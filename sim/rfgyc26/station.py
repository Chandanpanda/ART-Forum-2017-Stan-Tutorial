"""WHERE TO STAND: one solver for a question this project answers four times.

Every mission in this codebase repeatedly asks the same thing in different
words: *where do I put the chassis so that when I actuate this thing, the
payload ends up in that region, and my body is somewhere legal?*  Today
there are four separate hand-computed answers to it --

    AgentA.KIT_STATION   three poses, chosen with a ruler and a comment
    robot2's shake pose  the zone centre, minus a tail offset
    robot2._zone_pt      a grid search with its own scoring
    nav.capture_approach a ladder of stand-offs, its own scoring again

-- and every one of them is a constant or a bespoke search that has to be
re-derived by hand when the field, the chassis or the effector changes.
The PCC_L episode is what this module exists to prevent: the station had
been at x 240 since it was written, the 235 mm-wide body spanned x 122..357,
and the west sticker column stands at x 160, so every approach drove
through four patients and two runs in six beached on one.  The fix was to
work out, on paper, that the body must clear x 170 and the hopper's lip
must stay west of x 200, giving a feasible window of [292.5, 310] -- and
then to write 300 into the parameters.

That derivation is a function, not a number.  Given a region, an effector,
a footprint and a map, it is a small search, and once it is a search it
answers for any zone, any chassis and any field.

THE ANSWER IS A MARGIN, NOT A POSE.  What makes a station good is not
where it is but how much can go wrong before it fails, and there are two
independent ways for it to fail: the payload lands outside the region, or
the body hits something.  Both are distances in millimetres, so both are
comparable, and the margin of a pose is the smaller of them.  Ranking by
margin is what turns "it worked on the seeds I tried" into "it tolerates
this many millimetres of arrival error", which is a claim a rig can check.

Pure numpy and dataclasses -- no mujoco, no field constants.  This runs on
the Pi and it runs on somebody else's field.
"""
from dataclasses import dataclass, field as _field

import numpy as np


# --------------------------------------------------------------- geometry
@dataclass(frozen=True)
class Footprint:
    """A chassis in body frame: +x forward, +y left, origin at the axle.

    `pts` are sample points on and inside the hull -- the corners at least,
    plus anything that sticks out.  Clearance is evaluated at these points,
    so a shape with a long nose wants a point on the nose.
    """
    pts: tuple
    inscribed: float
    circumscribed: float

    @classmethod
    def rect(cls, length, width, axle=0.5, extra=()):
        """A box `length` by `width` with the axle `axle` of the way back
        from the nose (0.5 = centred, which is what a differential drive
        with the wheels amidships has)."""
        fwd, aft = length * axle, -length * (1.0 - axle)
        hw = width / 2.0
        pts = [(fwd, hw), (fwd, -hw), (aft, hw), (aft, -hw),
               (fwd, 0.0), (aft, 0.0), (0.0, hw), (0.0, -hw), (0.0, 0.0)]
        pts += list(extra)
        return cls(tuple(pts), width / 2.0,
                   float(max(np.hypot(x, y) for x, y in pts)))

    def world(self, x, y, th_deg):
        """The sample points in world frame for a pose."""
        a = np.radians(th_deg)
        c, s = np.cos(a), np.sin(a)
        p = np.asarray(self.pts, float)
        return np.stack([x + p[:, 0]*c - p[:, 1]*s,
                         y + p[:, 0]*s + p[:, 1]*c], axis=1)


@dataclass(frozen=True)
class Effector:
    """Where a payload ends up, in body frame, when this is actuated.

    A hopper that discharges over a flank, a pocket that leaves its cargo
    where the mouth was, a plow that shoves a piece ahead of itself -- all
    three are an offset, an uncertainty, and (for the ones that release
    more than one thing) a step between successive items.

        hopper  offset (0, 140)   spread 12   stride (28, 0)  count 6
        pocket  offset (26, 0)    spread 20
        tray    offset (-71, 0)   spread 25   stride (0, 28)  count 2
    """
    offset: tuple
    spread: float = 0.0
    stride: tuple = (0.0, 0.0)
    count: int = 1

    def deposits(self, x, y, th_deg):
        """Where the items land, in world frame, for a pose."""
        a = np.radians(th_deg)
        c, s = np.cos(a), np.sin(a)
        out = []
        for k in range(self.count):
            lx = self.offset[0] + k * self.stride[0]
            ly = self.offset[1] + k * self.stride[1]
            out.append((x + lx*c - ly*s, y + lx*s + ly*c))
        return out

    @property
    def reach(self):
        """How far from the chassis the furthest item can be."""
        k = max(self.count - 1, 0)
        return float(np.hypot(self.offset[0] + k*self.stride[0],
                              self.offset[1] + k*self.stride[1])
                     + self.spread)


@dataclass
class Stand:
    """A candidate pose and what it tolerates, in millimetres."""
    pose: tuple                 # (x, y, heading_deg)
    margin: float               # the smaller of the two below
    inside: float               # least distance of an item to the region edge
    clear: float                # least clearance of the body to an obstacle
    deposits: tuple = ()

    def __repr__(self):
        return ("<stand (%.0f, %.0f, %.0f) margin %.0f mm "
                "[inside %.0f, clear %.0f]>"
                % (self.pose[0], self.pose[1], self.pose[2],
                   self.margin, self.inside, self.clear))


# ----------------------------------------------------------------- solver
def _edge_distance(pt, box):
    """Distance from a point to a rectangle's boundary, positive inside."""
    x, y = pt
    return min(x - box[0], box[2] - x, y - box[1], box[3] - y)


def stand_for(region, eff, foot, cmap, headings=16, res=None, prefer=None,
              keep_out=(), avoid=(), avoid_r=0.0, top=8, need=0.0,
              need_clear=None, bounds=None):
    """Poses from which `eff` puts its payload inside `region`, best first.

    region    (x0, y0, x1, y1) the payload must land in
    eff       an Effector -- what the robot is about to actuate
    foot      a Footprint -- the chassis, for the clearance half
    cmap      a nav.CostMap, for the obstacle field
    headings  how many orientations to try around the circle
    prefer    (x, y) to break ties toward, e.g. where the robot is now
    keep_out  rectangles the BODY may not enter (tape it must not cross,
              floor another robot has reserved)
    avoid     points the body must stay `avoid_r` from (pieces already down)
    need      reject anything with less margin than this
    bounds    (x0, y0, x1, y1) to search in; default is the region grown by
              the effector's reach

    Returns up to `top` Stands sorted by margin, then by nearness to
    `prefer`.  An empty list means the effector cannot reach the region
    from anywhere legal -- which is a real answer, and one worth having
    before the robot drives somewhere to find out.
    """
    res = float(res or cmap.res)
    dist = cmap.clearance()
    reach = eff.reach + 1.0
    if bounds is None:
        bounds = (region[0] - reach, region[1] - reach,
                  region[2] + reach, region[3] + reach)
    x0 = max(bounds[0], foot.circumscribed * 0.25)
    y0 = max(bounds[1], foot.circumscribed * 0.25)
    x1 = min(bounds[2], cmap.w - foot.circumscribed * 0.25)
    y1 = min(bounds[3], cmap.h - foot.circumscribed * 0.25)
    if x1 <= x0 or y1 <= y0:
        return []
    xs = np.arange(x0, x1 + 1e-9, res)
    ys = np.arange(y0, y1 + 1e-9, res)
    avoid = np.asarray(avoid, float).reshape(-1, 2) if len(avoid) else None

    out = []
    for th in np.linspace(0.0, 360.0, headings, endpoint=False):
        for x in xs:
            for y in ys:
                # the payload first: it is the cheaper test and it rejects
                # most of the grid
                dep = eff.deposits(x, y, th)
                inside = min(_edge_distance(p, region) for p in dep)
                inside -= eff.spread
                if inside <= need:
                    continue
                pts = foot.world(x, y, th)
                ii = np.clip((pts[:, 0] // cmap.res).astype(int), 0, cmap.nx-1)
                jj = np.clip((pts[:, 1] // cmap.res).astype(int), 0, cmap.ny-1)
                clear = float(dist[ii, jj].min())
                if clear <= need:
                    continue
                if any(b[0] <= px <= b[2] and b[1] <= py <= b[3]
                       for b in keep_out for px, py in pts):
                    continue
                if avoid is not None and avoid_r > 0.0:
                    d = np.hypot(pts[:, None, 0] - avoid[None, :, 0],
                                 pts[:, None, 1] - avoid[None, :, 1])
                    if d.size and d.min() < avoid_r:
                        continue
                m = min(inside, clear)
                if m <= need:
                    continue
                near = 0.0 if prefer is None else \
                    float(np.hypot(x - prefer[0], y - prefer[1]))
                out.append((m, -near, Stand((float(x), float(y), float(th)),
                                            m, float(inside), clear,
                                            tuple(dep))))
    out.sort(key=lambda t: (-t[0], -t[1]))
    return [t[2] for t in out[:top]]


def best_stand(*a, **kw):
    """stand_for()'s first answer, or None."""
    got = stand_for(*a, **kw)
    return got[0] if got else None


def reachable(stands, planner, keep=1):
    """Filter stands by whether the robot can actually GET to them.

    A CLEAR STATION IS NOT A CLEAR APPROACH, and assuming otherwise is the
    mistake this module was written to stop making.  Worked out by hand,
    robot 1's PCC_L station looked like it drove through the west sticker
    column; the solver disagreed, and the solver was right -- at the
    station the chassis spans y 788..1073 and the stickers top out at 773,
    so the pose was always clear.  What was not clear was the TRAVERSE to
    it, 200 mm further south, where the same body spans y 587..872.  The
    station was moved east and the column stopped being hit, but only
    because the traverse follows the station's x.

    So a caller that needs to arrive somewhere asks for both: the pose that
    can do the job, and a path to it.  `planner(pose) -> (path, seconds)`
    is whatever motion planner the caller uses -- nav.plan bound to its map
    and radii, usually.

    Returns [(stand, path, seconds), ...] for the first `keep` that work.
    """
    out = []
    for st in stands:
        path, secs = planner(st.pose)
        if path is not None and np.isfinite(secs):
            out.append((st, path, float(secs)))
            if len(out) >= keep:
                break
    return out
