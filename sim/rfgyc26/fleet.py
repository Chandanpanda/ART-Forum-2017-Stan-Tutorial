"""The fleet executive: two robots, one field, no collisions, no idling.

WHY THIS FILE EXISTS.  Until now the two robots avoided each other by ONE
mechanism, and it was open-loop: robot 2 planned around a set of corridors
robot 1 was *expected* to occupy, derived from robot 1's schedule.  Robot 1
did not know robot 2 existed at all.  That fails in both directions and did:

    T+0.8   A_rear x r2_flare_l, 2.1 N -- robot 2 sets off across robot 1's
            tail, because robot 1 is not in robot 2's map
    T+62    robot 2 stops at (690, 1130) and does not move again for 58 s
            of a 120 s match, while robot 1 drives to HOSPITAL 194 mm away

A reservation is a promise about the future; a promise is not a sensor.  So
the executive is three layers, and each catches what the one above it
cannot:

  1  RESERVATIONS over a topological decomposition of the field.  The free
     space is not open -- it is two big rooms joined by two narrow bridges
     (the pinches, 176 mm of gap for a chassis planned at 150), plus the
     quadrant the beam seal owns for 35 s and the four destination zones.
     Those are RESOURCES.  An agent acquires every resource a leg needs
     before it starts the leg, all-or-nothing, in a fixed global order --
     so there is no hold-and-wait, and with a strict priority between two
     agents there is no deadlock (Coffman's conditions, minus two of them).

  2  LIVE MUTUAL AVOIDANCE.  Reservations are only as good as the model of
     where a robot will be, and robots deviate.  So each agent also treats
     the other's MEASURED footprint, swept forward over a short horizon, as
     a hard obstacle in its own costmap.  This is the layer robot 1 was
     missing entirely, and it is the one that would have caught T+0.8.

  3  PREEMPTION.  Robot 2 is a detached actuator, not a peer: robot 1 does
     not negotiate with it, it commands it.  When robot 1 needs a resource
     robot 2 holds, robot 2 is told to leave, and it leaves toward its own
     next job rather than to a dead corner -- a yield that costs the fleet
     nothing but the transit it was going to make anyway.

Priority is robot 1 over robot 2, and that is measured rather than assumed:
robot 1's stations are worth 18 to 45 points for 10 to 35 seconds of work,
robot 2's best remaining task is a patient at +8 for 11 s.  Where the rates
ever cross, PRIORITY below is the single place to change it.

Pure numpy and params, like nav.py: this runs on the Pi.
"""
import numpy as np

from .params import Field, AgentA

PRIORITY = {"r1": 0, "r2": 1}          # lower wins

# ---------------------------------------------------------------- resources
# The decomposition is deliberately coarse.  Only regions where two chassis
# CANNOT pass are worth serialising; everywhere else the live-avoidance layer
# is cheaper and more honest than a booking system.  Rectangles are
# (x0, y0, x1, y1) and the order of this dict IS the acquisition order, which
# is what makes the protocol deadlock-free.
REGIONS = {
    # THE TWO DOORS, LEARNED RATHER THAN DRAWN (F152).  These were two
    # rectangles typed from a drawing, and they were wrong in a way that
    # matters: a door here is not a fact about the field, it is a fact about
    # the field PLUS the twelve patients standing in it -- on a cleared
    # board there is no pinch at all, and the patients move all match.
    #
    # learn_doors() below finds them instead, as the bottleneck of the
    # widest path between the two halves (world.widest_path).  It re-derives
    # these two to within 6 and 45 mm from the map alone, so they stay here
    # as the value to use before anything has been surveyed, and as the
    # thing the check compares against.
    "PINCH_E": (846.0, 430.0, 966.0, 640.0),
    "PINCH_W": (196.0, 430.0, 316.0, 640.0),
    # the deployment box, where both robots start within 225 mm of each other
    "BOX":     (643.0, 0.0, 1143.0, 300.0),
    # the quadrant the beam seal works in for 35 s
    "SEAL":    (0.0, 70.0, 420.0, 900.0),
    # destinations: shared between the kit column and the patient column
    "HOSP":    Field.HOSPITAL,
    "PCC_L":   Field.PCC_L,
    "PCC_R":   Field.PCC_R,
    "RECOVERY": Field.RECOVERY,
}
ORDER = list(REGIONS)

# Discs down the long axis that CONTAIN each chassis (F130).  n discs each
# cover a sub-rectangle of half-length a/n and half-width b, so the radius
# that still contains the corners is sqrt((a/n)^2 + b^2):
#     robot 1  285 x 235, a=142.5 b=117.5   ->  3 discs of 127 at -95, 0, 95
#     robot 2  156 x 110, a=78   b=55       ->  2 discs of  67 at -39, 39
def _chain(length, width, n):
    a, b = length/2.0, width/2.0
    r = float(np.hypot(a/n, b))
    return [(float(-a + (2*k+1)*a/n), r) for k in range(n)]


BODY = {"r1": _chain(AgentA.L, AgentA.W, 3),
        "r2": _chain(156.0, 110.0, 2)}

# ------------------------------------------------------- robot 1's own floor
# BOTH ROBOTS SCORE IN THE SAME THREE RECTANGLES.  Kits and patients are
# both delivered to HOSPITAL, PCC_L and PCC_R; only RECOVERY belongs to one
# robot alone.  So the partition that gave robot 2 nothing but RECOVERY caps
# the patient column at +2 against the referee, and the board with it at
# about 142 -- the ceiling is the partition, not the driving (F126).
#
# Sharing a zone failed once before, and the reason is worth keeping: robot
# 2 PUSHED patients then, so it could not choose where one came to rest, and
# it left them standing in the floor robot 1 needs.  With the gate it places
# them, so the question becomes a geometric one with an answer -- which floor
# does robot 1 actually need?
#
# Less than you would think.  Robot 1 parks BESIDE each zone and discharges
# over its flank; it never drives in.  At the hospital station its 285 x 235
# body covers x 594..829 of a rectangle that runs x 471..671 -- the eastern
# 77 mm of it -- and the reverse-and-pivot that follows adds a disc of
# radius 185 about a point 200 mm south.  Everything else inside the tape is
# floor no wheel touches, and that is where a patient goes.
# kit_hazard and the rectangles it was built from are gone (F148): robot 1
# solves for where it stands and publishes the floor that pose occupies, via
# reserve_floor above.  A model of where another robot "usually" is cannot
# survive that robot learning to choose.
def covered(prims, x, y):
    """Is (x, y) inside any of them?"""
    for pr in prims:
        if pr[0] == "rect":
            if pr[1] <= x <= pr[3] and pr[2] <= y <= pr[4]:
                return True
        elif (x-pr[1])**2 + (y-pr[2])**2 <= pr[3]**2:
            return True
    return False



def learn_doors(cmap, radius, south=None, north=None):
    """Replace the PINCH regions with the passages this board actually has.

    Keeps the keys, and therefore ORDER -- the fixed global acquisition
    order is what makes the reservation protocol deadlock-free, so it must
    not depend on what the map happened to look like.
    """
    from . import world
    south = south or (Field.W / 2.0, Field.H * 0.13)
    north = north or (Field.W / 2.0, Field.H * 0.89)
    found = world.doors(cmap, south, north, radius=radius, keep=2,
                        bite=radius * 4.0)
    if len(found) < 2:
        return dict(REGIONS)                 # no pinch: leave the prior
    found.sort(key=lambda g: g[0][0])        # west first, then east
    span = radius * 2.0                      # the body's own width, not a guess
    for key, ((gx, gy), _w) in zip(("PINCH_W", "PINCH_E"), found):
        REGIONS[key] = (gx - span, gy - span, gx + span, gy + span)
    return dict(REGIONS)


def regions_on(pts, pad=0.0):
    """Every region a polyline passes through."""
    out = []
    for name, (x0, y0, x1, y1) in REGIONS.items():
        hit = False
        for a, b in zip(pts[:-1], pts[1:]) or ():
            n = max(2, int(np.hypot(b[0]-a[0], b[1]-a[1]) / 25.0))
            for t in np.linspace(0.0, 1.0, n):
                x = a[0] + t*(b[0]-a[0])
                y = a[1] + t*(b[1]-a[1])
                if x0-pad <= x <= x1+pad and y0-pad <= y <= y1+pad:
                    hit = True
                    break
            if hit:
                break
        if not hit and len(pts) == 1:
            x, y = pts[0]
            hit = x0-pad <= x <= x1+pad and y0-pad <= y <= y1+pad
        if hit:
            out.append(name)
    return [n for n in ORDER if n in out]


def region_of(x, y, pad=0.0):
    return regions_on([(x, y)], pad=pad)


class Agent:
    """What the executive knows about one robot."""

    def __init__(self, name, radius, priority, body=None):
        self.name = name
        self.radius = float(radius)       # circumscribed, mm
        self.priority = int(priority)
        # A DISC IS THE WRONG SHAPE FOR A 285 x 235 ROBOT (F130).  Robot 1's
        # circumscribed radius is 185, so a single disc claims a 370 mm
        # circle for a body 235 mm across -- sixty per cent more floor than
        # it occupies, in every direction at once, and robot 2 plans and
        # flinches against that phantom.  A chain of discs down the long
        # axis covers the same rectangle as a stadium instead: same
        # guarantee (it still CONTAINS the body), 253 mm wide rather than
        # 370.  body is [(local_x, radius), ...]; None means the disc.
        self.body = list(body) if body else [(0.0, float(radius))]
        self.pose = None                  # (x, y, th_deg), measured
        self.vel = 0.0                    # mm/s along the heading
        self.path = []                    # committed, world mm
        self.held = set()
        self.evict = None                 # region it has been told to leave
        self.busy_until = 0.0


class Fleet:
    """The traffic manager.  One object, shared: robot 2's controller runs on
    robot 1's Pi, so there is no wire between them and no protocol to get
    wrong -- which is exactly why a CENTRALISED manager is the right answer
    here rather than a distributed negotiation."""

    def __init__(self):
        self.agents = {}
        self.t = 0.0
        self._stamp = 0            # bumped on every observation
        self._hz = {}
        self.board = {}            # id -> (id, x, y, kind), shared
        self.floors = {}           # name -> key -> [(x, y, r)]

    def join(self, name, radius, body=None):
        self.agents[name] = Agent(name, radius, PRIORITY.get(name, 9),
                                  body=body if body is not None
                                  else BODY.get(name))
        return self.agents[name]

    # -------------------------------------------------------- the board
    # WHAT ONE ROBOT SEES, BOTH ROBOTS KNOW.  The two controllers run on the
    # same Pi (design doc 15.5), so there is no wire between them and no
    # excuse for robot 1 planning against a field it believes to be empty.
    # It did: robot 1 has never held a costmap at all, and its kit approach
    # drove through four patients for a season because they were not on a
    # map it did not have.  Robot 2 surveys the pieces at the gun and
    # watches them move; this is where it says so.
    def see(self, pieces):
        """Publish observed loose pieces as [(id, x, y, kind), ...]."""
        self.board = {p[0]: tuple(p) for p in pieces}
        self._stamp += 1

    def pieces(self, kinds=None):
        """Everything seen, optionally filtered by kind."""
        out = list(self.board.values())
        if kinds is not None:
            out = [p for p in out if len(p) > 3 and p[3] in kinds]
        return out

    # --------------------------------------------------------- the floors
    # WHERE AN AGENT WILL BE STANDING, published by the agent that worked it
    # out.  This replaces a rectangle drawn by eye round a station constant:
    # once robot 1 solves for where to stand (F148) the old rectangle
    # describes a pose nobody uses, and the floor it really needs is just
    # its footprint at the chosen pose plus wherever its effector throws.
    def reserve_floor(self, name, key, discs):
        self.floors.setdefault(name, {})[key] = tuple(discs)
        self._stamp += 1

    def floor_of(self, name, key=None):
        """[(x, y, r), ...] another agent has said it will occupy."""
        d = self.floors.get(name, {})
        if key is not None:
            return list(d.get(key, ()))
        return [c for v in d.values() for c in v]

    # ------------------------------------------------------------ observing
    def observe(self, name, pose, vel=0.0, path=None, t=None):
        a = self.agents.get(name)
        if a is None:
            return
        a.pose = tuple(float(v) for v in pose)
        a.vel = float(vel)
        self._stamp += 1
        if path is not None:
            a.path = [(float(x), float(y)) for x, y in path]
        if t is not None:
            self.t = float(t)

    def track(self, rb=None, ctl=None, t=None):
        """Feed the executive from the cameras, once per control tick.

        THE EXECUTIVE IS ONLY AS CURRENT AS ITS LAST FIX (F124).  The first
        version learned a robot's pose as a side effect of that robot being
        commanded -- so at the gun, while robot 1 held station for two
        seconds, the executive did not know robot 1 existed, robot 2's map
        was empty, and robot 2 set off straight across robot 1's tail.  Both
        robots carry ArUco plates and the same overhead pair sees both of
        them; there is no reason for either pose to be stale.
        """
        if rb is not None:
            x, y, th = rb.pose
            self.observe("r1", (x, y, th),
                         vel=float(abs(getattr(rb, "_odo_v",
                                               [0.0, 0.0])[0])))
        if ctl is not None:
            self.observe("r2", ctl.pose,
                         vel=abs(sum(getattr(ctl, "_cmd", (0.0, 0.0)))) / 2.0)
        if t is not None:
            self.t = float(t)

    # -------------------------------------------------- live mutual avoidance
    def hazard(self, whose, horizon=1.6, step=0.4):
        """Cached per (agent, observation): this is called from both robots'
        drive paths at 50 Hz, and rebuilding the list every tick made a
        twelve-seed board take three times as long for an answer that cannot
        change between two observations."""
        key = (whose, round(horizon, 2), round(step, 2), self._stamp)
        hit = self._hz.get(key)
        if hit is not None:
            return hit
        out = self._hazard(whose, horizon, step)
        self._hz = {key: out}
        return out

    def _hazard(self, whose, horizon=1.6, step=0.4):
        """Discs covering where every OTHER agent is and is about to be.

        Returns [(x, y, r), ...] to stamp into a costmap.  The horizon is
        short on purpose: a metre of predicted path is a guess, but the next
        second and a half is nearly a fact, and it is enough for a robot
        travelling at 200-350 mm/s to stop or steer round.
        """
        out = []
        for a in self.agents.values():
            if a.name == whose or a.pose is None:
                continue
            x, y, th = a.pose
            c0, s0 = np.cos(np.radians(th)), np.sin(np.radians(th))
            for lx, r in a.body:
                out.append((x + lx*c0, y + lx*s0, r))
            if horizon <= 0.0:
                continue                       # measured footprint only
            # sweep forward along the committed path when there is one, or
            # along the current heading when there is not
            if a.path:
                d = 0.0
                px, py = x, y
                for qx, qy in a.path:
                    seg = float(np.hypot(qx-px, qy-py))
                    while d < horizon * max(a.vel, 60.0) and seg > 1e-6:
                        f = min(1.0, (step * max(a.vel, 60.0)) / seg)
                        px, py = px + f*(qx-px), py + f*(qy-py)
                        seg *= (1.0 - f)
                        d += step * max(a.vel, 60.0)
                        out.append((px, py, a.radius))   # swept: a guess

                    if d >= horizon * max(a.vel, 60.0):
                        break
                    px, py = qx, qy
            else:
                c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
                for k in range(1, int(horizon / step) + 1):
                    r = k * step * max(a.vel, 0.0)
                    out.append((x + c*r, y + s*r, a.radius))
        return out

    def gap(self, whose, pose=None):
        """Millimetres of SURFACE clearance to the nearest other agent.

        Negative means the two body models overlap.  Measured positions
        only -- this is what a speed limit should follow, and a prediction
        is not a measurement.
        """
        me = self.agents.get(whose)
        if me is None:
            return 1e9
        x, y = (pose or me.pose or (0.0, 0.0, 0.0))[:2]
        rad = me.radius
        best = 1e9
        for hx, hy, hr in self.hazard(whose, horizon=0.0):
            best = min(best, float(np.hypot(hx-x, hy-y)) - hr - rad)
        return best

    def stamp(self, cmap, whose, horizon=1.6, grow=0.0):
        """Put the other agents into this costmap as static obstacles."""
        for x, y, r in self.hazard(whose, horizon=horizon):
            cmap.add_disc(x, y, r + grow)
        return cmap

    def clear_ahead(self, whose, pts, margin=0.0, horizon=0.0):
        """horizon=0 means MEASURED POSITIONS ONLY, and that is the default
        on purpose (F124).  A bumper that also refuses the other robot's
        PREDICTED sweep is not a bumper, it is a planner with no ability to
        replan: measured, robot 1 stopped dead with robot 2 274 mm away
        because a disc 1.6 s along robot 2's path fell near its nose, and it
        spent nine seconds of one match frozen that way -- three and a half
        of them inside the beam window, which is what took the seal from
        55.8 to 16.3 points.  Prediction belongs in the costmap, where a
        planner can route round it.  Contact avoidance belongs here."""
        """Is this leg clear of the other agents right now?

        The last line of defence, checked by the tracker rather than the
        planner: a planner that ran a tenth of a second ago is describing a
        board that has moved.
        """
        me = self.agents.get(whose)
        rad = (me.radius if me else 95.0)
        for x, y, r in self.hazard(whose, horizon=horizon):
            for a, b in zip(pts[:-1], pts[1:]):
                seg = np.array([b[0]-a[0], b[1]-a[1]], float)
                L2 = float(seg @ seg)
                if L2 < 1e-9:
                    d = float(np.hypot(x-a[0], y-a[1]))
                else:
                    t = float(np.clip(((x-a[0])*seg[0] + (y-a[1])*seg[1]) / L2,
                                      0.0, 1.0))
                    d = float(np.hypot(x - (a[0]+t*seg[0]),
                                       y - (a[1]+t*seg[1])))
                if d < r + rad + margin:
                    return False
        return True

    # ------------------------------------------------------- reservations
    def owner(self, region):
        for a in self.agents.values():
            if region in a.held:
                return a.name
        return None

    def claim(self, whose, regions, evict=True):
        """Acquire regions all-or-nothing, in the global order.

        ALL-OR-NOTHING IS THE WHOLE POINT.  An agent that takes what it can
        and waits for the rest is holding-and-waiting, and two of those
        deadlock.  This either returns True with everything held, or returns
        False having taken nothing.
        """
        me = self.agents[whose]
        want = [n for n in ORDER if n in set(regions)]
        blocked = []
        for n in want:
            o = self.owner(n)
            if o is not None and o != whose:
                blocked.append((n, o))
        if blocked:
            if not evict:
                return False
            # PREEMPTION: a higher-priority agent does not queue.  It tells
            # the holder to leave and takes the region when it has.
            for n, o in blocked:
                other = self.agents[o]
                if me.priority < other.priority:
                    other.evict = n
                else:
                    return False
            return False               # not yet -- ask again once it moves
        me.held |= set(want)
        return True

    def release(self, whose, regions=None):
        me = self.agents[whose]
        me.held = set() if regions is None else (me.held - set(regions))

    def must_leave(self, whose):
        """The region this agent has to get out of, if any.

        EVICTION KEYS ON WHERE A ROBOT IS, NOT ON WHAT IT HOLDS.  The first
        version only evicted an agent that had booked the region, which is
        exactly the case that never causes trouble -- a booked region is one
        both parties already agree about.  The collision that actually
        happened was robot 2 standing in the deployment box it had never
        claimed while robot 1 swung its tail through it.  So: any agent
        physically inside a region a HIGHER-priority agent holds is told to
        leave, booking or no booking.
        """
        a = self.agents.get(whose)
        if a is None or a.pose is None:
            return None
        here = set(region_of(a.pose[0], a.pose[1], pad=a.radius))
        for n in ORDER:
            if n not in here:
                continue
            o = self.owner(n)
            if o is not None and o != whose and \
                    self.agents[o].priority < a.priority:
                a.evict = n
                return n
        a.evict = None
        return None

    def vacated(self, whose):
        a = self.agents.get(whose)
        if a is not None and a.evict is not None:
            a.held.discard(a.evict)
            a.evict = None


def escape_from(region, pose, radius=95.0, margin=40.0, away_from=None,
                toward=None):
    """Where a yielding robot goes: out of `region`, by the shortest exit
    that does not walk into the robot it is yielding TO.

    Straight out of a face, because the point is to stop being in the way in
    the fewest millimetres -- but "nearest face" alone once sent robot 2 out
    of the deployment box heading due west, into the robot that had just
    preempted it.  Ties break toward wherever the caller's next job is, so a
    yield costs the fleet only the transit it was going to make anyway.
    """
    x0, y0, x1, y1 = REGIONS[region]
    x, y, _th = pose
    outs = [(x0 - radius - margin, y), (x1 + radius + margin, y),
            (x, y0 - radius - margin), (x, y1 + radius + margin)]
    # 60 mm, not 90: the bound is the chassis HALF-WIDTH plus slop, and a
    # 90 mm margin rejected the only sane exit from the deployment box --
    # robot 2's own start pose sits at x 1055, which a 90 mm bound calls
    # off-field.
    ok = [(px, py) for px, py in outs
          if 60.0 < px < Field.W - 60.0 and 60.0 < py < Field.H - 60.0]
    if not ok:
        return None

    def cost(p):
        c = float(np.hypot(p[0]-x, p[1]-y))
        if away_from is not None:
            d = float(np.hypot(p[0]-away_from[0], p[1]-away_from[1]))
            if d < 380.0:
                c += (380.0 - d) * 3.0        # do not exit into them
        if toward is not None:
            c += 0.35 * float(np.hypot(p[0]-toward[0], p[1]-toward[1]))
        return c
    return min(ok, key=cost)
