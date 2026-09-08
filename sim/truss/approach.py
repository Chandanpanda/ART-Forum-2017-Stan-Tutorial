"""Where the head goes to wind a joint, and what it has to clear to get
there -- station.stand_for for a gantry.

The question is the same one the competition robot asks before every
drop: put the effector so the payload lands where it must, with the body
somewhere legal, and say HOW MUCH can go wrong before it fails.  Here the
effector is a C-ring, the payload is a hoop of thread, and "legal" means
the ring, its spool and the head above them touch nothing through the
whole cycle: descending onto the chord, turning eighteen times round it,
and traversing to the next joint.

THE ANSWER IS A MARGIN, IN MILLIMETRES.  A pose is not good because it
looked fine; it is good because the nearest rod, pin or cradle is this
far from the head's swept volume, and a rig can check that number.

HOW IT IS MEASURED.  The head is a handful of solids of revolution and
boxes in the ring's own frame -- an annulus, a spool, a carriage box --
and the distance from a point to each of those is a closed form.  The
rods and the fixture are capsules, sampled densely along their axes.  So
the clearance is exact to the rod sampling pitch, a millimetre, which is
the difference between "the solver says 1.2 mm" and "the solver says 1.2
mm and a 2 mm rod fell between its sample points".  (The first version
sampled the HEAD as a point cloud and passed a truss whose other chords
ran straight through the spool.)

Three sweeps:

    seated    the full annulus (the gap vanishes over a revolution), the
              spool's torus, the head box, at the chord axis
    descent   the C-ring parked with its gap toward the joint's face,
              from the lift height down to seated
    traverse  the same ring parked gap-down, at the lift height, from
              this joint to the next

and the lift height is the SMALLEST that clears the traverse -- lifting
150 mm between every joint is what the brief's Z stroke allows, not what
the cycle can afford.  numpy only; takes a geometry and a fixture, so it
answers for any truss.
"""
from dataclasses import dataclass
from math import radians, sin, cos, pi

import numpy as np

from .spec import Ring, Head, Process, ring_r_in, ring_r_out
from .geometry import theta_chord_up


@dataclass
class Approach:
    """One joint's winding station."""
    joint:     int
    theta:     float            # cage angle
    centre:    np.ndarray       # ring centre when seated (world, mm)
    lift:      float            # mm above the chord to traverse at
    margin:    float            # the least of the three clearances
    seated:    float
    descent:   float
    traverse:  float
    gap_down:  float = 0.0      # ring azimuth to park the gap at for the
                                # descent and the lift; 0 is straight down

    def __repr__(self):
        c = self.centre
        return ("<approach j%d theta %.0f at (%.0f, %.1f, %.1f) lift %.0f "
                "margin %.1f [seat %.1f desc %.1f trav %.1f] gap %.0f>"
                % (self.joint, self.theta, c[0], c[1], c[2], self.lift,
                   self.margin, self.seated, self.descent, self.traverse,
                   self.gap_down))


# ---------------------------------------------------------- head solids
# Ring frame: x along the truss, z up, origin at the ring centre.  Azimuth
# psi is measured from straight down toward +y, so a point (x, y, z) has
# psi = atan2(y, -z).

def _annulus_dist(P, r0, r1, xh, gap=None):
    """Distance from points P [N,3] to an annular cylinder about x: radii
    r0..r1, half-width xh.  With gap=(centre_deg, half_deg) the sector is
    missing and a point inside it measures to the nearer edge face."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    rho = np.hypot(y, z)
    dx = np.maximum(np.abs(x) - xh, 0.0)
    dr = np.maximum(np.maximum(r0 - rho, rho - r1), 0.0)
    d = np.hypot(dx, dr)
    if gap is None:
        return d
    gc, gh = radians(gap[0]), radians(gap[1])
    psi = np.arctan2(y, -z)
    rel = (psi - gc + pi) % (2 * pi) - pi           # azimuth from the gap centre
    inside = np.abs(rel) < gh
    if not np.any(inside):
        return d
    # in the gap: distance to the nearer edge face, a radial half-plane
    # {rho in [r0, r1], |x| <= xh} at psi = gc +- gh
    edge = gc + np.where(rel > 0, gh, -gh)
    # decompose the point into (along-edge radial, perpendicular)
    ey, ez = np.sin(edge), -np.cos(edge)
    along = y * ey + z * ez                          # radial coordinate on the edge
    perp = np.abs(y * (-ez) + z * ey)                # off the edge plane
    da = np.maximum(np.maximum(r0 - along, along - r1), 0.0)
    d_edge = np.sqrt(dx ** 2 + da ** 2 + perp ** 2)
    return np.where(inside, d_edge, d)


def _box_dist(P, lo, hi):
    """Distance from points to an axis-aligned box [lo, hi]."""
    q = np.maximum(np.maximum(lo - P, P - hi), 0.0)
    return np.linalg.norm(q, axis=1)


def _rot_x_pts(P, deg):
    c, s = cos(radians(deg)), sin(radians(deg))
    y, z = P[:, 1], P[:, 2]
    return np.stack([P[:, 0], c * y - s * z, s * y + c * z], axis=1)


def head_distance(P, rotating, gap_az=0.0):
    """Distance from points (ring frame) to the head's solids.

    rotating=True is the seated case -- everything on the ring sweeps a
    full circle, so the gap is gone and the spool is a torus.  Otherwise
    the ring is parked with its gap at gap_az (0 = straight down) and the
    spool at Ring.SPOOL_AZ from it, as a box on the rim.
    """
    r_in, r_out = ring_r_in(), ring_r_out()
    sr, st, sx = Ring.SPOOL
    hx, hy, z0, z1 = Ring.HEAD_BOX
    d = _box_dist(P, np.array([-hx, -hy, z0]), np.array([hx, hy, z1]))
    if rotating:
        d = np.minimum(d, _annulus_dist(P, r_in, r_out, Ring.W / 2.0))
        d = np.minimum(d, _annulus_dist(P, r_out, r_out + sr, sx / 2.0))
        return d
    d = np.minimum(d, _annulus_dist(P, r_in, r_out, Ring.W / 2.0,
                                    gap=(gap_az, Ring.GAP / 2.0)))
    # the spool: a box standing on the rim at azimuth gap_az + SPOOL_AZ.
    # Rotate the points so that azimuth is straight up, then box-test.
    Q = _rot_x_pts(P, -(gap_az + Ring.SPOOL_AZ - 180.0))
    d = np.minimum(d, _box_dist(Q, np.array([-sx / 2.0, -st / 2.0, r_out]),
                                np.array([sx / 2.0, st / 2.0, r_out + sr])))
    return d


# ------------------------------------------------------------ the world
class Obstacles:
    """Every capsule the head may not touch, at one cage angle, sampled
    along its axis every `pitch` mm."""

    def __init__(self, geom, fixture, theta, pitch=1.0, pts=None, rad=None, skip=()):
        self.geom, self.fixture, self.theta = geom, fixture, theta
        if pts is not None:
            self.pts, self.rad = pts, rad
            return
        p0, p1, rr = geom.rods_at(theta)
        if len(skip):
            keep = np.array([r.index not in skip for r in geom.rods])
            p0, p1, rr = p0[keep], p1[keep], rr[keep]
        if fixture is not None:
            f0, f1, fr = fixture.obstacles(theta)
            if len(f0):
                p0, p1, rr = (np.vstack([p0, f0]), np.vstack([p1, f1]),
                              np.concatenate([rr, fr]))
        pts, rad = [], []
        for a, b, r in zip(p0, p1, rr):
            n = max(2, int(np.ceil(np.linalg.norm(b - a) / pitch)) + 1)
            s = np.linspace(0.0, 1.0, n)[:, None]
            pts.append(a[None, :] + s * (b - a)[None, :])
            rad.append(np.full(n, float(r)))
        self.pts = np.concatenate(pts, axis=0)
        self.rad = np.concatenate(rad)

    def near(self, x_lo, x_hi):
        keep = (self.pts[:, 0] + self.rad >= x_lo) & (self.pts[:, 0] - self.rad <= x_hi)
        return Obstacles(self.geom, self.fixture, self.theta,
                         pts=self.pts[keep], rad=self.rad[keep])

    def clearance(self, centre, rotating, gap_az=0.0, shifts=((0.0, 0.0, 0.0),)):
        """Least surface clearance to the head at `centre` (world), for
        every offset in `shifts` applied to the head."""
        return self.nearest(centre, rotating, gap_az, shifts)[0]

    def nearest(self, centre, rotating, gap_az=0.0, shifts=((0.0, 0.0, 0.0),)):
        """(clearance, the obstacle point that sets it, the head offset it
        was at) -- the number and the reason."""
        if not len(self.rad):
            return float("inf"), None, None
        best, where, at = float("inf"), None, None
        for s in shifts:
            P = self.pts - (np.asarray(centre, float) + np.asarray(s, float))[None, :]
            d = head_distance(P, rotating, gap_az) - self.rad
            i = int(np.argmin(d))
            if d[i] < best:
                best, where, at = float(d[i]), self.pts[i].copy(), np.asarray(s, float)
        return best, where, at


# ----------------------------------------------------------- the solver
def _reach(centre, dx=0.0):
    """x-extent the head can occupy around a station and its traverse."""
    r = Ring.HEAD_BOX[0] + Ring.SPOOL[0] + 5.0
    return (float(centre[0]) + min(0.0, dx) - r, float(centre[0]) + max(0.0, dx) + r)


def gap_azimuth(n_face):
    """Where to park the gap for a joint whose diagonals leave along
    n_face (a y-z unit vector in the world at this cage angle).

    THE GAP HAS TO ADMIT THE WHOLE CLUSTER, NOT JUST THE CHORD.  The ring
    descends along -z, so the chord enters at azimuth 0; the two mitred
    ends lie against the chord along n_face, which for a chord-up joint
    is 30 degrees off straight down -- exactly the edge of a 60 degree
    gap.  Parked straight down, the gap's edge shaves past the diagonals
    at a millimetre; parked toward the face, both pass through the
    opening.  "Stop with the gap down" (brief 4.5) is the right idea and
    the wrong number; station() measures the number.
    """
    psi = np.degrees(np.arctan2(n_face[1], -n_face[2]))
    lim = Ring.GAP / 2.0 - 8.0
    return float(np.clip(psi / 2.0, -lim, lim))


def seated_clearance(obs, centre, band=0.0):
    """The ring turns at every x across the band, so the seated test is
    taken at both ends of it as well as the centre."""
    hb = band / 2.0
    shifts = [(0.0, 0.0, 0.0)] if hb <= 0 else [(-hb, 0.0, 0.0), (0.0, 0.0, 0.0), (hb, 0.0, 0.0)]
    return obs.near(float(centre[0]) - hb - 40.0, float(centre[0]) + hb + 40.0).clearance(
        centre, True, shifts=shifts)


def descent_clearance(obs, centre, lift, gap_az=0.0, step=1.0, band=0.0):
    """Least clearance of the parked ring anywhere on its way down, at
    either end of the band (the descent happens at the band's start,
    whichever way the run goes)."""
    hs = np.arange(0.0, lift + 1e-9, step)
    xs = [0.0] if band <= 0 else [-band / 2.0, band / 2.0]
    return obs.near(float(centre[0]) - band - 40.0, float(centre[0]) + band + 40.0).clearance(
        centre, False, gap_az, shifts=[(x, 0.0, h) for x in xs for h in hs])


def traverse_clearance(obs, centre, lift, dx, gap_az=0.0, step=3.0):
    """Least clearance of the parked ring sliding `dx` along x at `lift`."""
    n = max(2, int(abs(dx) / step) + 1)
    return obs.near(*_reach(centre, dx)).clearance(
        centre, False, gap_az, shifts=[(s, 0.0, lift) for s in np.linspace(0.0, dx, n)])


def min_lift(obs, centre, dx, need, gap_az=0.0, h_max=80.0, step=0.5):
    """The smallest lift at which the traverse to the next joint keeps
    `need` mm of clearance, or None if none under h_max does."""
    o = obs.near(*_reach(centre, dx))
    last_bad = 0.0
    for h in np.arange(0.0, h_max + 1e-9, 4.0):
        if traverse_clearance(o, centre, h, dx, gap_az) >= need:
            for hf in np.arange(last_bad, h + 1e-9, step):
                if traverse_clearance(o, centre, hf, dx, gap_az) >= need:
                    return float(hf)
            return float(h)
        last_bad = h
    return None


def station(geom, fixture, joint, theta, obs=None, need=None, next_dx=None):
    """The Approach for one joint at one cage angle, or None if the ring
    cannot even be seated there."""
    need = Process.LIFT_CLEAR if need is None else need
    obs = Obstacles(geom, fixture, theta) if obs is None else obs
    c, n = geom.joint_at(joint, theta)
    band = geom.t.band
    seat = seated_clearance(obs, c, band=band)
    if seat <= 0.0:
        return None
    if next_dx is None:
        js = geom.joints_on(joint.chord)
        i = [j.index for j in js].index(joint.index)
        next_dx = (js[i + 1].x - joint.x) if i + 1 < len(js) else \
                  (js[i - 1].x - joint.x if i > 0 else geom.t.run)
    lift = min_lift(obs, c, next_dx, need, gap_az=0.0)
    if lift is None:
        return None
    lift = lift + Process.LIFT_CLEAR
    # The halfway angle is the right neighbourhood; the best angle in it
    # is measured.  A 60 degree gap holding a chord and a diagonal 30
    # degrees apart has a few millimetres to share, and where the optimum
    # lies depends on the rods' radii and on what else is near.
    base = gap_azimuth(n)
    gap, desc = base, -1e9
    lim = Ring.GAP / 2.0 - 8.0
    for d in (0.0, -5.0, 5.0, -10.0, 10.0):
        cand = float(np.clip(base + d, -lim, lim))
        v = descent_clearance(obs, c, lift, gap_az=cand, band=band)
        if v > desc + 1e-9:
            gap, desc = cand, v
    trav = traverse_clearance(obs, c, lift, next_dx, gap_az=0.0)
    return Approach(joint.index, float(theta), c, float(lift),
                    min(seat, desc, trav), seat, desc, trav, gap_down=gap)


def _probe_joints(js):
    """A chord's joints alternate faces, so its stations come in two
    interior kinds plus the two ends; four joints stand for them all
    while an angle is being chosen."""
    if len(js) <= 4:
        return list(js)
    return [js[0], js[1], js[2], js[-1]]


def plan_chord(geom, fixture, k, theta=None, span=30.0, step=10.0):
    """Every joint on chord k, at the cage angle that gives the chord's
    tightest joint the most room.  theta=None searches chord-up +- span
    on four representative joints, then solves every joint at the
    winner.  Ties within a millimetre go to the nominal chord-up angle."""
    js = geom.joints_on(k)
    base = theta_chord_up(k)
    if theta is not None:
        cands = [float(theta)]
    else:
        cands = [base + d for d in np.arange(-span, span + 1e-9, step)]
    best_val, best_theta = None, None
    for th in cands:
        obs = Obstacles(geom, fixture, th)
        vals = []
        for j in _probe_joints(js):
            a = station(geom, fixture, j, th, obs=obs)
            if a is None:
                vals = None
                break
            vals.append(a.margin)
        if vals is None:
            continue
        nominal = abs(((th - base) + 180.0) % 360.0 - 180.0)
        key = (round(min(vals), 0), -nominal)
        if best_val is None or key > best_val:
            best_val, best_theta = key, th
    if best_theta is None:
        return None, None
    obs = Obstacles(geom, fixture, best_theta)
    out = [station(geom, fixture, j, best_theta, obs=obs) for j in js]
    if any(a is None for a in out):
        return None, None
    return best_theta, out


def plan_truss(geom, fixture, **kw):
    """{chord: (theta, [Approach, ...])} for the whole truss."""
    return {k: plan_chord(geom, fixture, k, **kw) for k in range(geom.t.n_chords)}


def index_lift(geom, fixture):
    """How high the ring centre must be above the truss axis for the cage
    to turn under it: the cage's swept radius plus the head's lowest
    point, plus clearance."""
    r_out = ring_r_out()
    lowest = max(r_out, r_out + Ring.SPOOL[0])
    return fixture.cage_swept_r() + lowest + Process.LIFT_CLEAR


# ------------------------------------------------------- the held rod
def held_rod(mid, yaw, half):
    """Ends of a rod held level through its midpoint, `yaw` degrees from +x."""
    u = np.array([cos(radians(yaw)), sin(radians(yaw)), 0.0])
    return mid - half * u, mid + half * u


def yaw_sweep(yaw0, yaw1, step=3.0):
    """The yaws the wrist passes through from yaw0 to yaw1 the short way
    round -- the way schedule.yaw_to and the process turn it."""
    d = ((yaw1 - yaw0) + 180.0) % 360.0 - 180.0
    n = max(2, int(abs(d) / step) + 1)
    return yaw0 + d * np.linspace(0.0, 1.0, n)


def segment_clearance(obs, p0, p1, r):
    """Least surface clearance between the capsule (p0, p1, r) and the
    obstacles' sampled capsules."""
    if not len(obs.rad):
        return float("inf")
    u = p1 - p0
    L = float(np.linalg.norm(u))
    u = u / L
    v = obs.pts - np.asarray(p0, float)[None, :]
    s = np.clip(v @ u, 0.0, L)
    d = np.linalg.norm(v - s[:, None] * u[None, :], axis=1) - obs.rad - r
    return float(d.min())


def held_rod_head_clearance(half, r, yaw, extension):
    """Clearance between a held rod and the head's own solids (ring, spool,
    head box), in the ring frame: the grip point is Head.grip_x() along x
    and TIP_PARK above the ring centre less the gripper's extension."""
    mid = np.array([Head.grip_x(), 0.0, Head.TIP_PARK - extension])
    p0, p1 = held_rod(mid, yaw, half)
    n = max(2, int(2.0 * half) + 1)
    P = p0[None, :] + np.linspace(0.0, 1.0, n)[:, None] * (p1 - p0)[None, :]
    return float((head_distance(P, False, 0.0) - r).min())


def yaw_height(obs, place, yaw0, yaw1, half, r, need, z_max, stroke, step=0.5):
    """The lowest height (the rod's midpoint, world) at which a held rod
    can be lowered to at yaw0 -- the gripper's extension, the last
    `stroke` mm straight down -- and then turned to yaw1 the short way,
    keeping `need` mm from every obstacle.  None if nothing under z_max
    does.

    THE YAW WAITS FOR THE EXTENSION.  Carried retracted, a rod sits
    TIP_PARK above the ring's centre and beside its plate, so a rod
    parallel to the plate is clear and every other yaw swings its ends
    through the rim (measured: turned retracted, a diagonal stopped at 37
    of its 135 degrees against the ring, slipped in the pads, and missed
    its cradles).  So a rod is turned with the gripper out, below the
    rim, and this says how high above its seat that can happen.
    """
    place = np.asarray(place, float)
    o = obs.near(float(place[0]) - half - r - 5.0, float(place[0]) + half + r + 5.0)
    yaws = yaw_sweep(yaw0, yaw1)

    def clear(z):
        mid = np.array([place[0], place[1], z])
        for y in yaws:
            if segment_clearance(o, *held_rod(mid, y, half), r) < need:
                return False
        for h in np.arange(0.0, stroke + 1e-9, 2.0):
            if segment_clearance(o, *held_rod(mid + np.array([0.0, 0.0, h]), yaw0, half),
                                 r) < need:
                return False
        return True

    z0 = float(place[2])
    last_bad = z0
    for z in np.arange(z0, z_max + 1e-9, 4.0):
        if clear(z):
            for zf in np.arange(last_bad, z + 1e-9, step):
                if clear(zf):
                    return float(zf)
            return float(z)
        last_bad = z
    return None


# -------------------------------------------------------- the post loop
def post_loop(obs, post_x, chord_z, need=None, h_max=80.0, step=0.5):
    """Where the head runs round an anchor post to hook the strand: the
    ring-centre height and the y leg of a rectangle about the post, with
    the parked ring (gap down) clearing everything by `need` along it.
    Returns (z, leg, half_x, clearance) or None.

    The exit guide must pass the post on both sides by the post's radius
    plus the clearance -- that is the leg -- and the plate must pass fully
    beyond the post in x -- that is the half-width.  The height is the
    lowest at which that rectangle clears.  Measured on the 300 mm truss
    at the winding lift (14 mm over the chord): the post's lower end lies
    in the ring's plate for any leg over 2 mm, and the rectangle the plan
    had drawn by hand (legs of four post radii) hooked the ring on the
    post, jammed the x axis, and put the first look 60 mm from its joint.
    """
    from .spec import Cage
    need = Process.SEAT_CLEAR if need is None else need
    leg = Cage.POST_R + need
    a = Head.ring_axial_half() + Cage.POST_R + need
    o = obs.near(post_x - a - 30.0, post_x + a + 30.0)
    xs = np.linspace(-a, a, 9)
    ys = np.linspace(-leg, leg, 5)
    shifts = [(sx, sy, 0.0) for sx in xs for sy in (-leg, leg)] + \
             [(sx, sy, 0.0) for sx in (-a, a) for sy in ys]

    def clear(h):
        return o.clearance(np.array([post_x, 0.0, chord_z + h]), False, 0.0, shifts=shifts)

    last_bad = 0.0
    for h in np.arange(0.0, h_max + 1e-9, 4.0):
        if clear(h) >= need:
            for hf in np.arange(last_bad, h + 1e-9, step):
                c = clear(hf)
                if c >= need:
                    return float(chord_z + hf), float(leg), float(a), float(c)
            return float(chord_z + h), float(leg), float(a), float(clear(h))
        last_bad = h
    return None


def head_points(rotating, gap_az_deg=0.0, n_az=36):
    """The head's solids as a point cloud in the ring frame -- for
    rendering and for rigs that want to look at the envelope; the solver
    itself uses head_distance."""
    r_in, r_out = ring_r_in(), ring_r_out()
    sr, st, sx = Ring.SPOOL
    pts = []
    rs = np.linspace(r_in, r_out, 4)
    xs = np.linspace(-Ring.W / 2.0, Ring.W / 2.0, 3)
    if rotating:
        psi = np.linspace(0.0, 2 * pi, n_az, endpoint=False)
    else:
        g = radians(Ring.GAP / 2.0)
        psi = radians(gap_az_deg) + np.linspace(g, 2 * pi - g, n_az)
    R, PSI, X = np.meshgrid(rs, psi, xs, indexing="ij")
    pts.append(np.stack([X.ravel(), (R * np.sin(PSI)).ravel(),
                         (-R * np.cos(PSI)).ravel()], axis=1))
    return np.concatenate(pts, axis=0)
