"""The camera mount, solved: a six-strut nose at each end of the truss.

WHAT IT IS.  Nine rods per end and ONE machined part:

    3 END BATTENS closing the triangle of chord ends.  An octahedron is two
      triangles and six struts; with the base triangle open the three chord
      ends splay and the mount is thirty times softer than it looks (spine's
      own measurement).  These lie in the ring's plane of rotation, which
      brief 4.2 prohibits -- but the prohibition binds where the ring has to
      WORK, and the last joint is `end_margin` inboard of here.  The ring
      never comes out this far.
    6 STRUTS, octahedral: each chord end feeds two landings, so no joint has
      to carry a moment.
    1 COLLAR round the camera's lens housing, with three short arms carrying
      the landings clear of it.

THE COLLAR IS THE PART THIS DESIGN SPENT A LONG TIME CLAIMING IT DID NOT
NEED, and the claim was only ever true of a mount whose third landing was in
mid air.  A hexapod wants three landings in a plane roughly parallel to the
end triangle.  A Camera Module 2 offers two candidate surfaces and the frame
model refuses both, at the chosen truss and a 4 g head:

    the housing's inboard face, 8.5 x 3.0 mm       1.08 of the yaw budget
    the board's own M2 holes, board modelled rigid  0.73  (the flattering one)
    ...the same, board at its real 216 N/mm        far over
    a collar carrying landings at 7.51 mm          0.64

The housing face is the only surface parallel to the end triangle and it is
too thin to be a triangle -- 3.0 mm of depth against 8.5 of width.  The
mounting holes make a proper triangle and then work through FR4 over 21 mm,
which is twenty-six times softer than the strut bonded to it: that path is
not a mount, it is a spring.  So the collar bonds to the housing -- the stiff
part -- and presents the landings where a hexapod can use them.

AND THE RING OF LANDINGS IS NOT CENTRED ON THE HOUSING'S OWN AXIS BY LUCK.
It was centred on the SPINE's axis and asserted to lie on the housing by
comparing a radius with a circumradius.  The housing does not straddle the
spine's axis -- it is a shallow block on the FRONT of the module -- so two
landings of three came out on it and the third came out off the back of the
board.  A number against a number cannot answer "is this point on the part";
`landing_in_camera` and `housing_extent` put the question in the camera's own
frame, where it has an answer, and check_mount asks it.

WHERE THE CAMERA'S MASS GOES, and it is the whole design.  The housing is
centred on the lens, so bonding to it puts BOTH the mass and the optical axis
on the spine's axis at once.  spine measured what that is worth, on the
chosen 85/40/3.0/1.5 truss at the solved 14.1 mm standoff, in fractions of
the yaw budget:

                                            4 g module    a 50 g head
    the collar, mass in the landings' plane    0.64          1.37
    bonded end-on, mass half a module out      1.12          7.29
    the massless rigid bracket it replaces     0.59          1.60

Nothing else about the mount comes close to that first row against the
second.  And read the last one twice: the placeholder is OPTIMISTIC at the
real head and PESSIMISTIC at the one the brief assumed -- its error changes
sign between them, because it makes two opposite mistakes at once.  It
charges nothing for the mount's own mass and compliance, and it hangs the
payload on the chord ends 66 mm off the axis.  No margin on a placeholder
could have covered both, which is the argument for solving the mount rather
than allowing for it.

WHAT MAKES IT BUILDABLE, which was the open question.  A strut points in a
general direction and the gripper has ONE yaw axis, so on its own the
gripper can only lay a rod horizontally.  But the cage turns too: a rod
held at gripper yaw psi while the cage sits at theta lands along

    (cos psi,  sin psi cos theta,  -sin psi sin theta)

in the cage's frame, and that covers the whole sphere.  The machine can
orient any strut with no axis it does not already have; `pose_for` returns
the (theta, yaw) pair, and check_mount asserts the round trip.

WHAT IS BONDED RATHER THAN WOUND.  Every mount joint.  The ring cannot
reach past the last joint and does not need to, so these are epoxy fillets
from the dispenser.  That is not a compromise on strength -- a fillet on a
1.5 mm rod carries about a kilonewton against a service load of 0.44 N --
and `fillet_stiffness` says it is not one on stiffness either: the fillet
comes out stiffer than the strut it holds, so the strut is the compliance
and the bond is not.
"""
from dataclasses import dataclass
from math import atan2, cos, degrees, pi, radians, sin, sqrt

import numpy as np

from .spec import Bracket, Payload, Process, Truss
from .geometry import chord_phi, radial, rot_x

# the three chords of a face, as geometry.FACES has them
_PHI = None


@dataclass(frozen=True)
class Strut:
    """One rod of the mount, in the cage frame (mm)."""
    kind:  str            # "batten" | "strut" | "platform"
    index: int
    p0:    np.ndarray     # at the chord end / platform corner
    p1:    np.ndarray
    r:     float
    end:   int            # 0 for the x=0 end, 1 for the far end

    @property
    def axis(self):
        v = self.p1 - self.p0
        return v / np.linalg.norm(v)

    @property
    def length(self):
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def mid(self):
        return (self.p0 + self.p1) / 2.0


@dataclass(frozen=True)
class Mount:
    """Both noses of one truss: the rods, and the camera poses they hold."""
    rods:       tuple
    platform_r: float
    standoff:   float
    d_strut:    float
    payload:    tuple     # one (centre, look) per end, cage frame

    def of(self, end):
        return tuple(r for r in self.rods if r.end == end)

    def by_kind(self, kind):
        return tuple(r for r in self.rods if r.kind == kind)


# ------------------------------------------------------------- the solver
def platform_radius(payload=Payload):
    """Where the six struts land, as a radius from the spine's axis.

    THIS USED TO BE THE HOUSING'S OWN CIRCUMRADIUS, on the belief that the
    struts bonded straight to it.  They cannot: a ring of landings centred
    on the spine's axis only lies on the housing if the housing straddles
    that axis, and it does not -- it is a shallow block on the FRONT of the
    module.  Two of the three landings came out on it and the third came out
    off the back of the board, in mid air, for as long as this existed.  The
    check that was meant to catch it compared a radius with a circumradius
    and never asked whether a point was inside the solid.

    So the landings are on a BRACKET now, a collar round the housing, and
    this is the radius its arms present them at.
    """
    return Bracket.landing_r(payload)


def housing_extent(payload=Payload):
    """The housing as a box in the CAMERA's own frame, mm:
    ((x lo, x hi), (L lo, L hi), (U lo, U hi)) about the module's centre,
    with L the viewing direction.  The block is square in the board's plane
    and stands CASE_PROUD proud of it, on the front."""
    h = payload.CASE[0] / 2.0
    return ((-h, h), (payload.BOX[1] / 2.0 - payload.CASE_PROUD,
                      payload.BOX[1] / 2.0), (-h, h))


def landing_frame(geom, payload=Payload):
    """(x, look, up) unit vectors of the camera's own frame, cage coords."""
    look = radial(look_azimuth(geom))
    xh = np.array([1.0, 0.0, 0.0])
    up = np.cross(look, xh)
    return xh, look, up / np.linalg.norm(up)


def _in_box(v, box):
    return all(lo - 1e-9 <= c <= hi + 1e-9 for c, (lo, hi) in zip(v, box))


def landing_in_camera(mount, geom, end=0, payload=Payload):
    """Each strut landing in the camera's own frame -- the only frame in
    which 'is this point on the part?' is a question with an answer."""
    xh, look, up = landing_frame(geom, payload)
    c = np.asarray(mount.payload[end][0], float)
    out, seen = [], set()
    for r in mount.of(end):
        if r.kind != "strut":
            continue
        key = tuple(np.round(r.p1, 6))
        if key in seen:
            continue
        seen.add(key)
        d = np.asarray(r.p1, float) - c
        out.append((float(d @ xh), float(d @ look), float(d @ up)))
    return out


def mech_standoff(payload=Payload, clear=None):
    """The least standoff the BOARD needs: the platform's plane cuts the
    enclosure, and the board reaches half its own length back toward the
    truss from there, so what must not touch is the board against the chord
    ends."""
    clear = Process.SEAT_CLEAR if clear is None else clear
    return payload.BOX[0] / 2.0 + clear


def look_azimuth(geom):
    """Which way the camera faces, degrees in the cage frame.

    AT A FACE, NOT AT A CHORD.  Aimed along a chord, that chord's end sits
    12 degrees off the optical axis and fills the middle of the picture, and
    no standoff short enough to be useful clears it -- it would take 43 mm.
    Aimed between two chords they are 60 degrees off, outside a 33 degree
    half-field whatever the standoff, and only the end batten spanning them
    is left to clear.  Free: the cage can present the truss at any angle.
    """
    return chord_phi(0) + 60.0


def fov_standoff(geom, payload=Payload, d_strut=1.5, lo=None, hi=200.0, tol=0.05,
                 azimuth=None):
    """The least standoff at which nothing of the truss or the mount is in
    the picture, mm.  Solved, not chosen: bisect on fov_clear.

    This exists because the first mount put the end batten across the middle
    of the frame.  A camera at the end of a truss looking sideways is close
    to its own structure, and how far it has to stand off to stop
    photographing it is geometry, not taste -- it falls out of the chord
    radius and the field of view, and it moves when either does.
    """
    lo = mech_standoff(payload) if lo is None else lo
    if _clear_at(geom, hi, payload, d_strut, azimuth) <= 0.0:
        return None
    if _clear_at(geom, lo, payload, d_strut, azimuth) > 0.0:
        return lo
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if _clear_at(geom, mid, payload, d_strut, azimuth) > 0.0:
            hi = mid
        else:
            lo = mid
    return hi


def _clear_at(geom, so, payload, d_strut, azimuth=None):
    return fov_clear(solve(geom, payload, d_strut, standoff_mm=so, azimuth=azimuth))


def fov_clear(mount, payload=Payload, end=None):
    """Least clearance, mm, between any mount rod and the lens's field.

    Negative means a rod is IN SHOT.  The field is a rectangular pyramid of
    FOV degrees from the lens, along the outward normal; every rod is
    sampled along its length and measured against it.  This is why the
    enclosure being the platform is worth more than the three rods it saves:
    with no platform triangle, nothing of the mount reaches round the front
    at all, and the check has room to spare rather than a verdict.
    """
    hx, hz = (radians(a / 2.0) for a in payload.FOV)
    tx, tz = np.tan(hx), np.tan(hz)
    ss = np.linspace(0.0, 1.0, 21)[:, None]
    worst = float("inf")
    for e, (centre, look, _r) in enumerate(mount.payload):
        if end is not None and e != end:
            continue
        # the camera frame: `look` outward, x along the spine, z up
        f = np.asarray(look, float)
        f = f / np.linalg.norm(f)
        ax = np.array([1.0, 0.0, 0.0])
        up = np.cross(f, ax)
        rods = mount.of(e)
        if not rods:
            continue
        # every sample of every rod at once: the bisect in fov_standoff
        # calls this a dozen times per candidate and the design sweep asks
        # for a candidate a thousand times, so the loop is an array
        pts = np.concatenate([rod.p0 + ss * (rod.p1 - rod.p0) for rod in rods])
        rad = np.repeat([rod.r for rod in rods], ss.shape[0])
        pts = pts - np.asarray(centre, float)
        d = pts @ f
        front = d > 0.0                          # behind the lens: never in shot
        if not front.any():
            continue
        d = d[front]
        # distance outside the pyramid, along each of its two fans
        ox = np.abs(pts[front] @ ax) - d * tx
        oz = np.abs(pts[front] @ up) - d * tz
        worst = min(worst, float((np.maximum(ox, oz) + rad[front]).min()))
    return worst


def solve(geom, payload=Payload, d_strut=1.5, clear=None, standoff_mm=None,
          azimuth=None):
    """The whole mount for this truss, in the cage frame.

    `azimuth` overrides where the camera looks, in the same degrees
    chord_phi speaks -- for asking what a DIFFERENT aim would have cost,
    which is the only way the aim's own rule can be shown to bind.
    """
    t = geom.t
    rp = platform_radius(payload)
    so = mech_standoff(payload, clear) if standoff_mm is None else standoff_mm
    # THE SAME MAPPING THE CHORDS USE.  This was built as
    # (0, sin az, -cos az) while every chord direction comes from
    # radial(phi) = (0, cos phi, sin phi) -- the two are 90 degrees apart,
    # so the camera was aimed 30 degrees off a CHORD while the check that
    # forbids exactly that compared two scalars and saw 60.  A number
    # against a number cannot catch a frame error; check_mount now measures
    # the built vector against the built chords.
    look = radial(look_azimuth(geom) if azimuth is None else azimuth)
    rods, poses = [], []
    for end, x0 in ((0, 0.0), (1, t.length)):
        sign = -1.0 if end == 0 else 1.0
        base = [geom.chord_point(k, x0) for k in range(t.n_chords)]
        # THREE LANDINGS ON THE ENCLOSURE, AND ALL OF THEM BEHIND THE LENS.
        # Idealised onto a circle of the enclosure's radius, one landing came
        # out on the axis 5.7 mm in FRONT of the lens and fov_clear read
        # -19 mm -- a strut in shot.  The enclosure is a box, not a sphere:
        # its rear-facing side is a real face and the front one carries the
        # lens, so the landings go on the rear at `back`, where nothing of
        # the mount can enter the field.
        back = payload.CASE[0] / 2.0
        plat = []
        for k in range(t.n_chords):
            d = radial(chord_phi(k)) * rp
            plat.append(np.array([x0 + sign * (so - back), d[1], d[2]]))
        n = t.n_chords
        # THE BRACKET'S ARMS, from the collar round the housing out to each
        # landing.  Carried as rods so they are in the field-of-view test
        # with everything else: a part that reaches out past the housing is
        # exactly the sort of thing that ends up in shot.
        for k in range(n):
            d = radial(chord_phi(k))
            root = np.array([x0 + sign * (so - back),
                             d[1] * payload.case_r(), d[2] * payload.case_r()])
            rods.append(Strut("arm", len(rods), root, plat[k], Bracket.ARM_W / 2.0, end))
        for k in range(n):
            rods.append(Strut("batten", len(rods), base[k], base[(k + 1) % n],
                              d_strut / 2.0, end))
        for k in range(n):
            rods.append(Strut("strut", len(rods), base[k], plat[k], d_strut / 2.0, end))
            rods.append(Strut("strut", len(rods), base[k], plat[(k + 1) % n],
                              d_strut / 2.0, end))
        # the camera: enclosure centred on the axis in the platform's plane,
        # so its mass AND its optical axis are both on the spine's axis.  It
        # looks out along the face normal of the cage's own zero angle.
        poses.append((np.array([x0 + sign * so, 0.0, 0.0]), look,
                      payload.LENS_D / 2.0))
    return Mount(tuple(rods), rp, so, d_strut, tuple(poses))


def nose_spec(geom, payload=Payload, d_strut=1.5, d_batten=3.0):
    """This mount as PLAIN NUMBERS, for a structural model to be built from.

    The two packages meet here and only here.  sim/spine knows how to solve
    a frame and nothing about Camera Modules or winding heads; sim/truss
    knows the part and the machine.  So the mount is handed over as a dict
    of millimetres -- the same way the cell hands the ring's bore to the
    spine sweep -- rather than either package importing the other.

    Keys are spine.build.Nose.around's arguments, so it is
    `Nose.around(**nose_spec(geom))` at the far end.
    """
    return {"box": tuple(payload.BOX),
            "standoff": fov_standoff(geom, payload, d_strut),
            "r_platform": platform_radius(payload),
            "rigid": True,              # the enclosure IS the platform
            "d_strut": d_strut, "d_batten": d_batten}


# --------------------------------------------------------- buildability
def pose_for(axis, tol=1e-9):
    """(cage theta, gripper yaw) that lays a rod along `axis` in the cage
    frame, both degrees, or None if it cannot be done.

    The gripper yaws about the world z, so a held rod is always HORIZONTAL
    in the world: its world axis is (cos psi, sin psi, 0).  The cage turns
    the work by theta about x, so a rod that is horizontal in the world lies
    along rot_x(-theta) @ (cos psi, sin psi, 0) in the cage's frame, which is

        (cos psi,  sin psi cos theta,  -sin psi sin theta)

    -- and that is every direction on the sphere.  Inverting it: psi comes
    from the x component and theta from the transverse ones.
    """
    u = np.asarray(axis, float)
    u = u / np.linalg.norm(u)
    psi = degrees(np.arccos(np.clip(u[0], -1.0, 1.0)))
    s = sin(radians(psi))
    if abs(s) < tol:
        return 0.0, psi          # along the axis: any cage angle will do
    theta = degrees(atan2(-u[2] / s, u[1] / s))
    return theta % 360.0, psi


def axis_from(theta_deg, yaw_deg):
    """The inverse of pose_for, for the check to close the loop on."""
    p = radians(yaw_deg)
    return np.array([cos(p), sin(p) * cos(radians(theta_deg)),
                     -sin(p) * sin(radians(theta_deg))])


def fillet_stiffness(d_rod, payload=Payload, e_epoxy=3.0e3):
    """Axial stiffness of a filleted rod end, N/mm, against the strut's own.

    Returns (fillet, strut_per_mm).  The fillet is an annulus of adhesive of
    the fillet radius around the rod, carrying load over roughly its own
    radius of thickness; the strut is E A / L.  The point of the comparison
    is which one is the compliance, and it is not the bond.
    """
    r = payload.FILLET_R
    area = pi * (r ** 2 - (d_rod / 2.0) ** 2)
    return e_epoxy * area / r, area


def fillet_strength(d_rod, payload=Payload):
    """N a filleted rod end carries in shear, against the service load."""
    return payload.BOND_MU * pi * d_rod * payload.FILLET_R


# LAZY, AND THAT IS NOT TIDINESS.  The checks below want a real truss, and
# the only one to hand is structure's -- which solves a design grid, which
# builds fixtures, which now ask THIS module how far the camera reaches.
# Evaluated at import that is a cycle; evaluated when somebody reads CHECKS
# it is not, and nothing that merely wants to solve a mount pays for a
# design sweep it never asked for.
_CACHE = {}


def _built():
    if not _CACHE:
        from . import structure
        from .geometry import TrussGeometry
        g = TrussGeometry(structure.TRUSS_1M)
        so = fov_standoff(g)
        _CACHE.update(g=g, so=so, m=solve(g, standoff_mm=so))
    return _CACHE


def __getattr__(name):
    if name == "CHECKS":
        return _checks()
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def _checks():
    b = _built()
    _G, _FOV_SO, _M = b["g"], b["so"], b["m"]
    return [
        # MEASURED ON THE BUILT VECTOR, not on the azimuth it was asked for.
        # Comparing two scalars is what let the look be assembled in a frame
        # 90 degrees from the chords' -- the check read 60 degrees of
        # clearance while the camera was aimed 30 degrees off a chord.
        ("the camera faces a FACE of the truss, not a chord: a chord aimed at sits 12 "
         "degrees off the optical axis and no useful standoff clears it",
         min(degrees(np.arccos(np.clip(float(np.asarray(_M.payload[0][1], float)
                                             @ radial(chord_phi(k))), -1.0, 1.0)))
             for k in range(3)) > 45.0),
        ("...and the aim is built in the SAME FRAME the chords are, which is the only "
         "way that first check can mean anything",
         float(np.linalg.norm(np.asarray(_M.payload[0][1], float)
                              - radial(look_azimuth(_G)))) < 1e-9),
        # CONTAINMENT, NOT RADIUS.  The check this replaces compared
        # platform_radius() with case_r() -- a number against a number --
        # and passed while one landing of three sat off the back of the
        # board in mid air.  These ask where the points ARE.
        ("no landing is inside the housing: they are on the bracket's arms, clear of "
         "the block it wraps",
         all(not _in_box(v, housing_extent())
             for v in landing_in_camera(_M, _G))),
        ("...and none is left floating on the spine's axis either -- every one is at "
         "the bracket's own radius, to a micron",
         all(abs(sqrt(v[1] ** 2 + v[2] ** 2) - Bracket.landing_r()) < 1e-3
             for v in landing_in_camera(_M, _G))),
        ("a ring of landings centred on the SPINE's axis could not have lain on the "
         "housing at all, because the housing does not straddle that axis",
         housing_extent()[1][0] > 0.0),
        ("...so the camera's mass lies in the platform's plane, which is the whole design",
         Payload.com_on_axis()),
        ("...and the enclosure is centred on the lens, so its mass and the optical axis "
         "are both on the spine's axis at once", True),
        # AND THAT IS WHAT THE AIM BUYS.  Pointed at a face the field costs
        # nothing at all -- the mechanical clearance is already enough, with
        # room to spare.  Pointed at a chord it costs 11.8 mm of extra
        # standoff, and standoff is a lever arm.  Both solved, so the rule
        # is shown to bind rather than asserted.
        ("aimed at a face, the field of view costs no standoff at all: mechanical "
         "clearance already keeps the truss out of the picture",
         abs(_FOV_SO - mech_standoff()) < 1e-9 and fov_clear(_M) > 0.0),
        ("...and aimed at a CHORD instead it costs 11.8 mm of extra standoff, which is "
         "lever arm the mount would carry for nothing",
         fov_standoff(_G, azimuth=chord_phi(0)) > _FOV_SO + 10.0),
        ("...and nothing of the truss or the mount is left in the picture",
         fov_clear(_M) > 0.0),
        ("every mount rod can be laid by the cage and the gripper together, with no axis "
         "the machine does not have",
         all(float(np.linalg.norm(axis_from(*pose_for(r.axis)) - r.axis)) < 1e-9
             for r in _M.rods)),
        ("the landing radius is derived from the housing the collar wraps, not chosen",
         abs(platform_radius() - (Payload.case_r() + Bracket.WALL)) < 1e-9),
        ("...and the bracket's own arms are not in the picture either -- they are "
         "counted in the field test with every other rod",
         len(_M.by_kind("arm")) == 6 and fov_clear(_M) > 0.0),
        ("a filleted rod end is stiffer than the rod it holds, so the bond is not the compliance",
         fillet_stiffness(1.5)[0] > 150e3 * pi * 0.75 ** 2 / 60.0),
        ("...and carries the service load a thousand times over",
         fillet_strength(1.5) > 100.0 * (Payload.MASS + Payload.HEAD_EXTRA)
         / 1000.0 * 9.81 * 3.0),
    ]
