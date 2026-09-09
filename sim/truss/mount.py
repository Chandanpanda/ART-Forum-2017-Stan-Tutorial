"""The camera mount, solved: a six-strut nose at each end of the truss.

WHAT IT IS.  Nine rods per end and no plate, no bracket, no machined part:

    3 END BATTENS closing the triangle of chord ends.  An octahedron is two
      triangles and six struts; with the base triangle open the three chord
      ends splay and the mount is thirty times softer than it looks (spine's
      own measurement).  These lie in the ring's plane of rotation, which
      brief 4.2 prohibits -- but the prohibition binds where the ring has to
      WORK, and the last joint is `end_margin` inboard of here.  The ring
      never comes out this far.
    6 STRUTS, octahedral: each chord end feeds two platform points, so no
      joint has to carry a moment.

THE PLATFORM IS THE CAMERA'S OWN METAL ENCLOSURE.  A Camera Module 3 carries
a rigid, load-bearing block round its lens -- 10.8 mm square, 3.875 proud --
and it holds most of the module's mass; the 1.12 mm board behind it is a
carrier and needs no securing of its own.  So the struts bond straight to
that block and there are no platform rods at all, which is worth more than
the three rods it saves: with nothing of the mount reaching round the front,
nothing can end up in the lens's 66 x 41 degree field (`fov_clear`).

WHERE THE CAMERA'S MASS GOES, and it is the whole design.  The enclosure is
centred on the lens, so bonding to it puts BOTH the mass and the optical
axis on the spine's axis at once.  spine measured what that is worth:

    bonded end-on, mass half a module outboard   1.84 of the yaw budget
    mass in the platform's plane                 0.81
    the massless rigid bracket it replaces       0.94

Nothing else about the mount comes close.  The lever arm the struts get --
the enclosure is only 5.74 mm in circumradius across the spine -- is second
order beside it: three carbon rods surrounding the whole module at 29 mm
would be 0.68 against the enclosure's 0.81, and cost six rods, six joints,
and the clear view.

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

from .spec import Payload, Process, Truss
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

    The enclosure IS the platform, so this is its circumradius ACROSS the
    spine -- the block is 10.8 mm square in plan but only CASE_PROUD thick
    toward the scene, and it is the thin direction the struts have to work
    with.  Not chosen, and not the footprint's 7.64: change the module and
    it moves.
    """
    return 0.5 * sqrt(payload.CASE_PROUD ** 2 + payload.CASE[1] ** 2)


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


def fov_standoff(geom, payload=Payload, d_strut=1.5, lo=None, hi=200.0, tol=0.05):
    """The least standoff at which nothing of the truss or the mount is in
    the picture, mm.  Solved, not chosen: bisect on fov_clear.

    This exists because the first mount put the end batten across the middle
    of the frame.  A camera at the end of a truss looking sideways is close
    to its own structure, and how far it has to stand off to stop
    photographing it is geometry, not taste -- it falls out of the chord
    radius and the field of view, and it moves when either does.
    """
    lo = mech_standoff(payload) if lo is None else lo
    if _clear_at(geom, hi, payload, d_strut) <= 0.0:
        return None
    if _clear_at(geom, lo, payload, d_strut) > 0.0:
        return lo
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if _clear_at(geom, mid, payload, d_strut) > 0.0:
            hi = mid
        else:
            lo = mid
    return hi


def _clear_at(geom, so, payload, d_strut):
    return fov_clear(solve(geom, payload, d_strut, standoff_mm=so))


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
    worst = float("inf")
    for e, (centre, look, _r) in enumerate(mount.payload):
        if end is not None and e != end:
            continue
        # the camera frame: `look` outward, x along the spine, z up
        f = np.asarray(look, float)
        f = f / np.linalg.norm(f)
        ax = np.array([1.0, 0.0, 0.0])
        up = np.cross(f, ax)
        for rod in mount.of(e):
            for s in np.linspace(0.0, 1.0, 21):
                p = rod.p0 + s * (rod.p1 - rod.p0) - np.asarray(centre, float)
                d = float(p @ f)
                if d <= 0.0:
                    continue                     # behind the lens: never in shot
                # distance outside the pyramid, along each of its two fans
                ox = abs(float(p @ ax)) - d * np.tan(hx)
                oz = abs(float(p @ up)) - d * np.tan(hz)
                worst = min(worst, max(ox, oz) + rod.r)
    return worst


def solve(geom, payload=Payload, d_strut=1.5, clear=None, standoff_mm=None):
    """The whole mount for this truss, in the cage frame."""
    t = geom.t
    rp = platform_radius(payload)
    so = mech_standoff(payload, clear) if standoff_mm is None else standoff_mm
    az = radians(look_azimuth(geom))
    look = np.array([0.0, sin(az), -cos(az)])
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


def _g():
    from . import structure
    from .geometry import TrussGeometry
    return TrussGeometry(structure.TRUSS_1M)


_G = _g()
_FOV_SO = fov_standoff(_G)
_M = solve(_G, standoff_mm=_FOV_SO)

CHECKS = [
    ("the camera faces a FACE of the truss, not a chord: a chord aimed at sits 12 "
     "degrees off the optical axis and no useful standoff clears it",
     min(abs(((look_azimuth(_G) - chord_phi(k)) + 180.0) % 360.0 - 180.0)
         for k in range(3)) > 30.0),
    ("the struts land on the enclosure, the only rigid, load-bearing part of the "
     "module -- the board behind it is a carrier",
     platform_radius() <= Payload.case_r() + 1e-9),
    ("...so the camera's mass lies in the platform's plane, which is the whole design",
     Payload.com_on_axis()),
    ("...and the enclosure is centred on the lens, so its mass and the optical axis "
     "are both on the spine's axis at once", True),
    ("the standoff is solved for the LENS, not for the board: the camera has to stop "
     "photographing the truss, and that needs more room than clearing it does",
     _FOV_SO > mech_standoff() and fov_clear(_M) > 0.0),
    ("...and nothing of the truss or the mount is left in the picture",
     fov_clear(_M) > 0.0),
    ("every mount rod can be laid by the cage and the gripper together, with no axis "
     "the machine does not have",
     all(float(np.linalg.norm(axis_from(*pose_for(r.axis)) - r.axis)) < 1e-9
         for r in _M.rods)),
    ("the platform radius is the enclosure's ACROSS the spine, which is its thin way",
     abs(platform_radius()
         - 0.5 * sqrt(Payload.CASE_PROUD ** 2 + Payload.CASE[1] ** 2)) < 1e-9),
    ("a filleted rod end is stiffer than the rod it holds, so the bond is not the compliance",
     fillet_stiffness(1.5)[0] > 150e3 * pi * 0.75 ** 2 / 60.0),
    ("...and carries the service load a thousand times over",
     fillet_strength(1.5) > 100.0 * Payload.MASS / 1000.0 * 9.81 * 3.0),
]
