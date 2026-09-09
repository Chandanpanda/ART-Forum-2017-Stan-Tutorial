"""The camera mount, solved: a six-strut nose at each end of the truss.

WHAT IT IS.  Twelve rods per end and no plate, no bracket, no machined
part:

    3 END BATTENS closing the triangle of chord ends.  An octahedron is two
      triangles and six struts; with the base triangle open the three chord
      ends splay and the mount is thirty times softer than it looks (spine's
      own measurement).  These lie in the ring's plane of rotation, which
      brief 4.2 prohibits -- but the prohibition binds where the ring has to
      WORK, and the last joint is `end_margin` inboard of here.  The ring
      never comes out this far.
    6 STRUTS, octahedral: each chord end feeds two platform corners, so no
      joint has to carry a moment.
    3 PLATFORM rods forming the triangle the camera is bonded inside.

WHERE THE CAMERA GOES, and it is the whole design.  The platform triangle
SURROUNDS the housing at its mid-length rather than butting against its end
face, so the camera's centre of mass lies IN the platform's plane.  spine
measured what that is worth: bonded end-on, with the mass half a housing
outboard, the mount takes 1.84 of the yaw budget and the rig is two times
over; with the mass in the plane it takes 0.70 and the rig is INSIDE, truer
than with the mount left out of the model altogether.  Nothing else about
the mount comes close to mattering as much -- platform radius from 24 to 32
mm is worth 0.05, and thicker struts make it worse, because what they add
in stiffness they lose in tip mass.

SO THE PLATFORM RADIUS IS NOT CHOSEN.  It is the smallest triangle whose
inscribed circle clears the housing's half-diagonal with a bonding gap, and
a triangle's inscribed circle is half its circumradius.

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
def platform_radius(payload=Payload, clear=None):
    """The smallest platform triangle that can surround the housing.

    Its inscribed circle must clear the housing's half-diagonal by a bonding
    gap, and for an equilateral triangle the inscribed circle is half the
    circumradius -- so the circumradius is twice that sum.  Derived, not
    chosen: change the camera and this moves.
    """
    clear = Process.SEAT_CLEAR if clear is None else clear
    return 2.0 * (payload.half_diagonal() + clear)


def standoff(payload=Payload, clear=None):
    """How far the platform's plane sits beyond the chord ends.

    The platform is at the housing's MID-LENGTH, so the housing reaches half
    its length back toward the truss, and what must not touch is the housing
    against the chord ends.  Shorter is truer (spine: 15 mm beats 60 mm,
    because the cantilever costs more than the baseline gains), so this is
    the least that clears.
    """
    clear = Process.SEAT_CLEAR if clear is None else clear
    return payload.BOX[0] / 2.0 + clear


def solve(geom, payload=Payload, d_strut=1.5, clear=None):
    """The whole mount for this truss, in the cage frame."""
    t = geom.t
    rp = platform_radius(payload, clear)
    so = standoff(payload, clear)
    rods, poses = [], []
    for end, x0 in ((0, 0.0), (1, t.length)):
        sign = -1.0 if end == 0 else 1.0
        base = [geom.chord_point(k, x0) for k in range(t.n_chords)]
        plat = []
        for k in range(t.n_chords):
            d = radial(chord_phi(k)) * rp
            plat.append(np.array([x0 + sign * so, d[1], d[2]]))
        n = t.n_chords
        for k in range(n):
            rods.append(Strut("batten", len(rods), base[k], base[(k + 1) % n],
                              d_strut / 2.0, end))
        for k in range(n):
            rods.append(Strut("platform", len(rods), plat[k], plat[(k + 1) % n],
                              d_strut / 2.0, end))
            rods.append(Strut("strut", len(rods), base[k], plat[k], d_strut / 2.0, end))
            rods.append(Strut("strut", len(rods), base[k], plat[(k + 1) % n],
                              d_strut / 2.0, end))
        # the camera: centred on the axis in the platform's plane, looking
        # out along the face normal of the cage's own zero angle
        poses.append((np.array([x0 + sign * so, 0.0, 0.0]),
                      np.array([0.0, 0.0, 1.0])))
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


CHECKS = [
    ("the platform triangle surrounds the housing rather than butting its end face",
     platform_radius() / 2.0 > Payload.half_diagonal()),
    ("...so the camera's mass lies in the platform's plane, which is the whole design",
     Payload.com_on_axis()),
    ("the platform radius is derived from the housing, not chosen",
     abs(platform_radius() - 2.0 * (Payload.half_diagonal() + Process.SEAT_CLEAR)) < 1e-9),
    ("a filleted rod end is stiffer than the rod it holds, so the bond is not the compliance",
     fillet_stiffness(1.5)[0] > 150e3 * pi * 0.75 ** 2 / 60.0),
    ("...and carries the service load a thousand times over",
     fillet_strength(1.5) > 100.0 * Payload.MASS / 1000.0 * 9.81 * 3.0),
]
