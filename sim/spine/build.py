"""The candidate spines, as frames: a Warren truss and the tube it must beat.

Both come back as a Model with the same three landmarks -- a `left` and a
`right` camera face and a central mount -- so every metric downstream is
computed the same way for both.  That is the only way the comparison
means anything.

Millimetres stop here: the arguments are in millimetres because that is
how stock and drawings are quoted, and everything returned is SI.

THE LATTICE.  Three chords on a circle of radius R = side/sqrt(3), running
the full length uncut, and a web between each adjacent pair.  `web`
chooses which web, and the choice turns out to matter more than any
dimension:

    "warren"  one diagonal per bay per face, the zigzag the brief
              specifies.  Per bay this is 3 chords + 3 diagonals for 3 new
              nodes: SIX members where rigidity needs NINE.  A pin-jointed
              count over the whole 1 m spine gives 60 members and 9
              supports against 99 freedoms, so it is thirty mechanisms
              short and stands up only by bending 2 and 3 mm rods.  Its
              first four modes are the section triangle breathing.
    "x"       both diagonals of each bay in each face.  Nine members per
              bay for three nodes: exactly rigid, and still no member
              crosses a chord at ninety degrees, so the winding head can
              still reach every joint.
    "batten"  the warren zigzag plus a transverse member closing the
              triangle at each station.  Also exactly rigid, lighter than
              "x", and NOT BUILDABLE by the cell in sim/truss: a batten
              lies in the ring's plane of rotation, which is the one thing
              brief section 4.2 prohibits.  It is here as the control that
              says the softness is the topology and not the solver.
"""
from dataclasses import dataclass

import numpy as np

from .model import Model, Member, Section, PointMass
from . import material as mat

MM = 1e-3


def _stations(length_m, pitch_m):
    """Bay stations along the spine, symmetric about the centre, with a
    station AT the centre so the mount has somewhere to hold."""
    half = length_m / 2.0
    n = max(int(round(half / pitch_m)), 1)
    xs = np.linspace(-n * pitch_m, n * pitch_m, 2 * n + 1)
    # squeeze the lattice to the length asked for rather than overrun it
    return xs * (half / (n * pitch_m)) + half


WEBS = ("warren", "x", "batten")


@dataclass(frozen=True)
class Nose:
    """A hexapod camera mount at each end of the spine, all mm.

    THE PLATFORM IS PERPENDICULAR TO THE SPINE'S AXIS, parallel to the
    triangle of chord ends it is strutted from.  Two PARALLEL triangles and
    six struts is the octahedron and it is well conditioned; two
    PERPENDICULAR triangles and six struts is not, and the difference is a
    factor of three in yaw -- 0.50 of the budget against 1.43, measured.

    WHICH FACE OF THE CAMERA BONDS TO IT is what took three attempts.  The
    camera looks along the model's +y (metrics.Pose), so its back face is an
    x-z plane and cannot lie against a y-z platform: a plane containing x
    crosses a plane at constant x in a LINE, so it would meet the platform's
    three rods at two points, not three lines.  But a cuboid has more than
    one face.  Its INBOARD END face is a y-z plane -- the platform's own
    plane -- and bonded there the housing gets three long line contacts and
    still looks sideways.  So the camera is bonded end-on and hangs
    outboard, which is why `payload_x` is here: the mass ends up half a
    housing beyond the platform, and that offset is a moment arm under a
    lateral manoeuvre which a mass sitting on the platform would not have.

    AND `payload_x` DECIDES EVERYTHING.  It is how far the camera's centre of
    mass sits outboard of the platform's plane, and it is the whole mount
    problem:

        payload_x = 12.5 (bonded end-on)      1.84 of the budget
        payload_x = 0    (CoM in the plane)   0.67

    -- against 0.94 for the massless rigid placeholder the mount replaces.
    (Both at d_strut 1.5.  DO NOT read the strut diameter off that sweep:
    see the null below.)
    Got right the mount does not cost accuracy, it BUYS it, because the
    placeholder hung the camera's mass on the chord ends 66 mm off the axis
    where it drives the end triangle, and the nose brings it onto the axis.
    Got wrong by half a housing it nearly triples the mount's share and puts
    the rig two times over budget.

    So the platform triangle SURROUNDS the housing at its mid-length rather
    than butting against its end face, and r_platform is then not free
    either: the triangle's inscribed circle (r_platform / 2) has to clear
    the housing's half-diagonal.  Beyond that, r_platform barely matters --
    24 to 32 mm is 0.67 to 0.72.

    A NULL, WHICH IS NOT AN OPTIMUM.  Swept on strut diameter the yaw under
    a manoeuvre reads 2714, 523, 994, 2118, 3041, 3840 microdegrees at 0.8,
    1.0, 1.2, 1.5, 2.0, 3.0 mm -- a dip at 1.0 that looks like the answer and
    is not.  Signed, it goes -2714, -1338, -337, +415, ... : it CHANGES SIGN
    between 1.0 and 1.1, because two contributions oppose there -- the
    truss's own end rotation carried in through the struts, and the camera's
    inertia deflecting them.  1.0 mm is a stocked diameter, so this is a trap
    laid exactly where a designer would step.  The null moves with every
    guess it depends on: at 1.0 mm a camera of 15 g instead of 50 flips the
    sign again (-337 to +386).  So the diameter is chosen AWAY from it, and
    check_spine asserts the sign change is still there rather than letting
    someone rediscover the dip as a feature.
    """
    standoff:    float = 15.0
    r_platform:  float = 24.0
    # THE END BATTEN IS WHAT SETS THE FIRST MODE, and it is not the struts.
    # The mode at 180 Hz is the truss's own section triangle breathing at
    # mid-span -- the nose's nodes move at 0.17 of the amplitude, so the nose
    # is loading that mode, not causing it.  What the nose changed is how
    # well the END TRIANGLE is closed: it replaced the fictitious bracket, a
    # 20 x 3 mm section, with a rod, and 20 x 3 has 180 times the second
    # moment of a 1.5 mm rod.  So this diameter is swept separately.
    d_batten:    float = 3.0
    payload_x:   float = 0.0       # the camera's CoM, outboard of the platform
    payload_box: tuple = (25.0, 11.5, 24.0)   # mm, along x / y / z [VERIFY]
    d_strut:     float = 1.5
    d_platform:  float = 1.5

    @staticmethod
    def around(box=None, standoff=15.0, d_strut=1.5, d_batten=3.0, clearance=1.5):
        """A nose whose platform triangle surrounds `box` (mm, x/y/z) at its
        mid-length, so the camera's centre of mass lies in the platform's
        plane.  The triangle's inscribed circle must clear the housing's
        half-diagonal in the y-z section, and a triangle's inscribed circle
        is half its circumradius."""
        box = (25.0, 11.3, 23.862) if box is None else box   # RP-008153-DS-1
        half_diag = 0.5 * (box[1] ** 2 + box[2] ** 2) ** 0.5
        return Nose(standoff=standoff, r_platform=2.0 * (half_diag + clearance),
                    payload_x=0.0, payload_box=tuple(box),
                    d_strut=d_strut, d_platform=d_strut, d_batten=d_batten)


def warren_truss(length, side, alpha, d_chord, d_diag, material=None,
                 tip_mass=0.0, pinned_diagonals=False, web="warren",
                 brackets=True, nose=None, name=None):
    """A three-chord lattice.  Lengths in mm, mass in g, angle in deg.

    `nose` mounts the camera on a HEXAPOD instead of on the chord ends:
    Nose(standoff, r_platform, d_strut).  Six struts run from the three
    chord ends to a platform triangle of three more rods, and the camera's
    mass sits on the platform, `standoff` mm beyond the truss.  Two things
    change and they pull opposite ways: the BASELINE grows by twice the
    standoff, which is a straight accuracy win because range error goes as
    Z^2/B, and the cantilever grows with it, which costs yaw.  Which wins is
    a measurement, not an argument -- sweep it.

    Without a nose the camera's mass sits on the chord ends and the end
    triangle is closed by a massless, infinitely stiff `bracket`.  That was
    always a placeholder for a mount nobody had designed, and it is
    optimistic in exactly the way a placeholder is: it charges the mount no
    compliance and no mass.
    """
    if web not in WEBS:
        raise ValueError("web must be one of %s" % (WEBS,))
    material = material or mat.CARBON_UD
    L = length * MM
    s = side * MM
    R = s / np.sqrt(3.0)
    pitch = s / np.tan(np.radians(alpha))
    xs = _stations(L, pitch)
    ns = len(xs)
    phi = np.radians([90.0, 210.0, 330.0])          # three chords at 120
    nodes = []
    for k in range(3):
        for x in xs:
            nodes.append([x, R * np.cos(phi[k]), R * np.sin(phi[k])])
    nid = lambda k, i: k * ns + i
    sec_c, sec_d = Section.rod(d_chord * MM), Section.rod(d_diag * MM)
    members = []
    for k in range(3):
        for i in range(ns - 1):
            members.append(Member(nid(k, i), nid(k, i + 1), sec_c, material, tag="chord%d" % k))
    diag = lambda i, j: Member(i, j, sec_d, material, pinned=pinned_diagonals, tag="diag")
    for f in range(3):
        a, b = f, (f + 1) % 3
        for i in range(ns - 1):
            lo, hi = (a, b) if i % 2 == 0 else (b, a)
            members.append(diag(nid(lo, i), nid(hi, i + 1)))
            if web == "x":
                members.append(diag(nid(hi, i), nid(lo, i + 1)))
    if web == "batten":
        for i in range(ns):
            for k in range(3):
                members.append(Member(nid(k, i), nid((k + 1) % 3, i), sec_d, material,
                                      pinned=pinned_diagonals, tag="batten"))
    left = [nid(k, 0) for k in range(3)]
    right = [nid(k, ns - 1) for k in range(3)]
    mid = ns // 2
    mount = [nid(k, mid) for k in range(3)]
    if nose is not None:
        # ---- THE HEXAPOD NOSE, one at each end.
        # Six struts, octahedral: each chord end feeds two platform corners,
        # so no joint has to carry a moment and the platform's six degrees of
        # freedom are all taken by strut axial stiffness.  The platform is
        # three more rods -- a triangle the camera is bonded inside -- and it
        # is there because it gives the struts a lever arm against a camera
        # YAW; the housing's own end face is what caps how big it can be.
        sec_s = Section.rod(nose.d_strut * MM)
        sec_p = Section.rod(nose.d_platform * MM)
        rp = nose.r_platform * MM
        payload_nodes = []
        for base, sign in ((left, -1.0), (right, +1.0)):
            # THE BASE TRIANGLE HAS TO BE CLOSED TOO.  An octahedron is two
            # triangles and six struts; leave the base one open and the
            # three chord ends splay, and the mount reads 30 times softer
            # than it is (measured: 262568 udeg against 8574).  The warren
            # web does not close it -- its last diagonals go back to the
            # previous station -- which is what the fictitious `bracket`
            # was standing in for.
            #
            # AND A REAL ROD MAY DO IT HERE, though brief 4.2 forbids a
            # member in the ring's plane of rotation: the prohibition binds
            # where the ring has to WORK, and the last joint is end_margin
            # inboard of this.  The ring never comes out this far.
            for k in range(3):
                members.append(Member(base[k], base[(k + 1) % 3],
                                      Section.rod(nose.d_batten * MM), material,
                                      tag="endbatten"))
            x0 = nodes[base[0]][0]
            plat = []
            for k in range(3):
                plat.append(len(nodes))
                nodes.append([x0 + sign * nose.standoff * MM,
                              rp * np.cos(phi[k]), rp * np.sin(phi[k])])
            for k in range(3):
                members.append(Member(plat[k], plat[(k + 1) % 3], sec_p, material,
                                      tag="platform"))
                members.append(Member(base[k], plat[k], sec_s, material, tag="strut"))
                members.append(Member(base[k], plat[(k + 1) % 3], sec_s, material,
                                      tag="strut"))
            # THE CAMERA'S MASS WHERE IT ACTUALLY IS: a node at the housing's
            # centre of mass, half a housing outboard of the platform, tied to
            # the platform by the housing itself.  A metal box is far stiffer
            # than these struts, so it is three stiff massless members -- but
            # it is NOT on the platform, and that offset is a moment arm the
            # first model did not charge for.
            com = len(nodes)
            nodes.append([x0 + sign * (nose.standoff + nose.payload_x) * MM, 0.0, 0.0])
            for k in range(3):
                members.append(Member(plat[k], com, Section.rect(0.010, 0.003),
                                      mat.BRACKET, tag="housing"))
            payload_nodes.append(com)
            if base is left:
                left = plat
            else:
                right = plat
        brackets = False        # the nose replaces it, with real members
    if brackets:
        # THE CAMERA BRACKET CLOSES THE END TRIANGLE.  A plate bolted to
        # three chord ends is a rigid body, and without it in the model the
        # end triangle distorts and a rigid-body fit to it reports a
        # rotation that is really a shape change.  Massless: the bracket's
        # own mass is inside the head mass.
        sec_b = Section.rect(0.020, 0.003)
        for ring in (left, right):
            for k in range(3):
                members.append(Member(ring[k], ring[(k + 1) % 3], sec_b,
                                      mat.BRACKET, tag="bracket"))
    if nose is not None:
        # on the housing's own centre of mass, not spread over the platform,
        # and WITH the box's rotational inertia -- see PointMass
        masses = [PointMass.box(n, tip_mass * 1e-3,
                                [v * MM for v in nose.payload_box])
                  for n in payload_nodes]
    else:
        masses = [PointMass(n, tip_mass * 1e-3 / 3.0) for n in left + right]
    m = Model(nodes, members, masses,
              faces={"left": left, "right": right, "mount": mount},
              name=name or "%s %.0f/%.0f/%.1f/%.1f" % (web, side, alpha, d_chord, d_diag),
              depth=2.0 * R + d_chord * MM)
    m.fix(mount)
    m.web = web
    return m


def maxwell_count(model):
    """(members + supports, 3 * nodes) -- Maxwell's count for a pin-jointed
    frame.  Short of it, the assembly is a mechanism and whatever
    stiffness it has comes from bending its members, which for a 2 mm rod
    is almost none.  A frame model still solves; it just answers with a
    number the beam idealisation never predicted."""
    n_sup = int(model.fixed.reshape(-1, Model.DOF)[:, :3].sum())
    return len(model.members) + n_sup, 3 * model.n


def tube(length, d_out, wall, material=None, tip_mass=0.0, n_elem=24, name=None):
    """A monolithic tube, the alternative the truss has to beat."""
    material = material or mat.CARBON_WRAP
    L = length * MM
    n_elem += n_elem % 2                    # keep a node at the centre
    xs = np.linspace(0.0, L, n_elem + 1)
    nodes = [[x, 0.0, 0.0] for x in xs]
    sec = Section.tube(d_out * MM, wall * MM)
    members = [Member(i, i + 1, sec, material, tag="tube") for i in range(len(xs) - 1)]
    mid = len(xs) // 2
    masses = [PointMass(0, tip_mass * 1e-3), PointMass(len(xs) - 1, tip_mass * 1e-3)]
    m = Model(nodes, members, masses,
              faces={"left": [0], "right": [len(xs) - 1], "mount": [mid]},
              name=name or "tube %.0fx%.1f" % (d_out, wall),
              depth=d_out * MM)
    m.fix(mid)
    return m
