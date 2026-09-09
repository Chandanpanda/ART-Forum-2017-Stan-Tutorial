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


def effective_alpha(length, side, alpha):
    """The web angle the lattice is ACTUALLY built at, degrees.

    _stations rounds to a whole number of bays and then squeezes them onto
    the length asked for, so the alpha in a candidate's name is a request,
    not a fact: at a 100 mm section, 35 and 40 degrees are the SAME truss,
    built at 38.66.  Any rule applied to the requested angle is being
    applied to a truss that does not exist -- and the winding head's bore
    is such a rule, scaling as tan(alpha), so the rounding is the
    difference between a joint the head enters and one it does not.
    """
    xs = _stations(length * MM, side * MM / np.tan(np.radians(alpha)))
    return float(np.degrees(np.arctan(side * MM / (xs[1] - xs[0]))))


WEBS = ("warren", "x", "batten")


@dataclass(frozen=True)
class Nose:
    """A hexapod camera mount at each end of the spine, all mm.

    THE PLATFORM IS PERPENDICULAR TO THE SPINE'S AXIS, parallel to the
    triangle of chord ends it is strutted from.  Two PARALLEL triangles and
    six struts is the octahedron and it is well conditioned; two
    PERPENDICULAR triangles and six struts is not, and the difference is a
    factor of three in yaw -- 0.50 of the budget against 1.43, measured.

    WHAT THE STRUTS BOND TO is what took three attempts.  Bonding the
    camera's INBOARD END FACE against the platform is the tidy answer -- a
    y-z face against a y-z plane, three long line contacts -- and it is the
    wrong one: the housing then hangs entirely outboard and its mass sits
    half a module beyond the platform, which is a moment arm under a
    lateral manoeuvre.  That is what `payload_x` measures, and it is the
    single most expensive number in the mount.

    The answer is to bond to the module's own lens housing at its mid-length
    instead, so the mass sits IN the platform's plane and on the spine's
    axis, and `payload_x` is zero.  Which face touches what is then a bonding
    question for the cell, not a structural one.

    AND `payload_x` DECIDES EVERYTHING.  It is how far the camera's centre of
    mass sits outboard of the platform's plane, and it is the whole mount
    problem.  On the chosen 85/40/3.0/1.5 truss, in fractions of the budget:

        payload_x = 12.5 (bonded end-on)      1.22
        payload_x = 0    (CoM in the plane)   0.67

    -- against 0.61 for the massless rigid placeholder the mount replaces.
    (All at d_strut 1.5.  DO NOT read the strut diameter off that sweep:
    see the null below.)

    THE PLACEHOLDER IS NOT A BOUND, IN EITHER DIRECTION.  At the brief's
    assumed 50 g head the same three rows read 6.92, 1.44 and 1.60: the
    placeholder was PESSIMISTIC there and is OPTIMISTIC here.  It makes two
    opposite mistakes at once -- it charges nothing for the mount's own mass
    and compliance, and it hangs the payload on the chord ends 66 mm off the
    axis -- and which one dominates depends on what the camera weighs.  So
    the mount is modelled, not allowed for.

    r_platform IS DERIVED, NOT CHOSEN -- but it is no longer the payload's
    own radius.  The struts were assumed to land straight on the module's
    lens housing; they cannot.  A ring of landings centred on the spine's
    axis only lies on the housing if the housing straddles that axis, and it
    does not -- it is a shallow block on the FRONT of the module.  Two
    landings of three came out on it and the third came out off the back of
    the board, in mid air.

    Neither surface the camera actually offers will hold the budget: the
    housing's own face is the only one parallel to the end triangle and is
    too thin to be a triangle (1.08 of the budget), and the board's own
    mounting holes make a proper triangle and then work through FR4 that is
    26 times softer than the strut bonded to it.  So the landings sit on a
    small rigid COLLAR bonded round the housing, at its circumradius plus a
    wall, and `platform_rigid` is true because that collar is the stiff
    thing.  What the radius is barely matters -- swept 3 to 24 mm the budget
    moves 9% -- where ROTATING the landing triangle off the chords moves it
    20%.  So it is taken as small as the collar allows.

    A NULL, WHICH IS NOT AN OPTIMUM.  Swept on strut diameter at a 115 mm
    section the signed yaw under a manoeuvre goes -167, +337, +590, +741,
    +884 microdegrees at 0.6, 0.8, 1.0, 1.2, 1.5 mm: it CHANGES SIGN,
    because two contributions oppose there -- the truss's own end rotation
    carried in through the struts, and the camera's inertia deflecting them
    -- and an unsigned sweep shows a dip that looks like an answer.  Above
    the null there is no interior optimum at all; yaw rises with every
    millimetre, so thinner is better right down to it.

    AND THE NULL MOVES WITH THE NUMBERS IT DEPENDS ON.  At a stocked 1.0 mm
    strut the same geometry sits on opposite sides of it at 50 g and at 5 g.
    When the head was assumed to be 50 g the null sat exactly at 1.0 mm -- a
    stocked diameter, a trap laid where a designer would step.  It is now
    below 0.8.  check_spine asserts the sign change rather than letting
    someone rediscover the dip as a feature.
    """
    standoff:    float = 15.0
    r_platform:  float = 24.0
    # THE PLATFORM MAY BE THE PAYLOAD'S OWN HOUSING.  A camera module
    # carries a stiff square holder round its lens; when the
    # struts bond to that, the three platform rods are not needed at all --
    # which also means nothing of the mount can end up in front of the lens.
    # The lever arm is then the enclosure's, and small; that is second order
    # once the mass is in the plane, and the model says by how much.
    platform_rigid: bool = False
    # THE END BATTEN IS WHAT SETS THE FIRST MODE, and it is not the struts.
    # The mode at 180 Hz is the truss's own section triangle breathing at
    # mid-span -- the nose's nodes move at 0.17 of the amplitude, so the nose
    # is loading that mode, not causing it.  What the nose changed is how
    # well the END TRIANGLE is closed: it replaced the fictitious bracket, a
    # 20 x 3 mm section, with a rod, and 20 x 3 has 180 times the second
    # moment of a 1.5 mm rod.  So this diameter is swept separately.
    d_batten:    float = 3.0
    payload_x:   float = 0.0       # the camera's CoM, outboard of the platform
    payload_box: tuple = (25.0, 9.0, 23.862)  # mm, along x / y / z.  A
                                              # FALLBACK: the caller passes
                                              # the real part (mount.nose_spec)
    d_strut:     float = 1.5
    d_platform:  float = 1.5
    # WHERE THE SIX STRUTS ACTUALLY LAND, if not on a circle.  Three
    # (dx, y, z) offsets in mm from the platform's centre, so a landing
    # pattern that is NOT a ring around the spine -- a camera's own
    # mounting holes, say, which lie in the board's plane and therefore
    # contain the spine's axis rather than crossing it -- can be modelled
    # and priced instead of assumed.  None keeps the ring.
    landings:    tuple = None
    # HOW STIFF THE PAYLOAD IS BETWEEN A LANDING AND ITS OWN MASS, N/mm.
    # None means rigid, which is what a stiff housing is.  It is NOT what a
    # circuit board is: a strut bonded through one of the module's mounting
    # holes works through FR4 over 21 mm, and that is 26 times softer than
    # the strut itself.  Modelling it rigid is how a mount through the holes
    # comes back looking better than it is.
    link_k:      float = None
    # WHICH CHORD END FEEDS WHICH LANDING, as (chord, landing) pairs.  The
    # octahedron's own pattern -- each chord to two consecutive landings --
    # only exists when there are three of them.  The mount has four
    # crossings and six struts, and which crossing a strut runs to is
    # solved in truss.mount, not guessed here.
    strut_pairs: tuple = None
    # THE MOUNT'S OWN MACHINED MASS, g, carried at the payload's node.  The
    # collar is not free: at the size the struts need it is the same order
    # as the camera, and a mount that is not charged for its own part is
    # the placeholder this class replaced.
    plate_g:     float = 0.0
    # THE PLATFORM'S OWN SECTION, mm^2 and mm^4, when the ring is a real
    # part rather than a placeholder.  `platform_rigid` says the four
    # landings are tied by something infinitely stiff; the mount's landings
    # are tied by a folded aluminium band and four carbon rods, and its
    # OUT-OF-PLANE bending is the whole compliance between a strut and the
    # camera.  Given these, the ring carries it.
    #
    # THIS REPLACED AN AXIAL `link_k` FROM EACH LANDING TO THE CAMERA, and
    # the replacement is not cosmetic: those four links all pass through
    # the camera's own centre, so they restrain no TWIST about the optical
    # axis at all.  The model answered with a 16 Hz mode that is a
    # mechanism of the idealisation, not of the part -- the aperture bond
    # holds that twist over four walls of housing.  A ring with a real
    # section has no such freedom.
    plat_A:      float = None
    plat_I:      float = None

    @staticmethod
    def around(box=None, standoff=15.0, d_strut=1.5, d_batten=3.0, clearance=1.5,
               rigid=False, r_platform=None, landings=None, link_k=None,
               strut_pairs=None, plate_g=0.0, plat_A=None, plat_I=None):
        """A nose whose platform triangle surrounds `box` (mm, x/y/z) at its
        mid-length, so the camera's centre of mass lies in the platform's
        plane.  The triangle's inscribed circle must clear the housing's
        half-diagonal in the y-z section, and a triangle's inscribed circle
        is half its circumradius.

        `r_platform` overrides that when the struts land on something the
        payload already has -- the module's own lens housing -- rather
        than on three rods laid round it.  The caller passes the radius of
        whatever is being bonded to (with `rigid`, since it is then not a
        rod triangle); THIS package does not know what a Camera Module is.
        """
        # a fallback only -- sim/truss hands the real part over in
        # mount.nose_spec, and this package knows nothing about cameras
        box = (25.0, 9.0, 23.862) if box is None else box
        half_diag = 0.5 * (box[1] ** 2 + box[2] ** 2) ** 0.5
        r = 2.0 * (half_diag + clearance) if r_platform is None else float(r_platform)
        return Nose(standoff=standoff, r_platform=r,
                    platform_rigid=rigid, landings=landings, link_k=link_k,
                    strut_pairs=strut_pairs, plate_g=plate_g,
                    plat_A=plat_A, plat_I=plat_I,
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
            xp = x0 + sign * nose.standoff * MM
            plat = []
            for k in range(3 if nose.landings is None else len(nose.landings)):
                plat.append(len(nodes))
                if nose.landings is None:
                    nodes.append([xp, rp * np.cos(phi[k]), rp * np.sin(phi[k])])
                else:
                    dx, dy, dz = nose.landings[k]
                    nodes.append([xp + sign * dx * MM, dy * MM, dz * MM])
            npl = len(plat)
            if nose.plat_A is not None and nose.plat_I is not None:
                # The band's own area and its WEAK second moment, in both
                # bending axes.  A rectangle would put the strong axis
                # somewhere -- 4.5 x 1.5 is nine times stiffer edge-on than
                # flat -- and which way a ring member ends up rolled is not
                # something this model decides.  So it takes the soft
                # reading in both, which is the reading that cannot flatter.
                A = nose.plat_A * MM * MM
                I = nose.plat_I * MM ** 4
                sec_ring = Section("collar band", A, I, I, 2.0 * I, 2.0 * I)
                mat_ring = mat.BRACKET
            elif nose.platform_rigid:
                sec_ring, mat_ring = Section.rect(0.010, 0.003), mat.BRACKET
            else:
                sec_ring, mat_ring = sec_p, material
            for k in range(npl):
                members.append(Member(plat[k], plat[(k + 1) % npl],
                                      sec_ring, mat_ring, tag="platform"))
            pairs = (nose.strut_pairs if nose.strut_pairs is not None
                     else tuple((k, (k + j) % npl) for k in range(npl) for j in (0, 1)))
            for bk, pk in pairs:
                members.append(Member(base[bk], plat[pk], sec_s, material, tag="strut"))
            # THE CAMERA'S MASS WHERE IT ACTUALLY IS: a node at the housing's
            # centre of mass, half a housing outboard of the platform, tied to
            # the platform by the housing itself.  The housing is far stiffer
            # than these struts -- measured, even in plastic -- so it is three
            # stiff massless members; but
            # it is NOT on the platform, and that offset is a moment arm the
            # first model did not charge for.
            com = len(nodes)
            nodes.append([x0 + sign * (nose.standoff + nose.payload_x) * MM, 0.0, 0.0])
            # EVERY landing, not the first three.  With a three-landing ring
            # those were the same set; the mount has four crossings, and the
            # fourth was reaching the camera only through the platform ring.
            for k in range(npl):
                if nose.link_k is None:
                    sec_h = Section.rect(0.010, 0.003)
                else:
                    L = float(np.linalg.norm(np.array(nodes[plat[k]])
                                             - np.array(nodes[com])))
                    # A = k L / E, so the member reproduces the measured
                    # stiffness over the length it actually spans
                    sec_h = Section.rod(2.0 * np.sqrt(
                        (nose.link_k * 1000.0) * L / mat.BRACKET.E / np.pi))
                members.append(Member(plat[k], com, sec_h,
                                      mat.BRACKET, tag="housing"))
            payload_nodes.append(com)
            # THE FACE IS THE CAMERA, NOT THE MOUNT.  This read `plat` -- the
            # three platform landings -- so every pose in metrics was the
            # MOUNT's pose, and any compliance between the mount and the
            # camera it carries was invisible.  With a stiff housing the two
            # are the same to three figures; with a strut bonded through a
            # circuit board they are not, and that is exactly the case the
            # model existed to price.  face_pose reads a single node's own
            # rotation, which is what a rigid body's six freedoms are.
            if base is left:
                left = [com]
            else:
                right = [com]
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
        masses = [PointMass.box(n, (tip_mass + nose.plate_g) * 1e-3,
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
