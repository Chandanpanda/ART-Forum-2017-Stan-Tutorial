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


def warren_truss(length, side, alpha, d_chord, d_diag, material=None,
                 tip_mass=0.0, pinned_diagonals=False, web="warren",
                 brackets=True, name=None):
    """A three-chord lattice.  Lengths in mm, mass in g, angle in deg."""
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
