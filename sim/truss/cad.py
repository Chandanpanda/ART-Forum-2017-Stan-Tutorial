"""Printable geometry for the ring drive, derived from `spec.Ring`.

WHY THIS IS CODE AND NOT A DRAWING.  Every dimension that decides whether
the drive works is already solved -- `Ring.mesh()` fixes the module, both
tooth counts and the centre distance; `Ring.race_mouth()` fixes the arc the
rail may occupy; `Ring.pinion_az()` fixes where the three shafts go.  A CAD
file would be a transcription of those, and a transcription is a constant:
it can only be wrong silently, and it goes stale the moment the solve moves.
So the meshes are generated, `check_cad` re-derives them, and the STLs in
`sim/cad/` are build output.

WHAT IS *NOT* DERIVED, and is therefore a fabrication decision this module
owns.  These are fits and hardware facts, not answers to computations, and
each is named so it can be measured on the first print rather than argued:

  * `FIT_*`         -- printed-part allowances.  A first article measures
                       these; they are the only numbers here expected to
                       change, and they change in one place.
  * bearing, belt and screw dimensions -- MR63 is 3 x 6 x 2.5 because that
                       is what an MR63 is.  `Ring.BEARING_OD` already says 6.
  * the pinion's stack height, hub radius and the plate's standoff length
                       -- these follow from the axial layout: the gear band
                       must sit in the ring's own 4 mm, the belt must clear
                       the rail's outer face, and the plate must clear the
                       belt.  `axial_layout()` solves that chain and
                       `check_cad` asserts the clearances it claims.

THE ONE AMBIGUITY IN THE SPEC, recorded rather than hidden.  `Ring.GROOVE_D
= 2.0 # radial depth` and `GROOVE_W = 2.0 # axial width` describe a channel
whose section is 2 x 2 at mean radius 15, but a channel cut *radially* at
r 15 has no surface to be cut from -- the bore is at 10 and the rim is
teeth.  The groove is therefore cut AXIALLY, 2.0 deep into one face, and is
2.0 wide radially (r 14 to 16).  `groove_capacity` uses only the product, so
the thread budget is identical either way; what changes is which face the
thread leaves by, and that is now a fact of the part rather than a reading.
"""
from math import pi, cos, sin, tan, acos, asin, atan2, radians, degrees, hypot
import struct

import numpy as np

from .spec import Ring, ring_r_in, ring_r_out, rail_r_out

# ----------------------------------------------------------------- fits
# Printed-part allowances.  A FIRST ARTICLE MEASURES THESE.  They are here,
# named, in one place, precisely so that "the bearing was loose" is a
# one-line change and not a hunt.
FIT_BEARING = 0.10      # mm on diameter, MR63 seat: slip fit, retained by
                        # a shoulder and a drop of retainer
FIT_SHAFT   = 0.20      # mm on diameter, 3 mm shaft through a printed bore
FIT_SCREW   = 0.40      # mm on diameter, M3 free-running hole
FIT_PRESS   = -0.05     # mm on diameter, 3 mm shaft pressed into the plate
FIT_ROD     = 0.15      # mm on diameter, a CF rod seated in a V

# ------------------------------------------------------- hardware facts
BEARING_OD  = Ring.BEARING_OD       # 6.0, an MR63
BEARING_ID  = 3.0
BEARING_W   = Ring.BEARING_W        # 2.5
BEARING_RACE_ID = 4.8   # mm, the outer race's own bore -- a shoulder must
                        # bear here and NOT on the rotating inner race
M3          = 3.0
BELT_PITCH  = 2.0       # GT2
BELT_W      = 6.0
BELT_PLO    = 0.254     # GT2 pitch-line offset: pulley OD = PD - 2*PLO
BELT_GROOVE_R = 0.747   # mm.  The GT2 groove taken as a circular arc: a
                        # chord of 1.494 at the OD with a 0.762 sagitta is
                        # an arc of (w^2/4 + h^2)/2h = 0.747, which is the
                        # half-width the Gates section quotes.  A printed
                        # pulley at 2 mm pitch is a tooth's shape, not a
                        # tenth of a tooth's; this is the shape.
NEMA17_SQ   = 42.3
NEMA17_BC   = 31.0      # bolt spacing, square
NEMA17_BOSS = 22.0      # locating boss diameter


# ============================================================ mesh basics
class Mesh:
    """A triangle soup that knows whether it is a solid."""

    def __init__(self, name="part"):
        self.name = name
        self.tris = []

    def tri(self, a, b, c):
        self.tris.append((np.asarray(a, float), np.asarray(b, float),
                          np.asarray(c, float)))

    def quad(self, a, b, c, d):
        """a-b-c-d round the face.  Degenerate halves are dropped, which is
        how a collapsed edge (a cone's apex) stays manifold."""
        if not _same(a, b) and not _same(b, c) and not _same(c, a):
            self.tri(a, b, c)
        if not _same(a, c) and not _same(c, d) and not _same(d, a):
            self.tri(a, c, d)

    def extend(self, other):
        self.tris.extend(other.tris)
        return self

    # ------------------------------------------------------------ checks
    def volume(self):
        """Signed volume, mm^3.  Positive iff the normals face out."""
        return sum(float(np.dot(a, np.cross(b, c))) for a, b, c in self.tris) / 6.0

    def flip(self):
        self.tris = [(a, c, b) for a, b, c in self.tris]
        return self

    def orient(self):
        if self.volume() < 0:
            self.flip()
        return self

    def edges(self):
        """Every directed edge, keyed on quantised vertices."""
        out = {}
        for a, b, c in self.tris:
            k = (_key(a), _key(b), _key(c))
            for i in range(3):
                e = (k[i], k[(i + 1) % 3])
                out[e] = out.get(e, 0) + 1
        return out

    def open_edges(self):
        """Directed edges with no opposite twin -- a hole, or a flipped
        triangle.  A closed, consistently wound solid has none."""
        e = self.edges()
        return [k for k, n in e.items() if e.get((k[1], k[0]), 0) != n]

    def contains(self, p, d=(0.5773, 0.5774, 0.5775)):
        """Is `p` inside the solid?  Ray cast, crossings counted.

        The only test that reads the mesh the way a slicer will.  Edge
        counting says a surface closes; this says which side of it a point
        is on, which is what a groove IS.
        """
        p = np.asarray(p, float)
        d = np.asarray(d, float)
        d = d / np.linalg.norm(d)
        n = 0
        for a, b, c in self.tris:
            e1, e2 = b - a, c - a
            h = np.cross(d, e2)
            det = float(e1 @ h)
            if abs(det) < 1e-12:
                continue
            f = 1.0 / det
            sv = p - a
            u = f * float(sv @ h)
            if u < 0.0 or u > 1.0:
                continue
            q = np.cross(sv, e1)
            v = f * float(d @ q)
            if v < 0.0 or u + v > 1.0:
                continue
            if f * float(e2 @ q) > 1e-9:
                n += 1
        return bool(n % 2)

    def bbox(self):
        v = np.array([p for t in self.tris for p in t])
        return v.min(axis=0), v.max(axis=0)

    # ------------------------------------------------------------ output
    def stl(self, path):
        """Binary STL, mm.  Robu's uploader and every slicer read this."""
        with open(path, "wb") as f:
            f.write(b"\0" * 80)
            f.write(struct.pack("<I", len(self.tris)))
            for a, b, c in self.tris:
                n = np.cross(b - a, c - a)
                ln = np.linalg.norm(n)
                n = n / ln if ln > 1e-12 else np.zeros(3)
                f.write(struct.pack("<12fH", *n, *a, *b, *c, 0))
        return len(self.tris)


def _key(p, q=1e-6):
    return (round(float(p[0]) / q), round(float(p[1]) / q), round(float(p[2]) / q))


def _same(a, b, tol=1e-9):
    return float(np.dot(np.subtract(a, b), np.subtract(a, b))) < tol * tol


# ------------------------------------------------------- triangulation
def ear_clip(poly):
    """Indices of a triangulation of a simple polygon, CCW or CW.

    THE STRICT TEST IS NOT ENOUGH, and the way it failed is worth keeping.
    A strict ear needs cross > 0, and these profiles are full of COLLINEAR
    vertices that are not adjacent -- the rail's section has four points on
    r 18.2, the V-block's has four on its shoulder line.  Clip down to
    those and every remaining triple is collinear, no strict ear exists,
    and the clipper stops two triangles short: the caps came out with a
    hole in them and the part read as 8 open edges.  So a pass that finds
    no strict ear takes the best FLAT one instead.  A zero-area facet
    contributes nothing to the volume and its edges still pair, which is
    what closedness is measured on.
    """
    n = len(poly)
    idx = list(range(n))
    if _area(poly) < 0:
        idx.reverse()
    out = []
    for _ in range(n + 2):
        while len(idx) > 3:
            for strict in (True, False):
                for i in range(len(idx)):
                    a, b, c = idx[i - 1], idx[i], idx[(i + 1) % len(idx)]
                    x = _cross(poly[a], poly[b], poly[c])
                    if x < (1e-12 if strict else -1e-9):
                        continue
                    if any(_inside(poly[a], poly[b], poly[c], poly[k])
                           for k in idx if k not in (a, b, c)):
                        continue
                    out.append((a, b, c))
                    idx.pop(i)
                    break
                else:
                    continue
                break
            else:
                break
        break
    if len(idx) == 3:
        out.append(tuple(idx))
    return out


def _area(p):
    return sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1]
               for i in range(len(p))) / 2.0


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _inside(a, b, c, p):
    d1, d2, d3 = _cross(a, b, p), _cross(b, c, p), _cross(c, a, p)
    return (d1 > 0 and d2 > 0 and d3 > 0) or (d1 < 0 and d2 < 0 and d3 < 0)


# ------------------------------------------------------------- revolve
def revolve(profile, thetas, wrap, name="part"):
    """Sweep a closed (r, z) profile about the z axis.

    `profile` is a closed loop of (r, z); an `r` may be a callable of theta,
    which is what lets one solid carry two different tooth patterns in two
    different z bands.  `thetas` are radians, ascending.  `wrap` closes the
    sweep on itself; otherwise the two ends are capped, which is what makes
    a C rather than a ring.
    """
    P, T = len(profile), len(thetas)
    V = np.empty((T, P, 3))
    for i, t in enumerate(thetas):
        ct, st = cos(t), sin(t)
        for j, (rs, z) in enumerate(profile):
            r = rs(t) if callable(rs) else float(rs)
            V[i, j] = (r * ct, r * st, z)
    m = Mesh(name)
    last = T if wrap else T - 1
    for i in range(last):
        i2 = (i + 1) % T
        for j in range(P):
            j2 = (j + 1) % P
            m.quad(V[i, j], V[i2, j], V[i2, j2], V[i, j2])
    if not wrap:
        for i, sgn in ((0, -1), (T - 1, 1)):
            flat = [(float(rs(thetas[i])) if callable(rs) else float(rs), z)
                    for rs, z in profile]
            for a, b, c in ear_clip(flat):
                if sgn < 0:
                    m.tri(V[i, a], V[i, b], V[i, c])
                else:
                    m.tri(V[i, a], V[i, c], V[i, b])
    return m.orient()


# ========================================================= tooth profiles
def involute_r(n, m, alpha_deg, r_root, r_tip, backlash=0.0):
    """psi(r) for one flank, and the tooth's angular half-widths.

    Same textbook involute as `drive.involute_tooth`, but as a RADIUS
    function of angle rather than a polygon, because a swept solid wants
    r(theta) and a MuJoCo prism wants (y, z).
    """
    a = radians(alpha_deg)
    r_p = n * m / 2.0
    r_b = r_p * cos(a)
    inv = lambda t: tan(t) - t
    s_p = pi * m / 2.0 - backlash
    half_b = s_p / (2.0 * r_p) + inv(a)         # psi at the base circle
    psi = lambda r: half_b - inv(acos(min(1.0, r_b / r)))
    return psi, half_b, r_b


def gear_r(n, m, alpha_deg, r_root, r_tip, phase_deg=0.0, keep=None,
           backlash=0.0):
    """EXACT r(theta) on an involute gear's outline.

    psi(r) is the flank's angle from the tooth centre and it decreases with
    r, so the outline at an angular offset d from a tooth centre is the r
    where psi(r) = |d| -- one bisection, not a lookup.  The lookup was the
    first version and it was wrong in a way worth recording: the pinion's
    sweep is sampled at the union of its GEAR angles and its BELT angles,
    and a nearest-neighbour r() snapped every belt-only angle onto the
    nearest gear sample, putting a step in the flank at each one.  The
    profile looked right in a list of numbers and was a saw in the part.
    """
    psi, half_b, r_b = involute_r(n, m, alpha_deg, r_root, r_tip, backlash)
    r_lo = max(r_root, r_b)
    half_root = half_b if r_root < r_b - 1e-12 else psi(r_root)
    psi_tip = psi(r_tip)
    pitch = 2.0 * pi / n
    ph = radians(phase_deg)

    def r(t):
        k = round((t - ph) / pitch)
        d = abs(t - ph - k * pitch)
        if keep is not None and not keep(degrees(ph + k * pitch) % 360.0):
            return r_root
        if d >= half_root:
            return r_root
        if d <= psi_tip:
            return r_tip
        lo, hi = r_lo, r_tip
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if psi(mid) > d:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    r.half_root = half_root
    r.psi_tip = psi_tip
    return r


def gear_arc(n, m, alpha_deg, r_root, r_tip, phase_deg, keep, t0, t1,
             pts=9, arc_pts=3, backlash=0.0):
    """[(theta, r)] round an involute gear, ascending, over [t0, t1] degrees.

    `keep(centre_deg)` says which teeth exist -- absent ones leave the root
    circle, which is what the ring's 60-degree gap is.  `phase_deg` offsets
    the whole pattern.
    """
    psi, half_b, r_b = involute_r(n, m, alpha_deg, r_root, r_tip, backlash)
    r_lo = max(r_root, r_b)
    half_root = half_b if r_root < r_b - 1e-12 else psi(r_root)
    flank = [r_lo + (r_tip - r_lo) * i / (pts - 1.0) for i in range(pts)]
    out = [(radians(t0), r_root)]
    pitch = 360.0 / n
    for j in range(-1, 2 * n + 1):
        c = phase_deg + j * pitch
        if not (t0 - pitch <= c <= t1 + pitch) or not keep(c):
            continue
        lo, hi = c - degrees(half_root), c + degrees(half_root)
        if lo < t0 - 1e-9 or hi > t1 + 1e-9:
            continue                         # a part tooth is not a tooth
        seg = []
        if r_root < r_b - 1e-12:             # radial run-in below the base
            seg.append((lo, r_root))
        for r in flank:
            seg.append((c - degrees(psi(r)), r))
        pt = degrees(psi(r_tip))
        for i in range(1, arc_pts):
            seg.append((c - pt + 2 * pt * i / arc_pts, r_tip))
        for r in reversed(flank):
            seg.append((c + degrees(psi(r)), r))
        if r_root < r_b - 1e-12:
            seg.append((hi, r_root))
        # root arc from wherever we were to this tooth's start
        prev = degrees(out[-1][0])
        for i in range(1, arc_pts):
            out.append((radians(prev + (lo - prev) * i / arc_pts), r_root))
        out.extend((radians(a), r) for a, r in seg)
    prev = degrees(out[-1][0])
    for i in range(1, arc_pts + 1):
        out.append((radians(prev + (t1 - prev) * i / arc_pts), r_root))
    # strictly ascending
    keepl = [out[0]]
    for t, r in out[1:]:
        if t > keepl[-1][0] + 1e-12:
            keepl.append((t, r))
    return keepl


def gt2_r(n_teeth):
    """r(theta) for a printed GT2 pulley of `n_teeth`.

    The groove is the circular arc described at BELT_GROOVE_R, swung on the
    OD circle: a ray at theta leaves the groove where it cuts that circle,
    which is R cos t - sqrt(rho^2 - R^2 sin^2 t), and lands on the OD
    everywhere else.
    """
    R = n_teeth * BELT_PITCH / (2.0 * pi) - BELT_PLO
    rho = BELT_GROOVE_R
    step = 2.0 * pi / n_teeth
    half = asin(min(1.0, rho / R))

    def r(t):
        d = (t + step / 2.0) % step - step / 2.0
        if abs(d) >= half:
            return R
        s = R * sin(d)
        return R * cos(d) - (rho * rho - s * s) ** 0.5
    r.R = R
    r.half = half
    r.pitch_r = n_teeth * BELT_PITCH / (2.0 * pi)
    return r


def gt2_thetas(n_teeth, arc_pts=7):
    """Angles that resolve every groove of a GT2 pulley."""
    step = 2.0 * pi / n_teeth
    half = asin(min(1.0, BELT_GROOVE_R /
                    (n_teeth * BELT_PITCH / (2.0 * pi) - BELT_PLO)))
    out = []
    for j in range(n_teeth):
        c = j * step
        out.append(c - half)
        for i in range(1, arc_pts):
            out.append(c - half + 2 * half * i / arc_pts)
        out.append(c + half)
        out.append(c + step / 2.0)
    return sorted(t % (2 * pi) for t in out)


# ============================================================== extrusion
def extrude(profile, length, name="part", frame=None):
    """Extrude a closed 2-D loop (u, v) along w by `length`.

    `frame` maps local (u, v, w) into the part's own output frame -- a
    3x3 whose columns are the local axes, so one profile serves a V-block
    on the chord and the same V-block on a diagonal.
    """
    P = len(profile)
    A = np.eye(3) if frame is None else np.asarray(frame, float)
    V = np.empty((2, P, 3))
    for i, w in enumerate((0.0, float(length))):
        for j, (u, v) in enumerate(profile):
            V[i, j] = A @ np.array([u, v, w], float)
    m = Mesh(name)
    for j in range(P):
        j2 = (j + 1) % P
        m.quad(V[0, j], V[1, j], V[1, j2], V[0, j2])
    for a, b, c in ear_clip(list(profile)):
        m.tri(V[0, a], V[0, b], V[0, c])
        m.tri(V[1, a], V[1, c], V[1, b])
    return m.orient()


# ============================================================ the layout
def axial_layout(standoff_l=12.0, pulley_clear=1.5):
    """The axial chain, along the ring's own axis, in mm.

    Everything downstream of the ring's 4 mm hangs off three facts: the
    gear band must sit IN that 4 mm, the belt must clear the rail's outer
    face, and the plate must stand off the rail by a bought standoff.  So
    the pinion's hub length is what closes the chain, not a chosen number.
    """
    w2 = Ring.W / 2.0
    rail_out = w2 + Ring.RACE_CLEAR + Ring.RACE_T       # 5.05
    plate_x = rail_out + standoff_l                     # the plate's face
    pocket = BEARING_W + 0.10                           # a bearing plus a film
    belt = 0.5 + BELT_W + 0.4 + 0.5                     # flange, belt, flange
    hub = plate_x - w2 - pocket - belt
    lay = dict(rail_in=w2 + Ring.RACE_CLEAR, rail_out=rail_out,
               plate_x=plate_x, pocket=pocket, belt_band=belt, hub=hub,
               standoff_l=standoff_l,
               # pinion-local z, 0 at the far end, +z toward the plate
               z_pocket_a=pocket, z_gear=pocket + Ring.W,
               z_hub=pocket + Ring.W + hub,
               z_belt=pocket + Ring.W + hub + belt,
               z_top=pocket + Ring.W + hub + belt + pocket)
    lay["x0"] = -w2 - pocket        # where pinion z = 0 sits on the x axis
    lay["belt_x0"] = lay["x0"] + lay["z_hub"]
    lay["clear_belt_rail"] = lay["belt_x0"] - rail_out
    lay["pulley_clear"] = pulley_clear
    return lay


def pinion_bands(lay=None):
    """[(x0, x1, outer_r)] of the pinion's stack in the ASSEMBLY frame."""
    M = Ring.mesh()
    L = axial_layout() if lay is None else lay
    hub = M.r_pinion - 1.25 * M.m
    tip = M.r_pinion + M.m
    fl = gt2_r(20).R + 1.1
    x = L["x0"]
    z = (L["z_pocket_a"], L["z_gear"], L["z_hub"], L["z_belt"], L["z_top"])
    r = (hub, tip, hub, fl, hub)
    out, prev = [], 0.0
    for zi, ri in zip(z, r):
        out.append((x + prev, x + zi, ri))
        prev = zi
    return out


def rail_sections(lay=None):
    """[(|x|0, |x|1, r0, r1)] of the rail's own section."""
    M = Ring.mesh()
    L = axial_layout() if lay is None else lay
    return [(0.0, L["rail_in"], ring_r_out() + Ring.RACE_CLEAR, rail_r_out()),
            (L["rail_in"], L["rail_out"], M.r_pitch - 1.25 * M.m, SKIRT_R)]


def pinion_window_deg(clear=0.6):
    """Half-angle of the window a pinion needs in the rail, degrees.

    WHAT THE FIRST VERSION GOT WRONG, because it is the kind of thing only a
    check finds.  It reasoned: the gear's tip reaches in to ring-radius 18.4,
    inside the back wall's 20.05, so the teeth pass under the wall and only
    the HUB needs a window -- 14.93 degrees.  But a pinion is a DISC at
    r 26.4, and the further round it you look the further OUT in ring-radius
    its rim sits: at ring-radius 23.05, the back wall's own outer edge, the
    gear reaches 16.94 degrees off the pinion's azimuth.  `check_cad` sampled
    the printed mesh against the printed rail and found 1.47 mm of the gear
    inside the flange.

    So the window is a maximum over every (pinion band, rail section) pair
    whose axial extents overlap, and it is the geometry that decides which
    feature binds, not a sentence about which one looks like it should.
    """
    M = Ring.mesh()
    c = M.centre
    worst = 0.0
    for bx0, bx1, rho in pinion_bands():
        a0, a1 = sorted((abs(bx0), abs(bx1))) if bx0 * bx1 >= 0 else (0.0, max(abs(bx0), abs(bx1)))
        rho = rho + clear
        for sx0, sx1, r0, r1 in rail_sections():
            if min(a1, sx1) <= max(a0, sx0) + 1e-12:
                continue
            if rho >= c:                          # a disc over the axis
                return 180.0
            r_peak = (c * c - rho * rho) ** 0.5
            for r in (min(max(r_peak, r0), r1),):
                if not (c - rho <= r <= c + rho):
                    continue
                k = (c * c + r * r - rho * rho) / (2.0 * c * r)
                if abs(k) <= 1.0:
                    worst = max(worst, degrees(acos(k)))
    return worst


def race_arcs(win=None):
    """[(t0, t1)] degrees the rail may occupy: not the mouth, not a pinion."""
    win = pinion_window_deg() if win is None else win
    cuts = [(-Ring.race_mouth() / 2.0, Ring.race_mouth() / 2.0)]
    cuts += [(a - win, a + win) for a in Ring.pinion_az()]
    cuts = sorted(((a % 360.0), (b % 360.0)) for a, b in cuts)
    out = []
    for i, (a, b) in enumerate(cuts):
        nxt = cuts[(i + 1) % len(cuts)][0]
        t0, t1 = b, nxt + (360.0 if nxt < b else 0.0)
        if t1 - t0 > 1.0:
            out.append((t0, t1))
    return out


# ================================================================= parts
FIT_GEAR = 0.10         # mm of backlash across the PAIR, half on each.  A
                        # cut module-0.8 pair runs about 0.05; a printed one
                        # will not hold that, and a first article says what
                        # it does hold.


def ring(name="ring"):
    """The C: 300 degrees of a 48-tooth rim, its thread groove, its bore.

    The tooth pattern is offset HALF A PITCH so that the gap's two end
    faces fall at space centres -- otherwise 60 degrees lands on two tooth
    centres and the part ships with a half tooth at each end.  Eight whole
    teeth are then absent, which is what `Ring.teeth_in_gap()` says.
    """
    M = Ring.mesh()
    r_root, r_tip = M.r_pitch - 1.25 * M.m, ring_r_out()
    half = Ring.GAP / 2.0
    th = gear_arc(M.n_ring, M.m, Ring.PRESSURE_ANGLE, r_root, r_tip,
                  phase_deg=360.0 / M.n_ring / 2.0, keep=lambda c: True,
                  t0=half, t1=360.0 - half, backlash=FIT_GEAR / 2.0)
    th = [(t, r) for t, r in th]
    hw = degrees(gear_r(M.n_ring, M.m, Ring.PRESSURE_ANGLE, r_root, r_tip,
                        backlash=FIT_GEAR / 2.0).half_root)
    keep = lambda c: (half + hw) <= c <= (360.0 - half - hw)
    r_of = gear_r(M.n_ring, M.m, Ring.PRESSURE_ANGLE, r_root, r_tip,
                  phase_deg=360.0 / M.n_ring / 2.0, keep=keep,
                  backlash=FIT_GEAR / 2.0)
    g0, g1 = Ring.GROOVE_R - Ring.GROOVE_W / 2.0, Ring.GROOVE_R + Ring.GROOVE_W / 2.0
    W, r_in, ch = Ring.W, ring_r_in(), 0.6
    # THE GROOVE OPENS ON THE WORK SIDE, not the belt side.  The part's own
    # +z becomes the assembly's +x, which is where the plate, the three
    # pulleys and the belt are; a thread paying off that face runs past all
    # of them on its way to the joint.  Cut at z = 0 it opens away from the
    # drive, and the only thing on that side is the work.
    prof = [(r_in, ch), (r_in + ch, 0.0), (g0, 0.0), (g0, Ring.GROOVE_D),
            (g1, Ring.GROOVE_D), (g1, 0.0), (r_of, 0.0), (r_of, W),
            (r_in, W)]
    m = revolve(prof, [t for t, _ in th], wrap=False, name=name)
    m.info = dict(groove_open_at=0.0, groove_floor=Ring.GROOVE_D,
                  teeth=sum(1 for i in range(1, len(th))
                            if th[i][1] >= r_tip - 1e-9 and th[i - 1][1] < r_tip - 1e-9),
                  arc=360.0 - Ring.GAP, groove_mm3=(g1 - g0) * Ring.GROOVE_D *
                  2 * pi * Ring.GROOVE_R * (360.0 - Ring.GAP) / 360.0)
    return m


def pinion(name="pinion", lay=None):
    """18 teeth, two MR63 seats, and the GT2 pulley that phases it.

    THE PULLEY IS PRINTED AND INTEGRAL, and that is a decision with a
    reason.  A bought pulley would have to clamp on a boss, and the only
    boss a 3 mm shaft and a 6 mm bearing leave room for is thinner than the
    grub screw that would tighten on it.  Printing it also removes a
    quantisation: a bought pulley sets the gear's phase in whole belt teeth
    (18 degrees at 20T), and `check_drive` jams on a quarter of a pitch (5
    degrees), so phase has to be a setting and not a step.  Here it is set
    by where the belt is clamped and by the tensioner, both continuous.
    """
    M = Ring.mesh()
    L = axial_layout() if lay is None else lay
    r_root = M.r_pinion - 1.25 * M.m
    gear = gear_arc(M.n_pinion, M.m, Ring.PRESSURE_ANGLE, r_root,
                    M.r_pinion + M.m, 0.0, lambda c: True, 0.0, 360.0,
                    backlash=FIT_GEAR / 2.0)
    g_of = gear_r(M.n_pinion, M.m, Ring.PRESSURE_ANGLE, r_root,
                  M.r_pinion + M.m, backlash=FIT_GEAR / 2.0)
    belt = gt2_r(20)
    thetas = sorted(set([t for t, _ in gear] + list(gt2_thetas(20))))
    r_pk = (BEARING_OD + FIT_BEARING) / 2.0
    r_sh = BEARING_RACE_ID / 2.0 + 0.10
    r_hub, r_fl = r_root, belt.R + 1.1
    z1, z2, z3, z4, z5 = (L["z_pocket_a"], L["z_gear"], L["z_hub"],
                          L["z_belt"], L["z_top"])
    prof = [(r_pk, 0.0), (r_hub, 0.0), (r_hub, z1),
            (g_of, z1), (g_of, z2),
            (r_hub, z2), (r_hub, z3),
            (r_fl, z3), (r_fl, z3 + 0.5),
            (belt, z3 + 0.5), (belt, z4 - 0.5),
            (r_fl, z4 - 0.5), (r_fl, z4),
            (r_hub, z4), (r_hub, z5), (r_pk, z5),
            (r_pk, z4), (r_sh, z4), (r_sh, z1), (r_pk, z1)]
    m = revolve(prof, thetas, wrap=True, name=name)
    m.info = dict(pocket_r=r_pk, shoulder_r=r_sh, hub_r=r_hub,
                  pulley_od=2 * belt.R, height=z5)
    return m


def race(t0, t1, name="race"):
    """One arc of the C-channel: back wall at the tips, flanges to the root,
    and a skirt for the standoffs to be bonded to."""
    r_root = Ring.mesh().r_pitch - 1.25 * Ring.mesh().m
    r0, r1 = ring_r_out() + Ring.RACE_CLEAR, rail_r_out()
    L = axial_layout()
    skirt = SKIRT_R
    h = 2.0 * L["rail_out"]
    t_fl = Ring.RACE_T
    prof = [(r_root, 0.0), (skirt, 0.0), (skirt, t_fl), (r1, t_fl),
            (r1, h - t_fl), (skirt, h - t_fl), (skirt, h), (r_root, h),
            (r_root, h - t_fl), (r0, h - t_fl), (r0, t_fl), (r_root, t_fl)]
    n = max(12, int(round((t1 - t0) / 0.75)))
    th = [radians(t0 + (t1 - t0) * i / n) for i in range(n + 1)]
    m = revolve(prof, th, wrap=False, name=name)
    m.info = dict(arc=t1 - t0, skirt=skirt, channel=(r_root, r0, r1),
                  width=h, flange=t_fl)
    return m


def vblock(d_rod, length, name="vblock", h_axis=45.0, foot_w=26.0,
           col_w=14.0, foot_h=6.0):
    """A V-block: a rod sits in a 90-degree V, half proud, on a pedestal.

    Half proud on purpose -- the rod is bonded in, and a V you can see into
    is a V you can see the bond in.  A 90-degree V puts the rod's centre
    r*sqrt(2) above the apex, which is the only arithmetic here.
    """
    rr = (d_rod + FIT_ROD) / 2.0
    v = rr * 2.0 ** 0.5
    prof = [(-foot_w / 2, 0.0), (foot_w / 2, 0.0), (foot_w / 2, foot_h),
            (col_w / 2, foot_h), (col_w / 2, h_axis), (v, h_axis),
            (0.0, h_axis - v), (-v, h_axis), (-col_w / 2, h_axis),
            (-col_w / 2, foot_h), (-foot_w / 2, foot_h)]
    m = extrude(prof, length, name)
    m.info = dict(d_rod=d_rod, v_half=v, h_axis=h_axis, length=length)
    return m


# ==================================================================== DXF
class Dxf:
    """R12 ENTITIES only -- every laser cutter and Robu's uploader read it.

    Layers: CUT for the outline and holes, SCRIBE for the alignment marks a
    bonded part is set against.  A scribe line is how the V-blocks land
    concentric with a ring nobody can measure to on a bench.
    """

    def __init__(self):
        self.e = []

    def line(self, p, q, layer="CUT"):
        self.e.append(("LINE", layer, (p, q)))

    def circle(self, c, r, layer="CUT"):
        self.e.append(("CIRCLE", layer, (c, r)))

    def arc(self, c, r, a0, a1, layer="CUT"):
        self.e.append(("ARC", layer, (c, r, a0, a1)))

    def poly(self, pts, layer="CUT", close=True):
        for i in range(len(pts) - (0 if close else 1)):
            self.line(pts[i], pts[(i + 1) % len(pts)], layer)

    def text(self, p, msg, h=3.0, layer="SCRIBE"):
        self.e.append(("TEXT", layer, (p, msg, h)))

    def slot(self, c, d, r, length, layer="CUT"):
        """A slot of width 2r, its centre swept `length` along unit `d`."""
        d = np.asarray(d, float)
        d = d / np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        a, b = np.asarray(c, float) - d * length / 2, np.asarray(c, float) + d * length / 2
        self.line(a + n * r, b + n * r, layer)
        self.line(a - n * r, b - n * r, layer)
        self.arc(a, r, degrees(atan2(n[1], n[0])),
                 degrees(atan2(-n[1], -n[0])), layer)
        self.arc(b, r, degrees(atan2(-n[1], -n[0])),
                 degrees(atan2(n[1], n[0])), layer)

    def write(self, path):
        o = ["0", "SECTION", "2", "ENTITIES"]
        for kind, layer, a in self.e:
            if kind == "LINE":
                (p, q) = a
                o += ["0", "LINE", "8", layer, "10", "%.4f" % p[0], "20",
                      "%.4f" % p[1], "30", "0.0", "11", "%.4f" % q[0], "21",
                      "%.4f" % q[1], "31", "0.0"]
            elif kind == "CIRCLE":
                (c, r) = a
                o += ["0", "CIRCLE", "8", layer, "10", "%.4f" % c[0], "20",
                      "%.4f" % c[1], "30", "0.0", "40", "%.4f" % r]
            elif kind == "TEXT":
                (q, msg, h) = a
                o += ["0", "TEXT", "8", layer, "10", "%.4f" % q[0], "20",
                      "%.4f" % q[1], "30", "0.0", "40", "%.3f" % h, "1", msg]
            else:
                (c, r, a0, a1) = a
                o += ["0", "ARC", "8", layer, "10", "%.4f" % c[0], "20",
                      "%.4f" % c[1], "30", "0.0", "40", "%.4f" % r,
                      "50", "%.4f" % a0, "51", "%.4f" % a1]
        o += ["0", "ENDSEC", "0", "EOF"]
        open(path, "w").write("\n".join(o) + "\n")
        return len(self.e)


# ======================================================== the flat frame
# The plate's frame: the ring's axis at the origin, +z up, +y to the right,
# and the ring's azimuth measured from straight DOWN as everywhere else --
# az -> (r sin az, -r cos az).  The plate is LASER CUT, not printed: it is
# a 6 mm slab whose whole job is to hold three shafts, a motor and six
# standoffs concentric, and acrylic cut in one pass is flatter and stiffer
# than the same shape printed in layers.
H_AXIS    = 45.0        # mm, the rods' height above the base
PLATE_T   = 6.0
BASE_T    = 6.0
MOTOR_AZ  = 123.0       # between pinions 1 and 2, which is what keeps the
                        # belt's quadrilateral convex -- see belt_path()
# WHY THE MOTOR STANDS THIS FAR OUT.  At 40 mm it sat on top of a rail arc:
# its boss slot covered that arc's middle and its two inner bolt slots
# covered both ends, and `standoff_az` could not place a single hole in the
# arc -- which is the placement rule finding a real collision, not a bug in
# the rule.  Pushed out until its inner bolt circle clears the standoff
# circle, it also lands the belt on a standard closed loop; `check_cad`
# asserts both, so this number cannot quietly go back.
MOTOR_R   = 52.0
MOTOR_SLOT = 12.0       # mm of radial travel: the tensioner
SLOT_AZ   = 270.0       # the diagonal leaves the bore this way
SLOT_HALF = 7.0
# The rail's skirt is the pad a standoff is BONDED to, so it has to be wider
# than a standoff's end, not just wide enough to exist.  At r 27 the short
# rail arcs -- which the widened pinion window cut to 17.8 degrees -- had
# 7.8 mm of arc and 3.95 mm of radius to take a 6 mm pad, and `standoff_az`
# could not place a hole in them at all.
SKIRT_R   = 30.0
STANDOFF_R = None       # set below, from the skirt
TAB_W, TAB_H = 20.0, 8.0
STANDOFF_R = (rail_r_out() + SKIRT_R) / 2.0


def _pt(az, r):
    a = radians(az)
    return np.array([r * sin(a), -r * cos(a)])


def belt_path(motor_az=MOTOR_AZ, motor_r=MOTOR_R):
    """The belt's vertices, its length, and whether it is a belt at all.

    ONE BELT ROUND THE OUTSIDE OF FOUR PULLEYS.  That drives all four the
    same way and phases them rigidly, which is what the drive needs and
    what three independent servos could not do.  It only works while the
    four centres are CONVEX -- a motor tucked in behind a pinion puts that
    pinion inside the hull and the belt never touches it.  So convexity is
    returned, not assumed.
    """
    M = Ring.mesh()
    pts = [_pt(a, M.centre) for a in Ring.pinion_az()]
    pts.insert(1, _pt(motor_az, motor_r))          # between pinion 0 and 1
    n = len(pts)
    cr = [float(np.cross(pts[(i + 1) % n] - pts[i], pts[(i + 2) % n] - pts[(i + 1) % n]))
          for i in range(n)]
    convex = all(c > 0 for c in cr) or all(c < 0 for c in cr)
    r_p = gt2_r(20).pitch_r
    span = sum(float(np.linalg.norm(pts[(i + 1) % n] - pts[i])) for i in range(n))
    return dict(pts=pts, convex=convex, span=span,
                length=span + 2.0 * pi * r_p, pitch_r=r_p)


def standoff_az(arcs=None, inset=9.0):
    """Where the six standoffs go: inside a rail arc, out of the rod slot,
    out from under the motor.  Placed by rule and CHECKED, because a hole
    that lands under a motor is exactly the mistake a drawing makes."""
    arcs = race_arcs() if arcs is None else arcs
    caps = motor_cuts()
    def ok(az):
        p = _pt(az, STANDOFF_R)
        if p[0] < 0.0 and abs(p[1]) < SLOT_HALF + 2.0:
            return False                           # in the rod slot
        return all(_seg_gap(p, a, b) > r + 4.0 for a, b, r in caps)
    out = []
    for t0, t1 in arcs:
        ins = min(inset, (t1 - t0) / 3.0)
        lo, hi = t0 + ins, t1 - ins
        good = [float(c) for c in np.arange(lo, hi + 1e-9, 0.5) if ok(c)]
        if not good:
            raise ValueError("no standoff fits the rail arc %.1f..%.1f" % (t0, t1))
        out += ([good[0], good[-1]] if (hi - lo) > 30.0 and
                good[-1] - good[0] > 20.0 else [good[len(good) // 2]])
    return out


def motor_cuts():
    """The motor's cuts in the plate as capsules (a, b, r): four bolt slots
    and the boss.  The BODY is not a cut -- a standoff screw under it is a
    countersink, which is a build note, not an interference."""
    mc, rad = _pt(MOTOR_AZ, MOTOR_R), _pt(MOTOR_AZ, 1.0)
    n = np.array([-rad[1], rad[0]])
    out = []
    for sy in (-1, 1):
        for sn in (-1, 1):
            c = mc + rad * sy * NEMA17_BC / 2 + n * sn * NEMA17_BC / 2
            out.append((c - rad * MOTOR_SLOT / 2, c + rad * MOTOR_SLOT / 2,
                        (M3 + FIT_SCREW) / 2.0))
    out.append((mc - rad * MOTOR_SLOT / 2, mc + rad * MOTOR_SLOT / 2,
                (NEMA17_BOSS + 2.0) / 2.0))
    return out


def under_motor(az, r=STANDOFF_R):
    """True if this standoff's screw head lands under the motor's body, and
    therefore has to be countersunk."""
    p, mc, rad = _pt(az, r), _pt(MOTOR_AZ, MOTOR_R), _pt(MOTOR_AZ, 1.0)
    n = np.array([-rad[1], rad[0]])
    d = p - mc
    return (abs(float(d @ rad)) < NEMA17_SQ / 2 + MOTOR_SLOT / 2 and
            abs(float(d @ n)) < NEMA17_SQ / 2)


def _seg_gap(p, a, b):
    d = np.asarray(b, float) - np.asarray(a, float)
    L = float(d @ d)
    t = 0.0 if L < 1e-12 else max(0.0, min(1.0, float((p - a) @ d) / L))
    return float(np.linalg.norm(p - (a + t * d)))


def plate_extent(margin=6.0, step=5.0):
    """The plate's own outline, from the features it has to contain.

    A plate size is not a design decision, it is the bounding box of the
    motor's slots, the shafts, the standoffs and the rail's skirt, plus a
    margin.  Written by hand it is a constant that goes stale the first
    time the motor moves -- which it just did.
    """
    pts = [np.array([0.0, -H_AXIS])]
    for a, b, r in motor_cuts():
        for p in (a, b):
            pts += [p + np.array([sx * r, sy * r])
                    for sx in (-1, 1) for sy in (-1, 1)]
    for az in Ring.pinion_az():
        pts.append(_pt(az, Ring.mesh().centre))
    for az in standoff_az():
        pts.append(_pt(az, STANDOFF_R))
    pts += [np.array([sx * SKIRT_R, sy * SKIRT_R])
            for sx in (-1, 1) for sy in (-1, 1)]
    P = np.array(pts)
    up = lambda v: float(np.ceil((v + margin) / step) * step)
    return (-up(-P[:, 0].min()), up(P[:, 0].max()),
            -H_AXIS, up(P[:, 1].max()))


def plate_tabs():
    """[(y, width)] of the tabs that drop into the base, at the quarter
    points of whatever outline the features asked for -- the plate is
    lopsided because the motor is on one flank, and a tab written at a
    fixed y falls off the narrow side."""
    y0, y1, _, _ = plate_extent()
    return [(y0 + (y1 - y0) * f, TAB_W) for f in (0.25, 0.75)]


def plate_dxf():
    """The back plate, 6 mm, as a laser-cutter DXF."""
    M = Ring.mesh()
    d = Dxf()
    y0, y1, z0, z1 = plate_extent()
    # outline, with two tabs dropping into the base
    tabs = plate_tabs()
    pts = [(y1, z1), (y1, z0)]
    for c, w in sorted(tabs, reverse=True):
        pts += [(c + w / 2, z0), (c + w / 2, z0 - TAB_H),
                (c - w / 2, z0 - TAB_H), (c - w / 2, z0)]
    pts += [(y0, z0), (y0, z1)]
    d.poly(pts)
    # the rod slot: the bore, and the diagonal's way out
    u = _pt(SLOT_AZ, 1.0)
    d.line((0.0, SLOT_HALF), (u[0] * abs(y0), SLOT_HALF))
    d.line((0.0, -SLOT_HALF), (u[0] * abs(y0), -SLOT_HALF))
    d.arc((0.0, 0.0), SLOT_HALF, -90.0, 90.0)
    # three pinion shafts, pressed
    for az in Ring.pinion_az():
        d.circle(_pt(az, M.centre), (3.0 + FIT_PRESS) / 2.0)
    # six standoffs, bonded to the rail's skirt
    for az in standoff_az():
        d.circle(_pt(az, STANDOFF_R), (M3 + FIT_SCREW) / 2.0)
    # the motor, on radial slots -- the slots ARE the belt tensioner
    mc, rad = _pt(MOTOR_AZ, MOTOR_R), _pt(MOTOR_AZ, 1.0)
    n = np.array([-rad[1], rad[0]])
    for sy in (-1, 1):
        for sn in (-1, 1):
            d.slot(mc + rad * sy * NEMA17_BC / 2 + n * sn * NEMA17_BC / 2,
                   rad, (M3 + FIT_SCREW) / 2.0, MOTOR_SLOT)
    d.slot(mc, rad, (NEMA17_BOSS + 2.0) / 2.0, MOTOR_SLOT)
    # scribe: where the rail's skirt lands, so the standoffs can be set to it
    d.arc((0.0, 0.0), SKIRT_R, 0.0, 360.0, "SCRIBE")
    d.arc((0.0, 0.0), ring_r_out(), 0.0, 360.0, "SCRIBE")
    d.text((y0 + 4.0, z1 - 9.0), "RIG PLATE %.0fmm  ring-drive" % PLATE_T, 4.0)
    for az in Ring.pinion_az():
        q = _pt(az, M.centre) + _pt(az, 1.0) * 5.0
        d.text((q[0] - 4.0, q[1]), "P%.2f PRESS" % (3.0 + FIT_PRESS), 2.2)
    q = _pt(standoff_az()[0], STANDOFF_R) + _pt(standoff_az()[0], 1.0) * 4.0
    d.text((q[0] - 4.0, q[1]), "M3 x6 STANDOFF", 2.2)
    d.text((mc[0] - 12.0, mc[1] + NEMA17_SQ / 2 + 4.0), "NEMA17 - SLOTS TENSION", 2.5)
    d.text((u[0] * abs(y0) * 0.6, SLOT_HALF + 2.5), "ROD SLOT", 2.5)
    return d


def base_slots():
    """[(x, y, w, h)] of the slots the plate's tabs drop into, in the base's
    frame.  Returned so `check_cad` can tie them to `plate_tabs()` rather
    than trust that two functions were edited together."""
    L = axial_layout()
    xc = L["plate_x"] + PLATE_T / 2.0
    return [(xc, yc, PLATE_T + 0.2, tw) for yc, tw in plate_tabs()]


def base_dxf(alpha):
    """The base, 6 mm.  Frame: +x along the ring's axis toward the plate,
    +y as the plate's own y.  The V-blocks are BONDED to it against scribe
    lines, not bolted to holes -- the blocks are set against an assembled
    ring, which is the only thing on the bench that knows where the axis
    really is."""
    _ALPHA = alpha
    L = axial_layout()
    d = Dxf()
    pts = [np.array([L["plate_x"] + PLATE_T, 0.0])]
    for (a, b), e, w in vjig_stations(_ALPHA):
        n = np.array([-e[1], e[0]])
        pts += [q + n * sn * w / 2 for q in (a, b) for sn in (-1, 1)]
    py0, py1, _, _ = plate_extent()
    pts += [np.array([L["plate_x"], py0]), np.array([L["plate_x"], py1])]
    P = np.array(pts)
    mg, st = 12.0, 5.0
    up = lambda v: float(np.ceil((v + mg) / st) * st)
    x0, x1 = -up(-P[:, 0].min()), up(P[:, 0].max())
    y0, y1 = -up(-P[:, 1].min()), up(P[:, 1].max())
    d.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    # the plate stands in two slots
    xc = L["plate_x"] + PLATE_T / 2.0
    for yc, tw in plate_tabs():
        d.poly([(xc - (PLATE_T + 0.2) / 2, yc - tw / 2),
                (xc + (PLATE_T + 0.2) / 2, yc - tw / 2),
                (xc + (PLATE_T + 0.2) / 2, yc + tw / 2),
                (xc - (PLATE_T + 0.2) / 2, yc + tw / 2)])
    # scribes: what the assembly's FOOTPRINT is, and the two V-blocks'.
    # THE FIRST VERSION DREW A CIRCLE OF THE RING'S OWN RADIUS HERE, and it
    # is the kind of error only a picture finds: the ring stands vertical in
    # the y-z plane, so what it puts on the base is a 2*rail_out by 2*skirt
    # rectangle, not a 40 mm circle.  A V-block bonded to that circle would
    # have been 20 mm out.
    d.poly([(-L["rail_out"], -SKIRT_R), (L["rail_out"], -SKIRT_R),
            (L["rail_out"], SKIRT_R), (-L["rail_out"], SKIRT_R)], "SCRIBE")
    d.text((-L["rail_out"] + 1.0, SKIRT_R + 2.0), "RAIL FOOTPRINT", 2.5)
    d.text((x0 + 16.0, y1 - 8.0), "RIG BASE %.0fmm" % BASE_T, 4.0)
    for i, ((a, b), e, w) in enumerate(vjig_stations(_ALPHA)):
        n = np.array([-e[1], e[0]])
        d.poly([tuple(a + n * w / 2), tuple(b + n * w / 2),
                tuple(b - n * w / 2), tuple(a - n * w / 2)], "SCRIBE")
        d.text(tuple((a + b) / 2 - np.array([8.0, 1.0])),
               "VBLOCK %s" % ("CHORD" if i == 0 else "DIAG"), 2.5)
    for sx in (x0 + 10, x1 - 10):
        for sy in (y0 + 10, y1 - 10):
            d.circle((sx, sy), 4.4 / 2.0)
    return d


def vjig_stations(alpha, near=20.0, length=25.0, foot_w=26.0,
                  clear=3.0):
    """Where the two V-blocks sit on the base, in the base's frame.

    The chord runs along the ring's own axis; the diagonal leaves the joint
    at the truss's own angle, IN THE BASE'S PLANE rather than the machine's
    vertical -- which puts both rods at one height and makes both blocks the
    same part.  The rig is not testing the approach (`check_approach` is);
    it is testing whether a band can be laid on a crossing.
    """
    if alpha is None:
        raise ValueError("the V-jig's angle is the TRUSS's alpha -- pass it")
    a = radians(float(alpha))
    e_c, e_d = np.array([-1.0, 0.0]), np.array([-cos(a), sin(a)])
    chord = (e_c * near, e_c * (near + length))
    # WHERE THE DIAGONAL'S BLOCK GOES IS SOLVED, NOT CHOSEN.  Written at the
    # same stand-off as the chord's, the two footprints overlapped: at 40
    # degrees the diagonal's inner corner landed at (-23.7, 2.9), which is
    # inside a block spanning x -45..-20 and y +-13.  `check_cad` read it as
    # -7.3 mm.  So walk it out until the rectangles are clear.
    near_d = near
    while near_d < near + 200.0:
        d = (e_d * near_d, e_d * (near_d + length))
        if _rect_gap(chord, e_c, foot_w, d, e_d, foot_w) >= clear:
            break
        near_d += 0.5
    return [(chord, e_c, foot_w), ((e_d * near_d, e_d * (near_d + length)),
                                   e_d, foot_w)]


def _rect_corners(seg, e, w):
    n = np.array([-e[1], e[0]])
    a, b = seg
    return [a + n * w / 2, b + n * w / 2, b - n * w / 2, a - n * w / 2]


def _rect_gap(s1, e1, w1, s2, e2, w2):
    """Separating-axis gap between two rectangles: positive is clear."""
    A, B = _rect_corners(s1, e1, w1), _rect_corners(s2, e2, w2)
    best = -1e9
    for e in (e1, np.array([-e1[1], e1[0]]), e2, np.array([-e2[1], e2[0]])):
        pa = [float(q @ e) for q in A]
        pb = [float(q @ e) for q in B]
        best = max(best, max(min(pb) - max(pa), min(pa) - max(pb)))
    return best
