"""The derived bat in MuJoCo: one body per part, built from product.py's
elements, nothing placed by hand.

WHY BOXES.  A MuJoCo mesh collides as its convex hull, so a handle with a
bay in it cannot be a mesh: the cells would meet the hull of the handle,
not its bay.  So every prism is laid down as boxes -- its section's grid,
run-length encoded row by row, each run a box along x -- and a thin shell
(the sleeve, the collar, the pommel cap's skirt) as staves round its
outline.  Rods are cylinders, boxes are boxes, a point mass is a small
sphere.  A prism's boxes share the prism's derived mass by volume, so the
model weighs exactly what product.py says, and MuJoCo's own inertia
integration is a second, independent integration of how it is spread.

WHY LINERS.  None of those boxes collide: at RES_MJ a voxel is coarser
than the clearances being checked.  The fits that position something --
the cells in the bay, the board on its three datums, the chords in their
sockets -- are measured against LINERS: thin staves tangent to the
analytic cavity, so 0.25 mm is measured against the drawing, not against
a 0.5 mm voxel.  A liner ring of N staves is a polygon round its circle;
the error that leaves is its sagitta, r (1/cos(pi/N) - 1), which the
check is allowed and no more.

FRAME: world = bat frame (mm here, metres in the XML).  Each part is a
free body at the origin, so mj_forward reports contacts between parts
(bodies without joints are welded to the world and never collide), and
an exploded view is a qpos.  Gravity is off: M0 looks at the bat, it does
not drop it.

COLLISION BITS (contype / conaffinity):
    1  liners       the analytic cavity a fit is measured against
    2  located      the cells, the PCB, the chords: what the liners locate
    0  everything else
"""
from math import pi, sqrt, cos, sin, tan

import numpy as np

from truss.spec import mm
from truss.geometry import chord_phi, radial
from .spec import Board, Cell, Print, Rules, Knit
from . import product as P

RES_MJ = 0.5          # mm, the grid a solid section is laid down on
N_RING = 48           # staves round a liner's circle
N_ARC = 8             # staves round a shell's corner (an arc of 120 deg) or a quarter circle
LINER_T = 0.5         # mm, a liner stave's thickness
LINER, LOCATED = 1, 2
WRITTEN = 1e-3        # mm: every position and size is written to the micrometre ("%.6f" m)
POINT_R = 0.6         # mm, the sphere a point mass is drawn as (it has a sphere's inertia)

EX, EY, EZ = np.eye(3)


def box(name, centre, half, xax, yax, rgba, ctype, caff, mass):
    """truss.mjcf's box, with the mass written to nine significant figures:
    a stave of sleeve weighs a hundredth of a gram, which six decimals of a
    kilogram round to two figures."""
    return ('<geom name="%s" type="box" pos="%s" size="%.6f %.6f %.6f" '
            'xyaxes="%.6f %.6f %.6f %.6f %.6f %.6f" rgba="%s" contype="%d" '
            'conaffinity="%d" mass="%.9g"/>'
            % (name, _v(centre), mm(half[0]), mm(half[1]), mm(half[2]),
               xax[0], xax[1], xax[2], yax[0], yax[1], yax[2], rgba, ctype, caff, mass / 1000.0))


def cylinder(name, p0, p1, r, rgba, ctype, caff, mass):
    return ('<geom name="%s" type="cylinder" fromto="%s %s" size="%.6f" rgba="%s" '
            'contype="%d" conaffinity="%d" mass="%.9g"/>'
            % (name, _v(p0), _v(p1), mm(r), rgba, ctype, caff, mass / 1000.0))


def _rgba(c, a=None):
    r, g, b, a0 = c
    return "%.3f %.3f %.3f %.3f" % (r, g, b, a0 if a is None else a)


def sagitta(r, n=N_RING):
    """How far a ring of n flat staves stands off its circle, at most."""
    return r * (1.0 / cos(pi / n) - 1.0)


# ------------------------------------------------------------ sections
def _runs(sec, res=RES_MJ):
    """The section's grid as runs: (y0, y1, z0, z1) per run, row by row."""
    y0, y1, z0, z1 = sec.bbox()
    ys = np.arange(y0 + res / 2.0, y1, res)
    zs = np.arange(z0 + res / 2.0, z1, res)
    if len(ys) == 0 or len(zs) == 0:
        return []
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    m = sec.mask(Y, Z)
    out = []
    for j, z in enumerate(zs):
        col = m[:, j].astype(np.int8)
        d = np.diff(np.concatenate([[0], col, [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        for a, b in zip(starts, ends):
            out.append((ys[a] - res / 2.0, ys[b - 1] + res / 2.0, z - res / 2.0, z + res / 2.0))
    return out


def _outline(shape, n_arc=N_ARC):
    """Points round a circle or a rounded triangle, counter-clockwise in
    (y, z), with the outward normal at each: the path a shell's staves
    follow."""
    if shape.kind == "circle":
        cy, cz, r = shape.p
        th = np.linspace(0.0, 2.0 * pi, 4 * n_arc + 1)[:-1]
        return [(np.array([cy + r * cos(a), cz + r * sin(a)]), np.array([cos(a), sin(a)])) for a in th]
    R, rc = shape.p
    V = [np.asarray(R * radial(chord_phi(k))[1:]) for k in range(3)]
    # orient counter-clockwise
    if np.cross(V[1] - V[0], V[2] - V[0]) < 0:
        V = V[::-1]
    pts = []
    for i in range(3):
        a, b, c = V[i - 1], V[i], V[(i + 1) % 3]
        n_in = np.array([(b - a)[1], -(b - a)[0]]) / np.linalg.norm(b - a)    # outward of edge a->b
        n_out = np.array([(c - b)[1], -(c - b)[0]]) / np.linalg.norm(c - b)   # outward of edge b->c
        t0, t1 = np.arctan2(n_in[1], n_in[0]), np.arctan2(n_out[1], n_out[0])
        if t1 < t0:
            t1 += 2.0 * pi
        for t in np.linspace(t0, t1, n_arc + 1):
            nn = np.array([cos(t), sin(t)])
            pts.append((b + rc * nn, nn))
    return pts


def _is_shell(sec):
    """A solid less one hole of its own kind and centre: a shell, laid
    down as staves along its middle line rather than as a voxel ring."""
    if len(sec.holes) != 1:
        return False
    s, h = sec.solid, sec.holes[0]
    if s.kind != h.kind or s.kind not in ("circle", "rtri"):
        return False
    if s.kind == "circle":
        return abs(s.p[0] - h.p[0]) < 1e-9 and abs(s.p[1] - h.p[1]) < 1e-9 and h.p[2] < s.p[2]
    return abs(s.p[0] - h.p[0]) < 1e-9 and h.p[1] < s.p[1]


def _shell_staves(sec, x0, x1):
    """(centre, half, tangent) of each stave of a shell section."""
    s, h = sec.solid, sec.holes[0]
    t = (s.p[-1] - h.p[-1])
    mid = P.Shape(s.kind, s.p[:-1] + ((s.p[-1] + h.p[-1]) / 2.0,))
    pts = _outline(mid)
    out = []
    for i in range(len(pts)):
        (pa, _), (pb, _) = pts[i], pts[(i + 1) % len(pts)]
        seg = pb - pa
        L = float(np.linalg.norm(seg))
        if L < 1e-9:
            continue
        tan_ = np.array([0.0, seg[0] / L, seg[1] / L])
        c = (pa + pb) / 2.0
        out.append((np.array([(x0 + x1) / 2.0, c[0], c[1]]), ((x1 - x0) / 2.0, L / 2.0, t / 2.0), tan_))
    return out


# ------------------------------------------------------------------ parts
def part_geoms(name, part, rgba=None, res=RES_MJ):
    """Visual, mass-carrying geoms for one part, in the bat frame."""
    col = _rgba(part.rgba if rgba is None else rgba)
    g = []
    k = 0

    def nm():
        nonlocal k
        k += 1
        return "%s/v%d" % (name, k)
    for x0, x1, sec in part.prisms:
        # the prism's derived mass, shared among its boxes by volume: the
        # model weighs what product.py says, and only the spread is the grid's
        m_prism = part.rho * P.moments(sec)[0] * (x1 - x0)
        if _is_shell(sec):
            boxes = [(c, half, tan_) for c, half, tan_ in _shell_staves(sec, x0, x1)]
        else:
            boxes = [(np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0]),
                      ((x1 - x0) / 2.0, (y1 - y0) / 2.0, (z1 - z0) / 2.0), EY)
                     for y0, y1, z0, z1 in _runs(sec, res)]
        vol = sum(8.0 * h[0] * h[1] * h[2] for _, h, _ in boxes)
        for c, half, yax in boxes:
            g.append(box(nm(), c, half, EX, yax, col, 0, 0,
                         mass=m_prism * 8.0 * half[0] * half[1] * half[2] / vol))
    for p0, p1, r, m in part.rods:
        g.append(cylinder(nm(), p0, p1, r, col, 0, 0, mass=m))
    for c, half, m in part.boxes:
        g.append(box(nm(), c, half, EX, EY, col, 0, 0, mass=m))
    for p, m in part.points:
        g.append('<geom name="%s" type="sphere" pos="%s" size="%.6f" rgba="%s" contype="0" '
                 'conaffinity="0" mass="%.9f"/>' % (nm(), _v(p), mm(POINT_R), col, m / 1000.0))
    return g


def _v(p):
    return "%.6f %.6f %.6f" % (mm(p[0]), mm(p[1]), mm(p[2]))


# ------------------------------------------------------------------ liners
def liner_ring(prefix, cy, cz, r, x0, x1, n=N_RING, t=LINER_T):
    """n staves whose inner faces circumscribe the circle (cy, cz, r)."""
    out = []
    half_w = r * tan(pi / n) * 1.02          # a hair long, so the corners close
    for i in range(n):
        a = 2.0 * pi * (i + 0.5) / n
        nn = np.array([0.0, cos(a), sin(a)])
        c = np.array([(x0 + x1) / 2.0, cy, cz]) + nn * (r + t / 2.0)
        tan_ = np.array([0.0, -sin(a), cos(a)])
        out.append(box("%s/%d" % (prefix, i), c, ((x1 - x0) / 2.0, half_w, t / 2.0), EX, tan_,
                       "0.9 0.2 0.2 0.0", LINER, LOCATED, mass=1e-6))
    return out


def _slab(name, lo, hi):
    """An axis-aligned liner box from corner lo to corner hi."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return box(name, (lo + hi) / 2.0, tuple((hi - lo) / 2.0), EX, EY, "0.9 0.2 0.2 0.0",
               LINER, LOCATED, mass=1e-6)


def liners(d):
    """{group: [geom xml]} on the part each belongs to, and the fits they
    measure: {group: (part, located geom prefix, expected gap mm, kind)}.
    kind: "clear" (a gap of at least `expected`), "datum" (touching),
    "press" (penetrating by `expected`)."""
    lay, core, t = d.lay, d.core, d.frame.truss
    T = Print.WALL
    out, fits = {}, {}
    # the bay, round the cells, mouth to + plate
    out["core:bay"] = liner_ring("core/liner/bay", 0.0, -core.e, core.r_bay, lay.x_face, lay.x_plus)
    fits["core:bay"] = ("cell", Print.CLEAR, "clear")
    # the board's three datums, its hook and its pocket roof
    zl, zf, zc = core.z_ledge, core.z_floor, core.z_ceiling
    x_lo, x_hi = lay.x_hook, lay.x_stop
    out["core:datum_y"] = [_slab("core/liner/datum_y", (x_lo, core.y_datum - T, zf), (x_hi, core.y_datum, zl))]
    fits["core:datum_y"] = ("board/pcb", 0.0, "datum")
    out["core:ledges"] = [
        _slab("core/liner/ledges/a", (x_lo, core.y_datum - T, zl), (x_hi, core.pocket[0], zl + T)),
        _slab("core/liner/ledges/b", (x_lo, core.pocket[1], zl), (x_hi, core.y_board + T, zl + T))]
    fits["core:ledges"] = ("board/pcb", 0.0, "datum")
    out["core:stop"] = [_slab("core/liner/stop", (x_hi, core.y_datum - T, zf), (x_hi + T, core.y_board + T, zc))]
    fits["core:stop"] = ("board/pcb", 0.0, "datum")
    out["core:hook"] = [_slab("core/liner/hook", (lay.x_face, core.y_board - core.hook, core.z_pcb),
                              (lay.x_hook, core.y_board + T, zl))]
    fits["core:hook"] = ("board/pcb", lay.x_b0 - lay.x_hook, "clear")
    out["core:roof"] = [_slab("core/liner/roof", (x_lo, core.pocket[0], zc), (x_hi, core.pocket[1], zc + T))]
    fits["core:roof"] = ("board/module", Print.CLEAR, "clear")
    # the chords: a sliding socket in the shoulder, a press socket in the tip cap
    ss, ps = [], []
    for k in range(3):
        y, z = R_axis(t.R, k)
        ss += liner_ring("core/liner/sockets/%d" % k, y, z, t.d_chord / 2.0 + Print.CLEAR,
                         lay.x_c0, lay.x_blade0, n=N_RING)
        ps += liner_ring("tip_cap/liner/press/%d" % k, y, z, t.d_chord / 2.0 - Print.INTERFERENCE,
                         lay.x_tip0, lay.x_c1, n=N_RING)
    out["core:sockets"] = ss
    fits["core:sockets"] = ("frame/chord", Print.CLEAR, "clear")
    out["tip_cap:press"] = ps
    fits["tip_cap:press"] = ("frame/chord", Print.INTERFERENCE, "press")
    return out, fits


def R_axis(R, k):
    """(y, z) of chord k's axis."""
    return tuple(R * radial(chord_phi(k))[1:])


def located_geoms(d):
    """{part: [geom xml]}: the collision stand-ins of what the liners locate."""
    lay, core, t = d.lay, d.core, d.frame.truss
    out = {}
    for i in range(Rules.N_CELLS):
        x1 = lay.x_plus - i * Cell.L_MAX
        out["cell%d" % (i + 1)] = [cylinder("cell%d/c/body" % (i + 1),
                                            (x1 - Cell.L_MAX, 0.0, -core.e), (x1 - Cell.PIP_H, 0.0, -core.e),
                                            Cell.D_MAX / 2.0, "0 0 0 0", LOCATED, LINER, mass=1e-6)]
    pcb, mod = d.parts["board"].boxes
    out["board"] = [box("board/pcb", pcb[0], pcb[1], EX, EY, "0 0 0 0", LOCATED, LINER, mass=1e-6),
                    box("board/module", mod[0], mod[1], EX, EY, "0 0 0 0", LOCATED, LINER, mass=1e-6)]
    chords = []
    shift = np.array([lay.x_c0, 0.0, 0.0])
    for i, r in enumerate(d.geom.rods):
        if r.kind == "chord":
            chords.append(cylinder("frame/chord%d" % i, r.p0 + shift, r.p1 + shift, r.r,
                                   "0 0 0 0", LOCATED, LINER, mass=1e-6))
    out["frame"] = chords
    return out


# ------------------------------------------------------------------ bands
INK = 0.15            # mm, how proud of the knit a band is drawn: a picture, not a measurement


def band_parts(d):
    """The two bands, printed on the sleeve over the blade, as parts of
    their own so a cut view cuts them too.  Ink: no mass worth having."""
    out = {}
    hull = d.offsets[0]
    for i, (c, _) in enumerate(d.bands):
        sec = P.Section(P.rtri(d.R, hull + Knit.T + INK), (P.rtri(d.R, hull + Knit.T),))
        rgba = tuple(Rules.BAND_RGB[i]) + (1.0,)
        out["band%d" % (i + 1)] = P.Part("band%d" % (i + 1), "printed on the sleeve", rgba,
                                         prisms=[(c - Rules.BAND_W / 2.0, c + Rules.BAND_W / 2.0, sec)],
                                         rho=1e-12)
    return out


# ------------------------------------------------------------------ scene
ORDER = ("core", "pommel_cap", "board", "contacts", "spring", "cell1", "cell2", "frame",
         "tip_cap", "collar", "sleeve", "band1", "band2")


def scene(d, width=1280, height=800, floor=True):
    """The bat lying face down along x, every part a free body at its
    design pose."""
    lin, _ = liners(d)
    loc = located_geoms(d)
    bodies = []
    for name in ORDER:
        if name not in d.parts:
            continue
        g = part_geoms(name, d.parts[name])
        for key, xs in lin.items():
            if key.split(":")[0] == name:
                g += xs
        g += loc.get(name, [])
        if name == "sleeve":
            for k, bp in band_parts(d).items():
                g += part_geoms(k, bp)
        bodies.append('    <body name="%s" pos="0 0 0">\n      <freejoint name="%s"/>\n      %s\n    </body>'
                      % (name, name, "\n      ".join(g)))
    L = mm(d.bat.length)
    z_low = min(P.part_extent(p)[2] for p in d.parts.values())
    fl = ('    <geom name="floor" type="plane" pos="%.4f 0 %.6f" size="%.3f %.3f 0.01" material="floor" '
          'contype="0" conaffinity="0"/>' % (L / 2.0, mm(z_low - 0.5), L, L)) if floor else ""
    return """<mujoco model="bat_%s">
  <compiler angle="degree" autolimits="true" boundmass="1e-9" boundinertia="1e-15"/>
  <option timestep="0.001" gravity="0 0 0"/>
  <visual>
    <headlight ambient="0.55 0.55 0.55" diffuse="0.45 0.45 0.45" specular="0.1 0.1 0.1"/>
    <quality shadowsize="2048"/>
    <map znear="0.001" zfar="20"/>
    <global offwidth="%d" offheight="%d"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.94 0.94 0.92"
             rgb2="0.88 0.88 0.86" width="300" height="300"/>
    <material name="floor" texture="grid" texrepeat="8 8" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light name="key" pos="%.3f -0.6 0.9" dir="0 0.5 -1" diffuse="0.5 0.5 0.5"/>
%s
%s
  </worldbody>
</mujoco>
""" % (d.bat.name, width, height, L / 2.0, fl, "\n".join(bodies))


def build(d, **kw):
    """(model, data) at the design pose, forward-computed."""
    import mujoco
    m = mujoco.MjModel.from_xml_string(scene(d, **kw))
    data = mujoco.MjData(m)
    mujoco.mj_forward(m, data)
    return m, data


# ------------------------------------------------------------ measuring
def _ids(m, prefix):
    import mujoco
    out = []
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if nm.startswith(prefix):
            out.append(g)
    return out


def measure_liners(d, m, data, distmax=5.0):
    """Every liner fit as built: [(group, kind, expected mm, measured mm,
    allowed error mm)].  `measured` is the smallest signed distance
    between the liner's staves and what it locates (negative: they
    overlap); `allowed` is the stave ring's sagitta and a micrometre of
    narrowphase."""
    import mujoco
    _, fits = liners(d)
    lay, core, t = d.lay, d.core, d.frame.truss
    radius = {"core:bay": core.r_bay, "core:sockets": t.d_chord / 2.0 + Print.CLEAR,
              "tip_cap:press": t.d_chord / 2.0 - Print.INTERFERENCE}
    out = []
    fromto = np.zeros(6)
    for group, (located, expected, kind) in fits.items():
        part, name = group.split(":")
        ga = _ids(m, "%s/liner/%s" % (part, name))
        gb = _ids(m, located)
        best = distmax
        for a in ga:
            for b in gb:
                best = min(best, mujoco.mj_geomDistance(m, data, a, b, mm(distmax), fromto) * 1000.0)
        allowed = (sagitta(radius[group]) if group in radius else 0.0) + 1e-3
        meas = -best if kind == "press" else best
        out.append((group, kind, expected, meas, allowed))
    return out


def contacts(m, data):
    """[(geom a, geom b, dist mm)] at the current pose."""
    import mujoco
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        out.append((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2), c.dist * 1000.0))
    return out


def composite(m, data):
    """(mass g, centre mm, inertia about the centre g mm^2) of every body
    but the world, from MuJoCo's own integration of the geoms."""
    M = m.body_mass[1:].sum()
    c = (m.body_mass[1:, None] * data.xipos[1:]).sum(0) / M
    I = np.zeros((3, 3))
    for b in range(1, m.nbody):
        Rb = data.ximat[b].reshape(3, 3)
        Ib = Rb @ np.diag(m.body_inertia[b]) @ Rb.T
        dd = data.xipos[b] - c
        I += Ib + m.body_mass[b] * ((dd @ dd) * np.eye(3) - np.outer(dd, dd))
    return M * 1000.0, c * 1000.0, I * 1e9


# ------------------------------------------------------------ looking
# Transparency does not work on a voxel part: a ray through the handle
# crosses a hundred boxes, and a hundred faint faces are opaque.  So the
# views CUT instead -- a part clipped to z below a plane, or to a thin slab
# across x -- the way a drawing shows a section.
def _clip_rod(p0, p1, r, m, lo, hi, axis):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    a, b = p0[axis], p1[axis]
    if abs(b - a) < 1e-12:
        return (p0, p1, r, m) if lo <= a <= hi else None
    t0, t1 = sorted(((lo - a) / (b - a), (hi - a) / (b - a)))
    t0, t1 = max(t0, 0.0), min(t1, 1.0)
    if t1 - t0 < 1e-9:
        return None
    return (p0 + t0 * (p1 - p0), p0 + t1 * (p1 - p0), r, m * (t1 - t0))


def cut_part(part, z_cut=None, x_slab=None):
    """The part with everything above z_cut, or outside the slab
    (x0, x1), taken away.  Masses go with what is kept, so a cut model
    is still a model."""
    from dataclasses import replace
    prisms, rods, boxes, points = [], [], [], []
    for x0, x1, sec in part.prisms:
        if x_slab is not None:
            x0, x1 = max(x0, x_slab[0]), min(x1, x_slab[1])
            if x1 - x0 < 1e-9:
                continue
        if z_cut is not None:
            y0, y1, z0, z1 = sec.bbox()
            if z0 >= z_cut:
                continue
            if z1 > z_cut:
                sec = P.Section(sec.solid, sec.holes + (P.rect(y0 - 1.0, y1 + 1.0, z_cut, z1 + 1.0),))
        prisms.append((x0, x1, sec))
    for p0, p1, r, m in part.rods:
        q = (p0, p1, r, m)
        if x_slab is not None:
            q = _clip_rod(p0, p1, r, m, x_slab[0], x_slab[1], 0)
        if q is not None:
            rods.append(q)
    for c, h, m in part.boxes:
        lo, hi = np.asarray(c) - h, np.asarray(c) + h
        if x_slab is not None:
            lo[0], hi[0] = max(lo[0], x_slab[0]), min(hi[0], x_slab[1])
            if hi[0] - lo[0] < 1e-9:
                continue
        boxes.append(((lo + hi) / 2.0, tuple((hi - lo) / 2.0), m))
    for pt, m in part.points:
        if x_slab is None or x_slab[0] <= pt[0] <= x_slab[1]:
            points.append((pt, m))
    return replace(part, prisms=prisms, rods=rods, boxes=boxes, points=points)


def look_scene(d, parts, width, height, lookat, floor=False, offsets=None):
    """A scene of `parts` (name -> Part) for looking at, nothing else in it;
    `offsets` (name -> mm) moves a part bodily, for an exploded view."""
    bodies = []
    offsets = offsets or {}
    for name in ORDER:
        if name not in parts:
            continue
        g = part_geoms(name, parts[name])
        if g:
            bodies.append('    <body name="%s" pos="%s">\n      %s\n    </body>'
                          % (name, _v(offsets.get(name, (0.0, 0.0, 0.0))), "\n      ".join(g)))
    z_low = min(P.part_extent(p)[2] for p in d.parts.values())
    L = mm(d.bat.length)
    fl = ('    <geom name="floor" type="plane" pos="%.4f 0 %.6f" size="%.3f %.3f 0.01" material="floor"/>'
          % (L / 2.0, mm(z_low - 0.5), L, L)) if floor else ""
    return """<mujoco model="look">
  <compiler angle="degree" inertiafromgeom="false"/>
  <visual>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
    <map znear="0.0005" zfar="20"/>
    <global offwidth="%d" offheight="%d"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.94 0.94 0.92"
             rgb2="0.86 0.86 0.84" width="300" height="300"/>
    <material name="floor" texture="grid" texrepeat="16 16" reflectance="0.0"/>
  </asset>
  <worldbody>
    <light name="key" pos="%.4f %.4f %.4f" dir="0.3 0.4 -1" diffuse="0.45 0.45 0.45"/>
%s
%s
  </worldbody>
</mujoco>
""" % (width, height, mm(lookat[0]), mm(lookat[1]) - 0.3, mm(lookat[2]) + 0.6, fl, "\n".join(bodies))


def views(d, width=960, height=600):
    """[(label, image)] of the bat: whole, the frame bare, the handle cut
    open at the board and at the cells, the shoulder and the tip cut open,
    four slices across it, and the parts exploded.  Every one is a picture
    of the MODEL as derived -- the point is to see whether it is the bat."""
    import mujoco
    lay, core = d.lay, d.core
    out = []

    def shot(label, parts, lookat, dist, az, el, floor=False, offsets=None):
        m = mujoco.MjModel.from_xml_string(look_scene(d, parts, width, height, lookat, floor=floor,
                                                      offsets=offsets))
        data = mujoco.MjData(m)
        mujoco.mj_forward(m, data)
        with mujoco.Renderer(m, height, width) as rend:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = np.asarray(lookat, float) / 1000.0
            cam.distance, cam.azimuth, cam.elevation = dist / 1000.0, az, el
            rend.update_scene(data, cam)
            out.append((label, rend.render().copy()))

    L = d.bat.length
    allp = dict(d.parts)
    allp.update(band_parts(d))
    shot("whole bat, face down", allp, (L / 2.0, 0, 0), 1.05 * L, 110, -24, floor=True)
    bare = {k: v for k, v in allp.items() if k not in ("sleeve", "band1", "band2")}
    shot("the frame bare: sleeve off", bare, (L / 2.0 + 0.1 * L, 0, 0), 0.8 * L, 110, -24, floor=True)
    shells = ("core", "pommel_cap", "sleeve", "collar", "tip_cap", "band1", "band2")
    h = lay.x_floor_end

    def cut(z, x_hi=None, drop=()):
        return {k: (cut_part(v, z_cut=z) if k in shells else v) for k, v in allp.items()
                if (x_hi is None or P.part_extent(v)[5] < x_hi) and k not in drop}
    zb = core.z_pcb + Board.PCB_T / 2.0
    shot("handle cut at the board's mid-plane", cut(zb, lay.x_shoulder), (h / 2.0, 0, 0), 0.9 * h, 115, -55)
    shot("...the board out: hook, tabs, strips, window", cut(zb, lay.x_shoulder, drop=("board",)),
         (lay.x_stop / 2.0, 0, zb), 0.7 * lay.x_stop, 125, -58)
    shot("handle cut through the cells' axis", cut(-core.e, lay.x_shoulder), (h / 2.0, 0, -core.e),
         0.9 * h, 115, -55)
    xs = lay.x_shoulder
    shot("shoulder cut at the bat's axis", cut(0.0), (xs + 12, 0, 0), 120, 125, -55)
    shot("tip cut at the bat's axis", cut(0.0), (lay.x_tip0, 0, 0), 120, 55, -55)
    for label, x in (("slice at the IMU", lay.x_imu),
                     ("slice through the bay", (lay.x_stop + lay.x_floor) / 2.0),
                     ("slice through the sockets", (lay.x_c0 + lay.x_blade0) / 2.0),
                     ("slice through the first band", d.bands[0][0])):
        sl = {k: cut_part(v, x_slab=(x - 0.4, x + 0.4)) for k, v in allp.items()}
        sl = {k: v for k, v in sl.items() if v.prisms or v.rods or v.boxes or v.points}
        r = max(P.part_extent(v)[4] for v in sl.values())
        shot(label, sl, (x, 0, 0), 4.2 * r, 0.0, -2.0)
    # exploded: each part lifted by its place in the build, seen from the side
    lift = {k: (0.0, 0.0, (i - 3) * 40.0) for i, k in enumerate(ORDER)}
    lift["band1"] = lift["band2"] = lift["sleeve"]
    shot("exploded, in build order", allp, (L / 2.0, 0, 160), 1.25 * L, 90, -8, offsets=lift)
    return out
