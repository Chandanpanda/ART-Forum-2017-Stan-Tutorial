"""The calibration station's rig as a MuJoCo scene, built as built.

    RigBuildDraw.draw(rig, cams, rng)   how far this build lies from its
                   drawing: every camera's pose, lens, colour and light, the
                   gimbal's centre, axes and encoder zeros, each marker on
                   its stem, the bar's balls as made and as certified
    scene(rig, site, cams, b, payload)  the MJCF: the gimbal on its posts
                   with the bat (or the certified bar) in its clamp, the
                   gantry parked over it, the eight cameras and their ring
                   lights -- as built when b is given, as drawn when not

The scene is kinematic: the gimbal's two hinges are set and mj_kinematics
run; nothing is stepped, because nothing here is dynamics (CLAUDE.md:
physics only for mechanisms).

GEOM GROUPS.  0 the structure (rings, posts, clamp, gantry, floor);
1 the bat's own geoms, every part of it (thousands, for pictures and for
the model camera's occlusion); 2 a few boxes enclosing the bat, for the
placement solver's rays, never drawn; 3 the markers and the bar's balls,
which no ray stops at, so a ray reaches a ball unless something else is in
the way; 4 the balls' stems, apart so that a ball's own stem, which pulls
its centroid (vision.stem_bias_px) but does not hide it from a camera,
can be told from anything else in a ray's way (vision.clear_of).  A
renderer must show groups 0, 1, 3 and 4 and hide group 2 (vision.py does).

COLOURS ARE REFLECTANCES.  A designer gives the bat's colours (the bands,
the sleeve) as sRGB, gamma-encoded; a print of one reflects its decoded,
linear value, and that is what a geom's rgba is here, since MuJoCo lights
linearly (colour.decode).  The card's patches are given as measured
linear reflectances, and go in as they are.

THE RING LIGHTS FALL OFF.  A ring light 60 mm across at a metre or more is
a point at the lens: its light on a surface falls as the inverse square of
the distance (attenuation 0 0 1, in metres).  The colour exposure is set
for white at the nearest any surface of the bat can come to a camera,
vision.expose_mm, so nothing nearer clips.

A MARKER IS EMISSIVE.  A retroreflective ball lit by the ring light round
the lens that looks at it returns that light to the lens far brighter than
anything diffuse (spec.Marker.RETRO_GAIN), evenly over its disc: an
emissive white sphere is that, for one camera at a time.  The bat's
surfaces are matte (no specular), as the knit and the card are.
"""
from dataclasses import dataclass, field
from math import pi, sqrt, cos, sin, atan, degrees, radians, ceil

import numpy as np

from truss.spec import mm
from ..spec import RingCam, Marker, Gimbal, Clamp, CalBar, RigBuild, Rig, Material, Print, Stepper
from .. import mjcf as bat_mjcf
from .lens import Camera, Lens, rodrigues, DIST
from . import spec as RS

C_FRAME = "0.55 0.56 0.60 1"
C_RING = "0.62 0.64 0.68 1"
C_CLAMP = "0.18 0.19 0.22 1"
C_FLOOR = "0.20 0.20 0.21 1"
C_BAR = "0.08 0.08 0.09 1"
STRUCT, BAT, PROXY, BALL, STEM = 0, 1, 2, 3, 4
N_RING = 72                    # boxes a ring is drawn with: a 5-degree polygon
DEFAULT_DENSITY = 1000.0       # kg/m^3, MuJoCo's own default: what an unweighed geom weighs at


def _v(p):
    return "%.6f %.6f %.6f" % (mm(p[0]), mm(p[1]), mm(p[2]))


def _box(name, c, half, rgba, group=STRUCT, xyaxes=None):
    ax = "" if xyaxes is None else ' xyaxes="%s"' % " ".join("%.9f" % v for v in xyaxes)
    return ('<geom name="%s" type="box" pos="%s" size="%.6f %.6f %.6f" rgba="%s" group="%d"%s/>'
            % (name, _v(c), mm(half[0]), mm(half[1]), mm(half[2]), rgba, group, ax))


def _cyl(name, p0, p1, r, rgba, group=STRUCT):
    return ('<geom name="%s" type="cylinder" fromto="%s %s" size="%.6f" rgba="%s" group="%d"/>'
            % (name, _v(p0), _v(p1), mm(r), rgba, group))


def _ball(name, c, r):
    return ('<geom name="%s" type="sphere" pos="%s" size="%.6f" material="retro" group="%d"/>'
            % (name, _v(c), mm(r), BALL))


def _ring(prefix, R, S, rgba, n=N_RING):
    """A square-section ring of centreline radius R in the body's x-y
    plane, as n boxes."""
    out = []
    half_len = R * np.tan(pi / n) + S / 2.0 * np.tan(pi / n)
    for k in range(n):
        t = 2.0 * pi * (k + 0.5) / n
        c = (R * cos(t), R * sin(t), 0.0)
        out.append(_box("%s%d" % (prefix, k), c, (S / 2.0, half_len, S / 2.0), rgba,
                        xyaxes=(cos(t), sin(t), 0.0, -sin(t), cos(t), 0.0)))
    return out


def _ring_foot(ball, R_ring, S):
    """Where a ring marker's stem leaves the tube: its outer corner."""
    p = np.asarray(ball, float)
    rho = np.hypot(p[0], p[1])
    u = np.array([p[0] / rho, p[1] / rho, 0.0])
    return (R_ring + S / 2.0) * u + np.array([0.0, 0.0, np.sign(p[2]) * S / 2.0])


def _bar_foot(ball):
    """Where a bar ball's stem leaves the bar's tube."""
    p = np.asarray(ball, float)
    rho = np.hypot(p[1], p[2])
    return np.array([p[0], 0.0, 0.0]) + np.array([0.0, p[1] / rho, p[2] / rho]) * CalBar.TUBE_D / 2.0


def _feet(rig, markers, bar):
    feet = [_ring_foot(p, rig.R_i, Gimbal.SECTION) for p in markers] + [_bar_foot(p) for p in bar]
    P = np.vstack([np.asarray(markers, float).reshape(-1, 3), np.asarray(bar, float).reshape(-1, 3)])
    return np.asarray(feet) - P


def stems(rig, markers, bar=()):
    """(K, 3) unit vectors, body frame: from each ball's centre toward its
    stem, the ring's markers then the bar's balls -- the side from which a
    stem pulls a ball's centroid (vision.stem_bias_px)."""
    v = _feet(rig, markers, bar)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def stem_lengths(rig, markers, bar=()):
    """(K,) mm: each stem from its ball's surface to the tube it stands on."""
    return np.linalg.norm(_feet(rig, markers, bar), axis=1) - rig.ball_r


def _stem(name, ball, R_ring, S, r_ball):
    """The stem from the ring tube's outer corner to a ball."""
    p = np.asarray(ball, float)
    foot = _ring_foot(p, R_ring, S)
    d = p - foot
    end = foot + d * (1.0 - r_ball / np.linalg.norm(d))
    return _cyl(name, foot, end, Marker.STEM_D / 2.0, C_FRAME, group=STEM)


# ============================================================ THE BUILD
def _monotonic(L, r_max):
    """A real lens's distortion is monotonic over its field: the radius in
    the image grows with the radius in the world, out to the corner."""
    r = np.linspace(0.0, r_max, 400)
    r2 = r * r
    rad = 1.0 + 3.0 * L.k1 * r2 + 5.0 * L.k2 * r2 * r2 + 7.0 * L.k3 * r2 ** 3
    return bool(np.all(rad > 0.0))


@dataclass
class RigBuildDraw:
    """Where this build lies from its drawing (spec.RigBuild, 1 sigma), drawn
    once.  The scene is built from it; the calibration never sees it."""
    cams: list                 # [lens.Camera] as built
    colour: list               # [3x3] each sensor's colour response: raw = M @ true
    light: np.ndarray          # each ring light's output, as a fraction of its drawing
    centre: np.ndarray         # mm, the gimbal's centre from its drawing
    axis_o: np.ndarray         # the outer axis, unit, rig frame
    axis_i: np.ndarray         # the inner axis, unit, outer body's frame
    offset_i: np.ndarray       # mm, the inner axis off the outer, outer body's frame
    zero: np.ndarray           # deg, (outer, inner) encoder zeros: the hinge is at command + zero
    markers: np.ndarray        # (K, 3) mm body frame, as built
    bar: np.ndarray            # (N, 3) mm body frame, the bar's balls as made
    bar_cert: np.ndarray       # (N, 3) mm, and as its certificate gives them
    stem: float = 1.0          # the stems' pull on a centroid, of the area model's (spec.RigBuild.STEM)

    @classmethod
    def draw(cls, rig, cams, rng):
        B = RigBuild
        out = []
        for c in cams:
            L0 = c.lens
            centre = c.centre + rng.normal(0.0, B.CAM_POS, 3)
            R = rodrigues(rng.normal(0.0, radians(B.CAM_TILT), 3)) @ c.R
            f = L0.fx * (1.0 + rng.normal(0.0, B.FOCAL))
            cx, cy = L0.cx + rng.normal(0.0, B.CENTRE_PX), L0.cy + rng.normal(0.0, B.CENTRE_PX)
            # the corner of the sensor, undistorted, is where monotonicity is tested
            r_corner = 1.5 * np.hypot(*L0.ray(0.0, 0.0))
            while True:
                dk = rng.normal(0.0, B.DIST)
                L = Lens(L0.W, L0.H, f, f, cx, cy, L0.k1 + dk[0], L0.k2 + dk[1], L0.k3 + dk[2],
                         L0.p1 + dk[3], L0.p2 + dk[4])
                if _monotonic(L, r_corner):
                    break
            out.append(Camera.at(c.name, L, centre, R, **c.meta))
        # a channel leaks into its neighbours, never against them: no light
        # reads below dark (drawn signed, a 1 % green under a red read 2-5 %
        # once the sensor clipped the impossible negative light)
        colour = [np.eye(3) * (1.0 + rng.normal(0.0, B.COLOUR_GAIN, 3))
                  + (1.0 - np.eye(3)) * np.abs(rng.normal(0.0, B.COLOUR_MIX, (3, 3))) for _ in cams]
        light = 1.0 + rng.normal(0.0, B.LIGHT, len(cams))
        centre = rng.normal(0.0, B.CENTRE, 3)
        axis_o = rodrigues(np.array([0.0, *rng.normal(0.0, radians(B.AXIS_SKEW), 2)])) @ np.array([1.0, 0.0, 0.0])
        a, c_ = rng.normal(0.0, radians(B.AXIS_SKEW), 2)
        axis_i = rodrigues(np.array([a, 0.0, c_])) @ np.array([0.0, 1.0, 0.0])
        off = rng.normal(0.0, B.AXIS_OFFSET, 3)
        off -= axis_i * (off @ axis_i)                   # an offset along the axis is no offset
        zero = rng.normal(0.0, B.ENCODER_ZERO, 2)
        markers = rig.markers + rng.normal(0.0, B.MARKER, rig.markers.shape)
        bar = rig.bar + rng.normal(0.0, B.MARKER, rig.bar.shape)
        cert = bar + rng.normal(0.0, CalBar.CERT, bar.shape)
        stem = 1.0 + rng.normal(0.0, B.STEM)
        return cls(out, colour, light, centre, axis_o, axis_i, off, zero, markers, bar, cert, stem)

    @classmethod
    def drawing(cls, rig, cams):
        """The build as drawn: no errors anywhere."""
        n = len(cams)
        return cls(list(cams), [np.eye(3)] * n, np.ones(n), np.zeros(3), np.array([1.0, 0.0, 0.0]),
                   np.array([0.0, 1.0, 0.0]), np.zeros(3), np.zeros(2), rig.markers.copy(), rig.bar.copy(),
                   rig.bar.copy())


# ============================================================ THE CANVAS
def canvas(cams):
    """(W, H, [f]): the one image size every camera's ideal pinhole picture
    is rendered at, and each camera's focal length in it, so that warping
    it through the camera's own lens fills the whole sensor at no coarser
    than the sensor's own pixels: f is the camera's larger focal length,
    and the picture reaches every ray the sensor's edge sees."""
    W = H = 0
    fs = []
    for c in cams:
        L = c.lens
        f = max(L.fx, L.fy)
        u = np.concatenate([np.linspace(0, L.W, 64), np.full(64, L.W), np.linspace(0, L.W, 64), np.zeros(64)])
        v = np.concatenate([np.zeros(64), np.linspace(0, L.H, 64), np.full(64, L.H), np.linspace(0, L.H, 64)])
        x, y = L.ray(u, v)
        W = max(W, 2 * int(ceil(f * np.max(np.abs(x)))) + 4)
        H = max(H, 2 * int(ceil(f * np.max(np.abs(y)))) + 4)
        fs.append(f)
    return W, H, fs


# ============================================================ THE SCENE
def bat_geoms(rig):
    """The bat at calibration, in the bat frame: every part's own geoms,
    for pictures and the model camera, and boxes enclosing it slab by slab
    along its axis, for the solver's rays."""
    d = rig.d
    parts = dict(RS.cal_parts(d))
    parts.update(bat_mjcf.band_parts(d))
    g = []
    from .colour import decode
    for name in bat_mjcf.ORDER:
        if name in parts:
            lin = tuple(decode(parts[name].rgba[:3])) + tuple(parts[name].rgba[3:])
            g += [s.replace("<geom ", '<geom group="%d" ' % BAT, 1)
                  for s in bat_mjcf.part_geoms(name, parts[name], rgba=lin)]
    proxy = []
    n = 12
    xs = np.linspace(rig.x0, rig.x1, n + 1)
    for k in range(n):
        ext = []
        for p in parts.values():
            q = bat_mjcf.cut_part(p, x_slab=(xs[k], xs[k + 1]))
            if q.prisms or q.rods or q.boxes or q.points:
                ext.append(RS.part_extent(q))
        if not ext:
            continue
        y0, y1 = min(e[0] for e in ext), max(e[1] for e in ext)
        z0, z1 = min(e[2] for e in ext), max(e[3] for e in ext)
        proxy.append(_box("bat_proxy%d" % k, ((xs[k] + xs[k + 1]) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0),
                          ((xs[k + 1] - xs[k]) / 2.0, (y1 - y0) / 2.0, (z1 - z0) / 2.0), "1 0 0 0", PROXY))
    return g, proxy


# ============================================================ THE GIMBAL
def _kgm3(g_mm3):
    """kg/m^3 from spec's g/mm^3."""
    return g_mm3 * 1.0e6


def _printed(half):
    """kg/m^3 of a printed block of half-sizes `half` (mm): its walls solid,
    its inside at the slicer's infill (spec.Print)."""
    full = 8.0 * half[0] * half[1] * half[2]
    core = np.prod([max(2.0 * h - 2.0 * Print.WALL, 0.0) for h in half])
    return _kgm3(Print.RHO) * (full - core * (1.0 - Print.INFILL)) / full


def _weigh(line, rho=None, mass=None):
    """A geom line with its density (kg/m^3) or its mass (kg) written in."""
    if rho is not None:
        return line.replace("<geom ", '<geom density="%.9g" ' % rho, 1)
    return line.replace("<geom ", '<geom mass="%.9g" ' % mass, 1)


def gimbal_xml(rig, b, payload="bat", weighed=False, inner_extra=()):
    """[lines]: the gimbal's moving bodies, the outer member on its hinge
    and the inner one on its own, the payload in the clamp, as built (b).

    weighed: every geom given its material's density -- the rings square
    aluminium tube (spec.Gimbal.WALL), the hubs aluminium, the stems and
    the live centre steel, the balls their moulded cores, the clamp a
    printed block, the bat its parts' own masses and the solver's proxy
    boxes none -- and the inner axis' motor on the outer ring at one inner
    hub with its counterweight at the other (spec.Gimbal).  Not in the
    rendered scene: the cameras were placed against the gimbal as M2 drew
    it, without that motor [VERIFY: the motor's mount, then re-place].
    `inner_extra`: lines added to the inner body (sites)."""
    if weighed and payload not in ("bat", "none"):
        raise ValueError("only the bat, or nothing, is weighed in the clamp")
    S = Gimbal.SECTION
    r = rig.ball_r
    tube = _kgm3(Material.ALU) * (S * S - (S - 2.0 * Gimbal.WALL) ** 2) / (S * S)
    W = (lambda line, **k: _weigh(line, **k)) if weighed else (lambda line, **k: line)
    c0 = b.centre
    outer = ['<body name="outer" pos="%s">' % _v(c0),
             '  <joint name="alpha" type="hinge" axis="%.9f %.9f %.9f" pos="0 0 0"/>' % tuple(b.axis_o)]
    outer += ["  " + W(g, rho=tube) for g in _ring("ring_o", rig.R_o, S, C_RING)]
    for s in (-1.0, 1.0):
        outer.append("  " + W(_cyl("hub_o%+d" % int(s), (s * (rig.R_o + S / 2.0), 0.0, 0.0),
                                   (s * (rig.post_x - Gimbal.POST / 2.0), 0.0, 0.0), Gimbal.HUB[0] / 2.0, C_FRAME),
                              rho=_kgm3(Material.ALU)))
        outer.append("  " + W(_cyl("hub_i%+d" % int(s), (0.0, s * (rig.R_i + S / 2.0), 0.0),
                                   (0.0, s * (rig.R_o - S / 2.0), 0.0), Gimbal.HUB[0] / 2.0, C_FRAME),
                              rho=_kgm3(Material.ALU)))
    if weighed:
        y = rig.R_o + S / 2.0 + Stepper.BODY_L / 2.0
        half = (Stepper.FRAME / 2.0, Stepper.BODY_L / 2.0, Stepper.FRAME / 2.0)
        for name, s in (("motor_i", 1.0), ("counter_i", -1.0)):
            outer.append("  " + _weigh(_box(name, (0.0, s * y, 0.0), half, C_CLAMP), mass=Stepper.MASS / 1000.0))
    # the inner member: ring, markers, clamp, card, live centre, payload
    inner = ['<body name="inner" pos="%s">' % _v(b.offset_i),
             '  <joint name="beta" type="hinge" axis="%.9f %.9f %.9f" pos="0 0 0"/>' % tuple(b.axis_i)]
    inner += ["  " + e for e in inner_extra]
    inner += ["  " + W(g, rho=tube) for g in _ring("ring_i", rig.R_i, S, C_RING)]
    for k, p in enumerate(b.markers):
        inner.append("  " + W(_stem("stem%d" % k, p, rig.R_i, S, r), rho=_kgm3(Material.STEEL)))
        inner.append("  " + W(_ball("marker%d" % k, p, r), rho=_kgm3(Material.BALL)))
    for bx in rig.clamp:
        inner.append("  " + W(_box(bx.name, bx.c, bx.half, C_CLAMP), rho=_printed(bx.half)))
    for name, c, n, rgb in rig.card:
        half = np.where(np.abs(n) > 0.5, 0.25, Clamp.CARD_PATCH / 2.0)      # thin along its normal
        cc = np.asarray(c, float) + 0.25 * np.asarray(n, float)
        inner.append("  " + W(_box(name, cc, half, "%.4f %.4f %.4f 1" % tuple(rgb)), rho=_kgm3(Print.RHO)))
    tip = rig.x1 - rig.x_mid
    inner.append("  " + W(_cyl("live_centre", (tip, 0.0, 0.0), (rig.R_i - S / 2.0, 0.0, 0.0), Clamp.CENTRE_D / 2.0,
                               C_CLAMP), rho=_kgm3(Material.STEEL)))
    if payload in ("bat", "proxy"):
        g, proxy = bat_geoms(rig)
        if weighed:
            proxy = [_weigh(p, mass=0.0) for p in proxy]
        inner.append('  <body name="bat" pos="%s">' % _v((-rig.x_mid, 0.0, 0.0)))
        inner += ["    " + s for s in (g if payload == "bat" else []) + proxy]
        inner.append("  </body>")
    elif payload == "bar":
        x0 = rig.x0 - rig.x_mid
        inner.append("  " + _cyl("bar_tube", (x0, 0.0, 0.0), (tip, 0.0, 0.0), CalBar.TUBE_D / 2.0, C_BAR))
        for k, p in enumerate(b.bar):
            rho = np.hypot(p[1], p[2])
            u = np.array([0.0, p[1] / rho, p[2] / rho])
            foot = _bar_foot(p)
            inner.append("  " + _cyl("bar_stem%d" % k, foot, np.asarray(p) - u * r, Marker.STEM_D / 2.0, C_FRAME,
                                     group=STEM))
            inner.append("  " + _ball("bar%d" % k, p, r))
    inner.append("</body>")
    outer += ["  " + s for s in inner]
    outer.append("</body>")
    return outer


def scene(rig, site, cams, b=None, payload="bat", size=None, overview=True):
    """(xml, info) of the rig in its station, built as b says (as drawn if
    b is None), the bat or the certified bar ("bar") in the clamp, only the
    boxes enclosing the bat ("proxy": the placement solver's), or nothing
    ("none").  `cams` are the cameras as drawn; their as-built
    poses come from b.  `size` is the (W, H) every camera renders at -- the
    canvas (vision.py) -- and the offscreen buffer is made that big."""
    b = RigBuildDraw.drawing(rig, cams) if b is None else b
    built = b.cams
    if size is None:
        Wc, Hc, fs = canvas(built)
    else:
        Wc, Hc = size
        fs = [max(c.lens.fx, c.lens.fy) for c in built]
    world = []
    # the floor, the gantry parked at home over the gimbal, the posts
    world.append('<geom name="floor" type="plane" pos="0 0 %.6f" size="%.3f %.3f 0.01" rgba="%s" group="%d"/>'
                 % (mm(-rig.h_c), mm(3.0 * rig.standoff), mm(3.0 * rig.standoff), C_FLOOR, STRUCT))
    for o in site.occluders:
        if o.name.startswith("gpost"):
            continue
        world.append(_box("gantry_" + o.name, o.c, o.half, C_FRAME))
    for s in (-1.0, 1.0):
        x = s * rig.post_x
        top = Gimbal.HUB[0] / 2.0
        world.append(_box("post%+d" % int(s), (x, 0.0, (-rig.h_c + top) / 2.0),
                          (Gimbal.POST / 2.0, Gimbal.POST / 2.0, (rig.h_c + top) / 2.0), C_FRAME))
    world += gimbal_xml(rig, b, payload)
    # the cameras, as built, each with its ring light
    for k, (c, f) in enumerate(zip(built, fs)):
        fovy = 2.0 * degrees(atan((Hc / 2.0) / f))
        xa, ya = c.R[0], -c.R[1]
        world.append('<camera name="%s" pos="%s" xyaxes="%s" fovy="%.9f"/>'
                     % (c.name, _v(c.centre), " ".join("%.9f" % v for v in list(xa) + list(ya)), fovy))
        half_diag = degrees(atan(np.hypot(Wc, Hc) / 2.0 / f))
        world.append('<light name="light%d" pos="%s" dir="%.9f %.9f %.9f" cutoff="%.3f" exponent="0" '
                     'attenuation="0 0 1" '
                     'diffuse="%.4f %.4f %.4f" specular="0 0 0" ambient="0 0 0" castshadow="false"/>'
                     % ((k, _v(c.centre)) + tuple(c.axis) + (min(half_diag + 5.0, 89.0),) + (b.light[k],) * 3))
    if overview:
        eye = np.array([-1.2, -1.6, 1.0]) / np.linalg.norm([-1.2, -1.6, 1.0]) * 2.2 * rig.standoff
        from .lens import look_at
        Rv = look_at(eye, (0.0, 0.0, 0.0))
        world.append('<camera name="overview" pos="%s" xyaxes="%s" fovy="40"/>'
                     % (_v(eye), " ".join("%.9f" % v for v in list(Rv[0]) + list(-Rv[1]))))
    zfar = 4.0 * rig.standoff / 1000.0 + 2.0
    xml = """<mujoco model="rig">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option gravity="0 0 0"/>
  <visual>
    <headlight active="0" ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/>
    <map znear="0.002" zfar="%.3f"/>
    <quality shadowsize="0"/>
    <global offwidth="%d" offheight="%d"/>
  </visual>
  <asset>
    <material name="matte" specular="0" shininess="0" reflectance="0"/>
    <material name="retro" rgba="1 1 1 1" emission="1" specular="0" shininess="0"/>
  </asset>
  <default>
    <geom material="matte" contype="0" conaffinity="0" density="%g"/>
  </default>
  <worldbody>
%s
  </worldbody>
</mujoco>
""" % (zfar, Wc, Hc, DEFAULT_DENSITY, "\n".join("    " + w for w in world))
    info = {"canvas": (Wc, Hc), "f_canvas": fs, "cams": [c.name for c in built],
            "markers": ["marker%d" % k for k in range(len(b.markers))],
            "bar": ["bar%d" % k for k in range(len(b.bar))] if payload == "bar" else []}
    return xml, info


# ============================================================ TRUTH
def set_gimbal(m, d, a, b_, build):
    """Command the gimbal to (a, b) deg: the hinges go to command + their
    encoder zero, and the kinematics run.  The scene's truth, not the
    controller's belief -- the controller has only (a, b)."""
    import mujoco
    d.qpos[m.joint("alpha").qposadr[0]] = radians(a + build.zero[0])
    d.qpos[m.joint("beta").qposadr[0]] = radians(b_ + build.zero[1])
    mujoco.mj_kinematics(m, d)
    mujoco.mj_camlight(m, d)


def body_pose(m, d):
    """(R, t mm): the inner body's pose in the rig frame, truth."""
    bid = m.body("inner").id
    return d.xmat[bid].reshape(3, 3).copy(), d.xpos[bid] * 1000.0


def stem_ids(m, n_markers, n_bar=0):
    """(n_markers + n_bar,) the geom id of each ball's own stem, the ring's
    markers then the bar's balls (-1 where the scene has none)."""
    import mujoco
    names = ["stem%d" % k for k in range(n_markers)] + ["bar_stem%d" % k for k in range(n_bar)]
    return np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names], int)


def balls_world(m, d, names):
    """(N, 3) mm: where named balls are, truth."""
    return np.array([d.geom_xpos[m.geom(n).id] * 1000.0 for n in names])
