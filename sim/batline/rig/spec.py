"""The calibration station's rig, derived from the bat and the hardware.

Station 4 turns the bat through every orientation in a two-axis gimbal
while eight cameras round it measure the clamp's pose in every frame
(plan, milestone M2).  As in station/spec.py nothing here is typed: the
gimbal is as big as the bat at calibration and the facts make it; the
cameras stand as close as a catalogue lens lets one of them hold the whole
gimbal and still see the far side's balls; where they may stand is
wherever the gantry, the floor, the aisle and the gimbal's sweep are not.

    design(d)      the gimbal (rings, clamp, live centre, the markers, its
                   height), the ball size, the lens and the standoff, and
                   the certified bar.  Closed form and fast: line.py sizes
                   station 4 from it
    site(rig, st, dg, mod)   the rig placed in its station: where a camera
                   may not stand (keep-outs) and what stands between a
                   camera and the gimbal (the gantry parked, the posts)
    grid(n, offset)  gimbal angles over both axes' whole turn

FRAMES.  The rig frame has its origin at the gimbal's centre as drawn, x
along the outer axis -- the aisle, and the bat's axis when the gimbal is
parked -- y into the room, z up: the station frame, shifted.  The body
frame is the inner ring's: the bat's own frame (x from the pommel, z
toward the blade's back) moved so the bat's middle is at the origin.  As
drawn, the body's pose at gimbal angles (a, b) is R = Rx(a) Ry(b), t = 0:
the outer axis is x, the inner is the outer ring's y.  The gimbal as built
is not that, which is the point -- rig/mjcf.py builds it as built, and
only the cameras say where the body is.

WHY THE MARKERS ARE ON THE RING.  The bat is the thing calibrated; it
cannot carry the targets that measure it.  The inner ring is the stiffest,
widest thing that turns with it: 0.1 mm of noise on balls 0.5 m apart is
0.01 degrees, the order the budget asks for (README, "the rig's budget").
The clamp fixes the bat to the ring; how the bat sits in the clamp is
M3's to measure from the bands.
"""
from dataclasses import dataclass, field
from math import sqrt, pi, sin, cos, atan, asin, radians, degrees

import numpy as np

from ..spec import RingCam, Marker, Gimbal, Clamp, CalBar, RigBuild, Rig, Module
from ..product import Fit, part_extent
from .lens import Lens

# fitted at pack, after calibration (line.py, station 5); the cells are the
# customer's
LATER = ("pommel_cap", "spring")
GOLDEN = radians(180.0 * (3.0 - sqrt(5.0)))     # the golden angle: N stems spread evenly round a tube


def cal_parts(d):
    """The bat's parts as it comes to calibration."""
    return {k: p for k, p in d.parts.items() if k not in LATER and p.source != "customer"}


def bat_envelope(d):
    """(x0, x1, r): the bat at calibration's extent along its axis and its
    farthest point from the axis, its printed bands included."""
    from ..mjcf import band_parts
    parts = dict(cal_parts(d))
    parts.update(band_parts(d))
    ext = [part_extent(p) for p in parts.values()]
    return min(e[5] for e in ext), max(e[6] for e in ext), max(e[4] for e in ext)


def clear():
    """mm the gimbal's moving parts keep from each other and from anything
    standing: Module.Z_SIGMAS of where the build may have put them -- the
    centre, the inner axis against the outer, a marker on its stem."""
    return Module.Z_SIGMAS * sqrt(RigBuild.CENTRE ** 2 + RigBuild.AXIS_OFFSET ** 2 + RigBuild.MARKER ** 2)


def cam_radius():
    """mm, a camera's bounding sphere about its lens: the ring light across,
    the board's depth behind."""
    return 0.5 * sqrt(max(RingCam.BODY[0], RingCam.LIGHT_D) ** 2 + RingCam.BODY[2] ** 2)


def lens_drawn(f_mm, k1, k2):
    """A catalogue lens as drawn on the sensor: centred, square pixels."""
    f = f_mm / RingCam.PIXEL
    return Lens(RingCam.W, RingCam.H, f, f, RingCam.W / 2.0, RingCam.H / 2.0, k1, k2)


# ================================================================ DESIGN
@dataclass
class Box:
    """An axis-aligned box: centre and half sizes, mm."""
    name: str
    c: tuple
    half: tuple


@dataclass
class RigDesign:
    d: object                  # the bat (BatDesign)
    ball_d: float              # mm, every marker's and the bar's balls
    lens: tuple                # (focal mm, k1, k2), from RingCam.LENSES
    standoff: float            # mm, the gimbal's centre to each camera's lens
    x0: float                  # the bat's extent along its axis, bat frame
    x1: float
    x_mid: float               # bat frame x of the body's origin
    r_bat: float               # mm, the bat's farthest point from its axis
    R_i: float                 # mm, the inner ring's centreline
    R_o: float                 # mm, the outer ring's
    R_inner: float             # mm, the inner member's swept radius: a ball
    R_s: float                 # mm, the whole gimbal's
    clear: float
    h_c: float                 # mm, the gimbal's centre above the floor datum
    post_x: float              # mm, each post's centre from the centre, along x
    centre_l: float            # mm, the live centre's reach from the ring to the tip
    markers: np.ndarray        # (K, 3) body frame, the inner ring's balls as drawn
    bar: np.ndarray            # (N, 3) body frame, the certified bar's balls
    clamp: list                # [Box] body frame
    card: list                 # [(name, centre, normal, rgb)] body frame: the colour card's patches
    view_r: float              # mm, the sphere every camera must hold whole
    fits: list = field(default_factory=list)

    @property
    def ball_r(self):
        return self.ball_d / 2.0

    def lens_drawn(self):
        return lens_drawn(*self.lens)

    def bat_to_body(self, p):
        return np.asarray(p, float) - np.array([self.x_mid, 0.0, 0.0])

    def footprint(self):
        """(along x, across y) mm: the gimbal on its posts, from above."""
        return (2.0 * (self.post_x + Gimbal.POST / 2.0), 2.0 * (self.R_s + self.clear))

    def reach(self):
        """mm round the gimbal's centre a camera may stand at, its body
        included: what the station's footprint must take in."""
        return self.standoff + cam_radius()

    def work(self):
        """((along, across), top): the bat lying in the parked gimbal's
        clamp -- the only part of the fixture the station's head reaches --
        and its underside's height above the floor datum."""
        return (self.x1 - self.x0, 2.0 * self.r_bat), self.h_c - self.r_bat



def _marker_signs(theta):
    """Which side of the ring's plane each ball stands: half each side,
    and no half-turn of the ring about any of its three axes maps the set
    onto itself, so a pose is never ambiguous.  The first such pattern in
    counting order."""
    K = len(theta)
    th = np.radians(theta)

    def same(a, b):
        return abs((a - b + pi) % (2.0 * pi) - pi) < 1e-9

    turns = (lambda t, s: (t + pi, s),          # about z
             lambda t, s: (-t, -s),             # about x
             lambda t, s: (pi - t, -s))         # about y
    for code in range(2 ** K):
        s = np.array([1.0 if (code >> k) & 1 else -1.0 for k in range(K)])
        if s.sum() != 0.0:
            continue
        symmetric = False
        for turn in turns:
            if all(any(same(turn(th[i], s[i])[0], th[j]) and turn(th[i], s[i])[1] == s[j] for j in range(K))
                   for i in range(K)):
                symmetric = True
                break
        if not symmetric:
            return s
    raise ValueError("no asymmetric pattern of %d balls" % K)


def _gimbal(d, D):
    """The gimbal for this bat, with balls of diameter D."""
    lay = d.lay
    x0, x1, r_bat = bat_envelope(d)
    x_mid = (x0 + x1) / 2.0
    S = Gimbal.SECTION
    jaw_r = lay.r_grip + Clamp.JAW_T
    # the inner ring holds the clamp's back on one side and the live centre
    # on the other; the clamp side is the longer, so it sets the ring
    back = x0 - Clamp.STOP_T - Clamp.BACK - x_mid
    R_i = -back + S / 2.0
    centre_l = R_i - S / 2.0 - (x1 - x_mid)
    clamp = [Box("clamp_back", ((back + x0 - Clamp.STOP_T - x_mid) / 2.0, 0.0, 0.0),
                 (Clamp.BACK / 2.0, jaw_r, jaw_r)),
             Box("clamp_stop", (x0 - Clamp.STOP_T / 2.0 - x_mid, 0.0, 0.0), (Clamp.STOP_T / 2.0, jaw_r, jaw_r)),
             Box("clamp_jaws", ((x0 + lay.x_grip0 + Clamp.JAW_L) / 2.0 - x_mid, 0.0, 0.0),
                 ((lay.x_grip0 + Clamp.JAW_L - x0) / 2.0, jaw_r, jaw_r))]
    # the colour card: its patches in a row along the clamp's flanks and
    # its underside.  The gimbal turns a flank's normal only about the
    # outer axis, so no turn faces a flank down the aisle; it can face the
    # underside anywhere
    c0, c1 = back, lay.x_grip0 + Clamp.JAW_L - x_mid
    n = len(Clamp.CARD_RGB)
    card = []
    for tag, nrm in (("p", (0.0, 1.0, 0.0)), ("n", (0.0, -1.0, 0.0)), ("d", (0.0, 0.0, -1.0))):
        for i, rgb in enumerate(Clamp.CARD_RGB):
            x = (c0 + c1) / 2.0 + (i - (n - 1) / 2.0) * Clamp.CARD_PATCH
            card.append(("card%s%d" % (tag, i), (x, nrm[1] * jaw_r, nrm[2] * jaw_r), nrm, rgb))
    # the markers: PER_GAP in each gap between the ring's four fixtures (the
    # live centre at 0 deg, a bearing at 90, the clamp at 180, a bearing at
    # 270), each on a stem leaving the tube's outer corner at 45 degrees
    theta = [90.0 * g + 90.0 * (j + 1) / (Rig.PER_GAP + 1) for g in range(4) for j in range(Rig.PER_GAP)]
    signs = _marker_signs(theta)
    a = S / 2.0 + (Rig.STANDOFF * D + D / 2.0) / sqrt(2.0)
    rho = R_i + a
    markers = np.array([(rho * cos(radians(t)), rho * sin(radians(t)), s * a) for t, s in zip(theta, signs)])
    # what the inner member sweeps as both axes turn: a ball
    corner = lambda b: max(np.linalg.norm(np.array(b.c) + np.array(sx) * np.array(b.half))
                           for sx in [(i, j, k) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
    R_inner = max([float(np.max(np.linalg.norm(markers, axis=1))) + D / 2.0,
                   sqrt((R_i + S / 2.0) ** 2 + (S / 2.0) ** 2),
                   sqrt(max(x_mid - x0, x1 - x_mid) ** 2 + r_bat ** 2)]
                  + [corner(b) for b in clamp])
    c = clear()
    R_o = R_inner + c + S / 2.0
    R_s = sqrt((R_o + S / 2.0) ** 2 + (S / 2.0) ** 2)
    h_c = R_s + c
    post_x = R_o + S / 2.0 + Gimbal.HUB[1] + Gimbal.POST / 2.0
    # the certified bar: between the clamp's jaws and the tip, its balls
    # spread along it and round it
    xa = lay.x_grip0 + Clamp.JAW_L - x_mid + D
    xb = x1 - x_mid - D
    rb = CalBar.TUBE_D / 2.0 + Rig.STANDOFF * D + D / 2.0
    N = CalBar.BALLS
    bar = np.array([(xa + (xb - xa) * k / (N - 1), rb * cos(k * GOLDEN), rb * sin(k * GOLDEN)) for k in range(N)])
    view_r = max(R_inner, float(np.max(np.linalg.norm(bar, axis=1))) + D / 2.0) \
        + Module.Z_SIGMAS * sqrt(RigBuild.CENTRE ** 2 + RigBuild.CAM_POS ** 2)

    fits = []
    fits.append(Fit("the clamp's jaws hold the grip, short of the shoulder",
                    lay.x_shoulder - (lay.x_grip0 + Clamp.JAW_L), "jaws to x %.1f" % (lay.x_grip0 + Clamp.JAW_L)))
    fits.append(Fit("the pogo block in the jaws lands on the window",
                    min(lay.window[0] - x0, lay.x_grip0 + Clamp.JAW_L - lay.window[1]),
                    "window x %.2f..%.2f under jaws x %.2f..%.2f" % (lay.window[0], lay.window[1], x0,
                                                                     lay.x_grip0 + Clamp.JAW_L)))
    fits.append(Fit("the live centre reaches the tip from the ring", centre_l, "%.1f mm long" % centre_l))
    fits.append(Fit("the colour card fits along the clamp", (c1 - c0) - n * Clamp.CARD_PATCH,
                    "%d patches of %.0f mm on %.1f mm" % (n, Clamp.CARD_PATCH, c1 - c0)))
    fits.append(Fit("the outer ring clears the inner member's sweep by %.2f mm" % c,
                    (R_o - S / 2.0) - R_inner - c, "inner sweep %.1f, outer ring's inside %.1f" % (R_inner, R_o - S / 2.0)))
    fits.append(Fit("the posts stand clear of the gimbal's sweep",
                    post_x - Gimbal.POST / 2.0 - R_s - c, "posts' faces at +-%.1f, sweep %.1f"
                    % (post_x - Gimbal.POST / 2.0, R_s)))
    return dict(x0=x0, x1=x1, x_mid=x_mid, r_bat=r_bat, R_i=R_i, R_o=R_o, R_inner=R_inner, R_s=R_s, clear=c,
                h_c=h_c, post_x=post_x, centre_l=centre_l, markers=markers, bar=bar, clamp=clamp, card=card,
                view_r=view_r, fits=fits)


def _magnification(L, r_max):
    """The smallest of a lens's radial and tangential magnification (its
    distorted over its undistorted image) out to undistorted radius r_max:
    where barrel distortion shrinks a ball's image most."""
    r = np.linspace(0.0, r_max, 200)
    r2 = r * r
    tang = 1.0 + L.k1 * r2 + L.k2 * r2 * r2 + L.k3 * r2 ** 3
    rad = 1.0 + 3.0 * L.k1 * r2 + 5.0 * L.k2 * r2 * r2 + 7.0 * L.k3 * r2 ** 3
    return float(min(tang.min(), rad.min()))


def _optics(g, D, lens):
    """The standoff this lens needs and what it gives: (standoff, disc px,
    fits).  A camera aimed at the gimbal's centre must hold the whole view
    sphere in its narrow (vertical) field, after the principal point and
    the aim are off by Module.Z_SIGMAS of the build, so it stands where
    that sphere is tangent to the field; and from there the farthest ball
    must still image MIN_PX across where the lens shrinks images most."""
    L = lens_drawn(*lens)
    edge = Module.Z_SIGMAS * RigBuild.CENTRE_PX
    _, y = L.ray(L.cx, edge)
    half = atan(abs(float(np.atleast_1d(y)[0]))) - radians(Module.Z_SIGMAS * RigBuild.CAM_TILT)
    d_fov = g["view_r"] / sin(half) if half > 0.0 else float("inf")
    d = max(d_fov, g["R_s"] + g["clear"] + cam_radius())
    if not np.isfinite(d):
        return d, 0.0, [Fit("the lens's field outlasts the build's aim", degrees(half))]
    far = float(np.max(np.linalg.norm(g["markers"], axis=1)))
    mag = _magnification(L, np.tan(asin(min(g["view_r"] / d, 1.0))))
    px = L.fx * mag * D / (d + far)
    fits = [Fit("one camera holds the whole gimbal in its narrow field from %.0f mm (deg)" % d,
                degrees(half - asin(g["view_r"] / d)),
                "%.1f mm lens, half-field %.2f deg after the build" % (lens[0], degrees(half))),
            Fit("the far side's %.1f mm balls image %.0f px across" % (D, RingCam.MIN_PX),
                px - RingCam.MIN_PX, "%.1f px at %.0f mm, the lens shrinking images to %.3f" % (px, d + far, mag)),
            Fit("the cameras stand clear of the gimbal's sweep", d - g["R_s"] - g["clear"] - cam_radius(),
                "lens at %.0f, sweep %.1f" % (d, g["R_s"]))]
    return d, px, fits


def design(d):
    """The rig for bat design d: of every ball on the menu and every lens in
    the catalogue that works, the one whose cameras stand closest -- the
    station's footprint along the aisle is twice that -- and, at the same
    standoff, the smallest ball."""
    best = None
    for D in Marker.D_MENU:
        g = _gimbal(d, D)
        for lens in RingCam.LENSES:
            sd, px, ofits = _optics(g, D, lens)
            if not all(f.ok for f in ofits):
                continue
            if best is None or sd < best[2] - 1e-9:
                best = (D, lens, sd, g, ofits)
    if best is None:
        raise ValueError("no ball on the menu and lens in the catalogue lets a camera see the whole gimbal "
                         "and its farthest ball at %.0f px" % RingCam.MIN_PX)
    D, lens, sd, g, ofits = best
    fits = list(g.pop("fits")) + ofits
    return RigDesign(d=d, ball_d=D, lens=lens, standoff=sd, fits=fits, **g)


# ============================================================ GIMBAL ANGLES
def grid(n, offset=0.0):
    """(a, b) deg: n x n gimbal angles over both axes' whole turn, shifted
    by `offset` of a step -- the solver places the cameras on one grid and
    the check holds them to the grid half a step over, which it never saw."""
    step = 360.0 / n
    a = -180.0 + (np.arange(n) + offset) * step
    A, B = np.meshgrid(a, a, indexing="ij")
    return np.stack([A.ravel(), B.ravel()], axis=1)


def rot_x(deg):
    t = radians(deg)
    return np.array([[1.0, 0.0, 0.0], [0.0, cos(t), -sin(t)], [0.0, sin(t), cos(t)]])


def rot_y(deg):
    t = radians(deg)
    return np.array([[cos(t), 0.0, sin(t)], [0.0, 1.0, 0.0], [-sin(t), 0.0, cos(t)]])


def body_R(a, b):
    """The body's rotation at gimbal angles (a, b) deg, as drawn."""
    return rot_x(a) @ rot_y(b)


# ================================================================== SITE
@dataclass
class Site:
    """The rig in its station: the gimbal's centre in the station frame,
    and in the rig frame where a camera may not stand and what stands
    between a camera and the gimbal."""
    centre: tuple              # (x, y, z) of the gimbal's centre, station frame
    keepout_boxes: list        # [Box] rig frame, already grown by a camera's radius
    keepout_r: float           # mm, no camera's lens nearer the centre than this
    floor_z: float             # rig frame: no lens below this
    bounds: tuple              # ((x lo, x hi), (y lo, y hi)) rig frame: the station's floor
    occluders: list            # [Box] rig frame: the gantry parked at home, its frame, the posts
    fits: list = field(default_factory=list)


def site(rig, st, dg, mod):
    """The rig placed in station `st` (line.py's calibration station, its
    module `mod` derived): the keep-outs and occluders, in the rig frame."""
    from ..station.spec import frame
    xy = (lambda u, v: (u, v)) if dg.along else (lambda u, v: (v, u))
    cx, cy = xy(*st.fixture)
    C = np.array([cx, cy, rig.h_c])
    rc = cam_radius()
    fr = frame(mod)
    A, H = mod.axes, mod.head
    b = fr["beam"]
    px0, px1 = fr["posts"]
    ry0, ry1 = fr["rails"]
    rail_top = fr["rail_z"] + b / 2.0

    def sbox(name, lo, hi, grow=0.0):
        lo, hi = np.asarray(lo, float) - C - grow, np.asarray(hi, float) - C + grow
        return Box(name, tuple((lo + hi) / 2.0), tuple((hi - lo) / 2.0))

    # the gantry: everywhere its head and bridge go, its rails and posts --
    # one box from the floor to the rails' top over the frame's plan
    keep = [sbox("the gantry and its travel", (px0 - b / 2.0, ry0 - b / 2.0, 0.0),
                 (px1 + b / 2.0, ry1 + b / 2.0, rail_top), rc)]
    # the gimbal's posts and bearings
    for s in (-1.0, 1.0):
        x = s * rig.post_x
        keep.append(Box("a gimbal post", (x, 0.0, -rig.h_c / 2.0 + Gimbal.HUB[0] / 4.0),
                        (Gimbal.POST / 2.0 + rc, Gimbal.POST / 2.0 + rc, rig.h_c / 2.0 + Gimbal.HUB[0] / 4.0 + rc)))
    W, Dp = st.footprint if dg.along else st.footprint[::-1]
    bounds = ((-cx + rc, W - cx - rc), (-cy + rc, Dp - cy - rc))

    # occluders: the frame, the bridge and the head parked at home, posts
    occ = []
    for i, yy in enumerate((ry0, ry1)):
        occ.append(sbox("xrail%d" % i, (px0, yy - b / 2.0, fr["rail_z"] - b / 2.0), (px1, yy + b / 2.0, rail_top)))
        for j, xx in enumerate((px0, px1)):
            occ.append(sbox("post%d%d" % (i, j), (xx - b / 2.0, yy - b / 2.0, 0.0), (xx + b / 2.0, yy + b / 2.0, rail_top)))
    hx, hy, hz = A["x"].home, A["y"].home, A["z"].home
    occ.append(sbox("bridge", (hx - b, ry0, fr["rail_z"] - b / 2.0), (hx + b, ry1, rail_top)))
    occ.append(sbox("head", (hx - H.plate[0] / 2.0, hy - H.plate[1] / 2.0, hz + H.z_extended),
                    (hx + H.plate[0] / 2.0, hy + H.plate[1] / 2.0 + H.camera[1], fr["rail_z"])))
    for s in (-1.0, 1.0):
        occ.append(Box("gpost%+d" % int(s), (s * rig.post_x, 0.0, -rig.h_c / 2.0),
                       (Gimbal.POST / 2.0, Gimbal.POST / 2.0, rig.h_c / 2.0)))

    fits = []
    lowest = A["z"].hi + H.z_retracted
    fits.append(Fit("the gantry's head at home clears the gimbal's sweep",
                    lowest - (rig.h_c + rig.R_s) - rig.clear, "retracted tips at %.1f, sweep's top %.1f"
                    % (lowest, rig.h_c + rig.R_s)))
    near = min(np.hypot(xx - cx, yy - cy) for xx in (px0, px1) for yy in (ry0, ry1)) - b / sqrt(2.0)
    fits.append(Fit("the gantry's posts stand clear of the gimbal's sweep", near - rig.R_s - rig.clear,
                    "nearest post %.1f from the centre" % near))
    fits.append(Fit("the gantry's rails pass over the gimbal's sweep",
                    fr["rail_z"] - b / 2.0 - (rig.h_c + rig.R_s) - rig.clear,
                    "rails at %.1f" % fr["rail_z"]))
    return Site(tuple(C), keep, rig.R_s + rig.clear + rc, -rig.h_c + rc, bounds, occ, fits)


def allowed(site, P):
    """Which lens positions P (N x 3, rig frame) a camera may stand at."""
    P = np.atleast_2d(np.asarray(P, float))
    ok = np.linalg.norm(P, axis=1) >= site.keepout_r
    ok &= P[:, 2] >= site.floor_z
    (x0, x1), (y0, y1) = site.bounds
    ok &= (P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)
    for b in site.keepout_boxes:
        inside = np.all(np.abs(P - np.asarray(b.c)) <= np.asarray(b.half), axis=1)
        ok &= ~inside
    return ok


def report(rig):
    out = []
    w = out.append
    w("calibration rig: %.1f mm balls, %.1f mm lens (k1 %.2f), cameras %.0f mm from the centre"
      % (rig.ball_d, rig.lens[0], rig.lens[1], rig.standoff))
    w("  bat x %.1f..%.1f (middle %.1f), reach %.1f from its axis" % (rig.x0, rig.x1, rig.x_mid, rig.r_bat))
    w("  rings: inner %.1f, outer %.1f; sweeps: inner %.1f, whole %.1f; centre %.1f above the floor; "
      "posts at +-%.1f" % (rig.R_i, rig.R_o, rig.R_inner, rig.R_s, rig.h_c, rig.post_x))
    w("  %d markers, %d bar balls; view sphere %.1f" % (len(rig.markers), len(rig.bar), rig.view_r))
    for f in rig.fits:
        w("    %s %-70s %9.3f  %s" % ("ok  " if f.ok else "FAIL", f.name[:70], f.margin, f.detail))
    return "\n".join(out)
