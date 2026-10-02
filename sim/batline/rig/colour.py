"""The bands' colours, read back by the rig's cameras.

The phone finds the bat by its two printed bands, so the line stores each
band's colour as printed, and refuses a bat whose bands are not the colours
the design gave them (spec.Rules.BAND_RGB, judged by Rig.DE_MAX).

ONE CONVENTION.  A designer's colours (the bands, the sleeve) are sRGB,
gamma-encoded; what a print of one reflects is its decoded, linear value
(decode), and a camera's raw reading is linear in that.  The card's patches
are certified as linear reflectances (spec.Clamp.CARD_RGB).  So everything
here is linear until the end, where a band's colour is encoded back to sRGB
to be stored and compared with the design (CIE76 in CIELAB, D65).

A camera's raw reading of a surface is its reflectance times the light on
it -- the ring light's output, falling as the inverse square of the
distance (vision.expose_mm) -- times the cosine of the light's angle,
through the sensor's own colour response.  Three things take those apart:

  THE CARD.  Patches of known reflectance in a row on each of the clamp's
  flanks and its underside (spec.Clamp: white, red, green, blue), each row
  flat.  The gimbal stops where each camera can read a row whole and
  squarely (card_poses).  Each patch's reading over its geometry (the
  cosine and the fall-off, from the tracked pose) fits A = s M: the
  sensor's response M (each channel's gain and its mixing) times the light
  s, by weighted least squares over the patches' three channels, each
  weighted by the sensor's own noise (spec.RingCam) -- four patches for
  nine unknowns, over every pose the card faces the camera at.  What the
  fit leaves is tested against that noise (CardFit.chi2).

  THE SLEEVE.  A^-1 of a sleeve reading over its geometry is the sleeve's
  reflectance, measured, not assumed: a white knit a little darker than
  drawn would otherwise darken every band read against it.

  THE SLEEVE BESIDE THE BAND.  A band and the sleeve next to it on the
  same flat face share the face's normal, the light and the camera, so
  A^-1 of the band's reading over A^-1 of the sleeve's, channel by
  channel, each sample over its own geometry, times the sleeve's measured
  reflectance, is the band's reflectance -- whatever is left of s, the
  cosine and the renderer's shading.

Samples stay EDGE_PX from every edge in the image: a face's, a band's.
Each camera reads each face it sees squarely enough to keep that margin
on a face's middle 70 % (FACE_USE); the reads are averaged, weighted by
their samples.  Whether a camera sees a sample is cast in the rig as
drawn, carried to the clamp's tracked pose (carry), with a guard band in
the surface's own plane round it.
"""
from dataclasses import dataclass, field
from math import pi

import numpy as np

from ..spec import Clamp, Rules, Module
from .. import product as Pd
from ..mjcf import band_parts
from . import vision as V

EDGE_PX = 3.0                  # px a sample keeps from an edge: the bilinear read's 1, the render's
                               # antialiasing's 1, and 1 for the pose and the calibration
FACE_USE = 0.7                 # of a face's width sampled: its flat middle, clear of the rounded corners
N_T, N_X = 5, 3                # samples across a face and along a band (or its sleeve)


def faces(R, rc):
    """[(p0 (2,), p1 (2,), n (2,))] the rounded triangle's three flat faces
    in (y, z), each from end to end, with its outward normal (as
    batline/mjcf._outline draws them)."""
    Vt = [np.asarray(v) for v in Pd._tri_vertices(R)]
    if np.cross(Vt[1] - Vt[0], Vt[2] - Vt[0]) < 0:
        Vt = Vt[::-1]
    out = []
    for i in range(3):
        a, b = Vt[i], Vt[(i + 1) % 3]
        n = np.array([(b - a)[1], -(b - a)[0]]) / np.linalg.norm(b - a)
        out.append((a + rc * n, b + rc * n, n))
    return out


def band_geometry(rig):
    """[(name, x0, x1, faces of the band's outer surface, faces of the
    sleeve's, s0, s1)], x in the body frame (the bat's, shifted to the
    gimbal's centre), the faces in the section's (y, z); s0..s1 the run of
    the sleeve the band is printed on, where nothing stands proud of it."""
    out = []
    for name, part in band_parts(rig.d).items():
        x0, x1, sec = part.prisms[0]
        R, rc = sec.solid.p
        _, rs = sec.holes[0].p
        # the sleeve's run the band lies on: where its outer surface is the band's inner one
        runs = [(a, b) for a, b, ss in rig.d.parts["sleeve"].prisms if ss.solid.p == (R, rs)]
        if not runs:
            raise ValueError("%s lies on no run of the sleeve" % name)
        s0, s1 = runs[0]
        out.append((name, x0 - rig.x_mid, x1 - rig.x_mid, faces(R, rc), faces(R, rs),
                    s0 - rig.x_mid, s1 - rig.x_mid))
    return out


def decode(rgb):
    """sRGB (0..1, gamma-encoded, as a designer gives a colour) to linear
    (IEC 61966-2-1)."""
    c = np.clip(np.asarray(rgb, float), 0.0, None)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def encode(lin):
    """Linear to sRGB, decode's inverse."""
    c = np.clip(np.asarray(lin, float), 0.0, None)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1.0 / 2.4) - 0.055)


def linear_to_lab(lin):
    """CIELAB (D65) of linear sRGB-primaries colours (..., 3)."""
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = np.clip(np.asarray(lin, float), 0.0, None) @ M.T / np.array([0.95047, 1.0, 1.08883])
    e = 216.0 / 24389.0
    f = np.where(xyz > e, np.cbrt(xyz), (24389.0 / 27.0 * xyz + 16.0) / 116.0)
    return np.stack([116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])],
                    axis=-1)


def srgb_to_lab(rgb):
    """CIELAB (D65) of sRGB colours (..., 3) in 0..1."""
    return linear_to_lab(decode(rgb))


def delta_e(a, b):
    """CIE76."""
    return float(np.linalg.norm(srgb_to_lab(a) - srgb_to_lab(b)))


def carry(pred, f):
    """(M, c): the map Y -> M Y + c that carries a point of the calibration's
    world, where the clamp was tracked at pose f, into the rig as drawn,
    posed at the command (pred_set): so a ray cast there meets the drawn
    clamp, bat and card where the tracked ones are.  The structure that does
    not turn with the clamp is then off by the gimbal's few millimetres of
    build, which the guard band is for."""
    from . import mjcf as RM
    R0, t0 = RM.body_pose(pred.m, pred.d)
    M = R0 @ np.asarray(f.R).T
    return M, t0 - M @ np.asarray(f.t)


def _visible(pred, cam, X, guard, normal=None, carried=None):
    """Which surface points X (calibration world) camera `cam` sees with a
    guard band: rays to each point and to V.RIM points `guard` round it, in
    the surface's own plane when its normal is given (else square to the
    line of sight), none stopping more than `guard` short on anything a
    colour ray can hit (the marker balls too) -- in the rig as drawn, posed
    at the command, carried (carry) to the tracked pose."""
    import mujoco
    X = np.atleast_2d(np.asarray(X, float))
    C0 = np.asarray(cam.centre, float)
    n = None if normal is None else np.asarray(normal, float)
    if carried is not None:
        M, c = carried
        C0, X = M @ C0 + c, X @ M.T + c
        n = None if n is None else M @ n
    if n is None:
        vec, dist = V.outline_rays(C0, X, guard)
    else:
        e1 = np.cross(n, (0.0, 0.0, 1.0))
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(n, (1.0, 0.0, 0.0))
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        q = V.rim()
        pts = X[:, None, :] + guard * (q[None, :, :1] * e1 + q[None, :, 1:] * e2)
        vv = pts.reshape(-1, 3) - C0
        dist = np.linalg.norm(vv, axis=1)
        vec = vv / dist[:, None]
    N = len(vec)
    gid = np.zeros(N, np.int32)
    hit = np.zeros(N)
    mujoco.mj_multiRay(pred.m, pred.d, C0 / 1000.0, vec.ravel(), V.LIT_GROUPS, 1, -1, gid, hit, None, N, -1)
    blocked = (hit >= 0.0) & (hit * 1000.0 < dist - guard)
    return ~blocked.reshape(len(X), -1).any(axis=1)


def geometry(cam, W, nw, d_exp):
    """(N,): what a sample at W (world) on a surface of normal nw reads per
    unit reflectance, against white square to the light at d_exp: the
    cosine times the light's fall-off."""
    to = cam.centre - np.atleast_2d(W)
    dist = np.linalg.norm(to, axis=1)
    return (to @ nw) / dist * (d_exp / dist) ** 2


def sensor_var(raw):
    """(N, 3): the variance of a raw read (1.0 the full well), from the
    sensor's own facts: shot noise, read noise and the step of its BITS
    above the black level (vision._sensor)."""
    from ..spec import RingCam
    span = 2 ** RingCam.BITS - 1 - RingCam.BLACK_LEVEL
    e = np.clip(np.asarray(raw, float), 0.0, None) * RingCam.FULL_WELL
    return (e + RingCam.READ_NOISE ** 2) / RingCam.FULL_WELL ** 2 + 1.0 / (12.0 * span * span)


def _face_samples(cam, R, t, x0, x1, face, n_x=N_X, n_t=N_T):
    """Sample points on one flat face between x0 and x1 (bat frame, body
    pose (R, t)), their pixels, and whether each keeps EDGE_PX from the
    face's and the strip's edges; and the cosine of the face to the camera."""
    p0, p1, n = face
    lo = (1.0 - FACE_USE) / 2.0
    ts = np.linspace(lo, 1.0 - lo, n_t)
    xs = np.linspace(x0, x1, n_x + 2)[1:-1]
    T, Xx = np.meshgrid(ts, xs)
    yz = p0[None, :] + T.ravel()[:, None] * (p1 - p0)[None, :]
    P = np.column_stack([Xx.ravel(), yz])
    W = P @ R.T + t
    nw = R @ np.array([0.0, n[0], n[1]])
    to = cam.centre - W
    cos = (to @ nw) / np.linalg.norm(to, axis=1)
    u, v, z = cam.project(W)
    # the same points carried to each of the four edges, in the image
    ok = (z > 0) & cam.lens.inside(u, v, EDGE_PX)
    for dx, dt in ((x0 - P[:, 0], 0.0), (x1 - P[:, 0], 0.0)):
        Q = P.copy()
        Q[:, 0] += dx
        ue, ve, _ = cam.project(Q @ R.T + t)
        ok &= np.hypot(ue - u, ve - v) >= EDGE_PX
    for tt in (0.0, 1.0):
        yz_e = p0[None, :] + tt * (p1 - p0)[None, :]
        Q = np.column_stack([P[:, 0], np.broadcast_to(yz_e, (len(P), 2))])
        ue, ve, _ = cam.project(Q @ R.T + t)
        ok &= np.hypot(ue - u, ve - v) >= EDGE_PX
    return W, np.stack([u, v], axis=1), ok & (cos > 0.0), cos


@dataclass
class CardFit:
    A: list                    # per camera (3, 3) or None: raw = A @ reflectance, over the geometry
    patches: list              # per camera: patches read
    resid: list                # per camera: rms of the fit, raw units
    A_sd: list = field(default_factory=list)      # per camera (3, 3): A's 1 sigma, from the sensor's noise
    chi2: list = field(default_factory=list)      # per camera: what the fit leaves, over that noise
    dof: list = field(default_factory=list)


def _patch(cam, R, t, c, nrm, pred, guard, carried=None):
    """(uv (n, 2), W (n, 3), nw (3,)) of a card patch's samples -- the
    centres of the pixels that fall inside it, EDGE_PX in from its edges at
    its foreshortened scale, each read alone (a read at a pixel's centre is
    that pixel), so that no two share a pixel's noise -- through camera
    `cam` with the body at (R, t), or None when the camera cannot read the
    patch (facing away, too oblique to keep a pixel inside, off the sensor,
    or any of it hidden in the scene `pred` is posed in)."""
    c, nrm = np.asarray(c, float), np.asarray(nrm, float)
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.cross(nrm, e1)
    face = c + 0.5 * nrm                                   # the patch's face (rig/mjcf.py)
    nw = R @ nrm
    Wc = R @ face + t
    to = cam.centre - Wc
    depth = float(np.linalg.norm(to))
    cos = float(to @ nw) / depth
    if cos <= 0.0:
        return None
    h = Clamp.CARD_PATCH / 2.0 - EDGE_PX * depth / (cam.lens.fx * cos)
    if h <= 0.0:
        return None
    sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * h
    u, v, z = cam.project((face + sq[:, :1] * e1 + sq[:, 1:] * e2) @ R.T + t)
    if not (np.all(z > 0) and np.all(cam.lens.inside(u, v, EDGE_PX))):
        return None
    jj, ii = np.meshgrid(np.arange(np.floor(u.min()), np.ceil(u.max()) + 1),
                         np.arange(np.floor(v.min()), np.ceil(v.max()) + 1))
    uv = np.column_stack([jj.ravel(), ii.ravel()]) + 0.5
    d = cam.ray_world(uv[:, 0], uv[:, 1])
    s = ((Wc - cam.centre) @ nw) / (d @ nw)
    W = cam.centre + s[:, None] * d
    q = (W - t) @ R - face
    keep = (np.abs(q @ e1) <= h) & (np.abs(q @ e2) <= h)
    if not keep.any():
        return None
    uv, W = uv[keep], W[keep]
    if not _visible(pred, cam, W, guard, nw, carried).all():
        return None                                        # a patch partly hidden is not read
    return uv, W, nw


def fit_card(vision, cams, track_fits, poses, rig, pred, guard, falloff=True, floor=None):
    """CardFit: each camera's response from the card, over the poses given
    (the gimbal commands and the clamp's fitted pose at each).  `falloff`
    False ignores the light's fall-off with distance: a must-fail.
    `floor`, if given, is what the camera adds beyond the sensor's own
    noise, floor(cam, raw) -> (N, 3) variance, the same in every sample of
    a read (a rendered canvas's 8-bit step); a real camera has none."""
    C = len(cams)
    d_exp = V.expose_mm(rig)
    rows = [[] for _ in range(C)]
    n_seen = [0] * C
    for (a, b), f in zip(poses, track_fits):
        if f is None:
            continue
        pred_set(pred, a, b)
        cr = carry(pred, f)
        for name, c, nrm, rgb in rig.card:
            for k, cam in enumerate(cams):
                got = _patch(cam, f.R, f.t, c, nrm, pred, guard, cr)
                if got is None:
                    continue
                uv, W, nw = got
                g = geometry(cam, W, nw, d_exp)
                if not falloff:
                    g = g * (np.linalg.norm(cam.centre - W, axis=1) / d_exp) ** 2
                raw = vision.colour(a, b, k, uv)
                y = (raw / g[:, None]).mean(axis=0)
                var = (sensor_var(raw) / g[:, None] ** 2).mean(axis=0) / len(g)
                if floor is not None:
                    var = var + (floor(k, raw) / g[:, None] ** 2).mean(axis=0)
                rows[k].append((np.asarray(rgb, float), y, var))
                n_seen[k] += 1
    out = CardFit([], n_seen, [], [], [], [])
    for k in range(C):
        if len({tuple(r[0]) for r in rows[k]}) < 3:
            for lst in (out.A, out.A_sd):
                lst.append(None)
            out.resid.append(np.nan)
            out.chi2.append(np.nan)
            out.dof.append(0)
            continue
        X = np.array([r[0] for r in rows[k]])
        Y = np.array([r[1] for r in rows[k]])
        Wt = 1.0 / np.array([r[2] for r in rows[k]])
        keep = np.ones(len(X), bool)
        for _ in range(2):
            # a patch the drawing said was clear and was not reads far off
            # the rest: drop any more than Z_SIGMAS of its own noise off and
            # fit again
            At, sd = _wls(X[keep], Y[keep], Wt[keep])
            z = np.abs(X @ At - Y) * np.sqrt(Wt)
            new = z.max(axis=1) <= Module.Z_SIGMAS * max(1.0, float(np.median(z)))
            if new.all() or len({tuple(x) for x in X[new]}) < 3 or np.array_equal(new, keep):
                break
            keep = new
        At, sd = _wls(X[keep], Y[keep], Wt[keep])
        e = X[keep] @ At - Y[keep]
        out.A.append(At.T)
        out.A_sd.append(sd.T)
        out.resid.append(float(np.sqrt(np.mean(e ** 2))))
        out.chi2.append(float(np.sum(e * e * Wt[keep])))
        out.dof.append(int(3 * (keep.sum() - 3)))
    return out


def _wls(X, Y, Wt):
    """(A^T (3, 3), its sd): Y = X A^T, each channel by weighted least
    squares with weights Wt (n, 3)."""
    At, sd = np.zeros((3, 3)), np.zeros((3, 3))
    for j in range(3):
        w = Wt[:, j]
        H = X.T @ (X * w[:, None])
        Hi = np.linalg.inv(H)
        At[:, j] = Hi @ (X.T @ (w * Y[:, j]))
        sd[:, j] = np.sqrt(np.diag(Hi))
    return At, sd


def card_poses(rig, cams, pred, poses, guard, per_cam=2):
    """(a, b) deg: where the gimbal stops to read the card -- for each
    camera, the `per_cam` of `poses` at which it can read the most patches
    whole, most squarely, as drawn.  A camera that can read none at any of
    them gets none (fit_card then has no A for it)."""
    from . import spec as RS
    score = np.zeros((len(cams), len(poses)))
    for i, (a, b) in enumerate(poses):
        pred_set(pred, a, b)
        R = RS.body_R(a, b)
        for name, c, nrm, rgb in rig.card:
            for k, cam in enumerate(cams):
                got = _patch(cam, R, np.zeros(3), c, nrm, pred, guard)
                if got is not None:
                    score[k, i] += float(geometry(cam, got[1], got[2], V.expose_mm(rig)).mean())
    pick = set()
    for k in range(len(cams)):
        best = np.argsort(-score[k])[:per_cam]
        pick |= {int(i) for i in best if score[k, i] > 0.0}
    return np.asarray(poses)[sorted(pick)]


def verdict(reads, want):
    """(ok, [dE]): each band as read against the colour it should be --
    the line refuses the bat when any is more than Rig.DE_MAX off."""
    from ..spec import Rig
    de = [float(delta_e(r.rgb, w)) if r.n else np.inf for r, w in zip(reads, want)]
    return all(e <= Rig.DE_MAX for e in de), de


def pred_set(pred, a, b):
    """The rig as drawn at gimbal command (a, b), for occlusion rays."""
    from . import mjcf as RM
    RM.set_gimbal(pred.m, pred.d, a, b, pred.drawn)


@dataclass
class BandRead:
    name: str
    rgb: np.ndarray            # the band's colour as read, sRGB 0..1 (encode of lin)
    n: int                     # samples behind it
    per_cam: dict = field(default_factory=dict)    # cam -> (linear, samples)
    lin: np.ndarray = None     # its linear reflectance
    sleeve: np.ndarray = None  # the sleeve's linear reflectance, as measured (the same for every band)


def read_bands(vision, cams, card, track_fits, poses, rig, pred, guard, correct=True):
    """[BandRead]: each band's colour, from every camera that sees a flat
    face of it squarely enough, against the sleeve beside it, the sleeve
    measured through the card.  `correct` False reads with no card (A the
    identity): a must-fail."""
    geo = band_geometry(rig)
    d_exp = V.expose_mm(rig)
    acc = {name: {} for name, *_ in geo}
    sl, n_sl = np.zeros(3), 0
    for (a, b), f in zip(poses, track_fits):
        if f is None:
            continue
        pred_set(pred, a, b)
        cr = carry(pred, f)
        for name, x0, x1, bf, sf, s0, s1 in geo:
            w = x1 - x0
            b0, b1 = max(x0, s0), min(x1, s1)                  # the band where it is outermost
            strips = [(max(x0 - w / 2.0, s0), x0), (x1, min(x1 + w / 2.0, s1))]
            strips = [(p, q) for p, q in strips if q > p]
            for k, cam in enumerate(cams):
                if correct and card.A[k] is None:
                    continue
                Ainv = np.linalg.inv(card.A[k]) if correct else np.eye(3)
                for face_b, face_s in zip(bf, sf):
                    nw = f.R @ np.array([0.0, face_b[2][0], face_b[2][1]])
                    Wb, uvb, okb, _ = _face_samples(cam, f.R, f.t, b0, b1, face_b)
                    if not okb.all():
                        continue                               # a face is read whole or not at all
                    side, side_w = [], []
                    for p0, p1 in strips:
                        Ws, uvs, oks, _ = _face_samples(cam, f.R, f.t, p0, p1, face_s)
                        if oks.all() and _visible(pred, cam, Ws, guard, nw, cr).all():
                            side.append(uvs)
                            side_w.append(Ws)
                    if not side or not _visible(pred, cam, Wb, guard, nw, cr).all():
                        continue
                    Ws = np.vstack(side_w)
                    gb, gs = geometry(cam, Wb, nw, d_exp), geometry(cam, Ws, nw, d_exp)
                    rb = Ainv @ (vision.colour(a, b, k, uvb) / gb[:, None]).mean(axis=0)
                    rs_all = (vision.colour(a, b, k, np.vstack(side)) / gs[:, None]) @ Ainv.T
                    rs = rs_all.mean(axis=0)
                    ratio = rb / np.maximum(rs, 1e-9)
                    n0 = acc[name].get(k, (np.zeros(3), 0))
                    acc[name][k] = (n0[0] + ratio * len(uvb), n0[1] + len(uvb))
                    sl += rs_all.sum(axis=0)
                    n_sl += len(rs_all)
    sleeve = sl / n_sl if n_sl else np.full(3, np.nan)
    out = []
    for name, *_ in geo:
        per = {k: (sm / n * sleeve, n) for k, (sm, n) in acc[name].items()}
        n = sum(v[1] for v in per.values())
        lin = sum(v[0] * v[1] for v in per.values()) / n if n else np.full(3, np.nan)
        out.append(BandRead(name, encode(lin) if n else lin, n, per, lin, sleeve))
    return out
