"""The bands' colours, read back by the rig's cameras.

The phone finds the bat by its two printed bands, so the line stores each
band's colour as printed, and refuses a bat whose bands are not the colours
the design gave them (spec.Rules.BAND_RGB, judged by Rig.DE_MAX).

A camera's raw reading of a surface is its colour times the light on it
times the cosine of the light's angle, through the sensor's own colour
response.  Two things take those apart:

  THE CARD.  Patches of known colour in a row on each of the clamp's
  flanks and its underside (spec.Clamp: white, red, green, blue), each row
  flat.  The gimbal stops where each camera can read a row whole and
  squarely (card_poses).  Their readings,
  each over its cosine, fit A = s M: the sensor's response M (each
  channel's gain and its mixing) times the light s, by least squares over
  the patches' three channels -- four patches for nine unknowns.  A
  camera's A is fitted over every pose its card faces it at.

  THE SLEEVE BESIDE THE BAND.  A band and the sleeve next to it on the
  same flat face share the face's normal, the light and the camera, so
  A^-1 of the band's reading over A^-1 of the sleeve's, channel by
  channel, times the sleeve's own colour (the design's), is the band's
  colour -- whatever s, the cosine and the renderer's shading are.

Samples stay EDGE_PX from every edge in the image: a face's, a band's.
Each camera reads each face it sees squarely enough to keep that margin
on a face's middle 70 % (FACE_USE); the reads are averaged, weighted by
their samples.  The colour difference is CIE76 in CIELAB (D65), the
colours taken as sRGB as a designer gives them, which is how the model's
own colours are given.
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


def srgb_to_lab(rgb):
    """CIELAB (D65) of sRGB colours (..., 3) in 0..1."""
    c = np.clip(np.asarray(rgb, float), 0.0, None)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    e = 216.0 / 24389.0
    f = np.where(xyz > e, np.cbrt(xyz), (24389.0 / 27.0 * xyz + 16.0) / 116.0)
    return np.stack([116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])],
                    axis=-1)


def delta_e(a, b):
    """CIE76."""
    return float(np.linalg.norm(srgb_to_lab(a) - srgb_to_lab(b)))


def _visible(pred, cam, X, guard):
    """Which surface points X (world) camera `cam` sees with a guard band:
    rays to each point and to V.RIM points `guard` round it, square to the
    line of sight, none stopping more than `guard` short on anything a
    colour ray can hit (the marker balls too) -- in the rig as drawn, which
    lies up to the build's tolerance from the rig as built."""
    import mujoco
    C0 = cam.centre
    vec, dist = V.outline_rays(C0, np.atleast_2d(X), guard)
    N = len(vec)
    gid = np.zeros(N, np.int32)
    hit = np.zeros(N)
    mujoco.mj_multiRay(pred.m, pred.d, C0 / 1000.0, vec.ravel(), V.LIT_GROUPS, 1, -1, gid, hit, None, N, -1)
    blocked = (hit >= 0.0) & (hit * 1000.0 < dist - guard)
    return ~blocked.reshape(len(np.atleast_2d(X)), -1).any(axis=1)


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
    A: list                    # per camera (3, 3) or None: raw = A @ rgb, over the cosine
    patches: list              # per camera: patches read
    resid: list                # per camera: rms of the fit, raw units


def _patch(cam, R, t, c, nrm, pred, guard):
    """(uv (9, 2), cos (9,)) of a card patch's samples -- a 3 x 3 grid
    EDGE_PX in from its edges at its foreshortened scale -- through camera
    `cam` with the body at (R, t), or None when the camera cannot read the
    patch whole (facing away, too oblique, off the sensor, or any of it
    hidden in the scene `pred` is posed in)."""
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
    g = np.linspace(-h, h, 3)
    G1, G2 = np.meshgrid(g, g)
    P = face + G1.ravel()[:, None] * e1 + G2.ravel()[:, None] * e2
    W = P @ R.T + t
    u, v, z = cam.project(W)
    if not (np.all(z > 0) and np.all(cam.lens.inside(u, v, EDGE_PX))):
        return None
    if not _visible(pred, cam, W, guard).all():
        return None                                        # a patch partly hidden is not read
    cs = ((cam.centre - W) @ nw) / np.linalg.norm(cam.centre - W, axis=1)
    return np.stack([u, v], 1), cs


def fit_card(vision, cams, track_fits, poses, rig, pred, guard):
    """CardFit: each camera's response from the card, over the poses given
    (the gimbal commands and the clamp's fitted pose at each)."""
    C = len(cams)
    rows = [[] for _ in range(C)]
    n_seen = [0] * C
    for (a, b), f in zip(poses, track_fits):
        if f is None:
            continue
        pred_set(pred, a, b)
        for name, c, nrm, rgb in rig.card:
            for k, cam in enumerate(cams):
                got = _patch(cam, f.R, f.t, c, nrm, pred, guard)
                if got is None:
                    continue
                uv, cs = got
                raw = vision.colour(a, b, k, uv)
                rows[k].append((np.asarray(rgb, float), (raw / cs[:, None]).mean(axis=0)))
                n_seen[k] += 1
    A, resid = [], []
    for k in range(C):
        if len({tuple(r[0]) for r in rows[k]}) < 3:
            A.append(None)
            resid.append(np.nan)
            continue
        X = np.array([r[0] for r in rows[k]])
        Y = np.array([r[1] for r in rows[k]])
        At = np.linalg.lstsq(X, Y, rcond=None)[0]               # Y = X A^T
        # a patch the drawing said was clear and was not reads far off the
        # rest: drop any off by Z_SIGMAS times the median patch and fit again
        e = np.linalg.norm(X @ At - Y, axis=1)
        keep = e <= Module.Z_SIGMAS * np.median(e)
        if not keep.all() and len({tuple(x) for x in X[keep]}) >= 3:
            X, Y = X[keep], Y[keep]
            At = np.linalg.lstsq(X, Y, rcond=None)[0]
        A.append(At.T)
        resid.append(float(np.sqrt(np.mean((X @ At - Y) ** 2))))
    return CardFit(A, n_seen, resid)


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
                    score[k, i] += got[1].mean()
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
    rgb: np.ndarray            # the band's colour as read (sRGB, 0..1)
    n: int                     # samples behind it
    per_cam: dict = field(default_factory=dict)    # cam -> (rgb, samples)


def read_bands(vision, cams, card, track_fits, poses, rig, pred, guard, sleeve_rgb, correct=True):
    """[BandRead]: each band's colour, from every camera that sees a flat
    face of it squarely enough, against the sleeve beside it."""
    geo = band_geometry(rig)
    acc = {name: {} for name, *_ in geo}
    for (a, b), f in zip(poses, track_fits):
        if f is None:
            continue
        pred_set(pred, a, b)
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
                    Wb, uvb, okb, cos = _face_samples(cam, f.R, f.t, b0, b1, face_b)
                    if not okb.all():
                        continue                               # a face is read whole or not at all
                    side = []
                    for p0, p1 in strips:
                        Ws, uvs, oks, _ = _face_samples(cam, f.R, f.t, p0, p1, face_s)
                        if oks.all() and _visible(pred, cam, Ws, guard).all():
                            side.append(uvs)
                    if not side or not _visible(pred, cam, Wb, guard).all():
                        continue
                    rb = (Ainv @ vision.colour(a, b, k, uvb).mean(axis=0))
                    rs = (Ainv @ vision.colour(a, b, k, np.vstack(side)).mean(axis=0))
                    rgb = rb / np.maximum(rs, 1e-9) * np.asarray(sleeve_rgb, float)
                    n0 = acc[name].get(k, (np.zeros(3), 0))
                    acc[name][k] = (n0[0] + rgb * len(uvb), n0[1] + len(uvb))
    out = []
    for name, *_ in geo:
        per = {k: (sm / n, n) for k, (sm, n) in acc[name].items()}
        n = sum(v[1] for v in per.values())
        rgb = sum(v[0] * v[1] for v in per.values()) / n if n else np.full(3, np.nan)
        out.append(BandRead(name, rgb, n, per))
    return out
