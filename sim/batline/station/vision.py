"""The head camera's look: where a dark disc's centre is, in pixels.

Two implementations of station hal.VisionHAL with one contract, as
truss/vision.py has for the cell:

  ModelVision   the camera as built -- its pose read from the scene, not
                from the drawing -- each disc's true centre projected
                through it, and each look's noise (HeadCam.CENTROID_PX).
                Fast; what the executor and the checks run on.
  PixelVision   a rendered frame: the dark blobs near the pixel asked
                about, the one that is round and wholly in view, its centre
                weighted by how dark each pixel is, since an edge pixel is
                part disc and part ground.  Slow; check_gantry holds the
                model camera to it, which is what makes CENTROID_PX a
                measurement rather than a guess.

The discs are the regions' fiducials, the reference pin's top and the
tokens' marks: whatever the scene names fid_*, *_fid or pin.  Neither
camera turns pixels into millimetres; commissioning measures how.

Pixel coordinates are continuous: pixel (row i, column j) covers
[j, j+1) x [i, i+1), so its centre is (j + 0.5, i + 0.5) and the principal
point of a centred lens is (W/2, H/2).  v grows down the image.
"""
from math import tan, radians

import numpy as np
import mujoco

from ..spec import HeadCam
from . import hal

DISC_FILL = np.pi / 4.0          # a disc fills pi/4 of its bounding box


def discs(model):
    """[(geom id, radius m, half height m)] of everything the camera looks
    for: the scene's naming is the contract."""
    out = []
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name.startswith("fid_") or name.endswith("_fid") or name == "pin":
            out.append((g, float(model.geom_size[g][0]), float(model.geom_size[g][1])))
    return out


def project(model, data, p, cam="head_cam"):
    """(u, v, depth m) of world point p (m) through a MuJoCo camera, the
    way MuJoCo renders it: looking down -z, x right, y up."""
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    R = data.cam_xmat[cid].reshape(3, 3)
    q = R.T @ (np.asarray(p, float) - data.cam_xpos[cid])
    depth = -q[2]
    f = (HeadCam.H / 2.0) / tan(radians(model.cam_fovy[cid]) / 2.0)
    return HeadCam.W / 2.0 + f * q[0] / depth, HeadCam.H / 2.0 - f * q[1] / depth, depth


class ModelVision(hal.VisionHAL):
    def __init__(self, model, data, rng=None):
        self.m, self.d = model, data
        self.rng = rng if rng is not None else np.random.default_rng(1)
        self.discs = discs(model)
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
        self.f = (HeadCam.H / 2.0) / tan(radians(model.cam_fovy[cid]) / 2.0)

    def truth(self):
        """[(u, v, radius px)] of every disc wholly in view, noise-free."""
        out = []
        for g, r, hh in self.discs:
            top = self.d.geom_xpos[g] + self.d.geom_xmat[g].reshape(3, 3)[:, 2] * hh
            u, v, depth = project(self.m, self.d, top)
            if depth <= 0.0:
                continue
            rp = self.f * r / depth
            if rp <= u <= HeadCam.W - rp and rp <= v <= HeadCam.H - rp:
                out.append((u, v, rp))
        return out

    def locate(self, near=None):
        near = (HeadCam.W / 2.0, HeadCam.H / 2.0) if near is None else near
        seen = self.truth()
        if not seen:
            return None
        u, v, _ = min(seen, key=lambda s: (s[0] - near[0]) ** 2 + (s[1] - near[1]) ** 2)
        e = self.rng.normal(0.0, HeadCam.CENTROID_PX, 2)
        return float(u + e[0]), float(v + e[1])


# ================================================================ PIXELS
def otsu(vals):
    """The threshold that best splits a set of grey levels in two."""
    h, edges = np.histogram(vals, bins=256, range=(0.0, 256.0))
    p = h / max(h.sum(), 1)
    w = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + 0.5))
    mt = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (mt * w - mu) ** 2 / (w * (1.0 - w))
    s[~np.isfinite(s)] = 0.0
    return float(edges[int(np.argmax(s)) + 1])


def components(mask):
    """4-connected components of a boolean image, by runs and union-find:
    [(area, (i0, i1, j0, j1) inclusive bbox, run list)]."""
    runs, parent = [], []

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    prev = []
    for i in range(mask.shape[0]):
        row = mask[i]
        if not row.any():
            prev = []
            continue
        dd = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        starts, ends = np.flatnonzero(dd == 1), np.flatnonzero(dd == -1)
        cur = []
        for j0, j1 in zip(starts, ends):
            k = len(runs)
            runs.append((i, int(j0), int(j1)))
            parent.append(k)
            for kp in prev:
                _, p0, p1 = runs[kp]
                if p0 < j1 and j0 < p1:
                    ra, rb = find(k), find(kp)
                    if ra != rb:
                        parent[ra] = rb
            cur.append(k)
        prev = cur
    groups = {}
    for k in range(len(runs)):
        groups.setdefault(find(k), []).append(runs[k])
    out = []
    for rl in groups.values():
        area = sum(j1 - j0 for _, j0, j1 in rl)
        i0, i1 = min(r[0] for r in rl), max(r[0] for r in rl)
        j0, j1 = min(r[1] for r in rl), max(r[2] for r in rl) - 1
        out.append((area, (i0, i1, j0, j1), rl))
    return out


def _dilate(mask, n):
    out = mask.copy()
    for _ in range(n):
        m = out.copy()
        m[1:, :] |= out[:-1, :]
        m[:-1, :] |= out[1:, :]
        m[:, 1:] |= out[:, :-1]
        m[:, :-1] |= out[:, 1:]
        out = m
    return out


class PixelVision(hal.VisionHAL):
    """`camera` is a truss.hal.CameraHAL: frame() gives the image."""

    # a disc is round to within this: its fill of its bounding box, and the
    # box's aspect -- loose enough for a tilted lens and a 28-sided render,
    # tight enough to refuse a bar or a corner
    FILL = (0.65, 0.92)
    ASPECT = (0.8, 1.25)

    def __init__(self, camera, rng=None, half_window=None):
        self.cam = camera
        self.rng = rng
        # where to look round `near`: the largest disc there is (the pin, 8
        # mm) from the nearest look, and the drawing's worst error round it
        self.half = half_window or 200
        self.last = None             # what the last look saw, for a report

    def locate(self, near=None, img=None):
        if img is None:
            img, _t = self.cam.frame()
        g = img.astype(float).mean(axis=2)
        H, W = g.shape
        near = (W / 2.0, H / 2.0) if near is None else near
        i0, i1 = max(int(near[1]) - self.half, 0), min(int(near[1]) + self.half, H)
        j0, j1 = max(int(near[0]) - self.half, 0), min(int(near[0]) + self.half, W)
        win = g[i0:i1, j0:j1]
        thr = otsu(win.ravel())
        dark = win < thr
        best = None
        for area, (a0, a1, b0, b1), rl in components(dark):
            h, w = a1 - a0 + 1, b1 - b0 + 1
            if min(h, w) < HeadCam.MIN_PX * 0.5:
                continue
            if a0 == 0 or b0 == 0 or a1 == win.shape[0] - 1 or b1 == win.shape[1] - 1:
                continue                                    # not wholly in view
            fill, aspect = area / float(h * w), w / float(h)
            if not (self.FILL[0] <= fill <= self.FILL[1] and self.ASPECT[0] <= aspect <= self.ASPECT[1]):
                continue
            cu, cv = j0 + (b0 + b1 + 1) / 2.0, i0 + (a0 + a1 + 1) / 2.0
            dist = (cu - near[0]) ** 2 + (cv - near[1]) ** 2
            if best is None or dist < best[0]:
                best = (dist, rl, (a0, a1, b0, b1))
        if best is None:
            self.last = None
            return None
        _d, rl, (a0, a1, b0, b1) = best
        # sub-pixel: every pixel in or touching the blob, weighted by how much
        # of it is dark -- (light - grey) / (light - dark)
        m = np.zeros_like(dark)
        for i, s, e in rl:
            m[i, s:e] = True
        ring = _dilate(m, 2)
        light = float(np.median(win[~_dilate(dark, 2)])) if (~_dilate(dark, 2)).any() else float(win.max())
        core = m & ~_dilate(~m, 1)
        darkv = float(np.median(win[core])) if core.any() else float(win[m].min())
        wgt = np.clip((light - win) / max(light - darkv, 1e-6), 0.0, 1.0) * ring
        ii, jj = np.nonzero(ring)
        ww = wgt[ii, jj]
        u = j0 + float((ww * (jj + 0.5)).sum() / ww.sum())
        v = i0 + float((ww * (ii + 0.5)).sum() / ww.sum())
        self.last = {"thr": thr, "light": light, "dark": darkv, "px": float(2.0 * np.sqrt(ww.sum() / np.pi))}
        return u, v
