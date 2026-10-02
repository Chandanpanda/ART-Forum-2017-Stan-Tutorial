"""The rig's eight cameras: what each sees of the gimbal, as blobs and colours.

Two implementations of one contract (RigVisionHAL), as station/vision.py
has for the head camera:

  ModelVision   the cameras as built (their poses and true lenses, which
                no procedure here may read), each marker ball that is in
                view (its centre and RIM rays round its outline clear: a
                cap under 1 % of its disc can hide between two rays, which
                a rendered blob would show as a shift of up to 0.17 px; the
                tracker's guard band refuses those balls) projected
                through them with the centroid's noise
                (RingCam.CENTROID_PX), and colours by casting the pixel's
                ray into the scene and lighting what it hits.  Fast; what
                self-calibration and the checks run on.
  PixelVision   rendered frames.  MuJoCo draws an ideal pinhole picture on
                one shared canvas (mjcf.canvas); it is warped through the
                camera's true lens, given the light, the sensor's colour
                response, shot and read noise, and cut to 10 bits; blobs are
                found in it and their centroids weighted by how bright each
                pixel is.  Slow; check_cameras holds its centroids to the
                CENTROID_PX budget the model camera draws its noise at
                (they come in at about a third of it).

Neither returns anything labelled: a snap is, per camera, the blobs in no
order, and a colour read is the raw sensor values at the pixels asked
about.  Which blob is which ball is the tracker's job (track.py), and how
raw values become a colour is colour.py's.

TWO EXPOSURES.  A marker snap: every camera on one trigger, each with its
own ring light strobed, short, so a retroreflective ball (spec.Marker) is a
bright uniform disc and everything diffuse sits RETRO_GAIN below it.  A
colour snap: one camera at a time with its own light, long enough that a
white diffuse surface facing the light fills the same level.

EXPOSURE.  The brightest thing an exposure is for -- a marker, or white
facing the light at expose_mm, the nearest the inner member's surfaces
come to a camera -- is set to EXPOSE of the sensor's full well, so a light
or a channel Module.Z_SIGMAS brighter than drawn (RigBuild.LIGHT,
COLOUR_GAIN) still does not clip.  A ring light falls off as the inverse
square of the distance (rig/mjcf.py), so the same white further off reads
less, in both cameras.

COLOUR.  A surface's rgba in the scene is its linear reflectance
(rig/mjcf.py; colour.decode of a designer's sRGB), and a raw read is
linear in it.
"""
from abc import ABC, abstractmethod
from math import sqrt

import numpy as np

from ..spec import RingCam, Marker, RigBuild, Module, Rig
from .lens import Camera
from . import mjcf as RM

EXPOSE = 1.0 / (1.0 + Module.Z_SIGMAS * sqrt(RigBuild.LIGHT ** 2 + RigBuild.COLOUR_GAIN ** 2))
RIM = 8                        # rays round a ball's outline the model camera checks, besides its centre
CANVAS_BITS = 8                # what MuJoCo's renderer writes per channel
# the colour canvas's second, brighter draw: enough that the canvas's step
# is a quarter of the sensor's (PixelVision._colour_canvas)
BRACKET = 2.0 ** (RingCam.BITS + 2 - CANVAS_BITS)
HOT = 0.1                      # of a marker's level: canvas pixels a marker frame warps round (far above
                               # the diffuse world's 1 / RETRO_GAIN, far below a ball's edge pixels)
HALO = 6                       # px round them: a blob, the two-pixel ring its centroid reads, and its ground
RING = 2                       # px round a blob its centroid reads: an edge pixel is part ball
GAP = RING + 1.0               # px two balls' images keep apart to be read alone: the ring, and a pixel of blur
SEEN_GROUPS = np.array([1, 1, 0, 0, 1, 0], np.uint8)     # what a ray stops at: structure, the bat, stems
LIT_GROUPS = np.array([1, 1, 0, 1, 1, 0], np.uint8)      # what a colour ray hits: and the balls
STEM_GROUP = np.array([0, 0, 0, 0, 1, 0], np.uint8)      # rig/mjcf.py's group 4


def expose_mm(rig):
    """mm: the distance the colour exposure is set at -- the nearest any
    surface on the gimbal's inner member (the bat, the clamp, the card)
    can come to a camera."""
    return rig.standoff - rig.R_inner


def rim(n=RIM):
    """(n + 1, 2): a ball's centre and n points round its outline, as
    (cos, sin) of the outline's angle."""
    t = 2.0 * np.pi * np.arange(n) / n
    return np.vstack([[0.0, 0.0], np.stack([np.cos(t), np.sin(t)], axis=1)])


FILL = 0.95                    # of a ball's radius: where the model camera's outline rays aim, just inside
                               # the edge, so a stem meeting the ball behind its outline does not count


def outline_rays(C0, X, rho, n=RIM):
    """(K * (n + 1), 3) unit directions from C0 to each ball's centre and
    n points rho round it, square to the line of sight, and the distance
    to each point."""
    w = X - C0
    w = w / np.linalg.norm(w, axis=1, keepdims=True)
    e1 = np.cross(w, (0.0, 0.0, 1.0))
    flat = np.linalg.norm(e1, axis=1) < 1e-6
    e1[flat] = np.cross(w[flat], (1.0, 0.0, 0.0))
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(w, e1)
    q = rim(n)
    pts = X[:, None, :] + rho * (q[None, :, :1] * e1[:, None, :] + q[None, :, 1:] * e2[:, None, :])
    vec = pts.reshape(-1, 3) - C0
    dist = np.linalg.norm(vec, axis=1)
    return vec / dist[:, None], dist


def clear_of(m, d, C0, X, r, rho=None, groups=SEEN_GROUPS, n=RIM, own=None):
    """(K,) bool: which balls (centres X, mm, radius r) C0 sees whole.  Rays
    go to each centre and n points rho round it (FILL r if None); one is
    blocked when it meets anything before the ball's own near surface on
    that ray -- or, for rho past the ball's edge (a guard band), before the
    ball's front.  Balls themselves stop no ray (rig/mjcf.py's group 3).

    own (K,): each ball's own stem's geom id (mjcf.stem_ids), or -1.  A
    ball's own stem does not hide it: seen from its own side it crosses
    the disc, and what it does there is pull the centroid, which
    stem_bias_px models and the tracker corrects.  A ray whose first hit is
    the ball's own stem is cast again past the stems, so whatever stands
    behind that stem (the ring at its foot) still counts.  Without `own`
    a stem hides its own ball: then whether a ball is seen depends on how
    its stem's image falls between the n rays, not on the geometry."""
    import mujoco
    rho = FILL * r if rho is None else rho
    vec, dist = outline_rays(C0, X, rho, n)
    q = np.hypot(*rim(n).T) * rho
    front = np.tile(np.where(q < r, np.sqrt(np.maximum(r * r - q * q, 0.0)), r), len(X))
    N = len(vec)
    gid = np.zeros(N, np.int32)
    hit = np.zeros(N)
    mujoco.mj_multiRay(m, d, C0 / 1000.0, vec.ravel(), groups, 1, -1, gid, hit, None, N, -1)
    if own is not None and groups[4]:
        mine = (hit >= 0.0) & (gid == np.repeat(np.asarray(own, int), n + 1)) & (np.repeat(np.asarray(own), n + 1) >= 0)
        if mine.any():
            k = np.flatnonzero(mine)
            g2 = np.zeros(len(k), np.int32)
            h2 = np.zeros(len(k))
            rest = (np.asarray(groups, np.uint8) & (1 - STEM_GROUP)).astype(np.uint8)
            mujoco.mj_multiRay(m, d, C0 / 1000.0, vec[k].ravel(), rest, 1, -1, g2, h2, None, len(k), -1)
            hit[k] = h2
    blocked = (hit >= 0.0) & (hit * 1000.0 < dist - front)
    return ~blocked.reshape(len(X), n + 1).any(axis=1)


def image_radius(cam, X, r):
    """px: the radius of each ball's image, where the lens shrinks it."""
    L = cam.lens
    Xc = cam.to_cam(X)
    z = np.maximum(Xc[:, 2], 1e-9)
    x, y = Xc[:, 0] / z, Xc[:, 1] / z
    r2 = x * x + y * y
    tang = 1.0 + L.k1 * r2 + L.k2 * r2 * r2 + L.k3 * r2 ** 3
    rad = 1.0 + 3.0 * L.k1 * r2 + 5.0 * L.k2 * r2 * r2 + 7.0 * L.k3 * r2 ** 3
    return L.fx * r / z * np.sqrt(np.abs(tang * rad))


def stem_bias_px(cam, X, S, rr, r):
    """(K,) px, and (K, 2) the unit image direction it pushes: how far each
    ball's stem pulls its centroid, by the area it hides.  A stem leaves
    the ball at the point its direction S (world, unit, ball toward stem)
    meets the surface; seen from the stem's side (s = S . w > 0, w toward
    the camera) it crosses the disc from that point's image out to the rim,
    a strip 2 rho wide and r (1 - sqrt(1 - s^2)) long whose middle is
    r (1 + sqrt(1 - s^2)) / 2 out, so it takes rho s^2 / pi off the
    centroid, in mm, away from the stem.  Seen from the far side it hides
    behind the ball and takes nothing.

    That is the model's shape, and it is right (check_cameras: after it,
    what is left along the stem is what is left across it).  Its scale is
    not: a centroid weights a ball's edge as the image makes it, so the
    real pull is some multiple of this one, the same for every ball on the
    rig, which self-calibration solves (track.py, spec.RigBuild.STEM)."""
    w = cam.centre - X
    w = w / np.linalg.norm(w, axis=1, keepdims=True)
    sw = np.einsum("ij,ij->i", S, w)
    rho = Marker.STEM_D / 2.0
    bias = rho * np.maximum(sw, 0.0) ** 2 / np.pi * rr / r
    u0, v0, _ = cam.project(X)
    u1, v1, _ = cam.project(X + S)
    dvec = np.stack([u1 - u0, v1 - v0], axis=1)
    n = np.linalg.norm(dvec, axis=1, keepdims=True)
    return bias, -dvec / np.where(n > 1e-12, n, 1.0)


def stem_clear(cam, X, S, L, r):
    """(K,) bool: the stem model holds -- each stem (length L mm from its
    ball's surface) runs out past its ball's outline in the image, which
    it does while it is seen at least r / (r + L) side-on.  Nearer end-on
    it stops inside the disc and the tube it stands on is what lies over
    the ball."""
    w = cam.centre - X
    w = w / np.linalg.norm(w, axis=1, keepdims=True)
    sw = np.einsum("ij,ij->i", S, w)
    return sw <= np.sqrt(1.0 - (r / (r + np.asarray(L, float))) ** 2)


def wholly_on(cam, u, v, rr):
    L = cam.lens
    return (u - rr >= 0.0) & (u + rr <= L.W) & (v - rr >= 0.0) & (v + rr <= L.H)


def apart(u, v, rr, ok, front=None):
    """Which of the images `ok` keeps GAP px from every other ball's in
    front of the camera: two balls whose images meet are one blob, and a
    neighbour inside the ring a centroid reads drags it."""
    out = ok.copy()
    idx = np.flatnonzero(ok)
    near = np.flatnonzero(ok if front is None else (front | ok))
    for i in idx:
        for j in near:
            if j != i and np.hypot(u[i] - u[j], v[i] - v[j]) < rr[i] + rr[j] + GAP:
                out[i] = False
                break
    return out


# ================================================================ CONTRACT
class RigVisionHAL(ABC):
    """The rig's eight cameras.  `cams` is the drawing: the procedures
    start from it and calibrate it; the cameras as built are the scene's."""
    cams: list

    @abstractmethod
    def snap(self, a, b):
        """The gimbal at (a, b) deg commanded, one marker exposure of every
        camera: per camera an (N, 3) array of blobs (u, v px, area px^2),
        in no order."""

    @abstractmethod
    def colour(self, a, b, cam, uv):
        """The gimbal at (a, b), camera `cam`'s colour exposure, read at
        pixels uv (N, 2): (N, 3) raw linear values, 1.0 the full well."""


def audit(backend):
    """The contract's methods the backend lacks."""
    return [n for n in ("snap", "colour") if not callable(getattr(backend, n, None))]


class Contract(RigVisionHAL):
    """A backend seen through the contract alone: its cams, snap and colour.
    What the checks hand the procedures, so one that reached past the
    contract for the scene (its model, its build, its truth) raises where it
    reached, instead of passing on what a real rig would not have told it."""
    __slots__ = ("_backend",)

    def __init__(self, backend):
        object.__setattr__(self, "_backend", backend)

    @property
    def cams(self):
        return self._backend.cams

    def snap(self, a, b):
        return self._backend.snap(a, b)

    def colour(self, a, b, cam, uv):
        return self._backend.colour(a, b, cam, uv)

    def __getattr__(self, name):
        raise AttributeError("%r is not the rig's to read: a procedure has cams, snap and colour" % name)

    def __setattr__(self, name, value):
        raise AttributeError("a procedure does not change its cameras (%r)" % name)


class _Scene:
    """The as-built scene both cameras look into, and the truth a check may
    read of it (never the procedures: track, calibrate, colour)."""

    def __init__(self, rig, site, cams, build, payload, size=None):
        import mujoco
        self.rig, self.site, self.cams, self.build, self.payload = rig, site, list(cams), build, payload
        self.xml, self.info = RM.scene(rig, site, self.cams, build, payload=payload, size=size)
        self.m = mujoco.MjModel.from_xml_string(self.xml)
        self.d = mujoco.MjData(self.m)
        self.balls = self.info["markers"] + self.info["bar"]
        self.r = rig.ball_r
        self._at = None

    def pose(self, a, b):
        if self._at != (a, b):
            RM.set_gimbal(self.m, self.d, a, b, self.build)
            self._at = (a, b)

    # ------------------------------------------------ truth, for checks only
    def truth_pose(self, a, b):
        """(R, t mm): the inner member's pose, body -> rig frame."""
        self.pose(a, b)
        return RM.body_pose(self.m, self.d)

    def truth_balls(self, a, b):
        """{name: (3,) mm}: where every ball is."""
        self.pose(a, b)
        return dict(zip(self.balls, RM.balls_world(self.m, self.d, self.balls)))

    def truth_cams(self):
        return self.build.cams

    def paint(self, part, rgb):
        """Print part (a bat part's name: "band1", "sleeve") in rgb, sRGB
        as a designer gives it: what the line put on this bat, which the
        cameras then read.  The print reflects its decoded value."""
        from .colour import decode
        self.paint_linear(part, decode(rgb))

    def paint_linear(self, part, lin):
        """Print part with linear reflectance `lin`."""
        import mujoco
        rgb = np.asarray(lin, float)
        n = 0
        for g in range(self.m.ngeom):
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if name.startswith(part + "/"):
                self.m.geom_rgba[g, :3] = rgb
                n += 1
        if not n:
            raise ValueError("no part %r in the scene" % part)
        if hasattr(self, "_img"):
            self._img, self._col = {}, None


# ================================================================ MODEL
class ModelVision(RigVisionHAL, _Scene):
    def __init__(self, rig, site, cams, build, payload="bat", rng=None):
        _Scene.__init__(self, rig, site, cams, build, payload, size=(64, 64))
        self.rng = rng if rng is not None else np.random.default_rng(1)
        self.sigma = RingCam.CENTROID_PX
        self.stems = RM.stems(rig, build.markers, build.bar if payload == "bar" else ())
        self.own = RM.stem_ids(self.m, len(build.markers), len(build.bar) if payload == "bar" else 0)
        self.d_exp = expose_mm(rig)

    def snap(self, a, b):
        self.pose(a, b)
        X = RM.balls_world(self.m, self.d, self.balls)
        R, _ = RM.body_pose(self.m, self.d)
        S = self.stems @ R.T
        out = []
        for c in self.build.cams:
            u, v, z = c.project(X, self.r)
            rr = image_radius(c, X, self.r)
            ok = (z > self.r) & wholly_on(c, u, v, rr)
            if ok.any():
                ok[ok] = clear_of(self.m, self.d, c.centre, X[ok], self.r, own=self.own[ok])
            ok = apart(u, v, rr, ok, z > self.r)
            bias, dirn = stem_bias_px(c, X, S, rr, self.r)
            bias = bias * self.build.stem
            u, v = u + bias * dirn[:, 0], v + bias * dirn[:, 1]
            n = int(ok.sum())
            e = self.rng.normal(0.0, self.sigma, (n, 2))
            blobs = np.stack([u[ok] + e[:, 0], v[ok] + e[:, 1], np.pi * rr[ok] ** 2], axis=1)
            out.append(blobs[self.rng.permutation(n)])
        return out

    def colour(self, a, b, cam, uv):
        """Each pixel's ray, cast from the camera as built through its true
        lens, lights what it hits by the camera's own light: the surface's
        colour times the light times the cosine between its normal and the
        ray (the light sits at the lens), then the sensor's response, then
        noise.  The light falls off as the inverse square of the hit's
        distance, white at expose_mm filling EXPOSE.  A marker returns what
        it returns to its own lens."""
        import mujoco
        self.pose(a, b)
        c = self.build.cams[cam]
        uv = np.atleast_2d(np.asarray(uv, float))
        vec = c.ray_world(uv[:, 0], uv[:, 1])
        N = len(vec)
        gid = np.zeros(N, np.int32)
        hit = np.zeros(N)
        nrm = np.zeros(3 * N)
        mujoco.mj_multiRay(self.m, self.d, c.centre / 1000.0, vec.ravel(), LIT_GROUPS, 1, -1, gid, hit, nrm, N, -1)
        nrm = nrm.reshape(N, 3)
        rgb = np.zeros((N, 3))
        for i in range(N):
            if hit[i] < 0.0:
                continue
            g = gid[i]
            mat = self.m.geom_matid[g]
            if mat >= 0 and self.m.mat_emission[mat] > 0.0:
                rgb[i] = 1.0                                   # a marker: the brightest thing there is
                continue
            # a geom's own rgba wins over its material's unless it is MuJoCo's default
            own = self.m.geom_rgba[g]
            col = self.m.mat_rgba[mat, :3] if mat >= 0 and np.allclose(own, (0.5, 0.5, 0.5, 1.0)) else own[:3]
            rgb[i] = col * max(0.0, float(-nrm[i] @ vec[i])) * (self.d_exp / (hit[i] * 1000.0)) ** 2
        raw = EXPOSE * self.build.light[cam] * rgb @ self.build.colour[cam].T
        return _sensor(raw, self.rng)


def _sensor(raw, rng):
    """Light (1.0 the full well) to what the sensor reads: shot noise and
    read noise, on the sensor's black level, cut to BITS and clipped, the
    black level taken off again -- so a dark pixel reads its noise either
    side of zero, and 1.0 is still the full well."""
    e = np.clip(raw, 0.0, None) * RingCam.FULL_WELL
    e = e + rng.normal(0.0, 1.0, e.shape) * np.sqrt(e + RingCam.READ_NOISE ** 2)
    top = 2 ** RingCam.BITS - 1
    span = top - RingCam.BLACK_LEVEL
    code = np.clip(np.round(e / RingCam.FULL_WELL * span + RingCam.BLACK_LEVEL), 0, top)
    return (code - RingCam.BLACK_LEVEL) / span


# ================================================================ PIXELS
def warp_lut(lens, f_canvas, Wc, Hc, step=8):
    """(U, V) float32 (H, W): where on the canvas each sensor pixel's centre
    looks, through `lens` -- the true one.  The undistortion is solved on
    every step-th pixel and interpolated between: the map is smooth, and
    its curvature over 8 px moves it by well under a thousandth of a pixel."""
    W, H = lens.W, lens.H
    ju = np.append(np.arange(0, W, step), W - 1) + 0.5
    iv = np.append(np.arange(0, H, step), H - 1) + 0.5
    UU, VV = np.meshgrid(ju, iv)
    x, y = lens.ray(UU, VV)
    Uc = f_canvas * x + Wc / 2.0
    Vc = f_canvas * y + Hc / 2.0
    cols, rows = np.arange(W) + 0.5, np.arange(H) + 0.5

    def interp(G):
        A = np.empty((len(iv), W))
        for i in range(len(iv)):
            A[i] = np.interp(cols, ju, G[i])
        out = np.empty((H, W))
        for j in range(W):
            out[:, j] = np.interp(rows, iv, A[:, j])
        return out.astype(np.float32)

    return interp(Uc), interp(Vc)


def remap(img, U, V):
    """Bilinear sample of img (h, w[, c]) at continuous canvas coordinates
    (U, V), pixel centres at +0.5; outside the canvas is black."""
    h, w = img.shape[:2]
    x, y = U - 0.5, V - 0.5
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    fx, fy = x - x0, y - y0
    inside = (x0 >= 0) & (y0 >= 0) & (x0 < w - 1) & (y0 < h - 1)
    x0c, y0c = np.clip(x0, 0, w - 2), np.clip(y0, 0, h - 2)
    if img.ndim == 3:
        fx, fy, inside = fx[..., None], fy[..., None], inside[..., None]
    a = img[y0c, x0c]
    b = img[y0c, x0c + 1]
    c = img[y0c + 1, x0c]
    d = img[y0c + 1, x0c + 1]
    out = (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy
    return np.where(inside, out, 0.0)


def blobs(img, level, noise):
    """(N, 3) u, v, area of the bright blobs in a marker frame (values 0-1):
    every 4-connected group of pixels above half of `level`, each centroid
    weighted by how much of each pixel the ball fills -- (value - ground) /
    (plateau - ground), clipped, over the blob and two pixels round it,
    since an edge pixel is part ball; the ground is the median just outside
    that, the plateau the median of the blob's interior."""
    from ..station.vision import components, _dilate
    hot = img > 0.5 * level
    out = []
    H, W = img.shape
    for area, (i0, i1, j0, j1), rl in components(hot):
        if area < 2:
            continue
        a0, a1, b0, b1 = max(i0 - 5, 0), min(i1 + 6, H), max(j0 - 5, 0), min(j1 + 6, W)
        win = img[a0:a1, b0:b1]
        mask = np.zeros(win.shape, bool)
        for i, s, e in rl:
            mask[i - a0, s - b0:e - b0] = True
        core = mask & ~_dilate(~mask, 1)
        plateau = float(np.median(win[core])) if core.sum() >= 3 else float(win[mask].max())
        ring = _dilate(mask, RING)
        outside = ~_dilate(mask, RING + 1)
        ground = float(np.median(win[outside])) if outside.any() else 0.0
        wgt = np.clip((win - ground) / max(plateau - ground, 3.0 * noise), 0.0, 1.0) * ring
        ii, jj = np.nonzero(ring)
        ww = wgt[ii, jj]
        s = ww.sum()
        out.append((b0 + float((ww * (jj + 0.5)).sum() / s), a0 + float((ww * (ii + 0.5)).sum() / s), float(s)))
    return np.array(out).reshape(-1, 3)


class PixelVision(RigVisionHAL, _Scene):
    """One renderer, one process (MuJoCo's rule here): the check runs this
    in a worker of its own."""

    def __init__(self, rig, site, cams, build, payload="bat", rng=None):
        _Scene.__init__(self, rig, site, cams, build, payload)
        import mujoco
        self.rng = rng if rng is not None else np.random.default_rng(1)
        Wc, Hc = self.info["canvas"]
        self.Wc, self.Hc = Wc, Hc
        self.renderer = mujoco.Renderer(self.m, Hc, Wc)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        for g in (RM.STRUCT, RM.BAT, RM.BALL, RM.STEM):
            self.opt.geomgroup[g] = 1
        self.luts = [warp_lut(c.lens, f, Wc, Hc) for c, f in zip(build.cams, self.info["f_canvas"])]
        self.retro = [i for i in range(self.m.nmat) if self.m.mat_emission[i] > 0.0]
        self.base = self.m.light_diffuse.copy()
        self.d_exp = expose_mm(rig)
        self._img = {}
        self._col = None
        span = 2 ** RingCam.BITS - 1 - RingCam.BLACK_LEVEL
        # one pixel's noise at the background, in the image's units
        self.noise = sqrt(RingCam.READ_NOISE ** 2) / RingCam.FULL_WELL + 0.5 / span
        self.last = None

    def close(self):
        self.renderer.close()

    def _render(self, cam, retro, light):
        """The canvas picture (uint8) from camera `cam` (as built), lit by
        its own light alone at `light` x its build's output, markers at
        `retro`."""
        m = self.m
        m.light_active[:] = 0
        m.light_active[cam] = 1
        m.light_diffuse[cam] = light * (self.d_exp / 1000.0) ** 2    # its fall-off is in metres
        for i in self.retro:
            m.mat_emission[i] = retro
        self.renderer.update_scene(self.d, camera=self.build.cams[cam].name, scene_option=self.opt)
        return self.renderer.render()

    def _colour_canvas(self, cam):
        """The colour exposure's canvas (float, 1.0 the canvas's top), fine
        enough for the sensor: the renderer writes 8 bits, coarser than the
        sensor's 10 in a dark channel (a 1 % green under the light reads a
        few counts), so it is drawn twice, the second time BRACKET times
        brighter, and each pixel's channel taken from the bright one unless
        that one clipped."""
        lo = self._render(cam, 1.0, EXPOSE).astype(np.float32) / 255.0
        hi = self._render(cam, 1.0, EXPOSE * BRACKET).astype(np.float32) / 255.0
        return np.where(hi < 1.0, hi / BRACKET, lo)

    def floor(self, cam, raw):
        """(N, 3): the variance the canvas's 8-bit step adds to colour reads
        `raw` of camera `cam`, beyond the sensor's own noise -- a renderer's
        floor, not a camera's, the same in every pixel of a flat patch, so
        a check states it rather than expect the sensor's noise alone."""
        raw = np.atleast_2d(np.asarray(raw, float))
        M = self.build.light[cam] * np.asarray(self.build.colour[cam])
        c = np.clip(raw, 0.0, None) @ np.linalg.inv(M).T                # the canvas each read came from
        step = np.where(c < 1.0 / BRACKET, 1.0 / BRACKET, 1.0) / (2 ** CANVAS_BITS - 1)
        return (step ** 2 / 12.0) @ (M ** 2).T

    def frame(self, a, b, cam, mode="marker"):
        """The sensor's frame (H, W) or (H, W, 3), 0-1, as it reads it.

        A marker frame is warped only round what is bright on the canvas
        -- each bright canvas pixel sent through the true lens to the
        sensor, and HALO sensor pixels round it -- since everything else
        on it is the diffuse world RETRO_GAIN down, a few counts that no
        blob is found in and no centroid reads (blobs takes its ground
        from just outside each blob, inside the warped region)."""
        key = (a, b, cam, mode)
        if key in self._img:
            return self._img[key]
        self.pose(a, b)
        k = self.build.light[cam]
        c = self.build.cams[cam]
        L = c.lens
        U, V = self.luts[cam]
        if mode == "marker":
            raw = self._render(cam, EXPOSE, EXPOSE / Marker.RETRO_GAIN)
            ii, jj = np.nonzero(raw[..., 1] > HOT * EXPOSE * 255.0)
            img = np.zeros((L.H, L.W))
            if len(ii):
                f = self.info["f_canvas"][cam]
                u, v = L.pixel((jj + 0.5 - self.Wc / 2.0) / f, (ii + 0.5 - self.Hc / 2.0) / f)
                ok = (u >= 0) & (u < L.W) & (v >= 0) & (v < L.H)
                mask = np.zeros((L.H, L.W), bool)
                mask[v[ok].astype(int), u[ok].astype(int)] = True
                from ..station.vision import _dilate
                ri, rj = np.nonzero(_dilate(mask, HALO))
                g = raw.astype(np.float32).mean(axis=2) / 255.0
                img[ri, rj] = _sensor(k * remap(g, U[ri, rj], V[ri, rj]), self.rng)
        else:
            canvas = self._colour_canvas(cam)
            rgb = remap(canvas, U, V)
            img = _sensor(k * rgb @ self.build.colour[cam].T, self.rng)
        self._img = {key: img}
        return img

    def snap(self, a, b):
        out = []
        for k in range(len(self.cams)):
            img = self.frame(a, b, k, "marker")
            out.append(blobs(img, EXPOSE, self.noise))
        self.last = img
        return out

    def colour(self, a, b, cam, uv):
        """Bilinear between the four sensor pixels round each point; only
        those pixels are warped and read (each once a frame, so a pixel
        read twice has one noise), since a colour read samples a few
        hundred of the sensor's two million."""
        uv = np.atleast_2d(np.asarray(uv, float))
        key = (a, b, cam)
        if self._col is None or self._col[0] != key:
            self.pose(a, b)
            self._col = (key, self._colour_canvas(cam), {})
        _, canvas, done = self._col
        L = self.build.cams[cam].lens
        x, y = uv[:, 0] - 0.5, uv[:, 1] - 0.5
        j0 = np.clip(np.floor(x).astype(int), 0, L.W - 2)
        i0 = np.clip(np.floor(y).astype(int), 0, L.H - 2)
        fx, fy = (x - j0)[:, None], (y - i0)[:, None]
        U, V = self.luts[cam]
        corners = [(i0, j0), (i0, j0 + 1), (i0 + 1, j0), (i0 + 1, j0 + 1)]
        flat = np.unique(np.concatenate([i * L.W + j for i, j in corners]))
        new = np.array([f for f in flat if f not in done], int)
        if len(new):
            ii, jj = np.divmod(new, L.W)
            rgb = remap(canvas, U[ii, jj], V[ii, jj])
            val = _sensor(self.build.light[cam] * rgb @ self.build.colour[cam].T, self.rng)
            done.update(zip(new.tolist(), val))
        px = [np.array([done[int(i * L.W + j)] for i, j in zip(ci, cj)]) for ci, cj in corners]
        return (px[0] * (1 - fx) + px[1] * fx) * (1 - fy) + (px[2] * (1 - fx) + px[3] * fx) * fy
