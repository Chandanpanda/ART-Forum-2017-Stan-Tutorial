"""A rig camera: a pinhole with Brown-Conrady distortion, as calibrated.

The model every part of the rig shares -- the placement solver, the model
camera, the rendered camera's warp, the bundle adjustment and the clamp's
pose -- so a pixel means the same thing everywhere.  It is OpenCV's model
(Brown 1966, Conrady 1919), because that is the model the real cameras'
calibration will be done in:

    x, y   = X_c / Z_c, Y_c / Z_c                      normalised, undistorted
    r2     = x^2 + y^2
    radial = 1 + k1 r2 + k2 r2^2 + k3 r2^3
    x_d    = x radial + 2 p1 x y + p2 (r2 + 2 x^2)
    y_d    = y radial + p1 (r2 + 2 y^2) + 2 p2 x y
    u, v   = fx x_d + cx, fy y_d + cy

Frames.  World points are in mm in the rig's frame (rig/spec.py).  A
camera's frame is computer vision's: x right, y DOWN the image, z forward
along the optical axis; X_c = R X_w + t.  MuJoCo's camera looks down its
-z with y up, so its xyaxes are (R[0], -R[1]) -- mjcf.py does that, and
nothing else needs to know.

Pixels are continuous: pixel (row i, column j) covers [j, j+1) x [i, i+1),
its centre is (j + 0.5, i + 0.5), and a centred lens's principal point is
(W/2, H/2) -- as in station/vision.py.

A ball's image.  A sphere of radius r whose centre is at C in the camera
frame images as an ellipse whose centre is NOT the projection of C: in
normalised coordinates it is C_xy C_z / (C_z^2 - r^2).  A retroreflective
ball seen by its own ring light is a uniformly bright disc, so its
centroid is that ellipse's centre.  project(..., radius=r) returns it; at
the rig's ranges the difference is a few hundredths of a pixel, which is
the size of the noise, so it is not left out.
"""
from dataclasses import dataclass, field, replace

import numpy as np

DIST = ("k1", "k2", "k3", "p1", "p2")


# ================================================================ SO(3)
def skew(v):
    v = np.asarray(v, float)
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rodrigues(r):
    """Rotation matrix of the rotation vector r (rad)."""
    r = np.asarray(r, float)
    th = float(np.linalg.norm(r))
    K = skew(r)
    if th < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1.0 - np.cos(th)) / th ** 2 * (K @ K)


def rvec(R):
    """The rotation vector of R (rad), stable at small angles and near pi."""
    R = np.asarray(R, float)
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = float(np.arccos(c))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-7:
        return w / 2.0
    if np.pi - th < 1e-4:
        # near pi: the axis from the symmetric part
        B = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(B)))
        a = B[:, i] / np.sqrt(max(B[i, i], 1e-300))
        a = a / np.linalg.norm(a)
        if np.dot(a, w) < 0:
            a = -a
        return a * th
    return w * th / (2.0 * np.sin(th))


def angle_between(Ra, Rb):
    """deg: the angle of the rotation that takes Ra to Rb."""
    c = np.clip((np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """R (world to camera) of a camera at `eye` looking at `target`, its
    image's x level with the world's horizontal where it can be."""
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    u = np.asarray(up, float)
    x = np.cross(z, u)               # right: forward x up, as a level camera has it
    if np.linalg.norm(x) < 1e-9:     # looking straight up or down
        x = np.cross(z, (1.0, 0.0, 0.0)) if abs(z[0]) < 0.9 else np.cross(z, (0.0, 1.0, 0.0))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)               # down the image
    return np.stack([x, y, z])


# ================================================================ LENS
@dataclass
class Lens:
    """Intrinsics in pixels and Brown-Conrady distortion, on a W x H sensor."""
    W: int
    H: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    @property
    def dist(self):
        return np.array([getattr(self, k) for k in DIST])

    def distort(self, x, y):
        """Normalised undistorted -> normalised distorted."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        r2 = x * x + y * y
        rad = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3))
        xd = x * rad + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x)
        yd = y * rad + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y
        return xd, yd

    def _ddistort(self, x, y):
        """The 2x2 Jacobian of distort, as (dxd/dx, dxd/dy, dyd/dx, dyd/dy)."""
        r2 = x * x + y * y
        rad = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3))
        drad = self.k1 + r2 * (2.0 * self.k2 + 3.0 * self.k3 * r2)     # d rad / d r2
        a = rad + 2.0 * x * x * drad + 2.0 * self.p1 * y + 6.0 * self.p2 * x
        b = 2.0 * x * y * drad + 2.0 * self.p1 * x + 2.0 * self.p2 * y
        c = 2.0 * x * y * drad + 2.0 * self.p1 * x + 2.0 * self.p2 * y
        d = rad + 2.0 * y * y * drad + 6.0 * self.p1 * y + 2.0 * self.p2 * x
        return a, b, c, d

    def undistort(self, xd, yd, iters=20):
        """Normalised distorted -> normalised undistorted, by Newton's
        method from the distorted point (it converges in a few steps
        anywhere the distortion is monotonic, which is the whole of any
        real lens's field)."""
        xd, yd = np.asarray(xd, float), np.asarray(yd, float)
        x, y = xd.copy(), yd.copy()
        for _ in range(iters):
            fx_, fy_ = self.distort(x, y)
            ex, ey = fx_ - xd, fy_ - yd
            a, b, c, d = self._ddistort(x, y)
            det = a * d - b * c
            x = x - (d * ex - b * ey) / det
            y = y - (-c * ex + a * ey) / det
            if np.max(np.abs(ex)) + np.max(np.abs(ey)) < 1e-14:
                break
        return x, y

    def to_pixels(self, xd, yd):
        return self.fx * np.asarray(xd) + self.cx, self.fy * np.asarray(yd) + self.cy

    def from_pixels(self, u, v):
        return (np.asarray(u, float) - self.cx) / self.fx, (np.asarray(v, float) - self.cy) / self.fy

    def pixel(self, x, y):
        """Normalised undistorted -> pixel."""
        return self.to_pixels(*self.distort(x, y))

    def ray(self, u, v):
        """Pixel -> normalised undistorted (the ray's x/z, y/z)."""
        return self.undistort(*self.from_pixels(u, v))

    def inside(self, u, v, margin=0.0):
        u, v = np.asarray(u), np.asarray(v)
        return (u >= margin) & (u <= self.W - margin) & (v >= margin) & (v <= self.H - margin)

    def half_fov(self):
        """(rad, rad): the undistorted half-angles to the middle of each
        edge of the sensor."""
        x0, _ = self.ray(0.0, self.cy)
        x1, _ = self.ray(float(self.W), self.cy)
        _, y0 = self.ray(self.cx, 0.0)
        _, y1 = self.ray(self.cx, float(self.H))
        return (float(np.arctan(min(abs(x0), abs(x1)))), float(np.arctan(min(abs(y0), abs(y1)))))

    def params(self):
        return np.array([self.fx, self.fy, self.cx, self.cy] + list(self.dist))

    def with_params(self, p):
        p = np.asarray(p, float)
        return replace(self, fx=p[0], fy=p[1], cx=p[2], cy=p[3], k1=p[4], k2=p[5], k3=p[6], p1=p[7], p2=p[8])


# ============================================================== CAMERA
@dataclass
class Camera:
    """A lens at a pose: X_c = R X_w + t (mm)."""
    name: str
    lens: Lens
    R: np.ndarray
    t: np.ndarray
    meta: dict = field(default_factory=dict)

    @classmethod
    def at(cls, name, lens, centre, R, **meta):
        R = np.asarray(R, float)
        return cls(name, lens, R, -R @ np.asarray(centre, float), dict(meta))

    @property
    def centre(self):
        return -self.R.T @ self.t

    @property
    def axis(self):
        """The optical axis, in the world."""
        return self.R[2].copy()

    def to_cam(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        return X @ self.R.T + self.t

    def project(self, X, radius=0.0):
        """(u, v, z) of world points X (N x 3, mm): the pixel of each point,
        or of the centre of a ball of `radius` around it, and its depth.
        Points behind the camera come back with z <= 0; it is the caller's
        to drop them."""
        C = self.to_cam(X)
        z = C[:, 2]
        zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
        if radius:
            s = zs / np.maximum(zs * zs - radius * radius, 1e-9)
        else:
            s = 1.0 / zs
        x, y = C[:, 0] * s, C[:, 1] * s
        u, v = self.lens.pixel(x, y)
        return u, v, z

    def sees(self, X, radius=0.0, margin=0.0):
        """Which world points land on the sensor in front of the camera."""
        u, v, z = self.project(X, radius)
        return (z > radius) & self.lens.inside(u, v, margin)

    def ray_world(self, u, v):
        """Unit world directions of pixels (u, v)."""
        x, y = self.lens.ray(u, v)
        d = np.stack([np.atleast_1d(x), np.atleast_1d(y), np.ones(np.size(x))], axis=1)
        d = d @ self.R                       # = (R^T d^T)^T
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    # ------------------------------------------- bundle adjustment's view
    def params(self):
        """[rvec(3), t(3), fx, fy, cx, cy, k1, k2, k3, p1, p2]: 15."""
        return np.concatenate([rvec(self.R), self.t, self.lens.params()])

    def with_params(self, p):
        p = np.asarray(p, float)
        return Camera(self.name, self.lens.with_params(p[6:]), rodrigues(p[:3]), p[3:6].copy(), dict(self.meta))


N_PARAMS = 15


def distortion_px(true, est, pts=None, n=41):
    """px: the worst difference between two lenses' pixels for the same
    rays, over `pts` (normalised undistorted points) or a grid over the
    true lens's sensor.  How far apart two calibrations put a point."""
    if pts is None:
        us, vs = np.meshgrid(np.linspace(0, true.W, n), np.linspace(0, true.H, n))
        x, y = true.ray(us.ravel(), vs.ravel())
    else:
        x, y = pts
    u1, v1 = true.pixel(x, y)
    u2, v2 = est.pixel(x, y)
    return float(np.max(np.hypot(u1 - u2, v1 - v2)))
