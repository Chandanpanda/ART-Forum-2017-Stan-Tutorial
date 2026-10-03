"""The gimbal's motion and what it does to a point on the clamp, in closed
form.  numpy only: line.py times a plan with it, the fit models the IMU
with it, and check_imu holds it to MuJoCo.

UNITS.  Everything here is SI -- metres, seconds, radians, kg -- because
the IMU reads m/s^2 and rad/s.  The rig's modules work in millimetres and
degrees; callers convert at the boundary.

PATHS.  Each hinge moves along a piecewise-constant acceleration: holds,
trapezoids (truss.motion's, the same law the gantry runs) and constant-rate
spins.  A Path is built a segment at a time and then evaluated exactly --
angle, rate and acceleration at any instant -- so the plan, the
simulation and the fit all read the one profile.  Where an acceleration
changes is a corner; the stepper rings after each (settle()).

TWO CHAINS.  The gimbal is two hinges in series, written two ways:
  Chain   as MuJoCo builds it (rig/mjcf.py): the outer body on its hinge at
          the centre, the inner body on its hinge at an offset in the outer.
          Its parameters are the build's (RigBuildDraw), so only the
          simulation and the checks hold a true one; the planner holds the
          drawing's.  rnea() runs the dynamics on it.
  Kin     the product of exponentials (Brockett 1984; Park 1994): each
          hinge a unit twist in the world at the encoders' zero, and the
          clamp's pose there, T(q) = exp([xi1] q1) exp([xi2] q2) M0.  The
          encoders' zeros are not parameters -- M0 absorbs them -- so
          the model is minimal: 4 + 4 + 6 = 14 parameters (Okamura & Park
          1996).  This is what the station fits from the cameras' still
          poses (fit_kin), in the cameras' world, whatever that is.

THE POINT ON THE CLAMP.  For a point r fixed in the clamp (its frame:
the inner body's), the specific force an accelerometer there reads is
  f = R^T (a_O - g) + w' x r + w x (w x r)
in the clamp frame, with a_O the clamp origin's acceleration in the world,
w and w' the clamp's angular rate and acceleration in its own frame, and
g gravity's acceleration (pointing down).  A gyro there reads w plus the
Earth's turn, R^T Omega.
"""
from dataclasses import dataclass, field
from math import sin, cos, radians, sqrt, pi

import numpy as np

from truss.motion import trap_time

E = np.eye(3)


# ================================================================== SO(3)
def skew(v):
    v = np.asarray(v, float)
    O = np.zeros(v.shape[:-1] + (3, 3))
    O[..., 0, 1], O[..., 0, 2] = -v[..., 2], v[..., 1]
    O[..., 1, 0], O[..., 1, 2] = v[..., 2], -v[..., 0]
    O[..., 2, 0], O[..., 2, 1] = -v[..., 1], v[..., 0]
    return O


def exp_so3(phi):
    """Rotation matrices (..., 3, 3) of rotation vectors (..., 3)."""
    phi = np.asarray(phi, float)
    th = np.linalg.norm(phi, axis=-1)[..., None, None]
    K = skew(phi)
    small = th < 1e-6
    ths = np.where(small, 1.0, th)
    A = np.where(small, 1.0 - th ** 2 / 6.0, np.sin(ths) / ths)
    B = np.where(small, 0.5 - th ** 2 / 24.0, (1.0 - np.cos(ths)) / ths ** 2)
    return E + A * K + B * (K @ K)


def log_so3(R):
    """Rotation vectors of rotation matrices, stable near 0 and near pi."""
    R = np.asarray(R, float)
    tr = np.trace(R, axis1=-2, axis2=-1)
    c = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(c)
    w = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.sin(th)
    fac = np.where(th < 1e-6, 0.5 + th ** 2 / 12.0, th / (2.0 * np.where(s == 0.0, 1.0, s)))
    out = w * fac[..., None]
    near = th > pi - 1e-3                # sin(th) ~ 0: read the axis off R + R^T
    if np.any(near):
        Rn = R[near]
        thn = th[near]
        Bn = (Rn + np.swapaxes(Rn, -1, -2)) / 2.0 - np.cos(thn)[:, None, None] * E
        ax = np.empty((len(Rn), 3))
        for k in range(len(Rn)):
            d = np.diag(Bn[k])
            i = int(np.argmax(d))
            v = Bn[k][:, i] / sqrt(max(d[i], 1e-300))
            if v @ w[near][k] < 0.0:
                v = -v
            ax[k] = v / np.linalg.norm(v)
        out[near] = ax * thn[:, None]
    return out


def rot(axis, ang):
    """Rotations (n, 3, 3) about one fixed unit axis by angles (n,)."""
    ang = np.asarray(ang, float)
    K = skew(np.asarray(axis, float))
    s, c = np.sin(ang)[..., None, None], np.cos(ang)[..., None, None]
    return E + s * K + (1.0 - c) * (K @ K)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def perp_basis(w):
    """Two unit vectors square to w and to each other."""
    w = unit(w)
    a = E[int(np.argmin(np.abs(w)))]
    u = unit(np.cross(w, a))
    return u, np.cross(w, u)


# ================================================================== PATHS
@dataclass
class Axis:
    """One hinge's path: piece k starts at t[k] at angle q[k] (rad) and rate
    v[k] (rad/s) and keeps acceleration a[k] (rad/s^2) until t[k+1]."""
    t: list = field(default_factory=list)
    q: list = field(default_factory=list)
    v: list = field(default_factory=list)
    a: list = field(default_factory=list)

    def at(self, t):
        t = np.asarray(t, float)
        T = np.asarray(self.t)
        k = np.clip(np.searchsorted(T, t, side="right") - 1, 0, len(T) - 1)
        dt = t - T[k]
        q, v, a = np.asarray(self.q)[k], np.asarray(self.v)[k], np.asarray(self.a)[k]
        return q + v * dt + 0.5 * a * dt * dt, v + a * dt, a


@dataclass
class Segment:
    """What a stretch of the path is for: kind 'still', 'move' or 'spin'."""
    kind: str
    t0: float
    t1: float
    q0: tuple                  # rad, both hinges at its start
    q1: tuple                  # and at its end
    hinge: int = -1            # a spin's
    rate: float = 0.0          # rad/s, a spin's cruise
    tag: str = ""


class Path:
    """A two-hinge path built a segment at a time from rest at q0 (rad)."""

    def __init__(self, q0):
        self.axes = (Axis(), Axis())
        self.q = np.array(q0, float)
        self.T = 0.0
        self.segments = []

    def _piece(self, h, t, q, v, a):
        ax = self.axes[h]
        if ax.t and abs(ax.t[-1] - t) < 1e-12:
            ax.q[-1], ax.v[-1], ax.a[-1] = q, v, a
        else:
            ax.t.append(t)
            ax.q.append(q)
            ax.v.append(v)
            ax.a.append(a)

    def _rest(self):
        for h in (0, 1):
            self._piece(h, self.T, float(self.q[h]), 0.0, 0.0)

    def hold(self, dur, tag=""):
        q = tuple(self.q)
        self._rest()
        self.segments.append(Segment("still", self.T, self.T + dur, q, q, tag=tag))
        self.T += dur
        return self

    def move(self, q1, vmax, amax, tag=""):
        """Rest to rest to q1, both hinges arriving together: each runs its
        own trapezoid stretched to the slower one's time (truss.motion.Move)."""
        q1 = np.array(q1, float)
        d = q1 - self.q
        Tk = [trap_time(d[h], vmax[h], amax[h]) for h in (0, 1)]
        Tm = max(Tk)
        q0 = tuple(self.q)
        if Tm <= 0.0:
            return self
        for h in (0, 1):
            dist = abs(d[h])
            s = Tk[h] / Tm if Tm > 0 else 0.0
            sg = 1.0 if d[h] >= 0.0 else -1.0
            t0, qh = self.T, float(self.q[h])
            if dist < 1e-12 or s == 0.0:
                self._piece(h, t0, qh, 0.0, 0.0)
                continue
            a = amax[h] * s * s
            if dist <= vmax[h] ** 2 / amax[h]:           # a triangle
                th = Tm / 2.0
                vp = a * th
                self._piece(h, t0, qh, 0.0, sg * a)
                self._piece(h, t0 + th, qh + sg * 0.5 * a * th * th, sg * vp, -sg * a)
            else:
                ta = vmax[h] / amax[h] / s
                vc = vmax[h] * s
                self._piece(h, t0, qh, 0.0, sg * a)
                self._piece(h, t0 + ta, qh + sg * 0.5 * a * ta * ta, sg * vc, 0.0)
                self._piece(h, t0 + Tm - ta, float(q1[h]) - sg * 0.5 * a * ta * ta, sg * vc, -sg * a)
        self.T += Tm
        self.q = q1
        self.segments.append(Segment("move", self.T - Tm, self.T, q0, tuple(q1), tag=tag))
        self._rest()
        return self

    def spin(self, h, turns, w, amax, tag=""):
        """One hinge through `turns` (signed) at cruise rate w (rad/s), from
        rest to rest, ramping at amax; the other held."""
        d = 2.0 * pi * turns
        sg = 1.0 if d >= 0.0 else -1.0
        dist = abs(d)
        ta = w / amax
        if dist <= w * ta:
            raise ValueError("a spin of %.2f turns never reaches %.1f deg/s at %.1f deg/s^2"
                             % (turns, np.degrees(w), np.degrees(amax)))
        tc = (dist - w * ta) / w
        q0 = tuple(self.q)
        t0, qh = self.T, float(self.q[h])
        self._piece(h, t0, qh, 0.0, sg * amax)
        self._piece(h, t0 + ta, qh + sg * 0.5 * w * ta, sg * w, 0.0)
        self._piece(h, t0 + ta + tc, qh + sg * (0.5 * w * ta + w * tc), sg * w, -sg * amax)
        o = 1 - h
        self._piece(o, t0, float(self.q[o]), 0.0, 0.0)
        Tm = 2.0 * ta + tc
        self.T += Tm
        self.q = self.q.copy()
        self.q[h] = qh + d
        self.segments.append(Segment("spin", t0, self.T, q0, tuple(self.q), hinge=h, rate=sg * w, tag=tag))
        self._rest()
        return self

    def state(self, t):
        """(q, qd, qdd), each (n, 2): both hinges at times t."""
        out = [ax.at(t) for ax in self.axes]
        return tuple(np.stack([out[0][k], out[1][k]], -1) for k in range(3))

    def corners(self):
        """[(t, (da_outer, da_inner))]: where either hinge's acceleration
        jumps, and by how much (rad/s^2)."""
        ev = {}
        for h, ax in enumerate(self.axes):
            a_prev = 0.0
            for t, a in zip(ax.t, ax.a):
                if abs(a - a_prev) > 0.0:
                    key = round(t, 9)
                    ev.setdefault(key, np.zeros(2))[h] += a - a_prev
                a_prev = a
            if abs(a_prev) > 0.0:
                ev.setdefault(round(self.T, 9), np.zeros(2))[h] -= a_prev
        return sorted(ev.items())


# ================================================================ CHAINS
@dataclass
class Chain:
    """The gimbal as MuJoCo builds it, SI: the outer body at `centre` on a
    hinge about axis_o (world); the inner body at offset_i in the outer
    body on a hinge about axis_i (outer frame); each hinge at its encoder
    reading plus `zero` (rad)."""
    centre: np.ndarray
    axis_o: np.ndarray
    axis_i: np.ndarray
    offset_i: np.ndarray
    zero: np.ndarray

    @classmethod
    def drawn(cls):
        return cls(np.zeros(3), E[0].copy(), E[1].copy(), np.zeros(3), np.zeros(2))

    @classmethod
    def of(cls, centre_mm, axis_o, axis_i, offset_mm, zero_deg):
        """From a build's own units (rig/mjcf.RigBuildDraw: mm and deg)."""
        return cls(np.asarray(centre_mm, float) / 1000.0, unit(axis_o), unit(axis_i),
                   np.asarray(offset_mm, float) / 1000.0, np.radians(np.asarray(zero_deg, float)))

    def kin(self):
        """The same gimbal as a product of exponentials in encoder angles."""
        R1z = rot(self.axis_o, self.zero[0])
        w1, p1 = self.axis_o.copy(), self.centre.copy()
        w2 = R1z @ self.axis_i
        p2 = self.centre + R1z @ self.offset_i
        R0 = R1z @ rot(self.axis_i, self.zero[1])
        return Kin(w1, p1, w2, p2, R0, p2.copy())


@dataclass
class Kin:
    """The product-of-exponentials model of the gimbal in some world frame,
    SI: hinge k turns about unit w_k through point p_k (as they lie with
    both encoders at zero); M0 = (R0, t0) is the clamp's pose there."""
    w1: np.ndarray
    p1: np.ndarray
    w2: np.ndarray
    p2: np.ndarray
    R0: np.ndarray
    t0: np.ndarray

    def pose(self, q):
        """(R (n,3,3), t (n,3)): the clamp's pose at encoder angles q (n,2)."""
        q = np.atleast_2d(q)
        R1, R2 = rot(self.w1, q[:, 0]), rot(self.w2, q[:, 1])
        u = np.einsum("nij,j->ni", R2, self.t0 - self.p2)
        x = self.p1 + np.einsum("nij,nj->ni", R1, self.p2 - self.p1 + u)
        return R1 @ R2 @ self.R0, x

    def motion(self, q, qd, qdd):
        """(R, x, w_c, wd_c, a_O): pose, the clamp's rate and angular
        acceleration in its own frame, and its origin's acceleration in the
        world, at hinge states (n, 2) each."""
        q, qd, qdd = (np.atleast_2d(v) for v in (q, qd, qdd))
        R1, R2 = rot(self.w1, q[:, 0]), rot(self.w2, q[:, 1])
        R = R1 @ R2 @ self.R0
        u = np.einsum("nij,j->ni", R2, self.t0 - self.p2)
        r1 = np.einsum("nij,nj->ni", R1, self.p2 - self.p1 + u)
        x = self.p1 + r1
        w2w = np.einsum("nij,j->ni", R1, self.w2)
        om1 = qd[:, :1] * self.w1
        om = om1 + qd[:, 1:] * w2w
        al = qdd[:, :1] * self.w1 + qdd[:, 1:] * w2w + qd[:, 1:] * np.cross(om1, w2w)
        du = qd[:, 1:] * np.cross(self.w2, u)
        ud = np.einsum("nij,nj->ni", R1, du)
        ddu = qdd[:, 1:] * np.cross(self.w2, u) + qd[:, 1:] ** 2 * np.cross(self.w2, np.cross(self.w2, u))
        a = (qdd[:, :1] * np.cross(self.w1, r1) + np.cross(om1, np.cross(om1, r1)) + 2.0 * np.cross(om1, ud)
             + np.einsum("nij,nj->ni", R1, ddu))
        Rt = np.swapaxes(R, 1, 2)
        return R, x, np.einsum("nij,nj->ni", Rt, om), np.einsum("nij,nj->ni", Rt, al), a

    # --- the minimal parameters, as a local perturbation, for fit_kin
    def perturb(self, d):
        """The model moved by d (14,): each axis turned by (2) and its point
        shifted by (2) square to it, then the clamp's pose at zero by a
        rotation and a shift (6), all in world axes."""
        d = np.asarray(d, float)
        out = []
        for w, p, k in ((self.w1, self.p1, 0), (self.w2, self.p2, 4)):
            u, v = perp_basis(w)
            wn = rot(u, [d[k]])[0] @ (rot(v, [d[k + 1]])[0] @ w)
            pn = p + d[k + 2] * u + d[k + 3] * v
            out += [unit(wn), pn]
        R0 = exp_so3(d[8:11]) @ self.R0
        return Kin(out[0], out[1], out[2], out[3], R0, self.t0 + d[11:14])


def pose_residual(kin, q, R_obs, t_obs):
    """(n, 6): the model's pose against observed ones, [dphi (world axes,
    left), dt (m)]: R_model = exp(dphi) R_obs."""
    R, t = kin.pose(q)
    return np.concatenate([log_so3(R @ np.swapaxes(R_obs, 1, 2)), t - t_obs], axis=1)


def fit_kin(q, R_obs, t_obs, cov, start, iters=30, tol=1e-12):
    """The gimbal's 14 kinematic parameters from still poses: hinge readings
    q (n, 2) rad, observed poses R_obs (n,3,3) and t_obs (n,3) m with their
    covariances cov (n,6,6) in [rad, m].  Gauss-Newton from `start` (the
    drawing will do: it converges from a degree and a few mm).  Returns
    (Kin, Sigma (14,14) of the perturbation at the solution, chi2, dof)."""
    n = len(q)
    Wh = np.linalg.cholesky(np.linalg.inv(cov))          # W = L L^T: whitened r = L^T r
    kin = start
    eps = 1e-7

    def white(k):
        r = pose_residual(k, q, R_obs, t_obs)
        return np.einsum("nji,nj->ni", Wh, r).ravel()

    r = white(kin)
    for _ in range(iters):
        J = np.empty((r.size, 14))
        for j in range(14):
            dj = np.zeros(14)
            dj[j] = eps
            J[:, j] = (white(kin.perturb(dj)) - white(kin.perturb(-dj))) / (2.0 * eps)
        H = J.T @ J
        step = -np.linalg.solve(H, J.T @ r)
        kin = kin.perturb(step)
        r = white(kin)
        if np.max(np.abs(step)) < tol:
            break
    J = np.empty((r.size, 14))
    for j in range(14):
        dj = np.zeros(14)
        dj[j] = eps
        J[:, j] = (white(kin.perturb(dj)) - white(kin.perturb(-dj))) / (2.0 * eps)
    Sig = np.linalg.inv(J.T @ J)
    return kin, Sig, float(r @ r), 6 * n - 14


# ============================================================== DYNAMICS
@dataclass
class Inertial:
    """A rigid body's mass properties in its own frame, SI: mass, centre of
    mass, inertia tensor about the centre of mass."""
    m: float
    c: np.ndarray
    I: np.ndarray


def rnea(chain, outer, inner, q, qd, qdd, g, armature=(0.0, 0.0)):
    """(n, 2) N m: each hinge's torque to make the motion (q, qd, qdd) on
    Chain `chain` -- encoder angles -- with bodies `outer` and `inner`
    (Inertial, everything each carries) and gravity's acceleration g
    (world, pointing down): Newton-Euler, outward then inward (Featherstone
    2008, ch. 5), plus each motor's rotor reflected through its belt."""
    q, qd, qdd = (np.atleast_2d(v) for v in (q, qd, qdd))
    th = q + chain.zero
    R1 = rot(chain.axis_o, th[:, 0])
    R2 = R1 @ rot(chain.axis_i, th[:, 1])
    a1 = chain.axis_o
    a2 = np.einsum("nij,j->ni", R1, chain.axis_i)
    w1 = qd[:, :1] * a1
    al1 = qdd[:, :1] * a1
    w2 = w1 + qd[:, 1:] * a2
    al2 = al1 + qdd[:, 1:] * a2 + qd[:, 1:] * np.cross(w1, a2)
    off = np.einsum("nij,j->ni", R1, chain.offset_i)
    o2 = chain.centre + off
    a_o2 = np.cross(al1, off) + np.cross(w1, np.cross(w1, off))
    c1 = np.einsum("nij,j->ni", R1, outer.c)
    c2 = np.einsum("nij,j->ni", R2, inner.c)
    acc1 = np.cross(al1, c1) + np.cross(w1, np.cross(w1, c1))
    acc2 = a_o2 + np.cross(al2, c2) + np.cross(w2, np.cross(w2, c2))
    F1 = outer.m * (acc1 - g)
    F2 = inner.m * (acc2 - g)
    I1 = R1 @ outer.I @ np.swapaxes(R1, 1, 2)
    I2 = R2 @ inner.I @ np.swapaxes(R2, 1, 2)
    N1 = np.einsum("nij,nj->ni", I1, al1) + np.cross(w1, np.einsum("nij,nj->ni", I1, w1))
    N2 = np.einsum("nij,nj->ni", I2, al2) + np.cross(w2, np.einsum("nij,nj->ni", I2, w2))
    M2 = N2 + np.cross(c2, F2)                                   # about o2
    tau2 = np.einsum("ni,ni->n", a2, M2)
    M1 = N1 + np.cross(c1, F1) + M2 + np.cross(off, F2)           # about the centre
    tau1 = M1 @ a1
    return np.stack([tau1, tau2], -1) + np.asarray(armature) * qdd


def hinge_inertia(chain, outer, inner, q, armature=(0.0, 0.0)):
    """(n, 2) kg m^2: each hinge's own inertia at angles q, the other held
    -- the diagonal of the joint-space mass matrix, which is what a
    stepper's stiffness rings against."""
    q = np.atleast_2d(q)
    z = np.zeros_like(q)
    out = []
    for h in (0, 1):
        e = np.zeros_like(q)
        e[:, h] = 1.0
        out.append(rnea(chain, outer, inner, q, z, e, np.zeros(3), armature)[:, h])
    return np.stack(out, -1)


# ================================================================ EARTH
OMEGA_E = 7.292115e-5          # rad/s, the Earth's sidereal rate (IERS)


def gravity(lat_deg, alt_m):
    """m/s^2: normal gravity on the WGS 84 ellipsoid (Somigliana's closed
    form, NIMA TR8350.2 eq. 4-1) less the free-air gradient at height
    (0.3086 mGal/m)."""
    s2 = sin(radians(lat_deg)) ** 2
    g0 = 9.7803253359 * (1.0 + 0.00193185265241 * s2) / sqrt(1.0 - 0.00669437999013 * s2)
    return g0 - 3.086e-6 * alt_m


def north_rig(heading_deg):
    """Unit north in the rig frame (x the rig's axis, z up), the rig's x
    axis `heading_deg` east of north."""
    h = radians(heading_deg)
    return np.array([cos(h), sin(h), 0.0])


def earth_rate(lat_deg, up, north):
    """rad/s: the Earth's turn as a vector, given up and north (any frame;
    north is made square to up)."""
    up = unit(up)
    n = unit(north - up * (north @ up))
    phi = radians(lat_deg)
    return OMEGA_E * (cos(phi) * n + sin(phi) * up)


def specific_force(R, a_O, w_c, wd_c, r_c, g):
    """(n, 3) m/s^2 in the clamp frame at clamp point r_c (3,) or (n, 3)."""
    Rt = np.swapaxes(R, 1, 2)
    return (np.einsum("nij,nj->ni", Rt, a_O - g) + np.cross(wd_c, r_c)
            + np.cross(w_c, np.cross(w_c, r_c)))


# ================================================================ SETTLE
def ring(w_n, zeta):
    """(decay rate, damped frequency) rad/s of a hinge on its stepper's
    holding stiffness."""
    return zeta * w_n, w_n * sqrt(max(1.0 - zeta * zeta, 0.0))


def settle(da, w_n, zeta, tol):
    """s for a ring after an acceleration step da (rad/s^2) to fall below
    tol (rad).  The hinge follows its command through the stepper's
    stiffness as a second-order system: an acceleration step leaves an
    error da / w_n^2 that rings down at zeta w_n (its envelope is
    sqrt(1/(1 - zeta^2)) times that, at most)."""
    A = abs(da) / w_n ** 2 / sqrt(max(1.0 - zeta * zeta, 1e-12))
    if A <= tol:
        return 0.0
    return float(np.log(A / tol) / (zeta * w_n))


def ringing(corners, t, w_n, zeta):
    """(e, ed, edd) (n, 2): each hinge's lag behind its command at times t,
    the superposed response to every acceleration step so far (an error e
    with e'' + 2 zeta w_n e' + w_n^2 e = the command's acceleration, the
    damping the driver's, on the error).  corners as Path.corners(); w_n
    (2,) per hinge, or (len(corners), 2): each hinge's natural frequency
    as it stood at each corner."""
    t = np.asarray(t, float)
    e = np.zeros((len(t), 2))
    ed = np.zeros_like(e)
    edd = np.zeros_like(e)
    W = np.broadcast_to(np.asarray(w_n, float), (len(corners), 2))
    for h in (0, 1):
        for (tc, da), wn in zip(corners, W[:, h]):
            if da[h] == 0.0:
                continue
            s, wd = ring(wn, zeta)
            k = wn ** 2
            tau = t - tc
            m = tau > 0.0
            x = tau[m]
            ex, c, sn = np.exp(-s * x), np.cos(wd * x), np.sin(wd * x)
            A = da[h] / k
            e[m, h] += A * (1.0 - ex * (c + s / wd * sn))
            ed[m, h] += A * ex * (k / wd) * sn
            edd[m, h] += A * k * ex * (c - s / wd * sn)
    return e, ed, edd
