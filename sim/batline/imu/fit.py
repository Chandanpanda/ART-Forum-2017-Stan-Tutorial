"""The bat's IMU calibration at station 4, from what the station has: the
board's stream, the gimbal's encoders and the cameras' still poses.  numpy
only; the planner (plan.py) designs the plan against this same model.

WHAT IS MEASURED, AND WHEN.  The plan (plan.py) turns the gimbal through
still poses and single-hinge spins.  A move rings the steppers; the plan
waits until the ringing is below what the windows can see and calls the
rest a QUIET interval: a still, or a spin's cruise at constant rate.  Only
quiet intervals are used; moves and ramps are not.  In each:
  still   the whole interval's mean of every channel: one observation
  spin    means over windows of W samples (part.window_noise sizes W so
          neighbouring windows are nearly independent)

THE MODEL.  In the clamp's frame (the gimbal's inner body), for a window's
samples at station times t_k = T0 + rho k / ODR - delay (the board's own
clock, read through its output filter's delay):

  y_a = M_a (f0 + A r) + b_a + c_a dT
  y_g = M_g (w + R^T Omega) + G' (f0 + A r) + b_g + c_g dT

  f0 = R^T (a_O + g u)   the clamp origin's specific force, u gravity's up
  A  = [w']x + [w]x^2    so f0 + A r is the specific force at the IMU, r
  w, w'                  the clamp's rate and angular acceleration
  R, a_O                 its attitude and its origin's acceleration
each averaged over the window's samples.  The gimbal's kinematics (kin.Kin,
14 parameters) give R, w, w', a_O from the encoders; Omega is the Earth's
turn and g the site's gravity (spec.Site), dT the die's temperature less
25 C from the stream's own temperature codes.

M_a and M_g are full 3x3 maps from the clamp's frame to the chip's
outputs.  Fitted as such, the model is linear in them -- an analytic
Jacobian, a closed-form start -- and the IMU's mounting is read out of
them afterwards (terms()): M_a^T = U H, the polar decomposition, gives the
chip's axes U (the mounting) and its symmetric scale and cross-axis
H - I; the gyro's is M_g U - I.  The polar split depends on no order of
axes, as an RQ split would.

THE TWO STAGES.
  1  the gimbal's kinematics from the cameras' still poses and the
     encoders there (kin.fit_kin), in the cameras' world, with the
     cameras' own covariance.
  2  the 46 parameters below by Gauss-Newton, together with a refinement
     dk of the gimbal's 14 kinematic parameters whose prior is stage 1's
     estimate and covariance (a sequential adjustment: the cameras' fit
     enters as a prior, not as a fact).  The IMU sees the hinges' axes far
     better than a few poses do, and a kinematic error held fixed is not
     a constant rotation the maps absorb: a hinge's axis off by its
     cameras' sigma turns gravity about the wrong axis in every spin of
     that hinge, a first harmonic in the forces across it that no IMU
     term takes up.  The model is linear in dk (its Jacobian on the
     windows' centres, Model.kin_slopes, renewed with the clock), which is
     exact to its square: dk is a few hundredths of a degree.
Priors only on nuisances: the gyro's g-sensitivity, the temperature
coefficients, the clock's rate and its start (each its datasheet's
spread), and the gimbal's refinement (the cameras').  The targets rest on
data alone.

NOISE.  Each bat measures its own, as pure error: the scatter of
W-sample means about what any steady signal there must follow -- a
still's straight line in time, a spin's line plus harmonics of the spun
angle (every term of a steady spin turns with the spun angle: SPIN
MOTION) -- with no calibration term in it.  The outputs are whole codes,
and the gyro's noise is too small to dither the rounding, so a mean's
variance depends on where its true value falls between two codes
(part.code_var).  Where a still's true rate falls half way between two
codes its codes toggle between the two whatever the noise, and the
stills alone cannot measure it; a spin's signal sweeps across the codes
and can.  So the noise is solved from both, every window's scatter at its
own place (part.code_noise), and every observation's variance is taken
at its own place, the fit's prediction of it (reweigh()).  A mean also
carries the rounding's bias, which noise smaller than a code does not
dither away (Widrow, Kollar & Liu 1996, "Statistical theory of
quantization", eq. 7), as its own variance.

THE TESTS.  Each block's chi-square is on its residuals' degrees of
freedom (its count less the fit's leverage there).  A spin block is
tested by its lack of fit (Draper & Smith, "Applied Regression
Analysis", 3rd ed., sec. 2.1): the model's chi-square less the pure
error's, on the harmonics' count less the leverage -- what the model
misses of the spins' smooth signals, against the noise their own windows
measured.  The poses' block is stage 1's chi-square and the
refinement's against its prior.

PARAMETERS (theta, 46):
  MA 0:9   BA 9:12   CA 12:15   M_a (row-major), b_a m/s^2, c_a m/s^2/C
  MG 15:24 BG 24:27  CG 27:30   M_g, b_g rad/s, c_g rad/s/C
  GS 30:39                      G' = G R^T, the gyro's g-sensitivity G
                                with the force in the clamp's frame, R
                                the chip's axes there (which the fit
                                reads as U, the mounting), rad/s per m/s^2
  RR 39:42                      r m, the IMU's sensing centre, clamp frame
  UU 42:44                      gravity's up in the cameras' world, as a
                                tangent step from a start
  T0 44, RHO 45                 the board's clock: its first sample at
                                station time T0, a sample every rho/ODR s
"""
from dataclasses import dataclass, field
from functools import lru_cache
from math import sqrt, pi, log

import numpy as np

from ..spec import Imu, Site, Quality, Board, Print, Clamp, Gimbal
from .. import imu_fusion_sim as study
from ..rig.calib import chi2_quantile
from ..rig.pose import normal_quantile
from . import kin as K
from .part import G0, D2R, T_REF, lsb, physical, window_noise, impulse, code_var, code_var_range, code_noise, Stream

MA, BA, CA = slice(0, 9), slice(9, 12), slice(12, 15)
MG, BG, CG = slice(15, 24), slice(24, 27), slice(27, 30)
GS, RR, UU = slice(30, 39), slice(39, 42), slice(42, 44)
T0, RHO = 44, 45
NP = 46
LINEAR = np.r_[0:42]               # every parameter the model is linear in, r aside


# ================================================================ FACTS
@dataclass
class Facts:
    """What the station knows of where it stands and of the part's output
    path, SI."""
    g: float                       # m/s^2, the site's gravity
    lat: float                     # deg
    north: np.ndarray              # north in the cameras' world: the rig's, to its build
    delay: float                   # s, the output filter's delay at low frequency

    @classmethod
    def site(cls):
        fc = Imu.FILTER_BW
        if Imu.FILTER_ORDER != 2:
            raise NotImplementedError("only the 2nd-order output filter is modelled")
        # a 2nd-order Butterworth's group delay at DC: sqrt(2) / (2 pi fc)
        return cls(K.gravity(Site.LAT, Site.ALT), Site.LAT, K.north_rig(Site.HEADING), sqrt(2.0) / (2.0 * pi * fc))


# ================================================================ TERMS
# what is judged, as the study states its targets (imu_fusion_sim.CALIBRATED,
# each a 1 sigma error over bats); the clock is judged as a gyro scale,
# since a rate integrated on the wrong clock is a scale error
_C = study.CALIBRATED
GROUPS = (("accel bias", 3, _C.acc_bias * 1000.0, "mg"),
          ("accel scale", 3, _C.acc_scale, ""),
          ("accel cross", 3, _C.acc_cross, ""),
          ("gyro bias", 3, _C.gyro_bias, "dps"),
          ("gyro scale", 3, _C.gyro_scale, ""),
          ("gyro cross", 6, _C.gyro_cross, ""),
          ("mount", 3, _C.mount_deg, "deg"),
          ("lever", 3, _C.lever_mm, "mm"),
          ("clock", 1, _C.gyro_scale, ""))
NT = sum(g[1] for g in GROUPS)
GROUP_OF = np.concatenate([[k] * g[1] for k, g in enumerate(GROUPS)])
TAU = np.concatenate([[g[2]] * g[1] for g in GROUPS])
_OFF = ((0, 1), (0, 2), (1, 2))
_OFF6 = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))


def thresholds():
    """(NT,): the largest 1 sigma a term's fit may have.  A station is
    qualified on CAL_QUAL_BATS bats, and its RMS error in each group must
    then meet the group's target; a healthy station fails that with
    probability CAL_QUAL_ALPHA, shared among the groups (Bonferroni).  The
    RMS of nu squared N(0, s^2) errors is under tau with probability
    1 - alpha_g when s = tau / k, k^2 = chi2_{1-alpha_g}(nu) / nu."""
    a_g = Quality.CAL_QUAL_ALPHA / len(GROUPS)
    out = []
    for name, n, tau, _ in GROUPS:
        nu = Quality.CAL_QUAL_BATS * n
        k = sqrt(chi2_quantile(nu, a_g) / nu)
        out += [tau / k] * n
    return np.array(out)


def polar(M):
    """(U, H): M = U H, U a rotation, H symmetric positive definite."""
    Uu, s, Vt = np.linalg.svd(M)
    U = Uu @ Vt
    if np.linalg.det(U) < 0.0:
        raise ValueError("the accelerometer's map is a reflection")
    return U, Vt.T @ np.diag(s) @ Vt


def terms(th, dT_cal, U_ref):
    """(NT,) the judged terms of parameters th, biases at the die's mean
    temperature during calibration (dT_cal above 25 C), the mounting as a
    rotation vector (deg) from U_ref."""
    Ma, Mg = th[MA].reshape(3, 3), th[MG].reshape(3, 3)
    U, H = polar(Ma.T)
    Eg = Mg @ U - np.eye(3)
    S = H - np.eye(3)
    out = np.empty(NT)
    out[0:3] = (th[BA] + th[CA] * dT_cal) / G0 * 1000.0
    out[3:6] = np.diag(S)
    out[6:9] = [S[i, j] for i, j in _OFF]
    out[9:12] = (th[BG] + th[CG] * dT_cal) / D2R
    out[12:15] = np.diag(Eg)
    out[15:21] = [Eg[i, j] for i, j in _OFF6]
    out[21:24] = np.degrees(K.log_so3(U @ U_ref.T))
    out[24:27] = th[RR] * 1000.0
    out[27] = th[RHO] - 1.0
    return out


def mounting(th):
    return polar(th[MA].reshape(3, 3).T)[0]


def terms_jacobian(th, dT_cal):
    """(NT, NP) d terms / d theta at th, the mounting about its own value."""
    U0 = mounting(th)
    J = np.zeros((NT, NP))
    for j in list(range(42)) + [RHO]:
        h = 1e-7 * max(1.0, abs(th[j]))
        tp, tm = th.copy(), th.copy()
        tp[j] += h
        tm[j] -= h
        J[:, j] = (terms(tp, dT_cal, U0) - terms(tm, dT_cal, U0)) / (2.0 * h)
    return J


# ================================================================ NOISE
def sample_var():
    """Per unit noise density squared: the variance of one output sample."""
    return window_noise(1)[0]


def quant_var(sigma, step):
    """The variance, over where a constant falls between codes, of the bias
    a mean of its quantised samples keeps when Gaussian noise of sigma
    rides on it: the quantiser's characteristic function at the noise's
    (Widrow, Kollar & Liu 1996, eq. 7) -- the sum of its harmonics."""
    k = np.arange(1, 64)
    return float(step ** 2 * np.sum(np.exp(-4.0 * pi ** 2 * k ** 2 * sigma ** 2 / step ** 2) / (2.0 * pi ** 2 * k ** 2)))


def datasheet_nd():
    """(6,) noise density, SI per rtHz: gyro then accel."""
    return np.array([Imu.GYRO_ND * D2R] * 3 + [Imu.ACC_ND * G0] * 3)


def steps():
    lg, la = lsb()
    return np.array([lg] * 3 + [la] * 3)


def window_W():
    """The window: the fewest samples, a power of two, whose neighbouring
    means correlate by no more than 1 / sqrt(2 CAL_QUAL_BATS) -- the
    resolution of a qualification's Monte Carlo -- so the pure error may
    count windows as independent degrees of freedom (NOISE).  What a window
    weighs in a spin's slow signal is not its own variance: its neighbours'
    correlation adds to it twice over, 7 to 9 % here, and obs_var(run=True)
    carries it."""
    lim = 1.0 / sqrt(2.0 * Quality.CAL_QUAL_BATS)
    W = 1
    while abs(window_noise(W)[1]) > lim:
        W *= 2
    return W


def code_sd(nd2):
    """(6,) codes: the noise of one sample, density nd2 (squared)."""
    return np.sqrt(np.asarray(nd2, float) * sample_var()) / steps()


def obs_var(nd2, n, nu="most", run=False):
    """(.., 6) variance of an observation: an n-sample mean of codes of
    noise of density nd2 (squared, per channel), rounded (part.code_var)
    where the true value falls nu of a code past one ((.., 6) in codes) --
    or, before it is known, where that is "least", "mean" or "most" -- plus
    the rounding's bias: fixed over a still, a fresh draw for each window
    of a spin, whose signal moves.  With run, a spin's window: what it
    weighs in the spin's slow signal, its neighbours' correlation in it
    (part.code_var)."""
    s = code_sd(nd2)
    q = _quant_vars(tuple(np.asarray(nd2, float)))
    if isinstance(nu, str):
        k = ("least", "mean", "most").index(nu)
        return np.array([code_var_range(float(sc), int(n), 64, bool(run))[k] for sc in s]) * steps() ** 2 + q
    nu = np.asarray(nu, float)
    v = np.stack([code_var(float(s[c]), nu[..., c].ravel(), int(n), run).reshape(nu.shape[:-1])
                  for c in range(6)], -1)
    return v * steps() ** 2 + q


def noise_dof():
    """The fewest degrees of freedom the noise may be measured on: then its
    variance is known to sqrt(2 / dof), no worse than the resolution of a
    qualification's Monte Carlo, 1 / sqrt(2 CAL_QUAL_BATS) (as window_W)."""
    return 4 * Quality.CAL_QUAL_BATS


def still_noise_dof(n, W):
    """What a still of n samples gives the noise: its W-windows' scatter
    about their own straight line, if it has the four windows that leave
    that line two degrees of freedom to spare."""
    m = int(n) // W
    return m - 2 if m >= 4 else 0


def spin_noise_dof(n, W):
    """What a spin's n quiet samples give the noise: its W-windows' scatter
    about its own smooth signal (spin_basis), if it has windows to spare."""
    m = int(n) // W
    return m - P_SMOOTH if m >= 2 * P_SMOOTH else 0


def f_quantile(d1, d2, alpha):
    """The F(d1, d2) value exceeded with probability alpha, by Paulson's
    cube-root normal approximation (Paulson 1942, Ann. Math. Stat. 13:
    233) -- the chi-square's (calib.chi2_quantile) when the variance it is
    scaled by was itself measured on d2 degrees of freedom."""
    a, b = 2.0 / (9.0 * d1), 2.0 / (9.0 * d2)
    z = normal_quantile(1.0 - alpha)
    A = (1.0 - b) ** 2 - z * z * b
    B = -2.0 * (1.0 - a) * (1.0 - b)
    C = (1.0 - a) ** 2 - z * z * a
    x = (-B + sqrt(B * B - 4.0 * A * C)) / (2.0 * A)
    return x ** 3


@lru_cache(maxsize=256)
def _quant_vars(nd2):
    return np.array([quant_var(sqrt(v * sample_var()), s) for v, s in zip(nd2, steps())])


# ============================================================ OBSERVATIONS
# SPIN MOTION.  At a steady commanded rate a hinge still lags its command by
# the torque it carries over its stepper's stiffness, and that torque turns
# with the spun angle: gravity's is its first harmonic, the bodies' products
# of inertia at a steady rate its second, and nothing else turns with it.
# So each hinge, the held one too, is a straight line in time plus NH
# harmonics of the spun angle, fitted to the encoders (observations()).
NH = 2
# a steady spin's signals (NOISE): a line in time, for the die's warming,
# and harmonics of the spun angle up to those the motion's own NH make
# when they meet gravity's turn and each other, 2 NH + 1
SMOOTH_K = 2 * NH + 1
P_SMOOTH = 2 + 2 * SMOOTH_K


@dataclass
class Quiet:
    """An interval of the plan in which nothing rings, station time."""
    kind: str                  # "still" or "spin"
    t0: float
    t1: float
    item: int                  # the plan's item it belongs to
    hinge: int = -1
    rate: float = 0.0          # rad/s, the spin's commanded rate


@dataclass
class Obs:
    """The observations, one row each: a still's mean or a spin's window."""
    moving: np.ndarray         # (n,) bool
    k0: np.ndarray             # (n,) first sample
    k1: np.ndarray             # (n,) one past the last
    q0: np.ndarray             # (n, 2) rad, the hinges at tref
    qd: np.ndarray             # (n, 2) rad/s
    tref: np.ndarray           # (n,) station time
    dT: np.ndarray             # (n,) mean die temperature less 25 C
    y: np.ndarray              # (n, 6) mean outputs, SI (gyro, accel)
    var: np.ndarray            # (n, 6)
    quiet: np.ndarray          # (n,) the quiet interval each came from
    hinge: np.ndarray = None   # (n,) a spin's hinge, -1 at a still
    harm: np.ndarray = None    # (n, 2, 2 NH) each hinge's ripple about its straight line (SPIN MOTION)

    def __post_init__(self):
        n = len(self.k0)
        if self.hinge is None:
            self.hinge = np.array([int(np.argmax(np.abs(v))) if m else -1 for v, m in zip(self.qd, self.moving)], int)
        if self.harm is None:
            self.harm = np.zeros((n, 2, 2 * NH))

    def __len__(self):
        return len(self.k0)


@dataclass
class Agg:
    """Each observation's kinematics, averaged over its samples."""
    Rt: np.ndarray             # (n, 3, 3) mean R^T
    a: np.ndarray              # (n, 3) mean R^T a_O
    w: np.ndarray              # (n, 3) mean clamp rate
    A: np.ndarray              # (n, 3, 3) mean [w']x + [w]x^2


def state(obs, rows, t):
    """(q, qd, qdd), each (len(t), 2): the hinges at station times t, each
    on observation rows[i]'s motion (SPIN MOTION)."""
    q0, qd, tau = obs.q0[rows], obs.qd[rows], t - obs.tref[rows]
    q = q0 + qd * tau[:, None]
    v = qd.copy()
    a = np.zeros_like(q)
    hm = obs.harm[rows]
    if np.any(hm):
        h = np.maximum(obs.hinge[rows], 0)
        i = np.arange(len(rows))
        th, w = q[i, h], qd[i, h]
        for k in range(1, NH + 1):
            c, s_ = np.cos(k * th), np.sin(k * th)
            A, B = hm[:, :, 2 * k - 2], hm[:, :, 2 * k - 1]
            q = q + A * c[:, None] + B * s_[:, None]
            v = v + (k * w)[:, None] * (B * c[:, None] - A * s_[:, None])
            a = a - ((k * w) ** 2)[:, None] * (A * c[:, None] + B * s_[:, None])
    return q, v, a


def spin_basis(obs, rows, T0_, rho, facts):
    """(len(rows), P_SMOOTH): what a steady spin's signal at windows rows
    (one spin's) can be made of, whatever the calibration: 1, the time from
    the spin's middle, and cos, sin of k times the spun angle, k = 1 ..
    SMOOTH_K, the angle at each window's centre on the spin's line."""
    rows = np.asarray(rows)
    kc = 0.5 * (obs.k0[rows] + obs.k1[rows] - 1)
    tau = T0_ + rho * kc / Imu.ODR - facts.delay - obs.tref[rows]
    h = obs.hinge[rows]
    i = np.arange(len(rows))
    ang = obs.q0[rows][i, h] + obs.qd[rows][i, h] * tau
    return np.stack([np.ones_like(tau), tau] + [f(k * ang) for k in range(1, SMOOTH_K + 1)
                                                for f in (np.cos, np.sin)], 1)


def _pure_error(B, Y):
    """Least squares of Y (m, c) on B (m, p): (residuals' sum of squares (c,),
    each row's redundancy 1 - h (m,), the fitted values (m, c))."""
    Q, _ = np.linalg.qr(B)
    fit_ = Q @ (Q.T @ Y)
    return np.sum((Y - fit_) ** 2, 0), 1.0 - np.sum(Q * Q, 1), fit_


def _sample_index(obs, mv, T0_, rho, facts):
    """Every sample of the moving windows mv: (count per window, window of
    each sample, sample number, station time)."""
    cnt = obs.k1[mv] - obs.k0[mv]
    idx = np.repeat(np.arange(len(mv)), cnt)
    k = np.concatenate([np.arange(a_, b_) for a_, b_ in zip(obs.k0[mv], obs.k1[mv])])
    return cnt, idx, k, T0_ + rho * k / Imu.ODR - facts.delay


def _kinematics(kin, q, qd, qdd):
    """Per sample at hinge states (q, qd, qdd): (R^T, R^T a_O, the clamp's
    rate, [w']x + [w]x^2), the clamp's own frame."""
    R, x, wc, wd, aO = kin.motion(q, qd, qdd)
    RtS = np.swapaxes(R, 1, 2)
    return RtS, np.einsum("nij,nj->ni", RtS, aO), wc, K.skew(wd) + K.skew(wc) @ K.skew(wc)


def _mean_by(cnt, idx):
    if np.all(cnt == cnt[0]):
        def mean(v):
            return v.reshape((len(cnt), cnt[0]) + v.shape[1:]).mean(1)
    else:
        def mean(v):
            out = np.zeros((len(cnt),) + v.shape[1:])
            np.add.at(out, idx, v)
            return out / cnt.reshape((-1,) + (1,) * (v.ndim - 1))
    return mean


def aggregates(kin, obs, T0_, rho, facts, slopes=False):
    """Agg at the clock (T0, rho): every window's kinematics averaged over
    its samples.  With slopes, also (d Agg / d T0, d Agg / d rho): each
    sample's rate of change (central differences along its own motion)
    averaged as the samples are, times 1 and k / ODR."""
    n = len(obs)
    out = [Agg(np.zeros((n, 3, 3)), np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3, 3)))
           for _ in range(3 if slopes else 1)]
    st = ~obs.moving
    if np.any(st):
        R, x = kin.pose(obs.q0[st])
        out[0].Rt[st] = np.swapaxes(R, 1, 2)
    mv = np.nonzero(obs.moving)[0]
    if len(mv):
        cnt, idx, k, t = _sample_index(obs, mv, T0_, rho, facts)
        rows = mv[idx]
        mean = _mean_by(cnt, idx)
        X = _kinematics(kin, *state(obs, rows, t))
        for f, v in zip(("Rt", "a", "w", "A"), X):
            getattr(out[0], f)[mv] = mean(v)
        if slopes:
            h = 1e-5                       # s: central differences err by (w h)^2 / 6, 1e-8 at the fastest spin
            Xp, Xm = _kinematics(kin, *state(obs, rows, t + h)), _kinematics(kin, *state(obs, rows, t - h))
            lever = (k / Imu.ODR)
            for f, vp, vm in zip(("Rt", "a", "w", "A"), Xp, Xm):
                D = (vp - vm) / (2.0 * h)
                getattr(out[1], f)[mv] = mean(D)
                getattr(out[2], f)[mv] = mean(D * lever.reshape((-1,) + (1,) * (D.ndim - 1)))
    return out if slopes else out[0]


def centre_aggregates(kin, obs, T0_, rho, facts):
    """Agg with each window's kinematics at its centre instead of averaged:
    for sensitivities only (kin_jacobian), where a window's curvature
    (its angle squared over 24, under 1e-3 at the gimbal's fastest spin)
    does not matter."""
    n = len(obs)
    g = Agg(np.zeros((n, 3, 3)), np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3, 3)))
    st = ~obs.moving
    if np.any(st):
        R, x = kin.pose(obs.q0[st])
        g.Rt[st] = np.swapaxes(R, 1, 2)
    mv = np.nonzero(obs.moving)[0]
    if len(mv):
        kc = 0.5 * (obs.k0[mv] + obs.k1[mv] - 1)
        t = T0_ + rho * kc / Imu.ODR - facts.delay
        g.Rt[mv], g.a[mv], g.w[mv], g.A[mv] = _kinematics(kin, *state(obs, mv, t))
    return g


class Model:
    """The model of the observations at parameters theta: the gimbal as
    stage 1 gave it, refined by dk (THE TWO STAGES), gravity's up a tangent
    step from u0.

    THE CLOCK.  The aggregates depend on the clock (T0, rho) only through
    when each window's samples fall; the model holds them exactly at one
    clock and to first order about it (linearize()), so a Gauss-Newton
    step costs no pass over the samples.  fit() moves the point to each
    solution and solves again until the solution stays put.

    THE GIMBAL.  The refinement moves the aggregates to first order,
    dAgg / dk on the windows' centres (kin_slopes), taken at the same
    clock; the prediction is linear in the aggregates, so the maps'
    Jacobian and the refinement's are both exact for that model."""

    def __init__(self, kin, obs, facts, u0):
        self.kin, self.obs, self.facts = kin, obs, facts
        self.u0 = K.unit(u0)
        e1, e2 = K.perp_basis(self.u0)
        self.E = np.stack([e1, e2], 1)
        self._lin = None
        self._kd = None
        self.dk = np.zeros(14)

    def linearize(self, T0_, rho):
        g, d0, dr = aggregates(self.kin, self.obs, T0_, rho, self.facts, slopes=True)
        self._lin = (float(T0_), float(rho), g, d0, dr)
        self._kd = None

    def kin_slopes(self):
        """Agg of (14, n, ...): d Agg / d dk at the linearized clock,
        forward differences on the windows' centres (centre_aggregates)."""
        if self._kd is None:
            t0, r0 = self._lin[:2]
            args = (self.obs, t0, r0, self.facts)
            g0 = centre_aggregates(self.kin, *args)
            h = 1e-6
            parts = []
            for j in range(14):
                d = np.zeros(14)
                d[j] = h
                gj = centre_aggregates(self.kin.perturb(d), *args)
                parts.append([(getattr(gj, f) - getattr(g0, f)) / h for f in ("Rt", "a", "w", "A")])
            self._kd = Agg(*(np.stack([p_[i] for p_ in parts]) for i in range(4)))
        return self._kd

    @property
    def point(self):
        return self._lin[:2]

    def up(self, U):
        return K.unit(self.u0 + self.E @ U)

    def omega(self, u):
        return K.earth_rate(self.facts.lat, u, self.facts.north)

    def agg(self, T0_, rho):
        if self._lin is None:
            self.linearize(T0_, rho)
        t0, r0, g, d0, dr = self._lin
        a, b = float(T0_) - t0, float(rho) - r0
        if a != 0.0 or b != 0.0:
            g = Agg(*(getattr(g, f) + a * getattr(d0, f) + b * getattr(dr, f) for f in ("Rt", "a", "w", "A")))
        if np.any(self.dk):
            kd = self.kin_slopes()
            g = Agg(*(getattr(g, f) + np.tensordot(self.dk, getattr(kd, f), 1) for f in ("Rt", "a", "w", "A")))
        return g

    def fbar(self, th, g):
        u = self.up(th[UU])
        return self.facts.g * np.einsum("nij,j->ni", g.Rt, u) + g.a + np.einsum("nij,j->ni", g.A, th[RR])

    def _kinematic_part(self, th, g):
        """(n, 6): the part of the prediction that the aggregates carry
        (everything but the biases and the temperature terms) -- linear
        in them."""
        u = self.up(th[UU])
        f = self.fbar(th, g)
        Ma, Mg, Gs = th[MA].reshape(3, 3), th[MG].reshape(3, 3), th[GS].reshape(3, 3)
        wm = g.w + np.einsum("nij,j->ni", g.Rt, self.omega(u))
        return np.concatenate([wm @ Mg.T + f @ Gs.T, f @ Ma.T], 1)

    def predict(self, th, g=None):
        g = self.agg(th[T0], th[RHO]) if g is None else g
        dT = self.obs.dT[:, None]
        y = self._kinematic_part(th, g)
        y[:, 0:3] += th[BG] + dT * th[CG]
        y[:, 3:6] += th[BA] + dT * th[CA]
        return y

    def jacobian(self, th, time=True):
        """(n, 6, NP) d predict / d theta: analytic, but for gravity's
        direction's tiny pull on the Earth's rate (central differences);
        the clock's columns are the aggregates' slopes."""
        g = self.agg(th[T0], th[RHO])
        n = len(self.obs)
        J = np.zeros((n, 6, NP))
        u = self.up(th[UU])
        f = self.fbar(th, g)
        Ma, Mg, Gs = th[MA].reshape(3, 3), th[MG].reshape(3, 3), th[GS].reshape(3, 3)
        wm = g.w + np.einsum("nij,j->ni", g.Rt, self.omega(u))
        dT = self.obs.dT
        I3 = np.eye(3)
        for i in range(3):
            J[:, 3 + i, 3 * i:3 * i + 3] = f                      # M_a row i
            J[:, i, 15 + 3 * i:15 + 3 * i + 3] = wm               # M_g row i
            J[:, i, 30 + 3 * i:30 + 3 * i + 3] = f                # G' row i
        J[:, 3:6, BA] = I3
        J[:, 3:6, CA] = dT[:, None, None] * I3
        J[:, 0:3, BG] = I3
        J[:, 0:3, CG] = dT[:, None, None] * I3
        J[:, 3:6, RR] = np.einsum("ij,njk->nik", Ma, g.A)
        J[:, 0:3, RR] = np.einsum("ij,njk->nik", Gs, g.A)
        # gravity's up: f moves with it, and so (by 1e-4 of a part in a
        # thousand) does the Earth's rate as the station reads it
        v = self.u0 + self.E @ th[UU]
        du = (I3 - np.outer(u, u)) / np.linalg.norm(v) @ self.E            # (3, 2)
        df = self.facts.g * np.einsum("nij,jk->nik", g.Rt, du)
        J[:, 3:6, UU] = np.einsum("ij,njk->nik", Ma, df)
        hU = 1e-6
        dOm = np.stack([(self.omega(self.up(th[UU] + hU * e)) - self.omega(self.up(th[UU] - hU * e))) / (2 * hU)
                        for e in np.eye(2)], 1)
        J[:, 0:3, UU] = np.einsum("ij,njk->nik", Gs, df) + np.einsum("ij,njk->nik", Mg,
                                                                     np.einsum("nij,jk->nik", g.Rt, dOm))
        if time and np.any(self.obs.moving):
            _, _, _, d0, dr = self._lin
            J[:, :, T0] = self._kinematic_part(th, d0)
            J[:, :, RHO] = self._kinematic_part(th, dr)
        return J

    def kin_jacobian(self, th):
        """(n, 6, 14) d predict / d dk: the prediction along each of the
        aggregates' slopes (kin_slopes), which it is linear in."""
        if self._lin is None:
            self.linearize(th[T0], th[RHO])
        kd = self.kin_slopes()
        return np.stack([self._kinematic_part(th, Agg(kd.Rt[j], kd.a[j], kd.w[j], kd.A[j])) for j in range(14)], 2)


def priors(t0_guess):
    """(mean, sigma) of the nuisances' priors: the datasheet's spreads."""
    mean = np.zeros(NP)
    sig = np.full(NP, np.inf)
    sig[GS] = Imu.GYRO_GSENS * D2R / G0
    sig[CA] = Imu.ACC_TC * G0
    sig[CG] = Imu.GYRO_TC * D2R
    mean[RHO], sig[RHO] = 1.0, Imu.ODR_TOL
    mean[T0], sig[T0] = t0_guess, Imu.START_S
    return mean, sig


# ============================================================ THE INPUTS
@dataclass
class PoseFix:
    """The cameras' pose of the clamp at a still, SI: rotation (clamp ->
    world), position m, covariance [dphi rad (world, left), dx m]."""
    t: float
    R: np.ndarray
    x: np.ndarray
    cov: np.ndarray


@dataclass
class CalInputs:
    """Everything the fit is given, and nothing else (check_imu's scan holds
    the fit to it)."""
    stream: object             # part.Stream
    enc_t: np.ndarray          # (m,) station time of each encoder reading
    enc_q: np.ndarray          # (m, 2) rad, the encoders (counts times their step)
    poses: list                # [PoseFix], one per still, in the plan's order
    quiet: list                # [Quiet], the plan's
    starts: np.ndarray         # station times the gimbal leaves rest
    W: int                     # the plan's window
    t_on: float                # station time the board was powered
    theta0: np.ndarray         # the plan's nominal parameters
    s: np.ndarray              # (NT,) the plan's thresholds
    spread: np.ndarray         # (NT,) the parts' and the build's spreads, 1 sigma
    lag: float                 # rad, the largest lag the plan's motion can leave
    path: object               # kin.Path, the commanded motion


@dataclass
class CalResult:
    theta: np.ndarray
    Sigma: np.ndarray          # (NP, NP), the gimbal's refinement marginalised
    kin: object                # stage 1's, from the cameras
    Sigma_kin: np.ndarray      # its covariance: the refinement's prior
    helmert: float             # stage 1's chi-square per dof
    terms: np.ndarray          # (NT,)
    Sigma_terms: np.ndarray
    dT_cal: float
    blocks: dict               # name -> (chi2, dof)
    nd: np.ndarray             # (6,) measured noise density, SI
    obs: Obs
    converged: bool
    iters: int
    rails: int                 # railed samples in what was used
    gaps: int                  # samples the stream lacks (dropped) in what was used
    follow: float              # rad, the largest lag the encoders showed
    model: object = None
    short: int = 0             # spins that turned less than a whole turn quietly, not used
    nd_dof: int = 0            # the degrees of freedom the noise was measured on
    failed: str = ""           # why no calibration could be read from the solution, if none could
    dk: np.ndarray = None      # (14,) the IMU's refinement of the gimbal: kin.perturb(dk) is the joint answer
    lost: int = 0              # quiet intervals the stream does not hold in full: it was cut short (outside())


# ================================================================ TIMING
def on_counter(stream):
    """(Stream, missing (n,) bool): the stream laid on the board's counter,
    sample k at index k, from 0 to its last.  A sample the link dropped is
    a counter the stream skips; it takes the last sample before it (the
    first, before any) and missing marks it, so that a drop costs only a
    quiet interval it falls inside, which it fails, and does not move
    every sample after it by one."""
    k = np.asarray(stream.k, int)
    n = int(k[-1]) + 1 if len(k) else 0
    if len(k) == n and (n == 0 or (k[0] == 0 and np.all(np.diff(k) == 1))):
        return stream, np.zeros(n, bool)
    c = np.arange(n)
    i = np.maximum(np.searchsorted(k, c, side="right") - 1, 0)
    return Stream(c, stream.acc[i], stream.gyro[i], stream.temp[i], stream.rail[i]), k[i] != c


def _gyro_rate(stream):
    g, a, T = physical(on_counter(stream)[0])
    return g


def nominal_clock(inp):
    """(T0, rho) as the datasheet has them: the first sample START_S after
    power-up, a sample every 1 / ODR."""
    return inp.t_on + Imu.START_S, 1.0


def rough_clock(inp, facts, kin):
    """(T0, rho, u): the board's clock to about a sample over the whole
    plan, by matching the gyro's rate to the encoders' -- the norm of each,
    which the two share whatever the IMU's mounting, scale and bias (its
    bias the mean of the stream's first 200 samples, the gimbal at rest for
    them: plan.boot_s) -- over every first sample from power-up to twice
    the datasheet's boot and every rate the part's tolerance allows: the
    usual start for an IMU's clock against another sensor, cross-correlated
    angular rates (Mair et al. 2011; Kelly & Sukhatme 2014).  What is
    matched is each norm's rate of change over 10 ms, the moves' starts and
    stops: the norms themselves are mostly the spins' long plateaus, whose
    overlap barely changes with the clock (on M3's kid a first sample 30
    ms off cost them 0.06% of their match), so a clock 66 ms early and
    0.12% fast matched as well as the true one.  Searched first on
    10-sample means, every rate that moves the plan's last sample by less
    than one of them, then at full rate about the best; u (s) is how far it
    may leave any of the plan's samples: the 10 ms the matched rate of
    change is taken over, which its peak cannot resolve finer than (on 30
    of M3's bats it was 7.5 ms off at worst).  timing() refines it.
    Without it timing() extrapolated from its first start: on M3's adult
    plan the next start from rest came 23 s later, a clock 1.2% off put it
    285 ms from where timing looked, and the pair it took there, 140 ms
    wrong, passed the tolerance test."""
    D = 10
    yg = _gyro_rate(inp.stream)
    g = np.linalg.norm(yg - yg[:200].mean(0), axis=1)
    n = len(g)
    box = np.ones(D) / D
    dg = np.gradient(np.convolve(g, box, "same"))
    qd = np.gradient(inp.enc_q, inp.enc_t, axis=0)
    e = np.linalg.norm(kin.motion(inp.enc_q, qd, np.zeros_like(qd))[2], axis=1)
    de = np.gradient(np.convolve(e, np.ones(int(round(D / Imu.ODR * Gimbal.ENCODER_HZ))), "same"), inp.enc_t)
    T0n, _ = nominal_clock(inp)
    T = n / Imu.ODR
    span = accept_z() * Imu.ODR_TOL

    def best(sig, k, h, rhos, M):
        """(rho, shift) maximising sum (sig - mean) de(T0n + rho k / ODR -
        delay + shift), shifts m h for |m| <= M, by FFT per rho."""
        v = sig - sig.mean()
        L = 1 << int(np.ceil(np.log2(len(v) + 2 * M + 1)))
        V = np.conj(np.fft.rfft(v, L))
        j = np.arange(-M, len(v) + M)
        top = (-np.inf, 0.0, 0.0)
        for r in rhos:
            t = T0n + r * (k[0] + (k[1] - k[0]) * j) / Imu.ODR - facts.delay
            c = np.fft.irfft(np.fft.rfft(np.interp(t, inp.enc_t, de), L) * V, L)[:2 * M + 1]
            m = int(np.argmax(c))
            if c[m] > top[0]:
                top = (c[m], r, (m - M) * h)
        return top[1], top[2]
    nd = n // D
    kd = D * np.arange(nd) + 0.5 * (D - 1)
    h1, d1 = D / Imu.ODR, D / Imu.ODR / T
    rho, sh = best(dg[kd.astype(int)], kd, h1, np.arange(1.0 - span, 1.0 + span + d1, d1),
                   int(np.ceil(Imu.START_S / h1)))
    h2, d2 = 1.0 / Imu.ODR, d1 / 5.0
    T0n += sh
    rho, sh = best(dg, np.arange(n, dtype=float), h2, rho + d2 * np.arange(-5, 6), D)
    return T0n + sh, rho, h1


def turn_angle():
    """rad: how far the clamp turns from rest before timing() marks a
    start: a hundred of the encoders' codes, so their resolution moves the
    mark by a hundredth of the angle -- about half a percent of the time
    from rest to it, a constant acceleration's angle going as its square
    (M3's slowest start reaches it in 0.2 s: 1 ms) -- while the gyro's
    baseline, its mean over a hundred samples, is far finer."""
    return 100.0 * 2.0 * pi / 2 ** Gimbal.ENCODER_BITS


def timing(inp, facts, kin):
    """(T0, rho, pairs) to start from: rough_clock refined at each time the
    gimbal leaves rest, the moment the clamp has turned turn_angle() from
    where it rested, by the encoders (station time) and by the gyro, its
    rate less its baseline summed (sample number, through the filter's
    delay); a straight line through the pairs, refitted as they come.  A
    pair further from the rough clock than its u is a start the gyro did
    not see as the encoders did (a channel that does not move) and is not
    used.

    From rest: a start the encoders do not show still through the gyro's
    baseline before it -- the move that follows a spin's stop, the gimbal
    still turning -- is not used, since that baseline would take the spin's
    last motion for the gyro's bias (on M3's plans it put such pairs 120
    to 300 ms off).  An angle, not a rate: the encoders read angle, and
    their rate, a difference of codes, is too coarse to time a slow start
    by (a rate threshold put M3's slow starts up to 44 ms early)."""
    yg = _gyro_rate(inp.stream)
    nk = len(yg)
    lg, _ = lsb()
    rest = 40.0 * lg                   # far above the gyro's noise, far below any move's peak
    turn = turn_angle()
    dt_e = float(np.median(np.diff(inp.enc_t)))
    qd = np.gradient(inp.enc_q, inp.enc_t, axis=0)
    R_e, _, w_e, _, _ = kin.motion(inp.enc_q, qd, np.zeros_like(qd))
    w_enc = np.linalg.norm(w_e, axis=1)
    T0_, rho, u = rough_clock(inp, facts, kin)
    T0r, rhor = T0_, rho
    pre, look = int(0.1 * Imu.ODR), int(0.4 * Imu.ODR)
    # the encoders still through the gyro's baseline wherever u puts it
    n_rest = int(np.ceil((2 * pre / Imu.ODR + u) / dt_e))
    pairs = []

    def cross(ang, i):
        """The fractional index at which ang passes turn, i its first past."""
        return i - 1 + (turn - ang[i - 1]) / (ang[i] - ang[i - 1])
    for ts in inp.starts:
        je = int(np.searchsorted(inp.enc_t, ts))
        if je < n_rest or np.any(w_enc[je - n_rest:je] > rest):
            continue
        # the clamp's turn from where it rested, by the encoders' pose
        seg = slice(je, min(je + int(1.0 / dt_e), len(inp.enc_t)))
        D = np.einsum("ji,njk->nik", R_e[je], R_e[seg])
        ang = np.arcsin(np.minimum(0.5 * np.linalg.norm(
            np.stack([D[:, 2, 1] - D[:, 1, 2], D[:, 0, 2] - D[:, 2, 0], D[:, 1, 0] - D[:, 0, 1]], 1), axis=1), 1.0))
        hit = np.nonzero(ang > turn)[0]
        if not len(hit) or hit[0] == 0:
            continue
        te = float(np.interp(cross(ang, hit[0]), np.arange(len(ang)), inp.enc_t[seg]))
        ks = int(round((ts - T0_ + facts.delay) * Imu.ODR / rho))
        a, b = ks - 2 * pre, ks - pre
        if a < 0 or ks + look >= nk:
            continue
        # the gyro at rest: a mean, since a median of codes moves by whole
        # half-codes, which summed over the turn's time is a few percent of it
        base = np.mean(yg[a:b], axis=0)
        ang = np.linalg.norm(np.cumsum(yg[b:ks + look] - base, 0) * rho / Imu.ODR, axis=1)
        hit = np.nonzero(ang > turn)[0]
        if not len(hit) or hit[0] == 0:
            continue
        # the summed rate at sample k is the turn through its end: k + 1/2
        # on the samples' own times
        k = b + cross(ang, hit[0]) + 0.5
        if abs(te - (T0r + rhor * k / Imu.ODR - facts.delay)) > u:
            continue
        P = np.array(pairs + [(k, te)])
        if len(P) == 1:
            T0_c, rho_c = P[0, 1] - rho * P[0, 0] / Imu.ODR + facts.delay, rho
        else:
            X = np.stack([np.ones(len(P)), P[:, 0] / Imu.ODR], 1)
            c, *_ = np.linalg.lstsq(X, P[:, 1], rcond=None)
            T0_c, rho_c = float(c[0] + facts.delay), float(c[1])
        pairs.append(tuple(P[-1]))
        T0_, rho = T0_c, rho_c
    return T0_, rho, len(pairs)


# ================================================================ PREPARE
# what each quiet interval loses at either end before its samples are used:
# timing() places the board's clock to about a sample (3.4 ms at worst, at
# the far end of a plan, on 40 of M3's bats; check_calibration holds every
# bat within this), and a sample taken outside the quiet interval would see
# the move; five samples' worth
MARGIN = 0.005


def fir_s():
    """s: the output filter's memory of what came before (part.impulse)."""
    return len(impulse(Imu.ODR)) / Imu.ODR


def used_span(Q):
    """(t_a, t_b) station times: the part of a quiet interval the fit uses --
    past the filter's memory of the move before it, MARGIN in from either
    end."""
    return Q.t0 + MARGIN + fir_s(), Q.t1 - MARGIN


def _segment_encoders(inp, t0, t1):
    m = (inp.enc_t >= t0) & (inp.enc_t < t1)
    return inp.enc_t[m], inp.enc_q[m]


def quiet_samples(Q, T0_, rho, facts):
    """(ka, kb): the samples a quiet interval's used span (used_span) falls
    on at the board's clock (T0, rho), the first and one past the last --
    not yet clipped to the stream, which may not hold them."""
    ta, tb = used_span(Q)

    def k_of(t):
        return (t - T0_ + facts.delay) * Imu.ODR / rho

    return int(np.ceil(k_of(ta))), int(np.floor(k_of(tb)))


def outside(inp, T0_, rho, facts):
    """How many of the plan's quiet intervals the stream does not hold in
    full at the clock (T0, rho): their used spans begin before its first
    sample or end past its last.  The board sends from its power-up to the
    plan's end, so on a whole stream this is none; a stream cut short (a
    brown-out on the pogo pins) loses the intervals past the cut, and
    observations() uses only what of them it holds."""
    n = int(inp.stream.k[-1]) + 1 if len(inp.stream.k) else 0
    return sum(1 for Q in inp.quiet for ka, kb in [quiet_samples(Q, T0_, rho, facts)] if ka < 0 or kb > n)


def observations(inp, T0_, rho, facts, nd2=None):
    """(Obs, measured nd^2 (6,), its dof, rails, gaps, short): the quiet intervals' samples,
    a margin inside each for the clock's uncertainty."""
    stream, missing = on_counter(inp.stream)
    yg, ya, Tdie = physical(stream)
    y = np.concatenate([yg, ya], 1)
    dT = Tdie - T_REF
    n = len(y)
    W = inp.W
    rows = []
    noise = []
    rails = gaps = short = 0
    for qi, Q in enumerate(inp.quiet):
        ta, tb = used_span(Q)
        ka, kb = quiet_samples(Q, T0_, rho, facts)
        ka, kb = max(ka, 0), min(kb, n)
        if kb - ka < W:
            continue
        rails += int(np.sum(stream.rail[ka:kb]))
        gaps += int(np.sum(missing[ka:kb]))
        te, qe = _segment_encoders(inp, Q.t0, Q.t1)
        if Q.kind == "still":
            q0 = qe.mean(0)
            rows.append((False, ka, kb, q0, np.zeros(2), 0.5 * (Q.t0 + Q.t1), dT[ka:kb].mean(), y[ka:kb].mean(0), qi,
                         -1, np.zeros((2, 2 * NH))))
            m = (kb - ka) // W
            if still_noise_dof(kb - ka, W):
                Y = y[ka:ka + m * W].reshape(m, W, 6).mean(1)
                sse, w, f_ = _pure_error(np.stack([np.ones(m), np.arange(m)], 1), Y)
                noise.append((sse, w, f_ / steps()))
        else:
            h_ = Q.hinge
            if abs(Q.rate) * (tb - ta) < 2.0 * pi:
                short += 1                 # SPIN MOTION: the ripple needs a whole turn
                continue
            q0, qd, hm, tm = spin_motion(te, qe, h_)
            for k0 in range(ka, kb - W + 1, W):
                rows.append((True, k0, k0 + W, q0, qd, tm, dT[k0:k0 + W].mean(), y[k0:k0 + W].mean(0), qi, h_, hm))
    mv = np.array([r[0] for r in rows], bool)
    yy = np.array([r[7] for r in rows]).reshape(-1, 6)
    obs = Obs(moving=mv, k0=np.array([r[1] for r in rows]), k1=np.array([r[2] for r in rows]),
              q0=np.array([r[3] for r in rows]), qd=np.array([r[4] for r in rows]),
              tref=np.array([r[5] for r in rows]), dT=np.array([r[6] for r in rows]),
              y=np.array([r[7] for r in rows]), var=np.ones((len(rows), 6)), quiet=np.array([r[8] for r in rows]),
              hinge=np.array([r[9] for r in rows], int), harm=np.array([r[10] for r in rows]).reshape(-1, 2, 2 * NH))
    # the spins' pure error (NOISE): each spin's windows about its own
    # smooth signal
    for qi in np.unique(obs.quiet[mv]) if np.any(mv) else ():
        idx = np.flatnonzero(mv & (obs.quiet == qi))
        if len(idx) >= 2 * P_SMOOTH:
            sse, w, f_ = _pure_error(spin_basis(obs, idx, T0_, rho, facts), obs.y[idx])
            noise.append((sse, w, f_ / steps()))
    nd_dof = int(round(sum(float(np.sum(w)) for _, w, _ in noise)))
    if nd2 is None:
        nd2 = measure_noise(noise, W)
    # until the fit predicts each mean, a still's own mean places it; a
    # spin's window is too noisy to, and takes the average place
    reweigh(obs, nd2, np.where(mv[:, None], np.nan, yy))
    return obs, nd2, nd_dof, rails, gaps, short


def measure_noise(noise, W):
    """(6,) nd^2: the density whose rounded W-sample means scatter, each
    window at its own place, as the pure error did ([(sse (6,) SI^2, each
    window's redundancy (m,), each window's place (m, 6) codes)], one per
    still or spin)."""
    w = np.concatenate([w_ for _, w_, _ in noise]) if noise else np.zeros(0)
    nd_dof = float(np.sum(w))
    if nd_dof <= 0:
        raise ValueError("no still or spin is long enough to measure the noise on")
    sse = np.sum([s_ for s_, _, _ in noise], 0)
    nu = np.concatenate([p for _, _, p in noise])
    st, sv = steps(), sample_var()
    out = np.empty(6)
    for c in range(6):
        s_lin = sqrt(max(sse[c] / nd_dof / window_noise(W)[0] * sv, 1e-30)) / st[c]
        s_c = code_noise(sse[c] / st[c] ** 2, w, nu[:, c], W, s_lin)
        out[c] = (s_c * st[c]) ** 2 / sv
    return out


def reweigh(obs, nd2, y):
    """Each observation's variance where its true value y (.., 6, SI)
    falls between codes, a spin's window's in its run (obs_var); a NaN
    row takes the average place."""
    nu = np.asarray(y, float) / steps()
    cnt = obs.k1 - obs.k0
    var = np.empty((len(obs), 6))
    for n, run in {(int(c), bool(m)) for c, m in zip(cnt, obs.moving)}:
        m = (cnt == n) & (obs.moving == run)
        known = m & ~np.any(np.isnan(nu), 1)
        if np.any(known):
            var[known] = obs_var(nd2, n, nu[known], run)
        if np.any(m & ~known):
            var[m & ~known] = obs_var(nd2, n, "mean", run)
    obs.var = var


def spin_motion(te, qe, h):
    """(q0 (2,), qd (2,), harm (2, 2 NH), tref): a spin's hinges from its
    encoders (SPIN MOTION): each a line plus the spun angle's harmonics;
    the held hinge's line flat, since its command is.  The angle the
    harmonics turn with is the fitted line's, so the fit is redone once on
    the line it gives."""
    tm = float(te.mean())
    tau = te - tm
    c = np.linalg.lstsq(np.stack([np.ones_like(tau), tau], 1), qe[:, h], rcond=None)[0]
    q0, qd, hm = np.zeros(2), np.zeros(2), np.zeros((2, 2 * NH))
    for _ in range(2):
        th = c[0] + c[1] * tau
        hcols = [f(k * th) for k in range(1, NH + 1) for f in (np.cos, np.sin)]
        for j in (0, 1):
            base = [np.ones_like(tau)] + ([tau] if j == h else [])
            cj = np.linalg.lstsq(np.stack(base + hcols, 1), qe[:, j], rcond=None)[0]
            q0[j] = cj[0]
            qd[j] = cj[1] if j == h else 0.0
            hm[j] = cj[len(base):]
        c = np.array([q0[h], qd[h]])
    return q0, qd, hm, tm


# ================================================================ STAGE 1
def stage1(inp):
    """The gimbal's kinematics from the stills' poses: (Kin, Sigma, chi2/dof)."""
    st = [Q for Q in inp.quiet if Q.kind == "still"]
    q = np.array([_segment_encoders(inp, Q.t0, Q.t1)[1].mean(0) for Q in st])
    R = np.array([p.R for p in inp.poses])
    x = np.array([p.x for p in inp.poses])
    cov = np.array([p.cov for p in inp.poses])
    kin, Sig, chi2, dof = K.fit_kin(q, R, x, cov, K.Chain.drawn().kin())
    # the cameras' covariance is their qualification's, and the verdict's
    # poses test asks whether this bat's poses kept to it.  Scaling it by
    # chi2 / dof here would put a four-dof draw into every sigma.
    return kin, Sig, chi2 / dof


# ================================================================ STAGE 2
def initial(model, th0, T0_, rho):
    """The linear parameters in closed form, the rest from the plan."""
    obs = model.obs
    th = th0.copy()
    th[T0], th[RHO] = T0_, rho
    th[UU] = 0.0
    th[CA] = th[CG] = 0.0
    th[GS] = 0.0
    g = model.agg(T0_, rho)
    w = 1.0 / obs.var
    st = ~obs.moving
    # gravity's up first: each still's accelerometer, through the plan's
    # mounting, back into the world
    Ma0 = th0[MA].reshape(3, 3)
    v = np.einsum("nji,nj->ni", g.Rt[st], np.linalg.solve(Ma0, (obs.y[st, 3:6] - th0[BA]).T).T)
    model.u0 = K.unit(v.sum(0))
    e1, e2 = K.perp_basis(model.u0)
    model.E = np.stack([e1, e2], 1)
    # then the accelerometer, linear given gravity and r
    for _ in range(2):
        f = model.fbar(th, g)
        X = np.concatenate([f, np.ones((len(obs), 1))], 1)
        for i in range(3):
            c = np.linalg.lstsq(X * np.sqrt(w[:, 3 + i:4 + i]), obs.y[:, 3 + i] * np.sqrt(w[:, 3 + i]), rcond=None)[0]
            th[3 * i:3 * i + 3], th[9 + i] = c[:3], c[3]
        mv = obs.moving
        if np.any(mv):
            Ma = th[MA].reshape(3, 3)
            f0 = model.facts.g * np.einsum("nij,j->ni", g.Rt[mv], model.up(th[UU])) + g.a[mv]
            res = obs.y[mv, 3:6] - f0 @ Ma.T - th[BA]
            Jr = np.einsum("ij,njk->nik", Ma, g.A[mv])
            th[RR] = np.linalg.lstsq(Jr.reshape(-1, 3), res.ravel(), rcond=None)[0]
    # the gyro, linear given the rest
    u = model.up(th[UU])
    wm = g.w + np.einsum("nij,j->ni", g.Rt, model.omega(u))
    f = model.fbar(th, g)
    X = np.concatenate([wm, np.ones((len(obs), 1))], 1)
    for i in range(3):
        c = np.linalg.lstsq(X * np.sqrt(w[:, i:i + 1]), obs.y[:, i] * np.sqrt(w[:, i]), rcond=None)[0]
        th[15 + 3 * i:15 + 3 * i + 3], th[24 + i] = c[:3], c[3]
    return th


@dataclass
class KinPrior:
    """The gimbal's refinement in stage 2 (THE TWO STAGES): dk = L eta,
    eta's prior N(0, I), L stage 1's covariance's Cholesky factor."""
    L: np.ndarray
    eta: np.ndarray = field(default_factory=lambda: np.zeros(14))

    @property
    def dk(self):
        return self.L @ self.eta


def solve(model, th, prior_mean, prior_sig, kp, iters=40, tol=1e-9):
    """Gauss-Newton on theta and the gimbal's refinement kp.eta, with their
    priors, Levenberg-Marquardt damped only if a step fails to lower the
    cost.  Returns (theta, H (NP + 14 square), whitened residuals, converged,
    iterations); kp.eta is left at the solution."""
    obs = model.obs
    wh = 1.0 / np.sqrt(obs.var)
    pm = np.isfinite(prior_sig)
    P = np.zeros(NP)
    P[pm] = 1.0 / prior_sig[pm] ** 2
    Pall = np.r_[P, np.ones(14)]

    def cost(t, e):
        model.dk = kp.L @ e
        r = (obs.y - model.predict(t)) * wh
        return float(np.sum(r * r) + np.sum(P[pm] * (t[pm] - prior_mean[pm]) ** 2) + e @ e), r

    def jac(t, e):
        model.dk = kp.L @ e
        return np.concatenate([(model.jacobian(t) * wh[:, :, None]).reshape(-1, NP),
                               ((model.kin_jacobian(t) @ kp.L) * wh[:, :, None]).reshape(-1, 14)], 1)

    eta = kp.eta.copy()
    c, r = cost(th, eta)
    lam = 0.0
    converged = False
    it = 0
    for it in range(1, iters + 1):
        J = jac(th, eta)
        H = J.T @ J + np.diag(Pall)
        gvec = J.T @ r.ravel() - Pall * np.r_[np.where(pm, th - prior_mean, 0.0), eta]
        D = np.diag(np.diag(H))
        while True:
            step = np.linalg.solve(H + lam * D, gvec)
            tn, en = th + step[:NP], eta + step[NP:]
            cn, rn = cost(tn, en)
            if cn <= c * (1.0 + 1e-12):
                break
            lam = max(lam * 10.0, 1e-6)
            if lam > 1e12:
                kp.eta = eta
                model.dk = kp.dk
                return th, H, r, False, it
        th, eta, c, r = tn, en, cn, rn
        lam *= 0.1 if lam > 1e-9 else 0.0
        sd = 1.0 / np.sqrt(np.diag(H))
        if np.max(np.abs(step) / sd) < tol * 1e3:
            converged = True
            break
    kp.eta = eta
    J = jac(th, eta)
    H = J.T @ J + np.diag(Pall)
    return th, H, r, converged, it


def solve_clock(model, th, prior_mean, prior_sig, kp, rounds=8):
    """solve() with the aggregates exact at the clock it starts from (Model,
    THE CLOCK), moved to each solution until a solve from there moves
    nothing by ROUND_TOL of its sigma: the answer then rests on exact
    aggregates."""
    total = 0
    for _ in range(rounds):
        x0 = np.r_[th, kp.eta]
        model.linearize(th[T0], th[RHO])
        th, H, r, conv, it = solve(model, th, prior_mean, prior_sig, kp)
        total += it
        if conv and np.max(np.abs(np.r_[th, kp.eta] - x0) * np.sqrt(np.diag(H))) < ROUND_TOL:
            return th, H, r, True, total
    return th, H, r, False, total


# sigmas: a round that moves nothing further has converged.  The arithmetic
# itself moves a solution by a few millionths of a sigma from one exact
# linearization to the next (the cost's own rounding, a sum over 10^5
# rows); a ten-thousandth of a sigma is under any qualification's
# resolution and well above that floor
ROUND_TOL = 1e-4


def stuck(stream):
    """The channels whose code never changed through the plan: every plan
    turns every axis, and the noise alone moves a code."""
    out = []
    for name, c in (("gyro", stream.gyro), ("accel", stream.acc)):
        for j, ax in enumerate("xyz"):
            if len(c) and np.all(c[:, j] == c[0, j]):
                out.append("%s %s" % (name, ax))
    return out


def unobserved(obs, n):
    """Raise ValueError naming what the observations cannot show at all:
    gravity's up with no still (initial() reads it from the stills), the
    lever arm with no spin (A r is nothing at rest, and r has no prior).
    Only a stream cut short (outside()) leaves a plan's observations so."""
    if not np.any(~obs.moving):
        raise ValueError("no still inside the stream (%d samples): gravity's up is unobserved" % n)
    if not np.any(obs.moving):
        raise ValueError("no spin inside the stream (%d samples): the lever arm is unobserved" % n)


def fit(inp, facts=None, use=None):
    """The bat's calibration (CalResult) from its inputs.  `use` switches
    terms off for the must-fails ({"gsens": False, ...}); production never
    passes it.  A stream no calibration can be read from -- one cut short
    before the noise, a still or a spin could be measured, or a solution
    short of rank -- gives a CalResult whose `failed` says why, not an
    exception: the station refuses the bat, for its reason."""
    facts = Facts.site() if facts is None else facts
    use = {} if use is None else use
    qc = inp.path.state(inp.enc_t)[0]
    follow = float(np.max(np.abs(qc - inp.enc_q)))      # how far the gimbal fell behind its command anywhere
    dead = stuck(inp.stream)
    if dead:
        # the clock is not timed yet: what the stream holds is counted on
        # the datasheet's, which errs by no more than its tolerance
        # (Imu.ODR_TOL) of the time since power-up
        return CalResult(np.array(inp.theta0, float), np.full((NP, NP), np.nan), None, np.full((14, 14), np.nan),
                         np.nan, np.full(NT, np.nan), np.full((NT, NT), np.nan), 0.0, {}, np.full(6, np.nan),
                         None, False, 0, 0, 0, follow, failed="%s: one code through the whole plan, stuck"
                         % ", ".join(dead), lost=outside(inp, *nominal_clock(inp), facts))
    kin, Sig_k, helmert = stage1(inp)
    T0_, rho, npairs = timing(inp, facts, kin)
    clock = (T0_, rho)                 # where the observations were last placed
    th, obs, model, kp = np.array(inp.theta0, float), None, None, None
    nd2, nd_dof, rails, gaps, short, it = np.full(6, np.nan), 0, 0, 0, 0, 0
    try:
        obs, nd2, nd_dof, rails, gaps, short = observations(inp, T0_, rho, facts)
        unobserved(obs, len(inp.stream.k))
        model = Model(kin, obs, facts, np.array([0.0, 0.0, 1.0]))
        th = initial(model, inp.theta0, T0_, rho)
        pmean, psig = priors(T0_)
        for name, sl in (("gsens", GS), ("tempco", np.r_[CA, CG])):
            if use.get(name, True) is False:
                psig[sl] = 1e-30
                pmean[sl] = 0.0
                th[sl] = 0.0
        kp = KinPrior(np.linalg.cholesky(Sig_k))
        th, H, r, conv, it = solve_clock(model, th, pmean, psig, kp)
        # the clock moved the windows by a sample or more: place them
        # again, and measure the noise again on them -- on windows placed
        # by timing() alone, one that strays into a move reads it as noise
        if abs(th[T0] - T0_) * Imu.ODR > 0.5 or abs(th[RHO] - rho) * len(inp.stream.k) > 0.5:
            clock = (th[T0], th[RHO])
            obs, nd2, nd_dof, rails, gaps, short = observations(inp, th[T0], th[RHO], facts)
            unobserved(obs, len(inp.stream.k))
            u0, E = model.u0, model.E
            model = Model(kin, obs, facts, u0)
            model.E = E
            model.dk = kp.dk
            th, H, r, conv, it2 = solve_clock(model, th, pmean, psig, kp)
            it += it2
        # each mean's variance where the fit puts its true value (NOISE)
        reweigh(obs, nd2, model.predict(th))
        th, H, r, conv, it2 = solve_clock(model, th, pmean, psig, kp)
        it += it2
        return _read(inp, kin, Sig_k, helmert, model, kp, obs, th, H, r, nd2, nd_dof, conv, it, rails, gaps,
                     short, follow, outside(inp, *clock, facts))
    except (ValueError, np.linalg.LinAlgError) as e:
        # a stream or a solution no calibration can be read from (cut
        # short, a map turned inside out, information short of rank): the
        # verdict says so
        nan = np.full((NT, NT), np.nan)
        return CalResult(th, np.full((NP, NP), np.nan), kin, Sig_k, helmert, np.full(NT, np.nan), nan,
                         die_mean(obs) if obs is not None and len(obs) else 0.0, {}, np.sqrt(nd2), obs, False, it,
                         rails, gaps, follow, model, short, nd_dof, str(e), None if kp is None else kp.dk,
                         lost=outside(inp, *clock, facts))


def die_mean(obs):
    """degC above 25: the die's mean temperature over the samples the fit
    used, each row by its samples (a still's mean counts as its length)."""
    return float(np.average(obs.dT, weights=obs.k1 - obs.k0))


def _read(inp, kin, Sig_k, helmert, model, kp, obs, th, H, r, nd2, nd_dof, conv, it, rails, gaps, short, follow,
          lost=0):
    """CalResult from the solution: covariance, the blocks' chi-squares,
    the judged terms."""
    Hi = np.linalg.inv(H)
    Sigma = Hi[:NP, :NP]
    wh = 1.0 / np.sqrt(obs.var)
    J = np.concatenate([(model.jacobian(th) * wh[:, :, None]).reshape(-1, NP),
                        ((model.kin_jacobian(th) @ kp.L) * wh[:, :, None]).reshape(-1, 14)], 1)
    lev = np.einsum("ij,jk,ik->i", J, Hi, J).reshape(len(obs), 6)
    # each block's chi-square on its count less the fit's leverage there;
    # a spin block's as its lack of fit (THE TESTS)
    blocks = {}
    st, mv = ~obs.moving, obs.moving
    for name, cols in (("still gyro", slice(0, 3)), ("still accel", slice(3, 6))):
        if np.any(st):
            blocks[name] = (float(np.sum(r[st, cols] ** 2)), float(np.sum(1.0 - lev[st, cols])))
    spins = [np.flatnonzero(mv & (obs.quiet == qi)) for qi in np.unique(obs.quiet[mv])] if np.any(mv) else []
    spins = [i for i in spins if len(i) >= 2 * P_SMOOTH]
    for name, cols in (("spin gyro", range(0, 3)), ("spin accel", range(3, 6))):
        if not spins:
            continue
        rows = np.concatenate(spins)
        c2 = float(np.sum(r[rows][:, list(cols)] ** 2))
        dof = float(np.sum(1.0 - lev[rows][:, list(cols)]))
        pe = n_pe = 0.0
        for idx in spins:
            B = spin_basis(obs, idx, th[T0], th[RHO], model.facts)
            for c in cols:
                # the pure error on the fit's own weights, so the model's
                # residual splits into it and the lack of fit
                sse, w, _ = _pure_error(B * wh[idx, c:c + 1], (obs.y[idx, c] * wh[idx, c])[:, None])
                pe += float(sse[0])
                n_pe += float(np.sum(w))
        blocks[name] = (c2 - pe, dof - n_pe)
    dof1 = 6 * len(inp.poses) - 14
    blocks["poses"] = (helmert * dof1 + float(kp.eta @ kp.eta), dof1 + 14.0 - float(np.trace(Hi[NP:, NP:])))
    dT_cal = die_mean(obs)
    U0 = mounting(th)
    tt = terms(th, dT_cal, U0)
    Jt_ = terms_jacobian(th, dT_cal)
    St = Jt_ @ Sigma @ Jt_.T
    return CalResult(th, Sigma, kin, Sig_k, helmert, tt, St, dT_cal, blocks, np.sqrt(nd2), obs, conv, it,
                     rails, gaps, follow, model, short, nd_dof, dk=kp.dk, lost=lost)


# ================================================================ VERDICT
@dataclass
class Verdict:
    ok: bool
    why: list                  # [str]: every reason, the first decides


BLOCKS = ("still gyro", "still accel", "spin gyro", "spin accel", "poses")   # fit()'s, in its order


def accept_z(n_tests=len(BLOCKS) + 1, alpha=None):
    """How many spreads from nominal verdict lets a judged term lie, the
    part's spread and the fit's sigma together: alpha shared over every
    term and every test (Bonferroni)."""
    alpha = Quality.CAL_QUAL_ALPHA if alpha is None else alpha
    return normal_quantile(1.0 - alpha / (2.0 * NT * n_tests))


def verdict(res, inp, alpha=None):
    """Accept or reject a bat, with every reason.  A healthy bat on a
    healthy station is rejected with probability about alpha, shared
    among the tests (Bonferroni): Quality.CAL_QUAL_ALPHA."""
    alpha = Quality.CAL_QUAL_ALPHA if alpha is None else alpha
    n_tests = len(res.blocks) + 1
    # a stream cut short is the cause of whatever else follows from it, so
    # it comes first
    why = ["the stream (%d samples, %.1f s) does not hold %d of the plan's %d quiet intervals in full: it was cut short"
           % (len(inp.stream.k), len(inp.stream.k) / Imu.ODR, res.lost, len(inp.quiet))] if res.lost else []
    if res.failed:
        return Verdict(False, why + ["the fit failed: " + res.failed])
    if not res.converged:
        why.append("the fit did not converge")
    if res.rails:
        why.append("%d railed samples in what was used" % res.rails)
    if res.gaps:
        why.append("gaps in the stream: %d samples missing inside what was used" % res.gaps)
    if res.short:
        why.append("%d spins turned less than a whole turn quietly" % res.short)
    if res.follow > inp.lag:
        why.append("following error: the gimbal fell %.3f deg behind its command (%.3f allowed)"
                   % (np.degrees(res.follow), np.degrees(inp.lag)))
    if res.nd_dof < noise_dof():
        why.append("the noise was measured on %d dof, fewer than %d" % (res.nd_dof, noise_dof()))
    for name, (c2, dof) in res.blocks.items():
        # the poses' block rests on the cameras' covariance, the gimbal's
        # refinement on its prior; the IMU's blocks are scaled by noise
        # measured on three channels' nd_dof each
        lim = chi2_quantile(dof, alpha / n_tests) if name == "poses" else \
            dof * f_quantile(dof, 3.0 * max(res.nd_dof, 1), alpha / n_tests)
        if c2 > lim:
            why.append("%s: chi-square %.0f on %.0f dof, above %.0f" % (name, c2, dof, lim))
    sd = np.sqrt(np.diag(res.Sigma_terms))
    bad = np.nonzero(sd > inp.s)[0]
    for j in bad:
        why.append("%s %d: sigma %.3g over its %.3g" % (GROUPS[GROUP_OF[j]][0], j, sd[j], inp.s[j]))
    # how far a term lies from the drawing is the part's own spread and the
    # fit's error together: where the fit's sigma rivals the spread (the
    # lever arm's), the spread alone would refuse healthy bats
    nom = terms(inp.theta0, res.dT_cal, mounting(res.theta))
    z = accept_z(n_tests, alpha)
    off = np.abs(res.terms - nom) / np.sqrt(inp.spread ** 2 + sd ** 2)
    for j in np.nonzero(off > z)[0]:
        why.append("%s %d: %.1f of its spread from nominal" % (GROUPS[GROUP_OF[j]][0], j, off[j]))
    return Verdict(not why, why)
