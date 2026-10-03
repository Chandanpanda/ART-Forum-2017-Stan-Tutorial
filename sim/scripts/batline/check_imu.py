"""Tier 1: the board's IMU and the station's model of it -- the gimbal's
closed-form motion (imu/kin.py) against MuJoCo, the steppers' lag, the
simulated part (imu/model.py) against its datasheet (imu/part.py), and the
wall between the station's procedures and the simulation's truth.

    python3 sim/scripts/batline/check_imu.py [-v]

The plan's contract for this suite (README, Milestone 3): it fails when a
simulated IMU disagrees with the physics in a known motion, or saturates
where it should not.  The physics was developed on the kid's bat; the
adult's (check_bat's second size) it has never seen.  Each bat's station
is drawn as built (sim.Station.draw: the gimbal's centre, axes and encoder
zeros off the drawing, the base off level) and its bat drawn into the
clamp (sim.draw_bat), so no axis and no lever arm is the drawing's.

Every tolerance is derived where it is used, and says from what: float
arithmetic as Higham's gamma_n of the roundings an entry passes through, a
format's last written digit, an integrator's own error bound, the
truncated filter's own remainder, an LSB, a chi-square quantile, a Monte
Carlo's own standard error.  The statistical ones are held at
Module.Z_SIGMAS, the module's convention, shared among a check's tests
(Bonferroni).  Each check has a must-fail: a deliberately wrong variant the
same test must reject, so no check passes because it cannot fail.

PHYSICS, each bat:
   0  the scene is the build: MuJoCo's compiled chain -- the centre, both
      axes, the inner axis' offset -- is the drawn build to the MJCF's last
      written digit, its bodies unturned.  Must fail: the drawing's chain.
   1  still poses (12, gravity's direction spread over the clamp's sphere
      by a spherical Fibonacci set): MuJoCo's accelerometer and gyro at the
      IMU's site equal kin's f = R^T (g up) and w = 0; sim.signals, the
      path the simulation runs, equals the same with the Earth's turn
      R^T Omega added; Omega itself equals the turn built from east-north-
      up at every heading.  Must fail: R where R^T belongs; gravity off;
      the Earth's turn dropped; the heading taken anticlockwise.
   2  constant spin of each hinge at its fastest rate (plan.limits'
      w_spin; on the kid, again at the plan's own spins): MuJoCo's sensors
      equal kin.motion + specific_force.  Must fail: a hinge's axis
      flipped.
   3  a ramp of each hinge and a two-axis move: the w' x r term present,
      kin's angular acceleration MuJoCo's.  Must fail: w' x r dropped.
   4  inertia and torque at random states: kin.hinge_inertia and kin.rnea
      on rig/inertia.py's numpy inertials equal MuJoCo's mass matrix
      (mj_fullM) and inverse dynamics (qfrc_inverse).  Must fail: the
      rotors' armature left out; the solver's proxy boxes weighed.
   5  the steppers' lag ODE (sim._lag, RK4) after one step of command
      acceleration on a constant inertia: its linear spring (k=) equals
      kin.ringing's closed form within RK4's own error bound; the stepper's
      sine law, at a step whose static lag is one encoder code, equals the
      linear spring within the sine's own (N e)^2/6 shortfall, as the
      ring's largest gain carries it.  Must fail: the damping, or the
      stiffness, doubled; the command at the plan's spin rates, where the
      pull-out torque is less than holding; a step of command at the
      pull-out torque, which the sine law refuses as a lost step.

SENSOR, the kid's plan where one is needed:
   6  f_int: doubling sim.internal_rate's rate moves no noiseless sample
      by more than LSB/sqrt(12), about each hinge's hardest corner (the
      seconds it probes) and at the start of every quiet interval it did
      not probe, where the fit's data begin.  Must fail: f_int = ODR.
   7  the output filter: model.fir with part.impulse has the gain and
      phase of the analytic Butterworth (bilinear, prewarped) at bw/2 and
      bw, -3 dB and -90 deg at bw; a tone at ODR - 50 Hz comes out of
      model.sense attenuated by |H|.  Must fail: decimating before
      filtering.
   8  noise: a simulated still's W-sample means scatter as nd^2
      part.window_noise(W), and single samples as nd^2 window_noise(1),
      within chi-square bounds, every channel.  Must fail: the noise added
      after the filter; its sd set to the density itself.
   9  rounding: part.code_var equals a Monte Carlo of model.sense's own
      path (noise on, codes on) at nu = 0, 1/4, 1/2 of a code, within the
      Monte Carlo's standard error, Z shared among the 12 comparisons (6
      channels, n = 1 and W).  Must fail: the white-quantisation model,
      rejected at nu = 0 or 1/2 by the same test.
  10  range and codes: LSB = FS / 32768 from spec.Imu; a code reads
      k LSB as k at both ends of the range, flagged as a rail exactly at
      the two end codes (the station sees only codes); codes stay within
      int16 driven to four times full scale.  Must fail: FS / 65536; no
      clip.
  11  saturation, per axis: on the part plan.limits budgets for (scale and
      cross-axis at its quantile, gain 1 + s), codes are exact to half an
      LSB up to the plan's per-axis limit FS u / (1 + s), u =
      part.unrailed() keeping the top code unread, and on a part of
      gain 1 - s past full scale while the output is in range; an output
      at and past full scale rails and is flagged on its own axis alone.
      Must fail: clipping the vector's norm; clipping before the gain.
  12  no false saturation: the plan's motion at its peaks, as the motors
      run it, through the quantile parts and parts drawn at the spreads:
      no rail; the study's rated swing (imu_fusion_sim's Truth for
      Rules.RATED_SWING) fits the +-4000 dps / 32 g part.  Must fail: the
      same swing on a 2000 dps part rails.
  13  clock: over the plan's duration run_station's count
      (sim.sample_count) is T / (T_nom (1+eps)) within +-1, model.sense
      sends that many, and the stream's own codes (a ramp) measure that
      count.  Must fail: sample_count's count on the nominal clock, against
      what the codes measure.
  14  warm-up: the output's bias follows the tempco times the die's
      first-order rise after power-up, and the die's codes follow the
      rise.  Must fail: the rise left out; the tempco's sign flipped.
  15  truth separation: the production modules (imu/fit, plan, kin, part,
      record, golden; line.py, which times the station by them) import
      neither MuJoCo nor a TRUTH module (scene, model, sim, judge), by
      their source and in a fresh interpreter; the kid's plan designs
      without loading either.  Must fail: fifteen planted slips, each
      caught by the same scan.
  16  the plan's thinned spin ranking (plan.SPIN_RANK_PER_TURN through
      plan.nominal_obs) gives every term's sigma within 1 % of every
      window's on the kid's plan (plan.py's claim).  Must fail: the kept
      windows not standing for the ones skipped.
"""
import os
import re
import sys
import time
from multiprocessing import get_context

# four workers on four cores: one BLAS thread each, or the solves fight
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from truss.glenv import headless        # noqa: E402  (before mujoco)
headless()
import numpy as np                       # noqa: E402

from batline import product              # noqa: E402
from batline.spec import (KID, Bat, Imu, Module, Quality, Site, Gimbal, Rules, Stepper)   # noqa: E402

VERBOSE = "-v" in sys.argv
RESULTS = []
# check_bat's second size: a bat the physics was never developed on
ADULT = Bat("adult", length=860.0, handle=300.0, grip_d=34.0, width=108.0, depth=62.0)
BATS = {"developed": (KID, 7), "unseen": (ADULT, 8)}
Z = Module.Z_SIGMAS
EPS = np.finfo(float).eps
HERE = os.path.dirname(os.path.abspath(__file__))
# float agreement, Higham's gamma_n = n eps / (1 - n eps) (Higham 2002, Accuracy
# and Stability of Numerical Algorithms, sec. 3.1) over the longest chain of
# roundings an entry passes through, both sides together.  MuJoCo's reading
# at a site: three frames composed as quaternions (7 a product), one turned
# into a matrix (6), a matrix-vector product (5), the velocities and
# accelerations carried body to body as cross products (3 each, about six of
# them): about 50.  kin's: two Rodrigues rotations (about 10 each), three
# matrix products (5 each), the rigid body's terms (about 15): about 50.
N_FLOAT = 128
# the kid's plan, designed once, in the one worker that needs it (never written to disk)
PLAN = {}
TRUTH = ("scene", "model", "sim", "judge")


def gamma(n):
    return n * EPS / (1.0 - n * EPS)


def fft_round(L, gain):
    """Rounding of a convolution through the FFT at length L (model.fir): a
    forward transform, a product and an inverse, each of log2 L butterfly
    stages of a few roundings (Higham 2002, sec. 24.1), on a signal of unit
    size through a filter of absolute gain `gain`."""
    return gamma(int(3 * 5 * np.log2(L))) * (1.0 + gain)


def check(name, ok, detail="", secs=0.0):
    RESULTS.append((name, bool(ok), detail, secs))
    return bool(ok)


class Out:
    """A worker's results, each with the seconds since the one before it."""

    def __init__(self):
        self.rows = []
        self.t = time.time()

    def __call__(self, name, ok, detail=""):
        now = time.time()
        self.rows.append((name, bool(ok), detail, now - self.t))
        self.t = now
        return bool(ok)


def _case(bat):
    from batline.rig import spec as RS
    d = product.design(bat)
    return d, RS.design(d)


def _station(bat, seed):
    """(d, rig, Station as built, SimBat drawn into its clamp, the IMU's site placed there)."""
    from batline.imu import sim as S
    d, rig = _case(bat)
    rng = np.random.default_rng(seed)
    st = S.Station.draw(rig, rng)
    sb = S.draw_bat(d, rig, rng)
    st.scene.place(sb.r, sb.R_ci)
    return d, rig, st, sb


def built_chain(sc):
    """kin.Chain as MuJoCo compiled the scene: the outer body's place, both
    hinge axes and the inner body's offset as the model holds them, the
    encoder zeros as Scene.set adds them.  kin's arithmetic is held to
    MuJoCo's on this; check 0 holds this to the build."""
    from batline.imu.kin import Chain
    m = sc.m
    return Chain(m.body("outer").pos.copy(), m.jnt_axis[m.joint("alpha").id].copy(),
                 m.jnt_axis[m.joint("beta").id].copy(), m.body("inner").pos.copy(), sc.zero.copy())


def _mj_site(sc, q, qd, qdd, g_w):
    """(w, f) chip axes: MuJoCo's gyro and accelerometer at the IMU's site."""
    sc.gravity(g_w)
    w = np.empty((len(q), 3))
    f = np.empty((len(q), 3))
    for k in range(len(q)):
        sc.set(q[k], qd[k], qdd[k])
        w[k] = sc.read("imu_gyro")
        f[k] = sc.read("imu_acc")
    return w, f


def _kin_site(kn, q, qd, qdd, sb, g_w, drop_wd=False):
    """(w, f, scale) chip axes from kin: the clamp's motion and the rigid
    body's law at the bat's point; scale is the sum of the specific
    force's terms' sizes, what a float tolerance is a fraction of."""
    from batline.imu import kin as K
    R, x, w, wd, a = kn.motion(q, qd, qdd)
    f = K.specific_force(R, a, w, np.zeros_like(wd) if drop_wd else wd, sb.r, g_w)
    rn = np.linalg.norm(sb.r)
    scale = (np.linalg.norm(g_w) + np.linalg.norm(a, axis=1) + np.linalg.norm(w, axis=1) ** 2 * rn
             + np.linalg.norm(wd, axis=1) * rn)
    return w @ sb.R_ci, f @ sb.R_ci, scale


def _signals(st, sb, q, qd, qdd):
    """(w, f) chip axes as the simulation makes them: MuJoCo's clamp motion
    (scene.inner_motion) through sim.signals, the Earth's turn added."""
    from batline.imu import sim as S, scene as SC
    R, x, w, wd, a, tau = SC.inner_motion(st.scene, q, qd, qdd)
    z = np.zeros((len(q), 2))
    mot = S.Motion(np.zeros(len(q)), q, qd, qdd, z, R, x, w, wd, a, tau)
    return S.signals(st, mot, sb)


# ================================================================ PHYSICS
def physics(arg):
    label, bat, seed = arg
    t_start = time.time()
    out = Out()
    try:
        _physics(label, bat, seed, out)
    except Exception as e:                          # a crash is a failure, with its reason
        import traceback
        out("%s: physics ran to the end" % label, False, "%s: %s" % (type(e).__name__, traceback.format_exc()[-800:]))
    return out.rows, time.time() - t_start


def _physics(label, bat, seed, out):
    from batline.imu import kin as K, plan as P, scene as SC
    from batline.rig import mjcf as RM
    d, rig, st, sb = _station(bat, seed)
    sc = st.scene
    lim = P.limits(d, rig)
    chain = built_chain(sc)
    kn = chain.kin()
    g_w = st.g_w()

    # ---- 0: the scene compiled the build, to the MJCF's last written digit
    b = st.build
    drawn = SC.chain_of(b)
    # mjcf._v writes metres to 6 decimals: half a micrometre; an axis is
    # written to 9 decimals and normalised, which at most doubles it
    tol_p, tol_a = 0.5e-6, 2.0 * np.sqrt(3.0) * 0.5e-9
    dp = max(np.max(np.abs(chain.centre - drawn.centre)), np.max(np.abs(chain.offset_i - drawn.offset_i)))
    da = max(np.max(np.abs(chain.axis_o - drawn.axis_o)), np.max(np.abs(chain.axis_i - drawn.axis_i)))
    unturned = all(np.array_equal(sc.m.body(n).quat, [1.0, 0.0, 0.0, 0.0]) for n in ("outer", "inner"))
    out("%s: MuJoCo's gimbal is the build to the MJCF's last digit (0)" % label,
        dp <= tol_p * (1.0 + gamma(N_FLOAT)) and da <= tol_a * (1.0 + gamma(N_FLOAT)) and unturned,
        "centre/offset %.2e m (digit %.1e), axes %.2e (digit %.1e), bodies unturned %s"
        % (dp, tol_p, da, tol_a, unturned))
    dw = K.Chain.drawn()
    dp0 = max(np.max(np.abs(chain.centre - dw.centre)), np.max(np.abs(chain.offset_i - dw.offset_i)))
    da0 = max(np.max(np.abs(chain.axis_o - dw.axis_o)), np.max(np.abs(chain.axis_i - dw.axis_i)))
    out("must fail: %s: the drawing's chain as MuJoCo's (0)" % label, dp0 > tol_p or da0 > tol_a,
        "the drawing is %.2e m and %.2e off it" % (dp0, da0))

    # ---- 1: still poses
    qs = np.array([P.poses_for(v)[0] for v in P.fibonacci(12)])
    z = np.zeros_like(qs)
    w_mj, f_mj = _mj_site(sc, qs, z, z, g_w)
    R = kn.pose(qs)[0]
    up_c = np.einsum("nji,j->ni", R, -g_w)                       # R^T (g up)
    f_k = up_c @ sb.R_ci
    tol = gamma(N_FLOAT) * st.g
    e_f = np.max(np.abs(f_mj - f_k))
    e_w = np.max(np.abs(w_mj))
    out("%s: still poses: MuJoCo's IMU reads kin's R^T (g up) and no rate (1)" % label,
        e_f <= tol and e_w <= tol, "12 poses: |f - kin| %.2e m/s^2, |w| %.2e rad/s, tolerance %.2e" % (e_f, e_w, tol))
    f_bad = np.einsum("nij,j->ni", R, -g_w) @ sb.R_ci
    e_bad = np.max(np.abs(f_mj - f_bad))
    out("must fail: %s: R where R^T belongs (1)" % label, e_bad > tol, "|f - bad| %.3f m/s^2" % e_bad)
    _, f_off = _mj_site(sc, qs, z, z, np.zeros(3))
    e_off = np.max(np.abs(f_off - f_k))
    out("must fail: %s: gravity off (1)" % label, e_off > tol, "|f - kin| %.3f m/s^2" % e_off)
    # the simulation's own path: clamp motion through sim.signals, the Earth added
    Om = st.earth()
    w_s, f_s = _signals(st, sb, qs, z, z)
    w_e = np.einsum("nji,j->ni", R, Om) @ sb.R_ci
    e_sf = np.max(np.abs(f_s - f_k))
    e_sw = np.max(np.abs(w_s - w_e))
    tol_w = gamma(N_FLOAT) * np.linalg.norm(Om)
    out("%s: still poses: sim.signals gives kin's f and the Earth's turn R^T Omega (1)" % label,
        e_sf <= tol and e_sw <= tol_w, "|f| %.2e (tol %.2e), |w - R^T Omega| %.2e rad/s (tol %.2e)"
        % (e_sf, tol, e_sw, tol_w))
    out("must fail: %s: the Earth's turn dropped (1)" % label, np.max(np.abs(w_s)) > tol_w,
        "|w - 0| %.2e rad/s" % np.max(np.abs(w_s)))
    # Omega built from east-north-up: the rig's x heading h east of north,
    # z up, y = z x x; the Earth turns at OMEGA_E about the polar axis,
    # (0, cos lat, sin lat) in ENU
    worst, worst_bad = 0.0, 0.0
    for lat in (-90.0, -Site.LAT, 0.0, Site.LAT, 90.0):
        for k in range(7):                       # headings round the circle, none symmetric with another
            h = 360.0 * k / 7.0
            xr = np.array([np.sin(np.radians(h)), np.cos(np.radians(h)), 0.0])
            zr = np.array([0.0, 0.0, 1.0])
            Renu = np.stack([xr, np.cross(zr, xr), zr])
            Om_enu = K.OMEGA_E * np.array([0.0, np.cos(np.radians(lat)), np.sin(np.radians(lat))])
            ref = Renu @ Om_enu
            got = K.earth_rate(lat, zr, K.north_rig(h))
            bad = K.earth_rate(lat, zr, K.north_rig(-h))
            worst = max(worst, np.max(np.abs(got - ref)))
            worst_bad = max(worst_bad, np.max(np.abs(bad - ref)))
    up = st.up
    tilted = abs(np.linalg.norm(Om) - K.OMEGA_E) + abs(Om @ up - K.OMEGA_E * np.sin(np.radians(Site.LAT)))
    tol_e = gamma(N_FLOAT) * K.OMEGA_E
    out("%s: the Earth's turn is east-north-up's at every heading, and the tilted base's is its size and "
        "its vertical share (1)" % label, worst <= tol_e and tilted <= tol_e,
        "|Omega - ENU| %.2e, tilted %.2e rad/s, tolerance %.2e" % (worst, tilted, tol_e))
    out("must fail: %s: the heading taken anticlockwise (1)" % label, worst_bad > tol_e,
        "|Omega - ENU| %.2e rad/s" % worst_bad)

    # ---- 2: constant spins at each hinge's fastest
    spins = [(h, p, s * lim.w_spin[h]) for h in (0, 1) for p in np.linspace(-np.pi, np.pi, 4, endpoint=False)
             for s in (1.0, -1.0)]
    spin_checks(out, label, "each hinge at its fastest (limits' w_spin %.2f, %.2f rad/s)"
                % tuple(lim.w_spin), st, sb, chain, spins)

    # ---- 3: a ramp of each hinge, and a two-axis move
    ramp = []
    for h in (0, 1):
        a = lim.a_spin(h, lim.w_spin[h])
        T = lim.w_spin[h] / a
        for p in np.linspace(-np.pi, np.pi, 3, endpoint=False):
            for t in np.linspace(0.0, T, 8):
                q, qd, qdd = np.zeros(2), np.zeros(2), np.zeros(2)
                q[h], q[1 - h] = 0.5 * a * t * t, p
                qd[h], qdd[h] = a * t, a
                ramp.append((q, qd, qdd))
    # the two still poses farthest apart on both hinges at once
    i, j = max(((i, j) for i in range(len(qs)) for j in range(len(qs))),
               key=lambda ij: np.min(np.abs(qs[ij[0]] - qs[ij[1]])))
    path = K.Path(qs[i]).move(qs[j], lim.w_max, lim.a_max)
    tt = np.linspace(0.0, path.T, 400)
    q, qd, qdd = path.state(tt)
    both = np.all(qdd != 0.0, axis=1) & np.all(qd != 0.0, axis=1)
    sel = np.flatnonzero(both)[:: max(1, int(np.sum(both)) // 24)]
    states = [np.array([r[0] for r in ramp] + list(q[sel])), np.array([r[1] for r in ramp] + list(qd[sel])),
              np.array([r[2] for r in ramp] + list(qdd[sel]))]
    w_mj, f_mj = _mj_site(sc, *states, g_w)
    w_k, f_k, scale = _kin_site(kn, *states, sb, g_w)
    wd_mj = np.empty((len(states[0]), 3))
    for k in range(len(states[0])):
        sc.set(states[0][k], states[1][k], states[2][k])
        wd_mj[k] = sc.ang_acc()
    R_, x_, w_, wd_k, a_ = kn.motion(*states)
    tol = gamma(N_FLOAT) * scale
    e_f = np.max(np.abs(f_mj - f_k) / tol[:, None])
    e_w = np.max(np.abs(w_mj - w_k) / tol[:, None])
    tol_wd = gamma(N_FLOAT) * (np.linalg.norm(wd_k, axis=1) + np.linalg.norm(w_, axis=1) ** 2)
    e_wd = np.max(np.abs(wd_mj - wd_k) / tol_wd[:, None])
    out("%s: ramps and a two-axis move: MuJoCo's IMU and angular acceleration are kin's, w' x r in (3)" % label,
        max(e_f, e_w, e_wd) <= 1.0, "%d ramp states, %d of the move with both hinges turning and accelerating: "
        "worst f %.2f, w %.2f, w' %.2f of their tolerances (gamma_%d of each state's terms)"
        % (len(ramp), len(sel), e_f, e_w, e_wd, N_FLOAT))
    _, f_bad, _ = _kin_site(kn, *states, sb, g_w, drop_wd=True)
    e_bad = np.max(np.abs(f_mj - f_bad) / tol[:, None])
    out("must fail: %s: w' x r dropped (3)" % label, e_bad > 1.0,
        "worst %.2e of the tolerance (%.3f m/s^2)" % (e_bad, np.max(np.abs(f_mj - f_bad))))

    # ---- 4: inertia and torque
    Io, Ii = sc.numpy_inertials()
    arm = SC.armature()
    rng = np.random.default_rng(seed + 1000)
    n = 40
    Q = rng.uniform(-np.pi, np.pi, (n, 2))
    Qd = rng.uniform(-1.0, 1.0, (n, 2)) * lim.w_spin
    Qdd = rng.uniform(-1.0, 1.0, (n, 2)) * lim.a_max
    ngeom = sc.m.ngeom
    tol_n = gamma(ngeom + N_FLOAT)               # sums over every geom, in another order
    e_M = e_tau = e_arm = 0.0
    M_mj = np.empty((n, 2, 2))
    for k in range(n):
        M_mj[k] = sc.mass_matrix(Q[k])
    cols = [K.rnea(chain, Io, Ii, Q, np.zeros_like(Q), np.eye(2)[h][None].repeat(n, 0), np.zeros(3), (arm, arm))
            for h in (0, 1)]
    M_np = np.stack(cols, -1)                   # (n, 2 rows, 2 columns)
    diag = K.hinge_inertia(chain, Io, Ii, Q, (arm, arm))
    scale_M = np.abs(np.diagonal(M_mj, axis1=1, axis2=2)).max(1)
    e_M = max(np.max(np.abs(M_np - M_mj) / (tol_n * scale_M[:, None, None])),
              np.max(np.abs(diag - np.diagonal(M_mj, axis1=1, axis2=2)) / (tol_n * scale_M[:, None])))
    tau_mj = np.empty((n, 2))
    sc.gravity(g_w)
    for k in range(n):
        sc.set(Q[k], Qd[k], Qdd[k])
        tau_mj[k] = sc.torque()
    tau_np = K.rnea(chain, Io, Ii, Q, Qd, Qdd, g_w, (arm, arm))
    zq = np.zeros_like(Q)
    scale_t = (np.abs(K.rnea(chain, Io, Ii, Q, zq, Qdd, np.zeros(3), (arm, arm))).sum(1)
               + np.abs(K.rnea(chain, Io, Ii, Q, Qd, zq, np.zeros(3))).sum(1)
               + np.abs(K.rnea(chain, Io, Ii, Q, zq, zq, g_w)).sum(1))
    e_tau = np.max(np.abs(tau_np - tau_mj) / (tol_n * scale_t[:, None]))
    m_mj = sc.m.body("outer").mass[0], sc.m.body("inner").mass[0] + sc.m.body("bat").mass[0]
    out("%s: the numpy inertials' mass matrix and torques are MuJoCo's (4)" % label, e_M <= 1.0 and e_tau <= 1.0,
        "%d states: worst M %.3f, tau %.3f of gamma_%d (%d geoms summed); bodies %.4f, %.4f kg as MuJoCo's "
        "%.4f, %.4f" % (n, e_M, e_tau, ngeom + N_FLOAT, ngeom, Io.m, Ii.m, m_mj[0], m_mj[1]))
    no_arm = K.hinge_inertia(chain, Io, Ii, Q, (0.0, 0.0))
    e_arm = np.max(np.abs(no_arm - np.diagonal(M_mj, axis1=1, axis2=2)) / (tol_n * scale_M[:, None]))
    out("must fail: %s: the rotors' armature left out (4)" % label, e_arm > 1.0,
        "worst %.2e of the tolerance (armature %.3g kg m^2)" % (e_arm, arm))
    from batline.rig.inertia import inertials
    lines = [ln.replace(' mass="0" ', " ", 1) if "bat_proxy" in ln else ln for ln in sc.bodies]
    Ip = inertials("\n".join(lines), density=RM.DEFAULT_DENSITY)
    e_px = np.max(np.abs(K.hinge_inertia(chain, Ip["outer"], Ip["inner"], Q, (arm, arm))
                         - np.diagonal(M_mj, axis1=1, axis2=2)) / (tol_n * scale_M[:, None]))
    out("must fail: %s: the solver's proxy boxes weighed at MuJoCo's default density (4)" % label, e_px > 1.0,
        "inner %.3f kg against %.3f; worst %.2e of the tolerance" % (Ip["inner"].m, Ii.m, e_px))

    # ---- 5: the steppers' lag ODE against its closed form
    lag_checks(out, label, sc, lim)


def spin_checks(out, label, what, st, sb, chain, spins):
    """Check 2 at spins [(hinge, park rad, rate rad/s)], each sampled at a
    dozen angles round its turn."""
    from batline.imu.kin import Chain
    q, qd = [], []
    for h, park, rate in spins:
        for ang in np.linspace(-np.pi, np.pi, 12, endpoint=False):
            a = np.zeros(2)
            a[h], a[1 - h] = ang, park
            v = np.zeros(2)
            v[h] = rate
            q.append(a)
            qd.append(v)
    q, qd = np.array(q), np.array(qd)
    qdd = np.zeros_like(q)
    g_w = st.g_w()
    kn = chain.kin()
    w_mj, f_mj = _mj_site(st.scene, q, qd, qdd, g_w)
    w_k, f_k, scale = _kin_site(kn, q, qd, qdd, sb, g_w)
    tol = gamma(N_FLOAT) * scale
    e_f = np.max(np.abs(f_mj - f_k) / tol[:, None])
    e_w = np.max(np.abs(w_mj - w_k) / tol[:, None])
    w_s, f_s = _signals(st, sb, q, qd, qdd)
    R = kn.pose(q)[0]
    Om = np.einsum("nji,j->ni", R, st.earth()) @ sb.R_ci
    e_sf = np.max(np.abs(f_s - f_k) / tol[:, None])
    e_sw = np.max(np.abs(w_s - w_k - Om) / tol[:, None])
    out("%s: spins, %s: MuJoCo's IMU and sim.signals are kin.motion + specific_force (2)" % (label, what),
        max(e_f, e_w, e_sf, e_sw) <= 1.0, "%d states: worst f %.2f, w %.2f, signals f %.2f, w %.2f of gamma_%d of "
        "each state's terms" % (len(q), e_f, e_w, e_sf, e_sw, N_FLOAT))
    flips = []
    for h in (0, 1):
        m = np.abs(qd[:, h]) > 0.0
        if not np.any(m):
            continue
        flip = Chain(chain.centre, -chain.axis_o if h == 0 else chain.axis_o,
                     -chain.axis_i if h == 1 else chain.axis_i, chain.offset_i, chain.zero)
        wb, fb, _ = _kin_site(flip.kin(), q[m], qd[m], qdd[m], sb, g_w)
        flips.append(max(np.max(np.abs(fb - f_mj[m]) / tol[m, None]), np.max(np.abs(wb - w_mj[m]) / tol[m, None])))
    worst = min(flips)
    out("must fail: %s: spins, %s: the spun hinge's axis flipped (2)" % (label, what), worst > 1.0,
        "each spun hinge's flip: %s of the tolerance" % ", ".join("%.2e" % v for v in flips))


def lag_checks(out, label, sc, lim):
    """Check 5: one step of command acceleration on each hinge's own inertia
    at the zeros (MuJoCo's, the armature in), integrated by sim._lag on a
    grid at 4 x ODR -- the coarsest internal_rate starts from, where RK4's
    error is largest -- with the command's rate held at zero, so each
    motor's torque is its holding torque throughout.

    The linear spring (k=, the stiffness), against kin.ringing's closed
    form.  RK4's error bound on this linear system: in its two modes
    e^(lambda t), |lambda| = w_n, each step's amplification misses
    e^(h lambda) by at most |h w_n|^5 / 120 e^(h w_n) (the exponential's
    Taylor remainder), and n steps compound it to n delta (e^(-zeta w_n h)
    + delta)^(n-1); the step response's modes have weight A w_n / (2 w_d)
    each, A = da / w_n^2.  Rounding: 8 roundings of the state a step.

    The stepper's sine law (k None), at a small step -- its static lag one
    encoder code, the least the station resolves and where the plan's
    settling ends (kin.settle's tolerance) -- against the linear spring.
    The sine's lag less the spring's obeys the spring's own equation driven
    by d = k e - T sin(N e), which, the stiffness being the sine's slope at
    rest (N T = k), is at most k |e| (N e)^2 / 6: the sine's own shortfall.
    A forcing moves the lag by at most the L1 norm of the impulse response
    times its largest size, and k times that norm is coth(zeta pi / (2
    sqrt(1 - zeta^2))): the most a ring can build up any forcing.  Each
    run's RK4 error is added, bounded as above (the sine's modes are the
    spring's to the same shortfall).  Must fail: the same comparison with
    the command at the plan's spin rates, where the pull-out curve leaves
    the motors less than their holding torque; and a step of command torque
    at the pull-out torque, whose ring asks more than the motor has, which
    the sine law must refuse as a lost step."""
    from batline.imu import kin as K, plan as P, sim as S
    M0 = np.diag(sc.mass_matrix(np.zeros(2)))
    k = lim.stiffness()
    c = lim.damping()
    wn = np.sqrt(k / M0)
    zeta = c / (2.0 * np.sqrt(k * M0))
    da = lim.a_max                                   # the hardest corner a move makes
    h = 1.0 / (4.0 * Imu.ODR)
    code = 2.0 * np.pi / 2 ** Gimbal.ENCODER_BITS
    T = max(K.settle(da[j], wn[j], zeta[j], code) for j in (0, 1))      # until the ring is under a code
    t = np.arange(int(np.ceil(T / h)) + 1) * h
    n = len(t)
    M = np.tile(np.diag(M0), (n, 1, 1))
    tau = np.tile(M0 * da, (n, 1))
    w0 = np.zeros((n, 2))                            # the command's rate, held at zero

    def closed(j):
        e, ed, _ = K.ringing([(0.0, da)], t, wn, zeta[j])
        return e[:, j], ed[:, j]

    def bound(j, da_=da):
        wd = wn[j] * np.sqrt(1.0 - zeta[j] ** 2)
        A = da_[j] / wn[j] ** 2
        delta = (h * wn[j]) ** 5 / 120.0 * np.exp(h * wn[j])
        N = np.arange(n)
        b = 2.0 * A * wn[j] / (2.0 * wd) * N * delta * (np.exp(-zeta[j] * wn[j] * h) + delta) ** np.maximum(N - 1, 0)
        return b + N * 8.0 * EPS * A

    def worst(e, ed):
        out_ = 0.0
        for j in (0, 1):
            E, Ed = closed(j)
            B = bound(j) + 1e-300
            out_ = max(out_, np.max(np.abs(e[:, j] - E) / B), np.max(np.abs(ed[:, j] - Ed) / (B * wn[j])))
        return out_
    e, ed, _ = S._lag(t, tau, M, c, w0, k=k)
    r = worst(e, ed)
    out("%s: the lag ODE (sim._lag), its linear spring, is kin.ringing's closed form within RK4's bound (5)" % label,
        r <= 1.0, "a step of %.2f, %.2f rad/s^2 on w_n %.1f, %.1f rad/s, zeta %.3f, %.3f, %.2f s at h = 1/%.0f: worst "
        "%.2f of the bound (largest lag %.2e rad)" % (da[0], da[1], wn[0], wn[1], zeta[0], zeta[1], T, 1.0 / h, r,
                                                      np.max(np.abs(e))))
    for what, kf, cf in (("the damping doubled", 1.0, 2.0), ("the stiffness doubled", 2.0, 1.0)):
        e2, ed2, _ = S._lag(t, tau, M, c * cf, w0, k=k * kf)
        r2 = worst(e2, ed2)
        out("must fail: %s: %s (5)" % (label, what), r2 > 1.0, "worst %.2e of the bound" % r2)

    # the stepper's own law at a small step, against the linear spring
    N_e = Stepper.TEETH * Gimbal.GEAR                # electrical radians an axis radian
    gain = 1.0 / np.tanh(zeta * np.pi / (2.0 * np.sqrt(1.0 - zeta ** 2)))     # k times the impulse response's L1 norm
    da_s = code * wn ** 2                            # a static lag of one code
    tau_s = np.tile(M0 * da_s, (n, 1))

    def against(w_c):
        """(worst |e_sine - e_spring| over its bound, N max|e| per hinge,
        |e_sine - e_spring| over the spring's largest lag per hinge)."""
        e_l, _, _ = S._lag(t, tau_s, M, c, w_c, k=k)
        e_s, _, _ = S._lag(t, tau_s, M, c, w_c)
        em = np.max(np.abs(e_s), 0)
        gap = np.abs(e_s - e_l)
        B = np.stack([gain[j] * em[j] * (N_e * em[j]) ** 2 / 6.0 + 2.0 * bound(j, da_s) for j in (0, 1)], 1)
        return np.max(gap / B), N_e * em, np.max(gap, 0) / np.max(np.abs(e_l), 0)
    rs, x_s, rel_s = against(w0)
    out("%s: the stepper's sine law (sim._lag) at a step of a code's lag is the linear spring within the sine's "
        "own shortfall (5)" % label, rs <= 1.0,
        "N|e| up to %.4f, %.4f rad: (N e)^2/6 %.1e, %.1e through the ring's gain %.2f, %.2f; the two differ by "
        "%.1e, %.1e of the largest lag; worst %.2f of the bound" % (x_s[0], x_s[1], x_s[0] ** 2 / 6.0,
                                                                   x_s[1] ** 2 / 6.0, gain[0], gain[1], rel_s[0],
                                                                   rel_s[1], rs))
    T_hold, T_spin = P._pullout(np.zeros(2)), P._pullout(lim.w_spin)
    rb, _, rel_b = against(np.tile(lim.w_spin, (n, 1)))
    out("must fail: %s: the same with the command at the plan's spin rates (%.2f, %.2f rad/s), the pull-out torque "
        "%.0f%%, %.0f%% of holding (5)" % (label, lim.w_spin[0], lim.w_spin[1], 100.0 * T_spin[0] / T_hold[0],
                                           100.0 * T_spin[1] / T_hold[1]), rb > 1.0,
        "the two differ by %.1e, %.1e of the largest lag; worst %.2e of the bound" % (rel_b[0], rel_b[1], rb))
    # a step of command torque at the pull-out torque: the sine holds it only
    # at pi/2, and its ring, (1 + overshoot) of it, is more than the motor has
    ring = 1.0 + np.exp(-zeta * np.pi / np.sqrt(1.0 - zeta ** 2))
    try:
        S._lag(t, np.tile(T_hold, (n, 1)), M, c, w0)
        lost = "no error"
    except ValueError as ex:
        lost = str(ex)
    out("must fail: %s: a step of command at the pull-out torque, its ring %.2f, %.2f times that, is a lost step "
        "(5)" % (label, ring[0], ring[1]), "loses steps" in lost, lost)


# ================================================================ THE PLAN
def _truth_loaded():
    """MuJoCo (any part of it loads the package) and the TRUTH modules, as loaded here."""
    return sorted(k for k in sys.modules if k == "mujoco" or k in tuple("batline.imu." + t for t in TRUTH))


def plan_worker(_):
    t_start = time.time()
    out = Out()
    try:
        _plan_worker(out)
    except Exception as e:
        import traceback
        out("the kid's plan: its checks ran to the end", False,
            "%s: %s" % (type(e).__name__, traceback.format_exc()[-800:]))
    return out.rows, time.time() - t_start


def _plan_worker(out):
    from batline.imu import plan as P
    d, rig = _case(KID)
    before = _truth_loaded()
    plan = P.for_design(d)
    PLAN["kid"] = plan
    after = _truth_loaded()
    out("the kid's plan designs without MuJoCo or a TRUTH module loaded (15)", not after,
        "%d items, %.1f s; loaded before %s, after %s" % (len(plan.items), plan.duration_s, before or "none",
                                                          after or "none"))
    from batline.imu import sim as S                 # noqa: F401  (the detector must see it)
    out("must fail: the same test after importing sim (15)", bool(_truth_loaded()),
        "sees %s" % ", ".join(_truth_loaded()))
    from batline.imu import kin as K, model as MD, part as PT
    _, _, st, sb = _station(KID, BATS["developed"][1])
    chain = built_chain(st.scene)

    # ---- 2 again, at the plan's own spins
    sp = [(it.hinge, it.park, s * it.rate) for it in plan.items if isinstance(it, P.Spin) for s in (1.0, -1.0)]
    if sp:
        spin_checks(out, "kid", "the plan's own %d" % (len(sp) // 2), st, sb, chain, sp)

    # ---- 6: the internal rate
    f_int, moved = S.internal_rate(st, plan)
    lg, la = PT.lsb()
    scale = np.array([lg] * 3 + [la] * 3)
    pw = MD.PowerUp(np.zeros(3), np.zeros(3), -1e6, Site.AMBIENT)
    n = int(Imu.ODR * 0.9)                  # as internal_rate reads its second: 0.9 s of samples from 0.05 s in
    T = plan.duration_s

    def probe(fs, t0):
        mot = S.motion(st, plan, fs, span=(t0, t0 + 1.0))
        w, f = S.signals(st, mot, sb)
        _, _, ys = MD.sense(sb.err, pw, mot.t, w, f, t0 + 0.05, n, None, noise=False, quantise=False)
        return ys / scale
    # the seconds internal_rate probes (about each hinge's hardest corner, its
    # largest acceleration step) and the ones it does not: each quiet
    # interval's first second, where the fit's data begin
    spans = []
    for h in (0, 1):
        cs = [c for c in plan.tour.corners if c[1][h] != 0.0]
        if cs:
            tc = max(cs, key=lambda c: abs(c[1][h]))[0]
            spans.append(min(max(tc - 0.5, 0.0), max(T - 1.0, 0.0)))
    tt = np.linspace(0.0, T, int(T * 200) + 1)                  # where internal_rate itself probes
    t_ir = float(tt[int(np.argmax(np.abs(plan.path.state(tt)[2]).max(1)))])
    t_ir = min(max(t_ir - 0.5, 0.0), max(T - 1.0, 0.0))
    n_own = len(spans)
    for Q in plan.quiet:
        t0 = min(max(Q.t0 - 0.05, 0.0), max(T - 1.0, 0.0))
        if all(abs(t0 - s_) >= 0.5 for s_ in spans):
            spans.append(t0)
    rule = 1.0 / np.sqrt(12.0)
    moves, halves, odr = [], [], []
    for k, t0 in enumerate(spans):
        a, b2 = probe(f_int, t0), probe(2.0 * f_int, t0)
        moves.append(np.max(np.abs(a - b2)))
        halves.append(np.max(np.abs(probe(f_int / 2.0, t0) - a)) if k < n_own else np.nan)
        if k < n_own:
            odr.append(np.max(np.abs(probe(Imu.ODR, t0) - probe(2.0 * Imu.ODR, t0))))
    out("the internal rate: doubling f_int moves no noiseless sample by more than LSB/sqrt(12) (6)",
        max(moves) <= rule, "f_int %.0f Hz (its own probes; the max-acceleration second from %.2f s: %.3f codes); "
        "about each hinge's hardest corner (from %s s) doubling moves %s codes, the doubling before %s; at %d quiet "
        "starts it did not probe, at most %.3f; against %.3f"
        % (f_int, t_ir, moved, ", ".join("%.2f" % s_ for s_ in spans[:n_own]),
           ", ".join("%.3f" % v for v in moves[:n_own]), ", ".join("%.3f" % v for v in halves[:n_own]),
           len(spans) - n_own, max(moves[n_own:], default=0.0), rule))
    out("must fail: f_int = ODR (6)", max(odr) > rule, "doubling moves %s codes" % ", ".join("%.3f" % v for v in odr))

    # ---- 12: the plan's motion at its peaks, through parts at the spreads
    rate_checks(out, plan, st, sb, chain, f_int)

    # ---- 13: the clock over the plan
    clock_checks(out, plan, sb)

    # ---- 16: the thinned spin ranking
    th = plan.theta0
    kn = K.Chain.drawn().kin()
    W = plan.W
    val = P.Value(th, plan.s, plan.spread, weak=False)
    q_st = [plan.tour.path.state(np.array([Q.t0]))[0][0] for Q in plan.tour.quiet if Q.kind == "still"]
    full = val.sigma(P._exact_info(plan.tour, th, kn, W))
    obs = P.nominal_obs(plan.tour.quiet, plan.tour.path, W, per_turn=P.SPIN_RANK_PER_TURN)
    thin = val.sigma(P.information(obs, th, kn, poses_q=q_st, with_kin=True))
    rel = np.max(np.abs(thin / full - 1.0))
    n_full = len(P.nominal_obs(plan.tour.quiet, plan.tour.path, W))
    out("the thinned spin ranking (%d windows a turn) gives every sigma within 1%% of every window's (16)"
        % P.SPIN_RANK_PER_TURN, rel <= 0.01, "%d rows for %d: worst %.2f%%" % (len(obs), n_full, 100.0 * rel))
    # the kept windows each standing for one, not for the m they replace
    m_of = {}
    for qi, Q in enumerate(plan.tour.quiet):
        if Q.kind == "spin":
            m_of[qi] = max(1, int(2.0 * np.pi * Imu.ODR / (abs(Q.rate) * W * P.SPIN_RANK_PER_TURN)))
    mult = np.array([m_of.get(int(qi), 1) if mv else 1 for qi, mv in zip(obs.quiet, obs.moving)], float)
    obs.var = obs.var * mult[:, None]
    bad = val.sigma(P.information(obs, th, kn, poses_q=q_st, with_kin=True))
    rel_b = np.max(np.abs(bad / full - 1.0))
    out("must fail: the kept windows not standing for the ones skipped (16)", rel_b > 0.01,
        "worst %.1f%%" % (100.0 * rel_b))


def quantile_part(sign=1.0):
    """The part plan.limits budgets for: scale and cross-axis at the
    quantile z it takes (no channel of a healthy part rails but with
    probability CAL_QUAL_ALPHA / 12), laid where they add -- every
    channel's gain on an input with equal components is 1 + sign s,
    s = z sqrt(scale^2 + 2 cross^2): E_ii = z scale^2 / S, E_ij = z cross^2
    / S, the likeliest such E -- and the accelerometer's bias at the same
    quantile, as acc_fs budgets it.  The gyro's bias and g-sensitivity
    are left out, as gyro_fs leaves them out."""
    from batline.imu import model as MD
    from batline.imu.part import G0
    from batline.rig.pose import normal_quantile
    z = normal_quantile(1.0 - Quality.CAL_QUAL_ALPHA / 12.0)

    def E(ss, sc):
        S = np.sqrt(ss ** 2 + 2.0 * sc ** 2)
        M = np.full((3, 3), z * sc ** 2 / S)
        np.fill_diagonal(M, z * ss ** 2 / S)
        return sign * M
    ba = sign * z * Imu.ACC_BIAS * G0 * np.ones(3)
    return MD.ImuErrors(Ea=E(Imu.ACC_SCALE, Imu.ACC_CROSS), Eg=E(Imu.GYRO_SCALE, Imu.GYRO_CROSS), ba=ba,
                        bg=np.zeros(3), G=np.zeros((3, 3)), ca=np.zeros(3), cg=np.zeros(3), eps=0.0), z


def plan_limits():
    """(gyro rad/s, accel m/s^2, s_g, s_a): each axis' input the plan
    allows a part at its quantile, as plan.limits takes it: FS u / (1 + s),
    the accelerometer's less its bias at the quantile (the total specific
    force, gravity in: plan's acc_fs + g)."""
    from batline.imu.part import G0, D2R, unrailed
    _, z = quantile_part()
    s_g = z * np.sqrt(Imu.GYRO_SCALE ** 2 + 2.0 * Imu.GYRO_CROSS ** 2)
    s_a = z * np.sqrt(Imu.ACC_SCALE ** 2 + 2.0 * Imu.ACC_CROSS ** 2)
    u = unrailed()
    return (Imu.GYRO_FS * u * D2R / (1.0 + s_g), Imu.ACC_FS * u * G0 / (1.0 + s_a) - z * Imu.ACC_BIAS * G0, s_g, s_a)


def rate_checks(out, plan, st, sb, chain, f_int):
    """Check 12, the plan's half: its motion's peaks, as the motors run it."""
    from batline.imu import kin as K, model as MD, sim as S
    from batline.imu.part import impulse
    T = plan.duration_s
    tt = np.arange(0.0, T, 1.0 / Gimbal.ENCODER_HZ)          # as finely as the station reads the hinges
    q, qd, qdd = plan.path.state(tt)
    R, x, w, wd, a = chain.kin().motion(q, qd, qdd)
    f = K.specific_force(R, a, w, wd, sb.r, st.g_w()) @ sb.R_ci
    wc = (w + np.einsum("nji,j->ni", R, st.earth())) @ sb.R_ci
    lim_g, lim_a, _, _ = plan_limits()
    peaks = sorted({int(np.argmax(np.abs(wc).max(1))), int(np.argmax(np.abs(f).max(1)))})
    rng = np.random.default_rng(BATS["developed"][1] + 12)
    parts = [quantile_part(+1.0)[0], quantile_part(-1.0)[0]] + [MD.draw_errors(rng) for _ in range(8)]
    rails = 0
    worst = 0.0
    for i in peaks:
        t0 = min(max(tt[i] - 0.5, 0.0), max(T - 1.0, 0.0))
        mot = S.motion(st, plan, f_int, span=(t0, t0 + 1.0))
        w_s, f_s = S.signals(st, mot, sb)
        t_first = t0 + (len(impulse(f_int)) + 1) / f_int            # past the filter's start
        for p in parts:
            pw = MD.power_up(rng, 0.0, Site.AMBIENT)
            n = S.sample_count(mot.t[-1], t_first, p.eps)                  # run_station's count
            stream, _, ys = MD.sense(p, pw, mot.t, w_s, f_s, t_first, n, rng)
            rails += int(np.sum(stream.rail))
            rail_code = 2 ** (Imu.BITS - 1)
            worst = max(worst, np.max(np.abs(stream.gyro)) / rail_code, np.max(np.abs(stream.acc)) / rail_code)
    out("no false saturation: the plan's motion at its peaks through %d parts at the spreads rails nowhere (12)"
        % len(parts), rails == 0,
        "peaks %.0f dps and %.2f g at the IMU (the plan's per-axis limits %.0f dps, %.1f g); the worst code "
        "%.1f%% of the rail" % (np.degrees(np.abs(wc).max()), np.abs(f).max() / 9.80665, np.degrees(lim_g),
                                lim_a / 9.80665, 100.0 * worst))


def clock_checks(out, plan, sb):
    """Check 13: a ramp on the gyro's x over the plan, the part's clock at
    Z sigmas of the datasheet's spread either way and drawn: the count
    run_station takes (sim.sample_count) is T / (T_nom (1 + eps)) within
    one, model.sense sends that many, and the codes measure it.  The
    must-fail is the same function's count on the nominal clock, against
    what the codes measure."""
    from batline.imu import model as MD, sim as S
    from batline.imu.part import lsb, impulse
    rng = np.random.default_rng(BATS["developed"][1] + 13)
    fs = 4.0 * Imu.ODR                      # a ramp needs no finer grid: internal_rate's start
    T_end = plan.duration_s
    t = np.arange(int(np.ceil(T_end * fs)) + 1) / fs
    lg, _ = lsb()
    FS = Imu.GYRO_FS * np.pi / 180.0
    slope = 1.8 * FS / T_end                 # -0.9 FS to 0.9 FS over the plan
    w = np.zeros((len(t), 3))
    w[:, 0] = slope * (t - T_end / 2.0)
    f = np.zeros((len(t), 3))
    rows, bad = [], []
    for eps in (Z * Imu.ODR_TOL, -Z * Imu.ODR_TOL, MD.draw_errors(rng).eps):
        err = MD.ImuErrors(np.zeros((3, 3)), np.zeros((3, 3)), np.zeros(3), np.zeros(3), np.zeros((3, 3)),
                           np.zeros(3), np.zeros(3), float(eps))
        pw = MD.power_up(rng, 0.0, Site.AMBIENT)
        t_first = Imu.START_S + rng.uniform(0.0, 1.0) / Imu.ODR
        n = S.sample_count(T_end, t_first, eps)               # run_station's count
        stream, tk, _ = MD.sense(err, pw, t, w, f, t_first, n, rng)
        span = T_end - t_first
        want = span / ((1.0 / Imu.ODR) * (1.0 + eps))
        # the codes' own clock: past the filter's start, a line through k
        k0 = int(np.ceil(len(impulse(fs)) / fs * Imu.ODR)) + 1
        k = np.arange(k0, len(stream.gyro))
        X = np.stack([np.ones(len(k)), k], 1)
        c, *_ = np.linalg.lstsq(X, stream.gyro[k, 0].astype(float), rcond=None)
        eps_m = c[1] * lg * Imu.ODR / slope - 1.0
        n_m = span * Imu.ODR / (1.0 + eps_m)
        rows.append((eps, n, len(stream.gyro), want, n_m))
        if len(bad) < 2:                       # the two at the spread's Z sigmas
            bad.append(abs(S.sample_count(T_end, t_first, 0.0) - n_m))
    ok = all(abs(n - want) <= 1.0 and sent == n and abs(n - n_m) <= 1.0 for _, n, sent, want, n_m in rows)
    out("the clock: over the plan run_station's count is T / (T_nom (1+eps)) within one, the part sends it, and its "
        "codes say so (13)", ok,
        "; ".join("eps %+.4f: sample_count %d, sent %d, T/(T_nom(1+eps)) %.2f, from the codes %.2f" % r for r in rows))
    out("must fail: the nominal clock's count, against the codes (13)", min(bad) > 1.0,
        "off by %s samples" % ", ".join("%.0f" % b for b in bad))


# ================================================================ SENSOR
def sensor_worker(_):
    t_start = time.time()
    out = Out()
    try:
        _sensor_worker(out)
    except Exception as e:
        import traceback
        out("the sensor checks ran to the end", False, "%s: %s" % (type(e).__name__, traceback.format_exc()[-800:]))
    return out.rows, time.time() - t_start


def _perfect(eps=0.0):
    from batline.imu import model as MD
    z3 = np.zeros(3)
    return MD.ImuErrors(np.zeros((3, 3)), np.zeros((3, 3)), z3, z3, np.zeros((3, 3)), z3, z3, eps)


def _still_pw():
    from batline.imu import model as MD
    from batline.imu.part import T_REF
    return MD.PowerUp(np.zeros(3), np.zeros(3), 0.0, T_REF)


def _held(err, X, fs=None):
    """(codes (k, 6) gyro then accel, rail (k,), DC gain): the part reading
    each row of X (k, 6: gyro rad/s, accel m/s^2, chip axes), each held on
    the internal grid longer than the filter's memory and read at the end
    of its hold -- where the (truncated) filter has forgotten the row
    before."""
    from batline.imu import model as MD
    from batline.imu.part import impulse
    fs = 4.0 * Imu.ODR if fs is None else fs
    per = int(round(fs / Imu.ODR))
    hold = int(np.ceil(len(impulse(fs)) / per)) + 2              # output samples a row is held
    k = len(X)
    n_int = k * hold * per + per
    t = np.arange(n_int) / fs
    Y = np.repeat(np.asarray(X, float), hold * per, axis=0)
    Y = np.vstack([Y, Y[-1:].repeat(per, 0)])
    stream, tk, ys = MD.sense(err, _still_pw(), t, Y[:, :3], Y[:, 3:], 0.0, k * hold, None, noise=False)
    idx = np.arange(1, k + 1) * hold - 1
    codes = np.concatenate([stream.gyro[idx], stream.acc[idx]], 1)
    return codes, stream.rail[idx], float(impulse(fs).sum())


def _out_of(err, X):
    """(k, 6) the part's noiseless output for rows X: (I + E) x + G f + b."""
    X = np.asarray(X, float)
    g = X[:, :3] @ (np.eye(3) + err.Eg).T + X[:, 3:] @ err.G.T + err.bg
    a = X[:, 3:] @ (np.eye(3) + err.Ea).T + err.ba
    return np.concatenate([g, a], 1)


def _sensor_worker(out):
    from batline.imu import model as MD
    from batline.imu.part import G0, D2R, lsb, impulse, butterworth2
    lg, la = lsb()
    step = np.array([lg] * 3 + [la] * 3)
    FS = np.array([Imu.GYRO_FS * D2R] * 3 + [Imu.ACC_FS * G0] * 3)
    full = 2 ** (Imu.BITS - 1)
    lo, hi = -full, full - 1

    # ---- 7: the output filter
    fc = Imu.FILTER_BW

    def H_exact(f, fs):
        """The analog 2nd-order Butterworth through the bilinear transform,
        its cut-off prewarped: H(f) = H_a(j tan(pi f/fs) / tan(pi fc/fs))."""
        s = 1j * np.tan(np.pi * f / fs) / np.tan(np.pi * fc / fs)
        return 1.0 / (1.0 + np.sqrt(2.0) * s + s * s)

    def remainder(fs):
        """sum |h_k| past where part.impulse cut it: the recursion carried
        on, with no input, from its last two outputs, until rho^k is under
        eps (rho the poles' radius)."""
        h = impulse(fs)
        b, a = butterworth2(fc, fs)
        rho = max(abs(np.roots(a)))
        y1, y2 = h[-1], h[-2]
        tot = 0.0
        for _ in range(int(np.ceil(np.log(EPS) / np.log(rho))) + 2):
            y0 = -a[1] * y1 - a[2] * y2
            tot += abs(y0)
            y2, y1 = y1, y0
        return tot, float(np.abs(h).sum())

    rows, worst = [], 0.0
    for fs in (4.0 * Imu.ODR, 16.0 * Imu.ODR, 32.0 * Imu.ODR):
        h = impulse(fs)
        R_t, S_h = remainder(fs)
        for f in (fc / 2.0, fc):
            t = np.arange(int(np.ceil(fs * (64.0 / f + len(h) / fs)))) / fs     # 64 periods past the filter's memory
            y = MD.fir(np.sin(2.0 * np.pi * f * t)[:, None], h)[:, 0]
            m = np.arange(len(t)) >= len(h)
            X = np.stack([np.sin(2.0 * np.pi * f * t[m]), np.cos(2.0 * np.pi * f * t[m])], 1)
            cf, *_ = np.linalg.lstsq(X, y[m], rcond=None)
            Hm = cf[0] + 1j * cf[1]
            L = 1 << int(np.ceil(np.log2(len(t) + len(h))))
            tol = R_t + fft_round(L, S_h)
            worst = max(worst, abs(Hm - H_exact(f, fs)) / tol)
            rows.append((fs, f, abs(Hm), np.degrees(np.angle(Hm))))
    at_bw = [r for r in rows if r[1] == fc]
    m3db = max(abs(r[2] - 1.0 / np.sqrt(2.0)) for r in at_bw)
    m90 = max(abs(r[3] + 90.0) for r in at_bw)
    tail = max(remainder(fs)[0] / remainder(fs)[1] for fs in (4.0 * Imu.ODR, 16.0 * Imu.ODR, 32.0 * Imu.ODR))
    out("the output filter's gain and phase are the analytic Butterworth's at bw/2 and bw (7)", worst <= 1.0,
        "worst %.2f of its tolerance (the cut tail + FFT rounding); at bw %.9f off -3 dB, %.2e deg off -90; "
        "%s; part.impulse's cut tail is %.1e of the whole (it means to keep it under 1e-9)"
        % (worst, m3db, m90, ", ".join("%.0f Hz at %.0f: %.6f %.3f deg" % (r[1], r[0], r[2], r[3]) for r in rows),
           tail))
    # a tone near ODR - 50 Hz through model.sense: attenuated by |H| before
    # it is sampled; decimated first, it would alias to 50 Hz whole
    fs = 16.0 * Imu.ODR
    f_t = Imu.ODR - 50.0
    T = 1.0
    t = np.arange(int(np.ceil(T * fs)) + 1) / fs
    wv = np.zeros((len(t), 3))
    wv[:, :] = (0.5 * FS[0]) * np.sin(2.0 * np.pi * f_t * t)[:, None]
    t_first = np.ceil(len(impulse(fs)) / (fs / Imu.ODR)) / Imu.ODR      # on the grid, past the filter's start
    n = int((T - t_first) * Imu.ODR) - 1
    _, tk, ys = MD.sense(_perfect(), _still_pw(), t, wv, np.zeros_like(wv), t_first, n, None, noise=False,
                         quantise=False)
    X = np.stack([np.sin(2.0 * np.pi * f_t * tk), np.cos(2.0 * np.pi * f_t * tk)], 1)
    cf, *_ = np.linalg.lstsq(X, ys[:, 0], rcond=None)
    gain = np.hypot(*cf) / (0.5 * FS[0])
    R_t, S_h = remainder(fs)
    L = 1 << int(np.ceil(np.log2(len(t) + len(impulse(fs)))))
    tol = R_t + fft_round(L, S_h)
    want = abs(H_exact(f_t, fs))
    out("a tone at ODR - 50 Hz leaves model.sense attenuated by |H| (7)", abs(gain - want) <= tol,
        "%.0f Hz: gain %.9f, |H| %.9f (the analog's %.6f), tolerance %.1e" % (f_t, gain, want,
                                                                         abs(1.0 / (1.0 + np.sqrt(2.0) * 1j * f_t / fc
                                                                                    - (f_t / fc) ** 2)), tol))
    # decimated first: the tone read at the sample instants, then filtered at ODR
    xs = (0.5 * FS[0]) * np.sin(2.0 * np.pi * f_t * tk)
    yd = MD.fir(xs[:, None], impulse(Imu.ODR))[:, 0]
    k0 = len(impulse(Imu.ODR))
    cf, *_ = np.linalg.lstsq(X[k0:], yd[k0:], rcond=None)
    gain_d = np.hypot(*cf) / (0.5 * FS[0])
    out("must fail: decimating before filtering (7)", abs(gain_d - want) > tol,
        "gain %.4f against |H| %.4f" % (gain_d, want))

    # ---- 8: noise
    noise_checks(out)

    # ---- 10: range and codes
    want_lsb = (Imu.GYRO_FS * np.pi / 180.0 / 32768.0, Imu.ACC_FS * 9.80665 / 32768.0)
    ok_lsb = full == 32768 and all(abs(a - b) <= EPS * b for a, b in zip(lsb(), want_lsb))
    ks = np.array([lo, lo + 1, -1, 0, 1, hi - 1, hi], float)
    codes, rail, _ = _held(_perfect(), ks[:, None] * step[None, :])
    ok_k = np.array_equal(codes, np.repeat(ks[:, None], 6, 1)) and np.array_equal(rail, np.isin(ks, (lo, hi)))
    out("one LSB is FS / 32768 (spec.Imu), and k LSB reads k at both ends of the range (10)", ok_lsb and ok_k,
        "LSB %.6e rad/s, %.6e m/s^2; codes %s" % (lsb() + ([int(v) for v in codes[:, 0]],)))
    bad_lsb = (Imu.GYRO_FS * D2R / 2.0 ** Imu.BITS, Imu.ACC_FS * G0 / 2.0 ** Imu.BITS)
    out("must fail: FS / 65536 as the LSB (10)", not all(abs(a - b) <= EPS * b for a, b in zip(bad_lsb, want_lsb)),
        "%.6e against %.6e rad/s" % (bad_lsb[0], want_lsb[0]))
    levels = np.linspace(-4.0, 4.0, 33)
    Xr = levels[:, None] * FS[None, :]
    codes, rail, _ = _held(_perfect(), Xr)
    i16 = np.iinfo(np.int16)
    in16 = codes.min() >= i16.min and codes.max() <= i16.max
    out("codes stay within int16 driven to four times full scale (10)", in16,
        "codes %d..%d; %d of %d rows flagged" % (codes.min(), codes.max(), int(rail.sum()), len(rail)))
    raw = np.round(_out_of(_perfect(), Xr) / step)
    out("must fail: no clip (10)", not (raw.min() >= i16.min and raw.max() <= i16.max),
        "codes %d..%d" % (raw.min(), raw.max()))

    # ---- 11: saturation, per axis
    saturation_checks(out, step, FS, lo, hi)

    # ---- 12: the study's rated swing
    swing_checks(out)

    # ---- 14: warm-up
    warmup_checks(out)

    # ---- 15: the wall between production and truth
    separation_checks(out)


def _windows(ys, n, per):
    """(m, 6) means of the first n samples of each run of `per`."""
    m = len(ys) // per
    return ys[: m * per].reshape(m, per, -1)[:, :n].mean(1)


def _stills(value, ns, m, fs, rng, quantise):
    """({n: (m, 6)} means of the first n samples of m windows, in codes, and
    the windows' spacing): a perfect part reading `value` (6,) still,
    through model.sense in chunks, each its own stream begun past its
    filter's start.  The windows lie past the truncated filter's memory
    (part.impulse at fs) from each other, so their noise is independent."""
    from batline.imu import model as MD
    from batline.imu.part import impulse, lsb
    lg, la = lsb()
    step = np.array([lg] * 3 + [la] * 3)
    h = impulse(fs)
    gap = int(np.ceil(len(h) / (fs / Imu.ODR))) + 1
    per = max(ns) + gap
    n_grid = (1 << 19) - len(h) - 2               # a chunk: its FFT at 2^19
    t = np.arange(n_grid) / fs
    W = np.broadcast_to(np.asarray(value[:3], float), (n_grid, 3))
    F = np.broadcast_to(np.asarray(value[3:], float), (n_grid, 3))
    got = {n: [] for n in ns}
    total = 0
    while total < m:
        t_first = (len(h) + 1) / fs + rng.uniform(0.0, 1.0) / Imu.ODR
        n = int((t[-1] - t_first) * Imu.ODR) - 1
        stream, _, ys = MD.sense(_perfect(), _still_pw(), t, W, F, t_first, n, rng, noise=True, quantise=quantise)
        c = (np.concatenate([stream.gyro, stream.acc], 1).astype(float) if quantise else ys / step)
        for k in ns:
            got[k].append(_windows(c, k, per))
        total += len(got[ns[0]][-1])
    return {k: np.concatenate(v)[:m] for k, v in got.items()}, per


def noise_checks(out):
    """Check 8.  The test resolves the noise's variance as finely as the fit
    measures it (fit.noise_dof: sqrt(2 / dof)) at Z sigmas: m = Z^2 dof
    windows; run at 32 x ODR, the rate part.window_noise models the part's
    filter at.  The noise is zero-mean about a known still, so each mean's
    square over its model variance is chi-square on one dof."""
    from math import erf
    from batline.imu import fit as F
    from batline.imu.part import window_noise, lsb
    from batline.rig.calib import chi2_quantile
    rng = np.random.default_rng(BATS["developed"][1] + 8)
    W = F.window_W()
    m = int(Z * Z * F.noise_dof())
    fs = 32.0 * Imu.ODR
    lg, la = lsb()
    step = np.array([lg] * 3 + [la] * 3)
    nd = F.datasheet_nd()
    means, per = _stills(np.zeros(6), (1, W), m, fs, rng, quantise=False)
    # Z sigmas two-sided, shared among 6 channels x 2 window lengths
    alpha = (1.0 - erf(Z / np.sqrt(2.0))) / 12.0
    lo_q, hi_q = chi2_quantile(m, 1.0 - alpha / 2.0), chi2_quantile(m, alpha / 2.0)

    def stat(mn, n, scale=1.0):
        sig2 = nd ** 2 * window_noise(n)[0] / step ** 2                  # codes^2
        return np.sum((mn * scale) ** 2, 0) / sig2

    def inside(v):
        return np.all((v >= lo_q) & (v <= hi_q))
    s_w, s_1 = stat(means[W], W), stat(means[1], 1)
    out("noise: a still's single samples and W-means scatter as nd^2 part.window_noise, every channel (8)",
        inside(s_w) and inside(s_1), "W %d, %d windows %d samples apart at %.0f Hz: chi2/m single %s, W-mean %s; "
        "bounds %.3f..%.3f" % (W, m, per, fs, " ".join("%.3f" % v for v in s_1 / m),
                               " ".join("%.3f" % v for v in s_w / m), lo_q / m, hi_q / m))
    # after the filter: white at the output instants, nd sqrt(fs/2) a sample
    white = rng.normal(0.0, 1.0, (m * per, 6)) * nd * np.sqrt(fs / 2.0) / step
    s_a = stat(_windows(white, W, per), W)
    out("must fail: the noise added after the filter (8)", np.all((s_a < lo_q) | (s_a > hi_q)),
        "chi2/m %s" % " ".join("%.1f" % v for v in s_a / m))
    # the density itself as each internal sample's sd: the same linear path, sqrt(2/fs) of it
    s_b = stat(means[W], W, np.sqrt(2.0 / fs))
    out("must fail: the noise's sd set to the density itself (8)", np.all((s_b < lo_q) | (s_b > hi_q)),
        "chi2/m %s" % " ".join("%.2e" % v for v in s_b / m))


def rounding_worker(_):
    t_start = time.time()
    out = Out()
    try:
        _rounding(out)
    except Exception as e:
        import traceback
        out("the rounding check ran to the end", False, "%s: %s" % (type(e).__name__, traceback.format_exc()[-800:]))
    return out.rows, time.time() - t_start


def _rounding(out):
    """Check 9: each channel of a still placed nu of a code past a code (the
    gyro's three at 0, 1/4, 1/2, the accelerometer's the same), model.sense
    with noise and codes; the variance of n-sample means of codes against
    part.code_var, n = 1 and the fit's W, within the Monte Carlo's own
    standard error (the spread of the squared deviations over sqrt m) at Z
    sigmas two-sided, shared among its 12 comparisons (6 channels x 2 window
    lengths, as check 8 shares them) -- m = Z^2 noise_dof windows, so it
    resolves the variance as finely as the fit measures the noise.  Run at
    32 x ODR, the rate part.py's correlations are computed at, so the
    rounding is what is tested."""
    from math import erf
    from batline.imu import fit as F
    from batline.imu.part import code_var, window_noise, lsb
    from batline.rig.pose import normal_quantile
    rng = np.random.default_rng(BATS["developed"][1] + 9)
    W = F.window_W()
    m = int(Z * Z * F.noise_dof())
    fs = 32.0 * Imu.ODR
    lg, la = lsb()
    step = np.array([lg] * 3 + [la] * 3)
    nus = np.array([0.0, 0.25, 0.5, 0.0, 0.25, 0.5])
    s = F.code_sd(F.datasheet_nd() ** 2)
    rows, worst, white_out = [], 0.0, []
    ns = (1, W)
    # Z sigmas two-sided, shared among 6 channels x 2 window lengths
    z9 = normal_quantile(1.0 - (1.0 - erf(Z / np.sqrt(2.0))) / (2.0 * len(ns) * len(nus)))
    means, _ = _stills(nus * step, ns, m, fs, rng, quantise=True)
    for n in ns:
        d = means[n] - means[n].mean(0)
        v = np.sum(d * d, 0) / (m - 1)
        se = np.std(d * d, 0, ddof=1) / np.sqrt(m)
        for c in range(6):
            model = code_var(float(s[c]), [nus[c]], n)[0]
            white = s[c] ** 2 * window_noise(n)[0] / window_noise(1)[0] + 1.0 / (12.0 * n)
            worst = max(worst, abs(v[c] - model) / se[c])
            rows.append("%s nu %.2f n %d: MC %.4f +- %.4f, code_var %.4f, white %.4f"
                        % ("gyro" if c < 3 else "acc", nus[c], n, v[c], se[c], model, white))
            if c < 3 and nus[c] in (0.0, 0.5):
                white_out.append(abs(v[c] - white) / se[c])
    out("rounding: part.code_var is model.sense's own path at nu = 0, 1/4, 1/2 of a code (9)", worst <= z9,
        "%d windows each, s %.3f, %.3f codes; worst %.2f sigma of the Monte Carlo against %.2f (Z %.1f shared "
        "among %d); %s" % (m, s[0], s[3], worst, z9, Z, len(ns) * len(nus), "; ".join(rows)))
    out("must fail: the white-quantisation model, at nu = 0 or 1/2 (9)", max(white_out) > z9,
        "the gyro's, %s sigma against %.2f" % (", ".join("%.1f" % v for v in white_out), z9))


def saturation_checks(out, step, FS, lo, hi):
    """Check 11."""
    from batline.imu.part import impulse
    lim_g, lim_a, s_g, s_a = plan_limits()
    Ep, _ = quantile_part(+1.0)
    Em, _ = quantile_part(-1.0)
    signs = np.array([[a, b, c] for a in (1.0, -1.0) for b in (1.0, -1.0) for c in (1.0, -1.0)])
    # in range: every sign pattern a code under the plan's per-axis limits
    # on the gain-(1+s) part; on the gain-(1-s) part, inputs past full scale
    # by half its spread, while its output stays in range
    Xin_p = np.concatenate([signs * (lim_g - step[0]), signs * (lim_a - step[3])], 1)
    ones = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    Xin_m = np.concatenate([ones * FS[0] * (1.0 + s_g / 2.0), ones * FS[3] * (1.0 + s_a / 2.0)], 1)
    dc_err = abs(1.0 - impulse(4.0 * Imu.ODR).sum())

    def exact(fn):
        worst, flagged = 0.0, 0
        for err, X in ((Ep, Xin_p), (Em, Xin_m)):
            codes, rail, _ = fn(err, X)
            y = _out_of(err, X) / step
            worst = max(worst, np.max(np.abs(codes - y) - dc_err * np.abs(y)))
            flagged += int(np.sum(rail))
        return worst, flagged

    def railed(fn):
        """One channel at a time at and past full scale (FS itself, one code
        past, twice, four times, either way), the others at half scale: that
        channel at its rail and the sample flagged, the others exact.  -FS
        reads -32768, two's complement's own code for it, which is also the
        rail's: the station sees only codes, so it is flagged too (10)."""
        bad = []
        for err in (_perfect(), Ep):
            for c in range(6):
                for lvl, sgs in ((1.0, (1.0, -1.0)), (1.0 + 1.0 / 32768.0, (1.0, -1.0)), (2.0, (1.0, -1.0)),
                                 (4.0, (1.0, -1.0))):
                    for sg in sgs:
                        y = 0.5 * FS * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
                        y[c] = sg * lvl * FS[c]
                        A = np.zeros((6, 6))
                        A[:3, :3] = np.eye(3) + err.Eg
                        A[:3, 3:] = err.G
                        A[3:, 3:] = np.eye(3) + err.Ea
                        b = np.concatenate([err.bg, err.ba])
                        x = np.linalg.solve(A, y - b)
                        codes, rail, _ = fn(err, x[None])
                        yc = y / step
                        others = np.delete(np.arange(6), c)
                        ok = (rail[0] and codes[0, c] == (hi if sg > 0 else lo)
                              and np.all(np.abs(codes[0, others] - yc[others]) <= 0.5 + dc_err * np.abs(yc[others])))
                        if not ok:
                            bad.append((c, lvl, sg))
        return bad

    def sense_fn(err, X):
        return _held(err, X)
    w_e, n_f = exact(sense_fn)
    bad = railed(sense_fn)
    out("saturation: codes exact to half an LSB up to the plan's per-axis limit FS u/(1+s), and past full scale "
        "where the part's gain brings the output back in range (11)", w_e <= 0.5 and n_f == 0,
        "s %.4f gyro, %.4f accel (z %.2f); worst |code - y/LSB| %.3f, %d flagged"
        % (s_g, s_a, quantile_part()[1], w_e, n_f))
    out("saturation: an output at and past full scale rails and is flagged on its own axis, the others exact (11)",
        not bad, "%d cases fail: %s" % (len(bad), bad[:6]) if bad else "6 channels x 8 levels x 2 parts")

    def norm_fn(err, X):
        """Clipping on each sensor's vector norm."""
        y = _out_of(err, X)
        rail = np.zeros(len(y), bool)
        for sl, F_ in ((slice(0, 3), FS[0]), (slice(3, 6), FS[3])):
            nrm = np.linalg.norm(y[:, sl], axis=1)
            over = nrm > F_
            y[over, sl] *= (F_ / nrm[over])[:, None]
            rail |= over
        return np.clip(np.round(y / step), lo, hi), rail, 1.0

    def before_fn(err, X):
        """Clipping the true input at full scale, then the part's gain."""
        X = np.asarray(X, float)
        Xc = np.clip(X, -FS, FS)
        y = _out_of(err, Xc)
        return np.round(y / step), np.any(Xc != X, axis=1), 1.0
    for what, fn in (("clipping the vector's norm", norm_fn), ("clipping before the gain", before_fn)):
        w_b, n_b = exact(fn)
        bad_b = railed(fn)
        out("must fail: %s (11)" % what, w_b > 0.5 or n_b > 0 or bool(bad_b),
            "worst |code - y/LSB| %.1f, %d in-range flagged, %d rail cases wrong" % (w_b, n_b, len(bad_b)))


def swing_checks(out):
    """Check 12, the study's half: the rated swing at the IMU, through the
    part plan.limits budgets for."""
    from batline import imu_fusion_sim as study
    from batline.imu import model as MD
    from batline.imu.part import impulse
    sw = next(s for s in study.swings() if s.name == Rules.RATED_SWING)
    tr = study.Truth(sw)
    fs = 16.0 * Imu.ODR
    t = np.arange(int(np.floor(tr.t[-1] * fs)) + 1) / fs
    w = np.stack([np.interp(t, tr.t[:-1] + study.DT / 2.0, tr.w[:, i]) for i in range(3)], 1)
    f = np.stack([np.interp(t, tr.t[:-1] + study.DT / 2.0, tr.f[:, i]) for i in range(3)], 1)
    rng = np.random.default_rng(BATS["developed"][1] + 120)
    t_first = (len(impulse(fs)) + 1) / fs
    n = int((t[-1] - t_first) * Imu.ODR) - 1

    def run(part):
        stream, _, _ = MD.sense(part, MD.power_up(rng, 0.0, Site.AMBIENT), t, w, f, t_first, n, rng)
        return int(np.sum(stream.rail)), int(max(np.max(np.abs(stream.gyro)), np.max(np.abs(stream.acc))))
    parts = [quantile_part(+1.0)[0], quantile_part(-1.0)[0]]
    res = [run(p) for p in parts]
    out("the study's rated swing (%s) fits the +-%.0f dps / %.0f g part at the plan's quantile (12)"
        % (sw.name, Imu.GYRO_FS, Imu.ACC_FS), all(r[0] == 0 for r in res),
        "peaks %.0f dps, %.1f g (the study's); rails %s, the largest code %s"
        % (tr.gyro_peak_dps, tr.acc_peak_g, [r[0] for r in res], [r[1] for r in res]))
    keep = Imu.GYRO_FS
    try:
        Imu.GYRO_FS = 2000.0
        rails = run(parts[0])[0]
    finally:
        Imu.GYRO_FS = keep
    out("must fail: the same swing on a 2000 dps part (12)", rails > 0, "%d of %d samples railed" % (rails, n))


def warmup_checks(out):
    """Check 14: a part with its tempcos at Z sigmas reading nothing, sampled
    a second at a time at the instants the die has risen k/8 of its way
    (k = 1..7) and just after power-up.  The expectation is built from
    spec.Imu's constants directly -- T = T_amb + SELF_HEAT (1 - e^(-t /
    HEAT_TAU)), b = b0 + c (T - 25 C) + b_on -- and put through the same
    (separately checked, 7) filter; the two then differ by rounding alone:
    an FFT's log2 L roundings of the largest value."""
    from batline.imu import model as MD
    from batline.imu.part import impulse, T_REF, G0, D2R
    fs = 4.0 * Imu.ODR
    h = impulse(fs)
    sg = np.array([1.0, -1.0, 1.0])
    err = MD.ImuErrors(np.zeros((3, 3)), np.zeros((3, 3)), Imu.ACC_BIAS * G0 * sg, Imu.GYRO_BIAS * D2R * sg,
                       np.zeros((3, 3)), Z * Imu.ACC_TC * G0 * sg, Z * Imu.GYRO_TC * D2R * sg, 0.0)
    T_amb = Site.AMBIENT - Z * Site.AMBIENT_SD          # a cold morning: the die's rise from well under 25 C
    pw = MD.PowerUp(Imu.ACC_POWERUP * G0 * sg, Imu.GYRO_POWERUP * D2R * sg, 0.0, T_amb)
    times = [Imu.START_S] + [-Imu.HEAT_TAU * np.log(1.0 - k / 8.0) for k in range(1, 8)]

    def expect(t, rise=True, tc=1.0):
        Td = T_amb + Imu.SELF_HEAT * (1.0 - np.exp(-t / Imu.HEAT_TAU)) * (1.0 if rise else 0.0)
        bg = err.bg + tc * err.cg * (Td[:, None] - T_REF) + pw.bg_on
        ba = err.ba + tc * err.ca * (Td[:, None] - T_REF) + pw.ba_on
        return np.concatenate([bg, ba], 1), Td
    worst = worst_r = worst_s = 0.0
    temp_ok = temp_bad = True
    for t0 in times:
        t = t0 + np.arange(int(fs * 1.0) + 1) / fs
        zero = np.zeros((len(t), 3))
        t_first = t[0] + (len(h) + 1) / fs
        n = int((t[-1] - t_first) * Imu.ODR) - 1
        stream, tk, ys = MD.sense(err, pw, t, zero, zero, t_first, n, None, noise=False)
        E, _ = expect(t)
        u = (tk - t[0]) * fs
        i = np.minimum(np.floor(u).astype(int), len(t) - 2)
        fr = (u - i)[:, None]

        def through(x):
            y = MD.fir(x, h)
            return y[i] * (1.0 - fr) + y[i + 1] * fr
        L = 1 << int(np.ceil(np.log2(len(t) + len(h))))
        tol = fft_round(L, np.abs(h).sum()) * np.max(np.abs(E), 0)
        worst = max(worst, np.max(np.abs(ys - through(E)) / tol))
        worst_r = max(worst_r, np.max(np.abs(ys - through(expect(t, rise=False)[0])) / tol))
        worst_s = max(worst_s, np.max(np.abs(ys - through(expect(t, tc=-1.0)[0])) / tol))
        Tk = T_amb + Imu.SELF_HEAT * (1.0 - np.exp(-tk / Imu.HEAT_TAU))
        temp_ok &= bool(np.array_equal(stream.temp, np.round((Tk - T_REF) / Imu.TEMP_LSB).astype(int)))
        temp_bad &= bool(np.array_equal(stream.temp, np.round((T_amb - T_REF) / Imu.TEMP_LSB) * np.ones(n, int)))
    out("warm-up: the bias follows the tempco times the die's rise, and the die's codes the rise (14)",
        worst <= 1.0 and temp_ok, "%d seconds sampled from %.1f to %.0f s after power-up: worst %.2f of the "
        "rounding tolerance; die codes %s" % (len(times), times[0], times[-1], worst,
                                               "exact" if temp_ok else "off"))
    out("must fail: the die's rise left out (14)", worst_r > 1.0 and not temp_bad,
        "worst %.2e of the tolerance" % worst_r)
    out("must fail: the tempco's sign flipped (14)", worst_s > 1.0, "worst %.2e of the tolerance" % worst_s)


# ================================================================ TRUTH SEPARATION
PRODUCTION = ("imu/fit.py", "imu/plan.py", "imu/kin.py", "imu/part.py", "imu/record.py", "imu/golden.py",
              "line.py")
# the ways a production module could reach the truth, planted as if in batline/imu
SLIPS = ("from . import sim", "import mujoco", "from .model import sense", "from batline.imu import judge",
         "from .scene import Scene, inner_motion", "import batline.imu.model as MD", "from . import kin, sim as S",
         "from ..imu import scene", "import importlib\nm = importlib.import_module('batline.imu.sim')",
         "S = __import__('batline.imu.sim')", "from mujoco import mj_inverse",
         "def f():\n    from .judge import errors\n    return errors",
         "try:\n    import mujoco as mj\nexcept ImportError:\n    mj = None", "import os, mujoco",
         "from batline.imu.sim import run_station")


def imports_truth(src, package="batline.imu"):
    """What in Python source `src` (a module of `package`) imports MuJoCo or
    a TRUTH module: every import statement wherever it stands, relative ones
    resolved, and any import by name at run time (importlib, __import__),
    which no scan can follow.  Code only: strings and comments are not
    imports."""
    import ast
    import io
    import tokenize
    hits = []

    def truth(full):
        p = full.split(".")
        return p[0] == "mujoco" or (len(p) >= 3 and p[0] == "batline" and p[1] == "imu" and p[2] in TRUTH)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            hits += [a.name for a in node.names if truth(a.name)]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")[: len(package.split(".")) - (node.level - 1)]
                full = ".".join(base + ([node.module] if node.module else []))
            else:
                full = node.module or ""
            if truth(full):
                hits.append(full)
            hits += [full + "." + a.name for a in node.names if truth(full + "." + a.name)]
    toks = [t.string for t in tokenize.generate_tokens(io.StringIO(src).readline)
            if t.type == tokenize.NAME]
    hits += [t for t in toks if t in ("importlib", "__import__", "import_module")]
    return hits


def separation_checks(out):
    import subprocess
    root = os.path.join(HERE, "..", "..", "batline")
    hits, scanned = [], []
    for rel in PRODUCTION:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            continue
        pkg = "batline.imu" if rel.startswith("imu/") else "batline"
        scanned.append(rel)
        hits += ["%s: %s" % (rel, h) for h in imports_truth(open(path).read(), pkg)]
    out("no production module imports MuJoCo or a TRUTH module, by its source (15)", not hits,
        "%s: %s" % (", ".join(scanned), "; ".join(hits) or "none"))
    missed = [s for s in SLIPS if not imports_truth(s)]
    out("must fail: the scan passes a production module that reaches the truth (%d ways) (15)" % len(SLIPS),
        not missed, "missed: %s" % missed if missed else "every one caught")
    # and as an interpreter loads them, each in a fresh one
    sim_dir = os.path.abspath(os.path.join(HERE, "..", ".."))
    code = ("import sys; sys.path.insert(0, %r); import importlib; importlib.import_module(sys.argv[1]); "
            "print(' '.join(sorted(k for k in sys.modules if k == 'mujoco' or k in %r)))"
            % (sim_dir, tuple("batline.imu." + t for t in TRUTH)))
    loaded = []
    for rel in scanned:
        mod = "batline." + rel[:-3].replace("/", ".")
        p = subprocess.run([sys.executable, "-c", code, mod], capture_output=True, text=True)
        got = p.stdout.strip() if p.returncode == 0 else "failed to import: %s" % p.stderr.strip()[-200:]
        if got:
            loaded.append("%s: %s" % (mod, got))
    out("no production module loads MuJoCo or a TRUTH module, in a fresh interpreter (15)", not loaded,
        "; ".join(loaded) or "%d modules, none" % len(scanned))
    p = subprocess.run([sys.executable, "-c", code, "batline.imu.sim"], capture_output=True, text=True)
    out("must fail: the same load test on sim.py (15)", bool(p.stdout.strip()), "sees %s" % p.stdout.strip())


# ================================================================ MAIN
JOBS = (("plan", plan_worker, None), ("rounding", rounding_worker, None), ("sensor", sensor_worker, None),
        ("physics developed", physics, ("kid",) + BATS["developed"]),
        ("physics unseen", physics, ("adult",) + BATS["unseen"]))


def job(i):
    name, fn, arg = JOBS[i]
    rows, el = fn(arg)
    return name, rows, el


def main():
    t0 = time.time()
    import mujoco
    # the slowest first, so the pool's last worker is not left with it
    with get_context("spawn").Pool(min(len(JOBS), os.cpu_count() or 1)) as pool:
        done = pool.map(job, range(len(JOBS)), chunksize=1)
    for name, rows, el in done:
        for nm, ok, det, secs in rows:
            check(nm, ok, det, secs)
        if VERBOSE:
            print("  (%s: %.0f s)" % (name, el))
    def number(row):
        m = re.search(r"\((\d+)\)$", row[0])
        return int(m.group(1)) if m else 99
    RESULTS.sort(key=number)
    bad = sum(1 for _, ok, _, _ in RESULTS if not ok)
    for nm, ok, det, secs in RESULTS:
        print("  %s %5.1fs  %s%s" % ("ok  " if ok else "FAIL", secs, nm,
                                     ("  [%s]" % det) if det and (VERBOSE or not ok) else ""))
    print("check_imu: %d checks, %d failed, %.0f s (MuJoCo %s)"
          % (len(RESULTS), bad, time.time() - t0, mujoco.__version__))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
