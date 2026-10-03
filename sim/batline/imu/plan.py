"""The calibration plan for a bat design: where the gimbal holds still,
which spins it makes and how fast -- designed, from the targets, the part's
datasheet, the gimbal's motors and masses and the cameras' budget.  numpy
only: line.py times station 4 by it.

    limits(d)       how fast and how hard each hinge may turn, and why
    nominal(d)      the bat's IMU as drawn: the fit's parameters there
    spreads(d)      how far a part and its seat lie from the drawing, 1 sigma
    design(d)       the plan (CalPlan), designed to the bats' thresholds
                    (or to s, golden.day_design's)
    for_design(d)   the plan golden.day_design picks, once per bat: what
                    line.py calls

THE DESIGN is optimal experimental design over a candidate set (Fedorov
1972; the greedy with a cost is Krause & Golovin's, 2014):
  candidates  stills: gravity's direction in the clamp's frame, from a
              spherical Fibonacci set (Gonzalez 2010), on either of the two
              gimbal poses that give it; spins: either hinge, the other
              parked at each of a set of angles, at each of a set of rates
  value       V = sum_j max(0, sigma_j^2 / s_j^2 - 1): how far the fit's
              predicted 1 sigma (the same model, fit.Model, at the drawing)
              is from each term's threshold (fit.thresholds)
  greedy      add the candidate that lowers V most per second it costs --
              its move, its settling, its dwell -- at its cheapest place in
              the tour; a still's dwell is chosen by golden section
  check       build the tour as the motors would run it, place its quiet
              intervals, and recompute V from those exactly; carry on
              while V > 0
  prune       drop any item whose removal keeps V = 0
  order       2-opt on the tour (Croes 1958)
  refine      the best spin's rate, then its park, by golden section
              between the grid's neighbours
  density     the candidate set is doubled until doubling it no longer
              buys a plan one settle shorter
Weak priors at the parts' spreads rank the first candidates, before the
data can; the check that ends the design has none on the targets.

MOVES.  Rest to rest, both hinges arriving together (truss.motion's
trapezoids), each at an acceleration chosen against what it costs in
settling: an acceleration step of da leaves the hinge ringing at
da / w_n^2 on its stepper's stiffness (kin.settle), and the next quiet
interval starts when the ringing is below what a window can see -- where
summed over the interval's windows it adds less than one to the fit's
chi-square.  w_n is the hinge's own where the step happens: the outer
hinge's inertia turns with the inner's angle, so a move's last corner
rings, and decays, at the still's frequency, not the one it left.  The tour
is closed: it ends back at the zeros, where the gantry unloads the parked
gimbal and the next run counts its cable's wrap from.

THE MOTORS.  A hybrid stepper's torque is a sine of its lag, T sin(N e),
T its pull-out torque at the rate and N the electrical radians an axis
radian; past N e = pi/2 it slips a step.  So a hinge's accelerations are
budgeted against T with the ring each step of the command leaves: the
step response peaks at 1 + exp(-zeta pi / sqrt(1 - zeta^2)) of its step,
zeta the hinge's smallest damping ratio over its poses, and gravity's and
the rates' torques ride on it, raised by how near the motion's harmonics
come to the hinge's natural frequency (CalPlan.lag bounds the lag the
same way).
"""
from dataclasses import dataclass, field
from functools import lru_cache
from math import sqrt, pi, log, exp, radians, degrees, ceil
import hashlib

import numpy as np

from ..spec import (Imu, Board, Gimbal, Stepper, Site, Quality, Clamp, Print, Pogo, Rig, Line, RigBuild)
from ..rig import spec as RS
from ..rig.inertia import inertials
from ..rig.pose import normal_quantile
from . import kin as K
from . import fit as F
from .part import G0, D2R, lsb, unrailed, window_noise, impulse

GOLDEN = (sqrt(5.0) - 1.0) / 2.0
SPIN_RANK_PER_TURN = 128           # windows a turn keeps when a spin is ranked (nominal_obs): a
                                   # numerical method's resolution, not the plan's -- check_imu
                                   # holds the thinned sigmas within 1% of every window's (0.2%
                                   # on the kid's tour; 64 a turn gave 0.7%, 32 gave 1.6%)
DESIGN_PLACE = "most"              # where between two codes the plan takes each mean's true
                                   # value to fall (fit.obs_var): a bat's is a draw the plan
                                   # cannot know, so it takes the worst, and every bat's sigma
                                   # then meets its threshold whatever the draw


# ================================================================ THE BAT
def board_nominal(d, rig):
    """(r m, R): the IMU's sensing centre in the clamp's frame and the
    chip's axes there (columns), as drawn: the board on its stop, its top
    face on the ledges, the chip at Board.IMU_AT on it."""
    lay, core = d.lay, d.core
    p = np.array([lay.x_imu, core.y_datum + Board.W / 2.0 + Board.IMU_AT[1], core.z_ledge + Board.IMU_Z])
    R = np.array(Board.IMU_AXES, float).T
    if abs(np.linalg.det(R) - 1.0) > 1e-9 or np.max(np.abs(R.T @ R - np.eye(3))) > 1e-9:
        raise ValueError("spec.Board.IMU_AXES is not a rotation")
    return rig.bat_to_body(p) / 1000.0, R


def seat_pivot(rig):
    """m, clamp frame: what a bat turns about as it settles in the jaws."""
    jaws = [b for b in rig.clamp if b.name == "clamp_jaws"][0]
    return np.array(jaws.c, float) / 1000.0


def nominal(d, rig):
    r, R = board_nominal(d, rig)
    th = np.zeros(F.NP)
    th[F.MA] = R.T.ravel()
    th[F.MG] = R.T.ravel()
    th[F.RR] = r
    th[F.T0] = Imu.START_S
    th[F.RHO] = 1.0
    return th


def spreads(d, rig):
    """(NT,) 1 sigma of each judged term about the drawing: the part's
    datasheet spreads and how the board and the bat sit, as the fit's terms
    read them (the polar split puts half of an unsymmetric cross-axis error
    into the mounting)."""
    r, R = board_nominal(d, rig)
    dT = Site.AMBIENT_SD + Imu.SELF_HEAT
    out = np.empty(F.NT)
    out[0:3] = sqrt(Imu.ACC_BIAS ** 2 + Imu.ACC_POWERUP ** 2 + (Imu.ACC_TC * dT) ** 2) * 1000.0
    out[3:6] = Imu.ACC_SCALE
    out[6:9] = Imu.ACC_CROSS / sqrt(2.0)
    out[9:12] = sqrt(Imu.GYRO_BIAS ** 2 + Imu.GYRO_POWERUP ** 2 + (Imu.GYRO_TC * dT) ** 2)
    out[12:15] = Imu.GYRO_SCALE
    out[15:21] = sqrt(Imu.GYRO_CROSS ** 2 + Imu.ACC_CROSS ** 2 / 2.0)
    # the mounting, chip axes: tilt about x and y, turn about z on the
    # board; the board on its ledges across its width and along its length;
    # the seat; the cross-axis's turn
    tilt = np.radians([Board.CHIP_TILT, Board.CHIP_TILT, Board.CHIP_ROT])
    slot = np.array([Print.TOL / Board.W, Print.TOL / Board.L, 0.0])
    var = R @ np.diag(tilt ** 2) @ R.T + np.diag(slot ** 2) + np.eye(3) * (radians(Clamp.SEAT_ROT) ** 2
                                                                         + Imu.ACC_CROSS ** 2 / 2.0)
    out[21:24] = np.degrees(np.sqrt(np.diag(var)))
    # the lever, clamp axes: the chip placed on the board's face (along and
    # across it: clamp x and y), the bat seated across its own axis (y and
    # z: it lies on its stop along x), the board along its slot
    lever = np.sqrt(Print.TOL ** 2 + np.array([1.0, 1.0, 0.0]) * Board.CHIP_POS ** 2
                    + np.array([0.0, 1.0, 1.0]) * Clamp.SEAT_POS ** 2
                    + (np.linalg.norm(r - seat_pivot(rig)) * 1000.0 * radians(Clamp.SEAT_ROT)) ** 2
                    + np.array([Board.L_TOL ** 2 / 3.0, 0.0, 0.0]))
    out[24:27] = lever
    out[27] = Imu.ODR_TOL
    return out


# ================================================================ LIMITS
def _pullout(w_axis):
    """N m at the axis, at axis rate w (rad/s): the motor's pull-out torque
    through the belt; zero past the curve's end."""
    rpm = np.abs(w_axis) * Gimbal.GEAR * 60.0 / (2.0 * pi)
    c = np.array(Stepper.PULLOUT)
    return np.where(rpm <= c[-1, 0], np.interp(rpm, c[:, 0], c[:, 1]), 0.0) * Gimbal.GEAR


def _harmonics(b):
    """(n, 5): 1 and the first two harmonics of angles b."""
    return np.stack([np.ones_like(b), np.cos(b), np.sin(b), np.cos(2.0 * b), np.sin(2.0 * b)], 1)


@dataclass
class Limits:
    """How the gimbal may move, each limit with what binds it.  A move
    turns both hinges at once; a spin turns one, the other held."""
    w_max: np.ndarray              # (2,) rad/s, a move's
    a_max: np.ndarray              # (2,) rad/s^2, a move's
    w_spin: np.ndarray             # (2,) rad/s, a spin's top rate
    why_w: list
    why_a: list
    why_spin: list
    zeta: float
    chain: object                  # kin.Chain as drawn
    outer: object                  # kin.Inertial
    inner: object
    arm: float                     # kg m^2, each rotor through its belt
    r_max: float                   # m, the IMU's farthest from the gimbal's centre
    lag_static: float              # rad, the most gravity can hold a hinge off its command
    tg: np.ndarray                 # (2,) N m, gravity's largest torque on each hinge
    Mmax: np.ndarray               # (2,) kg m^2, each hinge's largest inertia
    _wn: dict = field(default_factory=dict, repr=False)            # caches, this gimbal's own
    _moves: dict = field(default_factory=dict, repr=False)
    _shapes: dict = field(default_factory=dict, repr=False)

    def stiffness(self):
        """N m/rad at an axis: a hybrid stepper's holding stiffness, its
        torque's slope at a full step's zero (T_hold sin(N_r theta)),
        through the belt."""
        return Stepper.HOLD_NM * Stepper.TEETH * Gimbal.GEAR ** 2

    def damping(self):
        """(2,) N m s/rad at each axis: the driver's damping, a constant of
        the motor and its driver.  Stepper.ZETA is a ratio as a ring-down
        at the gimbal's zero measures it; elsewhere the same damping on
        another inertia rings down at another ratio."""
        if "c" not in self._wn:
            M0 = K.hinge_inertia(self.chain, self.outer, self.inner, np.zeros((1, 2)), (self.arm, self.arm))[0]
            self._wn["c"] = 2.0 * self.zeta * np.sqrt(self.stiffness() * M0)
        return self._wn["c"]

    def decay(self, h, wn):
        """1/s: hinge h's ringing decays as e^(-s t) where its natural
        frequency is wn: s = c / 2M = c wn^2 / 2k."""
        return float(self.damping()[h] * wn ** 2 / (2.0 * self.stiffness()))

    def w_n(self, q):
        """(2,) rad/s: each hinge's natural frequency at angles q."""
        key = tuple(np.round(np.ravel(q), 12))
        if key not in self._wn:
            self._wn[key] = self.w_ns(np.atleast_2d(q))[0]
        return self._wn[key]

    def w_ns(self, Q):
        """(n, 2) rad/s: each hinge's natural frequency at each of angles Q
        (n, 2).  A serial chain's inertia does not turn with its first
        hinge, and its second turns the inner body's inertia and centre of
        mass through first and second harmonics of its angle: five samples
        of kin.hinge_inertia fix it exactly."""
        if "harm" not in self._wn:
            b = np.linspace(0.0, 2.0 * pi, 5, endpoint=False)
            Mb = K.hinge_inertia(self.chain, self.outer, self.inner, np.stack([np.zeros(5), b], 1),
                                 (self.arm, self.arm))
            self._wn["harm"] = np.linalg.solve(_harmonics(b), Mb)
        return np.sqrt(self.stiffness() / (_harmonics(np.asarray(Q, float)[:, 1]) @ self._wn["harm"]))

    def w_n_min(self):
        """(2,) rad/s: each hinge's lowest natural frequency, at its largest
        inertia."""
        return np.sqrt(self.stiffness() / self.Mmax)

    def overshoot(self):
        """(2,): how far past a step of command each hinge's lag rings, of
        the step, exp(-zeta pi / sqrt(1 - zeta^2)), at its smallest damping
        ratio over its poses: the driver's damping is fixed, so the ratio is
        least where the inertia is most."""
        z = self.damping() / (2.0 * np.sqrt(self.stiffness() * self.Mmax))
        return np.exp(-z * pi / np.sqrt(1.0 - z * z))

    def resonance(self, w):
        """(2,): how much each hinge's lag under a load turning at rate w is
        raised by the load's harmonics (fit.NH of them: gravity's at w, the
        bodies' products of inertia at 2w) against its lowest natural
        frequency, 1 / (1 - (NH w / w_n)^2); inf at the resonance or past
        it."""
        x = (F.NH * abs(float(w)) / self.w_n_min()) ** 2
        return np.where(x < 1.0, 1.0 / np.maximum(1.0 - x, 1e-300), np.inf)

    def a_spin(self, h, w):
        """rad/s^2: the most hinge h may accelerate alone up to rate w: its
        motor's pull-out torque there, less gravity's as the spin's
        harmonics raise it, over its inertia and the ring each step of the
        ramp leaves."""
        room = _pullout(w) - self.tg[h] * self.resonance(w)[h]
        return float(room / (self.Mmax[h] * (1.0 + self.overshoot()[h])))


def _gimbal_masses(rig):
    from ..rig import mjcf as RM
    b = RM.RigBuildDraw.drawing(rig, [])
    I = inertials("\n".join(RM.gimbal_xml(rig, b, payload="bat", weighed=True)))
    return I["outer"], I["inner"]


def limits(d, rig=None):
    rig = RS.design(d) if rig is None else rig
    outer, inner = _gimbal_masses(rig)
    chain = K.Chain.drawn()
    arm = Stepper.ROTOR_J * Gimbal.GEAR ** 2
    r, _ = board_nominal(d, rig)
    r_max = float(np.linalg.norm(r))
    g = K.gravity(Site.LAT, Site.ALT)
    z = normal_quantile(1.0 - Quality.CAL_QUAL_ALPHA / 12.0)       # no channel of a healthy part rails
    cands = []
    w_motor = Stepper.PULLOUT[-1][0] * 2.0 * pi / 60.0 / Gimbal.GEAR
    cands.append((w_motor, "the motor's pull-out curve ends at %.0f rpm" % Stepper.PULLOUT[-1][0]))
    # the rate that keeps every gyro channel off its rail: less its offset
    # and what the g-sensitivity makes of the largest force the accelerometer
    # can read, over its scale and cross-axis gain
    gyro_fs = (Imu.GYRO_FS * unrailed() - z * Imu.GYRO_BIAS - z * Imu.GYRO_GSENS * Imu.ACC_FS) * D2R \
        / (1.0 + z * sqrt(Imu.GYRO_SCALE ** 2 + 2.0 * Imu.GYRO_CROSS ** 2))
    cands.append((gyro_fs, "the gyro's full scale"))
    acc_fs = Imu.ACC_FS * unrailed() * G0 / (1.0 + z * sqrt(Imu.ACC_SCALE ** 2 + 2.0 * Imu.ACC_CROSS ** 2)) \
        - z * Imu.ACC_BIAS * G0 - g
    cands.append((sqrt(acc_fs / r_max), "the accelerometer's full scale at the IMU"))
    hold = 2.0 * Board.PAD_MU * Pogo.N * Pogo.FORCE / (Board.MASS / 1000.0)
    cands.append((sqrt(max(hold - g, 0.0) / r_max), "the pogo block's grip on the board"))
    w_cap, why_cap = min(cands)
    # the torques over the gimbal's every pose: gravity's, each hinge's
    # inertia and its pull on the other, the rates' own
    a_, b_ = np.meshgrid(np.linspace(-pi, pi, 25), np.linspace(-pi, pi, 25))
    Q = np.stack([a_.ravel(), b_.ravel()], 1)
    zq = np.zeros_like(Q)
    gw = np.array([0.0, 0.0, -g])
    tg = np.max(np.abs(K.rnea(chain, outer, inner, Q, zq, zq, gw)), 0)
    M = np.zeros((2, 2))
    for h in (0, 1):
        e = np.zeros_like(Q)
        e[:, h] = 1.0
        M[:, h] = np.max(np.abs(K.rnea(chain, outer, inner, Q, zq, e, np.zeros(3), (arm * (h == 0), arm * (h == 1)))), 0)
    # the limits filled in below; what they are budgeted by (the hinges'
    # natural frequencies, their ring) is the gimbal's own
    lim = Limits(None, None, None, [], [], [], Stepper.ZETA, chain, outer, inner, arm, r_max, 0.0, tg,
                 np.diag(M).copy())
    ring = 1.0 + lim.overshoot()

    def rate_torque(w):
        tc = np.zeros(2)
        for s0 in (-1.0, 1.0):
            for s1 in (-1.0, 1.0):
                qd = np.tile([s0 * w, s1 * w], (len(Q), 1))
                tc = np.maximum(tc, np.max(np.abs(K.rnea(chain, outer, inner, Q, qd, zq, np.zeros(3))), 0))
        return tc

    def move_a(w):
        """Both hinges at rate w and accelerating together, anywhere: the
        accelerations whose torques the motors have at w, on top of
        gravity's and the rates' as the motion's harmonics raise them, with
        the ring: a move with no cruise steps its acceleration by 2a at its
        middle, and the lag rings past that step by its overshoot."""
        amp = lim.resonance(w)
        if not np.all(np.isfinite(amp)):
            return None
        room = _pullout(np.array([w, w])) - amp * (tg + rate_torque(w))
        if np.any(room <= 0.0):
            return None
        a = np.linalg.solve(M, room / (2.0 * ring))
        return a if np.all(a > 0.0) else None

    # a move's rate: the one that turns both hinges half a turn soonest
    def half_turn(w):
        a = move_a(w)
        return np.inf if a is None else max(pi / w + w / a[h] for h in (0, 1))
    lo, hi = 1e-3 * w_cap, w_cap
    if move_a(lo) is None:
        raise ValueError("the gimbal's motors cannot move it against gravity at any rate")
    if move_a(hi) is None:                    # the fastest the torque allows, by bisection
        a_, b_ = lo, hi
        for _ in range(50):
            m = 0.5 * (a_ + b_)
            a_, b_ = (m, b_) if move_a(m) is not None else (a_, m)
        hi = a_
    x1, x2 = hi - GOLDEN * (hi - lo), lo + GOLDEN * (hi - lo)
    c1, c2 = half_turn(x1), half_turn(x2)
    for _ in range(40):
        if c1 < c2:
            hi, x2, c2 = x2, x1, c1
            x1 = hi - GOLDEN * (hi - lo)
            c1 = half_turn(x1)
        else:
            lo, x1, c1 = x1, x2, c2
            x2 = lo + GOLDEN * (hi - lo)
            c2 = half_turn(x2)
    w = x1 if c1 < c2 else x2
    if half_turn(w_cap) <= half_turn(w):
        w, why_w = w_cap, why_cap
    else:
        why_w = "the rate that turns both hinges half a turn soonest within the motors' torque"
    a = move_a(w)
    if a is None:
        raise ValueError("the gimbal's motors cannot move it against gravity")
    why_a = ("the pull-out torque at %.0f deg/s, less gravity's and the rates' (raised by their harmonics), "
             "over the inertias and the ring of a 2a step" % degrees(w))
    # a spin's top rate: the cap, unless the motor has no torque left
    # there to accelerate against gravity, or the held hinge cannot hold
    # what the spin throws on it -- each as the spin's harmonics raise it,
    # which also keeps them off either hinge's natural frequency
    w_spin = np.full(2, w_cap)
    why_spin = [why_cap, why_cap]
    hold_t = _pullout(0.0)
    for h in (0, 1):
        while True:
            qd = np.zeros_like(Q)
            qd[:, h] = w_spin[h]
            held = np.max(np.abs(K.rnea(chain, outer, inner, Q, qd, zq, gw)[:, 1 - h]))
            amp = lim.resonance(w_spin[h])
            short = [why for why, ok in
                     (("a harmonic of the spin at a hinge's natural frequency", np.all(np.isfinite(amp))),
                      ("the motor's torque at speed", _pullout(w_spin[h]) - tg[h] * amp[h] > 0.0),
                      ("the held hinge's torque", held * amp[1 - h] < hold_t)) if not ok]
            if not short:
                break
            w_spin[h] *= 0.98
            why_spin[h] = ", ".join(short)
    lim.w_max, lim.a_max, lim.w_spin = np.array([w, w]), a, w_spin
    lim.why_w, lim.why_a, lim.why_spin = [why_w] * 2, [why_a] * 2, why_spin
    # the stepper's sine law: gravity alone holds a hinge off its command
    # by the angle whose sine is its torque over the holding torque
    lim.lag_static = float(np.arcsin(min(np.max(tg) / hold_t, 1.0)) / (Stepper.TEETH * Gimbal.GEAR))
    return lim


# ================================================================ RINGING
def noise_floor(W):
    """(gyro, accel) SI: one W-window mean's noise, by the datasheet, where
    the rounding leaves it least (fit.obs_var): what ringing is measured
    against."""
    nd2 = F.datasheet_nd() ** 2
    v = F.obs_var(nd2, W, "least")
    return sqrt(v[0]), sqrt(v[3])


def ring_tol(W, s):
    """(gyro, accel): how much ringing a quiet interval may start with, its
    decay rate s: summed over the interval's windows, A^2 e^(-2 s t) / 2
    in units of a window's variance, it adds less than one to the
    chi-square."""
    fg, fa = noise_floor(W)
    k = sqrt(2.0 / (1.0 + Imu.ODR / (2.0 * s * W)))
    return fg * k, fa * k


def settle_after(corners, lim, W, w_cruise=0.0):
    """s after the last corner until both channels are quiet: corners
    [(t, da (2,), w_n (2,))] since the last quiet interval.  Each hinge's
    corners ring on its own frequency and decay; the hinges share a
    channel's tolerance equally."""
    if not corners:
        return 0.0
    t_last = max(c[0] for c in corners)
    out = 0.0
    g = K.gravity(Site.LAT, Site.ALT)
    for h in (0, 1):
        cs = [(t, da[h], wn[h]) for t, da, wn in corners if da[h] != 0.0]
        if not cs:
            continue
        wn = cs[-1][2]
        s = lim.decay(h, wn)
        tg, ta = ring_tol(W, s)
        A = sum(abs(da) / w ** 2 / sqrt(1.0 - (lim.decay(h, w) / w) ** 2) * exp(-lim.decay(h, w) * (t_last - t))
                for t, da, w in cs)
        for amp, tol in ((wn * A, tg), ((wn ** 2 * lim.r_max + g + 2.0 * abs(w_cruise) * wn * lim.r_max) * A, ta)):
            if amp > tol / 2.0:
                out = max(out, log(amp / (tol / 2.0)) / s)
    return out


def _move_corners(d, lim, f):
    """[(t, da (2,))] of a coordinated rest-to-rest move by d (2,) at a
    fraction f of each hinge's top acceleration, and its duration."""
    vm, am = lim.w_max, lim.a_max * f
    Tk = [float(K.trap_time(d[h], vm[h], am[h])) for h in (0, 1)]
    Tm = max(Tk)
    out = []
    if Tm <= 0.0:
        return out, 0.0
    for h in (0, 1):
        if abs(d[h]) < 1e-12 or Tk[h] == 0.0:
            continue
        s = Tk[h] / Tm
        a = am[h] * s * s * np.sign(d[h])
        if abs(d[h]) <= vm[h] ** 2 / am[h]:
            out += [(0.0, _on(h, a)), (Tm / 2.0, _on(h, -2.0 * a)), (Tm, _on(h, a))]
        else:
            ta = vm[h] / am[h] / s
            out += [(0.0, _on(h, a)), (ta, _on(h, -a)), (Tm - ta, _on(h, -a)), (Tm, _on(h, a))]
    return out, Tm


def _on(h, v):
    e = np.zeros(2)
    e[h] = v
    return e


def move_ring(q0, q1, lim, f):
    """([(t, da (2,), w_n (2,))], duration): a move's corners (_move_corners)
    from q0 to q1, each with the hinges' natural frequencies at the pose
    where it happens -- its last ones at q1, whose frequency the ring
    carries into the still."""
    cs, Tm = _move_corners(np.asarray(q1, float) - np.asarray(q0, float), lim, f)
    if not cs:
        return [], Tm
    path = K.Path(q0).move(q1, lim.w_max, lim.a_max * f)
    wn = lim.w_ns(path.state(np.array([t for t, _ in cs]))[0])
    return [(t, da, w) for (t, da), w in zip(cs, wn)], Tm


def move_cost(q0, q1, lim, W):
    """(seconds, f): a move and its settling, at the acceleration fraction
    that makes their sum least (golden section on log f)."""
    key = (W,) + tuple(np.round(np.r_[q0, q1], 9))
    if key not in lim._moves:
        lim._moves[key] = _move_cost(q0, q1, lim, W)
    return lim._moves[key]


def _move_cost(q0, q1, lim, W):
    d = np.asarray(q1, float) - np.asarray(q0, float)
    if np.max(np.abs(d)) < 1e-9:
        return 0.0, 1.0

    def cost(lf):
        cs, Tm = move_ring(q0, q1, lim, exp(lf))
        return Tm + settle_after(cs, lim, W)
    lo, hi = log(1e-3), 0.0
    x1, x2 = hi - GOLDEN * (hi - lo), lo + GOLDEN * (hi - lo)
    c1, c2 = cost(x1), cost(x2)
    for _ in range(12):
        if c1 < c2:
            hi, x2, c2 = x2, x1, c1
            x1 = hi - GOLDEN * (hi - lo)
            c1 = cost(x1)
        else:
            lo, x1, c1 = x1, x2, c2
            x2 = lo + GOLDEN * (hi - lo)
            c2 = cost(x2)
    cE = cost(0.0)
    if cE <= min(c1, c2):
        return cE, 1.0
    return (c1, exp(x1)) if c1 < c2 else (c2, exp(x2))


# ================================================================ ITEMS
@dataclass(frozen=True)
class Still:
    q: tuple                   # rad, (outer, inner)
    dwell: float               # s of quiet


@dataclass(frozen=True)
class Spin:
    hinge: int
    park: float                # rad, the other hinge
    rate: float                # rad/s
    f: float                   # of the hinge's top acceleration, for its ramps


def _span():
    return 2.0 * pi * Gimbal.WRAP


def takt_s():
    """s: the most the station may spend on one bat, the line's takt at its
    default volume (spec.Line): no item of a plan may be longer."""
    return Line().takt_s


def spin_rates(lim, h, n):
    """n rates for hinge h, log-spaced, from the slowest whose two ways
    over the cable's wrap fit inside the takt to the fastest it may spin."""
    lo = 2.0 * 2.0 * _span() / takt_s()
    return np.exp(np.linspace(log(lo), log(lim.w_spin[h]), n))


def spin_shape(sp, lim, W):
    """(ramp s, cruise s, quiet start after the ramp s, duration of one
    way): a spin over the cable's whole wrap, one way."""
    key = (sp, W)
    if key not in lim._shapes:
        lim._shapes[key] = _spin_shape(sp, lim, W)
    return lim._shapes[key]


def _spin_shape(sp, lim, W):
    a = lim.a_spin(sp.hinge, sp.rate) * sp.f
    if a <= 0.0:
        return None
    ta = sp.rate / a
    dist = 2.0 * _span()
    tc = (dist - sp.rate * ta) / sp.rate
    if tc <= 0.0:
        return None
    # the spun hinge's inertia does not turn with its own angle: its ring
    # is at the park's frequency all the way round
    wn = lim.w_n(_spin_start(sp, 1.0))
    st = settle_after([(0.0, _on(sp.hinge, a), wn), (ta, _on(sp.hinge, -a), wn)], lim, W, sp.rate)
    return ta, tc, st, 2.0 * ta + tc


def _spin_start(sp, sgn):
    q = np.zeros(2)
    q[sp.hinge] = -sgn * _span()
    q[1 - sp.hinge] = sp.park
    return q


def best_spin_f(sp, lim, W):
    """The ramp's acceleration fraction that leaves the most quiet cruise
    per second of spin.  A cruise must turn the hinge a whole turn quietly:
    less, and the fit cannot tell the ripple a turning load puts on the
    hinges from their rate (fit, SPIN MOTION)."""
    def val(lf):
        s = spin_shape(Spin(sp.hinge, sp.park, sp.rate, exp(lf)), lim, W)
        if s is None:
            return -1.0
        ta, tc, st, T = s
        if (tc - st - 2.0 * F.MARGIN - F.fir_s()) * sp.rate < 2.0 * pi:      # fit.used_span
            return -1.0
        return (tc - st) / T
    lo, hi = log(1e-3), 0.0
    best = max(np.linspace(lo, hi, 25), key=val)
    return Spin(sp.hinge, sp.park, sp.rate, exp(best)), val(best)


def _nearest(q_target, q_now):
    """The same pose as q_target, each hinge by whole turns nearest q_now,
    inside the cable's wrap."""
    out = np.array(q_target, float)
    for h in (0, 1):
        k = np.round((q_now[h] - out[h]) / (2.0 * pi))
        v = out[h] + 2.0 * pi * k
        while v > _span():
            v -= 2.0 * pi
        while v < -_span():
            v += 2.0 * pi
        out[h] = v
    return out


# ================================================================ THE TOUR
@dataclass
class Tour:
    items: list
    path: object               # kin.Path, encoder angles, commanded
    quiet: list                # [fit.Quiet]
    starts: list               # station times the gimbal leaves a still
    corners: list              # [(t, da (2,), w_n (2,) at the pose there)]
    T: float                   # s, back at rest at the zeros


def boot_s():
    """s from power until the first item may start: the board's start, the
    timing's baseline (fit.timing: 2 x 100 samples) and the filter's memory."""
    return Imu.START_S + 200.0 / Imu.ODR + len(impulse(Imu.ODR)) / Imu.ODR


def return_s(q, lim):
    """s: the move from rest at q home to the zeros, at the motors' own
    acceleration -- no quiet interval follows it to settle for."""
    return _move_corners(-np.asarray(q, float), lim, 1.0)[1]


def build(items, lim, W):
    """The tour the station runs for these items, from rest at the zeros and
    back to them: the gantry unloads the parked gimbal there, and the next
    run counts its cable's wrap from there."""
    path = K.Path((0.0, 0.0))
    path.hold(boot_s(), tag="boot")
    quiet, starts, corners = [], [], []
    for i, it in enumerate(items):
        q_now = path.q.copy()
        if isinstance(it, Still):
            q1 = _nearest(it.q, q_now)
            _, f = move_cost(q_now, q1, lim, W)
            t0 = path.T
            if np.max(np.abs(q1 - q_now)) > 1e-12:
                starts.append(t0)
            path.move(q1, lim.w_max, lim.a_max * f, tag="move %d" % i)
            cs = [(t0 + t, da, wn) for t, da, wn in move_ring(q_now, q1, lim, f)[0]]
            corners += cs
            st = settle_after(cs, lim, W) if cs else 0.0
            tq = path.T + st
            path.hold(st + it.dwell, tag="still %d" % i)
            quiet.append(F.Quiet("still", tq, tq + it.dwell, i))
        else:
            sgn = -1.0 if q_now[it.hinge] > 0.0 else 1.0
            qs = _spin_start(it, sgn)
            qs[1 - it.hinge] = _nearest([0.0, it.park] if it.hinge == 0 else [it.park, 0.0], q_now)[1 - it.hinge]
            _, f = move_cost(q_now, qs, lim, W)
            t0 = path.T
            if np.max(np.abs(qs - q_now)) > 1e-12:
                starts.append(t0)
                path.move(qs, lim.w_max, lim.a_max * f, tag="to spin %d" % i)
                corners += [(t0 + t, da, wn) for t, da, wn in move_ring(q_now, qs, lim, f)[0]]
            shape = spin_shape(it, lim, W)
            if shape is None:
                raise ValueError("spin %s has no quiet cruise" % (it,))
            ta, tc, st, Tone = shape
            a = lim.a_spin(it.hinge, it.rate) * it.f
            wn = lim.w_n(qs)
            for sg in (sgn, -sgn):
                t0 = path.T
                path.spin(it.hinge, sg * 2.0 * Gimbal.WRAP, it.rate, a, tag="spin %d" % i)
                corners += [(t0, _on(it.hinge, sg * a), wn), (t0 + ta, _on(it.hinge, -sg * a), wn),
                            (t0 + ta + tc, _on(it.hinge, -sg * a), wn), (t0 + Tone, _on(it.hinge, sg * a), wn)]
                quiet.append(F.Quiet("spin", t0 + ta + st, t0 + ta + tc, i, it.hinge, sg * it.rate))
    q_end = path.q.copy()
    if np.max(np.abs(q_end)) > 1e-12:
        t0 = path.T
        starts.append(t0)
        path.move(np.zeros(2), lim.w_max, lim.a_max, tag="home")
        corners += [(t0 + t, da, wn) for t, da, wn in move_ring(q_end, np.zeros(2), lim, 1.0)[0]]
    return Tour(list(items), path, quiet, starts, corners, path.T)


# ============================================================ INFORMATION
def _pose_jac(kin, q):
    """(6, 14): a pose's residual against the gimbal's parameters."""
    out = np.empty((6, 14))
    R, x = kin.pose(np.atleast_2d(q))
    for j in range(14):
        dj = np.zeros(14)
        dj[j] = 1e-7
        out[:, j] = (K.pose_residual(kin.perturb(dj), np.atleast_2d(q), R, x)[0]
                     - K.pose_residual(kin.perturb(-dj), np.atleast_2d(q), R, x)[0]) / 2e-7
    return out


def pose_weight():
    """(6,) 1 / variance of a camera pose: the rig's budget (place.budget),
    per axis -- what M2 holds the cameras to."""
    from ..rig.place import budget
    deg, mm = budget()
    return 1.0 / np.array([radians(deg)] * 3 + [mm / 1000.0] * 3) ** 2


@lru_cache(maxsize=None)
def gimbal_spread():
    """(14,) 1 sigma of each of the gimbal's kinematic parameters about the
    drawing, in kin.perturb's order (each axis turned (2) and its point
    shifted (2), then the clamp's pose at zero (3 + 3)), in the cameras'
    world: the build's spreads as rig/mjcf.RigBuildDraw draws them
    (RigBuild: the centre, each axis' skew, the inner axis' offset square to
    it, the encoders' zeros) seen from a world turned and shifted off the
    rig's frame by about the zeros and the centre (sim.Station.draw),
    carried through the chain to first order.  Each parameter's own: the
    build moves some combinations not at all (the clamp's origin stays on
    the inner axis, the centre and the world's shift move alike), so their
    covariance is singular.  Any set of poses that fixes the 14 gives the
    same answer, since a built gimbal is a product of exponentials too."""
    B, E = RigBuild, np.eye(3)
    sk, ez = radians(B.AXIS_SKEW), radians(B.ENCODER_ZERO)
    ce, of = B.CENTRE / 1000.0, B.AXIS_OFFSET / 1000.0
    k0 = K.Chain.drawn().kin()
    g = np.linspace(0.0, 2.0 * pi, 3, endpoint=False)
    Q = np.array([(a, b) for a in g for b in g])
    J = np.concatenate([_pose_jac(k0, q) for q in Q])
    z3, z2 = np.zeros(3), np.zeros(2)

    def seen(centre=z3, tilt_o=z3, tilt_i=z3, off=z3, zero=z2, rw=z3, tw=z3):
        R, x = K.Chain(centre, K.exp_so3(tilt_o) @ E[0], K.exp_so3(tilt_i) @ E[1], off, zero).kin().pose(Q)
        Rw = K.exp_so3(rw)
        return Rw.T @ R, (x - tw) @ Rw
    # each source at 1 sigma: (keyword, its direction)
    src = ([("centre", ce * E[i]) for i in range(3)] + [("tilt_o", sk * E[i]) for i in (1, 2)]
           + [("tilt_i", sk * E[i]) for i in (0, 2)] + [("off", of * E[i]) for i in (0, 2)]
           + [("zero", ez * np.eye(2)[i]) for i in range(2)]
           + [("rw", ez * E[i]) for i in range(3)] + [("tw", ce * E[i]) for i in range(3)])
    h = 1e-3
    A = np.empty((14, len(src)))
    for j, (kw, v) in enumerate(src):
        dr = (K.pose_residual(k0, Q, *seen(**{kw: h * v})) - K.pose_residual(k0, Q, *seen(**{kw: -h * v}))).ravel()
        A[:, j] = -np.linalg.lstsq(J, dr / (2.0 * h), rcond=None)[0]
    return np.linalg.norm(A, axis=1)


def die_dT(t):
    return Imu.SELF_HEAT * (1.0 - np.exp(-np.asarray(t, float) / Imu.HEAT_TAU)) + (Site.AMBIENT - 25.0)


def nominal_obs(quiet, path, W, t0_board=None, per_turn=None):
    """fit.Obs at the drawing for a tour's quiet intervals, as the fit will
    cut them (its margins), with the datasheet's noise.  With per_turn, a
    spin keeps only every m-th window, m as large as leaves per_turn of
    them in each turn, each standing for m (its variance over m): the
    windows' information is a smooth function of the hinge's angle and of
    time, so an even subset sums to it (the trapezoid rule on a periodic
    function) -- for ranking candidates, never for a check."""
    T0_ = Imu.START_S if t0_board is None else t0_board
    facts = F.Facts.site()
    nd2 = F.datasheet_nd() ** 2
    rows = []
    for qi, Q in enumerate(quiet):
        ta, tb = F.used_span(Q)
        ka = int(ceil((ta - T0_ + facts.delay) * Imu.ODR))
        kb = int(np.floor((tb - T0_ + facts.delay) * Imu.ODR))
        if kb - ka < W:
            continue
        q, qd, _ = path.state(np.array([Q.t0]))
        if Q.kind == "still":
            rows.append((False, ka, kb, q[0], np.zeros(2), Q.t0, float(die_dT(0.5 * (Q.t0 + Q.t1))), qi))
        else:
            m = 1
            if per_turn is not None:
                m = max(1, int(2.0 * pi * Imu.ODR / (abs(Q.rate) * W * per_turn)))
            for k0 in range(ka, kb - W + 1, W * m):
                rows.append((True, k0, k0 + W, q[0], qd[0], Q.t0,
                             float(die_dT(T0_ + (k0 + W / 2.0) / Imu.ODR)), qi, m))
    var = {}
    V = []
    for r in rows:
        key = (r[0], r[2] - r[1])
        if key not in var:
            var[key] = F.obs_var(nd2, r[2] - r[1], DESIGN_PLACE, run=r[0])
        V.append(var[key] / (r[8] if len(r) > 8 else 1))
    return F.Obs(moving=np.array([r[0] for r in rows], bool), k0=np.array([r[1] for r in rows], int),
                 k1=np.array([r[2] for r in rows], int), q0=np.array([r[3] for r in rows]).reshape(-1, 2),
                 qd=np.array([r[4] for r in rows]).reshape(-1, 2), tref=np.array([r[5] for r in rows]),
                 dT=np.array([r[6] for r in rows]), y=np.zeros((len(rows), 6)), var=np.array(V).reshape(-1, 6),
                 quiet=np.array([r[7] for r in rows], int))


@dataclass
class Info:
    """Information, additive over items: the fit's parameters (H), their
    coupling to the gimbal's refinement (C), the poses' on the gimbal (Hk),
    the IMU's on it (Kk), and the degrees of freedom the stills and spins
    give the noise (dof)."""
    H: np.ndarray
    C: np.ndarray
    Hk: np.ndarray
    Kk: np.ndarray = None
    dof: float = 0.0

    def __post_init__(self):
        if self.Kk is None:
            self.Kk = np.zeros((14, 14))

    def __add__(self, o):
        return Info(self.H + o.H, self.C + o.C, self.Hk + o.Hk, self.Kk + o.Kk, self.dof + o.dof)

    @classmethod
    def zero(cls):
        return cls(np.zeros((F.NP, F.NP)), np.zeros((F.NP, 14)), np.zeros((14, 14)))


def information(obs, th, kin, poses_q=(), with_kin=True):
    facts = F.Facts.site()
    m = F.Model(kin, obs, facts, np.array([0.0, 0.0, 1.0]))
    wh = 1.0 / np.sqrt(obs.var)
    C, Kk = np.zeros((F.NP, 14)), np.zeros((14, 14))
    if len(obs):
        J = (m.jacobian(th) * wh[:, :, None]).reshape(-1, F.NP)
        H = J.T @ J
        if with_kin:
            Jk = (m.kin_jacobian(th) * wh[:, :, None]).reshape(-1, 14)
            C, Kk = J.T @ Jk, Jk.T @ Jk
    else:
        H = np.zeros((F.NP, F.NP))
    Hk = np.zeros((14, 14))
    pw = pose_weight()
    for q in poses_q:
        Jp = _pose_jac(kin, q)
        Hk += Jp.T @ (pw[:, None] * Jp)
    return Info(H, C, Hk, Kk, noise_dof_of(obs))


def noise_dof_of(obs):
    """The degrees of freedom obs's stills and spins give the noise (fit,
    NOISE): a spin by its span, whatever windows a ranking kept."""
    W = F.window_W()
    dof = sum(F.still_noise_dof(n, W) for n in (obs.k1 - obs.k0)[~obs.moving])
    for qi in np.unique(obs.quiet[obs.moving]):
        m = obs.moving & (obs.quiet == qi)
        dof += F.spin_noise_dof(int(obs.k1[m].max() - obs.k0[m].min()), W)
    return float(dof)


def spd_inv(H):
    """The inverse of a symmetric information matrix, or None where the data
    leave a direction undetermined: scaled to unit diagonal, its smallest
    eigenvalue under the rank tolerance numpy's matrix_rank uses."""
    d = np.diag(H)
    if np.any(d <= 0.0):
        return None
    D = 1.0 / np.sqrt(d)
    A = D[:, None] * H * D[None, :]
    w = np.linalg.eigvalsh(A)              # (eigvalsh, not eigh: a threaded BLAS makes eigh 20x slower at this size)
    if w[0] <= w[-1] * len(w) * np.finfo(float).eps:
        return None
    return D[:, None] * np.linalg.inv(A) * D[None, :]


class Value:
    """V and each term's sigma from information, with priors."""

    def __init__(self, th, s, spread, weak=True):
        self.th, self.s = th, s
        self.Jt = F.terms_jacobian(th, float(die_dT(Imu.HEAT_TAU)))
        pm, psig = F.priors(Imu.START_S)
        P = np.zeros(F.NP)
        fin = np.isfinite(psig)
        P[fin] = 1.0 / psig[fin] ** 2
        # gravity's up is known to a radian before any still: a prior that
        # only keeps the matrix invertible
        P[F.UU] = 1.0
        self.P = np.diag(P)
        self.Pw = np.zeros((F.NP, F.NP))
        self.Pk = np.zeros((14, 14))
        if weak:
            # the parts' spreads, on the terms: a weak prior that ranks the
            # first candidates and is gone before the design is checked
            Jt = self.Jt
            Wt = np.diag(1.0 / spread ** 2)
            self.Pw = Jt.T @ Wt @ Jt
            # and the gimbal's build, seen from the cameras' world (its
            # rotation and shift unknown): each parameter at its spread
            self.Pk = np.diag(1.0 / gimbal_spread() ** 2)

    def cov(self, info):
        """(NT, NT) the judged terms' covariance, or None if the
        information is short of rank or the poses alone leave the gimbal
        undetermined (stage 1 needs them to).  The fit's (fit, THE TWO
        STAGES): the parameters and the gimbal's refinement together, the
        refinement's prior the poses' information."""
        if spd_inv(info.Hk + self.Pk) is None:
            return None
        Hi = spd_inv(np.block([[info.H + self.P + self.Pw, info.C],
                               [info.C.T, info.Hk + info.Kk + self.Pk]]))
        if Hi is None:
            return None
        return self.Jt @ Hi[:F.NP, :F.NP] @ self.Jt.T

    def sigma(self, info):
        St = self.cov(info)
        if St is None:
            return np.full(F.NT, np.inf)
        return np.sqrt(np.maximum(np.diag(St), 0.0))

    def V(self, info):
        """(V, sigma): with the noise's measurement held to its own
        target (fit.noise_dof) the same way, its variance as dof^-1."""
        sg = self.sigma(info)
        noise = max(0.0, F.noise_dof() / max(info.dof, 0.5) - 1.0)
        fin = np.isfinite(self.s)          # a term with no threshold asks nothing
        return float(np.sum(np.maximum(0.0, sg[fin] ** 2 / self.s[fin] ** 2 - 1.0)) + noise), sg


# ============================================================ CANDIDATES
def fibonacci(n):
    """n unit vectors spread evenly over the sphere (Gonzalez 2010)."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = i * pi * (3.0 - sqrt(5.0))
    r = np.sqrt(1.0 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def poses_for(dv):
    """The gimbal's two (outer, inner) angles that put gravity's DOWN along
    dv (unit, clamp frame) with the gimbal as drawn: R = Rx(a) Ry(b) gives
    down = (sin b cos a, -sin a, -cos a cos b)."""
    out = []
    s = -dv[1]
    for a in (np.arcsin(np.clip(s, -1.0, 1.0)), pi - np.arcsin(np.clip(s, -1.0, 1.0))):
        c = np.cos(a)
        if abs(c) < 1e-9:
            b = 0.0
        else:
            b = np.arctan2(dv[0] / c, -dv[2] / c)
        a = (a + pi) % (2.0 * pi) - pi
        out.append((float(a), float(b)))
    return out


# ================================================================ DESIGN
@dataclass
class CalPlan:
    items: list
    tour: Tour
    lim: Limits
    W: int
    theta0: np.ndarray
    s: np.ndarray
    spread: np.ndarray
    sigma: np.ndarray              # (NT,) predicted 1 sigma at the drawing
    V: float
    density: tuple                 # (directions, parks, rates) the design converged at
    log: list = field(default_factory=list)

    @property
    def duration_s(self):
        """s from power to the gimbal parked at its zeros again."""
        return self.tour.T

    @property
    def path(self):
        return self.tour.path

    @property
    def quiet(self):
        return self.tour.quiet

    @property
    def starts(self):
        return np.array(self.tour.starts)

    @property
    def hash(self):
        # the items' values as bytes, rounded to a nanounit (and -0 made 0):
        # the same plan hashes the same whatever prints its numbers
        h = hashlib.sha1()
        for i in self.items:
            h.update(type(i).__name__.encode())
            h.update((np.round(np.hstack([np.ravel(v) for v in vars(i).values()]).astype("<f8"), 9) + 0.0).tobytes())
        return h.hexdigest()[:8]

    def lag(self):
        """rad: the most a hinge may fall behind its command, for the fit's
        following-error test.  On the stepper's small-signal spring k, the
        lag is what the commanded motion needs of the motor -- gravity,
        every inertia, the bodies on each other (kin.rnea on the drawing) --
        over k, plus each corner's ring (its step response less the step's
        steady share, which the torque already holds, at the frequency of
        the pose where it happens, summed with their phases); the torque
        less the hinge's own inertia's share is raised by how near the
        harmonics of the faster hinge's rate come to the hinge's lowest
        natural frequency, 1 / (1 - (NH w / w_n)^2): a spun hinge forces
        the held one too.  The motor's sine law (T sin(N e), T its pull-out
        torque at the command's rate) holds that spring torque k e at the
        lag arcsin(k e / T) / N; plus one encoder code.  Raises ValueError
        where k e reaches T: the plan loses steps there."""
        if "_lag" in self.__dict__:
            return self.__dict__["_lag"]
        lim = self.lim
        k = lim.stiffness()
        t = np.arange(0.0, self.duration_s, 1.0 / Gimbal.ENCODER_HZ)
        q, qd, qdd = self.path.state(t)
        g = np.array([0.0, 0.0, -K.gravity(Site.LAT, Site.ALT)])
        tau = K.rnea(lim.chain, lim.outer, lim.inner, q, qd, qdd, g, (lim.arm, lim.arm))
        own = K.hinge_inertia(lim.chain, lim.outer, lim.inner, q, (lim.arm, lim.arm)) * qdd
        x = (F.NH * np.max(np.abs(qd), 1)[:, None] / lim.w_n_min()[None, :]) ** 2
        if np.any(x >= 1.0):
            raise ValueError("the plan turns a load at a hinge's resonance")
        e = tau / k
        for tc, da, w in self.tour.corners:
            m = t >= tc
            for h in (0, 1):
                if da[h] == 0.0:
                    continue
                s_ = lim.decay(h, w[h])
                wd = sqrt(w[h] ** 2 - s_ ** 2)
                u = t[m] - tc
                e[m, h] -= da[h] / w[h] ** 2 * np.exp(-s_ * u) * (np.cos(wd * u) + s_ / wd * np.sin(wd * u))
        spring = k * np.abs(e) + np.abs(tau - own) * x / (1.0 - x)
        T = _pullout(qd)
        if np.any(spring >= T):
            i, h = np.argwhere(spring >= T)[0]
            raise ValueError("hinge %d at %.2f s needs %.2f N m of a %.2f N m pull-out torque: the plan loses steps"
                             % (h, t[i], spring[i, h], T[i, h]))
        out = np.arcsin(spring / T) / (Stepper.TEETH * Gimbal.GEAR)
        self.__dict__["_lag"] = float(out.max()) + 2.0 * pi / 2 ** Gimbal.ENCODER_BITS
        return self.__dict__["_lag"]

    def binding(self):
        return {"rate": self.lim.why_w, "acceleration": self.lim.why_a}


def design(d, rig=None, density=None, log_=None, s=None):
    """The plan for bat design d (CalPlan).  `s` overrides the thresholds."""
    rig = RS.design(d) if rig is None else rig
    lim = limits(d, rig)
    W = F.window_W()
    th = nominal(d, rig)
    s = F.thresholds() if s is None else np.asarray(s, float)
    spread = spreads(d, rig)
    kin = K.Chain.drawn().kin()
    logs = [] if log_ is None else log_
    dens = (16, 4, 9) if density is None else density
    cache = {}
    best = None
    one_settle = _typical_settle(lim, W)
    while True:
        plan = _design_at(d, rig, lim, W, th, s, spread, kin, dens, logs, cache)
        logs.append("density %s: %d items, %.1f s, V %.3g" % (dens, len(plan.items), plan.duration_s, plan.V))
        if density is not None:
            return plan
        if plan.V <= 0.0 and best is not None and best.V <= 0.0 \
                and plan.duration_s > best.duration_s - one_settle:
            # a denser candidate set no longer buys a plan a settle shorter
            return plan if plan.duration_s < best.duration_s else best
        if best is None or (plan.V <= 0.0 and (best.V > 0.0 or plan.duration_s < best.duration_s)):
            best = plan
        if dens[0] >= 256:
            return best
        # each set holds the last: spherical Fibonacci sets don't nest, but
        # the parks and the log-spaced rates do
        dens = (dens[0] * 2, dens[1] * 2, dens[2] * 2 - 1)


def _typical_settle(lim, W):
    """s: the settle after the hardest stop the gimbal makes."""
    q = np.zeros(2)
    return max(settle_after([(0.0, _on(h, lim.a_max[h]), lim.w_n(q))], lim, W) for h in (0, 1))


def _still_samples(D):
    """Samples a still of D s of quiet gives the fit (fit.used_span)."""
    return int((D - 2.0 * F.MARGIN - F.fir_s()) * Imu.ODR)


def _golden_max(f, lo, hi, n=14):
    x1, x2 = hi - GOLDEN * (hi - lo), lo + GOLDEN * (hi - lo)
    f1, f2 = f(x1), f(x2)
    for _ in range(n):
        if f1 > f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - GOLDEN * (hi - lo)
            f1 = f(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + GOLDEN * (hi - lo)
            f2 = f(x2)
    return (f1, x1) if f1 > f2 else (f2, x2)


def _design_at(d, rig, lim, W, th, s, spread, kin, dens, logs, cache=None):
    n_dir, n_park, n_rate = dens
    cache = {} if cache is None else cache
    val_w = Value(th, s, spread, weak=True)
    val = Value(th, s, spread, weak=False)
    facts = F.Facts.site()
    nd2 = F.datasheet_nd() ** 2
    pw = pose_weight()
    poses = [q for dv in fibonacci(n_dir) for q in poses_for(dv)]
    rates = {h: spin_rates(lim, h, n_rate) for h in (0, 1)}

    def spin_at(h, p, w):
        """(Spin, its quiet share) at hinge h, park p, rate w, its ramp chosen."""
        key = ("f", h, round(float(p), 12), round(float(w), 12))
        if key not in cache:
            cache[key] = best_spin_f(Spin(h, float(p), float(w), 1.0), lim, W)
        return cache[key]
    spins = []
    for h in (0, 1):
        for p in np.linspace(-pi, pi, n_park, endpoint=False):
            for w in rates[h]:
                sp, v = spin_at(h, p, w)
                if v > 0.0:
                    spins.append(sp)
    dT_ref = float(die_dT(Imu.HEAT_TAU))

    def pose_rows(q):
        """A still's Jacobians at the drawing: its mean (the dwell changes
        only its noise), the gimbal's pull on it, the camera pose's."""
        key = ("pose", q)
        if key not in cache:
            obs = F.Obs(np.zeros(1, bool), np.zeros(1, int), np.ones(1, int), np.array([q]), np.zeros((1, 2)),
                        np.zeros(1), np.array([dT_ref]), np.zeros((1, 6)), np.ones((1, 6)), np.zeros(1, int))
            m = F.Model(kin, obs, facts, np.array([0.0, 0.0, 1.0]))
            cache[key] = (m.jacobian(th, time=False)[0], m.kin_jacobian(th)[0], _pose_jac(kin, q))
        return cache[key]

    def still_imu(q, D):
        J, Jk, _ = pose_rows(q)
        n = max(_still_samples(D), 1)
        w = 1.0 / F.obs_var(nd2, n, DESIGN_PLACE)
        return Info(J.T @ (w[:, None] * J), J.T @ (w[:, None] * Jk), np.zeros((14, 14)), Jk.T @ (w[:, None] * Jk),
                    F.still_noise_dof(n, W))

    def still_pose(q):
        Jp = pose_rows(q)[2]
        return Info(np.zeros((F.NP, F.NP)), np.zeros((F.NP, 14)), Jp.T @ (pw[:, None] * Jp))

    def spin_info(sp, t_at):
        """A spin's information as if it began at t_at: computed once where
        it begins alone (its windows thinned, nominal_obs), then moved -- a
        later start only adds to the clock's lever (T0's column into rho's)
        and warms the die (each bias's column into its temperature
        coefficient's)."""
        key = ("spin", sp)
        if key not in cache:
            tour = build([sp], lim, W)
            obs = nominal_obs(tour.quiet, tour.path, W, per_turn=SPIN_RANK_PER_TURN)
            cache[key] = (information(obs, th, kin), tour.quiet[0].t0)
        inf, t0 = cache[key]
        M = np.eye(F.NP)
        M[F.T0, F.RHO] = t_at - t0
        dT = float(die_dT(t_at) - die_dT(t0))
        for b, c in ((F.BA, F.CA), (F.BG, F.CG)):
            M[np.r_[b], np.r_[c]] = dT
        return Info(M.T @ inf.H @ M, M.T @ inf.C, np.zeros((14, 14)), inf.Kk, inf.dof)

    def spin_time(sp):
        return 2.0 * spin_shape(sp, lim, W)[3]

    D_min = (4.0 * W + 1.0) / Imu.ODR + 2.0 * F.MARGIN + F.fir_s()
    D_max = takt_s()
    # stage 1 takes the gimbal's kinematics from the stills' poses alone:
    # start from the fewest stills that determine them, each the one that
    # adds most to log det (D-optimal, Fedorov 1972; the weak prior only
    # ranks the first)
    items = []
    Hk = np.zeros((14, 14))
    while spd_inv(Hk) is None:
        held = {it.q for it in items}
        free = [q for q in poses if q not in held]
        if not free:
            raise ValueError("no set of the candidate stills determines the gimbal's kinematics")
        q = max(free, key=lambda q: np.linalg.slogdet(Hk + still_pose(q).Hk + val_w.Pk)[1])
        it = Still(q, round(ceil(D_min * 1000.0) / 1000.0, 3))
        items.insert(_insert_at(items, it, lim, W), it)
        Hk = Hk + still_pose(q).Hk
    for _ in range(400):
        tour = build(items, lim, W)
        t_spin = {}
        for Q in tour.quiet:
            t_spin.setdefault(Q.item, Q.t0)
        info = Info.zero()
        for k, it in enumerate(items):
            if isinstance(it, Still):
                info = info + still_imu(it.q, it.dwell) + still_pose(it.q)
            else:
                info = info + spin_info(it, t_spin.get(k, tour.T))
        Vt, _ = val.V(info)
        if Vt <= 0.0 and items:
            plan = _check(items, lim, W, th, s, spread, kin, val, dens)
            if plan.V <= 0.0:
                plan = _prune(plan, lim, W, th, s, spread, kin, val, dens)
                return _order(plan, lim, W, th, s, spread, kin, val, dens)
            info = _exact_info(plan.tour, th, kin, W)
            Vt, _ = val.V(info)
        Vw, _ = val_w.V(info)
        rank = val_w if Vw > 0.0 else val
        V0 = Vw if Vw > 0.0 else Vt
        best = (0.0, None)
        held = {it.q for it in items if isinstance(it, Still)}
        for q in poses:
            if q in held:
                continue
            mc = _insert_cost(items, q, q, lim, W)
            base = info + still_pose(q)

            def r_new(lD):
                return (V0 - rank.V(base + still_imu(q, exp(lD)))[0]) / (mc + exp(lD))
            r, x = _golden_max(r_new, log(D_min), log(D_max))
            if r > best[0]:
                best = (r, ("new", Still(q, exp(x))))
        for k, it in enumerate(items):
            if not isinstance(it, Still):
                continue
            have = still_imu(it.q, it.dwell)

            def r_ext(lD):
                add = still_imu(it.q, it.dwell + exp(lD))
                inf = Info(info.H - have.H + add.H, info.C - have.C + add.C, info.Hk, info.Kk - have.Kk + add.Kk,
                           info.dof - have.dof + add.dof)
                return (V0 - rank.V(inf)[0]) / exp(lD)
            r, x = _golden_max(r_ext, log(D_min), log(D_max))
            if r > best[0]:
                best = (r, ("extend", k, exp(x)))
        def spin_ratio(sp):
            qs = _spin_start(sp, 1.0)
            mc = _insert_cost(items, qs, qs, lim, W)
            return (V0 - rank.V(info + spin_info(sp, tour.T))[0]) / (mc + spin_time(sp))
        best_spin = (0.0, None)
        for sp in spins:
            r = spin_ratio(sp)
            if r > best_spin[0]:
                best_spin = (r, sp)
        if best_spin[1] is not None and best_spin[0] > best[0]:
            # the grid brackets the best spin: its rate, then its park, by
            # golden section between the grid's neighbours
            sp = best_spin[1]
            h = sp.hinge
            step = log(rates[h][1] / rates[h][0])

            def by_rate(lw):
                c, v = spin_at(h, sp.park, exp(lw))
                return spin_ratio(c) if v > 0.0 else -np.inf
            r1, x1 = _golden_max(by_rate, max(log(sp.rate) - step, log(rates[h][0])),
                                 min(log(sp.rate) + step, log(rates[h][-1])))
            if r1 > best_spin[0]:
                best_spin = (r1, spin_at(h, sp.park, exp(x1))[0])
            sp = best_spin[1]
            dp = 2.0 * pi / n_park

            def by_park(p):
                c, v = spin_at(h, p, sp.rate)
                return spin_ratio(c) if v > 0.0 else -np.inf
            r2, x2 = _golden_max(by_park, sp.park - dp, sp.park + dp)
            if r2 > best_spin[0]:
                best_spin = (r2, spin_at(h, x2, sp.rate)[0])
            best = (best_spin[0], ("new", best_spin[1]))
        if best[1] is None:
            break
        act = best[1]
        logs.append("  V %.3g (%s): %s" % (V0, "weak priors" if rank is val_w else "data alone",
                                          act[0] + " " + (repr(act[1]) if act[0] == "new" else
                                                           "still %d by %.3f s" % (act[1], act[2]))))
        if act[0] == "extend":
            k = act[1]
            items[k] = Still(items[k].q, round(items[k].dwell + act[2], 3))
        else:
            it = act[1]
            if isinstance(it, Still):
                it = Still(it.q, round(it.dwell, 3))
            items.insert(_insert_at(items, it, lim, W), it)
    return _check(items, lim, W, th, s, spread, kin, val, dens)


def _shifted(path, off):
    p = K.Path(path.axes[0].q[0:1] + path.axes[1].q[0:1])
    p.axes = tuple(K.Axis([t + off for t in ax.t], list(ax.q), list(ax.v), list(ax.a)) for ax in path.axes)
    p.T = path.T + off
    p.segments = path.segments
    p.q = path.q
    return p


def _ends(it):
    if isinstance(it, Still):
        return np.array(it.q), np.array(it.q)
    q = _spin_start(it, 1.0)
    return q, q


def _insertions(items, q_in, q_out, lim, W):
    """[s]: what an item entered at q_in and left at q_out adds to the tour
    in each place it could go, before item k or last (the moves added, less
    the move it replaces).  The tour is closed: the last place is before
    the return to the zeros, which unwinds the cable whole."""
    out = []
    prev = np.zeros(2)
    for k in range(len(items) + 1):
        a = move_cost(prev, _nearest(q_in, prev), lim, W)[0]
        if k < len(items):
            nxt = _ends(items[k])[0]
            b = move_cost(q_out, _nearest(nxt, q_out), lim, W)[0]
            c = move_cost(prev, _nearest(nxt, prev), lim, W)[0]
            prev = _ends(items[k])[1]
        else:
            b, c = return_s(_nearest(q_out, prev), lim), return_s(prev, lim)
        out.append(a + b - c)
    return out


def _insert_cost(items, q_in, q_out, lim, W):
    """s: the cheapest place in the tour for an item entered at q_in and
    left at q_out."""
    return min(_insertions(items, q_in, q_out, lim, W))


def _insert_at(items, it, lim, W):
    return int(np.argmin(_insertions(items, *_ends(it), lim, W)))


def _exact_info(tour, th, kin, W):
    obs = nominal_obs(tour.quiet, tour.path, W)
    q_st = [tour.path.state(np.array([Q.t0]))[0][0] for Q in tour.quiet if Q.kind == "still"]
    return information(obs, th, kin, poses_q=q_st, with_kin=True)


def _check(items, lim, W, th, s, spread, kin, val, dens):
    tour = build(items, lim, W)
    info = _exact_info(tour, th, kin, W)
    V, sg = val.V(info)
    return CalPlan(list(items), tour, lim, W, th, s, spread, sg, V, dens)


def _prune(plan, lim, W, th, s, spread, kin, val, dens):
    """Drop whole items, the costliest first, while the plan still meets
    every threshold on the tour as it would run."""
    items = list(plan.items)
    k = 0
    order = sorted(items, key=lambda it: -(it.dwell if isinstance(it, Still) else 2.0 * spin_shape(it, lim, W)[3]))
    for it in order:
        trial = [x for x in items if x is not it]
        if not trial:
            continue
        p = _check(trial, lim, W, th, s, spread, kin, val, dens)
        if p.V <= 0.0:
            items, plan = trial, p
    return plan


def _order(plan, lim, W, th, s, spread, kin, val, dens):
    """2-opt on the tour's order, by its real duration."""
    items = list(plan.items)
    best_T = build(items, lim, W).T
    improved = True
    while improved:
        improved = False
        for i in range(len(items) - 1):
            for j in range(i + 1, len(items)):
                trial = items[:i] + items[i:j + 1][::-1] + items[j + 1:]
                T = build(trial, lim, W).T
                if T < best_T - 1e-6:
                    items, best_T, improved = trial, T, True
    p = _check(items, lim, W, th, s, spread, kin, val, dens)
    return p if p.V <= 0.0 else plan


def terms_cov(plan):
    """(NT, NT) the judged terms' covariance the plan predicts at the
    drawing, on the tour as it runs (what plan.sigma is the diagonal of)."""
    kin = K.Chain.drawn().kin()
    info = _exact_info(plan.tour, plan.theta0, kin, plan.W)
    return Value(plan.theta0, plan.s, plan.spread, weak=False).cov(info)


def for_design(d):
    """The bats' plan for d's bat, as golden.day_design picks it (the plan
    for the bats alone, or the one that also lets one golden run watch the
    rig), designed once per bat."""
    from . import golden as G
    return G.for_design(d).plan
