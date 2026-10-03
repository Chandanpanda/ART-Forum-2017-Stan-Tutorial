"""A bat through station 4 in simulation: the motion as the motors make
it, the board where it sits, the IMU's stream, the encoders, the cameras'
poses.  TRUTH: only the checks import this; the fit sees what run_station
hands it (fit.CalInputs) and nothing else.

THE STATION, drawn once (Station.draw): the gimbal as built
(rig.mjcf.RigBuildDraw), the rig's base off level (Site.LEVEL), the
cameras' world -- "where the gimbal said the gauge sample was" (rig/calib.py),
so a rigid transform off the rig's frame by about the encoders' zeros and
the gimbal's centre (RigBuild.ENCODER_ZERO, CENTRE).

THE MOTION (motion()), once per station and plan.  The plan commands each
hinge's angle (plan.CalPlan.path).  A hybrid stepper pulls its rotor toward
the commanded angle by its sine law, so a hinge lags its command by e, where

    M(q) e'' + c e' + T(q_c') sin(N e) = tau(q_c, q_c', q_c'')

at the axis: N the electrical radians an axis radian (Stepper.TEETH x
Gimbal.GEAR), T the motor's pull-out torque through the belt at the
command's rate (plan._pullout: the holding torque at rest, less at speed),
c its driver's damping (plan.Limits.damping), and tau the torque the
commanded motion needs -- every inertia, the bodies' pull on each other,
gravity -- from MuJoCo's inverse dynamics on the gimbal as built and
weighed (scene.py).  At rest the sine's slope is the holding stiffness
(plan.Limits.stiffness): kin.ringing's linear spring, which the plan settles
by, is its small-signal model.  The torque is at its most where N|e| is
pi/2; past it the torque falls as the lag grows and the rotor slips a step,
so the simulation refuses the plan there (ValueError) rather than run a
hinge its encoder would catch off its command.  plan.limits sizes the
accelerations so the ring after each step of command stays inside the
pull-out torque, and check_calibration holds the simulated peak N|e| under
pi/2.  Taking tau and M along the command rather than the true motion is
first order in the lag: what it leaves out is the torque's slope in the
hinges times e, small beside the motor's own slope N T.  The true hinges are
the command less e; MuJoCo gives the clamp's attitude, rate, angular
acceleration and its origin's acceleration there.  The grid is the IMU's
internal rate (model.internal_rate).

THE BAT (draw_bat): the part's errors (model.draw_errors), the chip on its
board (placed on the board's face, along and across it), the board in its
slot on its stop, the bat in the jaws (seated across its axis, the stop
holding it along) -- each spread as plan.spreads() states it, so the plan
is designed against the bats this draws.  The board stays on its stop: the
station seats it toe-down before the pogo block closes, and the plan keeps
the specific force along the slot under the block's grip (plan.limits).

THE STREAM (run_station): the IMU's true rate and specific force at its
sensing centre -- the Earth's turn added, gravity from the tilted base --
through model.sense.  The board is powered as the plan starts; its first
sample comes Imu.START_S later at an unknown phase of the station's clock.
The encoders read the true hinges less their zeros, cut to ENCODER_BITS, at
ENCODER_HZ on the station's clock.  The cameras give the clamp's pose at
each still, in their world, with noise at the rig's budget (place.budget):
the worst M2 allows, as check_cameras holds the rig to it.
"""
from dataclasses import dataclass
from math import radians, sqrt, pi

import numpy as np

from ..spec import Imu, Site, Gimbal, Stepper, Board, Print, Clamp, RigBuild
from ..rig import mjcf as RM
from . import kin as K
from . import model as MD
from . import plan as P
from . import fit as F
from .scene import Scene, at_point, inner_motion


# ================================================================ STATION
@dataclass
class Station:
    build: object              # rig.mjcf.RigBuildDraw
    scene: object              # scene.Scene, as built
    up: np.ndarray             # gravity's up in the rig's frame: the base off level
    world: tuple               # (R, t m): the cameras' world -> the rig's frame
    g: float

    @classmethod
    def draw(cls, rig, rng, cams=()):
        """A station as built; with the rig's cameras (rig/place.py) when
        it is to be filmed through them."""
        b = RM.RigBuildDraw.draw(rig, list(cams), rng)
        tilt = rng.normal(0.0, radians(Site.LEVEL), 2)
        up = K.exp_so3(np.array([tilt[0], tilt[1], 0.0])) @ np.array([0.0, 0.0, 1.0])
        Rw = K.exp_so3(rng.normal(0.0, radians(RigBuild.ENCODER_ZERO), 3))
        tw = rng.normal(0.0, RigBuild.CENTRE / 1000.0, 3)
        return cls(b, Scene(rig, b), up, (Rw, tw), K.gravity(Site.LAT, Site.ALT))

    @classmethod
    def drawing(cls, rig):
        b = RM.RigBuildDraw.drawing(rig, [])
        return cls(b, Scene(rig, b), np.array([0.0, 0.0, 1.0]), (np.eye(3), np.zeros(3)),
                   K.gravity(Site.LAT, Site.ALT))

    def g_w(self):
        return -self.g * self.up

    def earth(self):
        """rad/s, the Earth's turn in the rig's frame."""
        return K.earth_rate(Site.LAT, self.up, K.north_rig(Site.HEADING))


# ================================================================ MOTION
@dataclass
class Motion:
    t: np.ndarray              # (n,) station time, the internal grid
    q: np.ndarray              # (n, 2) the true hinges, encoder angles
    qd: np.ndarray
    qdd: np.ndarray
    lag: np.ndarray            # (n, 2) command less truth
    R: np.ndarray              # (n, 3, 3) the clamp in the rig's frame
    x: np.ndarray              # (n, 3) m
    w: np.ndarray              # (n, 3) rad/s, clamp frame, relative to the room
    wd: np.ndarray             # (n, 3) rad/s^2, clamp frame
    a: np.ndarray              # (n, 3) m/s^2, the clamp origin's acceleration, clamp frame
    tau: np.ndarray            # (n, 2) N m, what the motors gave


def _command_torques(scene, path, t, g_w):
    """(tau (n, 2), M (n, 2, 2)) along the command."""
    qc, qdc, qddc = path.state(t)
    scene.gravity(g_w)
    tau = np.empty((len(t), 2))
    M = np.empty((len(t), 2, 2))
    for k in range(len(t)):
        scene.set(qc[k], qdc[k], qddc[k])
        tau[k] = scene.torque()
        M[k] = scene.mass()
    return tau, M, (qc, qdc, qddc)


def _lag(t, tau, M, c, w_c, k=None):
    """e (n, 2), e', e'': M e'' + c e' + T(w_c) sin(N e) = tau from rest,
    the stepper's own law (THE MOTION), RK4 on the grid with tau, M and the
    command's rate straight between its points.  w_c (n, 2) is the
    command's axis rates on t, which set each motor's pull-out torque T.
    Raises ValueError where a hinge's N|e| reaches pi/2: past it the motor's
    torque falls as the lag grows, and the rotor slips a step.  k (N m/rad,
    scalar or (2,)) given: the linear spring k e in its place, the law's
    small-signal model (kin.ringing's), and w_c is not read."""
    n = len(t)
    e = np.zeros((n, 2))
    v = np.zeros((n, 2))
    acc = np.zeros((n, 2))
    Mi = np.linalg.inv(M)
    N = Stepper.TEETH * Gimbal.GEAR                    # electrical radians an axis radian
    if k is None:
        w_c = np.asarray(w_c, float)
        T, Tm = P._pullout(w_c), P._pullout(0.5 * (w_c[:-1] + w_c[1:]))

        def pull(T_, e_):
            return T_ * np.sin(N * e_)
    else:
        T = np.broadcast_to(np.asarray(k, float), (n, 2))
        Tm = T[1:]

        def pull(k_, e_):
            return k_ * e_

    def f(Mi_, tau_, T_, e_, v_):
        return Mi_ @ (tau_ - c * v_ - pull(T_, e_))
    acc[0] = f(Mi[0], tau[0], T[0], e[0], v[0])
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        Mm, tm = 0.5 * (Mi[i] + Mi[i + 1]), 0.5 * (tau[i] + tau[i + 1])
        k1v, k1a = v[i], acc[i]
        k2v = v[i] + 0.5 * h * k1a
        k2a = f(Mm, tm, Tm[i], e[i] + 0.5 * h * k1v, k2v)
        k3v = v[i] + 0.5 * h * k2a
        k3a = f(Mm, tm, Tm[i], e[i] + 0.5 * h * k2v, k3v)
        k4v = v[i] + h * k3a
        k4a = f(Mi[i + 1], tau[i + 1], T[i + 1], e[i] + h * k3v, k4v)
        e[i + 1] = e[i] + h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        v[i + 1] = v[i] + h / 6.0 * (k1a + 2.0 * k2a + 2.0 * k3a + k4a)
        acc[i + 1] = f(Mi[i + 1], tau[i + 1], T[i + 1], e[i + 1], v[i + 1])
    if k is None:
        x = N * np.abs(e)
        past = np.argwhere(x >= pi / 2.0)
        if len(past):
            i, j = past[0]
            raise ValueError("hinge %d, at %.3f s (N|e| up to %.2f rad against pi/2), passed its pull-out angle: "
                             "the plan loses steps" % (j, t[i], x[:, j].max()))
    return e, v, acc


def motion(station, plan, f_int, lags=True, span=None):
    """Motion: the plan as the motors run it on this station, on a grid at
    f_int from 0 to the plan's end (or over span, the lag from rest at its
    start).  Raises ValueError where a hinge passes its pull-out angle
    (_lag).  lags=False: the hinges exactly on their command (for the
    checks)."""
    lim = plan.lim
    t0, t1 = (0.0, plan.duration_s) if span is None else span
    t = t0 + np.arange(int(np.ceil((t1 - t0) * f_int)) + 1) / f_int
    tau, M, (qc, qdc, qddc) = _command_torques(station.scene, plan.path, t, station.g_w())
    if lags:
        e, ed, edd = _lag(t, tau, M, lim.damping(), qdc)
    else:
        e = ed = edd = np.zeros_like(qc)
    q, qd, qdd = qc - e, qdc - ed, qddc - edd
    R, x, w, wd, a, _ = inner_motion(station.scene, q, qd, qdd)
    return Motion(t, q, qd, qdd, e, R, x, w, wd, a, tau)


# ================================================================ THE BAT
@dataclass
class SimBat:
    err: object                # model.ImuErrors, chip axes
    r: np.ndarray              # m, the sensing centre, clamp frame
    R_ci: np.ndarray           # the chip's axes (columns), clamp frame
    T_amb: float               # degC, the room this day


def draw_bat(d, rig, rng, scale=1.0, seat_rng=None):
    """A bat in the clamp: its part, its chip on its board, its board on its
    stop, itself in the jaws (THE BAT).  seat_rng, if given, draws what a
    re-clamp on another day redraws -- the seat and the room -- so a golden
    bat keeps its part from rng."""
    sr = rng if seat_rng is None else seat_rng
    r0, R0 = P.board_nominal(d, rig)
    pivot = P.seat_pivot(rig)
    tilt = np.radians([Board.CHIP_TILT, Board.CHIP_TILT, Board.CHIP_ROT]) * rng.normal(0.0, 1.0, 3)
    R_chip = R0 @ K.exp_so3(tilt) @ R0.T                       # about the chip's own axes
    slot = np.array([Print.TOL / Board.W, Print.TOL / Board.L, 0.0]) * rng.normal(0.0, 1.0, 3)
    R_slot = K.exp_so3(slot)
    R_seat = K.exp_so3(sr.normal(0.0, radians(Clamp.SEAT_ROT), 3))
    # the chip sits on its board's face, along and across it (clamp x and
    # y, as board_nominal lays the board); the jaws centre the bat across
    # its axis (y and z), its stop holding it along x
    face, across = np.array([1.0, 1.0, 0.0]), np.array([0.0, 1.0, 1.0])
    dp = (rng.normal(0.0, Print.TOL, 3) + rng.normal(0.0, Board.CHIP_POS, 3) * face
          + sr.normal(0.0, Clamp.SEAT_POS, 3) * across
          + np.array([rng.uniform(-Board.L_TOL, Board.L_TOL), 0.0, 0.0])) / 1000.0
    r_board = r0 + dp
    r = pivot + R_seat @ (r_board - pivot)
    R_ci = R_seat @ R_slot @ R_chip @ R0
    T_amb = Site.AMBIENT + sr.normal(0.0, Site.AMBIENT_SD)
    return SimBat(MD.draw_errors(rng, scale), r, R_ci, T_amb)


# ================================================================ TRUTH
@dataclass
class Truth:
    theta: np.ndarray          # (NP,) the fit's parameters as they truly are
    U: np.ndarray              # the chip's axes' rotation in the fit's gauge (fit.mounting of theta)
    bat: SimBat
    pw: object                 # model.PowerUp
    t_first: float
    eps: float


def true_theta(bat, pw, t_first):
    """The fit's parameters (fit.py) as this bat truly is, clamp frame; T0
    the first sample's station time; gravity's direction left at zero (a
    nuisance no term reads)."""
    e = bat.err
    th = np.zeros(F.NP)
    Rt = bat.R_ci.T
    th[F.MA] = ((np.eye(3) + e.Ea) @ Rt).ravel()
    th[F.MG] = ((np.eye(3) + e.Eg) @ Rt).ravel()
    th[F.GS] = (e.G @ Rt).ravel()
    th[F.BA] = e.ba + pw.ba_on
    th[F.BG] = e.bg + pw.bg_on
    th[F.CA] = e.ca
    th[F.CG] = e.cg
    th[F.RR] = bat.r
    th[F.T0] = t_first
    th[F.RHO] = 1.0 + e.eps
    return th


# ================================================================ STATION RUN
def signals(station, mot, bat):
    """(w, f) chip axes at the IMU's sensing centre: the rigid body's own law
    at the bat's point, the Earth's turn added, gravity from the tilted base."""
    w, f = at_point((mot.R, mot.x, mot.w, mot.wd, mot.a), bat.r, station.g_w())
    w = w + np.einsum("nji,j->ni", mot.R, station.earth())
    return w @ bat.R_ci, f @ bat.R_ci


def encoders(station, mot, t_enc):
    """(n, 2) rad: the encoders at station times t_enc -- the hinges as
    MuJoCo turns them, less their zeros (the command's frame), cut to
    whole codes."""
    q = np.stack([np.interp(t_enc, mot.t, mot.q[:, h]) for h in (0, 1)], 1)
    step = 2.0 * pi / 2 ** Gimbal.ENCODER_BITS
    return np.round(q / step) * step


def pose_fixes(station, mot, plan, rng, budget=None):
    """[fit.PoseFix]: the cameras' pose of the clamp at each still, in their
    world, with noise at the rig's budget."""
    from ..rig.place import budget as rig_budget
    deg, mm = rig_budget() if budget is None else budget
    sd = np.array([radians(deg)] * 3 + [mm / 1000.0] * 3)
    cov = np.diag(sd ** 2)
    Rw, tw = station.world
    out = []
    for Q in plan.quiet:
        if Q.kind != "still":
            continue
        tm = 0.5 * (Q.t0 + Q.t1)
        i = int(np.argmin(np.abs(mot.t - tm)))
        R = Rw.T @ mot.R[i]
        x = Rw.T @ (mot.x[i] - tw)
        n = rng.normal(0.0, 1.0, 6) * sd
        out.append(F.PoseFix(tm, K.exp_so3(n[:3]) @ R, x + n[3:], cov.copy()))
    return out


def sample_count(t_end, t_first, eps):
    """The samples the board sends from its first, at station time
    t_first, to t_end, on a clock of fractional error eps: as many whole
    sample periods, (1 + eps) / Imu.ODR each, as the span holds."""
    return int(np.floor((t_end - t_first) * Imu.ODR / (1.0 + eps)))


def run_station(station, mot, plan, bat, rng, budget=None):
    """(fit.CalInputs, Truth): one bat through the plan."""
    t_on = 0.0
    pw = MD.power_up(rng, t_on, bat.T_amb)
    t_first = t_on + Imu.START_S + rng.uniform(0.0, 1.0) / Imu.ODR
    n = sample_count(mot.t[-1], t_first, bat.err.eps)
    w, f = signals(station, mot, bat)
    stream, t_k, _ = MD.sense(bat.err, pw, mot.t, w, f, t_first, n, rng)
    t_enc = np.arange(0.0, mot.t[-1], 1.0 / Gimbal.ENCODER_HZ)
    inp = F.CalInputs(stream=stream, enc_t=t_enc, enc_q=encoders(station, mot, t_enc),
                      poses=pose_fixes(station, mot, plan, rng, budget), quiet=plan.quiet, starts=plan.starts,
                      W=plan.W, t_on=t_on, theta0=plan.theta0, s=plan.s, spread=plan.spread, lag=plan.lag(),
                      path=plan.path)
    th = true_theta(bat, pw, t_first)
    return inp, Truth(th, F.mounting(th), bat, pw, t_first, bat.err.eps)


def nominal_bat(plan):
    """The bat as drawn: a perfect part where the drawing puts it."""
    R = plan.theta0[F.MA].reshape(3, 3).T
    err = MD.ImuErrors(np.zeros((3, 3)), np.zeros((3, 3)), np.zeros(3), np.zeros(3), np.zeros((3, 3)),
                       np.zeros(3), np.zeros(3), 0.0)
    return SimBat(err, plan.theta0[F.RR].copy(), R, Site.AMBIENT)


def internal_rate(station, plan):
    """(f_int, moved) for this plan (model.internal_rate): the highest any
    of the plan's hardest seconds asks for, as the motors run them -- where
    its command's acceleration is largest, and about each hinge's largest
    acceleration step (a corner), where the steppers ring hardest."""
    T = plan.duration_s
    tt = np.linspace(0.0, T, int(T * 200) + 1)
    _, _, qdd = plan.path.state(tt)
    centres = [float(tt[int(np.argmax(np.abs(qdd).max(1)))])]
    for h in (0, 1):
        cs = [c for c in plan.tour.corners if c[1][h] != 0.0]
        if cs:
            centres.append(float(max(cs, key=lambda c: abs(c[1][h]))[0]))
    bat = nominal_bat(plan)
    lg, la = MD.lsb()
    scale = np.array([lg] * 3 + [la] * 3)
    n = int(Imu.ODR * 0.9)
    pw = MD.PowerUp(np.zeros(3), np.zeros(3), -1e6, Site.AMBIENT)
    best = None
    for tc in centres:
        t0 = min(max(tc - 0.5, 0.0), max(T - 1.0, 0.0))

        def probe(fs, t0=t0):
            mot = motion(station, plan, fs, span=(t0, t0 + 1.0))
            w, f = signals(station, mot, bat)
            _, _, ys = MD.sense(bat.err, pw, mot.t, w, f, t0 + 0.05, n, None, noise=False, quantise=False)
            return ys / scale
        got = MD.internal_rate(probe)
        if best is None or got[0] > best[0]:
            best = got
    return best
