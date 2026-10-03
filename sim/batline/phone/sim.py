"""The factory iPhone's test in simulation.  TRUTH: simulation and checks
only; the test (test.py) and the app (app.py) see the HALs and nothing else.

THE GIMBAL follows its commanded paths exactly, on the drawing's
kinematics with its base level: Tier 0.  imu/sim.py holds a gimbal's lag
to its steppers' law in MuJoCo, and check_calibration the fit that sees
through it; here the rig's reference and the bat's motion are one.

THE BAT is imu/sim.draw_bat's: its part, its chip on its board in the
clamp.  The seat's errors ride in the board's place; the bat's own points
-- the bands, the sweet spot -- are the drawing's, each band moved along
the axis by its print and the sleeve's place, uniform within Knit's
tolerances.  The record's frame is then the bat's.

THE IMU: the board's stream for whichever power-up the firmware asks for
(board.sim.Bench's source_for), made a packet at a time by imu/model.sense
from the rigid body's law at the chip (kin.specific_force, the Earth's
turn added), each packet's internal grid run up from before it by the
output filter's memory, at the rate model.internal_rate settles on for the
swing's hardest second.

THE RECORD: what station 4's fit writes for this bat this power-up, for a
bat that met every calibration target -- its true parameters
(imu/sim.true_theta) moved by one draw of the targets
(imu_fusion_sim.CALIBRATED: the mounting turned, the scale and cross-axis,
the biases, the lever and the clock each off by its 1 sigma).
check_calibration holds the fit to those targets on 100 bats; running the
80-second calibration here would test the fit again, not the phone.

THE CAMERA: each frame stamped on the phone's clock (the bench's central's)
at the middle of its first row's exposure; each band seen at its own row's
moment (the rolling shutter) through the commissioned view (camera.View),
its centroid off by Phone.SIGMA_PX on each axis; a band outside the image
is not found.  The centroid is the middle of the band's silhouette across
its image: the sleeve's section over the blade traced on product's own
grid (Shape.mask at product.RES, its boundary cells), put round the
band's centre, every point through the pinhole and the extremes taken
square to the axis's image -- the app's corner rule (app.seen) checked
against a picture of the section, not against itself.
"""
from dataclasses import dataclass
from math import pi
from types import SimpleNamespace

import numpy as np

from ..spec import Imu, Mcu, Link, Phone, Site, Knit, Gimbal, Firmware as FWS
from .. import imu_fusion_sim as study, product as P_
from ..rig import spec as RS
from ..imu import kin as K, model as MD, fit as F, record as R, sim as IS
from ..imu.part import G0, D2R, T_REF, lsb, impulse
from ..board import sim as BS, procedures as PR, power as P
from . import test as T
from .hal import Frame, GimbalHAL, CameraHAL

UP = np.array([0.0, 0.0, 1.0])


# ================================================================ THE GIMBAL
class Timeline:
    """The hinges against true time: at rest at q0 until the first path,
    then each path from the moment it was started (a later one takes over
    from where the last left the gimbal)."""

    def __init__(self, q0):
        self.q0 = np.array(q0, float)
        self.runs = []

    def add(self, t0, path):
        self.runs.append((float(t0), path))

    def still(self, t0, t1):
        """Whether no path runs anywhere in [t0, t1] (a path ends at rest)."""
        return all(t1 < s or t0 > s + path.T for s, path in self.runs)

    def state(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        q = np.tile(self.q0, (len(t), 1))
        qd, qdd = np.zeros_like(q), np.zeros_like(q)
        for t0, path in self.runs:
            m = t >= t0
            if np.any(m):
                q[m], qd[m], qdd[m] = path.state(t[m] - t0)
        return q, qd, qdd


def start_of(path):
    return np.array([ax.q[0] for ax in path.axes])


class SimGimbal(GimbalHAL):
    def __init__(self, bench, timeline):
        self.b = bench
        self.tl = timeline

    def _q(self):
        return self.tl.state(self.b.t())[0][0]

    def run(self, path):
        step = 2.0 * pi / 2 ** Gimbal.ENCODER_BITS
        if np.max(np.abs(start_of(path) - self._q())) > step:
            raise ValueError("a path must start where the gimbal stands")
        self.tl.add(self.b.t(), path)

    def encoders(self):
        step = 2.0 * pi / 2 ** Gimbal.ENCODER_BITS
        return np.round(self._q() / step) * step


# ================================================================ THE IMU
def signals(timeline, kin, bat, T_, g_w, earth):
    """(w, f) chip axes, SI, at true times T_: the rigid body's law at the
    chip, the Earth's turn added."""
    q, qd, qdd = timeline.state(T_)
    Rc, x, wc, wdc, a = kin.motion(q, qd, qdd)
    f = K.specific_force(Rc, a, wc, wdc, bat.r, g_w)
    w = wc + np.einsum("nji,j->ni", Rc, earth)
    return w @ bat.R_ci, f @ bat.R_ci


class MotionSource(BS.Source):
    """The board's IMU in the gimbal this power-up, a packet at a time."""

    def __init__(self, bat, pw, timeline, kin, g_w, earth, rng, t_first, f_int):
        super().__init__(t_first, bat.err.eps)
        self.bat, self.pw, self.tl, self.kin = bat, pw, timeline, kin
        self.g_w, self.earth, self.rng, self.f_int = g_w, earth, rng, f_int

    def codes(self, k0, n):
        ta, tb = float(self.t(k0)), float(self.t(k0 + n - 1))
        lead = len(impulse(self.f_int)) / self.f_int
        # held still, the signal is a constant and any rate past the
        # filter's band gives the same samples: board.sim.StillSource's
        fs = self.f_int if not self.tl.still(ta - lead, tb) else min(self.f_int, BS.StillSource.RATE)
        lead = len(impulse(fs)) / fs
        T_ = np.arange(ta - lead, tb + 2.0 / fs, 1.0 / fs)
        w, f = signals(self.tl, self.kin, self.bat, T_, self.g_w, self.earth)
        st, _, _ = MD.sense(self.bat.err, self.pw, T_, w, f, ta, n, self.rng)
        return np.concatenate([st.acc, st.gyro], axis=1).astype(int), int(st.temp[-1])


_rates = {}


def internal_rate(sp, bat_name):
    """Hz: model.internal_rate over the second about the swing's start,
    where its acceleration steps from rest, on a perfect part where the
    drawing puts it."""
    if bat_name in _rates:
        return _rates[bat_name]
    tl = Timeline(sp.q_a)
    tl.add(0.0, sp.path)
    zero = MD.ImuErrors(np.zeros((3, 3)), np.zeros((3, 3)), np.zeros(3), np.zeros(3), np.zeros((3, 3)),
                        np.zeros(3), np.zeros(3), 0.0)
    bat = IS.SimBat(zero, sp.r_imu, sp.R_imu, Site.AMBIENT)
    pw = MD.PowerUp(np.zeros(3), np.zeros(3), -1e6, Site.AMBIENT)
    lg, la = lsb()
    scale = np.array([lg] * 3 + [la] * 3)
    t0 = max(sp.t_stance - 0.5, 0.0)
    n = int(Imu.ODR * 0.9)
    g_w = -sp.g * UP
    earth = K.earth_rate(Site.LAT, UP, K.north_rig(Site.HEADING))

    def probe(fs):
        T_ = t0 + np.arange(int(fs) + 1) / fs
        w, f = signals(tl, sp.kin, bat, T_, g_w, earth)
        _, _, ys = MD.sense(zero, pw, T_, w, f, t0 + 0.05, n, None, noise=False, quantise=False)
        return ys / scale
    _rates[bat_name] = MD.internal_rate(probe)[0]
    return _rates[bat_name]


# ================================================================ THE RECORD
def record_for(bat, pw, t_first, T_die, rng, serial, grade=study.CALIBRATED):
    """CalRecord of this bat this power-up as a fit that met the targets
    writes it (THE RECORD), clamp frame."""
    th = IS.true_theta(bat, pw, t_first)
    U, Hm = F.polar(th[F.MA].reshape(3, 3).T)
    U2 = U @ K.exp_so3(rng.normal(0.0, grade.mount_deg * D2R, 3))

    def sym(s_scale, s_cross):
        S = np.diag(rng.normal(0.0, s_scale, 3))
        for i, j in F._OFF:
            S[i, j] = S[j, i] = rng.normal(0.0, s_cross)
        return S

    def full(s_scale, s_cross):
        E = np.diag(rng.normal(0.0, s_scale, 3))
        for i, j in F._OFF6:
            E[i, j] = rng.normal(0.0, s_cross)
        return E
    Eg = th[F.MG].reshape(3, 3) @ U - np.eye(3)
    G = th[F.GS].reshape(3, 3) @ U
    th[F.MA] = ((Hm + sym(grade.acc_scale, grade.acc_cross)) @ U2.T).ravel()
    th[F.MG] = ((np.eye(3) + Eg + full(grade.gyro_scale, grade.gyro_cross)) @ U2.T).ravel()
    th[F.GS] = (G @ U2.T).ravel()
    th[F.BA] += rng.normal(0.0, grade.acc_bias * G0, 3)
    th[F.BG] += rng.normal(0.0, grade.gyro_bias * D2R, 3)
    th[F.RR] += rng.normal(0.0, grade.lever_mm / 1000.0, 3)
    th[F.RHO] += rng.normal(0.0, grade.gyro_scale)
    res = SimpleNamespace(theta=th, dT_cal=T_die - T_REF, Sigma_terms=np.diag(F.TAU ** 2), blocks={})
    # the simulation's record rests on no run of the plan: plan hash and rig id zero
    return R.CalRecord.from_result(res, serial, FWS.VERSION, 20730, 4, 0, 0)


def datasheet_record(sp, serial, T_die):
    """The record an app that ignored the bat's would assume: every error
    zero, the chip where the drawing puts it (check_link's must-fail)."""
    zero = MD.ImuErrors(np.zeros((3, 3)), np.zeros((3, 3)), np.zeros(3), np.zeros(3), np.zeros((3, 3)),
                        np.zeros(3), np.zeros(3), 0.0)
    bat = IS.SimBat(zero, sp.r_imu, sp.R_imu, Site.AMBIENT)
    th = IS.true_theta(bat, MD.PowerUp(np.zeros(3), np.zeros(3), 0.0, Site.AMBIENT), 0.0)
    res = SimpleNamespace(theta=th, dT_cal=T_die - T_REF, Sigma_terms=np.diag(F.TAU ** 2), blocks={})
    return R.CalRecord.from_result(res, serial, FWS.VERSION, 20730, 4, 0, 0)


# ================================================================ THE CAMERA
def outline(d):
    """(n, 3) m, clamp frame about the axis (x = 0): the boundary cells of
    the sleeve's section over the blade on product's grid."""
    sh = P_.rtri(d.R, d.offsets[0] + Knit.T)
    y0, y1, z0, z1 = sh.bbox()
    Y, Z = np.meshgrid(np.arange(y0 - 2.0 * P_.RES, y1 + 2.0 * P_.RES, P_.RES),
                       np.arange(z0 - 2.0 * P_.RES, z1 + 2.0 * P_.RES, P_.RES), indexing="ij")
    m = sh.mask(Y, Z)
    inner = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    edge = m.copy()
    edge[1:-1, 1:-1] &= ~inner
    return np.column_stack([np.zeros(edge.sum()), Y[edge], Z[edge]]) / 1000.0


class SimCamera(CameraHAL):
    def __init__(self, bench, timeline, kin, view, bands, rng, rim):
        self.b, self.tl, self.kin, self.view, self.bands, self.rng = bench, timeline, kin, view, bands, rng
        self.rim = rim                         # outline(d)
        self.t0 = None

    def _uv(self, t):
        Rc, x = self.kin.pose(self.tl.state(t)[0])
        c = x[0] + self.bands @ Rc[0].T
        a = self.view.project(c + 1e-3 * Rc[0][:, 0]) - self.view.project(c)
        nrm = np.stack([-a[:, 1], a[:, 0]], 1) / np.linalg.norm(a, axis=1, keepdims=True)
        out = np.empty((2, 2))
        for i in (0, 1):
            uv = self.view.project(c[i] + self.rim @ Rc[0].T)
            mid = self.view.project(c[i])[0]
            s = (uv - mid) @ nrm[i]
            out[i] = mid + 0.5 * (s.min() + s.max()) * nrm[i]
        return out

    def start(self):
        self.t0 = self.b.t()
        self.phase = float(self.rng.uniform(0.0, 1.0 / Phone.FPS))

    def stop(self):
        c = self.b.central
        out = []
        for s in np.arange(c.station(self.t0) + self.phase, c.now(), 1.0 / Phone.FPS):
            t = c.true(s)
            first = self._uv(t)
            uv = np.stack([self._uv(t + first[i, 1] / Phone.H * Phone.READOUT_S)[i] for i in (0, 1)])
            ok = self.view.in_image(uv)
            out.append(Frame(float(s), uv + self.rng.normal(0.0, Phone.SIGMA_PX, (2, 2)) if ok else None))
        self.t0 = None
        return out


# ================================================================ A BAT AT THE TEST
@dataclass
class PhoneBench:
    bench: object              # board.sim.Bench, the phone its central
    station: object            # test.PhoneStation
    db: object                 # imu.record.FactoryDB
    serial: str
    bat: object                # imu.sim.SimBat
    record: object             # the CalRecord station 4 wrote
    bands: np.ndarray          # (2, 3) m, the bands' true centres, clamp frame
    timeline: object
    plan: object               # test.SwingPlan


_rigs = {}
_rims = {}


def ready(d, seed, per=None):
    """A calibrated bat in station 4's gimbal, its record written over the
    station's dongle, the factory iPhone taking over the bench."""
    rng = np.random.default_rng(seed)
    sp = T.slow_swing(d)
    if d.bat not in _rigs:
        _rigs[d.bat] = RS.design(d)
    bat = IS.draw_bat(d, _rigs[d.bat], rng)
    shift = rng.uniform(-Knit.PRINT_TOL, Knit.PRINT_TOL, 2) + rng.uniform(-Knit.PLACE_TOL, Knit.PLACE_TOL)
    bands = sp.geom.bands + np.outer(shift / 1000.0, sp.geom.axis)
    tl = Timeline(np.zeros(2))
    board = BS.draw_board(rng)
    board.err = bat.err
    board.truth.eps = bat.err.eps
    f_int = internal_rate(sp, d.bat)
    g_w = -sp.g * UP
    earth = K.earth_rate(Site.LAT, UP, K.north_rig(Site.HEADING))

    def source_for(t_on):
        pw = MD.power_up(rng, t_on, bat.T_amb)
        t_first = t_on + Imu.START_S + float(rng.uniform(0.0, 1.0)) / Imu.ODR
        return MotionSource(bat, pw, tl, sp.kin, g_w, earth, rng, t_first, f_int)
    b = BS.Bench(board, rng, central="dongle", source_for=source_for)
    serial = "BAT%05d" % seed
    b.supply.source(P.v_fresh(), PR.supply_limit())
    b.clock.wait(1e-3)
    b.swd.connect()
    b.swd.erase_all()
    b.swd.program(0, PR.image())
    b.swd.program(Mcu.UICR_SERIAL, serial.encode().ljust(16, b"\0"))
    b.swd.reset()
    b.clock.wait(Imu.START_S + 2.0 * Link.ADV_JITTER)
    src = b.fw.src_imu
    rec = record_for(bat, src.pw, src.t_first, float(MD.die_temperature(src.pw, b.t())), rng, serial)
    db = R.FactoryDB()
    R.commit(PR.BleLink(b.radio, serial), db, rec, np.eye(F.NT), "simulated")
    b.radio.disconnect()
    b.swap("phone", per)
    if d.bat not in _rims:
        _rims[d.bat] = outline(d)
    st = T.PhoneStation(b.clock, b.radio, SimCamera(b, tl, sp.kin, sp.view, bands, rng, _rims[d.bat]),
                        SimGimbal(b, tl))
    return PhoneBench(b, st, db, serial, bat, rec, bands, tl, sp)


def truth_ss(pb):
    """m, the camera's level frame: the bat's sweet spot where the gimbal
    stands now."""
    sp = pb.plan
    Rc, x = sp.kin.pose(pb.timeline.state(pb.bench.t())[0])
    return sp.view.level(x[0] + Rc[0] @ sp.geom.ss)[0]
