"""The factory iPhone's test at station 4: the slow swing it films, the
limit the fused sweet spot is held to, and the test.  Production.

    geometry(d, rig)     the bat as the app knows it (app.BatGeom)
    slow_swing(d)        SwingPlan: the gimbal's paths, the camera, the times,
                         the turn that catches the most bad records
    GimbalTruth          the swing as imu_fusion_sim.Truth states a swing
    budget(d)            the limit: the fused error's second moment where the
                         swing ends, from the accuracy study's Monte Carlo
    iphone_test(...)     pair, film the swing, fuse, judge
    plan_s(d)            its planned time, for line.py

THE SWING.  The test is there to catch a bat whose record does not fit
it, so its swing is the one that tells such a bat from a good one best.
The gimbal turns the inner hinge rest to rest at its limits
(imu/plan.limits), through a turn centred on the clamp's level pose; the
hinge turns the bat's axis in the clamp's x-z plane, and the camera looks
along y at it from Rules.PLAY_Z through the fold mirror (camera.py), so
the bat is square on to it.  The turn starts at what the limits cover in
the game's own window, the rated swing's backlift and downswing
(imu_fusion_sim.swings(), Rules.RATED_SWING: Tb + Td), and doubles while
that buys power: the accuracy study run on the swing twice, once with the
calibrated part (the budget, THE LIMIT) and once with the same part on its
datasheet alone (UNCAL, the heading off by its mounting), and the power is
the share of the second the budget rejects.  A longer turn widens both:
past a point a bad record's error grows no faster than a good one's
budget.  The doubling stops once the power gains no more than
Module.Z_SIGMAS of the two estimates' binomial sampling, at a whole turn,
or where the bands would leave the phone's image.  An uncalibrated bat
whose errors happen to be small passes, and should: the test judges the
sweet spot, not the record.

THE STANCE.  The study levels over imu_fusion_sim.LEVEL_S and anchors on
STANCE_FRAMES still frames, and the app takes its gyro's rest from the
first LEVEL_S (app.moved); the app's frames must also clear the clock's
error, half an exposure and a readout at either end (app.fuse), and the
anchor stops the output filter's memory short of the swing.  The swing's
end holds that memory and the last packet's journey to the phone (its
host delay at the Z tail).  The clock needs board.sync.listen_s of stream
at the phone's interval to meet Quality.SYNC_S; what that asks beyond the
rest goes to the stance, where the bat is still and it costs nothing.
Before the stance the bat rings on its steppers after the move to it; the
stance starts once the ring moves the farthest band by less than its still
centroid noise (kin.settle).

THE LIMIT.  The accuracy study the product's targets come from, run on
this swing: imu_fusion_sim.run with the calibrated grade of the rated part
(CAL_WIDE), the bat's drawn geometry, gravity's size from each trial's
own stance as the app takes it, the anchor's errors as the app's own
pose solve states them at this camera (app.anchor_cov: the frames'
centroid noise and the bands' print and place; the heading the rig gives
it off by the record's mounting about up, the grade's 1 sigma), no
velocity error (the bat is still) and no clock error (it is still again
where it is judged).  Its
trials' errors across the line of sight, X and Z, give a second moment;
enlarged to the chi-square bound of its own sampling at the Z tail, it is
what a healthy bat's error is drawn from.  The verdict holds the error's
Mahalanobis distance squared to the chi-square on two degrees of freedom
at Quality.TEST_ALPHA, whose quantile is -2 ln(alpha) exactly.  Depth,
along the line of sight, is judged by nothing: the study's finding is that
it only shifts the contact time.

THE REFERENCE is the rig's: the encoders where the swing ends, through the
gimbal's kinematics (the drawing's: check_calibration fits the built one),
put the sweet spot as drawn on the bat in the station's frame, and the
phone's commissioned pose there (camera.View) puts it in the camera's.
The same, at the stance, gives the app the bat's heading (heading), which
the game learns at home and two bands cannot fix (app.py, step 3).
"""
from dataclasses import dataclass, replace
from math import ceil, erfc, floor, log, sqrt

import numpy as np

from ..spec import Imu, Link, Phone, Quality, Rules, Module, Est, Site, Knit
from .. import imu_fusion_sim as study, product
from ..rig import spec as RS
from ..rig.calib import chi2_quantile
from ..imu import kin as K, plan as IP, part as PT, record as R
from ..board import procedures as PR, sync as SY, hal as H
from . import app as A, camera as C

TAIL = 0.5 * erfc(Module.Z_SIGMAS / sqrt(2.0))
BUDGET_TRIALS = 2000           # each Monte Carlo's trials: the sampling bound enlarges the budget 14% at the Z tail; power's binomial sd is 1.1% at most
BUDGET_SEED = 4                # seeded, so the limit is the same every time
UNCAL = replace(study.DATASHEET, name="datasheet, the rated 32 g / 4000 dps part",
                gyro_fs_dps=study.CAL_WIDE.gyro_fs_dps, acc_fs_g=study.CAL_WIDE.acc_fs_g)


# ================================================================ THE BAT AS THE APP KNOWS IT
def geometry(d, rig):
    """app.BatGeom of design d in the clamp's frame (rig.bat_to_body): the
    bands' centres and the sweet spot on the bat's axis, and the corners of
    the sleeve's section over the blade, which the bands are printed on."""
    pts = np.array([[c, 0.0, 0.0] for c, _ in d.bands] + [[d.lay.x_sweet, 0.0, 0.0]])
    p = rig.bat_to_body(pts) / 1000.0
    yz = product.rtri(d.R, d.offsets[0] + Knit.T).corners() / 1000.0
    hull = np.column_stack([np.zeros(len(yz)), yz])
    return A.BatGeom(p[:2].copy(), p[2].copy(), np.array([1.0, 0.0, 0.0]), hull)


def rated_window():
    """s from the stance's anchor to contact in the rated swing."""
    sw = next(s for s in study.swings() if s.name == Rules.RATED_SWING)
    return sw.Tb + sw.Td


def reach(T, w, a):
    """rad a rest-to-rest trapezoid at rate w and acceleration a covers in
    T: truss.motion.trap_time inverted."""
    return a * T * T / 4.0 if T <= 2.0 * w / a else w * (T - w / a)


def stance_spare():
    """s a stance frame keeps from either end of the stance (app.fuse)."""
    return Module.Z_SIGMAS * Quality.SYNC_S + Phone.EXPOSURE_S / 2.0 + Phone.READOUT_S


def last_packet_s():
    """s from a sample to its arrival at the phone, at the Z tail: the rest
    of its packet's samples, an interval's wait, the host's floor and its
    exponential spread's Z-tail quantile, -ln(TAIL) of it."""
    return (H.samples_per_packet() / Imu.ODR + Link.PHONE_CI + Est.HOST_FLOOR_S["phone"]
            - log(TAIL) * Est.HOST_SPREAD_S["phone"])


# ================================================================ THE SWING
@dataclass
class SwingPlan:
    q_a: np.ndarray            # rad, the stance's hinge angles
    q_b: np.ndarray            # and where the swing ends
    to_a: object               # kin.Path: from the gimbal's zeros (where the calibration's tour ends) to q_a
    settle: float              # s, the ring after it
    swing: object              # kin.Path: q_a to q_b
    path: object               # kin.Path: the stance, the swing, the end -- the budget's motion
    t_stance: float
    t_swing: float
    t_end: float
    n_frames: int              # stance frames the app is sure of
    view: object               # camera.View, the station's frame
    kin: object                # kin.Kin, the drawn gimbal
    geom: object               # app.BatGeom
    r_imu: np.ndarray          # m, the IMU as drawn, clamp frame
    R_imu: np.ndarray          # its axes there
    lim: object                # imu.plan.Limits
    g: float                   # m/s^2, the site's
    power: float = float("nan")    # the share of uncalibrated bats the test rejects (THE SWING)
    tried: tuple = ()          # ((turn rad, power), ...) as the design doubled the turn
    stop: str = ""             # why it stopped there

    @property
    def record_s(self):
        return self.t_stance + self.t_swing + self.t_end

    def points(self, q):
        """(3, 3) m, station frame: the two bands and the sweet spot at hinge angles q."""
        Rc, x = self.kin.pose(np.atleast_2d(q))
        return x[0] + np.vstack([self.geom.bands, self.geom.ss]) @ Rc[0].T


_plans = {}
_bases = {}


class _Unfit(ValueError):
    """A turn the station cannot film as THE SWING asks."""


def _base(d, rig):
    """What every turn tried on design d shares: the gimbal's limits, the
    bat as the app knows it, the IMU as drawn, the drawn gimbal, gravity."""
    if d.bat not in _bases:
        rig = RS.design(d) if rig is None else rig
        r0, R0 = IP.board_nominal(d, rig)
        _bases[d.bat] = (IP.limits(d, rig), geometry(d, rig), r0, R0, K.Chain.drawn().kin(),
                         K.gravity(Site.LAT, Site.ALT))
    return _bases[d.bat]


def _plan_for(base, turn):
    """SwingPlan turning the inner hinge through `turn` rad, rest to rest,
    centred on the clamp's level pose (THE SWING, THE STANCE); _Unfit where
    the bands leave the phone's image."""
    lim, geom, r0, R0, kin, g = base
    q_a, q_b = np.array([0.0, -turn / 2.0]), np.array([0.0, turn / 2.0])
    swing = K.Path(q_a).move(q_b, lim.w_max, lim.a_max, tag="swing")
    to_a = K.Path(np.zeros(2)).move(q_a, lim.w_max, lim.a_max, tag="to the stance")
    # the camera, square on to the swing plane, at the span the bands and
    # the sweet spot sweep over the frames it films
    q = swing.state(np.append(np.arange(0.0, swing.T, 1.0 / Phone.FPS), swing.T))[0]
    Rc, x = kin.pose(q)
    pts = (x[:, None, :] + np.einsum("nij,kj->nki", Rc, np.vstack([geom.bands, geom.ss]))).reshape(-1, 3)
    view = C.view_at(0.5 * (pts.min(0) + pts.max(0)), [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
    if not view.in_image(view.project(pts)):
        raise _Unfit("the bands leave the phone's image over a %.0f deg turn" % np.degrees(turn))
    # the ring after the move to the stance, on the inner hinge's stiffness
    r_far = float(np.max(np.linalg.norm(np.vstack([geom.bands, geom.ss]), axis=1)))
    tol = Phone.SIGMA_PX * Rules.PLAY_Z / Phone.F_PX / r_far
    wn = float(lim.w_n(q_a)[1])
    settle = K.settle(2.0 * lim.a_max[1], wn, lim.decay(1, wn) / wn, tol)
    # the stance and the end (THE STANCE)
    mem = len(PT.impulse(Imu.ODR)) / Imu.ODR
    spare = stance_spare()
    t_swing = swing.T
    t_stance = max(study.LEVEL_S, study.STANCE_FRAMES / Phone.FPS + 2.0 * spare) + mem
    t_end = mem + last_packet_s()
    t_stance += max(0.0, SY.listen_s(Link.PHONE_CI, "phone") - (t_stance + t_swing + t_end))
    path = K.Path(q_a).hold(t_stance, "stance").move(q_b, lim.w_max, lim.a_max, "swing").hold(t_end, "end")
    n = int(floor((t_stance - mem - 2.0 * spare) * Phone.FPS))
    return SwingPlan(q_a, q_b, to_a, settle, swing, path, t_stance, t_swing, t_end, n, view, kin, geom, r0, R0,
                     lim, g)


def _errors(sp, truth, grade, heading_sd, N, seed):
    """((N, 2) m, (4, 4)): the accuracy study's errors across the line of
    sight, X and Z, where swing sp ends, with an IMU of `grade` and the
    app's anchor at this camera (app.anchor_cov, the heading off by
    heading_sd rad); and that anchor covariance."""
    p, Rb = bat_pose(sp, sp.q_a)
    cov = A.anchor_cov(p, Rb, sp.geom, sp.view, sp.n_frames, heading_sd)
    anchor = study.Anchor("the factory iPhone's stance", "stance", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    out = study.run(truth, grade, anchor, N=N, seed=seed, l_imu=sp.r_imu, l_ss=sp.geom.ss, r_mount=sp.R_imu,
                    g=sp.g, anchor_cov=cov, stance_g=True)
    return out["e"][:, [0, 2]], cov


def _budget_of(e, cov):
    N = len(e)
    raw = e.T @ e / N
    return Budget(raw * N / chi2_quantile(N, 1.0 - TAIL), raw, N, e.mean(0), cov)


def _power(sp, N, seed):
    """(power, Budget): the share of uncalibrated bats (UNCAL, the heading
    off by its mounting) whose error the swing's budget rejects, and the
    budget, each from N trials."""
    truth = GimbalTruth(sp)
    bud = _budget_of(*_errors(sp, truth, study.CAL_WIDE, study.CAL_WIDE.mount_deg * study.D2R, N, seed))
    e, _ = _errors(sp, truth, UNCAL, UNCAL.mount_deg * study.D2R, N, seed + 1)
    d2 = np.einsum("ni,ij,nj->n", e, np.linalg.inv(bud.S), e)
    return float(np.mean(d2 > bud.limit)), bud


def slow_swing(d, rig=None):
    """SwingPlan for design d, designed once per bat (THE SWING, THE
    STANCE); its budget (THE LIMIT) at the defaults comes with it."""
    if d.bat in _plans:
        return _plans[d.bat]
    base = _base(d, rig)
    lim, N = base[0], BUDGET_TRIALS
    turn = reach(rated_window(), lim.w_max[1], lim.a_max[1])
    sp = _plan_for(base, turn)
    pw, bud = _power(sp, N, BUDGET_SEED)
    tried, stop = [(turn, pw)], "a whole turn"
    while 2.0 * turn <= 2.0 * np.pi:
        try:
            nxt = _plan_for(base, 2.0 * turn)
        except _Unfit as e:
            stop = str(e)
            break
        pw2, bud2 = _power(nxt, N, BUDGET_SEED)
        tried.append((2.0 * turn, pw2))
        se = sqrt((pw * (1.0 - pw) + pw2 * (1.0 - pw2)) / N)
        if pw2 - pw <= Module.Z_SIGMAS * se:
            stop = "doubling it again gains %.3f of power, within %.0f sigma of the trials' %.3f" \
                   % (pw2 - pw, Module.Z_SIGMAS, se)
            break
        sp, pw, bud, turn = nxt, pw2, bud2, 2.0 * turn
    sp.power, sp.tried, sp.stop = pw, tuple(tried), stop
    _plans[d.bat] = sp
    _budgets[(d.bat, N, BUDGET_SEED)] = bud
    return sp


class GimbalTruth:
    """The plan's motion in the camera's level frame as imu_fusion_sim.Truth
    states a swing, for its run(): at each 1 kHz node the IMU's and the
    bat's attitudes and the sweet spot; over each interval the IMU's exact
    turn (the log of its relative rotation, what run's integrator steps by)
    and its mean specific force (the trapezoid over study.SUB sub-steps of
    kin.specific_force, exact on the path)."""

    def __init__(self, sp):
        if abs(study.DT - 1.0 / Imu.ODR) > 1e-15:
            raise ValueError("the study's IMU runs at %.0f Hz, the bat's at %.0f" % (1.0 / study.DT, Imu.ODR))
        n = int(ceil(sp.path.T / study.DT))
        tf = np.arange(n * study.SUB + 1) * study.DT_F
        q, qd, qdd = sp.path.state(tf)
        Rc, x, wc, wdc, a = sp.kin.motion(q, qd, qdd)
        f = K.specific_force(Rc, a, wc, wdc, sp.r_imu, np.array([0.0, 0.0, -sp.g])) @ sp.R_imu
        Rl = np.einsum("ji,njk->nik", sp.view.R, Rc)
        xl = sp.view.level(x)
        ps = xl + np.einsum("nij,j->ni", Rl, sp.geom.ss)
        node = np.arange(n + 1) * study.SUB
        self.Rb = Rl[node]
        self.Ri = self.Rb @ sp.R_imu
        self.ps = ps[node]
        self.vs = np.gradient(ps, study.DT_F, axis=0)[node]
        self.w = K.log_so3(np.swapaxes(self.Ri[:-1], 1, 2) @ self.Ri[1:]) / study.DT
        wts = np.ones(study.SUB + 1)
        wts[0] = wts[-1] = 0.5
        idx = np.arange(n)[:, None] * study.SUB + np.arange(study.SUB + 1)[None, :]
        self.f = np.einsum("ksd,s->kd", f[idx], wts / study.SUB)
        self.t = node * study.DT_F
        # the anchor: the last node whose interval after it is still (run
        # sets the IMU's velocity there from that interval's rate); judged
        # at the first node of the end's rest
        self.k_s = int(floor(sp.t_stance / study.DT)) - 1
        self.k_top = self.k_s
        self.k_c = int(ceil((sp.t_stance + sp.t_swing) / study.DT))


@dataclass
class Budget:
    S: np.ndarray              # (2, 2) m^2: the X, Z error's second moment, enlarged by its sampling bound
    raw: np.ndarray            # (2, 2) m^2: as the trials gave it
    N: int
    mean: np.ndarray           # (2,) m: the trials' mean error
    anchor: np.ndarray         # (4, 4): app.anchor_cov at the stance, the plan's frames

    @property
    def limit(self):
        """The Mahalanobis distance squared a healthy bat stays under but
        for Quality.TEST_ALPHA: chi-square on 2 dof, -2 ln(alpha)."""
        return -2.0 * log(Quality.TEST_ALPHA)


_budgets = {}


def bat_pose(sp, q):
    """(p m, Rb): the bat's frame in the camera's level frame at hinge
    angles q, by the rig's kinematics and the phone's commissioned pose."""
    Rc, x = sp.kin.pose(np.atleast_2d(q))
    return sp.view.level(x)[0], sp.view.R.T @ Rc[0]


def heading(sp, q):
    """rad: the bat's heading about up at hinge angles q, as the app's
    attitude states it (app.heading_of)."""
    return A.heading_of(bat_pose(sp, q)[1])


def budget(d, N=BUDGET_TRIALS, seed=BUDGET_SEED):
    """Budget for design d (THE LIMIT): at the defaults, the one the swing
    was designed with."""
    sp = slow_swing(d)
    key = (d.bat, N, seed)
    if key not in _budgets:
        _budgets[key] = _budget_of(*_errors(sp, GimbalTruth(sp), study.CAL_WIDE,
                                            study.CAL_WIDE.mount_deg * study.D2R, N, seed))
    return _budgets[key]


def judge(ss, ref, bud):
    """(ok, d2, e (2,) m): the fused sweet spot against the rig's across the
    line of sight."""
    e = (np.asarray(ss) - np.asarray(ref))[[0, 2]]
    d2 = float(e @ np.linalg.solve(bud.S, e))
    return d2 <= bud.limit, d2, e


# ================================================================ THE TEST
@dataclass
class PhoneStation:
    """What the factory iPhone's test drives: the phone's clock and
    Bluetooth (board.hal), its camera and the gimbal (phone.hal)."""
    clock: H.ClockHAL
    radio: H.RadioHAL
    camera: object
    gimbal: object


def iphone_test(st, serial, db, d, app_record=None):
    """procedures.Verdict of the factory iPhone's test on the calibrated bat
    in the gimbal: it pairs and reads its record back (procedures.pair),
    films the slow swing with the stream on, fuses (app.fuse) and holds the
    sweet spot to the rig's (judge).  app_record, if given, is the record
    the app fuses with in place of the bat's (check_link's must-fail)."""
    sp = slow_swing(d)
    bud = budget(d)
    run = PR._Run(st)
    ci = Link.PHONE_CI
    try:
        t0 = st.clock.now()
        p = PR.pair(st.clock, st.radio, serial, db)
        run("pairs", p.ok, p.why, t0)
        rec = R.unpack(p.record) if app_record is None else app_record
        t0 = st.clock.now()
        q = st.gimbal.encoders()
        to_a = K.Path(q).move(sp.q_a, sp.lim.w_max, sp.lim.a_max, tag="to the stance")
        st.gimbal.run(to_a)
        st.clock.wait(to_a.T + sp.settle)
        psi = heading(sp, st.gimbal.encoders())
        st.radio.stream(True, PR.timeout(PR.request_s(ci, 0.0, "phone")))
        st.camera.start()
        st.clock.wait(sp.t_stance)
        st.gimbal.run(sp.swing)
        st.clock.wait(sp.t_swing + sp.t_end)
        frames = st.camera.stop()
        raw = st.radio.listen(0.0)
        st.radio.stream(False, PR.timeout(PR.request_s(ci, 0.0, "phone")))
        pk = [H.parse_packet(t, b) for t, b in raw]
        run("films", len(pk) >= 2 and len(frames) > 0, "%d packets and %d frames" % (len(pk), len(frames)), t0)
        t0 = st.clock.now()
        try:
            fix = A.fuse(pk, frames, rec, sp.geom, sp.view, psi, study.CAL_WIDE.mount_deg * study.D2R)
        except (A.FusionError, ValueError) as e:
            run("fuses", False, "the app could not fuse: %s" % e, t0)
        run("fuses", True, "", t0, frames=fix.n_frames)
        Rc, x = sp.kin.pose(np.atleast_2d(st.gimbal.encoders()))
        ref = sp.view.level(x[0] + Rc[0] @ sp.geom.ss)[0]
        ok, d2, e = judge(fix.ss, ref, bud)
        run("sweet spot", ok, "the fused sweet spot lies %.1f mm right and %.1f mm up of the rig's: d^2 %.1f "
            "against %.1f" % (e[0] * 1e3, e[1] * 1e3, d2, bud.limit), t0, d2=d2, e=e, fix=fix, ref=ref)
        st.radio.disconnect()
        return PR.Verdict(True, "", run.steps, serial)
    except PR._Fail as e:
        try:
            st.radio.disconnect()
        except H.LinkError:
            pass
        return PR.Verdict(False, str(e), run.steps, serial)


def plan(d):
    """{step: planned s} of iphone_test on a healthy bat, from the facts and
    the swing's plan: line.py times it by these."""
    sp = slow_swing(d)
    req = PR.request_s(Link.PHONE_CI, 0.0, "phone")
    return {"pairs": PR.pair_plan_s(),
            "films": sp.to_a.T + sp.settle + req + sp.record_s + req,
            "fuses": 0.0}


def plan_s(d):
    return sum(plan(d).values())
