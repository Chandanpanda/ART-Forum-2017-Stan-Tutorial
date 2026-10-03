"""Tier 1: the bat's IMU calibration at station 4 -- the plan, the fit on
simulated bats, what the verdict rejects, the record written to the bat,
the golden bat's daily check of the station, and the line's use of the
plan.

    python3 sim/scripts/batline/check_calibration.py [-v]

The plan's contract for this suite: it runs 100 simulated bats and fails
when the fit misses a planted error by more than its target, or when the
golden-bat check fails to catch a drifted rig.  Every bat has its own part
errors, chip, board, seat and power-up (sim.draw_bat, model.power_up),
drawn at the spreads the plan was designed against; the station is drawn
once per design (its gimbal as built, its base off level, the cameras'
world), and the motion is simulated once per (station, plan) -- the
steppers' lag under every load (sim.motion) -- and shared by its bats.
The kid's bat is the development case; the adult's (check_bat's second
size) is one the plan and the fit were never tuned on.

A. PLAN, each design:  the predicted sigma of every term within its
   threshold, the information of full rank; the plan the same when
   designed twice; the binding limits named; every spin a whole quiet
   turn; the simulated lag within the plan's bound, and its peak N|e|
   (N the electrical radians an axis radian) under pi/2, where a stepper
   slips a step -- a plan whose simulated or bounded lag passes it fails
   here, naming the hinge, and the suite stops, since its bats would run
   a motion the motors cannot make.  Must fail: the plan without its
   spins misses its thresholds.
B. FIT, 100 kid bats and 20 adult:  per-group RMS error within its target
   (the exit test); no term biased; the fit's sigmas cover its errors (each
   bat's NEES, pooled, inside chi-square bounds -- the bounds' ratio is
   the test's resolution, the scale of covariance it can tell from the
   fit's, and is reported); every fit converges in less time than the
   plan takes; the clock the fit starts from (fit.timing), on which it
   first places every window and measures the noise, within the windows'
   margin of the bat's own clock (on M3's first plans it was 70 ms off at
   the adult's first quiet sample and 470 ms at its last, the windows it
   placed straddled moves and read them as noise, and every adult bat was
   refused for sigmas 4.7 times too large; the kid's, 14 and 70 ms off,
   showed nothing).  Must fail: the cross-axis terms judged raw, as planted,
   instead of in the polar gauge (accuracy fails).  OMITTED TERMS: each of the
   gyro's g-sensitivity and the temperature coefficients is dropped from
   the fit on 20 bats; the bias that omission predicts on each bat (the
   fit's own Jacobian, the bat's true value) says whether a target is then
   missed, and the suite holds the omission to that prediction either way;
   each omitted fit's error is the bias its own normal equations predict
   (its own weights, the gimbal's refinement kept) within its own sigma.
C. REJECTION:  healthy bats, kid and adult, are rejected no more often
   than Faults.REJECT["calibration"] allows (a binomial bound).  A stuck gyro
   axis, a noisy accelerometer axis, a part 8 sigma off its scale, a gap
   in the stream inside a used interval, and a stream cut short -- the
   board silent from mid-way through the plan's first quiet interval,
   where nothing can be fitted, or through its last, where the rest can --
   are each rejected, each for its own reason; every fault is placed on
   the bat's own clock, which runs up to Imu.ODR_TOL off the datasheet's
   (seconds of samples by the plan's end).  A gap mid-move, where nothing
   is read, costs nothing: the fit is the gap-free one's.
D. RECORD:  the record packs and unpacks bit-exact; every judged term
   reads back as the fit's within a hundredth of its target, the biases
   moved by the stored coefficients across the die's plausible excursion
   (where a coefficient's unit would show), and the g-sensitivity (G' U,
   also with the chip turned off every axis, where G' U and G' differ),
   the sigmas and the chi-squares as the fit's within a hundredth of what
   each serves; a bat-frame record (to_bat, an arbitrary rigid transform) states
   the mounting and lever in that frame, reads back as the fit's through
   it and refuses to be read without it; every field's float32 resolution
   is within a hundredth of its target; every single-bit flip and every
   truncation is refused, an unknown major version is refused; the
   database refuses an UPDATE, a DELETE, an upsert and an INSERT OR
   REPLACE onto a stored id, the row's bytes and inputs hash unchanged; a
   corrupted bat is restored from the database.  Must fail: a CRC that
   skips the header passes a flipped version; a database with only its
   UPDATE and DELETE triggers lets a REPLACE overwrite a stored row.
E. GOLDEN:  a golden day is the bats' plan run r times, the bat clamped
   once; r is the fewest at which every watched group reaches lambda* on
   the plan's own covariance.  The terms its redraw hides: the seat's
   (spec.Clamp.SEAT_*) left to the cameras' drift check, the power-up's
   (spec.Imu.*_POWERUP) a stated gap, no camera seeing them.  Two golden
   bats certified on Quality.GOLDEN_RUNS days each, each certificate
   setting the runs its bat's days then take; undrifted days stop no
   more often than alpha_day allows, their T^2 follows the law the fit's
   covariances, the certificate and a rig that redraws as the spec says
   predict (a little under chi-square_p: the certificate's scatter is
   floored at the redraw), and each day's runs agree with each other as
   their sigmas say.  The rig fault
   planted is an encoder-scale one: a hinge's encoders read q (1 + eps) of
   a motion that did not change, a fault in what turns their codes into
   angles (a gear-ratio drift would move the axis off its command, the
   encoders on the axis reading it true, and show only as following
   error).  At the smallest size that puts bats out of target it stops
   the line at the power its non-centrality predicts, and each bat's
   following-error test's reach for it is reported; drifts the fit absorbs
   (an encoder's zero moved a degree, the cameras' world turned a degree)
   do not stop it; attribute() tells a drifted rig from an aged bat.  Must
   fail: a test on the biases alone misses the drift.  At twice that size
   on production bats, judged on the bats their verdict passes, the check
   names the layer that catches it: each bat's verdict (past its
   following-error test's reach, every bat refused for following error),
   or the golden day (the bats it passes out of target), or neither, which
   fails.  SYNTHETIC (golden.py's own functions on drawn runs, no fits):
   a day with a run that gave no calibration (its fit failed, its terms
   not numbers, its stream cut short) stops the line, and a commissioning
   day with a failed run refuses the cert; day-to-day scatter planted
   along one watched direction (rho = 2 of the day's measurement there)
   stops the line no more often than alpha_day allows.  Must fail: the
   same scatter left out of the test (Sigma_rep zero) stops it more often.
F. LINE:  the line model's calibration op is the plan's duration and its
   day job the golden day's; the hand-written recipe (Est.STILL_POSES and
   the rest) is gone from spec and line.  Must fail: the scan finds a
   recipe name planted back.
"""
import dataclasses
import os
import pickle
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from math import radians, sqrt, pi
from multiprocessing import get_context
from types import SimpleNamespace

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from truss.glenv import headless        # noqa: E402  (before mujoco)
headless()
import numpy as np                      # noqa: E402

from batline import product                                  # noqa: E402
from batline.spec import KID, Bat, Imu, Quality, Faults, Site, Stepper, Gimbal, Clamp, Line  # noqa: E402
from batline.rig.calib import chi2_quantile                   # noqa: E402
from batline.rig.pose import normal_quantile                  # noqa: E402
from batline.imu import fit as F, plan as P, golden as G      # noqa: E402

VERBOSE = "-v" in sys.argv
RESULTS = []
ADULT = Bat("adult", length=860.0, handle=300.0, grip_d=34.0, width=108.0, depth=62.0)
BATS = {"kid": KID, "adult": ADULT}
N_KID = Quality.CAL_QUAL_BATS          # the qualification's own count
N_ADULT = 20
N_OMIT = 20
N_DAYS = 10                            # golden days: each is r runs of the plan
N_DRIFT = 6
N_ABSORB = 3
ALPHA = Quality.CAL_QUAL_ALPHA         # every statistical bound in the suite
NOISY = 6.0                            # white noise planted on a noisy axis, its datasheet sample sd's: the fit
                                       # measures about 5 times the density (a fault's size, not a threshold)
RHO = 2.0                              # day-to-day scatter planted on synthetic golden days, its variance along
                                       # its direction of the day's measurement's there (a fault's size)
N_SCATTER = 2000                       # synthetic golden certs, a day each: alpha_day's binomial bound is then
                                       # 14 stops, of 5.6 expected
SEED = {"kid": 41, "adult": 42, "synthetic": 43}
HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "..", "..")


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def binom_upper(n, p, alpha):
    """The fewest k with P(Binomial(n, p) > k) <= alpha."""
    from math import comb
    tail = 1.0
    for k in range(n + 1):
        tail -= comb(n, k) * p ** k * (1.0 - p) ** (n - k)
        if tail <= alpha:
            return k
    return n


def binom_lower(n, p, alpha):
    """The most k with P(Binomial(n, p) < k) <= alpha."""
    return n - binom_upper(n, 1.0 - p, alpha)


# ============================================================ WORKERS
# A worker holds, per key, the plan, the station (without its MuJoCo scene:
# the signals need only the motion) and the motion, read from TMP as
# memory maps, so 4 workers share one copy.
_W = {}


def _load(tmp, key):
    if key not in _W:
        from batline.imu import sim as S
        with open(os.path.join(tmp, key + ".pkl"), "rb") as f:
            meta = pickle.load(f)
        arr = {k: np.load(os.path.join(tmp, key, k + ".npy"), mmap_mode="r") for k in meta["fields"]}
        mot = S.Motion(**arr)
        st = S.Station(meta["build"], None, meta["up"], meta["world"], meta["g"])
        _W[key] = (meta, st, mot)
    return _W[key]


def _scene(bat, build):
    from batline.rig import spec as RS
    from batline.imu.scene import Scene
    k = ("scene", bat, id(build))
    if k not in _W:
        d = product.design(BATS[bat])
        _W[k] = Scene(RS.design(d), build)
    return _W[k]


def design_task(which):
    """(which, GoldenPlan, seconds): the bats' plan and its golden day as
    production designs them (golden.day_design)."""
    t = time.time()
    try:
        return which, G.day_design(product.design(BATS[which])), time.time() - t
    except ValueError as e:                        # no golden day can watch it: infeasible, and why
        return which, e, time.time() - t


def torque_task(args):
    """Command torques and mass matrices over one chunk of the grid."""
    from batline.imu import sim as S
    bat, build, plan, t, g_w = args
    scene = _scene(bat, build)
    tau, M, _ = S._command_torques(scene, plan.path, t, g_w)
    return tau, M


def inner_task(args):
    """The clamp's motion from the true hinges over one chunk."""
    from batline.imu.scene import inner_motion
    bat, build, q, qd, qdd = args
    R, x, w, wd, a, _ = inner_motion(_scene(bat, build), q, qd, qdd)
    return R, x, w, wd, a


def _seated(d, rig, seed, seat_seed):
    """A bat whose part, chip and board come from seed and whose seat in
    the jaws (and the room) from seat_seed: a golden bat clamped again."""
    from batline.imu import sim as S
    return S.draw_bat(d, rig, np.random.default_rng(seed), seat_rng=np.random.default_rng(seat_seed))


class _Offset:
    """A commanded path read from a reference turned by dq."""

    def __init__(self, path, dq):
        self.path, self.dq = path, dq

    def state(self, t):
        q, qd, qdd = self.path.state(t)
        return q + self.dq, qd, qdd


def _perturb(inp, mode, arg, truth=None):
    """The inputs as a faulty bat or a drifted rig would give them; a
    fault in the stream is placed on the bat's own clock (truth's T0,
    rho), where the fit will find it, not on the datasheet's."""
    from batline.imu import kin as K
    s = inp.stream

    def span(Q):
        return F.quiet_samples(Q, truth.theta[F.T0], truth.theta[F.RHO], F.Facts.site())

    def drop(cut):
        return dataclasses.replace(inp, stream=dataclasses.replace(
            s, k=np.delete(s.k, cut), acc=np.delete(s.acc, cut, 0), gyro=np.delete(s.gyro, cut, 0),
            temp=np.delete(s.temp, cut), rail=np.delete(s.rail, cut)))
    if mode == "stuck":
        g = s.gyro.copy()
        g[:, 1] = g[0, 1]
        return dataclasses.replace(inp, stream=dataclasses.replace(s, gyro=g))
    if mode == "noisy":
        rng = np.random.default_rng(arg)
        sd = F.code_sd(F.datasheet_nd() ** 2)[3]
        a = s.acc.copy()
        a[:, 0] = np.round(a[:, 0] + rng.normal(0.0, NOISY * sd, len(a))).astype(a.dtype)
        return dataclasses.replace(inp, stream=dataclasses.replace(s, acc=a))
    if mode == "gap":
        # five samples dropped from the middle of the first still's used span
        ka, kb = span([q for q in inp.quiet if q.kind == "still"][0])
        return drop(np.arange(5) + (ka + kb) // 2 - 2)
    if mode == "gap-move":
        # five dropped where nothing reads them (gap_move_at)
        k = int(round((gap_move_at(inp)[0] - truth.theta[F.T0]) * Imu.ODR / truth.theta[F.RHO]))
        return drop(np.arange(5) + k - 2)
    if mode in ("cut-first", "cut-last"):
        # the board falls silent mid-way through the plan's first quiet
        # interval's used span (nothing after it: no fit) or its last's
        # (all but the end)
        Q = min(inp.quiet, key=lambda q: q.t0) if mode == "cut-first" else max(inp.quiet, key=lambda q: q.t1)
        n = sum(span(Q)) // 2
        return dataclasses.replace(inp, stream=dataclasses.replace(
            s, k=s.k[:n], acc=s.acc[:n], gyro=s.gyro[:n], temp=s.temp[:n], rail=s.rail[:n]))
    if mode == "enc_scale":
        # hinge h's encoders read q (1 + eps) of a motion that did not
        # change: a fault in what turns their codes into angles.  Not a
        # gear-ratio drift: that moves the axis off its command, the
        # encoder on the axis reading it true, and shows only as following
        # error, since the fit takes its kinematics from the encoders
        h, eps = arg
        q = inp.enc_q.copy()
        q[:, h] *= 1.0 + eps
        return dataclasses.replace(inp, enc_q=q)
    if mode == "zero":
        # the homing's reference moved a degree: the encoders, which define
        # it, and the command both read the hinge a degree off where it is
        dq = np.zeros(2)
        dq[arg] = radians(1.0)
        return dataclasses.replace(inp, enc_q=inp.enc_q + dq, path=_Offset(inp.path, dq))
    if mode == "world":
        Rr = K.exp_so3(np.radians([0.6, -0.5, 0.6]))          # a degree, off every axis
        return dataclasses.replace(inp, poses=[F.PoseFix(p.t, Rr @ p.R, Rr @ p.x, p.cov) for p in inp.poses])
    return inp


def gap_move_at(inp):
    """(t, margin s): the station time furthest inside a move from anything
    the fit reads -- the quiet intervals either side, and each start's
    window in timing() (fit.timing reads the gyro from 0.2 s before a start
    to 0.4 s after) -- and how far that is from the nearest of them."""
    Qs = sorted(inp.quiet, key=lambda q: q.t0)
    best = (None, -np.inf)
    for a, b in zip(Qs[:-1], Qs[1:]):
        lo, hi = a.t1, b.t0
        for t in np.linspace(lo, hi, 201)[1:-1]:
            m = min([t - lo, hi - t] + [t - (ts + 0.4) if t > ts else ts - 0.2 - t for ts in inp.starts])
            if m > best[1]:
                best = (float(t), float(m))
    return best


def follow_reach(inp, h):
    """The least encoder-scale drift eps on hinge h (_perturb "enc_scale")
    that fit.verdict's following-error test refuses on these inputs: where
    fit.fit's follow, max |command - encoders| over both hinges, passes the
    plan's lag.  follow is convex in eps (the largest of |d - eps q|), so
    the drifts it allows are an interval about zero, and bisection finds
    its end.  The encoders are the station's, the same for every bat: past
    it the test refuses every bat, short of it none."""
    d = inp.path.state(inp.enc_t)[0] - inp.enc_q
    other = float(np.max(np.abs(d[:, 1 - h])))
    qh, dh = inp.enc_q[:, h], d[:, h]

    def follow(eps):
        return max(other, float(np.max(np.abs(dh - eps * qh))))
    if follow(0.0) > inp.lag:
        return 0.0
    # where |q| is largest, eps |q| - |d| passes the lag by hi
    lo, hi = 0.0, 2.0 * (inp.lag + float(np.max(np.abs(dh)))) / float(np.max(np.abs(qh)))
    while hi - lo > 1e-6 * hi:
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if follow(m) <= inp.lag else (lo, m)
    return hi


def omission_bias(res, truth, cols):
    """(NT,) the bias in the judged terms that leaving cols out of the fit
    res made predicts for this bat: the kept parameters -- the gimbal's
    refinement eta among them -- take up J_o theta_o through res's own
    normal equations, at its own weights (omitted-variable bias), and the
    judged terms lose what theta_o put in them directly (a bias read at the
    die's mean temperature loses c dT), theta_o the bat's true value."""
    th = res.theta
    model = res.model
    model.dk = res.dk
    wh = 1.0 / np.sqrt(res.obs.var)
    L = np.linalg.cholesky(res.Sigma_kin)
    J = np.concatenate([(model.jacobian(th) * wh[:, :, None]).reshape(-1, F.NP),
                        ((model.kin_jacobian(th) @ L) * wh[:, :, None]).reshape(-1, 14)], 1)
    keep = np.r_[np.setdiff1d(np.arange(F.NP), cols), F.NP + np.arange(14)]
    _, psig = F.priors(th[F.T0])
    P_ = np.r_[np.where(np.isfinite(psig), 1.0 / np.where(np.isfinite(psig), psig, 1.0) ** 2, 0.0), np.ones(14)]
    Hk = J[:, keep].T @ J[:, keep] + np.diag(P_[keep])
    A = np.linalg.solve(Hk, J[:, keep].T @ J[:, cols])
    Tj = np.concatenate([F.terms_jacobian(th, res.dT_cal), np.zeros((F.NT, 14))], 1)
    return Tj[:, keep] @ (A @ truth.theta[cols]) - Tj[:, cols] @ truth.theta[cols]


def bat_task(args):
    """One bat through the station and the fit: a small dict.  kind "bat"
    draws a production bat from seed; kind "golden" is run seed[2] of
    golden bat seed[0] on day seed[1]: the day draws its seat and its room,
    the run its power-up and its noise."""
    tmp, key, seed, kind, mode, arg = args
    from batline.imu import sim as S, judge as J
    from batline.rig import spec as RS
    meta, st, mot = _load(tmp, key)
    plan = meta["plan"]
    d = product.design(BATS[meta["bat"]])
    rig = RS.design(d)
    if kind == "golden":
        bat = _seated(d, rig, seed[0], seed[1])
        rng = np.random.default_rng([seed[1], seed[2], 7919])
        if mode == "aged":
            bat.err.Eg[np.diag_indices(3)] += arg
    else:
        rng = np.random.default_rng(seed)
        bat = S.draw_bat(d, rig, rng)
        if mode == "scale8":
            bat.err.Ea[0, 0] = 8.0 * Imu.ACC_SCALE
    inp, tr = S.run_station(st, mot, plan, bat, rng)
    reach = follow_reach(inp, arg[0]) if mode == "enc_scale" else None
    inp = _perturb(inp, mode, arg, tr)
    use = {"gsens": False} if mode == "no-gsens" else {"tempco": False} if mode == "no-tempco" else None
    t = time.time()
    res = F.fit(inp, use=use)
    tf = time.time() - t
    v = F.verdict(res, inp)
    out = dict(seed=seed, kind=kind, mode=mode, ok=v.ok, why=v.why, conv=res.converged, iters=res.iters,
               fit_s=tf, e=J.errors(res, tr), tt=F.terms(tr.theta, res.dT_cal, tr.U),
               sd=np.sqrt(np.diag(res.Sigma_terms)), Sigma_terms=res.Sigma_terms, s=inp.s,
               terms=res.terms, blocks=res.blocks, short=res.short, follow=res.follow, lag=inp.lag,
               nd=res.nd / F.datasheet_nd(),
               Ea=bat.err.Ea.copy(), Eg=bat.err.Eg.copy())
    if mode == "healthy" and arg == "omit":
        out["omit"] = {name: omission_bias(res, tr, cols) for name, cols in OMIT.items()}
    if mode in OMIT:
        out["B"] = omission_bias(res, tr, OMIT[mode])
    if kind == "golden":
        out["res"] = _slim(res)
    if arg == "record":
        out["res"] = res
    if reach is not None:
        out["reach"] = reach
    if mode == "gap-move":
        out["gap_margin"] = gap_move_at(inp)[1]
    if mode == "healthy":
        out["clock0"] = clock_error(inp, tr)
    return out


def clock_error(inp, tr):
    """(s, pairs): how far fit.timing's clock -- where the fit's starts,
    and where it first places every window and measures the noise --
    puts the plan's quiet samples from where the bat's own clock does, at
    the worst of the first's and the last's ends (the error is a line in
    the sample number), and how many starts it timed."""
    facts = F.Facts.site()
    T0_, rho, n = F.timing(inp, facts, F.stage1(inp)[0])
    T0t, rhot = tr.theta[F.T0], tr.theta[F.RHO]
    ks = [k for Q in inp.quiet for k in F.quiet_samples(Q, T0t, rhot, facts)]
    err = [abs((T0_ - T0t) + (rho - rhot) * k / Imu.ODR) for k in (min(ks), max(ks))]
    return max(err), n


OMIT = {"no-gsens": np.arange(F.GS.start, F.GS.stop),
        "no-tempco": np.r_[np.arange(F.CA.start, F.CA.stop), np.arange(F.CG.start, F.CG.stop)]}


def _slim(res):
    """What golden.py reads of a CalResult."""
    return dataclasses.replace(res, obs=None, model=None)


# ============================================================ PREPARE
def prepare(pool, tmp, which, plan, seed):
    """Draw the station, find f_int, simulate the motion in chunks across
    the pool (the lag serially between), and leave it in tmp for the
    workers; return the meta.  A plan the motors cannot run -- its own lag
    bound (CalPlan.lag) or the simulated lag (sim._lag, the stepper's sine
    law) past the pull-out angle -- gives no motion: the meta then holds
    only the plan and why ("lost")."""
    from batline.imu import sim as S
    from batline.rig import spec as RS
    bat = "adult" if which == "adult" else "kid"
    try:
        plan.lag()
    except ValueError as err:
        return dict(plan=plan, bat=bat, lost="the plan's lag bound: %s" % err)
    d = product.design(BATS[bat])
    rig = RS.design(d)
    st = S.Station.draw(rig, np.random.default_rng(seed))
    fi = S.internal_rate(st, plan)[0]
    T = plan.duration_s
    t = np.arange(int(np.ceil(T * fi)) + 1) / fi
    chunks = np.array_split(np.arange(len(t)), 16)
    tm = pool.map(torque_task, [(bat, st.build, plan, t[c], st.g_w()) for c in chunks])
    tau = np.concatenate([x[0] for x in tm])
    M = np.concatenate([x[1] for x in tm])
    qc, qdc, qddc = plan.path.state(t)
    try:
        e, ed, edd = S._lag(t, tau, M, plan.lim.damping(), qdc)
    except ValueError as err:
        return dict(plan=plan, bat=bat, lost="the simulated lag: %s" % err)
    q, qd, qdd = qc - e, qdc - ed, qddc - edd
    im = pool.map(inner_task, [(bat, st.build, q[c], qd[c], qdd[c]) for c in chunks])
    R, x, w, wd, a = (np.concatenate([m[i] for m in im]) for i in range(5))
    fields = dict(t=t, q=q, qd=qd, qdd=qdd, lag=e, R=R, x=x, w=w, wd=wd, a=a, tau=tau)
    os.makedirs(os.path.join(tmp, which))
    for k, v in fields.items():
        np.save(os.path.join(tmp, which, k + ".npy"), v)
    meta = dict(plan=plan, bat=bat, build=st.build, up=st.up, world=st.world, g=st.g, f_int=fi,
                fields=list(fields), max_lag=float(np.abs(e).max()),
                peak_Ne=Stepper.TEETH * Gimbal.GEAR * np.abs(e).max(0))
    with open(os.path.join(tmp, which + ".pkl"), "wb") as f:
        pickle.dump(meta, f)
    return meta


# ============================================================ A. PLAN
def lost_step(label, meta):
    """The simulated peak N|e| of each hinge under pi/2, where the stepper's
    torque, T sin(N e), is at its most and past which the rotor slips a
    step -- the simulation's own law, whatever the plan's bound says; a
    plan prepare() found the motors cannot run fails here, for its reason."""
    if "lost" in meta:
        return check("%s: the simulated peak N|e| stays under pi/2, where a stepper slips a step" % label, False,
                     meta["lost"])
    pk = np.asarray(meta["peak_Ne"], float)
    return check("%s: the simulated peak N|e| stays under pi/2, where a stepper slips a step (%.2f, %.2f rad)"
                 % (label, pk[0], pk[1]), np.all(pk < pi / 2.0),
                 "%.0f%% and %.0f%% of pi/2" % tuple(100.0 * pk / (pi / 2.0)))


def plan_checks(label, plan, meta, twin=None):
    sb = F.thresholds()
    check("%s: every term's predicted sigma within its threshold, each no looser than the bats' (V %.3g)"
          % (label, plan.V), plan.V <= 0.0 and np.all(plan.sigma <= plan.s * (1.0 + 1e-9))
          and np.all(plan.s <= sb * (1.0 + 1e-12)),
          "worst sigma/s %.3f; %d thresholds held tighter for the golden bat, to %.3f of the bats' at least"
          % (float(np.max(plan.sigma / plan.s)), int(np.sum(plan.s < sb * (1.0 - 1e-12))), float(np.min(plan.s / sb))))
    St = P.terms_cov(plan)
    ok = St is not None and np.all(np.isfinite(St)) and np.linalg.eigvalsh(St).min() > 0.0
    check("%s: the plan's information is of full rank" % label, ok)
    if twin is not None:
        check("%s: designed twice, the same plan (%s, %s)" % (label, plan.hash, twin.hash), plan.hash == twin.hash)
    b = plan.binding()
    check("%s: the binding limits are named" % label, all(b["rate"]) and all(b["acceleration"]),
          "; ".join("%s: %s" % (k, ", ".join(v)) for k, v in b.items()))
    if "lost" not in meta:                         # a lost plan has no simulated lag, and maybe no bound
        check("%s: the simulated lag stays within the plan's bound (%.3f of %.3f deg)"
              % (label, np.degrees(meta["max_lag"]), np.degrees(plan.lag())), meta["max_lag"] <= plan.lag())
    lost_step(label, meta)
    # must fail: without its spins the plan cannot see the gyro's scale
    items = [i for i in plan.items if not isinstance(i, P.Spin)]
    tour = P.build(items, plan.lim, plan.W)
    info = P._exact_info(tour, plan.theta0, P.K.Chain.drawn().kin(), plan.W)
    sg = P.Value(plan.theta0, plan.s, plan.spread, weak=False).sigma(info)
    check("must fail (%s): the plan without its spins misses its thresholds" % label, np.any(~(sg <= plan.s)),
          "%d terms over" % int(np.sum(~(sg <= plan.s))))


# ============================================================ B. FIT
def group_rms(E):
    return np.array([sqrt(float(np.mean(E[:, F.GROUP_OF == k] ** 2))) for k in range(len(F.GROUPS))])


def fit_checks(label, runs, n_expect):
    E = np.array([r["e"] for r in runs])
    SD = np.array([r["sd"] for r in runs])
    n = len(runs)
    check("%s: %d of %d fits converged" % (label, sum(r["conv"] for r in runs), n_expect),
          n == n_expect and all(r["conv"] for r in runs))
    rms = group_rms(E)
    tau = np.array([g[2] for g in F.GROUPS])
    check("%s: every group's RMS error within its target, %d bats" % (label, n), np.all(rms <= tau),
          ", ".join("%s %.2f" % (g[0], r / t) for g, r, t in zip(F.GROUPS, rms, tau)) + " of target")
    zb = normal_quantile(1.0 - ALPHA / (2.0 * F.NT))
    bias = np.abs(E.mean(0)) / np.sqrt(np.mean(SD ** 2, 0) / n)
    check("%s: no term biased (|mean error| within %.1f of its standard error)" % (label, zb), np.all(bias <= zb),
          "worst %.2f, term %d" % (float(bias.max()), int(np.argmax(bias))))
    # the test's resolution: a covariance scaled by c scales NEES by 1/c,
    # so the data accept the fit's scaled by NEES/hi .. NEES/lo and refuse
    # any covariance off by more than hi/lo, wherever NEES lies
    nees = np.array([r["e"] @ np.linalg.solve(r["Sigma_terms"], r["e"]) for r in runs])
    lo, hi = chi2_quantile(n * F.NT, 1.0 - ALPHA / 2.0), chi2_quantile(n * F.NT, ALPHA / 2.0)
    check("%s: the fit's covariance covers its errors (pooled NEES %.0f on %d dof, in [%.0f, %.0f])"
          % (label, nees.sum(), n * F.NT, lo, hi), lo <= nees.sum() <= hi,
          "the test resolves a covariance %.2f times off: it accepts the fit's scaled by %.2f .. %.2f"
          % (hi / lo, nees.sum() / hi, nees.sum() / lo))
    slow = max(r["fit_s"] for r in runs)
    return E, SD, nees, (lo, hi), slow


def raw_cross_errors(runs):
    """Cross-axis errors against the cross-axis terms as planted -- the raw
    off-diagonals of each part's maps -- not as the polar gauge reads them
    (the gauge puts an unsymmetric term's antisymmetric half into the
    mounting)."""
    out = []
    for r in runs:
        fitted = r["e"] + r["tt"]
        e = r["e"].copy()
        e[6:9] = fitted[6:9] - np.array([r["Ea"][i, j] for i, j in F._OFF])
        e[15:21] = fitted[15:21] - np.array([r["Eg"][i, j] for i, j in F._OFF6])
        out.append(e)
    return np.array(out)


def omission_checks(base, by, tau):
    """Each omission against its own prediction: the healthy fits' errors
    plus the bias leaving the term out predicts (omission_bias on the full
    fit, the bat's true value) say which groups miss their targets; where
    the prediction is clear of the target by the RMS's own sampling spread,
    the omitted fits must agree.  And each omitted fit's error is the bias
    its own normal equations predict (omission_bias on the omitted fit, at
    its own weights, which move with its own predictions) plus noise its
    own covariance states: each group's sum of squared residuals over the
    omitted fits' sigmas is chi-square on its count, both tails."""
    Eh = np.array([r["e"] for r in base])
    n = len(base)
    z = normal_quantile(1.0 - ALPHA / (2.0 * len(F.GROUPS)))
    for name in OMIT:
        runs = by[("kid", "bat", name)]
        Eo = np.array([r["e"] for r in runs])
        B = np.array([r["omit"][name] for r in base])
        pred, obs = group_rms(Eh + B), group_rms(Eo)
        nu = n * np.array([g[1] for g in F.GROUPS])
        spread = pred * z / np.sqrt(2.0 * nu)
        clear = np.abs(pred - tau) > spread
        agree = np.all((pred[clear] > tau[clear]) == (obs[clear] > tau[clear]))
        miss = [g[0] for g, m in zip(F.GROUPS, pred > tau) if m]
        check("omitting %s: the groups it predicts out of target (%s) are the ones that miss"
              % (name[3:], ", ".join(miss) or "none"), agree,
              ", ".join("%s %.2f/%.2f" % (g[0], p_ / t, o / t) for g, p_, o, t in zip(F.GROUPS, pred, obs, tau)))
        Bo = np.array([r["B"] for r in runs])
        SDo = np.array([r["sd"] for r in runs])
        Z2 = ((Eo - Bo) / SDo) ** 2
        q = np.array([float(np.sum(Z2[:, F.GROUP_OF == k])) for k in range(len(F.GROUPS))])
        dof = len(runs) * np.array([g[1] for g in F.GROUPS])
        a_ = ALPHA / (2.0 * len(F.GROUPS))
        lo = np.array([chi2_quantile(d, 1.0 - a_) for d in dof])
        hi = np.array([chi2_quantile(d, a_) for d in dof])
        k = int(np.argmax(np.maximum(q / hi, lo / q)))
        check("omitting %s: each omitted fit's error is the bias its own fit predicts, within its own sigma"
              % name[3:], np.all((q >= lo) & (q <= hi)), "worst %s, %.0f on %d (in [%.0f, %.0f]); RMS %.2f sigma"
              % (F.GROUPS[k][0], q[k], dof[k], lo[k], hi[k], sqrt(q[k] / dof[k])))


def rejection_checks(by):
    reason = {"stuck": ("a stuck gyro axis", "stuck"),
              "noisy": ("a noisy accelerometer axis (%.1f times its datasheet's noise, as measured)"
                        % by[("kid", "bat", "noisy")][0]["nd"][3], "sigma"),
              "scale8": ("a part 8 sigma off its accelerometer scale", "of its spread from nominal"),
              "gap": ("a gap in the stream inside a still", "gaps in the stream"),
              "cut-first": ("a stream cut short in the plan's first quiet interval", "cut short"),
              "cut-last": ("a stream cut short in the plan's last quiet interval", "cut short")}
    for m, (what, key) in reason.items():
        r = by[("kid", "bat", m)][0]
        check("%s is rejected, for its reason (\"%s\")" % (what, key),
              not r["ok"] and any(key in w for w in r["why"]), "; ".join(r["why"][:3]) or "accepted")
    # the same bat as the first healthy one, five samples dropped mid-move:
    # the fit reads the stream by its counter, so nothing it uses moves
    g = by[("kid", "bat", "gap-move")][0]
    h = [r for r in by[("kid", "bat", "healthy")] if r["seed"] == g["seed"]][0]
    dz = float(np.max(np.abs(g["terms"] - h["terms"]) / h["sd"]))
    check("a gap in the stream mid-move, where nothing is read, costs nothing: the fit is the gap-free one's",
          g["ok"] == h["ok"] and dz <= 1e-6, "largest change %.2g sigma, %.2f s from anything read; %s"
          % (dz, g["gap_margin"], "; ".join(g["why"][:2]) or "accepted"))


def record_checks(check, res, plan):
    """D. The record (imu/record.py): what is written to the bat and kept in
    the factory's database, for one real fit (res) of plan."""
    import datetime
    import sqlite3
    import zlib
    from batline.imu import record as RC, fit as F, kin as K
    from batline.imu.part import G0, D2R

    def flip(b, bits):
        out = bytearray(b)
        for k in bits:
            out[k // 8] ^= 1 << (k % 8)
        return bytes(out)

    def refused(b, **kw):
        try:
            RC.unpack(b, **kw)
        except RC.RecordError as e:
            return str(e)
        return ""

    def resign(b, at, value):
        """b with byte `at` set to value and its CRC made good again."""
        out = bytearray(b)
        out[at] = value
        return bytes(out[:RC.CRC_AT]) + RC.crc(out).to_bytes(4, "little")

    def reason(r):
        return next((k for k in ("magic", "major", "length", "CRC") if k in r), r or "passed")

    # the plan's hash, as the record stores it
    try:
        ph = plan.hash
        ok_h, why = isinstance(ph, str) and len(ph) == 8 and int(ph, 16) >= 0, ph
    except Exception as e:                             # noqa: BLE001 -- a broken hash is this check's finding
        ph, ok_h, why = "00000000", False, "%s: %s" % (type(e).__name__, e)
    check("record: the plan's hash reads as the record's 8 hex digits", ok_h, why)
    serial = "KID-000001"
    rec = RC.CalRecord.from_result(res, serial, firmware=0x010000, date=datetime.date(2026, 10, 2), station=4,
                                   rig_id=1, plan_hash=ph)
    b = RC.pack(rec)
    back = RC.unpack(b)
    same = len(b) == RC.SIZE and RC.pack(back) == b and np.array_equal(back.values().view(np.uint32),
                                                                        rec.values().view(np.uint32))
    check("record: packs to %d bytes and unpacks bit for bit" % RC.SIZE, same,
          "%d bytes, %d value fields, %d bytes of pad" % (len(b), RC.NF, RC.PAD))

    # every judged term, as stored, against the fit's: within 1 % of its
    # target (the design's rule).  The mounting is read from the drawing's
    # axes, so it is a real rotation, not res.terms' zero.  The biases are
    # moved by the stored coefficients from the stored T_cal to the fit's
    # dT_cal, and to either end of the die's plausible excursion about it
    # (accept_z of the room's day-to-day spread, and the die's self-heating,
    # as resolution() bounds T_cal): at dT_cal alone the move is T_cal's
    # rounding, and a coefficient in the wrong unit, or none, would pass.
    # The g-sensitivity is no judged term: it is held, as the fit's G' in
    # the chip's axes (G' U, the record's table), to its datasheet spread
    # / 100, resolution()'s tolerance for it
    U_ref = F.mounting(plan.theta0)
    zx = RC.accept_z() * Site.AMBIENT_SD + Imu.SELF_HEAT
    e = np.max([np.abs(back.judged_terms(U_ref, dT) - F.terms(res.theta, dT, U_ref)) / (F.TAU / 100.0)
                for dT in (res.dT_cal - zx, res.dT_cal, res.dT_cal + zx)], 0)
    j = int(np.argmax(e))
    Gw = (res.theta[F.GS].reshape(3, 3) @ F.mounting(res.theta)).ravel() * G0 / D2R
    eG = float(np.max(np.abs(back.gyro_G.astype(float) - Gw))) / (Imu.GYRO_GSENS / 100.0)
    # ... and G' U, not G', where the two differ: the drawing's chip axes
    # may be the clamp's (U near I), so the same fit with its chip turned
    # off every axis (its accelerometer map turned, which turns U)
    Rq = K.exp_so3(np.random.default_rng(5).normal(0.0, 1.0, 3))
    th_q = res.theta.copy()
    th_q[F.MA] = (res.theta[F.MA].reshape(3, 3) @ Rq.T).ravel()
    back_q = RC.unpack(RC.pack(RC.CalRecord.from_result(dataclasses.replace(res, theta=th_q), serial,
                                                        firmware=0x010000, date=datetime.date(2026, 10, 2),
                                                        station=4, rig_id=1, plan_hash=ph)))
    Gq = (th_q[F.GS].reshape(3, 3) @ F.mounting(th_q)).ravel() * G0 / D2R
    eG = max(eG, float(np.max(np.abs(back_q.gyro_G.astype(float) - Gq))) / (Imu.GYRO_GSENS / 100.0))
    check("record: every judged term reads back within 1%% of its target (%d terms), the biases across the die's "
          "excursion (%.1f degC either side of the fit's), and the g-sensitivity (G' U, its chip turned too) within "
          "1%% of its spread" % (F.NT, zx), e.max() <= 1.0 and eG <= 1.0,
          "worst %s %d at %.2g of tau/100; g-sensitivity at %.2g of its spread/100"
          % (F.GROUPS[F.GROUP_OF[j]][0], j, e[j], eG))

    # the sigmas and the chi-squares as the fit gave them, each within 1 %
    # of what it serves (resolution()'s tolerances: the group's target, the
    # statistic's own spread over healthy bats); the clock's coefficient
    # zero until one is measured (the record's table)
    unit = np.array([RC.PPM if g[3] == "" else 1.0 for g in F.GROUPS])
    sd = np.sqrt(np.diag(res.Sigma_terms))
    sg = np.array([sd[F.GROUP_OF == k].max() for k in range(len(F.GROUPS))]) * unit
    es = float(np.max(np.abs(back.sigma.astype(float) - sg) / (np.array([g[2] for g in F.GROUPS]) * unit / 100.0)))
    ec, nan_ok = 0.0, True
    for i, b_ in enumerate(RC.BLOCKS):
        c2, dof = res.blocks.get(b_, (np.nan, 0))
        if dof > 0:
            ec = max(ec, abs(float(back.chi2[i]) - c2 / dof) / (sqrt(2.0 / dof) / 100.0))
        else:
            nan_ok = nan_ok and bool(np.isnan(back.chi2[i]))
    check("record: the stored sigmas and chi-squares are the fit's, within 1%% of what each serves",
          es <= 1.0 and ec <= 1.0 and nan_ok and float(back.rho_tc[0]) == 0.0,
          "sigmas at %.2g, chi-squares at %.2g of their tolerance; a missing block NaN: %s; rho_tc %g"
          % (es, ec, nan_ok, float(back.rho_tc[0])))

    # a bat-frame record: the mounting and the lever stated through to_bat
    # (R, p mm), the clamp's frame into the bat's.  No bat frame exists yet
    # (the record's NOT YET), so an arbitrary rigid transform: a turn off
    # every axis and an offset the lever's own size.  Stored, they are the
    # fit's turned into that frame; read back through it, the fit's terms;
    # read without it, refused
    rng = np.random.default_rng(4)
    r_mm = res.theta[F.RR] * 1000.0
    tb = (K.exp_so3(rng.normal(0.0, 1.0, 3)), rng.normal(0.0, 1.0, 3) * float(np.linalg.norm(r_mm)))
    bat_rec = RC.unpack(RC.pack(RC.CalRecord.from_result(res, serial, firmware=0x010000,
                                                        date=datetime.date(2026, 10, 2), station=4, rig_id=1,
                                                        plan_hash=ph, to_bat=tb)))
    gi = {g[0]: k for k, g in enumerate(F.GROUPS)}
    eb = np.abs(bat_rec.judged_terms(U_ref, res.dT_cal, to_bat=tb) - F.terms(res.theta, res.dT_cal, U_ref)) \
        / (F.TAU / 100.0)
    turn = float(np.degrees(np.linalg.norm(K.log_so3(RC.rot_of(bat_rec.quat) @ (tb[0] @ F.mounting(res.theta)).T))))
    shift = float(np.max(np.abs(bat_rec.lever.astype(float) - (tb[0] @ r_mm + tb[1]))))
    blind = []
    for name, read in (("rotation", bat_rec.rotation), ("lever_clamp", bat_rec.lever_clamp)):
        try:
            read()
            blind.append(name)
        except RC.RecordError:
            pass
    bad_b = [r.field for r in RC.resolution(plan, res, tb).values() if not r.ok]
    check("record: a bat-frame record states the mounting and lever in the bat's frame, reads back as the fit's "
          "through to_bat, and is refused without it",
          bat_rec.frame == RC.FRAME_BAT and eb.max() <= 1.0 and turn <= F.GROUPS[gi["mount"]][2] / 100.0
          and shift <= F.GROUPS[gi["lever"]][2] / 100.0 and not blind and not bad_b,
          "frame %d; terms at %.2g of tau/100; stated %.2g deg and %.2g mm off R U and R r + p; read without "
          "to_bat: %s; resolution over: %s" % (bat_rec.frame, eb.max(), turn, shift, blind or "refused",
                                              bad_b or "none"))

    # each field's step -- float32's ULP at its largest plausible value,
    # T_cal's count -- against 1 % of the target it serves
    rows = list(RC.resolution(plan, res).values())
    bad = ["%s %.2g" % (r.field, r.ulp / r.tol) for r in rows if not r.ok]
    w = max(rows, key=lambda r: r.ulp / r.tol)
    check("record: every field's step at its largest plausible value is within 1%% of its target "
          "(%d float32 fields and T_cal)" % sum(r.kind == "f32" for r in rows), not bad,
          "worst %s at %.2g of its tolerance (%s)%s" % (w.field, w.ulp / w.tol, w.basis,
                                                        "; over: %s" % bad if bad else ""))

    # every single-bit flip, every truncation
    why = {}
    missed = []
    for k in range(8 * RC.SIZE):
        r = refused(flip(b, [k]))
        if not r:
            missed.append(k)
        why[reason(r)] = why.get(reason(r), 0) + 1
    check("record: every one of the %d single-bit flips is refused" % (8 * RC.SIZE), not missed,
          "missed bits %s; refused for %s" % (missed[:8], why))
    short = [n for n in range(RC.SIZE) if not refused(b[:n])]
    check("record: every truncation (0..%d bytes) and an overlong record are refused" % (RC.SIZE - 1),
          not short and refused(b + b"\0"), "passed: %s" % short[:8])

    # versions: an unknown major is refused even with its CRC made good; a
    # newer minor unpacks, its appended field kept and written back as read
    maj = [refused(resign(b, 4, v)) for v in (RC.MAJOR - 1, RC.MAJOR + 1)]
    check("record: an unknown major version is refused, its CRC made good", all("major" in m for m in maj),
          "; ".join(maj))
    newer = bytearray(b)
    newer[5] = RC.MINOR + 1
    newer[RC.PAD_AT:RC.PAD_AT + 4] = np.float32(770.0).tobytes()          # a later minor's first field
    newer = bytes(newer[:RC.CRC_AT]) + RC.crc(newer).to_bytes(4, "little")
    r = refused(newer)
    ok = not r
    if ok:
        nr = RC.unpack(newer)
        ok = (nr.minor == RC.MINOR + 1 and np.array_equal(nr.values().view(np.uint32), rec.values().view(np.uint32))
              and RC.pack(nr) == newer)
    check("record: a newer minor version unpacks, its appended field kept", ok, r or "minor %d" % (RC.MINOR + 1))

    # the factory's database: history, newest last, the covariance bit for bit
    db = RC.FactoryDB(":memory:")
    h0 = "0" * 64
    rec2 = RC.CalRecord(**{**vars(rec), "date": rec.date + 1})          # the same bat, recalibrated a day later
    i1 = db.store(serial, b, res.Sigma, h0)
    i2 = db.store(serial, RC.pack(rec2), res.Sigma, h0)
    hist = db.history(serial)
    check("record: the database keeps every store with its covariance, newest last",
          i2 > i1 and [x.id for x in hist] == [i1, i2] and db.latest(serial) == RC.pack(rec2)
          and hist[0].Sigma.dtype == np.float64 and np.array_equal(hist[0].Sigma, res.Sigma),
          "ids %s, Sigma %s" % ([x.id for x in hist], hist[0].Sigma.shape))

    # no statement writes over a stored row: an UPDATE, a DELETE, an upsert
    # (which is an UPDATE) and an INSERT OR REPLACE (which deletes the row
    # without firing a delete trigger) onto its id, each refused by the
    # database itself, the row then as stored
    def overwrite(dbx, i):
        """({way: refused}, row i as stored (b, h0), the serial's ids) after
        each way of writing another record over row i of dbx; rolled back."""
        row = (i, serial, RC.pack(rec2), np.eye(2).tobytes(), "2,2", "f" * 64, "then")
        ins = ("INSERT%s INTO records (id, serial, record, sigma, sigma_shape, inputs_hash, stored_at) "
               "VALUES (?, ?, ?, ?, ?, ?, ?)")
        ways = (("UPDATE", "UPDATE records SET record = ?, inputs_hash = ? WHERE id = ?", (row[2], row[5], i)),
                ("DELETE", "DELETE FROM records WHERE id = ?", (i,)),
                ("upsert", ins % "" + " ON CONFLICT (id) DO UPDATE SET record = excluded.record, "
                 "inputs_hash = excluded.inputs_hash", row),
                ("INSERT OR REPLACE", ins % " OR REPLACE", row))
        out = {}
        for way, sql, args in ways:
            try:
                dbx.con.execute(sql, args)
                out[way] = False
            except sqlite3.DatabaseError:
                out[way] = True
        rows = dbx.history(serial)
        got = [x for x in rows if x.id == i]
        same = len(got) == 1 and got[0].record == b and got[0].inputs_hash == h0
        dbx.con.rollback()
        return out, same, [x.id for x in rows]

    tried, same, ids = overwrite(db, i1)
    check("record: the database refuses an UPDATE, a DELETE, an upsert and an INSERT OR REPLACE onto a stored id, "
          "the row's bytes and inputs hash as stored", all(tried.values()) and same and ids == [i1, i2],
          "accepted: %s; row %d as stored: %s; ids %s"
          % ([k for k, v in tried.items() if not v] or "none", i1, same, ids))

    # commit: one bad write is rewritten; two refuse and store nothing
    link = RC.MemoryLink()
    link.corrupt([8 * RC.FLOAT_AT + 3])                              # one bit of the accel bias
    w1 = RC.commit(link, db, rec, res.Sigma, h0)
    n_before = len(db.history(serial))
    link.corrupt([8 * RC.FLOAT_AT + 3], writes=2)
    twice = ""
    try:
        RC.commit(link, db, rec, res.Sigma, h0)
    except RC.RecordError as e_:
        twice = str(e_)
    check("record: commit rewrites a bat whose read-back differs, and refuses one that differs twice, storing "
          "nothing", w1.writes == 2 and w1.bad_bits == [1, 0] and twice and len(db.history(serial)) == n_before,
          "first: %d writes %s; second: %s" % (w1.writes, w1.bad_bits, twice or "accepted"))

    # the database restores a corrupted bat
    link.flash = b
    link.damage([8 * 100 + 5, 8 * 300 + 1])
    broken = refused(link.read())
    wr = RC.restore(link, db, serial)
    check("record: the database restores a corrupted bat", broken and link.read() == db.latest(serial)
          and not refused(link.read()), "the bat's record refused (%s); restored in %d write(s)" % (broken, wr.writes))

    # must fail: a CRC that skips the 8-byte header passes a flipped version
    # byte -- the minor, which nothing else checks -- so the real one covers it
    def skip(body):
        return zlib.crc32(bytes(body[RC.HEAD.size:RC.CRC_AT])) & 0xFFFFFFFF
    bs = RC.pack(rec, crc_of=skip)
    passed = [k for k in range(8 * RC.HEAD.size) if not refused(flip(bs, [k]), crc_of=skip)]
    caught = all(refused(flip(b, [k])) for k in range(8 * RC.HEAD.size))
    check("must fail: a CRC that skips the %d-byte header passes a flipped version byte" % RC.HEAD.size,
          bool(passed) and caught and all(k // 8 == 5 for k in passed),
          "header bits it passes: %s (byte 5, the minor version); the record's CRC refuses all %d"
          % (passed, 8 * RC.HEAD.size))
    db.close()

    # must fail: a database with only its UPDATE and DELETE triggers -- the
    # schema without records_fresh, in a file as the line keeps one -- lets
    # an INSERT OR REPLACE write over a stored row
    before = re.sub(r"\s*CREATE TRIGGER IF NOT EXISTS records_fresh\b.*?END;", "", RC.FactoryDB.SCHEMA, flags=re.S)

    class _Before(RC.FactoryDB):
        SCHEMA = before
    with tempfile.TemporaryDirectory() as td:
        odb = _Before(os.path.join(td, "factory.db"))
        j1 = odb.store(serial, b, res.Sigma, h0)
        tried_o, same_o, _ = overwrite(odb, j1)
        odb.close()
    check("must fail: a database with only its UPDATE and DELETE triggers lets a REPLACE overwrite a stored row",
          before != RC.FactoryDB.SCHEMA and tried_o["UPDATE"] and tried_o["DELETE"]
          and not tried_o["INSERT OR REPLACE"] and not same_o,
          "refused: %s; accepted: %s; the row as stored afterwards: %s"
          % ([k for k, v in tried_o.items() if v] or "none", [k for k, v in tried_o.items() if not v] or "none",
             same_o))


def golden_jobs(tmp, gb, day, r, mode="golden", arg=None, first=0):
    """A golden day: r runs of the plan, golden bat gb clamped once (from
    run `first`: the runs a day already run lacks)."""
    return [(tmp, "kid", (gb, day, n), "golden", mode, arg) for n in range(first, r)]


def days_of(runs):
    """{(bat, day): [CalResult of each run, in order]}."""
    out = {}
    for r in sorted(runs, key=lambda r: r["seed"]):
        out.setdefault(r["seed"][:2], []).append(r["res"])
    return out


def redraw_terms(d, rig):
    """(seat, power-up): (NT,) the terms whose redraw (golden.repeatability)
    the seat's spec (Clamp.SEAT_*) and the power-up's (Imu.*_POWERUP)
    carry -- found by setting each source to zero and seeing whose redraw
    it took, not by naming groups."""
    base = G.repeatability(d, rig)
    out = []
    for cls, names in ((Clamp, [n for n in vars(Clamp) if n.startswith("SEAT_")]),
                       (Imu, [n for n in vars(Imu) if n.endswith("_POWERUP")])):
        kept = {n: getattr(cls, n) for n in names}
        try:
            for n in names:
                setattr(cls, n, 0.0)
            out.append(G.repeatability(d, rig) < base)
        finally:
            for n, v in kept.items():
                setattr(cls, n, v)
    return tuple(out)


def _groups_of(mask):
    """The groups (NT,) mask touches, in fit.GROUPS' order."""
    return ", ".join(g[0] for k, g in enumerate(F.GROUPS) if np.any(mask & (F.GROUP_OF == k))) or "none"


def t2_law(cert, Sigmas, rep):
    """(mean, c, f): the law of the sum of T^2 over healthy days against
    cert, the days' measurements Sigmas (each its Day.Sigma) and the rig
    redrawing by rep and no more -- c chi-square_f, the two moments matched
    (Satterthwaite).  Each day's deviation from the certified mean carries
    its measurement, its redraw, and the certified mean's error, which
    every day shares: Cov(d_i, d_j) = (Sigma_c + R) / K off the diagonal,
    Sigma_i + R + that on it; T^2 sums d_i' C_i^-1 d_i, C_i the test's
    covariance (golden.test_cov), so its mean is tr(B S) and its variance
    2 tr((B S)^2), B the C_i^-1 down the diagonal and S the d's stacked."""
    R = np.diag(rep ** 2)
    p, n = len(rep), len(Sigmas)
    common = (cert.Sigma_c + R) / cert.K
    S = np.kron(np.ones((n, n)), common)
    B = np.zeros((n * p, n * p))
    for i, Si in enumerate(Sigmas):
        b = slice(i * p, (i + 1) * p)
        S[b, b] += Si + R
        B[b, b] = np.linalg.inv(G.test_cov(cert, Si))
    BS = B @ S
    m, v = float(np.trace(BS)), 2.0 * float(np.sum(BS * BS.T))
    return m, v / (2.0 * m), 2.0 * m * m / v


def golden_synthetic(gp, prod):
    """E, synthetic: golden.py's own functions (combine, certify, daily) on
    drawn runs, no fits -- each run's watched terms drawn about its day's
    from the plan's own covariance, as the fit's sigmas say they scatter.
    A day with a run that gave no calibration (its fit failed, its terms
    not numbers, its stream cut short) stops the line, and a commissioning
    day with one refuses the cert.  Then N_SCATTER certs, each on K days
    and judging one more, the days scattering by the spec's redraw
    (gp.rep) and by RHO times the day's own measurement along one watched
    direction: certify's scatter, measured from the days alone, keeps the
    stops within alpha_day's binomial bound; must fail: the same days
    tested without it (Sigma_rep zero) stop more often.  There a day's r runs, of
    one covariance, are drawn as the one run they combine to (the plan's
    covariance over r), which certify and daily cannot tell apart, so the
    certs cost what as many runs do."""
    K = Quality.GOLDEN_RUNS
    r_ = gp.repeats
    St = P.terms_cov(gp.plan)
    idx = np.flatnonzero(gp.watched)
    p = len(idx)
    a = G.alpha_day()
    rng = np.random.default_rng(SEED["synthetic"])
    Lr = np.linalg.cholesky(St[np.ix_(idx, idx)])

    def run(at, Sigma, **how):
        t = np.zeros(F.NT)
        t[idx] = at
        return SimpleNamespace(**{"terms": t, "Sigma_terms": Sigma, "failed": "", "lost": 0, **how})

    def day(shift):
        return [run(shift + Lr @ rng.normal(size=p), St) for _ in range(r_)]

    # a run that gave no calibration, as fit.fit returns each, in place of
    # a healthy one
    nan_t, nan_S = np.full(F.NT, np.nan), np.full((F.NT, F.NT), np.nan)
    bad = {"a failed fit": lambda r: dict(terms=nan_t, Sigma_terms=nan_S,
                                          failed="no spin inside the stream: the lever arm is unobserved"),
           "terms not numbers": lambda r: dict(terms=np.where(np.arange(F.NT) == idx[0], np.nan, r.terms)),
           "a stream cut short": lambda r: dict(lost=1)}

    def spoil(r, what):
        return SimpleNamespace(**{**vars(r), **bad[what](r)})
    healthy = [day(rng.normal(size=p) * gp.rep[idx]) for _ in range(K + 1)]
    cert = G.certify(healthy[:K], gp, prod)
    V = {}
    for what in bad:
        runs = list(healthy[K])
        runs[r_ // 2] = spoil(runs[r_ // 2], what)
        V[what] = G.daily(cert, runs)
    check("golden (synthetic): a day with a run that gave no calibration stops the line (%s)" % ", ".join(V),
          all(v.stop and v.T2 == np.inf for v in V.values()),
          "; ".join("%s: T^2 %g, %s" % (w_, v.T2, v.worst_group) for w_, v in V.items())
          + "; the same day whole: T^2 %.1f" % G.daily(cert, healthy[K]).T2)
    days = [list(x) for x in healthy[:K]]
    days[K // 2][0] = spoil(days[K // 2][0], "a failed fit")
    try:
        G.certify(days, gp, prod)
        why = ""
    except ValueError as e_:
        why = str(e_)
    check("golden (synthetic): a commissioning day with a failed run refuses the cert", "commissioning day" in why,
          why or "certified")

    # day-to-day scatter along one watched direction, RHO times the day's
    # measurement there
    Ld = np.linalg.cholesky(St[np.ix_(idx, idx)] / r_)
    u = rng.normal(size=p)
    v = Ld @ (u / np.linalg.norm(u))
    St_day = St / r_
    stops = stops0 = 0
    for _ in range(N_SCATTER):
        shift = sqrt(RHO) * rng.normal(size=(K + 1, 1)) * v + rng.normal(size=(K + 1, p)) * gp.rep[idx]
        days = [[run(x, St_day)] for x in shift[:K] + rng.normal(size=(K, p)) @ Ld.T]
        c = G.certify(days, gp, prod)
        # the day judged is run as its certificate says
        n_ = c.repeats if c.feasible else r_
        Sj = St[np.ix_(idx, idx)] / n_
        judged = [run(shift[K] + np.linalg.cholesky(Sj) @ rng.normal(size=p), St / n_)]
        stops += G.daily(c, judged).stop
        stops0 += G.daily(dataclasses.replace(c, Sigma_rep=np.zeros_like(c.Sigma_rep)), judged).stop
    lim = binom_upper(N_SCATTER, a, ALPHA)
    check("golden (synthetic): day-to-day scatter %g times the day's measurement along a watched direction, and the "
          "redraw, stops %d of %d days, at most %d at alpha_day %.4f" % (RHO, stops, N_SCATTER, lim, a), stops <= lim)
    check("must fail (synthetic): the same scatter left out of the test (Sigma_rep zero) stops %d of %d days, more "
          "than %d" % (stops0, N_SCATTER, lim), stops0 > lim)


def drift_layer(runs, eps, tau):
    """(layer or None, detail): which layer catches a rig drift of size eps
    on these production bats (bat_task's dicts, each with its following-
    error test's reach for the drift, follow_reach).  Each bat's verdict,
    when eps is past every bat's reach and every bat is refused for
    following error; else the golden day, when the bats the verdict passes
    are out of target, and the reach says which bats the test refused;
    else neither."""
    reach = np.array([r["reach"] for r in runs])
    past = eps > reach
    fe = np.array([r["follow"] > r["lag"] for r in runs])
    acc = [r for r in runs if r["ok"]]
    rms = group_rms(np.array([r["e"] for r in acc])) if acc else np.zeros(len(tau))
    if np.all(past) and np.all(fe) and not acc:
        layer = "each bat's verdict, every bat refused for following error"
    elif acc and np.any(rms > tau) and np.array_equal(past, fe):
        layer = "the golden day: the %d bats the verdict passes are out of target" % len(acc)
    else:
        layer = None
    return layer, ("%d of %d pass the verdict, %d refused for following error, whose reach is %.2g..%.2g; the "
                   "passed bats: %s" % (len(acc), len(runs), int(fe.sum()), reach.min(), reach.max(),
                                        ", ".join("%s %.2f" % (g[0], r / t) for g, r, t in zip(F.GROUPS, rms, tau))
                                        + " of target" if acc else "none"))


def golden_checks(pool, tmp, gp, prod, by, GB, step):
    tau = np.array([g[2] for g in F.GROUPS])
    groups = np.array([g[0] for g in F.GROUPS])
    r_ = gp.repeats
    St = P.terms_cov(gp.plan)
    lam_pred = G.group_lambdas(St / r_, prod, gp.watched, gp.rep)
    w = ~np.isnan(lam_pred)
    check("golden: its day, the plan %d times (%.0f s), reaches lambda* %.1f in every watched group (%s)"
          % (r_, gp.duration_s, gp.lam, ", ".join(groups[w])), np.all(lam_pred[w] >= gp.lam * (1.0 - 1e-9)),
          ", ".join("%s %.0f" % (g, l) for g, l in zip(groups[w], lam_pred[w]))
          + "; day_design weighed %s" % ", ".join("a %.1f s plan %d times (the station's day %.0f s)"
                                                   % (T_, n_, (Line().per_day + n_) * T_) for T_, n_ in gp.considered))
    if r_ > 1:
        lam_less = G.group_lambdas(St / (r_ - 1), prod, gp.watched, gp.rep)
        check("golden: %d runs are the fewest (with %d, %s falls to %.1f)"
              % (r_, r_ - 1, groups[int(np.nanargmin(lam_less))], float(np.nanmin(lam_less))),
              np.nanmin(lam_less) < gp.lam)
    # the terms its redraw hides, by where their redraw comes from: the
    # seat's are the cameras' to watch, the power-up's no camera sees
    from batline.rig import spec as RS
    d = product.design(KID)
    seat, pu = redraw_terms(d, RS.design(d))
    hidden = ~gp.watched
    check("golden: the terms its redraw hides are the seat's and the power-up's; the seat's (%s) are left to the "
          "cameras' drift check" % _groups_of(hidden & seat), not np.any(hidden & ~seat & ~pu),
          "hidden by neither: %s" % _groups_of(hidden & ~seat & ~pu))
    check("golden: a stated gap, unwatched and seen by no camera: the power-up's terms (%s)"
          % _groups_of(hidden & pu & ~seat), True,
          "their power-up redraw hides the smallest drift that matters (golden.watched), and the cameras' drift "
          "check sees the seat, not the part")
    D = days_of(by[("golden", "golden", "golden")])
    K = Quality.GOLDEN_RUNS
    try:
        certs = [G.certify([D[(gb, 5000 + 100 * j + k)] for k in range(K)], gp, prod) for j, gb in enumerate(GB)]
    except ValueError as e_:                       # a commissioning run that gave no calibration: no cert
        check("golden: both bats certified on %d days each" % K, False, str(e_))
        return
    check("golden: both bats certified on %d days each, every watched group at lambda* with their scatter, at the "
          "runs a day each certificate sets (%s)" % (K, ", ".join(str(c.repeats) for c in certs)),
          all(c.feasible for c in certs),
          "; ".join("Q %.0f on %d dof, min lambda %.0f" % (c.Q, gp.p * (K - 1), np.nanmin(c.lambdas))
                    for c in certs) + "; the plan's own %d" % r_)
    # each bat's days from here on are run as its certificate says; the
    # first bat's undrifted and absorbed days were run at the plan's r_
    # with round 1, and get the runs they lack
    R0, R1 = (c.repeats if c.feasible else r_ for c in certs)
    if R0 > r_:
        extra = [j for k in range(N_DAYS) for j in golden_jobs(tmp, GB[0], 6000 + k, R0, first=r_)]
        extra += [j for i, (m, arg) in enumerate((("zero", 0), ("world", None))) for k in range(N_ABSORB)
                  for j in golden_jobs(tmp, GB[0], 6500 + 10 * i + k, R0, m, arg, first=r_)]
        for (_, _, _, kind, mode, _), r in zip(extra, pool.map(bat_task, extra, chunksize=1)):
            by[("golden", kind, mode)].append(r)
        D = days_of(by[("golden", "golden", "golden")])
    a = G.alpha_day()
    V = [G.daily(certs[0], D[(GB[0], 6000 + k)][:R0]) for k in range(N_DAYS)]
    stops = sum(v.stop for v in V)
    check("golden: %d of %d undrifted days stop the line, at most %d at alpha_day %.4f"
          % (stops, len(V), binom_upper(len(V), a, ALPHA), a), stops <= binom_upper(len(V), a, ALPHA))
    T2 = sum(v.T2 for v in V)
    m_, c_, f_ = t2_law(certs[0], [v.day.Sigma for v in V], gp.rep[np.flatnonzero(gp.watched)])
    lo, hi = c_ * chi2_quantile(f_, 1.0 - ALPHA / 2.0), c_ * chi2_quantile(f_, ALPHA / 2.0)
    check("golden: undrifted, T^2 follows the law its covariances predict on a rig that redraws as the spec says "
          "(sum %.0f, predicted %.0f, in [%.0f, %.0f])" % (T2, m_, lo, hi), lo <= T2 <= hi,
          "%.2f of chi-square_%d's mean a day, predicted %.2f: the certificate's scatter, floored at the redraw, "
          "is above what a rig that redraws by no more shows" % (T2 / (len(V) * gp.p), gp.p, m_ / (len(V) * gp.p)))
    Q = sum(v.day.Q for v in V if v.day is not None)        # a day of no day stopped the line above
    dq = sum(v.day.n - 1 for v in V if v.day is not None) * gp.p
    if dq > 0:                                     # one run a day has no siblings to agree with
        lo, hi = chi2_quantile(dq, 1.0 - ALPHA / 2.0), chi2_quantile(dq, ALPHA / 2.0)
        check("golden: each day's runs agree with each other as their sigmas say (Q %.0f on %d dof, in [%.0f, %.0f])"
              % (Q, dq, lo, hi), lo <= Q <= hi)
    for m, what in (("zero", "an encoder's zero moved a degree"), ("world", "the cameras' world turned a degree")):
        Dm = days_of(by[("golden", "golden", m)])
        V = [G.daily(certs[0], runs[:R0]) for runs in Dm.values()]
        k = sum(v.stop for v in V)
        check("golden: %s, which the fit absorbs, stops %d of %d days (at most %d)"
              % (what, k, len(V), binom_upper(len(V), a, ALPHA)), k <= binom_upper(len(V), a, ALPHA),
              "T^2 " + ", ".join("%.0f" % v.T2 for v in V))

    # the encoder-scale drift: its direction in the terms (one run with and
    # without a drift of the gyro scale's target, the same noise: the
    # response is linear), and the smallest size that puts a group of
    # production bats out of target
    D2 = G.margins(prod)
    eps, crit, resp = {}, {}, {}
    for h in (0, 1):
        base = [r for r in by[("golden", "golden", "direction")] if r["seed"] == (GB[0], 6900 + h, 0)][0]
        drift = [r for r in by[("golden", "golden", "enc_scale")] if r["seed"] == (GB[0], 6900 + h, 0)][0]
        v = (drift["terms"] - base["terms"]) / step
        v[F.GROUP_OF == 6] = 0.0          # the mounting is read about each run's own (fit.terms)
        n2 = np.array([float(np.sum(v[F.GROUP_OF == k] ** 2)) for k in range(len(F.GROUPS))])
        e_g = np.where(n2 > 0.0, np.sqrt(D2 / np.where(n2 > 0.0, n2, 1.0)), np.inf)
        eps[h], crit[h], resp[h] = float(e_g.min()), int(np.argmin(e_g)), v
    jobs = [j for h in (0, 1) for k in range(N_DRIFT)
            for j in golden_jobs(tmp, GB[0], 7000 + 100 * h + k, R0, "enc_scale", (h, eps[h]))]
    jobs += golden_jobs(tmp, GB[1], 7500, R1, "enc_scale", (0, eps[0]))
    jobs += golden_jobs(tmp, GB[1], 7501, R1)
    age = sqrt(D2[4] / 3.0)                 # the gyro's scale, each axis: the group's margin
    jobs += golden_jobs(tmp, GB[0], 7502, R0, "aged", age)
    h_mf = max((0, 1), key=lambda h: eps[h])
    jobs += [(tmp, "kid", 1000 + i, "bat", "enc_scale", (h_mf, 2.0 * eps[h_mf])) for i in range(N_OMIT)]
    out = pool.map(bat_task, jobs, chunksize=1)
    rb = {}
    for (_, key, seed, kind, mode, arg), r in zip(jobs, out):
        rb.setdefault((kind, mode, arg[0] if mode == "enc_scale" else None), []).append(r)
    thr = chi2_quantile(gp.p, a)
    idx = np.flatnonzero(gp.watched)
    bmask = np.isin(F.GROUP_OF, [0, 3])
    # the biases as certified: the same commissioning days, read over the biases
    bc = [G.combine(D[(GB[0], 5000 + k)], bmask) for k in range(K)]
    b_cert = np.mean([x.terms for x in bc], 0)
    b_rep = gp.rep[bmask]
    for h in (0, 1):
        runs_h = [r for r in rb[("golden", "enc_scale", h)] if r["seed"][0] == GB[0]]
        Dh = days_of(runs_h)
        V = [G.daily(certs[0], runs) for runs in Dh.values()]
        # the drift's non-centrality through the test's own covariance
        # (golden.test_cov), over the days that gave one
        Sd = [v_.day.Sigma for v_ in V if v_.day is not None]
        b = (resp[h] * eps[h])[idx]
        lam = float(b @ np.linalg.solve(G.test_cov(certs[0], np.mean(Sd, 0)), b)) if Sd else float("nan")
        pw = G.noncentral_sf(thr, gp.p, lam) if Sd else 1.0
        k = sum(v_.stop for v_ in V)
        # each run's following-error test: the least drift of this kind it
        # refuses (follow_reach), the station's encoders the same for every run
        reach = np.array([r["reach"] for r in runs_h])
        check("golden: an encoder-scale drift on hinge %d of %.2g, the least that puts %s out of target, has lambda "
              "%.0f (lambda* %.0f)" % (h, eps[h], groups[crit[h]], lam, gp.lam), lam >= gp.lam,
              "the following-error test's reach for it %.2g..%.2g over the day's runs: at %.2g times that it "
              "refuses %s" % (reach.min(), reach.max(), eps[h] / reach.min() if reach.min() > 0.0 else np.inf,
                              "every bat" if eps[h] > reach.max() else
                              "no bat" if eps[h] <= reach.min() else "some bats"))
        check("golden: ... and stops %d of %d days, at least %d at its power %.3f"
              % (k, len(V), binom_lower(len(V), pw, ALPHA), pw), k >= binom_lower(len(V), pw, ALPHA),
              "T^2 " + ", ".join("%.0f" % v_.T2 for v_ in V) + "; worst groups " +
              ", ".join(sorted(set(v_.worst_group for v_ in V))))
        # must fail: the biases alone do not see it (a day that gives no
        # day stops the line, as daily does)
        kb = 0
        for runs in Dh.values():
            try:
                x = G.combine(runs, bmask)
            except ValueError:
                kb += 1
                continue
            d_ = x.terms - b_cert
            Cb = x.Sigma * (1.0 + 1.0 / K) + np.diag(b_rep ** 2)
            kb += float(d_ @ np.linalg.solve(Cb, d_)) > chi2_quantile(int(bmask.sum()), a)
        check("must fail: a test on the biases alone misses the hinge-%d encoder-scale drift (%d of %d days)"
              % (h, kb, len(Dh)), kb <= binom_upper(len(Dh), a, ALPHA))
    g1 = days_of([r for r in rb[("golden", "enc_scale", 0)] if r["seed"][0] == GB[1]])
    g0 = days_of([r for r in rb[("golden", "enc_scale", 0)] if r["seed"][0] == GB[0]])
    v_rig = (G.daily(certs[0], g0[(GB[0], 7000)]), G.daily(certs[1], g1[(GB[1], 7500)]))
    v_bat = (G.daily(certs[0], days_of(rb[("golden", "aged", None)])[(GB[0], 7502)]),
             G.daily(certs[1], days_of(rb[("golden", "golden", None)])[(GB[1], 7501)]))
    check("golden: attribute() blames the rig when both bats moved and the bat when one aged",
          G.attribute(*v_rig) == "rig" and G.attribute(*v_bat) == "bat",
          "rig day: %s, aged bat: %s (T^2 %.0f, %.0f; %.0f, %.0f)"
          % (G.attribute(*v_rig), G.attribute(*v_bat), v_rig[0].T2, v_rig[1].T2, v_bat[0].T2, v_bat[1].T2))
    # must fail: twice that drift on production bats is caught, and by
    # which layer (drift_layer), judged on the bats the line would ship
    e2 = 2.0 * eps[h_mf]
    layer, why = drift_layer(rb[("bat", "enc_scale", h_mf)], e2, tau)
    check("must fail: the hinge-%d encoder-scale drift at twice that size (%.2g) on production bats is caught, by %s"
          % (h_mf, e2, layer or "neither layer"), layer is not None, why)


def line_checks(prod, gp):
    from batline import line as L
    d = product.design(KID)
    sts, _dg = L.stations(d, timed=False)
    st = sts["calibration"]
    ops = {o.what: o.s for o in st.jobs["bat"]}
    cal = [s_ for w_, s_ in ops.items() if "calibration plan" in w_]
    check("line: the calibration station's op is the plan's duration (%.1f s)" % prod.duration_s,
          len(cal) == 1 and abs(cal[0] - prod.duration_s) < 1e-9, str(cal))
    gd = [o.s for o in st.jobs["day"] if "golden" in o.what and o.how == "plan"]
    check("line: the day job carries the golden day, the plan %d times (%.0f s)" % (gp.repeats, gp.duration_s),
          len(gd) == 1 and abs(gd[0] - gp.duration_s) < 1e-9, str(gd))
    names = ("GIMBAL_RATE", "STILL_POSES", "STILL_S", "SPIN_AXES", "SPIN_TURNS", "SPIN_RPS", "LEVER_SPINS",
             "LEVER_S", "GOLDEN_S")
    srcs = {f: open(os.path.join(SIM, "batline", f)).read() for f in ("spec.py", "line.py")}
    srcs["demo_rig.py"] = open(os.path.join(HERE, "demo_rig.py")).read()
    found = [(f, n) for f, src in srcs.items() for n in names if re.search(r"\b%s\b" % n, src)]
    check("line: the hand-written recipe is gone from spec, line and the rig demo", not found, str(found))
    planted = srcs["line.py"] + "\n    x = Est.STILL_POSES * Est.STILL_S\n"
    check("must fail: the scan finds a recipe name planted back",
          any(re.search(r"\b%s\b" % n, planted) for n in names))


# ============================================================ MAIN
def main():
    t_start = time.time()
    tmp = tempfile.mkdtemp(prefix="check_calibration_")
    try:
        return _main(tmp, t_start)
    except Exception:
        _report(t_start, {})                       # what ran before it, then the traceback
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _report(t_start, clock):
    """Print the results; the exit code."""
    import mujoco
    bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det and (VERBOSE or not ok) else ""))
    if VERBOSE and clock:
        print("  " + ", ".join("%s %.0f s" % kv for kv in clock.items()))
    print("check_calibration: %d checks, %d failed, %.0f s (MuJoCo %s)"
          % (len(RESULTS), bad, time.time() - t_start, mujoco.__version__))
    return 1 if bad else 0


def _main(tmp, t_start):
    ctx = get_context("spawn")
    clock = {}
    with ctx.Pool(min(4, os.cpu_count() or 1)) as pool:
        t0 = time.time()
        got = pool.map(design_task, ["kid", "kid", "adult"], chunksize=1)
        bad = [(w_, x) for w_, x, _ in got if isinstance(x, Exception)]
        if bad:                                    # golden_plan: infeasible, and why
            for w_, x in bad:
                check("golden: the %s's plan and golden day can be designed" % w_, False, str(x))
            return _report(t_start, clock)
        gp, gp2, gpa = (x[1] for x in got)
        kid, kid2, adult = gp.plan, gp2.plan, gpa.plan
        # what line.py reads (golden.for_design), as the pool designed it
        G._days[KID], G._days[ADULT] = gp, gpa
        clock["designs"] = time.time() - t0
        t0 = time.time()
        metas = {"kid": prepare(pool, tmp, "kid", kid, SEED["kid"]),
                 "adult": prepare(pool, tmp, "adult", adult, SEED["adult"])}
        clock["motions"] = time.time() - t0
        if any("lost" in m for m in metas.values()):
            # a plan the motors cannot run: the suite stops, since its bats
            # would run a motion that is not the plan (lost_step says which)
            plan_checks("kid", kid, metas["kid"], twin=kid2)
            plan_checks("adult (unseen)", adult, metas["adult"])
            return _report(t_start, clock)

        # ------------------------------------------------ the first round
        t0 = time.time()
        jobs = [(tmp, "kid", 1000 + i, "bat", "healthy",
                 "omit" if i < N_OMIT else "record" if i == N_OMIT else None) for i in range(N_KID)]
        jobs += [(tmp, "kid", 1000 + i, "bat", m, None) for m in OMIT for i in range(N_OMIT)]
        jobs += [(tmp, "adult", 2000 + i, "bat", "healthy", None) for i in range(N_ADULT)]
        jobs += [(tmp, "kid", 3000 + k, "bat", m, 3000 + k)
                 for k, m in enumerate(("stuck", "noisy", "scale8", "gap", "cut-first", "cut-last"))]
        jobs += [(tmp, "kid", 1000, "bat", "gap-move", None)]
        # the golden bats run the kid's plan on the kid's station
        K = Quality.GOLDEN_RUNS
        GB = (51, 52)                                  # the two golden bats
        r_ = gp.repeats
        jobs += [j for i, gb in enumerate(GB) for k in range(K) for j in golden_jobs(tmp, gb, 5000 + 100 * i + k, r_)]
        jobs += [j for k in range(N_DAYS) for j in golden_jobs(tmp, GB[0], 6000 + k, r_)]
        jobs += [j for i, (m, arg) in enumerate((("zero", 0), ("world", None))) for k in range(N_ABSORB)
                 for j in golden_jobs(tmp, GB[0], 6500 + 10 * i + k, r_, m, arg)]
        # the encoder-scale drift's direction in the terms: one run, with
        # and without a drift of the gyro scale's target
        step = F.GROUPS[4][2]
        jobs += [(tmp, "kid", (GB[0], 6900 + h, 0), "golden", m, (h, step) if m == "enc_scale" else "direction")
                 for h in (0, 1) for m in ("golden", "enc_scale")]
        out = pool.map(bat_task, jobs, chunksize=1)
        clock["round 1"] = time.time() - t0
        by = {}
        for (_, key, seed, kind, mode, arg), r in zip(jobs, out):
            by.setdefault(("golden" if kind == "golden" else key, kind,
                           "direction" if arg == "direction" else mode), []).append(r)

        # ------------------------------------------------ A. plan
        plan_checks("kid", kid, metas["kid"], twin=kid2)
        plan_checks("adult (unseen)", adult, metas["adult"])

        # ------------------------------------------------ B. fit
        healthy = by[("kid", "bat", "healthy")]
        E, SD, nees, nb, slow = fit_checks("kid", healthy, N_KID)
        fit_checks("adult (unseen)", by[("adult", "bat", "healthy")], N_ADULT)
        check("kid: every fit takes less time than the plan (slowest %.1f s of %.1f)" % (slow, kid.duration_s),
              slow < kid.duration_s)
        check("kid: no spin turned less than a whole quiet turn", all(r["short"] == 0 for r in healthy))
        check("kid: no healthy bat lagged its command past the plan's bound",
              all(r["follow"] <= r["lag"] for r in healthy))
        for label, runs in (("kid", healthy), ("adult (unseen)", by[("adult", "bat", "healthy")])):
            ce = np.array([r["clock0"][0] for r in runs])
            check("%s: timing()'s clock places every quiet sample within the windows' margin (%.0f ms) of the "
                  "bat's own clock, every bat" % (label, F.MARGIN * 1e3), np.all(ce <= F.MARGIN),
                  "worst %.2f ms, median %.2f; %d..%d starts timed of the plan's %d"
                  % (ce.max() * 1e3, np.median(ce) * 1e3, min(r["clock0"][1] for r in runs),
                     max(r["clock0"][1] for r in runs), len(metas[label.split()[0]]["plan"].starts)))
        rawE = raw_cross_errors(healthy)
        raw = group_rms(rawE)
        tau = np.array([g[2] for g in F.GROUPS])
        check("must fail: cross-axis terms judged as planted, not in the polar gauge, miss their targets",
              raw[2] > tau[2] and raw[5] > tau[5], "accel cross %.1f, gyro cross %.1f of target"
              % (raw[2] / tau[2], raw[5] / tau[5]))
        omission_checks(healthy[:N_OMIT], by, tau)

        # ------------------------------------------------ C. rejection
        rej = sum(not r["ok"] for r in healthy)
        lim = binom_upper(N_KID, Faults.REJECT["calibration"], ALPHA)
        check("kid: %d of %d healthy bats rejected, at most %d at the line's reject rate %.3f"
              % (rej, N_KID, lim, Faults.REJECT["calibration"]), rej <= lim,
              "; ".join(r["why"][0] for r in healthy if not r["ok"]))
        ah = by[("adult", "bat", "healthy")]
        rej = sum(not r["ok"] for r in ah)
        lim = binom_upper(N_ADULT, Faults.REJECT["calibration"], ALPHA)
        check("adult (unseen): %d of %d healthy bats rejected, at most %d at the line's reject rate %.3f"
              % (rej, N_ADULT, lim, Faults.REJECT["calibration"]), rej <= lim,
              "; ".join(r["why"][0] for r in ah if not r["ok"]))
        rejection_checks(by)

        # ------------------------------------------------ D. record
        rec = [r for r in healthy if "res" in r][0]
        record_checks(check, rec["res"], kid)

        # ------------------------------------------------ E. golden, then its second round
        t0 = time.time()
        golden_checks(pool, tmp, gp, kid, by, GB, step)
        clock["golden"] = time.time() - t0
    t0 = time.time()
    golden_synthetic(gp, kid)
    clock["golden, synthetic"] = time.time() - t0

    # ---------------------------------------------------- F. line
    line_checks(kid, gp)
    return _report(t_start, clock)


if __name__ == "__main__":
    sys.exit(main())
