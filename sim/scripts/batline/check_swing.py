"""Tier 2: the swing rig at station 4 -- one bat in Est.SWING_EVERY swung
through the rated swing at full speed on its dummy pack, streaming to the
station's dongle, to show its IMU does not rail and its cells do not let go.

    python3 sim/scripts/batline/check_swing.py [-v]

The plan's contract for this suite (README, Milestone 4): it fails if the
IMU saturates at the rated swing, or the battery circuit opens for longer
than the hold-up time.  Forward physics: the rig, its motors' loop and the
sprung pack in MuJoCo (swing/sim.py), the board and its link as
check_link's (board/sim.py), the test itself production code
(swing/rig.py) that sees only the HALs.

Every tolerance says where it comes from: float rounding, a differencing
step's own error, a drive's cycle, a Monte Carlo's standard error.  Each
check has a must-fail: a deliberately wrong variant the same test must
reject, so no check passes because it cannot fail.

THE RIG (swing/rig.py, no physics):
   1  the rig loads the pack at least as hard as the rated swing does, both
      ways: its largest pull toward the spring and its largest push onto
      the + plate cover the 3-D swing's own (imu_fusion_sim.Swing.pose,
      second differences at the pack's centre).  The detail states what
      the planar rig leaves out at the chip: the swing's tilt and roll
      rates.  Must fail: the bat turned about fixed hands.
   2  rig.torques (Newton-Euler, the rig's own mass model) equals MuJoCo's
      inverse dynamics of the rig as swing/sim.py builds it, along the
      swing, to the model file's printed precision.  Must fail: the bat's
      centre of mass taken at the hands.
   3  each joint's drive meets its four limits over the swing (peak and
      RMS torque, speed, inertia ratio), and the motor below it in
      Servo.CATALOGUE has no ratio that does.  Must fail: the arm on the
      motor below its own.
   4  the rig's own moves, park to the stance and back, load the pack no
      harder than the swing does.  Must fail: the move back to the nearest
      whole turn, which stands the bat on its pommel on the way.

THE PHYSICS (swing/sim.py, MuJoCo):
   5  the pack settles at rest in the rig under its spring, each contact
      carrying the spring's force; and halving the contacts' stiffness
      (sim.RING doubled) moves the run's least + plate force by under a
      hundredth.  Must fail: the contacts twice as stiff, which MuJoCo
      holds in a chatter that never settles.
   6  the rig follows its plan within one drive cycle's travel at each
      joint's peak rate, no command at its clip.  Must fail: the arm on
      the motor below its own.
   7  MuJoCo's gyro and accelerometer equal rig.imu_signals on MuJoCo's own
      state, to float rounding.  Must fail: the chip taken at the hands.
   8  the circuit never opens for longer than the board's hold-up
      (power.holdup_s, its capacitor at the low tolerance, the dongle's
      interval) on the dummy pack and on the shortest pair of cells the
      bay takes.  Must fail: a cap spring whose preload is half the
      swing's pull on the pack.
   9  the spring holds the pack, its tip coil's third of its mass added, at
      the swing's largest pull, on the shortest pair (its weakest preload).
      Must fail: the same half-strength spring.
  10  the chip's largest true rate and force on any axis through the run
      stay inside what a part at the calibration's quantile reads with no
      rail (imu/part.usable).  Must fail: a 2000 dps part.

THE TEST (rig.swing_test through the HALs):
  11  healthy bats pass every step.  Must fail: the half-strength spring is
      refused at "rides through the swing" and the 2000 dps part at "never
      rails".
  12  the test takes its planned time (rig.plan_s, what line.py books) on
      the mean healthy bat.  Must fail: a plan without the move back.
  13  the simulated rig, clock and radio meet their HALs.  Must fail: a rig
      with no angles().
  14  no production module of the rig (rig.py, hal.py) imports or loads
      MuJoCo or a TRUTH module.  Must fail: the same scan on swing/sim.py.

THE PHONE'S PICTURE (station 4's other test, phone/picture.py, MuJoCo):
  15  where the factory iPhone's app puts each band's centroid (app.seen:
      the middle of the band's silhouette, off the axis on the bat's
      rounded-triangle section) lies within Phone.SIGMA_PX, a still
      frame's own centroid noise, of the rendered band's pixels' centroid,
      at poses across the slow swing's turn.  The rendered picture is
      what showed the offset (demo_phone).  Must fail: each band's
      centre, which the app took before.
"""
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from math import erfc, sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from truss.glenv import headless         # noqa: E402  (before mujoco: check 15 renders)
headless()
import numpy as np                       # noqa: E402
import mujoco                            # noqa: E402

from batline.spec import (Module, Servo, CapSpring, Cell, Imu, Quality, Board, Link, Rules, SwingRig,  # noqa: E402
                          KID)
from batline.rig.pose import normal_quantile      # noqa: E402
from batline import product, imu_fusion_sim as study   # noqa: E402
from batline.product import MassProps             # noqa: E402
from batline.imu.part import usable               # noqa: E402
from batline.board import power as P, hal as H, procedures as PR   # noqa: E402
from batline.swing import rig as SR, sim as SS, hal as SH          # noqa: E402
from check_link import imports_truth, package_of  # noqa: E402

VERBOSE = "-v" in sys.argv
RESULTS = []
Z = Module.Z_SIGMAS
TAIL = 0.5 * erfc(Z / sqrt(2.0))
EPS = np.finfo(float).eps
G = study.G
STUDY_DT = 1e-5                # s: second differences on the swing's analytic pose
# product frame -> the study's bat frame (columns: along, face normal, their cross):
# product -z is the face (swing/rig.py THE FRAME)
TO_STUDY = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
HEALTHY = 6                    # bats through the station's test
PRODUCTION = ("swing/rig.py", "swing/hal.py")


def zshare(m):
    return normal_quantile(1.0 - TAIL / m)


def check(name, ok, detail="", secs=None):
    now = time.time()
    RESULTS.append((name, bool(ok), detail, now - check.t if secs is None else secs))
    check.t = now
    return bool(ok)


check.t = time.time()


@contextmanager
def patched(obj, **kw):
    old = {k: getattr(obj, k) for k in kw}
    for k, v in kw.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


D = product.design(KID)
RIG = SR.fit(D)
MO = SR.motion(D)
F_MAX = SR.swing_load(RIG)


def pack_centre(d, pack=SR.DUMMY):
    """m, product frame from the hands: the pack's centre."""
    cells, _ = SR.seated(d, pack)
    return np.array([(cells[0][1] + cells[-1][0]) / 2.0 - d.lay.x_hands, 0.0, -d.core.e]) / 1000.0


def smaller(dr):
    """The Drive of the motor below dr's in the catalogue at its best ratio,
    or None."""
    i = Servo.CATALOGUE.index(dr.motor)
    if i == 0:
        return None
    j = 0 if dr is SR.choose(RIG)[0] else 1
    q, qd, qdd = SR.swing_state(RIG, SR.swing_grid(RIG))
    tau = SR.torques(RIG, q, qd, qdd)
    belt = 1.0 if j == 0 else SwingRig.BELT_RATIO
    opts = SR.options(Servo.CATALOGUE[i - 1], belt, tau[:, j], qd[:, j], qdd[:, j], SR.load_inertia(RIG)[j])
    return min(opts, key=lambda o: o.worst()[1])


def weak_rate(d, pack=SR.DUMMY):
    """N/mm: a cap spring preloaded to half the swing's pull on the pack."""
    _, sl = SR.seated(d, pack)
    return 0.5 * pack.n * pack.m * 1e-3 * F_MAX / (CapSpring.L_FREE - sl)


def holdup():
    """s: the board's hold-up on the dummy pack, its capacitor at the low
    tolerance, streaming to the dongle at the test's interval."""
    return P.holdup_s(Rules.N_CELLS * P.dummy_volts(), Board.HOLD_C * (1.0 - Board.HOLD_TOL), Link.CI_MIN,
                      Link.PER_EVENT["dongle"])


# ================================================================ THE RIG
def rig_checks():
    d, rig, sw = D, RIG, RIG.swing
    # 1: the pack's load, the 3-D swing's against the rig's, on one grid
    t = np.arange(0.0, sw.t_end, STUDY_DT)
    Rb, Hh = sw.pose(t)
    pk = Hh + np.einsum("nij,j->ni", Rb, TO_STUDY @ pack_centre(d))
    acc = (pk[2:] - 2.0 * pk[1:-1] + pk[:-2]) / STUDY_DT ** 2
    f3 = np.einsum("ni,ni->n", acc - study.GW, Rb[1:-1, :, 0])
    q, qd, qdd = SR.swing_state(rig, t[1:-1])
    fr = SR.pack_load(rig, q, qd, qdd)
    # rounding: the study's second difference, and the rig's at D_FINE (its
    # angles' rounding over D_FINE^2, at the pack's reach from the hub)
    reach = rig.L + float(np.linalg.norm(pack_centre(d)))
    tol = 4.0 * EPS * (np.max(np.abs(pk)) / STUDY_DT ** 2 + np.max(np.abs(q)) * reach / SR.D_FINE ** 2)
    covers = fr.max() >= f3.max() - tol and fr.min() <= f3.min() + tol
    # the chip's rates: what the planar rig leaves out
    p_imu, R_imu = rig.imu
    Rc = Rb @ TO_STUDY @ R_imu
    w3 = study.log_so3(np.einsum("nji,njk->nik", Rc[:-1], Rc[1:])) / STUDY_DT
    wr, _ = SR.imu_signals(rig, q, qd, qdd)
    check("the rig loads the pack at least as hard as the rated swing, both ways (1)", covers,
          "pull toward the spring %.3f g (swing %.3f), push onto the plate %.2f g (swing %.2f); the chip's "
          "peak rate on each axis %s dps on the rig, %s dps in the swing (its tilt and roll, left out)"
          % (fr.max() / G, f3.max() / G, -fr.min() / G, -f3.min() / G,
             np.array2string(np.degrees(np.abs(wr).max(0)), precision=0),
             np.array2string(np.degrees(np.abs(w3).max(0)), precision=0)))
    fixed = SR.pack_load(replace(rig, L=0.0), q, qd, qdd)
    check("must fail: the bat turned about fixed hands (1)",
          not (fixed.max() >= f3.max() - tol and fixed.min() <= f3.min() + tol),
          "pull %.2f g, push %.2f g" % (fixed.max() / G, -fixed.min() / G))

    # 2: Newton-Euler against MuJoCo's inverse dynamics
    s = SS.Sim(d)
    m, dd = s.m, s.data
    tg = SR.swing_grid(rig)[::25]
    q, qd, qdd = SR.swing_state(rig, tg)
    ne = SR.torques(rig, q, qd, qdd)
    (_, ip), (_, iq), (_, ir) = s.j["phi"], s.j["q2"], s.j["rot"]
    M = np.zeros((m.nv, m.nv))
    mj = np.zeros_like(ne)
    for k in range(len(tg)):
        s.place(q[k])
        dd.qvel[:] = 0.0
        dd.qvel[ip], dd.qvel[iq], dd.qvel[ir] = qd[k, 0], qd[k, 1] - qd[k, 0], qd[k, 1]
        mujoco.mj_forward(m, dd)
        try:
            mujoco.mj_fullM(m, dd, M)                # MuJoCo >= 3.3 (as imu/scene.py)
        except TypeError:
            mujoco.mj_fullM(m, M, dd.qM)
        a = np.zeros(m.nv)
        a[ip], a[iq], a[ir] = qdd[k, 0], qdd[k, 1] - qdd[k, 0], qdd[k, 1]
        f = M @ a + dd.qfrc_bias
        f_phi = f[ip] - m.dof_armature[ip] * a[ip]
        mj[k] = f_phi - f[iq], f[iq]                 # (phi, q2) -> the joints to the room
    peak = np.abs(ne).max(0)
    gap = np.abs(mj - ne).max(0) / peak
    check("rig.torques equals MuJoCo's inverse dynamics along the swing (2)", np.all(gap < 1e-6),
          "worst %.1e of the arm's peak %.1f N m, %.1e of the wrist's %.2f N m (the model prints 9 digits)"
          % (gap[0], peak[0], gap[1], peak[1]))
    at_hands = MassProps(rig.bat.m, np.zeros(3), rig.bat.I)
    wrong = np.abs(mj - SR.torques(rig, q, qd, qdd, bat=at_hands)).max(0) / peak
    check("must fail: the bat's centre of mass at the hands (2)", not np.all(wrong < 1e-6),
          "worst %.2f and %.2f of the peaks" % tuple(wrong))

    # 3: the motors
    rows, ok = [], True
    for name, dr in zip(("arm", "wrist"), SR.choose(rig)):
        w_, share = dr.worst()
        lower = smaller(dr)
        ok &= share <= 1.0 and (lower is None or lower.worst()[1] > 1.0)
        rows.append("%s: %s, worst %s %.2f; %s" % (name, dr.name, w_, share,
                                                   "the smallest made" if lower is None else
                                                   "%s best %s %.2f" % (lower.name, *lower.worst())))
    check("each joint's drive meets its four limits; the motor below it does not (3)", ok, "; ".join(rows))
    weak = smaller(SR.choose(rig)[0])
    check("must fail: the arm on the motor below its own (3)", weak.worst()[1] > 1.0,
          "%s: %s %.2f" % (weak.name, *weak.worst()))

    # 4: the rig's own moves
    tt = np.arange(0.0, MO.T, STUDY_DT)
    run = SR.pack_load(rig, *MO.state(tt))
    s_ = np.linspace(0.0, 1.0, SR.S_GRID)
    curv = []
    for mv in (MO.to, MO.back):
        fl = SR.pack_load(rig, *[x * k for x, k in zip(mv.path(s_), (1.0, 1.0 / mv.T, 1.0 / mv.T ** 2))])
        curv.append(np.max(np.abs(np.diff(fl, 2))))          # its curvature over one step of S_GRID, squared
    tol4 = max(curv) / 8.0                                    # a peak between samples: (ds)^2/8 of it
    check("the rig's own moves load the pack no harder than the swing (4)", run.max() <= F_MAX + tol4,
          "%.4f g over the run, the swing's %.4f g (sampling allows %.1e g); park at %.0f deg, the moves %.3f "
          "and %.3f s" % (run.max() / G, F_MAX / G, tol4 / G, np.degrees(MO.park[1]), MO.to.T, MO.back.T))
    w, a = SR.move_limits(rig, SR.choose(rig))
    q1 = MO.back.q0
    near = SR.Move(q1, np.array([MO.park[0], 2.0 * np.pi * round(q1[1] / (2.0 * np.pi))]), w, a)
    fb = SR.pack_load(rig, *near.state(np.arange(0.0, near.T, STUDY_DT)))
    check("must fail: the move back to the nearest whole turn (4)", not fb.max() <= F_MAX + tol4,
          "%.2f g" % (fb.max() / G))


# ================================================================ THE PHYSICS
RUNS = {}


def run_of(key, **kw):
    if key not in RUNS:
        s = SS.Sim(D, **kw)
        RUNS[key] = (s, s.run(MO))
    return RUNS[key]


def physics_checks():
    d, rig = D, RIG
    # 5: the pack at rest, and the contacts' stiffness
    s = SS.Sim(d)
    s.settle(MO.park)
    fc = s.contact_forces()
    fs = SR.spring_force(d, SR.DUMMY)
    rest = np.all(np.abs(fc - fs) <= 1e-3 * fs)
    s0, r0 = run_of("dummy")
    with patched(SS, RING=2.0 * SS.RING):
        r2 = SS.Sim(d).run(MO)
    moved = abs(r2.force.min() - r0.force.min()) / r0.force.min()
    check("the pack settles at rest under its spring; a softer contact changes nothing (5)",
          rest and moved < 1e-2,
          "at rest %s N, the spring %.3f N; least plate force over the run %.3f N, %.3f N at twice the "
          "softness (%.1e)" % (np.array2string(fc, precision=3), fs, r0.force.min(), r2.force.min(), moved))
    try:
        with patched(SS, RING=SS.RING / 2.0):
            SS.Sim(d).settle(MO.park)
        chatter = False
    except ValueError:
        chatter = True
    check("must fail: the contacts twice as stiff (5)", chatter, "it never settles" if chatter else "it settled")

    # 6: tracking
    lim = np.abs(SR.swing_state(rig, SR.swing_grid(rig))[1]).max(0) * Servo.CYCLE_S
    err = np.abs(r0.err).max(0)
    check("the rig follows its plan within a drive cycle's travel at peak rate, no command clipped (6)",
          np.all(err <= lim) and not r0.sat.any(),
          "arm %.2f mrad of %.1f, wrist %.2f mrad of %.1f; peak commands %.2f and %.2f N m"
          % (err[0] * 1e3, lim[0] * 1e3, err[1] * 1e3, lim[1] * 1e3, *np.abs(r0.cmd).max(0)))
    weak = smaller(SR.choose(rig)[0])
    _, rw = run_of("weak arm", drives=(weak, SR.choose(rig)[1]))
    ew = np.abs(rw.err).max(0)
    check("must fail: the arm on the motor below its own (6)", not (np.all(ew <= lim) and not rw.sat.any()),
          "arm %.1f mrad, clipped %.1f%% of steps" % (ew[0] * 1e3, 100.0 * rw.sat[:, 0].mean()))

    # 7: the IMU's law
    w, f = SR.imu_signals(rig, r0.q, r0.qd, r0.qdd)
    gw = np.abs(r0.w - w).max() / np.abs(r0.w).max()
    gf = np.abs(r0.f - f).max() / np.abs(r0.f).max()
    check("MuJoCo's gyro and accelerometer equal the rigid body's law on its own state (7)",
          gw < 1e-6 and gf < 1e-6, "worst %.1e of the peak rate, %.1e of the peak force" % (gw, gf))
    w2, f2 = SR.imu_signals(replace(rig, imu=(np.zeros(3), rig.imu[1])), r0.q, r0.qd, r0.qdd)
    bad = np.abs(r0.f - f2).max() / np.abs(r0.f).max()
    check("must fail: the chip at the hands (7)", not bad < 1e-6, "worst %.2f of the peak force" % bad)

    # 8: the circuit
    hold = holdup()
    _, rs = run_of("shortest", pack=SR.SHORTEST)
    longest = {nm: max([b - a for a, b in r.opens], default=0.0) for nm, r in (("dummy", r0), ("shortest", rs))}
    check("the circuit never opens past the board's hold-up, on the dummy pack and the shortest pair (8)",
          all(v <= hold for v in longest.values()),
          "longest open %.2f and %.2f ms of %.1f; least plate force %.2f and %.2f N"
          % (longest["dummy"] * 1e3, longest["shortest"] * 1e3, hold * 1e3, r0.force.min(), rs.force.min()))
    _, rk = run_of("weak spring", rate=weak_rate(d))
    lk = max([b - a for a, b in rk.opens], default=0.0)
    check("must fail: a spring at half the swing's pull (8)", not lk <= hold,
          "%d opens, the longest %.1f ms" % (len(rk.opens), lk * 1e3))

    # 9: the spring's hold, statically
    tip = CapSpring.MASS / 3.0
    _, sl = SR.seated(d, SR.SHORTEST)
    pull = (SR.SHORTEST.n * SR.SHORTEST.m + tip) * 1e-3 * F_MAX
    fs_short = SR.spring_force(d, SR.SHORTEST)
    check("the spring holds the shortest pair, its tip coil added, at the swing's pull (9)", pull < fs_short,
          "%.3f N of %.2f; the tip coil's share %.3f N" % (pull, fs_short, tip * 1e-3 * F_MAX))
    fw = weak_rate(d) * (CapSpring.L_FREE - SR.seated(d, SR.DUMMY)[1])
    check("must fail: the half-strength spring (9)", not pull < fw, "%.3f N of %.3f" % (pull, fw))

    # 10: no rail
    z = normal_quantile(1.0 - Quality.CAL_QUAL_ALPHA / 12.0)
    gy, ac = usable(z)
    pw, pf = np.abs(r0.w).max(), np.abs(r0.f).max()
    check("the chip's largest rate and force stay inside what the quantile part reads with no rail (10)",
          pw <= gy and pf <= ac, "%.0f of %.0f dps, %.1f of %.1f g" % (np.degrees(pw), np.degrees(gy), pf / G, ac / G))
    with patched(Imu, GYRO_FS=2000.0):
        gy2, _ = usable(z)
    check("must fail: a 2000 dps part (10)", not pw <= gy2, "%.0f of %.0f dps" % (np.degrees(pw), np.degrees(gy2)))


# ================================================================ THE TEST
def test_checks():
    d = D
    secs, bad = [], []
    for i in range(HEALTHY):
        sb = SS.ready(d, 3000 + i)
        v = SR.swing_test(sb.station, sb.serial, d)
        secs.append(v.seconds)
        if not v.ok:
            bad.append("%s: %s" % (sb.serial, v.why))
    check("%d healthy bats pass the swing test (11)" % HEALTHY, not bad, "; ".join(bad) or "all")
    sb = SS.ready(d, 3100, rate=weak_rate(d))
    vk = SR.swing_test(sb.station, sb.serial, d)
    with patched(Imu, GYRO_FS=2000.0):
        sb = SS.ready(d, 3101)
        vg = SR.swing_test(sb.station, sb.serial, d)
    stop = lambda v: v.steps[-1].name if not v.ok else "passed"     # noqa: E731
    check("must fail: the half-strength spring and the 2000 dps part, each at its step (11)",
          stop(vk) == "rides through the swing" and stop(vg) == "never rails",
          "spring: %s (%s); part: %s (%s)" % (stop(vk), vk.why, stop(vg), vg.why))

    # 12: time
    secs = np.array(secs)
    plan = SR.plan_s(d)
    se = secs.std(ddof=1) / sqrt(len(secs))
    within = secs.mean() <= plan + zshare(2) * se and secs.max() <= PR.timeout(plan)
    check("the swing test takes %.2f s on the mean healthy bat, planned %.2f (12)" % (secs.mean(), plan), within,
          "se %.3f s; the longest %.2f s, the overrun %.1f s" % (se, secs.max(), PR.timeout(plan)))
    short = plan - MO.back.T
    check("must fail: a plan without the move back (12)", not secs.mean() <= short + zshare(2) * se,
          "it plans %.2f s" % short)

    # 13: HALs
    sb = SS.ready(d, 3200)
    missing = []
    for obj, c in ((sb.station.rig, SH.SwingRigHAL), (sb.station.clock, H.ClockHAL), (sb.station.radio, H.RadioHAL)):
        missing += H.audit(obj, (c,))
    check("the simulated rig, clock and radio meet their HALs (13)", not missing, ", ".join(missing) or "all")

    class Half:
        def run(self, motion):
            pass
    check("must fail: a rig with no angles() (13)", H.audit(Half(), (SH.SwingRigHAL,)) == ["SwingRigHAL.angles"], "")

    # 14: the truth wall
    root = os.path.join(HERE, "..", "..", "batline")
    hits = []
    for rel in PRODUCTION:
        hits += ["%s: %s" % (rel, h) for h in imports_truth(open(os.path.join(root, rel)).read(), package_of(rel))]
    import check_link as CL
    sim_dir = os.path.abspath(os.path.join(HERE, "..", ".."))
    code = ("import sys; sys.path.insert(0, %r); import importlib; importlib.import_module(sys.argv[1]); "
            "print(' '.join(sorted(k for k in sys.modules if k == 'mujoco' or k in %r)))" % (sim_dir, CL.TRUTH))
    loaded = []
    for rel in PRODUCTION:
        mod = "batline." + rel[:-3].replace("/", ".")
        p = subprocess.run([sys.executable, "-c", code, mod], capture_output=True, text=True)
        got = p.stdout.strip() if p.returncode == 0 else "failed to import: %s" % p.stderr.strip()[-200:]
        if got:
            loaded.append("%s: %s" % (mod, got))
    check("no production module of the rig imports or loads MuJoCo or a TRUTH module (14)",
          not hits and not loaded, "; ".join(hits + loaded) or "%d modules, none" % len(PRODUCTION))
    seen = imports_truth(open(os.path.join(root, "swing/sim.py")).read(), package_of("swing/sim.py"))
    check("must fail: the same scan on swing/sim.py (14)", bool(seen), ", ".join(seen))


VIEW_POSES = 5                 # gimbal poses the picture is rendered at, evenly over the slow swing's turn


def view_checks():
    from batline.spec import Phone
    from batline.rig import spec as RS
    from batline.phone import test as PT, app as A, picture as PP
    sp = PT.slow_swing(D)
    v = sp.view
    pic = PP.Picture(D, sp, RS.design(D))
    model, centre, rows = [], [], []
    try:
        for k in np.linspace(0.0, 1.0, VIEW_POSES):
            q = sp.q_a + k * (sp.q_b - sp.q_a)
            pic.pose(q)
            got, n = pic.silhouettes()
            Rc, x = sp.kin.pose(np.atleast_2d(q))
            uv = v.project_level(A.seen(v.level(x[0])[0], v.R.T @ Rc[0], sp.geom))
            uc = v.project(x[0] + sp.geom.bands @ Rc[0].T)
            model.append(np.linalg.norm(uv - got, axis=1))
            centre.append(np.linalg.norm(uc - got, axis=1))
            rows.append("%.0f deg: %s px over %s px" % (np.degrees(q[1]), "/".join("%.2f" % e for e in model[-1]),
                                                       "/".join("%d" % c for c in n)))
    finally:
        pic.close()
    model, centre = np.array(model), np.array(centre)
    check("the app's band centroids lie on the rendered bands' silhouettes, within a still frame's %.1f px (15)"
          % Phone.SIGMA_PX, np.all(model <= Phone.SIGMA_PX),
          "largest %.2f px; %s" % (np.nanmax(model), "; ".join(rows)))
    check("must fail: each band's centre (15)", not np.all(centre <= Phone.SIGMA_PX),
          "%.2f to %.2f px off" % (np.nanmin(centre), np.nanmax(centre)))


def main():
    t0 = time.time()
    for fn in (rig_checks, physics_checks, test_checks, view_checks):
        t = time.time()
        fn()
        if VERBOSE:
            print("  (%s: %.0f s)" % (fn.__name__, time.time() - t))
    bad = sum(1 for _, ok, _, _ in RESULTS if not ok)
    for nm, ok, det, secs in RESULTS:
        print("  %s %5.1fs  %s%s" % ("ok  " if ok else "FAIL", secs, nm,
                                     ("  [%s]" % det) if det and (VERBOSE or not ok) else ""))
    print("check_swing: %d checks, %d failed, %.0f s" % (len(RESULTS), bad, time.time() - t0))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
