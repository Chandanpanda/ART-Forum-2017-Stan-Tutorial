"""WHERE THINGS ACTUALLY LAND.  Run this on every change.

Both robots plan against a model of their own effectors -- "the hopper
discharges 140 mm over the flank", "the tray ejects 71 mm off the tail",
"a released patient stays where the pocket mouth was".  Nothing measured
any of that.  Twice in one session a wrong model shipped:

  * the hopper's stride was assumed sideways when the kits step 28 mm
    along the body's FORWARD axis, which made a solver reject the hospital
    station that demonstrably works
  * the tray's ejection offset was read off one log by eye

Both are cheap to measure and neither is a matter of opinion: drive the
effector in the simulator, look at where the payload came to rest, express
it in BODY frame, and compare it with the number the planners use.  When
these disagree, the planner is planning in a world that does not exist.

The measured values are also exactly what station.Effector wants, so this
suite is what keeps the pose solver honest about this machine.

    python3 sim/scripts/check_effectors.py [-v]
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import mujoco

from rfgyc26 import fleet, mjcf, nav, robot2, station
from rfgyc26.params import AgentA, Field, M2, Piece, Robot2 as R2
from rfgyc26.robot import AgentARobot

VERBOSE = "-v" in sys.argv
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def to_body(pose, x, y):
    """A world point in the robot's frame: +x forward, +y left."""
    a = np.radians(pose[2])
    dx, dy = x - pose[0], y - pose[1]
    return (dx*np.cos(a) + dy*np.sin(a), -dx*np.sin(a) + dy*np.cos(a))


def scene(pose, r2=False, r2_pose=None):
    rng = np.random.default_rng(11)
    m = mujoco.MjModel.from_xml_string(mjcf.scene_full_match(
        [(100.0, 100.0), (160.0, 190.0), (220.0, 110.0)], rng=rng,
        robot_pose=pose, r2=r2, r2_pose=r2_pose))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def xy(m, d, name):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    return None if b < 0 else (float(d.xpos[b][0]*1000),
                               float(d.xpos[b][1]*1000))


# ------------------------------------------------------ robot 1's hoppers
def measure_hopper(dest, pose, settle=2.0):
    """Open one hopper and report where its kits land, in body frame."""
    m, d = scene(pose)
    rb = AgentARobot(m, d, rng=np.random.default_rng(3))
    for _ in range(120):
        mujoco.mj_step(m, d)
    n = rb.open_hopper(dest)
    for _ in range(int(settle / m.opt.timestep)):
        mujoco.mj_step(m, d)
    pts = []
    for i in M2.KIT_GROUPS[dest]:
        p = xy(m, d, "kit%d" % i)
        if p and p[0] < 2000.0:
            pts.append(to_body(rb.pose, *p))
    return n, pts


def main():
    # ------------------------------------------------------------ hoppers
    for dest in ("HOSP", "PCC_L"):
        want = M2.HOPPER[dest]
        got = []
        for pose in ((600.0, 600.0, 90.0), (600.0, 600.0, 0.0),
                     (500.0, 700.0, 45.0)):
            n, pts = measure_hopper(dest, pose)
            if not pts:
                continue
            got.append((pose[2], n, pts))
        check("%s: the hopper releases its whole group" % dest,
              bool(got) and all(n == len(M2.KIT_GROUPS[dest])
                                for _, n, _ in got),
              str([(h, n) for h, n, _ in got]))
        if got:
            # the FIRST kit's body-frame offset is what M2.HOPPER claims
            first = [min(pts, key=lambda p: np.hypot(*p)) for _, _, pts in got]
            spread = max(np.hypot(a[0]-b[0], a[1]-b[1])
                         for a in first for b in first)
            # THE SPREAD IS A CALIBRATION, NOT A TOLERANCE.  Kits tumble
            # out and settle differently at different headings; what the
            # planners need is not that it be small but that the number
            # they plan with is not smaller than the truth.  An optimistic
            # spread aims at the edge of a zone and misses it.
            check("%s: the measured discharge spread is not worse than "
                  "M2.HOPPER_SPREAD says" % dest,
                  spread / 2.0 <= M2.HOPPER_SPREAD + 1e-6,
                  "measured +-%.0f mm, declared +-%.0f"
                  % (spread/2.0, M2.HOPPER_SPREAD))
            mean = (float(np.mean([p[0] for p in first])),
                    float(np.mean([p[1] for p in first])))
            check("%s: ...and it is where M2.HOPPER says" % dest,
                  np.hypot(mean[0]-want[0], mean[1]-want[1]) < 60.0,
                  "measured (%.0f, %.0f) vs declared (%.0f, %.0f)"
                  % (mean[0], mean[1], want[0], want[1]))
            # THE STRIDE, which is what was got wrong: successive kits step
            # along the body's forward axis, not sideways.
            _, _, pts = max(got, key=lambda g: len(g[2]))
            if len(pts) >= 2:
                pts = sorted(pts, key=lambda p: p[0])
                dxs = [pts[i+1][0]-pts[i][0] for i in range(len(pts)-1)]
                dys = [pts[i+1][1]-pts[i][1] for i in range(len(pts)-1)]
                check("%s: successive kits step along the body's FORWARD axis"
                      % dest,
                      abs(np.mean(dxs)) > abs(np.mean(dys)) + 8.0,
                      "step (%.0f, %.0f) mm per kit"
                      % (np.mean(dxs), np.mean(dys)))
                # and station.Effector built from these must reproduce them
                eff = station.Effector(offset=want, spread=M2.HOPPER_SPREAD,
                                       stride=(float(np.mean(dxs)), 0.0),
                                       count=len(M2.KIT_GROUPS[dest]))
                pred = eff.deposits(0.0, 0.0, 0.0)
                err = max(np.hypot(p[0]-q[0], p[1]-q[1])
                          for p, q in zip(sorted(pred), sorted(pts)))
                check("%s: station.Effector predicts the real landing points"
                      % dest, err < 45.0, "worst %.0f mm" % err)

    # ------------------------------------------------- robot 2's kit tray
    m, d = scene((150.0, 150.0, 0.0), r2=True, r2_pose=(600.0, 400.0, 90.0))
    link = robot2.SimLink(m, d, rng=np.random.default_rng(5))
    spot = robot2.sim_spot(m, d, np.random.default_rng(7))
    ctl = robot2.R2Controller(link, spot, clock=lambda: d.time)
    for _ in range(10):
        ctl.tick()
        link.step(20)
    p0 = ctl.pose
    g = ctl.shake_out(4)
    t0 = d.time
    while d.time - t0 < 6.0:
        try:
            next(g)
        except StopIteration:
            break
        link.step(20)
    for _ in range(60):
        link.step(20)
    tray = [to_body(p0, *xy(m, d, "kit%d" % i))
            for i in M2.KIT_GROUPS["PCC_R"]
            if xy(m, d, "kit%d" % i)[0] < 2000.0]
    check("robot 2's tray ejects both kits", len(tray) == 2,
          "%d landed" % len(tray))
    if tray:
        back = float(np.mean([p[0] for p in tray]))
        check("...off the TAIL, behind the axle", back < 0.0,
              "body-frame x %.0f mm" % back)
        check("...at the offset R2.EJECT_BACK declares",
              abs(abs(back) - R2.EJECT_BACK) < 35.0,
              "measured %.0f mm vs declared %.0f"
              % (abs(back), R2.EJECT_BACK))

    # --------------------------------------------- robot 2's pocket release
    m, d = scene((150.0, 150.0, 0.0), r2=True, r2_pose=(500.0, 600.0, 0.0))
    link = robot2.SimLink(m, d, rng=np.random.default_rng(5))
    spot = robot2.sim_spot(m, d, np.random.default_rng(7))
    ctl = robot2.R2Controller(link, spot, clock=lambda: d.time)
    for _ in range(10):
        ctl.tick()
        link.step(20)
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cyl0_f")
    a = m.jnt_qposadr[j]
    d.qpos[a:a+3] = [(500.0 + R2.CAPTURE_X + 55.0)/1000.0, 0.600, 0.011]
    mujoco.mj_forward(m, d)

    def run(gen, cap):
        t0 = d.time
        while d.time - t0 < cap:
            try:
                next(gen)
            except StopIteration as e:
                return e.value
            link.step(20)
        return None

    run(ctl.capture(*xy(m, d, "cyl0")), 8.0)
    seated = ctl.holding(*xy(m, d, "cyl0"))
    check("robot 2's pocket seats a patient driven into it", bool(seated))
    if seated:
        b = to_body(ctl.pose, *xy(m, d, "cyl0"))
        check("...at CAPTURE_X ahead of the axle",
              abs(b[0] - R2.CAPTURE_X) < 30.0 and abs(b[1]) < 30.0,
              "body frame (%.0f, %.0f) vs CAPTURE_X %.0f"
              % (b[0], b[1], R2.CAPTURE_X))
        pose_at_release = ctl.pose
        run(ctl.release(150.0), 8.0)
        left = xy(m, d, "cyl0")
        pred = station.Effector(offset=(R2.CAPTURE_X, 0.0),
                                spread=25.0).deposits(*pose_at_release)[0]
        check("...and a release leaves it where the mouth was",
              np.hypot(left[0]-pred[0], left[1]-pred[1]) < 60.0,
              "left at (%.0f, %.0f), predicted (%.0f, %.0f)"
              % (left[0], left[1], pred[0], pred[1]))

    # ------------------------------------------------ robot 1's beam seats
    m, d = scene((400.0, 400.0, 0.0))
    rb = AgentARobot(m, d, rng=np.random.default_rng(3))
    for _ in range(200):
        mujoco.mj_step(m, d)
    for i, want in ((1, AgentA.BEAM1_LOCAL), (2, AgentA.BEAM2_LOCAL)):
        p = xy(m, d, "beam%d" % i)
        b = to_body(rb.pose, *p)
        check("beam %d rides where AgentA.BEAM%d_LOCAL says" % (i, i),
              np.hypot(b[0]-want[0], b[1]-want[1]) < 25.0,
              "body frame (%.0f, %.0f) vs declared (%.0f, %.0f)"
              % (b[0], b[1], want[0], want[1]))

    # --------------------------------------------- the fleet's body models
    for who, L, W in (("r1", AgentA.L, AgentA.W), ("r2", 156.0, 110.0)):
        chain = fleet.BODY[who]
        a, hw = L/2.0, W/2.0
        bad = 0
        for lx in np.linspace(-a, a, 40):
            for ly in np.linspace(-hw, hw, 40):
                if not any(np.hypot(lx-cx, ly) <= r + 1e-9 for cx, r in chain):
                    bad += 1
        check("fleet.BODY[%s] contains the whole %0.f x %0.f chassis"
              % (who, L, W), bad == 0, "%d sample points outside" % bad)

    # ------------------------------------------------- robot 2's drivetrain
    m, d = scene((150.0, 150.0, 0.0), r2=True, r2_pose=(600.0, 750.0, 0.0))   # open floor: NOT on the lab plate
    link = robot2.SimLink(m, d, rng=np.random.default_rng(5))
    spot = robot2.sim_spot(m, d, np.random.default_rng(7))
    ctl = robot2.R2Controller(link, spot, clock=lambda: d.time)
    for _ in range(10):
        ctl.tick()
        link.step(20)
    p0 = ctl.pose
    V, T = 250.0, 1.5
    for _ in range(int(T * 50)):
        link.cmd(V, V, 150)
        ctl.tick()
        link.step(20)
    p1 = ctl.pose
    gone = float(np.hypot(p1[0]-p0[0], p1[1]-p0[1]))
    check("robot 2 drives about as far as it is told",
          abs(gone - V*T) < 0.30 * V*T,
          "%.0f mm travelled for %.0f commanded" % (gone, V*T))
    check("...in a straight line", abs(robot2._wrap(p1[2]-p0[2])) < 12.0,
          "yaw drifted %.1f deg" % robot2._wrap(p1[2]-p0[2]))
    # A PIVOT SCRUBS AND THE GEOMETRY IS A LIE.  (vr - vl) / track is what a
    # rolling differential drive would turn; a real one pivoting in place
    # drags both tyres sideways and turns considerably less.  check_hal says
    # the same of robot 1 -- "a pivot's scrub is VISIBLE as
    # commanded-over-actual, which is the estimator's process noise, not a
    # bug".  So measure the ratio rather than assert the formula: what
    # matters is that it is stable and less than one, because anything that
    # plans a pivot duration off the geometry will under-run it.
    p0 = ctl.pose
    for _ in range(int(1.0 * 50)):
        link.cmd(-V, V, 150)
        ctl.tick()
        link.step(20)
    turned = abs(robot2._wrap(ctl.pose[2] - p0[2]))
    geom = np.degrees(2.0 * V / R2.TRACK)
    scrub = turned / geom
    check("a pivot turns LESS than the rolling geometry predicts (scrub)",
          0.20 < scrub < 1.0,
          "%.0f of %.0f deg/s -- scrub factor %.2f" % (turned, geom, scrub))
    # An ARC is what the robot actually drives (no_pivot carries), and there
    # the wheels roll, so the geometry should hold.
    p0 = ctl.pose
    vl, vr = 200.0, 320.0
    for _ in range(int(1.0 * 50)):
        link.cmd(vl, vr, 150)
        ctl.tick()
        link.step(20)
    turned = abs(robot2._wrap(ctl.pose[2] - p0[2]))
    geom = abs(np.degrees((vr - vl) / R2.TRACK))
    check("an arc turns about as fast as the rolling geometry predicts",
          abs(turned - geom) < 0.35 * geom,
          "%.0f deg turned for %.0f predicted" % (turned, geom))

    n_bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det else ""))
    print("check_effectors: %d checks, %d failed" % (len(RESULTS), n_bad))
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
