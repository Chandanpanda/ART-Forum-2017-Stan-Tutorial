"""Phase 0e: the estimator, against the oracle.  Run on every change.

Everything in route.py now navigates on est.pose; this suite is the only
place the belief and the truth meet.  Three layers:

  * MOTION MODEL -- dead reckoning alone (gyro heading, coast-corrected
    wheels) holds millimetres over a scripted transit, and a stall does NOT
    walk the belief through a wall.
  * FIXES -- one camera look pulls a deliberately-wrong belief back in.
  * THE MISSION -- a full match, model camera, tracking |est - truth| every
    control tick.  The percentile gates are what the route actually needs:
    transits tolerate a couple of centimetres, and every terminal that needs
    better (dock, beams) closes on relative measurements, not on this.

    python3 sim/scripts/check_estimator.py [-v]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mujoco
from rfgyc26 import mjcf, hal
from rfgyc26.params import Field, AgentA
from rfgyc26.robot import AgentARobot, SimClock
from rfgyc26.route import mission_agent_a

VERBOSE = "-v" in sys.argv
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def build(pose=(300., 660., 0.), seed=0):
    rng = np.random.default_rng(seed)
    xml = mjcf.scene_full_match([(2500., 2400.), (2600., 2500.), (2700., 2600.)],
                                robot_pose=pose, rng=rng, kits_aboard=False)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    rb = AgentARobot(m, d, rng=rng, nav="est")
    rb.fingers(True); rb.gate(False); rb.intake(False)
    rb.cradle(1, True); rb.cradle(2, True)
    clk = SimClock(m, d)
    for _ in range(50):
        clk.tick()
    return m, d, rb, clk


def err(rb):
    ex, ey, eth = rb.pose
    tx, ty, tth = rb.pose_truth
    return np.hypot(ex-tx, ey-ty), abs((eth-tth+180.0) % 360.0 - 180.0)


def main():
    # ------------------------------------------------- dead reckoning alone
    m, d, rb, clk = build()
    def leg(v, w, secs):
        for _ in range(int(secs*hal.Clock.HZ)):
            rb.drive(v, w); clk.tick()
        rb.stop()
        for _ in range(15):
            rb.stop(); clk.tick()
    leg(200.0, 0.0, 2.0)
    e1, _ = err(rb)
    check("one 400 mm leg: coast-corrected odometry inside 4 mm",
          e1 < 4.0, "%.2f mm" % e1)
    leg(0.0, 120.0, 0.75)                        # ~90 deg pivot
    leg(180.0, 0.0, 1.5)
    leg(0.0, -120.0, 0.75)
    leg(180.0, 0.0, 1.5)
    e2, eth = err(rb)
    check("1.2 m + two pivots, NO fixes: position inside 25 mm",
          e2 < 25.0, "%.1f mm" % e2)
    check("...and heading rides the gyro, not the scrubbing wheels "
          "(inside 1.0 deg; wheels alone would be ~25)",
          eth < 1.0, "%.2f deg" % eth)

    # ------------------------------------------------------- stall freezing
    # x 250: WEST of the laboratory plate.  At x 600 the run crosses the
    # plate and the wheels SLIP on it rather than stall -- traction loss is
    # invisible to StallGuard on the real chip too, and the first version of
    # this check discovered that instead of what it meant to test.
    m, d, rb, clk = build(pose=(250., 400., 270.0))   # nose toward south wall
    for _ in range(int(4.0*hal.Clock.HZ)):
        rb.drive(180.0, 0.0); clk.tick()             # 720 mm commanded, wall at ~255
    rb.stop()
    e3, _ = err(rb)
    check("a wall stall FREEZES the belief instead of walking it through "
          "the wall (error under 25 mm after 500 mm of stalled command)",
          e3 < 25.0, "%.1f mm" % e3)

    # ------------------------------------------------------------ slot fix
    m, d, rb, clk = build(pose=(431.5, 158.0, 270.0), seed=3)
    est = rb._est()
    est.x += 20.0; est.y -= 15.0
    est.var_xy = 400.0                            # a belief that KNOWS it is lost
    e0 = err(rb)[0]
    rb.see_lab()                                  # one look = one fix
    e4 = err(rb)[0]
    check("one camera look pulls a 25 mm belief error under 6 mm",
          e0 > 20.0 and e4 < 6.0, "%.1f -> %.2f mm" % (e0, e4))
    check("...and the estimator counted it as a fix, not a rejection",
          est.fixes >= 1 and est.rejected == 0,
          "%d fixes, %d rejected" % (est.fixes, est.rejected))

    # -------------------------------------------------- the whole mission
    m, d, rb, clk = build(seed=1)                 # unused build; fresh below
    rng = np.random.default_rng(1)
    pts = []
    while len(pts) < 3:
        p = (rng.uniform(60, 245), rng.uniform(80, 230))
        if all(np.hypot(p[0]-q[0], p[1]-q[1]) > 90 for q in pts):
            pts.append(p)
    xml = mjcf.scene_full_match(pts, rng=rng)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    rb = AgentARobot(m, d, rng=rng, nav="est")
    gen = mission_agent_a(rb, list(Field.LAB_HOLE_X), mjcf.LAB_HOLE_Y,
                          AgentA.AXLE_X - AgentA.CHUTE_X,
                          log=lambda *a, **k: None, clock=lambda: d.time)
    errs, dock_entry = [], []
    looked = False
    while d.time < 120.0:
        try:
            next(gen)
        except StopIteration:
            break
        for _ in range(20):
            mujoco.mj_step(m, d)
        e, _ = err(rb)
        errs.append(e)
        n_fix = rb.est.fixes if rb.est else 0
        if not looked and n_fix > 0:
            looked = True
        if looked and rb.est and rb.est.fixes > len(dock_entry):
            dock_entry.append(e)
    errs = np.array(errs)
    p50, p95, mx = (np.percentile(errs, 50), np.percentile(errs, 95),
                    errs.max())
    # 25/90, set by what consumes it: transits steer on 25-40 mm waypoint
    # tolerances and every terminal closes on relative measurements.  The
    # p95 tail is the plow-slip transient and the fix-less second half, both
    # bounded by the wall datum's 120 mm gate; step 5's executor schedules
    # camera looks where the variance says they are worth their seconds,
    # which is what will tighten these.
    check("full mission: median belief error under 35 mm",
          p50 < 35.0, "median %.1f mm" % p50)
    check("full mission: 95th percentile under 90 mm",
          p95 < 90.0, "p95 %.1f mm (max %.1f)" % (p95, mx))
    check("the camera fixed the belief during the mission",
          rb.est is not None and rb.est.fixes >= 3,
          "%d fixes, %d rejected" % (rb.est.fixes if rb.est else 0,
                                     rb.est.rejected if rb.est else 0))
    e_after = np.array(dock_entry)
    check("right after a fix the belief is inside 8 mm (what the dock's "
          "step-across decision consumes)",
          len(e_after) > 0 and float(np.median(e_after)) < 8.0,
          "median %.1f mm over %d fixes" % (float(np.median(e_after))
                                            if len(e_after) else -1,
                                            len(e_after)))

    # --------------------------------------------------------------- summary
    fails = [r for r in RESULTS if not r[1]]
    for name, ok, detail in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("PASS" if ok else "FAIL", name,
                                  ("  [%s]" % detail) if detail else ""))
    print("%d checks, %d failed" % (len(RESULTS), len(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
