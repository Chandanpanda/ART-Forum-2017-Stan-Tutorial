"""Phase 0c: the HAL contract, with physics.  Run this on every change.

hal.py is the file both machines agree on; this suite proves the simulator
actually honours it.  Three kinds of check:

  * CONFORMANCE -- the sim backend implements every verb, and the verbs mean
    what the contract says (odometry integrates the quantised command over
    the time it was in force, resets on read, never blocks).
  * PHYSICS -- straight-line odometry agrees with ground truth to stepper
    accuracy, and a pivot's scrub is VISIBLE as commanded-over-actual, which
    is the estimator's process noise, not a bug.  If either stops being
    true the drive model changed and step 3's estimator design is stale.
  * THE RATCHET -- mission code's ground-truth reads are pinned at ZERO
    (step 3): no rb.d, no rb.m, no pose_truth anywhere in route.py.  .pose
    is the ESTIMATOR's belief there, and its count is pinned too so pose
    consumption is at least deliberate.  A new oracle read fails here.

    python3 sim/scripts/check_hal.py [-v]
"""
import os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mujoco
from rfgyc26 import hal, mjcf
from rfgyc26.params import Chassis
from rfgyc26.robot import AgentARobot, SimClock

VERBOSE = "-v" in sys.argv
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def build():
    xml = mjcf.scene_full_match([(2500., 2400.), (2600., 2500.), (2700., 2600.)],
                                robot_pose=(300., 660., 0.),
                                rng=np.random.default_rng(0), kits_aboard=False)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    rb = AgentARobot(m, d)
    rb.fingers(True); rb.gate(False); rb.blade(False); rb.feed(False)
    rb.cradle(1, True); rb.cradle(2, True)
    mujoco.mj_forward(m, d)
    clk = SimClock(m, d)
    for _ in range(50):                      # settle on the floor
        clk.tick()
    return m, d, rb, clk


def main():
    m, d, rb, clk = build()

    # ------------------------------------------------------------ conformance
    check("sim backend implements the full drive+device contract",
          hal.audit(rb) == [], ", ".join(hal.audit(rb)) or "complete")
    check("backend is registered against the ABCs, not duck-typed",
          isinstance(rb, hal.DriveHAL) and isinstance(rb, hal.DeviceHAL))

    check("SimClock derives the decimation from the model's own timestep",
          clk.decim == int(round(hal.Clock.PERIOD / m.opt.timestep)),
          "decim %d, timestep %.4f" % (clk.decim, m.opt.timestep))
    t0 = clk.now()
    clk.tick()
    check("one tick is one control period on the match clock",
          abs((clk.now() - t0) - hal.Clock.PERIOD) < 1e-9,
          "%.4f s" % (clk.now() - t0))

    ticks = [0]
    def gen3():
        for _ in range(3):
            ticks[0] += 1
            yield
    t0 = clk.now()
    done = clk.run(gen3())
    check("Clock.run drives a mission generator one tick per yield",
          done and ticks[0] == 3 and abs((clk.now()-t0) - 3*hal.Clock.PERIOD) < 1e-9,
          "%d yields, %.3f s" % (ticks[0], clk.now()-t0))

    # -------------------------------------------------------------- odometry
    rb.odometry()                            # zero the accumulator
    x0, y0, th0 = rb.pose_truth
    for _ in range(int(2.0 * hal.Clock.HZ)):
        rb.drive(200.0, 0.0)
        clk.tick()
    rb.stop()
    for _ in range(10):
        clk.tick()
    dl, dr = rb.odometry()
    x1, y1, th1 = rb.pose_truth
    truth = float(np.hypot(x1-x0, y1-y0))
    odo = 0.5*(dl+dr)
    check("straight run: odometry within 3% of ground truth",
          truth > 350.0 and abs(odo - truth) < 0.03*truth,
          "odo %.1f mm vs truth %.1f mm (%+.1f%%)"
          % (odo, truth, 100.0*(odo-truth)/max(truth, 1e-9)))
    check("straight run: left and right agree (no phantom yaw in the booking)",
          abs(dl - dr) < 0.02*max(odo, 1.0), "dL %.1f  dR %.1f" % (dl, dr))

    d2 = rb.odometry()
    check("odometry resets on read and books nothing while stopped",
          abs(d2[0]) < 1e-6 and abs(d2[1]) < 1e-6, "%.3g, %.3g" % d2)

    # A pivot: commanded travel over-reads the true arc (wheel scrub).  The
    # RATIO is the number the estimator's process noise is built on.
    th0 = rb.pose_truth[2]
    for _ in range(int(1.2 * hal.Clock.HZ)):
        rb.drive(0.0, 120.0)
        clk.tick()
    rb.stop()
    for _ in range(10):
        clk.tick()
    dl, dr = rb.odometry()
    dth_odo = np.degrees((dr - dl) / Chassis.TRACK)
    dth_true = (rb.pose_truth[2] - th0 + 180.0) % 360.0 - 180.0
    check("pivot: odometry heading never under-reads the true turn",
          dth_true > 30.0 and dth_odo >= 0.95*dth_true,
          "odo %.1f deg vs truth %.1f deg (efficiency %.2f)"
          % (dth_odo, dth_true, dth_true/max(dth_odo, 1e-9)))
    check("pivot: the scrub is bounded (efficiency 0.5..1.05)",
          0.5 < dth_true/max(dth_odo, 1e-9) < 1.05,
          "efficiency %.2f" % (dth_true/max(dth_odo, 1e-9)))

    # ------------------------------------------------------------ robot-2 link
    class Wire(hal.LinkHAL):
        def __init__(self): self.lines = []
        def send(self, line): self.lines.append(line)
        def recv(self): return None
    w = Wire()
    w.cmd(120.4, -120.4, 500); w.halt(); w.shake(3)
    check("LinkHAL encodes the v0 grammar exactly once, in the base class",
          w.lines == ["V 120 -120 500", "K", "SHAKE 3"], "; ".join(w.lines))
    n = hal.NullLink()
    n.cmd(0, 0, 0); n.halt(); n.shake()
    check("NullLink swallows commands and reports silence",
          n.recv() is None)

    # ------------------------------------------------------------- the ratchet
    # Step 3 landed: mission code reads NO ground truth.  .pose there is the
    # estimator's belief (nav="est"), so its count is pinned rather than
    # banned; the oracle (pose_truth, rb.d, rb.m) is banned outright.
    src = open(os.path.join(os.path.dirname(__file__), "..",
                            "rfgyc26", "route.py")).read()
    PINNED = {r"rb\.d\b": 0,          # the oracle's data -- referee/checks only
              r"rb\.m\b": 0,
              r"pose_truth": 0,       # the oracle pose, banned in mission code
              r"\.pose\b": 66,        # the map-frame BELIEF (transit; the
                                      # F148 station solver reads it ten
                                      # more times, all as planner inputs:
                                      # map origin, tie-break, path start,
                                      # arrival test)
              r"pose_odo": 6}         # the odometry frame (dock terminals)
    for pat, cap in PINNED.items():
        got = len(re.findall(pat, src))
        check("route.py ground-truth ratchet: %s <= %d" % (pat, cap),
              got <= cap, "found %d" % got)

    # THE RATCHET THAT MATTERS NOW.  Counting .pose was a proxy for how
    # scripted the mission is, and a proxy that gets raised whenever it
    # fires is not a ratchet.  The thing CLAUDE.md actually forbids is
    # measurable directly: coordinate-scale literals in the mission layers,
    # every one of which is either a fact that belongs in params or a pose
    # that belongs in a solver.  These caps only ever come DOWN.  If a
    # change needs one raised, it is adding an offender, and the offender
    # is the change.
    r2src = open(os.path.join(os.path.dirname(__file__), "..",
                              "rfgyc26", "robot2.py")).read()
    for nm, text, cap in (("route.py", src, 246), ("robot2.py", r2src, 246)):
        got = len(re.findall(r"(?<![\w.])\d{2,4}\.\d+", text))
        check("%s carries no MORE hardcoded coordinates than it did (<= %d)"
              % (nm, cap), got <= cap, "found %d" % got)

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
