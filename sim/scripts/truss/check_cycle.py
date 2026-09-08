"""Tier 1: a truss made end to end, and judged.

The 300 mm truss from empty racks to a wound, dosed, kept-in-the-fixture
truss, through schedule.plan and process.Executor against the MuJoCo
cell, with the model camera doing the pre-wind looks.  Then the
inspector says what was made.  This is the suite that can see the thing
the cell is for; the others each test a mechanism.

    python3 sim/scripts/truss/check_cycle.py [-v]
"""
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import mujoco

from truss import (structure, geometry, fixture, mjcf, cell, approach, schedule,
                   process, vision, inspector, band)
from truss.spec import Vision, Process
from truss.geometry import TrussGeometry, rot_x

VERBOSE = "-v" in sys.argv
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def main():
    t = structure.TRUSS_300
    g = TrussGeometry(t)
    fx = fixture.Fixture(g)
    st = approach.plan_truss(g, fx, span=0.0)
    P = schedule.plan(g, fx, st)
    m = mujoco.MjModel.from_xml_string(mjcf.scene_cell(g, fx, stage="empty"))
    d = mujoco.MjData(m)
    c = cell.CellSim(m, d, g, fx, rng=np.random.default_rng(3))
    clk = cell.SimClock(m, d, c)
    for _ in range(25):
        clk.tick()
    vis = vision.ModelVision(c, rng=np.random.default_rng(4))
    msgs = []
    ex = process.Executor(c, c, c, c, c, c, clk, g, fx, P, vision=vis, log=msgs.append)
    t0 = time.time()
    states = clk.run(ex.run())
    wall = time.time() - t0
    check("the whole plan runs to the end", states is not None and len(ex.timeline) == len(P.ops),
          "%d of %d ops, %.0f s wall" % (len(ex.timeline), len(P.ops), wall))
    check("no step timed out or closed on nothing", not msgs, "; ".join(msgs[:4]))
    # -------------------------------------------------------- the truss
    errs = {r.index: inspector.seat_error(g, r, *c.rod_pose(r.index), theta=c.cage_truth())
            for r in g.rods}
    rep = inspector.inspect_truss(g, states, seated=errs, cycle_s=d.time)
    check("every rod is seated within %.2f mm at the end" % inspector.SEAT_TOL,
          all(e < inspector.SEAT_TOL for e in errs.values()),
          "worst %.3f mm" % max(errs.values()))
    check("every joint carries its full band, centred, anchored and dosed",
          rep.n_ok == len(rep.joints),
          "; ".join("j%d: %s" % (j.joint, ", ".join(j.why)) for j in rep.joints if not j.ok)[:200])
    turns = [j.turns for j in rep.joints]
    check("the fiducial counted the recipe's turns on every joint",
          all(abs(n - t.turns) <= inspector.TURN_TOL for n in turns),
          "%s" % [round(n, 1) for n in turns])
    cen = [j.centre_err for j in rep.joints]
    check("every band is centred on its joint within the tolerance (the looks worked)",
          max(cen) <= inspector.CENTRE_TOL, "worst %.2f mm" % max(cen))
    check("...and the bias the looks carry is the camera's, not the gantry's",
          max(cen) <= 3.0 * np.hypot(*vis.bias) + 0.4,
          "worst %.2f vs bias %.2f" % (max(cen), np.hypot(*vis.bias)))
    # every drop landed on its joint's band: within the band's half-width
    # along the chord and on the cluster
    hits = []
    for jix, p in c.hits.items():
        j = g.joints[jix]
        cj = g.chord_point(j.chord, j.x)
        hits.append((jix, float(p[0] - cj[0]), float(np.hypot(p[1] - cj[1], p[2] - cj[2]))))
    check("every joint received its drop", len(hits) == t.n_joints,
          "%d of %d" % (len(hits), t.n_joints))
    r_drop = {j.joint: band.drop_radius_mm(j.dosed_mg) for j in rep.joints}
    check("every drop landed inside its band, on the cluster",
          all(abs(dx) <= t.band / 2.0 + 0.5 and
              rr <= g.cluster_reach(g.joints[jix], 0.0) + r_drop[jix] + 0.5
              for jix, dx, rr in hits),
          " ".join("j%d:%+.1f/%.1f" % h for h in hits)[:160])
    check("the inspector passes the truss", rep.ok, repr(rep))
    check("thread and resin on the joints stay inside the joint budget",
          rep.mass_g <= t.mass_joints * 1.05, "%.2f g of %.2f" % (rep.mass_g, t.mass_joints))
    # ------------------------------------------------------- the clock
    check("the simulated cycle keeps to the plan within 15%%",
          abs(d.time / P.total - 1.0) < 0.15, "%.0f s simulated, %.0f planned" % (d.time, P.total))
    check("nothing on the head touches the truss or the fixture at the end",
          c.contacts_between(mjcf.HEAD_B, mjcf.ROD | mjcf.CAGE) == 0)
    if VERBOSE:
        s = schedule.summary(P)
        print("  plan %.1f min by phase %s" % (s["total_min"], {k: round(v, 1) for k, v in s["phase_min"].items()}))
        for op, dt in ex.slow[:8]:
            print("  slow: %s took %.1f s" % (op, dt))
        for j in rep.joints:
            print("  joint %2d  %.1f turns  %.2f mm band  centre %+.2f  dosed %3.0f mg" %
                  (j.joint, j.turns, j.width, j.centre_err, j.dosed_mg))

    bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det and (VERBOSE or not ok) else ""))
    print("check_cycle: %d checks, %d failed" % (len(RESULTS), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
