"""Phase 0h: the map and the planners.  Run on every change.

nav.py is what replaced the project's hand-picked waypoints, so it gets the
same treatment every other load-bearing module got: assertions that state
what it must do, not what it happens to do today.

    python3 sim/scripts/check_nav.py [-v]
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from rfgyc26 import nav
from rfgyc26.params import Field

VERBOSE = "-v" in sys.argv
RESULTS = []
R2_IN, R2_CIRC = 55.0, 93.0


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def main():
    cm = nav.CostMap.field()
    grid = cm.inflated(R2_IN, R2_CIRC)
    check("the field map is the right shape at 20 mm",
          (cm.nx, cm.ny) == (58, 60), "%dx%d" % (cm.nx, cm.ny))

    # the plate is an obstacle, and it is the RIGHT obstacle
    i, j = cm.cell(570.0, 435.0)
    check("the laboratory plate is a hard obstacle (F11: nothing climbs 5 mm)",
          grid[i, j] >= nav.BLOCKED)
    i, j = cm.cell(570.0, 700.0)
    check("...and the field north of it is free",
          grid[i, j] < nav.BLOCKED)

    # walls block, and inflation keeps a body-width off them
    i, j = cm.cell(20.0, 600.0)
    check("the west wall blocks a robot centre 20 mm from it",
          grid[i, j] >= nav.BLOCKED)

    # A ROUTE AROUND THE PLATE, computed not typed
    p, secs = nav.plan(cm, (570.0, 205.0), (570.0, 700.0), R2_IN, R2_CIRC)
    xs = [q[0] for q in p] if p else []
    check("a straight-line-blocked leg routes AROUND the plate",
          p is not None and (min(xs) < 340.0 or max(xs) > 800.0),
          "x %.0f..%.0f" % (min(xs), max(xs)) if p else "no path")

    # planning is fast enough to be done live
    t = time.time()
    for _ in range(20):
        nav.plan(cm, (100.0, 100.0), (1050.0, 1100.0), R2_IN, R2_CIRC)
    ms = (time.time() - t) / 20.0 * 1000.0
    check("a corner-to-corner plan costs under 30 ms", ms < 30.0,
          "%.1f ms" % ms)

    # SPACE-TIME: the same leg is refused during a reservation, allowed after
    cm2 = nav.CostMap.field()
    cm2.add_corridor([(950.0, 250.0), (940.0, 655.0)], 185.0, 58.0, 73.0)
    grid2 = cm2.inflated(R2_IN, R2_CIRC)
    a = nav.astar(grid2, (1000.0, 150.0), (1000.0, 900.0), cm2.res,
                  t0=60.0, speed=300.0, cmap=cm2)
    b = nav.astar(grid2, (1000.0, 150.0), (1000.0, 900.0), cm2.res,
                  t0=90.0, speed=300.0, cmap=cm2)
    check("a reserved corridor blocks a path that would be there during it",
          a is None)
    check("...and does not block one that would arrive after it",
          b is not None)

    # the strict/soft split: work is refused, escape is not
    p_strict, _ = nav.plan(cm2, (1000.0, 400.0), (1000.0, 900.0), R2_IN,
                           R2_CIRC, t0=60.0, speed=300.0, strict=True)
    p_soft, _ = nav.plan(cm2, (1000.0, 400.0), (1000.0, 900.0), R2_IN,
                         R2_CIRC, t0=60.0, speed=300.0)
    check("strict planning refuses to enter a reservation to do work",
          p_strict is None)
    check("...but a robot standing inside one can still be told to leave",
          p_soft is not None)

    # a blocked START is normal (the robot finishes a push nose-to-puck)
    cm3 = nav.CostMap.field()
    cm3.add_disc(600.0, 600.0, 12.0)
    g3 = cm3.inflated(R2_IN, R2_CIRC)
    i, j = cm3.cell(600.0, 600.0)
    p3, _ = nav.plan(cm3, (600.0, 600.0), (300.0, 300.0), R2_IN, R2_CIRC)
    check("a plan out of a blocked start cell still exists",
          g3[i, j] >= nav.BLOCKED and p3 is not None)

    # ------------------------------------------------------- push planning
    cm4 = nav.CostMap.field()
    HOSP = (511.0, 941.0, 631.0, 1141.0)
    legs, secs = nav.plan_push(cm4, (160.0, 763.0), HOSP, robot=(160.0, 600.0))
    check("a west patient reaches the hospital in at most three pushes",
          legs is not None and len(legs) <= 3,
          "%s legs, %.1f s" % (len(legs) if legs else "no", secs))
    if legs:
        lx, ly = legs[-1]
        check("...and the last leg actually lands it inside the zone",
              HOSP[0] <= lx <= HOSP[2] and HOSP[1] <= ly <= HOSP[3],
              "(%.0f, %.0f)" % (lx, ly))

    # a patient walled into a corner by others is honestly refused
    cm5 = nav.CostMap.field()
    for dx, dy in ((0, 60), (60, 0), (60, 60), (0, -60), (-60, 0)):
        cm5.add_disc(80.0 + dx, 600.0 + dy, 12.0)
    legs5, _ = nav.plan_push(cm5, (80.0, 600.0), HOSP, robot=(300.0, 600.0))
    check("an unreachable patient is refused, not improvised",
          legs5 is None, "legs %s" % (legs5,))

    t = time.time()
    nav.plan_push(nav.CostMap.field(), (983.0, 763.0),
                  (983.0, 1021.0, 1103.0, 1141.0), robot=(900.0, 700.0))
    check("a push plan costs under 1.5 s (the opening plan has 12 to make)",
          time.time() - t < 1.5, "%.2f s" % (time.time() - t))

    fails = [r for r in RESULTS if not r[1]]
    for name, ok, detail in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("PASS" if ok else "FAIL", name,
                                  ("  [%s]" % detail) if detail else ""))
    print("%d checks, %d failed" % (len(RESULTS), len(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
