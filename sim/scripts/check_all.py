"""EVERY MODEL CHECK, IN ONE COMMAND.  Run this before and after any change.

    python3 sim/scripts/check_all.py          # the fast tier, ~30 s
    python3 sim/scripts/check_all.py --slow   # everything, including render

The point of a single entry is that there is no decision to make about
which suites are relevant -- deciding that is how a change to params.py
ships without anyone running the suite that would have caught it.

The fast tier is everything that does not render or simulate a whole
match.  It is short enough that it costs nothing to run it every time, and
it covers the questions that have actually gone wrong here: is the board
what we think it is, does each robot's geometry match its parameters, and
does each effector put its payload where the planners believe it does.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# name, why it exists -- ordered cheapest first so a broken model fails fast
FAST = [
    ("check_geometry",  "params re-derived against each other, no physics"),
    ("check_board",     "the field, the pieces, the referee's zones"),
    ("check_model",     "the built model's geometry, with physics"),
    ("check_effectors", "where each effector's payload actually lands"),
    ("check_r2_pocket", "robot 2's capture pocket and gate"),
    ("check_delivery",  "one patient, end to end, and the referee's verdict"),
    ("check_station",   "the pose solver, incl. a field it has never seen"),
    ("check_morphology", "the chassis outline as a packing + board problem"),
    ("check_nav",       "costmap, A*, space-time windows"),
    ("check_planner",   "the task DP against exhaustive enumeration"),
    ("check_trajectory", "the pure-pursuit tracker"),
    ("check_hal",       "the HAL contract and robot 1's drivetrain"),
]
SLOW = [
    ("check_estimator", "the belief against the oracle, over a full match"),
    ("check_perception", "the pixel pipeline against rendered truth"),
]


def run(name, why, verbose):
    path = os.path.join(HERE, name + ".py")
    if not os.path.exists(path):
        print("  ??  %-18s missing" % name)
        return None
    t0 = time.time()
    p = subprocess.run([sys.executable, path], capture_output=True, text=True)
    el = time.time() - t0
    tail = [l for l in (p.stdout or "").strip().splitlines() if l.strip()]
    last = tail[-1] if tail else (p.stderr or "").strip().splitlines()[-1:]
    ok = p.returncode == 0 and "FAIL" not in (p.stdout or "")
    print("  %s %-18s %5.1fs  %s" % ("ok  " if ok else "FAIL", name, el,
                                     last if isinstance(last, str) else last))
    if not ok or verbose:
        for line in (p.stdout or "").splitlines():
            if "FAIL" in line or verbose:
                print("        " + line)
        if p.stderr.strip():
            print("        stderr: " + p.stderr.strip().splitlines()[-1])
    return ok


def main():
    verbose = "-v" in sys.argv
    suites = FAST + (SLOW if "--slow" in sys.argv else [])
    print("model checks (%s tier)\n" % ("full" if "--slow" in sys.argv else "fast"))
    t0 = time.time()
    results = [(n, run(n, w, verbose)) for n, w in suites]
    bad = [n for n, ok in results if ok is False]
    print("\n%d suites in %.0f s -- %s"
          % (len(results), time.time() - t0,
             "all pass" if not bad else "FAILED: " + ", ".join(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
