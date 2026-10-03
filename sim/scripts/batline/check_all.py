"""EVERY BATLINE CHECK, IN ONE COMMAND.  Run this before and after any
change to sim/batline.

    python3 sim/scripts/batline/check_all.py          # every suite there is
    python3 sim/scripts/batline/check_all.py -v       # and every line of each

Same discipline as sim/scripts/check_all.py and the truss cell's: one
entry, no choosing which suite is relevant.  M0 has two suites, M1 one
(check_gantry), M2 one (check_cameras), M3 two (check_imu,
check_calibration) and M4 two (check_link, Tier 0, and check_swing, the
plan's first Tier 2 rig), several minutes on four cores, and they run
every time.  The plan put the swing among the slow suites; it runs in
under a minute, so it runs with the rest.  The cloth and the rollers
join SLOW when their rigs exist, if they need it.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

FAST = [
    ("check_bat",  "the bat derived from its spec, two sizes, and built in MuJoCo"),
    ("check_line", "the line sized from the bat, and a month of it at 300 and 1,000"),
    ("check_gantry", "the station module: repeatability, lost steps, commissioning and every op, "
                     "on two travels"),
    ("check_cameras", "the calibration station's rig: placement, self-calibration, pose, drift and colour, "
                      "on two bats, model and rendered cameras"),
    ("check_imu", "the simulated IMU against the physics in known motions: the gimbal's dynamics, "
                  "the part's errors, its rails and its noise"),
    ("check_calibration", "the IMU's calibration: the plan, the fit on 100 bats and an unseen size, "
                          "rejection, the record, the golden bat's daily check"),
    ("check_link", "the bat's board: power, firmware and link, station 2's self-test, the record over the "
                   "air, the clock, the factory iPhone's test"),
    ("check_swing", "the swing rig: the arm fitted to the rated swing, its motors, the sprung pack in "
                    "MuJoCo, no rail and no open past the hold-up, the station's test, the phone's picture"),
]
SLOW = []


def run(name, why, verbose):
    path = os.path.join(HERE, name + ".py")
    if not os.path.exists(path):
        print("  ??   %-16s missing" % name)
        return None
    t0 = time.time()
    p = subprocess.run([sys.executable, path] + (["-v"] if verbose else []),
                       capture_output=True, text=True, env=dict(os.environ))
    el = time.time() - t0
    tail = [l for l in (p.stdout or "").strip().splitlines() if l.strip()]
    last = tail[-1] if tail else ((p.stderr or "").strip().splitlines() or ["?"])[-1]
    ok = p.returncode == 0 and "FAIL" not in (p.stdout or "")
    print("  %s %-16s %5.1fs  %s" % ("ok  " if ok else "FAIL", name, el, last))
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
    print("batline checks\n")
    t0 = time.time()
    results = [(n, run(n, w, verbose)) for n, w in suites]
    bad = [n for n, ok in results if ok is False]
    print("\n%d suites in %.0f s -- %s"
          % (len(results), time.time() - t0,
             "all pass" if not bad else "FAILED: " + ", ".join(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
