"""EVERY TRUSS-CELL CHECK, IN ONE COMMAND.  Run this before and after any
change to sim/truss.

    python3 sim/scripts/truss/check_all.py          # the fast tier
    python3 sim/scripts/truss/check_all.py --slow   # + the cable rig and rendered vision

Same discipline as sim/scripts/check_all.py: one entry, no choosing which
suite is relevant.  The fast tier is Tier 0 arithmetic and Tier 1 rigid
physics; the slow tier is the Tier 2 thread rig and the rendered camera.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

FAST = [
    ("check_geometry",  "the spec's assertions and the truss re-derived from it"),
    ("check_structure", "the beam model against the brief; the optimiser's answer"),
    ("check_approach",  "a station for every joint, with a measured margin"),
    ("check_schedule",  "the plan: every rod, every joint, inside the clock"),
    ("check_model",     "the built MuJoCo cell against the parameters"),
    ("check_hal",       "the HAL contract and the axes' honesty"),
    ("check_view",      "the viewer's camera: the mouse is live, the keys return"),
    ("check_load",      "the loader: capture, seating, retention"),
    ("check_cycle",     "a truss made end to end, judged by the inspector"),
]
SLOW = [
    ("check_drive",     "the ring's drive: a free ring, real teeth, one belt"),
    ("check_ring",      "one joint wound with a real thread"),
    ("check_vision",    "the chord found in rendered frames"),
]


def run(name, why, verbose):
    path = os.path.join(HERE, name + ".py")
    if not os.path.exists(path):
        print("  ??   %-16s missing" % name)
        return None
    t0 = time.time()
    env = dict(os.environ)
    # each suite names its own context through truss.glenv; osmesa is a
    # Linux-only name and setting it here breaks the suites on Windows
    p = subprocess.run([sys.executable, path], capture_output=True, text=True, env=env)
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
    print("truss cell checks (%s tier)\n" % ("full" if "--slow" in sys.argv else "fast"))
    t0 = time.time()
    results = [(n, run(n, w, verbose)) for n, w in suites]
    bad = [n for n, ok in results if ok is False]
    print("\n%d suites in %.0f s -- %s"
          % (len(results), time.time() - t0,
             "all pass" if not bad else "FAILED: " + ", ".join(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
