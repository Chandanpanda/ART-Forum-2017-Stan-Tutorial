"""EVERY SPINE CHECK, IN ONE COMMAND.  Run before and after any change to
sim/spine.

    python3 sim/scripts/spine/check_all.py

Separate from the truss cell's suite because it answers a separate
question: that one asks whether the machine can build a spine, this one
whether the spine is worth building.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = [
    ("check_frame", "the element and the solver against closed forms"),
    ("check_spine", "the lattices, the tube, and what the metrics mean"),
]


def main():
    verbose = "-v" in sys.argv
    print("spine checks\n")
    t0 = time.time()
    bad = []
    for name, _why in SUITES:
        path = os.path.join(HERE, name + ".py")
        p = subprocess.run([sys.executable, path], capture_output=True, text=True)
        tail = [l for l in (p.stdout or "").strip().splitlines() if l.strip()]
        last = tail[-1] if tail else ((p.stderr or "").strip().splitlines() or ["?"])[-1]
        ok = p.returncode == 0 and "FAIL" not in (p.stdout or "")
        print("  %s %-14s %s" % ("ok  " if ok else "FAIL", name, last))
        if not ok or verbose:
            for line in (p.stdout or "").splitlines():
                if "FAIL" in line or verbose:
                    print("        " + line)
            if p.stderr.strip():
                print("        stderr: " + p.stderr.strip().splitlines()[-1])
        if not ok:
            bad.append(name)
    print("\n%d suites in %.0f s -- %s"
          % (len(SUITES), time.time() - t0, "all pass" if not bad else "FAILED: " + ", ".join(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
