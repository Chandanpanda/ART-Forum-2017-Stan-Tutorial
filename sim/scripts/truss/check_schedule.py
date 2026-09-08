"""Tier 0: the plan -- every rod placed, every joint wound and dosed,
inside the travels and inside the clock, and the closed-form estimate
the optimiser used still tracks it.

    python3 sim/scripts/truss/check_schedule.py [-v]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np

from truss import structure, geometry, fixture, approach, schedule, motion
from truss.spec import Gantry, Head
from truss.geometry import TrussGeometry, theta_face_up, theta_chord_up

VERBOSE = "-v" in sys.argv
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def run(t, tag):
    g = TrussGeometry(t)
    fx = fixture.Fixture(g)
    st = approach.plan_truss(g, fx, span=0.0)
    P = schedule.plan(g, fx, st)
    s = schedule.summary(P)
    check("%s: every joint is wound once" % tag, s["joints"] == t.n_joints,
          "%d of %d" % (s["joints"], t.n_joints))
    check("%s: every rod is placed once" % tag, s["rods"] == t.n_chords + t.n_diag)
    check("%s: every joint is dosed" % tag, P.count("dose") == t.n_joints)
    check("%s: six loading indexes and one per chord run" % tag,
          s["indexes"] >= 2 * t.n_chords, "%d" % s["indexes"])
    check("%s: the cycle is inside the 45 min budget" % tag, s["total_min"] < 45.0,
          "%.1f min" % s["total_min"])
    check("%s: phases account for the total" % tag,
          abs(sum(s["phase_min"].values()) - s["total_min"]) < 1e-6)
    check("%s: no op runs backwards or overlaps" % tag,
          all(o.dt >= 0 for o in P.ops) and
          all(abs(P.ops[i].t1 - P.ops[i + 1].t0) < 1e-6 for i in range(len(P.ops) - 1)))
    # travels
    moves = [o for o in P.ops if o.kind == "move"]
    xs = [o.args["pos"]["x"] for o in moves]
    ys = [abs(o.args["pos"]["y"]) for o in moves]
    zs = [o.args["pos"]["z"] for o in moves]
    check("%s: X stays inside the travel" % tag,
          max(xs) - min(xs) <= Gantry.X_TRAVEL, "%.0f..%.0f" % (min(xs), max(xs)))
    check("%s: Y stays inside the travel" % tag, max(ys) <= Gantry.Y_TRAVEL / 2.0,
          "%.0f" % max(ys))
    check("%s: Z stays inside the travel" % tag,
          max(zs) - min(zs) <= Gantry.Z_TRAVEL, "%.0f..%.0f" % (min(zs), max(zs)))
    # every move's duration is the axes' own trapezoid
    ok = True
    prev = None
    for o in P.ops:
        if o.kind == "move":
            if prev is not None:
                want = motion.coordinated_time(
                    {k: o.args["pos"][k] - prev[k] for k in ("x", "y", "z")},
                    Gantry.V_MAX, Gantry.A_MAX) + Gantry.SETTLE_S
                ok &= abs(want - o.dt) < 1e-6
            prev = dict(o.args["pos"])
        elif o.kind == "wind":
            prev["x"] += o.args["feed"]
    check("%s: every move takes what its trapezoid says" % tag, ok)
    # loading order and orientation
    rel = [o for o in P.ops if o.kind == "release"]
    chords_first = all(g.rods[o.args["rod"]].kind == "chord" for o in rel[:t.n_chords])
    check("%s: chords load before diagonals" % tag, chords_first)
    ok = True
    theta = 0.0
    for o in P.ops:
        if o.kind == "index":
            theta = o.args["theta"]
        if o.kind == "release":
            r = g.rods[o.args["rod"]]
            want = theta_chord_up(r.chord) if r.kind == "chord" else theta_face_up(r.face)
            ok &= abs(((theta - want) + 180.0) % 360.0 - 180.0) < 1e-6
    check("%s: each rod is released at its own loading angle" % tag, ok)
    # the dose follows its wind
    kinds = [o.kind for o in P.ops]
    ok = True
    for i, o in enumerate(P.ops):
        if o.kind == "wind":
            nxt = [q.kind for q in P.ops[i + 1:i + 12]]
            ok &= "dose" in nxt
    check("%s: every wind is followed by its dose within a few ops" % tag, ok)
    est = structure.cycle_estimate(t)
    check("%s: the optimiser's closed-form estimate tracks the plan within 30%%" % tag,
          abs(est / s["total_min"] - 1.0) < 0.30,
          "estimate %.1f, plan %.1f min" % (est, s["total_min"]))
    P2 = schedule.plan(g, fx, st, interleave=False)
    check("%s: dosing joint by joint is no slower than a separate pass" % tag,
          P.total <= P2.total + 1e-6,
          "%.1f vs %.1f min" % (P.total / 60.0, P2.total / 60.0))
    return P, s


def main():
    P, s = run(structure.TRUSS_1M, "1m")
    if VERBOSE:
        print("  1m: %.1f min, by phase %s" % (s["total_min"],
              {k: round(v, 1) for k, v in s["phase_min"].items()}))
        print("      by kind %s" % {k: round(v, 1) for k, v in s["kind_min"].items()})
    run(structure.TRUSS_300, "300")
    bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det and (VERBOSE or not ok) else ""))
    print("check_schedule: %d checks, %d failed" % (len(RESULTS), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
