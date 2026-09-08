"""Tier 2, slow: one joint wound with a real thread (truss.rig).

The band arithmetic Tier 1 plans and counts with -- a turn is a hoop of
the cluster's perimeter, laid where the feed had reached, at the recipe's
pitch -- is measured here against a cable wound by physics on the 300 mm
truss's second joint: turns wrapped, thread consumed, where the hoops
lie and how tight, whether the diagonals moved, and the question the
joint exists to answer: with the keeper released and the joint turned
face down, does the wound band alone hold the mitred diagonals?  A
control winds nothing and lets go.

    python3 sim/scripts/truss/check_ring.py [-v]
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np

from truss import structure, fixture, rig, inspector
from truss.geometry import TrussGeometry

VERBOSE = "-v" in sys.argv
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def main():
    t = structure.TRUSS_300
    g = TrussGeometry(t)
    fx = fixture.Fixture(g)
    j = g.joints_on(0)[1]
    R = rig.JointRig(g, fx, j, turns=4)
    check("the rig compiles: a cable of %d segments on the joint's cluster" % len(R.cable),
          R.m.nbody > 10 and len(R.cable) > 20, "nbody %d" % R.m.nbody)
    R.settle(0.3)
    e0 = R.seat_errors()
    check("the diagonals sit where the fixture put them before the wind",
          max(e0.values()) < inspector.SEAT_TOL, "%.3f mm" % max(e0.values()))
    yielded = R.keeper_yield()
    check("the keeper is a clamp here: it yields under the winding tension by less than "
          "the placement tolerance", yielded < inspector.SEAT_TOL, "%.3f mm under %.1f N"
          % (yielded, rig.RIG_TENSION))
    per_step = R.wind()
    check("a rig step costs single-digit milliseconds", per_step < 0.01, "%.1f ms" % (per_step * 1e3))
    wrapped, xs, rho = R.hoops()
    check("the spindle made the turns it was given", abs(R.spun_turns() - R.turns) < 0.15,
          "%.2f of %d" % (R.spun_turns(), R.turns))
    check("the thread wrapped the cluster once per turn", abs(wrapped - R.turns) < 0.5,
          "%.2f turns of thread for %d of spindle" % (wrapped, R.turns))
    band = (R.x_start - R.pitch, R.x_start + R.turns * R.pitch + R.pitch)
    inside = np.mean((xs >= band[0]) & (xs <= band[1])) if len(xs) else 0.0
    check("the hoops lie in the band the feed described", inside > 0.9,
          "%.0f%% in %.1f..%.1f; span %.1f..%.1f" % (100 * inside, band[0], band[1],
                                                     xs.min() if len(xs) else 0, xs.max() if len(xs) else 0))
    # ------------------------------------------------- WHAT A HOOP GOES ROUND
    # The band arithmetic first said a hoop is the cluster's convex hull
    # wherever it lies.  This is the measurement that refuted it.
    rad = R.hoop_radii()
    d_t = 2.0 * R.r_cable
    lo = np.abs(rad[:, 0]) > t.bind_half(d_t) + 1.0
    hi = np.abs(rad[:, 0]) < t.bind_half(d_t) - 0.3
    on_chord = t.d_chord / 2.0 + R.r_cable
    check("outside the binding width the thread beds on the CHORD, under the diagonals -- "
          "it does not ride their hull",
          lo.sum() > 5 and abs(float(np.median(rad[lo, 1])) - on_chord) < 0.25,
          "median radius %.2f for a chord+thread of %.2f, where the hull reaches %.2f"
          % (float(np.median(rad[lo, 1])) if lo.sum() else -1, on_chord,
             float(np.max(rad[lo, 2])) if lo.sum() else -1))
    check("...and the closed form says which hoops those are (Truss.bind_half)",
          not hi.sum() or float(np.median(rad[hi, 1])) >= float(np.median(rad[lo, 1])) - 0.1,
          "bind half %.2f mm of a %.1f mm band; inside median %.2f, outside %.2f"
          % (t.bind_half(d_t), t.band, float(np.median(rad[hi, 1])) if hi.sum() else -1,
             float(np.median(rad[lo, 1])) if lo.sum() else -1))
    arc = R.arc_on_cluster()
    want = R.thread_expected()
    check("the thread the hoops took is what the corrected arithmetic says, within 15%%",
          abs(arc / want - 1.0) < 0.15,
          "%.1f mm on the cluster, %.1f by hoop_perimeter (the hull everywhere would say %.1f)"
          % (arc, want, sum(g.cluster_perimeter(j, float(x)) for x in
                            R.x_start + (np.arange(R.turns) + 0.5) * R.pitch)))
    e1 = R.seat_errors()
    # THE CONTROL FOR "DID THE THREAD MOVE IT": a bare joint turned through
    # the same angle.  Four turns leave the joint at a different attitude,
    # and a rod hanging in its keeper at that attitude is displaced by
    # gravity alone -- which is most of what the wound rig shows.
    ang = np.degrees(R.angle()) % 360.0
    C = rig.JointRig(g, fx, j, turns=4, thread=False)
    C.settle(0.3)
    c0 = C.seat_errors()
    C.turn_to(float(ang), seconds=2.0)
    C.settle(0.5)
    c1 = C.seat_errors()
    moved = max(e1.values()) - max(e0.values())
    turned = max(c1.values()) - max(c0.values())
    check("the winding moves the diagonals no further than turning them there does: the thread "
          "does not drag a clamped rod out of its seat",
          moved < turned + inspector.SEAT_TOL,
          "%.3f mm wound, %.3f mm by the same rotation bare" % (moved, turned))
    # ---------------------------------- WHAT THE BAND DOES NOT DO
    # The keeper is released and the joint turned over, wound and bare.
    R.release_keepers()
    R.settle(0.5)
    R.turn_to(180.0, seconds=2.0)
    R.settle(1.0)
    e2 = R.seat_errors()
    C.release_keepers()
    C.settle(0.5)
    C.turn_to(180.0, seconds=2.0)
    C.settle(1.0)
    e3 = C.seat_errors()
    check("a band of dry thread does NOT retain the mitres: released and turned face down the "
          "diagonals leave, wound or bare -- the keeper holds until the resin cures",
          max(e2.values()) > 5.0 and max(e3.values()) > 5.0,
          "wound %.0f mm off, bare %.0f mm" % (max(e2.values()), max(e3.values())))
    if VERBOSE:
        print("  wrapped %.2f turns, %.1f mm on the cluster (%.1f expected), pay-out %.1f mm" %
              (wrapped, arc, want, R.pay_out()))
        print("  hoops x %.2f..%.2f, radius median %.3f, keeper yield %.3f mm" %
              (xs.min(), xs.max(), float(np.median(rho)), yielded))
        print("  seat errors: before %s, wound %s, released face down %s, bare %s" %
              ({k: round(v, 3) for k, v in e0.items()}, {k: round(v, 3) for k, v in e1.items()},
               {k: round(v, 1) for k, v in e2.items()}, {k: round(v, 1) for k, v in e3.items()}))

    bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det and (VERBOSE or not ok) else ""))
    print("check_ring: %d checks, %d failed" % (len(RESULTS), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
