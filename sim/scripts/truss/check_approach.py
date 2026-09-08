"""Tier 0: the approach solver -- every joint gets a station, the claims a
station makes are true, and the solver refuses what cannot be done.

    python3 sim/scripts/truss/check_approach.py [-v]
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np

from truss import structure, geometry, fixture, approach
from truss.spec import Truss, Gantry, Process, Ring
from truss.geometry import TrussGeometry, theta_chord_up, rot_x

VERBOSE = "-v" in sys.argv
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def solve(t, tag, **kw):
    g = TrussGeometry(t)
    fx = fixture.Fixture(g)
    t0 = time.time()
    st = approach.plan_truss(g, fx, **kw)
    el = time.time() - t0
    ok = all(v[1] is not None for v in st.values())
    check("%s: every chord has a station for every joint (%.0f s)" % (tag, el), ok)
    if not ok:
        return g, fx, st
    aps = [a for _th, al in st.values() for a in al]
    check("%s: every station has positive margin" % tag,
          all(a.margin > 0.0 for a in aps),
          "least %.2f mm" % min(a.margin for a in aps))
    check("%s: seated, the ring keeps the design clearance off the cluster" % tag,
          all(a.seated >= Process.SEAT_CLEAR - 0.2 for a in aps),
          "least %.2f" % min(a.seated for a in aps))
    check("%s: the lift between joints is small -- millimetres, not the Z stroke" % tag,
          all(0.0 < a.lift < Gantry.Z_TRAVEL / 4.0 for a in aps),
          "lifts %s" % sorted(set(round(a.lift) for a in aps)))
    for k, (th, al) in st.items():
        zs = [float((rot_x(th) @ g.chord_point(m, 0.0))[2]) for m in range(t.n_chords)]
        check("%s chord %d: the chosen cage angle puts the chord on top" % (tag, k),
              zs[k] == max(zs), "theta %.0f" % th)
        gaps = [a.gap_down for a in al]
        check("%s chord %d: the parked gap alternates with the joints' faces" % (tag, k),
              all(gaps[i] * gaps[i + 1] < 0 for i in range(len(gaps) - 1)),
              str([round(x) for x in gaps]))
    check("%s: the cage can index under the head within the Z stroke" % tag,
          approach.index_lift(g, fx) + Ring.OD / 2.0 <= Gantry.Z_TRAVEL,
          "%.0f mm" % approach.index_lift(g, fx))
    # THE CLAIMS ARE TRUE: re-measure one station with the rods sampled
    # four times as densely, so a claim that leaned on a sampling hole
    # would fall
    th, al = st[0]
    a = al[1]
    obs = approach.Obstacles(g, fx, th, pitch=0.25)
    seat = approach.seated_clearance(obs, a.centre)
    check("%s: a dense re-sample of the seated ring agrees with the claim" % tag,
          abs(seat - a.seated) < 0.15, "%.2f claimed, %.2f dense" % (a.seated, seat))
    desc = approach.descent_clearance(obs, a.centre, a.lift, gap_az=a.gap_down, step=0.25)
    check("%s: ...and so does a fine descent" % tag,
          desc >= a.descent - 0.15, "%.2f claimed, %.2f fine" % (a.descent, desc))
    # the reason for gap_azimuth: gap straight down is worse
    down = approach.descent_clearance(obs, a.centre, a.lift, gap_az=0.0, step=0.25)
    check("%s: parking the gap toward the face beats straight down" % tag,
          desc >= down - 1e-6, "toward face %.2f, straight down %.2f" % (desc, down))
    # and the point-cloud picture of the head sits inside the analytic one
    cloud = approach.head_points(True, n_az=72)
    d = approach.head_distance(cloud, True)
    check("%s: the head's own surface points measure zero distance to it" % tag,
          float(d.max()) < 1e-6, "%.3g" % float(d.max()))
    return g, fx, st


def main():
    solve(structure.TRUSS_1M, "1m")
    solve(structure.TRUSS_300, "300")
    # a truss it has never seen
    other = Truss(length=600.0, side=70.0, alpha=50.0, d_chord=2.0, d_diag=1.5, name="other")
    solve(other, "other")
    # ---------------------------------------------------- refusals
    steep = Truss(length=400.0, side=60.0, alpha=55.0, d_chord=3.0, d_diag=2.0, name="steep")
    g = TrussGeometry(steep)
    fx = fixture.Fixture(g)
    j = g.joints_on(0)[1]
    a = approach.station(g, fx, j, theta_chord_up(0))
    check("a diagonal too steep for the rim is refused, not seated anyway",
          a is None, repr(a))
    tiny = Truss(length=300.0, side=30.0, alpha=45.0, d_chord=2.0, d_diag=1.0, name="tiny")
    g = TrussGeometry(tiny)
    fx = fixture.Fixture(g)
    a = approach.station(g, fx, g.joints_on(0)[1], theta_chord_up(0))
    check("a section narrower than the ring's sweep is refused", a is None, repr(a))

    bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det and (VERBOSE or not ok) else ""))
    print("check_approach: %d checks, %d failed" % (len(RESULTS), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
