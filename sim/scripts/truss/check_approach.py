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
from truss.spec import Truss, Gantry, Process, Ring, Head, Cage
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
    seat = approach.seated_clearance(obs, a.centre, band=t.band)
    check("%s: a dense re-sample of the seated ring agrees with the claim" % tag,
          abs(seat - a.seated) < 0.15, "%.2f claimed, %.2f dense" % (a.seated, seat))
    desc = approach.descent_clearance(obs, a.centre, a.lift, gap_az=a.gap_down, step=0.25,
                                      band=t.band)
    check("%s: ...and so does a fine descent" % tag,
          desc >= a.descent - 0.15, "%.2f claimed, %.2f fine" % (a.descent, desc))
    # the reason for gap_azimuth: gap straight down is worse -- and the
    # band's ends are where it is worst, which is why the station measures
    # its descent there (at the joint's centre alone, straight down wins
    # by 0.2 mm on this truss and loses by 1.3 mm at the band's end)
    down = approach.descent_clearance(obs, a.centre, a.lift, gap_az=0.0, step=0.25, band=t.band)
    check("%s: parking the gap toward the face beats straight down" % tag,
          desc >= down - 1e-6, "toward face %.2f, straight down %.2f" % (desc, down))
    # and the point-cloud picture of the head sits inside the analytic one
    cloud = approach.head_points(True, n_az=72)
    d = approach.head_distance(cloud, True)
    check("%s: the head's own surface points measure zero distance to it" % tag,
          float(d.max()) < 1e-6, "%.3g" % float(d.max()))
    # THE HELD ROD AND THE HEAD.  Retracted, a rod sits beside the ring's
    # plate: parallel to it (a diagonal from its rack) or through its bore
    # (a chord); turned to its placing yaw while retracted it is in the
    # rim, which is why the plan turns it with the gripper out
    half_d, half_c = t.L_cut / 2.0, t.length / 2.0
    carried = min(approach.held_rod_head_clearance(half_d, t.d_diag / 2.0, 90.0, 0.0),
                  approach.held_rod_head_clearance(half_c, t.d_chord / 2.0, 0.0, 0.0))
    check("%s: a rod carried retracted at its rack yaw clears the head" % tag,
          carried >= Process.SEAT_CLEAR, "%.2f mm" % carried)
    struck = max(approach.held_rod_head_clearance(half_d, t.d_diag / 2.0, y, 0.0)
                 for y in (t.alpha, -t.alpha))
    check("%s: ...and turned to its placing yaw retracted it would be IN the rim -- "
          "the yaw waits for the extension" % tag, struck < 0.0, "%.2f mm" % struck)
    out = min(approach.held_rod_head_clearance(half_d, t.d_diag / 2.0, y, Head.GRIP_STROKE)
              for y in approach.yaw_sweep(90.0, -t.alpha))
    check("%s: with the gripper out it clears the head through the whole sweep" % tag,
          out >= Process.SEAT_CLEAR, "%.2f mm" % out)
    # and over the fixture, every diagonal has a height to turn at, a few
    # millimetres over its seat rather than the stroke
    cruise = approach.index_lift(g, fx)
    rise = []
    for r in g.diags:
        th = fx.theta_for_loading(r)
        obs = approach.Obstacles(g, fx, th, skip=(r.index,))
        place, yaw, _ = fx.place_pose(r, th)
        z = approach.yaw_height(obs, place, 90.0, yaw, half_d, r.r, Process.SEAT_CLEAR,
                                z_max=cruise, stroke=Head.GRIP_STROKE)
        rise.append(None if z is None else z - float(place[2]))
    check("%s: every diagonal can be turned over its cradles, millimetres above its seat" % tag,
          all(h is not None and 0.0 < h < Head.GRIP_STROKE / 2.0 for h in rise),
          "rises %s" % sorted(set(round(h, 1) for h in rise if h is not None)))
    return g, fx, st


def main():
    solve(structure.TRUSS_1M, "1m")
    solve(structure.TRUSS_300, "300")
    # a truss it has never seen
    other = Truss(length=600.0, side=70.0, alpha=40.0, d_chord=2.0, d_diag=1.5, name="other")
    solve(other, "other")
    # THE CORNER FINDING: a 50-degree diagonal with 1.9 mm of radial slack
    # under the rim clears the rim's corner by only 1.9 cos(50), and the
    # closed form must say so
    steep50 = Truss(length=600.0, side=70.0, alpha=50.0, d_chord=2.0, d_diag=1.5)
    g = TrussGeometry(steep50)
    fx = fixture.Fixture(g)
    a = approach.station(g, fx, g.joints_on(0)[1], theta_chord_up(0))
    from truss.spec import r_in_needed, ring_r_in
    # the diagonals against the rim alone, at the joint's centre: with the
    # fixture in and across the band the station's own number is smaller
    jj = g.joints_on(0)[1]
    alone = approach.Obstacles(g, None, theta_chord_up(0),
                               skip=[r.index for r in g.rods if r.index not in jj.diags])
    seat_c = approach.seated_clearance(alone, a.centre) if a is not None else -1.0
    check("a diagonal passing the rim's corner obliquely clears it by the slack times cos(alpha)",
          a is not None and abs(seat_c - (ring_r_in() - g.cluster_reach(g.joints_on(0)[1], Ring.W / 2.0))
                                * np.cos(np.radians(50.0))) < 0.1,
          "%.2f measured at the centre, %.2f across the band" % (seat_c, a.seated if a else -1))
    check("...which is exactly what the spec's closed form now charges for",
          r_in_needed(steep50) > ring_r_in(), "needs %.2f of %.1f" % (r_in_needed(steep50), ring_r_in()))
    # ------------------------------- THE END PLATE IS A DISC, NOT A BALL
    # A capsule swept about the plate's 3 mm thickness with the plate's
    # radius is a ball 120 mm across.  On a 92 mm section the first joint
    # stands clear of it and the error was invisible; on a 115 mm section
    # it forbids both end joints and the truss cannot be planned at all.
    deep = Truss(length=1000.0, side=115.0, alpha=40.0, d_chord=3.0, d_diag=1.5, name="deep")
    g = TrussGeometry(deep)
    fx = fixture.Fixture(g)
    f0, f1, fr = fx.obstacles(0.0)
    x0 = -(fx.end_free() + Cage.END_PLATE_T)
    near = [(a, b, r) for a, b, r in zip(f0, f1, fr)
            if abs(a[0] - (x0 + Cage.END_PLATE_T / 2.0)) < 1e-6]
    check("the end plate is modelled by samples no thicker than the plate",
          bool(near) and max(r for _a, _b, r in near) <= Cage.END_PLATE_T / 2.0 + 1e-9,
          "%d spokes, thickest %.2f mm against a %.1f mm plate"
          % (len(near), max((r for _a, _b, r in near), default=-1), Cage.END_PLATE_T))
    check("...and it reaches the radius the plate reaches",
          bool(near) and abs(max(np.linalg.norm(b[1:]) for _a, b, _r in near)
                             - fx.plate_r()) < 1e-6,
          "%.1f mm against %.1f" % (max((np.linalg.norm(b[1:]) for _a, b, _r in near),
                                        default=-1), fx.plate_r()))
    st = approach.plan_truss(g, fx, span=0.0)
    check("a deep section's END joints get stations: the plate does not swallow them",
          all(v[1] is not None for v in st.values()),
          "chords refused: %s" % [k for k, v in st.items() if v[1] is None])

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
