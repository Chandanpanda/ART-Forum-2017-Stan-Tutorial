"""Tier 0, no physics: the spec's own assertions, and the truss geometry
re-derived against the rules that produced it.  Run this on every change.

    python3 sim/scripts/truss/check_geometry.py [-v]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np

from truss import spec, structure, geometry, fixture
from truss.spec import Truss, Gantry, Cage, Ring, Process, ring_r_in
from truss.geometry import TrussGeometry, theta_chord_up, theta_face_up, rot_x

VERBOSE = "-v" in sys.argv
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def truss_checks(t, tag):
    g = TrussGeometry(t)
    fx = fixture.Fixture(g)
    js0 = g.joints_on(0)
    check("%s: joints are the chord's node grid, one every run" % tag,
          len(js0) == t.n_nodes and
          all(abs((js0[i + 1].x - js0[i].x) - t.run) < 1e-6 for i in range(len(js0) - 1)),
          "%d joints, run %.1f" % (len(js0), t.run))
    check("%s: successive joints on a chord alternate faces" % tag,
          all(js0[i].face != js0[i + 1].face for i in range(len(js0) - 1)))
    check("%s: every interior joint takes exactly two mitred ends, of its own face" % tag,
          all(len(j.diags) == 2 and all(g.rods[d].face == j.face for d in j.diags)
              for j in g.joints if 0 < j.node < t.n_nodes - 1)
          and all(len(j.diags) == 1 for j in g.joints if j.node in (0, t.n_nodes - 1)))
    check("%s: the grid is centred on the chord" % tag,
          abs(js0[0].x - (t.length - js0[-1].x)) < 1e-6)
    # a diagonal's centreline crosses each chord's tangent plane at the joint
    ok, worst = True, 0.0
    for r in g.diags:
        for end, jix in zip((r.p0, r.p1), r.ends):
            j = g.joints[jix]
            axis = g.chord_point(j.chord, j.x)
            dist = float(np.linalg.norm(end - axis))
            worst = max(worst, abs(dist - t.d_chord / 2.0))
            ok &= abs(dist - t.d_chord / 2.0) < 1e-6 and abs(end[0] - j.x) < 1e-6
    check("%s: diagonal centrelines meet the chords on their tangent planes" % tag,
          ok, "worst %.3g mm" % worst)
    a = np.degrees(np.arctan2(t.s_eff, t.run))
    check("%s: the diagonal centreline is at alpha to the chord" % tag,
          abs(a - t.alpha) < 1e-6, "%.2f vs %.2f" % (a, t.alpha))
    check("%s: the band covers the mitre face with margin" % tag,
          t.band >= t.mitre_face + 2.0,
          "band %.1f, face %.1f" % (t.band, t.mitre_face))
    # THE TOPOLOGY CLAIM (brief 4.2), measured on the cluster itself
    j = js0[1]
    sweep = Ring.W / 2.0 + t.band / 2.0
    reach = max(g.cluster_reach(j, dx) for dx in np.linspace(-sweep, sweep, 19))
    ca = np.cos(np.radians(t.alpha))
    check("%s: within the ring's band the cluster stays inside the rim, with the "
          "clearance taken at the rim's corner" % tag,
          (ring_r_in() - reach) * ca >= Process.SEAT_CLEAR - 1e-6,
          "reach %.2f, slack %.2f x cos = %.2f vs %.1f"
          % (reach, ring_r_in() - reach, (ring_r_in() - reach) * ca, Process.SEAT_CLEAR))
    check("%s: ...and the closed form agrees with the sampled cluster to 0.5 mm" % tag,
          abs((spec.r_in_needed(t) - Process.SEAT_CLEAR / ca) - reach) < 0.5,
          "formula %.2f, sampled %.2f" % (spec.r_in_needed(t) - Process.SEAT_CLEAR / ca, reach))
    tm = g.thread_mass()
    check("%s: thread on the truss is a fraction of the joint budget" % tag,
          0.0 < tm < t.mass_joints * 0.5,
          "%.3f g thread, %.2f g joint budget" % (tm, t.mass_joints))
    # WHAT A HOOP GOES ROUND.  A diagonal touches the chord at the joint
    # and nowhere else, so a hoop laid away from the joint finds a gap and
    # beds on the chord under it -- measured wound, in check_ring.  The
    # closed form for where that starts is Truss.bind_half, and the hoop
    # perimeter has to change there and only there.
    bh = t.bind_half()
    gap = lambda dx: abs(dx) * np.tan(np.radians(t.alpha)) - t.d_diag / (2.0 * np.cos(np.radians(t.alpha)))
    check("%s: at the binding half-width the members are exactly a thread apart" % tag,
          abs(gap(bh) - t.thread_d) < 1e-9,
          "%.3f mm at dx %.3f, thread %.2f" % (gap(bh), bh, t.thread_d))
    check("%s: ...and they touch each other only within d_diag / 2 sin alpha of the joint" % tag,
          abs(gap(t.d_diag / (2.0 * np.sin(np.radians(t.alpha))))) < 1e-9)
    chord_hoop = np.pi * (t.d_chord + t.thread_d)
    check("%s: a hoop inside the binding width goes round both members, one outside "
          "goes round the chord alone" % tag,
          g.hoop_perimeter(j, bh * 0.5) > chord_hoop + 0.3
          and abs(g.hoop_perimeter(j, bh + 0.5) - chord_hoop) < 1e-9,
          "inside %.2f, outside %.2f, chord alone %.2f"
          % (g.hoop_perimeter(j, bh * 0.5), g.hoop_perimeter(j, bh + 0.5), chord_hoop))
    check("%s: the band binds over a fraction of its width, so the hull everywhere "
          "overstates the thread" % tag,
          bh < t.band / 2.0 and g.thread_on_joint(j) < sum(
              g.cluster_perimeter(j, float(dx)) for dx in
              (np.arange(t.turns) * t.band / t.turns - t.band / 2.0 + t.band / t.turns / 2.0)),
          "binds over %.1f of %.1f mm" % (2 * bh, t.band))
    # pins
    for k in range(t.n_chords):
        xs = fx.pin_xs(k)
        gaps = np.diff(xs)
        near = min(abs(x - j.x) for x in xs for j in g.joints_on(k))
        check("%s chord %d: pins never further apart than %.0f mm" % (tag, k, Cage.PIN_PITCH),
              len(xs) > 1 and gaps.max() <= Cage.PIN_PITCH + 1e-6,
              "max %.1f" % gaps.max())
        check("%s chord %d: ...and never inside a joint's exclusion" % (tag, k),
              near >= fx.joint_exclusion() - 1e-6,
              "nearest %.1f vs %.1f" % (near, fx.joint_exclusion()))
    # cradles clear of the ring's envelope, axially
    worst = 1e9
    ca = np.cos(np.radians(t.alpha))
    for c in fx.cradles:
        j = g.joints[c.joint]
        # the block lies along the diagonal, so its half-length projects
        # onto the chord axis by cos(alpha); its round end caps do not
        near = abs(float(c.centre[0]) - j.x) - (Cage.CRADLE_L / 2.0 * ca + fx.cradle_r())
        worst = min(worst, near)
    check("%s: cradles sit outside the ring's swept band, with clearance" % tag,
          worst >= spec.Head.ring_axial_half() + Process.SEAT_CLEAR - 1e-6,
          "nearest %.1f vs %.1f" % (worst, spec.Head.ring_axial_half() + Process.SEAT_CLEAR))
    check("%s: thread posts stand beyond both chord ends" % tag,
          all((p.p0[0] < 0.0) if p.end == 0 else (p.p0[0] > t.length) for p in fx.posts))
    lo, hi = fx.x_range()
    check("%s: the cell's X travel covers posts and racks" % tag,
          hi - lo <= Gantry.X_TRAVEL, "%.0f..%.0f needs %.0f of %.0f"
          % (lo, hi, hi - lo, Gantry.X_TRAVEL))
    check("%s: ...and its Y travel reaches the chord rack" % tag,
          fx.y_reach() <= Gantry.Y_TRAVEL / 2.0,
          "%.0f of +-%.0f" % (fx.y_reach(), Gantry.Y_TRAVEL / 2.0))
    # orientation for loading
    for k in range(t.n_chords):
        th = theta_chord_up(k)
        zs = [float((rot_x(th) @ g.chord_point(m, 0.0))[2]) for m in range(t.n_chords)]
        check("%s: theta_chord_up(%d) puts chord %d highest" % (tag, k, k),
              zs[k] == max(zs) and zs[k] > max(z for m, z in enumerate(zs) if m != k) + 1.0)
    for f, (a_, b_) in enumerate(geometry.FACES):
        th = theta_face_up(f)
        pa = rot_x(th) @ g.chord_point(a_, 0.0)
        pb = rot_x(th) @ g.chord_point(b_, 0.0)
        pc = rot_x(th) @ g.chord_point(3 - a_ - b_, 0.0)
        check("%s: theta_face_up(%d) levels its chords above the third" % (tag, f),
              abs(pa[2] - pb[2]) < 1e-6 and pa[2] > pc[2] + 1.0)
    r = g.diags[0]
    mid, yaw, th = fx.place_pose(r)
    check("%s: a diagonal is placed level, at +-alpha in the horizontal plane" % tag,
          abs(mid[2] - t.R / 2.0) < 1e-6 and abs(abs(yaw) - t.alpha) < 3.0,
          "z %.1f (R/2 %.1f), yaw %.1f" % (mid[2], t.R / 2.0, yaw))
    r = g.chords[1]
    mid, yaw, th = fx.place_pose(r)
    check("%s: a chord is placed uppermost, along x" % tag,
          abs(mid[2] - t.R) < 1e-6 and abs(yaw) < 1e-6)
    # the loader's capture range beats the gantry's error
    check("%s: a V-notch captures more than the gantry and screw growth miss by" % tag,
          fx.capture_range(g.chords[0]) > Gantry.REPEAT + spec.stepper_scale_sigma() * t.length,
          "%.2f mm capture vs %.2f mm error"
          % (fx.capture_range(g.chords[0]), Gantry.REPEAT + spec.stepper_scale_sigma() * t.length))


def main():
    for name, ok in spec.CHECKS:
        check("spec: " + name, ok)
    truss_checks(structure.TRUSS_1M, "1m")
    truss_checks(structure.TRUSS_300, "300")
    # A TRUSS IT HAS NEVER SEEN: not a default, not the brief's
    other = Truss(length=600.0, side=70.0, alpha=40.0, d_chord=2.0, d_diag=1.5,
                  name="other")
    truss_checks(other, "other")
    # ---------------------------------------------- THE RING'S SOLVERS
    # These are cached because approach.head_distance calls them in its
    # innermost loop (uncached, check_approach went from 4 minutes to over
    # 50).  A cache that ignores its inputs is worse than a slow solver, so
    # this is the check that it does not.
    import time as _t
    t0 = _t.time()
    for _ in range(50000):
        Ring.mesh()
        Ring.pinion_az()
    check("the ring's mesh and pinion solvers are cheap enough to call from an "
          "inner loop", _t.time() - t0 < 1.0, "50k of each in %.3f s" % (_t.time() - t0))
    m0, az0, mouth0 = Ring.mesh(), Ring.pinion_az(), Ring.race_mouth()
    keep = (Ring.RACE_CLEAR, Ring.OD, Ring.GAP)
    try:
        Ring.RACE_CLEAR = keep[0] + 0.05
        moved_mouth = abs(Ring.race_mouth() - mouth0) > 1e-6
        Ring.RACE_CLEAR = keep[0]
        Ring.OD = keep[1] + 8.0
        m1 = Ring.mesh()
        moved_mesh = (m1.n_ring, m1.m) != (m0.n_ring, m0.m)
        Ring.OD = keep[1]
        # 45 rather than any old number: the gap must still be a whole
        # number of teeth or mesh() rightly refuses (90 degrees has no
        # admissible module at all, which is itself worth knowing)
        Ring.GAP = 45.0
        moved_az = abs(Ring.pinion_az()[0] - az0[0]) > 1e-6
    finally:
        Ring.RACE_CLEAR, Ring.OD, Ring.GAP = keep
    check("...and the cache is keyed on what they read: a different clearance, rim "
          "or gap re-solves", moved_mouth and moved_mesh and moved_az,
          "mouth %s, mesh %s, pinions %s" % (moved_mouth, moved_mesh, moved_az))
    check("...and restoring the spec restores the answer",
          Ring.mesh() == m0 and Ring.pinion_az() == az0
          and abs(Ring.race_mouth() - mouth0) < 1e-12)

    # motion profiles
    from truss import motion
    for d, v, a in ((10.0, 40.0, 300.0), (500.0, 60.0, 400.0), (0.5, 40.0, 300.0)):
        T = motion.trap_time(d, v, a)
        ts = np.linspace(0.0, T, 400)
        xs = np.array([motion.trap_position(tt, d, v, a) for tt in ts])
        vmax = np.max(np.abs(np.diff(xs) / np.diff(ts))) if T > 0 else 0.0
        check("motion: a %.1f mm move ends where it should and never exceeds vmax" % d,
              abs(xs[-1] - d) < 1e-6 and vmax <= v * 1.02 and T > 0,
              "T %.3f s, peak %.1f mm/s" % (T, vmax))
    mv = motion.Move({"x": 0.0, "z": 10.0}, {"x": 100.0, "z": 30.0},
                     Gantry.V_MAX, Gantry.A_MAX)
    p = mv.at(mv.T)
    check("motion: a coordinated move arrives on every axis together",
          abs(p["x"] - 100.0) < 1e-6 and abs(p["z"] - 30.0) < 1e-6)
    p = mv.at(mv.T / 2.0)
    check("motion: ...and runs a straight line between them",
          abs((p["x"] - 0.0) / 100.0 - (p["z"] - 10.0) / 20.0) < 0.02)

    bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det and (VERBOSE or not ok) else ""))
    print("check_geometry: %d checks, %d failed" % (len(RESULTS), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
