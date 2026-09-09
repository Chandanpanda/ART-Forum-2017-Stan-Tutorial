"""The spines and the metrics: what the model says, and why to believe it.

check_frame validates the element against closed forms.  This validates
what is built out of it -- that a lattice behaves like the beam it is
idealised as when it is braced, that it does not when it is not, and that
the metrics measure what they claim.  Several of these exist because the
first version of this package got them wrong.

    python3 sim/scripts/spine/check_spine.py [-v]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np

from spine import build, duty, metrics, material as mat, sweep
from spine.model import Model, Member, Section, PointMass

VERBOSE = "-v" in sys.argv
RESULTS = []
D = duty.QUADROTOR


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def truss(**kw):
    kw.setdefault("tip_mass", D.tip_mass_g)
    return build.warren_truss(1000, 92, 40, 3, 2, **kw)


def yaw_of(m, direction=(0, 1, 0)):
    u = m.solve(m.accel_load(direction, D.manoeuvre_g * D.g))
    return abs(metrics.relative_pose(m, u).yaw)


def main():
    # ------------------------------------------- THE COUNTING ARGUMENT
    for web, short in (("warren", True), ("x", False), ("batten", False)):
        m = truss(web=web)
        have, need = build.maxwell_count(m)
        check("the %s web is %s a rigid pin-jointed frame" % (web, "NOT" if short else ""),
              (have < need) == short, "%d members and supports against %d freedoms" % (have, need))

    # --------------------------------- THE FINDING: CLOSE THE END TRIANGLE
    # Without a bracket across the three chord ends the tip mass rides a
    # section-distortion mechanism, and the spine is six times softer than
    # its own beam idealisation.  With one -- any one -- it is not.
    open_ends, closed = truss(brackets=False), truss(brackets=True)
    f_open = open_ends.modes(1)[0][0]
    f_closed = closed.modes(1)[0][0]
    check("a lattice with open end triangles is far softer than the beam it is idealised as",
          f_closed / f_open > 3.0, "%.0f Hz open, %.0f Hz closed" % (f_open, f_closed))
    strap = truss(brackets=False)
    sec = Section.rect(0.006, 0.001)
    for ring in (strap.faces["left"], strap.faces["right"]):
        for k in range(3):
            strap.members.append(Member(ring[k], ring[(k + 1) % 3], sec, mat.BRACKET))
    strap._K = strap._M = None
    check("...and the bracket that closes it need not be stiff, only present",
          abs(strap.modes(1)[0][0] / f_closed - 1.0) < 0.02,
          "a 6x1 mm strap gives %.0f Hz against %.0f for a 20x3 plate"
          % (strap.modes(1)[0][0], f_closed))
    check("once the ends are closed the single-diagonal web is as stiff as an X, and lighter",
          truss(web="x").modes(1)[0][0] / f_closed < 1.1
          and closed.structural_mass < truss(web="x").structural_mass,
          "%.0f Hz at %.1f g against %.0f Hz at %.1f g"
          % (f_closed, closed.structural_mass * 1e3, truss(web="x").modes(1)[0][0],
             truss(web="x").structural_mass * 1e3))

    # ------------------------------ THE LATTICE AGAINST ITS IDEALISATION
    # A braced lattice must reproduce the beam the brief reasons with:
    # I = (n/2) A R^2 for three chords on a circle.
    m = truss(web="batten")
    R = 0.092 / np.sqrt(3.0)
    A = np.pi * 0.003 ** 2 / 4.0
    EI = mat.CARBON_UD.E * 1.5 * A * R ** 2
    P, l = 1.0, 0.5
    m2 = truss(web="batten", tip_mass=0.0)
    f = np.zeros(m2.n * 6)
    for nd in m2.faces["right"]:
        f[m2.dofs(nd)[1]] = P / 3.0
    u = m2.solve(f)
    _t, r = m2.face_pose(u, "right")
    check("a braced lattice's end rotates P l^2 / 2EI, the idealisation the brief reasons with",
          abs(abs(r[2]) / (P * l ** 2 / (2 * EI)) - 1.0) < 0.15,
          "%.4e rad against %.4e" % (abs(r[2]), P * l ** 2 / (2 * EI)))

    # --------------------------------------- THE FACTOR OF TWO
    # The budget is between the faces.  A centre-mounted spine's two
    # cantilevers point opposite ways, so their faces rotate in opposite
    # senses and the errors ADD: the tip slope of one half is half the
    # answer, and the brief compares that half against the whole budget.
    u = closed.solve(closed.accel_load([0, 1, 0], D.manoeuvre_g * D.g))
    _tl, rl = closed.face_pose(u, "left")
    _tr, rr = closed.face_pose(u, "right")
    check("the two camera faces yaw in opposite senses, so the relative error is twice one tip",
          rl[2] * rr[2] < 0 and abs(abs(rr[2] - rl[2]) / (2 * abs(rr[2])) - 1.0) < 0.05,
          "left %+.3e, right %+.3e rad" % (rl[2], rr[2]))

    # ------------------------------------------- WHICH LOAD DRIVES YAW
    yaws = {ax: yaw_of(closed, d) for ax, d in
            (("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1)))}
    check("fore-aft acceleration is what yaws the pair; along the baseline does almost nothing",
          yaws["y"] > 10.0 * yaws["x"],
          {k: "%.2e" % v for k, v in yaws.items()})

    # --------------------------- RIGID-BODY ROTATION COSTS NOTHING
    # The case where an open lattice should lose to a closed tube is
    # torsion.  It never arises: an angular acceleration of the aircraft
    # turns both faces together, and calibration only sees the difference.
    worst = 0.0
    for v in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        pr = metrics.relative_pose(closed, closed.solve(
            closed.angular_accel_load(v, D.ang_accel)))
        worst = max(worst, max(abs(x) for x in (pr.yaw, pr.pitch, pr.roll)))
    lin = metrics.relative_pose(closed, closed.solve(
        closed.accel_load([0, 1, 0], D.manoeuvre_g * D.g)))
    check("an angular acceleration of the whole aircraft disturbs the PAIR not at all",
          worst < 1e-4 * abs(lin.yaw), "%.2e rad against %.2e for the linear case"
          % (worst, abs(lin.yaw)))
    check("...so the lattice's torsional softness, where a tube would win, never gets to matter",
          worst < 1e-9, "%.2e rad" % worst)
    pz = metrics.relative_pose(closed, closed.solve(
        closed.accel_load([0, 0, 1], D.manoeuvre_g * D.g)))
    check("a vertical manoeuvre costs relative ROLL, which breaks rectification rather "
          "than biasing depth", abs(pz.roll) > 10.0 * abs(pz.yaw),
          "roll %.2e against yaw %.2e rad, a factor of %.0f"
          % (abs(pz.roll), abs(pz.yaw), abs(pz.roll) / abs(pz.yaw)))

    # -------------------------------------------------- THE CONVERSION
    # dZ = Z^2 dtheta / B, and the focal length cancels.  Checked against
    # the disparity arithmetic it comes from, at two focal lengths.
    Z, B, dth = 100.0, 1.0, 1e-5
    for f_px in (800.0, 2400.0):
        d = f_px * B / Z
        direct = f_px * B / (d + f_px * dth) - Z
        check("range error at f = %.0f px is Z^2 dtheta / B, whatever the lens" % f_px,
              abs(abs(direct) / (Z * Z * dth / B) - 1.0) < 1e-3,
              "%.4f m direct, %.4f m by the formula" % (abs(direct), Z * Z * dth / B))
    p = metrics.Pose(yaw=dth, pitch=0.0, roll=0.0, d_baseline=0.0, baseline=B)
    check("...and that is what Pose.range_error reports",
          abs(p.range_error(Z) / (Z * Z * dth / B) - 1.0) < 1e-12)
    p2 = metrics.Pose(yaw=0.0, pitch=0.0, roll=0.0, d_baseline=1e-5, baseline=B)
    check("a stretched baseline is a pure scale error, Z dB / B",
          abs(p2.range_error(Z) / (Z * 1e-5 / B) - 1.0) < 1e-12)

    # ------------------------------------------------ THE TUBE, FAIRLY
    tube = build.tube(1000, 43, 1, tip_mass=D.tip_mass_g)
    check("the tube carries the section depth it actually has, not the spread of its nodes",
          abs(tube.depth - 0.043) < 1e-9, "%.4f m" % tube.depth)
    check("...so a gradient across it bows it, which is a monolithic tube's classic error",
          metrics.evaluate(tube, D)["yaw_grad_udeg"] > 1e3,
          "%.0f udeg" % metrics.evaluate(tube, D)["yaw_grad_udeg"])
    bare = build.tube(1000, 43, 1, tip_mass=0.0)
    f = np.zeros(bare.n * 6)
    f[bare.dofs(bare.faces["right"][0])[1]] = 1.0
    u = bare.solve(f)
    _t, r = bare.face_pose(u, "right")
    sec_t = Section.tube(0.043, 0.001)
    want = 1.0 * 0.5 ** 2 / (2.0 * mat.CARBON_WRAP.E * sec_t.Iz)
    check("a tube modelled here matches the beam formula it is quoted by",
          abs(abs(r[2]) / want - 1.0) < 1e-6, "%.4e vs %.4e" % (abs(r[2]), want))

    # ---------------------------------------------------- SANITY
    m = truss()
    u = m.solve(m.accel_load([0, 1, 0], 0.0))
    check("no load, no motion", np.allclose(u, 0.0))
    r = metrics.evaluate(closed, D)
    check("the evaluation reports every metric the sweep ranks on",
          all(k in r for k in ("mass_g", "f1_hz", "yaw_accel_udeg", "yaw_grad_udeg",
                               "yaw_static_udeg", "budget_used", "dz_static_m",
                               "yaw_vib_udeg", "modes_in_band")))
    check("a heavier duty costs more budget than a gentler one",
          metrics.evaluate(closed, duty.SURVEY_MAST)["yaw_accel_udeg"]
          < r["yaw_accel_udeg"],
          "mast %.0f, quadrotor %.0f udeg"
          % (metrics.evaluate(closed, duty.SURVEY_MAST)["yaw_accel_udeg"], r["yaw_accel_udeg"]))
    big = build.warren_truss(1000, 130, 40, 3, 2, tip_mass=D.tip_mass_g)
    check("a deeper section yaws less, as the second moment says it must",
          yaw_of(big) < yaw_of(closed) * 0.6,
          "%.3e at 130 mm against %.3e at 92" % (yaw_of(big), yaw_of(closed)))

    # ------------------------------------------ WHAT THE CELL CAN BUILD
    # THE SWEEP MUST KNOW, and the check is here because it cost a wrong
    # answer: without the bore, sweep_spine named 120/45/3.0/1.5 as the
    # lightest design holding the accuracy budget, and every design holding
    # that budget wants a 45-degree web or steeper, which the head's 20 mm
    # bore cannot enter.  28 designs met the budget and none could be made.
    import importlib.util
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_spine.py")
    _sp = importlib.util.spec_from_file_location("sweep_spine", _p)
    _ss = importlib.util.module_from_spec(_sp)
    _sp.loader.exec_module(_ss)
    bore = _ss.bore_margin(1000.0)
    cands = sweep.truss_grid([92, 115, 120], [35, 40, 45, 50])
    rows = sweep.run(cands, D, 1000.0, buildable=bore)
    check("the sweep carries a buildability margin for every candidate",
          all("build_mm" in r for r in rows) and any(r["build_mm"] < 0 for r in rows)
          and any(r["build_mm"] > 0 for r in rows),
          "%d of %d candidates the cell cannot build"
          % (sum(1 for r in rows if r["build_mm"] < 0), len(rows)))
    check("...and it refuses them: a design the head cannot enter is not feasible",
          all("buildable" in r["violates"] for r in rows if r["build_mm"] < 0))
    shallow = [r for r in rows if r["candidate"].alpha <= 40 and r["candidate"].d_diag <= 1.5]
    steep = [r for r in rows if r["candidate"].alpha >= 50]
    check("the bore limits the WEB ANGLE, not the section -- which is why it costs "
          "accuracy, since the web angle is what buys it",
          max(r["build_mm"] for r in shallow) > 0 > max(r["build_mm"] for r in steep),
          "shallowest web has %+.2f mm, steepest %+.2f"
          % (max(r["build_mm"] for r in shallow), max(r["build_mm"] for r in steep)))
    from truss.spec import Ring
    chosen_row = [r for r in rows if "115/40/3.0/1.5" in r["name"]]
    check("...and the design chosen.py records does fit the head as built",
          bool(chosen_row) and chosen_row[0]["build_mm"] > 0,
          "%+.3f mm of a %.1f mm bore, with %.2f of run-out charged"
          % (chosen_row[0]["build_mm"] if chosen_row else -9.9, Ring.ID, Ring.RUN_OUT))

    bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det and (VERBOSE or not ok) else ""))
    print("check_spine: %d checks, %d failed" % (len(RESULTS), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
