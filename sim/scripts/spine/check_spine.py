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
    # Measured against the OPEN end, which is the comparison that means
    # something: the plate buys 194 Hz over an open triangle and a 6x1 mm
    # strap buys 187 of them.  (Against the plate alone it is a 2% gap,
    # which is a tolerance argument and moves with the head mass.)
    f_strap = strap.modes(1)[0][0]
    check("...and the bracket that closes it need not be stiff, only present: a strap "
          "recovers almost all of what a plate does",
          (f_strap - f_open) / (f_closed - f_open) > 0.9,
          "a 6x1 mm strap recovers %.0f%% of the plate's gain (%.0f Hz open, "
          "%.0f strap, %.0f plate)"
          % (100.0 * (f_strap - f_open) / (f_closed - f_open), f_open, f_strap, f_closed))
    # THE LIGHT HEAD CHANGED THIS ANSWER.  With the brief's 50 g heads the
    # payload was two thirds of the moving mass and the X web's extra shear
    # stiffness bought nothing measurable.  With the real 5 g module the
    # truss carries mostly ITSELF -- 54 g of structure against 10 g of
    # cameras -- so the X's 20 g of extra carbon is 20 g of extra manoeuvre
    # load, and it comes back with a HIGHER first mode and a WORSE angular
    # error.  Stiffening a self-loaded structure by adding material to it
    # is the trap this check exists to name.
    xw = truss(web="x")
    r_w = metrics.evaluate(closed, D)
    r_x = metrics.evaluate(xw, D)
    check("once the ends are closed the single-diagonal web is lighter than an X AND "
          "truer: with a 5 g head the truss is its own load, so the X's extra carbon "
          "buys frequency and costs accuracy",
          closed.structural_mass < xw.structural_mass
          and r_w["budget_used"] < r_x["budget_used"]
          and r_x["f1_hz"] > r_w["f1_hz"],
          "warren %.0f Hz %.1f g %.2f of budget; X %.0f Hz %.1f g %.2f"
          % (r_w["f1_hz"], closed.structural_mass * 1e3, r_w["budget_used"],
             r_x["f1_hz"], xw.structural_mass * 1e3, r_x["budget_used"]))
    check("...and that is arithmetic, not luck: the structure outweighs the two heads "
          "it carries by four to one",
          closed.structural_mass * 1e3 > 4.0 * 2.0 * D.tip_mass_g,
          "%.1f g of structure against 2 x %.1f g of camera"
          % (closed.structural_mass * 1e3, D.tip_mass_g))

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

    # ------------------------------------------------ THE CAMERA MOUNT
    # The mount is part of the design, not an accessory: modelled as the
    # massless rigid `bracket` it was placeheld as, it charges no mass and no
    # compliance AND puts the camera's mass on the chord ends 66 mm off the
    # axis.  These are the three claims the nose rests on.
    nose = build.Nose.around()
    def yaw_acc(n, tip=D.tip_mass_g, side=115, alpha=40, dc=3.0, dd=1.5):
        m = build.warren_truss(1000, side, alpha, dc, dd, tip_mass=tip, nose=n)
        f = m.accel_load([0, 1, 0], D.manoeuvre_g * D.g)
        return np.degrees(metrics.relative_pose(m, m.solve(f)).yaw) * 1e6
    def with_nose(n, tip=D.tip_mass_g):
        return metrics.evaluate(build.warren_truss(1000, 115, 40, 3.0, 1.5,
                                                   tip_mass=tip, nose=n), D)
    ref = with_nose(None)
    got = with_nose(nose)
    # THE PLACEHOLDER IS NOT A BOUND, IN EITHER DIRECTION, and that is the
    # argument for modelling the mount rather than a tolerance on it.  The
    # massless rigid bracket makes two opposite errors at once: it hangs
    # the payload on the chord ends, 66 mm off the axis (pessimistic, and
    # it dominates at a heavy head), and it charges nothing at all for the
    # mount's own mass and compliance (optimistic, and it dominates at a
    # light one).  Between the brief's 50 g and the module's 5 g the net
    # error changes SIGN -- 28% pessimistic to 9% optimistic -- so no
    # margin on the placeholder could have covered both.
    ref50, got50 = with_nose(None, 50.0), with_nose(nose, 50.0)
    check("the massless rigid bracket is not a bound on the mount in EITHER direction: "
          "its error changes sign between a 50 g head and the real 5 g one, so the "
          "mount has to be in the model",
          (ref["budget_used"] - got["budget_used"])
          * (ref50["budget_used"] - got50["budget_used"]) < 0.0,
          "at 5 g: %.3f placeholder against %.3f modelled; at 50 g: %.3f against %.3f"
          % (ref["budget_used"], got["budget_used"],
             ref50["budget_used"], got50["budget_used"]))
    off = build.Nose.around()
    off = build.Nose(standoff=off.standoff, r_platform=off.r_platform,
                     payload_x=off.payload_box[0] / 2.0, payload_box=off.payload_box,
                     d_strut=off.d_strut, d_platform=off.d_platform,
                     d_batten=off.d_batten)
    bad, bad50 = with_nose(off), with_nose(off, 50.0)
    check("...and bonding the camera END-ON instead, so its mass sits half a housing "
          "outboard of the platform, is worse at any head mass",
          bad["budget_used"] > got["budget_used"]
          and bad50["budget_used"] > got50["budget_used"],
          "%.2f of the budget against %.2f at 5 g, %.2f against %.2f at 50 g"
          % (bad["budget_used"], got["budget_used"],
             bad50["budget_used"], got50["budget_used"]))
    check("...and the penalty is mass times lever arm, so it grows with the payload: "
          "21% at 5 g, 103% at 50 g",
          (bad50["budget_used"] / got50["budget_used"] - 1.0)
          > 2.0 * (bad["budget_used"] / got["budget_used"] - 1.0),
          "+%.0f%% at 5 g, +%.0f%% at 50 g"
          % (100.0 * (bad["budget_used"] / got["budget_used"] - 1.0),
             100.0 * (bad50["budget_used"] / got50["budget_used"] - 1.0)))
    # THE NULL.  A sweep of strut diameter shows a dip in manoeuvre yaw.  It
    # is not an optimum: the yaw changes SIGN there, because two terms of
    # opposite sense cancel -- the payload's inertia pulling the platform
    # one way and the struts' own inertia pulling it the other.  A design
    # sitting on it is a design sitting on a cancellation.
    thin = yaw_acc(build.Nose.around(d_strut=0.6))
    lo = yaw_acc(build.Nose.around(d_strut=0.8))
    check("the strut diameter has a NULL, not an optimum: the manoeuvre yaw changes "
          "SIGN across it, so a sweep's dip is a cancellation and must not be "
          "designed to",
          thin * lo < 0.0, "%+.0f udeg at 0.6 mm, %+.0f at 0.8" % (thin, lo))
    check("...and above the null there is no interior optimum at all -- yaw rises "
          "with every millimetre of strut, so thinner is better right down to it",
          all(yaw_acc(build.Nose.around(d_strut=d))
              < yaw_acc(build.Nose.around(d_strut=d + 0.5))
              for d in (1.0, 1.5, 2.0)),
          "%+.0f at 1.0 mm to %+.0f at 2.5"
          % (yaw_acc(build.Nose.around(d_strut=1.0)),
             yaw_acc(build.Nose.around(d_strut=2.5))))
    # AND IT MOVES WITH THE PAYLOAD.  The same geometry at a stocked 1.0 mm
    # strut sits on either side of the null depending only on what the head
    # weighs: the null was at 1.0 mm when the head was assumed to be 50 g,
    # and the real 5 g module puts it below 0.8.  Nothing about the mount
    # changed; a number nobody had measured did.
    heavy = yaw_acc(build.Nose.around(d_strut=1.0), tip=50.0)
    light = yaw_acc(build.Nose.around(d_strut=1.0), tip=D.tip_mass_g)
    check("...and the null MOVES with the head mass: at a stocked 1.0 mm strut the "
          "same geometry sits on opposite sides of it at 50 g and at 5 g",
          heavy * light < 0.0,
          "%+.0f udeg at 50 g, %+.0f at %.0f g" % (heavy, light, D.tip_mass_g))

    # --------------------------------- THE DUTY'S HEAD IS THE PART'S HEAD
    # The two packages keep their own numbers on purpose -- spine solves
    # frames for any product, truss knows this one -- but a duty whose head
    # mass has drifted away from the part being flown is the failure this
    # whole re-run was: 50 g stood in duty.py against a 4 g module for as
    # long as nobody compared them.  So they are compared.
    from truss.spec import Payload as _P
    check("the duty's camera head is the module the truss package measured, plus its "
          "stated termination allowance -- not a number of its own",
          abs(D.tip_mass_g - (_P.MASS + _P.HEAD_EXTRA)) < 1e-9,
          "%.1f g duty against %.1f + %.1f g part"
          % (D.tip_mass_g, _P.MASS, _P.HEAD_EXTRA))

    # ------------------------------------- THE ANGLE IN THE NAME IS A REQUEST
    # _stations rounds the bay pitch to a whole number of bays, so the web
    # angle a candidate is named for is not the one it is built at.  That
    # would be a labelling nuisance except that the winding head's bore
    # rule scales as tan(alpha): asked about the requested angle it passed
    # designs whose real joints the head cannot enter.
    check("the web angle a lattice is BUILT at is not the one it was asked for: "
          "at a 100 mm section 35 and 40 degrees are the same truss",
          abs(build.effective_alpha(1000, 100, 35)
              - build.effective_alpha(1000, 100, 40)) < 1e-9
          and abs(build.effective_alpha(1000, 100, 35) - 35.0) > 3.0,
          "35 and 40 both build at %.2f" % build.effective_alpha(1000, 100, 35))
    check("...and it is the built angle the model has, not the label",
          abs(np.degrees(np.arctan(
              0.100 / (build.warren_truss(1000, 100, 35, 3, 1)
                       .nodes[1][0] - build.warren_truss(1000, 100, 35, 3, 1)
                       .nodes[0][0])))
              - build.effective_alpha(1000, 100, 35)) < 1e-6,
          "%.4f deg" % build.effective_alpha(1000, 100, 35))
    from truss.spec import Truss as _T, ring_r_in as _rin, r_in_needed as _need, Ring as _R

    def _bore(side, alpha, dc, dd):
        return _rin() - _R.RUN_OUT - _need(_T(length=1000.0, side=side, alpha=alpha,
                                              d_chord=dc, d_diag=dd))
    fooled = [(s_, a_, dc, dd) for s_ in range(60, 165, 5)
              for a_ in (30, 35, 40, 45, 50, 55, 60)
              for dc in (1.0, 1.5, 2.0, 3.0) for dd in (1.0, 1.5, 2.0, 3.0)
              if dd <= dc
              and _bore(s_, a_, dc, dd) >= 0.0
              > _bore(s_, build.effective_alpha(1000.0, s_, a_), dc, dd)]
    check("...and asking the bore rule about the REQUESTED angle passes designs the "
          "head cannot wind -- which is why the sweep asks it about the built one",
          len(fooled) > 0, "%d of the wide grid, e.g. %s" % (len(fooled), fooled[0]))

    # ------------------------------------------ WHAT THE CELL CAN BUILD
    # THE SWEEP MUST KNOW, and the check is here because it cost a wrong
    # answer twice.  Without the bore, sweep_spine named a 45-degree web
    # the head's 20 mm bore cannot enter.  With the bore but without the
    # members' own modes and the loader's reach, it named a 1.0 mm web that
    # rings inside the propeller band on a truss the gantry cannot lay the
    # racks for.  1062 of 1470 candidates hold the accuracy budget; SEVEN
    # break no rule.
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
    # ...AND THE RECORD IS HELD TO ALL OF IT.  Read from chosen.py rather
    # than typed here, so a design that stops meeting a rule cannot go on
    # being the recorded answer -- which is how a 115 mm section stayed on
    # record 0.16 mm short of the bore it needed.
    from truss.spec import Ring
    from spine import chosen as chosen_mod
    ch = chosen_mod.OPTIMAL_1M
    cc = sweep.Candidate("truss", side=ch.side, alpha=ch.alpha,
                         d_chord=ch.d_chord, d_diag=ch.d_diag, web=ch.web)
    check("the design chosen.py records fits the winding head as BUILT, run-out charged",
          _ss.bore_margin(ch.length)(cc) > 0.0,
          "%+.3f mm of a %.1f mm bore, with %.2f of run-out charged"
          % (_ss.bore_margin(ch.length)(cc), Ring.ID, Ring.RUN_OUT))
    check("...and breaks no other rule the cell or the members impose either",
          not _ss.cell_rules(ch.length)(cc), str(_ss.cell_rules(ch.length)(cc)))
    ch_row = metrics.evaluate(
        build.warren_truss(ch.length, ch.side, ch.alpha, ch.d_chord, ch.d_diag,
                           tip_mass=D.tip_mass_g, web=ch.web,
                           nose=_ss.nose_for(ch.length)(cc)), D)
    check("...and holds the accuracy budget, the mass ceiling and the first mode, "
          "with the solved mount on it",
          ch_row["budget_used"] <= 1.0 and ch_row["mass_g"] <= 70.0
          and ch_row["f1_hz"] >= 200.0,
          "%.2f of budget, %.1f g, %.0f Hz"
          % (ch_row["budget_used"], ch_row["mass_g"], ch_row["f1_hz"]))

    bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det and (VERBOSE or not ok) else ""))
    print("check_spine: %d checks, %d failed" % (len(RESULTS), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
