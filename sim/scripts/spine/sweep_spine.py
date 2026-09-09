"""The design sweep: which spine holds the camera's calibration, per gram.

    python3 sim/scripts/spine/sweep_spine.py            # the 1 m spine, quadrotor duty
    python3 sim/scripts/spine/sweep_spine.py --mast     # the same sweep, a gentler duty
    python3 sim/scripts/spine/sweep_spine.py --length 300
    python3 sim/scripts/spine/sweep_spine.py --wide     # a bigger grid, slower

Everything printed is computed from the frame model in sim/spine, whose
element is validated against closed forms by check_frame and whose
lattices are validated against the beam idealisation by check_spine.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np

from spine import build, duty as duty_mod, metrics, sweep, material as mat
from spine.material import Material


def named(length, duty, nose=None):
    """The designs already on the table, for the comparison to start from.

    They carry the same mount as the grid does, or the comparison is
    between a design with a camera on it and one without.
    """
    def n(side, alpha, dc, dd):
        return nose(sweep.Candidate("truss", side=side, alpha=alpha,
                                    d_chord=dc, d_diag=dd)) if nose else None
    out = []
    for web in ("warren", "x"):
        out.append(build.warren_truss(length, 92, 40, 3, 2, tip_mass=duty.tip_mass_g,
                                      web=web, nose=n(92, 40, 3, 2),
                                      name="derived truss, %s web" % web))
    out.append(build.warren_truss(length, 92, 40, 3, 2, tip_mass=duty.tip_mass_g,
                                  brackets=False, nose=n(92, 40, 3, 2),
                                  name="derived truss, ends OPEN"))
    out.append(build.warren_truss(length, 90, 45, 3, 2, tip_mass=duty.tip_mass_g,
                                  nose=n(90, 45, 3, 2), name="the brief's truss"))
    out.append(build.tube(length, 43, 1.0, tip_mass=duty.tip_mass_g,
                          name="carbon tube 43x1"))
    out.append(build.tube(length, 20, 1.0, tip_mass=duty.tip_mass_g,
                          name="carbon tube 20x1"))
    out.append(build.tube(length, 40, 2.0, material=mat.ALUMINIUM,
                          tip_mass=duty.tip_mass_g, name="aluminium tube 40x2"))
    return out


def sensitivity(length, duty, base, nose=None):
    """Which guessed material property the answer actually turns on.

    Every number in material.py is a typical value for the class rather
    than a measurement of the stock in the shop.  This says which one to
    put on a rig first: the one whose 20% moves the answer most.
    """
    rows = []
    for fld in ("E", "G", "alpha", "rho"):
        for k in (0.8, 1.2):
            kw = {"name": mat.CARBON_UD.name, "E": mat.CARBON_UD.E, "G": mat.CARBON_UD.G,
                  "rho": mat.CARBON_UD.rho, "alpha": mat.CARBON_UD.alpha}
            kw[fld] = kw[fld] * k
            # the SAME structure as `base`, mount included -- a ratio against a
            # model with a camera on it, taken from one without, is not a
            # sensitivity, it is the mount showing up in every row
            m = build.warren_truss(length, 92, 40, 3, 2, material=Material(**kw),
                                   tip_mass=duty.tip_mass_g, nose=nose)
            r = metrics.evaluate(m, duty)
            rows.append((fld, k, r["yaw_static_udeg"] / base["yaw_static_udeg"] - 1.0,
                         r["f1_hz"] / base["f1_hz"] - 1.0))
    return rows


def bore_margin(length):
    """mm of ring bore a candidate leaves over -- negative means the
    winding head cannot get round its joint.  Returns a predicate.

    It scales as tan(alpha), so it is the WEB ANGLE that the head limits,
    and the web angle is what buys accuracy: every design in the sweep that
    holds the quadrotor budget wants 45 degrees or steeper, and none of
    them fits a 20 mm bore.  This is the trade the product actually has.
    """
    from truss.spec import Truss, ring_r_in, r_in_needed, Ring

    def margin(c):
        if c.kind != "truss":
            return float("inf")
        # AT THE ANGLE THE LATTICE IS ACTUALLY BUILT AT, not the one asked
        # for.  The bay pitch rounds to a whole number of bays, so 100/35
        # and 100/40 are one truss built at 38.66 -- and r_in_needed scales
        # as tan(alpha), so asking this rule about the requested angle
        # passed 40 candidates in the wide grid whose real joints the head
        # cannot enter.
        t = Truss(length=length, side=c.side,
                  alpha=build.effective_alpha(length, c.side, c.alpha),
                  d_chord=c.d_chord, d_diag=c.d_diag)
        return ring_r_in() - Ring.RUN_OUT - r_in_needed(t)
    return margin


def nose_for(length):
    """A camera mount solved for the candidate's OWN section.

    The standoff is not a constant.  The lens has a 66 x 41 degree field
    and the truss it is bonded to is inside it, so how far the camera must
    stand off the chord ends falls out of the section: 23 mm at a 92 mm
    triangle, 36 mm at 140.  That matters here because standoff is a lever
    arm -- a deeper section buys stiffness and pays part of it back in
    mount offset -- and a sweep that shares one nose across the grid prices
    the deep sections as if they got theirs free.

    Supplied from the script, like the bore: sim/spine stays a frame
    solver, and everything about lenses and winding heads lives in
    sim/truss.
    """
    from truss import mount, geometry
    from truss.spec import Truss

    def make(c):
        if c.kind != "truss":
            return None
        t = Truss(length=length, side=c.side,
                  alpha=build.effective_alpha(length, c.side, c.alpha),
                  d_chord=c.d_chord, d_diag=c.d_diag)
        return build.Nose.around(**mount.nose_spec(geometry.TrussGeometry(t),
                                                   d_strut=c.d_diag))
    return make


# The global structural rules belong to THIS model -- it is the better one
# -- and everything else belongs to the package that owns the machine.
SPINE_OWNS = ("slope", "f1", "mass", "section envelope")


def cell_rules(length):
    """Every rule the members and the CELL impose, handed over whole from
    the package that owns them.

    The split is by which model is better at the question.  sim/spine's
    frame solver owns mass, the first mode and the angular budget.  It has
    no view at all of a diagonal's own first mode against the propeller
    band, of whether the winding ring fits round a joint, of the qualified
    joint angle, of chord buckling, of the build's cycle time, or of how
    far the gantry must reach to lay the racks -- and each of those has
    rejected a design this sweep ranked first.  Without them the sweep's
    answer for a metre truss was a 1.0 mm web whose diagonals ring inside
    the propeller band.
    """
    from truss import structure
    from truss.spec import Truss

    def rules(c):
        if c.kind != "truss":
            return []
        t = Truss(length=length, side=c.side,
                  alpha=build.effective_alpha(length, c.side, c.alpha),
                  d_chord=c.d_chord, d_diag=c.d_diag)
        return [r for r in structure.violations(t) if r not in SPINE_OWNS]
    return rules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=float, default=1000.0)
    ap.add_argument("--mast", action="store_true", help="the survey-mast duty")
    ap.add_argument("--wide", action="store_true", help="a bigger grid")
    ap.add_argument("--section-max", type=float, default=100.0,
                    help="mm the airframe can swallow (the brief's own limit)")
    ap.add_argument("--f1-min", type=float, default=None,
                    help="Hz, a floor on the first mode; the brief asks 200")
    ap.add_argument("--mass-max", type=float, default=70.0, help="g")
    ap.add_argument("--any-bore", action="store_true",
                    help="ignore whether the winding head can enter the joint")
    ap.add_argument("--no-mount", action="store_true",
                    help="the massless rigid placeholder instead of the solved mount "
                         "-- for showing what leaving it out costs, not for choosing")
    a = ap.parse_args()
    duty = duty_mod.SURVEY_MAST if a.mast else duty_mod.QUADROTOR
    print("spine sweep: %.0f mm baseline, %s duty (%.1f g manoeuvre, %.0f K gradient, "
          "%.3f deg budget, accuracy quoted at %.0f m)"
          % (a.length, duty.name, duty.manoeuvre_g, duty.gradient_K,
             duty.yaw_budget_deg, duty.range_m))

    print("\n--- the designs already on the table " + "-" * 42)
    print(metrics.HEADER)
    base = None
    the_nose = None if a.no_mount else nose_for(a.length)
    for m in named(a.length, duty, the_nose):
        r = metrics.evaluate(m, duty)
        if "derived truss, warren" in m.name:
            base = r
        print(metrics.row(r))

    sides = [70, 80, 90, 100, 110, 120, 130, 140]
    alphas = [35, 40, 45, 50]
    if a.wide:
        sides = list(range(60, 165, 5))
        alphas = [30, 35, 40, 45, 50, 55, 60]
    cands = sweep.truss_grid(sides, alphas, nose=the_nose) \
        + sweep.tube_grid([20, 30, 43, 60], [0.5, 1.0, 2.0])
    t0 = time.time()
    # WHAT THE CELL CAN BUILD is the fourth constraint, and it is supplied
    # from here rather than imported by the spine package -- the choosing
    # and the building stay separate, but the choosing has to know.  The
    # ring's bore, less what its centre wanders in the raceway, against the
    # radius the joint's diagonals need at the ring's own width.
    rows = sweep.run(cands, duty, length=a.length, section_max_mm=a.section_max,
                     mass_max_g=a.mass_max, f1_min_hz=a.f1_min,
                     buildable=None if a.any_bore else bore_margin(a.length),
                     rejects=None if a.any_bore else cell_rules(a.length))
    ok = sweep.feasible(rows)
    print("\n--- %d candidates in %.0f s, %d inside the constraints (section <= %.0f mm, "
          "mass <= %.0f g%s) %s"
          % (len(rows), time.time() - t0, len(ok), a.section_max, a.mass_max,
             ", f1 >= %.0f Hz" % a.f1_min if a.f1_min else "", "-" * 4))
    print(metrics.HEADER)
    for r in ok[:10]:
        print(metrics.row(r))

    print("\n--- the Pareto front: nothing feasible is both lighter and truer " + "-" * 15)
    print(metrics.HEADER)
    for r in sweep.pareto(ok):
        print(metrics.row(r))

    best = sweep.smallest_meeting(rows)
    print("\n--- the answer " + "-" * 64)
    if best is None:
        pool = ok or rows
        inside = min(pool, key=lambda r: r["budget_used"])
        print("  NO candidate in this grid holds the %.3f deg budget at %.1f g."
              % (duty.yaw_budget_deg, duty.manoeuvre_g))
        print("  The closest that IS feasible is %s at %.2f of the budget, %.1f g."
              % (inside["name"], inside["budget_used"], inside["mass_g"]))
        loose = min(rows, key=lambda r: r["budget_used"])
        if loose["violates"]:
            print("  Relax %s and %s holds it at %.2f, %.1f g."
                  % (" and ".join(loose["violates"]), loose["name"],
                     loose["budget_used"], loose["mass_g"]))
        # WHICH CONSTRAINT IS ACTUALLY BUYING THE ACCURACY
        held = [r for r in rows if r["budget_used"] <= 1.0]
        if held:
            no_bore = [r for r in held if "buildable" not in r["violates"]]
            worst = max(held, key=lambda r: -r["build_mm"])
            print("  %d designs hold the budget; %d of them the cell can build."
                  % (len(held), len(no_bore)))
            if not no_bore:
                print("  ALL of them want more ring bore than the head has -- the "
                      "shallowest is short by %.2f mm (%s, a %.0f degree web).  The "
                      "bore is the binding constraint on the product's accuracy."
                      % (-max(r["build_mm"] for r in held),
                         max(held, key=lambda r: r["build_mm"])["name"],
                         max(held, key=lambda r: r["build_mm"])["candidate"].alpha))
        print("  Of that, %.0f%% is the manoeuvre and %.0f%% the thermal gradient."
              % (100 * inside["yaw_accel_udeg"] / inside["yaw_static_udeg"],
                 100 * inside["yaw_grad_udeg"] / inside["yaw_static_udeg"]))
    else:
        print("  The lightest design inside the budget is %s at %.1f g, using %.2f of it."
              % (best["name"], best["mass_g"], best["budget_used"]))
    b = sweep.binding(rows)
    if b:
        print("  Of the forty truest designs overall, %s."
              % ", ".join("%d are refused by %s" % (v, k) for k, v in sorted(b.items())))
    trusses = [r for r in ok if r["candidate"].kind == "truss"]
    tubes = [r for r in ok if r["candidate"].kind == "tube"]
    if trusses and tubes:
        bt, bu = trusses[0], tubes[0]
        print("  Best truss %s: %.1f g, %.2f m of range error at %.0f m."
              % (bt["name"], bt["mass_g"], bt["dz_static_m"], duty.range_m))
        print("  Best tube  %s: %.1f g, %.2f m.  The truss is %.1fx truer at %.2fx the mass."
              % (bu["name"], bu["mass_g"], bu["dz_static_m"],
                 bu["dz_static_m"] / bt["dz_static_m"], bt["mass_g"] / bu["mass_g"]))

    print("\n--- which guess to measure first " + "-" * 46)
    print("  a 20%% error in each assumed property, and what it moves")
    print("  %-8s %8s %14s %12s" % ("property", "change", "yaw budget", "first mode"))
    for fld, k, dyaw, df in sensitivity(a.length, duty, base, the_nose(sweep.Candidate(
        "truss", side=92, alpha=40, d_chord=3, d_diag=2)) if the_nose else None):
        print("  %-8s %+7.0f%% %13.1f%% %11.1f%%" % (fld, 100 * (k - 1), 100 * dyaw, 100 * df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
