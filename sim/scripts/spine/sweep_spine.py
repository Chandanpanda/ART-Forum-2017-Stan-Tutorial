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


def named(length, duty):
    """The designs already on the table, for the comparison to start from."""
    out = []
    for web in ("warren", "x"):
        out.append(build.warren_truss(length, 92, 40, 3, 2, tip_mass=duty.tip_mass_g,
                                      web=web, name="derived truss, %s web" % web))
    out.append(build.warren_truss(length, 92, 40, 3, 2, tip_mass=duty.tip_mass_g,
                                  brackets=False, name="derived truss, ends OPEN"))
    out.append(build.warren_truss(length, 90, 45, 3, 2, tip_mass=duty.tip_mass_g,
                                  name="the brief's truss"))
    out.append(build.tube(length, 43, 1.0, tip_mass=duty.tip_mass_g,
                          name="carbon tube 43x1"))
    out.append(build.tube(length, 20, 1.0, tip_mass=duty.tip_mass_g,
                          name="carbon tube 20x1"))
    out.append(build.tube(length, 40, 2.0, material=mat.ALUMINIUM,
                          tip_mass=duty.tip_mass_g, name="aluminium tube 40x2"))
    return out


def sensitivity(length, duty, base):
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
            m = build.warren_truss(length, 92, 40, 3, 2, material=Material(**kw),
                                   tip_mass=duty.tip_mass_g)
            r = metrics.evaluate(m, duty)
            rows.append((fld, k, r["yaw_static_udeg"] / base["yaw_static_udeg"] - 1.0,
                         r["f1_hz"] / base["f1_hz"] - 1.0))
    return rows


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
    a = ap.parse_args()
    duty = duty_mod.SURVEY_MAST if a.mast else duty_mod.QUADROTOR
    print("spine sweep: %.0f mm baseline, %s duty (%.1f g manoeuvre, %.0f K gradient, "
          "%.3f deg budget, accuracy quoted at %.0f m)"
          % (a.length, duty.name, duty.manoeuvre_g, duty.gradient_K,
             duty.yaw_budget_deg, duty.range_m))

    print("\n--- the designs already on the table " + "-" * 42)
    print(metrics.HEADER)
    base = None
    for m in named(a.length, duty):
        r = metrics.evaluate(m, duty)
        if "derived truss, warren" in m.name:
            base = r
        print(metrics.row(r))

    sides = [70, 80, 90, 100, 110, 120, 130, 140]
    alphas = [35, 40, 45, 50]
    if a.wide:
        sides = list(range(60, 165, 5))
        alphas = [30, 35, 40, 45, 50, 55, 60]
    cands = sweep.truss_grid(sides, alphas) + sweep.tube_grid([20, 30, 43, 60], [0.5, 1.0, 2.0])
    t0 = time.time()
    rows = sweep.run(cands, duty, length=a.length, section_max_mm=a.section_max,
                     mass_max_g=a.mass_max, f1_min_hz=a.f1_min)
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
    for fld, k, dyaw, df in sensitivity(a.length, duty, base):
        print("  %-8s %+7.0f%% %13.1f%% %11.1f%%" % (fld, 100 * (k - 1), 100 * dyaw, 100 * df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
