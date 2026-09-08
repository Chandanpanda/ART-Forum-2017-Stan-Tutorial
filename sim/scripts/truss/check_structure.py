"""Tier 0: the structural model against the brief's own arithmetic, and
the design optimiser against the rules it serves.

The brief works one truss by hand (2.3): a 90 mm triangle, 3 mm chords,
EI 4.3e9 N.mm^2, 0.0025 degrees at the tip, 60 g, 228 Hz.  If the model
cannot reproduce that it is not a model of the same thing; if it can,
every other number it produces inherits the credit.

    python3 sim/scripts/truss/check_structure.py [-v]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np

from truss import spec, structure
from truss.spec import Truss, Load, Stock, ring_r_in

VERBOSE = "-v" in sys.argv
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


class _Load(Load):
    pass


def main():
    # ------------------------------------------------ the brief's truss
    brief = Truss(length=1000.0, side=90.0, alpha=45.0, d_chord=3.0, d_diag=2.0,
                  name="brief")
    m = structure.analyse(brief)
    check("brief's truss: EI within 10% of 4.3e9 N.mm^2",
          abs(m["EI"] / 4.3e9 - 1.0) < 0.10, "%.3g" % m["EI"])
    check("brief's truss: I within 10% of 28,600 mm^4",
          abs(m["I"] / 28600.0 - 1.0) < 0.10, "%.0f" % m["I"])
    check("brief's truss: tip slope within 15% of 0.0025 deg under 3 g",
          abs(m["slope_deg"] / 0.0025 - 1.0) < 0.15, "%.5f deg" % m["slope_deg"])
    check("brief's truss: mass within 15% of 60 g",
          abs(m["mass"] / 60.0 - 1.0) < 0.15, "%.1f g" % m["mass"])
    # 228 Hz is a bending-only cantilever figure; with the web's shear
    # compliance the mode is lower.  Both are asserted, because the gap
    # between them is the diagonals' contribution and it is real.
    l = structure.cantilever(brief)
    k_b = 3.0 * structure.EI(brief) / l ** 3 * 1000.0
    m_eff = Load.TIP_MASS * 1e-3 + 0.2357 * brief.mass * 0.5 * 1e-3
    f_bend = np.sqrt(k_b / m_eff) / (2 * np.pi)
    check("brief's truss: bending-only first mode within 15% of 228 Hz",
          abs(f_bend / 228.0 - 1.0) < 0.15, "%.0f Hz" % f_bend)
    check("...and the web's shear takes it lower, not higher",
          m["f1"] < f_bend, "%.0f with shear" % m["f1"])
    # the tube the brief rejects: 20 x 1 roll-wrapped at 70 GPa, I 2701
    I_tube = np.pi / 64.0 * (20.0 ** 4 - 18.0 ** 4)
    slope_tube = np.degrees(structure.tip_load() * 500.0 ** 2 / (2.0 * 70e3 * I_tube))
    check("brief's tube: 0.056 deg, failing the budget 11x",
          abs(slope_tube / 0.056 - 1.0) < 0.10 and slope_tube / Load.SLOPE_BUDGET > 10.0,
          "%.4f deg" % slope_tube)
    # THE BRIEF'S 228 Hz IS BENDING-ONLY.  With the 2 mm web's shear
    # compliance in the model the same truss sits at 195 Hz, five under
    # its own 200 Hz rule -- 45 degrees at 2 mm is a hair short, and the
    # optimiser's 40-degree, 92 mm answer clears it.  A finding, and one a
    # bench would confirm before a 228 was written on a datasheet.
    check("brief's truss fails its own first-mode rule once the web's shear counts, "
          "and nothing else",
          set(structure.violations(brief)) == {"f1"}, str(structure.violations(brief)))

    # -------------------------------------------------- the defaults
    for t, L in ((structure.TRUSS_1M, 1000.0), (structure.TRUSS_300, 300.0)):
        again, grid = structure.design(L, name=t.name)
        check("%s: is what design() returns -- nobody typed it" % t.name,
              again == t, "%s vs %s" % (again, t))
        check("%s: breaks no rule" % t.name, not structure.violations(t),
              str(structure.violations(t)))
        mm = structure.analyse(t)
        check("%s: slope under the budget with the design margin" % t.name,
              mm["slope_deg"] <= Load.SLOPE_BUDGET / Load.SLOPE_SF,
              "%.5f deg" % mm["slope_deg"])
        check("%s: the ring's rim clears the diagonals" % t.name,
              mm["ring_rim_margin"] >= 0.0, "%.2f mm" % mm["ring_rim_margin"])
        feas = [g for g in grid if not g[2]]
        check("%s: it is the lightest feasible candidate" % t.name,
              all(g[1]["mass"] >= mm["mass"] - 1e-9 for g in feas),
              "%d feasible of %d" % (len(feas), len(grid)))
    # ------------------------------------------------ what binds
    rules = structure.binding_rules(structure._GRID_1M)
    check("1m: the first mode and the slope are the binding rules",
          "f1" in list(rules)[:3] and "slope" in list(rules)[:3], str(list(rules)[:4]))
    rules = structure.binding_rules(structure._GRID_300)
    check("300: the ring, not the load, is what binds a short truss",
          any(r.startswith("ring") for r in list(rules)[:3]), str(list(rules)[:4]))
    # ------------------------------------------------ the rules move it
    class Wider(Load):
        SECTION_MAX = 140.0
    wider, _ = structure.design(1000.0, load=Wider)
    check("a wider envelope buys a lighter truss (the envelope binds)",
          wider is not None and wider.mass <= structure.TRUSS_1M.mass + 1e-9,
          "%.1f g at side %.0f" % (wider.mass, wider.side) if wider else "none")

    class Stricter(Load):
        SLOPE_SF = 4.0
    strict, _ = structure.design(1000.0, load=Stricter)
    check("halving the angular budget costs mass or feasibility",
          strict is None or strict.mass > structure.TRUSS_1M.mass - 1e-9,
          "%.1f g" % strict.mass if strict else "infeasible")
    # the ring rule: a steep diagonal needs a bigger ring
    steep = Truss(length=1000.0, side=92.0, alpha=55.0, d_chord=3.0, d_diag=2.0)
    check("a 55-degree diagonal does not fit this ring's rim",
          spec.r_in_needed(steep) > ring_r_in(),
          "needs %.1f, has %.1f" % (spec.r_in_needed(steep), ring_r_in()))
    # mass arithmetic: diagonal mass per face is L rho / cos(alpha), independent of depth
    a = Truss(length=1000.0, side=60.0, alpha=45.0, end_margin=0.0)
    b = Truss(length=1000.0, side=120.0, alpha=45.0, end_margin=0.0)
    check("diagonal mass hardly depends on depth at fixed angle (the brief's insight)",
          abs(a.mass_diags - b.mass_diags) / a.mass_diags < 0.15,
          "%.1f vs %.1f g" % (a.mass_diags, b.mass_diags))
    check("...while the second moment goes with the square of it",
          abs(structure.section_I(b) / structure.section_I(a) - 4.0) < 0.05)
    # cycle estimate inside the brief's budget
    check("1m: the closed-form cycle estimate is inside the 45 min budget",
          structure.cycle_estimate(structure.TRUSS_1M) < 45.0,
          "%.1f min" % structure.cycle_estimate(structure.TRUSS_1M))

    bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det and (VERBOSE or not ok) else ""))
    print("check_structure: %d checks, %d failed" % (len(RESULTS), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
