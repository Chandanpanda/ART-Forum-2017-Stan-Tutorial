"""The design sweep: which spine holds calibration best, per gram.

RANKED ON RANGE ERROR, NOT ON STIFFNESS.  Every candidate is solved and
its compliances converted into metres of range error at the duty's stated
range, so a design that is stiff in the wrong direction, or thermally
sensitive, or resonant in the propeller band, is penalised by exactly what
it costs the customer.  Mass is the other axis, because mass is the only
reason a lattice exists at all.

The tube is swept alongside as a candidate rather than quoted from a
table.  If it wins, the cell in sim/truss should not be built.
"""
from dataclasses import dataclass, asdict, replace
import itertools

import numpy as np

from . import build, metrics, material as mat

# what the shop keeps, so a sweep cannot recommend a rod nobody sells
STOCK_MM = (1.0, 1.5, 2.0, 3.0)


@dataclass(frozen=True)
class Candidate:
    kind:  str = "truss"
    side:  float = 92.0
    alpha: float = 40.0
    d_chord: float = 3.0
    d_diag:  float = 2.0
    web:   str = "warren"
    d_out: float = 43.0        # tube only
    wall:  float = 1.0
    # THE CAMERA MOUNT IS PART OF THE DESIGN, not an accessory bolted on
    # after.  Left out, the sweep charges the mount no mass and no
    # compliance and lands the camera's mass on the chord ends 66 mm off
    # the axis -- which is not conservative, it is a different structure.
    nose:  object = None

    def model(self, length, duty, material=None):
        if self.kind == "truss":
            return build.warren_truss(length, self.side, self.alpha, self.d_chord,
                                      self.d_diag, material=material or mat.CARBON_UD,
                                      tip_mass=duty.tip_mass_g, web=self.web,
                                      nose=self.nose)
        return build.tube(length, self.d_out, self.wall,
                          material=material or mat.CARBON_WRAP, tip_mass=duty.tip_mass_g)

    @property
    def label(self):
        if self.kind == "truss":
            return "%s %.0f/%.0f/%.1f/%.1f" % (self.web, self.side, self.alpha,
                                               self.d_chord, self.d_diag)
        return "tube %.0fx%.1f" % (self.d_out, self.wall)


def truss_grid(sides, alphas, chords=STOCK_MM, diags=STOCK_MM, webs=("warren",),
               nose=None):
    """Every truss the shop could cut, with the diagonal never fatter than
    the chord it lies on.

    `nose` may be a Nose or a CALLABLE of the candidate.  It has to be able
    to be a callable: the camera's standoff is solved from the section it
    is bolted to -- a deeper truss puts more of itself in a 66 degree field
    -- so a single nose shared across the grid would price the deep
    sections as if they were free of it.  What that function knows about
    cameras stays outside this package.
    """
    out = []
    for s, a, dc, dd, w in itertools.product(sides, alphas, chords, diags, webs):
        if dd > dc:
            continue
        c = Candidate("truss", side=s, alpha=a, d_chord=dc, d_diag=dd, web=w)
        out.append(replace(c, nose=nose(c)) if callable(nose) else replace(c, nose=nose))
    return out


def tube_grid(d_outs, walls):
    return [Candidate("tube", d_out=d, wall=w) for d, w in itertools.product(d_outs, walls)
            if w < d / 4.0]


def run(candidates, duty, length=1000.0, mass_max_g=None, section_max_mm=None,
        f1_min_hz=None, buildable=None, rejects=None):
    """Evaluate every candidate; returns rows sorted by range error.

    CONSTRAINTS ARE APPLIED, NOT ASSUMED AWAY.  Range error falls
    monotonically with section depth and with chord diameter, so an
    unconstrained sweep does not find an optimum, it finds the edge of the
    grid and reports it with a straight face.  What stops it is packaging
    (the section has to fit the airframe) and the modes (a deep section on
    thin diagonals is statically excellent and dynamically soft).  Every
    row keeps a `violates` list so a rejected design can still be read.

    AND EVERY RULE THIS MODEL CANNOT SEE.  `rejects` is an optional
    predicate on a Candidate returning a list of broken rule names, and it
    is how the rules that belong to another model get applied here without
    this package learning them.  A frame solver has no view of a diagonal's
    own first mode against a propeller band, of whether a winding ring fits
    round a joint, or of how long the cell takes to build the thing --
    and each of those has rejected a design this sweep ranked first.

    AND WHETHER THE MACHINE CAN MAKE IT.  `buildable` is an optional
    predicate on a Candidate returning a margin in mm -- positive is
    buildable -- and it is the fourth constraint.  It is passed IN rather
    than imported, so this package stays independent of any one cell: the
    caller supplies its own machine's rule.  It is not optional in
    practice.  Without it this sweep named 120/45/3.0/1.5 as the lightest
    design holding the accuracy budget, and every design holding that
    budget wants a web of 45 degrees or steeper, which the winding head's
    bore cannot enter -- 28 designs met the budget and NONE of them could
    be made.  A sweep that does not know what the shop can build is a
    sweep that answers a different question.
    """
    rows = []
    for c in candidates:
        m = c.model(length, duty)
        r = metrics.evaluate(m, duty)
        r["candidate"] = c
        r["name"] = c.label
        bad = []
        if section_max_mm and c.kind == "truss" and c.side > section_max_mm:
            bad.append("section")
        if section_max_mm and c.kind == "tube" and c.d_out > section_max_mm:
            bad.append("section")
        if mass_max_g and r["mass_g"] > mass_max_g:
            bad.append("mass")
        if f1_min_hz and r["f1_hz"] < f1_min_hz:
            bad.append("f1")
        r["build_mm"] = float(buildable(c)) if buildable is not None else float("inf")
        if r["build_mm"] < 0.0:
            bad.append("buildable")
        if rejects is not None:
            bad.extend(rejects(c))
        r["violates"] = bad
        r["feasible"] = not bad
        rows.append(r)
    rows.sort(key=lambda r: (not r["feasible"], r["dz_static_m"]))
    return rows


def feasible(rows):
    return [r for r in rows if r["feasible"]]


def pareto(rows, x="mass_g", y="dz_static_m"):
    """The designs nothing else beats on both axes."""
    out = []
    for r in sorted(rows, key=lambda r: (r[x], r[y])):
        if not out or r[y] < out[-1][y] - 1e-12:
            out.append(r)
    return out


def smallest_meeting(rows, key="budget_used", limit=1.0, by="mass_g"):
    """The lightest FEASIBLE design that stays inside the budget, or None."""
    ok = [r for r in rows if r["feasible"] and r[key] <= limit]
    return min(ok, key=lambda r: r[by]) if ok else None


def binding(rows):
    """Which constraint the best infeasible designs run into -- the one to
    argue with if the answer is not good enough."""
    out = {}
    for r in sorted(rows, key=lambda r: r["dz_static_m"])[:40]:
        for v in r["violates"]:
            out[v] = out.get(v, 0) + 1
    return out
