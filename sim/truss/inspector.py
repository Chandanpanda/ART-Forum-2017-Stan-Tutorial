"""What was made, judged: the referee for a truss.

Pure function of the final state -- band states, where the rods ended up,
what landed on each joint, the clock -- against the recipe and the brief's
acceptance table (6).  Nothing here is scored on intent: a band the
planner meant to lay counts only if the ring's angle history says it was
laid, and a rod counts as seated only if it is measured where the fixture
says it should be.  numpy only, so the planner and the checks share one
definition of "good".
"""
from dataclasses import dataclass, field

import numpy as np

from .spec import Load
from . import band as _band


# process tolerances: what a joint must satisfy to be the joint the DoE
# qualified.  The recipe's band is centred on the joint by construction;
# CENTRE_TOL is how far it may miss and still cover the mitre with margin.
TURN_TOL     = 0.5        # turns short of the recipe still counted complete
WIDTH_TOL    = 0.15       # fraction of the band
CENTRE_TOL   = 1.0        # mm
DOSE_MIN     = 0.8        # fraction of the planned dose that must land
SEAT_TOL     = 0.10       # mm, the brief's placement claim
STRAIGHT_TOL = 0.5        # mm over the length (brief 6)


@dataclass
class JointReport:
    joint:   int
    turns:   float
    width:   float
    centre_err: float
    thread_mm: float
    dosed_mg: float
    drop_hit: bool
    why:     list = field(default_factory=list)

    @property
    def ok(self):
        return not self.why


@dataclass
class TrussReport:
    joints:   list
    seated:   dict            # rod -> error mm, from measured positions
    mass_g:   float           # thread + resin
    cycle_min: float
    why:      list = field(default_factory=list)

    @property
    def n_ok(self):
        return sum(1 for j in self.joints if j.ok)

    @property
    def ok(self):
        return not self.why and all(j.ok for j in self.joints)

    def __repr__(self):
        return ("<truss %d/%d joints, %d rods seated of %d, %.2f g on the joints, "
                "%.1f min%s>" % (self.n_ok, len(self.joints),
                                 sum(1 for e in self.seated.values() if e <= SEAT_TOL),
                                 len(self.seated), self.mass_g, self.cycle_min,
                                 "" if self.ok else ": " + "; ".join(self.why)))


def inspect_joint(geom, joint, state, dose_mg=None):
    """One joint against the recipe."""
    t = geom.t
    r = JointReport(joint.index, state.turns, state.width,
                    abs(state.centre - joint.x), state.thread, state.dosed,
                    state.drop_hit)
    if state.turns < t.turns - TURN_TOL:
        r.why.append("%.1f of %d turns" % (state.turns, t.turns))
    if state.turns > 0 and abs(state.width - t.band) > WIDTH_TOL * t.band:
        r.why.append("band %.1f mm, recipe %.1f" % (state.width, t.band))
    if state.turns > 0 and r.centre_err > CENTRE_TOL:
        r.why.append("band centre %.1f mm off the joint" % r.centre_err)
    if not state.anchored:
        r.why.append("run not anchored")
    want = _band.resin_dose_mg(t, state.thread) if dose_mg is None else dose_mg
    if state.dosed < DOSE_MIN * want:
        r.why.append("dosed %.0f of %.0f mg" % (state.dosed, want))
    return r


def seat_error(geom, rod, p0_meas, p1_meas, theta=0.0):
    """How far a measured rod lies from where the fixture holds it: the
    larger of its two ends' distances from the nominal centreline."""
    from .geometry import rot_x
    Rm = rot_x(theta)
    a, b = Rm @ rod.p0, Rm @ rod.p1
    if rod.kind == "diag":
        a, b = (Rm @ p for p in geom.diag_body_ends(rod))
    u = (b - a) / np.linalg.norm(b - a)

    def off(p):
        v = np.asarray(p, float) - a
        return float(np.linalg.norm(v - (v @ u) * u))
    return max(off(p0_meas), off(p1_meas))


def inspect_truss(geom, states, seated=None, cycle_s=0.0, load=Load):
    """The whole truss.  `states` maps joint index -> BandState; `seated`
    maps rod index -> measured (p0, p1, theta) or a precomputed error."""
    t = geom.t
    js = []
    thread = resin = 0.0
    for j in geom.joints:
        s = states.get(j.index)
        if s is None:
            s = _band.BandState(j.index)
        js.append(inspect_joint(geom, j, s))
        thread += s.thread
        resin += s.dosed
    errs = {}
    for k, v in (seated or {}).items():
        if isinstance(v, (int, float)):
            errs[k] = float(v)
        else:
            errs[k] = seat_error(geom, geom.rods[k], *v)
    mass = _band.thread_mass_mg(thread, t) / 1000.0 + resin / 1000.0
    rep = TrussReport(js, errs, mass, cycle_s / 60.0)
    bad = [j for j in js if not j.ok]
    if bad:
        rep.why.append("%d joints fail" % len(bad))
    unseated = [k for k, e in errs.items() if e > SEAT_TOL]
    if unseated:
        rep.why.append("%d rods off their seat" % len(unseated))
    cap = load.MASS_MAX if t.length > 600 else load.MASS_MAX_300
    if t.mass_chords + t.mass_diags + mass > cap:
        rep.why.append("%.1f g over the %.0f g ceiling" % (t.mass_chords + t.mass_diags + mass, cap))
    if rep.cycle_min > 45.0:
        rep.why.append("%.0f min cycle" % rep.cycle_min)
    return rep
