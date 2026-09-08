"""What a truss is worth to the camera, and which truss to make.

THE GEOMETRY IS AN ANSWER, NOT AN INPUT.  The brief gives a 90 mm section
at 45 degrees with 3 mm chords and 2 mm diagonals, and a table showing it
beats every tube.  It also gives the reasons: a 0.005 degree yaw budget,
50 g heads on 500 mm cantilevers under 3 g, a 70 g ceiling, a prop
excitation band, and a ring that can only turn round a joint whose
diagonals stay inside its rim.  Those are the specification; the 90 and
the 45 are one solution to it.  This module makes the solution a function
of the specification, so it is re-derived the day a number changes -- and
so the trade-offs the brief argues in prose become a sweep anybody can
rerun.

The model is a beam, which is what the brief's own arithmetic uses and
what a centre-mounted spine is:

    I    = (n/2) A_chord R^2        three chords on a circle of radius R
    EI   = E I                      the chords carry bending; the web
                                    carries shear
    slope at the tip   P l^2 / 2EI  the number the camera cares about
    K_shear            1.5 E A_diag sin(a) cos^2(a)   Warren web, three
                                    faces averaged over load direction
    f1                 a cantilever with a tip mass, bending + shear
    f_diag, f_chord    the members' own first modes, pinned-pinned

Shear deformation does not rotate the section, so it does not enter the
slope; it enters the deflection and therefore the first mode, and that is
what stops the optimiser from flattening the diagonals to nothing.  Joint
compliance is ignored -- it is the epoxy's job and the bench's to measure.
numpy only.
"""
from math import pi, sin, cos, tan, radians, sqrt

import numpy as np

from .spec import (Truss, Stock, Load, Ring, Gantry, Head, Cage, Process,
                   Dispenser, Cutter, Magazine, Gripper,
                   ring_r_in, ring_r_out, ring_swept_r, r_in_needed)
from . import motion


# ------------------------------------------------------------- stiffness
def section_I(t):
    """Second moment of the chord areas about any centroidal axis, mm^4.
    n equally spaced chords on a circle contribute (n/2) A R^2 whichever
    way the section bends -- the reason a triangle is as good as a
    square here."""
    return (t.n_chords / 2.0) * Stock.area(t.d_chord) * t.R ** 2 \
        + t.n_chords * Stock.I_rod(t.d_chord)


def EI(t):
    return Stock.E * section_I(t)


def shear_stiffness(t):
    """Force per unit shear strain of the web, N.  One diagonal per bay per
    face at angle a carries the face's share of the shear; averaged over
    load direction three faces give 3/2 of one."""
    a = radians(t.alpha)
    return 1.5 * Stock.E * Stock.area(t.d_diag) * sin(a) * cos(a) ** 2


def tip_load(load=Load):
    """N at each camera head under the manoeuvre load."""
    return load.TIP_MASS * 1e-3 * load.G * load.LATERAL_G


def cantilever(t):
    """Centre mount: each half is a cantilever."""
    return t.length / 2.0


def tip_slope_deg(t, P=None, load=Load):
    """Bending slope at the tip, degrees -- the camera's yaw error."""
    P = tip_load(load) if P is None else P
    l = cantilever(t)
    return np.degrees(P * l * l / (2.0 * EI(t)))


def tip_deflection(t, P=None, load=Load):
    P = tip_load(load) if P is None else P
    l = cantilever(t)
    return P * l ** 3 / (3.0 * EI(t)) + P * l / shear_stiffness(t)


def tip_stiffness(t):
    """N/mm at the tip, bending and shear in series."""
    l = cantilever(t)
    return 1.0 / (l ** 3 / (3.0 * EI(t)) + l / shear_stiffness(t))


def f1(t, load=Load):
    """First bending mode with a camera head on each end, Hz."""
    k = tip_stiffness(t) * 1000.0                      # N/m
    m_half = t.mass * (cantilever(t) / t.length) * 1e-3
    m = load.TIP_MASS * 1e-3 + 0.2357 * m_half          # kg, Rayleigh
    return sqrt(k / m) / (2.0 * pi)


def _pinned_mode(d, L_mm):
    """First mode of a pinned-pinned rod, Hz."""
    EI_si = Stock.E * 1e6 * Stock.I_rod(d) * 1e-12      # N.m^2
    rhoA = Stock.rho_lin(d) * 1e-3                       # kg/m
    L = L_mm * 1e-3
    return (pi / 2.0) * sqrt(EI_si / (rhoA * L ** 4))


def f_diag(t):
    return _pinned_mode(t.d_diag, t.L_cut)


def f_chord(t):
    """A chord between two joints."""
    return _pinned_mode(t.d_chord, t.run)


def chord_force(t, P=None, load=Load):
    """Axial force in the most loaded chord under the tip load, N."""
    P = tip_load(load) if P is None else P
    M = P * cantilever(t)
    return M * t.R / section_I(t) * Stock.area(t.d_chord)


def chord_buckling_sf(t, load=Load):
    P_cr = pi ** 2 * Stock.E * Stock.I_rod(t.d_chord) / t.run ** 2
    return P_cr / max(chord_force(t, load=load), 1e-9)


# ------------------------------------------------------------- the ring
def ring_fit(t):
    """The two closed-form clearances the ring needs at a joint, mm.
    Positive is room.  approach.py sweeps the real thing; these are what
    the optimiser can afford to evaluate thousands of times."""
    rim = ring_r_in() - r_in_needed(t)
    # the ring turns round one chord; the other two are `side` away
    others = t.side - (ring_swept_r() + t.d_chord / 2.0 + Process.SEAT_CLEAR)
    # ...and the cage's spine runs down the axis, R away
    spine = t.R - ring_swept_r() - 2.0 * Process.SEAT_CLEAR - Cage.SPINE_R_MIN
    return rim, others, spine


def gap_fits(t):
    """The chord passes through the gap with clearance."""
    from .spec import gap_chord
    return gap_chord() - (t.d_chord + 2.0 * Process.SEAT_CLEAR)


# ------------------------------------------------------------ the clock
def lift_height(t):
    """Closed-form per-joint lift: the C-ring's lowest solid point, at the
    gap's edge, must clear the chord.  approach.py measures the true one
    against the diagonals; this is the floor it cannot go under."""
    return (ring_r_out() * cos(radians(Ring.GAP / 2.0)) + t.d_chord / 2.0
            + Process.LIFT_CLEAR)


def joint_time(t):
    """Seconds per joint: approach, look, lower, wind, park, lift, dose."""
    V, A = Gantry.V_MAX, Gantry.A_MAX
    h = lift_height(t)
    wind = t.turns / Ring.RPM * 60.0 + Ring.SPINUP_S + 0.5
    lower_lift = 2.0 * motion.trap_time(h, V["z"], A["z"])
    dose = (2.0 * motion.trap_time(Head.disp_x(), V["x"], A["x"])
            + 2.0 * motion.trap_time(h + Head.TIP_PARK, V["z"], A["z"])
            + Dispenser.DOSE_S + Process.DOSE_SETTLE_S)
    traverse = motion.trap_time(t.run, V["x"], A["x"])
    return (Gantry.SETTLE_S + Process.VISION_SETTLE_S + lower_lift + wind
            + dose + traverse)


def rod_time(t, is_chord):
    """Seconds to fetch one rod from its rack and seat it."""
    V, A = Gantry.V_MAX, Gantry.A_MAX
    reach = t.R + t.d_chord + Magazine.CLEAR + 3 * Magazine.CHORD_PITCH
    travel = 2.0 * motion.trap_time(reach, V["y"], A["y"])
    strokes = 4.0 * motion.trap_time(Head.GRIP_STROKE * 0.75, V["z"], A["z"])
    yaw = 0.0 if is_chord else (90.0 - t.alpha) / Gripper.YAW_V
    along = motion.trap_time(t.length / 2.0 if is_chord else t.run, V["x"], A["x"])
    return travel + strokes + 2.0 * Gripper.JAW_CLOSE_S + yaw + along \
        + 2.0 * Gantry.SETTLE_S


def index_time(t):
    """A cage index: retract over the cage, turn, come back down."""
    V, A = Gantry.V_MAX, Gantry.A_MAX
    clear = t.R + t.d_chord + Cage.NOTCH_DEPTH + ring_swept_r() + Process.LIFT_CLEAR
    return 120.0 / (6.0 * Cage.THETA_RPM) + 2.0 * motion.trap_time(clear, V["z"], A["z"])


def run_ends_time(t):
    """Anchor at the start of a chord run, cut at its end."""
    V, A = Gantry.V_MAX, Gantry.A_MAX
    legs = Process.ANCHOR_LEGS * motion.trap_time(4.0 * Cage.POST_R + 6.0, V["y"], A["y"])
    return 2.0 * legs + Process.ANCHOR_S + Cutter.CUT_S


def cycle_estimate(t):
    """Closed-form minutes for one truss: load, wind, dose.  schedule.py
    plans the real thing; this is the optimiser's proxy and check_schedule
    holds the two together."""
    n_rods = t.n_chords + t.n_diag
    load = (t.n_chords * rod_time(t, True) + t.n_diag * rod_time(t, False)
            + 2 * t.n_chords * index_time(t))       # chord-up, face-up
    wind = t.n_joints * joint_time(t) + t.n_chords * (index_time(t) + run_ends_time(t))
    return (load + wind) * Process.SPEED_FACTOR / 60.0


# ------------------------------------------------------------ verdicts
def analyse(t, load=Load):
    rim, others, spine = ring_fit(t)
    return {
        "I": section_I(t), "EI": EI(t),
        "slope_deg": tip_slope_deg(t, load=load),
        "slope_test_deg": tip_slope_deg(t, P=load.TEST_FORCE, load=load),
        "deflection": tip_deflection(t, load=load),
        "f1": f1(t, load), "f_diag": f_diag(t), "f_chord": f_chord(t),
        "buckling_sf": chord_buckling_sf(t, load),
        "mass": t.mass, "mass_chords": t.mass_chords,
        "mass_diags": t.mass_diags, "mass_joints": t.mass_joints,
        "n_joints": t.n_joints, "n_diag": t.n_diag, "run": t.run,
        "L_cut": t.L_cut, "mitre_face": t.mitre_face,
        "ring_rim_margin": rim, "ring_others_margin": others,
        "ring_spine_margin": spine,
        "gap_margin": gap_fits(t),
        "cycle_min": cycle_estimate(t),
    }


def mass_ceiling(t, load=Load):
    return load.MASS_MAX if t.length > 600.0 else load.MASS_MAX_300


def violations(t, m=None, load=Load, cycle_max_min=45.0):
    """Which rules this truss breaks.  Empty means feasible."""
    m = analyse(t, load) if m is None else m
    lo, hi = load.BLADE_PASS
    bad = []
    if m["slope_deg"] > load.SLOPE_BUDGET / load.SLOPE_SF:
        bad.append("slope")
    if m["f1"] < load.F1_MIN:
        bad.append("f1")
    if m["f_diag"] < load.MEMBER_F_MIN:
        bad.append("f_diag under the blade-pass ceiling")
    if m["f_chord"] < load.MEMBER_F_MIN:
        bad.append("f_chord under the blade-pass ceiling")
    if not (load.ALPHA_RANGE[0] <= t.alpha <= load.ALPHA_RANGE[1]):
        bad.append("alpha outside the qualified joint range")
    if m["buckling_sf"] < load.BUCKLE_SF:
        bad.append("chord buckling")
    if m["mass"] > mass_ceiling(t, load):
        bad.append("mass")
    if t.side > load.SECTION_MAX:
        bad.append("section envelope")
    if m["ring_rim_margin"] < 0.0:
        bad.append("ring rim")
    if m["ring_others_margin"] < 0.0:
        bad.append("ring vs other chords")
    if m["ring_spine_margin"] < 0.0:
        bad.append("ring vs spine")
    if m["gap_margin"] < 0.0:
        bad.append("gap")
    if m["cycle_min"] > cycle_max_min:
        bad.append("cycle time")
    if t.band < t.mitre_face + 2.0:
        bad.append("band shorter than the mitre")
    if t.n_nodes < 3:
        bad.append("too few joints to be a truss")
    return bad


def design(length, load=Load, name=None, sides=None, alphas=None,
           d_chords=(2.0, 3.0), d_diags=(1.0, 1.5, 2.0), cycle_max_min=45.0,
           **fields):
    """The lightest truss of this length that meets every rule.

    Returns (truss, grid) where grid is every candidate with its verdict,
    kept because the SHAPE of the feasible region says which rule binds --
    and that is the answer worth having when a rule changes.  Ties in mass
    go to the larger slope margin.
    """
    sides = np.arange(24.0, load.SECTION_MAX + 1e-9, 2.0) if sides is None else sides
    alphas = np.arange(25.0, 66.0, 1.0) if alphas is None else alphas
    grid, best, best_key = [], None, None
    for dc in d_chords:
        for dd in d_diags:
            if dd > dc:
                continue
            for a in alphas:
                for s in sides:
                    t = Truss(length=float(length), side=float(s), alpha=float(a),
                              d_chord=float(dc), d_diag=float(dd),
                              name=name or "truss_%d" % int(length), **fields)
                    if t.length - 2.0 * t.end_margin < 2.0 * t.run:
                        continue
                    m = analyse(t, load)
                    bad = violations(t, m, load, cycle_max_min)
                    grid.append((t, m, bad))
                    if not bad:
                        key = (m["mass"], m["slope_deg"])
                        if best_key is None or key < best_key:
                            best, best_key = t, key
    return best, grid


def binding_rules(grid):
    """How often each rule was the reason a candidate failed."""
    out = {}
    for _t, _m, bad in grid:
        for b in bad:
            out[b] = out.get(b, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# ------------------------------------------------------- the defaults
# THE DEFAULT TRUSSES ARE DERIVED.  check_structure asserts they are still
# what design() returns, so nobody can quietly type a section in.
TRUSS_1M, _GRID_1M = design(1000.0, name="camera_1m")
TRUSS_300, _GRID_300 = design(300.0, name="camera_300")
