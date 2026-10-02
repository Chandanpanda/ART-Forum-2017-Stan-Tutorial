"""Tier 0 and Tier 1: the bat derived from its spec -- every fit, every
overlap, every mass and both bands -- on the default bat and on a size the
derivation has never seen; then the derived bat built in MuJoCo and
measured against the derivation.

    python3 sim/scripts/batline/check_bat.py [-v]

The plan's contract for this suite: it fails when a part no longer fits
the spec (board in slot, cells in bay, caps, collar, thread, tube), mass
or balance leaves the spec, or a band leaves its zone.

A check that cannot see a bad bat cannot fail, so this one also derives
bats that MUST fail, and fails if they pass.  Most are the default bat with
one spec pushed just past the room its fit reports -- the handle shortened
by the stack's margin and a hair, the thread deepened by the root's -- and
must fail by exactly that hair, which proves two things at once: that the
fit can fail, and that its margin is the real room, not a number that
merely looks like one.  The rest push a spec past what any frame, printer
or knit can take: a grip no collar passes, a bed the core cannot lie on.
"""
import os
import sys
from contextlib import contextmanager
from dataclasses import replace
from math import sqrt, pi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np

from truss import structure
from truss.spec import Gantry
from batline import product as P
from batline import spec as S
from batline.spec import KID, Bat, Board, Print, Knit, CapSpring, Printer, Rules

VERBOSE = "-v" in sys.argv
RESULTS = []
# A bat the derivation was never developed on: an adult's, longer, with a
# longer handle, a thicker grip and a bigger envelope.  Test data, not a rule.
ADULT = Bat("adult", length=860.0, handle=300.0, grip_d=34.0, width=108.0, depth=62.0)
HAIR = 0.05            # mm (or g) past a margin: well above the arithmetic's noise


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def say(*a):
    if VERBOSE:
        print(*a)


@contextmanager
def patched(obj, **kw):
    """Spec attributes changed for one derivation, and put back."""
    old = {k: getattr(obj, k) for k in kw}
    for k, v in kw.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


def fit_of(d, start):
    return next(f for f in d.fits if f.name.startswith(start))


# ================================================================ the spec
def spec_checks():
    for name, ok in S.CHECKS:
        check("spec: " + name, ok)


# ======================================================= one derived bat
def raster_bound(perimeter, res):
    """The most a raster at `res` can miscount an area: every cell the
    boundary crosses lies within half a diagonal of it."""
    return perimeter * res * sqrt(2.0)


def derived(bat):
    tag = bat.name
    try:
        d = P.design(bat)
    except P.NoFrame as e:
        check("%s: a frame fits" % tag, False, str(e))
        return None
    check("%s: a frame fits" % tag, True)
    for f in d.fits:
        check("%s: %s" % (tag, f.name), f.ok, "%.3f%s" % (f.margin, ("; " + f.detail) if f.detail else ""))
    t = d.frame.truss
    xr, yr = structure.reach(t)
    check("%s: the truss cell's gantry reaches every joint of the frame" % tag,
          xr <= Gantry.X_TRAVEL and yr <= Gantry.Y_TRAVEL / 2.0,
          "needs %.0f x %.0f of %.0f x %.0f" % (xr, 2 * yr, Gantry.X_TRAVEL, Gantry.Y_TRAVEL))

    # THE SEARCH'S ESTIMATE, against the parts: the frame was chosen on an
    # estimate of what the blade carries; it must still hold the rated swing
    # with what the parts actually weigh, where they actually are
    lay = d.lay
    load = P.SwingLoad(lay.x_hands, np.arange(0.0, bat.length + 5.0, 5.0))
    tip = d.parts["tip_cap"].props
    Gu, (Gp,) = load.lever_terms(lay.x_blade0, lay.x_c1, [tip.c[0]])
    sleeve_blade = Knit.GSM * 1e-6 * d.sleeve["P_blade"] * (lay.x_tip0 - lay.x_blade0)
    M = P.peak_moment_of(Gu, Gp)(t.mass + sleeve_blade, tip.m)
    sf = P.chord_buckling_sf(t, M)
    check("%s: the frame holds the rated swing with the parts' real masses, not the search's estimate"
          % tag, sf >= Rules.BUCKLE_SF,
          "buckling x%.2f (search said x%.2f, rule x%.1f)" % (sf, d.frame.buckling_sf, Rules.BUCKLE_SF))

    # THE ANALYTIC AREA'S PRECONDITION, on the sections the search estimates
    # with: holes inside the solid and apart, so solid - holes is the area
    hull, sh, c_in, c_out = d.offsets
    socks = tuple(P.circle(y, z, t.d_chord / 2.0 + Print.CLEAR) for y, z in P._tri_vertices(d.R))
    for name, sec in (("shoulder", P.Section(P.rtri(d.R, sh), socks)),
                      ("collar", P.Section(P.rtri(d.R, c_out), (P.rtri(d.R, c_in),)))):
        y0, y1, z0, z1 = sec.bbox()
        Y, Z = np.meshgrid(np.arange(y0, y1, P.RES), np.arange(z0, z1, P.RES), indexing="ij")
        solid = sec.solid.mask(Y, Z)
        holes = [h.mask(Y, Z) for h in sec.holes]
        inside = all(not (h & ~solid).any() for h in holes)
        apart = all(not (a & b).any() for i, a in enumerate(holes) for b in holes[i + 1:])
        A = P.moments(sec)[0]
        bound = raster_bound(sec.perimeter, P.RES)
        check("%s: the %s's holes lie inside it and apart, so its analytic area holds" % (tag, name),
              inside and apart and abs(A - sec.area_exact) <= bound,
              "grid %.2f, analytic %.2f mm^2, raster bound %.2f" % (A, sec.area_exact, bound))

    # OVERLAPS: only the declared ones, at their declared areas
    found = {}
    for a, b, x, area in P.interferences(d):
        found.setdefault(frozenset((a, b)), []).append((x, area))
    declared = P.declared_overlaps(d)
    stray = {k: v for k, v in found.items() if k not in declared}
    check("%s: no two parts occupy the same space, except the declared fits" % tag, not stray,
          "; ".join("%s at x %.1f, %.2f mm^2" % ("/".join(sorted(k)), v[0][0], v[0][1])
                    for k, v in stray.items()))
    r = t.d_chord / 2.0
    perim = {frozenset(("collar", "sleeve")): P.rtri(d.R, sh + Knit.T).perimeter
             + P.rtri(d.R, c_in).perimeter,
             frozenset(("tip_cap", "frame")): 3.0 * 2.0 * pi * (2.0 * r - Print.INTERFERENCE)}
    for k, area in declared.items():
        got = found.get(k, [])
        bound = raster_bound(perim[k], 0.1)
        check("%s: %s overlap as declared" % (tag, " and ".join(sorted(k))),
              got and all(abs(a - area) <= bound for _, a in got),
              "declared %.2f mm^2, found %s, raster bound %.1f"
              % (area, [round(a, 2) for _, a in got] or "none", bound))

    # THE BANDS: printed where band_print says, each lands on its zone's
    # centre; with the print's and the placement's scatter it stays inside
    tol = Knit.PRINT_TOL + Knit.PLACE_TOL
    for i, ((c, (lo, hi)), u) in enumerate(zip(d.bands, d.band_print)):
        x = P.blank_to_frame(d, u)
        check("%s: band %d, printed %.0f mm from the blank's tip, lands on its zone" % (tag, i + 1, u),
              abs(x - c) <= 1e-6 and lo - 1e-9 <= x - tol and x + tol <= hi + 1e-9,
              "at %.3f, zone %.1f-%.1f" % (x, lo, hi))
    say(P.report(d))
    return d


# ====================================================== bats that must fail
def must_fail(what, run, fit_start, expect_margin=None):
    """Derive a bat that must fail `fit_start`; with expect_margin, its
    margin must be exactly that -- the spec pushed past by a hair."""
    d = run()
    f = fit_of(d, fit_start)
    exact = expect_margin is None or abs(f.margin - expect_margin) <= 1e-6
    check("must fail: %s" % what, not f.ok and exact,
          "'%s' %.4f%s" % (f.name, f.margin,
                           "" if expect_margin is None else " (expected %.4f)" % expect_margin))
    say("  %-60s %.4f" % (what, f.margin))
    return d


def must_fail_frame(what, bat, reason):
    try:
        P.design(bat)
    except P.NoFrame as e:
        top = next(iter(e.binding), None)
        check("must fail: %s" % what, top == reason, "NoFrame, binding %s" % list(e.binding.items())[:2])
        return
    check("must fail: %s" % what, False, "a frame was found")


def negative_controls(d):
    """The default bat, one spec at a time pushed past its fit's room."""
    lay, core, k = d.lay, d.core, KID
    h = HAIR

    # CELLS IN THE BAY: the handle shortened past the stack's room
    m = fit_of(d, "the stack and its floor").margin
    must_fail("cells: a handle %.2f mm shorter than the stack needs" % h,
              lambda: P.design(replace(k, handle=k.handle - m - h)), "the stack and its floor", -h)

    # BOARD IN THE SLOT, CELLS IN THE BAY, THE THREAD: everything under the
    # thread's root, which a deeper thread or a thinner grip squeezes
    m = fit_of(d, "the bay, the slot").margin
    with patched(Print, THREAD_DEPTH=Print.THREAD_DEPTH + m + h):
        must_fail("thread: a thread %.2f mm deeper than the root allows" % h, lambda: P.design(k),
                  "the bay, the slot", -h)
    must_fail("board and cells: a grip %.2f mm thinner than the bay and slot need" % (2 * h),
              lambda: P.design(replace(k, grip_d=k.grip_d - 2.0 * (m + h))), "the bay, the slot", -h)

    # BOARD IN THE SLOT: a board whose length scatters more than its finger's tab allows
    m = fit_of(d, "finger 1 lands").margin
    with patched(Board, L_TOL=Board.L_TOL + m + h):
        must_fail("board: an outline tolerance %.2f mm past the finger's tab" % h, lambda: P.design(k),
                  "finger 1 lands", -h)

    # THE POMMEL CAP: a spring whose base coil misses the return ring
    f = fit_of(d, "the spring's base coil")
    ring_in, ring_out = core.r_bay, core.r_bay + core.w_between - Print.CLEAR
    grow = (ring_out - CapSpring.D_BASE / 2.0) <= (CapSpring.D_BASE / 2.0 - ring_in)
    with patched(CapSpring, D_BASE=CapSpring.D_BASE + (2.0 if grow else -2.0) * (f.margin + h)):
        must_fail("cap: a spring's base coil %.2f mm off the return ring" % h, lambda: P.design(k),
                  "the spring's base coil", -h)

    # THE COLLAR: a grip fatter than the largest frame's collar can pass
    g_max = 0.0
    for dc in Rules.D_CHORDS:
        sides = P.frame_sides(k, dc)
        if len(sides):
            c_in = P._section_offsets(dc)[2]
            g_max = max(g_max, 2.0 * (P.rtri_inradius(sides[-1] / sqrt(3.0), c_in) - Print.CLEAR))
    must_fail_frame("collar: a grip %.2f mm fatter than any frame's collar passes" % (2 * h),
                    replace(k, grip_d=g_max + 2.0 * h), "the collar cannot pass the grip")

    # THE TUBE: sized by formula from the widest section; if the formula
    # missed a part by more than the tube's room, the parts must say so
    m = fit_of(d, "the bat, every part counted, slides into its tube").margin
    real = P.rtri_circumradius
    with patched(P, rtri_circumradius=lambda R, rc: real(R, rc) - (m + h)):
        must_fail("tube: a tube formula that misses a part by %.2f mm" % h, lambda: P.design(k),
                  "the bat, every part counted, slides into its tube", -h)

    # MASS AND BALANCE, once the spec sets them
    m_all = sum(p.mass for p in d.parts.values())
    must_fail("mass: a limit %.2f g under the bat" % h,
              lambda: P.design(replace(k, mass_max=m_all - h)), "the bat, cells in", -h)
    bal = d.balance
    must_fail("balance: a range starting %.2f mm past the balance point" % h,
              lambda: P.design(replace(k, balance=(bal + h / k.length, 1.0))), "its balance point", -h)

    # THE BANDS: zones that need more room than the blade has
    m = fit_of(d, "the two bands' zones stay apart").margin
    with patched(Knit, PLACE_TOL=Knit.PLACE_TOL + (m + h) / 4.0):
        must_fail("bands: a placement scatter that runs the two zones %.2f mm into each other" % h,
                  lambda: P.design(k), "the two bands' zones stay apart", -h)
    # ...and a knit that necks 0.01 less than Knit says, so a band printed
    # where band_print says lands short of where it should
    tol = Knit.PRINT_TOL + Knit.PLACE_TOL
    worst = max(abs(P.blank_to_frame(d, u, Knit.AXIAL_PER_HOOP - 0.01) - c) + tol - (hi - c)
                for (c, (lo, hi)), u in zip(d.bands, d.band_print))
    check("must fail: bands: a knit necking 0.01 less than its spec puts a band out of its zone",
          worst > 0.0, "%.2f mm past the zone's edge" % worst)
    say("  band 1 moves %.2f mm per 0.01 of the knit's necking: M7's sleeve rig has to measure it"
        % abs(P.blank_to_frame(d, d.band_print[0], Knit.AXIAL_PER_HOOP - 0.01) - d.bands[0][0]))

    # PRINTING: a bed smaller than the core can lie on, even across its diagonal
    span, wid = d.printing["core"]["footprint"]
    side = (span + wid) / sqrt(2.0) - h
    with patched(Printer, BUILD=(side, side, Printer.BUILD[2])):
        must_fail("printing: a bed %.2f mm short of the core's diagonal" % h, lambda: P.design(k),
                  "every printed part fits a printer's bed")

    # NO FRAME: an envelope shallower than the smallest frame the grid tries
    depth = k.depth
    while any(len(P.frame_sides(replace(k, depth=depth), dc)) for dc in Rules.D_CHORDS):
        depth -= 1.0
    must_fail_frame("frame: an envelope %.0f mm deep, under the smallest frame" % depth,
                    replace(k, depth=depth), "the envelope is narrower than the smallest frame the grid tries")


# ============================================================ in MuJoCo
def grid_shift(part):
    """How far the model's boxes can sit from product.py's material: a
    voxel's diagonal for a solid prism, a stave's sagitta for a shell, and
    for everything else the micrometre the XML is written to."""
    from batline import mjcf
    s = mjcf.WRITTEN * sqrt(3.0) / 2.0
    for _, _, sec in part.prisms:
        if mjcf._is_shell(sec):
            n = 4 * mjcf.N_ARC if sec.solid.kind == "circle" else 3 * mjcf.N_ARC
            r = sec.solid.p[-1]
            s = max(s, r * (1.0 - np.cos(pi / n)))
        else:
            s = max(s, mjcf.RES_MJ * sqrt(2.0))
    return s


def in_mujoco(d):
    import mujoco
    from batline import mjcf
    tag = d.bat.name
    m, data = mjcf.build(d)
    say("  %s in MuJoCo %s: %d geoms, %d bodies" % (tag, mujoco.__version__, m.ngeom, m.nbody))

    # THE FITS, against the liners: what positions a part, to the micrometre
    for group, kind, expected, measured, allowed in mjcf.measure_liners(d, m, data):
        check("%s mj: %s fit (%s) measures as derived" % (tag, group, kind),
              abs(measured - expected) <= allowed,
              "expected %.4f, measured %.4f, allowed %.4f mm" % (expected, measured, allowed))
    _, fits = mjcf.liners(d)
    bad = []
    for a, b, dist in mjcf.contacts(m, data):
        liner, other = (a, b) if "/liner/" in a else (b, a)
        if "/liner/" not in liner:
            bad.append("%s on %s" % (a, b))
            continue
        part, rest = liner.split("/liner/")
        group = "%s:%s" % (part, rest.split("/")[0])
        located, expected, kind = fits.get(group, ("?", 0.0, "?"))
        if not other.startswith(located) or kind == "clear":
            bad.append("%s on %s" % (liner, other))
        elif kind == "datum" and abs(dist) > 1e-3:
            bad.append("%s on %s at %.4f" % (liner, other, dist))
        elif kind == "press" and abs(-dist - expected) > mjcf.sagitta(t_r(d)) + 1e-3:
            bad.append("%s on %s at %.4f" % (liner, other, dist))
    check("%s mj: every contact is a datum touching or the press fit pressing" % tag, not bad,
          "; ".join(bad[:4]))

    # THE MASSES: each body weighs its part, sits where it does and spreads
    # its mass as it does, within what the grid can move it
    L = d.bat.length
    bodies = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(1, m.nbody)}
    check("%s mj: every derived part is a body, and every body a part" % tag, bodies == set(d.parts),
          "missing %s, extra %s" % (sorted(set(d.parts) - bodies), sorted(bodies - set(d.parts))))
    bodies &= set(d.parts)
    for b in range(1, m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        if name not in bodies:
            continue
        pp = d.parts[name].props
        aux = 1e-6 * sum(1 for g in range(m.ngeom) if m.geom_bodyid[g] == b
                         and m.geom_contype[g] + m.geom_conaffinity[g] > 0)    # liners, stand-ins
        mb = m.body_mass[b] * 1000.0
        dm = mb - pp.m
        delta = grid_shift(d.parts[name])
        dc = float(np.linalg.norm(data.xipos[b] * 1000.0 - pp.c))
        Rb = data.ximat[b].reshape(3, 3)
        Ib = Rb @ np.diag(m.body_inertia[b]) @ Rb.T * 1e9
        dI = np.abs(np.diag(Ib) - np.diag(pp.I))
        spheres = 0.4 * sum(mp for _, mp in d.parts[name].points) * mjcf.POINT_R ** 2
        bound = (2.0 * delta * np.sqrt(pp.m * np.diag(pp.I)) + pp.m * delta ** 2 + aux * L * L
                 + spheres + 1e-6 * np.diag(pp.I))
        check("%s mj: the %s weighs what it is derived to" % (tag, name),
              abs(dm - aux) <= 1e-6 * pp.m + 1e-9, "%.6f vs %.6f g" % (mb, pp.m))
        check("%s mj: the %s's centre and inertia are its part's, within the grid" % (tag, name),
              dc <= delta + aux * L / pp.m + 1e-6 and (dI <= bound).all(),
              "centre off %.4f mm (grid %.3f); inertia off %s of %s g mm^2"
              % (dc, delta, np.round(dI, 1), np.round(bound, 1)))
    M, c, _ = mjcf.composite(m, data)
    pw = d.props(True)
    aux = 1e-6 * sum(1 for g in range(m.ngeom) if m.geom_contype[g] + m.geom_conaffinity[g] > 0)
    check("%s mj: the bat weighs what it is derived to" % tag,
          abs(M - aux - pw.m) <= 1e-6 * pw.m, "%.4f vs %.4f g" % (M, pw.m))
    check("%s mj: the bat's balance point is the derived one" % tag,
          abs(c[0] - pw.c[0]) <= 1e-3, "%.4f vs %.4f mm" % (c[0], pw.c[0]))
    return mujoco.__version__


def t_r(d):
    return d.frame.truss.d_chord / 2.0 - Print.INTERFERENCE


def main():
    from truss.glenv import headless
    headless()                       # before mujoco
    spec_checks()
    designs = [derived(KID), derived(ADULT)]
    if designs[0] is not None:
        negative_controls(designs[0])
    version = "?"
    for d in designs:
        if d is not None:
            version = in_mujoco(d)
    bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det and (VERBOSE or not ok) else ""))
    print("check_bat: %d checks, %d failed (MuJoCo %s)" % (len(RESULTS), bad, version))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
