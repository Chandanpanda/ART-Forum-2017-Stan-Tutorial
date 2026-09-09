"""Does the fixture come out?

    python3 sim/scripts/truss/check_collapse.py [-v]

THE FINDING THIS SUITE EXISTS FOR.  The cage was drawn as a thing to wind
onto and never as a thing to get back.  The truss closes around it: 96 pins
and 54 cradles, every one of them 15 to 25 mm outside the only hole they
could leave through.  Nothing in the cell noticed, because every check asked
whether a feature was in the right PLACE and none asked whether it could
ever be anywhere else.

So the first check here is that the cage as drawn is trapped.  If that one
ever passes trivially, the rest of the suite is measuring nothing.
"""
import os
import sys
from math import degrees, radians, sqrt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np

from truss import geometry, fixture, collapse
from truss.spec import Truss, Cage, Process, Stock, Load
from spine.chosen import OPTIMAL_1M, OPTIMAL_300

RESULTS = []
VERBOSE = "-v" in sys.argv


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))


def cases():
    for ch in (OPTIMAL_1M, OPTIMAL_300):
        t = Truss(**ch.as_truss_kwargs())
        g = geometry.TrussGeometry(t)
        yield ch.name, t, g, collapse.Collapse(fixture.Fixture(g))


def main():
    for name, t, g, c in cases():
        f = c.f
        tag = "[%s] " % name
        exit_r = c.exit_radius()

        # ---------------------------------------------- IT IS TRAPPED
        check(tag + "the cage as drawn CANNOT leave the truss it built: every "
              "locating feature stands outside the section's own exit hole",
              c.feature_radius() > exit_r,
              "features reach %.2f mm, the exit is %.2f -- %d pins and %d cradles, "
              "all of them outside" % (c.feature_radius(), exit_r,
                                       len(f.pins), len(f.cradles)))
        check(tag + "...and the exit hole is the SECTION's, not the chord circle's: a "
              "diagonal lies in a face, so what withdraws has to clear the "
              "inscribed circle",
              abs(exit_r - (t.side / (2.0 * sqrt(3.0))
                            - max(t.d_chord, t.d_diag) / 2.0
                            - Process.SEAT_CLEAR)) < 1e-9,
              "%.2f mm of %.2f inscribed" % (exit_r, t.side / (2.0 * sqrt(3.0))))

        # ---------------------------------------------- IT COLLAPSES
        psi = c.fold_angle()
        check(tag + "the collapsed mandrel is inside that hole",
              c.folded_radius(psi) <= exit_r,
              "%.2f mm folded at %.1f degrees, against a %.2f exit"
              % (c.folded_radius(psi), degrees(psi), exit_r))
        check(tag + "...and so is a BROKEN knee, which is a loose two-bar chain and "
              "can be anywhere within a link of its anchor",
              c.knee_radius() <= exit_r,
              "%.2f mm of reach against %.2f" % (c.knee_radius(), exit_r))
        check(tag + "AND IT ACTUALLY COMES OUT: swept the length of the truss, the "
              "collapsed mandrel clears every chord and every diagonal",
              c.withdraws() > 0.0, "%.2f mm at the tightest" % c.withdraws())

        # ---------------------------------------------- THE PARALLELOGRAM
        for sh, (kind, kk, phi) in enumerate(c.shafts):
            arms = c.arms_of(sh)
            L = [a.length for a in arms]
            ups = np.array([a.up for a in arms])
            check(tag + "shaft %d's arms are a PARALLELOGRAM -- equal, parallel -- so "
                  "the rails translate instead of rotating and the V-notches stay "
                  "square to the rods as they retreat" % sh,
                  max(L) - min(L) < 1e-9
                  and float(np.abs(ups - ups[0]).max()) < 1e-9,
                  "%d arms, %.3f mm each, %s rail" % (len(arms), L[0], kind))
        for k in range(t.n_chords):
            free = f.free_spans(k)
            check(tag + "chord %d's rail runs only where the ring does not: one piece "
                  "per free interval between joints, so the head passes through the "
                  "gaps" % k,
                  tuple(c.segments("chord", k)) == tuple(free)
                  and all(not any(j.x - f.joint_exclusion() < bb
                                  and aa < j.x + f.joint_exclusion()
                                  for j in g.joints_on(k)) for aa, bb in free),
                  "%d pieces, %.0f..%.0f mm long, %d joints cleared"
                  % (len(free), min(bb - aa for aa, bb in free),
                     max(bb - aa for aa, bb in free), len(g.joints_on(k))))
        nface = len([1 for kind, *_ in c.rails if kind == "face"])
        check(tag + "...and the face rails, sixty degrees from any chord where the "
              "head never goes, run whole",
              nface == t.n_chords, "%d face rails for %d faces" % (nface, t.n_chords))
        check(tag + "every pin is on a chord rail and every cradle on a face rail -- "
              "nothing is left behind on an arm of its own",
              all(any(aa <= p.x <= bb for aa, bb in c.segments("chord", p.chord))
                  for p in f.pins)
              and all(any(aa <= float(cr.apex[0]) <= bb
                          for aa, bb in c.segments("face", 0)) for cr in f.cradles),
              "%d pins, %d cradles" % (len(f.pins), len(f.cradles)))
        check(tag + "SIX locks, not one per rail piece: the arms at one azimuth are "
              "keyed to a shaft on the spine's surface, which is the one radius the "
              "head can never reach",
              len(c.braces) == len(c.shafts) == 2 * t.n_chords,
              "%d braces for %d rail pieces, shafts at r %.2f"
              % (len(c.braces), len(c.rails), c.shaft_r()))
        from truss.spec import Bracket
        ei_rail = Bracket.E * Cage.ARM_W * c.rail_h() ** 3 / 12.0
        ei_chord = Stock.E * np.pi * t.d_chord ** 4 / 64.0
        check(tag + "the rail is held straighter between its arms than the chord it "
              "carries is between its pins, which is what sets the arm pitch",
              c.arm_pitch() ** 4 / ei_rail <= Cage.PIN_PITCH ** 4 / ei_chord + 1e-6,
              "arms every %.0f mm on EI %.3g, pins every %.0f on EI %.3g"
              % (c.arm_pitch(), ei_rail, Cage.PIN_PITCH, ei_chord))

        # ---------------------------------------------- THE LOCK
        for b in c.braces:
            a = c.arms_of(b.shaft)[len(c.arms_of(b.shaft)) // 2]
            cc = float(np.linalg.norm(b.attach - a.base))
            def span_at(ps):
                at = a.base + a.up * (cc * np.cos(ps)) + np.array(
                    [-cc * np.sin(ps), 0.0, 0.0])
                return float(np.linalg.norm(at - b.anchor))
            check(tag + "shaft %d's load DRIVES THE LOCK: the way the rails want to "
                  "fold is the way that shortens the brace, so the working load puts "
                  "it in compression and pushes the knee harder onto its stop" % b.shaft,
                  span_at(radians(1.0)) < span_at(0.0),
                  "%.4f mm of brace at rest, %.4f a degree into the fold"
                  % (span_at(0.0), span_at(radians(1.0))))
            check(tag + "shaft %d's over-centre offset is inside the bracket it has to "
                  "be in -- above the linkage's own slop and the stop pin's bearing, "
                  "below what a hand can break" % b.shaft,
                  b.d_lo <= b.delta <= b.d_hi,
                  "%.3f mm, in [%.3f, %.1f]" % (b.delta, b.d_lo, b.d_hi))
            f_knee = c.working_load() * 2.0 * b.delta / b.p
            check(tag + "...and breaking shaft %d's lock is a push a hand can make"
                  % b.shaft,
                  f_knee <= Cage.HAND_F,
                  "%.1f N at the knee against %.0f a hand gives" % (f_knee, Cage.HAND_F))
            f_stop = c.working_load() * b.p / (2.0 * b.delta)
            sig = f_stop / (Cage.PIVOT_D * Cage.LINK_T)
            check(tag + "...and shaft %d's stop pin is inside its bearing allowable, "
                  "which is what the offset cannot go below" % b.shaft,
                  sig <= Cage.PIVOT_SIGMA,
                  "%.1f N/mm2 of %.0f" % (sig, Cage.PIVOT_SIGMA))

        # ---------------------------------------------- ONE PULL
        check(tag + "one draw rod trips all six shafts: the stroke is the worst "
              "knee's and every brace is the same mechanism",
              abs(c.draw_stroke() - max(b.stroke for b in c.braces)) < 1e-9
              and len(c.braces) == 2 * t.n_chords,
              "%.2f mm of stroke, %d braces" % (c.draw_stroke(), len(c.braces)))
        check(tag + "...and the rod fits the spine's bore with the spine still thicker "
              "than the cage's own floor",
              c.draw_bore() < 2.0 * f.spine_r()
              and f.spine_r() - c.draw_bore() / 2.0 >= Cage.LINK_T / 2.0,
              "%.2f mm of bore in a %.2f mm spine, %.2f of wall"
              % (c.draw_bore(), 2.0 * f.spine_r(),
                 f.spine_r() - c.draw_bore() / 2.0))
        check(tag + "...and the stroke is pulled from OUTSIDE the truss, at the end "
              "plate, which is the only place a hand can reach",
              c.draw_stroke() < f.end_free(),
              "%.2f mm of pull in %.1f mm of end freedom"
              % (c.draw_stroke(), f.end_free()))

        # ---------------------------------------------- THE FOLD HAS ROOM
        shift = abs(c.posed(psi)[1])
        check(tag + "the rails have the axial room to swing down into",
              shift <= f.end_free() + Cage.END_PLATE_T,
              "%.1f mm of swing against %.1f of end room"
              % (shift, f.end_free() + Cage.END_PLATE_T))

    # ---------------------------------------------- AND AT THE EXTREMES
    worst = None
    for side in (60.0, 85.0, 100.0, Load.SECTION_MAX):
        for dd in Stock.DIAMETERS:
            t = Truss(length=1000.0, side=side, alpha=40.0, d_chord=3.0, d_diag=dd)
            g = geometry.TrussGeometry(t)
            c = collapse.Collapse(fixture.Fixture(g))
            w = c.withdraws()
            if worst is None or w < worst[0]:
                worst = (w, side, dd)
    check("the mandrel comes out of every section and web the cell is specified to "
          "build, not just the chosen one",
          worst[0] > 0.0,
          "tightest %.2f mm, at a %.0f mm section on %.1f mm web"
          % (worst[0], worst[1], worst[2]))

    bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det and (VERBOSE or not ok) else ""))
    print("check_collapse: %d checks, %d failed" % (len(RESULTS), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
