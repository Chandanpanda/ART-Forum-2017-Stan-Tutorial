"""IS THE ROBOT THE RIGHT SIZE?  Run this on every change.

Every other suite checks that the robot's parts agree with each other.  This
one checks the outline itself, because the outline was never derived: 235 is
an ASSERTION in params.py that the chassis width equals two carried beams,
and 285 is what a beam pushed by a tail stop reaches.  Neither number came
from anything the robot needs, and both are on the critical path of every
route it plans.

So: pack what has to be inside, hang what may stick out, and ask the board
the same three questions the mission asks -- fits, stands, routes.

    python3 sim/scripts/check_morphology.py [-v]
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from rfgyc26 import mjcf, morphology as mo, route, world
from rfgyc26.params import (AgentA as A, Chassis as C, Field, Piece as P,
                            Vision as V)

VERBOSE = "-v" in sys.argv
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


# WHAT HAS TO BE INSIDE ROBOT 1, from the parameters that describe the
# mechanisms rather than from the outline they were fitted into.  The length
# is one conveyor line -- pick up at the nose, carry aft, magazine, drop --
# so those three runs add; everything else sits above or below it and only
# spends width.
MODULES = (
    mo.Module("intake", run=(A.ROLL_AXIS_X + A.ROLL_TIP_R) - A.BELT_NOSE_X,
              half_w=A.ROLL_W / 2.0, datum="nose"),
    mo.Module("belt", run=P.DISC_D, half_w=C.BELT_W / 2.0),
    mo.Module("magazine", run=2 * A.ESC_XHALF, half_w=A.ESC_Y + A.TRIM_Y),
    mo.Module("drive", run=0.0, half_w=C.TRACK / 2.0 + C.WHEEL_W / 2.0,
              datum="axle", stacks=True),
    mo.Module("sweep", run=0.0, half_w=A.CAPTURE_OPEN / 2.0, datum="nose",
              stacks=True),
    mo.Module("cameras", run=0.0, half_w=V.BASELINE / 2.0 + 15.0,
              datum="tail", stacks=True),
    mo.Module("kits", run=0.0, half_w=P.KIT_Y / 2.0 * 3, stacks=True),
)
RUN, HALF = mo.packing(MODULES)
SHELL = 8.0
# The beams ride just outboard of the internal packing, NOT outboard of the
# shell: the pockets are open-sided channels whose outer face is the beam
# itself (F44), so what the beam has to clear is the structure, and the
# loaded envelope is the same whatever the shell does.
PY = HALF + 4.0 + P.BEAM_W / 2.0


def make(L, W):
    return mo.Design("A%03.0fx%03.0f" % (L, W), L, W, modules=MODULES,
                     shell=SHELL,
                     cargo=(mo.Cargo("beam1", P.BEAM1_L, P.BEAM_W, (0.0, PY)),
                            mo.Cargo("beam2", P.BEAM2_L, P.BEAM_W, (0.0, -PY))))


def main():
    # ------------------------------------------------------- the packing
    check("the packing minimum is derived, not typed",
          0.0 < RUN < A.L and 0.0 < 2 * HALF + SHELL < A.W,
          "%.0f x %.0f vs today's %.0f x %.0f"
          % (RUN, 2 * HALF + SHELL, A.L, A.W))
    check("...and the drivetrain is what binds the width, not the beams",
          max(m.half_w for m in MODULES) == C.TRACK / 2.0 + C.WHEEL_W / 2.0,
          max(MODULES, key=lambda m: m.half_w).name)
    today = make(A.L, A.W)
    check("today's chassis holds everything, with room to spare",
          today.packs and min(today.slack) > 0.0,
          "slack %.0f mm long, %.0f mm wide" % today.slack)
    check("a chassis below the packing minimum is refused",
          not make(RUN - 1.0, A.W).packs
          and not make(A.L, 2 * HALF + SHELL - 1.0).packs)
    check("a module that grows grows the robot",
          mo.packing(MODULES + (mo.Module("x", 10.0, 500.0),))
          == (RUN + 10.0, 500.0))

    # ------------------------------------------------------ the envelope
    # The spec's one hard number about the loaded robot: "the swept radius is
    # set by the beam ends at 185 regardless of chassis shape".
    check("a loaded robot 1 sweeps the 185 mm the specification claims",
          abs(today.swept(True) - 185.0) < 1.0, "%.1f mm" % today.swept(True))
    check("...and an empty one sweeps no more than a loaded one",
          today.swept(False) <= today.swept(True) + 1e-9,
          "%.1f empty, %.1f loaded" % (today.swept(False), today.swept(True)))
    # THE BEAM IS ALLOWED TO OVERHANG.  This is the whole point: shrinking
    # the chassis does not shrink the payload, so the loaded envelope stops
    # changing once the body is narrower than the beams -- and the empty one
    # keeps shrinking.
    small = make(RUN, 2 * HALF + SHELL)
    mid = make(RUN, (2 * HALF + SHELL + A.W) / 2.0)
    check("shrinking the chassis leaves the LOADED envelope alone",
          abs(small.swept(True) - mid.swept(True)) < 1e-6,
          "%.1f at %.0f wide, %.1f at %.0f wide"
          % (small.swept(True), small.width, mid.swept(True), mid.width))
    check("...but shrinks the EMPTY one by a third",
          small.swept(False) < 0.75 * today.swept(False),
          "%.0f vs %.0f mm" % (small.swept(False), today.swept(False)))

    # ------------------------------------------------------- the board
    lay = mjcf.m2_layout(np.random.default_rng(6))
    pieces = [(i, x, y, "patient") for i, (x, y, _c) in enumerate(lay)]
    soft = world.board_map(pieces=pieces, shove=True)
    hard = world.board_map(pieces=pieces)
    bare = world.board_map()
    regions = {k: (route.KIT_TAPE[k], route.kit_effector(k))
               for k in route.KIT_TAPE}
    start = A.START_POSE[:2]
    walls = world.doors(bare, (571.0, 150.0), (571.0, 1050.0), radius=60.0,
                        keep=3, bite=260.0)
    check("the cleared board has at least one passage north", len(walls) >= 1,
          str([(int(g[0]), int(g[1]), round(w)) for g, w in walls]))

    # A GATE'S WIDTH IS A CLEARANCE, WHICH IS A RADIUS.  Getting this wrong
    # by a factor of two rejected every design this file could generate,
    # including the one that is driving around the board today.
    if walls:
        w0 = walls[0][1]
        # The inscribed radius is the SMALLER half-extent, so a long thin
        # body gets through a gap its width would not suggest -- it goes in
        # end-on.  Both dimensions have to exceed the gate before the gate
        # refuses it, which is the honest statement and the one that catches
        # a factor-of-two slip in either direction.
        fit = mo.evaluate(make(2 * w0, 2 * w0), soft, regions, doors=walls,
                          start=start)
        no = mo.evaluate(make(2 * w0 + 40.0, 2 * w0 + 40.0), soft, regions,
                         doors=walls, start=start)
        check("a body whose inscribed radius equals the gate fits it",
              all(fit.fits.values()), str(fit.fits))
        check("...and one 20 mm bigger every way does not",
              not any(no.fits.values()), str(no.fits))

    tv = mo.evaluate(today, soft, regions, doors=walls, start=start,
                     strict=hard)
    check("today's robot can do today's job on a real board", tv.ok, repr(tv))
    check("...and needs to shove a patient to reach every kit zone",
          tv.shoves == len(regions), "%d of %d" % (tv.shoves, len(regions)))

    # NO FEASIBLE ROBOT 1 AVOIDS SHOVING, which is the finding that turns
    # world.board_map(shove=True) from a workaround into the design.  The
    # patients leave gaps a body would have to be narrower than the
    # drivetrain to use.
    gaps = world.doors(hard, (571.0, 150.0), (571.0, 1050.0), radius=40.0,
                       keep=3, bite=220.0)
    check("the patients leave gaps narrower than the packing floor",
          gaps and min(w for _g, w in gaps) < HALF,
          "%s mm of clearance vs %.0f needed"
          % ([round(w) for _g, w in gaps], HALF))

    # --------------------------------------------------------- the search
    best, bestd, grid = mo.shrink(
        make, soft, regions, doors=walls, start=start,
        lengths=np.arange(RUN, A.L + 1.0, 35.0),
        widths=np.arange(2 * HALF + SHELL, A.W + 1.0, 25.0), strict=hard)
    check("the search finds something smaller than today",
          best is not None and best.area < tv.area,
          "%s, %.0f%% of today's footprint"
          % (bestd.name if bestd else "-",
             100.0 * best.area / tv.area if best else 0))
    check("...and never returns a design the board refuses",
          all(v.ok for _d, v in grid if v is best) and (best is None or best.ok))
    check("...and every design it rejected, it rejected for a reason",
          all(v.ok or not v.packs or not all(v.fits.values())
              or any(s is None for s in v.stands.values())
              or any(r is None for r in v.reaches.values())
              for _d, v in grid))

    # THE PIVOT THAT F46 SAYS DOES NOT EXIST.  The corridor beside the
    # laboratory offers 170 mm of clearance; a robot cannot turn in place
    # anywhere its swept circle does not fit.  That is what forced the beams
    # to be carried 12 mm off the floor -- and it is a size problem, so it
    # has a size answer.
    d = bare.clearance()
    j0, j1 = bare.cell(0.0, Field.LAB_PLATE[1])[1], \
        bare.cell(0.0, Field.LAB_PLATE[3])[1]
    corr = float(d[:, j0:j1 + 1].max())
    check("today's empty robot cannot pivot beside the laboratory",
          today.swept(False) > corr,
          "swept %.0f mm, corridor %.0f mm" % (today.swept(False), corr))
    check("...and the smallest one can",
          best is not None and bestd.footprint(loaded=False).circumscribed
          <= corr,
          "swept %.0f mm" % (bestd.swept(False) if bestd else -1))

    n_bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det else ""))
    print("check_morphology: %d checks, %d failed" % (len(RESULTS), n_bad))
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
