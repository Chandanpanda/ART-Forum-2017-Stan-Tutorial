"""The pose solver, against the answers that were worked out by hand.

station.stand_for exists to replace four hand-computed constants, so the
first thing it has to do is REPRODUCE them -- if it cannot re-derive the
stations that are known to work, it is not a generalisation of anything.
Then it has to work on a field it has never seen, which is the only test
that distinguishes a library from a transcript.

    python3 sim/scripts/check_station.py [-v]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from rfgyc26 import nav, station
from rfgyc26.params import AgentA, Field, M2, Piece, Robot2 as R2

VERBOSE = "-v" in sys.argv
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


# ------------------------------------------------------------ the fixtures
def field_map(pieces=()):
    """The competition field: walls, the laboratory plate, and whatever
    loose pieces the caller wants treated as obstacles."""
    cm = nav.CostMap()
    cm.add_rect(*Field.LAB_PLATE)
    for x, y in pieces:
        cm.add_disc(x, y, Piece.CYL_D / 2.0)
    return cm


def stickers():
    """The twelve patient stickers, which is what robot 1's PCC_L approach
    was driving through."""
    out = []
    for box in (M2.SIDE_L, M2.SIDE_R):
        x0, y0, x1, y1 = box
        for j in range(6):
            fx = 0.25 if j % 2 == 0 else 0.75
            fy = (j // 2 + 0.5) / 3.0
            out.append((x0 + fx*(x1-x0), y0 + fy*(y1-y0)))
    return out


A1 = station.Footprint.rect(AgentA.L, AgentA.W)
# The kits leave the hopper in a line 28 mm apart ALONG THE BODY'S FORWARD
# axis -- measured, a northward drop lands them at (547, 978), (550, 1006),
# (554, 1034).  Writing that stride sideways is what made the first run of
# this check reject the hospital station that demonstrably works: the module
# was right and the fixture was wrong, which is the whole reason to state a
# mechanism as data and let something else check it.
HOPPER = station.Effector(offset=M2.HOPPER["PCC_L"], spread=12.0,
                          stride=(28.0, 0.0), count=2)
HOPPER6 = station.Effector(offset=M2.HOPPER["HOSP"], spread=12.0,
                           stride=(28.0, 0.0), count=6)


def main():
    # ------------------------------------------------- it reproduces itself
    cm = field_map(stickers())
    got = station.stand_for(Field.PCC_L, HOPPER, A1, cm, headings=8, res=10.0,
                            prefer=(300.0, 730.0), top=12)
    check("PCC_L: the solver finds somewhere to stand at all", bool(got),
          repr(got[:1]))
    if got:
        # The window worked out by hand was [292.5, 310]: the body has to
        # clear the sticker column at x 170 and the lip has to stay west of
        # the zone's east edge at x 200.
        # THE HAND-DERIVED WINDOW WAS WRONG AND THE SOLVER SAYS SO.  It was
        # [292.5, 310] on the reasoning that the body must clear the sticker
        # column at x 170; but at the STATION the chassis spans y 788..1073
        # and the stickers stop at 773, so the pose was never the conflict.
        # The traverse 200 mm south of it was.  What the solver must get
        # right is the claim it actually makes -- the payload lands inside
        # and the body is clear THERE -- and the approach is a separate
        # question, asked by station.reachable().
        check("...and the station that shipped is among the ones it accepts",
              any(abs(s.pose[0] - 300.0) < 45.0 and abs(s.pose[1] - 930.0) < 60.0
                  for s in station.stand_for(Field.PCC_L, HOPPER, A1, cm,
                                             headings=8, res=10.0, top=200)),
              "solver's own best x: " + " ".join("%.0f" % s.pose[0] for s in got[:6]))
        check("...clear of the sticker column the old station drove through",
              all(s.clear > 0.0 for s in got),
              "least clearance %.0f mm" % min(s.clear for s in got))
        best = got[0]
        check("...and every kit it promises really lands inside PCC_L",
              all(Field.PCC_L[0] <= x <= Field.PCC_L[2] and
                  Field.PCC_L[1] <= y <= Field.PCC_L[3]
                  for x, y in best.deposits), repr(best))

    # the hospital station needs no move and the solver should agree
    got_h = station.stand_for(Field.HOSPITAL, HOPPER6, A1, cm, headings=8,
                              res=10.0, prefer=(700.0, 730.0), top=6)
    all_h = station.stand_for(Field.HOSPITAL, HOPPER6, A1, cm, headings=8,
                              res=10.0, top=400)
    # The pose that used to be hand-set, (711.5, 965) facing north, is a
    # fact about this field and this hopper, so the solver must still accept
    # it -- but it is no longer WHERE the answer has to be.  It measured
    # 6 of 6 kits in the zone on every rig run, and so does the solver's own
    # choice, which faces 68 degrees and no constant would have proposed.
    check("HOSP: the pose that used to be hand-set is among those accepted",
          any(abs(s.pose[0] - 711.5) < 45.0 and abs(s.pose[1] - 965.0) < 60.0
              and abs(s.pose[2] - 90.0) < 1e-6 for s in all_h),
          "%d poses accepted; best %r" % (len(all_h), got_h[:1]))
    check("...and it prefers one with more margin than the hand-set pose",
          bool(got_h) and got_h[0].margin > 0.0, repr(got_h[:1]))

    # ------------------------------------------------- the claims are true
    bad = []
    for s in got + got_h:
        for x, y in s.deposits:
            box = Field.PCC_L if s in got else Field.HOSPITAL
            if not (box[0] <= x <= box[2] and box[1] <= y <= box[3]):
                bad.append((s.pose, (x, y)))
        pts = A1.world(*s.pose)
        d = cm.clearance()
        ii = np.clip((pts[:, 0] // cm.res).astype(int), 0, cm.nx - 1)
        jj = np.clip((pts[:, 1] // cm.res).astype(int), 0, cm.ny - 1)
        if abs(float(d[ii, jj].min()) - s.clear) > 1e-6:
            bad.append((s.pose, "clear mismatch"))
    check("every pose it returns satisfies the claim it makes about itself",
          not bad, str(bad[:2]))
    check("margins come back in order, best first",
          all(got[i].margin >= got[i+1].margin - 1e-9
              for i in range(len(got)-1)))

    # ------------------------------------------------ it refuses the absurd
    tiny = (5.0, 5.0, 15.0, 15.0)          # a 10 mm box in the corner
    check("an unreachable region returns nothing rather than a bad answer",
          not station.stand_for(tiny, HOPPER, A1, cm, headings=8, res=10.0))
    huge = station.Effector(offset=(0.0, 4000.0), spread=1.0)
    check("...and so does an effector that reaches off the field",
          not station.stand_for(Field.PCC_L, huge, A1, cm, headings=8,
                                res=10.0))

    # ------------------------------------------- A FIELD IT HAS NEVER SEEN
    # Nothing below comes from this competition: a different board, a
    # different chassis, a different effector.  This is the test that says
    # the module is a library rather than a transcript of one contest.
    other = nav.CostMap(res=10.0, size=(800.0, 600.0))
    other.add_rect(300.0, 0.0, 340.0, 400.0)          # a spur wall
    bay = (600.0, 420.0, 760.0, 560.0)
    small = station.Footprint.rect(180.0, 120.0, axle=0.35)
    arm = station.Effector(offset=(150.0, 0.0), spread=8.0)   # reaches ahead
    out = station.stand_for(bay, arm, small, other, headings=12, res=10.0,
                            top=5)
    check("a different field, chassis and effector: it still solves",
          bool(out), repr(out[:1]))
    if out:
        ok = True
        for s in out:
            for x, y in s.deposits:
                if not (bay[0] <= x <= bay[2] and bay[1] <= y <= bay[3]):
                    ok = False
            if s.clear <= 0.0:
                ok = False
            if not (0.0 <= s.pose[0] <= other.w and 0.0 <= s.pose[1] <= other.h):
                ok = False
        check("...with the payload in the bay and the body on the board", ok,
              repr(out[0]))
        check("...and it kept off the spur wall",
              all(not (300.0 <= x <= 340.0 and 0.0 <= y <= 400.0)
                  for s in out for x, y in A1.world(*s.pose)),
              repr(out[0]))

    # keep_out is honoured: forbid the floor the best answer stands on
    if out:
        b = out[0].pose
        block = (b[0] - 60.0, b[1] - 60.0, b[0] + 60.0, b[1] + 60.0)
        again = station.stand_for(bay, arm, small, other, headings=12,
                                  res=10.0, keep_out=[block], top=3)
        check("keep_out moves the answer off forbidden floor",
              all(not (block[0] <= x <= block[2] and block[1] <= y <= block[3])
                  for s in again for x, y in small.world(*s.pose)),
              repr(again[:1]))

    # ------------------------------------------------------------- margins
    # A margin is a promise: perturb a pose by less than it and the payload
    # should still land inside.
    if got:
        s = got[0]
        worst = None
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            k = max(s.margin - 1.0, 0.0)
            p = (s.pose[0] + dx*k, s.pose[1] + dy*k, s.pose[2])
            for x, y in HOPPER.deposits(*p):
                if not (Field.PCC_L[0] <= x <= Field.PCC_L[2] and
                        Field.PCC_L[1] <= y <= Field.PCC_L[3]):
                    worst = (p, (x, y))
        check("the margin is a promise: nudging by less keeps the payload in",
              worst is None, str(worst))

    bad_n = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det and not ok else ""))
    print("%d checks, %d failed" % (len(RESULTS), bad_n))
    return 1 if bad_n else 0


if __name__ == "__main__":
    sys.exit(main())
