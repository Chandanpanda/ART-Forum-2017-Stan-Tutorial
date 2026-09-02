"""Phase 0f: the planner, against brute force and the stopwatch.

Two things can be wrong with a planner: the optimiser (silently suboptimal
plans) and the model (optimal plans over fantasy numbers).  This suite
attacks both:

  * OPTIMALITY -- the DP's best value is compared against EXHAUSTIVE
    enumeration of every subset and order (about two thousand sequences:
    cheap, and it leaves the DP nowhere to hide).
  * INVARIANTS -- schedules fit the clock, the seal accepts no successor,
    no forbidden corridor transition appears, value degrades monotonically
    as the start slips, and the cheap station is sacrificed before the
    dear one.
  * CALIBRATION -- the model's block times are held against the measured
    phase stamps of four full runs (seeds 1/5/8/12).  When the behaviours
    get faster or slower, this fails, and the constants get re-measured --
    the planner is only as honest as its stopwatch.

    python3 sim/scripts/check_planner.py [-v]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from itertools import permutations, combinations
import numpy as np
from rfgyc26 import planner

VERBOSE = "-v" in sys.argv
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def brute_best(t0, at="SWEEP", done=()):
    """The certain answer: every subset, every order."""
    todo = [n for n in planner.NODES if n not in done]
    best_v, best_t = 0.0, t0
    for k in range(1, len(todo) + 1):
        for sub in combinations(todo, k):
            for order in permutations(sub):
                if "BEAMS" in order and order[-1] != "BEAMS":
                    continue
                if any(planner.RANK[order[i]] > planner.RANK[order[i+1]]
                       for i in range(len(order)-1)):
                    continue           # the proven topology is one-way
                t, prev, ok, seal_start = t0, at, True, None
                for name in order:
                    t += planner.TRAVEL.get((prev, name), 8.0)
                    # F127: the seal is two tasks wearing one name, and what
                    # has to fit before the buzzer is beam 2's commitment.
                    # Beam 1 is attempted only if BEAM1_TAIL is left, and the
                    # start-time pricing below charges for it when it is not.
                    floor = planner.BEAM2_TIME if name == "BEAMS" \
                        else planner.DUR[name]
                    if t + floor > planner.MATCH_END + 1e-6:
                        ok = False
                        break
                    if name == "BEAMS":
                        seal_start = t
                    t += planner.DUR[name]
                    prev = name
                if not ok:
                    continue
                v = planner._value(list(order) + list(done)) \
                    - planner._value(list(done))
                # the seal is priced by WHEN IT STARTS (F117): begun after
                # BEAM_FULL_BY it banks beam 2 alone and beam 1 is refused
                if seal_start is not None and \
                        seal_start > planner.BEAM_FULL_BY + 1e-9:
                    v -= planner.BEAM1_VALUE
                if v > best_v + 1e-9 or (abs(v - best_v) < 1e-9 and t < best_t):
                    best_v, best_t = v, t
    return best_v


def main():
    # ------------------------------------------------------------ optimality
    worst = None
    for t0 in np.arange(22.0, 92.0, 3.5):
        dp = planner.plan(float(t0)).value
        bf = brute_best(float(t0))
        if abs(dp - bf) > 1e-6 and worst is None:
            worst = (t0, dp, bf)
    check("DP matches exhaustive enumeration at every start time",
          worst is None, "t0=%.1f: DP %+.0f vs brute %+.0f" % worst
          if worst else "22..92 s, 20 starts")
    mid = planner.plan(60.0, at="L2", done=["L1", "L2"])
    check("...and from a mid-match state too",
          abs(mid.value - brute_best(60.0, at="L2", done=("L1", "L2"))) < 1e-6,
          repr(mid))

    # ------------------------------------------------------------ invariants
    bad = []
    values = []
    for t0 in np.arange(22.0, 110.0, 2.0):
        p = planner.plan(float(t0))
        names = [n for n, _, _ in p.tasks]
        values.append(p.value)
        if "BEAMS" in names and names[-1] != "BEAMS":
            bad.append((t0, "seal not last"))
        prev = "SWEEP"
        t = float(t0)
        for name, ts, du in p.tasks:
            tr = planner.TRAVEL.get((prev, name), 8.0)
            if tr > 1e8:
                bad.append((t0, "forbidden corridor %s->%s" % (prev, name)))
            t += tr + du
            prev = name
        if t > planner.MATCH_END + 1e-6:
            bad.append((t0, "overruns the clock: %.1f" % t))
    check("every schedule fits the clock, seals last, avoids forbidden "
          "corridors", not bad, str(bad[:2]) if bad else "45 starts")
    check("value never increases as the start slips later",
          all(values[i] >= values[i+1] - 1e-9 for i in range(len(values)-1)))
    # The prior prices a dock at its honest mean (11 s), so the opening
    # plan is CONSERVATIVE -- all three slots, the hospital and the seal --
    # and the full board is bought adaptively: one fast dock re-prices the
    # rest and the replan re-admits PCC_L.  Plan to the mean, upgrade on
    # good news; never promise on the fast tail.
    # With the seal budgeted at its p90 the OPENING allocation is lean --
    # slots, hospital, seal -- and the cheap stations are bought back
    # in-match when the docks come in fast.  Plan to the honest tail,
    # upgrade on good news.
    p_full = planner.plan(24.5)
    names = {n for n, _, _ in p_full.tasks}
    check("an on-time sweep keeps at least slots, hospital and seal "
          "(134 points)",
          {"L1", "L2", "KH", "BEAMS"} <= names and
          p_full.value >= 134.0 - 1e-6, repr(p_full))
    dropped = {"L1", "L2", "L3", "KH", "KL", "BEAMS"} - names
    p_full.observe("L", 8.7)                    # docks running FAST...
    p_full.complete("L1", 24.5 + 11.0)
    p_full.observe("L", 8.7)
    p_full.complete("L2", 24.5 + 20.5)
    back = dropped & {n for n, _, _ in p_full.tasks}
    # ...UNLESS BUYING IT BACK WOULD COST THE SEAL ITS SECOND BEAM (F117).
    # A returned station pushes everything after it later, and once the
    # seal starts past BEAM_FULL_BY it banks 25 instead of 70.  An 18-point
    # slot does not pay for a 45-point beam, so a planner that still bought
    # it back would be wrong; what has to hold is that the time IS found
    # and spent on the most valuable thing available.
    seal = {n: t for n, t, _ in p_full.tasks}.get("BEAMS")
    check("...and fast docks buy a dropped station back, unless it would "
          "cost beam 1",
          bool(back) or (seal is not None and
                         seal + planner.DUR["L3"] + 1.0 > planner.BEAM_FULL_BY),
          "dropped %s, returned %s, seal at T+%s"
          % (sorted(dropped), sorted(back),
             "%.0f" % seal if seal is not None else "-"))
    p_28 = planner.plan(28.5)
    names = {n for n, _, _ in p_28.tasks}
    check("under pressure the seal and the hospital always survive "
          "(70- and 28-point stations outlive the cheap ones)",
          "KH" in names and "BEAMS" in names and p_28.value >= 125.0 - 1e-6,
          repr(p_28))

    # ------------------------------------------------- schedule arithmetic
    p = planner.plan(26.0)
    name0 = p.tasks[0][0]
    tail = 0.0
    prev = None
    for name, _, du in p.tasks:
        if prev is not None:
            tail += planner.TRAVEL.get((prev, name), 8.0)
        tail += du
        prev = name
    check("latest_start leaves exactly the tail's room",
          abs(p.latest_start(name0) - (planner.MATCH_END - tail)) < 1e-6,
          "%.1f vs %.1f" % (p.latest_start(name0), planner.MATCH_END - tail))
    p.observe("L", 16.0)
    check("a slow dock re-prices the remaining docks, clamped",
          all(abs(p._dur[n] - 16.0) < 1e-6 for n in ("L1", "L2", "L3")),
          str({n: p._dur[n] for n in ("L1", "L2", "L3")}))
    p.observe("L", 3.0)
    check("...and cannot collapse below the clamp either",
          all(p._dur[n] >= 0.7*16.0 - 1e-6 for n in ("L1", "L2", "L3")))

    # A station dropped under pressure RETURNS when the match gets ahead.
    p = planner.plan(31.0)                      # pressure: something dropped
    dropped = {"L1", "L2", "L3", "KH", "KL", "BEAMS"} - \
              {n for n, _, _ in p.tasks}
    p.complete(p.tasks[0][0], 34.0)             # finished FAST (model said ~37+)
    back = dropped & {n for n, _, _ in p.tasks}
    check("a station dropped under pressure returns when time is found",
          not dropped or bool(back),
          "dropped %s, returned %s" % (sorted(dropped), sorted(back)))

    # ------------------------------------------------------- calibration
    # Measured phase stamps, truth-nav full runs (seeds 1/5/8/12): the block
    # from one station's start to the next station's start.
    MEASURED = {
        ("SWEEP", "L1"): [12.0, 12.7, 11.7],     # dock 1 total (s5's 17.8
                                                 # step-across outlier priced
                                                 # by observe, not the mean)
        ("L1", "L2"):    [10.8, 10.8, 10.2],     # dock 2 total
        ("KH", "KL"):    [9.4, 9.7, 9.9, 10.1],  # HOSP->PCC_L block
        ("L3", "KH"):    [15.0, 15.5, 16.4],     # F88 dispatch: seed-4
                                                 # mission block + L2/L1
                                                 # rig composites
        ("KL", "BEAMS"): [34.0, 35.0, 36.0],     # transit + both beams
    }
    off = []
    for (a, b), obs in MEASURED.items():
        model = planner.TRAVEL[(a, b)] + \
                (planner.DUR[b] if b != "BEAMS" else planner.DUR["BEAMS"])
        m = float(np.mean(obs))
        if not (0.7*m <= model <= 1.35*m):
            off.append("%s->%s model %.1f vs measured %.1f" % (a, b, model, m))
    check("the cost model stays within 35%% of the stopwatch, block by block",
          not off, "; ".join(off) or "%d blocks" % len(MEASURED))

    # ------------------------------------------------- the value model (F109)
    # The planner's value function used to be a hand-set constant per station
    # and it disagreed with the referee by 20 points a match: KL was priced
    # "2 kits x3 +10 = 16" on the assumption PCC_R stays empty, so the DP
    # dropped it for a lab dock worth 23 and the fleet forfeited the +20
    # distribution bonus.  These pin _value to the referee itself.
    from rfgyc26 import referee as _ref
    from rfgyc26.params import Field as _F, M2 as _M2

    def _kits(*zones):
        pts = []
        for z, n in zones:
            c = ((z[0]+z[2])/2.0, (z[1]+z[3])/2.0)
            pts += [c]*n
        return _ref.score_kits(pts)[0]

    for pcc, want_kl in ((0, 16.0), (2, 36.0)):
        base = planner._value(["KH"], pcc_r=pcc)
        got = planner._value(["KH", "KL"], pcc_r=pcc) - base
        check("PCC_L is worth %+.0f when robot 2 lands %d PCC_R kits"
              % (want_kl, pcc), abs(got - want_kl) < 1e-6, "got %+.1f" % got)

    check("a full robot-1 board is worth exactly samples+kits+beams",
          abs(planner._value(["L1", "L2", "L3", "KH", "KL", "BEAMS"],
                             pcc_r=2) - 170.0) < 1e-6,
          "%.1f" % planner._value(["L1", "L2", "L3", "KH", "KL", "BEAMS"],
                                  pcc_r=2))
    check("the kit arithmetic IS the referee's, not a copy",
          abs((planner._value(["KH", "KL"], pcc_r=2)
               - planner._value(["KH", "KL"], pcc_r=0))
              - (_kits((_F.HOSPITAL, 6), (_F.PCC_L, 2), (_F.PCC_R, 2))
                 - _kits((_F.HOSPITAL, 6), (_F.PCC_L, 2)))) < 1e-6)

    # and the beam constant is the referee's too: a perfect seal, scored
    beam1 = (_F.BEAM1_CENTRE[0], _F.BEAM1_CENTRE[1], 30.0,
             np.array([1.0, 0.0, 0.0, 0.0]))
    a = np.radians(90.0) / 2.0
    beam2 = (_F.BEAM2_CENTRE[0], _F.BEAM2_CENTRE[1], 30.0,
             np.array([np.cos(a), 0.0, 0.0, np.sin(a)]))
    seal = _ref.score_beams([beam1, beam2])[0]
    check("BEAM_SEAL_VALUE equals the referee's score for a perfect seal",
          abs(planner.BEAM_SEAL_VALUE - seal) < 1e-6,
          "planner %.0f vs referee %.0f" % (planner.BEAM_SEAL_VALUE, seal))

    # the seal is priced by when it starts (F117)
    check("BEAM_FULL_BY leaves beam 2 and then beam 1's tail",
          abs(planner.BEAM_FULL_BY
              - (120.0 - planner.BEAM1_TAIL - planner.BEAM2_TIME)) < 1e-6,
          "%.1f" % planner.BEAM_FULL_BY)
    early = planner.plan(60.0, at="L2", todo=["KH", "KL", "BEAMS"],
                         done=["L1", "L2"])
    late = planner.plan(78.0, at="KH", todo=["KL", "BEAMS"], done=["L1", "KH"])
    starts = {n: t for n, t, _ in late.tasks}
    # Test the price of the SEAL, not the total of the plan it rides in:
    # once a late seal is admitted (F127) the tour can afford PCC_L beside
    # it, and a threshold on late.value then fails for the right answer.
    late_nb = planner.plan(78.0, at="KH", todo=["KL"], done=["L1", "KH"])
    worth = late.value - late_nb.value
    check("a seal that cannot finish both beams is priced at 25, not 70",
          ("BEAMS" not in starts) or starts["BEAMS"] <= planner.BEAM_FULL_BY
          or abs(worth - (planner.BEAM_SEAL_VALUE - planner.BEAM1_VALUE))
          < 1e-6, "%r -- seal worth %+.0f" % (late, worth))
    # F127.  The DP used to test the seal's FULL duration against the buzzer,
    # so a late robot found it infeasible and dropped it: measured, five
    # seeds of twelve ended with both beams still aboard and half a minute
    # on the clock, scoring 0 where beam 2 alone was worth 25.
    done4 = ["L1", "L2", "KH", "KL"]
    partial = planner.plan(90.0, at="KL", todo=["BEAMS"], done=done4)
    check("a seal too late for beam 1 is still attempted for beam 2",
          "BEAMS" in [n for n, _, _ in partial.tasks], repr(partial))
    check("...and priced at 25 when it is",
          abs(partial.value - (planner.BEAM_SEAL_VALUE
                               - planner.BEAM1_VALUE)) < 1e-6, repr(partial))
    hopeless = planner.plan(100.0, at="KL", todo=["BEAMS"], done=done4)
    check("...and refused once even beam 2 cannot land",
          "BEAMS" not in [n for n, _, _ in hopeless.tasks], repr(hopeless))
    check("...and one that can is still worth taking early", "BEAMS" in
          [n for n, _, _ in early.tasks], repr(early))

    # the decision this all exists for
    planner.FLEET_PCC_R = 2
    keep = planner.plan(50.3, at="L2", todo=["L3", "KH", "KL", "BEAMS"],
                        done=["L1", "L2"])
    check("with PCC_R promised, the tour keeps PCC_L",
          "KL" in [n for n, _, _ in keep.tasks], repr(keep))

    # --------------------------------------------------------------- summary
    fails = [r for r in RESULTS if not r[1]]
    for name, ok, detail in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("PASS" if ok else "FAIL", name,
                                  ("  [%s]" % detail) if detail else ""))
    print("%d checks, %d failed" % (len(RESULTS), len(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
