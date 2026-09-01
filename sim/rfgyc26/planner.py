"""Route planning: the match as a prize-collecting tour over fixed stations.

Everything on this robot's board stands at a KNOWN position -- the only
randomness (sample positions inside the quarantine) is absorbed by the sweep
macro.  So the match after the sweep is a small, exactly-solvable problem:

    choose a subset of prize-bearing stations, and an order, maximising the
    referee's points subject to the 120 s clock, precedence, and the
    corridor travel times this field actually allows

-- a prize-collecting TSP with precedence.  Six free nodes: exact dynamic
programming over (subset, last) is microseconds, so the optimum is COMPUTED,
not tuned, and recomputed whenever reality disagrees with the model (a slow
dock, a jammed leg).  This file replaces every hand-set budget the route
carried (BEAM_BUDGET, KIT_BUDGET, the lab deadline arithmetic): those were
worst-case constants, wrong in both directions by construction -- the third
laboratory slot was skipped on all twelve seeds for want of seconds the kit
loop was not actually using.

Same code on both machines: numpy and params only.

THE COST MODEL IS MEASURED, NOT ASSUMED.  Durations and corridor times come
from phase-stamped runs of the very behaviours the schedule dispatches
(seeds 1/5/8/12; see check_planner, which holds the model against a
measured end-to-end timeline and fails if they drift apart).  The corridors
are the route's own: the east dogleg around the laboratory (the plate
cannot be crossed), the y-730 traverse south of the kit drops (F77), the
west lane the beams stage from.

VALUES ARE THE REFEREE'S -- literally: _value() scores a projected board
with referee.score_discs/score_kits rather than keeping a second copy of the
scoring rules.  That is not tidiness.  The kit column's +20 distribution
bonus and -10 empty-zone penalties couple robot 1's PCC_L drop to robot 2's
PCC_R drop, so a per-station constant CANNOT price KL correctly (F109), and
the constant this file used to carry was wrong by 20 points a match.
"""
import numpy as np

MATCH_END   = 118.0        # rules give 120; the last release needs to settle
# ---------------------------------------------------------------- durations
# Service time at each station, seconds, measured (mean of the phase logs).
DUR = {
    # A dock is BIMODAL: 8.5-10 when the arrival lands inside the tracked
    # reverse's authority, 14-17 when the step-across fires ("13 is what a
    # dock MEASURES", the old route knew).  11.0 is the honest mean;
    # observe() re-prices within the match as the modes reveal themselves.
    "L1": 11.0, "L2": 11.0, "L3": 11.0,
    "KH": 4.0, "KL": 4.0,              # turn north, drop, wait, back off
    # The MEAN of the whole seal.  A terminal task's overrun costs nothing
    # scheduled after it -- but see BEAM_FULL_BY: the seal's own second
    # beam IS downstream value, and it is priced by start time rather than
    # protected by padding this number (F117).
    "BEAMS": 31.0,
}
# Beam 1's own tail (staging dance + run-in + release), measured 17.2-19.8:
# once beam 2 stands, beam 1 is attempted only with this much clock left --
# a seal that dies mid-beam-1 drags beam 2 with it, and 25 banked points
# beat 0 heroic ones.
BEAM1_TAIL = 18.5
# The latest the seal may SET OFF and still have beam 1 attempted at all:
# beam 2 takes BEAM2_TIME to land, and beam 1 needs BEAM1_TAIL after that.
BEAM_FULL_BY = 120.0 - 18.5 - 18.0
# The sweep is the fixed opening act (randomised cargo -> sensor-terminated
# dwells), measured 24-28 s from the gun to the laboratory pivot line; the
# mission plans from wherever its own clock actually stands when it ends.
SWEEP_NOMINAL = 27.0

# ------------------------------------------------------------------ travel
# Corridor times between station exits and entries, seconds.  These encode
# the field's topology: lab pivot line along y~205, the east dogleg at
# x~950 (the only way past the laboratory), the y-730 traverse between kit
# longitudes, the descent to the west beam lane.  Pairs the optimiser
# should never want are present but honest (long).
def _lab_hop(i, j):
    return 1.0 + 0.9*abs(i - j)        # adjacent slots are 140 mm apart

TRAVEL = {}
for i, a in enumerate(("L1", "L2", "L3")):
    for j, b in enumerate(("L1", "L2", "L3")):
        if a != b:
            TRAVEL[(a, b)] = _lab_hop(i, j)
for a, i in (("L1", 0), ("L2", 1), ("L3", 2)):
    TRAVEL[("SWEEP", a)] = 2.2 + 0.9*i          # back out of the quarantine
    # Laboratory -> kit zones goes the LONG way by construction: dogleg
    # east of the plate, climb, swing west to the hospital lip -- ONE
    # pursuit since F88 (the 15.5 s the old chain measured was largely a
    # tail-corner grind on the plate edge: the dock-line departure pivot
    # was illegal and nothing logged it).  Re-measured on the fixed
    # dispatch: L3 block 15.0 (mission), L1/L2 12.1/11.2 s of pursuit plus
    # the 1.6 s back-away (rigs) -- travel is that minus the 4.0 service.
    TRAVEL[(a, "KH")] = 12.4 - 0.7*i
    # The direct-to-PCC_L plan (hospital already served or dropped) rides
    # the same climb plus the ~470 mm west swing: the KH measurement plus
    # 2.6 s.  Rarely planned; feasibility errs on the long side.
    TRAVEL[(a, "KL")] = 15.0 - 0.7*i
    TRAVEL[(a, "BEAMS")] = 6.5 + 0.9*i          # down to the west lane
    # Kit-zone -> laboratory is FORBIDDEN, not merely slow: the dock's
    # approach (pursue to the pivot line) would cross the laboratory plate
    # from the north.  A corridor-aware approach earns these back in step
    # 5; until then the tour must not contain them.
    TRAVEL[("KH", a)] = 1e9
    TRAVEL[("KL", a)] = 1e9
TRAVEL[("SWEEP", "KH")] = 14.0                  # kits before the lab: the
TRAVEL[("SWEEP", "KL")] = 16.0                  # dogleg from the west corner
                                                # (old chain's +1.5 over a
                                                # lab start, on F88's base)
TRAVEL[("SWEEP", "BEAMS")] = 5.0
TRAVEL[("KH", "KL")] = 5.5                      # the y-730 traverse (F77)
TRAVEL[("KL", "KH")] = 5.5
TRAVEL[("KH", "BEAMS")] = 9.0                   # down the west side
TRAVEL[("KL", "BEAMS")] = 5.0                   # why the loop ends at PCC_L

NODES = ("L1", "L2", "L3", "KH", "KL", "BEAMS")

# ===================================================== what a board is WORTH
# THE SCORE IS NOT SEPARABLE ACROSS ROBOTS, AND HAND-SET VALUES CANNOT SAY SO
# (F109).  This planner used to price each station with a constant:
#
#     VALUE = {"L1": 18, "L2": 18, "L3": 18, "KH": 28, "KL": 16, "BEAMS": 70}
#
# with a special case bolted on for the one joint bonus it knew about (all
# three samples, +5).  KL's 16 was derived on paper as "2 kits x3, +10 for
# the zone not being empty" -- correct ONLY on a board where PCC_R stays
# empty, which the comment beside it cheerfully admitted: "PCC_R belongs to
# robot 2 and is not this robot's problem."
#
# It is exactly this robot's problem.  The kit column pays +20 for the
# distribution 6/2/2 and -10 for each EMPTY zone, so robot 1's PCC_L drop
# and robot 2's PCC_R drop are worth far more together than apart:
#
#     robot 2 misses PCC_R:  KH alone -2   KH+KL 14   -> KL is worth 16
#     robot 2 lands  PCC_R:  KH alone 14   KH+KL 50   -> KL is worth 36
#
# Priced at 16, KL loses the tour to L3 (18) and is dropped at T+50 on most
# seeds; the two kits then ride in the hopper until they fall out beside the
# beams.  Measured over twelve seeds, that is the difference between a kit
# column of 2/50 and one of 50/50.  Robot 2 lands PCC_R on 10 of 12 seeds
# (21 of 24 kits, 7.5 s), so the good case is the normal one.
#
# The fix is not a better constant.  It is to stop keeping a second copy of
# the scoring rules: value(S) is the REFEREE's score for the board that task
# set leaves behind, and the marginal value of a task falls out of the DP
# for free -- joint bonuses, empty-zone penalties, sample closure and all.
from . import referee
from .params import Field, M2

# Robot 2's contract with the fleet.  Its opening act is two kits into
# PCC_R; route.py lowers this the moment robot 2 reports it failed, and the
# next replan re-prices KL accordingly.
FLEET_PCC_R = 2

# THE SEAL IS WORTH 70 IF IT STARTS IN TIME AND 25 IF IT DOES NOT (F117).
# seal_quarantine places beam 2 first and then refuses to begin beam 1 with
# less than BEAM1_TAIL on the clock -- correctly, since a seal that dies
# mid-beam-1 drags beam 2 off its line with it.  So a schedule that sets
# the seal off too late does not lose a little of its 70 points, it loses
# 45 of them, and a planner that prices the seal at a flat 70 will happily
# buy an 18-point laboratory slot with the second beam.  Measured, three
# seeds of twelve banked 25/70 that way.
#
# The fix is not to pad DUR["BEAMS"] -- that applies the worst case to
# every seed and costs a slot on all of them.  It is to price the seal by
# WHEN IT STARTS, which the DP already knows, and let it choose.
BEAM2_TIME    = 18.0       # transit plus beam 2 alone, measured
BEAM1_VALUE   = 45.0       # beam 1 (+25) and the closure it makes (+20)
BEAM_SEAL_VALUE = 70.0     # +25 each and +20 for closure; check_planner pins
                           # this to referee.score_beams on a perfect seal


def _centre(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _value(mask_names, pcc_r=None):
    """The referee's score for the board this task set would leave behind.

    Undelivered samples are stranded in the magazine (-3 each) and
    undelivered kits stay aboard, which is what the referee sees when the
    buzzer goes; patients are robot 2's column and cancel out of every
    comparison this DP makes, so they are left out.
    """
    names = set(mask_names)
    pcc_r = FLEET_PCC_R if pcc_r is None else pcc_r

    xyz = []
    for k, nm in enumerate(("L1", "L2", "L3")):
        if nm in names:
            xyz.append((Field.LAB_HOLE_X[k], referee.LAB_HOLE_Y, 2.5))
    xyz += [(2000.0, 2000.0, 10.0)] * (3 - len(xyz))       # still aboard

    kits = []
    if "KH" in names:
        kits += [_centre(Field.HOSPITAL)] * M2.KIT_PLAN["HOSP"]
    if "KL" in names:
        kits += [_centre(Field.PCC_L)] * M2.KIT_PLAN["PCC_L"]
    kits += [_centre(Field.PCC_R)] * int(pcc_r)

    return (referee.score_discs(xyz)[0] + referee.score_kits(kits)[0]
            + (BEAM_SEAL_VALUE if "BEAMS" in names else 0.0))

# THE PROVEN TOPOLOGY (F86).  The DP is free over SUBSETS but not over
# structure: slots west-to-east, then kits hospital-first, then the seal.
# Measured lesson, four 12-seed boards' worth: every behaviour was tuned in
# ONE entry basin (the seal from PCC_L's backoff above all -- entered from
# anywhere else it failed outright on seeds with 40 s on the clock), and a
# schedule that wanders outside those basins loses more to behaviour
# breakage than it wins on paper.  Optimal SELECTION inside the reliable
# structure; the exotic orders come back in step 5 when the rebuilt
# transits can back them.
RANK = {"L1": 0, "L2": 1, "L3": 2, "KH": 3, "KL": 4, "BEAMS": 5}


class Schedule:
    """The plan: ordered tasks with planned starts, plus the arithmetic the
    route used to do with constants.  tasks: [(name, t_start, dur)]."""

    def __init__(self, tasks, value, t0, dur, travel):
        self.tasks, self.value = list(tasks), value
        self._t0, self._dur, self._travel = t0, dict(dur), dict(travel)
        self.done = []

    def __repr__(self):
        s = " -> ".join("%s@%.0f" % (n, t) for n, t, _ in self.tasks)
        return "<plan %+.0f pts: %s>" % (self.value, s or "nothing")

    def next_task(self):
        return self.tasks[0][0] if self.tasks else None

    def latest_start(self, name):
        """Latest the task may START and still leave the rest of the plan
        its measured time -- the deadline the behaviours abort against.
        Replaces MATCH - BEAM_BUDGET - KIT_BUDGET and its cousins."""
        names = [n for n, _, _ in self.tasks]
        if name not in names:
            return MATCH_END
        i = names.index(name)
        t = MATCH_END
        for j in range(len(names) - 1, i - 1, -1):
            t -= self._dur[names[j]]
            if j > i:
                t -= self._travel.get((names[j-1], names[j]), 8.0)
        return t

    def complete(self, name, clock_now):
        """A task finished; drop it and REPLAN from here.  Milliseconds, so
        every boundary replans: the schedule adapts to how the match is
        actually going instead of how it was supposed to go."""
        self.done.append(name)
        # replan over EVERYTHING not yet done -- a station dropped under
        # pressure earlier comes back if the match got ahead of schedule
        new = plan(clock_now, at=name,
                   todo=[n for n in NODES if n not in self.done],
                   done=self.done, dur=self._dur)
        self.tasks, self.value = new.tasks, new.value
        return self

    def observe(self, kind, actual_s):
        """Measured reality updates the model for the REST of this match:
        one slow dock predicts the next (the est = last-dock trick the
        route used, made global).  Clamped -- one outlier is not a trend."""
        for n in list(self._dur):
            if n.startswith(kind):
                self._dur[n] = float(np.clip(actual_s, 0.7*self._dur[n],
                                             1.6*self._dur[n]))


def plan(t_now, at="SWEEP", todo=NODES, done=(), dur=None):
    """Exact DP (Held-Karp with precedence) over the remaining stations.

    State = (visited-subset, last station); transition cost = corridor
    travel + service duration; BEAMS accepts no successor (the robot is
    boxed into the south-west once the quarantine is sealed, F44).  Value
    is the referee's; ties break toward finishing earlier.  At six nodes
    this is exact and instant -- there is nothing to tune.
    """
    dur = dict(DUR if dur is None else dur)
    todo = [n for n in todo if n not in done]
    idx = {n: i for i, n in enumerate(todo)}
    n = len(todo)
    # best[(mask, last)] = earliest finish; parent for reconstruction
    best, parent = {}, {}
    r_at = RANK.get(at, -1)            # SWEEP has no rank: everything follows
    for name in todo:
        if RANK[name] < r_at:
            # The proven topology binds the FIRST hop too.  Without this, a
            # seal that aborted early "completed" BEAMS and the replan
            # cheerfully scheduled PCC_L from inside the south-west box the
            # sealed quarantine leaves the robot in (F44) -- over a default
            # 8 s corridor that does not exist.  Measured, seed 4.
            continue
        t = t_now + TRAVEL.get((at, name), 8.0) + dur[name]
        if t <= MATCH_END + 1e-6:      # epsilon: sums this long carry float dust
            m = 1 << idx[name]
            key = (m, idx[name])
            if t < best.get(key, 1e9):
                best[key] = t
                parent[key] = None
    for mask in sorted(range(1, 1 << n), key=lambda m: bin(m).count("1")):
        for last in range(n):
            key = (mask, last)
            if key not in best:
                continue
            if todo[last] == "BEAMS":
                continue               # nothing follows the seal
            t0 = best[key]
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                if RANK[todo[nxt]] < RANK[todo[last]]:
                    continue           # the proven topology is one-way
                t = t0 + TRAVEL.get((todo[last], todo[nxt]), 8.0) + dur[todo[nxt]]
                if t > MATCH_END + 1e-6:
                    continue
                k2 = (mask | (1 << nxt), nxt)
                if t < best.get(k2, 1e9):
                    best[k2] = t
                    parent[k2] = key
    # pick the subset with the most points, finishing earliest
    def names_of(mask):
        return [todo[i] for i in range(n) if mask & (1 << i)]
    top, top_key = -1e9, None
    for key, t in best.items():
        v = _value(names_of(key[0]) + list(done))
        # price the seal by its start time, not by whether it is in the set
        if todo[key[1]] == "BEAMS":
            if t - dur["BEAMS"] > BEAM_FULL_BY + 1e-9:
                v -= BEAM1_VALUE
        if v > top + 1e-9 or (abs(v - top) < 1e-9 and
                              (top_key is None or t < best[top_key])):
            top, top_key = v, key
    tasks = []
    key = top_key
    while key is not None:
        tasks.append(todo[key[1]])
        key = parent[key]
    tasks.reverse()
    # timestamps for the log and the deadlines
    out, t, prev = [], t_now, at
    for name in tasks:
        t += TRAVEL.get((prev, name), 8.0)
        out.append((name, t, dur[name]))
        t += dur[name]
        prev = name
    base = _value(list(done))
    return Schedule(out, top - base if top_key else 0.0, t_now, dur, TRAVEL)
