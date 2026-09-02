# Working rules for this repository

This project is a **test bench for a reusable robotics codebase**, not a
solution to one competition. The ITU field is the fixture; the deliverable
is the algorithms. If the work would have to be redone by hand for a
different field, chassis or effector, it is not the deliverable.

---

## THE RULE

**Never hardcode a pose, a path, a waypoint, a lane, a station or a
threshold. Never tune a movement constant by trying values against the
board. Every geometric and scheduling decision is computed at runtime from
the task spec, the robot spec and the map.**

This is not a style preference. It is the thing that has repeatedly failed.

### The recognition test

Before writing any number into code, ask:

> If the field moved 50 mm, or the chassis got 20 mm wider, or the hopper
> moved to the other flank — would a human have to re-derive this number?

If yes, it is not a number. It is a function you have not written yet.

### The other test

> Did I arrive at this value by trying it and looking at the score?

If yes, delete it. A/B-ing constants against a 24-seed board is not
engineering; the board's standard error is ±7 to ±13 points per column, so
most of what that procedure "learns" is noise. This has been measured, in
this repository, repeatedly.

---

## Why this keeps happening (read this after a compaction)

The failure mode is specific and it recurs *every time context is lost*:

1. Context compacts. The reasoning behind the architecture is gone; the
   code is still there.
2. `route.py` (1677 lines, ~460 numeric literals) and `robot2.py` (2109
   lines, ~430) are read as *the way this project works*, because they are
   long, confident and full of measurement comments.
3. A problem is diagnosed correctly.
4. The fix is written as **another constant in the same style**, because
   that is what the surrounding code looks like.
5. The board moves a few points, inside its own noise. Nothing generalises.
   The next compaction repeats it.

Worked example, from this repository, so it is recognisable:

> Robot 1's kit approach was driving through four patients. The station was
> at x 240; the body spans ±117.5; the sticker column is at x 160. On paper
> the body must clear x 170 and the hopper's lip must land west of x 200,
> giving a window of [292.5, 310]. **300 was written into params.py.**
>
> The window was *wrong*. At the station (y 930) the chassis spans y
> 788..1073 and the stickers stop at 773 — the pose was never in conflict.
> The *traverse* 200 mm south of it was. The change helped only because
> moving the station drags the traverse with it. A solver found this in one
> second; the hand derivation had shipped.

**A constant can only be wrong silently.** That is the whole argument.

---

## What to do instead

| the question | the answer |
|---|---|
| where do I stand to put X in region Y? | `station.stand_for(region, effector, footprint, costmap)` — returns poses ranked by **margin in mm** |
| can I get there? | `station.reachable(stands, planner)` — a clear station is not a clear approach |
| how do I get from A to B? | `nav.plan` on a costmap. Never a scripted turn-drive-turn chain. |
| what is this task worth? | call `referee.score_*` on a projected board. Never a hand-assigned point value. |
| what should each robot do? | market/auction allocation over one shared task set (ST-SR-TA). Not one DP for robot 1 and a greedy for robot 2 that ignore each other. |
| how do the robots avoid each other? | prioritised planning for the plan, **ORCA/RVO** for the reflex. Not a bespoke escape law. |
| how long will this take? | plan it with the motion planner, then correct with a measured pace factor. Never a table of transcribed corridor times. |

Prefer the industry-standard algorithm with a name over a bespoke one.
If you are inventing a geometric law, stop: it already exists and is
better tested than yours.

---

## Constants that ARE legitimate

The rule is about *derived* quantities, not about facts.

**Legitimate** — these are measurements of the world, and they belong in
`params.py` or the task spec:

- physical dimensions of hardware: chassis 285 × 235, wheel radius, gate
  travel, pocket depth, hopper offset from the axle
- device limits: stall torque, max step rate, servo horn-to-horn time
- the rules of the game: zone rectangles, piece sizes, scoring, match length
- calibration values **a rig measures and can re-measure**: the tail
  ejection offset, the placement spread of a hopper, the pace factor

**Not legitimate** — these are answers to computations:

- station poses, staging points, waypoints, lane positions, hold points
- inset "aiming" rectangles derived from real zones
- speed limits chosen to make a specific manoeuvre work
- confidence/priority weights picked to order two options
- corridor travel-time tables
- any rectangle drawn by eye around a region of the field

If a constant in the second list exists, it is a TODO with a number in it.

---

## Current known offenders

These are the hand-set values still in the tree, to be replaced by
computation. Do not add to this list; work it down.

- `AgentA.KIT_STATION` (3 poses) → `station.stand_for`
- `AgentA.BEAM1_STATION` / `BEAM2_STATION` / `*_LOCAL` → same
- `route.KIT_LOOP_Y`, `KIT_BACKOFF`, `KIT_HEADING`, `KIT_ORDER`, `WP`,
  `SWEEP_LANE_MIN/MAX`, `SWEEP_REACH`, and the scripted
  `turn_to → drive_straight → dress_onto_line` approach chains
- `planner.TRAVEL` (~30 transcribed corridor times), `planner.DUR`,
  `planner.RANK` (a hand-fixed task order)
- `robot2.ZONES` (inset rectangles), `HOLD`, `EDGE_HARD`, `CARRY_PAD`,
  `BACK_OUT`, `CLEAR_NEAR/FAR/SLOW`, `_TURN_RUNGS`, `KIT_V`, `CARRY_V/W`
- `fleet.REGIONS` (8 rectangles drawn by eye), `DIS_*`, `FLEET_NEAR/CRAWL`
- `nav.BODY_PTS`, `nav.PUSH_HEADINGS` — one robot's chassis living inside
  the navigation library

`nav.CostMap` was fixed this way already: it read the field size from
module constants, which quietly made every map in the project this
contest's map. It now takes a size, defaulting to the old values.

---

## Measurement discipline

- **Boards do not steer.** A 24-seed board has a standard error of ±7
  (total) and ±13 (beams). Do not conclude anything from a change smaller
  than twice that. Use it to accept or reject, never to search.
- **Rigs steer.** Isolate one mechanism, vary one thing, report the failure
  mode. `check_station`, `rig_kits`, `rig_carryv`, `rig_column` are the
  pattern.
- **A rig that cannot fail is not a measurement.** `rig_cycle` once
  returned "the generator finished" instead of what it returned, reported
  10 of 12 deliveries, and the referee scored −6.
- **Physics is 84% of a match's wall time** (~400 µs/step, 311 collidable
  geoms, nv 233) and none of it is needed to evaluate allocation or
  scheduling. Develop those against a fast kinematic tier; use MuJoCo to
  validate mechanisms (capture, placement, discharge) only.
- Report scores as `X / 250` where X is the mean and 250 is the full-game
  maximum.

---

## Commits

Commit and push to `master` directly. Write what was measured and what it
means, not what was changed — the diff already says that.
