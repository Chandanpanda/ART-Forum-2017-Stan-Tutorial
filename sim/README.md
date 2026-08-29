# RFGYC'26 Agent A — MuJoCo simulation

A working MuJoCo model of **Agent A** (beams + samples) from the Rev C two-robot
system, plus the demonstration you asked for: the robot sweeps the quarantine,
picks a Ø56 sample disc off the floor onto its inclined conveyor, and posts it
through the base gate into a Ø60 laboratory hole.

Everything here was built and run before it was handed over. **Read
"Status" below before you trust any part of it** — some of it works end to end,
some of it does not yet, and the difference matters.

---

## 1. Install and run (Windows, Python 3.13)

You have already done step 1.

```powershell
pip install mujoco numpy          # numpy is required; mujoco you already have
cd path\to\ART-Forum-2017-Stan-Tutorial\sim

python scripts\check_geometry.py     # 1. no physics: derived geometry + assertions
python scripts\demo_belt.py          # 2. the conveyor, isolated (fast, ~10 s)
python scripts\demo_capture.py       # 3. the "pick" half: 3 discs -> magazine
python scripts\demo_post.py          # 4. the "place" half: gate -> 3 lab holes
python scripts\demo_pick_place.py    # 5. the full Agent A mission (~60 s wall clock)

python scripts\demo_pick_place.py --gui        # interactive viewer
python scripts\demo_pick_place.py --seed 7     # different random disc placement
```

`models\scene_pick_place.xml` is regenerated on every run and can be opened
directly in MuJoCo's `simulate.exe` if you have it.

Optional, for video: `pip install "imageio[ffmpeg]"` then add `--video`.

---

## 2. Status — what actually works

| | |
|---|---|
| Model generation from one parameter file | **works** |
| 11 geometry assertions | **all pass** |
| Drive: straight line / turn in place | **works** — 0.9 % slip; 70 % turn efficiency |
| Conveyor carry on the incline | **works** — 59.2 mm/s against a 60 mm/s belt |
| Passive accumulation at a closed gate | **works** — 0.134 N vs 0.118 N predicted |
| **Pick: 3 samples off the floor into the magazine** | **works — 24 of 24** over 8 randomised matches, all three *seated* every time |
| **Magazine escapement: one piece per stroke** | **works — 3 of 3**; the bore rangefinder reads 3 → 2 → 1 |
| **Place: all three lab holes** | **works — 3 of 3, +45**, dock error **0.0 mm** |
| Closed-loop chute docking, in-mission | **works** — **1.5–2.1 mm at all three holes** |
| Referee scoring (Senior sample rules) | **works** |
| **Full mission end to end** | **+50 — all three samples placed — in 7 of 8 randomised matches** |

```
seed      1    2    3    4    5    6    7    8
score   +50  +50  +50  +50  +50  +27  +50  +50      (+50 = all three placed)
time    114  110  109  128  114  168  110  109  s     (match budget is generous)
```

Seed 6 posts two of three: the piece released at hole 1 misses the slot and is
then swept off the plate as the robot departs. Docking was 1.5 mm there, so this
is release scatter, not aim — and F21 explains why 2 mm is the entire budget.

## 3. Findings — things the simulator discovered about the design

These came out of getting the model to run, and they are the real value here.

**F1. The passive scoop cannot convey the disc. The sweeper fingers' stroke is
load-bearing, not optional.**
After ~19 mm of engagement (5 mm disc ÷ tan 15°) the disc has left the floor
entirely, yet its rear edge is still ~46 mm short of the belt nose — and nothing
then moves it, so it simply rides along with the robot. Measured over a full
parameter sweep of ramp friction (0.08–0.40) and approach speed (100–250 mm/s):
the disc never advanced past the ramp tip. For a piece to stay floor-supported
until it reaches a belt nose 17.5 mm up, the ramp would have to be ≤ 4.4°, which
a Ø16 roller forbids. **Spec §5's "the fingers do not lift — the robot's forward
motion does" is not sufficient on its own.** The ~110° finger sweep is what
closes this gap, and the Mechanism Explainer's 15.6° of arm travel is nowhere
near enough. This is worth a bench test before committing to the intake.

**F2. Agent A cannot turn in place in the deployment box.**
Swept radius is 185 mm (the beam pockets run the full length, so the corners
cannot be chamfered below Za 60 — spec §4.2). The start pose is 140 mm from the
south wall. The robot jams. It must leave the box nose-first and only pivot in
open field, ≥ 185 mm from any wall. The route now does this.

**F3. A sweep "mouth centred Y ≈ 70" is geometrically impossible.**
The chassis is 235 wide, so at heading 180 its south edge would be 47.5 mm
*inside* the south wall. Passes must run at Y ≥ 130. With the 165 capture band,
two passes at Y 130 and Y 215 cover Y 47.5–297.5, which is enough — but spec §9's
figures need correcting.

**F4. The wheel contact patch dominates turn-in-place scrub, and the spec's
scrub calculation omits it.** Spec §4.3 computes 0.03 N·m from the ball
transfers against 3.9 N·m available — a >100× margin. But a 22 mm-wide wheel's
own contact patch resists yaw far more than that: measured turn efficiency was
21 % at 22 mm, 44 % at 12 mm, 60 % at 8 mm, 70 % at 6 mm.

**F5. The lab hole position in the Explainer is geometrically impossible.**
A Ø60 hole centred at Y 372 on a plate spanning Y 360–510 overhangs the plate's
south edge by 18 mm. The model uses **Y 400** — the nearest value that keeps the
bore fully inside the 150-deep plate. Recorded as **R11**, `[VERIFY]`.

**F6. Disc-in-hole clearance is 2 mm, and the chamfer really does do all the
work.** Posting succeeds when the chute is centred and fails when the dock drifts
by ~8–15 mm. Spec §6.4's claim that the 45° chamfer "absorbs ±10 of robot
position error" is the single assumption the whole sample mission rests on, and
it needs the mock-up test in §10.10.

**F7. The belt tail (Xa 35) and the chute axis (Xa 33) are 2 mm apart, but the
bore has 1 mm of clearance.** A Ø56 disc tipping off the tail therefore cannot
land centred in a Ø58 bore — it overshoots by ~5 mm and jams on the rim. Both are
now on the same station (Xa 36).

**F8. The converging guides start 75 mm too late.** Spec §7 begins them at Xa 200,
but pieces board at the scoop tip (Xa 275). Two Ø56 discs fit abreast on a 116 belt,
so in that un-guided run they bridge and the queue stalls — the spec's own "#1 jam
risk", reproduced for Agent A. Guides now converge hard at the tip; the discs then
single-file dead centre (measured lateral spread ±1.6 mm).

**F9. A Ø58 bore is not achievable from a belt-tip drop.** Modelled at Ø66 (5 mm
clearance). With 1 mm the disc jams every time. The real machine needs a wider
bore or positive placement.

**F10. There is no legal pivot between the south wall and the lab plate.**
Agent A needs 185 mm of swept radius; the corridor at 351 < x < 791 is 360 mm wide
and needs 370. Turning there beaches the chassis on the plate with its drive
wheels off the floor. The route therefore pivots once, west or east of the plate,
and thereafter only reverses — which is the tail-in posture anyway.

**F11. Agent A cannot REVERSE up a square 3 mm plate edge.** The Ø20 ball
transfers stall on the step and the dock halts 80–320 mm short. This answers the
spec's `[VERIFY §10.2]` question in the negative *for the reverse direction* —
forwards over it is fine. The plate needs a ramped or taped edge; modelled here as
a 1 mm decal.

**F12. The chamfer is rebuilt, and it caps out at ~7 mm — not the spec's ±10.**
Segments are now oriented with `xyaxes` (tangential x, up-slope y) instead of three
composed eulers, which is what put the old ones 30 mm out of place. Measured
envelope: 5.4 mm tall, 34.8 mm radius. It funnels a disc dropped up to **7 mm**
off-centre straight into the slot; at 9 mm the disc rests on the cone.

**That paragraph was wrong and is corrected by F21 below** — I had said the fix
was "either a recessed counterbore in the plate, or more clearance at the robot's
tail". The laboratory is a *supplied field element*, so the first is not
available; and re-measured properly, the chamfer is not doing the work anyway.

**F13. What was actually breaking hole 3: the boundary tape.** Not the chamfer.
The 20 mm deployment-box tape was modelled as a collision geom, and its 0.4 mm
step caught the Ø20 ball transfers at **8–10 N**, stopping the robot dead on the
box boundary — which the eastern approach to hole 3 crosses. Real adhesive tape is
an optical marker, ~0.1 mm; it is now visual-only, and the TCRT line array reads
it by position. The same class of bug applied to the lab plate: the robot could
not reverse up even a 1 mm step, so the plate now sits on its own collision bit
(4) and interacts with game pieces but not the robot. With both fixed, all three
holes dock to 1.8–2.0 mm.

**F14. A gravity magazine does not seat its last piece.** Discs 1 and 2 stack
properly because each is pushed down by the next. The third has nothing above it
and lands on the stack rather than settling into it — measured at Za 34.6 with 22°
of tilt where a seated disc sits at 24.2 and 0.4°. It is then only loosely
retained and shakes free in transit. The fix is a **positive feed**: a plunger on
the bore axis, parked with its face at Za 84 and stroked down to Za 26. Two
strokes seat the column every time.

The first attempt at this was *worse than nothing* and the reason is worth
keeping: it parked a Ø30 foot 12 mm above the collar, and a piece tipping off the
belt tail reared up into it and wedged. **0 of 3 with the ram, 3 of 3 without it,
same model.** Anything parked close over the mouth does that. Parking at Za 84 is
33 mm clear of the highest a piece ever reaches (Za 51, its top at the discharge
plane), so the drop path is empty. 58 mm of stroke is more than a servo horn
gives directly — use a 29 mm crank or a rack — and keep the force low: at 14 N an
earlier build pushed a disc *through* the bore wall.

**F15. The bore needs a raised rear collar, and that collar is what centres the
piece.** Not the chamfer, not the bore. A Ø56 disc sliding aft at the Za 51.5
discharge plane is halted the instant its rim touches the collar's inner face at
r 33, which leaves its centre at dx = −(33 − 28) = **−5 mm** — inside the bore,
every time, with no sensing and no tuning. Without it the disc sails over the
Za 30 rim, jams against the chassis rear wall 36 mm aft of the axis, and simply
stays there, flat, held by belt friction. The collar has to be taller above the
discharge plane than the hold-down gap (F16), or a piece climbs over it instead
of being stopped.

**F16. Bore geometry alone cannot stop a coin jam. The piece has to arrive
flat.** A disc tilts freely inside *any* bore wider than itself — there is no
bore, chamfer or funnel that prevents it. Left to itself the piece enters
edge-first and stands up inside the tube at 63°, and everything behind it piles
up. The fix is upstream and costs almost nothing: a **hold-down strip 8 mm above
the belt** over the last ~120 mm of its run. A 5 mm disc can then rise only 3 mm,
which caps the droop at about 6°, so it stays flat until its trailing edge clears
the tail and then falls flat. It also makes shingling impossible on the guided
run — two discs need 10 mm of channel and there are 8. One folded strip of the
same 1 mm sheet as the chassis: the cheapest part on the robot and the one that
makes the magazine work.

It must end **at** the tail. Carried past it, the same strip blocks the rotation
the piece needs in order to drop, and nothing seats at all.

**F17. The tail roller belongs one disc radius forward of the chute axis — not
over it.** This supersedes F7. A piece is supported until its *trailing* edge
clears the tail, so it is released when its centre is one radius aft of the tail.
Put the tail at `CHUTE_X + 28` and the release point is the bore axis. Measured,
one disc, otherwise identical model:

| belt tail Xa | 36 | 50 | 56 | 60 | 64 |
|---|---|---|---|---|---|
| seats? | never | never | yes | yes | **yes, every time** |

With the two aligned (the old F7) the piece is already half over the bore when it
starts to overhang: it tips in edge-first and jams, 0 of 3. Moving the tail
forward also frees the bore axis, which is what lets the feed plunger (F14) sit
where it should.

**F18. The guides have to climb with the belt.** They were built at a fixed
height, and the belt rises 46 mm over its run — so near the intake the walls
floated above the pieces entirely and a disc passed underneath them, out to the
belt edge, where it jammed. One sample per match was lost this way. The walls are
now generated between two 3-D points and stand perpendicular to the belt face.

**F19. A single sliding gate cannot meter.** It has to clear a Ø56 disc to
release one, and by then it has released the column. This was masked for a long
time because the stack used to hang up on itself; once F14–F17 made it seat
properly, one stroke dropped all three and two of them landed in the same lab
hole. The magazine needs a real **escapement**: a shelf that carries the column,
and a thin retainer that slides in at the joint above the bottom piece.

Three things had to be right, and each was found by breaking it:

* **Two actuators, not one.** Built as a single stepped slide — the classic coin
  escapement — the shelf arrives from one side exactly as the retainer leaves
  from the other, so the column is handed over in mid-drop and tips out of the
  bore. Sequenced retainer-in → shelf-out → shelf-in → retainer-out, every
  transfer lands on something already in place.
* **The retainer is a 1 mm knife with a rolled leading edge.** At 3 mm it
  straddled the joint. Square-ended, it met the second disc's rim head-on and
  drove it sideways out of the bore. A round edge that starts *below* the joint
  cams the column up instead — and it is self-correcting, because the bottom disc
  cannot go down while the shelf is under it.
* **The robot has to know how many are left.** With one piece there is no joint
  to enter and the retainer must stay parked. A rangefinder looking down the bore
  from Za 70 gives 5 mm of range per piece. It has to be **on the axis**: at
  r 25 it missed a disc sitting 3.4 mm off centre and reported an empty magazine,
  which cost that piece.


**F20. The sweeper fingers pivot at the nose and reach aft, so "open" makes the
intake channel diverge.** Tips out at ±82.5 with pivots at ±74, the channel
*widens* from 148 mm at the mouth to 165 at the belt — it funnels nothing. That
looked like the reason a sample more than ~45 mm off the sweep line got bulldozed
into the west wall instead of collected, but raking the fingers in only moved the
step one bay forward (F22 is the real cause). Captures over 8 randomised matches,
24 samples in all, with the guides starting at the mouth width: **tips ±82.5
(open) 24/24, tips ±58 (raked) 21/24, tips ±40 22/24.** Open wins.

**F21. The laboratory is a supplied field element, and the 45° lead-in the robot
spec relies on is an assumption about someone else's part.** The rulebook (§3.2)
describes it as wood with "3 marked slots of 60 mm" and says nothing about a
chamfer. So a counterbore is *not an option the team has* — and measuring it
properly, it would not help much anyway. Dropping a disc from gate height at
increasing offsets from the slot centre:

| offset from slot centre (mm) | 0 | 1 | 2 | 3 | 4+ |
|---|---|---|---|---|---|
| 4 mm assumed chamfer | in | in | — | — | — |
| 2 mm chamfer | in | in | — | — | — |
| **no chamfer (as supplied)** | in | in | **in** | — | — |

The capture radius is **2–3 mm either way** — that is just the (60 − 56)/2 = 2 mm
radial clearance. A 4 mm-wide, 4 mm-tall lead-in is far too small to matter, and
it slightly *hurts* because a disc can rest on the cone instead of tipping in.
The earlier "±7 mm" figure was measured on a 3 mm plate before F11 thinned it.

So the posting budget is **2 mm, full stop**, and it has to be met by the robot:
by docking accuracy and by a repeatable release, not by geometry at the hole.
`Field.LAB_CHAMFER = 0.0` runs the sim with a plain bored slot — worth using as
the default assumption until someone measures a real laboratory.

**F22. The guides have to start as wide as the sweeper mouth.** They began at the
*belt* width (116) while the mouth is 148 wide at the finger pivots, so a piece
more than 30 mm off the sweep line met the guide's **leading edge** square-on
instead of its inner face, and was pushed along the field instead of funnelled
in. Starting them at 148 and tapering to 62 makes the hand-off continuous:
capture went from 22/24 to **24/24**, with all three seated in every seed.

One trap with this: the belt face is at Za 0.4 that far forward, so a wall foot
1 mm below it lands *under the floor plane*. These are robot-class geoms, so they
plough — every mission ran to the 240 s timeout until the foot was clamped.

---

## 4. Modelling decisions you should know about

Each of these is a deliberate, documented departure from the drawings.

| Decision | Why |
|---|---|
| **One continuous conveyor** from the scoop tip to the tail roller, instead of a 0.5 mm shim feeding a belt whose nose is at Z 17.5 | F1: a passive shim provably cannot bridge the gap. This stands in for the finger stroke. |
| **Wheel collision proxy 6 mm wide**, full 22 mm visual | F4: a rigid cylinder line-contact over-predicts scrub; a real tyre's patch does not behave that way. `Chassis.WHEEL_COLLISION_W` is the tuning knob. |
| **Scoop excluded from the floor plane** (collision bit 2, not 1) | Two rigid bodies both bottoming at z = 0 can never slide under each other. The real 0.5 mm knife edge sits below the disc's under-face; this reproduces that. |
| **Chute base is a two-blade escapement**, not a shutter | A flap at Za 11 with 8 mm to the plate cannot swing, and a plain shutter releases the whole column (F19). Spec §6.4's "one disc per stroke" *is* an escapement. |
| **Positive-feed plunger on the bore axis** | F14: gravity does not seat the last piece. Parked at Za 84 so it is never in the drop path. |
| **Hold-down strip over the belt tail** | F16: bore geometry cannot stop a coin jam; the piece has to arrive flat. |
| **Ball transfers on compliant contact** | The spec's "rear pair on 1.5 compliant mounts". Rigid balls left the chassis on a two-wheel line contact and it rocked, repeatedly lifting the drive wheels. |
| Thin parts given thicker collision proxies (belt 4 mm, shim 1.5 mm) | Standard practice; visual geometry keeps the real dimensions. |

---

## 5. Two MuJoCo traps that cost real time here

Worth knowing before you edit anything:

1. **`ctrlrange` is not converted to radians.** The compiler's `angle="degree"`
   converts *joint* ranges but leaves *actuator* `ctrlrange` exactly as written.
   Commanding a hinge position actuator with `-5.4` meant −5.4 **radians**, which
   slammed the finger into its limit and threw the whole robot across the field.
   All angular actuators here are commanded in radians.
2. **Friction combines as the element-wise MAXIMUM.** A geom with
   `friction="0.05"` against a floor with `friction="0.6"` gets **0.6**. Low-friction
   parts need an explicit `<contact><pair>`. This is why the ball transfers
   behaved like rubber feet at first.

---

## 6. Layout

```
sim/
  rfgyc26/
    params.py      single source of truth -- every dimension, plus 11 assertions
    mjcf.py        generates the MJCF from params.py (never hand-edit the XML)
    robot.py       Agent A: stepper drive with step-loss, servos, sensors
    route.py       the mission, written as generators; every action time-guarded
    referee.py     Senior scoring for the sample mission
  scripts/         check_geometry, demo_belt, demo_post, demo_pick_place
  models/          generated MJCF (regenerated on every run)
```

Change a dimension in `params.py`, re-run `check_geometry.py`, and the model
follows. That is the point of the split — the sim cannot silently drift from the
drawings.

---

## 7. Next steps, in the order I would do them

1. **Close the docking loop on a physical feature.** This is now the single
   biggest lever: the posting budget is 2 mm (F21) and dead-reckoned docking
   lands at 1.5–2.1. The spec's own answer — reverse until the chassis stalls
   against the laboratory structure, then step a known offset — turns the Y error
   into the repeatability of a mechanical stop. Use the TCRT line array on the
   plate markings for X.
2. **Model the sweeper finger stroke in contact** and re-test F1 — still the one
   finding that could change the physical design.
3. Run the whole mission with `Field.LAB_CHAMFER = 0.0` as the default, since
   that is what the rulebook actually promises.
4. Then Agent B: same chassis, plus the lane cassette, camera triage and gates.
