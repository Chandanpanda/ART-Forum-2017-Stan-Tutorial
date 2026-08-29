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
python scripts\demo_beams.py         # 5. the beam phase alone: seal the quarantine
python scripts\demo_pick_place.py    # 6. the full Agent A match (~2 min wall clock)

python scripts\demo_pick_place.py --gui        # interactive viewer
python scripts\demo_pick_place.py --seed 7     # different random disc placement
```

`models\scene_pick_place.xml` is regenerated on every run and can be opened
directly in MuJoCo's `simulate.exe` if you have it.

Optional, for video: `pip install "imageio[ffmpeg]"` then add `--video`.

---

### Watching it in the viewer

```
python scripts\demo_pick_place.py --gui              # paced to real time
python scripts\demo_pick_place.py --gui --xray      # chassis transparent (X toggles)
python scripts\demo_pick_place.py --gui --speed 0.25 # quarter speed
python scripts\demo_pick_place.py --speed 0          # headless, as fast as it goes
```

`--gui` uses MuJoCo's `launch_passive`, which hands the physics loop to the
script — so the viewer does no pacing of its own and, unthrottled, the run is
about 3x real time. `--speed` sleeps to match a wall-clock rate; it is a ceiling,
not a floor, and it does not touch the physics (same seed, same score).

In the viewer: **press `X` to make the chassis plates transparent** and watch the
belt, hold-down, chute, collar, escapement and feed plunger work inside — press
it again to bring them back. `--xray` starts that way. It is a rendering change
only: `geom_rgba` does not touch contact, so an x-rayed run is bit-identical.

**Left-drag orbits, right-drag pans, scroll zooms.** `[` and `]`
cycle the fixed cameras — `field` (the whole 2000x1000 arena), `lab`, `quar`, and
`A_chase` which follows the robot; `Esc` returns to the free camera and `Tab`
toggles the side panel. The free camera opens framing the whole field, at which
distance the robot reads as a single box — zoom in or press `]`.

### Auditing what is actually simulated

```
python scripts\model_report.py
```

Prints the compiled model — every joint, actuator, sensor and collision geom the
solver really integrates — and then lists the seven places the model is a
**stand-in** rather than a simulation. Read that second list before quoting any
number from here.

## 2. Status — where this actually stands

**The quarantine is sealed in 10 of 12 randomised matches, inside the 120 s
clock**, and the mean score at the buzzer is **+69** — against +44 for the
sample task alone, and +27 for the run that started this round of work.

| | |
|---|---|
| Model generation from one parameter file | **works** |
| 23 geometry assertions | **all pass** |
| Conveyor carry on the incline | **works** — 59.2 mm/s against a 60 mm/s belt |
| **Pick: samples off the floor into the magazine** | **works — 24 of 24**, all three seated |
| **Magazine escapement: one piece per stroke** | **works** — in the mission, not just on the bench |
| **Dock a slot in a 6 mm laboratory** | **works — 0.7–2.1 mm**, ~15 s each |
| **Seal the quarantine with both beams** | **works — 10 of 12**, +70 |
| **Full match inside 120 s** | **11 of 12** |

```
seed            1    2    3    4    5    6    7    8    9   10   11   12
samples        +9   +9   -9   +9  +27   +9   +9   +9   +9   +9   +9   +9
beams         +70  +70   +0  +70  +70  +25  +70  +70  +70  +70  +70  +70
AT BUZZER     +79  +79   -9  +79  +97  +34  +79  +79  +79  +79  +79  +79
finish (s)    108  110    -   98  113   97  106  100  102  100  115  102
mean at the buzzer  +69.4      sealed 10/12      over the clock  1/12
```

Where this came from, same twelve seeds:

| | mean at buzzer | sealed | over 120 s |
|---|---|---|---|
| samples only, before this round | +44 | — | 5/12 |
| samples only, after the clock work | +37 | — | 2/12 |
| **samples + beams** | **+69** | **10/12** | **1/12** |

One more measurement worth having: dropping the beam budget to 44 s buys the
laboratory a **second** slot, worth +18 — and costs far more than it earns.
The seal then starts 15 s later and misses: over the same twelve seeds the mean
fell from +69 to about +50, with four matches running past the buzzer with no
beams down at all. The 70-point task has no slack, so it takes the clock first.

**Read the sample column honestly: it is one slot, not three.** The two tasks do
not both fit in 120 s, so the laboratory runs on a budget and stops when the
beams need the clock (F51). One posted sample and a sealed corner is +79; three
samples and no seal is +50. Seed 3 is the one match that still fails outright —
it loses the magazine early and then has nothing to post.

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

**F23. The queue against a closed gate never goes static, and the peak load is
4× the mean.** The gate force was being read at a single instant after 8 s. Over
the following 2 s it actually ranges **0.000 to 0.468 N**, mean 0.146 — the
pieces creep on the moving belt, unload the gate completely, then re-seat against
it. A one-instant reading therefore lands anywhere in that band and lands
differently on different machines, which is exactly what happened: 0.134 N here
and 0.219 N on the same commit under Windows. The demo now averages over 2 s and
prints the range.

The mean is close to the µ·m·g·4 = 0.118 N the spec quotes, so the accumulation
claim stands. **Size the gate servo for the peak, not the mean** — 0.47 N against
a 37 mm shelf is ~0.017 N·m of holding torque, still nothing for an MG90S, but it
is 4× what the spec's figure implies and it arrives as an impact, not a load.

**F24. WITHDRAWN — it was wrong.** It claimed Agent A climbs a square 3 mm plate
edge unaided and that F11 had been overtaken. The evidence was a rig in which the
robot was **launched at initialisation** — it crossed the field in 2 s of a
90 mm/s drive and never touched the laboratory at all. See F33 for what actually
happens. The mission results quoted alongside it were real (they ran against a
solid 1 mm laboratory, which the robot does climb); the 3 mm claim was not.

**F25. The intake has a one-millimetre cliff, and the spec's roller is on the
wrong side of it.** The powered surface at the intake was modelled reaching the
floor (Za −0.3) — convenient, and not buildable. Sweeping that height, capture
over 8 randomised matches (24 samples):

| powered edge, Za | −0.3 | 3 | 4 | 5 | 6 | 9.5 | 13 | 17.5 |
|---|---|---|---|---|---|---|---|---|
| samples captured | 24/24 | 24/24 | 24/24 | 24/24 | **0/24** | 0/24 | 0/24 | 0/24 |

It is not a gradual falloff — it is a step between 5 and 6 mm, and the reason is
geometric: the powered edge has to get **under the piece's half-thickness**
(2.5 mm for a 5 mm disc) or it merely shunts it along the floor. Spec §3.1 puts
the belt nose at Za 17.5, set by the Ø16 roller. **That cannot pick anything up.**

The fix is a standard conveyor detail: a **knife-edge (nose-bar) transfer**
instead of a roller. A Ø2–3 mm nose bar with a ≤1.5 mm belt, its underside
0.5 mm off the floor, puts the belt face at 3.5–5 mm — just inside the cliff.
`Chassis.BELT_NOSE_Z` is now 3.0, and two assertions in `params.py` fail loudly
if the design drifts back toward a roller.

**F26. The mission is 40–60 s over the match limit.** The rules give 120 s (g.1);
the route takes 157–184. Everything above about scores is what the robot achieves
*eventually* — at the buzzer, seed 1 scores +25, not +50. This is now the top
risk, ahead of anything mechanical, and it is a route problem rather than a
physics one: 50 s goes on the hole-3 dock alone and 44 s on sweep dwell.

**F27. The "shaking without moving" during a dock was the lead-in cone.** Watching
a run in the viewer, the robot would stall mid-dock and vibrate for tens of
seconds. It is not a torque limit — instrumented, the drive sits at **0 % force
saturation** and the wheel joints turn at exactly the commanded speed. The robot
is held: `A_ball2 vs labcone2` at **6.7 N**. The rear ball transfers drive into
the lab-hole lead-in and wedge against it, because the lead-in was modelled as a
4 mm cone standing proud of a 1 mm plate. A chamfer is cut *into* a plate; on a
1 mm plate it is at most 1 mm deep. Modelled correctly, as a countersink within
the plate thickness, nothing catches — and a dock that used to burn its whole
55 s guard now converges.

If you see this in the viewer, that is what it is: the robot commanding a move it
cannot make because something is holding it, not a controller being timid.

**F28. Removing the fictions cost 10 points, and that is the useful part.** Same
code, same 12 seeds, only the laboratory model changed:

| laboratory model | inside 120 s | mean score at the buzzer |
|---|---|---|
| 4 mm lead-in as a raised collar (not real) | 5/12 | +36 |
| no lead-in at all | 2/12 | +19 |
| 1 mm countersink inside the plate (correct) | 2/12 | +26 |

The raised collar was catching near-misses that otherwise scatter. It cannot
exist, so the robot has to dock accurately enough not to need it. Posting
tolerance is the 2 mm radial clearance (F21) and in-mission docking lands at
1.5–2.1 — right on the edge, which is why the result is a coin-flip once the
collar is gone. **The fix is a physical datum, not tuning:** the raised gains
that would close the gap faster were tried and made it worse (dock error
1.9 → 3.8 mm, and hole 3 stopped converging at all).

**F29. Departing a hole by returning to its pivot station cost a third of the
match.** After posting, `dock_and_post` drove back to the station it had
approached from — then the next hole picked its own station, often on the other
side of the field. On one hole that round trip spent the entire 35 s guard and
never arrived. Departing forward off the plate instead, and letting the next hole
choose its own approach, took the typical mission from 158 s to about 105.

**F33. Agent A cannot drive onto the laboratory, and the laboratory is thicker
than the model says.** Two findings that only make sense together.

Re-tested on a rig that verifies the robot has settled before it drives — the
check the F24 rig lacked — reversing onto a square laboratory edge:

| laboratory thickness | 1 mm | 3 mm | 6 mm | 9 mm | 12 mm |
|---|---|---|---|---|---|
| result | climbs, 62 N | **blocked** | blocked | blocked | blocked |

The chute stops at Y 355–359 against a plate edge at 360. It does not get on at
all. That is the original F11, and it stands.

Meanwhile the model's 1 mm laboratory cannot be right. The rules require a sample
to finish **"completely inside"** a slot (2.1) and a sample is a **5 mm** disc, so
the laboratory is at least a sample thick. At 1 mm the disc falls through the slot
until it rests on the floor and stands **4 mm proud** of the surface — not inside
anything, and with only 2 mm under the robot's tail it is knocked straight out as
the robot departs. That is visible in the viewer, and it is what a user watching
an x-rayed run reported: *"the first drop is correct but after the movement of the
robot it gets displaced."*

Put together: **a laboratory thick enough to satisfy the rules is one this robot
cannot dock at.** Set `Field.LAB_PLATE_T = 6.0` and the mission collapses to one
sample or none. This is the top open risk in the whole project — bigger than the
match clock, bigger than the intake — because it is not a tuning problem.

The shape of the fix is clear and is parameterised in `params.py`, but **not
adopted**, because each half regressed the mission on its own and they need a
docking controller built to expect them:

* `Chassis.BALL_REAR_X = 90` (from 40) — moves the rear ball transfers forward so
  the tail **overhangs** the laboratory instead of climbing onto it. Docking puts
  the chute on a slot 40 mm inside the plate edge, so balls more than 40 mm
  forward of the chute stay off the plate; 90 leaves 14 mm of margin. Verified in
  isolation: with this, the robot reaches the slot at 1, 3 and 6 mm.
* `Chassis.TAIL_CLEAR = 14` with `TAIL_X = 90` — steps the shell up aft of the
  balls so the overhanging tail clears a wooden structure. Verified in isolation:
  it removes the rear-wall and side-plate contacts at 6 mm.

**F34. The slot probes are in, the datum algorithm is not.** Two downward
rangefinders (`a_probe_l`, `a_probe_r`) now straddle the chute axis 40 mm forward
of it, standing in for the spec's TCRT array. They read the laboratory surface
and see a slot as a step, which is what an edge-timing datum needs: reverse across
a slot, time when each probe crosses its rim, and the two chord lengths give both
the along-track centre and the lateral offset in closed form —
`e = (c_L² − c_R²) / 320` for probes at ±20 mm on a Ø60 slot.

It is not wired into the route yet, and the reason is F33: with the model's 1 mm
laboratory the step is 1 mm and the robot's own pitch as it crosses the edge
drifts the reading by as much, so the probes cannot resolve it. With the 6 mm
laboratory the rules imply, the signal is six times larger and trivially
detectable — but then the robot cannot get to the slot at all. **The datum is
blocked behind the overhanging-tail redesign, not behind the sensing.**

**F36. There IS a legal pivot south of the laboratory. F10 was wrong.** F10 said
the corridor between the south wall and the laboratory is 360 mm against a 370 mm
swept circle, so the robot could only turn west or east of the plate — which
forced every dock after the first onto a diagonal approach, and a diagonal drags
the rear ball transfers across the laboratory edge, where they jam (F33). Turning
0° → 270° with a 6 mm laboratory, measured:

| axle Y | 150 | 170 | 180 | 190 | 210 |
|---|---|---|---|---|---|
| result | wall, 136 N | wall, 102 N | wall, 89 N | **clean** | **clean** |

The swept circle is the *chassis*, and the chassis floor sits at Za 6 while the
laboratory is 6 mm tall — the corners pass straight over it. The only robot parts
low enough to touch are the ball transfers and the wheels, and both are well
inboard. So the pivot goes directly south of each slot, the approach is square,
and the cross-field trips disappear.

**F37. A 0.4 s wait is worth 100 seconds of match time.** Docking closes on the
chute, which is 106 mm behind the axle — so a residual yaw rate of 1°/s moves the
chute 1.9 mm/s. Starting the terminal controller while the chassis was still
settling from its turn made a dock that takes 15 s from rest burn three 20 s
passes and still finish 4 mm out. One `wait(rb, 0.4)` before the terminal took
seed 1 from **227 s to 125 s** and the dock error from 4.8 mm to 1.9 mm.

**F38. The slot-probe datum cannot be validated in this simulator, and that is a
statement about the simulator.** The probes are built — two downward rangefinders
40 mm forward of the chute at ±20 mm, standing in for the spec's TCRT array, with
the edge-timing maths worked out: reverse across a slot, record where each probe
crosses the rim, and the lateral error follows from the difference,
`e ≈ (s_L − s_R) / 1.79` for a Ø60 slot.

But a physical datum exists to correct **odometry drift**, and this robot has
none: `rb.pose` reads the simulator's ground truth, so the controller already
knows exactly where its chute is. The residual 2 mm is the controller failing to
converge, not the robot failing to know. Wiring the datum in would measure an
error that is zero by construction.

Making it meaningful needs the robot navigating on its own dead-reckoned estimate
— `rb.odo_steps` and the `--step-loss` injection are already there for it — so
that drift is real and the datum has something to recover. That is the honest
next step, and it is a bigger job than the datum itself: `route.py` reads true
pose throughout.

**F39. The rear ball transfers had zero margin, and that was the bimodal dock.**
With them at Xa 90 the geometry said the ball would sit 14 mm south of the
laboratory edge at the dock point. Measured, its surface sat at Y 360 — on the
edge. So each dock was a coin flip: graze it and converge in 15 s, catch it and
freeze. The frozen state is unmistakable once you look at it —

```
fore=-5.5  left=2.6  herr=-0.3   v=-11.0  w=-8.3   |vel|=0.09
A_ball3/lab_s 13 N
```

— identical every tick, the controller commanding 11 mm/s into a 13 N contact
until its guard expired, then doing it twice more. Moving the balls to Xa 115
gives the margin the arithmetic assumed and the failure disappears. **Check
clearances by measuring a geom's world position, not by adding up offsets.**

**F40. Steppers are position devices; do not servo them continuously into a 2 mm
budget.** The terminal commanded `v = 2·fore` and the chassis coasted straight
past: at `fore = −2.8` it asked for 5.6 mm/s while the robot was still moving at
13. It hunted around the target and never landed. Creeping instead — 0.18 s of
motion, 0.26 s stopped, error re-measured on a stationary robot — converges, and
the dock is only accepted when the chassis is actually stopped (`|v| < 3 mm/s`),
so it cannot book an error it is about to move away from.

Scope matters as much as the idea: applied across the whole endgame the duty
cycle spends 60% of the time stopped, which fixed the hunting seeds and pushed
the marginal ones over the match clock. Restricted to the last 8 mm it does both.

**F41. The escapement shelf was closing on the piece it had just released.** The
magazine metered 3 of 3 on the bench and still lost samples in the mission, which
is the signature of a timing bug rather than a geometry one. Watching the stack
through a release, in robot-frame coordinates:

```
hole 1, works        gate out  d1(z= 5.4)     gate back  d1(z=2.5)   landed
hole 2, fails        gate out  d2(z= 7.2)     gate back  d2(z=9.1, dy -10.7)
```

At hole 2 the released disc was still at Za 7.2 — barely below the shelf line at
Za 8 — when the shelf came back after 0.28 s. The returning shelf caught it,
swept it **10.7 mm sideways**, and left it jammed half in the bore, where the
departure then dragged it along the laboratory at 3 N.

It does not fall freely, which is why 0.28 s was not enough: the retainer's knife
lip is resting on the piece, so it is *released* rather than dropped, and it
leaves at zero velocity with the lip still in contact. Holding the shelf open for
0.60 s costs 1 s across the whole match and fixes it — every hole now reads
`gate out z≈5.6 → gate back z≈2.9 → landed z=2.5`.

The bench rig never caught this because it loads the magazine by placing discs at
fixed heights, which leaves them centred and free. A stack the sweeper built sits
a few millimetres off-axis, and off-axis is where the shelf can reach it.

---

**F44. The beam pockets cannot have outboard walls, because the robot does not
fit beside its own cargo.** The sealed corner is 280 × 250 and Agent A is 285
long, so the moment either beam is down the robot no longer fits alongside the
other one. Every arrangement with a conventional boxed pocket left the chassis
5–25 mm inside a placed beam. What does fit is a channel whose outboard face
**is the beam**: an inner wall at |Ya − 117.5| = 95.5, an open bottom, an end
stop that does the pushing, and a lip that drops away to release. The loaded
envelope is then exactly the 235 spec §4.2 asks for, and the robot's own
structure stops 20 mm inboard of the beam it carries. The first build kept the
old side plates on Ya 0/235 — exactly where the beam now rides — and the robot
stalled on its own cargo 56 mm short of the wall.

**F45. A beam dragging on the floor cannot cross the laboratory, and that alone
makes the task unsolvable.** With a beam aboard the swept circle is 185 mm. The
corridor south of the laboratory is 360 mm, so a pivot there needs 370. West of
the plate the gap is 351.5. East of it, 351.5. North of it the robot would have
to get there first. Result: **with a beam on the floor there is no legal pivot
anywhere in the southern half of the field**, and the two beam stations are
90° apart. The fix is to carry the beams 12 mm clear (F46) — then they pass
over the 6 mm plate exactly as the chassis does, and every pivot F36 measured
comes back.

**F46. So the beams are carried on lifting cradles, and Rev C's "never lifted"
has to go.** One servo per pocket, 34 mm of stroke. It also removes the beam
sled drag the spec budgets 1.1 N for. Two attempts failed first: a cradle that
only drops to floor level leaves the beam resting half on the shelf and the
robot tows it 6 mm off station as it leaves, and a cradle that retracts
outboard as it drops carries the beam 12 mm sideways and yaws it 9° before it
lands. What works is a plain vertical drop deep enough to put a **20 mm
retaining lip** below the field as well — 34 mm — because a tall hook cannot be
got out of the way by lowering at all.

**F47. The escapement retainer was parked inside a beam pocket.** 80 mm long on
a 74 mm stroke, so parked it reached Ya 114 — 16.5 mm into the pocket, at
exactly the height a carried beam rides. It rubbed the beam all match and
dragged the placed one off station. A blade cannot be parked clear of a Ø66
bore *and* stay inboard of Ya 95.5 unless it is shorter: 62 mm on a 64 mm
stroke does both, and a Ø56 disc only needs ±28 mm of support, so the
escapement's job is unchanged. **This one is a real packaging clash in the Rev C
drawings**, not a simulation artefact.

**F48. The release is 45 mm, not a beam length.** There is no 280 mm of straight
line in the corner to withdraw down. There does not need to be: once the cradle
is down, the only thing still touching the beam is the end stop behind it, so
backing that off by a few tens of millimetres IS the release. What the robot
then needs is 185 mm of separation before it may pivot — that is a different
number, and the route has to buy it explicitly.

**F49. There is exactly one order and one pair of lanes.** Beam 2 (south wall)
first, beam 1 (west wall) second, and beam 1 approached from the NORTH side.
The reason is turning room: beam 2's station sits 118 mm from beam 1's east
end, so if beam 1 goes down first, beam 2's station can never be entered — no
heading change is available anywhere on its approach lane. Reversed, and with
beam 1's body on Y 270–465 (clear of beam 2 by 20 mm), every leg is a straight
run and every pivot has its 185 mm. Two further limits set the lanes: the
robot's forward shell sits at Za 6 and the plate top is at Za 6, so driving
over the laboratory drags at 9 N — the beam-phase lane is X 185–254; and beam
1's line cannot be driven onto, so the robot **shuffles** onto it (turn 30° off
the line, run 75 mm, turn back, run back — a differential-drive parallel park).

**F50. A carried beam has to be clamped, not carried loose.** Modelled as a free
body resting in its channel, a 200 g beam slops around inside the pocket
clearance and its inertia arrives a moment after the chassis's on every start,
stop and turn. With the beams aboard the laboratory dock went from 1.5 mm in
14 s to **38 mm in 87 s**, and the whole match collapsed. The real cradle wedges
the beam against the inner wall, which is a clamp; modelled as a weld that
releases when the cradle drops, the dock is unaffected.

**F52. The beam end stops must start ABOVE the laboratory, and this one
masqueraded as something else entirely.** Built at Za 2 they hang 4 mm into a
6 mm plate, and the robot ploughs it on every reverse dock: hole 1 went from
1.4 mm in 15 s to **57 mm in 87 s**. The obvious suspect was the beams' 380 g,
so the test was a three-way comparison — no beams, beams present but massless,
beams at full mass — and all three failed *identically*. That is what said the
cargo was innocent and its pocket hardware was not. A carried beam sits at
Za 12–72, so a stop spanning 16–52 engages it and still clears the plate.
The same trap caught the cradle: its shelves and lips are on the belt's
collision bit, not the floor's, so they can drop below the field plane.

**F51. Both tasks do not fit in 120 s, and the samples are the ones to cut.**
Samples ≈ 70 s, seal ≈ 52 s. At 50 points against 70, the laboratory is the
phase that gives way — and the order is forced anyway: the beams seal the
quarantine, so they cannot precede the sweep, and once beam 1 is down the robot
is **boxed in** (north of a 60 mm obstacle, west of a laboratory it cannot
climb, 87 mm of gap against a 235 mm body). The beams have to be last.

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
  scripts/         check_geometry, demo_belt, demo_capture, demo_post,
                   demo_beams, demo_pick_place, model_report
  models/          generated MJCF (regenerated on every run)
```

Change a dimension in `params.py`, re-run `check_geometry.py`, and the model
follows. That is the point of the split — the sim cannot silently drift from the
drawings.

---

## 7. What the build has to change, before you order anything

These are the Rev C departures the beam work forced. All of them are cheap; the
point of listing them here is that none is optional.

| Change | Why | Cost |
|---|---|---|
| **Beam pockets lose their outboard walls** — inner wall on Ya 95.5/139.5, open bottom, the beam's own face is the envelope | The robot does not fit beside its own cargo once either beam is down (F44) | none, it is less material |
| **Beams are carried 12 mm clear and set down to place** — one cradle servo per pocket, 34 mm stroke, 20 mm retaining lip | A dragging beam cannot cross the 6 mm laboratory, and that left no legal pivot in the southern half of the field (F45/F46) | 2 × MG90S-class servo + a cam |
| **The escapement retainer shortens to 62 mm on a 64 mm stroke** | At 80/74 it parks 16.5 mm inside a beam pocket, at exactly the height a carried beam rides (F47) | none, it is a smaller part |
| **The pocket end stops start at Za 16, not at the floor** | Below Za 6 they plough the laboratory on every reverse dock (F52) | none |
| **The cradle wedges the beam against the pocket wall** | A loose beam's inertia arrives late and destroys the dock (F50) | shape of the lip |
| Sweeper finger pivots may need 5–8 mm aft | Only if you keep the beam's forward face flush with the chassis nose | none |

Bench-test before committing: the intake at a ≤5 mm powered edge (F25), and
turn efficiency for `Chassis.WHEEL_COLLISION_W`.

## 8. Next steps, in the order I would do them

1. **Seed 3, and the sample column.** One match in twelve loses the magazine
   early and scores −9; the rest post one slot out of three because the seal
   takes the clock. Both are the same problem — the sample phase is 70 s and
   should be 45. The sweep dwell and the three-pass dock are where it is.
2. **Odometry instead of ground truth** (F38), and the slot-probe datum on top
   of it. Until the robot can be wrong about where it is, the datum has nothing
   to correct — and that is the difference between these numbers and the real
   robot's. It matters more for the beams than for the samples: every beam lane
   here is dead reckoning between two wall stalls.
3. **Bench-test the intake before ordering the belt** (F25). A knife edge at
   ≤5 mm picks up; a Ø16 roller never does.
4. **Measure turn efficiency on the real robot** and set
   `Chassis.WHEEL_COLLISION_W` from it.
5. **Bench-test the beam cradle**: 200 g at 108 mm off the pocket wall, 34 mm
   of lift, and the release has to leave the beam standing within 5 mm.
6. Then Agent B.
