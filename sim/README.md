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
| **Pick: 3 discs off the floor into the magazine** | **works — 3 of 3**, `demo_capture.py` |
| **Place: gate stroke into all three lab holes** | **works — 3 of 3, +45**, `demo_post.py` |
| Closed-loop chute docking | **works** — 0.1–0.3 mm from a standing start |
| Sensors: rangefinder, insidesite | **verified** |
| Referee scoring (Senior sample rules) | **works** |
| **Full mission end to end** | **+27 — 2 of 3 samples placed** |

Both halves of pick-and-place now work completely in isolation, and the full
mission places two of the three samples. The remaining gap is dock precision at
**hole 3 only**: it is the one hole reached from the eastern pivot, its approach
is diagonal, and a straight reverse leaves ~15 mm of lateral error against the
~10 mm a Ø60/Ø56 hole will accept. Holes 1 and 2 converge to ~2 mm. Forcing hole
3 onto the western pivot is worse (a 420 mm blind reverse drifts ~150 mm).

The honest fix is the one the spec already specifies and this model had to remove:
a **45° chamfer** at the hole. Spec §6.4 says it "absorbs ±10 of robot position
error", and that is exactly the margin hole 3 is missing. See F12.

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

**F12. Removing the hole chamfer costs ~10 mm of dock tolerance, and that is
exactly the margin hole 3 needs.** The chamfer had to come out of the model
because the cone helper's tilted segments protruded above the plate and the
chassis rear wall crashed into them at 33 N. Rebuilding it properly is the single
highest-value next step — it is what closes the last third of the mission.

---

## 4. Modelling decisions you should know about

Each of these is a deliberate, documented departure from the drawings.

| Decision | Why |
|---|---|
| **One continuous conveyor** from the scoop tip to the tail roller, instead of a 0.5 mm shim feeding a belt whose nose is at Z 17.5 | F1: a passive shim provably cannot bridge the gap. This stands in for the finger stroke. |
| **Wheel collision proxy 6 mm wide**, full 22 mm visual | F4: a rigid cylinder line-contact over-predicts scrub; a real tyre's patch does not behave that way. `Chassis.WHEEL_COLLISION_W` is the tuning knob. |
| **Scoop excluded from the floor plane** (collision bit 2, not 1) | Two rigid bodies both bottoming at z = 0 can never slide under each other. The real 0.5 mm knife edge sits below the disc's under-face; this reproduces that. |
| **Chute base gate slides** rather than flapping | A flap at Za 11 with 8 mm to the plate cannot swing. Spec §6.4's "one disc per stroke" reads like an escapement anyway. |
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

1. **Model the sweeper finger stroke in contact** and re-test F1 — this is the
   one finding that could change the physical design.
2. Close the docking loop on a physical feature (the spec's own rule): use the
   front ToF against the lab plate edge instead of dead-reckoned pose. That
   should fix two of the three postings (F6).
3. Multi-disc magazine behaviour — check that a second disc arriving does not
   push the first back out of the open-fronted bore.
4. Then Agent B: same chassis, plus the lane cassette, camera triage and gates.
