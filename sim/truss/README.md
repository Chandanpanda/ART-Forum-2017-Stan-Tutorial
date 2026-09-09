# The truss cell — a fabrication cell for wound carbon-fibre Warren trusses, as a model

`sim/truss` is the design of an automated cell that builds the stereo-rig
truss of the brief — three carbon chords, mitred diagonals laid on them at
±α, every joint lashed with a band of Kevlar thread wound by a C-ring, a
drop of epoxy on each band, baked — and the checks that say what the
design can and cannot do.  It follows the rules of this repository: no
pose, path or threshold is written down; each is computed from a truss
spec, the machine's spec and the fixture, so the same code answers for a
300 mm test piece and the metre-long rig, and would answer for a truss it
has never seen.

```
python3 sim/scripts/truss/check_all.py          # the fast tier, ~12 min
python3 sim/scripts/truss/check_all.py --slow   # + the cable rig and the rendered camera
python3 sim/scripts/truss/demo_cell.py --gui    # watch a truss being made
```

On Windows the interpreter is `python`, and the paths use backslashes.
Do not set `MUJOCO_GL`: `osmesa` is a Linux-only backend name and MuJoCo
raises on it before it loads, so `truss/glenv.py` picks one per platform
and hands the viewer none at all.

## Three tiers

| tier | what is real | what it is for |
|---|---|---|
| 0 arithmetic (`spec`, `geometry`, `structure`, `fixture`, `approach`, `schedule`, `band`, `inspector`) | closed forms and small solvers, numpy only, milliseconds | choosing the truss, sizing the machine, every pose and every clearance in mm, the plan and its clock |
| 1 rigid physics (`mjcf`, `cell`, `hal`, `process`, `vision`) | the whole cell in MuJoCo: gantry, ring, gripper, dispenser, cage, rods; the band is counted, not simulated | does the plan survive contact — loading, seating, indexing, winding stations, dosing — and does the process built on the HAL make the truss the plan promised |
| 2 the thread (`rig`) | one joint, a cable with bending stiffness wound by a spindle under a fixed exit guide, at a tension | what a hoop actually goes round, how much thread it takes, whether winding disturbs a clamped rod, and what the dry band does and does not hold |

Tier 0 is where every number lives.  Tier 1 exists because the expensive
mistakes are not bad plans but plans made against a world that is not the
world; every suite below caught one.  Tier 2 exists because a band of
thread is the one part of the process no rigid body can stand in for.

## What was decided, and what decided it

Everything below is a measurement from one of the suites, not a choice.

**The truss.**  `structure.design()` sweeps side, α and stock against the
brief's rules — tip slope under the camera load with a safety factor of 2,
first mode ≥ 200 Hz, member modes ≥ 330 Hz (the propellers' band), α in
35–55°, section ≤ 100 mm, and two rules of the cell's own: the ring must
fit round the joint and the cage's spine must fit under the ring.  The
brief's own 45°, 3 mm/2 mm truss comes out with f₁ = 195 Hz and its
diagonals 1 mm short of the ring's rim; the optimiser's metre truss is
side 92, α 40°, 3/2 mm stock, 54.9 g, 27 joints, and the 300 mm test
piece is side 66, α 45°, 2/1 mm, 7.4 g, 12 joints, spine-bound.  The
brief's own 300 mm section, 40 mm across, is stiff enough several times
over and cannot be wound at all: the ring sweeps 20 mm of radius about the
chord it is on, and a 40 mm section leaves no room for the cage's spine or
the other two chords.  A short truss's section has a lower bound set by the
winding head, not by the loads.

**The ring.**  Its inner radius is a closed form,
`r_in ≥ d_c/2 + (W/2 + band/2)·tan α + d_d/(2 cos α) + clear/cos α`: the
plate's half-width AND half the band sweep, because the ring turns at every
x across the band, and the rim's corner meets an oblique diagonal at
cos α of the radial slack (`check_approach` measures the corner on a 50°
truss to 0.1 mm).  A 10 mm wide plate fit no truss once the band was
charged; the plate is 4 mm.  The gap parks *toward the joint's face*, not
straight down: at the band's ends, where the descent is tightest, toward
the face clears 1.9 mm and straight down 0.6 (metre truss).  Between
joints the ring lifts 13–14 mm, the least that clears the next pin — not
the Z stroke.

**The drive, which was the design's weakest claim.**  A ring that encircles
a rod cannot have a shaft through it — the rod is where the shaft would go
— so it is driven on its rim, and the first version drove it with two
friction wheels and hung a thread spool on the same rim the wheels ran on.
Three faults, all measured: the orbiting spool struck each wheel once a
revolution; nothing located the ring (two wheels are a drive, not a
bearing); and the thread alone asked 0.022 N·m against a slip torque of
0.03 that was a guess.  What replaced it:

* **The ring is its own spool.**  A whole metre truss is 7.8 m of 0.15 mm
  thread — 139 mm³, less than a drop — so it winds into a groove in the
  ring's own web, the way a toroidal winder's shuttle carries its wire.
  The rim is left for the drive and the head's swept radius falls from 32
  mm to the rim, which is what let the cage's spine grow from 3 mm to 8.
* **Teeth, not friction, and the tooth counts are SOLVED.**  The first
  toothed version wrote 72 teeth on the rim and a 12-tooth pinion of 6 mm
  pitch radius: module 0.556 against module 1.0, two gears that cannot
  mesh.  `Ring.mesh()` now solves it — a whole number of teeth on the rim
  AND in the gap so the far side arrives in phase, the smallest pinion that
  neither undercuts nor swallows its own bearing, Lewis at the root with a
  factor of 3, and of the survivors the coarsest that costs no envelope.
  It picks module 0.8: 48 teeth on the rim, 8 of them in the gap, an
  18-tooth pinion, 1:2.67, 6.0 MPa in the root.
* **Three pinions, phased, on one belt.**  `Ring.pinion_az()` places them
  by maximising the smaller of two margins in degrees — clear of the mouth,
  and further apart than a gap plus two contact arcs so one gap cannot
  unmesh two.  It answers 66/180/294 with 18.6° in hand; hand-picked
  110/205/300 had 48° of mouth margin and 11 of spacing, spending margin
  where it was already rich.
* **The turns are known at the motor.**  Friction slipped, so the old head
  counted its turns by watching a fiducial go past.  A toothed rim cannot,
  so `cell.angle()` is an encoder read and the only error left is the
  quantisation: 0.034° at the ring.

**What the drive rig found that no arithmetic did.**  `check_drive` builds
the ring as a **free body** — no hinge, no weld, six degrees of freedom —
held by nothing but a C-channel raceway and three involute pinions, and
turns it against the thread.  Two findings came out of it:

*The raceway does not locate the ring where it matters.*  A contact needs
rail at that azimuth AND ring at that azimuth; the rail is missing over the
mouth and the ring over its gap, so side by side they leave a **dead arc**
where nothing can touch it.  Pushed into that arc the ring runs until rail
half a dead arc away catches it — obliquely, and an oblique catch is a
wedge.  With the mouth at 100° the dead arc was 160°, the catch 80° off,
and a 2 N thread became 7 N on the rail: **the drive jammed every time the
gap crossed a pinion**, at every clearance under 0.12 mm, and turned only
at 0.15 or looser — where the ring travelled far enough to fetch up against
a pinion instead.  A design that needs a *loose* fit to turn is being held
by the wrong part.

*The mouth was 100° for a reason that does not hold.*  It was sized to
contain the ring's 60° gap.  Only the chord ever reaches the rail's radius,
and only at one azimuth — at the ring's own x-window the diagonals are
inside 5 mm of the chord's axis and never see the rail at 23 mm.  So
`Ring.race_mouth()` solves it instead: the narrowest mouth that still costs
no envelope, `2·acos(rim/rail)`, which is 59.6°.  The dead arc falls to
120°, the wedge from 0.05 to 0.013 N·m, and the drive turns at **every**
clearance from 0.05 to 0.20 mm, at RPM_MAX, at twice the tension.  The
clearance is then free to be what it should have been all along, an H7/g6
running fit.

What the rig measures on one revolution of the settled design: the ring
arrives where the shaft says to 0.30° of a 7.50° tooth, crossing a pinion
with the gap costs 0.04°, at least two pinions are engaged throughout, the
steady demand is 0.028 N·m of a 0.120 rating against a thread's own 0.022,
and the ring's centre moves 0.072 mm — which the spec carries as
`Ring.RUN_OUT` and charges against both the bore and the swept radius.
Press **one** pinion on a quarter of a pitch out and it jams; shift all
three together and it does not, because the ring is free and simply turns
half a tooth to suit — the phase that matters is relative, and the check
that shifted all three was passing for the wrong reason.

**The fixture.**  Pins hold the chords in 90° V-notches at a computed
pitch, kept out of every joint's band sweep and off every chord's midpoint
(the gripper needs it); cradles hold the diagonals, pushed along them until
a seated head clears them by `SEAT_CLEAR`; the spine is as fat as the
ring's swept radius allows.  A V is a kinematic locator: `check_load` releases a
chord with deliberate arrival errors and it seats within 0.001 mm; the
measured capture range is 2.2 mm against the closed form's 1.5.  A rod is
kept by welding it to the cage once seated (a magnet or clip on the
machine): a free diagonal turned face down leaves its cradles by 189 mm,
so the keeper is a requirement, holding a millinewton.  Kept rods leave
the fixture's collision set — welded *and* in contact, a diagonal drifted
30° over one index.

**The gripper.**  Flat pads, on a yaw servo and a 40 mm stroke, beside
the ring.  Two things measured that no drawing showed: a pad reaching more
than 0.4 mm below the rod's axis lands on the V's flanks before the rod,
and a rod turned while retracted — 6 mm above the ring's centre, beside
its plate — swings its ends through the rim (−0.5 mm clearance at ±α;
extended, 16.5 mm).  The sweep stopped at 37 of 135° against the ring and
the rod slipped in the pads.  So the yaw waits for the extension, at a
height `approach.yaw_height` solves — 4.5 mm over the seat on the 300 mm
truss — and `check_schedule` pins the order.

**The magazine.**  Chords beside the cage, three deep; diagonals in a rack
at each x end, along y, pitched 11 mm (10 let a pad catch the next
slot's flank).  A rack beside the cage would need a 500 mm Y axis.  X
travel is 1400 mm, the brief's own 400 mm past the truss.

**The plan.**  Chords load chord-up, diagonals face-up (six cage indexes);
then a serpentine over the chords, each run anchored round a post at both
ends, each joint looked at, parked, descended on, wound sensor-terminated
with an x feed, lifted and dosed on the spot.  Dosing on the spot is 1.4
min quicker than a separate pass.  The metre truss takes ~23 min, ~10 of
them loading; the 300 mm piece 9.4 min.  The post loop's shape is solved
too: the rectangle a first draft drew by hand hooked the ring's plate on
the post's lower end, jammed the x axis, and put the first look 60 mm
from its joint; `approach.post_loop` finds the height (17.5 mm over the
chord) and leg (3 mm) that clear by `SEAT_CLEAR`.

**The camera.**  Where it rides is a rendered finding, four times over: on
the carriage over the ring plane it photographs its own carriage box and
the head; outboard of the dispenser it sees the chord but the diagonals —
the mitre that gives the joint's x — hide behind the nearest pin's flanks;
in a bay outboard of the ring's plane it sees both, the far diagonal through
the ring's gap, which the plan parks downward for the look.  The fourth is
the STANDOFF, and it was the head's envelope that used to set it: the bay
sits just outside the head's widest static radius, so when the spool came
off the rim the camera came in with it — range 55 mm to 47, field 22.7 mm of
chord to 19.4 — and the path lost the first joint of every chord.  The
standoff is solved for `LOOK_FIELD` now as well as for the envelope.  The look is
always taken with the joint on the camera's side of the plate: through
the gap from the far side one diagonal is lost and the joint's x came back
up to 2 mm off.  The pixel path (`vision.PixelVision`) fits the chord's
centreline and both diagonals' lines near where the plan predicts them
and back-projects the difference; it is good to `Vision.look_sigma(α)`
along the chord and hundredths across it.  That is a LAW, not a constant,
and the difference mattered: written as one number measured on a 45° truss
(0.15 mm), it read the optimiser's move to a 35° web as a broken camera.
The joint's x comes from where the diagonals' LINES meet the chord and
their fit carries a lever arm of 1/sin α, so a shallower web is a worse
look in proportion — 0.20 mm at 35°, 0.16 at 45°, measured on rendered
frames. The model camera draws its noise from the law, not from the
sensor's 0.006 mm per edge.

**The dispenser.**  The drop hangs from the tip, so the tip stands off by
two drop radii plus the fall; a tip 1 mm over the cluster spawned the drop
inside the chord and the solver threw seven of twelve to the floor.  A
drop sticks the instant it touches (judged every physics step: judged
once a control tick, one that had met the crest was already rolling off)
and then stops colliding — resin wicks in; welded while still pressing on
the chord, each drop walked it a millimetre out of its notches.

**The cycle.**  `check_cycle` makes the 300 mm truss end to end through
`schedule.plan` and `process.Executor` against the MuJoCo cell: 377 ops,
562 s simulated against 558 planned, every rod seated within 0.005 mm,
every band 18.0 turns and centred within 0.16 mm (the camera's 0.07 mm
bias and the look), every drop on its joint, the head touching nothing at
the end.  The band's centre once read 7.9 mm off on every joint of the
return run: its width was added to the start whichever way the feed ran.

**The thread, and the band model it corrected.**  Tier 2 is a lathe: the
joint turns under a fixed guide at `Ring.EXIT_R` and the thread pays out
under a constant tension — a ring that carries its thread in its own web
still cannot carry a pay-out reserve round the cage.  MuJoCo's contacts scale with the mass they act
on, so the thread is a proxy — 1 g a millimetre, twice its radius, 0.5 N
instead of 2 N — and `rig.py`'s docstring says which ratios survive.

It refuted the band arithmetic.  Tier 0 had a hoop taking the cluster's
convex hull wherever it lay; wound, the hoops sat at 1.19 mm radius where
that hull reaches 6.3.  **A diagonal touches its chord at the joint and
nowhere else** — at `dx` along the chord it stands off by
`dx·tan α − d_diag/(2 cos α)` — so a hoop laid outside

    bind_half = d_diag / (2 sin α)  +  d_thread / tan α

finds a gap and beds on the chord *under* the diagonals, binding nothing.
On the 300 mm truss that is ±0.86 mm of a 8 mm band, on the metre truss
±1.73.  `Truss.bind_half` and `geometry.hoop_perimeter` are the closed
forms; the wound thread matches the corrected arithmetic to 3%, where the
hull model overstated it by a third.  The consequences: the recipe's band is mostly
thread on a chord, the thread mass was overstated (so the resin dose was
understated), and **a dry band does not retain the mitres** — released and
turned face down the diagonals leave, wound or bare.  The keeper holds
until the resin cures, which is what the plan does: `check_schedule` pins
that no keeper is ever released.

Two things the rig cleared rather than condemned: the winding does not
drag a clamped rod out of its seat (0.119 mm during a wind against 0.118
mm from the same rotation with no thread at all — gravity at the new
attitude, not the thread), and the thread it consumes is the arithmetic's.

## The payload, and the two numbers that moved everything

The truss is a stereo camera's spine, so the camera is a spec, not an
accessory. Three vendor drawings describe two candidate modules, and reading
all three is what caught the mistakes.

**The mass was never on any of them, and it was carried at 50 g.** The
vendor's comparison table weighs a Camera Module 2 at **3 g**. Everything
downstream of a tip mass an order of magnitude too large was answering a
different question: the section, the binding constraint and the choice of
web all moved when it was corrected. What the spine package found is in
`sim/spine/README.md`.

**And Module 3's lens moves.** Its sensor-assembly datasheet
(`RP-009992-DS-1`) gives an open-loop voice coil whose postural difference
is ±50 µm at a fixed drive current — 1.06% of the image distance, and so
1.06% of every range reported. At 100 m that is 1.06 m against the 0.84 m
that is the *entire* inter-camera budget the truss exists to hold. It is an
intrinsic; no rigidity touches it, and holding the drive current does not
remove it, because the postural term is what the coil does at a fixed
current.

**So the product uses Camera Module 2, which has no lens actuator at all.**
`Module2` and `Module3` both live in `spec.py`, the rejected part kept
beside the chosen one, and `check_geometry` asserts the trade rather than
the conclusion: Module 2's shorter focal length on bigger pixels costs
0.07 m of stochastic floor at 100 m to remove 1.06 m of drift.

**`RP-008149-DS-1`, Module 2's mechanical drawing**, read from the PDF's own
geometry rather than its dimension labels — everything scales at 35.525
units/mm, and against that:

| | measured | labelled |
|---|---|---|
| board | 24.996 × 23.871 mm | 25 × 23.862 |
| corners | 4 × 2.0 mm radius | 2.0 |
| holes | 4 × ⌀2.168 mm | 2.2 |
| hole pitch | 21.00 × 12.526 mm | 12.5 |
| hole pattern centre | 3.680 mm below the board's | — |
| lens housing | 8.500 × 8.500 mm square | 8.5 |
| optical axis | 2.477 mm above the board's centre | 9.462 to the top edge |

Module 2 and Module 3 share a board outline, a hole pattern and an optical
axis height to a hundredth of a millimetre — they are mechanically
interchangeable, which is why the mount did not have to be redrawn.

Two things the drawing does not give, both flagged in `Module2`: it is a top
view, so there is no height for the housing, and there is no barrel circle on
it. The optics are cross-checked instead — the published 62.2 × 48.8° field
and the measured 3.68 × 2.76 mm image area agree on one focal length to a
third of a percent.

**And a photograph settles what the top view cannot.** The lens assembly is a
*low square base with a tall round barrel* standing on it, so the base — the
part the six struts bond to — is a minority of the module's ~8 mm of
stand-off. `check_mount` sweeps that height over everything it could
plausibly be and finds the structure does not turn on it: 2% of the yaw
budget across 1.5–8 mm, and the standoff and the field clearance do not move
at all. **One** conclusion does turn on it, and it is now a caliper reading
rather than an argument: whether the plastic housing or the struts are the
compliance is decided by `CASE_E × CASE_PROUD` against the strut's 5.6 kN/mm,
so at the weakest plastic a housing could be, the base has to stand 1.87 mm
tall. The photograph says it comfortably does; a caliper and a material name
would say it for certain.

## The camera mount

`mount.py` solves a thirteen-rod nose at each end — three end battens closing
the triangle of chord ends, a four-rod **tic-tac-toe grid** round the camera,
six struts from the chord ends to the grid's four crossings — plus **one
laser-cut aluminium collar**, a frame with a closed square aperture that the
lens housing stands through.

**The collar is parallel to the board, and that took a render to find.** It
was drawn as a fin standing on edge in the plane perpendicular to the spine —
which is where a hexapod's platform wants to be, and is the one plane a plate
bonded to the lens housing cannot lie in. A fin through the camera's centre is
*inside the camera*:

| element | how far inside the module |
|---|---|
| collar 0, collar 2 | 4.50 mm — the fin's back half, buried in the PCB |
| grid 5, grid 6 | 4.50 mm — the two long tic-tac-toe rods laid along it |
| strut 14, strut 16 | 1.74 mm |

Six of nineteen elements at one end. Every number the mount reported was for a
part that cannot be made, and **nothing static asked**: every check about the
mount asked whether a feature was in the right *place*, and none asked whether
it was inside the part. `payload_clearance` asks now, in the camera's own
frame, against the module's real solid — which is *not* one 9 mm slab. Behind
the board's front face the module is solid over its whole outline; in front of
it there is only the housing, and the collar goes there. Tested against a
single slab the collar reads 2.25 mm inside a board it is *resting on*.

A plate on the housing can only lie parallel to the board. It is the one
orientation in which the aperture is a closed hole — four walls of bond,
51.0 mm² against the fin's three-sided 21.8 — and in which nothing of the
mount is behind the module's front face at all.

**The grid sits one clearance outside the module's silhouette**, and that is
not tidiness. The camera looks out through a *face*, so one chord lies
directly behind it, and a strut from that chord clears the board only by
staying outboard of it in x the whole way. Drawn 4.5 mm further in the mount
goes back through the PCB, and `check_mount` asserts exactly that, so the rule
cannot be quietly relaxed.

**Why a collar at all**, measured on the frame model. The camera offers two
surfaces of its own and both are refused:

| where the six struts could land | fraction of the yaw budget |
|---|---|
| the housing's own inboard face, 8.5 × 3.0 mm | **1.08** — over |
| the board's four M2 holes, board modelled rigid | 0.73 — the flattering model |
| …the same, board at its real 216 N/mm | far over |
| **the collar, its mass and its bending both charged** | **0.82** |

The housing face is the only surface parallel to the end triangle and is too
thin to *be* a triangle. The mounting holes make a proper triangle and then
work through FR4 over 21 mm — **26 times softer than the strut bonded to it**:
that path is not a mount, it is a spring.

**The collar is flat, and that is a measurement.** A 1.5 mm frame reaching
15 mm from the aperture to a crossing is 149 N/mm out of its own plane against
the strut's 5838 — 38× softer, the same failure the circuit board was rejected
for — so the obvious fix is to fold its rim into a channel. With the platform
ring carrying the band's real section and the plate's mass charged:

| fold | plate | corner | yaw budget |
|---|---|---|---|
| 0.0 mm | 2.49 g | 149 N/mm | **0.815** |
| 3.0 mm | 3.97 g | 2156 N/mm | 0.848 |
| 4.9 mm | 4.91 g | 6125 N/mm | 0.869 — what the stiffness rule asks for |
| rigid, at 3.97 g | | ∞ | 0.849 |

The last row is the point: a plate assumed *infinitely stiff* at the folded
plate's mass is no better than the folded plate, so what the fold buys was
already worth nothing. The rule the fillet is held to — be stiffer than the
strut you hold — is the wrong rule for a part whose mass is a tenth of the
head's.

**And the field test had a sign error.** `fov_clear` added the rod's radius
where it should have subtracted it, so it measured the *far* side of every rod
and let the near side into the picture: at the mechanical standoff it read
−0.05 mm where the end batten was 1.55 mm in shot. The solved standoff moves
14.09 → 15.59 mm.

Three things are still solved rather than chosen. The camera faces a **face**
of the truss, not a chord (aimed along a chord it needs 28.9 mm of standoff
instead of 15.6, and standoff is a lever arm). The **standoff** is bisected on
`fov_clear` until nothing is in a 62.2 × 48.8° field. And every rod's direction
is checked against what the machine has: a rod held at gripper yaw ψ with the
cage at θ lands along `(cos ψ, sin ψ cos θ, −sin ψ sin θ)`, which covers the
whole sphere, so `pose_for` returns the pair and no axis has to be added.

Every mount joint is an epoxy fillet: the ring cannot reach past the last truss
joint and does not need to. A fillet on a 1.5 mm rod carries 141 N against a
service load of 0.147 N, and comes out stiffer than the strut it holds — so the
strut is the compliance and the bond is not.

## The whole build, end to end

```
python3 sim/scripts/truss/demo_assembly.py                 # 300 mm, headless
python3 sim/scripts/truss/demo_assembly.py --1m            # the metre truss
python3 sim/scripts/truss/demo_assembly.py --video out.mp4 # record it
python3 sim/scripts/truss/demo_assembly.py --gui           # watch it live
python3 sim/scripts/truss/demo_assembly.py --shots DIR     # a still per phase
```

One plan, from racks a person has filled by hand to a truss with a camera on
each end. The mount is a **phase of the plan**, not a drawing hung on the
cage: thirteen rods an end plus the camera on its carrier, at cage angles the
same worm indexes and gripper yaws the same servo turns, then an epoxy fillet
at every rod end.

```
955 ops, 21.2 min simulated
9 rods loaded and seated, 9 joints wound, 9 bands dosed
28 mount parts placed, 52 fillets laid
<truss 9/9 joints, 9 rods seated of 9, 1.35 g on the joints>
```

Nothing in it is a scripted animation — every pose comes from `schedule.plan`
and is executed by `process.Executor` through the HAL against the same cell
the checks use, so if it collides here it collides in `check_approach`.

**Six things the run found that no static check was asking.** They are worth
listing because each is a class of question the suites did not have:

- **The mount rods fell on the floor.** A truss rod is released a millimetre
  above a V and the V takes it. The mount has no V's — its rods are laid onto
  the collar and onto each other — so the keeper has to take each one while
  the jaws are still shut. On the machine that is a tack before the gripper
  lets go.
- **The rack put its own blocks under the gripper's pads.** Two V-blocks
  `BLOCK_IN` from each end is the layout for a 300 mm chord; on a 42 mm strut
  it lands them 0.08 mm from the middle, which is where the pads close.
  `Magazine.blocks` solves the layout from the rod's length now, and returns
  *nothing* below the length at which nothing is left.
- **The camera is not a rod in its receptacle.** It is a 24 mm slab on a 9 mm
  base with a pin out of one side. On V-blocks under the boss it rolls off;
  in a flat nest it topples and drags itself out; with a wall-thickness of
  slack it slides 1.7 mm and puts its own board 0.3 mm inside where the pads
  close. It gets a printed pocket to the carrier's own fit, near wall split
  at the boss and stopping below the pads' reach, with a detent.
- **And the boss was too short — for the head, not the pads.** The ring's
  raceway reaches to within 14.3 mm of the grip point at the collar's offset,
  and the collar's top is 14.9 above it, so the head landed on the plate
  before the jaws reached the boss. `Carrier.boss_l` is now whichever is
  longer, the pads' rule or the head's envelope: 42.9 mm against 14.0.
- **The bond pass was aimed under the work.** The mount's rods sit all round
  the cage and the dispenser only reaches over the top; sent to cage
  coordinates at θ = 0 it stalled the z axis against the fixture and shook
  the truss off its own V's. Every fillet turns its own point up now.
- **The end battens were below the z axis's floor.** `rot_x(θ+180)` with the
  yaw negated is the same line in the cage's frame, but the rod's *position*
  flips with it; three battens an end came out 4.7 mm under what the gantry
  can reach. `poses_for` offers both and the plan takes the higher.

What is still open: 46 of the 955 ops overran their planned duration, all of
them `axes settle` or `stroke settle` in the mount and bond phases — the head
pressing on something as it works inside the cage. The plan completes and the
inspector passes the truss, but those are real contacts and the approach
solver does not yet plan the mount's own moves the way it plans the ring's.

## The fixture had to come out, and could not

The cage was drawn as a thing to wind onto and never as a thing to get back.
The truss closes around it:

| | |
|---|---|
| the exit — the section's inscribed circle less the fattest rod | **21.54 mm** |
| the pins' notch floors reach | 46.95 mm |
| the cradles reach | 36.18 mm |

96 pins and 54 cradles on the metre truss, every one of them 15 to 25 mm
outside the only hole they could leave through. `collapse.py` is a collapsible
mandrel, and each of its four ideas is forced by a measurement:

- **Rails, not arms.** An arm 39 mm long lying down takes 36 mm of axial room,
  and the metre truss has 1.2 m of arm for 1.0 m of spine. Arms that swing
  together carrying a rail fold once — the rail *translates* — so the room
  needed is one arm's. A **parallelogram** is what keeps it translating, and
  that matters: a V-notch that rotated as it retreated would rake the rod.
- **The chord rails are segmented.** They sit under the chords, which is
  exactly where the ring comes down to wind. So they run only on the free
  intervals `fixture.free_spans` already computes, and the head passes through
  the gaps — twelve pieces per chord, not one.
- **The face rails are not.** The cradles sit sixty degrees from any chord,
  where the head never goes.
- **Six shafts, six locks.** Thirty-nine rail pieces would be thirty-nine
  locks. Every arm at one azimuth is keyed to a torsion shaft on the spine's
  surface — the one radius the head can never reach — so one shaft turns a
  chord's worth of rail pieces together.

The lock is an **over-centre knee**, driven *harder* onto its stop by the load
it carries; the only way out is to push the knee back across, which is a motion
the load never applies. Its offset is bracketed, not picked: below the floor
the stop's pin brinells or the linkage's own slop swallows the lock, above the
ceiling a hand cannot break it. One draw rod down the spine's bore trips all
six from outside the truss.

| | 85 mm truss | 70 mm truss |
|---|---|---|
| fold | 80.2° | 84.5° |
| collapsed radius | 20.04 mm | 16.71 mm |
| axial swing | 34.9 mm | 28.1 mm |
| over-centre offset | 1.50 mm, in [1.50, 242] | 1.50 mm, in [1.50, 175] |
| draw stroke | 6.00 mm | 6.00 mm |
| **withdraws with** | **3.75 mm to spare** | **3.00 mm** |

`check_collapse` asserts the *trapped* state first: if the cage ever stops
being trapped when erect, the rest of the suite is measuring nothing. The rails
are in the approach solver's obstacle set — a structure the path planner cannot
see is how a fixture ends up unbuildable — and every joint still has a station.

Collapsing it is a manual step, outside the cycle.

### The cell could not pick up a camera, and could not be made to

Two findings from asking the machine rather than the geometry, both in
`check_mount`.

**The gripper cannot hold the module.** Its jaws open 8 mm, sized for a 3 mm
rod with clearance. The module is 11.3 mm through its thin way and its metal
enclosure is 10.8 mm square. The only thing on it inside 8 mm is the ⌀5.75
lens barrel, which is the one part a machine must not touch.

**And opening the jaws is not the fix.** The magazine's slot pitch is bounded
below by `JAW_OPEN/2` — a rod has to clear the pads that straddle its
neighbour — the diagonal racks pitch along x, and the gantry is nearly full.
Jaws wide enough for the enclosure put the racks at **1422 mm of a 1400 mm
axis**. The machine would need a longer X axis to be able to pick up a camera.

So the module is pressed into a printed **carrier** at kitting — the same
manual step as pre-cutting the rods — and the carrier presents a boss of the
*largest stock diameter*. To the loader it is then a chord: the jaws already
span it, the V already captures it, the rack pitch does not move. The boss
is **radial**, straight out of the camera's back, so it is 180° from the
optical axis and can never be in shot, and being radial rather than axial it
costs the cage no room at all. It grips the board's long edges, never the
enclosure — the enclosure is what the six struts bond to.

**And the camera did not fit inside the cage.** The mount goes on *in* the
cage: its rods are laid by the same gripper and bonded by the same
dispenser, and the cage is what presents their angles. `Cage.END_FREE` was
20 mm; the solved standoff puts the module's outer face 33.9 mm past the
chord ends on the chosen truss and 38.0 mm at the largest section the cell
is specified to build. The camera landed inside the end plate. It is now
40 mm, derived from the payload, the field of view and `SECTION_MAX` rather
than typed — and `check_mount` re-derives it, so it cannot drift away from
the camera it was sized for.

That room is not free. The end racks move out with the end plate, and the
chosen truss now wants **1399 mm of the gantry's 1400** — up from 1359. The
model says so rather than the cell finding out.

## The suites

450 checks in thirteen suites, about 12 minutes for the full tier.

| suite | tier | what it would have caught |
|---|---|---|
| `check_geometry` | 0 | a spec assertion that no longer holds; a truss whose derived quantities disagree |
| `check_structure` | 0 | the beam model against the brief; an optimiser that runs to the sweep's edge; a section the winder cannot enter; a truss whose racks the gantry cannot reach the ends of |
| `check_mount` | 0 | a camera the gripper cannot hold; a fix that costs more axis than the machine has; a module that lands in the end plate; a strut in shot; **a strut landing on nothing at all** |
| `check_approach` | 0 | a station with no margin; a sampling hole a rod fell through; a retracted rod in the rim; a post loop through the fixture |
| `check_schedule` | 0 | a rod released at the wrong angle; a yaw with the gripper in; a loop that touches |
| `check_model` | 1 | a V whose flanks stood proud; a head on the spine; a ring not the solver's |
| `check_hal` | 1 | a backend missing a verb; an axis that believes its own truth; process code that reads the simulator; a GL backend named where the name does not exist; a ring servo that chattered once the ring got light |
| `check_view` | 1 | a demo that opens on a fixed camera, where the mouse is dead; a camera key the viewer had already bound |
| `check_load` | 1 | a rod that seats 0.7 mm high; a capture range smaller than claimed; a rod that falls out; a fixed wait for an index that got longer |
| `check_cycle` | 1 | anything above that survives to the truss: bands off-centre, drops on the floor, rods pressed out |
| `check_drive` | 2 | a mesh of two different modules; a pinion phase that jams; a raceway that does not locate the ring; a rig whose "failure" case cannot fail |
| `check_ring` | 2 | a hoop model that is not what a thread does; thread mass off by a third; a band believed to retain what it does not |
| `check_vision` | 1 (slow) | a camera that photographs its own carriage; a look that is worse than the model says; a field too small for the joint; a sigma written as a number when it is a law |

When a check fails, suspect the check: three of the first vision failures
were the checks' cameras, not the code.

## Reading the code

```
spec.py        every fact: stock, rules, ring, gantry, head, gripper, cage, magazine, dispenser, vision, process
geometry.py    a Truss -> its rods, joints, clusters, faces, at any cage angle
structure.py   beam and member models; design(); the two default trusses
fixture.py     pins, cradles, posts, racks, orientations, pick/place poses, obstacles
approach.py    head solids as closed forms; obstacles as sampled capsules; stations, lifts, yaw height, post loop
drive.py       Tier 2: the ring as a free body in its raceway, involute teeth, one belted shaft
band.py        the band as arithmetic: pitch, thread, dose, drop size
schedule.py    the plan: ops with durations, from the stations
inspector.py   what was made, from the band states and the seats
hal.py         the contracts the process is written against
mjcf.py        the cell as MJCF, generated from the specs
cell.py        the MuJoCo backend of the HAL, with truth accessors for the checks
process.py     the plan executed through the HAL, one control tick per yield
vision.py      the look: a model camera and the rendered pixel path
view.py        the viewer's camera: pan, orbit, zoom, follow, on keys the viewer leaves alone
glenv.py       which GL backend to name, decided by the platform
rig.py         Tier 2: the one-joint lathe with a cable thread
```
