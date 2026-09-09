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
fit round the joint and the cage's spine must fit under the spool.  The
brief's own 45°, 3 mm/2 mm truss comes out with f₁ = 195 Hz and its
diagonals 1 mm short of the ring's rim; the optimiser's metre truss is
side 92, α 40°, 3/2 mm stock, 54.9 g, 27 joints, and the 300 mm test
piece is side 66, α 45°, 2/1 mm, 7.4 g, 12 joints, spine-bound.  The
brief's own 300 mm section, 40 mm across, is stiff enough several times
over and cannot be wound at all: the ring sweeps 32 mm of radius about the
chord it is on, leaving no room for the cage's spine or the other two
chords.  A short truss's section has a lower bound set by the winding
head, not by the loads.

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

**The fixture.**  Pins hold the chords in 90° V-notches at a computed
pitch, kept out of every joint's band sweep and off every chord's midpoint
(the gripper needs it); cradles hold the diagonals, pushed along them until
a seated head clears them by `SEAT_CLEAR`; the spine is as fat as the
spool's sweep allows.  A V is a kinematic locator: `check_load` releases a
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

**The camera.**  Where it rides is a rendered finding, three times over: on
the carriage over the ring plane it photographs its own carriage box and
the spool; outboard of the dispenser it sees the chord but the diagonals —
the mitre that gives the joint's x — hide behind the nearest pin's flanks;
in a bay beside the spool it sees both, the far diagonal through the
ring's gap, which the plan parks downward for the look.  The look is
always taken with the joint on the camera's side of the plate: through
the gap from the far side one diagonal is lost and the joint's x came back
up to 2 mm off.  The pixel path (`vision.PixelVision`) fits the chord's
centreline and both diagonals' lines near where the plan predicts them
and back-projects the difference; it is good to `LOOK_SIGMA` = 0.15 mm
along the chord — the lever arm of 1/sin α on the diagonals' lateral fit —
and hundredths across it.  The model camera draws its noise from that
measured figure, not from the sensor's 0.005 mm per edge.

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
under a constant tension — a spool on a 20 mm ring cannot carry a 50 mm
reserve round the cage.  MuJoCo's contacts scale with the mass they act
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

## The suites

374 checks in eleven suites, about 13 minutes for the full tier.

| suite | tier | what it would have caught |
|---|---|---|
| `check_geometry` | 0 | a spec assertion that no longer holds; a truss whose derived quantities disagree |
| `check_structure` | 0 | the beam model against the brief; an optimiser that runs to the sweep's edge; a section the winder cannot enter |
| `check_approach` | 0 | a station with no margin; a sampling hole a rod fell through; a retracted rod in the rim; a post loop through the fixture |
| `check_schedule` | 0 | a rod released at the wrong angle; a yaw with the gripper in; a loop that touches |
| `check_model` | 1 | a V whose flanks stood proud; a spool on the spine; a ring not the solver's |
| `check_hal` | 1 | a backend missing a verb; an axis that believes its own truth; process code that reads the simulator; a GL backend named where the name does not exist |
| `check_view` | 1 | a demo that opens on a fixed camera, where the mouse is dead; a camera key the viewer had already bound |
| `check_load` | 1 | a rod that seats 0.7 mm high; a capture range smaller than claimed; a rod that falls out |
| `check_cycle` | 1 | anything above that survives to the truss: bands off-centre, drops on the floor, rods pressed out |
| `check_ring` | 2 | a hoop model that is not what a thread does; thread mass off by a third; a band believed to retain what it does not |
| `check_vision` | 1 (slow) | a camera that photographs its own carriage; a look that is worse than the model says |

When a check fails, suspect the check: three of the first vision failures
were the checks' cameras, not the code.

## Reading the code

```
spec.py        every fact: stock, rules, ring, gantry, head, gripper, cage, magazine, dispenser, vision, process
geometry.py    a Truss -> its rods, joints, clusters, faces, at any cage angle
structure.py   beam and member models; design(); the two default trusses
fixture.py     pins, cradles, posts, racks, orientations, pick/place poses, obstacles
approach.py    head solids as closed forms; obstacles as sampled capsules; stations, lifts, yaw height, post loop
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
