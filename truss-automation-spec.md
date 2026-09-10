# Automated carbon fibre truss fabrication cell

## Design brief for robotics team

> **Revision 2 — the numbers below are now simulated, not estimated.**
>
> `sim/truss` models this cell: the structure, the fixture, the head and
> the process as arithmetic; the whole machine in MuJoCo; and one joint
> wound with a physical thread. Eleven check suites, 374 checks, run from
> `python3 sim/scripts/truss/check_all.py --slow`, and
> `sim/truss/README.md` is the design record.
>
> Where a figure in this brief was measured, it has been replaced and the
> measurement named. Six of them changed the design: the first mode of the
> §2.3 truss once the web's shear is counted, the 300 mm section (which
> the winding head cannot enter), the gap's parked orientation, the width
> of the band that actually binds, whether a dry band retains the mitres,
> and the yaw axis §4.3 says is not required.

---

## 1. Purpose

Build a machine that fabricates carbon fibre lattice trusses, 300 mm to 1000 mm long, with no manual handling of joints. The trusses are structural spines for long-range stereo camera heads.

Target output: one 1 m truss per 45 minutes of unattended machine time, at a repeatability that allows a single calibration procedure to be valid across units.

---

## 2. Why the structure matters (context for the team)

### 2.1 The product

A stereo depth camera with a **1 m baseline** — two global-shutter cameras 1 m apart, rigidly connected, feeding dense RGB-D and semantic segmentation for drone, USV and inspection autonomy.

Long baseline exists because stereo depth error grows with the square of range:

```
ΔZ = Z² · Δd / (f · B)
```

Where `Z` = range, `B` = baseline, `f` = focal length in pixels, `Δd` = disparity matching error.

At 1920×1200 with an 80° horizontal FOV (f ≈ 1145 px) and Δd = 0.25 px:

| Range | 1 m baseline | 240 mm baseline | 75 mm (typical COTS) |
|---|---|---|---|
| 10 m | ±2.2 cm | ±9.1 cm | ±29 cm |
| 25 m | ±13.6 cm | ±57 cm | ±1.8 m |
| 50 m | ±55 cm | ±2.3 m | ±7.3 m |
| 100 m | ±2.2 m | ±9.1 m | ±29 m |

No commercially available stereo camera exceeds a 250 mm baseline. The gap is not demand — it is that conventional structures become too heavy at that span to fly.

### 2.2 The constraint that drives everything

Stereo depth assumes the relative pose of the two cameras is known. That pose is established once, by calibration, and is thereafter assumed constant.

**A relative yaw error between the two cameras produces a disparity bias of `f · δθ`.**

At f = 1145 px:

- 0.005° of relative rotation → 0.1 px disparity bias
- At 100 m range, where true disparity is only 11.5 px, that 0.1 px is **±0.9 m of range error**
- At 0.02° the induced error reaches ±3.5 m and swamps the sensor's own noise floor

**Design budget: total angular drift between camera mounting faces must stay under 0.005° across all operating loads and temperatures.**

This budget is why the structure is the product. The baseline sets the accuracy ceiling; the structure determines whether that ceiling is ever reached.

Note that this cannot be fixed in software. Online extrinsic estimation (as used in VIO/SLAM stacks) can absorb slow drift, but relative yaw is weakly observable — it is mathematically near-indistinguishable from a global depth scale error — and no online method tracks flex occurring within a single exposure.

### 2.3 Why a truss and not a tube

Load case: camera head 50 g at each end, 500 mm cantilever from a central mount, 3 g lateral manoeuvre load. Tip slope must stay under 0.005°.

| 1 m spine option | E (GPa) | I (mm⁴) | EI (N·mm²) | Tip slope @ 3 g | Mass |
|---|---|---|---|---|---|
| 20 mm × 1 mm CF tube (roll-wrapped) | ~70 | 2,701 | 1.9 × 10⁸ | 0.056° — **fails 11×** | 90 g |
| 43 mm × 1 mm CF tube (hypothetical) | ~70 | 30,000 | 2.0 × 10⁹ | 0.0057° — marginal | 200 g |
| 40 × 2 mm aluminium tube | 69 | 43,000 | 3.0 × 10⁹ | 0.0038° | 645 g |
| **Warren truss, 90 mm chord triangle, 3 mm chords** | 150 | 28,600 | **4.3 × 10⁹** | **0.0025°** | **60 g** |
| **As built: 92 mm triangle, 40° diagonals** | 150 | 29,926 | 4.5 × 10⁹ | 0.0023° | 54.9 g |

The truss row is this brief's own hand calculation; the model reproduces
it to within 10% on I, EI, slope and mass (`check_structure`). Note that
90 mm there is the **side of the chord triangle**, not the chord-circle
diameter — 28,600 mm⁴ requires chords 52 mm from the neutral axis. The
row beneath it is what the optimiser returns when it is given the rules of
§6 and the envelope of §4 and left free (`structure.design`).

Three points for the design team:

1. **Roll-wrapped tube is the wrong carbon.** Fibres lie in woven cloth at ±45°, giving ~70 GPa axial. Pultruded rod has unidirectional fibres, ~150 GPa. Same material, roughly double the stiffness per gram.

2. **Second moment scales with the square of chord separation.** A tube places material 10 mm from the neutral axis; the truss places chords 45–52 mm out. That geometric leverage is where the performance comes from, not the material.

3. **Tube stock above 20 mm OD is not readily available in India at reasonable cost.** This constrains the tube option out of contention independently of the physics.

Secondary benefit: bonded joints provide structural damping. A monolithic tube rings.

**Corrected: 228 Hz is a bending-only figure.** A Warren truss carries a
real shear compliance in its web, and including it the 90 mm truss above
comes out at **195 Hz — below this brief's own 200 Hz rule in §6**. The
optimiser's answer (92 mm triangle, 40° diagonals) reaches 201 Hz at
54.9 g. Either way the mode sits inside propeller blade-pass excitation
(200–330 Hz for 5–7" props at 6,000–10,000 rpm), so **soft-mounting the
spine to the airframe is mandatory regardless**, and the individual
members must also be checked: a diagonal's own pinned-pinned mode is
384 Hz on the 1 m truss, and holding it above 330 Hz is what stops the
optimiser making the lattice coarser still.

---

## 3. The truss

### 3.1 Geometry

**Warren truss, triangular cross-section.**

- Three continuous chords at 120°, running the full length uncut
- Diagonal web members in a zigzag pattern between adjacent chord pairs
- **No vertical members.** This is a hard constraint — see §4.2

Geometry is **derived, not tabulated**: `structure.design()` sweeps the
triangle, the diagonal angle and the stock against the rules of §6 plus
two the machine imposes (the ring must fit round a joint; the cage's spine
must fit under the ring). These are its answers.

| Parameter | 300 mm truss | 1000 mm truss |
|---|---|---|
| Chord triangle side | 62 mm | 92 mm |
| Chord diameter | 2 mm pultruded | 3 mm pultruded |
| Diagonal diameter | 1 mm pultruded | 2 mm pultruded |
| Bay pitch | 86 mm | 106 mm |
| Diagonal angle to chord | 35° | 40° |
| Joints per truss | 9 | 27 |
| Mass | 6.7 g | 54.9 g |
| Tip slope at 3 g | 0.0010° | 0.0023° |
| First mode / member mode | 380 / 338 Hz | 201 / 384 Hz |
| Ring's bore margin at a joint | 2.28 mm | 0.12 mm |

**The 40 mm section of revision 1 cannot be built by this machine.** It is
stiff enough several times over — 0.0011°, 502 Hz — but the ring sweeps
20 mm of radius about the chord it winds, and a 40 mm triangle puts the
cage's spine and the other two chords inside that. A short truss's section
has a lower bound set by the winding head, not by the loads, and only a
model of the fabrication finds it (`check_structure`).

The 300 mm truss's numbers moved when the head did. Its section is bound by
the machine, not the loads, so the head's swept radius falling from 32 mm to
20 (the spool came off the rim, §4.5) let the optimiser take a smaller,
shallower truss: side 62 at 35° instead of 66 at 45°, three bays instead of
four, 9 joints instead of 12. The metre truss is load-bound and did not
move.

Chord stock: pultruded solid rod, 230 GPa fibre at ~65 % volume fraction, Tg 170 °C, 11.3 g/m for 3 mm. Axial CTE near zero.

### 3.2 The joint

**One joint type only.** Two mitred diagonal ends lying against a continuous chord, bound by a wound thread band saturated with epoxy.

- Diagonal ends cut at the diagonal angle, presenting a flat face of
  `d_diag / sin α` along the chord — **3.1 mm on the 1 m truss, 1.4 mm on
  the 300 mm**, not 6 mm; 6 mm would need a 4 mm diagonal
- Two diagonals meet at each joint, forming a shallow V against the chord
- 18 hoop turns of 0.15 mm Kevlar, laid as an 8 mm band
- Wound dry, under controlled tension
- Low-viscosity laminating epoxy applied after winding, wicked through by capillary action
- Target joint mass: **0.15 g**

Load paths: epoxy shear carries axial load along the chord; compression
bears directly into the chord; the thread is hoop reinforcement across the
bond.

**Two corrections from winding a joint with a real thread** (Tier 2,
`check_ring` — a cable with bending stiffness, wound under tension on a
spindle):

1. **Most of the band does not bind.** A diagonal touches its chord at the
   joint and nowhere else, standing off by `dx·tan α − d_diag/(2 cos α)` a
   distance `dx` along it. A hoop laid outside

   ```
   bind_half = d_diag / (2 sin α)  +  d_thread / tan α
   ```

   finds a gap and beds on the chord *underneath* the diagonals. Wound,
   the hoops sit at 1.19 mm radius where the members' convex hull reaches
   6.29. That is ±0.86 mm of the 8 mm band on the 300 mm truss and
   ±1.73 mm on the 1 m. The rest is thread on a chord: harmless, and it
   was overstating the thread mass by a third, which understated the resin
   dose by the same. Widening the band buys bond area, not grip.

2. **A dry band does not retain the mitres.** Released from the fixture
   and turned face down, the diagonals leave — wound (292 mm) or bare
   (281 mm). Nothing in the process may free a keeper before cure, which
   is now a checked invariant of the plan. The bake in §3.3 is therefore
   load-bearing in a second sense: it is the only thing that ever holds
   the joint.

What the thread does do is not disturb the placement: a wind moves a
clamped diagonal 0.119 mm, against 0.118 mm for the same rotation with no
thread in the machine at all.

**Do not thicken the resin.** Fillers block capillary wicking. If a joint tests weak, add a small fumed-silica fillet at the diagonal corner as a secondary operation — but test without first.

### 3.3 Cure

Bake the complete truss at **80 °C for 2 hours while still pinned in the fixture**.

This is not optional. Two things depend on it:

- The epoxy reaches full cure and a Tg above any operating condition
- Every joint locks while the thread is still under winding tension and the chords are held straight, so no residual stress remains to relax over subsequent weeks

Suggested system: Araldite LY556 with HY917 hardener and DY070 accelerator (~120 °C Tg after bake). Confirm schedule against the current datasheet. Room-temperature systems reaching only 70–80 °C Tg are marginal for a drone in direct sun.

---

## 4. Machine requirements

### 4.1 Governing principle

**Geometry is held by the fixture, not by the robot.**

The gantry has one head and the truss has 60 joints that must remain positioned for the duration of the cure. A pick-and-place approach would require either 60 grippers or 60 sequential cure cycles. Instead the fixture holds all rods permanently; the head visits joints one at a time and binds them.

Motorised axes are for *process*, not for *holding*. Steppers drift when warm and carry backlash; hardened pins hold ±0.02 mm indefinitely.

### 4.2 The enabling topology constraint

A ring-shaped winding head with a gap in its circumference can be lowered onto a **single** rod. It cannot encircle two crossing rods — any closed thread loop around a crossing is topologically linked with both members and cannot be closed without a hand-off.

The Warren geometry avoids this entirely:

- Chords are continuous and never crossed by anything
- Diagonals lie *against* the chord, not through it
- A diagonal stays close to the chord across the wrap zone — but "inside
  the ring's clearance envelope" is a computation, not an inspection. The
  bore must clear

  ```
  r_in ≥ d_c/2 + (W/2 + band/2)·tan α + d_d/(2 cos α) + clear/cos α
  ```

  where `W` is the ring plate's width and `band` its axial sweep, because
  the ring turns at **every** x across the band, and the last term is the
  rim's corner meeting an oblique diagonal at `cos α` of the radial slack.
  A 10 mm plate fits no truss once that is charged; the plate is **4 mm**.
  The brief's own §2.3 truss is 1 mm short of the rim on this rule.

**A vertical web member would sit at 90° to the chord, directly in the ring's plane of rotation, and block it.** This is the single reason vertical members are prohibited. Any geometry change proposed by the team must preserve this property.

### 4.3 Axes

| Axis | Function | Notes |
|---|---|---|
| **θ (cage rotation)** | Indexes the truss about its own axis to present each chord uppermost | 120° steps. NEMA 17 through worm gear — self-locking required so winding torque cannot back-drive it |
| **X** | Traverse along truss axis | **1,400 mm** — the fixture plus the diagonal racks span 1,326 mm for a 1 m truss, measured, not the 1,200 assumed |
| **Y** | Cross-axis positioning, and reaching the magazines | **±150 mm** — visual centring needs ±2 mm, but the chord rack sits ±134 mm off the axis |
| **Z** | Lowers ring onto joint, lifts clear | 150 mm travel is ample: the cage indexes under the head at 95 mm and the lift between joints is **13 mm**, the least that clears the next pin |
| **W (gripper yaw)** | Turns a rod from the rack's orientation to its placing angle | **±95°; see below** |
| **Ring rotation** | Continuous winding | Toothed rim, three phased pinions on one belt, see §4.5 |

**A yaw axis IS required, once rod loading is automated.** For winding the
brief is right: chords are parallel to the axis and the ring plane never
turns. But the diagonals lie at ±α, they are fetched from a rack that
holds them all parallel, and something has to turn them. A yaw servo on
the gripper is the cheapest place to put that rotation; ±95° covers a rack
rod at 90° and a placing angle of ±55°.

It carries a constraint that is not obvious and cost a rebuild to find:
**the rod may only be turned with the gripper extended.** Retracted, a
carried rod sits beside the ring's plate, and turning it there sweeps its
ends through the rim — the clearance is −0.5 mm at ±α, and in simulation
the wrist stalled at 37° of its 135° and the rod slipped in the pads.
Extended below the rim it clears by 16.5 mm. The height at which to do
this is computed per rod, and comes out a few millimetres above the seat.

Recommended for X/Y/Z: NEMA 17 direct-coupled to SFU1204 ballscrew on MGN12 linear rail. Avoid belt reduction — belt compliance sits exactly where positional confidence is needed. Repeatability target ±0.05 mm is ample; do not chase microstepping resolution, as thermal growth of a steel screw (12 ppm/K, ≈36 µm over 300 mm at 10 °C) exceeds it.

### 4.4 Fixture / cage

- Holds three chords at 120° on locating pins at **50 mm intervals** (chord straightness becomes the dominant geometric error once joints are stiff — do not use 100 mm spacing at 1 m). Pin positions are computed, not spaced blindly: a pin may not stand inside any joint's band sweep, and it must keep clear of each chord's midpoint, which is where the loader grips it
- Small cradles at each joint hold the two mitred diagonal ends seated against the chord. Their position along the diagonal is computed by pushing them outward until a seated ring clears them
- **A V-notch is the locator, and it is what delivers the ±0.1 mm.** A rod released anywhere in the V's mouth rolls to its apex line: measured, a chord released a millimetre above its notches seats within 0.001 mm, and the capture range is 2.2 mm against a closed form of 1.5
- **Every rod needs a keeper, and it must hold through the bake.** With the cage turned face down a free diagonal leaves its cradles entirely (189 mm). The force is only the rod's own weight, about a millinewton, and winding does not add to it — but nothing may release before the resin cures
- Rotates as an assembly on the θ axis
- **Must survive the 80 °C bake with the truss in situ** — material selection and differential expansion need attention here
- One fixture per truss geometry; changing baseline means a new fixture, which should be machinable in an afternoon

### 4.5 Ring winding head

The critical subsystem. Everything else is conventional motion control.

- C-ring, 40 mm outer diameter and 20 mm bore, with a **60° gap** in its circumference, on a **4 mm** plate (§4.2 — a 10 mm plate fits no truss the loads would choose)
- **Corrected: the ring is its own spool.** Revision 1 hung a spool block on the rim and paid 32 mm of swept radius for it — the number that set how fat the cage's spine could be. A whole metre truss is 7.8 m of 0.15 mm thread: 139 mm³, less than one epoxy drop. It winds into a **groove in the ring's own web** (mean radius 15 mm, 2 × 2 mm, 60 % packing → 10.7 m), the way a toroidal winder's shuttle carries its wire. Nothing protrudes, the rim is left for the drive, the swept radius is the rim itself, and the cage's spine grows from 3 mm to 8
- **Corrected: driven on its teeth, not by friction.** Two friction wheels 90° apart were three faults at once: the orbiting spool struck each wheel once a revolution; two wheels are a drive and not a bearing, and nothing else located the ring; and the thread alone asks 0.022 N·m against a slip torque of 0.03 that was a guess — friction slips silently, which is why the turns had to be counted by watching a fiducial go past
- **The tooth counts are solved, not chosen** (`Ring.mesh()`). The first toothed draft wrote 72 teeth on the rim and a 12-tooth pinion of 6 mm pitch radius: module 0.556 against module 1.0 — two gears that cannot mesh, and nothing caught it, because a tooth count can only ever be wrong silently. The solver takes the rim's tip circle as given (nothing may stand proud of the swept radius), requires a whole number of teeth on the rim **and in the gap** so the far side arrives in phase, takes the smallest pinion that neither undercuts (2/sin²α teeth) nor swallows its own bearing, checks Lewis at the root with a factor of 3, refuses any module whose pinion would hang below the rim, and of the survivors takes the coarsest. Answer: **module 0.8, 48 teeth on the rim, 8 of them in the gap, an 18-tooth pinion, 1:2.67, 6.0 MPa in the root**
- **Three pinions on one belt, phased** (`Ring.pinion_az()`, `drive.pinion_phase()`). Placed by maximising the smaller of two margins in degrees — clear of the mouth, and further apart than a gap plus two contact arcs, so one gap cannot unmesh two: **66°/180°/294°, 18.6° in hand**. One motor and a belt, not three servos: three independently commanded shafts on a closed loop are over-constrained, and the rig jammed with all three saturated and the ring turning backwards
- **The turns are known at the motor.** A toothed rim cannot slip, so the ring's angle is an encoder read on the pinion shaft and the only error left is quantisation: 4000 counts through 1:2.67 is **0.034° at the ring**. The fiducial is gone
- **Corrected: the raceway's mouth is 60°, not 100°, and it is solved** (`Ring.race_mouth()`). It was sized to contain the ring's 60° gap. That is not the requirement: only the chord ever reaches the rail's radius, and only at one azimuth — at the ring's own x-window the diagonals are inside 5 mm of the chord's axis and never see the rail at 23 mm. What does pull the other way is depth, because the rail's lowest point sits at `rail·cos(mouth/2)` and a narrow mouth hangs it below the rim the head has to get down past. So the mouth is the narrowest that still costs no envelope, `2·acos(rim/rail)` = **59.6°**. §7 records what the 100° version did
- **Corrected: it must stop with the gap toward the joint's face, not straight down.** "Gap down" is the right idea and the wrong number. The gap has to admit the whole cluster, and the two mitred ends lie against the chord along the face normal, 30° off vertical at a chord-up joint — exactly the edge of a 60° gap. Measured at the band's ends, where the descent is tightest: parked toward the face the ring clears by 1.9 mm, parked straight down by 0.6 mm. The angle is solved per joint and alternates in sign as the joints alternate faces

**What the drive is measured against** (`check_drive`, Tier 2): the ring is
built as a **free body** — no hinge, no weld, six degrees of freedom — held
by nothing but the C-channel raceway and three pinions with real involute
teeth, and turned against the thread. One revolution: the ring arrives where
the shaft says to **0.30° of a 7.50° tooth**, crossing a pinion with the gap
costs **0.04°**, at least two pinions are engaged throughout, the steady
demand is **0.028 N·m of a 0.120 rating** against a thread's own 0.022, and
the ring's centre moves **0.072 mm**. Still true at RPM_MAX and twice the
tension. Press one pinion on a quarter of a pitch out and it jams.

Prior art: this is an orbital cable-wrapping head, scaled down. The mechanism is proven; the engineering is in miniaturisation, tension control and the fact that the drive has to survive the gap going past it.

**Thread strategy:** do not cut between joints. Run one continuous thread per chord — tight bands at each joint, a single straight strand between. This reduces anchoring operations from 120 to 6 and the inter-joint strand is negligible in mass.

### 4.6 Epoxy dispensing

**Corrected: dose each joint as it is wound, not in a separate pass.** The
dispenser rides on its own stroke beside the ring, so there is no tool to
swap, and the head is already over the joint. Interleaving saves 1.4 min
on the 1 m truss and removes a whole traverse of the truss.

- Syringe beside the ring on its own Z stroke, stepper-driven plunger
- One metered drop per joint, sized from the joint's mass budget less the thread actually on it
- **The nozzle must stand off by two drop radii plus the fall, not by a nominal millimetre.** The drop hangs from the tip before it lets go; at a 1 mm standoff it is born inside the chord. Measured before this was fixed: seven of twelve drops missed the joint
- Low-viscosity resin only — it must wick through the thread by capillary action. Visual confirmation: the wrap turns translucent
- **Do not use a static mixing nozzle in a continuous-dispense configuration.** Two-part epoxy cures inside the nozzle. Either use a pre-mixed batch with adequate pot life, or plan for nozzle replacement every 20 minutes

### 4.7 Process sequence

Load first, in six cage indexes: chords with their chord uppermost, then
diagonals with each face uppermost. For each rod — fetch from the rack,
carry at the index height, descend to the computed turning height, extend,
yaw to the placing angle, lower to a millimetre above the seat, release
into the V, engage the keeper.

Then for each chord:

1. θ indexes so the target chord is uppermost
2. The head loops the thread around the anchor post at that end (the loop's height and size are solved against the fixture — drawn by eye, it hooked the ring on the post and jammed the X axis)
3. Vision locates the actual joint (§5), and the head trims onto it
4. The ring parks its gap at the solved angle; Z lowers it onto the chord
5. Ring rotates 18 turns at controlled tension while X feeds one band width
6. Ring re-parks; Z lifts 13 mm clear; the dispenser doses the joint it has just wound
7. X traverses to the next joint; repeat
8. At the last joint, loop the far post and cut

Then θ indexes 120° and the next chord is wound, running back the other
way. Three thread runs total. Transfer the fixture with the truss to the
oven and bake — **the keepers stay engaged throughout** (§3.2).

Simulated end to end, the 300 mm truss takes 9.4 minutes against 9.3
planned; the 1 m truss plans at 24.1 minutes, of which 11.0 is loading,
11.4 winding and 1.6 dosing.

---

## 5. Vision subsystem

Existing hardware available: multiple global-shutter stereo rigs (16 mm, 24 mm baselines), high-resolution stereo pair, Jetson NX and AGX Xavier.

Three required functions:

**Pre-wind joint localisation.** The fixture will not be perfect to ±1 mm at 1 m. The head must visually servo the ring onto the actual chord position before lowering. This is the function that makes fixture tolerance a non-issue and should be prioritised.

*Where the camera goes is a result, not a mounting detail.* Rendered from
three candidate positions: on the carriage over the ring's plane it
photographs its own carriage box and the spool; outboard of the dispenser
it sees the chord, but the nearest pin's flanks hide the diagonals — and
the diagonals are what give the joint its position **along** the chord.
It belongs in a bay between the ring's envelope and the dispenser, above
the spool's sweep, looking in at about 53 mm range. From there both
diagonals are in view, the far one through the ring's own gap, which the
plan has parked downward anyway.

Two further findings. The look must be taken with the joint on the
camera's side of the ring's plate: sighting past the plate from the far
side loses one diagonal and the joint's x came back 2.0 mm out, against
0.14 mm from the near side. And the measurement is a fit of the chord's
centreline and both diagonals' lines against where the plan predicts them,
which is good to **0.15 mm along the chord and hundredths across it** —
the along-chord figure carries a `1/sin α` lever arm from the diagonals'
fit. The sensor is not the limit: it resolves an edge to 0.005 mm. The
camera-to-ring bracket is, at 0.10 mm, and it is drawn once per run and
never averaged away.

**In-process verification.** Camera on the head observing the thread as it lands. Turn count from a fiducial on the ring; tension inferred from thread sag between spool and chord. Sag indicates tensioner slip — halt and flag.

**Post-cure QC.** Stereo measurement of all three chords along their full length for straightness and twist. This produces a per-unit quality number that ships with the product and will be the first thing a serious customer asks for.

**On reinforcement learning:** not recommended for the core sequence. Lower, wind, lift, traverse is fully deterministic with no hidden state — a lookup table is sufficient and more debuggable. The only contact-rich subproblem that might justify learning is thread anchoring and lift-off handling (§7).

---

## 6. Acceptance criteria

| Test | Requirement |
|---|---|
| Single joint pull-out | Withstand 50 N axial on the diagonal without debond |
| Joint mass | ≤ 0.20 g measured |
| Truss mass, 1 m | ≤ 70 g |
| Tip slope under 1.5 N applied at 500 mm cantilever | ≤ 0.005° |
| First bending mode, 50 g tip masses | ≥ 200 Hz |
| Chord straightness after cure | ≤ 0.5 mm deviation over 1 m |
| Thermal drift, camera-face to camera-face, 20 → 50 °C soak | ≤ 0.005° |
| Cycle time, 1 m truss, unattended | ≤ 45 min |
| Unit-to-unit repeatability, tip slope | ≤ 10 % spread across 10 units |

Simulated against these, on the derived geometries: 1 m truss mass 54.9 g,
tip slope 0.0023°, first mode 201 Hz, cycle 24.1 min; 300 mm truss 7.4 g,
0.0009°, 377 Hz, 9.4 min. Placement into the fixture seats every rod
within 0.005 mm of nominal, and every band lands centred within 0.16 mm.
The four criteria a simulation cannot reach — pull-out, chord straightness
after cure, thermal drift and unit-to-unit spread — are exactly the four
that need the bench, and they are the ones §8 builds first.

The thermal drift figure is the one that determines whether the product works. It should be measured using the stereo cameras themselves: calibrate, soak, cool, recalibrate, diff the extrinsics.

---

## 7. Known open problems

Flagged honestly rather than assumed solved:

1. **Thread anchoring at the start of each chord run, and termination at the end.** Six anchor and six cut operations per truss. The path is now specified — a rectangular loop around a post at each chord end, at a height and size solved against the fixture — but what actually holds the thread there (a CA tack, a clip, a heated cutter) is untested and remains the least-specified part of the design.

2. **Lift-off with a trailing strand.** The tensioner must pay out freely as the head traverses between joints without slack accumulating. Unmodelled: the Tier 2 rig turns the work under a fixed guide, so it never traverses.

3. **Fixture thermal behaviour during bake.** Differential expansion between fixture and truss over 60 °C could induce exactly the residual stress the bake is meant to eliminate.

4. ~~**Rod loading into the fixture.**~~ **Now in scope and modelled.** Chords lie in a rack beside the cage, diagonals in a rack at each end of the truss with a gripper turning each one to its placing angle. Simulated, every rod of the 300 mm truss is fetched, carried, turned and seated within 0.005 mm, in the time the plan allows. What is not answered: the rods must be pre-cut to length by hand, and nothing yet verifies a rod was picked up.

5. **Chord straightness verification before winding** — a bowed chord bonded straight will spring back after release.

6. **The band binds over a fraction of its width** (§3.2). Whether 18 turns of which ~4 grip is the right recipe is a question for the pull-out test, not the simulation. If grip is what the joint needs, a narrower band at the same turn count is the change to try first.

7. **Nothing in the model is a strength test.** The geometry, the reach, the clearances and the clock are simulated; the bond is not. §8 stands unchanged.

8. **The drive torque and the rail's friction are still `[VERIFY]`.** `Ring.DRIVE_TORQUE` (0.12 N·m at the ring, a NEMA 8 through 4:1) and `Ring.RACE_MU` (0.15, a dry rail) are the two numbers `check_drive` measures a demand *against* rather than measures. The demand it does measure is 0.028 N·m steady, so there is room for both to be a good deal worse than assumed — but the raceway's own friction coefficient is the term the wedge multiplies, and a real rail on a film is nearer 0.05 than 0.15.

9. **THE RING'S BORE IS NOW THE BINDING CONSTRAINT ON THE PRODUCT'S ACCURACY, and this is the open problem with the most at stake.** The bore a joint needs scales as `tan α`, so the winding head limits the **web angle** — and the web angle is what buys stereo accuracy. Swept over sides 80–180 mm and webs 35–55°, **not one design meets mass, first mode, the yaw budget and the bore together**; every design that holds the budget wants a 45° web or steeper, and none of those fits the 20 mm bore. Relaxing the bore by 1.5 mm (`Ring.ID` 20.0 → 21.5) unlocks 28 of them, the best being `warren 120/45/3.0/1.5` at 45.7 g, 0.88 of the budget and 0.77 m of range error at 100 m. Relaxing anything else unlocks nothing, except the first mode, which is not ours to trade away. What opening the bore costs is a re-check of the head: `EXIT_R` sits at 11.0 mm and would have 0.25 mm over a 10.75 mm bore, so the exit guide moves with it. **This trades the machine's head against the product's accuracy and it is a decision, not a derivation.** Until it is made, the chosen design (§3.1) sits at 0.94 of the budget and misses the 200 Hz mode by 1 Hz. `sweep.run` now takes the bore as a fourth constraint and `check_spine` asserts it is carried — without it the sweep named the best of a 28-strong field that the cell cannot build, with a straight face.

10. **The ring's run-out is a raceway property nothing has built yet.** `Ring.RUN_OUT` = 0.08 mm is what `check_drive` measures with three pinions and a C-channel, and the spec charges it against both the ring's bore (which leaves the metre truss 0.12 mm) and its swept radius. It is a *calibration*, in the sense of §CLAUDE.md: a constant with a suite that re-measures it. A real raceway with a real bearing surface may do better or worse, and the metre truss's bore margin is thin enough that it matters which.

11. ~~**The ring's bore binds the product's accuracy.**~~ **No longer true, and item 9 above is kept only as a record of how it read before the camera was weighed.** With a 4 g head the budget is met at an 85 mm section with +0.37 mm of bore to spare. What binds now is `MEMBER_F_MIN` — see §7.2.

12. **The flat flex is an unmodelled load path into the camera.** A photograph of the part shows what no drawing does: a wide, stiff ribbon leaving the board's edge, and the board is the thing whose pose the whole structure exists to hold. A taut cable pulls; a cable clamped to the airframe pulls with every manoeuvre and every degree of temperature. Nothing in `sim/spine` carries it — the head is a point mass and an inertia — and `Payload.HEAD_EXTRA` charges its *mass* but not its *stiffness*. The mitigation is standard and cheap (a service loop, with the strain relief anchored to the truss's own end batten rather than to the camera or to the airframe), but it is a requirement on the harness that nobody has written down, and it belongs with the joint pull-out test in §8 rather than in the model.

### 7.1 What the drive review changed, and what it cost

Recorded because the mistake generalises. The head's drive was reviewed
after a reader asked the obvious question — *where does the motor shaft go,
if the rod is on the axis?* — and every step of the answer had already been
written down wrongly:

| written | actual | how it was found |
|---|---|---|
| two friction wheels on the rim | the spool orbited on the same rim and struck each wheel once a revolution | reading the two envelopes against each other; a head is not checked against itself |
| a slip torque of 0.03 N·m | the thread alone asks 0.022, and friction slips silently | arithmetic |
| a split gate to close the gap for winding | there is nowhere for a gate to park: aside it scrapes the chord, radial it fouls the pin arm | the clearance solver refused it |
| 72 rim teeth, a 12-tooth pinion at 6 mm | module 0.556 against module 1.0 — the drive could not have turned | sizing the rig; nothing else would ever have said so |
| three velocity servos, one per pinion | a closed loop over-constrained: all three saturated, the ring turned backwards | the rig |
| pinions at 110/205/300 | 48° of mouth margin and 11 of spacing; the solver's 66/180/294 has 18.6 of each | the solver |
| a raceway mouth of 100° | 160° of dead arc; the ring wedged and the drive jammed at every clearance under 0.12 mm | the rig, and only the rig |

The last one is the one worth remembering. The 100° mouth was not a typo
or a rounding — it was a **correct deduction from a false premise** (that
the raceway's mouth must contain the ring's gap). It cost nothing visible:
the design still turned, because the ring wandered far enough to be caught
by a pinion instead of the rail. It would have been built, and it would
have needed a deliberately sloppy raceway to work at all, and nobody would
have known why.

### 7.2 What the camera's own datasheet changed, and what it cost

The same shape of mistake, one layer up. The load case in §2.3 is *camera
head 50 g at each end* and every structural conclusion in this document
descends from it. Nobody had weighed one. A Raspberry Pi Camera Module 3 —
the part this rig is built around — is **4 g** on the vendor's own
comparison table, and the modules it supersedes are 3 g. The tip mass was
an order of magnitude out, and it sets the tip force, dominates the
Rayleigh mass, and therefore fixes both the angular budget and the first
mode.

Re-swept with the real head, the mount modelled, and every rule the cell
and the members impose applied together (1470 candidates, 60–160 mm
sections, 30–60° webs):

| said | is |
|---|---|
| the 0.005° budget cannot be met inside a 100 mm section | met at **85 mm** |
| the winding head's **bore** is the binding constraint on accuracy | the chosen design clears it by +0.37 mm |
| the design is **section-limited** | it is limited by the **diagonals' own first mode** |
| 0 designs meet mass, f1, budget and bore together | **7** break no rule at all |
| the answer is `115/40/3.0/1.5`, 0.82 m at 100 m | `85/40/3.0/1.5`, **0.56 m** |

What binds now is `MEMBER_F_MIN`: 831 of the 1062 candidates that hold the
accuracy budget are refused because a 1.0 mm web rings inside the
propellers' 200–330 Hz band. Give up that one rule and the sweep reaches
0.358 m at 46.5 g — 36% truer and six grams lighter. That rule is about
fatigue at a bonded joint, not about the cameras, and **nobody has pulled a
joint with a 1.0 mm diagonal at blade-pass.** It is now the most valuable
bench test in the programme, worth more than any remaining geometric choice.

Three second-order things fell out of the same re-run, each of which had
been silently wrong:

- **The mount was a placeholder, and a placeholder is not a bound.** A
  massless rigid bracket charges nothing for the mount's own mass and
  compliance, and hangs the payload on the chord ends 66 mm off the axis.
  Between a 50 g head and a 4 g one its error changes *sign*. No margin
  could have covered both.
- **The web angle in a design's name is a request, not a fact.** The bay
  pitch rounds to a whole number of bays, so at a 100 mm section 35° and
  40° are the same truss, built at 38.66°. The bore rule scales as `tan α`
  and was being asked about the requested angle: 40 candidates passed it
  that the head cannot actually enter.
- **The loader's reach was not a design rule and should always have been.**
  A slenderer truss with a finer web needs *longer* diagonal racks, so the
  gantry's X travel is not monotonic in anything the structural rules see.
  With the light head the optimiser's first answer was a 58 mm section
  needing 1425 mm of a 1400 mm axis.

### 7.3 The camera's own datasheet, part two: the lens moves

`RP-009992-DS-1` (the Module 3 sensor assembly) confirms every mechanical
number that module's drawing gives — 10.8 ±0.15 square, 3.875 ±0.15 body
height, a ⌀5.75 barrel. Then it says the thing neither the brief nor the
mechanical drawing does.

**Module 3 focuses with an open-loop voice coil, and its tolerances are
larger than the whole structural budget.** In the units §2.2 writes the
budget in — range error at 100 m on a 1 m baseline:

| term | what it is | at 100 m |
|---|---|---|
| **the entire inter-camera budget** | 0.005° relative yaw | **0.84 m** |
| AF postural difference ±50 µm | lens shift with **orientation**, at fixed drive | **1.06 m** |
| AF hysteresis 8 µm | | 0.17 m |
| AF dynamic tilt 8′ | lens axis against the sensor, over the stroke | 1.7 px of principal point per mm of lever — **5–14 m** |

An axial lens shift changes the image distance, which is what a calibration
measures as the focal length. Stereo reads `Z = f B / d`, so a fractional
error in `f` is the same fractional error in `Z`, and unlike a relative yaw
it does **not** cancel between the two cameras: both focal lengths enter the
same way. This is an *intrinsic*, and no amount of structural rigidity
touches it.

**So the product uses Camera Module 2, which has no lens actuator at all.**
Its focus is set on a thread and then stays where it is put. That is the
decision this section exists to record, and it is a swap of parts rather
than a mitigation: locking a voice coil by holding its drive current does
not remove the postural term, because the postural term *is* what the coil
does at a fixed current.

**What the swap costs, and it is not nothing.** Module 2 has a shorter focal
length on bigger pixels — 3.04 mm and 1.12 µm against 4.74 and 1.4 — so a
pixel subtends 0.368 mrad instead of 0.295. Depth precision is proportional
to that, so the stochastic floor goes from 0.28 m to 0.35 m at 100 m. Seven
centimetres given up to remove a metre and six: a fifteen-to-one trade, and
`check_geometry` asserts the arithmetic rather than the conclusion.

Module 2 also brings a **plastic** lens housing where Module 3 had a
load-bearing metal can, which is the surface the six mount struts bond to.
It is not the soft part — a stubby block in the weakest plastic it could be
is 2.7× the axial stiffness of the 1.5 mm struts it carries — and
`check_mount` asserts the modulus at which that would flip (1.1 GPa, below
any engineering thermoplastic) rather than a margin somebody chose.

Recorded here because it is the same failure as §7.1 and §7.2 one layer
further out: a number nobody had looked up, quietly setting the answer.
`Module2` and `Module3` both live in `spec.py` — the rejected part kept
beside the chosen one, because the next person to compare 11.9 megapixels
against 8 will want to know what it cost.

---

## 8. Recommended build order

Do not build the gantry first.

1. **Bench-mounted ring head** over a single fixed chord with two mitred diagonals in a printed cradle. Wind by machine, apply resin by hand, bake, destructively test.
2. Only if the joint passes §6, proceed to the rotating cage and fixture.
3. Then the XYZ gantry.
4. Then vision servo.
5. Dispensing last.

The joint recipe — turn count, tension, wrap length — cannot be derived analytically. Establish it empirically with a ~20-sample design of experiments against a load tester before writing any winding program. The simulation deliberately treats the recipe as an input: it says where the thread goes and how much of it, never how strong the result is. Everything downstream depends on reproducing a known-good joint, and a machine that lays a beautiful wrap that does not hold is the most expensive possible failure mode.
