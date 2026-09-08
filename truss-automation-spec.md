# Automated carbon fibre truss fabrication cell

## Design brief for robotics team

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
| **Warren truss, 90 mm depth, 3 mm chords** | 150 | 28,600 | **4.3 × 10⁹** | **0.0025°** | **60 g** |

Three points for the design team:

1. **Roll-wrapped tube is the wrong carbon.** Fibres lie in woven cloth at ±45°, giving ~70 GPa axial. Pultruded rod has unidirectional fibres, ~150 GPa. Same material, roughly double the stiffness per gram.

2. **Second moment scales with the square of chord separation.** A tube places material 10 mm from the neutral axis; the truss places chords 45–52 mm out. That geometric leverage is where the performance comes from, not the material.

3. **Tube stock above 20 mm OD is not readily available in India at reasonable cost.** This constrains the tube option out of contention independently of the physics.

Secondary benefit: ~60 bonded joints provide structural damping. A monolithic tube rings. First bending mode is approximately 228 Hz for the truss versus 157 Hz for an equivalent tube — both fall within propeller blade-pass excitation (200–330 Hz for 5–7" props at 6,000–10,000 rpm), so **soft-mounting the spine to the airframe is mandatory regardless**. This is a system requirement, not a truss requirement, but the team should know it.

---

## 3. The truss

### 3.1 Geometry

**Warren truss, triangular cross-section.**

- Three continuous chords at 120°, running the full length uncut
- Diagonal web members in a zigzag pattern between adjacent chord pairs
- **No vertical members.** This is a hard constraint — see §4.2

| Parameter | 300 mm truss | 1000 mm truss |
|---|---|---|
| Cross-section depth (chord circle diameter) | 40 mm | 90 mm |
| Chord diameter | 3 mm pultruded | 3 mm pultruded |
| Diagonal diameter | 2 mm pultruded | 2 mm pultruded |
| Bay pitch | ~62 mm | ~100 mm |
| Diagonal angle to chord | 45° | 45° |
| Joints per truss | ~18 | ~60 |
| Target mass | 19 g | 60 g |

Chord stock: pultruded solid rod, 230 GPa fibre at ~65 % volume fraction, Tg 170 °C, 11.3 g/m for 3 mm. Axial CTE near zero.

### 3.2 The joint

**One joint type only.** Two mitred diagonal ends lying against a continuous chord, bound by a wound thread band saturated with epoxy.

- Diagonal ends cut at 45°, presenting a ~6 mm flat face along the chord (not point contact)
- Two diagonals meet at each joint, forming a shallow V against the chord
- 15–20 hoop turns of 0.1–0.2 mm Kevlar or fine glass thread, laid as an 8 mm band
- Wound dry, under controlled tension
- Low-viscosity laminating epoxy applied after winding, wicked through by capillary action
- Target joint mass: **0.15 g**

Load paths: hoop tension resists diagonal pull-out; epoxy shear carries axial load along the chord; compression bears directly into the chord.

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
- At 45°, a diagonal stays within 5 mm of the chord across the 6 mm wrap zone — inside the ring's clearance envelope

**A vertical web member would sit at 90° to the chord, directly in the ring's plane of rotation, and block it.** This is the single reason vertical members are prohibited. Any geometry change proposed by the team must preserve this property.

### 4.3 Axes

| Axis | Function | Notes |
|---|---|---|
| **θ (cage rotation)** | Indexes the truss about its own axis to present each chord uppermost | 120° steps. NEMA 17 through worm gear — self-locking required so winding torque cannot back-drive it |
| **X** | Traverse along truss axis | 1,200 mm usable travel (400 mm clearance beyond max truss length) |
| **Y** | Cross-axis positioning | ±50 mm sufficient; used for visual centring on the chord |
| **Z** | Lowers ring onto joint, lifts clear | 150 mm travel |
| **Ring rotation** | Continuous winding | Friction-driven, see §4.5 |

No yaw axis is required. All chords are parallel to the truss axis, so the ring plane is fixed for the entire job. This is a significant simplification and should be preserved.

Recommended for X/Y/Z: NEMA 17 direct-coupled to SFU1204 ballscrew on MGN12 linear rail. Avoid belt reduction — belt compliance sits exactly where positional confidence is needed. Repeatability target ±0.05 mm is ample; do not chase microstepping resolution, as thermal growth of a steel screw (12 ppm/K, ≈36 µm over 300 mm at 10 °C) exceeds it.

### 4.4 Fixture / cage

- Holds three chords at 120° on locating pins at **50 mm intervals** (chord straightness becomes the dominant geometric error once joints are stiff — do not use 100 mm spacing at 1 m)
- Small cradles at each joint hold the two mitred diagonal ends seated against the chord
- Rotates as an assembly on the θ axis
- **Must survive the 80 °C bake with the truss in situ** — material selection and differential expansion need attention here
- One fixture per truss geometry; changing baseline means a new fixture, which should be machinable in an afternoon

### 4.5 Ring winding head

The critical subsystem. Everything else is conventional motion control.

- C-ring, approximately 40 mm outer diameter, with a **60° gap** in its circumference
- Carries a thread spool and a passive tensioner (felt pad and spring is the standard solution)
- Driven by **two friction wheels 90° apart** on the ring's outer edge, so one is always engaged when the gap passes the other
- Must stop with the gap oriented downward, so the head can lift straight up clear of the diagonals (which lie in the horizontal plane)

Prior art: this is an orbital cable-wrapping head, scaled down. The mechanism is proven; the engineering is in miniaturisation and tension control.

**Thread strategy:** do not cut between joints. Run one continuous thread per chord — tight bands at each joint, a single straight strand between. This reduces anchoring operations from 120 to 6 and the inter-joint strand is negligible in mass.

### 4.6 Epoxy dispensing

Separate operation, after all winding is complete.

- Syringe on the Z axis, stepper-driven plunger
- One metered drop per joint
- Low-viscosity resin only — it must wick through the thread by capillary action. Visual confirmation: the wrap turns translucent
- **Do not use a static mixing nozzle in a continuous-dispense configuration.** Two-part epoxy cures inside the nozzle. Either use a pre-mixed batch with adequate pot life, or plan for nozzle replacement every 20 minutes

### 4.7 Process sequence

For each chord:

1. θ indexes so the target chord is uppermost
2. Vision locates the actual joint position (see §5)
3. Z lowers the ring onto the chord through the gap
4. Ring rotates 15–20 turns at controlled tension
5. Ring stops with gap down; Z lifts clear
6. X traverses to next joint; repeat

Then θ indexes 120° and the next chord is wound. Three thread runs total.

After all winding: swap to dispenser, apply resin to all joints, transfer fixture with truss to oven, bake.

---

## 5. Vision subsystem

Existing hardware available: multiple global-shutter stereo rigs (16 mm, 24 mm baselines), high-resolution stereo pair, Jetson NX and AGX Xavier.

Three required functions:

**Pre-wind joint localisation.** The fixture will not be perfect to ±1 mm at 1 m. The head must visually servo the ring onto the actual chord position before lowering. This is the function that makes fixture tolerance a non-issue and should be prioritised.

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

The thermal drift figure is the one that determines whether the product works. It should be measured using the stereo cameras themselves: calibrate, soak, cool, recalibrate, diff the extrinsics.

---

## 7. Known open problems

Flagged honestly rather than assumed solved:

1. **Thread anchoring at the start of each chord run, and termination at the end.** Six anchor and six cut operations per truss. Options include a CA tack, a small clip, or a heated cutter. This is the least-specified part of the design.

2. **Lift-off with a trailing strand.** The tensioner must pay out freely as the head traverses between joints without slack accumulating.

3. **Fixture thermal behaviour during bake.** Differential expansion between fixture and truss over 60 °C could induce exactly the residual stress the bake is meant to eliminate.

4. **Rod loading into the fixture.** Currently assumed to be a manual or semi-automatic operation at the start of each cycle. Full automation of this step is a later phase.

5. **Chord straightness verification before winding** — a bowed chord bonded straight will spring back after release.

---

## 8. Recommended build order

Do not build the gantry first.

1. **Bench-mounted ring head** over a single fixed chord with two mitred diagonals in a printed cradle. Wind by machine, apply resin by hand, bake, destructively test.
2. Only if the joint passes §6, proceed to the rotating cage and fixture.
3. Then the XYZ gantry.
4. Then vision servo.
5. Dispensing last.

The joint recipe — turn count, tension, wrap length — cannot be derived analytically. Establish it empirically with a ~20-sample design of experiments against a load tester before writing any winding program. Everything downstream depends on reproducing a known-good joint, and a machine that lays a beautiful wrap that does not hold is the most expensive possible failure mode.
