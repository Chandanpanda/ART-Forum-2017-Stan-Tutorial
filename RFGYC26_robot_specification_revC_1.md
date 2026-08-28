# RFGYC'26 Senior — Two-Robot System Specification — REV C

**Purpose:** self-contained engineering brief for the Rev C drawing set, ITU Robotics for Good Youth Challenge 2026–27, Senior. Assumes no prior context. All dimensions in millimetres. `[VERIFY]` marks values to confirm on a physical field mock-up before manufacture.

**Rev C supersedes Rev B's manipulation architecture entirely.** The mission analysis, field geometry, scoring logic, beam end-state and route plans from Rev B carry over unchanged and are restated here in compressed form (§1, §9). What changes is *how pieces move inside a robot*: Rev B's push-fed lanes, rear chutes and gravity ducts are replaced by a single inclined accumulation conveyor with camera triage and gated swim lanes, common to both agents.

## Revision history

| Rev | Change |
|---|---|
| A | Initial issue. |
| B | Discharge-geometry rule (front ingests, rear discharges at floor level, tail-in/nose-out); beam choreography corrected to axial-only separations; Agent A dispenser clash closed by a deck-ring belly snout. |
| **C** | **C1.** Single **common chassis** for both agents: inclined conveyor 116 wide at 11°. Agents differ only in the printed lane cassette and the software. **C2.** Drive changed to **2 × NEMA 17 steppers + TMC2209**, GT2 2:1, Ø60 wheels inboard flanking the belt, **track 150**, four Ø20 ball transfers. Encoders and bump-pad microswitches deleted — StallGuard replaces them. **C3.** **Turning geometry governed:** drive axle placed on the fore-aft centroid; corner chamfers 40 × 45°; turning ratio τ defined and bounded (§4). **C4.** Intake becomes a **two-finger sweeper on one MG996R** feeding a 0.5 shim scoop — the fingers sweep, they never lift. **C5.** Triage by **downward camera in a shrouded tunnel + passive plow diverter on one MG90S**; TCS34725 retained as arbiter. **C6.** Buffering by **swim lanes on the moving belt**, accumulation against MG90S flap gates; belt always on. **C7.** Discharge by **converging funnel over the tail roller** — gravity drop past the rear face. Mutually-exclusive gate interlock. **C8.** Agent A's disc dispensing returns to the rear face: the belt is the lift Rev B lacked, so a short vertical Ø58 chute now works (§6.4). **C9.** Battery 3S → **4S 14.8 V**. |
| **C.1** | Changes forced by the BOM review (§13). **D1.** Slat apron → **flat elastomer belt on crowned rollers** — 106 slats and ~424 M2 screws were not a buildable assembly. **D2.** Agent A's disc lane deleted; the **Ø58 chute becomes the magazine** — three Ø56 discs need 168 of lane and only ~120 exists. **D3.** The belt nose cannot sit at floor level (roller radius), so the shim ramp is a **scoop** driven by the robot's forward motion; it consumes 48–65 of the fore-aft budget, and all belt/lane/triage stations shift aft accordingly. **D4.** Mass and torque re-checked against the actual parts (§13.3). |

---

## 1. Mission context (carried from Rev B)

Two robots, 1143 × 1181 walled table, one 120 s fully autonomous round, both starting inside a 480 × 280 deployment box marked with 20 tape.

| # | Task | Objects | Carried at start | Agent |
|---|---|---|---|---|
| 1 | Medical kits to three zones | 10 cubes 25 × 25 × 20, ~9 g | Yes | B |
| 2 | Samples from quarantine → lab plate | 3 discs Ø56 × 5, ~8 g | No | A |
| 3 | Seal quarantine with two beams | 280 × 60 × 20 (~200 g), 250 × 60 × 20 (~180 g) | Yes | A |
| 4 | Colour-sort 12 patient markers | Ø20 × 20, ~5 g, red/yellow/green | No | B |

Field frame: origin inside bottom-left, X right 0→1143, Y up-field 0→1181, walls 19–20 thick and 65–70 tall. Quarantine X 0–280 / Y 0–280. Deployment box X 643–1123 `[VERIFY]`, recovery zone (RZ) inside it. Lab plate 440 × 150 × 3, three Ø60 through-holes, pitch 140 `[VERIFY]`. Hospital and two PCCs along the top wall, ~180 deep `[VERIFY]`. Twelve Ø20 patient stickers in fixed positions, colour randomised — waypoint navigation plus classification, never search.

**Beam end state** (unchanged): both standing on a 20 face, 60 tall, free-standing, untouched at the buzzer. Beam 1 (280) from the left wall along Y 250–270; beam 2 (250) from the bottom wall along X 280–300, its end face butting beam 1's south side face at ≈ (280, 250). Static tip-over 18.4°.

**Governing discharge rule** (unchanged from Rev B §2.1): the front face ingests; discharge exits past a robot edge at floor level; deposits are tail-in and departed nose-out. Neither robot ever travels over a piece it has placed, nor releases a piece beneath its own chassis.

---

## 2. Architecture — one chassis, two builds

Both agents are the **same rolling chassis and the same conveyor**. They differ in three printed parts (lane cassette, front guides, tail funnel), the presence of the beam pockets on A, and the firmware.

```
FRONT                                                          REAR
[sweeper fingers] → [scoop] → ═══ belt, 11° up ═══ → [tail roller]
                                   ↑            ↑    ↑         ↓
                          converge+camera   diverter lanes  funnel → floor
```

| | Agent A | Agent B |
|---|---|---|
| Owns | Beams + samples | Kits + cylinders |
| Envelope L × W × H | 285 × 235 × 175 | 265 × 180 × 200 |
| Lane cassette | none — converging guides to a Ø58 chute-magazine | 4 lanes: kits 30, R 26, Y 26, G 26 |
| Triage camera | **No** — single class | Yes |
| Servos | 4 (sweeper, disc gate, 2 finger sets) | 7 (sweeper, diverter, 4 gates, kit escapement) |
| Steppers / aux motors | 2 / 1 belt | 2 / 1 belt |

**Why the belt is the right primitive here:** every actuator becomes zero-force. The belt supplies all motive energy; servos supply only *direction* (diverter blade) and *permission* (lane gates). Nothing lifts, grips against load, or pushes a queue. The 11° incline also buys 53 mm of height at the tail, which is what makes gravity discharge past the rear face possible — and it is the lift that Rev B did not have, which is why Rev B's rear-duct dispenser had to be rejected (0.6° of fall against a 17° friction angle). That objection does not apply to Rev C.

**System totals:** 4 steppers, 2 belt motors, 11 servos, 1 camera, 1 TCS34725, 4 VL53L1X, 2 MPU6050, 2 TCRT5000 arrays, 1 beam-contact microswitch, 2 ESP32-S3, 2 e-stops. **Zero encoders, zero bump-pad switches.**

---

## 3. Common chassis

### 3.1 Width budget — the binding constraint

| | Agent A (235) | Agent B (180) |
|---|---|---|
| Belt | 116 | 116 |
| Wheels 2 × 22 | 44 | 44 |
| Frame walls + clearance | 20 | 20 |
| Beam pockets 2 × 24 | 48 | — |
| **Total** | **228** of 235 | **180** of 180 |

Lateral stations, both agents, from the body centreline:

| Feature | Agent A (Ya, body CL 117.5) | Agent B (Xb, body CL 90) |
|---|---|---|
| Beam pockets | 0–24, 211–235 | — |
| Wheel bands (22) | 31.5–53.5, 181.5–203.5 | 4–26, 154–176 |
| **Wheel centres → track** | 42.5 / 192.5 → **150** | 15 / 165 → **150** |
| Belt (116) | 59.5–175.5 | 32–148 |
| Belt-to-wheel gap | 6 | 6 |

### 3.2 Conveyor

| Item | Spec |
|---|---|
| Type | Inclined **flat elastomer belt**, accumulation duty, **always running** |
| Belt | 1.5 EPDM or neoprene sheet, 116 wide, loop cut to length, **lap joint** with contact adhesive + 20 backing patch |
| Rollers | Printed PETG Ø16, **crowned 0.8 across the width**, on 5 shafts in 625ZZ bearings; tail roller flanged and silicone-taped for grip |
| Tracking | Crown on both rollers + tail flanges; sprung idler at the nose sets tension |
| Incline | **11°**; belt top **Z 17.5** at the nose roller, **Z 51** at the tail |
| Support | 3 acrylic slide plate under the upper run |
| Drive | N20 100 rpm on the tail roller, ~60 mm/s, ~150 mA |
| Placement | A: nose Xa 210, tail Xa 35 (run 175). B: nose Yb 202, tail Yb 10 (run 192) |

**This replaces the Rev C slat apron, which the BOM killed.** A slatted apron at 10 pitch needs 53 printed slats per robot — 106 in total, ~640 g of PETG, and about 424 M2 screws into the GT2 belt clamps. That is a week of assembly and 424 chances to build in a misalignment. A crowned roller self-centres a flat belt, which is standard practice and needs no fasteners at all: two rollers, one joined loop, one idler.

**Accumulation:** a piece held at a closed gate rests on the belt while it slides beneath. Retarding force per piece ≈ µ·m·g ≈ 0.6 × 0.005 × 9.81 ≈ **0.03 N**; four pieces ≈ 0.12 N against a belt that can pull ~30 N. This is the one-directional friction bias you specified — it holds the queue packed against the gate with no actuator and no sensing.

**The belt surface must stay smooth — no cleats, no lips.** A cleat makes the belt carry positively, and at ~30 N of stall force it would crush the queue against a closed gate. Friction-only carry is the whole point. Carrying capacity on the incline is not at risk: rubber-on-wood µ ≈ 0.6 against tan 11° = 0.19, a 3× margin.

**Roll-back:** a cylinder that topples onto its side can roll down the incline. A fixed **6 mm retention lip immediately aft of the scoop crest** (static, not on the belt) catches it and returns it to the belt. It cannot escape the robot.

### 3.3 Drive

| Item | Spec | Note |
|---|---|---|
| Motors | 2 × NEMA 17, 42 × 42 × 40, ~4 kg·cm | ~₹500 each |
| Driver | **TMC2209**, 1/8 microstep | ~₹350 — see below |
| Reduction | GT2-6, 20T → 40T, **2:1**, idler tensioner | centre distance ~93 |
| Wheels | Ø60 × 22, inboard, **stub axles on outboard bearing plates** | no through-shaft — it would cross the belt |
| Track | **150** | both agents |
| Ground support | 4 × Ø20 ball transfer at the corners | rear pair on 1.5 compliant mounts |
| Supply | **4S 14.8 V 1500 mAh**, hard case, bolted | 3S caps speed at ~0.3 m/s on back-EMF |

Resolution and speed: 200 steps/rev × 2:1 = 400 steps/rev at the wheel; Ø60 → 188.5 circumference → **0.47 mm/full-step**, 0.059 mm at 1/8. 0.4 m/s = 850 full-steps/s — trivial for an ESP32 with FastAccelStepper. Turn-in-place: 1000 full-steps per 360°, **0.36°/step**.

Torque: required at the wheel ≈ 5 N·cm; available after the 2:1 ≈ 50 N·cm. Margin ~10×.

**Why TMC2209 rather than A4988:** StallGuard reports mechanical stall electrically. Drive into a field wall, read the stall, stop square — that is the wall-squaring reference, and it deletes all four bump-pad microswitches, their brackets and their wiring on both agents. Silent operation and lower heat are secondary benefits.

**The honest cost of steppers:** a skipped step is silent — nothing tells the firmware the odometry is now wrong. This is survivable only because **every terminal placement closes on a physical feature** — wall stall, gate, tape crossing, contact lever — and never on dead reckoning. That rule from Rev B is now load-bearing. Break it and steppers will lose a round.

**Motors are mounted at the front-outboard**, Z 55–95, clear of the triage tunnel laterally (motors at Xb 6–48 / 132–174; camera tunnel at Xb 66–114). Battery and electronics sit at the rear, Z 55–95, to trim the CG onto the axle (§4.3).

---

## 4. Turning geometry — governed, not incidental

Turning behaviour is treated as a controlled quantity in Rev C. Define:

> **Turning ratio τ = swept turning-circle diameter ÷ largest body plan dimension.**

τ = 1.0 means the robot turns entirely within its own footprint. **Target τ ≤ 1.25.**

### 4.1 Axle on the centroid

The single largest lever is the fore-aft position of the drive axle. Turn-in-place happens about the axle midpoint, so swept radius is the distance from that point to the farthest body corner.

| Axle position | Agent B swept radius | τ |
|---|---|---|
| Yb 210 (rear-biased, as in an earlier draft) | 228 | 1.72 |
| **Yb 132.5 (fore-aft centroid)** | 160 | 1.21 |
| Centroid + 40 corner chamfers | **142** | **1.07** |

**Drive axle sits on the fore-aft centroid on both agents:** Agent A Xa 142.5, Agent B Yb 132.5. This is why the wheels must run on stub axles — at that station the belt top sits at Z 30, exactly where a through-shaft would be.

### 4.2 Corner chamfers

**40 × 45° chamfers on all four corners.** Free — it is only how the deck plates are cut — and worth 11 % of swept radius. Agent B: swept radius 160 → 142, τ = 283/265 = **1.07**.

**Agent A is floor-limited and cannot reach this.** Its beam pockets run the full length at both flanks, so the corners cannot be chamfered below Za 60, and with a 280 beam aboard the swept radius is set by the beam ends at **185** regardless of chassis shape. Agent A therefore runs τ = 370/285 = **1.30**, and the Rev B rule stands unchanged: **rotate only at ≥ 200 from any placed beam.** Chamfer A above Za 60 anyway for wall clearance.

### 4.3 Castor scrub and CG

Ball transfers, not swivel castors: a swivel castor must be dragged into alignment at the start of every rotation, which is precisely the transient that skips steps.

Scrub torque in turn-in-place ≈ µ_ball · N · r ≈ 0.05 × 4 N × 0.142 m ≈ **0.03 N·m**. Available turning torque ≈ 2 × 26 N × 0.075 m ≈ **3.9 N·m**. Margin >100×, so rotation — the highest-risk manoeuvre for a stepper — has the largest margin in the design.

CG is trimmed to within **25 of the axle** by placing the motors forward and the battery aft. Drive wheels then carry ~85 % of weight (traction), the forward ball pair is lightly loaded, and the rear pair acts as an anti-tip stop under braking and during tail-in reversing.

---

## 5. Intake — sweeper fingers

Your two-finger, one-servo effector, with the impossible degree of freedom removed.

- **One MG996R** drives a 1:1 spur pair so two fingers **counter-rotate closed** through ~110°. Fingers are 90 long, 25 tall, printed PETG with a 3 silicone lip.
- **The fingers do not lift — the robot's forward motion does.** A Ø16 nose roller puts the belt top at Z 17.5, not at the floor, so a **scoop** bridges the gap: a 0.5 spring-steel shim, leading edge 0.3 above the surface, rising to the belt nose. The piece is stationary in the world frame while the robot advances, so the scoop slides under it and the piece rides up — a dustpan, not a lift. Nothing is actuated.
- **Scoop run:** Agent B **48 at 20°** (Ø20 cylinders only, which climb easily); Agent A **65 at 15°** (gentler for a 5 mm disc, which must not be flicked). This is why the belt noses sit at Yb 202 / Xa 210 rather than at the front face — the scoop owns the front 48–65 mm.
- The fingers only sweep the 165-wide capture band inward onto the 116 belt. They are therefore also the intake funnel — one part doing both jobs.
- Open at 165 capture, closed at 116 (belt width). Cycle ~0.4 s. Default state during transit: **open**, so anything the robot drives into is captured rather than plowed.

This is the walk-over insurance in the collecting direction: a stray cylinder brushed on any leg is ingested and re-sorted rather than displaced for −3.

---

## 6. Triage, lanes and discharge

### 6.1 Convergence and classification (Agent B)

| Station | Yb | Spec |
|---|---|---|
| Convergence wedge | 202 → 172 | 116 → 30, **asymmetric** — one wall straight, one at 22° |
| Camera tunnel | 172 → 157 | opaque PETG shroud, 2 × white LED constant current |
| Camera | 167, Zb 90 | 60 standoff, looking down |
| TCS34725 | 162, Zb 42 | arbiter, flush in the tunnel floor |
| Plow diverter pivot | 152 | MG90S, blade 90 long |

At 60 mm/s a piece dwells ~0.6 s in the ROI — about 12 frames at QQVGA. Classification is mean RGB of a centre ROI against a white reference taken at t = 0 inside the shroud; three widely separated hues, hard thresholds, no ML.

**Camera:** XIAO ESP32S3 Sense (~₹1800) — clean pinout, coexists with the rest of the bus. ESP32-CAM (~₹500) is the budget alternative but is awkward to integrate; if used, give it its own MCU and pass the class over UART.

**Keep the TCS34725.** The camera's real contribution is *timing* — knowing exactly when the piece is under the blade — while a shrouded photodiode is the more repeatable colour instrument. Two cheap sensors agreeing is worth more than one expensive one guessing.

**The convergence wedge is the #1 jam risk in this design.** Two cylinders arriving abreast will bridge across a symmetric throat. The asymmetric wedge is standard anti-bridging geometry; verify at §10.5 with 50 ingests before trusting it.

### 6.2 Diverter

One MG90S holds a **passive plow blade** at one of four angles; the moving belt does all the work of translating the piece along the blade face. Lateral deflection up to 43 over a 90 blade run = 25° maximum. The servo never pushes a load, only holds an angle against a light side force.

### 6.3 Swim lanes (Agent B cassette)

Printed PETG dividers 2.5 thick standing 30 above the belt, bonded to the frame, **not touching the belt surface**.

| Lane | Contents | Width | Xb |
|---|---|---|---|
| L1 | Kits | 30 | 34–64 |
| L2 | Red | 26 | 66.5–92.5 |
| L3 | Yellow | 26 | 95–121 |
| L4 | Green | 26 | 123.5–149.5 |

Lane run Yb 152 → 45, **length 107**. Four 25 kits need 100 and four Ø20 cylinders need 80, so L1 is the binding lane with 7 to spare. Each lane ends in an **MG90S flap gate** that drops into the lane; the belt runs on beneath it.

**Kit metering.** The belt packs kits into a queue, so a plain gate cannot meter 6 / 2 / 2. L1 stages **4 kits on the belt**; the other **6 sit in a vertical 27 × 27 tube** above the lane's rear with an MG90S shutter at its base. The shutter is the escapement — the only metered mechanism in the design. Everything else is dump-through-a-gate.

### 6.4 Discharge

Converging guides funnel all four lanes to a single spout over the **tail roller**, from which pieces fall 53 mm to the floor **behind the rear face**. Because belt speed is only 60 mm/s the horizontal throw is negligible: pieces land 5–15 behind the rear face, clear of the chassis, satisfying the Rev B tail-in rule.

**Mutual-exclusion interlock (firmware, mandatory):** *at most one lane gate is open at any instant.* The converging funnel is only free of cross-contamination because of this. Write it as a hardware-style interlock in the gate driver, not as a convention in the route script.

**Agent A — the chute is also the magazine.** Three Ø56 discs in a row need 168 of lane, and Agent A's belt run leaves about 120 — so Agent A has **no disc lane at all**. Converging guides deliver each disc to the tail, where it drops into a **vertical Ø58 chute at axis Xa 33** that doubles as the magazine: 40 of usable height holds up to 8 discs at 5 thick, and only 3 are ever carried. An **MG90S gate at the chute base** meters one disc per stroke.

The gate tip sits **Za 11 — 8 above the 3 plate**, with a 45° chamfered lead-out. Drop is 8, so exit velocity is ~0.4 m/s; a light foam brush at the tip damps it further. Disc-in-hole radial clearance is (60 − 56)/2 = 2; the chamfer absorbs ±10 of robot position error.

This deletes a lane cassette, a lane gate and the foam-pinch detail from Agent A. It also decouples collection order from dispense order — discs can arrive in any sequence and still be posted one at a time.

Agent A therefore **docks the lab in reverse**, tail over the plate. This is the Rev B rear-face option, now valid because the belt provides the lift.

`[VERIFY §10.2]` Tail overhang to reach a hole ≈ 75, while the drive wheels sit 108 forward of the chute — so the wheels may just roll onto the 3 plate. With Ø60 wheels a 3 step is a non-event mechanically, but confirm the plate does not shift. Fallback if it does: a 30 deployable tongue on one MG90S carrying the chute aft, deployed only after the start (the start envelope must stay within 285).

---

## 7. Agent A — station schedule

Envelope 285 (Xa, fore-aft) × 235 (Ya) × 175 (Za). Origin rear-right, floor. +Xa forward.

| Station | Xa | Ya / Za |
|---|---|---|
| Sweeper fingers, MG996R | pivots 278 | capture 165 → 116 |
| **Scoop, 0.5 shim @ 15°** | 275 → 210 | run 65, leading edge 0.3 above floor |
| Belt nose roller Ø16 | 210 | belt top Za 17.5 |
| Roll-back retention lip | 203 | 6 tall, static |
| Converging guides to the chute | 200 → 50 | 116 → 62 |
| Belt tail roller Ø16 | 35 | belt top Za 51 |
| **Disc chute / magazine Ø58** | axis 33 | Za 51 → 11, capacity 8, carries 3 |
| Disc gate MG90S | 33 | at the chute base, one disc per stroke |
| Drive axle (stub) | **142.5** | track 150, axle Za 30 |
| Steppers ×2 | 200–242 | Ya 26–68 / 167–209, Za 55–95 |
| Battery + electronics | 40–110 | Za 55–95 |
| Ball transfers ×4 | 40, 245 | at the chamfer corners |
| Beam pocket L (beam 1, 280) | 0 → 285, open front | Ya 211–235 |
| Beam pocket R (beam 2, 250) | 0 → 258, 8 float | Ya 0–24 |
| Fingers L / R, MG996R common shaft | 245, 30 / 235, 30 | 50 tall, 3 end-stop hooks |
| Beam-contact microswitch | pocket R leading end | projects 0.5–1 |
| ToF ×2 | left flank 200 Za 75; front 45° | VL53L1X |
| TCRT5000 array | front underside, shrouded | tape fixes |
| E-stop Ø20 | 40 | Za 175 |

**Beam carriage is carried over from Rev B unchanged** and must not be disturbed: open-bottomed, open-ended flank pockets; beams stand on the field and are never lifted; release is **axial separation only**, never lateral, never with yaw while a pocket overlaps a beam end; fingers swing up and outboard. Beam 1 set 5 short with a chamfered end; beam 2 biased 5 inboard.

A 280 beam cannot ride a 116 belt inside a 285 body — the beams stay in the pockets. This is the one manipulation in the system the conveyor does not touch.

---

## 8. Agent B — station schedule

Envelope 265 (Yb, fore-aft) × 180 (Xb) × 200 (Zb). Origin rear-left, floor. +Yb forward.

| Station | Yb | Xb / Zb |
|---|---|---|
| Sweeper fingers, MG996R | pivots 253 | capture 165 → 116 |
| **Scoop, 0.5 shim @ 20°** | 250 → 202 | run 48, leading edge 0.3 above floor |
| Belt nose roller Ø16 | 202 | belt top Zb 17.5 |
| Roll-back retention lip | 195 | 6 tall, static |
| Steppers ×2 | 205–247 | Xb 6–48 / 132–174, Zb 55–95 |
| Convergence wedge | 202 → 172 | 116 → 30, asymmetric |
| Camera + tunnel | 157–172 | Xb 66–114, camera Zb 90 |
| TCS34725 | 162 | tunnel floor |
| Plow diverter, MG90S | 152 | blade 90 |
| Lanes L1–L4 | 152 → 45 | per §6.3 |
| Kit escapement tube 27 × 27, MG90S | 125–152 | Xb 34–64, Zb 60–170, 6 kits |
| Lane gates ×4, MG90S | 45 | flaps |
| Discharge funnel | 45 → 10 | 116 → 40 |
| Belt tail roller Ø16 | 10 | belt top Zb 51 |
| Drive axle (stub) | **132.5** | track 150, axle Zb 30 |
| Battery + electronics | 20–75 | Zb 55–95 |
| Ball transfers ×4 | 25, 240 | at the chamfer corners |
| ToF ×2 | front centre Zb 60; right flank | VL53L1X |
| TCRT5000 array | front underside, shrouded | tape and RZ fixes |
| E-stop Ø20 | 60 | Zb 200 |

---

## 9. Operations (carried from Rev B)

**Agent A, 120 s:** two sweep passes of the quarantine (mouth centred Y ≈ 70 then 210), ~30 s → three reverse dockings at the lab, ~28 s → beam 1 on the left wall, ~20 s → beam 2 to contact closure, ~22 s → park **east of the sealed corner**, reserve ~20 s. **Hard abort at T−40 s:** if the disc phase is incomplete, abandon it and place beams. The 70-point task must never die downstream of the 50-point one.

**Agent B, P1 (aggressive, round 1):** right sticker columns, 6 markers (~28 s) → cross mid-field on tape fixes → left columns, 6 (~28 s) → PCC-L tail-in, 2 kits + 2 yellow (~8 s) → Hospital, 6 kits + 4 red (~10 s) → PCC-R, 2 kits + 2 yellow (~8 s) → RZ, greens, park (~10 s). One visit per zone. **T−50 abort** jumps straight to the delivery chain: the kits are aboard regardless, so the −30 empty-zone penalty is always killed.

**Agent B, P2 (safe, later rounds):** north pass depositing 2 / 6 / 2 tail-in (~35 s) → both sticker rows (~50 s) → three discharges + home (~30 s). Second visits offset ≥ 100 laterally within the zone.

Ranking is best-single-round, so run P1 first.

**Localisation.** Steppers give exact open-loop odometry for gross navigation; every terminal closes on a physical feature:

| Terminal | Reference |
|---|---|
| Wall squaring, all cases | **TMC2209 StallGuard** on both motors |
| Lab holes ×3 | bottom-wall stall + step offset + Ø58 chamfer capture ±10 |
| Beam 1 / beam 2 | left-wall stall / bottom-wall stall + contact lever |
| H and PCC tail-in depth | front ToF to the top wall, tail 50 inside the zone |
| RZ entry | front TCRT array crossing the box tape |
| Mid-field | TCRT crossings of the 20 tape figures; MPU6050 heading hold |

**Inter-agent deconfliction:** Agent A's post-beam park (X ≈ 300–530, Y ≈ 350–500, from ~T+100 s) and Agent B's P1 left-column leg (X ≈ 150–350, ~T+55–75 s) are temporally separated.

---

## 10. Open items — physical mock-up

Build the quarantine corner, the lab plate and one healthcare strip first.

1. **Beam seating and the "touch" criterion** — T-joint against the 20 floor line; whether a ≤ 1 gap at the contact lever satisfies the referee.
2. **Lab plate position, hole pitch (140 assumed), and whether the drive wheels mount the 3 plate** (§6.4) — decides whether the deployable tongue is needed.
3. **Deployment box X offset (643 assumed); sticker grid; zone depths.**
4. **Belt tracking** — run the loop for 10 minutes loaded and confirm it stays centred on the crowns; if it walks, add a V-guide or fall back to the slat apron.
5. **Lap-joint durability** — the belt joint passes over a Ø16 roller thousands of times per match day; inspect after a full day of testing.
6. **Accumulation friction on the real surface** — confirm ~0.03 N/piece and that a full lane does not creep past a closed gate.
7. **Convergence wedge jam rate** — 50 ingests per lane at worst-case entry angle and doubled arrivals; adjust the asymmetric wall until zero.
8. **Scoop pickup rate** — a Ø20 cylinder at 20° and a Ø56 × 5 disc at 15°, at approach speeds 0.1–0.4 m/s; confirm the piece climbs rather than being pushed along the floor.
9. **StallGuard threshold calibration** per venue surface — and note that StallGuard is unreliable below ~0.1 m/s, so wall squaring must be done at moderate speed and then backed off.
8. **Step-loss audit** — instrument a full P1 run and compare commanded vs measured pose at three checkpoints; if loss is non-zero, lower acceleration before adding sensors.
9. **Camera classification margins** across venue lighting; log hue separation and confirm the TCS34725 agrees.
10. **Disc gate exit velocity** with the foam brush — target ≤ 0.4 m/s, no bounce-out of the Ø60 hole.
11. **CG trim** — measure and shim to within 25 of the axle.

## 11. Compliance checklist

- [ ] Both robots and all game elements inside 480 × 280 at start; nothing beyond the tape
- [ ] Fully autonomous after activation; no operator link
- [ ] Accessible, identifiable e-stop per robot; SDG 3 decoration per robot
- [ ] No sharp edges, loose parts, unsecured battery or cabling
- [ ] Nothing placed on the field outside the box; no off-robot infrastructure
- [ ] **Gate interlock: never more than one lane gate open**
- [ ] **No terminal placement relies on dead reckoning alone**
- [ ] Walk-over audit: no release under the chassis; no path over a placed piece; no rotation within 200 of a placed beam
- [ ] Design document and robot video prepared for submission

---

## 12. Drawing brief — Rev C sheet list

Orthographic line drawing, monochrome plus one accent for game elements, mm dimensions, leader callouts, no perspective. REV field = C.

**Sheet 1 — Deployment packing plan.** Both agents loaded in the box; beam orientation constraint inset retained.

**Sheet 2 — Common chassis GA, three views.** The shared module: belt 116 at 11°, stub axles at the centroid, track 150, motors forward-outboard with the GT2 2:1 run, four ball transfers, 40 corner chamfers. Section through the belt showing the crowned roller, belt, slide plate and sprung idler.

**Sheet 3 — Turning geometry.** Plan overlay of the swept circle for three axle positions (Yb 210 / centroid / centroid + chamfers) with τ tabulated; castor scrub and CG-trim diagram; Agent A's beam-loaded swept radius of 185 with the ≥ 200 rotation-clearance circle.

**Sheet 4 — Intake.** Sweeper fingers open and closed (phantom), 1:1 spur pair, scoop section at 15° (A) and 20° (B) with the run dimensioned to the nose roller, roll-back retention lip, capture band 165 → 116.

**Sheet 5 — Agent B GA and cassette.** Lane layout with widths, kit escapement tube, gates, converging discharge funnel, camera tunnel, diverter blade at all four angles (phantom).

**Sheet 6 — Triage detail.** Section on the belt centreline: convergence wedge (asymmetric, dimensioned), shrouded tunnel, camera standoff 60, TCS34725 in the floor, plow blade geometry with the 25° maximum deflection.

**Sheet 7 — Discharge detail.** Tail roller, funnel convergence, 51 drop, landing zone 5–15 behind the rear face, tail-in deposit posture at a healthcare zone. Inset: why under-chassis release fails (20 piece vs 6 clearance).

**Sheet 8 — Agent A GA.** Beam pockets carried from Rev B (open-ended, end-stop hooks, contact lever), converging guides, Ø58 chute-magazine with its base gate and the 8 stand-off above the plate, reverse lab-docking posture with the wheel-on-plate question annotated.

**Sheet 9 — Field operations plan.** Agent A route with the T−40 abort; Agent B P1 and P2 overlays with the T−50 abort; all `[VERIFY]` field geometry marked.

---

## 13. Bill of materials

Prices are indicative Indian retail (robu.in / Amazon.in / local fabricator), inclusive of GST, and should be re-quoted before ordering. Everything listed is commodity stock with no lead time. `⚠` marks the line items worth scrutinising before committing.

### 13.1 Structure and consumables (shared)

| Item | Spec | Qty | Unit ₹ | ₹ | Note |
|---|---|---:|---:|---:|---|
| Cast acrylic, laser cut | 3 mm, 600 × 450, cut to drawing | 2 sheets | 1,200 | 2,400 | Chennai fabricator, material + cutting |
| PETG filament | 1.75 mm, 1 kg, any brand | 1 | 1,200 | 1,200 | ~550 g of printed parts across both agents |
| Spring steel shim | 0.5 mm, 100 × 200 | 1 | 180 | 180 | scoops; feeler-gauge stock is equivalent |
| Contact adhesive + backing | for the belt lap joint | 1 | 120 | 120 | ⚠ joint is a wear point |
| LiPo balance charger | 4S capable, 50 W | 1 | 2,000 | 2,000 | shared |
| **Subtotal** | | | | **5,900** | |

### 13.2 Per-agent bill

Quantities are per robot. Where A and B differ, both are given.

**Drivetrain**

| Item | Spec | Qty A | Qty B | Unit ₹ | Note |
|---|---|---:|---:|---:|---|
| Stepper motor | NEMA 17, 42 × 42 × 40, 1.8°, ≥ 4 kg·cm | 2 | 2 | 550 | |
| Stepper driver | TMC2209 module | 2 | 2 | 350 | ⚠ StallGuard needs UART — see §13.4 |
| GT2 pulley 20T | 5 mm bore | 2 | 2 | 90 | motor end |
| GT2 pulley 40T | 8 mm bore | 2 | 2 | 150 | axle end, 2:1 |
| GT2-6 belt | 200 mm closed loop | 2 | 2 | 110 | |
| Drive wheel | Ø60 × 22, 8 mm bore | 2 | 2 | 150 | |
| Stub shaft | 8 mm × 40, mild steel | 2 | 2 | 60 | |
| Bearing | 608ZZ | 4 | 4 | 25 | 2 per stub axle |
| Ball transfer unit | Ø20 nylon or steel ball | 4 | 4 | 100 | ⚠ not swivel castors — see §4.3 |
| **Subtotal** | | | | **3,420** | same both agents |

**Conveyor**

| Item | Spec | Qty A | Qty B | Unit ₹ | Note |
|---|---|---:|---:|---:|---|
| Elastomer sheet | 1.5 mm EPDM/neoprene, 150 × 600 | 1 | 1 | 350 | ⚠ gasket shop, not robu |
| Roller | printed PETG Ø16, crowned 0.8 | 2 | 2 | — | filament above |
| Idler + tension spring | printed roller + extension spring | 1 | 1 | 40 | |
| Shaft | 5 mm × 130 | 2 | 2 | 40 | |
| Bearing | 625ZZ | 4 | 4 | 20 | |
| Gearmotor | N20, 100 rpm, 12 V | 1 | 1 | 250 | belt drive |
| Silicone tape | for tail-roller grip | 1 | 1 | 80 | |
| **Subtotal** | | | | **≈ 900** | same both agents |

**Actuation**

| Item | Qty A | Qty B | Unit ₹ | Purpose |
|---|---:|---:|---:|---|
| MG996R servo | 3 | 1 | 350 | A: sweeper + 2 beam finger sets. B: sweeper |
| MG90S servo | 1 | 6 | 150 | A: disc gate. B: diverter + 4 lane gates + kit escapement |
| **Subtotal** | **1,200** | **1,250** | | |

**Sensing and control**

| Item | Qty A | Qty B | Unit ₹ | Note |
|---|---:|---:|---:|---|
| ESP32-S3 DevKitC-1 | 1 | 1 | 900 | |
| VL53L1X ToF | 2 | 2 | 350 | |
| MPU6050 IMU | 1 | 1 | 150 | |
| TCRT5000 5-ch line array | 1 | 1 | 200 | shrouded |
| TCS34725 colour | — | 1 | 450 | arbiter |
| XIAO ESP32S3 Sense | — | 1 | 1,800 | ⚠ camera — see §13.4 |
| White LED + resistor | — | 2 | 10 | tunnel illumination |
| **Subtotal** | **1,950** | **4,220** | | |

**Power and wiring**

| Item | Spec | Qty | Unit ₹ | Note |
|---|---|---:|---:|---|
| LiPo battery | 4S 14.8 V 1500 mAh 25C, hard case | 2 | 1,200 | ⚠ two per robot per competition day |
| Buck converter | 5 V 5 A | 1 | 250 | servo rail |
| E-stop | Ø22 latching NC mushroom | 1 | 250 | homologation gate |
| Main switch + fuse | 10 A | 1 | 200 | |
| Connectors + wire | XT60, JST, 18/22 AWG silicone, heatshrink | set | 500 | |
| **Subtotal** | | | **3,600** | same both agents |

**Fasteners**

| Item | Qty | ₹ |
|---|---:|---:|
| M3 socket cap 8/12/16, nyloc nuts, brass standoffs | set | 500 |
| M2 × 6 for servo horns and brackets | set | 100 |
| **Subtotal** | | **600** |

### 13.3 Totals, mass and margin

| | Agent A | Agent B |
|---|---:|---:|
| Drivetrain | 3,420 | 3,420 |
| Conveyor | 900 | 900 |
| Actuation | 1,200 | 1,250 |
| Sensing and control | 1,950 | 4,220 |
| Power and wiring | 3,600 | 3,600 |
| Fasteners | 600 | 600 |
| Structure share | 1,900 | 1,900 |
| **Per agent** | **13,570** | **15,890** |

**System ≈ ₹29,460**, plus ~₹2,000 shared charger and a recommended 20 % spares allowance (a spare stepper, driver, servo set and belt blank) → **budget ≈ ₹38,000**.

Mass, from the actual parts:

| Group | Agent A | Agent B |
|---|---:|---:|
| Acrylic structure | 350 g | 350 g |
| Printed parts | 380 g | 350 g |
| Belt, rollers, shafts | 200 g | 200 g |
| 2 × NEMA 17 | 560 g | 560 g |
| Wheels + ball transfers | 280 g | 280 g |
| Servos | 165 g | 133 g |
| Battery | 170 g | 170 g |
| Electronics + wiring | 150 g | 200 g |
| Payload | 380 g (beams) | 90 g (kits) |
| **Total** | **≈ 2.6 kg** | **≈ 2.3 kg** |

Torque re-check at 2.6 kg with beams: rolling resistance ≈ 0.5 N, acceleration at 0.6 m/s² ≈ 1.6 N, beam sled drag ≈ 1.1 N → **≈ 3.2 N total, 1.6 N per wheel, 4.8 N·cm at the wheel** against **≈ 50 N·cm available** after the 2:1. Margin ~10×. Turn-in-place scrub (§4.3) remains ~0.03 N·m against ~3.9 N·m. **The drivetrain is not close to any limit** — the mass growth from steppers is comfortably absorbed.

Current draw: 2 steppers at ~0.9 A + belt motor 0.15 A + servo rail peaks ~2 A → ~2.5 A average, ~4 A peak. A 1500 mAh 4S sustains a match with large margin; two packs per robot cover a competition day with charging between rounds.

### 13.4 What the BOM review changed, and what to scrutinise

Three findings, all now folded into the spec above:

1. **The slat apron was not buildable.** 53 slats per robot at 10 pitch means 106 printed parts, ~640 g of PETG and roughly **424 M2 screws** into belt clamps — a week of assembly and 424 chances to introduce a misalignment. Replaced by a flat elastomer belt on crowned rollers: two printed rollers, one joined loop, zero fasteners. This is the single biggest build-time saving in Rev C.1.
2. **Agent A's disc lane did not fit.** Three Ø56 discs in a row need 168 mm; the available belt run after the scoop is about 120. The Ø58 chute now serves as the magazine (40 of height, capacity 8, only 3 carried), which deletes Agent A's lane cassette entirely — the lane gate simply becomes the chute base gate, so the servo count is unchanged.
3. **The belt nose cannot be at floor level.** A Ø16 roller puts the belt top at Z 17.5. The shim ramp therefore became a **scoop** whose energy comes from the robot driving forward, and it consumes 48–65 mm of fore-aft budget — which is why every belt, triage and lane station moved aft, and why the lane length recomputed to 107.

Four line items still deserve scrutiny before ordering:

- **Belt material and joint (⚠).** The lap joint is the only wear point in the drivetrain and it passes over a Ø16 roller continuously. Source the sheet from a gasket supplier, make three loops, and destructively test one. If tracking or joint life fails at mock-up, the slat apron is the documented fallback — expensive in labour, but it works.
- **Camera (⚠).** XIAO ESP32S3 Sense at ₹1,800 is a third of Agent B's sensing budget. ESP32-CAM at ~₹500 saves ₹1,300 but has an awkward pinout and poor bus coexistence; if chosen, give it a dedicated MCU and pass only the class over UART. Note that the TCS34725 is doing the actual colour work — the camera's contribution is timing, so the cheap board is defensible.
- **TMC2209 UART (⚠).** StallGuard requires UART configuration, not just STEP/DIR. Up to four drivers share one UART line via MS1/MS2 address straps — wire this deliberately, because retrofitting it after the loom is built is unpleasant. Also note StallGuard is unreliable below ~0.1 m/s, so every wall-squaring move must approach at moderate speed and then back off.
- **Ball transfers, not swivel castors (⚠).** A swivel castor must be dragged into alignment at the start of every rotation, which is exactly the transient that skips steps. Do not substitute on price.
