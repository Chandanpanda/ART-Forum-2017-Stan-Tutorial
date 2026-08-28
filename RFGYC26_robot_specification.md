# RFGYC'26 Senior — Two-Robot System Specification

**Purpose of this document:** a self-contained engineering brief for producing technical diagrams of a two-robot system built for the ITU Robotics for Good Youth Challenge 2026–2027 (Senior category). It assumes no prior context. All dimensions are millimetres unless stated. Masses are estimates derived from wood density 500–800 kg/m³, since the competition rulebook does not specify them.

**Provisional dimensions** are marked `[VERIFY]`. These depend on measurements not given in the rulebook and must be confirmed against a physical field mock-up before manufacture. They are given here as best estimates so the drawing can be produced.

---

## 1. Mission context (background for the illustrator)

Two robots operate together on a 1143 × 1181 mm walled table field for a single 120-second fully autonomous round. No human contact is permitted after start. Both robots must begin entirely inside a 480 × 280 mm deployment box marked with 20 mm black tape.

Four tasks are performed:

| # | Task | Objects | Carried at start? |
|---|------|---------|-------------------|
| 1 | Deliver medical kits to three destination zones | 10 wooden cubes, 25 × 25 × 20 mm, ~9 g | **Yes** |
| 2 | Extract samples from a walled corner and post into a lab plate | 3 wooden discs, Ø56 × 5 mm, ~8 g | No — collected on field |
| 3 | Seal the corner with two free-standing beams | 2 wooden beams, 280 × 60 × 20 and 250 × 60 × 20 mm, ~200 g and ~180 g | **Yes** |
| 4 | Collect and colour-sort patient markers into three destinations | 12 wooden cylinders, Ø20 × 20 mm, ~5 g, in red / yellow / green | No — collected on field |

**Task dependency:** the beams seal the corner that contains the samples. Sample extraction must therefore complete before beam placement. Beam placement is the last operation in that quadrant.

### 1.1 Field reference frame

Origin at the inside bottom-left corner of the field. X to the right (0 → 1143). Y up-field (0 → 1181). Walls are 19–20 mm thick, 65–70 mm tall, on all four sides.

| Zone | Location | Notes |
|---|---|---|
| Quarantine zone | X 0–280, Y 0–280 | Bottom-left. Two field walls form two sides; the two beams form the other two. Contains the 3 samples in random positions. |
| Deployment box | Y 0–280, right-hand side, 480 wide | `[VERIFY]` exact X offset |
| Laboratory plate | Bottom-centre, structure 440 × 150 × 3 mm with three Ø60 through-holes | `[VERIFY]` hole pitch and X/Y position |
| Hospital (H) | Top-centre | Destination: 6 kits + 4 red cylinders |
| Primary care centres (PCC ×2) | Top-left and top-right corners | Destination: 2 kits each + 4 yellow cylinders |
| Recovery zone (RZ) | Inside the deployment box | Destination: 4 green cylinders |
| Patient stickers | 12 × Ø20 marks, six per side, ~100 mm column pitch, two rows | Positions fixed; **colour assignment randomised each match** |

### 1.2 Beam end state (drives the whole Agent A design)

Both beams must finish **standing on a 20 mm-wide face, 60 mm tall, free-standing, with nothing touching them.** Beam 1 touches the left field wall and runs along Y ≈ 280. Beam 2 touches the bottom field wall and runs along X ≈ 280. The two beams must touch each other end-to-end, forming a convex corner at approximately (280, 280) that points into the field.

Static tip-over angle = atan(10 / 30) = **18.4°**. Release must occur with near-zero residual lateral velocity.

`[VERIFY]` The sum of beam lengths (280 + 250 = 530) does not obviously close a 280 × 280 perimeter. The correct seating relative to the 20 mm floor line is undetermined from the rulebook. **This is why Agent A uses contact detection on the second beam rather than open-loop positioning.**

---

## 2. Architecture summary

Two independently controlled differential-drive robots. No arm, no gripper, no suction, no camera, no inverse kinematics anywhere in the system.

Every manipulation is either **bulk intake into a magazine** or **gated release from a magazine**. The only closed-loop perception task is 3-class colour classification of the cylinders.

| | Agent A | Agent B |
|---|---|---|
| Owns | Beams + samples | Kits + cylinders |
| Operating region | Lower-left quadrant | Top of field + side rows |
| Envelope (L × W × H) | 285 × 235 × 175 | 180 × 265 × 200 |
| Drive motors | 2 | 2 |
| Auxiliary motors | 1 (intake roller) | 1 (feed roller) |
| Servos | 3 | 6 |
| Perception | Contact + ranging only | Contact + ranging + 1 colour sensor |

**System totals:** 4 drive motors, 2 auxiliary motors, 9 servos, 1 colour sensor, 4 ToF rangefinders, 2 IMUs, 6 microswitches, 2 microcontrollers, 2 emergency stops.

---

## 3. Deployment box packing plan

Box: 480 (X) × 280 (Y). Origin at the box's bottom-left corner. Nothing may extend beyond the boundary; touching the tape is permitted.

```
Y=280 ┌────────────────────────────────────────────────────────┐
      │                                            ┌──────────┐│
      │  ┌──────────────────────────────────────┐  │          ││
      │  │            AGENT A                   │  │ AGENT B  ││
      │  │         285 × 235                    │  │ 180×265  ││
      │  │  [beam pocket L — 280 mm beam]       │  │          ││
      │  │  [beam pocket R — 250 mm beam]       │  │          ││
      │  └──────────────────────────────────────┘  └──────────┘│
Y=0   └────────────────────────────────────────────────────────┘
      X=0        X=5                    X=290  X=296     X=476  X=480
```

| Item | X range | Y range |
|---|---|---|
| Agent A envelope | 5 → 290 | 5 → 240 |
| Agent B envelope | 296 → 476 | 5 → 270 |
| Inter-robot gap | 6 mm | — |
| Boundary margin | 4–5 mm all round | — |

**Critical constraint:** the 280 mm beam cannot lie along the box's 280 mm axis — zero clearance would breach the boundary rule. Both beams must be carried with their long axis along the 480 mm axis. This single constraint fixes Agent A's orientation in the box and is the reason Agent A consumes 285 of the available 480 mm.

The 10 medical kits are carried inside Agent B's magazine tubes. The 2 beams are carried in Agent A's side pockets, resting directly on the field surface.

---

## 4. Common platform (both agents)

| Subsystem | Specification |
|---|---|
| Chassis | Two-tier, 4 mm laser-cut acrylic or 3 mm aluminium plate, 40 mm deck separation, M3 standoffs |
| Drive motors | 2 × 12 V metal-gear DC, ~300 rpm, integral quadrature encoder (e.g. Pololu 25D or JGB37-520) |
| Wheels | Ø65 × 25 mm, rubber tyre |
| Configuration | Differential drive, single Ø25 mm ball castor |
| Ground clearance | 6 mm |
| Motor driver | Dual H-bridge, TB6612FNG or DRV8833 |
| Controller | ESP32-S3 development board |
| IMU | BNO055 (or MPU6050) for heading hold |
| Battery | 3S LiPo 11.1 V 1500 mAh in a **hard case, rigidly bolted** |
| Emergency stop | Ø20 mm red mushroom latching switch, top deck, breaks battery main. **Mandatory for homologation** |
| Wiring | Fully loomed and strain-relieved; no loose cable permitted at inspection |
| Max speed | 0.45 m/s |
| Acceleration limit | 0.6 m/s² while beams are loaded (tip-over protection) |
| Decoration | Removable SDG 3 (health and well-being) themed shell panels. Robots that fail visual inspection may not compete |

**Surface note for both agents:** the international final surface is specular white whiteboard finish, and national-stage surfaces may be MDF, painted board, or printed tarpaulin. Friction coefficient is undefined and varies between venues. All terminal positioning is therefore referenced to physical features, never to open-loop odometry. Any IR reflectance sensing must be shrouded.

---

## 5. AGENT A — beams and samples

Envelope **285 (fore-aft) × 235 (lateral) × 175 (tall)**. Local frame: origin at the rear-right corner at floor level. +Xa forward, +Ya to the left, +Za up. Forward at deployment points toward −X in field coordinates (toward the quarantine corner).

### 5.1 Layout by station

| Station | Xa position | Description |
|---|---|---|
| Disc scoop | 250 → 285 | Front face, full assembly |
| Disc magazine | 195 → 250 | Vertical stack tube |
| Dispenser snout | 240 → 275, low | Below scoop, forward-facing |
| Drive axle | 120 | Both wheels, track 175 |
| Electronics deck | 60 → 200, Za 60 → 110 | Controller, driver, IMU |
| Battery | 20 → 90, Za 10 → 45 | Low and central |
| Castor | 35 | Rear |
| Beam pocket L | 15 → 275, Ya 211 → 235 | Left side, full length |
| Beam pocket R | 15 → 265, Ya 0 → 24 | Right side |
| E-stop | 40, Za 175 | Top deck, rear |

### 5.2 Disc intake (bulk collection)

Collects all three Ø56 × 5 mm discs in one sweep without knowing where they are. This is what makes the Senior random sample placement irrelevant.

| Parameter | Value |
|---|---|
| Mouth width | 165 (constrained by the two beam pockets) |
| Lead-in ramp | 0.5 mm spring-steel shim, 165 × 60 mm, 12° to floor, leading edge 0.3 mm above surface |
| Intake roller | Ø30 × 165 mm, 8 mm silicone foam over a Ø14 core |
| Roller axis height | 26 mm |
| Roller-to-ramp gap | 3 mm (disc is 5 mm — compression fit) |
| Roller drive | N20 gearmotor, 1000 rpm, timing belt, surface speed ~1.2 m/s |
| Sweep coverage | 280 mm quarantine width covered in **two passes** at 165 mm |

### 5.3 Disc magazine and dispenser

| Parameter | Value |
|---|---|
| Stack tube | Ø62 internal, 45 mm tall, capacity 3 discs with margin |
| Bottom gate | 1 × MG90S servo, sliding shutter, 6 mm slot — releases exactly one disc per actuation |
| Dispenser snout | Forward-facing chute, outside diameter 54 mm, 45° chamfered lead-in |
| Snout tip height | 8 mm above floor |
| Target | Ø60 through-hole in a 3 mm plate — **2 mm radial clearance** |

**Design principle:** the chamfered snout converts up to 10 mm of robot positioning error into a successful insertion. Precision is delegated to a passive mechanical feature rather than to control.

### 5.4 Beam carriage (the critical subsystem — 70 of 250 points)

Beams are **not lifted**. They ride upright on the field surface inside open-sided pockets in the chassis flanks, retained laterally. This eliminates any lift or re-orientation mechanism and makes release a purely lateral separation.

| Parameter | Left pocket | Right pocket |
|---|---|---|
| Beam carried | 280 × 60 × 20 mm | 250 × 60 × 20 mm |
| Pocket depth (lateral) | 24 mm | 24 mm |
| Pocket length | 285 mm | 265 mm |
| Retaining fingers | 2 (fore at Xa 245, aft at Xa 30) | 2 (fore at Xa 235, aft at Xa 30) |
| Finger height | 50 mm (beam is 60 mm — fully constrains tipping) |
| Finger actuation | 1 × MG996R per pocket, both fingers on a common shaft, 90° swing upward and outward |
| Contact switch | — | 1 × microswitch at the forward end, actuated by beam-to-beam contact |

Estimated sled drag: 2 beams ≈ 380 g at µ ≈ 0.3 → ~1.1 N. Negligible for the drivetrain.

### 5.5 Beam placement sequence (draw as a 3-step strip)

**Step 1 — place the 280 mm beam.** Agent A approaches the left field wall with its long axis along field X. Squares against the wall using the left ToF plus the left bump switch. Advances until the front bumper registers the corner region `[VERIFY]`. Left fingers swing clear. Robot translates in +Y, leaving the beam standing on the line.

**Step 2 — reposition.** Reverse, rotate 90°, approach the bottom field wall. Square using the front bumper plus the forward ToF.

**Step 3 — place the 250 mm beam with contact closure.** Advance along the bottom wall until the **contact microswitch** on the right pocket registers the leading end of beam 2 touching the already-placed beam 1. Stop immediately. Right fingers swing clear. Reverse in pure translation.

**Why contact detection rather than a rigid pre-formed L:** the acceptance criteria form a closed chain — each beam touches a wall and the two touch each other. Placed open-loop, beam 2 must be positioned relative to beam 1's *achieved* pose, and odometry drift of 12–24 mm over the approach exceeds the tolerance. The contact switch closes that loop directly and is robust to the unresolved seating geometry noted in §1.2. A rigid L-cradle would need no sensing but its bounding box would consume the entire deployment box, eliminating Agent B.

**No rotation during retreat.** Any yaw while withdrawing will clip a beam end and topple it.

### 5.6 Sensing

| Sensor | Qty | Location | Function |
|---|---|---|---|
| VL53L1X ToF | 2 | Left flank (Xa 200), forward-left 45° | Wall squaring, standoff |
| Bump microswitch | 2 | Front bumper, ±60 mm from centreline | Wall contact, corner detection |
| Bump microswitch | 1 | Right pocket, forward end | **Beam-to-beam contact closure** |
| Quadrature encoders | 2 | Drive motors | Odometry, gross navigation |
| IMU | 1 | Centre deck | Heading hold |

### 5.7 Timing budget (120 s)

| Phase | Duration |
|---|---|
| Traverse to quarantine, sweep pass 1 | 18 s |
| Sweep pass 2 | 12 s |
| Traverse to lab, square up, dispense 3 discs | 28 s |
| Return to quarantine, place beam 1 | 20 s |
| Reposition, place beam 2, withdraw | 22 s |
| Reserve | 20 s |

**Hard abort at T−40 s:** if the disc phase has not completed, abandon it and jump directly to beam placement regardless of magazine state. The 70-point task sits downstream of the 50-point task; without this timeout a single intake jam forfeits both.

---

## 6. AGENT B — kits and cylinders

Envelope **180 (lateral) × 265 (fore-aft) × 200 (tall)**. Local frame: origin at the rear-left corner at floor level. +Yb forward (up-field, +Y in field coordinates).

### 6.1 Layout by station

| Station | Yb position | Description |
|---|---|---|
| Funnel mouth | 235 → 265 | Front face |
| Feed roller | 215 | Above throat |
| Throat and colour station | 190 → 205 | Shrouded |
| Diverter gate | 175 | 3-position |
| Sort bins ×3 | 100 → 170 | Side by side across Xb |
| Drive axle | 110 | Track 145 |
| Kit magazine ×2 | 30 → 55 | Vertical tubes |
| Kit drop slot | 40 | Floor aperture |
| Battery + electronics | 55 → 100 | Low and central |
| Castor | 235 | Front |
| E-stop | 60, Zb 200 | Top deck |

### 6.2 Kit magazine (pure transport, zero sensing)

| Parameter | Value |
|---|---|
| Tubes | 2 vertical, 27 × 27 mm internal, 110 mm tall |
| Capacity | 5 kits each (5 × 20 mm = 100 mm stack) |
| Gate | 1 × MG90S per tube, sliding shutter, 22 mm stroke, one 20 mm cube per actuation |
| Discharge | Through a 30 × 60 mm floor slot at Yb 40, gravity drop from 6 mm |
| Xb positions | 55 and 125 |

Delivery pattern: 6 kits at the hospital, 2 at each PCC. Getting at least one kit into all three zones early is worth 30 points of avoided penalty and should be scheduled before any task that risks stranding the robot.

### 6.3 Cylinder intake, classification, and sorting

The only perception loop in the system. Note that the 12 sticker positions are **fixed and always occupied** — only the colour assignment randomises. This is waypoint navigation plus classification, not search.

| Stage | Specification |
|---|---|
| Funnel mouth | 90 mm wide × 40 mm tall at the front face, tapering over 70 mm to a 24 × 24 mm throat |
| Throat floor | Flush with field surface, 0.3 mm steel entry lip |
| Feed roller | Ø24 mm foam, N20 gearmotor, axis 18 mm high, draws one cylinder at a time rearward |
| Colour sensor | TCS34725, mounted looking down into the throat, 12 mm standoff, 20 × 20 mm sample window |
| Illumination | 2 × white LED inside the shroud, constant current |
| Shroud | Opaque 3D-printed enclosure, fully excluding ambient light |
| Calibration | White-reference read at t = 0 before the round begins |
| Classification | HSV hue thresholding, 3 classes (red / yellow / green), widely separated — no ML required |
| Diverter | 1 × MG90S rotating a 3-way gate at the throat exit, positions −35° / 0° / +35° |

**Shrouding is not optional.** The competition surface is specular white and will corrupt any unshrouded reading.

### 6.4 Sort bins

| Parameter | Value |
|---|---|
| Quantity | 3 (red, yellow, green) |
| Internal section | 24 × 24 mm |
| Height | 90 mm — capacity 4 cylinders each |
| Xb positions | 30, 90, 150 |
| Discharge gate | 1 × MG90S sliding shutter per bin (3 total), bottom discharge |

Discharge sequence: red bin at the hospital, yellow bin at a PCC, green bin at the recovery zone. The recovery zone lies inside the deployment box, so the final discharge returns the robot to its start position naturally.

### 6.5 Sensing

| Sensor | Qty | Location | Function |
|---|---|---|---|
| TCS34725 | 1 | Throat, shrouded | Cylinder classification |
| VL53L1X ToF | 2 | Front, right flank | Zone standoff, wall squaring |
| Bump microswitch | 2 | Front bumper | Wall and structure contact |
| Quadrature encoders | 2 | Drive motors | Odometry |
| IMU | 1 | Centre deck | Heading hold |

### 6.6 Timing budget (120 s)

| Phase | Duration |
|---|---|
| Traverse north, dispense 6 / 2 / 2 | 35 s |
| Side row 1 — collect 6 cylinders | 25 s |
| Side row 2 — collect 6 cylinders | 25 s |
| Three sorted discharges, return home | 30 s |
| Reserve | 5 s |

---

## 7. Localisation strategy

Wheel odometry drifts 1–2 % over a ~1.2 m path, giving 12–24 mm terminal error. The tightest placement requirement is 2 mm radial at the lab slots. Odometry is therefore used for gross navigation only, and **every terminal placement closes the loop on a physical feature.**

| Terminal | Reference method |
|---|---|
| Laboratory slots | Chamfered dispenser snout + wall square-up |
| Quarantine corner, beam 1 | Left wall bump-and-square |
| Quarantine corner, beam 2 | Bottom wall square-up + beam-to-beam contact switch |
| Hospital / PCC | Coarse zones (~180 × 300 mm) — odometry sufficient |
| Recovery zone | 20 mm black tape line-crossing fix |

---

## 8. Diagram brief

Produce the following sheets. Style: **orthographic engineering line drawing**, monochrome with a single accent colour used only for game elements (beams, discs, kits, cylinders). Dimensioned in millimetres with leader-line callouts. No perspective renders, no shading, no photorealism. Label every numbered station from the tables above.

**Sheet 1 — Deployment packing plan.** Top view of the 480 × 280 box with both robots to scale in their start positions. Show the two beams inside Agent A's pockets and the kit tubes inside Agent B. Dimension all envelopes, gaps, and boundary margins. Annotate the constraint that the 280 mm beam must lie along the 480 mm axis.

**Sheet 2 — Agent A, three views.** Top, front, and left side. Call out: disc scoop assembly, ramp angle, roller, magazine tube, dispenser snout, both beam pockets, retaining fingers, contact microswitch, drive axle, castor, battery, e-stop.

**Sheet 3 — Agent A, beam placement sequence.** Three-panel strip showing the quarantine corner in plan view: (1) beam 1 released against the left wall, (2) reposition and rotate 90°, (3) beam 2 advanced until contact closure, released, withdrawn. Show the 18.4° tip-over margin as an inset detail with the beam cross-section.

**Sheet 4 — Agent A, disc intake detail.** Section view through the scoop showing the 0.5 mm ramp at 12°, the Ø30 roller at 26 mm axis height, the 3 mm gap, a Ø56 × 5 mm disc mid-ingest, and the path into the stack tube.

**Sheet 5 — Agent B, three views.** Top, front, and right side. Call out: funnel mouth, feed roller, shrouded colour station, diverter gate, three sort bins, two kit tubes, floor drop slot, drive axle, castor, e-stop.

**Sheet 6 — Agent B, sorting flow.** Section view following one cylinder from funnel mouth through the throat, past the colour sensor, through the three-position diverter, into its bin. Adjacent block diagram: intake → classify → divert → buffer → discharge, with the three destinations labelled.

**Sheet 7 — Field operations plan.** Plan view of the full 1143 × 1181 field with all zones labelled. Overlay Agent A's path in one line weight and Agent B's in another. Mark the T−40 s abort decision point on Agent A's path.

---

## 9. Open items requiring physical mock-up

These must be resolved before manufacture. Build the quarantine corner, the laboratory plate, and a strip of the healthcare zone first.

1. **Beam seating geometry.** Whether the beams sit inside, on, or outside the 20 mm floor line. Determines whether 280 + 250 mm closes the perimeter and sets the contact-switch trigger position.
2. **Laboratory plate hole pitch.** Sets the dispenser snout position and the traverse increment between the three dispense events.
3. **Laboratory plate X/Y position** relative to the quarantine zone and the deployment box.
4. **Deployment box X offset** within the field width.
5. **Surface friction** on the actual competition material — affects sled drag, acceleration limits, and every open-loop segment.

## 10. Compliance checklist

- [ ] Both robots fit entirely within 480 × 280 mm with all game elements loaded
- [ ] Nothing crosses or extends beyond the black tape boundary at start
- [ ] Both robots fully autonomous after activation; no communication link to any operator
- [ ] Accessible, clearly identifiable emergency stop on each robot
- [ ] SDG 3 health-themed decoration on **both** robots
- [ ] No sharp edges, no loose parts, no exposed or unsecured battery, no unrestrained cabling
- [ ] Nothing placed on the field outside the deployment box; no off-robot infrastructure
- [ ] Design document and working-robot video prepared for submission
