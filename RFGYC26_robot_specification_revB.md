# RFGYC'26 Senior — Two-Robot System Specification — **REV B**

**Purpose:** a self-contained engineering brief for producing technical diagrams of a two-robot system for the ITU Robotics for Good Youth Challenge 2026–2027 (Senior category). It assumes no prior context. All dimensions in millimetres unless stated. Masses estimated from wood density 500–800 kg/m³ (not specified by the rulebook).

**Provisional dimensions** are marked `[VERIFY]` — to be confirmed on a physical field mock-up before manufacture.

---

## REVISION HISTORY — what changed from Rev A (drawing set RFGYC26-DWG-001…007)

| # | Change | Sheets affected |
|---|--------|-----------------|
| B1 | **New governing principle (§3): all discharges exit past the robot's rear face at floor level; deposit tail-in, depart nose-out.** Rev A's kit floor-drop and bottom-discharge bins are deleted — a 20 mm piece released under a 6 mm-clearance chassis strands the robot on its own payload. | 1, 5, 6, 7, new 8 |
| B2 | Agent B sort bins replaced by **three open-bottomed floor-level lanes**, push-fed at the front, flap-gated at the rear face. No vertical bin, no leaf retainer, no drop height. | 5, 6 |
| B3 | Agent B kit tubes keep gravity feed but release into **open-bottomed rear bays**; cubes are left behind on a forward creep, never under the chassis. | 5 |
| B4 | Agent B funnel mouth widened **90 → 160**; front face becomes almost entirely intake. Front castor replaced by **two rear-corner castors**. Front bump pads deleted (shoulders too thin); terminal references move to a **rear-mounted TCRT5000 line array** + front ToF. | 5, 6 |
| B5 | Agent B carries **two mission programs** (P1 single-loop, P2 kits-first) with a **T−50 s abort** in P1. | 7 |
| B6 | Agent A disc dispensing moves to the **rear face** (Rev A CLASH ⟨1⟩, option a). Snout as drawn in Rev A review: **ID 58 / OD 64** (OD 54 could not pass a Ø56 disc). Lab approached in reverse. | 2, 4 |
| B7 | Agent A track **175 → 162** (Rev A CLASH ⟨2⟩ — wheel encroached pocket walls by 6.5). | 2 |
| B8 | **Beam choreography corrected.** Rev A Sheet 3 contained an impossible lateral translation for a differential drive and a length paradox (a 250 beam flush-forward on a 285 robot cannot reach a wall the robot's rear hits first). Rev B: beams swap pockets, pockets become **open-ended**, and both separations are **axial**. Full sequence in §6.5. | 2, 3 |
| B9 | Rear bump pads added to Agent A (Xa 0, Ya ±60) — bottom-wall staging for beam 2 and lab docking datum. Retaining fingers gain **3 mm end-stop hooks** at each pocket's open end (accel/brake retention). | 2, 3 |
| B10 | IMU **BNO055 → MPU6050** (availability/cost, India); stationary bias calibration at t = 0; drift over 120 s ≤ 2°, irrelevant since every terminal closes on a physical feature. **TCRT5000 5-channel line arrays added to both agents** (shrouded). | 2, 5 |
| B11 | Rev A ⟨V⟩ resolutions adopted: disc stack tube datum Za 13→58; pocket L drawn full length; beam 1 set 5 short with chamfered end, beam 2 biased 5 inboard; front bump pads Za 45–60. | 2, 3, 4 |

---

## 1. Mission context (background for the illustrator)

Two robots operate together on a 1143 × 1181 mm walled table for one 120-second fully autonomous round. No human contact after start. Both robots begin entirely inside a 480 × 280 mm deployment box marked with 20 mm black tape.

| # | Task | Objects | Carried at start? |
|---|------|---------|-------------------|
| 1 | Deliver medical kits to three destination zones | 10 cubes, 25 × 25 × 20, ~9 g | **Yes** |
| 2 | Extract samples from a walled corner, post into a lab plate | 3 discs, Ø56 × 5, ~8 g | No — collected |
| 3 | Seal the corner with two free-standing beams | 280 × 60 × 20 (~200 g) and 250 × 60 × 20 (~180 g) | **Yes** |
| 4 | Collect and colour-sort patient markers | 12 cylinders, Ø20 × 20, ~5 g, red/yellow/green | No — collected |

**Task dependency:** the beams seal the corner containing the samples → extraction completes before beam placement; beam placement is the last operation in that quadrant.

### 1.1 Field reference frame

Origin at the inside bottom-left corner. X right (0→1143), Y up-field (0→1181). Walls 19–20 thick, 65–70 tall, all sides.

| Zone | Location |
|---|---|
| Quarantine | X 0–280, Y 0–280. Contains the 3 samples in random positions |
| Deployment box | 480 × 280, bottom, X-offset 643 `[VERIFY]` |
| Laboratory plate | 440 × 150 × 3 with three Ø60 through-holes, bottom-centre, hole pitch 140 `[VERIFY]` |
| Hospital (H) | Top-centre, against the top wall, zone depth ~180 |
| PCC ×2 | Top corners, against the top wall |
| Recovery zone (RZ) | Inside the deployment box |
| Patient stickers | 12 × Ø20 — two columns per side, two rows; column/row positions `[VERIFY]` (assumed rows Y ≈ 531 and 731; left columns X ≈ 150/250; right columns mirrored) |

### 1.2 Beam end state

Both beams finish **standing on a 20 face, 60 tall, free-standing, nothing touching them.** Beam 1 touches the left wall; beam 2 touches the bottom wall; the beams touch each other; the convex corner points into the field. Static tip-over angle **18.4°** (CG 30 over a 20 base).

Closure as drawn `[VERIFY]`: beam 1 from the left wall along Y = 250→270, spanning X 0→280. Beam 2 from the bottom wall along X = 280→300, spanning Y 0→250. Corner joint at (280, 250): beam 2's north end face meets beam 1's south side face near beam 1's east end. Sealed opening 280 × 250. 280 + 250 cannot close a 280 × 280 perimeter end-to-end; exact seating against the 20 floor line is mock-up item §10.1.

---

## 2. Architecture summary

Two independent differential-drive robots. No arm, no gripper, no suction, no camera, no lift anywhere. Every manipulation is **bulk intake into a floor-level buffer** or **gated release past a robot edge**. The only perception loop is 3-class colour classification.

| | Agent A | Agent B |
|---|---|---|
| Owns | Beams + samples | Kits + cylinders |
| Region | Lower-left quadrant | Top of field + side rows |
| Envelope L × W × H | 285 × 235 × 175 | 180 × 265 × 200 |
| Drive / aux motors | 2 / 1 (intake roller) | 2 / 1 (feed roller) |
| Servos | 3 (2 finger sets, 1 disc gate) | 6 (1 diverter, 3 lane flaps, 2 kit shutters) |
| Perception | Contact + ranging + line array | Same + 1 colour sensor |

**System totals:** 4 drive motors, 2 aux motors, 9 servos, 1 × TCS34725, 4 × VL53L1X, 2 × MPU6050, 2 × TCRT5000 arrays, 7 microswitches, 2 × ESP32-S3, 2 e-stops.

---

## 3. GOVERNING PRINCIPLE — discharge geometry (new in Rev B)

> **The front face ingests. The rear face discharges, at floor level, past the robot's edge. Deposits are made tail-in and departed nose-out. A robot never travels over anything it has placed.**

Rationale: game pieces are 20–25 tall; chassis clearance is 6. Any piece released under the chassis strands the robot on its own payload and rakes the piece on departure (Rev A kit floor-drop and bin bottom-drop — both deleted). Displacing a patient cylinder costs −3; pushing a kit out of a zone forfeits +3 and threatens the +20 distribution bonus.

**Standard deposit kinematic (draw once, Sheet 8):**
1. Approach the zone nose-out (facing away), reverse tail-first across the zone boundary to a depth of **150** (zone depth 180). Stop reference: rear line array on the zone tape + front ToF to the far wall.
2. Open the relevant rear gate(s).
3. **Creep forward 100–120 at ≤ 0.1 m/s.** Pieces rest on the field inside open-bottomed bays/lanes; ground friction holds them; they exit relatively through the open rear and trail from 150 to ~40 inside the boundary — all inside the zone.
4. Close gates, drive off forward. Nothing placed is ever under or ahead of the robot again.

The same open-bottomed, fence-and-leave mechanism family is used for the beams (Agent A pockets), the cylinders (Agent B lanes), and the kits (Agent B rear bays). One concept, three scales.

---

## 4. Deployment box packing plan

Box 480 (X) × 280 (Y), origin bottom-left of the box.

| Item | X | Y |
|---|---|---|
| Agent A envelope | 5 → 290 | 5 → 240 |
| Agent B envelope | 296 → 476 | 5 → 270 |
| Gap A–B | 6 | — |
| Margin to tape | 4–5 | 5–40 |

Both beams lie with their long axis parallel to the 480 axis (a 280 beam on the 280 axis has zero clearance — not permitted; annotate as in Rev A Sheet 1). **Beam 1 (280) rides in pocket R, flush with Agent A's front face. Beam 2 (250) rides in pocket L, flush with the rear face.** Beams and all kits are within the envelopes; lanes and bays are empty at start.

---

## 5. Common platform

| Subsystem | Specification |
|---|---|
| Chassis | Two-tier, 3–4 mm laser-cut acrylic (or 3 mm Al), decks at Z ≈ 6–10 and 50–54, M3 standoffs |
| Drive | 2 × 12 V metal-gear DC ~300 rpm with quadrature encoders (JGB37-520 class), Ø65 × 25 wheels, differential |
| Track | **Agent A 162** (Rev B7), Agent B 145 |
| Castors | A: Ø25 ball, rear. B: **2 × Ø25 ball, rear corners** (Rev B4) |
| Ground clearance | 6 |
| Driver / MCU | TB6612FNG · ESP32-S3 |
| IMU | **MPU6050** (Rev B10), stationary bias cal at t = 0 |
| Line sensing | **TCRT5000 5-channel array, shrouded**: A front underside, B rear underside (Rev B10) |
| Battery | 3S 11.1 V 1500 mAh, hard case, bolted |
| E-stop | Ø20 red mushroom, latching, top deck, breaks battery main — homologation-mandatory |
| Speed / accel | 0.45 m/s max; ≤ 0.6 m/s² with beams loaded; ≤ 0.1 m/s during any discharge creep |
| Decoration | Removable SDG 3 shell panels; visual inspection is a hard gate |

Surface: specular white whiteboard at the international final; MDF/paint/tarpaulin nationally; friction undefined and venue-dependent. All IR sensing shrouded; no open-loop terminal moves.

---

## 6. AGENT A — beams and samples

Envelope **285 × 235 × 175**. Local origin rear-right at floor; +Xa forward, +Ya left, +Za up. Deploys facing field −X (toward quarantine).

### 6.1 Station schedule

| Station | Xa | Ya / Za |
|---|---|---|
| Disc scoop | 250→285 | Ya 35→200 |
| Disc stack tube Ø62 int × 45 | 195→250 | Za 13→58 |
| **Disc dispenser snout (rear, Rev B6)** | **0→35** | duct from tube, ID 58 / OD 64, tip 8 above plate |
| Drive axle | 120 | **track 162** |
| Electronics / battery | 60→200 / 20→90 | Za 60→110 / 10→45 |
| Castor Ø25 | 35 | Ya 117.5 |
| **Pocket R — beam 1 (280), open at FRONT** | 0→285, beam flush front | Ya 0→24 |
| **Pocket L — beam 2 (250), open at REAR** | 0→285, beam flush rear | Ya 211→235 |
| Fingers (2 per pocket, 50 tall, MG996R common shaft, swing up + outboard) | R: 245, 30 · L: 250, 35 | with **3 mm end-stop hooks** at each open end (Rev B9) |
| Beam-contact microswitch | pocket L forward region | lever leads beam 2's leading face by 2 |
| Front bump pads ×2 | 285 | Ya ±60 of CL, Za 45→60 |
| **Rear bump pads ×2 (Rev B9)** | 0 | Ya ±60 of CL, Za 45→60 |
| ToF-1 / ToF-2 | left flank Xa 200, Za 75 / front 45° | wall standoff |
| TCRT array | front underside, shrouded | tape crossings |
| E-stop | 40 | Za 175 |

Pocket section as Rev A Section A-A: open-bottomed, open-sided, beam standing on the field, 0.5 slack per side; **plus open at one end per the schedule above.** Release is finger-clear followed by **axial separation only** — the beam exits through the pocket's open end as the robot drives along its own axis. Never lateral, never with yaw while a pocket overlaps a beam (18.4° margin, Rev A Detail C carries over unchanged).

### 6.2 Disc intake (unchanged from Rev A Sheet 4)

165 mouth · 0.5 shim ramp at 12°, leading edge 0.3 above surface · Ø30 × 165 roller, 8 silicone foam on Ø14 core, axis 26, 3 gap (5 disc = compression fit) · N20 + belt, ~1.2 m/s surface · two passes cover the 280 quarantine width — random placement needs no sensing.

### 6.3 Disc dispense (rear, Rev B6)

Gravity duct from the stack tube to a rear-face snout, ID 58 / OD 64, 45° chamfered lead-out, tip 8 above the plate, MG90S gate, one disc per stroke. **Lab approached in reverse**: rear pads square on the bottom wall, TCRT confirms plate-edge tape `[VERIFY]` , dock 3 times at the hole pitch. Chamfer absorbs 10 of position error; disc-in-hole clearance is 2 radial.

### 6.4 Beam assignment and the length arithmetic (why Rev B8 exists)

Robot length 285. Beam 1 = 280 flush-front in pocket R → with front pads on the left wall, beam 1 spans X 0→280 exactly; the robot's spare 5 sits aft. Beam 2 = 250 flush-rear in pocket L → with rear pads on the bottom wall, beam 2 spans Y ≈ 0→250 exactly; the robot's spare 35 sits forward, clear of the wall. *(A 250 beam flush-forward can never reach a wall — the robot's rear strikes it 35 short. This is the Rev A Sheet 3 paradox.)*

### 6.5 Beam placement choreography (Sheet 3 — redraw entirely)

Precondition: discs extracted (or T−40 abort fired). Quarantine floor is empty and drivable.

1. **Beam 1.** Robot faces west (−X) in the corridor Y ≈ 15–250 (body inside the quarantine footprint — permitted; nothing remains there). Front pads square on the left wall. Beam 1's west end face reaches the wall (flush-front, pads flush). Fingers R swing clear. **Reverse east, pure axial translation ≥ 320**; beam 1 exits the pocket's open front and stands at Y 250–270 `[VERIFY]`, X 0→280.
2. **Reposition.** Continue east to X ≥ 480, rotate 90° CCW to face north **at ≥ 200 from any placed beam** (robot half-diagonal ≈ 185), then translate to the beam-2 lane: west flank at X ≈ 302 (beam 2's outboard face lands at X = 280–300, biased 5 inboard per Rev B11).
3. **Stage.** Reverse south until **rear pads square on the bottom wall**. Beam 2's south end is now at the wall.
4. **Close.** Creep north at ≤ 0.05 m/s until the **pocket-L forward microswitch** (lever leading the beam face by 2) contacts beam 1's south face — stop, creep the final 2. Beam-to-beam and beam-to-wall are now both contact-confirmed by a single axial move, because 250 exactly spans the gap in the drawn closure.
5. **Release and depart.** Fingers L clear. **Advance north, pure axial translation**; beam 2 exits the pocket's open rear. Departure corridor X ≈ 302–537 clears the mid-field tape spine and the right sticker columns `[VERIFY]`. Park north-east of the sealed corner. **Nothing approaches either beam again.**

Corridor check `[VERIFY]`: the beam-1 approach keeps the body south of Y ≈ 250, well clear of the assumed left sticker rows at Y ≈ 531/731. Confirm sticker geometry at mock-up (§10.5).

### 6.6 Timing (120 s)

Sweep 1 + 2: 30 · lab (3 reverse dockings): 28 · beam 1: 18 · reposition + beam 2: 24 · reserve: 20. **Hard abort at T−40 s → jump to step 6.5-1 regardless of magazine state.**

---

## 7. AGENT B — kits and cylinders

Envelope **180 × 265 × 200**. Local origin rear-left at floor; +Yb forward (field +Y), +Xb right, +Zb up.

### 7.1 Station schedule

| Station | Yb | Xb / Zb |
|---|---|---|
| **Funnel mouth 160 × 40 (Rev B4)** | 235→265 | Xb 10→170, tapering over 70 to a 24 × 24 throat |
| Feed roller Ø24 foam, N20 | 215 | Zb 18 — one cylinder at a time |
| Shrouded colour station | 190→205 | TCS34725, 12 standoff, 2 × white LED, 20 × 20 window |
| 3-way diverter, MG90S | 175 | −35° / 0° / +35° |
| **Lanes ×3 — open-bottomed, 24 wide, floor level (Rev B2)** | 55→170 | Xb 18→42 (R) · 78→102 (Y) · 138→162 (G); radiused entries R8 |
| **Lane flap gates ×3, MG90S** | rear face 0 | one per lane |
| Kit tubes ×2, 27 × 27 int × 110, 5 cubes each | 30→57 | Xb 55 / 125; shutter MG90S, 22 stroke, one cube per stroke |
| **Kit rear bays ×2 — open-bottomed, rear-open (Rev B3)** | 0→28 | 30 wide, under each tube; cube lands on the field at 2 drop |
| Drive axle | 110 | track 145 |
| **Castors ×2 Ø25 (Rev B4)** | 15 | Xb 20 / 160 |
| **TCRT array (Rev B4)** | rear underside, shrouded | zone-tape and RZ-tape fixes, tail-first |
| ToF ×2 | front centre-high / right flank | top-wall standoff, wall follow |
| Battery + electronics | 58→100 | Zb 10→50 |
| E-stop Ø20 | 60 | Zb 200 |

Rear-face budget: 3 × 24 lane gates + 2 × 30 kit bays = 132 of 180 — fits with webs between.

### 7.2 Sorting chain

Intake (160 mouth → 24 throat) → classify (roller stopped while the read settles; HSV hue, 3 classes, white-reference at t = 0) → divert → **lane buffer**: each cylinder rides the field surface, pushed aft by the queue (push force ≈ 0.06 N — the roller does not notice) → **discharge** by the standard §3 kinematic. One cylinder in the throat at a time. Capacity 4 per lane equals one colour exactly — a mis-classification overfills one lane; log hue margins in testing. **Jam risk moved from the Rev A leaf retainer to the lane entry: radius entries generously (R8) and run 50 ingests per lane on the mock-up (§10.6).**

Yellow 2-2 split (worth +8): meter by creep distance — flap open, creep 45 (two cylinders at ~21 pitch), flap shut. P2 instead puts all four yellows at one PCC (no penalty, forfeit +8).

### 7.3 Two mission programs (Rev B5) — both uploaded, chosen per round

**P1 — single loop (high ceiling, primary).** Start → NE lane → right sticker columns, collect 6 → cross the mid-field tape → left columns, collect 6 → PCC-left: tail-in, 2 kits + 2 yellows → Hospital: tail-in, 6 kits + 4 reds → PCC-right: tail-in, 2 kits + 2 yellows → south along the east lane → RZ: tail-in on the box tape, 4 greens → creep forward 100, park. One visit per zone; every deposit tail-in/nose-out; nothing revisited. Budget: collect 55 · four deposits + transit 55 · reserve 10. **Hard abort at T−50 s: jump straight to the deposit sequence with whatever is aboard — the kits are aboard regardless, so the −30 empty-zone penalty is always killed.**
**P2 — kits-first (guaranteed floor, fallback).** North pass depositing 6/2/2 tail-in at three zones (35 s), then collect rows (50 s), then three colour deposits + home (30 s). Second visits land at the opposite end of each ~300-wide zone from the first deposit — no overlap with placed pieces.

Best-single-round ranking: run P1 in round 1, choose by result thereafter.

---

## 8. Localisation

Odometry drifts 12–24 over ~1.2 m; tightest requirement is 2 radial at the lab. Odometry is gross navigation only.

| Terminal | Reference |
|---|---|
| Lab slots | Rear pads on bottom wall + TCRT + chamfered snout |
| Beam 1 | Front pads on left wall |
| Beam 2 | Rear pads on bottom wall + beam-contact switch |
| H / PCC deposits | Rear TCRT on zone tape + front ToF to top wall, tail-in to 150 |
| RZ | Rear TCRT on box tape |
| Mid-field | TCRT line-crossing fixes on the 20 tape; IMU heading hold |

---

## 9. Diagram brief — Rev B drawing set

Same conventions as Rev A (orthographic, mono + one accent for game elements, mm dimensions, ⟨V⟩ flags, title blocks; REV field = **B**). Redraw:

**Sheet 1 — Packing plan.** Update: beam-to-pocket assignment (beam 1/R flush-front, beam 2/L flush-rear), Agent B rear-corner castors, rear gate cluster, 160 mouth. Keep the beam-orientation constraint inset.
**Sheet 2 — Agent A GA.** Update: rear dispenser duct and snout (clash resolved — remove phantom), track 162, rear pads, open-ended pockets with end-stop hooks, TCRT array, MPU6050. Section A-A gains the open end + hook.
**Sheet 3 — Beam placement.** Redraw entirely per §6.5: five panels (place 1 axial-exit · reposition with 200 rotate-clearance circle · stage on rear pads · creep-to-contact · axial depart north). Keep Detail C (18.4°) unchanged. Annotate the length arithmetic of §6.4.
**Sheet 4 — Intake & dispense.** Section B-B unchanged; Detail D moves to the rear face, reverse-docking shown against the bottom wall.
**Sheet 5 — Agent B GA.** Major redraw: 160 mouth, three open-bottomed lanes with radiused entries, rear flap gates, kit tubes over rear-open bays, twin rear castors, rear TCRT. Detail E becomes tube-over-bay.
**Sheet 6 — Sorting flow.** Section C-C: cylinder path entirely at field level end-to-end; lane replaces bin; add the discharge creep. Chain becomes intake → classify → divert → lane → tail-in discharge.
**Sheet 7 — Field operations.** Two overlays: P1 and P2 (differing line styles), tail-in deposit stops marked at each zone, abort markers T−40 (A) and T−50 (B), Agent A corridors from §6.5 with the sticker-clearance note.
**Sheet 8 (new) — Deposit kinematic.** The §3 standard deposit as a four-panel side-view strip at a healthcare zone: tail-in to 150 → gates open → creep 100–120 with the piece trail dimensioned inside the boundary → nose-out departure. One inset: why under-chassis release fails (20 piece vs 6 clearance).

---

## 10. Open items — physical mock-up (build the quarantine corner, lab plate, one healthcare strip, one sticker column first)

1. Beam seating vs the 20 floor line; corner joint geometry at (280, 250); contact-switch trigger offset.
2. Lab plate hole pitch (140 assumed) and X/Y position; whether a tape feature exists for the reverse dock.
3. Deployment box X-offset (643 assumed).
4. Surface friction per venue — creep speeds, sled drag, accel limits.
5. Sticker column/row coordinates — Agent A corridor clearance (§6.5) and P1 collection order.
6. Lane-entry jam rate: 50 ingests per lane, radiused entries; hue-margin log across venue lighting.
7. Rear-pad flushness vs beam 2's wall end; front-pad flushness vs beam 1's wall end.

## 11. Compliance checklist

- [ ] Both robots + all game elements entirely within 480 × 280 at start; nothing beyond the tape
- [ ] Fully autonomous after activation; no operator link
- [ ] Accessible, identifiable e-stop per robot; SDG 3 decoration per robot
- [ ] No sharp edges, loose parts, or unsecured battery/cabling
- [ ] Nothing placed on the field outside the deployment box; no off-robot infrastructure
- [ ] Robots never touch a placed beam after release; final parks clear of the sealed corner
- [ ] Design document + working-robot video prepared for submission
