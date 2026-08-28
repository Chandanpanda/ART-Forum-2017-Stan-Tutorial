# RFGYC'26 Senior — MuJoCo Simulation Plan

Simulate the two-robot Rev C system (Agent A + Agent B) on the ITU Robotics for Good Youth
Challenge 2026–27 Senior field. Supersedes the PyBullet draft: MuJoCo is a better fit, and the
reason is specific rather than general.

---

## 0. Why MuJoCo — verified, not assumed

Everything in this section was checked against `mujoco==3.12.0` installed from PyPI, not recalled.

**License.** Apache 2.0. The wheel ships the full Apache text in
`mujoco-3.12.0.dist-info/licenses/LICENSE`. DeepMind open-sourced MuJoCo in 2022; `pip install
mujoco` is the first-party binding, actively developed. PyBullet's license (Zlib) was never the
problem — its maintenance status was.

**MuJoCo has a native conveyor.** This is the decisive fact. `<geom surfacevel="...">` is a
6-vector (linear + angular) surface velocity in the geom's local frame, auto-enabled when any geom
sets it (`mjModel.flg_surfacevel`). The entire "there is no belt, so inject contact forces every
step and hope" section of the PyBullet plan collapses to one attribute:

```xml
<geom name="belt" type="box" size="0.096 0.058 0.002"
      surfacevel="0.060 0 0  0 0 0" friction="0.6 0.005 0.0001" condim="6"/>
```

Checked: a 5 g Ø20 × 20 cylinder placed on an 11° belt running at 60 mm/s is carried up-slope at
**59.2 mm/s** — 0.8 mm/s of slip, which is what µ 0.6 against tan 11° = 0.19 predicts. Four
cylinders queued against a closed gate settle and hold, with **0.131 N** of normal force on the
gate against the spec's hand-calculated 0.116 N (§3.2). The design's central claim — passive
accumulation with no actuator and no sensor — reproduces out of the box.

**Sensors are declared, not coded.** MuJoCo's `<sensor>` block covers almost the entire sensing
suite that the PyBullet plan was going to hand-roll from `rayTest`:

| Rev C hardware | MuJoCo sensor | Checked |
|---|---|---|
| VL53L1X ToF ×4 | `rangefinder` | ✓ returns 0.2900 for a wall at 290 mm |
| MPU6050 ×2 | `gyro` + `accelerometer` (both take a `noise` attribute) | — |
| TCRT5000 line array | `tactile`, or `rangefinder` rays onto tape geoms | — |
| Beam-contact microswitch | `touch` | — |
| TMC2209 StallGuard | `actuatorfrc` / `jointactuatorfrc` saturation | — |
| Camera ROI timing | `campprojection` | — |
| **Referee zone tests** | **`insidesite`** | ✓ returns 1 inside / 0 outside |
| "Untouched at the buzzer" | `contact` | — |

`insidesite` is the happy surprise: *"is this piece inside this zone"* is a built-in sensor type, so
most of the scoring module becomes MJCF declarations rather than point-in-polygon code.

**Contact model.** `condim=6` gives sliding, torsional and rolling friction as first-class per-geom
parameters (`friction="slide torsion roll"`), which is exactly what a 5 g wooden cylinder on a
whiteboard needs. Contacts are soft via `solref`/`solimp`, and `margin`/`gap` are per-geom — there
is no global collision margin to fight. **The Froude-scaling fallback from the PyBullet plan is
dropped entirely**; the tests above ran in true SI at `timestep=0.0005`.

**MJX for the Monte Carlo.** `mujoco-mjx` (separate install, pulls JAX) runs batched physics on
GPU/TPU, turning Phase 4's N ≥ 500 rounds into N ≥ 10,000 in the same wall clock. Two caveats:
MJX supports a subset of MuJoCo, so **Tier 1 must stay on primitive geoms**; and I could not verify
whether MJX implements `surfacevel` (it ships separately and was not installed here). That does not
matter — Tier 1 abstracts manipulation and never needs the belt. Confirm before relying on it.

**What we give up.** Nothing that matters here. URDF ecosystem reuse is irrelevant because the model
is generated from a parameter file either way, and MJCF is the better generation target: it has
`<default>` class inheritance, `<replicate>`, includes, first-class `<sensor>` and
`<contact><pair>`, none of which URDF can express.

---

## 1. Source map

| Artefact | Commit | Standing |
|---|---|---|
| `Robotics for good youth challenge.pdf` | `a9288be` | Rulebook — external authority |
| `RFGYC26_robot_specification_revC_1.md` | `8956e65` | Authoritative internal spec |
| `RFGYC26 Robot Drawing Set revC.dc.html` (9 sheets) | `8956e65` | Authoritative geometry |
| `RFGYC26 Mechanism Explainer.dc.html` | `8956e65`, upd. `7fb8ca9` | **Newest** — explicitly supersedes spec §6.2–§6.3 on the diverter and lane walls |
| `RFGYC26 Explainer.dc.html` | `8956e65` | Newest — only numeric source for field zone polygons |
| Rev A/B spec, Rev A/B drawing set, `*.png` | `37faa2c`, `a9288be`, `1601df1` | Superseded — ignore |
| `support.js`, `doc-page.js` | `37faa2c` | Rendering runtime, no robot logic |

The repository's `0_welcome/` … `6_about_stan/` trees are an unrelated Stan tutorial sharing the
repo.

---

## 2. Resolutions

Ten contradictions, all resolved. Each resolution states the reasoning, because the reasoning is
what makes it reviewable — and three of them turned up only while resolving the other seven.

### R1 — Belt incline: adopt 11° on both agents; the tail height is derived

**The conflict.** Nose belt-top Z 17.5, tail Z 51, Agent B run 192. At 11°, 192 mm rises 37.3 mm,
putting the tail at Z 54.8. Sheet 2's "rises 33.5 over the run" is self-consistent only for Agent
A's 175 run. The Mechanism Explainer silently uses gradient `0.1745` — and arctan(0.1745) =
**9.90°**, which is the radian value of 10° almost certainly written where `tan(11°)` was meant.

**What is actually free.** The nose height is not negotiable: a Ø16 roller resting on the floor puts
its axis at Z 8, so belt top = 8 + 8 + 1.5 = 17.5. The runs are fixed by the station schedules
(A 175, B 192), which are in turn fixed by the scoop consuming the front 65/48. **Incline and tail
height are therefore a single degree of freedom** — pin one, derive the other.

**Decision: pin the incline at 11°.** The decisive argument is the common-chassis claim. Pinning the
tail at Z 51 on both agents forces *different inclines* (A 11.0°, B 9.9°), which means different
belt frames, slide plates and roller mounts — the "one chassis, built twice" premise breaks. Pinning
11° keeps the chassis genuinely common and pushes the difference into the tail height, and the
**tail funnel is already a per-agent printed part** (§2). The friction argument survives either way
(a shallower belt has *more* carry margin, not less), so it does not decide anything.

**Adopt:** incline 11° both agents. Agent A tail belt-top **Z 51.5** (lift 34.0), Agent B tail
belt-top **Z 54.8** (lift 37.3).

**Propagated.** Agent B's discharge drop becomes 54.8, not 51. Throw at 60 mm/s is
60 · √(2 · 0.0548/9.81) = **6.3 mm**, against 6.2 mm at Z 51 — so the 5–15 mm landing band survives
both readings and this contradiction turns out to be low-risk downstream. It does change the tail
funnel geometry and the rear-face clearance for a 20 mm kit, which is why it still has to be settled.
Fix `0.1745` → `0.19438` in the Mechanism Explainer, and §2's "buys 53 mm of height at the tail"
(a typo for 33.5) → 34.0 / 37.3.

### R2 — Diverter: adopt the vane, pivoted at two-thirds

**The conflict.** Spec §6.2 and Sheet 6 describe a 90 mm plow blade pivoted at its upstream end at
Yb 152, deflecting 43 mm at 25°. The Mechanism Explainer describes a 66 mm vane pivoted at its
middle, angles 0/±12/±25°, tip throw ±18.6.

**Decision: the vane.** A 90 mm blade pivoted at its upstream end at Yb 152 sweeps Yb 152 → 62,
consuming 90 of the 107 nominal lane run and leaving 17 mm of lane. That is unbuildable, so the
spec's version is simply out of date.

**But the vane as stated is not self-consistent**, and this resolves it: a 66 mm blade pivoted at its
exact middle throws 33 · sin 25° = ±13.9, not ±18.6. **Pivot it at two-thirds instead — upstream arm
44, downstream arm 22.** Then 44 · sin 25° = **18.59**, reproducing the stated ±18.6 exactly, while
preserving the stated behaviour ("the upstream arm catches the piece and starts steering it while
the downstream arm swings the other way"). A mid-pivot cannot produce the number the same document
quotes; a 2/3 pivot produces it to three significant figures.

**Adopt:** vane 66 long, pivot Yb 152 at two-thirds (44 upstream / 22 downstream), four held angles
0 / ±12 / ±25°, tip throw ±18.6. Swept envelope **Yb 132–192**.

*Source update (`7fb8ca9`).* The Mechanism Explainer now carries an explicit note — "the 66 vane and
the lane walls from Yb 108 supersede §6.2–§6.3; Sheets 5 and 6 still show the spec pivot and need
reconciling" — which confirms the direction of this resolution. **The mid-pivot inconsistency
itself is unchanged**: the file still reads "pivoted at its middle" beside a ±18.6 tip throw, which
a mid-pivot cannot produce. The two-thirds split stands as the only reading that reconciles them.

### R3 — The vane sets a trajectory; the lane noses finish the move

**The conflict.** Belt centreline Xb 90; lane centres L1 49, L2 79.5, L3 108, L4 136.5. Green needs
46.5 mm of lateral travel against a quoted 43 mm maximum — and under R2 the vane only throws ±18.6,
reaching Xb 71.4–108.6. It can aim directly at L2 and L3 and at neither L1 nor L4.

**Decision: the vane is a trajectory-setter, not a placer.** The piece leaves the vane with a lateral
velocity component `v_lat = v_belt · tan θ` and keeps drifting across the belt until a lane
divider's pointed nose intercepts it. This is what the Mechanism Explainer means by "the lane walls
start aft of its swept envelope and their pointed noses finish the lateral move" — that sentence is
load-bearing, not decorative.

**The geometry closes, but barely.** Drift rate at 25° is tan 25° = 0.466 mm lateral per mm of belt
travel. Reaching L4 needs 136.5 − 108.6 = 27.9 mm more lateral, hence **59.9 mm of belt travel**.
The fan zone (throat exit Yb 172 → divider noses Yb 108) is 64 mm, giving **29.8 mm of drift against
27.9 needed — 1.9 mm of margin.** L1 is comfortable (22.4 mm needed, 48 mm of travel).

**Adopt:** staggered divider noses — L1 and L4 noses furthest aft at Yb 108 (they need the whole fan
zone; `7fb8ca9` **independently confirms Yb 108** as the wall line, and its withdrawn "fan zone
91–157" annotation converts to Yb 108–174 against the 108–172 derived here), L2 and L3 noses further forward at ≈ Yb 125 (they need less drift and gain jam protection
from earlier walls). Model the noses as wedges. **This is now the headline correctness experiment**,
displacing the convergence wedge, which remains the headline *jam* experiment.

### R4 — Lane run is 63; kits move to two tandem tubes

**The conflict.** Spec §6.3 and Sheet 5 give the lane run as Yb 152 → 45 = 107. The Mechanism
Explainer says 63.

**Decision: 63**, unchanged and re-confirmed by `7fb8ca9` (lane run 63, three cylinders on the belt,
the fourth in the flare). The spec measured from the vane pivot and ignored the swept envelope. Under R2/R3
the numbers close exactly: fan zone Yb 172 → 108 (64) plus dividered lane Yb 108 → 45 (63) equals
the 127 available from the throat exit to the gate. The spec's 107 double-counts the fan zone.

**Consequence A — cylinder lanes hold 3, not 4.** 3 × 20 = 60 ≤ 63; a fourth queues back in the fan
zone, which is exactly the Explainer's "the 4th waits in the flare". With randomised colours a
single side of the field can hold four of one colour, so this needs a firmware guard, not a hope:
**Agent B counts pieces per lane — it knows every class — and triggers an early discharge leg the
moment any lane reaches 3.** Costs nothing.

**Consequence B — L1 cannot stage 4 kits.** 4 × 25 = 100 > 63. And the spec's own tube capacity is
off by one: Zb 60–170 is 110 mm of usable height, which holds **5** kits at 20 mm, not 6. So the
stated 4-on-belt + 6-in-tube arrangement is short on both halves.

**Adopt: two tandem escapement tubes in L1, 5 kits each, nothing staged on the belt.** Both 27 × 27
at Zb 60–170, in series along Yb inside L1's 63 mm run (54 mm of tube footprint, ~9 mm to the gate —
confirm the packing in CAD). This restores what Rev A/B already had ("KIT MAGAZINE ×2 — 5 KITS
EACH") and costs one extra MG90S shutter: **Agent B goes 7 → 8 servos, system total 11 → 12.**

### R5 — Believe the rulebook: sweep the ground zones, don't visit waypoints

**The conflict.** The spec assumes "twelve stickers in fixed positions, colour randomised — waypoint
navigation plus classification, never search." The rulebook says three separate times (§b Step 2,
§d.2.8, §h.3.2) that for Senior the *position* is randomised each match.

**Decision: believe the rulebook.** The asymmetry is brutal — a wrong assumption here costs the whole
80-point mission, and the rulebook is the external authority. But the problem is bounded: still 12
stickers, 6 per side, inside the two side ground zones.

**And the fix costs nothing, because Agent B never needed to see them.** Replace the four column
descents with a **boustrophedon sweep of each side ground zone**. The sweeper is default-open at a
165 mm capture band, so two passes at 165 pitch cover a ~200 mm strip exhaustively regardless of
where the pieces sit. Triage happens on the belt, so pickup order is irrelevant. Two passes per side
is comparable in distance to the two column descents P1 already drives.

This promotes the design's own "walk-over insurance in the collecting direction" from a safety net
to the primary strategy — no new hardware, no vision, and it removes the main fragility in the P1/P2
split at the same time.

### R6 — Beam 1 centre is Y 260

**The conflict.** Spec §1 places beam 1 along Y 250–270 (centre 260); the Explainer draws it at
centre 252.5.

**Decision: the spec, and it is self-proving.** Beam 2 is 250 long from the bottom wall along
X 280–300, so its north end face lands at Y 250. Beam 1's south face must butt it there, so beam 1
spans Y 250–270. **Centre Y 260.** With beam 1 "set 5 short with a chamfered end" it spans X 0–275,
centre X 137.5. Beam 2 centre is **(290, 125)** — the Explainer's 127 is the same class of slip.

### R7 — Plan to −5 for a misplaced patient

**The conflict.** The Senior scoring table says −3; penalties §j.4.4 says −5.

**Decision: plan to −5, report both, ask the referees at registration.** §j.4.4 introduces itself as
"mission-specific penalties for the Senior category, *already included in the Game procedure
section*" — so the two are meant to match and one is a transcription error. §j is the section whose
job is penalties and whose other Senior figures are internally consistent. Planning to the harsher
number is the safe asymmetry.

**This reframes the sensing budget.** At −5, a misclassified marker is a **10-point swing** (+5
forfeited, −5 incurred). Across 12 markers that is 120 points of classification-dependent swing
against a 250-point maximum. **Classification accuracy is the highest-leverage number in the
system** — which retroactively justifies keeping the TCS34725 as an arbiter alongside the camera,
and makes "refuse to classify" a strategy worth simulating rather than a failure state. See R10.

### R8 — Kit escapement tube capacity is 5, not 6

Surfaced while resolving R4. Zb 60–170 = 110 mm ÷ 20 mm per kit = 5. Folded into R4's two-tube
resolution.

### R9 — "Buys 53 mm of height at the tail" is a typo

Surfaced while resolving R1. §2 says 53; Sheet 2 says 33.5 (= 51 − 17.5). Under R1's adopted 11° the
correct figures are **34.0 (Agent A) and 37.3 (Agent B)**.

### R10 — The wrong-zone penalty has two incompatible wordings

Surfaced while resolving R7, and it is worth more than it looks. The scoring table penalises a
patient *"outside their correct destination zone"* — any location, including still aboard the robot.
§j.4.4 penalises a patient *"placed in an incorrect destination zone"* — which requires placement in
a zone at all.

Under §j.4.4's wording, **retaining an unclassifiable marker on board is free** and strictly better
than guessing (0 rather than −5). Under the table's wording it costs the same as misplacing it, and
the optimal play is to always guess. These are opposite strategies, so this is not academic.

**Adopt:** implement both as a ruleset flag; make the default the pessimistic reading (retention is
penalised); and **put this to the referees as a direct question** — "does a marker still aboard at
the buzzer incur the penalty?" It is the highest-value question on the list.

---

## 3. Field and scoring reference

Field 1143 (X) × 1181 (Y), walls 19–20 thick × 65–70 tall, origin bottom-left inside the walls.
These polygons exist as numbers in exactly one place — the Explainer's canvas code.

| Zone | Extent (mm) | Zone | Extent (mm) |
|---|---|---|---|
| Quarantine | X 0–280, Y 0–280 | Hospital | X 471.5–671.5, Y 901–1181 |
| Lab plate 440×150×3 | X 351.5–791.5, Y 360–510 | PCC-L / PCC-R | X 0–200 / 943–1143, Y 981–1181 |
| Lab holes Ø60 `[V]` | X 431.5/571.5/711.5, Y 372 | Recovery zone | X 700–900, Y 190–270 |
| Deployment box `[V]` | X 643–1123, Y 0–280 | Stickers ×12 `[R5]` | randomised in the side ground zones |
| Beam 1 final `[R6]` | 280×20, centre (137.5, 260) | Beam 2 final `[R6]` | 20×250, centre (290, 125) |

Pieces: 3 discs Ø56 × 5 (~8 g); 10 kits 25 × 25 × 20 (~9 g); 12 cylinders Ø20 × 20 (~5 g), 4 each
red/yellow/green; beam 1 280 × 60 × 20 (~200 g), beam 2 250 × 60 × 20 (~180 g).

Senior scoring: sample in a slot +15 ea, all three +5, stranded sample −3 ea, sample left in
quarantine −5 ea; beam placed +25 ea, perimeter closed +20; kit in a valid area +3 ea, correct 6/2/2
+20, empty destination zone −10 ea; patient in the correct zone +5 ea, all red in H +6, yellow split
+8, all green in RZ +6, wrong zone −5 `[R7]`. Round penalties −20 each for touching the robot,
manipulating the field, restarting the program, or leaving the field.

**Maximum 250** — samples 50, beams 70, kits 50, patients 80. Ranking is **best single round**, so
any Monte Carlo reports the upper tail, not the mean.

---

## 4. Two simulators, one geometry source

The match sim and the mechanism sim have incompatible requirements. Build two, joined by a
deliberately narrow interface.

| | Tier 1 — Match sim | Tier 2 — Mechanism rig |
|---|---|---|
| Scope | Whole field, both agents, 27 pieces | One agent, a 600 mm floor patch, one station |
| Timestep | 0.002–0.005 s | 0.0005 s (verified stable) |
| Geoms | Primitives only, for MJX compatibility | Anything |
| Manipulation | Abstracted — capture / hold / release | Simulated in contact |
| Answers | Routes, timing, aborts, deconfliction, score spread | Jam rate, pickup rate, drift accuracy, landing band |
| Target | ≥ 20× real time; batched under MJX | 0.2× real time is fine |

The bridge is a small table of empirical rates measured in Tier 2 and consumed by Tier 1 —
`p_capture(speed)`, `p_jam(arrival_rate)`, `p_lane_correct(colour)`, `t_gate_to_floor`,
`landing_offset` mean and σ. That is the entire coupling.

**Phase 0 is one geometry source.** A single `params.yaml` holding every dimension, with a `VERIFY`
flag on each value the spec tagged and each resolution above, and **assertions that re-derive the
over-determined quantities and fail loudly**: belt rise against run against incline (R1), vane throw
against pivot split (R2), drift budget against fan-zone length (R3), lane capacity against piece
count (R4), tube capacity against height (R8), the width budget (116 + 44 + 20 = 180 exactly), and τ
as swept diameter over body length. MJCF is generated from that file, never hand-written.

---

## 5. What to simulate, what to declare, what to stub

**Simulate in contact** — these carry the design's arguments:

- **Belt** — `surfacevel` on the belt geom (§0). Both agents, 11° (R1).
- **Scoop** — a real hinge with a torsion spring and a 25° limit. Sheet 4's whole argument is that it
  trips up when reversed, over the 20 tape and the 3 lab plate.
- **Diverter vane and staggered lane noses** (R2, R3) — the tightest margin in the design.
- **Convergence wedge** — asymmetric, one wall at 22°. The spec's declared #1 jam risk.
- **Ball transfers ×4** — low-friction spheres, **never swivel castors**; Sheet 3 rejects castors
  because they must be dragged into alignment, and modelling them as castors reproduces a failure
  the design engineered out.
- **Chassis** — chamfered hull, not a plain box; the whole turning-ratio argument depends on the
  40 × 45° chamfers.
- **Beams** — the one manipulation that cannot be abstracted. Agent A never lifts them: they stand
  on the field, dragged in open-bottomed pockets, released by axial separation only. The scoring
  criterion — *"standing upright on its long side, stable by itself, fully released"* — is directly
  checkable from final tilt, settled velocity, and a `contact` sensor against both robots. The
  tip-over margin is atan(20/60) = 18.43°, and the release transient is where it gets spent.

**Declare as sensors** — see the table in §0. `rangefinder` for ToF, `gyro`/`accelerometer` for the
IMU, `touch` for the beam-contact lever, `actuatorfrc` saturation for StallGuard, `insidesite` for
every referee zone test, `contact` for "untouched at the buzzer".

**Stub analytically:**

- **Steppers** — wheel control quantised to 0.47 mm/full-step, with an **explicit step-loss model**.
  The spec calls a skipped step "silent", so inject skips under acceleration and measure the damage.
- **StallGuard** — honour "unreliable below ~0.1 m/s" by disabling detection under that speed.
- **Camera + TCS34725** — ground-truth colour in the tunnel ROI plus a confusion matrix and latency.
  Given R7, sweep the misclassification rate hard; it is the highest-leverage parameter in the model.
- **Servos** — MuJoCo `<position>` actuators with `forcerange` and a first-order lag (~0.4 s sweeper
  cycle).

**The gate interlock belongs in the driver.** At most one lane gate open at any instant — the spec is
emphatic that this is a hardware-style interlock, not a route-script convention, because the shared
discharge funnel is only free of cross-contamination because of it. Implement it as a mutex in the
gate abstraction and assert on it every step. R4 adds a second interlock: **the kit escapement may
not fire while a marker is in the fan zone.**

**Three free audits** that catch exactly the class of bug that costs a round: flag any yaw command
whose swept R 185 circle intersects a placed beam (Agent A runs τ = 1.30 and must rotate only at
≥ 200 mm from a placed beam); flag any release under a chassis; flag any path over a placed piece.

---

## 6. Validation — the spec grades its own homework

| Assertion | Status |
|---|---|
| Retarding force ≈ 0.03 N/piece, 0.12 N for four | **✓ measured 0.131 N against 0.116 N predicted** |
| Carry margin µ 0.6 vs tan 11° = 0.19 | **✓ measured 0.8 mm/s slip at 60 mm/s** |
| Beam tip-over 18.4° | ✓ atan(20/60) = 18.43° analytically |
| Disc exit ~0.4 m/s from an 8 mm drop | ✓ √(2·9.81·0.008) = 0.396 m/s |
| 0.47 mm/full-step, 0.36°/step | ✓ π·150 / 0.47 = 1002 steps per 360° |
| Landing band 5–15 mm behind the rear face | ✓ 6.3 mm under R1's adopted 11° |
| τ = 1.07 (B chamfered) / 1.30 (A beam-loaded) | to measure — sweep the swept circle |
| Drift budget reaches L4 | **1.9 mm of margin (R3) — headline correctness test** |
| Convergence wedge jam rate, 50 ingests per lane | headline jam test — Monte Carlo doubled arrivals |
| Lane capacity 3 cylinders, kit tubes 5 each | resolved by R4/R8; confirm the packing in CAD |

---

## 7. Phasing

**P0 — Geometry truth, no physics.** `params.yaml`, the MJCF generator, the assertions, a top-down
render. Land all ten resolutions as parameters with their `VERIFY` flags.
*Deliverable: a rendered field matching Sheet 9, and a failing assertion for every unresolved number.*

**P1 — Tier 1 skeleton.** Field, walls, zones, 27 pieces, both chassis with driven wheels and ball
transfers. Manipulation abstracted. Referee wired to `insidesite` sensors.
*Deliverable: a scripted 120 s match that scores.*

**P2 — Navigation realism.** Stepper quantisation and step-loss, StallGuard wall squaring, ToF, tape
crossings, IMU heading hold. Replace the Explainer's hand-authored waypoint tables with a route
executor that closes every terminal on a physical feature — and with **R5's boustrophedon ground-zone
sweeps** in place of column descents.
*Deliverable: the T−40 and T−50 aborts exercised under odometry drift, against randomised sticker positions.*

**P3 — Tier 2 mechanism rig.** In risk order, which R2–R4 have reshuffled: **diverter drift accuracy
first** (1.9 mm of margin), then the convergence wedge, then scoop pickup, then belt accumulation
(already partly validated in §0), then discharge and landing band, then Agent A's chute and beam
release.
*Deliverable: the empirical rate table, and answers to spec §10 items 4, 6, 7, 8, 10.*

**P4 — Monte Carlo under MJX.** Randomise sticker positions and colours, venue friction,
misclassification rate, step loss. N ≥ 10,000 if MJX is available, N ≥ 500 otherwise.
*Deliverable: the 90th-percentile score, and the top three failure modes ranked by points lost.*

**P5 — Firmware in the loop (optional).** Swap the Python route executor for the real ESP32 state
machine behind a shim.

---

## 8. Risks

- **The diverter drift margin is 1.9 mm** (R3). This is now the thinnest number in the design and it
  only exists because R2 and R4 forced the vane's real reach into the open. If P3 shows it does not
  close on the real surface, the fix is more fan zone — which costs lane run, which costs lane
  capacity, which R4 has already made tight. Test it first.
- **Two firmware interlocks are load-bearing**, not conventions: one gate at a time, and no kit
  escapement while a marker is in the fan zone. Both belong in the driver with assertions.
- **Sim-to-real on friction.** Everything hinges on µ ≈ 0.6 wood-on-whiteboard, and the rulebook says
  the surface varies by stage — MDF, whiteboard, or printed tarpaulin. Sweep µ; never fix it.
- **Two referee questions are worth more than any amount of simulation** (R7, R10): is the wrong-zone
  penalty −3 or −5, and does a marker still aboard at the buzzer incur it? The second one selects
  between opposite strategies. Ask at registration.
- **MJX feature coverage is unverified.** Keep Tier 1 on primitive geoms and confirm before depending
  on batched runs.
- **Scope.** P0–P2 plus the P4 Monte Carlo deliver most of the decision value. Enter P3 station by
  station, driven by whichever open item is blocking the physical build.
