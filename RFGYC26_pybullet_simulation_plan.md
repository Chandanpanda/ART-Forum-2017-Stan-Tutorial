# RFGYC'26 Senior — PyBullet Simulation Plan

**Scope:** simulate the two-robot Rev C system (Agent A + Agent B) on the ITU Robotics for Good
Youth Challenge 2026–27 Senior field, in PyBullet.

**Sources read (commits `37faa2c`, `a9288be`, `1601df1`, `8956e65`):**

| Artefact | Commit | Status | Role in this plan |
|---|---|---|---|
| `Robotics for good youth challenge.pdf` | `a9288be` | Authoritative (external) | Field, pieces, rules, scoring, penalties |
| `RFGYC26_robot_specification_revC_1.md` | `8956e65` | Authoritative (internal) | Station schedules, masses, tolerances, open items §10 |
| `RFGYC26 Robot Drawing Set revC.dc.html` (9 sheets) | `8956e65` | Authoritative (internal) | Dimensioned geometry, section details |
| `RFGYC26 Mechanism Explainer.dc.html` | `8956e65` | **Newest** — supersedes on 2 points | Station-by-station piece path, revised diverter |
| `RFGYC26 Explainer.dc.html` | `8956e65` (updated) | Newest | Field zone polygons, both robots' routes + timings |
| `RFGYC26_robot_specification.md`, `_revB.md`, `Robot Drawing Set.dc.html`, `*.png` | `37faa2c`/`a9288be`/`1601df1` | **Superseded** | Ignore (Rev A/B: push-fed lanes, rear chutes, gravity ducts) |
| `support.js`, `doc-page.js` | `37faa2c` | Infrastructure | Rendering runtime only — no robot logic |

Note that the repository's `0_welcome/` … `6_about_stan/`, `demo/` and `README.md` are an unrelated
Stan/Bayesian-statistics tutorial. The RFGYC26 material is a separate body of work sharing the repo.

---

## Part 1 — What the source actually specifies

### 1.1 The rules that bind a simulator

- 120 s, fully autonomous, no intervention after the start. Both robots start inside a
  480 × 280 mm taped box together with **all** beams and kits.
- Field 1143 (X) × 1181 (Y) mm, walls 19–20 thick × 65–70 tall. Origin bottom-left inside the walls.
- Senior scoring (per round; ranking is **best single round**, so the sim must optimise peak,
  not mean):

| Item | Points |
|---|---|
| Sample in a lab slot | +15 each |
| All three samples in slots | +5 |
| Sample outside quarantine **and** outside lab at the end | −3 each |
| Sample left inside quarantine | −5 each |
| Beam upright, stable, free-standing, released | +25 each |
| Perimeter closed by 2 beams + 2 walls | +20 |
| Kit in a valid marked area | +3 each |
| Correct 6 / 2 / 2 kit distribution | +20 |
| Destination zone left with no kits | −10 each |
| Patient in the correct zone | +5 each |
| All 4 red in H / yellow split across both PCCs / all 4 green in RZ | +6 / +8 / +6 |
| Patient in a wrong zone | −3 (table) / −5 (penalties §j.4.4) — **rulebook contradicts itself** |
| Touching robot / manipulating field / restarting program / robot leaves field | −20 each |

  Theoretical maximum ≈ **250** (samples 50, beams 70, kits 50, patients 80).

- Pieces: 3 discs Ø56 × 5 (~8 g); 10 kits 25 × 25 × 20 (~9 g); 12 cylinders Ø20 × 20 (~5 g),
  4 red / 4 yellow / 4 green; beam 1 280 × 60 × 20 (~200 g), beam 2 250 × 60 × 20 (~180 g).

### 1.2 Field geometry (mm, from the Explainer's zone polygons — the only place these are numeric)

| Zone | Extent |
|---|---|
| Quarantine | X 0–280, Y 0–280 |
| Laboratory plate (440 × 150 × 3) | X 351.5–791.5, Y 360–510 |
| Lab holes Ø60 | X 431.5 / 571.5 / 711.5 at Y 372, pitch 140 `[VERIFY]` |
| Hospital (H) | X 471.5–671.5, Y 901–1181 |
| PCC-L / PCC-R | X 0–200 / X 943–1143, Y 981–1181 |
| Recovery zone (RZ) | X 700–900, Y 190–270 (inside the deployment box) |
| Deployment box (20 tape) | X 643–1123, Y 0–280 `[VERIFY offset]` |
| Patient stickers Ø20 | X ∈ {100, 200, 943, 1043} × Y ∈ {481, 581, 681} |
| Beam 1 final | 280 × 20 footprint, 60 tall, from the left wall along Y 250–270 |
| Beam 2 final | 20 × 250 footprint, 60 tall, from the bottom wall along X 280–300 |

Beam contact point ≈ (280, 250). Static tip-over angle atan(20/60) = **18.43°** — matches the
spec's 18.4°.

### 1.3 The Rev C architecture in one paragraph

Both agents are **the same rolling chassis and the same conveyor**; they differ only in a printed
lane cassette, front guides, tail funnel, Agent A's beam pockets, and firmware. A flat 1.5 mm
elastomer belt, 116 wide, runs continuously at ~60 mm/s up an 11° incline on crowned Ø16 rollers,
from a nose roller (belt top Z 17.5) to a tail roller (belt top Z 51). Drive is 2 × NEMA 17 through
GT2 2:1 onto Ø60 × 22 wheels on **stub axles at the fore-aft centroid** (A: Xa 142.5, B: Yb 132.5),
track 150, with four Ø20 ball transfers at 40 × 45° chamfered corners. **Every actuator is
zero-force:** the belt supplies all motive energy, servos supply only direction (diverter) and
permission (gates); nothing lifts, grips or pushes a queue.

The chain of stations, from the Mechanism Explainer:

```
1 Capture   sweeper fingers, 165 → 116, one MG996R, 1:1 spur pair, ~0.4 s, default OPEN in transit
2 Scoop     0.5 shim, sprung ~1.5 N onto the floor, free to trip 25° up; A 65 @15°, B 48 @20°
3 Uptake    belt 116 @ 11°, ~60 mm/s, friction-only carry (µ≈0.6 vs tan 11° = 0.19, ~3× margin)
4 Triage    convergence wedge 116→30 (one wall straight, one at 22°), shrouded camera + TCS34725
5 Divert    one MG90S vane, four held angles, piece translated by the belt along the blade face
6 Accumulate  swim lane, queue rests against a closed MG90S flap gate; ~0.03 N drag per piece
7 Discharge  one gate opens; converging funnel 116 → 40 over the tail roller
8 Deposit   gravity drop 51 mm past the rear face; lands 5–15 mm aft; depart nose-out
```

Agent A has **no triage and no lane cassette** — one class of piece. Its converging guides
(116 → 62) feed a Ø58 vertical **chute that is also the magazine** (Za 51 → 11, capacity 8,
carries 3), with an MG90S base gate 8 mm above the 3 mm lab plate; Agent A therefore **docks the
lab in reverse**. Its two beams never touch the belt: they stand on the field in open-bottomed,
open-ended flank pockets and are released by **axial separation only**.

Localisation is deliberately open-loop for gross navigation (0.47 mm/full-step, 0.36°/step
turn-in-place) with **every terminal placement closing on a physical feature** — TMC2209
StallGuard wall stall, a tape crossing, a gate, or a contact lever. Zero encoders, zero bump
switches.

### 1.4 Discrepancies found across the four commits

These are not nitpicks — each one changes what a simulation would predict, and each should be
resolved (or parameterised) before any model is built.

1. **Agent B's belt incline is over-determined.** Nose belt-top Z 17.5, tail Z 51, run 192
   (Yb 202 → 10). At a true 11°, 192 mm of run rises 37.3 mm, putting the tail at **Z 54.8**, not
   51. Sheet 2's "rises 33.5 over the run" is self-consistent only for **Agent A's 175 run**
   (175 × tan 11° = 34.0 → Z 51.5 ✓). The Mechanism Explainer resolves this silently: its piece
   path uses the gradient `z = 17.5 + (x−63)·0.1745`, and **arctan(0.1745) = 9.90°**, not 11°.
   Pick one: 11° with a Z 54.8 tail (bigger drop, longer throw), or ~9.9° with a Z 51 tail.
2. **The diverter changed and the spec was not updated.** Spec §6.2 and Sheet 6: a **90 mm plow
   blade pivoted at its upstream end** at Yb 152, max lateral deflection 43 mm = 25°. Mechanism
   Explainer (newest): a **66 mm vane pivoted at its middle**, angles 0/±12/±25°, tip throw ±18.6,
   working in a "divider-free fan zone" whose lane walls start aft of the swept envelope and whose
   pointed noses "finish the lateral move".
3. **The plow cannot reach L4 on its own.** Belt centreline is Xb 90; lane centres are L1 49,
   L2 79.5, L3 108, L4 136.5. Green (L4) needs **46.5 mm** of lateral travel, against the spec's
   quoted 43 mm maximum. The Mechanism Explainer's pointed lane noses are therefore load-bearing,
   not decorative — the last ~4 mm is done by the wall, not the vane.
4. **Lane run: 107 or 63?** Spec §6.3 and Sheet 5 say the lane run is Yb 152 → 45 = **107**, with
   L1 binding at 4 kits × 25 = 100. The Mechanism Explainer's station 6 says lane run **63**, "3
   cylinders on the belt, 4th waits in the flare" — i.e. the fan zone consumes ~44 mm of the
   nominal run. If 63 is right, **L1 cannot stage 4 kits** (4 × 25 = 100 > 63).
5. **Senior patient positions.** The spec (§1) assumes "twelve Ø20 patient stickers in **fixed
   positions**, colour randomised — waypoint navigation plus classification, never search." The
   rulebook (§b Step 2, §d.2.8, §h.3.2) says for Senior "the **position** of the triaged patients
   is randomised in each match" and "the arrangement may vary in each match". If the rulebook means
   what it says, Agent B's entire route plan (descend each column from above, top-down) is invalid
   and it needs search/detect. **This is the single largest strategic risk in the design**, and it
   is cheap to quantify in simulation.
6. **Beam 1 final Y.** Spec §1: "from the left wall along **Y 250–270**" (centre Y 260). The
   Explainer places it at centre Y 252.5 (spanning 242.5–262.5). 7.5 mm matters when the pass/fail
   criterion is "touches the black line" and "touches beam 2 at one end".
7. **Wrong-zone patient penalty is −3 in the rulebook's Senior table and −5 in its penalties
   section.** Make it a config knob and report both.

---

## Part 2 — The PyBullet plan

### 2.0 Decide what the simulator is *for* before building it

The sim earns its cost only if it answers questions the team cannot cheaply answer on a bench.
Three tiers of value, in priority order:

- **A. Strategy and scoring.** Given randomised sample and patient placement, what does the route
  plan actually score, and how often does the T−40 / T−50 abort logic save the round? This is a
  Monte Carlo problem, and it needs almost no mechanism fidelity.
- **B. Mechanism risk.** The spec's own §10 open-items list is a list of simulation experiments:
  wedge jam rate (§10.7), scoop pickup rate (§10.8), accumulation creep past a closed gate (§10.6),
  belt tracking (§10.4), disc gate exit velocity (§10.10), CG trim (§10.11). Answering these in sim
  before cutting acrylic is where the money is.
- **C. Firmware-in-the-loop.** Run the actual route/state-machine code against the sim. Worth doing
  only after A and B exist.

Everything below is organised so that A ships first and B slots in behind it.

### 2.1 Two simulation tiers, one geometry source

**Do not try to build one simulator.** The match sim and the mechanism sim have incompatible
requirements (120 s of wall-clock-competitive stepping vs. millimetre contact fidelity), and
merging them produces something that is both slow and wrong.

| | **Tier 1 — Match sim** | **Tier 2 — Mechanism sim** |
|---|---|---|
| Scope | Whole field, both agents, all 27 pieces | One agent, a 600 × 600 floor patch, one station at a time |
| Timestep | 1/240 s | 1/1000 s (or Froude-scaled — §2.4) |
| Robot | Chassis box + 2 driven wheels + 4 low-friction spheres | Full: belt, scoop, fingers, wedge, vane, gates, funnel, chute |
| Manipulation | **Abstracted**: capture / hold / release events with timing + failure rates *sampled from Tier 2* | Simulated in contact |
| Answers | Routes, timing budget, aborts, deconfliction, score distribution | Jam rate, pickup rate, creep, landing band, exit velocity |
| Target speed | ≥ 20× real time headless | ≥ 0.2× real time is fine |

The bridge between them is a small table of **empirical rates** — `p_capture(approach_speed)`,
`p_jam(arrival_rate)`, `t_gate_to_floor`, `landing_offset_mean/σ` — measured in Tier 2 and consumed
by Tier 1. That is the whole coupling; keep it that narrow.

### 2.2 Single source of truth for geometry (do this first, before any physics)

Right now the numbers live in three places that already disagree (§1.4). Step one is a single
`geometry.py` / `params.yaml` holding every dimension in millimetres, with:

- one entry per station from the §7 and §8 station schedules;
- every field polygon from §1.2;
- an explicit `VERIFY` flag on each value the spec tagged `[VERIFY]`, plus the seven items in §1.4;
- **assertions** that re-derive the over-determined values and fail loudly: belt rise vs. run vs.
  incline, lane widths summing inside the 116 belt, width budget (116 + 44 + 20 = 180 exactly),
  A's beam-pocket budget (228 of 235), lane capacity vs. piece count, τ = swept diameter ÷ body
  length.

URDFs are then **generated** from this file, never hand-written. This is what stops the sim from
silently drifting away from the drawings — and it would have caught the 11°/9.9° problem on day one.

### 2.3 The hard problem: PyBullet has no conveyor

The entire Rev C manipulation architecture is *friction transport on a moving surface with passive
accumulation*. PyBullet has no belt primitive and no surface-velocity property (unlike MuJoCo).
Two viable models, and I recommend building **both** because agreement between them is the only
cheap validation available:

**Model 1 — contact-force injection (primary).** Each step, for every contact on the belt link:

```
v_slip = v_belt_world − v_piece_at_contact
F      = µ · normalForce · unit(v_slip)             # Coulomb traction, along the slip
F      = clamp(F, mass · |v_slip| / dt)             # never overshoot belt speed
applyExternalForce(piece, +F, at contact point)
applyExternalForce(chassis, −F, at contact point)   # reaction — this is not optional
```

This reproduces exactly the physics the spec argues from. A free piece accelerates to belt speed
and F decays to zero; a piece held at a closed gate sits at full slip and feels a steady
µ·m·g = 0.6 × 0.005 × 9.81 = **0.029 N** — the spec's 0.03 N, derived rather than asserted. And
because the reaction goes back onto the chassis, the "four pieces ≈ 0.12 N against a belt that can
pull ~30 N" claim becomes a measurable output instead of an assumption.

**Model 2 — roller bank (cross-check).** 12–16 velocity-controlled Ø16 cylinders spanning the
116 mm width at ω = v/r = 0.060/0.008 = 7.5 rad/s. Slower and chattier, but structurally
independent. If Model 1 and Model 2 agree on jam rate and landing band, trust the number; if they
don't, neither is trustworthy yet.

Do **not** model the belt as an articulated loop of links — it is expensive and unstable at this
scale, and buys nothing the two models above don't give.

### 2.4 Units, scale and solver settings — the thing that will actually break

The pieces are 5–9 g and 20–56 mm; the shim is 0.5 mm and the lab plate is 3 mm. Bullet's default
contact tolerances and collision margins are tuned for roughly metre-scale, kilogram-scale objects.
Left alone, thin parts will jitter, tunnel, or float. Plan for it explicitly:

1. **Start in true SI metres** with `fixedTimeStep = 1/1000`, `numSolverIterations ≈ 150`,
   `numSubSteps = 1` (sub-steps interact badly with `applyExternalForce`, which is cleared each
   step — apply forces every step, not every sub-step).
2. **Thicken thin collision proxies.** The 0.5 mm shim gets a ~3 mm collision box with 0.5 mm
   visual geometry; likewise the belt sheet and the lane dividers. Collision and visual geometry
   do not have to match, and here they must not.
3. **Set contact material properties explicitly** on every piece: `lateralFriction`,
   **`rollingFriction`** and **`spinningFriction`** (without these, a Ø20 wooden cylinder rolls
   forever and the "roll-back retention lip" question becomes meaningless),
   `contactStiffness`/`contactDamping`, and `linearDamping`/`angularDamping` at small non-zero
   values.
4. **Fallback if contact behaviour stays bad: Froude scaling.** Scale all lengths ×10 and all
   masses ×1000, keep g = 9.81. Friction coefficients are dimensionless and unchanged; forces scale
   ×1000; times scale by √10 ≈ 3.16 (the sim runs in "slow motion" and you convert back). The
   0.5 mm shim becomes 5 mm and every one of Bullet's tolerances becomes comfortable. Keep the
   conversion in one place so results are always reported in real units.

### 2.5 Robot model decomposition

Generate two URDFs from the shared geometry file. Common chassis, per-agent cassette.

**Rigid + jointed (simulate in contact):**

- Chassis: box with 40 × 45° chamfers (a chamfered convex hull, not a plain box — the whole
  turning-ratio argument on Sheet 3 depends on the chamfers), ground clearance 6.
- 2 × continuous wheel joints, Ø60 × 22, track 150, at the fore-aft centroid on stub axles.
- 4 × ball transfers: low-friction spheres (`lateralFriction` ~0.05, `rollingFriction` ~0.0) — **not**
  swivel castors. Sheet 3 explicitly rejects castors because they must be dragged into alignment;
  modelling them as castors would reproduce a failure mode the design has designed out.
- Belt link (inclined thin box) + the traction model of §2.3.
- Scoop: a revolute joint on the nose-roller shaft with a torsion spring (~1.5 N preload at the
  tip) and a 25° travel limit. **This must be a real joint, not a fixed part** — the whole Sheet 4
  argument is that the scoop trips up when reversed, over the 20 mm tape and over the 3 mm plate.
- Sweeper fingers: 2 revolute joints coupled 1:1, position-controlled, ~55° arm rake
  (165 → 116 capture), default open.
- Agent B: convergence wedge (static, asymmetric — one wall straight, one at 22°), diverter vane
  (revolute, 4 held angles — model **both** variants from §1.4 item 2), 4 lane gates (revolute
  flaps), kit escapement shutter, 4 lane dividers, converging discharge funnel.
- Agent A: converging guides, Ø58 chute-magazine (a tube with a base gate), 2 open-bottomed,
  open-ended beam pockets, beam-contact lever.

**Stubbed analytically (do not simulate the physics):**

| Hardware | Sim stand-in |
|---|---|
| Steppers + TMC2209 microstepping | Wheel velocity/position control quantised to 0.47 mm/full-step; **plus an explicit step-loss model** — the spec calls a skipped step "silent", so inject skips at high acceleration and measure the damage |
| StallGuard | Wheel-torque saturation or contact-normal-force threshold; honour the "unreliable below ~0.1 m/s" rule by disabling detection under that speed |
| Camera + TCS34725 | Read the ground-truth colour of the body in the tunnel ROI; add a confusion matrix and a latency; sweep misclassification rate |
| VL53L1X ToF ×2 | `p.rayTest` + range noise + FOV cone |
| TCRT5000 5-ch array | `p.rayTest` down onto tape polygons (analytic), or read a floor texture |
| MPU6050 | Integrate base angular velocity with bias + random walk |
| Beam-contact microswitch | Contact-point query between pocket-R lead and beam 1 |
| Servos (MG996R / MG90S) | Position control with a rate limit and a settling time (~0.4 s sweeper cycle) |

**Interlock to enforce in the driver, not the route script** (per §6.4, and it is a compliance
checklist item): **at most one lane gate open at any instant.** Implement it as a hardware-style
mutex in the gate abstraction so a route bug cannot violate it, and assert on it every step.

### 2.6 The beams are a genuine physics test, not a scripted event

Everything else can be abstracted; the beams cannot. Agent A never lifts them — they stand on the
field and are dragged in open-bottomed pockets, then released by **axial separation only, never
lateral, never with yaw while a pocket overlaps a beam end**. In PyBullet that is exactly a
friction-drag problem between a free 200 g rigid body and two pocket walls, and it is worth
simulating properly because:

- the scoring criterion — "standing upright on its long side, **stable by itself**, fully
  released" — is directly checkable from the final tilt and contact set;
- the 18.43° tip-over margin is tight, and the release transient is where it gets used up;
- the "**rotate only at ≥ 200 mm from a placed beam**" rule (Agent A runs τ = 1.30) can be
  **automatically audited** every step: flag any yaw command whose swept R 185 circle intersects a
  placed beam. Same for the walk-over rules — no release under a chassis, no path over a placed
  piece. These three audits are cheap and catch exactly the class of bug that costs a round.

### 2.7 The referee module

Implement Senior scoring as a **pure function of the final world state** plus an event log:

```
score(world_state, event_log, ruleset) -> {total, breakdown, violations}
```

Zone membership is a point-in-polygon test against §1.2; "sample completely inside a laboratory
slot" needs a containment test against the Ø60 hole, not a centroid test; "beam upright and stable"
needs a tilt threshold plus a settled-velocity check; "free-standing, untouched at the buzzer"
needs a final contact check against both robots. Make the −3/−5 wrong-zone ambiguity (§1.4 item 7)
a `ruleset` parameter and report both totals.

With the referee as a pure function, the whole thing becomes a benchmark: run N randomised matches,
get a score distribution, and — since ranking is best-single-round — read the **upper tail**, not
the mean.

### 2.8 Validation: the spec grades its own homework

The spec asserts a dozen numbers. Turn each into a regression test; a model that reproduces them is
trustworthy, and a model that doesn't has found either a modelling bug or a design error.

| Assertion (source) | Sim check |
|---|---|
| Retarding force ≈ 0.03 N/piece, 0.12 N for four (§3.2) | Measure the belt reaction on a queued piece |
| Carry margin µ 0.6 vs tan 11° = 0.19, ~3× (§3.2) | Sweep µ down until pieces slip back; expect ~0.19 |
| Landing band 5–15 mm behind the rear face (§6.4) | Ballistic drop from the tail at 60 mm/s — **note this depends on resolving 11° vs 9.9°, Z 51 vs Z 54.8** |
| Disc exit ~0.4 m/s from an 8 mm drop (§6.4) | √(2·9.81·0.008) = 0.396 m/s ✓ analytically; check bounce-out of the Ø60 hole |
| Beam tip-over 18.4° (§1) | atan(20/60) = 18.43° ✓; check the release transient stays inside it |
| τ = 1.07 (B, chamfered) / 1.30 (A, beam-loaded) (§4) | Sweep the swept circle in sim and measure |
| Scrub 0.03 N·m vs 3.9 N·m available (§4.3) | Measure turn-in-place torque draw |
| 0.47 mm/full-step, 0.36°/step, 1000 steps/360° (§3.3) | π·150 / 0.47 = 1002 steps ✓ |
| Wedge is "the #1 jam risk", 50 ingests per lane (§10.7) | **The headline experiment** — Monte Carlo doubled arrivals at worst-case entry angle |
| L1 stages 4 kits (§6.3) | Depends on lane run 107 vs 63 (§1.4 item 4) — 4 × 25 = 100 does not fit in 63 |
| Plow reaches all four lanes (§6.2) | 46.5 mm needed to L4 vs 43 mm quoted (§1.4 item 3) |

Four of these are already known to be in tension with the drawings. That is the plan working.

### 2.9 Phasing

**Phase 0 — Geometry truth (no physics).** `params.yaml` + `geometry.py` + assertions + a matplotlib
top-down render of the field and both robots. Resolve or explicitly park the seven items in §1.4.
*Deliverable: a rendered field that matches Sheet 9, and a failing assertion for every unresolved
number.*

**Phase 1 — Tier 1 skeleton.** Field, walls, zones, all 27 pieces, both chassis with driven wheels
and ball transfers. Manipulation entirely abstracted (teleport-on-capture, drop-on-release). Wire
in the referee. *Deliverable: a scripted 120 s match that scores.*

**Phase 2 — Navigation realism.** Stepper quantisation + step-loss, StallGuard wall squaring, ToF,
TCRT tape crossings, IMU heading hold. Replace the Explainer's hand-authored `PATH_A`/`PATH_B`
waypoint tables with a route executor that closes every terminal on a physical feature — the
spec's load-bearing rule. *Deliverable: the T−40 / T−50 abort logic exercised under odometry drift.*

**Phase 3 — Tier 2 mechanism rig.** One station at a time, in this order (cheapest-to-highest
risk-reduction): belt traction and accumulation → scoop pickup → convergence wedge → diverter and
lanes → discharge and landing band → Agent A chute and beam release. *Deliverable: the empirical
rate table of §2.1, and answers to spec §10 items 4, 6, 7, 8, 10.*

**Phase 4 — Monte Carlo.** Randomise sample positions, patient colours **and positions** (test both
readings of the rulebook — §1.4 item 5), venue friction, lighting/misclassification, step-loss rate.
N ≥ 500 matches. Report the score distribution and, because ranking is best-of, the 90th percentile.
*Deliverable: the expected competition score and the top three failure modes ranked by points lost.*

**Phase 5 (optional) — firmware in the loop.** Swap the Python route executor for the real ESP32
state machine over a shim.

### 2.10 Risks, honestly

- **The belt is the whole design, and PyBullet models it worst.** If neither belt model of §2.3
  reproduces stable accumulation, Tier 2 loses most of its value and the mechanism questions go
  back to the bench. Build the belt rig in Phase 3 *first* and treat it as a go/no-go.
- **Thin geometry.** 0.5 mm shim, 1.5 mm belt, 3 mm plate. §2.4 mitigates this, but expect to spend
  real time here, and expect to end up Froude-scaled.
- **Sim-to-real on friction.** Everything hinges on µ ≈ 0.6 wood-on-whiteboard and on the venue
  surface, which the rulebook says varies by stage (MDF, whiteboard, printed tarpaulin). Treat µ as
  a swept parameter, never a constant, and report score-vs-µ.
- **Garbage in.** Seven numbers are known-inconsistent and several more are `[VERIFY]`. A confident
  simulation built on unresolved geometry is worse than no simulation — hence Phase 0 first.
- **Scope.** Tier 2 at full fidelity across every station is a large build. Phases 0–2 plus the
  Phase 4 Monte Carlo deliver most of the decision value; Phase 3 should be entered station by
  station, driven by which §10 open item is currently blocking the build.
