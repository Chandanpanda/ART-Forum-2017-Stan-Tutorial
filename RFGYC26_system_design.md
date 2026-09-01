# RFGYC'26 — System Design: planning, control, perception, and the sim/hardware interface

**Status: design baseline for the software rebuild.  Supersedes the route-tuning
workflow; leaves every mechanical finding (F1–F81) and the MuJoCo model as they
are.**

The simulator answered the mechanism questions: the intake takes 14/15 samples
honestly, the magazine posts them, the beams seat, the kits land in their zones.
What it has NOT answered is the clock, and it cannot answer it the way we have
been asking — by editing a hand-written script and re-running twelve seeds.  The
match is a route-optimization problem and a tracking-control problem, both
classical, both with known algorithms.  This document designs that system, on
top of the code that exists, such that **the software developed in MuJoCo is the
identical software that runs on the real robot** — a Raspberry Pi 5 with two
rigidly-mounted, calibrated cameras — with only the lowest interface layer
swapped.

---

## 1. The kit that leaves robot 1's plan: PCC_R

Robot 1 hands the **PCC_R zone (two kits)** to robot 2.  Robot 1 keeps HOSP (6)
and PCC_L (2).

Why PCC_R and not PCC_L:

* **It is the one zone on robot 2's natural path.**  Robot 2 starts in the
  deployment box (643–1123, 0–280), against the east wall.  PCC_R
  (943–1143, 981–1181) is a single straight run ~900 mm north along that same
  wall — the easiest possible first act for a two-motor, camera-less robot: a
  wall on its right the whole way, a plow-drop at the end, done in the opening
  ten seconds while robot 1 is still sweeping the quarantine in the opposite
  corner.  No crossing of the field, no crossing of the 5 mm laboratory plate
  (which small wheels may not climb), no interference with robot 1.
* **Robot 2 returns there anyway.**  Yellow patients from the east side area
  (SIDE_R, x 943–1103) are delivered to PCC_R, so the corner is on its beat for
  the rest of the match.
* **Removing PCC_L instead would be the worst of both.**  Robot 1's kit loop
  deliberately *ends* at PCC_L because the beam phase stages 480 mm from it;
  ending at HOSP instead adds ~250 mm to the highest-risk transit of the match.
  And robot 2 would have to cross the entire field diagonally, around the
  laboratory, to reach the far corner — precisely what a dead-reckoned robot
  cannot do.

What it buys robot 1: the PCC_R spur (the leg east to (903, 930), the turn, the
drop, the 200 mm backoff, the re-aim) — **5–7 s** off the kit loop, and one
fewer terminal manoeuvre that can go wrong.

Scoring accounting (referee-measured): robot 1 alone then books kits
24 − 10 (empty PCC_R) = **+14**; robot 2's two kits add +6, clear the −10, and
complete the 6/2/2 distribution for +20 — system kit total **+50**, unchanged.
The raw `mrun` board therefore *drops by construction* (−36 moves to robot 2's
ledger) the day `M2.KIT_AGENT_A` is flipped to `("HOSP", "PCC_L")`.  Progress is
tracked on the **robot-1 subtotal** (samples + beams + own-kit share, ceiling
50 + 70 + 14 = **134**), which `diag.py`'s `sNNbNNkNNpNN` breakdown already
separates.  The flip lands together with the planner (§13 step 4), not before:
flipped against the current hand-written route it collapsed the board once
already.

---

## 2. Where the system stands, and why it is slow

Baseline at HEAD (12 seeds, honest referee, scored at the buzzer): **+93.2**
raw, robot-1 subtotal ≈ 127/170 (all three kit zones still on robot 1).
48/48 geometry checks and 20/20 built-model checks green.

The profiler says the match is not spent scoring.  Of ~120 s:

| where the time goes            | measured | what it is                                    |
|--------------------------------|---------:|-----------------------------------------------|
| sweep passes + dwells          |  ~23.5 s | two lanes, sensor-terminated dwells (already lean) |
| laboratory approaches          |  ~17 s   | pivots, look-station legs, tracked reverses    |
| kit loop transit               |  ~19.3 s | dogleg, east climb, L-traverses, drops         |
| beam transit + staging         |  ~32.6 s | pivots, diagonals, dressing shuffles           |
| actual scoring contact         |  ~10 s   | posting strokes, hopper flaps, seating pushes  |

Roughly **75 s of repositioning**, and it decomposes into exactly the defects a
planner and a tracker remove:

1. **Every leg ends at v = 0.**  The route is turn → drive → stop → turn →
   drive.  A pivot costs 1.9–3.2 s (measured, `turn_to` table in route.py), and
   the route runs ~20 of them.  Most connect legs whose headings differ by
   20–60° — angles a moving robot takes as a 190 mm arc at speed for ~0 extra
   cost (60 °/s at 200 mm/s, the caps already proven in `stall_drive`).
2. **Budgets are constants, so they are wrong in both directions.**
   `BEAM_BUDGET = 34`, `KIT_BUDGET = 30` are worst-case reservations.  The
   third laboratory slot is skipped 12/12 seeds, missing its deadline by
   1–5 s, while actual beam+kit phases frequently finish inside their budgets —
   the slack exists, but a constant cannot move it.
3. **The order is hard-coded.**  sweep → lab → kits → beams is *a* feasible
   order, chosen by hand.  With PCC_R gone the station set changes, and the
   right order under each seed's timing is a computation, not an opinion.
4. **Known fixes are stranded by phase cost.**  Stall-terminated beam-1 seating
   (0.5 mm residual vs 10 mm, worth the T-joint) costs 1.3 s and is reverted
   only because the current schedule cannot afford 1.3 s.  A plan that recovers
   10 s affords it trivially.
5. **Everything navigates on ground truth.**  `rb.pose` reads `d.xpos`;
   `align_reverse` and `stall_drive` read `d.qvel`.  On the real robot none of
   that exists — and the terminal behaviours that already work relative to a
   *measurement* (camera dock, wall stalls) are exactly the ones that will
   survive.  The transit layer has no real-hardware story at all yet.

Items 1–4 are the planning/control problem; item 5 is the interface problem.
This design addresses all five.

---

## 3. Architecture

Layered, dependencies pointing strictly downward.  Everything above the HAL is
**shared verbatim between simulation and hardware**.

```
┌───────────────────────────────────────────────────────────────┐
│ MISSION EXECUTOR      schedule sequencing, deadline policy,   │
│                       replanning triggers, robot-2 tasking    │
├───────────────────────────────────────────────────────────────┤
│ ROUTE PLANNER         station graph + precedence DAG,         │
│  (offline + replan)   prize-collecting tour DP, time model    │
│ TRAJECTORY GENERATOR  polyline → arc-blended path + velocity  │
│                       profile (accel & curvature limited)     │
├───────────────────────────────────────────────────────────────┤
│ CONTROLLERS           path tracker (pure pursuit + v-sched),  │
│                       terminal/guarded moves (wall stall,     │
│                       tracked reverse, dress, seat), process  │
│                       control (belt/roller/dwell)             │
├───────────────────────────────────────────────────────────────┤
│ STATE ESTIMATOR       wheel odometry ⊕ camera fixes ⊕ wall    │
│                       stalls → (x, y, θ) + confidence         │
├───────────────────────────────────────────────────────────────┤
│ PERCEPTION            pixels → slot poses, sample positions,  │
│                       cylinder colours, robot-2 marker,       │
│                       landmark fixes                          │
├───────────────────────────────────────────────────────────────┤
│ HAL                   drive, devices, sensors, cameras,       │
│                       robot-2 link, clock/tick               │
├───────────────┬───────────────────────────────────────────────┤
│ SimBackend    │ PiBackend                                     │
│ (MuJoCo)      │ (Pi 5: steppers/TMC2209, N20+PWM, servos,     │
│               │  picamera2 ×2, Bluetooth to the Pico)         │
└───────────────┴───────────────────────────────────────────────┘
```

The referee, `check_geometry.py`, `check_model.py`, and the scratch rigs sit
*outside* this stack: they are the test harness and keep their ground-truth
access.

---

## 4. HAL — the one layer that changes

### 4.1 Contract

`sim/rfgyc26/hal.py` defines the interface; `robot.py` becomes its MuJoCo
implementation (a mechanical extraction — the method bodies already exist).

```python
class DriveHAL:
    def drive(self, v_mm_s, omega_deg_s): ...      # body-frame velocity request
    def stop(self): ...
    def odometry(self):                            # (dL_mm, dR_mm) wheel travel
        ...                                        # since the previous call
    def stalled(self, thresh=0.42) -> bool: ...    # TMC2209 StallGuard / sim torque

class DeviceHAL:                                   # one call per mechanism
    def intake(self, collecting, rpm=None): ...    # knife servo + brush N20
    def fingers(self, opened): ...
    def feed(self, down): ...                      # magazine paddle
    def gate(self, opened): ...                    # escapement leaves
    def blade(self, inserted): ...                 # retainer leaves
    def cradle(self, which, carry): ...
    def cradle_down(self, which) -> bool: ...
    def trim(self, y_mm): ...
    def trim_at(self) -> float: ...
    def trim_settled(self) -> bool: ...
    def open_hopper(self, dest) -> int: ...
    def mag_count(self) -> int: ...                # bore reflectance ray

class CameraHAL:
    def frames(self):                              # (imgL, imgR, t_s) BGR uint8,
        ...                                        # Vision.W × Vision.H
    def calib(self) -> StereoCalib: ...            # K, dist, R|T cam→robot, both eyes

class LinkHAL:                                     # robot-2 (§10)
    def send(self, cmd: bytes): ...
    def recv(self) -> bytes | None: ...

class Clock:
    def now(self) -> float: ...                    # match seconds
    def tick(self): ...                            # advance one 20 ms control period
```

### 4.2 The tick model — why the behaviours port unchanged

`route.py` is already written as **generators resumed at 50 Hz**: every `yield`
is one control period.  That structure is kept as the scheduler contract:

* **SimBackend.tick()** steps `mj_step` the right number of physics substeps.
* **PiBackend.tick()** sleeps to the next 20 ms wall-clock boundary and latches
  sensor reads.

No behaviour, controller, or planner code knows which one it is running on.

### 4.3 What is banned above the HAL

The migration is defined by removing three ground-truth reads from mission
code:

| today                                    | replacement                                     |
|------------------------------------------|-------------------------------------------------|
| `rb.pose` (reads `d.xpos`) everywhere    | `est.pose` from the state estimator (§6)        |
| `rb.d.qvel` in `align_reverse`, `stall_drive` | "distance not accruing" from `odometry()`  |
| `rb.see_lab()` (synthetic geometry)      | `perception.slots(frames)` — same return shape  |

Ground truth survives only in the referee, the check suites, and an estimator
error report (§12).  `see_lab`'s synthetic model is *kept* as a fast test
double (`--model-camera` flag) so 12-seed planner regressions don't pay for
rendering; the rendered pipeline is the default and the one that gates
releases.

### 4.4 Hardware notes (for later; the contract is what matters now)

Drive NEMA17s via TMC2209 (UART, StallGuard = `stalled()`; commanded steps =
`odometry()` — steppers below stall don't slip, which makes odometry exact
except during stall/scrub events, and those are flagged).  N20 roller with
encoder + PWM (`intake(rpm=…)` is already a control).  MG90S servos on a PWM
hat or an RP2040 I/O co-processor — the HAL doesn't care.  Cameras: two Camera
Module 3 Wide on the one laser-cut plate (Vision class: 130 mm baseline, 45°
pitch, 8° toe), calibrated once with the standard OpenCV stereo flow; the same
YAML loads in both backends.

---

## 5. Perception — fed from cameras in both worlds

The MJCF gains two `<camera>` elements at `Vision.CAM_*` (the mounts already
exist as geometry); **SimBackend.frames() = offscreen `mujoco.Renderer` at
1280×720**, PiBackend.frames() = picamera2 captures.  One pipeline consumes
both:

1. **Laboratory slots** (the precision task).  Undistort → adaptive threshold →
   contour → **ellipse fit of the top rim** (the rim, not the blob centroid —
   Vision.DET_BIAS documents the 2.4 mm centroid bias) → per-eye centre +
   apparent diameter → stereo triangulation *and* known-diameter mono range
   (two physics that must agree, or the measurement is refused) → the same
   `[(x_mm, y_mm, mode)]` robot-frame list `see_lab` returns today.
   `look_lab`, `pick_slot`, `dock_and_post` run unchanged.
   Acceptance: ≤ 2 mm lateral error at ≤ 200 mm range on rendered frames, per
   the Vision error budget.
2. **Quarantine samples** (the only random positions on robot 1's board).
   Ø56 discs against the field: colour/contrast segmentation in the quarantine
   box, centres triangulated.  Used opportunistically — the aft-looking rig
   sees the swept lane on every reverse-out — to (a) choose/skip sweep lanes,
   and (b) answer "is the quarantine actually empty" before leaving, which
   replaces the failed blind third-pass experiments (F59) with a *conditional*
   recovery pass that fires only on a seen stray.
3. **Cylinder colours** (random colours at fixed stickers).  No detection
   needed: project the 12 known sticker positions through the calibrated rig
   whenever one is in frame, sample the patch, classify R/Y/G.  Consumed by
   robot 2's task plan, not robot 1's route.
4. **Robot-2 tracking.**  One ArUco marker on robot 2's top plate; pose from
   either eye.  This is robot 2's entire localization (its Pico dead-reckons
   between fixes and is corrected over the link).
5. **Landmark fixes for the estimator**: the lab slot line (full x, y, θ fix),
   wall/corner lines, zone tape.  Cheap, and only needed at ~1 Hz.

Perception runs at `Vision.FPS = 20`, and only in the phases that need it —
it does not sit in the 50 Hz loop.

---

## 6. State estimation

A 3-state (x, y, θ) filter, deliberately simple:

* **Predict** from `odometry()` differential-drive kinematics each tick.
  Stepper odometry is exact in straight lines; **pivots are the drift source**
  (measured turn efficiency 0.21–0.70 vs contact width — scrub) so pivot
  segments carry inflated process noise, and the planner's arc-blended paths
  (§8) reduce drift by mostly *not pivoting*.
* **Update** from: slot-line fixes (§5.5) when the laboratory is in frame;
  **wall stalls** (`stall_drive` already is the spec's "left/bottom-wall
  stall" datum — a 1-D position reset plus a θ square-up); kit-drop wall
  proximity; start-pose known exactly.
* Confidence gates behaviour: a terminal approach may demand σ ≤ 10 mm and
  insert a cheap fix (one look, one wall touch) if the estimate is worse.

Crucially, the design keeps the property the dock already proved: **terminal
scoring actions never consume the world estimate** — they close on relative
measurements (camera slot vector, wall stall, trim on measured lateral).  The
estimator only has to be good enough to *arrive at the look station*, which is
±20 mm territory, not ±2 mm.

---

## 7. Global route planning — the heart

### 7.1 The problem, formalized

Everything on robot 1's board is at a **fixed, known position** except the
three quarantine samples (random inside a known 280×280 box).  The cylinders
are fixed-position/random-colour and belong to robot 2.  So robot 1's match is:

> Choose a subset of prize-bearing stations, an order, and motions connecting
> them, to maximize score subject to a 120 s deadline, precedence constraints,
> and differential-drive kinematics — a **prize-collecting TSP (orienteering
> problem) with precedence, with sequence- and state-dependent travel times.**

Station set (service pose fixed by the mechanism — heading is *not* a free
variable at stations, which keeps the search small):

| node | service pose | prize | duration model |
|------|--------------|------:|----------------|
| SWEEP (macro: lanes from perception, want=3) | y-lanes, θ=180 | enables slots | measured 20–24 s, sensor-terminated |
| L1, L2, L3 (lab slots) | (hole_x, ~160), θ=270 | ~17 each | measured 6–11 s, `est = last dock` |
| KH (HOSP drop) | (711.5, 965), θ=90 | 18 + zone | ~2.5 s |
| KL (PCC_L drop) | (240, 930), θ=90 | 6 + zone | ~2.5 s |
| B2 (beam 2) | (—, —), θ=270 | 70 with B1+T-joint | ~10 s |
| B1 (beam 1) | derived from B2 stall datum, θ=180 | — | ~10 s incl. 1.3 s stall-seat |
| R (stray recovery, conditional) | from perception | recovers a slot's 17 | only if a stray is *seen* |

Precedence DAG: SWEEP ≺ {L1..L3} (cargo), SWEEP ≺ B2 (beams seal the
quarantine with anything still inside), B2 ≺ B1 (beam 1's line is *derived*
from beam 2's stall datum, F54), {B1, B2} terminal (the robot is boxed into
the south-west afterwards — F44).  Kit drops are unconstrained; their order is
the planner's to choose, not a convention.

### 7.2 Travel-time model (the cost matrix)

`T[i][j][world_state]` computed by path search, not guessed:

* **Field polygon** with inflated obstacles: walls and lab plate at the 185 mm
  swept radius (187 loaded), side-area cylinder strips as keep-outs (robot 2's
  pieces; the walk-over audit applies), **placed-piece keep-outs that appear as
  tasks complete** — dropped kits (the F77 lesson) and placed beams.  World
  state = {beams placed?, kits dropped?}, a handful of variants, precomputed.
* **Shortest path** on a visibility graph over that polygon; each polyline is
  costed with the motion model: v ≤ 220 mm/s, the measured accel ramp, arc
  blends at R ≥ v/ω_max (191 mm at full speed — tighter corners take a
  computed slow-through speed, v = ω_max·R), pivots only at true reversals,
  costed from the measured `turn_to` table.
* Station entry/exit headings are fixed (§7.1), so `T[i][j]` includes the
  turn-out and turn-in exactly once, correctly.

The first cost matrix is calibrated against `prof.py` measurements of the
existing route (same legs, known times); disagreement > 10 % is a model bug to
fix before the optimizer is trusted.

### 7.3 The solver

Twelve-ish nodes: **exact dynamic programming (Held-Karp over subsets with
precedence pruning)** — 2^12 × 12 states, milliseconds in Python.  Value =
score of visited prizes; feasibility = arrival times within 120 s with the
duration models; tie-break = finish earlier.  No metaheuristics, no tuning: at
this size the optimum is simply computed.

Output: a **schedule** — ordered (node, planned start, planned duration,
slack), replacing `MATCH − BEAM_BUDGET − KIT_BUDGET` and every other constant
deadline in route.py.

### 7.4 Replanning

The same DP re-runs from current state (position, time, cargo, world state)
whenever: a phase ends off-plan by more than its slack, a stall/blocked event
fires, or perception changes the board (stray sample seen; quarantine
confirmed empty).  Milliseconds per replan means the executor can afford to do
it at every phase boundary.  This is what finally makes the third slot
rational: it is attempted exactly when the measured remaining cost of
{kits, beams} leaves ≥ one measured dock time, not when a constant says so.

---

## 8. Trajectory generation and tracking control

* **Trajectory generator**: planner polyline → arc-blended path (blend radius
  ≥ 191 mm where speed allows, else slow-through) → time-parameterized
  velocity profile under v_max, a_max (measured from the stepper ramps), and
  corner speed caps.  Transit legs touch v = 0 **only** at service poses and
  true reversals.
* **Path tracker**: pure pursuit with speed-proportional lookahead plus
  cross-track PD, running on `est.pose` — an upgrade of the proven `pursue`
  and the crab-capped line law from `stall_drive`/`line_drive` (gains and
  caps carry over: they were measured against this chassis).
* **Terminal library** (kept, re-hosted on the HAL): `stall_drive` (guarded
  wall stall), `reverse_track` (camera-vector reverse with trim handoff),
  `dress_safe`/`dress_onto_line` (differential-drive park), seating pushes,
  `dwell_until_loaded` (process control).  These already close on relative
  measurements; they are the part of the current system that was *right*.
* **Adopted with the new schedule**: stall-terminated beam-1 seating (0.5 mm
  vs 10 mm, +1.3 s) and the beam-1 protrusion station — both proven, both
  currently reverted only for phase cost.

Expected recoveries against §2 (bounded by measurements, to be verified by the
profiler, not asserted): pivot elimination on transit ~8–12 s, no
stop-at-waypoint chaining ~4–6 s, PCC_R spur 5–7 s, budget slack reclaimed
2–5 s.  The third slot needs 1–5 s; both beams and all three slots need
~10–15 s.  The margin is there.

---

## 9. Mission executor

A thin sequencer, not a framework: consume the schedule; run each node's
behaviour generator with its guard; stamp actual vs planned (the analysis loop
is *built in*, per-phase, every run); trigger replans (§7.4); apply the yield
policy on projected overrun — drop the lowest score-per-remaining-second node,
which is the existing deadline idea generalized from one hard-coded comparison
to all of them.  Logs stay in the current honest style: measured, stamped,
grep-able.

---

## 10. Robot 2 — interface reserved now, built in step #3

Decided architecture (not revisited here): a small differential-drive robot —
two DC motors, two drivers, small wheels, a small battery, a plow, a Pico W —
**no camera, no autonomy**.  It is in effect a *detached actuator of robot 1*:
robot 1's Pi 5 and stereo rig are its perception and its planner, the
Bluetooth link is its wiring loom, and the hardware being imperfect is
tolerable because every job it has is push/pull with a plow (position
tolerance comes from the plow's width) plus the kit-shake manoeuvre, which is
perfected in simulation before the firmware ships.  This document only fixes
the contract so the planner and HAL are ready:

* **LinkHAL protocol v0** (fits in one line each, ASCII, checksummed):
  `V <left> <right> <ms>` (velocity for a duration), `K` (keepalive/stop),
  `SHAKE <n>` (the escapement shake manoeuvre).  20 Hz command rate, dead-man
  stop on 250 ms silence.
* **Tracking**: one ArUco marker, top plate (§5.4).
* **Tasking**: robot 2's node set (PCC_R kit drop first, then sticker→zone
  plow legs per classified colour) runs through the *same* planner machinery;
  the two schedules exchange keep-out time windows (robot 2 stays off robot
  1's active corridor and vice versa — with PCC_R offloaded their default
  territories barely intersect).
* In simulation, robot 2 is a second MJCF body driven through the same LinkHAL
  interface, so its control software is also sim-tested unchanged.

---

## 11. Scoring model used by the planner

From the referee (all measured, Senior table): samples 50 (three slots),
beams 70 (both, T-joint within tolerance), kits: +3/kit in zone, −10/empty
zone, +20 exact 6/2/2; patients +5 right / −5 wrong / −3 adrift (robot 2's
ledger, −36 if untouched).  Robot-1 subtotal ceiling after the PCC_R handoff:
**134** (50 + 70 + 14).  System ceiling with robot 2 nominal: **250**.
Current: ≈127 with three kit zones; target after this rebuild: **≥ 128 of
134** robot-1 subtotal (3 slots, both beams with T-joint, 8/8 kits, ≤ 6 pts
conceded to seed variance), inside 115 s so 5 s of reserve absorbs hardware
reality.

---

## 12. Test and validation plan

The two standing suites stay and gate every commit (48 geometry + 20 model
checks).  New, in the same style — small, loud, measuring the thing that
matters:

* **check_planner.py** — fixed boards → schedule respects precedence, fits the
  clock, is deterministic, and its predicted leg times match `prof.py`
  measurements within 10 %; perturbed starts still solve.
* **check_perception.py** — rendered frames at known poses → slot error ≤ 2 mm
  at ≤ 200 mm, both range estimates agree, refusal on disagreement; 12/12
  sticker colours; marker pose error ≤ 5 mm at 1 m.
* **check_estimator.py** — 12-seed runs → |est − truth| ≤ 15 mm for 95 % of
  the match, and the terminal phases *never consume* pose worse than their
  gate (this is the check that proves ground-truth removal, by construction).
* **Board regression** — `mrun` 12 seeds: steps 1–3 of the migration must hold
  +93.2 (pure refactors); steps 4–6 must beat it on the robot-1 subtotal; the
  schedule's predicted vs actual per-phase report ships in `diag.py`.
* Later, on hardware: the same behaviours against the PiBackend, calibration
  residuals into the same YAML, and the sim's estimator-error report re-run
  against motion-capture-free truth (wall-touch audits).

---

## 13. Build order

Each step lands committed with its checks green; no step mixes a refactor with
a behaviour change.

1. **hal.py + SimBackend** — extract the interface from `robot.py`; route.py
   consumes the HAL.  Pure refactor: board holds +93.2.
2. **Cameras in MJCF + rendered perception** — `mujoco.Renderer` frames, slot
   pipeline, A/B against the synthetic `see_lab` model; `check_perception`
   green; dock still lands on rendered vision.
3. **Odometry + estimator** — primitives move from `rb.pose`/`d.qvel` to
   `est.pose`/odometry; ground truth quarantined to referee/checks;
   `check_estimator` green; board holds.
4. **planner.py** — cost model calibrated against `prof.py`; DP solver;
   schedule replaces the constant budgets; **`KIT_AGENT_A` flips to
   ("HOSP", "PCC_L") here**, with the plan that exploits it.
5. **trajectory.py + tracker** — arc-blended transit, executor on the
   schedule, stall-seated beam 1 re-adopted; per-phase actual-vs-plan report.
   *Landed.*  Board **+65.5** (from +53.4), docks 85%, the full robot-1
   subtotal of 134 reached on two seeds.  The step's find (F88): the kit
   departure pivot at the dock line was never legal — the tail's 185 mm
   sweep vs 155 mm to the plate edge — and the silent grind was hiding
   inside the "15.5 s" hospital leg; cured by a 55 mm back-away plus one
   strict-knee pursuit to the lip (KH block 24 s → 15).  Est-nav re-run
   after the fix: **−2.3** with docks at 24% and the sweep itself
   degrading — the estimator suite is green (median 14.6 mm, p95 < 90),
   so the gap lives in the *behaviours'* tolerance to belief error, not
   in the filter; the flip stays deferred and step 6 owns the diagnosis.
6. **Re-profile and close** — the analysis loop on the *optimal* plan; chase
   the residuals the report names, not hunches.  Exit: §11 target on 12 seeds.
   Named residuals from the step-5 boards: dock service variance 10–16 s
   (six seeds trade PCC_L's 16 pts for the seal under slow docks), the
   beam-1 tail (staging 13.3 s measured vs 18.5 budgeted for the whole
   tail; two crab-arrival 10 mm misses), and the est-nav dock/sweep gap.
7. **Step #3 of the project** — robot 2 (second MJCF body + LinkHAL + its
   schedule), then PiBackend bring-up on the bench.

---

## 14. File map

| file | today | after |
|------|-------|-------|
| `params.py` | single source of truth | unchanged (gains Vision pipeline + planner constants) |
| `mjcf.py` | model builder | + two `<camera>` elements, robot-2 body (step 7) |
| `robot.py` | actuation + synthetic sensing + ground truth | SimBackend of the HAL |
| **`hal.py`** | — | the interface (§4) |
| **`perception.py`** | — | §5 pipeline, both backends |
| **`estimator.py`** | — | §6 filter |
| **`planner.py`** | — | §7 graph, cost model, DP, schedules |
| **`trajectory.py`** | — | §8 paths + profiles |
| `route.py` | primitives + hand-written mission | controller/terminal library + behaviours; `mission_agent_a` shrinks to executor + schedule |
| `referee.py`, `check_*.py` | harness | unchanged, + three new checks (§12) |
| **`pi_hal.py`** | — | PiBackend (hardware bring-up) |

---

*The mechanisms are measured and settled; the clock is a computation we have
not yet done.  This design does it once, properly, and the same code that
proves it in MuJoCo drives the aluminium.*
