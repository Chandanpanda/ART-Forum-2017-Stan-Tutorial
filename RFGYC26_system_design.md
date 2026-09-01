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
   *First increment landed.*  The dock lever paid: the step-across misses
   all measured 37–45 mm, just past the 36 mm inline-absorption threshold,
   so the tracked reverse's crab authority went 5°→8° and the threshold to
   46 — **docks 94 %** (62/66 over 24 seeds), all-three-slots on 18/24,
   five seeds at the full 144 robot-1 ceiling; slots are effectively
   solved.  24-seed mean +57.2 with the beams as the one dominant variance
   term (b70 ×10 / b25 ×9 / b0 ×3, ~25 pts of swing a seed).  Tried and
   reverted, with the traces kept: moving the dock-line pivots to a
   "minimum-graze" y (the rigid south wall deflects worse than the plate's
   5 mm lip — stiffness beats depth, F89), and both a tail-first reverse
   capture and a capture+dress hybrid for beam 2's descent (staged
   beautifully and stalled 11 mm high at 83° three-for-three: the run-in's
   crab law needs its heading rendezvous on the CCW side, which the slow
   wall-grinding flip accidentally provides — the aim-taper redesign of
   stall_drive is the named follow-up, F91).  New finds: the beam-1 b25s
   are the F87 patient pinned between the SW nose corner and the west wall
   (contact-dump proven, cyl1 removal cures seed 9 outright — F90;
   robot 2 remains the cure); a planner hole let an aborted seal schedule
   PCC_L from inside the sealed box (RANK now binds the first hop too).
   Est-nav diagnosed end-to-end (F92): belief is excellent while fixes
   flow (camera-era p95 22 mm, post-fix median 3 mm) and the second half
   is a fix drought — no absolute reference north of the laboratory, 60–90
   mm by the seal, and the seal's shuffles defeat the sustained-stall
   freeze (alternating pushes reset the counter; the belief crossed the
   south wall by 270 mm while the truth wiggled in place).  Cure list for
   the est-nav pass: scheduled datum touches in the north half, a freeze
   that survives shuffles, and only then the F85 flip.  Remaining for this
   step's exit: the beam variance (robot 2's patient clearance buys most
   of it) and finishing inside 115 s.
7. **Step #3 of the project** — robot 2 (second MJCF body + LinkHAL + its
   schedule), then PiBackend bring-up on the bench.
   *Robot 2 exists and the fleet plays.*  The body (150×110, narrow for the
   two 191 mm field pinches, 120 mm plow pocket, front-low kit tray whose
   1.2 mm tail lip the SHAKE ratchet hops), the SimLink firmware end of the
   LinkHAL v0 wire (gain lottery, deadband, 250 ms dead-man, SHAKE as a
   firmware macro), and the Pi-side controller closed over robot 1's camera
   at 5 Hz with online gain calibration — swap SimLink for the Bluetooth
   socket and nothing above the wire changes.  **Fleet board: +63.3 mean,
   max +126 over 12 seeds** (robot-1 solo was +57.2): kits 10/10 with both
   bonuses proven, patients scoring for the first time (best p−10 against
   the −36 untouched baseline).  The hard lessons are F94–F98: the push
   catalog is wall geometry (mid columns push ±65° diagonals, edge columns
   only along themselves); pushes must be camera-verified per leg (open-loop
   pushes delivered nothing and nobody knew); there is no robot-2 lane
   through robot 1's half (six exit routes died on seed dice before the F82
   east-wall spawn — robot 1's start moved 144 mm west to make room); every
   interior parking spot belongs to some robot-1 artery, so robot 2 works
   the east columns EARLY (they are robot-1-free until the T+56 climb),
   holds the north-east dead corner through the climb, mops the climb-pile
   after T+74, and parks where nothing ever drives.  The WEST columns are
   deliberately untouched in this build: five configurations of west-side
   pushing each re-rolled robot 1's seal into failures — that corridor
   cannot host a 10–30 s noisy push pipeline and a seal.  **Roadmap to the
   250-ceiling / mean-200 goal, in order of yield**: (a) halve the push
   cycle (approach transits dominate; the primitives converge but spend
   8–20 s where the budget says 6–8) — that alone re-opens the west window
   before the seal and its ~+45 of patient/bonus value; (b) the beam-1
   tail rebuild (stall_drive aim-taper, F91) to lift b70 from ~40 % toward
   85 %; (c) the west-phase re-entry with a corridor-state validator
   (leave the corridor STRICTLY cleaner than found or do not enter);
   (d) the est-nav program (F92) for the day the field takes the oracle
   away.  PiBackend bring-up unchanged: BtLink speaks the same grammar.

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

## 15. Robot 2 control architecture — the rebuild

Watching the fleet play, the verdict is unambiguous and it is not a tuning
verdict: **robot 1 was built as an autonomy stack and robot 2 was built as a
script.**  Everything that looks wrong on screen follows from that one fact.

### 15.1 What is actually wrong

| symptom | mechanism |
|---|---|
| "basically blind, not reacting" | `_classify()` runs at T+0 and at two re-look boundaries. The plan is built from a **snapshot** and then executed to the end. Nothing re-decides. |
| "not looking at colour to make a route" | Colour *is* read — but it only selects one of a handful of **hand-written route templates** keyed on which sticker column the puck started in. The chain should be colour → destination zone → a route *computed* against the live board. It is colour → my hand-derived catalogue. |
| "gets stuck multiple times" | `goto()` is a straight-line heading-P pursuit. **There is no map.** The lab plate, the walls, robot 1, the other eleven pucks exist in no data structure the controller can see. It cannot avoid an obstacle because it has no obstacles. |
| "should be easy to route away from elevated areas" | Exactly — and today "routing" is *me* hand-picking waypoint chains like (880, 420) → (885, 700) and pasting them in. The plate must be **data in a costmap**, not a constant in a mission. |
| coordination is brittle | `stop_at=52`, `wait_until(74)` — wall-clock constants guessing where robot 1 will be, while robot 1's actual `Schedule` sits unread in the same process. |
| retries fail the same way twice | A jam backs off 90 mm and re-attempts the **identical** approach into the identical obstacle. No memory, no alternative. |

The design document already promised the cure and the build did not honour it:
*"robot 2's node set … runs through the **same planner machinery**."*  It does
not.  §15.2–15.7 make that true, and every symptom above becomes structurally
impossible rather than tuned away.

### 15.2 Layer 0 — the hardware model was wrong in two ways

**Wheels 22 mm → 32 mm radius (64 mm dia).**  Not cosmetic: the axle rises
10 mm, so plow ground clearance stops being a knife-edge constant (the "plow
dug into the floor" failure was cured with a 3 mm tweak — at 32 mm it cannot
happen), obstacles like tape edges and the box lip stop mattering, and the
caster-compliance nose-dive that caused it disappears with the geometry.

**Model the motor, not a clamp.**  Today: a velocity servo with an arbitrary
force clamp — unphysically stiff near zero speed and too weak at stall, which
is precisely the combination that reads as "gets stuck."  Instead model the
real part, an N20-class 6 V metal-gear motor, with its torque–speed line:

```
τ(ω) = τ_stall · (1 − ω / ω_noload)      τ_stall ≈ 0.06 N·m, ω_noload ≈ 21 rad/s
```

At 32 mm that is ~1.9 N of tractive force per wheel, ~3.7 N total against a
0.45 kg robot — roughly 8 m/s² available, and pushing force an order of
magnitude beyond what a 5 g puck needs.  The point is not "make it strong";
it is that **the acceleration limits the local planner (§15.6) reasons about
become physical numbers** instead of guesses.

**Add the encoders.**  Two quadrature encoders on a Pico W cost about two
dollars and turn dead-reckoning from *commanded velocity × learned gain* into
actual odometry.  The v0 wire already has an unused return direction
(`recv()` → `None`): give it `O <ticks_l> <ticks_r>` at 20 Hz.  This is the
cheapest accuracy upgrade on the entire robot and it makes the 5 Hz camera a
*corrector* rather than the only source of truth.

### 15.3 Layer 1 — the opening survey, then a tracker that verifies it

Perception has **two jobs with different tempos**, and conflating them is part
of what went wrong.

**The opening survey (once, ~0.5 s at the gun) is the important one.**  It is
the act that turns a randomised board into a known one: a stereo pass that
returns the twelve cylinder positions *with colours* and the three disc
positions.  Everything the joint plan (§15.5) decides rests on it, so it is
worth spending half a second doing properly — several frames, outlier
rejection, and an explicit confidence per detection.  A cylinder the survey is
unsure about is planned as *low confidence*, which the task selector prices
(verify-before-committing costs a second) rather than gambling on.

**The tracker (5–10 Hz, for the rest of the match) does not re-decide
anything** — it *verifies*.  It answers "is the world still where the plan
thinks it is," which feeds the invalidation triggers, the delivery checks and
the local detours, and nothing else.

Both use the same machinery, on robot 1's Pi, and neither uses `geom_rgba`:

1. **Detect** — colour-gated blobs on the field plane, through the same run-CCL
   and circle-fit machinery `perception.py` already uses to find lab slots.
2. **Back-project** — `Eye.ground()` puts each blob on the field in millimetres.
3. **Associate** — nearest-neighbour with a gate (Hungarian if crowded) into
   persistent tracks: `{id, x, y, P, colour, last_seen}`.
4. **Occlusion** — an unseen track keeps its position with growing covariance
   and a `stale` flag; the planner treats stale tracks as "probably there,
   verify before committing," which is a *policy*, not a crash.

Output is a live **`BoardState`** — twelve puck tracks, robot 2's ArUco pose,
robot 1's estimator pose, zone occupancy counts — published at 5–10 Hz.  Every
layer below consumes it.  This is what "reacting to the current state" means
concretely.

### 15.4 Layer 2 — the world model: a time-varying costmap

A 20 mm occupancy grid over the field: 58 × 60 ≈ 3 400 cells.  Four composed
layers:

* **Static** — walls, and **the laboratory plate as a hard obstacle**.  This is
  the user's "route away from elevated areas," and it is one line of map
  initialisation rather than a hand-drawn corridor.
* **Dynamic** — the twelve puck tracks and robot 1's current footprint.
* **Predictive (space–time)** — robot 1's *planned* corridors.  Its `Schedule`
  knows it docks L2 at T+41 and runs the kit dogleg T+58–73; rasterise each
  planned task's swept corridor into a time-indexed keep-out.
* **Sticky** — a decaying cost bump wherever a jam actually happened, so a
  retry cannot re-run the same failure.

Inflation is the standard two-radius scheme: hard-block inside the inscribed
radius (55 mm), exponentially decaying cost out to the circumscribed radius
(93 mm) — so the planner prefers corridor centres but *can* thread a 191 mm
pinch when the value justifies it.

### 15.5 Layer 3 — ONE joint fleet plan, computed once at the gun

**The board is fully observable at T+0 and deterministic thereafter.**  The
only randomness the match contains — the three sample positions inside the
quarantine, and which colour stands on which of the twelve stickers — is
*visible from the start line*.  Everything else (zones, plate, walls, beam
stations, kit hoppers) is fixed by the rulebook.  So this is not a problem
that wants continuous re-decision; it is a **fully-observable deterministic
planning problem**, and the right treatment is the one `planner.py` already
gives robot 1: **solve it once, properly, and then execute the answer.**

That correction matters, because an early draft of this section had the task
layer re-solving at 2–5 Hz.  That is wrong, and not merely wasteful:
continuously re-deciding *what to do* makes a robot thrash between plans as
noise moves the objective around, and it destroys the property a competition
robot needs most — **a plan you can inspect, validate and dry-run before the
match starts.**  Feedback belongs in the *execution* of a plan, not in the
*choice* of one.  The distinction is the whole architecture:

> **Plan once. Track continuously. Repair on invalidation.**

**The plan is a single fleet-wide object**, not two plans that avoid each
other at runtime:

1. **Perceive** (~0.5 s at the gun): one stereo pass gives the twelve cylinder
   positions with colours and the three disc positions.  Combined with the
   fixed geometry, the world is now completely known.
2. **Assign** tasks to robots.  Mostly forced by capability — robot 1 owns the
   discs, the beams, HOSP and PCC_L; robot 2 owns PCC_R and the cylinders —
   but the *split* is an output, not an axiom: if robot 1's tour comes back
   with slack it can take a cylinder, and if it comes back overloaded it can
   shed the PCC_L kits.
3. **Route each robot** — a prize-collecting tour (§15.5a) over its own tasks,
   where each task's cost comes from the push planner (§15.5b) and the path
   planner (§15.5c) rather than from a straight-line guess.
4. **Coordinate the two into one space–time plan** (§15.5d) so the routes
   cannot collide, by construction, before the match starts.
5. **Emit** a *timed reference plan* per robot: an ordered list of
   `(action, path, expected start, expected end)`, plus the shared
   reservation table.  This object is the contract.  It can be printed,
   plotted, replayed, and checked against the referee's scoring *before a
   single motor turns.*

**Repair, not replan.**  Execution deviates — cheap motors, uncertain pushes,
bimodal dock times.  The plan is only re-solved when a **precondition is
actually violated**, and then only the affected suffix:

| trigger | repair |
|---|---|
| a push left the cylinder > 60 mm from its predicted spot | re-solve that delivery's remaining legs from the observed position |
| a robot is > 6 s behind its scheduled task start | re-solve the remaining tour with the true clock (drops the cheapest task) |
| a task's precondition is gone (zone occupied, cylinder moved by the other robot) | re-solve that task, keep the rest |
| a path is blocked by something not in the plan | local A* detour around it, same plan |

Nothing else triggers a re-solve.  A re-solve costs microseconds, so the
constraint is not compute — it is **stability**: the plan should change when
the world genuinely diverges from it and at no other time.

**(a) Task selection — the same prize-collecting solver robot 1 uses.**
Nodes are deliverable pucks plus the kit drop.  Value is the referee's
marginal points, *including the set bonuses* — 4 reds in HOSP +6, yellows 2/2
across the PCCs +8, 4 greens in RECOVERY +6.  Those bonuses make combinations
worth more than the sum of their parts, which is exactly what a hand-ordered
priority list can never see and what an optimiser sees for free.  Cost comes
from (b) and (c), measured.  Constraints: the clock and the space–time
windows.  Twelve nodes with dominance pruning solves **exactly**, in
microseconds — so the opening tour is genuinely optimal rather than
heuristic, and it is re-solved only on the invalidation triggers above,
precisely as `planner.Schedule.complete()` already does for robot 1.  Reuse
`planner.py`; do not write a second one.

**(b) Push planning — non-prehensile manipulation as a search.**
For a puck at **p** bound for zone Z, search over straight pushes:

* *action*: push along heading θ (16 discrete) for distance d (quantised);
* *feasible* iff, **computed from the costmap**: the approach pose
  `p − θ̂·(PLOW_X + margin)` is free *and reachable* by (c); the swept corridor
  of body-plus-plow along d is clear of everything but the target; and the
  release pose is free so the robot can back out;
* *goal*: puck inside Z's inset rectangle;
* *cost*: approach + push + release time;
* *search*: A*, heuristic = range-to-zone ÷ push speed.  Depth ≤ 3, branching
  16 × 5 — milliseconds.

This **derives** the "±65° diagonals from mid columns, edge columns along
themselves" catalogue instead of asserting it, and it handles the layouts the
catalogue cannot: displaced pucks, robot 1's plow-pile, any referee
randomisation.  Crucially, *"this puck is unreachable"* becomes a computed
output — infinite cost, so the task selector spends those seconds elsewhere —
instead of a hand-written `continue`.

**(c) Path planning — A* on the costmap, computed with the plan.**
Grid A* (Theta* if we want any-angle smoothness) from pose to approach pose,
over the composed map, sampling the space–time layer at the *planned* arrival
time.  3 400 cells is sub-millisecond, so every leg of the opening plan gets a
real, obstacle-free path at the gun — **this is what ends "gets stuck," and it
is what routes around the plate: by construction, with no hand-picked
waypoints.**  At run time the path is a *reference to be tracked*, not a
suggestion to be recomputed; only an obstacle that is not in the plan (the
other robot out of position, a cylinder somewhere unexpected) triggers a local
detour, which is a small bounded A* between two points of the existing path.

**(d) Fleet coordination — the two routes are solved as one.**
Robot 1 is the higher-priority agent: it carries 144 of the board's points and
its behaviours have the tighter tolerances, so it plans first and freely,
producing a **reservation table** — its swept corridor as a function of time.
Robot 2 then plans in the *space–time residual*, i.e. its A* runs on a map
where robot 1's corridor is blocked **only during the window robot 1 occupies
it**.  That is prioritised planning, the standard and by far the cheapest
multi-agent path-finding scheme, and it is exactly right here because the
priority order is not a tie.

Two properties come out of doing this at the gun rather than reactively:

* **Collisions are impossible in the plan**, so the runtime job shrinks to
  *tracking* and the two robots stop negotiating for space at 20 Hz.  Every
  robot-on-robot failure this project has measured — the 35 s wrestle, the
  mutual corner-lock, the three wrecked sweeps, the seal re-rolls — was two
  independent plans discovering each other at run time.
* **The "leave-clean" invariant becomes a planning constraint** rather than a
  hope: robot 2 may not *deposit* a cylinder inside any corridor robot 1's
  reservation table claims later.  The push planner simply refuses those
  push targets.  That is the principled form of F87/F90 (the seal-corridor
  patient) and F98 (every interior park belongs to some artery), and it means
  robot 2's parking spot is computed — the free cell maximising distance from
  all remaining reservations — instead of being a constant I picked.

If the residual ever leaves robot 2 with no feasible route to a valuable
cylinder, that is a *fleet* answer, not a deadlock: the joint solve can pay
robot 1 a few seconds of delay to open the corridor, and compare the two
totals.  A single objective over both robots is the only way that trade is
even expressible.

### 15.6 Layer 4 — control: three nested loops, and we only have one

A plan is worthless if the robot does not execute it, and "does robot 2 have
control to make sure it follows commands?" has an uncomfortable answer today:
**there is an outer loop and no inner loop.**  The full cascade a cheap DC
differential drive needs is three loops at three rates, each closing a
different error:

| loop | rate | runs on | feedback | closes |
|---|---|---|---|---|
| **wheel velocity** | 100–200 Hz | Pico W | quadrature encoders | *this wheel is not turning at the speed I asked* |
| **path tracking** | 10–20 Hz | Pi | pose belief | *the chassis is off the planned path* |
| **task verification** | per leg | Pi | camera | *the cylinder did not end up where the plan said* |

Today we have the middle loop (a proportional heading law on a camera-corrected
belief) and the outer one (per-leg re-spotting, F96 — worth +21 points the day
it went in).  **The inner loop does not exist at all.**  The Pico is told
"200 mm/s" and sets a PWM proportional to it; whether the wheel actually turns
at 200 mm/s depends on the motor's gain, the battery's charge, the load on the
plow, and the deadband.  The Pi compensates with a *single learned scalar* per
robot, which cannot represent any of those — they are per-wheel, nonlinear,
and time-varying within a match as the cells sag.

**The fix is the cheapest item in this document.**  Two quadrature encoders
(~$2) and ~30 lines of firmware: a PI controller per wheel on measured
velocity, at 100 Hz, on the Pico.  Then `V 200 200 150` on the wire *means*
200 mm/s, and every layer above it stops paying for the lie.  Concretely it
buys:

* the ±15 % gain lottery, the deadband and battery sag are all rejected by the
  inner loop instead of being modelled by the outer one;
* dead reckoning between 5 Hz camera fixes becomes **odometry** (encoder ticks)
  instead of *commanded velocity × a scalar* — at 300 mm/s that is 60 mm of
  open-loop travel per fix interval today, and a few millimetres after;
* the planner's time estimates become trustworthy, which is what makes a
  *timed* fleet plan (§15.5d) executable at all;
* it is the one change that ports to the real robot with no calibration
  argument: encoders measure the truth on both sides of the sim boundary.

The v0 wire already has the return direction to carry it (`recv()` → `None`
today): add `O <ticks_l> <ticks_r>` at 20 Hz, and the same LinkHAL contract now
closes the loop on hardware exactly as it does in MuJoCo.

**The middle loop becomes a path tracker, not a point-chaser.**  `goto()`
today drives at a *point*, which is why it cuts corners into obstacles; the
plan now hands it a *path*, and the tracker's job is to stay on it.  Replace
the proportional heading law with a **Dynamic Window Approach** at 10–20 Hz:

* sample the admissible (v, ω) window from current speed and the *physical*
  acceleration limits §15.2 now provides;
* roll each candidate forward ~1.2 s through the differential-drive model,
  using the per-wheel gains the controller already learns online;
* score = w₁·progress along the global path + w₂·clearance from the costmap
  + w₃·speed − w₄·path deviation − w₅·control effort;
* execute the winner, decomposed to wheel commands with deadband compensation.

Pure pursuit — which `trajectory.py` already implements for robot 1 — is a
*path follower*: it assumes the path is safe.  DWA is a *local planner*: it
refuses to drive into things, slows for clearance, and produces the
route-around-the-plate behaviour without hand-holding.  For the push phase the
same controller runs with a tightened window (low v, small |ω|) so the puck
stays in the pocket.

The principled version of the final push segment is a short-horizon **MPC on
the puck**: predict puck motion through the plow contact model and optimise
(v, ω) so *the puck* tracks the push line.  DWA-with-tight-limits is the 90 %
version at a tenth of the complexity; start there, and keep MPC in reserve for
the last few millimetres of zone-edge precision.

### 15.7 Layer 5 — executor and fleet coordination

**Behaviour tree with a real recovery ladder:**
IDLE → NAVIGATE(approach) → ALIGN → PUSH → RELEASE → VERIFY, and on failure
escalate only as far as needed: (i) re-spot and re-verify *(exists today)*;
(ii) back out and replan the **path**; (iii) mark sticky cost and replan the
**push** — a different approach direction; (iv) abandon and re-solve **task
selection**.  Today only (i) exists, which is why a jam retries into the same
wall.

**Coordination becomes prioritised planning, not clock constants.**  Robot 1 is
the higher-priority agent and plans freely (it already plans optimally); robot
2 plans in the space–time residual.  Both live on the same Pi, so robot 1's
schedule is a function call.  Two rules replace every magic number:

* **hard** — robot 2's path may not intersect a corridor robot 1 has reserved
  during the window it is reserved;
* **soft, the "leave-clean" invariant** — robot 2 may not *place* anything (a
  pushed puck, or itself) inside a corridor robot 1 will use later.

That single invariant is the principled form of two findings we paid for
empirically: the F87/F90 seal-corridor patient, and the F98 discovery that
every interior parking spot belongs to some robot-1 artery.  Parking stops
being a constant and becomes a computation: *the free cell maximising distance
from all of robot 1's remaining reserved corridors.*

### 15.8 Cost, and why this is the cheap path

Everything above fits comfortably on a Pi 5 at 10 Hz: A* over 3 400 cells,
~200 DWA rollouts, a ≤12-node prize-collecting solve — single-digit
milliseconds per cycle.  In code it is roughly: costmap 150 lines, A* 80, push
search 120, DWA 120, tracker 150, behaviour tree 150 — ~800 lines, plus
`planner.py` reused rather than duplicated.

That is less code than the accumulated hand-tuned mission it replaces, and the
difference in kind matters more than the size: the score would then follow
from **capability** rather than from constants I fitted to twelve seeds.  The
honest retrospective on why this happened: robot 2 was built inside-out (body →
primitives → mission), with each layer patched reactively when a board came
back bad.  Robot 1 was built outside-in (perception → estimator → planner →
control → mission) — and robot 1 is the one that works.

**Build order for the rebuild**, each step independently measurable on the
fleet board:

1. **Costmap + A\*** — kills the stuck failures and deletes every hand-picked
   waypoint.  Largest behaviour change per line of code.
2. **Encoders + per-wheel PI on the Pico** — ~30 lines and $2, and it is what
   makes every command above it mean what it says.  Do it early: the layers
   above are all calibrated against a drive that currently lies.
3. **Push search** — unlocks the west columns and every displaced cylinder;
   turns the hand-derived catalogue into a computation.
4. **The joint fleet plan** — one solve at the gun, reservation table,
   leave-clean as a constraint.  Retires every wall-clock window constant.
5. **DWA path tracking**, then the survey/tracker split, then folding robot
   2's task selection into `planner.py`.

The first four are where the score is; the fifth is where the robustness is.

---

*The mechanisms are measured and settled; the clock is a computation we have
not yet done.  This design does it once, properly, and the same code that
proves it in MuJoCo drives the aluminium.*
