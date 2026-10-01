# IMU bat: automated build and calibration line

A snapshot, 1 October 2026, of the design doc of the same name.

`sim/batline` is the design of an unattended line that builds, calibrates and tests the IMU bat, and the study that decided the bat. Nothing in it is wired into the check suites yet.

| file | what it is |
| --- | --- |
| `README.md` | this design: the bat, the stations, transport, calibration, testing, unattended running, capacity, how it maps onto `sim/truss` and `sim/rfgyc26`, build order, risks |
| `imu_fusion_sim.py` | Monte Carlo of how well an IMU bat, anchored by the phone camera, knows its sweet spot at contact. numpy only: `python3 sim/batline/imu_fusion_sim.py [--quick]` |
| `imu_fusion_results.md` | what that simulation found |
| `results_raw.md` | its full output |
| `line-layout.png` | the line at a glance |

Every number in this folder is a model or a planning estimate, not a measurement.

## Summary

One unattended line can build, calibrate and test the bat at 300 a month and stretch to 1,000. It uses the machines you already have: the truss assembler, the XYZ gantry, the 5-DOF arm, the mecanum robot and the multi-view camera rig. Only the electronics board is bought in, and customers fit their own AA batteries, so the line never handles a cell.

Running 16 hours a day, 300 a month means one finished bat every 96 minutes. At 1,000 a month it is one every 29 minutes. Speed is not the hard part; running for hours with nobody there is.

Five rules shape everything below:

- **Design the bat for the robot.** Every part goes in with one straight move against a datum. No cables, separate screws, batteries or loose fabric.
- **Measure, don't place precisely.** The camera rig measures where each part ended up, and those measurements become the bat's calibration, stored on the bat.
- **Coarse by robot, fine by dock.** The mecanum robot only brings a tray near a station. A kinematic dock sets its exact position.
- **Every bat is calibrated and tested.** A sample is also swung at full speed against a stereo reference.
- **Nothing waits for a person.** A failed bat goes to a reject rack, the line keeps running, and you get a message.

All times and costs here are planning estimates, not measurements.

## The bat, designed for the robot

The bat is six parts, and each goes in with one straight move against a datum. Only the board is bought.

| Part | Made by | Goes in by |
| --- | --- | --- |
| Handle core: battery tube, board slot, datum faces | printed in-house (my default) | gantry places it in the frame jig |
| Blade frame | truss assembler | built onto the core |
| Electronics board | bought, ready-assembled | slides into the slot and snaps against two datum faces |
| Battery contacts | bought springs | pressed into the core |
| Printed sleeve with two tracking bands | printed in-house | pushed off a mandrel onto the bat |
| End caps | printed in-house | pressed on along the bat's axis |

**The board** should carry:

- **An nRF52840-class microcontroller.** It runs straight from 2 AA cells (1.7 to 3.6 V) with no boost converter. Bluetooth streaming draws a few milliamps, so 2 alkaline AA should last on the order of 200 hours of play (estimate).
- **A ±32 g / ±4000 dps IMU**, for example the TDK ICM-42686-P (check the datasheet). The simulation showed ±2000 dps saturating on hard swings.
- **Test pads on one face for pogo pins:** power, programming and serial. The line powers, flashes and calibrates the bat without a battery.
- **A hold-up capacitor and brownout-safe firmware**, because AA cells can bounce off their contacts in a hard swing.
- **Wake on motion from the IMU**, so there is no power switch to fit.

**Three rules keep the calibration valid for the life of the bat:**

- The board seats in the fixed core, never in the removable battery cap. Otherwise every battery change would move the calibrated IMU.
- The customer fits 2 AA in line through a screw-on pommel cap. The line never touches a battery.
- The sleeve's two tracking bands sit in reserved zones of fixed colour, about 30 mm wide and 0.45 m apart. Everything else on the sleeve is free artwork.

## The line at a glance

![The robot carries every tray; docks and stations supply the precision](line-layout.png)

Every station has one dock. The robot sets a tray on it and later takes it away; all precise handling happens inside the station.

Steps 1 to 3 make identical frames to stock. An order enters at step 4 with its artwork, and the bought board only goes in at step 5, so boards are never tied up in unsold bats. You restock the stores and empty the two racks on your own schedule; nothing on the line waits for you.

## The stations

Each station reads the tray's tag and each bat's serial, does one job, checks its own work, and adds what it measured to the bat's record.

### 1. Print farm

Prints the handle cores, the blade-tip end caps and the pommel caps. Unattended printing needs printers that clear their own beds, such as belt printers or printers with an automatic plate changer, so finished parts drop into a tray at the dock. A camera at the dock checks the count, and that each core's board slot and battery bay are clear.

### 2. Frame cell

This is the truss assembler you already have. Its XYZ gantry loads carbon rods from magazines into a fixture on a collapsible mandrel, the C-ring winds Kevlar thread round each joint, and the dispenser puts epoxy on each band. For the bat, the handle core sits on the end of the mandrel, so the chords seat into its sockets and the frame is built onto it.

The bat's frame is a new task spec, and the existing code computes the fixture, the winding clearances and the cycle time from it. The cell's camera finds each joint before winding, and the inspector judges the finished frame from the ring's counted turns and where each rod was measured.

### 3. Cure rack

Wet frames cure on their mandrels in a rack, so the cell can start the next frame at once. When a frame has cured, the robot brings its mandrel back to the cell. The cell pulls the draw rod to collapse the mandrel (the one-pull release already designed in `collapse.py`), slides the frame off and reuses the mandrel. Finished frames wait on the rack as stock until an order needs them.

The number of mandrels follows from the cure time and the takt, so it is computed once the epoxy is chosen.

### 4. Sleeve

The one station that handles fabric, and the riskiest. It prints the customer's artwork, with the two tracking bands in their reserved zones, onto polyester knit by dye sublimation. A laser cuts the panel, which also seals the polyester's edges, and an ultrasonic head seams it into a tube; the cutter and the seam head ride an XYZ gantry.

A mandrel stretches the tube, a linear axis pushes the frame into it along the bat's axis, and the mandrel withdraws so the sleeve closes onto the frame. A press fits the blade-tip end cap, which clamps the sleeve's end. A camera checks the bands' colours and order against the artwork file; their exact positions are measured at step 6.

### 5 to 8. Finishing cell

The cell is built around the 5-DOF arm, which stays in this cell. The robot docks one tray of sleeved bats, and the arm takes each bat through four steps and puts it, packed, on an outbound tray.

- **5. Electronics in.** The arm slides the board into its own slot beside the battery bay, from the pommel end, until it snaps against its datum faces, then presses the contact spring in. A pogo block touches the board's pads at the pommel end to power it, flash the firmware, write the serial number and run a self-test: the IMU answers and reads gravity within its datasheet tolerance, and Bluetooth advertises. The board goes in this late so it never sees the epoxy's cure, and only ever into a bat that has an order.
- **6. Calibrate.** The arm loads the bat into a 2-axis gimbal inside the multi-view camera rig. The calibration section below covers what it measures.
- **7. Test.** A factory iPhone running the real app pairs with the bat and watches it at playing distance while the arm moves it slowly. A sample of bats also goes to an enclosed swing rig. The testing section below has the detail.
- **8. Pack.** The arm screws on the empty pommel cap, slides the bat into a cardboard tube with a quick-start card, caps the tube and applies a label with the serial and order number. A tube makes packing one axial move, which folding a box is not. A camera reads the label back against the order.

While the gimbal runs a calibration, the arm packs the previous bat and fits the next board, so one arm keeps up with the line.

## Moving bats between stations

- **Trays, never loose bats.** The robot carries trays. A tray holds several bats in V-cradles (four is my default) and carries a tag the stations read. Part totes, mandrels and packed tubes ride in trays of the same footprint, so one robot and one kind of dock move everything.
- **Coarse by robot, fine by dock.** Every dock is the same: two rails the robot drives between, carrying three balls. Each tray has three matching V-grooves underneath with chamfered lead-ins, a kinematic coupling. The robot only has to arrive within the lead-ins' capture range. When it lowers its lift, the tray drops onto the balls and lands in the same place every time, to well under a millimetre. The station's camera reads the tray's tag to confirm what arrived.
- **Where to stop is computed, never taught.** The stopping pose comes from the solver the competition robot already uses (`station.stand_for`), which ranks poses by their margin in millimetres against the dock's capture range. Moving a station is an edit to the floor map, not a re-teach.
- **Navigation.** Paths come from `nav.plan` on a costmap of the room built from the floor plan. The robot localises from wheel odometry and fiducial markers on the floor or walls, and the dock's camera corrects it over the last metre. Mecanum wheels let it slide sideways into a dock instead of shuffling.
- **One robot carries the volume.** Each tray makes about six to eight trips between print farm and outbound, shared by the bats on it, so a bat costs about two robot moves: a few minutes of robot time against a takt of 29 to 96 minutes (estimate).
- **But one robot is a single point of failure.** Every station's input and output buffers are sized to keep it working through a robot outage of the length you choose. A second robot then buys redundancy rather than throughput, and the fleet executive already in the code (`fleet.py`) handles two robots sharing narrow aisles.
- **Its own battery.** The robot charges at a dock between jobs, and the scheduler plans around its charge like any other resource. A LiFePO4 pack keeps the fire risk low; it is the least fire-prone common lithium chemistry.

## Calibration: the camera rig is the reference

The calibration cell needs neither a precise clamp nor a precise gimbal. The multi-view cameras measure where the bat actually is, at every moment, in 6-DOF, and the calibration fits the IMU's errors against that measured truth.

**What happens to each bat** takes a few minutes, mostly gimbal motion (estimate):

1. **Power and data.** The pogo block on the gimbal's clamp powers the board through a slip ring. The bat streams over Bluetooth, the link it will use in play.
2. **Geometry.** The cameras measure the bat's axis, its length, the sweet spot, and the two bands' positions and colours. These define the bat frame the game uses.
3. **Accelerometer.** The gimbal holds the bat still in a set of orientations, where gravity is the only input. The cameras measure each orientation, and a least-squares fit gives each axis's bias, scale and cross-axis error (the standard multi-position method). The same still periods give the gyro's bias and its sensitivity to gravity.
4. **Gyroscope.** The gimbal spins the bat through whole turns about each axis. The angle turned is known exactly from the counted turns, which the cameras confirm, so the fit gives each gyro axis's scale and cross-axis error.
5. **Mounting.** A spin about the bat's own long axis shows which direction the IMU takes as "along the bat", so the board's angle in the bat is measured, not assumed.
6. **Lever arm.** A spin with the IMU off the spin axis produces a centripetal acceleration that grows with its distance from the axis. With the cameras' geometry, that places the IMU relative to the sweet spot and the bands.
7. **Timing.** The offset between the bat's clock and the cameras' clock is estimated from the motion itself, by matching the gyro's rate to the rate the cameras see, as open-source camera-IMU calibrators such as Kalibr do.

**The target.** The simulation's 6 to 12 mm result assumed these residuals after calibration: gyro bias 0.02 dps, scale and cross-axis 0.05 %, accelerometer bias 2 mg, mounting 0.2°, lever arm 0.5 mm. Before any hardware exists, a check plants known errors in a simulated IMU, runs the gimbal plan through a camera model, and requires the fit to recover them to these values. Bias also drifts with temperature and each power-up, so the game keeps re-learning it in play; the factory's job is the terms that drift little.

**Keeping the rig honest.** Each shift starts with a golden bat, a reference bat measured many times. If its numbers move by more than the rig's own measured repeatability, the cell stops calibrating and messages you, rather than shipping bats calibrated by a drifting rig. The cameras are recalibrated against a target on the same trigger.

**Stored on the bat**, in its flash and in the factory database under its serial:

- serial number, firmware version, calibration date and temperature
- gyro and accelerometer bias, scale and cross-axis terms, and the gyro's gravity sensitivity
- the IMU's rotation and position in the bat frame
- the bat's geometry: axis, length, sweet spot, and the two bands' positions and colours
- a checksum

The game reads this record over Bluetooth when the bat pairs. The factory copy lets a corrupted record be restored, and lets a returned bat be compared with its day of manufacture.

## Testing: every bat slowly, a sample at full speed

**Every bat** is tested in the finishing cell straight after calibration:

- **Pairing.** A factory iPhone running the real app pairs with the bat and reads its calibration back.
- **Bands at playing distance.** The iPhone must find both bands from where a player's phone would stand. A mirror can fold that distance into the cell.
- **A slow fused swing.** The arm moves the bat along a slow path while the app fuses the IMU with its camera. The multi-view rig measures the same motion, and the fused sweet spot must agree with it within the product's accuracy budget for that speed. The limit comes from the budget, never from a number picked to pass.
- **Current draw**, asleep and streaming, measured through the pogo block. This catches a board that would flatten its cells early.

**A sample at full speed.** The arm cannot swing a bat at 20 to 30 m/s, so an enclosed swing rig beside the cell does. A motor swings the bat through a full arc while your Pi stereo rig, at high frame rate and short exposure, measures the true sweet spot to compare with the bat's own estimate.

During the swing the bat runs from a dummy pack of two AA-sized supercapacitors, weighted like two AA cells, which the rig charges. So the line still handles no batteries, and the swing still tests the contacts for bounce.

The first bat of each batch is sampled. After that, the sampling rate is set from how much bats turn out to differ, once a few hundred have been measured. Every result goes into the bat's record.

## Running with nobody there

- **One record per bat.** Each bat's record moves through fixed states: core printed, frame wound, cured, sleeved, electronics in, calibrated, tested, packed, then shipped or rejected. Every station writes what it measured. If the controller restarts, it rebuilds the line's state from the records and the tray tags.
- **Build to stock, finish to order.** Frames are made ahead whenever the frame cell is free. An order starts at the sleeve with its artwork, so it waits only for steps 4 to 8.
- **Who moves what next.** Stations, docks, mandrels and the robot are resources, and transport jobs are allocated by auction over one shared task list, as the repo's rules require. With one robot this reduces to a priority queue, and it extends to two unchanged. Travel times come from the path planner with a measured pace factor, never from a table.
- **Buffers instead of babysitting.** Buffer sizes are computed from the stations' cycle times and the longest outage you want the line to ride through, not picked. The stores hold enough for the time between your visits, and you get a reorder message when stock falls to a supplier's lead time.
- **Nothing hangs.** Every action has a timeout, the rule the competition robot's missions already follow. A step retries once only where a retry is safe, such as re-docking a tray. Otherwise the bat goes to the reject rack with its record, the station runs its own self-check, you get a phone notification, and the line carries on.
- **Stop only what threatens many bats.** A station that fails its self-check takes itself offline. A golden-bat failure stops calibration. A safety fault stops everything.
- **Safe when empty.** No cells are on the line. The printers, and the cure rack if it is heated, have independent thermal cut-outs, and a smoke detector cuts the power and messages you.
- **Visible from your phone.** The records and a camera view of the line are on your phone.

What you still do: restock the stores, empty the outbound and reject racks, and maintain the machines, on your own schedule.

## Capacity and cycle time

At 16 hours a day, 300 bats a month is one every 96 minutes, and 1,000 a month is one every 29 minutes. The times below are planning estimates unless marked as coming from the repo, and each is replaced by a measurement as its station is built.

| Station | Time per bat | At 300 a month | At 1,000 a month |
| --- | --- | --- | --- |
| Print farm | about 2 hours of printing, if a core and its caps print together (estimate) | 2 printers | 5 printers |
| Frame cell | not yet planned for a bat; the 300 mm test truss simulates at 9.4 min and the 1 m truss plans at 24.1 min (repo) | 1 cell | 1 cell, if the bat frame plans under 29 min |
| Cure rack | off the cell; set by the epoxy | mandrels = cure time ÷ takt, plus one | same rule |
| Sleeve | a few minutes of fitting; printing runs in batches (estimate) | 1 | 1 |
| Finishing cell | 5 to 10 min, mostly gimbal time (estimate) | 1 arm | 1 arm |
| Robot | about 2 moves per bat (estimate) | 1 | 1, plus 1 for redundancy |

Printers needed = printing hours per bat ÷ takt in hours, rounded up.

**What it means.** At 300 a month every station has slack, which leaves room for retries, maintenance and rejects. At 1,000 a month the printers and the frame cell set the pace, and both scale by adding a machine rather than by redesigning. Before anything is built, the whole line runs as a fast simulation with no physics, at both volumes and with robot outages and rejects injected, to size the buffers.

## How it maps onto the existing code

The line becomes a third fixture for the same codebase, beside the competition robot (`sim/rfgyc26`) and the truss cell (`sim/truss`). Most of the hard parts already exist in a general form. My default is a new package, `sim/batline`, with its own spec, hardware layer, model and check suite.

| Part of the line | Reuses | New work |
| --- | --- | --- |
| Frame cell | all of `sim/truss`: `spec`, `structure`, `fixture`, `collapse`, `approach`, `schedule`, `motion`, `process`, `hal`, `cell`, `vision`, `inspector` | the bat as a truss spec, swing loads in place of camera loads, the handle core as a fixture part |
| Docking | `station.stand_for` and `station.reachable` | the dock as the region, the lift as the effector, robot plus tray as the footprint |
| Robot transport | `nav` (its costmap already takes a size), `world`, `trajectory`, `estimator` | the workshop's floor map, and mecanum kinematics, since both competition robots are differential drive |
| A second robot | `fleet`: reservations, live avoidance, priority | aisles as the shared resources |
| Scheduling | the pattern of `planner` (an exact solver over a small task set) and the auction rule in `CLAUDE.md` | a flow-shop scheduler with buffers, with travel times from `nav.plan` and a measured pace factor |
| Calibration cell | the IMU error model from the fusion simulation (`imu_fusion_sim.py`, in this folder) as the simulated bat, and the two-implementation camera pattern of `truss/vision.py` | the gimbal, the calibration fit, the stored record |
| Test verdicts | `truss/inspector` and `rfgyc26/referee`: a pure function of what was measured against an acceptance table | the bat's acceptance table, from the accuracy budget |
| Hardware layer | both `hal.py` contracts: one file, a simulator backend and a machine backend, raise rather than lie | gimbal, pogo link, camera rig, arm, robot lift, printers |
| Review | `truss/filmstrip.py` contact sheets | none |

**New checks, each one a rig that can fail:**

- `check_calibration` plants known errors in a simulated IMU, runs the gimbal plan through the camera model, and requires the fit to recover them within the calibrated budget.
- `check_dock` compares the robot's arrival error, from the estimator, with the dock's capture range, as a margin in millimetres.
- `check_tray` measures the kinematic coupling's repeatability in MuJoCo, a mechanism question where physics earns its cost.
- `check_line` runs the whole line as a fast event model at 300 and 1,000 a month, with outages and rejects injected, and reports buffer levels and missed takt.

The repo's rules carry over unchanged: no dock pose, path, buffer size or pass limit is written by hand. Each is computed from the specs of the bat, the machines and the floor, so a new bat size is a new spec, not a re-tune.

## Build order

Each step replaces hand work, so bats can ship before the whole line exists.

1. **Measure real swings.** Put a ±32 g / ±4000 dps IMU logger in a prop bat, and film swings with the iPhone and your Pi stereo rig. This settles the IMU part, the accelerations the bat must survive, and how close the fusion gets to the simulation. It is the cheapest step, and every later one depends on it.
2. **The calibration cell, hand-loaded.** Gimbal, multi-view rig, calibration fit and golden bat. Prove that calibration buys the accuracy the simulation promised, by comparing calibrated and uncalibrated bats on the stereo swing test. This is the IP, so it comes first.
3. **The bat, designed for the robot.** Printed core, board slot, pogo access at the pommel end, sleeve. Hand-assemble the first 10 to 20 bats and put them in players' hands.
4. **Sleeve fitting**, prototyped alongside step 3, because it is the riskiest station.
5. **The frame cell for the bat**: the bat as a truss spec, the mandrel with the core, the cure rack.
6. **The finishing cell's arm**: board insertion, gimbal loading, the slow test swing and packing.
7. **Robot, trays and docks**, developed on the fast simulated line first, then on the floor.
8. **Stores and racks**, with stock counts and reorder messages.
9. **Unattended runs**: a watched overnight run, then unwatched ones, logging every stop and reject.

## Risks and open questions

**Risks, most serious first:**

- **Fabric handling.** Fitting a stretchy sleeve by machine is the step most likely to fail. If it cannot be made reliable, the fallback is printed panels that clip onto the frame.
- **Swings harsher than modelled.** More acceleration means more error, possible IMU saturation, and AA cells bouncing on their springs. Build step 1 measures this.
- **Carbon near the antenna.** Carbon fibre conducts and can weaken a 2.4 GHz antenna. Keep the antenna in the printed handle, clear of the chords, and test range early.
- **Heading at home.** The largest error in the simulation is the IMU's heading relative to the customer's phone. The factory cannot calibrate it, so the game must re-estimate it on every swing.
- **The robot and the floor.** Mecanum rollers dislike uneven or dusty floors, and one robot is a single point of failure. A sealed, clean floor and buffers sized for an outage cover both.
- **A frame the head can wind.** The winding ring sets a minimum truss section; the 300 mm test truss needed a 66 mm side. The bat's shape has to pass the existing clearance check before it is committed.
- **Epoxy cure** sets the number of mandrels and how long frames take to reach stock, so choose the epoxy with that in mind.
- **Radio approval.** Selling a Bluetooth product in India needs WPC approval; check what a pre-certified module on the bought board covers. I have not verified the current rules.
- **Google TV.** Whether the Streamer accepts a custom Bluetooth LE peripheral is unverified. The iPhone path works either way, at a slightly longer connection interval.

**Open questions for you:**

- **Bat sizes.** Kid and adult? Each is a spec rather than a redesign, but each needs its own tray cradles and mandrels.
- **Battery count.** Two AA in series suit the board directly. Three or four would need a regulator and add about 23 g per cell, so two is my default.
- **How long the line runs between your visits.** Overnight, a weekend or a week sets the size of every buffer and store.
