# IMU bat + iPhone camera: how accurate is it at contact?

Simulation results, 1 October 2026. Every number here comes from a model, not
from a measurement. The script is `imu_fusion_sim.py` in this folder, and the
full output is in `results_raw.md`.

## Bottom line

- **With a per-unit calibrated IMU**, the bat's sweet spot is known to about
  **6 to 12 mm typically, and 12 to 31 mm for the worst 5% of swings**. That is
  the error across the ball's line at the moment of contact.
- **Without per-unit calibration** it is **2 to 4 cm typically** (4 to 8 cm
  for the worst 5%). That is the "few cm" figure from earlier in the thread.
- **The stereo global-shutter console** would get about **1 mm** across the
  ball's line (analytic estimate).
- **Depth errors are mostly harmless.** The phone sits at the TV, so its weak
  axis (depth) lies along the ball's line. An error there mainly shifts *when*
  contact happens, by about a millisecond, which moves the shot direction by a
  degree or two.
- **Compute, heat and latency all favour the IMU bat.** Contact latency is set
  by the Bluetooth link, not the camera.

## What was modelled

- **The swing.** A bat swings through a plane: a still stance, a 0.45 s
  backlift, then a downswing that peaks at contact. The hands ride along.
  There are four swings: a kid's drive, an adult drive, a hard drive and an
  adult pull (a flat swing). The IMU sits in the handle cap, 60 cm from the
  sweet spot.

  | swing | peak gyro (dps) | peak accel at IMU (g) | sweet-spot speed (m/s) |
  |---|---|---|---|
  | kid, drive | 1004 | 2.0 | 11.0 |
  | adult, drive | 1506 | 5.0 | 19.7 |
  | hard, drive | 2317 | 11.9 | 30.3 |
  | adult, pull | 1506 | 5.2 | 19.7 |

- **The IMU.** The model includes bias, scale factor, cross-axis, nonlinearity,
  g-sensitivity, noise, quantisation and saturation. It also includes how the
  board sits in the bat and where it sits. The values are typical
  consumer-MEMS datasheet figures from memory, so check them against the part
  you choose. "Calibrated" means a per-unit calibration station measured scale,
  cross-axis, bias and mounting.
- **The camera.** It is used only as an anchor while the bat is slow: at the
  end of the stance, or at the top of the backlift. From there the IMU alone
  carries the estimate to contact. The anchor errors are derived from the
  iPhone's geometry at 4 m: 2 mm across, 15 to 18 mm in depth. The IMU's
  heading relative to the camera is assumed known to 1°. Blurred mid-swing
  frames are not used, which makes the model pessimistic. A real filter would
  use both anchors and the blurred frames.

## Error at contact, across the ball's line (mm, median / 95th percentile)

| swing | calibrated, stance anchor | calibrated, top anchor | uncalibrated, stance anchor |
|---|---|---|---|
| kid, drive | 6.5 / 13.2 | 9.2 / 20.9 | 30.2 / 61.0 |
| adult, drive | 6.4 / 13.4 | 10.5 / 28.0 | 24.8 / 51.9 |
| hard, drive* | 5.8 / 12.2 | 9.9 / 26.5 | 20.8 / 42.6 |
| adult, pull | 7.8 / 16.8 | 12.3 / 31.2 | 38.0 / 78.3 |

\* Calibrated columns use a ±32 g / ±4000 dps part; the uncalibrated column uses ±16 g / ±2000 dps.

Orientation at contact, calibrated (95th percentile): the bat axis is within
0.6° for drives. For the pull the bat-axis error is 1.9°, because heading
error shows up there. The face angle is within 2.0° for drives and 0.8° for
the pull. Both follow the heading error almost one to one.

**A ±2000 dps gyro saturates on the hard swing.** The bat-axis error then
jumps to 6.8° and the depth error to 66 mm. Use a ±32 g / ±4000 dps part.

## What drives the error

For a calibrated IMU, adult drive, the largest single sources are (median /
95th percentile, mm):

| source | stance anchor | top anchor |
|---|---|---|
| heading of IMU vs camera (1°) | 3.1 / 9.0 | 8.8 / 25.4 |
| camera anchor position | 2.4 / 4.9 | 2.5 / 5.1 |
| camera anchor velocity | 1.5 / 3.2 | 3.6 / 7.4 |
| accelerometer bias (2 mg) | 1.6 / 5.0 | 2.2 / 5.5 |
| accelerometer nonlinearity | 2.8 / 5.2 | 0.1 / 0.3 |
| everything else, each | 2.4 or less | 2.8 or less |

Without calibration, the accelerometer bias (12.5 / 37.5) and the gyro and
accelerometer cross-axis terms (10.4 / 28.8 and 9.2 / 21.3) take over.

**What this means:**

- **The heading error grows with how far the bat travels toward or away from
  the camera since the last anchor.** That is why the top anchor, with about
  0.5 m of travel to contact, is worse than the stance anchor, where the bat
  comes back near where it started. The filter must keep re-estimating
  heading on every swing. The bat's travel in depth is exactly what makes
  heading observable.
- **Once the IMU is calibrated, the camera and the filter limit the accuracy,
  not the IMU.**

## Sensitivity (adult drive, calibrated, stance anchor)

| change | median / 95th (mm) | face angle 95th (deg) |
|---|---|---|
| baseline (heading 1.0°) | 6.4 / 13.4 | 2.0 |
| heading known to 0.5° | 5.4 / 11.4 | 1.1 |
| heading known to 0.3° | 5.2 / 11.1 | 0.7 |
| heading known to 2.0° | 8.5 / 20.2 | 4.0 |
| poorer camera (1 px noise, 5 mm bias) | 8.4 / 17.1 | 2.0 |
| IMU warmed up (gyro scale 0.15%, accel bias 4 mg) | 7.5 / 16.4 | 2.0 |
| no per-unit calibration, biases learned online | 18.0 / 39.7 | 2.7 |
| no per-unit calibration at all | 24.8 / 51.9 | 2.7 |

**Per-unit calibration is the single largest lever.** It is a station your
assembly line could run: a turntable or tumble jig at the end of the line.

## How fast the IMU-only estimate degrades

Adult drive, calibrated, last camera fix at the end of the stance:

| time since fix (s) | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.65 (contact) |
|---|---|---|---|---|---|---|
| median (mm) | 2.9 | 6.8 | 8.1 | 7.5 | 8.0 | 6.4 |
| 95th (mm) | 6.5 | 18.7 | 22.2 | 19.0 | 19.9 | 13.4 |

Even if the camera misses the top of the backlift (an occluded bat), the
estimate stays near 1 to 2 cm for most of a second.

## Compute, latency, heat, cost

| | IMU bat + iPhone | Global-shutter stereo console |
|---|---|---|
| across-ball accuracy at contact | 6 to 12 mm typical, 12 to 31 mm worst 5% (calibrated) | about 1 mm |
| latency to a contact decision | about 10 to 20 ms over Bluetooth (estimate). Late camera fixes only correct the past. | about 10 to 20 ms: exposure, readout, ROI matching, Wi-Fi (estimate) |
| extra work on the iPhone | finding two colour bands in a small region the IMU predicts. This should be negligible next to segmentation; worth measuring on the hot phone. | none for tracking, but compositing needs registration between the two devices |
| fusion compute | error-state Kalman filter at 1 kHz, microseconds per step, can run on the TV | stereo ROI matching on a Pi-class board |
| parts | bat about INR 1,000, plus a calibration station | console INR 6,000 to 12,000, plus passive props |
| the prop | battery, charging, Bluetooth pairing | passive, battery-free, personalised |

- **Latency.** On iPhone, the shortest Bluetooth LE connection interval is
  about 15 ms. On Android it is 7.5 to 15 ms, so pairing the bat directly with
  the Google TV is the lower-latency path. I have not checked that the
  Streamer allows a custom Bluetooth LE peripheral.
- **Composited video.** At 30 fps the real bat is a streak about 30 cm long
  in the frame at contact. A 1 cm pose error cannot be seen there. In my
  judgment, about 1 cm is also below what a player can sense against a
  virtual ball. Where it can show is in edge calls (nicks): a ball within
  about a centimetre of the bat's edge can be called the wrong way.

## What would change this

- **Real swings could be harsher than modelled** (2 to 12 g at the handle
  here). More acceleration means more error and possible saturation. Measure
  this first.
- **Camera detection could be worse than assumed**, through lighting,
  occlusion or a stance that isn't still. The "poorer camera" row shows the
  effect.
- **The heading could be estimated worse than 1°.** That is the largest term.
- **Not modelled:** bat flex, Bluetooth packet loss and jitter, temperature
  swings beyond the "warm" row, and the filter's own tuning.

## Suggested next experiment

1. Put a ±32 g / ±4000 dps IMU logger in a prop bat.
2. Film real swings with the iPhone (for the anchors) and, at the same time,
   with your Pi stereo rig at high frame rate and short exposure as the
   reference.
3. Compare the IMU-propagated sweet spot against the stereo reference
   through the swing.

This also measures the real handle accelerations.
