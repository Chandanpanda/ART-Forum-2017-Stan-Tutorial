# IMU bat + phone camera: error at contact (Monte Carlo)

1500 trials per row. Errors are for the bat sweet spot, 60 cm from the IMU. p50 = median, p95 = 95th percentile. Estimates from a model, not measurements.

Integrator check (no errors at all, 0.65 s from stance): 0.017 mm lateral, 0.102 mm depth.

## Swings

| swing | peak gyro (dps) | peak accel at IMU (g) | sweet-spot speed at contact (m/s) |
|---|---|---|---|
| kid, straight drive | 1004 | 2.0 | 11.0 |
| adult, straight drive | 1506 | 5.0 | 19.7 |
| hard, straight drive | 2317 | 11.9 | 30.3 |
| adult, pull (flat swing) | 1506 | 5.2 | 19.7 |

## Camera anchor assumed (derived in camera_anchor)

| anchor | position lateral / depth (mm) | velocity lateral / depth (mm/s) | heading (deg) | clock sync (ms) |
|---|---|---|---|---|
| top of backlift | 2.1 / 17.7 | 15 / 194 | 1.0 | 1.0 |
| end of stance | 2.0 / 15.4 | 2 / 2 | 1.0 | 1.0 |

## Error at contact

| swing | IMU | anchor | lateral p50 / p95 (mm) | depth p50 / p95 (mm) | bat axis p95 (deg) | face angle p95 (deg) |
|---|---|---|---|---|---|---|
| kid, straight drive | datasheet, no per-unit calibration | top of backlift | 18.9 / 43.6 | 45 / 129 | 3.17 | 2.69 |
| kid, straight drive | datasheet, no per-unit calibration | end of stance | 30.2 / 61.0 | 30 / 86 | 3.17 | 2.69 |
| kid, straight drive | calibrated per unit | top of backlift | 9.2 / 20.9 | 42 / 123 | 0.55 | 2.00 |
| kid, straight drive | calibrated per unit | end of stance | 6.5 / 13.2 | 14 / 41 | 0.55 | 2.00 |
| kid, straight drive | calibrated, 32 g / 4000 dps part | top of backlift | 9.2 / 20.9 | 42 / 123 | 0.55 | 2.00 |
| kid, straight drive | calibrated, 32 g / 4000 dps part | end of stance | 6.5 / 13.3 | 14 / 41 | 0.55 | 2.00 |
| adult, straight drive | datasheet, no per-unit calibration | top of backlift | 23.9 / 58.5 | 35 / 106 | 3.36 | 2.65 |
| adult, straight drive | datasheet, no per-unit calibration | end of stance | 24.8 / 51.9 | 25 / 75 | 3.36 | 2.65 |
| adult, straight drive | calibrated per unit | top of backlift | 10.5 / 28.0 | 30 / 95 | 0.60 | 2.02 |
| adult, straight drive | calibrated per unit | end of stance | 6.4 / 13.4 | 17 / 51 | 0.60 | 2.02 |
| adult, straight drive | calibrated, 32 g / 4000 dps part | top of backlift | 10.5 / 27.9 | 30 / 95 | 0.60 | 2.02 |
| adult, straight drive | calibrated, 32 g / 4000 dps part | end of stance | 6.5 / 13.5 | 17 / 51 | 0.60 | 2.02 |
| hard, straight drive | datasheet, no per-unit calibration | top of backlift | 24.5 / 57.1 | 66 / 152 | 8.67 | 2.84 |
| hard, straight drive | datasheet, no per-unit calibration | end of stance | 20.8 / 42.6 | 66 / 130 | 8.67 | 2.84 |
| hard, straight drive | calibrated per unit | top of backlift | 10.2 / 24.4 | 65 / 135 | 6.79 | 2.02 |
| hard, straight drive | calibrated per unit | end of stance | 6.2 / 12.4 | 66 / 121 | 6.79 | 2.02 |
| hard, straight drive | calibrated, 32 g / 4000 dps part | top of backlift | 9.9 / 26.5 | 30 / 83 | 0.61 | 2.02 |
| hard, straight drive | calibrated, 32 g / 4000 dps part | end of stance | 5.8 / 12.2 | 23 / 65 | 0.61 | 2.02 |
| adult, pull (flat swing) | datasheet, no per-unit calibration | top of backlift | 27.6 / 62.9 | 37 / 110 | 3.42 | 2.62 |
| adult, pull (flat swing) | datasheet, no per-unit calibration | end of stance | 38.0 / 78.3 | 26 / 80 | 3.42 | 2.62 |
| adult, pull (flat swing) | calibrated per unit | top of backlift | 12.3 / 31.2 | 34 / 101 | 1.95 | 0.81 |
| adult, pull (flat swing) | calibrated per unit | end of stance | 7.8 / 16.8 | 17 / 51 | 1.95 | 0.81 |
| adult, pull (flat swing) | calibrated, 32 g / 4000 dps part | top of backlift | 12.3 / 31.2 | 34 / 101 | 1.93 | 0.81 |
| adult, pull (flat swing) | calibrated, 32 g / 4000 dps part | end of stance | 7.9 / 16.9 | 17 / 51 | 1.93 | 0.81 |

## Sensitivity: adult straight drive, anchored at end of stance

One assumption changed per row from the calibrated case.

| change | lateral p50 / p95 (mm) | face angle p95 (deg) |
|---|---|---|
| baseline: calibrated, heading 1.0 deg | 6.4 / 13.4 | 2.02 |
| heading known to 0.3 deg | 5.2 / 11.1 | 0.71 |
| heading known to 0.5 deg | 5.4 / 11.4 | 1.05 |
| heading known to 2.0 deg | 8.5 / 20.2 | 3.96 |
| poorer camera: 1 px noise, 5 mm bias floor | 8.4 / 17.1 | 2.02 |
| IMU warmed up: gyro scale 0.15%, accel bias 4 mg | 7.5 / 16.4 | 2.02 |
| no per-unit calibration, biases estimated online | 18.0 / 39.7 | 2.65 |
| no per-unit calibration at all | 24.8 / 51.9 | 2.65 |

## Error budget: adult straight drive, calibrated per unit, anchored at end of stance

Each source alone; lateral error at contact.

| source | lateral p50 (mm) | lateral p95 (mm) |
|---|---|---|
| yaw | 3.1 | 9.0 |
| acc_nl | 2.8 | 5.2 |
| acc_bias | 1.6 | 5.0 |
| anchor_pos | 2.4 | 4.9 |
| anchor_vel | 1.5 | 3.2 |
| gyro_cross | 0.9 | 2.4 |
| acc_cross | 0.9 | 2.1 |
| gyro_nl | 1.1 | 2.0 |
| acc_scale | 0.7 | 1.8 |
| gyro_gsens | 0.7 | 1.8 |
| gyro_scale | 0.5 | 1.5 |
| mount | 0.5 | 1.5 |
| gyro_bias | 0.3 | 0.6 |
| gyro_noise | 0.2 | 0.6 |
| acc_noise | 0.2 | 0.4 |
| lever | 0.1 | 0.3 |
| range | 0.3 | 0.3 |
| sync | 0.1 | 0.2 |

## Error budget: adult straight drive, calibrated per unit, anchored at top of backlift

Each source alone; lateral error at contact.

| source | lateral p50 (mm) | lateral p95 (mm) |
|---|---|---|
| yaw | 8.8 | 25.4 |
| anchor_vel | 3.6 | 7.4 |
| acc_bias | 2.2 | 5.5 |
| anchor_pos | 2.5 | 5.1 |
| mount | 1.0 | 2.8 |
| gyro_nl | 1.1 | 2.0 |
| lever | 0.7 | 1.9 |
| gyro_cross | 0.7 | 1.8 |
| acc_cross | 0.6 | 1.7 |
| gyro_gsens | 0.7 | 1.5 |
| gyro_scale | 0.5 | 1.4 |
| gyro_bias | 0.2 | 0.4 |
| acc_nl | 0.1 | 0.3 |
| sync | 0.1 | 0.2 |
| range | 0.2 | 0.2 |
| gyro_noise | 0.1 | 0.2 |
| acc_noise | 0.1 | 0.2 |
| acc_scale | 0.1 | 0.1 |

## Error budget: adult straight drive, datasheet, no per-unit calibration, anchored at end of stance

Each source alone; lateral error at contact.

| source | lateral p50 (mm) | lateral p95 (mm) |
|---|---|---|
| acc_bias | 12.5 | 37.5 |
| gyro_cross | 10.4 | 28.8 |
| acc_cross | 9.2 | 21.3 |
| acc_scale | 3.9 | 10.7 |
| yaw | 3.1 | 9.0 |
| gyro_scale | 3.2 | 8.9 |
| mount | 2.5 | 7.4 |
| acc_nl | 2.8 | 5.2 |
| anchor_pos | 2.4 | 4.9 |
| anchor_vel | 1.5 | 3.2 |
| gyro_nl | 1.1 | 2.0 |
| gyro_gsens | 0.7 | 1.8 |
| gyro_bias | 0.6 | 1.5 |
| lever | 0.5 | 1.3 |
| gyro_noise | 0.2 | 0.6 |
| acc_noise | 0.2 | 0.4 |
| range | 0.3 | 0.3 |
| sync | 0.1 | 0.2 |

## How fast the IMU-only estimate degrades (adult drive, calibrated, anchored at end of stance)

| time since last camera anchor (s) | lateral p50 (mm) | lateral p95 (mm) |
|---|---|---|
| 0.00 | 2.4 | 5.0 |
| 0.05 | 2.4 | 5.1 |
| 0.10 | 2.9 | 6.5 |
| 0.15 | 4.7 | 12.3 |
| 0.20 | 6.8 | 18.7 |
| 0.25 | 8.1 | 22.3 |
| 0.30 | 8.1 | 22.2 |
| 0.35 | 7.6 | 20.2 |
| 0.40 | 7.5 | 19.0 |
| 0.45 | 7.6 | 18.8 |
| 0.50 | 8.0 | 19.9 |
| 0.55 | 9.2 | 23.0 |
| 0.60 | 8.6 | 19.8 |
| 0.65 | 6.4 | 13.4 |

## Global-shutter stereo console, for comparison (analytic)

OV9281-class, f = 914 px, 0.15 px centroid noise, 4 m: lateral 0.7 mm; depth 6.2 mm at 0.6 m baseline, 3.7 mm at 1 m.

