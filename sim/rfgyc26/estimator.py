"""State estimation: where the robot BELIEVES it is.

Same code on both machines -- numpy and params only.  Everything above the
HAL now navigates on this estimate; the simulator's ground truth is for the
referee, the check suites, and this module's own error report.

WHAT THE SENSORS ACTUALLY GIVE, on this robot (and how each is wrong):

  * Commanded-step odometry (DriveHAL.odometry).  Steppers below stall do
    not slip, so straight running is exact to the command -- except that the
    command is not quite the truth: the sim's velocity servo lets the
    chassis ride ~1% past it (measured +1.1% over 400 mm, check_hal), and a
    real drivetrain has its own scale error from tyre diameter.  Both are
    CALIBRATED OUT by ODO_SCALE, the standard bench ritual (UMBmark).  What
    cannot be calibrated: wheels scrub in a pivot (efficiency 0.71 measured
    -- a 90 deg pivot mis-booked by wheels alone would be 26 deg wrong), and
    a stalled stepper counts steps it did not make.
  * The gyro (DriveHAL.gyro_z).  Yaw rate is measured PHYSICALLY, so the
    pivot scrub problem disappears: heading integrates the gyro, never the
    wheels.  The cost is bias -- drawn once per match in the backend, like
    every calibration error in this project -- and integrated bias is the
    heading drift the camera fixes must absorb.
  * Camera slot fixes (fed by robot.see_lab, both vision modes).  The three
    laboratory slots are at KNOWN world positions; a measurement of any of
    them in the robot frame is an absolute position fix, and two at once
    also fix heading (the slot line's pitch is known to a fifth of a
    millimetre -- params.Vision).  These arrive exactly when precision is
    about to be spent: on every dock approach.
  * StallGuard.  While the drivers saturate, commanded steps are fiction:
    translation is FROZEN for the duration.  (This also keeps est.speed()
    honest for stall_drive's is-it-moving test.)

The filter is a 3-state EKF with a diagonal covariance -- position sigma
grows with distance and turn (scrub also translates the axle a little),
shrinks on fixes, and gates them: an innovation far outside the current
uncertainty is refused, not averaged in, same philosophy as perception.
"""
import numpy as np
from .params import Field, Chassis, AgentA

# Straight-line odometry scale: commanded -> true, bench-calibrated
# (UMBmark).  The sim's measured "+1.1% over 400 mm" turned out NOT to be a
# scale error at all -- see COAST_TAU -- so the scale is unity.
ODO_SCALE   = 1.0
# Stopping coast: the sim's velocity servo lets the chassis ride ~v*tau past
# a commanded stop (measured 4.5 mm from 200 mm/s -> tau 22 ms), which a
# step counter cannot see.  It is a PER-STOP error proportional to the speed
# at the stop, so modelling it as a distance scale would corrupt every short
# leg; instead each wheel books v_prev*tau extra when its command collapses
# to zero.  On real TMC2209 steppers holding torque makes tau ~ 0 -- this
# constant is bench-calibrated per machine like ODO_SCALE.
COAST_TAU   = 0.015            # s; the ride-through varies 8-22 ms
                               # with load, so this centres the band
COAST_V     = 30.0             # mm/s; below this a "stop" books no coast
# StallGuard is unreliable in transients (accelerating from rest saturates
# the drivers too), so steps become fiction only after the saturation has
# HELD this many flush intervals -- and the travel booked during the run-up
# was fiction as well, so it is rewound when the freeze engages.
STALL_HOLD  = 12               # flushes (~0.24 s at 50 Hz)
# Process noise, per flush interval: floor + per-mm + per-deg (a scrubbing
# pivot walks the axle centre as well as mis-reading yaw).
Q_FLOOR_MM  = 0.03
Q_PER_MM    = 0.006
Q_PER_DEG   = 0.15
# TRACTION SLIP is the odometry killer StallGuard cannot see: plowing three
# discs through the quarantine, the wheels ran ~30% faster than the chassis
# (measured, +90 mm booked in 3 s) with the drivers nowhere near torque
# saturation.  The DIFFERENTIAL part of slip is observable -- the yaw the
# wheels imply disagrees with the yaw the gyro measured -- so every degree
# of that mismatch buys process noise.  It cannot fix the estimate; it makes
# the estimate ADMIT it is degrading, which is what lets the wall datum and
# the slot fixes gate correctly afterwards.
Q_SLIP_PER_DEG = 1.2
Q_TH_FLOOR  = 0.002            # deg; gyro noise integrates slowly
Q_TH_PER_DEG = 0.004           # gyro scale error over a turn
# The WALL DATUM (the spec's own "left-wall stall").  A sustained stall
# while heading into a known wall pins the along-axis coordinate: the axle
# is one body reach from the wall face.  A stall with a beam carried proud
# of the shell presses ~3 mm early -- inside this sigma -- and a stall
# against something that is NOT the wall (a mid-field jam) is rejected by
# the same covariance gate every other fix uses.
BODY_REACH  = AgentA.L - AgentA.AXLE_X          # 142.5: axle to either end
WALL_SIGMA  = 4.0
# Measurement noise for one slot fix (Vision budget: pipeline well under a
# millimetre, mount bias ~1, mono range a little worse).
R_SLOT_MM   = 2.5
R_TH_DEG    = 0.6              # two-slot pitch-line heading
GATE_MM     = 45.0             # association: nearest hole must be this close
GATE_SIGMA  = 4.0              # ...and the innovation inside 4 sigma + slack


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


class Estimator:
    def __init__(self, x_mm, y_mm, th_deg):
        # The start pose is knowledge, not measurement: the team places the
        # robot in the deployment box by hand, against its corner jig.
        self.x, self.y, self.th = float(x_mm), float(y_mm), float(th_deg)
        # THE ODOMETRY FRAME: the same increments, never fixed.  Fixes JUMP
        # the map-frame belief -- that is their job -- and a terminal that
        # froze a target in a frame that then jumps drives to a point that
        # no longer means anything (measured: the dock fell from 67% to 20%
        # the day navigation went honest).  Terminals therefore track in
        # THIS frame: smooth, locally exact, drifting only mm over the
        # seconds a terminal lasts.  The standard map/odom split.
        self.xo, self.yo, self.tho = self.x, self.y, self.th
        self.var_xy = 4.0                      # mm^2 -- a couple of mm of jig
        self.var_th = 0.25                     # deg^2
        self._speed = 0.0
        self._vl = self._vr = 0.0              # last interval's wheel speeds
        self._stall_run = 0                    # consecutive saturated flushes
        self._stall_hist = []                  # (dx, dy) booked during run-up
        self._stall_dir = 1.0                  # sign of travel entering a stall
        self.fixes = 0
        self.rejected = 0

    # ------------------------------------------------------------- predict
    def predict(self, dl_mm, dr_mm, dth_gyro_deg, dt_s, stall_raw=False):
        """One flush interval: wheel travel, gyro yaw increment, and the RAW
        driver-saturation flag (StallGuard / actuator force).  The estimator
        owns the judgement: sustained saturation means the steps are being
        counted, not made -- freeze translation and rewind the run-up."""
        if stall_raw:
            self._stall_run += 1
            if self._stall_run == STALL_HOLD:
                # The freeze engages: everything booked while saturated was
                # fiction.  Take it back out, admit the uncertainty a wall
                # contact leaves -- then ask whether this stall IS the wall
                # datum, which turns the failure into a fix.
                for hx, hy, hxo, hyo in self._stall_hist:
                    self.x -= hx; self.y -= hy
                    self.xo -= hxo; self.yo -= hyo
                self._stall_hist = []
                self.var_xy += 25.0
                self._wall_datum()
        else:
            self._stall_run = 0
            self._stall_hist = []
        if self._stall_run >= STALL_HOLD:
            dl_mm = dr_mm = 0.0
        if dt_s > 0:
            vl, vr = dl_mm/dt_s, dr_mm/dt_s
            # book the stopping coast the step counter cannot see
            if abs(self._vl) > COAST_V and abs(vl) < 1.0:
                dl_mm += self._vl * COAST_TAU
            if abs(self._vr) > COAST_V and abs(vr) < 1.0:
                dr_mm += self._vr * COAST_TAU
            self._vl, self._vr = vl, vr
        ds = ODO_SCALE * 0.5*(dl_mm + dr_mm)
        dth = dth_gyro_deg                     # heading is the GYRO's
        t = np.radians(self.th + 0.5*dth)
        dx, dy = ds*np.cos(t), ds*np.sin(t)
        self.x += dx
        self.y += dy
        to = np.radians(self.tho + 0.5*dth)
        dxo, dyo = ds*np.cos(to), ds*np.sin(to)
        self.xo += dxo
        self.yo += dyo
        self.tho = _wrap(self.tho + dth)
        if stall_raw and self._stall_run < STALL_HOLD:
            # provisional in BOTH frames: rewound on engage
            self._stall_hist.append((dx, dy, dxo, dyo))
        if abs(ds) > 1e-6:
            self._stall_dir = 1.0 if ds > 0 else -1.0
        self.th = _wrap(self.th + dth)
        self._speed = abs(ds)/dt_s if dt_s > 0 else 0.0
        slip = abs(np.degrees((dr_mm - dl_mm)/Chassis.TRACK) - dth)
        q = Q_FLOOR_MM + Q_PER_MM*abs(ds) + Q_PER_DEG*abs(dth) \
            + Q_SLIP_PER_DEG*slip
        self.var_xy += q*q
        qt = Q_TH_FLOOR + Q_TH_PER_DEG*abs(dth)
        self.var_th += qt*qt

    def _wall_datum(self):
        """A sustained stall, read against the map: if the travel direction
        points into a field wall, the axle is one body reach from its face.
        Applied through wall_fix, so the covariance gate -- not a special
        case -- decides whether this stall can be the wall at all."""
        a = self.th if self._stall_dir >= 0 else self.th + 180.0
        for ang, axis, wall in ((0.0, "x", Field.W), (180.0, "x", 0.0),
                                (90.0, "y", Field.H), (270.0, "y", 0.0)):
            if abs(_wrap(a - ang)) < 25.0:
                self.wall_fix(axis, (wall - BODY_REACH) if wall > 0.0
                              else BODY_REACH, sigma_mm=WALL_SIGMA,
                              gate_mm=120.0)
                return

    def speed(self):
        """Believed speed, mm/s.  Zero while stalled -- see the header."""
        return self._speed

    # ---------------------------------------------------------------- fixes
    def slot_fix(self, meas_robot, holes_world):
        """Absolute update from lab-slot measurements.

        meas_robot: [(x_r, y_r, ...), ...] robot-frame, from see_lab.
        holes_world: [(wx, wy), ...] the KNOWN slot centres.
        Association is nearest-hole under the current estimate; a fix whose
        innovation exceeds the gate is rejected and counted, never blended.
        """
        if not meas_robot:
            return
        t = np.radians(self.th)
        c, s = np.cos(t), np.sin(t)
        pairs = []                             # (world_meas, world_hole)
        for mr in meas_robot:
            xr, yr = mr[0], mr[1]
            wx = self.x + xr*c - yr*s
            wy = self.y + xr*s + yr*c
            d, hole = min(((np.hypot(wx-hx, wy-hy), (hx, hy))
                           for hx, hy in holes_world), key=lambda v: v[0])
            if d < GATE_MM:
                pairs.append(((wx, wy), hole))
            else:
                self.rejected += 1
        if not pairs:
            return
        # Heading first, from the pitch between two associated slots: the
        # difference of two measurements, so the mount bias drops out of it.
        if len(pairs) >= 2:
            (m0, h0), (m1, h1) = pairs[0], pairs[-1]
            if np.hypot(h1[0]-h0[0], h1[1]-h0[1]) > 60.0:
                a_m = np.degrees(np.arctan2(m1[1]-m0[1], m1[0]-m0[0]))
                a_h = np.degrees(np.arctan2(h1[1]-h0[1], h1[0]-h0[0]))
                innov = _wrap(a_h - a_m)
                sig = np.sqrt(self.var_th + R_TH_DEG**2)
                if abs(innov) < GATE_SIGMA*sig + 2.0:
                    k = self.var_th / (self.var_th + R_TH_DEG**2)
                    self.th = _wrap(self.th + k*innov)
                    self.var_th *= (1.0 - k)
                else:
                    self.rejected += 1
        # Then position, from the mean innovation of every associated slot.
        inx = float(np.mean([h[0]-m[0] for m, h in pairs]))
        iny = float(np.mean([h[1]-m[1] for m, h in pairs]))
        r2 = R_SLOT_MM**2 / len(pairs)
        sig = np.sqrt(self.var_xy + r2)
        if np.hypot(inx, iny) > GATE_SIGMA*sig + 10.0:
            self.rejected += 1
            return
        k = self.var_xy / (self.var_xy + r2)
        self.x += k*inx
        self.y += k*iny
        self.var_xy *= (1.0 - k)
        self.fixes += 1

    def wall_fix(self, axis, value_mm, sigma_mm=5.0, gate_mm=None):
        """1-D absolute fix from a wall stall: the spec's own datum.
        axis 'x' or 'y'; value is where that coordinate physically is.

        gate_mm overrides the covariance gate, because a wall press earns a
        PHYSICAL gate the filter cannot derive: common-mode wheel slip grows
        real error the covariance never saw (both wheels slipping equally is
        unobservable -- measured, 90 mm booked with sigma claiming 16), and
        the map guarantees the space near a wall is EMPTY: nothing else
        within ~120 mm of a wall face along a cardinal approach can hold a
        sustained stall.  A placed beam's face sits 155+ mm out -- outside
        the gate, correctly rejected."""
        cur = self.x if axis == "x" else self.y
        r2 = sigma_mm*sigma_mm
        innov = value_mm - cur
        gate = gate_mm if gate_mm is not None \
            else GATE_SIGMA*np.sqrt(self.var_xy + r2) + 10.0
        if abs(innov) > gate:
            self.rejected += 1
            return
        k = self.var_xy / (self.var_xy + r2)
        if axis == "x":
            self.x += k*innov
        else:
            self.y += k*innov
        self.var_xy *= (1.0 - 0.5*k)           # only one axis got information
        self.fixes += 1

    # ---------------------------------------------------------------- state
    @property
    def pose(self):
        return self.x, self.y, self.th

    @property
    def odo_pose(self):
        """The jump-free frame terminals track in (see __init__)."""
        return self.xo, self.yo, self.tho

    def sigma(self):
        """(mm, deg) 1-sigma -- behaviours may demand a fix below a gate."""
        return float(np.sqrt(self.var_xy)), float(np.sqrt(self.var_th))
