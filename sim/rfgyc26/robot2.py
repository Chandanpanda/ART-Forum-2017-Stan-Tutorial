"""Robot 2: the detached actuator (design doc section 10).

Three layers, split exactly where the hardware splits:

  * SimLink -- the PICO W SIDE.  Decodes the LinkHAL v0 wire (the same
    grammar hal.LinkHAL encodes) into wheel commands, applies the DC
    plant's indignities where the real driver sits -- per-motor gain
    lottery, deadband, integer quantisation -- enforces the 250 ms
    dead-man, and owns the SHAKE macro as firmware.  Swap this class for
    a Bluetooth socket (BtLink, section 4.4) and NOTHING above changes.

  * R2Controller -- the PI SIDE.  Robot 2 has no sensors, so this runs on
    robot 1: a `spot` callable (robot 1's stereo rig finding the ArUco)
    is the only feedback, ~5 Hz with centimetre noise.  Between fixes the
    controller dead-reckons on its own commands scaled by CALIBRATION
    it learns as it drives: the per-side gain estimate is updated from
    every camera-observed displacement, which is the "learn to control
    the cheap motors" loop -- on real hardware the same code calibrates
    the real motors.

  * primitives -- face / goto / push_to / shake_out, all generators in
    the route.py convention (one yield per 50 Hz tick), commanding at
    20 Hz with 150 ms of dead-man margin.  Tolerances are camera-sized:
    nothing here pretends to millimetres, because every job is a plow
    push whose tolerance is the plow's width.
"""
import numpy as np
from . import hal
from .params import Robot2 as R2

TICK = hal.Clock.PERIOD                      # 50 Hz mission tick
_CMD_EVERY = max(1, int(round(hal.Clock.HZ / R2.CMD_HZ)))   # ticks per send


# ============================================================== the Pico side
class SimLink(hal.LinkHAL):
    """The firmware end of the wire, on MuJoCo.

    Construct with the model/data and a per-match rng: the gain lottery is
    drawn HERE, because that is where the real spread lives (motor + driver
    + tyre tolerances).  The mission driver calls step() once per control
    tick, before physics."""

    def __init__(self, m, d, rng=None):
        self.m, self.d = m, d
        rng = rng or np.random.default_rng(0)
        self._al = _aid(m, "r2_drive_l")
        self._ar = _aid(m, "r2_drive_r")
        self.gain_l = float(1.0 + R2.GAIN_SD * rng.standard_normal())
        self.gain_r = float(1.0 + R2.GAIN_SD * rng.standard_normal())
        self._vl = self._vr = 0.0
        self._expire = -1.0
        self._shake = 0.0                    # busy-until clock for the macro
        self._shake_t0 = 0.0
        self.sent = 0

    # --- wire ---
    def send(self, line):
        self.sent += 1
        f = line.split()
        now = self.d.time
        if f[0] == "V" and len(f) == 4:
            if now >= self._shake:           # firmware ignores V mid-shake
                self._vl, self._vr = float(int(f[1])), float(int(f[2]))
                self._expire = now + min(int(f[3]), 1000) / 1000.0
        elif f[0] == "K":
            self._vl = self._vr = 0.0
            self._expire = now
        elif f[0] == "SHAKE" and len(f) == 2:
            n = max(1, min(int(f[1]), 10))
            self._shake_t0 = now
            self._shake = now + n * 0.55     # n ratchet cycles
            self._vl = self._vr = 0.0

    def recv(self):
        return None

    # --- plant ---
    def _wheel(self, v, gain):
        # driver deadband, then the gain lottery; the wire already quantised
        if abs(v) < R2.DEADBAND:
            return 0.0
        return v * gain

    def step(self):
        """Apply the current command state to the actuators.  One call per
        control tick; the dead-man and the shake macro live here."""
        now = self.d.time
        if now < self._shake:
            # THE RATCHET (rigged, F93): collect against the front rail at
            # full reverse, then a full-jerk forward pulse -- the kit
            # sprints the tray and hops the 1.2 mm tail lip.  0.55 s/cycle.
            ph = (now - self._shake_t0) % 0.55
            v = -380.0 if ph < 0.25 else (380.0 if ph < 0.45 else 0.0)
            vl = vr = v
        elif now > self._expire:
            vl = vr = 0.0                    # dead-man: silence stops it
        else:
            vl, vr = self._vl, self._vr
        self.d.ctrl[self._al] = self._wheel(vl, self.gain_l) / R2.WHEEL_R
        self.d.ctrl[self._ar] = self._wheel(vr, self.gain_r) / R2.WHEEL_R

    def busy(self):
        return self.d.time < self._shake


def _aid(m, name):
    import mujoco
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if i < 0:
        raise KeyError(name)
    return i


# ============================================================ robot 1's eyes
def sim_spot(m, d, rng, sigma_mm=8.0, sigma_deg=2.5):
    """Model-camera tracking of robot 2's ArUco, the same convention as
    robot 1's synthetic `see_lab`: truth plus calibrated noise.  The render
    pipeline replaces this on the Pi; the CONTROLLER cannot tell."""
    import mujoco
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot2")

    def spot():
        x, y = d.xpos[b][:2] * 1000.0
        q = d.xquat[b]
        th = np.degrees(np.arctan2(2*(q[0]*q[3] + q[1]*q[2]),
                                   1 - 2*(q[2]**2 + q[3]**2)))
        return (x + rng.normal(0.0, sigma_mm),
                y + rng.normal(0.0, sigma_mm),
                th + rng.normal(0.0, sigma_deg))
    return spot


# =============================================================== the Pi side
def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


class R2Controller:
    """Closed loop over the camera, open loop between fixes.

    The belief is last-fix + command dead-reckoning through the learned
    gains; every new fix both corrects the belief and refines the gains
    (scale from distance covered, turn bias from heading drift on nominally
    straight legs).  Fix cadence ~5 Hz: every job here tolerates that."""

    FIX_EVERY = 10                            # ticks between camera looks

    def __init__(self, link, spot, clock):
        self.link, self.spot, self.clock = link, spot, clock
        x, y, th = spot()
        self.x, self.y, self.th = x, y, th
        self.scale = 1.0                      # learned: commanded -> real
        self.turn_bias = 0.0                  # deg/s of veer at straight cmd
        self._tick = 0
        self._cmd = (0.0, 0.0)
        self._dr = np.zeros(3)                # dead-reckoned delta since fix
        self._fix_xy = (x, y, th)

    # ---- belief upkeep -------------------------------------------------
    def _integrate(self):
        vl, vr = self._cmd
        v = 0.5 * (vl + vr) * self.scale
        w = np.degrees((vr - vl) * self.scale / R2.TRACK) + \
            (self.turn_bias if abs(vl) > 50 and abs(vr) > 50 else 0.0)
        self.th += w * TICK
        self.x += v * np.cos(np.radians(self.th)) * TICK
        self.y += v * np.sin(np.radians(self.th)) * TICK
        self._dr += (v * TICK, 0.0, w * TICK)

    def _maybe_fix(self):
        self._tick += 1
        if self._tick % self.FIX_EVERY:
            return
        x, y, th = self.spot()
        # calibration: compare camera displacement with the dead-reckoned
        # one over the window; only meaningful when the robot really moved
        cam_d = float(np.hypot(x - self._fix_xy[0], y - self._fix_xy[1]))
        dr_d = float(self._dr[0])
        if abs(dr_d) > 25.0 and cam_d > 15.0:
            r = float(np.clip(cam_d / abs(dr_d), 0.6, 1.6))
            self.scale = float(np.clip(0.9*self.scale + 0.1*self.scale*r,
                                       0.6, 1.5))
        self.x, self.y, self.th = x, y, th
        self._fix_xy = (x, y, th)
        self._dr[:] = 0.0

    # ---- wire ----------------------------------------------------------
    def _drive(self, vl, vr):
        vl = float(np.clip(vl, -R2.V_MAX, R2.V_MAX))
        vr = float(np.clip(vr, -R2.V_MAX, R2.V_MAX))
        self._cmd = (vl, vr)
        if self._tick % _CMD_EVERY == 0:
            self.link.cmd(vl, vr, 150)

    def stop(self):
        self._cmd = (0.0, 0.0)
        self.link.halt()

    def tick(self):
        """One 50 Hz step of bookkeeping; call from inside primitives."""
        self._integrate()
        self._maybe_fix()

    @property
    def pose(self):
        return self.x, self.y, self.th

    # ---- primitives ----------------------------------------------------
    def face(self, th_t, tol=6.0, cap_s=4.0):
        n = int(cap_s * hal.Clock.HZ)
        for _ in range(n):
            err = _wrap(th_t - self.th)
            if abs(err) < tol:
                break
            w = np.clip(3.2 * err, -R2.W_MAX, R2.W_MAX)
            dv = np.radians(w) * R2.TRACK / 2.0
            # keep the wheels outside the deadband or nothing turns
            if abs(dv) < R2.DEADBAND + 12.0:
                dv = np.sign(dv) * (R2.DEADBAND + 12.0)
            self._drive(-dv, dv)
            self.tick()
            yield
        self.stop()
        for _ in range(4):
            self.tick(); yield

    def goto(self, tx, ty, v_max=320.0, tol=25.0, cap_s=12.0, slow_into=None):
        """Drive to a point: initial face if far off-bearing, then a heading-P
        pursuit with a speed ramp.  slow_into caps speed near the goal."""
        b0 = np.degrees(np.arctan2(ty - self.y, tx - self.x))
        if abs(_wrap(b0 - self.th)) > 35.0:
            yield from self.face(b0, tol=10.0)
        n = int(cap_s * hal.Clock.HZ)
        for _ in range(n):
            dx, dy = tx - self.x, ty - self.y
            dist = float(np.hypot(dx, dy))
            if dist < tol:
                break
            err = _wrap(np.degrees(np.arctan2(dy, dx)) - self.th)
            if abs(err) > 55.0:
                yield from self.face(np.degrees(np.arctan2(dy, dx)), tol=8.0)
                continue
            v = min(v_max, max(90.0, 2.2 * dist))
            if slow_into is not None:
                v = min(v, slow_into)
            v *= max(0.25, 1.0 - abs(err) / 80.0)
            w = np.clip(2.6 * err, -140.0, 140.0)          # deg/s of yaw
            dv = np.radians(w) * R2.TRACK / 2.0
            self._drive(v - dv, v + dv)
            self.tick()
            yield
        self.stop()
        for _ in range(4):
            self.tick(); yield

    def push_to(self, tx, ty, v=150.0, tol=30.0, cap_s=16.0):
        """A plow push: same pursuit, gentler -- the puck must stay in the
        pocket, so turns are arc-limited and speed is constant-low."""
        n = int(cap_s * hal.Clock.HZ)
        for _ in range(n):
            dx, dy = tx - self.x, ty - self.y
            dist = float(np.hypot(dx, dy))
            if dist < tol:
                break
            err = _wrap(np.degrees(np.arctan2(dy, dx)) - self.th)
            w = np.clip(1.6 * err, -55.0, 55.0)            # gentle: keep the
            dv = np.radians(w) * R2.TRACK / 2.0            # puck in the pocket
            self._drive(v - dv, v + dv)
            self.tick()
            yield
        self.stop()
        for _ in range(4):
            self.tick(); yield

    def back_off(self, mm_, v=200.0):
        n = int(mm_ / v * hal.Clock.HZ)
        for _ in range(n):
            self._drive(-v, -v)
            self.tick()
            yield
        self.stop()
        for _ in range(4):
            self.tick(); yield

    def shake_out(self, n=4):
        self.link.shake(n)
        for _ in range(int(n * 0.55 * hal.Clock.HZ) + 10):
            self.tick()
            yield
        self.stop()
