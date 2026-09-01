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
from . import hal, nav
from .params import Robot2 as R2

R2_INSCRIBED = 55.0        # no orientation fits inside this
R2_CIRCUM = 93.0           # every orientation fits outside it

TICK = hal.Clock.PERIOD                      # 50 Hz mission tick
_CMD_EVERY = max(1, int(round(hal.Clock.HZ / R2.CMD_HZ)))   # ticks per send


# ============================================================== the Pico side
class WheelServo:
    """The inner loop -- the one the first build did not have (F99).

    A cheap DC gearmotor does not go the speed you ask it to.  Its torque
    constant is a lottery, its driver has stiction, its battery sags, and the
    plow's load changes every second.  The first build told the Pico
    "200 mm/s", set a PWM proportional to that, and compensated at the Pi with
    a SINGLE LEARNED SCALAR -- which can represent none of those, because they
    are per-wheel, nonlinear and time-varying.

    So: a PI controller per wheel, on measured encoder velocity, at 1 kHz,
    inside the firmware.  Then `V 200 200 150` on the wire MEANS 200 mm/s, and
    every layer above stops paying for the lie.  This class is the firmware;
    it is deliberately written in the terms a Pico has (ticks, dt, a torque
    command) so it ports across as arithmetic, not as an idea."""

    def __init__(self, gain=1.0, rng=None):
        self.gain = float(gain)          # the motor's own torque constant
        self.i = 0.0                     # integrator, N.m
        self.ticks = 0                   # encoder accumulator
        self._frac = 0.0
        self.target = 0.0                # mm/s

    def update(self, v_meas, dt):
        """One firmware tick: measured wheel velocity in, torque out."""
        err = self.target - v_meas
        self.i = float(np.clip(self.i + R2.SERVO_KI * err * dt,
                               -R2.TAU_STALL, R2.TAU_STALL))
        if abs(self.target) < 1e-6:
            self.i *= 0.9                # bleed down when commanded to stop
        tau = R2.SERVO_KP * err + self.i
        # the plant's own limit: an N20 cannot exceed its stall torque, and
        # what it CAN deliver falls off with speed (the torque-speed line)
        w = abs(v_meas) / R2.WHEEL_R
        avail = R2.TAU_STALL * max(0.0, 1.0 - w / R2.W_NOLOAD)
        tau = float(np.clip(tau, -avail, avail))
        # stiction: below breakaway the shaft simply does not move
        if abs(tau) < R2.STICTION and abs(v_meas) < 5.0:
            tau = 0.0
        return tau * self.gain

    def count(self, v_meas, dt):
        """Quadrature accumulation, quantised exactly as the hardware is."""
        rev = v_meas * dt / (2.0 * np.pi * R2.WHEEL_R)
        self._frac += rev * R2.ENC_CPR
        whole = int(self._frac)
        self._frac -= whole
        self.ticks += whole
        return self.ticks


class SimLink(hal.LinkHAL):
    """The firmware end of the wire, on MuJoCo.

    Owns: the v0 grammar decode, the 250 ms dead-man, the SHAKE macro, the two
    WheelServos, and the `O <l> <r>` odometry report the Pi consumes.  The
    per-motor gain lottery is drawn HERE, because that is where the real
    spread lives -- and now the servo has to REJECT it rather than the Pi
    having to model it."""

    def __init__(self, m, d, rng=None):
        self.m, self.d = m, d
        rng = rng or np.random.default_rng(0)
        self._al = _aid(m, "r2_drive_l")
        self._ar = _aid(m, "r2_drive_r")
        import mujoco
        self._jl = m.jnt_dofadr[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, "r2_w_l")]
        self._jr = m.jnt_dofadr[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, "r2_w_r")]
        self.gain_l = float(1.0 + R2.GAIN_SD * rng.standard_normal())
        self.gain_r = float(1.0 + R2.GAIN_SD * rng.standard_normal())
        self.servo_l = WheelServo(self.gain_l)
        self.servo_r = WheelServo(self.gain_r)
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
        """`O <ticks_l> <ticks_r>` -- the return direction the first build
        left as None.  This is what turns the Pi's dead reckoning from
        commanded-velocity-times-a-scalar into actual odometry."""
        return "O %d %d" % (self.servo_l.ticks, self.servo_r.ticks)

    def wheel_speeds(self):
        """Measured, in mm/s -- what the encoders see."""
        return (float(self.d.qvel[self._jl]) * R2.WHEEL_R,
                float(self.d.qvel[self._jr]) * R2.WHEEL_R)

    # --- plant ---
    def step(self, n=20):
        """One 50 Hz control tick: the COMMAND layer once, then n physics
        substeps with the inner servo loop closed around each one.

        THE LINK OWNS THE SUBSTEPPING, which is not an accident of the sim: the
        inner loop only means anything if it reads a velocity that has changed
        since it last wrote a torque.  The first attempt ran twenty servo
        iterations against one frozen qvel and simply wound its integrator up.
        Drivers call this INSTEAD of their own mj_step loop."""
        import mujoco
        now = self.d.time
        if now < self._shake:
            # THE RATCHET (F93): collect against the front rail at full
            # reverse, then a full-jerk forward pulse -- the kit sprints the
            # tray and hops the 1.2 mm tail lip.  0.55 s per cycle.
            ph = (now - self._shake_t0) % 0.55
            v = -R2.V_MAX if ph < 0.25 else (R2.V_MAX if ph < 0.45 else 0.0)
            vl = vr = v
        elif now > self._expire:
            vl = vr = 0.0                    # dead-man: silence stops it
        else:
            vl, vr = self._vl, self._vr
        self.servo_l.target = vl
        self.servo_r.target = vr
        dt = float(self.m.opt.timestep)
        for _ in range(max(1, n)):
            ml, mr = self.wheel_speeds()
            self.d.ctrl[self._al] = self.servo_l.update(ml, dt)
            self.d.ctrl[self._ar] = self.servo_r.update(mr, dt)
            self.servo_l.count(ml, dt)
            self.servo_r.count(mr, dt)
            mujoco.mj_step(self.m, self.d)

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
    """The Pi side: belief, path following, and the push primitive.

    THREE LOOPS, and this class owns the outer two (the inner one is
    WheelServo, on the Pico).  The belief is encoder odometry corrected by
    camera fixes; the tracker is a Dynamic Window Approach over the costmap,
    so refusing to drive into things is a property of the controller rather
    than something the mission has to route around by hand.
    """

    FIX_EVERY = 10                            # ticks between camera looks

    def __init__(self, link, spot, clock, cmap=None):
        self.link, self.spot, self.clock = link, spot, clock
        self.cmap = cmap
        # THE BELIEF IS BORN LAZILY, on first use -- not here (F99).  A pose
        # read at construction time comes from an un-forwarded MjData and is
        # all zeros, so every plan started from the south-west corner and the
        # robot spent the match chasing a path it was never on.  Robot 1's
        # estimator learned this exact lesson in step 3 (F85); the fix is the
        # same one.
        self._born = False
        self.x = self.y = self.th = 0.0
        self._tick = 0
        self._cmd = (0.0, 0.0)
        self._ticks = self._read_ticks()
        self.jams = 0
        self.blocked = False                  # last follow_path gave up

    # ---- belief: encoder odometry, camera-corrected ---------------------
    def _read_ticks(self):
        line = self.link.recv()
        if not line:
            return (0, 0)
        f = line.split()
        return (int(f[1]), int(f[2])) if f[0] == "O" else (0, 0)

    def _integrate(self):
        """Differential-drive dead reckoning on ENCODER TICKS -- not on the
        commanded velocity times a learned scalar, which is what the first
        build had and which could not represent the deadband, the per-wheel
        gain spread or the battery's sag (F99)."""
        tl, tr = self._read_ticks()
        dl = (tl - self._ticks[0]) / R2.ENC_CPR * 2.0 * np.pi * R2.WHEEL_R
        dr = (tr - self._ticks[1]) / R2.ENC_CPR * 2.0 * np.pi * R2.WHEEL_R
        self._ticks = (tl, tr)
        ds = 0.5 * (dl + dr)
        dth = np.degrees((dr - dl) / R2.TRACK)
        self.th += dth
        self.x += ds * np.cos(np.radians(self.th - 0.5 * dth))
        self.y += ds * np.sin(np.radians(self.th - 0.5 * dth))

    def _maybe_fix(self):
        self._tick += 1
        if self._tick % self.FIX_EVERY:
            return
        x, y, th = self.spot()
        self.x, self.y, self.th = x, y, th

    def _birth(self):
        if not self._born:
            self.x, self.y, self.th = self.spot()
            self._ticks = self._read_ticks()
            self._born = True

    def tick(self):
        self._birth()
        self._integrate()
        self._maybe_fix()

    @property
    def pose(self):
        self._birth()
        return self.x, self.y, self.th

    # ---- wire ------------------------------------------------------------
    def _drive(self, v, w):
        """(v mm/s, w deg/s) -> wheel speeds -> the wire, at 20 Hz."""
        dv = np.radians(w) * R2.TRACK / 2.0
        vl = float(np.clip(v - dv, -R2.V_MAX, R2.V_MAX))
        vr = float(np.clip(v + dv, -R2.V_MAX, R2.V_MAX))
        self._cmd = (vl, vr)
        if self._tick % _CMD_EVERY == 0:
            self.link.cmd(vl, vr, 150)

    def stop(self):
        self._cmd = (0.0, 0.0)
        self.link.halt()

    # ---- DWA -------------------------------------------------------------
    # Acceleration limits are PHYSICAL now: 2*TAU_STALL/WHEEL_R over the mass.
    A_LIN = 2000.0                            # mm/s^2 usable (of ~8900 avail)
    A_ANG = 900.0                             # deg/s^2
    HORIZON = 0.7                             # s of rollout
    N_V, N_W = 7, 15

    # The body's corners in the chassis frame: what the footprint check
    # actually tests.  A DISC MODEL CANNOT REPRESENT THIS ROBOT (F99) --
    # inflating by the inscribed radius (55, the half-WIDTH) leaves the nose
    # 75 mm out free to enter an obstacle, and it did: the tracker drove the
    # chassis onto the laboratory plate and pressed there, wheels spinning,
    # while its disc model reported clear.  Inflating by the circumscribed
    # radius instead would be safe but would close both 191 mm pinches this
    # robot was narrowed to fit.  So: plan on the disc, VETO ON THE BODY.
    _CORNERS = [(78.0, 55.0), (78.0, -55.0), (-78.0, 55.0), (-78.0, -55.0),
                (78.0, 0.0), (0.0, 0.0)]

    def _hits(self, x, y, a, body):
        ca, sa = np.cos(a), np.sin(a)
        for lx, ly in self._CORNERS:
            wx = x + lx * ca - ly * sa
            wy = y + lx * sa + ly * ca
            i = int(np.clip(wx // nav.RES, 0, body.shape[0] - 1))
            j = int(np.clip(wy // nav.RES, 0, body.shape[1] - 1))
            if body[i, j] >= nav.BLOCKED:
                return True
        return False

    def _dwa(self, path, i_ahead, v_max, w_max, grid, body=None):
        """Sample the admissible window, roll each candidate forward, keep the
        best.  Score = progress toward the carrot + clearance + speed, minus
        cross-track error.  A candidate whose rollout touches a blocked cell
        is discarded outright, which is what makes 'stuck' structurally
        impossible rather than something a watchdog notices afterwards."""
        px, py, th = self.x, self.y, np.radians(self.th)
        v0 = 0.5 * (self._cmd[0] + self._cmd[1])
        w0 = np.degrees((self._cmd[1] - self._cmd[0]) / R2.TRACK)
        dt = 1.0 / hal.Clock.HZ
        vs = np.linspace(max(-v_max, v0 - self.A_LIN * dt * 4),
                         min(v_max, v0 + self.A_LIN * dt * 4), self.N_V)
        ws = np.linspace(max(-w_max, w0 - self.A_ANG * dt * 4),
                         min(w_max, w0 + self.A_ANG * dt * 4), self.N_W)
        cand = [(v, w) for v in vs for w in ws]
        # ESCAPES, always available whatever the accel window says.  A
        # differential drive can ALWAYS spin on the spot and can always back
        # up, and those two are exactly the moves that rescue a chassis whose
        # forward window has gone empty.  Without them the tracker declared
        # itself stuck four times a leg (measured) in situations a driver
        # would have turned out of in half a second.
        for w in (-w_max, -0.6 * w_max, 0.6 * w_max, w_max):
            cand.append((0.0, w))
            cand.append((-140.0, 0.35 * w))
        cand.append((-170.0, 0.0))
        cx, cy = path[i_ahead]
        best, best_s = (0.0, 0.0), -1e18
        steps = 6
        h = self.HORIZON / steps
        for v, w in cand:
            if True:
                x, y, a = px, py, th
                imminent, pen = False, 0.0
                for k in range(steps):
                    a += np.radians(w) * h
                    x += v * np.cos(a) * h
                    y += v * np.sin(a) * h
                    if body is not None:
                        if self._hits(x, y, a, body):
                            # HARD-REJECT ONLY WHAT IS IMMINENT.  Rejecting
                            # any rollout that grazes an inflated halo empties
                            # the window everywhere near a wall -- measured,
                            # five of six legs jammed within seconds.  The
                            # A* path is already clear; DWA's job is to track
                            # it and to not hit what the path did not know
                            # about.  So: veto the next 2 steps, and merely
                            # dislike the rest.
                            if k < 2:
                                imminent = True
                                break
                            pen += 600.0 / (k + 1)
                            break
                        # NO soft clearance term.  A* already routed for
                        # clearance; charging it again here made the tracker
                        # pay ~100 points to move at all inside the inflation
                        # band -- which is most of the field -- against a
                        # speed reward of 20, so it crawled the whole match
                        # at 80 mm/s against a 340 limit (measured).  DWA's
                        # job is tracking and imminent-collision veto; the
                        # clearance judgement belongs to the planner.
                if imminent:
                    continue
                d_end = np.hypot(cx - x, cy - y)
                head = abs(_wrap(np.degrees(np.arctan2(cy - y, cx - x))
                                 - np.degrees(a)))
                # forward progress is worth more than spinning: the escapes
                # must rescue the tracker, not become its favourite move
                s = (-2.0 * d_end - 1.2 * head + 0.25 * max(v, 0.0)
                     - 0.20 * abs(min(v, 0.0)) - pen)
                if s > best_s:
                    best_s, best = s, (v, w)
        return best

    # ---- primitives ------------------------------------------------------
    def face(self, th_t, tol=6.0, cap_s=4.0):
        n = int(cap_s * hal.Clock.HZ)
        for _ in range(n):
            err = _wrap(th_t - self.th)
            if abs(err) < tol:
                break
            w = float(np.clip(3.2 * err, -R2.W_MAX, R2.W_MAX))
            if abs(w) < 40.0:
                w = 40.0 * np.sign(w)
            self._drive(0.0, w)
            self.tick()
            yield
        self.stop()
        for _ in range(3):
            self.tick(); yield

    def follow_path(self, path, v_max=340.0, w_max=200.0, tol=30.0,
                    cap_s=None, grid=None):
        """Track a planned path with DWA.  Returns True on arrival."""
        if not path:
            return True
        self._birth()
        body = None
        if grid is None and self.cmap is not None:
            grid = self.cmap.inflated(R2_INSCRIBED, R2_CIRCUM)
            body = self.cmap.inflated(6.0, 8.0)
        if cap_s is None:
            cap_s = 4.0 + nav.path_length(path) / 120.0
        n = int(cap_s * hal.Clock.HZ)
        gx, gy = path[-1]
        jx, jy, jn = self.x, self.y, 0
        for _ in range(n):
            if np.hypot(gx - self.x, gy - self.y) < tol:
                self.stop()
                for _ in range(3):
                    self.tick(); yield
                return True
            i = self._carrot(path, 170.0)
            v, w = self._dwa(path, i, v_max, w_max, grid, body)
            if abs(v) < 8.0 and abs(w) < 8.0:
                # the window is empty -- every rollout hits something.  Back
                # out, mark the spot, and let the caller replan.
                self.jams += 1
                if self.cmap is not None:
                    self.cmap.add_sticky(self.x, self.y)
                yield from self.back_off(90.0)
                self.blocked = True
                return False
            self._drive(v, w)
            self.tick()
            yield
            jn += 1
            if jn >= 50:
                if np.hypot(self.x - jx, self.y - jy) < 12.0:
                    self.jams += 1
                    if self.cmap is not None:
                        self.cmap.add_sticky(self.x, self.y)
                    yield from self.back_off(90.0)
                    self.blocked = True
                    return False
                jx, jy, jn = self.x, self.y, 0
        self.stop()
        return np.hypot(gx - self.x, gy - self.y) < tol * 2.0

    def _carrot(self, path, look):
        """Index of the point `look` mm ahead along the path."""
        best, bd = 0, 1e18
        for k, (x, y) in enumerate(path):
            d = np.hypot(x - self.x, y - self.y)
            if d < bd:
                bd, best = d, k
        acc = 0.0
        for k in range(best, len(path) - 1):
            acc += np.hypot(path[k+1][0] - path[k][0],
                            path[k+1][1] - path[k][1])
            if acc >= look:
                return k + 1
        return len(path) - 1

    def goto(self, tx, ty, v_max=340.0, tol=30.0, cap_s=None, t_now=None,
             tries=4):
        """Plan a path to (tx, ty), follow it, and REPLAN when the tracker
        runs out of admissible moves.

        The replan is the whole recovery story (F99).  A tracked path is a
        reference, and the chassis drifts off it -- 59 mm was enough, measured:
        from there the straight line to the carrot clipped a patient the PATH
        went around, DWA correctly vetoed every candidate, and the first build
        simply gave up 700 mm short.  Replanning from where the robot actually
        IS costs half a millisecond and is the correct answer; the sticky cost
        the tracker just dropped stops the new plan repeating the old one."""
        self._birth()
        self.blocked = False
        for attempt in range(max(1, tries)):
            if self.cmap is None:
                path = [(self.x, self.y), (tx, ty)]
            else:
                t0 = t_now if t_now is not None else (
                    self.clock() if self.clock else 0.0)
                path, _ = nav.plan(self.cmap, (self.x, self.y), (tx, ty),
                                   R2_INSCRIBED, R2_CIRCUM, t0=t0, speed=v_max)
                if path is None:
                    self.blocked = True
                    return False
                path = list(path) + [(tx, ty)]
            ok = yield from self.follow_path(path, v_max=v_max, tol=tol,
                                             cap_s=cap_s)
            if ok:
                self.blocked = False
                return True
            if np.hypot(tx - self.x, ty - self.y) < tol * 1.6:
                self.blocked = False
                return True
        return False

    def push_to(self, tx, ty, v=170.0, tol=35.0, cap_s=16.0):
        """A plow push: straight, slow, and gently steered so the puck stays
        in the pocket.  No costmap avoidance here on purpose -- the push
        corridor was proved clear by the push planner before we committed,
        and swerving mid-push is how you lose the cargo."""
        n = int(cap_s * hal.Clock.HZ)
        jx, jy, jn = self.x, self.y, 0
        for _ in range(n):
            dx, dy = tx - self.x, ty - self.y
            if float(np.hypot(dx, dy)) < tol:
                break
            err = _wrap(np.degrees(np.arctan2(dy, dx)) - self.th)
            self._drive(v * max(0.4, 1.0 - abs(err) / 90.0),
                        float(np.clip(1.6 * err, -55.0, 55.0)))
            self.tick()
            yield
            jn += 1
            if jn >= 60:
                if np.hypot(self.x - jx, self.y - jy) < 12.0:
                    self.stop()
                    return False
                jx, jy, jn = self.x, self.y, 0
        self.stop()
        for _ in range(3):
            self.tick(); yield
        return True

    def back_off(self, mm_, v=220.0):
        n = int(mm_ / v * hal.Clock.HZ)
        for _ in range(n):
            self._drive(-v, 0.0)
            self.tick()
            yield
        self.stop()
        for _ in range(3):
            self.tick(); yield

    def shake_out(self, n=4):
        self.link.shake(n)
        for _ in range(int(n * 0.55 * hal.Clock.HZ) + 10):
            self.tick()
            yield
        self.stop()


# ============================================================== the mission
# Field facts the push catalog is built from (params.Field, m2_layout):
# sticker columns x 80/160 (west) and 983/1063 (east), rows y 537 (bottom),
# 650, 763 (top); zones HOSP (471-671, >=901), RECOVERY (700-900, 190-270),
# PCC_L (<=200, >=981), PCC_R (>=943, >=981).  What a puck can do is wall
# geometry (F94, measured the hard way -- the first catalog parked the
# approach INSIDE the wall's corner sweep and the pivot batted the pucks):
#
#   * every column pushes NORTH or SOUTH along itself freely;
#   * mid columns (160/983) also take +-65-degree diagonals: the approach
#     point 133 mm behind the puck then keeps the body's 93 mm corner
#     circle 11 mm clear of the wall.  Shallower angles do not fit, so an
#     eastward escape off the west columns is ~0.42 mm of x per mm pushed;
#   * edge columns (80/1063) have standing room for NOTHING but the
#     column itself: their reds and greens stay adrift (-3), priced;
#   * within a column: the bottom row's south-going and the top row's
#     north-going pushes cross nothing; the middle row waits for whichever
#     neighbour its own push direction crosses.
#
# Approaches arrive PRE-FACED: _push stages 240 mm behind the puck along
# the push line first (when that point is in-field), so the goto's arrival
# heading IS the push heading and no plow-sweeping pivot happens within
# reach of the puck (F94's second lesson).

DIAG = 65.0                                   # degrees off east-west
_ZONES = {"red": (511.0, 941.0, 631.0, 1141.0),      # HOSP inset 40
          "green": (740.0, 215.0, 860.0, 245.0),     # RECOVERY inset
          "yellowW": (40.0, 1021.0, 160.0, 1141.0),  # PCC_L inset
          "yellowE": (983.0, 1021.0, 1103.0, 1141.0)}


def _generic(i, x, y, c):
    """A one-leg push from wherever the puck IS to the nearest inset point
    of its destination zone; None when it is already home or the approach
    would leave the field."""
    key = c if c != "yellow" else ("yellowE" if x > 570.0 else "yellowW")
    x0, y0, x1, y1 = _ZONES[key]
    tx = float(np.clip(x, x0, x1))
    ty = float(np.clip(y, y0, y1))
    if abs(tx - x) < 4.0 and abs(ty - y) < 4.0:
        return None                          # already inside
    ux, uy, n = _norm(tx - x, ty - y)
    ax, ay = x - ux * 190.0, y - uy * 190.0
    # margin 70, not 95: robot 1's climb piles pucks 40 mm off the north
    # wall, and their east-west approaches live in that band -- arriving
    # pre-faced keeps the corner circle out of the wall (F97)
    if not _infield(ax, ay, margin=70.0):
        return None
    # carry 35 mm INTO the zone so the release leaves it inside
    return (2 if n < 200.0 else 6, i, x, y,
            [(tx + ux * 35.0, ty + uy * 35.0)])


def _classify(m, d=None):
    """Robot 1's colour classifier, model-camera convention: in render mode
    perception.classify_patch does this from pixels; the mission cannot
    tell.  With `d`, positions are CURRENT -- the east column is wherever
    robot 1's climb plowed it, not where the stickers were (F94)."""
    import mujoco
    if d is None or float(abs(d.xpos).sum()) < 1e-9:
        # match start: the live data has not been forwarded yet -- read the
        # spawn state (measured: planning from an unforwarded d aimed every
        # push at (0,0) and the robot chased the corner for 100 s)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
    out = []
    for i in range(64):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cyl%d" % i)
        if b < 0:
            break
        x, y = d.xpos[b][:2] * 1000.0
        g = m.geom_rgba[m.body_geomadr[b]]
        col = ("red" if g[0] > 0.6 and g[1] < 0.4 else
               ("green" if g[1] > 0.5 and g[0] < 0.5 else "yellow"))
        out.append((i, float(x), float(y), col))
    return out


def _infield(x, y, margin=95.0):
    return margin <= x <= 1143.0 - margin and margin <= y <= 1181.0 - margin


def _norm(dx, dy):
    n = float(np.hypot(dx, dy))
    return dx / n, dy / n, n


def _push(ctl, px, py, ux, uy, dist, v=175.0):
    """Acquire the puck at (px,py) and push it `dist` mm along unit (ux,uy).
    Stage 240 back along the line when the field allows, so the approach
    ARRIVES facing the push; face() only trims what is left."""
    from .params import Robot2 as R2
    hd = np.degrees(np.arctan2(uy, ux))
    sx, sy = px - ux * 240.0, py - uy * 240.0
    ax, ay = px - ux * (R2.PLOW_X + 52.0), py - uy * (R2.PLOW_X + 52.0)
    if _infield(sx, sy):
        yield from ctl.goto(sx, sy, v_max=310.0, tol=42.0, cap_s=6.0)
    yield from ctl.goto(ax, ay, v_max=250.0, tol=26.0, slow_into=160.0,
                        cap_s=6.0)
    if abs((hd - ctl.th + 180.0) % 360.0 - 180.0) > 10.0:
        yield from ctl.face(hd, tol=6.0, cap_s=3.0)
    tx, ty = px + ux * dist, py + uy * dist
    yield from ctl.push_to(tx - ux * (R2.PLOW_X - 10.0),
                           ty - uy * (R2.PLOW_X - 10.0), v=v,
                           cap_s=7.0 + dist / v)
    yield from ctl.back_off(95.0)


def _plan_side(pats, side, want="all"):
    """Ordered push tasks for one side.  Each: (prio, i, x, y, legs) where
    legs = [(ux, uy, dist), ...] applied in sequence with re-acquisition.
    Encodes the F94 catalog: which colour can leave which column, and the
    within-column crossing order."""
    east = side == "E"
    mid_x = 983.0 if east else 160.0
    s = -1.0 if east else 1.0               # x-direction away from the wall
    col = {}
    for i, x, y, c in pats:
        col.setdefault(round(x), []).append((i, x, y, c))
    tasks = []
    ca, sa = np.cos(np.radians(DIAG)), np.sin(np.radians(DIAG))
    pccy = 1010.0

    def diag_to_recovery(x, y):
        """leg A: down-and-out at -DIAG to y~240, leg B: flat to RECOVERY."""
        d = (y - 240.0) / sa
        lx = x + s * ca * d
        bx = 845.0 if east else 800.0
        return [(lx, 240.0), (bx, 240.0)]

    def diag_to_hosp(x, y):
        """leg A: up-and-out at +DIAG to y<=870, leg B: aimed into HOSP.
        From the east the entry aims HIGH (640, 1075): robot 1's kit pile
        lands at (535-590, 920-960) at T+66-72 and the east phase runs
        after it -- the north-east entry corridor clears the pile."""
        d = (min(870.0, y + 333.0) - y) / sa
        lx, ly = x + s * ca * d, y + sa * d
        if east:
            bx, by = float(np.clip(lx - 140.0, 560.0, 640.0)), 1075.0
        else:
            bx, by = float(np.clip(lx + 140.0, 495.0, 640.0)), 1005.0
        return [(lx, ly), (bx, by)]

    for cx, pucks in col.items():
        is_mid = abs(cx - mid_x) < 40.0
        for i, x, y, c in pucks:
            if want == "nogreens" and c == "green":
                continue
            if want == "greens" and c != "green":
                continue
            row = round(y)
            off_sticker = min(abs(x - 80.0), abs(x - 160.0),
                              abs(x - 983.0), abs(x - 1063.0)) > 60.0 or \
                min(abs(y - 537.0), abs(y - 650.0), abs(y - 763.0)) > 60.0
            if off_sticker:
                # robot 1's plow relocated it (the climb pile): a generic
                # straight push to the nearest inset point of its zone
                g = _generic(i, x, y, c)
                if g:
                    tasks.append(g)
                continue
            if not east and not is_mid:
                # x-80 stays UNTOUCHED (F98): it is west of the descent's
                # swath, so it is no seal hazard -- and a failed push there
                # strands a puck mid-corridor, which is one.  Its yellows'
                # +5s wait for a faster push pipeline.
                continue
            if c == "yellow":
                # NORTH, bottom-up (a row's approach nudges the row below;
                # re-spotting heals it, F96/F97) on a wall-standoff line
                # (F94: at x 80 the wing grazed the wall).  With non-yellow
                # mates ABOVE, the pocket would train them into the PCC at
                # -5 each: stop the train short of the zone, then finish
                # the yellow alone on a re-spotted second leg.
                xv = float(np.clip(x, 100.0, 1043.0))
                mates_above = any(cy > y + 40.0 and cc != "yellow"
                                  for _, _, cy, cc in pucks)
                legs = [(xv, 925.0), (xv, pccy)] if mates_above \
                    else [(xv, pccy)]
                tasks.append((10 + (row - 500) // 100, i, x, y, legs))
                continue
            elif not east and is_mid:
                # WEST mid column non-yellows (F97): the wall locks every
                # scoring diagonal, but these pucks ARE the seal-corridor
                # hazard (F87/F90) -- the fleet's founding job.  Park them
                # NORTH, out of the corridor (y > 780) and south of PCC_L,
                # after the yellows have gone through; the mop-up upgrades
                # any that get displaced east.
                # park at (108, 902): out of the corridor (y > 780), out
                # of PCC_L (y < 981), and west of robot 1's PCC_L approach
                # swath (x >= 122) -- the first park spot sat inside that
                # swath and the drop scattered both park and kits (F98)
                tasks.append((14 + (row - 500) // 100, i, x, y,
                              [(108.0, 902.0)]))
                continue
            elif not is_mid:
                # edge reds/greens: priced adrift.  x-80 sits west of the
                # descent's swath (122+), so it is not a seal hazard.
                continue
            elif c == "green":
                if row >= 700:
                    # top green: NE out (crosses nothing), then the long
                    # field push to RECOVERY
                    d = 107.0 / sa
                    lx, ly = x + s * ca * d, 870.0
                    bx, by = (845.0, 240.0) if east else (800.0, 240.0)
                    tasks.append((5, i, x, y, [(lx, ly), (bx, by)]))
                else:
                    # bottom/middle green: down-and-out; middle waits for
                    # the bottom sticker to clear (prio order handles it)
                    tasks.append((1 if row < 600 else 3, i, x, y,
                                  diag_to_recovery(x, y)))
            elif c == "red":
                if row >= 700:
                    tasks.append((5, i, x, y, diag_to_hosp(x, y)))
                else:
                    # bottom red goes NE early only if the row above it is
                    # already leaving (handled by prio: bottom-diag-up runs
                    # AFTER top/middle rows cleared)
                    tasks.append((7 if row < 600 else 6, i, x, y,
                                  diag_to_hosp(x, y)))
    tasks.sort(key=lambda k: k[0])
    return tasks


def _run_tasks(ctl, tasks, spot_puck, log, t, clock, stop_at, placed,
               hard_stop=None):
    """Camera-verified delivery: every leg starts from where the puck IS
    (robot 1's classifier re-spots it), and a leg that left it more than
    75 mm from its target gets ONE straight retry from the actual spot --
    open-loop pushes lost half their cargo and nobody knew (F96)."""
    for prio, i, x, y, legs in tasks:
        if clock() > stop_at:
            log(t() + "window over here")
            return
        t0 = clock()
        for k, (tx, ty) in enumerate(legs):
            for attempt in (0, 1):
                if clock() > stop_at or clock() - t0 > 16.0:
                    log(t() + "push %d out of time" % i)
                    break
                px, py = spot_puck(i)
                ux, uy, n = _norm(tx - px, ty - py)
                if n < 70.0:
                    break                    # this leg is done enough
                if hard_stop is not None and \
                        clock() + 9.0 + n / 160.0 > hard_stop:
                    # would still be pushing when the window slams: a task
                    # that overran T+52 put robot 2 inside robot 1's climb
                    # and cost the whole kit phase (F98)
                    log(t() + "push %d would overrun the window" % i)
                    return
                log(t() + "push %d leg %d%s: %.0f mm at %.0f deg"
                    % (i, k, "r" if attempt else "", n,
                       np.degrees(np.arctan2(uy, ux))))
                yield from _push(ctl, px, py, ux, uy, n + 12.0)
        placed[i] = spot_puck(i)


def mission_robot2(ctl, m, d=None, log=print, clock=None):
    """The whole match for the detached actuator.  One yield per tick."""
    t = (lambda: "") if clock is None else (lambda: "T+%5.1f R2 " % clock())
    pats = _classify(m, d)
    east = [p for p in pats if p[1] > 500.0]

    import mujoco as _mj
    _cb = {i: _mj.mj_name2id(m, _mj.mjtObj.mjOBJ_BODY, "cyl%d" % i)
           for i, _, _, _ in pats}
    _dd = d

    def spot_puck(i):
        if _dd is None:
            return (0.0, 0.0)
        p = _dd.xpos[_cb[i]]
        return float(p[0] * 1000.0), float(p[1] * 1000.0)
    log(t() + "colours: " + " ".join("%d%s@%.0f,%.0f" % (i, c[0], x, y)
                                     for i, x, y, c in pats))
    placed = {}

    def wait_until(ts):
        ctl.stop()
        while clock() < ts:
            ctl.tick()
            yield

    # ---- P0: kits FIRST, from the east-box spawn (F82/F95).  Six exit
    # routes through robot 1's half of the field all died on seed dice --
    # a 35 s wrestle, a mutual corner-lock, three wrecked sweeps -- because
    # no lane through it clears the sweep band, the parked body, the plate
    # and the samples at once.  The doc had the answer all along: robot 2
    # starts AGAINST THE EAST WALL (robot 1's spawn moved 144 mm west to
    # make room), so its opening act crosses nothing robot 1 will ever
    # touch: north-west under the sticker rows, up the east pinch it was
    # sized for, SHAKE into PCC_R with the north wall as the backstop.
    yield from ctl.goto(880.0, 420.0, v_max=340.0, tol=36.0)
    yield from ctl.goto(885.0, 700.0, v_max=320.0, tol=34.0)
    yield from ctl.goto(1040.0, 1080.0, v_max=300.0, tol=30.0)
    yield from ctl.face(270.0, tol=7.0)
    log(t() + "SHAKE: kits out against the north wall")
    yield from ctl.shake_out(4)

    # ---- P1: the EAST columns, EARLY (F98).  Robot 1 lives in the
    # south and west until its kit climb at ~T+56; the east stickers are
    # pristine and robot-1-free until then.  Working them now means the
    # climb's plow-pile later contains only what is left.  The WEST side
    # is not visited at all in this build: five configurations of west
    # pushing each re-rolled robot 1's seal into b0s -- the corridor
    # cannot host a 10-30 s noisy push pipeline and a seal.  The west
    # yellows' +20 and the F87 patient cure return when the pushes are
    # twice as fast; the roadmap owns it.
    yield from _run_tasks(ctl, _plan_side(east, "E"), spot_puck, log, t,
                          clock, stop_at=52.0, placed=placed, hard_stop=55.0)

    # hold in the north-east dead corner through robot 1's climb window
    log(t() + "holding at (1085, 880) through robot 1's climb")
    while clock() < 74.0:
        px, py, _ = ctl.pose
        if abs(px - 1085.0) > 90.0 or abs(py - 880.0) > 90.0:
            yield from ctl.goto(1085.0, 880.0, v_max=320.0, tol=45.0)
        ctl.stop()
        for _ in range(int(1.0 * hal.Clock.HZ)):
            ctl.tick()
            yield

    # ---- P3: east side once robot 1 has left it.  LOOK AGAIN each
    # round: robot 1's climb piles the east column near HOSP (measured),
    # and every push moves the map -- plan from the live positions until
    # the clock or the work runs out.
    yield from wait_until(74.0)
    while clock() < 108.0:
        east = [p for p in _classify(m, d) if p[1] > 400.0
                and p[0] not in placed]
        log(t() + "east re-look: " + " ".join(
            "%d%s@%.0f,%.0f" % (i, c[0], x, y) for i, x, y, c in east))
        tasks = _plan_side(east, "E")
        if not tasks:
            break
        n0 = len(placed)
        yield from _run_tasks(ctl, tasks, spot_puck, log, t, clock,
                              stop_at=112.0, placed=placed)
        if len(placed) == n0:
            break                            # no progress: stop replanning

    # ---- P4: PARK AND HOLD in the dead ground (F98).  "Done" is not a
    # state a detached actuator may improvise: left loose it was found in
    # the seal corridor being plowed by its own teammate.  The box's east
    # end is robot-1-dead after T+66; the hold re-checks, because robot 1
    # CAN shove this robot.
    log(t() + "parking in the north-east dead corner")
    while True:
        px, py, _ = ctl.pose
        if abs(px - 1085.0) > 90.0 or abs(py - 880.0) > 90.0:
            yield from ctl.goto(1085.0, 880.0, v_max=320.0, tol=45.0)
        ctl.stop()
        for _ in range(int(2.0 * hal.Clock.HZ)):
            ctl.tick()
            yield
