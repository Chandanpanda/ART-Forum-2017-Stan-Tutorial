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
        # BATCH THE PHYSICS BETWEEN SERVO TICKS (F103).  Stepping MuJoCo one
        # step at a time from Python, with a servo update wrapped round each,
        # cost ~1.7 ms per step against the ~0.03 ms the engine needs -- the
        # physics was a rounding error next to the interpreter overhead, and
        # a 12-seed board took ten minutes.  The servo runs at SERVO_HZ and
        # mj_step loops in C between ticks, which is both 5x faster and a
        # more honest model of a Pico running MicroPython.
        step = float(self.m.opt.timestep)
        every = max(1, int(round(1.0 / (R2.SERVO_HZ * step))))
        left = max(1, n)
        while left > 0:
            k = min(every, left)
            ml, mr = self.wheel_speeds()
            dt = k * step
            self.d.ctrl[self._al] = self.servo_l.update(ml, dt)
            self.d.ctrl[self._ar] = self.servo_r.update(mr, dt)
            self.servo_l.count(ml, dt)
            self.servo_r.count(mr, dt)
            mujoco.mj_step(self.m, self.d, nstep=k)
            left -= k

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
             tries=4, strict=False):
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
                                   R2_INSCRIBED, R2_CIRCUM, t0=t0, speed=v_max,
                                   strict=strict)
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
# ONE PLAN AT THE GUN (design doc section 15.5), then track it and repair it.
# The board is fully observable from the start line -- the only randomness a
# match holds is which colour stands on which sticker, and the camera sees
# that -- so the mission is not a script of hand-timed phases any more.  It
# is: survey, build the map, reserve robot 1's corridors in space-time, price
# every delivery with the push planner, take them in value order, and re-price
# against the live board after each one.

ZONES = {"HOSP": (511.0, 941.0, 631.0, 1141.0),
         "RECOVERY": (730.0, 200.0, 870.0, 260.0),
         "PCC_L": (40.0, 1021.0, 160.0, 1141.0),
         "PCC_R": (983.0, 1021.0, 1103.0, 1141.0)}
DEST = {"red": "HOSP", "green": "RECOVERY"}

# ROBOT 1'S RESERVATIONS.  Its planner knows its own schedule; until the two
# solvers are merged this is that schedule's measured shape -- the corridors
# it occupies and when.  Robot 2's A* runs on the residual, so the two cannot
# collide by construction instead of negotiating at 20 Hz (F95/F98).
def robot1_reservations(cmap):
    """Robot 1's corridors, as space-time keep-outs.

    THE WIDTH AND THE WINDOWS ARE BOTH BUDGETS.  Reserving robot 1's full
    185 mm swept radius for the whole of every phase leaves robot 2 with
    almost no field and it simply stops working (measured: it declared eight
    of twelve patients unreachable and parked at T+26).  These are the
    centrelines with a 150 mm half-width -- robot 1's body plus a working gap,
    not its worst-case sweep -- over the windows its phases actually occupy.
    When the two planners merge, this function is replaced by a read of
    robot 1's own Schedule."""
    cmap.add_corridor([(60.0, 130.0), (700.0, 130.0)], 150.0, 0.0, 26.0)
    cmap.add_corridor([(60.0, 215.0), (700.0, 215.0)], 150.0, 8.0, 28.0)
    cmap.add_corridor([(390.0, 230.0), (740.0, 230.0)], 150.0, 26.0, 58.0)
    cmap.add_corridor([(780.0, 220.0), (945.0, 260.0), (935.0, 650.0),
                       (770.0, 790.0)], 150.0, 56.0, 72.0)
    cmap.add_corridor([(730.0, 850.0), (700.0, 960.0)], 150.0, 62.0, 76.0)
    cmap.add_corridor([(660.0, 930.0), (250.0, 930.0)], 150.0, 72.0, 84.0)
    cmap.add_corridor([(240.0, 860.0), (240.0, 700.0)], 150.0, 78.0, 92.0)
    cmap.add_corridor([(190.0, 620.0), (180.0, 200.0)], 150.0, 86.0, 121.0)
    cmap.add_corridor([(300.0, 375.0), (140.0, 370.0)], 150.0, 100.0, 121.0)


def survey(m, d=None):
    """The opening perception act: every patient's position and colour.
    Model-camera convention, exactly like robot 1's synthetic see_lab -- in
    render mode perception.classify_patch does this from pixels and nothing
    above can tell the difference."""
    import mujoco
    if d is None or float(abs(d.xpos).sum()) < 1e-9:
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


def _norm(dx, dy):
    n = float(np.hypot(dx, dy))
    if n < 1e-9:
        return 1.0, 0.0, 0.0
    return dx / n, dy / n, n


def zone_of(colour, x):
    if colour == "yellow":
        return ZONES["PCC_R"] if x > 570.0 else ZONES["PCC_L"]
    return ZONES[DEST[colour]]


def _board_map(pucks, skip=None):
    """A fresh costmap with robot 1's reservations and every patient except
    the one being pushed."""
    cm = nav.CostMap.field()
    robot1_reservations(cm)
    for i, x, y, c in pucks:
        if skip is not None and i == skip:
            continue
        cm.add_disc(x, y, 12.0)
    return cm


def mission_robot2(ctl, m, d=None, log=print, clock=None):
    """Survey, plan, execute, repair.  One yield per 50 Hz tick."""
    t = (lambda: "") if clock is None else (lambda: "T+%5.1f R2 " % clock())
    now = clock if clock else (lambda: 0.0)
    pucks = survey(m, d)
    log(t() + "survey: " + " ".join("%d%s@%.0f,%.0f" % (i, c[0], x, y)
                                    for i, x, y, c in pucks))

    def live(i):
        """Where the tracker says puck i is now."""
        import mujoco
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cyl%d" % i)
        return (float(d.xpos[b][0] * 1000.0), float(d.xpos[b][1] * 1000.0))

    def refresh():
        return [(i, *live(i), c) for i, _, _, c in pucks]

    done = set()

    # ---- the kits: PCC_R, from the east-box spawn -----------------------
    ctl.cmap = _board_map(pucks)
    ok = yield from ctl.goto(1043.0, 1075.0, v_max=360.0, tol=45.0)
    yield from ctl.face(270.0, tol=8.0)
    log(t() + "SHAKE: kits into PCC_R")
    yield from ctl.shake_out(4)
    yield from ctl.goto(1040.0, 900.0, v_max=340.0, tol=50.0)

    # ---- the patients, cheapest-first, re-priced after every delivery ---
    while now() < 108.0:
        board = refresh()
        best = None
        for i, x, y, c in board:
            if i in done:
                continue
            z = zone_of(c, x)
            if z[0] <= x <= z[2] and z[1] <= y <= z[3]:
                done.add(i)
                continue
            cm = _board_map(board, skip=i)
            # leave-clean: never park a patient where robot 1 still has to
            # go -- EXCEPT in a destination zone (F102).  Robot 1's kit
            # corridor runs straight through the hospital, so an unqualified
            # mask forbade delivering a red to its own zone and robot 2 then
            # declared the whole board undeliverable and quit at T+49 with
            # sixty seconds in hand.  A scored patient is not an obstacle.
            avoid = np.zeros((cm.nx, cm.ny), dtype=bool)
            for mask, w0, w1 in cm._windows:
                if w1 > now():
                    avoid |= mask
            for zx0, zy0, zx1, zy1 in ZONES.values():
                gx, gy = cm._grid_xy()
                avoid &= ~((gx >= zx0 - 30.0) & (gx <= zx1 + 30.0) &
                           (gy >= zy0 - 30.0) & (gy <= zy1 + 30.0))
            legs, secs = nav.plan_push(cm, (x, y), z, robot=ctl.pose[:2],
                                       avoid=avoid)
            if legs is None:
                continue
            # value is the referee's: +5 delivered and +3 not-adrift = 8
            if best is None or secs < best[1]:
                best = (i, secs, legs, cm, (x, y))
        if best is None:
            # NOT a reason to stop: reservations expire, and the board keeps
            # changing under robot 1's wheels.  Wait for the next window and
            # look again -- quitting here threw away sixty seconds.
            if now() > 104.0:
                break
            log(t() + "nothing deliverable yet; waiting")
            for _ in range(int(3.0 * hal.Clock.HZ)):
                ctl.tick()
                yield
            continue
        i, secs, legs, cm, p0 = best
        if now() + secs > 112.0:
            log(t() + "%.0f s of work left, %.0f s of clock -- stopping"
                % (secs, 112.0 - now()))
            break
        log(t() + "puck %d: %d legs, %.1f s" % (i, len(legs), secs))
        ctl.cmap = cm
        failed = False
        for k, (tx, ty) in enumerate(legs):
            px, py = live(i)
            ux, uy, n = _norm(tx - px, ty - py)
            if n < 55.0:
                continue
            # THE APPROACH DOES NOT NEED TO BE PRECISE (F101), and asking
            # for precision is what cost the deliveries.  What the push
            # actually requires is: be BEHIND the puck, FACING the push
            # direction, with the puck inside a 120 mm pocket.  The first
            # version drove to an exact pose 130 mm back on a 34 mm
            # tolerance and abandoned the puck when the tracker could not
            # nail it -- 12-20 s a puck, nothing delivered.  So: park loosely
            # well behind it, turn onto the push line, then CLOSE the last
            # 120 mm in a straight line.  The straight run is what funnels
            # the puck into the pocket, and it needs no planner at all.
            hd = float(np.degrees(np.arctan2(uy, ux)))
            sx, sy = px - ux * 250.0, py - uy * 250.0
            ok = yield from ctl.goto(sx, sy, v_max=330.0, tol=70.0,
                                     tries=2, strict=True)
            # 300, not 190: the straight close below covers the rest, and
            # abandoning from inside a chassis-length of the staging point
            # is throwing away a delivery that was nearly made
            if not ok and np.hypot(ctl.pose[0] - sx, ctl.pose[1] - sy) > 300.0:
                failed = True
                break
            yield from ctl.face(hd, tol=6.0)
            # close on the puck: the plow's mouth does the centring
            yield from ctl.push_to(px - ux * 40.0, py - uy * 40.0, v=200.0,
                                   tol=45.0, cap_s=4.0)
            yield from ctl.push_to(tx - ux * 60.0, ty - uy * 60.0, v=180.0,
                                   cap_s=6.0 + n / 150.0)
            yield from ctl.back_off(95.0)
            if now() > 112.0:
                break
        done.add(i)
        if failed:
            log(t() + "puck %d: approach blocked, moving on" % i)

    # ---- park in the dead corner ----------------------------------------
    log(t() + "parking")
    ctl.cmap = _board_map(refresh())
    yield from ctl.goto(1085.0, 880.0, v_max=340.0, tol=60.0)
    ctl.stop()
    while True:
        ctl.tick()
        yield
