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
from . import hal, nav, trajectory
from .params import Robot2 as R2

R2_INSCRIBED = 75.0        # no orientation fits inside this
R2_CIRCUM = 98.0          # every orientation fits outside it
                           # (the capture pocket is part of the body)

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

    def drive(self, v, w):
        """The DriveHAL verb, so robot 1's PROVEN tracker can steer this
        robot unchanged (F107).  trajectory.track_waypoints consumes only
        .pose / .drive / .stop, and it is the same pure-pursuit that has
        carried robot 1's kit dogleg through six green checks -- where
        robot 2's bespoke DWA could not cross the board reliably."""
        self._drive(v, w)

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
    _CORNERS = [(-78.0, 55.0), (-78.0, -55.0), (78.0, 55.0), (78.0, -55.0),
                (70, 38), (70, -38), (70, 0.0), (0.0, 0.0)]

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

    def _dwa(self, path, i_ahead, v_max, w_max, grid, body=None,
             carry=False):
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
        if carry:
            # CARRYING: never stop, never spin (F106).  The pocket holds a
            # patient only while the chassis is driving INTO it; a stationary
            # pivot swings the walls sideways past a puck that has nothing
            # behind it and flings it out the mouth -- measured, 4 captures
            # of 4 survived the approach and none survived a 90-degree turn
            # in place.  So a laden robot turns by driving arcs.
            cand = [(v, w) for v, w in cand if v > 60.0]
            if not cand:
                cand = [(90.0, w) for w in
                        np.linspace(-0.5 * w_max, 0.5 * w_max, 9)]
        else:
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
                    cap_s=None, grid=None, carry=False):
        """Track a planned path with robot 1's pure-pursuit tracker.

        THE BESPOKE DWA IS RETIRED HERE (F107).  It was the right idea and
        it never became reliable: across a twelve-leg grid it delivered six,
        and in the delivery rig it left the chassis 906 mm short of a
        stand-off it had been asked for -- after which everything downstream
        (capture, carry, release) was measuring a robot that had never
        arrived.  trajectory.track_waypoints is the same code robot 1 uses,
        it is checked, and this project's whole premise is that both robots
        run the same software.  The body-footprint veto survives as a
        SAFETY check rather than as the steering law.
        """
        if not path:
            return True
        self._birth()
        if grid is None and self.cmap is not None:
            grid = self.cmap.inflated(6.0, 8.0)
        if cap_s is None:
            cap_s = 5.0 + nav.path_length(path) / 110.0
        pts = [(float(x), float(y)) for x, y in path]
        gx, gy = pts[-1]
        gen = trajectory.track_waypoints(
            self, pts, v_max=v_max, v_end=110.0 if not carry else 90.0,
            tol_end=tol, lookahead=0.62, strict=False)
        n = int(cap_s * hal.Clock.HZ)
        jx, jy, jn = self.x, self.y, 0
        for _ in range(n):
            try:
                next(gen)
            except StopIteration as e:
                self.stop()
                for _ in range(3):
                    self.tick(); yield
                ok = bool(e.value) or \
                    np.hypot(gx - self.x, gy - self.y) < tol * 1.6
                self.blocked = not ok
                return ok
            self.tick()
            yield
            jn += 1
            if jn >= 55:
                if np.hypot(self.x - jx, self.y - jy) < 14.0:
                    self.jams += 1
                    if self.cmap is not None:
                        self.cmap.add_sticky(self.x, self.y)
                    yield from self.back_off(95.0)
                    self.blocked = True
                    return False
                jx, jy, jn = self.x, self.y, 0
        self.stop()
        ok = np.hypot(gx - self.x, gy - self.y) < tol * 1.6
        self.blocked = not ok
        return ok

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

    def goto(self, tx, ty, v_max=340.0, w_max=200.0, tol=30.0, cap_s=None,
             t_now=None, tries=4, strict=False, carry=False):
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
            ok = yield from self.follow_path(path, v_max=v_max, w_max=w_max,
                                             tol=tol, cap_s=cap_s,
                                             carry=carry)
            if ok:
                self.blocked = False
                return True
            if np.hypot(tx - self.x, ty - self.y) < tol * 1.6:
                self.blocked = False
                return True
        return False

    def capture(self, px, py, cap_s=5.0):
        """Close the last ~120 mm onto a patient so it seats in the pocket.
        Straight, slow, and blind to the costmap on purpose: the target IS
        an obstacle and we mean to touch it."""
        for _ in range(int(cap_s * hal.Clock.HZ)):
            dx, dy = px - self.x, py - self.y
            n = float(np.hypot(dx, dy))
            if n < R2.CAPTURE_X + 12.0:
                break
            err = _wrap(np.degrees(np.arctan2(dy, dx)) - self.th)
            self._drive(150.0 * max(0.35, 1.0 - abs(err) / 80.0),
                        float(np.clip(1.4 * err, -40.0, 40.0)))
            self.tick()
            yield
        # settle against the stop
        for _ in range(int(0.4 * hal.Clock.HZ)):
            self._drive(110.0, 0.0)
            self.tick()
            yield
        self.stop()
        for _ in range(3):
            self.tick(); yield

    def holding(self, px, py):
        """Is the patient still in the pocket?  Cheap and honest: where the
        camera says it is, in the chassis frame."""
        a = np.radians(self.th)
        dx, dy = px - self.x, py - self.y
        lx = dx * np.cos(a) + dy * np.sin(a)
        ly = -dx * np.sin(a) + dy * np.cos(a)
        return (R2.STOP_X - 12.0 < lx < R2.STOP_X + R2.POCKET_D + 26.0
                and abs(ly) < R2.POCKET_W / 2.0 + 16.0)

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
# RESTORED TO THE 62/250 CONFIGURATION (F107), on top of the capture
# hardware.  The capture pocket is proven in isolation -- 4/4 including a
# 90-degree turn and a 400 mm carry -- but the capture-based MISSION built
# around it in one sitting scored 41/250 against this one's 62, with
# patients no better and kits and beams worse.  Mechanism and integration
# are different problems: the mechanism stays, the mission goes back to what
# measured best, and the capture integration earns its way in through rigs
# under match conditions before it touches a board again.
#
# Pushing with a capture pocket is strictly better than pushing with a
# plow anyway: the puck seats against the stop instead of riding loose in
# front of a blade.
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

# ROBOT 1'S RESERVATIONS, read from robot 1's own live plan (F112).
# Robot 2's A* runs on the residual, so the two cannot collide by
# construction instead of negotiating at 20 Hz (F95/F98).
# The corridor each of robot 1's tasks actually occupies, as a centreline.
# These are its own route's lanes: the east dogleg round the laboratory
# plate (which cannot be crossed), the y-730 traverse south of the kit
# drops, the climb to the hospital lip, the west lane the beams stage from.
R1_LANES = {
    "SWEEP": [[(60.0, 130.0), (700.0, 130.0)], [(60.0, 215.0), (700.0, 215.0)]],
    "L1":    [[(390.0, 230.0), (500.0, 230.0)]],
    "L2":    [[(500.0, 230.0), (640.0, 230.0)]],
    "L3":    [[(640.0, 230.0), (780.0, 230.0)]],
    "KH":    [[(780.0, 220.0), (945.0, 260.0), (935.0, 650.0), (770.0, 790.0)],
              [(730.0, 850.0), (700.0, 960.0)]],
    "KL":    [[(660.0, 930.0), (250.0, 930.0)]],
    "BEAMS": [[(240.0, 860.0), (240.0, 700.0)], [(190.0, 620.0), (180.0, 200.0)],
              [(300.0, 375.0), (140.0, 370.0)]],
}
R1_HALF = 150.0        # robot 1's body plus a working gap, not its worst-case
                       # 185 mm sweep: reserving the sweep for the whole of
                       # every phase left robot 2 with no field at all and it
                       # simply stopped working (measured: eight of twelve
                       # patients declared unreachable, parked at T+26).


def robot1_reservations(cmap, schedule=None, t_now=0.0):
    """Robot 1's corridors, as space-time keep-outs.

    THE WINDOWS COME FROM ROBOT 1'S OWN PLAN when one is offered (F112).
    Robot 2's controller runs on robot 1's Pi, so there is no reason to
    guess: route.mission_agent_a publishes its live Schedule on the robot
    at every replan, and each remaining task's window is exactly the
    interval that plan gives it -- travel in, plus service.

    Hardcoded windows were defensible while robot 1 ran one fixed order.
    They stopped being defensible the moment its planner started choosing
    between the third laboratory slot and PCC_L, because the running order
    now differs seed to seed.  A corridor reserved for 72-84 s while robot
    1 is in it at 60-70 is worse than no reservation: robot 2 reads the
    lane as free and parks in it.  Without a schedule (rigs, and robot 1's
    own opening survey before it has planned) this falls back to the
    measured shape of the nominal plan.
    """
    if schedule is None or not getattr(schedule, "tasks", None):
        for nm, lanes in R1_LANES.items():
            for lane in lanes:
                cmap.add_corridor(lane, R1_HALF, 0.0, 121.0
                                  if nm == "BEAMS" else 121.0)
        return cmap
    from . import planner
    prev = "SWEEP"
    for name, t0, dur in schedule.tasks:
        travel = planner.TRAVEL.get((prev, name), 8.0)
        # the lane is occupied from when robot 1 sets off for it until it
        # has finished there; a margin either side covers the model's own
        # error, which observe() is still correcting as the match runs
        w0, w1 = t0 - travel - 3.0, t0 + dur + 3.0
        for lane in R1_LANES.get(name, ()):
            cmap.add_corridor(lane, R1_HALF, max(w0, t_now - 1.0), w1)
        prev = name
    return cmap


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


def _board_map(pucks, skip=None, sched=None, t_now=0.0):
    """A fresh costmap with robot 1's reservations and every patient except
    the one(s) being handled.  skip is an index or a set of them."""
    cm = nav.CostMap.field()
    robot1_reservations(cm, schedule=sched, t_now=t_now)
    drop = set() if skip is None else (
        {skip} if isinstance(skip, (int, np.integer)) else set(skip))
    for i, x, y, c in pucks:
        if i in drop:
            continue
        cm.add_disc(x, y, 12.0)
    return cm



# ==================================================== the patient mission
# WHY THIS IS AN ORDERING PROBLEM BEFORE IT IS A CONTROL PROBLEM (F110).
#
# The twelve patients stand in two 2x3 blocks pressed against the side
# walls: 80 mm apart across the field, 113 mm along it.  This chassis is
# 110 mm wide, so it cannot pass BETWEEN two neighbours -- the removal
# order is forced by geometry, not chosen by a planner:
#
#     west block   x=160 must come out before x=80 is reachable at all
#     east block   x=983 must come out before x=1063
#
# Across a column the robot approaches broadside from the field centre,
# and its 110 mm width clears easily between neighbours 226 mm apart, so
# the three rows are free in any order.  The old mission asked plan_push
# for a cheapest-first ordering that knew nothing of this, spent its legs
# re-staging at a 250 mm stand-off three times per delivery, and mostly
# reported "nothing deliverable from here".
#
# The clock decides HOW MANY.  A delivery is approach + capture + carry +
# release; carries run 400 mm (a block to its own PCC) to 700 mm (to
# HOSPITAL or RECOVERY), so one costs 10-15 s and the match affords about
# eight of the twelve.  Which eight is worth real points: the referee pays
# +6 for four reds in HOSPITAL, +8 for yellows 2/2 across the PCCs and +6
# for four greens in RECOVERY, and no per-patient constant can see a set
# bonus.  So the selection is value-per-second over what is legal RIGHT
# NOW, and the value is referee.score_cylinders on the projected board --
# the same discipline the fleet's kit pricing needed (F109).
#
# Robot 1 owns the third constraint.  Its beam seal takes the west lane
# from T+86 and the west block's only approach corridor runs through it,
# so transits are planned STRICT against robot1_reservations' space-time
# windows: robot 2 works the east block first (it starts there, and its
# kits are there) and is refused the west corridor once the seal opens
# rather than negotiating for it at 20 Hz.
# ZONE OWNERSHIP IS A PARTITION, AND THE GEOMETRY LEAVES NO CHOICE (F111).
#
# HOSPITAL, PCC_L and PCC_R are destinations for BOTH scoring columns.
# Robot 1 drops six kits into HOSPITAL at T+68 and two into PCC_L at T+72;
# robot 2 then carries patients into the same rectangles.  Releasing a
# patient leaves the flare tips 88 mm past it, and HOSPITAL is 200 mm wide
# with the pile down its centre at x 546-554, so a 110 mm chassis cannot
# stand anywhere in that zone without touching the pile -- from the south,
# from the west, or deep from the north, all three were checked and all
# three foul it.  PCC_L is worse: robot 1's kits land 6 mm inside the
# southern lip, exactly where a carry releases.
#
# The arithmetic settles it even where the geometry is arguable.  A
# delivered patient is worth +8.  Shovelling the HOSPITAL pile costs the
# six kits (-18), the empty-zone penalty (-10) and the 6/2/2 bonus (-20):
# up to -48.  There is no version of that trade worth taking, and the
# measured boards agree -- robot 1 alone lands 6 and 2 in the same spots on
# every seed, while a fleet board's kit column swings between -30 and +50.
#
# So the fleet partitions by ZONE rather than negotiating inside one:
#
#     robot 1   HOSPITAL, PCC_L        (kits)
#     robot 2   RECOVERY, PCC_R        (its own kits, dropped deep first)
#
# PCC_R is the proof that the partition is the right shape: robot 2 shakes
# its kits out backwards to the zone's far edge, 185 mm deep, then releases
# patients at the near edge 150 mm away, and nothing has ever disturbed
# them.  Robot 2 therefore delivers the greens (RECOVERY, which holds no
# kits at all) and the east-side yellows, and leaves the reds and west
# yellows where they are: -3 each is cheaper than the alternative.
R2_ZONES = ("RECOVERY", "PCC_R")

CARRY_V = 190.0            # F106: arcs only, never a pivot, with a puck
CARRY_W = 90.0
APPROACH_V = 330.0


def _zone_pt(zone, puck):
    """Where inside the zone to put this patient: the nearest legal point,
    so a delivery is the shortest carry that still scores."""
    return (float(np.clip(puck[0], zone[0] + 35.0, zone[2] - 35.0)),
            float(np.clip(puck[1], zone[1] + 35.0, zone[3] - 35.0)))


def _board_now(pucks, live, placed):
    """(x, y, colour) for all twelve, as the tracker currently believes."""
    return [(placed[i][0], placed[i][1], c) if i in placed else (*live(i), c)
            for i, _, _, c in pucks]


def _marginal(board, i, zone):
    """Referee points gained by putting patient i in that zone -- set
    bonuses, wrong-zone penalties and the adrift baseline included."""
    from . import referee
    after = list(board)
    after[i] = ((zone[0] + zone[2]) / 2.0, (zone[1] + zone[3]) / 2.0,
                board[i][2])
    return (referee.score_cylinders(after)[0]
            - referee.score_cylinders(board)[0])


def _capture_pose(app, puck):
    """Where the chassis ends up once the puck is seated."""
    ux, uy = np.cos(np.radians(app[2])), np.sin(np.radians(app[2]))
    return (puck[0] - ux * R2.CAPTURE_X, puck[1] - uy * R2.CAPTURE_X)


def _spoil_point(cm, pucks, live, near, drop):
    """Somewhere harmless to put a patient that is merely IN THE WAY.

    Moving a blocker is FREE: the referee pays -3 for a patient outside
    every destination zone whether it is standing on its sticker or shoved
    300 mm sideways.  So the peel that the packing forces -- outer column
    before inner, because a 110 mm chassis cannot pass between neighbours
    80 mm apart -- costs nothing but seconds.  What it must not do is
    create a NEW problem: the spoil point is clear of every zone (a patient
    dropped in the wrong one is -5, and one dropped on robot 1's kits is
    far worse), clear of the other patients, and clear of the corridors
    robot 1 has reserved.
    """
    grid = cm.inflated(R2_INSCRIBED, R2_CIRCUM)
    gx, gy = cm._grid_xy()
    ok = grid < nav.BLOCKED
    for z in ZONES.values():
        ok &= ~((gx >= z[0] - 90.0) & (gx <= z[2] + 90.0) &
                (gy >= z[1] - 90.0) & (gy <= z[3] + 90.0))
    for j, _, _, _ in pucks:
        if j in drop:
            continue
        px, py = live(j)
        ok &= (gx - px) ** 2 + (gy - py) ** 2 > 200.0 ** 2
    if not ok.any():
        return None
    d2 = (gx - near[0]) ** 2 + (gy - near[1]) ** 2
    d2 = np.where(ok, d2, np.inf)
    i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
    return float(gx[i, j]), float(gy[i, j])


def _blocker(pucks, live, i, zone, robot, t0, sched=None):
    """The neighbour whose removal makes patient i deliverable, if any.

    Only true neighbours are candidates: at 80 mm across and 113 along, a
    patient blocks the one beside it and nothing further away, so this is a
    handful of costmap builds rather than a search.
    """
    px, py = live(i)
    cand = sorted((j for j, _, _, _ in pucks if j != i),
                  key=lambda j: np.hypot(live(j)[0]-px, live(j)[1]-py))
    for j in cand:
        if np.hypot(live(j)[0]-px, live(j)[1]-py) > 260.0:
            break
        cm = _board_map([(k, *live(k), c) for k, _, _, c in pucks],
                        skip={i, j}, sched=sched, t_now=t0)
        if _price(cm, robot, (px, py), zone, t0) is not None:
            return j
    return None


def _price(cm, robot, puck, zone, t0):
    """(seconds, approach) for one delivery, or None if it cannot be done.

    Both legs are planned STRICT: a transit that would cross a corridor
    robot 1 has reserved is not cheap, it is unavailable.
    """
    app = nav.capture_approach(cm, puck)
    if app is None:
        return None
    _, s1 = nav.plan(cm, robot, app[:2], R2_INSCRIBED, R2_CIRCUM,
                     t0=t0, speed=APPROACH_V, strict=True)
    if not np.isfinite(s1):
        return None
    tgt = _zone_pt(zone, puck)
    _, s2 = nav.plan(cm, _capture_pose(app, puck), tgt, R2_INSCRIBED,
                     R2_CIRCUM, t0=t0 + s1 + 3.0, speed=CARRY_V, strict=True)
    if not np.isfinite(s2):
        return None
    #        approach   turn+capture   carry   release+back off
    return float(s1 + 4.0 + s2 + 3.0), app


def _deliver(ctl, i, live, target, app, log, t, zone=None, what="patient"):
    """One patient, end to end: stand off, face it, take it, carry it, let
    go.  Returns True only if the RESULT is what was wanted -- inside the
    zone for a delivery, or actually moved for a clearance.  Nothing here
    assumes; the camera checks after every stage, because a capture that
    silently failed used to be carried all the way to a zone and released
    into thin air."""
    ok = yield from ctl.goto(app[0], app[1], v_max=APPROACH_V, tol=70.0,
                             tries=3)
    gap = float(np.hypot(ctl.pose[0] - app[0], ctl.pose[1] - app[1]))
    if gap > 190.0:
        log(t() + "  %s %d: never reached the stand-off (%.0f mm)"
            % (what, i, gap))
        return False
    px, py = live(i)
    yield from ctl.face(float(np.degrees(np.arctan2(py - ctl.pose[1],
                                                    px - ctl.pose[0]))),
                        tol=5.0)
    p0 = live(i)
    yield from ctl.capture(*live(i))
    if not ctl.holding(*live(i)):
        log(t() + "  %s %d: capture missed" % (what, i))
        return False
    tx, ty = target
    ux, uy, _ = _norm(tx - live(i)[0], ty - live(i)[1])
    yield from ctl.goto(tx - ux * R2.CAPTURE_X, ty - uy * R2.CAPTURE_X,
                        v_max=CARRY_V, w_max=CARRY_W, tol=55.0, tries=3,
                        carry=True)
    yield from ctl.back_off(150.0)
    fx, fy = live(i)
    if zone is None:
        good = np.hypot(fx - p0[0], fy - p0[1]) > 120.0
    else:
        good = zone[0] <= fx <= zone[2] and zone[1] <= fy <= zone[3]
    log(t() + "  %s %d: %s at (%.0f, %.0f)"
        % (what, i, "done" if good else "short", fx, fy))
    return bool(good)


def _work_patients(ctl, pucks, live, log, t, now, deadline=112.0,
                   sched=None):
    """Value-per-second greedy over whatever is legal right now, with the
    peel the packing forces.

    Two kinds of move.  A DELIVERY earns the referee's marginal points for
    putting a patient in a zone robot 2 owns.  A CLEARANCE earns nothing
    directly and costs nothing either -- a patient outside every zone is
    -3 wherever it stands -- but it is how the inner column becomes
    reachable at all, because this chassis cannot pass between neighbours
    80 mm apart.  Deliveries first; a clearance only when nothing can be
    delivered, and only when it unlocks something that can.
    """
    placed, spent = {}, set()
    _sc = sched if callable(sched) else (lambda: sched)

    def wanted(i, c, p):
        """The zone robot 2 may deliver this patient to, or None."""
        zn = DEST.get(c) if c != "yellow" else \
            ("PCC_R" if p[0] > 570.0 else "PCC_L")
        return ZONES[zn] if zn in R2_ZONES else None

    while now() < deadline:
        board = _board_now(pucks, live, placed)
        full = [(i, *live(i), c) for i, _, _, c in pucks]
        best = None
        for i, _, _, c in pucks:
            if i in placed or i in spent:
                continue
            p = live(i)
            zone = wanted(i, c, p)
            if zone is None:
                continue                   # robot 1 owns that rectangle
            if zone[0] <= p[0] <= zone[2] and zone[1] <= p[1] <= zone[3]:
                placed[i] = p              # already home
                continue
            gain = _marginal(board, i, zone)
            if gain <= 0.0:
                continue
            cmi = _board_map(full, skip=i, sched=_sc(), t_now=now())
            pr = _price(cmi, ctl.pose[:2], p, zone, now())
            if pr is None:
                continue
            secs, app = pr
            if now() + secs > deadline + 6.0:
                continue
            rate = gain / max(secs, 1.0)
            if best is None or rate > best[0]:
                best = (rate, i, gain, secs, app, zone, cmi,
                        _zone_pt(zone, p), "patient")

        if best is None:
            # ---- nothing deliverable: is something merely IN THE WAY? ----
            for i, _, _, c in pucks:
                if i in placed or i in spent:
                    continue
                zone = wanted(i, c, live(i))
                if zone is None:
                    continue
                j = _blocker(pucks, live, i, zone, ctl.pose[:2], now(),
                             sched=_sc())
                if j is None or j in spent:
                    continue
                cmj = _board_map(full, skip=j, sched=_sc(), t_now=now())
                sp = _spoil_point(cmj, pucks, live, live(j), {i, j})
                if sp is None:
                    continue
                app = nav.capture_approach(cmj, live(j))
                if app is None:
                    continue
                _, s1 = nav.plan(cmj, ctl.pose[:2], app[:2], R2_INSCRIBED,
                                 R2_CIRCUM, t0=now(), speed=APPROACH_V,
                                 strict=True)
                if not np.isfinite(s1):
                    continue
                log(t() + "patient %d is blocked by %d -- clearing it to "
                    "(%.0f, %.0f)" % (i, j, sp[0], sp[1]))
                best = (0.0, j, 0.0, s1 + 12.0, app, None, cmj, sp, "blocker")
                break

        if best is None:
            log(t() + "nothing deliverable from here")
            return
        _, i, gain, secs, app, zone, cm, tgt, what = best
        if what == "patient":
            log(t() + "patient %d: %+.0f pts in %.0f s (%.2f pts/s)"
                % (i, gain, secs, gain / max(secs, 1.0)))
        ctl.cmap = cm
        good = yield from _deliver(ctl, i, live, tgt, app, log, t,
                                   zone=zone, what=what)
        if what == "blocker":
            if not good:
                spent.add(i)           # could not be shifted: stop trying
        elif good:
            placed[i] = live(i)
        else:
            spent.add(i)               # one honest attempt each; the clock
                                       # is worth more than a second try


def mission_robot2(ctl, m, d=None, log=print, clock=None, rb=None):
    """Survey, plan, execute, repair.  One yield per 50 Hz tick.

    rb is robot 1, when the fleet is running one: robot 2's controller
    lives on robot 1's Pi, so it reads that robot's published Schedule and
    reserves its corridors in space-time from the plan robot 1 is actually
    following (F112) rather than from a remembered one.
    """
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

    def sched():
        return getattr(rb, "schedule", None) if rb is not None else None

    # ---- the kits: PCC_R, from the east-box spawn -----------------------
    ctl.cmap = _board_map(pucks, sched=sched(), t_now=now())
    ok = yield from ctl.goto(1043.0, 1075.0, v_max=360.0, tol=45.0)
    yield from ctl.face(270.0, tol=8.0)
    log(t() + "SHAKE: kits into PCC_R")
    yield from ctl.shake_out(4)
    yield from ctl.goto(1040.0, 900.0, v_max=340.0, tol=50.0)

    # ---- the patients: go there, GRAB it, carry it, let go --------------
    yield from _work_patients(ctl, pucks, live, log, t, now,
                              sched=sched)

    # ---- park in the dead corner ----------------------------------------
    log(t() + "parking")
    ctl.cmap = _board_map(refresh(), sched=sched(), t_now=now())
    yield from ctl.goto(1085.0, 880.0, v_max=340.0, tol=60.0)
    ctl.stop()
    while True:
        ctl.tick()
        yield
