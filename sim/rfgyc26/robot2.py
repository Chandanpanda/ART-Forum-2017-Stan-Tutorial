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
from . import hal, nav, trajectory, world, fleet as fleetmod
from .params import Robot2 as R2, Field, Piece

R2_INSCRIBED = 75.0        # no orientation fits inside this
CARRY_PAD    = 20.0        # extra room a LOADED drive asks for (F136)
CLEAR_NEAR   = 85.0        # mm of clearance at which speed is floored
CLEAR_FAR    = 130.0       # ...and above which it is unrestricted
CLEAR_SLOW   = 110.0       # mm/s through a pinch (F121, F138)
EJECT_BACK   = R2.EJECT_BACK   # rig-measured; see check_effectors
KIT_V        = 230.0       # mm/s with kits loose on the tray
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
        try:
            self._ag_l = _aid(m, "r2_gate_l_srv")
            self._ag_r = _aid(m, "r2_gate_r_srv")
        except Exception:
            self._ag_l = self._ag_r = None
        self._gate_want = 0.0                # 0 shut, 1 open
        self._gate_at = 0.0
        self._gate_t0 = 0.0
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
        elif f[0] == "G" and len(f) == 2:
            # the servo takes GATE_T to travel; the firmware just writes the
            # pulse width and the horn gets there when it gets there
            self._gate_want = 1 if int(f[1]) else 0
            self._gate_t0 = now

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
        # the gate servo, slewed at the horn's real speed
        if self._ag_l is not None:
            frac = 1.0 if R2.GATE_T <= 0 else min(
                1.0, (now - self._gate_t0) / R2.GATE_T)
            here = self._gate_at + (self._gate_want - self._gate_at) * frac
            if frac >= 1.0:
                self._gate_at = float(self._gate_want)
            span = np.radians(R2.GATE_OPEN)
            self.d.ctrl[self._ag_l] = -span * here
            self.d.ctrl[self._ag_r] = span * here
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
        self.fleet = None          # rfgyc26.fleet.Fleet, set by the runner
        self.fleet_name = "r2"
        self.cmap = cmap
        self.escaping = False
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
        # tell the executive where we are, every control tick.  This is the
        # only thing robot 2 owes the fleet, and it is what lets robot 1
        # refuse to drive into it (F124).
        if self.fleet is not None:
            self.fleet.observe(self.fleet_name, (self.x, self.y, self.th),
                               vel=abs(self._cmd[0] + self._cmd[1]) / 2.0)

    @property
    def pose(self):
        self._birth()
        return self.x, self.y, self.th

    # ---- wire ------------------------------------------------------------
    # ---- the escape reflex ----------------------------------------------
    ESC_V = 220.0             # mm/s of retreat -- brisk, this is a reflex
    ESC_K = 2.2               # deg/s of turn per degree of bearing error
    ESC_MARGIN = 10.0         # mm of slack on top of the two radii

    def _closing_hazard(self):
        """The nearest other-robot footprint inside the reflex bubble, or None.

        Measured positions only.  A prediction belongs in the costmap, where
        a planner can route round it; down here a prediction is just a way
        to flinch at something that is not there (F124).
        """
        best, bd = None, 1e9
        me = self.fleet.agents.get(self.fleet_name)
        rad = me.radius if me else R2_CIRCUM
        for hx, hy, hr in self.fleet.hazard(self.fleet_name, horizon=0.0):
            d = float(np.hypot(hx - self.x, hy - self.y))
            if d < hr + rad + self.ESC_MARGIN and d < bd:
                best, bd = (hx, hy, hr), d
        return best

    def _escape(self, haz):
        """Wheel speeds that strictly increase separation from `haz`.

        Reversing straight out is right only when the hazard is dead ahead;
        from a glancing bearing it walks the tail into it.  Distance grows
        at -v.cos(b) for a bearing b, so the sign of v follows the sign of
        cos(b) -- back away from something in front, drive away from
        something behind -- and the turn drives |b| to 0 or 180, which is
        where that rate is largest.  One command, and it converges from any
        relative bearing.

        The arc is checked against the static map before it is sent, and the
        other sign tried if it is blocked: a reflex that reverses into a
        wall has only changed which obstacle it hits.
        """
        b = np.radians(self.th)
        brg = np.arctan2(haz[1] - self.y, haz[0] - self.x) - b
        brg = (brg + np.pi) % (2*np.pi) - np.pi
        ahead = abs(brg) <= np.pi/2
        # err: how far to swing the nose so the escape runs along the axis
        want = 0.0 if ahead else np.copysign(np.pi, brg)   # aim the axis
        err = np.degrees(brg - want)
        body = None
        if self.cmap is not None:
            body = self.cmap.inflated(6.0, 8.0)
        for sgn in ((-1.0, 1.0) if ahead else (1.0, -1.0)):
            v = sgn * self.ESC_V
            if body is not None:
                a2 = b + np.radians(np.clip(self.ESC_K*err, -120.0, 120.0)) * 0.3
                nx = self.x + v * 0.30 * np.cos(a2)
                ny = self.y + v * 0.30 * np.sin(a2)
                if self._hits(nx, ny, a2, body):
                    continue
            w = float(np.clip(self.ESC_K * err, -R2.W_MAX, R2.W_MAX))
            dv = np.radians(w) * R2.TRACK / 2.0
            return (float(np.clip(v - dv, -R2.V_MAX, R2.V_MAX)),
                    float(np.clip(v + dv, -R2.V_MAX, R2.V_MAX)))
        return None                 # boxed in -- see the caller

    def _drive(self, v, w):
        """(v mm/s, w deg/s) -> wheel speeds -> the wire, at 20 Hz."""
        dv = np.radians(w) * R2.TRACK / 2.0
        vl = float(np.clip(v - dv, -R2.V_MAX, R2.V_MAX))
        vr = float(np.clip(v + dv, -R2.V_MAX, R2.V_MAX))
        # THE ESCAPE REFLEX (F125).  This layer used to zero the command,
        # mirroring a veto on robot 1's side, and two agents that both stop
        # for each other wedge instead of avoiding: rendered on seed 3 they
        # sat in contact from T+72 to the end of the match.  Robot 1 no
        # longer brakes at all, so the whole avoidance burden is here -- and
        # a burden discharged by standing still is not discharged.  Robot 2
        # therefore MOVES AWAY, which is the one response that cannot
        # deadlock: it strictly increases separation every tick it runs.
        if self.fleet is not None:
            haz = self._closing_hazard()
            if haz is not None:
                esc = self._escape(haz)
                self.escaping = esc is not None
                if esc is not None:
                    vl, vr = esc
                # esc is None: boxed in, so the commanded motion stands.
                # Standing still achieves nothing when the other robot
                # never stops -- and it is what froze this robot on the
                # start line, 262 mm from a parked robot 1, for a whole
                # match (F129).  The planner chose that motion against a
                # costmap this robot's footprint is already stamped into;
                # it is the best information available.
            else:
                self.escaping = False
        # SPEED FOLLOWS CLEARANCE, BUT ONLY WHERE THE CLEARANCE IS LOW
        # (F138).  This limit used to be computed in goto() from
        # min_clearance over the WHOLE path and applied to the whole drive,
        # so one 85 mm pinch anywhere along 900 mm of route put the robot at
        # 110 mm/s for all of it -- and with eleven patients, two walls, the
        # laboratory plate and robot 1 all stamped, nearly every path has
        # one such point.  Measured on the phase rig, that is most of the
        # delivery cycle: approach 9.9 s and carry 8.2 s of 28.5, an
        # effective 61 mm/s against a 330 mm/s cap.
        #
        # A pinch is a property of a PLACE, not of a route through it.  Here
        # the governor sees where the chassis actually is, every tick, so
        # the tight bit is slow and the open floor is not.
        if self.cmap is not None and v != 0.0:
            gap = float(self.cmap.clearance()[self.cmap.cell(self.x, self.y)])
            if gap < CLEAR_FAR:
                lim = float(np.interp(gap, [CLEAR_NEAR, CLEAR_FAR],
                                      [CLEAR_SLOW, abs(v)]))
                lim = max(lim, CLEAR_SLOW)
                if abs(v) > lim:
                    k = lim / abs(v)
                    vl, vr = vl * k, vr * k
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
        pts = [(float(x), float(y)) for x, y in path]
        # SPEED FOLLOWS CLEARANCE (F121).  The two ways north out of the
        # deployment box are pinches with about 26 mm of slack for this
        # chassis, and a tracker running at 360 mm/s carries more
        # cross-track error than that: measured, robot 2 clipped the
        # laboratory plate's south-east corner four seconds into the match
        # and stayed wedged against it until the buzzer, on seed after
        # seed.  It fits at a walk and it does not fit at a run, so the
        # tightest point on the path sets the speed for the leg.
        if self.cmap is not None:
            # (the speed limit itself now lives in _drive, where it can be
            #  evaluated WHERE THE ROBOT IS -- see F138 there)
            pass
        if cap_s is None:
            cap_s = 5.0 + nav.path_length(path) / 110.0
        gx, gy = pts[-1]
        gen = trajectory.track_waypoints(
            self, pts, v_max=v_max, v_end=110.0 if not carry else 90.0,
            tol_end=tol, lookahead=0.62, strict=False,
            w_max=w_max if carry else None, no_pivot=carry)
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
                # A LOADED CARRY ASKS FOR MORE ROOM THAN AN EMPTY DRIVE
                # (F136).  Planning at the inscribed radius says the body
                # fits, which is true of the PATH and not of the driving:
                # the tracker corners inside its own tolerance, and in a
                # corridor the plan only just fits that difference is a
                # patient shoved out of the way.  Measured on the cycle rig,
                # carrying patient 0 up the west edge pushed patient 2 from
                # (80, 650) to (81, 1069) -- 419 mm, and into PCC_L, where a
                # green is -5 rather than -3.  So ask for the margin, and
                # take the tight route only when there is no other.
                pad = CARRY_PAD if carry else 0.0
                path, _ = nav.plan(self.cmap, (self.x, self.y), (tx, ty),
                                   R2_INSCRIBED + pad, R2_CIRCUM + pad,
                                   t0=t0, speed=v_max, strict=strict)
                if path is None and pad:
                    path, _ = nav.plan(self.cmap, (self.x, self.y), (tx, ty),
                                       R2_INSCRIBED, R2_CIRCUM, t0=t0,
                                       speed=v_max, strict=strict)
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

    def gate(self, open_, settle=None):
        """Drive the capture gate and wait for the horn to arrive."""
        self.link.gate(open_)
        for _ in range(int((R2.GATE_T if settle is None else settle)
                           * hal.Clock.HZ) + 4):
            self.tick()
            yield

    def release(self, mm_=150.0):
        """Let a patient go: open the gate, then back away from it.

        This is the half a passive pocket could not do (F123).  With the
        gate shut the patient is captive through a reverse -- which is what
        makes the inner columns reachable -- so it has to be told when to
        stop being captive."""
        yield from self.gate(True)
        yield from self.back_off(mm_)
        yield from self.gate(False)

    def capture(self, px, py, cap_s=5.0):
        """Close the last ~120 mm onto a patient so it seats in the pocket.
        Straight, slow, and blind to the costmap on purpose: the target IS
        an obstacle and we mean to touch it.  The gate opens to take it and
        shuts behind it."""
        yield from self.gate(True)
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
        # settle against the stop, THEN shut the gate behind it
        for _ in range(int(0.4 * hal.Clock.HZ)):
            self._drive(110.0, 0.0)
            self.tick()
            yield
        self.stop()
        yield from self.gate(False)
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

    def carry_turn(self, th_t, radius=130.0, v=170.0, tol=28.0, cap_s=7.0):
        """Come round onto a heading WITHOUT PIVOTING, cargo aboard.

        Every patient stands against a side wall and every destination is
        toward the middle of the field, so a delivery's carry almost always
        begins 100-180 degrees off its bearing.  A pivot answers that in
        two seconds and loses the puck -- the pocket is open at the front,
        and the chassis frame shows a seated puck walking from x 46 to the
        flare tips at 79 inside 1.2 s of a 100 deg/s stand-and-turn (F113).
        A differential drive does not need the pivot: a forward arc of
        radius v/w turns just as far, and at 170 mm/s and 75 deg/s that is
        130 mm, which this field has room for almost everywhere.

        Both turn directions are tried against the body footprint, nearer
        first; if neither fits, the caller still has the cargo and can be
        told so rather than grinding into a wall.
        """
        w = np.degrees(v / max(radius, 1.0))
        occ = None if self.cmap is None else \
            (self.cmap.inflated(6.0, 8.0) >= nav.BLOCKED)

        def fits(x, y, th, ahead):
            if occ is None:
                return True
            a = np.radians(th)
            x += ahead * np.cos(a)
            y += ahead * np.sin(a)
            for lx, ly in R2.BODY_PTS:
                wx = x + lx*np.cos(a) - ly*np.sin(a)
                wy = y + lx*np.sin(a) + ly*np.cos(a)
                i, j = self.cmap.cell(wx, wy)
                if occ[i, j]:
                    return False
            return True

        def probe(sgn, r):
            th_probe = self.th
            for _ in range(6):
                th_probe += sgn * 15.0
                if not fits(self.x, self.y, th_probe, r * 0.9):
                    return False
            return True

        err = _wrap(th_t - self.th)
        sign = 1.0 if err > 0 else -1.0
        # A LADDER OF RADII, NOT ONE (F139).  At 170 mm/s and 75 deg/s the
        # arc is 130 mm, and where 130 does not fit this used to drive it
        # anyway: neither probe direction cleared, `sign` kept its opening
        # guess, and the loop ran its whole seven-second budget as a blind
        # 1.2 m arc with a patient aboard.  Measured on the phase rig that
        # was four deliveries in nine, and it is how the neighbours end up
        # scattered.  A tighter arc is the same manoeuvre at a lower speed
        # -- what F113 forbids is the PIVOT, not the radius -- so try 130,
        # then 95, then 70, each way, and only then borrow room behind.
        found = None
        for r in (radius, radius * 0.73, radius * 0.54):
            for s_try in (sign, -sign):
                if probe(s_try, r):
                    found = (s_try, r)
                    break
            if found:
                break
        if found:
            sign, radius = found
            v = radius * np.radians(w)          # keep w, shrink the speed
        else:
            # NOTHING FITS FORWARD: BACK OUT FIRST (F133).  The gate is shut
            # and holds the patient through the reverse, so the room a
            # wall-side capture does not have in front of it can be
            # borrowed from behind.  Straight line, no radius needed.
            room = 0.0 if occ is None else _back_room(occ, self.cmap,
                                                      (self.x, self.y), self.th)
            if room >= 60.0:
                yield from self.back_off(room)
                err = _wrap(th_t - self.th)
                sign = 1.0 if err > 0 else -1.0
                for r in (radius, radius * 0.73, radius * 0.54):
                    for s_try in (sign, -sign):
                        if probe(s_try, r):
                            found = (s_try, r)
                            break
                    if found:
                        break
                if found:
                    sign, radius = found
                    v = radius * np.radians(w)
            if not found:
                # Nothing fits from anywhere: say so rather than grinding.
                self.stop()
                for _ in range(3):
                    self.tick(); yield
                return False
        # budget the arc rather than a flat seven seconds: what it needs is
        # the angle over the rate, and anything past that is a failure
        # burning clock the delivery has not got
        cap_s = min(cap_s, 1.5 + abs(_wrap(th_t - self.th)) / max(w, 1.0) * 1.7)
        for _ in range(int(cap_s * hal.Clock.HZ)):
            err = _wrap(th_t - self.th)
            if abs(err) < tol:
                break
            self._drive(v, sign * w)
            self.tick()
            yield
        self.stop()
        for _ in range(3):
            self.tick(); yield
        return abs(_wrap(th_t - self.th)) < tol * 2.0

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
ZONE_NAME = {v: k for k, v in ZONES.items()}   # rect -> name, for the chooser
# ZONES IS AN AIMING BOX, NOT THE TAPE (F143).  Every rectangle above is
# inset 40 mm so a carry that stops inside its tolerance still lands inside
# the line.  Judging the RESULT by the same box calls a delivery short when
# the referee would pay for it -- measured on seed 19, patient 7 came to
# rest at (1042, 1017), four millimetres under the aiming box's south edge
# and forty inside PCC_R's actual one.  It was marked failed, marked spent,
# and left out of the board the colour bonuses are priced against.
TAPE = {"HOSP": Field.HOSPITAL, "RECOVERY": Field.RECOVERY,
        "PCC_L": Field.PCC_L, "PCC_R": Field.PCC_R}
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
    # THE SEAL IS A QUADRANT, NOT A LANE (F122).  Three centrelines
    # described the transit into the south-west and none of the work: the
    # staging dance, the two run-ins, the withdrawals and the shuffles all
    # happen inside roughly x < 420, y < 900, and robot 1 spends thirty-five
    # seconds there doing them.  Reserving only the way in let robot 2 walk
    # into the middle of it -- beams 55.8 -> 45.8/70 the moment robot 2
    # became mobile enough to get there.  Robot 2 has no business in that
    # quadrant at all: its own destination zone is RECOVERY at x 730-870.
    "BEAMS": [[(240.0, 860.0), (240.0, 700.0)], [(190.0, 620.0), (180.0, 200.0)],
              [(300.0, 375.0), (140.0, 370.0)],
              [(60.0, 120.0), (380.0, 120.0)], [(60.0, 300.0), (380.0, 300.0)],
              [(60.0, 480.0), (380.0, 480.0)], [(60.0, 660.0), (380.0, 660.0)],
              [(60.0, 840.0), (380.0, 840.0)]],
}
# ROBOT 1'S OWN SWEPT RADIUS, and no less (F122).  150 was "its body plus a
# working gap", chosen when reserving the full 185 left robot 2 with no
# field -- but that was with windows spanning the whole match, before the
# schedule made them the intervals robot 1 is actually in a lane.  The
# corridor mask is inflated by ROBOT 2's radius on top of this (see
# CostMap.inflated), so this number is robot 1's alone, and robot 1 sweeps
# 185 mm.  Reserving 150 was reserving less than the robot.
R1_HALF = 185.0


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
    from . import planner
    if schedule is None or not getattr(schedule, "tasks", None):
        # THE FALLBACK IS THE NOMINAL PLAN, NOT A CLOSED BOARD.  Robot 1
        # does not publish a schedule until its opening sweep ends at
        # T+27, which is exactly when robot 2 finishes its kits and wants
        # its first patient -- and a fallback that reserved every lane for
        # the whole match answered "nothing deliverable from here" and
        # parked it at T+26 for the remaining 95 seconds.  Reserving
        # everything is not caution, it is a different wrong answer.
        schedule = planner.plan(planner.SWEEP_NOMINAL, at="SWEEP")
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


def kits_home(m, d=None, which=(8, 9)):
    """How many of robot 2's own kits are inside PCC_R.

    Model-camera convention, exactly like survey(): in render mode this is
    two white rectangles found in a known rectangle of the field, which is
    the easiest perception task on either robot.
    """
    import mujoco
    from .params import Field
    if d is None:
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
    z = Field.PCC_R
    n = 0
    for i in which:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "kit%d" % i)
        if b < 0:
            continue
        x, y = d.xpos[b][:2] * 1000.0
        if z[0] <= x <= z[2] and z[1] <= y <= z[3]:
            n += 1
    return n


def _norm(dx, dy):
    n = float(np.hypot(dx, dy))
    if n < 1e-9:
        return 1.0, 0.0, 0.0
    return dx / n, dy / n, n


def zone_of(colour, x):
    if colour == "yellow":
        return ZONES["PCC_R"] if x > 570.0 else ZONES["PCC_L"]
    return ZONES[DEST[colour]]


def _board_map(pucks, skip=None, sched=None, t_now=0.0, flt=None):
    """A fresh costmap with robot 1's reservations, robot 1's LIVE position,
    and every patient except the one(s) being handled.

    The reservations are a promise about the future; the live stamp is a
    measurement of the present (F124).  Robot 2 needs both, and it had
    neither of the second: it set off across robot 1's tail in the first
    second of the match because robot 1 simply was not on its map.
    """
    return world.board_map(pieces=pucks, skip=() if skip is None else skip,
                           schedule=sched, reserve=robot1_reservations,
                           t_now=t_now, fleet=flt, whose="r2")



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
# The rule that falls out is simpler than a schedule: NO ROBOT ENTERS A
# ZONE THAT HOLDS SCORED KITS.  Robot 2's own PCC_R kits are no different
# from robot 1's -- the first version of this partition let robot 2 deliver
# east-side yellows into PCC_R on the grounds that its kits sit 185 mm deep
# and a release lands 150 mm away from them, and the board disagreed: the
# kit column fell from 41/50 to 19.75/50, with PCC_R empty on ten of twelve
# seeds, for a patient column that gained one point.  150 mm of paper
# clearance is not clearance once the approach, the shuffle and the
# back-off are counted.
#
#     robot 1   HOSPITAL, PCC_L
#     robot 2   RECOVERY -- the one destination zone that holds no kits
#
# But "never" is too blunt, and the schedule robot 1 now publishes (F112)
# says exactly how blunt.  A kit zone is only dangerous once the kits are
# IN it.  Robot 1 reaches HOSPITAL around T+65 and PCC_L around T+75, so
# there is a real window from the end of robot 2's own kit run at T+25
# until then in which those rectangles are empty and a patient dropped
# there is worth its full +8.  Robot 2 closes each zone at robot 1's
# scheduled arrival, with a margin, and never reopens it.
#
# THE WINDOW WAS TRIED ONCE AND THE BOARD SAID NO -- to the mechanism, not
# to the idea.  Opening HOSPITAL and PCC_L until robot 1's scheduled arrival
# lost 34 points a match: kits 19.75 -> 3.0, beams 58.75 -> 33.3, for 2.5
# points of patients.  The diagnosis then was "an empty rectangle is not a
# safe one -- robot 2 leaves a patient standing in the spot robot 1 must
# drive through".  That diagnosis was right, and it is a statement about a
# robot that PUSHED: a plow cannot choose where a puck comes to rest, so
# "put it in the zone" was the whole of the placement policy and the zone
# is only 120 mm of usable width.
#
# Two things have changed since, and both bear directly on it.  The gate
# means robot 2 now PLACES a patient at a chosen point rather than shoving
# it to a stop, and fleet.kit_hazard means the point can be chosen against
# a model of the floor robot 1 actually needs -- which is well under half
# of each zone, because robot 1 parks BESIDE a zone and discharges over its
# flank rather than driving in.  _zone_pt maximises clearance from that
# floor, from the kit pile beside it, and from every patient already down.
#
# And the arithmetic says this is not optional.  Against the referee:
#
#     robot 2 in RECOVERY only         patients  +2      board ceiling ~142
#     + HOSPITAL                       patients +40      board ceiling ~180
#     + both PCCs                      patients +80      board ceiling  250
#
# All twelve patients start outside every destination zone, which is -36
# before either robot moves.  RECOVERY holds four of them, so the partition
# that reserved every other zone for robot 1 caps the patient column at +2
# and the whole board around 142 -- the target is out of reach by
# construction, however well the two robots drive.  A -34 experiment is a
# reason to change the mechanism, not a reason to accept a ceiling.
R2_ZONES = ("RECOVERY", "HOSP", "PCC_L", "PCC_R")
ZONE_TASK = {"HOSP": "KH", "PCC_L": "KL"}
CLOSE_MARGIN = 10.0        # be out before robot 1 sets off, not as it arrives
REOPEN_MARGIN = 4.0        # and back in only once it has actually gone


def _r1_clear(zname, flt):
    """Is robot 1 physically out of that rectangle right now?"""
    if flt is None:
        return True
    a = flt.agents.get("r1")
    if a is None or a.pose is None:
        return True
    box = fleetmod.REGIONS.get(zname)
    if box is None:
        return True
    r = a.radius
    return not (box[0]-r <= a.pose[0] <= box[2]+r and
                box[1]-r <= a.pose[1] <= box[3]+r)


def zone_open(zname, now, sched, flt=None):
    """May robot 2 deliver into this zone right now?

    Three windows, not one.  Robot 1 visits a kit zone once, for about ten
    seconds, somewhere near the middle of the match; the rectangle is
    robot 2's before that and robot 2's again afterwards, and the second
    window is the larger of the two.  The old test knew only the first and
    read "not in the plan any more" -- which is what a FINISHED task looks
    like -- as a reason to stay out for good.
    """
    if zname not in R2_ZONES:
        return False
    task = ZONE_TASK.get(zname)
    if task is None:
        return True                       # RECOVERY, PCC_R: no robot 1 kits
    if sched is None or not getattr(sched, "tasks", None):
        from . import planner
        sched = planner.plan(planner.SWEEP_NOMINAL, at="SWEEP")
    from . import planner
    travel = planner.TRAVEL.get(("L3", task), 8.0)
    for name, t0, dur in sched.tasks:
        if name == task:
            if now < t0 - travel - CLOSE_MARGIN:
                return True               # robot 1 has not set off yet
            if now > t0 + dur + REOPEN_MARGIN:
                return _r1_clear(zname, flt)
            return False                  # it is on its way, or there
    # Not in the plan: robot 1 has either finished the task or dropped it
    # under time pressure.  Either way nothing more is coming, so the only
    # question left is whether it is still standing in the rectangle.
    return _r1_clear(zname, flt)


HOLD = (1085.0, 880.0)     # the north-east dead corner
# THE GATE RAISED THE CARRY SPEED TOO (F146).  190 mm/s was set for the OPEN
# pocket, where the cargo rode against a stop with nothing in front of it and
# any lateral acceleration walked it out of the mouth -- the same fact that
# forbade the pivot (F106).  Shut, the gate is what holds it, and the rig
# says the speed stopped mattering: a 700 mm dogleg carried a patient 3 of 3
# at 190, 270, 350 and 420 mm/s, with the seated gap growing only from about
# 25 mm to 35.  What has NOT changed is the geometry -- carry_turn's arcs and
# every clearance around them are validated at a 130 mm radius -- so the turn
# rate rises with the speed to hold that radius: 300/130 rad/s is 132 deg/s.
CARRY_V = 300.0
CARRY_W = 130.0
APPROACH_V = 330.0


EDGE_HARD = 12.0           # ZONES is already inset 40 mm from the tape


def _zone_pt(zone, puck, zname=None, avoid=(), keep=()):
    """Where inside the zone to put this patient.

    35 mm of inset was arithmetic, not engineering.  The carry ends when
    the tracker is within its 55 mm tolerance and the release then backs
    away, so a target 35 mm inside the line can land 20 mm outside it --
    measured, a green carried the whole width of the board and stopped at
    x 691 against RECOVERY's edge at 700, nine millimetres short, for
    nothing.  Inset by the tolerance instead, and collapse to the middle
    when the zone is too small to allow it (RECOVERY is only 80 mm deep).

    INSIDE THE TAPE IS NOT ENOUGH IN A SHARED ZONE (F126).  Three of the
    four destination zones also take robot 1's kits, and a patient left on
    the floor robot 1 parks on is a patient robot 1 shoulders back out --
    that is what cost 34 points a match the first time these zones were
    opened.  Robot 1's floor is known (fleet.kit_hazard: it parks beside
    the zone and discharges over its flank, so it covers well under half of
    one), so the choice is made by clearance rather than by proximity:
    among the legal points, take the one furthest from everything already
    down, and break ties by the shortest carry.  With nothing down yet that
    is the middle of the free floor, which is exactly right.
    """
    lo = []
    for a, b in ((zone[0], zone[2]), (zone[1], zone[3])):
        # RECOVERY is 60 mm deep in ZONES terms and must hold four: an
        # inset that is generous in a 200 mm box leaves nothing in a 60 mm
        # one, so it yields once the zone is the smaller constraint.
        inset = min(EDGE_HARD, (b - a) / 2.0 - 20.0)
        lo.append((a + inset, b - inset))
    (x0, x1), (y0, y1) = lo
    # WHERE ROBOT 1 SAID IT WILL BE, not a rectangle drawn round a station
    # constant (F148).  Robot 1 now solves for its own pose and publishes
    # the floor that pose occupies -- its footprint and its kit pile -- so
    # this keeps off the real thing instead of a model of a former one.
    haz = [(x, y, r + Piece.CYL_D/2.0 + 12.0) for x, y, r in keep]
    best, bs = None, -1e18
    for x in np.linspace(x0, x1, 21):
        for y in np.linspace(y0, y1, 21):
            if any((x-hx)**2 + (y-hy)**2 <= hr*hr for hx, hy, hr in haz):
                continue
            clear = min((float(np.hypot(x-ax, y-ay)) for ax, ay in avoid),
                        default=400.0)
            if clear < Piece.CYL_D + 5.0:   # 20 mm bodies: do not touch
                continue
            edge = min(x-zone[0], zone[2]-x, y-zone[1], zone[3]-y)
            # Keeping off the tape and keeping off the last patient are the
            # same requirement, so they are the same term: the worst of the
            # two, capped where more room stops helping.  Then the shorter
            # carry, at a millimetre of score per centimetre of drive.
            sc = min(clear, edge, 90.0) - 0.1*float(np.hypot(x-puck[0],
                                                             y-puck[1]))
            if sc > bs:
                best, bs = (float(x), float(y)), sc
    if best is not None:
        return best
    # Nowhere clear: fall back to the old nearest-comfortably-inside point
    # rather than refusing the delivery.  A patient in the zone is +8 even
    # if robot 1 later nudges it; a patient never carried is -3 for certain.
    return (float(np.clip(puck[0], x0, x1)), float(np.clip(puck[1], y0, y1)))


def _placed_in(placed, zone):
    """Patients already standing in that rectangle -- what a new one must
    not be put on top of."""
    return [q for q in placed.values()
            if zone[0] <= q[0] <= zone[2] and zone[1] <= q[1] <= zone[3]]


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


def _blocker(pucks, live, i, zone, robot, t0, sched=None, flt=None):
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
                        skip={i, j}, sched=sched, t_now=t0, flt=flt)
        if _price(cm, robot, (px, py), zone, t0) is not None:
            return j
    return None


BACK_OUT = 200.0        # how far the loaded pocket may reverse before turning


def _body_free(occ, cm, x, y, th):
    a = np.radians(th)
    for lx, ly in R2.BODY_PTS:
        i, j = cm.cell(x + lx*np.cos(a) - ly*np.sin(a),
                       y + lx*np.sin(a) + ly*np.cos(a))
        if occ[i, j]:
            return False
    return True


def _back_room(occ, cm, pose, th0, want=BACK_OUT, step=25.0):
    """How far the chassis can reverse along -th0 before something is in it."""
    a = np.radians(th0)
    got = 0.0
    while got + step <= want:
        x = pose[0] - (got + step) * np.cos(a)
        y = pose[1] - (got + step) * np.sin(a)
        if not _body_free(occ, cm, x, y, th0):
            break
        got += step
    return got


def _can_turn_out(cm, pose, th0, th_t, radius=130.0):
    """Can the robot leave a capture pose, loaded, onto the carry bearing?

    THE REVERSE IS PART OF THE ANSWER NOW (F133).  This asked only whether a
    forward arc fitted, which was the right question for a PASSIVE pocket --
    that build could not reverse without shedding the puck, so a capture
    with no forward arc was a capture the robot could not leave.  The servo
    gate changed the mechanism and nothing here noticed: shut, it holds a
    patient through 250 mm of reverse (check_r2_pocket, BACK 6/6), which is
    exactly the room a wall-side capture is short of.

    Measured on the seed 6 board, the forward-only test refused seven of
    twelve patients -- the whole east side, every one of them a ten to
    thirteen second delivery -- and robot 2 never attempted them in any
    match.  So: try the arc from the capture pose, and if nothing fits,
    back straight out as far as the map allows and try again from there.
    """
    # AND IT MUST MODEL WHAT carry_turn ACTUALLY DOES.  That behaviour
    # tries three radii each way before it borrows room behind (F139); a
    # predicate that only ever asks about the widest one refuses deliveries
    # the robot could make.  Measured over twelve seeds' opening boards,
    # radius-130-only refused 57 of 144 patients -- forty per cent of the
    # column, every one of them on this test alone.
    return turn_out_margin(cm, pose, th0, th_t, radius) > 0.0


# HOW COMFORTABLY, NOT JUST WHETHER (F147).  Admitting more deliveries is
# only worth it if they land, and the four rungs below are not equally
# likely to: an arc that fits at 130 mm from where the robot stands is the
# manoeuvre these clearances were measured on, while one that needs 70 mm
# after backing 200 mm out of a sticker column is the same manoeuvre with
# every margin spent.  Measured, opening the gate on the tight ones (F145)
# priced 9.9 patients of 12 instead of 7.2 and the board's patient column
# went DOWN 6.2 points -- the extra attempts cost a full cycle each and
# mostly failed.
#
# These weights are an ORDERING, not calibrated probabilities.  What they
# have to get right is that a comfortable delivery outranks a marginal one
# of equal value, and that a marginal one still outranks doing nothing.
# _price divides its quote by this, which is the expected-cost form: an
# attempt that succeeds half the time costs twice the seconds per point.
_TURN_RUNGS = (1.0, 0.72, 0.45)          # 130 mm arc, 95 mm, 70 mm
_TURN_BACKED = 0.55                      # ...and after borrowing room behind


def turn_out_margin(cm, pose, th0, th_t, radius=130.0):
    """0 if the robot cannot leave this capture pose loaded, else how
    comfortably it can, 1 being the full arc from where it stands."""
    occ = cm.inflated(6.0, 8.0) >= nav.BLOCKED
    ladder = (radius, radius * 0.73, radius * 0.54)
    for r, w in zip(ladder, _TURN_RUNGS):
        if _arc_out(occ, cm, pose, th0, th_t, r):
            return w
    back = _back_room(occ, cm, pose, th0)
    if back < 60.0:
        return 0.0
    a = np.radians(th0)
    pose = (pose[0] - back*np.cos(a), pose[1] - back*np.sin(a))
    for r, w in zip(ladder, _TURN_RUNGS):
        if _arc_out(occ, cm, pose, th0, th_t, r):
            return w * _TURN_BACKED
    return 0.0


def _arc_out(occ, cm, pose, th0, th_t, radius=130.0):
    """Is there a forward arc from (pose, th0) onto th_t that the body fits?

    The same sweep carry_turn will actually drive, checked before the robot
    commits to a capture it cannot leave.  Either direction counts; a
    delivery only needs one of them.
    """
    err = _wrap(th_t - th0)
    for sgn in (1.0 if err > 0 else -1.0, -1.0 if err > 0 else 1.0):
        x, y, th = pose[0], pose[1], th0
        turned, ok = 0.0, True
        while turned < abs(err) - 1e-9 and turned < 200.0:
            dth = sgn * 10.0
            th += dth
            turned += 10.0
            # advance along the arc by the same 10 degrees
            x += radius * np.radians(10.0) * np.cos(np.radians(th))
            y += radius * np.radians(10.0) * np.sin(np.radians(th))
            a = np.radians(th)
            for lx, ly in R2.BODY_PTS:
                i, j = cm.cell(x + lx*np.cos(a) - ly*np.sin(a),
                               y + lx*np.sin(a) + ly*np.cos(a))
                if occ[i, j]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return True
    return False


def _price(cm, robot, puck, zone, t0, zname=None, avoid=(), keep=()):
    """(seconds, approach) for one delivery, or None if it cannot be done.

    A RESERVATION IS A PREDICTION, SO IT PRICES A TRANSIT RATHER THAN
    FORBIDDING ONE (F128).  Both legs used to be planned strict -- a path
    across a corridor robot 1 had reserved was unavailable, not expensive --
    and measured over robot 1's own published plan that is not a
    restriction, it is a shutdown:

        T+20   12 of 12 patients reachable strict,  12 soft
        T+46    0 of 12                          ,   8 soft
        T+70    0 of 12                          ,   6 soft
        T+96    0 of 12                          ,  12 soft

    Robot 1's tour touches the box, the laboratory, both pinches, the
    hospital, PCC_L and the seal quadrant -- which is most of the board --
    so from T+46 the strict test answers "nothing is deliverable" for the
    remaining seventy-four seconds of every match.  Traced on seed 6, robot
    2 attempts two patients in 120 s, fails both, and then stages and parks
    with ten still standing on their stickers.  That is the whole -29/80.

    Hard constraints belong to what is MEASURED; a booking made sixty
    seconds ago is a guess, and a guess deserves a price.  The soft retry
    charges 4000 a cell for crossing a live window, which buys the way
    round wherever there is one and only crosses when there is not.  The
    protections that actually keep the two apart are unchanged and there
    are three of them: robot 1's measured footprint is a hard obstacle in
    this same costmap, zone_open still refuses a delivery into a zone robot
    1 is about to want, and the executive can still order this robot out of
    a region outright.
    """
    # APPROACH FROM THE SIDE THE DESTINATION IS NOT ON (F113).  The carry
    # is where deliveries were dying: the pocket is open at the front, so
    # any stand-and-turn walks the puck out of the mouth, and a robot that
    # captured facing east and must deliver west has to turn 180 degrees
    # with the cargo aboard.  Arriving along the puck-to-zone line removes
    # the turn instead of surviving it -- the robot ends the capture
    # already pointing where the carry goes, and the whole delivery is a
    # straight run in, a stop, and a straight run out.  This is the one
    # thing a plow always got right and the first pocket mission threw
    # away.
    tgt = _zone_pt(zone, puck, zname, avoid, keep)
    heading = float(np.degrees(np.arctan2(tgt[1] - puck[1],
                                          tgt[0] - puck[0])))
    app = nav.capture_approach(cm, puck, R2.BODY_PTS, prefer=heading)
    if app is None:
        return None
    # CAN IT GET OUT AGAIN?  A capture pose is not a delivery: the robot
    # ends the capture nose-first at the patient, and if the patient stands
    # 80 mm off a wall the nose is 27 mm off it too, with no forward arc
    # available and no reverse allowed while the pocket is loaded.  Six of
    # twelve rig deliveries died in exactly that corner, at 21-31 s apiece,
    # and the schedule cannot afford to discover it by driving there.  So
    # the turn-out is part of the price: no clear arc, no delivery.
    conf = turn_out_margin(cm, _capture_pose(app, puck), app[2], heading)
    if conf <= 0.0:
        return None
    _, s1 = nav.plan(cm, robot, app[:2], R2_INSCRIBED, R2_CIRCUM,
                     t0=t0, speed=APPROACH_V, strict=False)
    if not np.isfinite(s1):
        return None
    _, s2 = nav.plan(cm, _capture_pose(app, puck), tgt,
                     R2_INSCRIBED + CARRY_PAD, R2_CIRCUM + CARRY_PAD,
                     t0=t0 + s1 + 3.0, speed=CARRY_V, strict=False)
    if not np.isfinite(s2):        # quote the tight route rather than refuse
        _, s2 = nav.plan(cm, _capture_pose(app, puck), tgt, R2_INSCRIBED,
                         R2_CIRCUM, t0=t0 + s1 + 3.0, speed=CARRY_V,
                         strict=False)
    if not np.isfinite(s2):
        return None
    #        approach   turn+capture   carry   release+back off
    # ...divided by how likely the leaving is (F147): expected seconds per
    # point, not best-case seconds per point.
    return float((s1 + 4.0 + s2 + 3.0) / conf), app


# THE STOPWATCH SAYS TWICE (F134).  _price quotes path-length over speed for
# the two legs plus fixed allowances for the turn, the capture and the
# release.  Measured on the isolating rig, every one of seven deliveries took
# about double the quote:
#
#     quoted  13.2 13.3 11.1 10.1 16.8 14.6 13.9
#     took    26.0 27.5 18.6 18.4 39.1 35.1 23.9      mean ratio 2.0
#
# A path length is not a drive.  The tracker corners inside its own
# tolerance, replans when it drifts off, and pays for every acceleration; on
# top of that a delivery begins wherever the LAST one ended, so the quote's
# first leg is measured from a pose the robot has yet to reach.  None of
# that is a bug to remove -- it is what driving costs -- but a scheduler
# betting on the quoted number buys twelve deliveries out of a budget that
# affords five, and then discovers it at the buzzer.
#
# So the quote is scaled: by the measured mean at the gun, and by this
# robot's own stopwatch once the match has provided one.  Successes only --
# a failed attempt's duration is mostly timeout, and would teach the wrong
# lesson.  This is what planner.observe() does for robot 1, and it is the
# same argument.
PACE0 = 2.0                # measured actual/quoted, seven deliveries
PACE_GAIN = 0.3            # how fast the stopwatch overrides the prior
PACE_CLAMP = (1.2, 3.5)


def _wrong_zone(x, y, want):
    """The name of a destination zone this point is in that is NOT the one
    we are delivering to, or None."""
    for zn, box in ZONES.items():
        if box is want:
            continue
        if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
            return zn
    return None


def _deliver(ctl, i, live, target, app, log, t, zone=None, what="patient"):
    """One patient, end to end: stand off, face it, take it, carry it, let
    go.  Returns True only if the RESULT is what was wanted -- inside the
    zone for a delivery, or actually moved for a clearance.  Nothing here
    assumes; the camera checks after every stage, because a capture that
    silently failed used to be carried all the way to a zone and released
    into thin air."""
    j0 = ctl.jams
    p0 = (ctl.pose[0], ctl.pose[1])
    ok = yield from ctl.goto(app[0], app[1], v_max=APPROACH_V, tol=70.0,
                             tries=3)
    gap = float(np.hypot(ctl.pose[0] - app[0], ctl.pose[1] - app[1]))
    if gap > 190.0:
        # say WHY, not just that: blocked means the planner refused, jams
        # mean the tracker was stopped by something, and neither means the
        # same thing to whoever reads this next
        moved = float(np.hypot(ctl.pose[0] - p0[0], ctl.pose[1] - p0[1]))
        log(t() + "  %s %d: never reached the stand-off (%.0f mm short, "
            "moved %.0f, %d jams%s)"
            % (what, i, gap, moved, ctl.jams - j0,
               ", no path" if ctl.blocked and ctl.jams == j0 else ""))
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
    # come round onto the carry bearing on an ARC before asking the
    # tracker for anything (F113): pure pursuit answers a 150-degree
    # opening error with a stand-and-turn, and a stand-and-turn empties
    # the pocket.
    yield from ctl.carry_turn(float(np.degrees(np.arctan2(ty - ctl.pose[1],
                                                          tx - ctl.pose[0]))))
    if not ctl.holding(*live(i)):
        log(t() + "  %s %d: lost it coming round" % (what, i))
        return False
    yield from ctl.goto(tx - ux * R2.CAPTURE_X, ty - uy * R2.CAPTURE_X,
                        v_max=CARRY_V, w_max=CARRY_W, tol=55.0, tries=3,
                        carry=True)
    # NEVER LET GO INSIDE THE WRONG ZONE (F131).  A carry that falls short
    # is only worth -3, the same as a patient nobody touched -- but a carry
    # that falls short ON TOP OF another destination zone is -5, and the
    # difference is free to avoid: back out along the way we came and drop
    # it on plain floor instead.  Measured on seed 6, one of two patients
    # robot 2 got hold of finished in a zone it did not belong in.
    if zone is not None:
        for _ in range(4):
            hx, hy = ctl.pose[0] + ux*R2.CAPTURE_X, ctl.pose[1] + uy*R2.CAPTURE_X
            wrong = _wrong_zone(hx, hy, zone)
            if wrong is None:
                break
            log(t() + "  %s %d: short, and over %s -- backing off before "
                "the release" % (what, i, wrong))
            yield from ctl.goto(ctl.pose[0] - ux*220.0, ctl.pose[1] - uy*220.0,
                                v_max=CARRY_V, w_max=CARRY_W, tol=70.0,
                                tries=1, carry=True)
    yield from ctl.release(150.0)
    fx, fy = live(i)
    if zone is None:
        good = np.hypot(fx - p0[0], fy - p0[1]) > 120.0
    else:
        box = TAPE.get(ZONE_NAME.get(zone), zone)     # the line, not the box
        good = box[0] <= fx <= box[2] and box[1] <= fy <= box[3]
    log(t() + "  %s %d: %s at (%.0f, %.0f)"
        % (what, i, "done" if good else "short", fx, fy))
    return bool(good)


def _staging(ctl, pucks, live, placed, spent, flt):
    """Where to stand while waiting: near the next job, out of robot 1's way.

    A robot that has nothing to do RIGHT NOW still has somewhere better to
    be than where it is.  This picks the stand-off of the patient it would
    most like next -- so the wait is spent shortening the next approach --
    and falls back to the north-east dead corner, which is the one part of
    the board robot 1's route never visits.
    """
    best, bestd = None, 1e18
    for i, _, _, c in pucks:
        if i in placed or i in spent:
            continue
        p = live(i)
        zn = DEST.get(c) if c != "yellow" else \
            ("PCC_R" if p[0] > 570.0 else "PCC_L")
        if zn not in R2_ZONES:
            continue
        d = float(np.hypot(p[0]-ctl.pose[0], p[1]-ctl.pose[1]))
        if d < bestd:
            bestd, best = d, p
    if best is None:
        return HOLD
    # 260 mm back from it along the line to the field centre: outside the
    # patient block, inside the approach.
    ux, uy, _n = _norm(571.0 - best[0], 620.0 - best[1])
    x = float(np.clip(best[0] + ux * 260.0, 110.0, 1030.0))
    y = float(np.clip(best[1] + uy * 260.0, 110.0, 1070.0))
    if flt is not None:
        for reg in fleetmod.region_of(x, y, pad=R2_CIRCUM):
            if flt.owner(reg) not in (None, "r2"):
                return HOLD
    return (x, y)


def _work_patients(ctl, pucks, live, log, t, now, deadline=112.0,
                   sched=None, flt=None):
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
    pace = [PACE0]                              # F134, corrected as we go
    idling = False
    yields = 0
    _flt = flt
    _sc = sched if callable(sched) else (lambda: sched)

    def wanted(i, c, p):
        """The zones robot 2 may deliver this patient to right now.

        A YELLOW HAS TWO HOMES AND THE CHOICE IS WORTH 8 POINTS (F135).  The
        referee pays that bonus for yellows split EXACTLY two and two
        between the PCCs, and this used to send each one to whichever PCC
        was nearer in x -- so a board that deals three yellows east sends
        all three east and the bonus is gone before the first is carried.
        Offering both and letting _marginal price them is all it takes: the
        referee already knows about the split, and the fourth yellow's
        marginal value tells the truth about which side needs it.  Distance
        still decides when the points are equal, because the rate does.
        """
        names = ["PCC_R", "PCC_L"] if c == "yellow" else [DEST.get(c)]
        if c == "yellow" and p[0] <= 570.0:
            names.reverse()                     # nearer one first, for ties
        return [ZONES[z] for z in names
                if z and zone_open(z, now(), _sc(), _flt)]

    while now() < deadline:
        # THE EXECUTIVE SPEAKS FIRST.  Robot 2 is a detached actuator: when
        # robot 1 wants a region robot 2 is standing in, robot 2 leaves.  It
        # leaves toward its own next job, so the yield costs the fleet only
        # the transit it was going to make anyway.
        if _flt is not None:
            leave = _flt.must_leave("r2")
            if leave is not None and yields < 6:
                other = _flt.agents.get("r1")
                aim = _staging(ctl, pucks, live, placed, spent, _flt)
                out = fleetmod.escape_from(
                    leave, ctl.pose, radius=R2_CIRCUM,
                    away_from=(other.pose[:2] if other and other.pose else None),
                    toward=aim)
                log(t() + "robot 1 wants %s -- yielding to (%.0f, %.0f)"
                    % (leave, out[0], out[1]) if out else
                    t() + "robot 1 wants %s -- nowhere to yield" % leave)
                if out is not None:
                    ctl.cmap = _board_map(
                        [(i, *live(i), c) for i, _, _, c in pucks],
                        sched=_sc(), t_now=now(), flt=_flt)
                    yield from ctl.goto(out[0], out[1], v_max=320.0, tol=70.0,
                                        tries=2)
                # ALWAYS BURN A TICK HERE.  When there is nowhere to yield to
                # this branch used to `continue` without yielding at all --
                # a generator that never yields is a hang, and two of twelve
                # seeds sat in it burning eleven minutes of CPU apiece.  The
                # counter is the second belt: an agent that cannot get out of
                # the way six times running is not going to, and the right
                # answer then is to carry on with its own work and let the
                # reactive veto keep it out of trouble.
                yields += 1
                for _ in range(int(0.4 * hal.Clock.HZ)):
                    ctl.tick()
                    yield
                _flt.vacated("r2")
                idling = False
                continue
        board = _board_now(pucks, live, placed)
        full = [(i, *live(i), c) for i, _, _, c in pucks]
        if _flt is not None:            # the shared board, kept live
            _flt.see([(i, *live(i), "patient") for i, _, _, _ in pucks])
        best = None
        for i, _, _, c in pucks:
            if i in placed or i in spent:
                continue
            p = live(i)
            zones = wanted(i, c, p)
            if not zones:
                continue                   # robot 1 owns those rectangles
            if any(z[0] <= p[0] <= z[2] and z[1] <= p[1] <= z[3]
                   for z in zones):
                placed[i] = p              # already home
                continue
            cmi = None
            for zone in zones:
                gain = _marginal(board, i, zone)
                if gain <= 0.0:
                    continue
                if cmi is None:            # one map serves both zones
                    cmi = _board_map(full, skip=i, sched=_sc(),
                                     t_now=now(), flt=_flt)
                zn_ = ZONE_NAME.get(zone)
                here = _placed_in(placed, zone)
                keep = [] if _flt is None else _flt.floor_of("r1", zn_)
                pr = _price(cmi, ctl.pose[:2], p, zone, now(), zn_, here, keep)
                if pr is None:
                    continue
                raw, app = pr
                secs = raw * pace[0]
                if now() + secs > deadline + 6.0:
                    continue
                rate = gain / max(secs, 1.0)
                if best is None or rate > best[0]:
                    best = (rate, i, gain, secs, app, zone, cmi,
                            _zone_pt(zone, p, zn_, here, keep),
                            "patient", raw)

        if best is None:
            # ---- nothing deliverable: is something merely IN THE WAY? ----
            for i, _, _, c in pucks:
                if i in placed or i in spent:
                    continue
                zs = wanted(i, c, live(i))
                if not zs:
                    continue
                zone = zs[0]
                j = _blocker(pucks, live, i, zone, ctl.pose[:2], now(),
                             sched=_sc(), flt=_flt)
                if j is None or j in spent:
                    continue
                cmj = _board_map(full, skip=j, sched=_sc(), t_now=now(), flt=_flt)
                sp = _spoil_point(cmj, pucks, live, live(j), {i, j})
                if sp is None:
                    continue
                app = nav.capture_approach(
                    cmj, live(j), R2.BODY_PTS,
                    prefer=float(np.degrees(np.arctan2(sp[1]-live(j)[1],
                                                       sp[0]-live(j)[0]))))
                if app is None:
                    continue
                _, s1 = nav.plan(cmj, ctl.pose[:2], app[:2], R2_INSCRIBED,
                                 R2_CIRCUM, t0=now(), speed=APPROACH_V,
                                 strict=False)
                if not np.isfinite(s1):
                    continue
                log(t() + "patient %d is blocked by %d -- clearing it to "
                    "(%.0f, %.0f)" % (i, j, sp[0], sp[1]))
                best = (0.0, j, 0.0, s1 + 12.0, app, None, cmj, sp, "blocker",
                        s1 + 12.0)
                break

        if best is None:
            # NOTHING DELIVERABLE **YET** IS NOT NOTHING TO DO (F124).
            # Robot 2's one destination zone sits inside robot 1's
            # laboratory and kit corridors from T+24 to T+81, so the honest
            # answer at T+45 is "not now", not "never" -- and the old code
            # said never and parked for the remaining seventy seconds.
            # Measured on seed 1: fifty-eight seconds of a hundred and
            # twenty, motionless, while robot 1 worked round it.
            #
            # So: stand somewhere USEFUL and ask again.  Useful means out of
            # every region robot 1 holds, and as close as that allows to the
            # patient this robot most wants next -- the wait becomes the
            # first half of the next approach instead of dead time.
            if not idling:
                log(t() + "nothing deliverable yet (%d placed, %d spent) -- "
                    "staging" % (len(placed), len(spent)))
                idling = True
                stage = _staging(ctl, pucks, live, placed, spent, _flt)
                if stage is not None:
                    ctl.cmap = _board_map(full, sched=_sc(), t_now=now(),
                                          flt=_flt)
                    yield from ctl.goto(stage[0], stage[1], v_max=300.0,
                                        tol=80.0, tries=2)
                ctl.stop()
            for _ in range(int(1.5 * hal.Clock.HZ)):
                ctl.tick()
                yield
            continue
        idling = False
        yields = 0
        _, i, gain, secs, app, zone, cm, tgt, what, raw = best
        t_go = now()
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
            if raw > 3.0:                       # F134: correct the quote
                r = (now() - t_go) / raw
                pace[0] = float(np.clip(
                    (1.0 - PACE_GAIN) * pace[0] + PACE_GAIN * r, *PACE_CLAMP))
        else:
            spent.add(i)               # one honest attempt each; the clock
                                       # is worth more than a second try


def mission_robot2(ctl, m, d=None, log=print, clock=None, rb=None,
                   flt=None):
    """Survey, plan, execute, repair.  One yield per 50 Hz tick.

    flt is the fleet executive (rfgyc26.fleet): robot 2 tells it where it
    is on every tick, plans around robot 1's live footprint, and gets out of
    the way when it is told to.  rb is robot 1, when the fleet is running
    one: robot 2's controller
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

    def publish():
        """Tell the fleet where the pieces are, so ROBOT 1 can plan too."""
        if flt is not None:
            flt.see([(i, *live(i), "patient") for i, _, _, _ in pucks])

    ctl.fleet = flt
    publish()

    def sched():
        return getattr(rb, "schedule", None) if rb is not None else None

    def yielding():
        """Has the executive told us to get out of somewhere?"""
        return None if flt is None else flt.must_leave("r2")

    # ---- the kits: PCC_R, from the east-box spawn -----------------------
    # AND CHECK THEY LANDED (F115).  This drop is worth 36 points to the
    # fleet -- it is what turns robot 1's PCC_L delivery from a 16-point
    # task into the 6/2/2 bonus -- and in isolation it works on ten of
    # twelve seeds.  In a match it was failing silently: the kits stayed on
    # the tray through the shake and then dribbled out over the next
    # twenty-five seconds while robot 2 drove away, landing in a trail from
    # (747, 717) to (1046, 1001).  Nothing noticed, because nothing looked.
    # The camera can see two white boxes in a zone; so look, and shake
    # again from a re-squared pose if they are not there.
    # STAND IN THE MIDDLE AND CHECK BEFORE THROWING.  The shake ejects off
    # the tail, about 95 mm behind the axle, and PCC_R is 200 mm square --
    # so from the zone's CENTRE the kits land inside it whatever the
    # heading, and the only thing that matters is actually being there.
    # Arriving on a 45 mm tolerance and shaking anyway put them 5-40 mm
    # outside the line often enough to cost the 6/2/2 bonus on seven seeds
    # of twelve.  Each stage is checked against the estimate before the
    # next one is allowed to matter.
    zx, zy = (Field.PCC_R[0] + Field.PCC_R[2]) / 2.0, \
             (Field.PCC_R[1] + Field.PCC_R[3]) / 2.0
    for attempt in range(4):
        ctl.cmap = _board_map(pucks, sched=sched(), t_now=now(), flt=flt)
        # AIM THE EJECTION POINT, NOT THE AXLE (F144).  The shake walks the
        # kits over the TAIL lip, measured 71 mm behind the axle, and facing
        # south that puts them 71 mm NORTH of wherever the chassis stands.
        # From the zone's centre they land 29 mm short of the north line
        # with nothing in hand for arrival error; standing 71 mm south of it
        # instead puts them on the middle of the zone, a hundred either way.
        #
        # And carry them at a carrying speed.  This leg used to ask for
        # 360 mm/s, which was harmless while a path-global clearance limit
        # held the robot to 110 (F138) and is not now that the limit is
        # local.  A kit on an open tray has a lateral acceleration budget
        # like any other cargo.
        ax, ay = zx, zy - EJECT_BACK
        for _ in range(2):
            yield from ctl.goto(ax, ay, v_max=KIT_V, tol=30.0, tries=3)
            if np.hypot(ctl.pose[0] - ax, ctl.pose[1] - ay) < 55.0:
                break
        for _ in range(2):
            yield from ctl.face(270.0, tol=6.0)
            if abs(_wrap(ctl.th - 270.0)) < 14.0:
                break
        log(t() + "SHAKE: kits into PCC_R (attempt %d, at %.0f,%.0f hdg %.0f)"
            % (attempt + 1, ctl.pose[0], ctl.pose[1], ctl.pose[2]))
        yield from ctl.shake_out(4)
        n_in = kits_home(m, d)
        if n_in >= 2:
            log(t() + "  both kits are in PCC_R")
            break
        log(t() + "  only %d kit(s) landed -- going round again" % n_in)
        yield from ctl.back_off(150.0)
    yield from ctl.goto(1040.0, 900.0, v_max=340.0, tol=50.0)

    # ---- the patients: go there, GRAB it, carry it, let go --------------
    yield from _work_patients(ctl, pucks, live, log, t, now,
                              sched=sched, flt=flt)

    # ---- park in the dead corner ----------------------------------------
    log(t() + "parking")
    ctl.cmap = _board_map(refresh(), sched=sched(), t_now=now(), flt=flt)
    yield from ctl.goto(1085.0, 880.0, v_max=340.0, tol=60.0)
    ctl.stop()
    while True:
        ctl.tick()
        yield
