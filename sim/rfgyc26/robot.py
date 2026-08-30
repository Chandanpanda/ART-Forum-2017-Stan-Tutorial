"""Agent A hardware abstraction: drive, servos, sensors.

The drive is a stepper model -- commanded wheel speed is quantised to whole
full-steps per second, and step loss can be injected, because the spec calls a
skipped step 'silent' and that is the design's main odometry risk.
"""
import numpy as np, mujoco
from .params import Chassis, AgentA, Piece, mm

WHEEL_R = Chassis.WHEEL_D / 2000.0          # m
HALF_TRACK = Chassis.TRACK / 2000.0         # m
STEP_RAD = 2 * np.pi / Chassis.STEPS_PER_REV


class AgentARobot:
    def __init__(self, model, data, step_loss=0.0, rng=None):
        self.m, self.d = model, data
        self.rng = rng or np.random.default_rng(0)
        self.step_loss = step_loss
        gid = lambda t, n: mujoco.mj_name2id(model, t, n)
        self.bid   = gid(mujoco.mjtObj.mjOBJ_BODY, "agentA")
        self.a_l   = gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "a_drive_l")
        self.a_r   = gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "a_drive_r")
        self.a_fl  = gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "a_finger_l")
        self.a_fr  = gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "a_finger_r")
        self.a_shim  = gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "a_shim")
        self.a_roller= gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "a_roller")
        self.a_gate= gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "a_gate")
        self.a_feed= gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "a_feed")
        self.a_blade = gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "a_blade")
        self.a_cr = [gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "a_cradle1"),
                     gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "a_cradle2")]
        self.eq_beam = [gid(mujoco.mjtObj.mjOBJ_EQUALITY, "beam%d_hold" % i)
                        for i in (1, 2)]
        self.s_tof = gid(mujoco.mjtObj.mjOBJ_SENSOR, "a_tof")
        self.s_gyro= gid(mujoco.mjtObj.mjOBJ_SENSOR, "a_gyro")
        self.s_mag = gid(mujoco.mjtObj.mjOBJ_SENSOR, "a_mag")
        self.s_pl  = gid(mujoco.mjtObj.mjOBJ_SENSOR, "a_probe_l")
        self.s_pr  = gid(mujoco.mjtObj.mjOBJ_SENSOR, "a_probe_r")
        self.odo_steps = np.zeros(2)          # commanded steps, the robot's belief

    # ---------------------------------------------------------------- state
    @property
    def pose(self):
        """(x_mm, y_mm, heading_deg) ground truth."""
        p = self.d.xpos[self.bid]
        q = self.d.qpos[3:7]
        th = np.degrees(np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2)))
        return p[0]*1000.0, p[1]*1000.0, th

    def to_local(self, world_xyz):
        x, y, th = self.pose
        t = np.radians(th)
        dx, dy = world_xyz[0]*1000.0 - x, world_xyz[1]*1000.0 - y
        return (dx*np.cos(-t) - dy*np.sin(-t), dx*np.sin(-t) + dy*np.cos(-t), world_xyz[2]*1000.0)

    def chute_xy(self, offset_mm):
        """World position of the O58 chute axis -- it sits `offset` behind the axle."""
        x, y, th = self.pose
        t = np.radians(th)
        return x - offset_mm*np.cos(t), y - offset_mm*np.sin(t)

    def tof_mm(self):
        v = self.d.sensordata[self.m.sensor_adr[self.s_tof]]
        return 1e9 if v < 0 else v*1000.0

    # ------------------------------------------------------------ actuation
    def drive(self, v_mm_s, omega_deg_s):
        """Differential drive.  Wheel speeds are quantised to whole steps/s."""
        v, w = v_mm_s/1000.0, np.radians(omega_deg_s)
        wl = (v - w*HALF_TRACK) / WHEEL_R
        wr = (v + w*HALF_TRACK) / WHEEL_R
        out = []
        for om in (wl, wr):
            steps = np.round(om / STEP_RAD)                    # whole steps/s
            if self.step_loss and self.rng.random() < self.step_loss:
                steps -= np.sign(steps)                        # a silent skipped step
            out.append(steps * STEP_RAD)
        self.d.ctrl[self.a_l], self.d.ctrl[self.a_r] = out
        self.odo_steps += np.array(out) / STEP_RAD * self.m.opt.timestep

    def stop(self):
        self.d.ctrl[self.a_l] = self.d.ctrl[self.a_r] = 0.0

    def fingers(self, opened=True):
        # RADIANS -- actuator ctrlrange is not converted by compiler angle="degree"
        if self.a_fl < 0:            # fingerless build: ctrl[-1] would hit the cradle
            return
        a = np.radians(AgentA.FINGER_OPEN if opened else AgentA.FINGER_RAKE)
        self.d.ctrl[self.a_fl] = a
        self.d.ctrl[self.a_fr] = -a

    def intake(self, collecting, rpm=None):
        """Knife down + brush roller spinning, or roller stopped + knife up.

        The knife is servo-lifted SHIM_LIFT deg on every non-collecting leg: pressed
        to the floor it bulldozes already-placed discs.  Ordering matters on
        the real machine -- spinning fingers strike a LIFTED knife tip (they
        clear a lowered one by ~2 mm) -- so the roller stops before the knife
        lifts and the knife drops before the roller starts; here both commands
        land the same tick and the model's masks make the order moot.
        """
        if self.a_shim < 0:
            return
        if collecting:
            self.d.ctrl[self.a_shim]   = np.radians(AgentA.SHIM_DROOP)
            self.d.ctrl[self.a_roller] = 2*np.pi*(rpm or AgentA.ROLL_RPM)/60.0
        else:
            self.d.ctrl[self.a_roller] = 0.0
            self.d.ctrl[self.a_shim]   = -np.radians(AgentA.SHIM_LIFT)

    def feed(self, down):
        """Positive-feed plunger.  Parked its face is 33 mm above the highest a
        piece ever reaches, so the drop path stays clear; one stroke presses the
        column down onto the stack and re-seats anything shaken loose."""
        self.d.ctrl[self.a_feed] = -mm(AgentA.FEED_STROKE) if down else 0.0

    def gate(self, opened):
        """Escapement shelf: carries the column, slides clear to release one."""
        self.d.ctrl[self.a_gate] = mm(AgentA.ESC_Y) if opened else 0.0

    def blade(self, inserted):
        """Escapement retainer: a 1 mm knife that takes the column at the joint
        above the bottom disc, so the shelf can slide out from under just that
        one.  Parked it is clear of the bore."""
        self.d.ctrl[self.a_blade] = -mm(AgentA.ESC_BLADE_PARK) if inserted else 0.0

    def cradle(self, which, carry):
        """Beam cradle 1 (pocket R, beam 1) or 2 (pocket L, beam 2).

        carry=True lifts the beam CARRY_Z off the field -- which is how it
        crosses the laboratory and how the robot is allowed to pivot at all
        (F46).  carry=False sets it down on the field; the shelves then sit in
        the floor plane and the robot can simply back away from the piece.
        """
        a = self.a_cr[which - 1]
        if a >= 0:
            self.d.ctrl[a] = 0.0 if carry else mm(AgentA.CARRY_Z + AgentA.CRADLE_DROP)
        # The clamp goes with it: carried means wedged against the pocket wall,
        # released means standing on the field on its own (F50).
        e = self.eq_beam[which - 1]
        if e >= 0:
            self.d.eq_active[e] = 1 if carry else 0

    def cradle_down(self, which, tol=0.6):
        """Has the cradle finished its stroke?  A limit switch reads this."""
        j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "A_cr%d_j" % which)
        full = AgentA.CARRY_Z + AgentA.CRADLE_DROP
        return self.d.qpos[self.m.jnt_qposadr[j]]*1000.0 > full - tol

    def probe_mm(self):
        """Raw slot-probe ranges, mm.  -1 means no return at all."""
        l = self.d.sensordata[self.m.sensor_adr[self.s_pl]]
        r = self.d.sensordata[self.m.sensor_adr[self.s_pr]]
        return (-1.0 if l < 0 else l*1000.0, -1.0 if r < 0 else r*1000.0)

    def over_slot(self, ref_mm, step_mm=0.5):
        """(left, right) -- is each probe looking into a slot?

        `ref_mm` is the range the probe reads over the laboratory SURFACE, which
        the robot learns on the way in rather than assuming: the rulebook gives
        the laboratory no thickness, so the step into a slot is not a number we
        are entitled to know in advance.
        """
        l, r = self.probe_mm()
        return (l < 0 or l > ref_mm + step_mm, r < 0 or r > ref_mm + step_mm)

    def mag_count(self):
        """Pieces in the magazine, from the bore rangefinder.  Empty reads the
        shelf at Za 11; each piece brings the surface up 5 mm."""
        r = self.d.sensordata[self.m.sensor_adr[self.s_mag]]
        if r < 0:                        # no return at all
            return 0
        top = 70.0 - r*1000.0            # site is at Za 70 looking down
        return int(max(0.0, round((top - AgentA.CHUTE_Z0) / Piece.DISC_T)))

    # ------------------------------------------------------------ stallguard
    def stalled(self, thresh=0.42):
        """TMC2209 StallGuard stand-in: both drivers at torque saturation.
        Unreliable below ~0.1 m/s, exactly as the real part is."""
        return (abs(self.d.actuator_force[self.a_l]) > thresh and
                abs(self.d.actuator_force[self.a_r]) > thresh)
