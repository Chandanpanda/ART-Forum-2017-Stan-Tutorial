"""Agent A hardware abstraction: drive, servos, sensors.

The drive is a stepper model -- commanded wheel speed is quantised to whole
full-steps per second, and step loss can be injected, because the spec calls a
skipped step 'silent' and that is the design's main odometry risk.
"""
import numpy as np, mujoco
from .params import Chassis, AgentA, mm

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
        self.a_gate= gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "a_gate")
        self.s_tof = gid(mujoco.mjtObj.mjOBJ_SENSOR, "a_tof")
        self.s_gyro= gid(mujoco.mjtObj.mjOBJ_SENSOR, "a_gyro")
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
        a = np.radians(-5.4 if opened else 10.2)
        self.d.ctrl[self.a_fl] = a
        self.d.ctrl[self.a_fr] = -a

    def gate(self, opened):
        self.d.ctrl[self.a_gate] = 0.070 if opened else 0.0

    # ------------------------------------------------------------ stallguard
    def stalled(self, thresh=0.42):
        """TMC2209 StallGuard stand-in: both drivers at torque saturation.
        Unreliable below ~0.1 m/s, exactly as the real part is."""
        return (abs(self.d.actuator_force[self.a_l]) > thresh and
                abs(self.d.actuator_force[self.a_r]) > thresh)
