"""The hardware abstraction layer -- the ONE file both machines agree on.

Everything above this interface (perception, estimation, planning, control,
the mission itself) is the same source file in MuJoCo and on the Raspberry
Pi 5.  Everything below it exists twice: robot.py implements it against the
simulator, pi_hal.py (bring-up, step 7) implements it against the aluminium.
This module therefore imports NOTHING but the standard library -- no mujoco,
no numpy -- and defines no behaviour, only contracts.

Units, everywhere, no exceptions: millimetres, degrees, seconds, and the
robot's own frame is +x out the nose, +y out the LEFT flank, heading CCW
positive with 0 along world +x.  A backend that cannot honour a call must
raise, not lie.

What is deliberately NOT here:

  * pose.  There is no oracle on the real field.  Position comes from the
    state estimator (step 3), which consumes odometry() and camera fixes
    through this interface like any other client.  The simulator's ground
    truth (AgentARobot.pose, .d) remains available to the referee and the
    check suites -- and, until step 3 lands, to route.py, whose remaining
    direct reads are pinned by check_hal.py so they can shrink but not grow.
  * velocity.  Same reason: stall/settle tests are phrased on odometry
    increments, which both backends can produce honestly.
"""
from abc import ABC, abstractmethod


class DriveHAL(ABC):
    """The differential drive: two steppers through TMC2209 drivers."""

    @abstractmethod
    def drive(self, v_mm_s, omega_deg_s):
        """Command body velocity: v along the nose, omega CCW positive.

        The backend converts to wheel rates and quantises to whole full
        steps per second -- the command actually sent is the quantised one,
        on both machines.  Takes effect from this control tick until the
        next drive()/stop().
        """

    @abstractmethod
    def stop(self):
        """Both wheels to zero, immediately.  Equivalent to drive(0, 0)."""

    @abstractmethod
    def odometry(self):
        """(dL_mm, dR_mm): signed wheel travel since the previous call.

        COMMANDED-step integration on both backends -- the TMC2209 does not
        report actual rotation, it counts the steps it was told to make, and
        the simulator books exactly the same thing (including any injected
        step loss).  Below stall a stepper does not slip, so this is exact
        in straight running; in a pivot the wheels scrub and commanded
        travel over-reads the true arc -- that mismatch is the estimator's
        process noise (design doc paragraph 6), not an error in this call.
        Never blocks; resets its own accumulator on read.
        """

    @abstractmethod
    def stalled(self, thresh=0.42):
        """Both drivers at torque saturation -- TMC2209 StallGuard, or the
        simulator's actuator force.  Unreliable below ~0.1 m/s on both
        machines; callers gate it on distance covered (see stall_drive)."""

    @abstractmethod
    def gyro_z(self):
        """Yaw rate, deg/s, CCW positive -- the IMU on the deck (the model's
        A_imu gyro; an MPU6050-class part on the real robot).  Wheels scrub
        in a pivot (efficiency 0.71 measured), so heading integrates THIS,
        never the wheels.  The backend owns the sensor's imperfection: a
        per-match bias plus per-read noise, drawn like every calibration
        error in this project -- once, not averaged away."""


class DeviceHAL(ABC):
    """Every mechanism that is not a wheel.  One verb per mechanism.

    All of these are fire-and-forget position commands to servos or small
    DC motors; the three that have feedback say so.
    """

    @abstractmethod
    def intake(self, collecting, rpm=None):
        """Knife shim down + brush roller at `rpm` (default ROLL_RPM), or
        roller stopped + knife lifted.  On hardware the roller stops BEFORE
        the knife lifts and the knife drops before the roller starts --
        spinning fingers strike a lifted knife tip."""

    @abstractmethod
    def fingers(self, opened=True):
        """Guide fingers to the mouth width (opened) or raked to the belt."""

    @abstractmethod
    def feed(self, down):
        """Positive-feed plunger: one stroke presses the column onto the
        stack and re-seats anything perched."""

    @abstractmethod
    def gate(self, opened):
        """Escapement shelf: two leaves carrying the column, retracting to
        opposite sides to release exactly one piece."""

    @abstractmethod
    def blade(self, inserted):
        """Escapement retainer: two 1 mm knives taking the column at the
        joint above the bottom disc."""

    @abstractmethod
    def cradle(self, which, carry):
        """Beam cradle 1 or 2: carry=True holds the beam CARRY_Z clear of
        the floor, carry=False sets it down and frees it."""

    @abstractmethod
    def cradle_down(self, which):
        """Limit switch: has cradle `which` finished its down-stroke?"""

    @abstractmethod
    def trim(self, y_mm):
        """Aim the posting head: +y is the robot's LEFT, clipped to TRIM_Y."""

    @abstractmethod
    def trim_at(self):
        """Where the trim slide actually is (mm) -- the servo's feedback."""

    @abstractmethod
    def trim_settled(self, tol=0.25):
        """Is the slide within tol of its command?"""

    @abstractmethod
    def open_hopper(self, dest):
        """Open one destination's kit flap ("HOSP" | "PCC_L" | "PCC_R").
        The only verb Mission 2 needs: kits start aboard, grouped.  Returns
        how many kits were released."""

    @abstractmethod
    def mag_count(self):
        """Pieces in the magazine, from the bore rangefinder.  Over- and
        under-reads transiently while a piece crosses the beam (F60);
        callers demand a HELD count before trusting it."""


class StereoCalib:
    """The calibration that rides with a camera pair -- one object, one
    file format (OpenCV YAML), produced by the same chessboard flow on the
    bench and by construction in the simulator.

    fx, fy, cx, cy   per-eye pinhole intrinsics, pixels
    dist             per-eye distortion (k1 k2 p1 p2 k3), zero in sim
    R, T             each eye's rotation (3x3, row-major tuple) and
                     position (mm) in the ROBOT frame
    size             (width, height) pixels
    """
    def __init__(self, size, left, right):
        self.size, self.left, self.right = size, left, right


class CameraHAL(ABC):
    """The stereo pair on the tail mast.  Implemented in step 2:
    mujoco.Renderer offscreen frames in sim, picamera2 captures on the Pi.
    Until then the synthetic geometric model (AgentARobot.see_lab) stands in
    for the whole camera+perception stack as its test double."""

    @abstractmethod
    def frames(self):
        """(imgL, imgR, t_s): BGR uint8 arrays of calib().size, and the
        capture time on the Clock's own axis.  Blocks at most one frame."""

    @abstractmethod
    def calib(self):
        """The StereoCalib for this pair.  Constant for the match."""


class LinkHAL(ABC):
    """The Bluetooth link to robot 2 -- which is not a peer, it is a
    DETACHED ACTUATOR of this robot (decided): two DC motors, two drivers,
    a small battery, a Pico W, a plow, no sensors robot 1 relies on.  All
    perception and planning for it happen here; the wire carries dumb verbs
    and the firmware's only autonomy is the dead-man stop.

    Wire protocol v0, one ASCII line per command, newline-terminated,
    trailing "*XX" hex checksum of everything before it:

        V <left_mm_s> <right_mm_s> <ms>   run wheels for a duration
        K                                 keepalive / stop now
        SHAKE <n>                         n cycles of the kit-shake manoeuvre
                                          (the escapement robot 2 doesn't have
                                          a servo for -- perfected in sim)

    The firmware stops both motors 250 ms after the last valid line, so a
    dropped link parks the robot instead of driving it into a wall.  The
    same grammar drives the simulated robot 2 body (step 7) so the shake
    and the plow legs are tuned against the same parser."""

    @abstractmethod
    def send(self, line):
        """Queue one protocol line (str, no newline).  Non-blocking."""

    @abstractmethod
    def recv(self):
        """Next line from robot 2 ("A <seq>" acks, battery reports), or
        None.  Non-blocking."""

    # The grammar, encoded once -- both backends inherit these.
    def cmd(self, left_mm_s, right_mm_s, ms):
        self.send("V %d %d %d" % (round(left_mm_s), round(right_mm_s), round(ms)))

    def halt(self):
        self.send("K")

    def shake(self, n=3):
        self.send("SHAKE %d" % n)


class NullLink(LinkHAL):
    """No robot 2 attached (every match until step 7).  Swallows sends."""
    def send(self, line):
        pass

    def recv(self):
        return None


class Clock(ABC):
    """Match time and the 50 Hz control tick.

    Mission code is written as generators: every `yield` is one control
    period.  The scheduler advances them by calling tick() -- which steps
    the physics CTRL_DECIM substeps in the simulator, and sleeps to the
    next 20 ms wall boundary on the Pi.  Nothing above the HAL knows which."""

    HZ = 50.0
    PERIOD = 1.0 / HZ

    @abstractmethod
    def now(self):
        """Seconds on the match clock.  Starts near zero, never rewinds."""

    @abstractmethod
    def tick(self):
        """Advance exactly one control period."""

    def run(self, gen, seconds=None):
        """Drive a mission generator: next(gen) then tick(), until it stops
        or `seconds` of clock have elapsed.  Returns True if it finished."""
        t0 = self.now()
        for _ in gen:
            self.tick()
            if seconds is not None and self.now() - t0 >= seconds:
                return False
        return True


def audit(rb, contracts=(DriveHAL, DeviceHAL)):
    """Every abstract method of `contracts` that `rb` is missing.  Empty
    list == conformant.  ABC registration already enforces this for
    subclasses; this catches duck-typed backends and rigs too."""
    missing = []
    for c in contracts:
        for name in sorted(getattr(c, "__abstractmethods__", ())):
            if not callable(getattr(rb, name, None)):
                missing.append("%s.%s" % (c.__name__, name))
    return missing
