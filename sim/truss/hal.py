"""The hardware abstraction layer for the cell -- the ONE file both the
simulator and the controller agree on.

Everything above this (the plan, the process generators, vision, the
inspector) is the same source in MuJoCo and on the machine.  Everything
below it exists twice: cell.py implements it against the simulator; the
firmware bridge implements it against the steppers.  This module imports
nothing but the standard library and defines no behaviour, only
contracts.

Units, everywhere: millimetres, degrees, seconds.  The cell frame is the
TRUSS frame: +x along the chord from its start, +z up, y across, origin
on the cage's axis.  A backend that cannot honour a call must raise, not
lie.

What is deliberately NOT here: pose truth.  The axes are steppers and
report what they were told; the ring reports a fiducial; the rods are
where the fixture's notches put them.  The simulator's ground truth
stays available to the inspector and the checks, never to the process.
"""
from abc import ABC, abstractmethod


class AxesHAL(ABC):
    """The gantry and the two tool strokes, all steppers on screws, plus
    the gripper's yaw servo.  Axis names:

        x y z   the carriage; the RING CENTRE is the tool point
        d       dispenser stroke, mm DOWN from park (0 = parked)
        g       gripper stroke, mm DOWN from park
        w       gripper yaw, degrees
    """

    @abstractmethod
    def goto(self, axis, value):
        """Setpoint for this control tick.  The backend quantises to whole
        steps and streams it; a profile is the caller's job (motion.Move),
        exactly as a step generator on the machine would receive it."""

    @abstractmethod
    def at(self, axis):
        """Where the axis BELIEVES it is: the last quantised setpoint.
        Steppers have no encoder; this is bookkeeping, not measurement."""

    @abstractmethod
    def settled(self, axis, tol=0.05):
        """Has the axis stopped within tol of its setpoint?  On hardware a
        step generator answers from its queue; the simulator from the
        joint's own error."""

    @abstractmethod
    def limits(self):
        """{axis: (lo, hi)} the travels, in the axis' own units."""


class RingHAL(ABC):
    """The friction-driven C-ring: spin, park, and read the fiducial."""

    @abstractmethod
    def spin(self, rpm):
        """Run at rpm (signed).  0 stops the drive; the ring coasts."""

    @abstractmethod
    def park(self, az_deg):
        """Close the loop on the fiducial to hold the gap at az_deg
        (0 = straight down, positive toward +y).  Non-blocking; poll
        parked()."""

    @abstractmethod
    def angle(self):
        """Gap azimuth from the fiducial, degrees, unwrapped."""

    @abstractmethod
    def parked(self, tol=None):
        """Within tol (default Ring.STOP_TOL) of the park target and
        stopped."""


class CageHAL(ABC):
    @abstractmethod
    def index(self, theta_deg):
        """Command the worm to theta.  Non-blocking; poll indexed()."""

    @abstractmethod
    def theta(self):
        """Believed cage angle -- the worm's step count."""

    @abstractmethod
    def indexed(self, tol=None):
        pass

    @abstractmethod
    def keep(self, rod_index):
        """Engage the keeper at this rod's station, so the fixture holds
        it through every cage angle (Cage.KEEPER)."""

    @abstractmethod
    def kept(self, rod_index):
        pass


class VisionHAL(ABC):
    """Pre-wind localisation (brief 5): where the joint really is,
    relative to the ring.  A camera measures relative geometry; the
    backend owns the sensor's imperfection -- a calibration bias drawn
    once per run, noise per look."""

    @abstractmethod
    def locate(self, joint_index):
        """(dx, dy): how far the joint's centre and the chord's axis lie
        from the ring's plane and centre, in the cell frame, mm.  Add
        them to the head's x and y to be over the joint.  None if the
        chord is not in view."""


class GripperHAL(ABC):
    @abstractmethod
    def close(self):
        """Close the jaws on whatever is between the pads."""

    @abstractmethod
    def open(self):
        pass

    @abstractmethod
    def holding(self):
        """The jaw switch: closed on something."""


class DispenserHAL(ABC):
    @abstractmethod
    def dose(self, mg):
        """Meter one drop.  Returns when the drop has left the nozzle."""

    @abstractmethod
    def dosed(self):
        """Drops dispensed so far."""


class CutterHAL(ABC):
    @abstractmethod
    def cut(self):
        """Sever the strand between the exit guide and the post."""


class CameraHAL(ABC):
    """The camera on the carriage, looking down past the ring."""

    @abstractmethod
    def frame(self):
        """(img BGR uint8 [H,W,3], t_s)."""

    @abstractmethod
    def calib(self):
        """Pinhole intrinsics and the camera's pose in the HEAD frame
        (mm), as a dict: f, cx, cy, pos, R."""


class Clock(ABC):
    """The control tick.  Process code is generators: every yield is one
    control period.  tick() steps the physics in the simulator and sleeps
    to the boundary on the machine."""

    HZ = 50.0
    PERIOD = 1.0 / HZ

    @abstractmethod
    def now(self):
        pass

    @abstractmethod
    def tick(self):
        pass

    def run(self, gen, seconds=None):
        """Drive a process generator: next(gen) then tick(), until it
        stops or `seconds` elapse.  Returns the generator's return value,
        or None if it was cut off."""
        t0 = self.now()
        while True:
            try:
                next(gen)
            except StopIteration as e:
                return e.value
            self.tick()
            if seconds is not None and self.now() - t0 >= seconds:
                return None


def audit(backend, contracts=(AxesHAL, RingHAL, CageHAL, GripperHAL,
                              DispenserHAL, CutterHAL)):
    """Every abstract method of `contracts` the backend is missing."""
    missing = []
    for c in contracts:
        for name in sorted(getattr(c, "__abstractmethods__", ())):
            if not callable(getattr(backend, name, None)):
                missing.append("%s.%s" % (c.__name__, name))
    return missing
