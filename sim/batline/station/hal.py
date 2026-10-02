"""The station module's hardware abstraction layer: the one file the
simulator and the controller agree on, as truss/hal.py is for the cell.

Everything above it -- the executor, commissioning, the jobs -- is the same
source in MuJoCo and on the machine.  Below it, sim.py implements it
against MuJoCo and a firmware bridge would implement it against the
drivers.  This module defines contracts and no behaviour.

Units: millimetres, degrees, seconds, newtons.  Coordinates are the HEAD
POINT's (the tool plate's underside, centre) in the station frame, as the
controller BELIEVES them: whole steps since the home switch tripped.  What
the axes really did is not here, and neither is any part's pose: the
process knows where things are only by looking (VisionHAL) and touching
(ForceHAL).  A backend that cannot honour a call raises; it does not lie.
"""
from abc import ABC, abstractmethod

from truss.hal import Clock, CameraHAL, audit as _audit   # noqa: F401  (the same tick, the same camera)


class AxesHAL(ABC):
    """x, y, z: three steppers with an encoder on each shaft."""

    @abstractmethod
    def goto(self, axis, value, fine=False):
        """Setpoint for this control tick, in believed mm.  The backend
        rounds it to a whole step, or with fine to a microstep (a touch's
        creep); the profile is the caller's (truss motion.Move), as a step
        generator on the machine would receive it."""

    @abstractmethod
    def at(self, axis):
        """The last setpoint, as rounded: where the axis believes it is."""

    @abstractmethod
    def encoder(self, axis):
        """Where the motor's shaft encoder puts the axis, believed mm.  It
        lags at() while moving; after a move it differs from at() only if
        the motor dropped steps -- which it then does by whole electrical
        cycles, four full steps at a time."""

    @abstractmethod
    def settled(self, axis, tol=0.01):
        """The encoder within tol of the setpoint and not moving.  The
        carriage beyond the drive's give may still be ringing: the encoder
        cannot see it, and nor can this."""

    @abstractmethod
    def limits(self):
        """{axis: (lo, hi)} in believed mm."""

    @abstractmethod
    def home(self, axis):
        """Start homing from wherever the axis is: seek the switch, back
        off it, creep onto it, and make the creep's trip point the axis'
        home coordinate.  Non-blocking; poll homed()."""

    @abstractmethod
    def homed(self, axis):
        pass


class ToolsHAL(ABC):
    """Each tool on its own two-position air slide, with reed switches at
    both ends."""

    @abstractmethod
    def extend(self, kind):
        pass

    @abstractmethod
    def retract(self, kind):
        pass

    @abstractmethod
    def extended(self, kind):
        pass

    @abstractmethod
    def retracted(self, kind):
        pass


class HoldHAL(ABC):
    """The tools that take hold of a part: the gripper's jaws, the vacuum
    cup, the electro-permanent magnet, the spindle's collet."""

    @abstractmethod
    def grip(self, kind):
        """Close the jaws or the collet, open the valve, pulse the magnet on."""

    @abstractmethod
    def release(self, kind):
        pass

    @abstractmethod
    def holding(self, kind):
        """The tool's own sensor: jaws stopped short of closed, the vacuum
        switch, the magnet's hall sensor, the collet's switch."""


class SpindleHAL(ABC):
    @abstractmethod
    def spin(self, rpm):
        """Turn at rpm (signed, positive anticlockwise seen from above);
        0 stops it.  The motor's current limit is its torque limit."""

    @abstractmethod
    def angle(self):
        """Degrees, from the motor's encoder, unwrapped."""

    @abstractmethod
    def stalled(self):
        """At its torque limit and not turning: a cap or nut seated."""


class ProbeHAL(ABC):
    @abstractmethod
    def contacts(self):
        """Per pogo pin, whether it closes a circuit: compressed into its
        working window."""


class ForceHAL(ABC):
    """The 3-axis load cell between the Z carriage and the tool plate."""

    @abstractmethod
    def force(self):
        """(fx, fy, fz) N that the work pushes the tool with, head frame,
        since the last tare; one reading, with its noise."""

    @abstractmethod
    def tare(self):
        """Zero the reading here: call with nothing touching the tool."""


class VisionHAL(ABC):
    """The head camera's measurement: where a dark disc's centre is in the
    image.  Converting pixels to station millimetres is commissioning's
    job, not the camera's."""

    @abstractmethod
    def locate(self, near=None):
        """(u, v) in pixels of the disc nearest pixel `near` (default the
        image centre), or None if none is in view."""


CONTRACTS = (AxesHAL, ToolsHAL, HoldHAL, SpindleHAL, ProbeHAL, ForceHAL)


def audit(backend, contracts=CONTRACTS):
    """Every abstract method of `contracts` the backend is missing."""
    return _audit(backend, contracts)
