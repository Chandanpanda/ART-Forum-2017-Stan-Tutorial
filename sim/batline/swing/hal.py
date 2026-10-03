"""The swing rig as the station drives it.  Production: the abstract
contract only; swing/sim.py holds the simulated backend, and the real one
comes with the hardware.
"""
from abc import ABC, abstractmethod


class SwingRigHAL(ABC):
    """The enclosed swing rig's two servo axes (swing/rig.py)."""

    @abstractmethod
    def run(self, motion):
        """Start a rig.Motion now, from the park pose it starts at, and
        return at once; the rig follows it to its end and holds there."""

    @abstractmethod
    def angles(self):
        """(2,) rad: the arm's and the bat's angles to the room, from the
        motors' encoders."""
