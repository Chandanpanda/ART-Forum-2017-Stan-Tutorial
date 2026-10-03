"""The factory iPhone test's instruments besides the phone's Bluetooth
(board.hal.RadioHAL): station 4's gimbal and the phone's camera, as the
test drives them.  Production: abstract contracts only; phone/sim.py holds
the simulated backends, and the real ones come with the hardware.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Frame:
    """One frame's band finds: its time on the phone's clock (the middle of
    the first row's exposure, s) and each band's centroid (2, 2) px, rows
    in the design's band order; None where a band was not found."""
    t: float
    uv: Optional[np.ndarray]


class GimbalHAL(ABC):
    """Station 4's two-axis gimbal (imu/plan.py drives the same one)."""

    @abstractmethod
    def run(self, path):
        """Start an imu.kin.Path now, from the angles it starts at, and
        return at once; the gimbal follows it to its end and holds there."""

    @abstractmethod
    def encoders(self):
        """(2,) rad: both hinges' encoders now, whole codes."""


class CameraHAL(ABC):
    """The factory iPhone's camera running the app's band finder."""

    @abstractmethod
    def start(self):
        """Begin filming at Phone.FPS."""

    @abstractmethod
    def stop(self):
        """Stop, and return [Frame] since start()."""
