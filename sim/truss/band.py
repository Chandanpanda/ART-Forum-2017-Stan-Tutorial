"""The wound band as arithmetic: what Tier 0 plans with and Tier 1 counts.

A band is what a ring turning N times at pitch p lays on the cluster: N
hoops, each the cluster's hull perimeter where it lies, spread over
N * p of chord.  Everything the inspector asks of a joint -- how many
turns, how wide, centred where, covering which mitre, weighing what -- is
a function of the ring's angle history and the x feed, and that is all
this module holds.  Physics of the thread itself is Tier 2's business.
"""
from dataclasses import dataclass
from math import pi

from .spec import Dispenser


@dataclass
class BandState:
    """One joint's winding, as it progresses."""
    joint:     int
    x_start:   float = 0.0      # chord x where the first hoop was laid
    direction: float = 1.0      # which way along x the feed ran from it
    pitch:     float = 0.0      # mm of x per turn
    angle:     float = 0.0      # ring angle accumulated since the start, deg
    anchored:  bool = False     # this chord's run has a start post
    dosed:     float = 0.0      # mg of resin landed on the band
    thread:    float = 0.0      # mm laid, from the perimeters
    drop_hit:  bool = False     # the drop landed on the band (Tier 1 measures)

    @property
    def turns(self):
        return self.angle / 360.0

    @property
    def width(self):
        return self.turns * self.pitch

    @property
    def centre(self):
        # measured: with the return run's feed running to -x, a centre
        # taken as x_start + width/2 put every one of that chord's bands
        # a full band width off the joint
        return self.x_start + self.direction * self.width / 2.0

    def lay(self, d_angle, perimeter):
        """The ring turned d_angle more; the hoop it laid was `perimeter`."""
        self.angle += abs(d_angle)
        self.thread += abs(d_angle) / 360.0 * perimeter


def pitch_for(truss):
    return truss.band / truss.turns


def band_start(joint_x, truss, direction=+1):
    """Where the first hoop goes so the band is centred on the joint and
    grows in the direction of travel."""
    return joint_x - direction * truss.band / 2.0


def thread_mass_mg(length_mm, truss):
    return length_mm * pi * (truss.thread_d / 2.0) ** 2 * truss.thread_rho


def resin_dose_mg(truss, thread_mm):
    """The joint's mass budget less the thread on it."""
    return max(truss.joint_mass * 1000.0 - thread_mass_mg(thread_mm, truss), 0.0)


def drop_radius_mm(dose_mg):
    """A dose as a sphere of resin."""
    v = dose_mg / 1000.0 / Dispenser.RESIN_RHO      # cm^3
    return (3.0 * v / (4.0 * pi)) ** (1.0 / 3.0) * 10.0
