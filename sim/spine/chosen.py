"""The design the sweep chose, and what it was chosen for.

A RECORD, NOT A CONSTANT.  Every number here is an output of
scripts/spine/sweep_spine.py, and the provenance below says what it was
optimised against and what it cost.  Re-run the sweep when the duty, the
material data or the constraints change, and replace this.

THE CHOICE, on the quadrotor duty, a 1 m baseline and 100 m accuracy:

    a 115 mm chord triangle, 40 degree diagonals, 3 mm chords, 1.5 mm web

against the 92/40/3.0/2.0 the cell was first built for:

    mass              45.5 g   against 54.4 g      16% lighter
    range error       0.82 m   against 1.24 m      34% truer at 100 m
    yaw budget        0.94     against 1.42        inside it, not over
    first mode         199 Hz  against 195 Hz      no worse
    build time        21.6 min against 24.1 min    24 joints, not 27

WHAT IT COSTS, and neither is free:

  * The section grows from 92 to 115 mm, past the 100 mm the brief
    assumes the airframe can swallow.  This is THE trade: the accuracy
    budget cannot be met inside 100 mm at 3 g by any design in the sweep,
    with any stock, in either web.  Someone has to decide whether the
    airframe or the accuracy gives.
  * The 1.5 mm diagonals put their own first mode under the 330 Hz the
    propellers reach.  A diagonal singing at 300 Hz is a local mode; what
    it costs the cameras is in the transmissibility column, not in its
    frequency, and there it is no worse than the design it replaces.

WHAT IT DOES NOT COST: the winding head.  The ring's bore is sized by the
diagonal ANGLE, not by the section, so at 40 degrees this fits the head
already designed.  A 45 degree web would have been slightly truer and
needed 0.7 mm more bore.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Chosen:
    name:    str
    length:  float          # mm
    side:    float
    alpha:   float
    d_chord: float
    d_diag:  float
    web:     str = "warren"

    def as_truss_kwargs(self):
        """The keyword arguments truss.spec.Truss takes, so the cell can
        be built for this design without this package being imported by
        it -- the choosing and the building stay separate."""
        return dict(length=self.length, side=self.side, alpha=self.alpha,
                    d_chord=self.d_chord, d_diag=self.d_diag, name=self.name)


OPTIMAL_1M = Chosen("optimal_1m", 1000.0, 115.0, 40.0, 3.0, 1.5)
# the short test piece, the same sweep at 300 mm: a shorter cantilever
# needs far less section, and the ring's minimum is what binds instead
OPTIMAL_300 = Chosen("optimal_300", 300.0, 66.0, 45.0, 2.0, 1.0)
