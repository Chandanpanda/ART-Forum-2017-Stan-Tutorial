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

---------------------------------------------------------------------------
RE-RUN, after the winding head's own review, with the cell's BORE added to
the sweep as a fourth constraint (sweep_spine.bore_margin).  Every number
above still holds.  Two things changed around it, and the second is the
one to read:

  * THIS DESIGN NOW MISSES f1 BY 1 Hz.  199 against the brief's 200.  That
    is 0.5%, against a modulus that is a typical value and moves f1 by 10%
    per 20% of itself, so it is not a real distinction -- but it is why
    sweep_spine no longer names this design, and it should not be quietly
    rounded away.  The E measurement is what settles it.
  * THE RING'S BORE IS NOW THE BINDING CONSTRAINT ON THE PRODUCT'S
    ACCURACY, and nothing in the tree said so before.  The bore a joint
    needs scales as tan(alpha), so the head limits the WEB ANGLE -- and the
    web angle is what buys accuracy.  Swept over sides 80..180 and webs
    35..55 degrees:

        all four constraints (mass, f1, budget, bore)     0 designs
        give up the BORE (open the ring 1.5 mm)          28 designs
        give up f1 (soft-mount harder)                  124 designs
        give up the budget                                0 designs
        give up mass                                      0 designs

    So the bore is the cheapest thing to give, by a wide margin, and it is
    the only one that gives a design meeting everything else:

        warren 120/45/3.0/1.5   45.7 g   budget 0.88   0.77 m at 100 m
                                f1 200 Hz, and 0.76 mm short of bore

    Ring.ID is 20.0 mm.  ID 21.5 buys that design: 6% truer than this one,
    0.2 g heavier, and f1 met rather than missed.  What it costs is a
    re-check of the head -- EXIT_R sits at 11.0 and would have 0.25 mm of
    clearance over a 10.75 mm bore, which is too little, so the exit guide
    moves too.  THAT IS A DECISION, NOT A DERIVATION, and it is not made
    here: it trades the machine's head against the product's accuracy, and
    both are someone's to weigh.

WHAT IT DOES NOT COST: the winding head, as built.  At 40 degrees this
design fits the bore with +0.45 mm to spare (measured, with Ring.RUN_OUT
charged against it).  A 45 degree web is truer and does not fit -- which
is the whole finding above.
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
