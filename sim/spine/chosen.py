"""The design the sweep chose, and what it was chosen for.

A RECORD, NOT A CONSTANT.  Every number here is an output of
scripts/spine/sweep_spine.py, and the provenance below says what it was
optimised against and what it cost.  Re-run the sweep when the duty, the
material data or the constraints change, and replace this.

THE CHOICE, on the quadrotor duty, a 1 m baseline and 100 m accuracy, with
a Raspberry Pi Camera Module 2 at each end:

    an 85 mm chord triangle, 40 degree diagonals, 3 mm chords, 1.5 mm web

    mass                52.5 g      of a 70 g ceiling
    range error         0.55 m      at 100 m, 0.55% of the range
    yaw budget          0.64        of 0.005 degrees between the faces
    first mode           281 Hz     against the 200 the brief asks
    ring bore          +0.37 mm     the winding head enters every joint
    cell                21.9 min    30 joints; X 1383 of 1400, Y 130 of 150

Of 1482 candidates swept over sections 60..160 mm and webs 30..60 degrees,
in every stock diameter, SEVEN break no rule at all.  One of the seven is
marginally truer -- 90/35/3.0/2.0 at 0.541 m against 0.546 -- and costs 9.3
grams for those five millimetres, which is not a trade worth making; it also
sits at 201 Hz against this design's 281.  The lightest of the seven is
75/35/3.0/1.5 at 51.1 g and 0.66 m.  So the feasible set spans 1.4 g and
0.12 m end to end, and the choice inside it is close to arbitrary.

---------------------------------------------------------------------------
WHAT MOVED, AND WHY THIS REPLACES THE 115 mm ANSWER

The previous record chose 115/40/3.0/1.5 and said the ring's bore was the
binding constraint on the product's accuracy.  Both were artefacts of ONE
number that had never been measured: the camera head was carried at 50 g, a
placeholder from the brief, against a module the vendor's own table weighs
at 3 g.  Everything downstream of a tip mass more than ten times too large
was answering a different question.

With the real head:

  * THE TRUSS IS ITS OWN LOAD.  52.6 g of structure carries 2 x 4 g of
    camera, so six sevenths of the manoeuvre load is the spine's own mass.
    Adding carbon to stiffen it now costs accuracy as often as it buys it
    -- the X web comes back with a HIGHER first mode and a WORSE angular
    error than the single-diagonal one, which at 50 g it did not.
  * THE SECTION NO LONGER BINDS.  The budget is met at 85 mm, inside the
    100 mm the airframe swallows.  The old answer's headline cost -- "the
    accuracy budget cannot be met inside 100 mm by any design in the
    sweep" -- was a consequence of the placeholder, not of the physics.
  * NEITHER DOES THE BORE.  This design clears the winding head's 20 mm
    bore by 0.37 mm with the ring's run-out charged against it.  Opening
    the ring is still worth something -- give up the bore and 100/45/3.0/2.0
    reaches 0.46 m -- but it costs 13 g to buy 0.08 m, where before it was
    the only way to meet the budget at all.
  * WHAT BINDS INSTEAD IS THE DIAGONALS' OWN FIRST MODE.  1109 of the 1470
    candidates hold the accuracy budget.  What refuses them, most often
    first, is a 1.0 mm web whose members ring inside the propellers'
    200-330 Hz band: 857 of the 1109.  Give up that one rule and the sweep
    reaches 100/40/3.0/1.0 -- 46.5 g and 0.345 m, six grams lighter and
    36% truer than the design above.

    THAT IS THE TRADE TO PUT ON A BENCH NEXT.  MEMBER_F_MIN is a rule
    about fatigue at a joint, not about the cameras: what a singing
    diagonal costs the calibration is in the transmissibility, and the
    model says that column is no worse.  If a pulled joint survives a 1.0
    mm diagonal at blade-pass, the product is 36% more accurate for six
    grams less.  Nobody has pulled one.

AND THE PART CHANGED, for a reason that has nothing to do with structure.
Module 3 focuses with an open-loop voice coil whose postural difference is
+-50 um at a fixed drive current -- 1.06% of the image distance and so
1.06% of every range it reports, which at 100 m is more than the entire
inter-camera budget this spine exists to hold.  Module 2 has no lens
actuator.  It costs 0.07 m of stochastic floor to remove 1.06 m of drift,
and it is a gram lighter, which is why the numbers above are slightly
better than the ones they replace rather than slightly worse.

WHAT THE MOUNT COSTS, now that it is modelled rather than placeheld: the
camera stands 14.1 mm off the chord ends -- the mechanical clearance and
nothing more, because aimed at a FACE of the truss the 62.2 x 48.8 degree
field costs no standoff at all.  It read 21.1 mm until the aim was found to
be assembled in a frame 90 degrees from the chords', pointing the camera 30
degrees off one.

AND THE MOUNT HAS ONE MACHINED PART AFTER ALL: a small rigid collar bonded
round the module's lens housing, whose three arms carry the landings the six
struts reach for.  The struts were assumed to bond straight to the housing;
they cannot.  A ring of landings centred on the spine's axis only lies on the
housing if the housing straddles that axis, and it does not -- two landings
of three came out on it and the third came out off the back of the board.
Neither surface the camera really offers will hold the budget: its housing
face is the only one parallel to the end triangle and is too thin to be a
triangle (1.08), and its own mounting holes make a proper triangle and then
work through FR4 that is 26 times softer than the strut bonded to it.  The
collar reads 0.64.

Both were checks comparing a number with a number -- an azimuth against an
azimuth, a radius against a circumradius -- where the question was about
geometry.  check_mount asks the geometric question now, in the camera's own
frame.

THE MASSLESS RIGID BRACKET the mount replaces is still not a bound on it in
either direction: its error changes sign between a 50 g head and a 4 g one,
0.59 against 0.64 here and 1.60 against 1.37 there.  No margin could have
covered both, which is the argument for solving the mount rather than
allowing for it.
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


OPTIMAL_1M = Chosen("optimal_1m", 1000.0, 85.0, 40.0, 3.0, 1.5)
# The short test piece, the same sweep at 300 mm: a quarter of the
# cantilever needs a fraction of the section, so the accuracy budget is
# spent at 0.30 and what binds is the cell -- the ring's rim, the joint
# angle it has qualified, and having enough bays to be a truss at all.
OPTIMAL_300 = Chosen("optimal_300", 300.0, 70.0, 45.0, 1.0, 1.0)
