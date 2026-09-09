"""The camera spine, judged by what the camera needs: calibration held.

A SEPARATE QUESTION FROM sim/truss.  That package answers "can the cell
build it".  This one answers "is it worth building" -- and it must answer
for a carbon tube too, in the same solver, or the comparison is rhetoric.
Nothing here imports the automation package, and nothing here knows about
winding, fixtures or rings.

THE ONE NUMBER.  A stereo pair measures range by disparity, Z = f B / d.
Rotate one camera by dtheta about the vertical and the disparity shifts by
f dtheta, so the range error is

    dZ = Z^2 dtheta / B          (the focal length cancels)

and a change in the baseline itself gives dZ = Z dB / B.  Both are metres
of range error at a stated range, which is the currency the customer buys
in, so every compliance in this package is converted into it.

WHAT IS MODELLED: linear elastic statics and normal modes of a 3-D frame,
under acceleration and temperature.  That is the calibration question and
all of it.  Crash, creep, moisture and fatigue are deliberately absent --
they belong to particular applications, and a spine on a survey mast has
no crash case at all.

UNITS ARE SI THROUGHOUT: metres, kilograms, newtons, pascals, kelvin,
radians.  Millimetres stop at the builders in build.py.  (Every mixed-unit
stiffness matrix is wrong in a way that looks plausible.)
"""
