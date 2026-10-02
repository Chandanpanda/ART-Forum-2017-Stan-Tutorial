"""The station module (plan, milestone M1): one gantry, copied.

    spec.py        a station derived from what its head must reach: the head,
                   the travel, the drive on each axis, its limits and dynamics
    mjcf.py        that module in MuJoCo, as built: each axis a drive and a
                   carriage with the drive's play between them, the tools on
                   a 3-axis load cell, the head camera, the docks, the
                   fixture, the reference pin and the test pieces
    sim.py         the MuJoCo backend of hal.py: steppers by their own law,
                   encoders, home switches, the load cell, the camera
    hal.py         the contracts the controller and the simulator share
    vision.py      a dark disc's centre: from the camera as built (fast), or
                   from a rendered frame (what makes the fast one honest)
    ops.py         the executor: home, move, look, touch, pick, place, press,
                   screw, probe -- each a generator with a timeout, each
                   waiting on a sensor, never on a duration
    commission.py  the station measuring itself from the drawing: camera,
                   regions and every tool, against the reference pin

sim/scripts/batline/check_gantry.py holds it to its repeatability on the
station it was developed on and on one it has never seen;
demo_station.py films it.
"""
