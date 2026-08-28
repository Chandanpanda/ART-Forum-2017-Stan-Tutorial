"""Agent A mission.

Two hard constraints the simulator surfaced, both real:

  * Agent A's swept radius is 185 mm (beam pockets run the full length, so the
    corners cannot be chamfered below Za 60 -- spec 4.2).  It therefore CANNOT
    turn in place inside the 480x280 deployment box, nor anywhere within 185 mm
    of a wall.  The route leaves the box nose-first and only pivots in open field.
  * The body is 235 wide, so a sweep "mouth centred Y = 70" (spec 9) is
    impossible -- the chassis would be inside the south wall.  Passes run at
    Y 130 and Y 215; the 165 capture band then covers Y 47.5 ... 297.5.

Every action is time-guarded so a mission can fail but never hang.
"""
import numpy as np

HZ = 50.0
def _wrap(a): return (a + 180.0) % 360.0 - 180.0


def guard(gen, seconds, name=""):
    """Run gen for at most `seconds` of control time."""
    n = int(seconds * HZ)
    for i, v in enumerate(gen):
        if i > n:
            return
        yield v


def wait(rb, seconds):
    for _ in range(int(seconds * HZ)):
        rb.stop(); yield


def drive_straight(rb, dist_mm, speed=200.0):
    """Hold the current heading and cover a distance.  No pivoting."""
    x0, y0, hold = rb.pose
    while True:
        x, y, th = rb.pose
        gone = np.hypot(x-x0, y-y0)
        if gone >= abs(dist_mm):
            rb.stop(); return
        v = np.sign(dist_mm) * min(speed, max(70.0, (abs(dist_mm)-gone)*3.0))
        rb.drive(v, np.clip(2.0*_wrap(hold-th), -25, 25)); yield


def turn_to(rb, heading, tol=2.0, wmax=110.0):
    while True:
        err = _wrap(heading - rb.pose[2])
        if abs(err) < tol:
            rb.stop(); return
        rb.drive(0.0, np.clip(2.6*err, -wmax, wmax)); yield


def pursue(rb, x, y, speed=200.0, tol=25.0, wmax=60.0):
    """Pure pursuit -- steers while moving, never pivots.  Safe near walls."""
    while True:
        px, py, th = rb.pose
        if np.hypot(x-px, y-py) < tol:
            rb.stop(); return
        err = _wrap(np.degrees(np.arctan2(y-py, x-px)) - th)
        if abs(err) > 100:                      # target is behind: stop, caller decides
            rb.stop(); return
        v = speed * max(0.35, 1.0 - abs(err)/110.0)
        rb.drive(v, np.clip(1.8*err, -wmax, wmax)); yield


def reverse_to(rb, x, y, speed=110.0, tol=6.0):
    hold = rb.pose[2]
    while True:
        px, py, th = rb.pose
        t = np.radians(th)
        back = -((x-px)*np.cos(t) + (y-py)*np.sin(t))
        if back < tol:
            rb.stop(); return
        rb.drive(-min(speed, max(45.0, back*2.2)),
                 np.clip(1.6*_wrap(hold-th), -25, 25)); yield


def sweep_line(rb, y, x_to, speed=140.0):
    """Collecting run along a constant-Y line, fingers open, heading held at 180."""
    rb.fingers(True)
    yield from guard(turn_to(rb, 180.0), 8.0)
    while rb.pose[0] > x_to:
        th = rb.pose[2]
        lat = np.clip(0.6*(y - rb.pose[1]), -12, 12)     # hold the line
        rb.drive(speed, np.clip(2.0*_wrap(180.0-th) - lat, -22, 22)); yield
    rb.stop()
    # The belt runs continuously, so a piece already aboard keeps travelling aft at
    # ~60 mm/s regardless of the robot.  Measured: 240 mm of belt run = ~4 s.  Dwell
    # so the magazine is loaded before the next manoeuvre.
    yield from wait(rb, 5.0)


def dock_and_post(rb, hole_x, hole_y, chute_offset, stroke=0.28, log=print):
    """Reverse over a lab hole; one gate stroke meters one disc."""
    stage_y = hole_y - chute_offset - 210.0
    yield from guard(pursue(rb, hole_x, stage_y, speed=200.0, tol=30.0), 22.0)
    yield from guard(turn_to(rb, 270.0), 10.0)
    yield from guard(pursue(rb, hole_x, stage_y + 40.0, speed=90.0, tol=18.0), 8.0)
    yield from guard(turn_to(rb, 270.0), 6.0)
    yield from guard(reverse_to(rb, hole_x, hole_y - chute_offset), 14.0)
    yield from wait(rb, 0.4)
    rb.gate(True)
    yield from wait(rb, stroke)
    rb.gate(False)
    yield from wait(rb, 0.7)
    yield from guard(drive_straight(rb, 150.0, speed=150.0), 8.0)   # depart nose-out


def mission_agent_a(rb, holes, hole_y, chute_offset, log=print):
    log("  leaving the deployment box nose-first (no pivot: swept R 185 > 140 to the wall)")
    yield from guard(drive_straight(rb, 300.0, speed=220.0), 10.0)
    log("  sweep pass 1, mouth on Y 130")
    yield from guard(pursue(rb, 430.0, 130.0, speed=220.0, tol=40.0), 20.0)
    yield from guard(sweep_line(rb, 130.0, 158.0), 30.0)
    log("  sweep pass 2, mouth on Y 215")
    yield from guard(reverse_to(rb, 400.0, 130.0), 16.0)
    yield from guard(turn_to(rb, 90.0), 10.0)
    yield from guard(pursue(rb, 430.0, 215.0, speed=200.0, tol=40.0), 20.0)
    yield from guard(sweep_line(rb, 215.0, 158.0), 30.0)
    log("  reverse-docking the laboratory")
    yield from guard(reverse_to(rb, 430.0, 215.0), 18.0)
    yield from guard(turn_to(rb, 60.0), 10.0)
    for i, hx in enumerate(holes):
        log("    hole %d (x=%.1f)" % (i+1, hx))
        yield from dock_and_post(rb, hx, hole_y, chute_offset, log=log)
    log("  parking clear of the lab")
    yield from guard(pursue(rb, 900.0, 200.0, speed=220.0, tol=40.0), 20.0)
    rb.stop()
