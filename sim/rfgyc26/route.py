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
from .params import Chassis

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


def back_to(rb, x, y, speed=130.0, tol=25.0):
    """Reverse toward a point, steering the tail -- no pivot.

    After a sweep the robot faces west with the lab behind it, and it is often
    below Y 185 where a pivot would jam the corner on the south wall (F2).  This
    lets it get back out east AND gain clearance in one move, so the next turn is
    legal.
    """
    while True:
        px, py, th = rb.pose
        if np.hypot(x-px, y-py) < tol:
            rb.stop(); return
        err = _wrap(np.degrees(np.arctan2(y-py, x-px)) - (th + 180.0))
        if abs(err) > 100:
            rb.stop(); return
        # differential drive: rotating the body rotates the tail's velocity vector
        # directly, so steer WITH the error (a steered-wheel car would be inverted)
        rb.drive(-speed*max(0.35, 1.0-abs(err)/110.0), np.clip(1.8*err, -45, 45))
        yield


def sweep_line(rb, y, x_to, speed=140.0):
    """Collecting run along a constant-Y line, fingers open, heading held at 180.

    Fingers OPEN is right once the guides start at the mouth width (F22): the
    channel is then continuous from the pivots inward and the fingers only have
    to stay out of its way.  Raking them to the belt width instead re-creates the
    step one bay further forward, and capture drops from 24/24 to 21/24.
    """
    rb.fingers(True)
    yield from guard(turn_to(rb, 180.0), 8.0)
    while rb.pose[0] > x_to:
        th = rb.pose[2]
        lat = np.clip(1.6*(y - rb.pose[1]), -20, 20)     # hold the line
        rb.drive(speed, np.clip(2.0*_wrap(180.0-th) - lat, -22, 22)); yield
    rb.stop()
    # The belt runs continuously, so a piece already aboard keeps travelling aft at
    # ~60 mm/s regardless of the robot.  Measured: 240 mm of belt run = ~4 s.  Dwell
    # so the magazine is loaded before the next manoeuvre.
    # 240 mm of belt run at ~60 mm/s = 4 s per piece, and pieces queue, so a full
    # sweep needs the belt to clear before the next manoeuvre disturbs the stack.
    yield from wait(rb, Chassis.SWEEP_DWELL)


def align_reverse(rb, chute_offset, tx, ty, heading, tol=2.5, max_ticks=900):
    """Back the CHUTE onto a target, steering on its measured cross-track error.

    Docking cannot be dead-reckoned here: the rear ball transfers ride up the
    3 mm lab plate (the spec's own [VERIFY 10.2] question), which pitches the
    chassis and walks the chute several mm.  Disc-in-hole radial clearance is
    only 2 mm, so the terminal has to close on the chute's actual position.
    """
    for _ in range(max_ticks):
        px, py, th = rb.pose
        cx, cy = rb.chute_xy(chute_offset)
        t = np.radians(th)
        # error of the CHUTE, in the robot frame: +fore is toward the nose
        ex, ey = tx - cx, ty - cy
        fore =  ex*np.cos(t) + ey*np.sin(t)
        left = -ex*np.sin(t) + ey*np.cos(t)
        if abs(fore) < tol and abs(left) < tol:
            rb.stop(); return True
        herr = _wrap(heading - th)
        # Two regimes.  Coarse: reverse fast holding the commanded heading.
        # Endgame: the chute is 106.5 mm behind the axle, so 1 deg of yaw swings it
        # 1.9 mm sideways -- far more lateral authority than the heading term needs.
        # Fighting both at once settles into an equilibrium ~15 mm off (hole 3), so
        # once we are close, null the lateral error and let the heading float.
        # DO NOT raise these to save time.  Tried: threshold 18, w +/-22, endgame
        # 55 mm/s and 15 deg/s.  Dock error went 1.9 -> 3.8 mm against a 2 mm
        # posting budget, and hole 3 stopped converging at all (240 s timeout).
        # The endgame is slow because it has to be.
        if abs(fore) > 25.0:
            w = np.clip(1.4*herr - 1.2*left, -18, 18)
            cap = 140.0 if abs(fore) > 60.0 else 60.0
            v = np.clip(fore*2.5, -cap, cap)
        else:
            w = np.clip(-3.2*left, -10, 10)
            v = np.clip(fore*2.0, -35, 35) if abs(fore) > tol else 0.0
        rb.drive(v, w)
        yield
    rb.stop()
    return False


# The only stations that clear the south wall (>=185), the side walls and the lab
# plate rectangle by more than the 185 mm swept radius.  There is no legal pivot
# directly south of the plate: that corridor is 360 mm wide and needs 370.
PIVOT_W = (230.0, 195.0)
PIVOT_E = (900.0, 190.0)
PIVOT_Y = 195.0          # the corridor line, south of the laboratory


def nearest_pivot(hole_x, hole_y):
    """Turn directly SOUTH OF THE SLOT, and approach it square.

    F36 retires F10.  F10 said there was no legal pivot in the corridor between
    the south wall and the laboratory -- 360 mm of room against a 370 mm swept
    circle -- so every dock had to start from a station west or east of the
    plate, and every approach after the first was therefore DIAGONAL.  That is
    fatal once the laboratory is solid and a sample thick: on a diagonal the rear
    ball transfers cross its edge, and a O20 ball cannot climb 6 mm (F33/F35).

    But the swept circle is the CHASSIS, and the chassis floor is at Za 6 while
    the laboratory is 6 mm tall.  The corners pass over it.  The only robot parts
    low enough to touch are the ball transfers and the wheels, and those are well
    inboard.  Measured, turning 0 -> 270 with a 6 mm laboratory:

        y 150   scrapes the south wall, 136 N
        y 170   scrapes, 102 N
        y 180   scrapes, 89 N
        y 190   CLEAN
        y 210   CLEAN

    So the pivot goes directly south of each slot, the approach is square, the
    balls never meet the edge, and the long cross-field trips disappear.
    """
    return (hole_x, PIVOT_Y)

def reseat(rb, cycles=1):
    """One stroke of the positive-feed plunger and back."""
    for _ in range(cycles):
        rb.feed(True)
        yield from wait(rb, 0.7)
        rb.feed(False)
        yield from wait(rb, 0.5)


def settle_stack(rb, cycles=2):
    """Seat the magazine with the positive-feed paddle.

    The last piece in has nothing above it to push it down.  It arrives centred
    (the collar sees to that) but lands ON the stack rather than settling into
    it -- measured: disc 3 perched at Za 34.6 with 22 deg of tilt where a seated
    disc sits at 24.2 and 0.4.  One paddle sweep through the mouth pushes it flat;
    two makes it certain.  The fore-aft jog between sweeps frees anything lightly
    wedged against the bore wall.
    """
    for _ in range(cycles):
        rb.feed(True)
        yield from wait(rb, 0.8)
        rb.feed(False)
        yield from wait(rb, 0.5)
        yield from guard(drive_straight(rb,  22.0, speed=160.0), 3.0)
        yield from guard(drive_straight(rb, -22.0, speed=160.0), 3.0)
    rb.feed(False)
    yield from wait(rb, 0.6)


def dock_and_post(rb, hole_x, hole_y, chute_offset, stroke=0.28, aboard=0,
                  depart=None, log=print, clk=None):
    """Reverse the chute onto a lab hole along a straight line, then meter one disc.

    F10: with a 185 mm swept radius there is NO legal pivot between the south wall
    (needs y >= 185) and the lab plate (needs y <= 175 at 351 < x < 791) -- the
    corridor is 360 mm wide and the swept circle needs 370.  Turning there beaches
    the chassis on the 3 mm plate with its drive wheels off the floor.

    But the dock does not require a particular heading: the chute lies on the
    robot's own axis, so ANY heading works provided the robot is positioned to
    suit.  So it pivots once, west of the plate, to face directly AWAY from the
    hole -- and then simply reverses in a straight line until the chute is on it.
    """
    lap = (lambda w: None) if clk is None else (lambda w: log("        %-14s T+%5.1f" % (w, clk())))
    pv = nearest_pivot(hole_x, hole_y)
    lap("start")
    # get onto the correct pivot station first (turning only where it is legal)
    if np.hypot(rb.pose[0]-pv[0], rb.pose[1]-pv[1]) > 60.0:
        px, py, _ = rb.pose
        yield from guard(turn_to(rb, np.degrees(np.arctan2(pv[1]-py, pv[0]-px))), 22.0)
        yield from guard(pursue(rb, pv[0], pv[1], speed=220.0, tol=30.0), 35.0)
    lap("at pivot")
    # Two passes.  A single straight reverse leaves a few mm of lateral error that
    # align_reverse cannot null (correcting it means rotating, and rotating swings
    # the chute).  Pulling forward and re-aiming from closer in fixes that -- which
    # is the job the spec's 45 deg chamfer would otherwise do (it absorbs +/-10).
    # THREE SHORT PASSES, not two long ones.  A pass that is going to work
    # converges in 7-14 s; one that is not will happily spend its whole guard
    # hunting -- measured, a single approach burned all 55 s and then succeeded
    # in 14 s on the retry, which is 40 s of a 120 s match thrown away.  Capping
    # each pass at 20 s and allowing a third bounds the worst case AND lowers the
    # typical case, because pulling forward and re-aiming is what actually fixes
    # a bad approach.
    for attempt in range(3):
        px, py, _ = rb.pose
        th = np.degrees(np.arctan2(py - hole_y, px - hole_x))   # face away from hole
        yield from guard(turn_to(rb, th, tol=1.2), 22.0)
        # Let the chassis come to rest before the terminal.  align_reverse closes
        # on the CHUTE, 106 mm behind the axle, so any residual yaw rate is
        # 1.9 mm of chute movement per degree -- starting it while the robot is
        # still settling is what made a dock that takes 15 s from rest burn three
        # 20 s passes in the mission.
        yield from wait(rb, 0.4)
        yield from guard(align_reverse(rb, chute_offset, hole_x, hole_y, th,
                                       tol=2.0, max_ticks=2600), 20.0)
        cx, cy = rb.chute_xy(chute_offset)
        lap("pass %d done" % attempt)
        if np.hypot(cx-hole_x, cy-hole_y) < 4.0 or attempt == 2:
            break
        yield from guard(drive_straight(rb, 90.0, speed=140.0), 10.0)
    cx, cy = rb.chute_xy(chute_offset)
    log("      docked: chute(%.1f,%.1f) vs hole(%.1f,%.1f) err %.1f mm"
        % (cx, cy, hole_x, hole_y, np.hypot(cx-hole_x, cy-hole_y)))
    lap("docked")
    yield from reseat(rb)             # seat the stack before metering
    # Escapement, sequenced (F19).  The retainer takes the column at the joint
    # above the bottom piece so the shelf releases exactly one; with only one
    # piece left there is no joint to enter and the retainer stays parked -- the
    # bore rangefinder is what tells the robot which case it is in.
    n = rb.mag_count()
    if n == 0 and aboard:
        # A perched piece is INVISIBLE to the bore ray: it sits off-axis and the
        # beam passes it.  "Empty" while pieces are still aboard therefore means
        # jammed, not empty -- ram it and look again rather than stroking the
        # escapement at nothing.
        log("      bore reads empty with %d still aboard -- re-seating" % aboard)
        yield from reseat(rb, cycles=2)
        n = rb.mag_count()
    log("      magazine holds %d" % n)
    rb.blade(n >= 2)
    yield from wait(rb, 0.5)
    rb.gate(True)
    yield from wait(rb, stroke)
    rb.gate(False)
    yield from wait(rb, 0.5)
    rb.blade(False)
    yield from wait(rb, 0.9)
    # depart nose-out: the robot already faces away from the hole, so driving
    # forward retraces the approach line straight back to the pivot station
    lap("posted")
    # Depart by driving FORWARD off the plate, not by returning to the pivot
    # station this hole was approached from.  Going back west after hole 2 only
    # to set off east for hole 3 cost 35 s -- the whole guard, because the pursue
    # never even arrived.  The next hole's approach picks its own station, so all
    # that is needed here is to get the tail clear of the plate.  Forward travel
    # is capped by the south wall: docked, the axle sits at Y ~293 and the nose
    # is 142.5 ahead of it.
    # Depart onto the PIVOT LINE, not past it.  Docked, the axle sits ~106 mm
    # south of the slot; driving a fixed 130 mm put it at y 163, and turning
    # there scrapes the south wall (clean only at y >= 190, F36), so every
    # inter-hole turn was slow and sloppy.  Stop where the next turn is clean.
    px, py, th = rb.pose
    d_out = depart if depart is not None else max(40.0, py - PIVOT_Y)
    yield from guard(drive_straight(rb, d_out, speed=220.0), 8.0)
    lap("departed")


def mission_agent_a(rb, holes, hole_y, chute_offset, log=print, clock=None):
    # THE MATCH IS 120 s (rules g.1).  Every phase is stamped so the budget
    # is visible in the log, not discovered at the end.
    t = (lambda: "") if clock is None else (lambda: "T+%5.1f  " % clock())
    log(t() + "leaving the deployment box nose-first (no pivot: swept R 185 > 140 to the wall)")
    yield from guard(drive_straight(rb, 300.0, speed=220.0), 10.0)
    log(t() + "sweep pass 1, mouth on Y 130")
    yield from guard(pursue(rb, 430.0, 130.0, speed=220.0, tol=40.0), 20.0)
    yield from guard(sweep_line(rb, 130.0, 158.0), 45.0)
    log(t() + "sweep pass 2, mouth on Y 215")
    yield from guard(back_to(rb, 470.0, 235.0), 22.0)
    yield from guard(turn_to(rb, 180.0), 12.0)
    yield from guard(pursue(rb, 430.0, 215.0, speed=200.0, tol=40.0), 20.0)
    yield from guard(sweep_line(rb, 215.0, 158.0), 45.0)
    log(t() + "settling the magazine")
    yield from guard(settle_stack(rb), 30.0)
    log(t() + "reverse-docking the laboratory")
    yield from guard(back_to(rb, PIVOT_W[0], PIVOT_W[1]), 25.0)
    for i, hx in enumerate(holes):
        log(t() + "  hole %d (x=%.1f)" % (i+1, hx))
        yield from dock_and_post(rb, hx, hole_y, chute_offset,
                                 aboard=len(holes) - i, log=log, clk=clock)
    log(t() + "parking clear of the lab")
    yield from guard(pursue(rb, 900.0, 200.0, speed=220.0, tol=40.0), 20.0)
    rb.stop()
