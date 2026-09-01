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
from .params import Chassis, AgentA, Field, Piece, M2

HZ = 50.0
MATCH = 120.0            # rules g.1
HOLE_BUDGET = 10.0       # measured cost of one sensed dock plus its post (F68)
# 56, and this was measured rather than guessed.  At 44 the laboratory gets a
# second slot -- worth +18 -- and the seal then starts 15 s later and misses:
# over twelve seeds the mean went from +69 to about +50, with four matches
# running past the buzzer with no beams down at all.  The seal is 70 points
# and it is the phase with no slack, so it gets the clock it needs first.
# 34, not 39: the seal used to start from the laboratory and now starts from
# PCC_L, which the kit loop visits last precisely because it is 370 mm from
# where beam 2 stages rather than 530.
BEAM_BUDGET = 34.0
# MEASURED on the isolated rig: 27-28 s from the laboratory's exit to the last
# drop, delivering the full 6/2/2 for +50.  30 leaves a little for a leg that
# has to be re-aimed.
KIT_BUDGET  = 30.0
# 13, which is what a dock MEASURES: 11-16 s from arriving at the pivot to
# being clear of the plate again.  At 9 the robot starts approaches it cannot
# finish -- one match began its third slot at T+66, left the plate at T+84,
# and the seal then ran 7 s past the buzzer and scored nothing.  A slot is
# worth 18 and the seal 70; when they compete, the seal wins.
MIN_DOCK    = 6.0        # floor on the estimate; an F68 dock is 6-11 s
# How close the chassis has to be before the trim slide takes over.  The scan
# recovers +/-22 mm to under a millimetre, so this is deliberately well inside
# it: every millimetre of chassis error is a millimetre of trim stroke spent,
# and the stroke also has to cover where the slot actually is.
LAT_OK      = 14.0
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


def drive_straight(rb, dist_mm, speed=200.0, stall_s=1.0):
    """Hold the current heading and cover a distance.  No pivoting.

    Returns True if it covered the distance, False if it stopped moving --
    the caller can then do something about it instead of the guard silently
    absorbing the whole manoeuvre (F43).
    """
    x0, y0, hold = rb.pose
    last, held = 0.0, 0
    while True:
        x, y, th = rb.pose
        gone = np.hypot(x-x0, y-y0)
        if gone >= abs(dist_mm):
            rb.stop(); return True
        if gone - last < 1.0:
            held += 1
            if held > int(stall_s*HZ):
                rb.stop(); return False
        else:
            last, held = gone, 0
        v = np.sign(dist_mm) * min(speed, max(70.0, (abs(dist_mm)-gone)*3.0))
        rb.drive(v, np.clip(2.0*_wrap(hold-th), -25, 25)); yield


def turn_to(rb, heading, tol=2.0, wmax=160.0, free=True):
    """Turn in place, and give up if the turn is not actually happening.

    F43.  A pivot that scrapes a wall looks exactly like a pivot that is
    working: the controller commands yaw, the steppers turn, and the chassis
    does not move.  With nothing watching, turn_to happily spends its whole
    guard that way -- 22 s of a 120 s match, twice over in one observed run.
    So watch the heading: if it has not changed by 1 deg in 1.2 s while yaw is
    commanded, the robot is jammed.  Back off 45 mm along its own axis (which
    is what frees a corner) and try once more; if that fails too, hand back to
    the caller rather than burning the guard.
    """
    # MEASURED, mean of four 90 deg pivots, with and without beams aboard:
    #
    #     wmax   tol 2.0    tol 4.0    tol 6.0
    #      110    3.2 s      2.4 s      2.0 s
    #      160    2.5 s      2.1 s      1.9 s
    #      220    2.7 s      2.1 s      1.7 s
    #
    # The TOLERANCE dominates, not the rate cap -- a P controller spends its
    # time in the exponential tail.  220 is not reliably better than 160
    # because it overshoots and comes back.  So: 160 by default, and every
    # caller asks for the loosest tolerance its own job can stand.  The
    # laboratory pays six pivots a match, the kit loop four and the beam phase
    # more; half a second each is ten seconds of a 120 s match.
    stuck, th0, held = 0, rb.pose[2], 0
    while True:
        err = _wrap(heading - rb.pose[2])
        if abs(err) < tol:
            rb.stop(); return True
        if abs(_wrap(rb.pose[2] - th0)) < 1.0:
            held += 1
            if held > int(1.2*HZ):
                if not free or stuck:
                    rb.stop(); return False
                stuck, held = 1, 0
                yield from guard(drive_straight(rb, -45.0, speed=120.0), 2.5)
                th0 = rb.pose[2]
                continue
        else:
            th0, held = rb.pose[2], 0
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


def dwell_until_loaded(rb, cap=None, quiet=4.6, want=None):
    """Wait for the belt to finish delivering, not for a fixed time (F42).

    The belt runs continuously, so a piece picked up at the end of a pass is
    still 240 mm from the magazine when the robot stops -- about 4 s at
    60 mm/s.  The old fix was a flat SWEEP_DWELL on every pass, and at 10 s
    twice that is 20 s of a 120 s match spent standing still with nothing
    moving for most of it.  Measured over the seed sweep: the bore count
    settles 1.5-4.5 s into the dwell and never changes again.

    So watch the bore rangefinder instead: hold until the count has been
    steady for `quiet` seconds, then go.  A pass that delivered nothing
    leaves after `quiet`; a pass with a piece still in transit waits exactly
    as long as it has to.  The cap is still SWEEP_DWELL, so this can only be
    faster than the old behaviour, never slower.
    """
    cap = Chassis.SWEEP_DWELL if cap is None else cap
    n, steady, full = rb.mag_count(), 0.0, 0.0
    for _ in range(int(cap * HZ)):
        rb.stop()
        c = rb.mag_count()
        # If everything that exists is already aboard there is nothing left to
        # wait for.  The robot knows the match has three samples in it, and the
        # bore rangefinder tells it how many it is holding, so a pass that
        # swept the lot can leave the instant the last one seats.  Worth 4-5 s,
        # and it costs nothing when the pass was not that lucky.
        # ...but only on a count that HOLDS.  F60 says the ray over-reads as
        # freely as it under-reads, and a piece still riding the belt crosses
        # under it reading like a full stack for a few ticks.  Trusting one
        # sample ended a sweep at T+13.8 with one disc aboard (seed 11) --
        # the exit needs the same steadiness the quiet timer already uses.
        full = full + 1.0/HZ if (want is not None and c >= want) else 0.0
        if full >= 1.2:
            return
        steady = steady + 1.0/HZ if c == n else 0.0
        n = c
        if steady >= quiet:
            return
        yield


def sweep_line(rb, y, x_to, speed=140.0, want=None):
    """Collecting run along a constant-Y line, fingers open, heading held at 180.

    Fingers OPEN is right once the guides start at the mouth width (F22): the
    channel is then continuous from the pivots inward and the fingers only have
    to stay out of its way.  Raking them to the belt width instead re-creates the
    step one bay further forward, and capture drops from 24/24 to 21/24.
    """
    rb.fingers(True)
    yield from guard(turn_to(rb, 180.0), 8.0)
    rb.intake(True)                     # knife down, brush up to speed (F64)
    while rb.pose[0] > x_to:
        th = rb.pose[2]
        lat = np.clip(1.6*(y - rb.pose[1]), -20, 20)     # hold the line
        rb.drive(speed, np.clip(2.0*_wrap(180.0-th) - lat, -22, 22)); yield
    rb.stop()
    # Roller keeps spinning through the dwell -- a piece bitten in the last
    # 100 mm is still on the shim when the wheels stop.  The knife lifts and
    # the brush stops only once the bore count says delivery is over.
    yield from dwell_until_loaded(rb, want=want)
    rb.intake(False)


# The only stations that clear the south wall (>=185), the side walls and the lab
# plate rectangle by more than the 185 mm swept radius.  There is no legal pivot
# directly south of the plate: that corridor is 360 mm wide and needs 370.
PIVOT_W = (230.0, 195.0)
PIVOT_E = (900.0, 190.0)
# 205, not 195.  F36 measured the pivot as clean at y >= 190 and scraping at
# 180, and 195 left 5 mm -- less than the heading error the departure leaves.
# When it does scrape, turn_to has no way to know: it commands yaw into a
# 5-18 N wall contact and burns its whole guard.  One observed match spent
# 30 s getting from hole 2 to hole 3's pivot for that reason, and another 20 s
# grinding the south wall during the park.  10 mm of margin costs nothing.
PIVOT_Y = 205.0          # the corridor line, south of the laboratory


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

# ================================================== F69 SEEING THE LABORATORY
# F68 built the dock around two reflectance probes on the posting head: a servo
# sweep to find the slot's rims, a plate-edge crossing to datum the range, and
# a short dead-reckoned run at the end.  It worked -- +/-22 mm of capture, under
# a millimetre of residual, 6 s a slot against 17 -- but every millimetre of it
# was bought by MOVING something, because a reflectance sensor only answers
# "is there anything 14 mm below this one point".
#
# A calibrated stereo camera answers the question the robot is actually asking.
# From the pivot line it sees all three slots at once, in three dimensions, and
# the sweep, the edge crossing and the world coordinates all go away.  What
# survives is the trim slide: a differential drive still cannot move sideways.


def reverse_track(rb, sw, trim_mm, heading, speed=150.0, stop_at=0.0, k=0.010,
                  psi_max=14.0):
    """Reverse onto a measured slot, steering out the lateral error on the way.

    The chassis usually arrives 20-40 mm off the slot, and the trim slide only
    has 22.  Correcting that by turning east, driving, and turning back costs
    two pivots and 3.6 s a slot -- 11 s of a 120 s match to move 30 mm sideways.
    But the robot is about to reverse 130 mm anyway, and a reverse with a little
    steering in it lands somewhere else: de/dt = v*sin(psi), so 8 deg of heading
    error held over 130 mm is 18 mm of lateral travel, for nothing.

    So this steers to a heading that closes the cross-track error and then
    straightens as the range runs out, handing the last few millimetres to the
    trim slide, which is what the trim slide is for.  psi_max is deliberately
    small: a big angle would land the range short and swing the bore, and the
    goal here is not to null the error but to get it inside the slide's stroke.
    """
    t = np.radians(heading)
    ux, uy = -np.cos(t), -np.sin(t)              # the reverse direction
    while True:
        # ODOMETRY frame throughout: sw was frozen in it, and a camera fix
        # jumping the map frame mid-reverse must not yank the target.
        x, y, th = rb.pose_odo
        rem = range_to(rb, sw, trim_mm)
        if rem <= stop_at:
            rb.stop(); return
        # cross-track: where the BORE is, across the reverse line
        bx = x + BORE_X*np.cos(np.radians(th)) - trim_mm*np.sin(np.radians(th))
        by = y + BORE_X*np.sin(np.radians(th)) + trim_mm*np.cos(np.radians(th))
        e = (bx - sw[0])*(-uy) + (by - sw[1])*(ux)
        # straighten out over the last 45 mm so the bore is not still swinging
        lim = np.radians(psi_max) * min(1.0, max(rem - 45.0, 0.0)/60.0)
        psi_des = float(np.clip(-k*e, -lim, lim))
        psi = np.radians(_wrap(th - heading))
        rb.drive(-min(speed, max(35.0, 3.0*rem)),
                 float(np.clip(np.degrees(3.2*(psi_des - psi)), -55.0, 55.0)))
        yield


def look_lab(rb, n=10, want=3):
    """Average n frames and return the slots the rig can see.

    Returns [(x_mm, y_mm, mode)] in the ROBOT frame at the last frame, ordered
    by the robot's own +Ya -- which at a dock heading is the line the three
    slots lie on.

    EACH FRAME IS CONVERTED TO WORLD BEFORE IT IS AVERAGED.  Averaging in the
    robot frame silently assumes the robot is not moving, and it is: look_lab
    runs straight after a drive, so the chassis is still coasting.  Ten frames
    at 50 Hz is 0.2 s, which at 170 mm/s is 34 mm of travel smeared into the
    answer -- measured, one slot came back 9.9 mm out on a measurement whose
    noise is about one, and that sample missed its hole.  A world-frame mean is
    right whether the robot is moving or not, which also means the robot may
    look WHILE it drives rather than stopping to.

    Averaging is worth the fifth of a second.  Per-frame noise is independent
    between frames and falls as 1/sqrt(n); the calibration bias does not, and is
    not meant to -- it is the same every frame, which is why the PITCH between
    two slots (a difference of two measurements) is good to a fifth of a
    millimetre while their absolute position is good to about two.
    """
    acc = []
    for _ in range(n):
        x, y, th = rb.pose_odo
        t = np.radians(th)
        for lx_, ly_, lz_, mode in rb.see_lab():
            acc.append((x + lx_*np.cos(t) - ly_*np.sin(t),
                        y + lx_*np.sin(t) + ly_*np.cos(t), mode))
        yield
    groups = []
    for wx, wy, mode in acc:
        for g in groups:
            if np.hypot(g[0][0]-wx, g[0][1]-wy) < 35.0:
                g.append((wx, wy, mode)); break
        else:
            groups.append([(wx, wy, mode)])
    x, y, th = rb.pose_odo
    t = np.radians(th)
    out = []
    for g in groups:
        if len(g) < max(2, n//3):          # seen in too few frames to trust
            continue
        mx = float(np.mean([v[0] for v in g]))
        my = float(np.mean([v[1] for v in g]))
        dx, dy = mx - x, my - y
        out.append((dx*np.cos(-t) - dy*np.sin(-t),
                    dx*np.sin(-t) + dy*np.cos(-t),
                    "stereo" if any(v[2] == "stereo" for v in g) else "mono"))
    out.sort(key=lambda v: v[1])
    rb.lab_seen = out


def pick_slot(seen, want_y):
    """Choose the measured slot nearest a wanted lateral offset."""
    if not seen:
        return None
    return min(seen, key=lambda v: abs(v[1] - want_y))


def reseat(rb, cycles=1):
    """One stroke of the positive-feed plunger and back."""
    for _ in range(cycles):
        rb.feed(True)
        yield from wait(rb, 0.7)
        rb.feed(False)
        yield from wait(rb, 0.5)


def settle_stack(rb, cycles=2, want=None):
    """Seat the magazine with the positive-feed paddle.

    The last piece in has nothing above it to push it down.  It arrives centred
    (the collar sees to that) but lands ON the stack rather than settling into
    it -- measured: disc 3 perched at Za 34.6 with 22 deg of tilt where a seated
    disc sits at 24.2 and 0.4.  One paddle sweep through the mouth pushes it flat;
    two makes it certain.  The fore-aft jog between sweeps frees anything lightly
    wedged against the bore wall.
    """
    for i in range(cycles):
        # Stop early once the bore can SEE everything the robot is carrying.  A
        # perched piece is invisible to the bore ray (it sits off-axis and the
        # beam passes it), so a full count is proof the stack is seated flat --
        # which is the only thing the second cycle was ever for.  Worth 2.7 s.
        if i and want is not None and rb.mag_count() >= want:
            break
        rb.feed(True)
        yield from wait(rb, 0.8)
        rb.feed(False)
        yield from wait(rb, 0.5)
        yield from guard(drive_straight(rb,  22.0, speed=160.0), 3.0)
        yield from guard(drive_straight(rb, -22.0, speed=160.0), 3.0)
    rb.feed(False)
    yield from wait(rb, 0.6)


import os


# The bore's own place in the robot frame: CHUTE_X measured from the axle,
# negative because it is behind it.
BORE_X = AgentA.CHUTE_X - AgentA.AXLE_X
# WHERE THE ROBOT STANDS TO LOOK.  The pivot line is 88 mm before the dock and
# the camera cannot focus disparity closer than 104 -- so from the pivot the
# laboratory is inside the blind cone and only the nearest slot shows, in the
# RGB fallback.  Backing off to axle Y 158 puts the slots 133 mm away, in
# stereo, with the nose still 17 mm clear of the south wall.  It costs 45 mm
# out and 45 mm back: under half a second, for the difference between seeing
# the laboratory and guessing at it.
LOOK_Y      = 158.0


def slot_world(rb, tgt):
    """A measured slot, in world mm, frozen at the pose it was measured from.

    Converting once and then driving to a WORLD point is what makes re-measuring
    safe: the remaining distance is recomputed from wherever the robot now is,
    so a look that fails leaves the previous answer still correct instead of
    replaying a leg that has already been driven.  (Not doing this drove the
    approach twice and put the bore 30 mm past the slot.)
    """
    x, y, th = rb.pose_odo
    t = np.radians(th)
    return (x + tgt[0]*np.cos(t) - tgt[1]*np.sin(t),
            y + tgt[0]*np.sin(t) + tgt[1]*np.cos(t))


def _bore_dy(rb, sw):
    """Lateral offset from the bore's own axis to a measured slot, robot frame."""
    x, y, th = rb.pose_odo
    t = np.radians(th)
    bx, by = x + BORE_X*np.cos(t), y + BORE_X*np.sin(t)
    return -(sw[0]-bx)*np.sin(t) + (sw[1]-by)*np.cos(t)


def range_to(rb, sw, trim_mm):
    """How far to REVERSE to put the bore (at robot x=BORE_X, y=trim) on sw."""
    x, y, th = rb.pose_odo
    t = np.radians(th)
    bx = x + BORE_X*np.cos(t) - trim_mm*np.sin(t)
    by = y + BORE_X*np.sin(t) + trim_mm*np.cos(t)
    return -((sw[0]-bx)*np.cos(t) + (sw[1]-by)*np.sin(t))


def dock_and_post(rb, hole_x, hole_y, chute_offset, stroke=0.60, aboard=0,
                  depart=None, log=print, clk=None, deadline=None):
    """Look at the slot, drive the vector, aim the head, meter one disc.

    THE TERMINAL IS NOW ENTIRELY RELATIVE.  What the robot needs is the slot's
    position with respect to its own bore, and that is what the camera gives it:
    reverse by (BORE_X - sx) and trim by sy.  No world coordinate enters, so the
    dock is no better or worse than the camera -- it does not also depend on the
    navigation being right, which is the property that has to survive the move
    from ground truth to SLAM.

    hole_x is still passed in, but only to say WHICH slot this is: the robot
    drives to roughly the right place on the pivot line and then takes whichever
    slot the camera reports nearest to the one it meant.
    """
    lap = (lambda w: None) if clk is None else (lambda w: log("        %-14s T+%5.1f" % (w, clk())))
    pv = nearest_pivot(hole_x, hole_y)
    lap("start")
    px, py, ph = rb.pose
    if abs(py - PIVOT_Y) < 55.0 and abs(_wrap(ph - 270.0)) < 25.0:
        # ALREADY ON THE PIVOT LINE FACING THE PLATE, which is where the last
        # slot's departure leaves the robot.  Two turns and a straight run beat
        # a pursue: measured 3.7 s against 6.5, three times a match.  pursue has
        # to curve in from wherever it is and then be squared up afterwards.
        dx = pv[0] - px
        if abs(dx) > 12.0:
            h = 0.0 if dx > 0 else 180.0
            yield from guard(turn_to(rb, h, tol=6.0), 8.0)
            yield from guard(drive_straight(rb, abs(dx), speed=220.0), 8.0)
    elif np.hypot(px-pv[0], py-pv[1]) > 60.0:
        yield from guard(turn_to(rb, np.degrees(np.arctan2(pv[1]-py, pv[0]-px))), 9.0)
        yield from guard(pursue(rb, pv[0], pv[1], speed=220.0, tol=20.0), 35.0)
    lap("at pivot")
    # Square to the plate.  Loose on purpose: 4 deg of yaw is 0.2% on the range
    # and the lateral is measured, not dead-reckoned.  Squaring to 1.5 deg cost
    # 3.4 s of a 16 s dock and bought nothing.
    # 6 deg, because nothing downstream needs better: the lateral is measured
    # and the tracked reverse steers, and 6 deg of yaw is 0.5% on the range.
    yield from guard(turn_to(rb, 270.0, tol=6.0), 9.0)
    yield from wait(rb, 0.15)
    lap("squared")

    # LOOK -- from far enough back that the camera can actually focus on it.
    if rb.pose[1] > LOOK_Y + 8.0:
        yield from guard(drive_straight(rb, rb.pose[1] - LOOK_Y, speed=200.0), 5.0)
    yield from guard(look_lab(rb), 3.0)
    seen = getattr(rb, "lab_seen", [])
    if not seen:
        log("      no slot in view -- falling back to dead reckoning")
    lap("looked (%d)" % len(seen))

    # Only step across when the reverse itself cannot absorb it.  A tracked
    # reverse takes out 15-20 mm on the way in for free; beyond about 45 the
    # heading angle it would need starts costing range, and a pivot is cheaper.
    seen = getattr(rb, "lab_seen", [])
    tgt0 = pick_slot(seen, 0.0)
    dy0 = tgt0[1] if tgt0 else (hole_x - rb.pose[0])
    # 36: the tracked reverse takes out about 16 mm over the 112 mm it has and
    # the trim slide 25, so anything inside their sum needs no pivot at all.
    if abs(dy0) > 36.0:
        # AND THE PIVOT CANNOT HAPPEN HERE.  The look station is at Y 158 and
        # the swept radius is 185, so a turn on the spot puts the robot's corner
        # 27 mm through the south wall.  It grinds, the re-look then comes back
        # empty, and the dock falls through to dead reckoning -- measured, 21.5
        # mm out and the sample lost.  Back onto the pivot line first, which
        # F36 measured clean at Y >= 190, and return to the look station after.
        log("      %+.1f mm off the slot -- stepping across" % dy0)
        yield from guard(drive_straight(rb, -(PIVOT_Y + 5.0 - rb.pose[1]),
                                        speed=200.0), 4.0)
        h = 0.0 if dy0 > 0 else 180.0
        yield from guard(turn_to(rb, h, tol=3.0), 9.0)
        yield from guard(drive_straight(rb, abs(dy0), speed=180.0), 8.0)
        yield from guard(turn_to(rb, 270.0, tol=6.0), 9.0)
        yield from guard(drive_straight(rb, rb.pose[1] - LOOK_Y, speed=200.0), 5.0)
        yield from guard(look_lab(rb), 3.0)
    lap("lined up")

    # DRIVE THE VECTOR.  Re-measuring on the way in is free and it removes the
    # approach's own error: the slot stays in frame down to ~70 mm before the
    # dock, by which point stereo has handed over to the RGB sensor's view of a
    # circle whose diameter is known.
    tgt = pick_slot(getattr(rb, "lab_seen", []), 0.0)
    if tgt is not None:
        sw = slot_world(rb, tgt)
        rb.trim(float(np.clip(tgt[1], -AgentA.TRIM_Y, AgentA.TRIM_Y)))
        # In to where the slot is about to leave the frame, then look again from
        # close range -- where a millimetre of disparity is worth a third of a
        # millimetre -- and finish on that.
        # Stop at the NEAR EDGE OF THE CAMERA'S BAND, not at some round number.
        # Below about 105 mm the robot's own tail shell cuts the near rim of the
        # slot out of the frame, so a look taken closer than that returns
        # nothing and the leg is wasted.
        # THE STEERING BELONGS ON THE LONG LEG.  This one is only ~23 mm --
        # from the look station to the near edge of the camera's band -- and
        # putting the authority here gave 4 deg over the 112 mm that follows,
        # which is 8 mm of correction.  Measured: a slot that wanted 33 mm of
        # lateral got 18 and the dock landed 14.6 mm out.
        yield from guard(reverse_track(rb, sw, rb.trim_at(), 270.0,
                                       speed=170.0, stop_at=112.0, psi_max=5.0), 8.0)
        yield from guard(look_lab(rb, n=6), 2.5)
        # TRACK THE SAME SLOT.  Closer in, the nearest slot's rim leaves the
        # frame before its neighbours' do -- so a re-look can come back with the
        # two OUTER slots and not the one being docked, and "nearest to zero"
        # would then cheerfully switch targets 140 mm sideways.  Match against
        # where the slot already measured should now be.
        t2 = pick_slot(getattr(rb, "lab_seen", []), _bore_dy(rb, sw))
        if t2 is not None and abs(t2[1] - _bore_dy(rb, sw)) > 40.0:
            t2 = None
        if t2 is not None:
            tgt, sw = t2, slot_world(rb, t2)
        rb.trim(float(np.clip(_bore_dy(rb, sw), -AgentA.TRIM_Y, AgentA.TRIM_Y)))
        yield from guard(reverse_track(rb, sw, rb.trim_at(), 270.0,
                                       speed=95.0, psi_max=15.0), 9.0)
        # AIM THE HEAD LAST.  The tracked reverse steers on purpose -- that is
        # how it takes 16 mm of lateral error out over 112 mm -- so a trim set
        # before it is a trim set against a pose the robot then left.  Measured
        # with the trim set early: 2.4, 7.3 and 6.3 mm of residual on a
        # measurement good to about one.  Set from where the robot has actually
        # ended up, it is the measurement's own error and nothing else.
        rb.stop()
        rb.trim(float(np.clip(_bore_dy(rb, sw), -AgentA.TRIM_Y, AgentA.TRIM_Y)))
        for _ in range(int(0.7*HZ)):
            if rb.trim_settled():
                break
            yield
    else:
        cx, cy = rb.chute_xy(chute_offset)
        yield from guard(drive_straight(rb, -(hole_y - cy), speed=110.0), 12.0)
    rb.stop()
    yield from wait(rb, 0.2)
    cx, cy = rb.chute_xy(chute_offset)
    th = np.radians(rb.pose[2])
    tr = rb.trim_at()
    bx, by = cx - tr*np.sin(th), cy + tr*np.cos(th)
    log("      docked: bore(%.1f,%.1f) vs hole(%.1f,%.1f) err %.1f mm  (trim %+.1f, %s)"
        % (bx, by, hole_x, hole_y, np.hypot(bx-hole_x, by-hole_y), tr,
           tgt[2] if tgt else "blind"))
    lap("docked")

    if not aboard or rb.mag_count() < aboard:
        yield from reseat(rb)
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
    yield from wait(rb, 0.4)
    # Hold the leaves open long enough for the piece to clear them (F41), and
    # leave them open until the head is off the slot (F55).  Both findings were
    # made against a single sliding shelf, which swept a released disc 10.7 mm
    # sideways on its return and could pinch one against the slot's countersink
    # hard enough to bolt the robot to the laboratory.  Two leaves retracting to
    # opposite sides apply no net side force -- measured, a released disc moves
    # 0.11 mm laterally instead of 4.6 -- but the sequencing costs nothing.
    rb.gate(True)
    yield from wait(rb, stroke)
    lap("posted")
    px, py, th = rb.pose
    d_out = depart if depart is not None else max(40.0, py - PIVOT_Y)
    yield from guard(drive_straight(rb, d_out, speed=220.0), 8.0)
    for _ in range(3):
        if rb.pose[1] <= PIVOT_Y + 35.0:
            break
        log("      departure blocked at Y %.0f -- rocking free" % rb.pose[1])
        yield from guard(drive_straight(rb, -45.0, speed=140.0), 4.0)
        yield from guard(drive_straight(rb, rb.pose[1] - PIVOT_Y + 45.0,
                                        speed=200.0), 7.0)
    # Clear of the slot -- safe to close under the column again, and to bring
    # the head back to centre so the belt tail feeds the bore.
    rb.gate(False)
    rb.trim(0.0)
    yield from wait(rb, 0.35)
    rb.blade(False)
    yield from wait(rb, 0.4)
    lap("departed")


# ==================================================== MISSION 2: THE KITS
# 80 points of swing for a driving job.  Agent A currently scores -30 on the
# kits -- three empty destination zones at -10 each -- and the full 6/2/2
# distribution is +50.  Nothing has to be picked up: rules g.1 let the kits
# start ON the robot, so they are loaded into three hoppers before the match
# and delivery is one flap per zone.
#
# ALL THREE ZONES OR IT IS BARELY WORTH GOING.  Measured on the referee:
# nothing -30, hospital alone -2, hospital plus one PCC +14, the full set +50.
# The last two kits sit in the opposite corner of the field, 943 mm away, and
# they are worth 36 points on their own.
#
# The order is chosen to END in the west, because the beam phase stages there:
# lab -> HOSPITAL -> PCC_R -> PCC_L -> beams is 2987 mm, and the obvious
# alternative (PCC_L before PCC_R) is the same distance but finishes 943 mm
# from where the beams start.
# THE LABORATORY IS IN THE WAY.  A straight run from the lab pivot to the
# hospital crosses the plate at x 759, and the robot cannot climb a 6 mm edge
# (F11) -- measured, it spent 20.7 s going nowhere.  The loop therefore goes
# EAST of the plate (x >= 909 clears it, and x <= 950 leaves the 185 mm swept
# radius clear of the east wall), north, then west along the top, and comes
# down the west side to where the beams stage.
#
# Every drop is made facing NORTH, which costs one turn each and is what makes
# the landing position a property of the design rather than of the arrival
# angle.  The station is the kit's target minus the hopper's own 78 mm offset.
# ORDER AND TRAVERSE: NEVER DRIVE ALONG THE LATITUDE YOU JUST UNLOADED ON.
#
# Every hopper discharges over a flank, so each drop lands ~140 mm to one SIDE
# of its station -- and all three stations sit on the same y 930 line.  Driving
# straight from one station to the next therefore runs the robot along the very
# latitude it has just covered in kits.  Measured on the old route: HOSP's six
# land correctly at (592, 923), inside the zone; the robot reverses 200, runs
# the diagonal to PCC_L, and at T+70 its north-west corner sweeps the pile from
# (589, 925) to (397, 1022) -- 74 mm outside.  Two kits every match, and with
# them the +20 for the 6/2/2 distribution.  The spec's own walk-over audit
# forbids exactly this: "no path over a placed piece".
#
# Reordering does not fix it -- any east-west traverse at y 930 crosses some
# drop -- so the TRAVERSE moves instead.  The robot already reverses 200 mm out
# of each zone before pivoting; it now stays down there, runs to the next
# station's longitude at y 730, and only then turns north.  An L instead of a
# diagonal.  At y 730 the body spans 587-872, clear of every drop at y >= 920.
KIT_ORDER   = ("PCC_R", "HOSP", "PCC_L")
KIT_APPROACH = (950.0, 250.0)          # the dogleg east of the laboratory
KIT_LOOP_Y   = 730.0                   # = station y - KIT_BACKOFF: the traverse
# DROP CENTRALLY, THEN REVERSE OUT BEFORE TURNING.
#
# The first attempt put each station where a PIVOT was also legal.  The swept
# radius is 185 (F44: the beam pockets run the full length, so the corners
# cannot be chamfered below Za 60) and the corner zones are 200 mm boxes against
# two walls, which leaves a 15 x 15 mm square in PCC_R -- and a 740 mm leg
# arrives nowhere near that well.  Measured: the robot landed 50 mm east of it,
# could not then pivot without grinding two walls, and the loop delivered 2 kits
# of 10 and scored -14.
#
# So the two requirements are separated.  The DROP happens in the middle of the
# zone, where a 40 mm arrival error is harmless; then the robot REVERSES 140 mm,
# which is a straight line and needs no radius at all, and pivots from there.
# The stations are all at y 1030 and the pivots all at y 890, which is also
# north of the side areas the patients stand in.
# HOSP SITS 35 mm FURTHER NORTH THAN THE OTHER TWO.  Its hopper discharges
# level with the axle (local x 0..56) while the PCCs' sit forward of it
# (x 84..112), so a HOSP kit lands ~10 mm BEHIND the station and a PCC kit
# ~84 mm ahead.  The hospital's south edge is at y 901, so at a 930 station the
# six-kit scatter reached 896 -- five millimetres out, and it cost the +20
# distribution bonus.  At 965 the same scatter lands 30 mm inside, and the
# robot's north edge is still 74 mm clear of the top wall.
KIT_STATION = {"PCC_R": (903.0, 930.0),
               "HOSP":  (711.5, 965.0),
               "PCC_L": (240.0, 930.0)}
KIT_HEADING = 90.0
# FAR ENOUGH THAT THE PIVOT CLEARS THE KITS IT JUST DROPPED.  They land 140 mm
# to one side, so after reversing R the robot's centre is sqrt(140^2 + R^2) from
# them, and the swept radius is 185 plus the kit's own 18 -- so R must exceed
# 147.  At 140 the circle clipped them and the next leg spent 13 s moving 170 mm
# (measured, and it does the same with every cylinder removed from the field, so
# it was never the patients).  200 also lands the pivot at y 830, just north of
# the side areas.
KIT_BACKOFF = 200.0


def deliver_kits(rb, log=print, clk=None, deadline=None, order=KIT_ORDER):
    """Drive the northern loop and open one hopper in each destination zone."""
    t = (lambda: "") if clk is None else (lambda: "T+%5.1f  " % clk())
    def leg(tx, ty, speed=230.0, cap=14.0, tol=32.0):
        # Aim, run, and re-aim if the run did not land close enough.  A single
        # turn-and-drive carries its heading error the whole way -- 3 deg over a
        # 740 mm leg is 39 mm -- and the second pass halves whatever the first
        # left, which is cheaper than a tight turn tolerance on the first.
        for _ in range(2):
            px, py, _ = rb.pose
            d_ = np.hypot(tx-px, ty-py)
            if d_ < tol:
                return
            # THE TURN TOLERANCE HAS TO SCALE WITH THE LEG, and the obvious
            # way round is the wrong one.  A heading error is a LATERAL error
            # multiplied by the distance: 8 deg over the 780 mm run up the east
            # side is 108 mm, which walked the robot from x 950 to x 1009 and
            # jammed it against the east wall with 134 mm of clearance and a
            # 185 mm swept radius.  Ask for whatever angle keeps the miss under
            # 25 mm, and let the second pass clean up the rest.
            tol_ = float(np.clip(np.degrees(np.arctan2(25.0, d_)), 2.0, 10.0))
            yield from guard(turn_to(rb, np.degrees(np.arctan2(ty-py, tx-px)),
                                     tol=tol_), 9.0)
            yield from guard(drive_straight(rb, d_, speed=speed), cap)

    # get east of the laboratory before turning north
    if rb.pose[1] < 320.0:
        yield from leg(*KIT_APPROACH, speed=230.0, cap=8.0)
    for dest in order:
        if deadline is not None and clk is not None and clk() > deadline:
            log(t() + "  kits: %s abandoned at the beam deadline" % dest)
            break
        tx, ty = KIT_STATION[dest]
        log(t() + "  kits -> %s (%.0f, %.0f)" % (dest, tx, ty))
        # Traverse SOUTH of the drops, then turn up into the zone.
        if abs(rb.pose[0] - tx) > 60.0:
            yield from leg(tx, KIT_LOOP_Y, cap=10.0)
        yield from leg(tx, ty)
        # 8 deg is enough: the hopper mouth is 78 mm off the centreline, so
        # 8 deg of heading error moves the landing point 11 mm, against zone
        # margins of 50 mm and more.
        yield from guard(turn_to(rb, KIT_HEADING, tol=8.0), 8.0)
        rb.stop()
        yield from wait(rb, 0.2)
        n = rb.open_hopper(dest)
        px, py, th = rb.pose
        hx_, hy_ = M2.HOPPER[dest]
        tr = np.radians(th)
        kx = px + hx_*np.cos(tr) - hy_*np.sin(tr)
        ky = py + hx_*np.sin(tr) + hy_*np.cos(tr)
        log("      dropped %d kit(s); lip at (%.0f, %.0f)%s"
            % (n, kx, ky, "" if _in_zone(dest, kx, ky) else "  -- OUTSIDE THE ZONE"))
        yield from wait(rb, 0.5)
        # Reverse out of the corner before pivoting: reversing needs no swept
        # radius, and the kits are outboard of the track so nothing is run over.
        yield from guard(drive_straight(rb, -KIT_BACKOFF, speed=220.0), 5.0)


def _in_zone(dest, x, y):
    box = {"HOSP": Field.HOSPITAL, "PCC_L": Field.PCC_L, "PCC_R": Field.PCC_R}[dest]
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def mission_agent_a(rb, holes, hole_y, chute_offset, log=print, clock=None):
    # THE MATCH IS 120 s (rules g.1).  Every phase is stamped so the budget
    # is visible in the log, not discovered at the end.
    t = (lambda: "") if clock is None else (lambda: "T+%5.1f  " % clock())
    log(t() + "leaving the deployment box nose-first (no pivot: swept R 185 > 140 to the wall)")
    yield from guard(drive_straight(rb, 300.0, speed=220.0), 10.0)
    # THE LANES ARE 130 AND 215, AND THE THIRD PASS IS THE FIX.
    #
    # The mouth collects a piece whose centre is within about 55 mm of the lane;
    # the chassis is 235 wide.  So every pass carries ~62 mm of bulldozer beyond
    # each side of what it can actually take, and a sample shoved NORTH leaves
    # the quarantine, where the second lane cannot reach it -- seed 3's went
    # (60, 226) -> (37, 273) and, worse, into beam 1's footprint, so the beam
    # landed on it and lost its own +25 as well.  Seeds 5 and 12, the same.
    #
    # Moving the lanes does not fix it, it moves it: at Y 178 a sample 45 mm off
    # the lane meets the guide's leading edge instead of its inner face, jams the
    # nose and yaws the chassis 25 deg, and the pass then ploughs the other two
    # (measured on seed 1 -- 1 of 3 collected, against 3 of 3 at Y 130).  These
    # two lanes were tuned against 24 randomised samples and they collect; the
    # defect is what happens to the ones they miss.
    #
    # So keep them, and go and get the strays.  A third lane at Y 265 covers the
    # strip a Y 130 pass shoves into.
    #
    # ONLY THE THIRD PASS IS CONDITIONAL.  Skipping the second when the bore
    # already reads three looked like 12 s for nothing, and it is not: the bore
    # is a rangefinder down the magazine and it reads the top of the stack, so a
    # piece that arrives perched -- which is the whole reason settle_stack
    # exists -- reads as a full magazine.  Measured on seed 1: three counted
    # after one pass, one actually aboard, the other two still on the field, and
    # the match scored +7.  The count is good enough to decide whether to spend
    # a bonus pass; it is not good enough to skip a pass the robot needs.
    def _pass(lane, ret):
        if ret is not None:
            # Back out ALONG the field, then let sweep_line find its own lane.
            # Do NOT pursue() it: the target is under 100 mm away and the loaded
            # turn radius at sweep speed is 190, so pursue circles the point
            # instead of arriving -- 20 s of a 120 s match, measured.
            yield from guard(back_to(rb, 470.0, ret), 22.0)
            yield from guard(turn_to(rb, 180.0), 12.0)
        else:
            yield from guard(pursue(rb, 430.0, lane, speed=220.0, tol=40.0), 20.0)
        yield from guard(sweep_line(rb, lane, 158.0, want=3), 45.0)

    log(t() + "sweep pass 1, mouth on Y 130")
    yield from _pass(130.0, None)
    log(t() + "sweep pass 2, mouth on Y 215")
    yield from _pass(215.0, 235.0)
    log(t() + "settling the magazine")
    yield from guard(settle_stack(rb, want=len(holes)), 30.0)
    # THERE IS NO THIRD LANE, AND IT WAS TRIED.  A pass at Y 265 covers the strip
    # a Y 130 pass shoves into, so it looked like the way to get the strays back.
    # It is not, twice over:
    #
    #  * Gated on the bore count BEFORE settling it fires when nothing is
    #    missing -- a perched piece does not read (F60) -- and it then shoves
    #    whatever IS still in the quarantine north and out of it.  Seeds 1, 2
    #    and 3 all swept a lane they did not need and all three finished with
    #    fewer samples than they started the pass with.
    #  * Gated after settling, so the count is honest, it fires on the right
    #    matches and still does not pay: on seed 3 it spent 41 s and recovered
    #    nothing, and the match went from +50 to +32.
    #
    # A sample that has been shoved out of the quarantine is 20 points; 41 s of a
    # 120 s match is the beam task.  The recovery has to be cheaper than this or
    # it does not belong in the route -- see F59 and next steps.

    # ORDER: SAMPLES, THEN BEAMS -- and the samples get whatever the beams do
    # not need, not the other way round.
    #
    #  * The beams SEAL the quarantine, so they cannot go down before the
    #    samples are swept out of it: the corner is 280 x 250 once closed and
    #    the robot is 285 long.
    #  * Once beam 1 is down the robot is BOXED IN.  It finishes north of a
    #    60 mm obstacle, west of a laboratory its ball transfers cannot climb,
    #    and the gap between the two is 87 mm against a 235 mm body.  Getting
    #    back out to the laboratory means going round the north of the plate --
    #    about 25 s.  So the beams have to be the last thing the robot does.
    #  * They are 70 points against the laboratory's 50, and both together do
    #    not fit in 120 s.  Spec section 9's rule decides it: the 70-point task
    #    must never die downstream of the 50-point one.  So the laboratory runs
    #    on a budget and stops when the beams need the clock.
    log(t() + "reverse-docking the laboratory")
    yield from guard(back_to(rb, PIVOT_W[0], PIVOT_W[1]), 25.0)
    # THE LABORATORY RUNS TO A DEADLINE, NOT A BUDGET.  A per-slot budget has to
    # be pessimistic enough for the worst dock, and then it throws away slots
    # the robot had time for: one measured match abandoned its third slot with
    # 56 s left and then sealed the quarantine in 39.  A deadline lets the robot
    # spend every second it actually has and stop the moment the beams need it,
    # including part-way through an approach.
    # THE LABORATORY NOW YIELDS TO TWO THINGS, NOT ONE.  A slot is worth +18.
    # The beam seal is +70 and the kit loop is +80 of swing (-30 to +50), so
    # both outrank it, and the deadline is what is left after both.
    dl, est = MATCH - BEAM_BUDGET - KIT_BUDGET, MIN_DOCK
    for i, hx in enumerate(holes):
        # ...and the estimate of what a slot costs is MEASURED, not assumed.
        # Docks run 11-17 s depending on how the approach goes, and a constant
        # is wrong in both directions: too small and the robot starts a slot it
        # cannot finish, losing the 70-point seal to gain 18; too large and it
        # gives up a slot it had time for.  Time the last one instead.
        if clock is not None and clock() + est > dl:
            log(t() + "  hole %d not started: T+%.0f + %.0f would pass %.0f"
                % (i+1, clock(), est, dl))
            break
        log(t() + "  hole %d (x=%.1f)" % (i+1, hx))
        t0 = clock() if clock else 0.0
        yield from dock_and_post(rb, hx, hole_y, chute_offset,
                                 aboard=len(holes) - i, log=log, clk=clock,
                                 deadline=dl)
        if clock is not None:
            est = max(MIN_DOCK, clock() - t0)
    # KITS BEFORE BEAMS, and northwards from the laboratory.  The beams seal the
    # quarantine and box the robot into the south-west corner (F44), so they are
    # always last; the kit loop is chosen to finish at PCC_L, 723 mm from where
    # beam 2 stages, rather than at PCC_R which is 943 mm further away.
    log(t() + "delivering the medical kits")
    yield from guard(deliver_kits(rb, log=log, clk=clock,
                                  deadline=MATCH - BEAM_BUDGET),
                     KIT_BUDGET + 8.0)
    yield from seal_quarantine(rb, log=log, clk=clock)
    rb.stop()


# ============================================================ BEAM PLACEMENT
# The 70-point task.  Two rigid beams seal the quarantine corner: beam 1 (280)
# from the west wall along Y 250-270, beam 2 (250) from the south wall along
# X 280-300, its north end face butting beam 1's south face at (280, 250).
#
# Three things make this hard, and all three are geometry rather than control:
#
#  * The sealed corner is 280 x 250 and Agent A is 285 long, so ONCE EITHER
#    BEAM IS DOWN the robot no longer fits beside the other one.  Every station
#    therefore has to work from outside the corner, which fixes the approach
#    axis of each placement exactly (F44).
#  * A beam dragging on the floor cannot cross the 6 mm laboratory, and the
#    swept circle with a beam aboard is 189 mm against a 360 mm corridor.  That
#    left no legal pivot anywhere south of the laboratory.  Carrying the beams
#    12 mm clear (F46) is what buys the pivots back.
#  * The release cannot be a beam-length withdrawal (280 mm of straight line
#    the corner does not have).  With the cradles down, nothing but the end
#    stop touches the piece, so backing the stop off 45 mm IS the release.


def stall_drive(rb, v, hold_heading, max_mm=400.0, thresh=0.30, settle=0.35,
                line=None, crab_max=12.0, gain=1.0, taper_mm=70.0):
    """Drive until the drive torque saturates -- the spec's wall-stall datum.

    This is the localisation primitive the spec calls "left-wall stall" and
    "bottom-wall stall": no encoder, no ToF, just push until the motors say the
    robot has stopped.  StallGuard on a TMC2209 is exactly this signal.
    """
    x0, y0, _ = rb.pose
    held = 0
    for _ in range(int(max_mm/abs(v)*HZ) + int(3*HZ)):
        x, y, th = rb.pose
        gone = np.hypot(x-x0, y-y0)
        if gone > max_mm:
            rb.stop(); return False
        moving = rb.speed()
        # A robot accelerating from rest is ALSO at torque saturation with a
        # near-zero velocity, so StallGuard reads "stalled" the instant it is
        # asked to move.  The real chip has the same blind spot -- the datasheet
        # says it is unreliable below about 0.1 m/s -- so ignore it until the
        # robot has actually covered some ground.
        if gone > 20.0 and rb.stalled(thresh) and moving < 6.0:
            held += 1
            if held > int(settle*HZ):
                rb.stop(); return True
        else:
            held = 0
        aim = hold_heading
        if line is not None:
            # Follow the LINE, not just the heading.  The wall gives the robot
            # its position along the approach; nothing gives it the position
            # across, so a heading-only run-in lands the beam parallel to its
            # target and 12-15 mm off it.  Crab onto the line by aiming a few
            # degrees off it, which is a proper line follower rather than a
            # yaw-rate nudge that the heading term then cancels.
            t = np.radians(hold_heading)
            lat = -(x - line[0])*np.sin(t) + (y - line[1])*np.cos(t)
            # Gain 1.0: one degree of crab per millimetre off the line.  At
            # 0.4 the loop has a 1.2 s time constant against a 1.25 s run-in
            # and settles 12 mm out -- enough to lose the closure bonus.
            #
            # But cap the crab at 7 deg on the RUN-IN, where the free approach
            # legs use 22.  The run-in ends by driving the beam into a wall,
            # and a crabbed robot presents the beam's CORNER to that wall: at
            # 22 deg the beam leads the chassis by 3 mm, so the stall triggers
            # on a corner touch with the cross-track error still uncorrected.
            # Measured, that put beam 2 down 58 mm east of its line.
            # ...and taper it to zero over the last 70 mm.  Whatever crab is
            # still on when the beam meets the wall is the angle the beam is
            # laid down at, and the referee wants it within 10 deg of its line.
            # A 12 mm residual is 12 deg, which is a fail on yaw alone.
            rem = abs((line[0] - x)*np.cos(t) + (line[1] - y)*np.sin(t))
            taper = float(np.clip(rem / taper_mm, 0.0, 1.0))
            aim = hold_heading - np.clip(gain*lat, -crab_max, crab_max) * \
                  taper * (1.0 if v > 0 else -1.0)
        # 3.5, not 2.2: the crab angle has to be REACHED, not approached.  At
        # 2.2 the robot spent the first 80 mm of a 150 mm run-in still turning
        # onto its crab, corrected 16 of the 33 mm it needed, and put beam 1
        # down 35 mm north of its line.
        # +/-60 deg/s, not 26.  A crab leg lasts about half a second and at
        # 26 deg/s the chassis is still turning onto its crab when the leg
        # ends -- 85 mm of travel bought 9 mm of cross-track instead of 32.
        # 60 deg/s at 200 mm/s is a 190 mm radius, well inside what the
        # drivetrain does elsewhere in the route.
        rb.drive(v, np.clip(3.5*_wrap(aim - th), -60, 60))
        yield
    rb.stop(); return False



def line_drive(rb, v, head, line, dist_mm, side=None):
    """Cover `dist_mm` along the approach axis while converging onto it.

    Same law as stall_drive's run-in, distance-terminated instead of
    torque-terminated.  Used to back out to the staging point: the robot has to
    arrive on the line, not merely parallel to it, because the wall stall only
    fixes the coordinate ALONG the line.
    """
    x0, y0, _ = rb.pose
    t = np.radians(head)
    while True:
        x, y, th = rb.pose
        if np.hypot(x-x0, y-y0) >= dist_mm:
            rb.stop(); return
        lat = -(x - line[0])*np.sin(t) + (y - line[1])*np.cos(t)
        aim = head - np.clip(1.0*lat, -22.0, 22.0) * (1.0 if v > 0 else -1.0)
        # One-sided, when the caller says so: with a beam already on the field
        # the crab is safe on one side of the approach and a collision on the
        # other (see dress_safe).  Give up the correction rather than take it
        # from the wrong side -- this is a transit, and the run-in that follows
        # is what actually has to be on the line.
        if side is not None and side*_wrap(aim - head) < 0.0:
            aim = head
        # +/-60 deg/s, not 26.  A crab leg lasts about half a second and at
        # 26 deg/s the chassis is still turning onto its crab when the leg
        # ends -- 85 mm of travel bought 9 mm of cross-track instead of 32.
        # 60 deg/s at 200 mm/s is a 190 mm radius, well inside what the
        # drivetrain does elsewhere in the route.
        rb.drive(v, np.clip(3.5*_wrap(aim - th), -60, 60)); yield


def dress_onto_line(rb, head, line, net_mm=0.0, tol=10.0, phi=12.0, leg=75.0,
                    passes=7):
    """Shuffle sideways onto an approach line.  A differential-drive park.

    A crabbing run-in clears about a third of its length in cross-track, so
    beam 1's 75 mm lateral shift wants 220 mm of straight line.  There is not
    220 mm: at heading 180 the robot may not go east of Xa 249 (its rear ball
    transfers reach the laboratory edge and climb it -- measured, a crabbing
    shuffle beached itself there) nor much west of Xa 170 (its nose reaches the
    wall).  Nor does crabbing work in short legs: at 200 mm/s a 45 mm leg lasts
    0.22 s, the chassis turns 6 deg in that time, and eight legs moved the robot
    4 mm sideways in total.

    So establish the crab angle FIRST, in place, then drive it: turn phi off the
    line, run a leg, turn phi the other way, run back.  Each pair is a pure
    lateral translation of 2 * leg * sin(phi) and no net travel along the line.

    phi is 12 deg, not 30, and the lane is X 190-330 rather than 185-254.  Both
    are set by the beam already on the field: at 30 deg the chassis's rear
    pocket-side corner swings 170 mm south-east of the axle, and on the western
    lane that lands it inside beam 2 -- it shoved the placed beam 45 mm and
    yawed it 24 deg.  Kept east of X 190 the same corner clears beam 2's east
    face whatever the heading, and 12 deg keeps the rear ball transfers west of
    the laboratory edge they cannot climb.
    """
    for i in range(passes):
        x, y, _ = rb.pose
        t = np.radians(head)
        lat = -(x - line[0])*np.sin(t) + (y - line[1])*np.cos(t)
        if abs(lat) < tol:
            break
        fwd = (i % 2 == 0)
        a = min(phi, max(8.0, abs(lat)))          # ease off as it closes
        aim = head - np.sign(lat)*a*(1.0 if fwd else -1.0)
        yield from guard(turn_to(rb, aim, tol=2.5), 5.0)
        step = min(leg, abs(lat)/max(np.sin(np.radians(a)), 0.1))
        yield from guard(drive_straight(rb, step if fwd else -step,
                                        speed=200.0), 4.0)
    yield from guard(turn_to(rb, head, tol=1.5), 7.0)


def dress_safe(rb, head, line, side=1.0, tol=5.0, phi=12.0, leg=80.0,
               passes=9, recock=True):
    """Shuffle onto a line WITHOUT EVER CRABBING TO THE WRONG SIDE OF IT.

    dress_onto_line alternates the crab angle either side of the approach
    heading.  That is free on an empty field and fatal once something is
    standing on it, because the robot's cargo does not sit on the axle: beam 1
    is carried 137 mm ahead of the pivot and 107 mm to the left, so a few
    degrees of yaw swing its mid-section a long way in Y.  Measured, at the
    moment beam 1 is being lined up:

        heading    clearance from the carried beam's south face
                   to beam 2's north end face, robot on the line
          192          +23 .. +7 mm     (lifts away)
          180           +3.8 mm         (the design clearance)
          173           -10 .. -1 mm    (through it)
          168           -21 .. -5 mm

    -- so the crab is safe on one side and a collision on the other, and
    dress_onto_line spends half its legs on the wrong one.  On seed 6 that
    dragged the beam already placed 20 mm east and 9 deg round.  Nothing looks
    wrong at the time: beam 2 still scores its +25, and the T-joint it no longer
    makes is not read until the buzzer.

    So pin the crab to the safe side and get the other direction of travel by
    REVERSING instead of by crabbing the other way -- at head + side*a, forward
    moves the robot one way across the line and reverse moves it back.  Each
    cycle then squares up and runs its own along-track travel back off, because
    four descending legs in a row would otherwise walk the carried beam into
    the west wall.
    """
    t = np.radians(head)
    for _ in range(passes):
        x, y, _ = rb.pose
        lat = -(x - line[0])*np.sin(t) + (y - line[1])*np.cos(t)
        if abs(lat) < tol:
            break
        # Forward at head + side*a changes lat by +side*sin(a) per mm travelled,
        # reverse by -side*sin(a).  The crab angle never changes sign; only the
        # gear does.
        a    = min(phi, max(6.0, abs(lat)))
        fwd  = (lat < 0.0) == (side > 0.0)
        step = min(leg, abs(lat)/max(np.sin(np.radians(a)), 0.1))
        yield from guard(turn_to(rb, head + side*a, tol=2.0), 5.0)
        yield from guard(drive_straight(rb, step if fwd else -step,
                                        speed=200.0), 4.0)
        # Square up BEFORE running the station back.  The return leg is the one
        # part of the cycle that has to be flat: it is driven at the approach
        # heading, where the design clearance is 3.8 mm and nothing is spare.
        #
        # recock=False leaves the along-track travel where it fell.  That halves
        # the cycle -- one turn and one leg instead of two of each, 2.6 s against
        # 4 -- and it is right whenever the caller has something else that will
        # put the station back, which place_beam does: it line_drives to its own
        # staging point before the run-in anyway.
        yield from guard(turn_to(rb, head, tol=1.5), 5.0)
        if recock:
            back = step*np.cos(np.radians(a))
            yield from guard(drive_straight(rb, -back if fwd else back,
                                            speed=200.0), 4.0)
    yield from guard(turn_to(rb, head, tol=1.5), 6.0)

def place_beam(rb, which, log=print, clk=None, withdraw=0.0,
               withdraw_line=None, back=170.0, station=None, crab=12.0,
               gain=1.0, taper_mm=70.0, side=None):
    """Set one beam down against its wall and back away from it."""
    lap = (lambda w: None) if clk is None else \
          (lambda w: log("        %-16s T+%5.1f  axle (%6.1f,%6.1f,%6.1f)"
                         % ((w, clk()) + rb.pose)))
    st  = station or (AgentA.BEAM1_STATION if which == 1
                      else AgentA.BEAM2_STATION)
    ax, ay, head = st
    # Stage on the approach axis, one robot length back, then run it in.
    #   beam 1 approaches westward  (forward, rear stop presses)
    #   beam 2 approaches southward (reverse,  front stop presses)
    fwd = (which == 1)
    t = np.radians(head)
    sx, sy = (ax - back*np.cos(t), ay - back*np.sin(t)) if fwd else \
             (ax + back*np.cos(t), ay + back*np.sin(t))
    log("      beam %d: staging at (%.0f, %.0f) heading %.0f" % (which, sx, sy, head))
    # THE CALLER OWNS THE NAVIGATION AND, IN PARTICULAR, OWNS THE PIVOTS.
    # Turning at the staging point is what wrecked the first working version:
    # beam 1's staging sits 121 mm from beam 2's north end, inside the 185 mm
    # loaded swept circle, so the pivot swept the beam that had just been placed
    # 60 mm across the field.  By the time place_beam is called the robot is on
    # the station heading and behind the staging point; from here on it is a
    # straight line and a wall.
    yield from guard(turn_to(rb, head, tol=1.5), 10.0)
    px, py, _ = rb.pose
    t = np.radians(head)
    along = (sx - px)*np.cos(t) + (sy - py)*np.sin(t)     # + is toward the nose
    if abs(along) > 15.0:
        yield from guard(line_drive(rb, 200.0 if along > 0 else -200.0, head,
                                    (ax, ay), abs(along), side=side), 12.0)
    lap("staged")
    # Run in and stall on the wall.  The beam is still carried clear, so the
    # piece that meets the wall is the beam's own end face, not the chassis.
    ok = yield from guard(stall_drive(rb, 120.0 if fwd else -120.0, head,
                                      max_mm=back + 120.0, line=(ax, ay),
                                      crab_max=crab, gain=gain,
                                      taper_mm=taper_mm), 16.0)
    lap("wall stall")
    # STRAIGHTEN AND RESEAT.  The run-in crabs to kill its cross-track, and
    # whatever crab is left when the beam meets the wall is the angle the beam
    # is laid down at -- 5 deg of it swings a 250 beam's far end 12 mm sideways,
    # which is enough to miss the T-joint even with both beams inside their
    # own tolerances.  So back off, square up, and come in again with the crab
    # switched off.  Three seconds, and the beam lands parallel to its wall.
    if abs(_wrap(head - rb.pose[2])) > 1.5:
        yield from guard(drive_straight(rb, -30.0 if fwd else 30.0, speed=120.0), 3.0)
        yield from guard(turn_to(rb, head, tol=1.0), 6.0)
        yield from guard(stall_drive(rb, 90.0 if fwd else -90.0, head,
                                     max_mm=70.0), 6.0)
        lap("squared")
    # SET IT DOWN WHILE STILL PRESSING.  Stopping first lets the chassis rebound
    # off the wall it is leaning on -- 25 N of contact, and the beam is still on
    # the shelves, so it comes back east with the robot and lands 5 mm out.
    # Holding a light push through the drop pins the beam against the wall until
    # it is standing on the field.
    # Square up first, then release with ZERO yaw command.  The beam inherits
    # the chassis's angular velocity at the instant the clamp lets go, and a
    # heading-correction term running through the release hands it 3-4 deg/s --
    # which it keeps until floor friction stops it, several degrees later.  A
    # 250 beam yawed 4 deg has its far end 9 mm out of place, and that is the
    # T-joint.  Straight push, no steering.
    yield from guard(turn_to(rb, head, tol=0.8), 5.0)
    rb.cradle(which, False)
    push = 35.0 if fwd else -35.0
    for _ in range(int(1.0*HZ)):
        rb.drive(push, 0.0); yield
    rb.stop()
    yield from wait(rb, 0.4)
    lap("set down")
    # Read the datum HERE, not at the first touch: the seating push moves the
    # beam a further 2-3 mm, and 3 mm is the whole T-joint tolerance.
    rb.beam_stall = getattr(rb, "beam_stall", {})
    rb.beam_stall[which] = rb.pose
    yield from guard(drive_straight(rb, -AgentA.BEAM_BACKOFF if fwd
                                    else AgentA.BEAM_BACKOFF, speed=90.0), 4.0)
    # The cradle STAYS DOWN.  Its shelves run the length of the beam, so
    # raising them after a 45 mm back-off would simply pick the piece up again;
    # left down they sit in the floor plane, under the beam, touching nothing.
    yield from wait(rb, 0.4)
    lap("released")
    # WITHDRAW ALONG THE APPROACH AXIS.  A placed beam is a 60 mm obstacle and
    # the loaded swept circle is 185 mm, so until the robot is that far from it
    # there is no legal pivot -- and after the release it is 95 mm away.  Every
    # heading change has to wait for this leg.
    if withdraw > 0:
        # Crab back onto a lane the next pivot can use.  Straight out along the
        # station line leaves the robot 182 mm from the west wall, and the
        # loaded swept circle is 185 -- so the pivot after it would put a beam
        # end through the wall.
        wl = withdraw_line or (ax, ay)
        yield from guard(line_drive(rb, -200.0 if fwd else 200.0, head, wl,
                                    withdraw), 9.0)
        lap("withdrawn")
    x, y, th = rb.pose
    log("      beam %d released, axle at (%.1f, %.1f, %.0f) [target %.1f, %.1f, %.0f]"
        % (which, x, y, th, ax, ay, head))


def seal_quarantine(rb, log=print, clk=None):
    """Both beams, in the only order and along the only lanes that work.

    Every pivot below is at a point checked against the 185 mm loaded swept
    circle, the walls, the 6 mm laboratory (which the rolling contacts cannot
    climb, F33) and whichever beam is already down.  There are exactly two
    such places in the quarantine half of the field, and they are both on the
    x ~ 200 lane north of the corner -- which is why the route looks indirect.
    """
    t = (lambda: "") if clk is None else (lambda: "T+%5.1f  " % clk())
    log(t() + "sealing the quarantine: beam 2 (south wall) first")
    # Turn onto the bearing BEFORE pursuing.  pursue() refuses a target more
    # than 100 deg off the nose and returns instantly, and after the last
    # laboratory dock the robot faces south with the quarantine behind its
    # shoulder -- so both approach legs used to no-op and place_beam was left
    # to find the station from 400 mm away with a crab controller.  That cost
    # 28 s of a 120 s match.
    # Arrive ON beam 2's approach lane, not near it.  The run-in is 220 mm and
    # the crab is capped at 12 deg on it, which is worth about 45 mm of
    # cross-track; a pivot 40 mm off the lane spends all of that and still puts
    # the beam down 67 mm east.  X 200 is the westmost pivot the loaded swept
    # circle allows (188.6 mm from the wall), and it is 22 mm off the lane.
    # TURN, THEN RUN STRAIGHT.  pursue() steers continuously toward a point and
    # from the far side of the field it circles it instead of arriving: from
    # the third laboratory slot it spent its whole 18 s guard and finished
    # 76 mm out and 37 deg off, which then cost another 20 s of shuffling.  A
    # heading hold over a measured distance lands within a few millimetres and
    # takes 2.6 s.  The waypoint is Y 230 rather than 300 so the run stays
    # south of the laboratory -- at 300 the north wheel tracks its edge -- and
    # X 192 rather than 200 because everything left over is shuffled off by
    # hand afterwards at about 3 s per leg.  192 is as far west as the pivot
    # can go: the loaded swept circle is 187 mm and the wall is at zero.
    WP = (192.0, 230.0)
    # GET OFF THE LABORATORY FIRST.  A dock that fails to depart leaves the
    # robot at Y 293 with a drive wheel against the plate edge, where it can
    # neither turn nor drive -- and the beam phase then spends 24 s discovering
    # that.  The robot is always facing away from the slot at this point, so
    # driving forward is driving south; no pivot needed, which is the only
    # thing that works from there.
    px, py, pth = rb.pose
    if py > 240.0 and abs(_wrap(pth - 270.0)) < 70.0:
        log("      still on the laboratory at Y %.0f -- driving clear" % py)
        yield from guard(drive_straight(rb, py - PIVOT_Y, speed=200.0), 8.0)
        yield from guard(drive_straight(rb, py - PIVOT_Y, speed=200.0), 6.0)
    px, py, _ = rb.pose
    log("      transit: from (%.0f, %.0f, %.0f) to the lane" % rb.pose)
    yield from guard(turn_to(rb, np.degrees(np.arctan2(WP[1]-py, WP[0]-px)),
                             tol=3.0), 9.0)
    log("      turned:  (%.0f, %.0f, %.0f)" % rb.pose)
    yield from guard(drive_straight(rb, np.hypot(WP[0]-px, WP[1]-py),
                                    speed=220.0), 12.0)
    log("      ran in:  (%.0f, %.0f, %.0f)" % rb.pose)
    yield from guard(turn_to(rb, AgentA.BEAM2_STATION[2], tol=2.0), 9.0)
    # Then shuffle the last 20-140 mm onto the lane.  How far off the pivot
    # leaves the robot depends on how far it came -- from the third slot it
    # arrives 140 mm out -- and beam 2's 220 mm run-in cannot absorb that.
    # The same park that gets the robot onto beam 1's line gets it onto this
    # one, and here there is nothing placed yet to swing into.
    # tol 5, not 10.  The run-in crabs to kill whatever cross-track is left,
    # and it ends by driving the beam into a wall -- so whatever crab is still
    # on at the stall is the angle the beam gets laid down at.  10 mm of
    # residual is 10 deg of yaw, and the referee wants the beam within 10.
    # tol 18, not 5.  This used to shuffle the last 15 mm onto the lane by hand
    # at 5.6 s a time, and it does not have to: place_beam line_drives 130 mm to
    # its own staging point before the run-in, crabbing onto the lane as it goes,
    # and the run-in itself is 220 mm with a proper line follower on it.  The
    # shuffle is a safety net for a pivot that lands badly, not the way the robot
    # is supposed to get onto its line.
    yield from guard(dress_onto_line(rb, AgentA.BEAM2_STATION[2],
                                     AgentA.BEAM2_STATION[:2],
                                     tol=18.0, passes=2), 8.0)
    # HOW FAR TO WITHDRAW IS THE WHOLE COST OF THE NEXT PHASE.  The pivot after
    # this one has to clear beam 2, and every millimetre of over-retreat is a
    # millimetre beam 1's approach has to be parked back down again -- at about
    # 6 s per 40 mm, because the park is a shuffle and not a drive.
    #
    # Derived rather than padded.  Turning at (177.5, y) with beam 1 still
    # aboard, the binding radius is the carried beam's far corner, 184.7 mm at
    # 99.4 mm of x-offset from beam 2's north-west corner, so y >= 406.  The
    # release leaves the axle at 187.5, hence 222 with 4 mm to spare.  It was
    # and loses the T-joint: the pivot then clears beam 2 by 4 mm, not 12.
    yield from place_beam(rb, 2, log=log, clk=clk, withdraw=230.0, back=220.0,
                          gain=1.2, taper_mm=45.0)
    log(t() + "beam 1 (west wall)")
    # DERIVE BEAM 1's LINE FROM WHERE BEAM 2 ACTUALLY LANDED (F54).
    # Beam 2's north end face is at its stall axle + STOP2_X, and beam 1 has to
    # butt that face.  A fixed line cannot do it: the wall stall leaves the beam
    # 0-3 mm off the wall depending on how hard it arrived, and 3 mm is the
    # whole T-joint tolerance.  Worse, the pocket-L end stop rides on beam 1's
    # own south face, so a line 2 mm too far south does not merely open the
    # joint -- it drives the stop through the beam already on the field.
    # The robot knows the number: it stalled there.
    # ...minus a measured stall-release offset.  The raw estimate assumes the
    # beam's north face is ON the stop face at the stall, and it is not quite:
    # by the time the beam is standing free it has settled 2 mm south of that
    # (measured +2.0 and +2.1 on seeds 1 and 8, F64 -- the heavier nose changed
    # the stall micro-dynamics and every T-joint opened to 3-4 mm).  Beam 1's
    # own placement lands within 0.5 mm of its line, so this bias was the whole
    # T-joint regression.
    STALL_RELEASE = 2.0
    st2 = getattr(rb, "beam_stall", {}).get(2)
    n_end = (st2[1] + AgentA.STOP2_X - STALL_RELEASE) if st2 else Piece.BEAM2_L
    line1 = (AgentA.BEAM1_STATION[0], n_end + AgentA.POCKET_Y + Piece.BEAM_W/2.0 + 1.0)
    log("      beam 2's north face read at Y %.1f -> beam 1 line Y %.1f"
        % (n_end, line1[1]))
    # COME DOWN THE DIAGONAL, DO NOT SHUFFLE DOWN.  The robot is now at
    # (194, 458) and beam 1's line is 87 mm south of that: a lateral shuffle
    # costs 1.9 s per 17 mm and there is no 10 s to spend.  Reversing at 150 deg
    # covers both axes at once -- 113 mm of travel is 98 east and 57 south --
    # and lands on the staging X with about 30 mm still to come off, which is
    # what the run-in's crab is for.
    #
    # 150 IS A FLOOR, NOT A CHOICE.  Steeper diagonals land closer to the line
    # and drag the carried beam's trailing half through beam 2 on the way:
    # measured at the same start, 147.5 clears by 1.6 mm, 145 by -7, 140 by -25.
    # Shallower ones are safe but leave more for the run-in than it can take.
    # STEP EAST BEFORE ANY PIVOT.  Beam 2's lane is Xa 177.5 and the CHASSIS
    # half-diagonal is 184.7, so from the withdrawal point the robot is boxed
    # into headings 55.6 .. 124.4 and cannot reach 180 by turning at all, in
    # either direction -- a front corner goes 7.2 mm inside the west wall at
    # heading 140.  The carried beam is not even involved; this pivot was never
    # legal, loaded or empty.
    #
    # turn_to hides it.  It sees a pivot that is not happening, backs off 45 mm
    # along its own axis and tries again, and on seed 7 that back-off drove the
    # carried beam into the beam just placed: 10 mm east, 5 deg round, T-joint
    # gone.  The score said +100 and nothing in the log said why.
    #
    # Forward at 65 deg is the cheapest heading in the box that gains X, and
    # 44 mm of it puts the axle at (196, 462), where the pivot clears the wall
    # by 8 mm and beam 2 by 10.
    px, py, _ = rb.pose
    yield from guard(turn_to(rb, 65.0, tol=2.5), 8.0)
    yield from guard(drive_straight(rb, float(np.clip(
        (196.0 - px)/np.cos(np.radians(65.0)), 25.0, 90.0)), speed=180.0), 5.0)
    log("      stepped east to (%.0f, %.0f) -- the pivot is legal there" % rb.pose[:2])
    STAGE_X = 292.0                       # the lab plate caps it; see below
    px, py, _ = rb.pose
    tgt = (STAGE_X, line1[1] + 30.0)
    h2  = float(np.clip((np.degrees(np.arctan2(tgt[1]-py, tgt[0]-px)) + 180.0) % 360.0,
                        150.0, 176.0))
    yield from guard(turn_to(rb, h2, tol=2.5), 9.0)
    px, py, _ = rb.pose
    run = (STAGE_X - px)/max(0.2, -np.cos(np.radians(h2)))
    yield from guard(drive_straight(rb, -float(np.clip(run, 30.0, 220.0)),
                                    speed=180.0), 7.0)
    log("      down the diagonal at %.0f deg to (%.0f, %.0f)" % ((h2,) + rb.pose[:2]))
    # Safety net only.  If the diagonal lands badly the shuffle picks it up, and
    # it is the ONE-SIDED shuffle: beam 2 is on the field and the carried beam
    # only clears its north end face by 3.8 mm, so crabbing below 180 puts the
    # cargo through the piece already placed (see dress_safe).
    # ...and shuffle it down to within 3 mm FIRST.  The run-in's crab is a
    # cascaded loop -- crab angle commanded from cross-track, yaw rate commanded
    # from crab angle -- and the inner loop's 0.29 s makes it sluggish, not
    # tight: from 11 mm out it cleared 7 and laid beam 1 down 4 mm high, which
    # is a 7 mm air gap at the T-joint.  Winding the gain up to 2.5 did not fix
    # it, it made it overshoot: seed 6 went 6 mm PAST the line and drove the
    # beam into beam 2, losing both.  So do not ask the run-in for authority it
    # has not got.  Two shuffle cycles cost 5 s and hand it a 3 mm error, which
    # is inside what it can actually clear.
    yield from guard(dress_safe(rb, 180.0, (line1[0], line1[1] + 2.0),
                                side=+1.0, tol=4.0, phi=20.0, leg=115.0,
                                passes=3, recock=False), 10.0)
    # Stage no further east than Xa 292.  At heading 180 the shell's FORWARD
    # section -- the part at Za 6, which cannot ride over the 6 mm laboratory --
    # reaches axle + 52.5, and the plate starts at 351.5.  The first version
    # staged at 312 and ploughed the plate edge for the whole run-in, which
    # yawed beam 1 by 8 deg and left it 28 mm out.
    #
    # crab 20, not 12.  The run-in now arrives ~30 mm NORTH of the line and has
    # 150 mm to take it off; at 12 deg that is 24 mm of authority and beam 1
    # lands 6 mm high, which is most of the T-joint budget.  Twenty is free
    # here: the correction is northward, so the crab tips the carried beam AWAY
    # from beam 2, and the taper still squares it up before the wall.
    yield from place_beam(rb, 1, log=log, clk=clk, withdraw=60.0, back=150.0,
                          station=(line1[0], line1[1], 180.0), crab=24.0,
                          gain=1.2, taper_mm=45.0, side=+1.0)
