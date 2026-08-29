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
from .params import Chassis, AgentA, Field, Piece

HZ = 50.0
MATCH = 120.0            # rules g.1
HOLE_BUDGET = 17.0       # measured cost of one reverse dock plus its post
# 56, and this was measured rather than guessed.  At 44 the laboratory gets a
# second slot -- worth +18 -- and the seal then starts 15 s later and misses:
# over twelve seeds the mean went from +69 to about +50, with four matches
# running past the buzzer with no beams down at all.  The seal is 70 points
# and it is the phase with no slack, so it gets the clock it needs first.
BEAM_BUDGET = 39.0       # measured cost of the two-beam seal from the lab
# 13, which is what a dock MEASURES: 11-16 s from arriving at the pivot to
# being clear of the plate again.  At 9 the robot starts approaches it cannot
# finish -- one match began its third slot at T+66, left the plate at T+84,
# and the seal then ran 7 s past the buzzer and scored nothing.  A slot is
# worth 18 and the seal 70; when they compete, the seal wins.
MIN_DOCK    = 11.0       # floor on the estimate; a clean dock is 11-12 s
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


def turn_to(rb, heading, tol=2.0, wmax=110.0, free=True):
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
    n, steady = rb.mag_count(), 0.0
    for _ in range(int(cap * HZ)):
        rb.stop()
        c = rb.mag_count()
        # If everything that exists is already aboard there is nothing left to
        # wait for.  The robot knows the match has three samples in it, and the
        # bore rangefinder tells it how many it is holding, so a pass that
        # swept the lot can leave the instant the last one seats.  Worth 4-5 s,
        # and it costs nothing when the pass was not that lucky.
        if want is not None and c >= want:
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
    while rb.pose[0] > x_to:
        th = rb.pose[2]
        lat = np.clip(1.6*(y - rb.pose[1]), -20, 20)     # hold the line
        rb.drive(speed, np.clip(2.0*_wrap(180.0-th) - lat, -22, 22)); yield
    rb.stop()
    yield from dwell_until_loaded(rb, want=want)


def align_reverse(rb, chute_offset, tx, ty, heading, tol=2.5, max_ticks=900):
    """Back the CHUTE onto a target, steering on its measured cross-track error.

    Docking cannot be dead-reckoned here: the rear ball transfers ride up the
    3 mm lab plate (the spec's own [VERIFY 10.2] question), which pitches the
    chassis and walks the chute several mm.  Disc-in-hole radial clearance is
    only 2 mm, so the terminal has to close on the chute's actual position.
    """
    import os
    _dbg = os.environ.get("DOCK_DEBUG")
    _n = 0
    for _ in range(max_ticks):
        _n += 1
        px, py, th = rb.pose
        cx, cy = rb.chute_xy(chute_offset)
        t = np.radians(th)
        # error of the CHUTE, in the robot frame: +fore is toward the nose
        ex, ey = tx - cx, ty - cy
        fore =  ex*np.cos(t) + ey*np.sin(t)
        left = -ex*np.sin(t) + ey*np.cos(t)
        # Only accept the dock when the chassis is actually STOPPED.  Accepting
        # mid-coast books an error the robot is about to move away from.
        if abs(fore) < tol and abs(left) < tol and \
                np.linalg.norm(rb.d.qvel[:2])*1000.0 < 3.0:
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
            # CREEP AND SETTLE, not continuous control.  The drive is a pair of
            # steppers -- position devices -- and the chassis carries enough
            # momentum that commanding 5 mm/s still leaves it coasting at 13,
            # so a continuous terminal hunts around the target and never lands
            # inside 2 mm.  Move in short bursts and let it stop between them:
            # the error is then measured on a stationary robot, which is the
            # only way a 2 mm budget is meetable.
            w = np.clip(-3.2*left, -10, 10)
            v = np.clip(fore*2.0, -35, 35) if abs(fore) > tol else 0.0
            # Creep only in the LAST few millimetres.  Applied across the whole
            # endgame the duty cycle spends 60% of the time stopped, which fixed
            # the seeds that were hunting and pushed the marginal ones over the
            # match clock instead.
            if abs(fore) < 8.0 and abs(left) < 8.0 and _n % 16 >= 9:
                v, w = 0.0, 0.0
        if _dbg and _n % 25 == 0:
            import mujoco as _mj, numpy as _np
            hits, _f = {}, _np.zeros(6)
            for _c in range(rb.d.ncon):
                _co = rb.d.contact[_c]
                n1 = _mj.mj_id2name(rb.m, _mj.mjtObj.mjOBJ_GEOM, _co.geom1) or ""
                n2 = _mj.mj_id2name(rb.m, _mj.mjtObj.mjOBJ_GEOM, _co.geom2) or ""
                for a, b in ((n1, n2), (n2, n1)):
                    if a.startswith("A_") and not b.startswith("A_") and b != "floor":
                        _mj.mj_contactForce(rb.m, rb.d, _c, _f)
                        hits[(a, b)] = max(hits.get((a, b), 0.0), abs(_f[0]))
            _b3 = _mj.mj_name2id(rb.m, _mj.mjtObj.mjOBJ_GEOM, "A_ball3")
            _by = rb.d.geom_xpos[_b3][1]*1000 if _b3 >= 0 else 0.0
            print("        [dock] fore=%7.1f left=%7.1f herr=%6.1f v=%6.1f w=%6.1f "
                  "|vel|=%5.2f ball3_y=%6.1f  %s"
                  % (fore, left, herr, v, w, _np.linalg.norm(rb.d.qvel[:2])*1000, _by,
                     ", ".join("%s/%s %.0fN" % (a, b, f) for (a, b), f in
                               sorted(hits.items(), key=lambda kv: -kv[1])[:3]) or "-"))
        rb.drive(v, w)
        yield
    rb.stop()
    return False


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


def dock_and_post(rb, hole_x, hole_y, chute_offset, stroke=0.60, aboard=0,
                  depart=None, log=print, clk=None, deadline=None):
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
        yield from guard(turn_to(rb, np.degrees(np.arctan2(pv[1]-py, pv[0]-px))), 9.0)
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
    over = (lambda: clk is not None and deadline is not None and clk() > deadline)
    for attempt in range(3):
        # A dock that is still hunting when the clock runs out has to stop
        # hunting: the beam phase behind it is worth 70 points and the slot in
        # front is worth 18.  Give up the approach, keep the departure -- the
        # robot must still get off the plate whatever happens.
        if over() and attempt:
            log("      hole abandoned mid-approach at the beam deadline")
            break
        px, py, _ = rb.pose
        th = np.degrees(np.arctan2(py - hole_y, px - hole_x))   # face away from hole
        yield from guard(turn_to(rb, th, tol=1.2), 9.0)
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
    # Re-seat before metering ONLY if the bore disagrees with what should be
    # aboard.  When it already reads the full count the stack is demonstrably
    # flat and the stroke is 1.2 s of a 120 s match for nothing -- 3.6 s over
    # three slots, which is most of a fourth dock.
    if not aboard or rb.mag_count() < aboard:
        yield from reseat(rb)
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

    def _stack(tag):
        if not os.environ.get("ESC_DEBUG"):
            return
        import mujoco as _mj
        out = []
        for _i in range(3):
            _b = _mj.mj_name2id(rb.m, _mj.mjtObj.mjOBJ_BODY, "disc%d" % _i)
            if _b < 0:
                continue
            _p = rb.to_local(rb.d.xpos[_b])
            out.append("d%d(%6.1f,%5.1f,%5.1f)" % (_i, _p[0], _p[1], _p[2]))
        log("        [esc %-9s] %s" % (tag, "  ".join(out)))

    _stack("before")
    rb.blade(n >= 2)
    yield from wait(rb, 0.5)
    _stack("blade in")
    # HOLD THE SHELF OPEN LONG ENOUGH FOR THE PIECE TO CLEAR IT (F41).  At 0.28 s
    # the released disc was still at Za 7.2 -- barely below the shelf line at 8 --
    # when the shelf came back, and the returning shelf caught it and swept it
    # 10.7 mm sideways, where it jammed half in the bore and was then dragged
    # along by the departure.  It does not fall freely: the retainer's knife lip
    # rests on it, so it is released rather than dropped.
    rb.gate(True)
    yield from wait(rb, stroke)
    _stack("gate out")
    # ...AND LEAVE IT OPEN UNTIL THE ROBOT HAS LEFT THE SLOT (F55).  Closing it
    # here is what bolts the robot to the laboratory: a disc that perches on the
    # slot's countersink instead of dropping through ends up pinched between
    # that countersink and the returning shelf, 4 N on the shelf and 3 N on the
    # cone, and the robot cannot then drive off in ANY direction -- rocking does
    # not free it.  It still scores, and everything after it is lost.
    # With the retainer in, the column is already supported without the shelf,
    # so there is no reason to close it until the chute is clear of the slot.
    # (On the last disc there is nothing left in the bore to hold up at all.)
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
    # F55.  THE ROBOT CAN BE BOLTED TO THE LABORATORY BY ITS OWN SAMPLE.  A disc
    # that perches on a slot's countersink instead of dropping through ends up
    # pinched between that countersink and the escapement shelf directly above
    # it -- measured at 4 N on the shelf and 3 N on the cone -- and the robot
    # then cannot drive off in any direction.  It still SCORES (it is inside
    # the slot), but everything downstream is dead: one observed match lost the
    # entire beam phase to it, and the symptom looked like a failed pivot 20 s
    # later and half a field away.
    #
    # Rocking frees it.  Reversing slides the shelf off the piece the other
    # way, and the second attempt then leaves cleanly.  Three tries, and the
    # last of them re-aims, because if it is still stuck after that the robot
    # is not stuck on a disc.
    for _ in range(3):
        if rb.pose[1] <= PIVOT_Y + 35.0:
            break
        log("      departure blocked at Y %.0f -- rocking free" % rb.pose[1])
        yield from guard(drive_straight(rb, -45.0, speed=140.0), 4.0)
        yield from guard(drive_straight(rb, rb.pose[1] - PIVOT_Y + 45.0,
                                        speed=200.0), 7.0)
    # Clear of the slot -- now it is safe to put the shelf back under the column.
    rb.gate(False)
    yield from wait(rb, 0.4)
    _stack("gate back")
    rb.blade(False)
    yield from wait(rb, 0.5)
    _stack("blade out")
    lap("departed")


def mission_agent_a(rb, holes, hole_y, chute_offset, log=print, clock=None):
    # THE MATCH IS 120 s (rules g.1).  Every phase is stamped so the budget
    # is visible in the log, not discovered at the end.
    t = (lambda: "") if clock is None else (lambda: "T+%5.1f  " % clock())
    log(t() + "leaving the deployment box nose-first (no pivot: swept R 185 > 140 to the wall)")
    yield from guard(drive_straight(rb, 300.0, speed=220.0), 10.0)
    # TWO LANES, BUT NOT THESE TWO -- AND THE NORTH ONE FIRST.
    #
    # Samples are randomised in Y 80 .. 230 (rules 2.1, and random_discs).  The
    # sweeper takes a piece whose centre is within about 55 mm of its lane -- the
    # finger tips are at +/-82.5 and the disc is O56, so beyond ~55 the centre is
    # outside the tips and the piece is not funnelled, it is SHOVED.  The chassis
    # is 235 wide, so each pass carries 62 mm of bulldozer beyond each side of
    # what it can actually collect.
    #
    # Lanes at 130 and 215 covered the band between them, but the first pass
    # moved pieces the second one was relying on: seed 3's sample went
    # (60, 226) -> (37, 273), out of the quarantine AND into beam 1's footprint,
    # so the beam landed on it and lost its own +25 too.  Seeds 5 and 12 lost one
    # the same way.  Three of twelve matches, all the same 62 mm.
    #
    # Y 178 then Y 120 is the pair where that cannot happen:
    #     pass 1 collects 123 .. 233  -- its north bulldozer band starts ABOVE
    #                                   the highest a sample can be, so nothing
    #                                   is ever shoved out of the quarantine
    #     pass 2 collects  65 .. 175  -- and picks up whatever pass 1 pushed
    #                                   south, which is the only way pass 1 can
    #                                   push anything
    # Between them they cover 65 .. 233 with a 52 mm overlap, and every sample
    # that moves at all moves TOWARD the next pass.
    log(t() + "sweep pass 1, mouth on Y 178")
    yield from guard(pursue(rb, 430.0, 178.0, speed=220.0, tol=40.0), 20.0)
    yield from guard(sweep_line(rb, 178.0, 158.0, want=3), 45.0)
    # Second lane only if the bore says something is still out there.  When the
    # samples happen to fall in the north half this is the whole pass saved --
    # 10 s, and the beams need every one of them.
    if rb.mag_count() < 3:
        log(t() + "%d aboard -- second lane, mouth on Y 120" % rb.mag_count())
        # Back out to Y 195: the southernmost legal pivot with the beams aboard
        # is 184.7 (swept R against the south wall) and everything further north
        # is lane the next pass has to crab back off.  And DO NOT pursue() the
        # lane from there -- the target is ~90 mm away and the loaded turn radius
        # at sweep speed is 190, so pursue circles the point instead of arriving
        # (measured: 20 s of a 120 s match spent going round it).  sweep_line
        # holds its own lane; let it.
        yield from guard(back_to(rb, 470.0, 195.0), 22.0)
        yield from guard(turn_to(rb, 180.0), 12.0)
        yield from guard(sweep_line(rb, 120.0, 158.0, want=3), 45.0)
    log(t() + "settling the magazine")
    yield from guard(settle_stack(rb, want=len(holes)), 30.0)

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
    dl, est = MATCH - BEAM_BUDGET, MIN_DOCK
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
        moving = np.linalg.norm(rb.d.qvel[:2])*1000.0
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
    # The pivot lane.  Two hard limits set it: the robot's shell must stay west
    # of the laboratory's edge (Xa 351.5 - 97 = 254, because the forward shell
    # sits at Za 6 and the plate top is Za 6, and driving over it drags at 9 N),
    # and a pivot needs 185 mm from the west wall.  So every heading change in
    # the beam phase happens on X 185-254.  215 is the middle of it.
    LANE = 255.0
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
    st2 = getattr(rb, "beam_stall", {}).get(2)
    n_end = (st2[1] + AgentA.STOP2_X) if st2 else Piece.BEAM2_L
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
