"""Trajectory generation and tracking: transits that do not stop.

The profiler's verdict on the old route was ~75 s of repositioning in a
120 s match, and most of it is STRUCTURAL: every leg ends at v=0, pivots in
place (1.9-3.2 s each, ~20 a match), then accelerates from rest.  A
differential drive does not need any of that between stations -- it needs a
path whose corners it can take at speed, a tracker that steers while
moving, and a terminal that hands over to the station's own guarded moves
(wall stalls, camera docks) at the right pose and speed.

Two primitives, both generators in the route.py convention (one yield per
50 Hz tick), both consuming only rb.pose / rb.drive -- so they run on the
estimator's belief the day navigation goes honest:

  * track_waypoints -- pure pursuit along a polyline with corner speed
    caps: the carrot slides along the path, the chassis cuts corners with
    a radius it can actually drive, speed drops INTO a sharp corner and
    recovers out of it, and the leg ends at a controlled handoff speed.
    Replaces turn_to + drive_straight + turn_to chains on the long
    transits (the kit dogleg and climb above all).

  * capture_line -- the landing pattern: acquire an approach LINE, not a
    point.  From anywhere in a sensible entry cone, pursue a virtual
    target that slides along the line toward the station; convergence puts
    the chassis ON the line and SQUARE by construction, at an arrival
    speed the wall-stall run-in wants.  This is what replaces the beam
    approaches' turn-square-shuffle dances -- and it attacks the seal's
    top measured failure directly: a crabbed arrival presents the beam's
    corner to the wall, fires the stall early (11 mm short, 6 deg yawed,
    referee refuses), where a square arrival cannot.

The tuning constants are conservative and shared: yaw rate capped where
the chassis's own controllers cap it, lookahead proportional to speed with
floors that keep the pursuit stable at crawl.
"""
import numpy as np

W_MAX      = 60.0        # deg/s of steering while moving -- 190 mm radius at
                         # 200 mm/s, the same cap the crab laws already trust
PIVOT_W    = 140.0       # deg/s when a standing turn is unavoidable (entry
                         # far outside the pursuit cone)


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def track_waypoints(rb, pts, v_max=220.0, v_end=110.0, tol_end=30.0,
                    lookahead=0.55, reverse=False, strict=False):
    """Pure pursuit through pts = [(x, y), ...]; ends near the last point
    at ~v_end.  Corners are cut by the lookahead circle -- that is the
    point -- so waypoints are corridor knees, not poses to visit exactly.
    reverse=True drives TAIL-FIRST (back_to's steering law), keeping the
    chassis heading for a reverse terminal that follows -- the seal's
    descent uses it to arrive ready to press without an about-face.
    strict=True actually VISITS each knee (advance radius 45 mm instead of
    the lookahead): for corridor-critical knees whose whole point is the
    detour -- the seal's east knee exists to keep the tail off a patient
    sticker, and the lookahead cheerfully cut it and plowed the column.
    Returns True on arrival, False if progress stalled (caller decides)."""
    pts = [np.asarray(p, float) for p in pts]
    goal_i = 0                                  # the waypoint being pursued
    last_d, held = 1e9, 0
    pivoting = False
    while True:
        px, py, th = rb.pose
        p = np.array([px, py])
        end_d = float(np.linalg.norm(pts[-1] - p))
        if goal_i >= len(pts) - 1 and end_d < tol_end:
            return True
        # hand the pursuit the NEXT knee once this one is inside the
        # lookahead circle -- that is what rounds the corner.  (The first
        # version pursued the FINAL point from the start and beelined
        # diagonally across the laboratory plate.)
        v_now = max(v_end, min(v_max, 3.0*end_d))
        L = max(90.0, lookahead*v_now)
        adv = 45.0 if strict else L
        while goal_i < len(pts) - 1 and \
                float(np.linalg.norm(pts[goal_i] - p)) < adv:
            goal_i += 1
        tgt = pts[goal_i]
        # Slow INTO a coming corner, not merely inside it.  The corner
        # slowdown below keys on the CURRENT heading error, which a strict
        # knee hides until the target switches -- entering an 80-degree
        # knee at 210 mm/s is a 200 mm arc, and from the kit dogleg's knee
        # that arc carried the nose corner into the east wall and pinned it
        # there (measured: 15 s astride the watchdog's jitter floor).  The
        # turn at the next knee is knowable in advance; cap the approach.
        if goal_i < len(pts) - 1:
            seg = tgt - p
            d_knee = float(np.linalg.norm(seg))
            if d_knee < 240.0 and d_knee > 1e-6:
                a1 = np.degrees(np.arctan2(seg[1], seg[0]))
                nseg = pts[goal_i+1] - tgt
                a2 = np.degrees(np.arctan2(nseg[1], nseg[0]))
                turn = abs(_wrap(a2 - a1))
                v_c = v_now * max(0.3, 1.0 - turn/110.0)
                blend = float(np.clip((240.0 - d_knee)/160.0, 0.0, 1.0))
                v_now = v_now*(1.0 - blend) + v_c*blend
        eff_th = th + (180.0 if reverse else 0.0)
        err = _wrap(np.degrees(np.arctan2(tgt[1]-py, tgt[0]-px)) - eff_th)
        if pivoting or abs(err) > 95.0:
            # Target behind the shoulder: stand and turn -- rare, and only
            # at entry.  WITH HYSTERESIS: keep standing until the error is
            # small.  Bailing out at 94 deg had the chassis open its arc
            # 74 deg off the bearing -- from the kit dispatch that drove it
            # into the laboratory plate's edge and it ground there to the
            # watchdog (measured: 2.7 s, every match).  A standing turn is
            # cheap; an arc that opens sideways is not.
            pivoting = abs(err) > 22.0
            if pivoting:
                rb.drive(0.0, np.clip(3.0*err, -PIVOT_W, PIVOT_W))
                yield
                continue
        # corner slowdown: the heading error IS the curvature demand
        v = v_now * max(0.35, 1.0 - abs(err)/95.0)
        if reverse:
            rb.drive(-v_now*max(0.35, 1.0 - abs(err)/110.0),
                     np.clip(1.8*err, -45.0, 45.0))
        else:
            rb.drive(v, np.clip(2.2*err, -W_MAX, W_MAX))
        # watchdog: distance to the end must keep falling.  The epsilon is
        # 4 mm, not 0.5: a chassis pinned on a wall still CREEPS about a
        # millimetre a second under the scrubbing wheels, and that jitter
        # fed the finer threshold forever (measured: 15 s pinned, watchdog
        # never fired).  Real transits cover 4 mm in a few ticks.
        if end_d < last_d - 4.0:
            last_d, held = end_d, 0
        else:
            held += 1
            if held > 125:                       # 2.5 s of no progress
                rb.stop()
                return False
        yield


def capture_line(rb, ax, ay, head_deg, gate_mm, v_cruise=200.0,
                 v_arrive=95.0, tol_cross=12.0, tol_head=12.0,
                 reverse=False):
    """Acquire the line THROUGH (ax, ay) at heading head_deg, arriving at
    the GATE -- gate_mm before (ax, ay) along the line -- on-line, square,
    at ~v_arrive.  Returns True at the gate; False on overshoot or stall
    (the caller's dress/turn fallbacks still exist for that).

    The virtual target sits on the line, ahead of the chassis's own
    projection by a speed-scaled lookahead, never past the gate: chasing
    it is what curves the chassis onto the line, and the shrinking
    lookahead near the gate is what straightens it out.

    reverse=True captures TAIL-FIRST: head_deg stays the direction of
    MOTION, the chassis heading ends at head_deg+180, and the steering
    follows back_to's lesson (a reversing differential drive steers WITH
    the bearing error).  This is what the beam-2 lane wants: the run-in is
    a reverse wall-stall, so capturing in reverse hands place_beam a
    staged chassis with NO about-face -- and no pivot west of x 192, where
    none is legal.

    CERTIFIED ENVELOPE (check_trajectory): entries to 90 mm of cross-track
    and 35 deg of heading, EXCEPT the crossing diagonals -- far off the
    line while pointing hard across it (|lat x heading| beyond ~40 mm x
    27 deg, opposite signs), which need an S-curve one capture arc cannot
    give and end ~13 mm across the line.  No caller enters there: the
    seal's normalised basin arrives 62 mm off and near-parallel.  Outside
    the envelope the primitive refuses and the caller's turn_to fallback
    stands.
    """
    t = np.radians(head_deg)
    ux, uy = np.cos(t), np.sin(t)
    gate_s = -float(gate_mm)                    # along-track of the gate
    last_gap, held = 1e9, 0
    err_prev = None
    pivoting = False
    sgn = -1.0 if reverse else 1.0
    while True:
        px, py, th = rb.pose
        eff_th = th + (180.0 if reverse else 0.0)
        rx, ry = px - ax, py - ay
        s = rx*ux + ry*uy                       # along-track (0 at the station)
        e = -rx*uy + ry*ux                      # cross-track, left positive
        herr = _wrap(head_deg - eff_th)
        if s >= gate_s - 15.0:
            rb.drive(sgn*v_arrive, np.clip(2.4*herr - sgn*0.35*e, -35, 35))
            if abs(e) < tol_cross and abs(herr) < tol_head:
                return True                     # handoff: on-line and square
            if s > gate_s + 70.0:
                rb.stop()
                return False                    # past the gate, still crooked
            yield
            continue
        gap = gate_s - s
        v = v_arrive + (v_cruise - v_arrive) * \
            float(np.clip(min(gap/260.0, 1.2 - abs(e)/120.0,
                              1.2 - abs(herr)/70.0), 0.0, 1.0))
        # The lookahead grows with the CROSS-TRACK error: a short carrot at
        # a 90 mm offset asks for a 37-degree intercept and the chassis
        # overshoots the line and oscillates (measured on the entry grid);
        # 1.7x the offset caps the intercept near 30 degrees and the
        # approach becomes the asymptote it is supposed to be.
        L = max(100.0, 0.6*v, 1.7*abs(e))
        if reverse:
            L *= 1.4        # a reversing chassis converges underdamped on
                            # the forward lookahead (crossed the line by
                            # 12-18 mm, measured); a shallower intercept
                            # is the structural damping
        s_t = min(s + L, gate_s + 60.0)         # the carrot never leads past
        tx_, ty_ = ax + s_t*ux, ay + s_t*uy     # ...the gate by much
        err = _wrap(np.degrees(np.arctan2(ty_-py, tx_-px)) - eff_th)
        if pivoting or abs(err) > 95.0:
            # stand and turn the MOTION direction toward the carrot -- and
            # keep standing until it is nearly there (same hysteresis as
            # the tracker: an arc that opens 70+ deg off the bearing goes
            # somewhere no caller has checked).
            pivoting = abs(err) > 22.0
            if pivoting:
                rb.drive(0.0, np.clip(3.0*err, -PIVOT_W, PIVOT_W))
                yield
                continue
        # ...with a touch of damping: entries from the far side of the
        # cone arrived at the gate still rolling out of the intercept
        # (5-13 deg past square, measured); the derivative term kills the
        # roll-out without slowing the capture.
        d_err = 0.0 if err_prev is None else (err - err_prev)
        err_prev = err
        if reverse:
            # back_to's proven reverse law, verbatim: gentler gain, wider
            # clip, NO derivative.  The forward gains mirrored into reverse
            # pirouetted on the spot (measured: 98 mm in 10.9 s).
            rb.drive(-v*max(0.35, 1.0 - abs(err)/110.0),
                     np.clip(1.8*err, -45.0, 45.0))
        else:
            rb.drive(v*max(0.35, 1.0 - abs(err)/95.0),
                     np.clip(2.0*err + 7.0*d_err, -W_MAX, W_MAX))
        # progress = closing the gate AND closing the line: a curving entry
        # spends seconds mostly lateral, and a gate-only watchdog shot it.
        # Epsilon 4 mm for the same reason as the tracker's: pinned wheels
        # still creep past a half-millimetre threshold.
        prog = gap + abs(e)
        if prog < last_gap - 4.0:
            last_gap, held = prog, 0
        else:
            held += 1
            if held > 125:
                rb.stop()
                return False
        yield
