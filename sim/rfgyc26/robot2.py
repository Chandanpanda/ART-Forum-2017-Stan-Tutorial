"""Robot 2: the detached actuator (design doc section 10).

Three layers, split exactly where the hardware splits:

  * SimLink -- the PICO W SIDE.  Decodes the LinkHAL v0 wire (the same
    grammar hal.LinkHAL encodes) into wheel commands, applies the DC
    plant's indignities where the real driver sits -- per-motor gain
    lottery, deadband, integer quantisation -- enforces the 250 ms
    dead-man, and owns the SHAKE macro as firmware.  Swap this class for
    a Bluetooth socket (BtLink, section 4.4) and NOTHING above changes.

  * R2Controller -- the PI SIDE.  Robot 2 has no sensors, so this runs on
    robot 1: a `spot` callable (robot 1's stereo rig finding the ArUco)
    is the only feedback, ~5 Hz with centimetre noise.  Between fixes the
    controller dead-reckons on its own commands scaled by CALIBRATION
    it learns as it drives: the per-side gain estimate is updated from
    every camera-observed displacement, which is the "learn to control
    the cheap motors" loop -- on real hardware the same code calibrates
    the real motors.

  * primitives -- face / goto / push_to / shake_out, all generators in
    the route.py convention (one yield per 50 Hz tick), commanding at
    20 Hz with 150 ms of dead-man margin.  Tolerances are camera-sized:
    nothing here pretends to millimetres, because every job is a plow
    push whose tolerance is the plow's width.
"""
import numpy as np
from . import hal
from .params import Robot2 as R2

TICK = hal.Clock.PERIOD                      # 50 Hz mission tick
_CMD_EVERY = max(1, int(round(hal.Clock.HZ / R2.CMD_HZ)))   # ticks per send


# ============================================================== the Pico side
class SimLink(hal.LinkHAL):
    """The firmware end of the wire, on MuJoCo.

    Construct with the model/data and a per-match rng: the gain lottery is
    drawn HERE, because that is where the real spread lives (motor + driver
    + tyre tolerances).  The mission driver calls step() once per control
    tick, before physics."""

    def __init__(self, m, d, rng=None):
        self.m, self.d = m, d
        rng = rng or np.random.default_rng(0)
        self._al = _aid(m, "r2_drive_l")
        self._ar = _aid(m, "r2_drive_r")
        self.gain_l = float(1.0 + R2.GAIN_SD * rng.standard_normal())
        self.gain_r = float(1.0 + R2.GAIN_SD * rng.standard_normal())
        self._vl = self._vr = 0.0
        self._expire = -1.0
        self._shake = 0.0                    # busy-until clock for the macro
        self._shake_t0 = 0.0
        self.sent = 0

    # --- wire ---
    def send(self, line):
        self.sent += 1
        f = line.split()
        now = self.d.time
        if f[0] == "V" and len(f) == 4:
            if now >= self._shake:           # firmware ignores V mid-shake
                self._vl, self._vr = float(int(f[1])), float(int(f[2]))
                self._expire = now + min(int(f[3]), 1000) / 1000.0
        elif f[0] == "K":
            self._vl = self._vr = 0.0
            self._expire = now
        elif f[0] == "SHAKE" and len(f) == 2:
            n = max(1, min(int(f[1]), 10))
            self._shake_t0 = now
            self._shake = now + n * 0.55     # n ratchet cycles
            self._vl = self._vr = 0.0

    def recv(self):
        return None

    # --- plant ---
    def _wheel(self, v, gain):
        # driver deadband, then the gain lottery; the wire already quantised
        if abs(v) < R2.DEADBAND:
            return 0.0
        return v * gain

    def step(self):
        """Apply the current command state to the actuators.  One call per
        control tick; the dead-man and the shake macro live here."""
        now = self.d.time
        if now < self._shake:
            # THE RATCHET (rigged, F93): collect against the front rail at
            # full reverse, then a full-jerk forward pulse -- the kit
            # sprints the tray and hops the 1.2 mm tail lip.  0.55 s/cycle.
            ph = (now - self._shake_t0) % 0.55
            v = -380.0 if ph < 0.25 else (380.0 if ph < 0.45 else 0.0)
            vl = vr = v
        elif now > self._expire:
            vl = vr = 0.0                    # dead-man: silence stops it
        else:
            vl, vr = self._vl, self._vr
        self.d.ctrl[self._al] = self._wheel(vl, self.gain_l) / R2.WHEEL_R
        self.d.ctrl[self._ar] = self._wheel(vr, self.gain_r) / R2.WHEEL_R

    def busy(self):
        return self.d.time < self._shake


def _aid(m, name):
    import mujoco
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if i < 0:
        raise KeyError(name)
    return i


# ============================================================ robot 1's eyes
def sim_spot(m, d, rng, sigma_mm=8.0, sigma_deg=2.5):
    """Model-camera tracking of robot 2's ArUco, the same convention as
    robot 1's synthetic `see_lab`: truth plus calibrated noise.  The render
    pipeline replaces this on the Pi; the CONTROLLER cannot tell."""
    import mujoco
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot2")

    def spot():
        x, y = d.xpos[b][:2] * 1000.0
        q = d.xquat[b]
        th = np.degrees(np.arctan2(2*(q[0]*q[3] + q[1]*q[2]),
                                   1 - 2*(q[2]**2 + q[3]**2)))
        return (x + rng.normal(0.0, sigma_mm),
                y + rng.normal(0.0, sigma_mm),
                th + rng.normal(0.0, sigma_deg))
    return spot


# =============================================================== the Pi side
def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


class R2Controller:
    """Closed loop over the camera, open loop between fixes.

    The belief is last-fix + command dead-reckoning through the learned
    gains; every new fix both corrects the belief and refines the gains
    (scale from distance covered, turn bias from heading drift on nominally
    straight legs).  Fix cadence ~5 Hz: every job here tolerates that."""

    FIX_EVERY = 10                            # ticks between camera looks

    def __init__(self, link, spot, clock):
        self.link, self.spot, self.clock = link, spot, clock
        x, y, th = spot()
        self.x, self.y, self.th = x, y, th
        self.scale = 1.0                      # learned: commanded -> real
        self.turn_bias = 0.0                  # deg/s of veer at straight cmd
        self._tick = 0
        self._cmd = (0.0, 0.0)
        self._dr = np.zeros(3)                # dead-reckoned delta since fix
        self._fix_xy = (x, y, th)

    # ---- belief upkeep -------------------------------------------------
    def _integrate(self):
        vl, vr = self._cmd
        v = 0.5 * (vl + vr) * self.scale
        w = np.degrees((vr - vl) * self.scale / R2.TRACK) + \
            (self.turn_bias if abs(vl) > 50 and abs(vr) > 50 else 0.0)
        self.th += w * TICK
        self.x += v * np.cos(np.radians(self.th)) * TICK
        self.y += v * np.sin(np.radians(self.th)) * TICK
        self._dr += (v * TICK, 0.0, w * TICK)

    def _maybe_fix(self):
        self._tick += 1
        if self._tick % self.FIX_EVERY:
            return
        x, y, th = self.spot()
        # calibration: compare camera displacement with the dead-reckoned
        # one over the window; only meaningful when the robot really moved
        cam_d = float(np.hypot(x - self._fix_xy[0], y - self._fix_xy[1]))
        dr_d = float(self._dr[0])
        # calibrate only on healthy motion: a JAM (commanded travel, no
        # camera travel) is not a gain -- it drove scale to the old 0.6
        # floor once and quarter-speed dead-reckoning poisoned every leg
        # after.  Physical DC spreads are +-15%; clamp accordingly.
        if abs(dr_d) > 25.0 and cam_d > 0.5 * abs(dr_d):
            r = float(np.clip(cam_d / abs(dr_d), 0.7, 1.4))
            self.scale = float(np.clip(0.9*self.scale + 0.1*self.scale*r,
                                       0.8, 1.3))
        self.x, self.y, self.th = x, y, th
        self._fix_xy = (x, y, th)
        self._dr[:] = 0.0

    # ---- wire ----------------------------------------------------------
    def _drive(self, vl, vr):
        vl = float(np.clip(vl, -R2.V_MAX, R2.V_MAX))
        vr = float(np.clip(vr, -R2.V_MAX, R2.V_MAX))
        self._cmd = (vl, vr)
        if self._tick % _CMD_EVERY == 0:
            self.link.cmd(vl, vr, 150)

    def stop(self):
        self._cmd = (0.0, 0.0)
        self.link.halt()

    def tick(self):
        """One 50 Hz step of bookkeeping; call from inside primitives."""
        self._integrate()
        self._maybe_fix()

    @property
    def pose(self):
        return self.x, self.y, self.th

    # ---- primitives ----------------------------------------------------
    def face(self, th_t, tol=6.0, cap_s=4.0):
        n = int(cap_s * hal.Clock.HZ)
        for _ in range(n):
            err = _wrap(th_t - self.th)
            if abs(err) < tol:
                break
            w = np.clip(3.2 * err, -R2.W_MAX, R2.W_MAX)
            dv = np.radians(w) * R2.TRACK / 2.0
            # keep the wheels outside the deadband or nothing turns
            if abs(dv) < R2.DEADBAND + 12.0:
                dv = np.sign(dv) * (R2.DEADBAND + 12.0)
            self._drive(-dv, dv)
            self.tick()
            yield
        self.stop()
        for _ in range(4):
            self.tick(); yield

    def goto(self, tx, ty, v_max=320.0, tol=25.0, cap_s=12.0, slow_into=None):
        """Drive to a point: initial face if far off-bearing, then a heading-P
        pursuit with a speed ramp.  slow_into caps speed near the goal.
        A JAM (robot-robot contact, mostly) backs off and retries twice,
        then gives up rather than grinding the match away."""
        b0 = np.degrees(np.arctan2(ty - self.y, tx - self.x))
        if abs(_wrap(b0 - self.th)) > 35.0:
            yield from self.face(b0, tol=10.0)
        n = int(cap_s * hal.Clock.HZ)
        jx, jy, jn, jams = self.x, self.y, 0, 0
        for _ in range(n):
            jn += 1
            if jn >= 60:
                if np.hypot(self.x - jx, self.y - jy) < 14.0 and \
                        abs(self._cmd[0]) + abs(self._cmd[1]) > 100.0:
                    jams += 1
                    if jams > 2:
                        self.stop()
                        return
                    yield from self.back_off(90.0)
                jx, jy, jn = self.x, self.y, 0
            dx, dy = tx - self.x, ty - self.y
            dist = float(np.hypot(dx, dy))
            if dist < tol:
                break
            err = _wrap(np.degrees(np.arctan2(dy, dx)) - self.th)
            if abs(err) > 55.0:
                yield from self.face(np.degrees(np.arctan2(dy, dx)), tol=8.0)
                continue
            v = min(v_max, max(90.0, 2.2 * dist))
            if slow_into is not None:
                v = min(v, slow_into)
            v *= max(0.25, 1.0 - abs(err) / 80.0)
            w = np.clip(2.6 * err, -140.0, 140.0)          # deg/s of yaw
            dv = np.radians(w) * R2.TRACK / 2.0
            self._drive(v - dv, v + dv)
            self.tick()
            yield
        self.stop()
        for _ in range(4):
            self.tick(); yield

    def push_to(self, tx, ty, v=150.0, tol=30.0, cap_s=16.0):
        """A plow push: same pursuit, gentler -- the puck must stay in the
        pocket, so turns are arc-limited and speed is constant-low."""
        n = int(cap_s * hal.Clock.HZ)
        jx, jy, jn = self.x, self.y, 0
        for _ in range(n):
            jn += 1
            if jn >= 70:
                if np.hypot(self.x - jx, self.y - jy) < 14.0:
                    self.stop()
                    return                   # jammed mid-push: abandon it
                jx, jy, jn = self.x, self.y, 0
            dx, dy = tx - self.x, ty - self.y
            dist = float(np.hypot(dx, dy))
            if dist < tol:
                break
            err = _wrap(np.degrees(np.arctan2(dy, dx)) - self.th)
            w = np.clip(1.6 * err, -55.0, 55.0)            # gentle: keep the
            dv = np.radians(w) * R2.TRACK / 2.0            # puck in the pocket
            self._drive(v - dv, v + dv)
            self.tick()
            yield
        self.stop()
        for _ in range(4):
            self.tick(); yield

    def back_off(self, mm_, v=200.0):
        n = int(mm_ / v * hal.Clock.HZ)
        for _ in range(n):
            self._drive(-v, -v)
            self.tick()
            yield
        self.stop()
        for _ in range(4):
            self.tick(); yield

    def shake_out(self, n=4):
        self.link.shake(n)
        for _ in range(int(n * 0.55 * hal.Clock.HZ) + 10):
            self.tick()
            yield
        self.stop()


# ============================================================== the mission
# Field facts the push catalog is built from (params.Field, m2_layout):
# sticker columns x 80/160 (west) and 983/1063 (east), rows y 537 (bottom),
# 650, 763 (top); zones HOSP (471-671, >=901), RECOVERY (700-900, 190-270),
# PCC_L (<=200, >=981), PCC_R (>=943, >=981).  What a puck can do is wall
# geometry (F94, measured the hard way -- the first catalog parked the
# approach INSIDE the wall's corner sweep and the pivot batted the pucks):
#
#   * every column pushes NORTH or SOUTH along itself freely;
#   * mid columns (160/983) also take +-65-degree diagonals: the approach
#     point 133 mm behind the puck then keeps the body's 93 mm corner
#     circle 11 mm clear of the wall.  Shallower angles do not fit, so an
#     eastward escape off the west columns is ~0.42 mm of x per mm pushed;
#   * edge columns (80/1063) have standing room for NOTHING but the
#     column itself: their reds and greens stay adrift (-3), priced;
#   * within a column: the bottom row's south-going and the top row's
#     north-going pushes cross nothing; the middle row waits for whichever
#     neighbour its own push direction crosses.
#
# Approaches arrive PRE-FACED: _push stages 240 mm behind the puck along
# the push line first (when that point is in-field), so the goto's arrival
# heading IS the push heading and no plow-sweeping pivot happens within
# reach of the puck (F94's second lesson).

DIAG = 65.0                                   # degrees off east-west
_ZONES = {"red": (511.0, 941.0, 631.0, 1141.0),      # HOSP inset 40
          "green": (740.0, 215.0, 860.0, 245.0),     # RECOVERY inset
          "yellowW": (40.0, 1021.0, 160.0, 1141.0),  # PCC_L inset
          "yellowE": (983.0, 1021.0, 1103.0, 1141.0)}


def _generic(i, x, y, c):
    """A one-leg push from wherever the puck IS to the nearest inset point
    of its destination zone; None when it is already home or the approach
    would leave the field."""
    key = c if c != "yellow" else ("yellowE" if x > 570.0 else "yellowW")
    x0, y0, x1, y1 = _ZONES[key]
    tx = float(np.clip(x, x0, x1))
    ty = float(np.clip(y, y0, y1))
    if abs(tx - x) < 4.0 and abs(ty - y) < 4.0:
        return None                          # already inside
    ux, uy, n = _norm(tx - x, ty - y)
    ax, ay = x - ux * 190.0, y - uy * 190.0
    if not _infield(ax, ay):
        return None
    # carry 35 mm INTO the zone so the release leaves it inside
    return (2 if n < 200.0 else 6, i, x, y,
            [(tx + ux * 35.0, ty + uy * 35.0)])


def _classify(m, d=None):
    """Robot 1's colour classifier, model-camera convention: in render mode
    perception.classify_patch does this from pixels; the mission cannot
    tell.  With `d`, positions are CURRENT -- the east column is wherever
    robot 1's climb plowed it, not where the stickers were (F94)."""
    import mujoco
    if d is None or float(abs(d.xpos).sum()) < 1e-9:
        # match start: the live data has not been forwarded yet -- read the
        # spawn state (measured: planning from an unforwarded d aimed every
        # push at (0,0) and the robot chased the corner for 100 s)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
    out = []
    for i in range(64):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cyl%d" % i)
        if b < 0:
            break
        x, y = d.xpos[b][:2] * 1000.0
        g = m.geom_rgba[m.body_geomadr[b]]
        col = ("red" if g[0] > 0.6 and g[1] < 0.4 else
               ("green" if g[1] > 0.5 and g[0] < 0.5 else "yellow"))
        out.append((i, float(x), float(y), col))
    return out


def _infield(x, y, margin=95.0):
    return margin <= x <= 1143.0 - margin and margin <= y <= 1181.0 - margin


def _norm(dx, dy):
    n = float(np.hypot(dx, dy))
    return dx / n, dy / n, n


def _push(ctl, px, py, ux, uy, dist, v=175.0):
    """Acquire the puck at (px,py) and push it `dist` mm along unit (ux,uy).
    Stage 240 back along the line when the field allows, so the approach
    ARRIVES facing the push; face() only trims what is left."""
    from .params import Robot2 as R2
    hd = np.degrees(np.arctan2(uy, ux))
    sx, sy = px - ux * 240.0, py - uy * 240.0
    ax, ay = px - ux * (R2.PLOW_X + 52.0), py - uy * (R2.PLOW_X + 52.0)
    if _infield(sx, sy):
        yield from ctl.goto(sx, sy, v_max=310.0, tol=42.0)
    yield from ctl.goto(ax, ay, v_max=250.0, tol=26.0, slow_into=160.0)
    if abs((hd - ctl.th + 180.0) % 360.0 - 180.0) > 10.0:
        yield from ctl.face(hd, tol=6.0)
    tx, ty = px + ux * dist, py + uy * dist
    yield from ctl.push_to(tx - ux * (R2.PLOW_X - 10.0),
                           ty - uy * (R2.PLOW_X - 10.0), v=v,
                           cap_s=7.0 + dist / v)
    yield from ctl.back_off(95.0)


def _plan_side(pats, side, want="all"):
    """Ordered push tasks for one side.  Each: (prio, i, x, y, legs) where
    legs = [(ux, uy, dist), ...] applied in sequence with re-acquisition.
    Encodes the F94 catalog: which colour can leave which column, and the
    within-column crossing order."""
    east = side == "E"
    mid_x = 983.0 if east else 160.0
    s = -1.0 if east else 1.0               # x-direction away from the wall
    col = {}
    for i, x, y, c in pats:
        col.setdefault(round(x), []).append((i, x, y, c))
    tasks = []
    ca, sa = np.cos(np.radians(DIAG)), np.sin(np.radians(DIAG))
    pccy = 1010.0

    def diag_to_recovery(x, y):
        """leg A: down-and-out at -DIAG to y~240, leg B: flat to RECOVERY."""
        d = (y - 240.0) / sa
        lx = x + s * ca * d
        bx = 845.0 if east else 800.0
        return [(lx, 240.0), (bx, 240.0)]

    def diag_to_hosp(x, y):
        """leg A: up-and-out at +DIAG to y<=870, leg B: aimed into HOSP.
        From the east the entry aims HIGH (640, 1075): robot 1's kit pile
        lands at (535-590, 920-960) at T+66-72 and the east phase runs
        after it -- the north-east entry corridor clears the pile."""
        d = (min(870.0, y + 333.0) - y) / sa
        lx, ly = x + s * ca * d, y + sa * d
        if east:
            bx, by = float(np.clip(lx - 140.0, 560.0, 640.0)), 1075.0
        else:
            bx, by = float(np.clip(lx + 140.0, 495.0, 640.0)), 1005.0
        return [(lx, ly), (bx, by)]

    for cx, pucks in col.items():
        is_mid = abs(cx - mid_x) < 40.0
        for i, x, y, c in pucks:
            if want == "nogreens" and c == "green":
                continue
            if want == "greens" and c != "green":
                continue
            row = round(y)
            off_sticker = min(abs(x - 80.0), abs(x - 160.0),
                              abs(x - 983.0), abs(x - 1063.0)) > 60.0 or \
                min(abs(y - 537.0), abs(y - 650.0), abs(y - 763.0)) > 60.0
            if off_sticker:
                # robot 1's plow relocated it (the climb pile): a generic
                # straight push to the nearest inset point of its zone
                g = _generic(i, x, y, c)
                if g:
                    tasks.append(g)
                continue
            if c == "yellow":
                # north to the own-side PCC along a wall-standoff line: the
                # edge columns sit 20 mm inside the plow wing, so the ROBOT
                # tracks x in [100, 1043] and the pocket carries the puck
                # (F94: at x 80 the wing grazed the wall and shed it)
                xv = float(np.clip(x, 100.0, 1043.0))
                tasks.append((10 + (763 - row) // 100, i, xv, y,
                              [(xv, pccy)]))
                continue
            elif not is_mid:
                continue                     # edge reds/greens: priced adrift
            elif c == "green":
                if row >= 700:
                    # top green: NE out (crosses nothing), then the long
                    # field push to RECOVERY
                    d = 107.0 / sa
                    lx, ly = x + s * ca * d, 870.0
                    bx, by = (845.0, 240.0) if east else (800.0, 240.0)
                    tasks.append((5, i, x, y, [(lx, ly), (bx, by)]))
                else:
                    # bottom/middle green: down-and-out; middle waits for
                    # the bottom sticker to clear (prio order handles it)
                    tasks.append((1 if row < 600 else 3, i, x, y,
                                  diag_to_recovery(x, y)))
            elif c == "red":
                if row >= 700:
                    tasks.append((5, i, x, y, diag_to_hosp(x, y)))
                else:
                    # bottom red goes NE early only if the row above it is
                    # already leaving (handled by prio: bottom-diag-up runs
                    # AFTER top/middle rows cleared)
                    tasks.append((7 if row < 600 else 6, i, x, y,
                                  diag_to_hosp(x, y)))
    tasks.sort(key=lambda k: k[0])
    return tasks


def _run_tasks(ctl, tasks, spot_puck, log, t, clock, stop_at, placed):
    """Camera-verified delivery: every leg starts from where the puck IS
    (robot 1's classifier re-spots it), and a leg that left it more than
    75 mm from its target gets ONE straight retry from the actual spot --
    open-loop pushes lost half their cargo and nobody knew (F96)."""
    for prio, i, x, y, legs in tasks:
        if clock() > stop_at:
            log(t() + "window over here")
            return
        t0 = clock()
        for k, (tx, ty) in enumerate(legs):
            for attempt in (0, 1):
                if clock() > stop_at or clock() - t0 > 24.0:
                    log(t() + "push %d out of time" % i)
                    break
                px, py = spot_puck(i)
                ux, uy, n = _norm(tx - px, ty - py)
                if n < 70.0:
                    break                    # this leg is done enough
                log(t() + "push %d leg %d%s: %.0f mm at %.0f deg"
                    % (i, k, "r" if attempt else "", n,
                       np.degrees(np.arctan2(uy, ux))))
                yield from _push(ctl, px, py, ux, uy, n + 12.0)
        placed[i] = spot_puck(i)


def mission_robot2(ctl, m, d=None, log=print, clock=None):
    """The whole match for the detached actuator.  One yield per tick."""
    t = (lambda: "") if clock is None else (lambda: "T+%5.1f R2 " % clock())
    pats = _classify(m, d)
    west = [p for p in pats if p[1] < 500.0]

    import mujoco as _mj
    _cb = {i: _mj.mj_name2id(m, _mj.mjtObj.mjOBJ_BODY, "cyl%d" % i)
           for i, _, _, _ in pats}
    _dd = d

    def spot_puck(i):
        if _dd is None:
            return (0.0, 0.0)
        p = _dd.xpos[_cb[i]]
        return float(p[0] * 1000.0), float(p[1] * 1000.0)
    log(t() + "colours: " + " ".join("%d%s@%.0f,%.0f" % (i, c[0], x, y)
                                     for i, x, y, c in pats))
    placed = {}

    def wait_until(ts):
        ctl.stop()
        while clock() < ts:
            ctl.tick()
            yield

    # ---- P0: kits FIRST, from the east-box spawn (F82/F95).  Six exit
    # routes through robot 1's half of the field all died on seed dice --
    # a 35 s wrestle, a mutual corner-lock, three wrecked sweeps -- because
    # no lane through it clears the sweep band, the parked body, the plate
    # and the samples at once.  The doc had the answer all along: robot 2
    # starts AGAINST THE EAST WALL (robot 1's spawn moved 144 mm west to
    # make room), so its opening act crosses nothing robot 1 will ever
    # touch: north-west under the sticker rows, up the east pinch it was
    # sized for, SHAKE into PCC_R with the north wall as the backstop.
    yield from ctl.goto(880.0, 420.0, v_max=340.0, tol=36.0)
    yield from ctl.goto(885.0, 700.0, v_max=320.0, tol=34.0)
    yield from ctl.goto(1040.0, 1080.0, v_max=300.0, tol=30.0)
    yield from ctl.face(270.0, tol=7.0)
    log(t() + "SHAKE: kits out against the north wall")
    yield from ctl.shake_out(4)

    # ---- P1: cross west on the y-1080 lane (empty until robot 1's kit
    # phase at T+66) and drop onto the west columns from above.
    yield from ctl.goto(700.0, 1080.0, v_max=300.0, tol=36.0)
    yield from ctl.goto(300.0, 990.0, v_max=300.0, tol=36.0)

    # ---- P2: west columns (the seal-corridor cure) ---------------------
    # greens wait: their RECOVERY leg crosses the dock band (x 380-760,
    # y<322), which robot 1 vacates around T+58
    yield from _run_tasks(ctl, _plan_side(west, "W", "nogreens"), spot_puck,
                          log, t, clock, stop_at=58.0, placed=placed)
    yield from wait_until(58.0)
    yield from _run_tasks(ctl, _plan_side(west, "W", "greens"), spot_puck,
                          log, t, clock, stop_at=76.0, placed=placed)

    # ---- P3: east columns once robot 1 has left the east side.  LOOK
    # AGAIN first: robot 1's climb has plowed the east column somewhere
    # else entirely (measured: four pucks in a pile 40 mm from HOSP).
    yield from wait_until(74.0)
    east = [p for p in _classify(m, d) if p[1] > 500.0]
    log(t() + "east re-look: " + " ".join(
        "%d%s@%.0f,%.0f" % (i, c[0], x, y) for i, x, y, c in east))
    yield from _run_tasks(ctl, _plan_side(east, "E"), spot_puck, log, t,
                          clock, stop_at=112.0, placed=placed)
    ctl.stop()
    log(t() + "done: %d pucks delivered or staged" % len(placed))
