"""The station's operations, run through the HAL one control tick at a time.

The plan's executor: move, look, pick, place, insert, press, screw, probe,
release -- plus home and touch, which commissioning is built from.  Each
is a generator (every yield is one 50 Hz tick, truss.hal.Clock), each has
a timeout, and each returns a Done saying what it did and whether it
worked, never assuming it did.  Like truss/process.py it never reaches
around the HAL: where a plan would write a duration it waits for the
sensor that says the step is done -- the encoder settled, the slide's reed
switch, the jaws' switch, the load cell, the spindle's stall.

WHERE THINGS ARE comes from a StationFrame (commission.py): the drawing's
until the station has commissioned itself, the measured one after.  Every
coordinate the executor sends is the head point's, believed; every target
it is given is a station point, which the frame turns into a head point
for the tool that is to reach it.

TIMEOUTS ARE THE PLAN'S OWN TIMES, overrun.  An op that takes
Module.OVERRUN times what its sensors should need has failed: a slide
that has not reached its switch, jaws that have closed on nothing, a
spindle that never stalls.

A MOVE THAT LOST STEPS IS A FAILED MOVE.  The encoder is the only thing
that can tell; after every move the executor compares it with the count,
and a difference of an electrical cycle is a crash or a jam, not a
rounding.  It reports it; it does not carry on believing.
"""
from dataclasses import dataclass, field

import numpy as np

from truss import motion
from truss.spec import Gantry
from ..spec import (Module, LoadCell, HomeSwitch, ToolSlide, Gripper2, Spindle, Pogo, Artefact, HeadCam,
                    Stepper)
from . import spec as SS

AXES = SS.AXES
IX = {"x": 0, "y": 1, "z": 2}


@dataclass
class Done:
    ok: bool
    op: str
    why: str = ""
    data: dict = field(default_factory=dict)
    t0: float = 0.0
    t1: float = 0.0


class Executor:
    """`hal` implements every station contract (station/hal.py); `vision`
    is a VisionHAL; `frame` a commission.StationFrame."""

    def __init__(self, hal, clock, mod, frame, vision=None, log=None):
        self.hal, self.clock, self.mod, self.frame = hal, clock, mod, frame
        self.vision = vision
        self.log = log or (lambda s: None)
        self.done = []                 # every op's Done, in order
        self.lost = []                 # (op, axis, steps) every time the encoder disagreed
        self.held = {}                 # tool -> what it holds, by its own sensor

    # ============================================================== WAITS
    def _ticks(self, seconds):
        return max(int(np.ceil(seconds * self.clock.HZ - 1e-9)), 1)

    def _wait(self, seconds):
        for _ in range(self._ticks(seconds)):
            yield

    def _until(self, pred, planned, what):
        """Wait for pred, at most Module.OVERRUN times what it should take."""
        limit = Module.OVERRUN * max(planned, self.clock.PERIOD)
        t0 = self.clock.now()
        while not pred():
            if self.clock.now() - t0 > limit:
                self.log("      timeout after %.2f s: %s" % (self.clock.now() - t0, what))
                return False
            yield
        return True

    def _record(self, d, t0):
        d.t0, d.t1 = t0, self.clock.now()
        self.done.append(d)
        return d

    # ============================================================= HOMING
    def home(self, axes=("z", "y", "x")):
        """Z first and alone -- it lifts every tool off the work -- then the
        rest together."""
        t0 = self.clock.now()
        groups = [a for a in axes if a == "z"], [a for a in axes if a != "z"]
        for grp in groups:
            if not grp:
                continue
            # seek the switch at a quarter speed -- from where the encoder
            # says the axis is, or from the far end of its travel if it has
            # never been homed -- back off it at that speed, and creep back
            # onto it at the switch's own
            back = 2.0 * SS.home_zone()
            seek = {a: (abs(self.hal.encoder(a) - self.mod.axes[a].home) if self.hal.homed(a)
                        else self.mod.axes[a].travel) for a in grp}
            planned = max((seek[a] + back) / (self.mod.axes[a].v_max / 4.0) + 2.0 * back / HomeSwitch.CREEP_V
                          for a in grp)
            for a in grp:
                self.hal.home(a)
            ok = yield from self._until(lambda: all(self.hal.homed(a) for a in grp), planned, "homing %s" % grp)
            if not ok:
                return self._record(Done(False, "home", "%s never tripped its switch" % "".join(grp)), t0)
        return self._record(Done(True, "home"), t0)

    # ============================================================= MOTION
    def _check_lost(self, op):
        bad = {}
        for a in AXES:
            step = self.mod.axes[a].drive.step
            dn = (self.hal.encoder(a) - self.hal.at(a)) / step
            if abs(dn) > 1.0:
                bad[a] = dn
                self.lost.append((op, a, dn))
        return bad

    def move(self, goal, frac=1.0, op="move"):
        """A coordinated rest-to-rest move of the axes in `goal` (believed
        head point), at frac of the module's limits; then the encoder
        settled, the settle time, and the lost-step check."""
        t0 = self.clock.now()
        for a, v in goal.items():
            lo, hi = self.hal.limits()[a]
            if not lo <= v <= hi:
                return self._record(Done(False, op, "%s %.3f is outside %.3f .. %.3f" % (a, v, lo, hi)), t0)
        start = {a: self.hal.at(a) for a in goal}
        mv = motion.Move(start, goal, {a: self.mod.axes[a].v_max * frac for a in goal},
                         {a: self.mod.axes[a].a_max * frac for a in goal})
        t = 0.0
        while t < mv.T:
            for a, v in mv.at(t).items():
                self.hal.goto(a, v)
            t += self.clock.PERIOD
            yield
        for a, v in goal.items():
            self.hal.goto(a, v)
        ok = yield from self._until(lambda: all(self.hal.settled(a) for a in goal), Gantry.SETTLE_S, "settle")
        yield from self._wait(Gantry.SETTLE_S)
        bad = self._check_lost(op)
        if bad:
            return self._record(Done(False, op, "lost steps: " + ", ".join("%s %+.1f" % kv for kv in bad.items()),
                                     {"lost": bad}), t0)
        return self._record(Done(ok, op, "" if ok else "did not settle", {"T": mv.T}), t0)

    def lift(self, z=None):
        """Z up to the carrying height (or z), alone."""
        z = self.mod.z_carry if z is None else z
        if abs(self.hal.at("z") - z) < 1e-6:
            return Done(True, "lift")
        return (yield from self.move({"z": z}, op="lift"))

    def travel(self, head):
        """To a head point by the safe route: lift, cross, lower."""
        d = yield from self.lift()
        if not d.ok:
            return d
        d = yield from self.move({"x": head[0], "y": head[1]}, op="travel")
        if not d.ok:
            return d
        if len(head) > 2 and head[2] is not None:
            d = yield from self.move({"z": head[2]}, op="lower")
        return d

    # ============================================================== TOOLS
    def slide(self, kind, out=True):
        t0 = self.clock.now()
        (self.hal.extend if out else self.hal.retract)(kind)
        sensor = (lambda: self.hal.extended(kind)) if out else (lambda: self.hal.retracted(kind))
        ok = yield from self._until(sensor, ToolSlide.TIME_S, "%s %s" % (kind, "out" if out else "in"))
        return self._record(Done(ok, "extend" if out else "retract", "" if ok else "no reed switch", {"tool": kind}), t0)

    # ============================================================= LOOKING
    def look(self, p, z_surface, looks=2, over=None):
        """Where the disc nearest station point p (x, y) is, in the station
        frame: the camera put over p from the carrying height -- or, given
        the region `over`, from that region's own (StationModule.z_over) --
        and the disc found, then the camera centred on it and the disc found
        again, so the answer never leans on the lens's edges or on a tall
        disc's side."""
        t0 = self.clock.now()
        if self.vision is None:
            return self._record(Done(False, "look", "no camera"), t0)
        target = np.asarray(p[:2], float)
        z_look = self.mod.z_over.get(over, self.mod.z_carry) if over is not None else self.mod.z_carry
        for _ in range(looks):
            hx, hy = self.frame.camera_head(target, z_look, z_surface)
            d = yield from self.travel((hx, hy, None if abs(z_look - self.mod.z_carry) < 1e-6 else z_look))
            if not d.ok:
                return self._record(Done(False, "look", d.why), t0)
            uv = self.vision.locate()
            if uv is None:
                return self._record(Done(False, "look", "nothing in view near %.1f, %.1f" % tuple(target)), t0)
            head = np.array([self.hal.at("x"), self.hal.at("y")])
            target = self.frame.to_station(head, self.hal.at("z"), uv, z_surface)
        off = float(np.hypot(uv[0] - HeadCam.W / 2.0, uv[1] - HeadCam.H / 2.0))
        return self._record(Done(True, "look", data={"p": tuple(float(v) for v in target), "uv": uv,
                                                      "head": tuple(head), "off_px": off}), t0)

    # ============================================================ TOUCHING
    def touch(self, axis, sign, reach, op="touch"):
        """Creep along `axis` (sign +1 or -1) until the work pushes back, at
        most `reach` mm; then back to where it started.  Contact is where the
        load cell's force, fitted against position once it has risen out of
        its noise, extrapolates to nothing -- not where it first crossed a
        threshold, which is late by however soft the work is.  The noise is
        the reading's own, measured on the way in.

        Twice.  A step a tick first, to find the work.  Against something
        hard a reading then carries up to a step's worth of the head's
        stiffness, which on X or Y can be more than the preload holding the
        carriage on its flank, and a fit over readings that pushed it off
        would be a fit over the play.  So back off, and in again by as few
        microsteps a tick as raise the force by about one threshold: a
        sixteenth of a step against steel, whole steps against a spring.
        The fit then takes readings until they span TOUCH_POINTS thresholds,
        and never past SS.touch_cap -- the preload, if the work pushes away
        from home.  The second pass starts back from where the first one's
        readings put the contact; one that starts in contact backs off twice
        as far and tries again."""
        t0 = self.clock.now()
        ax = self.mod.axes[axis]
        step = ax.drive.step
        micro = step / Stepper.MICROSTEPS
        start = self.hal.at(axis)
        end = start + sign * reach
        lo, hi = self.hal.limits()[axis]
        if not lo <= end <= hi:
            return self._record(Done(False, op, "touch would leave the travel"), t0)
        cap = SS.touch_cap(ax, sign)
        self.hal.tare()
        yield from self._wait(Gantry.SETTLE_S)
        self.hal.tare()
        i = IX[axis]
        noise = [None, None]                     # mean, sd of the free readings, moving

        def creep(frm, length, inc, enough):
            """Readings in contact [(x, f)], in a row, creeping from frm by
            inc a tick for at most length mm until enough(readings); None if
            the very first reading was one."""
            free, pts, pos, first = [], [], frm, True
            while abs(pos - frm) < length:
                pos += sign * inc
                self.hal.goto(axis, pos, True)
                yield
                f = -sign * self.hal.force()[i]
                x = self.hal.at(axis)
                # the first readings are free travel by construction -- the
                # caller starts a touch clear of what it expects to meet --
                # and they are the noise a contact has to rise out of
                if noise[1] is None:
                    free.append(f)
                    if len(free) == Module.TOUCH_FREE:
                        noise[:] = float(np.mean(free)), max(float(np.std(free)), LoadCell.NOISE)
                    continue
                if f - noise[0] > Module.Z_SIGMAS * noise[1]:
                    if first:
                        return None
                    pts.append((x, f - noise[0]))
                    if enough(pts):
                        break
                elif pts:
                    pts = []                     # a spike, not a contact
                first = False
            return pts

        # find it: two readings in a row, so a spike does not count
        found = yield from creep(start, reach, max(HomeSwitch.CREEP_V * self.clock.PERIOD, step),
                                 lambda p: len(p) >= 2)
        pts, why = [], ""
        if found:
            (x1, f1), (x2, f2) = found
            thr = Module.Z_SIGMAS * noise[1]
            # how stiff the work is, as the find saw it -- no softer than a
            # threshold a step, so soft work is measured a step at a time
            k1 = max((f2 - f1) / abs(x2 - x1), thr / step)
            inc = min(max(round(thr / k1 / micro), 1) * micro, step)
            # contact is about f1 / k1 before the first reading that felt
            # it, and that reading lagged its setpoint by up to a step
            gap = f1 / k1 + 2.0 * step
            span = Module.TOUCH_POINTS * thr
            enough = lambda p: (len(p) >= Module.TOUCH_POINTS and p[-1][1] - p[0][1] >= span) or p[-1][1] >= cap
            for _ in range(3):
                back = x1 - sign * gap
                d = yield from self.move({axis: back}, frac=0.25, op=op + " off")
                if not d.ok:
                    why = d.why
                    break
                pts = yield from creep(back, gap + cap / k1, inc, enough)
                if pts is not None:
                    break
                gap *= 2.0
        # back off before anything else, at the creep speed
        d_back = yield from self.move({axis: start}, frac=0.25, op=op + " back")
        if not found:
            return self._record(Done(False, op, "met nothing in %.1f mm" % reach), t0)
        if why or pts is None:
            return self._record(Done(False, op, why or "still touching %.3f mm back from %.3f" % (gap, x1)), t0)
        if len(pts) < Module.TOUCH_POINTS or pts[-1][1] - pts[0][1] < span:
            return self._record(Done(False, op, "lost the contact it found at %.3f" % x1 if len(pts) < 2 else
                                     "%.1f N/mm is too soft to fit under %.1f N" % (k1, cap)), t0)
        x = np.array([p[0] for p in pts])
        f = np.array([p[1] for p in pts])
        k, b = np.polyfit(x, f, 1)
        x0 = -b / k
        ok = d_back.ok and k * sign > 0.0
        return self._record(Done(ok, op, "" if ok else "force fell as it pushed",
                                 {"at": float(x0), "k": float(abs(k)), "noise": noise[1], "pts": pts,
                                  "found": found, "inc": inc}), t0)

    def touch_contacts(self, reach, op="touch contacts"):
        """Lower until every pogo pin closes its circuit -- the block's own
        sensor -- at most `reach` mm; the head z where the last one closed."""
        t0 = self.clock.now()
        start = self.hal.at("z")
        if start - reach < self.hal.limits()["z"][0]:
            return self._record(Done(False, op, "touch would leave the travel"), t0)
        step = self.mod.axes["z"].drive.step
        per_tick = max(HomeSwitch.CREEP_V * self.clock.PERIOD, step)
        z, at = start, None
        while start - z < reach:
            z -= per_tick
            self.hal.goto("z", z)
            yield
            if all(self.hal.contacts()):
                at = self.hal.at("z")
                break
        d_back = yield from self.move({"z": start}, frac=0.25, op=op + " back")
        if at is None:
            return self._record(Done(False, op, "the pins never all closed in %.1f mm" % reach), t0)
        return self._record(Done(d_back.ok, op, d_back.why, {"at": float(at)}), t0)

    # =================================================== TAKING HOLD OF THINGS
    def _tip_to(self, kind, p, extended=True):
        """Head point putting this tool's tip on station point p."""
        return self.frame.head_for(kind, p, extended)

    def pick(self, kind, p, grip_z=None):
        """Take the part whose top centre is station point p with `kind`:
        over it, the tool out, down to where the tool takes it (a face tool
        on the top; the gripper's jaws round its side), hold, and up.
        Picked means the tool's own sensor says so."""
        t0 = self.clock.now()
        z = p[2] if grip_z is None else grip_z
        head = self._tip_to(kind, (p[0], p[1], z))
        for step in (self.travel((head[0], head[1], None)), self.slide(kind, True),
                     self.move({"z": head[2]}, frac=0.5, op="down")):
            d = yield from step
            if not d.ok:
                return self._record(Done(False, "pick", d.why, {"tool": kind}), t0)
        self.hal.grip(kind)
        planned = Gripper2.CLOSE_S if kind == "gripper" else ToolSlide.TIME_S
        held = yield from self._until(lambda: self.hal.holding(kind), planned, "%s takes hold" % kind)
        yield from self._wait(Gantry.SETTLE_S)
        d = yield from self.lift()
        d2 = yield from self.slide(kind, False)
        still = self.hal.holding(kind)
        self.held[kind] = still
        ok = held and still and d.ok and d2.ok
        return self._record(Done(ok, "pick", "" if ok else ("never took hold" if not held else "dropped it"),
                                 {"tool": kind}), t0)

    def place(self, kind, p, grip_z=None):
        """Put what `kind` holds down with its top centre at station point p."""
        t0 = self.clock.now()
        z = p[2] if grip_z is None else grip_z
        head = self._tip_to(kind, (p[0], p[1], z))
        for step in (self.travel((head[0], head[1], None)), self.slide(kind, True),
                     self.move({"z": head[2]}, frac=0.5, op="down")):
            d = yield from step
            if not d.ok:
                return self._record(Done(False, "place", d.why, {"tool": kind}), t0)
        d = yield from self.release(kind)
        return self._record(Done(d.ok, "place", d.why, {"tool": kind}), t0)

    def release(self, kind):
        t0 = self.clock.now()
        self.hal.release(kind)
        planned = Gripper2.CLOSE_S if kind == "gripper" else ToolSlide.TIME_S
        let_go = yield from self._until(lambda: not self.hal.holding(kind), planned, "%s lets go" % kind)
        yield from self._wait(planned)
        d = yield from self.lift()
        d2 = yield from self.slide(kind, False)
        self.held[kind] = False
        ok = let_go and d.ok and d2.ok
        return self._record(Done(ok, "release", "" if ok else "still holding", {"tool": kind}), t0)

    # ============================================================== PRESSES
    def press(self, kind, p, depth, f_max, op="press"):
        """Push `kind` down `depth` mm past where it meets the work at
        station point p, recording force against depth; stop early at
        f_max.  Insertion is a press with a force window on the way in."""
        t0 = self.clock.now()
        head = self._tip_to(kind, (p[0], p[1], p[2] + SS.work_gap()))
        for step in (self.travel((head[0], head[1], None)), self.slide(kind, True),
                     self.move({"z": head[2]}, frac=0.5, op="down")):
            d = yield from step
            if not d.ok:
                return self._record(Done(False, op, d.why, {"tool": kind}), t0)
        c = yield from self.touch("z", -1, 2.0 * SS.work_gap(), op=op + " find")
        if not c.ok:
            yield from self.lift()
            yield from self.slide(kind, False)
            return self._record(Done(False, op, c.why, {"tool": kind}), t0)
        z0 = c.data["at"]
        self.hal.tare()
        curve, z = [], self.hal.at("z")
        step = self.mod.axes["z"].drive.step
        per_tick = max(HomeSwitch.CREEP_V * self.clock.PERIOD, step)
        stopped = ""
        while z > z0 - depth:
            z -= per_tick
            self.hal.goto("z", z)
            yield
            f = self.hal.force()[2]
            curve.append((z0 - self.hal.at("z"), f))
            if f > f_max:
                stopped = "stopped at %.1f N, %.3f mm in" % (f, z0 - self.hal.at("z"))
                break
        yield from self._wait(Gantry.SETTLE_S)
        d = yield from self.lift()
        d2 = yield from self.slide(kind, False)
        ok = d.ok and d2.ok and not stopped
        return self._record(Done(ok, op, stopped, {"tool": kind, "z0": z0, "curve": curve}), t0)

    def insert(self, kind, p, depth, f_window):
        """A press that must stay inside its force window (lo at the end,
        hi throughout): what a snap fit or a press fit is judged by."""
        lo, hi = f_window
        d = yield from self.press(kind, p, depth, hi, op="insert")
        if d.ok and d.data["curve"] and d.data["curve"][-1][1] < lo:
            d.ok, d.why = False, "only %.1f N at full depth" % d.data["curve"][-1][1]
        return d

    # ============================================================= SCREWING
    def screw(self, p, turns_max, rpm=None):
        """Run a nut (or cap) down its thread with the spindle until it
        seats: the collet onto its top, hold, turn clockwise seen from above
        while Z follows the lead, until the spindle stalls."""
        t0 = self.clock.now()
        rpm = Spindle.RPM if rpm is None else rpm
        head = self._tip_to("spindle", (p[0], p[1], p[2] + SS.work_gap()))
        for step in (self.travel((head[0], head[1], None)), self.slide("spindle", True),
                     self.move({"z": head[2]}, frac=0.5, op="down")):
            d = yield from step
            if not d.ok:
                return self._record(Done(False, "screw", d.why), t0)
        c = yield from self.touch("z", -1, 2.0 * SS.work_gap(), op="screw find")
        if not c.ok:
            yield from self.lift()
            return self._record(Done(False, "screw", c.why), t0)
        yield from self.move({"z": c.data["at"]}, frac=0.25, op="onto")
        self.hal.grip("spindle")
        held = yield from self._until(lambda: self.hal.holding("spindle"), ToolSlide.TIME_S, "collet")
        if not held:
            yield from self.lift()
            return self._record(Done(False, "screw", "the collet never took it"), t0)
        a0 = self.hal.angle()
        lead = Artefact.STUD_LEAD
        self.hal.spin(-rpm)
        z = self.hal.at("z")
        seated = False
        planned = turns_max * 60.0 / rpm
        t1 = self.clock.now()
        while self.clock.now() - t1 < Module.OVERRUN * planned:
            # Z follows the thread: what the spindle turned since last tick
            turned = (a0 - self.hal.angle()) / 360.0
            zt = c.data["at"] - turned * lead
            if zt < self.hal.limits()["z"][0]:
                break
            self.hal.goto("z", zt)
            yield
            if self.hal.stalled():
                seated = True
                break
        self.hal.spin(0.0)
        turns = (a0 - self.hal.angle()) / 360.0
        yield from self.release("spindle")
        return self._record(Done(seated, "screw", "" if seated else "never seated",
                                 {"turns": float(turns)}), t0)

    # ============================================================== PROBING
    def probe(self, p):
        """Land the pogo block on the pad at station point p until every pin
        is in its working window, read them, and lift."""
        t0 = self.clock.now()
        head = self._tip_to("pogo", (p[0], p[1], p[2] + SS.work_gap()))
        for step in (self.travel((head[0], head[1], None)), self.slide("pogo", True),
                     self.move({"z": head[2]}, frac=0.5, op="down")):
            d = yield from step
            if not d.ok:
                return self._record(Done(False, "probe", d.why), t0)
        z = self.hal.at("z")
        step = self.mod.axes["z"].drive.step
        per_tick = max(HomeSwitch.CREEP_V * self.clock.PERIOD, step)
        floor = z - 2.0 * SS.work_gap() - Pogo.WORK[1]
        made = []
        while z > floor:
            z -= per_tick
            self.hal.goto("z", z)
            yield
            made = self.hal.contacts()
            if all(made):
                break
        yield from self._wait(Gantry.SETTLE_S)
        made = self.hal.contacts()
        d = yield from self.lift()
        d2 = yield from self.slide("pogo", False)
        ok = all(made) and d.ok and d2.ok
        return self._record(Done(ok, "probe", "" if ok else "%d of %d pins made contact" % (sum(made), len(made)),
                                 {"contacts": list(made)}), t0)
