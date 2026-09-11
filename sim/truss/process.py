"""The process: the plan, executed through the HAL, with the sensing the
plan assumed.

schedule.plan() decides what happens and in what order; this module makes
it happen, one control tick per yield, and never reaches around the HAL.
The same source runs against cell.py in MuJoCo and against the firmware
bridge on the machine.  Where the plan wrote a duration, the process
waits for the SENSOR that says the step is done -- the axis settled, the
cage indexed, the ring parked, the fiducial having counted the turns --
so a simulated cycle can be slower than the plan, and the check that
compares the two is the honest one.

The band state is kept HERE, from the fiducial: a turn is a turn the
ring's angle history says it made, laid at the x the feed had reached.
The inspector reads these states; it does not get them from the plan.
"""
import numpy as np

from .spec import (Gantry, Head, Gripper, Dispenser, Cutter,
                   Process)
from . import motion, band as _band


class Executor:
    """Runs a Plan.  Everything it touches is a HAL object."""

    def __init__(self, axes, ring, cage, gripper, dispenser, cutter, clock,
                 geom, fixture, plan, vision=None, log=None):
        self.axes, self.ring, self.cage = axes, ring, cage
        self.grip, self.disp, self.cutter = gripper, dispenser, cutter
        self.clock, self.geom, self.fixture, self.plan = clock, geom, fixture, plan
        self.vision = vision
        self.log = log or (lambda s: None)
        self.states = {j.index: _band.BandState(j.index) for j in geom.joints}
        self.corr = np.zeros(2)          # the last look's (dx, dy)
        self.timeline = []               # (op, t0, t1)
        self.looks = {}                  # joint -> (dx, dy) measured
        self.released = {}               # rod -> clock time
        self.slow = []                   # ops that overran the plan
        self.bonds = []                  # (rod, point, t) fillets laid
        self.direction = +1

    # ------------------------------------------------------------ waits
    def _wait(self, seconds):
        n = int(round(seconds * self.clock.HZ))
        for _ in range(max(n, 0)):
            yield

    def _until(self, pred, timeout, what):
        t0 = self.clock.now()
        while not pred():
            if self.clock.now() - t0 > timeout:
                self.log("      timeout: %s" % what)
                return False
            yield
        return True

    # ------------------------------------------------------------ moves
    def _move(self, goal, axes=("x", "y", "z")):
        start = {a: self.axes.at(a) for a in axes if a in goal}
        mv = motion.Move(start, {a: goal[a] for a in start}, Gantry.V_MAX, Gantry.A_MAX)
        t = 0.0
        while t < mv.T:
            for a, v in mv.at(t).items():
                self.axes.goto(a, v)
            t += self.clock.PERIOD
            yield
        for a, v in mv.goal.items():
            self.axes.goto(a, v)
        yield from self._until(lambda: all(self.axes.settled(a) for a in mv.goal),
                               2.0 + Gantry.SETTLE_S, "axes settle")
        yield from self._wait(Gantry.SETTLE_S)

    def _stroke(self, axis, target):
        start = self.axes.at(axis)
        T = motion.trap_time(target - start, Gantry.V_MAX["z"], Gantry.A_MAX["z"])
        t = 0.0
        while t < T:
            self.axes.goto(axis, start + motion.trap_position(t, target - start,
                                                              Gantry.V_MAX["z"], Gantry.A_MAX["z"]))
            t += self.clock.PERIOD
            yield
        self.axes.goto(axis, target)
        yield from self._until(lambda: self.axes.settled(axis, 0.1), 1.5, "stroke settle")

    def _yaw(self, target):
        """Turn the gripper to `target`, ABSOLUTELY.

        THE YAW HAS HARD STOPS AT +-GRIP_YAW, so there is no wrap-around to
        be short about -- and taking the short way round is how a rod gets
        laid off its own line.  From -90 to +90 the difference wraps to
        -180, the servo is commanded to -270, it clamps at its stop, and
        every batten and strut picked after the first is held 185 degrees
        from where the plan believes it is: five degrees off, once the
        rod's own end-for-end symmetry is taken out.

        Measured on rig_mount: four battens a truss at 8.2 degrees and
        4.43 mm with their midpoints dead on to 0.03 and NOTHING TOUCHING
        THEM.  It read as a collision for as long as it was only read as a
        number; the trace showed the pick yaw sitting at -95.0 when the op
        had asked for 90.
        """
        start = self.axes.at("w")
        goal = float(target)
        if abs(goal) > Head.GRIP_YAW + 1e-9:
            self.log("      yaw %.1f is outside the servo's +-%.0f stop"
                     % (goal, Head.GRIP_YAW))
            goal = max(-Head.GRIP_YAW, min(Head.GRIP_YAW, goal))
        d = goal - start
        T = abs(d) / Gripper.YAW_V
        t = 0.0
        while t < T:
            self.axes.goto("w", start + d * (t / T))
            t += self.clock.PERIOD
            yield
        self.axes.goto("w", goal)
        yield from self._until(lambda: self.axes.settled("w", 0.5), 1.0, "yaw settle")

    # --------------------------------------------------------- the ops
    def run(self):
        for op in self.plan.ops:
            t0 = self.clock.now()
            yield from self._do(op)
            t1 = self.clock.now()
            self.timeline.append((op, t0, t1))
            if t1 - t0 > op.dt * 1.5 + 0.5:
                self.slow.append((op, t1 - t0))
        return self.states

    def _do(self, op):
        k, a = op.kind, op.args
        if k == "move":
            goal = dict(a["pos"])
            if op.phase in ("wind", "dose"):
                goal["x"] += float(self.corr[0])
                goal["y"] += float(self.corr[1])
            yield from self._move(goal)
        elif k == "index":
            self.cage.index(a["theta"])
            yield from self._until(self.cage.indexed, op.dt + 3.0, "cage index")
        elif k == "yaw":
            yield from self._yaw(a["yaw"])
        elif k == "extend":
            axis = "g" if a["tool"] == "grip" else "d"
            full = Head.GRIP_STROKE if axis == "g" else Head.DISP_STROKE
            yield from self._stroke(axis, float(a.get("mm", full)))
        elif k == "retract":
            axis = "g" if a["tool"] == "grip" else "d"
            yield from self._stroke(axis, 0.0)
        elif k == "grip":
            if hasattr(self.grip, "free_from_rack"):
                self.grip.free_from_rack(a["rod"])
            self.grip.close()
            yield from self._wait(Gripper.JAW_CLOSE_S)
            yield from self._until(self.grip.holding, 0.5, "grip rod %d" % a["rod"])
            if not self.grip.holding():
                self.log("      rod %d: jaws closed on nothing" % a["rod"])
        elif k == "release":
            if a.get("tack"):
                # nothing under it: the keeper takes it before the jaws move
                self.cage.keep(a["rod"])
                yield from self._wait(0.2)
                self.grip.open()
                yield from self._wait(Gripper.JAW_CLOSE_S + 0.2)
            else:
                self.grip.open()
                yield from self._wait(Gripper.JAW_CLOSE_S + 0.4)  # settle in the V
                self.cage.keep(a["rod"])
            self.released[a["rod"]] = self.clock.now()
        elif k == "look":
            self.corr[:] = 0.0
            yield from self._wait(Process.VISION_SETTLE_S)
            if self.vision is not None:
                got = self.vision.locate(a["joint"])
                if got is not None:
                    # THE LOOK MEASURES WHERE THE JOINT IS; THE PLAN SAYS WHERE
                    # IT SHOULD BE.  The ring is parked a half-band from the
                    # joint on purpose, so the correction is the measured
                    # offset less the offset the geometry predicts from the
                    # head's believed position -- not the measurement itself
                    # (applied raw, it walked every band onto the joint's
                    # centre and the ring into the pins).
                    j = self.geom.joints[a["joint"]]
                    cj, _n = self.geom.joint_at(j, self.cage.theta())
                    want = (float(cj[0]) - self.axes.at("x"), float(cj[1]) - self.axes.at("y"))
                    self.corr[:] = (got[0] - want[0], got[1] - want[1])
                    self.looks[a["joint"]] = tuple(self.corr)
                    # trim onto the chord before descending
                    yield from self._move({"x": self.axes.at("x") + self.corr[0],
                                           "y": self.axes.at("y") + self.corr[1]}, axes=("x", "y"))
                else:
                    self.log("      joint %d: chord not in view" % a["joint"])
        elif k == "park":
            self.ring.park(float(a["gap"]))
            yield from self._until(self.ring.parked, op.dt + 2.0, "ring park")
        elif k == "wind":
            yield from self._wind(op)
        elif k == "dose":
            j = a["joint"]
            st = self.states[j]
            mg = _band.resin_dose_mg(self.geom.t, st.thread)
            self.disp.dose(mg, joint=j)
            st.dosed += mg
            yield from self._wait(Dispenser.DOSE_S + Process.DOSE_SETTLE_S)
        elif k == "bond":
            # A MOUNT JOINT: one epoxy fillet, from the same dispenser.
            # It is not a band, so it has no BandState -- the joint states
            # are the truss's, and a fillet on a strut is not one of them.
            from . import mount as _mount
            mg = _mount.fillet_mass_mg(2.0 * self.geom.all_rods[a["rod"]].r)
            self.disp.dose(mg, joint=None)
            self.bonds.append((a["rod"], tuple(a["at"]), self.clock.now()))
            yield from self._wait(Dispenser.DOSE_S + Process.DOSE_SETTLE_S)
        elif k == "anchor":
            chord, _end = a["post"]
            for j in self.geom.joints_on(chord):
                self.states[j.index].anchored = True
            yield from self._wait(Process.ANCHOR_S)
        elif k == "cut":
            self.cutter.cut()
            yield from self._wait(Cutter.CUT_S)
        else:
            yield from self._wait(op.dt)

    def _wind(self, op):
        """Spin until the fiducial has counted the turns, feeding x so the
        band grows in the direction of travel."""
        a = op.args
        j = self.geom.joints[a["joint"]]
        st = self.states[j.index]
        turns, feed = a["turns"], float(a["feed"])
        st.pitch = abs(feed) / turns
        st.direction = 1.0 if feed >= 0.0 else -1.0
        x0 = self.axes.at("x")
        st.x_start = x0
        a0 = self.ring.angle()
        last = a0
        self.ring.spin(a["rpm"])
        t0 = self.clock.now()
        while True:
            ang = self.ring.angle()
            d = ang - last
            last = ang
            prog = abs(ang - a0)
            # the hoop laid this tick lies where the feed has reached
            dx = np.sign(feed) * min(prog / (360.0 * turns), 1.0) * abs(feed)
            st.lay(d, self.geom.hoop_perimeter(j, dx - feed / 2.0))
            self.axes.goto("x", x0 + dx)
            if prog >= 360.0 * turns:
                break
            if self.clock.now() - t0 > op.dt * 2.0 + 5.0:
                self.log("      joint %d: wind timed out at %.1f turns" % (j.index, prog / 360.0))
                break
            yield
        self.ring.spin(0.0)
        self.axes.goto("x", x0 + feed)
        yield from self._until(lambda: self.axes.settled("x"), 1.0, "feed settle")
