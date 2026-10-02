"""The station module's MuJoCo backend of hal.py.

THE AXES ARE STEPPERS, AND THE PHYSICS DECIDES WHERE THEY END UP.  The
controller sets a step count each tick; this backend streams the steps
evenly through the tick, as a step generator does, and turns the count
into a force on the drive's slide by the stepper's own law -- F =
F_peak(v) sin(pi/2 e/step), e the rotor's lag -- so a motor asked for
more than it has drops four full steps at a time and stays there, as a
real one does.  The drive's give is a second force, on the carriage's own
slide: nothing inside the backlash, the drive's stiffness beyond it,
softer the further a screw's nut runs from its motor -- and the axis'
preload, the constant force toward home that keeps that play closed,
pulling from the frame so the motor carries it as well.
Rail drag is a tendon's friction.  The encoder reads the rotor, so it
sees lost steps and cannot see the carriage ring.

What the controller never sees: the screw's thermal growth (a scale drawn
once per run, as truss.cell does, and changed by warm() to make a shift
pass), which moves the nut along the screw and so the carriage, not the
rotor; where a home switch really trips; how the station was built.

Three things happen by constraint rather than by contact, each a declared
stand-in: a tool HOLDS a part by a weld that switches on only when the
part is where that tool could take it (the truss cell's grip weld); the
spindle's collet takes a part by its top face; and a nut stays where it
was left on its self-locking thread, welded, until the collet takes it.

Everything below the "truth" marker exists only in simulation: the checks
read it, the executor never does.
"""
from math import pi, sin, sqrt, copysign

import numpy as np
import mujoco

from truss.spec import stepper_scale_sigma
from truss import hal as truss_hal
from ..spec import (Stepper, HomeSwitch, LoadCell, Gripper2, Vacuum, Magnet, Spindle, Pogo, ToolSlide, Artefact,
                    Rail, Module)
from . import hal, spec as SS

J, A_, B_, S_, E_ = (mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR, mujoco.mjtObj.mjOBJ_BODY,
                     mujoco.mjtObj.mjOBJ_SITE, mujoco.mjtObj.mjOBJ_EQUALITY)
_RPM, _TQ = (tuple(float(r) for r, _ in Stepper.PULLOUT), tuple(float(t) for _, t in Stepper.PULLOUT))


def _pullout(rpm):
    """Stepper.PULLOUT, interpolated; nothing past its end."""
    if rpm >= _RPM[-1]:
        return 0.0
    for i in range(1, len(_RPM)):
        if rpm < _RPM[i]:
            f = (rpm - _RPM[i - 1]) / (_RPM[i] - _RPM[i - 1])
            return _TQ[i - 1] + f * (_TQ[i] - _TQ[i - 1])
    return _TQ[-1]


def _quat_from_mat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def drive_damping(ax):
    """N s/m on the rotor's lag behind its field: Stepper.ZETA of critical
    on the rotor's magnetic spring -- the driver's current loop."""
    return 2.0 * Stepper.ZETA * sqrt(ax.drive.k_magnet() * 1e3 * ax.drive.reflected() * 1e-3)


class _Axis:
    """One axis: its firmware (steps, homing) and its two slides."""

    def __init__(self, sim, a, eps):
        m = sim.m
        ax = sim.mod.axes[a]
        self.a, self.ax, self.dr = a, ax, ax.drive
        jd, jp = (mujoco.mj_name2id(m, J, "%s_drive" % a), mujoco.mj_name2id(m, J, "%s_play" % a))
        self.qd, self.vd = int(m.jnt_qposadr[jd]), int(m.jnt_dofadr[jd])
        self.qp, self.vp = int(m.jnt_qposadr[jp]), int(m.jnt_dofadr[jp])
        self.act = mujoco.mj_name2id(m, A_, "%s_motor" % a)
        self.body = mujoco.mj_name2id(m, B_, a)
        self.step = self.dr.step
        self.f_peak = self.dr.f_magnet()
        self.half = self.dr.backlash() / 2.0 * 1e-3            # m
        self.m_carr = ax.moving * 1e-3
        self.eps = eps                                          # the screw's growth since the count began
        # believed coordinate = b0 + n step.  n is the count the controller
        # asked for by the end of this tick, n_now the count streamed so far
        self.n, self.n_now, self.n_from, self.b0 = 0, 0, 0, ax.home
        self.set = ax.home
        self.toward = ax.toward                                 # which way the switch is, and the preload pulls
        self.state, self.trip, self.creep_to = "homed?", None, None
        self.k, self.c = 0.0, 0.0
        self.c_rotor = drive_damping(ax)
        self.period = truss_hal.Clock.PERIOD
        self.update_spring(sim.d)

    # ------------------------------------------------------------ firmware
    def believed_to_n(self, b, fine=False):
        if fine:
            return round((b - self.b0) / self.step * Stepper.MICROSTEPS) / float(Stepper.MICROSTEPS)
        return int(round((b - self.b0) / self.step))

    def carriage(self, d):
        """mm, the carriage's true position in the station frame."""
        return self.ax.home + 1000.0 * (d.qpos[self.qd] + d.qpos[self.qp])

    def encoder(self, d):
        q = self.dr.lead / Stepper.ENCODER_CPR
        return self.b0 + round(d.qpos[self.qd] * 1000.0 / q) * q

    def stream(self, frac):
        """The step generator: frac of the way through the tick, the count
        has gone that fraction of the way to this tick's setpoint -- in whole
        steps, or, if the driver interpolates, smoothly between them."""
        span = (self.n - self.n_from) * frac
        self.n_now = self.n_from + (span if Stepper.INTERPOLATE else int(round(span)))

    def update_spring(self, d):
        """The drive's stiffness where the nut now is: a screw softens as
        it runs from the motor.  Once a tick; the nut moves a fraction of a
        millimetre in one."""
        self.k = self.ax.k_drive(self.carriage(d)) * 1e3                    # N/m
        self.c = 2.0 * Rail.ZETA * sqrt(self.k * self.m_carr)

    def forces(self, d):
        """Before each physics step: the stepper's sine law on the drive,
        the drive's give on the carriage.

        What a stepper has at speed is set by how fast it is STEPPED (its
        current cannot rise in a shorter step), and what damps its rotor is
        the field it lags, so both follow the step rate, not the rotor's
        own speed: a damper on the rotor's absolute speed would brake a
        fast move with more than the motor has.  The screw's growth carries
        the nut eps further along for every millimetre it is from the
        motor's end, so it shifts where the give is centred."""
        step_m = self.step * 1e-3
        e = self.n_now * step_m - d.qpos[self.qd]
        v_field = (self.n - self.n_from) * step_m / self.period
        rpm = abs(v_field) * 1000.0 / self.dr.lead * 60.0
        d.ctrl[self.act] = (self.f_peak * (_pullout(rpm) / Stepper.HOLD_NM) * sin(0.5 * pi * e / step_m)
                            - self.c_rotor * (d.qvel[self.vd] - v_field))
        dl = d.qpos[self.qp] - self.eps * d.qpos[self.qd]
        if dl > self.half:
            f = -self.k * (dl - self.half) - self.c * d.qvel[self.vp]
        elif dl < -self.half:
            f = -self.k * (dl + self.half) - self.c * d.qvel[self.vp]
        else:
            f = 0.0
        # the preload pulls the carriage from the frame, as a constant-force
        # spring does: a force on the carriage is one on both slides, so the
        # motor carries it too (spec.derive sized the motor for it)
        d.qfrc_applied[self.vp] = f + self.toward * self.ax.preload
        d.qfrc_applied[self.vd] = self.toward * self.ax.preload


class StationSim(hal.AxesHAL, hal.ToolsHAL, hal.HoldHAL, hal.SpindleHAL, hal.ProbeHAL, hal.ForceHAL):
    def __init__(self, model, data, mod, info, build=None, rng=None, thermal=True):
        self.m, self.d, self.mod, self.info = model, data, mod, info
        self.build = build
        self.rng = rng if rng is not None else np.random.default_rng(0)
        sd = stepper_scale_sigma() if thermal else 0.0
        self.axes = {a: _Axis(self, a, float(self.rng.normal(0.0, sd))) for a in SS.AXES}
        gid = lambda t, n: mujoco.mj_name2id(model, t, n)
        self.tools = tuple(mod.head.tools)
        self.j_slide = {k: int(model.jnt_qposadr[gid(J, "slide_%s" % k)]) for k in self.tools}
        self.a_slide = {k: gid(A_, "slide_%s" % k) for k in self.tools}
        self.s_tip = {k: gid(S_, "tip_%s" % k) for k in self.tools}
        self.s_head = gid(S_, "head")
        self.lc = [int(model.jnt_qposadr[gid(J, n)]) for n in ("lc_x", "lc_y", "lc_z")]
        self.k_lc = LoadCell.RATED / LoadCell.DEFLECT              # N/mm
        self._tare = np.zeros(3)
        # the amplifier integrates over its conversion, a control period:
        # the reading is the tick's mean deflection, not one instant of the
        # head ringing on the cell after a full step
        self._lc_sum, self._lc_n = np.zeros(3), 0
        self._lc_last = np.zeros(3)
        if "gripper" in self.tools:
            self.j_jaw = int(model.jnt_qposadr[gid(J, "jaw_l")])
            self.a_jaw = gid(A_, "jaws")
        if "spindle" in self.tools:
            self.j_sp = int(model.jnt_qposadr[gid(J, "spindle")])
            self.v_sp = int(model.jnt_dofadr[gid(J, "spindle")])
            self.a_sp = gid(A_, "spindle")
        if "pogo" in self.tools:
            self.j_pogo = [int(model.jnt_qposadr[gid(J, "pogo%d" % i)]) for i in range(Pogo.N)]
        # the parts a tool may take, and the welds that would hold them
        self.parts = list(info["tokens"]) + (["nut"] if gid(B_, "nut") >= 0 else [])
        self.b_part = {p: gid(B_, p) for p in self.parts}
        self.eq = {kp: gid(E_, name) for kp, name in info["welds"].items()}
        self.lock = gid(E_, "nut_lock")
        if self.lock >= 0:
            mujoco.mj_forward(model, data)
            self._weld_now(self.lock, self.b_part["nut"], 0)          # body1 the nut, body2 the world
        self.holder_body = {"gripper": gid(B_, "jaw_l"), "vacuum": gid(B_, "tool_vacuum"),
                            "magnet": gid(B_, "tool_magnet"), "spindle": gid(B_, "collet")}
        self._want = {k: False for k in self.tools}
        self._held = {k: None for k in self.tools}
        self._dist = None
        self._block = None
        self.log = []
        mujoco.mj_forward(model, data)
        for ax in self.axes.values():
            ax.update_spring(data)

    # ================================================================ AXES
    def goto(self, axis, value, fine=False):
        ax = self.axes[axis]
        if ax.state in ("seek", "back", "creep"):
            raise RuntimeError("axis %s is homing" % axis)
        lo, hi = self.limits()[axis]
        if not lo - ax.step <= value <= hi + ax.step:
            raise ValueError("%s %.3f is outside its travel %.3f .. %.3f" % (axis, value, lo, hi))
        ax.n = ax.believed_to_n(float(value), fine)
        ax.set = ax.b0 + ax.n * ax.step

    def at(self, axis):
        return self.axes[axis].set

    def encoder(self, axis):
        return self.axes[axis].encoder(self.d)

    def settled(self, axis, tol=0.01):
        ax = self.axes[axis]
        v = abs(self.d.qvel[ax.vd]) * 1000.0
        return abs(ax.encoder(self.d) - ax.set) < tol and v < 20.0 * tol

    def limits(self):
        return {a: (ax.ax.lo, ax.ax.hi) for a, ax in self.axes.items()}

    def home(self, axis):
        """The usual two-speed homing: seek the switch at a quarter of the
        axis' speed from wherever the carriage is, back off it, then creep
        onto it at the speed its repeat is quoted at.  The seek's trip only
        finds the switch; the creep's is the one that becomes home."""
        ax = self.axes[axis]
        # the count starts again from where the encoder says the rotor is:
        # after lost steps they differ by whole electrical cycles, which the
        # field cannot tell apart, and every arrival from here on is judged
        # against the encoder
        ax.n = ax.n_now = ax.n_from = ax.believed_to_n(ax.encoder(self.d))
        ax.set = ax.b0 + ax.n * ax.step
        ax.state = "seek"
        ax.trip = self.mod.axes[axis].home + (self.build.home.get(axis, 0.0) if self.build else 0.0)
        self._march = getattr(self, "_march", {})
        self._march[axis] = (ax.set, None, self.mod.axes[axis].v_max / 4.0)

    def homed(self, axis):
        return self.axes[axis].state == "homed"

    def _serve_homing(self, dt):
        for a, ax in self.axes.items():
            if ax.state == "back":
                b, goal, v = self._march[a]
                stepv = v * dt
                b = goal if abs(goal - b) <= stepv else b + copysign(stepv, goal - b)
                self._march[a] = (b, goal, v)
                ax.n = ax.believed_to_n(b)
                ax.set = ax.b0 + ax.n * ax.step
                if b == goal and abs(ax.encoder(self.d) - ax.set) < 2.0 * ax.step:
                    ax.state = "creep"
                    # where it trips this time: as mounted, and as it repeats
                    ax.trip = (self.mod.axes[a].home + (self.build.home.get(a, 0.0) if self.build else 0.0)
                               + float(self.rng.normal(0.0, HomeSwitch.REPEAT)))
                    self._march[a] = (b, None, HomeSwitch.CREEP_V)
            elif ax.state in ("seek", "creep"):
                b, _g, v = self._march[a]
                b += ax.toward * v * dt
                self._march[a] = (b, None, v)
                ax.n = ax.believed_to_n(b)
                ax.set = ax.b0 + ax.n * ax.step

    def _check_switches(self):
        for a, ax in self.axes.items():
            if ax.state not in ("seek", "creep"):
                continue
            if (ax.carriage(self.d) - ax.trip) * ax.toward < 0.0:
                continue
            if ax.state == "seek":
                # found it: back off, from where the count has got to
                b = self._march[a][0]
                ax.state = "back"
                ax.creep_to = b - ax.toward * 2.0 * SS.home_zone()
                self._march[a] = (b, ax.creep_to, self.mod.axes[a].v_max / 4.0)
            else:
                # latched: where the field is at the trip is the home
                # coordinate, and the generator stops there.  An
                # interpolating driver may be part way through a step; the
                # origin takes the fraction, so every count from here on is
                # still a whole number of steps from home
                ax.n = ax.n_from = ax.n_now
                ax.b0 = self.mod.axes[a].home - ax.n * ax.step
                ax.set = self.mod.axes[a].home
                ax.state = "homed"

    # =============================================================== TOOLS
    def extend(self, kind):
        self.d.ctrl[self.a_slide[kind]] = ToolSlide.FORCE

    def retract(self, kind):
        self.d.ctrl[self.a_slide[kind]] = -ToolSlide.FORCE

    def extended(self, kind):
        return self.d.qpos[self.j_slide[kind]] * 1000.0 >= ToolSlide.STROKE - 0.5

    def retracted(self, kind):
        return self.d.qpos[self.j_slide[kind]] * 1000.0 <= 0.5

    # ================================================================ HOLD
    def grip(self, kind):
        self._want[kind] = True
        if kind == "gripper":
            self.d.ctrl[self.a_jaw] = Gripper2.STROKE * 1e-3

    def release(self, kind):
        self._want[kind] = False
        p = self._held.get(kind)
        if p is not None:
            self.d.eq_active[self.eq[(kind, p)]] = 0
            self._held[kind] = None
            if p == "nut" and kind == "spindle":
                self._weld_now(self.lock, self.b_part[p], 0)          # the thread locks where it was left
        if kind == "gripper":
            self.d.ctrl[self.a_jaw] = 0.0

    def holding(self, kind):
        return self._held.get(kind) is not None

    def _weld_now(self, eq, b1, b2):
        """Activate a weld holding body2 where it is relative to body1."""
        R1 = self.d.xmat[b1].reshape(3, 3)
        R2 = self.d.xmat[b2].reshape(3, 3)
        self.m.eq_data[eq][0:3] = 0.0
        self.m.eq_data[eq][3:6] = R1.T @ (self.d.xpos[b2] - self.d.xpos[b1])
        self.m.eq_data[eq][6:10] = _quat_from_mat(R1.T @ R2)
        self.m.eq_data[eq][10] = 1.0
        self.d.eq_active[eq] = 1

    def _part_geom(self, p):
        """(centre, top z, radius, axis) of a part, world mm."""
        b = self.b_part[p]
        c = self.d.xpos[b] * 1000.0
        R = self.d.xmat[b].reshape(3, 3)
        if p == "nut":
            h, r = Artefact.NUT_H, Artefact.NUT_AF / 2.0
        else:
            h, r = Artefact.TOKEN_H, Artefact.TOKEN_D / 2.0
        return c, c[2] + h / 2.0 * R[2, 2], r, R[:, 2], h

    def _serve_holds(self):
        """A tool asked to hold takes a part the moment the geometry lets
        it, and the jaws stop closing once they have."""
        for k in self.tools:
            if not self._want.get(k) or self._held.get(k) is not None or k not in self.holder_body:
                continue
            tip = self.d.site_xpos[self.s_tip[k]] * 1000.0
            for p in self.parts:
                if (k, p) not in self.eq:
                    continue
                c, top, r, axis, h = self._part_geom(p)
                dxy = float(np.hypot(c[0] - tip[0], c[1] - tip[1]))
                ok = False
                if k == "gripper":
                    gap = 2.0 * (Gripper2.STROKE - self.d.qpos[self.j_jaw] * 1000.0)
                    bottom = top - h
                    overlap = min(top, tip[2] + Gripper2.JAW[2]) - max(bottom, tip[2])
                    ok = (gap <= 2.0 * r + 0.4 and abs(c[1] - tip[1]) < 1.0 and abs(c[0] - tip[0]) < Gripper2.JAW[0]
                          and overlap > 2.0)
                elif k == "vacuum":
                    ok = (p != "nut" and abs(top - tip[2]) < 0.5 and dxy < Vacuum.CUP_D / 2.0
                          and np.degrees(np.arccos(min(1.0, abs(axis[2])))) < Vacuum.TILT)
                elif k == "magnet":
                    ok = abs(top - tip[2]) < Magnet.GAP and dxy < Magnet.D / 2.0
                elif k == "spindle":
                    ok = abs(top - tip[2]) < 0.5 and dxy < (Spindle.COLLET_D / 2.0 - r) + 1.0
                if ok:
                    self._weld_now(self.eq[(k, p)], self.holder_body[k], self.b_part[p])
                    self._held[k] = p
                    if p == "nut" and k == "spindle":
                        self.d.eq_active[self.lock] = 0                     # only a turn moves it now
                    if k == "gripper":
                        self.d.ctrl[self.a_jaw] = self.d.qpos[self.j_jaw] + 0.2e-3
                    break

    # ============================================================= SPINDLE
    def spin(self, rpm):
        self.d.ctrl[self.a_sp] = float(rpm) * 2.0 * pi / 60.0

    def angle(self):
        return float(np.degrees(self.d.qpos[self.j_sp]))

    def stalled(self):
        f = abs(float(self.d.actuator_force[self.a_sp]))
        return f >= 0.95 * Spindle.TORQUE and abs(self.d.qvel[self.v_sp]) < 0.5

    # =============================================================== PROBE
    def contacts(self):
        out = []
        for j in self.j_pogo:
            c = self.d.qpos[j] * 1000.0
            out.append(Pogo.WORK[0] <= c <= Pogo.WORK[1])
        return out

    # =============================================================== FORCE
    def _raw_force(self):
        return self._lc_last * 1000.0 * self.k_lc

    def force(self):
        f = self._raw_force() - self._tare + self.rng.normal(0.0, LoadCell.NOISE, 3)
        return tuple(float(x) for x in f)

    def tare(self):
        self._tare = self._raw_force()

    # ================================================================ TICK
    def tick_hook(self, dt):
        """The firmware's own loops, once a control tick."""
        if self._lc_n:
            self._lc_last = self._lc_sum / self._lc_n
        self._lc_sum, self._lc_n = np.zeros(3), 0
        for ax in self.axes.values():
            ax.n_from = ax.n_now
        self._serve_homing(dt)
        self._serve_holds()
        for ax in self.axes.values():
            ax.update_spring(self.d)

    def pre_step(self, frac=1.0):
        for ax in self.axes.values():
            ax.stream(frac)
            ax.forces(self.d)
        if self._dist is not None:
            a, f = self._dist
            self.d.qfrc_applied[self.axes[a].vp] += f
            self.d.qfrc_applied[self.axes[a].vd] += f
        if self._block is not None:
            a, at, side = self._block
            ax = self.axes[a]
            into = (ax.carriage(self.d) - at) * side * 1e-3                   # m past the stop
            if into > 0.0:
                v = (self.d.qvel[ax.vd] + self.d.qvel[ax.vp]) * side
                f = -side * max(ax.k * into + ax.c * v, 0.0)
                self.d.qfrc_applied[ax.vp] += f
                self.d.qfrc_applied[ax.vd] += f

    def post_step(self):
        self._check_switches()
        self._lc_sum += [self.d.qpos[j] for j in self.lc]
        self._lc_n += 1

    # =============================================================== TRUTH
    # Sim only: the checks read these; the executor never does.
    def head_truth(self):
        return self.d.site_xpos[self.s_head] * 1000.0

    def tip_truth(self, kind):
        return self.d.site_xpos[self.s_tip[kind]] * 1000.0

    def carriage(self, axis):
        return self.axes[axis].carriage(self.d)

    def part_truth(self, p):
        return self.d.xpos[self.b_part[p]] * 1000.0

    def lost_steps(self, axis):
        """Full steps the rotor is behind (+) or ahead of its count, after a
        move: what the encoder should have caught."""
        ax = self.axes[axis]
        return (ax.n_now * ax.step - self.d.qpos[ax.qd] * 1000.0) / ax.step

    def warm(self, axis, eps):
        """The screw (or rack) has grown by eps since the step count began."""
        self.axes[axis].eps = float(eps)

    def disturb(self, axis, newtons=None):
        """Push the carriage along its axis with a force from outside (or
        stop pushing): a jam, a crash, a process force.  From outside, so on
        both slides: on the play alone it would push the drive back as hard
        as the carriage and the motor would never feel it."""
        self._dist = None if newtons is None else (axis, float(newtons))

    def block(self, axis, at=None):
        """Put something in the carriage's way at station coordinate `at`
        (or take it away): a jam.  A stop on the far side of where the
        carriage is now, as stiff and as damped as the drive itself."""
        if at is None:
            self._block = None
            return
        side = 1.0 if at > self.axes[axis].carriage(self.d) else -1.0
        self._block = (axis, float(at), side)


class SimClock(truss_hal.Clock):
    """One tick: the firmware's loops, then the physics steps of one
    control period, each preceded by the forces the backend computes."""

    def __init__(self, model, data, sim, hook=None):
        self.m, self.d, self.sim = model, data, sim
        self.decim = int(round(self.PERIOD / model.opt.timestep))
        self.hook = hook                     # called once a tick, after physics: a filmstrip, a logger

    def now(self):
        return float(self.d.time)

    def tick(self):
        self.sim.tick_hook(self.PERIOD)
        m, d, sim = self.m, self.d, self.sim
        for i in range(self.decim):
            sim.pre_step((i + 1.0) / self.decim)
            mujoco.mj_step(m, d)
            sim.post_step()
        if self.hook is not None:
            self.hook(self.d)


class SimCamera(truss_hal.CameraHAL):
    """The head camera, rendered.  Needs an offscreen GL context."""

    def __init__(self, model, data):
        from ..spec import HeadCam
        self.m, self.d = model, data
        self.W, self.H = HeadCam.W, HeadCam.H
        self._r = None

    def frame(self):
        if self._r is None:
            self._r = mujoco.Renderer(self.m, self.H, self.W)
        self._r.update_scene(self.d, camera="head_cam")
        return self._r.render().copy(), float(self.d.time)

    def calib(self):
        from ..spec import HeadCam
        return {"f": HeadCam.F / HeadCam.PIXEL, "cx": self.W / 2.0, "cy": self.H / 2.0}


def stiffen_riders(m):
    """MuJoCo softens each contact by an estimate of the inverse mass it
    acts on: its bodies' own, averaged over x, y and z at the reference
    pose.  A body on a light joint of its own -- a 0.5 g pogo plunger on
    its spring, a 10 g jaw -- averages that one light direction with the
    head's heavy ones, so every contact it makes comes out as soft as the
    light direction alone: a plunger pushed sideways into the reference pin
    gave 1.3 N/mm where the magnet gave 220, and on its pad it sank 0.3 mm
    into the steel.  Everything riding the tool plate is held by the head
    in every direction but its own one, so it takes the plate's estimate."""
    plate = mujoco.mj_name2id(m, B_, "plate")
    for b in range(m.nbody):
        p = b
        while p not in (0, plate):
            p = int(m.body_parentid[p])
        if p == plate and b != plate:
            m.body_invweight0[b] = m.body_invweight0[plate]


def scale_couplings(m):
    """MuJoCo softens a coupling of two joints by the sum of their inverse
    inertias, as if it were one to one.  A thread couples a turn to a lead:
    the 8 g nut's own spin, 3e-7 kg m^2, made the coupling 28,000 times
    softer than the nut it carries, and the nut fell 3.4 mm down its stud
    before the thread caught it.  The turning joint's share is scaled by
    the coupling's ratio squared, which is what it contributes."""
    for e in range(m.neq):
        if m.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT or m.eq_obj2id[e] < 0:
            continue
        ratio = float(m.eq_data[e][1])
        dof = int(m.jnt_dofadr[m.eq_obj2id[e]])
        m.dof_invweight0[dof] *= ratio ** 2


def build(mod, build=None, pieces=(), rng=None, thermal=True, render=None):
    """(model, data, backend, clock) for a station as built."""
    from . import mjcf as SM
    from ..spec import HeadCam
    xml, info = SM.scene(mod, build, pieces, render=render or (HeadCam.W, HeadCam.H))
    m = mujoco.MjModel.from_xml_string(xml)
    stiffen_riders(m)
    scale_couplings(m)
    d = mujoco.MjData(m)
    sim = StationSim(m, d, mod, info, build, rng, thermal)
    return m, d, sim, SimClock(m, d, sim)
