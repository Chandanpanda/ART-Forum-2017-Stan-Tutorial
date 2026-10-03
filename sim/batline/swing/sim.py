"""The swing rig in MuJoCo.  TRUTH: simulation and checks only; the test
(rig.swing_test) sees the HALs and nothing else.

THE RIG is rig.py's: the arm on its hinge at the hub, the bat on the
wrist's hinge at the arm's end, each body carrying rig.py's own mass
properties (so a check can hold rig.torques to MuJoCo's dynamics), and the
wrist's motor on the hub coupled to the bat's angle to the room through
its belt -- a tendon equality: the rotor's angle is the arm's plus the
wrist's.  Each rotor through its gearbox is its joint's armature.  Forward
physics, every step.

THE LOOP: computed torque (rig.torques on the plan, plus each rotor's
inertia) every step, as a drive's current loop follows its interpolated
command, and a PD on each motor's encoder at the drive's cycle
(Servo.CYCLE_S), held between.  The PD is critically damped
at a twentieth of the cycle's rate: Franklin, Powell and Workman (Digital
Control of Dynamic Systems, 3rd ed., 11.2) sample at twenty times the
closed loop's bandwidth for a response with no visible lag.  Each motor's
command is clipped at its peak torque.

THE PACK: each cell a body on a slide along the bay, its ends meeting the
+ plate and the next cell in MuJoCo contacts -- the + pip a ball its own
diameter on a flat, one frictionless contact point (a flat pip on a flat
plate gives MuJoCo four points, each taking the whole stiffness, and the
contact gains energy at the stiffness used here), -- the cap spring on the last
cell's slide (CapSpring.RATE, preloaded to the stack's length, the slide
stopped where the spring goes solid).  The spring's tip coil rides the last
cell and is folded into it: a third of the spring's mass at the swing's
largest acceleration pushes a few hundredths of a newton against its
preload (check_swing states it).  The circuit is closed while both
contacts carry force; each break goes to the firmware
(board.firmware.open_circuit), which browns out if its capacitor cannot
carry the board through it.  A contact rings at half the step's rate (the
stiffest MuJoCo integrates stably, a hundred times the cap spring's own
rate on the pack), damped to ContactSet.RESTITUTION; the spring's solid
stop is the same.

THE IMU: MuJoCo's gyro and accelerometer at the chip as drawn, every
step.  The step is the internal rate model.internal_rate settles on for
the planned run, so imu/model.sense takes the readings as they come.
"""
from dataclasses import dataclass
from math import pi, sqrt, log, ceil

import numpy as np
import mujoco

from ..spec import Servo, CapSpring, ContactSet, Imu, Site, Link, Rules, DummyCell, Cell
from ..imu import model as MD, kin as K
from ..imu.part import impulse, lsb
from ..board import sim as BS, power as P, procedures as PR
from . import rig as SR
from .hal import SwingRigHAL


def damping_ratio(e):
    """The damping ratio of a linear spring-damper that leaves a contact at
    e times the speed it struck at."""
    return -log(e) / sqrt(pi * pi + log(e) ** 2)


def _inertial(mp):
    I = mp.I
    return ('<inertial pos="%.9g %.9g %.9g" mass="%.9g" fullinertia="%.9g %.9g %.9g %.9g %.9g %.9g"/>'
            % (*mp.c, mp.m, I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]))


def _quat(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


# ================================================================ THE STEP
_rates = {}


def internal_rate(d):
    """Hz: model.internal_rate over the planned run at the chip, a perfect
    part where the drawing puts it."""
    if d.bat in _rates:
        return _rates[d.bat]
    rig, mo = SR.fit(d), SR.motion(d)
    zero = MD.ImuErrors(np.zeros((3, 3)), np.zeros((3, 3)), np.zeros(3), np.zeros(3), np.zeros((3, 3)),
                        np.zeros(3), np.zeros(3), 0.0)
    pw = MD.PowerUp(np.zeros(3), np.zeros(3), -1e6, Site.AMBIENT)
    lg, la = lsb()
    scale = np.array([lg] * 3 + [la] * 3)
    t0 = mo.t_swing
    n = int(rig.swing.t_end * Imu.ODR) - 2

    def probe(fs):
        T_ = t0 - 0.05 + np.arange(int(ceil((rig.swing.t_end + 0.1) * fs)) + 1) / fs
        w, f = SR.imu_signals(rig, *mo.state(T_))
        _, _, ys = MD.sense(zero, pw, T_, w, f, t0, n, None, noise=False, quantise=False)
        return ys / scale
    _rates[d.bat] = MD.internal_rate(probe)[0]
    return _rates[d.bat]


# ================================================================ THE SCENE
RING = 4.0


def xml(rig, drives, pack, dt, rate=CapSpring.RATE, restitution=ContactSet.RESTITUTION):
    d = rig.d
    lay = d.lay
    hx = lay.x_hands
    da, dw = drives
    cells, spring_len = SR.seated(d, pack)
    F0 = rate * (CapSpring.L_FREE - spring_len)
    z = -d.core.e / 1000.0
    h = ContactSet.T / 2000.0                      # m: half a contact element's thickness
    w = min(pack.D, 2.0 * d.core.r_bay) / 4000.0   # m: half its width, inside the cell's end
    r_pip = Cell.PIP_D / 2000.0                    # m: the + pip as a ball its diameter
    zeta = damping_ratio(restitution)
    # MuJoCo's solref is (the damping's time constant, the damping ratio): a
    # natural frequency of half the step's rate is 2 dt / zeta
    sol = 'solref="%.9g %.9g"' % (RING * dt / zeta, zeta)
    p_imu, R_imu = rig.imu
    q_imu = _quat(R_imu)
    bat = SR.bat_props(d, None)
    cell_bodies = []
    for i, (x0, x1) in enumerate(cells):
        L = (x1 - x0) / 1000.0
        c = SR.cell_props(d, pack)[i]
        mp = SR.si(c)
        xc = ((x0 + x1) / 2.0 - hx) / 1000.0
        last = i == len(cells) - 1
        joint = ('<joint name="cell%d" type="slide" axis="1 0 0"%s/>'
                 % (i, (' stiffness="%.9g" springref="%.9g" limited="true" range="%.9g 1"'
                        % (rate * 1000.0, F0 / (rate * 1000.0), -(spring_len - CapSpring.L_SOLID) / 1000.0)
                    + ' solreflimit="%.9g %.9g"' % (RING * dt / zeta, zeta))
                    if last else ""))
        cell_bodies.append("""
        <body name="cell%d" pos="%.9g 0 %.9g">
          %s
          %s
          <geom name="cell%d_body" type="cylinder" fromto="%.9g 0 0 %.9g 0 0" size="%.9g" contype="0" conaffinity="0" mass="0" rgba="0.75 0.62 0.15 1"/>
          <geom name="cell%d_plus" type="sphere" pos="%.9g 0 0" size="%.9g" contype="0" conaffinity="0" mass="0"/>
          <geom name="cell%d_minus" type="box" pos="%.9g 0 0" size="%.9g %.9g %.9g" contype="0" conaffinity="0" mass="0"/>
        </body>""" % (i, xc, z, joint, _inertial(MassAt(mp, -np.array([xc + hx / 1000.0, 0.0, z]))),
                      i, -L / 2.0, L / 2.0, pack.D / 2000.0,
                      i, L / 2.0 - r_pip, r_pip, i, -L / 2.0 + h, h, w, w))
    pairs = ['<pair geom1="plate" geom2="cell0_plus" condim="1" %s/>' % sol]
    pairs += ['<pair geom1="cell%d_minus" geom2="cell%d_plus" condim="1" %s/>' % (i, i + 1, sol)
              for i in range(len(cells) - 1)]
    s_t = SR.SwingRig.TUBE[0] / 2000.0
    xml_ = """<mujoco model="swing rig">
  <compiler angle="radian"/>
  <option timestep="%(dt).12g" gravity="%(g)s" integrator="implicitfast"/>
  <visual><global offwidth="1280" offheight="800"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.82 0.80" rgb2="0.74 0.74 0.72" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light pos="0 -2 2" dir="0 1 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" pos="0 0 %(floor).9g" size="3 3 0.01" material="grid" contype="0" conaffinity="0"/>
    <geom name="hub" type="cylinder" pos="%(hx).9g 0.06 %(hz).9g" euler="1.5708 0 0" size="0.05 0.05" rgba="0.3 0.3 0.32 1" contype="0" conaffinity="0"/>
    <body name="rotor" pos="%(hx).9g 0 %(hz).9g">
      <joint name="rot" type="hinge" axis="0 -1 0" armature="%(arm_w).12g"/>
      <inertial pos="0 0 0" mass="1" diaginertia="1e-6 1e-6 1e-6"/>
    </body>
    <body name="arm" pos="%(hx).9g 0 %(hz).9g">
      <joint name="phi" type="hinge" axis="0 -1 0" armature="%(arm_a).12g"/>
      %(arm_in)s
      <geom type="box" pos="%(Lh).9g 0 0" size="%(Lh).9g %(st).9g %(st).9g" rgba="0.70 0.72 0.75 1" contype="0" conaffinity="0" mass="0"/>
      <body name="bat" pos="%(L).9g 0 0">
        <joint name="q2" type="hinge" axis="0 -1 0"/>
        %(bat_in)s
        <site name="imu" pos="%(pi)s" quat="%(qi)s" size="0.003"/>
        <geom name="shaft" type="capsule" fromto="%(x0).9g 0 0 %(x1).9g 0 0" size="%(rg).9g" rgba="0.20 0.35 0.65 0.35" contype="0" conaffinity="0" mass="0"/>
        <geom name="plate" type="box" pos="%(xp).9g 0 %(z).9g" size="%(h).9g %(w).9g %(w).9g" contype="0" conaffinity="0" mass="0" rgba="0.8 0.8 0.8 1"/>
        %(cells)s
      </body>
    </body>
  </worldbody>
  <contact>
    %(pairs)s
  </contact>
  <tendon>
    <fixed name="abs"><joint joint="phi" coef="1"/><joint joint="q2" coef="1"/></fixed>
    <fixed name="rot"><joint joint="rot" coef="1"/></fixed>
  </tendon>
  <equality>
    <tendon tendon1="rot" tendon2="abs" polycoef="0 1 0 0 0" solref="%(eq).12g 1"/>
  </equality>
  <actuator>
    <motor name="arm" joint="phi" gear="%(Na).9g" ctrllimited="true" ctrlrange="-%(Ta).9g %(Ta).9g"/>
    <motor name="wrist" joint="rot" gear="%(Nw).9g" ctrllimited="true" ctrlrange="-%(Tw).9g %(Tw).9g"/>
  </actuator>
  <sensor>
    <gyro name="gyro" site="imu"/>
    <accelerometer name="acc" site="imu"/>
  </sensor>
</mujoco>
""" % dict(dt=dt, g="%.9g %.9g %.9g" % tuple(rig.g), floor=-rig.L - 1.0, hx=rig.hub[0], hz=rig.hub[1],
           arm_w=da_rotor(dw), arm_a=da_rotor(da), arm_in=_inertial(rig.arm), Lh=rig.L / 2.0, st=s_t, L=rig.L,
           bat_in=_inertial(bat), pi="%.9g %.9g %.9g" % tuple(p_imu), qi="%.9g %.9g %.9g %.9g" % tuple(q_imu),
           x0=-hx / 1000.0, x1=(lay.x_tip1 - hx) / 1000.0, rg=lay.r_grip / 1000.0,
           xp=(lay.x_plus - hx) / 1000.0 + h, z=z, h=h, w=w, cells="".join(cell_bodies),
           pairs="\n    ".join(pairs), eq=2.0 * dt, Na=da.N, Ta=da.motor[2], Nw=dw.N, Tw=dw.motor[2])
    return xml_


def da_rotor(dr):
    """kg m^2: a drive's rotor as its joint feels it, through its gearbox."""
    return dr.motor[5] * dr.N ** 2


def MassAt(mp, shift):
    """mp with its centre moved by `shift` (a body's frame from the bat's)."""
    return SR.MassProps(mp.m, mp.c + shift, mp.I)


# ================================================================ THE RUN
@dataclass
class Run:
    t: np.ndarray              # (n,) s from the motion's start, each step's start
    q: np.ndarray              # (n, 2) rad: (phi, alpha), true
    w: np.ndarray              # (n, 3) rad/s, chip axes, the room still
    f: np.ndarray              # (n, 3) m/s^2
    closed: np.ndarray         # (n,) bool: both contacts carry force
    opens: list                # [(t0, t1)] s: each break of the circuit
    force: np.ndarray          # (n,) N: the + plate's contact force
    cmd: np.ndarray            # (n, 2) N m: each motor's command
    sat: np.ndarray            # (n, 2) bool: a command at its clip
    err: np.ndarray            # (n, 2) rad: plan less truth
    qd: np.ndarray             # (n, 2) rad/s: (phi, alpha)'s rates, true
    qdd: np.ndarray            # (n, 2) rad/s^2: and accelerations
    cells: np.ndarray          # (n, pack.n) m: each cell's slide from where it was drawn seated


class Sim:
    """One rig with one pack in MuJoCo."""

    def __init__(self, d, pack=SR.DUMMY, rate=CapSpring.RATE, restitution=ContactSet.RESTITUTION, drives=None,
                 dt=None):
        self.d, self.pack = d, pack
        self.rig = SR.fit(d)
        self.drives = SR.choose(self.rig) if drives is None else drives
        self.dt = 1.0 / internal_rate(d) if dt is None else dt
        self.per = int(round(Servo.CYCLE_S / self.dt))
        if abs(self.per * self.dt - Servo.CYCLE_S) > 1e-9 * Servo.CYCLE_S:
            raise ValueError("the drive's cycle is not a whole number of physics steps")
        self.xml = xml(self.rig, self.drives, pack, self.dt, rate, restitution)
        self.m = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.m)
        m = self.m
        self.j = {n: (m.joint(n).qposadr[0], m.joint(n).dofadr[0]) for n in ("phi", "q2", "rot")}
        self.jc = np.array([m.joint("cell%d" % i).qposadr[0] for i in range(pack.n)])
        self.gplate = m.geom("plate").id
        self.pairs = [(m.geom("plate").id, m.geom("cell0_plus").id)]
        self.pairs += [(m.geom("cell%d_minus" % i).id, m.geom("cell%d_plus" % (i + 1)).id)
                       for i in range(pack.n - 1)]
        self.s_gyro = slice(m.sensor("gyro").adr[0], m.sensor("gyro").adr[0] + 3)
        self.s_acc = slice(m.sensor("acc").adr[0], m.sensor("acc").adr[0] + 3)
        wc = 2.0 * pi / Servo.CYCLE_S / 20.0
        self.kp, self.kd = wc * wc, 2.0 * wc
        self.res = np.array([2.0 * pi / 2 ** Servo.ENCODER_BITS / dr.N for dr in self.drives])

    def place(self, q):
        """The rig still at (phi, alpha) q, the pack seated."""
        dd = self.data
        mujoco.mj_resetData(self.m, dd)
        dd.qpos[self.j["phi"][0]] = q[0]
        dd.qpos[self.j["q2"][0]] = q[1] - q[0]
        dd.qpos[self.j["rot"][0]] = q[1]
        mujoco.mj_forward(self.m, dd)

    def angles(self):
        dd = self.data
        return np.array([dd.qpos[self.j["phi"][0]], dd.qpos[self.j["phi"][0]] + dd.qpos[self.j["q2"][0]]])

    def contact_forces(self):
        """(n_pairs,) N: each contact's normal force, summed over its points."""
        dd = self.data
        out = np.zeros(len(self.pairs))
        f6 = np.zeros(6)
        for i in range(dd.ncon):
            c = dd.contact[i]
            key = (c.geom1, c.geom2)
            for k, pr in enumerate(self.pairs):
                if key == pr or key == pr[::-1]:
                    mujoco.mj_contactForce(self.m, dd, i, f6)
                    out[k] += f6[0]
        return out

    def _drive(self, Qd, QDd, QDDd, record):
        """Steps the rig along the plan (Qd, QDd, QDDd), one row a step;
        each step's records if `record`."""
        rig, dd, n = self.rig, self.data, len(Qd)
        J = SR.load_inertia(rig)
        Jr = np.array([da_rotor(dr) for dr in self.drives])
        N = np.array([dr.N for dr in self.drives])
        Tpk = np.array([dr.motor[2] for dr in self.drives])
        FF = (SR.torques(rig, Qd, QDd, QDDd) + Jr * QDDd) / N
        out = (np.zeros((n, 2)), np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, len(self.pairs))),
               np.zeros((n, 2)), np.zeros((n, 2), bool), np.zeros((n, 2)), np.zeros((n, 2)),
               np.zeros((n, len(self.jc)))) if record else None
        ip, iq = self.j["phi"][1], self.j["q2"][1]
        enc_prev, fb = None, np.zeros(2)
        for k in range(n):
            if k % self.per == 0:
                enc = np.round(self.angles() / self.res) * self.res
                vel = QDd[k] if enc_prev is None else (enc - enc_prev) / Servo.CYCLE_S
                enc_prev = enc
                fb = (J + Jr) * (self.kp * (Qd[k] - enc) + self.kd * (QDd[k] - vel)) / N
            raw = FF[k] + fb
            dd.ctrl[:] = np.clip(raw, -Tpk, Tpk)
            q = self.angles()
            qd = np.array([dd.qvel[ip], dd.qvel[ip] + dd.qvel[iq]])
            cq = dd.qpos[self.jc]
            mujoco.mj_step(self.m, dd)                 # its sensors, contacts and qacc are the step's start's
            if record:
                Q, W, F_, FC, U, S, QD, QDD, CQ = out
                Q[k], QD[k], CQ[k] = q, qd, cq
                QDD[k] = dd.qacc[ip], dd.qacc[ip] + dd.qacc[iq]
                W[k] = dd.sensordata[self.s_gyro]
                F_[k] = dd.sensordata[self.s_acc]
                FC[k] = self.contact_forces()
                U[k] = dd.ctrl
                S[k] = np.abs(raw) > Tpk
        return out

    def settle(self, q):
        """The rig held at q from rest until the pack's forces stop changing:
        blocks of one contact period until a block's spread is under a
        thousandth of its mean, a second at most."""
        self.place(q)
        blk = int(ceil(2.0 * pi * RING))
        z = np.zeros((blk, 2))
        for _ in range(int(ceil(1.0 / (blk * self.dt)))):
            fc = self._drive(q + z, z, z, True)[3]
            if np.all(fc > 0.0) and np.all(np.ptp(fc, 0) < 1e-3 * fc.mean(0)):
                return
        raise ValueError("the pack did not settle in the rig")

    def run(self, mo, extra=0.0):
        """Run: the rig through Motion mo from its park pose, the pack
        settled there first, and `extra` s held at its end."""
        q0 = mo.state(np.array([0.0]))[0][0]
        self.settle(q0)
        n = int(ceil((mo.T + extra) / self.dt))
        T = np.arange(n) * self.dt
        Qd, QDd, QDDd = mo.state(T)
        Q, W, F_, FC, U, S, QD, QDD, CQ = self._drive(Qd, QDd, QDDd, True)
        C = np.all(FC > 0.0, axis=1)
        return Run(T, Q, W, F_, C, breaks(T, C, self.dt), FC[:, 0], U, S, Qd - Q, QD, QDD, CQ)


def breaks(t, closed, dt):
    """[(t0, t1)]: each stretch the circuit is open, from its first open
    step to its next closed one."""
    edge = np.diff(np.concatenate([[0], (~np.asarray(closed)).astype(int), [0]]))
    starts, ends = np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)
    return [(float(t[a]), float(t[b]) if b < len(t) else float(t[-1] + dt)) for a, b in zip(starts, ends)]


# ================================================================ THE IMU ON THE BENCH
class Timeline:
    """What the chip feels against true time: at rest at the park pose
    until the run, the run's readings during it, at rest where it ended."""

    def __init__(self, rest, earth):
        self.rest = rest                           # (w (3,), f (3,)) still at park
        self.earth = earth                         # rad/s, the rig's frame
        self.runs = []                             # [(t0, Run, (w, f) at the end)]

    def add(self, t0, run, end):
        self.runs.append((float(t0), run, end))

    def still(self, t0, t1):
        return all(t1 < s or t0 > s + r.t[-1] for s, r, _ in self.runs)

    def signals(self, T_):
        T_ = np.asarray(T_, float)
        w = np.tile(self.rest[0], (len(T_), 1))
        f = np.tile(self.rest[1], (len(T_), 1))
        for t0, r, (w1, f1) in self.runs:
            s = T_ - t0
            m = s >= 0.0
            inside = m & (s <= r.t[-1])
            for i in range(3):
                w[inside, i] = np.interp(s[inside], r.t, r.w[:, i])
                f[inside, i] = np.interp(s[inside], r.t, r.f[:, i])
            after = s > r.t[-1]
            w[after], f[after] = w1, f1
        return w, f


class SwingSource(BS.Source):
    """The board's IMU on the rig this power-up, a packet at a time."""

    def __init__(self, err, pw, timeline, rng, t_first, fs):
        super().__init__(t_first, err.eps)
        self.err, self.pw, self.tl, self.rng, self.fs = err, pw, timeline, rng, fs

    def codes(self, k0, n):
        ta, tb = float(self.t(k0)), float(self.t(k0 + n - 1))
        fs = self.fs if not self.tl.still(ta - len(impulse(self.fs)) / self.fs, tb) \
            else min(self.fs, BS.StillSource.RATE)
        lead = len(impulse(fs)) / fs
        T_ = np.arange(ta - lead, tb + 2.0 / fs, 1.0 / fs)
        w, f = self.tl.signals(T_)
        st, _, _ = MD.sense(self.err, self.pw, T_, w, f, ta, n, self.rng)
        return np.concatenate([st.acc, st.gyro], axis=1).astype(int), int(st.temp[-1])


def still_reading(rig, q, earth):
    """(w, f) chip axes: the chip still on the rig at (phi, alpha) q."""
    z = np.zeros((1, 2))
    w, f = SR.imu_signals(rig, np.atleast_2d(q), z, z)
    Rb = SR.rot3(np.atleast_1d(q[1]))[0]
    return w[0] + rig.imu[1].T @ Rb.T @ earth, f[0]


class SimSwingRig(SwingRigHAL):
    def __init__(self, bench, sim, timeline, earth):
        self.b, self.sim, self.tl, self.earth = bench, sim, timeline, earth
        self.q = None
        self.last = None

    def run(self, mo):
        q0 = mo.state(np.array([0.0]))[0][0]
        if self.q is not None:
            off = q0 - self.q
            off[1] -= 2.0 * pi * round(off[1] / (2.0 * pi))   # the wrist's encoder counts whole turns
            if np.max(np.abs(off)) > np.max(self.sim.res):
                raise ValueError("a motion must start where the rig stands")
        t0 = self.b.t()
        r = self.sim.run(mo)
        Rb = SR.rot3(r.q[:, 1])
        r.w = r.w + np.einsum("ij,nkj,k->ni", self.sim.rig.imu[1].T, Rb, self.earth)
        self.tl.add(t0, r, still_reading(self.sim.rig, r.q[-1], self.earth))
        for a, b in r.opens:
            self.b.fw.open_circuit(t0 + a, t0 + b)
        self.q = r.q[-1]
        self.t_end = t0 + r.t[-1]
        self.last = r

    def angles(self):
        q = self.q if self.q is not None else self.park
        return np.round(q / self.sim.res) * self.sim.res


@dataclass
class SwingBench:
    bench: object              # board.sim.Bench, the dongle its central
    station: object            # rig.SwingStation
    serial: str
    sim: Sim
    timeline: Timeline


def ready(d, seed, pack=SR.DUMMY, rate=CapSpring.RATE, hold_c=None, per=None):
    """A flashed bat on the dummy pack in the rig at its park pose,
    advertising to the station's dongle."""
    rng = np.random.default_rng(seed)
    sim = Sim(d, pack, rate)
    mo = SR.motion(d)
    rig = sim.rig
    up = -rig.g / np.linalg.norm(rig.g)
    earth = K.earth_rate(Site.LAT, up, K.north_rig(Site.HEADING))
    tl = Timeline(still_reading(rig, mo.park, earth), earth)
    board = BS.draw_board(rng)
    if hold_c is not None:
        board.truth.hold_c = hold_c
    fs = 1.0 / sim.dt

    def source_for(t_on):
        pw = MD.power_up(rng, t_on, Site.AMBIENT)
        t_first = t_on + Imu.START_S + float(rng.uniform(0.0, 1.0)) / Imu.ODR
        return SwingSource(board.err, pw, tl, rng, t_first, fs)
    b = BS.Bench(board, rng, central="dongle", per=per, source_for=source_for)
    serial = "BAT%05d" % seed
    # the line flashes the board at station 2 (board.procedures): here it
    # arrives flashed, and the pack powers it in the rig
    b.fw.image = PR.image()
    b.fw.uicr = serial.encode().ljust(16, b"\0") + b.fw.uicr[16:]
    b.fw.supply(b.t(), Rules.N_CELLS * P.dummy_volts(), Rules.N_CELLS * DummyCell.ESR)
    b.clock.wait(Imu.START_S + 2.0 * Link.ADV_JITTER)
    rg = SimSwingRig(b, sim, tl, earth)
    rg.park = mo.park
    return SwingBench(b, SR.SwingStation(b.clock, b.radio, rg), serial, sim, tl)
