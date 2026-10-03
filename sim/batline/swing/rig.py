"""The swing rig at station 4: one bat in Est.SWING_EVERY swung through the
rated swing at full speed with its board streaming, to show its IMU does
not rail and its cells do not let go.  Production, numpy only.

    fit(d)            Rig: the hub and the arm fitted to the rated swing's
                      hand path, the bodies' masses
    swing_state(rig)  the swing as the rig makes it: each motor's angle, rate
                      and acceleration
    imu_signals(...)  what the chip feels on the rig, by the rigid body's law
    torques(rig, q)   what each motor's joint must give (Newton-Euler)
    choose(rig)       each joint's motor and gearbox from Servo.CATALOGUE
    motion(d)         the rig's run: from its park pose to the stance, the
                      swing, back
    swing_test(...)   the station's procedure
    plan(d), plan_s   its planned time, for line.py

THE RIG.  The rated swing (imu_fusion_sim.swings(), Rules.RATED_SWING)
turns the bat in a plane while the hands ride an arc in it.  The rig keeps
both in the plane: an arm on a hub carries the hands round a circle fitted
to their arc -- Kasa's algebraic fit (Kasa 1976, IEEE Trans. Instrum.
Meas. 25: 8), on points spread evenly along the arc, not in time, or the
still stance would outweigh the swing -- and the bat turns on a wrist at
the arm's end through the swing's own angle in the plane.  The wrist's
motor stands on the hub and turns the wrist through a belt
(SwingRig.BELT_RATIO), so it sets the bat's angle to the room, not to the
arm, and no motor rides the arm.  The rig leaves out the swing's roll of
the face, its tilt out of the plane (Swing.tilt_amp_deg) and the hands'
few millimetres off the circle (Rig.dev); check_swing holds what the rig
does to the cells and the IMU against the swing's own.

THE FRAME.  The rig's x and z are the swing plane's axes (Swing.P: along
the ball's line, and up for a drive); y = z x x.  Angles turn x toward z
(about -y): phi the arm's, alpha the bat's axis', both to the room.  At
alpha = 0 the bat's frame (product.py) is the rig's: the face, product -z,
is the face normal the swing leads with, so the swing turns the bat about
its own y.  The bat's frame here starts at the hands, product's
lay.x_hands.

THE MOTORS.  Each joint's torque over the swing comes from Newton-Euler on
the two bodies (torques), with each motor's rotor reflected through its
gearbox; a motor and ratio serve a joint when its peak torque, its rated
torque against the swing's RMS, its top speed and Servo.INERTIA_RATIO all
hold.  choose takes the smallest motor in the catalogue that has a ratio
doing so, and of its ratios the one with the most room left.

THE TEST.  The bat rides the rig on the dummy pack (spec.DummyCell),
connected to the station's dongle and streaming.  The rig runs from its
park pose (the bat level, face down, where the head lays it in) to the
stance, swings, and goes back, each move rest to rest on one minimum-jerk
profile for both joints (Flash and Hogan 1985, J. Neurosci. 5: 1688): a
servo's moves are jerk-limited, and a trapezoid's step in acceleration is
a step of force on the pack that no drive's current loop delivers.  The
bat must still be there on the same
boot after (the battery circuit never opened past the hold-up), its
stream unbroken, and no sample at a rail.
"""
from dataclasses import dataclass
from math import pi, sqrt, log, erfc

import numpy as np

from ..spec import Rules, Servo, SwingRig, Material, Link, Imu, Est, Cell, DummyCell, CapSpring, Module
from .. import imu_fusion_sim as study
from ..product import MassProps, rod_props
from ..imu import plan as IP
from ..board import procedures as PR, hal as H

D_FINE = 1e-6                  # s, the central differences' step on the study's analytic swing
TAIL = 0.5 * erfc(Module.Z_SIGMAS / sqrt(2.0))


# ================================================================ THE BODIES
def si(mp):
    """product.MassProps (g, mm, g mm^2) in kg, m, kg m^2."""
    return MassProps(mp.m * 1e-3, mp.c * 1e-3, mp.I * 1e-9)


@dataclass(frozen=True)
class Pack:
    """What sits in the bay: n cells each L mm long, D mm across and m g."""
    name: str
    L: float
    D: float
    m: float
    n: int = Rules.N_CELLS


DUMMY = Pack("the dummy pack", DummyCell.L, DummyCell.D, Cell.MASS)
SHORTEST = Pack("the shortest pair of cells", Cell.L_MIN, Cell.D_MIN, Cell.MASS)


def seated(d, pack):
    """[(x0, x1) mm] of each cell, product frame, from the + plate back,
    pressed home by the spring; and the spring's length then (mm)."""
    x_plus = d.lay.x_plus
    cells = [(x_plus - (i + 1) * pack.L, x_plus - i * pack.L) for i in range(pack.n)]
    return cells, cells[-1][0] - d.lay.t_end


def spring_force(d, pack):
    """N the cap spring presses the pack onto the + plate with."""
    return CapSpring.RATE * (CapSpring.L_FREE - seated(d, pack)[1])


def cell_props(d, pack):
    """[MassProps] of each cell where it sits, product frame, g mm."""
    z = -d.core.e
    return [rod_props((x0, 0.0, z), (x1, 0.0, z), pack.D / 2.0, pack.m) for x0, x1 in seated(d, pack)[0]]


def bat_props(d, pack=None):
    """MassProps of what turns on the wrist, SI, the bat's frame at the
    hands: the bat without its cells, the clamp at the hands, and the pack
    seated if one is given."""
    out = d.props(cells=False) + MassProps(SwingRig.CLAMP_MASS, (d.lay.x_hands, 0.0, 0.0))
    for c in (cell_props(d, pack) if pack is not None else []):
        out = out + c
    out = si(out)
    return MassProps(out.m, out.c - np.array([d.lay.x_hands * 1e-3, 0.0, 0.0]), out.I)


def arm_props(L):
    """MassProps of the arm, SI, its own frame: x from the hub to the wrist.
    The tube and the wrist's housing at its end."""
    s, t = SwingRig.TUBE
    si_ = s - 2.0 * t
    m = Material.ALU * (s * s - si_ * si_) * L * 1e3 * 1e-3          # kg
    k2 = (s * s + si_ * si_) * 1e-6                                  # m^2: a square tube's section
    tube = MassProps(m, (L / 2.0, 0.0, 0.0), m * np.diag([k2 / 6.0, L * L / 12.0 + k2 / 12.0,
                                                          L * L / 12.0 + k2 / 12.0]))
    return tube + MassProps(SwingRig.WRIST_MASS * 1e-3, (L, 0.0, 0.0))


def rot2(a):
    """(n, 2, 2): the turn by a in the x-z plane, x toward z."""
    c, s = np.cos(a), np.sin(a)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


def rot3(a):
    """(n, 3, 3): the same turn as a 3-D rotation (about -y)."""
    c, s = np.cos(a), np.sin(a)
    z, o = np.zeros_like(c), np.ones_like(c)
    return np.stack([np.stack([c, z, -s], -1), np.stack([z, o, z], -1), np.stack([s, z, c], -1)], -2)


def cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def perp(r):
    """J r: r turned a quarter, x toward z."""
    return np.stack([-r[..., 1], r[..., 0]], -1)


# ================================================================ THE FIT
@dataclass
class Rig:
    d: object
    swing: object              # imu_fusion_sim.Swing, the rated one
    hub: np.ndarray            # (2,) m, x z: the arm's pivot from the hands' arc's origin (Swing.Hc)
    L: float                   # m, hub to wrist
    dev: float                 # m, the hands' farthest off the circle
    g: np.ndarray              # (3,) m/s^2, gravity in the rig's frame
    bat: MassProps             # SI, the bat's frame at the hands, the dummy pack in
    arm: MassProps             # SI, the arm's frame
    imu: tuple                 # (p m, R): the chip in the bat's frame at the hands


def kasa(P):
    """(centre (2,), radius) of the circle through points P (n, 2) by Kasa's
    linear least squares."""
    A = np.column_stack([P, np.ones(len(P))])
    c = np.linalg.lstsq(A, np.sum(P * P, 1), rcond=None)[0]
    centre = c[:2] / 2.0
    return centre, float(sqrt(c[2] + centre @ centre))


def plane_axes(sw):
    """(3, 3): the rig's x, y, z in the study's world."""
    x, z = sw.P[:, 0], sw.P[:, 1]
    return np.stack([x, np.cross(z, x), z], 1)


_rigs = {}


def fit(d, g=None):
    """Rig for design d (THE RIG), once per bat."""
    if d.bat in _rigs and g is None:
        return _rigs[d.bat]
    sw = next(s for s in study.swings() if s.name == Rules.RATED_SWING)
    # the hands' arc, sampled evenly along the bat's angle between its ends
    th = sw.theta(np.array([sw.t_top, sw.t_end]))
    t_arc = arc_times(sw, np.linspace(th.min(), th.max(), 721))
    P = hands(sw, t_arc)
    hub, L = kasa(P)
    dev = float(np.max(np.abs(np.linalg.norm(P - hub, axis=1) - L)))
    g_w = study.GW if g is None else np.asarray(g, float)
    p, R = IP.imu_in_bat(d)
    imu = ((p - np.array([d.lay.x_hands, 0.0, 0.0])) / 1000.0, R)
    rig = Rig(d, sw, hub, L, dev, plane_axes(sw).T @ g_w, bat_props(d, DUMMY), arm_props(L), imu)
    if g is None:
        _rigs[d.bat] = rig
    return rig


def arc_times(sw, thetas):
    """Times in the swing, from its top to its end, at which the bat's angle
    takes each of `thetas` (the swing's angle is monotone from the top to
    the follow-through's end)."""
    t = np.linspace(sw.t_top, sw.t_end, 200001)
    th = sw.theta(t)
    order = np.argsort(th)
    return np.interp(thetas, th[order], t[order])


def hands(sw, t):
    """(n, 2) m: the hands in the swing plane, x z, from the arc's origin."""
    _, H_ = sw.pose(np.asarray(t, float))
    return (H_ - sw.Hc) @ sw.P


# ================================================================ THE SWING AS THE RIG MAKES IT
def swing_angles(rig, t):
    """(n, 2) rad: (phi, alpha) at swing times t, unwrapped from the stance."""
    h = hands(rig.swing, t) - rig.hub
    phi = np.arctan2(h[:, 1], h[:, 0])
    alpha = rig.swing.theta(t) - pi / 2.0
    return np.stack([phi, alpha], 1)


def swing_state(rig, t):
    """(q, qd, qdd), each (n, 2): (phi, alpha), their rates and
    accelerations at swing times t, by central differences on the study's
    analytic swing."""
    t = np.asarray(t, float)
    a, b, c = (swing_angles(rig, t + k * D_FINE) for k in (-1.0, 0.0, 1.0))
    for x in (a, c):                               # one branch of atan2 across the three
        x -= 2.0 * pi * np.round((x - b) / (2.0 * pi))
    return b, (c - a) / (2.0 * D_FINE), (c - 2.0 * b + a) / D_FINE ** 2


def points(rig, q):
    """((n, 2) wrist, (n, 2) bat axis) in the rig's plane from the hub, at
    (phi, alpha) q."""
    q = np.atleast_2d(q)
    u = np.stack([np.cos(q[:, 0]), np.sin(q[:, 0])], 1)
    return rig.L * u, np.stack([np.cos(q[:, 1]), np.sin(q[:, 1])], 1)


def imu_signals(rig, q, qd, qdd):
    """((n, 3) rad/s, (n, 3) m/s^2): the chip's rate and specific force,
    its own axes, at (phi, alpha) states: the rigid body's law at the chip
    as drawn (Rig.imu), the room still."""
    p, R = rig.imu
    phi, al = q[:, 0], q[:, 1]
    u = np.stack([np.cos(phi), np.sin(phi)], 1)
    Rb = rot3(al)
    r = np.einsum("nij,j->ni", Rb, p)[:, [0, 2]]
    a2 = rig.L * (qdd[:, :1] * perp(u) - qd[:, :1] ** 2 * u) + qdd[:, 1:] * perp(r) - qd[:, 1:] ** 2 * r
    a = np.zeros((len(q), 3))
    a[:, [0, 2]] = a2
    f = np.einsum("nji,nj->ni", Rb, a - rig.g)
    w = np.zeros((len(q), 3))
    w[:, 1] = -qd[:, 1]                            # alpha turns about -y, the bat's own y
    return w @ R, f @ R


# ================================================================ THE TORQUES
def torques(rig, q, qd, qdd, bat=None):
    """(n, 2) N m: what the arm's joint and the wrist's (to the room) must
    give at states (phi, alpha), by Newton-Euler on the arm and the bat
    (THE MOTORS); `bat` is the bat's MassProps if not the rig's."""
    bat = rig.bat if bat is None else bat
    g2 = rig.g[[0, 2]]
    phi, al = q[:, 0], q[:, 1]
    u = np.stack([np.cos(phi), np.sin(phi)], 1)
    rb = np.einsum("nij,j->ni", rot2(al), bat.c[[0, 2]])
    aW = rig.L * (qdd[:, :1] * perp(u) - qd[:, :1] ** 2 * u)
    aC = aW + qdd[:, 1:] * perp(rb) - qd[:, 1:] ** 2 * rb
    Fw = bat.m * (aC - g2)
    tau_w = bat.I[1, 1] * qdd[:, 1] + cross2(rb, Fw)
    I_hub = rig.arm.about(np.zeros(3))[1, 1]
    ra = np.einsum("nij,j->ni", rot2(phi), rig.arm.c[[0, 2]])
    tau_a = I_hub * qdd[:, 0] - cross2(ra, rig.arm.m * np.broadcast_to(g2, ra.shape)) + cross2(rig.L * u, Fw)
    return np.stack([tau_a, tau_w], 1)


def load_inertia(rig):
    """(2,) kg m^2: each joint's inertia as its motor sees it, the other
    held (THE MOTORS)."""
    return np.array([rig.arm.about(np.zeros(3))[1, 1] + rig.bat.m * rig.L ** 2,
                     rig.bat.about(np.zeros(3))[1, 1]])


# ================================================================ THE MOTORS
@dataclass
class Drive:
    motor: tuple               # a Servo.CATALOGUE row
    ratio: float               # the gearbox's
    N: float                   # motor turns a joint turn: the gearbox and, at the wrist, the belt
    use: dict                  # {what: share of the motor's limit the swing takes}

    @property
    def name(self):
        return "%s through %.0f:1" % (self.motor[0], self.ratio)

    def worst(self):
        return max(self.use.items(), key=lambda kv: kv[1])


def drive_use(motor, N, tau, w, wd, J_load):
    """{limit: share} of a motor through N turning a joint's tau, w, wd."""
    _, T_rated, T_peak, _, rpm_max, J_r = motor
    Tm = J_r * N * wd + tau / (N * Servo.GEAR_EFF)
    return {"peak torque": float(np.max(np.abs(Tm)) / T_peak),
            "rated torque (rms)": float(np.sqrt(np.mean(Tm ** 2)) / T_rated),
            "speed": float(np.max(np.abs(w)) * N / (rpm_max * 2.0 * pi / 60.0)),
            "inertia ratio": float(J_load / (N * N * J_r) / Servo.INERTIA_RATIO)}


def options(motor, belt, tau, w, wd, J_load):
    """[Drive] of one motor at every ratio."""
    return [Drive(motor, r, r * belt, drive_use(motor, r * belt, tau, w, wd, J_load)) for r in Servo.RATIOS]


def swing_grid(rig):
    """s, the swing's own times on the drive's command cycle."""
    return np.arange(0.0, rig.swing.t_end + 0.5 * Servo.CYCLE_S, Servo.CYCLE_S)


_drives = {}


def choose(rig):
    """(Drive arm, Drive wrist): THE MOTORS."""
    key = rig.d.bat
    if key in _drives and rig is _rigs.get(key):
        return _drives[key]
    q, qd, qdd = swing_state(rig, swing_grid(rig))
    tau = torques(rig, q, qd, qdd)
    J = load_inertia(rig)
    out = []
    for j, belt in ((0, 1.0), (1, SwingRig.BELT_RATIO)):
        pick = None
        for motor in Servo.CATALOGUE:
            ok = [o for o in options(motor, belt, tau[:, j], qd[:, j], qdd[:, j], J[j]) if o.worst()[1] <= 1.0]
            if ok:
                pick = min(ok, key=lambda o: o.worst()[1])
                break
        if pick is None:
            raise ValueError("no motor in Servo.CATALOGUE turns the %s through the rated swing"
                             % ("arm", "wrist")[j])
        out.append(pick)
    out = tuple(out)
    if rig is _rigs.get(key):
        _drives[key] = out
    return out


# ================================================================ THE RUN
def static_torque(rig):
    """(2,) N m: the most gravity asks of each joint anywhere."""
    a = np.linspace(-pi, pi, 73)
    A, B = np.meshgrid(a, a)
    q = np.stack([A.ravel(), B.ravel()], 1)
    return np.max(np.abs(torques(rig, q, np.zeros_like(q), np.zeros_like(q))), 0)


def move_limits(rig, drives):
    """(w (2,) rad/s, a (2,) rad/s^2) for the rig's own moves, park to stance
    and back: each motor at its rated speed and the rated torque left over
    from gravity's most, on the joint's inertia and its rotor's."""
    J = load_inertia(rig)
    ts = static_torque(rig)
    w, a = np.zeros(2), np.zeros(2)
    for j, dr in enumerate(drives):
        _, T_rated, _, rpm, _, J_r = dr.motor
        w[j] = rpm * 2.0 * pi / 60.0 / dr.N
        room = T_rated * dr.N * Servo.GEAR_EFF - ts[j]
        if room <= 0.0:
            raise ValueError("the %s's motor cannot hold the rig against gravity at its rated torque"
                             % ("arm", "wrist")[j])
        a[j] = room / (J[j] + J_r * dr.N ** 2)
    return w, a


class Move:
    """Rest to rest from q0 to q1 (rad), both joints on the quintic
    10 s^3 - 15 s^4 + 6 s^5 (THE TEST), as short as each joint's speed w
    and acceleration a (rad/s, rad/s^2) allow: its peak rate is 15/8 and
    its peak acceleration 10/sqrt(3) of the mean's."""

    V_PEAK = 15.0 / 8.0
    A_PEAK = 10.0 / sqrt(3.0)

    def __init__(self, q0, q1, w, a, T_min=0.0):
        self.q0 = np.asarray(q0, float)
        self.d = np.asarray(q1, float) - self.q0
        D = np.abs(self.d)
        self.T = float(max(np.max(self.V_PEAK * D / np.asarray(w, float)),
                           np.max(np.sqrt(self.A_PEAK * D / np.asarray(a, float))), T_min))

    def path(self, s):
        """(q, q', q''), each (n, 2): the move at fractions s of its time,
        rates per unit of s; at time T, q' / T and q'' / T^2."""
        s = np.clip(np.atleast_1d(np.asarray(s, float)), 0.0, 1.0)
        p = s ** 3 * (10.0 - 15.0 * s + 6.0 * s * s)
        pd = 30.0 * s * s * (1.0 - s) ** 2
        pdd = 60.0 * s * (1.0 - s) * (1.0 - 2.0 * s)
        return self.q0 + np.outer(p, self.d), np.outer(pd, self.d), np.outer(pdd, self.d)

    def state(self, t):
        """(q, qd, qdd), each (n, 2), at t s from the move's start."""
        t = np.atleast_1d(np.asarray(t, float))
        if self.T <= 0.0:
            z = np.zeros((len(t), 2))
            return self.q0 + z, z, z.copy()
        q, q1, q2 = self.path(t / self.T)
        return q, q1 / self.T, q2 / self.T ** 2


S_GRID = 2001                  # samples along a move where its load on the pack is held


def pack_load(rig, q, qd, qdd, pack=None):
    """(n,) m/s^2: the specific force along the bat toward its tip at the
    pack's centre, at (phi, alpha) states.  The pack leaves its + plate
    when this times the pack's mass passes the spring's force."""
    pack = DUMMY if pack is None else pack
    cells, _ = seated(rig.d, pack)
    p = np.array([(cells[0][1] + cells[-1][0]) / 2.0 - rig.d.lay.x_hands, 0.0, -rig.d.core.e]) / 1000.0
    u = np.stack([np.cos(q[:, 0]), np.sin(q[:, 0])], 1)
    axis = np.stack([np.cos(q[:, 1]), np.sin(q[:, 1])], 1)
    r = np.einsum("nij,j->ni", rot3(q[:, 1]), p)[:, [0, 2]]
    a2 = rig.L * (qdd[:, :1] * perp(u) - qd[:, :1] ** 2 * u) + qdd[:, 1:] * perp(r) - qd[:, 1:] ** 2 * r
    return np.einsum("ni,ni->n", a2 - rig.g[[0, 2]], axis)


def swing_load(rig, pack=None):
    """m/s^2: the most pack_load the swing itself asks."""
    tg = swing_grid(rig)
    return float(np.max(pack_load(rig, *swing_state(rig, tg), pack)))


def gentle(rig, q0, q1, w, a, f_max, pack=None):
    """Move from q0 to q1, stretched until its load on the pack nowhere
    passes f_max (m/s^2), or None if the bat at rest somewhere on it would
    already pass it.  The load at time T is the still load plus the
    move's own (at T = 1) over T^2."""
    mv = Move(q0, q1, w, a)
    q, q1_, q2_ = mv.path(np.linspace(0.0, 1.0, S_GRID))
    still = pack_load(rig, q, 0.0 * q, 0.0 * q, pack)
    own = pack_load(rig, q, q1_, q2_, pack) - still
    room = f_max - still
    if np.any(room <= 0.0):
        return None
    m = own > 0.0
    T2 = float(np.max(own[m] / room[m])) if np.any(m) else 0.0
    return Move(q0, q1, w, a, sqrt(T2))


class Motion:
    """The rig's run in (phi, alpha): the move to the stance, the swing, the
    move back.  state(t) -> (q, qd, qdd), each (n, 2)."""

    def __init__(self, rig, park, to, back):
        self.rig, self.park, self.to, self.back = rig, park, to, back
        self.t_swing = to.T
        self.t_back = to.T + rig.swing.t_end
        self.T = self.t_back + back.T

    def state(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        q, qd, qdd = (np.zeros((len(t), 2)) for _ in range(3))
        for lo, hi, fn in ((-np.inf, self.t_swing, lambda s: self.to.state(np.maximum(s, 0.0))),
                           (self.t_swing, self.t_back, lambda s: swing_state(self.rig, s - self.t_swing)),
                           (self.t_back, np.inf, lambda s: self.back.state(np.minimum(s - self.t_back, self.back.T)))):
            m = (t >= lo) & (t < hi)
            if np.any(m):
                q[m], qd[m], qdd[m] = fn(t[m])
        return q, qd, qdd


_motions = {}


def motion(d):
    """Motion of design d's run (THE TEST).  The park pose is the stance's
    arm with the bat level and face down (alpha a whole number of turns);
    each move takes the whole turn, of the two either side, that gets it
    there soonest without loading the pack harder than the swing does."""
    if d.bat in _motions:
        return _motions[d.bat]
    rig = fit(d)
    w, a = move_limits(rig, choose(rig))
    f_max = swing_load(rig)
    q0 = swing_state(rig, np.array([0.0]))[0][0]
    q1 = swing_state(rig, np.array([rig.swing.t_end]))[0][0]

    def level(alpha):
        k = np.floor(alpha / (2.0 * pi))
        return [2.0 * pi * k, 2.0 * pi * (k + 1.0)]

    tos = [(gentle(rig, np.array([q0[0], al]), q0, w, a, f_max), al) for al in level(q0[1])]
    tos = [(mv, al) for mv, al in tos if mv is not None]
    backs = [gentle(rig, q1, np.array([q0[0], al]), w, a, f_max) for al in level(q1[1])]
    backs = [mv for mv in backs if mv is not None]
    if not tos or not backs:
        raise ValueError("no move between park and the swing keeps the pack's load within the swing's")
    to, al0 = min(tos, key=lambda x: x[0].T)
    back = min(backs, key=lambda mv: mv.T)
    mo = Motion(rig, np.array([q0[0], al0]), to, back)
    _motions[d.bat] = mo
    return mo


# ================================================================ THE TEST
@dataclass
class SwingStation:
    clock: H.ClockHAL
    radio: H.RadioHAL          # the station's dongle
    rig: object                # swing.hal.SwingRigHAL


def rails(codes):
    """Whether any code sits at either end of its range (imu/model.sense's
    rail, as a station sees it)."""
    lo, hi = -2 ** (Imu.BITS - 1), 2 ** (Imu.BITS - 1) - 1
    return bool(np.any((codes == lo) | (codes == hi)))


def tail_s(ci):
    """s from a sample to its arrival at the dongle, at the Z tail: the rest
    of its packet, an interval's wait, the host's floor and its exponential
    spread's Z-tail quantile (as phone.test.last_packet_s, for the dongle)."""
    return (H.samples_per_packet() / Imu.ODR + ci + Est.HOST_FLOOR_S["dongle"]
            - log(TAIL) * Est.HOST_SPREAD_S["dongle"])


def swing_test(st, serial, d):
    """procedures.Verdict of the full-speed swing on the bat in the rig (THE
    TEST)."""
    mo = motion(d)
    run = PR._Run(st)
    ci = Link.CI_MIN
    req = PR.timeout(PR.request_s(ci))
    try:
        t0 = st.clock.now()
        ok = st.radio.connect(serial, ci, PR.timeout(Link.ADV_SLOW_S + 2.0 * ci))
        run("connects", ok, "it would not connect", t0)
        t0 = st.clock.now()
        try:
            _, _, boots0, _ = st.radio.info(req)
            st.radio.stream(True, req)
        except H.LinkError as e:
            run("streams", False, "no answer: %s" % e, t0)
        st.rig.run(mo)
        st.clock.wait(mo.T + tail_s(ci))
        raw = st.radio.listen(0.0)
        try:
            st.radio.stream(False, req)
            _, _, boots1, _ = st.radio.info(req)
            reset = boots1 != boots0
        except H.LinkError:
            reset = True
        pk = [H.parse_packet(t, b) for t, b in raw]
        k = np.array([p.k0 for p in pk])
        whole = len(k) > 1 and bool(np.all(np.diff(k) == H.samples_per_packet()))
        run("rides through the swing", not reset and whole,
            "the board reset during the swing: its cells let go for longer than its hold-up" if reset
            else "its stream broke: %d packets" % len(pk), t0, packets=len(pk))
        n = len(pk) * H.samples_per_packet()
        need = int(mo.T * Imu.ODR)
        run("streams the whole run", n >= need, "%d samples of the run's %d" % (n, need), t0)
        codes = np.concatenate([p.codes for p in pk])
        run("never rails", not rails(codes), "%d samples at a rail" % int(np.sum(np.any(
            (codes == -2 ** (Imu.BITS - 1)) | (codes == 2 ** (Imu.BITS - 1) - 1), axis=1))), t0, codes=codes)
        st.radio.disconnect()
        return PR.Verdict(True, "", run.steps, serial)
    except PR._Fail as e:
        try:
            st.radio.disconnect()
        except H.LinkError:
            pass
        return PR.Verdict(False, str(e), run.steps, serial)


def plan(d):
    """{step: planned s} of swing_test on a healthy bat."""
    mo = motion(d)
    ci = Link.CI_MIN
    req = PR.request_s(ci)
    return {"connects": Link.ADV_FAST_S / 2.0 + Link.CI_STEP + ci / 2.0,
            "rides through the swing": 2.0 * req + mo.T + tail_s(ci) + 2.0 * req}


def plan_s(d):
    return sum(plan(d).values())
