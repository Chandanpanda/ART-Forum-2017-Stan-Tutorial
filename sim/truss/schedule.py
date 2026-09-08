"""The cycle as a list of operations with times: load, wind, dose.

The brief's process sequence (4.7) is a paragraph.  This turns it into a
plan the machine executes and the clock can be checked against: every op
has a start time and a duration computed from the axes' own motion
profiles (motion.py), the ring's speed, and the fixture's poses.  Tier 1
runs the same list through the HAL, so "planned 26 minutes, simulated 27"
is a sentence the check suite can utter.

DECISIONS TAKEN HERE, WITH THE REASON:

  * Chords load chord-up, diagonals load face-up.  A chord drops into its
    V-notches only when the notches open upward; a diagonal drops into
    its two cradles only when its face is level.  Six cage indexes.
  * Chord runs alternate direction.  The thread is continuous along a
    chord, so a run may start at either end; starting where the last one
    finished saves a metre of traverse per chord.
  * The dose follows the wind, joint by joint.  The dispenser rides the
    head on its own stroke, so nothing is swapped and no second pass is
    made; a joint is dosed the moment its band is complete, while the
    band is still on top.  `interleave=False` plans the brief's separate
    pass instead, so the two can be compared on the clock.
  * The head cruises at the index height while carrying a rod and at the
    per-joint lift while winding, both computed by approach.py.
"""
from dataclasses import dataclass, field

import numpy as np

from .spec import (Gantry, Ring, Head, Gripper, Cage, Dispenser, Cutter, Vision,
                   Process)
from . import motion, approach as _approach, band as _band
from .geometry import rot_x


@dataclass
class Op:
    kind:  str
    t0:    float
    dt:    float
    phase: str
    args:  dict = field(default_factory=dict)

    @property
    def t1(self):
        return self.t0 + self.dt

    def __repr__(self):
        a = " ".join("%s=%s" % (k, (("%.1f" % v) if isinstance(v, float) else v))
                     for k, v in self.args.items() if k != "pos")
        return "<%7.1f +%5.1f %-6s %-8s %s>" % (self.t0, self.dt, self.phase, self.kind, a)


class Plan:
    def __init__(self):
        self.ops = []
        self.t = 0.0

    def add(self, kind, dt, phase, **args):
        op = Op(kind, self.t, float(dt), phase, args)
        self.ops.append(op)
        self.t += float(dt)
        return op

    @property
    def total(self):
        return self.t

    def by_phase(self):
        out = {}
        for o in self.ops:
            out[o.phase] = out.get(o.phase, 0.0) + o.dt
        return out

    def by_kind(self):
        out = {}
        for o in self.ops:
            out[o.kind] = out.get(o.kind, 0.0) + o.dt
        return out

    def count(self, kind):
        return sum(1 for o in self.ops if o.kind == kind)


# ------------------------------------------------------------ the head
class HeadState:
    """Where the ring centre is, and what the cage is at.  The planner
    moves this; the process generators move the real one the same way."""

    def __init__(self, x, y, z, theta):
        self.pos = {"x": x, "y": y, "z": z}
        self.theta = theta
        self.yaw = 0.0


def grip_z(z_rod):
    """Ring-centre z that puts a fully extended gripper's rod at z_rod."""
    return z_rod - Head.TIP_PARK + Head.GRIP_STROKE


def look_x(t, joint):
    """Ring-centre x for the pre-wind look: the joint half a band along on
    the CAMERA'S side of the ring's plate, whichever way the run goes.
    Measured (check_vision): seen through the plate's gap from the far
    side, one diagonal of two is lost and the joint's x comes back up to
    2 mm off; on the camera's side both are in view and it is within
    LOOK_SIGMA.  On a run toward the camera's side this is the band start
    and costs nothing; on the return run it is a band's width away."""
    side = 1.0 if Vision.cam_pos()[0] >= 0.0 else -1.0
    return float(joint.x) - side * t.band / 2.0


def move_time(a, b):
    return motion.coordinated_time({k: b[k] - a[k] for k in b if k in a},
                                   Gantry.V_MAX, Gantry.A_MAX)


def stroke_time(d):
    return motion.trap_time(d, Gantry.V_MAX["z"], Gantry.A_MAX["z"])


def index_time(d_theta):
    """The worm: THETA_RPM at the cage, the short way round."""
    d = abs((d_theta + 180.0) % 360.0 - 180.0)
    return d / (6.0 * Cage.THETA_RPM) + Gantry.SETTLE_S


# ------------------------------------------------------------- planning
def plan(geom, fixture, stations, interleave=True, start=None):
    """The whole cycle.  `stations` is approach.plan_truss's output."""
    t = geom.t
    P = Plan()
    cruise = _approach.index_lift(geom, fixture)
    hs = HeadState(*(start or (-Cage.POST_OFF, 0.0, cruise)), theta=0.0)
    obstacles = {}

    def obs_at(theta):
        if theta not in obstacles:
            obstacles[theta] = _approach.Obstacles(geom, fixture, theta)
        return obstacles[theta]

    def move(phase, **goal):
        g = dict(hs.pos); g.update(goal)
        dt = move_time(hs.pos, g) + Gantry.SETTLE_S
        hs.pos = g
        return P.add("move", dt, phase, pos=dict(g), **goal)

    def index(phase, theta):
        if hs.pos["z"] < cruise - 1e-6:
            move(phase, z=cruise)
        if abs(((theta - hs.theta) + 180.0) % 360.0 - 180.0) > 1e-6:
            P.add("index", index_time(theta - hs.theta), phase, theta=float(theta))
            hs.theta = float(theta)

    def yaw_to(phase, yaw):
        d = abs(((yaw - hs.yaw) + 180.0) % 360.0 - 180.0)
        if d > 1e-6:
            P.add("yaw", d / Gripper.YAW_V, phase, yaw=float(yaw))
            hs.yaw = float(yaw)

    # ------------------------------------------------------------ load
    rods = list(geom.chords) + list(geom.diags)
    for r in rods:
        theta = fixture.theta_for_loading(r)
        index("load", theta)
        pick, pick_yaw = fixture.pick_pose(r)
        place, yaw, _ = fixture.place_pose(r, theta)
        gx = Head.grip_x()
        # over the slot, at cruise; the gripper's grip point is gx along x
        move("load", x=float(pick[0]) - gx, y=float(pick[1]), z=cruise)
        yaw_to("load", pick_yaw)
        move("load", z=grip_z(float(pick[2])))
        P.add("extend", stroke_time(Head.GRIP_STROKE), "load", tool="grip")
        P.add("grip", Gripper.JAW_CLOSE_S, "load", rod=r.index)
        P.add("retract", stroke_time(Head.GRIP_STROKE), "load", tool="grip")
        move("load", z=cruise)
        move("load", x=float(place[0]) - gx, y=float(place[1]))
        if abs(((yaw - hs.yaw) + 180.0) % 360.0 - 180.0) > 1e-6:
            # THE YAW WAITS FOR THE EXTENSION (approach.yaw_height): a rod
            # turned beside the ring while retracted swings through its
            # rim.  Come down to the height the solver gives, put the
            # gripper out, turn there, then finish the descent.
            half = (t.L_cut if r.kind == "diag" else r.length) / 2.0
            z_yaw = _approach.yaw_height(obs_at(theta), place, hs.yaw, yaw, half, r.r,
                                         Process.SEAT_CLEAR, z_max=cruise,
                                         stroke=Head.GRIP_STROKE)
            if z_yaw is None:
                raise ValueError("rod %d cannot be turned to %.0f over its seat" % (r.index, yaw))
            move("load", z=grip_z(z_yaw))
            P.add("extend", stroke_time(Head.GRIP_STROKE), "load", tool="grip")
            yaw_to("load", yaw)
            move("load", z=grip_z(float(place[2])) + 3.0)
        else:
            move("load", z=grip_z(float(place[2])) + 3.0)
            P.add("extend", stroke_time(Head.GRIP_STROKE), "load", tool="grip")
        # release a millimetre ABOVE the seat and let the V take it: lowered
        # onto the flanks while still gripped, the axis stalls against the
        # fixture and the rod is pressed rather than seated
        move("load", z=grip_z(float(place[2])) + Process.DROP_IN)
        P.add("release", Gripper.JAW_CLOSE_S + 0.4, "load", rod=r.index,
              pose=(float(place[0]), float(place[1]), float(place[2])), yaw=float(yaw))
        P.add("retract", stroke_time(Head.GRIP_STROKE), "load", tool="grip")
        move("load", z=cruise)

    # ------------------------------------------------------------ wind
    def post_loop(phase, post, theta):
        """Hook the strand round a post: a rectangle about it at the
        height and leg approach.post_loop solves for."""
        px = float(post.p0[0])
        zc = float((rot_x(theta) @ geom.chord_point(post.chord, 0.0))[2])
        lp = _approach.post_loop(obs_at(theta), px, zc, Process.SEAT_CLEAR, h_max=cruise - zc)
        if lp is None:
            raise ValueError("no clear loop round post %d/%d" % (post.chord, post.end))
        z, leg, a, _cl = lp
        move(phase, x=px - a, y=0.0, z=z + Process.LIFT_CLEAR)
        for dx, dy in ((0, leg), (2 * a, 0), (0, -leg), (-2 * a, 0)):
            move(phase, x=hs.pos["x"] + dx, y=hs.pos["y"] + dy)
        P.add("anchor", Process.ANCHOR_S, phase, post=(post.chord, post.end))

    direction = +1
    for k in range(t.n_chords):
        theta, aps = stations[k]
        if aps is None:
            raise ValueError("no approach for chord %d" % k)
        index("wind", theta)
        js = geom.joints_on(k)
        order = list(range(len(js))) if direction > 0 else list(range(len(js) - 1, -1, -1))
        first_end = 0 if direction > 0 else 1
        post = next(p for p in fixture.posts if p.chord == k and p.end == first_end)
        post_loop("wind", post, theta)
        for n, i in enumerate(order):
            j, a = js[i], aps[i]
            c = a.centre
            xb = _band.band_start(j.x, t, direction)
            move("wind", x=look_x(t, j), y=float(c[1]), z=float(c[2] + a.lift))
            P.add("look", Process.VISION_SETTLE_S, "wind", joint=j.index)
            if abs(look_x(t, j) - xb) > 1e-9:
                move("wind", x=float(xb))
            P.add("park", 0.3, "wind", gap=float(a.gap_down))
            move("wind", z=float(c[2]))
            wind_s = t.turns / Ring.RPM * 60.0 + Ring.SPINUP_S
            P.add("wind", wind_s, "wind", joint=j.index, turns=t.turns,
                  feed=float(direction * t.band), rpm=Ring.RPM)
            hs.pos["x"] = float(xb + direction * t.band)
            P.add("park", 0.5 + Ring.STOP_TOL / (6.0 * Ring.RPM), "wind", gap=float(a.gap_down))
            move("wind", z=float(c[2] + a.lift))
            P.add("park", 0.3, "wind", gap=0.0)
            if interleave:
                dose(P, hs, move, geom, j, a, "dose")
        last_end = 1 if direction > 0 else 0
        post = next(p for p in fixture.posts if p.chord == k and p.end == last_end)
        post_loop("wind", post, theta)
        P.add("cut", Cutter.CUT_S, "wind", chord=k)
        direction = -direction

    # -------------------------------------------------- the separate pass
    if not interleave:
        for k in range(t.n_chords):
            theta, aps = stations[k]
            index("dose", theta)
            for j, a in zip(geom.joints_on(k), aps):
                dose(P, hs, move, geom, j, a, "dose")
    move("done", z=cruise)
    return P


def dose(P, hs, move, geom, j, a, phase):
    """One drop on one band: nozzle over the joint, extend to standoff,
    dispense, retract.  The ring is at its lift, off to one side."""
    c = a.centre
    # the drop hangs from the tip: two radii of it, then the fall.  (A
    # tip one millimetre over the cluster spawned the drop inside the
    # chord and the solver threw it to the floor: seven of twelve.)
    r_drop = _band.drop_radius_mm(_band.resin_dose_mg(geom.t, geom.thread_on_joint(j)))
    top = float(c[2]) + geom.cluster_reach(j, 0.0) + 2.0 * r_drop + Dispenser.STANDOFF
    move(phase, x=float(j.x) - Head.disp_x(), y=float(c[1]), z=float(c[2] + a.lift))
    ext = (hs.pos["z"] + Head.TIP_PARK) - top
    P.add("extend", stroke_time(ext), phase, tool="disp", mm=float(ext))
    P.add("dose", Dispenser.DOSE_S + Process.DOSE_SETTLE_S, phase, joint=j.index)
    P.add("retract", stroke_time(ext), phase, tool="disp")


def summary(P):
    """Minutes by phase and by kind, and the totals a check pins."""
    ph = {k: v / 60.0 for k, v in P.by_phase().items()}
    kd = {k: v / 60.0 for k, v in P.by_kind().items()}
    return {"total_min": P.total / 60.0, "phase_min": ph, "kind_min": kd,
            "joints": P.count("wind"), "rods": P.count("release"),
            "indexes": P.count("index")}
