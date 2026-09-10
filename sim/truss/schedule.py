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

from math import atan2, degrees

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
    look_sigma.  On a run toward the camera's side this is the band start
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
    # ------------------------------------------------------------ mount
    # THE CAMERA GOES ON IN THE CAGE.  Its rods are laid by the same
    # gripper at cage angles the same worm indexes, and its bonds are laid
    # by the same dispenser -- which is why the mount was solved for poses
    # the machine already has rather than for the tidiest geometry.
    if geom.mount_rods:
        mount_phase(P, hs, move, index, yaw_to, geom, fixture, cruise)
    move("done", z=cruise)
    return P


def theta_up(p):
    """The cage angle that brings a point on the work to the top, degrees.

    The gantry reaches down; the cage is what presents a face to it.  Every
    other phase already knows this -- `theta_for_loading` for a rod, the
    approach solver's station for a joint -- and the mount's fillets are no
    different."""
    return degrees(atan2(float(p[1]), float(p[2]))) % 360.0


def mount_phase(P, hs, move, index, yaw_to, geom, fixture, cruise):
    """Both noses: three battens, the HEAD KIT on its carrier, six struts,
    and a fillet at every rod end.

    THE GRID IS NOT LAID HERE.  Its four crossings are wound to each other
    with the same thread the truss's joints use, and this cell cannot make
    that joint: its ring is 40 mm across in a raceway 54 across, and by the
    time the mount goes on, the crossings are inside the end triangle with
    the camera behind them.  Station B makes the sub-assembly on its own
    jig and it arrives wound, bonded and seated on the module -- so what
    this phase fetches is one part, and the crossings are already there
    when the struts reach for them.

    ORDER IS MECHANICAL, not arbitrary.  The battens close the chord
    triangle first, because everything else is measured from it.  The KIT
    goes on next, because it carries the crossings.  Then the struts, which
    land where two grid rods already cross: a strut laid before its
    crossing exists would be a rod bonded to air.
    """
    from . import mount as _mount
    order = {"mbatten": 0, "mcam": 1, "mstrut": 2}
    gx = Head.grip_x()
    # ...AND HIGH ENOUGH THAT THE ROD, HUNG BELOW THE HEAD, CLEARS THE WORK.
    # The swing costs the whole stroke in height, so the carriage has to be
    # a stroke plus the cage's own radius above the axis to make room for it.
    swing_z = max(cruise,
                  geom.R + geom.t.d_chord / 2.0 + Process.SEAT_CLEAR
                  + Head.yaw_stroke(geom.t.d_chord / 2.0) - Head.TIP_PARK)
    for end in (0, 1):
        rods = sorted([r for r in geom.mount_rods if r.chord == end],
                      key=lambda r: (order[r.kind], r.index))
        for r in rods:
            # EITHER CAGE ANGLE LAYS THE ROD; only one of them puts it
            # where the gantry can reach.  rot_x(theta+180) with the yaw
            # negated is the same line in the cage's frame, but the rod's
            # POSITION flips with it, and the three end battens came out
            # 4.7 mm below the z axis's own floor at the first solution.
            # Take the one that holds the rod higher.
            # ...AND ONLY THE ONES THE YAW SERVO CAN ACTUALLY REACH.  It has
            # a stop at +-GRIP_YAW; asked for more it clamps, lays the rod
            # off its own line, and answers "settled" -- which is how four
            # struts a truss came out 27.7 mm out at their ends while every
            # op reported success.  A pose outside the stop is not a pose.
            # THE CAMERA IS NOT REVERSIBLE.  A rod laid end-for-end is the
            # same rod; a camera laid end-for-end is a boss where the lens
            # goes.  Only the cage's own symmetry is on offer for it.
            offer = _mount.poses_for(r.axis, reversible=(r.kind != "mcam"))
            poses = [p for p in offer if abs(p[1]) <= Head.GRIP_YAW]
            if not poses:
                raise ValueError(
                    "mount rod %d wants a gripper yaw outside +-%.0f deg: %s"
                    % (r.index, Head.GRIP_YAW, [round(p[1], 1) for p in offer]))
            theta, yaw = max(poses, key=lambda p: float((rot_x(p[0]) @ r.mid)[2]))
            index("mount", theta)
            s = fixture.slot_of(r.index)
            pick = (s.p0 + s.p1) / 2.0
            place = rot_x(theta) @ r.mid
            move("mount", x=float(pick[0]) - gx, y=float(pick[1]), z=cruise)
            yaw_to("mount", 90.0)
            move("mount", z=grip_z(float(pick[2])))
            P.add("extend", stroke_time(Head.GRIP_STROKE), "mount", tool="grip")
            P.add("grip", Gripper.JAW_CLOSE_S, "mount", rod=r.index)
            P.add("retract", stroke_time(Head.GRIP_STROKE), "mount", tool="grip")
            move("mount", z=cruise)
            move("mount", x=float(place[0]) - gx, y=float(place[1]))
            # TURN IT BELOW THE HEAD, NOT INSIDE IT.  Retracted, the grip
            # point sits TIP_PARK above the ring's centre -- inside its bore
            # -- and a rod swung about that point sweeps its own half-length
            # out of the bore and into the annulus.  Measured on rig_mount:
            # a 39.7 mm strut turned to -33.7 degrees sat 0.67 mm inside
            # `ring7`, and went down 71.6 degrees off its own line with the
            # yaw servo reporting the commanded angle and every op reporting
            # success.  Extended past the head's own rim it turns in clear
            # air at any yaw, which is why this is a stroke and not a filter
            # on the poses: filtered, two struts an end have no pose left.
            sw = Head.yaw_stroke(r.r)
            move("mount", z=float(swing_z))
            P.add("extend", stroke_time(sw), "mount", tool="grip", mm=float(sw))
            yaw_to("mount", yaw)
            # ...AND IT DOES NOT COME BACK UP.  Turned below the head, the
            # rod is at a yaw that does NOT clear the head retracted -- so
            # retracting after the turn drags it straight back through the
            # annulus.  Measured: `rod13_g` against `ring16`, and four
            # struts a truss laid 6 degrees off their own line by it.  The
            # carriage descends with the rod still hung below it and the
            # stroke finishes the last few millimetres.
            # DOWN TO WHERE THE ROD GOES, not to a hover above it.  A truss
            # rod is released a millimetre above a V and the V takes it;
            # THE MOUNT HAS NO V's, and the keeper welds each rod where the
            # tool left it -- so a millimetre of drop-in is a millimetre of
            # permanent error on every part of the nose, and it was the
            # floor under every reading in rig_mount.
            move("mount", z=grip_z(float(place[2])))
            P.add("extend", stroke_time(Head.GRIP_STROKE - sw), "mount",
                  tool="grip")
            # TACKED, NOT DROPPED.  A truss rod is released a millimetre
            # above a V and the V takes it.  THE MOUNT HAS NO V's -- its
            # rods are laid onto the collar and onto each other -- so the
            # keeper has to take each one while the jaws are still shut.
            # On the machine that is a tack of adhesive before the gripper
            # lets go; released first and welded after, every mount rod
            # fell on the floor (measured, in the assembly run).
            P.add("release", Gripper.JAW_CLOSE_S + 0.4, "mount", rod=r.index,
                  pose=tuple(float(v) for v in place), yaw=float(yaw),
                  tack=True)
            P.add("retract", stroke_time(Head.GRIP_STROKE), "mount", tool="grip")
            move("mount", z=cruise)
        # ---- the bonds.  EVERY MOUNT JOINT IS AN EPOXY FILLET: the ring
        # cannot reach past the last truss joint, and does not need to --
        # a fillet on a 1.5 mm rod carries a kilonewton against a service
        # load under half a newton, and comes out stiffer than the strut.
        # EVERY BOND POINT COMES UP TO THE NOZZLE.  The mount's rods sit all
        # round the cage; the dispenser only reaches over the top, and the
        # first version sent it to cage coordinates at theta zero -- which
        # put it under the work half the time, stalled the z axis against
        # the fixture, and shook the truss off its own V's.  The cage turns
        # for a fillet exactly as it turns for a rod.  Sorted by angle, so
        # the worm indexes once per group rather than once per fillet.
        pts = []
        for r in rods:
            if r.kind == "mcam":
                continue
            for q in (r.p0, r.p1):
                q = np.asarray(q, float)
                pts.append((theta_up(q), r.index, q))
        for th, ri, q in sorted(pts, key=lambda t: (round(t[0], 3), t[1])):
            index("bond", th)
            w = rot_x(th) @ q                    # where it is once turned up
            move("bond", x=float(w[0]) - Head.disp_x(), y=float(w[1]),
                 z=float(w[2]) + Head.DISP_STROKE + Process.LIFT_CLEAR)
            P.add("extend", stroke_time(Head.DISP_STROKE), "bond", tool="disp",
                  mm=float(Head.DISP_STROKE))
            P.add("bond", Dispenser.DOSE_S + Process.DOSE_SETTLE_S, "bond",
                  rod=ri, at=tuple(float(v) for v in w))
            P.add("retract", stroke_time(Head.DISP_STROKE), "bond", tool="disp")
        move("bond", z=cruise)


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
