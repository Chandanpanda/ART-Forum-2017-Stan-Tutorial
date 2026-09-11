"""Station B: the sub-assembly that makes the camera's head kit.

THE PROBLEM IT EXISTS FOR.  The mount's four tic-tac-toe rods are WOUND to
each other at four crossings -- the same thread the truss's joints use, so
the joint is a joint and not a dab of resin on a tangent.  The main cell
cannot make those joints.  Its ring is 40 mm across in a raceway 54 mm
across, and by the time the mount goes on, the crossings are inside the end
triangle with the camera behind them.  There is no approach, and no
clearance argument produces one.

So the crossings are wound before the camera ever reaches the truss, on a
station whose only job is that one part:

    the module goes into a nest, lens UP, by hand
    B lays the COLLAR over the lens housing and bonds its four walls
    B lays the four GRID rods ON A JIG, in clear air, two layers
    B WINDS each of the four crossings and doses them
    B LIFTS the finished frame off the jig and sets it on the collar
    B beads the two bands where the frame lands on the plate
    the finished kit comes out by hand

WHY THE FRAME IS NOT BUILT ON THE CAMERA.  It was, and it could not be
wound: a ring centred on a crossing reaches eleven millimetres below it,
and four of those millimetres down it is already ten millimetres in --
over a plate thirty across, on a board twenty-five across, in a nest.
Measured on a rig, the ring touched the board 78,000 times and turned
0.02 of a turn.  Nothing orbits a joint that has a plate under it, so the
joints are made before the plate is under them.

WHAT MAKES IT THE RIGHT PART.  Every dimension here comes from `Bracket`
and `Payload` -- the same numbers `mount.solve` builds the nose from -- so
the kit B makes is the kit A expects to pick up.  `check_stationb` asserts
that, crossing by crossing, against `mount.solve`'s own landings.

THE FRAME.  B works in its own coordinates and the module lies flat in it:

    B's x  <-  the camera's x (which becomes the truss's spine axis)
    B's y  <-  the camera's `up`
    B's z  <-  the camera's `look` -- so the lens points UP at the head

That is the one orientation in which a three-axis gantry can reach all four
crossings and lay all four rods without turning the part over.

WHY THE WINDER NEEDS NO ROTARY.  A hoop laid in the plane perpendicular to
one rod of a crossing encircles BOTH of them, because the other rod passes
through that plane at the crossing.  So one ring axis -- along the lower
layer -- serves all four crossings, and B has no fixture axis at all.
"""
from dataclasses import dataclass
from math import sqrt

import numpy as np

from .spec import (Bracket, Payload, StationB, Stock, Gripper, Head, Process,
                   Dispenser)


@dataclass(frozen=True)
class Rod:
    """One rod of the kit, in B's frame (mm)."""
    kind:  str            # "collar" | "grid"
    index: int
    p0:    np.ndarray
    p1:    np.ndarray
    r:     float
    layer: int = 0

    @property
    def axis(self):
        v = self.p1 - self.p0
        return v / np.linalg.norm(v)

    @property
    def length(self):
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def mid(self):
        return (self.p0 + self.p1) / 2.0


class Kit:
    """The head kit: what B makes and A picks up.

    Built in B's frame with the module's centre at the origin, so the
    module's board lies in z = 0 and the lens looks up +z.
    """

    def __init__(self, payload=None, d_rod=1.5, clear=None):
        self.p = Payload if payload is None else payload
        self.d = float(d_rod)
        self.clear = clear
        self.gx, self.gu = Bracket.grid_half(self.p, self.d, clear)
        self.px, self.pu = Bracket.plate_half(self.p, self.d, clear)
        self.l0 = Bracket.layer_l(0, self.p, self.d)
        self.l1 = Bracket.layer_l(1, self.p, self.d)
        self.seat = Bracket.seat_l(self.p)

    # ------------------------------------------------------------ pieces
    def collar_z(self):
        """(lo, hi) the collar plate occupies in B's z, mm."""
        return self.seat

    @property
    def rods(self):
        """The four grid rods, in the order they are laid.

        LAYER 0 FIRST, and not by preference: layer 1 lies ON layer 0 at
        every crossing.  Laid the other way round the second pair has
        nothing under it but the collar's own aperture."""
        over = Bracket.overrun(self.d)
        out = []
        for su in (+1.0, -1.0):                      # layer 0, along B's x
            out.append(Rod("grid", len(out),
                           np.array([-self.gx - over, su * self.gu, self.l0]),
                           np.array([+self.gx + over, su * self.gu, self.l0]),
                           self.d / 2.0, 0))
        for sx in (+1.0, -1.0):                      # layer 1, along B's y
            out.append(Rod("grid", len(out),
                           np.array([sx * self.gx, -self.gu - over, self.l1]),
                           np.array([sx * self.gx, +self.gu + over, self.l1]),
                           self.d / 2.0, 1))
        return tuple(out)

    def crossings(self):
        """The four wound joints, in B's frame.  Same points `mount.solve`
        lands its struts on -- that is the whole contract between the two
        machines."""
        lx = 0.5 * (self.l0 + self.l1)
        return tuple(np.array([sx * self.gx, su * self.gu, lx])
                     for sx in (+1.0, -1.0) for su in (+1.0, -1.0))

    def wind_axis(self, i=0):
        """Which way the winder's bore points at a crossing: along the
        crossing's own DIAGONAL.

        NOT along either rod, and that took a frame to see.  A ring can
        only orbit something that lies along its axis: put the bore on the
        lower rod and the UPPER one lies in the ring's plane, so the
        annulus sweeps straight into it and the drive stalls -- measured,
        one crossing of four wound and the rest at zero turns.

        On the diagonal both rods are at 45 degrees to the plane.  Each
        pierces it once, at the crossing, inside the bore; nine millimetres
        out along a rod it is already seven millimetres clear of a ring
        1.5 mm thick.  And that is the plane the lashing wants anyway: over
        the top rod in one quadrant, under the bottom one, up through the
        opposite quadrant -- the loop that ties the two together rather
        than gripping one of them.
        """
        return np.array([1.0, -1.0, 0.0]) / sqrt(2.0)

    def wind_yaw(self):
        """The ring's own yaw, degrees, so its bore lies on that diagonal."""
        return -45.0

    def rod_off_plane(self, s=None):
        """How far off the ring's mid-plane a rod is, `s` mm out from the
        crossing.  The margin the diagonal buys."""
        s = self.gx if s is None else s
        return float(abs(s) / sqrt(2.0))

    def entry_half_angle(self):
        """Half the angle, about the winder's own centre, that the crossing's
        two rods occupy as the ring comes down on them -- what the mouth has
        to be turned to before it descends.

        A rod is inside the ring's slab only while it is within half a ring
        width of its plane, which on the diagonal is a couple of millimetres
        either side of the crossing; at the bore's radius that subtends this
        angle from straight down."""
        n = self.wind_axis()
        t = np.array([-n[1], n[0], 0.0])
        u = self.rods[0].axis
        half = StationB.RING_W / 2.0 + self.d / 2.0
        t_max = half * abs(float(u @ t)) / abs(float(u @ n))
        return float(np.degrees(np.arcsin(min(1.0, t_max / StationB.ring_r_in()))))

    def park_tol(self):
        """How far off its home the mouth may be and still swallow the
        crossing, degrees.  Not a tolerance I chose: the mouth's half-angle
        less the angle the rods need."""
        return StationB.RING_GAP / 2.0 - self.entry_half_angle()

    def hoop_perimeter(self):
        """One turn of thread round a crossing, mm.

        Two rods at right angles, one lying on the other: the hoop is the
        convex hull of the pair in the plane it is laid in -- two half-circles
        of the rod, joined across the stack."""
        return float(np.pi * self.d + 2.0 * self.d)

    def thread_len(self, turns=None):
        n = StationB.TURNS if turns is None else turns
        return 4.0 * n * self.hoop_perimeter()

    # --------------------------------------------------------- the jig
    #
    # THE FRAME IS BUILT AND WOUND OFF THE CAMERA, and that is forced, not
    # chosen.  The crossings sit 4 mm above the collar, and the collar sits
    # on a board 25 x 24 mm.  A ring centred on a crossing reaches 11 mm
    # below it -- through the board and into the nest.  Measured on a rig:
    # turning there, the ring touched `b_board` and two nest walls 78,000
    # times and moved 0.02 of a turn.  Standing the collar on a pedestal
    # does not help: the ring's rim, four millimetres down, is 10.4 mm in
    # along the diagonal, which is still over a plate 30 mm across.
    #
    # No orbiting tool fits over a plate.  So the four rods are laid on a
    # JIG in clear air, the four crossings are wound there, and the
    # finished frame -- one rigid part now -- is lifted onto the collar as
    # a whole.  Which is what the machine was asked for: a station that
    # threads the tic-tac-toe, and a collar that then goes into the camera.
    def wind_solid_clear(self, p, z_top, q):
        """How far a vertical post at (x, y) `p` rising to `z_top` stays
        out of the winder's swept solid at crossing `q`, mm.  Positive is
        clear.

        THE RING IS A THIN ANNULUS AND THERE ARE TWO WAYS TO BE CLEAR OF
        IT.  A post can stand outside its RIM -- more than r_out away in
        the ring's own plane -- or outside its PLANE, more than half its
        width off the axis.  A post only needs one of them, and on this kit
        it is always the second: the bore lies on the crossing's diagonal,
        so a post under a rod is 0.707 mm off the plane for every
        millimetre it stands back along that rod.  Take the better of the
        two and a jig that a radius test rejects fits with 3 mm to spare.
        """
        n = self.wind_axis()
        t = np.array([-n[1], n[0], 0.0])          # in-plane and horizontal
        v = np.array([float(p[0]) - float(q[0]),
                      float(p[1]) - float(q[1]), 0.0])
        axial = abs(float(v @ n)) - StationB.RING_W / 2.0
        dz = max(0.0, float(q[2]) - float(z_top))
        radial = sqrt(float(v @ t) ** 2 + dz * dz) - StationB.ring_r_out()
        return max(axial, radial)

    def work_above_crossing(self):
        """How far the tallest part of the kit stands above the plane the
        winder's centre sits in, mm.  What the carriage's other tools have
        to be seated clear of."""
        lx = 0.5 * (self.l0 + self.l1)
        return max(float(r.p0[2]) + r.r for r in self.rods) - lx

    def post_offset(self):
        """How far out along a layer-0 rod the jig's posts stand, mm.

        AS FAR AS THE WINDER ALLOWS.  The posts want to be far apart -- they
        are the frame's whole support base while it is being wound -- and
        the only thing pushing them in is the ring, which has to orbit the
        crossing at the rod's end.  A post is clear of the ring once it is
        RING_W/2 off the ring's plane, and standing back `a` from the
        crossing along the rod puts it (u . t) * a off that plane.  So the
        bound is the clearance divided by that cosine, and there is no
        choosing left to do."""
        n = self.wind_axis()
        t = np.array([-n[1], n[0], 0.0])
        u = self.rods[0].axis
        need = (StationB.RING_W / 2.0 + StationB.POST_D / 2.0
                + Process.SEAT_CLEAR)
        return self.gx - need / abs(float(u @ t))

    def jig_z(self):
        """Where layer 0 lies above the jig's base plate, mm.

        The winder reaches r_out below the crossing it is turning on, and
        the plate is what it would hit."""
        lx = 0.5 * (self.l0 + self.l1)
        return (StationB.JIG_BASE_T + StationB.ring_r_out()
                + Process.SEAT_CLEAR - (lx - self.l0))

    def jig_y(self):
        """Where the jig sits along B's y, mm: beside the nest, far enough
        that the winder working the near pair of crossings is clear of the
        nest's own walls."""
        n = self.wind_axis()
        t = np.array([-n[1], n[0], 0.0])
        reach = abs(float(t[1])) * StationB.ring_r_out() \
            + abs(float(n[1])) * StationB.RING_W / 2.0
        near = max(self.gu + reach,                       # the ring
                   self.gu + Bracket.overrun(self.d),     # the rods
                   self.gu + StationB.POST_D)             # the base plate
        return (Payload.BOX[2] / 2.0 + StationB.NEST_WALL
                + Process.SEAT_CLEAR + near)

    def jig_origin(self):
        """Where the kit's own origin lands on the jig, in B's frame."""
        return np.array([0.0, self.jig_y(), self.jig_z() - self.l0])

    def jig_rods(self):
        """The four rods where the JIG holds them -- the kit's own geometry,
        translated."""
        o = self.jig_origin()
        return tuple(Rod(r.kind, r.index, r.p0 + o, r.p1 + o, r.r, r.layer)
                     for r in self.rods)

    def jig_crossings(self):
        o = self.jig_origin()
        return tuple(q + o for q in self.crossings())

    def jig_posts(self):
        """(x, y, z_top) of the four posts.

        TWO UNDER EACH LAYER-0 ROD and none under layer 1, because layer 1
        rests on layer 0 at all four crossings exactly as it does in the
        finished mount -- so the jig holds the part the way the part holds
        itself, and the two rods that have nothing over them are left free
        for the jaws that lift the frame off.

        Four posts on a rectangle, not two on a line: two posts under the
        middles of the layer-0 rods would put the frame's own centre of
        mass ON the line joining them, and it would fall off the first time
        the ring touched it.

        The z is the ROD'S OWN CENTRELINE at that station -- the point the
        post has to carry.  How the post ends (a V, and how deep) is the
        rack's business and lives in the MJCF, not here."""
        o = self.jig_origin()
        a = self.post_offset()
        z = self.l0 + float(o[2])
        return tuple((float(sx * a), float(su * self.gu + o[1]), float(z))
                     for sx in (+1.0, -1.0) for su in (+1.0, -1.0))

    def jig_base(self):
        """(centre, half-extents) of the jig's base plate, mm."""
        o = self.jig_origin()
        hx = self.post_offset() + StationB.POST_D
        hy = self.gu + StationB.POST_D
        t = StationB.JIG_BASE_T
        return (np.array([0.0, float(o[1]), t / 2.0]),
                np.array([hx, hy, t / 2.0]))

    def post_clearance(self):
        """Least clearance from any jig post to the winder's swept solid at
        any crossing, mm.  What check_stationb holds the jig against."""
        best = 1e9
        for (x, y, z) in self.jig_posts():
            for q in self.jig_crossings():
                best = min(best, self.wind_solid_clear((x, y), z, q))
        return best

    def jig_floor_clear(self):
        """How far the winder's lowest reach stays above the jig's base
        plate, mm."""
        q = self.jig_crossings()[0]
        return (float(q[2]) - StationB.ring_r_out()) - StationB.JIG_BASE_T

    # ------------------------------------------------------ the transfer
    def frame_rod(self):
        """Which rod the jaws take the finished frame by.

        IT HAS TO BE A LAYER-1 ROD, and that is geometry, not preference.
        The jaws reach PAD_UNDER below the rod they hold; on a layer-0 rod
        that is 2.5 mm above the board, which is INSIDE the collar the
        frame is being set down on.  On layer 1 it is 3.5, and the collar's
        face is at 3.0.  Half a millimetre, and it is the difference
        between placing the frame and shovelling the collar off the
        camera.

        And it is taken at its MIDDLE, which is the one span on the whole
        frame with nothing under it and nothing over it -- the jig's posts
        are under layer 0, and layer 1's own crossings are 8.9 mm away on
        either side."""
        return self.rods[2]

    def frame_grip(self):
        """(point, yaw) the jaws take the finished frame at, in B's frame
        with the frame still on the jig."""
        r = self.frame_rod()
        o = self.jig_origin()
        yaw = 0.0 if abs(float(r.axis[0])) > 0.5 else 90.0
        return (r.mid + o, yaw)

    def frame_grip_margin(self):
        """(under the pads, along the rod) mm of clearance the transfer
        grip has: how far the pads' underside stays above the collar's own
        face when the frame is set down, and how far the pads stay off the
        nearest crossing."""
        r = self.frame_rod()
        under = (float(r.p0[2]) - r.r - Gripper.PAD_UNDER) - self.seat[1] + r.r
        along = min(abs(float(np.linalg.norm(q[:2] - r.mid[:2])))
                    for q in self.crossings()) - Gripper.PAD_L / 2.0
        return (under, along)

    # ------------------------------------------------------- the bonds
    def obstacles(self, where="kit"):
        """Everything standing on the table around a dose point, as
        (x0, x1, y0, y1, top) in B's frame.

        A DOSE POINT IS A CORNER AND A CORNER HAS A WALL.  The nozzle goes
        STANDOFF above the work, but "the work" at the collar's aperture is
        a plate at 3.0 with a lens housing beside it standing to 4.5, and
        at a wound crossing it is the lower rod with the upper one on top.
        A height taken off the floor of the corner is a height inside the
        wall of it -- measured, the nozzle drove into the housing four
        times in one run."""
        from .mount import payload_solid
        out = []
        if where == "kit":
            for e in payload_solid(self.p):
                out.append((e[0][0], e[0][1], e[2][0], e[2][1], e[1][1]))
            out.append((-self.px, self.px, -self.pu, self.pu, self.seat[1]))
            rods = self.rods
        else:
            rods = self.jig_rods()
            for (jx, jy, jz) in self.jig_posts():
                h = StationB.POST_D / 2.0
                out.append((jx - h, jx + h, jy - h, jy + h, jz))
        for r in rods:
            lo, hi_ = np.minimum(r.p0, r.p1), np.maximum(r.p0, r.p1)
            out.append((lo[0] - r.r, hi_[0] + r.r, lo[1] - r.r, hi_[1] + r.r,
                        float(hi_[2]) + r.r))
        return tuple(out)

    def nozzle_z(self, q, where="kit"):
        """Where the nozzle's tip sits to dose the point `q`, mm."""
        clear = Dispenser.NOZZLE_D / 2.0 + Dispenser.DROP_SPREAD
        top = float(q[2])
        for (x0, x1, y0, y1, t) in self.obstacles(where):
            if (x0 - clear <= q[0] <= x1 + clear
                    and y0 - clear <= q[1] <= y1 + clear):
                top = max(top, t)
        return top + Dispenser.STANDOFF

    def aperture_bonds(self):
        """Where the collar is dosed onto the lens housing: one bead on
        each of the aperture's four walls."""
        apx, apu = Bracket.aperture(self.p)
        z = self.seat[1]
        return tuple(np.array([dx, dy, z]) for dx, dy in
                     ((apx / 2.0, 0.0), (-apx / 2.0, 0.0),
                      (0.0, apu / 2.0), (0.0, -apu / 2.0)))

    def collar_beads(self):
        """The two lines of adhesive that hold the frame to the collar.

        LAYER 0 LIES ON THE COLLAR'S TWO LONG BANDS -- that is what the
        plate's outline is for, and `mount.solve` says so in as many words:
        a line of adhesive, not a dab.  Layer 1 never touches the plate; it
        is held by the four wound crossings.  So there are two beads, and
        each runs the length of the band under its rod."""
        bx = self.px - Bracket.band_w(self.d) / 2.0
        out = []
        for r in self.rods:
            if r.layer != 0:
                continue
            a = np.array([-bx, float(r.p0[1]), self.seat[1]])
            b = np.array([+bx, float(r.p0[1]), self.seat[1]])
            out.append((a, b))
        return tuple(out)

    def bead_doses(self, a, b):
        """How many metered doses a bead of adhesive takes.

        ONE PER BAND WIDTH.  The bond land is the band the rod lies on, so
        a bead laid at the band's own pitch runs together into one fillet
        instead of a row of dabs."""
        L = float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
        return max(1, int(np.ceil(L / Bracket.band_w(self.d))))

    # ---------------------------------------------------- the traverse
    def table_top(self):
        """The tallest thing on B's table, mm -- what the carriage has to
        cross over to get from one part of its job to another."""
        tops = [self.seat[1], self.l1 + self.d / 2.0,     # the kit on the camera
                Payload.BOX[1] / 2.0,                     # the module
                StationB.JIG_BASE_T,
                self.jig_z() + (self.l1 - self.l0) + self.d / 2.0]
        for _k, _i, a, _b in self.rack_slots():
            tops.append(float(a[2]) + self.d / 2.0)
        return max(tops)

    def cruise_z(self):
        """The height the CARRIAGE crosses the table at, mm.

        THE RING IS THE LOWEST THING ON THE CARRIAGE AND IT DOES NOT
        RETRACT.  It hangs r_out below the carriage's own centre wherever
        the carriage goes, so the cruise height is set by the winder and
        never by whichever tool happens to be working.  Flown at the
        working tool's height instead, the ring is eleven millimetres deep
        in the jig."""
        return self.table_top() + StationB.ring_r_out() + StationB.LIFT_CLEAR

    def lowest_tip(self):
        """The lowest a tool tip has to reach, mm.  The collar's face where
        it waits on its shelf, as it happens."""
        return min(self.seat[0] + Bracket.SHEET / 2.0,
                   self.l0 - self.d / 2.0, self.seat[1])

    # ------------------------------------------------------- the racks
    def rack_pitch(self):
        """How far apart the waiting rods lie, mm.

        WHAT SETS IT IS THE GRIPPER, not the rods: the jaws have to come
        down onto one rod without their body fouling the next.  So the
        pitch is the jaw body's own width plus the process clearance -- 11
        and 12 were numbers I typed."""
        return Gripper.BODY_W + Process.SEAT_CLEAR

    def rack_slots(self):
        """Where the four rods and the collar wait, in B's frame.

        Beside the nest on the -y side, laid along B's x so the gripper
        takes them at yaw zero and never has to turn a rod it is carrying
        over the part.  The jig is on the +y side and the nest between
        them, so nothing the gripper carries crosses anything it has
        already built."""
        out = []
        pitch = self.rack_pitch()
        y0 = -(Payload.BOX[2] / 2.0 + StationB.NEST_WALL + Process.SEAT_CLEAR
               + pitch / 2.0)
        for i, r in enumerate(self.rods):
            y = y0 - i * pitch
            hl = r.length / 2.0
            out.append(("grid", i, np.array([-hl, y, self.l0]),
                        np.array([+hl, y, self.l0])))
        y = y0 - len(self.rods) * pitch - self.pu
        out.append(("collar", 0, np.array([-self.px, y, self.seat[0]]),
                    np.array([+self.px, y, self.seat[0]])))
        return out

    def extent(self):
        """(x, y, z) half-extents of everything B holds, mm -- what its
        gantry has to cover.  The jig is part of that: it is the tallest
        thing on the table and the far end of the y travel."""
        xs, ys = [self.px], [self.pu]
        zs = [self.seat[1], self.l1 + self.d,
              self.jig_z() + (self.l1 - self.l0) + self.d]
        for r in self.rods:
            xs += [abs(r.p0[0]), abs(r.p1[0])]
            ys += [abs(r.p0[1]), abs(r.p1[1])]
        for r in self.jig_rods():
            xs += [abs(r.p0[0]), abs(r.p1[0])]
            ys += [abs(r.p0[1]), abs(r.p1[1])]
        for _k, _i, a, b in self.rack_slots():
            xs += [abs(a[0]), abs(b[0])]
            ys += [abs(a[1]), abs(b[1])]
        return (max(xs), max(ys), max(zs))


# ======================================================== the machine
import mujoco                                          # noqa: E402

from . import motion                                   # noqa: E402
from .schedule import Plan                             # noqa: E402


AXES = ("x", "y", "z", "g", "d", "v", "w")
_JOINT = {"x": "bx", "y": "by", "z": "bz", "g": "bg", "d": "bd", "v": "bv",
          "w": "bw"}
_ACT = {"x": "ba_x", "y": "ba_y", "z": "ba_z", "g": "ba_g", "d": "ba_d",
        "v": "ba_v", "w": "ba_w"}


class CellB:
    """Station B's HAL, bound to B's own joints in the shared model.

    The same shape as `cell.CellSim` and deliberately no more: B has three
    slides, a gripper on a stroke and a yaw, a vacuum tip on a stroke, a
    dispenser on a stroke and a winder on a hinge.  It reads its axes from
    the model and never from the parts, exactly as the cell's does -- a HAL
    that can see the truth is a HAL that will be asked for it.
    """

    def __init__(self, model, data, kit, origin=None, rng=None):
        from .spec import StationB
        self.m, self.d, self.kit = model, data, kit
        self.rng = rng or np.random.default_rng(0)
        self.o = np.asarray(StationB.ORIGIN if origin is None else origin, float)
        gid = lambda t, n: mujoco.mj_name2id(model, t, n)
        J, A, B, S, E = (mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR,
                         mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_SITE,
                         mujoco.mjtObj.mjOBJ_EQUALITY)
        self.j = {a: gid(J, _JOINT[a]) for a in AXES}
        self.a = {a: gid(A, _ACT[a]) for a in AXES}
        self.a_f, self.a_ring = gid(A, "ba_f"), gid(A, "ba_ring")
        self.j_ring = gid(J, "bring")
        self.b_gw, self.b_vac = gid(B, "bgw"), gid(B, "bvac")
        self.b_mod = gid(B, "b_module")
        self.s_grip = gid(S, "b_grip_pt")
        self.s_cup, self.s_noz = gid(S, "b_cup_tip"), gid(S, "b_nozzle_tip")
        self.names = ["bcollar"] + ["bgrid%d" % i for i in range(len(kit.rods))]
        self.b_part = {n: gid(B, n) for n in self.names}
        self.eq_keep = {n: gid(E, "bkeep_%s" % n) for n in self.names}
        self.eq_hold = {n: gid(E, "bhold_%s" % n) for n in self.names}
        self.eq_vac = {n: gid(E, "bvac_%s" % n) for n in self.names}
        self.eq_jig = {n: gid(E, "bjig_%s" % n) for n in self.names}
        self.eq_frm = [gid(E, "bfrm%d" % i) for i in range(1, len(kit.rods))]
        self.tz = StationB.tool_lift(kit)
        # THE AXES READ IN B'S OWN FRAME, not the world's.  The bodies are
        # built at B's origin, so the joint's zero is already there; adding
        # the world offset back put every commanded y half a metre outside
        # a 200 mm axis, the servo clamped, and nothing ever settled -- 41
        # timeouts in one run, and four rods that never left the rack.
        # WHERE EACH AXIS READS ZERO, in B's own frame.  Not the axis body's
        # own `body_pos` -- these are nested, and the carriage's build
        # HEIGHT sits on `bx`, the outermost, while `bz` is at zero
        # relative to its parent.  Read off `bz` alone the z axis was 38 mm
        # out and every descent stopped short: the tools reached for the
        # rods and closed on air, and the frames showed four rods that
        # never left the rack.  Walk the chain.
        self._off = {}
        for k, (a, b) in enumerate((("x", "bx"), ("y", "by"), ("z", "bz"))):
            i, acc = gid(B, b), 0.0
            while i > 0:
                acc += float(model.body_pos[i][k])
                i = int(model.body_parentid[i])
            self._off[a] = acc * 1000.0 - float(self.o[k])
        for a in ("g", "d", "v", "w"):
            self._off[a] = 0.0
        self._set = {a: self._read(a) for a in AXES}
        for a in AXES:
            self.goto(a, self._set[a])
        self._closing = False
        self._held = None
        self._grasp = ()
        self._vac = False
        self._vac_held = None
        self._ring_w = 0.0
        self.wound = {}                 # crossing -> turns laid
        self.bonds = []

    # ------------------------------------------------------------ axes
    def _read(self, a):
        adr = self.m.jnt_qposadr[self.j[a]]
        v = float(self.d.qpos[adr]) * (1.0 if a == "w" else 1000.0)
        if a == "w":
            v = float(np.degrees(self.d.qpos[adr]))
        return v + self._off.get(a, 0.0)

    def goto(self, axis, value):
        self._set[axis] = float(value)
        v = float(value) - self._off.get(axis, 0.0)
        self.d.ctrl[self.a[axis]] = np.radians(v) if axis == "w" else v / 1000.0

    def at(self, axis):
        return self._read(axis)

    def settled(self, axis, tol=0.05):
        return abs(self.at(axis) - self._set[axis]) <= tol

    def rate(self, axis):
        """How fast an axis is actually moving, mm/s (deg/s for the yaw).

        A TOOL THAT LANDS ON THE WORK NEVER REACHES ITS COMMANDED
        POSITION, and it should not: the cup is supposed to touch the
        collar and the jaws are supposed to close on a rod.  Position
        error alone cannot tell that from a servo that has failed, so the
        stroke asks this as well -- at the target, or stopped."""
        adr = self.m.jnt_dofadr[self.j[axis]]
        v = float(self.d.qvel[adr])
        return np.degrees(v) if axis == "w" else v * 1000.0

    # --------------------------------------------------------- gripper
    def close(self):
        """Close the jaws onto the rod -- onto its own diameter, which is
        what the tendon's travel has to be asked for.  3.5 mm was a number
        and it left the pads 4.5 mm apart round a 1 mm rod."""
        self._closing = True
        self.d.ctrl[self.a_f] = -(Gripper.JAW_OPEN - self.kit.d) / 1000.0

    def open(self):
        self._closing = False
        self.d.ctrl[self.a_f] = 0.0
        for n in self._grasp:
            self.d.eq_active[self.eq_hold[n]] = 0
        self._held, self._grasp = None, ()

    def holding(self):
        return self._held is not None

    # ---------------------------------------------------------- vacuum
    def vac_on(self):
        self._vac = True

    def vac_off(self):
        self._vac = False
        if self._vac_held is not None:
            self.d.eq_active[self.eq_vac[self._vac_held]] = 0
            self._vac_held = None

    def vac_holding(self):
        return self._vac_held is not None

    # ----------------------------------------------------------- winder
    def spin(self, rpm):
        self._ring_w = float(rpm) * 6.0          # deg/s
        self.d.ctrl[self.a_ring] = np.radians(self._ring_w)

    def angle(self):
        return float(np.degrees(self.d.qpos[self.m.jnt_qposadr[self.j_ring]]))

    def mouth_error(self, az=0.0):
        """How far the winder's mouth is from `az`, degrees, signed."""
        return ((self.angle() - float(az)) + 180.0) % 360.0 - 180.0

    def parked(self, az=0.0, tol=None):
        """A GAPPED RING IS ONLY PARKED WHEN ITS MOUTH IS FACING THE WORK.
        Stopping the drive is not parking it: the ring came off crossing 0
        wherever fourteen turns left it, was lowered onto crossing 1 through
        its own rim, and drove the rods round -- 32 degrees of yaw on one of
        them, welded there, and every crossing after it built on that."""
        tol = self.kit.park_tol() if tol is None else tol
        return abs(self._ring_w) < 1e-9 and abs(self.mouth_error(az)) <= tol

    # -------------------------------------------------------- keeper
    def keep(self, name, to="module"):
        """Weld a part where it now lies -- B's equivalent of the cell's
        keeper.

        TWO PLACES TO KEEP IT, because B builds in two places: on the
        MODULE, which is what the finished kit is, and on the JIG, which is
        where the frame is made.  `bkeep_` welds to the module and would
        carry a jigged rod half way across the table with it."""
        if to == "jig":
            self._weld(self.eq_jig[name], 0, self.b_part[name])
        else:
            self._weld(self.eq_keep[name], self.b_mod, self.b_part[name])

    def free_jig(self):
        """Let the jig go.  Called with the jaws already closed on the
        frame, never before -- a frame let go of by both at once falls."""
        for n in self.names:
            self.d.eq_active[self.eq_jig[n]] = 0

    def join_frame(self):
        """THE FRAME IS ONE PART NOW.  Four crossings wound and dosed, and
        the tic-tac-toe stops being four rods that touch -- which is the
        whole reason it can be picked up by a single member.  Recorded in
        the pose it cured in, on the jig."""
        for i, eq in enumerate(self.eq_frm, start=1):
            self._weld(eq, self.b_part["bgrid0"], self.b_part["bgrid%d" % i])

    def joined(self):
        return all(self.d.eq_active[e] for e in self.eq_frm)

    def _weld(self, eq, b1, b2):
        R1 = self.d.xmat[b1].reshape(3, 3)
        R2 = self.d.xmat[b2].reshape(3, 3)
        self.m.eq_data[eq][0:3] = 0.0
        self.m.eq_data[eq][3:6] = R1.T @ (self.d.xpos[b2] - self.d.xpos[b1])
        r = R1.T @ R2
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, r.flatten())
        self.m.eq_data[eq][6:10] = q
        self.m.eq_data[eq][10] = 1.0
        self.d.eq_active[eq] = 1

    # ------------------------------------------------------ the servos
    def tick_hook(self, dt):
        self._serve_grip()
        self._serve_vac()

    def _nearest(self, site, want_layer=None):
        p = self.d.site_xpos[site] * 1000.0
        best, bd = None, 1e9
        for n, b in self.b_part.items():
            d = float(np.linalg.norm(self.d.xpos[b] * 1000.0 - p))
            if d < bd:
                best, bd = n, d
        return best, bd

    def _serve_grip(self):
        if not self._closing or self._held is not None:
            return
        n, d = self._nearest(self.s_grip)
        if n is None or not n.startswith("bgrid") or d >= 4.0:
            return
        # A JOINED FRAME IS HELD BY ALL OF IT, and this is not a convenience.
        # Ten millimetres of pad clamping a rod does resist twist about that
        # rod -- but the rod's own axial inertia is 5.6e-12 kg m^2, and a
        # single weld carrying the frame through it holds position to 0.04 mm
        # and orientation not at all: measured, the frame rolled 5.6 degrees
        # about the held rod the instant the jig let go, and stayed there.
        # Every member welded to the jaws puts each rod's TRANSVERSE inertia
        # -- a thousand times larger -- across the same rotation.
        grasp = tuple(k for k in self.names if k.startswith("bgrid")) \
            if self.joined() else (n,)
        for k in grasp:
            self._weld(self.eq_hold[k], self.b_gw, self.b_part[k])
        self._held, self._grasp = n, grasp

    def _serve_vac(self):
        if not self._vac or self._vac_held is not None:
            return
        n, d = self._nearest(self.s_cup)
        if n == "bcollar" and d < 6.0:
            self._weld(self.eq_vac[n], self.b_vac, self.b_part[n])
            self._vac_held = n


class Runner:
    """Runs B's plan against B's HAL, one control tick per yield.

    Deliberately its own loop and not `process.Executor`: that one owns the
    truss's band states, its joint list and its fiducial, none of which
    exist here.  Sharing it would mean teaching it about a machine it does
    not drive."""

    HZ = 50.0

    def __init__(self, cell, plan, clock=None, log=None):
        from .spec import StationB
        self.c, self.plan = cell, plan
        self.clock = clock
        self.log = log or (lambda s: None)
        self.S = StationB
        self.timeline = []
        self.slow = []

    def _now(self):
        return self.c.d.time

    def _wait(self, seconds):
        for _ in range(int(round(seconds * self.HZ))):
            yield

    def _until(self, pred, timeout, what):
        t0 = self._now()
        while not pred():
            if self._now() - t0 > timeout:
                self.log("      timeout: %s" % what)
                return False
            yield
        return True

    def _move(self, goal, axes=("x", "y", "z")):
        start = {a: self.c.at(a) for a in axes if a in goal}
        mv = motion.Move(start, {a: goal[a] for a in start},
                         self.S.V_MAX, self.S.A_MAX)
        t = 0.0
        while t < mv.T:
            for a, v in mv.at(t).items():
                self.c.goto(a, v)
            t += 1.0 / self.HZ
            yield
        for a, v in mv.goal.items():
            self.c.goto(a, v)
        yield from self._until(lambda: all(self.c.settled(a) for a in mv.goal),
                               2.0, "B axes settle")
        yield from self._wait(self.S.SETTLE_S)

    def _stroke(self, axis, target):
        start = self.c.at(axis)
        T = motion.trap_time(target - start, self.S.V_MAX["z"], self.S.A_MAX["z"])
        t = 0.0
        while t < T:
            self.c.goto(axis, start + motion.trap_position(
                t, target - start, self.S.V_MAX["z"], self.S.A_MAX["z"]))
            t += 1.0 / self.HZ
            yield
        self.c.goto(axis, target)
        # AT THE TARGET, OR STOPPED.  A tool that has landed on the work
        # cannot reach its commanded position and must not be waited for.
        yield from self._until(
            lambda: self.c.settled(axis, 0.1) or abs(self.c.rate(axis)) < 0.2,
            1.5, "B stroke settle")

    def _yaw(self, target):
        """Turn the jaws to `target`, ABSOLUTELY -- the axis has hard stops,
        so there is no wrap-around to be short about.  Wrapped, a 180 degree
        command comes out as -180 and the servo clamps; it laid four
        battens a truss five degrees off their own line in the main cell
        before it was found, and B's axis has the same stops."""
        start = self.c.at("w")
        d = float(target) - start
        T = max(abs(d) / Gripper.YAW_V, 1e-6)
        t = 0.0
        while t < T:
            self.c.goto("w", start + d * (t / T))
            t += 1.0 / self.HZ
            yield
        self.c.goto("w", float(target))
        yield from self._until(lambda: self.c.settled("w", 0.5), 1.0, "B yaw")

    def run(self):
        for op in self.plan.ops:
            t0 = self._now()
            yield from self._do(op)
            t1 = self._now()
            self.timeline.append((op, t0, t1))
            if t1 - t0 > op.dt * 1.5 + 0.5:
                self.slow.append((op, t1 - t0))
        return self.c.wound

    def _do(self, op):
        k, a = op.kind, op.args
        if k == "move":
            yield from self._move(dict(a["pos"]))
        elif k == "yaw":
            yield from self._yaw(a["yaw"])
        elif k in ("extend", "retract"):
            axis = {"grip": "g", "disp": "d", "vac": "v"}[a["tool"]]
            full = self.S.tool_stroke(self.c.kit)
            yield from self._stroke(axis, 0.0 if k == "retract"
                                    else float(a.get("mm", full)))
        elif k == "grip":
            self.c.close()
            yield from self._wait(0.3)
            yield from self._until(self.c.holding, 0.6, "B grip %s" % a["part"])
            if not self.c.holding():
                self.log("      %s: jaws closed on nothing" % a["part"])
        elif k == "vac":
            self.c.vac_on()
            yield from self._wait(0.2)
            yield from self._until(self.c.vac_holding, 0.6, "B vac %s" % a["part"])
            if not self.c.vac_holding():
                self.log("      %s: the cup picked up nothing" % a["part"])
        elif k == "release":
            # NOTHING UNDER IT: the keeper takes the part before the tool
            # lets go, the same reason the mount's rods are tacked.
            for nm in a.get("parts", [a["part"]]):
                self.c.keep(nm, a.get("to", "module"))
            yield from self._wait(0.2)
            if a.get("tool") == "vac":
                self.c.vac_off()
            else:
                self.c.open()
            yield from self._wait(0.3)
        elif k == "join":
            self.c.join_frame()
            yield from self._wait(op.dt)
        elif k == "free":
            # THE JAWS ARE ALREADY ON IT.  Order matters: the jig lets go
            # second, never first.
            if not self.c.holding():
                self.log("      the jaws are not on the frame; jig held")
            else:
                self.c.free_jig()
            yield from self._wait(op.dt)
        elif k == "bead":
            yield from self._bead(op)
        elif k == "wind":
            yield from self._wind(op)
        elif k == "park":
            yield from self._park(a.get("az", 0.0))
        elif k == "bond":
            self.c.bonds.append((a.get("what"), self._now()))
            yield from self._wait(op.dt)
        else:
            yield from self._wait(op.dt)

    def _park(self, az=0.0):
        """Turn the mouth back to where the crossing goes in.

        The rate is not a speed I picked: it is the one that cannot
        overshoot the mouth's own tolerance in a single control tick."""
        tol = self.c.kit.park_tol()
        rpm = tol * self.HZ / 2.0 / 6.0
        if abs(self.c.mouth_error(az)) > tol:
            self.c.spin(-rpm if self.c.mouth_error(az) > 0 else rpm)
            ok = yield from self._until(
                lambda: abs(self.c.mouth_error(az)) <= tol,
                360.0 / (rpm * 6.0) + 1.0, "B mouth to %.0f deg" % az)
            if not ok:
                self.log("      mouth stuck %.0f deg off"
                         % self.c.mouth_error(az))
        self.c.spin(0.0)
        yield from self._wait(0.2)

    def _bead(self, op):
        """A LINE of adhesive, not a dab: the nozzle traverses the band at
        the rate that lays one metered dose per band width, so the drops
        run together.  Where the frame meets the collar the bond is the
        whole band -- that is what the plate's outline is for."""
        a, b = np.asarray(op.args["a"], float), np.asarray(op.args["b"], float)
        T = max(op.dt, 1e-3)
        t = 0.0
        while t < T:
            f = t / T
            p = a + (b - a) * f
            for ax, v in (("x", p[0]), ("y", p[1])):
                self.c.goto(ax, float(v) - op.args["off"][{"x": 0, "y": 1}[ax]])
            t += 1.0 / self.HZ
            yield
        self.c.bonds.append((op.args.get("what"), self._now()))
        yield from self._wait(0.2)

    def _wind(self, op):
        """Spin until the hinge has turned the turns asked for.  The thread
        is not simulated as a strand here -- what B has to prove is that
        the ring can GET to the crossing and turn there, and that is the
        hinge's own angle."""
        i, turns = op.args["crossing"], op.args["turns"]
        a0 = self.c.angle()
        self.c.spin(self.S.RING_RPM)
        yield from self._wait(self.S.RING_SPINUP_S)
        t0 = self._now()
        while abs(self.c.angle() - a0) < 360.0 * turns:
            if self._now() - t0 > op.dt * 2.0 + 5.0:
                self.log("      crossing %d: wind timed out at %.1f turns"
                         % (i, abs(self.c.angle() - a0) / 360.0))
                break
            yield
        self.c.spin(0.0)
        self.c.wound[i] = abs(self.c.angle() - a0) / 360.0
        yield from self._wait(0.3)


def plan(kit, cruise=None):
    """B's whole cycle, as ops.

    THE ORDER IS MECHANICAL, and every step of it is forced by something
    that cannot be worked around:

      1. the COLLAR goes on the module and its four aperture walls are
         dosed -- first, because a cup can only pick a flat part off a
         shelf, and the shelf is under everything else;
      2. the four grid rods go on the JIG, layer 0 then layer 1, because
         layer 1 lies ON layer 0 at every crossing;
      3. the four crossings are WOUND and then dosed -- resin before
         thread is resin the thread has to be pulled through;
      4. the frame, now one part, is LIFTED off the jig by a layer-1 rod
         and set on the collar;
      5. the two bands the frame lands on are BEADED.

    Nothing here is placed at a height or a station that was typed in.
    The cruise height is the winder's, the strokes are the drop from it,
    and every x and y is a point on the kit or the jig.
    """
    from .spec import StationB, Bracket, Payload, Dispenser, Process, Gripper
    P = Plan()
    hi = kit.cruise_z() if cruise is None else float(cruise)
    tz = StationB.tool_lift(kit)
    at = {"x": 0.0, "y": 0.0, "z": hi, "w": 0.0}
    OFF = {"grip": StationB.grip_off(), "disp": StationB.disp_off(),
           "vac": StationB.vac_off(), None: (0.0, 0.0)}

    def move(phase, tool=None, **goal):
        """Put `tool`'s tip where the goal says, by moving the carriage the
        tool's own offset away from it."""
        off = OFF.get(tool, (0.0, 0.0))
        g = dict(at)
        g.update(goal)
        if "x" in goal:
            g["x"] = float(goal["x"]) - off[0]
        if "y" in goal:
            g["y"] = float(goal["y"]) - off[1]
        dt = _move_time(at, g)
        at.update(g)
        return P.add("move", dt + StationB.SETTLE_S, phase,
                     pos={k: g[k] for k in "xyz"})

    def yaw(phase, w):
        d = abs(((w - at["w"]) + 180.0) % 360.0 - 180.0)
        if d > 1e-6:
            P.add("yaw", d / Gripper.YAW_V, phase, yaw=float(w))
            at["w"] = float(w)

    def down(phase, tool, z, what="extend"):
        """Extend `tool` until its tip is at `z`.  The tip sits `tz` above
        the carriage when the tool is home -- see StationB.tool_lift."""
        s = at["z"] + tz - float(z)
        P.add(what, _stroke_time(s), phase, tool=tool, mm=float(s))
        return s

    def up(phase, tool, s):
        P.add("retract", _stroke_time(s), phase, tool=tool)

    slots = {(k, i): (np.asarray(a, float), np.asarray(b, float))
             for k, i, a, b in kit.rack_slots()}

    # ---------------------------------------------------- 1. the collar
    a, b = slots[("collar", 0)]
    c = (a + b) / 2.0
    face = float(c[2]) + Bracket.SHEET / 2.0        # the cup lands on metal
    move("collar", "vac", x=float(c[0]), y=float(c[1]))
    s = down("collar", "vac", face)
    P.add("vac", 0.4, "collar", part="bcollar")
    up("collar", "vac", s)
    move("collar", "vac", x=0.0, y=0.0)
    s = down("collar", "vac", kit.seat[1])
    P.add("release", 0.5, "collar", part="bcollar", tool="vac")
    up("collar", "vac", s)
    for q in kit.aperture_bonds():
        move("collar", "disp", x=float(q[0]), y=float(q[1]))
        s = down("collar", "disp", kit.nozzle_z(q, "kit"))
        P.add("bond", Dispenser.DOSE_S + Process.DOSE_SETTLE_S, "collar",
              what="aperture")
        up("collar", "disp", s)

    # ------------------------------------------- 2. the frame, on the jig
    jrods = kit.jig_rods()
    for r, jr in zip(kit.rods, jrods):
        a, b = slots[("grid", r.index)]
        c = (a + b) / 2.0
        nm = "bgrid%d" % r.index
        move("frame", "grip", x=float(c[0]), y=float(c[1]))
        yaw("frame", 0.0)
        s = down("frame", "grip", float(c[2]))
        P.add("grip", Gripper.JAW_CLOSE_S, "frame", part=nm)
        up("frame", "grip", s)
        move("frame", "grip", x=float(jr.mid[0]), y=float(jr.mid[1]))
        yaw("frame", 0.0 if abs(float(jr.axis[0])) > 0.5 else 90.0)
        # DOWN TO WHERE THE PART GOES, not to a hover above it.  The
        # keeper welds a part where the tool left it, so a millimetre of
        # release clearance is a millimetre of permanent error -- and the
        # frame collects it twice, once onto the jig and once onto the
        # collar.  The tool lands the part; the servo stops when it does.
        s = down("frame", "grip", float(jr.mid[2]))
        P.add("release", Gripper.JAW_CLOSE_S + 0.4, "frame", part=nm, to="jig")
        up("frame", "grip", s)

    # --------------------------------------- 3. wind them, then dose them
    jx = kit.jig_crossings()
    for i, q in enumerate(jx):
        move("wind", None, x=float(q[0]), y=float(q[1]))
        P.add("park", _park_time(kit), "wind", az=0.0)
        move("wind", None, z=float(q[2]))
        P.add("wind", StationB.TURNS / StationB.RING_RPM * 60.0
              + StationB.RING_SPINUP_S, "wind", crossing=i,
              turns=StationB.TURNS)
        P.add("park", _park_time(kit), "wind", az=0.0)
        move("wind", None, z=hi)
    for i, q in enumerate(jx):
        move("wind", "disp", x=float(q[0]), y=float(q[1]))
        s = down("wind", "disp", kit.nozzle_z(q, "jig"))
        P.add("bond", Dispenser.DOSE_S + Process.DOSE_SETTLE_S, "wind",
              what="crossing%d" % i)
        up("wind", "disp", s)
    P.add("join", Process.DOSE_SETTLE_S, "wind")

    # ------------------------------------------------- 4. the transfer
    gr, gyaw = kit.frame_grip()
    hold = kit.frame_rod()
    move("place", "grip", x=float(gr[0]), y=float(gr[1]))
    yaw("place", gyaw)
    s = down("place", "grip", float(gr[2]))
    P.add("grip", Gripper.JAW_CLOSE_S, "place", part="bgrid%d" % hold.index)
    P.add("free", 0.3, "place")
    up("place", "grip", s)
    move("place", "grip", x=float(hold.mid[0]), y=float(hold.mid[1]))
    s = down("place", "grip", float(hold.mid[2]))
    # ONE TACK, NOT FOUR.  The frame is already welded to itself; welding
    # every rod to the module as well closes three loops of constraint on
    # four bodies that weigh a fifth of a gram between them.  Tack the rod
    # that actually lands on the collar and let the frame carry the rest.
    # TACK EVERY MEMBER, for the same reason the jaws hold every member:
    # one weld on one rod leaves the frame free to roll about that rod's own
    # axis, and it does -- 3 degrees, measured, the moment the jaws opened.
    # The redundancy is consistent (all four are recorded in the same pose)
    # and it is also the truth: the bead bonds both layer-0 rods down and the
    # wound crossings carry the other two.
    P.add("release", Gripper.JAW_CLOSE_S + 0.4, "place",
          part="bgrid%d" % hold.index,
          parts=["bgrid%d" % r.index for r in kit.rods])
    up("place", "grip", s)

    # ---------------------------------------------- 5. bead it onto the plate
    for i, (a, b) in enumerate(kit.collar_beads()):
        n = kit.bead_doses(a, b)
        move("bead", "disp", x=float(a[0]), y=float(a[1]))
        s = down("bead", "disp", max(kit.nozzle_z(a, "kit"),
                                     kit.nozzle_z(b, "kit")))
        P.add("bead", n * Dispenser.DOSE_S, "bead", a=list(map(float, a)),
              b=list(map(float, b)), off=list(StationB.disp_off()),
              what="band%d" % i, doses=n)
        up("bead", "disp", s)
        at["x"] = float(b[0]) - StationB.disp_off()[0]
        at["y"] = float(b[1]) - StationB.disp_off()[1]

    move("done", None, x=0.0, y=0.0, z=hi)
    return P


def _move_time(a, b):
    from .spec import StationB
    return max(motion.trap_time(b[k] - a[k], StationB.V_MAX[k], StationB.A_MAX[k])
               for k in "xyz")


def _park_time(kit):
    """Worst case for turning the mouth home: half a turn at the park rate."""
    rpm = kit.park_tol() * Runner.HZ / 2.0 / 6.0
    return 180.0 / (rpm * 6.0) + 0.2


def _stroke_time(d):
    from .spec import StationB
    return motion.trap_time(abs(d), StationB.V_MAX["z"], StationB.A_MAX["z"])
