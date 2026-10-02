"""A station, derived from what its head must reach.

The plan's station module is one machine copied five times: three axes on
rails, a head carrying that station's tools on a 3-axis load cell, one
camera looking down, a fixture and docks (plan, "The station module").
What differs between the copies is what the head must reach, so that is
the only input -- a StationSpec: the tools, and the regions (docks,
fixture, test pieces) with the tools that work on each.  The rest is
derived here:

    the head      the tools in a row across the bat with their tips level,
                  and the camera at the row's end, far enough out that its
                  cone clears them
    the travel    every region swept by every tool that works on it, every
                  fiducial under the camera, plus the home switch's zone
    the drives    per axis, the first drive in Module.DRIVES that reaches
                  the rated speed at that length: a screw whips, a rack
                  does not.  No step coarser than Module.STEP_MAX
    the preload   per axis, a constant force toward home that keeps the
                  drive's play closed whichever way the axis moves
    the limits    the rated speed and acceleration, or less where the
                  drive's critical speed, ball-return limit or pull-out
                  force (at Module.TORQUE_SF, against the preload) cannot
                  give them
    the dynamics  each axis' moving mass from the parts it carries, the
                  drive train's inertia reflected to the carriage, its
                  stiffness along the travel, its backlash and the
                  stepper's magnetic stiffness -- what station/mjcf.py
                  builds the physics from, and what check_gantry holds the
                  physics against

WHY A PRELOAD, NOT A ONE-WAY APPROACH.  Every drive has play, and a rack's
is twice the module's repeatability (Rack: 15 arcmin at the gearbox, 0.05
mm at the mesh).  Ending every move travelling the same way (CNC's G60)
was this module's first answer, and check_gantry's rig showed it does
nothing: a full-stepped rotor rings at the end of every move, the carriage
leaves the driving flank and stops anywhere in the play, and one-way
approaches spread 0.076 mm on the rack, no better than both ways.  What
works is the standard one: a constant force on each carriage toward its
home (a constant-force spring; on Z, a counterbalance that overbalances),
larger than anything that could pull the carriage off the loaded flank --
the rails' drag plus the carriage's own mass at full acceleration, plus,
on Z, the heaviest part it carries -- at Module.TORQUE_SF.  The play is
then never crossed and the approach direction stops mattering.  A process
force must push the carriage the way the preload does (x and y: work
pushed away from home; z: pressed down), or stay under the preload.

Nothing here is typed: a region is where it is because line.py laid the
station out, and a tool is where it is because of its neighbours' sizes.
"""
from dataclasses import dataclass, field
from math import pi, sqrt, tan, atan2, degrees, radians, log

import numpy as np

from truss.spec import Gantry, stepper_scale_sigma
from ..spec import (Stepper, SCREWS, ScrewEnds, Rack, Rail, Frame, HomeSwitch, LoadCell, HeadCam,
                    Fiducial, Gripper2, Vacuum, Magnet, Pusher, Pogo, Spindle, ToolSlide, Build, Module,
                    Artefact)
from ..product import Fit

G = 9.81e-3                    # N per gram
AXES = ("x", "y", "z")
TOOLS = {"gripper": Gripper2, "vacuum": Vacuum, "magnet": Magnet, "pusher": Pusher, "pogo": Pogo,
         "spindle": Spindle}
# the tools that take hold of a part, and so can carry the test token
PICKERS = ("gripper", "vacuum", "magnet", "spindle")


def tool_height(kind):
    """A tool's height from its slide's carriage to its tip."""
    T = TOOLS[kind]
    if kind == "gripper":
        return T.BODY[2] + T.JAW[2]
    if kind == "pogo":
        return T.BODY[2] + T.TRAVEL
    return T.BODY[2]


# ================================================================ INPUT
@dataclass(frozen=True)
class Region:
    """A place the head works on, in the station frame (x along the bat --
    the module's x -- y across it, z up, from the station's floor datum).

    `reach` names the tools that must reach every point of it; "camera"
    means its fiducials must come under the camera.  `tall` is the tallest
    thing standing on it, above its top."""
    name: str
    centre: tuple
    size: tuple
    top: float = 0.0
    tall: float = 0.0
    reach: tuple = ()
    fiducials: bool = True

    @property
    def lo(self):
        return (self.centre[0] - self.size[0] / 2.0, self.centre[1] - self.size[1] / 2.0)

    @property
    def hi(self):
        return (self.centre[0] + self.size[0] / 2.0, self.centre[1] + self.size[1] / 2.0)

    def fiducial_points(self):
        """Three discs in an L, at three corners, each inset by its keep-out,
        as a PCB panel's three global fiducials: the region's place and turn
        and, since the head's steps are the ruler, the ruler's scale along x
        and along y and their squareness across it.  The first two are
        opposite corners.  Two were enough for the place and the turn, but
        a rack and a screw grow apart: with a similarity fit to two, the
        adult's parts tray was 0.08 mm out at a corner nobody looked at."""
        if not self.fiducials:
            return []
        inset = Fiducial.D * Fiducial.CLEAR
        (x0, y0), (x1, y1) = self.lo, self.hi
        return [(x0 + inset, y0 + inset, self.top), (x1 - inset, y1 - inset, self.top),
                (x0 + inset, y1 - inset, self.top)]


@dataclass(frozen=True)
class StationSpec:
    """What a station's head carries and where it has to work."""
    name: str
    tools: tuple
    regions: tuple
    drives: tuple = Module.DRIVES
    payload: float = 0.0       # g, the heaviest part a tool carries
    force: float = 0.0         # N, the largest process force a tool applies
    # a ball a fixture sweeps while the head waits at home, ((x, y, z), r):
    # the calibration station's gimbal turns the bat through every
    # orientation.  Z homes high enough for the retracted tips to clear its
    # top, and the frame's posts stand outside it
    sweep: tuple = None

    @property
    def overhead(self):
        """mm above the floor datum the head must clear at home."""
        return 0.0 if self.sweep is None else self.sweep[0][2] + self.sweep[1]


# ================================================================ HEAD
@dataclass
class ToolMount:
    kind: str
    tip: tuple                 # (x, y, z) of the tip, retracted, from the head point
    body: tuple                # the tool's own size
    spacer: float              # mm of spacer under the plate, so every tip is level
    slot: float                # mm the tool occupies across the row


@dataclass
class Head:
    """The head point is the centre of the tool plate's underside: every
    axis coordinate is where the controller believes it is."""
    tools: dict
    plate: tuple               # (x, y, thickness)
    camera: tuple              # (x, y, z): the lens's front, from the head point
    half_fov: tuple            # deg, the camera's half-angles along (x, y)
    z_retracted: float
    z_extended: float
    mass: float                # g below the load cell: plate, slides, spacers, tools

    def tip(self, kind, extended=True):
        x, y, z = self.tools[kind].tip
        return (x, y, z - (ToolSlide.STROKE if extended else 0.0))


def head_layout(tools):
    """Tools in a row along y, in the order given, tips level; the camera
    beyond the row's +y end.

    Across the bat, because the tools work along it: the pusher pushes
    along the bat and every insertion is along its axis (plan, "How the
    harder fits are made"), so the head's x faces are kept clear.  The
    camera's image is laid long side along x, so its narrow half-angle is
    the one that has to clear the row."""
    slots = [max(TOOLS[k].BODY[1], ToolSlide.BODY[1]) for k in tools]
    row = sum(slots) + Module.TOOL_GAP * (len(slots) - 1)
    depth = max([TOOLS[k].BODY[0] for k in tools] + [ToolSlide.BODY[0]])
    drop = max(ToolSlide.BODY[2] + tool_height(k) for k in tools)
    out, y = {}, -row / 2.0
    for k, s in zip(tools, slots):
        out[k] = ToolMount(k, (0.0, y + s / 2.0, -drop), TOOLS[k].BODY,
                           drop - ToolSlide.BODY[2] - tool_height(k), s)
        y += s + Module.TOOL_GAP
    half = (degrees(atan2(HeadCam.W * HeadCam.PIXEL / 2.0, HeadCam.F)),
            degrees(atan2(HeadCam.H * HeadCam.PIXEL / 2.0, HeadCam.F)))
    # the lens's front level with the plate's underside: the cone has to
    # pass the nearest tool all the way down to its tip
    cone = drop * tan(radians(half[1]))
    cam_y = row / 2.0 + max(Module.TOOL_GAP + HeadCam.BOARD / 2.0, cone)
    plate = (depth, row, Frame.PLATE_T)
    mass = (sum(TOOLS[k].MASS + ToolSlide.MASS for k in tools)
            + plate[0] * plate[1] * plate[2] * Frame.RHO_AL
            + sum(m.spacer * depth * ToolSlide.BODY[1] * Frame.RHO_AL * 0.3 for m in out.values()))
    return Head(out, plate, (0.0, cam_y, 0.0), half, -drop, -drop - ToolSlide.STROKE, mass)


# ================================================================ DRIVES
def _screw_j(s, length):
    """kg m^2 of a steel screw spinning on its axis (its nominal diameter)."""
    return ScrewEnds.RHO_STEEL * pi * s.d0 ** 4 * length / 32.0 * 1e-9


@dataclass
class Drive:
    """One axis' drive: a ball screw or a rack and pinion through a
    planetary gearbox, n of them in parallel, each on its own stepper."""
    kind: str
    name: str
    n: int
    lead: float                # mm of travel a motor revolution
    length: float              # mm: between a screw's bearings, or of rack
    screw: object = None
    ratio: int = 0

    @property
    def step(self):
        return self.lead / Gantry.STEPS_PER_REV

    @property
    def w(self):
        """rad of motor per metre of travel."""
        return 2.0 * pi / (self.lead * 1e-3)

    @property
    def efficiency(self):
        return ScrewEnds.EFFICIENCY if self.kind == "screw" else Rack.EFFICIENCY

    def j_motor(self):
        """kg m^2 at one motor's shaft."""
        if self.kind == "screw":
            return Stepper.ROTOR_J + ScrewEnds.COUPLING_J + _screw_j(self.screw, self.length)
        return Stepper.ROTOR_J + Rack.GEAR_J

    def reflected(self):
        """g: the drive trains' inertia as a mass at the carriage."""
        return self.n * self.j_motor() * self.w ** 2 * 1000.0

    def f_magnet(self):
        """N: the motors' holding force at the carriage -- the peak of the
        stepper's sine law, which nothing beyond it can hold."""
        return self.n * Stepper.HOLD_NM * self.w

    def k_magnet(self):
        """N/mm: the slope of that law at no lag, T_hold x N_r reflected."""
        return self.n * Stepper.HOLD_NM * Stepper.TEETH * self.w ** 2 * 1e-3

    def rpm(self, v):
        return v / self.lead * 60.0

    def torque(self, v):
        """N m one motor has at v mm/s: the pull-out curve, nothing past its end."""
        rpm, T = zip(*Stepper.PULLOUT)
        return float(np.interp(self.rpm(abs(v)), rpm, T, right=0.0))

    def f_pullout(self, v):
        """N all motors can push with at v before they drop steps."""
        return self.n * self.torque(v) * self.w * self.efficiency

    def k_drive(self, s):
        """N/mm, motor shaft to carriage, with the nut s mm along the travel
        from the motor's end.  A screw is a bar loaded between its fixed
        bearing and the nut, so it softens as the nut runs away from the
        motor; a rack does not."""
        if self.kind == "screw":
            sc = self.screw
            x = ScrewEnds.END_L + sc.nut_l / 2.0 + max(s, 0.0)
            shaft = ScrewEnds.E_STEEL * pi / 4.0 * sc.d_root ** 2 / x
            coupling = ScrewEnds.COUPLING_K * self.w ** 2 * 1e-3
            parts = (shaft, sc.nut_k * 1e3, ScrewEnds.BEARING_K * 1e3, coupling)
        else:
            r = Rack.MODULE * Rack.PINION_Z / 2.0 * 1e-3
            gear = Rack.GEAR_K * 60.0 * 180.0 / pi / r ** 2 * 1e-3      # at the output, to the pitch line
            parts = (gear, Rack.MESH_C * Rack.FACE * 1e3)
        return self.n / sum(1.0 / k for k in parts)

    def backlash(self):
        if self.kind == "screw":
            return self.screw.play
        r = Rack.MODULE * Rack.PINION_Z / 2.0
        return radians(Rack.GEAR_PLAY / 60.0) * r + Rack.MESH_PLAY

    def v_limit(self):
        """(mm/s, why): the fastest this drive may turn, and what sets it."""
        lims = [(max(r for r, _ in Stepper.PULLOUT) * self.lead / 60.0, "the motor's curve")]
        if self.kind == "screw":
            sc = self.screw
            n_crit = ScrewEnds.LAMBDA * 1e7 * sc.d_root / self.length ** 2
            lims += [(n_crit * self.lead / 60.0, "the screw whips"),
                     (ScrewEnds.DN_MAX / sc.d0 * self.lead / 60.0, "the balls' return")]
        return min(lims)

    def moving_mass(self):
        """g of the drive that rides with the carriage: a nut, or the motor,
        gearbox and pinion of a rack drive."""
        if self.kind == "screw":
            return 0.0
        return self.n * (Stepper.MASS + Rack.GEAR_MASS)

    def fixed_mass(self):
        if self.kind == "screw":
            return self.n * (Stepper.MASS + pi / 4.0 * self.screw.d0 ** 2 * self.length * ScrewEnds.RHO_STEEL)
        return self.n * self.length * Rack.MASS


def candidates(kind, n, travel):
    if kind == "screw":
        return [Drive("screw", s.name, n, s.lead, travel + s.nut_l + 2.0 * ScrewEnds.END_L, screw=s)
                for s in SCREWS]
    if kind == "rack":
        circ = pi * Rack.MODULE * Rack.PINION_Z
        ratios = [r for r in Rack.RATIOS if circ / r / Gantry.STEPS_PER_REV <= Module.STEP_MAX]
        if not ratios:
            return []
        r = ratios[0]
        return [Drive("rack", "rack m%g z%d, %d:1" % (Rack.MODULE, Rack.PINION_Z, r), n, circ / r,
                      travel + 2.0 * Rail.BLOCK_L, ratio=r)]
    raise ValueError("no drive called %r" % kind)


def limits(drive, axis, moving, drag, unbalanced):
    """(v_max, a_max, why) of an axis on this drive: the rated values, cut
    to what the drive can turn and what its motors can push at
    Module.TORQUE_SF.  `unbalanced` is the gravity the drive carries (N)."""
    v_lim, why = drive.v_limit()
    v = min(Module.V_RATED[axis], v_lim)
    held = [why] if v_lim < Module.V_RATED[axis] else []
    m = (moving + drive.reflected()) * 1e-3          # kg
    a = Module.A_RATED[axis]
    spare = drive.f_pullout(v) / Module.TORQUE_SF - drag - unbalanced
    if spare < m * a * 1e-3:
        a = max(spare / m * 1e3, 0.0)
        held.append("its inertia: %.0f kg reflected" % (drive.reflected() * 1e-3))
    return v, a, "; ".join(held) or "rated"


def choose(axis, travel, moving, drag, unbalanced, n, kinds):
    """The first drive, in the order `kinds` tries them, that is as fine as
    Module.STEP_MAX and reaches the rated speed and acceleration; failing
    that, whichever comes closest."""
    best = None
    for kind in kinds:
        for dr in candidates(kind, n, travel):
            v, a, why = limits(dr, axis, moving + dr.moving_mass(), drag, unbalanced)
            fine = dr.step <= Module.STEP_MAX + 1e-12
            if fine and v >= Module.V_RATED[axis] - 1e-9 and a >= Module.A_RATED[axis] - 1e-9:
                return dr, v, a, why
            score = (fine, v, a)
            if best is None or score > best[0]:
                best = (score, dr, v, a, why)
    return best[1:]


# ================================================================ AXES
@dataclass
class Axis:
    name: str
    lo: float                  # mm, the head point's coordinate at each end of the travel
    hi: float
    home: float                # where it homes: x and y low, z high (homing never lowers a tool)
    preload: float             # N, the constant force pulling the carriage toward home
    drive: Drive
    moving: float              # g this axis carries, its own carriage included
    drag: float                # N of rail drag
    v_max: float
    a_max: float
    why: str                   # what holds v_max (or a_max) below rated, else "rated"

    @property
    def travel(self):
        return self.hi - self.lo

    @property
    def toward(self):
        """-1 if home is at the low end, +1 at the high: the preload's sign."""
        return -1 if self.home <= self.lo + 1e-9 else +1

    def s(self, q):
        """mm from the motor's end (the home end) to coordinate q."""
        return abs(q - self.home)

    def k_drive(self, q):
        return self.drive.k_drive(self.s(q))

    def k_total(self, q):
        """N/mm from the commanded position to the carriage: the stepper's
        magnetic spring in series with the drive."""
        return 1.0 / (1.0 / self.drive.k_magnet() + 1.0 / self.k_drive(q))

    def mass(self):
        """kg the drive accelerates: the carriage and its drive's inertia."""
        return (self.moving + self.drive.reflected()) * 1e-3

    def modes(self, q):
        """rad/s of the two modes the axis rings in: the rotor on its
        magnetic spring, the carriage on the drive."""
        k1, k2 = self.drive.k_magnet() * 1e3, self.k_drive(q) * 1e3
        m1, m2 = self.drive.reflected() * 1e-3, self.moving * 1e-3
        K = np.array([[k1 + k2, -k2], [-k2, k2]])
        Minv = np.diag([1.0 / m1, 1.0 / m2])
        return tuple(sorted(np.sqrt(np.abs(np.linalg.eigvals(Minv @ K)))))


# ================================================================ MODULE
@dataclass
class StationModule:
    """One copy of the module, derived for one station."""
    spec: StationSpec
    head: Head
    axes: dict
    regions: dict
    fiducials: list            # (region, (x, y, z)) in the station frame
    clear: float               # mm a moving tool clears everything by
    z_safe: float              # head z at which the retracted tips clear everything ...
    z_carry: float             # ... and at which a part hanging from them does too
    look: dict                 # region -> mm, the lens to the region's top at z_carry
    counterbalance: float      # N holding Z up: the Z assembly's weight and Z's preload
    fits: list = field(default_factory=list)
    z_over: dict = field(default_factory=dict)   # region -> the head z it looks down at it from

    def tip(self, kind, extended=True):
        return self.head.tip(kind, extended)

    def head_for(self, kind, p, extended=True):
        """The head point that puts this tool's tip on station point p,
        from the drawing (commissioning measures the truth)."""
        t = self.tip(kind, extended)
        return tuple(p[i] - t[i] for i in range(3))

    def camera_over(self, p):
        c = self.head.camera
        return (p[0] - c[0], p[1] - c[1])

    def mm_per_px(self, rng):
        return rng * HeadCam.PIXEL / HeadCam.F


def home_zone():
    """mm beyond the last working position that the home switch may trip
    in: Module.Z_SIGMAS of where it was mounted and how it repeats."""
    return Module.Z_SIGMAS * (Build.HOME + HomeSwitch.REPEAT)


def preload(axis, drag, carried, payload):
    """N toward home that keeps a carriage on its loaded flank: more than
    the drag, the carriage's own mass (g) at the rated acceleration and
    the payload's weight (N) can pull it off, at Module.TORQUE_SF.  The
    drive's inertia is on the far side of the play, so it does not count."""
    return Module.TORQUE_SF * (drag + carried * 1e-3 * Module.A_RATED[axis] * 1e-3 + payload)


def _clearance():
    """What a moving tool keeps between itself and anything below it:
    Module.Z_SIGMAS of everything not yet measured in z -- where the tool
    sits on its plate, where the home switch trips, a printed part's
    height."""
    from ..spec import Print
    return Module.Z_SIGMAS * sqrt(Build.TOOL_Z ** 2 + Build.HOME ** 2 + Print.TOL ** 2)


def region_build(name):
    """(mm, deg), 1 sigma: how far a region may be built from its drawing,
    in place and in turn about its centre.  The fixture is located by
    dowels on the frame; a dock is set on the floor by hand; the reference
    pin stands in the fixture."""
    if name in ("fixture", "pin"):
        return Build.FIXTURE_XY, 0.0
    return Build.DOCK_XY, Build.DOCK_YAW


def build_margin(r, i, offset):
    """mm, along axis i, by which the head may have to go past where the
    drawing puts region r's edge for a tool (or the camera) to reach it as
    built, to Module.Z_SIGMAS: the region's place, its turn swinging its
    far side across, and how far the tool is from where it was drawn
    (offset, 1 sigma)."""
    xy, yaw = region_build(r.name)
    swing = radians(yaw) * r.size[1 - i] / 2.0
    return Module.Z_SIGMAS * sqrt(xy ** 2 + swing ** 2 + offset ** 2)


def gaps():
    """mm a touch starts clear of what it expects to meet, and goes on
    past it: Module.Z_SIGMAS of what is not yet measured, plus the free
    travel it reads its noise over.  In z: a tool's height and the home
    switch.  In x and y: a tool on its plate and the camera on its
    carriage."""
    free = (Module.TOUCH_FREE + 1) * Module.STEP_MAX
    z = Module.Z_SIGMAS * sqrt(Build.TOOL_Z ** 2 + Build.HOME ** 2) + free
    xy = Module.Z_SIGMAS * sqrt(Build.TOOL_XY ** 2 + Build.CAM_XY ** 2) + free
    return z, xy


def touch_cap(ax, sign):
    """N a touch's readings may rise to, creeping along axis ax in direction
    sign.  Work that pushes back toward home pushes the way the preload
    does, and only the load cell's rating limits it; work that pushes away
    from home is held off by the preload alone, and past what the preload
    was sized to hold the carriage would leave its flank.  Both at
    Module.TORQUE_SF."""
    if -sign == ax.toward:
        return LoadCell.RATED / Module.TORQUE_SF
    return ax.preload / Module.TORQUE_SF


def work_gap():
    """mm a press, a screw or a probe starts above the top it expects, once
    the station has measured itself: a part's height to Module.Z_SIGMAS of
    the print's scatter, and the free travel a touch reads its noise over."""
    from ..spec import Print
    return Module.Z_SIGMAS * Print.TOL + (Module.TOUCH_FREE + 1) * Module.STEP_MAX


def sink(kind):
    """mm a tool's extended tip goes below the surface it works on: the
    pogo block's pins compress to make contact; every other tool stops on
    what it meets."""
    return Pogo.WORK[1] if kind == "pogo" else 0.0


def derive(spec):
    """The station module that does spec's work."""
    head = head_layout(spec.tools)
    regions = {r.name: r for r in spec.regions}
    clear = _clearance()
    fids = [(r.name, p) for r in spec.regions for p in r.fiducial_points()]

    # ---------------------------------------------------- x and y travel
    # every point of every region a tool works on, and every fiducial under
    # the camera, wherever the build may have put them
    lo, hi = [np.inf, np.inf], [-np.inf, -np.inf]
    for r in spec.regions:
        for kind in r.reach:
            if kind == "camera":
                continue
            if kind not in head.tools:
                raise ValueError("%s works on %s, but the head has no %s" % (kind, r.name, kind))
            t = head.tip(kind)
            for i in range(2):
                m = build_margin(r, i, Build.TOOL_XY)
                lo[i] = min(lo[i], r.lo[i] - t[i] - m)
                hi[i] = max(hi[i], r.hi[i] - t[i] + m)
    for name, p in fids:
        h = (p[0] - head.camera[0], p[1] - head.camera[1])
        for i in range(2):
            m = build_margin(regions[name], i, Build.CAM_XY)
            lo[i], hi[i] = min(lo[i], h[i] - m), max(hi[i], h[i] + m)

    # ------------------------------------------------------- z travel
    tops = [r.top for r in spec.regions]
    tallest = max(r.top + r.tall for r in spec.regions)
    part = max(r.tall for r in spec.regions)
    # an extended tip down to the lowest top, as deep as a tool sinks into
    # it, and as far again as a touch that expects it there has to search
    z_low = min(tops) - max(sink(k) for k in spec.tools) - gaps()[0] - head.z_extended
    z_safe = tallest + clear - head.z_retracted       # retracted tips clear of everything
    z_carry = z_safe + part                           # ... with a part hanging from one
    # over each region, the height that clears what stands on it and on
    # every region the head overlaps there, a part hanging: where the head
    # looks down from.  Where every region stands at one height it is
    # z_carry; where they do not (the calibration station's clamp stands a
    # gimbal's height over its dock), a camera looking from the highest
    # sees the lowest too small to find a fiducial
    grow = np.hypot(head.plate[0], head.plate[1] + 2.0 * head.camera[1]) / 2.0

    def over(r):
        near = [q for q in spec.regions
                if all(q.lo[i] - grow <= r.hi[i] and r.lo[i] <= q.hi[i] + grow for i in range(2))]
        return max(q.top + q.tall for q in near) + clear - head.z_retracted + part
    z_over = {r.name: over(r) for r in spec.regions}
    look = {r.name: z_over[r.name] + head.camera[2] - r.top for r in spec.regions}

    # ------------------------------------------- masses, from the parts
    # Z: everything below the load cell, the load cell, the Z plate and its
    # carriages, the camera.  The Z plate spans the row and the camera.
    zp = (head.plate[0], head.camera[1] + HeadCam.BOARD / 2.0 + head.plate[1] / 2.0,
          (z_carry - z_low) + Gantry.BLOCK_PITCH + Rail.BLOCK_L)
    m_z = (head.mass + LoadCell.MASS + HeadCam.MASS + 2.0 * Rail.BLOCK_MASS
           + zp[1] * zp[2] * Frame.PLATE_T * Frame.RHO_AL)
    axes = {}

    def build(name, a_lo, a_hi, moving, n, drag, payload, home_low):
        # the preload is sized on the carriage, which on a rack carries its
        # own motors, so it needs the drive, which is chosen against the
        # preload: two passes
        hz, pre = home_zone(), preload(name, drag, moving, payload)
        span = a_hi - a_lo + hz
        for _ in range(2):
            dr, v, a, why = choose(name, span, moving, drag, pre, n, spec.drives)
            pre = preload(name, drag, moving + dr.moving_mass(), payload)
        lo_, hi_ = (a_lo - hz, a_hi) if home_low else (a_lo, a_hi + hz)
        return Axis(name, lo_, hi_, lo_ if home_low else hi_, pre, dr, moving + dr.moving_mass(), drag, v, a, why)

    # Z carries the Z assembly, its weight counterbalanced and its preload
    # pulling it up, against which it carries the heaviest part a tool
    # picks up.  It homes at the top: homing never lowers a tool -- and
    # home is high enough that the retracted tips clear the overhead
    z_top = max(z_carry, spec.overhead + clear - head.z_retracted) if spec.overhead > 0.0 else z_carry
    axes["z"] = build("z", z_low, z_top, m_z, 1, 2.0 * Rail.DRAG, spec.payload * G, False)
    # Y carries Z, its drive, the Y carriage plate and its blocks
    yp = (zp[0] + 2.0 * Frame.BEAM, zp[1])
    m_y = (m_z + axes["z"].drive.fixed_mass() + 2.0 * Rail.BLOCK_MASS
           + (axes["z"].travel + Gantry.BLOCK_PITCH + Rail.BLOCK_L) * Rail.MASS
           + yp[0] * yp[1] * Frame.PLATE_T * Frame.RHO_AL)
    axes["y"] = build("y", lo[1], hi[1], m_y, 1, 2.0 * Rail.DRAG, 0.0, True)
    # X carries the bridge: Y, its beam and rail, its drive, four blocks
    beam = axes["y"].travel + yp[1] + 2.0 * Frame.BEAM
    m_x = (m_y + axes["y"].drive.fixed_mass() + beam * (Frame.BEAM_MASS + Rail.MASS)
           + 2.0 * Module.X_DRIVES * Rail.BLOCK_MASS)
    axes["x"] = build("x", lo[0], hi[0], m_x, Module.X_DRIVES, 2.0 * Module.X_DRIVES * Rail.DRAG, 0.0, True)

    counterbalance = m_z * G + axes["z"].preload
    mod = StationModule(spec, head, axes, regions, fids, clear, z_safe, z_carry, look, counterbalance,
                        z_over=z_over)
    mod.fits = fits(mod)
    return mod


# ================================================================ FITS
def settle_s(ax, q):
    """Seconds for the slower mode to ring down from a full step to a tenth
    of the repeatability, at Stepper.ZETA."""
    w = min(ax.modes(q))
    return log(ax.drive.step / (Module.REPEAT / 10.0)) / (Stepper.ZETA * w) if ax.drive.step > Module.REPEAT / 10.0 else 0.0


def fits(mod):
    out = []

    def fit(name, margin, detail=""):
        out.append(Fit(name, float(margin), detail))

    sp = mod.spec
    for k, ax in mod.axes.items():
        dr = ax.drive
        fit("%s: a step is no coarser than the module's %.3f mm" % (k, Module.STEP_MAX),
            Module.STEP_MAX - dr.step, "%s, %.4f mm" % (dr.name, dr.step))
        fit("%s: the drive reaches the rated %.0f mm/s at %.0f mm of travel"
            % (k, Module.V_RATED[k], ax.travel), ax.v_max - Module.V_RATED[k],
            "%s: %.1f mm/s (%s)" % (dr.name, ax.v_max, ax.why))
        fit("%s: and the rated %.0f mm/s^2" % (k, Module.A_RATED[k]), ax.a_max - Module.A_RATED[k],
            "%.0f mm/s^2" % ax.a_max)
        need = ax.mass() * ax.a_max * 1e-3 + ax.drag + ax.preload
        fit("%s: the motors push %.0fx what the hardest move asks, at full speed" % (k, Module.TORQUE_SF),
            dr.f_pullout(ax.v_max) / max(need, 1e-9) - Module.TORQUE_SF,
            "%.0f N at %.0f rpm against %.1f N" % (dr.f_pullout(ax.v_max), dr.rpm(ax.v_max), need))
        if dr.kind == "screw":
            v_lim, _ = dr.v_limit()
            fit("%s: the screw stays under its critical speed and ball-return limit" % k,
                v_lim - ax.v_max, "%.0f mm/s allowed on %.0f mm of %s" % (v_lim, dr.length, dr.name))
        far = ax.lo if ax.toward > 0 else ax.hi
        fit("%s: the axis rings down inside the settle time, at the far end" % k,
            Gantry.SETTLE_S - settle_s(ax, far),
            "%.3f s; modes %.0f and %.0f Hz" % ((settle_s(ax, far),) + tuple(w / 2 / pi for w in ax.modes(far))))
    # a press: the tool's slide must hold it, the load cell must read it,
    # and Z must hold it on its magnets with the margin
    if sp.force > 0.0:
        z = mod.axes["z"]
        fit("the tool slides hold the largest press", ToolSlide.FORCE - sp.force,
            "%.0f N against %.0f" % (ToolSlide.FORCE, sp.force))
        fit("the load cell reads the largest press", LoadCell.RATED - sp.force)
        fit("z holds the largest press with the margin",
            z.drive.f_magnet() / (sp.force + z.preload) - Module.TORQUE_SF)
    # a fixture that moves while the head waits at home
    if sp.overhead > 0.0:
        z = mod.axes["z"]
        lowest = z.hi + mod.head.z_retracted
        fit("the head at home clears what moves under it by %.1f mm" % mod.clear,
            lowest - sp.overhead - mod.clear, "retracted tips at %.1f, over %.1f" % (lowest, sp.overhead))
        (sx, sy, _), r = sp.sweep
        fr = frame(mod)
        near = min(np.hypot(px - sx, py - sy) for px in fr["posts"] for py in fr["rails"]) - Frame.BEAM / sqrt(2.0)
        fit("the frame's posts stand outside what moves", near - r, "nearest post %.1f from it" % near)
    # the camera: a fiducial wide enough to find, from the look height
    for name, rng in mod.look.items():
        if not mod.regions[name].fiducials:
            continue
        px = Fiducial.D / mod.mm_per_px(rng)
        fit("the camera sees a fiducial on %s at %.0f px, from %.0f mm" % (name, HeadCam.MIN_PX, rng),
            px - HeadCam.MIN_PX, "%.1f px" % px)
    return out


def frame(mod):
    """The module's fixed frame, as station/mjcf.py draws it: two X rails
    on four posts, one each side of the travel, high enough for the bridge
    to carry Y and Z over everything.  {"rail_z", "rails" (y of each),
    "posts" (x of each pair), "beam"}, in the station frame."""
    A, H = mod.axes, mod.head
    stack = A["z"].travel + 2.0 * Rail.BLOCK_L + LoadCell.SIZE[2] + Frame.PLATE_T
    edge = 60.0                # mm the frame stands past the travel: a drawing, as station/mjcf.py's
    rails = [A["y"].lo - edge, A["y"].hi + H.camera[1] + edge]
    posts = (A["x"].lo - edge, A["x"].hi + edge)
    sw = mod.spec.sweep
    if sw is not None:
        # a post inside the swept ball's plan would stand in its way: the
        # rails move out until their posts clear it
        (sx, sy, _), r = sw
        reach = r + Frame.BEAM / sqrt(2.0)
        dx = min(abs(p - sx) for p in posts)
        if dx < reach:
            half = sqrt(reach ** 2 - dx ** 2)
            rails = [min(rails[0], sy - half), max(rails[1], sy + half)]
    return {"rail_z": A["z"].hi + stack, "stack": stack, "rails": tuple(rails), "posts": posts,
            "beam": Frame.BEAM}


def thermal_reach():
    """mm from home beyond which a shift's warm-up (3 sigma) moves the head
    by more than the repeatability: past it, a stale look is not good
    enough and the executor looks again before it places."""
    return Module.REPEAT / (3.0 * stepper_scale_sigma())


# ============================================================ FROM THE LINE
# The plan's table "The stations": what each head carries.  The frame
# station's head is the truss cell's own (M5); as a module it carries the
# gripper that moves mandrels and frames between its docks and its cage.
STATION_TOOLS = {
    "frame": ("gripper",),
    "electronics": ("magnet", "gripper", "pusher", "pogo"),
    "sleeve": ("gripper", "pusher"),
    "calibration": ("gripper",),
    "pack": ("gripper", "magnet", "spindle", "pusher", "vacuum"),
}
# which regions each tool works on: the gripper moves bats, frames, tubes
# and stretchers between every dock and the fixture; the part-takers take
# parts from the parts trays to the fixture; the pusher and the pogo block
# only ever work at the fixture
WORKS_ON = {"gripper": "all", "magnet": "parts", "vacuum": "parts", "spindle": "parts",
            "pusher": "fixture", "pogo": "fixture"}


def _parts_dock(name):
    return name in ("parts", "cores")


def payload(d, docks):
    """g: the heaviest single thing the head lifts from these docks -- a bat
    as shipped (every part but the customer's cells), a tube, a core, a
    frame.  A dock of small parts lifts less than any of these.  Mandrels
    and stretchers are not modelled yet (M5, M7); a station that lifts
    them counts the bat until then."""
    shipped = sum(p.mass for p in d.parts.values() if p.source != "customer")
    holds = {"bats": shipped, "tubes": d.tube["mass"], "cores": d.parts["core"].mass,
             "frames": d.parts["frame"].mass}
    return max([holds.get(k, 0.0) for k in docks] + [shipped])


def from_line(st, dg, d, tools=None, drives=None):
    """A StationSpec from one of line.py's stations: its docks' trays and
    its fixture as regions, the reference pin at the fixture's corner."""
    tools = tuple(STATION_TOOLS[st.name] if tools is None else tools)
    # line.py lays a station out on the floor, u along the aisle and v into
    # the station; x is the bat's axis, which runs along u when the bats lie
    # along the aisle.  A tray's length runs along its bats, so in the
    # station's frame a tray is (length, width) whichever way it lies.
    xy = (lambda u, v: (u, v)) if dg.along else (lambda u, v: (v, u))
    lift = d.tube["od"]
    tray = (d.tray["length"], d.tray["width"])

    def reach(where):
        out = ["camera"]
        for k in tools:
            w = WORKS_ON[k]
            if w == "all" or (w == "parts" and _parts_dock(where)) or w == where:
                out.append(k)
        return tuple(out)

    regions = []
    for k in st.docks:
        regions.append(Region(k, xy(st.dock_u[k], dg.depth / 2.0), tray, 0.0, lift, reach(k)))
    fc = xy(*st.fixture)
    fs = xy(*st.fixture_size)
    # where the station says the head works on only part of its fixture
    # (the calibration station's head reaches the gimbal's clamp, not the
    # gimbal), that part is the region, at its own height
    work = getattr(st, "work", None)
    if work is not None:
        fs, top = work
    else:
        top = 0.0
    regions.append(Region("fixture", fc, fs, top, lift, reach("fixture")))
    regions.append(pin_region(regions[-1], tools))
    sweep = getattr(st, "sweep", None)
    if sweep is not None:
        (u, v, z), r = sweep
        sweep = (xy(u, v) + (z,), r)
    return StationSpec(st.name, tools, tuple(regions), Module.DRIVES if drives is None else tuple(drives),
                       payload=payload(d, st.docks), sweep=sweep)


def keepout(tools):
    """mm round a test piece a tool needs to come at it from any side: the
    widest tool's half-size and the piece's clearance."""
    return max(max(TOOLS[k].BODY[0], TOOLS[k].BODY[1]) for k in tools) / 2.0 + Module.TOOL_GAP


def pin_region(fixture, tools, at=0):
    """The reference pin's region, in the fixture's -x, -y corner (or the
    at-th place along its x): every tool touches it from four sides and
    from above, so the region is the pin and a tool's width all round."""
    side = Artefact.PIN_D + 2.0 * keepout(tools)
    x0, y0 = fixture.lo
    c = (x0 + side / 2.0 + at * side, y0 + side / 2.0)
    return Region("pin", c, (side, side), fixture.top, Artefact.PIN_H, ("camera",) + tuple(tools), False)


def report(mod):
    out = []
    w = out.append
    w("station %s: tools %s" % (mod.spec.name, ", ".join(mod.spec.tools)))
    for k, ax in mod.axes.items():
        dr = ax.drive
        w("  %s %7.1f mm (%.1f .. %.1f, home %.1f)  %-22s step %.4f  %5.1f mm/s %4.0f mm/s^2 (%s)"
          % (k, ax.travel, ax.lo, ax.hi, ax.home, dr.name, dr.step, ax.v_max, ax.a_max, ax.why))
        far = ax.lo if ax.toward > 0 else ax.hi
        w("      moving %.0f g + drive %.0f g; play %.3f mm, preloaded %.1f N; stiffness %.1f..%.1f N/um; "
          "modes %s Hz" % (ax.moving, dr.reflected(), dr.backlash(), ax.preload, ax.k_total(ax.home) / 1e3,
                           ax.k_total(far) / 1e3, "/".join("%.0f" % (w_ / 2 / pi) for w_ in ax.modes(far))))
    w("  head: plate %.0f x %.0f; tips %.1f retracted, %.1f extended; camera at y %.1f; %.0f g below "
      "the load cell; counterbalance %.1f N" % (mod.head.plate[0], mod.head.plate[1], mod.head.z_retracted,
                                               mod.head.z_extended, mod.head.camera[1], mod.head.mass,
                                               mod.counterbalance))
    w("  clear %.2f mm; z safe %.1f, carrying %.1f; looks from %s mm"
      % (mod.clear, mod.z_safe, mod.z_carry, ", ".join("%.0f" % v for v in sorted(set(mod.look.values())))))
    for f in mod.fits:
        w("    %s %-70s %9.3f  %s" % ("ok  " if f.ok else "FAIL", f.name[:70], f.margin, f.detail))
    return "\n".join(out)
