"""The station module in MuJoCo, built from its derived StationModule.

EACH AXIS IS TWO BODIES.  A drive body, whose slide is the motor's shaft
expressed as travel (the drive train's inertia reflected onto it as the
joint's armature), and the carriage on it, whose slide is the drive's own
give: free inside the backlash, then the drive's stiffness, which for a
screw softens as the nut runs away from the motor.  Neither force is
MuJoCo's -- the backend (sim.py) computes both before every physics step:
the stepper's sine law on the drive, which drops a whole electrical cycle
(four full steps) when the lag passes two, and the backlash and stiffness
on the carriage.  Rail drag is a tendon's friction on the carriage's
absolute position.  So repeatability, lost steps and deflection under a
press come out of the physics, not out of a formula the check could only
agree with.

BELOW THE Z CARRIAGE, the 3-axis load cell is three stiff slides; the tool
plate hangs from them and every tool from the plate on its own air slide.
A counterbalance holds the Z assembly up (gravcomp on the Z carriage, sized
to everything below it); Z's preload, which sim.py applies as it does
every axis', pulls it up beyond that, so Z's motor carries the preload and
what a tool picked.  The camera is on the Z carriage, not the plate, so it
never weighs on the cell.

THE BUILD IS NOT THE DRAWING.  A Build draws, once per station, how far
each tool, the camera, each dock and the fixture lie from where the module
says, and the scene is built AS BUILT.  The controller only ever has the
drawing (the StationModule); commissioning finds the rest.

COLLISION BITS:
    1  HEAD    tool tips, jaws, pins, cup, collet: what touches the work
    2  WORK    region plates, the test pieces' fixed parts
    4  PART    things that move when touched: tokens, the nut, the plunger
Frame, carriages and tool bodies above the tips collide with nothing.
"""
from dataclasses import dataclass, field
from math import atan2, degrees, radians, pi, sqrt, cos, sin

import numpy as np

from truss.spec import mm
from ..spec import (LoadCell, HeadCam, Fiducial, Gripper2, Vacuum, Magnet, Pusher, Pogo, Spindle,
                    ToolSlide, Build, Artefact, Frame, Rail, Print)
from . import spec as SS

HEAD, WORK, PART = 1, 2, 4
C_FRAME = "0.55 0.56 0.60 0.35"
C_CARR = "0.40 0.42 0.46 1"
C_TOOL = "0.25 0.27 0.30 1"
C_PLATE = "0.93 0.93 0.90 1"
C_DARK = "0.05 0.05 0.06 1"
C_STEEL = "0.55 0.57 0.60 1"
C_PART = "0.85 0.55 0.20 1"


def _v(p):
    return "%.6f %.6f %.6f" % (mm(p[0]), mm(p[1]), mm(p[2]))


def box(name, centre, half, rgba, ctype=0, caff=0, mass=0.0, extra=""):
    """A box geom; mass in grams, and a geom drawn only for the eye weighs
    nothing (MuJoCo would otherwise give it water's density)."""
    m = ' mass="%.9g"' % (mass / 1000.0)
    return ('<geom name="%s" type="box" pos="%s" size="%.6f %.6f %.6f" rgba="%s" contype="%d" '
            'conaffinity="%d"%s%s/>' % (name, _v(centre), mm(half[0]), mm(half[1]), mm(half[2]), rgba,
                                        ctype, caff, m, extra))


def cyl(name, centre, r, half_h, rgba, ctype=0, caff=0, mass=0.0, extra=""):
    m = ' mass="%.9g"' % (mass / 1000.0)
    return ('<geom name="%s" type="cylinder" pos="%s" size="%.6f %.6f" rgba="%s" contype="%d" '
            'conaffinity="%d"%s%s/>' % (name, _v(centre), mm(r), mm(half_h), rgba, ctype, caff, m, extra))


# ================================================================ BUILD
@dataclass
class BuildDraw:
    """One built station's departures from its drawing (spec.Build, 1
    sigma).  Tools and camera are in the head frame; docks and fixture in
    the station frame; home switches along their axes."""
    tool: dict = field(default_factory=dict)      # kind -> (dx, dy, dz)
    cam: tuple = (0.0, 0.0, 0.0)                  # (dx, dy, yaw deg)
    cam_tilt: tuple = (0.0, 0.0)                  # deg about x, y
    region: dict = field(default_factory=dict)    # name -> (dx, dy, yaw deg)
    home: dict = field(default_factory=dict)      # axis -> mm

    @classmethod
    def draw(cls, mod, rng):
        b = cls()
        n = lambda s, k=None: rng.normal(0.0, s, k)
        for k in mod.head.tools:
            b.tool[k] = (float(n(Build.TOOL_XY)), float(n(Build.TOOL_XY)), float(n(Build.TOOL_Z)))
        b.cam = (float(n(Build.CAM_XY)), float(n(Build.CAM_XY)), float(n(Build.CAM_TILT)))
        b.cam_tilt = (float(n(Build.CAM_TILT)), float(n(Build.CAM_TILT)))
        for name in mod.regions:
            xy, yaw = SS.region_build(name)
            b.region[name] = (float(n(xy)), float(n(xy)), float(n(yaw)) if yaw else 0.0)
        # the pin is part of the fixture: it moves with it
        if "pin" in b.region and "fixture" in b.region:
            b.region["pin"] = b.region["fixture"]
        for a in SS.AXES:
            b.home[a] = float(n(Build.HOME))
        return b

    def region_pose(self, mod, name):
        """(R 2x2, t 2): station point as built = R p + t, for a point p of
        region `name` as drawn (rotation about the region's centre)."""
        dx, dy, yaw = self.region.get(name, (0.0, 0.0, 0.0))
        c = np.array(mod.regions[name].centre, float) if name in mod.regions else np.zeros(2)
        th = radians(yaw)
        R = np.array([[cos(th), -sin(th)], [sin(th), cos(th)]])
        return R, c - R @ c + np.array([dx, dy])

    def place(self, mod, name, p):
        """Where drawn point p (x, y[, z]) of region `name` was built."""
        R, t = self.region_pose(mod, name)
        q = R @ np.asarray(p[:2], float) + t
        return (float(q[0]), float(q[1])) + tuple(p[2:])


# ================================================================ PIECES
@dataclass(frozen=True)
class Piece:
    """A test piece on a region: drawn at `at` (x, y) on that region."""
    kind: str                  # pin | token | plunger | stud | pad
    name: str
    region: str
    at: tuple

    @property
    def size(self):
        """(x, y) footprint, drawn."""
        A = Artefact
        d = {"pin": A.PIN_D, "token": A.TOKEN_D + 2.0 * (Print.CLEAR + A.NEST_WALL),
             "plunger": A.PLUNGER_D, "stud": A.NUT_AF, "pad": A.PAD_W}[self.kind]
        return (d, d)


def test_plate(mod, pickers=None):
    """The check's test pieces, laid out from the module: the reference pin
    where from_line put it; a plunger, a stud and nut and a pad for the
    pogo block in a row along the fixture after it, each with a tool's
    keep-out round it; a token in its nest for each pick tool on the first
    dock its tool works on."""
    tools = tuple(mod.head.tools)
    pin = mod.regions["pin"]
    fx = mod.regions["fixture"]
    side = pin.size[0]
    out = [Piece("pin", "pin", "pin", pin.centre)]
    x = pin.centre[0] + side
    for kind, need in (("plunger", ("pusher", "gripper")), ("stud", ("spindle",)), ("pad", ("pogo",))):
        if not any(k in tools for k in need):
            continue
        out.append(Piece(kind, kind, "fixture", (x, pin.centre[1])))
        x += side
    if x - side / 2.0 > fx.hi[0]:
        raise ValueError("the test pieces need %.0f mm of fixture, it has %.0f" % (x - fx.lo[0], fx.size[0]))
    pickers = [k for k in (pickers or SS.PICKERS) if k in tools]
    for i, k in enumerate(pickers):
        dock = next(r for r in mod.spec.regions if k in r.reach and r.name not in ("fixture", "pin"))
        pitch = Artefact.TOKEN_D + 2.0 * SS.keepout(tools)
        lo = dock.lo
        out.append(Piece("token", "token_%s" % k, dock.name,
                         (lo[0] + pitch / 2.0 + i * pitch, dock.centre[1])))
    return out


# ================================================================ SCENE
def timestep(mod):
    """The largest step at which the explicit drive springs stay stable
    with room: omega dt <= 1 on the stiffest mode of any axis, anywhere in
    its travel, capped at 0.25 ms."""
    w = 0.0
    for ax in mod.axes.values():
        for q in (ax.lo, ax.hi):
            w = max(w, max(ax.modes(q)))
    return min(2.5e-4, 1.0 / w)


def _masses(mod):
    """g on each body, from the module's own budget: each carriage carries
    the next one down, so its body is the difference."""
    A = mod.axes
    lc = LoadCell.MASS / 2.0
    tools = sum(SS.TOOLS[k].MASS for k in mod.head.tools)
    plate = mod.head.mass - tools + lc
    z = A["z"].moving - mod.head.mass - LoadCell.MASS + lc
    y = A["y"].moving - A["z"].moving
    x = A["x"].moving - A["y"].moving
    return {"x": x, "y": y, "z": z, "plate": plate}


def preamble(dt, w, h):
    """Options, lights and defaults.  Every end stop (a tool slide's, a
    jaw's, a pogo pin's) and every mechanical coupling (the jaws' rack, a
    nut's thread) is as hard as the step lets a constraint be stable, a time
    constant of two steps: MuJoCo's default (20 ms) lets a 60 g tool on a
    40 N air slide sit 6 mm past its stroke, and let the driven jaw shove a
    token a millimetre across before the other caught up."""
    return """  <compiler angle="degree" autolimits="true"/>
  <option timestep="%g" integrator="implicitfast" cone="elliptic" impratio="3" gravity="0 0 -9.81"/>
  <visual>
    <headlight ambient="0.80 0.80 0.80" diffuse="0.35 0.35 0.35" specular="0.05 0.05 0.05"/>
    <quality shadowsize="2048"/>
    <map znear="0.002" zfar="30"/>
    <global offwidth="%d" offheight="%d"/>
  </visual>
  <default>
    <geom condim="3" friction="0.6 0.005 0.0001" solref="0.004 1" solimp="0.95 0.99 0.001"/>
    <joint solreflimit="%g 1"/>
    <equality solref="%g 1"/>
  </default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.80 0.80 0.78" rgb2="0.74 0.74 0.72"
             width="300" height="300"/>
    <material name="floor" texture="grid" texrepeat="20 20" reflectance="0.02"/>
  </asset>""" % (dt, w, h, 2.0 * dt, 2.0 * dt)


def scene(mod, build=None, pieces=(), render=(HeadCam.W, HeadCam.H)):
    """The station as built, as MJCF.  Returns (xml, info): info names what
    the backend needs that the XML cannot say -- the timestep, the body
    masses, which welds pair which tool with which part."""
    b = build if build is not None else BuildDraw()
    H = mod.head
    A = mod.axes
    dt = timestep(mod)
    m = _masses(mod)
    world, eq, act, ten, excl = [], [], [], [], []

    # ---------------------------------------------------------- floor
    x0, x1 = A["x"].lo - 300.0, A["x"].hi + 300.0
    y0, y1 = A["y"].lo - 300.0, A["y"].hi + 300.0
    world.append('<geom name="floor" type="plane" pos="%s" size="%.3f %.3f 0.1" material="floor" '
                 'contype="%d" conaffinity="%d"/>' % (_v(((x0 + x1) / 2, (y0 + y1) / 2, -1.0)),
                                                     mm(x1 - x0) / 2, mm(y1 - y0) / 2, WORK, PART))

    # -------------------------------------------------- regions, as built
    for name, r in mod.regions.items():
        if name == "pin":
            continue
        R, t = b.region_pose(mod, name)
        c = R @ np.array(r.centre) + t
        yaw = degrees(atan2(R[1, 0], R[0, 0]))
        world.append('<body name="reg_%s" pos="%s" euler="0 0 %.6f">' % (name, _v((c[0], c[1], r.top)), yaw))
        world.append('  ' + box("reg_%s" % name, (0, 0, -2.5), (r.size[0] / 2, r.size[1] / 2, 2.5), C_PLATE,
                                WORK, PART))
        for i, p in enumerate(r.fiducial_points()):
            q = np.array(p[:2]) - np.array(r.centre)
            world.append('  ' + cyl("fid_%s_%d" % (name, i), (q[0], q[1], 0.025), Fiducial.D / 2.0, 0.025,
                                    C_DARK))
        world.append('</body>')

    # -------------------------------------------------- the test pieces
    info = {"dt": dt, "masses": m, "tokens": [], "welds": {}, "pieces": {}}
    for pc in pieces:
        r = mod.regions[pc.region]
        p = b.place(mod, pc.region, pc.at)
        top = r.top
        info["pieces"][pc.name] = (pc.kind, (p[0], p[1], top))
        A_ = Artefact
        if pc.kind == "pin":
            world.append(cyl("pin", (p[0], p[1], top + A_.PIN_H / 2.0), A_.PIN_D / 2.0, A_.PIN_H / 2.0,
                             C_DARK, WORK, HEAD | PART))
        elif pc.kind == "pad":
            world.append(box("pad", (p[0], p[1], top + A_.PAD_T / 2.0), (A_.PAD_W / 2.0, A_.PAD_W / 2.0,
                                                                          A_.PAD_T / 2.0),
                             "0.75 0.62 0.25 1", WORK, HEAD))
        elif pc.kind == "plunger":
            world += ['<body name="plunger" pos="%s">' % _v((p[0], p[1], top + A_.PLUNGER_TRAVEL + 5.0)),
                      '  <joint name="plunger" type="slide" axis="0 0 1" range="%.6f 0" stiffness="%.6f" '
                      'damping="%.6f"/>' % (mm(-A_.PLUNGER_TRAVEL), A_.PLUNGER_K * 1e3, 2.0),
                      '  ' + cyl("plunger_cap", (0, 0, 0), A_.PLUNGER_D / 2.0, 5.0, C_STEEL, PART, HEAD,
                                 mass=20.0),
                      '</body>',
                      box("plunger_base", (p[0], p[1], top + A_.PLUNGER_TRAVEL / 2.0),
                          (A_.PLUNGER_D / 2.0 + 3.0, A_.PLUNGER_D / 2.0 + 3.0, A_.PLUNGER_TRAVEL / 2.0),
                          C_FRAME)]
        elif pc.kind == "stud":
            h = A_.STUD_L + A_.NUT_H
            world += [cyl("stud", (p[0], p[1], top + h / 2.0), 5.0, h / 2.0, C_STEEL),
                      '<body name="nut" pos="%s">' % _v((p[0], p[1], top + A_.STUD_L + A_.NUT_H / 2.0)),
                      '  <joint name="nut_z" type="slide" axis="0 0 1" range="%.6f 0" damping="0.5"/>'
                      % mm(-A_.STUD_L),
                      '  <joint name="nut_w" type="hinge" axis="0 0 1" damping="0.0005"/>',
                      '  ' + cyl("nut", (0, 0, 0), A_.NUT_AF / 2.0, A_.NUT_H / 2.0, C_STEEL, PART, HEAD,
                                 mass=8.0),
                      '  ' + box("nut_mark", (A_.NUT_AF / 2.0 - 2.0, 0, A_.NUT_H / 2.0 + 0.05), (1.5, 0.8, 0.05),
                                 C_DARK),
                      '</body>']
            # a right-hand thread: turning it clockwise seen from above
            # (negative about +z) runs it down, lead per turn
            eq.append('<joint name="thread" joint1="nut_z" joint2="nut_w" polycoef="0 %.9f 0 0 0"/>'
                      % (mm(A_.STUD_LEAD) / (2.0 * pi)))
            # and it is self-locking: nothing but a turn moves the nut, so it
            # stays where it was left -- a weld to the world, which sim.py
            # lets go while the collet holds it.  MuJoCo's dry friction on a
            # turning joint this light, behind a thread, did not hold it: a
            # 0.01 rad/s nudge spun it up to the 38 rad/s where the friction
            # equals the damping, and it ran 3.6 mm down unturned
            eq.append('<weld name="nut_lock" body1="nut" solref="0.004 1" solimp="0.98 0.999 0.001"/>')
        elif pc.kind == "token":
            wall, cl = A_.NEST_WALL, Print.CLEAR
            half_in = A_.TOKEN_D / 2.0 + cl
            for k, (dx, dy, hx, hy) in enumerate(((half_in + wall / 2, 0, wall / 2, half_in + wall),
                                                   (-half_in - wall / 2, 0, wall / 2, half_in + wall),
                                                   (0, half_in + wall / 2, half_in, wall / 2),
                                                   (0, -half_in - wall / 2, half_in, wall / 2))):
                world.append(box("%s_nest%d" % (pc.name, k), (p[0] + dx, p[1] + dy, top + A_.TOKEN_H / 4.0),
                                 (hx, hy, A_.TOKEN_H / 4.0), "0.30 0.55 0.75 1", WORK, PART))
            mass = A_.TOKEN_RHO * pi / 4.0 * A_.TOKEN_D ** 2 * A_.TOKEN_H
            world += ['<body name="%s" pos="%s">' % (pc.name, _v((p[0], p[1], top + A_.TOKEN_H / 2.0 + 0.01))),
                      '  <freejoint name="%s"/>' % pc.name,
                      '  ' + cyl(pc.name, (0, 0, 0), A_.TOKEN_D / 2.0, A_.TOKEN_H / 2.0, C_STEEL, PART,
                                 HEAD | WORK | PART, mass=mass),
                      '  ' + cyl(pc.name + "_fid", (0, 0, A_.TOKEN_H / 2.0 + 0.025), Fiducial.D, 0.025, C_DARK),
                      '</body>']
            info["tokens"].append(pc.name)

    # ------------------------------------------------------ the frame
    # two X rails on posts, one each side of the travel, high enough for
    # the bridge to carry Y and Z over everything: drawn, not collided
    rail_z = A["z"].hi + _z_stack(mod)
    ry0, ry1 = A["y"].lo - 60.0, A["y"].hi + H.camera[1] + 60.0
    for i, yy in enumerate((ry0, ry1)):
        world.append(box("xrail%d" % i, ((A["x"].lo + A["x"].hi) / 2.0, yy, rail_z),
                         ((A["x"].hi - A["x"].lo) / 2.0 + 60.0, Frame.BEAM / 2.0, Frame.BEAM / 2.0), C_FRAME))
        for j, xx in enumerate((A["x"].lo - 60.0, A["x"].hi + 60.0)):
            world.append(box("post%d%d" % (i, j), (xx, yy, rail_z / 2.0), (Frame.BEAM / 2, Frame.BEAM / 2,
                                                                            rail_z / 2.0), C_FRAME))

    # ------------------------------------------------------ the axes
    # the head point at home: x and y at their low ends, z at its top
    home = (A["x"].home, A["y"].home, A["z"].home)
    # a review camera riding the Z carriage: the tool row from the side, a
    # little above, far enough back that the whole row fits the narrower
    # (vertical) field with a tool's slot to spare -- every tip, retracted
    # and extended, and the part under it, wherever the head goes
    fovy, down = 45.0, radians(30.0)       # a view, not a machine: what reads well
    reach = (H.plate[1] / 2.0 + max(t.slot for t in H.tools.values())) / np.tan(radians(fovy / 2.0))
    aim = np.array([0.0, 0.0, H.z_extended])
    tip_cam = ('<camera name="tip" pos="%s" xyaxes="0 -1 0 %.6f 0 %.6f" fovy="%.1f"/>'
               % (_v(aim + reach * np.array([-cos(down), 0.0, sin(down)])), sin(down), cos(down), fovy))
    body = []
    rng_ = lambda a: (mm(A[a].lo - A[a].home - SS.home_zone()), mm(A[a].hi - A[a].home + SS.home_zone()))
    dz = _z_stack(mod)
    body += [
        '<body name="x_drive" pos="%s">' % _v((home[0], 0.0, 0.0)),
        '  <joint name="x_drive" type="slide" axis="1 0 0" range="%.6f %.6f" armature="%.6f"/>'
        % (rng_("x") + (A["x"].drive.reflected() * 1e-3,)),
        '  <geom name="x_nut" type="sphere" size="0.005" mass="0.001" contype="0" conaffinity="0" rgba="0 0 0 0"/>',
        '  <body name="x" pos="0 0 0">',
        '    <joint name="x_play" type="slide" axis="1 0 0"/>',
        '    ' + box("bridge", (0.0, (ry0 + ry1) / 2.0, rail_z), (Frame.BEAM, (ry1 - ry0) / 2.0, Frame.BEAM / 2.0),
                     C_CARR, mass=m["x"]),
        '    <body name="y_drive" pos="%s">' % _v((0.0, home[1], 0.0)),
        '      <joint name="y_drive" type="slide" axis="0 1 0" range="%.6f %.6f" armature="%.6f"/>'
        % (rng_("y") + (A["y"].drive.reflected() * 1e-3,)),
        '      <geom name="y_nut" type="sphere" size="0.005" mass="0.001" contype="0" conaffinity="0" rgba="0 0 0 0"/>',
        '      <body name="y" pos="0 0 0">',
        '        <joint name="y_play" type="slide" axis="0 1 0"/>',
        '        ' + box("ycar", (0.0, H.camera[1] / 2.0, rail_z - Frame.BEAM), (H.plate[0] / 2.0 + Frame.BEAM,
                                                                              H.plate[1] / 2.0 + H.camera[1] / 2.0
                                                                              + 15.0, 6.0), C_CARR, mass=m["y"]),
        '        <body name="z_drive" pos="%s">' % _v((0.0, 0.0, home[2])),
        '          <joint name="z_drive" type="slide" axis="0 0 1" range="%.6f %.6f" armature="%.6f"/>'
        % (rng_("z") + (A["z"].drive.reflected() * 1e-3,)),
        '          <geom name="z_nut" type="sphere" size="0.005" mass="0.001" contype="0" conaffinity="0" '
        'rgba="0 0 0 0"/>',
        '          <body name="z" pos="0 0 0" gravcomp="%.6f">' % _gravcomp(mod, m),
        '            <joint name="z_play" type="slide" axis="0 0 1"/>',
        '            ' + box("zplate", (0.0, H.camera[1] / 2.0, LoadCell.SIZE[2] + Frame.PLATE_T + dz / 2.0),
                             (Frame.PLATE_T / 2.0, H.plate[1] / 2.0 + H.camera[1] / 2.0, dz / 2.0), C_CARR,
                             mass=m["z"] - HeadCam.MASS),
        '            ' + tip_cam,
    ]
    # the camera, as built: on the Z carriage, looking straight down
    cx, cy, yaw = b.cam
    tx, ty = b.cam_tilt
    cam = np.array(H.camera) + np.array([cx, cy, 0.0])
    Rm = _rot_z(yaw) @ _rot_x(tx) @ _rot_y(ty)
    xa, ya = Rm @ np.array([1.0, 0.0, 0.0]), Rm @ np.array([0.0, 1.0, 0.0])
    fovy = 2.0 * degrees(atan2(HeadCam.H * HeadCam.PIXEL / 2.0, HeadCam.F))
    body += [
        '            ' + box("cam_board", (cam[0], cam[1], cam[2] + HeadCam.LENS_L), (HeadCam.BOARD / 2.0,
                                                                                     HeadCam.BOARD / 2.0, 1.0),
                             "0.10 0.35 0.15 1", mass=HeadCam.MASS * 0.7),
        '            ' + cyl("cam_lens", (cam[0], cam[1], cam[2] + HeadCam.LENS_L / 2.0), 7.0,
                             HeadCam.LENS_L / 2.0, C_DARK, mass=HeadCam.MASS * 0.3),
        '            <camera name="head_cam" pos="%s" xyaxes="%.9f %.9f %.9f %.9f %.9f %.9f" fovy="%.6f"/>'
        % ((_v(cam),) + tuple(xa) + tuple(ya) + (fovy,)),
        '            <site name="cam" pos="%s" size="0.001"/>' % _v(cam),
    ]
    # the load cell: three stiff slides, the tool plate on the last
    k_lc = LoadCell.RATED / LoadCell.DEFLECT * 1e3          # N/m
    m_below = (m["plate"] + sum(SS.TOOLS[k].MASS for k in H.tools)) * 1e-3
    c_lc = 2.0 * LoadCell.ZETA * sqrt(k_lc * m_below)
    lc_top = LoadCell.SIZE[2] + Frame.PLATE_T
    body += [
        '            ' + box("lc_body", (0.0, 0.0, Frame.PLATE_T + LoadCell.SIZE[2] / 2.0),
                             tuple(s / 2.0 for s in LoadCell.SIZE), "0.70 0.70 0.20 1", mass=0.001),
        '            <body name="lc_x" pos="0 0 0">',
        '              <joint name="lc_x" type="slide" axis="1 0 0" stiffness="%.6f" damping="%.6f"/>' % (k_lc, c_lc),
        '              <geom name="lc_x" type="sphere" size="0.002" mass="0.001" contype="0" conaffinity="0" '
        'rgba="0 0 0 0"/>',
        '              <body name="lc_y" pos="0 0 0">',
        '                <joint name="lc_y" type="slide" axis="0 1 0" stiffness="%.6f" damping="%.6f"/>' % (k_lc, c_lc),
        '                <geom name="lc_y" type="sphere" size="0.002" mass="0.001" contype="0" conaffinity="0" '
        'rgba="0 0 0 0"/>',
        '                <body name="plate" pos="0 0 0">',
        '                  <joint name="lc_z" type="slide" axis="0 0 1" stiffness="%.6f" damping="%.6f"/>' % (k_lc, c_lc),
        '                  ' + box("plate", (0.0, 0.0, Frame.PLATE_T / 2.0), (H.plate[0] / 2.0, H.plate[1] / 2.0,
                                                                            Frame.PLATE_T / 2.0), C_CARR,
                                   mass=m["plate"]),
        '                  <site name="head" pos="0 0 0" size="0.001"/>',
    ]
    # each tool: a spacer and an air slide under the plate, the tool on
    # the slide's carriage, its tip site where the drawing (plus the
    # build's error) puts it
    welds = {}
    for k, tm in H.tools.items():
        e = b.tool.get(k, (0.0, 0.0, 0.0))
        tip = np.array(tm.tip) + np.array(e)
        top = tip[2] + SS.tool_height(k)                   # the tool's top, on its slide's carriage
        cslide = (tip[0], tip[1], top + ToolSlide.BODY[2] / 2.0)
        T = SS.TOOLS[k]
        body += [
            '                  ' + box("slide_%s" % k, cslide, tuple(s / 2.0 for s in ToolSlide.BODY), C_CARR),
            '                  ' + box("spacer_%s" % k, (tip[0], tip[1], (top + ToolSlide.BODY[2]) / 2.0),
                                       (ToolSlide.BODY[0] / 2.0, ToolSlide.BODY[1] / 2.0,
                                        max(-(top + ToolSlide.BODY[2]) / 2.0, 0.5)), C_CARR),
            '                  <body name="tool_%s" pos="0 0 0">' % k,
            '                    <joint name="slide_%s" type="slide" axis="0 0 -1" range="0 %.6f" damping="%.6f"/>'
            % (k, mm(ToolSlide.STROKE), ToolSlide.FORCE * ToolSlide.TIME_S / mm(ToolSlide.STROKE)),
            '                    <site name="tip_%s" pos="%s" size="0.001"/>' % (k, _v(tip)),
        ]
        body += ['                    ' + g for g in _tool(k, tip, T, info)]
        body.append('                  </body>')
        act.append('<motor name="slide_%s" joint="slide_%s" gear="1" ctrlrange="%.3f %.3f"/>'
                   % (k, k, -ToolSlide.FORCE, ToolSlide.FORCE))
        if k == "gripper":
            eq.append('<joint name="jaws" joint1="jaw_l" joint2="jaw_r" polycoef="0 1 0 0 0"/>')
            act.append('<position name="jaws" joint="jaw_l" kp="%.3f" kv="%.3f" forcerange="%.3f %.3f"/>'
                       % (2.0e4, 40.0, -Gripper2.FORCE, Gripper2.FORCE))
        if k == "spindle":
            act.append('<velocity name="spindle" joint="spindle" kv="%.6f" forcerange="%.4f %.4f"/>'
                       % (0.05, -Spindle.TORQUE, Spindle.TORQUE))
    body += ['                </body>', '              </body>', '            </body>',
             '          </body>', '        </body>', '      </body>', '    </body>', '  </body>', '</body>']
    world += body

    # welds: each tool that takes hold, to each thing it may take
    holders = {"gripper": "jaw_l", "vacuum": "tool_vacuum", "magnet": "tool_magnet", "spindle": "collet"}
    parts = list(info["tokens"]) + (["nut"] if any(pc.kind == "stud" for pc in pieces) else [])
    for k, b1 in holders.items():
        if k not in H.tools:
            continue
        for p in parts:
            name = "hold_%s_%s" % (k, p)
            eq.append('<weld name="%s" body1="%s" body2="%s" active="false" solref="0.004 1" '
                      'solimp="0.98 0.999 0.001"/>' % (name, b1, p))
            welds[(k, p)] = name
    info["welds"] = welds

    for a in SS.AXES:
        act.append('<motor name="%s_motor" joint="%s_drive" gear="1" ctrlrange="-1e6 1e6"/>' % (a, a))
        ten.append('<fixed name="%s_abs" frictionloss="%.6f"><joint joint="%s_drive" coef="1"/>'
                   '<joint joint="%s_play" coef="1"/></fixed>' % (a, A[a].drag, a, a))
    # what MuJoCo does not exclude on its own: a grandparent's geoms
    excl += ['<exclude body1="z" body2="plate"/>', '<exclude body1="z" body2="lc_y"/>']
    for k in H.tools:
        excl.append('<exclude body1="plate" body2="tool_%s"/>' % k)
        excl.append('<exclude body1="z" body2="tool_%s"/>' % k)

    # review cameras: the whole station from the aisle (the tool row's own
    # rides the Z carriage, above)
    span = max(A["x"].hi - A["x"].lo, 600.0)
    cams = ['<camera name="overview" pos="%s" xyaxes="1 0 0 0 0.55 0.835"/>'
            % _v(((A["x"].lo + A["x"].hi) / 2.0, A["y"].lo - 0.85 * span, 0.75 * span)),
            ]
    xml = """<mujoco model="station_%s">
%s
  <worldbody>
    %s
%s
  </worldbody>
  <contact>
%s
  </contact>
  <equality>
%s
  </equality>
  <tendon>
%s
  </tendon>
  <actuator>
%s
  </actuator>
</mujoco>
""" % (mod.spec.name, preamble(dt, render[0], render[1]), "\n    ".join(cams),
       "\n".join("    " + w for w in world),
       "\n".join("    " + e for e in excl), "\n".join("    " + e for e in eq),
       "\n".join("    " + t for t in ten), "\n".join("    " + a for a in act))
    return xml, info


def _z_stack(mod):
    """mm from the head point up to the Y carriage: the Z plate's height."""
    return mod.axes["z"].travel + 2.0 * Rail.BLOCK_L + LoadCell.SIZE[2] + Frame.PLATE_T


def _gravcomp(mod, m):
    """The counterbalance: the whole Z assembly's weight, carried by the Z
    carriage, which MuJoCo lets a body do with gravcomp above 1."""
    below = m["plate"] + sum(SS.TOOLS[k].MASS for k in mod.head.tools) + 0.002e3
    return (m["z"] + below) / m["z"]


def _rot_x(deg):
    t = radians(deg)
    return np.array([[1, 0, 0], [0, cos(t), -sin(t)], [0, sin(t), cos(t)]])


def _rot_y(deg):
    t = radians(deg)
    return np.array([[cos(t), 0, sin(t)], [0, 1, 0], [-sin(t), 0, cos(t)]])


def _rot_z(deg):
    t = radians(deg)
    return np.array([[cos(t), -sin(t), 0], [sin(t), cos(t), 0], [0, 0, 1]])


def _tool(k, tip, T, info):
    """A tool's geoms (in the plate's frame, on its slide's body), its
    tip at `tip`.  Only what touches the work collides."""
    x, y, z = tip
    h = SS.tool_height(k)
    if k == "gripper":
        jaw = T.JAW
        out = [box("g_body", (x, y, z + jaw[2] + T.BODY[2] / 2.0), tuple(s / 2.0 for s in T.BODY), C_TOOL,
                   mass=T.MASS - 20.0)]
        for s, tag in ((1.0, "l"), (-1.0, "r")):
            yy = y + s * (T.STROKE + jaw[1] / 2.0)
            out += ['<body name="jaw_%s" pos="%s">' % (tag, _v((x, yy, z + jaw[2] / 2.0))),
                    '  <joint name="jaw_%s" type="slide" axis="0 %d 0" range="0 %.6f" damping="2"/>'
                    % (tag, int(-s), mm(T.STROKE)),
                    '  ' + box("jaw_%s" % tag, (0, 0, 0), (jaw[0] / 2.0, jaw[1] / 2.0, jaw[2] / 2.0), C_DARK, HEAD,
                               WORK | PART, mass=10.0, extra=' friction="1.0 0.005 0.0001"'),
                    '</body>']
        return out
    if k == "vacuum":
        return [box("v_body", (x, y, z + h / 2.0 + 4.0), (T.BODY[0] / 2.0, T.BODY[1] / 2.0, h / 2.0 - 4.0), C_TOOL,
                    mass=T.MASS - 5.0),
                cyl("v_cup", (x, y, z + 2.0), T.CUP_D / 2.0, 2.0, "0.15 0.15 0.15 1", HEAD, WORK | PART, mass=5.0)]
    if k == "magnet":
        return [box("m_body", (x, y, z + h / 2.0), tuple(s / 2.0 for s in T.BODY), "0.60 0.15 0.12 1", HEAD,
                    WORK | PART, mass=T.MASS)]
    if k == "pusher":
        return [box("p_body", (x, y, z + h / 2.0), tuple(s / 2.0 for s in T.BODY), "0.20 0.40 0.70 1", HEAD,
                    WORK | PART, mass=T.MASS)]
    if k == "pogo":
        out = [box("pogo_body", (x, y, z + T.TRAVEL + T.BODY[2] / 2.0), tuple(s / 2.0 for s in T.BODY),
                   "0.15 0.45 0.25 1", HEAD, WORK | PART, mass=T.MASS - T.N * 0.5)]
        k_pin = T.FORCE / ((T.WORK[0] + T.WORK[1]) / 2.0) * 1e3
        for i in range(T.N):
            px = x + (i - (T.N - 1) / 2.0) * T.PITCH
            out += ['<body name="pogo%d" pos="%s">' % (i, _v((px, y, z + T.TRAVEL / 2.0))),
                    '  <joint name="pogo%d" type="slide" axis="0 0 1" range="0 %.6f" stiffness="%.6f" '
                    'damping="0.02"/>' % (i, mm(T.TRAVEL), k_pin),
                    '  ' + cyl("pogo%d" % i, (0, 0, 0), T.TIP_D / 2.0, T.TRAVEL / 2.0, "0.85 0.70 0.20 1", HEAD,
                               WORK, mass=0.5),
                    '</body>']
        return out
    if k == "spindle":
        return [box("s_body", (x, y, z + h / 2.0 + 10.0), (T.BODY[0] / 2.0, T.BODY[1] / 2.0, h / 2.0 - 10.0),
                    C_TOOL, mass=T.MASS - 30.0),
                '<body name="collet" pos="%s">' % _v((x, y, z + 5.0)),
                '  <joint name="spindle" type="hinge" axis="0 0 1" damping="0.0002" armature="0.0001"/>',
                '  ' + cyl("collet", (0, 0, 0), T.COLLET_D / 2.0, 5.0, "0.75 0.75 0.78 1", HEAD, WORK | PART,
                           mass=30.0),
                '  ' + box("collet_mark", (T.COLLET_D / 2.0 - 2.0, 0, -5.05), (1.5, 0.8, 0.05), C_DARK),
                '</body>']
    raise ValueError(k)
