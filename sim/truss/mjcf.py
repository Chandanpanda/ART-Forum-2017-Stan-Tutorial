"""Generate the cell's MJCF from a truss, its geometry and its fixture.

Nothing here is hand-placed: every rod, pin, cradle, slot and tool sits
where geometry.py and fixture.py computed it from the spec, so the model
cannot drift from the parameters and a new truss is a new model with no
editing.  Self-contained scene strings (no <include>) so each opens in
MuJoCo's simulate on its own.

WORLD FRAME = TRUSS FRAME: x along the chord from its start, z up, the
cage's axis on the x axis.  The head's tool point is the RING CENTRE;
the carriage joints read directly in that frame, which is what the plan
and the HAL use.

COLLISION BITS (contype / conaffinity):
    1  rods         collide with everything that matters
    2  fixture      pins, cradles, spine, plates, posts, rack slots
    4  head         ring, spool, tools, carriage box
    8  drops
The head collides with rods and fixture on purpose: a crash the solver
did not predict is a contact this model reports.
"""
from math import pi, cos, sin, radians, degrees, atan2, sqrt

import numpy as np

from .spec import (mm, Truss, Ring, Gantry, Head, Gripper, Cage, Magazine,
                   Dispenser, Vision, ring_r_in, ring_r_out, rail_r_out)
from . import band as _band

C_ROD, C_CAGE, C_HEAD = "0.12 0.12 0.13 1", "0.55 0.58 0.62 1", "0.85 0.45 0.15 1"
C_RING, C_RACK, C_BAND = "0.20 0.55 0.85 1", "0.70 0.66 0.55 1", "0.90 0.80 0.20 0.0"
C_DROP = "0.95 0.90 0.35 0.9"

# collision masks
ROD, CAGE, HEAD_B, DROP = 1, 2, 4, 8


def _v(p):
    """mm vector -> metres string."""
    return "%.6f %.6f %.6f" % (mm(p[0]), mm(p[1]), mm(p[2]))


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _perp(u, prefer):
    """A unit vector perpendicular to u, as close to `prefer` as possible."""
    u = _unit(u)
    p = np.asarray(prefer, float)
    p = p - (p @ u) * u
    if np.linalg.norm(p) < 1e-9:
        p = np.cross(u, [1.0, 0.0, 0.0])
        if np.linalg.norm(p) < 1e-9:
            p = np.cross(u, [0.0, 1.0, 0.0])
    return _unit(p)


def box(name, centre, size, xax, yax, rgba, ctype, caff, mass=None, extra=""):
    """A box at `centre` (mm) with half-sizes `size` (mm), x along xax, y
    along yax (MuJoCo derives z), collision masks as given."""
    m = ' mass="%.6f"' % (mass / 1000.0) if mass is not None else ""
    return ('<geom name="%s" type="box" pos="%s" size="%.6f %.6f %.6f" '
            'xyaxes="%.6f %.6f %.6f %.6f %.6f %.6f" rgba="%s" contype="%d" '
            'conaffinity="%d"%s%s/>'
            % (name, _v(centre), mm(size[0]), mm(size[1]), mm(size[2]),
               xax[0], xax[1], xax[2], yax[0], yax[1], yax[2], rgba, ctype, caff,
               m, extra))


def v_flanks(prefix, apex, axis, up, length, depth, thick, half_angle_deg, rgba,
             ctype, caff):
    """Two boxes forming a V-groove: apex line along `axis` through `apex`,
    opening along `up`, flanks at +-half_angle from `up`."""
    u = _unit(axis)
    up = np.asarray(up, float)
    up = _unit(up - (up @ u) * u)
    t = np.cross(u, up)
    b = radians(half_angle_deg)
    h = depth / cos(b)                      # flank length
    out = []
    for s, tag in ((1.0, "a"), (-1.0, "b")):
        w = sin(b) * s * t + cos(b) * up     # along the flank, from the apex
        n = np.cross(u, w)
        # THE FLANK'S INNER FACE MEETS THE APEX, not its centre plane: a
        # box centred on the apex line stands half its thickness proud,
        # and a rod then seats 0.7 mm high in every V -- measured, the
        # first thing the built model got wrong.  Shift each flank away
        # from the groove's interior by half its thickness.
        outward = -np.sign(float(n @ up)) if abs(float(n @ up)) > 1e-9 else 1.0
        c = np.asarray(apex, float) + w * (h / 2.0) + n * (outward * thick / 2.0)
        out.append(box("%s_%s" % (prefix, tag), c, (length / 2.0, h / 2.0, thick / 2.0),
                       u, w, rgba, ctype, caff))
    return out


def capsule(name, p0, p1, r, rgba, ctype, caff, mass=None, cls=""):
    m = ' mass="%.6f"' % (mass / 1000.0) if mass is not None else ""
    return ('<geom name="%s" type="capsule" fromto="%s %s" size="%.6f" rgba="%s" '
            'contype="%d" conaffinity="%d"%s%s/>'
            % (name, _v(p0), _v(p1), mm(r), rgba, ctype, caff, m, cls))


def cylinder(name, p0, p1, r, rgba, ctype, caff, mass=None, extra=""):
    m = ' mass="%.6f"' % (mass / 1000.0) if mass is not None else ""
    return ('<geom name="%s" type="cylinder" fromto="%s %s" size="%.6f" rgba="%s" '
            'contype="%d" conaffinity="%d"%s%s/>'
            % (name, _v(p0), _v(p1), mm(r), rgba, ctype, caff, m, extra))


# ---------------------------------------------------------------- preamble
def preamble(timestep):
    return """  <compiler angle="degree" autolimits="true"/>
  <option timestep="%g" integrator="implicitfast" cone="elliptic"
          impratio="3" gravity="0 0 -9.81"/>
  <visual>
    <!-- lit like a vision cell, flat and bright: matte black rods against
         a light fixture from every angle.  Measured: with a strong
         directional component the cage's shaded faces rendered as dark as
         the chord, and no grey threshold separated them. -->
    <headlight ambient="0.85 0.85 0.85" diffuse="0.35 0.35 0.35" specular="0.05 0.05 0.05"/>
    <quality shadowsize="2048"/>
    <map znear="0.005" zfar="20"/>
    <global offwidth="%d" offheight="%d"/>
  </visual>
  <default>
    <geom condim="3" friction="0.5 0.005 0.0001" solref="0.004 1"
          solimp="0.95 0.99 0.001"/>
    <default class="rod">
      <geom condim="3" friction="0.4 0.005 0.0001" solref="0.004 1"
            solimp="0.95 0.99 0.001"/>
    </default>
  </default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.94 0.94 0.92"
             rgb2="0.88 0.88 0.86" width="300" height="300"/>
    <material name="floor" texture="grid" texrepeat="8 8" reflectance="0.05"/>
  </asset>""" % (timestep, Vision.W, Vision.H)


# ------------------------------------------------------------------ cage
def cage_body(geom, fixture):
    """The rotating fixture as one body on a hinge about x."""
    t = geom.t
    out = ['<body name="cage" pos="0 0 0">',
           '  <joint name="cage" type="hinge" axis="1 0 0" damping="0.05" '
           'armature="1e-5"/>']
    x0, x1 = -(Cage.END_FREE + Cage.END_PLATE_T), t.length + Cage.END_FREE + Cage.END_PLATE_T
    out.append("  " + cylinder("spine", (x0, 0, 0), (x1, 0, 0), fixture.spine_r(), C_CAGE,
                               CAGE, ROD | HEAD_B, mass=80.0))
    plate_r = fixture.plate_r()
    for tag, xa in (("p0", x0), ("p1", t.length + Cage.END_FREE)):
        out.append("  " + cylinder("plate_%s" % tag, (xa, 0, 0), (xa + Cage.END_PLATE_T, 0, 0),
                                   plate_r, C_CAGE, CAGE, ROD | HEAD_B, mass=40.0))
    # pins: an arm from the spine out to the notch, then the V
    for i, p in enumerate(fixture.pins):
        root = np.array([p.x, 0.0, 0.0]) + p.up * (fixture.spine_r() - 1.0)
        arm_c = (root + p.apex) / 2.0
        L = float(np.linalg.norm(p.apex - root))
        out.append("  " + box("arm%d" % i, arm_c, (L / 2.0, Cage.ARM_W / 2.0, Cage.PIN_T / 2.0),
                              p.up, np.cross(np.array([1.0, 0, 0]), p.up), C_CAGE,
                              CAGE, ROD | HEAD_B, mass=1.5))
        out += ["  " + s for s in v_flanks("pin%d" % i, p.apex, (1.0, 0.0, 0.0), p.up,
                                           Cage.PIN_T, Cage.NOTCH_DEPTH, 0.8, Cage.NOTCH_ANGLE / 2.0,
                                           C_CAGE, CAGE, ROD | HEAD_B)]
    # cradles: a V along the diagonal, opening along the face's normal
    for i, c in enumerate(fixture.cradles):
        out += ["  " + s for s in v_flanks("cradle%d" % i, c.apex, c.axis, c.up,
                                           Cage.CRADLE_L, Cage.NOTCH_DEPTH * 0.8, Cage.CRADLE_T,
                                           Cage.CRADLE_ANGLE / 2.0, C_CAGE, CAGE, ROD | HEAD_B)]
        # a stem back to the spine's neighbourhood so the cradle is held
        stem0 = c.apex - c.up * 2.0
        stem1 = c.apex - c.up * 14.0
        out.append("  " + capsule("cstem%d" % i, stem0, stem1, 1.5, C_CAGE, CAGE, ROD | HEAD_B))
    for i, q in enumerate(fixture.posts):
        out.append("  " + cylinder("post%d" % i, q.p0, q.p1, Cage.POST_R, C_CAGE, CAGE,
                                   ROD | HEAD_B))
    # band sleeves: visual only, grown as the turns are counted
    for j in geom.joints:
        c = geom.chord_point(j.chord, j.x)
        r = geom.cluster_reach(j, 0.0) + t.thread_d
        out.append('  <geom name="band%d" type="cylinder" fromto="%s %s" size="%.6f" '
                   'rgba="%s" contype="0" conaffinity="0" mass="0"/>'
                   % (j.index, _v(c - np.array([0.05, 0, 0])), _v(c + np.array([0.05, 0, 0])),
                      mm(r), C_BAND))
    out.append("</body>")
    return out


# ------------------------------------------------------------------ rods
def rod_body(geom, rod, p0, p1):
    """A free rod at world endpoints p0, p1 (its centreline)."""
    t = geom.t
    if rod.kind == "chord":
        g = capsule("rod%d_g" % rod.index, p0 - (p0 + p1) / 2.0, p1 - (p0 + p1) / 2.0,
                    rod.r, C_ROD, ROD, ROD | CAGE | HEAD_B | DROP,
                    mass=t.mass_chords / t.n_chords, cls=' class="rod"')
    else:
        g = cylinder("rod%d_g" % rod.index, p0 - (p0 + p1) / 2.0, p1 - (p0 + p1) / 2.0,
                     rod.r, C_ROD, ROD, ROD | CAGE | HEAD_B | DROP,
                     mass=t.mass_diags / max(t.n_diag, 1), extra=' class="rod"')
    c = (p0 + p1) / 2.0
    return ('<body name="rod%d" pos="%s"><freejoint name="rod%d_f"/>%s</body>'
            % (rod.index, _v(c), rod.index, g))


def rods_in_racks(geom, fixture):
    out = []
    for rod in geom.rods:
        s = fixture.slot_of(rod.index)
        out.append(rod_body(geom, rod, s.p0, s.p1))
    return out


def rods_in_fixture(geom):
    out = []
    for rod in geom.rods:
        p0, p1 = (rod.p0, rod.p1) if rod.kind == "chord" else geom.diag_body_ends(rod)
        out.append(rod_body(geom, rod, p0, p1))
    return out


# ----------------------------------------------------------------- racks
def rack_geoms(geom, fixture):
    """A rod rests on TWO SHORT V-BLOCKS near its ends, and the middle is
    free: a V long enough to be a gutter buries a 1 mm rod below its own
    flanks, and no pad can get beside it (measured: jaws stopped 1.7 mm
    from a diagonal, on the flank).  The gripper takes every rod at its
    midpoint."""
    out = []
    drop_c = (geom.t.d_chord / 2.0) / sin(radians(Magazine.SLOT_ANGLE / 2.0))
    drop_d = (geom.t.d_diag / 2.0) / sin(radians(Magazine.SLOT_ANGLE / 2.0))
    for s in fixture.slots:
        rod = geom.rods[s.rod]
        drop = drop_c if rod.kind == "chord" else drop_d
        u = _unit(s.p1 - s.p0)
        mid = (s.p0 + s.p1) / 2.0
        half = float(np.linalg.norm(s.p1 - s.p0)) / 2.0
        for k, sign in enumerate((-1.0, 1.0)):
            centre = mid + u * (sign * (half - Magazine.BLOCK_IN)) - s.up * drop
            out += v_flanks("slot%d_%d" % (s.rod, k), centre, u, s.up, Magazine.BLOCK_L,
                            Magazine.SLOT_DEPTH, 1.0, Magazine.SLOT_ANGLE / 2.0, C_RACK, CAGE, ROD)
            # a bed under each block so a rod that misses cannot fall through
            out.append(box("slotbed%d_%d" % (s.rod, k), centre - s.up * 2.0,
                           (Magazine.BLOCK_L / 2.0, 4.0, 1.0), u,
                           np.cross(u, s.up) if abs(np.cross(u, s.up) @ s.up) < 0.5
                           else np.array([0, 1.0, 0]), C_RACK, CAGE, ROD))
    return out


# ------------------------------------------------------------------ head
def head_body(geom, fixture, z_lo, start=None):
    """The gantry carriage: x, y, z slides carrying the ring, the
    dispenser on its stroke, the gripper on its stroke and yaw."""
    t = geom.t
    sx, sy, sz = start if start is not None else (-Cage.POST_OFF, 0.0, z_lo + Gantry.Z_TRAVEL - 5.0)
    r_in, r_out = ring_r_in(), ring_r_out()
    hx, hy, hz0, hz1 = Ring.HEAD_BOX
    lo, hi = fixture.x_range()
    out = [
        '<body name="gx" pos="%.6f 0 0">' % mm(sx),
        '  <joint name="gx" type="slide" axis="1 0 0" range="%.6f %.6f" damping="5"/>'
        % (mm(lo - 5.0 - sx), mm(lo - 5.0 + Gantry.X_TRAVEL - sx)),
        # the carriage's own visual: a beam across the cell
        '  <geom name="gx_beam" type="box" pos="0 0 %.6f" size="0.02 %.6f 0.02" rgba="0.3 0.3 0.33 0.5" '
        'contype="0" conaffinity="0" mass="1.5"/>' % (mm(z_lo + Gantry.Z_TRAVEL + 60.0),
                                                       mm(Gantry.Y_TRAVEL / 2.0 + 40.0)),
        '  <body name="gy" pos="0 %.6f 0">' % mm(sy),
        '    <joint name="gy" type="slide" axis="0 1 0" range="%.6f %.6f" damping="5"/>'
        % (mm(-Gantry.Y_TRAVEL / 2.0 - sy), mm(Gantry.Y_TRAVEL / 2.0 - sy)),
        '    <geom name="gy_sled" type="box" pos="0 0 %.6f" size="0.03 0.03 0.015" rgba="0.3 0.3 0.33 0.6" '
        'contype="0" conaffinity="0" mass="1.0"/>' % mm(z_lo + Gantry.Z_TRAVEL + 40.0),
        '    <body name="gz" pos="0 0 %.6f" gravcomp="1">' % mm(sz),
        '      <joint name="gz" type="slide" axis="0 0 1" range="%.6f %.6f" damping="5"/>'
        % (mm(z_lo - sz), mm(z_lo + Gantry.Z_TRAVEL - sz)),
        # the carriage box above the ring -- what the solver sweeps
        '      ' + box("head_box", (0.0, 0.0, (hz0 + hz1) / 2.0), (hx, hy, (hz1 - hz0) / 2.0),
                       (1, 0, 0), (0, 1, 0), "0.35 0.35 0.38 0.35", HEAD_B, ROD | CAGE, mass=900.0),
        # ---- the ring ----
        '      <body name="ring" pos="0 0 0" gravcomp="1">',
        # the armature is the DRIVE's rotor reflected to this axis, not a
        # numerical placeholder: see Ring.drive_armature
        '        <joint name="ring" type="hinge" axis="1 0 0" damping="1e-5" armature="%.9f"/>'
        % Ring.drive_armature(),
        '        <site name="ring_exit" pos="0 0 %.6f" size="0.0005"/>' % mm(Ring.EXIT_R),
        '        <site name="ring_fid" pos="0 0 %.6f" size="0.0005"/>' % mm(-r_out),
    ]
    n = 24
    r_mid = (r_in + r_out) / 2.0
    arc = 2 * pi * r_mid / n
    for i in range(n):
        psi = 2 * pi * (i + 0.5) / n
        # gap centred straight down (psi = 0)
        d = abs(((degrees(psi) + 180.0) % 360.0) - 180.0)
        if d < Ring.GAP / 2.0:
            continue
        c = (0.0, r_mid * sin(psi), -r_mid * cos(psi))
        rad = (0.0, sin(psi), -cos(psi))
        tan_ = (0.0, cos(psi), sin(psi))
        out.append('        ' + box("ring%d" % i, c, (Ring.W / 2.0, arc * 0.55, (r_out - r_in) / 2.0),
                                    (1, 0, 0), tan_, C_RING, HEAD_B, ROD | CAGE,
                                    mass=Ring.MASS * 0.7 / (n * (360.0 - Ring.GAP) / 360.0)))
    # the thread, wound into a groove in the ring's own web: a visual band
    # inside the section, adding nothing to the envelope
    out.append('        ' + cylinder("thread", (-Ring.GROOVE_W / 2.0, 0, 0),
                                     (Ring.GROOVE_W / 2.0, 0, 0),
                                     Ring.GROOVE_R + Ring.GROOVE_D / 2.0,
                                     "0.90 0.80 0.20 0.35", 0, 0, mass=Ring.MASS * 0.1))
    out += [
        '      </body>',
        # ---- the raceway: what carries the ring, and does not turn.  Open
        # over the mouth, where the work comes in.
    ]
    M = Ring.mesh()
    n_race = 20
    r_race0, r_race1 = r_out + Ring.RACE_CLEAR, rail_r_out()
    rr_mid = (r_race0 + r_race1) / 2.0
    arc = 2 * pi * rr_mid / n_race
    for i in range(n_race):
        psi = 2 * pi * (i + 0.5) / n_race
        if abs(((degrees(psi) + 180.0) % 360.0) - 180.0) < Ring.race_mouth() / 2.0:
            continue
        c = (0.0, rr_mid * sin(psi), -rr_mid * cos(psi))
        tan_ = (0.0, cos(psi), sin(psi))
        out.append('      ' + box("race%d" % i, c, (Ring.W / 2.0 + Ring.RACE_T, arc * 0.55,
                                                    (r_race1 - r_race0) / 2.0),
                                  (1, 0, 0), tan_, "0.35 0.35 0.40 1", HEAD_B, ROD | CAGE,
                                  mass=6.0))
    # THE PINIONS ARE COSMETIC HERE and their teeth are not drawn: in the
    # cell the ring is a hinge on an actuator, so what the drive has to
    # prove -- that it locates the ring and turns it without losing a tooth
    # -- cannot be asked of a hinge.  truss.drive asks it, of real teeth.
    for k, az in enumerate(Ring.pinion_az()):
        a = radians(az)
        c = np.array([0.0, M.centre * sin(a), -M.centre * cos(a)])
        out.append('      ' + cylinder("pinion%d" % k, c - np.array([Ring.W / 2.0, 0, 0]),
                                       c + np.array([Ring.W / 2.0, 0, 0]), M.r_pinion + M.m,
                                       "0.85 0.55 0.15 1", HEAD_B, ROD | CAGE, mass=3.0))
    out += [
        # ---- the dispenser on its stroke ----
        '      <body name="disp" pos="%.6f 0 %.6f" gravcomp="1">' % (mm(Head.disp_x()), mm(Head.TIP_PARK)),
        '        <joint name="gd" type="slide" axis="0 0 -1" range="0 %.6f" damping="2"/>'
        % mm(Head.DISP_STROKE),
        '        ' + cylinder("nozzle", (0, 0, 0), (0, 0, 22.0), Dispenser.NOZZLE_D / 2.0 + 0.8,
                              "0.85 0.85 0.9 1", HEAD_B, ROD | CAGE, mass=5.0),
        '        ' + box("disp_body", (0, 0, 40.0), (Dispenser.BODY_W / 2.0, 6.0, 18.0),
                         (1, 0, 0), (0, 1, 0), C_HEAD, HEAD_B, ROD | CAGE, mass=60.0),
        '        <site name="nozzle_tip" pos="0 0 0" size="0.0005"/>',
        '      </body>',
        # ---- the gripper on its stroke and yaw ----
        '      <body name="grip" pos="%.6f 0 %.6f" gravcomp="1">' % (mm(Head.grip_x()), mm(Head.TIP_PARK)),
        '        <joint name="gg" type="slide" axis="0 0 -1" range="0 %.6f" damping="2"/>'
        % mm(Head.GRIP_STROKE),
        '        ' + box("grip_body", (0, 0, Gripper.FINGER_L + 14.0), (Gripper.BODY_W / 2.0, 9.0, 12.0),
                         (1, 0, 0), (0, 1, 0), C_HEAD, HEAD_B, ROD | CAGE, mass=80.0),
        '        <body name="gripw" pos="0 0 0" gravcomp="1">',
        '          <joint name="gw" type="hinge" axis="0 0 1" range="%.1f %.1f" damping="0.002"/>'
        % (-Head.GRIP_YAW, Head.GRIP_YAW),
        '          <site name="grip_pt" pos="0 0 0" size="0.0005"/>',
        # the yaw plate the fingers hang from
        '          ' + box("grip_plate", (0, 0, Gripper.FINGER_L + 6.0), (8.0, 8.0, 2.0),
                           (1, 0, 0), (0, 1, 0), C_HEAD, 0, 0, mass=6.0),
    ]
    for s, tag in ((1.0, "l"), (-1.0, "r")):
        y = s * (Gripper.JAW_OPEN / 2.0 + Gripper.PAD_T / 2.0)
        out += [
            '          <body name="finger_%s" pos="0 %.6f 0" gravcomp="1">' % (tag, mm(y)),
            '            <joint name="gf_%s" type="slide" axis="0 %d 0" range="%.6f 0" damping="0.5"/>'
            % (tag, int(-s), mm(-(Gripper.JAW_OPEN / 2.0 - 0.3))),
            # THE PADS STOP JUST UNDER THE ROD'S AXIS.  A rod in a 90-degree V
            # is held at its own radius above the apex, so a pad reaching
            # further below the axis than 0.4 mm lands on the flanks before
            # it lands on the rod (measured: jaws that could not close on a
            # racked chord).  The pads grip the rod's upper half.
            '            ' + box("pad_%s" % tag, (0, 0, Gripper.PAD_H / 2.0 - Gripper.PAD_UNDER),
                                 (Gripper.PAD_L / 2.0, Gripper.PAD_T / 2.0, Gripper.PAD_H / 2.0),
                                 (1, 0, 0), (0, 1, 0), "0.2 0.2 0.22 1", HEAD_B, ROD | CAGE, mass=3.0,
                                 extra=' friction="0.9 0.005 0.0001"'),
            '            ' + box("stem_%s" % tag, (0, 0, Gripper.FINGER_L / 2.0 + Gripper.PAD_H - Gripper.PAD_UNDER),
                                 (Gripper.PAD_L / 2.0, Gripper.PAD_T / 2.0, Gripper.FINGER_L / 2.0),
                                 (1, 0, 0), (0, 1, 0), C_HEAD, HEAD_B, ROD | CAGE, mass=4.0),
            '          </body>',
        ]
    fovy = 2.0 * degrees(atan2(Vision.H / 2.0, Vision.f_px()))
    out += [
        '        </body>',
        '      </body>',
        # ---- the camera: under the carriage, outboard of the dispenser,
        # looking in at the joint (Vision.cam_pos) ----
        '      <camera name="head_cam" pos="%s" xyaxes="%s %s" fovy="%.4f"/>'
        % (_v(Vision.cam_pos()), " ".join("%.6f" % c for c in Vision.cam_frame()[0]),
           " ".join("%.6f" % c for c in Vision.cam_frame()[1]), fovy),
        '      <site name="ring_centre" pos="0 0 0" size="0.0005"/>',
        '    </body>',
        '  </body>',
        '</body>',
    ]
    return out


def drops(n, park=(-400.0, 0.0, -200.0)):
    out = []
    for i in range(n):
        p = (park[0] - 8.0 * i, park[1], park[2])
        out.append('<body name="drop%d" pos="%s"><freejoint name="drop%d_f"/>'
                   '<geom name="drop%d_g" type="sphere" size="0.0012" rgba="%s" mass="0.0001" '
                   'contype="%d" conaffinity="%d"/></body>'
                   % (i, _v(p), i, i, C_DROP, DROP, ROD | CAGE))
    return out


# ------------------------------------------------------------ assembly
def z_floor(geom, fixture):
    """Lowest ring-centre height the carriage needs: seated on the lowest
    chord it ever winds is chord-up, R; the loader needs grip_z(R/2) which
    is higher.  Leave a little under R."""
    return geom.R - 15.0


def scene_cell(geom, fixture, stage="empty", start=None, timestep=5e-4, n_drops=None):
    """The whole cell.  stage="empty": rods in their racks, welds off;
    "loaded": rods in the fixture, welded to the cage."""
    t = geom.t
    n_drops = geom.t.n_joints if n_drops is None else n_drops
    z_lo = z_floor(geom, fixture)
    parts = []
    floor_z = -(geom.R + 120.0)
    parts.append('<geom name="floor" type="plane" pos="%.6f 0 %.6f" size="3 3 0.1" material="floor" '
                 'contype="%d" conaffinity="%d"/>' % (mm(t.length / 2.0), mm(floor_z), CAGE, ROD | DROP))
    parts += rack_geoms(geom, fixture)
    parts += cage_body(geom, fixture)
    parts += rods_in_racks(geom, fixture) if stage == "empty" else rods_in_fixture(geom)
    parts += head_body(geom, fixture, z_lo, start=start)
    parts += drops(n_drops)
    # ---- equality: a weld per rod to the cage (the keeper) and to the
    # gripper's yaw body (the grip), both inactive unless loaded
    eq = ["  <equality>",
          # a parallel gripper's jaws move together by a rack and pinion:
          # a rigid coupling, not a shared tendon (with the tendon alone a
          # blocked pad let the other run past the rod)
          '    <joint name="jaws_sym" joint1="gf_l" joint2="gf_r" polycoef="0 1 0 0 0"/>']
    for r in geom.rods:
        eq.append('    <weld name="keep%d" body1="cage" body2="rod%d" active="%s" '
                  'solref="0.004 1" solimp="0.98 0.999 0.001"/>'
                  % (r.index, r.index, "true" if stage == "loaded" else "false"))
        eq.append('    <weld name="hold%d" body1="gripw" body2="rod%d" active="false" '
                  'solref="0.004 1" solimp="0.98 0.999 0.001"/>' % (r.index, r.index))
    for i in range(n_drops):
        eq.append('    <weld name="stick%d" body1="cage" body2="drop%d" active="false" '
                  'solref="0.004 1" solimp="0.98 0.999 0.001"/>' % (i, i))
    eq.append("  </equality>")
    # ---- actuators
    act = """  <actuator>
    <position name="a_x" joint="gx" kp="40000" kv="600" forcerange="-%g %g"/>
    <position name="a_y" joint="gy" kp="30000" kv="450" forcerange="-%g %g"/>
    <position name="a_z" joint="gz" kp="30000" kv="450" forcerange="-%g %g"/>
    <position name="a_d" joint="gd" kp="3000" kv="40" forcerange="-30 30"/>
    <position name="a_g" joint="gg" kp="3000" kv="40" forcerange="-30 30"/>
    <position name="a_w" joint="gw" kp="0.8" kv="0.01" forcerange="-0.3 0.3"/>
    <position name="a_f" tendon="jaws" kp="300" kv="3" forcerange="-15 15"/>
    <position name="a_ring" joint="ring" kp="%.6f" kv="%.6f" forcerange="-%g %g"/>
    <position name="a_cage" joint="cage" kp="60" kv="2.5" forcerange="-6 6"/>
  </actuator>""" % (Gantry.STALL_N["x"], Gantry.STALL_N["x"], Gantry.STALL_N["y"], Gantry.STALL_N["y"],
                    Gantry.STALL_N["z"], Gantry.STALL_N["z"], Ring.servo_kp(),
                    Ring.servo_kv(), Ring.DRIVE_TORQUE, Ring.DRIVE_TORQUE)
    ten = """  <tendon>
    <fixed name="jaws">
      <joint joint="gf_l" coef="1"/>
      <joint joint="gf_r" coef="1"/>
    </fixed>
  </tendon>"""
    sen = """  <sensor>
    <jointpos name="s_ring" joint="ring" noise="0.0005"/>
    <jointpos name="s_cage" joint="cage"/>
    <framepos name="s_ring_centre" objtype="site" objname="ring_centre"/>
    <framepos name="s_grip" objtype="site" objname="grip_pt"/>
    <framepos name="s_nozzle" objtype="site" objname="nozzle_tip"/>
  </sensor>"""
    contact = ['  <contact>',
               '    <exclude body1="gz" body2="ring"/>',
               '    <exclude body1="gz" body2="disp"/>',
               '    <exclude body1="gz" body2="grip"/>',
               '    <exclude body1="grip" body2="gripw"/>',
               '    <exclude body1="gripw" body2="finger_l"/>',
               '    <exclude body1="gripw" body2="finger_r"/>',
               '    <exclude body1="finger_l" body2="finger_r"/>',
               # the finger stems run up inside the gripper's body: a
               # grandparent, which MuJoCo does not exclude on its own
               '    <exclude body1="grip" body2="finger_l"/>',
               '    <exclude body1="grip" body2="finger_r"/>',
               '    <exclude body1="gz" body2="finger_l"/>',
               '    <exclude body1="gz" body2="finger_r"/>',
               '    <exclude body1="gz" body2="gripw"/>',
               '    <exclude body1="ring" body2="disp"/>',
               '    <exclude body1="ring" body2="grip"/>',
               '  </contact>']
    cams = """
    <camera name="cell" pos="%.4f -0.9 0.7" xyaxes="1 0 0 0 0.6 0.8"/>
    <camera name="side" pos="%.4f -0.35 0.20" xyaxes="1 0 0 0 0.45 0.9"/>""" % (
        mm(t.length / 2.0), mm(t.length / 2.0))
    return """<mujoco model="truss_cell_%s">
%s
  <worldbody>%s
%s
  </worldbody>
%s
%s
%s
%s
%s
</mujoco>
""" % (t.name, preamble(timestep), cams, "\n".join("    " + p for p in parts),
       "\n".join(contact), "\n".join(eq), ten, act, sen)
