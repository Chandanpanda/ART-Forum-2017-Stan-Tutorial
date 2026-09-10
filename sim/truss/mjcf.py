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

from .spec import (mm, Truss, Ring, Gantry, Head, Gripper, Cage, Magazine, Stock,
                   Dispenser, Vision, Process, ring_r_in, ring_r_out, rail_r_out)
from .geometry import radial
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
def cage_body(geom, fixture, mount=None):
    """The rotating fixture as one body on a hinge about x.  With `mount`,
    the finished nose rides on it too -- by then the truss is cured and the
    whole assembly turns together."""
    t = geom.t
    out = ['<body name="cage" pos="0 0 0">',
           '  <joint name="cage" type="hinge" axis="1 0 0" damping="0.05" '
           'armature="1e-5"/>']
    ef = fixture.end_free()
    x0, x1 = -(ef + Cage.END_PLATE_T), t.length + ef + Cage.END_PLATE_T
    # THE BACKBONE STOPS AT THE NOSE.  It used to be STEPPED over it,
    # necked down to Cage.nose_spine_r() -- but the camera is mounted on
    # this same axis and its board contains it, so no radius clears it and
    # the tube ran 4.5 mm inside the module.  Over the nose the end plate
    # is carried on the machine's own bearing; what is drawn here is the
    # plate's hub, reaching in as far as the nose allows.
    nr = Cage.nose_spine_r()
    n0, n1 = fixture.nose_span(0), fixture.nose_span(1)
    hub0 = min(0.0, n0[0] - Process.SEAT_CLEAR) if n0 else 0.0
    hub1 = max(t.length, n1[1] + Process.SEAT_CLEAR) if n1 else t.length
    out.append("  " + cylinder("spine", (0.0, 0, 0), (t.length, 0, 0),
                               fixture.spine_r(), C_CAGE, CAGE, ROD | HEAD_B, mass=80.0))
    if hub0 > x0:
        out.append("  " + cylinder("spine_n0", (x0, 0, 0), (hub0, 0, 0), nr, C_CAGE,
                                   CAGE, ROD | HEAD_B, mass=4.0))
    if x1 > hub1:
        out.append("  " + cylinder("spine_n1", (hub1, 0, 0), (x1, 0, 0), nr, C_CAGE,
                                   CAGE, ROD | HEAD_B, mass=4.0))
    plate_r = fixture.plate_r()
    for tag, xa in (("p0", x0), ("p1", t.length + ef)):
        out.append("  " + cylinder("plate_%s" % tag, (xa, 0, 0), (xa + Cage.END_PLATE_T, 0, 0),
                                   plate_r, C_CAGE, CAGE, ROD | HEAD_B, mass=40.0))
    # THE COLLAPSING MANDREL.  Six torsion shafts on the spine's surface,
    # parallelogram arms off them, segmented rails on the arms, one
    # over-centre brace per shaft.  Drawn erect and locked, which is how it
    # spends the whole build; collapsing it is a manual step afterwards.
    col = fixture.collapse
    sx0, sx1 = col.shaft_span()
    for sh, (kind, kk, phi) in enumerate(col.shafts):
        up = radial(phi)
        a = np.array([sx0, 0.0, 0.0]) + up * col.shaft_r()
        b = np.array([sx1, 0.0, 0.0]) + up * col.shaft_r()
        out.append("  " + cylinder("shaft%d" % sh, a, b, Cage.PIVOT_D / 2.0,
                                   C_CAGE, CAGE, ROD | HEAD_B, mass=6.0))
    for i, (kind, sh, phi, r, (xa, xb)) in enumerate(col.rails):
        up = radial(phi)
        a = np.array([xa, 0.0, 0.0]) + up * r
        b = np.array([xb, 0.0, 0.0]) + up * r
        out.append("  " + box("rail%d" % i, (a + b) / 2.0,
                              (float(np.linalg.norm(b - a)) / 2.0,
                               Cage.ARM_W / 2.0, col.rail_h() / 2.0),
                              np.array([1.0, 0.0, 0.0]), np.cross(up, np.array([1.0, 0, 0])),
                              C_CAGE, CAGE, ROD | HEAD_B, mass=2.0))
    for i, a in enumerate(col.arms):
        out.append("  " + box("marm%d" % i, (a.base + a.tip) / 2.0,
                              (a.length / 2.0, Cage.LINK_W / 2.0, Cage.LINK_T / 2.0),
                              a.up, np.cross(np.array([1.0, 0, 0]), a.up),
                              C_CAGE, CAGE, ROD | HEAD_B, mass=1.0))
    for b in col.braces:
        out.append("  " + capsule("brace%da" % b.shaft, b.anchor, b.knee,
                                  Cage.LINK_T / 2.0, C_HEAD, CAGE, ROD | HEAD_B))
        out.append("  " + capsule("brace%db" % b.shaft, b.knee, b.attach,
                                  Cage.LINK_T / 2.0, C_HEAD, CAGE, ROD | HEAD_B))
    # the release rod, down the spine's bore.  IT STOPS AT THE NOSE TOO --
    # it is 6 mm across on the axis, and the camera is on the axis.  Its
    # pull is taken off the axis outside the kit's silhouette (see
    # Cage.nose_spine_r); what is drawn is the rod inside the truss, which
    # is where it is for the whole build.
    dx0 = (n0[1] + Process.SEAT_CLEAR) if n0 else (x0 - col.draw_stroke())
    dx1 = (n1[0] - Process.SEAT_CLEAR) if n1 else x1
    out.append("  " + cylinder("draw", (dx0, 0, 0), (dx1, 0, 0),
                               Cage.DRAW_D / 2.0, C_HEAD, CAGE, ROD | HEAD_B, mass=20.0))
    # pins: a post from the rail out to the notch, then the V
    for i, p in enumerate(fixture.pins):
        root = np.array([p.x, 0.0, 0.0]) + p.up * col.rail_r("chord", p.chord)
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
        # a stem down onto the face rail, which is what carries it
        stem0 = c.apex - c.up * 2.0
        stem1 = c.apex - c.up * max(3.0, float(np.linalg.norm(c.apex[1:]))
                                    - col.rail_r("face", 0))
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
    if mount is not None:
        out += mount_geoms(geom, mount)
    out.append("</body>")
    return out


# ------------------------------------------------------------------ rods
def rod_body(geom, rod, p0, p1):
    """A free rod at world endpoints p0, p1 (its centreline)."""
    t = geom.t
    if rod.kind == "mcam":
        # THE WHOLE HEAD KIT, as one part on a carrier boss: the module, the
        # COLLAR bonded round its lens housing, and the four GRID rods wound
        # to each other at four crossings.  All of it arrives from station B
        # already made -- this cell cannot wind those crossings -- so to the
        # loader it is one rigid thing to fetch, and it must be DRAWN as the
        # thing that arrives.  Drawn as a module with a solid plate on it,
        # the cell was planning against a part that does not exist: no grid
        # to land the struts on, and a plate over the lens.
        from .spec import Payload, Bracket, Carrier
        from .mount import collar_segments, payload_solid
        d = t.d_diag
        c = (p0 + p1) / 2.0
        look = -(p1 - p0) / float(np.linalg.norm(p1 - p0))
        xh = np.array([1.0, 0.0, 0.0])
        up = np.cross(look, xh)
        if np.linalg.norm(up) < 1e-9:
            up = np.array([0.0, 0.0, 1.0])
        up = up / np.linalg.norm(up)
        base = p0 + look * (Payload.BOX[1] / 2.0)      # the module's centre

        def at(dx, dl, du):
            """A point in the CAMERA's own frame -- the frame mount.solve
            builds every part of the mount in."""
            return base - c + dx * xh + dl * look + du * up

        gs = [cylinder("rod%d_g" % rod.index, p0 - c, p1 - c, rod.r,
                       C_ALU, ROD, ROD | CAGE | HEAD_B | DROP,
                       mass=Carrier.MASS, extra=' class="rod"')]
        # the module, as the two boxes it really occupies -- the collar
        # seats on the board's FRONT FACE with the housing standing through
        # its aperture, which one 9 mm slab cannot say
        for k, e in enumerate(payload_solid(Payload)):
            gs.append(box("rod%d_b%d" % (rod.index, k),
                          at(*(0.5 * (a[0] + a[1]) for a in e)),
                          tuple(0.5 * (a[1] - a[0]) for a in e),
                          xh, look, C_CAM if k == 0 else "0.10 0.10 0.12 1",
                          ROD, ROD | CAGE | HEAD_B | DROP,
                          mass=Payload.MASS if k == 0 else 0.3))
        # the collar: four bands, the ring round the aperture, four ribs
        segs = collar_segments(Payload, d)
        lc = 0.5 * sum(Bracket.seat_l())
        for k, (s0, s1, sw) in enumerate(segs):
            gs.append(box("rod%d_p%d" % (rod.index, k),
                          at(0.5 * (s0[0] + s1[0]), lc, 0.5 * (s0[1] + s1[1])),
                          (max(abs(s1[0] - s0[0]) / 2.0, sw / 2.0),
                           Bracket.SHEET / 2.0,
                           max(abs(s1[1] - s0[1]) / 2.0, sw / 2.0)),
                          xh, look, C_ALU, ROD, ROD | CAGE | HEAD_B | DROP,
                          mass=Bracket.mass(Payload, d) / len(segs)))
        # the tic-tac-toe, wound and bonded at station B
        gx, gu = Bracket.grid_half(Payload, d)
        over = Bracket.overrun(d)
        grid = ([((-gx - over, Bracket.layer_l(0, Payload, d), su * gu),
                  (+gx + over, Bracket.layer_l(0, Payload, d), su * gu))
                 for su in (+1.0, -1.0)]
                + [((sx * gx, Bracket.layer_l(1, Payload, d), -gu - over),
                    (sx * gx, Bracket.layer_l(1, Payload, d), +gu + over))
                   for sx in (+1.0, -1.0)])
        for k, (a, b) in enumerate(grid):
            qa, qb = at(*a), at(*b)
            gs.append(cylinder("rod%d_q%d" % (rod.index, k), qa, qb, d / 2.0,
                               C_ROD, ROD, ROD | CAGE | HEAD_B | DROP,
                               mass=Stock.rho_lin(d)
                               * float(np.linalg.norm(qb - qa)) / 1000.0))
        return ('<body name="rod%d" pos="%s"><freejoint name="rod%d_f"/>%s</body>'
                % (rod.index, _v(c), rod.index, "".join(gs)))
    if rod.kind.startswith("m"):
        # A MOUNT ROD.  Its mass is its own: the truss's per-rod averages
        # are for chords and diagonals, and a 30 mm strut charged as one of
        # twenty-four diagonals weighs three times what it is.
        L = float(np.linalg.norm(p1 - p0))
        g = cylinder("rod%d_g" % rod.index, p0 - (p0 + p1) / 2.0, p1 - (p0 + p1) / 2.0,
                     rod.r, C_ROD, ROD, ROD | CAGE | HEAD_B | DROP,
                     mass=Stock.rho_lin(2.0 * rod.r) * L / 1000.0,
                     extra=' class="rod"')
    elif rod.kind == "chord":
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
    for rod in geom.all_rods:
        s = fixture.slot_of(rod.index)
        out.append(rod_body(geom, rod, s.p0, s.p1))
    return out


def rods_in_fixture(geom, fixture=None, racked=()):
    """Every rod where the fixture holds it -- except the kinds named in
    `racked`, which start in their rack slots instead.

    WHY THE EXCEPTION EXISTS.  `rig_mount` wants a truss that is already
    built and a mount that is not, so that the mount phase has something to
    fetch.  Given "loaded" as it stood, the mount rods started welded at
    their nominal poses, every grip in the phase closed on nothing, and the
    rig reported the mount built to 0.44 mm without having laid any of it.
    A rig that cannot fail is not a measurement."""
    out = []
    for rod in geom.all_rods:
        if rod.kind in racked and fixture is not None:
            s = fixture.slot_of(rod.index)
            out.append(rod_body(geom, rod, s.p0, s.p1))
            continue
        p0, p1 = (rod.p0, rod.p1) if rod.kind != "diag" else geom.diag_body_ends(rod)
        out.append(rod_body(geom, rod, p0, p1))
    return out


# ----------------------------------------------------------------- mount
C_ALU, C_CAM, C_LENS = "0.72 0.75 0.78 1", "0.10 0.35 0.16 1", "0.06 0.06 0.08 1"


def mount_geoms(geom, mount, payload=None):
    """The camera mount, welded to the cage: the carbon of the nose, the
    laser-cut collar, and the module itself.

    Part of the CAGE body rather than free rods -- by the time the mount
    goes on, the truss is cured and the whole assembly turns together.  What
    this is for is looking at it: the numbers are check_mount's.
    """
    from .spec import Payload, Bracket
    from .geometry import radial
    payload = Payload if payload is None else payload
    out = []
    for r in mount.rods:
        if r.kind == "collar":
            # a STRIP OF SHEET, wide in its own plane and thin through it:
            # drawn as a rod it looked like a wire frame, and the plate's
            # real footprint on the board is what says whether it fits
            n = np.asarray(r.normal, float)
            n = n / np.linalg.norm(n)
            e = np.cross(r.axis, n)
            out.append("  " + box("collar%d" % r.index, r.mid,
                                  (float(np.linalg.norm(r.p1 - r.p0)) / 2.0,
                                   r.half_w, r.r),
                                  r.axis, e, C_ALU, CAGE, ROD | HEAD_B))
        else:
            col = {"grid": C_ROD, "strut": C_ROD, "batten": C_ROD}[r.kind]
            out.append("  " + capsule("m_%s%d" % (r.kind, r.index), r.p0, r.p1,
                                      r.r, col, CAGE, ROD | HEAD_B))
    # the module: a slab on the plate's plane, looking out along `look`
    for centre, look, _lr in mount.payload:
        look = np.asarray(look, float) / float(np.linalg.norm(look))
        up = np.cross(look, np.array([1.0, 0.0, 0.0]))
        up = up / np.linalg.norm(up)
        i = len(out)
        out.append("  " + box("cam%d" % i, centre,
                              (payload.BOX[0] / 2.0, payload.BOX[1] / 2.0,
                               payload.BOX[2] / 2.0),
                              np.array([1.0, 0.0, 0.0]), look, C_CAM, CAGE, ROD | HEAD_B))
        lens0 = centre + look * (payload.BOX[1] / 2.0)
        out.append("  " + cylinder("lens%d" % i, lens0,
                                   lens0 + look * 1.5, payload.LENS_D / 2.0,
                                   C_LENS, CAGE, ROD | HEAD_B))
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
        rod = geom.all_rods[s.rod]
        drop = drop_c if rod.kind in ("chord", "mcam") else drop_d
        u = _unit(s.p1 - s.p0)
        mid = (s.p0 + s.p1) / 2.0
        L = float(np.linalg.norm(s.p1 - s.p0))
        if rod.kind == "mcam":
            # THE CAMERA IS NOT A ROD IN ITS RECEPTACLE.  It is a module on
            # a printed carrier, and what the rack holds is the CARRIER --
            # a nest under the module's own body, with the boss standing
            # free for the jaws.  Held on V-blocks under the boss alone it
            # is a 4 g overhang on a 3 mm pin, and it rolls off.
            from .spec import Payload, Carrier, Process, Bracket
            look = -u
            base = s.p0 + look * (Payload.BOX[1] / 2.0)
            # A POCKET, NOT A SHELF.  The module is 24 mm tall on a 9 mm
            # base; stood on a flat nest it topples the moment the sim
            # settles and drags itself off by the boss -- measured.  Two
            # walls a bond gap outside its faces hold it upright, which is
            # what a printed kitting nest is.
            # A FIT, NOT A CLEARANCE.  At a wall-thickness of slack the
            # module slides 1.7 mm toward the boss as the sim settles, and
            # its near face ends up 0.3 mm inside where the pads close --
            # measured, as a stalled stroke on the board.  The nest is
            # printed to the part, so it is the carrier's own FIT.
            wall = 2.0
            hy = Payload.BOX[1] / 2.0 + Carrier.FIT
            # ...AND THE FIT IS ON THE BACK FACE ONLY, because the front is
            # where the product is.  The kit stands `kit_proud` in front of
            # the board with a collar and a wound tic-tac-toe on it; a wall
            # a fit's interference from the board's front face is a wall
            # inside the grid.  Sized against the module it cleared the
            # part that actually arrives by a tenth of a millimetre.
            fy = (Payload.BOX[1] / 2.0 - Payload.CASE_PROUD
                  + Carrier.kit_proud(Payload, geom.t.d_diag)
                  + Process.SEAT_CLEAR)
            kx, ku = Carrier.kit_half(Payload, geom.t.d_diag)
            # THE FLOOR IS A PAD UNDER THE BOARD, not a plate under the kit.
            # The grid hangs BELOW the module -- its layer-1 rods reach
            # 18.4 mm down where the board reaches 11.9 -- so a floor that
            # takes the module's own weight is a floor those rods stand in.
            # Measured on the built scene: 3.55 mm inside it.  The pad
            # therefore stops inboard of the grid in x and behind it in
            # look, and carries the board's underside where nothing else of
            # the kit reaches.
            gx, _gu = Bracket.grid_half(Payload, geom.t.d_diag)
            hx = min(Payload.BOX[0] / 2.0,
                     gx - geom.t.d_diag / 2.0 - Process.SEAT_CLEAR)
            back = Payload.BOX[1] / 2.0 + wall
            # ...and it stops BEHIND THE COLLAR, not at the board's front
            # face.  The collar overhangs the board by 2.5 mm on every side,
            # so under the module its own bands hang lower than the board
            # does and a pad flush with the board's face is a pad against
            # them -- measured on the built scene, touching at 0.00 mm.
            front = Bracket.seat_l(Payload)[0] - Process.SEAT_CLEAR
            pt = 4.0                                             # pad thickness
            out.append(box("nest%d" % s.rod,
                           base + look * (0.5 * (front - back))
                           - s.up * (Payload.BOX[2] / 2.0 + pt / 2.0),
                           (hx, 0.5 * (front + back), pt / 2.0),
                           np.array([1.0, 0.0, 0.0]), look, C_RACK, CAGE, ROD))
            # The FAR wall is whole; the NEAR one is SPLIT, because the
            # carrier's boss comes out through it and the jaws close on the
            # boss at its middle.  A whole near wall stands 4.5 mm from the
            # grip point and the pads land on it before the boss -- which
            # is what "jaws closed on nothing" was.
            # THE GAP IS THE JAWS', NOT THE BOSS'S.  At the yaw the boss is
            # taken at, the pads straddle it along x and reach JAW_OPEN/2 +
            # PAD_T out either side; a wall inside that is a wall the
            # gripper's stroke stalls against.
            # ...and the two pieces go at the module's own CORNERS, not
            # wherever is left over: that is as far from the grip point as
            # the nest reaches, and it is where a tab holds a board best.
            gap = Gripper.JAW_OPEN / 2.0 + Gripper.PAD_T + 1.0
            hw = 3.0
            # THE FRONT WALL IS A KEEPER, NOT A BEARING.  It stands one
            # process clearance in front of the whole kit -- the grid, not
            # the housing -- and touches nothing; what actually retains the
            # module is the carrier it is pressed into, which is a printed
            # part and not modelled here.  The BACK tabs keep the fit, and
            # the back is the direction the module was measured sliding.
            out.append(box("nestw%d_a" % s.rod,
                           base + look * (fy + wall / 2.0)
                           - s.up * (Payload.BOX[2] / 4.0),
                           (kx + wall, wall / 2.0, Payload.BOX[2] / 4.0),
                           np.array([1.0, 0.0, 0.0]), look, C_RACK, CAGE, ROD))
            for sx, tag in ((+1.0, "b"), (-1.0, "c")):
                # ...and they stop BELOW the pads' own reach: a tab whose
                # top is level with the boss's axis brushes the pad as it
                # closes, and a contact on the jaws is a contact the grip
                # sensor has to argue with.
                out.append(box("nestw%d_%s" % (s.rod, tag),
                               base - look * (hy + wall / 2.0)
                               - s.up * (Payload.BOX[2] / 4.0 + Gripper.PAD_H / 2.0)
                               + np.array([sx * (Payload.BOX[0] / 2.0 + wall - hw),
                                           0.0, 0.0]),
                               (hw, wall / 2.0, Payload.BOX[2] / 4.0),
                               np.array([1.0, 0.0, 0.0]), look, C_RACK, CAGE, ROD))
        for k, (off, bl) in enumerate(Magazine.blocks(L)):
            centre = mid + u * off - s.up * drop
            out += v_flanks("slot%d_%d" % (s.rod, k), centre, u, s.up, bl,
                            Magazine.SLOT_DEPTH, 1.0, Magazine.SLOT_ANGLE / 2.0, C_RACK, CAGE, ROD)
            # a bed under each block so a rod that misses cannot fall through
            out.append(box("slotbed%d_%d" % (s.rod, k), centre - s.up * 2.0,
                           (bl / 2.0, 4.0, 1.0), u,
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


# ------------------------------------------------- station B, the sub-assembly
C_NEST, C_KIT = "0.62 0.58 0.50 1", "0.80 0.82 0.85 1"


def station_b(kit, origin=None):
    """Station B's bodies, actuators and equalities, in the cell's frame.

    A SEPARATE MACHINE IN THE SAME SCENE.  It has its own three axes, its
    own gripper, its own winder and its own dispenser -- nothing is shared
    with the main cell but the floor -- because the two run at the same
    time on different parts.  Everything is prefixed `b` so no name can
    collide with the cell's.

    THE PART LIES FLAT AND THE LENS POINTS UP.  That is the one orientation
    in which three axes reach all four crossings and all four rods without
    the part being turned over, and it is why B needs no fixture axis.
    """
    from .spec import StationB, Payload, Bracket, Process
    o = np.asarray(StationB.ORIGIN if origin is None else origin, float)
    px, pu = kit.px, kit.pu
    lo, hi = kit.collar_z()
    out, eq, act = [], [], []

    def at(p):
        return o + np.asarray(p, float)

    # ---- the nest: a pocket the module drops into, lens up
    w = StationB.NEST_WALL
    bx, by = Payload.BOX[0] / 2.0, Payload.BOX[2] / 2.0
    out.append(box("b_nest", at((0.0, 0.0, -Payload.BOX[1] / 2.0 - StationB.NEST_H / 2.0)),
                   (bx + w, by + w, StationB.NEST_H / 2.0),
                   (1, 0, 0), (0, 1, 0), C_NEST, CAGE, ROD))
    # THE WALLS STOP UNDER THE BOARD'S FRONT FACE.  The collar is bigger
    # than the module -- 30 mm across a 25 mm board -- so a pocket as tall
    # as the module is a pocket the collar lands ON instead of passing into.
    # Measured off the first render of the station.
    w_top = lo - Process.SEAT_CLEAR
    w_lo = -Payload.BOX[1] / 2.0 - StationB.NEST_H
    for sx, sy, tag in ((1, 0, "xa"), (-1, 0, "xb"), (0, 1, "ya"), (0, -1, "yb")):
        c = at((sx * (bx + w / 2.0), sy * (by + w / 2.0), (w_top + w_lo) / 2.0))
        hx = w / 2.0 if sx else bx + w
        hy = by + w if sx else w / 2.0
        out.append(box("b_nestw_%s" % tag, c, (hx, hy, (w_top - w_lo) / 2.0),
                       (1, 0, 0), (0, 1, 0), C_NEST, CAGE, ROD))
    # ---- the racks: a V-block pair under each waiting rod, and a shelf
    # under the collar
    for knd, i, a, b in kit.rack_slots():
        if knd == "collar":
            # THE SHELF'S FACE IS WHERE THE RACK SAYS THE COLLAR LIES, less
            # its own thickness.  Two millimetres down was a number, and the
            # collar settled a quarter of a millimetre above where the plan
            # went looking for it.
            t = 1.5
            top = float(a[2]) - Bracket.SHEET / 2.0
            out.append(box("b_shelf", at((0.0, float(a[1]), top - t / 2.0)),
                           (px + 2.0, pu + 2.0, t / 2.0), (1, 0, 0), (0, 1, 0),
                           C_NEST, CAGE, ROD))
            continue
        u = _unit(b - a)
        L = float(np.linalg.norm(b - a))
        drop = (kit.d / 2.0) / sin(radians(Magazine.SLOT_ANGLE / 2.0))
        for k, (off, bl) in enumerate(Magazine.blocks(L)):
            c = at((a + b) / 2.0 + u * off - np.array([0, 0, drop]))
            out += v_flanks("b_slot%d_%d" % (i, k), c, u, np.array([0, 0, 1.0]),
                            bl, Magazine.SLOT_DEPTH, 1.0, Magazine.SLOT_ANGLE / 2.0,
                            C_RACK, CAGE, ROD)
    # ---- THE JIG: where the frame is built and wound, in clear air.
    # A plate and four posts, and every one of those positions comes out of
    # `Kit` -- the posts stand as far along the layer-0 rods as the winder's
    # own swept solid allows, and none stands under layer 1, because layer 1
    # rests on layer 0 exactly as it does in the finished mount.
    jc, jh = kit.jig_base()
    out.append(box("b_jigbase", at(jc), jh, (1, 0, 0), (0, 1, 0),
                   C_NEST, CAGE, ROD))
    drop = (kit.d / 2.0) / sin(radians(Magazine.SLOT_ANGLE / 2.0))
    for i, (jx, jy, jz) in enumerate(kit.jig_posts()):
        # the post carries the rod's centreline at jz; its V's apex sits
        # `drop` under that, the same geometry the rack's blocks use
        out.append(cylinder("b_post%d" % i, at((jx, jy, StationB.JIG_BASE_T)),
                            at((jx, jy, jz - drop)), StationB.POST_D / 2.0,
                            C_NEST, CAGE, ROD))
        out += v_flanks("b_postv%d" % i, at((jx, jy, jz - drop)),
                        np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]),
                        StationB.POST_D, kit.d / 2.0, 0.6,
                        Magazine.SLOT_ANGLE / 2.0, C_NEST, CAGE, ROD)
    # ---- the kit's own parts, as free bodies in their start poses
    for knd, i, a, b in kit.rack_slots():
        c = (np.asarray(a) + np.asarray(b)) / 2.0
        nm = "bcollar" if knd == "collar" else "bgrid%d" % i
        # THE KIT'S OWN PARTS DO NOT COLLIDE WITH EACH OTHER.  They are
        # designed touching -- layer 1 lies ON layer 0 at every crossing and
        # layer 0 lies ON the collar -- and once they are wound and bonded
        # they are one part.  Left colliding, the contact solver pushed
        # tangent rods apart while the frame's own welds pulled them back,
        # and the finished tic-tac-toe came off the jig 12 mm out of shape
        # and 25 degrees out of plane.  So drop ROD from what a kit part
        # answers to; it still collides with the jig, the nest and the head,
        # which is where a real crash would be.
        kaff = CAGE | HEAD_B
        if knd == "collar":
            from .mount import collar_segments
            segs = collar_segments(Payload, kit.d, kit.clear)
            g = "".join(
                box("%s_g%d" % (nm, k),
                    (0.5 * (s0[0] + s1[0]), 0.5 * (s0[1] + s1[1]), 0.0),
                    (max(abs(s1[0] - s0[0]) / 2.0, sw / 2.0),
                     max(abs(s1[1] - s0[1]) / 2.0, sw / 2.0),
                     Bracket.SHEET / 2.0),
                    (1, 0, 0), (0, 1, 0), C_KIT, ROD, kaff,
                    mass=Bracket.mass(Payload, kit.d) / len(segs))
                for k, (s0, s1, sw) in enumerate(segs))
        else:
            g = cylinder(nm + "_g", np.asarray(a) - c, np.asarray(b) - c, kit.d / 2.0,
                         C_ROD, ROD, kaff,
                         mass=Stock.rho_lin(kit.d) * float(np.linalg.norm(
                             np.asarray(b) - np.asarray(a))) / 1000.0)
        out.append('<body name="%s" pos="%s"><freejoint name="%s_f"/>%s</body>'
                   % (nm, _v(at(c)), nm, g))
        eq.append('    <weld name="bkeep_%s" body1="b_module" body2="%s" active="false" '
                  'solref="0.004 1" solimp="0.98 0.999 0.001"/>' % (nm, nm))
        eq.append('    <weld name="bhold_%s" body1="bgw" body2="%s" active="false" '
                  'solref="0.004 1" solimp="0.98 0.999 0.001"/>' % (nm, nm))
        eq.append('    <weld name="bvac_%s" body1="bvac" body2="%s" active="false" '
                  'solref="0.004 1" solimp="0.98 0.999 0.001"/>' % (nm, nm))
        # the JIG's own keeper: the frame is not built on the module, so
        # `bkeep_` (which welds to it) cannot hold a rod on the jig
        eq.append('    <weld name="bjig_%s" body1="world" body2="%s" active="false" '
                  'solref="0.004 1" solimp="0.98 0.999 0.001"/>' % (nm, nm))
    # ---- and the frame's own joints.  Once four crossings are wound and
    # cured the tic-tac-toe is ONE PART, which is the whole reason it can be
    # picked up by a single rod; these welds are that fact.
    for i in range(1, len(kit.rods)):
        eq.append('    <weld name="bfrm%d" body1="bgrid0" body2="bgrid%d" '
                  'active="false" solref="0.002 1" solimp="0.99 0.9999 0.001"/>'
                  % (i, i))
    # ---- the module itself, welded down: it is placed by hand and stays.
    # TWO BOXES, from `mount.payload_solid`, and not one slab -- the collar
    # SEATS ON THE BOARD'S FRONT FACE with the housing standing through its
    # aperture, so a module modelled as one 9 mm box puts the collar 1.5 mm
    # inside the part.  Built that way here, the nozzle went to dose the
    # aperture and drove into a housing that was not where the part says it
    # is.  The two machines read the module from the same function now.
    from .mount import payload_solid
    slab, hous = payload_solid(Payload)
    out.append('<body name="b_module" pos="%s">' % _v(at((0, 0, 0))))
    for nm, e, col, ms in (("b_board", slab, C_CAM, Payload.MASS),
                           ("b_hous", hous, "0.10 0.10 0.12 1", 0.3)):
        # payload_solid speaks the CAMERA's frame (x, look, up); B's is
        # (x, up, look), which is why the last two swap
        out.append("  " + box(nm,
                              (0.5 * (e[0][0] + e[0][1]),
                               0.5 * (e[2][0] + e[2][1]),
                               0.5 * (e[1][0] + e[1][1])),
                              (0.5 * (e[0][1] - e[0][0]),
                               0.5 * (e[2][1] - e[2][0]),
                               0.5 * (e[1][1] - e[1][0])),
                              (1, 0, 0), (0, 1, 0), col, CAGE, ROD, mass=ms))
    out.append("</body>")
    # ---- B's gantry and head
    ex, ey, ez = kit.extent()
    out += _station_b_head(kit, o, ex, ey, ez)
    act.append('    <position name="ba_x" joint="bx" kp="20000" kv="300" '
               'forcerange="-%g %g"/>' % (StationB.STALL_N["x"], StationB.STALL_N["x"]))
    act.append('    <position name="ba_y" joint="by" kp="20000" kv="300" '
               'forcerange="-%g %g"/>' % (StationB.STALL_N["y"], StationB.STALL_N["y"]))
    act.append('    <position name="ba_z" joint="bz" kp="20000" kv="300" '
               'forcerange="-%g %g"/>' % (StationB.STALL_N["z"], StationB.STALL_N["z"]))
    act.append('    <position name="ba_g" joint="bg" kp="2000" kv="30" '
               'forcerange="-25 25"/>')
    act.append('    <position name="ba_d" joint="bd" kp="2000" kv="30" '
               'forcerange="-25 25"/>')
    act.append('    <position name="ba_w" joint="bw" kp="0.8" kv="0.01" '
               'forcerange="-0.3 0.3"/>')
    act.append('    <position name="ba_f" tendon="bjaws" kp="300" kv="3" '
               'forcerange="-15 15"/>')
    act.append('    <velocity name="ba_ring" joint="bring" kv="0.02" '
               'forcerange="-0.5 0.5"/>')
    act.append('    <position name="ba_v" joint="bv" kp="2000" kv="30" '
               'forcerange="-25 25"/>')
    eq.append('    <joint name="bjaws_sym" joint1="bf_l" joint2="bf_r" '
              'polycoef="0 1 0 0 0"/>')
    ten = ('  <tendon>\n    <fixed name="bjaws">\n'
           '      <joint joint="bf_l" coef="1"/>\n'
           '      <joint joint="bf_r" coef="1"/>\n    </fixed>\n  </tendon>')
    return out, eq, act, ten


def _station_b_head(kit, o, ex, ey, ez):
    """B's carriage: three slides, a gripper on a stroke and a yaw, a small
    ring on a hinge, and a nozzle."""
    from .spec import StationB, Gripper, Dispenser, Payload
    z0 = float(o[2]) + kit.cruise_z() + StationB.LIFT_CLEAR
    stroke = StationB.tool_stroke(kit)
    # EVERY TOOL IS SEATED ABOVE THE RING'S OWN CENTRE.  They share a
    # carriage with it, and when the ring winds, that centre is ON the
    # crossing -- so a tool whose tip sat at the carriage's height sat at
    # the crossing's height, twenty millimetres away in x, which on this
    # kit is directly over a grid rod.  See StationB.tool_lift.
    tz = StationB.tool_lift(kit)
    r_in, r_out = StationB.ring_r_in(), StationB.ring_r_out()
    out = [
        '<body name="bx" pos="%.6f %.6f %.6f">' % (mm(o[0]), mm(o[1]), mm(z0)),
        '  <joint name="bx" type="slide" axis="1 0 0" range="%.6f %.6f" damping="4"/>'
        % (mm(-StationB.X_TRAVEL / 2.0), mm(StationB.X_TRAVEL / 2.0)),
        # a slide with no mass of its own is not a body MuJoCo will move
        '  ' + box("b_beam", (0, 0, 70.0), (8.0, StationB.Y_TRAVEL / 2.0, 8.0),
                   (1, 0, 0), (0, 1, 0), "0.3 0.3 0.33 0.4", 0, 0, mass=0.8),
        '  <body name="by" pos="0 0 0">',
        '    <joint name="by" type="slide" axis="0 1 0" range="%.6f %.6f" damping="4"/>'
        % (mm(-StationB.Y_TRAVEL / 2.0), mm(StationB.Y_TRAVEL / 2.0)),
        '    ' + box("b_sled", (0, 0, 58.0), (18.0, 18.0, 8.0),
                     (1, 0, 0), (0, 1, 0), "0.3 0.3 0.33 0.5", 0, 0, mass=0.6),
        '    <body name="bz" pos="0 0 0" gravcomp="1">',
        '      <joint name="bz" type="slide" axis="0 0 1" range="%.6f %.6f" damping="4"/>'
        % (mm(-StationB.Z_TRAVEL), 0.0),
        '      ' + box("b_head", (0, 0, 34.0), (26.0, 16.0, 8.0), (1, 0, 0), (0, 1, 0),
                       "0.35 0.35 0.38 0.4", HEAD_B, ROD | CAGE, mass=120.0),
        '      <site name="b_ring_centre" pos="0 0 0" size="0.0005"/>',
        # the winder: a gapped ring turning about B's x, mouth facing down
        # THE BORE LIES ON THE CROSSING'S DIAGONAL, not on either rod: a
        # ring can only orbit what lies along its axis, and on a rod the
        # OTHER rod is in the ring's plane.  See Kit.wind_axis.
        '      <body name="bring" pos="0 0 0" euler="0 0 %.1f" gravcomp="1">'
        % kit.wind_yaw(),
        '        <joint name="bring" type="hinge" axis="1 0 0" damping="2e-6" '
        'armature="2e-8"/>',
    ]
    n = 18
    r_mid = (r_in + r_out) / 2.0
    arc = 2 * pi * r_mid / n
    for i in range(n):
        psi = 2 * pi * (i + 0.5) / n
        if abs(((degrees(psi) + 180.0) % 360.0) - 180.0) < StationB.RING_GAP / 2.0:
            continue
        c = (0.0, r_mid * sin(psi), -r_mid * cos(psi))
        tan_ = (0.0, cos(psi), sin(psi))
        out.append('        ' + box("bring%d" % i, c,
                                    (StationB.RING_W / 2.0, arc * 0.55,
                                     (r_out - r_in) / 2.0),
                                    (1, 0, 0), tan_, C_RING, HEAD_B, ROD | CAGE,
                                    mass=0.6))
    out.append('      </body>')
    # the dispenser, outboard of the ring
    out += [
        '      <body name="bdisp" pos="%.6f %.6f %.6f" gravcomp="1">'
        % (mm(StationB.disp_off()[0]), mm(StationB.disp_off()[1]), mm(tz)),
        '        <joint name="bd" type="slide" axis="0 0 -1" range="0 %.6f" '
        'damping="2"/>' % mm(stroke),
        '        ' + cylinder("b_nozzle", (0, 0, 0), (0, 0, 16.0),
                              Dispenser.NOZZLE_D / 2.0,
                              "0.85 0.85 0.9 1", HEAD_B, ROD | CAGE, mass=3.0),
        '        <site name="b_nozzle_tip" pos="0 0 0" size="0.0005"/>',
        '      </body>',
        # the vacuum head, for the collar: FOUR PADS ON THE APERTURE RING,
        # which is the only metal at the plate's own centre.  See
        # StationB.vac_pads for why it is not a cup in the middle.
        '      <body name="bvac" pos="%.6f %.6f %.6f" gravcomp="1">'
        % (mm(StationB.vac_off()[0]), mm(StationB.vac_off()[1]), mm(tz)),
        '        <joint name="bv" type="slide" axis="0 0 -1" range="0 %.6f" '
        'damping="2"/>' % mm(stroke),
    ] + [
        '        ' + box("b_cup%d" % k, (c[0], c[1], 3.0), (h[0], h[1], 3.0),
                         (1, 0, 0), (0, 1, 0), "0.25 0.25 0.28 1",
                         HEAD_B, ROD | CAGE, mass=1.0)
        for k, (c, h) in enumerate(StationB.vac_pads(Payload))
    ] + [
        '        <site name="b_cup_tip" pos="0 0 0" size="0.0005"/>',
        '      </body>',
        # the gripper, on the other side
        '      <body name="bgrip" pos="%.6f %.6f %.6f" gravcomp="1">'
        % (mm(StationB.grip_off()[0]), mm(StationB.grip_off()[1]), mm(tz)),
        '        <joint name="bg" type="slide" axis="0 0 -1" range="0 %.6f" '
        'damping="2"/>' % mm(stroke),
        '        ' + box("b_gbody", (0, 0, 26.0), (7.0, 6.0, 8.0),
                         (1, 0, 0), (0, 1, 0), C_HEAD, HEAD_B, ROD | CAGE, mass=30.0),
        '        <body name="bgw" pos="0 0 0" gravcomp="1">',
        '          <joint name="bw" type="hinge" axis="0 0 1" range="-95 95" '
        'damping="0.002"/>',
        '          <site name="b_grip_pt" pos="0 0 0" size="0.0005"/>',
        '          ' + box("b_gplate", (0, 0, 14.0), (7.0, 7.0, 2.0), (1, 0, 0),
                           (0, 1, 0), C_HEAD, 0, 0, mass=4.0),
    ]
    for sgn, tag in ((1.0, "l"), (-1.0, "r")):
        y = sgn * (Gripper.JAW_OPEN / 2.0 + Gripper.PAD_T / 2.0)
        out += [
            '          <body name="bfinger_%s" pos="0 %.6f 0" gravcomp="1">' % (tag, mm(y)),
            # THE TRAVEL CLOSES.  With the axis the other way up the
            # finger at +y moved further +y on a negative command: the jaws
            # OPENED when the HAL told them to close, and every frame of
            # every run showed a rod between jaws standing 8 mm apart while
            # the weld did all the gripping.
            '            <joint name="bf_%s" type="slide" axis="0 %d 0" '
            'range="%.6f 0" damping="0.5"/>' % (tag, int(sgn),
                                                mm(-(Gripper.JAW_OPEN / 2.0 - 0.3))),
            '            ' + box("bpad_%s" % tag,
                                 (0, 0, Gripper.PAD_H / 2.0 - Gripper.PAD_UNDER),
                                 (Gripper.PAD_L / 2.0, Gripper.PAD_T / 2.0,
                                  Gripper.PAD_H / 2.0), (1, 0, 0), (0, 1, 0),
                                 "0.2 0.2 0.22 1", HEAD_B, ROD | CAGE, mass=2.0,
                                 extra=' friction="0.9 0.005 0.0001"'),
            '          </body>',
        ]
    out += ['        </body>', '      </body>', '    </body>', '  </body>', '</body>']
    return out


def scene_station_b(kit, timestep=5e-4, origin=(0.0, 0.0, 0.0)):
    """Station B on its own, for the rig and the demo.

    THE CELL IS 668 GEOMS AND B IS FORTY.  Stepping the whole plant to
    watch one small station make one small part costs ten minutes of wall
    clock for three of simulated time, and buys nothing: B shares the floor
    with the cell and nothing else, so a scene with only B in it is the
    same physics."""
    parts, eq, act, ten = station_b(kit, origin=origin)
    floor_z = origin[2] - 60.0
    body = ['<geom name="floor" type="plane" pos="%.6f %.6f %.6f" size="1 1 0.1" '
            'material="floor" contype="%d" conaffinity="%d"/>'
            % (mm(origin[0]), mm(origin[1]), mm(floor_z), CAGE, ROD | DROP)]
    body += parts
    return """<mujoco model="station_b">
%s
  <worldbody>
    <camera name="b_cell" pos="%.4f %.4f %.4f" xyaxes="1 0 0 0 0.5 0.87"/>
%s
  </worldbody>
  <equality>
%s
  </equality>
%s
  <actuator>
%s
  </actuator>
</mujoco>
""" % (preamble(timestep), mm(origin[0]), mm(origin[1]) - 0.16, mm(origin[2]) + 0.10,
       "\n".join("    " + p for p in body), "\n".join(eq), ten, "\n".join(act))


# ------------------------------------------------------------ assembly
def z_floor(geom, fixture):
    """Lowest ring-centre height the carriage needs: seated on the lowest
    chord it ever winds is chord-up, R; the loader needs grip_z(R/2) which
    is higher.  Leave a little under R."""
    return geom.R - 15.0


def scene_cell(geom, fixture, stage="empty", start=None, timestep=5e-4, n_drops=None,
               mount=None, kit=None):
    """The whole cell.  stage="empty": rods in their racks, welds off;
    stage="truss": the TRUSS built and welded to the cage, the MOUNT still in
    its racks -- what `rig_mount` runs against;
    "loaded": rods in the fixture, welded to the cage.  Pass `mount` (a
    truss.mount.Mount) to hang the finished camera nose on the cage."""
    t = geom.t
    n_drops = geom.t.n_joints if n_drops is None else n_drops
    z_lo = z_floor(geom, fixture)
    parts = []
    floor_z = -(geom.R + 120.0)
    parts.append('<geom name="floor" type="plane" pos="%.6f 0 %.6f" size="3 3 0.1" material="floor" '
                 'contype="%d" conaffinity="%d"/>' % (mm(t.length / 2.0), mm(floor_z), CAGE, ROD | DROP))
    parts += rack_geoms(geom, fixture)
    parts += cage_body(geom, fixture, mount=mount)
    racked = tuple(k for k in ("mbatten", "mcam", "mstrut", "mgrid")) \
        if stage == "truss" else ()
    parts += (rods_in_racks(geom, fixture) if stage == "empty"
              else rods_in_fixture(geom, fixture, racked))
    parts += head_body(geom, fixture, z_lo, start=start)
    parts += drops(n_drops)
    b_eq, b_act, b_ten = [], [], None
    if kit is not None:
        b_parts, b_eq, b_act, b_ten = station_b(kit)
        parts += b_parts
    # ---- equality: a weld per rod to the cage (the keeper) and to the
    # gripper's yaw body (the grip), both inactive unless loaded
    eq = ["  <equality>",
          # a parallel gripper's jaws move together by a rack and pinion:
          # a rigid coupling, not a shared tendon (with the tendon alone a
          # blocked pad let the other run past the rod)
          '    <joint name="jaws_sym" joint1="gf_l" joint2="gf_r" polycoef="0 1 0 0 0"/>']
    for r in geom.all_rods:
        if r.kind == "mcam" and (stage == "empty" or r.kind in racked):
            # A DETENT IN THE NEST.  The carrier is a 24 mm slab on a 9 mm
            # base with a 14 mm pin out of one side; sat loose in a pocket
            # it walks under the gripper's own approach and ends up tilted
            # across the grip point -- measured, twice.  A printed nest for
            # a part like this has a snap detent, and this is it: released
            # the instant the jaws close on the boss.
            eq.append('    <weld name="rack%d" body1="world" body2="rod%d" '
                      'active="true" solref="0.004 1" solimp="0.98 0.999 0.001"/>'
                      % (r.index, r.index))
        eq.append('    <weld name="keep%d" body1="cage" body2="rod%d" active="%s" '
                  'solref="0.004 1" solimp="0.98 0.999 0.001"/>'
                  % (r.index, r.index,
                     "true" if stage in ("loaded", "truss")
                     and r.kind not in racked else "false"))
        eq.append('    <weld name="hold%d" body1="gripw" body2="rod%d" active="false" '
                  'solref="0.004 1" solimp="0.98 0.999 0.001"/>' % (r.index, r.index))
    for i in range(n_drops):
        eq.append('    <weld name="stick%d" body1="cage" body2="drop%d" active="false" '
                  'solref="0.004 1" solimp="0.98 0.999 0.001"/>' % (i, i))
    eq += b_eq
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
%s  </actuator>""" .replace("%s", "\n".join(b_act) + ("\n" if b_act else "")) % (Gantry.STALL_N["x"], Gantry.STALL_N["x"], Gantry.STALL_N["y"], Gantry.STALL_N["y"],
                    Gantry.STALL_N["z"], Gantry.STALL_N["z"], Ring.servo_kp(),
                    Ring.servo_kv(), Ring.DRIVE_TORQUE, Ring.DRIVE_TORQUE)
    ten = """  <tendon>
    <fixed name="jaws">
      <joint joint="gf_l" coef="1"/>
      <joint joint="gf_r" coef="1"/>
    </fixed>
  </tendon>"""
    if b_ten:
        ten = ten[:-len("  </tendon>")] + b_ten[len("  <tendon>\n"):]
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
