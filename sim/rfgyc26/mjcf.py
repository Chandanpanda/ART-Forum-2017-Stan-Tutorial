"""Generate MJCF from params.py.  Nothing here is hand-tuned geometry --
every number comes from rfgyc26.params, so the model cannot drift from the
drawings.  Emits self-contained scene files (no <include>) so each one opens
standalone in MuJoCo's simulate.exe.

Agent A local frame:  +x forward, +y left, +z up, origin at the drive-axle
centre projected onto the floor.  local_x = Xa - 142.5,  local_y = Ya - 117.5.
"""
from math import cos, sin, radians, pi, atan2, degrees

def degrees_atan(rise, run):
    return degrees(atan2(rise, run))
from .params import Field, Piece, Chassis, AgentA, mm, BELT_TOP_TAIL_A

# R11 (new): the Explainer's hole centre Y 372 puts a O60 hole 18 mm outside a
# plate spanning Y 360-510.  400 is the nearest value that keeps the bore fully
# inside the 150-deep plate.  [VERIFY on the field mock-up]
LAB_HOLE_Y = 400.0

AX, AY = AgentA.AXLE_X, AgentA.W / 2.0        # 142.5, 117.5
def lx(Xa): return mm(Xa - AX)
def ly(Ya): return mm(Ya - AY)


# ------------------------------------------------------------------ helpers
def box(name, cx, cy, cz, sx, sy, sz, rgba, cls="static", euler=None, mass=None):
    e = f' euler="{euler[0]} {euler[1]} {euler[2]}"' if euler else ""
    m = f' mass="{mass}"' if mass is not None else ""
    return (f'<geom name="{name}" class="{cls}" type="box" '
            f'pos="{cx:.6f} {cy:.6f} {cz:.6f}" size="{sx:.6f} {sy:.6f} {sz:.6f}"'
            f'{e}{m} rgba="{rgba}"/>')

def wall(name, p0, p1, thick, h, rgba, cls="robot", extra=""):
    """A guide wall between two 3-D points, standing perpendicular to the belt.

    The walls have to CLIMB with the belt.  Built at a fixed height they float
    above the pieces near the intake -- the belt top is at Za 0 there and rises
    28 mm -- and a disc simply passes underneath, out to the belt edge, where it
    jams (measured: one sample per match left stranded at Xa 223, y -49).
    x-axis runs along the wall, y-axis is the horizontal normal; MuJoCo derives
    the third, so there is no euler composition to get wrong.
    """
    dx, dy, dz = (p1[i]-p0[i] for i in range(3))
    L = (dx*dx + dy*dy + dz*dz) ** 0.5
    nx, ny = -dy, dx
    nl = (nx*nx + ny*ny) ** 0.5 or 1.0
    return (f'<geom name="{name}" class="{cls}" type="box" '
            f'pos="{(p0[0]+p1[0])/2:.6f} {(p0[1]+p1[1])/2:.6f} '
            f'{(p0[2]+p1[2])/2 + h/2:.6f}" '
            f'size="{L/2:.6f} {thick/2:.6f} {h/2:.6f}" '
            f'xyaxes="{dx/L:.6f} {dy/L:.6f} {dz/L:.6f} {nx/nl:.6f} {ny/nl:.6f} 0" '
            f'rgba="{rgba}"{extra}/>')


def ring(prefix, cx, cy, z0, z1, r_in, wall, rgba, n=16, cls="static"):
    """Vertical annulus from n boxes -- a round bore out of primitives."""
    out, rm, hz = [], r_in + wall / 2.0, (z1 - z0) / 2.0
    arc = pi * 2 * rm / n
    for i in range(n):
        a = 2 * pi * i / n
        out.append(box(f"{prefix}_{i}", cx + rm*cos(a), cy + rm*sin(a), z0 + hz,
                       arc*0.62, wall/2.0, hz, rgba, cls,
                       euler=(0, 0, a*180/pi + 90.0)))
    return out

def cone(prefix, cx, cy, z0, r_in, r_out, rgba, n=24, cls="static", thick=0.0006,
         skip_deg=0.0, height=None):
    """45-degree lead-in chamfer, built from tangential boxes oriented with xyaxes.

    The previous version composed three euler angles and got the composition
    wrong: the segments stuck out ~30 mm past where they belonged, and the robot
    crashed into them at 33 N.  Specifying the box's x-axis (tangential) and
    y-axis (up the cone slope) directly removes all ambiguity -- MuJoCo derives z.

    The cone rises at 45 degrees, so its height equals (r_out - r_in).  Keep that
    small: when Agent A is docked, its chute-base gate sits at Za 8 and its rear
    wall at Za 6 directly above this ring, so anything taller is a collision.
    """
    out = []
    CONE_GEOMS[prefix] = []
    for i in range(n):
        ang = 2*pi*i/n
        if abs(((ang*180/pi + 180) % 360) - 180) < skip_deg:
            continue                      # leave the feed side open
        CONE_GEOMS[prefix].append(f"{prefix}_{i}")
        ca, sa = cos(ang), sin(ang)
        dr = r_out - r_in
        dz = dr if height is None else height  # default is a 45 deg chamfer
        L  = (dr*dr + dz*dz) ** 0.5
        rm = (r_in + r_out) / 2.0
        zm = z0 + dz / 2.0
        tx, ty = -sa, ca                       # tangential
        ux, uy, uz = ca*dr/L, sa*dr/L, dz/L    # up the slope
        half_t = pi*rm/n * 1.15                # tangential half-length, slight overlap
        half_s = L / 2.0
        out.append(
            f'<geom name="{prefix}_{i}" class="{cls}" type="box" '
            f'pos="{cx + rm*ca:.6f} {cy + rm*sa:.6f} {zm:.6f}" '
            f'size="{half_t:.6f} {half_s:.6f} {thick:.6f}" '
            f'xyaxes="{tx:.6f} {ty:.6f} 0 {ux:.6f} {uy:.6f} {uz:.6f}" '
            f'friction="0.08 0.002 0.0001" rgba="{rgba}"/>')
    return out


C_WOOD, C_DISC = "0.78 0.66 0.47 1", "0.85 0.72 0.50 1"
C_BODY, C_BELT = "0.31 0.37 0.84 1", "0.07 0.63 0.55 1"
C_WALL, C_PLATE, C_TAPE = "0.62 0.60 0.55 1", "0.76 0.81 0.87 1", "0.15 0.15 0.15 1"
C_ZONE = "0.80 0.86 0.80 0.35"


# ------------------------------------------------------------------ preamble
def preamble(timestep=0.001):
    return f"""  <compiler angle="degree" autolimits="true"/>
  <option timestep="{timestep}" integrator="implicitfast" cone="elliptic"
          impratio="3" gravity="0 0 -9.81"/>
  <size njmax="4000" nconmax="1500"/>
  <visual>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.65 0.65 0.65"/>
    <quality shadowsize="2048"/>
    <map znear="0.01" zfar="30"/>
  </visual>
  <default>
    <default class="static">
      <geom condim="3" friction="{Chassis.MU_PIECE} 0.005 0.0001"
            solref="0.004 1" solimp="0.95 0.99 0.001" contype="3" conaffinity="3"/>
    </default>
    <default class="piece">
      <geom condim="6" friction="{Chassis.MU_PIECE} 0.004 0.0002"
            solref="0.004 1" solimp="0.95 0.99 0.001" contype="3" conaffinity="3"/>
    </default>
    <default class="robot">
      <geom condim="3" friction="0.5 0.005 0.0001"
            solref="0.004 1" solimp="0.95 0.99 0.001" contype="3" conaffinity="3"/>
    </default>
    <default class="ball">
      <geom condim="3" friction="{Chassis.MU_BALL} 0.0001 0.0001"
            solref="0.004 1" solimp="0.95 0.99 0.001"/>
    </default>
    <default class="belt">
      <geom condim="6" friction="{Chassis.MU_PIECE} 0.004 0.0002"
            solref="0.003 1" solimp="0.97 0.99 0.001"/>
    </default>
  </default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.94 0.94 0.92"
             rgb2="0.88 0.88 0.86" width="300" height="300"/>
    <material name="floor" texture="grid" texrepeat="12 12" reflectance="0.05"/>
  </asset>"""


# --------------------------------------------------------------------- field
def _lab(g):
    """Plate geometry collides with game pieces, and with the robot only when
    Field.LAB_SOLID is set (see the note in params)."""
    bits = ' contype="5" conaffinity="5"' if Field.LAB_SOLID \
      else ' contype="4" conaffinity="4"'
    return g.replace('/>', bits + '/>')


def field_body(with_zones=True):
    o, W, H, t, h = [], Field.W, Field.H, Field.WALL_T, Field.WALL_H
    # contype/conaffinity bit 1 is the floor alone: the scoop is excluded from it
    # (bit 2 only) so its knife edge can sit below the floor plane and wedge under a
    # 5 mm disc.  Two rigid bodies both bottoming at z=0 can never slide under each
    # other; a real 0.5 shim works because its edge is below the disc's under-face.
    o.append(f'<geom name="floor" type="plane" contype="1" conaffinity="1" '
             f'condim="3" friction="{Chassis.MU_PIECE} 0.005 0.0001" '
             f'solref="0.004 1" solimp="0.95 0.99 0.001" '
             f'pos="{mm(W/2):.4f} {mm(H/2):.4f} 0" size="3 3 0.1" material="floor"/>')
    for nm, (cx, cy, sx, sy) in {
        "wall_s": (W/2, -t/2, W/2 + t, t/2),
        "wall_n": (W/2, H + t/2, W/2 + t, t/2),
        "wall_w": (-t/2, H/2, t/2, H/2),
        "wall_e": (W + t/2, H/2, t/2, H/2)}.items():
        o.append(box(nm, mm(cx), mm(cy), mm(h/2), mm(sx), mm(sy), mm(h/2), C_WALL))

    # lab plate: strips around three square cut-outs, each bored round + chamfered
    x0, y0, x1, y1 = Field.LAB_PLATE
    pt, r = Field.LAB_PLATE_T, Field.LAB_HOLE_D / 2.0
    hy0, hy1 = LAB_HOLE_Y - r, LAB_HOLE_Y + r
    o.append(_lab(box("lab_s", mm((x0+x1)/2), mm((y0+hy0)/2), mm(pt/2),
                 mm((x1-x0)/2), mm((hy0-y0)/2), mm(pt/2), C_PLATE)))
    o.append(_lab(box("lab_n", mm((x0+x1)/2), mm((hy1+y1)/2), mm(pt/2),
                 mm((x1-x0)/2), mm((y1-hy1)/2), mm(pt/2), C_PLATE)))
    edges = [x0] + [v for hx in Field.LAB_HOLE_X for v in (hx-r, hx+r)] + [x1]
    for i in range(0, len(edges)-1, 2):
        a, b = edges[i], edges[i+1]
        o.append(_lab(box(f"lab_m{i}", mm((a+b)/2), mm(LAB_HOLE_Y), mm(pt/2),
                          mm((b-a)/2), mm(r), mm(pt/2), C_PLATE)))
    # F11: the robot cannot REVERSE up a square 3 mm plate edge -- the O20 ball
    # transfers stall on the step and the dock halts 80-320 mm short.  This is the
    # spec's own [VERIFY 10.2] question, answered: the plate needs a ramped or
    # taped edge.  Modelled as a 12 mm ramp on the approach (south) edge.
    ramp_l = Field.LAB_EDGE_RAMP
    if ramp_l > 0:
        o.append(_lab(box("lab_ramp", mm((x0+x1)/2), mm(y0 - ramp_l/2), mm(pt/2),
                     mm((x1-x0)/2), mm(ramp_l/2), mm(pt/2), C_PLATE,
                     euler=(-14.0, 0, 0))))
    for i, hx in enumerate(Field.LAB_HOLE_X):
        o += [_lab(g) for g in ring(f"labring{i}", mm(hx), mm(LAB_HOLE_Y), 0.0, mm(pt), mm(r), mm(6), C_PLATE)]
        # 45 deg lead-in chamfer.  ASSUMED, not specified: the rulebook supplies
        # the laboratory as a wooden part with plain 60 mm slots (F21).  Capped
        # at r+4 -> 4 mm tall, because the docked robot's gate sits at Za 8 and
        # its rear wall at Za 6 right above this ring.
        if Field.LAB_CHAMFER > 0:
            o += [_lab(g) for g in cone(f"labcone{i}", mm(hx), mm(LAB_HOLE_Y),
                                        mm(pt), mm(r), mm(r + Field.LAB_CHAMFER),
                                        C_PLATE)]

    if with_zones:
        for nm, (a, b, c, d) in {"z_quar": Field.QUARANTINE, "z_dep": Field.DEPLOY_BOX}.items():
            o.append(f'<site name="{nm}" type="box" '
                     f'pos="{mm((a+c)/2):.4f} {mm((b+d)/2):.4f} 0.0005" '
                     f'size="{mm((c-a)/2):.4f} {mm((d-b)/2):.4f} 0.0005" rgba="{C_ZONE}"/>')
        # 20 mm boundary tape is an OPTICAL marker, not an obstacle -- it is
        # adhesive film, ~0.1 mm.  As a collision geom its 0.4 mm step caught the
        # O20 ball transfers at 8-10 N and stalled the robot dead on the box
        # boundary (this is what was killing the hole-3 dock).  Visual only; the
        # TCRT line array reads it by position, not by contact.
        for e in (0, 1):
            bx = Field.DEPLOY_BOX
            g = box(f"tape_h{e}", mm((bx[0]+bx[2])/2), mm(bx[1] if e == 0 else bx[3]),
                    0.0002, mm((bx[2]-bx[0])/2), mm(Field.TAPE_W/2), 0.0002, C_TAPE)
            o.append(g.replace('/>', ' contype="0" conaffinity="0"/>'))
    return o


# ------------------------------------------------------- Agent A robot body
CHUTE_GEOMS = []
CONE_GEOMS = {}          # prefix -> emitted geom names (cone() skips some arcs)


def _ring_gap(prefix, cx, cy, z0, z1, r_in, wall, rgba, n=16, skip_deg=0, cls="robot"):
    """Chute bore with the forward arc left open where the belt feeds it."""
    out, rm, hz = [], r_in + wall/2.0, (z1-z0)/2.0
    arc = pi*2*rm/n
    for i in range(n):
        a = 2*pi*i/n
        d = abs(((a*180/pi + 180) % 360) - 180)       # angle from +x
        if d < skip_deg:
            continue
        g = box(f"{prefix}_{i}", cx + rm*cos(a), cy + rm*sin(a), z0+hz,
                arc*0.62, wall/2.0, hz, rgba, cls, euler=(0, 0, a*180/pi + 90.0))
        CHUTE_GEOMS.append(f"{prefix}_{i}")
        out.append(g.replace('/>', ' friction="0.10 0.002 0.0001"/>'))
    return out


FINGER_PIVOT_Y = 74.0
FINGER_LEN     = 90.0
FINGER_OPEN    = AgentA.FINGER_OPEN     # left finger hinge angle, deg
FINGER_CLOSED  = AgentA.FINGER_RAKE
GATE_OPEN_MM   = 62.0


def agent_a_body(name="agentA", pose=None, with_beams=False):
    del CHUTE_GEOMS[:]
    """Returns (xml, actuator_xml, sensor_xml). pose = (field_x, field_y, heading_deg)."""
    px, py, ph = pose or AgentA.START_POSE
    b, o = [], []
    zc = mm(Chassis.GROUND_CLEAR)

    # ---- structure -------------------------------------------------------
    o.append(box("A_deck", 0, 0, mm(96.5), mm(AgentA.L/2), mm(AgentA.W/2), mm(1.5), C_BODY, "robot"))
    o.append(box("A_rear", lx(0)-mm(1.5), 0, mm(50), mm(1.5), mm(AgentA.W/2), mm(44), C_BODY, "robot"))
    for s, tag in ((1, "l"), (-1, "r")):
        o.append(box(f"A_side_{tag}", 0, s*mm(AgentA.W/2), mm(50),
                     mm(AgentA.L/2), mm(1.5), mm(44), C_BODY, "robot"))
        # beam-pocket inner wall, open-bottomed, full length
        # open-bottomed pocket: walls start at the ground-clearance line
        pz0 = Chassis.GROUND_CLEAR
        o.append(box(f"A_pocket_{tag}", 0, s*mm(93.5), mm((pz0+AgentA.POCKET_H)/2),
                     mm(AgentA.L/2), mm(1.5), mm((AgentA.POCKET_H-pz0)/2), C_BODY, "robot"))

    # ---- belt ------------------------------------------------------------
    # MODELLING DECISION: one continuous conveyor from the scoop tip (top surface
    # 0.3 BELOW the floor plane, so it wedges under a 5 mm disc) all the way to the
    # tail roller.  The real machine splits this into a 0.5 shim plus a belt whose
    # nose sits at Z 17.5, because a O16 roller cannot reach the floor -- but a
    # PASSIVE shim provably cannot convey the piece across that 65 mm gap (see the
    # Findings section of the README).  Treating the shim as an extension of the
    # belt is the simplest faithful stand-in for the sweeper fingers' active stroke.
    nose_x, tail_x = lx(AgentA.SCOOP_FROM), lx(AgentA.BELT_TAIL_X)
    NOSE_Z, TAIL_Z = Chassis.BELT_NOSE_Z, BELT_TOP_TAIL_A
    inc = degrees_atan(TAIL_Z - NOSE_Z, (nose_x - tail_x) * 1000.0)
    bl  = ((nose_x - tail_x)**2 + mm(TAIL_Z - NOSE_Z)**2) ** 0.5
    bcx = (nose_x + tail_x) / 2.0
    bcz = mm((NOSE_Z + TAIL_Z) / 2.0 - Chassis.BELT_T / 2.0)
    def belt_top(xa):
        f = (AgentA.SCOOP_FROM - min(xa, AgentA.SCOOP_FROM)) / \
            (AgentA.SCOOP_FROM - AgentA.BELT_TAIL_X)
        return NOSE_Z + min(f, 1.0) * (TAIL_Z - NOSE_Z)   # flat aft of the tail
    o.append(f'<geom name="A_belt" type="box" contype="2" conaffinity="2" condim="6" '
             f'friction="{Chassis.MU_PIECE} 0.004 0.0002" solref="0.003 1" '
             f'solimp="0.97 0.99 0.001" '
             f'pos="{bcx:.6f} 0 {bcz:.6f}" size="{bl/2:.6f} {mm(Chassis.BELT_W/2):.6f} '
             f'{mm(Chassis.BELT_T/2):.6f}" euler="0 {inc:.4f} 0" '
             f'surfacevel="{-mm(Chassis.BELT_SPEED):.6f} 0 0 0 0 0" rgba="{C_BELT}"/>')
    # F8: spec 7 converges 116 -> 62 gradually over Xa 200->50.  Three O56 discs
    # then bridge: two fit abreast on a 116 belt, and a gradual taper gives them
    # 75 mm of un-guided belt to bunch in.  Converge HARD right at the intake so
    # the channel is single-file (62 wide, 3 mm clearance each side) from the tip.
    # Guides climb with the belt (see wall()).  Converge HARD at the intake so
    # the channel is single-file from the tip: spec 7's gradual 116 -> 62 over
    # Xa 200->50 gives three O56 discs 75 mm of un-guided belt to bunch in, and
    # two of them fit abreast on a 116 belt (F8).
    GUIDE_END_X = 195.0                    # Xa where the taper finishes
    GUIDE_TOP_X = 272.0                    # Xa where it starts
    GH = 16.0                              # wall height above the belt face
    # The belt face is at Za 0.4 up here, so a wall foot 1 mm under it would sit
    # BELOW the floor plane -- and these are robot-class geoms, so they plough.
    # Every mission timed out until this clamp went in.
    def foot(xa): return mm(max(belt_top(xa) - 1.0, 2.0))
    for s_, tag in ((1, "l"), (-1, "r")):
        p0 = (lx(GUIDE_TOP_X), s_*mm(AgentA.GUIDE_FROM_W/2), foot(GUIDE_TOP_X))
        p1 = (lx(GUIDE_END_X), s_*mm(AgentA.GUIDE_TO_W/2), foot(GUIDE_END_X))
        p2 = (lx(AgentA.BELT_TAIL_X), s_*mm(AgentA.GUIDE_TO_W/2),
              foot(AgentA.BELT_TAIL_X))
        o.append(wall(f"A_guide_{tag}", p0, p1, mm(1.5), mm(GH), C_BODY))
        o.append(wall(f"A_lane_{tag}",  p1, p2, mm(1.5), mm(GH), C_BODY))

    # ---- hold-down strip over the belt tail (F16) -------------------------
    if AgentA.HOLD_GAP0 > 0:
        ha = (AgentA.HOLD_FROM, belt_top(AgentA.HOLD_FROM) + AgentA.HOLD_GAP0
              + AgentA.HOLD_T/2)
        hb = (AgentA.HOLD_TO,   belt_top(AgentA.HOLD_TO)   + AgentA.HOLD_GAP1
              + AgentA.HOLD_T/2)
        hinc = degrees_atan(ha[1]-hb[1], hb[0]-ha[0])
        hl   = ((hb[0]-ha[0])**2 + (ha[1]-hb[1])**2) ** 0.5
        o.append(box("A_hold", lx((ha[0]+hb[0])/2), 0, mm((ha[1]+hb[1])/2),
                     mm(hl/2), mm(AgentA.HOLD_W/2), mm(AgentA.HOLD_T/2),
                     "0.80 0.80 0.84 0.5", "robot",
                     euler=(0, hinc, 0)).replace(
                     '/>', ' friction="0.10 0.002 0.0001"/>'))

    # ---- chute-magazine + slide gate -------------------------------------
    cx = lx(AgentA.CHUTE_X)
    # Full bore now: its top (Za 41) sits below the belt underside (47.5), so a
    # disc tips off the tail and falls in from ABOVE.  While the bore extended past
    # the belt the ring blocked discs still riding it; with the front left open
    # instead, discs landed on the gate and rolled straight back out.
    o += _ring_gap("A_chute", cx, 0, mm(AgentA.CHUTE_Z0), mm(AgentA.CHUTE_Z1),
                   mm(AgentA.CHUTE_D/2), mm(5), "0.17 0.58 0.79 1")
    # 45 deg lead-in at the bore MOUTH.  A disc tipping off the belt tail lands up
    # to ~11 mm off-axis when the magazine is partly full; against a bare rim it
    # simply sat there (the third disc never stacked, and shook loose in transit).
    # Capped at r+5 -> 5 mm tall, clearing the belt underside at Za 47.5.
    # REAR ARC ONLY.  The belt slopes, so at the bore's front edge its underside is
    # only Za 43.8 -- a full ring here punches through the belt and stops every
    # piece riding it.  Discs overshoot the axis rearward (~11 mm), so the rear
    # half is the half that has to catch them anyway.
    # raised rear collar: the piece's aft stop, and what centres it (see params)
    if AgentA.CHUTE_Z2 > AgentA.CHUTE_Z1:
        o += _ring_gap("A_collar", cx, 0, mm(AgentA.CHUTE_Z1), mm(AgentA.CHUTE_Z2),
                       mm(AgentA.CHUTE_D/2), mm(AgentA.CHUTE_COLLAR_T),
                       "0.17 0.58 0.79 1", skip_deg=AgentA.LEAD_SKIP)
    if AgentA.LEAD_H > 0:
        o += cone("A_chutelead", cx, 0, mm(max(AgentA.CHUTE_Z1, AgentA.CHUTE_Z2)),
                  mm(AgentA.CHUTE_D/2), mm(AgentA.LEAD_R),
                  "0.17 0.58 0.79 1", cls="robot", skip_deg=AgentA.LEAD_SKIP,
                  height=mm(AgentA.LEAD_H), thick=0.0005)

    # ---- ball transfers ---------------------------------------------------
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        o.append(f'<geom name="A_ball{i}" class="ball" type="sphere" '
                 f'pos="{sx*mm(102.5):.5f} {sy*mm(80.0):.5f} {mm(10.0):.5f}" '
                 f'size="{mm(10):.5f}" rgba="0.4 0.4 0.42 1"/>')

    body = [f'<body name="{name}" pos="{mm(px):.5f} {mm(py):.5f} 0" euler="0 0 {ph}">',
            '  <freejoint name="A_free"/>',
            f'  <inertial pos="{mm(10):.4f} 0 {mm(45):.4f}" mass="{g(AgentA.MASS) if False else AgentA.MASS/1000.0:.4f}" '
            f'diaginertia="0.0159 0.0207 0.0252"/>']
    body += ["  " + s for s in o]

    # ---- wheels -----------------------------------------------------------
    for s, tag in ((1, "l"), (-1, "r")):
        body += [
            f'  <body name="A_wheel_{tag}" pos="0 {s*mm(Chassis.TRACK/2):.5f} {mm(Chassis.WHEEL_D/2):.5f}">',
            f'    <joint name="A_w_{tag}" type="hinge" axis="0 1 0" limited="false" damping="0.002"/>',
            f'    <geom name="A_wg_{tag}" class="robot" type="cylinder" zaxis="0 1 0" '
            f'size="{mm(Chassis.WHEEL_D/2):.5f} {mm(Chassis.WHEEL_COLLISION_W/2):.5f}" '
            f'mass="0.06" friction="1.2 0.005 0.0002" rgba="0.15 0.15 0.17 1"/>',
            f'    <geom name="A_wv_{tag}" type="cylinder" zaxis="0 1 0" contype="0" conaffinity="0" '
            f'size="{mm(Chassis.WHEEL_D/2):.5f} {mm(Chassis.WHEEL_W/2):.5f}" '
            f'mass="0" rgba="0.15 0.15 0.17 0.55"/>',
            '  </body>']

    # ---- scoop ------------------------------------------------------------
    # Height is set by the hinge's lower LIMIT (spring holds it there), not by
    # floor contact, so it can never dig in.  It keeps the 25 deg up-trip that
    # lets it skate over the 20 tape and the 3 lab plate in reverse.
    sx0, sz0 = lx(AgentA.SCOOP_TO), mm(9.5)
    TIP_Z  = -0.5                                    # top surface, below the floor
    run    = AgentA.SCOOP_FROM - AgentA.SCOOP_TO
    rise   = Chassis.BELT_TOP_NOSE - TIP_Z
    ang    = degrees_atan(rise, run)
    plate_l= (run**2 + rise**2) ** 0.5
    midx   = (AgentA.SCOOP_TO + AgentA.SCOOP_FROM)/2.0
    midz   = (Chassis.BELT_TOP_NOSE + TIP_Z)/2.0 - AgentA.SCOOP_T/2.0
    body += [
        f'  <body name="A_scoop" pos="{sx0:.5f} 0 {sz0:.5f}">',
        '    <joint name="A_scoop_j" type="hinge" axis="0 1 0" range="-25 0"'
        '           stiffness="0.30" springref="0" damping="0.004"/>',
        # FINDING (sim): a PASSIVE shim cannot convey the disc.  After ~19 mm of
        # engagement a 5 mm disc has left the floor entirely (5/tan15 deg), yet its
        # rear edge is still ~46 mm short of the belt nose, and nothing then moves
        # it -- it simply rides along with the robot.  Reaching a nose 17.5 mm up
        # while staying floor-supported would need a <=4.4 deg ramp, which the O16
        # roller forbids.  The sweeper fingers' active ~110 deg stroke (spec 5) is
        # therefore load-bearing, not optional.  Modelled here as surfacevel on the
        # shim (an extension of the belt); simulating the finger stroke in contact
        # is the honest next step.  See README "Findings".
        f'    <geom name="A_scoop_g" type="box" contype="0" conaffinity="0" condim="3" '
        f'surfacevel="{-mm(Chassis.BELT_SPEED):.6f} 0 0 0 0 0" '
        f'friction="0.40 0.002 0.0001" solref="0.004 1" solimp="0.95 0.99 0.001" '
        f'pos="{mm(midx - AgentA.SCOOP_TO):.5f} 0 {mm(midz - 9.5):.5f}" '
        f'size="{mm(plate_l/2):.5f} {mm(Chassis.BELT_W/2):.5f} {mm(AgentA.SCOOP_T/2):.5f}" '
        f'euler="0 {ang:.4f} 0" mass="0.018" rgba="0.72 0.72 0.75 1"/>',
        f'    <geom name="A_scoop_lip" type="cylinder" zaxis="0 1 0" contype="0" conaffinity="0" '
        f'condim="3" friction="0.40 0.002 0.0001" solref="0.004 1" solimp="0.95 0.99 0.001" '
        f'pos="{mm(run):.5f} 0 {mm(TIP_Z - 0.75 - 9.5):.5f}" '
        f'size="{mm(0.75):.5f} {mm(Chassis.BELT_W/2):.5f}" mass="0.002" rgba="0.62 0.62 0.66 1"/>',
        '  </body>']

    # ---- sweeper fingers ---------------------------------------------------
    for s, tag in ((1, "l"), (-1, "r")):
        body += [
            f'  <body name="A_finger_{tag}" pos="{lx(AgentA.SWEEP_PIVOT_X):.5f} '
            f'{s*mm(FINGER_PIVOT_Y):.5f} {mm(14.5):.5f}">',
            f'    <joint name="A_f_{tag}" type="hinge" axis="0 0 1" range="-30 30" damping="0.01"/>',
            f'    <geom name="A_fg_{tag}" class="robot" type="box" '
            f'pos="{-mm(FINGER_LEN/2):.5f} 0 0" '
            f'size="{mm(FINGER_LEN/2):.5f} {mm(2):.5f} {mm(12.5):.5f}" mass="0.03" '
            f'rgba="0.85 0.55 0.2 1"/>',
            '  </body>']

    # ---- positive feed: plunger on the bore axis (see params, F18) --------
    body += [
        f'  <body name="A_feed" pos="{lx(AgentA.FEED_X):.5f} 0 '
        f'{mm(AgentA.FEED_Z_UP):.5f}">',
        f'    <joint name="A_feed_j" type="slide" axis="0 0 1" '
        f'range="{-mm(AgentA.FEED_STROKE):.5f} 0" damping="0.06"/>',
        f'    <geom name="A_feed_g" class="robot" type="cylinder" '
        f'size="{mm(AgentA.FEED_D/2):.5f} {mm(1.5):.5f}" mass="0.012" '
        f'friction="0.10 0.002 0.0001" rgba="0.90 0.55 0.20 1"/>',
        '  </body>']

    # ---- chute base gate: slides to the left flank to release one disc ----

    esc = mm(AgentA.ESC_Y)
    half = mm(AgentA.CHUTE_D/2 + 4)
    body += [
        f'  <body name="A_gate" pos="{cx:.5f} 0 {mm(AgentA.CHUTE_Z0-1.5):.5f}">',
        f'    <joint name="A_gate_j" type="slide" axis="0 1 0" '
        f'range="0 {esc:.5f}" damping="0.05"/>',
        f'    <geom name="A_gate_g" class="robot" type="box" '
        f'size="{half:.5f} {half:.5f} {mm(AgentA.ESC_T):.5f}" '
        f'mass="0.008" rgba="0.9 0.35 0.2 1"/>',
        '  </body>',
        # retainer: parked clear of the bore at +ESC_Y, driven to 0 to hold the
        # column while the shelf is out from under it
        f'  <body name="A_blade" pos="{cx:.5f} {esc:.5f} '
        f'{mm(AgentA.ESC_BLADE_Z):.5f}">',
        f'    <joint name="A_blade_j" type="slide" axis="0 1 0" '
        f'range="{-esc:.5f} 0" damping="0.05"/>',
        f'    <geom name="A_blade_g" class="robot" type="box" '
        f'size="{half:.5f} {mm(AgentA.ESC_BLADE_Y):.5f} '
        f'{mm(AgentA.ESC_BLADE_T):.5f}" mass="0.005" rgba="0.9 0.55 0.2 1"/>',
        f'    <geom name="A_blade_lip" class="robot" type="cylinder" '
        f'zaxis="1 0 0" pos="0 {mm(AgentA.ESC_BLADE_Y):.5f} '
        f'{mm(AgentA.ESC_LIP_Z - AgentA.ESC_BLADE_Z):.5f}" '
        f'size="{mm(AgentA.ESC_LIP_R):.5f} {half:.5f}" mass="0.001" '
        f'rgba="0.9 0.55 0.2 1"/>',
        '  </body>']

    body += [f'  <site name="A_mag" pos="{cx:.5f} 0 {mm(70):.4f}" zaxis="0 0 -1"/>',
             f'  <site name="A_imu" pos="0 0 {mm(60):.4f}"/>',
             f'  <site name="A_tof" pos="{lx(AgentA.L):.5f} 0 {mm(45):.4f}" zaxis="1 0 0"/>',
             f'  <camera name="A_chase" pos="{-mm(560):.4f} 0 {mm(420):.4f}" xyaxes="0 -1 0 0.6 0 0.8"/>',
             '</body>']

    fs = -mm(AgentA.FEED_STROKE)
    nesc = -esc
    act = f"""
    <velocity name="a_drive_l" joint="A_w_l" kv="5.0" ctrlrange="-30 30" forcerange="-0.5 0.5"/>
    <velocity name="a_drive_r" joint="A_w_r" kv="5.0" ctrlrange="-30 30" forcerange="-0.5 0.5"/>
    <position name="a_finger_l" joint="A_f_l" kp="4" ctrlrange="-0.55 0.55"/>   <!-- RADIANS -->
    <position name="a_finger_r" joint="A_f_r" kp="4" ctrlrange="-0.55 0.55"/>   <!-- RADIANS -->
    <position name="a_gate" joint="A_gate_j" kp="900" kv="12" ctrlrange="0 {esc:.5f}" forcerange="-25 25"/>
    <position name="a_blade" joint="A_blade_j" kp="600" kv="10" ctrlrange="{nesc:.5f} 0" forcerange="-12 12"/>
    <position name="a_feed" joint="A_feed_j" kp="120" kv="5" ctrlrange="{fs:.5f} 0" forcerange="-4.0 4.0"/>"""

    sen = """
    <framepos    name="a_pos"  objtype="body" objname="agentA"/>
    <framequat   name="a_quat" objtype="body" objname="agentA"/>
    <gyro        name="a_gyro" site="A_imu" noise="0.002"/>
    <accelerometer name="a_acc" site="A_imu" noise="0.01"/>
    <rangefinder name="a_tof"  site="A_tof" noise="0.001"/>
    <!-- looks down the bore from Za 70, ON the axis: the feed plunger parks
         ABOVE the site so it is never in the ray, and a centred ray still hits
         a piece that has drifted a few mm (off-axis at r 25 it missed a disc
         sitting 3.4 mm off centre and read an empty magazine).  5 mm of range
         per piece is how the robot knows how many are left, and so whether the
         escapement needs its retainer at all. -->
    <rangefinder name="a_mag"  site="A_mag" noise="0.0005"/>
    <jointvel    name="a_wvel_l" joint="A_w_l"/>
    <jointvel    name="a_wvel_r" joint="A_w_r"/>
    <actuatorfrc name="a_frc_l" actuator="a_drive_l"/>
    <actuatorfrc name="a_frc_r" actuator="a_drive_r"/>"""
    return "\n".join(body), act, sen


# ------------------------------------------------------------------ pieces
def disc_body(i, x, y, z=None):
    z = z if z is not None else Piece.DISC_T/2 + 0.5
    return (f'<body name="disc{i}" pos="{mm(x):.5f} {mm(y):.5f} {mm(z):.5f}">'
            f'<freejoint name="disc{i}_f"/>'
            f'<geom name="disc{i}_g" class="piece" contype="7" conaffinity="7" type="cylinder" '
            f'size="{mm(Piece.DISC_D/2):.5f} {mm(Piece.DISC_T/2):.5f}" '
            f'mass="{Piece.DISC_M/1000.0:.4f}" rgba="{C_DISC}"/></body>')

def cyl_body(i, x, y, colour):
    rgba = {"red": "0.82 0.24 0.24 1", "yellow": "0.88 0.70 0.20 1",
            "green": "0.25 0.62 0.32 1"}[colour]
    return (f'<body name="cyl{i}" pos="{mm(x):.5f} {mm(y):.5f} {mm(Piece.CYL_H/2+0.5):.5f}">'
            f'<freejoint name="cyl{i}_f"/>'
            f'<geom name="cyl{i}_g" class="piece" type="cylinder" '
            f'size="{mm(Piece.CYL_D/2):.5f} {mm(Piece.CYL_H/2):.5f}" '
            f'mass="{Piece.CYL_M/1000.0:.4f}" rgba="{rgba}"/></body>')

def beam_body(i, x, y, length, heading_deg, mass):
    return (f'<body name="beam{i}" pos="{mm(x):.5f} {mm(y):.5f} {mm(Piece.BEAM_H/2+0.5):.5f}" '
            f'euler="0 0 {heading_deg}">'
            f'<freejoint name="beam{i}_f"/>'
            f'<geom name="beam{i}_g" class="piece" type="box" '
            f'size="{mm(length/2):.5f} {mm(Piece.BEAM_W/2):.5f} {mm(Piece.BEAM_H/2):.5f}" '
            f'mass="{mass/1000.0:.4f}" rgba="0.85 0.55 0.20 1"/></body>')


# ------------------------------------------------------------------ scenes
def contact_pairs(agent="A", n_discs=3):
    """Explicit friction pairs -- see note in scene()."""
    out = ["  <contact>"]
    for i in range(4):
        out.append(f'    <pair geom1="{agent}_ball{i}" geom2="floor" '
                   f'friction="{Chassis.MU_BALL} {Chassis.MU_BALL} 0.0001 0.0001 0.0001" '
                   f'solref="0.02 2.0" solimp="0.6 0.9 0.01"/>')
    # scoop is excluded from the floor by collision bitmask -- no pair needed
    # The gate must slide OUT FROM UNDER the disc.  MuJoCo takes the element-wise
    # MAX of the two geoms' friction, so a smooth gate against a 0.6 piece still
    # gets 0.6 -- and the disc simply rode the gate out of the chute.
    for i in range(n_discs):
        out.append(f'    <pair geom1="{agent}_feed_g" geom2="disc{i}_g" '
                   f'friction="0.06 0.06 0.0005 0.0001 0.0001" '
                   f'solref="0.004 1" solimp="0.95 0.99 0.001"/>')
        for gg in ("gate_g", "blade_g", "blade_lip"):
            out.append(f'    <pair geom1="{agent}_{gg}" geom2="disc{i}_g" '
                       f'friction="0.04 0.04 0.0005 0.0001 0.0001" '
                       f'solref="0.004 1" solimp="0.95 0.99 0.001"/>')
        # Converging guides are printed PETG: slippery.  Combined at the floor's
        # 0.6 the discs jammed across the throat instead of single-filing -- the
        # spec's own "#1 jam risk", reproduced.
        for g in ("guide_l", "guide_r", "lane_l", "lane_r"):
            out.append(f'    <pair geom1="{agent}_{g}" geom2="disc{i}_g" '
                       f'friction="0.08 0.08 0.001 0.0001 0.0001"/>')
        for nm in CHUTE_GEOMS + CONE_GEOMS.get("A_chutelead", []):
            out.append(f'    <pair geom1="{nm}" geom2="disc{i}_g" '
                       f'friction="0.08 0.08 0.001 0.0001 0.0001"/>')

    out.append("  </contact>")
    return "\n".join(out)


# Fixed viewpoints.  The free camera opens looking at the whole 2000x1000 field,
# which from a distance makes the robot read as a single box -- press [ or ] in
# the viewer to cycle onto these instead.
FIELD_CAMS = """
    <camera name="field"  pos="1.00 -0.55 1.75" xyaxes="1 0 0 0 1 0.75"/>
    <camera name="lab"    pos="0.57 -0.10 0.55" xyaxes="1 0 0 0 0.7 0.7"/>
    <camera name="quar"   pos="0.20 -0.25 0.45" xyaxes="1 0 0 0 0.6 0.8"/>
"""


def scene(name, bodies, actuators="", sensors="", timestep=0.001, contacts=""):
    return f"""<mujoco model="{name}">
{preamble(timestep)}
  <worldbody>{FIELD_CAMS}
{chr(10).join('    ' + b for b in bodies)}
  </worldbody>
{contacts}
  <actuator>{actuators}
  </actuator>
  <sensor>{sensors}
  </sensor>
</mujoco>
"""


def scene_pick_place(disc_positions, robot_pose=None, with_beams=False):
    body, act, sen = agent_a_body(pose=robot_pose, with_beams=with_beams)
    parts = field_body() + [body]
    for i, (x, y) in enumerate(disc_positions):
        parts.append(disc_body(i, x, y))
    if with_beams:
        parts.append(beam_body(1, 0, 0, Piece.BEAM1_L, 0, Piece.BEAM1_M))
        parts.append(beam_body(2, 0, 0, Piece.BEAM2_L, 0, Piece.BEAM2_M))
    return scene("rfgyc26_pick_place", parts, act, sen,
                 contacts=contact_pairs(n_discs=len(disc_positions)))


def scene_belt_rig(n_pieces=4, incline=None, speed=None):
    """Tier 2 rig: a bare inclined belt, side walls and a gate -- the
    accumulation experiment, isolated from everything else."""
    inc = incline if incline is not None else Chassis.BELT_INCLINE
    v   = speed   if speed   is not None else Chassis.BELT_SPEED
    L, Wd = 0.192, mm(Chassis.BELT_W)
    parts = [f'<geom name="floor" class="static" type="plane" pos="0 0 -0.25" '
             f'size="2 2 .1" material="floor"/>',
             f'<body name="rig" euler="0 {-inc} 0">',
             f'  <geom name="belt" class="belt" type="box" size="{L/2:.4f} {Wd/2:.4f} 0.002" '
             f'surfacevel="{mm(v):.5f} 0 0 0 0 0" rgba="{C_BELT}"/>',
             f'  <geom name="gate" class="static" type="box" pos="{L/2+0.002:.4f} 0 0.017" '
             f'size="0.002 {Wd/2:.4f} 0.015" rgba="0.9 0.35 0.2 1"/>',
             f'  <geom name="wallL" class="static" type="box" pos="0 {Wd/2+0.002:.4f} 0.017" '
             f'size="{L/2:.4f} 0.002 0.015" rgba="{C_WALL}"/>',
             f'  <geom name="wallR" class="static" type="box" pos="0 {-Wd/2-0.002:.4f} 0.017" '
             f'size="{L/2:.4f} 0.002 0.015" rgba="{C_WALL}"/>',
             '</body>']
    # placement verified against the standalone rig: a 5 g cylinder is carried
    # up the incline at 59.2 mm/s against a 60 mm/s belt, and four queued at a
    # closed gate hold at 0.131 N (spec 3.2 predicts 0.116 N).
    th = radians(-inc)
    for i in range(n_pieces):
        lxp, lzp = 0.020 + i*0.021, 0.012
        wx = lxp*cos(th) + lzp*sin(th)
        wz = -lxp*sin(th) + lzp*cos(th)
        parts.append(
            f'<body name="p{i}" pos="{wx:.5f} 0 {wz:.5f}" euler="0 {-inc} 0">'
            f'<freejoint name="p{i}_f"/>'
            f'<geom name="p{i}_g" class="piece" type="cylinder" '
            f'size="{mm(Piece.CYL_D/2):.5f} {mm(Piece.CYL_H/2):.5f}" '
            f'mass="{Piece.CYL_M/1000.0:.4f}" rgba="0.25 0.62 0.32 1"/></body>')
    return scene("rfgyc26_belt_rig", parts, timestep=0.0005)
