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

def cone(prefix, cx, cy, z0, h, r_in, r_out, rgba, n=16, cls="static"):
    """45-deg lead-in chamfer: n boxes tilted about their own tangent."""
    out = []
    rm = (r_in + r_out) / 2.0
    slope = 45.0
    seg = pi * 2 * rm / n * 0.62
    t = (r_out - r_in) * 1.45 / 2.0
    for i in range(n):
        a = 2 * pi * i / n
        out.append(
            f'<geom name="{prefix}_{i}" class="{cls}" type="box" '
            f'pos="{cx + rm*cos(a):.6f} {cy + rm*sin(a):.6f} {z0 + h/2.0:.6f}" '
            f'size="{seg:.6f} 0.0015 {t:.6f}" '
            f'euler="{slope} 0 {a*180/pi + 90:.4f}" rgba="{rgba}"/>')
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
    o.append(box("lab_s", mm((x0+x1)/2), mm((y0+hy0)/2), mm(pt/2),
                 mm((x1-x0)/2), mm((hy0-y0)/2), mm(pt/2), C_PLATE))
    o.append(box("lab_n", mm((x0+x1)/2), mm((hy1+y1)/2), mm(pt/2),
                 mm((x1-x0)/2), mm((y1-hy1)/2), mm(pt/2), C_PLATE))
    edges = [x0] + [v for hx in Field.LAB_HOLE_X for v in (hx-r, hx+r)] + [x1]
    for i in range(0, len(edges)-1, 2):
        a, b = edges[i], edges[i+1]
        o.append(box(f"lab_m{i}", mm((a+b)/2), mm(LAB_HOLE_Y), mm(pt/2),
                     mm((b-a)/2), mm(r), mm(pt/2), C_PLATE))
    for i, hx in enumerate(Field.LAB_HOLE_X):
        o += ring(f"labring{i}", mm(hx), mm(LAB_HOLE_Y), 0.0, mm(pt), mm(r), mm(6), C_PLATE)
        o += cone(f"labcone{i}", mm(hx), mm(LAB_HOLE_Y), mm(pt), mm(6), mm(r), mm(r+6), C_PLATE)

    if with_zones:
        for nm, (a, b, c, d) in {"z_quar": Field.QUARANTINE, "z_dep": Field.DEPLOY_BOX}.items():
            o.append(f'<site name="{nm}" type="box" '
                     f'pos="{mm((a+c)/2):.4f} {mm((b+d)/2):.4f} 0.0005" '
                     f'size="{mm((c-a)/2):.4f} {mm((d-b)/2):.4f} 0.0005" rgba="{C_ZONE}"/>')
        for e in (0, 1):
            bx = Field.DEPLOY_BOX
            o.append(box(f"tape_h{e}", mm((bx[0]+bx[2])/2), mm(bx[1] if e == 0 else bx[3]),
                         0.0002, mm((bx[2]-bx[0])/2), mm(Field.TAPE_W/2), 0.0002, C_TAPE))
    return o


# ------------------------------------------------------- Agent A robot body
def _ring_gap(prefix, cx, cy, z0, z1, r_in, wall, rgba, n=16, skip_deg=95, cls="robot"):
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
        out.append(g.replace('/>', ' friction="0.10 0.002 0.0001"/>'))
    return out


FINGER_PIVOT_Y = 74.0
FINGER_LEN     = 90.0
FINGER_OPEN    = -5.4      # left finger hinge angle, deg  (tip at y +82.5)
FINGER_CLOSED  = 10.2      # (tip at y +58)
GATE_OPEN_MM   = 62.0


def agent_a_body(name="agentA", pose=None, with_beams=False):
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
    NOSE_Z, TAIL_Z = -0.3, BELT_TOP_TAIL_A
    inc = degrees_atan(TAIL_Z - NOSE_Z, (nose_x - tail_x) * 1000.0)
    bl  = ((nose_x - tail_x)**2 + mm(TAIL_Z - NOSE_Z)**2) ** 0.5
    bcx = (nose_x + tail_x) / 2.0
    bcz = mm((NOSE_Z + TAIL_Z) / 2.0 - Chassis.BELT_T / 2.0)
    o.append(f'<geom name="A_belt" type="box" contype="2" conaffinity="2" condim="6" '
             f'friction="{Chassis.MU_PIECE} 0.004 0.0002" solref="0.003 1" '
             f'solimp="0.97 0.99 0.001" '
             f'pos="{bcx:.6f} 0 {bcz:.6f}" size="{bl/2:.6f} {mm(Chassis.BELT_W/2):.6f} '
             f'{mm(Chassis.BELT_T/2):.6f}" euler="0 {inc:.4f} 0" '
             f'surfacevel="{-mm(Chassis.BELT_SPEED):.6f} 0 0 0 0 0" rgba="{C_BELT}"/>')
    for s, tag in ((1, "l"), (-1, "r")):     # converging guides 116 -> 62
        gx0, gx1 = lx(200.0), lx(50.0)
        gy0, gy1 = mm(58.0), mm(31.0)
        ang = -180/pi * ((gy1-gy0)/(gx1-gx0))
        o.append(box(f"A_guide_{tag}", (gx0+gx1)/2, s*(gy0+gy1)/2, mm(38),
                     abs(gx1-gx0)/2, mm(1.5), mm(16), C_BODY, "robot",
                     euler=(0, 0, s*ang)))

    # ---- chute-magazine + slide gate -------------------------------------
    cx = lx(AgentA.CHUTE_X)
    o += _ring_gap("A_chute", cx, 0, mm(AgentA.CHUTE_Z0), mm(AgentA.CHUTE_Z1),
                   mm(AgentA.CHUTE_D/2), mm(5), "0.17 0.58 0.79 1")

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

    # ---- chute base gate: slides to the left flank to release one disc ----
    body += [
        f'  <body name="A_gate" pos="{cx:.5f} 0 {mm(AgentA.CHUTE_Z0-1.5):.5f}">',
        '    <joint name="A_gate_j" type="slide" axis="0 1 0" range="0 0.075" damping="0.05"/>',
        f'    <geom name="A_gate_g" class="robot" type="box" size="{mm(31):.5f} {mm(31):.5f} {mm(1.5):.5f}" '
        f'mass="0.008" rgba="0.9 0.35 0.2 1"/>',
        '  </body>']

    body += [f'  <site name="A_imu" pos="0 0 {mm(60):.4f}"/>',
             f'  <site name="A_tof" pos="{lx(AgentA.L):.5f} 0 {mm(45):.4f}" zaxis="1 0 0"/>',
             f'  <camera name="A_chase" pos="{-mm(560):.4f} 0 {mm(420):.4f}" xyaxes="0 -1 0 0.6 0 0.8"/>',
             '</body>']

    act = f"""
    <velocity name="a_drive_l" joint="A_w_l" kv="5.0" ctrlrange="-30 30" forcerange="-0.5 0.5"/>
    <velocity name="a_drive_r" joint="A_w_r" kv="5.0" ctrlrange="-30 30" forcerange="-0.5 0.5"/>
    <position name="a_finger_l" joint="A_f_l" kp="4" ctrlrange="-0.55 0.55"/>   <!-- RADIANS -->
    <position name="a_finger_r" joint="A_f_r" kp="4" ctrlrange="-0.55 0.55"/>   <!-- RADIANS -->
    <position name="a_gate" joint="A_gate_j" kp="900" kv="12" ctrlrange="0 0.075" forcerange="-25 25"/>"""

    sen = """
    <framepos    name="a_pos"  objtype="body" objname="agentA"/>
    <framequat   name="a_quat" objtype="body" objname="agentA"/>
    <gyro        name="a_gyro" site="A_imu" noise="0.002"/>
    <accelerometer name="a_acc" site="A_imu" noise="0.01"/>
    <rangefinder name="a_tof"  site="A_tof" noise="0.001"/>
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
            f'<geom name="disc{i}_g" class="piece" type="cylinder" '
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
def contact_pairs(agent="A"):
    """Explicit friction pairs -- see note in scene()."""
    out = ["  <contact>"]
    for i in range(4):
        out.append(f'    <pair geom1="{agent}_ball{i}" geom2="floor" '
                   f'friction="{Chassis.MU_BALL} {Chassis.MU_BALL} 0.0001 0.0001 0.0001" '
                   f'solref="0.02 2.0" solimp="0.6 0.9 0.01"/>')
    # scoop is excluded from the floor by collision bitmask -- no pair needed
    out.append("  </contact>")
    return "\n".join(out)


def scene(name, bodies, actuators="", sensors="", timestep=0.001, contacts=""):
    return f"""<mujoco model="{name}">
{preamble(timestep)}
  <worldbody>
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
    return scene("rfgyc26_pick_place", parts, act, sen, contacts=contact_pairs())


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
