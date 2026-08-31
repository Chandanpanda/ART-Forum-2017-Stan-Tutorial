"""Generate MJCF from params.py.  Nothing here is hand-tuned geometry --
every number comes from rfgyc26.params, so the model cannot drift from the
drawings.  Emits self-contained scene files (no <include>) so each one opens
standalone in MuJoCo's simulate.exe.

Agent A local frame:  +x forward, +y left, +z up, origin at the drive-axle
centre projected onto the floor.  local_x = Xa - 142.5,  local_y = Ya - 117.5.
"""
from math import cos, sin, radians, pi, atan, atan2, degrees

def degrees_atan(rise, run):
    return degrees(atan2(rise, run))
from .params import Field, Piece, Chassis, AgentA, M2, Vision, mm, BELT_TOP_TAIL_A

# R11 (new): the Explainer's hole centre Y 372 puts a O60 hole 18 mm outside a
# plate spanning Y 360-510.  400 is the nearest value that keeps the bore fully
# inside the 150-deep plate.  [VERIFY on the field mock-up]
LAB_HOLE_Y = 400.0

def local_to_world(px, py, heading_deg, lx_mm, ly_mm):
    """Robot-frame (mm) -> field-frame (mm).  Used to spawn the carried beams
    exactly where their pockets are, so nothing is hand-placed."""
    t = radians(heading_deg)
    return (px + lx_mm*cos(t) - ly_mm*sin(t), py + lx_mm*sin(t) + ly_mm*cos(t))


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


# The outer shell.  Everything interesting -- belt, guides, hold-down, chute,
# collar, escapement, feed plunger -- lives INSIDE these six plates, so from any
# useful camera angle they hide the whole machine.  set_xray() fades them.
SHELL_GEOMS = ["A_deck", "A_rear", "A_side_l", "A_side_r",
               "A_sidef_l", "A_sidef_r", "A_pocket_l", "A_pocket_r",
               "A_pocketf_l", "A_pocketf_r"]
XRAY_ALPHA = 0.10


def set_xray(m, on, alpha=XRAY_ALPHA):
    """Fade the chassis plates so the mechanism inside is visible.

    Purely a rendering change -- geom_rgba has no effect on contact, so the
    physics of an x-rayed run is bit-identical to a solid one.
    """
    import mujoco
    for name in SHELL_GEOMS:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        if g >= 0:
            m.geom_rgba[g][3] = alpha if on else 1.0
    return on


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
        # Countersink, INSIDE the plate: it rises from the underside to the top
        # face, never above it, so nothing can catch on it (see params).
        ch = min(Field.LAB_CHAMFER, pt)
        if ch > 0:
            o += [_lab(g) for g in cone(f"labcone{i}", mm(hx), mm(LAB_HOLE_Y),
                                        mm(pt - ch), mm(r), mm(r + ch),
                                        C_PLATE, height=mm(ch))]

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


def _finger_set(tip_r, width, tag, rgba, indent="      "):
    """The silicone fingers, as bodies that bend (F72).

    Each is hinged at the hub with a torsional spring, so it deflects when it
    meets something and springs back -- which is the whole reason a brush
    roller works on pieces of different heights.  With ROLL_FINGERS off they
    revert to F64's decoration and the rigid drum does the work, so the two
    models can be run against each other rather than argued about.
    """
    out, n = [], AgentA.FING_N
    hub, L = AgentA.FING_HUB_R, tip_r - AgentA.FING_HUB_R
    for k in range(n):
        ph = k * (360.0 / n)
        c, s_ = cos(radians(ph)), sin(radians(ph))
        if not AgentA.ROLL_FINGERS:
            r0 = mm((hub + tip_r) / 2.0)
            out.append(
                f'{indent}<geom type="box" contype="0" conaffinity="0" '
                f'pos="{r0*c:.5f} 0 {r0*s_:.5f}" euler="0 {-ph:.1f} 0" '
                f'size="{mm(L/2):.5f} {mm(width/2-2):.5f} 0.0015" '
                f'mass="0" rgba="{rgba}"/>')
            continue
        out += [
            f'{indent}<body name="A_f{tag}{k}" pos="{mm(hub)*c:.5f} 0 {mm(hub)*s_:.5f}" '
            f'euler="0 {-ph:.1f} 0">',
            f'{indent}  <joint name="A_f{tag}{k}_j" type="hinge" axis="0 1 0" '
            f'range="-75 75" stiffness="{AgentA.FING_K}" damping="{AgentA.FING_DAMP}" '
            f'armature="1e-7"/>',
            # BIT 3, WHICH ONLY GAME PIECES CARRY.  A finger is a separate
            # BODY, and MuJoCo's <exclude> is body-level -- so putting sixteen
            # of them on the intake's own bit 2 had them smashing into the knife
            # shim, which is excluded from the DRUM but knows nothing about its
            # fingers.  Discs went from 5 of 5 to 0 of 5.  A bit of their own
            # says what is actually meant: fingers touch pieces, nothing else.
            f'{indent}  <geom name="A_f{tag}{k}_g" type="box" contype="8" conaffinity="8" '
            f'condim="3" friction="1.10 0.02 0.0002" '
            f'solref="0.012 1" solimp="0.80 0.92 0.004" '
            f'pos="{mm(L/2):.5f} 0 0" '
            f'size="{mm(L/2):.5f} {mm(width/2-2):.5f} {mm(AgentA.FING_T/2):.5f}" '
            f'mass="0.0015" rgba="{rgba}"/>',
            f'{indent}</body>']
    return out


def agent_a_body(name="agentA", pose=None, with_beams=False):
    del CHUTE_GEOMS[:]
    """Returns (xml, actuator_xml, sensor_xml). pose = (field_x, field_y, heading_deg)."""
    px, py, ph = pose or AgentA.START_POSE
    b, o = [], []
    zc = mm(Chassis.GROUND_CLEAR)

    # ---- structure -------------------------------------------------------
    o.append(box("A_deck", 0, 0, mm(96.5), mm(AgentA.L/2),
                 mm(AgentA.POCKET_IN_Y), mm(1.5), C_BODY, "robot"))
    # Aft of Chassis.TAIL_X the shell is stepped up to TAIL_CLEAR: that stretch
    # overhangs the laboratory while the robot posts, and GROUND_CLEAR does not
    # clear a wooden structure (F31/F32).
    tz0, gz0, top = Chassis.TAIL_CLEAR, Chassis.GROUND_CLEAR, 94.0
    o.append(box("A_rear", lx(0)-mm(1.5), 0, mm((tz0+top)/2),
                 mm(1.5), mm(AgentA.POCKET_IN_Y), mm((top-tz0)/2), C_BODY, "robot"))
    # (the rear wall is aft of TAIL_X by definition, so it always takes the step)
    # Split into an aft (stepped-up) and a forward section ONLY when there is
    # actually a step.  With TAIL_CLEAR == GROUND_CLEAR the split would be two
    # geoms describing one plate, and even that reshuffles the contact ordering
    # enough to move a chaotic mission's score -- so emit one geom.
    stepped = Chassis.TAIL_CLEAR > Chassis.GROUND_CLEAR
    # F44: the shell's flanks stop at the pocket line, not at the 235 envelope.
    # The outer 20 mm of each side IS the beam.  Leaving the old side plates on
    # AgentA.W/2 put them exactly where the carried beam rides, and the robot
    # stalled on its own cargo 56 mm short of the wall.
    sw = AgentA.POCKET_IN_Y - 2.0
    for s, tag in ((1, "l"), (-1, "r")):
        if stepped:
            o.append(box(f"A_side_{tag}", (lx(0)+lx(Chassis.TAIL_X))/2, s*mm(sw),
                         mm((tz0+top)/2), (lx(Chassis.TAIL_X)-lx(0))/2, mm(1.5),
                         mm((top-tz0)/2), C_BODY, "robot"))
            o.append(box(f"A_sidef_{tag}", (lx(Chassis.TAIL_X)+lx(AgentA.L))/2,
                         s*mm(sw), mm((gz0+top)/2),
                         (lx(AgentA.L)-lx(Chassis.TAIL_X))/2, mm(1.5),
                         mm((top-gz0)/2), C_BODY, "robot"))
        else:
            o.append(box(f"A_side_{tag}", 0, s*mm(AgentA.W/2), mm((gz0+top)/2),
                         mm(AgentA.L/2), mm(1.5), mm((top-gz0)/2), C_BODY, "robot"))
        # BEAM POCKET INNER WALL (F44).  This is the ONLY fixed structure in the
        # pocket: outboard of it there is a 22 mm channel whose outer face is
        # the beam itself, so the loaded envelope is exactly 235 and the robot
        # stops 20 mm inboard of the beam it carries.  A boxed pocket with an
        # outboard wall does not fit the field -- see the note in params.
        pw = AgentA.POCKET_IN_Y
        if stepped:
            o.append(box(f"A_pocket_{tag}", (lx(0)+lx(Chassis.TAIL_X))/2, s*mm(pw),
                         mm((tz0+AgentA.POCKET_H)/2), (lx(Chassis.TAIL_X)-lx(0))/2,
                         mm(1.5), mm((AgentA.POCKET_H-tz0)/2), C_BODY, "robot"))
            o.append(box(f"A_pocketf_{tag}", (lx(Chassis.TAIL_X)+lx(AgentA.L))/2,
                         s*mm(pw), mm((gz0+AgentA.POCKET_H)/2),
                         (lx(AgentA.L)-lx(Chassis.TAIL_X))/2, mm(1.5),
                         mm((AgentA.POCKET_H-gz0)/2), C_BODY, "robot"))
        else:
            o.append(box(f"A_pocket_{tag}", 0, s*mm(pw),
                         mm((gz0+AgentA.POCKET_H)/2), mm(AgentA.L/2), mm(1.5),
                         mm((AgentA.POCKET_H-gz0)/2), C_BODY, "robot"))
    # END STOPS.  These are what actually push a beam into its wall: a plate
    # across the channel, behind beam 1 (so driving forward at heading 180
    # presses it west) and ahead of beam 2 (so reversing at heading 90 presses
    # it south).  Release is simply backing the stop off -- once the cradles
    # are down, nothing else in the pocket touches the piece.
    for sy, tag, sx in ((1, "1", AgentA.STOP1_X), (-1, "2", AgentA.STOP2_X)):
        # The stop's FACE lands on STOPn_X, which is the beam's own end face --
        # so the plate body sits one half-thickness outboard of it.  Centred on
        # STOPn_X instead, it overlaps the beam it is welded to by 2 mm and the
        # solver fights that at 1630 N for the whole match.
        sx = sx + (AgentA.STOP_T if sx > 0 else -AgentA.STOP_T)
        o.append(box(f"A_stop{tag}", lx(AgentA.AXLE_X + sx), sy*mm(AgentA.POCKET_Y),
                     mm(AgentA.STOP_Z0 + AgentA.STOP_H), mm(AgentA.STOP_T), mm(AgentA.STOP_W),
                     mm(AgentA.STOP_H), "0.85 0.35 0.20 1", "robot"))

    # ---- belt ------------------------------------------------------------
    # THE BELT NOW STARTS WHERE A BELT PHYSICALLY CAN (F64): at its nose
    # roller, Xa BELT_NOSE_X with its top face at spec 3.1's 17.5.  The old
    # model ran this box all the way to the scoop tip with its face 3 mm off
    # the floor and surfacevel doing the conveying -- a powered surface no
    # mechanism produced, and the simulator's single biggest stand-in (F1).
    # Engagement and conveyance below the nose now belong to mechanisms that
    # exist: the knife shim and the brush roller, built further down.
    # surfacevel on THIS box is honest -- it is exactly what a driven belt's
    # moving surface is.
    nose_x, tail_x = lx(AgentA.BELT_NOSE_X), lx(AgentA.BELT_TAIL_X)
    NOSE_Z, TAIL_Z = Chassis.BELT_TOP_NOSE, BELT_TOP_TAIL_A
    inc = degrees_atan(TAIL_Z - NOSE_Z, (nose_x - tail_x) * 1000.0)
    bl  = ((nose_x - tail_x)**2 + mm(TAIL_Z - NOSE_Z)**2) ** 0.5
    bcx = (nose_x + tail_x) / 2.0
    bcz = mm((NOSE_Z + TAIL_Z) / 2.0 - Chassis.BELT_T / 2.0)
    def belt_top(xa):
        """Top of the conveying surface at Xa: shim plane forward of the belt
        nose, belt plane aft of it, flat aft of the tail (guides and the
        hold-down strip are built off this)."""
        if xa > AgentA.SHIM_TIP_X:
            return AgentA.SHIM_T
        if xa > AgentA.SHIM_HINGE_X:
            f = (AgentA.SHIM_TIP_X - xa) / (AgentA.SHIM_TIP_X - AgentA.SHIM_HINGE_X)
            return AgentA.SHIM_T + f * (AgentA.SHIM_HINGE_Z + AgentA.SHIM_T/2
                                        - AgentA.SHIM_T)
        f = (AgentA.BELT_NOSE_X - min(xa, AgentA.BELT_NOSE_X)) / \
            (AgentA.BELT_NOSE_X - AgentA.BELT_TAIL_X)
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
    GUIDE_END_X = AgentA.GUIDE_END_X       # Xa where the taper finishes
    GUIDE_TOP_X = AgentA.GUIDE_TOP_X       # Xa where it starts
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

    # ---- chute-magazine, on the F68 trim slide ---------------------------
    # Everything from here to the lead-in belongs to the POSTING HEAD, which is
    # a body of its own on a lateral slide -- so it is collected in `head`, not
    # in the chassis's own geom list.
    head = []
    cx = lx(AgentA.CHUTE_X)
    # Full bore now: its top (Za 41) sits below the belt underside (47.5), so a
    # disc tips off the tail and falls in from ABOVE.  While the bore extended past
    # the belt the ring blocked discs still riding it; with the front left open
    # instead, discs landed on the gate and rolled straight back out.
    head += _ring_gap("A_chute", cx, 0, mm(AgentA.CHUTE_Z0), mm(AgentA.CHUTE_Z1),
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
        head += _ring_gap("A_collar", cx, 0, mm(AgentA.CHUTE_Z1), mm(AgentA.CHUTE_Z2),
                          mm(AgentA.CHUTE_D/2), mm(AgentA.CHUTE_COLLAR_T),
                          "0.17 0.58 0.79 1", skip_deg=AgentA.LEAD_SKIP)
    # SEAT TAPER (F70).  The bore is O66 for a O56 disc -- 5 mm of play, which
    # F9 put there because a disc tipping off the belt tail into a O58 bore
    # overshoots and jams on the rim.  That clearance is spent twice: once
    # catching the piece, and again at the dock, where the piece lands wherever
    # it happens to be sitting rather than on the bore's axis.  Measured at the
    # instant the escapement fires: 1.5, 3.9 and 4.6 mm off the axis, on top of
    # whatever the dock itself is off by.
    #
    # A taper in the LAST 8 mm gets both: O66 where the piece arrives, closing
    # to O59.5 where it rests, so a disc settles onto the shelf centred and
    # stays there.  It is a funnel, and it is the reason a funnel exists.
    if AgentA.SEAT_H > 0:
        head += cone("A_seat", cx, 0, mm(AgentA.CHUTE_Z0),
                     mm(AgentA.SEAT_R), mm(AgentA.CHUTE_D/2),
                     "0.17 0.58 0.79 1", cls="robot",
                     height=mm(AgentA.SEAT_H), thick=0.0008)
    if AgentA.LEAD_H > 0:
        head += cone("A_chutelead", cx, 0, mm(max(AgentA.CHUTE_Z1, AgentA.CHUTE_Z2)),
                  mm(AgentA.CHUTE_D/2), mm(AgentA.LEAD_R),
                  "0.17 0.58 0.79 1", cls="robot", skip_deg=AgentA.LEAD_SKIP,
                  height=mm(AgentA.LEAD_H), thick=0.0005)

    # ---- ball transfers ---------------------------------------------------
    # Skid lips ahead of the rear balls, so a ball meeting the laboratory edge on
    # a diagonal approach rides up instead of stopping dead (F35).
    if Chassis.SKID_LEN > 0 and Chassis.TAIL_CLEAR > Chassis.SKID_Z0:
        rise = Chassis.TAIL_CLEAR - Chassis.SKID_Z0
        ang  = degrees_atan(rise, Chassis.SKID_LEN)
        ln   = (Chassis.SKID_LEN**2 + rise**2) ** 0.5
        for sy, tag in ((1, "l"), (-1, "r")):
            o.append(box(f"A_skid_{tag}",
                         lx(Chassis.BALL_REAR_X + Chassis.SKID_LEN/2),
                         sy*mm(Chassis.BALL_Y), mm((Chassis.SKID_Z0+Chassis.TAIL_CLEAR)/2),
                         mm(ln/2), mm(Chassis.SKID_W/2), mm(0.8), C_BODY, "robot",
                         euler=(0, ang, 0)))

    for i, (bx, sy) in enumerate(((Chassis.BALL_FRONT_X,  1), (Chassis.BALL_FRONT_X, -1),
                                  (Chassis.BALL_REAR_X,   1), (Chassis.BALL_REAR_X,  -1))):
        o.append(f'<geom name="A_ball{i}" class="ball" type="sphere" '
                 f'pos="{lx(bx):.5f} {sy*mm(Chassis.BALL_Y):.5f} {mm(Chassis.BALL_D/2):.5f}" '
                 f'size="{mm(Chassis.BALL_D/2):.5f}" rgba="0.4 0.4 0.42 1"/>')

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

    # ---- knife shim (F64: engagement) --------------------------------------
    # A 0.5 spring-steel plate hinged over the belt nose, tip floating on the
    # field.  The step it presents to a disc is its own thickness plus a O1 tip
    # lip -- under the 2.5 mm half-thickness bar by construction.  The hinge is
    # SERVO-driven, not sprung: ctrl +SHIM_DROOP presses the tip onto the floor
    # (the force cap on the actuator IS the preload), ctrl -SHIM_LIFT swings the
    # tip 27 mm up for transit, because a floor-pressed knife bulldozes every
    # already-placed disc it crosses.  Collision: the plate rides bit 2 only
    # (pieces; never the floor, so the edge can stand under a disc's under-face
    # -- two rigid bodies both bottoming at z=0 can never slide under each
    # other); the tip lip alone also carries bit 0 so the FLOOR sets the
    # working height, not the chassis.  Chassis contacts are excluded by pair.
    s_run  = AgentA.SHIM_TIP_X - AgentA.SHIM_HINGE_X
    s_drop = AgentA.SHIM_HINGE_Z - AgentA.SHIM_T          # plate mid: hinge->tip
    s_len  = (s_run**2 + s_drop**2) ** 0.5
    s_ang  = degrees_atan(s_drop, s_run)
    SHIM_W = AgentA.SHIM_W                # narrower than the mouth -- see params
    body += [
        f'  <body name="A_shim" pos="{lx(AgentA.SHIM_HINGE_X):.5f} 0 '
        f'{mm(AgentA.SHIM_HINGE_Z):.5f}">',
        f'    <joint name="A_shim_j" type="hinge" axis="0 1 0" '
        f'range="{-AgentA.SHIM_LIFT:.1f} {AgentA.SHIM_DROOP:.1f}" damping="0.005"/>',
        f'    <geom name="A_shim_g" type="box" contype="4" conaffinity="4" condim="3" '
        f'friction="0.30 0.002 0.0001" solref="0.004 1" solimp="0.95 0.99 0.001" '
        f'pos="{mm(s_run/2):.5f} 0 {mm(-s_drop/2):.5f}" '
        f'size="{mm(s_len/2):.5f} {mm(SHIM_W/2):.5f} {mm(AgentA.SHIM_T/2):.5f}" '
        f'euler="0 {s_ang:.4f} 0" mass="0.028" rgba="0.72 0.72 0.75 1"/>',
        # The cutting edge.  A first build gave the knife a O1 rolled lip
        # standing ON the floor, and it BULLDOZED every disc: a bar at floor
        # level meets the disc rim with a horizontal normal -- a wall, not a
        # wedge.  A real knife is skived: its top plane runs down to an edge
        # of effectively zero thickness BELOW the disc's under-face.  So the
        # blade continues the plate top plane past the plate end to ~0.6 mm
        # under the floor line.  It is floor-EXCLUDED (bit 2 only), which is
        # the only honest way rigid bodies can model that; working height is
        # set by the servo against its droop stop, as the proven scoop hinge
        # always was, and the up-trip still rides over tape and the lab.
        f'    <geom name="A_shim_blade" type="box" contype="4" conaffinity="4" '
        f'condim="3" friction="0.30 0.002 0.0001" '
        f'solref="0.004 1" solimp="0.95 0.99 0.001" '
        f'pos="{mm(s_run/2 + (s_len/2 + 2.0)*cos(radians(s_ang)) + 0.15*sin(radians(s_ang))):.5f} 0 '
        f'{mm(-s_drop/2 - (s_len/2 + 2.0)*sin(radians(s_ang)) - 0.15*cos(radians(s_ang))):.5f}" '
        f'size="0.003 {mm(SHIM_W/2 - 1.0):.5f} 0.0001" '
        f'euler="0 {s_ang:.4f} 0" mass="0.003" rgba="0.80 0.80 0.85 1"/>',
        '  </body>']

    # ---- brush roller (F64: conveyance) ------------------------------------
    # Silicone fingers on a O20 hub, N20-driven, on a sprung swing arm.  It is
    # what carries the piece across the dead zone where F1 measured a passive
    # piece stranding: 19 mm engaged, off the floor, 46 mm short of the belt,
    # becalmed.  A O56 disc bridges the drum's bite to the belt nose (asserted
    # in params), so there is no un-powered stretch anywhere on the intake.
    # Collision proxy: a rigid drum inscribed at working finger squish
    # (R DRUM < R TIP), soft solref standing in for finger compliance; the big
    # tolerance -- riding over whatever comes -- is the ARM's sprung swing.
    # The drum touches PIECES ONLY (bit 2; floor and chassis masked out --
    # fingers brushing the floor or the shim transmit no useful force).  Spin
    # is a real hinge with a velocity actuator capped at N20 stall torque.
    # BUILD NOTE: a LIFTED shim tip sits inside the finger circle, so the
    # roller must spin only with the shim down -- robot.intake() owns that
    # ordering.
    a_dx = AgentA.ROLL_AXIS_X - AgentA.ARM_PIVOT_X
    a_dz = AgentA.ROLL_AXIS_Z - AgentA.ARM_PIVOT_Z
    a_len = (a_dx**2 + a_dz**2) ** 0.5
    a_ang = degrees_atan(-a_dz, a_dx)
    arm_k = AgentA.ARM_PRELOAD_N * mm(a_dx) / 0.1396      # preload at 8 deg past stop
    fingers = _finger_set(AgentA.ROLL_TIP_R, AgentA.ROLL_W, "d", "0.85 0.45 0.25 0.9",
                          indent="      ")
    body += [
        f'  <body name="A_arm" pos="{lx(AgentA.ARM_PIVOT_X):.5f} 0 '
        f'{mm(AgentA.ARM_PIVOT_Z):.5f}">',
        f'    <joint name="A_arm_j" type="hinge" axis="0 1 0" range="-40 0" '
        f'stiffness="{arm_k:.4f}" springref="8" damping="0.02"/>',
        f'    <geom name="A_arm_l" type="box" contype="0" conaffinity="0" '
        f'pos="{mm(a_dx/2):.5f} {mm(AgentA.ROLL_W/2+3):.5f} {mm(a_dz/2):.5f}" '
        f'euler="0 {a_ang:.4f} 0" size="{mm(a_len/2):.5f} 0.002 0.0025" '
        f'mass="0.008" rgba="0.35 0.37 0.40 1"/>',
        f'    <geom name="A_arm_r" type="box" contype="0" conaffinity="0" '
        f'pos="{mm(a_dx/2):.5f} {-mm(AgentA.ROLL_W/2+3):.5f} {mm(a_dz/2):.5f}" '
        f'euler="0 {a_ang:.4f} 0" size="{mm(a_len/2):.5f} 0.002 0.0025" '
        f'mass="0.008" rgba="0.35 0.37 0.40 1"/>',
        f'    <body name="A_drum" pos="{mm(a_dx):.5f} 0 {mm(a_dz):.5f}">',
        f'      <joint name="A_drum_j" type="hinge" axis="0 1 0" limited="false" '
        f'damping="0.0001"/>',
        f'      <geom name="A_drum_g" type="cylinder" zaxis="0 1 0" contype="4" '
        f'conaffinity="4" condim="3" friction="0.90 0.02 0.0002" '
        f'solref="0.008 1" solimp="0.90 0.95 0.002" '
        f'size="{mm(AgentA.FING_HUB_R if AgentA.ROLL_FINGERS else AgentA.ROLL_DRUM_R):.5f} '
        f'{mm(AgentA.ROLL_W/2):.5f}" '
        f'mass="0.050" rgba="0.30 0.32 0.35 0.35"/>',
        f'      <geom name="A_hub_v" type="cylinder" zaxis="0 1 0" contype="0" '
        f'conaffinity="0" size="0.010 {mm(AgentA.ROLL_W/2):.5f}" mass="0.01" '
        f'rgba="0.20 0.22 0.25 1"/>',
    ] + fingers + [
        '    </body>',
        '  </body>']

    # ---- trip bar (F71: lay the patients down before the mouth) ------------
    if AgentA.TRIP_BAR:
        body.append(
            f'  <geom name="A_trip" type="cylinder" zaxis="0 1 0" contype="4" '
            f'conaffinity="4" condim="3" friction="0.35 0.005 0.0001" '
            f'pos="{lx(AgentA.TRIP_X):.5f} 0 {mm(AgentA.TRIP_Z):.5f}" '
            f'size="{mm(AgentA.TRIP_R):.5f} {mm(AgentA.TRIP_W/2):.5f}" '
            f'mass="0.006" rgba="0.85 0.85 0.30 1"/>')

    # ---- upper roller (F71: the patients) ----------------------------------
    # Fixed axis, not sprung: it only has to be there when a tall piece arrives,
    # and a second swing arm would have to be tuned against the first.  Same
    # collision mask as the lower drum -- pieces only (bit 2) -- so it cannot
    # touch the floor, the shim or the chassis.
    if AgentA.UP_ROLL:
        up_f = _finger_set(AgentA.UP_TIP_R, AgentA.UP_W, "u", "0.25 0.65 0.85 0.9",
                           indent="    ")
        body += [
            f'  <body name="A_up" pos="{lx(AgentA.UP_AXIS_X):.5f} 0 '
            f'{mm(AgentA.UP_AXIS_Z):.5f}">',
            f'    <joint name="A_up_j" type="hinge" axis="0 1 0" limited="false" '
            f'damping="0.0001"/>',
            f'    <geom name="A_up_g" type="cylinder" zaxis="0 1 0" contype="4" '
            f'conaffinity="4" condim="3" friction="0.90 0.02 0.0002" '
            f'solref="0.008 1" solimp="0.90 0.95 0.002" '
            f'size="{mm(AgentA.FING_HUB_R if AgentA.ROLL_FINGERS else AgentA.UP_DRUM_R):.5f} '
            f'{mm(AgentA.UP_W/2):.5f}" '
            f'mass="0.050" rgba="0.20 0.45 0.60 0.35"/>',
        ] + up_f + ['  </body>']

    # ---- sweeper fingers ---------------------------------------------------
    for s, tag in ((1, "l"), (-1, "r")):
        body += [
            f'  <body name="A_finger_{tag}" pos="{lx(AgentA.SWEEP_PIVOT_X):.5f} '
            f'{s*mm(FINGER_PIVOT_Y):.5f} {mm(14.5):.5f}">',
            # PASSIVE SPRING, NOT A SERVO (F65).  The rake position is never
            # commanded anywhere in the mission -- the fingers sit at OPEN for
            # the whole match -- so the "actuator" was a torsion spring wearing
            # a servo's badge.  stiffness 4 at springref OPEN is bit-identical
            # to the old kp=4 position hold; two MG90S and two driver channels
            # leave the build.
            f'    <joint name="A_f_{tag}" type="hinge" axis="0 0 1" range="-30 30" '
            f'stiffness="4" springref="{s*AgentA.FINGER_OPEN:.2f}" damping="0.01"/>',
            f'    <geom name="A_fg_{tag}" class="robot" type="box" '
            f'pos="{-mm(FINGER_LEN/2):.5f} 0 0" '
            f'size="{mm(FINGER_LEN/2):.5f} {mm(2):.5f} {mm(12.5):.5f}" mass="0.03" '
            f'rgba="0.85 0.55 0.2 1"/>',
            '  </body>']

    # ---- POSTING HEAD on its lateral trim slide (F68) ---------------------
    # Bore, collar, escapement, feed plunger and the slot probes are ONE body
    # that slides +/-TRIM_Y across the robot.  Nothing about the drop changes --
    # every F9/F15/F16/F17/F19/F41/F55 finding is about what happens inside this
    # head, and the head is unchanged; what moves is where it is aimed.
    esc  = mm(AgentA.ESC_Y)
    bpk  = mm(AgentA.ESC_BLADE_PARK)
    half = mm(AgentA.ESC_HALF)
    trim = mm(AgentA.TRIM_Y)
    body += [f'  <body name="A_trim" pos="0 0 0">',
             f'    <joint name="A_trim_j" type="slide" axis="0 1 0" '
             f'range="{-trim:.5f} {trim:.5f}" damping="0.20"/>']
    body += ["    " + gm for gm in head]
    body += [
        # positive feed: plunger on the bore axis (see params, F18)
        f'    <body name="A_feed" pos="{lx(AgentA.FEED_X):.5f} 0 '
        f'{mm(AgentA.FEED_Z_UP):.5f}">',
        f'      <joint name="A_feed_j" type="slide" axis="0 0 1" '
        f'range="{-mm(AgentA.FEED_STROKE):.5f} 0" damping="0.06"/>',
        f'      <geom name="A_feed_g" class="robot" type="cylinder" '
        f'size="{mm(AgentA.FEED_D/2):.5f} {mm(1.5):.5f}" mass="0.012" '
        f'friction="0.10 0.002 0.0001" rgba="0.90 0.55 0.20 1"/>',
        '    </body>',
        # ESCAPEMENT, two leaves per stage (F68).  Leaf L closes to +y, leaf R
        # to -y; they overlap at the axis so there is no slit for a disc to
        # find.  Opening, they retract in opposite directions -- which is why
        # a released disc goes straight down instead of being swept sideways.
        f'    <body name="A_gate_l" pos="{cx:.5f} '
        f'{mm(AgentA.ESC_HALF - AgentA.ESC_OVER):.5f} {mm(AgentA.CHUTE_Z0-1.5):.5f}">',
        f'      <joint name="A_gate_l_j" type="slide" axis="0 1 0" '
        f'range="0 {mm(AgentA.ESC_Y):.5f}" damping="0.05"/>',
        f'      <geom name="A_gate_l_g" class="robot" type="box" '
        f'size="{mm(AgentA.ESC_XHALF):.5f} {mm(AgentA.ESC_HALF):.5f} '
        f'{mm(AgentA.ESC_T):.5f}" mass="0.005" rgba="0.9 0.35 0.2 1"/>',
        '    </body>',
        f'    <body name="A_gate_r" pos="{cx:.5f} '
        f'{-mm(AgentA.ESC_HALF - AgentA.ESC_OVER):.5f} {mm(AgentA.CHUTE_Z0-1.5):.5f}">',
        f'      <joint name="A_gate_r_j" type="slide" axis="0 -1 0" '
        f'range="0 {mm(AgentA.ESC_Y):.5f}" damping="0.05"/>',
        f'      <geom name="A_gate_r_g" class="robot" type="box" '
        f'size="{mm(AgentA.ESC_XHALF):.5f} {mm(AgentA.ESC_HALF):.5f} '
        f'{mm(AgentA.ESC_T):.5f}" mass="0.005" rgba="0.9 0.35 0.2 1"/>',
        '    </body>',
        # retainer leaves: parked clear of the bore, driven IN to hold the
        # column at the joint while the shelf is out from under it.  Each
        # carries the F47 rolled lip on its leading edge -- square, it met the
        # second disc's rim head-on and drove it out of the bore.
        f'    <body name="A_blade_l" pos="{cx:.5f} '
        f'{mm(AgentA.ESC_BLADE_HALF - AgentA.ESC_BLADE_OVER + AgentA.ESC_BLADE_PARK):.5f} '
        f'{mm(AgentA.ESC_BLADE_Z):.5f}">',
        f'      <joint name="A_blade_l_j" type="slide" axis="0 -1 0" '
        f'range="0 {mm(AgentA.ESC_BLADE_PARK):.5f}" damping="0.05"/>',
        f'      <geom name="A_blade_l_g" class="robot" type="box" '
        f'size="{mm(AgentA.ESC_BLADE_XHALF):.5f} {mm(AgentA.ESC_BLADE_HALF):.5f} '
        f'{mm(AgentA.ESC_BLADE_T):.5f}" mass="0.003" rgba="0.9 0.55 0.2 1"/>',
        f'      <geom name="A_blade_l_lip" class="robot" type="cylinder" '
        f'zaxis="1 0 0" pos="0 {-mm(AgentA.ESC_BLADE_HALF):.5f} '
        f'{mm(AgentA.ESC_LIP_Z - AgentA.ESC_BLADE_Z):.5f}" '
        f'size="{mm(AgentA.ESC_LIP_R):.5f} {mm(AgentA.ESC_BLADE_XHALF):.5f}" mass="0.001" '
        f'rgba="0.9 0.55 0.2 1"/>',
        '    </body>',
        f'    <body name="A_blade_r" pos="{cx:.5f} '
        f'{-mm(AgentA.ESC_BLADE_HALF - AgentA.ESC_BLADE_OVER + AgentA.ESC_BLADE_PARK):.5f} '
        f'{mm(AgentA.ESC_BLADE_Z):.5f}">',
        f'      <joint name="A_blade_r_j" type="slide" axis="0 1 0" '
        f'range="0 {mm(AgentA.ESC_BLADE_PARK):.5f}" damping="0.05"/>',
        f'      <geom name="A_blade_r_g" class="robot" type="box" '
        f'size="{mm(AgentA.ESC_BLADE_XHALF):.5f} {mm(AgentA.ESC_BLADE_HALF):.5f} '
        f'{mm(AgentA.ESC_BLADE_T):.5f}" mass="0.003" rgba="0.9 0.55 0.2 1"/>',
        f'      <geom name="A_blade_r_lip" class="robot" type="cylinder" '
        f'zaxis="1 0 0" pos="0 {mm(AgentA.ESC_BLADE_HALF):.5f} '
        f'{mm(AgentA.ESC_LIP_Z - AgentA.ESC_BLADE_Z):.5f}" '
        f'size="{mm(AgentA.ESC_LIP_R):.5f} {mm(AgentA.ESC_BLADE_XHALF):.5f}" mass="0.001" '
        f'rgba="0.9 0.55 0.2 1"/>',
        '    </body>',
        # bore rangefinder and the two slot probes ride WITH the head -- that is
        # the whole point: what they measure is the slot relative to the bore.
        f'    <site name="A_mag" pos="{cx:.5f} 0 {mm(70):.4f}" zaxis="0 0 -1"/>',
        '  </body>']

    # ---- beam cradles (F44/F46) -------------------------------------------
    # One per pocket: two L-hooks on a common vertical slide.  Up (ctrl 0) the
    # shelves carry the beam CARRY_Z off the floor, which is what lets it cross
    # the laboratory and lets the robot pivot at all; down (ctrl -CARRY_Z) the
    # beam stands on the field and the shelves sit in the floor plane, out of
    # the way.  The shelves are on the belt's collision bit, not the floor's,
    # exactly as the scoop is -- otherwise "down" would mean "jacked up on the
    # floor" and the beam would never touch the ground.
    cz = mm(AgentA.CARRY_Z + AgentA.CRADLE_DROP)
    for sy, tag, hooks in ((1, "1", AgentA.HOOK1_X), (-1, "2", AgentA.HOOK2_X)):
        body.append(f'  <body name="A_cradle{tag}" pos="0 {sy*mm(AgentA.POCKET_Y):.5f} '
                    f'{mm(AgentA.CARRY_Z):.5f}">')
        body.append(f'    <joint name="A_cr{tag}_j" type="slide" axis="0 0 -1" '
                    f'range="0 {mm(AgentA.CARRY_Z + AgentA.CRADLE_DROP):.5f}" '
                    f'damping="0.05"/>')
        for i, hx in enumerate(hooks):
            body.append(
                f'    <geom name="A_shelf{tag}_{i}" type="box" contype="2" conaffinity="2" '
                f'condim="3" friction="0.35 0.005 0.0001" '
                f'pos="{mm(hx):.5f} {sy*mm(5.0):.5f} {-mm(0.75):.5f}" '
                f'size="{mm(AgentA.HOOK_W):.5f} {mm(5.0):.5f} {mm(0.75):.5f}" '
                f'mass="0.006" rgba="0.95 0.62 0.15 1"/>')
            body.append(
                f'    <geom name="A_hook{tag}_{i}" type="box" contype="2" '
                f'conaffinity="2" condim="3" friction="0.35 0.005 0.0001" '
                f'pos="{mm(hx):.5f} {sy*mm(AgentA.HOOK_Y-AgentA.POCKET_Y):.5f} '
                f'{mm(AgentA.HOOK_H):.5f}" '
                f'size="{mm(AgentA.HOOK_W):.5f} {mm(AgentA.HOOK_T):.5f} '
                f'{mm(AgentA.HOOK_H):.5f}" mass="0.006" rgba="0.95 0.62 0.15 1"/>')
        body.append('  </body>')

    # ---- OAK-D on the tail mast (F69) -------------------------------------
    # A post from the deck carrying the camera over the tail, looking back and
    # down.  Modelled with mass and bulk because it is 136 g at Za 155 -- the
    # highest mass on the robot, and it moves the centre of gravity aft, which
    # is the direction that matters when the tail overhangs the laboratory.
    #
    # The MuJoCo camera stands for the depth map's frame.  DepthAI aligns depth
    # to the rectified LEFT mono by default, so one camera at the left sensor's
    # pose is the honest model of what the Pi receives; the right sensor exists
    # only inside the device.
    mx, mz = lx(Vision.CAM_X), mm(Vision.CAM_Z)
    hb = mm(Vision.BASELINE/2.0)
    body += [
        f'  <geom name="A_mast" class="robot" type="box" pos="{mx:.5f} 0 '
        f'{mm((96.5 + Vision.CAM_Z)/2):.5f}" size="{mm(6):.5f} {mm(30):.5f} '
        f'{mm((Vision.CAM_Z - 96.5)/2):.5f}" mass="{Vision.MAST_MASS/1000.0:.4f}" '
        f'rgba="0.35 0.35 0.38 1"/>',
        # ONE plate, two cameras: what makes the pair's own calibration hold is
        # that the baseline is a single laser-cut part, not two brackets that
        # can move relative to each other.
        f'  <geom name="A_camplate" class="robot" type="box" pos="{mx:.5f} 0 '
        f'{mz:.5f}" euler="0 {Vision.CAM_PITCH:.1f} 0" '
        f'size="{mm(3):.5f} {mm(Vision.BASELINE/2 + 14):.5f} {mm(16):.5f}" '
        f'mass="{2*Vision.CAM_MASS/1000.0:.4f}" rgba="0.85 0.55 0.10 1"/>']
    for sy, tag in ((1, "l"), (-1, "r")):
        # toed IN, so both cameras keep the dock zone in frame at close range
        yaw = -sy*Vision.TOE
        cp, sp = cos(radians(Vision.CAM_PITCH)), sin(radians(Vision.CAM_PITCH))
        cy_, sy_ = cos(radians(yaw)), sin(radians(yaw))
        # image +x is the robot's LEFT, image +y is up-image; MuJoCo looks -z
        xax = (-sy_, cy_, 0.0)
        yax = (-sp*cy_, -sp*sy_, cp)
        body.append(
            f'  <camera name="A_cam_{tag}" pos="{mx:.5f} {sy*hb:.5f} {mz:.5f}" '
            f'xyaxes="{xax[0]:.5f} {xax[1]:.5f} {xax[2]:.5f} '
            f'{yax[0]:.5f} {yax[1]:.5f} {yax[2]:.5f}" '
            f'fovy="{2*degrees(atan((Vision.H/2)/Vision.f_px())):.4f}"/>')

    body += [f'  <site name="A_imu" pos="0 0 {mm(60):.4f}"/>',
             f'  <site name="A_tof" pos="{lx(AgentA.L):.5f} 0 {mm(45):.4f}" zaxis="1 0 0"/>',
             f'  <camera name="A_chase" pos="{-mm(560):.4f} 0 {mm(420):.4f}" xyaxes="0 -1 0 0.6 0 0.8"/>',
             '</body>']

    UPACT = ('<velocity name="a_uproll" joint="A_up_j" kv="0.010" ctrlrange="0 60" '
             'forcerange="%.4f %.4f"/>' % (-AgentA.UP_TORQUE, AgentA.UP_TORQUE)
             ) if AgentA.UP_ROLL else ""
    fs = -mm(AgentA.FEED_STROKE)
    nesc = -bpk
    esc2, bpk2 = 2*esc, 2*bpk           # a tendon of two joints reads twice the stroke
    crn = mm(AgentA.CARRY_Z + AgentA.CRADLE_DROP)
    act = f"""
    <velocity name="a_drive_l" joint="A_w_l" kv="5.0" ctrlrange="-30 30" forcerange="-0.5 0.5"/>
    <velocity name="a_drive_r" joint="A_w_r" kv="5.0" ctrlrange="-30 30" forcerange="-0.5 0.5"/>
    <!-- knife servo: the force cap IS the tip preload (~0.3 N at the tip);
         down (+droop) saturates against the floor, up (-lift) clears transit -->
    <position name="a_shim" joint="A_shim_j" kp="0.8" kv="0.05"
              ctrlrange="{-radians(AgentA.SHIM_LIFT):.5f} {radians(AgentA.SHIM_DROOP):.5f}"
              forcerange="-0.02 0.02"/>
    <!-- brush roller: velocity servo capped at the N20's stall torque -->
    <velocity name="a_roller" joint="A_drum_j" kv="0.010"
              ctrlrange="0 60" forcerange="{-AgentA.ROLL_TORQUE} {AgentA.ROLL_TORQUE}"/>
    {UPACT}
    <!-- F68 trim slide: one MG90S through a Scotch yoke, carrying the whole
         posting head.  kv is high because the head must ARRIVE and stay put --
         a slide still ringing when the shelf opens is the same error the
         chassis used to make. -->
    <position name="a_trim" joint="A_trim_j" kp="400" kv="30"
              ctrlrange="{-trim:.5f} {trim:.5f}" forcerange="-14 14"/>
    <!-- One pinion drives both racks, so ONE actuator drives both leaves
         through a fixed tendon.  That is the real coupling, not two servos
         asked to agree. -->
    <position name="a_gate" tendon="A_gate_t" kp="900" kv="12"
              ctrlrange="0 {esc2:.5f}" forcerange="-25 25"/>
    <position name="a_blade" tendon="A_blade_t" kp="600" kv="10"
              ctrlrange="0 {bpk2:.5f}" forcerange="-12 12"/>
    <position name="a_feed" joint="A_feed_j" kp="120" kv="5" ctrlrange="{fs:.5f} 0" forcerange="-4.0 4.0"/>
    <position name="a_cradle1" joint="A_cr1_j" kp="600" kv="20" ctrlrange="0 {crn:.5f}" forcerange="-16 16"/>
    <position name="a_cradle2" joint="A_cr2_j" kp="600" kv="20" ctrlrange="0 {crn:.5f}" forcerange="-16 16"/>"""

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
    <!-- The TCRT slot and plate-edge probes are GONE (F69): the OAK-D measures
         the laboratory directly.  What is left below is the one rangefinder a
         camera cannot replace -- a_mag looks down a 60 mm tube. -->
    <jointvel    name="a_wvel_l" joint="A_w_l"/>
    <jointvel    name="a_wvel_r" joint="A_w_r"/>
    <actuatorfrc name="a_frc_l" actuator="a_drive_l"/>
    <actuatorfrc name="a_frc_r" actuator="a_drive_r"/>"""
    ten = f"""
  <tendon>
    <fixed name="A_gate_t">
      <joint joint="A_gate_l_j" coef="1"/>
      <joint joint="A_gate_r_j" coef="1"/>
    </fixed>
    <fixed name="A_blade_t">
      <joint joint="A_blade_l_j" coef="1"/>
      <joint joint="A_blade_r_j" coef="1"/>
    </fixed>
  </tendon>"""
    return "\n".join(body), act + "\n@@TENDON@@" + ten, sen


# ------------------------------------------------------------------ pieces
def disc_body(i, x, y, z=None):
    z = z if z is not None else Piece.DISC_T/2 + 0.5
    return (f'<body name="disc{i}" pos="{mm(x):.5f} {mm(y):.5f} {mm(z):.5f}">'
            f'<freejoint name="disc{i}_f"/>'
            f'<geom name="disc{i}_g" class="piece" contype="15" conaffinity="15" type="cylinder" '
            f'size="{mm(Piece.DISC_D/2):.5f} {mm(Piece.DISC_T/2):.5f}" '
            f'mass="{Piece.DISC_M/1000.0:.4f}" rgba="{C_DISC}"/></body>')

def cyl_body(i, x, y, colour):
    rgba = {"red": "0.82 0.24 0.24 1", "yellow": "0.88 0.70 0.20 1",
            "green": "0.25 0.62 0.32 1"}[colour]
    return (f'<body name="cyl{i}" pos="{mm(x):.5f} {mm(y):.5f} {mm(Piece.CYL_H/2+0.5):.5f}">'
            f'<freejoint name="cyl{i}_f"/>'
            # BIT 2, LIKE A SAMPLE.  Without it a patient cannot touch the
            # intake at all -- shim, blade and both rollers are on bit 2 -- and
            # the simulator dutifully reported ten of ten "bulldozed", which was
            # a masking artefact and not physics.  A finding that is really a
            # collision filter is the most expensive kind.
            f'<geom name="cyl{i}_g" class="piece" contype="15" conaffinity="15" type="cylinder" '
            f'size="{mm(Piece.CYL_D/2):.5f} {mm(Piece.CYL_H/2):.5f}" '
            f'mass="{Piece.CYL_M/1000.0:.4f}" rgba="{rgba}"/></body>')

def kit_body(i, x, y, z=None):
    """A medical kit.  Rules g.1 let these start ON the robot, so the mission
    never picks one up -- the only verb is release."""
    z = Piece.KIT_Z/2 + 0.5 if z is None else z
    return (f'<body name="kit{i}" pos="{mm(x):.5f} {mm(y):.5f} {mm(z):.5f}">'
            f'<freejoint name="kit{i}_f"/>'
            f'<geom name="kit{i}_g" class="piece" type="box" '
            f'size="{mm(Piece.KIT_X/2):.5f} {mm(Piece.KIT_Y/2):.5f} {mm(Piece.KIT_Z/2):.5f}" '
            f'mass="{Piece.KIT_M/1000.0:.4f}" rgba="0.93 0.93 0.95 1"/></body>')


def m2_layout(rng=None):
    """Twelve patients on their stickers, four of each colour, six per side.

    WHERE THEY STAND IS AN ASSUMPTION (M2.SIDE_L/SIDE_R) -- the rulebook gives
    no coordinates, only "two side areas... six cylinders on each side".  Senior
    randomises the arrangement every match, so the layout is drawn per seed and
    the colour order is shuffled: a route that only works for one colour order
    is a route that has not been tested.
    """
    import numpy as _np
    rng = rng or _np.random.default_rng(0)
    cols = list(M2.COLOURS) * 4
    rng.shuffle(cols)
    out, k = [], 0
    for box in (M2.SIDE_L, M2.SIDE_R):
        x0, y0, x1, y1 = box
        for j in range(6):
            fx = 0.25 if j % 2 == 0 else 0.75
            fy = (j // 2 + 0.5) / 3.0
            out.append((x0 + fx*(x1-x0), y0 + fy*(y1-y0), cols[k])); k += 1
    return out


def beam_body(i, x, y, length, heading_deg, mass, z0=0.5):
    return (f'<body name="beam{i}" pos="{mm(x):.5f} {mm(y):.5f} {mm(Piece.BEAM_H/2+z0):.5f}" '
            f'euler="0 0 {heading_deg}">'
            f'<freejoint name="beam{i}_f"/>'
            f'<geom name="beam{i}_g" class="piece" type="box" '
            f'size="{mm(length/2):.5f} {mm(Piece.BEAM_W/2):.5f} {mm(Piece.BEAM_H/2):.5f}" '
            f'mass="{mass/1000.0:.4f}" rgba="0.85 0.55 0.20 1"/></body>')


# ------------------------------------------------------------------ scenes
def contact_pairs(agent="A", n_discs=3):
    """Explicit friction pairs -- see note in scene()."""
    out = ["  <contact>"]
    # The knife shim rides inside the chassis envelope (guide walls seal to its
    # top face) and the lifted tip swings into the drum's collision circle, so
    # both pairings are masked out -- the real contacts are silent brushes.
    out.append(f'    <exclude body1="agentA" body2="A_shim"/>')
    out.append(f'    <exclude body1="A_shim" body2="A_drum"/>')
    if AgentA.UP_ROLL:
        out.append(f'    <exclude body1="agentA" body2="A_up"/>')
        out.append(f'    <exclude body1="A_shim" body2="A_up"/>')
        out.append(f'    <exclude body1="A_drum" body2="A_up"/>')
    # F68.  The posting head is a body now, so its bore ring and the escapement
    # riding on it would collide with the shell they live inside -- geometry
    # that was silent while they were all one body.  The shell's plates are
    # there to meet the FIELD; the mechanism inside them is clearance the CAD
    # owns, and CHECKS asserts the two extents that actually matter (the
    # retainer against the beam pocket, the shelf under a carried beam).
    _HEAD = ("A_trim", "A_gate_l", "A_gate_r", "A_blade_l", "A_blade_r", "A_feed")
    for _b in _HEAD:
        out.append(f'    <exclude body1="agentA" body2="{_b}"/>')
    for _i, _b in enumerate(_HEAD):
        for _c in _HEAD[_i+1:]:
            out.append(f'    <exclude body1="{_b}" body2="{_c}"/>')
    for i in range(4):
        out.append(f'    <pair geom1="{agent}_ball{i}" geom2="floor" '
                   f'friction="{Chassis.MU_BALL} {Chassis.MU_BALL} 0.0001 0.0001 0.0001" '
                   f'solref="0.02 2.0" solimp="0.6 0.9 0.01"/>')
    # shim plate is excluded from the floor by collision bitmask -- no pair
    # needed; only its tip lip touches the field
    # The gate must slide OUT FROM UNDER the disc.  MuJoCo takes the element-wise
    # MAX of the two geoms' friction, so a smooth gate against a 0.6 piece still
    # gets 0.6 -- and the disc simply rode the gate out of the chute.
    for i in range(n_discs):
        out.append(f'    <pair geom1="{agent}_feed_g" geom2="disc{i}_g" '
                   f'friction="0.06 0.06 0.0005 0.0001 0.0001" '
                   f'solref="0.004 1" solimp="0.95 0.99 0.001"/>')
        for gg in ("gate_l_g", "gate_r_g", "blade_l_g", "blade_r_g",
                   "blade_l_lip", "blade_r_lip"):
            out.append(f'    <pair geom1="{agent}_{gg}" geom2="disc{i}_g" '
                       f'friction="0.04 0.04 0.0005 0.0001 0.0001" '
                       f'solref="0.004 1" solimp="0.95 0.99 0.001"/>')
        # Converging guides are printed PETG: slippery.  Combined at the floor's
        # 0.6 the discs jammed across the throat instead of single-filing -- the
        # spec's own "#1 jam risk", reproduced.
        for g in ("guide_l", "guide_r", "lane_l", "lane_r"):
            out.append(f'    <pair geom1="{agent}_{g}" geom2="disc{i}_g" '
                       f'friction="0.08 0.08 0.001 0.0001 0.0001"/>')
        # The knife is polished spring steel; MuJoCo's element-wise MAX would
        # otherwise hand it the disc's 0.6 and brake every climb.
        for g in ("shim_g", "shim_blade"):
            out.append(f'    <pair geom1="{agent}_{g}" geom2="disc{i}_g" '
                       f'friction="0.30 0.30 0.002 0.0001 0.0001" '
                       f'solref="0.004 1" solimp="0.95 0.99 0.001"/>')
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


def scene(name, bodies, actuators="", sensors="", timestep=0.001, contacts="",
          equality=""):
    # agent_a_body returns its <tendon> block glued to the actuators behind a
    # marker, because a tendon is a top-level section but is only ever written
    # by whoever writes the actuators that pull on it.
    tendons = ""
    if "@@TENDON@@" in actuators:
        actuators, tendons = actuators.split("@@TENDON@@", 1)
    return f"""<mujoco model="{name}">
{preamble(timestep)}
  <worldbody>{FIELD_CAMS}
{chr(10).join('    ' + b for b in bodies)}
  </worldbody>
{contacts}
{equality}{tendons}
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
        px, py, ph = robot_pose or AgentA.START_POSE
        for i, (loc, L, M) in enumerate(((AgentA.BEAM1_LOCAL, Piece.BEAM1_L, Piece.BEAM1_M),
                                         (AgentA.BEAM2_LOCAL, Piece.BEAM2_L, Piece.BEAM2_M)), 1):
            wx, wy = local_to_world(px, py, ph, loc[0], loc[1])
            parts.append(beam_body(i, wx, wy, L, ph, M, z0=AgentA.CARRY_Z + 0.2))
    # A CARRIED BEAM IS CLAMPED, NOT LOOSE (F50).  Modelled as a free body
    # resting in its channel, a 200 g beam 285 mm long slops around inside the
    # 2 mm of pocket clearance, and its inertia arrives a moment after the
    # chassis's on every start, stop and turn.  That is not a detail: with the
    # beams aboard the laboratory dock went from 1.5 mm in 14 s to 38 mm in
    # 87 s.  The real pocket does not work that way either -- the cradle lifts
    # the beam and wedges it against the pocket's inner wall, which is a clamp.
    # So: welded while carried, free the instant the cradle drops.
    eq = ""
    if with_beams:
        eq = ("  <equality>\n"
              + "".join('    <weld name="beam%d_hold" body1="agentA" body2="beam%d" '
                        'solref="0.004 1" solimp="0.98 0.999 0.001"/>\n' % (i, i)
                        for i in (1, 2))
              + "  </equality>")
    return scene("rfgyc26_pick_place", parts, act, sen,
                 contacts=contact_pairs(n_discs=len(disc_positions)), equality=eq)


def scene_full_match(disc_positions, robot_pose=None, rng=None, kits_aboard=True):
    """The whole 250-point board: 3 samples, 2 beams, 10 kits, 12 patients.

    KITS START ABOARD by default, which is not a liberty -- rules g.1 say the
    kits "may be placed on the board, ON THE ROBOT, or incorporated into its
    mechanism".  That is what makes Mission 2 cheap: the robot never picks a kit
    up, it only opens a flap.  Stacked two-high in the flanks above the wheels,
    where the route plan puts the hoppers.
    """
    import numpy as _np
    rng = rng or _np.random.default_rng(0)
    body, act, sen = agent_a_body(pose=robot_pose, with_beams=True)
    parts = field_body() + [body]
    for i, (x, y) in enumerate(disc_positions):
        parts.append(disc_body(i, x, y))
    px, py, ph = robot_pose or AgentA.START_POSE
    for i, (loc, L, M) in enumerate(((AgentA.BEAM1_LOCAL, Piece.BEAM1_L, Piece.BEAM1_M),
                                     (AgentA.BEAM2_LOCAL, Piece.BEAM2_L, Piece.BEAM2_M)), 1):
        wx, wy = local_to_world(px, py, ph, loc[0], loc[1])
        parts.append(beam_body(i, wx, wy, L, ph, M, z0=AgentA.CARRY_Z + 0.2))
    for i, (cx, cy, col) in enumerate(m2_layout(rng)):
        parts.append(cyl_body(i, cx, cy, col))
    # KITS RIDE IN THREE HOPPERS, ONE PER DESTINATION.  Grouped before the
    # match, so delivery is "open hopper X" and never a sorting problem.
    kit_at = {}
    for dest, idx in M2.KIT_GROUPS.items():
        hx_, hy_ = M2.HOPPER[dest]
        w = M2.HOPPER_WIDE[dest]
        for k, i in enumerate(idx):
            kit_at[i] = (hx_ + (k % w)*M2.HOPPER_PITCH, hy_,
                         M2.HOPPER_Z + (k // w)*(Piece.KIT_Z + 3.0))
    for i in range(M2.N_KITS):
        if kits_aboard:
            lx_, ly_, lz_ = kit_at[i]
            wx, wy = local_to_world(px, py, ph, lx_, ly_)
            parts.append(kit_body(i, wx, wy, z=lz_))
        else:
            parts.append(kit_body(i, 700.0 + (i % 5)*35.0, 40.0 + (i // 5)*35.0))
    eq = ("  <equality>\n"
          + "".join('    <weld name="beam%d_hold" body1="agentA" body2="beam%d" '
                    'solref="0.004 1" solimp="0.98 0.999 0.001"/>\n' % (i, i)
                    for i in (1, 2))
          + "".join('    <weld name="kit%d_hold" body1="agentA" body2="kit%d" '
                    'solref="0.004 1" solimp="0.98 0.999 0.001"/>\n' % (i, i)
                    for i in range(M2.N_KITS) if kits_aboard)
          + "  </equality>")
    return scene("rfgyc26_full_match", parts, act, sen,
                 contacts=contact_pairs(n_discs=len(disc_positions)), equality=eq)


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
