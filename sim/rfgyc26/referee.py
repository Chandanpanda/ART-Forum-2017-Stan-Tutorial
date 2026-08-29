"""Senior scoring, as a pure function of the final world state."""
import numpy as np, mujoco
from .params import Field, Piece
from .mjcf import LAB_HOLE_Y

WRONG_ZONE_PENALTY = -5      # R7: table says -3, penalties section says -5


def _in(box, x, y):
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def score_discs(disc_xyz, ruleset_wrong=WRONG_ZONE_PENALTY):
    """disc_xyz: list of (x_mm, y_mm, z_mm). Returns (points, breakdown)."""
    placed, out, in_quar = 0, 0, 0
    detail = []
    for i, (x, y, z) in enumerate(disc_xyz):
        hole = None
        for h, hx in enumerate(Field.LAB_HOLE_X):
            if np.hypot(x - hx, y - LAB_HOLE_Y) <= (Field.LAB_HOLE_D - Piece.DISC_D)/2 + 8:
                hole = h; break
        if hole is not None and z < Field.LAB_PLATE_T + Piece.DISC_T:
            placed += 1; detail.append((i, "lab slot %d" % (hole+1), +15))
        elif _in(Field.QUARANTINE, x, y):
            in_quar += 1; detail.append((i, "left in quarantine", -5))
        else:
            out += 1; detail.append((i, "stranded", -3))
    pts = 15*placed - 5*in_quar - 3*out
    if placed == 3:
        pts += 5; detail.append((-1, "all three samples placed", +5))
    return pts, detail


# --------------------------------------------------------------------- beams
# R7/spec 1: +25 for each beam correctly placed, +20 when the perimeter closes.
# "Correctly placed" is read from the rulebook's own end state (spec 1): each
# beam standing on a 20 mm face, 60 tall, free-standing, along its nominal line
# and touching its wall.  "Closed" is the T-joint: beam 2's north end face
# against beam 1's south side face at about (280, 250).
BEAM_POS_TOL   = 25.0     # mm, centre error along and across the line
BEAM_TILT_TOL  = 8.0      # deg off upright before it is not "standing"
BEAM_WALL_TOL  = 6.0      # mm gap to the wall still counted as "from the wall"
BEAM_TOUCH_TOL = 3.0      # mm gap at the T-joint still counted as closure


def _beam_frame(q):
    """(yaw_deg, tilt_deg) from a body quaternion."""
    w, x, y, z = q
    yaw  = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    # the beam's own +z against world +z
    up   = np.array([2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)])
    tilt = np.degrees(np.arccos(np.clip(up[2], -1, 1)))
    return yaw, tilt


def score_beams(beams):
    """beams: [(x_mm, y_mm, z_mm, quat), ...] for beam 1 then beam 2.

    Returns (points, detail).  Nothing here is scored on intent: a beam that
    fell over, drifted or never left the pocket scores zero, and the closure
    bonus needs the two pieces actually within BEAM_TOUCH_TOL of each other.
    """
    detail, placed, ends = [], 0, {}
    targets = [(Field.BEAM1_CENTRE, 0.0,   "west wall"),
               (Field.BEAM2_CENTRE, 90.0,  "south wall")]
    for i, ((tx, ty), theading, wall) in enumerate(targets, 1):
        if i - 1 >= len(beams):
            continue
        x, y, z, q = beams[i-1]
        yaw, tilt = _beam_frame(q)
        L = Piece.BEAM1_L if i == 1 else Piece.BEAM2_L
        # ends of the beam along its own long axis
        t = np.radians(yaw)
        ends[i] = ((x - L/2*np.cos(t), y - L/2*np.sin(t)),
                   (x + L/2*np.cos(t), y + L/2*np.sin(t)))
        why = []
        if tilt > BEAM_TILT_TOL:
            why.append("tipped %.0f deg" % tilt)
        if abs(z - Piece.BEAM_H/2) > 8.0:
            why.append("not standing on the field (z=%.0f)" % z)
        if abs(((yaw - theading + 90) % 180) - 90) > 10.0:
            why.append("off heading by %.0f deg" % abs(((yaw-theading+90) % 180) - 90))
        if np.hypot(x - tx, y - ty) > BEAM_POS_TOL:
            why.append("centre off by %.0f mm" % np.hypot(x-tx, y-ty))
        touch = min(abs(ends[i][0][0]), abs(ends[i][1][0])) if i == 1 else \
                min(abs(ends[i][0][1]), abs(ends[i][1][1]))
        if touch > BEAM_WALL_TOL:
            why.append("%.0f mm off the %s" % (touch, wall))
        if why:
            detail.append((i, "beam %d NOT placed: %s" % (i, "; ".join(why)), 0))
        else:
            placed += 1
            detail.append((i, "beam %d placed on the %s" % (i, wall), +25))
    pts = 25*placed
    if placed == 2:
        # T-joint: the closest approach between beam 2's end faces and beam 1's
        # south side face.
        (x0, y0), (x1, y1) = ends[1]
        gap = min(_seg_gap((x0, y0), (x1, y1), e) for e in ends[2])
        if gap <= BEAM_TOUCH_TOL + Piece.BEAM_W:
            pts += 20
            detail.append((-1, "perimeter closed (T-joint gap %.1f mm)"
                           % max(0.0, gap - Piece.BEAM_W), +20))
        else:
            detail.append((-1, "perimeter NOT closed: %.0f mm gap at the T-joint"
                           % (gap - Piece.BEAM_W), 0))
    return pts, detail


def _seg_gap(a, b, p):
    a, b, p = np.array(a), np.array(b), np.array(p)
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t*ab)))
