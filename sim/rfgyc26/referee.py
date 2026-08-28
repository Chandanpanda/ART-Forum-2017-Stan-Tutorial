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
