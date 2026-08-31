"""Senior scoring, as a pure function of the final world state."""
import numpy as np, mujoco
from .params import Field, Piece, M2
from .mjcf import LAB_HOLE_Y

WRONG_ZONE_PENALTY = -5      # R7: table says -3, penalties section says -5


def _in(box, x, y):
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


# F67.  "COMPLETELY INSIDE" MEANS INSIDE (rules 2.1).
#
# This test used to allow the disc's centre 10 mm of radial error and its centre
# height up to LAB_PLATE_T + DISC_T.  A O56 disc lying FLAT ON TOP of a 6 mm
# plate has its centre at Za 8.5, so it passed both -- 9 mm off the hole and not
# in it at all, scored +15.  Every mission number in the README was flattered by
# that, the sweeps most of all, because a near miss that skates onto the plate
# looks exactly like a hit.
#
# The honest test is geometric and has no free parameters:
#   radial  -- a disc that is physically inside a O60 bore cannot be more than
#              (60-56)/2 = 2.0 mm off its axis.  3.0 allows for solver slop.
#   height  -- "completely inside" means the disc's TOP is not above the plate's
#              top: z + DISC_T/2 <= LAB_PLATE_T.  Seated on the field floor
#              inside the bore that is z = 2.5, with 1.0 of slack for a tilt.
SLOT_R_TOL  = 3.0
def _slot_z_max():
    return Field.LAB_PLATE_T - Piece.DISC_T/2.0 + 1.0


def score_discs(disc_xyz, ruleset_wrong=WRONG_ZONE_PENALTY):
    """disc_xyz: list of (x_mm, y_mm, z_mm). Returns (points, breakdown)."""
    placed, out, in_quar = 0, 0, 0
    detail = []
    for i, (x, y, z) in enumerate(disc_xyz):
        hole = None
        for h, hx in enumerate(Field.LAB_HOLE_X):
            if np.hypot(x - hx, y - LAB_HOLE_Y) <= SLOT_R_TOL:
                hole = h; break
        if hole is not None and z <= _slot_z_max():
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
        # T-JOINT: THE GAP BETWEEN THE TWO FOOTPRINTS, FACE TO FACE.
        #
        # This used to measure beam 2's centreline endpoint against beam 1's
        # centreline SEGMENT and then allow a whole beam width of slack.  Two
        # errors that mostly cancelled, and when they did not it scored joints
        # that are visibly open: beam 1 landing 5 mm high left a 7 mm air gap
        # and still collected the +20.  A closed perimeter is a physical
        # condition -- the pieces touch -- so measure the pieces, not their
        # axes, and allow only the tolerance.
        gap = _rect_gap(_rect(*beams[0][:2], _beam_frame(beams[0][3])[0],
                              Piece.BEAM1_L, Piece.BEAM_W),
                        _rect(*beams[1][:2], _beam_frame(beams[1][3])[0],
                              Piece.BEAM2_L, Piece.BEAM_W))
        if gap <= BEAM_TOUCH_TOL:
            pts += 20
            detail.append((-1, "perimeter closed (T-joint gap %.1f mm)"
                           % max(0.0, gap), +20))
        else:
            detail.append((-1, "perimeter NOT closed: %.1f mm gap at the T-joint"
                           % gap, 0))
    return pts, detail


def _rect(x, y, yaw, length, width):
    """World corners of a beam's floor footprint."""
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    return np.array([(x + sx*length/2*c - sy*width/2*s,
                      y + sx*length/2*s + sy*width/2*c)
                     for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))])


def _rect_gap(A, B):
    """Separation between two convex polygons; 0 once they touch or overlap."""
    best = 0.0
    for P, Q in ((A, B), (B, A)):
        for i in range(len(P)):
            e = P[(i + 1) % len(P)] - P[i]
            n = np.array([-e[1], e[0]])
            n = n/np.linalg.norm(n)
            pa, qa = P @ n, Q @ n
            best = max(best, qa.min() - pa.max(), pa.min() - qa.max())
    return float(best)



# ===================================================== MISSION 2 (healthcare)
# Senior tables, rules 3.2.  130 points, and the reason a Mission-1-only robot
# does NOT score 120: three empty kit zones is -30 before anything is counted.
KIT_ZONES = {"HOSP": Field.HOSPITAL, "PCC_L": Field.PCC_L, "PCC_R": Field.PCC_R}
CYL_ZONES = dict(KIT_ZONES, RECOVERY=Field.RECOVERY)


def _zone_of(x, y, zones):
    for name, box in zones.items():
        if _in(box, x, y):
            return name
    return None


def score_kits(kit_xy):
    """kit_xy: list of (x_mm, y_mm).  Returns (points, breakdown)."""
    per = {z: 0 for z in KIT_ZONES}
    detail, placed = [], 0
    for x, y in kit_xy:
        z = _zone_of(x, y, KIT_ZONES)
        if z is not None:
            per[z] += 1; placed += 1
    pts = 3 * placed
    detail.append((-1, "%d kits in valid areas" % placed, 3 * placed))
    if all(per[z] == M2.KIT_PLAN[z] for z in KIT_ZONES):
        pts += 20; detail.append((-1, "correct full distribution 6/2/2", +20))
    empty = [z for z in KIT_ZONES if per[z] == 0]
    if empty:
        pts -= 10 * len(empty)
        detail.append((-1, "empty kit zones: " + ", ".join(sorted(empty)),
                       -10 * len(empty)))
    return pts, detail


def score_cylinders(cyl):
    """cyl: list of (x_mm, y_mm, colour).  Returns (points, breakdown)."""
    right, detail = 0, []
    per = {c: {} for c in M2.COLOURS}
    for x, y, col in cyl:
        z = _zone_of(x, y, CYL_ZONES)
        per[col][z] = per[col].get(z, 0) + 1
        want = M2.CYL_DEST.get(col)
        ok = (z in ("PCC_L", "PCC_R")) if col == "yellow" else (z == want)
        if ok:
            right += 1
    pts = 5 * right
    detail.append((-1, "%d patients in the correct zone" % right, 5 * right))
    if per["red"].get("HOSP", 0) == 4:
        pts += 6; detail.append((-1, "all red in the hospital", +6))
    yl, yr = per["yellow"].get("PCC_L", 0), per["yellow"].get("PCC_R", 0)
    if yl == 2 and yr == 2:
        pts += 8; detail.append((-1, "yellows split evenly between the PCCs", +8))
    if per["green"].get("RECOVERY", 0) == 4:
        pts += 6; detail.append((-1, "all green in the recovery zone", +6))
    return pts, detail


def score_match(disc_xyz, beams, kit_xy=(), cyl=()):
    """The whole 250.  Mission 2 arguments default to empty, which is exactly
    the Mission-1-only robot -- and it correctly returns +90, not +120."""
    out, total = [], 0
    for name, (p, d) in (("samples", score_discs(disc_xyz)),
                         ("beams",   score_beams(beams)),
                         ("kits",    score_kits(list(kit_xy))),
                         ("patients", score_cylinders(list(cyl)))):
        total += p; out.append((name, p, d))
    return total, out
