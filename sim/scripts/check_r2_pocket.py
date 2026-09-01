"""Phase 0i: robot 2's capture pocket is EMPTY, and the planners believe the
robot that actually exists.

This check exists because of F108, which is the most expensive kind of bug
this project has produced: a mechanism that could not work, on any board, in
any seed, from the day it was built, while every layer above it reported
plausible-looking failures.  The chassis deck ran 25 mm past the flare tips
through the exact height band a patient occupies, so the funnel opened onto
a wall.  Twelve of twelve deliveries came back "capture missed"; the puck's
position in the chassis frame sat pinned at x = 105.0 -- the deck's front
edge plus a puck radius -- while the robot drove into it.

Nothing tested the pocket's own geometry.  So:

  * the clear volume is DECLARED in params.Robot2 and asserted here;
  * the footprint the planners use is asserted to CONTAIN every collidable
    geom, so the F107 class of bug (planners describing a shorter robot than
    the one that has to fit) cannot come back silently either.

    python3 sim/scripts/check_r2_pocket.py [-v]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
from rfgyc26 import mjcf, nav, robot2
from rfgyc26.params import Robot2 as R2, Piece

VERBOSE = "-v" in sys.argv
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def geoms():
    """Every COLLIDABLE geom of robot 2 and its child bodies, as
    (name, lo[3], hi[3]) in the CHASSIS frame with z from the floor.

    Read from the world through mj_forward and transformed back, not from
    geom_pos: a geom on a child body (the wheels, the retention fingers)
    carries a position local to THAT body, and reading it as if it were
    chassis-local put the wheels on the centreline and failed the clear
    volume for a robot that was fine.
    """
    xml = mjcf.scene_full_match([(2500., 2400.), (2600., 2500.), (2700., 2600.)],
                                rng=np.random.default_rng(0), r2=True,
                                r2_pose=(1055., 140., 90.))
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    b2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot2")
    kin = {b2}
    for b in range(m.nbody):
        if m.body_parentid[b] in kin:
            kin.add(b)
    rx, ry = d.xpos[b2][:2] * 1000.0
    q = d.xquat[b2]
    yaw = np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))
    c, sn = np.cos(-yaw), np.sin(-yaw)
    out = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] not in kin:
            continue
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue                                   # visual only
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "?"
        p = d.geom_xpos[g] * 1000.0
        s_ = m.geom_size[g] * 1000.0
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            s_ = np.array([s_[0], s_[0], s_[0]])
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            s_ = np.array([s_[0], s_[0], s_[1]])
        # the geom's own world orientation widens its axis-aligned box
        R = d.geom_xmat[g].reshape(3, 3)
        half = np.abs(R) @ s_
        wlo = np.array([p[0]-half[0], p[1]-half[1], p[2]-half[2]])
        whi = np.array([p[0]+half[0], p[1]+half[1], p[2]+half[2]])
        # world -> chassis (the box stays axis-aligned under a 90 deg yaw,
        # which is why the rig spawns the robot at exactly 90)
        cor = []
        for X in (wlo, whi):
            dx, dy = X[0] - rx, X[1] - ry
            cor.append((dx*c - dy*sn, dx*sn + dy*c, X[2]))
        lo = np.array([min(cor[0][k], cor[1][k]) for k in range(3)])
        hi = np.array([max(cor[0][k], cor[1][k]) for k in range(3)])
        out.append((nm, lo, hi))
    return out


POCKET = {"r2_stop", "r2_wall_l", "r2_wall_r", "r2_flare_l",
          "r2_flare_r", "r2_finger_l", "r2_finger_r"}
WHEELS = {"r2_wg_l", "r2_wg_r"}       # outboard, above the puck


def main():
    gs = geoms()
    check("robot 2 has geometry at all", len(gs) >= 8, "%d geoms" % len(gs))

    # ---- 1. THE CLEAR VOLUME ------------------------------------------
    # A patient is a 20 mm cylinder standing on the floor; the pocket must
    # be able to swallow one from the flare tips down to the back stop.
    tip_x = R2.STOP_X + R2.POCKET_D + R2.FLARE_L * np.cos(np.radians(R2.FLARE_ANG))
    # the volume starts at the STOP's front face, not its centre: the
    # mass block is allowed to sit right behind the stop, and does
    x0, x1 = R2.STOP_X + 3.0, tip_x + 12.0
    y_half, z_top = 40.0, 24.0
    if VERBOSE:
        print("  clear volume  x %.0f..%.0f  |y|<=%.0f  z 0..%.0f"
              % (x0, x1, y_half, z_top))
    bad = []
    for nm, lo, hi in gs:
        if nm in POCKET or nm in WHEELS:
            continue
        if (hi[0] > x0 and lo[0] < x1 and hi[1] > -y_half and lo[1] < y_half
                and hi[2] > 0.0 and lo[2] < z_top):
            bad.append("%s x[%.0f,%.0f] y[%.0f,%.0f] z[%.0f,%.0f]"
                       % (nm, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
    check("no chassis geom stands in the capture pocket (F108)",
          not bad, "; ".join(bad))

    # and the pocket must actually be deep enough to hold what it catches
    check("the pocket is deeper than a patient is wide",
          R2.POCKET_D >= Piece.CYL_D, "%.0f vs %.0f mm" % (R2.POCKET_D, Piece.CYL_D))
    check("the throat passes a patient with slop but not two",
          Piece.CYL_D + 8.0 < R2.POCKET_W < 2.0 * Piece.CYL_D,
          "throat %.0f, patient %.0f" % (R2.POCKET_W, Piece.CYL_D))
    mouth = R2.POCKET_W/2.0 + R2.FLARE_L*np.sin(np.radians(R2.FLARE_ANG))
    check("the funnel mouth is wider than the throat",
          mouth > R2.POCKET_W/2.0 + 8.0, "+-%.0f vs +-%.0f mm"
          % (mouth, R2.POCKET_W/2.0))

    # ---- 2. THE PLANNERS DESCRIBE THIS ROBOT (F107) --------------------
    # Every collidable geom must lie inside the footprint nav plans with,
    # and inside the circumscribed radius the costmap inflates by.
    hull = nav.BODY_PTS
    fx = max(p[0] for p in hull)
    rx = min(p[0] for p in hull)
    fy = max(abs(p[1]) for p in hull)
    over = []
    worst_r = 0.0
    for nm, lo, hi in gs:
        if hi[0] > fx + 1e-6 or lo[0] < rx - 1e-6 or max(abs(lo[1]), abs(hi[1])) > fy + 1e-6:
            over.append("%s x[%.0f,%.0f] y+-%.0f"
                        % (nm, lo[0], hi[0], max(abs(lo[1]), abs(hi[1]))))
        for cx in (lo[0], hi[0]):
            for cy in (lo[1], hi[1]):
                worst_r = max(worst_r, float(np.hypot(cx, cy)))
    check("every geom lies inside nav.BODY_PTS' extent (F107)",
          not over, "footprint x[%.0f,%.0f] y+-%.0f -- outside: %s"
          % (rx, fx, fy, "; ".join(over)))
    check("R2_CIRCUM covers the real circumscribed radius",
          robot2.R2_CIRCUM >= worst_r - 1e-6,
          "circum %.0f, real %.0f mm" % (robot2.R2_CIRCUM, worst_r))
    check("R2_INSCRIBED is not larger than R2_CIRCUM",
          robot2.R2_INSCRIBED <= robot2.R2_CIRCUM,
          "%.0f <= %.0f" % (robot2.R2_INSCRIBED, robot2.R2_CIRCUM))

    # THE PINCHES.  The chassis is narrow precisely so it can pass the
    # field's two 191 mm gaps; planning radius is what decides whether it
    # may try.  This is the constraint F107 traded against pocket reach.
    check("the planning radius still passes the field's 191 mm pinches",
          2.0 * robot2.R2_INSCRIBED < 191.0 - 20.0,
          "2*%.0f = %.0f mm of 191" % (robot2.R2_INSCRIBED,
                                       2.0*robot2.R2_INSCRIBED))

    # ---- 3. holding() AGREES WITH THE POCKET IT TESTS ------------------
    # A seated puck sits at CAPTURE_X on the centreline; holding() must say
    # yes there, and no once the puck is out past the flare tips.
    class _Fake:
        x = y = th = 0.0
    f = _Fake()
    inside = robot2.R2Controller.holding(f, R2.CAPTURE_X, 0.0)
    outside = robot2.R2Controller.holding(f, tip_x + 45.0, 0.0)
    aside = robot2.R2Controller.holding(f, R2.CAPTURE_X, mouth + 30.0)
    check("holding() is true for a puck seated at CAPTURE_X", inside)
    check("holding() is false for a puck shed off the nose", not outside)
    check("holding() is false for a puck knocked off to the side", not aside)
    check("a seated puck sits between the stop and the walls' end",
          R2.STOP_X < R2.CAPTURE_X < R2.STOP_X + R2.POCKET_D,
          "%.0f in (%.0f, %.0f)" % (R2.CAPTURE_X, R2.STOP_X,
                                    R2.STOP_X + R2.POCKET_D))

    ok = sum(1 for _, o, _ in RESULTS if o)
    for name, o, detail in RESULTS:
        if not o or VERBOSE:
            print("  %s %s%s" % ("PASS" if o else "FAIL", name,
                                 ("   [%s]" % detail) if detail else ""))
    print("\ncheck_r2_pocket: %d / %d" % (ok, len(RESULTS)))
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
