"""Phase 0b: MODEL truth, with physics.  Run this on every change.

check_geometry.py re-derives params against each other.  It cannot see what
the BUILT model actually looks like, and that is where the expensive faults
have all been: a knife commanded to droop with nothing to stop it, sitting
3.75 mm under the floor; a razor skived the wrong way, presenting the plate's
end as a 0.30 mm wall; capsules whose end caps put the finger tips 3 mm proud
of the circle the part makes; a hold-down clamp 8 mm tall on a path that has
to pass a 20 mm patient.  Every one of those was found by sweeping parameters
for hours and wondering why nothing moved.

So: build the scene, settle it, command the intake to COLLECT, and then
interrogate the geometry that results.  Each check is a sentence about the
machine that a person could argue with.  Anything red is either a bug or a
design decision that has not been made.

    python3 sim/scripts/check_model.py [-v]
"""
import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mujoco
from rfgyc26 import mjcf
from rfgyc26.params import AgentA, Chassis, Piece, Field
import rfgyc26.params as P
from rfgyc26.robot import AgentARobot

VERBOSE = "-v" in sys.argv
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


# --------------------------------------------------------------- the model
def build(settle=2000):
    xml = mjcf.scene_full_match([(2500., 2400.), (2600., 2500.), (2700., 2600.)],
                                robot_pose=(300., 600., 0.),
                                rng=np.random.default_rng(0), kits_aboard=False)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    rb = AgentARobot(m, d)
    rb.fingers(True); rb.gate(False); rb.blade(False); rb.feed(False)
    mujoco.mj_forward(m, d)
    rb.intake(True)                       # knife DOWN, brush spinning
    for _ in range(settle):
        mujoco.mj_step(m, d)
    return m, d, rb


def gname(m, g):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "?"


def bname(m, g):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "?"


PIECE_MASK = 15                            # what a piece's conaffinity can feel
BRUSH_BODIES = ("A_drum", "A_arm", "A_up")


def is_brush(m, g):
    b = bname(m, g)
    return b.startswith("A_f") or b in BRUSH_BODIES


def piece_visible(m, g):
    return bool(m.geom_contype[g] & PIECE_MASK)


def is_finger(m, g):
    return bname(m, g).startswith("A_f")


def lowest_z(m, d, g):
    """World z of the geom's lowest point, from MuJoCo's own local AABB so it
    is right for every geom type and orientation."""
    R = d.geom_xmat[g].reshape(3, 3)
    c = m.geom_aabb[g][:3] * 1000.0
    h = m.geom_aabb[g][3:] * 1000.0
    z = d.geom_xpos[g][2] * 1000.0 + float(R[2] @ c)
    return z - float(sum(abs(R[2, i]) * h[i] for i in range(3)))


def to_world(rb, d, xa, y, z):
    """Robot-frame (Xa, y, z) in mm -> world metres."""
    rp = d.xpos[rb.bid]
    R = d.xmat[rb.bid].reshape(3, 3)
    return rp + R @ (np.array([xa - AgentA.AXLE_X, y, z]) / 1000.0)


def cast(m, d, rb, xa, y, z0, up=True, skip=lambda g: False):
    """Ray from (xa,y,z0) straight up or down; first surface a piece can feel.
    Returns (z_mm, geom_name) or (None, '-')."""
    gid = np.zeros(1, np.int32)
    vec = np.array([0.0, 0.0, 1.0 if up else -1.0])
    R = d.xmat[rb.bid].reshape(3, 3)
    vec = R @ vec
    z = z0
    for _ in range(12):
        p = to_world(rb, d, xa, y, z)
        dist = mujoco.mj_ray(m, d, p, vec, None, 1, -1, gid)
        if gid[0] < 0 or dist < 0:
            return None, "-"
        g = int(gid[0])
        hit = z + (dist * 1000.0) * (1.0 if up else -1.0)
        if piece_visible(m, g) and not skip(g):
            return hit, gname(m, g)
        z = hit + (0.2 if up else -0.2)
    return None, "-"


# ------------------------------------------------------------------ probes
def transport_profile(m, d, rb, xs, y=0.0):
    """Top of the surface a piece rides at each station -- the knife plane
    forward of the belt nose, the belt aft of it.  The brush is not a
    transport surface, so it is skipped."""
    out = []
    for xa in xs:
        z, n = cast(m, d, rb, xa, y, 60.0, up=False, skip=lambda g: is_brush(m, g))
        out.append((xa, z, n))
    return out


def roof_profile(m, d, rb, xs, y=0.0):
    """Bottom of the first thing above the transport surface."""
    out = []
    for xa, z, _ in transport_profile(m, d, rb, xs, y):
        if z is None:
            out.append((xa, None, None, "-")); continue
        r, n = cast(m, d, rb, xa, y, z + 0.6, up=True,
                    skip=lambda g: is_finger(m, g))
        out.append((xa, z, r, n))
    return out


def main():
    m, d, rb = build()
    XS = [float(x) for x in range(int(AgentA.CHUTE_X) + 10, 286, 1)]

    # ---------------------------------------------------------- 1. the knife
    bl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "A_shim_blade")
    pl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "A_shim_g")
    tip = rb.to_local(d.geom_xpos[bl])
    tip_z = tip[2]
    check("the knife's cutting edge sits just under the field, not buried",
          -1.5 <= tip_z <= 0.0,
          "razor centre z %+.2f mm (want -1.5 .. 0.0)" % tip_z)

    sj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "A_shim_j")
    sa = math.degrees(d.qpos[m.jnt_qposadr[sj]])
    check("the knife rests on a stop when the servo presses it down",
          abs(sa) <= 0.5,
          "shim joint %+.2f deg under a %+.1f deg command" % (sa, AgentA.SHIM_DROOP))

    # razor and plate must be ONE ramp: no step where they meet
    prof = transport_profile(m, d, rb, [float(x) for x in range(
        int(AgentA.BELT_NOSE_X), int(AgentA.SHIM_TIP_X) + 8)])
    zs = [(x, z) for x, z, _ in prof if z is not None]
    steps = [(zs[i+1][0], zs[i][1] - zs[i+1][1]) for i in range(len(zs)-1)]
    worst = max(steps, key=lambda t: t[1]) if steps else (0, 0.0)
    check("the knife presents ONE unbroken ramp -- no step to climb",
          worst[1] <= 1.0,
          "biggest step %.2f mm at Xa %.0f" % (worst[1], worst[0]))

    # ------------------------------------------------ 2. nothing below the floor
    bad = []
    for g in range(m.ngeom):
        n = gname(m, g)
        if not n.startswith("A_") or not piece_visible(m, g):
            continue
        if n == "A_shim_blade" or n.startswith("A_ball"):
            continue      # the razor wedges under by design; the castors are sprung
        lo = lowest_z(m, d, g)
        if lo < -1.5:
            bad.append((n, lo))
    check("nothing a piece can feel is buried in the field (bar razor and castors)",
          not bad, "; ".join("%s %.1f" % b for b in bad[:4]))

    # --------------------------------------------------------- 3. the throat
    roofs = roof_profile(m, d, rb, XS)
    tight = min(((r - s, x, n) for x, s, r, n in roofs
                 if s is not None and r is not None), default=(999, 0, "-"))
    # SAMPLES ONLY, since F76.  The patients go to a second robot, so the
    # throat is sized for a 5 mm disc and nothing else has to fit.
    check("the throat passes a sample past every RIGID thing",
          tight[0] >= Piece.DISC_T + 2.0,
          "tightest %.1f mm at Xa %.0f (%s); a sample is %.0f"
          % (tight[0], tight[1], tight[2], Piece.DISC_T))

    gaps = [x for x, z, _ in prof if z is None]
    check("the conveying surface is continuous from knife tip to belt tail",
          not gaps, "no surface at Xa %s" % (gaps[:6],))

    # ---------------------------------------------------------- 4. the brush
    dr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "A_drum")
    tubes = [g for g in range(m.ngeom) if gname(m, g).startswith("A_fd")]
    rmax = 0.0
    for g in tubes:
        R = d.geom_xmat[g].reshape(3, 3)
        for s_ in (1, -1):
            v = (d.geom_xpos[g] + R[:, 2] * m.geom_size[g][1] * s_ - d.xpos[dr]) * 1000.0
            rmax = max(rmax, math.hypot(v[0], v[2]) + m.geom_size[g][0] * 1000.0)
    if tubes:
        check("the fingers sweep the circle the PART makes, not the model's",
              abs(rmax - AgentA.ROLL_TIP_R) <= 1.0,
              "swept radius %.1f mm, ROLL_TIP_R %.1f" % (rmax, AgentA.ROLL_TIP_R))

    dj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "A_drum_j")
    w = abs(d.qvel[m.jnt_dofadr[dj]])
    wc = 2 * math.pi * AgentA.ROLL_RPM / 60.0
    check("the brush actually turns at the speed it is told to, unloaded",
          w >= 0.90 * wc, "%.1f of %.1f rad/s (%.0f%%)" % (w, wc, 100 * w / wc))

    ax = rb.to_local(d.xpos[dr])
    ramp_here, _ = cast(m, d, rb, ax[0] + AgentA.AXLE_X, 0.0, 60.0,
                        up=False, skip=lambda g: is_brush(m, g))
    gap = (ax[2] - rmax) - (ramp_here if ramp_here is not None else 0.0)
    check("the brush clears the surface under it, so pieces carry it, not the ramp",
          gap >= -0.5, "finger tips %.1f mm %s the surface"
          % (abs(gap), "below" if gap < 0 else "above"))

    hub = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "A_drum_g")
    hub_lo = lowest_z(m, d, hub)
    surf_here = ramp_here if ramp_here is not None else 0.0
    lift = P._arm_lift()
    check("the roller can open far enough to admit a sample",
          hub_lo + lift - surf_here >= Piece.DISC_T,
          "hub bottom %.1f + %.1f of arm travel over a %.1f surface = %.1f; "
          "a sample is %.0f" % (hub_lo, lift, surf_here,
                                hub_lo + lift - surf_here, Piece.DISC_T))
    check("...and its driver can reach the speed it is commanded",
          AgentA.ROLL_RPM <= AgentA.ROLL_RPM_MAX,
          "%.0f rpm commanded, driver ceiling %.0f" % (AgentA.ROLL_RPM,
                                                       AgentA.ROLL_RPM_MAX))

    aj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "A_arm_j")
    if aj >= 0:
        arm = math.degrees(d.qpos[m.jnt_dofadr[aj]])
        check("...and its arm is resting on its own stop, not held up",
              abs(arm) <= 1.5, "arm %+.2f deg off its stop" % arm)

    # ------------------------------------------------------- 5. masks & pairs
    fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    def touches(a, b):
        return bool((m.geom_contype[a] & m.geom_conaffinity[b]) or
                    (m.geom_contype[b] & m.geom_conaffinity[a]))
    disc = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "disc0_g")
    cyl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cyl0_g")
    must = [("floor", fl), ("A_shim_g", pl), ("A_shim_blade", bl)]
    if tubes: must.append(("a finger", tubes[0]))
    for lbl, pid in (("a sample", disc), ("a patient", cyl)):
        miss = [n for n, g in must if g >= 0 and not touches(pid, g)]
        check("%s can feel the floor, the knife and the brush" % lbl,
              not miss, "cannot feel %s" % miss)

    if tubes:
        check("the fingers cannot feel the floor (they would sweep it flat)",
              not touches(tubes[0], fl))

    # MuJoCo takes the element-wise MAX of two frictions, so any surface meant
    # to be slippery needs an explicit <pair> or it silently runs at the
    # piece's own 0.6.  Check the ones the design leans on.
    pairs = {}
    for i in range(m.npair):
        pairs[(m.pair_geom1[i], m.pair_geom2[i])] = m.pair_friction[i][0]
        pairs[(m.pair_geom2[i], m.pair_geom1[i])] = m.pair_friction[i][0]
    slippery = ["A_shim_g", "A_shim_blade", "A_guide_l", "A_guide_r",
                "A_lane_l", "A_lane_r"]
    for lbl, pid in (("sample", disc), ("patient", cyl)):
        miss = []
        for n in slippery:
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
            if g >= 0 and pid >= 0 and (pid, g) not in pairs:
                miss.append(n)
        check("every slippery surface is declared slippery for a %s" % lbl,
              not miss, "running at MU_PIECE against %s" % miss[:4])

    # ---------------------------------------------------------- 6. dynamics
    zero = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "?"
            for b in range(1, m.nbody)
            if m.body_dofnum[b] > 0 and m.body_mass[b] <= 0.0]
    check("no free body has zero mass", not zero, "%s" % zero[:4])

    nan = not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel)))
    check("the model is numerically alive after settling", not nan)

    # ------------------------------------------------------------- 7. speeds
    check("the belt's surface runs AFT in the world during a sweep",
          Chassis.BELT_SPEED > P.SWEEP_SPEED_A,
          "belt %.0f vs sweep %.0f mm/s -> surface %+.0f mm/s"
          % (Chassis.BELT_SPEED, P.SWEEP_SPEED_A,
             P.SWEEP_SPEED_A - Chassis.BELT_SPEED))

    # ------------------------------------------------------------- report
    print("RFGYC'26 -- BUILT MODEL, settled and collecting\n" + "-" * 70)
    bad = 0
    for name, ok, detail in RESULTS:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", name))
        if detail and (VERBOSE or not ok):
            print("         %s" % detail)
        bad += not ok
    print("-" * 70)
    print("%d of %d model checks pass" % (len(RESULTS) - bad, len(RESULTS)))
    if VERBOSE:
        print("\n  throat profile (surface -> roof), every 10 mm")
        for x, s, r, n in roofs[::10]:
            print("   Xa %5.0f  rides %6s   roof %6s %-16s clear %s"
                  % (x, "%.1f" % s if s is not None else "-",
                     "%.1f" % r if r is not None else "-", n,
                     "%.1f" % (r - s) if (r is not None and s is not None) else "open"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
