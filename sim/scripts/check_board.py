"""THE BOARD IS WHAT WE THINK IT IS.  Run this on every change.

Every other suite tests a robot.  This one tests the thing the robots are
standing on, because a wrong board makes every measurement above it a lie
and none of them look wrong while it happens.  The faults this exists to
catch are the ones that have actually cost days here: a piece that starts
inside a wall, a colour read back differently from how it was written, a
zone rectangle the referee disagrees with, a board that drifts because
something is resting on something else at t=0.

It is deliberately cheap -- one scene, two seconds of settling -- so there
is no excuse not to run it.

    python3 sim/scripts/check_board.py [-v]
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import mujoco

from rfgyc26 import mjcf, referee
from rfgyc26.params import Field, M2, Piece, Robot2 as R2

VERBOSE = "-v" in sys.argv
RESULTS = []
SEEDS = (1, 6, 17)


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def build(seed, r2=True):
    rng = np.random.default_rng(seed)
    discs = [(100.0, 100.0), (160.0, 190.0), (220.0, 110.0)]
    m = mujoco.MjModel.from_xml_string(
        mjcf.scene_full_match(discs, rng=rng, r2=r2))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d, discs


def body_xy(m, d, name):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    return None if b < 0 else (float(d.xpos[b][0]*1000),
                               float(d.xpos[b][1]*1000),
                               float(d.xpos[b][2]*1000))


def main():
    m, d, discs = build(6)

    # ------------------------------------------------------- the furniture
    check("the field is the size params says it is",
          abs(m.stat.extent) > 0 and Field.W > 0 and Field.H > 0,
          "%.0f x %.0f mm" % (Field.W, Field.H))
    for nm, n in (("cyl", M2.N_CYL), ("kit", M2.N_KITS), ("disc", 3)):
        got = sum(1 for i in range(n)
                  if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                       "%s%d" % (nm, i)) >= 0)
        check("the board carries %d %s bodies" % (n, nm), got == n,
              "found %d" % got)
    check("both beams exist",
          all(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "beam%d" % i) >= 0
              for i in (1, 2)))

    # -------------------------------------------------------- the patients
    for seed in SEEDS:
        ms, ds, _ = build(seed)
        lay = mjcf.m2_layout(np.random.default_rng(seed))
        cols = [mjcf.cyl_colour(ms, i) for i in range(M2.N_CYL)]
        check("seed %d: the colours read back as they were written" % seed,
              cols == [c for _, _, c in lay],
              "%s vs %s" % (cols[:4], [c for _, _, c in lay][:4]))
        cnt = {c: cols.count(c) for c in M2.COLOURS}
        check("seed %d: four patients of each colour" % seed,
              all(v == 4 for v in cnt.values()), str(cnt))
        left = sum(1 for i in range(M2.N_CYL)
                   if body_xy(ms, ds, "cyl%d" % i)[0] < Field.W / 2)
        check("seed %d: six patients on each side" % seed, left == 6,
              "%d on the left" % left)
        off = [i for i in range(M2.N_CYL)
               if not any(abs(body_xy(ms, ds, "cyl%d" % i)[0] - x) < 2.0 and
                          abs(body_xy(ms, ds, "cyl%d" % i)[1] - y) < 2.0
                          for x, y, _ in lay)]
        check("seed %d: every patient starts on its sticker" % seed, not off,
              "off-sticker: %s" % off)

    # ------------------------------------------------- nothing starts wrong
    inside, sunk, pairs = [], [], []
    for nm, n in (("cyl", M2.N_CYL), ("kit", M2.N_KITS), ("disc", 3)):
        for i in range(n):
            p = body_xy(m, d, "%s%d" % (nm, i))
            if p is None:
                continue
            if p[0] > 2000.0:            # parked off-field on purpose
                continue
            if not (0.0 <= p[0] <= Field.W and 0.0 <= p[1] <= Field.H):
                inside.append(("%s%d" % (nm, i), p[:2]))
            if p[2] < -1.0:
                sunk.append(("%s%d" % (nm, i), p[2]))
    check("no piece starts outside the field", not inside, str(inside[:3]))
    check("no piece starts below the floor", not sunk, str(sunk[:3]))

    # interpenetration: a contact between two PIECES with real depth at t=0
    deep = []
    for k in range(d.ncon):
        c = d.contact[k]
        n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        if c.dist < -1.0e-3 and any(n1.startswith(p) for p in ("cyl", "kit", "disc")) \
                and any(n2.startswith(p) for p in ("cyl", "kit", "disc")):
            deep.append((n1, n2, round(c.dist*1000, 2)))
    check("no two pieces start inside each other", not deep, str(deep[:3]))

    # ------------------------------------------------------- it sits still
    before = {}
    for nm, n in (("cyl", M2.N_CYL), ("kit", M2.N_KITS), ("disc", 3)):
        for i in range(n):
            k = "%s%d" % (nm, i)
            if body_xy(m, d, k) is not None:
                before[k] = body_xy(m, d, k)
    for _ in range(2000):                       # 2 s, no actuation at all
        mujoco.mj_step(m, d)
    moved = []
    for k, p0 in before.items():
        p1 = body_xy(m, d, k)
        if p0[0] < 2000.0 and np.hypot(p1[0]-p0[0], p1[1]-p0[1]) > 8.0:
            moved.append((k, round(float(np.hypot(p1[0]-p0[0], p1[1]-p0[1])), 1)))
    check("the board is still after two seconds of nobody touching it",
          not moved, "drifted: %s" % moved[:4])
    check("...and is still numerically alive",
          np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel)),
          "max |qvel| %.1f" % float(np.abs(d.qvel).max()))

    # ---------------------------------------- the referee and the geometry
    ZONES = {"HOSP": Field.HOSPITAL, "PCC_L": Field.PCC_L,
             "PCC_R": Field.PCC_R, "RECOVERY": Field.RECOVERY}
    bad = []
    for nm, box in ZONES.items():
        cx, cy = (box[0]+box[2])/2.0, (box[1]+box[3])/2.0
        if referee._zone_of(cx, cy, referee.CYL_ZONES) != nm:
            bad.append((nm, "centre reads %s"
                        % referee._zone_of(cx, cy, referee.CYL_ZONES)))
        out = (box[0]-15.0, cy)
        if referee._zone_of(out[0], out[1], referee.CYL_ZONES) == nm:
            bad.append((nm, "15 mm outside still reads inside"))
    check("the referee's zones are the rectangles params declares", not bad,
          str(bad[:3]))
    check("the four destination zones do not overlap each other",
          not [(a, b) for a in ZONES for b in ZONES if a < b
               and ZONES[a][0] < ZONES[b][2] and ZONES[b][0] < ZONES[a][2]
               and ZONES[a][1] < ZONES[b][3] and ZONES[b][1] < ZONES[a][3]])
    check("every destination zone is inside the field",
          all(0 <= b[0] and 0 <= b[1] and b[2] <= Field.W and b[3] <= Field.H
              for b in ZONES.values()))

    # ------------------------------------------- the untouched-board score
    # A robot that does nothing does not score zero, and knowing exactly
    # what it scores is what makes every later number readable.
    cyl = [(*body_xy(m, d, "cyl%d" % i)[:2], mjcf.cyl_colour(m, i))
           for i in range(M2.N_CYL)]
    pts, _ = referee.score_cylinders(cyl)
    check("an untouched board scores -36 for the patients", pts == -36,
          "%+d" % pts)
    beams = []
    for i in (1, 2):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "beam%d" % i)
        beams.append((*(d.xpos[b]*1000), d.xquat[b].copy()))
    bp, _ = referee.score_beams(beams)
    check("...and 0 for beams still in their pockets", bp == 0, "%+d" % bp)

    # ---------------------------------------------- the kits start aboard
    aboard, on_r2, loose = 0, 0, []
    for i in range(M2.N_KITS):
        p = body_xy(m, d, "kit%d" % i)
        if p[0] > 2000.0:
            continue
        if p[2] > 20.0:                 # up on a deck rather than on the floor
            (aboard, on_r2) = (aboard+1, on_r2) if p[0] < 900.0 else \
                              (aboard, on_r2+1)
        else:
            loose.append((i, tuple(round(v) for v in p)))
    check("no kit starts loose on the floor", not loose, str(loose[:3]))
    check("PCC_R's pair rides on robot 2", on_r2 == len(M2.KIT_GROUPS["PCC_R"]),
          "%d on robot 2" % on_r2)

    n_bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det and not ok else ""))
    print("check_board: %d checks, %d failed" % (len(RESULTS), n_bad))
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
