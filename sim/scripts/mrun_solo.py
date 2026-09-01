"""SOLO board: robot 1 alone, robot 2 not even in the scene.

This is the control every fleet number has been missing.  A fleet mean is
only meaningful against the score the same seeds produce with robot 2
deleted -- otherwise "robot 2 scores 3 points on patients" hides the 20 it
may be costing robot 1 by standing in the hospital corridor.
"""
import os, sys, io, contextlib, multiprocessing as mp
os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, "/home/user/ART-Forum-2017-Stan-Tutorial/sim")
import numpy as np, mujoco
from rfgyc26 import mjcf, referee
from rfgyc26.params import Field, AgentA, M2
from rfgyc26.robot import AgentARobot
from rfgyc26.route import mission_agent_a
from rfgyc26 import planner

# NO ROBOT 2 MEANS NO PCC_R KITS.  The planner prices robot 1's PCC_L drop
# against what the fleet will deliver (F109); telling it to expect two kits
# that nothing is carrying would make this control flatter itself.
planner.FLEET_PCC_R = 0

CHUTE_OFFSET = AgentA.AXLE_X - AgentA.CHUTE_X


def random_discs(rng, n=3):
    pts = []
    while len(pts) < n:
        p = (rng.uniform(60, 245), rng.uniform(80, 230))
        if all(np.hypot(p[0]-q[0], p[1]-q[1]) > 90 for q in pts):
            pts.append(p)
    return pts


def one(seed, log=lambda *a, **k: None):
    rng = np.random.default_rng(seed)
    discs0 = random_discs(rng)
    xml = mjcf.scene_full_match(discs0, rng=rng, r2=False)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    rb = AgentARobot(m, d, rng=rng)
    g1 = mission_agent_a(rb, list(Field.LAB_HOLE_X), mjcf.LAB_HOLE_Y,
                         CHUTE_OFFSET, log=log, clock=lambda: d.time,
                         discs=discs0)
    d1 = False
    while d.time < 121.0:
        if not d1 and d.time >= 2.0:
            try:
                next(g1)
            except StopIteration:
                d1 = True
        for _ in range(20):
            mujoco.mj_step(m, d)
        if d1:
            break
    discs, beams, kits, cyl = [], [], [], []
    for i in range(3):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "disc%d" % i)
        p = d.xpos[b]*1000; discs.append((p[0], p[1], p[2]))
    for i in (1, 2):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "beam%d" % i)
        if b < 0: continue
        p = d.xpos[b]*1000
        beams.append((p[0], p[1], p[2], d.xquat[b].copy()))
    for i in range(M2.N_KITS):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "kit%d" % i)
        p_ = d.xpos[b]*1000; kits.append((p_[0], p_[1]))
    for i in range(M2.N_CYL):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cyl%d" % i)
        p_ = d.xpos[b]*1000
        g = m.geom_rgba[m.body_geomadr[b]]
        col = ("red" if g[0] > 0.6 and g[1] < 0.4 else
               ("green" if g[1] > 0.5 and g[0] < 0.5 else "yellow"))
        cyl.append((p_[0], p_[1], col))
    tot, out = referee.score_match(discs, beams, kits, cyl)
    part = {n: p for n, p, _ in out}
    det = "sam %3d/50  beam %3d/70  kit %3d/50  pat %3d/80" % (
        part.get("samples", 0), part.get("beams", 0),
        part.get("kits", 0), part.get("patients", 0))
    return seed, tot, det, d.time, part


def _w(s):
    with contextlib.redirect_stdout(io.StringIO()):
        return one(s)


if __name__ == "__main__":
    seeds = list(range(1, 13)) if len(sys.argv) < 2 else \
        [int(x) for x in sys.argv[1:]]
    with mp.Pool(min(12, len(seeds))) as pool:
        res = pool.map(_w, seeds)
    tots, cols = [], {"samples": [], "beams": [], "kits": [], "patients": []}
    print("seed  score      samples     beams       kits        patients"
          "     finish")
    for s, t_, det, tt, part in sorted(res):
        tots.append(t_)
        for k in cols:
            cols[k].append(part.get(k, 0))
        print("%4d  %4.0f/250   %s   %5.1f s" % (s, t_, det, tt))
    print("\n==============================================================")
    print("  SOLO MEAN  %.0f / 250        (best seed %.0f, worst %.0f)"
          % (np.mean(tots), max(tots), min(tots)))
    print("  columns:  sam %.1f/50  beam %.1f/70  kit %.1f/50  pat %.1f/80"
          % (np.mean(cols["samples"]), np.mean(cols["beams"]),
             np.mean(cols["kits"]), np.mean(cols["patients"])))
    print("==============================================================")
