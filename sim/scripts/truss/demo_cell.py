"""Watch the cell make a truss.

    python3 sim/scripts/truss/demo_cell.py                 # headless, prints the log
    python3 sim/scripts/truss/demo_cell.py --gui           # MuJoCo's passive viewer
    python3 sim/scripts/truss/demo_cell.py --gui --1m      # the metre truss (slow)
    python3 sim/scripts/truss/demo_cell.py --gui --speed 2 # 2x real time

In the viewer: `[` and `]` cycle the fixed cameras (`cell`, `side`, and
the head's own `head_cam`); press 0 / Esc for the free camera.  The wound
bands appear as yellow sleeves as the turns are counted.

On Windows the interpreter is `python`, not `python3`, and the path is
relative to where you are: from the `sim` directory, run
`python scripts/truss/demo_cell.py --gui` (Windows takes forward slashes).
Do not set MUJOCO_GL -- truss/glenv.py picks one per platform, and
`osmesa` is a name that only exists on Linux.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from truss import glenv                  # noqa: E402  (before mujoco)
# A WINDOW NEEDS THE PLATFORM'S OWN CONTEXT.  --gui asks for none, so
# MuJoCo opens wgl on Windows, cgl on macOS, glx on a Linux desktop;
# headless we name a software one, and only where that name is legal.
# (Forcing osmesa here made `import mujoco` raise on Windows.)
if "--gui" not in sys.argv:
    glenv.headless()
import numpy as np
import mujoco

from truss import (structure, geometry, fixture, mjcf, cell, approach, schedule,
                   process, vision, inspector, band as _band)
from truss.geometry import TrussGeometry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--1m", dest="one_m", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0, help="wall-clock pacing; 0 = unpaced")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    t = structure.TRUSS_1M if args.one_m else structure.TRUSS_300
    g = TrussGeometry(t)
    fx = fixture.Fixture(g)
    print("planning %s: %d rods, %d joints" % (t.name, len(g.rods), t.n_joints))
    st = approach.plan_truss(g, fx, span=0.0)
    P = schedule.plan(g, fx, st)
    s = schedule.summary(P)
    print("plan: %.1f min  %s" % (s["total_min"], {k: round(v, 1) for k, v in s["phase_min"].items()}))
    xml = mjcf.scene_cell(g, fx, stage="empty")
    out = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "truss_cell_%s.xml" % t.name), "w") as f:
        f.write(xml)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    c = cell.CellSim(m, d, g, fx, rng=np.random.default_rng(args.seed))
    clk = cell.SimClock(m, d, c)
    for _ in range(25):
        clk.tick()
    vis = vision.ModelVision(c, rng=np.random.default_rng(args.seed + 1))
    ex = process.Executor(c, c, c, c, c, c, clk, g, fx, P, vision=vis, log=print)
    gen = ex.run()
    band_ids = {j.index: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "band%d" % j.index)
                for j in g.joints}

    def show_bands():
        for jix, stt in ex.states.items():
            gid = band_ids[jix]
            if gid < 0 or stt.turns <= 0:
                continue
            m.geom_size[gid][1] = max(stt.width, 0.2) / 2000.0
            cx = g.chord_point(g.joints[jix].chord, stt.centre)
            m.geom_pos[gid] = cx / 1000.0
            m.geom_rgba[gid][3] = min(1.0, 0.2 + stt.turns / t.turns)

    last_op = 0
    t_wall = time.time()

    def step():
        nonlocal last_op
        try:
            next(gen)
        except StopIteration:
            return False
        clk.tick()
        if len(ex.timeline) > last_op:
            last_op = len(ex.timeline)
            op = ex.timeline[-1][0]
            if op.kind in ("wind", "release", "index", "dose", "cut"):
                print("%7.1f  %s" % (d.time, op))
            show_bands()
        return True

    if args.gui and not glenv.windowed():
        print("--gui needs a display; this session has none.  Run without it, or set "
              "DISPLAY.")
        args.gui = False
    if args.gui:
        from mujoco import viewer as mjviewer
        with mjviewer.launch_passive(m, d) as v:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            v.cam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "side")
            while v.is_running():
                if not step():
                    break
                v.sync()
                if args.speed > 0:
                    lag = d.time / args.speed - (time.time() - t_wall)
                    if lag > 0:
                        time.sleep(min(lag, 0.05))
    else:
        while step():
            pass
    errs = {r.index: inspector.seat_error(g, r, *c.rod_pose(r.index), theta=c.cage_truth())
            for r in g.rods}
    rep = inspector.inspect_truss(g, ex.states, seated=errs, cycle_s=d.time)
    print(rep)
    print("simulated %.1f min against %.1f planned" % (d.time / 60.0, P.total / 60.0))


if __name__ == "__main__":
    main()
