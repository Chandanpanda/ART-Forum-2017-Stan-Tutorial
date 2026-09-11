"""Station B making one head kit, in MuJoCo.

    python3 sim/scripts/truss/demo_stationb.py                  # headless
    python3 sim/scripts/truss/demo_stationb.py --sweep DIR      # frames, tiled
    python3 sim/scripts/truss/demo_stationb.py --gui            # watch it

WHAT IT RUNS.  The collar onto the lens housing on a four-pad vacuum head
and its four aperture walls dosed; the four grid rods laid ON A JIG, in
clear air, layer 0 then layer 1; each of the four crossings WOUND there and
dosed; the finished frame -- one part now -- lifted off the jig by a layer-1
rod and set on the collar; the two bands it lands on beaded.  The module
goes into the nest by hand and the finished kit comes out by hand, which is
where the QC step belongs.

Read the sweep.  `Kit` and `mount.solve` agree on the crossings to 1e-14 mm
and that is worth nothing if the machine cannot reach them; only the frames
say whether it did.  Everything expensive in this station was found in the
pictures and nothing was found in the numbers: the winder centred on a
crossing that had a camera under it, a collar drawn as a solid plate lying
over the lens, jaws that opened when the HAL said close, and a ring lowered
onto a joint through its own rim because parking it only ever stopped the
drive.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from truss import glenv                          # noqa: E402
if "--gui" not in sys.argv:
    glenv.headless()
import numpy as np                               # noqa: E402
import mujoco                                    # noqa: E402

from truss import (structure, geometry, fixture, mjcf, mount, stationb,   # noqa: E402
                   filmstrip)
from truss.spec import StationB                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--1m", dest="metre", action="store_true")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--sweep", metavar="DIR")
    ap.add_argument("--sweep-fps", type=float, default=None,
                    help="default: whatever gives about 120 frames over the "
                         "planned run, floored at the 0.1 fps sweep rate")
    ap.add_argument("--window", nargs=2, type=float, metavar=("T0", "T1"),
                    help="1 fps into a section that looked wrong")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--in-cell", action="store_true",
                    help="build B inside the whole plant instead of on its own "
                         "-- the same physics, and ten times the wall clock")
    a = ap.parse_args()

    t = structure.TRUSS_1M if a.metre else structure.TRUSS_300
    g = geometry.TrussGeometry(t)
    M = mount.solve(g, d_strut=t.d_diag,
                    standoff_mm=mount.fov_standoff(g, d_strut=t.d_diag))
    g.attach_mount(M)
    fx = fixture.Fixture(g)
    kit = stationb.Kit(d_rod=t.d_diag)
    P = stationb.plan(kit)
    print("station B, kit for %s: %d ops, %.2f min planned"
          % (t.name, len(P.ops), P.total / 60.0))

    if a.in_cell:
        m = mujoco.MjModel.from_xml_string(mjcf.scene_cell(g, fx, kit=kit, n_drops=16))
    else:
        m = mujoco.MjModel.from_xml_string(mjcf.scene_station_b(kit))
    d = mujoco.MjData(m)
    c = stationb.CellB(m, d, kit,
                       origin=StationB.ORIGIN if a.in_cell else (0.0, 0.0, 0.0),
                       rng=np.random.default_rng(a.seed))
    for _ in range(50):
        c.tick_hook(m.opt.timestep)
        mujoco.mj_step(m, d)

    strip = None
    if a.sweep:
        os.makedirs(a.sweep, exist_ok=True)
        rend = mujoco.Renderer(m, height=560, width=740)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        o = StationB.ORIGIN if a.in_cell else (0.0, 0.0, 0.0)
        # LOOK AT THE WHOLE TABLE.  The racks, the nest and the jig are the
        # three places B works and a frame that misses one of them cannot
        # show the thing that went wrong there.
        ex, ey, ez = kit.extent()
        lo_y = min(float(s_[2][1]) for s_ in kit.rack_slots())
        hi_y = float(kit.jig_y()) + kit.gu + kit.d
        cam.lookat[:] = [o[0] * 1e-3, (o[1] + 0.5 * (lo_y + hi_y)) * 1e-3,
                         (o[2] + ez / 2.0) * 1e-3]
        cam.distance = 2.0 * max(ex, 0.5 * (hi_y - lo_y)) * 1e-3
        cam.azimuth, cam.elevation = 122.0, -26.0
        if a.sweep_fps:
            fps = a.sweep_fps
        elif a.window:
            fps = 1.0
        else:
            fps = max(0.1, 120.0 / max(P.total, 1.0))
        strip = filmstrip.Filmstrip(rend, cam, fps=fps,
                                    window=tuple(a.window) if a.window else None)
        print("sweeping at %.2f fps%s" % (fps, " over %g..%g s" % tuple(a.window)
                                          if a.window else ""))

    msgs = []
    run = stationb.Runner(c, P, log=lambda s: (msgs.append(s), print("   ", s)))
    gen = run.run()
    last = 0
    t0 = time.time()
    steps = int(round(1.0 / (Runner_hz := 50.0) / m.opt.timestep))
    while True:
        try:
            next(gen)
        except StopIteration:
            break
        for _ in range(steps):
            c.tick_hook(m.opt.timestep)
            mujoco.mj_step(m, d)
        if len(run.timeline) > last:
            last = len(run.timeline)
            op = run.timeline[-1][0]
            if op.kind in ("wind", "release", "vac", "bond", "grip"):
                print("%7.1f  %s" % (d.time, op))
        if strip is not None:
            strip.maybe(d.time, d)
    if strip is not None:
        strip.maybe(d.time + 1e9, d)

    print("ran %d of %d ops in %.0f s wall, %.1f min simulated"
          % (len(run.timeline), len(P.ops), time.time() - t0, d.time / 60.0))
    print("wound %d of 4 crossings %s; %d bonds"
          % (sum(1 for v in c.wound.values() if v > 0.5),
             {k: round(v, 1) for k, v in sorted(c.wound.items())}, len(c.bonds)))
    # WHERE THE KIT ACTUALLY ENDED UP, against where `Kit` says it goes.
    # Not the model against itself: `Kit` is the same object `mount.solve`
    # lands its struts on, so this is the kit measured against the truss.
    # MEASURED AT THE ROD'S ENDS, not at its centre -- a rod rolled ten
    # degrees about its own axis has a centre error of zero, and it was
    # reading zero while the frames showed the thing lying over.
    o = np.asarray(StationB.ORIGIN if a.in_cell else (0.0, 0.0, 0.0), float)
    worst = 0.0
    for r in kit.rods:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bgrid%d" % r.index)
        R = d.xmat[b].reshape(3, 3)
        c = d.xpos[b] * 1000.0 - o
        half = R @ np.array([r.length / 2.0, 0.0, 0.0])   # racked along x
        got = (c - half, c + half)
        e = min(max(float(np.linalg.norm(got[0] - r.p0)),
                    float(np.linalg.norm(got[1] - r.p1))),
                max(float(np.linalg.norm(got[1] - r.p0)),
                    float(np.linalg.norm(got[0] - r.p1))))
        worst = max(worst, e)
        print("   grid%d  ends %6.2f mm off plan   at %s .. %s"
              % (r.index, e, np.round(got[0], 2), np.round(got[1], 2)))
    xs = [np.asarray(q) for q in kit.crossings()]
    print("kit placed within %.2f mm of the mount's own geometry "
          "(the struts land on %d crossings)" % (worst, len(xs)))
    if run.slow:
        by = {}
        for op, took in run.slow:
            by[(op.phase, op.kind)] = by.get((op.phase, op.kind), 0) + 1
        print("overran: %s" % ", ".join("%s/%s x%d" % (p, k, n)
                                        for (p, k), n in sorted(by.items())))
    if msgs:
        print("%d warnings" % len(msgs))

    if strip is not None:
        from PIL import Image
        for i, f in enumerate(strip.frames):
            Image.fromarray(f).save(os.path.join(a.sweep, "b%03d.png" % i))
        sheet = filmstrip.contact_sheet(
            strip.frames, os.path.join(a.sweep, "sweep_b.png"), scale=1,
            labels=["%.0fs" % v for v in strip.times])
        print("wrote %d frames and %s -- READ THE SHEET"
              % (len(strip.frames), sheet))
    return 0


if __name__ == "__main__":
    sys.exit(main())
