"""Tier 1, slow: the pre-wind look done on RENDERED frames.

ModelVision hands the process the truth plus a bracket bias; this suite
puts the pixel path in its place -- the head camera rendered offscreen,
PixelVision fitting the chord and its diagonals -- and asks the same
things of it: at the plan's look pose over every joint, does it find the
joint where the model camera says it is; does a deliberate head offset
come back as that offset; does the bracket's bias pass straight through;
how much does a frame's fit jitter.

    python3 sim/scripts/truss/check_vision.py [-v]
"""
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import mujoco

from truss import structure, fixture, mjcf, cell, approach, vision, schedule, band as _band
from truss.spec import Vision
from truss.geometry import TrussGeometry

VERBOSE = "-v" in sys.argv
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def main():
    t = structure.TRUSS_300
    g = TrussGeometry(t)
    fx = fixture.Fixture(g)
    m = mujoco.MjModel.from_xml_string(mjcf.scene_cell(g, fx, stage="loaded"))
    d = mujoco.MjData(m)
    c = cell.CellSim(m, d, g, fx, rng=np.random.default_rng(3), thermal=False)
    clk = cell.SimClock(m, d, c)
    for _ in range(25):
        clk.tick()
    cam = cell.SimCamera(m, d)
    pv = vision.PixelVision(cam, c, rng=np.random.default_rng(5))
    mv = vision.ModelVision(c, rng=np.random.default_rng(6))
    mv.bias[:] = 0.0
    mv.sigma = 0.0
    bias = pv.bias.copy()
    pv.bias[:] = 0.0
    st = approach.plan_truss(g, fx, span=0.0)
    t0 = time.time()
    cam.frame()
    t_first = time.time() - t0
    t0 = time.time()
    cam.frame()
    t_frame = time.time() - t0
    check("a frame renders offscreen in well under a look's settle time",
          t_frame < 1.0, "%.2f s (first %.2f s)" % (t_frame, t_first))
    # ------------------------------------------- every joint, look pose
    errs, misses, jitter = [], [], []
    direction = +1
    for k in range(t.n_chords):
        th, aps = st[k]
        c.index(th)
        for _ in range(300):
            clk.tick()
        c.park(0.0)
        for _ in range(60):
            clk.tick()
        for j, a in zip(g.joints_on(k), aps):
            c.goto("x", schedule.look_x(t, j)); c.goto("y", a.centre[1]); c.goto("z", a.centre[2] + a.lift)
            for _ in range(120):
                clk.tick()
            truth = mv.locate(j.index)
            got = pv.locate(j.index)
            if got is None or truth is None:
                misses.append(j.index)
                continue
            errs.append((j.index, got[0] - truth[0], got[1] - truth[1]))
            if j.index == g.joints_on(k)[1].index:
                reps = [pv.locate(j.index) for _ in range(3)]
                jitter.append(np.std([r[0] for r in reps if r]) if all(reps) else 9.9)
        direction = -direction
    ex = np.array([e[1] for e in errs]) if errs else np.array([9.9])
    ey = np.array([e[2] for e in errs]) if errs else np.array([9.9])
    check("the pixel path finds every joint at its look pose", not misses,
          "missed %s" % misses)
    check("...within LOOK_SIGMA rms of the model camera along the chord, the constant the model "
          "camera draws its noise from (MEASURED here)",
          np.sqrt(np.mean(ex ** 2)) <= Vision.LOOK_SIGMA and np.abs(ex).max() < 3.0 * Vision.LOOK_SIGMA,
          "rms %.3f, worst %.2f mm vs %.2f" % (np.sqrt(np.mean(ex ** 2)), np.abs(ex).max(), Vision.LOOK_SIGMA))
    check("...and to a tenth across it, where the chord's own centreline is the feature",
          np.abs(ey).max() < 0.15, "worst %.3f mm" % np.abs(ey).max())
    # the far side of the plate, for the record: the reason look_x exists
    th, aps = st[1]
    c.index(th)
    for _ in range(300):
        clk.tick()
    c.park(0.0)
    for _ in range(60):
        clk.tick()
    far = []
    for j, a in zip(g.joints_on(1), aps):
        c.goto("x", 2.0 * j.x - schedule.look_x(t, j)); c.goto("y", a.centre[1]); c.goto("z", a.centre[2] + a.lift)
        for _ in range(120):
            clk.tick()
        truth, got = mv.locate(j.index), pv.locate(j.index)
        far.append(abs(got[0] - truth[0]) if got and truth else 9.9)
    check("seen through the plate's gap from the far side the joint's x is worse -- why look_x picks a side",
          max(far) > np.abs(ex).max(), "far side worst %.2f vs %.2f" % (max(far), np.abs(ex).max()))
    check("the frame-to-frame jitter of a look is hundredths of a millimetre",
          max(jitter) < 0.05, "%s" % [round(v, 3) for v in jitter])
    # --------------------------------------------- a deliberate offset
    th, aps = st[0]
    c.index(th)
    for _ in range(300):
        clk.tick()
    c.park(0.0)
    for _ in range(60):
        clk.tick()
    j, a = g.joints_on(0)[1], aps[1]
    xb = schedule.look_x(t, j)
    rows = []
    for ox, oy in ((0.0, 0.0), (1.0, -0.6), (-1.5, 0.8), (0.5, 1.5)):
        c.goto("x", xb + ox); c.goto("y", a.centre[1] + oy); c.goto("z", a.centre[2] + a.lift)
        for _ in range(120):
            clk.tick()
        truth, got = mv.locate(j.index), pv.locate(j.index)
        rows.append((ox, oy, None if got is None else (got[0] - truth[0], got[1] - truth[1])))
    check("a head put down off the joint reads back the offset it was given",
          all(r[2] is not None and abs(r[2][0]) < 3.0 * Vision.LOOK_SIGMA and abs(r[2][1]) < 0.15
              for r in rows),
          " ".join("(%+.1f,%+.1f)->%s" % (r[0], r[1], None if r[2] is None else
                                            "%+.2f/%+.2f" % r[2]) for r in rows))
    # ------------------------------------------------------- the bias
    pv.bias[:] = bias
    c.goto("x", xb); c.goto("y", a.centre[1])
    for _ in range(120):
        clk.tick()
    truth, got = mv.locate(j.index), pv.locate(j.index)
    check("the bracket's bias passes straight through the look, as the model camera assumes",
          got is not None and abs((got[0] - truth[0]) - bias[0]) < 3.0 * Vision.LOOK_SIGMA
          and abs((got[1] - truth[1]) - bias[1]) < 0.15,
          "read %+.2f/%+.2f for a bias of %+.2f/%+.2f" % (got[0] - truth[0], got[1] - truth[1], bias[0], bias[1])
          if got else "no fix")
    check("the camera resolves a chord edge finer than the bracket's bias, so the bracket is the budget",
          Vision.mm_per_px(Vision.range_nominal()) * Vision.EDGE_SIGMA_PX < Vision.EXT_SIGMA / 5.0,
          "%.4f mm/edge vs %.2f bias" % (Vision.mm_per_px(Vision.range_nominal()) * Vision.EDGE_SIGMA_PX,
                                          Vision.EXT_SIGMA))

    bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det and (VERBOSE or not ok) else ""))
    print("check_vision: %d checks, %d failed" % (len(RESULTS), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
