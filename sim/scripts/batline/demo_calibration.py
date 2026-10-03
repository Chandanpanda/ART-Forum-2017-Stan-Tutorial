"""LOOK AT IT: a bat's IMU calibrated at station 4 -- the gimbal running the
plan, the board's stream, and what the fit made of it -- filmed.

    python3 sim/scripts/batline/demo_calibration.py                  # the kid's bat
    python3 sim/scripts/batline/demo_calibration.py --bat adult      # the adult's, which the plan was never tuned on
    python3 sim/scripts/batline/demo_calibration.py --window T0 T1   # and 1 fps inside [T0, T1] s of station time
    python3 sim/scripts/batline/demo_calibration.py --out DIR

The plan is designed from the targets (imu/plan.py), the station drawn as
check_calibration draws it but with the rig's cameras placed (so the
numbers are its own, not the check's), the motion simulated with the
steppers' lag under every load (imu/sim.py), and one bat, drawn at the
spreads the plan was designed against, is run through it and fitted.

Each frame is the overview and two of the eight cameras, rendered at the
hinges as the steppers turned them, beside three traces over the whole
plan with a cursor at the frame's time: the hinges' angles; the gyro as
the board sent it (decoded at the datasheet's scale), with the plan's
quiet intervals shaded (stills green, spins blue); and how far each hinge
lags its command, against the plan's bound (red).

What the frames are for is what check_calibration cannot see: the bat
somewhere in the clamp the drawing does not put it, the gimbal reaching
the frame, a still that is not still in the gyro, a spin that is not one
steady rate, lag that rings after a stop into the next quiet interval.
Default output: sim/out/batline/calibration/<bat>/ (git ignores sim/out).
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from truss.glenv import headless   # noqa: E402  (before mujoco)
headless()

import numpy as np                  # noqa: E402

import check_cameras as CC          # noqa: E402
from truss import filmstrip         # noqa: E402
from batline import product         # noqa: E402
from batline.spec import Imu        # noqa: E402

VIEW = (320, 200)                   # each camera's frame on the sheet, px
OVER = (320, 400)                   # the overview's
TRACE = (480, 400)                  # the traces'
SHOWN = (0, 4)                      # the two cameras shown: opposite sides of the ring
BATS = {"kid": ("developed", 41, 1000), "adult": ("unseen", 42, 2000)}     # check_calibration's seeds


def traces(plan, mot, inp, t_s, gyro_dps):
    """The three traces as one picture (PIL), and a function from station
    time to its x pixel."""
    from PIL import Image, ImageDraw
    W, H = TRACE
    im = Image.new("RGB", TRACE, (255, 255, 255))
    dr = ImageDraw.Draw(im)
    T = float(mot.t[-1])
    x0, x1 = 44, W - 6
    X = lambda t: x0 + (x1 - x0) * np.asarray(t, float) / T                       # noqa: E731
    rows = [(8, 128, "hinges, deg"), (140, 260, "gyro |w|, dps"), (272, 392, "lag, deg")]
    for Q in inp.quiet:
        c = (205, 240, 205) if Q.kind == "still" else (205, 220, 250)
        for y0, y1, _ in rows[1:2]:
            dr.rectangle([float(X(Q.t0)), y0, float(X(Q.t1)), y1], fill=c)

    def line(ts, ys, y0, y1, lo, hi, col):
        Y = y1 - (y1 - y0) * (np.asarray(ys, float) - lo) / max(hi - lo, 1e-12)
        dr.line(list(zip(X(ts).tolist(), Y.tolist())), fill=col, width=1)

    qc = np.degrees(plan.path.state(mot.t)[0])
    lo, hi = float(qc.min()), float(qc.max())
    for h, col in ((0, (200, 60, 0)), (1, (0, 90, 200))):
        line(mot.t, qc[:, h], rows[0][0], rows[0][1], lo, hi, col)
    rows[0] = rows[0][:2] + ("hinges %.0f..%.0f deg (outer orange, inner blue)" % (lo, hi),)
    g = gyro_dps
    line(t_s, g, rows[1][0], rows[1][1], 0.0, float(g.max()), (0, 0, 0))
    rows[1] = rows[1][:2] + ("gyro |w| 0..%.0f dps (stills green, spins blue)" % float(g.max()),)
    lag = np.degrees(np.abs(mot.lag))
    bound = float(np.degrees(plan.lag()))
    top = 1.1 * max(bound, float(lag.max()))
    for h, col in ((0, (200, 60, 0)), (1, (0, 90, 200))):
        line(mot.t, lag[:, h], rows[2][0], rows[2][1], 0.0, top, col)
    yb = rows[2][1] - (rows[2][1] - rows[2][0]) * bound / top
    dr.line([(x0, yb), (x1, yb)], fill=(220, 0, 0))
    rows[2] = rows[2][:2] + ("lag 0..%.2f deg, the plan's bound %.2f (red)" % (top, bound),)
    for y0, y1, lab in rows:
        dr.rectangle([x0, y0, x1, y1], outline=(120, 120, 120))
        dr.text((x0 + 3, y0 + 1), lab, fill=(0, 0, 0))
    for t in np.arange(0.0, T, 10.0):
        dr.text((float(X(t)) - 6, H - 8), "%.0f" % t, fill=(80, 80, 80))
    return im, X


def what_at(inp, t):
    for Q in inp.quiet:
        if Q.t0 <= t <= Q.t1:
            if Q.kind == "still":
                return "still (item %d)" % Q.item
            return "spin, hinge %d at %.0f dps (item %d)" % (Q.hinge, np.degrees(Q.rate), Q.item)
    return "moving"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bat", default="kid", choices=sorted(BATS))
    ap.add_argument("--window", type=float, nargs=2, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from PIL import Image, ImageDraw
    from batline.rig import place as PL, vision as V
    from batline.imu import plan as P, sim as S, fit as F, judge as J
    from batline.imu.part import physical

    which, seed, bat_seed = BATS[a.bat]
    out = a.out or os.path.join(HERE, "..", "..", "out", "batline", "calibration", a.bat)
    os.makedirs(out, exist_ok=True)
    t_wall = time.time()
    d, _st, rig, site = CC._station(CC.BATS[which][0])
    pl, _cov = PL.place(rig, site)
    cams = pl.cams
    print("%s: cameras placed, %.0f s" % (a.bat, time.time() - t_wall))
    plan = P.for_design(d)
    print("plan: %.1f s, %d items, designed in %.0f s" % (plan.duration_s, len(plan.items), time.time() - t_wall))
    st = S.Station.draw(rig, np.random.default_rng(seed), cams)
    f_int = S.internal_rate(st, plan)[0]
    mot = S.motion(st, plan, f_int)
    print("motion simulated at %.0f Hz, lag at most %.3f of %.3f deg, %.0f s"
          % (f_int, np.degrees(np.abs(mot.lag).max()), np.degrees(plan.lag()), time.time() - t_wall))
    rng = np.random.default_rng(bat_seed)
    bat = S.draw_bat(d, rig, rng)
    inp, tr = S.run_station(st, mot, plan, bat, rng)
    res = F.fit(inp)
    v = F.verdict(res, inp)
    e = J.errors(res, tr)
    print("fit: %s, %d iterations; %s" % ("accepted" if v.ok else "rejected: " + "; ".join(v.why[:2]), res.iters,
                                         ", ".join("%s %.2f" % (g[0], float(np.sqrt(np.mean(e[F.GROUP_OF == k] ** 2)))
                                                                / g[2]) for k, g in enumerate(F.GROUPS))
                                         + " of target"))
    g, _, _ = physical(inp.stream)
    t_s = tr.t_first + inp.stream.k * (1.0 + tr.eps) / Imu.ODR
    base, X = traces(plan, mot, inp, t_s, np.degrees(np.linalg.norm(g, axis=1)))

    pv = V.PixelVision(rig, site, cams, st.build, "bat", np.random.default_rng(3))
    sx, sy = VIEW[0] / cams[0].lens.W, VIEW[1] / cams[0].lens.H

    def frame(t):
        i = int(np.argmin(np.abs(mot.t - t)))
        aa, bb = (float(x) for x in np.degrees(mot.q[i]))
        pv.pose(aa, bb)
        m = pv.m
        m.light_active[:] = 1
        m.light_diffuse[:] = 0.5
        pv.renderer.update_scene(pv.d, camera="overview", scene_option=pv.opt)
        ov = np.asarray(Image.fromarray(pv.renderer.render()).resize(OVER))
        tiles = []
        for k in SHOWN:
            img = pv.frame(aa, bb, k, "colour")
            im = Image.fromarray((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)).resize(VIEW)
            ImageDraw.Draw(im).text((4, 2), cams[k].name, fill=(255, 255, 0))
            tiles.append(np.asarray(im))
        tr_im = base.copy()
        dr = ImageDraw.Draw(tr_im)
        x = float(X(t))
        dr.line([(x, 4), (x, TRACE[1] - 10)], fill=(0, 0, 0), width=2)
        return np.hstack([ov, np.vstack(tiles), np.asarray(tr_im)]), (aa, bb)

    fps = [(0.1, None)] + ([(1.0, tuple(a.window))] if a.window else [])
    T = float(mot.t[-1])
    for f, w in fps:
        T0, T1 = (0.0, T) if w is None else (max(0.0, w[0]), min(T, w[1]))
        frames, labels = [], []
        for t in np.arange(T0, T1, 1.0 / f):
            im, (aa, bb) = frame(t)
            frames.append(im)
            labels.append("%.0f s  (%.0f, %.0f) deg  %s" % (t, aa, bb, what_at(inp, t)))
        name = "sweep.png" if w is None else "window_%.0f_%.0f.png" % w
        filmstrip.contact_sheet(frames, os.path.join(out, name), cols=2, labels=labels)
        print("%d frames: %s" % (len(frames), os.path.abspath(os.path.join(out, name))))
    pv.close()
    print("%.0f s of station time in %.0f s wall" % (T, time.time() - t_wall))


if __name__ == "__main__":
    main()
