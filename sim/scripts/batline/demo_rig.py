"""LOOK AT IT: the calibration station's rig calibrating itself and then
measuring a bat, filmed through its own eight cameras.

    python3 sim/scripts/batline/demo_rig.py                  # the kid's bat, check_cameras' build
    python3 sim/scripts/batline/demo_rig.py --bat unseen     # the adult's, which the rig never saw
    python3 sim/scripts/batline/demo_rig.py --window T0 T1   # and 1 fps inside [T0, T1] s of station time
    python3 sim/scripts/batline/demo_rig.py --out DIR

The same placement, build draw and sequence as check_cameras: the solver
places the cameras, the certified bar turns through the 8 x 8 grid of
gimbal poses and the rig calibrates itself from it, then the bat turns
through the 4 x 4 grid a quarter step over (no pose calibration saw).  The gimbal turns both axes at
once at Est.GIMBAL_RATE and holds each pose Est.STILL_S, so the frames are
at 0.1 fps of that time (truss/filmstrip.py).  Each frame is the overview
and the eight cameras' colour exposures, rendered and warped through each
camera's true lens.  On the bar, red crosses are where the drawing (the
encoders' pose, the cameras as drawn) puts each ball -- what
self-calibration starts from; on the bat, green crosses are where the
calibrated rig puts each marker at the pose it tracked.

What the frames are for is what check_cameras cannot see: a camera whose
view is mostly the gantry, a ball every camera sees only edge-on, the bar
or the bat somewhere the scene did not mean, crosses that land beside the
balls.  Default output: sim/out/batline/rig/<bat>/ (git ignores sim/out).
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
from batline.spec import Est        # noqa: E402

VIEW = (320, 200)                   # each camera's frame on the sheet, px
OVER = (320, 400)                   # the overview's


def timeline(poses, t0=0.0, start=(0.0, 0.0)):
    """[(t0, t1, from, to, what)]: the gimbal turning to each pose, both
    axes at once the shorter way round, then held still."""
    segs, t, prev = [], t0, np.asarray(start, float)
    for p in np.asarray(poses, float):
        dlt = (p - prev + 180.0) % 360.0 - 180.0
        mv = float(np.abs(dlt).max()) / Est.GIMBAL_RATE
        segs.append((t, t + mv, prev, prev + dlt, "turning"))
        t += mv
        segs.append((t, t + Est.STILL_S, p, p, "held"))
        t += Est.STILL_S
        prev = p
    return segs


def at(segs, t):
    for t0, t1, p0, p1, what in segs:
        if t0 <= t <= t1:
            f = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            a, b = ((1.0 - f) * p0 + f * p1 + 180.0) % 360.0 - 180.0
            return float(a), float(b), what
    return None


def times(segs, fps, window=None):
    T0, T1 = segs[0][0], segs[-1][1]
    if window:
        T0, T1 = max(T0, window[0]), min(T1, window[1])
    return np.arange(T0, T1, 1.0 / fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bat", default="developed", choices=sorted(CC.BATS))
    ap.add_argument("--window", type=float, nargs=2, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from PIL import Image, ImageDraw
    from batline.rig import place as PL, spec as RS, mjcf as RM, vision as V, track as T

    bat, seed = CC.BATS[a.bat]
    out = a.out or os.path.join(HERE, "..", "..", "out", "batline", "rig", a.bat)
    os.makedirs(out, exist_ok=True)
    t_wall = time.time()
    d, st, rig, site = CC._station(bat)
    pl, _cov = PL.place(rig, site)
    cams = pl.cams
    b = RM.RigBuildDraw.draw(rig, cams, np.random.default_rng(seed))
    print("%s: cameras placed, %.0f s" % (bat.name, time.time() - t_wall))

    bar_poses, bat_poses = RS.grid(8), RS.grid(4, 0.25)
    s_bar = timeline(bar_poses)
    s_bat = timeline(bat_poses, s_bar[-1][1], bar_poses[-1])
    fps = [(0.1, None)] + ([(1.0, tuple(a.window))] if a.window else [])
    K = len(rig.markers)
    sx, sy = VIEW[0] / cams[0].lens.W, VIEW[1] / cams[0].lens.H

    def views(pv, aa, bb, crosses, colour):
        """The overview beside the eight views, crosses drawn on each."""
        import mujoco
        pv.pose(aa, bb)
        m = pv.m
        m.light_active[:] = 1
        m.light_diffuse[:] = 0.5
        pv.renderer.update_scene(pv.d, camera="overview", scene_option=pv.opt)
        ov = Image.fromarray(pv.renderer.render()).resize(OVER)
        tiles = []
        for k in range(len(cams)):
            img = pv.frame(aa, bb, k, "colour")
            im = Image.fromarray((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)).resize(VIEW)
            dr = ImageDraw.Draw(im)
            for u, v in crosses[k]:
                x, y = u * sx, v * sy
                dr.line([(x - 4, y), (x + 4, y)], fill=colour)
                dr.line([(x, y - 4), (x, y + 4)], fill=colour)
            dr.text((4, 2), cams[k].name, fill=(255, 255, 0))
            tiles.append(np.asarray(im))
        grid = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
        return np.hstack([np.asarray(ov), grid])

    sheets = {f: ([], []) for f, _w in fps}

    # the bar: the drawing's belief, as self-calibration starts
    pv = V.PixelVision(rig, site, cams, b, "bar", np.random.default_rng(1))
    pred = T.Predictor(rig, site, cams, "bar")
    P0 = np.vstack([rig.markers, b.bar_cert])
    for f, w in fps:
        for t in times(s_bar, f, w):
            aa, bb, what = at(s_bar, t)
            T0 = T.encoder_pose(aa, bb)
            pr = pred(cams, aa, bb, T0[:, :3], T0[:, 3], P0, rig.ball_r)
            cr = [[(uv[j] + pull[j]) for j in np.flatnonzero(front)] for uv, front, ok, pull in pr]
            sheets[f][0].append(views(pv, aa, bb, cr, (255, 40, 40)))
            sheets[f][1].append("%.0f s  bar %s at (%.0f, %.0f)" % (t, what, aa, bb))
    pv.close()
    del pv
    print("bar filmed, %.0f s; calibrating" % (time.time() - t_wall))
    cal = T.self_calibrate(V.Contract(V.ModelVision(rig, site, cams, b, "bar", np.random.default_rng(1))), rig,
                           site, bar_poses, b.bar_cert)
    print("calibrated: rms %.3f px, stems' scale %.3f, %.0f s" % (cal.sol.rms_px, cal.stem, time.time() - t_wall))

    # the bat: the calibrated rig's prediction at the pose it tracked
    pv = V.PixelVision(rig, site, cams, b, "bat", np.random.default_rng(3))
    pred = T.Predictor(rig, site, cams, "bat")
    P = cal.points.P[:K]
    for f, w in fps:
        for t in times(s_bat, f, w):
            aa, bb, what = at(s_bat, t)
            tr = T.track(V.Contract(pv), cal, rig, pred, aa, bb)
            cr = [[] for _ in cams]
            if tr.fit is not None:
                pr = pred(cal.sol.cams, aa, bb, tr.fit.R, tr.fit.t, P, rig.ball_r)
                cr = [[(uv[j] + cal.stem * pull[j]) for j in np.flatnonzero(front)] for uv, front, ok, pull in pr]
            sheets[f][0].append(views(pv, aa, bb, cr, (40, 255, 40)))
            sheets[f][1].append("%.0f s  bat %s at (%.0f, %.0f), %d sightings" % (t, what, aa, bb, len(tr.obs)))
    pv.close()

    for f, w in fps:
        frames, labels = sheets[f]
        name = "sweep.png" if w is None else "window_%.0f_%.0f.png" % w
        filmstrip.contact_sheet(frames, os.path.join(out, name), cols=2, labels=labels)
        print("%d frames: %s" % (len(frames), os.path.abspath(os.path.join(out, name))))
    print("%.0f s of station time in %.0f s wall" % (s_bat[-1][1], time.time() - t_wall))


if __name__ == "__main__":
    main()
