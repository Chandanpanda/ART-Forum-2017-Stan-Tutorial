"""LOOK AT IT: a station commissioning itself and running every op, filmed.

    python3 sim/scripts/batline/demo_station.py                     # the station the module was developed on
    python3 sim/scripts/batline/demo_station.py --station unseen    # the travel it has never seen
    python3 sim/scripts/batline/demo_station.py --window T0 T1      # and 1 fps inside [T0, T1] s of station time
    python3 sim/scripts/batline/demo_station.py --out DIR

The same station, the same build draw and the same sequence as
check_gantry's commissioning scenario: home, measure the camera, find the
regions, measure every tool against the pin, then every op on the test
pieces.  Filmed at 0.1 fps of SIMULATED time (truss/filmstrip.py) from two
cameras -- the overview from the aisle, and the tip camera riding the Z
carriage, which sees the tool row side on: every tip and the part under it.
Each frame is the two side by side, labelled with its station time and the
op that was running, and all of them are tiled on one contact sheet.
--window adds a 1 fps sheet of a stretch that looked wrong.

What the frames are for is what check_gantry cannot see: a tool sent
exactly where the frame said and the frame wrong, a token set back in its
nest on its side, a phase that ran and did nothing.  Default output:
sim/out/batline/station/<station>/ (git ignores sim/out).
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

import check_gantry as G            # noqa: E402
from truss import filmstrip         # noqa: E402

VIEW = (480, 300)                   # each camera's frame, px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", default="developed", choices=sorted(G.STATIONS))
    ap.add_argument("--window", type=float, nargs=2, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import mujoco
    from batline.station import commission as C, sim as SIM, mjcf as SM

    bat, name, seed = G.STATIONS[a.station]
    out = a.out or os.path.join(HERE, "..", "..", "out", "batline", "station", a.station)
    os.makedirs(out, exist_ok=True)
    mods, _ = G.bat_modules(bat)
    mod, label, seed = G.module(mods, a.station)

    # check_gantry's build draw, with an offscreen buffer the review views fit
    b = SM.BuildDraw.draw(mod, np.random.default_rng(seed))
    pcs = SM.test_plate(mod)
    W = max(VIEW[0], G.HeadCam.W)
    H = max(VIEW[1], G.HeadCam.H)
    m, d, sim, clock = SIM.build(mod, b, pcs, rng=np.random.default_rng(seed + 100), render=(W, H))
    r = mujoco.Renderer(m, VIEW[1], VIEW[0])
    strips = [filmstrip.Filmstrip(r, cam, fps=0.1) for cam in ("overview", "tip")]
    if a.window:
        strips += [filmstrip.Filmstrip(r, cam, fps=1.0, window=tuple(a.window)) for cam in ("overview", "tip")]

    def hook(data):
        for s in strips:
            s.maybe(data.time, data)

    clock.hook = hook
    ex = G._executor(mod, sim, clock, m, d, seed)
    t0 = time.time()
    fr, notes = clock.run(C.commission(ex, mod))
    print("%s: commissioning %s, %.0f s of station time" % (label, "done" if fr else "FAILED: %s"
                                                            % notes.get("failed"), d.time))
    t_comm = d.time
    res = G.ops(mod, label, b, pcs, m, d, sim, clock, ex, fr) if fr is not None else []
    for nm, ok, det in res:
        print("  %s  %s  [%s]" % ("ok  " if ok else "FAIL", nm, det))
    print("%.0f s of station time in %.0f s wall" % (d.time, time.time() - t0))

    def doing(t):
        """The innermost op running at station time t."""
        on = [dn for dn in ex.done if dn.t0 <= t <= dn.t1]
        if not on:
            return "commissioning" if t < t_comm else "ops"
        return min(on, key=lambda dn: dn.t1 - dn.t0).op

    def sheet(pair, path):
        ov, sd = pair
        frames = [np.hstack([f1, f2]) for f1, f2 in zip(ov.frames, sd.frames)]
        labels = ["%.0f s %s" % (t, doing(t)) for t in ov.times]
        filmstrip.contact_sheet(frames, path, cols=4, labels=labels)
        print("%d frames: %s" % (len(frames), os.path.abspath(path)))

    sheet(strips[:2], os.path.join(out, "sweep.png"))
    if a.window:
        sheet(strips[2:], os.path.join(out, "window_%.0f_%.0f.png" % tuple(a.window)))


if __name__ == "__main__":
    main()
