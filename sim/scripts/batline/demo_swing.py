"""LOOK AT IT: the swing rig at station 4 swinging a bat through the rated
swing on its dummy pack, filmed.

    python3 sim/scripts/batline/demo_swing.py                   # the kid's bat, the dummy pack
    python3 sim/scripts/batline/demo_swing.py --bat adult       # the adult's
    python3 sim/scripts/batline/demo_swing.py --weak            # a cap spring at half the swing's pull
    python3 sim/scripts/batline/demo_swing.py --window T0 T1    # and 200 fps inside [T0, T1] s of the run
    python3 sim/scripts/batline/demo_swing.py --out DIR

The run is check_swing's: the rig fitted to the rated swing, its motors
chosen, the motion from park to the stance, the swing and back, MuJoCo
stepping the rig, its servo loop and the sprung pack (swing/sim.py).  The
run lasts under two seconds, so the sweep is ten frames a simulated second
(the repo's 0.1 fps is for runs of minutes); --window looks closer.

Each frame is the rig seen square on to its plane, the arm on its hub and
the bat on the wrist at the angles MuJoCo reached, with the rated swing's
own hand path as grey dots (the arm's end should ride them), the cells
drawn where their slides put them (moved 50x along the bay so a
millimetre shows); beside it the run's traces with a cursor: the two
joints' angles, planned and true; the chip's rate against the
quantile part's rail (red); and the + plate's force with the pack's load
(the circuit opens where the force reaches zero).

What the frames are for is what check_swing cannot see: an arm that
does not follow the hands, a bat that turns the wrong way or stands on
its pommel, a stance or follow-through nobody meant, cells that move
when nothing says they should.  Default output: sim/out/batline/swing/<bat>/
(git ignores sim/out).
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
import mujoco                       # noqa: E402

from truss import filmstrip         # noqa: E402
from batline import product         # noqa: E402
from batline.spec import KID, Bat, Quality, CapSpring  # noqa: E402
from batline.rig.pose import normal_quantile     # noqa: E402
from batline.imu.part import usable              # noqa: E402
from batline.swing import rig as SR, sim as SS   # noqa: E402

VIEW = (520, 400)                   # the rig's frame on the sheet, px
TRACE = (480, 400)
SHOW_CELLS = 50.0                   # the cells' slides drawn this much larger
BATS = {"kid": KID,                 # the adult's as check_bat draws it
        "adult": Bat("adult", length=860.0, handle=300.0, grip_d=34.0, width=108.0, depth=62.0)}


def scene(sim, rig):
    """The run's model with the rated swing's hand path as dots and a camera
    square on to the rig's plane, framing the whole sweep."""
    sw = rig.swing
    th = sw.theta(np.array([sw.t_top, sw.t_end]))
    P = SR.hands(sw, SR.arc_times(sw, np.linspace(th.min(), th.max(), 40)))
    dots = "".join('<geom type="sphere" pos="%.5f 0.02 %.5f" size="0.006" rgba="0.45 0.45 0.45 1" '
                   'contype="0" conaffinity="0"/>' % (x, z) for x, z in P)
    reach = rig.L + (rig.d.lay.x_tip1 - rig.d.lay.x_hands) / 1000.0
    fovy = 50.0
    dist = 1.15 * reach / np.tan(np.radians(fovy / 2.0))
    cam = ('<camera name="side" pos="%.5f %.5f %.5f" xyaxes="1 0 0 0 0 1" fovy="%.1f"/>'
           % (rig.hub[0], -dist, rig.hub[1], fovy))
    xml = sim.xml.replace("</worldbody>", dots + cam + "\n  </worldbody>")
    m = mujoco.MjModel.from_xml_string(xml)
    return m, mujoco.MjData(m)


def traces(rig, mo, r, rail):
    from PIL import Image, ImageDraw
    W, H = TRACE
    im = Image.new("RGB", TRACE, (255, 255, 255))
    dr = ImageDraw.Draw(im)
    T = float(r.t[-1])
    x0, x1 = 44, W - 6
    X = lambda t: x0 + (x1 - x0) * np.asarray(t, float) / T                       # noqa: E731
    rows = [[8, 128, ""], [140, 260, ""], [272, 392, ""]]

    def line(ts, ys, y0, y1, lo, hi, col):
        Y = y1 - (y1 - y0) * (np.asarray(ys, float) - lo) / max(hi - lo, 1e-12)
        dr.line(list(zip(X(ts).tolist(), Y.tolist())), fill=col, width=1)

    step = max(1, len(r.t) // 4000)
    t = r.t[::step]
    qp = np.degrees(mo.state(t)[0])
    qt = np.degrees(r.q[::step])
    lo, hi = float(min(qp.min(), qt.min())), float(max(qp.max(), qt.max()))
    for h, col in ((0, (200, 60, 0)), (1, (0, 90, 200))):
        line(t, qp[:, h], rows[0][0], rows[0][1], lo, hi, (200, 200, 200))
        line(t, qt[:, h], rows[0][0], rows[0][1], lo, hi, col)
    rows[0][2] = "arm (orange), bat (blue) %.0f..%.0f deg; plan grey" % (lo, hi)
    g = np.degrees(np.abs(r.w[::step]).max(1))
    top = 1.1 * np.degrees(rail)
    line(t, g, rows[1][0], rows[1][1], 0.0, top, (0, 0, 0))
    yr = rows[1][1] - (rows[1][1] - rows[1][0]) * np.degrees(rail) / top
    dr.line([(x0, yr), (x1, yr)], fill=(220, 0, 0))
    rows[1][2] = "chip |w| 0..%.0f dps; the quantile part's rail %.0f (red)" % (top, np.degrees(rail))
    fp = r.force[::step]
    m_pack = SR.DUMMY.n * SR.DUMMY.m * 1e-3
    load = -m_pack * SR.pack_load(rig, r.q[::step], r.qd[::step], r.qdd[::step]) + SR.spring_force(rig.d, SR.DUMMY)
    hi2 = 1.1 * float(max(fp.max(), load.max()))
    line(t, load, rows[2][0], rows[2][1], 0.0, hi2, (0, 160, 0))
    line(t, fp, rows[2][0], rows[2][1], 0.0, hi2, (0, 0, 0))
    rows[2][2] = "+ plate force 0..%.1f N (black); spring less the pack's load (green)" % hi2
    for y0, y1, lab in rows:
        dr.rectangle([x0, y0, x1, y1], outline=(120, 120, 120))
        dr.text((x0 + 3, y0 + 1), lab, fill=(0, 0, 0))
    for tt in np.arange(0.0, T, 0.25):
        dr.text((float(X(tt)) - 8, H - 8), "%.2f" % tt, fill=(80, 80, 80))
    for tt, lab in ((mo.t_swing, "swing"), (mo.t_back, "back")):
        dr.line([(float(X(tt)), 4), (float(X(tt)), H - 10)], fill=(180, 180, 255))
        dr.text((float(X(tt)) + 2, H - 20), lab, fill=(80, 80, 200))
    return im, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bat", default="kid", choices=sorted(BATS))
    ap.add_argument("--weak", action="store_true")
    ap.add_argument("--window", type=float, nargs=2, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from PIL import Image, ImageDraw
    d = product.design(BATS[a.bat])
    name = a.bat + ("_weak" if a.weak else "")
    out = a.out or os.path.join(HERE, "..", "..", "out", "batline", "swing", name)
    os.makedirs(out, exist_ok=True)
    t_wall = time.time()
    rig, mo = SR.fit(d), SR.motion(d)
    kw = {}
    if a.weak:
        _, sl = SR.seated(d, SR.DUMMY)
        kw["rate"] = 0.5 * SR.DUMMY.n * SR.DUMMY.m * 1e-3 * SR.swing_load(rig) / (CapSpring.L_FREE - sl)
    sim = SS.Sim(d, **kw)
    r = sim.run(mo)
    print("%s: run %.2f s simulated at %.0f Hz in %.0f s; %d opens, longest %.1f ms"
          % (name, mo.T, 1.0 / sim.dt, time.time() - t_wall, len(r.opens),
             1e3 * max([b - a_ for a_, b in r.opens], default=0.0)))
    rail = usable(normal_quantile(1.0 - Quality.CAL_QUAL_ALPHA / 12.0))[0]
    base, X = traces(rig, mo, r, rail)
    m, dd = scene(sim, rig)
    ren = mujoco.Renderer(m, VIEW[1], VIEW[0])
    jp, jq, jr = (m.joint(n).qposadr[0] for n in ("phi", "q2", "rot"))
    jc = [m.joint("cell%d" % i).qposadr[0] for i in range(SR.DUMMY.n)]

    def frame(t):
        i = min(int(round(t / sim.dt)), len(r.t) - 1)
        q = r.q[i]
        dd.qpos[:] = 0.0
        dd.qpos[jp], dd.qpos[jq], dd.qpos[jr] = q[0], q[1] - q[0], q[1]
        for k, j in enumerate(jc):
            dd.qpos[j] = SHOW_CELLS * r.cells[i, k]
        mujoco.mj_forward(m, dd)
        ren.update_scene(dd, camera="side")
        img = Image.fromarray(ren.render())
        ImageDraw.Draw(img).text((6, 4), "arm %.0f, bat %.0f deg; plate %.2f N%s"
                                 % (np.degrees(q[0]), np.degrees(q[1]), r.force[i],
                                    "" if r.closed[i] else "  OPEN"), fill=(0, 0, 0))
        tr = base.copy()
        ImageDraw.Draw(tr).line([(float(X(t)), 4), (float(X(t)), TRACE[1] - 10)], fill=(0, 0, 0), width=2)
        return np.hstack([np.asarray(img), np.asarray(tr)])

    def phase(t):
        return "to the stance" if t < mo.t_swing else ("swing" if t < mo.t_back else "back to park")

    sheets = [(10.0, (0.0, float(r.t[-1])), "sweep.png")]
    if a.window:
        sheets.append((200.0, tuple(a.window), "window_%.3f_%.3f.png" % tuple(a.window)))
    for f, (T0, T1), fname in sheets:
        frames, labels = [], []
        for t in np.arange(T0, min(T1, float(r.t[-1])) + 1e-9, 1.0 / f):
            frames.append(frame(t))
            labels.append("%.3f s  %s" % (t, phase(t)))
        filmstrip.contact_sheet(frames, os.path.join(out, fname), cols=3, labels=labels)
        print("%d frames: %s" % (len(frames), os.path.abspath(os.path.join(out, fname))))
    ren.close()
    print("%.0f s wall" % (time.time() - t_wall))


if __name__ == "__main__":
    main()
