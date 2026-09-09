"""Build the truss the design sweep chose, in the same cell.

    python3 sim/scripts/truss/demo_optimal.py                 # headless
    python3 sim/scripts/truss/demo_optimal.py --gui           # watch it
    python3 sim/scripts/truss/demo_optimal.py --gui --300     # the short piece
    python3 sim/scripts/truss/demo_optimal.py --gui --speed 4

demo_cell.py builds the geometry the cell was first designed around.  This
builds the one sim/spine picked by solving what the CAMERA needs -- a
115 mm section on 40 degree diagonals with a 1.5 mm web -- and prints,
after the truss is made, what that design is worth to the stereo pair.
Same cell, same head, same process: only the truss spec differs, which is
the point of having made every pose a function of it.

The two demos are independent.  This one does not modify demo_cell, and
demo_cell does not know this exists.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from truss import glenv                  # noqa: E402  (before mujoco)
if "--gui" not in sys.argv:
    glenv.headless()
import numpy as np
import mujoco

from truss import (fixture, mjcf, cell, approach, schedule, process, vision,
                   inspector, view, structure as tstruct, band as _band)
from truss.spec import Truss, ring_swept_r
from truss.geometry import TrussGeometry

from spine import chosen, build as sbuild, duty as sduty, metrics as smetrics


def spine_report(ch, duty):
    """What this truss is worth to the camera, from sim/spine."""
    m = sbuild.warren_truss(ch.length, ch.side, ch.alpha, ch.d_chord, ch.d_diag,
                            tip_mass=duty.tip_mass_g, web=ch.web)
    return smetrics.evaluate(m, duty)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--300", dest="short", action="store_true", help="the 300 mm piece")
    ap.add_argument("--speed", type=float, default=1.0, help="wall-clock pacing; 0 = unpaced")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    ch = chosen.OPTIMAL_300 if a.short else chosen.OPTIMAL_1M
    duty = sduty.QUADROTOR
    t = Truss(**ch.as_truss_kwargs())
    g = TrussGeometry(t)
    fx = fixture.Fixture(g)

    sp = spine_report(ch, duty)
    was = tstruct.TRUSS_300 if a.short else tstruct.TRUSS_1M
    sp_was = spine_report(chosen.Chosen(was.name, was.length, was.side, was.alpha,
                                        was.d_chord, was.d_diag), duty)
    print("the design the sweep chose: %.0f mm of %.0f mm section, %.0f degree web, "
          "%.1f/%.1f mm stock" % (t.length, t.side, t.alpha, t.d_chord, t.d_diag))
    print("  %-22s %10s %10s" % ("", "chosen", "as first built"))
    for label, key, fmt in (("mass, g", "mass_g", "%10.1f"),
                            ("range error at 100 m", "dz_static_m", "%10.2f"),
                            ("yaw budget used", "budget_used", "%10.2f"),
                            ("first mode, Hz", "f1_hz", "%10.0f")):
        print(("  %-22s " + fmt + fmt) % (label, sp[key], sp_was[key]))
    print("  the cell says: %s" % (", ".join(tstruct.violations(t)) or "no objection"))

    st = approach.plan_truss(g, fx, span=0.0)
    if any(v[1] is None for v in st.values()):
        print("the cell cannot reach every joint of this design.")
        return 1
    P = schedule.plan(g, fx, st)
    s = schedule.summary(P)
    print("plan: %d rods, %d joints, %.1f min  %s"
          % (len(g.rods), t.n_joints, s["total_min"],
             {k: round(v, 1) for k, v in s["phase_min"].items()}))

    xml = mjcf.scene_cell(g, fx, stage="empty")
    out = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "truss_cell_%s.xml" % t.name), "w") as f:
        f.write(xml)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    c = cell.CellSim(m, d, g, fx, rng=np.random.default_rng(a.seed))
    clk = cell.SimClock(m, d, c)
    for _ in range(25):
        clk.tick()
    vis = vision.ModelVision(c, rng=np.random.default_rng(a.seed + 1))
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
            m.geom_pos[gid] = g.chord_point(g.joints[jix].chord, stt.centre) / 1000.0
            m.geom_rgba[gid][3] = min(1.0, 0.2 + stt.turns / t.turns)

    last_op = [0]
    t_wall = [time.time()]

    def step():
        try:
            next(gen)
        except StopIteration:
            return False
        clk.tick()
        if len(ex.timeline) > last_op[0]:
            last_op[0] = len(ex.timeline)
            op = ex.timeline[-1][0]
            if op.kind in ("wind", "release", "index", "dose", "cut"):
                print("%7.1f  %s" % (d.time, op))
            show_bands()
        return True

    def report():
        errs = {r.index: inspector.seat_error(g, r, *c.rod_pose(r.index), theta=c.cage_truth())
                for r in g.rods}
        rep = inspector.inspect_truss(g, ex.states, seated=errs, cycle_s=d.time)
        print(rep)
        print("simulated %.1f min against %.1f planned" % (d.time / 60.0, P.total / 60.0))
        print("and what it is worth: %.2f m of range error at %.0f m, %.0f%% of the yaw budget, "
              "%.1f g" % (sp["dz_static_m"], duty.range_m, 100 * sp["budget_used"], sp["mass_g"]))

    if a.gui and not glenv.windowed():
        print("--gui needs a display; this session has none.")
        a.gui = False
    if not a.gui:
        while step():
            pass
        report()
        return 0

    from mujoco import viewer as mjviewer
    rig = {"cam": None}

    def on_key(code):
        if rig["cam"] is not None:
            rig["cam"].key(code)

    with mjviewer.launch_passive(m, d, key_callback=on_key) as v:
        rig["cam"] = view.CameraRig(v, m, view.cell_frame(g, fx),
                                    follow=lambda: c.ring_centre() / 1000.0,
                                    close=view.close_frame(ring_swept_r()))
        print(view.HELP)
        t_wall[0] = time.time()
        while v.is_running():
            if not step():
                break
            rig["cam"].tick()
            v.sync()
            if a.speed > 0:
                lag = d.time / a.speed - (time.time() - t_wall[0])
                if lag > 0:
                    time.sleep(min(lag, 0.05))
        if not v.is_running():
            return 0
        report()
        print("\n  the truss is finished; the window is yours.  Close it to exit.")
        while v.is_running():
            clk.tick()
            rig["cam"].tick()
            v.sync()
            time.sleep(0.02)
    return 0


if __name__ == "__main__":
    sys.exit(main())
