"""Agent A, beam phase only: seal the quarantine corner with the two beams.

The 70-point task (spec 1, task 3).  Run it on its own so the beam geometry can
be judged without waiting for the sample mission:

    python scripts/demo_beams.py [--gui] [--xray] [--speed 1.0] [--only 1|2]
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mujoco
from rfgyc26 import mjcf, referee, view
from rfgyc26.params import Field, AgentA
from rfgyc26.robot import AgentARobot
from rfgyc26.route import place_beam, seal_quarantine, guard, turn_to, pursue

CTRL_DECIM = 20


def beam_state(m, d):
    out = []
    for i in (1, 2):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "beam%d" % i)
        p = d.xpos[b] * 1000.0
        out.append((p[0], p[1], p[2], d.xquat[b].copy()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--xray", action="store_true")
    ap.add_argument("--speed", type=float, default=0.0)
    ap.add_argument("--only", type=int, default=0, help="place just this beam")
    ap.add_argument("--start", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "H"), help="robot start pose")
    ap.add_argument("--timeout", type=float, default=120.0)
    a = ap.parse_args()

    # Start where the sample mission leaves off: west end of the quarantine,
    # facing west, both beams still in their pockets.
    pose = tuple(a.start) if a.start else (520.0, 230.0, 180.0)
    xml = mjcf.scene_pick_place([], robot_pose=pose, with_beams=True)
    path = os.path.join(os.path.dirname(__file__), "..", "models", "scene_beams.xml")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(xml)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    rb = AgentARobot(m, d)
    rb.fingers(True); rb.gate(False); rb.intake(False); rb.cradle(1, True); rb.cradle(2, True)
    mjcf.set_xray(m, a.xray)

    def script():
        # let the chassis settle on its wheels before anything is commanded
        for _ in range(40):
            rb.stop(); yield
        # beam 2 first -- see F49 in params: the other order cannot be driven
        if a.only:
            yield from guard(pursue(rb, 205.0, 430.0, speed=220.0, tol=30.0), 22.0)
            yield from guard(place_beam(rb, a.only, clk=lambda: d.time), 55.0)
        else:
            yield from seal_quarantine(rb, clk=lambda: d.time)

    rig = {"cam": None}

    def on_key(keycode):
        if rig["cam"] is not None:
            rig["cam"].key(keycode)

    viewer = None
    if a.gui:
        import mujoco.viewer as _mjv
        viewer = _mjv.launch_passive(m, d, key_callback=on_key)
        rig["cam"] = view.CameraRig(
            viewer, m,
            follow=lambda: (rb.pose[0]/1000.0, rb.pose[1]/1000.0, 0.06))
        print(view.HELP)
        if a.speed == 0.0:
            a.speed = 1.0
    run, k, done = script(), 0, False
    wall0 = time.perf_counter()
    while d.time < a.timeout:
        if k % CTRL_DECIM == 0 and not done:
            try: next(run)
            except StopIteration:
                done = True; rb.stop()
                print("beam phase complete at T+%.1f s" % d.time)
        mujoco.mj_step(m, d); k += 1
        if viewer is not None and k % 20 == 0:
            if not viewer.is_running(): break
            rig["cam"].tick()
            viewer.sync()
        if a.speed > 0 and k % 20 == 0:
            lag = d.time/a.speed - (time.perf_counter() - wall0)
            if lag > 0: time.sleep(lag)
        if done and d.time > 0 and k % CTRL_DECIM == 0:
            if not hasattr(main, "_t"): main._t = d.time
            if d.time > main._t + 2.0: break
    if viewer is not None: viewer.close()

    st = beam_state(m, d)
    print("\n--- final beam poses -------------------------------------")
    for i, (x, y, z, q) in enumerate(st, 1):
        yaw, tilt = referee._beam_frame(q)
        tgt = Field.BEAM1_CENTRE if i == 1 else Field.BEAM2_CENTRE
        print("  beam %d  (%7.1f, %7.1f, z=%5.1f)  yaw %6.1f  tilt %4.1f   "
              "target (%.1f, %.1f)  err %5.1f mm"
              % (i, x, y, z, yaw, tilt, tgt[0], tgt[1], np.hypot(x-tgt[0], y-tgt[1])))
    pts, detail = referee.score_beams(st)
    print("\n--- referee (beams) --------------------------------------")
    for _i, what, p in detail:
        print("  %-58s %+4d" % (what, p))
    print("  %-58s %+4d" % ("BEAM TOTAL", pts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
