"""Agent A end-to-end: sweep the quarantine, collect three sample discs on the
conveyor, reverse-dock the laboratory and post one disc into each hole.

    python scripts/demo_pick_place.py [--seed N] [--video] [--gui]
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mujoco
from rfgyc26 import mjcf, referee
from rfgyc26.params import Field, AgentA
from rfgyc26.robot import AgentARobot
from rfgyc26.route import mission_agent_a

CHUTE_OFFSET = AgentA.AXLE_X - AgentA.CHUTE_X          # 109.5 mm behind the axle
CTRL_DECIM = 20                                        # 1 kHz sim -> 50 Hz control


def random_discs(rng, n=3):
    """Senior: samples are randomised inside the quarantine each match."""
    pts = []
    while len(pts) < n:
        p = (rng.uniform(60, 245), rng.uniform(80, 230))
        if all(np.hypot(p[0]-q[0], p[1]-q[1]) > 90 for q in pts):
            pts.append(p)
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--video", action="store_true", help="write out/pick_place.mp4 frames")
    ap.add_argument("--gui", action="store_true", help="open the interactive viewer")
    # Viewer controls (standard MuJoCo): left-drag orbits, RIGHT-drag pans,
    # scroll zooms, [ and ] cycle the fixed cameras (field / lab / quar /
    # A_chase), Esc returns to the free camera, Tab toggles the side panel.
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--step-loss", type=float, default=0.0)
    ap.add_argument("--xray", action="store_true",
                    help="start with the chassis plates transparent; press X in "
                         "the viewer to toggle")
    ap.add_argument("--speed", type=float, default=0.0,
                    help="playback rate vs real time: 1.0 = real time, 0.25 = quarter "
                         "speed, 0 = as fast as the machine manages (default)")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    discs = random_discs(rng)
    print("sample discs at: " + ", ".join("(%.0f, %.0f)" % p for p in discs))

    xml = mjcf.scene_pick_place(discs)
    path = os.path.join(os.path.dirname(__file__), "..", "models", "scene_pick_place.xml")
    os.makedirs(os.path.dirname(path), exist_ok=True)   # .gitignore'd, so absent on a fresh clone
    open(path, "w").write(xml)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    rb = AgentARobot(m, d, step_loss=a.step_loss, rng=rng)
    rb.fingers(True); rb.gate(False)

    dbid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "disc%d" % i) for i in range(3)]
    mission = mission_agent_a(rb, Field.LAB_HOLE_X, mjcf.LAB_HOLE_Y, CHUTE_OFFSET,
                              clock=lambda: d.time)

    frames, renderer = [], None
    if a.video:
        try:
            renderer = mujoco.Renderer(m, 480, 720)
        except Exception as e:
            print("  (offscreen render unavailable: %s -- continuing without video)" % e)

    shown = {"xray": a.xray}
    mjcf.set_xray(m, a.xray)

    def on_key(keycode):
        # X toggles the chassis plates in and out of view.  Rendering only --
        # geom_rgba does not touch contact, so the run is unaffected.
        if keycode in (ord("X"), ord("x")):
            shown["xray"] = not shown["xray"]
            mjcf.set_xray(m, shown["xray"])
            print("  [X] chassis %s" % ("transparent" if shown["xray"] else "solid"))

    viewer = None
    if a.gui:
        import mujoco.viewer as _mjv          # 'import mujoco.viewer' would shadow the global
        viewer = _mjv.launch_passive(m, d, key_callback=on_key)
        print("  viewer: X toggles the chassis transparent, [ and ] cycle cameras,"
              " right-drag pans, scroll zooms")

    if a.gui and a.speed == 0.0:
        a.speed = 1.0            # unthrottled is unwatchable; pace the viewer
    t0, done, k = time.time(), False, 0
    wall0 = time.perf_counter()
    settle = 2500      # let the last release settle before the referee judges
    MATCH = 120.0      # rules g.1 -- what is not posted by then does not count
    at_buzzer = None
    while d.time < a.timeout:
        if at_buzzer is None and d.time >= MATCH:
            at_buzzer = [(d.xpos[b][0]*1000, d.xpos[b][1]*1000, d.xpos[b][2]*1000)
                         for b in dbid]
        if k % CTRL_DECIM == 0 and not done:
            try: next(mission)
            except StopIteration:
                done = True; rb.stop()
                print("mission complete at T+%.1f s" % d.time)
        mujoco.mj_step(m, d)
        k += 1
        if renderer is not None and k % 100 == 0:
            renderer.update_scene(d, camera=-1); frames.append(renderer.render())
        if viewer is not None and k % 20 == 0:
            if not viewer.is_running(): break
            viewer.sync()
        if a.speed > 0 and k % 20 == 0:
            # launch_passive hands the physics loop to US, so the viewer does no
            # pacing of its own -- without this the run is as fast as the CPU
            # allows, which is ~3x real time here.
            lag = d.time / a.speed - (time.perf_counter() - wall0)
            if lag > 0: time.sleep(lag)
        if done:
            settle -= 1
            if settle <= 0: break
    if viewer is not None: viewer.close()

    pos = [(d.xpos[b][0]*1000, d.xpos[b][1]*1000, d.xpos[b][2]*1000) for b in dbid]
    pts, detail = referee.score_discs(pos)
    print("\n--- final sample positions ------------------------------")
    for i, (x, y, z) in enumerate(pos):
        near = min((np.hypot(x-hx, y-mjcf.LAB_HOLE_Y), j) for j, hx in enumerate(Field.LAB_HOLE_X))
        print("  disc %d  (%7.1f, %7.1f, z=%5.2f)   nearest hole %d at %6.1f mm"
              % (i, x, y, z, near[1]+1, near[0]))
    print("\n--- referee ---------------------------------------------")
    for i, what, p in detail:
        print("  %-22s %+4d" % (("disc %d: %s" % (i, what)) if i >= 0 else what, p))
    print("  %-22s %+4d   (sim %.1f s in %.1f s wall clock)"
          % ("TOTAL", pts, d.time, time.time()-t0))
    # The match is 120 s (rules g.1).  Anything not posted by then does not count,
    # so state the verdict rather than leaving it implicit in the timings.
    print("  %-22s %s   (%.1f s of 120 s)"
          % ("match budget", "WITHIN" if d.time <= MATCH else
             "OVER by %.0f s" % (d.time - MATCH), d.time))
    # The score that would actually be awarded.  A run finishing at T+160 does not
    # score what it finished with -- it scores what was on the field at 2 minutes.
    bpts = pts if at_buzzer is None else referee.score_discs(at_buzzer)[0]
    print("  %-22s %+4d   <-- THE SCORE THAT COUNTS%s"
          % ("AT THE BUZZER", bpts,
             "" if at_buzzer is not None else "  (finished inside the match)"))
    if frames:
        out = os.path.join(os.path.dirname(__file__), "..", "out")
        os.makedirs(out, exist_ok=True)
        try:
            import imageio.v2 as imageio
            imageio.mimsave(os.path.join(out, "pick_place.mp4"), frames, fps=20)
            print("  wrote out/pick_place.mp4 (%d frames)" % len(frames))
        except ImportError:
            np.save(os.path.join(out, "pick_place_frames.npy"), np.array(frames[::5]))
            print("  %d frames -> out/pick_place_frames.npy "
                  "(pip install 'imageio[ffmpeg]' for mp4)" % len(frames))
    return 0 if pts > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
