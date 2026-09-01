"""Agent A end-to-end, on the WHOLE 250-point board.

Sweeps the quarantine, collects three sample discs onto the conveyor, docks the
laboratory on camera and posts one disc into each slot, delivers the hospital's
medical kits, and seals the quarantine with both beams -- all inside 120 s.

The scene is `scene_full_match`, and it is the CURRENT model in every respect:

  * knife shim faced with UHMW/PTFE tape (mu 0.12) -- the samples are no longer
    picked up by being shoved against the west wall
  * brush roller at Xa 260 / Za 31.4, silicone tubes on a O20 hub, finger tips
    on the shell line and clear of their own ramp
  * laboratory plate 5 mm, per the rulebook drawings: the slots are cut clean
    through, so a sample has to clear a 5 mm edge and drop in
  * only the destinations in M2.KIT_AGENT_A ride on this robot; the rest start
    in the deployment box, which is where the second robot sits
  * twelve patients on the field, NOT this robot's job -- they belong to the
    second robot, and they are here so the route has to drive around them

What you should see: the knife lifts for every transit and drops for the sweep;
the brush spins only with the knife down; the magazine fills and posts one disc
per slot; the kits fall out of a side hopper; both beams are set down against a
wall and the robot backs away from them.

    python scripts/demo_pick_place.py [--seed N] [--video] [--gui]
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mujoco
from rfgyc26 import mjcf, referee, view
from rfgyc26.params import Field, AgentA, M2
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
    ap.add_argument("--vision", choices=("model", "render"), default="model",
                    help="'render': the dock measures REAL frames from the tail "
                         "cameras through perception.LabPipeline (the pipeline "
                         "the Pi will run); 'model': the fast synthetic camera. "
                         "Headless boxes need MUJOCO_GL=osmesa for 'render'.")
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

    xml = mjcf.scene_full_match(discs, rng=rng)
    path = os.path.join(os.path.dirname(__file__), "..", "models", "scene_full_match.xml")
    os.makedirs(os.path.dirname(path), exist_ok=True)   # .gitignore'd, so absent on a fresh clone
    open(path, "w").write(xml)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    rb = AgentARobot(m, d, step_loss=a.step_loss, rng=rng, vision=a.vision)
    rb.fingers(True); rb.gate(False); rb.intake(False)
    rb.cradle(1, True); rb.cradle(2, True)      # beams carried clear of the field

    dbid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "disc%d" % i) for i in range(3)]
    bbid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "beam%d" % i) for i in (1, 2)]
    kbid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "kit%d" % i) for i in range(M2.N_KITS)]
    cbid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cyl%d" % i) for i in range(M2.N_CYL)]
    ccol = []
    for b in cbid:
        g = m.geom_rgba[m.body_geomadr[b]]
        ccol.append("red" if g[0] > 0.6 and g[1] < 0.4 else
                    ("green" if g[1] > 0.5 and g[0] < 0.5 else "yellow"))
    beams = lambda: [(d.xpos[b][0]*1000, d.xpos[b][1]*1000, d.xpos[b][2]*1000,
                      d.xquat[b].copy()) for b in bbid]
    kits  = lambda: [(d.xpos[b][0]*1000, d.xpos[b][1]*1000) for b in kbid]
    cyls  = lambda: [(d.xpos[b][0]*1000, d.xpos[b][1]*1000, c)
                     for b, c in zip(cbid, ccol)]
    mission = mission_agent_a(rb, Field.LAB_HOLE_X, mjcf.LAB_HOLE_Y, CHUTE_OFFSET,
                              clock=lambda: d.time)

    frames, renderer = [], None
    if a.video:
        try:
            renderer = mujoco.Renderer(m, 480, 720)
        except Exception as e:
            print("  (offscreen render unavailable: %s -- continuing without video)" % e)

    beams_buzzer = kits_buzzer = cyls_buzzer = None
    shown = {"xray": a.xray}
    mjcf.set_xray(m, a.xray)

    # The rig has to exist before the viewer can be asked for it and after the
    # key callback has been handed over, so the callback reaches it by box.
    rig = {"cam": None}

    def on_key(keycode):
        # Camera first: those keys are all non-letters (see rfgyc26.view), so
        # they cannot collide with anything below or inside the viewer.
        if rig["cam"] is not None and rig["cam"].key(keycode):
            return
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
        rig["cam"] = view.CameraRig(
            viewer, m,
            follow=lambda: (rb.pose[0]/1000.0, rb.pose[1]/1000.0, 0.06))
        print("  viewer: X toggles the chassis transparent")
        print(view.HELP)

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
            beams_buzzer, kits_buzzer, cyls_buzzer = beams(), kits(), cyls()
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
            rig["cam"].tick()
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
    print("\n--- final sample positions ------------------------------")
    for i, (x, y, z) in enumerate(pos):
        near = min((np.hypot(x-hx, y-mjcf.LAB_HOLE_Y), j) for j, hx in enumerate(Field.LAB_HOLE_X))
        print("  disc %d  (%7.1f, %7.1f, z=%5.2f)   nearest hole %d at %6.1f mm"
              % (i, x, y, z, near[1]+1, near[0]))

    # THE SCORE THAT COUNTS IS THE ONE AT 120 s (rules g.1).  A run that
    # finishes at T+160 scores what was on the field at two minutes, not what it
    # ended up with, so the whole board is snapshotted at the buzzer.
    if at_buzzer is None:
        at_buzzer, beams_buzzer, kits_buzzer, cyls_buzzer = pos, beams(), kits(), cyls()
        note = "  (finished inside the match)"
    else:
        note = ""
    total, parts = referee.score_match(at_buzzer, beams_buzzer, kits_buzzer, cyls_buzzer)
    print("\n--- referee, at the buzzer ------------------------------")
    for name, p, detail in parts:
        print("  %s" % name.upper())
        for i, what, q in detail:
            print("    %-56s %+4d" % (("disc %d: %s" % (i, what)) if i >= 0 else what, q))
        print("    %-56s %+4d" % ("subtotal", p))
    print("  %-58s %+4d%s" % ("TOTAL AT THE BUZZER", total, note))
    print("  %-58s %s" % ("match budget",
                          "WITHIN (%.1f s of 120)" % d.time if d.time <= MATCH
                          else "the route ran to %.1f s; scored at 120" % d.time))
    print("  (sim %.1f s in %.1f s wall clock)" % (d.time, time.time()-t0))
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
    return 0 if total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
