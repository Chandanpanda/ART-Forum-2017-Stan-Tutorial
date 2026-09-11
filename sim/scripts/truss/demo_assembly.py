"""The whole build, end to end, in MuJoCo: rods, thread, glue, cameras.

    python3 sim/scripts/truss/demo_assembly.py                  # 300 mm, headless
    python3 sim/scripts/truss/demo_assembly.py --gui            # watch it live
    python3 sim/scripts/truss/demo_assembly.py --gui --speed 20 # 20x real time
    python3 sim/scripts/truss/demo_assembly.py --gui --1m       # the metre truss
    python3 sim/scripts/truss/demo_assembly.py --video out.mp4  # record it
    python3 sim/scripts/truss/demo_assembly.py --shots DIR      # stills per phase

ON WINDOWS the interpreter is `python`, not `python3`, and the path is
relative to where you are; from the `sim` directory:

    python scripts\truss\demo_assembly.py --gui --speed 20

Forward slashes work too.  Do NOT set MUJOCO_GL -- truss/glenv.py picks one
per platform, and `osmesa` is a name that only exists on Linux.  The window
opens on the FREE camera framed on the cell, so the mouse is live from the
first frame: left-drag orbits, right-drag pans, scroll zooms; truss/view.py
adds arrows to pan, Home/End to orbit, - and = to zoom, 1/2/3 preset views,
4 close on the head, . to follow it, 0 back to the whole cell.  The window
stays open when the build is finished.

At --speed 1 the film is as long as the build (about twenty minutes of
simulated time); --speed 20 is the one to watch.

WHAT IT RUNS.  One plan, from racks that a person has filled by hand, to a
truss with a camera on each end:

    load    every chord and every diagonal, picked from its rack and
            dropped into the fixture's V's
    wind    every joint, the ring turning round the chord, the band laid
            where the fiducial says the turns went
    dose    resin onto every band
    mount   three battens and six struts an end, plus the HEAD KIT on its
            carrier -- module, collar and the wound tic-tac-toe, made
            whole at station B because this cell cannot wind those four
            crossings (see stationb.py) -- at cage angles the same worm
            indexes and gripper yaws the same servo turns
    bond    an epoxy fillet at every mount rod end, the six struts landing
            on crossings that arrived already wound

Nothing here is a scripted animation.  Every pose comes from the same
`schedule.plan` the checks run, executed by the same `process.Executor`
against the same MuJoCo cell, through the HAL -- so what you watch is what
the checks measured, and if it collides here it collides there.

THE FIXTURE'S OWN COLLAPSE IS NOT IN THIS.  It is a manual step: break six
over-centre knees with one pull on the draw rod and the mandrel withdraws
(see truss/collapse.py, and check_collapse for the numbers).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from truss import glenv                  # noqa: E402  (before mujoco)
if "--gui" not in sys.argv:
    glenv.headless()
import numpy as np                       # noqa: E402
import mujoco                            # noqa: E402

from truss import (structure, geometry, fixture, mjcf, cell, approach,   # noqa: E402
                   schedule, process, vision, inspector, mount, view,
                   filmstrip)
from truss.spec import Vision            # noqa: E402
from truss.geometry import TrussGeometry  # noqa: E402


def build(metre=False, seed=3):
    t = structure.TRUSS_1M if metre else structure.TRUSS_300
    g = TrussGeometry(t)
    M = mount.solve(g, d_strut=t.d_diag,
                    standoff_mm=mount.fov_standoff(g, d_strut=t.d_diag))
    g.attach_mount(M)
    fx = fixture.Fixture(g)
    st = approach.plan_truss(g, fx, span=0.0)
    P = schedule.plan(g, fx, st)
    # ONE DROP PER DOSE, AND THE MOUNT DOSES TOO.  The scene pre-allocates
    # the drop bodies, so it has to know how many joints AND how many
    # fillets the plan will lay, or the executor runs out mid-build.
    n_drops = sum(1 for o in P.ops if o.kind in ("dose", "bond")) + 4
    xml = mjcf.scene_cell(g, fx, stage="empty", n_drops=n_drops)
    return t, g, fx, P, xml, n_drops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--1m", dest="metre", action="store_true")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--video")
    ap.add_argument("--shots")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="wall-clock pacing under --gui; 0 = unpaced")
    ap.add_argument("--vspeed", type=float, default=40.0,
                    help="simulated seconds per recorded second, for --video")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--stalls", action="store_true",
                    help="list the ops whose sensor did not answer in time")
    ap.add_argument("--sweep", metavar="DIR",
                    help="frames at 0.1 fps over the whole run, tiled into one "
                         "contact sheet -- READ IT, this is the review")
    ap.add_argument("--sweep-fps", type=float, default=0.1)
    ap.add_argument("--window", nargs=2, type=float, metavar=("T0", "T1"),
                    help="with --sweep, capture only this stretch of simulated "
                         "time, and at --sweep-fps 1 by default")
    ap.add_argument("--nose", action="store_true",
                    help="frame the sweep on the x=0 nose instead of the cell")
    a = ap.parse_args()

    t, g, fx, P, xml, n_drops = build(a.metre, a.seed)
    s = schedule.summary(P)
    print("%s: %d rods + %d mount parts, %d joints, %d ops, %.1f min planned"
          % (t.name, len(g.rods), len(g.mount_rods), t.n_joints, len(P.ops),
             s["total_min"]))
    for k, v in sorted(s["phase_min"].items(), key=lambda kv: -kv[1]):
        print("    %-6s %6.2f min" % (k, v))

    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    c = cell.CellSim(m, d, g, fx, rng=np.random.default_rng(a.seed))
    clk = cell.SimClock(m, d, c)
    for _ in range(25):
        clk.tick()
    vis = vision.ModelVision(c, rng=np.random.default_rng(a.seed + 1))
    msgs = []
    ex = process.Executor(c, c, c, c, c, c, clk, g, fx, P, vision=vis,
                          log=lambda s2: (msgs.append(s2), print("   ", s2)))

    band_ids = {j.index: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                                           "band%d" % j.index)
                for j in g.joints}

    def show_bands():
        """The wound band as a sleeve that grows with the turns the fiducial
        counted -- the only part of the build that has no geometry of its
        own until it is made."""
        for jix, stt in ex.states.items():
            gid = band_ids[jix]
            if gid < 0 or stt.turns <= 0:
                continue
            m.geom_size[gid][1] = max(stt.width, 0.2) / 2000.0
            m.geom_pos[gid] = g.chord_point(g.joints[jix].chord, stt.centre) / 1000.0
            m.geom_rgba[gid][3] = min(1.0, 0.2 + stt.turns / t.turns)

    frames, shots = [], {}
    rend = None
    vid = None
    if a.video or a.shots:
        rend = mujoco.Renderer(m, height=Vision.H, width=Vision.W)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = [t.length / 2.0 * 1e-3, 0.0, 0.0]
        cam.distance = max(0.6, t.length * 2.2e-3)
        cam.azimuth, cam.elevation = 135.0, -18.0
    strip = None
    if a.sweep:
        # A SEPARATE CAMERA AND RENDERER: the sweep is the review, and it
        # has to keep working when nothing else is being recorded.
        srend = mujoco.Renderer(m, height=480, width=640)
        scam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(scam)
        if a.nose:
            scam.lookat[:] = [-0.010, 0.0, 0.0]
            scam.distance, scam.azimuth, scam.elevation = 0.22, 125.0, -12.0
        else:
            scam.lookat[:] = [t.length / 2.0 * 1e-3, 0.0, 0.0]
            scam.distance = max(0.6, t.length * 2.2e-3)
            scam.azimuth, scam.elevation = 135.0, -18.0
        fps = a.sweep_fps if a.window is None or a.sweep_fps != 0.1 else 1.0
        strip = filmstrip.Filmstrip(srend, scam, fps=fps,
                                    window=tuple(a.window) if a.window else None)
    if a.video:
        # A SEPARATE, SMALLER RENDERER FOR THE FILM.  A twenty-minute build
        # at twenty frames a second is six hundred frames, and six hundred
        # frames of 1280x720 is 1.7 GB held in RAM before anything is
        # written.  The stills stay full size; the film does not need to be.
        vid = mujoco.Renderer(m, height=360, width=640)
    period = a.vspeed / max(a.fps, 1)
    next_frame, phase = [0.0], [None]
    t0 = time.time()

    gen = ex.run()
    last_op = [0]

    def step():
        """One control tick.  False when the plan is finished."""
        clk.tick()
        try:
            next(gen)
        except StopIteration:
            return False
        if len(ex.timeline) > last_op[0]:
            last_op[0] = len(ex.timeline)
            op = ex.timeline[-1][0]
            if op.kind in ("wind", "release", "index", "dose", "bond", "cut"):
                print("%7.1f  %s" % (d.time, op))
            show_bands()
        nonlocal_phase = ex.timeline[-1][0].phase if ex.timeline else "load"
        if vid is not None and d.time >= next_frame[0]:
            vid.update_scene(d, cam)
            frames.append(vid.render().copy())
            next_frame[0] = d.time + period
        if strip is not None:
            strip.maybe(d.time, d)
        if a.shots and nonlocal_phase != phase[0] and rend is not None:
            rend.update_scene(d, cam)
            shots[nonlocal_phase] = rend.render().copy()
            phase[0] = nonlocal_phase
        return True

    if a.gui and not glenv.windowed():
        print("--gui needs a display; this session has none.  Run without it.")
        a.gui = False
    try:
        if a.gui:
            run_windowed(m, d, c, g, fx, step, a.speed)
        else:
            while step():
                pass
    except KeyboardInterrupt:
        print("interrupted")
    wall = time.time() - t0

    print("ran %d of %d ops in %.0f s wall, %.1f min simulated"
          % (len(ex.timeline), len(P.ops), wall, d.time / 60.0))
    print("%d fillets laid, %d bands wound, %d rods released"
          % (len(ex.bonds), sum(1 for st2 in ex.states.values() if st2.turns > 0),
             len(ex.released)))
    errs = {r.index: inspector.seat_error(g, r, *c.rod_pose(r.index),
                                          theta=c.cage_truth())
            for r in g.rods}
    rep = inspector.inspect_truss(g, ex.states, seated=errs, cycle_s=d.time)
    print(inspector.report(rep) if hasattr(inspector, "report") else rep)
    # WHERE THE MOUNT ACTUALLY ENDED UP, against `mount.solve`'s own drawing.
    # NOT THE MACHINE AGAINST ITSELF.  The mount phase once reported every
    # rod placed within 0.45 mm and 0.0 degrees of plan while building a
    # mess: that number was the tracker measured against its own target.
    # This one is the built part, in the cage's frame, against the geometry
    # the struts and the camera and the truss all have to agree about -- and
    # it is measured at the rods' ENDS, because a rod rolled about its own
    # axis has a centre error of zero.
    if g.mount_rods:
        Rb = geometry.rot_x(-c.cage_truth())
        worst, who = 0.0, ""
        for r in g.mount_rods:
            q0, q1 = c.rod_pose(r.index)
            a0, a1 = Rb @ np.asarray(q0, float), Rb @ np.asarray(q1, float)
            e = min(max(float(np.linalg.norm(a0 - r.p0)),
                        float(np.linalg.norm(a1 - r.p1))),
                    max(float(np.linalg.norm(a0 - r.p1)),
                        float(np.linalg.norm(a1 - r.p0))))
            if e > worst:
                worst, who = e, "%s%d" % (r.kind, r.index)
        print("mount built within %.2f mm of its own geometry (worst %s of %d "
              "parts; the head kit arrives whole from station B)"
              % (worst, who, len(g.mount_rods)))
    if msgs:
        print("%d warnings" % len(msgs))
    if ex.slow:
        by = {}
        for op, took in ex.slow:
            by[(op.phase, op.kind)] = by.get((op.phase, op.kind), 0) + 1
        print("ops that overran the plan, by phase: %s"
              % ", ".join("%s/%s x%d" % (p2, k, n)
                          for (p2, k), n in sorted(by.items(), key=lambda kv: -kv[1])))
        if a.stalls:
            for op, took in ex.slow:
                print("    %-6s %-8s planned %.2f s, took %.2f" % (op.phase, op.kind,
                                                                    op.dt, took))

    if a.shots:
        os.makedirs(a.shots, exist_ok=True)
        from PIL import Image
        for k, im in shots.items():
            Image.fromarray(im).save(os.path.join(a.shots, "phase_%s.png" % k))
        rend.update_scene(d, cam)
        Image.fromarray(rend.render()).save(os.path.join(a.shots, "phase_final.png"))
        # ...and a close-up of each nose, which is what the whole mount
        # phase was for and what a whole-cell view cannot show
        for end, sx in ((0, -1.0), (1, +1.0)):
            xc = (0.0 if end == 0 else t.length)
            cam.lookat[:] = [(xc + sx * 25.0) * 1e-3, 0.0, 0.0]
            cam.distance = 0.20
            cam.azimuth, cam.elevation = (125.0 if end == 0 else 55.0), -15.0
            rend.update_scene(d, cam)
            Image.fromarray(rend.render()).save(
                os.path.join(a.shots, "nose_end%d.png" % end))
        print("wrote %d stills to %s" % (len(shots) + 3, a.shots))
    if strip is not None:
        strip.maybe(d.time + 1e9, d)          # one last frame, whatever the rate
        sheet = filmstrip.contact_sheet(
            strip.frames, os.path.join(a.sweep, "sweep.png"),
            labels=["%.0fs" % v for v in strip.times])
        for i, f in enumerate(strip.frames):
            from PIL import Image
            Image.fromarray(f).save(os.path.join(a.sweep, "s%03d.png" % i))
        print("wrote %d sweep frames and %s -- READ THE SHEET"
              % (len(strip.frames), sheet))
    if a.video and frames:
        got = write_video(a.video, frames, a.fps)
        print("wrote %s: %d frames at %d fps" % (got, len(frames), a.fps))
    return 0


def run_windowed(m, d, c, g, fx, step, speed):
    """Drive the same `step` under MuJoCo's passive viewer.

    THE WINDOW OUTLIVES THE CYCLE.  What the demo is for is the thing it
    made, and closing on the last op leaves nothing to look at."""
    from mujoco import viewer as mjviewer
    from truss.spec import ring_swept_r
    rig = {"cam": None}

    def on_key(code):
        # the camera keys are all non-letters (truss/view.py says why), so
        # they cannot collide with the viewer's own visualisation flags
        if rig["cam"] is not None:
            rig["cam"].key(code)

    with mjviewer.launch_passive(m, d, key_callback=on_key) as v:
        rig["cam"] = view.CameraRig(v, m, view.cell_frame(g, fx),
                                    follow=lambda: c.ring_centre() / 1000.0,
                                    close=view.close_frame(ring_swept_r()))
        print(view.HELP)
        # the clock starts when the window is up: framing the camera and
        # compiling the first frame takes a moment, and pacing from before
        # that makes the demo sprint to catch up
        t_wall = time.time()
        while v.is_running():
            if not step():
                break
            rig["cam"].tick()
            v.sync()
            if speed > 0:
                lag = d.time / speed - (time.time() - t_wall)
                if lag > 0:
                    time.sleep(min(lag, 0.05))
        print("\n  the build is finished; the window is yours.  Close it to exit.")
        while v.is_running():
            rig["cam"].tick()
            v.sync()
            time.sleep(0.02)


def write_video(path, frames, fps):
    """MP4 through ffmpeg if it is there, otherwise a GIF through PIL."""
    import subprocess
    import shutil
    from PIL import Image
    if shutil.which("ffmpeg") and path.endswith(".mp4"):
        h, w, _ = frames[0].shape
        p = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", "%dx%d" % (w, h), "-r", str(fps), "-i", "-",
             "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", path],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        for f in frames:
            p.stdin.write(f.tobytes())
        p.stdin.close()
        p.wait()
        return path
    gif = path.rsplit(".", 1)[0] + ".gif"
    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(gif, save_all=True, append_images=ims[1:],
                duration=int(1000 / fps), loop=0)
    return gif


if __name__ == "__main__":
    sys.exit(main())
