"""The whole build, end to end, in MuJoCo: rods, thread, glue, cameras.

    python3 sim/scripts/truss/demo_assembly.py                  # 300 mm, headless
    python3 sim/scripts/truss/demo_assembly.py --1m             # the metre truss
    python3 sim/scripts/truss/demo_assembly.py --video out.mp4  # record it
    python3 sim/scripts/truss/demo_assembly.py --gui            # watch it live
    python3 sim/scripts/truss/demo_assembly.py --shots DIR      # stills per phase

WHAT IT RUNS.  One plan, from racks that a person has filled by hand, to a
truss with a camera on each end:

    load    every chord and every diagonal, picked from its rack and
            dropped into the fixture's V's
    wind    every joint, the ring turning round the chord, the band laid
            where the fiducial says the turns went
    dose    resin onto every band
    mount   thirteen rods an end plus the camera on its carrier, at cage
            angles the same worm indexes and gripper yaws the same servo
            turns -- the mount was solved for poses the machine already has
    bond    an epoxy fillet at every mount rod end

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
                   schedule, process, vision, inspector, mount, view)
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
    ap.add_argument("--speed", type=float, default=40.0,
                    help="simulated seconds per recorded second")
    ap.add_argument("--seed", type=int, default=3)
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
    if a.video:
        # A SEPARATE, SMALLER RENDERER FOR THE FILM.  A twenty-minute build
        # at twenty frames a second is six hundred frames, and six hundred
        # frames of 1280x720 is 1.7 GB held in RAM before anything is
        # written.  The stills stay full size; the film does not need to be.
        vid = mujoco.Renderer(m, height=360, width=640)
    period = a.speed / max(a.fps, 1)
    next_frame, phase = 0.0, None
    t0 = time.time()

    gen = ex.run()
    try:
        while True:
            clk.tick()
            try:
                next(gen)
            except StopIteration:
                break
            ph = ex.timeline[-1][0].phase if ex.timeline else "load"
            if vid is not None and d.time >= next_frame:
                vid.update_scene(d, cam)
                frames.append(vid.render().copy())
                next_frame = d.time + period
            if a.shots and ph != phase and rend is not None:
                rend.update_scene(d, cam)
                shots[ph] = rend.render().copy()
                phase = ph
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
    if msgs:
        print("%d warnings" % len(msgs))

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
    if a.video and frames:
        write_video(a.video, frames, a.fps)
        print("wrote %s: %d frames at %d fps" % (a.video, len(frames), a.fps))
    return 0


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
        return
    gif = path.rsplit(".", 1)[0] + ".gif"
    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(gif, save_all=True, append_images=ims[1:],
                duration=int(1000 / fps), loop=0)


if __name__ == "__main__":
    sys.exit(main())
