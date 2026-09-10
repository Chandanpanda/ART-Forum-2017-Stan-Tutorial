"""RIG: the mount phase alone, and where its parts actually end up.

    python3 sim/scripts/truss/rig_mount.py [--1m] [--shots DIR]

WHY A RIG.  The whole build is twenty-eight minutes of wall clock and the
mount is the last two of them, so every question about the mount cost a
full run to ask.  This starts from a truss that is already loaded and wound
-- `stage="loaded"` welds every rod to the cage -- and runs the mount and
bond ops only.

WHAT IT MEASURES, and it is the measurement the whole build was missing:
each mount part's ENDS, in the cage's own frame, against `mount.solve`'s
drawing.  Not the tracker against its own target, which is what "every rod
placed within 0.45 mm and 0.0 degrees" was while the thing being built was
a mess.  Reported per part, because one number cannot say WHICH part, and
split three ways, because one number cannot say WHOSE FAULT it is:

    plan    what the release op ASKED for, against the drawing
    track   where the part ended up, against what the op asked
    built   where it ended up, against the drawing

WHAT IT FOUND SO FAR.  `plan` is 0.00 on every part -- the solver is
exact.  All of it is `track`: two struts an end wanted a gripper yaw of
142.3 degrees against a 95 degree stop, and the servo clamped and answered
"settled" (fixed -- `mount.poses_for` offers the rod's own end-for-end
symmetry now, and `schedule` refuses a pose outside the stop).  What is
left is the HEAD KIT: it is tacked about 24 mm from its target and the two
struts laid straight after it inherit that.  Its design pose clears every
batten and strut by 4.5 mm, so it is the approach or the seating, not the
geometry.

STARTING IT: `stage="truss"` -- the truss built and welded, the mount still
racked.  Given "loaded" instead, the mount rods start welded at their
nominal poses, every grip closes on nothing and the rig reports 0.44 mm
without laying anything.  A rig that cannot fail is not a measurement.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from truss import glenv                          # noqa: E402
glenv.headless()
import numpy as np                               # noqa: E402
import mujoco                                    # noqa: E402

from truss import (structure, geometry, fixture, mjcf, cell, approach,   # noqa: E402
                   schedule, process, vision, mount)
from truss.schedule import Plan                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--1m", dest="metre", action="store_true")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--shots")
    ap.add_argument("--sweep", metavar="DIR",
                    help="frames tiled into one contact sheet -- READ IT")
    ap.add_argument("--sweep-fps", type=float, default=1.0)
    ap.add_argument("--window", nargs=2, type=float, metavar=("T0", "T1"))
    ap.add_argument("--end", type=int, default=0,
                    help="which nose to frame the sweep on")
    ap.add_argument("--trace", type=int, metavar="ROD",
                    help="follow one part through its own ops: what the op "
                         "asked of each axis, what the axis did, where the "
                         "part is, and what is touching it")
    ap.add_argument("--phases", default="mount,bond")
    a = ap.parse_args()

    t = structure.TRUSS_1M if a.metre else structure.TRUSS_300
    g = geometry.TrussGeometry(t)
    M = mount.solve(g, d_strut=t.d_diag,
                    standoff_mm=mount.fov_standoff(g, d_strut=t.d_diag))
    g.attach_mount(M)
    fx = fixture.Fixture(g)
    full = schedule.plan(g, fx, approach.plan_truss(g, fx, span=0.0))
    want = set(a.phases.split(","))
    P = Plan()
    op_theta, th_now = {}, 0.0
    for op in full.ops:
        if op.kind == "index":
            th_now = float(op.args.get("theta", th_now))
        if op.kind == "release" and "rod" in op.args:
            op_theta[op.args["rod"]] = th_now
        if op.phase in want:
            P.ops.append(op)
    n_drops = sum(1 for o in full.ops if o.kind in ("dose", "bond")) + 4
    m = mujoco.MjModel.from_xml_string(
        mjcf.scene_cell(g, fx, stage="truss", n_drops=n_drops))
    d = mujoco.MjData(m)
    c = cell.CellSim(m, d, g, fx, rng=np.random.default_rng(a.seed))
    clk = cell.SimClock(m, d, c)
    for _ in range(25):
        clk.tick()
    vis = vision.ModelVision(c, rng=np.random.default_rng(a.seed + 1))
    msgs = []
    ex = process.Executor(c, c, c, c, c, c, clk, g, fx, P, vision=vis,
                          log=lambda s: (msgs.append(s), print("   ", s)))
    strip = None
    if a.sweep:
        os.makedirs(a.sweep, exist_ok=True)
        from truss import filmstrip
        srend = mujoco.Renderer(m, height=560, width=740)
        scam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(scam)
        xc = 0.0 if a.end == 0 else t.length
        sx = -1.0 if a.end == 0 else 1.0
        scam.lookat[:] = [(xc + sx * 20.0) * 1e-3, 0.0, 0.0]
        scam.distance = 0.15
        scam.azimuth = 125.0 if a.end == 0 else 55.0
        scam.elevation = -12.0
        strip = filmstrip.Filmstrip(srend, scam, fps=a.sweep_fps,
                                    window=tuple(a.window) if a.window else None)
    print("%s: %d mount parts, %d ops of %d, %.1f min planned"
          % (t.name, len(g.mount_rods), len(P.ops), len(full.ops), P.total / 60.0))
    tr = None
    if a.trace is not None:
        b_tr = c.b_rod[a.trace]
        gname = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        gof = lambda b: range(m.body_geomadr[b],
                              m.body_geomadr[b] + m.body_geomnum[b])
        mine = set(gof(b_tr))

        def tr(tag, op=None):
            R0 = geometry.rot_x(-c.cage_truth())
            q0, q1 = c.rod_pose(a.trace)
            mid = R0 @ (0.5 * (np.asarray(q0) + np.asarray(q1)))
            r = [x for x in g.mount_rods if x.index == a.trace][0]
            hits = sorted({(round(d.contact[k].dist * 1000, 2),
                            gname(d.contact[k].geom1), gname(d.contact[k].geom2))
                           for k in range(d.ncon)
                           if d.contact[k].geom1 in mine
                           or d.contact[k].geom2 in mine})[:3]
            print("   %7.2f %-9s ax(%8.2f %7.2f %7.2f) w %6.1f gz %6.2f "
                  "grip %s held=%s kept=%d  mid %s (want %s, %5.2f off)  %s"
                  % (d.time, tag, c.at("x"), c.at("y"), c.at("z"), c.at("w"),
                     c.at("g"), np.round(c.grip_point(), 1),
                     c._held, int(c.kept(a.trace)), np.round(mid, 1),
                     np.round(r.mid, 1), float(np.linalg.norm(mid - r.mid)),
                     hits))

    # WHAT THE AXES ACTUALLY WERE WHEN EACH ROD WAS TACKED.  `plan` says
    # the solver is right and `track` says the tool reached the commanded
    # point -- neither can see the cage sitting a degree off its index or
    # the yaw servo a degree off its target, and a rod is laid by all four.
    at_lay = {}
    t0 = time.time()
    seen, near = [0], [False]
    for _ in ex.run():
        clk.tick()
        if strip is not None:
            strip.maybe(d.time, d)
        if len(ex.timeline) > seen[0] or tr is not None:
            pass
        if ex.timeline and len(ex.timeline) > seen[0]:
            _op = ex.timeline[-1][0]
            if _op.kind == "release" and "rod" in _op.args:
                at_lay[_op.args["rod"]] = (c.cage_truth(), c.at("w"))
        if tr is not None and len(ex.timeline) > seen[0]:
            seen[0] = len(ex.timeline)
            op = ex.timeline[-1][0]
            mine_op = op.args.get("rod") == a.trace
            if mine_op:
                near[0] = True
            if near[0]:
                tr("%s/%s" % (op.phase, op.kind), op)
            if op.kind == "retract" and c._held is None and not mine_op:
                near[0] = near[0] and c.kept(a.trace) == 0
    if strip is not None:
        strip.maybe(d.time + 1e9, d)
    print("ran %d of %d ops in %.0f s wall, %.1f min simulated; %d warnings"
          % (len(ex.timeline), len(P.ops), time.time() - t0, d.time / 60.0,
             len(msgs)))

    # WHOSE FAULT IS IT.  Three numbers per part, because one cannot say:
    #   plan   -- what the release op ASKED for, against the drawing
    #   track  -- where the part ended up, against what the op asked
    #   built  -- where it ended up, against the drawing
    # A big `plan` is a solver fault; a big `track` with a small `plan` is
    # the machine not getting there; a small `track` and a big `built` is
    # the measurement lying to itself, which is how this started.
    asked = {}
    for op in P.ops:
        if op.kind == "release" and "rod" in op.args and "pose" in op.args:
            asked[op.args["rod"]] = (np.asarray(op.args["pose"], float),
                                     float(op.args.get("yaw", 0.0)))
    R = geometry.rot_x(-c.cage_truth())
    rows = []
    for r in g.mount_rods:
        q0, q1 = c.rod_pose(r.index)
        a0, a1 = R @ np.asarray(q0, float), R @ np.asarray(q1, float)
        mid = 0.5 * (a0 + a1)
        e = min(max(float(np.linalg.norm(a0 - r.p0)),
                    float(np.linalg.norm(a1 - r.p1))),
                max(float(np.linalg.norm(a0 - r.p1)),
                    float(np.linalg.norm(a1 - r.p0))))
        want, want_yaw = asked.get(r.index, (None, 0.0))
        # the op's pose is in the WORLD at the cage angle it was laid at,
        # so bring it back the same way the part was brought back
        plan_e = trak = tilt = float("nan")
        th_at = None
        if want is not None:
            th_at = op_theta.get(r.index)
            w_cage = (geometry.rot_x(-th_at) @ want) if th_at is not None else want
            plan_e = float(np.linalg.norm(w_cage - r.mid))
            # TRACKING IS NOT JUST THE MIDPOINT.  A rod at the right place
            # and the wrong angle reads zero here, and three struts a truss
            # were doing exactly that -- sitting at a direction that matches
            # NO pose the plan offers, midpoint dead on.
            got_u = (a1 - a0) / float(np.linalg.norm(a1 - a0))
            want_u = np.asarray(mount.axis_from(th_at, want_yaw), float) \
                if th_at is not None else r.axis
            tilt = float(np.degrees(np.arccos(
                min(1.0, abs(float(got_u @ want_u))))))
            trak = float(np.linalg.norm(mid - w_cage))
        th_got, w_got = at_lay.get(r.index, (float("nan"), float("nan")))
        d_th = ((th_got - (th_at if th_at is not None else 0.0)) + 180.0) % 360.0 - 180.0
        d_w = w_got - want_yaw
        rows.append((e, plan_e, trak, tilt, d_th, d_w, r, a0, a1))
    print("  %-8s %3s %4s  %7s %7s %7s %7s %7s %7s"
          % ("kind", "ix", "end", "plan", "track", "tilt", "dcage", "dyaw", "built"))
    for e, pe, tk, tl, dth, dw, r, a0, a1 in sorted(rows, key=lambda v: -v[0]):
        print("  %-8s %3d %4d  %7.2f %7.2f %6.1fd %6.2fd %6.2fd %7.2f"
              "   at %s .. %s   want %s .. %s"
              % (r.kind, r.index, r.chord, pe, tk, tl, dth, dw, e,
                 np.round(a0, 1), np.round(a1, 1),
                 np.round(r.p0, 1), np.round(r.p1, 1)))
    print("mount built within %.2f mm and %.1f degrees of its own geometry"
          % (max(v[0] for v in rows),
             max((v[3] for v in rows if v[3] == v[3]), default=0.0)))
    if strip is not None:
        from PIL import Image
        from truss import filmstrip as _fs
        for i, f in enumerate(strip.frames):
            Image.fromarray(f).save(os.path.join(a.sweep, "r%03d.png" % i))
        sheet = _fs.contact_sheet(strip.frames,
                                 os.path.join(a.sweep, "sweep.png"),
                                 labels=["%.0fs" % v for v in strip.times])
        print("wrote %d frames and %s -- READ THE SHEET"
              % (len(strip.frames), sheet))
    if a.shots:
        os.makedirs(a.shots, exist_ok=True)
        from PIL import Image
        rend = mujoco.Renderer(m, height=720, width=960)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        for end, sx in ((0, -1.0), (1, +1.0)):
            xc = 0.0 if end == 0 else t.length
            cam.lookat[:] = [(xc + sx * 25.0) * 1e-3, 0.0, 0.0]
            cam.distance = 0.18
            cam.azimuth, cam.elevation = (125.0 if end == 0 else 55.0), -12.0
            rend.update_scene(d, cam)
            Image.fromarray(rend.render()).save(
                os.path.join(a.shots, "nose%d.png" % end))
        print("wrote 2 stills to %s -- LOOK AT THEM" % a.shots)
    return 0


if __name__ == "__main__":
    sys.exit(main())
