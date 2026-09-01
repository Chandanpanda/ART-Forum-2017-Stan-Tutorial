"""Phase 0d: the PIXEL pipeline, against rendered truth.  Run on every change.

The synthetic camera (robot.see_lab) asserts an error MODEL; this suite
measures the actual pipeline on actual frames: build the scene, park the
robot at a grid of dock poses, render the tail cameras, run
perception.LabPipeline, and compare against where the slots really are.
Requires an offscreen GL backend -- on a headless Linux box:

    MUJOCO_GL=osmesa python3 sim/scripts/check_perception.py [-v]

Gates are BANDED by what the consumer needs: the dock only ever closes on
the slot directly behind the tail (the trim axis is the tight one); the
neighbours only steer the 36 mm step-across decision.  The residual ~1 mm
range bias inside these gates is the MODEL's polygonal bore (16 box segments
standing in for a drilled circle), not the pipeline: it shrinks with segment
count and does not exist on hardware.  Widening the gates past these numbers
to get green is how a wrong dock gets confident -- do not.
"""
import os, sys
if os.name == "posix" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import time
import numpy as np, mujoco
from rfgyc26 import mjcf, perception
from rfgyc26.params import Field, Vision, M2
from rfgyc26.robot import AgentARobot, SimCameras, SimClock

VERBOSE = "-v" in sys.argv
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def build():
    xml = mjcf.scene_full_match([(2500., 2400.), (2600., 2500.), (2700., 2600.)],
                                robot_pose=(431.5, 158.0, 270.0),
                                rng=np.random.default_rng(0), kits_aboard=False)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    rb = AgentARobot(m, d)
    rb.fingers(True); rb.gate(False); rb.intake(False)
    rb.cradle(1, True); rb.cradle(2, True)
    clk = SimClock(m, d)
    for _ in range(50):
        clk.tick()
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "agentA")
    jadr = m.jnt_qposadr[m.body_jntadr[bid]]
    return m, d, rb, jadr


def teleport(m, d, jadr, x, y, th):
    d.qpos[jadr:jadr+2] = x/1000.0, y/1000.0
    t = np.radians(th)/2.0
    d.qpos[jadr+3:jadr+7] = np.cos(t), 0.0, 0.0, np.sin(t)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)


def robot_frame(pose, wx, wy):
    x, y, th = pose
    t = np.radians(th)
    dx, dy = wx - x, wy - y
    return (dx*np.cos(t) + dy*np.sin(t), -dx*np.sin(t) + dy*np.cos(t))


def main():
    m, d, rb, jadr = build()
    cams = SimCameras(m, d, rng=None)                  # ideal calibration
    pipe = perception.LabPipeline(cams.calib())

    # ------------------------------------------------ accuracy over the grid
    hits, per_call = [], []
    missed_docked = []
    poses = []
    # The legal look band: the first look happens at axle Y 158 (nose 15 mm
    # off the south wall) and the close re-look at ~Y 181 (bore 112 mm from
    # the slot, reverse_track's stop_at).  North of ~Y 190 the slot falls
    # into the blind cone under the tail; south of 158 the nose is in the
    # wall.  The grid covers that band plus the dock's angular tolerance.
    for hx in Field.LAB_HOLE_X:
        poses += [(hx, 158.0, 270.0), (hx + 25.0, 158.0, 266.0),
                  (hx - 20.0, 150.0, 274.0), (hx, 181.5, 270.0)]
    for pose in poses:
        teleport(m, d, jadr, *pose)
        imgL, imgR, _ = cams.frames()
        t0 = time.perf_counter()
        out = pipe.slots(imgL, imgR)
        per_call.append(time.perf_counter() - t0)
        truth = [robot_frame(pose, hx, mjcf.LAB_HOLE_Y)
                 for hx in Field.LAB_HOLE_X]
        docked_found = False
        for x, y, z, mode in out:
            err, (tx, ty) = min((np.hypot(t[0]-x, t[1]-y), t) for t in truth)
            hits.append((abs(ty), x-tx, y-ty, mode, pose))
            if abs(ty) < 60.0:
                docked_found = True
        if not docked_found and min(abs(t[1]) for t in truth) < 60.0 \
                and pose[1] > 130.0:
            missed_docked.append(pose)
    hits = [(b, ex, ey, mo, po) for b, ex, ey, mo, po in hits]
    docked = [h for h in hits if h[0] < 60.0]
    near = [h for h in hits if 60.0 <= h[0] < 200.0]
    far = [h for h in hits if h[0] >= 200.0]

    check("every look-band pose measures the slot it is parked at",
          not missed_docked, "missed at %s" % (missed_docked or "none"))
    w = max(abs(h[2]) for h in docked)
    check("docked slot: trim-axis error under 1.0 mm, every pose",
          w < 1.0, "worst %.2f mm over %d measurements" % (w, len(docked)))
    w = max(np.hypot(h[1], h[2]) for h in docked)
    check("docked slot: total error under 2.5 mm, every pose",
          w < 2.5, "worst %.2f mm" % w)
    w = max((abs(h[2]) for h in near), default=0.0)
    check("neighbour slots (60-200 mm off-axis): lateral under 2.5 mm",
          w < 2.5, "worst %.2f mm over %d" % (w, len(near)))
    w = max((abs(h[2]) for h in far), default=0.0)
    check("far slots (over 200 mm off-axis): lateral under 5 mm "
          "(vs the 36 mm step-across decision)",
          w < 5.0, "worst %.2f mm over %d" % (w, len(far)))
    st = [h for h in hits if h[3] == "stereo"]
    check("the docked slot is confirmed by BOTH eyes somewhere on the grid",
          any(h[0] < 60.0 for h in st), "%d stereo of %d" % (len(st), len(hits)))
    check("pipeline cost holds 20 FPS with margin (under 250 ms/pair)",
          np.median(per_call) < 0.25,
          "median %.0f ms" % (1000*np.median(per_call)))

    # --------------------------------------------------- calibration bias on
    cams_b = SimCameras(m, d, rng=np.random.default_rng(7))
    pipe_b = perception.LabPipeline(cams_b.calib())
    teleport(m, d, jadr, 431.5, 158.0, 270.0)
    imgL, imgR, _ = cams_b.frames()
    out = pipe_b.slots(imgL, imgR)
    errs = []
    for x, y, z, mode in out:
        tx, ty = min(((robot_frame((431.5, 158.0, 270.0), hx, mjcf.LAB_HOLE_Y))
                      for hx in Field.LAB_HOLE_X),
                     key=lambda t: np.hypot(t[0]-x, t[1]-y))
        errs.append(np.hypot(x-tx, y-ty))
    check("with the drawn mounting bias the docked slot still lands "
          "inside the dock's own budget",
          errs and min(errs) < 3.5, "best %.2f mm" % (min(errs) if errs else -1))

    # ------------------------------------------------------------- refusal
    # Tail toward the NORTH field: kit zones and open floor, no laboratory.
    # (Heading 90 here would aim the cameras straight AT the plate from its
    # north side -- the first version did, and the pipeline correctly
    # measured the real slots, failing the check for the check's own error.)
    teleport(m, d, jadr, 600.0, 700.0, 270.0)
    imgL, imgR, _ = cams.frames()
    check("no laboratory in view produces NO slots, not fabrications",
          pipe.slots(imgL, imgR) == [])

    # ------------------------------------------------------------- colours
    teleport(m, d, jadr, 350.0, 650.0, 0.0)           # tail toward SIDE_L
    imgL, imgR, _ = cams.frames()
    pose = rb.pose
    seen, wrong = 0, []
    for i in range(M2.N_CYL):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cyl%d" % i)
        g = m.body_geomadr[b]
        rgba = m.geom_rgba[g]
        truth_col = ("red" if rgba[0] > 0.6 and rgba[1] < 0.4 else
                     "green" if rgba[1] > 0.5 and rgba[0] < 0.5 else "yellow")
        p_w = d.geom_xpos[g]*1000.0
        # occlusion-checked from the left eye, as the pipeline would be used
        eye = cams.calib().left
        t = np.radians(pose[2])
        R2 = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        eye_w = np.array([*(np.array(pose[:2]) + R2 @ eye.T[:2]), eye.T[2]])
        vec = p_w - eye_w
        rng_m = np.linalg.norm(vec)/1000.0
        gid = np.zeros(1, np.int32)
        hit = mujoco.mj_ray(m, d, eye_w/1000.0 + vec/np.linalg.norm(vec)*0.045,
                            vec/np.linalg.norm(vec), None, 1, -1, gid)
        if not (0 <= hit <= rng_m + 0.02) or \
                (gid[0] >= 0 and m.geom_bodyid[gid[0]] != b):
            continue
        p_r = robot_frame(pose, p_w[0], p_w[1])
        got = perception.classify_patch(imgL, eye, (p_r[0], p_r[1], p_w[2]))
        if got is not None:
            seen += 1
            if got != truth_col:
                wrong.append((i, truth_col, got))
    check("cylinder colours read correctly at their known stickers "
          "(3+ visible, zero wrong)",
          seen >= 3 and not wrong, "%d read, wrong: %s" % (seen, wrong or "none"))

    # --------------------------------------------------------------- summary
    fails = [r for r in RESULTS if not r[1]]
    for name, ok, detail in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("PASS" if ok else "FAIL", name,
                                  ("  [%s]" % detail) if detail else ""))
    print("%d checks, %d failed" % (len(RESULTS), len(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
