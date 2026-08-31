"""The 'place' half of pick-and-place, isolated: Agent A squared up to a
laboratory hole, the OAK-D measuring where the slot actually is, the trim slide
aiming the head at it, and one escapement cycle metering out a single disc
(F68/F69)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mujoco
from rfgyc26 import mjcf, referee
from rfgyc26.robot import AgentARobot
from rfgyc26.route import look_lab, pick_slot, slot_world, reverse_track, LOOK_Y
from rfgyc26.params import Field, AgentA

HY  = mjcf.LAB_HOLE_Y
OFF = AgentA.AXLE_X - AgentA.CHUTE_X          # chute sits 109.5 mm behind the axle
m = mujoco.MjModel.from_xml_string(mjcf.scene_pick_place([(2500,2400),(2600,2500),(2700,2600)]))
gj = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "A_gate_l_j")]

total = 0
for hi, hx in enumerate(Field.LAB_HOLE_X):
    d = mujoco.MjData(m); rb = AgentARobot(m, d)
    rb.fingers(True); rb.gate(False); rb.blade(False); rb.intake(False); rb.feed(False)
    d.qpos[0], d.qpos[1] = hx/1000.0, LOOK_Y/1000.0
    d.qpos[3:7] = [np.cos(np.radians(135)), 0, 0, np.sin(np.radians(135))]   # yaw 270
    db = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "disc0")
    q  = m.jnt_qposadr[m.body_jntadr[db]]
    d.qpos[q:q+3] = [hx/1000.0, (LOOK_Y+OFF)/1000.0, 0.030]   # one disc in the magazine
    d.qpos[2] = 0.004                                      # clear the 3 mm plate on drop-in
    mujoco.mj_forward(m, d)
    for _ in range(1500): mujoco.mj_step(m, d)
    # LOOK, then drive the vector.  No iterated approach and no world
    # coordinate for the slot -- what the robot uses is where the camera says
    # the slot is relative to its own bore.
    def spin(gen, ticks=900):
        for _ in range(ticks):
            try: next(gen)
            except StopIteration: return
            for _ in range(20): mujoco.mj_step(m, d)
    spin(look_lab(rb))
    tgt = pick_slot(getattr(rb, "lab_seen", []), 0.0)
    if tgt is not None:
        sw = slot_world(rb, tgt)
        rb.trim(float(np.clip(tgt[1], -AgentA.TRIM_Y, AgentA.TRIM_Y)))
        spin(reverse_track(rb, sw, rb.trim_at(), 270.0, speed=110.0))
    rb.stop()
    for _ in range(300): mujoco.mj_step(m, d)
    dock = rb.pose
    cx, cy = rb.chute_xy(OFF)
    cx -= rb.trim_at()*np.sin(np.radians(dock[2]))
    cy += rb.trim_at()*np.cos(np.radians(dock[2]))
    rb.gate(True)
    for _ in range(500): mujoco.mj_step(m, d)
    stroke = d.qpos[gj]*1000
    rb.gate(False)
    for _ in range(2500): mujoco.mj_step(m, d)
    w = d.xpos[db]*1000
    pts, det = referee.score_discs([(w[0], w[1], w[2])])
    total += pts
    print("hole %d  bore(%.1f,%.1f) vs hole(%.1f,%.1f) err %.1f mm  trim %+.1f  disc->(%.0f,%.0f,%.1f)  %-14s %+d"
          % (hi+1, cx, cy, hx, HY, np.hypot(cx-hx, cy-HY), rb.trim_at(), w[0], w[1], w[2], det[0][1], pts))
print("\ntotal %+d   (a O56 disc in a O60 slot has 2.0 mm of radial clearance."
      "  This rig starts the robot already square and centred, so it exercises"
      " the MEASUREMENT and the drop, not the approach: the mission's own"
      " numbers are in demo_pick_place.)" % total)
