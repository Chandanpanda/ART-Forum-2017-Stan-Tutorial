"""The 'place' half of pick-and-place, isolated: Agent A docked in reverse over a
laboratory hole, one gate stroke metering one disc out of the chute-magazine."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mujoco
from rfgyc26 import mjcf, referee
from rfgyc26.robot import AgentARobot
from rfgyc26.route import align_reverse
from rfgyc26.params import Field, AgentA

HY  = mjcf.LAB_HOLE_Y
OFF = AgentA.AXLE_X - AgentA.CHUTE_X          # chute sits 109.5 mm behind the axle
m = mujoco.MjModel.from_xml_string(mjcf.scene_pick_place([(2500,2400),(2600,2500),(2700,2600)]))
gj = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "A_gate_j")]

total = 0
for hi, hx in enumerate(Field.LAB_HOLE_X):
    d = mujoco.MjData(m); rb = AgentARobot(m, d); rb.fingers(True); rb.gate(False); rb.intake(False)
    d.qpos[0], d.qpos[1] = hx/1000.0, (HY-OFF)/1000.0
    d.qpos[3:7] = [np.cos(np.radians(135)), 0, 0, np.sin(np.radians(135))]   # yaw 270
    db = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "disc0")
    q  = m.jnt_qposadr[m.body_jntadr[db]]
    d.qpos[q:q+3] = [hx/1000.0, HY/1000.0, 0.030]         # one disc in the magazine
    d.qpos[2] = 0.004                                      # clear the 3 mm plate on drop-in
    mujoco.mj_forward(m, d)
    for _ in range(1500): mujoco.mj_step(m, d)
    # close the dock on the chute's MEASURED position (see route.align_reverse)
    align = align_reverse(rb, OFF, hx, HY, 270.0)
    for _ in range(900):
        try: next(align)
        except StopIteration: break
        for _ in range(20): mujoco.mj_step(m, d)
    rb.stop()
    for _ in range(400): mujoco.mj_step(m, d)
    dock = rb.pose
    cx, cy = rb.chute_xy(OFF)
    rb.gate(True)
    for _ in range(500): mujoco.mj_step(m, d)
    stroke = d.qpos[gj]*1000
    rb.gate(False)
    for _ in range(2500): mujoco.mj_step(m, d)
    w = d.xpos[db]*1000
    pts, det = referee.score_discs([(w[0], w[1], w[2])])
    total += pts
    print("hole %d  chute(%.1f,%.1f) vs hole(%.1f,%.1f) err %.1f mm  stroke %.0f  disc->(%.0f,%.0f,%.1f)  %-14s %+d"
          % (hi+1, cx, cy, hx, HY, np.hypot(cx-hx, cy-HY), stroke, w[0], w[1], w[2], det[0][1], pts))
print("\ntotal %+d   (disc-in-hole radial clearance is 2 mm and the measured capture"
      " radius is 2-3 mm -- the lead-in buys nothing, see F21)" % total)
