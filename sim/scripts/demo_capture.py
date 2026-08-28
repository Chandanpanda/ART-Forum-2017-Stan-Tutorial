"""The 'pick' half, isolated: one sweep pass collecting three sample discs off
the floor onto the conveyor and stacking them in the chute-magazine."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mujoco
from rfgyc26 import mjcf
from rfgyc26.robot import AgentARobot
from rfgyc26.params import AgentA
from rfgyc26.route import guard, drive_straight, pursue, sweep_line

CH = -(AgentA.AXLE_X - AgentA.CHUTE_X)
DISCS = [(330, 130), (240, 133), (170, 128)]

m = mujoco.MjModel.from_xml_string(mjcf.scene_pick_place(DISCS))
d = mujoco.MjData(m)
rb = AgentARobot(m, d); rb.fingers(True); rb.gate(False)
dbs = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "disc%d" % i) for i in range(3)]

def seq():
    yield from guard(drive_straight(rb, 300.0, speed=220.0), 10.0)
    yield from guard(pursue(rb, 430.0, 130.0, speed=220.0, tol=40.0), 20.0)
    yield from guard(sweep_line(rb, 130.0, 158.0), 60.0)

print("three discs on the floor at " + ", ".join("(%d,%d)" % p for p in DISCS))
g, k = seq(), 0
while d.time < 60:
    if k % 20 == 0:
        try: next(g)
        except StopIteration: break
    mujoco.mj_step(m, d); k += 1

n = 0
for i, b in enumerate(dbs):
    lx, ly, lz = rb.to_local(d.xpos[b])
    inmag = abs(lx - CH) < 34 and abs(ly) < 34 and 8 < lz < 48
    n += inmag
    print("  disc%d  robot-frame (%7.1f, %6.1f, %6.1f)  %s"
          % (i, lx, ly, lz, "STACKED IN MAGAZINE" if inmag else "not captured"))
print("\n%d of 3 collected in %.1f s of match time  (chute axis is x=%.1f)" % (n, d.time, CH))
print("stack heights show them queued on the gate: the magazine is a simple column")
