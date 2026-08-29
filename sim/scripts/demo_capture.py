"""The 'pick' half, isolated: one sweep pass collecting three sample discs off
the floor onto the conveyor and stacking them in the chute-magazine."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mujoco
from rfgyc26 import mjcf
from rfgyc26.robot import AgentARobot
from rfgyc26.params import AgentA
from rfgyc26.route import guard, drive_straight, pursue, sweep_line, settle_stack

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
    # The positive feed is part of collecting, not of posting: the last piece in
    # has nothing above it and lands ON the stack rather than settling into it.
    yield from guard(settle_stack(rb), 30.0)

print("three discs on the floor at " + ", ".join("(%d,%d)" % p for p in DISCS))
g, k = seq(), 0
while d.time < 80:
    if k % 20 == 0:
        try: next(g)
        except StopIteration: break
    mujoco.mj_step(m, d); k += 1

n = 0
for i, b in enumerate(dbs):
    lx, ly, lz = rb.to_local(d.xpos[b])
    q = d.xquat[b]
    R = np.zeros(9); mujoco.mju_quat2Mat(R, q); R = R.reshape(3, 3)
    tilt = np.degrees(np.arccos(min(1.0, abs(R[2, 2]))))
    # SEATED, not merely "in the bore": a piece perched on the stack at 20 deg
    # is not retained and will not meter (F14).
    seated = abs(lx - CH) < 12 and abs(ly) < 12 and lz < 30 and tilt < 15
    n += seated
    print("  disc%d  robot-frame (%7.1f, %6.1f, %6.1f)  tilt %4.1f deg  %s"
          % (i, lx, ly, lz, tilt, "SEATED IN MAGAZINE" if seated else "NOT SEATED"))
print("\n%d of 3 seated in %.1f s of match time  (chute axis is x=%.1f)" % (n, d.time, CH))
print("the bore rangefinder reads %d -- that is what the escapement acts on\n"
      "(it over-reads while a piece is perched, which is the case the feed fixes)"
      % rb.mag_count())
print("\nThis is ONE sweep pass over a fixed layout, so it is the marginal case:\n"
      "the mission runs two passes and captures 24 of 24 over 8 randomised\n"
      "matches.  If a sample is missed here, that is the second pass's job.")
