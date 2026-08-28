"""Tier-2 rig: the conveyor, isolated.

This is the verified belt validation -- the model is written inline rather than
generated, so it is a fixed, reproducible reference for the two numbers the Rev C
spec asserts about the conveyor.  It is also the proof that MuJoCo's native
<geom surfacevel> is the right primitive for this design.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mujoco

INCLINE, SPEED, MU = 11.0, 0.060, 0.6
MASS, R, H = 0.005, 0.010, 0.010          # O20 x 20 wooden cylinder, 5 g

def rig(n, x0=0.020, gate=True):
    th = np.radians(-INCLINE); c, s = np.cos(th), np.sin(th)
    bodies = ""
    for i in range(n):
        x, z = x0 + i*0.021, 0.012
        bodies += f"""
    <body name="p{i}" pos="{x*c + z*s:.5f} 0 {-x*s + z*c:.5f}" euler="0 {-INCLINE} 0">
      <freejoint/>
      <geom name="p{i}_g" type="cylinder" size="{R} {H}" mass="{MASS}"
            condim="6" friction="{MU} 0.004 0.0002"/>
    </body>"""
    gate_geom = ('<geom name="gate" type="box" size="0.002 0.058 0.015" '
                 'pos="0.098 0 0.015"/>') if gate else ""
    return f"""<mujoco>
  <option timestep="0.0005" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 .1" pos="0 0 -0.2"
          friction="{MU} 0.005 0.0001"/>
    <body euler="0 {-INCLINE} 0">
      <geom name="belt"  type="box" size="0.096 0.058 0.002" condim="6"
            friction="{MU} 0.005 0.0001" surfacevel="{SPEED} 0 0  0 0 0"/>
      {gate_geom}
      <geom name="wallL" type="box" size="0.096 0.002 0.015" pos="0 0.060 0.015"/>
      <geom name="wallR" type="box" size="0.096 0.002 0.015" pos="0 -0.060 0.015"/>
    </body>{bodies}
  </worldbody>
</mujoco>"""

# --- 1. carry on the incline ------------------------------------------------
m = mujoco.MjModel.from_xml_string(rig(1, x0=-0.070, gate=False)); d = mujoco.MjData(m)
for _ in range(3000): mujoco.mj_step(m, d)      # 1.5 s -- still on the belt
v = np.linalg.norm(d.qvel[:3]) * 1000
print("carry   : piece %.1f mm/s against a %.0f mm/s belt  (slip %.1f mm/s)"
      % (v, SPEED*1000, SPEED*1000 - v))
print("          mu %.2f vs tan(%.0f) = %.3f  ->  %.1fx margin  [spec 3.2 claims ~3x]"
      % (MU, INCLINE, np.tan(np.radians(INCLINE)), MU/np.tan(np.radians(INCLINE))))

# --- 2. accumulation against a closed gate ----------------------------------
m = mujoco.MjModel.from_xml_string(rig(4)); d = mujoco.MjData(m)
gate = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "gate")
for _ in range(16000): mujoco.mj_step(m, d)     # 8 s
f6, tot = np.zeros(6), 0.0
for k in range(d.ncon):
    c = d.contact[k]
    if gate in (c.geom1, c.geom2):
        mujoco.mj_contactForce(m, d, k, f6); tot += abs(f6[0])
pred = MU * MASS * 9.81 * 4
print("accum   : %.4f N on the closed gate, %.4f N predicted by mu*m*g*4"
      % (tot, pred))
print("          [spec 3.2 quotes ~0.12 N for four pieces]")
print("          passive: no actuator and no sensor holds this queue")
