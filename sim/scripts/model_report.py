"""What is actually in the model -- and, just as important, what is NOT.

Run this before trusting any number the simulator prints.  It reads the compiled
MuJoCo model rather than the source, so it cannot flatter the design: every joint,
actuator and sensor listed here is one the solver really integrates.

    python scripts/model_report.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mujoco
from rfgyc26 import mjcf

JT = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
GT = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid",
      5: "cylinder", 6: "box", 7: "mesh"}
m = mujoco.MjModel.from_xml_string(mjcf.scene_pick_place([(300, 130), (240, 133), (170, 128)]))
nm = lambda t, i: mujoco.mj_id2name(m, t, i)

print("RFGYC'26 Agent A -- compiled model inventory")
print("=" * 62)
print("  bodies %d   geoms %d   joints %d   DOF %d   actuators %d   sensors %d"
      % (m.nbody, m.ngeom, m.njnt, m.nv, m.nu, m.nsensor))

print("\nDEGREES OF FREEDOM  (every one is integrated by the solver)")
for i in range(m.njnt):
    lim = ("  range %s" % np.round(m.jnt_range[i], 4)) if m.jnt_limited[i] else ""
    print("  %-12s %-6s%s" % (nm(mujoco.mjtObj.mjOBJ_JOINT, i), JT[m.jnt_type[i]], lim))

print("\nACTUATORS  (closed loop on the joint above; force-limited)")
for i in range(m.nu):
    j = m.actuator_trnid[i][0]
    f = (str(np.round(m.actuator_forcerange[i], 3)) if m.actuator_forcelimited[i]
         else "unlimited")
    print("  %-12s -> %-12s ctrl %-16s force %s"
          % (nm(mujoco.mjtObj.mjOBJ_ACTUATOR, i), nm(mujoco.mjtObj.mjOBJ_JOINT, j),
             np.round(m.actuator_ctrlrange[i], 4), f))

print("\nSENSORS  (what the robot is allowed to know)")
for i in range(m.nsensor):
    t = mujoco.mjtSensor(m.sensor_type[i]).name.replace("mjSENS_", "").lower()
    print("  %-12s %s" % (nm(mujoco.mjtObj.mjOBJ_SENSOR, i), t))

counts = {}
for i in range(m.ngeom):
    counts[GT.get(m.geom_type[i], "?")] = counts.get(GT.get(m.geom_type[i], "?"), 0) + 1
print("\nCOLLISION GEOMETRY  " + ", ".join("%s x%d" % kv for kv in sorted(counts.items())))
print("  All primitives, no meshes -- so it looks blocky, but every one of these")
print("  is a real contact surface, not decoration.  Zoom in (scroll) or press")
print("  [ / ] in the viewer to reach the 'lab' and 'quar' cameras.")

print("""
WHERE THE MODEL IS A STAND-IN  (read this before quoting any result)
======================================================================
These are deliberate and documented, not oversights.  Each one is a place the
simulator is NOT proving what it looks like it is proving.

1. THE SWEEPER STROKE IS NOT SIMULATED.  This is the big one (F1).  The scoop
   shim carries the belt's surfacevel, which stands in for the fingers' ~110 deg
   powered stroke.  A passive shim provably cannot convey a piece across that
   gap, so the fingers are load-bearing -- but they are modelled as static
   funnel walls plus a moving surface, not as a stroke in contact.  If one
   result here should be re-derived before you cut metal, it is this one.

2. THE BELT IS A surfacevel BOX, not a loop over two rollers.  MuJoCo's native
   conveyor primitive.  Validated against the spec's own numbers in demo_belt.py
   (59.2 mm/s carry against a 60 mm/s belt), so the carry claim is real -- but
   there is no belt tension, tracking, or roller compliance in here.

3. THE WHEELS COLLIDE ON A 6 mm PROXY, 22 mm wide visually (F4).  A rigid
   cylinder's line contact over-predicts scrub, which put turn-in-place at 21%
   efficiency against a plausible ~70%.  Chassis.WHEEL_COLLISION_W is the knob;
   the turn numbers move with it, so treat turn efficiency as calibrated, not
   predicted.

4. THE LAB PLATE DOES NOT TOUCH THE ROBOT (F11/F13).  It is on its own collision
   bit: it interacts with game pieces but not the chassis.  Without that, the
   ball transfers cannot reverse up even a 1 mm step and no docking succeeds.
   That is a real unsolved hardware question, hidden here by a bitmask.

5. THE STEPPERS ARE VELOCITY SOURCES with step quantisation and optional step
   loss -- not a torque/speed curve, and StallGuard is a torque-saturation
   threshold, not the real chip's algorithm.

6. MASS IS LUMPED into one inertial block for the chassis; only the moving
   sub-assemblies carry their own.

7. NO VISION.  Agent A's mission does not need the camera, so none is simulated.
   Agent B's triage will.
""")
print("Everything else -- the conveyor, chute, collar, hold-down, escapement,")
print("feed plunger, ball transfers, wheels, fingers -- is rigid-body contact")
print("physics with real joints and force-limited actuators.")
