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

1. THE INTAKE IS MODELLED AS A STRAIGHT POWERED BOX reaching Za 3, not as a belt
   wrapping a nose bar over a 0.5 mm shim, and the sweeper fingers are static
   funnel walls rather than a powered stroke.  What the model DOES now prove is
   the requirement: sweeping the powered edge height gives 24/24 captures at
   Za -0.3, 3, 4 and 5, and 0/24 at Za 6 and above.  A one-millimetre cliff,
   because the edge has to get under the piece's half-thickness.  That rules out
   the spec's O16 roller (belt face ~18) and calls for a knife-edge nose bar.
   Bench-test the intake alone before committing: one belt, one nose bar, a
   sloped board, and a disc.

2. THE LABORATORY IS SOLID AND 6 mm, AND THE ROBOT REACHES OVER IT.  Both of the
   old cheats here are gone.  It used to sit on its own collision bit so the
   chassis ignored it, and it used to be 1 mm -- a thickness in which a 5 mm
   sample cannot sit, so the rules would not have accepted a single posting.  It
   is now 6 mm (a sample thick, F32), the chassis really collides with it, and
   the rear ball transfers sit at Xa 115 so the tail overhangs rather than trying
   to climb an edge a O20 ball cannot climb (F33/F39).

3. THE WHEELS COLLIDE ON A 6 mm PROXY, 22 mm wide visually (F4).  A rigid
   cylinder's line contact over-predicts scrub, which put turn-in-place at 21%
   efficiency against a plausible ~70%.  Chassis.WHEEL_COLLISION_W is the knob.
   THIS ONE CANNOT BE SETTLED IN SIMULATION.  Measure it: mark the floor, command
   a 360 deg turn in place, and see how far round the robot actually gets.
   Efficiency = achieved / commanded.  Measured values by proxy width --
   22 mm: 0.21,  12 mm: 0.44,  8 mm: 0.60,  6 mm: 0.70 -- so set the parameter
   from the bench number and re-run, rather than trusting 0.70.

4. THE MATCH CLOCK IS TIGHT.  120 s (rules g.1) against runs of 90-171 s: 7 of
   12 finish inside it and the rest post two or three samples by the buzzer.
   demo_pick_place.py prints both the final score and the score AT THE BUZZER,
   which is the one that counts.

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
