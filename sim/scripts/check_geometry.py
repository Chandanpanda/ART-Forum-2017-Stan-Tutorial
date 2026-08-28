"""Phase 0: geometry truth, no physics.  Re-derives every over-determined
quantity and fails loudly.  Run this first, and after any params.py edit."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rfgyc26 import params as P
from rfgyc26.mjcf import LAB_HOLE_Y

print("RFGYC'26 Rev C -- derived geometry\n" + "-"*54)
print("  belt incline (adopted, R1)      %8.2f deg" % P.Chassis.BELT_INCLINE)
print("  belt run / rise, Agent A        %8.1f / %.1f mm" % (P.BELT_RUN_A, P.BELT_RISE_A))
print("  belt top at the tail (derived)  %8.2f mm" % P.BELT_TOP_TAIL_A)
print("  discharge throw at 60 mm/s      %8.2f mm  (spec band 5-15)" % P.THROW_A)
print("  drive resolution                %8.4f mm/full-step" % P.Chassis.MM_PER_STEP)
print("  turn-in-place resolution        %8.4f deg/step" % P.DEG_PER_STEP)
print("  beam static tip-over            %8.3f deg" % P.BEAM_TIP_OVER)
print("  lab hole centre Y (R11)         %8.1f mm  [VERIFY]" % LAB_HOLE_Y)
print("-"*54)
bad = 0
for name, ok in P.CHECKS:
    print("  [%s] %s" % ("PASS" if ok else "FAIL", name))
    bad += not ok
print("-"*54)
print("%d of %d checks pass" % (len(P.CHECKS)-bad, len(P.CHECKS)))
sys.exit(1 if bad else 0)
