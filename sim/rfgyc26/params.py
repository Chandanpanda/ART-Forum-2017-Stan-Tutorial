"""Single source of truth for every dimension in the RFGYC'26 Rev C system.

All lengths are MILLIMETRES, all masses GRAMS, all angles DEGREES.
Convert with mm(), g() at the point of use -- MuJoCo works in metres/kilograms.

Every value traces to the Rev C source set. Values the spec tagged [VERIFY], and
the ten contradictions resolved in RFGYC26_simulation_plan.md, are marked below.
check_geometry.py re-derives the over-determined quantities and fails loudly.
"""
from math import tan, radians, cos, sin, sqrt

# ---------------------------------------------------------------- conversions
def mm(x):  return x / 1000.0          # mm -> m
def g(x):   return x / 1000.0          # g  -> kg


# ============================================================ FIELD (rulebook)
class Field:
    W               = 1143.0           # X, right
    H               = 1181.0           # Y, up-field
    WALL_T          = 20.0
    WALL_H          = 65.0

    QUARANTINE      = (0.0,   0.0,   280.0,  280.0)     # x0,y0,x1,y1
    LAB_PLATE       = (351.5, 360.0, 791.5,  510.0)
    LAB_PLATE_T     = 3.0
    LAB_HOLE_X      = (431.5, 571.5, 711.5)             # [VERIFY] pitch 140
    LAB_HOLE_Y      = 372.0
    LAB_HOLE_D      = 60.0

    HOSPITAL        = (471.5, 901.0, 671.5, 1181.0)
    PCC_L           = (0.0,   981.0, 200.0, 1181.0)
    PCC_R           = (943.0, 981.0, 1143.0,1181.0)
    RECOVERY        = (700.0, 190.0, 900.0,  270.0)

    DEPLOY_BOX      = (643.0, 0.0, 1123.0, 280.0)       # [VERIFY] X offset
    TAPE_W          = 20.0

    # R6: beam 1 south face butts beam 2's north end face at Y 250.
    BEAM1_CENTRE    = (137.5, 260.0)                    # 280x20 footprint
    BEAM2_CENTRE    = (290.0, 125.0)                    # 20x250 footprint


# =========================================================== GAME PIECES
class Piece:
    DISC_D, DISC_T, DISC_M          = 56.0, 5.0, 8.0
    KIT_X, KIT_Y, KIT_Z, KIT_M      = 25.0, 25.0, 20.0, 9.0
    CYL_D, CYL_H, CYL_M             = 20.0, 20.0, 5.0
    BEAM1_L, BEAM_W, BEAM_H, BEAM1_M = 280.0, 20.0, 60.0, 200.0
    BEAM2_L, BEAM2_M                 = 250.0, 180.0


# ==================================================== COMMON CHASSIS (Rev C)
class Chassis:
    BELT_W          = 116.0
    BELT_T          = 4.0              # collision proxy; real belt is 1.5
    BELT_INCLINE    = 11.0             # R1: incline pinned, tail height derived
    BELT_SPEED      = 60.0             # mm/s
    ROLLER_D        = 16.0
    BELT_TOP_NOSE   = 17.5             # = roller_r + roller_r + belt_t(1.5)

    TRACK           = 150.0
    WHEEL_D         = 60.0
    WHEEL_W         = 22.0
    # CALIBRATION: a rigid cylinder-plane LINE contact over-predicts turn-in-place
    # scrub badly (22 mm -> 21% turn efficiency).  A real rubber tyre's patch does
    # not resist yaw that way.  Collision proxy narrower than the visual wheel;
    # measured turn efficiency vs this value: 22mm=0.21 12mm=0.44 8mm=0.60 6mm=0.70.
    WHEEL_COLLISION_W = 6.0
    BALL_D          = 20.0
    GROUND_CLEAR    = 6.0
    CHAMFER         = 40.0

    STEPS_PER_REV   = 200 * 2          # NEMA17 200 x GT2 2:1
    MM_PER_STEP     = 3.14159265358979 * WHEEL_D / STEPS_PER_REV   # 0.471

    MU_PIECE        = 0.6              # wood on whiteboard
    MU_BALL         = 0.05


# ==================================================== AGENT A station schedule
class AgentA:
    L, W, H         = 285.0, 235.0, 175.0
    AXLE_X          = 142.5            # fore-aft centroid; local origin
    MASS            = 2600.0 - Piece.BEAM1_M - Piece.BEAM2_M   # chassis only

    SWEEP_PIVOT_X   = 278.0
    CAPTURE_OPEN    = 165.0
    CAPTURE_CLOSED  = 116.0

    SCOOP_FROM      = 275.0
    SCOOP_TO        = 210.0            # = belt nose
    SCOOP_ANGLE     = 15.0
    SCOOP_T         = 1.5              # collision proxy; real shim is 0.5

    BELT_NOSE_X     = 210.0
    BELT_TAIL_X     = 35.0
    GUIDE_FROM_W    = 116.0
    GUIDE_TO_W      = 62.0

    CHUTE_X         = 33.0             # axis
    CHUTE_D         = 58.0
    CHUTE_Z0        = 11.0             # gate tip -- 8 above the 3 lab plate
    CHUTE_Z1        = 55.0
    CHUTE_CAP       = 8

    POCKET_L_Y      = (211.0, 235.0)   # beam 1, open at the FRONT
    POCKET_R_Y      = (0.0,   24.0)    # beam 2, open at the REAR
    POCKET_H        = 60.0

    START_POSE      = (974.5, 140.0, 180.0)   # field x, y, heading deg


# ============================================================ derived + checks
BELT_RUN_A   = AgentA.BELT_NOSE_X - AgentA.BELT_TAIL_X            # 175
BELT_RISE_A  = BELT_RUN_A * tan(radians(Chassis.BELT_INCLINE))    # 34.02
BELT_TOP_TAIL_A = Chassis.BELT_TOP_NOSE + BELT_RISE_A             # 51.5  (R1)
SCOOP_RUN_A  = AgentA.SCOOP_FROM - AgentA.SCOOP_TO                # 65
SCOOP_RISE_A = SCOOP_RUN_A * tan(radians(AgentA.SCOOP_ANGLE))     # 17.4

# discharge throw: piece leaves the tail at BELT_SPEED, falls BELT_TOP_TAIL
DROP_TIME_A  = sqrt(2 * (BELT_TOP_TAIL_A / 1000.0) / 9.81)
THROW_A      = Chassis.BELT_SPEED * DROP_TIME_A                   # ~6.2 mm

BEAM_TIP_OVER = 18.434948822922             # atan(20/60), degrees
STEPS_PER_360 = 3.14159265358979 * Chassis.TRACK / Chassis.MM_PER_STEP
DEG_PER_STEP  = 360.0 / STEPS_PER_360

CHECKS = [
    ("belt rise closes on the nose height",
     abs(BELT_TOP_TAIL_A - (Chassis.BELT_TOP_NOSE + BELT_RISE_A)) < 1e-9),
    ("scoop rise meets the belt nose within 0.2",
     abs(SCOOP_RISE_A - Chassis.BELT_TOP_NOSE) < 0.2),
    ("A width budget: belt + wheels + pockets fits 235",
     Chassis.BELT_W + 2*Chassis.WHEEL_W + 20 + 2*24 <= AgentA.W),
    ("axle is on the fore-aft centroid",
     abs(AgentA.AXLE_X - AgentA.L/2) < 1e-9),
    ("chute bore clears a disc by >= 1 per side",
     (AgentA.CHUTE_D - Piece.DISC_D)/2 >= 1.0),
    ("chute holds at least the 3 discs carried",
     (AgentA.CHUTE_Z1 - AgentA.CHUTE_Z0) / Piece.DISC_T >= 3),
    ("lab hole clears a disc by 2 per side",
     abs((Field.LAB_HOLE_D - Piece.DISC_D)/2 - 2.0) < 1e-9),
    ("landing throw sits inside the 5-15 band",
     4.0 <= THROW_A <= 15.0),
    ("turn resolution is ~0.36 deg/step",
     abs(DEG_PER_STEP - 0.36) < 0.01),
    ("beam tip-over is 18.4 deg",
     abs(BEAM_TIP_OVER - 18.4) < 0.05),
    ("belt nose height is roller-determined",
     abs(Chassis.BELT_TOP_NOSE - (Chassis.ROLLER_D + 1.5)) < 1e-9),
]
