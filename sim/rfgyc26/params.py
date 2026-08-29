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
    # F11: Agent A cannot REVERSE up a square 3 mm plate edge -- the O20 ball
    # transfers stall on the step and the dock halts 80-320 mm short of the hole.
    # That is the spec's [VERIFY 10.2] question, answered in the negative for the
    # REVERSE direction (forwards over it is fine).  The real plate needs a ramped
    # or taped edge; modelled here as a 1 mm decal so the robot can dock at all.
    LAB_PLATE_T     = 1.0
    LAB_HOLE_X      = (431.5, 571.5, 711.5)             # [VERIFY] pitch 140
    LAB_HOLE_Y      = 372.0
    LAB_HOLE_D      = 60.0
    # F21.  The rulebook describes the laboratory as a WOODEN field element with
    # "3 marked slots of 60 mm" (rules 3.2) and says nothing about a lead-in.  The
    # robot spec's 45 deg chamfer that "absorbs +/-10 of robot position error" is
    # therefore an assumption about someone else's part, and the team cannot cut
    # it -- the laboratory is supplied.  Set this to 0.0 to see what the posting
    # tolerance really is with a plain bored slot.
    LAB_CHAMFER     = 4.0              # radial width of the assumed 45 deg lead-in

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
    # F20.  The fingers pivot at the NOSE and reach aft, so "open" (tips outboard
    # at y +/-82.5) makes the channel between them DIVERGE.  That looked like the
    # reason a sample more than ~45 mm off the sweep line was bulldozed into the
    # west wall rather than collected -- but raking them in only moved the
    # problem (see F22, which is the actual cause).  Samples captured over 8
    # randomised matches, 24 in all, with the guides starting at the mouth width:
    #     tips +/-82.5 (open)  24/24        tips +/-58 (raked)  21/24
    # Open it is.  The rake position is kept because the fingers still need a
    # second setting, and because raking is what the spec's stroke describes.
    FINGER_OPEN     = -5.4             # deg; tips at y +/-82.5, for release
    FINGER_RAKE     = 10.2             # deg; tips at y +/-58, the collecting set
    CAPTURE_OPEN    = 165.0
    CAPTURE_CLOSED  = 116.0

    SCOOP_FROM      = 275.0
    SCOOP_TO        = 210.0            # = belt nose
    SCOOP_ANGLE     = 15.0
    SCOOP_T         = 1.5              # collision proxy; real shim is 0.5

    BELT_NOSE_X     = 210.0
    # F17 (supersedes F7).  The tail roller must stand ONE DISC RADIUS forward of
    # the chute axis -- not over it.  A piece is supported until its trailing edge
    # clears the tail, so it is released when its CENTRE is one radius aft of the
    # tail: put the tail at CHUTE_X + 28 and the release point is the bore axis,
    # and the piece drops in flat and centred.  With the two aligned (F7) the
    # piece is already half over the bore when it starts to overhang; it tips in
    # edge-first and stands up inside the bore -- a coin jam, 0 for 3 in the rig.
    # Measured, one disc, otherwise identical model:  tail 36 -> never seats;
    # 50 -> never seats;  56 -> seats;  60 -> seats;  64 -> seats every time.
    BELT_TAIL_X     = 64.0             # = CHUTE_X + Piece.DISC_D/2
    # F22.  The guides used to start at the belt width (116) while the sweeper
    # mouth is 148 wide at the pivots, so a piece more than 30 mm off the sweep
    # line met the guide's LEADING EDGE square-on instead of its inner face and
    # was bulldozed along instead of funnelled in.  Start them at the mouth width
    # and the hand-off is continuous.
    GUIDE_FROM_W    = 148.0
    GUIDE_TO_W      = 62.0

    # Moved forward from the spec's Xa 33 so the bore is not jammed against the
    # chassis rear wall.  Even at 36 there is only 3 mm between the bore's rear
    # inner face and the wall, which is why the collar below has to do the
    # centring rather than the wall.
    CHUTE_X         = 36.0             # axis
    # F9: the spec's O58 bore for a O56 disc leaves 1 mm of radial clearance, which
    # a disc tipping off the belt tail cannot hit -- it overshoots the axis by ~5 mm
    # and jams on the rim.  Modelled at O66 (5 mm clearance).  The real machine needs
    # either a wider bore or positive placement; 1 mm is not achievable from a drop.
    CHUTE_D         = 66.0
    CHUTE_Z0        = 11.0             # gate tip -- 8 above the 3 lab plate
    # Front of the bore stops at Za 30: with the tail roller now at Xa 64 (F17) a
    # O16 roller occupies Za 30..46 right above the bore's front rim, so the tube
    # is a C -- open at the front above Za 30, where the roller and the belt wrap
    # close it.  Three 5 mm discs stack to Za 26, so 30 still contains them.
    CHUTE_Z1        = 30.0
    CHUTE_CAP       = 8

    # ESCAPEMENT (F19).  The base gate cannot be a plain shutter.  It has to slide
    # clear of a O56 disc to release one, and by then it has released the whole
    # column -- with a properly seated stack (which is new: they used to hang up
    # on each other) one stroke dropped all three and two landed in the same lab
    # hole.  So there are two blades: the SHELF carries the column, and a thin
    # RETAINER slides in at the joint between the bottom disc and the next one.
    #
    # They must be driven separately.  Built as one stepped slide -- the classic
    # coin escapement -- the shelf arrives from one side exactly as the retainer
    # leaves from the other, so the column is handed over in mid-drop and tips
    # out of the bore (measured: cycle 1 clean, cycle 2 threw the last disc onto
    # the collar at Za 32).  Two small servos, sequenced retainer-in, shelf-out,
    # shelf-in, retainer-out, and each transfer happens onto something already
    # in place.
    #
    # The retainer is a 1 mm KNIFE, not a plate: at 3 mm it straddled the joint,
    # caught the second disc's rim and dragged it out of the bore sideways.
    ESC_Y           = 74.0             # stroke; clears the bore and stays inboard
    ESC_T           = 1.5              # shelf half-thickness
    ESC_BLADE_T     = 0.5              # retainer half-thickness
    ESC_BLADE_Z     = 17.3             # underside 16.8, over a disc topping at 16
    ESC_BLADE_Y     = 40.0             # retainer half-length; covers the bore
    # ...and its leading edge is a rolled lip, not a square end.  Square, it met
    # the second disc's rim head-on and drove it sideways out of the bore (dy -30,
    # Za 32).  A round edge that starts BELOW the joint cams the column up
    # instead, and it is self-correcting: the bottom disc cannot go down because
    # the shelf is under it, so the lip always ends up between the two.
    ESC_LIP_R       = 0.7
    ESC_LIP_Z       = 17.1             # lip underside 16.4, below the Za 16 joint

    # RAISED REAR COLLAR (F15).  The bore's rear arc carries on upward past the
    # discharge plane as a collar.  It is the piece's aft backstop, and it is what
    # centres it: a O56 disc sliding aft at the discharge plane is halted the
    # instant its rim touches the collar's inner face at r 33, which leaves its
    # centre at dx = -(33 - 28) = -5 mm -- inside the bore, every time, with no
    # sensing.  Without it the disc sails over the Za 30 rim and jams against the
    # chassis rear wall 36 mm aft of the axis (measured: it stops there and stays,
    # flat, held by belt friction, and never drops).  Rear arc only -- the front
    # is where the belt feeds.  Its top must be higher above the discharge plane
    # than HOLD_GAP0, or a piece can climb over it instead of being stopped.
    CHUTE_Z2        = 58.0             # collar top (must beat HOLD_GAP0, below)
    CHUTE_COLLAR_T  = 2.0
    LEAD_R          = 36.5             # top chamfer outer radius (0 = none)
    LEAD_H          = 0.0
    LEAD_SKIP       = 100.0            # leave +/-100 deg of the FRONT arc open

    # HOLD-DOWN (F16).  Bore geometry alone cannot stop a coin jam -- a disc tilts
    # freely inside any bore wider than itself -- so the piece has to arrive flat.
    # A strip 8 mm above the belt over the last ~120 mm of its run lets a 5 mm
    # disc rise only 3 mm, capping the droop at about 6 deg, so it stays flat
    # until its trailing edge clears the tail and then falls flat.  It also makes
    # shingling impossible on the guided run: two 5 mm discs need 10 mm and the
    # channel is 8.  Ends AT the tail -- carried past it, the same strip blocks
    # the rotation the piece needs in order to drop, and nothing seats.
    # One folded strip of the same 1 mm sheet as the chassis: the cheapest part
    # on the robot, and the one that makes the magazine work.
    HOLD_FROM       = 64.0             # Xa, at the tail roller (= BELT_TAIL_X)
    HOLD_TO         = 180.0            # Xa, upstream end (flared)
    HOLD_GAP0       = 8.0              # clear height above the belt at the tail
    HOLD_GAP1       = 14.0             # clear height at the upstream end
    HOLD_W          = 62.0             # matches the single-file lane
    HOLD_T          = 1.0

    # POSITIVE FEED (F14/F18).  A plunger on the bore axis: parked its face is at
    # Za 84, and one stroke presses the top of the column down onto the stack.
    #
    # Two things had to change before this could work.  The first attempt parked a
    # foot 12 mm above the collar and was worse than nothing -- a piece tipping off
    # the tail reared up into it and wedged (0/3 with the ram, 3/3 without it, same
    # model).  Anything parked close over the mouth does that.  Parking at Za 84 is
    # 33 mm clear of the highest a piece ever reaches (Za 51, its top at the
    # discharge plane), so the drop path is empty.  Second, the old tail at Xa 36
    # forced the foot 14 mm off-axis to miss the belt, so it pressed on a perched
    # disc's rear half and levered it out of the bore; with the tail at Xa 64 (F17)
    # the axis itself is clear and the foot presses through the centre.
    #
    # 58 mm of stroke is more than a servo horn gives directly: drive it with a
    # 29 mm crank on a standard servo, or a rack.  Force is deliberately low --
    # at 14 N an earlier build pushed a disc THROUGH the bore wall.
    FEED_X          = 36.0             # on the chute axis
    FEED_D          = 46.0             # foot; forward edge Xa 59, clear of the tail
    FEED_Z_UP       = 87.0             # parked: foot centre (face at 84)
    FEED_STROKE     = 58.0             # face down to Za 26, the top of a full stack

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
