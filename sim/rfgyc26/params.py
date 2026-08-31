"""Single source of truth for every dimension in the RFGYC'26 Rev C system.

All lengths are MILLIMETRES, all masses GRAMS, all angles DEGREES.
Convert with mm(), g() at the point of use -- MuJoCo works in metres/kilograms.

Every value traces to the Rev C source set. Values the spec tagged [VERIFY], and
the ten contradictions resolved in RFGYC26_simulation_plan.md, are marked below.
check_geometry.py re-derives the over-determined quantities and fails loudly.
"""
from math import tan, radians, cos, sin, sqrt, atan, atan2, degrees

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
    # F32.  1.0 IS KNOWN TO BE WRONG and is kept only because the robot cannot
    # yet cope with the right value.  The rules require a sample to end up
    # "completely inside" a slot (2.1) and a sample is a 5 mm disc, so a real
    # laboratory is at least 5 mm thick.  At 1 mm the disc drops through until it
    # rests on the floor and stands 4 mm PROUD of the surface -- not inside
    # anything, and knocked straight out again as the robot departs, which is
    # visible in the viewer.  Set this to 6.0 to see the real problem: the
    # mission collapses to one sample or none.  That is the top open risk.
    LAB_PLATE_T     = 6.0
    # The plate collides with the ROBOT, not just with game pieces.
    #
    # F24 WAS WRONG AND IS WITHDRAWN.  It claimed Agent A climbs a square 3 mm
    # edge unaided, on the strength of a rig in which the robot was LAUNCHED at
    # initialisation and crossed the field in 2 s of a 90 mm/s drive -- it never
    # touched the laboratory at all.  With a rig that checks the robot has
    # actually settled before it drives (F33), the original F11 stands: the robot
    # gets onto a 1 mm edge at 62 N and is STOPPED DEAD by 3 mm and above.
    LAB_SOLID       = True
    # No approach ramp.  One was modelled on the plate's south edge, but the
    # laboratory is a supplied part (F21) so assuming a ramp on it was never
    # legitimate.  The robot climbs the square edge without it.
    LAB_EDGE_RAMP   = 0.0
    LAB_HOLE_X      = (431.5, 571.5, 711.5)             # [VERIFY] pitch 140
    LAB_HOLE_Y      = 372.0
    LAB_HOLE_D      = 60.0
    # F21.  The rulebook describes the laboratory as a WOODEN field element with
    # "3 marked slots of 60 mm" (rules 3.2) and says nothing about a lead-in.  The
    # robot spec's 45 deg chamfer that "absorbs +/-10 of robot position error" is
    # therefore an assumption about someone else's part, and the team cannot cut
    # it -- the laboratory is supplied.  Set this to 0.0 to see what the posting
    # tolerance really is with a plain bored slot.
    # A chamfer is cut INTO the plate, so on a 1 mm plate it is at most 1 mm deep
    # and lies entirely below the top surface.  It was modelled as a 4 mm cone
    # rising ABOVE the plate -- a raised collar, not a chamfer -- and once the
    # plate became solid that collar became a 4 mm obstacle: the rear ball
    # transfers drive into it and wedge, 6.7 N against a cone, wheels turning at
    # the commanded speed and the robot going nowhere.  That is the "shaking
    # without moving" seen in the viewer, and it is why a dock could burn its
    # whole 55 s guard.  Deleting the chamfer outright was worse (near-misses
    # then scatter across the plate); modelling it correctly, as a countersink
    # within the plate thickness, obstructs nothing.
    LAB_CHAMFER     = 1.0              # = plate thickness; a bevel cannot exceed it

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
    # The old BELT_NOSE_Z (a powered face 3 mm off the floor at the scoop tip)
    # is GONE -- it was the simulator's biggest stand-in (F1), a surface no
    # mechanism produced.  Its measured cliff (<=5 mm captures 24/24, >=6 mm
    # 0/24) and its replacement live in the AgentA intake section: a knife
    # shim owns engagement, a driven brush roller owns conveyance, and the
    # belt now starts where a belt physically can, at its nose roller.
    # F64: the belt reaches FORWARD to meet the brush roller's bite, on the
    # same plane (tail height and the whole magazine untouched).  With the old
    # 210 nose, the drum's bite ended ~30 mm short of the belt and off-centre
    # discs stalled in the unpowered stretch (measured: parked at Xa ~220,
    # creeping).  The nose therefore sits at 241 on a O10 roller -- smaller
    # than the O16 drive roller, which stays at the tail.
    NOSE_ROLLER_D   = 10.0
    BELT_TOP_NOSE   = 11.5             # = nose_roller + belt(1.5)

    TRACK           = 150.0
    WHEEL_D         = 60.0
    WHEEL_W         = 22.0
    # CALIBRATION: a rigid cylinder-plane LINE contact over-predicts turn-in-place
    # scrub badly (22 mm -> 21% turn efficiency).  A real rubber tyre's patch does
    # not resist yaw that way.  Collision proxy narrower than the visual wheel;
    # measured turn efficiency vs this value: 22mm=0.21 12mm=0.44 8mm=0.60 6mm=0.70.
    WHEEL_COLLISION_W = 6.0
    BALL_D          = 20.0
    # F31.  The rear ball transfers moved forward, from Xa 40 to Xa 90, so the
    # tail OVERHANGS the laboratory instead of driving onto it.  Agent A cannot
    # climb the laboratory: measured on a rig that checks the robot has actually
    # settled before it drives (the earlier one had not, and reported the
    # opposite), it gets onto a 1 mm edge at 62 N and is STOPPED DEAD by 3 mm and
    # anything above.  The rulebook gives the laboratory no thickness at all, so
    # a design that has to drive onto it is a design that scores nothing if the
    # supplied part is 6 mm ply.
    # Geometry: docking puts the chute on a slot 40 mm inside the plate edge, so
    # the rear balls stay off the plate when they sit more than 40 mm forward of
    # the chute.  Xa 90 leaves 14 mm of margin.
    # DEFAULT IS STILL THE ORIGINAL 40.  Moving the balls forward is the right
    # answer to the geometry, but on its own it regressed the mission to one
    # sample everywhere -- it changes the support polygon the docking controller
    # was tuned against.  It is parameterised, documented and NOT adopted: the
    # move has to come with a docking controller that expects it.
    # 115, not 90.  At 90 the arithmetic said 14 mm of margin but the MEASURED
    # ball surface sits at Y 360 at the dock point -- exactly the laboratory edge
    # -- so a dock either grazes it or catches on it, and catching stalls the
    # robot 5.5 mm short with the terminal controller commanding 11 mm/s into a
    # 13 N contact forever.  That is the bimodal dock: 15 s or never.  Every
    # 1 mm here is 1 mm of margin.
    BALL_REAR_X     = 115.0            # overhangs the lab instead of climbing it
    BALL_FRONT_X    = 245.0
    BALL_Y          = 80.0
    GROUND_CLEAR    = 6.0
    # A step up in the shell aft of TAIL_X, for when the tail overhangs the
    # laboratory (F31).  DEFAULT EQUALS GROUND_CLEAR, i.e. no step: like the ball
    # move it is the right shape and it also regressed the mission on its own, so
    # it is parameterised and documented rather than adopted.  Set 14.0 with
    # BALL_REAR_X 90 to model the overhanging tail.
    TAIL_CLEAR      = 14.0
    # F35.  A skid: a bent lip sloping from just off the floor up to TAIL_CLEAR,
    # ahead of each rear ball transfer.  The overhanging tail (F31) keeps the
    # balls off the laboratory on a PERPENDICULAR dock, but there is no legal
    # place to turn south of the laboratory (the corridor is 360 wide and the
    # swept circle needs 370, F10), so every approach after the first is
    # DIAGONAL -- and on a diagonal the balls cross the edge anyway and jam at
    # 20 N.  A O20 ball cannot climb 6 mm: cos a = (10-6)/10 needs 2.25x the
    # supported weight as push.  A 20 deg wedge needs 0.36x, which the drive has.
    # SKID_LEN 0: the skid does not work and is not built.  A ramp only climbs a
    # step if its leading edge is ABOVE the step; below it, the two vertical
    # faces meet and jam, which is what a 6 mm laboratory does to it.  Kept as a
    # parameter so the finding is not re-discovered.
    SKID_LEN        = 0.0              # Xa run, ahead of the rear balls
    SKID_Z0         = 2.0              # leading edge, just clear of the floor
    SKID_W          = 26.0             # width, straddling each ball
    TAIL_X          = 90.0             # everything aft of this is stepped up
    CHAMFER         = 40.0

    STEPS_PER_REV   = 200 * 2          # NEMA17 200 x GT2 2:1
    MM_PER_STEP     = 3.14159265358979 * WHEEL_D / STEPS_PER_REV   # 0.471

    # Dwell at the end of a sweep pass, letting the belt clear before the robot
    # manoeuvres.  THE MATCH IS 120 s (rules g.1) and two passes here spend 44 of
    # them standing still, so it is the first place to look for time.  Measured
    # capture over 8 matches: 22 s -> 24/24, 14 s -> 23/24, 10 s -> 23/24,
    # 7 s -> 19/24.  Buying time here costs samples.
    SWEEP_DWELL     = 10.0             # s
    MU_PIECE        = 0.6              # wood on whiteboard
    MU_BALL         = 0.05


# ============================================================ PERCEPTION (F69)
class Vision:
    """Two Pi cameras on one rigid plate at the tail: the robot's primary sensor.

    WHY THIS REPLACES THE REFLECTANCE PROBES.  F68 built the laboratory dock
    around two TCRT-class rangefinders on the posting head -- a 1.5 s servo
    sweep to find the slot's rims, a plate-edge crossing to datum the range, and
    a dead-reckoned run at the end.  It worked, and every millimetre of it was
    bought by MOVING something, because a reflectance sensor answers one
    question: "is there anything 14 mm below this one point".  A camera answers
    the question the robot is actually asking -- where is the slot, in three
    dimensions, relative to me -- from 140 mm away, for all three slots at once.

    WHY A SELF-BUILT PAIR AND NOT A DEPTH CAMERA.  A packaged stereo module
    (OAK-D and the like) hands over a dense depth map, and pays for it with a
    MINIMUM RANGE: min_depth = f*B/disparity_levels, which for a 75 mm baseline
    at 800P is 697 mm.  The corridor south of the laboratory is 360 mm wide
    (F10), so the robot can never stand more than ~290 mm from its dock -- a
    sensor with a 700 mm floor is blind for the entire approach.  Dropping to
    400P and extended disparity brings it to 173 mm, and the whole design then
    has to be arranged around a blind cone it cannot avoid.

    None of that applies to a pair of cameras we mount ourselves, because the
    robot does not need a dense depth map.  It needs ONE feature: the centre of
    a O60 circle.  Found independently in each image and triangulated, that has
    no disparity search and therefore no minimum range at all.

    AND THE SENSOR IS NOT THE LIMIT -- the bracket is.  Worked through at
    1280x720 and 66 deg (f = 986 px), for a slot 200 mm away spanning 296 px:

        ellipse centre, few hundred boundary pixels    0.05 px -> 0.01 mm
        triangulated range, 130 mm baseline                    -> 0.02 mm
        range from the KNOWN O60 diameter, one camera          -> 0.03 mm
        ---------------------------------------------------------------
        camera-to-BORE calibration, a bracket on this robot        1.0 mm
        mast flex at 200 mm/s over a wooden floor, 0.2 deg        0.70 mm
        ellipse-centre perspective bias at 45 deg, uncorrected    0.46 mm
        ...the same, corrected by conic back-projection           0.10 mm

    So: spend the effort on a STIFF, SHORT mast, on both cameras being on ONE
    plate rather than two brackets, and on calibrating to the bore rather than
    only to each other.  Do not spend it on megapixels.  The two axes that
    matter -- lateral, which the trim slide has to spend, and range, which
    odometry already half knows -- are both dominated by the mount.

    TWO INDEPENDENT RANGES, DELIBERATELY.  Triangulation and known-diameter are
    different physics on the same image pair.  A competition robot that posts a
    sample into the wrong place loses 18 points and does it confidently; two
    measurements that must agree is what turns a wrong answer into a refusal.
    """
    # ---- the parts, and what is ours to choose ----
    W, H            = 1280, 720        # processing resolution, not sensor size
    # 102, the WIDE Camera Module 3, and the reason is coverage rather than
    # anything to do with precision.  Measured over the whole band the corridor
    # allows (d 140 down to 60), counting slots whose FULL RIM is in frame:
    #
    #     HFOV  toe        d=140 130 120 110 100 ...
    #      66     8          3   3   2   2   0        loses the slot it is docking
    #      66    12          2   2   2   2   0        toe alone does not fix it
    #      78     8          3   3   3   3   0
    #     102     8          3   3   3   3   0
    #
    # At 66 deg the NEAREST slot's rim leaves the frame first -- so the robot
    # keeps the two it is not docking and loses the one it is.  78 fixes it and
    # 102 does no better, but 78 is not a stock Pi lens and 102 is.  The cost is
    # focal length (518 px against 986), which takes the lateral noise from
    # 0.016 mm to 0.031 -- against a calibration bias of 1.0.  It is free.
    HFOV            = 102.0            # deg, Camera Module 3 Wide
    BASELINE        = 130.0            # mm; ours to pick, chassis allows 200
    TOE             = 8.0              # deg of convergence, so both see the dock
    FPS             = 20.0             # an ellipse fit does not need 60

    # ---- what the model treats as noise -----------------------------------
    FEAT_SIGMA_PX   = 0.08             # ellipse centre, per frame
    DIAM_SIGMA_PX   = 0.12             # apparent diameter, per frame
    # ---- ...and what it treats as BIAS, because that is what it is ---------
    # Drawn once per match, not per frame.  A calibration error is the same
    # every frame; a model that redraws it averages it away and flatters itself.
    EXT_SIGMA       = 1.0              # mm, camera-plate-to-bore
    EXT_ANG_SIGMA   = 0.25             # deg
    FLEX_DEG        = 0.20             # mast flex at FLEX_REF speed
    FLEX_REF        = 200.0            # mm/s
    # DETECTION BIAS, and it is the one that nearly went unmodelled.
    #
    # A slot is a O60 hole in a 6 mm plate, seen from 45 deg.  Its FLOOR is a
    # crescent, not a circle -- the near wall hides part of it -- so the
    # centroid of "everything below plate level" sits off the slot's axis, by an
    # amount that depends on where the camera is relative to that slot.
    # Measured against actual depth renders, with the modelled calibration bias
    # zeroed so only geometry is left:
    #
    #     slot straight behind the camera      -2.37 mm, at every range
    #     slot 140 mm to the far side          +1.56 mm, at every range
    #
    # Constant with range and signed by the viewing offset: a bias, and one
    # comparable to the entire rest of the budget.  THE FIX IS IN THE PIPELINE,
    # NOT THE OPTICS: fit the TOP RIM, which is a real circle and is never
    # occluded from above, instead of taking the centroid of what you can see
    # through it.  DET_BIAS is what a rim fit leaves behind; the 2.4 mm above is
    # what a naive blob centroid costs, and it is written down here so nobody
    # re-discovers it on the field.
    DET_BIAS        = 0.4              # mm, 1-sigma, per slot per match
    Z_MAX           = 1200.0           # beyond this a O60 slot is under 50 px

    # ---- MOUNT: on the deck, at the tail, looking back and down -----------
    #
    # Rearward, and that is considered rather than convenient.  The one task on
    # the board needing millimetre vision is the laboratory, and the laboratory
    # is a REVERSE dock -- the posting head is at Xa 36, behind the axle,
    # because that is where the belt discharges (F17).  A forward camera would
    # have to map the slots, turn 180 deg, and then dock on odometry through
    # the least repeatable manoeuvre the robot performs.  Looking back, the slot
    # is in frame for the whole approach, and the camera also sees every scoring
    # action the robot has just taken.
    #
    # The binding constraint is NOT range or precision -- it is the robot's own
    # tail.  The camera has to look over a shell 94 mm tall to see a slot 100 mm
    # behind it, and the model's own ray casts put the slot's near rim 0.3 mm
    # inside that shell.  Height, pitch and how far aft the plate sits are all
    # traded against the START ENVELOPE (285 long, spec 4.2), which is why the
    # plate is flush with the tail rather than past it.
    CAM_X           = 24.0             # Xa of the plate's centre
    CAM_Z           = 158.0            # Za  [VERIFY the height rule]
    CAM_PITCH       = 45.0             # deg below horizontal, facing -Xa
    CAM_MASS        = 12.0             # g each, Camera Module 3
    MAST_MASS       = 60.0             # g, mast + plate

    @classmethod
    def body_half_x(cls):
        """How far the tilted housing reaches along Xa from the plate centre."""
        from math import cos, sin, radians
        a = radians(cls.CAM_PITCH)
        return 14.0*cos(a) + 18.75*sin(a)

    @classmethod
    def f_px(cls):
        from math import tan, radians
        return (cls.W/2.0) / tan(radians(cls.HFOV/2.0))

    @classmethod
    def sigma_lat(cls, z_mm):
        """Lateral, from the feature's image position."""
        return z_mm * cls.FEAT_SIGMA_PX / cls.f_px()

    @classmethod
    def sigma_z_stereo(cls, z_mm):
        return z_mm*z_mm * cls.FEAT_SIGMA_PX * 1.414 / (cls.f_px() * cls.BASELINE)

    @classmethod
    def sigma_z_mono(cls, z_mm, diam_mm):
        """Range from a circle of KNOWN diameter: sigma_Z/Z = sigma_d/d_px."""
        d_px = diam_mm * cls.f_px() / z_mm
        return z_mm * cls.DIAM_SIGMA_PX / d_px


# ==================================================== MISSION 2 (healthcare)
class M2:
    """Kits and triaged patients -- rules 2.2/3.2.

    130 of the 250 points on the board live here, and Agent A currently scores
    NONE of it: three kit destination zones left empty is -30 before anything
    else is counted (Senior table).  So the honest Mission-1-only score is +90,
    not +120.
    """
    N_KITS          = 10
    KIT_PLAN        = {"HOSP": 6, "PCC_L": 2, "PCC_R": 2}   # the +20 distribution
    # WHICH KIT LIVES IN WHICH HOPPER.  A destination is a hopper, not a
    # sorting problem: the kits may start on the robot (rules g.1), so they are
    # loaded before the match already grouped, and the only verb the mission
    # needs is "open hopper X".  Three MG90S flaps, and no colour, no
    # singulation and no belt anywhere in it.
    #
    # THE DISTRIBUTION BONUS IS WHY ALL THREE MUST BE VISITED.  Measured on the
    # referee: nothing at all is -30, hospital alone is -2, hospital plus one
    # PCC is +14, and the full 6/2/2 is +50.  The last two kits, in the far
    # corner, are worth 36 points on their own.
    KIT_GROUPS      = {"HOSP": tuple(range(6)), "PCC_L": (6, 7), "PCC_R": (8, 9)}
    # Hopper discharge points in the ROBOT frame (x from the axle, y, and the
    # height the kit leaves at).
    #
    # THE KIT MUST LAND OUTSIDE THE ROBOT'S OWN FOOTPRINT, and that is not a
    # detail.  With the shafts at Ya 78 the kits fell between the wheels (Ya 75,
    # 22 wide) and the beam pockets -- inside the track.  The robot then could
    # not leave: reversing ran the rear over them and driving forward was into a
    # wall, because a corner zone is a 200 mm box and the swept radius is 185, so
    # there is nowhere to pivot either.  Measured: 2 kits of 10 delivered, -30.
    #
    # 140 puts the discharge 22 mm PROUD of the chassis, which the same MG90S
    # that opens the hopper provides as a short swing-out lip -- so the start
    # envelope is still 235 (spec 4.2) and the kit still clears the wheels at 86
    # and the beam pockets at 117.5.  With the kits outside the track the robot
    # can simply reverse out of the corner and pivot where there is room.
    #
    # PCC_R's is on the OTHER flank, and that is geometry too: the zone runs
    # x 943-1143 against a wall at 1143, so the robot's centre can never be east
    # of 1025, and a left-hand lip on a north-facing robot drops the kit at 785 --
    # nowhere near the zone.  A right-hand one drops it at 1040.
    # ...and FORWARD of the axle, which is about the north wall rather than the
    # kits.  A drop station has to be deep enough into a zone for the kit to
    # land inside it, and with a purely sideways discharge that put the robot at
    # y 1030 -- where, at any angle at all, its own corner is through the north
    # wall (the half-diagonal is 185, and at 153 deg the y extent alone is 169).
    # Measured: it ground along the wall at 4 mm/s for eight seconds, and the
    # stall detector never fired because it never quite stopped.
    #
    # Discharging 100 mm AHEAD of the axle lets the robot stand 100 mm further
    # south for the same landing point.  It also means reversing away from a
    # drop moves the wheels away from the kit rather than over it.
    # THE THREE HOPPERS MUST NOT OVERLAP IN SPACE, which sounds obvious and was
    # not: HOSP's upper layer sat directly on top of a PCC_L kit, and a kit
    # resting on a still-welded kit is resting on a shelf.  It rode the whole
    # loop and fell in the wrong zone -- 10 kits delivered, +30 instead of +50,
    # and the log cheerfully said "dropped 6".
    #
    # Layout on each flank, x measured forward from the axle:
    #   HOSP   3 wide x 2 high at x 0, 28, 56      (left)
    #   PCC_L  2 wide         at x 84, 112         (left)
    #   PCC_R  2 wide         at x 84, 112         (right)
    HOPPER          = {"HOSP":  (   0.0,  140.0),
                       "PCC_L": (  84.0,  140.0),
                       "PCC_R": (  84.0, -140.0)}
    HOPPER_WIDE     = {"HOSP": 3, "PCC_L": 2, "PCC_R": 2}   # kits per layer
    HOPPER_PITCH    = 28.0
    HOPPER_Z        = 88.0             # above a carried beam (tops at Za 72)
    # Twelve cylinders, four of each colour, six per side area.
    N_CYL           = 12
    COLOURS         = ("red", "yellow", "green")
    CYL_DEST        = {"red": "HOSP", "green": "RECOVERY"}  # yellow splits PCC_L/R
    # WHERE THE CYLINDERS START IS AN ASSUMPTION.  The rulebook says only "two
    # side areas of the field... six cylinders on each side" and gives no
    # coordinates (rules 2.2).  Modelled as mid-field left and right strips.
    # This is a [VERIFY] of the same class as F21 and F32 -- both of which cost
    # us a rebuild -- so it is one parameter, not a number sprinkled through the
    # code.  Sweep time is the second-tightest constraint in the route plan.
    SIDE_L          = (40.0,  480.0, 200.0, 820.0)          # x0,y0,x1,y1
    SIDE_R          = (943.0, 480.0, 1103.0, 820.0)
    STICKER_D       = 40.0


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

    # LOCKED INTAKE (F64): sprung knife shim + driven brush roller.  This
    # replaces the powered-surface stand-in that fed the belt from the scoop
    # tip (the old A_belt ran to Xa 275 with its face at 3 mm and surfacevel
    # doing the conveying -- a mechanism that never existed).  The two jobs are
    # now split between two mechanisms that do:
    #   ENGAGEMENT -- a 0.5 spring-steel knife shim, hinged over the belt nose,
    #   tip floating on the field.  The edge it presents to a disc is its own
    #   thickness, so the old 1 mm height spec (capture 24/24 at <=5 mm, 0/24
    #   at >=6 -- the powered edge had to get under the disc's 2.5 mm
    #   half-thickness) is met by construction, not by build accuracy.
    #   A servo also LIFTS it 25 deg for transit: a floor-pressed knife would
    #   bulldoze already-placed discs on every non-collecting leg.
    #   CONVEYANCE -- a brush roller over the shim: silicone fingers on a
    #   D20 hub, N20-driven, on a sprung swing arm.  It presses ~1.3 N down on
    #   the piece and drives it aft at the finger tips, which is what carries
    #   the disc across the dead zone where F1 measured a passive piece
    #   stranding (19 mm engaged, 46 mm short of the belt, becalmed).
    SHIM_TIP_X      = 272.0            # knife tip; = GUIDE_TOP_X, 13 in from the shell
    SHIM_HINGE_X    = 242.0            # hinge axis, hanging over the belt nose
    # Plate TOP at the hinge = belt nose top, FLUSH within 0.2 (the old scoop
    # check, relearned the hard way: built 0.75 proud "so the disc drops onto
    # the belt", the aft edge became a fulcrum -- the disc see-sawed on it,
    # the belt got 0.03 N of normal force and lost the tug-of-war to the
    # passive shim.  Measured: creeping at 2 mm/s, forever).
    SHIM_HINGE_Z    = 11.25            # plate mid-plane at the hinge
    SHIM_T          = 0.5              # spring steel; the presented step
    SHIM_MU         = 0.30            # knife face against a piece [VERIFY]
    SHIM_DROOP      = 6.0              # deg past nominal the servo presses (preload)
    SHIM_LIFT       = 35.0             # deg tip-up for transit (short shim needs more)
    ROLL_AXIS_X     = 262.0            # drum axis at rest
    ROLL_AXIS_Z     = 25.5             # rest height: collision drum bottom at 3.5
    ROLL_DRUM_R     = 22.0             # collision proxy: hub 10 + fingers at working squish
    ROLL_TIP_R      = 25.0             # = FING_HUB_R + FING_TUBE_L, asserted below
    ROLL_W          = 128.0            # across the mouth, inside the guide walls
    ROLL_RPM        = 300.0            # N20 nominal; sweep 200-500
    # The DRIVER'S ceiling, not a modelling constant.  It was hard-coded at
    # 60 rad/s (573 rpm) in the actuator, so every sweep above that was
    # silently clipped and 600, 900 and 1200 rpm all ran at 573 -- which is
    # exactly why those rows came out identical.  The N20 has an encoder and
    # a PWM driver, so rpm is a CONTROL we can set per piece if it helps.
    ROLL_RPM_MAX    = 1200.0           # driver ceiling
    ROLL_TORQUE     = 0.098            # N*m -- 1.0 kg*cm N20 stall, the honest cap
    ARM_SPRUNG      = True            # False = axis bolted down, tubes do it all
    ARM_PIVOT_X     = 205.0            # swing-arm pivot
    ARM_PIVOT_Z     = 60.0
    ARM_PRELOAD_N   = 1.3              # downforce at the drum, arm on its stop

    # THE FINGERS ARE SILICONE TUBES, IN ROWS (F72).
    #
    # F64 built the brush roller as a RIGID DRUM inscribed at "working finger
    # squish" (r 22 against a 25 mm tip circle), with the fingers drawn as
    # contype=0 decoration.  That is not the machine: the machine is a hub with
    # short lengths of silicone tube standing out of it, and the whole reason
    # such a roller works on objects of unknown size is that each tube bends
    # independently.  A rigid drum has one contact height; a brush has as many
    # as it has fingers.
    #
    # It matters most for the small pieces.  A O56 sample spans the drum and
    # meets every row at once, so a rigid cylinder is a fair stand-in for it --
    # which is why F64's numbers held.  A O20 patient meets ONE OR TWO ROWS,
    # and what those rows do is the whole mechanism.  Modelling it as a plate
    # across the full width answers for a piece that does not exist.
    #
    # STIFFNESS IS DERIVED FROM THE PART, not fitted.  A O6/O3 silicone tube
    # 15 mm long, E ~ 3 MPa (shore A 40): I = 5.96e-11 m^4, tip stiffness
    # 159 N/m, so 0.036 N*m/rad about the root and 0.38 g of finger.  0.3 N at
    # the tip bends it 7 degrees; 1 N bends it 24.
    # [VERIFY: press one finger against a gram scale and read the deflection.]
    # MEASURED, ON THE BENCH RIG (F74).  Robot held still, piece placed on the
    # ramp 12 mm forward of the axis, five lateral offsets, honest torque for
    # each speed (an N20 three times faster has a third of the stall torque):
    #
    #   roller                rpm   N*m     samples  upright  on side
    #   rigid drum O44        300  0.098      5/5      5/5      0/5
    #   rigid drum O44        600  0.049      5/5      0/5      0/5
    #   rigid drum O44        900  0.033      5/5      0/5      0/5
    #   tubes O20 hub, 20 mm  300  0.098      5/5      4/5      5/5
    #
    # For a O56 x 5 SAMPLE the two are identical -- every roller, every speed,
    # 5 of 5.  There is nothing to win there.  For a O20 PATIENT the tubes win
    # outright, and the drum's only route to matching (spin faster) costs the
    # torque it needs to do it.  A drum on its sprung arm does take an UPRIGHT
    # patient -- it rides up over it and puts it on the belt in 0.2 s -- but it
    # never once took one lying down.
    #
    # Speed is a control, not a constant (N20 + encoder + PWM driver), and the
    # tubes have a NARROW optimum: 150 rpm gives 0 of 10 on patients, 300 gives
    # 9 of 10, 600 and 900 give 0 of 10.  Fast tubes strike and reject.
    ROLL_FINGERS    = True             # False restores the F64 rigid drum
    FING_ROWS       = 5                # across the drum's width
    FING_AROUND     = 8                # per row, staggered half a pitch row to row
    FING_DAMP       = 0.0006
    FING_MASS       = 0.00038          # kg
    FING_TUBE_OD    = 6.0              # the ordered tube
    FING_TUBE_ID    = 3.0
    FING_TUBE_R     = FING_TUBE_OD/2.0
    FING_TUBE_L     = 15.0             # free length; 20 measured better [VERIFY]
    FING_E          = 3.0e6            # Pa, shore A 40 silicone  [VERIFY]
    FING_HUB_R      = 10.0

    # UPPER ROLLER (F71) -- the patients.
    #
    # The F64 intake is geometrically specific to a 5 mm flat disc, and the
    # simulator said so the first time it was asked: 10 patients of 10 were
    # BULLDOZED along the floor, upright and on their sides, at every offset
    # across the mouth.  A O20 x 20 cylinder standing on the floor is 20 tall,
    # and the drum's axis is at Za 25.5 -- so the fingers meet it near its TOP,
    # 5.5 mm below the axis, and push it over forwards instead of sweeping it
    # back onto the knife.  A 5 mm disc is only ever touched on its top face,
    # which is why the same roller carries it perfectly.
    #
    # An OVER-UNDER pair fixes it, and is what game-piece intakes do: a second
    # roller above, so a tall piece is caught between the two and driven up the
    # knife by both.  Sized so a disc passes underneath UNTOUCHED -- the lower
    # gap is 12 mm against a 5 mm disc -- so nothing about the validated disc
    # intake changes.  One more N20 and one more silicone tube.
    # OFF.  Measured over twelve matches with it fitted: mean +113.2 against
    # +125.1 without, and the laboratory dock fell from 30 samples of 36 to 26.
    # It buys nothing until the patient question is settled on a bench, and it
    # is not free -- a second roller in the mouth disturbs the disc path that
    # F64 spent a rebuild getting right.  Parameterised so it costs one line to
    # try again once there is a reason to.
    UP_ROLL         = False
    # Swept: Xa 252 / Za 49 takes a patient lying down 5 times in 5; every
    # other position in the grid managed 0-3.
    UP_AXIS_X       = 252.0
    UP_AXIS_Z       = 49.0             # fingers reach to Za 24 = 12 above a disc
    UP_DRUM_R       = 22.0
    UP_TIP_R        = 25.0
    UP_W            = 128.0
    UP_RPM          = 300.0
    UP_TORQUE       = 0.098

    # TRIP BAR (F71) -- a bent rod across the mouth, and the cheapest part on
    # the robot after the hold-down strip.
    #
    # A patient LYING DOWN goes in 5 times out of 5 once the upper roller is
    # sited (swept).  Standing up it goes in 1 time in 5, and the reason is not
    # the gap: a O20 x 20 cylinder has a 1:1 aspect ratio and its centre of mass
    # 10 mm up, so anything that pushes it above 10 rotates it instead of moving
    # it, and the knife's 0.2 mm edge WEDGES UNDER the base and stands it on the
    # ramp, where it topples sideways out of the mouth (measured: they finish
    # 80 mm off the lane).
    #
    # So do not try to convey it standing -- lay it down first.  A blunt bar at
    # Za 6-9 meets the cylinder BELOW its centre of mass and trips it over
    # towards the robot; a 5 mm sample passes underneath with a millimetre to
    # spare, so the disc intake is untouched and the bar needs no actuator.
    # OFF, for the same reason: it was built to lay a standing patient down and
    # it does not (0 of 5, and it made the lying case worse).  Kept because the
    # REASONING is worth preserving even though the part is not.
    TRIP_BAR        = False
    TRIP_X          = 285.0            # ahead of the knife tip at 272
    TRIP_Z          = 7.5              # centre; O3 rod spans 6.0-9.0
    TRIP_R          = 1.5
    TRIP_W          = 130.0            # across the mouth
    # The knife is NARROWER than the mouth on purpose.  The sweeper fingers
    # occupy z 2..27 wherever they swing (raked, their structure reaches in to
    # |y| ~56), and a lifted knife rotates its corners up through that plane --
    # a metal-on-metal clash in the real build, found when the first sim lift
    # jammed at 1.5 mm.  At 108 the knife clears every finger position at every
    # knife angle; a disc funnelled in from wider out crosses the knife's 0.5
    # side edge, which is nothing (F63's side-climb was a ~9 mm scoop edge).
    SHIM_W          = 108.0

    BELT_NOSE_X     = 241.0            # F64: forward to the drum bite; was 210
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
    # 148 IS A MEASURED CEILING, NOT A DRAFT (F63).  Widening it looks free and
    # is not: at 162 the mean over twelve matches falls +110 -> +69, at 175 into
    # a steeper taper further still.  The mouth is already 32 mm wider than the
    # PAN it feeds -- the scoop is belt-width, 116 -- so a funnelled piece
    # crosses bare floor and has to climb the scoop's side edge, and widening
    # only delivers more pieces into that wedge.  Widening the pan to match is
    # blocked by the front ball transfers at Xa 245, y +/-80.
    GUIDE_FROM_W    = 148.0
    GUIDE_TO_W      = 62.0
    # Promoted out of mjcf.py so the two ceilings above can be asserted.
    GUIDE_END_X     = 195.0            # Xa where the taper finishes
    GUIDE_TOP_X     = 272.0            # Xa where it starts

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
    # F70 seat taper: the bore closes to SEAT_R over the last SEAT_H, so a piece
    # settles centred instead of anywhere in 5 mm of play.  1.75 mm of radial
    # clearance at the seat, which a piece reaches by sliding down a 24 deg
    # funnel rather than by being dropped into it.
    SEAT_R          = 29.75
    SEAT_H          = 8.0
    # Front of the bore stops at Za 30: with the tail roller now at Xa 64 (F17) a
    # O16 roller occupies Za 30..46 right above the bore's front rim, so the tube
    # is a C -- open at the front above Za 30, where the roller and the belt wrap
    # close it.  Three 5 mm discs stack to Za 26, so 30 still contains them.
    CHUTE_Z1        = 30.0
    CHUTE_CAP       = 8

    # ================================================ POSTING-HEAD TRIM (F68)
    #
    # THE LABORATORY COSTS 53 s OF A 120 s MATCH.  Measured over six seeds:
    # 18.6 + 17.4 + 17.0 s for three slots worth 50 points, in a match that also
    # has to find Mission 2's 130.  The escapement is not what costs it -- 34 of
    # 36 samples go in -- the DOCK is.
    #
    # Why the dock is expensive: the mechanism's capture window is +/-4 mm
    # across and +/-3 mm along (measured, 23 of 81 on a +/-8 mm grid under F67's
    # honest slot test), because a O56 disc into a O60 slot has 2.0 mm of radial
    # clearance and nothing can widen that.  A differential-drive robot can only
    # correct LATERAL error by turning, and turning swings a chute 106 mm behind
    # the axle, so align_reverse hunts: three reverse passes, 11-17 s.  Measured
    # separately, ONE turn-and-reverse takes 3.2 s and leaves the lateral error
    # it started with, essentially untouched.
    #
    # So the last few millimetres are given to the HEAD instead of the chassis.
    # One MG90S through a 15 mm Scotch yoke slides the whole posting head -- bore,
    # collar, escapement, feed and the slot probes -- across the robot.  The
    # chassis then only has to arrive within the slide's reach, in ONE pass.
    #
    # This also survives the change that is coming.  Today the route navigates on
    # ground truth; the real robot will navigate on VIO, which does not deliver
    # 4 mm.  A trim slide closed on the probes needs the slot's position RELATIVE
    # TO THE HEAD, which is a local measurement, not a world coordinate.
    #
    # WHAT DOES NOT FIT, AND WHY THE DROP-THROUGH BORE SURVIVES.  A magazine that
    # discharges SIDEWAYS -- push the bottom disc out through a window, as a
    # vending machine does -- is one actuator instead of two and needs no bore.
    # The space is not there.  A shuttle carrying a disc from the bore to a
    # separate discharge hole needs >= 70 mm between the two O66 centres AND a
    # plate long enough to still roof the bore at full stroke: 132 mm of plate
    # sweeping 202 mm, inside a 191 mm shell.  Pushing the disc clear OUT of the
    # robot does fit (177 mm swept) but puts the drop point 118 mm off the
    # centreline, which stands the drive wheels exactly on the laboratory's 6 mm
    # edge -- the one contact F35 says jams the robot.  A rotary carrier is
    # worse: 180 deg of swing about an axis 35 mm off the bore needs an outer
    # radius of 67, so it reaches Ya 102, through the beam pocket wall.
    # 25, which is what the two-leaf escapement leaves room for: the shelf leaf
    # reaches Ya 95 at full trim and the retainer 84, against a beam pocket wall
    # at 95.5.  The single sliding shelf this replaced could not have afforded
    # eight.  It is the shelf that binds, so widening the trim any further means
    # shortening the leaves, and the leaves are already at their minimum -- a
    # shelf that retracts short of the bore wall PERCHES the disc instead of
    # dropping it (0 of 81 on the capture grid, measured).
    TRIM_Y          = 25.0             # +/- lateral stroke of the posting head
    TRIM_RATE       = 60.0             # mm/s; MG90S at 0.1 s/60 deg on a 22 mm crank

    # THE REFLECTANCE PROBES ARE GONE (F69).  Rev C specified a TCRT array to
    # find the slots, F30 sited it, F68 moved it onto the head and made it work
    # -- and then the OAK-D made all of it pointless.  A reflectance sensor
    # answers "is there something 14 mm below this one point", so the robot has
    # to move to ask again, and the dock was built around asking enough times: a
    # 1.5 s servo sweep, a plate-edge crossing to datum the range, and a blind
    # run at the end.  A stereo camera answers the question directly, from
    # 140 mm away, for all three slots at once.  Four sensors, their wiring and
    # 9 mm of the escapement's lateral extent all leave the build.
    #
    # The one rangefinder that STAYS is a_mag, looking down the magazine bore --
    # a 60 mm tube with a stack in it is exactly what a short-range reflectance
    # sensor is good at, and no camera can see inside it.

    # ESCAPEMENT (F19, rebuilt as a two-leaf iris by F68).
    #
    # The base gate cannot be a plain shutter: it has to slide clear of a O56
    # disc to release one, and by then it has released the whole column -- one
    # stroke dropped all three and two landed in the same lab hole.  So there
    # are two stages: a SHELF carrying the column, and a thin RETAINER entering
    # the joint between the bottom disc and the next one.  They must be driven
    # separately (built as one stepped slide the column is handed over in
    # mid-drop and tips out of the bore), and the retainer must be a 1 mm KNIFE
    # with a rolled leading lip -- at 3 mm it straddled the joint and dragged
    # the second disc sideways out of the bore, and square-ended it drove the
    # second disc out the same way.  All of that stands.
    #
    # WHAT CHANGED IS THE SIDEWAYS EXTENT, AND IT HAD TO.
    #
    # A single sliding shelf must roof a O66 bore closed AND be entirely out
    # from under a O56 disc open, so its stroke is at least 66 mm and it reaches
    # 33 + 66 = 99 mm off the head before any trim is added.  Measured: cut it
    # to 56 and the disc is not released, it PERCHES on the 7 mm of shelf still
    # under its far edge and the returning shelf then scoops it back up (0 of 81
    # on the capture grid, against 23 before).  There is no room in a 191 mm
    # shell for that stroke plus F68's trim slide.
    #
    # Two leaves closing from opposite sides halve it.  Each leaf roofs half the
    # bore and retracts 37 mm, so the far edge is 70 instead of 111 -- and the
    # release is SYMMETRIC, which fixes something the single shelf never did:
    # the returning shelf used to sweep a released disc sideways (10.7 mm,
    # measured, F41) and could pinch one against the slot's countersink hard
    # enough to bolt the robot to the laboratory (F55).  Two leaves meeting at
    # the axis apply no net side force at all.
    #
    # On the real machine one pinion drives both racks, which is how the model
    # couples them: one actuator on a fixed tendon, not two servos.
    ESC_OVER        = 2.0              # each leaf crosses the axis by this much
    ESC_GAP         = 2.0              # ...and clears the bore by this when open
    ESC_Y           = CHUTE_D/2 + ESC_OVER + ESC_GAP        # leaf stroke, 37
    ESC_HALF        = (CHUTE_D/2 + ESC_OVER)/2.0            # leaf half-length, 17.5
    ESC_XHALF       = CHUTE_D/2 + 4.0  # shelf leaf half-width along Xa; roofs the bore
    # The RETAINER is narrower along Xa than the shelf, and has to be: it only
    # takes the column on a chord, so it needs to reach under a O56 disc, not
    # roof a O66 bore -- and at the shelf's 37 it spans Xa -1..73, straight
    # under the plate-edge probes.  Blocked, they read a constant 2.2 mm, the
    # edge is never seen, and every dock spends its whole 14 s guard before
    # falling back to dead reckoning.  Measured: the dock went from 17 s to 28.
    ESC_BLADE_XHALF = Piece.DISC_D/2 + 2.0                  # 30
    ESC_T           = 1.5              # leaf half-thickness
    ESC_BLADE_T     = 0.5              # retainer half-thickness
    ESC_BLADE_Z     = 17.3             # underside 16.8, over a disc topping at 16
    # The retainer is two leaves as well, for the same reason and with the same
    # rolled lip.  Each reaches ESC_BLADE_Y past the axis, so the column is held
    # on a 2*ESC_BLADE_Y chord with its centre supported.  F47 cut this from 80
    # to 62 mm to keep it out of the beam pocket; two leaves take it to 46 and
    # leave room for the trim slide on top.
    ESC_BLADE_OVER  = 1.0
    ESC_BLADE_Y     = 23.0             # each leaf reaches this far past the axis
    ESC_BLADE_HALF  = (ESC_BLADE_Y + ESC_BLADE_OVER)/2.0
    # Back to the geometric minimum now that F69 has retired the slot probes it
    # used to have to park outboard of.  That is 9 mm of lateral extent given
    # back, and the trim slide takes it.
    ESC_BLADE_PARK  = CHUTE_D/2 + ESC_BLADE_OVER + ESC_GAP  # 36: its own stroke
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
    # F64: the strip reaches forward to the drum (which roofs the shim itself),
    # because the brush feeds pieces at several hundred mm/s where the old
    # surface crawled at 60 -- and a fast piece arriving pitched or shingled
    # sailed OVER a strip that started at 180 and parked on top of it
    # (measured: two tandem discs at z 42, on the strip's back).  The same one
    # straight plate, longer: a launch-catcher at the front, the F16
    # anti-shingle taper at the tail.
    HOLD_TO         = 235.0            # Xa, upstream end (5 clear of the drum)
    HOLD_GAP0       = 8.0              # clear height above the belt at the tail
    HOLD_GAP1       = 18.0             # clear height at the upstream end
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


    # ---------------------------------------------------------------- BEAMS
    # F44.  The pockets are NOT boxes with an outboard wall.  They cannot be:
    # the quarantine corner is 280x280, the two beams enclose 280x250 of it,
    # and Agent A is 285 long -- so once either beam is down, the robot no
    # longer fits beside the other one.  Every arrangement with an outboard
    # wall left the chassis 5-25 mm inside a placed beam.  What does fit is a
    # channel whose OUTBOARD face is the beam itself: an inner wall at
    # |Ya-117.5| = 95.5, an open bottom, an end stop that does the pushing,
    # and two hooks that drop outboard of the beam to retain it and lift clear
    # to release.  The loaded envelope is then exactly 235 (spec 4.2's number),
    # and the robot's own structure stops 20 mm inboard of the beam it carries.
    POCKET_IN_Y     = 95.5             # inner wall, |local y|; 1.5 thick
    POCKET_Y        = 107.5            # beam centreline, |local y|; 2*(107.5+10)=235
    POCKET_H        = 60.0             # = Piece.BEAM_H
    HOOK_Y          = 118.25           # |local y| of the retaining lip
    HOOK_T          = 0.75             # half-thickness (a bent tab)
    HOOK_W          = 14.0             # half-length along Xa
    # A short UPSTAND on the shelf's outboard edge, not a tall hook.  A tall
    # hook cannot be got out of the way by lowering -- it still stands beside
    # the placed beam and tows it -- and swinging it outboard while the shelf is
    # still under the piece drags the beam 12 mm sideways and yaws it 9 deg
    # (both measured).  A 20 mm lip retains a beam whose centre of mass is 30 mm
    # up, and one straight 34 mm drop takes the whole cradle below the field.
    HOOK_H          = 10.0             # half-height: lip is CARRY_Z .. +20
    HOOK_LIFT       = 64.0             # clears a 60 beam
    # Hook stations, local x.  Chosen so that on the approach to beam 2 no
    # hook ever crosses beam 1's footprint (F45): beam 1 lies across
    # x 0..280, y 250..270, and the pocket-L hooks pass 275 mm out from the
    # axle on that approach.
    # F53.  The pocket-L lips live on the FORWARD half of the beam, and that is
    # not arbitrary.  Beam 1's run-in and its shuffle both take the robot along
    # the north side of beam 2, and a lip at Xa 42 swings out to X 262-328 as
    # the chassis crabs -- straight through beam 2's east half, which it shoved
    # 30 mm and yawed 14 deg.  At Xa 172/242 the lips never enter beam 2's
    # X band at all, at any heading the approach uses.
    HOOK1_X         = ( 30.0, 100.0)  # pocket L, carries beam 1
    HOOK2_X         = ( -60.0,   40.0) # pocket R, carries beam 2
    # F46.  The beams are CARRIED CLEAR OF THE FLOOR and set down to place.
    # Rev C says they "stand on the field and are never lifted", and that is
    # what makes the task unsolvable: a beam dragging on the floor cannot cross
    # the 6 mm laboratory, so with a beam aboard the robot may not turn
    # anywhere the swept circle touches the plate -- and the swept circle is
    # 189 mm, the corridor south of the plate is 360 mm, and 2 x 189 > 360.
    # Combined with the walls that leaves NO legal pivot anywhere in the
    # southern half of the field with a beam aboard, which is fatal: the two
    # stations are 90 deg apart.
    # Carrying them 12 mm up costs one servo per pocket -- the same servo the
    # hooks already need -- and it buys back every pivot F36 measured, because
    # 12 mm clears the 6 mm plate exactly as the chassis does.
    CARRY_Z         = 12.0
    # ...and the shelves drop CLEAR OF THE FLOOR, not level with it.  Level
    # with it the placed beam still rests half on the shelf, and the robot
    # drove away dragging it 6.7 mm off station by friction alone.
    CRADLE_DROP     = 2*HOOK_H + 2.0   # takes the lip below the field too
    CRADLE_OUT      = 0.0              # F48: retracting outboard drags the beam
    # End stops.  Beam 1 is pushed WEST by a stop behind it (the robot drives
    # forward at heading 180); beam 2 is pushed SOUTH by a stop ahead of it
    # (the robot reverses at heading 90).  Both release by backing the stop
    # off the beam -- a few tens of millimetres, not a beam length, because
    # nothing else in the pocket touches the piece once the hooks are up.
    # Flush with the chassis ends: the beam's leading face and the shell's
    # own end reach the wall together, so nothing has to be trimmed off the
    # intake to give the beam a lead.  (At -147.5 the shell stalled 11.5 mm
    # early and beam 1 finished 12 mm off the wall.)
    STOP1_X         = -137.5           # pocket L, REAR: pushes beam 1 west
    STOP2_X         =  107.5           # pocket R, FRONT: pushes beam 2 south
    STOP_T          = 2.0
    # ...and the stop is exactly as wide as the beam, not wider.  1.5 mm of
    # overhang is 1.5 mm of the stop sitting south of beam 1's own south face,
    # which is the face that has to touch beam 2.
    STOP_W          = Piece.BEAM_W / 2.0
    # F52.  The end stops must start ABOVE the laboratory.  At Za 2 they hang
    # 4 mm into a 6 mm plate, and the robot ploughs it on every reverse dock:
    # hole 1 went from 1.4 mm in 15 s to 57 mm in 87 s, and it took a
    # three-way load comparison to see that the beams were not the cause --
    # their pocket hardware was.  A carried beam sits at Za 12..72, so a stop
    # spanning 16..52 engages it and still clears the plate.
    STOP_Z0         = 16.0
    STOP_H          = 18.0             # half-height: 16..52
    # Where each beam rides.  Beam 1's rear face sits on STOP1_X, beam 2's
    # forward face on STOP2_X.
    BEAM1_LOCAL     = (STOP1_X + Piece.BEAM1_L/2.0,  POCKET_Y)   # pocket L
    BEAM2_LOCAL     = (STOP2_X - Piece.BEAM2_L/2.0, -POCKET_Y)   # pocket R

    # Beam stations: (axle_x, axle_y, heading) at the moment of the wall stall,
    # and how far to back the stop off afterwards.  Both are derived, not
    # guessed -- check_geometry.py re-derives them from the pocket numbers.
    #   beam 1: heading 180, drives WEST, rear stop pushes, releases eastward
    #   beam 2: heading  90, reverses SOUTH, front stop pushes, releases north
    #
    # F49.  BEAM 2 GOES FIRST, AND THE ROBOT WORKS BEAM 1 FROM THE NORTH SIDE.
    # This is the only assignment of the four that closes, and the reason is
    # turning room.  With a beam aboard the swept radius is 185 mm; a placed
    # beam is an obstacle 60 mm tall; and the robot cannot pivot within 185 mm
    # of one.  Beam 2's station sits 118 mm from beam 1's east end, so if beam 1
    # goes down first, beam 2's station can never be entered -- there is no
    # heading change available anywhere on its approach lane.  Reversed, and
    # with beam 1 approached from the NORTH (body on Y 270-465, clear of beam 2
    # by 20 mm), every leg is a straight run and every pivot has its 185 mm.
    # The station Y is 369.5 rather than 367.5 because the cradle lip has to
    # clear beam 2's north end face as it goes past; that leaves a 2 mm gap at
    # the T-joint, which is inside the referee tolerance and is the "<= 1 mm at
    # the contact lever" question spec section 10 already flags.
    BEAM1_STATION   = (142.5, 369.5, 180.0)
    BEAM2_STATION   = (177.5, 142.5,  90.0)   # 5 inboard, per spec 7
    BEAM_BACKOFF    = 45.0

    START_POSE      = (974.5, 140.0, 180.0)   # field x, y, heading deg


# ============================================================ derived + checks
BELT_RUN_A   = AgentA.BELT_NOSE_X - AgentA.BELT_TAIL_X            # 146
BELT_RISE_A  = BELT_RUN_A * tan(radians(Chassis.BELT_INCLINE))    # 28.4
BELT_TOP_TAIL_A = Chassis.BELT_TOP_NOSE + BELT_RISE_A             # 45.9  (R1)
# Knife shim plane, from the hinge (top face over the belt nose) down to the
# floating tip.  The angle is DERIVED -- the two anchor points own it.
SHIM_RUN_A   = AgentA.SHIM_TIP_X - AgentA.SHIM_HINGE_X            # 30
SHIM_DROP_A  = (AgentA.SHIM_HINGE_Z + AgentA.SHIM_T/2) - AgentA.SHIM_T   # 11.0
SHIM_ANGLE_A = degrees(atan2(SHIM_DROP_A, SHIM_RUN_A))            # 20.1 deg
SHIM_TOP_HINGE_A = AgentA.SHIM_HINGE_Z + AgentA.SHIM_T/2          # 11.5
SHIM_SLOPE_A = SHIM_DROP_A / SHIM_RUN_A                           # 0.367


def ramp_z(x):
    """Top surface of the knife shim at Xa=x, shim on its stop (mm).

    This is the surface a piece actually rides.  It is NOT the floor: the whole
    intake happens on a 20 deg ramp that starts 0.5 mm off the field at Xa 272
    and reaches the belt nose height, 11.5, at Xa 242.
    """
    x = min(max(x, AgentA.SHIM_HINGE_X), AgentA.SHIM_TIP_X)
    return SHIM_TOP_HINGE_A - SHIM_SLOPE_A * (x - AgentA.SHIM_HINGE_X)


def brush_reach(z):
    """Forward-most Xa at which the finger-tip circle gets down to height z.

    Beyond this the brush cannot touch a piece of that height at all; behind it
    the brush is on the piece.  The knife's LEAD is SHIM_TIP_X minus this, and
    the lead is what decides whether the knife wedges under a piece or the
    brush shoves it away first.
    """
    dz = AgentA.ROLL_AXIS_Z - z
    if abs(dz) >= AgentA.ROLL_TIP_R:
        return None
    return AgentA.ROLL_AXIS_X + sqrt(AgentA.ROLL_TIP_R**2 - dz*dz)


# How far the finger tips clear the ramp directly under the axis.  Negative
# means the tubes are folded flat on the plate and have no reach left for a
# piece -- and, measured, they then get under the plate and jack the knife up.
ROLL_GAP_A   = (AgentA.ROLL_AXIS_Z - AgentA.ROLL_TIP_R
                - ramp_z(AgentA.ROLL_AXIS_X))

# The sweep leg's speed, which the belt has to beat (route.sweep_line).
SWEEP_SPEED_A = 140.0

# How far the finger tips must press into a piece to drive it, not graze it.
ROLL_BITE_A  = 2.0

# THE TUBE'S STIFFNESS IS THE TUBE'S, not a number that was tuned until the
# simulation behaved.  A cantilever of second moment I bent about its root:
#     I     = pi/64 * (OD^4 - ID^4)
#     k_rad = 3EI / L        N*m per radian at the root
# so changing the tube length changes the stiffness, as it does on the bench.
FING_I_A  = 3.14159265358979/64.0 * ((AgentA.FING_TUBE_OD/1000.0)**4
                                     - (AgentA.FING_TUBE_ID/1000.0)**4)
FING_K_A  = 3.0 * AgentA.FING_E * FING_I_A / (AgentA.FING_TUBE_L/1000.0)
AgentA.FING_K = FING_K_A


def _arm_lift(deg=40.0):
    """How far the sprung arm can carry the drum up before it hits its stop."""
    if not AgentA.ARM_SPRUNG:
        return 0.0
    dx = AgentA.ROLL_AXIS_X - AgentA.ARM_PIVOT_X
    dz = AgentA.ROLL_AXIS_Z - AgentA.ARM_PIVOT_Z
    t = radians(-deg)
    return (-dx*sin(t) + dz*cos(t)) - dz


# THE HEIGHT DIFFERENCE THE ROLLER HAS TO ABSORB.  Its rigid part must clear a
# 20 mm patient while its working surface presses down on a 5 mm sample -- a
# 17 mm swing, with the bite.  Only two things can give it: tube length (the
# tips reach below the hub) or arm travel (the whole drum lifts).  A BOLTED
# rigid roller has neither, and no height exists that takes both pieces.
# Measured: on its sprung arm a rigid drum with 3 mm of static clearance rides
# up over an upright patient and puts it on the belt in 0.2 s.
ROLL_ACCOM_A = ((AgentA.ROLL_TIP_R - AgentA.FING_HUB_R if AgentA.ROLL_FINGERS else 0.0)
                + _arm_lift())

# discharge throw: piece leaves the tail at BELT_SPEED, falls BELT_TOP_TAIL
DROP_TIME_A  = sqrt(2 * (BELT_TOP_TAIL_A / 1000.0) / 9.81)
THROW_A      = Chassis.BELT_SPEED * DROP_TIME_A                   # ~6.2 mm

BEAM_TIP_OVER = 18.434948822922             # atan(20/60), degrees
STEPS_PER_360 = 3.14159265358979 * Chassis.TRACK / Chassis.MM_PER_STEP
DEG_PER_STEP  = 360.0 / STEPS_PER_360

def _guide_ball_clearance():
    """Gap between the converging guide wall and the front ball transfer, mm.

    The wall runs from (GUIDE_TOP_X, GUIDE_FROM_W/2) to (GUIDE_END_X, GUIDE_TO_W/2)
    and the ball is a sphere sitting on the floor at (BALL_FRONT_X, +/-BALL_Y).
    Widening the mouth walks the wall straight across it -- which is why F63's
    widening experiments had to end the taper further forward to fit at all.
    """
    ax, ay = AgentA.GUIDE_END_X, AgentA.GUIDE_TO_W/2.0
    bx, by = AgentA.GUIDE_TOP_X, AgentA.GUIDE_FROM_W/2.0
    px, py = Chassis.BALL_FRONT_X, Chassis.BALL_Y
    dx, dy = bx - ax, by - ay
    t = ((px-ax)*dx + (py-ay)*dy) / max(dx*dx + dy*dy, 1e-9)
    t = min(1.0, max(0.0, t))
    d = ((px - (ax + t*dx))**2 + (py - (ay + t*dy))**2) ** 0.5
    return d - Chassis.BALL_D/2.0 - 0.75          # ball radius, half wall thickness


CHECKS = [
    ("belt rise closes on the nose height",
     abs(BELT_TOP_TAIL_A - (Chassis.BELT_TOP_NOSE + BELT_RISE_A)) < 1e-9),
    ("shim top meets the belt nose FLUSH within 0.2 (a proud edge is a fulcrum)",
     abs((AgentA.SHIM_HINGE_Z + AgentA.SHIM_T/2) - Chassis.BELT_TOP_NOSE) <= 0.2),
    ("a disc bridges the roller bite to the belt -- no dead zone by construction",
     AgentA.ROLL_AXIS_X - AgentA.SHIM_HINGE_X + 5.0 <= Piece.DISC_D),
    ("drum bite starts below a disc's top face",
     AgentA.ROLL_AXIS_Z - AgentA.ROLL_DRUM_R <= Piece.DISC_T - 1.0),
    ("lifted knife tip clears a placed disc by >= 10",
     AgentA.SHIM_HINGE_Z + AgentA.SHIM_T/2
     + SHIM_RUN_A * sin(radians(AgentA.SHIM_LIFT - SHIM_ANGLE_A))
     >= Piece.DISC_T + 10.0),
    ("roller fits inside the guide walls at its station",
     AgentA.ROLL_W/2.0 <= AgentA.GUIDE_TO_W/2.0
     + (AgentA.ROLL_AXIS_X - AgentA.GUIDE_END_X)
     / (AgentA.GUIDE_TOP_X - AgentA.GUIDE_END_X)
     * (AgentA.GUIDE_FROM_W - AgentA.GUIDE_TO_W)/2.0 - 2.0),
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
    ("belt nose height is nose-roller-determined",
     abs(Chassis.BELT_TOP_NOSE - (Chassis.NOSE_ROLLER_D + 1.5)) < 1e-9),
    # The one that actually decides whether the intake works.  Measured cliff
    # (F1 era): an edge at or below Za 5 captures 24/24, at Za 6 it captures
    # 0/24, because it has to get under the piece's half-thickness instead of
    # shunting it.  The knife shim satisfies it BY CONSTRUCTION -- the step it
    # presents is its own 0.5 thickness plus a O1 tip lip.
    ("knife edge stands under the piece's half-thickness",
     AgentA.SHIM_T + 1.0 <= Piece.DISC_T / 2.0),
    ("a bare roller nose could NOT satisfy that -- the shim is load-bearing",
     Chassis.ROLLER_D + 1.5 > Piece.DISC_T / 2.0 + 2.5),
    # --- beams ---------------------------------------------------------
    ("beam 1 lands on the west wall, spanning X 0-280",
     abs((AgentA.BEAM1_STATION[0] - (AgentA.STOP1_X + Piece.BEAM1_L))) < 1e-9),
    ("beam 1 lands within 5 mm of the Y 250-270 band (R6)",
     abs((AgentA.BEAM1_STATION[1] - AgentA.POCKET_Y) - Field.BEAM1_CENTRE[1]) <= 5.0),
    ("beam 2 lands on the south wall, spanning Y 0-250",
     abs(AgentA.BEAM2_STATION[1] - (Piece.BEAM2_L - AgentA.STOP2_X)) < 1e-9),
    ("the cradle lip clears beam 2's end face on beam 1's run-in",
     AgentA.BEAM1_STATION[1] - AgentA.HOOK_Y - AgentA.HOOK_T >= Piece.BEAM2_L),
    # Spec 7's "beam 2 biased 5 inboard", and it is not cosmetic: beam 1 ends
    # at X 280 and beam 2's nominal X 280-300 touches it along a line of zero
    # width, so any placement error at all opens the T-joint.  5 mm of overlap
    # is what makes the closure bonus robust.
    ("beam 2 lands 5 mm inboard of X 280-300, overlapping beam 1's end",
     abs((AgentA.BEAM2_STATION[0] + AgentA.POCKET_Y) - 285.0) < 1e-9),
    ("the carried beam clears the laboratory it has to cross",
     AgentA.CARRY_Z > Field.LAB_PLATE_T),
    ("the loaded envelope is still 235 wide (spec 4.2)",
     abs(2*(AgentA.POCKET_Y + Piece.BEAM_W/2) - AgentA.W) < 1e-9),
    ("the hooks lift clear of a 60 beam",
     AgentA.HOOK_LIFT >= Piece.BEAM_H + 2.0),
    # Both ceilings on the guide mouth, because both are one edit away (F63).
    ("the guide mouth clears the front ball transfer (F63)",
     _guide_ball_clearance() >= 3.0),
    ("...and stays inboard of the beam pocket wall",
     AgentA.GUIDE_FROM_W/2.0 + 0.75 <= AgentA.POCKET_IN_Y - 1.5),
    # F47 and F68 together: the retainer has to clear the beam pocket AT FULL
    # TRIM, not just at trim zero.  Every lateral extent on the head is now
    # spent twice, and this is the one that sits at a carried beam's height.
    ("the parked escapement retainer stays out of the beam pocket at full trim (F47/F68)",
     AgentA.ESC_BLADE_PARK + AgentA.ESC_BLADE_Y + AgentA.TRIM_Y <= AgentA.POCKET_IN_Y),
    # The shelf is allowed past the pocket wall because it passes UNDER a
    # carried beam -- but only just, so assert the clearance that makes it legal.
    ("the escapement shelf passes under a carried beam",
     AgentA.CHUTE_Z0 - 1.5 + AgentA.ESC_T < AgentA.CARRY_Z),
    ("...and at full trim the shelf leaf is still clear of a carried beam's foot",
     AgentA.CHUTE_D/2.0 + AgentA.ESC_Y + AgentA.TRIM_Y <= AgentA.POCKET_IN_Y),
    ("the closed shelf leaves roof the whole bore, overlapping at the axis",
     AgentA.ESC_OVER > 0 and 2*AgentA.ESC_HALF - AgentA.ESC_OVER >= AgentA.CHUTE_D/2.0),
    # THE ONE THAT WAS MISSING, AND IT COST A ZERO.  A shelf that retracts short
    # of the bore wall leaves the disc PERCHED on its edge instead of dropping
    # it, and the returning shelf then scoops it back up: 0 of 81 on the capture
    # grid, measured, when ESC_Y was cut from 74 to 56 on the old single shelf.
    ("the open shelf leaves clear the bore entirely, so nothing perches",
     AgentA.ESC_Y - AgentA.ESC_OVER >= AgentA.CHUTE_D/2.0),
    ("the closed retainer leaves take the column on a chord through its centre",
     AgentA.ESC_BLADE_Y >= Piece.DISC_D/4.0),
    ("the parked retainer is clear of the bore",
     AgentA.ESC_BLADE_PARK - AgentA.ESC_BLADE_OVER >= AgentA.CHUTE_D/2.0),
    # ------------------------------------------------------------------
    # F73: THE INTAKE'S LAWS, WRITTEN DOWN.
    #
    # Six model faults in a row were found the expensive way -- by sweeping
    # parameters and wondering why nothing moved.  Each one was a statement
    # about the machine that nothing in the file was checking.  These are those
    # statements.  A red line here is a design decision that has not been made
    # yet, not a nuisance.
    #
    # GETTING UNDER A PIECE.  A piece resting on the field is held by nothing
    # but mu_field * W.  Resolve a wedge of angle a and face friction mu_w and
    # the piece stays put -- letting the knife slide under -- only while
    #       a + atan(mu_w) <= atan(mu_field)
    # Mass, size and the roller all cancel: it is two angles.  Measured, a
    # sample climbs 54 mm having slipped 1.7 mm when this holds, and is
    # bulldozed from first touch when it does not.
    ("the knife can get UNDER a piece rather than shoving it",
     SHIM_ANGLE_A + degrees(atan(AgentA.SHIM_MU))
     <= degrees(atan(Chassis.MU_PIECE))),
    # ...AND THEN HOLD IT.  A piece SHORTER than the ramp ends up entirely on
    # the ramp with no floor contact left, and slides straight back down unless
    #       atan(mu_w) >= a
    # A O56 sample never gets here -- it bridges ramp to floor the whole way --
    # but a O20 patient does, twenty millimetres in.  The two conditions bound
    # the ramp at a <= min(phi_w, 31 - phi_w), best at mu_w 0.28.
    ("...and hold one shorter than the ramp instead of dropping it back out",
     degrees(atan(AgentA.SHIM_MU)) >= SHIM_ANGLE_A or
     Piece.CYL_H > SHIM_RUN_A),
    # THE THROAT.  Everything the intake swallows goes down one channel.  The
    # hold-down clamp was sized for a 5 mm disc (F16) and is 8 mm at the tail:
    # a 20 mm patient cannot enter the belt run at all, over 170 mm of it, and
    # no roller position, speed or finger count can change that.
    ("the throat passes the TALLEST piece, not just the flattest",
     min(AgentA.HOLD_GAP0, AgentA.HOLD_GAP1) >= Piece.CYL_H + 2.0),
    # THE BELT MUST OUTRUN THE ROBOT.  A piece is stationary in the WORLD and
    # every surface of the intake is moving forward at the sweep speed.  A belt
    # running aft at v gives its surface a world speed of (sweep - v): positive
    # and the belt carries the piece out of the mouth however well the knife
    # got under it.
    ("the belt beats the sweep, so its surface goes AFT in the world",
     Chassis.BELT_SPEED > SWEEP_SPEED_A),
    # THE BRUSH MUST NOT LEAN ON ITS OWN RAMP.  The arm's only down stop is its
    # nominal height, so whatever the brush rests on carries it: set with
    # interference against the knife, eight tubes at ~1 N lift the arm clear
    # and the brush rides the RAMP, never the piece.
    ("the brush clears the knife ramp under its own axis",
     ROLL_GAP_A >= 0.0),
    ("...and its finger tips stay inside the shell",
     AgentA.ROLL_AXIS_X + AgentA.ROLL_TIP_R <= AgentA.L),
    # ONE ROLLER, TWO PIECE HEIGHTS -- and it decides rigid vs brush outright.
    # The roller's RIGID part must pass over the tallest piece:
    #       Za - r_hub  >=  surface + CYL_H
    # and its tips must reach down onto the flattest one:
    #       Za - r_tip  <=  surface + DISC_T - bite
    # Subtract, and the surface and the height cancel:
    #       r_tip - r_hub  >=  CYL_H - DISC_T + bite
    # A rigid drum has r_tip == r_hub, so the left side is ZERO and no height
    # exists that takes both.  That is not a tuning result, it is arithmetic:
    # a single hard roller can never admit a 5 mm sample and a 20 mm patient.
    # A brush can, and the tube length is exactly how much margin it has.
    # Built: 25 - 10 = 15 against 20 - 5 + 2 = 17.  Three millimetres short.
    ("a single roller can take BOTH piece heights",
     ROLL_ACCOM_A >= Piece.CYL_H - Piece.DISC_T + ROLL_BITE_A),
    ("the roller's commanded speed is inside the driver's range",
     AgentA.ROLL_RPM <= AgentA.ROLL_RPM_MAX),
    ("the finger tip circle is the hub plus the tube, not a separate number",
     (not AgentA.ROLL_FINGERS) or
     abs(AgentA.ROLL_TIP_R - (AgentA.FING_HUB_R + AgentA.FING_TUBE_L)) < 1e-9),
    # F69: the camera is the primary sensor, so its geometry gets assertions.
    ("a O60 slot is hundreds of pixels across where the robot measures it",
     Field.LAB_HOLE_D * Vision.f_px() / 250.0 >= 100.0),
    ("...so the SENSOR is nowhere near the limit -- the bracket is",
     Vision.sigma_lat(250.0) < 0.1 and Vision.EXT_SIGMA >= 10*Vision.sigma_lat(250.0)),
    ("both cameras fit across the chassis, inboard of the beam pockets",
     Vision.BASELINE/2.0 + 15.0 <= AgentA.POCKET_IN_Y),
    ("the camera housing stays inside the robot's own length",
     Vision.CAM_X - Vision.body_half_x() >= -0.5),
    ("...and under the design envelope's roof",
     Vision.CAM_Z + 14.0*sin(radians(Vision.CAM_PITCH))
                  + 18.75*cos(radians(Vision.CAM_PITCH)) <= AgentA.H + 35.0),
    # The trim scan only works if BOTH probes can find their rim inside the
    # slide's stroke: the left crosses at trim = SLOTP_DY - R - e, so the
    # lateral error it can resolve is TRIM_Y - (SLOTP_DY - R) each way.
]
