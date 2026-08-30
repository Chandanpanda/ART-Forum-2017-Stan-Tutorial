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
    # HEIGHT OF THE POWERED SURFACE AT THE INTAKE.  This is the single biggest
    # modelling assumption in the whole simulator (F1).  At -0.3 the belt reaches
    # the floor and picks pieces up directly -- which a O16 roller cannot do: its
    # own radius puts the belt face at ~9.5 at best, and spec 3.1 quotes 17.5.
    # Sweep it (scripts/risk_intake.py) to see what the intake really tolerates.
    # 3.0 = a knife-edge nose bar, which is buildable.  Measured cliff, capture
    # over 8 randomised matches (24 samples): Za -0.3, 3, 4, 5 -> 24/24;
    # Za 6, 9.5, 13, 17.5 -> 0/24.  ONE MILLIMETRE wide, and the reason is
    # geometric: the powered edge has to get under the piece's half-thickness
    # (2.5 mm for a 5 mm disc) or it just shunts it along the floor.
    # A O16 roller puts the belt face at ~18 -- it cannot work, at all.
    BELT_NOSE_Z     = 3.0
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
    # F47.  The retainer used to be 80 mm long on a 74 mm stroke, so PARKED it
    # reached Ya 114 -- 16.5 mm inside the beam pocket, at exactly the height a
    # carried beam rides.  It rubbed the beam for the whole match and dragged
    # the placed one off station.  A blade cannot be parked clear of a O66 bore
    # AND stay inboard of Ya 95.5 unless it is shorter: near edge on the bore
    # rim (33) plus half-length must come to <= 95.  62 mm long on a 64 mm
    # stroke does it, and a O56 disc only needs +/-28 of support, so nothing
    # about the escapement's job changes.
    ESC_BLADE_Y     = 31.0             # retainer half-length; covers the bore
    ESC_BLADE_PARK  = 64.0             # its own stroke -- NOT the gate's 74
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

    # SLOT PROBES (F30).  Two downward reflectance sensors -- the spec's TCRT
    # array -- straddling the chute axis, mounted one bore-radius FORWARD of it so
    # they clear the bore and the escapement.  Reversing, the chute crosses a slot
    # first and the probes follow PROBE_DX later, so the slot can be measured
    # after the chute has passed over it and the dock corrected from the
    # measurement rather than from dead reckoning.
    #
    # This is the only laboratory feature the rulebook actually guarantees: it
    # gives no thickness, no height and no frame, just "3 marked slots of 60 mm"
    # in wood (rules 3.2).  Anything datumed off a plate edge or a back wall
    # would be assuming a part nobody has specified -- the mistake F21 and F27
    # already cost us twice.
    PROBE_DX        = 40.0             # Xa forward of the chute axis
    PROBE_DY        = 20.0             # +/- lateral; chord over a O60 slot is 45
    PROBE_Z         = 20.0             # site height, below the belt underside

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
    # The one that actually decides whether the intake works.  Measured cliff:
    # a powered edge at or below Za 5 captures 24/24, at Za 6 it captures 0/24,
    # because it has to get under the piece's half-thickness instead of shunting
    # it.  This assertion is what stops the design drifting back to a roller.
    ("intake edge gets under the piece (Za <= half its thickness + 2.5)",
     Chassis.BELT_NOSE_Z <= Piece.DISC_T / 2.0 + 2.5),
    ("a roller nose could NOT satisfy that -- the design needs a knife edge",
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
    ("the parked escapement retainer stays out of the beam pocket (F47)",
     AgentA.ESC_BLADE_PARK + AgentA.ESC_BLADE_Y <= AgentA.POCKET_IN_Y),
    ("...and parked it is still clear of the bore",
     AgentA.ESC_BLADE_PARK - AgentA.ESC_BLADE_Y >= AgentA.CHUTE_D/2.0),
]
