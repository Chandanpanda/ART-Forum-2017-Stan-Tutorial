"""The calibration station's camera rig (plan, milestone M2): a two-axis
gimbal that turns the bat, and eight cameras round it that measure the
clamp's pose in every frame and read the bands' colours.

    spec.py     the rig derived from the bat and the hardware: the gimbal,
                its markers, the ball size, the lens, the standoff, the
                certified bar; and the station's keep-outs
    place.py    where the eight cameras stand: a coverage solver held to
                spec.Rig's rule with any one camera lost
    mjcf.py     the rig in MuJoCo, as drawn or as built (RigBuildDraw:
                every camera's place, turn and lens, the gimbal's axes and
                zeros, the markers, the sensors' colour, the stems' pull)
    lens.py     the camera model every part shares: a pinhole with
                Brown-Conrady distortion, and a ball's image centre
    vision.py   what the cameras see, behind one contract (RigVisionHAL):
                ModelVision, fast, and PixelVision, rendered and warped
                through each camera's true lens
    track.py    prediction, association, self-calibration's loop (and the
                stems' pull on a centroid), tracking, the shift-start check
    calib.py    bundle adjustment of every camera's lens and pose from the
                certified bar; the drift test
    pose.py     a rigid body's pose from many cameras: the clamp's, in
                every frame
    colour.py   each camera's colour response from the clamp's card, and
                the bands' colours against the sleeve beside them
    judge.py    the results against the scene's truth: for the checks only
"""
