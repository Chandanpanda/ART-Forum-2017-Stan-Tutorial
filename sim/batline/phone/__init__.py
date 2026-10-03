"""The factory iPhone's test at station 4 (Milestone 4): the phone pairs
with the calibrated bat, reads its record back, films its two bands while
the gimbal swings it slowly, and the game app's fusion must put the sweet
spot where the rig says it is, within the accuracy budget at that swing.

Production (the station and the app run these):
    camera   the phone's camera at the player's distance, through a fold mirror
    hal      the two instruments the test drives besides the phone's radio:
             the gimbal and the phone's camera
    app      the game app's fusion: the record applied to the IMU, the
             stance anchored from the bands, the swing dead-reckoned
    test     the slow swing, the limit from the accuracy budget, the test
             itself and its planned time
TRUTH (simulation and checks only):
    sim      the gimbal's motion, the bat's IMU through it, the camera's
             frames, a calibrated bat's record
    picture  the phone's view of the bat rendered in MuJoCo, and where
             each band's pixels lie
"""
