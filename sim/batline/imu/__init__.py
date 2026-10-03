"""The board's IMU and its calibration at station 4 (README, Milestone 3).

Production, numpy only (line.py and the station import these):
    kin      the gimbal's motion and what it does to a point on the clamp
    part     the datasheet's IMU: codes, scale, its output filter
    plan     the calibration plan, designed from the targets and the limits
    fit      the bat's IMU calibration from its stream and the cameras' poses
    record   what is written to the bat and to the factory's database
    golden   the golden bat's daily check of the station

TRUTH, simulation and checks only (no procedure imports these):
    model    the part's errors, warm-up, clock, filter and codes
    scene    the weighed gimbal in MuJoCo with the IMU's sensors
    sim      a bat through the station: the motion as the motors make it,
             the encoders, the cameras' poses, the stream
    judge    the fit against the truth
"""
