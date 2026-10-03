"""The enclosed swing rig at station 4 (Milestone 4): one bat in
Est.SWING_EVERY swung through the rated swing at full speed on the dummy
pack, streaming, to show its IMU does not rail and its cells do not let go
for longer than the board's hold-up.

Production (the station runs these):
    rig      the arm and wrist fitted to the rated swing, the motors chosen,
             the run, the station's test and its planned time
    hal      the rig as the test drives it
TRUTH (simulation and checks only):
    sim      the rig in MuJoCo: its servo loop, the bat and its sprung pack,
             the contacts' breaks, the IMU's readings
"""
