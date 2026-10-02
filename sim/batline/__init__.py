"""The IMU bat's build-and-calibration line, as a MuJoCo test bench.

Third fixture for the reusable robotics codebase (see CLAUDE.md), beside
the competition robot (sim/rfgyc26) and the truss cell (sim/truss).  The
product is a cricket-style bat with a bought IMU board in a printed handle
and a carbon truss blade under a printed knit sleeve; the line builds it,
calibrates it against eight cameras and packs it, with a person visiting
only for the manual tasks the plan keeps on purpose.

The plan this package follows is "IMU bat line: MuJoCo implementation
plan" (12 milestones).  M0 is what is here so far:

    spec        facts, product rules and the task specs (a Bat, a Line)
    product     the bat derived from a Bat: every part, every fit, the
                frame's truss, the bands' zones, the masses
    mjcf        the derived bat in MuJoCo, one body per part, from
                primitives (a mesh collides as its hull, so a bay cannot
                be a mesh)
    line        station job estimates, the floor layout, trip times and
                every size the line needs (printers, mandrels, stores,
                racks), computed from a Line and a derived bat
    executive   des: the fast line model, a month of the line in seconds

and in sim/scripts/batline: check_bat and check_line (check_all runs
both) and demo_bat, which renders the derived bat to be looked at.

Tiers, as in the truss cell:

    0   arithmetic          spec, product, line, executive -- numpy only
    1   rigid MuJoCo        the bat (M0); stations, robot and docks later
    2   one mechanism       the sleeve's cloth, the rollers, the swing (later)

imu_fusion_sim.py is the study that chose the bat; product.py reads its
rated swing, so the frame is sized against the swing the accuracy claim
was made for.
"""
