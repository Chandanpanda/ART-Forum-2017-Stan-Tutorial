"""The line's executive: what the line does next, and when.

The plan ("Running the line") has one executive driven three ways -- the
fast line model, station islands and end to end.  M0 is the first: des
runs the executive's policy against timed resources for a month of the
line in seconds, with line.plan_line's times and sizes, and reports
whether they held.  The islands (M11) replace the timers with MuJoCo and
keep the policy.

    des     the fast line model: stations, the robot and the person as
            timed resources; trays, bats, orders and stores as state
"""
