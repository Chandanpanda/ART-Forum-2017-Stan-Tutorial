"""The carbon-fibre truss fabrication cell, as a MuJoCo test bench.

Second fixture for the reusable robotics codebase (see CLAUDE.md).  The
competition package next door is a mobile robot on a board; this one is a
gantry over a rotating fixture, winding thread around rod joints.  What the
two share is the discipline, not the code: one parameter file, a model
generated from it, a HAL that mission code never reaches around, solvers
instead of constants, and a check suite that measures the model before
anything is concluded from it.

Tiers, by what the thread is:

    0   arithmetic          spec, geometry, structure, approach, schedule,
                            inspector -- numpy only, runs on the controller
    1   rigid rods, the     the whole cell in MuJoCo; the thread is a band
        machine simulated   that becomes a weld once the turns are counted
    2   a real cable        one joint, a cable-composite thread, a tensioned
                            pay-out -- the contact-rich questions, slowly
"""
