"""WHAT A ROBOT BELIEVES IS ON THE BOARD, as a costmap.

One robot in this project navigates and the other does not.  Robot 2 builds
a map every time it plans; robot 1 has never touched `nav` at all -- it
drives a script of turn-and-drive legs to poses that were chosen with a
ruler, which is why its kit approach spent a season driving through four
patients without anything noticing.  A robot that cannot see the board
cannot avoid what is on it.

So the map-building moves out of one robot's mission file and becomes the
layer both of them share:

    static      the field's own furniture -- walls, the laboratory plate
    observed    pieces the cameras have found, as discs
    booked      another agent's published plan, as space-time windows
    live        another agent's measured footprint, swept forward

Each layer is optional and each is a different KIND of knowledge, which is
the distinction that matters when they disagree: a booking is a promise
about the future and belongs in a window a planner may pay to cross; a
measured footprint is the present and is a wall.  Confusing the two is how
robot 2 spent seventy-four seconds of every match believing nothing was
deliverable (F128).

Pure numpy and nav.  No mujoco, no mission.
"""
import numpy as np

from . import nav


def board_map(pieces=(), skip=(), cmap=None, res=None, size=None,
              schedule=None, reserve=None, t_now=0.0,
              fleet=None, whose=None, horizon=1.6, piece_r=12.0,
              extra=()):
    """The costmap to plan on.

    pieces     [(id, x, y, ...), ...] observed loose pieces
    skip       ids to leave out -- the one being picked up, usually
    schedule   another agent's published plan, stamped via `reserve`
    reserve    reserve(cmap, schedule, t_now) -- the task's own booking rule
    fleet      a fleet.Fleet, to stamp other agents' measured footprints
    whose      which agent is asking (so it is not stamped as its own wall)
    extra      [(x, y, r), ...] anything else to block out

    Everything is optional: a map with no arguments is just the field.
    """
    cm = nav.CostMap.field() if cmap is None else cmap
    if reserve is not None and schedule is not None:
        reserve(cm, schedule=schedule, t_now=t_now)
    if fleet is not None and whose is not None:
        fleet.stamp(cm, whose, horizon=horizon)
    drop = ({skip} if isinstance(skip, (int, np.integer)) else set(skip or ()))
    for p in pieces:
        if p[0] in drop:
            continue
        cm.add_disc(float(p[1]), float(p[2]), piece_r)
    for x, y, r in extra:
        cm.add_disc(float(x), float(y), float(r))
    return cm


def route_to(cmap, start, goal, foot, speed=250.0, t0=0.0, strict=False):
    """Plan a path and say how long it should take, for any footprint.

    A thin wrapper, but it is the one place that decides how a Footprint
    becomes the two radii nav.plan wants -- inscribed for the hard block,
    circumscribed for the decaying cost -- so callers stop each inventing
    their own pair.
    """
    path, secs = nav.plan(cmap, start[:2], goal[:2],
                          foot.inscribed, foot.circumscribed,
                          t0=t0, speed=speed, strict=strict)
    return path, secs
