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
              shove=False, shove_cost=900.0, extra=()):
    """The costmap to plan on.

    pieces     [(id, x, y, ...), ...] observed loose pieces
    skip       ids to leave out -- the one being picked up, usually
    schedule   another agent's published plan, stamped via `reserve`
    reserve    reserve(cmap, schedule, t_now) -- the task's own booking rule
    fleet      a fleet.Fleet, to stamp other agents' measured footprints
    whose      which agent is asking (so it is not stamped as its own wall)
    shove      treat the pieces as EXPENSIVE rather than impassable
    extra      [(x, y, r), ...] anything else to block out

    Everything is optional: a map with no arguments is just the field.

    A 5 g CYLINDER IS NOT A WALL TO A 4 kg ROBOT (F153).  Whether a loose
    piece blocks a route depends on who is asking.  Robot 2 is light and
    usually carrying, so a patient in the way is a wall.  Robot 1 is
    seventeen times its mass and shoves them aside without noticing -- it
    did exactly that, on a scripted path, for the whole life of this
    project.  The first version of its costmap made them hard obstacles for
    everyone, and robot 1 promptly declared both kit zones unreachable:
    the sticker columns leave 80 mm gaps and robot 1 is 235 mm wide, so
    there is no route at all.  Measured, that cost the kit column 34 points
    and skipped the zones outright.

    `shove` says the piece may be driven through at a price.  The price is
    what makes it a last resort rather than a habit -- the planner will pay
    several hundred millimetres of detour to avoid one, and will still get
    to the hospital when the only way there is through.
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
        if shove:
            cm.add_sticky(float(p[1]), float(p[2]), r=piece_r,
                          cost=shove_cost)
        else:
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


# ------------------------------------------------------------- topology
def widest_path(cmap, start, goal):
    """The route between two points that keeps the most room, and how much.

    WHERE ARE THE DOORS?  fleet.py carries eight rectangles drawn by eye,
    two of which -- "the two pinches, 176 mm of gap for a chassis planned at
    150" -- are the whole reason its reservation protocol exists.  A door is
    not a rectangle somebody typed; it is a property of the free space, and
    the property is this: of all the ways from here to there, take the one
    whose TIGHTEST point is as wide as possible.  That tightest point is the
    door.

    This is the bottleneck-shortest-path (maximum-capacity path) problem,
    and it is Dijkstra with max-min relaxation in place of sum-min: the
    value of reaching a cell is the narrowest clearance on the best route to
    it, and the most generous frontier is expanded first.

    Returns (path, width, gate): the route, its narrowest clearance in mm,
    and where that occurs.  No route gives (None, 0.0, None).
    """
    import heapq
    d = cmap.clearance()
    nx, ny = d.shape
    si, sj = cmap.cell(*start[:2])
    gi, gj = cmap.cell(*goal[:2])
    best = np.full((nx, ny), -1.0)
    prev = {}
    best[si, sj] = d[si, sj]
    q = [(-float(d[si, sj]), si, sj)]
    while q:
        w, i, j = heapq.heappop(q)
        w = -w
        if w < best[i, j]:
            continue
        if (i, j) == (gi, gj):
            break
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                a, b = i + di, j + dj
                if not (0 <= a < nx and 0 <= b < ny):
                    continue
                cand = min(w, float(d[a, b]))
                if cand > best[a, b]:
                    best[a, b] = cand
                    prev[(a, b)] = (i, j)
                    heapq.heappush(q, (-cand, a, b))
    if best[gi, gj] < 0:
        return None, 0.0, None
    path, node = [], (gi, gj)
    while node != (si, sj):
        path.append(cmap.centre(*node))
        node = prev.get(node)
        if node is None:
            return None, 0.0, None
    path.append(cmap.centre(si, sj))
    path.reverse()
    # THE ENDS ARE NOT DOORS.  The narrowest point on a route is usually
    # where it starts or finishes, because a start point is often chosen
    # tight against something; skip a body's length at each end so the gate
    # is a constriction the route passes THROUGH.
    skip = max(1, int(round(200.0 / cmap.res)))
    mid = path[skip:-skip] or path
    gate = min(mid, key=lambda p: float(d[cmap.cell(*p)]))
    return path, float(best[gi, gj]), gate


def doors(cmap, start, goal, radius, keep=2, bite=None):
    """Every distinct passage between two places, widest first.

    Take the widest path, note its gate, block that gate, and ask again --
    which is how to enumerate the ways through a wall without knowing in
    advance how many there are.  Stops when what is left is too tight for a
    body of `radius`.
    """
    bite = float(bite if bite is not None else radius * 2.0)
    work = cmap
    out = []
    for _ in range(keep):
        path, width, gate = widest_path(work, start, goal)
        if path is None or width < radius or gate is None:
            break
        out.append((gate, width))
        blocked = nav.CostMap(res=work.res, size=(work.w, work.h))
        blocked.static = cmap.static.copy()
        for g, _w in out:
            blocked.add_disc(g[0], g[1], bite)
        work = blocked
    return out
