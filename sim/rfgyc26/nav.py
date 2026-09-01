"""Navigation: the costmap and the path planner (design doc section 15.4/15.5c).

Both robots plan on the SAME map object, inflated to their own radius.  This
file replaces every hand-picked waypoint chain in the project: a route is
something the machine computes from geometry it can see, not something a human
pastes in as a pair of numbers.

    map  = CostMap.field()                    # walls + laboratory plate
    map.add_disc(x, y, r)                     # a patient, a placed beam...
    grid = map.inflated(R2_INSCRIBED, R2_CIRCUM)
    path = astar(grid, (x0, y0), (x1, y1))    # metres of millimetres, in ms

THE INFLATION IS THE WHOLE TRICK.  A planner that treats the robot as a point
drives its corners through walls; one that treats it as a disc of the
circumscribed radius cannot enter a 191 mm pinch that the chassis physically
fits.  So the standard two-radius scheme: HARD-BLOCK inside the inscribed
radius (no orientation can fit there), and a decaying COST out to the
circumscribed radius (some orientations fit; prefer not to, but go if the
value justifies it).  That is what lets robot 2 thread the plate/sticker
pinches on purpose while never clipping a wall by accident.

SPACE-TIME.  add_window() blocks a region only during [t0, t1).  A* carries
the arrival time along with the cost, so a corridor the other robot owns from
T+58 to T+73 is an obstacle if you would arrive at T+65 and open floor if you
would arrive at T+80.  That is what makes one fleet plan possible instead of
two plans discovering each other at 20 Hz.

Pure numpy, no mujoco: this file ships to the Pi unchanged.
"""
import heapq
import numpy as np

from .params import Field

RES = 20.0                    # mm per cell -- 58 x 60 over the field
FIELD_W = 1143.0
FIELD_H = 1181.0

# The cost a cell carries at the very edge of the inscribed zone, decaying to
# zero at the circumscribed radius.  120 is chosen against the A* step cost
# (one cell = 20): a 6-cell detour is worth taking to gain one cell of
# clearance, which keeps paths off walls without making them cowardly.
INFLATE_PEAK = 120.0
BLOCKED = 1e9


class CostMap:
    """An occupancy grid with obstacle discs, rectangles and time windows."""

    def __init__(self, res=RES):
        self.res = float(res)
        self.nx = int(np.ceil(FIELD_W / self.res))
        self.ny = int(np.ceil(FIELD_H / self.res))
        self.static = np.zeros((self.nx, self.ny), dtype=bool)
        self._windows = []          # (mask, t0, t1)
        self._sticky = np.zeros((self.nx, self.ny), dtype=float)
        self._cache = {}

    # ------------------------------------------------------------- indexing
    def cell(self, x, y):
        return (int(np.clip(x // self.res, 0, self.nx - 1)),
                int(np.clip(y // self.res, 0, self.ny - 1)))

    def centre(self, i, j):
        return ((i + 0.5) * self.res, (j + 0.5) * self.res)

    def _grid_xy(self):
        xs = (np.arange(self.nx) + 0.5) * self.res
        ys = (np.arange(self.ny) + 0.5) * self.res
        return np.meshgrid(xs, ys, indexing="ij")

    # ------------------------------------------------------------ obstacles
    def add_rect(self, x0, y0, x1, y1, mask=None):
        gx, gy = self._grid_xy()
        m = (gx >= x0 - self.res) & (gx <= x1 + self.res) & \
            (gy >= y0 - self.res) & (gy <= y1 + self.res)
        if mask is None:
            self.static |= m
            self._cache.clear()
        else:
            mask |= m
        return m

    def add_disc(self, x, y, r, mask=None):
        gx, gy = self._grid_xy()
        m = (gx - x) ** 2 + (gy - y) ** 2 <= (r + self.res * 0.5) ** 2
        if mask is None:
            self.static |= m
            self._cache.clear()
        else:
            mask |= m
        return m

    def add_window(self, mask, t0, t1):
        """Block `mask` only during [t0, t1) -- the other robot's corridor."""
        self._windows.append((np.asarray(mask, dtype=bool), float(t0),
                              float(t1)))

    def add_corridor(self, pts, half_w, t0, t1):
        """A swept corridor along a polyline, live for [t0, t1)."""
        gx, gy = self._grid_xy()
        m = np.zeros((self.nx, self.ny), dtype=bool)
        pts = [np.asarray(p, float) for p in pts]
        for a, b in zip(pts[:-1], pts[1:]):
            ab = b - a
            L2 = float(ab @ ab)
            if L2 < 1e-9:
                m |= (gx - a[0]) ** 2 + (gy - a[1]) ** 2 <= half_w ** 2
                continue
            t = np.clip(((gx - a[0]) * ab[0] + (gy - a[1]) * ab[1]) / L2,
                        0.0, 1.0)
            px, py = a[0] + t * ab[0], a[1] + t * ab[1]
            m |= (gx - px) ** 2 + (gy - py) ** 2 <= half_w ** 2
        self.add_window(m, t0, t1)
        return m

    def add_sticky(self, x, y, r=60.0, cost=400.0):
        """A jam happened here.  Cost, not a block: the cell may still be the
        only way through, but the planner should want an alternative."""
        gx, gy = self._grid_xy()
        self._sticky += cost * ((gx - x) ** 2 + (gy - y) ** 2 <= r ** 2)
        self._cache.clear()

    # ------------------------------------------------------------ inflation
    def inflated(self, inscribed, circumscribed, t=None):
        """Cost grid for a robot of these radii, at time t (None = static)."""
        key = (round(inscribed, 1), round(circumscribed, 1))
        if key not in self._cache:
            self._cache[key] = self._inflate(self.static, inscribed,
                                             circumscribed)
        grid = self._cache[key] + self._sticky
        if t is not None and self._windows:
            live = np.zeros((self.nx, self.ny), dtype=bool)
            for mask, t0, t1 in self._windows:
                if t0 <= t < t1:
                    live |= mask
            if live.any():
                grid = grid + self._inflate(live, inscribed,
                                            circumscribed, walls=False)
        return grid

    def windows_at(self, t):
        live = np.zeros((self.nx, self.ny), dtype=bool)
        for mask, t0, t1 in self._windows:
            if t0 <= t < t1:
                live |= mask
        return live

    def _inflate(self, occ, inscribed, circumscribed, walls=True):
        """Exact-ish distance inflation by successive dilation.  The grid is
        3 400 cells; a brute-force EDT over the obstacle cells is microseconds
        and avoids a scipy dependency the Pi image may not carry."""
        grid = np.zeros(occ.shape, dtype=float)
        gx, gy = self._grid_xy()
        dist = np.full(occ.shape, 1e9)
        if walls:
            # THE WALLS ARE SURFACES, NOT CELLS (F100).  Marking them by
            # `gx < 1` matched no cell centre at all -- the first cell's
            # centre is at 10 -- so the field boundary was simply ABSENT
            # from the map and every plan was free to route through it.
            # Distance to the boundary is exact and costs nothing.
            dist = np.minimum(dist, np.minimum(gx, FIELD_W - gx))
            dist = np.minimum(dist, np.minimum(gy, FIELD_H - gy))
        if occ.any():
            oi, oj = np.nonzero(occ)
            ox = (oi + 0.5) * self.res
            oy = (oj + 0.5) * self.res
            step = 512
            for s in range(0, len(oi), step):
                dx = gx[..., None] - ox[None, None, s:s + step]
                dy = gy[..., None] - oy[None, None, s:s + step]
                dist = np.minimum(dist,
                                  np.sqrt(dx * dx + dy * dy).min(axis=2))
        if not np.isfinite(dist).any() or (dist >= 1e9).all():
            return grid
        grid[dist <= inscribed] = BLOCKED
        band = (dist > inscribed) & (dist < circumscribed)
        if circumscribed > inscribed:
            f = (circumscribed - dist[band]) / (circumscribed - inscribed)
            grid[band] = INFLATE_PEAK * f * f
        return grid

    # ----------------------------------------------------------- factories
    @classmethod
    def field(cls, res=RES, plate=True):
        """Walls and the laboratory plate: the geometry that never moves.

        The plate is a HARD obstacle for both robots -- neither can climb its
        5 mm edge (F11, measured: 20.7 s going nowhere) -- and putting it in
        the map is what retires every hand-drawn dogleg in the project."""
        m = cls(res)
        gx, gy = m._grid_xy()
        # the walls: everything outside the playing surface, plus a half-cell
        # so a path never rides the boundary exactly
        # (the walls themselves are handled as boundary DISTANCE in
        #  _inflate -- see F100; marking cells here never worked)
        if plate:
            m.add_rect(*Field.LAB_PLATE)
        m._cache.clear()
        return m


# ================================================================== the A*
_NB = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
       (1, 1, 1.41421356), (1, -1, 1.41421356),
       (-1, 1, 1.41421356), (-1, -1, 1.41421356)]


def astar(grid, start, goal, res=RES, t0=0.0, speed=280.0, cmap=None,
          inscribed=None, circumscribed=None):
    """Shortest path on a cost grid, in field millimetres.

    Returns [(x, y), ...] from start to goal, or None if unreachable.  With
    `cmap` given the search is SPACE-TIME: each node carries its arrival time
    (t0 + travelled/speed) and a cell blocked by a live window at that time is
    refused -- so a corridor the other robot owns for fifteen seconds is an
    obstacle only to a path that would be there during them.
    """
    nx, ny = grid.shape
    si, sj = (int(np.clip(start[0] // res, 0, nx - 1)),
              int(np.clip(start[1] // res, 0, ny - 1)))
    gi, gj = (int(np.clip(goal[0] // res, 0, nx - 1)),
              int(np.clip(goal[1] // res, 0, ny - 1)))
    if grid[gi, gj] >= BLOCKED:
        gi, gj = _nearest_free(grid, gi, gj)
        if gi is None:
            return None
    if grid[si, sj] >= BLOCKED:
        # A BLOCKED START IS NORMAL, not an error: the robot finishes a push
        # nose-to-nose with the patient it just delivered, so its own cell is
        # inside that patient's halo.  Every neighbour is blocked too, the
        # search expands nothing, and the mission reads "unreachable" for a
        # board it could drive out of in half a second -- measured, eight
        # pucks abandoned in one tick.  Snap to the nearest enterable cell
        # and let the local planner cover the few centimetres.
        ni, nj = _nearest_free(grid, si, sj, radius=8)
        if ni is None:
            return None
        si, sj = ni, nj

    def h(i, j):
        dx, dy = abs(i - gi), abs(j - gj)
        return (max(dx, dy) + 0.41421356 * min(dx, dy)) * res

    open_ = [(h(si, sj), 0.0, si, sj)]
    came, gcost = {}, {(si, sj): 0.0}
    seen = set()
    windows = cmap is not None and cmap._windows
    while open_:
        _, g, i, j = heapq.heappop(open_)
        if (i, j) in seen:
            continue
        seen.add((i, j))
        if (i, j) == (gi, gj):
            return _trace(came, (si, sj), (gi, gj), res)
        for di, dj, w in _NB:
            ni, nj = i + di, j + dj
            if not (0 <= ni < nx and 0 <= nj < ny) or (ni, nj) in seen:
                continue
            c = grid[ni, nj]
            if c >= BLOCKED:
                continue
            ng = g + w * res + c * w
            if windows:
                arrive = t0 + ng / max(speed, 1.0)
                live = cmap.windows_at(arrive)
                if live[ni, nj]:
                    continue
            if ng < gcost.get((ni, nj), 1e18):
                gcost[(ni, nj)] = ng
                came[(ni, nj)] = (i, j)
                heapq.heappush(open_, (ng + h(ni, nj), ng, ni, nj))
    return None


def _nearest_free(grid, gi, gj, radius=6):
    """The goal cell is inside an inflation halo (a station IS near a wall).
    Take the closest cell that is actually enterable."""
    nx, ny = grid.shape
    best, bd = None, 1e9
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            i, j = gi + di, gj + dj
            if 0 <= i < nx and 0 <= j < ny and grid[i, j] < BLOCKED:
                d = di * di + dj * dj
                if d < bd:
                    best, bd = (i, j), d
    return best if best else (None, None)


def _trace(came, s, g, res):
    out, cur = [], g
    while cur != s:
        out.append(((cur[0] + 0.5) * res, (cur[1] + 0.5) * res))
        cur = came[cur]
    out.append(((s[0] + 0.5) * res, (s[1] + 0.5) * res))
    out.reverse()
    return out


def simplify(path, grid, res=RES, tol=BLOCKED):
    """Line-of-sight shortcutting: an 8-connected grid path is a staircase,
    and a differential drive should not drive staircases.  Keeps only the
    knees a straight run cannot skip."""
    if not path or len(path) < 3:
        return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _clear(path[i], path[j], grid, res, tol):
            j -= 1
        out.append(path[j])
        i = j
    return out


def _clear(a, b, grid, res, tol):
    n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / (res * 0.5)))
    xs = np.linspace(a[0], b[0], n)
    ys = np.linspace(a[1], b[1], n)
    ii = np.clip((xs // res).astype(int), 0, grid.shape[0] - 1)
    jj = np.clip((ys // res).astype(int), 0, grid.shape[1] - 1)
    return bool((grid[ii, jj] < tol).all())


def path_length(path):
    if not path or len(path) < 2:
        return 0.0
    p = np.asarray(path, float)
    return float(np.hypot(*(p[1:] - p[:-1]).T).sum())


def plan(cmap, start, goal, inscribed, circumscribed, t0=0.0, speed=280.0,
         soft_windows=True, strict=False):
    """The one call the rest of the project makes: inflate, search, simplify.
    Returns (path, seconds) or (None, inf).

    TWO STAGES, because a hard space-time block can be un-escapable: if the
    robot is STANDING inside a corridor the other robot owns right now, its
    own position is blocked and the strict search returns nothing -- but
    "get out" is exactly the plan we need most at that moment.  So: strict
    first (never enter a reserved corridor); on failure, retry with the
    windows as heavy COST instead of walls, which produces a path that
    leaves promptly and crosses only if there is no alternative.

    strict=True disables that second stage.  A reservation exists to keep
    robot 2 out of robot 1's way, and a fallback that quietly crosses it
    anyway is not a reservation -- measured, robot 2 went to work on the west
    columns during the seal window and cost the fleet both beams.  Use
    strict for WORK (if the corridor is busy, do something else) and the
    default for ESCAPE (get out, whatever it takes)."""
    grid = cmap.inflated(inscribed, circumscribed)
    raw = astar(grid, start, goal, cmap.res, t0=t0, speed=speed, cmap=cmap)
    if raw is None and soft_windows and not strict and cmap._windows:
        soft = grid.copy()
        for mask, w0, w1 in cmap._windows:
            if w1 > t0:                      # only windows still ahead of us
                soft[mask & (grid < BLOCKED)] += 4000.0
        raw = astar(soft, start, goal, cmap.res)
        grid = soft
    if raw is None:
        return None, float("inf")
    path = simplify(raw, grid, cmap.res)
    return path, path_length(path) / max(speed, 1.0)


# ====================================================== push planning (15.5b)
# Delivering a patient is NON-PREHENSILE MANIPULATION: the robot cannot carry
# the cylinder, only shove it, so the question "can this puck reach that zone"
# is a search, not a lookup.  The first build answered it with a hand-derived
# catalogue keyed on which sticker column a puck started in -- correct for the
# opening layout and wrong for every board state after the first push.  This
# derives the same answer from the map, and returns "unreachable" as an honest
# infinite cost the task selector can price.
#
# State = the puck's position.  Action = push along heading th for distance d.
# An action is feasible iff, ALL from the costmap:
#   * the approach pose (behind the puck along -th) is free and reachable,
#   * the robot's body sweeps the whole push without touching anything but
#     the target puck,
#   * the release pose (a short back-off) is free.

PUSH_HEADINGS = 16
PUSH_DISTS = (60.0, 120.0, 200.0, 300.0, 420.0, 560.0, 700.0)
PLOW_REACH = 92.0            # centre to just behind the pocket mouth
# the CAPTURE POCKET IS PART OF THE BODY (F107): its flare tips reach 77 mm
# ahead of the axle, and a planner that stops the footprint at 78 routes a
# robot 15 mm shorter than the one that has to fit.
# check_r2_pocket.py asserts these points contain every collidable geom, so
# this list can no longer drift away from the robot it claims to describe.
BODY_PTS = [(-78.0, 55.0), (-78.0, -55.0), (79.0, 55.0), (79.0, -55.0),
            (79.0, 40.0), (79.0, -40.0), (79.0, 0.0), (0.0, 0.0)]


def body_masks(cmap, n_head=PUSH_HEADINGS):
    """For each discrete heading, a boolean grid: can the CHASSIS sit with its
    centre in this cell facing this way?

    Precomputed because the push search asks the question about a million
    times -- 16 headings x 7 distances x ~15 sweep samples per state -- and
    doing it point-by-point cost 3.5 s per puck (measured), which is 40 s of
    an opening plan that has 120 s to spend.  As sixteen vectorised array
    ops it is milliseconds."""
    occ = cmap.inflated(6.0, 8.0) >= BLOCKED
    nx, ny = occ.shape
    gx, gy = cmap._grid_xy()
    out = []
    for k in range(n_head):
        a = 2.0 * np.pi * k / n_head
        ca, sa = np.cos(a), np.sin(a)
        bad = np.zeros((nx, ny), dtype=bool)
        for lx, ly in BODY_PTS:
            wx = gx + lx * ca - ly * sa
            wy = gy + lx * sa + ly * ca
            i = np.clip((wx // cmap.res).astype(int), 0, nx - 1)
            j = np.clip((wy // cmap.res).astype(int), 0, ny - 1)
            bad |= occ[i, j]
        out.append(~bad)
    return out


def _infield(x, y, margin=95.0):
    """Is a robot centre here inside the playing surface, with room?"""
    return (margin <= x <= FIELD_W - margin and
            margin <= y <= FIELD_H - margin)


def _body_free(body, x, y, a):
    ca, sa = np.cos(a), np.sin(a)
    for lx, ly in BODY_PTS:
        wx, wy = x + lx * ca - ly * sa, y + lx * sa + ly * ca
        i = int(np.clip(wx // RES, 0, body.shape[0] - 1))
        j = int(np.clip(wy // RES, 0, body.shape[1] - 1))
        if body[i, j] >= BLOCKED:
            return False
    return True


def push_actions(masks, res, px, py):
    """Every feasible single push from (px, py).  Yields (tx, ty, th, dist)."""
    nx, ny = masks[0].shape
    for k, ok_here in enumerate(masks):
        th = 2.0 * np.pi * k / len(masks)
        ux, uy = np.cos(th), np.sin(th)
        ax, ay = px - ux * PLOW_REACH, py - uy * PLOW_REACH
        i = int(np.clip(ax // res, 0, nx - 1))
        j = int(np.clip(ay // res, 0, ny - 1))
        if not ok_here[i, j]:
            continue                      # nowhere to stand behind it
        for d in PUSH_DISTS:
            n = max(2, int(d / 40.0))
            ss = np.linspace(0.0, d, n)
            ii = np.clip(((ax + ux * ss) // res).astype(int), 0, nx - 1)
            jj = np.clip(((ay + uy * ss) // res).astype(int), 0, ny - 1)
            if not ok_here[ii, jj].all():
                break                     # longer pushes only get worse
            yield (px + ux * d, py + uy * d, np.degrees(th), d)


def plan_push(cmap, puck, zone, robot=None, max_legs=3, speed=170.0,
              transit=300.0, avoid=None):
    """Push `puck` into `zone` = (x0, y0, x1, y1), in at most max_legs legs.

    Returns (legs, seconds) with legs = [(tx, ty), ...] -- the puck's target
    after each leg -- or (None, inf) when the map says it cannot be done.

    THREE LEGS, not two: a green on the west column has to get south past the
    laboratory plate and then 640 mm east, and no two straight pushes do that
    (measured -- every green came back NO PLAN at two).  And "cannot be done"
    is usually TEMPORARY: a puck's own column mates are obstacles here, so
    the top of a column is deliverable and the one beneath it is not until
    the top has gone.  That resolves itself, because the caller re-plans
    against the live board after every delivery.
    """
    x0, y0, x1, y1 = zone
    masks = cmap._push_masks if getattr(cmap, "_push_masks", None) is not None \
        else body_masks(cmap)
    cmap._push_masks = masks

    def parks_badly(x, y):
        """THE LEAVE-CLEAN INVARIANT (design doc 15.7).  A push may not DEPOSIT
        its puck inside a corridor the other robot still needs.  Robot 2 was
        obeying the reservations with its own body and then shoving patients
        into them -- five of twelve seeds finished with no beams at all, and
        the seed with the best patient score had the worst beams.  Keeping
        yourself out of the way is not the same as leaving the way clear."""
        if avoid is None:
            return False
        i = int(np.clip(x // cmap.res, 0, avoid.shape[0] - 1))
        j = int(np.clip(y // cmap.res, 0, avoid.shape[1] - 1))
        return bool(avoid[i, j])

    def inside(x, y):
        return x0 <= x <= x1 and y0 <= y <= y1

    def h(x, y):
        dx = max(x0 - x, 0.0, x - x1)
        dy = max(y0 - y, 0.0, y - y1)
        return float(np.hypot(dx, dy)) / speed

    start = (float(puck[0]), float(puck[1]))
    if inside(*start):
        return [], 0.0
    seen = {}
    heap = [(h(*start), 0.0, start, ())]
    best = (None, float("inf"))
    while heap:
        f, g, p, legs = heapq.heappop(heap)
        if len(legs) >= max_legs:
            continue
        key = (round(p[0] / 40.0), round(p[1] / 40.0), len(legs))
        if key in seen and seen[key] <= g:
            continue
        seen[key] = g
        for tx, ty, th, d in push_actions(masks, cmap.res, p[0], p[1]):
            # cost: get behind the puck, then push it
            ux, uy = np.cos(np.radians(th)), np.sin(np.radians(th))
            ax, ay = p[0] - ux * PLOW_REACH, p[1] - uy * PLOW_REACH
            frm = robot if not legs else legs[-1]
            reposition = (float(np.hypot(ax - frm[0], ay - frm[1])) / transit
                          if frm is not None else 0.0)
            ng = g + reposition + 1.2 + d / speed
            if parks_badly(tx, ty):
                continue
            nl = legs + ((tx, ty),)
            if inside(tx, ty):
                if ng < best[1]:
                    best = (list(nl), ng)
                continue
            if ng + h(tx, ty) < best[1]:
                heapq.heappush(heap, (ng + h(tx, ty), ng, (tx, ty), nl))
    return best if best[0] else (None, float("inf"))


def approach_pose(puck, target):
    """Where the robot must stand, and facing where, to push puck -> target."""
    ux, uy, _ = _unit(target[0] - puck[0], target[1] - puck[1])
    return (puck[0] - ux * PLOW_REACH, puck[1] - uy * PLOW_REACH,
            float(np.degrees(np.arctan2(uy, ux))))


def _unit(dx, dy):
    n = float(np.hypot(dx, dy))
    if n < 1e-9:
        return 1.0, 0.0, 0.0
    return dx / n, dy / n, n


# THE FIELD MARGIN IS THE PLANNING RADIUS, NOT A ROUND NUMBER (F116).
# 85 mm was a guess, and it rejected every stand-off directly north or
# south of the INNER patient columns, which stand at x 80 -- the one
# family of approaches that gives those six patients a way out, since
# an approach from the field centre leaves the nose 27 mm off the wall
# with no turning arc available.  A* hard-blocks at 75, so 78 is the
# honest bound and anything larger is throwing away reachable board.
def capture_approach(cmap, puck, prefer=None, standoffs=(185.0, 150.0, 128.0),
                     margin=78.0, seat=26.0):
    """Where to stand, and facing where, to CAPTURE this patient.

    A pushing robot must approach along the push line -- that coupling is
    what made the sticker columns unworkable, because the run-up to a puck
    passed straight through its neighbours.  A robot that CAPTURES has no
    such constraint: it may come from any side that is clear, and carry
    wherever it likes afterwards.  So this searches the headings and returns
    the cheapest clear one, preferring `prefer` (usually the carry bearing)
    when several work.

    Returns (x, y, heading_deg) or None.
    """
    masks = getattr(cmap, "_push_masks", None)
    if masks is None:
        masks = body_masks(cmap)
        cmap._push_masks = masks
    nx, ny = masks[0].shape
    best, best_c = None, 1e18
    for k, ok_here in enumerate(masks):
        th = 2.0 * np.pi * k / len(masks)
        ux, uy = np.cos(th), np.sin(th)
        # A LADDER OF STAND-OFFS, not one (F107).  A single 185 mm stand-off
        # rejected the whole x-80 column -- every heading that fitted put the
        # stand-off inside the wall margin, and every heading that cleared
        # the margin did not fit.  At 150 mm heading 202 is clear.  The
        # comfortable distance first, then whatever the corner allows.
        for standoff in standoffs:
            sx, sy = puck[0] - ux * standoff, puck[1] - uy * standoff
            if not _infield(sx, sy, margin=margin):
                continue
            i = int(np.clip(sx // cmap.res, 0, nx - 1))
            j = int(np.clip(sy // cmap.res, 0, ny - 1))
            if not ok_here[i, j]:
                continue
            # AND THE POSE IT ENDS IN, WHICH IS THE TIGHT ONE (F119).
            # This validated the STAND-OFF and nothing else -- a pose
            # 150-185 mm back, where the patient's neighbours are
            # comfortably far away.  The capture ends 26 mm from it,
            # with a 157 mm chassis among neighbours 80 and 113 mm
            # apart, and there the TAIL corner lands on the puck
            # behind.  Measured on the east block: every heading the
            # search offered put a corner inside a neighbour, the
            # turn-out check refused them all, and robot 2 called
            # twelve patients undeliverable on a board where three
            # had clean two-second carries.  Check the pose the robot
            # will actually be in when it has the thing.
            cx, cy = puck[0] - ux * seat, puck[1] - uy * seat
            ci = int(np.clip(cx // cmap.res, 0, nx - 1))
            cj = int(np.clip(cy // cmap.res, 0, ny - 1))
            if not ok_here[ci, cj]:
                continue
            # the run-in must be clear right up to the pocket's mouth
            run = np.linspace(0.0, max(0.0, standoff - PLOW_REACH), 6)
            ii = np.clip(((sx + ux * run) // cmap.res).astype(int), 0, nx - 1)
            jj = np.clip(((sy + uy * run) // cmap.res).astype(int), 0, ny - 1)
            if not ok_here[ii, jj].all():
                continue
            c = (0.0 if prefer is None else
                 abs((np.degrees(th) - prefer + 180.0) % 360.0 - 180.0))
            c += 0.25 * (185.0 - standoff)      # prefer room to line up
            if c < best_c:
                best_c, best = c, (sx, sy, float(np.degrees(th)))
            break
    return best
