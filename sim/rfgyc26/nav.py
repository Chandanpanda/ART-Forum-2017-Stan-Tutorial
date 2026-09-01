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
                grid = grid + self._inflate(live, inscribed, circumscribed)
        return grid

    def windows_at(self, t):
        live = np.zeros((self.nx, self.ny), dtype=bool)
        for mask, t0, t1 in self._windows:
            if t0 <= t < t1:
                live |= mask
        return live

    def _inflate(self, occ, inscribed, circumscribed):
        """Exact-ish distance inflation by successive dilation.  The grid is
        3 400 cells; a brute-force EDT over the obstacle cells is microseconds
        and avoids a scipy dependency the Pi image may not carry."""
        grid = np.zeros(occ.shape, dtype=float)
        if not occ.any():
            return grid
        oi, oj = np.nonzero(occ)
        gx, gy = self._grid_xy()
        ox = (oi + 0.5) * self.res
        oy = (oj + 0.5) * self.res
        # distance to the nearest obstacle cell centre, in blocks to bound RAM
        dist = np.full(occ.shape, 1e9)
        step = 512
        for s in range(0, len(oi), step):
            dx = gx[..., None] - ox[None, None, s:s + step]
            dy = gy[..., None] - oy[None, None, s:s + step]
            dist = np.minimum(dist, np.sqrt(dx * dx + dy * dy).min(axis=2))
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
        m.static |= (gx < 1.0) | (gx > FIELD_W - 1.0) | \
                    (gy < 1.0) | (gy > FIELD_H - 1.0)
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
         soft_windows=True):
    """The one call the rest of the project makes: inflate, search, simplify.
    Returns (path, seconds) or (None, inf).

    TWO STAGES, because a hard space-time block can be un-escapable: if the
    robot is STANDING inside a corridor the other robot owns right now, its
    own position is blocked and the strict search returns nothing -- but
    "get out" is exactly the plan we need most at that moment.  So: strict
    first (never enter a reserved corridor); on failure, retry with the
    windows as heavy COST instead of walls, which produces a path that
    leaves promptly and crosses only if there is no alternative."""
    grid = cmap.inflated(inscribed, circumscribed)
    raw = astar(grid, start, goal, cmap.res, t0=t0, speed=speed, cmap=cmap)
    if raw is None and soft_windows and cmap._windows:
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
