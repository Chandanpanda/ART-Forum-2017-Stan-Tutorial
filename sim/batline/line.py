"""The line, sized from a Line and a derived bat.

What each station does to one bat and how long it takes, where the
stations and racks stand along the robot's aisle, how long the robot takes
between any two docks, and how big everything has to be: printers,
mandrels, trays, stores, racks.  Nothing here runs the line --
executive/des.py runs a month of it against what this computes, and
check_line holds the two together.

HOW A TIME IS MADE.  Every op says which of four sources it came from:

    move     a gantry move: the truss cell's trapezoids (truss.motion) on
             the module's own axes (truss.spec.Gantry), over a distance
             computed from the station's layout -- never a typed duration
    stroke   a force-watched insertion at Est.INSERT_V, over a length the
             bat's own layout gives (product.Layout)
    plan     a planner's own time on this bat: the frame's (the truss
             cell's planner, approach.plan_truss and schedule.plan, the
             same check_schedule holds for the cell's own trusses), the
             IMU calibration's (imu/plan.py), and each station test's
             procedure (board/procedures.py, phone/test.py,
             swing/rig.py), each held to its simulation by its check
    est      a process nobody has timed: spec.Est, which names the
             milestone whose check replaces it

The robot's trips are the same trapezoids at the robot's own limits, over
legs measured on the floor layout.  Est.PACE scales all of it, as the
truss cell's SPEED_FACTOR does, once a rig has measured it.

HOW A SIZE IS MADE.  Orders arrive as a Poisson stream, so every store is
a base stock: the quantile, at Line.service, of what it hands out between
two refills -- counted in bats, because a bat takes its parts together.
Printed parts are reviewed at every visit and arrive at the next one, the
textbook periodic review with a lead time of one period, so the farm
prints a whole interval's usage at the service level and the store holds
two.  Mandrels are Little's law: the rate station 1 needs to win back the
longest outage inside one line day, times the cure.  Racks hold whatever
fills them between two visits.  Nothing is tuned against the event model;
the event model only says whether it held.

THE LAYOUT FOLLOWS FROM THE DOCK.  A tray lands on balls carried by two
rails the robot drives between, so it has to span the robot from rail to
rail.  The default robot's deck is wider than a tray of four bats is wide,
so the tray spans it lengthways: at a dock the bats lie ALONG the aisle,
docks stand side by side a tray's length apart, and each station's fixture
stands behind its docks with the gantry's x along the aisle.  dock_geometry
works that out for any robot and tray -- a robot narrower than a tray of
bats turns them across the aisle and packs the docks far closer.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from math import ceil, degrees, exp, floor, lgamma, log, pi, sqrt

import numpy as np

from truss import motion, approach, schedule, structure, fixture as truss_fixture, band as truss_band
from truss.geometry import TrussGeometry
from truss.spec import Gantry, Gripper, Ring, Dispenser
from .spec import (Line, Est, Faults, Person, Robot, Dock, Printer, Print, Rules, Epoxy, Thread,
                   Rod, Label, CapSpring, Tube, Module)
from . import product

STATIONS = ("frame", "electronics", "sleeve", "calibration", "pack")
# the bat's route, and the racks between, in the order they stand along the
# aisle -- the plan's route ("The bat's route"), which is an input
ROUTE = ("inbound", "frame", "cure", "stock", "sleeves", "electronics", "sleeve",
         "calibration", "pack", "outbound", "charger")


# ============================================================ SIZING MATHS
def poisson_q(mean, p):
    """The smallest k with P(Poisson(mean) <= k) >= p.  Summed in log
    space, so a mean in the thousands neither underflows nor stalls."""
    if mean <= 0.0:
        return 0
    k, log_term, cdf = 0, -mean, 0.0
    while True:
        cdf += exp(log_term)
        if cdf >= p:
            return k
        k += 1
        log_term += log(mean / k)


def mean_ceil(mean, k):
    """E[ceil(N / k)] for N ~ Poisson(mean): trays it takes to carry a
    random batch k at a time."""
    kmax = int(mean + 12.0 * sqrt(mean + 1.0) + 10)
    pmf = _poisson_pmf(mean, kmax)
    return float(pmf @ np.ceil(np.arange(kmax + 1) / k))


def _poisson_pmf(mean, kmax):
    k = np.arange(kmax + 1)
    if mean <= 0.0:
        return (k == 0).astype(float)
    lg = np.array([lgamma(i + 1.0) for i in k])
    pmf = np.exp(k * log(mean) - mean - lg)
    return pmf / pmf.sum()


def _binom_pmf(n, p_ok):
    j = np.arange(n + 1)
    if p_ok >= 1.0:
        return (j == n).astype(float)
    lg = np.array([lgamma(n + 1.0) - lgamma(i + 1.0) - lgamma(n - i + 1.0) for i in j])
    return np.exp(lg + j * log(p_ok) + (n - j) * log(1.0 - p_ok))


def stock_law(mean, plates, per_plate, p_fail, tol=1e-12):
    """What a printed store's level has to cover, under the farm's own rule.

    At each visit the person restarts just enough plates to bring the
    store back up to its level: whole plates, at most `plates` of them,
    each failing at p_fail, which nobody knows until the next visit, when
    the good ones arrive.  Demand is Poisson, `mean` a period.  How far the
    store sits below its level once the plates now printing are counted,
    Y (negative: a whole plate's overshoot), is a Markov chain:

        Y' = Y + D - per_plate * Binom(min(plates, ceil((Y + D) / per_plate)), 1 - p_fail)

    -- the capacitated base stock of Federgruen and Zipkin (1986), with a
    farm that prints only what it is short, so a failed plate is a
    shortfall even while printers stand idle.  The store runs dry in a
    period when Y + D + D' passes the level: plates started at one visit
    arrive at the next, so the level covers two periods' demand and the
    shortfall.  Returns (cdf of Y + D + D' from `lo`, lo, mean shortfall),
    or None when the farm cannot keep up at all.  Its stationary law is one
    linear solve; check_line holds it against a simulation of the rule."""
    if plates * per_plate * (1.0 - p_fail) <= mean:
        return None
    kmax = int(mean + 12.0 * sqrt(mean + 1.0) + 10)
    d_pmf = _poisson_pmf(mean, kmax)
    good = [_binom_pmf(q, 1.0 - p_fail) for q in range(plates + 1)]
    lo = -(per_plate - 1)
    n = 4 * (kmax + plates * per_plate)
    while True:
        size = n - lo + 1
        T = np.zeros((size, size))
        for y in range(lo, n + 1):
            for d in range(kmax + 1):
                need = y + d
                q = 0 if need <= 0 else min(plates, -(-need // per_plate))
                to = np.clip(need - per_plate * np.arange(q + 1), lo, n) - lo
                np.add.at(T[y - lo], to, d_pmf[d] * good[q])
        A = T.T - np.eye(size)
        A[-1, :] = 1.0
        b = np.zeros(size)
        b[-1] = 1.0
        pi = np.clip(np.linalg.solve(A, b), 0.0, None)
        if pi[-(kmax + 1):].sum() < tol or n > 20000:
            break
        n *= 2
    pi = pi / pi.sum()
    two = _poisson_pmf(2.0 * mean, int(2.0 * mean + 12.0 * sqrt(2.0 * mean + 1.0) + 10))
    ys = np.arange(lo, n + 1)
    return np.cumsum(np.convolve(pi, two)), lo, float(np.clip(ys, 0, None) @ pi)


def capacitated_stock(mean, plates, per_plate, p_fail, service):
    """(the level a printed store is refilled to, its mean shortfall): the
    least level that stock_law's store covers with probability `service`;
    (None, inf) if the farm cannot keep up."""
    law = stock_law(mean, plates, per_plate, p_fail)
    if law is None:
        return None, float("inf")
    cdf, lo, short = law
    return max(0, lo + int(np.searchsorted(cdf, service))), short


# ================================================================ THE DOCK
@dataclass
class DockGeometry:
    along: bool          # True: at a dock the bats lie along the aisle
    pitch: float         # mm between dock centres along the aisle
    depth: float         # mm a dock reaches into its station from the aisle's edge
    aisle: float         # mm, the robot's aisle, wall to wall
    span_margin: float   # mm the tray reaches past both rails (negative: it cannot)
    slots: list          # each bat slot's offset across the tray from its centre
    cradle: float        # mm, one slot's width


def dock_geometry(d):
    """Which way round a tray sits on a dock, and what that does to the
    floor.  A tray lands on balls carried by two rails the robot drives
    between, so whichever tray dimension lies across the robot must reach
    over its width, both clearances and both rails.  If both do, the
    shorter one goes across: the docks then pack closer along the aisle."""
    L, W = d.tray["length"], d.tray["width"]
    fwd, across = Robot.DECK
    need = across + 2.0 * (Dock.CLEAR + Dock.RAIL_W)
    options = [(dim, other, along) for dim, other, along in ((W, L, False), (L, W, True))
               if dim >= need]
    dim, other, along = min(options) if options else (L, W, True)
    k = d.tray["per_tray"]
    cradle = (W - 2.0 * Rules.LOAD_WALLS * Print.WALL) / k
    return DockGeometry(along=along, pitch=max(dim, need), depth=max(fwd, other) + Dock.CLEAR,
                        aisle=max(fwd, other) + 2.0 * Dock.CLEAR, span_margin=dim - need,
                        slots=[(i - (k - 1) / 2.0) * cradle for i in range(k)], cradle=cradle)


# ============================================================= A STATION
@dataclass
class Op:
    what: str
    s: float
    how: str             # move | stroke | plan | est


@dataclass
class Station:
    """One copy of the module: its docks in a row along the aisle, its
    fixture behind them, and its jobs as lists of ops.  `reach` is what the
    head point must cover, (x along the bat, y across it, z); `travel` is
    the module's, derived from it with the tools, the camera, the overshoot
    and the home switch (station/spec.py), and `module` is that module.
    `footprint` is (along the aisle, into the room)."""
    name: str
    docks: tuple
    dock_u: dict
    fixture: tuple       # (u, v) centre
    fixture_size: tuple  # (u, v)
    footprint: tuple
    # the part of the fixture the head works on, when it is not all of it:
    # ((along the bat, across), its top above the floor datum)
    work: tuple = None
    sweep: tuple = None       # ((u, v, z), r): a ball the fixture sweeps while the head waits (station/spec.py)
    reach: tuple = (0.0, 0.0, 0.0)
    travel: tuple = (0.0, 0.0, 0.0)
    module: object = None
    limits: tuple = None                          # (v, a) of each axis, from the module
    jobs: dict = field(default_factory=dict)      # job -> [Op]

    def job_s(self, job):
        return sum(o.s for o in self.jobs.get(job, ()))


class _Head:
    """A station's gantry, in the station's floor frame: u along the aisle,
    v into the station.  The module's x is the bat's axis, as in the truss
    cell, so which floor direction x runs in follows the dock.  `limits` is
    the station's own (v, a) per axis; without it, the module's rated ones."""

    def __init__(self, dg, lift, limits=None):
        self.dg = dg
        self.ax = ("x", "y") if dg.along else ("y", "x")
        self.lift = lift
        self.v, self.a = limits if limits is not None else (Module.V_RATED, Module.A_RATED)
        self.points = []

    def at(self, p):
        self.points.append(p)
        return p

    def bat_axis(self, p, s):
        """The point s mm along the bat's axis from p."""
        return (p[0] + s, p[1]) if self.dg.along else (p[0], p[1] + s)

    def move(self, a, b):
        d = {self.ax[0]: b[0] - a[0], self.ax[1]: b[1] - a[1]}
        return (motion.coordinated_time(d, self.v, self.a) + Gantry.SETTLE_S) * Est.PACE

    def z(self):
        return (motion.trap_time(self.lift, self.v["z"], self.a["z"]) + Gantry.SETTLE_S) * Est.PACE

    def transfer(self, a, b, rise=0.0):
        """The head at b goes to a, takes the part, brings it to b and
        leaves it there: two crossings, four strokes of the lift, a grip
        and a release.  Where b stands `rise` above a, the head goes down
        it empty and up it carrying."""
        self.at(a), self.at(b)
        climb = 2.0 * (motion.trap_time(rise, self.v["z"], self.a["z"]) * Est.PACE if rise > 0.0 else 0.0)
        return 2.0 * self.move(a, b) + 4.0 * self.z() + climb + 2.0 * Gripper.JAW_CLOSE_S

    def slide(self, length):
        """A free move along the bat's axis -- the module's x."""
        return motion.trap_time(length, self.v["x"], self.a["x"]) * Est.PACE


def _stroke(length, v=None):
    return length / (Est.INSERT_V if v is None else v)


def _station(name, docks, fixture_ab, dg, lift, limits=None, around=0.0):
    """Lay a station out: docks side by side along the aisle, the fixture
    behind them.  fixture_ab is the fixture's (along the bat, across it);
    `around` is how far round the fixture's centre the station must own the
    floor -- the calibration station's cameras -- short of the aisle."""
    fu, fv = fixture_ab if dg.along else fixture_ab[::-1]
    width = max(len(docks) * dg.pitch, fu, 2.0 * around)
    u0 = (width - len(docks) * dg.pitch) / 2.0
    dock_u = {k: u0 + (i + 0.5) * dg.pitch for i, k in enumerate(docks)}
    fixture = (width / 2.0, dg.depth + Dock.CLEAR + fv / 2.0)
    st = Station(name, tuple(docks), dock_u, fixture, (fu, fv),
                 (width, max(dg.depth + Dock.CLEAR + fv, fixture[1] + around)))
    return st, _Head(dg, lift, None if limits is None else limits.get(name))


def _tray_centre(st, dg, dock):
    return (st.dock_u[dock], dg.depth / 2.0)


def _slots(st, dg, dock):
    u, v = _tray_centre(st, dg, dock)
    return [(u, v + o) if dg.along else (u + o, v) for o in dg.slots]


def _mean_transfer(head, slots, b, rise=0.0):
    return sum(head.transfer(a, b, rise) for a in slots) / len(slots)


# ============================================================ THE FRAME
@contextmanager
def _gantry(limits):
    """The truss planner, timed on another gantry's axes: it reads
    truss.spec.Gantry's limits when it plans, so they are swapped for the
    station's own for as long as it does."""
    if limits is None:
        yield
        return
    old = Gantry.V_MAX, Gantry.A_MAX
    Gantry.V_MAX, Gantry.A_MAX = dict(limits[0]), dict(limits[1])
    try:
        yield
    finally:
        Gantry.V_MAX, Gantry.A_MAX = old


@lru_cache(maxsize=16)
def _truss_plan(t, limits=None):
    g = TrussGeometry(t)
    fx = truss_fixture.Fixture(g)
    with _gantry(limits):
        P = schedule.plan(g, fx, approach.plan_truss(g, fx, span=0.0))
    return schedule.summary(P)


def frame_plan(d, limits=None):
    """The truss cell's own plan for the bat's frame, minutes by phase, on
    station 1's axes when `limits` (v, a) gives them -- station 1 is the
    truss cell made long enough to reach its docks."""
    key = None if limits is None else tuple(tuple(sorted(x.items())) for x in limits)
    return _truss_plan(d.frame.truss, key)


def bom(d):
    """What one bat takes, by the station that uses it.  Thread and resin
    from the frame's own geometry: the truss cell's hoops (TrussGeometry)
    and its dose rule (band.resin_dose_mg)."""
    t, g = d.frame.truss, d.geom
    thread_m = sum(g.thread_per_chord(k) for k in range(t.n_chords)) / 1000.0
    resin_ml = sum(truss_band.resin_dose_mg(t, g.thread_on_joint(j))
                   for j in g.joints) / 1000.0 / Dispenser.RESIN_RHO
    return {"frame": {"core": 1, "rods": t.n_chords + t.n_diag, "thread_m": thread_m,
                      "resin_ml": resin_ml},
            "electronics": {"board": 1, "contacts": 1},
            "sleeve": {"tip_cap": 1, "collar": 1, "sleeve": 1},
            "pack": {"pommel_cap": 1, "spring": 1, "tube": 1, "tube_cap": 1, "label": 1}}


# ============================================================ THE STATIONS
def stations(d, drives=None, timed=True):
    """Every station's layout, module and jobs, computed from the derived
    bat.  Two passes: the layout and what each head must reach first, at
    the module's rated speeds; then each station's module is derived from
    that (station/spec.py) and every move is timed again on that station's
    own axes, since an axis too long for its drive is slower than rated.
    `drives` overrides Module.DRIVES, the order drives are tried in.
    `timed=False` skips the second pass, for a caller that wants only the
    modules: the jobs are then timed at the rated speeds."""
    from .station import spec as station_spec
    sts, dg = _layout(d)
    mods = {k: station_spec.derive(station_spec.from_line(st, dg, d, drives=drives)) for k, st in sts.items()}
    lim = {k: ({a: m.axes[a].v_max for a in "xyz"}, {a: m.axes[a].a_max for a in "xyz"})
           for k, m in mods.items()}
    if timed:
        sts, dg = _layout(d, lim)
    for k, st in sts.items():
        st.module, st.limits = mods[k], lim[k]
        st.travel = tuple(mods[k].axes[a].travel for a in "xyz")
    return sts, dg


def _layout(d, limits=None):
    """Every station's layout and jobs, its moves timed at `limits`
    (station -> (v, a) per axis) or at the module's rated speeds."""
    dg = dock_geometry(d)
    lay = d.lay
    L = d.tube["length"]
    lift = d.tube["od"]                 # the tallest thing a tray carries is a bat in its cradle
    out = {}

    # ---------------------------------------------------------------- 1
    # The frame station is the truss cell: its fixture is the cell's own
    # build envelope (structure.reach, the loader's racks included), and
    # its build is the cell's own plan.
    xr, yr = d.frame.reach
    st, h = _station("frame", ("cores", "mandrels A", "mandrels B", "frames"), (xr, 2.0 * yr), dg, lift,
                     limits)
    F = st.fixture
    core_mid = h.bat_axis(F, -(L / 2.0) + lay.x_shoulder / 2.0)    # the core rides the mandrel's end
    plan = frame_plan(d, None if limits is None else limits.get("frame"))
    thread_m = bom(d)["frame"]["thread_m"]
    reload_turns = thread_m * 1000.0 / (2.0 * pi * Ring.GROOVE_R)
    st.jobs["build"] = [
        Op("an empty mandrel into the cage", h.transfer(_tray_centre(st, dg, "mandrels A"), F), "move"),
        Op("a core onto the mandrel's end", h.transfer(_tray_centre(st, dg, "cores"), core_mid), "move"),
        Op("the frame loaded, wound and dosed (the truss cell's planner)",
           plan["total_min"] * 60.0 * Est.PACE, "plan"),
        Op("the ring's groove reloaded from the spool at its winding speed",
           (reload_turns / Ring.RPM * 60.0 + Ring.SPINUP_S) * Est.PACE, "move"),
        Op("the wound mandrel back into its tray", h.transfer(F, _tray_centre(st, dg, "mandrels A")), "move")]
    st.jobs["session"] = [Op("a new mixing nozzle, purged", Est.NOZZLE_S, "est")]
    st.jobs["collapse"] = [
        Op("a cured mandrel into the cage", h.transfer(_tray_centre(st, dg, "mandrels B"), F), "move"),
        Op("the draw rod pulled, the mandrel folded", Est.COLLAPSE_S, "est"),
        Op("the frame off it, into a bat tray", _mean_transfer(h, _slots(st, dg, "frames"), F), "move"),
        Op("the empty mandrel back into its tray", h.transfer(F, _tray_centre(st, dg, "mandrels B")), "move")]
    out["frame"] = (st, h)

    # ---------------------------------------------------------------- 2
    from .board import procedures as board_procedures
    st, h = _station("electronics", ("bats", "parts"), (L, dg.cradle), dg, lift, limits)
    F = st.fixture
    pommel = h.at(h.bat_axis(F, -L / 2.0))
    parts = _tray_centre(st, dg, "parts")
    st.jobs["bat"] = [
        Op("the bat from its slot into the cradle", _mean_transfer(h, _slots(st, dg, "bats"), F), "move"),
        Op("the contact set from the parts tray to the bay's mouth", h.transfer(parts, pommel), "move"),
        Op("pushed along the bay to the + plate", _stroke(lay.x_plus - lay.x_face), "stroke"),
        Op("pressed to its stop, force curve read", Est.PRESS_S, "est"),
        Op("the board from the parts tray to the slot's mouth", h.transfer(parts, pommel), "move"),
        Op("slid in until it snaps", _stroke(lay.x_stop - lay.x_face), "stroke"),
        Op("the pogo block down on the pads and up", 2.0 * h.z(), "move"),
        Op("flashed, serial written, self-test (board/procedures.py)", board_procedures.plan_s(), "plan"),
        Op("the battery path checked with a dummy cell", Est.DUMMY_CELL_S, "est"),
        Op("the bat back into its slot", _mean_transfer(h, _slots(st, dg, "bats"), F), "move")]
    st.jobs["tray"] = [Op("the trays' tags read", 2 * Est.TAG_READ_S, "est")]
    out["electronics"] = (st, h)

    # ---------------------------------------------------------------- 3
    blade = lay.x_tip1 - lay.x_shoulder
    st, h = _station("sleeve", ("bats", "stretchers", "parts"), (L + blade, dg.cradle), dg, lift, limits)
    F = st.fixture
    bat_mid = h.bat_axis(F, -blade / 2.0)        # the bat's cradle, the stretcher's holder ahead of it
    tip = h.at(h.bat_axis(bat_mid, L / 2.0))
    holder = h.at(h.bat_axis(tip, blade / 2.0))
    pommel = h.at(h.bat_axis(bat_mid, -L / 2.0))
    parts = _tray_centre(st, dg, "parts")
    st.jobs["bat"] = [
        Op("the bat from its slot into the cradle", _mean_transfer(h, _slots(st, dg, "bats"), bat_mid), "move"),
        Op("the tip cap from the parts tray onto the tip", h.transfer(parts, tip), "move"),
        Op("pressed home over the chords", _stroke(lay.socket_l) + Est.PRESS_S, "stroke"),
        Op("its stretcher into the holder, in line", h.transfer(_tray_centre(st, dg, "stretchers"), holder),
           "move"),
        Op("the bat pushed through it tip first, drawing the sleeve on",
           _stroke(blade, Est.SLEEVE_FEED_V), "stroke"),
        Op("the stretcher opened", Est.STRETCHER_S, "est"),
        Op("the empty stretcher back into its tray", h.transfer(holder, _tray_centre(st, dg, "stretchers")),
           "move"),
        Op("the collar from the parts tray over the pommel", h.transfer(parts, pommel), "move"),
        Op("slid down the grip", h.slide(lay.x_shoulder), "move"),
        Op("pushed home over the hem", _stroke(lay.x_blade0 - lay.x_shoulder), "stroke"),
        Op("the order code in the hem read", Est.CODE_READ_S, "est"),
        Op("the bat back into its slot", _mean_transfer(h, _slots(st, dg, "bats"), bat_mid), "move")]
    st.jobs["tray"] = [Op("the trays' tags read", 3 * Est.TAG_READ_S, "est")]
    out["sleeve"] = (st, h)

    # ---------------------------------------------------------------- 4
    # The gimbal turns the bat about its middle through every orientation,
    # so it sweeps a ball; its eight cameras stand round it (rig/spec.py
    # derives both from the bat).  The fixture is the gimbal on its posts;
    # the station owns the floor out to its cameras; the head works only on
    # the bat in the parked gimbal's clamp, and waits at home, above the
    # sweep, while the gimbal turns.
    from .rig import spec as rig_spec
    rig = rig_spec.design(d)
    st, h = _station("calibration", ("bats",), rig.footprint(), dg, lift, limits, around=rig.reach())
    st.work = rig.work()
    st.sweep = (st.fixture + (rig.h_c,), rig.R_s + rig.clear)
    F = st.fixture
    rise = rig.work()[1]
    # the gimbal's motion is the IMU calibration's own plan, designed from
    # the targets and the gimbal's limits (imu/plan.py), and the day starts
    # with a golden bat run through it as often as catching a drifted rig
    # takes (imu/golden.py)
    from .imu import plan as imu_plan, golden as imu_golden
    from .phone import test as phone_test
    from .swing import rig as swing_rig
    load = _mean_transfer(h, _slots(st, dg, "bats"), F, rise)
    bay = _stroke(lay.x_plus - lay.x_face)          # the pack's lead pip from the mouth to the + plate
    st.jobs["bat"] = [
        Op("the bat from its slot into the gimbal's clamp", load, "move"),
        Op("the pogo clamp closed and opened", 2.0 * h.z(), "move"),
        Op("the IMU calibration plan", imu_plan.for_design(d).duration_s, "plan"),
        Op("the factory iPhone's test (phone/test.py)", phone_test.plan_s(d), "plan"),
        Op("the bat back into its slot", load, "move")]
    # the swing rig stands beside the gimbal, its clamp where the head
    # reaches as it reaches the gimbal's; the rig pushes its dummy pack
    # home and closes the bay with a cap of its own carrying the product's
    # spring (the pommel cap goes on at the pack station)
    st.jobs["swing"] = [
        Op("the bat from its slot into the swing rig's clamp", load, "move"),
        Op("the dummy pack pushed home along the bay, the rig's cap closed", bay, "stroke"),
        Op("the full-speed swing test (swing/rig.py)", swing_rig.plan_s(d), "plan"),
        Op("the dummy pack drawn back out", bay, "stroke"),
        Op("the bat back into its slot", load, "move")]
    st.jobs["day"] = [
        Op("a golden bat from its slot into the clamp, before the day's first", load, "move"),
        Op("the pogo clamp closed and opened", 2.0 * h.z(), "move"),
        Op("the golden bat's runs of the plan", imu_golden.for_design(d).duration_s, "plan"),
        Op("the golden bat back into its slot", load, "move")]
    st.jobs["tray"] = [Op("the tray's tag read", Est.TAG_READ_S, "est")]
    out["calibration"] = (st, h)

    # ---------------------------------------------------------------- 5
    st, h = _station("pack", ("bats", "parts", "tubes"), (2.0 * L, 2.0 * dg.cradle), dg, lift, limits)
    F = st.fixture
    cradle = h.bat_axis(F, -L / 2.0)                            # the bat's cradle ...
    tube = h.bat_axis(F, L / 2.0)                               # ... in line with the tube's
    pommel = h.at(h.bat_axis(cradle, -L / 2.0))
    tube_end = h.at(h.bat_axis(tube, L / 2.0))
    magazine = h.at((tube[0], tube[1] + dg.cradle) if dg.along else (tube[0] + dg.cradle, tube[1]))
    parts = _tray_centre(st, dg, "parts")
    turns = lay.stub_l / Print.THREAD_PITCH
    st.jobs["bat"] = [
        Op("the bat from its slot into the cradle", _mean_transfer(h, _slots(st, dg, "bats"), cradle), "move"),
        Op("the pommel cap from the parts tray onto the spindle", h.transfer(parts, pommel), "move"),
        Op("its spring from the parts tray, pressed into it", h.transfer(parts, pommel) + Est.PRESS_S, "move"),
        Op("the cap screwed on at the thread's lead, to its torque",
           turns / (Est.SPINDLE_RPM / 60.0), "est"),
        Op("a tube from the magazine into its cradle", h.transfer(magazine, tube), "move"),
        Op("the blade pushed into the tube", h.slide(L), "move"),
        Op("the handle seated past the cap's skirt", _stroke(Tube.CAP_DEPTH), "stroke"),
        Op("the tube cap from its magazine, pressed on", h.transfer(magazine, tube_end) + Est.PRESS_S, "move"),
        Op("a label printed, pressed on and read back", Est.LABEL_S, "est"),
        Op("the packed tube into the tube tray", _mean_transfer(h, _slots(st, dg, "tubes"), tube), "move")]
    st.jobs["reject"] = [Op("a rejected bat onto the reject shelf",
                            _mean_transfer(h, _slots(st, dg, "bats"), magazine), "move")]
    st.jobs["tray"] = [Op("the trays' tags read", 3 * Est.TAG_READ_S, "est")]
    out["pack"] = (st, h)

    # every station: what its head point reaches, as (x, y, z) in the
    # module's axes -- the points it went to, the tray corners it picks
    # from and, for the frame, the truss cell's own reach
    hl, hw = d.tray["length"] / 2.0, d.tray["width"] / 2.0
    for name, (st, h) in out.items():
        pts = list(h.points)
        for k in st.docks:
            u, v = _tray_centre(st, dg, k)
            du, dv = (hl, hw) if dg.along else (hw, hl)
            pts += [(u - du, v - dv), (u + du, v + dv)]
        us, vs = [p[0] for p in pts], [p[1] for p in pts]
        du, dv = max(us) - min(us), max(vs) - min(vs)
        x, y = (du, dv) if dg.along else (dv, du)
        if name == "frame":
            x, y = max(x, d.frame.reach[0]), max(y, 2.0 * d.frame.reach[1])
        st.reach = (x, y, lift)
    return {k: v[0] for k, v in out.items()}, dg


# ============================================================== THE FLOOR
@dataclass
class Floor:
    dg: DockGeometry
    items: dict          # name -> (side, u0, width, depth)
    docks: dict          # "station/dock" or rack name -> (u, side)
    length: float        # the aisle's length

    def trip_s(self, a, b):
        """The robot from dock a to dock b: out to the aisle's centreline,
        along it, into b.  Each leg a trapezoid at the robot's limits; the
        last few centimetres onto the balls are Est.DOCK_S, not here."""
        if a == b:
            return 0.0
        (ua, _), (ub, _) = self.docks[a], self.docks[b]
        out = self.dg.depth / 2.0 + self.dg.aisle / 2.0
        legs = (out, abs(ua - ub), out)
        return sum(motion.trap_time(x, Robot.V_MAX, Robot.A_MAX) for x in legs) * Est.PACE


def floor_plan(sts, dg, racks):
    """Stations and racks along both sides of one aisle, in route order,
    each on whichever side is shorter so far.  `racks` is name -> number
    of tray positions; a rack is a row of docks at the docks' pitch."""
    ends = [0.0, 0.0]
    items, docks = {}, {}
    for name in ROUTE:
        if name in sts:
            st = sts[name]
            width, depth = st.footprint
        else:
            width, depth = max(racks.get(name, 1), 1) * dg.pitch, dg.depth
        side = 0 if ends[0] <= ends[1] else 1
        u0 = ends[side]
        items[name] = (side, u0, width, depth)
        ends[side] += width
        if name in sts:
            for k in sts[name].docks:
                docks["%s/%s" % (name, k)] = (u0 + sts[name].dock_u[k], side)
        else:
            docks[name] = (u0 + width / 2.0, side)
    return Floor(dg, items, docks, max(ends))


# ============================================================= THE SIZING
@dataclass
class Rates:
    """Bats a day through each station, at a demand: rejects from a station
    on are made up by starting more there."""
    shipped: float
    starts: dict         # station -> bats a day it works on
    sleeves: float       # sleeves a day the person makes
    rejects: dict        # station -> rejects a day


def rates(line):
    lam = line.per_day
    starts, y = {}, 1.0
    for k in reversed(STATIONS):
        y *= 1.0 - Faults.REJECT[k]
        starts[k] = lam / y
    rejects = {k: starts[k] * Faults.REJECT[k] for k in STATIONS}
    # a bat rejected before its sleeve is fitted hands its sleeve to the
    # next frame; from the sleeve station on, a reject costs a sleeve
    return Rates(lam, starts, starts["sleeve"], rejects)


def _footprint_on_tray(fp, tray_lw):
    """How many of a part's footprint one tray's nests hold: the printer's
    packing rule, on the tray, with a nest wall between parts."""
    gap = 2.0 * (Print.CLEAR + Print.WALL)
    return product._pack_on_bed(fp, bed=tray_lw, gap=gap)


def _kit_footprint(fps):
    """One nest that takes a station's whole kit for one bat: the parts
    side by side across, the longest along."""
    gap = 2.0 * (Print.CLEAR + Print.WALL)
    return (max(a for a, _ in fps), sum(b for _, b in fps) + gap * (len(fps) - 1))


def _part_fp(d, name):
    """A part's footprint lying on its side in a nest, (length, width).
    The spring as bought, free length and base coil -- in the bat it is
    compressed, which is not how it arrives."""
    if name in d.printing:
        return d.printing[name]["footprint"]
    if name == "spring":
        return (CapSpring.L_FREE, CapSpring.D_BASE)
    y0, y1, z0, z1, _, x0, x1 = product.part_extent(d.parts[name])
    return (x1 - x0, max(y1 - y0, z1 - z0))


@dataclass
class Plan:
    """Everything the event model needs, and everything check_line holds."""
    line: Line
    bat: object
    d: object
    sts: dict
    floor: Floor
    rates: Rates
    bom: dict
    t_bat: dict          # station -> seconds of station time per bat, all jobs averaged in
    sizes: dict
    fits: list
    notes: list


def plan_line(d, line=None, drives=None):
    """Size the line for a derived bat.  `drives`: see stations()."""
    line = Line() if line is None else line
    sts, dg = stations(d, drives)
    R = rates(line)
    B = bom(d)
    p = line.service
    T_v = float(line.visit_every_days)                 # days between visits
    T_r = float(line.restock_every_days)
    k_tray = line.bats_per_tray
    run_h = line.hours_day
    sizes, fits, notes = {}, [], []

    def fit(name, margin, detail=""):
        fits.append(product.Fit(name, float(margin), detail))

    # ------------------------------------------- station time per bat
    st = sts
    # Sleeves come a visit's worth at a time, and a tray of frames goes
    # out against one tray of sleeves, so a visit's N sleeves make
    # ceil(N / k) passes, not N / k: every tray-borne cost is per pass.
    m_visit = R.starts["electronics"] * T_v
    per_pass = m_visit / mean_ceil(m_visit, k_tray) if m_visit > 0 else float(k_tray)
    t_bat = {
        "frame": st["frame"].job_s("build") + st["frame"].job_s("collapse"),
        "electronics": st["electronics"].job_s("bat") + st["electronics"].job_s("tray") / per_pass,
        "sleeve": st["sleeve"].job_s("bat") + st["sleeve"].job_s("tray") / per_pass,
        "calibration": st["calibration"].job_s("bat") + st["calibration"].job_s("tray") / per_pass
        + st["calibration"].job_s("swing") / Est.SWING_EVERY,
        "pack": st["pack"].job_s("bat") + st["pack"].job_s("tray") / per_pass,
    }
    sizes["bats_per_pass"] = per_pass
    # the frame station also starts a nozzle each line day at least, and
    # the calibration station runs the golden bat each line day
    day_s = {"frame": st["frame"].job_s("session"), "calibration": st["calibration"].job_s("day")}

    # -------------------------------------------- printed parts, the farm
    # A plate comes off only at a visit, and a printer works through
    # Printer.PLATES of them between two if it has the time.  The farm is
    # capacitated, so it is sized the capacitated way: printers for the
    # mean at Line.farm_load, the stock for the shortfall that leaves.
    printed = {"core": "frame", "tip_cap": "sleeve", "collar": "sleeve", "pommel_cap": "pack"}
    farm = {}
    for part, where in printed.items():
        pr = d.printing[part]
        per = pr["per_plate"]
        plate_s = per * pr["print_s"]
        lam = R.starts[where] * T_v                   # parts a period
        per_visit = max(1, min(Printer.PLATES, int(floor(T_v * 86400.0 / plate_s))))
        mean_plates = lam / (per * (1.0 - Faults.PRINT_FAIL))
        printers = max(1, int(ceil(mean_plates / per_visit / line.farm_load)))
        level, short = capacitated_stock(lam, printers * per_visit, per, Faults.PRINT_FAIL, p)
        farm[part] = {"per_plate": per, "plate_h": plate_s / 3600.0, "printers": printers,
                      "per_visit": per_visit,
                      "mean_printers": mean_plates / per_visit, "order_up_to": level,
                      "shortfall": short, "station": where,
                      # what the uncapacitated rule -- every visit refilled in
                      # full at the service level -- would have bought instead
                      "uncapacitated": int(ceil(poisson_q(lam, p) / (per * (1.0 - Faults.PRINT_FAIL))
                                                / per_visit))}
        fit("a plate of %ss prints between two visits" % part.replace("_", " "),
            T_v * 24.0 - plate_s / 3600.0, "%.1f h a plate of %d" % (plate_s / 3600.0, per))
    sizes["printers"] = farm

    # ---------------------------------------------------- frame pipeline
    t_build = st["frame"].job_s("build")
    t_coll = st["frame"].job_s("collapse")
    lam1 = R.starts["frame"]
    # the rate station 1 needs to win back the longest outage in one line day
    r_req = min(lam1 * (run_h + line.outage_h) / run_h / (run_h * 3600.0), 1.0 / (t_build + t_coll))
    mandrels = r_req * Epoxy.HANDLING_H * 3600.0 + 2 * k_tray        # curing, plus one tray filling, one emptying
    m_trays = int(ceil(mandrels / k_tray))
    # one visit's sleeves, drawn at once, and the frames station 1 is still
    # filling into a tray when they are: at most a tray less one
    stock = poisson_q(R.starts["electronics"] * T_v, p) + k_tray - 1
    sizes["mandrels"] = {"count": m_trays * k_tray, "trays": m_trays, "rate_per_h": r_req * 3600.0}
    sizes["frame_stock"] = {"frames": stock, "trays": int(ceil(stock / k_tray))}
    fit("station 1 rebuilds an average visit's frames and rides the longest outage, in an "
        "interval's line hours",
        T_v * run_h - (lam1 * T_v * (t_build + t_coll) / 3600.0 + line.outage_h),
        "%.1f frames at %.0f min and %.0f h out, in %.0f h" % (lam1 * T_v, (t_build + t_coll) / 60.0,
                                                               line.outage_h, T_v * run_h))

    # ------------------------------------------- station parts trays
    kits = {"frame": ["core"], "electronics": ["board", "contacts"], "sleeve": ["tip_cap", "collar"],
            "pack": ["pommel_cap", "spring"]}
    trays = {}
    tray_lw = (d.tray["length"], d.tray["width"])
    for where, names in kits.items():
        fps = [_part_fp(d, n) for n in names]
        per = _footprint_on_tray(_kit_footprint(fps), tray_lw)
        lam = R.starts[where]
        need = poisson_q(lam * T_v, p)
        n = int(ceil(need / max(per, 1))) + 1                      # one interval's full trays, and the one in use
        if where == "frame":                                       # the core store IS its trays: the farm's level
            v = farm["core"]                                       # and a plate's overshoot
            n = max(n, int(ceil((v["order_up_to"] + v["per_plate"] - 1) / max(per, 1))) + 1)
        trays[where] = {"kits": per, "trays": n, "parts": names}
        fit("the %s station's parts tray outlasts the longest outage" % where,
            per - lam * line.outage_h / run_h, "%d kits a tray, %.1f used in %.0f h"
            % (per, lam * line.outage_h / run_h, line.outage_h))
    sizes["parts_trays"] = trays

    # ------------------------------------------------- bought stores
    bought = {"board": "electronics", "contacts": "electronics", "spring": "pack",
              "tube": "pack", "tube_cap": "pack"}
    sizes["bought"] = {k: poisson_q(R.starts[w] * T_r, p) for k, w in bought.items()}

    # ---------------------------------------------------- consumables
    f = B["frame"]
    need1 = poisson_q(lam1 * T_v, p)                                 # frames in one interval
    cons = {
        "rods": {"unit": "magazine", "per": Rod.MAGAZINE, "use": need1 * f["rods"]},
        "thread_m": {"unit": "spool", "per": Thread.SPOOL_M, "use": need1 * f["thread_m"]},
        # every frame may start a new nozzle: the worst case the cartridges must cover
        "resin_ml": {"unit": "cartridge", "per": Epoxy.CARTRIDGE_ML,
                     "use": need1 * (f["resin_ml"] + Epoxy.PURGE_ML)},
        "nozzles": {"unit": "nozzle", "per": 1, "use": need1},
        "labels": {"unit": "roll", "per": Label.ROLL, "use": poisson_q(R.starts["pack"] * T_v, p)},
    }
    for v in cons.values():
        v["units"] = int(ceil(v["use"] / v["per"]))
    sizes["consumables"] = cons
    fit("the ring's groove holds a whole frame's thread",
        Ring.groove_capacity(d.frame.truss.thread_d) - f["thread_m"],
        "%.1f m a frame, %.1f m a groove" % (f["thread_m"], Ring.groove_capacity(d.frame.truss.thread_d)))

    # ------------------------------------------------- trays and racks
    q_ship = poisson_q(R.shipped * T_v, p)
    q_sleeve = poisson_q(R.sleeves * T_v, p)
    bat_trays = sizes["frame_stock"]["trays"] + 1 + 4 + 1   # stock, at station 1, in stations 2-5, one spare
    tube_trays = int(ceil(q_ship / k_tray)) + 2              # a visit's tubes, one at station 5, one empty
    stretcher_trays = int(ceil(q_sleeve / k_tray)) + 2       # a visit's sleeves, one at station 3, one back
    sizes["trays"] = {"bat": bat_trays, "tube": tube_trays, "stretcher": stretcher_trays,
                      "mandrel": m_trays, "parts": sum(v["trays"] for v in trays.values())}
    sizes["racks"] = {
        "inbound": sum(v["trays"] for v in trays.values()),
        "cure": m_trays,
        "stock": bat_trays,
        "sleeves": stretcher_trays,
        "outbound": tube_trays,
        "charger": 1}
    sizes["shelves"] = {"reject shelf": poisson_q(sum(R.rejects[k] for k in STATIONS[1:]) * T_v, p),
                        "reject bin": poisson_q(R.rejects["frame"] * T_v, p)}
    fl = floor_plan(sts, dg, sizes["racks"])

    # --------------------------------------------------------- the robot
    # A tray's every hop: two handovers and the trip, and the robot's run
    # to the tray before it.  Per tray of bats:
    hops = [("stock", "electronics/bats"), ("electronics/bats", "sleeve/bats"),
            ("sleeve/bats", "calibration/bats"), ("calibration/bats", "pack/bats"),
            ("pack/bats", "stock"),
            ("sleeves", "sleeve/stretchers"), ("sleeve/stretchers", "sleeves"),
            ("outbound", "pack/tubes"), ("pack/tubes", "outbound"),
            ("frame/frames", "stock"), ("stock", "frame/frames"),
            ("frame/mandrels A", "cure"), ("cure", "frame/mandrels B")]
    per_tray = sum(fl.trip_s(a, b) + 2.0 * Est.DOCK_S for a, b in hops)
    dead = sum(fl.trip_s(hops[i][1], hops[(i + 1) % len(hops)][0]) for i in range(len(hops)))
    parts_hops = sum(2.0 * (fl.trip_s("inbound", "%s/%s" % (w, "cores" if w == "frame" else "parts"))
                            + 2.0 * Est.DOCK_S) / trays[w]["kits"] for w in trays)
    t_bat["robot"] = (per_tray + dead) / per_pass + parts_hops
    sizes["robot_s_per_bat"] = t_bat["robot"]

    # --------------------------------------------------------- the person
    per_bat = {
        "sleeves": Person.SLEEVE_MIN * R.sleeves / R.shipped,
        "printers": Person.UNLOAD_MIN * sum(R.starts[v["station"]] / v["per_plate"] / (1 - Faults.PRINT_FAIL)
                                            for v in farm.values()) / R.shipped,
        "parts": Person.STOCK_MIN * sum(R.starts[w] for w in kits) / R.shipped / len(kits),
        "dispatch": Person.DISPATCH_MIN,
        "rejects": Person.REJECT_MIN * sum(R.rejects.values()) / R.shipped,
        "rods": (Person.CUT_ROD_MIN * f["rods"] * lam1 / R.shipped) if line.rods_by_hand else 0.0,
        # each unit a person changes, per bat shipped: what a bat uses over
        # what a unit holds, a nozzle magazine refilled once a visit, and at
        # least one nozzle's purge a line day
        "consumables": Person.CONSUMABLE_MIN * (
            f["rods"] * lam1 / R.shipped / Rod.MAGAZINE
            + f["thread_m"] * lam1 / R.shipped / Thread.SPOOL_M
            + (f["resin_ml"] * lam1 + Epoxy.PURGE_ML) / R.shipped / Epoxy.CARTRIDGE_ML
            + R.starts["pack"] / R.shipped / Label.ROLL
            + 1.0 / (T_v * R.shipped)),
    }
    sizes["person_min_per_bat"] = per_bat

    # ---------------------------------------- capacity: who saturates first
    avail = {k: run_h * 3600.0 * (1.0 - _down_frac(k)) for k in STATIONS + ("robot",)}
    cap = {}
    for k in STATIONS + ("robot",):
        busy = avail[k] - day_s.get(k, 0.0)
        per = t_bat[k] * (R.starts[k] / R.shipped if k in R.starts else 1.0)
        cap[k] = busy / per * line.days_month                # bats a month it can ship
    sizes["capacity_month"] = cap
    sizes["utilisation"] = {k: line.demand_month / cap[k] for k in cap}
    for k in cap:
        fit("%s keeps up with %.0f a month" % (k, line.demand_month), cap[k] - line.demand_month,
            "capacity %.0f a month, %.0f%% busy" % (cap[k], 100.0 * line.demand_month / cap[k]))

    # --------------------------------------------------- geometric fits
    fit("a tray spans the dock's rails, robot clearance included", dg.span_margin,
        "%s: %.0f mm against %.0f" % ("lengthways" if dg.along else "across", d.tray["length"] if dg.along
                                      else d.tray["width"], Robot.DECK[1] + 2.0 * (Dock.CLEAR + Dock.RAIL_W)))
    for k in STATIONS:
        mod = sts[k].module
        for f in mod.fits:
            fit("station %s's module: %s" % (k, f.name), f.margin, f.detail)
    notes.append("frame build %.1f min by the truss cell's planner on station 1's axes (its closed form "
                 "says %.1f on the cell's own)" % (frame_plan(d, sts["frame"].limits)["total_min"],
                                                  structure.cycle_estimate(d.frame.truss)))
    return Plan(line, d.bat, d, sts, fl, R, B, t_bat, sizes, fits, notes)


def _down_frac(k):
    """Fraction of running time a resource spends clearing its own faults."""
    if k == "robot":
        return Faults.ROBOT_MTTR_MIN / 60.0 / (Faults.ROBOT_MTBF_H + Faults.ROBOT_MTTR_MIN / 60.0)
    return Faults.STATION_MTTR_MIN / 60.0 / (Faults.STATION_MTBF_H + Faults.STATION_MTTR_MIN / 60.0)


# ================================================================ REPORT
def report(P):
    """The plan as text: times, layout, sizes, fits."""
    out = []
    w = out.append
    S = P.sizes
    w("LINE  %s bat at %.0f a month: takt %.1f min over %.0f h a day"
      % (P.bat.name, P.line.demand_month, P.line.takt_s / 60.0, P.line.hours_day))
    w("")
    w("station time per bat (each source: move / stroke / plan / est)")
    for k in STATIONS:
        st = P.sts[k]
        by = {}
        for job in st.jobs.values():
            for o in job:
                by[o.how] = by.get(o.how, 0.0) + o.s
        dr = "/".join(st.module.axes[a].drive.kind for a in "xyz")
        w("  %-12s %6.1f min  travel %4.0f x %4.0f x %3.0f mm (%s)  [%s]"
          % (k, P.t_bat[k] / 60.0, st.travel[0], st.travel[1], st.travel[2], dr,
             "  ".join("%s %.0f s" % kv for kv in sorted(by.items()))))
    w("  %-12s %6.1f min  (trips, handovers and runs to the next tray)" % ("robot", P.t_bat["robot"] / 60.0))
    w("")
    dg = P.floor.dg
    w("floor: aisle %.1f m long, %.2f m wide; docks %.2f m apart, %.2f m deep; bats lie %s the aisle"
      % (P.floor.length / 1000.0, dg.aisle / 1000.0, dg.pitch / 1000.0, dg.depth / 1000.0,
         "along" if dg.along else "across"))
    w("")
    w("capacity, bats a month (busy at %.0f)" % P.line.demand_month)
    for k, c in sorted(S["capacity_month"].items(), key=lambda kv: kv[1]):
        w("  %-12s %7.0f  %5.1f%%" % (k, c, 100.0 * S["utilisation"][k]))
    w("")
    w("printers: %d" % sum(v["printers"] for v in S["printers"].values()))
    for k, v in S["printers"].items():
        w("  %-10s %2d (the mean needs %.1f; refilling every visit in full, %d)  %d a plate, %.1f h a "
          "plate, stock up to %d" % (k, v["printers"], v["mean_printers"], v["uncapacitated"],
                                     v["per_plate"], v["plate_h"], v["order_up_to"]))
    m = S["mandrels"]
    w("mandrels: %d in %d trays, for %.2f frames an hour" % (m["count"], m["trays"], m["rate_per_h"]))
    w("frame stock: %d frames in %d trays" % (S["frame_stock"]["frames"], S["frame_stock"]["trays"]))
    w("parts trays: " + ", ".join("%s %d of %d kits" % (k, v["trays"], v["kits"])
                                  for k, v in S["parts_trays"].items()))
    w("bought stores (a restock's worth): " + ", ".join("%s %d" % kv for kv in S["bought"].items()))
    w("consumables per visit: " + ", ".join("%s %.0f (%d %s)" % (k, v["use"], v["units"], v["unit"])
                                           for k, v in S["consumables"].items()))
    w("trays: " + ", ".join("%s %d" % kv for kv in S["trays"].items()))
    w("racks: " + ", ".join("%s %d" % kv for kv in S["racks"].items()))
    w("shelves: " + ", ".join("%s %d" % kv for kv in S["shelves"].items()))
    pm = S["person_min_per_bat"]
    w("person: %.1f min a bat (%s)" % (sum(pm.values()), ", ".join("%s %.2f" % kv for kv in pm.items())))
    for n in P.notes:
        w("note: " + n)
    w("")
    for f in P.fits:
        w("  %s %-72s %8.2f  %s" % ("ok  " if f.ok else "FAIL", f.name[:72], f.margin, f.detail))
    return "\n".join(out)


if __name__ == "__main__":
    import argparse
    from .spec import KID
    ap = argparse.ArgumentParser(description="The default bat, derived, and the line sized for it: "
                                 "python3 -m batline.line [--demand N], from sim/")
    ap.add_argument("--demand", type=float, default=Line.demand_month, help="bats a month")
    a = ap.parse_args()
    bat = product.design(KID)
    print(product.report(bat))
    print(report(plan_line(bat, Line(demand_month=a.demand))))
