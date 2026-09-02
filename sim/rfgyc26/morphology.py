"""THE CHASSIS IS A DESIGN VARIABLE, NOT A CONSTRAINT.

Every other module in this project plans *within* robot 1's 285 x 235 mm
outline.  That is the wrong place to stop.  The board's tightest passages
are 80 mm, the chassis is 235 wide, and no planner fixes a body that does
not fit -- F153 is exactly that story, a robot declaring both kit zones
unreachable and then being told to shove the patients aside instead.

And the 235 is not a measurement of anything the robot needs.  It is an
assertion in params.py::

    ("the loaded envelope is still 235 wide (spec 4.2)",
     abs(2*(AgentA.POCKET_Y + Piece.BEAM_W/2) - AgentA.W) < 1e-9)

-- the chassis width is DEFINED to equal the width of two carried beams.
The beam-carrying subsystem was allowed to set the vehicle's outline.  That
is backwards: a beam is a 280 x 20 stick, and a stick can be carried by a
much smaller vehicle if it is allowed to overhang.

So this module asks the size question the way it should have been asked:

    WHAT MUST FIT INSIDE      the intake chain, the drive, the magazine --
                              a packing problem, from params, not a drawing
    WHAT MAY HANG OUTSIDE     the beams; they sweep, but they do not have
                              to be contained
    WHAT DOES THE BOARD ALLOW world.doors, station.stand_for, nav.plan --
                              the same three questions the mission asks

LOADED AND EMPTY ARE DIFFERENT ROBOTS.  A 280 mm beam in a flank pocket
makes a 200 mm chassis behave like a 280 mm one *while it is aboard*, which
is why the specification says the swept radius is "set by the beam ends at
185 regardless of chassis shape".  A design therefore has two footprints,
and which one applies depends on the leg: the beams are placed in the first
thirty seconds and the remaining ninety are flown empty.  A robot that is
only big while it is doing the thing that needs it big is a smaller robot
for three quarters of a match.

Pure numpy, nav, station and world.  No mujoco, no mission, no field
constants -- the modules and regions are arguments, so this runs on
somebody else's robot.
"""
from dataclasses import dataclass, field as _field

import numpy as np

from . import station, world


# ------------------------------------------------------------- the inside
@dataclass(frozen=True)
class Module:
    """Something that must fit INSIDE the chassis.

    `run` is how much length it needs along the vehicle's own axis and
    `half_w` how far out from the centreline, both in millimetres.  `datum`
    says what it is pinned to:

        "nose"   the intake -- it works at the front by definition
        "tail"   anything that must reach the back face
        "axle"   the drive
        "free"   slides along the axis; the packer places it

    A CHASSIS IS NOT A BOX WITH THINGS RATTLING IN IT.  The length of this
    vehicle is one conveyor line -- pick up at the nose, carry aft on a
    belt, magazine, drop through the floor -- so the things on that line
    are laid END TO END and their runs add.  Everything else (the drive
    under the belt, the battery over it, the camera mast above that) is at
    a different height and only spends width.  Set `stacks=True` for those:
    they contribute half_w but not run.
    """
    name: str
    run: float
    half_w: float
    datum: str = "free"
    stacks: bool = False


def packing(modules):
    """(minimum length, minimum half-width) for a set of modules.

    The length is the conveyor line's total run; the width is whatever
    sticks out furthest.  Both are lower bounds and both come from the
    parameters that describe the mechanisms, so a mechanism that grows
    grows the robot.
    """
    run = sum(m.run for m in modules if not m.stacks)
    half = max([m.half_w for m in modules] or [0.0])
    return float(run), float(half)


# ------------------------------------------------------------ the outside
@dataclass(frozen=True)
class Cargo:
    """Something that rides on the chassis and MAY hang off it."""
    name: str
    length: float
    width: float
    at: tuple = (0.0, 0.0)        # body-frame centre
    along: float = 0.0            # degrees from the body's +x axis

    def hull(self):
        a = np.radians(self.along)
        c, s = np.cos(a), np.sin(a)
        return [(self.at[0] + dx*c - dy*s, self.at[1] + dx*s + dy*c)
                for dx in (-self.length/2.0, self.length/2.0)
                for dy in (-self.width/2.0, self.width/2.0)]


@dataclass
class Design:
    """A chassis, what is packed in it, what rides on it, what it throws."""
    name: str
    length: float
    width: float
    modules: tuple = ()
    cargo: tuple = ()
    effectors: dict = _field(default_factory=dict)
    shell: float = 8.0            # skin and clearance, total across the width

    # ------------------------------------------------------------ packing
    @property
    def packs(self):
        """Does what has to be inside actually fit inside?"""
        run, half = packing(self.modules)
        return (self.length + 1e-9 >= run
                and self.width + 1e-9 >= 2.0*half + self.shell)

    @property
    def slack(self):
        """(length, width) to spare over the packing minimum -- negative
        means it does not fit, which is more useful than a bare False."""
        run, half = packing(self.modules)
        return (self.length - run, self.width - (2.0*half + self.shell))

    # ------------------------------------------------------------- shapes
    def _pts(self, loaded):
        fwd, hw = self.length/2.0, self.width/2.0
        pts = [(fwd, hw), (fwd, -hw), (-fwd, hw), (-fwd, -hw),
               (fwd, 0.0), (-fwd, 0.0), (0.0, hw), (0.0, -hw), (0.0, 0.0)]
        if loaded:
            for c in self.cargo:
                pts += c.hull()
        return tuple(pts)

    def footprint(self, loaded=True):
        """The chassis as a planner sees it.

        The inscribed radius is the half-WIDTH, not half the smaller
        dimension: it is the radius that is safe at every heading, and a
        differential drive turns.  For a loaded design the cargo may be
        longer than the body, so take the narrowest half-extent over the
        hull rather than assuming the body sets it.
        """
        pts = self._pts(loaded)
        hw = max(abs(y) for _x, y in pts)
        hl = max(abs(x) for x, _y in pts)
        return station.Footprint(
            pts, min(hw, hl),
            float(max(np.hypot(x, y) for x, y in pts)))

    def swept(self, loaded=True):
        return self.footprint(loaded).circumscribed

    @property
    def area(self):
        return self.length * self.width


# ------------------------------------------------------------ evaluation
@dataclass
class Verdict:
    design: str
    packs: bool
    slack: tuple
    fits: dict                  # door -> bool (empty body)
    stands: dict                # region -> margin, None = nowhere to stand
    reaches: dict               # region -> seconds, None = no route
    clean: dict                 # region -> True if reachable touching nothing
    swept_loaded: float
    swept_empty: float
    area: float

    @property
    def ok(self):
        return (self.packs
                and all(self.fits.values())
                and all(v is not None for v in self.stands.values())
                and all(v is not None for v in self.reaches.values()))

    @property
    def shoves(self):
        """How many of its destinations this body can only reach by driving
        THROUGH a loose piece.

        THIS IS WHAT WIDTH COSTS, in a number.  The board's own furniture
        leaves plenty of room; the twelve patients leave 80 mm gaps, and a
        body that does not fit one has to shove.  Shoving is legal (F153 --
        4 kg against 5 g) and it is how this robot has always got to the kit
        zones, but every shove moves a patient robot 2 is scored on, so a
        design that never needs to is strictly better than one that does.
        """
        return sum(1 for v in self.clean.values() if not v)

    @property
    def cost(self):
        """Seconds of driving the board makes this design pay.  Infinite if
        it cannot do the job -- so smaller-is-better ranking never picks a
        design that does not work."""
        if not self.ok:
            return float("inf")
        return float(sum(self.reaches.values()))

    def __repr__(self):
        return ("<%s %.0fx%.0f  %s  swept %.0f/%.0f  %s>"
                % (self.design, 0, 0, "ok" if self.ok else "NO",
                   self.swept_empty, self.swept_loaded,
                   "%.1f s" % self.cost if self.ok else "-"))


def evaluate(design, cmap, regions, doors=(), start=None, loaded=False,
             headings=16, res=20.0, speed=230.0, strict=None):
    """Can this design do the job this board asks of it?

    regions  {name: (rect, effector)} it must deliver into
    doors    [(gate, width), ...] from world.doors on the CLEARED board --
             the field's own passages, which every design must use
    strict   the same board with the loose pieces as walls.  Optional, and
             it decides nothing: it only measures how many destinations the
             body is too big to reach without shoving something.
    loaded   plan the delivery legs with the cargo aboard.  False is the
             honest default for robot 1: the beams are placed first and
             every kit leg is flown empty.

    WHAT IS A GATE AND WHAT IS A PRICE.  The first version of this failed
    every design against the 80 mm gaps the patients leave, which is true
    and useless -- no robot that can carry a 280 mm beam fits an 80 mm gap,
    so the search had no feasible region at all.  The board's WALLS are a
    gate because nothing moves them; the board's PIECES are a price,
    because a robot that does not fit between two of them drives through
    one.  Mixing those up is the same error F153 was, in the other
    direction.
    """
    foot = design.footprint(loaded=loaded)
    empty = design.footprint(loaded=False)
    start = tuple(start) if start is not None else (cmap.w/2.0, cmap.h*0.13)

    # A DOOR'S WIDTH IS A RADIUS, NOT A GAP.  world.doors reports the
    # CLEARANCE at the gate -- the distance from a centre to the nearest
    # obstacle -- so the body that fits is the one whose inscribed radius is
    # no larger.  Doubling it (a gap) is off by two in the direction that
    # rejects every design, which is how this file first concluded that
    # nothing at all could work.
    fits = {}
    for k, (gate, width) in enumerate(doors):
        fits["door%d@%.0f,%.0f" % (k, gate[0], gate[1])] = \
            bool(empty.inscribed <= width)

    stands, reaches, clean = {}, {}, {}
    for name, (rect, eff) in regions.items():
        got = station.stand_for(rect, eff, foot, cmap, headings=headings,
                                res=res, need_clear=-np.inf, top=6,
                                prefer=start)
        stands[name] = got[0].margin if got else None
        reaches[name] = None
        clean[name] = False
        for st in got:
            _p, secs = world.route_to(cmap, start, st.pose, foot, speed=speed)
            if np.isfinite(secs):
                reaches[name] = float(secs)
                if strict is not None:
                    _q, t = world.route_to(strict, start, st.pose, foot,
                                           speed=speed)
                    clean[name] = bool(np.isfinite(t))
                break
    return Verdict(design.name, design.packs, design.slack, fits, stands,
                   reaches, clean, design.swept(True), design.swept(False),
                   design.area)


def sweep(designs, cmap, regions, doors=(), start=None, **kw):
    """Evaluate a family; feasible first, then smallest."""
    out = [evaluate(d, cmap, regions, doors=doors, start=start, **kw)
           for d in designs]
    out.sort(key=lambda v: (not v.ok, v.area))
    return out


def shrink(make, cmap, regions, doors=(), start=None,
           lengths=None, widths=None, **kw):
    """The smallest feasible chassis in a family, and what it cost to find.

    `make(length, width) -> Design`.  This is a search over the two
    dimensions the vehicle actually has, not a list somebody typed, so it
    answers again when the field or the payload changes.

    Returns (best Verdict, best Design, [(design, verdict), ...]) with the
    whole grid kept -- the shape of the feasible region is worth more than
    its corner, because it says WHICH dimension is binding.
    """
    lengths = np.asarray(lengths if lengths is not None
                         else np.arange(180.0, 300.0, 20.0), float)
    widths = np.asarray(widths if widths is not None
                        else np.arange(120.0, 260.0, 20.0), float)
    grid, best, bestd = [], None, None
    for L in lengths:
        for W in widths:
            d = make(float(L), float(W))
            if not d.packs:                 # cheap, and it rejects most
                v = Verdict(d.name, False, d.slack, {}, {}, {}, {},
                            d.swept(True), d.swept(False), d.area)
            else:
                v = evaluate(d, cmap, regions, doors=doors, start=start, **kw)
            grid.append((d, v))
            if v.ok and (best is None or v.area < best.area):
                best, bestd = v, d
    return best, bestd, grid
