"""Station B: the sub-assembly that makes the camera's head kit.

THE PROBLEM IT EXISTS FOR.  The mount's four tic-tac-toe rods are WOUND to
each other at four crossings -- the same thread the truss's joints use, so
the joint is a joint and not a dab of resin on a tangent.  The main cell
cannot make those joints.  Its ring is 40 mm across in a raceway 54 mm
across, and by the time the mount goes on, the crossings are inside the end
triangle with the camera behind them.  There is no approach, and no
clearance argument produces one.

So the crossings are wound before the camera ever reaches the truss, on a
station whose only job is that one part:

    the module goes into a nest, lens UP, by hand
    B lays the COLLAR over the lens housing and bonds its four walls
    B lays the four GRID rods on it, two layers, two crossings each
    B WINDS each of the four crossings
    B doses each crossing and the collar's aperture
    the finished kit comes out by hand

WHAT MAKES IT THE RIGHT PART.  Every dimension here comes from `Bracket`
and `Payload` -- the same numbers `mount.solve` builds the nose from -- so
the kit B makes is the kit A expects to pick up.  `check_stationb` asserts
that, crossing by crossing, against `mount.solve`'s own landings.

THE FRAME.  B works in its own coordinates and the module lies flat in it:

    B's x  <-  the camera's x (which becomes the truss's spine axis)
    B's y  <-  the camera's `up`
    B's z  <-  the camera's `look` -- so the lens points UP at the head

That is the one orientation in which a three-axis gantry can reach all four
crossings and lay all four rods without turning the part over.

WHY THE WINDER NEEDS NO ROTARY.  A hoop laid in the plane perpendicular to
one rod of a crossing encircles BOTH of them, because the other rod passes
through that plane at the crossing.  So one ring axis -- along the lower
layer -- serves all four crossings, and B has no fixture axis at all.
"""
from dataclasses import dataclass
from math import sqrt

import numpy as np

from .spec import Bracket, Payload, StationB, Stock, Gripper, Head, Process


@dataclass(frozen=True)
class Rod:
    """One rod of the kit, in B's frame (mm)."""
    kind:  str            # "collar" | "grid"
    index: int
    p0:    np.ndarray
    p1:    np.ndarray
    r:     float
    layer: int = 0

    @property
    def axis(self):
        v = self.p1 - self.p0
        return v / np.linalg.norm(v)

    @property
    def length(self):
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def mid(self):
        return (self.p0 + self.p1) / 2.0


class Kit:
    """The head kit: what B makes and A picks up.

    Built in B's frame with the module's centre at the origin, so the
    module's board lies in z = 0 and the lens looks up +z.
    """

    def __init__(self, payload=None, d_rod=1.5, clear=None):
        self.p = Payload if payload is None else payload
        self.d = float(d_rod)
        self.clear = clear
        self.gx, self.gu = Bracket.grid_half(self.p, self.d, clear)
        self.px, self.pu = Bracket.plate_half(self.p, self.d, clear)
        self.l0 = Bracket.layer_l(0, self.p, self.d)
        self.l1 = Bracket.layer_l(1, self.p, self.d)
        self.seat = Bracket.seat_l(self.p)

    # ------------------------------------------------------------ pieces
    def collar_z(self):
        """(lo, hi) the collar plate occupies in B's z, mm."""
        return self.seat

    @property
    def rods(self):
        """The four grid rods, in the order they are laid.

        LAYER 0 FIRST, and not by preference: layer 1 lies ON layer 0 at
        every crossing.  Laid the other way round the second pair has
        nothing under it but the collar's own aperture."""
        over = Bracket.OVERRUN
        out = []
        for su in (+1.0, -1.0):                      # layer 0, along B's x
            out.append(Rod("grid", len(out),
                           np.array([-self.gx - over, su * self.gu, self.l0]),
                           np.array([+self.gx + over, su * self.gu, self.l0]),
                           self.d / 2.0, 0))
        for sx in (+1.0, -1.0):                      # layer 1, along B's y
            out.append(Rod("grid", len(out),
                           np.array([sx * self.gx, -self.gu - over, self.l1]),
                           np.array([sx * self.gx, +self.gu + over, self.l1]),
                           self.d / 2.0, 1))
        return tuple(out)

    def crossings(self):
        """The four wound joints, in B's frame.  Same points `mount.solve`
        lands its struts on -- that is the whole contract between the two
        machines."""
        lx = 0.5 * (self.l0 + self.l1)
        return tuple(np.array([sx * self.gx, su * self.gu, lx])
                     for sx in (+1.0, -1.0) for su in (+1.0, -1.0))

    def wind_axis(self, i):
        """Which way the winder's bore points at crossing i: along the
        LOWER rod, so the hoop encircles both."""
        return np.array([1.0, 0.0, 0.0])

    def hoop_perimeter(self):
        """One turn of thread round a crossing, mm.

        Two rods at right angles, one lying on the other: the hoop is the
        convex hull of the pair in the plane it is laid in -- two half-circles
        of the rod, joined across the stack."""
        return float(np.pi * self.d + 2.0 * self.d)

    def thread_len(self, turns=None):
        n = StationB.TURNS if turns is None else turns
        return 4.0 * n * self.hoop_perimeter()

    # ------------------------------------------------------- the racks
    def rack_slots(self):
        """Where the four rods and the collar wait, in B's frame.

        Beside the nest on the -y side, on the rods' own pitch, laid along
        B's x so the gripper takes them at yaw zero and never has to turn a
        rod it is carrying over the part."""
        out = []
        y0 = -(self.pu + StationB.NEST_WALL + 12.0)
        for i, r in enumerate(self.rods):
            y = y0 - i * 11.0
            hl = r.length / 2.0
            out.append(("grid", i, np.array([-hl, y, self.l0]),
                        np.array([+hl, y, self.l0])))
        y = y0 - 4 * 11.0 - 12.0
        out.append(("collar", 0, np.array([-self.px, y, self.seat[0]]),
                    np.array([+self.px, y, self.seat[0]])))
        return out

    def extent(self):
        """(x, y, z) half-extents of everything B holds, mm -- what its
        gantry has to cover."""
        xs, ys, zs = [self.px], [self.pu], [self.seat[1], self.l1 + self.d]
        for r in self.rods:
            xs += [abs(r.p0[0]), abs(r.p1[0])]
            ys += [abs(r.p0[1]), abs(r.p1[1])]
        for _k, _i, a, b in self.rack_slots():
            xs += [abs(a[0]), abs(b[0])]
            ys += [abs(a[1]), abs(b[1])]
        return (max(xs), max(ys), max(zs))
