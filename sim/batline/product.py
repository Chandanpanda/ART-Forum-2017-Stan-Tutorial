"""The bat, derived from a Bat: every part, every fit, every mass.

NOTHING HERE IS A DRAWING.  A Bat gives an envelope -- length, handle,
grip, width, depth -- and every dimension inside it is solved from the
facts in spec.py: the bay from the IEC cell, the slot from the board's
drawing, the stack's length from the spring's working range, the frame's
truss from the rated swing and the machine that has to wind it, the sleeve
blank from the knit's stretch window, the bands' zones from where the
sleeve can be seen, the tube from the widest section.  A second size is a
second Bat, not a second drawing; check_bat runs one it has never seen.

FRAME.  x along the bat from the pommel end (0) to the toe; z toward the
back of the blade, so the face is down when the bat lies in a cradle; y
completes the right-handed frame.  The handle, the shoulder and the
frame's truss share the x axis.  The truss is truss.geometry's, moved
along x: chord 0 is the back's apex, chords 1 and 2 bound the face.

WHAT IS STILL A SIMPLIFICATION, so nobody mistakes it for a result:
  * the shoulder is a plate as thick as its sockets plus a floor; whether
    that plate carries the swing's moment into the handle is M5's
    (check_frame), not this module's;
  * the frame is sized against the rated swing as a cantilever from the
    shoulder (strength: chord buckling), and the bend that swing puts on
    the sweet spot is REPORTED, not ruled -- including the printed
    handle's share, which this module estimates and M5 must settle;
  * the tip cap's press fit is the coupon's number; whether it holds is
    the sleeve's job, since the sleeve's closed tip covers the cap.

MASSES.  Every part is a list of elements -- prisms of constant section,
rods, boxes, point masses -- and its mass properties are integrated from
them.  A section is a union of shapes less a union of holes, integrated on
a grid; its analytic area is kept beside it so the integrator is itself
checked.  Printed parts weigh what a slicer would lay down: walls solid,
the interior at the infill fraction.

numpy only.
"""
from dataclasses import dataclass, field
from functools import lru_cache
from math import pi, sqrt, tan, radians, asin, log, exp, floor, cos, sin

import numpy as np

from truss.spec import Truss, Head, Process as TrussProcess, Stock, Gantry
from truss import structure
from truss.geometry import TrussGeometry, chord_phi, radial
from .spec import (Bat, Cell, Print, Printer, Board, ContactSet, CapSpring, Knit,
                   Tube, Rules, Line)
from . import imu_fusion_sim as study

RES = 0.1          # mm, the grid a section is integrated on
SQ3 = sqrt(3.0)


# ================================================================ SECTIONS
def _tri_vertices(R):
    """(y, z) of the chord axes: the truss's own azimuths."""
    return [tuple(R * radial(chord_phi(k))[1:]) for k in range(3)]


def _dist_to_triangle(Y, Z, V):
    """Distance from each point to the FILLED triangle V, zero inside."""
    def cross(a, b, y, z):
        return (b[0] - a[0]) * (z - a[1]) - (b[1] - a[1]) * (y - a[0])
    c = [cross(V[i], V[(i + 1) % 3], Y, Z) for i in range(3)]
    inside = ((c[0] >= 0) & (c[1] >= 0) & (c[2] >= 0)) | ((c[0] <= 0) & (c[1] <= 0) & (c[2] <= 0))
    d = np.full(np.shape(Y), np.inf)
    for i in range(3):
        a, b = np.array(V[i]), np.array(V[(i + 1) % 3])
        ab = b - a
        tt = np.clip(((Y - a[0]) * ab[0] + (Z - a[1]) * ab[1]) / (ab @ ab), 0.0, 1.0)
        d = np.minimum(d, np.hypot(Y - (a[0] + tt * ab[0]), Z - (a[1] + tt * ab[1])))
    return np.where(inside, 0.0, d)


@dataclass(frozen=True)
class Shape:
    """A circle (cy, cz, r), a rectangle (y0, y1, z0, z1) or a ROUNDED
    TRIANGLE (R, rc): every point within rc of the triangle whose vertices
    are the chord axes at radius R.  That last one is the section of
    everything that wraps the frame -- its hull, the shoulder, the collar,
    the tip cap, the sleeve -- each an offset of the one before."""
    kind: str
    p: tuple

    def mask(self, Y, Z):
        if self.kind == "circle":
            cy, cz, r = self.p
            return (Y - cy) ** 2 + (Z - cz) ** 2 <= r * r
        if self.kind == "rect":
            y0, y1, z0, z1 = self.p
            return (Y >= y0) & (Y <= y1) & (Z >= z0) & (Z <= z1)
        R, rc = self.p
        return _dist_to_triangle(Y, Z, _tri_vertices(R)) <= rc

    @property
    def area(self):
        if self.kind == "circle":
            return pi * self.p[2] ** 2
        if self.kind == "rect":
            y0, y1, z0, z1 = self.p
            return (y1 - y0) * (z1 - z0)
        R, rc = self.p
        s = SQ3 * R
        return SQ3 / 4.0 * s * s + 3.0 * s * rc + pi * rc * rc

    @property
    def perimeter(self):
        if self.kind == "circle":
            return 2.0 * pi * self.p[2]
        if self.kind == "rect":
            y0, y1, z0, z1 = self.p
            return 2.0 * ((y1 - y0) + (z1 - z0))
        R, rc = self.p
        return 3.0 * SQ3 * R + 2.0 * pi * rc

    def bbox(self):
        if self.kind == "circle":
            cy, cz, r = self.p
            return (cy - r, cy + r, cz - r, cz + r)
        if self.kind == "rect":
            return self.p
        R, rc = self.p
        return (-SQ3 * R / 2.0 - rc, SQ3 * R / 2.0 + rc, -R / 2.0 - rc, R + rc)

    def corners(self):
        """(k, 2) mm, (y, z): the convex polygon this shape is grown from by
        its rounding -- a circle's centre, a rectangle's four corners, a
        rounded triangle's chord axes -- so its extent along any direction
        is theirs, widened by the rounding on each side."""
        if self.kind == "circle":
            return np.array([self.p[:2]], float)
        if self.kind == "rect":
            y0, y1, z0, z1 = self.p
            return np.array([(y0, z0), (y1, z0), (y1, z1), (y0, z1)], float)
        return np.array(_tri_vertices(self.p[0]), float)

    def r_max(self):
        """The farthest point from the bat's axis."""
        if self.kind == "circle":
            cy, cz, r = self.p
            return sqrt(cy * cy + cz * cz) + r
        if self.kind == "rect":
            y0, y1, z0, z1 = self.p
            return max(sqrt(y * y + z * z) for y in (y0, y1) for z in (z0, z1))
        return self.p[0] + self.p[1]

    def offset(self, d):
        """Grown by d (shrunk if negative).  A rectangle's offset is kept
        square-cornered: close enough for the stiffness it is used for."""
        if self.kind == "circle":
            cy, cz, r = self.p
            return circle(cy, cz, max(r + d, 0.0))
        if self.kind == "rect":
            y0, y1, z0, z1 = self.p
            return rect(y0 - d, y1 + d, z0 - d, z1 + d)
        R, rc = self.p
        return rtri(R, max(rc + d, 0.0))


def circle(cy, cz, r):
    return Shape("circle", (float(cy), float(cz), float(r)))


def rect(y0, y1, z0, z1):
    return Shape("rect", (float(y0), float(y1), float(z0), float(z1)))


def rtri(R, rc):
    return Shape("rtri", (float(R), float(rc)))


def rtri_inradius(R, rc):
    """Axis to a flat face of a rounded triangle."""
    return R / 2.0 + rc


def rtri_circumradius(R, rc):
    """Axis to the farthest point: a corner."""
    return R + rc


@dataclass(frozen=True)
class Section:
    """One solid shape less any holes inside it."""
    solid: Shape
    holes: tuple = ()

    def mask(self, Y, Z):
        m = self.solid.mask(Y, Z)
        for h in self.holes:
            m &= ~h.mask(Y, Z)
        return m

    @property
    def area_exact(self):
        """Analytic area -- valid only while every hole lies inside the
        solid and no two holes overlap.  The core's holes DO overlap (a
        groove runs into the bay), so only the frame search's estimate uses
        this, on the shoulder, tip cap and collar; check_bat asserts the
        precondition on the grid for those."""
        return self.solid.area - sum(h.area for h in self.holes)

    @property
    def perimeter(self):
        return self.solid.perimeter + sum(h.perimeter for h in self.holes)

    def bbox(self):
        return self.solid.bbox()


@lru_cache(maxsize=None)
def moments(sec):
    """(A, Sy, Sz, Jyy, Jzz, Jyz) of a section about the bat's axis, on a
    RES grid: area, first moments, and the integrals of y^2, z^2 and yz."""
    y0, y1, z0, z1 = sec.bbox()
    ys = np.arange(y0 + RES / 2.0, y1, RES)
    zs = np.arange(z0 + RES / 2.0, z1, RES)
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    m = sec.mask(Y, Z)
    dA = RES * RES
    y, z = Y[m], Z[m]
    return (m.sum() * dA, y.sum() * dA, z.sum() * dA,
            (y * y).sum() * dA, (z * z).sum() * dA, (y * z).sum() * dA)


def printed_I(sec):
    """The lesser principal second moment (mm^4) of a section printed the
    way a slicer builds it -- walls solid, the interior at the infill
    fraction -- about its own centroid.  Counting infill at its share of
    the material is OPTIMISTIC (a sparse lattice bends less than its share
    of the mass), so this is an upper bound on the stiffness."""
    inner = Section(sec.solid.offset(-Print.WALL), tuple(h.offset(Print.WALL) for h in sec.holes))
    o = np.array(moments(sec)) - (1.0 - Print.INFILL) * np.array(moments(inner))
    A, Sy, Sz, Jyy, Jzz, Jyz = o
    Iyy, Izz, Iyz = Jzz - Sz * Sz / A, Jyy - Sy * Sy / A, Jyz - Sy * Sz / A
    return float((Iyy + Izz) / 2.0 - sqrt(((Iyy - Izz) / 2.0) ** 2 + Iyz ** 2))


# ============================================================ MASS PROPS
class MassProps:
    """Mass (g), centre (mm) and inertia about that centre (g mm^2)."""

    def __init__(self, m=0.0, c=(0.0, 0.0, 0.0), I=None):
        self.m = float(m)
        self.c = np.asarray(c, float)
        self.I = np.zeros((3, 3)) if I is None else np.asarray(I, float)

    def about(self, p):
        """Inertia about a point p, by the parallel-axis theorem."""
        d = self.c - np.asarray(p, float)
        return self.I + self.m * ((d @ d) * np.eye(3) - np.outer(d, d))

    def __add__(self, o):
        m = self.m + o.m
        if m <= 0.0:
            return MassProps()
        c = (self.m * self.c + o.m * o.c) / m
        return MassProps(m, c, self.about(c) + o.about(c))

    def __radd__(self, o):
        return self if o == 0 else self.__add__(o)


def prism_props(x0, x1, sec, rho):
    A, Sy, Sz, Jyy, Jzz, Jyz = moments(sec)
    L = x1 - x0
    m = rho * A * L
    if m <= 0.0:
        return MassProps()
    x3 = (x1 ** 3 - x0 ** 3) / 3.0
    x2 = (x1 ** 2 - x0 ** 2) / 2.0
    I0 = rho * np.array([[L * (Jyy + Jzz), -x2 * Sy, -x2 * Sz],
                         [-x2 * Sy, A * x3 + L * Jzz, -L * Jyz],
                         [-x2 * Sz, -L * Jyz, A * x3 + L * Jyy]])
    c = np.array([(x0 + x1) / 2.0, Sy / A, Sz / A])
    return MassProps(m, c, I0 - m * ((c @ c) * np.eye(3) - np.outer(c, c)))


def rod_props(p0, p1, r, m):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0)
    u = (p1 - p0) / L
    Ip = m * (3.0 * r * r + L * L) / 12.0
    Ia = m * r * r / 2.0
    return MassProps(m, (p0 + p1) / 2.0, Ip * (np.eye(3) - np.outer(u, u)) + Ia * np.outer(u, u))


def box_props(c, half, m):
    a, b, h = half
    return MassProps(m, c, m / 3.0 * np.diag([b * b + h * h, a * a + h * h, a * a + b * b]))


def rod_volume(r, p0, p1):
    return pi * r * r * float(np.linalg.norm(np.asarray(p1, float) - np.asarray(p0, float)))


# ================================================================== PARTS
@dataclass
class Part:
    """One part of the bat, as elements in the bat frame.

    prisms  (x0, x1, Section): material of constant section along x
    rods    (p0, p1, r, m):    solid cylinders carrying their own mass
    boxes   (centre, half, m): axis-aligned in the bat frame
    points  (p, m):            masses with no size worth modelling
    `rho` is the prisms' density; a printed part's is its EFFECTIVE density,
    walls and infill together."""
    name: str
    source: str
    rgba: tuple
    prisms: list = field(default_factory=list)
    rods: list = field(default_factory=list)
    boxes: list = field(default_factory=list)
    points: list = field(default_factory=list)
    rho: float = 0.0

    @property
    def volume(self):
        return sum(moments(s)[0] * (x1 - x0) for x0, x1, s in self.prisms)

    @property
    def props(self):
        out = MassProps()
        for x0, x1, s in self.prisms:
            out = out + prism_props(x0, x1, s, self.rho)
        for p0, p1, r, m in self.rods:
            out = out + rod_props(p0, p1, r, m)
        for c, h, m in self.boxes:
            out = out + box_props(c, h, m)
        for p, m in self.points:
            out = out + MassProps(m, p)
        return out

    @property
    def mass(self):
        return self.props.m


def part_extent(part):
    """(y_min, y_max, z_min, z_max, r_max, x_min, x_max) of every element of
    a part: its envelope, measured, not formulated."""
    ys, zs, rs, xs = [], [], [], []
    for x0, x1, sec in part.prisms:
        y0, y1, z0, z1 = sec.bbox()
        ys += [y0, y1]
        zs += [z0, z1]
        rs.append(sec.solid.r_max())
        xs += [x0, x1]
    for p0, p1, r, _ in part.rods:
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        u = (p1 - p0) / np.linalg.norm(p1 - p0)
        e = r * np.sqrt(np.clip(1.0 - u * u, 0.0, 1.0))     # a disc's reach along each axis
        for q in (p0, p1):
            xs += [q[0] - e[0], q[0] + e[0]]
            ys += [q[1] - e[1], q[1] + e[1]]
            zs += [q[2] - e[2], q[2] + e[2]]
            rs.append(sqrt(q[1] ** 2 + q[2] ** 2) + float(np.hypot(e[1], e[2])))
    for c, h, _ in part.boxes:
        ys += [c[1] - h[1], c[1] + h[1]]
        zs += [c[2] - h[2], c[2] + h[2]]
        rs.append(sqrt((abs(c[1]) + h[1]) ** 2 + (abs(c[2]) + h[2]) ** 2))
        xs += [c[0] - h[0], c[0] + h[0]]
    for p, _ in part.points:
        ys.append(p[1])
        zs.append(p[2])
        rs.append(sqrt(p[1] ** 2 + p[2] ** 2))
        xs.append(p[0])
    return (min(ys), max(ys), min(zs), max(zs), max(rs), min(xs), max(xs))


RES_SHELL = 0.2       # mm, the grid the walls are found on (a wall is 6 cells)


def _grid_masks(secs, res):
    """Masks of several sections on one common grid, padded by a wall so an
    erosion never runs off its edge."""
    pad = Print.WALL + 2.0 * res
    bb = np.array([s_.bbox() for s_ in secs])
    ys = np.arange(bb[:, 0].min() - pad, bb[:, 1].max() + pad, res)
    zs = np.arange(bb[:, 2].min() - pad, bb[:, 3].max() + pad, res)
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    return [s_.mask(Y, Z) for s_ in secs]


def _erode_octagon(m, n):
    """m less everything within about n cells of its boundary: 4- and
    8-neighbour erosions in turn, an octagon within 8 % of a circle."""
    e = m.copy()
    for i in range(n):
        f = e.copy()
        f[1:, :] &= e[:-1, :]
        f[:-1, :] &= e[1:, :]
        f[:, 1:] &= e[:, :-1]
        f[:, :-1] &= e[:, 1:]
        if i % 2:
            f[1:, 1:] &= e[:-1, :-1]
            f[:-1, :-1] &= e[1:, 1:]
            f[1:, :-1] &= e[:-1, 1:]
            f[:-1, 1:] &= e[1:, :-1]
        e = f
    return e


@lru_cache(maxsize=None)
def wall_fraction(sec):
    """The share of a section a slicer lays as walls: everything within a
    wall's thickness of ANY boundary -- the outside, a hole, the union of
    holes that overlap (a groove into the bay, a window into the pocket),
    which a sum of perimeters would count twice."""
    m, = _grid_masks([sec], RES_SHELL)
    n = int(round(Print.WALL / RES_SHELL))
    tot = int(m.sum())
    return (tot - int(_erode_octagon(m, n).sum())) / tot if tot else 0.0


def printed_mass(prisms):
    """Grams a slicer lays down for these prisms: walls solid, the rest at
    the infill fraction.  The walls are each section's wall_fraction along
    its length, and a wall's thickness of skin over every face -- the two
    ends and wherever one section steps to the next (the cells the two
    sections do not share)."""
    prisms = sorted(prisms, key=lambda q: q[0])
    V = sum(moments(s_)[0] * (x1 - x0) for x0, x1, s_ in prisms)
    shell = sum(moments(s_)[0] * wall_fraction(s_) * (x1 - x0) for x0, x1, s_ in prisms)
    faces = moments(prisms[0][2])[0] + moments(prisms[-1][2])[0]
    for (_, xa, sa), (xb, _, sb) in zip(prisms[:-1], prisms[1:]):
        if abs(xa - xb) > 1e-6:                  # not touching: two more free ends
            faces += moments(sa)[0] + moments(sb)[0]
            continue
        ma, mb = _grid_masks([sa, sb], RES_SHELL)
        faces += float((ma ^ mb).sum()) * RES_SHELL ** 2
    shell = min(V, shell + faces * Print.WALL)
    return Print.RHO * (shell + Print.INFILL * (V - shell)), V


def printed_mass_exact(prisms):
    """The same, on the analytic areas: the design loop's version, which
    cannot afford a grid per candidate."""
    prisms = sorted(prisms, key=lambda q: q[0])
    V = sum(s.area_exact * (x1 - x0) for x0, x1, s in prisms)
    S = sum(s.perimeter * (x1 - x0) for x0, x1, s in prisms)
    areas = [s.area_exact for _, _, s in prisms]
    S += areas[0] + areas[-1] + sum(abs(a - b) for a, b in zip(areas[1:], areas[:-1]))
    shell = min(V, S * Print.WALL)
    return Print.RHO * (shell + Print.INFILL * (V - shell))


def printed(name, rgba, prisms):
    m, V = printed_mass(prisms)
    return Part(name, "printed", rgba, prisms=list(prisms), rho=m / V)


# ============================================================ THE SWING
class SwingLoad:
    """The rated swing's specific force along the bat.

    The swing is the accuracy study's own (imu_fusion_sim.swings()), so the
    bat is sized against the swing its accuracy was estimated for.  The
    study's bat frame starts at the HANDS; a bat's hands are the middle of
    its grip, which puts the study's IMU 0.10 m toward the pommel from them,
    where this bat's board is.  For a gram at station x[k], in m/s^2 with
    gravity included:

        f[t, k]       the push ACROSS the bat (3-vector, world frame)
        along[t, k]   the push ALONG it, positive toward the toe -- what
                      its pommel-side support must supply (negative: its
                      toe-side support does)"""

    def __init__(self, x_hands, xs, name=Rules.RATED_SWING, dt=2e-4):
        sw = next(s for s in study.swings() if s.name == name)
        t = np.arange(0.0, sw.t_end, dt)
        R, H = sw.pose(t)
        b = R[:, :, 0]
        xm = (np.asarray(xs, float) - x_hands) / 1000.0
        P = H[:, None, :] + b[:, None, :] * xm[None, :, None]
        a = (P[2:] - 2.0 * P[1:-1] + P[:-2]) / (dt * dt)
        f = a - study.GW
        bb = b[1:-1]
        self.along = np.einsum("tk,txk->tx", bb, f)
        self.f = f - self.along[..., None] * bb[:, None, :]
        self.t, self.x, self.name, self.x_hands = t[1:-1], np.asarray(xs, float), name, x_hands
        self.omega_max_dps = float(np.degrees(sw.om_max))
        self.k_contact = int(np.argmin(np.abs(self.t - sw.t_c)))

    def _w(self, x):
        k = int(np.clip(np.searchsorted(self.x, x) - 1, 0, len(self.x) - 2))
        return k, (x - self.x[k]) / (self.x[k + 1] - self.x[k])

    def at(self, x):
        """f at one station, interpolated, (n_t, 3)."""
        k, w = self._w(x)
        return (1.0 - w) * self.f[:, k] + w * self.f[:, k + 1]

    def along_at(self, x):
        """along at one station, interpolated, (n_t,)."""
        k, w = self._w(x)
        return (1.0 - w) * self.along[:, k] + w * self.along[:, k + 1]

    def lever_terms(self, x_root, x_tip, x_points):
        """(G_u, [G_p...]): the moment about the root per gram, as vectors in
        time (mm m/s^2), of a uniform gram spread over [x_root, x_tip] and
        of a gram at each point."""
        sel = (self.x >= x_root) & (self.x <= x_tip)
        lever = self.x[sel] - x_root
        Gu = np.einsum("k,tkd->td", lever, self.f[:, sel]) / sel.sum()
        Gp = [(xp - x_root) * self.at(xp) for xp in x_points]
        return Gu, Gp


def peak_moment_of(Gu, Gp):
    """The peak bending moment at the root, N mm, as a function of the
    uniform mass and one point mass: max over the swing of |m_u Gu + m_p Gp|,
    whose square expands into three dot products taken once -- which is
    what lets the frame search price two thousand candidates."""
    aa, ab, bb = (np.einsum("td,td->t", u, v) for u, v in ((Gu, Gu), (Gu, Gp), (Gp, Gp)))
    return lambda m_u, m_p: 1e-3 * sqrt(float((m_u * m_u * aa + 2.0 * m_u * m_p * ab
                                               + m_p * m_p * bb).max()))


def cantilever_deflection(load, x_root, x_tip, x_obs, m_u, points, EI, K):
    """Peak deflection at x_obs of a cantilever from x_root under the swing,
    mm: the uniform mass m_u over [x_root, x_tip] and point masses.  Euler-
    Bernoulli influence plus the web's shear (structure's own model)."""
    xi = x_obs - x_root
    sel = (load.x >= x_root) & (load.x <= x_tip)
    a = load.x[sel] - x_root
    infl = np.where(a >= xi, xi * xi * (3.0 * a - xi), a * a * (3.0 * xi - a)) / (6.0 * EI) \
        + np.minimum(a, xi) / K
    d = (m_u * 1e-3 / sel.sum()) * np.einsum("k,tkd->td", infl, load.f[:, sel])
    for xp, mp in points:
        ap = xp - x_root
        ip = (xi * xi * (3.0 * ap - xi) if ap >= xi else ap * ap * (3.0 * xi - ap)) / (6.0 * EI) \
            + min(ap, xi) / K
        d = d + mp * 1e-3 * ip * load.at(xp)
    return float(np.linalg.norm(d, axis=1).max())


# ======================================================= THE HANDLE'S SECTION
@dataclass
class CoreSection:
    """Where the bay, the slot and their walls sit in the handle.

    THE BAY IS OFF THE AXIS ON PURPOSE.  Two AA cells and a board side by
    side are not round, and the handle is: the bay is pushed away from the
    slot until the circle that holds both is smallest.  Solved, not chosen,
    so a different board moves it.

    THE BOARD IS DATUMED ACROSS THE SLOT, not centred in it.  Its -y edge
    sits on the channel's -y wall and a snap beam on the +y side presses it
    there.  Centred, the channel's play reaches both edges, and each edge
    strip must hold a ledge AND clear the parts across twice that play --
    with the placeholder board's 1 mm strip that misses by 0.05 mm worst
    case.  Datumed, the play goes into the beam, and the IMU sits where the
    datum puts it, not wherever the board rattled to: which is what a
    per-unit calibration needs."""
    e: float               # mm the bay's axis sits below the bat's (toward the face)
    r_bay: float
    w_between: float       # the wall between bay and slot, with the strips' grooves in it
    z_floor: float         # slot floor (the tabs lie on it)
    z_pcb: float           # the PCB's underside at the nominal finger height
    z_ledge: float         # the ledges: the PCB's top face rests against them (datum 2)
    z_ceiling: float       # the component pocket's roof
    y_datum: float         # the channel's -y wall: the board's -y edge sits on it (datum 3)
    y_board: float         # the nominal board's +y edge
    pocket: tuple          # (lo, hi) across the bat: the component pocket over the ledges
    y_relief: float        # the far wall of the space the beam opens into
    finger_h: float        # a finger's compressed height, nominal
    finger_margin: float   # mm each way inside the finger's window, with every scatter
    ledge_overlap: float   # mm the worse ledge holds its edge strip by, every scatter taken
    hook: float            # mm the snap's hook reaches over the board's edge strip
    beam: dict             # the snap beam: length, preload, deflections, forces
    r_need: float          # the smallest circle round all of it

    def holes(self, slot=True, relief=False):
        out = [circle(0.0, -self.e, self.r_bay)]
        if slot:
            out += [rect(self.y_datum, self.y_board, self.z_floor, self.z_ledge),
                    rect(self.pocket[0], self.pocket[1], self.z_ledge, self.z_ceiling)]
        if relief:
            out += [rect(self.y_board + Print.WALL, self.y_relief, self.z_pcb, self.z_ledge)]
        return tuple(out)


def solve_snap_beam(f_board):
    """The snap beam on the slot's +y side: a cantilever along x, rooted
    toward the toe, its hook at the mouth.  It is printed standing INTO the
    channel by a preload, so every board -- narrowest, with every print
    scatter against it -- is still pressed onto the datum hard enough that
    the swing (f_board, m/s^2 across the bat at the board) cannot lift it
    off.  Its length is Bayer's, y = 0.67 eps l^2 / h, for the deflection
    it takes as the WIDEST board rides past on the hook; its stiffness is a
    cantilever's, 3 E I / l^3.  The two depend on each other, so iterate.

    The beam is as deep as the PCB it presses (the edge it bears on) and
    as thin as the process prints."""
    h, b = Print.WALL, Board.PCB_T
    F_need = Board.MASS * f_board / 1000.0
    hook = Board.EDGE_KEEP - Board.L_TOL - Print.TOL
    scatter = Board.L_TOL + Print.TOL
    l = sqrt(h * (2.0 * scatter + hook) / (0.67 * Print.SNAP_STRAIN))
    for _ in range(200):
        k = Print.E * b * h ** 3 / (4.0 * l ** 3)
        d_min = F_need / k
        pre = d_min + scatter
        d_max = pre + scatter
        d_pass = d_max + hook
        l_new = sqrt(h * d_pass / (0.67 * Print.SNAP_STRAIN))
        if abs(l_new - l) < 1e-9:
            break
        l = l_new
    return {"l": l, "h": h, "b": b, "k": k, "preload": pre, "d_min": d_min, "d_max": d_max,
            "d_pass": d_pass, "F_need": F_need, "F_min": k * d_min, "F_max": k * d_max,
            "hook": hook}


def solve_core_section(f_board):
    """The bay and the slot, packed into the smallest circle."""
    r_bay = Cell.D_MAX / 2.0 + Print.CLEAR
    # the strips run in grooves in the wall's bay face, so the wall is a
    # wall plus a groove
    w_between = Print.WALL + ContactSet.T + Print.CLEAR
    # A FINGER'S HEIGHT IS A WINDOW, and the window has to swallow every
    # scatter between the tab and the ledge: the board's thickness and
    # the printed slot's.  Centred, the margin is what is left each way.
    h_lo = Board.FINGER_FREE - Board.FINGER_MAX
    h_hi = Board.FINGER_FREE - Board.FINGER_MIN
    h = (h_lo + h_hi) / 2.0
    scatter = Board.PCB_TOL + Print.TOL
    finger_margin = (h_hi - h_lo) / 2.0 - scatter
    # ACROSS: the datum wall, the ledges and the pocket.  Each ledge's inner
    # edge is one printed dimension from the datum (+-TOL): far enough in to
    # hold its edge strip, short enough to clear the parts.
    y_datum = -Board.W / 2.0
    y_board = y_datum + Board.W
    pocket = (y_datum + Board.EDGE_KEEP - Print.TOL,
              y_datum + Board.W - Board.EDGE_KEEP + Print.TOL)
    ledge_overlap = min(pocket[0] - y_datum - Print.TOL,
                        (y_board - Board.L_TOL) - pocket[1] - Print.TOL)
    beam = solve_snap_beam(f_board)
    # the beam's back, as the widest board rides past on the hook, and a
    # clearance behind it
    y_relief = y_board + Board.L_TOL + beam["hook"] + Print.WALL + Print.CLEAR

    def layout(e):
        z_floor = -e + r_bay + w_between
        z_pcb = z_floor + ContactSet.T + h
        z_ledge = z_pcb + Board.PCB_T
        z_ceiling = z_ledge + Board.TOP_H + Print.CLEAR
        corners = [(y_relief, z_pcb), (y_relief, z_ledge), (y_datum, z_floor), (y_board, z_floor),
                   (y_datum, z_ledge), (pocket[0], z_ceiling), (pocket[1], z_ceiling)]
        r = max([e + r_bay] + [sqrt(a * a + b * b) for a, b in corners])
        return r, z_floor, z_pcb, z_ledge, z_ceiling

    es = np.arange(0.0, r_bay, 0.005)
    rs = [layout(e)[0] for e in es]
    e = float(es[int(np.argmin(rs))])
    r, z_floor, z_pcb, z_ledge, z_ceiling = layout(e)
    return CoreSection(e, r_bay, w_between, z_floor, z_pcb, z_ledge, z_ceiling, y_datum, y_board,
                       pocket, y_relief, h, finger_margin, ledge_overlap, beam["hook"], beam, r)


# ================================================================ LAYOUT
@dataclass
class Layout:
    """Every station along the bat, mm from the pommel end.  Filled by
    design(); read by mjcf, line and the checks."""
    t_end: float = 0.0       # the pommel cap's end wall
    x_face: float = 0.0      # the stub's end face: bay and slot mouths, the ring on it
    x_hook: float = 0.0      # the snap's catch face, one wall in from the mouth
    x_b0: float = 0.0        # the board's pommel edge, the board on its stop
    x_stop: float = 0.0      # the slot's stop: the board's toe edge (datum 1)
    board_play: float = 0.0  # how far the pommel edge can sit off the hook, every board
    x_imu: float = 0.0
    window: tuple = (0.0, 0.0)
    window_y: tuple = (0.0, 0.0)   # the window across the bat
    x_grip0: float = 0.0     # the stub ends, the grip begins
    x_neg: float = 0.0       # the first cell's negative end, longest stack
    x_plus: float = 0.0      # the + plate's contact face
    x_floor: float = 0.0     # the bay's floor
    x_floor_end: float = 0.0
    x_shoulder: float = 0.0  # the shoulder's handle face = Bat.handle
    x_blade0: float = 0.0    # the shoulder's toe face: the frame's root
    x_c0: float = 0.0        # chord ends, in the shoulder's sockets
    x_c1: float = 0.0        # chord ends, in the tip cap's sockets
    x_tip0: float = 0.0      # the tip cap's handle face
    x_tip1: float = 0.0      # the tip cap's end face
    x_hands: float = 0.0     # the middle of the grip: the swing study's origin
    x_sweet: float = 0.0     # the study's sweet spot on this bat
    r_stub: float = 0.0      # the thread's major radius
    r_grip: float = 0.0
    stub_l: float = 0.0
    thread_l: float = 0.0
    thread_frac: float = 0.0  # of the thread's circumference the window cuts away
    socket_l: float = 0.0
    spring: tuple = (0.0, 0.0)   # spring length with the longest / shortest stack


# =============================================================== THE FRAME
@dataclass
class FrameChoice:
    truss: Truss
    M_rated: float          # N mm at the shoulder, rated swing, peak
    F_chord: float          # N in the most loaded chord
    buckling_sf: float
    blade_mass: float       # g, what the search minimised
    reach: tuple
    candidates: int
    feasible: int
    binding: dict           # rule -> how many candidates it rejected


def _section_offsets(d_chord):
    """Corner radii of everything that wraps the frame, outward: the chord
    hull, the shoulder and tip cap (a socket wall round each chord), the
    collar (a squeezed knit, then a load-carrying wall)."""
    hull = d_chord / 2.0
    shoulder = hull + Print.CLEAR + Print.WALL
    collar_in = shoulder + Knit.T * (1.0 - Knit.SQUEEZE)
    collar_out = collar_in + Rules.LOAD_WALLS * Print.WALL
    return hull, shoulder, collar_in, collar_out


def side_max(bat, d_chord):
    """The largest chord-to-chord side whose widest section -- the collar
    -- fits the bat's envelope.  A triangle of side s spans s + 2 rc across
    and (sqrt 3 / 2) s + 2 rc deep."""
    rc = _section_offsets(d_chord)[3]
    return min(bat.width - 2.0 * rc, (bat.depth - 2.0 * rc) * 2.0 / SQ3)


def frame_sides(bat, d_chord):
    """The sides the frame search tries for one chord size: structure.design's
    own grid, from 24 mm in steps of 2, cut where the envelope ends."""
    return np.arange(24.0, side_max(bat, d_chord) + 1e-9, 2.0)


def _frame_rules(t):
    """The truss cell's own rules for a truss it can build -- the subset of
    structure.violations that is about the machine and not the camera."""
    bad = []
    rim, others, spine = structure.ring_fit(t)
    if rim < 0.0:
        bad.append("ring rim")
    if others < 0.0:
        bad.append("ring vs other chords")
    if spine < 0.0:
        bad.append("ring vs spine")
    if structure.gap_fits(t) < 0.0:
        bad.append("gap")
    if not (Rules.ALPHA_RANGE[0] <= t.alpha <= Rules.ALPHA_RANGE[1]):
        bad.append("alpha outside the qualified joint range")
    if t.band < t.mitre_face + 2.0:
        bad.append("band shorter than the mitre")
    if t.n_nodes < 3:
        bad.append("too few joints to be a truss")
    return bad


def chord_force(t, M):
    """N in the most loaded chord under a root moment M (N mm): the
    section's own I, as structure.chord_force computes it."""
    return M * t.R / structure.section_I(t) * Stock.area(t.d_chord)


def chord_buckling_sf(t, M):
    P_cr = pi ** 2 * Stock.E * Stock.I_rod(t.d_chord) / t.run ** 2
    return P_cr / max(chord_force(t, M), 1e-9)


def _blade_masses(bat, lay, s, d_chord):
    """(m_uniform, m_tip, m_rest) for a candidate section: the sleeve over
    the blade spread with the frame, the tip cap at the tip, and the
    shoulder and collar, which load nothing but weigh something.  The sleeve
    at its STRETCHED area: an upper bound, which is the safe side."""
    R = s / SQ3
    hull, sh, c_in, c_out = _section_offsets(d_chord)
    socks = tuple(circle(y, z, d_chord / 2.0 + Print.CLEAR) for y, z in _tri_vertices(R))
    L_blade = lay.x_tip0 - lay.x_blade0
    sleeve = Knit.GSM * 1e-6 * rtri(R, hull).perimeter * L_blade
    tip = printed_mass_exact([(lay.x_tip0, lay.x_c1, Section(rtri(R, sh), socks)),
                              (lay.x_c1, lay.x_tip1, Section(rtri(R, sh)))])
    shoulder = printed_mass_exact([(lay.x_shoulder, lay.x_blade0, Section(rtri(R, sh), socks))])
    collar = printed_mass_exact([(lay.x_shoulder, lay.x_blade0,
                                  Section(rtri(R, c_out), (rtri(R, c_in),)))])
    return sleeve, tip, shoulder + collar


class NoFrame(ValueError):
    """No truss fits the bat at all; `binding` says which rules closed it."""

    def __init__(self, bat, binding, candidates):
        super().__init__("no frame fits %s (%d candidates): %r" % (bat.name, candidates, binding))
        self.binding, self.candidates = binding, candidates


def design_frame(bat, lay, load):
    """The lightest frame that the cell can wind and the rated swing cannot
    buckle, inside the envelope -- structure.design's search, on the bat's
    rules.  The candidates are structure.design's own grid."""
    L = lay.x_c1 - lay.x_c0
    em = lay.socket_l + Head.ring_axial_half() + TrussProcess.SEAT_CLEAR
    alphas = [a for a in np.arange(25.0, 66.0, 1.0)
              if Rules.ALPHA_RANGE[0] <= a <= Rules.ALPHA_RANGE[1]]
    x_tipc = (lay.x_tip0 + lay.x_tip1) / 2.0
    Gu, (Gp,) = load.lever_terms(lay.x_blade0, lay.x_c1, [x_tipc])
    peak = peak_moment_of(Gu, Gp)
    blade = {}                      # (side, chord) -> _blade_masses: the same for every alpha
    rows, n = [], 0
    binding = {}
    for dc in Rules.D_CHORDS:
        c_in = _section_offsets(dc)[2]
        sides = frame_sides(bat, dc)
        if not len(sides):
            why = "the envelope is narrower than the smallest frame the grid tries"
            binding[why] = binding.get(why, 0) + 1
        for dd in Rules.D_DIAGS:
            if dd > dc:
                continue
            for a in alphas:
                for s in sides:
                    t = Truss(length=float(L), side=float(s), alpha=float(a),
                              d_chord=float(dc), d_diag=float(dd), end_margin=float(em),
                              name=bat.name + "_frame")
                    if t.length - 2.0 * t.end_margin < 2.0 * t.run:
                        why = "the blade is shorter than two bays"
                        binding[why] = binding.get(why, 0) + 1
                        continue
                    n += 1
                    bad = _frame_rules(t)
                    # the collar goes on over the handle (the sleeve covers
                    # the toe), so the section has a floor as well as a roof
                    if rtri_inradius(t.R, c_in) < bat.grip_d / 2.0 + Print.CLEAR:
                        bad.append("the collar cannot pass the grip")
                    if (s, dc) not in blade:
                        blade[s, dc] = _blade_masses(bat, lay, s, dc)
                    sleeve, tip, rest = blade[s, dc]
                    M = peak(t.mass + sleeve, tip)
                    sf = chord_buckling_sf(t, M)
                    if sf < Rules.BUCKLE_SF:
                        bad.append("chord buckling under the rated swing")
                    for b in bad:
                        binding[b] = binding.get(b, 0) + 1
                    if not bad:
                        rows.append((t.mass + sleeve + tip + rest, -sf, t, M, sf))
    rows.sort(key=lambda r: (r[0], r[1]))
    for mass, _, t, M, sf in rows:
        xr, yr = structure.reach(t)
        if xr <= Gantry.X_TRAVEL and yr <= Gantry.Y_TRAVEL / 2.0:
            return FrameChoice(t, M, chord_force(t, M), sf, mass, (xr, yr), n, len(rows),
                               dict(sorted(binding.items(), key=lambda kv: -kv[1])))
        binding["outside the gantry's reach"] = binding.get("outside the gantry's reach", 0) + 1
    raise NoFrame(bat, dict(sorted(binding.items(), key=lambda kv: -kv[1])), n)


# ================================================================ RESULTS
@dataclass
class Fit:
    name: str
    margin: float           # mm (or the unit named), positive is room
    detail: str = ""

    @property
    def ok(self):
        return self.margin >= -1e-9


@dataclass
class BatDesign:
    bat: Bat
    core: CoreSection
    lay: Layout
    frame: FrameChoice
    geom: TrussGeometry
    parts: dict
    fits: list
    sleeve: dict
    bands: list             # (centre, (lo, hi)) for each band, x in the bat frame
    band_print: list        # distance from the blank's closed tip to each band's centre
    tube: dict
    tray: dict
    printing: dict
    swing: dict
    offsets: tuple          # corner radii: hull, shoulder, collar in, collar out

    @property
    def R(self):
        return self.frame.truss.R

    def props(self, cells=True):
        out = MassProps()
        for k, p in self.parts.items():
            if k.startswith("cell") and not cells:
                continue
            out = out + p.props
        return out

    @property
    def balance(self):
        """Balance point, as a fraction of the length from the pommel."""
        return self.props(cells=True).c[0] / self.bat.length

    def envelope(self):
        """(width, depth) of the widest section, sleeve and collar included."""
        R = self.R
        rc = self.offsets[3]
        rk = self.offsets[1] + Knit.T
        rc = max(rc, rk)
        w = max(SQ3 * R + 2.0 * rc, 2.0 * self.lay.r_grip)
        dpt = max(1.5 * R + 2.0 * rc, 2.0 * self.lay.r_grip)
        return w, dpt


def _rotz(phi):
    """Unit vector at azimuth phi in the y-z plane."""
    return np.array([0.0, cos(radians(phi)), sin(radians(phi))])


def _pack_on_bed(fp, bed=None, gap=None):
    """How many of a footprint (a, b) fit a bed: the better of the two grid
    orientations, or one laid on the diagonal when neither grid takes it."""
    bed = Printer.BUILD if bed is None else bed
    gap = Printer.PART_GAP if gap is None else gap
    A, B = bed[0], bed[1]
    a, b = fp
    best = 0
    for p, q in ((a, b), (b, a)):
        if p <= A and q <= B:
            best = max(best, int(floor((A + gap) / (p + gap))) * int(floor((B + gap) / (q + gap))))
    if best == 0:
        for th in np.radians(np.arange(0.0, 90.01, 0.5)):
            if a * cos(th) + b * sin(th) <= A and a * sin(th) + b * cos(th) <= B:
                return 1
    return best


def design(bat, line=None, iterations=12):
    """Derive the bat.  Returns a BatDesign, whose `fits` say where it does
    not fit -- nothing here raises on a bad bat, because a check that
    cannot see a bad bat cannot fail.  The one exception is a bat no frame
    fits at all (NoFrame): then there is no blade to put the rest on."""
    line = Line() if line is None else line
    lay = Layout()
    W = Print.WALL
    lay.t_end = Rules.LOAD_WALLS * W
    lay.x_face = lay.t_end + ContactSet.T
    lay.r_grip = bat.grip_d / 2.0
    # the stub: the thread's major radius is what the cap's wall leaves
    lay.r_stub = lay.r_grip - Print.CLEAR - W
    # THE BOARD: its toe edge on the stop (datum 1), which is where the
    # swing presses it at contact (a fit below says so); the slot as long
    # as the longest board and a sliding clearance, the hook one wall in
    # from the mouth.  Off its stop, the board can sit anywhere back to the
    # hook: board_play.
    lay.x_hook = lay.x_face + W
    lay.x_stop = lay.x_hook + Board.L + Board.L_TOL + Print.CLEAR
    lay.x_b0 = lay.x_stop - Board.L
    lay.board_play = 2.0 * Board.L_TOL + Print.CLEAR
    lay.x_imu = lay.x_b0 + Board.IMU_AT[0]
    # THE WINDOW exposes every pad wherever the board sits -- from its hook
    # to its stop along the bat, on its datum across it -- with the slot's
    # print scatter and the gantry's repeatability on top: the pogo block
    # is placed by the gantry, not by the board.
    pa = [a_ for a_, _ in Board.PADS_AT]
    pb = [b_ for _, b_ in Board.PADS_AT]
    reach = Board.PAD_D / 2.0 + Print.TOL + Gantry.REPEAT
    lay.window = (lay.x_hook + min(pa) - reach, lay.x_hook + lay.board_play + max(pa) + reach)
    lay.window_y = (min(pb) - reach, max(pb) + reach)     # the datum puts the board's centreline on the bat's
    half_w = max(abs(lay.window_y[0]), abs(lay.window_y[1]))
    lay.thread_frac = 2.0 * asin(min(1.0, half_w / lay.r_stub)) / (2.0 * pi)
    lay.thread_l = Print.THREAD_TURNS * Print.THREAD_PITCH / (1.0 - lay.thread_frac)
    lay.stub_l = max(lay.thread_l, lay.window[1] + W - lay.x_face)
    lay.x_grip0 = lay.x_face + lay.stub_l
    # THE STACK.  The cap's spring sits on the cap's inner face; the cells
    # run from its tip to the + plate.  The plate goes where the spring's
    # working range holds BOTH the longest and the shortest pair, centred.
    n = Rules.N_CELLS
    lo = lay.t_end + n * Cell.L_MAX + CapSpring.L_SOLID
    hi = lay.t_end + n * Cell.L_MIN + CapSpring.L_FREE - CapSpring.F_MIN / CapSpring.RATE
    lay.x_plus = (lo + hi) / 2.0
    lay.x_neg = lay.x_plus - n * Cell.L_MAX
    lay.spring = (lay.x_plus - n * Cell.L_MAX - lay.t_end, lay.x_plus - n * Cell.L_MIN - lay.t_end)
    lay.x_floor = lay.x_plus + ContactSet.T
    lay.x_floor_end = lay.x_floor + Rules.LOAD_WALLS * W
    # the far end
    lay.x_shoulder = bat.handle
    lay.x_tip1 = bat.length - Knit.T
    lay.x_hands = (lay.x_grip0 + lay.x_shoulder) / 2.0
    lay.x_sweet = lay.x_hands + 1000.0 * float(study.L_SS[0])
    load = SwingLoad(lay.x_hands, np.arange(0.0, bat.length + 5.0, 5.0))
    # the handle's section waits for the swing: the snap beam is sized to
    # hold the board on its datum against the hardest push across the bat
    # anywhere along the board
    f_board = max(float(np.linalg.norm(load.at(x), axis=1).max()) for x in (lay.x_b0, lay.x_stop))
    core = solve_core_section(f_board)

    # THE SOCKETS AND THE FRAME DEPEND ON EACH OTHER: the bond's length is
    # set by the chord force, which is set by the frame, which is chosen on
    # a length the sockets take out of the blade.  Iterate to the fixed point.
    # The socket only ever deepens, so the loop settles on a socket that
    # holds the frame it was chosen with, not one a hair short of it; a
    # micron is the most it is ever deeper than it needs to be.
    lay.socket_l, frame = 0.0, None
    for _ in range(iterations):
        lay.x_blade0 = lay.x_shoulder + lay.socket_l + Rules.LOAD_WALLS * W
        lay.x_c0 = lay.x_shoulder + Rules.LOAD_WALLS * W
        lay.x_c1 = lay.x_tip1 - Rules.LOAD_WALLS * W
        lay.x_tip0 = lay.x_c1 - lay.socket_l
        frame = design_frame(bat, lay, load)
        t = frame.truss
        need = frame.F_chord / (pi * t.d_chord * Print.BOND_TAU)
        if need <= lay.socket_l:
            break
        lay.socket_l = need + 1e-3

    t = frame.truss
    geom = TrussGeometry(t)
    R = t.R
    off = _section_offsets(t.d_chord)
    hull, sh, c_in, c_out = off

    # ------------------------------------------------------------ parts
    parts = {}
    sock = [circle(*(R * radial(chord_phi(k))[1:]), t.d_chord / 2.0 + Print.CLEAR)
            for k in range(3)]
    press = [circle(*(R * radial(chord_phi(k))[1:]), t.d_chord / 2.0 - Print.INTERFERENCE)
             for k in range(3)]
    r_minor = lay.r_stub - Print.THREAD_DEPTH
    r_stub_mean = lay.r_stub - Print.THREAD_DEPTH / 2.0
    win = rect(lay.window_y[0], lay.window_y[1], core.z_ceiling, lay.r_grip + 1.0)
    x_relief = lay.x_hook + core.beam["l"]           # the snap beam's root
    # THE STRIPS' GROOVES: each strip lies in a groove in the bay's roof, a
    # strip's thickness and a clearance deep -- which is what w_between
    # left room for -- from where it starts to where it ends.  The + strip
    # runs from its tab to the + plate, the return strip from the ring on
    # the end face to its tab.
    (fx0, fy0), (fx1, fy1) = Board.FINGERS_AT
    x_tab = [lay.x_hook + lay.board_play / 2.0 + fx for fx in (fx0, fx1)]
    z_groove = -core.e + core.r_bay + ContactSet.T + Print.CLEAR
    grooves = [(x_tab[0] - ContactSet.W / 2.0, lay.x_plus, fy0),
               (lay.x_face, x_tab[1] + ContactSet.W / 2.0, fy1)]

    def groove_holes(x):
        return tuple(rect(fy - ContactSet.W / 2.0 - Print.CLEAR, fy + ContactSet.W / 2.0 + Print.CLEAR,
                          -core.e, z_groove)
                     for x0_, x1_, fy in grooves if x0_ <= x < x1_)

    def core_section(x):
        """The core's section at x: the stub, the grip or the shoulder, less
        whatever runs through it there."""
        if x >= lay.x_shoulder:
            return Section(rtri(R, sh), tuple(sock) if x >= lay.x_c0 else ())
        solid = circle(0, 0, r_stub_mean if x < lay.x_grip0 else lay.r_grip)
        if x < lay.x_stop:
            holes = core.holes(slot=True, relief=x < x_relief)
        elif x < lay.x_floor:
            holes = core.holes(slot=False)
        else:
            holes = ()
        if lay.window[0] <= x < lay.window[1]:
            holes = holes + (win,)
        return Section(solid, holes + groove_holes(x))
    cuts = sorted({lay.x_face, lay.window[0], lay.window[1], lay.x_grip0, x_relief, lay.x_stop,
                   lay.x_floor, lay.x_shoulder, lay.x_c0, lay.x_blade0}
                  | {g_[0] for g_ in grooves} | {g_[1] for g_ in grooves})
    core_prisms = [(x0, x1, core_section((x0 + x1) / 2.0))
                   for x0, x1 in zip(cuts[:-1], cuts[1:]) if x1 - x0 > 1e-6]
    parts["core"] = printed("core", (0.93, 0.93, 0.90, 1.0), core_prisms)
    parts["tip_cap"] = printed("tip_cap", (0.85, 0.85, 0.82, 1.0), [
        (lay.x_tip0, lay.x_c1, Section(rtri(R, sh), tuple(press))),
        (lay.x_c1, lay.x_tip1, Section(rtri(R, sh)))])
    parts["collar"] = printed("collar", (0.20, 0.22, 0.26, 1.0), [
        (lay.x_shoulder, lay.x_blade0, Section(rtri(R, c_out), (rtri(R, c_in),)))])
    cap_in = lay.r_stub - Print.THREAD_DEPTH / 2.0 + Print.CLEAR
    parts["pommel_cap"] = printed("pommel_cap", (0.20, 0.22, 0.26, 1.0), [
        (0.0, lay.t_end, Section(circle(0, 0, lay.r_grip))),
        (lay.t_end, lay.x_grip0 - Print.CLEAR, Section(circle(0, 0, lay.r_grip),
                                                        (circle(0, 0, cap_in),)))])
    # the frame: the truss cell's rods, moved along the bat, each its own mass
    rods = []
    shift = np.array([lay.x_c0, 0.0, 0.0])
    for r in geom.rods:
        if r.kind == "chord":
            p0, p1 = r.p0 + shift, r.p1 + shift
        else:
            p0, p1 = (q + shift for q in geom.diag_body_ends(r))
        m = Stock.rho_lin(2.0 * r.r) * (t.length if r.kind == "chord" else t.L_cut) / 1000.0
        rods.append((p0, p1, r.r, m))
    joints = [(geom.chord_point(j.chord, j.x) + shift, t.joint_mass) for j in geom.joints]
    parts["frame"] = Part("frame", "built", (0.10, 0.10, 0.11, 1.0), rods=rods, points=joints)
    # the sleeve: a knit skin over shoulder, blade and tip cap; mass from the blank
    shell = [(lay.x_shoulder, lay.x_blade0, Section(rtri(R, sh + Knit.T), (rtri(R, sh),))),
             (lay.x_blade0, lay.x_tip0, Section(rtri(R, hull + Knit.T), (rtri(R, hull),))),
             (lay.x_tip0, lay.x_tip1, Section(rtri(R, sh + Knit.T), (rtri(R, sh),))),
             (lay.x_tip1, bat.length, Section(rtri(R, sh + Knit.T)))]

    # ------------------------------------------------- the sleeve blank
    P_blade = rtri(R, hull).perimeter
    P_cap = rtri(R, sh).perimeter
    rc_str_in = sh + Knit.T + Print.CLEAR + Gantry.REPEAT
    rc_str_out = rc_str_in + W
    P_str = rtri(R, rc_str_out).perimeter
    c_lo = P_str / (1.0 + Knit.STRETCH_MAX)
    c_hi = P_blade / (1.0 + Knit.STRETCH_MIN)
    C0 = sqrt(c_lo * c_hi) if c_hi > c_lo else (c_lo + c_hi) / 2.0
    e_blade, e_cap, e_str = P_blade / C0 - 1.0, P_cap / C0 - 1.0, P_str / C0 - 1.0
    lam = {k: 1.0 - Knit.AXIAL_PER_HOOP * e for k, e in
           (("blade", e_blade), ("cap", e_cap))}
    end_r = sqrt(rtri(R, sh).area / pi)          # the closed tip, as a disc
    segs = [("cap", lay.x_shoulder, lay.x_blade0), ("blade", lay.x_blade0, lay.x_tip0),
            ("cap", lay.x_tip0, lay.x_tip1)]
    relaxed_l = sum((b - a) / lam[k] for k, a, b in segs) + end_r
    m_sleeve = Knit.GSM * 1e-6 * (C0 * relaxed_l)
    V_shell = sum(moments(s)[0] * (x1 - x0) for x0, x1, s in shell)
    parts["sleeve"] = Part("sleeve", "bought, printed by hand", (0.97, 0.97, 0.97, 1.0),
                           prisms=shell, rho=m_sleeve / V_shell)
    sleeve = {"C0": C0, "window": (c_lo, c_hi), "stretch_blade": e_blade,
              "stretch_cap": e_cap, "stretch_stretcher": e_str, "relaxed_l": relaxed_l,
              "mass": m_sleeve, "P_blade": P_blade, "P_stretcher": P_str,
              "stretcher_rc": (rc_str_in, rc_str_out),
              "segs": segs, "end_r": end_r}

    # --------------------------------------------------------- the bands
    tol = Knit.PRINT_TOL + Knit.PLACE_TOL
    hb = Rules.BAND_W / 2.0
    xa = lay.x_blade0 + hb + tol
    xb = lay.x_tip1 - hb - tol
    bands = [(xa, (xa - tol, xa + tol)), (xb, (xb - tol, xb + tol))]

    def from_tip(x):
        """Distance along the RELAXED blank from its closed tip to x."""
        u = end_r
        for k, a, b in reversed(segs):
            lo_, hi_ = max(a, x), b
            if hi_ > lo_:
                u += (hi_ - lo_) / lam[k]
        return u
    band_print = [from_tip(c) for c, _ in bands]

    # ------------------------------------------------- bought and customer
    # board: the PCB, the module on top, the IMU; positions in the slot
    y_b = core.y_datum + Board.W / 2.0
    pcb_c = np.array([lay.x_b0 + Board.L / 2.0, y_b, core.z_pcb + Board.PCB_T / 2.0])
    pcb_h = (Board.L / 2.0, Board.W / 2.0, Board.PCB_T / 2.0)
    mod_h = (Board.L * 0.35 / 2.0, (Board.W - 2.0 * Board.EDGE_KEEP) / 2.0 * 0.9, Board.TOP_H / 2.0)
    mod_c = np.array([lay.x_b0 + Board.L * 0.70, 0.0, core.z_ledge + Board.TOP_H / 2.0])
    m_pcb = Board.MASS * 0.5
    parts["board"] = Part("board", "bought", (0.05, 0.40, 0.15, 1.0),
                          boxes=[(pcb_c, pcb_h, m_pcb), (mod_c, mod_h, Board.MASS - m_pcb)])
    # cells: the longest pair, pip toward the + plate
    for i in range(n):
        x1 = lay.x_plus - i * Cell.L_MAX
        x0 = x1 - Cell.L_MAX
        body = (np.array([x0, 0.0, -core.e]), np.array([x1 - Cell.PIP_H, 0.0, -core.e]))
        pip = (np.array([x1 - Cell.PIP_H, 0.0, -core.e]), np.array([x1, 0.0, -core.e]))
        m_pip = Cell.MASS * (Cell.PIP_D / Cell.D_MAX) ** 2 * Cell.PIP_H / Cell.L_MAX
        parts["cell%d" % (i + 1)] = Part("cell%d" % (i + 1), "customer", (0.75, 0.62, 0.15, 1.0),
                                         rods=[(*body, Cell.D_MAX / 2.0, Cell.MASS - m_pip),
                                               (*pip, Cell.PIP_D / 2.0, m_pip)])
    # contact set: the plate, the strips along the bay's roof, their tabs,
    # and the ring on the end face; steel, by volume
    z_strip = -core.e + core.r_bay + ContactSet.T / 2.0
    z_tab = core.z_floor + ContactSet.T / 2.0
    T, Ws = ContactSet.T, ContactSet.W
    boxes = []
    # each tab under the middle of where its finger can be (x_tab, above):
    # the board's pommel edge ranges over board_play from the hook
    for (x_a, x_b, y) in ((x_tab[0], lay.x_plus, fy0), (lay.x_face, x_tab[1], fy1)):
        boxes.append((np.array([(x_a + x_b) / 2.0, y, z_strip]), ((x_b - x_a) / 2.0, Ws / 2.0, T / 2.0)))
    for (x, y) in ((x_tab[0], fy0), (x_tab[1], fy1)):
        boxes.append((np.array([x, y, z_tab]), (Ws / 2.0, Ws / 2.0, T / 2.0)))
    plate_r = core.r_bay - Print.CLEAR
    ring_in, ring_out = core.r_bay, core.r_bay + core.w_between - Print.CLEAR
    rods = [(np.array([lay.x_plus, 0.0, -core.e]), np.array([lay.x_plus + T, 0.0, -core.e]), plate_r,
             ContactSet.RHO * pi * plate_r ** 2 * T)]
    ring_m = ContactSet.RHO * pi * (ring_out ** 2 - ring_in ** 2) * T
    parts["contacts"] = Part("contacts", "bought", (0.78, 0.78, 0.80, 1.0),
                             boxes=[(c, h, ContactSet.RHO * 8.0 * h[0] * h[1] * h[2]) for c, h in boxes],
                             rods=rods, points=[(np.array([lay.t_end + T / 2.0, 0.0, -core.e]), ring_m)])
    parts["spring"] = Part("spring", "bought", (0.65, 0.65, 0.68, 1.0), rods=[
        (np.array([lay.t_end, 0.0, -core.e]), np.array([lay.x_neg, 0.0, -core.e]),
         (CapSpring.D_BASE + CapSpring.D_TIP) / 4.0, CapSpring.MASS)])

    # --------------------------------------------------------- the fits
    fits = []

    def fit(name, margin, detail=""):
        fits.append(Fit(name, float(margin), detail))

    fit("two IEC cells slide into the bay, fattest included",
        (core.r_bay - Cell.D_MAX / 2.0) - Print.TOL,
        "bay %.2f mm for a %.1f mm cell" % (2 * core.r_bay, Cell.D_MAX))
    fit("the spring does not bottom on the longest pair",
        lay.spring[0] - CapSpring.L_SOLID, "%.2f mm long, solid at %.1f" % (lay.spring[0], CapSpring.L_SOLID))
    fit("...and still presses the shortest pair at its rated force",
        (CapSpring.L_FREE - CapSpring.F_MIN / CapSpring.RATE) - lay.spring[1],
        "%.2f mm long, %.2f N" % (lay.spring[1], (CapSpring.L_FREE - lay.spring[1]) * CapSpring.RATE))
    fit("the stack and its floor fit inside the handle",
        lay.x_shoulder - lay.x_floor_end, "floor ends at %.1f of a %.0f handle" % (lay.x_floor_end, bat.handle))
    fit("the bay, the slot, the snap's relief and their walls fit under the thread's root",
        r_minor - (core.r_need + W), "need %.2f under a root of %.2f" % (core.r_need + W, r_minor))
    fit("the ledges hold both edge strips and clear the parts, every scatter taken",
        core.ledge_overlap, "%.2f mm of ledge on the worse edge" % core.ledge_overlap)
    fit("the snap's hook reaches over the edge strip by a printable width and clears the parts",
        core.hook - Print.NOZZLE, "a %.2f mm hook" % core.hook)
    bm = core.beam
    fit("the snap beam fits beside the board, short of the stop's wall",
        (lay.x_stop - W) - x_relief,
        "a %.1f mm beam, preloaded %.2f mm: %.2f-%.2f N on the datum for %.2f N of swing"
        % (bm["l"], bm["preload"], bm["F_min"], bm["F_max"], bm["F_need"]))
    fit("each finger presses inside its working range, across the board's and the slot's scatter",
        core.finger_margin, "compressed to %.2f mm" % core.finger_h)
    fit("the longest board fits between the hook and the stop, the slot printed short",
        (lay.x_stop - lay.x_hook) - (Board.L + Board.L_TOL) - Print.TOL)
    for i, (fx, fy) in enumerate(Board.FINGERS_AT):
        fit("finger %d lands on its tab wherever the board sits" % (i + 1),
            Ws / 2.0 - Board.FINGER_W / 2.0 - max(lay.board_play / 2.0, 0.0) - Print.TOL)
    pad_y = [y_b + b_ for _, b_ in Board.PADS_AT]
    fit("the test pads and the window over them lie between the ledges",
        min(min(pad_y) - Board.PAD_D / 2.0 - (core.pocket[0] + Print.TOL),
            (core.pocket[1] - Print.TOL) - (max(pad_y) + Board.PAD_D / 2.0),
            lay.window_y[0] - core.pocket[0], core.pocket[1] - lay.window_y[1]))
    fit("the pommel cap's thread engages its turns, less the window's arc",
        lay.stub_l * (1.0 - lay.thread_frac) - Print.THREAD_TURNS * Print.THREAD_PITCH,
        "%.1f mm of stub, %.0f%% cut by the window" % (lay.stub_l, 100 * lay.thread_frac))
    fit("the test pads are under the cap, behind a whole wall",
        (lay.x_grip0 - W) - lay.window[1])
    fit("the cap's stub ends short of the shoulder, so there is a grip",
        lay.x_shoulder - lay.x_grip0, "%.0f mm of grip" % (lay.x_shoulder - lay.x_grip0))
    # THE SWING decides which datum the board sits on and whether the cells
    # keep their contacts: what a support must supply along the bat, per
    # gram, positive toward the toe
    x_bm = lay.x_b0 + Board.L / 2.0
    f_contact = float(load.along_at(x_bm)[load.k_contact])
    fit("at contact the swing presses the board onto its stop, not its hook",
        -f_contact / 9.81, "%.1f g toward the stop" % (-f_contact / 9.81))
    m_cells = Rules.N_CELLS * Cell.MASS
    x_cm = lay.x_plus - Rules.N_CELLS * Cell.L_MAX / 2.0
    pull = max(float(load.along_at(x_cm).max()), 0.0)
    F_spring = (CapSpring.L_FREE - lay.spring[1]) * CapSpring.RATE
    fit("the spring holds the cells on the + plate however the swing pulls them back",
        F_spring - m_cells * pull / 1000.0,
        "%.2f N for %.2f N (%.1f g back)" % (F_spring, m_cells * pull / 1000.0, pull / 9.81))
    fit("the spring's base coil lands on the return ring",
        min(CapSpring.D_BASE / 2.0 - ring_in, ring_out - CapSpring.D_BASE / 2.0),
        "coil %.1f on a ring %.1f-%.1f" % (CapSpring.D_BASE, 2 * ring_in, 2 * ring_out))
    fit("the chords slide into the shoulder's sockets",
        Print.CLEAR - Print.TOL)
    fit("the sockets bond the rated swing's chord force",
        lay.socket_l - frame.F_chord / (pi * t.d_chord * Print.BOND_TAU),
        "%.1f mm for %.0f N" % (lay.socket_l, frame.F_chord))
    fit("the first joint clears the shoulder by the ring's half-width and a seat",
        (t.x0) - (lay.socket_l + Head.ring_axial_half() + TrussProcess.SEAT_CLEAR))
    fit("the collar passes over the grip and the pommel cap",
        rtri_inradius(R, c_in) - lay.r_grip - Print.CLEAR)
    fit("the sleeve blank has a stretch window: taut on the blade",
        e_blade - Knit.STRETCH_MIN, "hoop strain %.2f" % e_blade)
    fit("...and not torn on the stretcher",
        Knit.STRETCH_MAX - e_str, "hoop strain %.2f" % e_str)
    fit("the two bands' zones stay apart",
        (bands[1][1][0] - hb) - (bands[0][1][1] + hb), "spacing %.0f mm" % (xb - xa))
    fit("the bands' zones lie on the visible sleeve",
        min(bands[0][1][0] - hb - lay.x_blade0, lay.x_tip1 - (bands[1][1][1] + hb)))
    # the envelope as the PARTS make it, not as the section formula says:
    # a part that grows past the formula shows here
    ext = np.array([part_extent(p) for p in parts.values()])
    wenv = ext[:, 1].max() - ext[:, 0].min()
    denv = ext[:, 3].max() - ext[:, 2].min()
    r_parts = float(ext[:, 4].max())
    fit("the widest section fits the bat's width", bat.width - wenv, "%.1f of %.0f" % (wenv, bat.width))
    fit("...and its depth", bat.depth - denv, "%.1f of %.0f" % (denv, bat.depth))
    m_all = sum(p.mass for p in parts.values())
    if bat.mass_max is not None:
        fit("the bat, cells in, is no heavier than its spec", bat.mass_max - m_all,
            "%.0f g of %.0f" % (m_all, bat.mass_max))
    if bat.balance is not None:
        bal = sum(p.mass * p.props.c[0] for p in parts.values()) / m_all / bat.length
        fit("its balance point lies in the spec's range",
            min(bal - bat.balance[0], bat.balance[1] - bal) * bat.length,
            "%.3f of the length, spec %.2f-%.2f" % (bal, bat.balance[0], bat.balance[1]))

    # ------------------------------------------------- the tube, the tray
    r_max = max(rtri_circumradius(R, c_out), rtri_circumradius(R, sh + Knit.T), lay.r_grip)
    c_tube = Print.TOL + Gantry.REPEAT
    tube_id = 2.0 * r_max + Tube.ID_TOL + 2.0 * c_tube
    tube_l = bat.length + 2.0 * c_tube
    tube = {"id": tube_id, "od": tube_id + 2.0 * Tube.WALL, "length": tube_l,
            "r_max": r_max,
            "mass": Tube.RHO * pi * ((tube_id / 2.0 + Tube.WALL) ** 2 - (tube_id / 2.0) ** 2) * tube_l}
    fit("the bat, every part counted, slides into its tube at the tube's tightest",
        (tube_id - Tube.ID_TOL) / 2.0 - r_parts - Print.TOL,
        "%.1f mm round the widest part, a %.1f mm tube" % (2.0 * r_parts, tube_id))
    fit("...and is no longer than it", tube_l - (ext[:, 6].max() - ext[:, 5].min()))
    k = line.bats_per_tray
    cradle = tube["od"] + 2.0 * (Print.CLEAR + W)          # the widest thing a tray carries
    tray = {"length": tube_l + 2.0 * (Tube.CAP_DEPTH + Print.CLEAR + Rules.LOAD_WALLS * W),
            "width": k * cradle + 2.0 * Rules.LOAD_WALLS * W, "per_tray": k}

    # ----------------------------------------------- printing, per bat
    printing = {}
    for name in ("core", "tip_cap", "collar", "pommel_cap"):
        p = parts[name]
        xs_ = [q[0] for q in p.prisms] + [q[1] for q in p.prisms]
        span = max(xs_) - min(xs_)
        bbs = [q[2].bbox() for q in p.prisms]
        wid = max(b[1] for b in bbs) - min(b[0] for b in bbs)
        dep = max(b[3] for b in bbs) - min(b[2] for b in bbs)
        # lying along x, on its flattest side: a part is printed on its side
        footprint = (span, max(wid, dep))
        per_plate = _pack_on_bed(footprint)
        printing[name] = {"mass": p.mass, "print_s": p.mass / Print.RHO / Printer.FLOW,
                          "footprint": footprint, "height": min(wid, dep),
                          "per_plate": per_plate}
    fit("every printed part fits a printer's bed",
        min(v["per_plate"] for v in printing.values()) - 1,
        ", ".join("%s %d" % (k_, v["per_plate"]) for k_, v in printing.items()))
    fit("...and is no taller than the printer",
        Printer.BUILD[2] - max(v["height"] for v in printing.values()))

    # ------------------------------------------------------- the swing
    EI = structure.EI(t)
    K = structure.shear_stiffness(t)
    sleeve_blade = Knit.GSM * 1e-6 * P_blade * (lay.x_tip0 - lay.x_blade0)
    x_tipc = parts["tip_cap"].props.c[0]
    d_blade = cantilever_deflection(load, lay.x_blade0, lay.x_c1, lay.x_sweet,
                                    t.mass + sleeve_blade, [(x_tipc, parts["tip_cap"].mass)], EI, K)
    # the handle's share, ESTIMATED: the hands span the grip, the moment in
    # it climbs from nothing at the top hand to the shoulder's at the bottom
    # one, and the IMU sits under the top hand.  The slope it puts between
    # the IMU and the shoulder, carried out to the sweet spot.
    EI_core = Print.E * printed_I(Section(circle(0, 0, lay.r_grip), core.holes(slot=False)))
    span = lay.x_shoulder - lay.x_grip0
    a0 = max(lay.x_imu - lay.x_grip0, 0.0)
    theta = frame.M_rated / EI_core * (span * span - a0 * a0) / (2.0 * span)
    d_handle = theta * (lay.x_sweet - lay.x_shoulder)
    swing = {"name": load.name, "omega_dps": load.omega_max_dps, "M_rated": frame.M_rated,
             "F_chord": frame.F_chord, "buckling_sf": frame.buckling_sf,
             "sweet_bend_blade": d_blade, "sweet_bend_handle": d_handle,
             "EI_blade": EI, "EI_handle": EI_core,
             "lever_budget": float(study.CALIBRATED.lever_mm)}

    return BatDesign(bat, core, lay, frame, geom, parts, fits, sleeve, bands, band_print,
                     tube, tray, printing, swing, off)


def blank_to_frame(d, u, axial_per_hoop=None):
    """Where a mark printed u mm from the relaxed blank's closed tip lands
    on the frame, the knit contracting along the bat by `axial_per_hoop`
    per unit of hoop strain (Knit's own unless given) -- design()'s
    from_tip, inverted.  With Knit's number it puts a band's print position
    back on its zone's centre; with the knit's real one, wherever that knit
    puts it."""
    a = Knit.AXIAL_PER_HOOP if axial_per_hoop is None else axial_per_hoop
    sl = d.sleeve
    strain = {"blade": sl["stretch_blade"], "cap": sl["stretch_cap"]}
    left = u - sl["end_r"]
    x = d.lay.x_tip1
    for k, x0, x1 in reversed(sl["segs"]):
        lam = 1.0 - a * strain[k]
        if left <= (x1 - x0) / lam:
            return x1 - left * lam
        left -= (x1 - x0) / lam
        x = x0
    return x - left                 # past the shoulder: the hem, unstretched


# ============================================================ INTERFERENCE
def declared_overlaps(d):
    """The parts that overlap ON PURPOSE, and by how much: {pair: mm^2 of
    section}.  Everything else that overlaps is a mistake."""
    t = d.frame.truss
    hull, sh, c_in, c_out = d.offsets
    r = t.d_chord / 2.0
    return {frozenset(("collar", "sleeve")):
            rtri(d.R, sh + Knit.T).area - rtri(d.R, c_in).area,          # the squeeze on the hem
            frozenset(("tip_cap", "frame")):
            3.0 * pi * (r * r - (r - Print.INTERFERENCE) ** 2)}          # the chords' press fit


def _segment_mask(Y, Z, x, p0, p1, r):
    """Points (x, Y, Z) inside the CYLINDER of radius r from p0 to p1 --
    flat ends, as rod_props weighs it, not a capsule's round ones."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    u = p1 - p0
    L2 = float(u @ u)
    t = ((x - p0[0]) * u[0] + (Y - p0[1]) * u[1] + (Z - p0[2]) * u[2]) / L2
    dx = x - (p0[0] + t * u[0])
    dy = Y - (p0[1] + t * u[1])
    dz = Z - (p0[2] + t * u[2])
    return (t >= 0.0) & (t <= 1.0) & (dx * dx + dy * dy + dz * dz <= r * r)


def _part_mask(part, x, Y, Z):
    m = np.zeros(Y.shape, bool)
    for x0, x1, sec in part.prisms:
        if x0 <= x < x1:
            m |= sec.mask(Y, Z)
    for p0, p1, r, _ in part.rods:
        if min(p0[0], p1[0]) - r <= x <= max(p0[0], p1[0]) + r:
            m |= _segment_mask(Y, Z, x, p0, p1, r)
    for c, h, _ in part.boxes:
        if c[0] - h[0] <= x <= c[0] + h[0]:
            m |= (np.abs(Y - c[1]) <= h[1]) & (np.abs(Z - c[2]) <= h[2])
    return m


def interferences(d, res=0.1, parts=None):
    """Every place two parts occupy the same space: [(a, b, x, mm^2)].

    Sections are compared, x by x, at the middle of every interval in
    which no part changes -- between every prism's ends, every rod's and
    every box's -- each part's section rasterised at `res`.  A hole's mask
    leaves out its own boundary, so a part that only TOUCHES another (the
    board on its datums, the sleeve on the chords) shares no cell with
    it; measured, at 0.05, 0.1 and 0.2 mm the only overlaps on a correct
    bat are the declared ones, at their declared areas.  What this cannot
    see is an overlap thinner than `res`."""
    parts = d.parts if parts is None else parts
    xs = set()
    spans = {}
    for k, p in parts.items():
        lo, hi = [], []
        for x0, x1, _ in p.prisms:
            xs.update((x0, x1))
            lo.append(x0)
            hi.append(x1)
        for p0, p1, r, _ in p.rods:
            a, b = sorted((p0[0], p1[0]))
            xs.update((a, b))
            lo.append(a - r)
            hi.append(b + r)
        for c, h, _ in p.boxes:
            xs.update((c[0] - h[0], c[0] + h[0]))
            lo.append(c[0] - h[0])
            hi.append(c[0] + h[0])
        if lo:
            spans[k] = (min(lo), max(hi))
    xs = sorted(xs)
    ext = {k: part_extent(p) for k, p in parts.items() if k in spans}
    out = []
    for xa, xb in zip(xs[:-1], xs[1:]):
        if xb - xa < 1e-6:
            continue
        x = (xa + xb) / 2.0
        here = [k for k, (a, b) in spans.items() if a <= x <= b]
        if len(here) < 2:
            continue
        y0 = min(ext[k][0] for k in here)
        y1 = max(ext[k][1] for k in here)
        z0 = min(ext[k][2] for k in here)
        z1 = max(ext[k][3] for k in here)
        Y, Z = np.meshgrid(np.arange(y0, y1 + res, res), np.arange(z0, z1 + res, res), indexing="ij")
        masks = {k: _part_mask(parts[k], x, Y, Z) for k in here}
        for i, a in enumerate(here):
            for b in here[i + 1:]:
                n = int((masks[a] & masks[b]).sum())
                if n:
                    out.append((a, b, x, n * res * res))
    return out


def report(d):
    """A plain summary, for -v and the README."""
    t = d.frame.truss
    pw, pn = d.props(True), d.props(False)
    out = ["%s: %.0f mm, handle %.0f, grip %.0f, envelope %.0f x %.0f"
           % (d.bat.name, d.bat.length, d.bat.handle, d.bat.grip_d, d.bat.width, d.bat.depth),
           "  frame: side %.0f, alpha %.0f, chords %.1f, diagonals %.1f; %d joints, %.1f g; "
           "%d of %d candidates feasible" % (t.side, t.alpha, t.d_chord, t.d_diag, t.n_joints,
                                            t.mass, d.frame.feasible, d.frame.candidates),
           "  rated swing (%s, %.0f dps): %.2f N m at the shoulder, %.0f N per chord, buckling x%.1f"
           % (d.swing["name"], d.swing["omega_dps"], d.swing["M_rated"] / 1000.0,
              d.swing["F_chord"], d.swing["buckling_sf"]),
           "  sweet spot moves %.2f mm with the blade's bend, about %.1f mm with the handle's "
           "(the study assumed %.1f mm of lever-arm error)"
           % (d.swing["sweet_bend_blade"], d.swing["sweet_bend_handle"], d.swing["lever_budget"]),
           "  mass %.0f g with cells, %.0f g as shipped; balance %.2f of the length"
           % (pw.m, pn.m, d.balance),
           "  bands %s mm from the pommel, %.0f apart; print them %s mm from the blank's tip"
           % ([round(c) for c, _ in d.bands], d.bands[1][0] - d.bands[0][0],
              [round(u) for u in d.band_print]),
           "  sleeve blank: %.0f mm round, %.0f mm long relaxed; tube %.0f id x %.0f"
           % (d.sleeve["C0"], d.sleeve["relaxed_l"], d.tube["id"], d.tube["length"]),
           "  parts: " + ", ".join("%s %.1f g" % (k, p.mass) for k, p in d.parts.items())]
    bad = [f for f in d.fits if not f.ok]
    out.append("  fits: %d, %d short%s" % (len(d.fits), len(bad),
                                            "".join("\n    SHORT %s (%.2f)" % (f.name, f.margin) for f in bad)))
    return "\n".join(out)
