"""The truss as rods, joints and faces, computed from a Truss spec.

FRAME.  The truss axis is +x, running from 0 to the chord length.  The
section lies in the y-z plane and chord k sits at azimuth 90 + 120 k
degrees, so chord 0 points straight up at cage angle zero.  The cage
turns the whole section about x by theta; every world position here is
"cage frame at theta = 0" unless a theta is passed.

WHY THIS IS A MODULE AND NOT A DRAWING.  Every downstream question -- where
the ring goes, what it must clear, where the loader puts a rod, how long
the thread is -- is a function of these rods and joints.  Computed once
from the spec, they answer for the 300 mm truss, the 1 m truss and the
truss nobody has designed yet; drawn by hand they would have to be drawn
again each time.  numpy only: this runs on the controller.
"""
from dataclasses import dataclass
from math import sqrt, tan, sin, cos, radians, atan2, degrees, pi

import numpy as np

from .spec import Truss

# face f joins chord f to chord f+1; node i of face f sits on the first
# chord when i is even and on the second when i is odd -- which is what
# interleaves the joints on every chord (see Truss)
FACES = ((0, 1), (1, 2), (2, 0))


def chord_phi(k, theta=0.0):
    """Azimuth of chord k in the y-z plane at cage angle theta, degrees."""
    return 90.0 + 120.0 * k + theta


def radial(phi_deg):
    """Unit vector in y-z at azimuth phi (from +y toward +z)."""
    a = radians(phi_deg)
    return np.array([0.0, cos(a), sin(a)])


def rot_x(theta_deg):
    """Rotation about +x by theta, as a 3x3."""
    c, s = cos(radians(theta_deg)), sin(radians(theta_deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def theta_chord_up(k):
    """Cage angle that puts chord k uppermost."""
    return (-120.0 * k) % 360.0


def theta_face_up(f):
    """Cage angle that puts face f's two chords level at the top."""
    return (-120.0 * f - 60.0) % 360.0


# ------------------------------------------------------------------ pieces
@dataclass(frozen=True)
class Rod:
    """A straight rod by its centreline.  For a diagonal p0/p1 are where
    the centreline crosses the two chords' tangent planes -- the joint
    centres -- and the material extends past them by the mitre."""
    kind:  str
    index: int
    p0:    np.ndarray
    p1:    np.ndarray
    r:     float
    chord: int = -1          # chords: which chord
    face:  int = -1          # diagonals: which face
    seg:   int = -1          # diagonals: which segment of the zigzag
    ends:  tuple = ()        # diagonals: (joint index at p0, at p1)

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


@dataclass(frozen=True)
class Joint:
    """Where two mitred diagonals of one face meet a chord."""
    index:  int
    chord:  int
    face:   int
    node:   int              # i within the face's zigzag
    x:      float
    diags:  tuple            # rod indices ending here: one at a chord end, else two
    n_face: np.ndarray       # unit y-z vector from this chord toward the face's other chord

    @property
    def interior(self):
        return len(self.diags) == 2


# ----------------------------------------------------------------- the truss
class TrussGeometry:
    def __init__(self, truss: Truss):
        self.t = truss
        t = truss
        self.R = t.R
        self.xs = t.joint_x()
        self.rods = []
        self.joints = []
        self._chord_axes = {}
        for k in range(t.n_chords):
            d = radial(chord_phi(k)) * self.R
            p0, p1 = d + np.array([0.0, 0, 0]), d + np.array([t.length, 0, 0])
            self._chord_axes[k] = (p0, p1)
            self.rods.append(Rod("chord", len(self.rods), p0, p1,
                                 t.d_chord / 2.0, chord=k))
        # joints first, so diagonals can name theirs
        jidx = {}
        for f, (a, b) in enumerate(FACES):
            for i, x in enumerate(self.xs):
                c = a if i % 2 == 0 else b
                other = b if c == a else a
                n = self.n_between(c, other)
                j = Joint(len(self.joints), c, f, i, x, (), n)
                jidx[(f, i)] = j.index
                self.joints.append(j)
        diags_at = {j.index: [] for j in self.joints}
        for f, (a, b) in enumerate(FACES):
            for i in range(len(self.xs) - 1):
                c0 = a if i % 2 == 0 else b
                c1 = b if c0 == a else a
                n01 = self.n_between(c0, c1)
                q0 = self.chord_point(c0, self.xs[i]) + n01 * (t.d_chord / 2.0)
                q1 = self.chord_point(c1, self.xs[i + 1]) - n01 * (t.d_chord / 2.0)
                j0, j1 = jidx[(f, i)], jidx[(f, i + 1)]
                r = Rod("diag", len(self.rods), q0, q1, t.d_diag / 2.0,
                        face=f, seg=i, ends=(j0, j1))
                self.rods.append(r)
                diags_at[j0].append(r.index)
                diags_at[j1].append(r.index)
        self.joints = [Joint(j.index, j.chord, j.face, j.node, j.x,
                             tuple(diags_at[j.index]), j.n_face)
                       for j in self.joints]
        self.chords = [r for r in self.rods if r.kind == "chord"]
        self.diags = [r for r in self.rods if r.kind == "diag"]

    # ------------------------------------------------------------ helpers
    def chord_point(self, k, x):
        p0, _ = self._chord_axes[k]
        return p0 + np.array([x, 0.0, 0.0])

    def n_between(self, k, m):
        """Unit y-z vector from chord k's axis toward chord m's."""
        v = radial(chord_phi(m)) - radial(chord_phi(k))
        return v / np.linalg.norm(v)

    def joints_on(self, k):
        """The joints of chord k, in x order."""
        return sorted((j for j in self.joints if j.chord == k), key=lambda j: j.x)

    def joints_of_face(self, f):
        return [j for j in self.joints if j.face == f]

    def diag_mitre_normals(self, rod):
        """Outward normals of the two tangent planes that cut a diagonal's
        ends: the face normal at each end, pointing from the chord into the
        rod.  (p0's plane normal, p1's plane normal.)"""
        j0, j1 = (self.joints[i] for i in rod.ends)
        return j0.n_face.copy(), j1.n_face.copy()

    def diag_body_ends(self, rod):
        """Endpoints of a FLAT-ENDED cylinder standing in for the mitred
        rod (Tier 1): pulled in from the tangent-plane crossings so the
        end face's nearest corner just touches the chord."""
        e = (self.t.d_diag / 2.0) / tan(radians(self.t.alpha))
        u = rod.axis
        return rod.p0 + u * e, rod.p1 - u * e

    # ----------------------------------------------------- rotated frames
    def rods_at(self, theta):
        """(p0 [N,3], p1 [N,3], r [N]) of every rod at cage angle theta."""
        Rm = rot_x(theta)
        p0 = np.array([r.p0 for r in self.rods]) @ Rm.T
        p1 = np.array([r.p1 for r in self.rods]) @ Rm.T
        rr = np.array([r.r for r in self.rods])
        return p0, p1, rr

    def joint_at(self, joint, theta):
        """(chord axis point at the joint, face normal) at cage angle theta."""
        Rm = rot_x(theta)
        return Rm @ self.chord_point(joint.chord, joint.x), Rm @ joint.n_face

    def chord_at(self, k, theta):
        Rm = rot_x(theta)
        p0, p1 = self._chord_axes[k]
        return Rm @ p0, Rm @ p1

    def cage_radius(self):
        """The radius the section sweeps: chord axes plus a chord."""
        return self.R + self.t.d_chord / 2.0

    # --------------------------------------------------------- the cluster
    def cluster_points(self, joint, dx, n=48):
        """Section of the joint's cluster by the plane x = joint.x + dx, as
        2-D points (u along n_face, v across it), for the band and the ring.

        The chord is a circle at the origin.  A diagonal on this side of
        the joint has its centreline at d_chord/2 + |dx| tan(alpha) along
        n_face, and its section by an x = const plane is an ellipse
        stretched by 1/cos(alpha) along n_face.  Beyond the rod's material
        (past the far mitre, or on a side with no diagonal) there is only
        the chord.
        """
        t = self.t
        a = radians(t.alpha)
        ang = np.linspace(0.0, 2.0 * pi, n, endpoint=False)
        pts = [np.stack([t.d_chord / 2.0 * np.cos(ang),
                         t.d_chord / 2.0 * np.sin(ang)], axis=1)]
        side = 1 if dx >= 0 else -1
        has = any(self._diag_toward(joint, r, side) for r in joint.diags)
        e = (t.d_diag / 2.0) / tan(a)         # the mitre's reach past the joint
        if has and abs(dx) <= t.L_diag + e:
            rho = t.d_chord / 2.0 + abs(dx) * tan(a)
            # material exists on this side only from -e (the short edge of
            # the mitre) outward
            if dx * side >= -e:
                pts.append(np.stack([rho + (t.d_diag / (2.0 * cos(a))) * np.cos(ang),
                                     (t.d_diag / 2.0) * np.sin(ang)], axis=1))
        return np.concatenate(pts, axis=0)

    def _diag_toward(self, joint, rod_index, side):
        r = self.rods[rod_index]
        other = r.ends[1] if r.ends[0] == joint.index else r.ends[0]
        return (self.joints[other].x - joint.x) * side > 0

    def cluster_perimeter(self, joint, dx):
        """Length of one hoop of thread at axial offset dx: the convex
        hull's perimeter."""
        return hull_perimeter(self.cluster_points(joint, dx))

    def hoop_perimeter(self, joint, dx, thread_d=None):
        """Length of one hoop of thread at axial offset dx -- what the
        THREAD goes round, which is the cluster's hull only where the
        members are within a thread of each other (Truss.bind_half) and
        the chord alone outside it, in both cases offset outward by the
        thread's own radius.

        The hull everywhere was the first model, and Tier 2 refuted it:
        see Truss.bind_half and check_ring.
        """
        t = self.t
        d_t = t.thread_d if thread_d is None else thread_d
        if abs(dx) <= t.bind_half(d_t):
            return hull_perimeter(self.cluster_points(joint, dx)) + pi * d_t
        return pi * (t.d_chord + d_t)

    def cluster_reach(self, joint, dx):
        """How far from the chord axis the cluster reaches at dx -- what
        the ring's inner radius must clear."""
        p = self.cluster_points(joint, dx)
        return float(np.hypot(p[:, 0], p[:, 1]).max())

    def thread_on_joint(self, joint):
        """Thread laid on one joint, mm: turns over the band, pitch = band
        over turns, each turn one hoop of the cluster where it lies."""
        t = self.t
        pitch = t.band / t.turns
        x = np.arange(t.turns) * pitch - t.band / 2.0 + pitch / 2.0
        return float(sum(self.hoop_perimeter(joint, float(dx)) for dx in x))

    def thread_per_chord(self, k):
        """Thread for one continuous run along chord k: bands plus the
        straight strands between them and to the posts."""
        js = self.joints_on(k)
        bands = sum(self.thread_on_joint(j) for j in js)
        strands = sum(max(js[i + 1].x - js[i].x - self.t.band, 0.0)
                      for i in range(len(js) - 1))
        return bands + strands

    def thread_mass(self):
        """Grams of thread on the whole truss."""
        t = self.t
        L = sum(self.thread_per_chord(k) for k in range(t.n_chords))
        return L * pi * (t.thread_d / 2.0) ** 2 * t.thread_rho * 1e-3


# ------------------------------------------------------------------- hull
def convex_hull(pts):
    """Andrew's monotone chain; pts [N,2] -> hull vertices CCW."""
    P = sorted(set(map(tuple, np.asarray(pts, float))))
    if len(P) <= 2:
        return np.array(P)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def hull_perimeter(pts):
    h = convex_hull(pts)
    if len(h) < 2:
        return 0.0
    d = np.diff(np.vstack([h, h[:1]]), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


# ------------------------------------------------------------ distances
def point_segment_distance(P, a, b):
    """Distances from points P [M,3] to segments a,b [N,3] -> [M,N]."""
    ab = b - a                                        # [N,3]
    L2 = np.maximum((ab * ab).sum(axis=1), 1e-12)     # [N]
    ap = P[:, None, :] - a[None, :, :]                # [M,N,3]
    t = np.clip((ap * ab[None, :, :]).sum(axis=2) / L2[None, :], 0.0, 1.0)
    c = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    return np.linalg.norm(P[:, None, :] - c, axis=2)


def clearance(P, p0, p1, rr, skip=None):
    """Least surface clearance from the points P to any rod, mm.  Negative
    means inside a rod.  `skip` masks rods out (a boolean array)."""
    d = point_segment_distance(P, p0, p1) - rr[None, :]
    if skip is not None and np.any(skip):
        d = d[:, ~np.asarray(skip, bool)]
    return float(d.min()) if d.size else float("inf")
