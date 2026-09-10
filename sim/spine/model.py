"""A spine as a 3-D frame: assemble, solve, and read the camera faces.

Nothing here knows what a truss is.  It takes nodes, members and sections,
and answers three questions: where does it go under a load, what are its
modes, and what does that do to the two camera faces.  A tube and a
lattice are the same object to this module, which is the point -- the
comparison the whole exercise turns on has to happen inside one solver.

A CAMERA FACE IS A SET OF NODES, and its pose is the rigid-body motion
that best explains their displacements.  That is what a bracket bolted to
three chord ends actually sees.  Where a face is a single node -- the end
of a tube, bonded to its section -- the node's own rotational freedoms
are the face's rotation, because that is what the bonded joint sees.  The
two definitions differ because the two mountings differ, not because the
code is inconsistent.

SI units.  Frame convention, fixed by what the cameras do:

    x   along the spine, and therefore the BASELINE
    y   forward, the optical axes
    z   up

so a rotation about z swings an optical axis toward or away from its
partner, which is the yaw that biases disparity, and a rotation about x
tips it up or down, which rectification can absorb.
"""
from dataclasses import dataclass, field

import numpy as np

from . import frame


# ------------------------------------------------------------- sections
@dataclass(frozen=True)
class Section:
    """Area and second moments, SI (m^2, m^4)."""
    name: str
    A:  float
    Iy: float
    Iz: float
    J:  float
    Ip: float          # polar second moment, for rotational inertia

    @staticmethod
    def rod(d):
        """Solid circular rod of diameter d (m)."""
        A = np.pi * d * d / 4.0
        I = np.pi * d ** 4 / 64.0
        return Section("rod %.1f mm" % (d * 1e3), A, I, I, 2.0 * I, 2.0 * I)

    @staticmethod
    def rect(w, h):
        """A rectangular bar, w across by h thick."""
        A = w * h
        return Section("bar %.0fx%.0f mm" % (w * 1e3, h * 1e3), A,
                       w * h ** 3 / 12.0, h * w ** 3 / 12.0,
                       w * h ** 3 / 3.0, A * (w * w + h * h) / 12.0)

    @staticmethod
    def tube(d_out, wall):
        d_in = d_out - 2.0 * wall
        A = np.pi * (d_out ** 2 - d_in ** 2) / 4.0
        I = np.pi * (d_out ** 4 - d_in ** 4) / 64.0
        return Section("tube %.1fx%.1f mm" % (d_out * 1e3, wall * 1e3),
                       A, I, I, 2.0 * I, 2.0 * I)


@dataclass(frozen=True)
class Member:
    i: int
    j: int
    section: Section
    material: object
    pinned: bool = False
    tag: str = ""


@dataclass
class PointMass:
    """A lumped mass at a node, kg, with its own ROTATIONAL inertia.

    The inertia is not decoration.  A point mass contributes to three
    translational freedoms and nothing else, so a node whose only members
    are massless has no rotational inertia at all and the mass matrix is
    singular -- Cholesky fails, with a message about positive definiteness
    that says nothing about the cause.  That went unnoticed while every mass
    sat on a node carrying real rods, whose consistent mass supplied it by
    accident.  A camera on a massless housing does not, and a camera is a
    box with a real inertia, so it is asked for.
    """
    node: int
    m: float
    inertia: tuple = (0.0, 0.0, 0.0)     # kg.m^2 about the node's own axes

    @staticmethod
    def box(node, mass_kg, sides_m):
        """A cuboid's mass and its three principal inertias about its own
        centre: m(b^2 + c^2)/12 and cyclic."""
        a, b, c = (float(v) for v in sides_m)
        m = float(mass_kg)
        return PointMass(node, m, (m * (b * b + c * c) / 12.0,
                                   m * (a * a + c * c) / 12.0,
                                   m * (a * a + b * b) / 12.0))


def _skew(r):
    return np.array([[0.0, -r[2], r[1]],
                     [r[2], 0.0, -r[0]],
                     [-r[1], r[0], 0.0]])


class Model:
    """Nodes, members, supports; assembly and the solves."""

    DOF = 6

    def __init__(self, nodes, members, masses=(), faces=None, name="", depth=None):
        self.nodes = np.asarray(nodes, float)
        self.members = list(members)
        self.masses = list(masses)
        self.faces = dict(faces or {})
        self.name = name
        # THE PHYSICAL SECTION DEPTH, which is not the spread of the nodes.
        # A tube is modelled as a line of nodes on its own axis and spans
        # 43 mm all the same; a thermal gradient normalised by the node
        # spread instead divides by zero and reports a tube bowing by a
        # kilometre.  Builders set this; the node spread is the fallback.
        self.depth = float(depth) if depth else float(
            2.0 * np.linalg.norm(self.nodes[:, 1:], axis=1).max())
        self.n = len(self.nodes)
        self.fixed = np.zeros(self.n * self.DOF, dtype=bool)
        self.springs = np.zeros(self.n * self.DOF)
        self._K = self._M = None

    # ------------------------------------------------------------ index
    def dofs(self, node):
        a = node * self.DOF
        return np.arange(a, a + self.DOF)

    def fix(self, nodes, dofs=range(6)):
        for nd in np.atleast_1d(nodes):
            for k in dofs:
                self.fixed[int(nd) * self.DOF + k] = True

    def ground_spring(self, nodes, k_lin, k_rot):
        """A compliant mount: each named node to ground, N/m and N.m/rad.

        Here because the mount can be the softest thing in the assembly,
        and a spine optimised while the bracket carries the compliance is
        an optimisation of the wrong part.
        """
        for nd in np.atleast_1d(nodes):
            d = self.dofs(int(nd))
            self.springs[d[:3]] += k_lin
            self.springs[d[3:]] += k_rot

    # --------------------------------------------------------- assembly
    def _element(self, mb):
        R, L = frame.local_axes(self.nodes[mb.i], self.nodes[mb.j])
        T = frame.transform(R)
        m, s = mb.material, mb.section
        k = frame.stiffness_local(m.E, m.G, s.A, s.Iy, s.Iz, s.J, L, mb.pinned)
        M = frame.mass_local(m.rho, s.A, s.Ip, L, mb.pinned)
        return T.T @ k @ T, T.T @ M @ T, T, L

    def assemble(self):
        if self._K is not None:
            return self._K, self._M
        nd = self.n * self.DOF
        K = np.zeros((nd, nd))
        M = np.zeros((nd, nd))
        for mb in self.members:
            kg, mg, _T, _L = self._element(mb)
            d = np.concatenate([self.dofs(mb.i), self.dofs(mb.j)])
            K[np.ix_(d, d)] += kg
            M[np.ix_(d, d)] += mg
        for pm in self.masses:
            d = self.dofs(pm.node)
            for t in range(3):
                M[d[t], d[t]] += pm.m
                M[d[3 + t], d[3 + t]] += float(pm.inertia[t])
        K[np.diag_indices(nd)] += self.springs
        self._K, self._M = K, M
        return K, M

    def free(self):
        return np.where(~self.fixed)[0]

    # ------------------------------------------------------------ loads
    def accel_load(self, direction, a):
        """Equivalent static load for an acceleration `a` (m/s^2) along
        `direction`: the same construction as gravity, so the structure
        deflects the way it is pushed."""
        K, M = self.assemble()
        r = np.zeros(self.n * self.DOF)
        u = np.asarray(direction, float)
        u = u / np.linalg.norm(u)
        for k in range(3):
            r[k::self.DOF] = u[k]
        return M @ (r * a)

    def angular_accel_load(self, axis, alpha, centre=None):
        """Equivalent static load for an angular acceleration `alpha`
        (rad/s^2) about `axis` through `centre` (default: the mount).

        THE CASE WHERE A LATTICE IS WEAKEST.  A closed tube carries torque
        in shear flow and is enormously stiff in it; an open three-chord
        web carries it in its diagonals.  Leaving this case out of the
        comparison would flatter the truss, so it is in.
        """
        if centre is None:
            centre = self.face_centre("mount") if "mount" in self.faces \
                else self.nodes.mean(axis=0)
        w = np.asarray(axis, float)
        w = alpha * w / np.linalg.norm(w)
        a = np.zeros(self.n * self.DOF)
        for nd in range(self.n):
            acc = np.cross(w, self.nodes[nd] - np.asarray(centre, float))
            a[self.dofs(nd)[:3]] = acc
        _K, M = self.assemble()
        return M @ a

    def thermal_load(self, field_fn, probe=None):
        """Equivalent nodal load for a temperature field, `field_fn(p)`
        giving the rise in kelvin at a point (m).

        The field is sampled at the member's midpoint for the mean rise
        and differenced across its own section for the gradient, so a
        member thick enough to see a gradient bows and a thin rod does not.
        """
        f = np.zeros(self.n * self.DOF)
        for mb in self.members:
            R, L = frame.local_axes(self.nodes[mb.i], self.nodes[mb.j])
            T = frame.transform(R)
            mid = 0.5 * (self.nodes[mb.i] + self.nodes[mb.j])
            dT = float(field_fn(mid))
            h = probe if probe else np.sqrt(mb.section.A)   # the member's own width
            gy = (float(field_fn(mid + R[1] * h)) - float(field_fn(mid - R[1] * h))) / (2.0 * h)
            gz = (float(field_fn(mid + R[2] * h)) - float(field_fn(mid - R[2] * h))) / (2.0 * h)
            fl = frame.thermal_local(mb.material.E, mb.section.A, mb.material.alpha, dT,
                                     mb.section.Iy, mb.section.Iz, gy, gz)
            d = np.concatenate([self.dofs(mb.i), self.dofs(mb.j)])
            f[d] += T.T @ fl
        return f

    def member_dT(self, field_fn):
        return np.array([float(field_fn(0.5 * (self.nodes[m.i] + self.nodes[m.j])))
                         for m in self.members])

    # ------------------------------------------------------------ solve
    def solve(self, f):
        """Displacements under load f, with the supports applied."""
        K, _M = self.assemble()
        free = self.free()
        Kff = K[np.ix_(free, free)]
        weak = np.where(np.abs(np.diag(Kff)) < 1e-12 * max(np.abs(np.diag(Kff)).max(), 1.0))[0]
        if len(weak):
            nd = free[weak[0]] // self.DOF
            raise ValueError("node %d has a freedom nothing restrains (dof %d): a bar-only "
                             "joint needs a continuous member through it"
                             % (nd, free[weak[0]] % self.DOF))
        u = np.zeros(self.n * self.DOF)
        u[free] = np.linalg.solve(Kff, f[free])
        return u

    def modes(self, n=8):
        """(frequencies Hz, shapes) of the n lowest modes, supports applied.

        Generalised symmetric eigenproblem by Cholesky of the mass matrix,
        which is positive definite because the element mass is consistent.
        """
        K, M = self.assemble()
        free = self.free()
        Kff, Mff = K[np.ix_(free, free)], M[np.ix_(free, free)]
        # A NODE WITH MASS AND NO ROTATIONAL INERTIA makes this singular, and
        # the message numpy gives for that says nothing about the cause.
        bad = np.where(np.diag(Mff) <= 0.0)[0]
        if len(bad):
            raise ValueError(
                "%d free degrees of freedom carry no inertia (first at index %d): a "
                "PointMass with no `inertia` on a node whose members are massless "
                "leaves its three rotations empty" % (len(bad), int(bad[0])))
        L = np.linalg.cholesky(Mff)
        Li = np.linalg.inv(L)
        A = Li @ Kff @ Li.T
        A = 0.5 * (A + A.T)
        w2, V = np.linalg.eigh(A)
        w2 = np.clip(w2, 0.0, None)
        f_hz = np.sqrt(w2) / (2.0 * np.pi)
        shapes = np.zeros((len(f_hz), self.n * self.DOF))
        Phi = Li.T @ V
        shapes[:, free] = Phi.T
        return f_hz[:n], shapes[:n]

    # ------------------------------------------------------------ faces
    def face_pose(self, u, name):
        """(translation, rotation) of a camera face, metres and radians."""
        ids = list(self.faces[name])
        if len(ids) == 1:
            d = self.dofs(ids[0])
            return u[d[:3]].copy(), u[d[3:]].copy()
        P = self.nodes[ids]
        c = P.mean(axis=0)
        A, b = [], []
        for k, nid in enumerate(ids):
            r = P[k] - c
            A.append(np.hstack([np.eye(3), -_skew(r)]))
            b.append(u[self.dofs(nid)[:3]])
        sol, *_ = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)
        return sol[:3], sol[3:]

    def face_centre(self, name):
        return self.nodes[list(self.faces[name])].mean(axis=0)

    # ----------------------------------------------------------- forces
    def member_forces(self, u, dT=None):
        """Axial force in each member, N, tension positive.  The thermal
        strain is subtracted: a free bar that has expanded carries none."""
        out = np.zeros(len(self.members))
        for k, mb in enumerate(self.members):
            R, L = frame.local_axes(self.nodes[mb.i], self.nodes[mb.j])
            d = np.concatenate([self.dofs(mb.i), self.dofs(mb.j)])
            ul = frame.transform(R) @ u[d]
            eps = (ul[6] - ul[0]) / L
            if dT is not None:
                eps -= mb.material.alpha * dT[k]
            out[k] = mb.material.E * mb.section.A * eps
        return out

    @property
    def mass(self):
        m = sum(mb.material.rho * mb.section.A *
                float(np.linalg.norm(self.nodes[mb.j] - self.nodes[mb.i]))
                for mb in self.members)
        return m + sum(pm.m for pm in self.masses)

    @property
    def structural_mass(self):
        return sum(mb.material.rho * mb.section.A *
                   float(np.linalg.norm(self.nodes[mb.j] - self.nodes[mb.i]))
                   for mb in self.members)
