"""One 3-D frame element: the whole of the physics in this package.

A prismatic member with axial, torsional and two-plane bending stiffness,
in SI units.  Everything else here assembles these.

WHY A FRAME AND NOT A BAR.  The chords run through every joint uncut, so
they carry bending; a pin-jointed bar model would leave the lattice a
mechanism in some topologies and would miss the local flexibility at the
mount and the ends, which is exactly where the camera faces are.  The
diagonals are a different question -- a wound and bonded lashing is
somewhere between a pin and a moment connection -- so they can be built
either way and the answer reported for both (`pinned` below).  That
bracket, rather than a guessed joint stiffness, is the honest treatment
until a rig measures one.

LOCAL AXES: x along the member from node i to node j; z as close to global
+z as the member allows, or global +x for a vertical member; y = z cross x.
DOF ORDER, per node: ux uy uz rx ry rz -- twelve for the element.

The consistent mass matrix is used rather than a lumped one.  A lumped
matrix has zeros on the rotational diagonal, which makes the generalised
eigenproblem singular and has to be patched with an invented rotary
inertia; the consistent matrix is positive definite as it stands.
"""
import numpy as np


def local_axes(p_i, p_j, up=None):
    """(3,3) matrix whose ROWS are the local x, y, z axes in global."""
    x = np.asarray(p_j, float) - np.asarray(p_i, float)
    L = float(np.linalg.norm(x))
    if L <= 0.0:
        raise ValueError("zero-length member")
    x = x / L
    if up is None:
        # a member within 1 degree of vertical has no useful projection of
        # global z; borrow global x instead
        up = np.array([1.0, 0.0, 0.0]) if abs(x[2]) > 0.9998 else np.array([0.0, 0.0, 1.0])
    up = np.asarray(up, float)
    y = np.cross(up, x)
    n = np.linalg.norm(y)
    if n < 1e-12:
        raise ValueError("the up vector is parallel to the member")
    y = y / n
    z = np.cross(x, y)
    return np.array([x, y, z]), L


def transform(R):
    """(12,12) block-diagonal rotation from global to local."""
    T = np.zeros((12, 12))
    for k in range(4):
        T[3 * k:3 * k + 3, 3 * k:3 * k + 3] = R
    return T


def stiffness_local(E, G, A, Iy, Iz, J, L, pinned=False):
    """(12,12) local stiffness.

    `pinned` releases the bending moments at both ends, which condenses
    the element to a bar: axial and torsion only.  (Condensing a pair of
    4EI/L, 2EI/L rotations leaves exactly zero transverse stiffness, so
    the release is done by omission rather than by inverting a 2x2.)
    """
    k = np.zeros((12, 12))
    k[0, 0] = k[6, 6] = E * A / L
    k[0, 6] = k[6, 0] = -E * A / L
    k[3, 3] = k[9, 9] = G * J / L
    k[3, 9] = k[9, 3] = -G * J / L
    if pinned:
        return k
    # bending in the local x-y plane: uy (1, 7) with rz (5, 11)
    a, b = 12.0 * E * Iz / L ** 3, 6.0 * E * Iz / L ** 2
    c, d = 4.0 * E * Iz / L, 2.0 * E * Iz / L
    idx = [1, 5, 7, 11]
    kb = np.array([[a, b, -a, b],
                   [b, c, -b, d],
                   [-a, -b, a, -b],
                   [b, d, -b, c]])
    k[np.ix_(idx, idx)] += kb
    # bending in the local x-z plane: uz (2, 8) with ry (4, 10); the
    # rotation's sign is opposite, which is where sign errors live
    a, b = 12.0 * E * Iy / L ** 3, 6.0 * E * Iy / L ** 2
    c, d = 4.0 * E * Iy / L, 2.0 * E * Iy / L
    idx = [2, 4, 8, 10]
    kb = np.array([[a, -b, -a, -b],
                   [-b, c, b, d],
                   [-a, b, a, b],
                   [-b, d, b, c]])
    k[np.ix_(idx, idx)] += kb
    return k


def mass_local(rho, A, Ip, L, pinned=False):
    """(12,12) consistent mass.  `Ip` is the polar second moment of area,
    for the torsional inertia."""
    m = rho * A * L
    M = np.zeros((12, 12))
    # axial
    M[0, 0] = M[6, 6] = m / 3.0
    M[0, 6] = M[6, 0] = m / 6.0
    # torsion
    jt = rho * Ip * L
    M[3, 3] = M[9, 9] = jt / 3.0
    M[3, 9] = M[9, 3] = jt / 6.0
    if pinned:
        # a bar still carries its own transverse mass; without it the
        # diagonals would be weightless sideways and the modes wrong
        for t in (1, 2):
            M[t, t] = M[t + 6, t + 6] = m / 3.0
            M[t, t + 6] = M[t + 6, t] = m / 6.0
        return M
    c = m / 420.0
    idx = [1, 5, 7, 11]
    mb = c * np.array([[156.0, 22.0 * L, 54.0, -13.0 * L],
                       [22.0 * L, 4.0 * L * L, 13.0 * L, -3.0 * L * L],
                       [54.0, 13.0 * L, 156.0, -22.0 * L],
                       [-13.0 * L, -3.0 * L * L, -22.0 * L, 4.0 * L * L]])
    M[np.ix_(idx, idx)] += mb
    idx = [2, 4, 8, 10]
    mb = c * np.array([[156.0, -22.0 * L, 54.0, 13.0 * L],
                       [-22.0 * L, 4.0 * L * L, -13.0 * L, -3.0 * L * L],
                       [54.0, -13.0 * L, 156.0, 22.0 * L],
                       [13.0 * L, -3.0 * L * L, 22.0 * L, 4.0 * L * L]])
    M[np.ix_(idx, idx)] += mb
    return M


def thermal_local(E, A, alpha, dT, Iy=0.0, Iz=0.0, gy=0.0, gz=0.0):
    """(12,) local equivalent nodal load for a temperature field.

    `dT` is the mean rise over the section; `gy`, `gz` are its gradients
    across the LOCAL section axes, in kelvin per metre.

    THE GRADIENT TERM IS NOT OPTIONAL IN A COMPARISON.  A beam whose
    temperature varies across its depth bows, at curvature alpha times the
    gradient: that is a mast leaning away from the sun, and it is the
    classic thermal error of a monolithic tube.  Left out, a tube modelled
    as a line of nodes reports exactly zero thermal distortion and beats
    every lattice for the wrong reason.

    A free bar must come out alpha*dT*L longer and a free beam must take
    curvature alpha*g; check_frame asserts both, which is what pins these
    signs.
    """
    f = np.zeros(12)
    N = E * A * alpha * dT
    f[0], f[6] = -N, +N
    Mz = E * Iz * alpha * gy
    f[5], f[11] = -Mz, +Mz
    My = E * Iy * alpha * gz
    f[4], f[10] = +My, -My
    return f
