"""The element and the solver, against closed forms.

Everything this package concludes rests on these forty lines of element
matrices, so they are checked against cases with exact answers before any
spine is judged by them.  A frame element with cubic shape functions is
EXACT for a beam loaded at its ends, so most of these are equalities, not
approximations, and the tolerances are tight on purpose: a sign error in
the bending block or a missing factor in the mass matrix would not shift
these numbers slightly, it would break them.

    python3 sim/scripts/spine/check_frame.py [-v]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np

from spine import frame
from spine.model import Model, Member, Section, PointMass
from spine.material import Material

VERBOSE = "-v" in sys.argv
RESULTS = []
MAT = Material("test steel", E=200e9, G=80e9, rho=7850.0, alpha=1.2e-5)


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def beam(n, L, sec, mat=MAT, along=(1, 0, 0)):
    u = np.asarray(along, float)
    u = u / np.linalg.norm(u)
    nodes = [(u * (i * L / (n - 1))).tolist() for i in range(n)]
    mem = [Member(i, i + 1, sec, mat) for i in range(n - 1)]
    return Model(nodes, mem, faces={"tip": [n - 1]}, depth=0.05)


def main():
    L, sec = 1.0, Section.rod(0.01)
    E, I, A = MAT.E, sec.Iy, sec.A
    J, G = sec.J, MAT.G

    # --------------------------------------------------- the matrices
    k = frame.stiffness_local(E, G, A, I, I, J, L)
    check("the element stiffness is symmetric", np.allclose(k, k.T))
    check("...and has exactly six rigid-body modes",
          np.linalg.matrix_rank(k, tol=1e-6 * abs(k).max()) == 6,
          "rank %d" % np.linalg.matrix_rank(k, tol=1e-6 * abs(k).max()))
    M = frame.mass_local(MAT.rho, A, sec.Ip, L)
    check("the consistent mass is symmetric and positive definite",
          np.allclose(M, M.T) and np.linalg.eigvalsh(M).min() > 0,
          "least eigenvalue %.3g" % np.linalg.eigvalsh(M).min())
    check("...and its total is the member's mass",
          abs(M[0, 0] + 2 * M[0, 6] + M[6, 6] - MAT.rho * A * L) < 1e-12,
          "%.6g vs %.6g" % (M[0, 0] + 2 * M[0, 6] + M[6, 6], MAT.rho * A * L))

    # ------------------------------------------------------- statics
    for tag, along, dof, rot in (("x", (1, 0, 0), 2, 4), ("y", (0, 1, 0), 2, 3),
                                 ("z", (0, 0, 1), 0, 4)):
        m = beam(11, L, sec, along=along)
        m.fix(0)
        f = np.zeros(m.n * 6)
        f[m.dofs(10)[dof]] = 1.0
        u = m.solve(f)
        got = u[m.dofs(10)[dof]]
        check("a cantilever along %s deflects P L^3 / 3EI" % tag,
              abs(got / (L ** 3 / (3 * E * I)) - 1.0) < 1e-9,
              "%.6e vs %.6e" % (got, L ** 3 / (3 * E * I)))
        slope = abs(u[m.dofs(10)[rot]])
        check("...and its tip slope is P L^2 / 2EI (the camera's yaw)",
              abs(slope / (L ** 2 / (2 * E * I)) - 1.0) < 1e-9,
              "%.6e vs %.6e" % (slope, L ** 2 / (2 * E * I)))
    m = beam(11, L, sec)
    m.fix(0)
    f = np.zeros(m.n * 6)
    f[m.dofs(10)[3]] = 1.0
    u = m.solve(f)
    check("a shaft twists T L / GJ, on the shear modulus and not on E",
          abs(u[m.dofs(10)[3]] / (L / (G * J)) - 1.0) < 1e-9,
          "%.6e vs %.6e" % (u[m.dofs(10)[3]], L / (G * J)))
    f = np.zeros(m.n * 6)
    f[m.dofs(10)[0]] = 1000.0
    u = m.solve(f)
    check("a bar stretches P L / EA", abs(u[m.dofs(10)[0]] / (1000.0 * L / (E * A)) - 1.0) < 1e-9)
    N = m.member_forces(u)
    check("...and every member of it carries the load",
          np.allclose(N, 1000.0, rtol=1e-9), "%.3f..%.3f N" % (N.min(), N.max()))

    # ------------------------------------------------------ thermal
    m = beam(11, L, sec)
    m.fix(0)
    u = m.solve(m.thermal_load(lambda p: 40.0))
    check("a free bar warmed 40 K grows alpha dT L, and no more",
          abs(u[m.dofs(10)[0]] - MAT.alpha * 40.0 * L) < 1e-14,
          "%.6e vs %.6e" % (u[m.dofs(10)[0]], MAT.alpha * 40.0 * L))
    dT = m.member_dT(lambda p: 40.0)
    check("...and carries no force while it does",
          np.allclose(m.member_forces(u, dT), 0.0, atol=1e-6),
          "%.3g N" % np.abs(m.member_forces(u, dT)).max())
    m2 = beam(11, L, sec)
    m2.fix(0, dofs=range(6))
    m2.fix(10, dofs=[0])
    u2 = m2.solve(m2.thermal_load(lambda p: 40.0))
    dT2 = m2.member_dT(lambda p: 40.0)
    want = -E * A * MAT.alpha * 40.0
    check("a bar held at both ends and warmed carries -EA alpha dT",
          abs(m2.member_forces(u2, dT2)[0] / want - 1.0) < 1e-9,
          "%.1f N vs %.1f" % (m2.member_forces(u2, dT2)[0], want))
    g = 100.0
    m3 = beam(11, L, Section.tube(0.04, 0.001))
    m3.fix(0)
    u3 = m3.solve(m3.thermal_load(lambda p: g * p[2], probe=0.02))
    check("a beam with a gradient across it bows at curvature alpha times the gradient",
          abs(-u3[m3.dofs(10)[4]] / (MAT.alpha * g * L) - 1.0) < 1e-9,
          "%.6e vs %.6e" % (-u3[m3.dofs(10)[4]], MAT.alpha * g * L))

    # -------------------------------------------------------- modes
    m = beam(21, L, sec)
    m.fix(0)
    f_hz, _ = m.modes(3)
    want = 3.51602 * np.sqrt(E * I / (MAT.rho * A * L ** 4)) / (2 * np.pi)
    check("a bare cantilever's first mode is the Euler-Bernoulli one",
          abs(f_hz[0] / want - 1.0) < 2e-3, "%.4f Hz vs %.4f" % (f_hz[0], want))
    check("...and the second is 6.27 times the first",
          abs(f_hz[2] / f_hz[0] / 6.2669 - 1.0) < 5e-3,
          "%.3f" % (f_hz[2] / f_hz[0]))
    m = beam(21, L, sec, )
    m.masses = [PointMass(20, 100.0 * MAT.rho * A * L)]
    m._K = m._M = None
    m.fix(0)
    f_hz, _ = m.modes(2)
    k_tip = 3 * E * I / L ** 3
    want = np.sqrt(k_tip / (100.0 * MAT.rho * A * L)) / (2 * np.pi)
    check("a cantilever under a heavy tip mass is the spring-mass oscillator",
          abs(f_hz[0] / want - 1.0) < 5e-3, "%.4f Hz vs %.4f" % (f_hz[0], want))
    free = beam(11, L, sec)
    f_hz, _ = free.modes(8)
    # the six rigid modes come back at the square root of machine epsilon
    # times the elastic ones, which is as close to zero as a Cholesky
    # reduction of a well-conditioned mass matrix can put them
    check("an unsupported structure has six rigid modes and no more",
          (f_hz[:6] < 1e-4 * f_hz[6]).all() and f_hz[6] > 1.0,
          "worst rigid %.2e Hz against a first elastic mode of %.2f" % (f_hz[:6].max(), f_hz[6]))

    # ------------------------------------------------- the machinery
    m = beam(11, L, sec)
    m.fix(0)
    f = np.zeros(m.n * 6)
    f[m.dofs(10)[2]] = 1.0
    u = m.solve(f)
    K, _ = m.assemble()
    check("the work of the load equals twice the strain energy",
          abs(float(f @ u) / float(u @ K @ u) - 1.0) < 1e-9,
          "%.9f" % (float(f @ u) / float(u @ K @ u)))
    m = beam(11, L, sec)
    m.fix(0)
    m.ground_spring(10, 1e3, 0.0)
    u2 = m.solve(f)
    check("a ground spring softens what it is attached to",
          u2[m.dofs(10)[2]] < u[m.dofs(10)[2]],
          "%.4e with the spring, %.4e without" % (u2[m.dofs(10)[2]], u[m.dofs(10)[2]]))
    m = beam(6, L, sec)
    m.members = [Member(mb.i, mb.j, mb.section, mb.material, pinned=True) for mb in m.members]
    m._K = m._M = None
    m.fix(0)
    try:
        m.solve(f)
        ok = False
    except ValueError as e:
        ok = "restrains" in str(e)
    check("a chain of bars is refused, not solved: the solver says which freedom is loose", ok)

    bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det and (VERBOSE or not ok) else ""))
    print("check_frame: %d checks, %d failed" % (len(RESULTS), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
