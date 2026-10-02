"""A rigid body's pose from many calibrated cameras -- the clamp's, in every
frame -- and the numerical core the rig's estimators share.

What is measured.  Each camera reports the centroid of every ball it sees,
to about 0.07 px (1 sigma) per axis.  That reprojection error is the one
quantity whose noise is known, so it is the quantity minimised: the pose is
the rigid motion of the body that puts every ball, through every camera's
calibrated lens (lens.Camera.project, with the ball's radius), nearest to
where it was seen.  Under Gaussian centroid noise that is the maximum-
likelihood pose (Hartley & Zisserman 2004, 4.2 and ch. 18).  Averaging a
pose per camera, or fitting the body's points to triangulated ones, would
weight the evidence by the geometry instead of by the noise; triangulation
is used here only to start.

The method:

  start   the encoders' pose if there is one; otherwise every ball seen by
          two or more cameras is triangulated linearly (the DLT of Hartley
          & Zisserman 2004, 12.2, in its inhomogeneous form on normalised,
          undistorted rays, which are already of order one, so Hartley's
          conditioning has nothing left to do) and the body's points are
          fitted onto them by the least-squares rigid map (Kabsch 1976;
          Umeyama 1991, whose determinant guard keeps it a rotation).
          triangulate() also refines a point by Gauss-Newton on its
          reprojection error, for callers that want the point; the pose's
          start does not, since the pose's own fit redoes that work.
  refine  Levenberg-Marquardt (Levenberg 1944; Marquardt 1963, with his
          scaling of the damping by the normal matrix's diagonal) on every
          observation's reprojection error.  Rotations are updated by a
          rotation vector on the left, R <- exp([dphi]x) R, never by Euler
          angles; for a body pose (body -> world) dphi is in world axes.
  robust  Huber's loss (Huber 1964) by iteratively reweighted least squares
          while the fit settles, so a bad centroid pulls with a bounded
          force; then every observation the other observations cannot
          explain is rejected, and a plain least-squares solve on the rest
          gives the answer and, from its normal matrix, the covariance.
          A centroid that is not a finite number is not used at all.

calib.py uses the same projection derivatives, robust rules and solver, so
a residual is judged by one rule in calibration and in operation.

The rules, and why each number is what it is:

  HUBER_K       1.345 robust sigmas: Huber's knee for 95 % asymptotic
                efficiency at the Gaussian (Huber 1964; Holland & Welsch
                1977).  It costs 5 % of the variance on clean data and
                bounds the pull of any one bad centroid.
  MAD_SIGMA     1 / Phi^-1(3/4) = 1.4826: the median absolute deviation's
                consistency factor for a Gaussian (Hampel 1974), computed
                below, not typed.
  MAD_EFF       8 phi(q)^2 q^2 = 0.368, q = Phi^-1(3/4): the MAD's
                asymptotic efficiency at the Gaussian, the variance of the
                sample median of |r| (1 / (4 N g(q)^2), g = 2 phi the
                density of |r|) against the standard deviation's; a MAD
                scale from N residuals is as good as a sum of squares of
                0.368 N.  Computed, not typed.
  rejection     an observation is rejected when the OTHER inliers cannot
                explain it: the mean-shift outlier test (Belsley, Kuh &
                Welsch 1980; Cook & Weisberg 1982, 2.2), on its 2-D
                residual.  Its leverage block G = J_i (J^T J)^-1 J_i^T (2x2)
                is how much of its noise the fit absorbs: an inlier's
                residual is shrunk to covariance sigma^2 (I - G), a rejected
                one's prediction is uncertain by sigma^2 (I + G), and
                d = r^T (I -+ G)^-1 r is in both cases the part of the sum
                of squares that the observation alone accounts for.  With
                the noise stated, d / sigma^2 is chi-square with 2 dof,
                tail exp(-d / 2 sigma^2).  Without it, the scale is the
                other inliers' own: RSS' on nu' = 2m - p dof (less 2 for an
                inlier, whose own residual is taken out), and
                (d / 2) / (RSS' / nu') is F(2, nu'), tail (1 + d / RSS')^
                (-nu'/2) -- exact under Gaussian noise.  (Treating a scale
                estimated from the same few residuals as if it were the
                noise lost a genuine centroid in 4 % of held-out frames,
                and in a quarter of frames of six observations, for a
                claimed 1 %.)  An observation is rejected when its tail is
                under FALSE_REJECT / n, n the observations tested, so a
                genuine one is lost in fewer than FALSE_REJECT of data sets
                (Bonferroni's bound).  FALSE_REJECT is 1 %: one data set in
                a hundred loses one good centroid, which costs nothing; a
                gross outlier of a few pixels is fifty sigmas and goes
                whatever the rate.  One the others cannot judge is kept:
                the only observation fixing some direction (I - G singular
                -- to within the inverse's round-off, n eps cond, which is
                what 1 - h can be known to: a camera fixed by barely more
                residuals than it has parameters has leverages of 1 - 1e-8
                that are 1 in exact arithmetic), or any, when there are too
                few dof for a scale.
                The first cut, before there is an inlier fit to judge by, is
                by the robust scale (the MAD, which outliers do not move).
                With the noise stated, a residual beyond reject_limit(n) =
                sqrt(2 ln(n / FALSE_REJECT)) sigmas goes: the same tail.
                Without it the MAD is corrected for the dof the fit took
                (by sqrt(N / nu), N residuals, nu = N - p) and its own
                sampling error is allowed for: it is worth MAD_EFF nu
                residuals, so a residual is judged by F(2, MAD_EFF nu).
                In a frame of a few balls that is what keeps the cut from
                taking genuine centroids on a scale that happened to come
                out small.
                Every later pass re-tests every observation against the
                inliers' fit, so a genuine one the cut took comes back, and
                takes inliers out one at a time, the worst first (Baarda
                1968's data snooping): each test assumes the other inliers
                right, and a fit's residuals are correlated, so two
                sightings of one ball that fail together may be one bad one
                and its victim -- and taking out both would leave the ball
                unseen.  The test of an observation does not depend on
                whether it is in the fit, so the passes do not oscillate
                over it.
  null space    a direction of the parameters the data do not fix is a zero
                eigenvalue of the (Jacobi scaled) normal matrix -- zero
                meaning under numpy's matrix_rank tolerance, n eps
                lambda_max, the round-off of forming it.  Every parameter
                with more than TINY of its unit vector in such a direction
                has NaN covariance: a pseudo-inverse would report the
                minimum-norm answer's spread instead, finite and wrong.
  STEP_TOL      0.01: converged when the Gauss-Newton step to the local
                minimum is under a hundredth of a standard deviation.  Its
                Mahalanobis length bounds the change of every derived
                quantity in units of that quantity's own sigma (Cauchy-
                Schwarz), so a smaller step is one the data cannot see.
  ROBUST_TOL    1: the robust phase only has to find the outliers.  Once its
                step is under one standard deviation (of what the residuals
                inside Huber's knee know), no genuine residual is more than
                about a sigma from where the final fit puts it, which cannot
                carry it across a rejection limit of four to five sigmas
                unless it already sat on it -- and the inlier set is
                re-tested after the final least squares in any case.
                Converging it further buys nothing and, where a direction is
                fixed only by outliers (a sample held by one bad ball), the
                reweighting creeps along the loss's linear part for many
                iterations.
  LAMBDA0       1e-3: where the damping starts (Press et al., Numerical
                Recipes, 15.5); from the first step on it follows Nielsen's
                rule (Nielsen 1999; Madsen, Nielsen & Tingleff 2004).
  TINY          sqrt(machine epsilon) = 1.5e-8: in px, far above the
                round-off of a pixel coordinate (eps x 2000 px = 4e-13 px)
                and far below any centroid noise, so a scale under it means
                an exactly determined fit whose residuals are round-off, and
                the scale is floored there rather than divided by.  As a
                fraction (a leverage's complement, a share of a null
                direction) it is far above the round-off of an eigenvector
                and far below any coupling the data mean.
"""
import math
from dataclasses import dataclass

import numpy as np

from .lens import angle_between, rodrigues


# ================================================================ RULES
def normal_quantile(p):
    """z with Phi(z) = p, by bisection on math.erfc: exact to round-off and
    needs nothing but the standard library."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    lo, hi = -40.0, 40.0                 # Phi(-40) underflows: every p a double holds lies inside
    while hi - lo > 4.0 * np.finfo(float).eps * max(1.0, abs(lo), abs(hi)):
        mid = 0.5 * (lo + hi)
        if mid in (lo, hi):
            break
        if 0.5 * math.erfc(-mid / math.sqrt(2.0)) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


HUBER_K = 1.345
MAD_SIGMA = 1.0 / normal_quantile(0.75)
MAD_EFF = 8.0 * (normal_quantile(0.75) * math.exp(-0.5 * normal_quantile(0.75) ** 2)) ** 2 / (2.0 * math.pi)
FALSE_REJECT = 0.01
STEP_TOL = 0.01
ROBUST_TOL = 1.0
LAMBDA0 = 1e-3
EPS = np.finfo(float).eps
TINY = math.sqrt(EPS)


def robust_sigma(r):
    """px: 1.4826 x the median absolute deviation of residuals r, both axes
    pooled (they share one noise)."""
    a = np.asarray(r, float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(MAD_SIGMA * np.median(np.abs(a - np.median(a))))


def reject_limit(n):
    """Sigmas: the 2-D residual a genuine observation exceeds in fewer than
    FALSE_REJECT of data sets of n observations, the scale known."""
    return math.sqrt(2.0 * math.log(max(n, 1) / FALSE_REJECT))


def _scale(r, sigma_px):
    """The scale residuals are judged by.  A stated noise is a floor: while
    the model is still far from the data (from the drawing), the residuals'
    own robust scale is larger, and judging them by the noise instead would
    weight every observation as an outlier and turn the fit into an L1 fit
    that converges slowly; once the fit reaches the noise, the noise rules."""
    s = robust_sigma(r)
    if not np.isfinite(s):
        s = 0.0
    if sigma_px:
        s = max(float(sigma_px), s)
    return max(s, TINY)


def _weights(r, s, mask, huber):
    """(n, 2) IRLS weights: Huber's psi(r)/r per residual, zero off the mask."""
    w = np.repeat(mask[:, None], 2, axis=1).astype(float)
    if huber:
        k = HUBER_K * s
        a = np.abs(np.where(np.isfinite(r), r, 0.0))
        w *= np.where(a <= k, 1.0, k / np.maximum(a, k))
    return w


def _cost(r, s, mask, huber):
    rr = r[mask]
    if not np.all(np.isfinite(rr)):
        return np.inf
    if not huber:
        return 0.5 * float(np.sum(rr * rr))
    k = HUBER_K * s
    a = np.abs(rr)
    return float(np.sum(np.where(a <= k, 0.5 * a * a, k * a - 0.5 * k * k)))


# ======================================================= LINEAR ALGEBRA
def _solve(H, g, lam, free):
    """d minimising the damped model: (H + lam diag(H)) d = -g over the
    free columns (Marquardt's scaling, done as a Jacobi scaling so the
    system is well conditioned whatever the units), zero elsewhere."""
    d = np.zeros(len(g))
    idx = np.flatnonzero(free)
    if idx.size == 0:
        return d
    D = np.sqrt(np.diag(H)[idx])
    A = H[np.ix_(idx, idx)] / np.outer(D, D)
    A[np.diag_indices_from(A)] += lam
    b = -g[idx] / D
    try:
        np.linalg.cholesky(A)
        y = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        y = np.linalg.lstsq(A, b, rcond=None)[0]
    d[idx] = y / D
    return d


def inverse(H, free, info=False):
    """The inverse of the normal matrix over the free columns, by the
    eigenvectors of its Jacobi scaling; NaN off the free columns and on
    every row and column of a parameter the data do not fix (the module
    docstring's "null space").  With info, (inverse, the number of
    directions the data do not fix, the scaled matrix's condition number
    over the rest)."""
    n = len(H)
    out = np.full((n, n), np.nan)
    idx = np.flatnonzero(free)
    k, cond = 0, 1.0
    if idx.size:
        D = np.sqrt(np.diag(H)[idx])
        A = H[np.ix_(idx, idx)] / np.outer(D, D)
        lam, V = np.linalg.eigh(0.5 * (A + A.T))
        null = lam <= len(lam) * EPS * lam.max()
        k = int(null.sum())
        cond = float(lam.max() / lam[~null].min())
        Ai = (V[:, ~null] / lam[~null]) @ V[:, ~null].T
        loose = np.linalg.norm(V[:, null], axis=1) > TINY
        Ai[loose, :] = np.nan
        Ai[:, loose] = np.nan
        out[np.ix_(idx, idx)] = Ai / np.outer(D, D)
    return (out, k, cond) if info else out


def skew_n(V):
    """(n, 3, 3): the cross-product matrix of each row of V."""
    V = np.asarray(V, float)
    K = np.zeros(V.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -V[..., 2], V[..., 1]
    K[..., 1, 0], K[..., 1, 2] = V[..., 2], -V[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -V[..., 1], V[..., 0]
    return K


# ================================================ PROJECTION DERIVATIVES
def projection_jacobian(cam, X, radius=0.0):
    """The derivatives of Camera.project(X, radius) at world points X (n, 3).

    Returns (Jcam, JX): Jcam (n, 2, 15) with respect to the camera's
    parameters in the order Camera.params() lists them, except that the
    rotation is the increment R <- exp([dphi]x) R (dphi in the camera's own
    axes); JX (n, 2, 3) with respect to the world point.  `radius` may be
    one value or one per point.  The chain is the model's own: camera
    frame, the ball's ellipse centre C_xy C_z / (C_z^2 - r^2), Brown-
    Conrady, pixels."""
    X = np.atleast_2d(np.asarray(X, float))
    n = len(X)
    r = np.broadcast_to(np.asarray(radius, float), (n,))
    L = cam.lens
    RX = X @ cam.R.T
    C = RX + cam.t
    z = C[:, 2]
    q = z * z - r * r
    s = z / q
    ds = -(z * z + r * r) / (q * q)
    x, y = C[:, 0] * s, C[:, 1] * s
    A = np.zeros((n, 2, 3))                      # d(x, y) / dC
    A[:, 0, 0], A[:, 0, 2] = s, C[:, 0] * ds
    A[:, 1, 1], A[:, 1, 2] = s, C[:, 1] * ds
    a, b, c, d = L._ddistort(x, y)               # d(xd, yd) / d(x, y)
    Dd = np.empty((n, 2, 2))
    Dd[:, 0, 0], Dd[:, 0, 1], Dd[:, 1, 0], Dd[:, 1, 1] = a, b, c, d
    dC = np.einsum("nij,njk->nik", Dd, A)
    dC[:, 0] *= L.fx
    dC[:, 1] *= L.fy
    xd, yd = L.distort(x, y)
    r2 = x * x + y * y
    J = np.zeros((n, 2, 15))
    J[:, :, 0:3] = np.einsum("nij,njk->nik", dC, -skew_n(RX))
    J[:, :, 3:6] = dC
    J[:, 0, 6], J[:, 1, 7] = xd, yd
    J[:, 0, 8], J[:, 1, 9] = 1.0, 1.0
    for j, pw in ((10, 1), (11, 2), (12, 3)):    # k1, k2, k3
        J[:, 0, j] = L.fx * x * r2 ** pw
        J[:, 1, j] = L.fy * y * r2 ** pw
    J[:, 0, 13], J[:, 1, 13] = L.fx * 2.0 * x * y, L.fy * (r2 + 2.0 * y * y)      # p1
    J[:, 0, 14], J[:, 1, 14] = L.fx * (r2 + 2.0 * x * x), L.fy * 2.0 * x * y      # p2
    return J, dC @ cam.R


def body_jacobian(cam, R, t, P, radius=0.0):
    """The derivatives of the pixels of body points P (n, 3) on bodies at
    poses (R (n, 3, 3) or (3, 3), t (n, 3) or (3,)), seen by one camera:
    (Jcam (n, 2, 15), Jpose (n, 2, 6), Jpoint (n, 2, 3)) -- by the camera's
    parameters (as projection_jacobian), by the pose increment [dphi (world
    axes, R <- exp([dphi]x) R), dt], and by the body-frame point.  Shared
    by bundle adjustment and the body's pose, so one chain rule serves
    both."""
    P = np.atleast_2d(np.asarray(P, float))
    R = np.broadcast_to(np.asarray(R, float), (len(P), 3, 3))
    RP = np.einsum("nij,nj->ni", R, P)
    Jc, dX = projection_jacobian(cam, RP + t, radius)
    Js = np.concatenate([np.einsum("nij,njk->nik", dX, -skew_n(RP)), dX], axis=2)   # dX/dphi = -[R P]x
    return Jc, Js, np.einsum("nij,njk->nik", dX, R)


def lens_jacobian(lens_, x, y):
    """(n, 2, 9): the pixel's derivatives at normalised undistorted points
    with respect to [fx, fy, cx, cy, k1, k2, k3, p1, p2] -- how far an error
    in the lens moves the pixel of a fixed ray."""
    x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
    xd, yd = lens_.distort(x, y)
    r2 = x * x + y * y
    J = np.zeros((len(x), 2, 9))
    J[:, 0, 0], J[:, 1, 1], J[:, 0, 2], J[:, 1, 3] = xd, yd, 1.0, 1.0
    for j, pw in ((4, 1), (5, 2), (6, 3)):
        J[:, 0, j], J[:, 1, j] = lens_.fx * x * r2 ** pw, lens_.fy * y * r2 ** pw
    J[:, 0, 7], J[:, 1, 7] = lens_.fx * 2.0 * x * y, lens_.fy * (r2 + 2.0 * y * y)
    J[:, 0, 8], J[:, 1, 8] = lens_.fx * (r2 + 2.0 * x * x), lens_.fy * 2.0 * x * y
    return J


def _normal(J, w, r, w2=None):
    """(J^T W J, J^T W r, J^T W2 J) from J (n, 2, p), for a small dense p."""
    Jw = J * w[:, :, None]
    H2 = None if w2 is None else np.einsum("nai,na,naj->ij", J, w2, J)
    return np.einsum("nai,naj->ij", Jw, J), np.einsum("nai,na->i", Jw, r), H2


# ============================================================ THE SOLVER
@dataclass
class LMFit:
    x: object
    r: np.ndarray              # (n, 2) px, predicted - observed, at x
    inliers: np.ndarray        # (n,) bool
    scale: float               # px: robust scale (MAD) of the inliers' residuals
    H: np.ndarray              # J^T J over the inliers at x, unweighted
    free: np.ndarray           # (p,) bool: the columns the data constrain
    iterations: int            # linearisations, all phases
    converged: bool
    lam: float = LAMBDA0       # the damping it ended with


def _lm(x, residual, normal, step, mask, sigma_px, huber, budget, say, tag, lam=LAMBDA0):
    """Levenberg-Marquardt on the Huber (or plain) cost of the masked
    residuals, from x and damping lam.  Returns (x, linearisations,
    converged, damping, the number of columns the masked data inform)."""
    nu = 2.0
    p = None
    r = residual(x)
    mask = mask & np.all(np.isfinite(r), axis=1)
    for it in range(budget):
        s = _scale(r[mask], sigma_px)
        w = _weights(r, s, mask, huber)
        cost = _cost(r, s, mask, huber)
        if huber:
            # the step is measured against what the residuals inside Huber's
            # knee know: a direction only outliers fix is settled by the
            # rejection that follows, not by more reweighting (ROBUST_TOL)
            w_in = np.repeat(mask[:, None], 2, axis=1) & (np.abs(np.where(np.isfinite(r), r, np.inf)) <= HUBER_K * s)
            H, g, Hq = normal(x, w, np.where(w > 0, r, 0.0), w_in.astype(float))
        else:
            H, g, Hq = normal(x, w, np.where(w > 0, r, 0.0))
            Hq = H
        free = np.diag(H) > 0
        p = int(free.sum())
        dgn = _solve(H, g, 0.0, free)
        dec = math.sqrt(max(float(dgn @ Hq @ dgn), 0.0)) / s
        say("      %s %2d: cost %.6g  scale %.4g px  GN step %.3g sd  lambda %.2g" % (tag, it, cost, s, dec, lam))
        if dec < (ROBUST_TOL if huber else STEP_TOL):
            return x, it + 1, True, lam, p
        while True:
            d = _solve(H, g, lam, free)
            xt = step(x, d)
            rt = residual(xt)
            ct = _cost(rt, s, mask, huber)
            pred = -float(d @ g) - 0.5 * float(d @ H @ d)
            if pred > 0 and ct < cost:
                rho = (cost - ct) / pred
                x, r = xt, rt
                lam *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
                nu = 2.0
                break
            lam *= nu
            nu *= 2.0
            length = math.sqrt(max(float(d @ Hq @ d), 0.0)) / s
            if length < (ROBUST_TOL if huber else STEP_TOL):
                # no step the data could see lowers the cost: at its
                # minimum to the precision the data supports
                say("      %s %2d: stalled at a step of %.3g sd" % (tag, it, length))
                return x, it + 1, True, lam, p
    return x, budget, False, lam, p


def _first_cut(r, act, sigma_px, p):
    """The first inlier set, before there is an inlier fit to judge by:
    each 2-D residual against the robust scale of them all (the module
    docstring's "rejection"), p the columns the fit has."""
    fin = act & np.all(np.isfinite(r), axis=1)
    e2 = np.sum(np.where(fin[:, None], r, 0.0) ** 2, axis=1)
    floor = math.log(FALSE_REJECT / max(int(act.sum()), 1))
    if sigma_px:
        s = _scale(r[act], sigma_px)
        return fin & (-e2 / (2.0 * s * s) >= floor)            # |r| <= reject_limit(n) s
    N = 2 * int(fin.sum())
    nu = N - (p or 0)
    if nu <= 0:
        return fin                                             # an exact fit: nothing can judge anything
    s = max(robust_sigma(r[fin]) * math.sqrt(N / nu), TINY)
    nu_e = MAD_EFF * nu
    return fin & (-0.5 * nu_e * np.log1p(e2 / (nu_e * s * s)) >= floor)


def _retest(x, r, inl, act, normal, hat, sigma_px):
    """The inlier set the fit at x supports: every active observation
    tested against the fit of the other inliers (the module docstring's
    "rejection").  NaN residuals are rejected."""
    fin = np.all(np.isfinite(r), axis=1)
    rows = np.flatnonzero(act & fin)
    inl = inl & fin
    w = np.repeat(inl[:, None], 2, axis=1).astype(float)
    H = normal(x, w, np.where(w > 0, r, 0.0))[0]
    free = np.diag(H) > 0
    Hi, _, cond = inverse(H, free, info=True)
    G = np.zeros((len(rows), 2, 2)) if hat is None else hat(x, Hi, rows)
    inside = inl[rows]
    M = np.eye(2) + np.where(inside, -1.0, 1.0)[:, None, None] * G
    det = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
    tr = M[:, 0, 0] + M[:, 1, 1]
    lam_min = 0.5 * (tr - np.sqrt(np.maximum(tr * tr - 4.0 * det, 0.0)))
    e = r[rows]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (M[:, 1, 1] * e[:, 0] ** 2 - (M[:, 0, 1] + M[:, 1, 0]) * e[:, 0] * e[:, 1]
             + M[:, 0, 0] * e[:, 1] ** 2) / det                                  # r^T M^-1 r
    # 1 - h is known only to the inverse's round-off, n eps cond: below it
    # the observation may be the only one fixing a direction, untestable
    judged = np.isfinite(d) & (d >= 0.0) & (lam_min > max(TINY, int(free.sum()) * EPS * cond))
    keep_unjudged = inside.copy()
    if sigma_px:
        s = _scale(r[act], sigma_px)
        logtail = -d / (2.0 * s * s)
    else:
        rss = float(np.sum(r[inl] ** 2))
        nu = 2 * int(inl.sum()) - int(free.sum())
        nu_o = np.where(inside, nu - 2, nu)
        rss_o = np.where(inside, rss - d, rss)
        no_dof = nu_o <= 0
        judged &= ~no_dof
        keep_unjudged |= no_dof                      # nothing can judge it: kept, and so is a rejected one
        exact = rss_o <= np.maximum(nu_o, 1) * TINY ** 2   # the others fit to round-off: TINY is the scale
        with np.errstate(divide="ignore", invalid="ignore"):
            logtail = np.where(exact, -d / (2.0 * TINY ** 2), -0.5 * nu_o * np.log1p(d / rss_o))
    keep = np.where(judged, logtail >= math.log(FALSE_REJECT / max(int(act.sum()), 1)), keep_unjudged)
    out = inside & ~keep
    if out.sum() > 1:
        # one at a time out of the fit, the worst first: each test assumes
        # the other inliers right, and a fit's residuals are correlated
        keep |= out & (np.arange(len(rows)) != np.argmin(np.where(out, logtail, np.inf)))
    new = np.zeros(len(act), bool)
    new[rows] = keep
    return new


def robust_lm(x, residual, normal, step, n, sigma_px=None, active=None, robust=True, max_iter=50, log=None,
              lam0=LAMBDA0, hat=None):
    """The rig's least-squares solver, shared by bundle adjustment and the
    body's pose.

      residual(x)       -> (n, 2) px, predicted - observed; NaN where the
                           state puts a ball behind its camera
      normal(x, w, r, w2=None)
                        -> (H, g, H2) = (J^T W J, J^T W r, J^T W2 J), W =
                           diag(w), w (n, 2) per residual; H2 None without w2
      step(x, d)        -> the state moved by d (one entry per column)
      hat(x, Hinv, rows) -> (len(rows), 2, 2): J_i Hinv J_i^T for those
                           observations, Hinv as inverse() gives it; None
                           takes every leverage as zero

    `active` marks the observations that may be used at all.  With
    `robust`, Huber IRLS first, then the first cut, plain least squares on
    the inliers, and the re-test against them, until the inlier set is
    stable; without it, plain least squares on `active`, its damping
    starting at lam0 (a solve continuing another passes that one's).
    Columns with no information (a zero diagonal) are held where they
    are."""
    say = log or (lambda m: None)
    act = np.ones(n, bool) if active is None else np.asarray(active, bool).copy()
    used, conv = 0, True
    if robust:
        # each phase starts from the last one's damping: it starts where the
        # last one stopped, near a minimum
        x, k, conv, lam, p = _lm(x, residual, normal, step, act, sigma_px, True, max_iter, say, "huber")
        used += k
        inl = None
        while True:
            r = residual(x)
            first = inl is None
            new = _first_cut(r, act, sigma_px, p) if first else _retest(x, r, inl, act, normal, hat, sigma_px)
            if not first and np.array_equal(new, inl):
                break
            if used >= max_iter:
                conv = False
                inl = new if first else inl
                break
            inl = new
            say("      %s: %d of %d rejected" % ("first cut" if first else "re-test", int(act.sum() - inl.sum()),
                                                 int(act.sum())))
            x, k, conv, lam, _ = _lm(x, residual, normal, step, inl, sigma_px, False, max_iter - used, say, "ls", lam)
            used += k
    else:
        inl = act
        x, used, conv, lam, _ = _lm(x, residual, normal, step, inl, sigma_px, False, max_iter, say, "ls", lam0)
    r = residual(x)
    inl = inl & np.all(np.isfinite(r), axis=1)
    w = np.repeat(inl[:, None], 2, axis=1).astype(float)
    H = normal(x, w, np.where(w > 0, r, 0.0))[0]
    return LMFit(x, r, inl, robust_sigma(r[inl]), H, np.diag(H) > 0, used, conv, lam)


# ============================================================== GEOMETRY
def kabsch(P, X, w=None):
    """(R, t): the rigid map minimising sum w |R P_i + t - X_i|^2 (Kabsch
    1976; Umeyama 1991, whose sign on the last singular vector keeps R a
    rotation when the points are nearly planar or noisy)."""
    P, X = np.asarray(P, float), np.asarray(X, float)
    w = np.ones(len(P)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    p0, x0 = w @ P, w @ X
    Hm = (P - p0).T @ ((X - x0) * w[:, None])
    U, _, Vt = np.linalg.svd(Hm)
    S = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ S @ U.T
    return R, x0 - R @ p0


def spans_plane(Q):
    """True if the points Q (n, 3) are at least three and not on one line,
    by numpy's numerical rank (tolerance: the largest singular value times
    the size times machine epsilon).  Nearly collinear points pass and say
    so in the covariance of what they fix: large, or NaN where the normal
    matrix cannot tell them from a line (inverse())."""
    Q = np.asarray(Q, float).reshape(-1, 3)
    return len(Q) >= 3 and np.linalg.matrix_rank(Q - Q.mean(axis=0)) >= 2


def _project(cams, ci, X, rad):
    """(n, 2) pixels of world points X seen by cameras ci with radii rad,
    by Camera.project grouped by camera and radius; NaN behind a camera."""
    out = np.full((len(ci), 2), np.nan)
    for c in np.unique(ci):
        for rho in np.unique(rad[ci == c]):
            idx = np.flatnonzero((ci == c) & (rad == rho))
            u, v, z = cams[c].project(X[idx], float(rho))
            ok = z > rho
            out[idx[ok], 0], out[idx[ok], 1] = u[ok], v[ok]
    return out


def _rays(cams, ci, uv):
    """(n, 2): the normalised, undistorted ray (x/z, y/z) of each pixel."""
    xy = np.empty((len(ci), 2))
    for c in np.unique(ci):
        idx = np.flatnonzero(ci == c)
        xy[idx, 0], xy[idx, 1] = cams[c].lens.ray(uv[idx, 0], uv[idx, 1])
    return xy


def _dlt(cams, ci, xy):
    """The linear triangulation of one point from its rays (Hartley &
    Zisserman 2004, 12.2, inhomogeneous): x (R_3 X + t_3) = R_1 X + t_1 and
    the same in y, for every camera, in least squares."""
    Rs = np.stack([cams[c].R for c in ci])
    ts = np.stack([cams[c].t for c in ci])
    A = np.concatenate([xy[:, :1] * Rs[:, 2] - Rs[:, 0], xy[:, 1:] * Rs[:, 2] - Rs[:, 1]])
    b = np.concatenate([ts[:, 0] - xy[:, 0] * ts[:, 2], ts[:, 1] - xy[:, 1] * ts[:, 2]])
    return np.linalg.lstsq(A, b, rcond=None)[0]


def triangulate(cams, obs_uv, radius=0.0):
    """(X, rms_px): the world point (mm) seen at pixels obs_uv, a list of
    (camera index, u, v), by the linear DLT on undistorted rays and then
    Gauss-Newton on the reprojection error of a ball of `radius`.  Needs two
    cameras or more with finite centroids; a centroid that is not a finite
    number says nothing and is not used."""
    ob = np.asarray(obs_uv, float).reshape(-1, 3)
    ob = ob[np.all(np.isfinite(ob), axis=1)]
    ci, uv = ob[:, 0].astype(int), ob[:, 1:]
    if len(np.unique(ci)) < 2:
        raise ValueError("a point needs two cameras with finite centroids to triangulate")
    X0 = _dlt(cams, ci, _rays(cams, ci, uv))
    rad = np.full(len(ci), float(radius))

    def residual(X):
        return _project(cams, ci, np.repeat(X[None], len(ci), axis=0), rad) - uv

    def normal(X, w, r, w2=None):
        J = np.zeros((len(ci), 2, 3))
        for c in np.unique(ci):
            idx = np.flatnonzero(ci == c)
            J[idx] = projection_jacobian(cams[c], np.repeat(X[None], len(idx), axis=0), rad[idx])[1]
        return _normal(J, w, r, w2)

    fit = robust_lm(X0, residual, normal, lambda X, d: X + d, len(ci), robust=False)
    return fit.x, float(np.sqrt(np.mean(fit.r[fit.inliers] ** 2)))


def as_pose(init):
    """(R, t) from a (3, 4) [R | t] or an (R, t) pair."""
    if isinstance(init, (tuple, list)):
        return np.array(init[0], float), np.array(init[1], float).reshape(3)
    M = np.asarray(init, float)
    return M[:3, :3].copy(), M[:3, 3].copy()


@dataclass
class PoseFit:
    R: np.ndarray              # body -> world
    t: np.ndarray              # mm
    rms_px: float              # rms reprojection residual of the inliers, per axis
    n_obs: int                 # inlier observations
    n_points: int              # distinct balls among them
    inliers: np.ndarray        # (n,) bool over the observations passed in
    cov: np.ndarray            # 6x6, 1 sigma^2: [dphi (rad, world axes), t (mm)]
    sigma_px: float            # the noise the covariance is scaled by
    scale_px: float            # robust scale of the inliers' residuals
    iterations: int
    converged: bool


def body_pose(cams, P, radius, obs, init=None, sigma_px=None, max_iter=50, log=None):
    """The body's pose from one frame's observations, or None.

    cams    calibrated lens.Camera list; obs: (camera index, point id, u, v)
    P       (K, 3) body-frame ball positions, mm; radius: (K,) or one, mm
    init    the encoders' pose, (3, 4) or (R, t); without it, the linear
            triangulation of every ball seen by two cameras or more, and
            Kabsch
    sigma_px  the centroid noise; None estimates it from the residuals

    None when fewer than three balls not on one line constrain the pose:
    seen by two cameras (to triangulate) without `init`, by one with it.
    An observation whose u or v is not a finite number is not used and is
    not an inlier.  The covariance is sigma^2 (J^T J)^-1 over the inliers,
    sigma the stated noise or, without one, the residuals' own (sum r^2 /
    (2n - 6)); NaN in any direction the inliers do not fix (balls on a line
    to round-off)."""
    P = np.asarray(P, float)
    rad_k = np.broadcast_to(np.asarray(radius, float), (len(P),))
    ob = np.asarray(obs, float).reshape(-1, 4)
    ci, pi, uv = ob[:, 0].astype(int), ob[:, 1].astype(int), ob[:, 2:4]
    n = len(ob)
    rad = rad_k[pi]
    fin = np.all(np.isfinite(uv), axis=1)
    if init is None:
        # the linear triangulation alone: refining each ball by Gauss-Newton
        # first would be undone by the pose's own fit, which sees them all
        good = np.flatnonzero(fin)
        xy = _rays(cams, ci[good], uv[good])
        seen, Xs = [], []
        for k in np.unique(pi[good]):
            sel = np.flatnonzero(pi[good] == k)
            if len(np.unique(ci[good[sel]])) < 2:
                continue
            seen.append(k)
            Xs.append(_dlt(cams, ci[good[sel]], xy[sel]))
        if not spans_plane(P[seen]):
            return None
        R0, t0 = kabsch(P[seen], np.array(Xs))
    else:
        if not spans_plane(P[np.unique(pi[fin])]):
            return None
        R0, t0 = as_pose(init)

    def residual(x):
        R, t = x
        return _project(cams, ci, P[pi] @ R.T + t, rad) - uv

    def normal(x, w, r, w2=None):
        R, t = x
        use = np.any(w > 0, axis=1)            # a rejected ball may be behind its camera
        J = np.zeros((n, 2, 6))
        for c in np.unique(ci[use]):
            idx = np.flatnonzero(use & (ci == c))
            J[idx] = body_jacobian(cams[c], R, t, P[pi[idx]], rad[idx])[1]
        return _normal(J, w, r, w2)

    def step(x, d):
        return rodrigues(d[:3]) @ x[0], x[1] + d[3:]

    def hat(x, Hi, rows):
        R, t = x
        J = np.zeros((len(rows), 2, 6))
        for c in np.unique(ci[rows]):
            k = np.flatnonzero(ci[rows] == c)
            J[k] = body_jacobian(cams[c], R, t, P[pi[rows[k]]], rad[rows[k]])[1]
        return np.einsum("nai,ij,nbj->nab", J, Hi, J)

    fit = robust_lm((R0, t0), residual, normal, step, n, sigma_px=sigma_px, active=fin, max_iter=max_iter, log=log,
                    hat=hat)
    inl = fit.inliers
    if not spans_plane(P[np.unique(pi[inl])]):
        return None
    m = int(inl.sum())
    rr = fit.r[inl]
    sig = float(sigma_px) if sigma_px else (math.sqrt(float(np.sum(rr * rr)) / (2 * m - 6)) if 2 * m > 6 else np.nan)
    cov = sig ** 2 * inverse(fit.H, fit.free)
    R, t = fit.x
    return PoseFit(R, t, float(np.sqrt(np.mean(rr ** 2))), m, len(np.unique(pi[inl])), inl, cov, sig, fit.scale,
                   fit.iterations, fit.converged)


def pose_error(R_est, t_est, R_true, t_true, at=None):
    """(deg, mm): the angle of the rotation between two poses, and how far
    apart they put the body point `at` (default the body origin)."""
    p = np.zeros(3) if at is None else np.asarray(at, float)
    a = np.asarray(R_est) @ p + np.asarray(t_est)
    b = np.asarray(R_true) @ p + np.asarray(t_true)
    return angle_between(R_est, R_true), float(np.linalg.norm(a - b))
