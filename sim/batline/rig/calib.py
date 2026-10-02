"""Self-calibration of the rig's cameras by bundle adjustment, and the
shift-start check that the calibration still holds.

The rig's 8 cameras are built, not drawn: each sits millimetres and a
degree from its drawing, its focal length is a percent or two off, its
principal point some pixels, and its lens bends rays by tens of pixels at
the edge of what it sees.  None of that is measured by hand.  The gimbal
turns a certified calibration bar (its balls' positions known to 0.01 mm)
and the ring's own markers (known only as drawn, +-0.3 mm) through many
orientations; every camera reports where it saw every ball; and one
least-squares problem solves everything at once -- every camera's pose,
focal lengths, principal point and Brown-Conrady distortion, every
sample's body pose, and the ring markers' body-frame positions -- for the
values that best explain every centroid.  That is bundle adjustment
(Triggs, McLauchlan, Hartley & Fitzgibbon 2000): the maximum-likelihood
estimate under the centroid noise, and the only estimate that uses every
observation for every unknown it bears on.

THE GAUGE (Triggs et al. 2000, 9).  The reprojections do not change if the
whole world moves rigidly, or if the body frame moves and every body point
moves with it, or if everything is scaled.  Such directions carry no
information, and one left free makes the normal equations singular.
Exactly enough is held:

  - the certified points' body-frame positions.  At least three of them,
    not on one line, fix the body frame's origin, orientation and scale:
    the bar is the only ruler the rig has, and the body frame is the bar's.
    How well it fixes the turn about its own length is the bar's geometry:
    only its balls' offsets across that length do, so a bar whose balls
    sit nearly on a line leaves every pose that turn's uncertainty, and
    its certificate's error over those offsets turns the body frame by an
    amount no observation can see;
  - one sample's body pose, at the gimbal's own reading of it: sample 0's,
    or, if sample 0 does not see three balls off a line, the first sample
    that does (and the Solution says so).  It fixes the world frame's 6
    dof.  The world is therefore "where the gimbal said that sample was",
    wrong by the encoders' error; every camera and every pose is right
    relative to it, and one rigid transform takes it to any other world.

Nothing else is held: 15 parameters per camera, 6 per other sample, 3 per
free ball.  `fix_dist` and `fix_intrinsics` hold more on request, never
less.

THE METHOD.  Levenberg-Marquardt (Marquardt 1963, his diagonal scaling) on
the full parameter vector, robust (pose.robust_lm, shared with the body's
pose, whose docstring gives the rules): Huber's loss by iteratively
reweighted least squares from the drawing, then rejection of each
observation the noise cannot explain, then plain least squares on the
inliers.  Rotations are updated on the left, R <- exp([dphi]x) R: for a
body pose (body -> world) dphi is in world axes; for a camera (world ->
camera) it is in the camera's own axes.  The Jacobian is analytic
(pose.projection_jacobian, through the ball's ellipse-centre correction),
assembled per observation into its 24 columns -- its camera's 15, its
sample's 6, its ball's 3 -- and accumulated into dense normal equations.
About 500 unknowns and 10^4 observations make those cheap; the Schur
complement that large problems need (Triggs et al. 2000, 6.1) buys nothing
at this size.  Every predicted pixel is lens.Camera.project's, with the
ball's radius.

WHAT CANNOT BE SOLVED IS DROPPED, AND SAID.  Before solving, and again
after rejection: a sample (other than the one held) whose observations do
not span three balls off a line cannot fix its pose; a free ball seen along
a single ray (one camera, one sample) cannot fix its depth; a camera with
fewer residuals than free parameters cannot fix itself; and cameras and
samples that share no observation with the rest of the rig (an island of
the camera-sample graph) can move together without changing a residual,
so only the part that holds the gauge is kept.  Their observations are not
used, they keep their starting values, and `Solution.warnings` says which
and why.  A column with no information left is held by the solver rather
than inverted, and any direction the data still do not fix (a zero
eigenvalue of the normal matrix: pose.inverse) gives every parameter in it
NaN covariance, is left out of n_params, and is named in the warnings.
What the data fix only POORLY is not dropped: a camera that saw a few balls
nearly on one line is solved, and its covariance (cov["cam"],
cov["centre"]) says how badly.  Whether that is good enough depends on how
well the drawing knew it, which the caller knows and this does not.

THE COVARIANCE.  After the final solve, sigma^2 (J^T J)^-1 over the
inliers, sigma the stated noise or, without one, the residuals' own
(sum r^2 / dof).  It bounds what the calibration can know: the checks
compare errors against it rather than against a tolerance someone chose.

THE SHIFT-START CHECK (drift).  With the calibration held, each sample's
body pose is re-estimated from new observations (pose.body_pose, robust),
and two questions are asked at alpha / 2 each.  Do the pose fits' inliers
fit to the noise -- the sum of their squared normalised residuals is
chi-square with 2N - 6S dof if the rig is as calibrated?  And does any
camera hold more of the rejected observations than chance gives it --
exactly hypergeometric whatever the detector's rate of gross centroids,
which strike every camera alike, whereas a moved camera has its own
observations rejected?  The first sees a small move, the second a large
one, and neither is moved by a gross centroid or two, which every real
data set has.  Given the Solution the cameras came from, the chi-square
also allows for the calibration's own uncertainty -- the calibrated values
are taken as observations with the covariance above, a prior whose
equations add as many unknowns as they add residuals, so the dof is
unchanged.  Without that, a sharp check on many observations sees the
calibration's own noise as drift and fails more often than alpha.  What no
covariance here carries is the certificate's own error: the certified
balls are held as exact, so the check reads a bar that differs from its
certificate as a small excess of chi-square.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from . import pose
from .lens import DIST, N_PARAMS, Camera, rodrigues

LENS = ("fx", "fy", "cx", "cy") + DIST      # Lens.params() order: columns 6.. of a camera
N_POSE, N_POINT = 6, 3


# ================================================================== DATA
@dataclass
class Obs:                     # one detection
    cam: int                   # camera index
    sample: int                # gimbal sample (frame) index
    point: int                 # ball id (index into the points table)
    u: float
    v: float


@dataclass
class Points:                  # the balls on the body
    P: np.ndarray              # (K, 3) body-frame positions, mm (initial guess for free ones)
    radius: np.ndarray         # (K,) ball radius, mm
    known: np.ndarray          # (K,) bool: certified (held fixed) or free (solved)


@dataclass
class Solution:
    cams: list                 # calibrated lens.Camera, same order and names
    poses: np.ndarray          # (S, 3, 4): body -> world [R | t] per sample
    P: np.ndarray              # (K, 3) body-frame positions (known ones unchanged)
    rms_px: float              # rms reprojection residual of the inliers, per axis
    sigma_px: float            # robust scale of the inliers' residuals (1.4826 x MAD per axis)
    inliers: np.ndarray        # (N,) bool over the observations passed in
    iterations: int            # linearisations, all phases
    per_cam_rms: np.ndarray    # (C,) px; NaN for a camera with no inliers
    cov: dict = field(default_factory=dict)    # 1-sigma; keys and units in _covariance
    warnings: list = field(default_factory=list)   # what was dropped or held, and why
    n_params: int = 0          # unknowns the data constrained
    gauge: int = 0             # the sample whose pose was held
    converged: bool = True
    used: np.ndarray = None    # (N,) bool: observations the data could use (pruning, not rejection);
    #                            used & ~inliers are the ones the outlier test rejected
    stem: float = np.nan       # the stems' pull, of the area model's, when it was solved (bundle_adjust's pull)
    stem_sd: float = np.nan    # its 1 sigma, marginal over everything else


def _arrays(obs):
    cam = np.array([o.cam for o in obs], int)
    smp = np.array([o.sample for o in obs], int)
    pt = np.array([o.point for o in obs], int)
    uv = np.array([[o.u, o.v] for o in obs], float).reshape(-1, 2)
    return cam, smp, pt, uv


# ============================================================ THE STATE
@dataclass
class _State:
    Rc: np.ndarray             # (C, 3, 3) world -> camera
    tc: np.ndarray             # (C, 3)
    lp: np.ndarray             # (C, 9) Lens.params()
    Rb: np.ndarray             # (S, 3, 3) body -> world
    tb: np.ndarray             # (S, 3)
    P: np.ndarray              # (K, 3) body frame
    k: float = 1.0             # the stems' scale

    def cameras(self, like):
        return [Camera(c.name, c.lens.with_params(self.lp[i]), self.Rc[i].copy(), self.tc[i].copy(), dict(c.meta))
                for i, c in enumerate(like)]


def _expand(d, cols):
    out = np.zeros(cols.shape)
    m = cols >= 0
    out[m] = d[cols[m]]
    return out


def _blocks(cams, Rb, tb, P, cam, smp, pt, rad):
    """Each observation's pixel derivatives: (n, 2, 15) by its camera's
    parameters, (n, 2, 6) by its sample's pose increment [dphi (world
    axes), dt], (n, 2, 3) by its ball's body-frame position
    (pose.body_jacobian, camera by camera)."""
    n = len(cam)
    Jc, Js, Jp = np.zeros((n, 2, N_PARAMS)), np.zeros((n, 2, N_POSE)), np.zeros((n, 2, N_POINT))
    for c in np.unique(cam):
        idx = np.flatnonzero(cam == c)
        Jc[idx], Js[idx], Jp[idx] = pose.body_jacobian(cams[c], Rb[smp[idx]], tb[smp[idx]], P[pt[idx]], rad[idx])
    return Jc, Js, Jp


def _predict(cams, Rb, tb, P, cam, smp, pt, rad):
    """(n, 2) px: Camera.project of every observation's ball, with its radius."""
    X = np.einsum("nij,nj->ni", Rb[smp], P[pt]) + tb[smp]
    return pose._project(cams, cam, X, rad)


# ============================================================== PRUNING
def _prune(cam, smp, pt, P, known, n_cam_free, use, held=None):
    """(use, gauge, warnings): drop what cannot be constrained, until
    nothing more drops.  The gauge is the first sample that sees three balls
    off a line, or `held` once one is held: a held pose needs no fixing,
    and what it sees still ties the world to the cameras."""
    warn = []
    while True:
        changed = False
        gauge = held
        for s in np.unique(smp[use]):
            sel = use & (smp == s)
            if s == held:
                continue
            if pose.spans_plane(P[np.unique(pt[sel])]):
                if gauge is None:
                    gauge = int(s)
                continue
            nb = len(np.unique(pt[sel]))
            warn.append("sample %d: %d observations of %d ball%s cannot fix its pose; not used"
                        % (s, int(sel.sum()), nb, "" if nb == 1 else "s"))
            use = use & ~sel
            changed = True
        for k in np.unique(pt[use]):
            if known[k]:
                continue
            sel = use & (pt == k)
            rays = np.unique(cam[sel] * (smp.max() + 1) + smp[sel])
            if len(rays) < 2:
                warn.append("ball %d: seen along a single ray (camera %d, sample %d); its position cannot be "
                            "solved, %d observations not used" % (k, cam[sel][0], smp[sel][0], int(sel.sum())))
                use = use & ~sel
                changed = True
        for c in np.unique(cam[use]):
            sel = use & (cam == c)
            if 2 * int(sel.sum()) < n_cam_free:
                warn.append("camera %d: %d observations cannot fix its %d parameters; not used"
                            % (c, int(sel.sum()), n_cam_free))
                use = use & ~sel
                changed = True
        part = _connected(cam, smp, use, held)
        if not np.array_equal(part, use):
            lost = use & ~part
            warn.append("camera%s %s and sample%s %s share no observation with the rest of the rig, so nothing "
                        "fixes where they are relative to it; %d observations not used"
                        % (_plural(cam[lost]), _ids(cam[lost]), _plural(smp[lost]), _ids(smp[lost]), int(lost.sum())))
            use = part
            changed = True
        if not changed:
            return use, gauge, warn


def _ids(a):
    return ", ".join(str(int(v)) for v in np.unique(a))


def _plural(a):
    return "" if len(np.unique(a)) == 1 else "s"


def _connected(cam, smp, use, held=None):
    """The used observations in one connected part of the camera-sample
    graph (an observation joins its camera and its sample): the part that
    holds `held`, or else the one with the most observations.  A part that
    shares no observation with the rest -- its own cameras and samples --
    can move rigidly in the world without changing a residual, a gauge
    freedom of its own that the one held pose does not fix (Triggs et al.
    2000, 9).  Free balls do not join parts: they are in the body frame,
    which moves with the samples."""
    if not np.any(use):
        return use
    C = int(cam.max()) + 1
    root = np.arange(C + int(smp.max()) + 1)

    def find(a):
        while root[a] != a:
            root[a] = root[root[a]]
            a = root[a]
        return a

    for c, s in np.unique(np.stack([cam[use], smp[use]], axis=1), axis=0):
        root[find(c)] = find(C + s)
    lab = np.array([find(c) for c in cam])
    labs, counts = np.unique(lab[use], return_counts=True)
    if held is not None and np.any(use & (smp == held)):
        keep = find(C + held)
    else:
        keep = labs[np.argmax(counts)]
    return use & (lab == keep)


# ======================================================= BUNDLE ADJUSTMENT
def bundle_adjust(cams0, poses0, points, obs, fix_dist=(), fix_intrinsics=False, sigma_px=None, max_iter=50,
                  log=None, pull=None, stem0=1.0) -> Solution:
    """Calibrate every camera, every sample's body pose and the free balls'
    body-frame positions from observations of the balls.

    cams0       the drawing's cameras (lens.Camera), the starting point
    poses0      (S, 3, 4) body -> world, the gimbal's reading of each sample
    points      Points: certified (known) balls fix the body frame and the
                scale and are held; free ones start at P and are solved
    obs         list of Obs
    fix_dist    names in lens.DIST held at their starting value
    fix_intrinsics  hold every lens parameter (fx, fy, cx, cy and the
                distortion): only poses are solved
    sigma_px    the centroid noise for the Huber knee and the rejection
                (a floor: see pose._scale); None estimates it each round
    max_iter    linearisations, all phases together
    pull        (N, 2) px or None: each observation's stem's pull at scale 1
                (vision.stem_bias_px).  With it each blob is the ball's image
                plus one scale times its pull, and that scale, from stem0,
                is solved with everything else -- so its sigma is the
                marginal one, net of what the cameras, poses and balls
                absorb of the pull (Solution.stem, stem_sd)
    """
    say = log or (lambda m: None)
    C, S, K = len(cams0), len(poses0), len(points.P)
    poses0 = np.asarray(poses0, float)
    P0 = np.asarray(points.P, float)
    known = np.asarray(points.known, bool)
    rad_k = np.broadcast_to(np.asarray(points.radius, float), (K,))
    cam, smp, pt, uv = _arrays(obs)
    N = len(cam)
    if N and (cam.min() < 0 or cam.max() >= C or smp.min() < 0 or smp.max() >= S or pt.min() < 0 or pt.max() >= K):
        raise ValueError("an observation names a camera, sample or ball that is not there")
    rad = rad_k[pt]

    held = np.zeros(N_PARAMS, bool)
    if fix_intrinsics:
        held[6:] = True
    for name in fix_dist:
        if name not in DIST:
            raise ValueError("unknown distortion term %r (lens.DIST: %s)" % (name, DIST))
        held[6 + LENS.index(name)] = True

    use, gauge, warnings = _prune(cam, smp, pt, P0, known, int((~held).sum()), np.ones(N, bool))
    if gauge is None:
        raise ValueError("no sample sees three balls off a line: nothing fixes the world frame")
    if gauge != 0:
        warnings.append("sample 0 cannot fix the world frame; sample %d's pose is held instead" % gauge)
    if not pose.spans_plane(P0[np.unique(pt[use & known[pt]])]):
        raise ValueError("fewer than three certified balls off a line are seen: the body frame and the scale are "
                         "not fixed")
    for c in range(C):
        if not np.any(use & (cam == c)):
            warnings.append("camera %d (%s): no usable observations; left at its drawing" % (c, cams0[c].name))

    # ------------------------------------------------------------ layout
    ncol = 0
    cam_col = -np.ones((C, N_PARAMS), int)
    for c in range(C):
        for j in np.flatnonzero(~held):
            cam_col[c, j] = ncol
            ncol += 1
    smp_col = -np.ones((S, N_POSE), int)
    for s in range(S):
        if s != gauge:
            smp_col[s] = np.arange(ncol, ncol + N_POSE)
            ncol += N_POSE
    pt_col = -np.ones((K, N_POINT), int)
    for k in range(K):
        if not known[k]:
            pt_col[k] = np.arange(ncol, ncol + N_POINT)
            ncol += N_POINT
    k_col = -1
    extra = []
    if pull is not None:
        pull = np.asarray(pull, float).reshape(N, 2)
        k_col = ncol
        ncol += 1
        extra = [np.full((N, 1), k_col)]
    cols = np.concatenate([cam_col[cam], smp_col[smp], pt_col[pt]] + extra, axis=1)
    cols[cols < 0] = ncol                                       # a sink row and column, dropped
    hidx = (cols[:, :, None] * (ncol + 1) + cols[:, None, :]).reshape(N, -1)

    # ---------------------------------------------------- the three calls
    def residual(x):
        r = _predict(x.cameras(cams0), x.Rb, x.tb, x.P, cam, smp, pt, rad) - uv
        return r if pull is None else r + x.k * pull

    def jac(x, rows):
        Jc, Js, Jp = _blocks(x.cameras(cams0), x.Rb, x.tb, x.P, cam[rows], smp[rows], pt[rows], rad[rows])
        return np.concatenate([Jc, Js, Jp] + ([] if pull is None else [pull[rows][:, :, None]]), axis=2)

    def normal(x, w, r, w2=None):
        sel = np.flatnonzero(np.any(w > 0, axis=1))
        J = jac(x, sel)

        def accumulate(ww):
            Hl = np.einsum("nai,na,naj->nij", J, ww, J)
            return np.bincount(hidx[sel].ravel(), Hl.ravel(), (ncol + 1) ** 2).reshape(ncol + 1, ncol + 1)[:ncol, :ncol]

        gl = np.einsum("nai,na->ni", J * w[sel][:, :, None], r[sel])
        g = np.bincount(cols[sel].ravel(), gl.ravel(), ncol + 1)[:ncol]
        return accumulate(w[sel]), g, None if w2 is None else accumulate(w2[sel])

    def step(x, d):
        dc, ds, dp = _expand(d, cam_col), _expand(d, smp_col), _expand(d, pt_col)
        return _State(np.stack([rodrigues(dc[c, :3]) @ x.Rc[c] for c in range(C)]), x.tc + dc[:, 3:6],
                      x.lp + dc[:, 6:], np.stack([rodrigues(ds[s, :3]) @ x.Rb[s] for s in range(S)]),
                      x.tb + ds[:, 3:], x.P + dp, x.k + (float(d[k_col]) if k_col >= 0 else 0.0))

    def hat(x, Hi, rows):
        # J_i Hinv J_i^T from each observation's 24 columns; the sink (held
        # parameters) is exact, so its row and column of Hinv are zero
        J = jac(x, rows)
        Hp = np.zeros((ncol + 1, ncol + 1))
        Hp[:ncol, :ncol] = Hi
        G = np.empty((len(rows), 2, 2))
        for c in np.unique(cam[rows]):               # a camera at a time: (n, 24, 24) blocks stay small
            k = np.flatnonzero(cam[rows] == c)
            cc = cols[rows[k]]
            G[k] = np.einsum("nai,nij,nbj->nab", J[k], Hp[cc[:, :, None], cc[:, None, :]], J[k], optimize=True)
        return G

    x0 = _State(np.stack([c.R for c in cams0]).astype(float), np.stack([c.t for c in cams0]).astype(float),
                np.stack([c.lens.params() for c in cams0]), poses0[:, :, :3].copy(), poses0[:, :, 3].copy(),
                P0.copy(), float(stem0))
    say("    bundle adjustment: %d cameras, %d samples, %d balls (%d free), %d observations (%d usable), %d unknowns"
        % (C, S, K, int((~known).sum()), N, int(use.sum()), ncol))
    fit = pose.robust_lm(x0, residual, normal, step, N, sigma_px=sigma_px, active=use, max_iter=max_iter, log=say,
                         hat=hat)

    # rejection may leave something unconstrained: prune again and re-solve
    use2, _, w2 = _prune(cam, smp, pt, P0, known, int((~held).sum()), fit.inliers, held=gauge)
    if w2:
        warnings += ["after rejection: " + m for m in w2]
        use = use & ~(fit.inliers & ~use2)          # what the second pruning took is not "rejected"
        fit2 = pose.robust_lm(fit.x, residual, normal, step, N, sigma_px=sigma_px, active=use2, robust=False,
                              max_iter=max(max_iter - fit.iterations, 1), log=say, lam0=fit.lam)
        fit2.iterations += fit.iterations
        fit = fit2
    if not fit.converged:
        warnings.append("not converged in %d linearisations" % max_iter)

    # ---------------------------------------------------------- the answer
    x, r, inl = fit.x, fit.r, fit.inliers
    cams = x.cameras(cams0)
    poses = np.concatenate([x.Rb, x.tb[:, :, None]], axis=2)
    rr = r[inl]
    per_cam = np.array([np.sqrt(np.mean(r[inl & (cam == c)] ** 2)) if np.any(inl & (cam == c)) else np.nan
                        for c in range(C)])
    Hi, n_null, _ = pose.inverse(fit.H, fit.free, info=True)
    n_free = int(fit.free.sum()) - n_null
    if n_null:
        warnings.append("%d direction%s of the solution not fixed by the data; every parameter in them has NaN "
                        "covariance" % (n_null, "" if n_null == 1 else "s"))
    dof = 2 * int(inl.sum()) - n_free
    sig = float(sigma_px) if sigma_px else math.sqrt(float(np.sum(rr * rr)) / max(dof, 1))
    cov = _covariance(Hi, sig, cams, cam_col, smp_col, pt_col, C)
    for c in range(C):
        if np.any(np.isnan(cov["cam"][c])) and np.any(use & (cam == c)):
            warnings.append("camera %d (%s): not fixed by the data (NaN covariance)" % (c, cams0[c].name))
    for s in range(S):
        if s != gauge and np.any(np.isnan(cov["pose"][s])):
            warnings.append("sample %d: pose not fixed by the data (NaN covariance); %s" % (
                s, "left at the gimbal's reading" if np.all(np.isnan(cov["pose"][s])) else "its value is arbitrary"))
    for k in range(K):
        if not known[k] and np.any(np.isnan(cov["point"][k])):
            warnings.append("ball %d: position not fixed by the data (NaN covariance); %s" % (
                k, "left as drawn" if np.all(np.isnan(cov["point"][k])) else "its value is arbitrary"))
    for m in warnings:
        say("    warning: " + m)
    say("    rms %.4f px over %d inliers (%d rejected), sigma %.4f px, %d unknowns, %d linearisations"
        % (float(np.sqrt(np.mean(rr ** 2))), int(inl.sum()), int(use.sum() - inl.sum()), fit.scale, n_free,
           fit.iterations))
    stem, stem_sd = np.nan, np.nan
    if k_col >= 0:
        stem, stem_sd = float(x.k), float(sig * np.sqrt(Hi[k_col, k_col]))
        if not np.isfinite(stem_sd):
            warnings.append("the stems' scale is not fixed by the data")
    return Solution(cams, poses, x.P.copy(), float(np.sqrt(np.mean(rr ** 2))), fit.scale, inl, fit.iterations,
                    per_cam, cov, warnings, n_free, gauge, fit.converged, use, stem, stem_sd)


def _covariance(Hi, sig, cams, cam_col, smp_col, pt_col, C):
    """1-sigma standard deviations (and the blocks the checks need) from
    sigma^2 (J^T J)^-1, Hi = (J^T J)^-1 as pose.inverse gives it.  A held
    parameter has 0, one the data do not fix NaN.

      cam       (C, 15)      [dphi (rad, camera axes), t (mm), fx, fy, cx, cy (px), k1..p2]
      cam_full  (C, 15, 15)  each camera's covariance block
      centre    (C, 3)       mm, the camera's centre in the world
      pose      (S, 6)       [dphi (rad, world axes), t (mm)]
      point     (K, 3)       mm, body frame
      cal       (15C + 3K)^2 the joint covariance of every camera's and
                             ball's parameters, the sample poses
                             marginalised: what drift() takes as a prior.
                             Held parameters, and any the data did not
                             fix, enter it as 0: exact
      sigma     the noise it is scaled by
    """
    Sig = sig ** 2 * Hi

    def block(cols):
        m = cols >= 0
        out = np.zeros((len(cols), len(cols)))
        c = cols[m]
        out[np.ix_(m, m)] = Sig[np.ix_(c, c)]
        return out

    def sd(cols):
        out = np.zeros(cols.shape)
        m = cols >= 0
        out[m] = np.sqrt(np.diag(Sig)[cols[m]])
        return out

    cam_full = np.stack([block(cam_col[c]) for c in range(C)])
    centre = np.zeros((C, 3))
    for c, cm in enumerate(cams):
        # c = -R^T t; with R <- exp(dphi) R: dc = -R^T [t]x dphi - R^T dt
        Jc = np.hstack([-cm.R.T @ pose.skew_n(cm.t), -cm.R.T])
        centre[c] = np.sqrt(np.diag(Jc @ cam_full[c][:6, :6] @ Jc.T))
    cal_cols = np.concatenate([cam_col.ravel(), pt_col.ravel()])
    cal = np.nan_to_num(block(cal_cols))
    return {"cam": sd(cam_col), "cam_full": cam_full, "centre": centre, "pose": sd(smp_col), "point": sd(pt_col),
            "cal": cal, "sigma": sig}


# =========================================================== DRIFT CHECK
def chi2_quantile(dof, alpha):
    """The chi-square value exceeded with probability alpha, by Wilson and
    Hilferty's cube-root normal approximation (Wilson & Hilferty 1931, "The
    distribution of chi-square", PNAS 17: 684): (chi2/dof)^(1/3) is nearly
    normal with mean 1 - 2/(9 dof) and variance 2/(9 dof).  At the dof a
    rig check has (hundreds to thousands) it is exact to a few parts in
    10^4."""
    h = 2.0 / (9.0 * dof)
    return dof * (1.0 - h + pose.normal_quantile(1.0 - alpha) * math.sqrt(h)) ** 3


def chi2_sf(chi2, dof):
    """P(chi-square_dof > chi2), by the same approximation."""
    return math.exp(chi2_logsf(chi2, dof))


def chi2_logsf(chi2, dof):
    """ln P(chi-square_dof > chi2), by the same approximation; dof need not
    be a whole number.  In logs, so two cameras far in the tail still
    compare."""
    h = 2.0 / (9.0 * dof)
    z = ((max(chi2, 0.0) / dof) ** (1.0 / 3.0) - (1.0 - h)) / math.sqrt(h)
    return _norm_logsf(z)


def _norm_logsf(z):
    """ln P(N(0,1) > z): erfc while it is above the smallest double, then
    the first term of its asymptotic series, -z^2/2 - ln(z sqrt(2 pi))
    (Abramowitz & Stegun 7.1.23), whose relative error 1/z^2 is under 1e-3
    out there."""
    q = 0.5 * math.erfc(z / math.sqrt(2.0))
    if q > 0.0:
        return math.log(q)
    return -0.5 * z * z - math.log(z * math.sqrt(2.0 * math.pi))


def hypergeom_logsf(k, N, n, R):
    """ln P(X >= k), X hypergeometric: the number of R items drawn without
    replacement from N that fall among a particular n of them.  Exact, by
    the sum of its terms in logs."""
    lo, hi = max(int(k), 0, R - (N - n)), min(n, R)
    if k <= max(0, R - (N - n)):
        return 0.0
    if lo > hi:
        return -math.inf

    def lchoose(a, b):
        return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)

    t = np.array([lchoose(n, j) + lchoose(N - n, R - j) for j in range(lo, hi + 1)]) - lchoose(N, R)
    return float(t.max() + math.log(np.sum(np.exp(t - t.max()))))


def _prior_factor(Sig):
    """L with L L^T = Sig, for a positive semi-definite Sig in mixed units:
    the eigenvectors of its correlation matrix, the numerically zero
    eigenvalues (held parameters, by numpy's matrix_rank tolerance)
    dropped."""
    d = np.sqrt(np.clip(np.diag(Sig), 0.0, None))
    m = d > 0
    D = d[m]
    Cr = Sig[np.ix_(m, m)] / np.outer(D, D)
    lam, V = np.linalg.eigh(Cr)
    keep = lam > lam.max() * len(lam) * np.finfo(float).eps
    L = np.zeros((len(Sig), int(keep.sum())))
    L[m] = D[:, None] * V[:, keep] * np.sqrt(lam[keep])
    return L


def drift(cams, points, obs, sigma_px, alpha=1e-3, init=None, cal=None, expected=None) -> dict:
    """The shift-start check: does the calibration still explain new
    observations to the noise?

    cams      the calibrated cameras, held fixed
    points    Points with the calibrated body-frame positions (Solution.P)
    obs       list of Obs: new samples of the target, any number of them
    sigma_px  the centroid noise the test is against
    alpha     the false-alarm rate: at most alpha (Bonferroni over the two
              questions below, alpha / 2 each)
    init      optional {sample: (3, 4) pose} from the gimbal; without it each
              pose starts from triangulation
    cal       the Solution the cameras came from: its covariance enters the
              chi-square, so an undisturbed rig fails at alpha and not more
    expected  optional {sample: (C,) counts}: how many balls each camera
              should have seen in that sample.  Without it a camera that
              sees nothing (dark, blocked, or knocked so far that nothing
              it sees can be matched) leaves no observations and so no
              evidence, and passes

    No pose is passed in: each sample's pose is re-estimated here, from
    the observations (pose.body_pose, robust, with sigma_px), because a pose
    from the gimbal would carry the gimbal's error into a test of the
    cameras.  The pose fit splits the observations into inliers and the
    ones the others cannot explain, and the check asks two questions:

      do the inliers fit to the noise?  sum |r|^2 / sigma^2 over them is
          chi-square with 2 n_in - 6 S dof; the limit is its 1 - alpha/2
          quantile.  A rejection limit some four sigmas out truncates the
          chi-square by under FALSE_REJECT x (c^2 + 2) ~ 0.2 a sample, far
          inside its spread.  This sees a camera that has moved by a little,
          its observations still inliers but each a few sigmas off.
      is one camera's share of the rejections more than chance?  A gross
          centroid (a reflection, two blobs merged) is a detector fault,
          not drift, and happens to any camera alike; a camera that has
          moved by more has most of its observations rejected.  Given R
          rejected of n, the number in camera c (of its n_c) is
          hypergeometric under the null whatever the fault rate (the
          conditional test of equal rates, as in Fisher's exact test), so
          no fault rate has to be assumed; each camera seen is tested at
          alpha / (2 C).  With `expected`, a ball a camera should have seen
          and did not is counted with its rejections, against everything
          the cameras should have seen: a camera that has gone dark or
          moved past matching is then the one with all of them.

    The rig passes when neither fails.  The worst camera is the one whose
    own evidence is strongest: the smaller of its rejected-share p and the
    p of its inliers' own chi-square (sum |r|^2 / sigma^2 against the
    dof its residuals keep, sum of 2 - h_i with h_i its leverage in its
    pose fit).  Neither is moved by one gross centroid, which a camera's
    rms over everything is.  A centroid that is not a finite number is
    counted (non_finite) and not used: it is a detector's fault, not the
    calibration's.

    Returns a dict: ok; chi2, dof, limit, p (the inliers' chi-square, its
    dof, its 1 - alpha/2 quantile, its p) and chi2_fixed (the same taking
    the calibration as exact); camera_limit (the p under which a camera's
    rejected share fails); per camera (C,): per_cam_n, per_cam_rejected,
    per_cam_p_rejected, per_cam_p_fit, per_cam_p (the smaller),
    per_cam_rms (px, its inliers), per_cam_median_px (|r| over all its
    observations); worst_cam, worst_index; why (what failed, empty when
    ok); n_obs, n_inliers, rejected, non_finite, skipped (samples whose
    pose could not be fitted), samples."""
    cam, smp, pt, uv = _arrays(obs)
    C = len(cams)
    P = np.asarray(points.P, float)
    rad_k = np.broadcast_to(np.asarray(points.radius, float), (len(P),))
    if cal is not None and cal.cov["cal"].shape[0] != N_PARAMS * C + N_POINT * len(P):
        raise ValueError("cal is the calibration of a different set of cameras or balls")
    finite = np.all(np.isfinite(uv), axis=1)
    fits, skipped = {}, []
    for s in np.unique(smp[finite]):
        sel = np.flatnonzero(finite & (smp == s))
        f = pose.body_pose(cams, P, rad_k, [(cam[j], pt[j], uv[j, 0], uv[j, 1]) for j in sel],
                           init=None if init is None or s not in init else init[s], sigma_px=sigma_px)
        if f is None:
            skipped.append(int(s))
        else:
            fits[int(s)] = f
    order = sorted(fits)
    idx = np.flatnonzero(finite & np.isin(smp, order))
    loc = np.searchsorted(order, smp[idx])
    inl = np.zeros(len(idx), bool)
    for j, s in enumerate(order):
        inl[loc == j] = fits[s].inliers                # body_pose saw them in this order
    Rb = np.stack([fits[s].R for s in order]) if order else np.zeros((0, 3, 3))
    tb = np.stack([fits[s].t for s in order]) if order else np.zeros((0, 3))
    c_, p_, rad = cam[idx], pt[idx], rad_k[pt[idx]]
    r = _predict(cams, Rb, tb, P, c_, loc, p_, rad) - uv[idx]
    n, n_in, S = len(idx), int(inl.sum()), len(order)
    dof = 2 * n_in - N_POSE * S
    out = {"skipped": skipped, "n_obs": n, "n_inliers": n_in, "rejected": n - n_in,
           "non_finite": int((~finite).sum()), "samples": order}
    if dof <= 0:
        out.update({"ok": False, "chi2": np.nan, "dof": dof, "limit": np.nan, "p": np.nan,
                    "per_cam_rms": np.full(C, np.nan), "worst_cam": None, "worst_index": None,
                    "why": "too few observations to test"})
        return out
    z2 = np.sum(np.where(inl[:, None], r, 0.0) ** 2, axis=1) / sigma_px ** 2     # inliers only
    chi2_fixed = float(z2.sum())
    chi2 = chi2_fixed
    Jc, Js, Jp = _blocks(cams, Rb, tb, P, c_, loc, p_, rad)
    if cal is not None:
        # min over the poses and the calibration's error dc, the latter
        # with its prior: |r + Js dxi + Jcal dc|^2 / sigma^2 + dc^T Sig^-1 dc,
        # dc = L z, over the inliers.  One Gauss-Newton step from the poses
        # just fitted.
        K = len(P)
        L = _prior_factor(cal.cov["cal"])
        ii = np.flatnonzero(inl)
        rows = np.arange(len(ii))
        Jcal = np.zeros((len(ii), 2, N_PARAMS * C + N_POINT * K))
        for j in range(N_PARAMS):
            Jcal[rows, :, N_PARAMS * c_[ii] + j] = Jc[ii, :, j]
        for j in range(N_POINT):
            Jcal[rows, :, N_PARAMS * C + N_POINT * p_[ii] + j] = Jp[ii, :, j]
        Jpose = np.zeros((len(ii), 2, N_POSE * S))
        for j in range(N_POSE):
            Jpose[rows, :, N_POSE * loc[ii] + j] = Js[ii, :, j]
        m = L.shape[1]
        A = np.vstack([np.hstack([Jpose.reshape(2 * len(ii), -1), Jcal.reshape(2 * len(ii), -1) @ L]) / sigma_px,
                       np.hstack([np.zeros((m, N_POSE * S)), np.eye(m)])])
        b = np.concatenate([-r[ii].ravel() / sigma_px, np.zeros(m)])
        y = np.linalg.lstsq(A, b, rcond=None)[0]
        e = A @ y - b
        chi2 = float(e @ e)
    limit = chi2_quantile(dof, alpha / 2.0)

    # each inlier's leverage in its own pose fit, h_i = tr(J_i (J^T J)^-1 J_i^T):
    # the share of the 6 pose dof its residual gave up
    lev = np.zeros(n)
    for j, s in enumerate(order):
        k = np.flatnonzero((loc == j) & inl)
        Hi = fits[s].cov / fits[s].sigma_px ** 2
        h = np.einsum("nai,ij,naj->n", Js[k], Hi, Js[k])
        lev[k] = np.where(np.isfinite(h), h, N_POSE / max(len(k), 1))
    n_c = np.bincount(c_, minlength=C)
    k_c = np.bincount(c_[~inl], minlength=C)
    t_c, f_c = n_c.copy(), k_c.copy()                 # trials and failures for the share test
    if expected is not None:
        e_c = np.zeros(C, int)
        for s in order:
            e_c += np.asarray(expected.get(s, np.zeros(C)), int)
        miss = np.maximum(e_c - n_c, 0)
        t_c, f_c = n_c + miss, k_c + miss
    chi2_c = np.bincount(c_, z2, minlength=C)
    dof_c = np.bincount(c_[inl], 2.0 - lev[inl], minlength=C)
    seen = t_c > 0
    N, F = int(t_c.sum()), int(f_c.sum())
    lp_rej = np.array([hypergeom_logsf(f_c[c], N, t_c[c], F) if seen[c] else 0.0 for c in range(C)])
    lp_fit = np.array([chi2_logsf(chi2_c[c], dof_c[c]) if dof_c[c] > 0 else 0.0 for c in range(C)])
    lp_cam = np.where(seen, np.minimum(lp_rej, lp_fit), np.inf)
    worst = int(np.argmin(lp_cam))
    cam_floor = math.log(alpha / (2.0 * max(int(seen.sum()), 1)))
    ok_fit = chi2 <= limit
    ok_rej = bool(np.all(lp_rej[seen] >= cam_floor))
    why = []
    if not ok_fit:
        why.append("the inliers' chi-square %.0f is over its limit %.0f" % (chi2, limit))
    for c in np.flatnonzero(seen & (lp_rej < cam_floor)):
        why.append("%s: %d of its %d observations unexplained or missing (%d of %d over all cameras; p %.2g)"
                   % (cams[c].name, f_c[c], t_c[c], F, N, math.exp(lp_rej[c])))
    per_cam = np.array([np.sqrt(np.mean(r[inl & (c_ == c)] ** 2)) if np.any(inl & (c_ == c)) else np.nan
                        for c in range(C)])
    with np.errstate(invalid="ignore"):
        med = np.array([np.nanmedian(np.hypot(*r[c_ == c].T)) if n_c[c] else np.nan for c in range(C)])
    out.update({"ok": bool(ok_fit and ok_rej), "chi2": chi2, "dof": dof, "limit": limit, "p": chi2_sf(chi2, dof),
                "chi2_fixed": chi2_fixed, "per_cam_rms": per_cam, "per_cam_median_px": med, "per_cam_n": n_c,
                "per_cam_rejected": k_c, "per_cam_missing": t_c - n_c,
                "per_cam_p": np.exp(np.where(seen, lp_cam, 0.0)),
                "per_cam_p_rejected": np.exp(lp_rej), "per_cam_p_fit": np.exp(lp_fit),
                "camera_limit": math.exp(cam_floor),
                "worst_cam": cams[worst].name, "worst_index": worst, "why": "; ".join(why)})
    return out
