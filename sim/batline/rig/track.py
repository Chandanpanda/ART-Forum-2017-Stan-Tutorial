"""From blobs to a calibrated rig, and from a calibrated rig to the clamp's
pose: prediction, association, self-calibration's loop, and tracking.

Nothing here reads the scene's truth.  It knows the rig as drawn (its
cameras, the gimbal's kinematics, where the balls were meant to be, the
certificate of the bar), what the cameras report (vision.RigVisionHAL),
and what it has solved for itself.

PREDICTION.  Where each ball should image, through the cameras as
believed, at the pose believed; and whether that image can be used: in
front, wholly on the sensor, touching no other ball's image (vision.apart),
its stem seen side-on enough for its pull to be known (vision.stem_clear),
and clear of everything the rig as drawn could put in front of it.  The
last is a guard band: rays round each ball at its radius plus GUARD, cast
in the scene as drawn, since the real structure lies up to the build's
tolerance from its drawing and a ball partly hidden reads a centroid that
is wrong rather than missing.

ASSOCIATION.  A blob is a ball's when each is the other's nearest (the
blob among blobs, the ball among every ball predicted in front of the
camera, usable or not, since a hidden ball's sliver is a blob too), it lies
within the gate, and it is nearer its ball's prediction than half the
distance from there to any other ball's -- so no other ball could have
made it.  Ambiguous blobs wait for a better prediction.

THE STEMS' PULL.  A stem seen from its own side hides a strip of its
ball and pulls the centroid away from it (vision.stem_bias_px), by up to
several times the centroid's noise.  The pull's shape is known from the
geometry and its scale is one number for the rig: self-calibration solves
it inside the bundle adjustment, with every camera, pose and ball
(calib.bundle_adjust's pull), so its sigma is net of what they absorb, and
tracking corrects every centroid by that scale times the model's pull.

SELF-CALIBRATION.  The bar in the clamp, the gimbal over a grid of poses
on both axes' whole turn; per round, associate against the current
belief, bundle-adjust everything with the stems' scale (calib.bundle_adjust),
re-predict, until the association is one already solved (unchanged, or
cycling: a ball on the edge of a test goes in and out with each round's
small change).  The first round's belief is the drawing, wrong by
tens of pixels (prior_px), and the area model's scale, 1; each camera's
bulk shift is taken out first, as the median offset of its unambiguous
matches.
"""
from dataclasses import dataclass, field
from math import radians, sqrt, tan

import numpy as np

from ..spec import RigBuild, RingCam, Module
from . import spec as RS
from . import mjcf as RM
from . import vision as V
from .calib import Obs, Points, bundle_adjust
from .pose import body_pose

ROUNDS = 8                     # rounds at most: the drawing, then each solution; the association settles
                               # in two, the stems' scale in a few more (each round leaves a part of its
                               # error that the solution's free markers absorbed)


def encoder_pose(a, b):
    """(3, 4) body -> rig as the gimbal's drawing says, at command (a, b) deg."""
    return np.hstack([RS.body_R(a, b), np.zeros((3, 1))])


def guard_mm():
    """mm a usable ball's image keeps from anything that could hide it: the
    build's tolerance of the rig's structure about the balls
    (rig/spec.clear), the same margin its moving parts keep."""
    return RS.clear()


def prior_px(rig):
    """px, 1 sigma: how far the drawing's prediction may lie from a blob --
    each camera's tilt, place, principal point, focal length and lens, and
    the gimbal's zeros and axes, at the far edge of the image."""
    L = rig.lens_drawn()
    f = L.fx
    x_edge = np.hypot(*L.ray(0.0, 0.0))                       # normalised radius of the corner
    k = RigBuild.DIST
    lens = f * np.hypot(k[0] * x_edge ** 3, k[1] * x_edge ** 5)
    body = f / rig.standoff * np.hypot(rig.R_s * radians(RigBuild.ENCODER_ZERO), RigBuild.CENTRE)
    return float(np.sqrt((f * tan(radians(RigBuild.CAM_TILT))) ** 2 + (f * RigBuild.CAM_POS / rig.standoff) ** 2
                         + RigBuild.CENTRE_PX ** 2 + (f * RigBuild.FOCAL * x_edge) ** 2 + lens ** 2 + body ** 2))


def kinematic_px(rig):
    """px, 1 sigma: how far the gimbal's drawn kinematics put a ball from
    where it is, once the cameras are calibrated -- its zeros, skew,
    offset and centre."""
    L = rig.lens_drawn()
    ang = radians(np.hypot(RigBuild.ENCODER_ZERO * sqrt(2.0), RigBuild.AXIS_SKEW))
    return float(L.fx / (rig.standoff - rig.R_s) * np.hypot(rig.R_s * ang,
                                                              np.hypot(RigBuild.AXIS_OFFSET, RigBuild.CENTRE)))


class Predictor:
    """Where balls image and which can be used, against the rig as drawn
    with `payload` in the clamp."""

    def __init__(self, rig, site, cams_drawn, payload):
        import mujoco
        self.rig = rig
        xml, info = RM.scene(rig, site, cams_drawn, None, payload=payload, size=(64, 64), overview=False)
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.drawn = RM.RigBuildDraw.drawing(rig, cams_drawn)
        self.guard = guard_mm()
        self.n_bar = len(rig.bar) if payload == "bar" else 0

    def __call__(self, cams, a, b, R, t, P, r):
        """Per camera (uv (K, 2), front (K,) bool, usable (K,) bool, pull
        (K, 2) px) of balls P (body frame, mm, radius r; the ring's markers,
        then the bar's balls if it is there) at body pose (R, t) -- the
        structure drawn at command (a, b).  uv is the ball's centre; its
        blob lies at uv + scale x pull (vision.stem_bias_px)."""
        RM.set_gimbal(self.m, self.d, a, b, self.drawn)
        P = np.asarray(P)
        X = P @ np.asarray(R).T + np.asarray(t)
        K = len(self.rig.markers)
        S = RM.stems(self.rig, P[:K], P[K:]) @ np.asarray(R).T
        L = RM.stem_lengths(self.rig, P[:K], P[K:])
        own = RM.stem_ids(self.m, K, min(len(P) - K, self.n_bar))
        own = np.r_[own, -np.ones(len(P) - len(own), int)]
        out = []
        for c in cams:
            u, v, z = c.project(X, r)
            rr = V.image_radius(c, X, r)
            front = z > r
            ok = front & V.wholly_on(c, u, v, rr) & V.stem_clear(c, X, S, L, r)
            if ok.any():
                ok[ok] = V.clear_of(self.m, self.d, c.centre, X[ok], r, rho=r + self.guard, own=own[ok])
            ok = V.apart(u, v, rr, ok, front)
            bias, dirn = V.stem_bias_px(c, X, S, rr, r)
            out.append((np.stack([u, v], axis=1), front, ok, bias[:, None] * dirn))
        return out


def associate(blobs, uv, front, usable, gate, shift=(0.0, 0.0)):
    """[(blob index, ball index)] for one camera."""
    if len(blobs) == 0 or not usable.any():
        return []
    q = uv + np.asarray(shift)
    live = np.flatnonzero(front)
    B = blobs[:, :2]
    D = np.hypot(B[:, None, 0] - q[None, live, 0], B[:, None, 1] - q[None, live, 1])     # (N, L)
    out = []
    for jl, k in enumerate(live):
        if not usable[k]:
            continue
        i = int(np.argmin(D[:, jl]))
        if D[i, jl] > gate or int(np.argmin(D[i])) != jl:
            continue
        others = np.delete(np.hypot(*(q[live] - q[k]).T), jl)
        if len(others) and D[i, jl] >= 0.5 * others.min():
            continue
        out.append((i, int(k)))
    return out


@dataclass
class SelfCal:
    sol: object                # calib.Solution
    poses: np.ndarray          # (S, 2) the gimbal's commands
    points: Points             # as solved (Solution.P)
    obs: list                  # the last round's association, each centroid corrected for its stem
    rounds: list = field(default_factory=list)    # (round, matched, inliers, rms px, stems' scale)
    snaps: list = None
    stem: float = 1.0          # the stems' pull, of the area model's (vision.stem_bias_px)
    stem_sd: float = np.inf    # its 1 sigma, marginal (calib.Solution.stem_sd)


def _matches(snaps, pr, gate, k, shifts=None):
    """[Obs] of every sample and camera, matched where each ball's blob
    should be at the stems' scale k -- the blobs' own centroids -- and
    (N, 2) each one's stem's pull at scale 1."""
    obs, pull = [], []
    for s, per in enumerate(pr):
        for c, (uv, front, ok, d) in enumerate(per):
            sh = (0.0, 0.0) if shifts is None else shifts[c]
            for i, j in associate(snaps[s][c], uv + k * d, front, ok, gate, sh):
                obs.append(Obs(c, s, j, float(snaps[s][c][i, 0]), float(snaps[s][c][i, 1])))
                pull.append(d[j])
    return obs, np.array(pull).reshape(-1, 2)


def self_calibrate(vision, rig, site, poses, bar_cert, sigma_px=RingCam.CENTROID_PX, log=None, fix_dist=()):
    """SelfCal: the rig calibrated from the bar in the clamp over `poses`.
    `vision` must be looking at the bar (payload "bar"); `bar_cert` is the
    bar's certificate, its balls in the body frame.

    sigma_px is the centroid error the solution's covariance is stated at:
    the rig's budget, CENTROID_PX, which check_cameras holds rendered
    frames to, what is left of the stems' pull included -- not the
    residuals' scatter, which is smaller and cannot see an error the
    solution absorbs."""
    say = log or (lambda m: None)
    cams0 = vision.cams
    snaps = [vision.snap(a, b) for a, b in poses]
    K = len(rig.markers)
    P0 = np.vstack([rig.markers, np.asarray(bar_cert, float)])
    known = np.r_[np.zeros(K, bool), np.ones(len(rig.bar), bool)]
    points = Points(P0.copy(), np.full(len(P0), rig.ball_r), known)
    pred = Predictor(rig, site, cams0, "bar")
    gate = Module.Z_SIGMAS * prior_px(rig)
    cams, Rt = cams0, np.array([encoder_pose(a, b) for a, b in poses])
    P = P0
    k = 1.0
    sol, last, rounds = None, set(), []
    for rnd in range(ROUNDS):
        pr = [pred(cams, a, b, Rt[s][:, :3], Rt[s][:, 3], P, rig.ball_r) for s, (a, b) in enumerate(poses)]
        shifts = _shifts(snaps, pr, gate, k) if rnd == 0 else None
        obs, pull = _matches(snaps, pr, gate, k, shifts)
        key = frozenset((o.cam, o.sample, o.point) for o in obs)
        if key in last:
            break                      # settled, or cycling between sets it has solved already
        last.add(key)
        # each round starts from the last one's solution: the gauge sample's
        # pose is still the gimbal's reading, since the solve held it there;
        # the stems' scale is solved with the rest
        sol = bundle_adjust(cams, Rt, Points(np.where(known[:, None], P0, P), points.radius, known), obs,
                            sigma_px=sigma_px, fix_dist=fix_dist, pull=pull, stem0=k)
        cams, Rt, P, k = sol.cams, sol.poses, sol.P, sol.stem
        rounds.append((rnd, len(obs), int(sol.inliers.sum()), sol.rms_px, k))
        say("round %d: %d matched, %d inliers, rms %.3f px, stems' scale %.3f +- %.3f" % (rounds[-1] + (sol.stem_sd,)))
    if sol is None:
        raise ValueError("self-calibration matched nothing")
    corrected = [Obs(o.cam, o.sample, o.point, o.u - k * d[0], o.v - k * d[1]) for o, d in zip(obs, pull)]
    return SelfCal(sol, np.asarray(poses), Points(P.copy(), points.radius, known), corrected, rounds, snaps, k,
                   sol.stem_sd)


def _shifts(snaps, pr, gate, k):
    """Per camera: the median offset of blob from prediction over its
    unambiguous matches -- the bulk of a camera's error, its tilt and its
    principal point, taken out before the first association."""
    C = len(pr[0])
    out = []
    for c in range(C):
        d = []
        for s in range(len(snaps)):
            uv, front, ok, pull = pr[s][c]
            q = uv + k * pull
            for i, j in associate(snaps[s][c], q, front, ok, gate):
                d.append(snaps[s][c][i, :2] - q[j])
        out.append(tuple(np.median(d, axis=0)) if len(d) else (0.0, 0.0))
    return out


@dataclass
class Track:
    fit: object                # pose.PoseFit, or None
    obs: list                  # (cam, ball, u, v), each centroid corrected for its stem
    seen: np.ndarray           # (K,) cameras each ball was matched in


def track(vision, cal, rig, pred, a, b, sigma_px=RingCam.CENTROID_PX):
    """The clamp's pose at gimbal command (a, b) from one snap, through the
    calibrated cameras, starting from the gimbal's own reading; its
    covariance stated at sigma_px, as self_calibrate's."""
    blobs = vision.snap(a, b)
    T0 = encoder_pose(a, b)
    P = cal.points.P[:len(rig.markers)]
    pr = pred(cal.sol.cams, a, b, T0[:, :3], T0[:, 3], P, rig.ball_r)
    gate = Module.Z_SIGMAS * kinematic_px(rig)
    obs = []
    for c, (uv, front, ok, pull) in enumerate(pr):
        q = cal.stem * pull
        for i, j in associate(blobs[c], uv + q, front, ok, gate):
            obs.append((c, j, float(blobs[c][i, 0] - q[j, 0]), float(blobs[c][i, 1] - q[j, 1])))
    seen = np.bincount([o[1] for o in obs], minlength=len(P))
    if not obs:
        return Track(None, obs, seen)
    fit = body_pose(cal.sol.cams, P, rig.ball_r, obs, init=T0, sigma_px=sigma_px)
    return Track(fit, obs, seen)


def drift_check(vision, cal, rig, pred, poses, sigma_px=RingCam.CENTROID_PX, alpha=None):
    """The shift-start check (calib.drift): the ring's markers seen at a few
    gimbal poses, matched against the calibration, and the residuals tested
    against the noise.  dict as calib.drift's.  Every ball the calibration
    says a camera should see and use counts: one it does not match is
    evidence against that camera, so a camera gone dark, or knocked past
    the association gate, fails rather than going quiet."""
    from .calib import drift, Obs as O
    obs, init, expected = [], {}, {}
    P = cal.points.P
    K = len(rig.markers)
    gate = Module.Z_SIGMAS * kinematic_px(rig)
    for s, (a, b) in enumerate(poses):
        blobs = vision.snap(a, b)
        T0 = encoder_pose(a, b)
        init[s] = T0
        pr = pred(cal.sol.cams, a, b, T0[:, :3], T0[:, 3], P[:K], rig.ball_r)
        expected[s] = np.array([int(ok.sum()) for _uv, _f, ok, _p in pr])
        for c, (uv, front, ok, pull) in enumerate(pr):
            q = cal.stem * pull
            for i, j in associate(blobs[c], uv + q, front, ok, gate):
                obs.append(O(c, s, j, float(blobs[c][i, 0] - q[j, 0]), float(blobs[c][i, 1] - q[j, 1])))
    kw = {} if alpha is None else {"alpha": alpha}
    return drift(cal.sol.cams, cal.points, obs, sigma_px, init=init, cal=cal.sol, expected=expected, **kw)
