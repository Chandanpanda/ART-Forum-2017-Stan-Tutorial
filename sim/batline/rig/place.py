"""Where the eight cameras stand: a coverage solver.

The rule (spec.Rig): at every pose the gimbal can take, with any one camera
lost, the clamp's pose rests on at least MIN_CHECKED markers each seen
whole by MIN_VIEWS cameras -- and the pose is fixed within the rig's share
of the calibration targets.  No camera stands in the gantry's way or the
gimbal's sweep, and none is placed by eye.

CANDIDATES.  Directions spread evenly over the sphere (a Fibonacci
lattice), each a camera at the rig's standoff looking at the gimbal's
centre with its image level, kept where rig/spec.allowed says a camera may
stand: outside the sweep, the gantry's box, the posts, above the floor, on
the station's own floor and off the aisle.

SEEN, WHOLE.  Candidate c sees marker k at pose p when the ball is in
front of it, its image lies on the sensor inside the margin the build may
shift it by, the image is MIN_PX across where the lens shrinks it, its stem is seen
side-on enough for its pull on the centroid to be known (vision.stem_clear;
track.py corrects it), the ball's centre and points round it, out to
the guard band the tracker keeps (track.guard_mm), are all in clear view
-- rays cast in the scene as drawn, against the structure and boxes
enclosing the bat (rig/mjcf.py's groups 0 and 2) -- and its image keeps
clear of every other ball's (vision.apart).  A ball partly hidden is not
seen: its centroid would be wrong, not missing.  This is the tracker's
own test of a usable ball (track.Predictor), with the placement's two
stand-ins: the boxes for whichever bat is in the clamp, and the margin for
where the build may shift the image.  A looser test here would place
cameras for views the tracker then refuses.

GOOD ANGLES.  Two cameras that see a ball from nearly the same side fix it
poorly along their common line.  What "good" means here is what the angles
are for: the body's pose.  At each pose the 6x6 Gauss-Newton information
of the body's pose from every ball every chosen camera sees, at the
centroid's own noise (RingCam.CENTROID_PX), says how well the cameras fix
it; the worst pose's 1-sigma, with any one camera lost, is the figure.

THE SEARCH, lexicographic on (the rule's deficit, the coverage's
shortfall, the margin's deficit, the worst sigma).  The coverage, Sum over
poses and markers of min(views, MIN_VIEWS), is monotone submodular, so
adding the camera that adds most, eight times, is within 1 - 1/e of the
best set (Nemhauser, Wolsey and Fisher 1978).  That greedy runs from every
candidate as the first camera, and each result is improved by 1-swaps (the
local search of facility location, Arya et al. 2004) on the shortfall,
which nearly always meets the rule on the way; when none does, the best
are swapped again on the rule itself.
"""
from dataclasses import dataclass, field
from math import sqrt, pi, radians, degrees

import numpy as np

from ..spec import RingCam, Rig, RigBuild, Module
from ..product import Fit
from .lens import Camera, look_at
from . import spec as RS
from . import vision as V

LIMB = V.RIM                   # points round a ball's outline a view must clear, besides its centre: the tracker's
PROXY_GROUPS = np.array([1, 0, 1, 0, 1, 0], np.uint8)    # the structure, the stems and the boxes round the bat


def fibonacci(n):
    """n unit directions spread evenly over the sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    th = pi * (1.0 + sqrt(5.0)) * i
    s = np.sqrt(1.0 - z * z)
    return np.stack([s * np.cos(th), s * np.sin(th), z], axis=1)


def candidates(rig, site, n):
    """[Camera] as drawn, at every allowed lattice direction."""
    L = rig.lens_drawn()
    P = fibonacci(n) * rig.standoff
    ok = RS.allowed(site, P)
    return [Camera.at("c%d" % i, L, P[i], look_at(P[i], (0.0, 0.0, 0.0)), dir=i) for i in np.where(ok)[0]]


def _local_mag(L, x, y):
    """The lens's smallest of radial and tangential magnification at each
    undistorted point: how much it shrinks a ball's image there."""
    r2 = x * x + y * y
    tang = 1.0 + L.k1 * r2 + L.k2 * r2 * r2 + L.k3 * r2 ** 3
    rad = 1.0 + 3.0 * L.k1 * r2 + 5.0 * L.k2 * r2 * r2 + 7.0 * L.k3 * r2 ** 3
    return np.minimum(tang, rad)


def margin_px(rig):
    """px a ball's image centre keeps from the sensor's edge: its own
    radius, plus Module.Z_SIGMAS of where the build may shift the image
    (the principal point, and the aim over the far side's range)."""
    L = rig.lens_drawn()
    r_img = L.fx * rig.ball_r / (rig.standoff - rig.R_inner)
    aim = L.fx * np.tan(radians(RigBuild.CAM_TILT))
    return r_img + Module.Z_SIGMAS * np.hypot(RigBuild.CENTRE_PX, aim)


@dataclass
class Coverage:
    poses: np.ndarray          # (P, 2) deg
    seen: np.ndarray           # (C, P, K) bool
    info: np.ndarray           # (C, P, 6, 6) the pose information each candidate gives, at each pose
    cams: list                 # the candidates


def coverage(rig, site, cams, poses, payload="proxy"):
    """Which candidate sees which marker whole at which pose, and the pose
    information each gives: rays cast in the scene as drawn."""
    import mujoco
    from . import mjcf as RM
    xml, _ = RM.scene(rig, site, cams[:1], None, payload=payload, size=(64, 64), overview=False)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    b = RM.RigBuildDraw.drawing(rig, cams[:1])
    K, C, NP = len(rig.markers), len(cams), len(poses)
    r = rig.ball_r
    margin = margin_px(rig)
    seen = np.zeros((C, NP, K), bool)
    info = np.zeros((C, NP, 6, 6))
    sig = RingCam.CENTROID_PX
    S0 = RM.stems(rig, rig.markers)
    L0 = RM.stem_lengths(rig, rig.markers)
    rho = r + RS.clear()                                 # the tracker's guard band (track.guard_mm)
    own = RM.stem_ids(m, K)
    for ip, (a, b_) in enumerate(poses):
        RM.set_gimbal(m, d, a, b_, b)
        Rb, tb = RM.body_pose(m, d)
        X = rig.markers @ Rb.T + tb                       # mm, rig frame
        S = S0 @ Rb.T
        for ic, cam in enumerate(cams):
            L = cam.lens
            u, v, z = cam.project(X, r)
            Xc = cam.to_cam(X)
            xs, ys = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
            px = 2.0 * L.fx * r / np.maximum(z, 1e-9) * _local_mag(L, xs, ys)
            ok = (z > r) & L.inside(u, v, margin) & (px >= RingCam.MIN_PX)
            ok &= V.stem_clear(cam, X, S, L0, r)
            if not ok.any():
                continue
            clear = np.zeros(K, bool)
            clear[ok] = V.clear_of(m, d, cam.centre, X[ok], r, rho=rho, groups=PROXY_GROUPS, n=LIMB, own=own[ok])
            vis = V.apart(u, v, V.image_radius(cam, X, r), ok & clear, z > r)
            seen[ic, ip] = vis
            if vis.any():
                info[ic, ip] = _pose_info(cam, X[vis], r) / sig ** 2
    return Coverage(np.asarray(poses), seen, info, cams)


def _pose_info(cam, X, r, h=1e-3):
    """J^T J of the pixels of balls X (mm, world) with respect to the body's
    pose: a small rotation (rad) and translation (mm) about the rig's
    origin, X' = X + w x X + t."""
    X = np.atleast_2d(X)
    n = len(X)
    E = h * np.eye(3)
    up, vp, _ = cam.project((X[:, None, :] + E[None]).reshape(-1, 3), r)
    um, vm, _ = cam.project((X[:, None, :] - E[None]).reshape(-1, 3), r)
    J = np.stack([(up - um).reshape(n, 3), (vp - vm).reshape(n, 3)], axis=1) / (2 * h)   # (n, 2, 3)
    A = np.concatenate([-np.stack([_skew(x) for x in X]), np.broadcast_to(np.eye(3), (n, 3, 3))], axis=2)
    Jp = J @ A                                                                          # (n, 2, 6)
    return np.einsum("nij,nik->jk", Jp, Jp)


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def pose_sigma(info_sum, at=None, R=None):
    """(deg, mm) per pose: the 1-sigma rotation of the body (its largest
    axis) and the 1-sigma position of the body points `at` (the largest
    axis of the worst; the origin if None), from summed information (P, 6,
    6).  The information is about a turn and a shift of the rig frame
    about its origin (_pose_info), so each body point is taken where it is
    at each pose: R (P, 3, 3), the body's rotation there (identity if
    None)."""
    bad = np.linalg.matrix_rank(info_sum) < 6
    cov = np.linalg.pinv(info_sum)
    rot = np.sqrt(np.linalg.eigvalsh(cov[:, :3, :3])[:, -1])
    pts = [np.zeros(3)] if at is None else at
    n = len(info_sum)
    pos = np.zeros(n)
    I3 = np.broadcast_to(np.eye(3), (n, 3, 3))
    for p in pts:
        q = np.broadcast_to(np.asarray(p, float), (n, 3)) if R is None else np.asarray(R) @ np.asarray(p, float)
        A = np.concatenate([-np.array([_skew(x) for x in q]), I3], axis=2)
        Cp = A @ cov @ A.transpose(0, 2, 1)
        pos = np.maximum(pos, np.sqrt(np.linalg.eigvalsh(Cp)[:, -1]))
    rot[bad], pos[bad] = np.inf, np.inf
    return np.degrees(rot), pos


@dataclass
class Placement:
    chosen: list               # indices into the coverage's candidates
    cams: list                 # [Camera] as drawn, named cam0..
    views: np.ndarray          # (P, K) cameras that see each marker whole at each pose
    deficit: int               # the rule's: checked markers short, over poses and each camera lost
    shortfall: int             # Sum over poses and markers of the views short of MIN_VIEWS
    worst_rot: float           # deg, 1 sigma, the worst pose's rotation, any one camera lost
    worst_pos: float           # mm, 1 sigma, the worst pose's worst bat point, any one camera lost
    log: list = field(default_factory=list)


def _short(count):
    """Views short of MIN_VIEWS, summed, for counts (..., P, K)."""
    return np.maximum(0, Rig.MIN_VIEWS - count).sum(axis=(-2, -1))


def _lack(count, need=Rig.MIN_CHECKED):
    """Checked markers short of `need`, summed over poses, for counts
    (..., P, K)."""
    return np.maximum(0, need - (count >= Rig.MIN_VIEWS).sum(axis=-1)).sum(axis=-1)


def deficit(seen, S, need=Rig.MIN_CHECKED):
    """The rule's deficit for cameras S: over every camera lost in turn,
    the checked markers short of `need`, summed over poses.  With need one
    more than the rule's, it is the margin's: how much of the rule rests on
    exactly MIN_CHECKED, which a pose between the grid's may not keep."""
    v = seen[S].sum(axis=0)
    return int(sum(_lack(v - seen[c], need) for c in S))


def _greedy(seen, first, n):
    """Greedy on saturated coverage from a forced first camera."""
    S, count = [first], seen[first].astype(int)
    for _ in range(n - 1):
        gain = np.minimum(count[None] + seen, Rig.MIN_VIEWS).sum(axis=(1, 2))
        gain[S] = -1
        best = int(np.argmax(gain))
        S.append(best)
        count = count + seen[best]
    return S


def _swap(seen, S, rule):
    """1-swaps, every candidate at once, on the shortfall alone, or with
    `rule` on (deficit, shortfall)."""
    S = list(S)
    big = seen.shape[1] * seen.shape[2] * Rig.MIN_VIEWS + 1
    cur = (deficit(seen, S) * big if rule else 0) + int(_short(seen[S].sum(axis=0)))
    improved = True
    while improved and cur > 0:
        improved = False
        for i in range(len(S)):
            base = seen[S].sum(axis=0) - seen[S[i]]
            new = base[None] + seen                                   # (C, P, K) with c in i's place
            key = _short(new)
            if rule:
                lack = _lack(base)                                    # c itself lost
                for j, t in enumerate(S):
                    if j != i:
                        lack = lack + _lack(new - seen[t][None])
                key = key + lack * big
            key[S] = cur + 1
            c = int(np.argmin(key))
            if key[c] < cur:
                S[i], cur, improved = c, int(key[c]), True
    return S


def _score(cov, S, at):
    """(deficit, shortfall, worst rotation deg, worst position mm, views):
    the sigmas the worst over poses and over each camera lost in turn."""
    seen = cov.seen
    views = seen[S].sum(axis=0)
    short = int(_short(views))
    total = cov.info[S].sum(axis=0)
    Rb = np.array([RS.body_R(a, b) for a, b in cov.poses])
    rot, pos = 0.0, 0.0
    for c in S:
        r_, p_ = pose_sigma(total - cov.info[c], at, Rb)
        rot, pos = max(rot, float(r_.max())), max(pos, float(p_.max()))
    return deficit(seen, S), short, rot, pos, views


def solve(cov, n=Rig.CAMERAS, at=None, log=None):
    """The n cameras.  From every candidate as the first: greedy on the
    coverage, then 1-swaps on the shortfall; ranked on (deficit, shortfall,
    the margin's deficit).  Only if no set meets the rule are the best
    swapped again on (deficit, shortfall).  Of the sets that tie, the one
    with the best worst sigma with a camera lost."""
    C = len(cov.cams)
    seen = cov.seen.astype(np.int16)

    def key(S):
        S = list(S)
        return (deficit(seen, S), int(_short(seen[S].sum(axis=0))), deficit(seen, S, Rig.MIN_CHECKED + 1))

    sets = {tuple(sorted(_swap(seen, _greedy(seen, f, n), False))) for f in range(C)}
    keys = {S: key(S) for S in sets}
    lo = min(keys.values())
    msgs = ["%d starts, %d distinct sets: least (deficit, shortfall, margin deficit) %s" % (C, len(sets), lo)]
    if lo[0] > 0:
        tops = [S for S in sets if keys[S][:2] == lo[:2]]
        sets = {tuple(sorted(_swap(seen, S, True))) for S in tops}
        keys = {S: key(S) for S in sets}
        lo = min(keys.values())
        msgs.append("no set met the rule; %d swapped on it: least %s" % (len(tops), lo))
    tied = [list(S) for S in sets if keys[S] == lo]
    scored = [(_score(cov, S, at), S) for S in tied]
    cur, S = min(scored, key=lambda x: x[0][2])
    msgs.append("%d tied; the best: worst rotation %.4f deg, position %.4f mm with a camera lost"
                % (len(tied), cur[2], cur[3]))
    if log:
        for m_ in msgs:
            log(m_)
    cams = [Camera(("cam%d" % k), cov.cams[c].lens, cov.cams[c].R, cov.cams[c].t, dict(cov.cams[c].meta))
            for k, c in enumerate(S)]
    return Placement(S, cams, cur[4], cur[0], cur[1], cur[2], cur[3], msgs)


def bat_points(rig):
    """The bat's ends and middle in the body frame: where the pose's
    position error is judged."""
    return [np.array([rig.x0 - rig.x_mid, 0.0, 0.0]), np.zeros(3), np.array([rig.x1 - rig.x_mid, 0.0, 0.0])]


def training(n):
    """(a, b) deg: the poses the cameras are placed against -- the n x n
    grid and the same grid shifted half a step on both axes, a quincunx,
    so no pose of the gimbal is more than a third of a step from one.
    Whether a ball is seen changes over a few degrees (a stem or the ring's
    own tube crossing it), so the placement is judged afterwards on poses it
    never saw (judge)."""
    return np.concatenate([RS.grid(n), RS.grid(n, 0.5)])


def place(rig, site, n_dirs=400, n_grid=12, log=None):
    """(Placement, Coverage): the eight cameras for this rig in its site."""
    cands = candidates(rig, site, n_dirs)
    cov = coverage(rig, site, cands, training(n_grid))
    return solve(cov, at=bat_points(rig), log=log), cov


def budget():
    """(deg, mm): the rig's share of the per-unit calibration's mounting
    and lever-arm targets (Rig.SHARE of imu_fusion_sim.CALIBRATED), 1 sigma."""
    from .. import imu_fusion_sim as study
    return Rig.SHARE * study.CALIBRATED.mount_deg, Rig.SHARE * study.CALIBRATED.lever_mm


@dataclass
class Judged:
    """How a set of cameras covers a set of poses."""
    deficit: int               # checked markers short of MIN_CHECKED, over poses and each camera lost
    least_checked: int         # the fewest checked markers at any pose with any one camera lost
    short_frac: float          # (pose, marker) pairs seen by fewer than MIN_VIEWS
    fewest: int                # the fewest cameras on any marker at any pose
    worst_rot: float           # deg, 1 sigma, any one camera lost
    worst_pos: float           # mm, 1 sigma, at the bat's ends and middle, any one camera lost
    poses: int


def judge(rig, site, cams, poses):
    """Judged: the cameras' coverage of `poses`, ray-cast afresh."""
    cov = coverage(rig, site, cams, poses)
    S = list(range(len(cams)))
    seen = cov.seen.astype(np.int16)
    v = seen.sum(axis=0)
    least = min(int(((v - seen[c]) >= Rig.MIN_VIEWS).sum(axis=1).min()) for c in S)
    _, _, rot, pos, _ = _score(cov, S, bat_points(rig))
    return Judged(deficit(seen, S), least, float((v < Rig.MIN_VIEWS).mean()), int(v.min()), rot, pos, len(poses))


def fits(rig, site, cams, j):
    """The rule as Fits, from a Judged."""
    b_rot, b_pos = budget()
    return [Fit("every camera stands where a camera may", -float(np.sum(~RS.allowed(site, [c.centre for c in cams]))),
                "%d cameras" % len(cams)),
            Fit("any one camera lost, every pose rests on %d markers seen by %d (fewest)"
                % (Rig.MIN_CHECKED, Rig.MIN_VIEWS), float(j.least_checked - Rig.MIN_CHECKED),
                "%d poses, %.1f%% of marker sightings short of %d" % (j.poses, 100.0 * j.short_frac, Rig.MIN_VIEWS)),
            Fit("any one camera lost, the clamp's rotation is fixed to its share (deg, 1 sigma)",
                b_rot - j.worst_rot, "%.4f of %.3f" % (j.worst_rot, b_rot)),
            Fit("any one camera lost, the bat's ends are fixed to their share (mm, 1 sigma)",
                b_pos - j.worst_pos, "%.4f of %.3f" % (j.worst_pos, b_pos))]
