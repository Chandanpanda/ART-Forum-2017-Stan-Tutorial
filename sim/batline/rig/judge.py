"""How the rig's results compare with the scene's truth: for the checks
only, never the procedures (they cannot see the truth; these can).

THE GAUGE.  A self-calibration's world is "where the gimbal said its
gauge sample was" (calib.py): right relative to itself, one rigid transform
from the rig's frame.  The body frame is the bar's certificate, which is
the truth to its 0.005 mm.  So every comparison first finds that one
transform -- Kabsch over the bar's and markers' balls at every calibration
sample, as solved against as they truly were -- and then compares poses
the calibration never saw.  Six numbers fitted to thousands of points
absorb nothing of a held-out error.
"""
from dataclasses import dataclass

import numpy as np

from .lens import rvec
from .pose import kabsch


def gauge(cal, scene):
    """(R, t): the calibration's world -> the rig frame."""
    est, true = [], []
    P = cal.sol.P
    for s, (a, b) in enumerate(cal.poses):
        Rt = cal.sol.poses[s]
        Rb, tb = scene.truth_pose(a, b)
        est.append(P @ Rt[:, :3].T + Rt[:, 3])
        true.append(P @ Rb.T + tb)
    return kabsch(np.vstack(est), np.vstack(true))


def reprojection(cal, scene, G):
    """Per camera (rms, max) px: where the calibrated camera puts every ball
    of every calibration sample against where the camera as built puts it
    -- the lens and the pose together, over what each camera looked at."""
    Rg, tg = G
    out = []
    P = cal.sol.P
    for c, (est, true) in enumerate(zip(cal.sol.cams, scene.truth_cams())):
        d = []
        for s, (a, b) in enumerate(cal.poses):
            Rb, tb = scene.truth_pose(a, b)
            X = P @ Rb.T + tb                                  # rig frame
            Xw = (X - tg) @ Rg                                 # the calibration's world
            u1, v1, z1 = true.project(X, scene.r)
            u2, v2, _ = est.project(Xw, scene.r)
            ok = (z1 > scene.r) & true.lens.inside(u1, v1)
            d.append(np.hypot(u1 - u2, v1 - v2)[ok])
        d = np.concatenate(d)
        out.append((float(np.sqrt(np.mean(d * d))), float(d.max())))
    return out


def intrinsics_z(cal, scene):
    """(C, 9): each camera's lens parameter error over its own stated
    sigma (calib.Solution.cov), the lens being free of the world's gauge."""
    out = []
    for c, (est, true) in enumerate(zip(cal.sol.cams, scene.truth_cams())):
        sd = cal.sol.cov["cam"][c, 6:]
        out.append((est.lens.params() - true.lens.params()) / np.where(sd > 0, sd, np.inf))
    return np.array(out)


@dataclass
class PoseErr:
    rot: np.ndarray            # (n, 3) deg: the turn from fitted to true, per world axis
    pos: np.ndarray            # (n, m, 3) mm: each bat point's error, per world axis
    z: np.ndarray              # (n, 6): the error over the fit's own covariance, whitened

    def rms_rot(self):
        """deg: per axis, over every pose and axis -- the 1 sigma the
        budget is stated in."""
        return float(np.sqrt(np.mean(self.rot ** 2)))

    def rms_pos(self):
        return float(np.sqrt(np.mean(self.pos ** 2)))


def pose_errors(fits, poses, scene, G, at):
    """PoseErr of fitted poses (pose.PoseFit, in the calibration's world)
    against the truth at gimbal commands `poses`, bat points `at` (body
    frame).  The rotation error is the fit's own parametrisation, R_true =
    exp([dphi]x) R_fit in world axes, so z whitens it by the fit's
    covariance: a fit that knows its own error has z ~ N(0, 1)."""
    Rg, tg = G
    rot, pos, z = [], [], []
    for f, (a, b) in zip(fits, poses):
        Rb, tb = scene.truth_pose(a, b)
        Re, te = Rg @ f.R, Rg @ f.t + tg
        dphi = rvec(Rb @ Re.T)                                 # rig axes
        rot.append(np.degrees(dphi))
        pos.append([(Rb @ p + tb) - (Re @ p + te) for p in at])
        # the same error in the fit's own world and parameters (pose.py: R <-
        # exp([dphi]x) R in world axes, t <- t + dt), where its covariance is
        Rw, tw = Rg.T @ Rb, Rg.T @ (tb - tg)
        e = np.r_[rvec(Rw @ f.R.T), tw - f.t]
        L = np.linalg.cholesky(f.cov)
        z.append(np.linalg.solve(L, e))
    return PoseErr(np.array(rot), np.array(pos), np.array(z))
