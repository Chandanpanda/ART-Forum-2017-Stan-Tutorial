"""The game app's fusion as the factory iPhone runs it: the stand-in the
plan names for the real app (imu_fusion_sim.py's fusion), fed only what a
phone has -- the bat's stream over Bluetooth, the band finder's centroids,
the bat's record and its design.  Production, numpy only.

    apply          the record applied to the codes: rate and force in the
                   bat's frame
    moved          where the bat turned, from the gyro alone
    seen           what the band finder's centroids sit over: each band's
                   silhouette's middle, off the axis on the bat's section
    solve_pose     the bat's position at rest, from its bands
    anchor_cov     what that pose's errors are, from the camera's noise and
                   the bands' tolerances
    dead_reckon    the IMU carried on from the anchor
    fuse           all of it: where the sweet spot is as the swing ends

THE APP'S WORLD is the camera's level frame (camera.py): X right, Y along
the line of sight, Z up -- the accuracy study's.  The bat's frame is the
record's: the clamp's (record.FRAME_CLAMP), the bat's axis along its x.

AS THE STUDY DOES IT (imu_fusion_sim.run, its stance anchor):
  1  the record applied, its table inverted: in the chip's axes
     f = H^-1 (y_a - b_a(T)) and w = (I + E_g)^-1 (y_g - G f - b_g(T)),
     each bias at the die's temperature from its packet; U turns both into
     the bat's frame;
  2  the stance: every sample before the gyro first leaves its rest -- a
     chi-square on its three axes, the rest its median over the stance's
     first imu_fusion_sim.LEVEL_S (the game's stance holds the bat still
     there), Module.Z_SIGMAS's tail shared over every sample by Bonferroni -- less
     the output filter's memory.  Its gyro mean is this power-up's bias
     (the Earth's turn in it), its force's mean levels the bat (the
     shortest turn taking it to up) and states gravity;
  3  the bat's heading about up, the one turn the force cannot see, is
     given.  Two bands 4 m away cannot fix it: only their perspective
     tells heading from depth, by a few millimetres of foreshortening, and
     each band's print moves them more (a solve for it too put the
     heading's 1 sigma at 4.5 degrees on the kid's bat, and failed to
     converge on 23 of 40 simulated bats).  The game learns it at home;
     the factory test states it from the rig (test.heading);
  4  the frames inside the stance, their times put on the samples' by the
     bat's clock (board.sync, as the game recovers it) with the clock's
     Z-sigma error, half an exposure and a readout to spare.  Their mean
     centroids fix the bat's position by Gauss-Newton on the two bands'
     design places, weighted by what moves each centroid -- its noise and
     the bands' print and place (anchor_cov): four numbers, three unknowns.
     A centroid sits over its band's silhouette, not its centre (seen):
     the section is a rounded triangle, so seen side-on with a flat face
     edge-on to the lens (station 4's case: the bat face down, the camera
     level) the silhouette's middle is R/4 off the axis, 7 mm on the
     kid's bat, 2.6 px at 4 m, which demo_phone's frames showed and
     check_swing renders; square on to a face it is on the axis;
  5  from the stance's last sample the IMU alone, from rest, to the first
     sample past the gyro's last movement and the filter's memory.
"""
from dataclasses import dataclass
from math import erfc, sqrt

import numpy as np

from ..spec import Imu, Knit, Module, Phone, Quality
from ..rig.calib import chi2_quantile
from ..imu import part as PT, kin as K, record as R
from ..board import sync as SY
from .. import imu_fusion_sim as study

TAIL = 0.5 * erfc(Module.Z_SIGMAS / sqrt(2.0))
UP = np.array([0.0, 0.0, 1.0])


class FusionError(ValueError):
    """What the app cannot fuse: a gap in the stream, no stance, no frame."""


@dataclass(frozen=True)
class BatGeom:
    """What the app knows of the bat from its design, m in the bat's frame:
    the two bands' centres (in the band finder's order), the sweet spot,
    the axis along which a band's print and place move it, and the corners
    of the section the bands are printed round (product.Shape.corners, on
    the axis's plane through the origin)."""
    bands: np.ndarray          # (2, 3)
    ss: np.ndarray             # (3,)
    axis: np.ndarray           # (3,) unit, pommel to toe
    hull: np.ndarray           # (k, 3)


@dataclass
class Fix:
    ss: np.ndarray             # (3,) m: the sweet spot as the swing ends, the camera's level frame
    ss0: np.ndarray            # (3,) m: and at the anchor
    psi: float                 # rad: the bat's heading about up at the anchor, as given
    g: float                   # m/s^2: gravity as the stance reads it
    cov: np.ndarray            # (4, 4): the anchor's sweet spot (m) and heading (rad), anchor_cov
    n_frames: int
    k: tuple                   # (the anchor, the swing's end): sample counters
    clock: object              # sync.ClockMap


# ================================================================ THE RECORD
def apply(rec, codes, temp):
    """(w rad/s, f m/s^2), each (n, 3) in the record's frame, from the
    codes (n, 6: accel then gyro) and each sample's die-temperature code."""
    if rec.frame != R.FRAME_CLAMP:
        raise FusionError("the record states its mounting in frame %d; the app reads the clamp's" % rec.frame)
    gl, al = PT.lsb()
    codes = np.asarray(codes, float)
    dT = (PT.T_REF + np.asarray(temp, float) * Imu.TEMP_LSB - rec.T_cal)[:, None]
    ba = (rec.acc_bias.astype(float) + rec.acc_tc.astype(float) * dT) * PT.G0 / 1000.0
    f = np.linalg.solve(np.eye(3) + rec.acc_scale(), (codes[:, :3] * al - ba).T).T
    Eg = rec.gyro_E.astype(float).reshape(3, 3) / R.PPM
    G = rec.gyro_G.astype(float).reshape(3, 3) * PT.D2R / PT.G0
    bg = (rec.gyro_bias.astype(float) + rec.gyro_tc.astype(float) * dT) * PT.D2R
    w = np.linalg.solve(np.eye(3) + Eg, (codes[:, 3:] * gl - f @ G.T - bg).T).T
    U = rec.rotation()
    return w @ U.T, f @ U.T


def sample_dt(rec):
    """s between samples, true time: the record's clock."""
    return (1.0 + float(rec.rho[0]) / R.PPM) / Imu.ODR


# ================================================================ STILL OR NOT
def gyro_sd():
    """rad/s, 1 sigma of one gyro sample about its rest: the datasheet's
    white noise through the output filter (part.window_noise) and the
    rounding to a code."""
    lg, _ = PT.lsb()
    return sqrt((Imu.GYRO_ND * PT.D2R) ** 2 * PT.window_noise(1)[0] + lg * lg / 12.0)


def moved(w, n_rest):
    """(first, last) index of the samples whose rate is off the rest by
    more than the noise allows anywhere in the recording, or None: the rest
    is the median of the first n_rest samples, which the stance holds still."""
    w = np.asarray(w, float)
    s2 = np.sum((w - np.median(w[:n_rest], 0)) ** 2, 1) / gyro_sd() ** 2
    hit = np.flatnonzero(s2 > chi2_quantile(3, TAIL / len(w)))
    return (int(hit[0]), int(hit[-1])) if len(hit) else None


def level(f):
    """The shortest turn taking the direction of f (the force at rest,
    which points up) to up."""
    u = np.asarray(f, float) / np.linalg.norm(f)
    ax = np.cross(u, UP)
    s, c = float(np.linalg.norm(ax)), float(u @ UP)
    if s == 0.0:
        return np.eye(3) if c > 0.0 else K.rot([1.0, 0.0, 0.0], [np.pi])[0]
    return K.exp_so3(ax / s * np.arctan2(s, c))


# ================================================================ THE POSE
def heading_of(Rb):
    """rad: the heading about up of the bat's frame Rb (level frame): the
    turn about up left once level(Rb^T up) is taken out."""
    Rz = Rb @ level(Rb.T @ UP).T
    return float(np.arctan2(Rz[1, 0], Rz[0, 0]))


def attitude(f, psi):
    """The bat's frame in the level frame from its force at rest f (bat
    frame) and its heading psi."""
    return K.rot(UP, [psi])[0] @ level(f)


def seen(p, Rb, geom, bands=None):
    """(2, 3) m, level frame: what the band finder's centroids sit over with
    the bat's frame at p and Rb (bands, the bat frame's band centres, the
    design's if None) -- each band's silhouette's middle, its centre only
    on a section symmetric about the axis.  From the camera (the level
    frame's origin) a band at c spans its section's extent along
    e = axis x c, square to the axis and the line of sight; the section is
    a convex polygon grown by its rounding, so that extent is the
    corners', widened alike on each side, and its middle lies
    (min + max)/2 of the corners along e off the axis.  Orthographic
    across the section: its depth at 4 m moves the middle by microns."""
    c = p + (geom.bands if bands is None else bands) @ Rb.T
    V = geom.hull @ Rb.T
    e = np.cross(Rb @ geom.axis, c)
    n = np.linalg.norm(e, axis=1, keepdims=True)
    if np.any(n <= 1e-9 * np.linalg.norm(c, axis=1, keepdims=True)):
        raise FusionError("a band is seen end on")
    e = e / n
    ve = e @ V.T
    return c + 0.5 * (ve.min(1) + ve.max(1))[:, None] * e


def _jac(p, Rb, geom, view):
    """(J (4, 3) d px / d p, Jb (4, 2) d px / d each band's place along the
    axis), by central differences: steps far under the solution's own
    spread and far over float rounding at 4 m."""
    def proj(pp, bands):
        return view.project_level(seen(pp, Rb, geom, bands)).ravel()
    h = 1e-6
    J = np.stack([(proj(p + h * e, geom.bands) - proj(p - h * e, geom.bands)) / (2.0 * h) for e in np.eye(3)], 1)
    Jb = np.empty((4, 2))
    for i in (0, 1):
        d = np.zeros((2, 3))
        d[i] = h * geom.axis
        Jb[:, i] = (proj(p, geom.bands + d) - proj(p, geom.bands - d)) / (2.0 * h)
    return J, Jb


def band_cov():
    """(2, 2) m^2: the bands' places along the axis about the design: each
    printed within Knit.PRINT_TOL on its own, the sleeve placed within
    Knit.PLACE_TOL moving both, each bound read as a rectangular
    distribution (GUM, JCGM 100:2008, 4.3.7: a / sqrt(3))."""
    p, s = Knit.PRINT_TOL / 1000.0 / sqrt(3.0), Knit.PLACE_TOL / 1000.0 / sqrt(3.0)
    return p * p * np.eye(2) + s * s * np.ones((2, 2))


def _weights(p, Rb, geom, view, n):
    """(J, Sm): the solve's Jacobian and its four centroids' covariance:
    each Phone.SIGMA_PX, averaged over n frames, and the bands' places
    (band_cov) through the projection."""
    J, Jb = _jac(p, Rb, geom, view)
    return J, Phone.SIGMA_PX ** 2 / n * np.eye(4) + Jb @ band_cov() @ Jb.T


def anchor_cov(p, Rb, geom, view, n, heading_sd):
    """(4, 4): the covariance of the sweet spot (m, level frame) the pose
    solve gives from n still frames at the bat's position p and frame Rb,
    and of the given heading (heading_sd rad, uncorrelated)."""
    J, Sm = _weights(p, Rb, geom, view, n)
    Wi = np.linalg.inv(Sm)
    out = np.zeros((4, 4))
    out[:3, :3] = np.linalg.inv(J.T @ Wi @ J)
    out[3, 3] = heading_sd ** 2
    return out


def solve_pose(uv, n, Rb, geom, view):
    """(p m, its (3, 3) covariance): the position of the bat's frame, at
    attitude Rb, that best puts its two bands at the mean centroids uv
    (2, 2) of n frames (Gauss-Newton, weighted by _weights), started on the
    centroids' mid ray at the depth their separation gives."""
    uv = np.asarray(uv, float).reshape(2, 2)
    d = Rb @ (geom.bands[1] - geom.bands[0])
    across = float(np.hypot(d[0], d[2]))
    sep = float(np.linalg.norm(uv[1] - uv[0]))
    if across <= 0.0 or sep <= 0.0:
        raise FusionError("the bands do not stand apart across the line of sight")
    y0 = view.f * across / sep
    um = uv.mean(0)
    p = np.array([(um[0] - view.w / 2.0) / view.f, 1.0, -(um[1] - view.h / 2.0) / view.f]) * y0 \
        - Rb @ geom.bands.mean(0)
    for _ in range(50):
        J, Sm = _weights(p, Rb, geom, view, n)
        Wi = np.linalg.inv(Sm)
        r = uv.ravel() - view.project_level(seen(p, Rb, geom)).ravel()
        try:
            step = np.linalg.solve(J.T @ Wi @ J, J.T @ Wi @ r)
        except np.linalg.LinAlgError:
            raise FusionError("the bands do not fix the bat's position")
        p = p + step
        if step @ (J.T @ Wi @ J) @ step <= 1e-12:     # a millionth of its own 1 sigma
            break
    else:
        raise FusionError("the pose solve did not converge")
    J, Sm = _weights(p, Rb, geom, view, n)
    return p, np.linalg.inv(J.T @ np.linalg.inv(Sm) @ J)


# ================================================================ THE SWING
def dead_reckon(p, R0, w, f, dt, g):
    """(p, v, R): the IMU from rest at p with the bat's frame at R0 (level
    frame) after the samples (w, f) (bat frame), each held over its
    interval dt: imu_fusion_sim.run's integrator (the force turned by the
    attitude at the interval's middle, gravity g straight down)."""
    p = np.array(p, float)
    v = np.zeros(3)
    Rk = np.array(R0, float)
    gw = -g * UP
    for wk, fk in zip(np.asarray(w, float), np.asarray(f, float)):
        a = Rk @ K.exp_so3(wk * dt / 2.0) @ fk + gw
        p = p + v * dt + 0.5 * a * dt * dt
        v = v + a * dt
        Rk = Rk @ K.exp_so3(wk * dt)
    return p, v, Rk


def samples(packets):
    """(k, codes (n, 6), temp (n,)) of every sample since the bat's last
    reset, counter order, each with its packet's die temperature.
    FusionError if any is missing."""
    k_last = [p.k0 + len(p.codes) - 1 for p in packets]
    kl, runs = SY.unwrap(k_last)
    pk = list(zip(kl, packets))[runs[-1]:]
    k = np.concatenate([kk - len(p.codes) + 1 + np.arange(len(p.codes)) for kk, p in pk])
    codes = np.concatenate([p.codes for _, p in pk])
    temp = np.concatenate([np.full(len(p.codes), p.temp) for _, p in pk])
    o = np.argsort(k, kind="stable")
    k, codes, temp = k[o], codes[o], temp[o]
    gaps = int(np.sum(np.diff(k) - 1)) if len(k) > 1 else 0
    if gaps or len(np.unique(k)) != len(k):
        raise FusionError("%d samples missing from the stream" % gaps)
    return k, codes, temp


def fuse(packets, frames, rec, geom, view, psi, heading_sd=0.0, central="phone"):
    """Fix: the sweet spot where the bat comes to rest after the swing, from
    the stream's packets (board.hal.Packet, in arrival order), the frames
    (phone.hal.Frame), the bat's record and its design, given its heading
    psi (rad, known to heading_sd)."""
    if len(packets) < 2:
        raise FusionError("%d stream packets" % len(packets))
    cm = SY.recover(packets, central)
    k, codes, temp = samples(packets)
    w, f = apply(rec, codes, temp)
    n_rest = int(round(study.LEVEL_S / sample_dt(rec)))
    if len(w) <= n_rest:
        raise FusionError("%d samples, not a stance's %d" % (len(w), n_rest))
    mv = moved(w, n_rest)
    if mv is None:
        raise FusionError("the gyro never left its rest")
    mem = len(PT.impulse(Imu.ODR))
    s, e = mv[0] - mem, mv[1] + mem
    if s < 1 or e >= len(k):
        raise FusionError("the recording does not hold a stance and the swing's end")
    bias = w[:s + 1].mean(0)
    fs = f[:s + 1].mean(0)
    g = float(np.linalg.norm(fs))
    Rb = attitude(fs, psi)
    spare = Module.Z_SIGMAS * Quality.SYNC_S + Phone.EXPOSURE_S / 2.0 + Phone.READOUT_S
    t_lo, t_hi = float(cm.at(k[0])) + spare, float(cm.at(k[s])) - spare
    still = [fr.uv for fr in frames if fr.uv is not None and t_lo <= fr.t <= t_hi]
    if not still:
        raise FusionError("no frame found both bands inside the stance")
    p0, _ = solve_pose(np.mean(still, 0), len(still), Rb, geom, view)
    r = rec.lever_clamp() / 1000.0
    p, _, Re = dead_reckon(p0 + Rb @ r, Rb, w[s:e] - bias, f[s:e], sample_dt(rec), g)
    return Fix(p + Re @ (geom.ss - r), p0 + Rb @ geom.ss, float(psi), g,
               anchor_cov(p0, Rb, geom, view, len(still), heading_sd), len(still), (int(k[s]), int(k[e])), cm)
