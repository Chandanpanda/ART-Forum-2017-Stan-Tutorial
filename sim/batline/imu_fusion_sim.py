#!/usr/bin/env python3
"""
IMU bat + phone camera: how well can the game know where the bat is at contact?

Monte Carlo over consumer-MEMS error terms on a parametric bat swing.

Model
-----
* World: X right (as the camera sees it), Y away from the camera along the
  ball's line, Z up. The phone sits at the TV, so the ball travels along +Y
  and the camera's weak axis (depth) is the ball's own axis.
* The camera is used only as an anchor. It supplies the bat's position and
  velocity at a slow moment (end of the stance, or the top of the backlift),
  with errors derived from camera geometry (camera_anchor below). From there
  the IMU alone carries the estimate to contact. Blurred mid-swing frames are
  NOT used, so this is pessimistic for the fusion design.
* Orientation is never corrected by the camera after the stance. It is
  levelled from the (biased) accelerometer while the bat is still, given a
  heading error (the camera's job), then integrated from the corrupted gyro.

Reported at contact, for the bat's sweet spot:
  lateral = sqrt(X^2 + Z^2): across the ball's line. Decides hit, miss, edge.
  depth   = |Y|: along the ball's line. Only shifts the contact time by
            depth / closing speed (1 cm at 35 m/s is 0.3 ms).
plus the bat-axis direction error and the face-angle (roll) error.

Everything here is an estimate from a model, not a measurement. The sensor
figures are typical consumer-MEMS datasheet values from memory; change them
in ImuGrade and re-run.

    python3 imu_fusion_sim.py            # full report (about a minute)
    python3 imu_fusion_sim.py --quick    # fewer trials
"""
import argparse
from dataclasses import dataclass, replace

import numpy as np

G = 9.80665
GW = np.array([0.0, 0.0, -G])
D2R = np.pi / 180.0


# ------------------------------------------------------------------ SO(3)
def skew(v):
    v = np.asarray(v, float)
    O = np.zeros(v.shape[:-1] + (3, 3))
    O[..., 0, 1] = -v[..., 2]
    O[..., 0, 2] = v[..., 1]
    O[..., 1, 0] = v[..., 2]
    O[..., 1, 2] = -v[..., 0]
    O[..., 2, 0] = -v[..., 1]
    O[..., 2, 1] = v[..., 0]
    return O


def exp_so3(phi):
    phi = np.asarray(phi, float)
    th = np.linalg.norm(phi, axis=-1)[..., None, None]
    K = skew(phi)
    small = th < 1e-9
    ths = np.where(small, 1.0, th)
    A = np.where(small, 1.0 - th ** 2 / 6, np.sin(ths) / ths)
    B = np.where(small, 0.5 - th ** 2 / 24, (1 - np.cos(ths)) / ths ** 2)
    return np.eye(3) + A * K + B * (K @ K)


def log_so3(R):
    tr = np.trace(R, axis1=-2, axis2=-1)
    th = np.arccos(np.clip((tr - 1) / 2, -1, 1))
    w = np.stack([R[..., 2, 1] - R[..., 1, 2],
                  R[..., 0, 2] - R[..., 2, 0],
                  R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.sin(th)
    fac = np.where(th < 1e-8, 0.5, th / (2 * np.where(s == 0, 1, s)))
    return w * fac[..., None]


def rot_axis(axis, ang):
    return exp_so3(np.asarray(ang, float)[..., None] * np.asarray(axis, float))


def T(R):
    return np.swapaxes(R, -1, -2)


# ------------------------------------------------------------------ swing
@dataclass
class Swing:
    """A bat swinging through a plane, hands riding along with it.

    theta is the bat angle in the swing plane: 0 = pointing straight down
    (contact), +150 deg = top of backlift (up and back), -120 deg = end of
    follow-through. Stance and backlift use a minimum-jerk profile; the
    downswing accelerates as sin^2 to peak angular speed at contact.
    """
    name: str
    Td: float                     # top of backlift -> contact (s)
    plane_tilt_deg: float = 0.0   # 0: vertical plane along the ball line (drive); 70: nearly flat (pull)
    plane_yaw_deg: float = 5.0
    Ts: float = 0.6               # still stance (s)
    Tb: float = 0.45              # backlift (s)
    th_stance_deg: float = 20.0
    th_top_deg: float = 150.0
    th_follow_deg: float = -120.0
    roll_k1: float = 0.20         # face roll (rad) per rad of swing angle
    roll_k2: float = -0.03
    tilt_amp_deg: float = 8.0     # bat leaves the swing plane by up to this
    hand_a: float = 0.25          # hands travel along the ball line (m)
    hand_b: float = 0.30          # hands rise (m)
    hands_contact: tuple = (0.15, 4.0, 0.85)

    def __post_init__(self):
        self.th_s = self.th_stance_deg * D2R
        self.th_top = self.th_top_deg * D2R
        self.th_fol = self.th_follow_deg * D2R
        self.om_max = 2 * self.th_top / self.Td
        self.Tf = -2 * self.th_fol / self.om_max
        self.t_top = self.Ts + self.Tb
        self.t_c = self.t_top + self.Td
        self.t_end = self.t_c + self.Tf
        Rz = rot_axis([0, 0, 1], self.plane_yaw_deg * D2R)
        e_a = Rz @ np.array([0.0, 1.0, 0.0])
        e_b = Rz @ rot_axis([0, 1, 0], self.plane_tilt_deg * D2R) @ np.array([0.0, 0.0, 1.0])
        self.P = np.stack([e_a, e_b], 1)
        self.Hc = np.array(self.hands_contact, float)

    def theta(self, t):
        t = np.asarray(t, float)
        th = np.full_like(t, self.th_s)
        m = (t >= self.Ts) & (t < self.t_top)
        tau = (t[m] - self.Ts) / self.Tb
        th[m] = self.th_s + (self.th_top - self.th_s) * (10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5)
        m = (t >= self.t_top) & (t < self.t_c)
        tau = (t[m] - self.t_top) / self.Td
        th[m] = self.th_top - self.om_max * self.Td * (tau / 2 - np.sin(np.pi * tau) / (2 * np.pi))
        m = t >= self.t_c
        tau = np.minimum((t[m] - self.t_c) / self.Tf, 1.0)
        th[m] = -self.om_max * self.Tf * (tau / 2 + np.sin(np.pi * tau) / (2 * np.pi))
        return th

    def pose(self, t):
        th = self.theta(t)
        P = self.P
        b = (P @ np.stack([np.sin(th), -np.cos(th)])).T          # bat axis, handle -> toe
        f = (P @ np.stack([-np.cos(th), -np.sin(th)])).T         # face normal (face leads)
        n = np.cross(b, f)
        Rp = np.stack([b, f, n], -1)
        psi = self.tilt_amp_deg * D2R * np.sin(th)
        phi = self.roll_k1 * th + self.roll_k2 * th ** 2
        R = Rp @ rot_axis([0, 1, 0], psi) @ rot_axis([1, 0, 0], phi)
        H = self.Hc + (P @ np.stack([self.hand_a * np.sin(th / 2),
                                     self.hand_b * (1 - np.cos(th / 2))])).T
        return R, H


# bat-frame offsets from the hands (x along the bat, handle -> toe)
L_IMU = np.array([-0.10, 0.0, 0.0])   # IMU in the handle cap, 10 cm above the hands
L_SS = np.array([0.50, 0.0, 0.0])     # sweet spot, 60 cm from the IMU
R_MOUNT = exp_so3(np.array([0.6, -0.4, 0.9]) * D2R)   # how the IMU board actually sits

DT_F = 1e-4      # truth grid
SUB = 10         # truth steps per IMU sample
DT = DT_F * SUB  # 1 kHz IMU


class Truth:
    """Ground-truth trajectory and the ideal 1 kHz IMU samples it produces."""

    def __init__(self, sw):
        self.sw = sw
        n_f = int(np.ceil((sw.t_end + 0.05) / DT_F))
        n_f = (n_f // SUB + 1) * SUB + 1
        tf = (np.arange(n_f + 2) - 1) * DT_F            # one pad step either side
        Rb, H = sw.pose(tf)
        Ri = Rb @ R_MOUNT
        pi = H + np.einsum('nij,j->ni', Rb, L_IMU)
        ps = H + np.einsum('nij,j->ni', Rb, L_SS)
        w_int = log_so3(T(Ri[1:-1]) @ Ri[2:]) / DT_F     # body rate over [j, j+1]
        acc = (pi[2:] - 2 * pi[1:-1] + pi[:-2]) / DT_F ** 2
        f_node = np.einsum('nji,nj->ni', Ri[1:-1], acc - GW)   # specific force, body
        K = (n_f - 1) // SUB
        self.w = w_int[:K * SUB].reshape(K, SUB, 3).mean(1)
        wts = np.ones(SUB + 1)
        wts[0] = wts[-1] = 0.5
        wts /= SUB
        idx = np.arange(K)[:, None] * SUB + np.arange(SUB + 1)[None, :]
        self.f = np.einsum('ksd,s->kd', f_node[idx], wts)
        node = np.arange(K + 1) * SUB
        self.t = node * DT_F
        self.Ri = Ri[1:-1][node]
        self.Rb = Rb[1:-1][node]
        self.pi = pi[1:-1][node]
        self.ps = ps[1:-1][node]
        self.vi = ((pi[2:] - pi[:-2]) / (2 * DT_F))[node]
        self.vs = ((ps[2:] - ps[:-2]) / (2 * DT_F))[node]
        self.acc_peak_g = np.linalg.norm(f_node, axis=1).max() / G
        self.gyro_peak_dps = np.abs(w_int).max() / D2R
        self.tip_speed = np.linalg.norm(self.vs[int(round(sw.t_c / DT))])
        self.k_s = int(round(sw.Ts / DT))
        self.k_top = int(round(sw.t_top / DT))
        self.k_c = int(round(sw.t_c / DT))


# ------------------------------------------------------------------ sensor
@dataclass
class ImuGrade:
    name: str
    gyro_fs_dps: float = 2000.0
    acc_fs_g: float = 16.0
    gyro_nd: float = 0.0028       # dps/rtHz
    acc_nd: float = 70e-6         # g/rtHz
    bw: float = 250.0             # Hz, noise bandwidth at 1 kHz ODR
    gyro_bias: float = 0.05       # dps, 1 sigma, residual after the filter's at-rest estimate
    gyro_scale: float = 0.003     # 1 sigma (datasheet tolerance about +-0.5%)
    gyro_cross: float = 0.006     # 1 sigma per off-diagonal (datasheet about +-1.25%)
    gyro_nl: float = 0.001        # +- fraction of FS, not calibrated
    gyro_gsens: float = 0.05      # dps/g, 1 sigma per element, not calibrated
    acc_bias: float = 0.015       # g, 1 sigma
    acc_scale: float = 0.003
    acc_cross: float = 0.005
    acc_nl: float = 0.002         # +- fraction of FS, not calibrated
    mount_deg: float = 1.0        # 1 sigma per axis: belief vs how the board sits in the bat
    lever_mm: float = 2.0         # 1 sigma per axis: belief vs where the IMU sits


DATASHEET = ImuGrade('datasheet, no per-unit calibration')
CALIBRATED = replace(DATASHEET, name='calibrated per unit',
                     gyro_bias=0.02, gyro_scale=0.0005, gyro_cross=0.0005,
                     acc_bias=0.002, acc_scale=0.0005, acc_cross=0.0005,
                     mount_deg=0.2, lever_mm=0.5)
CAL_WIDE = replace(CALIBRATED, name='calibrated, 32 g / 4000 dps part',
                   gyro_fs_dps=4000.0, acc_fs_g=32.0)


# ------------------------------------------------------------------ camera anchor
@dataclass
class Anchor:
    name: str
    at: str               # 'stance' or 'top'
    pos_lat: float        # m, 1 sigma per lateral axis (X, Z)
    pos_dep: float        # m, 1 sigma along Y
    vel_lat: float        # m/s
    vel_dep: float        # m/s
    yaw_deg: float = 1.0  # heading of the IMU relative to the camera, 1 sigma
    sync_ms: float = 1.0  # IMU clock vs game clock, 1 sigma


def camera_anchor(at, f_px=1500.0, Z=4.0, L=0.45, sig_px=0.5, floor_lat=0.002,
                  floor_dep_frac=0.0037, sig_px_moving=1.5, n_vel=9, T_vel=0.3):
    """Anchor errors from phone-camera geometry.

    f_px 1500: iPhone wide camera at 1080p (about 65 deg). Two colour bands
    0.45 m apart on the bat. sig_px: band centroid noise per frame on a still
    bat. floor_lat: lens/centroid bias that averaging cannot remove.
    floor_dep_frac: focal-length (0.3%) and band-spacing (0.2%) tolerance.
    Velocity at the top comes from the ~9 blurred backlift frames, bridged by
    the IMU's (accurate, short-term) relative motion.
    """
    lat1 = Z * sig_px / f_px
    dep1 = Z ** 2 * np.sqrt(2) * sig_px / (f_px * L)
    if at == 'stance':
        n = 15
        vlat = vdep = 0.002    # bat at rest, and the IMU knows it
    else:
        n = 3
        k = sig_px_moving / sig_px
        vlat = lat1 * k * np.sqrt(12) / (T_vel * np.sqrt(n_vel))
        vdep = dep1 * k * np.sqrt(12) / (T_vel * np.sqrt(n_vel))
    lat = np.hypot(lat1 / np.sqrt(n), floor_lat)
    dep = np.hypot(dep1 / np.sqrt(n), floor_dep_frac * Z)
    name = 'top of backlift' if at == 'top' else 'end of stance'
    return Anchor(name, at, lat, dep, vlat, vdep)


# ------------------------------------------------------------------ Monte Carlo
SOURCES = ['anchor_pos', 'anchor_vel', 'yaw', 'gyro_bias', 'gyro_scale', 'gyro_cross',
           'gyro_nl', 'gyro_gsens', 'gyro_noise', 'acc_bias', 'acc_scale', 'acc_cross',
           'acc_nl', 'acc_noise', 'range', 'mount', 'lever', 'sync']


def run(tr, grade, anchor, N=1500, seed=1, only=None, record_every=0.05):
    rng = np.random.default_rng(seed)

    def on(s):
        return only is None or s in only

    k_lev = 300                               # level over the last 0.3 s of stance
    k0, k1 = tr.k_s - k_lev, tr.k_c
    w_t, f_t = tr.w[k0:k1], tr.f[k0:k1]
    I3 = np.eye(3)

    def tmat(scale, cross, s_name, c_name):
        M = np.repeat(I3[None], N, 0)
        if on(s_name):
            M = M + rng.normal(0, scale, (N, 3))[:, :, None] * I3
        if on(c_name):
            C = rng.normal(0, cross, (N, 3, 3))
            C[:, [0, 1, 2], [0, 1, 2]] = 0
            M = M + C
        return M

    # gyro
    fs_w = grade.gyro_fs_dps * D2R
    w_m = np.einsum('nij,kj->nki', tmat(grade.gyro_scale, grade.gyro_cross, 'gyro_scale', 'gyro_cross'), w_t)
    if on('gyro_nl'):
        a = rng.uniform(-grade.gyro_nl, grade.gyro_nl, (N, 1, 3))
        x = np.clip(w_t / fs_w, -1, 1)[None]
        w_m = w_m + (a / 0.4) * fs_w * (x ** 3 - 0.6 * x)
    if on('gyro_gsens'):
        Gs = rng.normal(0, grade.gyro_gsens * D2R / G, (N, 3, 3))
        w_m = w_m + np.einsum('nij,kj->nki', Gs, f_t)
    if on('gyro_bias'):
        w_m = w_m + rng.normal(0, grade.gyro_bias * D2R, (N, 1, 3))
    if on('gyro_noise'):
        w_m = w_m + rng.normal(0, grade.gyro_nd * np.sqrt(grade.bw) * D2R, w_m.shape)
    if on('range'):
        lsb = fs_w / 32768
        w_m = np.round(np.clip(w_m, -fs_w, fs_w) / lsb) * lsb

    # accelerometer
    fs_a = grade.acc_fs_g * G
    f_m = np.einsum('nij,kj->nki', tmat(grade.acc_scale, grade.acc_cross, 'acc_scale', 'acc_cross'), f_t)
    if on('acc_nl'):
        a = rng.uniform(-grade.acc_nl, grade.acc_nl, (N, 1, 3))
        x = np.clip(f_t / fs_a, -1, 1)[None]
        f_m = f_m + (a / 0.4) * fs_a * (x ** 3 - 0.6 * x)
    if on('acc_bias'):
        f_m = f_m + rng.normal(0, grade.acc_bias * G, (N, 1, 3))
    if on('acc_noise'):
        f_m = f_m + rng.normal(0, grade.acc_nd * np.sqrt(grade.bw) * G, f_m.shape)
    if on('range'):
        lsb = fs_a / 32768
        f_m = np.round(np.clip(f_m, -fs_a, fs_a) / lsb) * lsb

    # what the filter believes about how the IMU sits in the bat
    dm = rng.normal(0, grade.mount_deg * D2R, (N, 3)) if on('mount') else np.zeros((N, 3))
    Rm_est = R_MOUNT @ exp_so3(dm)
    dl = rng.normal(0, grade.lever_mm / 1000, (N, 3)) if on('lever') else np.zeros((N, 3))
    lev_b = L_SS - (L_IMU + dl)               # IMU -> sweet spot, bat frame, as believed

    def bat_est(Ri_est):
        return Ri_est @ T(Rm_est)

    # level from the still stance, heading from the camera
    f_avg = f_m[:, :k_lev].mean(1)
    u = np.einsum('ij,nj->ni', tr.Ri[tr.k_s], f_avg)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    z = np.array([0.0, 0.0, 1.0])
    ax = np.cross(u, z)
    s = np.linalg.norm(ax, axis=1, keepdims=True)
    ang = np.arctan2(s, u @ z[:, None])
    Q = exp_so3(np.where(s > 1e-12, ax / np.where(s > 1e-12, s, 1) * ang, 0.0))
    yaw = rng.normal(0, anchor.yaw_deg * D2R, N) if on('yaw') else np.zeros(N)
    R = rot_axis([0, 0, 1], yaw) @ Q @ tr.Ri[tr.k_s]

    def camera_reset(k, R_imu):
        """Position/velocity from the camera's view of the bands at sample k."""
        dp = np.zeros((N, 3))
        dv = np.zeros((N, 3))
        if on('anchor_pos'):
            dp = rng.normal(0, 1, (N, 3)) * [anchor.pos_lat, anchor.pos_dep, anchor.pos_lat]
        if on('anchor_vel'):
            dv = rng.normal(0, 1, (N, 3)) * [anchor.vel_lat, anchor.vel_dep, anchor.vel_lat]
        lev_w = np.einsum('nij,nj->ni', bat_est(R_imu), lev_b)
        w_w = np.einsum('nij,nj->ni', R_imu, w_m[:, k - k0])
        return tr.ps[k] + dp - lev_w, tr.vs[k] + dv - np.cross(w_w, lev_w)

    p, v = camera_reset(tr.k_s, R)
    rec_every = int(round(record_every / DT))
    k_anchor = tr.k_top if anchor.at == 'top' else tr.k_s
    growth = []

    for k in range(tr.k_s, tr.k_c):
        if anchor.at == 'top' and k == tr.k_top:
            p, v = camera_reset(k, R)
        if k >= k_anchor and (k - k_anchor) % rec_every == 0:
            pss = p + np.einsum('nij,nj->ni', bat_est(R), lev_b)
            e = pss - tr.ps[k]
            growth.append(((k - k_anchor) * DT, np.hypot(e[:, 0], e[:, 2])))
        wk, fk = w_m[:, k - k0], f_m[:, k - k0]
        dR = exp_so3(wk * DT)
        Rmid = R @ exp_so3(wk * DT / 2)
        a = np.einsum('nij,nj->ni', Rmid, fk) + GW
        p = p + v * DT + 0.5 * a * DT * DT
        v = v + a * DT
        R = R @ dR

    Rb_e = bat_est(R)
    e = p + np.einsum('nij,nj->ni', Rb_e, lev_b) - tr.ps[tr.k_c]
    if on('sync'):
        e = e + tr.vs[tr.k_c] * rng.normal(0, anchor.sync_ms / 1000, (N, 1))
    Rb_t = tr.Rb[tr.k_c]
    rv = log_so3(T(Rb_t)[None] @ Rb_e)
    axis_err = np.arccos(np.clip(Rb_e[:, :, 0] @ Rb_t[:, 0], -1, 1)) / D2R
    growth.append(((tr.k_c - k_anchor) * DT, np.hypot(e[:, 0], e[:, 2])))
    return dict(lat=np.hypot(e[:, 0], e[:, 2]) * 1000, dep=np.abs(e[:, 1]) * 1000,
                axis=axis_err, roll=np.abs(rv[:, 0]) / D2R, growth=growth)


def pct(x, q):
    return float(np.percentile(x, q))


# ------------------------------------------------------------------ report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    N = 400 if args.quick else 1500

    fast_hands = dict(hand_a=0.5, hand_b=0.6)     # hands about 6.5 m/s at contact for Td 0.2
    swings = [Swing('kid, straight drive', Td=0.30),
              Swing('adult, straight drive', Td=0.20, **fast_hands),
              Swing('hard, straight drive', Td=0.13, **fast_hands),
              Swing('adult, pull (flat swing)', Td=0.20, plane_tilt_deg=70.0, **fast_hands)]
    truths = {sw.name: Truth(sw) for sw in swings}
    anchors = [camera_anchor('top'), camera_anchor('stance')]

    out = []
    w = out.append
    w('# IMU bat + phone camera: error at contact (Monte Carlo)\n')
    w(f'{N} trials per row. Errors are for the bat sweet spot, 60 cm from the IMU. '
      'p50 = median, p95 = 95th percentile. Estimates from a model, not measurements.\n')

    # sanity: perfect sensors, perfect anchor
    tr = truths['adult, straight drive']
    clean = run(tr, CALIBRATED, anchors[1], N=8, only=set())
    w(f'Integrator check (no errors at all, 0.65 s from stance): '
      f'{clean["lat"].max():.3f} mm lateral, {clean["dep"].max():.3f} mm depth.\n')

    w('## Swings\n')
    w('| swing | peak gyro (dps) | peak accel at IMU (g) | sweet-spot speed at contact (m/s) |')
    w('|---|---|---|---|')
    for sw in swings:
        t = truths[sw.name]
        w(f'| {sw.name} | {t.gyro_peak_dps:.0f} | {t.acc_peak_g:.1f} | {t.tip_speed:.1f} |')
    w('')

    w('## Camera anchor assumed (derived in camera_anchor)\n')
    w('| anchor | position lateral / depth (mm) | velocity lateral / depth (mm/s) | heading (deg) | clock sync (ms) |')
    w('|---|---|---|---|---|')
    for an in anchors:
        w(f'| {an.name} | {an.pos_lat*1e3:.1f} / {an.pos_dep*1e3:.1f} | '
          f'{an.vel_lat*1e3:.0f} / {an.vel_dep*1e3:.0f} | {an.yaw_deg:.1f} | {an.sync_ms:.1f} |')
    w('')

    w('## Error at contact\n')
    w('| swing | IMU | anchor | lateral p50 / p95 (mm) | depth p50 / p95 (mm) | bat axis p95 (deg) | face angle p95 (deg) |')
    w('|---|---|---|---|---|---|---|')
    for sw in swings:
        t = truths[sw.name]
        for grade in (DATASHEET, CALIBRATED, CAL_WIDE):
            for an in anchors:
                r = run(t, grade, an, N=N)
                w(f'| {sw.name} | {grade.name} | {an.name} | {pct(r["lat"],50):.1f} / {pct(r["lat"],95):.1f} | '
                  f'{pct(r["dep"],50):.0f} / {pct(r["dep"],95):.0f} | {pct(r["axis"],95):.2f} | {pct(r["roll"],95):.2f} |')
    w('')

    tr = truths['adult, straight drive']
    BIAS_ONLY = replace(DATASHEET, name='biases estimated online, no per-unit calibration',
                        gyro_bias=0.02, acc_bias=0.003)
    WARM = replace(CALIBRATED, name='calibrated, gyro scale drifted 0.15% (warm)',
                   gyro_scale=0.0015, acc_bias=0.004)
    w('## Sensitivity: adult straight drive, anchored at end of stance\n')
    w('One assumption changed per row from the calibrated case.\n')
    w('| change | lateral p50 / p95 (mm) | face angle p95 (deg) |')
    w('|---|---|---|')
    st = anchors[1]
    cases = [('baseline: calibrated, heading 1.0 deg', CALIBRATED, st),
             ('heading known to 0.3 deg', CALIBRATED, replace(st, yaw_deg=0.3)),
             ('heading known to 0.5 deg', CALIBRATED, replace(st, yaw_deg=0.5)),
             ('heading known to 2.0 deg', CALIBRATED, replace(st, yaw_deg=2.0)),
             ('poorer camera: 1 px noise, 5 mm bias floor', CALIBRATED,
              replace(camera_anchor('stance', sig_px=1.0, floor_lat=0.005), yaw_deg=1.0)),
             ('IMU warmed up: gyro scale 0.15%, accel bias 4 mg', WARM, st),
             ('no per-unit calibration, biases estimated online', BIAS_ONLY, st),
             ('no per-unit calibration at all', DATASHEET, st)]
    for label, grade, an in cases:
        r = run(tr, grade, an, N=N)
        w(f'| {label} | {pct(r["lat"],50):.1f} / {pct(r["lat"],95):.1f} | {pct(r["roll"],95):.2f} |')
    w('')

    for grade, an in ((CALIBRATED, anchors[1]), (CALIBRATED, anchors[0]), (DATASHEET, anchors[1])):
        w(f'## Error budget: adult straight drive, {grade.name}, anchored at {an.name}\n')
        w('Each source alone; lateral error at contact.\n')
        w('| source | lateral p50 (mm) | lateral p95 (mm) |')
        w('|---|---|---|')
        rows = []
        for s in SOURCES:
            r = run(tr, grade, an, N=N, only={s})
            rows.append((pct(r['lat'], 95), s, pct(r['lat'], 50)))
        for p95, s, p50 in sorted(rows, reverse=True):
            w(f'| {s} | {p50:.1f} | {p95:.1f} |')
        w('')

    w('## How fast the IMU-only estimate degrades (adult drive, calibrated, anchored at end of stance)\n')
    r = run(tr, CALIBRATED, anchors[1], N=N)
    w('| time since last camera anchor (s) | lateral p50 (mm) | lateral p95 (mm) |')
    w('|---|---|---|')
    for tt, lat in r['growth']:
        w(f'| {tt:.2f} | {pct(lat,50)*1e3:.1f} | {pct(lat,95)*1e3:.1f} |')
    w('')

    # stereo console for comparison (analytic)
    f9, Z, sp = 914.0, 4.0, 0.15
    w('## Global-shutter stereo console, for comparison (analytic)\n')
    w(f'OV9281-class, f = {f9:.0f} px, {sp} px centroid noise, 4 m: lateral '
      f'{Z*sp/f9*1e3:.1f} mm; depth {Z**2*np.sqrt(2)*sp/(f9*0.6)*1e3:.1f} mm at 0.6 m baseline, '
      f'{Z**2*np.sqrt(2)*sp/(f9*1.0)*1e3:.1f} mm at 1 m.\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
