"""The board's IMU in simulation: its errors, its warm-up, its clock, its
filter and its codes.  TRUTH: only the simulation (sim.py) and the checks
import this; the station's procedures see the codes it emits and nothing
else (the scan in check_imu holds them to that).

What a part is (spec.Imu, 1 sigma part to part):

  y_a = (I + Ea) f + ba + ca (T - 25 C) + ba_on                  accelerometer
  y_g = (I + Eg) w + G f + bg + cg (T - 25 C) + bg_on            gyro

in the chip's own axes, f the specific force and w the angular rate at its
sensing centre.  Ea and Eg are full 3x3: scale on the diagonal, cross-axis
off it (the die's misalignment included, so not symmetric).  ba_on and
bg_on are redrawn at every power-up (power_up).  T is the die's
temperature, which warms from the room after power-up (die_temperature).

How a sample is made, in the order the part makes it (sense):
  1. the signal above, at an internal rate f_int, with white noise of the
     part's density (variance nd^2 f_int / 2 a sample: nd is one-sided);
  2. the output path's low-pass (spec.Imu.FILTER_ORDER, FILTER_BW), a
     Butterworth by the bilinear transform at f_int, applied as its
     impulse response truncated where what is left is under 1e-9 of it;
  3. read at the part's own sample instants, t_k = t_first + k T (1 + eps):
     its oscillator is off by eps (ODR_TOL);
  4. cut to signed 16-bit codes at FS / 32768 a code, clipped to the codes
     there are, -32768..32767, each clipped sample flagged.
The die temperature rides in every sample, in its own codes.

f_int is not chosen: internal_rate() doubles it from the output rate until
doubling again moves no sample by more than the quantiser's own noise.
"""
from dataclasses import dataclass
from math import sqrt

import numpy as np

from ..spec import Imu
from .kin import exp_so3
from .part import G0, D2R, T_REF, lsb, impulse, Stream

@dataclass
class ImuErrors:
    """One part's errors, SI, in its own axes."""
    Ea: np.ndarray
    Eg: np.ndarray
    ba: np.ndarray             # m/s^2 at 25 C
    bg: np.ndarray             # rad/s at 25 C
    G: np.ndarray              # rad/s per m/s^2
    ca: np.ndarray             # m/s^2 per degC
    cg: np.ndarray             # rad/s per degC
    eps: float                 # its oscillator's rate error


@dataclass
class PowerUp:
    """One power-up: its offsets, when it came on (station time, s), the
    room, and where in its sample period the stream starts."""
    ba_on: np.ndarray
    bg_on: np.ndarray
    t_on: float
    T_amb: float


def draw_errors(rng, scale=1.0):
    """A part, from the datasheet's spreads (times `scale`)."""
    def mat(s_scale, s_cross):
        M = np.diag(rng.normal(0.0, s_scale, 3))
        off = ~np.eye(3, dtype=bool)
        M[off] = rng.normal(0.0, s_cross, 6)
        return M * scale
    return ImuErrors(Ea=mat(Imu.ACC_SCALE, Imu.ACC_CROSS), Eg=mat(Imu.GYRO_SCALE, Imu.GYRO_CROSS),
                     ba=rng.normal(0.0, Imu.ACC_BIAS * G0, 3) * scale,
                     bg=rng.normal(0.0, Imu.GYRO_BIAS * D2R, 3) * scale,
                     G=rng.normal(0.0, Imu.GYRO_GSENS * D2R / G0, (3, 3)) * scale,
                     ca=rng.normal(0.0, Imu.ACC_TC * G0, 3) * scale,
                     cg=rng.normal(0.0, Imu.GYRO_TC * D2R, 3) * scale,
                     eps=float(rng.normal(0.0, Imu.ODR_TOL)) * scale)


def power_up(rng, t_on, T_amb):
    return PowerUp(ba_on=rng.normal(0.0, Imu.ACC_POWERUP * G0, 3), bg_on=rng.normal(0.0, Imu.GYRO_POWERUP * D2R, 3),
                   t_on=float(t_on), T_amb=float(T_amb))


def die_temperature(pw, t):
    """degC at station times t: the room plus the board's own warming since
    power-up, first order."""
    tau = np.maximum(np.asarray(t, float) - pw.t_on, 0.0)
    return pw.T_amb + Imu.SELF_HEAT * (1.0 - np.exp(-tau / Imu.HEAT_TAU))


def biases(err, pw, T):
    """(ba, bg) at die temperature T, this power-up."""
    return (err.ba + err.ca * (T - T_REF) + pw.ba_on, err.bg + err.cg * (T - T_REF) + pw.bg_on)


def fir(x, h):
    """Causal filtering of each column of x (n, m) by h, through the FFT,
    started from rest at the first sample's value."""
    n = len(x)
    L = 1 << int(np.ceil(np.log2(n + len(h))))
    x0 = x[:1]
    X = np.fft.rfft(x - x0, L, axis=0)
    H = np.fft.rfft(h, L)
    return np.fft.irfft(X * H[:, None], L, axis=0)[:n] + x0 * h.sum()


# ================================================================ SAMPLES
def sense(err, pw, t_int, w_imu, f_imu, t_first, n, rng, noise=True, quantise=True):
    """The board's stream: n samples from its first, at station time
    t_first, of the true rate w_imu and specific force f_imu (chip axes,
    SI) given on the uniform internal grid t_int (station time).  Returns
    (Stream, t_k, the output before its codes, SI -- its noise in it unless
    noise=False)."""
    T = np.asarray(t_int, float)
    dt = T[1] - T[0]
    fs = 1.0 / dt
    temp = die_temperature(pw, T)
    ba, bg = biases(err, pw, temp[:, None])
    ya = f_imu + f_imu @ err.Ea.T + ba
    yg = w_imu + w_imu @ err.Eg.T + f_imu @ err.G.T + bg
    y = np.concatenate([yg, ya], axis=1)
    if noise:
        sd = np.array([Imu.GYRO_ND * D2R] * 3 + [Imu.ACC_ND * G0] * 3) * sqrt(fs / 2.0)
        y = y + rng.normal(0.0, 1.0, y.shape) * sd
    h = impulse(fs)
    y = fir(y, h)
    t_k = t_first + np.arange(n) * (1.0 + err.eps) / Imu.ODR
    if t_k[0] < T[0] or t_k[-1] > T[-1]:
        raise ValueError("the samples run outside the internal grid")
    u = (t_k - T[0]) / dt
    i = np.minimum(np.floor(u).astype(int), len(T) - 2)
    fr = (u - i)[:, None]
    ys = y[i] * (1.0 - fr) + y[i + 1] * fr
    Tk = die_temperature(pw, t_k)
    if noise:
        Tk = Tk + rng.normal(0.0, Imu.TEMP_NOISE, n)
    lg, la = lsb()
    scale = np.array([lg] * 3 + [la] * 3)
    lo, hi = -2 ** (Imu.BITS - 1), 2 ** (Imu.BITS - 1) - 1
    if quantise:
        # the rail as a station can see it: a code at either end, which a
        # value just inside reads as too
        c = np.clip(np.round(ys / scale), lo, hi).astype(np.int32)
        rail = np.any((c == lo) | (c == hi), axis=1)
        tc = np.round((Tk - T_REF) / Imu.TEMP_LSB).astype(np.int32)
    else:
        c = ys / scale
        rail = np.any((c < lo - 0.5) | (c > hi + 0.5), axis=1)
        tc = (Tk - T_REF) / Imu.TEMP_LSB
    return Stream(np.arange(n), c[:, 3:], c[:, :3], tc, rail), t_k, ys


def internal_rate(probe, start=None, max_rate=None):
    """The internal rate f_int: from 4 x the output rate, doubled until a
    doubling moves no noiseless output sample by more than the quantiser's
    noise, LSB / sqrt(12).  `probe(fs)` returns the noiseless outputs (n, 6)
    in codes at the same instants for internal rate fs."""
    fs = 4.0 * Imu.ODR if start is None else start
    top = 256.0 * Imu.ODR if max_rate is None else max_rate
    prev = probe(fs)
    while fs < top:
        nxt = probe(2.0 * fs)
        if np.max(np.abs(nxt - prev)) <= 1.0 / sqrt(12.0):
            return fs, float(np.max(np.abs(nxt - prev)))
        fs, prev = 2.0 * fs, nxt
    raise ValueError("the internal rate did not converge below %.0f Hz" % top)


def mount_matrix(rot_deg):
    return exp_so3(np.radians(np.asarray(rot_deg, float)))
