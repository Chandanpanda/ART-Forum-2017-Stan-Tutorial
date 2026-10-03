"""What the datasheet says about the board's IMU, and the stream it sends:
codes and their scale, the output filter, the noise of a window's mean.
Production code (the planner and the fit read it); the part's own errors
are the simulation's (model.py).
"""
from dataclasses import dataclass
from functools import lru_cache
from math import sqrt, pi, tan, erf, ceil, log, exp

import numpy as np

from ..spec import Imu

G0 = 9.80665                   # m/s^2 a g: the datasheet's unit (ISO 80000-3)
D2R = pi / 180.0
T_REF = 25.0                   # degC, where a datasheet states its offsets


def lsb():
    """(gyro rad/s, accel m/s^2) a code."""
    full = 2.0 ** (Imu.BITS - 1)
    return Imu.GYRO_FS * D2R / full, Imu.ACC_FS * G0 / full


def unrailed():
    """The fraction of full scale an output stays under to read no code
    at either end: the station sees only codes, so a value that rounds to
    the top one, 2^(BITS-1) - 1, is a rail to it."""
    full = 2.0 ** (Imu.BITS - 1)
    return (full - 1.5) / full


@dataclass
class Stream:
    """What the board sends: per sample its counter, six codes, the die
    temperature's code, and whether any axis read a code at its rail."""
    k: np.ndarray              # (n,) int
    acc: np.ndarray            # (n, 3) int16 codes
    gyro: np.ndarray           # (n, 3) int16 codes
    temp: np.ndarray           # (n,) int codes, Imu.TEMP_LSB a code about 25 C
    rail: np.ndarray           # (n,) bool


def physical(stream):
    """(gyro rad/s, accel m/s^2, die temperature degC) from the codes, at
    the datasheet's own scale: what a procedure reads."""
    lg, la = lsb()
    return stream.gyro * lg, stream.acc * la, T_REF + stream.temp * Imu.TEMP_LSB


# ================================================================ FILTER
def butterworth2(fc, fs):
    """(b, a) of the 2nd-order Butterworth low-pass at fc, sampled at fs,
    by the bilinear transform with the cut-off prewarped."""
    K = tan(pi * fc / fs)
    n = 1.0 / (1.0 + sqrt(2.0) * K + K * K)
    b0 = K * K * n
    return np.array([b0, 2.0 * b0, b0]), np.array([1.0, 2.0 * (K * K - 1.0) * n, (1.0 - sqrt(2.0) * K + K * K) * n])


def impulse(fs, tail=1e-9):
    """The output path's impulse response at fs, until the rest of it
    holds under `tail` of its whole."""
    return _impulse(float(fs), float(tail)).copy()


@lru_cache(maxsize=64)
def _impulse(fs, tail):
    if Imu.FILTER_ORDER != 2:
        raise NotImplementedError("only the 2nd-order output filter is modelled")
    b, a = butterworth2(Imu.FILTER_BW, fs)
    p = np.roots(a)
    # once the input is past, the response is its two modes, each its value
    # at the last sample times p^j: from the last two samples, y1 = m1 + m2
    # and y2 = m1 / p1 + m2 / p2, and the rest sums to under
    # sum |m_i| |p_i| / (1 - |p_i|)
    Vi = np.linalg.inv(np.array([[1.0, 1.0], [1.0 / p[0], 1.0 / p[1]]]))
    h = []
    x1 = x2 = y1 = y2 = 0.0
    total = 0.0
    k = 0
    while True:
        x0 = 1.0 if k == 0 else 0.0
        y0 = b[0] * x0 + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        h.append(y0)
        total += abs(y0)
        x2, x1, y2, y1 = x1, x0, y1, y0
        k += 1
        if k > 8:
            m = Vi @ np.array([y1, y2])
            if float(np.sum(np.abs(m) * np.abs(p) / (1.0 - np.abs(p)))) < tail * total:
                break
    return np.array(h)


@lru_cache(maxsize=None)
def _autocov(oversample=32):
    """(lags, r): the output noise's autocovariance at whole output samples,
    per unit white density squared (nd = 1 per rtHz): the filter run at
    `oversample` x the output rate, as the part runs it faster than it
    reads it out."""
    fs = oversample * Imu.ODR
    h = impulse(fs)
    r_full = np.correlate(h, h, "full") * fs / 2.0          # lags -L..L at fs, nd = 1
    L = len(h) - 1
    m = L // oversample
    r = r_full[L - m * oversample: L + m * oversample + 1: oversample]
    return np.arange(-m, m + 1), r


@lru_cache(maxsize=4096)
def window_noise(W, oversample=32):
    """(variance of a W-sample mean, correlation of two neighbouring means)
    of the output's noise per unit white density squared."""
    lags, r = _autocov(oversample)

    def cov(shift):
        d = lags - shift * W
        return float(np.sum(np.maximum(0, W - np.abs(d)) * r)) / (W * W)
    c0 = cov(0)
    return c0, cov(1) / c0


# ================================================================ CODES
# The output is rounded to a whole code after the filter.  Where the noise
# is well under a code -- the gyro's is 0.38 of one at the full scale a
# swing needs -- it does not dither the rounding: a mean of codes is half
# again noisier where the true value falls halfway between two codes than
# where it falls on one, so the noise a still measures need not be the
# noise a spin's window carries.  A mean's variance is therefore that of
# Gaussian noise with the filter's correlation, rounded: at lag 0 summed
# over the codes, at the other lags from the rounding error's Fourier
# series, round(x) - x = sum_m (-1)^m sin(2 pi m x) / (pi m), with Price's
# theorem for its covariance with the noise (Widrow & Kollar 2008,
# Quantization Noise, ch. 4 and 9).  Units are codes: s is the noise's sd
# per sample, nu where the true value falls, as a fraction of a code.
_erf = np.vectorize(erf, otypes=[float])


@lru_cache(maxsize=None)
def _rho(oversample=32):
    """The output noise's correlation at lags 0, 1, ... output samples."""
    lags, r = _autocov(oversample)
    m = len(lags) // 2
    return r[m:] / r[m]


def _lag0(s, nu):
    """(len(nu),) the variance of one sample's code."""
    span = int(ceil(12.0 * s)) + 2
    q = np.arange(-span, span + 2, dtype=float)
    z = (q[None, :] - nu[:, None]) / (s * sqrt(2.0))
    P = 0.5 * (_erf(z + 0.5 / (s * sqrt(2.0))) - _erf(z - 0.5 / (s * sqrt(2.0))))
    m1 = P @ q
    return P @ (q * q) - m1 * m1


def _lag(s, nu, rho):
    """(len(nu),) the covariance of two samples' codes whose noise
    correlates by rho, |rho| < 1: the series to where its terms fall
    under e^-30 of its first, at most _M_MAX of them."""
    M = min(int(ceil(sqrt(60.0 / (s * s * (1.0 - abs(rho)))) / (2.0 * pi))) + 1, _M_MAX)
    m = np.arange(1, M + 1)
    a = 2.0 * pi * m
    sg = (-1.0) ** m
    g1 = 2.0 * (sg * np.exp(-0.5 * a * a * s * s)) @ np.cos(np.outer(a, nu))
    A, B = a[:, None], a[None, :]
    L = -0.5 * s * s * (A * A + B * B)
    base = np.exp(L)
    e1 = np.exp(-s * s * (0.5 * (A - B) ** 2 + (1.0 - rho) * A * B)) - base
    # base (e^x - 1), x = -rho s^2 A B: by expm1 while x is small, and in
    # one exponent past that, where e^x alone would overflow (rho < 0, a
    # noise of several codes) though the product never exceeds 1
    x = -rho * s * s * A * B
    big = x > 30.0
    e2 = np.where(big, np.exp(L + np.where(big, x, 0.0)) - base, base * np.expm1(np.where(big, 0.0, x)))
    C = np.outer(sg / (pi * m), sg / (pi * m))
    # gathered by m - l and m + l: then each place costs 2M cosines, not M^2
    d = (np.subtract.outer(m, m) + M).ravel()
    E1 = np.bincount(d, (C * e1).ravel(), 2 * M + 1)
    E2 = np.bincount(np.add.outer(m, m).ravel(), (C * e2).ravel(), 2 * M + 1)
    k = np.arange(2 * M + 1)
    cee = 0.5 * (E1 @ np.cos(2.0 * pi * np.outer(k - M, nu)) - E2 @ np.cos(2.0 * pi * np.outer(k, nu)))
    return s * s * rho * (1.0 + 2.0 * g1) + cee


_M_MAX = 1024                  # terms: a noise of 1/600 of a code still converges
_SERIES_TAIL = 60.0            # the series' terms fall as e^-(2 pi s)^2 (1 - |rho|) / 2: past this, none is left


def dithered(s):
    """Whether noise s (codes) dithers the rounding fully: every term of
    the series under e^-30, so a code is the noise plus a white twelfth."""
    rho = _rho()
    return (2.0 * pi * s) ** 2 * (1.0 - float(np.max(np.abs(rho[1:])))) >= _SERIES_TAIL


def code_var(s, nu, n, run=False):
    """(len(nu),) codes^2: the variance of a mean of n successive codes,
    noise s codes a sample, the true value nu of a code past a code.  With
    run, what each such mean carries into a long run of them: its share of
    the variance of the run's mean, times the number of means -- its own
    variance and its covariances with the means after it, the filter's
    whole memory.  That is what a window weighs in anything that changes
    slowly across windows (a spin's harmonics), and it is more than its own
    variance by the neighbours' correlation (window_noise)."""
    nu = np.mod(np.atleast_1d(np.asarray(nu, float)), 1.0)
    if s <= 0.0:
        return np.zeros(len(nu))     # no noise: every code is the same code
    rho = _rho()
    L = len(rho) - 1 if run else min(int(n) - 1, len(rho) - 1)
    # pairs of samples dl apart: n - dl inside one mean; n a mean, in a run
    pairs = (lambda dl: float(n)) if run else (lambda dl: float(n - dl))
    if dithered(s):
        tot = n * (s * s + 1.0 / 12.0) + 2.0 * s * s * sum(pairs(d) * rho[d] for d in range(1, L + 1))
        return np.full(len(nu), tot / (float(n) * n))
    tot = n * _lag0(s, nu)
    for dl in range(1, L + 1):
        if rho[dl] != 0.0:
            tot = tot + 2.0 * pairs(dl) * _lag(s, nu, rho[dl])
    return tot / (float(n) * n)


@lru_cache(maxsize=4096)
def code_var_range(s, n, grid=64, run=False):
    """(least, mean, most) of code_var over where the true value falls."""
    v = code_var(s, (np.arange(grid) + 0.5) / grid, n, run)
    return float(v.min()), float(v.mean()), float(v.max())


def code_noise(sse, w, nu, n, s0):
    """s, codes a sample: the noise whose rounded n-sample means, each at
    its own place nu (its true value, codes), give the scatter sse (codes^2,
    summed) when each counts by its redundancy w (a regression's 1 - h:
    what the mean it is scattered about leaves it) -- solved on log s by
    the Illinois method from s0."""
    w, nu = (np.asarray(x, float).ravel() for x in (w, nu))
    target = float(np.sum(sse))
    if target <= 0.0:
        return 0.0                   # the codes never moved: a channel stuck, or no noise to see

    if not np.isfinite(target) or not np.all(np.isfinite(nu)):
        raise FloatingPointError("the scatter is not finite")

    def f(ls):
        return log(float(np.sum(w * code_var(exp(ls), nu, n)))) - log(target)
    a, b = log(s0) - 0.5, log(s0) + 0.5
    fa, fb = f(a), f(b)
    for _ in range(40):
        if fa <= 0.0:
            break
        a -= 1.0
        fa = f(a)
    for _ in range(40):
        if fb >= 0.0:
            break
        b += 1.0
        fb = f(b)
    if fa > 0.0 or fb < 0.0:
        return exp(a if fa > 0.0 else b)
    side = 0
    for _ in range(60):
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        if abs(fc) < 1e-10 or b - a < 1e-10:
            return exp(c)
        if fc * fb > 0.0:
            b, fb = c, fc
            if side == -1:
                fa *= 0.5
            side = -1
        else:
            a, fa = c, fc
            if side == 1:
                fb *= 0.5
            side = 1
    return exp(c)
