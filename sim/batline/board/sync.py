"""The bat's sample clock on a central's clock, from the stream alone.
Production, numpy only: the station, the factory iPhone's app and the game
all run it.

A packet leaves the bat once its last sample is taken, waits for the next
connection event, may be lost on air and resent, and crosses the central's
host stack, so each arrival is the sample's time plus a delay that is
never below a floor and usually above it.  Against the sample counter k,
the arrivals therefore lie on or above one line, t = t0 + rate (k - k0) +
floor: the line every point lies above that is highest at the points' mean
counter.  That is Moon, Skelly and Towsley's linear program (INFOCOM 1999),
and its optimum is the edge of the points' lower convex hull that spans
the mean (Zhang, Liu and Xia, INFOCOM 2002), found here by Andrew's
monotone chain in one sorted pass.  Unlike a least-squares fit it does not
average the delays, so the connection interval and the host's queueing,
which only ever add, do not bias it.

The floor itself -- radio, stack and host at their quickest -- cannot be
seen in arrivals; it is a property of the central model, measured once in
the lab (spec.Est.HOST_FLOOR_S) and subtracted.  The recovered clock is
then held to Quality.SYNC_S, the bat-to-phone offset the accuracy study
assumed.
"""
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, sqrt

import numpy as np

from ..spec import Imu, Est, Quality, Link
from . import hal as H

WRAP = 2 ** 32                 # the packet's counter is a u32


@dataclass(frozen=True)
class ClockMap:
    """Station time of sample k: t0 + rate (k - k0) - floor."""
    k0: float
    t0: float
    rate: float                # station seconds a count
    floor: float               # the central's delay floor, removed
    n: int                     # packets it rests on
    span: float                # s of station time they cover

    def at(self, k):
        return self.t0 + self.rate * (np.asarray(k, float) - self.k0) - self.floor

    @property
    def rho(self):
        """Station seconds per nominal sample interval: (1 + eps) of the
        bat's oscillator, times the station clock's own rate."""
        return self.rate * Imu.ODR


def unwrap(k):
    """The u32 counters as one rising sequence of ints.  A counter that
    falls by more than half the wrap is a wrap; one that falls by less is
    the bat having reset, and unwrap returns the start of each run."""
    k = np.asarray(k, dtype=np.int64)
    out = np.empty_like(k)
    runs = [0]
    base = 0
    for i in range(len(k)):
        if i and k[i] < k[i - 1]:
            if k[i - 1] - k[i] > WRAP // 2:
                base += WRAP
            else:
                runs.append(i)
                base = 0
        out[i] = k[i] + base
    return out, runs


def lower_hull(x, y):
    """Indices of the lower convex hull of points sorted by x (Andrew's
    monotone chain): a point stays only if the turn to it is to the left."""
    h = []
    for i in range(len(x)):
        while len(h) >= 2:
            a, b = h[-2], h[-1]
            if (x[b] - x[a]) * (y[i] - y[a]) - (y[b] - y[a]) * (x[i] - x[a]) <= 0.0:
                h.pop()
            else:
                break
        h.append(i)
    return h


def envelope(k, t):
    """(t0, rate) of the line under every (k, t) that is highest at the
    mean of k, with k0 that mean: Moon's linear program, solved exactly."""
    k = np.asarray(k, float)
    t = np.asarray(t, float)
    o = np.argsort(k, kind="stable")
    k, t = k[o], t[o]
    k0, tm = float(k.mean()), float(t.mean())
    x, y = k - k0, t - tm
    h = lower_hull(x, y)
    if len(h) < 2:
        raise ValueError("a clock needs packets at two counters at least")
    for a, b in zip(h, h[1:]):
        if x[a] <= 0.0 <= x[b]:
            rate = (y[b] - y[a]) / (x[b] - x[a])
            return tm + y[a] - rate * x[a], rate, k0
    raise ValueError("no hull edge spans the mean counter")


def recover(packets, central):
    """ClockMap from [Packet] (hal.Packet: arrival t, first counter k0, codes),
    using the run after the bat's last reset, with `central`'s floor removed."""
    if len(packets) < 2:
        raise ValueError("a clock needs two packets")
    k_last = [p.k0 + len(p.codes) - 1 for p in packets]
    k, runs = unwrap(k_last)
    i0 = runs[-1]
    k = k[i0:]
    t = np.array([p.t for p in packets[i0:]])
    t0, rate, k0 = envelope(k, t)
    return ClockMap(k0, t0, rate, floor(central), len(k), float(t.max() - t.min()))


def naive(packets, central):
    """The least-squares clock through the arrivals: what averaging the
    delays gives, for check_link's must-fail.  Not used by any procedure."""
    k = np.array([p.k0 + len(p.codes) - 1 for p in packets], float)
    t = np.array([p.t for p in packets])
    A = np.stack([np.ones_like(k), k - k.mean()], 1)
    (a, b), *_ = np.linalg.lstsq(A, t, rcond=None)
    return ClockMap(float(k.mean()), float(a), float(b), floor(central), len(k), float(t.max() - t.min()))


def link_floor():
    """s: the least a stream packet spends on air after its event opens --
    the central's empty packet, the bat's packet, a gap after each (the
    first slot of an event).  Part of the floor, from the facts; the host's
    part is the lab's (spec.Est.HOST_FLOOR_S)."""
    n = H.PACKET_HEAD.size + H.samples_per_packet() * H.SAMPLE.size
    return air_s(0) + air_s(n) + 2.0 * Link.T_IFS


def air_s(payload):
    return (payload + Link.LL_OVERHEAD) * 8.0 / Link.PHY_BPS


def floor(central):
    return Est.HOST_FLOOR_S[central] + link_floor()


@lru_cache(maxsize=256)
def envelope_rms(ci, central, n, trials=200):
    """s, the root mean square of the envelope's offset error on n packets,
    by Monte Carlo of the link as its facts describe it: packets made every
    samples_per_packet samples of an oscillator within Imu.ODR_TOL, each
    waiting for the next connection event at a random phase, resent an
    interval later for each loss (an exchange is lost if either of its two
    packets is, Link.PER apiece), then the host's exponential spread.  The waits are not independent -- the packet period and the
    interval beat -- and a sum of two delays has no density at zero, so no
    closed form fits; the model is cheap to run instead.  Seeded, so the
    plan is the same every time; check_link holds the simulated link to it."""
    rng = np.random.default_rng(12345 + n)
    spp = H.samples_per_packet()
    s = Est.HOST_SPREAD_S[central]
    k = np.arange(n) * spp
    err = np.empty(trials)
    for i in range(trials):
        dt = spp * (1.0 + rng.normal(0.0, Imu.ODR_TOL)) / Imu.ODR
        ready = np.arange(n) * dt
        wait = np.mod(rng.uniform(0.0, ci) - ready, ci)
        resent = rng.geometric((1.0 - Link.PER) ** 2, n) - 1
        t = ready + wait + resent * ci + rng.exponential(s, n)
        t0, rate, k0 = envelope(k, t)
        err[i] = t0 - k0 / spp * dt
    return float(np.sqrt(np.mean(err ** 2)))


def packets_needed(ci, central):
    """Packets a recovery rests on so that the envelope's own error and the
    floor's characterisation together stay inside Quality.SYNC_S: they are
    independent, so their mean squares add."""
    left = Quality.SYNC_S ** 2 - Est.HOST_FLOOR_SD ** 2
    if left <= 0.0:
        raise ValueError("the floor's characterisation alone spends the sync budget")
    n = 8
    while envelope_rms(ci, central, n) ** 2 > left:
        n *= 2
        if n > 1 << 16:
            raise ValueError("no stream this long recovers the clock to Quality.SYNC_S")
    lo, hi = n // 2, n
    while hi - lo > max(1, hi // 32):
        m = (lo + hi) // 2
        lo, hi = (lo, m) if envelope_rms(ci, central, m) ** 2 <= left else (m, hi)
    return hi


def slope_sd(ci, central, n=None):
    """A bound on the envelope's rate error, as a share: its offset error
    at either end of the span, against half the span between them.
    check_link holds the recovered rates to it."""
    n = packets_needed(ci, central) if n is None else n
    span = n * H.samples_per_packet() / Imu.ODR
    return 2.0 * envelope_rms(ci, central, n) / (span / 2.0)


def listen_s(ci, central):
    """s of streaming that brings packets_needed."""
    rate = Imu.ODR / H.samples_per_packet()
    return packets_needed(ci, central) / rate
