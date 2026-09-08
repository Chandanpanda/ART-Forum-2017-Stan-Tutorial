"""Axis motion: trapezoidal profiles for stepper-driven screws.

Both the schedule (how long will this take) and the Tier 1 backend (what
setpoint does the axis get this tick) read the SAME profile, which is how
the planned time and the simulated time cannot disagree about the
kinematics -- only about what the world did to them.  numpy only.
"""
from math import sqrt

import numpy as np


def trap_time(dist, vmax, amax):
    """Seconds to move `dist` from rest to rest under a trapezoid."""
    d = abs(float(dist))
    if d < 1e-9:
        return 0.0
    d_acc = vmax * vmax / amax           # distance to reach vmax and stop
    if d <= d_acc:                        # triangle
        return 2.0 * sqrt(d / amax)
    return d / vmax + vmax / amax


def trap_position(t, dist, vmax, amax):
    """Position along a rest-to-rest trapezoid at time t (0 at start)."""
    d = abs(float(dist))
    sgn = 1.0 if dist >= 0 else -1.0
    if d < 1e-9:
        return 0.0
    T = trap_time(d, vmax, amax)
    if t >= T:
        return sgn * d
    if t <= 0:
        return 0.0
    d_acc = vmax * vmax / amax
    if d <= d_acc:                        # triangle, peak at T/2
        th = T / 2.0
        if t < th:
            return sgn * 0.5 * amax * t * t
        tr = T - t
        return sgn * (d - 0.5 * amax * tr * tr)
    ta = vmax / amax
    if t < ta:
        return sgn * 0.5 * amax * t * t
    if t < T - ta:
        return sgn * (0.5 * amax * ta * ta + vmax * (t - ta))
    tr = T - t
    return sgn * (d - 0.5 * amax * tr * tr)


def coordinated_time(deltas, vmax, amax):
    """A multi-axis move takes as long as its slowest axis."""
    return max((trap_time(deltas[k], vmax[k], amax[k]) for k in deltas),
               default=0.0)


class Move:
    """A rest-to-rest move of several axes that all arrive together: every
    axis runs its own trapezoid stretched to the slowest one's duration, so
    a diagonal move is a straight line.  Sample with at(t)."""

    def __init__(self, start, goal, vmax, amax):
        self.start = dict(start)
        self.goal = dict(goal)
        self.axes = [k for k in goal if k in start]
        self.T = coordinated_time({k: goal[k] - start[k] for k in self.axes},
                                  vmax, amax)
        self._v, self._a = vmax, amax

    def at(self, t):
        out = {}
        for k in self.axes:
            d = self.goal[k] - self.start[k]
            Tk = trap_time(d, self._v[k], self._a[k])
            # stretch this axis's own profile to the shared duration
            tk = t * (Tk / self.T) if self.T > 0 else Tk
            out[k] = self.start[k] + trap_position(tk, d, self._v[k], self._a[k])
        return out

    @property
    def done_at(self):
        return self.T
