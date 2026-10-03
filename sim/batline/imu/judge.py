"""The fit against the truth.  TRUTH: for the checks only -- the station
cannot see a bat's true terms; these can.

Both sides are read the same way (fit.terms): the biases at the die's mean
temperature during calibration, the mounting as the rotation between the
fit's chip axes and the true ones (each the polar factor of its own
accelerometer map, in the clamp's frame), the lever arm at the board's
stop, where the swing holds it (sim.py, THE BAT).
"""
import numpy as np

from . import fit as F


def errors(res, truth):
    """(NT,): fitted less true, each term in its group's unit."""
    return F.terms(res.theta, res.dT_cal, truth.U) - F.terms(truth.theta, res.dT_cal, truth.U)


def z(res, truth):
    """(NT,): each error over the fit's own 1 sigma for it."""
    return errors(res, truth) / np.sqrt(np.diag(res.Sigma_terms))


def group_rms(E):
    """(groups,): RMS over bats and over each group's terms, E (bats, NT)."""
    return np.array([np.sqrt(np.mean(E[:, F.GROUP_OF == k] ** 2)) for k in range(len(F.GROUPS))])
