"""The golden bat: station 4 checking itself at the start of each line day
(README, Milestone 3).  numpy only; production.

WHAT IT GUARDS.  A rig fault the fit cannot absorb -- an encoder's scale
(it reads (1 + eps) q where its axis turned q: the fit takes the gimbal's
motion from the encoders, so it reads every bat's gyro scale wrong by
eps), the station's clock, an encoder's cyclic error -- moves every bat's
fitted terms the same way.  The gear ratio itself is not one: each
encoder is on its axis (spec.Gimbal), so a belt that slips or stretches
moves the axis away from its command while the encoders still read it
truly, the fit follows the axis, and the drift shows only as following
error, which every bat's own verdict bounds.  A bat is in target when its
group's RMS error is (fit.thresholds: per group, over bats and the group's
terms), so a drift matters once it moves a group's mean square past what
the plan's own sigmas leave of the target:

    |b_g|^2 / n_g  >  tau_g^2 - mean_g sigma^2      (sigma: the bats' plan)

WHAT IT CAN SEE.  The golden bat is clamped and powered up again each day.
A re-clamp redraws where it sits (spec.Clamp.SEAT_*), a power-up its
biases (spec.Imu.*_POWERUP), and neither is a drift.  A term whose redraw
alone hides the smallest drift that matters from the test cannot be
watched by a golden bat; which terms those are is computed (watched()),
and the cameras' own drift check (rig/track.drift_check) is what guards
the pose-side terms they leave (the mounting and the lever arm).

THE TEST.  A golden day is r runs of the bats' own plan, the bat clamped
once and powered again for each; the runs' terms, combined by their
information (combine), are the day's.  Runs, not one tour r times as
long: each is the plan as the bats run it, so the fit, its noise model and
its verdict are the ones production uses, and where r > 1 a run that
disagrees with its siblings shows (Day.Q).  Over the watched terms, d the day's
less the certified (the mean of K commissioning days, each with its own
measurement and its own redraw),

    T^2 = d' (Sigma_day + Sigma_rep + (Sigma_c + Sigma_rep) / K)^-1 d

against chi-square_p at alpha_day = GOLDEN_STOPS_YEAR / the days in a
year: a healthy rig stops the line no more often.  r (golden_plan) is the
fewest runs for which the smallest drift that matters, along the worst
direction of any group, has non-centrality lambda*: P(chi2_p(lambda*) >
threshold) = GOLDEN_POWER, by Patnaik's approximation.  Sigma_rep -- what
the golden bat scatters by from day to day beyond its fit's sigma -- is
measured at commissioning, term by term (certify): what its K days
scatter by beyond their sigmas, never less than the spec's redraw, and
guarded for being measured at all -- each term's days' variance taken
kappa = F(1, K - 1) / chi2_1 times, at alpha_day (scatter_factor), so
that one term's scatter, measured on K - 1 dof, still leaves a healthy
day's chance of passing its threshold at alpha_day.  Taken as measured,
K days' variances let a rig whose days scatter twice their measurement
along one watched direction stop the line 2.7 times as often as
alpha_day on K = 10 and 2.4 times on K = 20: per-term variances cannot
see a direction, and more days alone do not fix it.  Guarded, on K = 20
(kappa 1.31), it stops 1 such day in 2000, 0.2 times as often
(check_calibration, SYNTHETIC); on K = 10 (kappa 1.85) the
guard refuses 15% of rigs that scatter as the spec says and takes 2.5
runs a day on the rest (2000 drawn days each, M3's kid, on the plan
sized for ten days taken as measured).  So
commissioning is Quality.GOLDEN_RUNS = 20 days.  The plan is sized for
what the guarded scatter adds, in expectation, to a rig that scatters as
the spec says (scatter_floor); the certificate then sets the runs the
rig's own golden days take, on what its days showed.
A run whose fit failed, or whose terms are not numbers, stops the line
(daily).

THE BATS' PLAN IS THE GOLDEN BAT'S TOO.  r runs of a plan designed for the
bats alone can be many: nothing in its design values a term the golden bat
watches beyond the bats' own threshold, so a term can sit near it (on M3's
kid plan the gyro's z scale did, at 0.77 of its target) and its margin
then takes r = 27 runs.  day_design() also designs the plan with each
watched term held to what lets one run reach lambda*, and keeps whichever
of the two makes the calibration station's day shorter: the day's bats
(spec.Line.per_day) and the golden day, both runs of the same plan.

TWO GOLDEN BATS per design, run on alternate days.  After a stop the other
is run: both moved, the rig drifted; one, that bat aged (attribute).
"""
import dataclasses
from dataclasses import dataclass
from math import radians, sqrt

import numpy as np

from ..spec import Clamp, Imu, Line, Quality
from ..rig.calib import chi2_quantile, chi2_sf
from . import fit as F
from . import plan as P


def alpha_day(line=None):
    """A healthy rig's chance of stopping the line on a given day."""
    line = Line() if line is None else line
    return Quality.GOLDEN_STOPS_YEAR / (12.0 * line.days_month)


def noncentral_sf(x, p, lam):
    """P(chi2_p(lam) > x), by Patnaik's two-moment approximation (Patnaik
    1949, Biometrika 36: 202): the non-central chi-square as c chi2_nu."""
    c = (p + 2.0 * lam) / (p + lam)
    nu = (p + lam) ** 2 / (p + 2.0 * lam)
    return chi2_sf(x / c, nu)


def lambda_star(p, alpha=None, power=None):
    """The non-centrality at which a chi2_p test at alpha has the power."""
    alpha = alpha_day() if alpha is None else alpha
    power = Quality.GOLDEN_POWER if power is None else power
    x = chi2_quantile(p, alpha)
    lo, hi = 0.0, 1.0
    while noncentral_sf(x, p, hi) < power:
        lo, hi = hi, 2.0 * hi
    while hi - lo > 1e-9 * hi:
        m = 0.5 * (lo + hi)
        if noncentral_sf(x, p, m) < power:
            lo = m
        else:
            hi = m
    return hi


def scatter_factor(K, alpha=None):
    """kappa: what certify multiplies a term's days' variance by, measured
    on K - 1 dof, for a healthy day to pass that term's threshold with
    probability alpha_day still -- F(1, K - 1) over chi-square_1 at alpha,
    the squared Student-t quantile over the normal's (a prediction
    interval for one more day from K)."""
    alpha = alpha_day() if alpha is None else alpha
    return F.f_quantile(1, K - 1, alpha) / chi2_quantile(1, alpha)


def scatter_floor(K):
    """What certify's scatter adds, in expectation, to a term whose days
    scatter as their sigmas and the redraw say, as a fraction of that
    variance: E[(kappa X - 1)+], X a chi-square_nu over nu (its days'
    variance over its own, nu = K - 1) and kappa scatter_factor(K), which
    is kappa P(chi2_{nu+2} > nu / kappa) - P(chi2_nu > nu / kappa), since x
    times the chi-square_nu density is nu times the chi-square_{nu+2}'s."""
    nu, k = K - 1, scatter_factor(K)
    return k * chi2_sf(nu / k, nu + 2) - chi2_sf(nu / k, nu)


def repeatability(d, rig):
    """(NT,) 1 sigma of what a re-clamp and a power-up redraw in each term,
    by the spec, before commissioning measures it: the biases a power-up's,
    the mounting and lever arm the seat's (the lever turned about the seat's
    pivot, spec.Clamp, and moved across the bat's axis -- clamp y and z, the
    jaws centring it there while its stop holds it along x, as sim.draw_bat
    and plan.spreads seat it), the part's own scale, cross-axis and clock
    none."""
    r, _ = P.board_nominal(d, rig)
    v = (r - P.seat_pivot(rig)) * 1000.0
    across = np.array([0.0, 1.0, 1.0])
    out = np.zeros(F.NT)
    for k, (name, n, _, _) in enumerate(F.GROUPS):
        m = F.GROUP_OF == k
        if name == "accel bias":
            out[m] = Imu.ACC_POWERUP * 1000.0
        elif name == "gyro bias":
            out[m] = Imu.GYRO_POWERUP
        elif name == "mount":
            out[m] = Clamp.SEAT_ROT
        elif name == "lever":
            out[m] = np.sqrt(across * Clamp.SEAT_POS ** 2 + radians(Clamp.SEAT_ROT) ** 2 * (v @ v - v ** 2))
    return out


def margins(prod):
    """(groups,) |b_g|^2 at which a drift puts group g out of target:
    n_g (tau_g^2 - mean sigma^2), the bats' plan's sigmas."""
    out = np.zeros(len(F.GROUPS))
    for k, (_, n, tau, _) in enumerate(F.GROUPS):
        sg = prod.sigma[F.GROUP_OF == k]
        out[k] = max(0.0, n * (tau ** 2 - float(np.mean(sg ** 2))))
    return out


def watched(prod, rep, K=None):
    """(mask (NT,), p, lambda*): the terms a golden bat can watch, found at
    the p they make (lambda* grows with p).  The drift that matters must
    reach lambda* through the redraw and the measurement together; a term
    is watched when the redraw, as the test will carry it (rep^2 and what
    certify's floor adds to it over K days, scatter_floor), takes no more of
    that than the measurement may -- past that the golden plan would have
    to measure the bat finer than it can be seated or powered up, and the
    instrument that sees the seat (the cameras) is the one to watch it."""
    K = Quality.GOLDEN_RUNS if K is None else K
    D2 = margins(prod)[F.GROUP_OF]
    held = rep ** 2 * (1.0 + scatter_floor(K))
    p = F.NT
    seen = set()
    while True:
        lam = lambda_star(p)
        w = D2 / (2.0 * lam) >= held
        p_new = int(np.count_nonzero(w))
        if p_new == p or p_new in seen or p_new == 0:
            return w, max(p_new, 1), lambda_star(max(p_new, 1))
        seen.add(p)
        p = p_new


def group_lambdas(St, prod, w, rep, K=None, Sigma_rep=None):
    """(groups,) the non-centrality of the smallest drift that matters in
    each watched group, along the worst direction a drift can take: one
    that moves group g by |b_g|^2 = margins(g) and every other term however
    it likes.  Minimising b' C^-1 b over those leaves the group's own
    (marginal) block, so lambda_g = margins(g) / the largest eigenvalue of
    C's g block, C the test's covariance (test_cov: today's measurement St
    and redraw R, and the certified mean's, the mean of K days each with
    its own, (St + R) / K); nan for a group not watched.  R is Sigma_rep
    when given (a certificate's); else what certify will take on a rig
    that redraws by rep and scatters by no more: rep^2, and what its floor
    adds over K days on that and on the measurement, in expectation
    (scatter_floor)."""
    K = Quality.GOLDEN_RUNS if K is None else K
    idx = np.flatnonzero(w)
    Sw = St[np.ix_(idx, idx)]
    if Sigma_rep is None:
        r2 = rep[idx] ** 2
        R = np.diag(r2 + (np.diag(Sw) + r2) * scatter_floor(K))
    else:
        R = Sigma_rep
    return worst_lambdas((Sw + R) * (1.0 + 1.0 / K), prod, w)


def worst_lambdas(C, prod, w):
    """(groups,) margins(g) / the largest eigenvalue of C's g block, C (p,
    p) a test's covariance over the watched terms w (group_lambdas); nan
    for a group not watched."""
    idx = np.flatnonzero(w)
    D2 = margins(prod)
    out = np.full(len(F.GROUPS), np.nan)
    for k in range(len(F.GROUPS)):
        m = F.GROUP_OF[idx] == k
        if np.any(m):
            out[k] = D2[k] / float(np.linalg.eigvalsh(C[np.ix_(m, m)]).max())
    return out


@dataclass
class GoldenPlan:
    plan: object               # plan.CalPlan: the bats' own plan
    repeats: int               # runs of it a golden day takes
    watched: np.ndarray        # (NT,) bool
    p: int
    lam: float                 # lambda*
    lambdas: np.ndarray        # (groups,) each watched group's worst-direction lambda, predicted, floor and all
    rep: np.ndarray            # (NT,) the redraw it was sized for
    considered: tuple = ()     # day_design's candidates, (plan s, runs) each, this one among them

    @property
    def duration_s(self):
        return self.repeats * self.plan.duration_s


def golden_plan(d, rig=None, prod=None, rep=None, K=None):
    """The golden bat's day for design d: the bats' plan run r times, r the
    fewest for which every watched group's worst direction reaches lambda*
    on the plan's predicted covariance over r (the runs independent but for
    the redraw, which is the day's), with what certify's floor will add to
    it on a rig that scatters no more (group_lambdas).  A watched group the
    bats' own plan leaves no margin to its target -- its sigmas alone at or
    past it, so a drift of any size matters -- has no smallest drift to
    watch for, and the plan is infeasible (ValueError), as when the redraw
    hides them all."""
    from ..rig import spec as RS
    rig = RS.design(d) if rig is None else rig
    prod = P.for_design(d) if prod is None else prod
    rep = repeatability(d, rig) if rep is None else rep
    K = Quality.GOLDEN_RUNS if K is None else K
    w, p, lam = watched(prod, rep, K)
    if not np.any(w):
        raise ValueError("infeasible: repeatability -- the golden bat's redraw hides every drift that matters")
    D2 = margins(prod)
    gw = np.unique(F.GROUP_OF[w])
    none_left = [F.GROUPS[k][0] for k in gw if not (np.isfinite(D2[k]) and D2[k] > 0.0)]
    if none_left:
        raise ValueError("infeasible: margin -- the bats' own plan leaves %s no margin to its target, so every "
                         "drift matters and none is smallest" % ", ".join(none_left))
    # as r grows the measurement goes and the redraw stays, as the test
    # carries it: watched() keeps that under half of each margin over
    # lambda*, so some r reaches lambda*; a redraw passed in that does not
    # would send r without end
    held = rep ** 2 * (1.0 + scatter_floor(K)) * (1.0 + 1.0 / K)
    capped = [F.GROUPS[k][0] for k in gw if np.max(held[w & (F.GROUP_OF == k)]) * lam >= D2[k]]
    if capped:
        raise ValueError("infeasible: repeatability -- no number of runs takes %s past the redraw to lambda*"
                         % ", ".join(capped))
    St = P.terms_cov(prod)
    r = 1
    while True:
        lams = group_lambdas(St / r, prod, w, rep, K)
        if not np.all(np.isfinite(lams[gw]) & (lams[gw] > 0.0)):
            raise ValueError("infeasible: the plan's covariance gives no lambda in %s"
                             % ", ".join(F.GROUPS[k][0] for k in gw if not (np.isfinite(lams[k]) and lams[k] > 0.0)))
        short = gw[lams[gw] < lam]
        if not len(short):
            return GoldenPlan(prod, r, w, p, lam, lams, rep)
        # lambda grows as r while the measurement dominates the redraw; the
        # shortest group's shortfall sets the next r
        r = max(r + 1, int(np.ceil(r * np.max(lam / lams[short]))))


def one_run_thresholds(prod, rep, K=None):
    """(NT,) fit.thresholds with each term the golden bat watches on prod
    held to the 1 sigma at which one run of a plan reaches lambda*: in
    group_lambdas' covariance, a term's own share, (sigma^2 + rep^2) (1 +
    scatter_floor) (1 + 1/K), is at most its group's margin over lambda*.
    The margins are prod's, and a plan that measures the watched terms
    finer only widens them; group_lambdas on the plan designed to these
    says what r it then takes."""
    K = Quality.GOLDEN_RUNS if K is None else K
    w, _, lam = watched(prod, rep, K)
    D2 = margins(prod)[F.GROUP_OF]
    c = (1.0 + scatter_floor(K)) * (1.0 + 1.0 / K)
    s = F.thresholds()
    s2 = D2 / (lam * c) - rep ** 2
    if np.any(w & (s2 <= 0.0)):
        raise ValueError("infeasible: one run cannot see past the redraw in %s"
                         % ", ".join(sorted({F.GROUPS[k][0] for k in F.GROUP_OF[w & (s2 <= 0.0)]})))
    s[w] = np.minimum(s[w], np.sqrt(np.where(w, s2, 1.0)[w]))
    return s


def day_design(d, rig=None, line=None):
    """GoldenPlan (its .plan the bats' plan) for design d: of the plan
    designed for the bats' thresholds alone and the one designed also to
    one_run_thresholds on the first's margins, the one whose day at the
    calibration station -- line.per_day bats and the golden day's r runs,
    each the plan's duration -- is shorter.  A pair the golden bat cannot
    watch (golden_plan's ValueError) is not a candidate; if neither is,
    the first's reason is raised."""
    from ..rig import spec as RS
    rig = RS.design(d) if rig is None else rig
    line = Line() if line is None else line
    rep = repeatability(d, rig)
    base = P.design(d, rig)
    cands, why = [], None
    try:
        cands.append(golden_plan(d, rig, prod=base, rep=rep))
    except ValueError as e:
        why = e
    try:
        tight = P.design(d, rig, s=one_run_thresholds(base, rep))
        cands.append(golden_plan(d, rig, prod=tight, rep=rep))
    except ValueError as e:
        why = why or e
    if not cands:
        raise why
    best = min(cands, key=lambda gp: (line.per_day + gp.repeats) * gp.plan.duration_s)
    return dataclasses.replace(best, considered=tuple((gp.plan.duration_s, gp.repeats) for gp in cands))


_days = {}                     # bat -> GoldenPlan: day_design, once per bat


def for_design(d):
    """The bats' plan and its golden day for d's bat (day_design), designed
    once per bat; plan.for_design is its .plan."""
    if d.bat not in _days:
        _days[d.bat] = day_design(d)
    return _days[d.bat]


# ================================================================ A DAY
@dataclass
class Day:
    terms: np.ndarray          # (p,) the watched terms, the runs' information-weighted mean
    Sigma: np.ndarray          # (p, p) its covariance
    n: int                     # runs
    Q: float                   # the runs' scatter about it, chi-square on p (n - 1) dof


def combine(results, w):
    """Day from a golden day's runs (fit.CalResult each), over the watched
    terms w: each run weighted by its fit's own information, so the day's
    covariance is what the runs' sigmas together leave.  A run that gave no
    calibration -- its fit failed, its stream was cut short (fit.CalResult.lost:
    a cut run can still solve, on less than the plan the day was sized
    for), or its watched terms or their covariance are not finite, or that
    covariance is not positive definite -- gives no day: ValueError, naming
    the runs and why, since one run's NaN would otherwise carry through the
    day to a T^2 no threshold refuses."""
    idx = np.flatnonzero(w)
    bad = []
    for k, r in enumerate(results):
        if r.failed:
            bad.append("run %d: the fit failed (%s)" % (k, r.failed))
        elif getattr(r, "lost", 0):
            bad.append("run %d: its stream was cut short (%d quiet intervals lost)" % (k, r.lost))
        elif not (np.all(np.isfinite(r.terms[idx])) and np.all(np.isfinite(r.Sigma_terms[np.ix_(idx, idx)]))):
            bad.append("run %d: its watched terms are not finite" % k)
        else:
            try:
                np.linalg.cholesky(r.Sigma_terms[np.ix_(idx, idx)])
            except np.linalg.LinAlgError:
                bad.append("run %d: its watched terms' covariance is not positive definite" % k)
    if bad or not len(results):
        raise ValueError("the golden runs give no day: " + ("; ".join(bad) if bad else "there are none"))
    Wi = [np.linalg.inv(r.Sigma_terms[np.ix_(idx, idx)]) for r in results]
    S = np.linalg.inv(np.sum(Wi, 0))
    t = S @ np.sum([Wk @ r.terms[idx] for Wk, r in zip(Wi, results)], 0)
    Q = float(sum((r.terms[idx] - t) @ Wk @ (r.terms[idx] - t) for Wk, r in zip(Wi, results)))
    return Day(t, 0.5 * (S + S.T), len(results), Q)


# ================================================================ CERTIFY
@dataclass
class Cert:
    terms: np.ndarray          # (p,) the certified watched terms: the commissioning days' mean
    Sigma_c: np.ndarray        # (p, p) the days' mean covariance
    Sigma_rep: np.ndarray      # (p, p) the scatter beyond it: each term's days' variance, guarded, less the measurement's, the redraw at least
    K: int                     # days
    watched: np.ndarray        # (NT,) bool
    lambdas: np.ndarray        # (groups,) worst-direction lambdas with the measured scatter, at repeats runs a day
    feasible: bool             # some number of runs a day takes every watched group to lambda*
    plan_hash: str
    Q: float = 0.0             # the days' scatter about their mean, chi-square on p (K - 1) dof
    repeats: int = 0           # runs a golden day takes with this scatter: the fewest that reach lambda*


def certify(days, gp, prod):
    """Cert from K commissioning days (each a list of the golden plan's
    runs, the bat clamped again and powered up between days): their mean,
    their mean covariance, and the scatter beyond it, then how many runs a
    golden day takes for every watched group to reach lambda* with that
    scatter -- the day's measurement goes as the runs, the scatter and the
    certified mean's error stay -- or that none does (feasible False).

    The scatter is what K days can identify: each watched term's own
    variance from day to day, on K - 1 degrees of freedom.  A full matrix
    over p terms needs more days than terms, and a test that takes it only
    when the days show it misses, nearly always, a scatter small enough to
    hide from K days and large enough to stop the line many times more
    often than alpha_day.  So none is gated and none is set to zero: each
    term's is its days' variance, guarded for being measured
    (scatter_factor), less the measurement's, and never less than the
    spec's redraw (gp.rep).  golden_plan sizes the day for what this adds,
    in expectation, to a rig that scatters as the spec says, and K days'
    variance is that expectation give or take a third of it (K - 1 = 19
    dof): before the guard, on ten days, one of M3's kid golden bats'
    days scattered 1.95 times their lever arm's measurement and redraw,
    which left its day lambda 45 of 51.6 at the plan's one run and 60 at
    two.  So the certificate, not the plan, sets the runs the rig's golden
    days take.  A day whose runs give no day (combine) refuses the cert:
    ValueError, naming it."""
    w = gp.watched
    idx = np.flatnonzero(w)
    D_ = []
    for k, runs in enumerate(days):
        try:
            D_.append(combine(runs, w))
        except ValueError as e:
            raise ValueError("commissioning day %d: %s" % (k, e)) from e
    K = len(D_)
    if K < 2:
        raise ValueError("certify: %d commissioning days -- the scatter between days needs two or more" % K)
    T = np.array([x.terms for x in D_])
    Sc = np.mean([x.Sigma for x in D_], 0)
    D = T - T.mean(0)
    Q = float(np.sum(D * np.linalg.solve(Sc, D.T).T))
    v = scatter_factor(K) * np.var(T, axis=0, ddof=1) - np.diag(Sc)
    Srep = np.diag(np.maximum(v, gp.rep[idx] ** 2))
    gw = np.unique(F.GROUP_OF[idx])
    held = Srep + (Sc + Srep) / K          # what more runs a day do not shrink

    def lams_at(r):
        # test_cov of a day of r runs: the days were gp.repeats runs each
        return worst_lambdas(Sc * gp.repeats / r + held, prod, w)
    if np.any(worst_lambdas(held, prod, w)[gw] <= gp.lam):
        return Cert(T.mean(0), Sc, Srep, K, w, lams_at(gp.repeats), False, gp.plan.hash, Q, 0)
    r = 1
    while True:
        lams = lams_at(r)
        short = gw[lams[gw] < gp.lam]
        if not len(short):
            return Cert(T.mean(0), Sc, Srep, K, w, lams, True, gp.plan.hash, Q, r)
        r = max(r + 1, int(np.ceil(r * np.max(gp.lam / lams[short]))))


# ================================================================ DAILY
@dataclass
class GoldenVerdict:
    T2: float
    threshold: float
    stop: bool
    worst_group: str           # the group whose terms carry most of T^2; why, when the runs gave no day
    share: dict                # group -> its part of T^2
    day: Day                   # None when the runs gave no day


def test_cov(cert, Sigma_day, Sigma_rep=None):
    """(p, p) the covariance of a healthy day's watched terms less the
    certified: the day's own measurement and redraw, and the certified
    mean's -- the mean of K days, each with its own measurement and its own
    redraw, (Sigma_c + Sigma_rep) / K."""
    Sr = cert.Sigma_rep if Sigma_rep is None else Sigma_rep
    return Sigma_day + Sr + (cert.Sigma_c + Sr) / cert.K


def statistic(cert, day, Sigma_rep=None):
    """(T^2, p, per-term contributions) of a day (Day) against cert."""
    dvec = day.terms - cert.terms
    u = np.linalg.solve(test_cov(cert, day.Sigma, Sigma_rep), dvec)
    return float(dvec @ u), len(dvec), dvec * u


def daily(cert, runs, alpha=None):
    """GoldenVerdict on a day's runs (fit.CalResult each): stop the line
    when T^2 passes chi2_p at alpha_day, naming the group that carries most
    of it.  It fails closed: runs that give no day (combine) stop the line
    with T^2 infinite and the reason where the group would be, and a T^2
    that is not a number is not under the threshold."""
    alpha = alpha_day() if alpha is None else alpha
    thr = chi2_quantile(int(np.count_nonzero(cert.watched)), alpha)
    try:
        day = combine(runs, cert.watched)
    except ValueError as e:
        return GoldenVerdict(np.inf, thr, True, str(e), {}, None)
    T2, p, c = statistic(cert, day)
    g = F.GROUP_OF[np.flatnonzero(cert.watched)]
    share = {F.GROUPS[k][0]: float(np.sum(c[g == k])) for k in np.unique(g)}
    worst = max(share, key=share.get)
    return GoldenVerdict(T2, thr, not (T2 <= thr), worst, share, day)


def attribute(first, second):
    """After a stop on one golden bat, the other's verdict decides: both
    stopped, the rig drifted; only the first, that bat aged."""
    if not first.stop:
        return "none"
    return "rig" if second.stop else "bat"
