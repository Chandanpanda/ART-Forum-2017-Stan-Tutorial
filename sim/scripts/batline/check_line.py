"""Tier 0: the line, sized from the bat, then run month after month in
seconds.

    python3 sim/scripts/batline/check_line.py [-v]

The plan's contract for this suite: it fails when, at 300 a month with
injected failures, the takt slips, a store runs dry between visits, or a
buffer overflows -- and it reports which station saturates first at 1,000.

Three layers, each of which can fail on its own:

  the sizing maths   poisson_q against a direct sum; the printed stores'
                     chain against a simulation of the farm's own rule
  the static plan    every fit line.plan_line reports, at 300
  the event model    months of the line with station and robot faults and
                     the longest robot outage injected: every order inside
                     the promise, no store dry, no rack full

and lines that MUST fail -- one asked for more than its first station can
build, one restocked short, one with too few stretchers -- which have to
fail the way they should, or the months that pass prove nothing.
"""
import os
import sys
from dataclasses import replace
from math import exp, lgamma, log, sqrt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np

from batline import product, line as L
from batline.executive import des
from batline.spec import KID, Line, Faults

VERBOSE = "-v" in sys.argv
RESULTS = []
SEEDS = 10             # months at 300; each is a few tenths of a second


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def say(*a):
    if VERBOSE:
        print(*a)


# ======================================================== the sizing maths
def poisson_cdf(mean, k):
    """P(Poisson(mean) <= k), summed term by term: the check's own."""
    return sum(exp(j * log(mean) - mean - lgamma(j + 1.0)) for j in range(k + 1))


def sizing_maths(P):
    for mean in (0.4, 7.3, 64.0, 1500.0):
        for p in (0.9, 0.999):
            q = L.poisson_q(mean, p)
            check("poisson_q(%.1f, %.3f) = %d is the least k that covers it" % (mean, p, q),
                  poisson_cdf(mean, q) >= p > (poisson_cdf(mean, q - 1) if q else 0.0),
                  "cdf %.5f at %d" % (poisson_cdf(mean, q), q))
    # every printed store of the plan: the chain's stockout rate at its
    # level, and at levels it says are short, against the farm's rule run
    # period by period
    rng = np.random.default_rng(7)
    T_v = P.line.visit_every_days
    for part, v in P.sizes["printers"].items():
        mean = P.rates.starts[v["station"]] * T_v
        plates = v["printers"] * v["per_visit"]
        cdf, lo, _ = L.stock_law(mean, plates, v["per_plate"], Faults.PRINT_FAIL)
        level = v["order_up_to"]
        for lev in (level, max(lo, level - int(round(sqrt(2.0 * mean))))):
            chain = 1.0 - cdf[lev - lo]
            sim, se = simulate_farm(lev, mean, plates, v["per_plate"], Faults.PRINT_FAIL, rng)
            check("the %s store at %d runs dry as often as its chain says" % (part, lev),
                  abs(sim - chain) <= 4.0 * se + 1e-12,
                  "chain %.5f, simulated %.5f +- %.5f" % (chain, sim, se))
        check("the %s store's level lasts a period with probability %.3f" % (part, P.line.service),
              1.0 - cdf[level - lo] <= 1.0 - P.line.service, "dry %.5f a period" % (1.0 - cdf[level - lo]))


def simulate_farm(level, mean, plates, per, p_fail, rng, periods=200000, batches=100):
    """The person's rule, one visit at a time: unload the good plates,
    restart just enough plates to get back to the level, then a period's
    Poisson demand.  The fraction of periods the store ran short, and its
    standard error by batch means (dry periods come in runs)."""
    D = rng.poisson(mean, periods)
    U = rng.random((periods, plates))
    on_hand, arriving, dry = level, 0, np.zeros(periods, bool)
    for t in range(periods):
        on_hand += arriving
        need = level - on_hand
        q = 0 if need <= 0 else min(plates, -(-need // per))
        arriving = per * int((U[t, :q] >= p_fail).sum())
        dry[t] = D[t] > on_hand
        on_hand -= int(D[t])
    b = dry.reshape(batches, -1).mean(axis=1)
    return float(dry.mean()), float(b.std(ddof=1) / sqrt(batches))


# ======================================================== the static plan
def static_plan(P):
    for f in P.fits:
        check("plan at %.0f: %s" % (P.line.demand_month, f.name), f.ok,
              "%.2f%s" % (f.margin, ("; " + f.detail) if f.detail else ""))


# ===================================================== the event model
def months(P, n, outage=True):
    out = []
    for seed in range(n):
        out.append(des.run_month(P, seed=seed, outage_day=(3 + 2 * seed) if outage else None))
    return out


def held(P, rs):
    tag = "%.0f a month" % P.line.demand_month
    for r in rs:
        detail = des.summary(r).replace("\n", ";")
        check("%s, seed %d: every order ships inside the promise" % (tag, r.seed), r.takt_held, detail)
        check("%s, seed %d: no store runs dry" % (tag, r.seed), not r.stockouts,
              ", ".join("%s x%d" % (k, n) for k, (n, _) in r.kinds(r.stockouts).items()))
        check("%s, seed %d: no rack, shelf or stretcher overflows" % (tag, r.seed), not r.overflows,
              ", ".join("%s x%d" % (k, n) for k, (n, _) in r.kinds(r.overflows).items()))
        say(des.summary(r))


def busiest(P, rs):
    """(the plan's first resource to saturate, the event model's busiest)."""
    plan = max(P.sizes["utilisation"], key=P.sizes["utilisation"].get)
    mean_busy = {k: float(np.mean([r.busy[k] for r in rs])) for k in rs[0].busy}
    return plan, max(mean_busy, key=mean_busy.get), mean_busy


# =================================================== lines that must fail
def must_fail(d, P):
    # THE TAKT: asked for a fifth more than station 1 can build, sized for it
    over = 1.2 * P.sizes["capacity_month"]["frame"]
    Po = L.plan_line(d, Line(demand_month=over))
    rs = months(Po, 2)
    check("must fail: a line asked for %.0f a month, a fifth past station 1, slips its takt" % over,
          all(not r.takt_held for r in rs),
          "; ".join("%d late, %d unshipped" % (r.late - r.late_remade, r.unshipped) for r in rs))
    # A STORE: boards restocked to half a restock's mean use
    T_r = P.line.restock_every_days
    half = dict(P.sizes["bought"], board=int(0.5 * P.rates.starts["electronics"] * T_r))
    Ps = replace(P, sizes=dict(P.sizes, bought=half))
    rs = months(Ps, 2)
    check("must fail: boards restocked to half a week's use run dry",
          all(any(k.startswith("electronics") or k == "board" for k, _ in r.stockouts) for r in rs),
          "; ".join(", ".join(sorted(r.kinds(r.stockouts))) or "none" for r in rs))
    # A BUFFER: one tray of stretchers for a whole visit's sleeves
    trays = dict(P.sizes["trays"], stretcher=1)
    racks = dict(P.sizes["racks"], sleeves=1)
    Pb = replace(P, sizes=dict(P.sizes, trays=trays, racks=racks))
    rs = months(Pb, 2)
    check("must fail: one stretcher tray overflows with a visit's sleeves",
          all(any(k == "stretchers" for k, _ in r.overflows) for r in rs),
          "; ".join(", ".join(sorted(r.kinds(r.overflows))) or "none" for r in rs))


def main():
    d = product.design(KID)
    P = L.plan_line(d, Line())
    say(L.report(P))
    sizing_maths(P)
    static_plan(P)

    rs = months(P, SEEDS)
    held(P, rs)
    plan_top, sim_top, busy = busiest(P, rs)
    check("at 300 the plan and the event model agree on the busiest resource", plan_top == sim_top,
          "plan %s, event model %s" % (plan_top, sim_top))
    say("  at 300, busy: %s" % ", ".join("%s %.1f%%" % (k, 100 * v) for k, v in busy.items()))
    say("  the robot, per bat: the plan's estimate %.0f s, the event model's %.0f s"
        % (P.t_bat["robot"], np.mean([r.busy_per_bat["robot"] for r in rs])))
    say("  late only for a re-made sleeve: %d orders in %d months"
        % (sum(r.late_remade for r in rs), len(rs)))

    # 1,000 A MONTH: who saturates first, by the plan and by the line
    P1k = L.plan_line(d, Line(demand_month=1000.0))
    rk = months(P1k, 2)
    plan_top, sim_top, busy = busiest(P1k, rk)
    cap = P1k.sizes["capacity_month"]
    order = sorted(cap, key=cap.get)
    check("at 1,000 the plan and the event model agree on what saturates first", plan_top == sim_top,
          "plan %s (%.0f a month), event model %s (%.0f%% busy)"
          % (plan_top, cap[plan_top], sim_top, 100 * busy[sim_top]))
    must_fail(d, P)

    bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det and (VERBOSE or not ok) else ""))
    print("at 1,000 a month the %s station saturates first: it can make %.0f a month (then %s)"
          % (order[0], cap[order[0]], ", ".join("%s %.0f" % (k, cap[k]) for k in order[1:3])))
    print("check_line: %d checks, %d failed" % (len(RESULTS), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
