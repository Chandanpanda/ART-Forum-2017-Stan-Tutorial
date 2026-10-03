"""Tier 0: the bat's board as the line reaches it -- its power, the
simulated firmware and radio link against the derivations the station's
limits rest on, station 2's self-test, the calibration record over the air,
the bat's sample clock on a central's -- and the factory iPhone's test at
station 4.

    python3 sim/scripts/batline/check_link.py [-v]

The plan's contract for this suite (README, Milestone 4): it fails if the
self-test misses an injected fault, the record does not survive a round
trip, or the clock offset is not recovered; and (the plan's "slow fused
swing") if the app's fused sweet spot leaves the rig's by more than the
accuracy budget at the test's swing, or the test passes bats it should not.  No physics: the board is a
state machine and the link a channel model (board/firmware.py, radio.py),
and the gimbal follows its paths exactly (phone/sim.py); MuJoCo is
imported by imu/sim.py's bat drawing and never stepped.

Every tolerance says where it comes from: a datasheet's spread, an
instrument's accuracy, a Monte Carlo's own standard error (batch means), a
chi-square or binomial quantile, float rounding.  The statistical ones are
held at Module.Z_SIGMAS, its tail shared among a check's tests by
Bonferroni.  Each check has a must-fail: a deliberately wrong variant the
same test must reject, so no check passes because it cannot fail.

POWER, from the facts (board/power.py):
   1  the hold-up capacitor at its low tolerance carries the streaming
      board through Rules.BOUNCE_S from the cells' voltage at
      Rules.HOLD_LIFE, at the phone's interval (the game's case, and the
      station's hold-up test's).  Must fail: a 100 uF part.
   2  power.window_charge bounds the charge the simulated board draws in
      any window (a bounce, an interval, four, the hold-up time), streaming
      to either central for a minute, but for its Z tail: no more bursts of
      loss over it than the binomial allows.  Must fail: the mean draw
      times the window, at a bounce.
   3  the swing rig's dummy pack, charged as power.dummy_volts says: each
      supercapacitor under its rating, the pack inside the board's supply
      range, the cells' envelope.  Must fail: charged to the rating.
   4  the currents' limits re-derive the rules: the usable charge solves
      the cells' curve under the peak draw; a board at the streaming limit
      plays exactly Rules.PLAY_H, at the sleep limit sleeps
      Rules.SLEEP_DAYS; a simulated board plays at least PLAY_H; and the
      guard band passes no board over a limit on any reading the
      instrument's error allows.  Must fail: the limit read with no guard.

MODEL, the simulated board against the derivations:
   5  each state's mean current in the simulation equals power.draw's,
      within Z of the batch means' standard error and one event's charge.
      Must fail: the streaming draw at the other central's interval.
   6  a contact break: the streaming board browns out no sooner than
      power.holdup_s and no later than its capacitor would carry its steady
      draw alone, at 40 random phases.  Must fail: the hold-up from the
      mean draw.
   7  the record's commit is atomic: power lost at every moment across the
      erase and write leaves the old record or the new one, never neither,
      and both are seen.  Must fail: a firmware that writes one page in
      place.
   8  wake on motion: asleep, the board wakes and advertises WOKE within
      the advertising delay of the first sample Firmware.WOM_G off its
      reading as it slept; a still board's noise never comes near it.
      Must fail: a tap of half WOM_G.
   9  throughput: ten seconds' stream to each central arrives whole, no
      packet dropped or missing, as power.keeps_up predicts.  Must fail:
      one packet an event at 30 ms, which keeps_up and the simulation both
      refuse.

SELF-TEST, station 2 (board/procedures.station_test):
  10  of 200 healthy boards drawn from their parts' spreads, no more fail
      than Quality.TEST_ALPHA a board allows, at the binomial's Z tail.
      Must fail: the datasheet's typical spreads read as limits, a common
      misreading (z = 1).
  11  each of the planted faults (board/sim.FAULTS) is refused, at the step
      meant to catch it.  Must fail: the hold-up step skipped, and the
      board with no hold-up passes.
  12  time: the healthy runs' mean is within the plan (procedures.plan,
      which line.py times station 2 by) to Z of its standard error, and the
      longest within Module.OVERRUN of it.  Must fail: a plan that writes
      flash at the probe's rate, not the flash's.

RECORD over the air (procedures.BleLink, pair):
  13  a record committed over the station's dongle reads back bit for bit
      and the factory iPhone pairs to it within its plan.  Must fail: the
      database holding a newer record than the bat.
  14  the commit survives a lossy link (30 % of packets), the link dropped
      mid-write, and power lost mid-commit, each retried once.  Must fail:
      a link that never retries, under the dropped link.
  15  a stuck flash bit is refused and nothing stored; a passing flip is
      rewritten.  Must fail: a link that answers reads from its own cache.
  16  a record gone bad in the field is refused at pairing and restored
      from the database.  Must fail: pairing that trusts the bat's CRC
      alone.

CLOCK (board/sync.py):
  17  the bat's sample clock recovered from listen_s of stream, 100 boards
      a central: the offset's RMS within Quality.SYNC_S and within the
      model's own prediction (sync.envelope_rms), the rate's within
      sync.slope_sd, at the chi-square's Z tail.  Must fail: the least-
      squares clock; a quarter of the packets.
  18  counters that wrap and a bat that resets mid-stream recover the clock
      of the last run.  Must fail: the counters taken raw.
  19  the lower hull's edge is Moon's linear program's optimum, against a
      brute force over every pair, on random point sets.  Must fail: the
      hull's first edge.

CONTRACTS:
  20  the simulated benches meet every abstract method of the four board
      HALs and the factory iPhone's two (phone/hal).  Must fail: a supply
      missing limited().
  21  truth separation: the production modules (board/hal, power, sync,
      procedures; phone/camera, hal, app, test) import neither MuJoCo nor a
      TRUTH module (board/firmware, radio, sim; imu's scene, model, sim,
      judge; phone/sim, picture), by their source and in a fresh
      interpreter.  Must fail: the same scan on board/sim.py and on
      phone/sim.py.

THE FACTORY IPHONE, station 4 (phone/test.py, app.py; the kid's bat):
  22  the budget's truth is exact: with no error at all, the accuracy study
      on the gimbal's swing (test.GimbalTruth) puts the sweet spot where the
      drawing does, to within the Z tail of its trials' own standard error
      (chi-square on two degrees of freedom).  Must fail: the anchor at the
      stance's nearest sample, which starts the IMU a fraction of a sample
      into the move.
  23  30 calibrated bats through the whole test (phone/sim.ready: the
      bat drawn, its record written over the dongle, the iPhone pairing,
      filming the swing and fusing; each centroid the middle of its band's
      silhouette, which check_swing 15 holds the app's model of to a
      rendered picture): no more refused than
      Quality.TEST_ALPHA allows at the binomial's Z tail, and their errors
      no larger than the budget's: the sum of their d^2 under the chi-square
      on 2n degrees of freedom at its Z tail.  Must fail: the same bats
      held to the budget of the game-length swing (the turn the design
      starts from), which is not the swing they made.
  24  30 bats whose app fuses with the datasheet's record in place of
      the bat's: refused at least as often as the swing was designed to
      refuse them (test.slow_swing's power), at the binomial's Z tail.
      Must fail: the calibrated bats of 23, whose records fit.
  25  time: the healthy runs' mean is within the plan (test.plan, which
      line.py times station 4 by) to Z of its standard error, and the
      longest within Module.OVERRUN of it.  Must fail: a plan without the
      pairing.
"""
import os
import re
import struct
import subprocess
import sys
import time
import zlib
from math import erfc, sqrt, floor, comb, log
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np                       # noqa: E402

from batline.spec import (Imu, Board, Mcu, Link, Firmware as FWS, Module, Quality, Rules, Smu, Est,  # noqa: E402
                          DummyCell, Cell)
from batline.rig.pose import normal_quantile     # noqa: E402
from batline.rig.calib import chi2_quantile      # noqa: E402
from batline.imu import record as R              # noqa: E402
from batline.imu.part import lsb, G0             # noqa: E402
from batline.board import hal as H, power as P, sync as SY, procedures as PR, sim as S   # noqa: E402
from batline.board.firmware import Firmware, ADV, SLEEP                                # noqa: E402
from batline import product, imu_fusion_sim as study                                    # noqa: E402
from batline.spec import KID                                                           # noqa: E402
from batline.phone import test as PT, sim as PS, hal as PH                             # noqa: E402

VERBOSE = "-v" in sys.argv
RESULTS = []
Z = Module.Z_SIGMAS
TAIL = 0.5 * erfc(Z / sqrt(2.0))         # the one-sided tail Z sigmas leaves
EPS = np.finfo(float).eps
HERE = os.path.dirname(os.path.abspath(__file__))
IMG = PR.image()


def zshare(m):
    """Z with its tail shared among m tests (Bonferroni)."""
    return normal_quantile(1.0 - TAIL / m)


def check(name, ok, detail="", secs=None):
    now = time.time()
    RESULTS.append((name, bool(ok), detail, now - check.t if secs is None else secs))
    check.t = now
    return bool(ok)


check.t = time.time()


# ================================================================ BENCHES
def nominal(board):
    """The board with its parts at their datasheets' typical draw and its
    capacitor at its nominal value: what power.py derives for."""
    board.truth.i_scale = {k: 1.0 for k in board.truth.i_scale}
    board.truth.hold_c = Board.HOLD_C
    return board


def fresh(seed, central="dongle", per=None, board=None, firmware=Firmware, source_for=None, typical=False):
    """A board flashed, given its serial, reset and advertising, on a bench."""
    rng = np.random.default_rng(seed)
    bd = board if board is not None else S.draw_board(rng)
    if typical:
        nominal(bd)
    b = S.Bench(bd, rng, central=central, per=per, firmware=firmware, source_for=source_for)
    serial = "BAT%05d" % seed
    b.supply.source(P.v_fresh(), PR.supply_limit())
    b.clock.wait(1e-3)
    b.swd.connect()
    b.swd.erase_all()
    b.swd.program(0, IMG)
    b.swd.program(Mcu.UICR_SERIAL, serial.encode().ljust(16, b"\0"))
    b.swd.reset()
    b.clock.wait(Imu.START_S + 2.0 * Link.ADV_JITTER)
    return b, serial


def connect(b, serial, ci):
    if not b.radio.connect(serial, ci, PR.timeout(Link.ADV_SLOW_S + 2.0 * ci)):
        raise RuntimeError("%s would not connect" % serial)


def stream_on(b, ci):
    b.radio.stream(True, PR.timeout(PR.request_s(ci, 0.0, b.central.kind)))


def record(rng, serial):
    kw = {n: rng.normal(0.0, 1.0, k) for n, k, _ in R.FIELDS}
    kw["quat"] = np.array([1.0, 0.0, 0.0, 0.0])
    return R.CalRecord(serial=serial, firmware=FWS.VERSION, date=20730, station=4, rig_id=1,
                       plan_hash=0xABCDEF01, t_cal=2800, flags=1, frame=0, **kw)


def logged(fw):
    """Record every charge the board's radio and CPU spend: [(t, C)]."""
    log = []
    orig = fw.pulse

    def pulse(q):
        log.append((fw.t, q))
        orig(q)
    fw.pulse = pulse
    return log


# ================================================================ POWER
def power_checks():
    c_low = Board.HOLD_C * (1.0 - Board.HOLD_TOL)
    pe = Link.PER_EVENT["phone"]
    need = P.holdup_c_needed(Rules.BOUNCE_S, Link.PHONE_CI, pe)
    t_low = P.holdup_s(P.v_hold(), c_low, Link.PHONE_CI, pe)
    check("the hold-up capacitor at its low tolerance carries a %.0f ms bounce at %.2f V (1)"
          % (Rules.BOUNCE_S * 1e3, P.v_hold()), c_low >= need and t_low >= Rules.BOUNCE_S,
          "%.0f uF of %.0f needed; it carries %.1f ms" % (c_low * 1e6, need * 1e6, t_low * 1e3))
    t_bad = P.holdup_s(P.v_hold(), 100e-6, Link.PHONE_CI, pe)
    check("must fail: a 100 uF part (1)", not (100e-6 >= need and t_bad >= Rules.BOUNCE_S),
          "it carries %.1f ms" % (t_bad * 1e3))

    # 2: the bound against the simulated board's windows
    rows, episodes, n_win, mean_ok = [], 0, 0, True
    for central, ci in (("phone", Link.PHONE_CI), ("dongle", Link.CI_MIN)):
        b, serial = fresh(31 if central == "phone" else 32, central, typical=True)
        connect(b, serial, ci)
        stream_on(b, ci)
        b.clock.wait(0.5)
        log = logged(b.fw)
        b.clock.wait(60.0)
        t_b = b.fw.t
        T = np.array([t for t, _ in log])
        Q = np.cumsum([0.0] + [q for _, q in log])
        i_base = b.fw.base_current()
        pe = b.central.per_event
        for w in (Rules.BOUNCE_S, ci, 4.0 * ci, P.holdup_s(P.v_hold(), Board.HOLD_C, ci, pe)):
            # every window that starts at a charge: its pulses inside [t, t + w]
            i = np.flatnonzero(T + w <= t_b)
            j = np.searchsorted(T, T[i] + w, side="right")
            got = Q[j] - Q[i] + i_base * w
            bound = P.window_charge(w, ci, pe)
            over = T[i][got > bound]
            # an episode: overlapping windows over the bound are one burst of losses
            ep = int(np.sum(np.diff(over) > w)) + 1 if len(over) else 0
            episodes += ep
            n_win += int((t_b - T[0]) / w)
            rows.append("%s %.1f ms: worst %.1f of %.1f uC, %d over" % (central, w * 1e3, got.max() * 1e6,
                                                                       bound * 1e6, ep))
            if w == Rules.BOUNCE_S:
                mean_ok &= got.max() <= P.draw(ci).streaming * w
    most = binomial_most(n_win, TAIL, TAIL)
    check("power.window_charge bounds the simulated board's windows but for its tail, both centrals (2)",
          episodes <= most, "%d bursts over it in %d disjoint windows, at most %d allowed; %s"
          % (episodes, n_win, most, "; ".join(rows)))
    check("must fail: the mean draw times a bounce as the bound (2)", not mean_ok, "")

    fits = P.dummy_fits()
    check("the dummy pack, charged to %.2f V a supercapacitor, is safe and fits the cells' place (3)"
          % P.dummy_volts(), all(m >= 0.0 for _, m in fits),
          "; ".join("%s %.2f" % (w, m) for w, m in fits))
    bad = P.dummy_fits(DummyCell.V_RATED)
    check("must fail: charged to its %.1f V rating (3)" % DummyCell.V_RATED, not all(m >= 0.0 for _, m in bad),
          "; ".join("%s %.2f" % (w, m) for w, m in bad if m < 0.0))

    # 4: the limits re-derive the rules
    ah = P.usable_ah()
    u = ah / Cell.CAPACITY_AH
    v_end = Rules.N_CELLS * (P.cell_volts(u) - P.peak_current() * P.cell_ohms(u))
    lim = P.limits()
    ok = (abs(v_end - P.v_min()) <= 1e-9 and abs(P.play_hours(lim.streaming) - Rules.PLAY_H) <= 1e-9 * Rules.PLAY_H
          and abs(ah / lim.sleep - Rules.SLEEP_DAYS * 24.0) <= 1e-9 * Rules.SLEEP_DAYS * 24.0)
    i_sim = MEAN.get(("streaming", "phone"))
    played = P.play_hours(i_sim) if i_sim else 0.0
    # every true current over a limit, at the lowest reading the instrument allows
    over = [lim.streaming * (1.0 + x) for x in np.linspace(1e-6, 0.05, 200)]
    lowest = [i * (1.0 - Smu.ACC) - Z * Smu.FLOOR for i in over]
    leaks = sum(P.accept(m, lim.streaming) for m in lowest)
    check("the limits re-derive the rules; a simulated board plays %.0f h of %.0f; no board over a limit passes (4)"
          % (played, Rules.PLAY_H), ok and played >= Rules.PLAY_H and leaks == 0,
          "usable %.3f Ah to %.4f V; streaming limit %.2f mA, sleep %.1f uA; %d over-limit boards pass"
          % (ah, v_end, lim.streaming * 1e3, lim.sleep * 1e6, leaks))
    naive = sum(m <= lim.streaming for m in lowest)
    check("must fail: the limit read with no guard band (4)", naive > 0, "%d of %d boards over it pass" % (naive, len(over)))


# ================================================================ MODEL
MEAN = {}


def batch_mean(b, seconds, m=20):
    """(mean A, its standard error) of the board's draw over `seconds`, in
    m batches (the batch-means method: the batches are long against an
    interval, so their means are near independent).  The error carries
    the float rounding of differencing the board's running charge."""
    fw = b.fw
    out = []
    for _ in range(m):
        q0, t0 = fw.q, fw.t
        b.clock.wait(seconds / m)
        out.append((fw.q - q0) / (fw.t - t0))
    out = np.array(out)
    rounding = 4.0 * EPS * fw.q / (seconds / m)
    return float(out.mean()), float(out.std(ddof=1) / sqrt(m)) + rounding


def model_checks():
    rows, ok_all, bad_all = [], True, True
    z = zshare(5)
    # advertising, connected, streaming at both intervals, asleep
    b, serial = fresh(51, "dongle", typical=True)
    b.clock.wait(0.2)
    cases = []
    cases.append(("advertising", "dongle") + batch_mean(b, 10.0) + (P.draw().advert_fast, Mcu.Q_ADV, 10.0))
    connect(b, serial, Link.CI_MIN)
    cases.append(("connected", "dongle") + batch_mean(b, 10.0) + (P.draw(Link.CI_MIN).connected, Mcu.Q_EVENT, 10.0))
    stream_on(b, Link.CI_MIN)
    b.clock.wait(0.2)
    m, se = batch_mean(b, 10.0)
    cases.append(("streaming", "dongle", m, se, P.draw(Link.CI_MIN).streaming, Mcu.Q_EVENT + P.q_packet(), 10.0))
    wrong = abs(m - P.draw(Link.PHONE_CI).streaming) <= z * se + (Mcu.Q_EVENT + P.q_packet()) / 10.0
    b.radio.stream(False, 1.0)
    b.radio.sleep(1.0)
    b.clock.wait(0.1)
    cases.append(("asleep", "dongle") + batch_mean(b, 10.0) + (P.draw().sleep, 0.0, 10.0))
    b2, serial2 = fresh(52, "phone", typical=True)
    connect(b2, serial2, Link.PHONE_CI)
    stream_on(b2, Link.PHONE_CI)
    b2.clock.wait(0.2)
    cases.append(("streaming", "phone") + batch_mean(b2, 10.0)
                 + (P.draw(Link.PHONE_CI).streaming, Mcu.Q_EVENT + P.q_packet(), 10.0))
    for state, central, m, se, want, q1, T in cases:
        tol = z * se + q1 / T + 64 * EPS * want
        ok = abs(m - want) <= tol
        ok_all &= ok
        MEAN[(state, central)] = m
        rows.append("%s %s %.4f of %.4f mA (+-%.4f)" % (state, central, m * 1e3, want * 1e3, tol * 1e3))
    check("each state's simulated mean current equals power.draw's (5)", ok_all, "; ".join(rows))
    check("must fail: the dongle's stream judged at the phone's interval (5)", not wrong, "")

    # 6: brown-out on a contact break, streaming to the phone
    tb, lo_ok, hi_ok = [], True, True
    pe = Link.PER_EVENT["phone"]
    t_lo = P.holdup_s(P.v_hold(), Board.HOLD_C, Link.PHONE_CI, pe)
    for i in range(40):
        b, serial = fresh(600 + i, "phone", typical=True)
        connect(b, serial, Link.PHONE_CI)
        stream_on(b, Link.PHONE_CI)
        b.supply.source(P.v_hold(), PR.supply_limit())
        b.clock.wait(0.3 + float(b.rng.uniform(0.0, Link.PHONE_CI)))
        t0 = b.fw.t + 1e-6
        v0 = b.fw.v
        n0 = len(b.fw.browned)
        b.fw.open_circuit(t0, t0 + 0.2)
        b.clock.wait(0.1)
        if len(b.fw.browned) == n0:
            tb.append(np.inf)
            continue
        tb.append(b.fw.browned[n0] - t0)
        t_hi = Board.HOLD_C * (v0 - P.v_min()) / (Mcu.I_IDLE + Imu.I_RUN)
        lo_ok &= tb[-1] >= P.holdup_s(v0, Board.HOLD_C, Link.PHONE_CI, pe) - 1e-9
        hi_ok &= tb[-1] <= t_hi
    tb = np.array(tb)
    check("a contact break browns the board out no sooner than power.holdup_s, no later than its steady draw alone (6)",
          lo_ok and hi_ok and np.all(np.isfinite(tb)),
          "%.2f to %.2f ms after the break; holdup_s %.2f ms; the steady draw alone %.1f ms"
          % (tb.min() * 1e3, tb.max() * 1e3, t_lo * 1e3,
             Board.HOLD_C * (P.v_hold() - P.v_min()) / (Mcu.I_IDLE + Imu.I_RUN) * 1e3))
    t_mean = Board.HOLD_C * (P.v_hold() - P.v_min()) / P.draw(Link.PHONE_CI).streaming
    check("must fail: the hold-up from the mean draw (6)", not np.all(tb >= t_mean),
          "%.2f ms; %d of %d breaks browned out sooner" % (t_mean * 1e3, int(np.sum(tb < t_mean)), len(tb)))

    # 7: the commit, cut at every moment
    def sweep(fw_cls):
        outcomes = []
        rng = np.random.default_rng(7)
        A = R.pack(record(rng, "BAT00007"))
        B = R.pack(record(rng, "BAT00007"))
        span = Mcu.ERASE_PAGE_S + (4 + R.SIZE) / 4 * Mcu.FLASH_WORD_S
        for d in np.linspace(-0.005, span + 0.005, 41):
            b, serial = fresh(7, "dongle", firmware=fw_cls)
            b.board.truth.hold_c = 1e-9            # a cut is a brown-out at once
            connect(b, serial, Link.CI_MIN)
            fw = b.fw
            for data in (A, B):
                t = fw.t
                for off in range(0, R.SIZE, H.record_chunk()):
                    fw.receive(t, bytes([H.REC_STAGE]) + struct.pack("<H", off) + data[off:off + H.record_chunk()])
                fw.receive(t, bytes([H.REC_COMMIT]) + struct.pack("<I", zlib.crc32(data)))
                if data is A:
                    fw.advance(t + span + 0.01)
                else:
                    fw.open_circuit(t + max(d, 1e-6), t + 0.5)
                    fw.advance(t + 0.6)
            got = fw.current_record()
            outcomes.append("A" if got == A else "B" if got == B else "neither")
        return outcomes
    two = sweep(Firmware)

    class OnePage(Firmware):
        PAGES = 1
    one = sweep(OnePage)
    check("power lost anywhere across a commit leaves the old record or the new (7)",
          "neither" not in two and "A" in two and "B" in two,
          "41 cuts: %d old, %d new, %d neither" % (two.count("A"), two.count("B"), two.count("neither")))
    check("must fail: a firmware that writes one page in place (7)", "neither" in one,
          "%d of 41 cuts leave no record" % one.count("neither"))

    # 8: wake on motion
    thr = FWS.WOM_G * G0 / lsb()[1]                      # codes

    def tapped(step):
        n = 20000
        rng = np.random.default_rng(8)
        acc = np.tile([0, 0, int(round(G0 / lsb()[1]))], (n, 1)) + rng.integers(-2, 3, (n, 3))
        acc[12000:, 0] += int(round(step * thr))
        st = SimpleNamespace(acc=acc, gyro=rng.integers(-2, 3, (n, 3)), temp=np.zeros(n, int))
        b, serial = fresh(8, "dongle", source_for=lambda t_on: S.StreamSource(st, t_on + Imu.START_S, 0.0))
        connect(b, serial, Link.CI_MIN)
        b.radio.sleep(1.0)
        b.clock.wait(0.5)
        asleep = b.fw.state == SLEEP
        t_motion = b.fw.src_imu.t(12000)
        heard = b.radio.scan(18.0 - (b.fw.t - b.fw.src_imu.t_first))
        woke = [a for a in heard if a.serial == serial and a.state == H.WOKE]
        if not (woke and getattr(b.fw, "woke", False)):
            return asleep, None
        return asleep, b.fw.adv_since - t_motion
    asleep, lat = tapped(2.0)
    still, _ = fresh(9)
    c, _ = still.fw.src_imu.codes(0, 10000)
    most = float(np.abs(c[:, :3] - c[0, :3]).max()) / thr
    lat_max = 1.0 / Imu.ODR
    check("asleep, a tap of twice WOM_G wakes the board, which advertises WOKE (8)",
          asleep and lat is not None and -1e-12 <= lat <= lat_max and most < 1.0,
          "awake %s ms after the tap's first sample (at most one sample); a still board's 10 s stray %.3f of WOM_G"
          % ("%.3f" % (lat * 1e3) if lat is not None else "never", most))
    _, lat2 = tapped(0.5)
    check("must fail: a tap of half WOM_G (8)", lat2 is None, "")

    # 9: throughput
    rows, ok_all = [], True
    for central, ci, pe, seed in (("dongle", Link.CI_MIN, None, 91), ("phone", Link.PHONE_CI, None, 92),
                                  ("dongle", 0.030, 1, 93)):
        b, serial = fresh(seed, central)
        if pe is not None:
            b.central.per_event = pe
        connect(b, serial, ci)
        stream_on(b, ci)
        raw = b.radio.listen(10.0)
        pk = [H.parse_packet(t, d) for t, d in raw]
        k = np.array([p.k0 for p in pk])
        whole = b.fw.dropped == 0 and len(k) > 1 and np.all(np.diff(k) == H.samples_per_packet())
        margin = P.keeps_up(ci, b.central.per_event)
        rows.append((central, ci, b.central.per_event, whole, margin, b.fw.dropped, len(pk)))
    good = all(r[3] and r[4] > 0.0 for r in rows[:2])
    check("ten seconds' stream to each central arrives whole, as power.keeps_up predicts (9)", good,
          "; ".join("%s %.1f ms %d an event: %s, margin %.0f pk/s, %d dropped of %d"
                    % (c_, ci * 1e3, pe, "whole" if w else "broken", m, d, n) for c_, ci, pe, w, m, d, n in rows[:2]))
    r = rows[2]
    check("must fail: one packet an event at 30 ms (9)", not r[3] and r[4] < 0.0,
          "margin %.0f pk/s, %d dropped" % (r[4], r[5]))


# ================================================================ SELF-TEST
def run_station(seed, fault=None, arg=None):
    rng = np.random.default_rng(seed)
    b = S.Bench(S.draw_board(rng, fault, arg), rng)
    return PR.station_test(PR.Station(b.clock, b.supply, b.swd, b.radio), "BAT%05d" % seed), b


def binomial_most(n, p, tail):
    """The fewest failures k such that more than k in n, each with
    probability p, is rarer than tail."""
    k, cdf = 0, 0.0
    while True:
        cdf += comb(n, k) * p ** k * (1.0 - p) ** (n - k)
        if 1.0 - cdf <= tail:
            return k
        k += 1


def selftest_checks():
    N = 200
    verdicts = [run_station(1000 + i)[0] for i in range(N)]
    bad = [v for v in verdicts if not v.ok]
    most = binomial_most(N, Quality.TEST_ALPHA, TAIL)
    check("%d healthy boards: %d refused, at most %d allowed (10)" % (N, len(bad), most), len(bad) <= most,
          "; ".join("%s: %s" % (v.serial, v.why) for v in bad) or "none")
    z0 = PR.z_test
    try:
        PR.z_test = lambda n=None: 1.0                     # the datasheet's typical spreads read as limits
        loose = sum(not run_station(1000 + i)[0].ok for i in range(N))
    finally:
        PR.z_test = z0
    check("must fail: the datasheet's typical spreads read as limits (10)", loose > most, "%d refused" % loose)

    rows, missed = [], []
    for i, (f, (arg, step)) in enumerate(S.FAULTS.items()):
        v, _ = run_station(100 + i, f, arg)
        got = v.steps[-1].name if v.steps else None
        if v.ok or got != step:
            missed.append("%s: %s at %s" % (f, "passed" if v.ok else "refused", got))
        rows.append("%s at %s" % (f, got))
    check("every one of %d planted faults is refused at its step (11)" % len(S.FAULTS), not missed,
          "; ".join(missed) or "; ".join(rows))
    h0 = PR._holdup
    try:
        PR._holdup = lambda run, st, serial: None
        i = list(S.FAULTS).index("no_holdup")
        v, _ = run_station(100 + i, "no_holdup", S.FAULTS["no_holdup"][0])
    finally:
        PR._holdup = h0
    check("must fail: the hold-up step skipped (11)", v.ok, "the board with no hold-up %s" % ("passes" if v.ok else v.why))

    # 12: time
    good = [v for v in verdicts if v.ok]
    tot = np.array([v.seconds for v in good])
    plan = PR.plan_s()
    se = float(tot.std(ddof=1) / sqrt(len(tot)))
    ok = tot.mean() <= plan + zshare(2) * se and tot.max() <= PR.timeout(plan)
    check("station 2 takes %.2f s on the mean healthy board, planned %.2f (12)" % (tot.mean(), plan), ok,
          "se %.3f s; the longest %.2f s, the overrun %.1f s" % (se, tot.max(), PR.timeout(plan)))
    swd_w = Mcu.SWD_WORD_BITS / min(Mcu.SWD_HZ, PR.Probe.SWD_HZ)
    words = (len(IMG) + 3) // 4
    fast = plan - PR.plan()["flash"] + Mcu.ERASE_ALL_S + 2 * words * swd_w
    check("must fail: a plan that writes flash at the probe's rate (12)",
          not tot.mean() <= fast + zshare(2) * se, "it plans %.2f s" % fast)


# ================================================================ RECORD
class Dropping(H.RadioHAL):
    """A radio whose link drops (the bat walks out of range, the dongle's
    stack hangs) after `after` requests: the next request finds it gone."""

    def __init__(self, inner, after, cut=None):
        self.inner, self.after, self.n, self.cut = inner, after, 0, cut

    def scan(self, seconds):
        return self.inner.scan(seconds)

    def connect(self, addr, interval, timeout):
        return self.inner.connect(addr, interval, timeout)

    def disconnect(self):
        self.inner.disconnect()

    def connected(self):
        return self.inner.connected()

    def listen(self, seconds):
        return self.inner.listen(seconds)

    def request(self, payload, timeout):
        self.n += 1
        if self.n == self.after:
            if self.cut is not None:
                self.cut(payload)
            else:
                self.inner.disconnect()
                raise H.LinkLost("the link dropped")
        return self.inner.request(payload, timeout)


class OnceLink(PR.BleLink):
    """A link that never retries."""

    def _twice(self, what):
        self._connect()
        return what()


class CachingLink(PR.BleLink):
    """A link whose reads answer from what it last wrote."""

    def write(self, data):
        super().write(data)
        self._last = bytes(data)

    def read(self):
        return getattr(self, "_last", None) or super().read()


def record_checks():
    rng = np.random.default_rng(13)
    db = R.FactoryDB()
    b, serial = fresh(130)
    rec = record(rng, serial)
    link = PR.BleLink(b.radio, serial)
    w = R.commit(link, db, rec, np.eye(3), "h")
    back = link.read()
    b.swap("phone")
    p = PR.pair(b.clock, b.radio, serial, db)
    plan = PR.pair_plan_s()
    check("a record committed over the dongle reads back bit for bit; the factory iPhone pairs in %.0f ms of %.0f planned (13)"
          % (p.seconds * 1e3, plan * 1e3), w.writes == 1 and back == R.pack(rec) and p.ok
          and p.seconds <= PR.timeout(plan) / 2.0, "commit %s; pair %s" % (w, p.why or "ok"))
    newer = record(rng, serial)
    db.store(serial, R.pack(newer), np.eye(3), "h2")
    b.radio.disconnect()
    p2 = PR.pair(b.clock, b.radio, serial, db)
    check("must fail: the database holding a newer record than the bat (13)", not p2.ok, p2.why)

    # 14: robustness
    rows, ok_all = [], True
    b, serial = fresh(141, per=0.3)
    try:
        lk = PR.BleLink(b.radio, serial)
        w = R.commit(lk, R.FactoryDB(), record(rng, serial), np.eye(3), "h")
        rows.append("30%% lost: %d write(s), %d retries" % (w.writes, lk.retries))
    except (H.LinkError, R.RecordError) as e:
        ok_all = False
        rows.append("30%% lost: %s" % e)
    b, serial = fresh(142)
    lk = PR.BleLink(Dropping(b.radio, 2), serial)
    try:
        w = R.commit(lk, R.FactoryDB(), record(rng, serial), np.eye(3), "h")
        rows.append("dropped mid-write: %d retry" % lk.retries)
        ok_all &= lk.retries == 1
    except (H.LinkError, R.RecordError) as e:
        ok_all = False
        rows.append("dropped mid-write: %s" % e)
    b, serial = fresh(143)
    b.board.truth.hold_c = 1e-6                            # so the break browns it out inside the commit
    rec143 = record(rng, serial)

    def brown(payload):
        if payload[0] == H.REC_COMMIT:
            t = b.fw.t
            b.fw.open_circuit(t + 1e-6, t + 0.15)
    lk = PR.BleLink(Dropping(b.radio, 1 + -(-R.SIZE // H.record_chunk()), cut=brown), serial)
    try:
        w = R.commit(lk, R.FactoryDB(), rec143, np.eye(3), "h")
        ok_all &= len(b.fw.browned) >= 1 and lk.retries == 1 and b.fw.current_record() == R.pack(rec143)
        rows.append("power lost mid-commit: %d brown-out(s), %d retry" % (len(b.fw.browned), lk.retries))
    except (H.LinkError, R.RecordError) as e:
        ok_all = False
        rows.append("power lost mid-commit: %s" % e)
    check("the commit survives a lossy link, a dropped link and power lost mid-commit (14)", ok_all, "; ".join(rows))
    b, serial = fresh(142)
    try:
        R.commit(OnceLink(Dropping(b.radio, 2), serial), R.FactoryDB(), record(rng, serial), np.eye(3), "h")
        once = True
    except (H.LinkError, R.RecordError):
        once = False
    check("must fail: a link that never retries, under the dropped link (14)", not once, "")

    # 15: the flash itself
    b, serial = fresh(151)
    b.board.truth.faults["flash_bad"] = (1234,)
    db15 = R.FactoryDB()
    try:
        R.commit(PR.BleLink(b.radio, serial), db15, record(rng, serial), np.eye(3), "h")
        stuck_ok = False
    except R.RecordError:
        stuck_ok = db15.latest(serial) is None
    b, serial = fresh(152)
    b.fw.flips = [(777,)]
    rec152 = record(rng, serial)
    w = R.commit(PR.BleLink(b.radio, serial), R.FactoryDB(), rec152, np.eye(3), "h")
    check("a stuck flash bit is refused with nothing stored; a passing flip is rewritten (15)",
          stuck_ok and w.writes == 2 and b.fw.current_record() == R.pack(rec152),
          "stuck: %s; flip: %d writes" % ("refused" if stuck_ok else "stored", w.writes))
    b, serial = fresh(151)
    b.board.truth.faults["flash_bad"] = (1234,)
    db15 = R.FactoryDB()
    try:
        R.commit(CachingLink(b.radio, serial), db15, record(rng, serial), np.eye(3), "h")
        cached = db15.latest(serial) is None
    except R.RecordError:
        cached = True
    check("must fail: a link that answers reads from its own cache (15)", not cached,
          "it stored a record the bat does not hold" if not cached else "")

    # 16: a record gone bad in the field
    b, serial = fresh(160)
    db16 = R.FactoryDB()
    r1, r2 = record(rng, serial), record(rng, serial)
    lk = PR.BleLink(b.radio, serial)
    R.commit(lk, db16, r1, np.eye(3), "h")
    R.commit(lk, db16, r2, np.eye(3), "h")                 # two pages, each a good record
    cur = [i for i, pg in enumerate(b.fw.pages) if pg[4:] == R.pack(r2)][0]
    pg = bytearray(b.fw.pages[cur])
    pg[4 + 100] ^= 0x04                                    # the bat's newer page goes bad
    b.fw.pages[cur] = bytes(pg)
    b.radio.disconnect()
    b.swap("phone")
    bad = PR.pair(b.clock, b.radio, serial, db16)
    b.radio.disconnect()
    R.restore(PR.BleLink(b.radio, serial, Link.PHONE_CI, "phone"), db16, serial)
    b.radio.disconnect()
    good = PR.pair(b.clock, b.radio, serial, db16)
    check("a record gone bad in the field is refused at pairing and restored from the database (16)",
          not bad.ok and good.ok, "refused: %s; then %s" % (bad.why, "paired" if good.ok else good.why))
    # the bat's own CRC alone passes the older page, which is a good record -- of the wrong date
    crc_ok = R.unpack(bad.record).serial == serial if bad.record else False
    check("must fail: pairing that trusts the bat's CRC alone (16)", crc_ok,
          "the bat offers its older record, whose CRC holds")


# ================================================================ CLOCK
def recovered(seed, central):
    """[(offset error s, rate error, packets)] of one board's clock on
    `central`: the envelope, least squares, the envelope on a quarter of
    the packets."""
    ci = Link.CI_MIN if central == "dongle" else Link.PHONE_CI
    b, serial = fresh(seed)
    if central != "dongle":
        b.swap(central)
    connect(b, serial, ci)
    stream_on(b, ci)
    raw = b.radio.listen(SY.listen_s(ci, central))
    pk = [H.parse_packet(t, d) for t, d in raw]
    src, c = b.fw.src_imu, b.central
    out = []
    for fn, use in ((SY.recover, pk), (SY.naive, pk), (SY.recover, pk[:max(2, len(pk) // 4)])):
        cm = fn(use, central)
        e = float(cm.at(cm.k0) - c.station(src.t(cm.k0)))
        r = cm.rho / ((1.0 + src.eps) * c.rate) - 1.0
        out.append((e, r, len(use)))
    return out


def clock_checks():
    N = 100
    zq = TAIL / 4.0                                       # four comparisons
    rows, ok_all, ls_bad, few_bad = [], True, True, True
    for central, ci in (("dongle", Link.CI_MIN), ("phone", Link.PHONE_CI)):
        res = [recovered(1700 + i + (0 if central == "dongle" else 5000), central) for i in range(N)]
        e = np.array([r[0][0] for r in res])
        rr = np.array([r[0][1] for r in res])
        n = int(np.median([r[0][2] for r in res]))
        k = sqrt(chi2_quantile(N, zq) / N)
        rms, rrms = float(np.sqrt(np.mean(e ** 2))), float(np.sqrt(np.mean(rr ** 2)))
        model = sqrt(SY.envelope_rms(ci, central, n) ** 2 + Est.HOST_FLOOR_SD ** 2)
        slope = SY.slope_sd(ci, central, n)
        ok = rms <= Quality.SYNC_S * k and rms <= model * k and rrms <= slope * k
        ok_all &= ok
        rows.append("%s, %d packets: offset rms %.3f ms (budget %.2f, model %.3f, bound x%.2f), rate rms %.4f %% (bound %.4f)"
                    % (central, n, rms * 1e3, Quality.SYNC_S * 1e3, model * 1e3, k, rrms * 100, slope * 100))
        e_ls = np.array([r[1][0] for r in res])
        ls_bad &= float(np.sqrt(np.mean(e_ls ** 2))) > Quality.SYNC_S * k
        few = np.array([r[2][0] for r in res])
        few_bad &= float(np.sqrt(np.mean(few ** 2))) > Quality.SYNC_S * k
        rows.append("least squares %.2f ms; a quarter of the packets %.2f ms"
                    % (float(np.sqrt(np.mean(e_ls ** 2))) * 1e3, float(np.sqrt(np.mean(np.square(few)))) * 1e3))
    check("the bat's clock recovered to within Quality.SYNC_S on both centrals (17)", ok_all, "; ".join(rows))
    check("must fail: the least-squares clock (17)", ls_bad, "")
    check("must fail: a quarter of the packets (17)", few_bad, "")

    # 18: wraps and resets, on synthetic packets from a known line
    rng = np.random.default_rng(18)
    spp = H.samples_per_packet()
    ci = Link.CI_MIN

    def packets(k_first, n, t0, rate):
        k = k_first + spp * np.arange(n) + spp - 1
        ready = t0 + rate * (k - k_first)
        t = ready + np.mod(rng.uniform(0, ci) - ready, ci) + rng.exponential(Est.HOST_SPREAD_S["dongle"], n) \
            + SY.floor("dongle")
        return [H.Packet(float(ti), int((ki - spp + 1) % SY.WRAP), 0, np.zeros((spp, 6), int)) for ti, ki in zip(t, k)]
    rate = (1.0 + 3e-3) / Imu.ODR
    base = SY.WRAP - 60 * spp
    wrapped = packets(base, 120, 10.0, rate)
    plain = [H.Packet(p.t, p.k0 + (SY.WRAP if p.k0 < base else 0), 0, p.codes) for p in wrapped]
    cw, cp = SY.recover(wrapped, "dongle"), SY.recover(plain, "dongle")
    first = packets(5000, 50, 2.0, rate)
    second = packets(0, 100, 5.0, rate)
    cr, c2 = SY.recover(first + second, "dongle"), SY.recover(second, "dongle")
    ok = (abs(cw.t0 - cp.t0) <= 1e-9 and abs(cw.rate - cp.rate) <= 1e-15 and abs(cr.t0 - c2.t0) <= 1e-12
          and abs(cr.rate - c2.rate) <= 1e-18 and cw.n == 120 and cr.n == 100)
    check("counters that wrap, and a bat that resets mid-stream, recover the last run's clock (18)", ok,
          "wrap: t0 %.3g s and rate %.3g apart; reset: the last %d of %d packets" % (abs(cw.t0 - cp.t0),
                                                                               abs(cw.rate - cp.rate), cr.n,
                                                                               len(first) + len(second)))
    k_raw = np.array([p.k0 + spp - 1 for p in wrapped], float)
    t_raw = np.array([p.t for p in wrapped])
    t0r, rr_, k0r = SY.envelope(k_raw, t_raw)
    err = abs((t0r + rr_ * (cp.k0 - k0r) - cp.floor) - cp.at(cp.k0))
    check("must fail: the counters taken raw (18)", err > Quality.SYNC_S, "%.3g s off" % err)

    # 19: the hull against a brute-force linear program
    worst, first_bad = 0.0, 0
    for trial in range(100):
        n = 25
        k = np.sort(rng.choice(5000, n, replace=False)).astype(float)
        t = 0.3 + k * 1.001e-3 + rng.exponential(2e-3, n)
        t0, rate, k0 = SY.envelope(k, t)
        best = None
        for i in range(n):
            for j in range(i + 1, n):
                s = (t[j] - t[i]) / (k[j] - k[i])
                line = t[i] + s * (k - k[i])
                if np.all(t - line >= -64 * EPS * np.abs(t)):
                    v = t[i] + s * (k.mean() - k[i])
                    if best is None or v > best[0]:
                        best = (v, s)
        worst = max(worst, abs(best[0] - t0) / abs(t0), abs(best[1] - rate) / abs(rate))
        # the wrong edge: the hull's first
        x, y = k - k.mean(), t - t.mean()
        h = SY.lower_hull(x, y)
        a, bb = h[0], h[1]
        v_first = t.mean() + y[a] - (y[bb] - y[a]) / (x[bb] - x[a]) * x[a]
        first_bad += abs(v_first - best[0]) > 64 * EPS * abs(best[0])
    check("the lower hull's spanning edge is the linear program's optimum on 100 random sets (19)",
          worst <= 1e3 * EPS, "worst relative gap %.2g" % worst)
    check("must fail: the hull's first edge (19)", first_bad > 50, "%d of 100 differ" % first_bad)


# ================================================================ CONTRACTS
PRODUCTION = ("board/hal.py", "board/power.py", "board/sync.py", "board/procedures.py",
              "phone/camera.py", "phone/hal.py", "phone/app.py", "phone/test.py")
TRUTH = ("batline.board.firmware", "batline.board.radio", "batline.board.sim", "batline.imu.scene",
         "batline.imu.model", "batline.imu.sim", "batline.imu.judge", "batline.phone.sim", "batline.phone.picture",
         "batline.swing.sim")


def package_of(rel):
    return "batline." + os.path.dirname(rel).replace("/", ".")


def imports_truth(src, package):
    import ast
    hits = []

    def truth(full):
        return full.split(".")[0] == "mujoco" or any(full == t or full.startswith(t + ".") for t in TRUTH)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            hits += [a.name for a in node.names if truth(a.name)]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                full = ".".join(parts[: len(parts) - (node.level - 1)] + ([node.module] if node.module else []))
            else:
                full = node.module or ""
            if truth(full):
                hits.append(full)
            hits += [full + "." + a.name for a in node.names if truth(full + "." + a.name)]
        elif isinstance(node, ast.Name) and node.id in ("importlib", "__import__"):
            hits.append(node.id)
    return hits


def contract_checks():
    b, _ = fresh(200)
    missing = []
    for obj, c in ((b.clock, H.ClockHAL), (b.supply, H.SupplyHAL), (b.swd, H.SwdHAL), (b.radio, H.RadioHAL)):
        missing += H.audit(obj, (c,))
    pb = PS.ready(product.design(KID), 201)
    for obj, c in ((pb.station.gimbal, PH.GimbalHAL), (pb.station.camera, PH.CameraHAL),
                   (pb.station.clock, H.ClockHAL), (pb.station.radio, H.RadioHAL)):
        missing += H.audit(obj, (c,))
    check("the simulated benches meet the four board HALs and the iPhone's two (20)", not missing,
          ", ".join(missing) or "all")

    class Half:
        def source(self, v, i):
            pass

        def off(self):
            pass

        def mean_current(self, s):
            return 0.0
    check("must fail: a supply missing limited() (20)", H.audit(Half(), (H.SupplyHAL,)) == ["SupplyHAL.limited"], "")

    root = os.path.join(HERE, "..", "..", "batline")
    hits = []
    for rel in PRODUCTION:
        hits += ["%s: %s" % (rel, h) for h in imports_truth(open(os.path.join(root, rel)).read(), package_of(rel))]
    sim_dir = os.path.abspath(os.path.join(HERE, "..", ".."))
    code = ("import sys; sys.path.insert(0, %r); import importlib; importlib.import_module(sys.argv[1]); "
            "print(' '.join(sorted(k for k in sys.modules if k == 'mujoco' or k in %r)))" % (sim_dir, TRUTH))
    loaded = []
    for rel in PRODUCTION:
        mod = "batline." + rel[:-3].replace("/", ".")
        p = subprocess.run([sys.executable, "-c", code, mod], capture_output=True, text=True)
        got = p.stdout.strip() if p.returncode == 0 else "failed to import: %s" % p.stderr.strip()[-200:]
        if got:
            loaded.append("%s: %s" % (mod, got))
    check("no production module of the board or the iPhone imports or loads MuJoCo or a TRUTH module (21)",
          not hits and not loaded, "; ".join(hits + loaded) or "%d modules, none" % len(PRODUCTION))
    sims = {rel: imports_truth(open(os.path.join(root, rel)).read(), package_of(rel))
            for rel in ("board/sim.py", "phone/sim.py")}
    check("must fail: the same scan on board/sim.py and phone/sim.py (21)", all(sims.values()),
          "; ".join("%s sees %s" % (rel, ", ".join(h) or "nothing") for rel, h in sims.items()))


# ================================================================ THE FACTORY IPHONE
PHONE_N = 30


def binomial_least(n, p, tail):
    """The most successes k such that fewer than k in n, each with
    probability p, is rarer than tail."""
    k, cdf = 0, 0.0
    while True:
        pk = comb(n, k) * p ** k * (1.0 - p) ** (n - k)
        if cdf + pk > tail:
            return k
        cdf += pk
        k += 1


def phone_checks():
    d = product.design(KID)
    sp = PT.slow_swing(d)
    bud = PT.budget(d)
    anchor = study.Anchor("none", "stance", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # 22: the truth the budget runs on, with no error at all
    def bare(k_s):
        tr = PT.GimbalTruth(sp)
        if k_s is not None:
            tr.k_s = tr.k_top = k_s
        out = study.run(tr, study.CAL_WIDE, anchor, N=1, seed=1, only=set(), l_imu=sp.r_imu, l_ss=sp.geom.ss,
                        r_mount=sp.R_imu, g=sp.g, stance_g=True)
        e = out["e"][0, [0, 2]]
        return e, float(e @ np.linalg.solve(bud.raw / bud.N, e))
    lim = -2.0 * log(TAIL)                     # chi-square on 2 dof, exceeded with probability TAIL: exact
    e0, d0 = bare(None)
    check("with no error at all the budget's study lands on the drawn sweet spot (22)", d0 <= lim,
          "%.3f mm right, %.3f up: d^2 %.2g against its trials' standard error, bound %.1f; the swing turns "
          "%.1f deg (%s; stopped: %s)"
          % (e0[0] * 1e3, e0[1] * 1e3, d0, lim, np.degrees(sp.q_b[1] - sp.q_a[1]),
             ", ".join("%.1f deg %.2f" % (np.degrees(a), p) for a, p in sp.tried), sp.stop))
    e1, d1 = bare(int(round(sp.t_stance / study.DT)))
    check("must fail: the anchor at the stance's nearest sample (22)", d1 > lim,
          "%.3f mm right, %.3f up: d^2 %.0f" % (e1[0] * 1e3, e1[1] * 1e3, d1))

    # 23: healthy bats
    good = []
    for i in range(PHONE_N):
        pb = PS.ready(d, 2300 + i)
        good.append(PT.iphone_test(pb.station, pb.serial, pb.db, d))
    refused = [v for v in good if not v.ok]
    E = np.array([v.step("sweet spot").data["e"] for v in good if v.step("sweet spot")])
    D2 = np.array([v.step("sweet spot").data["d2"] for v in good if v.step("sweet spot")])
    most = binomial_most(PHONE_N, Quality.TEST_ALPHA, TAIL / 2.0)          # two tests share the tail
    n = len(D2)
    bound = chi2_quantile(2 * n, TAIL / 2.0)
    check("%d calibrated bats: %d refused, at most %d allowed; their errors within the budget (23)"
          % (PHONE_N, len(refused), most), len(refused) <= most and n == PHONE_N and D2.sum() <= bound,
          "sum d^2 %.1f of %.1f; rms %.2f mm right, %.2f up, the budget's %.2f and %.2f; %s"
          % (D2.sum(), bound, *np.sqrt(np.mean(E ** 2, 0)) * 1e3, *np.sqrt(np.diag(bud.raw)) * 1e3,
             "; ".join("%s: %s" % (v.serial, v.why) for v in refused) or "none refused"))
    base = PT._base(d, None)
    sp0 = PT._plan_for(base, sp.tried[0][0])
    bud0 = PT._budget_of(*PT._errors(sp0, PT.GimbalTruth(sp0), study.CAL_WIDE, study.CAL_WIDE.mount_deg * study.D2R,
                                     PT.BUDGET_TRIALS, PT.BUDGET_SEED))
    D2w = np.einsum("ni,ij,nj->n", E, np.linalg.inv(bud0.S), E)
    check("must fail: the same bats held to the game-length swing's budget (23)", D2w.sum() > bound,
          "sum d^2 %.0f; that budget's rms %.2f mm right, %.2f up" % (D2w.sum(), *np.sqrt(np.diag(bud0.raw)) * 1e3))

    # 24: the datasheet's record in the app
    bad = []
    for i in range(PHONE_N):
        pb = PS.ready(d, 2400 + i)
        T_die = float(PS.MD.die_temperature(pb.bench.fw.src_imu.pw, pb.bench.t()))
        bad.append(PT.iphone_test(pb.station, pb.serial, pb.db, d,
                                  app_record=PS.datasheet_record(sp, pb.serial, T_die)))
    caught = sum(not v.ok for v in bad)
    least = binomial_least(PHONE_N, sp.power, TAIL)
    why = {}
    for v in bad:
        if not v.ok:
            w = v.steps[-1].name
            why[w] = why.get(w, 0) + 1
    check("%d bats fused with the datasheet's record: %d refused, at least %d wanted (24)" % (PHONE_N, caught, least),
          caught >= least, "designed to refuse %.0f %%; refused at %s"
          % (sp.power * 100, ", ".join("%s %d" % kv for kv in why.items()) or "none"))
    check("must fail: the calibrated bats of 23 (24)", len(refused) < least, "%d refused" % len(refused))

    # 25: time
    secs = np.array([v.seconds for v in good if v.ok])
    plan = PT.plan_s(d)
    se = float(secs.std(ddof=1) / sqrt(len(secs)))
    ok = secs.mean() <= plan + zshare(2) * se and secs.max() <= PR.timeout(plan)
    check("station 4's iPhone test takes %.2f s on the mean healthy bat, planned %.2f (25)" % (secs.mean(), plan), ok,
          "se %.3f s; the longest %.2f s, the overrun %.1f s" % (se, secs.max(), PR.timeout(plan)))
    short = plan - PT.plan(d)["pairs"]
    check("must fail: a plan without the pairing (25)", not secs.mean() <= short + zshare(2) * se,
          "it plans %.2f s" % short)


# ================================================================ MAIN
def main():
    t0 = time.time()
    for fn in (model_checks, power_checks, selftest_checks, record_checks, clock_checks, contract_checks,
               phone_checks):
        t = time.time()
        fn()
        if VERBOSE:
            print("  (%s: %.0f s)" % (fn.__name__, time.time() - t))

    def number(row):
        m = re.search(r"\((\d+)\)$", row[0])
        return int(m.group(1)) if m else 99
    RESULTS.sort(key=number)
    bad = sum(1 for _, ok, _, _ in RESULTS if not ok)
    for nm, ok, det, secs in RESULTS:
        print("  %s %5.1fs  %s%s" % ("ok  " if ok else "FAIL", secs, nm,
                                     ("  [%s]" % det) if det and (VERBOSE or not ok) else ""))
    print("check_link: %d checks, %d failed, %.0f s" % (len(RESULTS), bad, time.time() - t0))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
