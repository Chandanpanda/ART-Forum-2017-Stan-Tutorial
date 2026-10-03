"""The board's power, derived from the parts' datasheets and the product's
rules: what it draws in each state, what the cells give it, how long its
capacitor carries it through an open contact, and the limits the station
holds a board's currents to.  Production, standard library only.

Nothing here is a chosen number.  The board runs straight from two cells
(README), so the least supply it works on is the larger of its two parts'
least, and every limit follows from that, the cells' curve and the rules
(spec.Rules: PLAY_H, SLEEP_DAYS, BOUNCE_S, HOLD_LIFE).
"""
from bisect import bisect_left
from dataclasses import dataclass
from math import ceil, floor, erfc, sqrt, comb

from ..spec import Cell, Board, Mcu, Imu, Link, Rules, Firmware, DummyCell, Smu, Module
from . import hal as H


# ================================================================ THE CELLS
def cell_volts(used):
    """V, one cell's open-circuit voltage with `used` of its capacity gone."""
    xs = [u for u, _ in Cell.CURVE]
    ys = [v for _, v in Cell.CURVE]
    u = min(max(float(used), xs[0]), xs[-1])
    i = max(1, bisect_left(xs, u))
    f = (u - xs[i - 1]) / (xs[i] - xs[i - 1])
    return ys[i - 1] + f * (ys[i] - ys[i - 1])


def cell_used_at(volts):
    """The share of capacity used when a cell's voltage has fallen to `volts`."""
    pts = list(Cell.CURVE)
    if volts >= pts[0][1]:
        return 0.0
    for (u0, v0), (u1, v1) in zip(pts, pts[1:]):
        if v1 <= volts <= v0:
            return u0 + (v0 - volts) / (v0 - v1) * (u1 - u0)
    return 1.0


def cell_ohms(used):
    """Its internal resistance, rising from fresh to the end of its curve."""
    return Cell.R_FRESH + min(max(used, 0.0), 1.0) * (Cell.R_END - Cell.R_FRESH)


def v_min():
    """V, the least supply the board works on: below either part's least,
    the stream is no longer the part's datasheet, so it counts as a
    brown-out whether or not the radio's own reset has tripped."""
    return max(Mcu.VDD_MIN, Imu.VDD_MIN, Mcu.V_BOR)


def v_fresh():
    """V, the supply with fresh cells: what the datasheets' currents are near."""
    return Rules.N_CELLS * cell_volts(0.0)


def v_hold():
    """V, the supply at the end of the life over which the board must ride
    through Rules.BOUNCE_S (Rules.HOLD_LIFE): where its hold-up is least."""
    return Rules.N_CELLS * cell_volts(Rules.HOLD_LIFE)


def usable_ah():
    """Ah the board gets from fresh cells: down to the voltage, under its
    heaviest draw, at which the supply meets v_min."""
    i_pk = peak_current()
    # solve N (V(u) - I R(u)) = v_min for u by bisection on the monotone curve
    lo, hi = 0.0, 1.0
    for _ in range(60):
        u = 0.5 * (lo + hi)
        v = Rules.N_CELLS * (cell_volts(u) - i_pk * cell_ohms(u))
        lo, hi = (u, hi) if v > v_min() else (lo, u)
    return lo * Cell.CAPACITY_AH


# ================================================================ THE DRAW
def air_s(payload, phy=Link.PHY_BPS):
    """s on air for one link-layer packet carrying `payload` bytes."""
    return (payload + Link.LL_OVERHEAD) * 8.0 / phy


def q_air(payload):
    """C one notification of `payload` bytes adds to a connection event on
    air: the measured 20-byte cost, plus each byte more at its air time."""
    return Mcu.Q_NOTIFY + max(0, payload - 20) * 8.0 / Link.PHY_BPS * Mcu.I_TX


def q_cpu():
    """C the CPU spends making one stream packet."""
    return Firmware.CPU_PACKET_S * Mcu.I_CPU


def q_packet(payload=None):
    """C one stream packet costs: made, then sent."""
    payload = H.notification_bytes() if payload is None else payload
    return q_air(payload) + q_cpu()


def packet_rate():
    """Stream packets a second at the IMU's rate."""
    return Imu.ODR / H.samples_per_packet()


@dataclass(frozen=True)
class Draw:
    """A, the board's mean draw in each state, and in the heaviest moment."""
    sleep: float
    advert_fast: float
    advert_slow: float
    connected: float
    streaming: float
    peak: float


def draw(ci=Link.PHONE_CI):
    """The board's mean current per state, from its parts' datasheets, with
    a central at connection interval `ci`.  An advertising interval is its
    setting plus the Core spec's random delay (mean half of ADV_JITTER); a
    stream packet the central missed is sent again, 1 / (1 - PER) sends a
    packet on average."""
    base = Mcu.I_IDLE
    sleep = base + Imu.I_WOM
    adv_f = sleep + Mcu.Q_ADV / (Link.ADV_FAST_S + Link.ADV_JITTER / 2.0)
    adv_s = sleep + Mcu.Q_ADV / (Link.ADV_SLOW_S + Link.ADV_JITTER / 2.0)
    conn = sleep + Mcu.Q_EVENT / ci
    sends = 1.0 / (1.0 - Link.PER)
    stream = (base + Imu.I_RUN + Mcu.Q_EVENT / ci
              + packet_rate() * (sends * q_air(H.notification_bytes()) + q_cpu()))
    return Draw(sleep, adv_f, adv_s, conn, stream, peak_current())


def keeps_up(ci, per_event):
    """Packets a second the link carries beyond what the stream makes (a
    margin; below zero the bat's queue fills and drops): per_event slots
    an interval, each needing both its packets through."""
    return per_event * (1.0 - Link.PER) ** 2 / ci - packet_rate()


def peak_current():
    """A at the heaviest instant: the radio sending, the CPU awake, the IMU running."""
    return Mcu.I_TX + Mcu.I_CPU + Imu.I_RUN


def losses(n_ev, ci):
    """Packets lost on air that can bear on a window holding n_ev events:
    the most that more of is rarer than Module.Z_SIGMAS's one-sided tail,
    binomially over the window's events and the one before (whose losses
    are resent inside it), each carrying the stream's packets for an
    interval and the closing exchange, each exchange lost if either of its
    two packets is (Link.PER apiece)."""
    n = (n_ev + 1) * (floor(packet_rate() * ci) + 2)
    p = 1.0 - (1.0 - Link.PER) ** 2
    tail = 0.5 * erfc(Module.Z_SIGMAS / sqrt(2.0))
    k, cdf = 0, 0.0
    while True:
        cdf += comb(n, k) * p ** k * (1.0 - p) ** (n - k)
        if 1.0 - cdf <= tail:
            return k
        k += 1


def window_charge(t, ci=Link.PHONE_CI, per_event=None):
    """C the streaming board can draw in any `t` seconds: its steady draw;
    every connection event that can fall inside t; the packets the stream
    makes inside t; and, on air, every packet those events can carry --
    each made since the interval before the first of them -- with every
    loss that can bear on the window sent again (losses()).  A bound but
    for a Module.Z_SIGMAS tail, which check_link holds the simulated board
    under."""
    per_event = Link.PER_EVENT["phone"] if per_event is None else per_event
    rate = packet_rate()
    n_ev = floor(t * (1.0 + Link.CLOCK_PPM * 1e-6) / ci) + 1   # event starts a closed window of t can hold,
    #                                                          the central's clock at its fastest
    made = floor(rate * t) + 1                   # packets made in it
    sent = min(n_ev * per_event, floor(rate * n_ev * ci) + 1 + losses(n_ev, ci))
    return ((Mcu.I_IDLE + Imu.I_RUN) * t + n_ev * Mcu.Q_EVENT + sent * q_air(H.notification_bytes())
            + made * q_cpu())


def holdup_s(v0=None, c=None, ci=Link.PHONE_CI, per_event=None):
    """s the board, streaming, runs on its capacitor alone from `v0` before
    its supply falls to v_min: the least t whose window_charge spends
    C (v0 - v_min)."""
    v0 = v_hold() if v0 is None else v0
    c = Board.HOLD_C if c is None else c
    q = c * (v0 - v_min())
    if q <= 0.0:
        return 0.0
    lo, hi = 0.0, 1.0
    if window_charge(hi, ci, per_event) < q:
        return hi
    for _ in range(60):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if window_charge(m, ci, per_event) < q else (lo, m)
    return lo


def holdup_c_needed(t=Rules.BOUNCE_S, ci=Link.PHONE_CI, per_event=None):
    """F, the least capacitor that carries the streaming board through t from v_hold."""
    return window_charge(t, ci, per_event) / (v_hold() - v_min())


# ================================================================ THE LIMITS
@dataclass(frozen=True)
class Limits:
    """What the station holds a board's currents to, A, and why."""
    streaming: float           # the cells last Rules.PLAY_H streaming
    advertising: float         # an advertising board is a playing one: the same
    sleep: float               # asleep, fresh cells last Rules.SLEEP_DAYS
    guard: float               # each reading must sit under limit (1 - guard), less its floor (accept)


def limits():
    ah = usable_ah()
    stream = ah / Rules.PLAY_H
    return Limits(streaming=stream, advertising=stream, sleep=ah / (Rules.SLEEP_DAYS * 24.0), guard=Smu.ACC)


def accept(measured, limit):
    """Guard-banded acceptance (ISO 14253-1): a reading conforms only if it
    sits inside the limit by the instrument's own error -- Smu.ACC of the
    reading and its floor at Module.Z_SIGMAS -- so a board whose true
    current is over the limit cannot pass on a low reading."""
    return measured + Module.Z_SIGMAS * Smu.FLOOR <= limit * (1.0 - Smu.ACC)


def play_hours(i_stream):
    return usable_ah() / i_stream


# ================================================================ THE DUMMY PACK
def dummy_volts():
    """V each of the swing rig's supercapacitors is charged to: the cells'
    voltage at the end of the hold-up's promise, where an open contact
    costs most."""
    return cell_volts(Rules.HOLD_LIFE)


def dummy_fits(v=None):
    """[(what, margin)]: the dummy pack, each supercapacitor charged to `v`,
    is safe for the board and itself and stands in for the cells it
    replaces."""
    v = dummy_volts() if v is None else v
    return [("each supercapacitor under its rating, V", DummyCell.V_RATED - v),
            ("the pack under the board's most, V", Mcu.VDD_MAX - Rules.N_CELLS * v),
            ("the pack above the board's least, V", Rules.N_CELLS * v - v_min()),
            ("a supercapacitor inside an AA's diameter, mm", Cell.D_MAX - DummyCell.D),
            ("...and its length, mm", Cell.L_MAX - DummyCell.L),
            ("...and as long as the shortest AA, mm", DummyCell.L - Cell.L_MIN)]


def dummy_droop(seconds, i=None):
    """V the pack falls by carrying the streaming board for `seconds`."""
    i = draw().streaming if i is None else i
    return i * seconds / (DummyCell.C / Rules.N_CELLS)
