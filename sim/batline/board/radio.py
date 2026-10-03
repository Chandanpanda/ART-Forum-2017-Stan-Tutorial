"""The radio link as a channel model, and the central at its far end.
TRUTH: simulation and checks only.

The plan's link: connection interval, latency spread, packet loss,
throughput, and the offset and drift between the bat's clock and the
station's.  Modelled at the level of the link layer's connection events,
because that is where each of those comes from:

  * the central opens an event every interval on ITS clock, which runs
    off true time by up to Link.CLOCK_PPM;
  * in an event the two sides take turns, up to the central's packets an
    event (Link.PER_EVENT); every packet is lost on air with Link.PER, and
    the event closes on a loss (the Core spec closes it on two running;
    one is the harsher reading);
  * a packet not acknowledged stays at the head of its queue and goes in
    the next event, so nothing is lost to the application -- latency is
    what a loss costs -- until nothing gets through for the supervision
    timeout and the link drops;
  * a delivered notification crosses the central's host stack: a floor,
    which this central unit draws about its model's characterised value,
    and an exponential spread above it (spec.Est.HOST_*).

What a packet's air time and its charge are comes from power.py, the same
derivation the station's limits rest on.
"""
import numpy as np

from ..spec import Link, Est, Mcu
from . import hal as H
from . import power as P
from .firmware import ADV


class Central:
    """A Bluetooth LE central (the station's dongle, the factory iPhone)
    against one simulated board's firmware."""

    def __init__(self, kind, fw, rng, per=None, t0=0.0):
        self.kind = kind
        self.fw = fw
        self.rng = rng
        self.per = Link.PER if per is None else per
        # its clock: station time s = s0 + rate t
        self.rate = 1.0 + float(rng.uniform(-Link.CLOCK_PPM, Link.CLOCK_PPM)) * 1e-6
        self.s0 = float(rng.uniform(0.0, 1000.0))
        self.floor = Est.HOST_FLOOR_S[kind] + float(rng.normal(0.0, Est.HOST_FLOOR_SD))
        self.spread = Est.HOST_SPREAD_S[kind]
        self.per_event = Link.PER_EVENT[kind]
        self.t = float(t0)
        self.conn = None           # (next anchor, interval), true seconds
        self.last_ok = None
        self.txq = []
        self.notes = []            # [(t_station, bytes)] stream notifications
        self.answers = []          # [(t_station, bytes)] control answers
        self.lost = 0              # packets lost on air
        self.events = 0

    # ------------------------------------------------------------ clocks
    def station(self, t):
        return self.s0 + self.rate * t

    def true(self, s):
        return (s - self.s0) / self.rate

    def now(self):
        return self.station(self.t)

    # ------------------------------------------------------------ time
    def run(self, t_end, stop=None):
        """Advance true time to t_end, holding connection events on the way;
        returns early once stop() holds after an event."""
        while self.conn is not None and self.conn[0] <= t_end:
            e, ci = self.conn
            self.t = max(self.t, e)
            self._event(e)
            if self.conn is not None:
                self.conn = (e + ci, ci)
            if stop is not None and stop():
                return True
        self.fw.advance(t_end)
        self.t = max(self.t, t_end)
        return bool(stop()) if stop is not None else False

    def _deliver(self, t_air_end, kind, data):
        t_arr = t_air_end + self.floor + float(self.rng.exponential(self.spread))
        (self.answers if kind == "answer" else self.notes).append((self.station(t_arr), data))

    def _event(self, e):
        fw = self.fw
        fw.advance(e)
        self.events += 1
        if not fw.alive():
            if fw.terminated:                                 # the bat said goodbye
                fw.terminated = False
                self.conn = None
            else:
                self._supervise(e)
            return
        fw.pulse(Mcu.Q_EVENT * fw._i("radio"))
        t = e
        if not fw.alive():                                # the event's own charge browned it out
            self._supervise(e)
            return
        for slot in range(self.per_event):
            c_has = bool(self.txq)
            p_has = fw.has_tx(t)
            if slot > 0 and not (c_has or p_has):
                break
            c_len = len(self.txq[0]) if c_has else 0
            t += P.air_s(c_len) + Link.T_IFS
            if self.rng.random() < self.per:              # the bat missed the central's packet
                self.lost += 1
                break
            if c_has:
                fw.receive(t, self.txq.pop(0))
            fw.advance(t)                                 # what became ready before its slot goes in it
            if not fw.alive():
                break
            head = fw.peek_tx(t)
            p_len = len(head[2]) if head is not None else 0
            if head is not None:
                fw.pulse(P.q_air(p_len) * fw._i("radio"))
                if not fw.alive():                        # browned out sending it: nothing arrives
                    break
            t += P.air_s(p_len) + Link.T_IFS
            if self.rng.random() < self.per:              # the central missed the bat's answer
                self.lost += 1
                break
            self.last_ok = t
            if head is not None:
                kind, data = fw.pop_tx(t)
                self._deliver(t, kind, data)
            if not fw.alive():
                break
        self._supervise(e)

    def _supervise(self, e):
        if self.last_ok is not None and e - self.last_ok > Link.SUPERVISION / self.rate:
            self.conn = None
            self.fw.disconnect(e)

    # ------------------------------------------------------------ the HAL's calls
    def scan(self, seconds):
        t1 = self.t + seconds / self.rate
        self.fw.adverts = [a for a in self.fw.adverts if a[0] > self.t]
        self.run(t1)
        heard = []
        for ta, data in self.fw.adverts:
            if self.t - seconds / self.rate <= ta <= self.t and self.rng.random() >= self.per:
                heard.append(H.parse_advert(self.station(ta), H.parse_advert(0, "", data).serial, data))
        self.fw.adverts = []
        return heard

    def connect(self, serial, interval, timeout):
        if interval < Link.CI_MIN - 1e-12 or abs(interval / Link.CI_STEP - round(interval / Link.CI_STEP)) > 1e-6:
            raise H.LinkError("a connection interval of %.4f s is not one the Core spec allows" % interval)
        if self.kind == "phone" and (interval < Link.PHONE_CI - 1e-12
                                     or abs(interval / Link.PHONE_CI - round(interval / Link.PHONE_CI)) > 1e-6):
            raise H.LinkError("iOS refuses a %.4f s interval (Apple: at least 15 ms, a multiple of it)" % interval)
        t_dead = self.t + timeout / self.rate
        self.fw.adverts = []
        while self.t < t_dead:
            fw = self.fw
            nxt = fw.adv_t if fw.state == ADV and fw.adv_t is not None and fw.adv_t > self.t else self.t + Link.ADV_FAST_S
            self.run(min(t_dead, nxt + 1e-9))
            for ta, data in self.fw.adverts:
                if H.parse_advert(0, "", data).serial == serial and self.rng.random() >= self.per:
                    e0 = ta + Link.CI_STEP + float(self.rng.uniform(0.0, interval / self.rate))
                    self.fw.adverts = []
                    if not self.fw.connect(e0):
                        return False
                    self.t = max(self.t, e0)
                    self.conn = (e0, interval / self.rate)
                    self.last_ok = e0
                    self.txq = []
                    self.notes = []
                    self.answers = []
                    return True
            self.fw.adverts = []
        return False

    def disconnect(self):
        if self.conn is not None:
            self.fw.disconnect(self.t)
        self.conn = None

    def connected(self):
        return self.conn is not None

    def request(self, payload, timeout):
        if self.conn is None:
            raise H.LinkLost("not connected")
        self.answers = []
        self.txq.append(bytes(payload))
        t_dead = self.t + timeout / self.rate
        got = self.run(t_dead, stop=lambda: bool(self.answers) or self.conn is None)
        if self.conn is None and not self.answers:
            raise H.LinkLost("the link dropped waiting for an answer to opcode %d" % payload[0])
        if not got:
            self.txq = [p for p in self.txq if p != bytes(payload)]
            raise H.LinkTimeout("no answer to opcode %d in %.3f s" % (payload[0], timeout))
        s, data = self.answers.pop(0)
        self.run(max(self.t, self.true(s)))
        return data

    def listen(self, seconds):
        self.run(self.t + seconds / self.rate)
        out = [(s, d) for s, d in self.notes if s <= self.now()]
        self.notes = [(s, d) for s, d in self.notes if s > self.now()]
        return out
