"""The bat's firmware and its board's supply, as a state machine in
simulated time.  TRUTH: simulation and checks only; no procedure imports
this (check_imu's separation scan holds it to that).

The plan's firmware: it boots, advertises, connects, streams IMU samples
in packets, writes and reads back the calibration record with a checksum,
wakes on motion, sleeps, and resets on a brown-out when the supply dips
for longer than the hold-up capacitor covers.  It speaks hal.py's bytes,
so the station's procedures cannot tell it from a real one.  A real
firmware can later run in Renode behind the same calls (the plan).

The supply: when its source is connected, the capacitor sits at the
source's volts less the drop across the path; when it is not, the
capacitor alone carries the board's draw, a steady current set by the
state plus a charge for every radio event, and below power.v_min() the
board has browned out.  Whatever the board was doing is lost: the link,
the staging buffer, the stream.  Flash is not: the image, the UICR's
serial, the two record pages and the boot counter stay.

Times are true (simulation) seconds; the central converts to its own.
"""
import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

from ..spec import Imu, Link, Mcu, Firmware as FW, Board
from ..imu.part import G0
from ..imu import record as R
from . import hal as H
from . import power as P

OFF, BOOT, ADV, CONN, SLEEP, DEAD = "off", "boot", "advertising", "connected", "asleep", "no firmware"
IMAGE_MAGIC = b"BATF"
IMAGE_HEAD = struct.Struct("<4sIII")              # magic, version, body length, crc32 of the body
PAGE_HEAD = struct.Struct("<I")                   # a record page: its sequence number, then the record


def image_version(img):
    """The firmware version an image carries, or None if it would not run."""
    if img is None or len(img) < IMAGE_HEAD.size:
        return None
    magic, ver, n, crc = IMAGE_HEAD.unpack_from(img)
    body = img[IMAGE_HEAD.size:IMAGE_HEAD.size + n]
    if magic != IMAGE_MAGIC or len(body) != n or zlib.crc32(body) != crc:
        return None
    return ver


@dataclass
class Truth:
    """One simulated board as built: its currents, its capacitor, its
    radio, and anything wrong with it."""
    i_scale: dict              # per draw term, its factor on the datasheet's typical
    hold_c: float              # F
    eps: float                 # the IMU's oscillator
    st_ratio: np.ndarray       # (6,) its self-test response, of the factory's
    who: int = Board.WHO_AM_I
    faults: dict = field(default_factory=dict)    # name -> argument


class Firmware:
    """The board: flash, supply, state, radio queue."""
    PAGES = 2                                     # record pages; check_link's must-fail writes one in place

    def __init__(self, truth, rng, source_for):
        self.tr = truth
        self.rng = rng
        self.source_for = source_for              # t_on -> a Source for this power-up
        # flash
        self.image = None
        self.uicr = b"\xff" * (4 * Mcu.UICR_WORDS)
        self.pages = [b"", b""]                   # each: PAGE_HEAD + record, or b"" erased
        self.boots = 0
        self.flips = []                           # planted: bit indices flipped in the next record writes
        # supply
        self.t = 0.0
        self.v = 0.0
        self.src = None                           # (volts, ohms) when connected
        self.q = 0.0                              # C drawn from the source since t = 0
        self.q_cap = 0.0                          # C drawn from the capacitor alone
        self.browned = []                         # times the board browned out
        self.opens = []                           # [(t_open, t_close)]: the contact breaks, scheduled
        self._open = False                        # inside one now
        self._held = None                         # the source the break disconnected
        # state
        self.state = OFF
        self._reset_volatile()

    # ------------------------------------------------------------ helpers
    def _reset_volatile(self):
        self.t_on = None
        self.src_imu = None
        self.link = False
        self.streaming = False
        self.k_next = None
        self.staged = bytearray(b"\xff" * R.SIZE)
        self.staged_n = set()
        self.txq = []                             # [(t_ready, kind, bytes)]
        self.pending = []                         # answers not ready yet
        self.dropped = 0
        self.adv_t = None
        self.adv_since = None
        self.adverts = []                         # [(t, bytes)] events since the central last read
        self.busy_until = 0.0
        self.terminated = False
        self.asleep_at = None

    @property
    def serial(self):
        return self.uicr[:16].split(b"\xff")[0].rstrip(b"\0")

    @property
    def version(self):
        return image_version(self.image)

    def _i(self, term):
        return self.tr.i_scale.get(term, 1.0)

    def base_current(self):
        """A, the steady draw in the present state, events excluded."""
        if self.state in (OFF,):
            return 0.0
        idle = Mcu.I_IDLE * self._i("idle") + self.tr.faults.get("leak", 0.0)
        if self.state == BOOT:
            return idle + Mcu.I_CPU * self._i("cpu")
        if self.state == DEAD:
            return idle
        if self.streaming:
            return idle + Imu.I_RUN * self._i("imu")
        return idle + Imu.I_WOM * self._i("wom")

    def short(self):
        return "short" in self.tr.faults

    # ------------------------------------------------------------ supply
    def supply(self, t, volts, ohms):
        self.advance(t)
        if self._open:
            self._held = (float(volts), float(ohms))
            return
        self.src = (float(volts), float(ohms))
        self._settle(t)

    def unsupply(self, t):
        self.advance(t)
        if self._open:
            self._held = None
        self.src = None

    def open_circuit(self, t0, t1):
        """The supply's contact breaks from t0 to t1 (a cell bouncing off
        its spring in a swing, a pogo pin lifting): the board runs on its
        capacitor alone between them."""
        if t1 <= t0 or t0 < self.t:
            raise ValueError("a break must lie ahead and end after it starts")
        self.opens.append((float(t0), float(t1)))
        self.opens.sort()

    def _settle(self, t):
        if self.src is None:
            return
        v_src, r = self.src
        if self.short():
            self.v = 0.0
            return
        self.v = v_src - self.base_current() * r
        if self.state == OFF and self.v >= Mcu.V_POR:
            self._boot(t)
        elif self.state != OFF and self.v < P.v_min():
            self._brown(t)

    def pulse(self, q):
        """A radio event's charge, now."""
        self.q += q
        if self.src is None and self.state != OFF:
            self.q_cap += q
            self.v -= q / self.tr.hold_c
            if self.v < P.v_min():
                self._brown(self.t)

    def _drain(self, t1):
        dt = t1 - self.t
        if dt <= 0.0:
            return
        i = self.base_current()
        self.q += i * dt
        if self.src is None and self.state != OFF:
            self.q_cap += i * dt
            v1 = self.v - i * dt / self.tr.hold_c
            if v1 < P.v_min():
                tb = self.t + (self.v - P.v_min()) * self.tr.hold_c / i
                self.t = tb
                self._brown(tb)
                self.v = P.v_min()
                return
            self.v = v1

    def _brown(self, t):
        self.browned.append(t)
        self.state = OFF
        self._reset_volatile()

    def _boot(self, t):
        self.boots += 1
        self.state = BOOT
        self._reset_volatile()
        self.t_on = t

    def reset(self, t):
        """The reset pin: a reboot with the supply up."""
        self.advance(t)
        if self.state != OFF:
            self._boot(t)

    # ------------------------------------------------------------ time
    def _next_event(self):
        ts = []
        if self.opens:
            ts.append(self.opens[0][1] if self._open else self.opens[0][0])
        if self.state == BOOT:
            ts.append(self.t_on + Imu.START_S)
        if self.state == ADV:
            ts.append(self.adv_t)
            ts.append(self.adv_since + FW.ADV_FOR_S)
        if self.state == SLEEP and self.src_imu is not None:
            w = self.src_imu.wakes_after(self.asleep_at, self.t)
            if w is not None:
                ts.append(w)
        if self.state == CONN and self.streaming:
            ts.append(self.src_imu.t(self.k_next + H.samples_per_packet() - 1))
        if self.pending:
            ts.append(min(p[0] for p in self.pending))
        return min(ts) if ts else None

    def advance(self, t1):
        while self.t < t1:
            te = self._next_event()
            tn = t1 if te is None or te > t1 else max(te, self.t)
            state0 = self.state
            self._drain(tn)
            if self.state != state0:              # browned out mid-interval
                continue
            self.t = tn
            if te is not None and te <= tn:
                self._on_event(tn)
        if self.src is not None:
            self._settle(self.t)

    def _on_event(self, t):
        if self.opens:
            t0, t1 = self.opens[0]
            if not self._open and t >= t0 - 1e-12:
                self._open, self._held, self.src = True, self.src, None
                return
            if self._open and t >= t1 - 1e-12:
                self.opens.pop(0)
                self._open, self.src, self._held = False, self._held, None
                self._settle(t)
                return
        if self.state == BOOT and t >= self.t_on + Imu.START_S - 1e-12:
            if self.version is None:
                self.state = DEAD
                return
            self.src_imu = self.source_for(self.t_on)
            self._advertise(t)
            return
        if self.state == ADV:
            if t >= self.adv_since + FW.ADV_FOR_S - 1e-12:
                self._asleep(t)
                return
            if t >= self.adv_t - 1e-12:
                if "radio_silent" not in self.tr.faults:
                    self.adverts.append((t, self.advert_bytes()))
                self.pulse(Mcu.Q_ADV * self._i("radio"))
                fast = (t - self.adv_since) < Link.ADV_FAST_FOR
                self.adv_t = t + (Link.ADV_FAST_S if fast else Link.ADV_SLOW_S) \
                    + float(self.rng.uniform(0.0, Link.ADV_JITTER))
                return
        if self.state == SLEEP:
            w = self.src_imu.wakes_after(self.asleep_at, t - 1e-9) if self.src_imu is not None else None
            if w is not None and w <= t + 1e-12:
                self._advertise(t, woke=True)
                return
        if self.state == CONN and self.streaming:
            n = H.samples_per_packet()
            if t >= self.src_imu.t(self.k_next + n - 1) - 1e-12:
                codes, temp = self.src_imu.codes(self.k_next, n)
                pk = H.PACKET_HEAD.pack(self.k_next % (2 ** 32), n, int(temp)) + codes.astype("<i2").tobytes()
                self.k_next += n
                self.pulse(P.q_cpu())
                if sum(1 for q in self.txq if q[1] == "stream") >= FW.QUEUE:
                    self.dropped += 1
                else:
                    self.txq.append((t, "stream", pk, None))
        if self.pending:
            ready = [p for p in self.pending if p[0] <= t + 1e-12]
            self.pending = [p for p in self.pending if p[0] > t + 1e-12]
            for t_ready, data, at_ready, after in ready:
                if at_ready is not None:
                    at_ready()
                if data is not None:
                    # answers go out ahead of stream packets, in their own order
                    i = sum(1 for q in self.txq if q[1] == "answer")
                    self.txq.insert(i, (t_ready, "answer", data, after))

    def _advertise(self, t, woke=False):
        self.state = ADV
        self.woke = woke
        self.adv_since = t
        self.adv_t = t + float(self.rng.uniform(0.0, Link.ADV_JITTER))

    def advert_bytes(self):
        s = self.serial[:16].ljust(16, b"\0")
        return H.ADVERT.pack(s, self.version or 0, self.boots % 65536, H.WOKE if getattr(self, "woke", False) else H.IDLE)

    # ------------------------------------------------------------ the link
    def connect(self, t):
        self.advance(t)
        if self.state != ADV:
            return False
        self.state = CONN
        self.link = True
        return True

    def disconnect(self, t):
        self.advance(t)
        if self.state == CONN:
            self.link = False
            self.streaming = False
            self.txq = []
            self._keep_flash()
            self._advertise(t)

    def _keep_flash(self):
        """The link is gone, so are its answers; a flash write under way
        finishes all the same (only power stops it)."""
        self.pending = [p for p in self.pending if p[1] is None]

    def alive(self):
        return self.state == CONN and self.link

    def has_tx(self, t):
        return any(q[0] <= t for q in self.txq)

    def pop_tx(self, t):
        """The packet at the queue's head, delivered: (kind, bytes)."""
        for i, q in enumerate(self.txq):
            if q[0] <= t:
                self.txq.pop(i)
                if q[3] is not None:
                    q[3]()
                return q[1], q[2]
        return None

    def peek_tx(self, t):
        for q in self.txq:
            if q[0] <= t:
                return q
        return None

    def receive(self, t, payload):
        """A CONTROL write arrives; its answer is queued when ready."""
        self.advance(t)
        if not self.alive():
            return
        op, body = payload[0], payload[1:]

        def answer(status, data=b"", delay=0.0, after=None):
            self.pending.append((t + delay, bytes([op | 0x80, status]) + bytes(data), None, after))
        if op == H.INFO:
            answer(H.OK, H.INFO_ANS.pack(self.serial[:16].ljust(16, b"\0"), self.version or 0,
                                         self.boots % 65536, 1 if self.streaming else 0))
        elif op == H.REC_READ:
            off, n = struct.unpack_from("<HB", body)
            rec = self.current_record()
            answer(H.OK, rec[off:off + n])
        elif op == H.REC_STAGE:
            off = struct.unpack_from("<H", body)[0]
            data = body[2:]
            if off + len(data) > R.SIZE:
                answer(H.BAD_ARG)
            else:
                self.staged[off:off + len(data)] = data
                self.staged_n.update(range(off, off + len(data)))
                answer(H.OK)
        elif op == H.REC_COMMIT:
            crc = struct.unpack_from("<I", body)[0]
            if len(self.staged_n) != R.SIZE or zlib.crc32(bytes(self.staged)) != crc:
                answer(H.BAD_CRC)
            else:
                self._commit(t, answer)
        elif op == H.STREAM:
            on = bool(body[0])
            if on and not self.streaming:
                self.k_next = self.src_imu.k_after(t)
            self.streaming = on
            if not on:
                self.txq = [q for q in self.txq if q[1] != "stream"]
            answer(H.OK)
        elif op == H.SLEEP:
            answer(H.OK, after=self._go_sleep)    # asleep once its answer is acknowledged
        elif op == H.IMU_TEST:
            st = np.clip(np.round(self.tr.st_ratio * 1000.0), -32768, 32767).astype(int)
            who = 0 if "imu_absent" in self.tr.faults else self.tr.who
            if "imu_absent" in self.tr.faults:
                st = np.zeros(6, int)
            answer(H.OK, H.IMU_ANS.pack(who, *[int(x) for x in st]), delay=Imu.ST_S)
        else:
            answer(H.BAD_ARG)

    def _go_sleep(self):
        self.terminated = True                    # it ends the link itself (LL_TERMINATE_IND)
        self.link = False
        self.streaming = False
        self.txq = []
        self._keep_flash()
        self._asleep(self.t)

    def _asleep(self, t):
        """Asleep, the IMU's wake-on-motion armed against its reading now."""
        self.state = SLEEP
        self.asleep_at = t

    # ------------------------------------------------------------ the record
    def current_record(self):
        """The newer page whose record's CRC holds, or an erased record."""
        best = None
        for pg in self.pages:
            if len(pg) != PAGE_HEAD.size + R.SIZE:
                continue
            seq = PAGE_HEAD.unpack_from(pg)[0]
            rec = pg[PAGE_HEAD.size:]
            if zlib.crc32(rec[:R.CRC_AT]) != struct.unpack_from("<I", rec, R.CRC_AT)[0]:
                continue
            if best is None or seq > best[0]:
                best = (seq, rec)
        return best[1] if best else b"\xff" * R.SIZE

    def _commit(self, t, answer):
        seqs = [PAGE_HEAD.unpack_from(pg)[0] if len(pg) == PAGE_HEAD.size + R.SIZE else -1 for pg in self.pages]
        cur = self.current_record()
        # the target: the page not holding the current record (the older, or an erased one)
        valid = [len(pg) == PAGE_HEAD.size + R.SIZE and pg[PAGE_HEAD.size:] == cur for pg in self.pages]
        target = 1 if valid[0] and (not valid[1] or seqs[0] >= seqs[1]) else 0
        if self.PAGES == 1:                       # in place: erase the current record, then write
            target = 0
        seq = max(seqs) + 1 if max(seqs) >= 0 else 0
        data = bytes(self.staged)
        if self.flips:
            data = R._flip(data, self.flips.pop(0))
        if "flash_bad" in self.tr.faults:
            data = R._flip(data, self.tr.faults["flash_bad"])
        t_erase = Mcu.ERASE_PAGE_S
        t_write = (PAGE_HEAD.size + R.SIZE) / 4 * Mcu.FLASH_WORD_S
        boots = self.boots

        def write():
            if self.boots != boots:                   # power was lost under it
                return
            self.pages[target] = PAGE_HEAD.pack(seq) + data
        # the erase happens at once; the write lands when it is done, and the
        # answer reports the firmware's own read-back
        self.pages[target] = b""
        self.pending.append((t + t_erase + t_write, None, write, None))
        ok = data == bytes(self.staged)
        answer(H.OK if ok else H.FLASH_FAIL, delay=t_erase + t_write + 1e-6)
        self.staged = bytearray(b"\xff" * R.SIZE)
        self.staged_n = set()

