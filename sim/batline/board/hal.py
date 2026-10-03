"""The bat's board as the line reaches it: the one file the simulated board
and the station's code agree on, as station/hal.py is for the gantry.

Three instruments and a clock, each a contract with no behaviour:

    ClockHAL     the bench's time; a procedure waits on it, never on a sleep
    SupplyHAL    the source-measure unit on the pogo block's power pins
    SwdHAL       the debug probe on SWDIO, SWCLK and reset
    RadioHAL     a Bluetooth LE central: the station's dongle, the factory
                 iPhone's app

The bat's own protocol over the radio is encoded ONCE, here, in RadioHAL's
methods (as rfgyc26's LinkHAL encodes its grammar), so the simulated
firmware and a real one parse the same bytes:

  Advertising: manufacturer data, little-endian
      16s serial (ASCII, NUL-padded), u32 firmware, u16 boots, u8 state

  CONTROL (write with response; the answer comes back as a notification)
      request   u8 opcode, payload
      answer    u8 opcode | 0x80, u8 status, payload

      INFO       0x01  ()                      -> 16s serial, u32 firmware, u16 boots, u8 state
      REC_READ   0x02  (u16 offset, u8 n)      -> n bytes of the record the bat holds
      REC_STAGE  0x03  (u16 offset, bytes)     -> ()   into a staging buffer in RAM
      REC_COMMIT 0x04  (u32 crc of the staged) -> ()   staged to flash, atomically
      STREAM     0x05  (u8 on)                 -> ()
      SLEEP      0x06  ()                      -> ()   answered, then the link drops
      IMU_TEST   0x07  ()                      -> u8 WHO_AM_I, 6 x i16 self-test
                                                  response, per mille of the
                                                  factory's: accel xyz, gyro xyz

  STREAM (notify), one packet a notification
      u32 counter of its first sample, u8 n, i16 die temperature code,
      then n x (accel xyz, gyro xyz) as i16 codes

  A commit writes the record to whichever of two flash pages does not hold
  the current one, with a sequence number, and the current one is the
  newer page whose CRC holds: a brown-out between the erase and the write
  leaves the old record readable.  That is the firmware's promise, and
  check_link holds the simulated firmware to it.

Units: seconds, volts, amps, bytes.  A backend that cannot honour a call
raises (LinkError); it does not lie.
"""
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..spec import Link
from truss.hal import audit as _audit

INFO, REC_READ, REC_STAGE, REC_COMMIT, STREAM, SLEEP, IMU_TEST = 1, 2, 3, 4, 5, 6, 7
OK, BAD_ARG, BAD_CRC, FLASH_FAIL, BUSY = 0, 1, 2, 3, 4
STATUS = {OK: "ok", BAD_ARG: "bad argument", BAD_CRC: "staged record's CRC differs",
          FLASH_FAIL: "flash read back wrong", BUSY: "busy"}
# advertised state
IDLE, WOKE = 0, 1

ADVERT = struct.Struct("<16sIHB")
INFO_ANS = struct.Struct("<16sIHB")
PACKET_HEAD = struct.Struct("<IBh")
SAMPLE = struct.Struct("<6h")
IMU_ANS = struct.Struct("<B6h")


class LinkError(Exception):
    """The instrument or the link failed to do what was asked."""


class Refused(LinkError):
    """The bat answered, and refused: `status` says why."""

    def __init__(self, op, status):
        super().__init__("opcode %d refused: %s" % (op, STATUS.get(status, status)))
        self.op, self.status = op, status


class LinkTimeout(LinkError):
    """No answer in the time allowed."""


class LinkLost(LinkError):
    """The connection dropped (the supervision timeout, or the bat reset)."""


def notification_bytes(mtu=Link.MTU):
    """Bytes one notification carries: the ATT MTU less its 3-byte header."""
    return mtu - 3


def samples_per_packet(mtu=Link.MTU):
    return (notification_bytes(mtu) - PACKET_HEAD.size) // SAMPLE.size


def record_chunk(mtu=Link.MTU):
    """Record bytes one REC_STAGE or REC_READ answer carries."""
    return notification_bytes(mtu) - 4


@dataclass
class Advert:
    t: float                   # station time it was heard
    addr: str
    serial: str
    firmware: int
    boots: int
    state: int


@dataclass
class Packet:
    t: float                   # station time it arrived
    k0: int                    # counter of its first sample
    temp: int
    codes: object              # (n, 6) int: accel xyz, gyro xyz


def parse_advert(t, addr, data):
    s, fw, boots, state = ADVERT.unpack(bytes(data)[:ADVERT.size])
    return Advert(t, addr, s.rstrip(b"\0").decode("ascii", "replace"), fw, boots, state)


def parse_packet(t, data):
    import numpy as np
    data = bytes(data)
    k0, n, temp = PACKET_HEAD.unpack_from(data)
    body = data[PACKET_HEAD.size:]
    if len(body) != n * SAMPLE.size:
        raise LinkError("a stream packet of %d samples carried %d bytes" % (n, len(body)))
    return Packet(t, k0, temp, np.frombuffer(body, "<i2").reshape(n, 6).astype(int))


class ClockHAL(ABC):
    @abstractmethod
    def now(self):
        """s, the bench's clock."""

    @abstractmethod
    def wait(self, seconds):
        """Let `seconds` pass."""


class SupplyHAL(ABC):
    """The source-measure unit on the pogo block's power and ground pins."""

    @abstractmethod
    def source(self, volts, limit):
        """Output on at `volts`, current-limited to `limit` A."""

    @abstractmethod
    def off(self):
        """Output open: the board runs on its own capacitor until it browns out."""

    @abstractmethod
    def mean_current(self, seconds):
        """A, the mean over the next `seconds` (blocks for them)."""

    @abstractmethod
    def limited(self):
        """True if the output has sat at its current limit since source()."""


class SwdHAL(ABC):
    """The debug probe on SWDIO, SWCLK and reset."""

    @abstractmethod
    def connect(self):
        """The debug port's IDCODE.  LinkError if nothing answers."""

    @abstractmethod
    def erase_all(self):
        """Erase the flash and the UICR."""

    @abstractmethod
    def program(self, addr, data):
        """Write `data` to flash at `addr` (word-aligned, erased)."""

    @abstractmethod
    def read(self, addr, n):
        """bytes: `n` from `addr`."""

    @abstractmethod
    def reset(self):
        """Pulse reset and let the core run."""


class RadioHAL(ABC):
    """A Bluetooth LE central.  The station's procedures and the factory
    iPhone's app reach the bat only through these."""

    @abstractmethod
    def scan(self, seconds):
        """[Advert] heard in the next `seconds`."""

    @abstractmethod
    def connect(self, addr, interval, timeout):
        """Connect at `interval` s; True once the bat has answered."""

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def connected(self):
        pass

    @abstractmethod
    def request(self, payload, timeout):
        """Write `payload` to CONTROL; the answer's bytes.  LinkTimeout,
        LinkLost."""

    @abstractmethod
    def listen(self, seconds):
        """[(t, bytes)]: STREAM notifications arriving in the next
        `seconds`, stamped on the station's clock as they arrive."""

    # -------------------------------------------- the grammar, encoded once
    def call(self, op, payload=b"", timeout=None):
        """The answer's payload; LinkError on a refused or malformed answer."""
        ans = bytes(self.request(bytes([op]) + bytes(payload), timeout))
        if len(ans) < 2 or ans[0] != (op | 0x80):
            raise LinkError("answer %r to opcode %d" % (ans[:4], op))
        if ans[1] != OK:
            raise Refused(op, ans[1])
        return ans[2:]

    def info(self, timeout=None):
        s, fw, boots, state = INFO_ANS.unpack(self.call(INFO, timeout=timeout))
        return s.rstrip(b"\0").decode("ascii", "replace"), fw, boots, state

    def read_record(self, size, timeout=None):
        out = b""
        n = record_chunk()
        while len(out) < size:
            m = min(n, size - len(out))
            part = self.call(REC_READ, struct.pack("<HB", len(out), m), timeout)
            if len(part) != m:
                raise LinkError("asked for %d record bytes, got %d" % (m, len(part)))
            out += part
        return out

    def write_record(self, data, crc, timeout=None):
        n = record_chunk()
        for off in range(0, len(data), n):
            self.call(REC_STAGE, struct.pack("<H", off) + data[off:off + n], timeout)
        self.call(REC_COMMIT, struct.pack("<I", crc), timeout)

    def stream(self, on, timeout=None):
        self.call(STREAM, bytes([1 if on else 0]), timeout)

    def sleep(self, timeout=None):
        self.call(SLEEP, timeout=timeout)

    def imu_test(self, timeout=None):
        who, *st = IMU_ANS.unpack(self.call(IMU_TEST, timeout=timeout))
        return who, [x / 1000.0 for x in st[:3]], [x / 1000.0 for x in st[3:]]


CONTRACTS = (ClockHAL, SupplyHAL, SwdHAL, RadioHAL)


def audit(backend, contracts=CONTRACTS):
    return _audit(backend, contracts)

