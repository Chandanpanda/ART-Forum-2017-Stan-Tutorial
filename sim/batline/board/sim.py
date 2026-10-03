"""Simulated boards on a simulated bench: the HAL's backends, the IMU's
output for one power-up, and boards drawn healthy or with a planted
fault.  TRUTH: simulation and checks only.

A bench is one board, one central and the pogo block's two instruments,
sharing one clock: the central's (a station's clock is its dongle's).
Every backend raises where the hardware would fail and never reports
what the hardware could not know.
"""
from dataclasses import dataclass, field
from math import radians, sin, cos

import numpy as np

from ..spec import Imu, Board, Mcu, Smu, Pogo, Est, Site, Probe, Firmware as FW
from ..imu import model as MD, kin as K
from ..imu.part import lsb, G0, D2R, window_noise
from . import hal as H
from . import power as P
from .firmware import Firmware, Truth
from .radio import Central

# each planted fault, its argument, and the step of station_test that must refuse it
FAULTS = {"short": (True, "power"), "swd_dead": (True, "swd"), "image_flip": (1000, "flash"),
          "uicr_fail": (True, "serial"), "radio_silent": (True, "advertises"), "imu_absent": (True, "imu answers"),
          "imu_selftest": ((4, 0.3), "imu self-test"),
          "accel_scale": ((1, 0.8), "imu self-test"),       # sensitivity 1.8x: its own self-test sees it in any pose
          "stuck_gyro": (1, "not stuck"), "stuck_accel": (2, "not stuck"),
          "accel_offset": ((0, 0.3), "gravity along"), "tilted": (10.0, "gravity along"),
          "accel_range": (True, "gravity size"),              # the firmware set +-16 g: every code doubled
          "gyro_offset": ((0, 10.0), "gyro still"), "noisy_accel": ((1, 3.0), "noise"),
          "clock": (0.25, "clock"),                          # the IMU set to 800 Hz, not 1 kHz
          "leak_stream": (30.0, "streaming current"),
          "no_holdup": (10e-6, "hold-up"),
          "leak_sleep": (0.5e-3, "sleep current")}           # small enough to ride through the hold-up


# ================================================================ THE IMU'S OUTPUT
class Source:
    """The board's IMU this power-up: its sample times and its codes."""

    def __init__(self, t_first, eps):
        self.t_first = float(t_first)
        self.eps = float(eps)

    def t(self, k):
        return self.t_first + np.asarray(k, float) * (1.0 + self.eps) / Imu.ODR

    def k_after(self, t):
        return max(0, int(np.floor((t - self.t_first) * Imu.ODR / (1.0 + self.eps))) + 1)

    def wakes_after(self, t_ref, t):
        """The time of the first sample after t whose acceleration differs
        from the sample at t_ref by more than Firmware.WOM_G on any axis
        (the IMU's wake-on-motion, compared with its reading as it was
        armed), or None."""
        return None

    def codes(self, k0, n):
        raise NotImplementedError


class StillSource(Source):
    """A bat lying still: gravity and the Earth's turn in the chip's axes,
    through the part's errors, filter and rounding (imu/model.sense), in
    blocks drawn as they are first asked for; with any IMU fault applied.
    It never wakes the board: its noise (milli-g) is far under
    Firmware.WOM_G, which check_link's wake check holds."""
    BLOCK = 2048
    RATE = 4.0 * Imu.ODR       # Hz, the internal rate: the signal is constant, so any past the filter's band

    def __init__(self, err, pw, f_chip, w_chip, rng, t_first, faults=None):
        super().__init__(t_first, err.eps)
        self.err, self.pw, self.rng = err, pw, rng
        self.f, self.w = np.asarray(f_chip, float), np.asarray(w_chip, float)
        self.faults = dict(faults or {})
        self.blocks = {}

    def _block(self, b):
        if b not in self.blocks:
            k0 = b * self.BLOCK
            t0 = float(self.t(k0)) - 0.05
            t1 = float(self.t(k0 + self.BLOCK)) + 2.0 / Imu.ODR
            fs = self.RATE
            T = np.arange(t0, t1, 1.0 / fs)
            n = len(T)
            st, _, _ = MD.sense(self.err, self.pw, T, np.tile(self.w, (n, 1)), np.tile(self.f, (n, 1)),
                                float(self.t(k0)), self.BLOCK, self.rng)
            c = np.concatenate([st.acc, st.gyro], axis=1).astype(int)
            fl = self.faults
            if "accel_range" in fl:
                c[:, :3] = np.clip(2 * c[:, :3], -2 ** 15, 2 ** 15 - 1)
            if "imu_absent" in fl:
                c[:] = 0
            if "stuck_gyro" in fl:
                c[:, 3 + fl["stuck_gyro"]] = self._stuck_code(3 + fl["stuck_gyro"], c)
            if "stuck_accel" in fl:
                c[:, fl["stuck_accel"]] = self._stuck_code(fl["stuck_accel"], c)
            if "noisy_accel" in fl:
                ax, factor = fl["noisy_accel"]
                sd = Imu.ACC_ND * G0 * np.sqrt(window_noise(1)[0]) / lsb()[1]
                c[:, ax] = np.round(c[:, ax] + self.rng.normal(0.0, sd * np.sqrt(factor ** 2 - 1.0), len(c)))
            self.blocks[b] = (c, st.temp.astype(int))
        return self.blocks[b]

    def _stuck_code(self, j, c):
        if not hasattr(self, "_stuck"):
            self._stuck = {}
        return self._stuck.setdefault(j, int(c[0, j]))

    def codes(self, k0, n):
        out, temp = [], 0
        k = k0
        while k < k0 + n:
            b, i = divmod(k, self.BLOCK)
            c, tc = self._block(b)
            m = min(n - (k - k0), self.BLOCK - i)
            out.append(c[i:i + m])
            temp = tc[i + m - 1]
            k += m
        return np.concatenate(out, 0), temp


class StreamSource(Source):
    """A stream the simulation made already (imu/sim's station run, a
    swing): its codes by counter."""

    def __init__(self, stream, t_first, eps):
        super().__init__(t_first, eps)
        self.c = np.concatenate([stream.acc, stream.gyro], axis=1).astype(int)
        self.temp = np.asarray(stream.temp, int)
        self._moved = {}

    def wakes_after(self, t_ref, t):
        if t_ref is None:
            return None
        k_ref = min(self.k_after(t_ref), len(self.c) - 1)
        if k_ref not in self._moved:
            thr = FW.WOM_G * G0 / lsb()[1]
            off = np.abs(self.c[k_ref + 1:, :3] - self.c[k_ref, :3]).max(1)
            self._moved[k_ref] = k_ref + 1 + np.flatnonzero(off > thr)
        moved = self._moved[k_ref]
        i = np.searchsorted(moved, self.k_after(t))
        return float(self.t(moved[i])) if i < len(moved) else None

    def codes(self, k0, n):
        if k0 + n > len(self.c):
            raise ValueError("the stream ends at sample %d" % len(self.c))
        return self.c[k0:k0 + n], self.temp[k0 + n - 1]


# ================================================================ A BOARD
@dataclass
class Lying:
    """How the bat lies in a station's cradle: its axis level to within
    the cradle's pitch, free to roll.  pitch is the truth; the station
    knows `measured` +- `sd` (deg)."""
    pitch: float
    roll: float
    measured: float = 0.0
    sd: float = Site.LEVEL


def chip_in_bat(rng, tilt_deg=0.0):
    """R (3x3) taking chip axes to the bat's, as built: the board's
    nominal axes (Board.IMU_AXES), the chip's placement on it, and any
    extra pitch of a board not seated home."""
    A = np.array(Board.IMU_AXES, float).T
    e = np.array([rng.normal(0.0, Board.CHIP_TILT), rng.normal(0.0, Board.CHIP_TILT) + tilt_deg,
                  rng.normal(0.0, Board.CHIP_ROT)])
    return A @ K.exp_so3(np.radians(e))


def gravity_in_chip(lying, R_bc):
    """(f, w) at rest in chip axes, SI: the specific force (up, g) and the Earth's turn."""
    R_wb = K.rot([0.0, 1.0, 0.0], [radians(lying.pitch)])[0] @ K.rot([1.0, 0.0, 0.0], [radians(lying.roll)])[0]
    R_wc = R_wb @ R_bc
    g = K.gravity(Site.LAT, Site.ALT)
    up = np.array([0.0, 0.0, 1.0])
    w = K.earth_rate(Site.LAT, up, K.north_rig(Site.HEADING))
    return R_wc.T @ (g * up), R_wc.T @ w


@dataclass
class SimBoard:
    truth: Truth
    err: object                # model.ImuErrors
    R_bc: np.ndarray
    lying: Lying
    serial: str = ""
    planted: dict = field(default_factory=dict)


def draw_board(rng, fault=None, arg=None, lying=None):
    """A board drawn from its parts' spreads, with one planted fault (a
    name from FAULTS and its argument) or none."""
    err = MD.draw_errors(rng)
    sc = {k: max(0.3, 1.0 + rng.normal(0.0, Est.CURRENT_SPREAD)) for k in ("idle", "cpu", "imu", "wom", "radio")}
    tr = Truth(i_scale=sc, hold_c=Board.HOLD_C * (1.0 + rng.uniform(-Board.HOLD_TOL, Board.HOLD_TOL)),
               eps=err.eps, st_ratio=1.0 + rng.normal(0.0, 0.05, 6))
    tilt = 0.0
    if fault is not None:
        if fault not in FAULTS:
            raise ValueError(fault)
        tr.faults[fault] = arg
        if fault == "no_holdup":
            tr.hold_c = arg
        elif fault == "clock":
            err.eps = tr.eps = arg
        elif fault == "imu_selftest":
            tr.st_ratio[arg[0]] = arg[1]
        elif fault == "accel_scale":
            err.Ea[arg[0], arg[0]] += arg[1]
            tr.st_ratio[arg[0]] *= 1.0 + arg[1]
        elif fault == "accel_offset":
            err.ba[arg[0]] += arg[1] * G0
        elif fault == "gyro_offset":
            err.bg[arg[0]] += arg[1] * D2R
        elif fault == "tilted":
            tilt = arg
        elif fault in ("leak_sleep",):
            tr.faults["leak"] = arg
        elif fault == "leak_stream":
            tr.i_scale["imu"] *= arg
    lying = lying or Lying(pitch=float(rng.normal(0.0, Site.LEVEL)), roll=float(rng.uniform(0.0, 360.0)))
    return SimBoard(tr, err, chip_in_bat(rng, tilt), lying)


# ================================================================ THE BENCH
class Bench:
    """One board under the pogo block with a central in range, on one
    clock.  Its four backends, one per contract, share it."""

    def __init__(self, board, rng, central="dongle", per=None, firmware=Firmware, source_for=None):
        self.board = board
        self.rng = rng
        imu_faults = {k: v for k, v in board.truth.faults.items()
                      if k in ("imu_absent", "stuck_gyro", "stuck_accel", "noisy_accel", "accel_range")}
        f, w = gravity_in_chip(board.lying, board.R_bc)

        def still(t_on):
            pw = MD.power_up(rng, t_on, Site.AMBIENT)
            t_first = t_on + Imu.START_S + float(rng.uniform(0.0, 1.0)) / Imu.ODR
            return StillSource(board.err, pw, f, w, rng, t_first, imu_faults)
        self.fw = firmware(board.truth, rng, source_for or still)
        self.central = Central(central, self.fw, rng, per=per)
        self.clock = SimClock(self)
        self.supply = SimSupply(self)
        self.swd = SimSwd(self)
        self.radio = SimRadio(self)

    def t(self):
        return self.central.t

    def swap(self, kind, per=None):
        """Another central takes over the bench (the factory iPhone after the
        station's dongle), from the same moment, on its own clock."""
        old = self.central
        if old.conn is not None:
            old.disconnect()
        self.central = Central(kind, self.fw, self.rng, per=per, t0=old.t)
        return self.central


class SimClock(H.ClockHAL):
    def __init__(self, b):
        self.b = b

    def now(self):
        return self.b.central.now()

    def wait(self, seconds):
        c = self.b.central
        c.run(c.t + seconds / c.rate)


class SimSupply(H.SupplyHAL):
    def __init__(self, b):
        self.b = b
        self._limit = None
        self._limited = False

    def source(self, volts, limit):
        if not (Smu.V_RANGE[0] <= volts <= Smu.V_RANGE[1]):
            raise H.LinkError("the supply cannot source %.2f V" % volts)
        self._limit = float(limit)
        self._limited = self.b.fw.short()
        self.b.fw.supply(self.b.t(), volts, 2.0 * Pogo.R_CONTACT)

    def off(self):
        self.b.fw.unsupply(self.b.t())

    def mean_current(self, seconds):
        fw, rng = self.b.fw, self.b.rng
        q0, t0 = fw.q, self.b.t()
        self.b.clock.wait(seconds)
        if self._limited:
            i = self._limit
        else:
            i = (fw.q - q0) / max(self.b.t() - t0, 1e-12)
            if self._limit is not None and i > self._limit:
                self._limited = True
                i = self._limit
        return float(i * (1.0 + rng.uniform(-Smu.ACC, Smu.ACC)) + rng.normal(0.0, Smu.FLOOR))

    def limited(self):
        return self._limited


class SimSwd(H.SwdHAL):
    def __init__(self, b):
        self.b = b

    def _ok(self):
        fw = self.b.fw
        if "swd_dead" in self.b.board.truth.faults or fw.src is None or fw.v < Mcu.V_POR:
            raise H.LinkError("no answer on SWD")

    def _time(self, n_bytes, write):
        words = (n_bytes + 3) // 4
        per = Mcu.SWD_WORD_BITS / min(Probe.SWD_HZ, Mcu.SWD_HZ)
        return words * (max(per, Mcu.FLASH_WORD_S) if write else per)

    def connect(self):
        self._ok()
        self.b.clock.wait(64 * Mcu.SWD_WORD_BITS / Probe.SWD_HZ)
        return Board.SWD_IDCODE

    def erase_all(self):
        self._ok()
        self.b.clock.wait(Mcu.ERASE_ALL_S)
        fw = self.b.fw
        fw.image = None
        fw.uicr = b"\xff" * (4 * Mcu.UICR_WORDS)
        fw.pages = [b"", b""]

    def program(self, addr, data):
        self._ok()
        data = bytes(data)
        self.b.clock.wait(self._time(len(data), True))
        fw, faults = self.b.fw, self.b.board.truth.faults
        if addr == 0:
            if "image_flip" in faults:
                bb = bytearray(data)
                bb[faults["image_flip"] % len(bb)] ^= 0x10
                data = bytes(bb)
            fw.image = data
        elif Mcu.UICR_SERIAL <= addr < Mcu.UICR_SERIAL + 4 * Mcu.UICR_WORDS:
            if "uicr_fail" in faults:
                return
            off = addr - Mcu.UICR_SERIAL
            u = bytearray(fw.uicr)
            u[off:off + len(data)] = bytes(x & y for x, y in zip(u[off:off + len(data)], data))
            fw.uicr = bytes(u)
        else:
            raise H.LinkError("no flash at 0x%08x" % addr)

    def read(self, addr, n):
        self._ok()
        self.b.clock.wait(self._time(n, False))
        fw = self.b.fw
        if addr == 0:
            return (fw.image or b"")[:n].ljust(n, b"\xff")
        if Mcu.UICR_SERIAL <= addr < Mcu.UICR_SERIAL + 4 * Mcu.UICR_WORDS:
            off = addr - Mcu.UICR_SERIAL
            return fw.uicr[off:off + n]
        raise H.LinkError("no flash at 0x%08x" % addr)

    def reset(self):
        self._ok()
        self.b.fw.reset(self.b.t())


class SimRadio(H.RadioHAL):
    def __init__(self, b):
        self.b = b

    def scan(self, seconds):
        return self.b.central.scan(seconds)

    def connect(self, addr, interval, timeout):
        return self.b.central.connect(addr, interval, timeout)

    def disconnect(self):
        self.b.central.disconnect()

    def connected(self):
        return self.b.central.connected()

    def request(self, payload, timeout):
        return self.b.central.request(payload, timeout)

    def listen(self, seconds):
        return self.b.central.listen(seconds)
