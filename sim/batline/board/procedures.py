"""What the line does to a bat's board, through the HAL only.  Production:
the same functions drive the simulated bench and, later, the real one.

    image          the firmware image the line flashes (a stand-in body)
    station_test   station 2: power, flash, serial, boot, advertise, the IMU's
                   answers, its still stream, the clock, the currents, the
                   hold-up -- one verdict with its reason
    BleLink        imu.record.BatLink over the radio: how station 4 writes
                   the calibration record and the factory iPhone reads it
    pair           the factory iPhone's pairing: find the bat by its serial,
                   connect, read its record back
    plan           every step's planned time, from the facts, for the line

Every limit is computed: the gravity and noise windows from the IMU's
datasheet spreads, the currents from the battery-life rules (power.py), the
hold-up from Rules.BOUNCE_S, and each statistical test's share of
Quality.TEST_ALPHA by Bonferroni.  Every wait is on a sensor or the link,
with a timeout of Module.OVERRUN times what it should take.
"""
import hashlib
import struct
import zlib
from dataclasses import dataclass, field
from math import sqrt, radians, sin, cos, ceil, log, atan, degrees

import numpy as np

from ..spec import (Imu, Board, Mcu, Link, Firmware, Module, Quality, Rules, Smu, Print, Probe, Site)
from ..imu import record as R, fit as F, part as PT, kin as K
from ..rig.pose import normal_quantile
from ..rig.calib import chi2_quantile
from . import hal as H
from . import power as P
from . import sync as SY


# ================================================================ THE IMAGE
def image(version=Firmware.VERSION, size=Firmware.IMAGE_BYTES):
    """The image the line flashes: a header the firmware checks and a body
    of `size` bytes (a stand-in until the firmware exists: what matters to
    the line is its size and that a bad copy is caught)."""
    body = bytearray()
    h = hashlib.sha256(b"bat firmware %d" % version).digest()
    while len(body) < size:
        h = hashlib.sha256(h).digest()
        body += h
    body = bytes(body[:size])
    return struct.pack("<4sIII", b"BATF", version, len(body), zlib.crc32(body)) + body


# ================================================================ THE LIMITS
TESTS = ("who", "selftest", "gravity along", "gravity size", "gyro still", "noise", "clock")


def z_test(n=len(TESTS) * 6):
    """Spreads a healthy board's reading may lie from nominal: Quality.TEST_ALPHA
    shared by Bonferroni over every channel of every statistical test."""
    return normal_quantile(1.0 - Quality.TEST_ALPHA / (2.0 * n))


def alpha_test(n=len(TESTS) * 6):
    return Quality.TEST_ALPHA / n


@dataclass(frozen=True)
class Fixture:
    """How the station believes the bat lies in its cradle: the bat's axis
    pitched `pitch` deg, known to `sd` (1 sigma).  Roll about the axis is
    free in a V cradle, so the test uses none."""
    pitch: float = 0.0
    sd: float = Site.LEVEL


def seat_tilt_deg():
    """1 sigma of the board's pitch in its slot: its thickness's tolerance
    and the printed slot's scatter, across the board's length."""
    return degrees(atan((Board.PCB_TOL + Print.TOL) / Board.L))


def gravity_sd(fixture):
    """(along, size) 1 sigma, in g, of a healthy still reading: the
    along-bat component and the magnitude, from the datasheet's spreads,
    the cradle's knowledge and the board's seat."""
    dT = Imu.SELF_HEAT + Site.AMBIENT_SD
    acc = Imu.ACC_BIAS ** 2 + Imu.ACC_POWERUP ** 2 + (Imu.ACC_TC * dT) ** 2
    ang = radians(fixture.sd) ** 2 + radians(Board.CHIP_TILT) ** 2 + radians(seat_tilt_deg()) ** 2
    along = sqrt(acc + 2.0 * Imu.ACC_CROSS ** 2 + ang)
    size = sqrt(acc + Imu.ACC_SCALE ** 2 + 2.0 * Imu.ACC_CROSS ** 2)
    return along, size


def gyro_sd():
    """dps, 1 sigma of a still gyro axis' mean: its offset, the power-up's,
    the temperature's and the g-sensitivity's under 1 g."""
    dT = Imu.SELF_HEAT + Site.AMBIENT_SD
    return sqrt(Imu.GYRO_BIAS ** 2 + Imu.GYRO_POWERUP ** 2 + (Imu.GYRO_TC * dT) ** 2 + 3.0 * Imu.GYRO_GSENS ** 2)


def noise_most():
    """(6,) codes^2, accel then gyro: the most a healthy channel's samples
    vary by, wherever its true value sits between two codes (imu.part)."""
    nd2 = F.datasheet_nd() ** 2
    s = F.code_sd(nd2)                                   # gyro xyz, accel xyz
    most = np.array([PT.code_var_range(float(x), 1)[2] for x in s])
    return np.concatenate([most[3:], most[:3]])


def noise_dof(n):
    """Effective degrees of freedom of a sample variance of n filtered
    samples: n over 1 + 2 sum rho_k^2, rho the output's correlation."""
    rho = PT._rho()
    return n / (1.0 + 2.0 * float(np.sum(rho[1:] ** 2)))


def clock_sd(ci, central, n=None):
    """1 sigma of a recovered sample rate, as a share: the part's
    oscillator, the central's clock (uniform within its tolerance) and the
    envelope's own slope error (sync.slope_sd), which check_link measures."""
    return sqrt(Imu.ODR_TOL ** 2 + (Link.CLOCK_PPM * 1e-6) ** 2 / 3.0 + SY.slope_sd(ci, central, n) ** 2)


def rate_packets(ci, central):
    """Stream packets the clock test rests on: enough that the envelope's
    slope error (sync.slope_sd) is no larger than the part's own rate
    tolerance, so the test judges the part and not the link."""
    n = SY.packets_needed(ci, central)
    while SY.slope_sd(ci, central, n) > Imu.ODR_TOL:
        n += 1
    return n


def stream_s(ci, central="dongle"):
    """s station 2 streams for: the longest of what the clock's offset,
    its rate, the noise test and the current's window each need."""
    rate = P.packet_rate()
    return max(SY.listen_s(ci, central), rate_packets(ci, central) / rate, noise_samples() / Imu.ODR,
               current_window())


def scan_events():
    """Advertising events a scan must cover so that hearing none from a
    healthy board is as rare as a test's share of TEST_ALPHA."""
    return int(ceil(log(alpha_test()) / log(Link.PER)))


def supply_limit():
    """A, the station's current limit: above a healthy board's heaviest
    draw by Module.Z_SIGMAS of the unit spread, so only a fault sits at it."""
    from ..spec import Est
    return P.peak_current() * (1.0 + Module.Z_SIGMAS * Est.CURRENT_SPREAD)


def current_window():
    """s a current is averaged over: enough radio events that one more or
    less moves the mean by less than the instrument's own accuracy."""
    return ceil(1.0 / Smu.ACC) * Link.ADV_FAST_S


def noise_samples():
    """Still samples the noise test rests on: enough that a channel at
    twice its datasheet noise fails it as surely as a healthy one passes
    (the two chi-square tails meet)."""
    z = normal_quantile(1.0 - alpha_test())
    # (s2/sigma2) ~ 1 +- sqrt(2/nu): the healthy limit 1 + z sqrt(2/nu) under 4 - z 4 sqrt(2/nu)
    nu = 2.0 * (5.0 * z / 3.0) ** 2
    rho = PT._rho()
    return int(ceil(nu * (1.0 + 2.0 * float(np.sum(rho[1:] ** 2)))))


# ================================================================ STEPS
@dataclass
class Step:
    name: str
    ok: bool
    why: str = ""
    t0: float = 0.0
    t1: float = 0.0
    data: dict = field(default_factory=dict)


@dataclass
class Verdict:
    ok: bool
    why: str
    steps: list
    serial: str

    @property
    def seconds(self):
        return self.steps[-1].t1 - self.steps[0].t0 if self.steps else 0.0

    def step(self, name):
        return next((s for s in self.steps if s.name == name), None)


class _Fail(Exception):
    pass


class _Run:
    def __init__(self, st):
        self.st = st
        self.steps = []

    def __call__(self, name, ok, why="", t0=None, **data):
        s = Step(name, bool(ok), "" if ok else why, self.st.clock.now() if t0 is None else t0,
                 self.st.clock.now(), data)
        self.steps.append(s)
        if not ok:
            raise _Fail(why)
        return s


@dataclass
class Station:
    """The instruments a procedure drives."""
    clock: H.ClockHAL
    supply: H.SupplyHAL
    swd: H.SwdHAL
    radio: H.RadioHAL


def request_s(ci, extra=0.0, central="dongle"):
    """s a request should take: out on the next event, back on the one after,
    across the host stack -- what its timeout is OVERRUN times."""
    from ..spec import Est
    return 2.0 * ci + extra + Est.HOST_FLOOR_S[central] + Est.HOST_SPREAD_S[central]


def timeout(planned):
    return Module.OVERRUN * planned


# ================================================================ STATION 2
def station_test(st, serial, fixture=Fixture(), img=None):
    """Station 2's work on a fresh board with the pogo block down.  Verdict
    with every step's reading and time."""
    img = image() if img is None else img
    run = _Run(st)
    ci = Link.CI_MIN
    try:
        _power(run, st)
        _flash(run, st, img)
        _serial(run, st, serial)
        _boot(run, st, serial)
        _connect(run, st, serial, ci, img)
        _imu(run, st)
        _stream(run, st, ci, fixture)
        _holdup(run, st, serial)
        _sleep(run, st, Link.PHONE_CI)
        st.supply.off()
        return Verdict(True, "", run.steps, serial)
    except _Fail as e:
        try:
            st.radio.disconnect()
            st.supply.off()
        except H.LinkError:
            pass
        return Verdict(False, str(e), run.steps, serial)


def _power(run, st):
    t0 = st.clock.now()
    lim = supply_limit()
    st.supply.source(P.v_fresh(), lim)
    st.clock.wait(Board.HOLD_C * (1.0 + Board.HOLD_TOL) * P.v_fresh() / lim)   # the capacitor charged at the limit
    run("power", not st.supply.limited(), "the supply sits at its %.1f mA limit: a short" % (lim * 1e3), t0)


def _flash(run, st, img):
    t0 = st.clock.now()
    try:
        idc = st.swd.connect()
    except H.LinkError:
        run("swd", False, "nothing answers on SWD", t0)
    run("swd", idc == Board.SWD_IDCODE, "SWD IDCODE 0x%08x, not the radio's" % idc, t0, idcode=idc)
    t0 = st.clock.now()
    st.swd.erase_all()
    tries = []
    for _ in range(2):
        st.swd.program(0, img)
        back = st.swd.read(0, len(img))
        tries.append(back == img)
        if tries[-1]:
            break
    run("flash", tries[-1], "the image read back wrong twice", t0, writes=len(tries))


def _serial(run, st, serial):
    t0 = st.clock.now()
    b = serial.encode("ascii")
    if len(b) > 16:
        raise ValueError("a serial is at most 16 ASCII bytes")
    b = b.ljust(16, b"\0")
    st.swd.program(Mcu.UICR_SERIAL, b)
    back = st.swd.read(Mcu.UICR_SERIAL, 16)
    run("serial", back == b, "the UICR read back %r" % back, t0)


def _boot(run, st, serial):
    t0 = st.clock.now()
    st.swd.reset()
    st.clock.wait(Imu.START_S)
    n = scan_events()
    seen = st.radio.scan(n * (Link.ADV_FAST_S + Link.ADV_JITTER))
    mine = [a for a in seen if a.serial == serial]
    run("advertises", bool(mine), "no advertisement with the serial in %d advertising intervals" % n, t0,
        heard=len(mine))
    t0 = st.clock.now()
    i = st.supply.mean_current(current_window())
    lim = P.limits().advertising
    run("advertising current", P.accept(i, lim), "%.3f mA advertising against %.3f mA" % (i * 1e3, lim * 1e3),
        t0, amps=i)


def _connect(run, st, serial, ci, img):
    t0 = st.clock.now()
    ok = st.radio.connect(serial, ci, timeout(Link.ADV_SLOW_S + 2.0 * ci))
    run("connects", ok, "it would not connect", t0)
    t0 = st.clock.now()
    try:
        s, fw, boots, _ = st.radio.info(timeout(request_s(ci)))
    except H.LinkError as e:
        run("info", False, "no answer to INFO: %s" % e, t0)
    ver = struct.unpack_from("<I", img, 4)[0]
    run("info", s == serial and fw == ver, "it reports %r firmware 0x%08x" % (s, fw), t0, boots=boots)


def _imu(run, st):
    t0 = st.clock.now()
    try:
        who, st_a, st_g = st.radio.imu_test(timeout(request_s(Link.CI_MIN, Imu.ST_S)))
    except H.LinkError as e:
        run("imu answers", False, "no answer to IMU_TEST: %s" % e, t0)
    run("imu answers", who == Board.WHO_AM_I, "WHO_AM_I reads 0x%02x" % who, t0)
    lo, hi = Imu.ST_RATIO
    bad = [n for n, r in zip(("accel x", "accel y", "accel z", "gyro x", "gyro y", "gyro z"), st_a + st_g)
           if not lo <= r <= hi]
    run("imu self-test", not bad, "its own self-test out of %.2f-%.2f on %s" % (lo, hi, ", ".join(bad)), t0,
        ratios=st_a + st_g)


def _collect(st, seconds):
    """[hal.Packet] streamed in the next `seconds`, with the mean current over them."""
    i = st.supply.mean_current(seconds)
    raw = st.radio.listen(0.0)
    return [H.parse_packet(t, d) for t, d in raw], i


def _stream(run, st, ci, fixture):
    t0 = st.clock.now()
    st.radio.stream(True, timeout(request_s(ci)))
    need = stream_s(ci)
    pk, i = _collect(st, need)
    run("streams", len(pk) >= 2, "%d stream packets in %.2f s" % (len(pk), need), t0)
    k = np.concatenate([p.k0 + np.arange(len(p.codes)) for p in pk])
    codes = np.concatenate([p.codes for p in pk])
    stream = PT.Stream(k, codes[:, :3], codes[:, 3:], np.zeros(len(k), int), np.zeros(len(k), bool))
    stuck = F.stuck(stream)
    run("not stuck", not stuck, "a channel holds one code: %s" % ", ".join(stuck), t0)
    gl, al = PT.lsb()
    a = codes[:, :3].mean(0) * al / K.gravity(Site.LAT, Site.ALT)          # g, chip axes
    w = codes[:, 3:].mean(0) * gl / PT.D2R                                 # dps
    z = z_test()
    sd_along, sd_size = gravity_sd(fixture)
    along_axis = np.array(Board.IMU_AXES, float)[:, 0]                     # the bat's axis in chip coordinates
    along = float(a @ along_axis) + sin(radians(fixture.pitch))
    run("gravity along", abs(along) <= z * sd_along,
        "gravity along the bat %.4f g off the cradle's: the board is not seated (limit %.4f)" % (along, z * sd_along),
        t0, along=along)
    size = float(np.linalg.norm(a)) - 1.0
    run("gravity size", abs(size) <= z * sd_size, "gravity reads %.4f g off 1 (limit %.4f)" % (size, z * sd_size),
        t0, size=size)
    run("gyro still", bool(np.all(np.abs(w) <= z * gyro_sd())),
        "a still gyro reads %s dps (limit %.2f)" % (np.round(w, 2), z * gyro_sd()), t0, gyro=w)
    var = codes.var(0, ddof=1)
    nu = noise_dof(len(codes))
    top = noise_most() * chi2_quantile(nu, alpha_test()) / nu
    names = ("accel x", "accel y", "accel z", "gyro x", "gyro y", "gyro z")
    loud = [n for n, v, t in zip(names, var, top) if v > t]
    run("noise", not loud, "noisier than its datasheet: %s" % ", ".join(loud), t0, ratio=var / noise_most())
    cm = SY.recover(pk, "dongle")
    dr = cm.rho - 1.0
    lim = z * clock_sd(ci, "dongle", len(pk))
    run("clock", abs(dr) <= lim, "its sample clock runs %.2f%% off (limit %.2f%%)" % (100 * dr, 100 * lim), t0, rho=cm.rho)
    lim = P.limits().streaming
    run("streaming current", P.accept(i, lim), "%.3f mA streaming against %.3f mA" % (i * 1e3, lim * 1e3), t0, amps=i)


def _holdup(run, st, serial):
    """The supply opened for Rules.BOUNCE_S with the board streaming to a
    central at the phone's interval (the game's case; power.holdup_c_needed
    is derived for it) at the voltage the hold-up is promised down to: it
    must still be there, on the same boot, after."""
    t0 = st.clock.now()
    ci = Link.PHONE_CI
    st.radio.disconnect()
    if not st.radio.connect(serial, ci, timeout(Link.ADV_SLOW_S + 2.0 * ci)):
        run("hold-up", False, "it would not reconnect at the phone's interval", t0)
    st.radio.stream(True, timeout(request_s(ci)))
    _, _, boots0, _ = st.radio.info(timeout(request_s(ci)))
    st.supply.source(P.v_hold(), supply_limit())
    st.clock.wait(2.0 * ci)
    st.supply.off()
    st.clock.wait(Rules.BOUNCE_S)
    st.supply.source(P.v_hold(), supply_limit())
    try:
        _, _, boots1, _ = st.radio.info(timeout(request_s(ci)))
        ok = boots1 == boots0
    except H.LinkError:
        ok = False
    run("hold-up", ok, "it reset when its supply opened for %.1f ms at %.2f V" % (Rules.BOUNCE_S * 1e3, P.v_hold()), t0)
    st.supply.source(P.v_fresh(), supply_limit())


def _sleep(run, st, ci):
    t0 = st.clock.now()
    st.radio.stream(False, timeout(request_s(ci)))
    st.radio.sleep(timeout(request_s(ci)))
    st.clock.wait(2.0 * ci)
    i = st.supply.mean_current(current_window())
    lim = P.limits().sleep
    run("sleep current", P.accept(i, lim), "%.1f uA asleep against %.1f uA" % (i * 1e6, lim * 1e6), t0, amps=i)


# ================================================================ THE RECORD OVER THE AIR
class BleLink(R.BatLink):
    """The bat's calibration flash over Bluetooth.  A write stages the
    record in chunks, then commits it with its CRC, which the firmware
    checks before it touches flash; a lost link or a missed answer is
    retried once on a fresh connection.  What the bat then holds is judged
    by reading it back (imu.record.commit compares every bit)."""

    def __init__(self, radio, serial, interval=Link.CI_MIN, central="dongle"):
        self.radio = radio
        self.serial = serial
        self.ci = interval
        self.central = central
        self.writes = 0
        self.reads = 0
        self.retries = 0

    def _connect(self):
        if not self.radio.connected():
            if not self.radio.connect(self.serial, self.ci, timeout(Link.ADV_SLOW_S + 2.0 * self.ci)):
                raise H.LinkLost("%s would not connect" % self.serial)

    def _twice(self, what):
        for attempt in range(2):
            try:
                self._connect()
                return what()
            except (H.LinkLost, H.LinkTimeout):
                if attempt:
                    raise
                self.retries += 1
                self.radio.disconnect()

    def write(self, data):
        self.writes += 1
        data = bytes(data)
        t = timeout(request_s(self.ci, Mcu.ERASE_PAGE_S + R.SIZE / 4 * Mcu.FLASH_WORD_S, self.central))

        def go():
            try:
                self.radio.write_record(data, zlib.crc32(data), t)
            except H.Refused as e:
                if e.status not in (H.FLASH_FAIL, H.BAD_CRC):
                    raise
                # the read back will show what the bat holds
        self._twice(go)

    def read(self):
        self.reads += 1
        t = timeout(request_s(self.ci, 0.0, self.central))
        return self._twice(lambda: self.radio.read_record(R.SIZE, t))


# ================================================================ THE FACTORY IPHONE
@dataclass
class Paired:
    ok: bool
    why: str
    seconds: float
    record: bytes = b""


def pair(clock, radio, serial, db):
    """The factory iPhone finds the bat by its serial, connects at the
    interval iOS allows, and reads back the record the database holds for
    it.  The bat must hold exactly that."""
    t0 = clock.now()
    ci = Link.PHONE_CI
    if not radio.connect(serial, ci, timeout(Link.ADV_SLOW_S + 2.0 * ci)):
        return Paired(False, "the phone could not connect", clock.now() - t0)
    want = db.latest(serial)
    try:
        got = BleLink(radio, serial, ci, "phone").read()
    except H.LinkError as e:
        return Paired(False, "the record would not read: %s" % e, clock.now() - t0)
    if want is None:
        return Paired(False, "the database holds no record of %s" % serial, clock.now() - t0, got)
    try:
        rec = R.unpack(got)
    except R.RecordError as e:
        return Paired(False, "the bat's record does not unpack: %s" % e, clock.now() - t0, got)
    ok = got == want and rec.serial == serial
    return Paired(ok, "" if ok else "the bat's record is not the database's", clock.now() - t0, got)


# ================================================================ THE PLAN
def plan(ci=Link.CI_MIN, img_bytes=None):
    """{step: planned s} of station_test on a healthy board, from the facts.
    line.py times station 2's bat by these; check_link holds the simulated
    procedure to them."""
    from ..spec import Est
    n_img = len(image()) if img_bytes is None else img_bytes
    swd_w = Mcu.SWD_WORD_BITS / min(Probe.SWD_HZ, Mcu.SWD_HZ)
    words = (n_img + 3) // 4
    lim = supply_limit()
    req = request_s(ci)
    p = {
        "power": Board.HOLD_C * (1.0 + Board.HOLD_TOL) * P.v_fresh() / lim,
        "swd": 64 * Mcu.SWD_WORD_BITS / Probe.SWD_HZ,
        "flash": Mcu.ERASE_ALL_S + words * max(swd_w, Mcu.FLASH_WORD_S) + words * swd_w,
        "serial": 4 * max(swd_w, Mcu.FLASH_WORD_S) + 4 * swd_w,
        "advertises": Imu.START_S + scan_events() * (Link.ADV_FAST_S + Link.ADV_JITTER),
        "advertising current": current_window(),
        "connects": Link.ADV_FAST_S / 2.0 + Link.CI_STEP + ci / 2.0,
        "info": req,
        "imu": request_s(ci, Imu.ST_S),
        "stream": req + stream_s(ci),
        "hold-up": (Link.ADV_FAST_S / 2.0 + Link.CI_STEP + Link.PHONE_CI / 2.0 + 3.0 * request_s(Link.PHONE_CI)
                    + 2.0 * Link.PHONE_CI + Rules.BOUNCE_S),
        "sleep": 2.0 * request_s(Link.PHONE_CI) + 2.0 * Link.PHONE_CI + current_window(),
    }
    return p


def plan_s(ci=Link.CI_MIN):
    return sum(plan(ci).values())


def pair_plan_s():
    """s the factory iPhone's pairing and record read take, planned."""
    ci = Link.PHONE_CI
    chunks = ceil(R.SIZE / H.record_chunk())
    return Link.ADV_SLOW_S / 2.0 + Link.CI_STEP + ci / 2.0 + chunks * request_s(ci, 0.0, "phone")
