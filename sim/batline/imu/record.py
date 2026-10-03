"""What station 4 writes to the bat after calibration, and what it keeps in
the factory's database.  Production: numpy and the standard library only;
the station imports it, and M4's firmware reads the same bytes.

    CalRecord      the 512 bytes written to the bat, as fields
    pack, unpack   bytes and back; unpack refuses anything it cannot trust
    resolution     float32's step at each field's largest plausible value,
                   against the 1 % of the target that field serves
    BatLink        how bytes reach the bat's flash (M4's firmware implements
                   it); MemoryLink is the simulation's, with planted bit flips
    FactoryDB      every record, its covariance and its inputs' hash, by
                   serial, kept for ever (sqlite3)
    commit         write, read back, compare; once more if they differ; then
                   the database
    restore        write the database's latest good record back to a bat

THE RECORD, version 1.0: 512 bytes, little-endian, struct-packed.
Offsets in bytes; every value field float32.

  off  size  type   field
    0     4  4s     magic b"BATC"
    4     1  u8     major version (1): a reader refuses any other
    5     1  u8     minor version (0): later minors only append fields, in
                    the pad, so a reader of minor 0 reads a later minor's
                    record and keeps what it does not know (CalRecord.tail)
    6     2  u16    length: 512, the whole record
    8    16  16s    serial, ASCII, NUL-padded
   24     4  u32    firmware version, as the line's firmware numbers it
   28     4  u32    calibration date: days since 1970-01-01 (the Unix day), UTC
   32     2  u16    station
   34     4  u32    rig-calibration id: the rig calibration the poses rest on
   38     4  u32    plan hash: plan.CalPlan.hash's 8 hex digits
   42     2  i16    T_cal, 0.01 degC: the die's mean temperature during
                    calibration, 25 + fit.CalResult.dT_cal, rounded; the
                    biases are stated at it as rounded, so the app's
                    b + c (T - T_cal) is the fit's whatever the rounding
   44     1  u8     flags: bit 0 set -- the biases are this power-up's (each
                    power-up redraws them, Imu.*_POWERUP, and the app
                    re-learns them in play); bits 1-7 zero in minor 0
   45     1  u8     frame of the mounting and the lever: 0 the clamp's (the
                    fit's own), 1 the bat's (from_result's to_bat)
   46     2  --     zero: puts every float on a 4-byte boundary
   48    12  3 f32  accel bias at T_cal, mg, chip axes
   60    24  6 f32  accel scale and cross-axis, ppm: H - I of the polar split
                    M_a^T = U H (fit.terms), xx yy zz xy xz yz
   84    12  3 f32  gyro bias at T_cal, dps, chip axes
   96    36  9 f32  gyro scale and cross-axis, ppm: E_g = M_g U - I, row-major
  132    36  9 f32  gyro g-sensitivity, dps/g, row-major, chip axes on both
                    sides: G = G' U, G' the fit's (clamp-frame force)
  168    12  3 f32  accel bias temperature coefficient, mg/degC
  180    12  3 f32  gyro bias temperature coefficient, dps/degC
  192     4  1 f32  clock: rho - 1, ppm -- with v this field, the board's
                    sample interval is (1 + v * 1e-6) / ODR of station
                    (true) time (rho / ODR, rho the fit's)
  196     4  1 f32  clock's temperature coefficient, ppm/degC: 0 until one is
                    measured [VERIFY: the part's oscillator against
                    temperature]
  200    16  4 f32  mounting: the unit quaternion (w x y z, w >= 0, Hamilton)
                    of U, which turns chip axes into the frame's
  216    12  3 f32  lever: the IMU's sensing centre, mm, in the frame
  228    36  9 f32  1 sigma per judged group (fit.GROUPS, in its order): the
                    largest of the group's terms, in its unit; the
                    dimensionless groups in ppm
  264    20  5 f32  chi-square per dof of each verdict block (BLOCKS, in its
                    order); NaN for a block the fit did not have
  284   224  --     pad: zero in minor 0; later minors' fields go here
  508     4  u32    zlib.crc32 of bytes 0..507: the header too, so a flipped
                    version byte cannot pass as a newer minor

The CRC (IEEE 802.3's CRC-32) has Hamming distance 4 over a record this
long, so it catches every error of up to three bits, and every burst up to
32 bits (Koopman 2002, "32-bit cyclic redundancy codes for Internet
applications", DSN): what flash wear and a noisy link do.

NOT YET IN THE RECORD.  The README also stores the bat's geometry -- its
axis, length, sweet spot, and the two bands' places and colours -- which
the cameras measure (station 4, step 2) and M3 does not yet.  A minor
version appends it in the pad, and the frame byte turns to 1 when the
bat's frame exists to state the mounting in.

THE UNITS.  The datasheet's (mg, dps, ppm, deg, mm) -- what the app's
filter and a person reading a returned bat's record think in.  The
design's rule for them: float32's step (ULP) at a field's largest
plausible value is at most 1 % of the target that field serves
(resolution()).  float32 carries 24 bits, so that holds wherever the
largest value is under 2^23 / 100 of the target; the closest field (the
mounting's quaternion) is about 150 times inside it on the kid's plan.
T_cal, an integer, is held to the same rule through what its count moves:
where the biases are stated, over which a bias carries c's own
uncertainty (resolution(), kind "i16").  Had the biases been stated at
the unrounded temperature, a count would move them by c itself, and at
the gyro's largest plausible c a 0.01 degC count would be 1.13 times over
the rule.
"""
import abc
import datetime
import hashlib
import sqlite3
import struct
import zlib
from dataclasses import dataclass, field, fields, is_dataclass
from math import sqrt, radians

import numpy as np

from ..spec import Imu, Quality, Site
from ..rig.calib import chi2_quantile
from . import fit as F
from . import kin as K
from .part import G0, D2R, T_REF

MAGIC = b"BATC"
MAJOR, MINOR = 1, 0
SIZE = 512
T_LSB = 0.01                       # degC a count of T_cal: the record's format
SESSION = 0x01                     # flags bit 0: the biases are this power-up's
FRAME_CLAMP, FRAME_BAT = 0, 1
BLOCKS = F.BLOCKS                                   # fit.fit's blocks, in its order

HEAD = struct.Struct("<4sBBH")                     # magic, major, minor, length
IDENT = struct.Struct("<16sIIHIIhBB2x")            # serial .. frame, then two zero bytes
FLOAT_AT = HEAD.size + IDENT.size                  # 48

# (name, count, unit): the float32 fields, in the record's order
FIELDS = (("acc_bias", 3, "mg"),
          ("acc_S", 6, "ppm"),
          ("gyro_bias", 3, "dps"),
          ("gyro_E", 9, "ppm"),
          ("gyro_G", 9, "dps/g"),
          ("acc_tc", 3, "mg/degC"),
          ("gyro_tc", 3, "dps/degC"),
          ("rho", 1, "ppm"),
          ("rho_tc", 1, "ppm/degC"),
          ("quat", 4, ""),
          ("lever", 3, "mm"),
          ("sigma", len(F.GROUPS), "group"),
          ("chi2", len(BLOCKS), ""))
NF = sum(n for _, n, _ in FIELDS)
CRC_AT = SIZE - 4
PAD_AT = FLOAT_AT + 4 * NF
PAD = CRC_AT - PAD_AT
if PAD < 0:
    raise ImportError("the record's fields overrun its %d bytes" % SIZE)
PPM = 1e6
_SPAN = {}
_o = 0
for _n, _c, _u in FIELDS:
    _SPAN[_n] = slice(_o, _o + _c)
    _o += _c
# each judged term's place in the record (fit.terms' order): (field, index)
_S_ORDER = [(i, i) for i in range(3)] + list(F._OFF)               # acc_S's elements as (row, col)
_TERM_AT = ([("acc_bias", i) for i in range(3)] + [("acc_S", i) for i in range(6)]
            + [("gyro_bias", i) for i in range(3)]
            + [("gyro_E", 4 * i) for i in range(3)] + [("gyro_E", 3 * i + j) for i, j in F._OFF6]
            + [("quat", None)] * 3 + [("lever", i) for i in range(3)] + [("rho", 0)])
if len(_TERM_AT) != F.NT:
    raise ImportError("record.py maps %d judged terms, fit.terms has %d" % (len(_TERM_AT), F.NT))


class RecordError(ValueError):
    """A record that cannot be written as asked, or cannot be trusted as read."""


# ================================================================ ROTATIONS
def quat_of(R):
    """The unit quaternion (w, x, y, z), w >= 0, of rotation R (Shepperd
    1978: from the largest of the four squared components, so no division
    by a small one)."""
    R = np.asarray(R, float)
    t = np.trace(R)
    d = np.array([1.0 + t, 1.0 + 2.0 * R[0, 0] - t, 1.0 + 2.0 * R[1, 1] - t, 1.0 + 2.0 * R[2, 2] - t])
    i = int(np.argmax(d))
    s = 2.0 * sqrt(d[i])
    if i == 0:
        q = [s / 4.0, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif i == 1:
        q = [(R[2, 1] - R[1, 2]) / s, s / 4.0, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif i == 2:
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4.0, (R[1, 2] + R[2, 1]) / s]
    else:
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4.0]
    q = np.array(q)
    q = -q if q[0] < 0.0 else q
    return q / np.linalg.norm(q)


def rot_of(q):
    """The rotation of quaternion q (w, x, y, z), normalised first: a
    float32 quaternion is unit only to its rounding."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(np.asarray(q, float))
    return np.array([[1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
                     [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
                     [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)]])


# ================================================================ THE RECORD
def unix_day(date):
    """Days since 1970-01-01 of a datetime.date (or an int already one)."""
    if isinstance(date, datetime.datetime):
        date = date.astimezone(datetime.timezone.utc).date() if date.tzinfo else date.date()
    if isinstance(date, datetime.date):
        return (date - datetime.date(1970, 1, 1)).days
    return int(date)


def _hash32(h):
    """CalPlan.hash (8 hex digits) or an int, as a u32."""
    if isinstance(h, str):
        if len(h) != 8 or any(c not in "0123456789abcdefABCDEF" for c in h):
            raise RecordError("a plan hash is 8 hex digits, not %r" % h)
        return int(h, 16)
    return int(h)


@dataclass(eq=False)
class CalRecord:
    """The record's fields, each value as float32 -- exactly what the bat
    holds (see the module's table for units and frames)."""
    serial: str
    firmware: int
    date: int                      # Unix day
    station: int
    rig_id: int
    plan_hash: int                 # u32
    t_cal: int                     # T_cal in counts of T_LSB degC
    flags: int
    frame: int
    acc_bias: np.ndarray
    acc_S: np.ndarray
    gyro_bias: np.ndarray
    gyro_E: np.ndarray
    gyro_G: np.ndarray
    acc_tc: np.ndarray
    gyro_tc: np.ndarray
    rho: np.ndarray
    rho_tc: np.ndarray
    quat: np.ndarray
    lever: np.ndarray
    sigma: np.ndarray
    chi2: np.ndarray
    minor: int = MINOR
    tail: bytes = field(default=bytes(PAD))        # the pad: a later minor's fields, kept as read

    def __post_init__(self):
        for name, n, _ in FIELDS:
            v = np.asarray(getattr(self, name), dtype=np.float32).ravel()
            if v.size != n:
                raise RecordError("%s holds %d values, not %d" % (name, v.size, n))
            setattr(self, name, v)
        self.tail = bytes(self.tail)
        if len(self.tail) != PAD:
            raise RecordError("the pad holds %d bytes, not %d" % (len(self.tail), PAD))

    @property
    def T_cal(self):
        """degC: the die's mean temperature during calibration."""
        return self.t_cal * T_LSB

    def values(self):
        """(NF,) float32: every value field, in the record's order."""
        return np.concatenate([getattr(self, n) for n, _, _ in FIELDS])

    @classmethod
    def from_result(cls, res, serial, firmware, date, station, rig_id, plan_hash, to_bat=None):
        """The record of a fit.CalResult.  to_bat (R, p mm), the clamp's
        frame into the bat's, states the mounting and the lever in the
        bat's frame; without it they stay in the clamp's (the fit's)."""
        th = np.asarray(res.theta, float)
        if not np.all(np.isfinite(th)):
            raise RecordError("the fit's parameters are not all finite")
        U, H = F.polar(th[F.MA].reshape(3, 3).T)
        S = H - np.eye(3)
        Eg = th[F.MG].reshape(3, 3) @ U - np.eye(3)
        dT = float(res.dT_cal)
        r = th[F.RR] * 1000.0
        if to_bat is None:
            frame, Uf, rf = FRAME_CLAMP, U, r
        else:
            Rb, pb = np.asarray(to_bat[0], float), np.asarray(to_bat[1], float)
            # a rotation built in float64 is one to rounding; the drawing's
            # own test (plan.board_nominal) allows 1e-9
            if abs(np.linalg.det(Rb) - 1.0) > 1e-9 or np.max(np.abs(Rb.T @ Rb - np.eye(3))) > 1e-9:
                raise RecordError("to_bat's R is not a rotation")
            frame, Uf, rf = FRAME_BAT, Rb @ U, Rb @ r + pb
        if isinstance(serial, (bytes, bytearray)):
            serial = bytes(serial).rstrip(b"\0").decode("ascii", "replace")
        sd = np.sqrt(np.diag(res.Sigma_terms))
        sig = [np.max(sd[F.GROUP_OF == k]) * (PPM if g[3] == "" else 1.0) for k, g in enumerate(F.GROUPS)]
        chi = [res.blocks[b][0] / res.blocks[b][1] if b in res.blocks and res.blocks[b][1] > 0 else np.nan
               for b in BLOCKS]
        # the biases are stated at T_cal as stored (rounded), not at dT_cal:
        # then the app's b + c (T - T_cal) is the fit's b + c (T - 25)
        # whatever the rounding, which moves only where they are stated
        t_cal = int(round((T_REF + dT) / T_LSB))
        if not -2 ** 15 <= t_cal < 2 ** 15:
            raise RecordError("T_cal %.2f degC does not fit the record's i16" % (T_REF + dT))
        dTs = t_cal * T_LSB - T_REF
        rec = cls(serial=str(serial), firmware=int(firmware), date=unix_day(date), station=int(station),
                  rig_id=int(rig_id), plan_hash=_hash32(plan_hash), t_cal=t_cal, flags=SESSION, frame=frame,
                  acc_bias=(th[F.BA] + th[F.CA] * dTs) / G0 * 1000.0,
                  acc_S=np.array([S[i, j] for i, j in _S_ORDER]) * PPM,
                  gyro_bias=(th[F.BG] + th[F.CG] * dTs) / D2R,
                  gyro_E=Eg.ravel() * PPM,
                  gyro_G=(th[F.GS].reshape(3, 3) @ U).ravel() * G0 / D2R,
                  acc_tc=th[F.CA] / G0 * 1000.0,
                  gyro_tc=th[F.CG] / D2R,
                  rho=[(th[F.RHO] - 1.0) * PPM], rho_tc=[0.0],
                  quat=quat_of(Uf), lever=rf, sigma=sig, chi2=chi)
        v = rec.values()
        live = np.ones(NF, bool)
        live[_SPAN["chi2"]] = False                # NaN there is "no such block"
        if not np.all(np.isfinite(v[live])):
            bad = sorted({FIELDS_OF[i] for i in np.flatnonzero(~np.isfinite(v) & live)})
            raise RecordError("not finite as float32: %s" % ", ".join(bad))
        return rec

    def rotation(self, to_bat=None):
        """U: chip axes into the clamp's frame, whatever frame is stored
        (to_bat, the same transform from_result was given, undoes a bat-frame
        record)."""
        R = rot_of(self.quat)
        if self.frame == FRAME_BAT:
            if to_bat is None:
                raise RecordError("a bat-frame record needs to_bat to give the clamp's frame")
            R = np.asarray(to_bat[0], float).T @ R
        return R

    def lever_clamp(self, to_bat=None):
        """mm, the clamp's frame."""
        r = self.lever.astype(float)
        if self.frame == FRAME_BAT:
            if to_bat is None:
                raise RecordError("a bat-frame record needs to_bat to give the clamp's frame")
            r = np.asarray(to_bat[0], float).T @ (r - np.asarray(to_bat[1], float))
        return r

    def judged_terms(self, U_ref, dT_cal=None, to_bat=None):
        """(NT,): the stored terms as fit.terms states them -- the biases
        at 25 + dT_cal degC (moved there from T_cal by the stored
        coefficients; at T_cal itself without it), the mounting as a
        rotation vector (deg) from U_ref, the lever in the clamp's frame --
        for comparison with fit.terms(theta, dT_cal, U_ref)."""
        move = 0.0 if dT_cal is None else T_REF + float(dT_cal) - self.T_cal
        S = np.zeros((3, 3))
        for (i, j), v in zip(_S_ORDER, self.acc_S.astype(float) / PPM):
            S[i, j] = S[j, i] = v
        Eg = self.gyro_E.astype(float).reshape(3, 3) / PPM
        out = np.empty(F.NT)
        out[0:3] = self.acc_bias.astype(float) + self.acc_tc.astype(float) * move
        out[3:6] = np.diag(S)
        out[6:9] = [S[i, j] for i, j in F._OFF]
        out[9:12] = self.gyro_bias.astype(float) + self.gyro_tc.astype(float) * move
        out[12:15] = np.diag(Eg)
        out[15:21] = [Eg[i, j] for i, j in F._OFF6]
        out[21:24] = np.degrees(K.log_so3(self.rotation(to_bat) @ np.asarray(U_ref, float).T))
        out[24:27] = self.lever_clamp(to_bat)
        out[27] = float(self.rho[0]) / PPM
        return out


FIELDS_OF = [n for n, c, _ in FIELDS for _ in range(c)]


def crc(b):
    """The record's CRC: bytes 0 .. CRC_AT-1, the header included."""
    return zlib.crc32(bytes(b[:CRC_AT])) & 0xFFFFFFFF


def pack(rec, crc_of=None):
    """The record's 512 bytes.  `crc_of` swaps the checksum for the
    must-fails; production never passes it."""
    crc_of = crc if crc_of is None else crc_of
    try:
        s = rec.serial.encode("ascii")
    except UnicodeEncodeError:
        raise RecordError("serial %r is not ASCII" % rec.serial)
    if len(s) > 16:
        raise RecordError("serial %r is longer than 16 bytes" % rec.serial)
    try:
        head = HEAD.pack(MAGIC, MAJOR, rec.minor, SIZE)
        ident = IDENT.pack(s, rec.firmware, rec.date, rec.station, rec.rig_id, rec.plan_hash, rec.t_cal,
                           rec.flags, rec.frame)
    except struct.error as e:
        raise RecordError("a header field is out of its range: %s" % e)
    body = head + ident + rec.values().astype("<f4").tobytes() + rec.tail
    return body + struct.pack("<I", crc_of(body))


def unpack(b, crc_of=None):
    """CalRecord from bytes, or RecordError: truncated or overlong, a bad
    magic, an unknown major version, a length field that disagrees with
    the bytes or with SIZE, or a CRC that does not match.  A newer minor of
    the same major unpacks: its appended fields are kept in `tail`.
    `crc_of` swaps the checksum for the must-fails; production never
    passes it."""
    crc_of = crc if crc_of is None else crc_of
    b = bytes(b)
    if len(b) < HEAD.size:
        raise RecordError("truncated: %d bytes, fewer than the %d-byte header" % (len(b), HEAD.size))
    magic, major, minor, length = HEAD.unpack_from(b, 0)
    if magic != MAGIC:
        raise RecordError("bad magic %r" % magic)
    if major != MAJOR:
        raise RecordError("unknown major version %d.%d (this reader knows %d)" % (major, minor, MAJOR))
    if length != SIZE:
        raise RecordError("the length field says %d, a version %d record is %d" % (length, MAJOR, SIZE))
    if len(b) != length:
        raise RecordError("%d bytes, the length field says %d" % (len(b), length))
    want = struct.unpack_from("<I", b, CRC_AT)[0]
    got = crc_of(b[:CRC_AT])
    if got != want:
        raise RecordError("CRC %08x, the record says %08x" % (got, want))
    s, fw, day, st, rig, ph, tc, flags, frame = IDENT.unpack_from(b, HEAD.size)
    try:
        serial = s.rstrip(b"\0").decode("ascii")
    except UnicodeDecodeError:
        raise RecordError("serial %r is not ASCII" % s)
    v = np.frombuffer(b, dtype="<f4", count=NF, offset=FLOAT_AT).astype(np.float32)
    kw = {n: v[_SPAN[n]] for n, _, _ in FIELDS}
    return CalRecord(serial=serial, firmware=fw, date=day, station=st, rig_id=rig, plan_hash=ph, t_cal=tc,
                     flags=flags, frame=frame, minor=minor, tail=b[PAD_AT:CRC_AT], **kw)


# ================================================================ RESOLUTION
@dataclass
class Resolution:
    """One field: its representation's step (float32's ULP at its largest
    plausible value; an integer's count) against 1 % of the target it
    serves, both in the field's unit."""
    field: str
    unit: str
    kind: str                      # "f32", or "i16" for T_cal
    magnitude: float               # the largest plausible value
    ulp: float                     # the representation's step there
    tol: float                     # target / 100 (or spread / 100), the binding element's
    basis: str

    @property
    def ok(self):
        return self.ulp <= self.tol


def accept_z(n_blocks=len(BLOCKS)):
    """How many spreads from nominal fit.verdict lets a judged term lie, the
    fit's sigma included (fit.accept_z).  A record is written only for a bat the verdict passed."""
    return F.accept_z(n_blocks + 1)


def _ulp(m):
    return float(np.spacing(np.float32(abs(m))))


def resolution(plan, res=None, to_bat=None):
    """{field: Resolution} for every stored float32 field, and T_cal.

    Largest plausible value: a judged term's nominal (plan.theta0) plus
    accept_z() of its spread and its threshold together, sqrt(plan.spread^2
    + plan.s^2) -- the verdict refuses any bat further out, or any whose
    sigma passes its threshold; a term with no target, accept_z() of its datasheet spread
    (the same tail); a sigma, its threshold (plan.s, the verdict refuses a
    larger one); a quaternion component, 1; a block's chi-square per dof,
    the verdict's limit (needs res, for the dof).  With to_bat (R, p) the
    lever is stored as R r + p; the verdict bounds each component of r on
    its own, so r lies in a box, nominal r0 plus or minus z f (f the
    spreads above), and a stored component's largest over it is
    |R r0 + p| + z |R| f, |R| taken elementwise -- the clamp frame's
    bound when R is I and p zero.  Tolerance: tau / 100 of
    the group the field serves (fit.GROUPS, imu_fusion_sim.CALIBRATED), or
    its datasheet spread / 100 where there is no target.  A quaternion
    component's error dq turns the rotation by up to 2 dq rad.  T_cal's
    count moves where the biases are stated, which costs a bias c's own
    1 sigma per degree (at most the datasheet's, c's prior)."""
    z = accept_z()
    tau = F.TAU
    unit = {k: (PPM if g[3] == "" else 1.0) for k, g in enumerate(F.GROUPS)}
    nom = F.terms(np.asarray(plan.theta0, float), 0.0, F.mounting(plan.theta0))
    far = np.sqrt(np.asarray(plan.spread, float) ** 2 + np.asarray(plan.s, float) ** 2)
    big = np.abs(nom) + z * far
    if to_bat is not None:
        Rb, pb = np.asarray(to_bat[0], float), np.asarray(to_bat[1], float)
        big[24:27] = np.abs(Rb @ nom[24:27] + pb) + z * (np.abs(Rb) @ far[24:27])
    rows = {}

    def put(name, mags, tols, basis, kind="f32", step=None):
        u = dict((n, un) for n, _, un in FIELDS).get(name, "degC")
        ulps = np.array([_ulp(m) for m in mags]) if step is None else np.full(len(mags), step)
        k = int(np.argmax(ulps / np.asarray(tols, float)))
        rows[name] = Resolution(name, u, kind, float(np.max(mags)), float(ulps[k]), float(tols[k]), basis)

    # the judged terms, each in its record field's unit
    per = {}
    for j, (fname, _) in enumerate(_TERM_AT):
        g = F.GROUP_OF[j]
        if fname == "quat":
            continue
        per.setdefault(fname, ([], []))
        per[fname][0].append(big[j] * unit[g])
        per[fname][1].append(tau[j] * unit[g] / 100.0)
    for fname, (m, t) in per.items():
        put(fname, m, t, "nominal + %.2f spreads; tau/100" % z)
    mount = F.GROUP_OF[_TERM_AT.index(("quat", None))]
    put("quat", [1.0] * 4, [radians(F.GROUPS[mount][2] / 100.0) / 2.0] * 4, "|q_i| <= 1; mount tau/100 rad / 2")
    gs, atc, gtc = Imu.GYRO_GSENS, Imu.ACC_TC * 1000.0, Imu.GYRO_TC
    put("gyro_G", [z * gs] * 9, [gs / 100.0] * 9, "%.2f datasheet spreads; spread/100" % z)
    put("acc_tc", [z * atc] * 3, [atc / 100.0] * 3, "%.2f datasheet spreads; spread/100" % z)
    put("gyro_tc", [z * gtc] * 3, [gtc / 100.0] * 3, "%.2f datasheet spreads; spread/100" % z)
    # no datasheet figure for the oscillator against temperature: its
    # whole tolerance over the die's excursion (plan.spreads' dT), 1 sigma
    dT = Site.AMBIENT_SD + Imu.SELF_HEAT
    rs = Imu.ODR_TOL * PPM / dT
    put("rho_tc", [z * rs], [rs / 100.0], "ODR_TOL over %.0f degC, %.2f of it; /100" % (dT, z))
    gk = np.array([F.GROUPS[k][2] for k in range(len(F.GROUPS))])
    smax = [np.max(np.asarray(plan.s)[F.GROUP_OF == k]) * unit[k] for k in range(len(F.GROUPS))]
    put("sigma", smax, gk * np.array([unit[k] for k in range(len(F.GROUPS))]) / 100.0, "threshold s; tau/100")
    if res is not None:
        m, t = [], []
        n_tests = len(res.blocks) + 1
        for b in BLOCKS:
            if b not in res.blocks:
                continue
            dof = res.blocks[b][1]
            if dof <= 0:
                continue
            a = Quality.CAL_QUAL_ALPHA / n_tests
            lim = chi2_quantile(dof, a) if b == "poses" else dof * F.f_quantile(dof, 3.0 * max(res.nd_dof, 1), a)
            m.append(lim / dof)
            t.append(sqrt(2.0 / dof) / 100.0)          # the statistic's own spread over healthy bats
        if m:
            put("chi2", m, t, "the verdict's limit; sqrt(2/dof)/100")
    # T_cal, an integer.  The biases are stated at T_cal as stored, so its
    # rounding moves only where they are stated: by up to a count, over
    # which a bias carries c's own uncertainty -- at most its prior's, the
    # datasheet spread, since the data only tighten it
    tol_T = min(F.GROUPS[0][2] / 100.0 / atc, F.GROUPS[3][2] / 100.0 / gtc)
    put("T_cal", [T_REF + z * Site.AMBIENT_SD + Imu.SELF_HEAT], [tol_T], "a count moves where a bias is "
        "stated, by c's 1 sigma a degC; tau/100 over it", kind="i16", step=T_LSB)
    return rows


# ================================================================ THE LINK
class BatLink(abc.ABC):
    """How the station reaches the bat's calibration flash.  M4's firmware
    implements it (over the pogo block's serial, or Bluetooth); the station
    only ever writes a whole record and reads one back."""

    @abc.abstractmethod
    def write(self, data):
        """Store `data` (bytes) as the bat's record."""

    @abc.abstractmethod
    def read(self):
        """bytes: the bat's record as it holds it."""


class MemoryLink(BatLink):
    """The simulation's bat: a bytes flash.  corrupt() plants bit flips in
    the next writes (the link's or the flash's), damage() in what is
    stored (a record gone bad in the field)."""

    def __init__(self, flash=b""):
        self.flash = bytes(flash)
        self.writes = 0
        self.reads = 0
        self._plant = []

    def corrupt(self, bits, writes=1):
        """Flip bit indices `bits` (byte k // 8, bit k % 8) in each of the
        next `writes` writes."""
        self._plant += [tuple(int(k) for k in bits)] * int(writes)

    def damage(self, bits):
        self.flash = _flip(self.flash, bits)

    def write(self, data):
        self.writes += 1
        bits = self._plant.pop(0) if self._plant else ()
        self.flash = _flip(bytes(data), bits)

    def read(self):
        self.reads += 1
        return self.flash


def _flip(b, bits):
    out = bytearray(b)
    for k in bits:
        out[k // 8] ^= 1 << (k % 8)
    return bytes(out)


# ================================================================ THE DATABASE
@dataclass
class Entry:
    id: int
    serial: str
    record: bytes
    Sigma: np.ndarray
    inputs_hash: str
    stored_at: str                 # ISO 8601, UTC


class FactoryDB:
    """Every record the line writes, with the fit's full covariance and a
    hash of what the fit was given, by serial.  History only grows: each
    store is a new row with a larger id, and the database itself refuses
    an UPDATE, a DELETE, or an INSERT onto an id it holds.  The last is
    a REPLACE (INSERT OR REPLACE, the usual upsert): it deletes the stored
    row without firing a delete trigger unless the connection turns on
    PRAGMA recursive_triggers, so records_stay alone lets it through.  The
    triggers live in the file and the schema runs on every open, so a
    database made before one existed gains it the first time it is
    opened here."""

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            serial TEXT NOT NULL,
            record BLOB NOT NULL,
            sigma BLOB NOT NULL,
            sigma_shape TEXT NOT NULL,
            inputs_hash TEXT NOT NULL,
            stored_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS records_serial ON records (serial, id);
        CREATE TRIGGER IF NOT EXISTS records_keep BEFORE UPDATE ON records
            BEGIN SELECT RAISE(ABORT, 'a stored record is never changed'); END;
        CREATE TRIGGER IF NOT EXISTS records_stay BEFORE DELETE ON records
            BEGIN SELECT RAISE(ABORT, 'a stored record is never deleted'); END;
        CREATE TRIGGER IF NOT EXISTS records_fresh BEFORE INSERT ON records
            WHEN EXISTS (SELECT 1 FROM records WHERE id = NEW.id)
            BEGIN SELECT RAISE(ABORT, 'a stored record is never replaced'); END;
    """

    def __init__(self, path=":memory:"):
        self.path = str(path)
        self.con = sqlite3.connect(self.path)
        self.con.executescript(self.SCHEMA)
        self.con.commit()

    def close(self):
        self.con.close()

    def store(self, serial, record, Sigma, inputs_hash):
        """The new row's id.  The record must unpack and carry `serial`."""
        record = bytes(record)
        rec = unpack(record)
        if rec.serial != serial:
            raise RecordError("record of %r stored under %r" % (rec.serial, serial))
        S = np.ascontiguousarray(Sigma, dtype="<f8")
        at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cur = self.con.execute("INSERT INTO records (serial, record, sigma, sigma_shape, inputs_hash, stored_at) "
                               "VALUES (?, ?, ?, ?, ?, ?)",
                               (serial, record, S.tobytes(), ",".join(str(n) for n in S.shape), str(inputs_hash), at))
        self.con.commit()
        return int(cur.lastrowid)

    def history(self, serial):
        """[Entry], oldest first."""
        rows = self.con.execute("SELECT id, serial, record, sigma, sigma_shape, inputs_hash, stored_at FROM records "
                                "WHERE serial = ? ORDER BY id", (serial,)).fetchall()
        out = []
        for i, s, rec, sig, shape, h, at in rows:
            shp = tuple(int(n) for n in shape.split(",")) if shape else ()
            out.append(Entry(int(i), s, bytes(rec), np.frombuffer(sig, dtype="<f8").reshape(shp).copy(), h, at))
        return out

    def latest(self, serial):
        """The newest record of `serial` that unpacks, as bytes, or None."""
        for (rec,) in self.con.execute("SELECT record FROM records WHERE serial = ? ORDER BY id DESC", (serial,)):
            try:
                unpack(rec)
            except RecordError:
                continue
            return bytes(rec)
        return None


def digest(obj):
    """sha256 hex of anything the fit is given (fit.CalInputs): arrays by
    dtype, shape and bytes; dataclasses, objects, lists and dicts by their
    contents in a fixed order; numbers by their exact repr."""
    h = hashlib.sha256()
    seen = set()

    def walk(o):
        if isinstance(o, np.ndarray) or isinstance(o, np.generic):
            a = np.ascontiguousarray(o)
            h.update(b"A" + a.dtype.str.encode() + repr(a.shape).encode())
            h.update(a.tobytes())
        elif o is None or isinstance(o, (bool, int, float, str, bytes, complex)):
            h.update(b"S" + type(o).__name__.encode() + repr(o).encode())
        elif isinstance(o, (list, tuple)):
            h.update(b"L%d" % len(o))
            for x in o:
                walk(x)
        elif isinstance(o, dict):
            h.update(b"D%d" % len(o))
            for k in sorted(o, key=repr):
                walk(k)
                walk(o[k])
        else:
            if id(o) in seen:
                h.update(b"R")
                return
            seen.add(id(o))
            h.update(b"O" + type(o).__name__.encode())
            names = [f.name for f in fields(o)] if is_dataclass(o) else sorted(vars(o))
            for n in names:
                h.update(n.encode())
                walk(getattr(o, n))
    walk(obj)
    return h.hexdigest()


# ================================================================ COMMIT
@dataclass
class Written:
    writes: int                    # 1, or 2 when the first read back differed
    bad_bits: list                 # per write, how many bits read back wrong
    row: int = 0                   # the database row, for commit


def _write_verified(link, b):
    """Write b, read it back; once more if it differs; RecordError if the
    second read back differs too."""
    bad = []
    for _ in range(2):
        link.write(b)
        back = bytes(link.read())
        n = _bits_differing(back, b)
        bad.append(n)
        if n == 0:
            return Written(len(bad), bad)
    raise RecordError("the bat read back %s wrong bits after %d writes" % (bad, len(bad)))


def _bits_differing(a, b):
    if len(a) != len(b):
        return 8 * max(len(a), len(b))
    x = np.frombuffer(a, np.uint8) ^ np.frombuffer(b, np.uint8)
    return int(np.unpackbits(x).sum())


def commit(link, db, rec, Sigma, inputs_hash):
    """Pack rec, write it to the bat over link, read it back and compare;
    rewrite once if they differ; RecordError if they still do (nothing is
    stored).  Then store it in db.  Returns Written."""
    b = pack(rec)
    w = _write_verified(link, b)
    w.row = db.store(rec.serial, b, Sigma, inputs_hash)
    return w


def restore(link, db, serial):
    """Write the database's latest good record of `serial` back to the bat
    (a corrupted bat, a replaced board's flash).  Returns Written."""
    b = db.latest(serial)
    if b is None:
        raise RecordError("the database holds no good record of %r" % serial)
    return _write_verified(link, b)
