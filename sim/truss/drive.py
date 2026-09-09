"""Tier 2: the ring's DRIVE, with teeth -- can this head be turned at all?

THE QUESTION.  A ring that encircles a rod cannot have a shaft through it,
because the rod is where the shaft would go.  Everything else in the cell
is planned as though the ring simply turns; this rig is the only place
that has to make it turn, out of parts, against a load.

WHAT IT BUILDS.  The ring is a FREE BODY.  It has no hinge, no axis, no
weld -- six degrees of freedom and nothing holding it but

  * a C-channel raceway: a back wall at the tooth tips and two flanges
    that reach in to the ring's root circle, over everything but the
    mouth the work comes in through;
  * three pinions on their own hinges, at Ring.pinion_az(), with real
    involute teeth meshing real involute teeth on the rim.

If the design is wrong the ring falls out of the frame, or the teeth jam,
or a pinion re-enters the gap butting a tooth.  None of those can be
argued away, which is the point: the analysis in spec.Ring says the mesh
is consistent, the gap is a whole number of teeth and two pinions are
always in contact.  Here that is a body in a frame, and it either turns or
it does not.

FOUR THINGS THIS RIG FOUND that no arithmetic in this project would have.
Kept as a list because each was a correct-looking design that did not
turn:

  1. The rim's 72 teeth and the pinion's 12 at 6 mm pitch radius were
     module 0.556 and module 1.0.  Ring.mesh solves the counts now.
  2. Three velocity servos, one per pinion, over-constrain a closed loop.
     One motor and a belt: one actuator, two joint equalities.
  3. A massless pinion on a stiff servo hammers.  The shaft carries a
     rotor inertia and its gains come from the stepper's own stiffness
     (KP), not from a bandwidth guess -- which was 700 times too soft and
     showed as 40 degrees of "slip" that was following error.
  4. The raceway does not locate the ring toward its own mouth.  With the
     mouth at 100 degrees the ring wedged and the drive JAMMED at every
     clearance under 0.12 mm; Ring.race_mouth is the fix, and the design
     now turns from 0.05 to 0.20.

THE TEETH ARE REAL INVOLUTES, not box stand-ins.  A box tooth entering a
box space jams unless the space is made so wide that the backlash swamps
what the rig is measuring; the involute is the profile that rolls, and
generating one is twenty lines (`involute_tooth`).  Each tooth is a convex
prism, so MuJoCo's hull of it is the tooth itself.

PROXIES, and what they cost (the discipline of truss.rig):

  * BACKLASH.  A module-0.8 pair is cut with about 0.05 mm of backlash.
    The contact solver's own compliance here is of that order, so a
    backlash under it is not a backlash, it is numerical.  The rig runs
    RIG_BACKLASH (0.12 mm).  This changes the LOST MOTION -- the rig
    cannot say what the machine's positioning backlash is -- and it does
    not change whether a tooth is lost, which is what is asked.
  * SPEED.  Ring.RPM is 60, and a revolution at 60 rpm is a second of
    physics, which at this timestep is twenty of wall clock.  The rig winds
    at RIG_RPM and reports the torque referred to the ring, where the
    inertial share of the demand is 1e-4 N.m against a thread's 0.022:
    speed is not what the demand is made of.
  * FRICTION at the flanges stands in for a bearing surface nobody has
    specified yet, and is deliberately high (RAIL_MU): it makes the torque
    demand pessimistic rather than flattering.
"""
from math import pi, cos, sin, acos, tan, radians, degrees

import numpy as np
import mujoco

from .spec import Ring, mm, ring_r_in, ring_r_out, rail_r_out

# ---- rig facts and proxies (see the module docstring)
TIMESTEP = 2e-4
RIG_BACKLASH = 0.12          # mm at the pitch circle, against the cut 0.05
RIG_RPM = 60.0
RAIL_MU = 0.20               # a dry rail; the machine's runs on a film
# THE DRIVE SHAFT'S SERVO IS A STEPPER, and a stepper is a position source
# with a known stiffness: its torque runs up to the rating over roughly one
# full step of lag, so kp is the rating over a step and there is nothing to
# tune.  (Sized instead from a bandwidth guess -- 4/SPINUP_S -- kp came out
# 700 times softer, and the ring trailed its command by 40 degrees under
# nothing worse than the thread's own drag.  That is a soft servo being
# read as a slipping mesh.)
#
# kv is critical damping on what the shaft actually carries, which is its
# own rotor plus the ring's inertia reflected through the mesh -- the ring
# is 20 times the shaft, so leaving it out would under-damp by a factor of
# four.
def _shaft_inertia():
    M = Ring.mesh()
    j_ring = (Ring.MASS / 1000.0) * (M.r_pitch / 1000.0) ** 2
    return Ring.REDUCTION ** 2 * Ring.MOTOR_INERTIA + j_ring / M.ratio ** 2


KP = Ring.shaft_kp()
KV = 2.0 * (KP * _shaft_inertia()) ** 0.5
TOOTH_PTS = 10               # points down one involute flank
TRACK_EVERY = 50             # steps between turn samples (0.01 s: 3.6 deg)
RING_RGBA = "0.20 0.55 0.85 1"
PIN_RGBA = "0.85 0.55 0.15 1"
RAIL_RGBA = "0.35 0.35 0.40 1"
# collision classes: the ring meets the pinions and the rails; the pinions
# and the rails are bolted to the same frame and never meet each other
C_RING, C_PIN, C_RAIL = 1, 2, 4


# --------------------------------------------------------------- geometry
def involute_tooth(n, m, alpha_deg, r_root, r_tip, backlash, pts=TOOTH_PTS):
    """One tooth's cross-section as a convex polygon of (y, z), centred on
    the +z axis, for a gear of `n` teeth and module `m`.

    Textbook involute: the half-thickness angle at radius r is

        psi(r) = s_p/(2 r_p) + inv(alpha) - inv(alpha_r),   inv(a) = tan a - a

    with alpha_r = acos(r_b / r).  Below the base circle the flank runs
    radially in to the root, which is where the involute's own tangent
    already points, so the section stays convex and MuJoCo's hull of it is
    the tooth.
    """
    a = radians(alpha_deg)
    r_p = n * m / 2.0
    r_b = r_p * cos(a)
    inv = lambda t: tan(t) - t
    s_p = pi * m / 2.0 - backlash            # tooth thickness at the pitch circle
    half_p = s_p / (2.0 * r_p) + inv(a)      # psi at the base circle
    flank = []
    r_lo = max(r_root, r_b)
    for i in range(pts):
        r = r_lo + (r_tip - r_lo) * i / (pts - 1.0)
        psi = half_p - inv(acos(min(1.0, r_b / r)))
        flank.append((r, psi))
    if r_root < r_b - 1e-9:                  # radial run-in below the base circle
        flank.insert(0, (r_root, half_p))
    right = [(r * sin(psi), r * cos(psi)) for r, psi in flank]
    left = [(-y, z) for y, z in reversed(right)]
    return left + right


def _prism(name, poly, half_w, rgba, contype, conaffinity, mass, pos=(0, 0, 0), az=0.0):
    """A tooth: the polygon extruded along x, rotated to azimuth `az` (deg
    from straight down, the mouth's centre) and placed at `pos`."""
    c, s = cos(radians(az)), sin(radians(az))
    v = []
    for y, z in poly:
        # the polygon is drawn about +z; azimuth 0 is straight DOWN
        yy, zz = c * y + s * z, s * y - c * z
        for x in (-half_w, half_w):
            v += [x, yy + pos[1], zz + pos[2]]
    verts = " ".join("%.7f" % mm(t) for t in v)
    return (('<mesh name="%s" vertex="%s"/>' % (name, verts)),
            ('<geom name="%s" type="mesh" mesh="%s" rgba="%s" contype="%d" '
             'conaffinity="%d" mass="%.8f" condim="3" friction="0.15 0.005 0.0001" '
             'solref="0.0008 1" solimp="0.99 0.999 0.0005"/>'
             % (name, name, rgba, contype, conaffinity, mass / 1000.0)))


def pinion_phase(az, mesh=None):
    """The angle a pinion at azimuth `az` must be built at, in degrees, so
    that its tooth meets the rim's space.

    A pinion tooth points at the ring's centre when its own phase is zero;
    a ring SPACE is at the pinion when the ring's pattern is half a pitch
    along.  Writing e for the ring pattern's phase at that azimuth, the
    pinion's is e - 1/2 of its own pitch -- and the two run at the same
    rate thereafter, because n_ring / n_pinion is exactly the ratio of the
    pitches.  Straight down (az 0) this gives -half a pinion pitch, which
    is a space facing the ring's tooth: the check that it is the right way
    round.
    """
    M = Ring.mesh() if mesh is None else mesh
    p_r, p_p = 360.0 / M.n_ring, 360.0 / M.n_pinion
    e = (az % p_r) / p_r
    return (e - 0.5) * p_p


class DriveRig:
    """The ring, its raceway and its three pinions, and nothing else."""

    def __init__(self, mouth_down=True, tension=2.0, rpm=RIG_RPM, gravity=True,
                 phase_error=0.0, accel=(0.0, 0.0, 0.0)):
        """phase_error: degrees of pinion phase to build in WRONG, so the rig
        can be made to fail on purpose.  A scalar goes on the FIRST pinion
        only, because a shift applied to all three is not an error: the ring
        is a free body and simply turns half a tooth to suit it.  What
        cannot be accommodated is one pinion out of step with the belt, and
        that is the mistake a machine actually makes.  A 3-sequence sets
        each pinion's error.  accel: the gantry's acceleration, mm/s^2,
        which the ring feels as its own weight would."""
        self.M = Ring.mesh()
        self.tension = tension
        self.rpm = rpm
        self.mouth_down = mouth_down
        self.phase_error = ((float(phase_error), 0.0, 0.0)
                            if np.isscalar(phase_error)
                            else tuple(float(x) for x in phase_error))
        self.accel = np.asarray(accel, float)
        self.xml = self._build(gravity)
        self.m = mujoco.MjModel.from_xml_string(self.xml)
        self.d = mujoco.MjData(self.m)
        nm = lambda t, n: mujoco.mj_name2id(self.m, t, n)
        B, J, A, S = (mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT,
                      mujoco.mjtObj.mjOBJ_ACTUATOR, mujoco.mjtObj.mjOBJ_SITE)
        self.b_ring = nm(B, "ring")
        self.j_ring = self.m.jnt_qposadr[nm(J, "ring_f")]
        self.a_pin = nm(A, "a_p0")
        self.j_pin = [self.m.jnt_qposadr[nm(J, "p%d" % k)] for k in range(3)]
        self.g_pin = [[i for i in range(self.m.ngeom)
                       if (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
                       .startswith("p%dt" % k)] for k in range(3)]
        self.g_ring = set(i for i in range(self.m.ngeom)
                          if (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
                          .startswith("rt"))
        self.s_exit = nm(S, "exit")
        self._cmd = 0.0          # the drive shaft's commanded angle, deg
        self._w = 0.0            # and its rate, deg/s
        self._theta = 0.0        # the ring's turn, unwrapped
        self._raw = 0.0
        mujoco.mj_forward(self.m, self.d)
        self.p0 = self.centre()
        self.q0 = self.d.qpos[self.j_ring + 3:self.j_ring + 7].copy()

    # ------------------------------------------------------------- build
    def _build(self, gravity):
        M = self.M
        assets, parts = [], []
        r_root_ring = M.r_pitch - 1.25 * M.m
        r_root_pin = M.r_pinion - 1.25 * M.m
        n_gap = int(round(Ring.teeth_in_gap()))
        p_r = 360.0 / M.n_ring

        # ---------------------------------------------------- THE RING
        # a free body: nothing locates it but the raceway and the pinions
        parts.append('<body name="ring" pos="0 0 0">')
        parts.append('  <freejoint name="ring_f"/>')
        parts.append('  <site name="exit" pos="0 0 %.6f" size="0.0005"/>' % mm(-Ring.EXIT_R))
        # the web, in arc boxes, over everything but the gap
        n_web, r_in = 24, ring_r_in()
        r_mid = (r_in + r_root_ring) / 2.0
        arc = 2 * pi * r_mid / n_web
        web_mass = Ring.MASS * 0.75 / max(1, sum(
            1 for i in range(n_web)
            if abs(((360.0 * (i + 0.5) / n_web + 180.0) % 360.0) - 180.0) >= Ring.GAP / 2.0))
        for i in range(n_web):
            az = 360.0 * (i + 0.5) / n_web
            if abs(((az + 180.0) % 360.0) - 180.0) < Ring.GAP / 2.0:
                continue
            a = radians(az)
            parts.append('  <geom name="rw%d" type="box" pos="0 %.6f %.6f" '
                         'xyaxes="1 0 0 0 %.6f %.6f" size="%.6f %.6f %.6f" rgba="%s" '
                         'contype="%d" conaffinity="%d" mass="%.8f"/>'
                         % (i, mm(r_mid * sin(a)), mm(-r_mid * cos(a)),
                            cos(a), sin(a), mm(Ring.W / 2.0), mm(arc * 0.55),
                            mm((r_root_ring - r_in) / 2.0), RING_RGBA,
                            C_RING, C_PIN | C_RAIL, web_mass / 1000.0))
        # the teeth: one per pitch, none in the gap
        tooth = involute_tooth(M.n_ring, M.m, Ring.PRESSURE_ANGLE,
                               r_root_ring, ring_r_out(), RIG_BACKLASH)
        t_mass = Ring.MASS * 0.25 / (M.n_ring - n_gap)
        for j in range(M.n_ring):
            az = j * p_r
            if abs(((az + 180.0) % 360.0) - 180.0) < Ring.GAP / 2.0 - 1e-9:
                continue
            a, gm = _prism("rt%d" % j, tooth, Ring.W / 2.0, RING_RGBA,
                           C_RING, C_PIN | C_RAIL, t_mass, az=az)
            assets.append(a)
            parts.append("  " + gm)
        parts.append('</body>')

        # ------------------------------------------------- THE RACEWAY
        # back wall at the tip circle plus a running clearance, and two
        # flanges reaching in to the ring's ROOT circle -- they face the
        # web, not the teeth, which is what a flange can bear on
        n_rail = 40
        r0, r1 = ring_r_out() + Ring.RACE_CLEAR, rail_r_out()
        rr = (r0 + r1) / 2.0
        arc = 2 * pi * rr / n_rail
        ax = Ring.W / 2.0 + Ring.RACE_CLEAR + Ring.RACE_T / 2.0
        f_mid = (r_root_ring + r1) / 2.0
        for i in range(n_rail):
            az = 360.0 * (i + 0.5) / n_rail
            if abs(((az + 180.0) % 360.0) - 180.0) < Ring.race_mouth() / 2.0:
                continue
            a = radians(az)
            fr = ' friction="%g 0.005 0.0001" solref="0.0008 1"' % RAIL_MU
            parts.append('<geom name="back%d" type="box" pos="0 %.6f %.6f" '
                         'xyaxes="1 0 0 0 %.6f %.6f" size="%.6f %.6f %.6f" rgba="%s" '
                         'contype="%d" conaffinity="%d"%s/>'
                         % (i, mm(rr * sin(a)), mm(-rr * cos(a)), cos(a), sin(a),
                            mm(ax + Ring.RACE_T / 2.0), mm(arc * 0.55), mm((r1 - r0) / 2.0),
                            RAIL_RGBA, C_RAIL, C_RING, fr))
            for tag, s in (("l", -1.0), ("r", 1.0)):
                parts.append('<geom name="fl%s%d" type="box" pos="%.6f %.6f %.6f" '
                             'xyaxes="1 0 0 0 %.6f %.6f" size="%.6f %.6f %.6f" rgba="%s" '
                             'contype="%d" conaffinity="%d"%s/>'
                             % (tag, i, mm(s * ax), mm(f_mid * sin(a)), mm(-f_mid * cos(a)),
                                cos(a), sin(a), mm(Ring.RACE_T / 2.0), mm(arc * 0.55),
                                mm((r1 - r_root_ring) / 2.0), RAIL_RGBA, C_RAIL, C_RING, fr))

        # -------------------------------------------------- THE PINIONS
        p_tooth = involute_tooth(M.n_pinion, M.m, Ring.PRESSURE_ANGLE,
                                 r_root_pin, M.r_pinion + M.m, RIG_BACKLASH)
        p_p = 360.0 / M.n_pinion
        for k, az in enumerate(Ring.pinion_az()):
            a = radians(az)
            y, z = M.centre * sin(a), -M.centre * cos(a)
            phase = pinion_phase(az, M) + self.phase_error[k]
            parts.append('<body name="p%d" pos="0 %.6f %.6f">' % (k, mm(y), mm(z)))
            parts.append('  <joint name="p%d" type="hinge" axis="1 0 0" damping="1e-6" '
                         'armature="%.9f"/>' % (k, Ring.REDUCTION ** 2 * Ring.MOTOR_INERTIA))
            parts.append('  <geom name="p%dh" type="cylinder" fromto="%.6f 0 0 %.6f 0 0" '
                         'size="%.6f" rgba="%s" contype="%d" conaffinity="%d" mass="0.003"/>'
                         % (k, mm(-Ring.W / 2.0), mm(Ring.W / 2.0), mm(r_root_pin),
                            PIN_RGBA, C_PIN, C_RING))
            for j in range(M.n_pinion):
                # the ring's centre lies at azimuth az + 180 as seen FROM
                # the pinion, and that is where tooth 0 points at phase 0.
                # (Written -az first: the top pinion still meshed, because
                # at az 180 the two agree mod 360, and the rig jammed with
                # the ring turning -8 degrees against a commanded 360.)
                loc = 180.0 + az + phase + j * p_p
                asset, gm = _prism("p%dt%d" % (k, j), p_tooth, Ring.W / 2.0, PIN_RGBA,
                                   C_PIN, C_RING, 0.3, az=loc)
                assets.append(asset)
                parts.append("  " + gm)
            parts.append('</body>')

        # ---- ONE MOTOR, THREE PINIONS, ONE BELT.
        # The first version gave each pinion its own velocity servo.  Three
        # servos on a closed loop fight: the ring geared to three
        # independently commanded shafts is over-constrained, and the rig
        # jammed with all three saturated and the ring turning backwards.
        # On the machine they are one shaft -- a belt round all three off a
        # single motor -- so that is what this is: one actuator, and the
        # other two pinions coupled to it by equality.  The whole drive's
        # rating lands on that one shaft.
        f_pin = Ring.DRIVE_TORQUE * M.ratio
        act = ('    <position name="a_p0" joint="p0" kp="%.6f" kv="%.6f" '
               'forcerange="%.6f %.6f"/>' % (KP, KV, -f_pin, f_pin))
        eq = ("  <equality>\n"
              + "\n".join('    <joint name="belt%d" joint1="p%d" joint2="p0" '
                          'polycoef="0 1 0 0 0" solref="0.0004 1"/>' % (k, k)
                          for k in (1, 2))
              + "\n  </equality>")
        gv = "0 0 -9.81" if gravity else "0 0 0"
        if not self.mouth_down:
            gv = "0 0 9.81" if gravity else "0 0 0"
        return """<mujoco model="drive_rig">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="%g" integrator="implicitfast" gravity="%s" cone="elliptic">
    <flag multiccd="enable"/>
  </option>
  <default>
    <geom condim="3" friction="0.15 0.005 0.0001" solref="0.0008 1" solimp="0.99 0.999 0.0005"/>
  </default>
  <visual><headlight ambient="0.7 0.7 0.7" diffuse="0.5 0.5 0.5"/></visual>
  <asset>
%s
  </asset>
  <worldbody>
%s
  </worldbody>
%s
  <actuator>
%s
  </actuator>
</mujoco>""" % (TIMESTEP, gv, "\n".join("    " + a for a in assets),
                "\n".join("    " + p for p in parts), eq, act)

    # -------------------------------------------------------------- read
    def centre(self):
        """The ring's centre, mm, in the raceway's frame."""
        return np.array(self.d.xpos[self.b_ring]) * 1000.0

    def wander(self):
        """How far the ring's centre has left where it started, mm."""
        return float(np.linalg.norm(self.centre() - self.p0))

    def wander_axial(self):
        """...along the rod's axis, which the flanges bound."""
        return abs(float((self.centre() - self.p0)[0]))

    def wander_radial(self):
        """...across it, which the back wall bounds."""
        return float(np.linalg.norm((self.centre() - self.p0)[1:]))

    def penetration(self):
        """The deepest the solver has let anything into anything else, mm.
        The bound on wander is the running clearance PLUS this: a contact
        that yields by 0.1 mm is 0.1 mm of extra room, and pretending
        otherwise turns solver compliance into a design finding."""
        if not self.d.ncon:
            return 0.0
        return max(0.0, -min(float(self.d.contact[i].dist)
                             for i in range(self.d.ncon)) * 1000.0)

    def _raw_angle(self):
        """The ring's rotation about x from a quaternion: WRAPPED to +-180,
        because that is all a quaternion can say."""
        q = self.d.qpos[self.j_ring + 3:self.j_ring + 7]
        qi = np.zeros(4)
        mujoco.mju_negQuat(qi, self.q0)
        dq = np.zeros(4)
        mujoco.mju_mulQuat(dq, q, qi)
        ax = np.zeros(3)
        mujoco.mju_quat2Vel(ax, dq, 1.0)
        return degrees(float(ax[0]))

    def track(self):
        """Accumulate the ring's turn.  Called often enough that a step is
        far under half a revolution -- otherwise the wrap is the reading,
        and the rig reports 25 degrees for a revolution and a bit."""
        r = self._raw_angle()
        d = (r - self._raw + 180.0) % 360.0 - 180.0
        self._theta += d
        self._raw = r
        return self._theta

    def ring_angle(self):
        """Degrees the ring has turned since the start, unwrapped."""
        return self._theta

    def pinion_angle(self, k):
        return degrees(float(self.d.qpos[self.j_pin[k]]))

    def engaged(self):
        """How many pinions have a ring TOOTH within reach -- the geometric
        claim spec.Ring.pinions_meshed makes, read off the ring's actual
        angle.  Contacts come and go as teeth hand over; engagement does
        not, and engagement is what locates the ring."""
        return Ring.pinions_meshed(self.ring_angle() % 360.0)

    def meshed(self):
        """How many pinions have a tooth actually in contact with the rim."""
        n = 0
        for k in range(3):
            gs = set(self.g_pin[k])
            for i in range(self.d.ncon):
                c = self.d.contact[i]
                if (c.geom1 in gs and c.geom2 in self.g_ring) or \
                   (c.geom2 in gs and c.geom1 in self.g_ring):
                    n += 1
                    break
        return n

    def torque_at_ring(self):
        """The drive shaft's torque, referred to the ring, N.m."""
        return abs(float(self.d.actuator_force[self.a_pin])) / self.M.ratio

    # -------------------------------------------------------- the load
    def _load(self):
        """The winding tension pulling on the ring's exit guide against the
        way it turns -- a force at a point, so the raceway carries the
        reaction the way it would on the machine -- plus whatever the
        gantry's acceleration adds to the ring's weight."""
        self.d.xfrc_applied[self.b_ring] = 0.0
        m_ring = float(self.m.body_mass[self.b_ring])
        if np.any(self.accel):
            self.d.xfrc_applied[self.b_ring, :3] = -m_ring * self.accel / 1000.0
        if not self.tension:
            return
        p = self.d.site_xpos[self.s_exit]
        c = self.d.xpos[self.b_ring]
        r = p - c
        rad = np.array([0.0, r[1], r[2]])
        n = np.linalg.norm(rad)
        if n < 1e-9:
            return
        rad = rad / n
        # AGAINST the way it turns.  A point at radius r moves along
        # (0, -z, y) under a positive turn about x, so the drag is the
        # other way; written without the minus the "tension" drove the
        # ring, it ran 17 degrees ahead of its command, and the torque
        # reading was the drag it was being helped by.
        tang = -np.array([0.0, -rad[2], rad[1]]) * np.sign(self.rpm)
        f = self.tension * tang
        # a force at a point is that force at the centre of mass plus its
        # moment about it -- xfrc_applied is a wrench, not a point load
        self.d.xfrc_applied[self.b_ring, :3] += f
        self.d.xfrc_applied[self.b_ring, 3:] = np.cross(p - self.d.xipos[self.b_ring], f)

    # ------------------------------------------------------------- drive
    def commanded_ring(self):
        """Where the drive has told the ring to be, degrees -- the shaft's
        commanded angle referred through the mesh."""
        return -self._cmd * self.M.ratio

    def _ramp(self, target_ring_deg, dt):
        """Advance the shaft's command toward a ring rate of `self.rpm`,
        accelerating over Ring.SPINUP_S the way the machine does."""
        w_max = self.rpm * 6.0 / abs(self.M.ratio)            # deg/s at the shaft
        # SPINUP_S is 0 to RPM, which is an ACCELERATION, not a time to
        # whatever speed is asked for.  Read as a fixed time it made the
        # ramp to RPM_MAX two and a half times as steep and the drive
        # saturated on it -- a rig result about the rig.
        a = (Ring.RPM * 6.0 / abs(self.M.ratio)) / Ring.SPINUP_S
        want = -target_ring_deg / self.M.ratio
        s = 1.0 if want > self._cmd else -1.0
        # brake in time: the distance this rate needs to stop in
        if abs(want - self._cmd) <= self._w ** 2 / (2.0 * a):
            self._w = max(0.0, self._w - a * dt)
        else:
            self._w = min(w_max, self._w + a * dt)
        step = s * self._w * dt
        if abs(step) > abs(want - self._cmd):
            step, self._w = want - self._cmd, 0.0
        self._cmd += step

    def step(self, n, target_ring_deg=None):
        peak, least = 0.0, 3
        for i in range(n):
            if target_ring_deg is not None:
                self._ramp(target_ring_deg, TIMESTEP)
            self.d.ctrl[self.a_pin] = radians(self._cmd)
            self._load()
            mujoco.mj_step(self.m, self.d)
            peak = max(peak, self.torque_at_ring())
            least = min(least, self.meshed())
            if i % TRACK_EVERY == 0:
                self.track()
        self.track()
        return peak, least

    def hold(self, seconds):
        for i in range(int(seconds / TIMESTEP)):
            self.d.ctrl[self.a_pin] = radians(self._cmd)
            self._load()
            mujoco.mj_step(self.m, self.d)
            if i % TRACK_EVERY == 0:
                self.track()
        self.track()

    def run(self, degrees_of_ring, record=None, settle=0.15):
        """Turn the ring through `degrees_of_ring` and stop, sampling as it
        goes.

        Returns (peak torque at the ring, fewest pinions ever in contact,
        the record).  A record row is (commanded ring angle, measured ring
        angle, shaft angle, wander, pinions in contact, pinions engaged,
        radial wander, axial wander, deepest penetration, torque at the
        ring, seconds).
        """
        rec = [] if record is None else record
        self.track()
        target = self.ring_angle() + degrees_of_ring
        # RUN UNTIL THE COMMAND ARRIVES, not for a computed time.  The
        # profile is trapezoidal at a fixed acceleration, so its duration
        # depends on the distance AND the speed asked for, and a run sized
        # by "distance over speed plus SPINUP_S" stopped a fifth of the way
        # through the ramp at RPM_MAX and reported the drive saturated.
        want = -target / self.M.ratio
        v = self.rpm * 6.0 / abs(self.M.ratio)
        acc = (Ring.RPM * 6.0 / abs(self.M.ratio)) / Ring.SPINUP_S
        d = abs(want - self._cmd)
        t_move = (2.0 * v / acc + (d - v * v / acc) / v if d >= v * v / acc
                  else 2.0 * (d / acc) ** 0.5)
        total = int((t_move + settle) / TIMESTEP)
        chunk = max(1, total // 300)
        peak, least, done = 0.0, 3, 0
        while done < total:
            n = min(chunk, total - done)
            p, l = self.step(n, target)
            done += n
            peak, least = max(peak, p), min(least, l)
            rec.append((self.commanded_ring(), self.ring_angle(),
                        self.pinion_angle(0), self.wander(), self.meshed(),
                        self.engaged(), self.wander_radial(), self.wander_axial(),
                        self.penetration(), self.torque_at_ring(), done * TIMESTEP))
        return peak, least, rec
