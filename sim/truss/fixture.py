"""The fixture, derived from the truss: pins, cradles, posts, magazines,
and the poses the loader needs to fill it.

GEOMETRY IS HELD BY THE FIXTURE (brief 4.1), so the fixture is the part of
the cell that knows where every rod goes -- and it is computed from the
truss, not drawn, because "one fixture per truss geometry, machinable in
an afternoon" (brief 4.4) is only true if the afternoon is a script.

Three things this module decides that the brief leaves open:

  * PINS GO BETWEEN JOINTS.  A 50 mm pin grid on a chord with joints
    every `run` lands some pins inside the ring's swept band, where the
    ring would strike them.  Pins therefore keep a computed axial distance
    from every joint, and the grid is phased to need as few moves as
    possible.
  * CRADLES SIT OUTSIDE THE RING'S SWEEP.  A cradle holds a diagonal's
    end, and the end is where the ring turns.  The cradle is pushed along
    the diagonal until it clears the ring's spool torus with margin.
  * THE ORIENTATION FOR LOADING IS A COMPUTED ANGLE.  A chord drops into
    its V-notches when its chord is uppermost; a diagonal drops into its
    two cradles when its face is uppermost and level.  Both are cage
    angles the geometry module derives.

numpy only.
"""
from dataclasses import dataclass
from functools import cached_property
from math import pi, sin, cos, tan, radians, atan2, degrees, sqrt

import numpy as np

from .spec import (Cage, Carrier, Magazine, Head, Gripper, Process,
                   notch_mouth)
from .geometry import (TrussGeometry, FACES, radial, chord_phi, rot_x,
                       theta_chord_up, theta_face_up)


@dataclass(frozen=True)
class Pin:
    chord:  int
    x:      float
    apex:   np.ndarray     # the V's floor line point, cage frame, theta 0
    up:     np.ndarray     # unit vector the V opens along (radially out)


@dataclass(frozen=True)
class Cradle:
    joint:  int
    rod:    int
    centre: np.ndarray     # on the diagonal's centreline, cage frame
    axis:   np.ndarray     # along the diagonal
    up:     np.ndarray     # the V opens along this (the face's outward normal)
    apex:   np.ndarray     # V floor point


@dataclass(frozen=True)
class Post:
    chord:  int
    end:    int            # 0 = the x0 end, 1 = the far end
    p0:     np.ndarray     # a radial pin the thread wraps
    p1:     np.ndarray


@dataclass(frozen=True)
class Slot:
    """A magazine slot: where a rod lies before it is loaded."""
    rod:    int
    p0:     np.ndarray     # rod centreline in world (the rack is static)
    p1:     np.ndarray
    up:     np.ndarray


class Fixture:
    # LAZY ON PURPOSE.  A fixture is cheap to name and expensive to lay
    # out -- the pins alone are 24 ms, and the design optimiser wants the
    # magazine's extent for ten thousand candidates.  Deferring each part
    # until it is read keeps ONE implementation of the layout instead of a
    # closed form beside it that can drift; a check pins the two together
    # only when there is one to pin.
    def __init__(self, geom: TrussGeometry):
        self.g = geom
        self.t = geom.t

    @cached_property
    def pins(self):
        return self._pins()

    @cached_property
    def cradles(self):
        return self._cradles()

    @cached_property
    def posts(self):
        return self._posts()

    @cached_property
    def slots(self):
        return self._magazine()

    @cached_property
    def nose_reach(self):
        """How far past the chord ends this truss's loaded camera carrier
        reaches -- what the cage's end freedom has to clear.

        Solved from the section, because the standoff is: 21.4 mm at an
        85 mm triangle, 35.7 at 140.  `end_free` is the CAGE's answer to
        the worst of these across everything the cell builds; this is the
        one truss's, and check_mount holds the first against the second.
        """
        from . import mount
        return Carrier.reach(mount.fov_standoff(self.g, d_strut=self.t.d_diag))

    def end_free(self):
        """Axial room between the chord ends and the end plate, mm.  A
        machine fact: the cage is built once (spec.Cage.END_FREE)."""
        return Cage.END_FREE

    # ------------------------------------------------------------- pins
    def joint_exclusion(self):
        """Axial half-width round a joint nothing fixed may enter: the
        ring's band or its spool, whichever is wider, plus HALF THE WOUND
        BAND -- the ring starts each band a half-band to one side of the
        joint and sweeps across it (measured: the spool met a pin arm
        11 mm from a joint at 0.4 turns) -- plus the arm that carries a
        pin (wider than the pin plate itself), plus clearance."""
        return (Head.ring_axial_half() + self.t.band / 2.0
                + max(Cage.PIN_T, Cage.ARM_W) / 2.0 + Process.SEAT_CLEAR)

    def pin_xs(self, k):
        """Pin stations along chord k.  Every free interval between two
        forbidden zones -- a joint's exclusion, the gripper's footprint at
        the chord's midpoint, the chord's ends -- gets pins at both ends
        and evenly between, never further apart than PIN_PITCH: a chord is
        supported within the exclusion distance of every joint and nowhere
        is a span longer than the brief's 50 mm.  Overlapping zones are
        merged first (the metre truss has a joint on its midpoint)."""
        t = self.t
        excl = self.joint_exclusion()
        gexcl = Gripper.PAD_L / 2.0 + Cage.PIN_T / 2.0 + 2.0
        zones = sorted([(j.x - excl, j.x + excl) for j in self.g.joints_on(k)]
                       + [(t.length / 2.0 - gexcl, t.length / 2.0 + gexcl)])
        merged = []
        for a, b in zones:
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        free = []
        lo = Cage.PIN_T
        for a, b in merged:
            free.append((lo, a))
            lo = b
        free.append((lo, t.length - Cage.PIN_T))
        xs = []
        for a, b in free:
            if b - a < Cage.PIN_T:
                continue
            n = int(np.ceil((b - a) / Cage.PIN_PITCH)) + 1
            xs.extend(np.linspace(a, b, n).tolist())
        return xs

    def _pins(self):
        t = self.t
        out = []
        for k in range(t.n_chords):
            xs = self.pin_xs(k)
            up = radial(chord_phi(k))
            for x in xs:
                axis_pt = self.g.chord_point(k, x)
                # a 90-degree V holds a rod of radius r with its axis r*sqrt2
                # above the apex; general angle: r / sin(half-angle)
                drop = (t.d_chord / 2.0) / sin(radians(Cage.NOTCH_ANGLE / 2.0))
                out.append(Pin(k, float(x), axis_pt - up * drop, up))
        return out

    # ---------------------------------------------------------- cradles
    def cradle_r(self):
        """The cradle block's collision radius round the diagonal."""
        return self.t.d_diag / 2.0 + Cage.CRADLE_T + 1.0

    def cradle_offset(self):
        """Along the diagonal from the joint centre to the cradle's centre:
        the block's nearest point clears the ring's swept envelope at that
        joint by the seating clearance."""
        a = radians(self.t.alpha)
        axial = (Head.ring_axial_half() + self.t.band / 2.0 + Process.SEAT_CLEAR
                 + self.cradle_r() + Cage.CRADLE_L * cos(a) / 2.0)
        return axial / cos(a)

    def face_normal_out(self, f):
        """Unit vector from the truss axis through the middle of face f."""
        a, b = FACES[f]
        v = radial(chord_phi(a)) + radial(chord_phi(b))
        return v / np.linalg.norm(v)

    def _cradles(self):
        """The closed-form offset puts the block outside the ring's axial
        band; on a steep diagonal the block also climbs into the spool's
        radial band, where the two corners meet.  So the offset is then
        PUSHED, half a millimetre at a time, until the block clears the
        seated head by the seating clearance -- measured with the same
        distance the approach solver uses, so the two cannot disagree."""
        from .approach import head_distance
        t = self.t
        off0 = self.cradle_offset()
        drop = (t.d_diag / 2.0) / sin(radians(Cage.CRADLE_ANGLE / 2.0))
        out = []
        for r in self.g.diags:
            u = r.axis
            up = self.face_normal_out(r.face)
            for end, (jix, sign) in enumerate(((r.ends[0], 1.0), (r.ends[1], -1.0))):
                j = self.g.joints[jix]
                axis_pt = self.g.chord_point(j.chord, j.x)
                q = r.p0 if end == 0 else r.p1
                off = off0
                for _ in range(40):
                    p = q + u * (sign * off)
                    a = p - u * (Cage.CRADLE_L / 2.0) - up * 2.0
                    b = p + u * (Cage.CRADLE_L / 2.0) - up * 2.0
                    s = np.linspace(0.0, 1.0, 13)[:, None]
                    P = a[None, :] + s * (b - a)[None, :] - axis_pt[None, :]
                    # the seated head is a body of revolution about the
                    # chord bar its carriage box, which sits above the
                    # chord when the chord is up -- away from a cradle.
                    # It sweeps the band's width, so test both ends.
                    hb = self.t.band / 2.0
                    d = min(float((head_distance(P + np.array([sx, 0.0, 0.0]), True)
                                   - self.cradle_r()).min()) for sx in (-hb, 0.0, hb))
                    if d >= Process.SEAT_CLEAR:
                        break
                    off += 0.5
                out.append(Cradle(jix, r.index, p, u, up, p - up * drop))
        return out

    # ------------------------------------------------------------ posts
    def _posts(self):
        t = self.t
        out = []
        for k in range(t.n_chords):
            up = radial(chord_phi(k))
            for end, x in enumerate((-Cage.POST_OFF, t.length + Cage.POST_OFF)):
                c = self.g.chord_point(k, x)
                out.append(Post(k, end, c - up * 6.0, c + up * 6.0))
        return out

    # --------------------------------------------------------- magazine
    def cage_swept_r(self):
        """The largest radius anything on the cage reaches."""
        return self.g.cage_radius() + Cage.NOTCH_DEPTH + Cage.PIN_T

    def _magazine(self):
        """Chords lie parallel to x beside the cage on the -y side, three
        slots deep.  Diagonals lie ALONG y in a rack beyond the +x end of
        the truss, pitched along x, centred on the truss axis: that is the
        clearance the brief already reserves past the truss (4.3), it
        keeps the loader's y reach to the chord rack, and the gripper's
        yaw turns a rod from +90 to +-alpha.  (A rack of diagonals beside
        the cage would need a 500 mm y axis -- measured, not guessed: the
        rods are 130 mm long and there are twenty-four of them.)"""
        t = self.t
        out = []
        z = self.g.R                        # rack tops level with the cage top
        y0 = -(self.cage_swept_r() + Magazine.CLEAR)
        for r in self.g.chords:
            y = y0 - r.chord * Magazine.CHORD_PITCH
            out.append(Slot(r.index, np.array([0.0, y, z]),
                            np.array([t.length, y, z]), np.array([0, 0, 1.0])))
        # one rack at each end, so a diagonal comes from the nearer one and
        # the mean fetch is a quarter of the truss, not half
        half = t.L_cut / 2.0
        edge = self.end_free() + Cage.END_PLATE_T + Magazine.END_CLEAR
        near = sorted(self.g.diags, key=lambda r: r.mid[0])
        n0 = len(near) // 2
        for i, r in enumerate(near[:n0]):
            x = -edge - i * Magazine.DIAG_PITCH
            out.append(Slot(r.index, np.array([x, -half, z]),
                            np.array([x, half, z]), np.array([0, 0, 1.0])))
        for i, r in enumerate(near[n0:]):
            x = t.length + edge + i * Magazine.DIAG_PITCH
            out.append(Slot(r.index, np.array([x, -half, z]),
                            np.array([x, half, z]), np.array([0, 0, 1.0])))
        return out

    def x_range(self):
        """The ring-centre x the cell must reach, (min, max): the posts,
        and every slot with the gripper's offset from the ring plane."""
        xs = [-Cage.POST_OFF, self.t.length + Cage.POST_OFF]
        for s in self.slots:
            xs.append(float(s.p0[0]) - Head.grip_x())
            xs.append(float(s.p1[0]) - Head.grip_x())
        return min(xs) - Gripper.BODY_W / 2.0, max(xs) + Gripper.BODY_W / 2.0

    def x_reach(self):
        lo, hi = self.x_range()
        return hi - lo

    def slot_of(self, rod_index):
        return next(s for s in self.slots if s.rod == rod_index)

    def y_reach(self):
        """Y travel the loader needs, either side of the truss axis."""
        ys = [abs(float(s.p0[1])) for s in self.slots] + \
             [abs(float(s.p1[1])) for s in self.slots]
        return max(ys) + Gripper.PAD_L

    # ------------------------------------------------------ orientation
    def theta_for_loading(self, rod):
        return theta_chord_up(rod.chord) if rod.kind == "chord" \
            else theta_face_up(rod.face)

    def place_pose(self, rod, theta=None):
        """Where the gripper's grip point goes to seat this rod: the rod's
        midpoint in world at the loading angle, and the yaw of its axis.
        Returns (point [3], yaw_deg, theta)."""
        theta = self.theta_for_loading(rod) if theta is None else theta
        Rm = rot_x(theta)
        p0, p1 = Rm @ rod.p0, Rm @ rod.p1
        if rod.kind == "diag":
            p0, p1 = (Rm @ p for p in self.g.diag_body_ends(rod))
        mid = (p0 + p1) / 2.0
        yaw = degrees(atan2(p1[1] - p0[1], p1[0] - p0[0]))
        return mid, yaw, theta

    def pick_pose(self, rod):
        s = self.slot_of(rod.index)
        mid = (s.p0 + s.p1) / 2.0
        yaw = degrees(atan2(s.p1[1] - s.p0[1], s.p1[0] - s.p0[0]))
        return mid, yaw

    # -------------------------------------------------------- obstacles
    def obstacles(self, theta):
        """Fixture solids as capsules (p0, p1, r) at cage angle theta, for
        the approach solver: pin arms, cradle blocks, posts."""
        Rm = rot_x(theta)
        p0, p1, rr = [], [], []
        for p in self.pins:
            root = np.array([p.x, 0.0, 0.0])
            p0.append(Rm @ root); p1.append(Rm @ p.apex); rr.append(Cage.ARM_W / 2.0)
        for c in self.cradles:
            a = c.centre - c.axis * (Cage.CRADLE_L / 2.0) - c.up * 2.0
            b = c.centre + c.axis * (Cage.CRADLE_L / 2.0) - c.up * 2.0
            p0.append(Rm @ a); p1.append(Rm @ b)
            rr.append(self.cradle_r())
        for q in self.posts:
            p0.append(Rm @ q.p0); p1.append(Rm @ q.p1); rr.append(Cage.POST_R)
        # the spine down the axis
        x0 = -(self.end_free() + Cage.END_PLATE_T)
        x1 = self.t.length + self.end_free() + Cage.END_PLATE_T
        p0.append(np.array([x0, 0.0, 0.0])); p1.append(np.array([x1, 0.0, 0.0]))
        rr.append(self.spine_r())
        # THE END PLATES ARE DISCS, AND A CAPSULE CANNOT BE ONE.  Swept
        # about a 3 mm segment, a capsule of the plate's radius is a BALL
        # 120 mm across, and it refuses every station within a plate radius
        # of the truss's end.  On the 92 mm section the first joint sits
        # far enough out that the ball never bites and the error went
        # unseen; on a 115 mm section it forbids both end joints and the
        # whole truss becomes unplannable.  Spokes: radial capsules as
        # thick as the plate, which is the shape the plate actually has.
        for xa in (x0, self.t.length + self.end_free()):
            mid = xa + Cage.END_PLATE_T / 2.0
            for k in range(Cage.PLATE_SPOKES):
                a = 2.0 * pi * k / Cage.PLATE_SPOKES
                v = np.array([0.0, cos(a), sin(a)]) * self.plate_r()
                p0.append(Rm @ np.array([mid, 0.0, 0.0]))
                p1.append(Rm @ (np.array([mid, 0.0, 0.0]) + v))
                rr.append(Cage.END_PLATE_T / 2.0)
        return np.array(p0), np.array(p1), np.array(rr)

    def spine_r(self):
        return max(Cage.spine_r(self.g.R), Cage.SPINE_R_MIN)

    def plate_r(self):
        """The end plates reach nearly to the chords."""
        return self.g.R - self.t.d_chord - 2.0

    def capture_range(self, rod):
        """How far off a V a rod may arrive and still drop in, mm."""
        return (notch_mouth() - 2.0 * rod.r) / 2.0
