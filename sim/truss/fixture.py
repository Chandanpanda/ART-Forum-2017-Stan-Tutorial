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

from .spec import (Cage, Carrier, Gantry, Magazine, Head, Gripper, Process,
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
    def collapse(self):
        """The collapsing mandrel this fixture rides on.  Lazy: the layout
        is what it needs, and the layout is lazy too."""
        from . import collapse as _c
        return _c.Collapse(self)

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

    def nose_span(self, end=0):
        """(x_in, x_out) the head kit occupies at this end, mm -- or None if
        no camera is being built onto this truss.

        NOTHING OF THE MANDREL MAY BE IN HERE.  The kit is centred on the
        cage's own axis -- that is the whole point of the mount, since a
        mass off the axis turns a manoeuvre into camera yaw -- so there is
        no radius a backbone can neck down to that clears it.  Not 3 mm,
        not 1: the module CONTAINS the axis.  `nose_spine_r` assumed the
        module's nearest face was 4.5 mm off it and stepped the tube down
        to 3.0; measured on the built scene, the tube was 4.5 mm inside the
        board and the camera would not seat because there was something
        already there.
        """
        cams = [r for r in getattr(self.g, "mount_rods", ()) if r.kind == "mcam"]
        if not cams:
            return None
        from .spec import Payload, Carrier
        d = self.t.d_diag
        kx = Carrier.kit_half(Payload, d)[0]
        xs = []
        for r in cams:
            if r.chord != end:
                continue
            xp = 0.5 * (float(r.p0[0]) + float(r.p1[0]))
            # the boss's midpoint is not the module's centre; the module is
            # at the boss's inboard end
            xp = float(r.p0[0]) if end == 0 else float(r.p0[0])
            xs.append(xp)
        if not xs:
            return None
        xc = xs[0]
        return (xc - kx, xc + kx)

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

    def end_exclusion(self):
        """Axial margin at each chord end that no pin may enter, mm.

        THE CHORD ENDS ARE NOT FREE ANY MORE.  The mount's three END
        BATTENS are laid right across them, in the plane the ends define,
        so a pin arm at the end of a chord is a pin arm in a batten's way.
        Measured on the built scene once every mount rod was measured
        rather than only the kit: the last arm stood 0.75 mm from a batten
        where the process wants 1.50, on a margin that was `Cage.PIN_T` --
        a pin's own thickness, which is a fact about the pin and says
        nothing about what is laid over it.

        With no camera on this truss the ends carry nothing and the margin
        falls back to the arm's half-thickness plus clearance, which is the
        3 mm it always was.
        """
        m = getattr(self.g, "mount", None)
        rb = max((r.r for r in m.rods if r.kind == "batten"), default=0.0) \
            if m is not None else 0.0
        return Cage.PIN_T / 2.0 + rb + Process.SEAT_CLEAR

    def free_spans(self, k):
        """The stretches of chord k nothing fixed may cross, merged: between
        one joint's exclusion zone and the next, and clear of the gripper's
        footprint at the midpoint.

        Pulled out of `pin_xs` because the COLLAPSING RAILS need the same
        intervals -- a rail that filled the gaps would sit inside the ring's
        bore at every joint on its chord -- and two copies of this would
        drift apart."""
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
        margin = self.end_exclusion()
        free, lo = [], margin
        for a, b in merged:
            free.append((lo, a))
            lo = b
        free.append((lo, t.length - margin))
        return [(a, b) for a, b in free if b - a >= Cage.PIN_T]

    def pin_xs(self, k):
        """Pin stations along chord k.  Every free interval between two
        forbidden zones -- a joint's exclusion, the gripper's footprint at
        the chord's midpoint, the chord's ends -- gets pins at both ends
        and evenly between, never further apart than PIN_PITCH: a chord is
        supported within the exclusion distance of every joint and nowhere
        is a span longer than the brief's 50 mm.  Overlapping zones are
        merged first (the metre truss has a joint on its midpoint)."""
        xs = []
        for a, b in self.free_spans(k):
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
    @cached_property
    def post_off(self):
        """How far outboard of the chord ends this truss's thread posts
        stand, mm.

        SOLVED, NOT SET.  A post is a radial pin on the chord's own line and
        the mount's struts leave that same line heading inward, so a post
        near the end is in the struts' cone: at the 8 mm the cage was drawn
        with, two struts an end were 0.66 mm inside a pin, and the four
        long ones came out of the mount phase six degrees off their own
        line with every axis reporting success.  `mount.post_station` scans
        the window `Cage.post_off_min/max` bound for the nearest station
        that clears the whole mount.

        With no mount on this truss there is nothing to clear and the post
        goes to the near bound -- which is what the cage was drawn with,
        and is why the number looked right for as long as nobody put a
        camera on it.
        """
        m = getattr(self.g, "mount", None)
        if m is None:
            return Cage.post_off_min()
        from . import mount as _mount
        off = _mount.post_station(self.g, m, Cage.POST_R, Cage.POST_L)
        if off is None:
            raise ValueError(
                "no thread post station clears the mount on this truss: "
                "the window is %.2f..%.2f mm past the chord ends"
                % (Cage.post_off_min(), Cage.post_off_max()))
        return off

    def _posts(self):
        t = self.t
        out = []
        off = self.post_off
        for k in range(t.n_chords):
            up = radial(chord_phi(k))
            for end, x in enumerate((-off, t.length + off)):
                c = self.g.chord_point(k, x)
                out.append(Post(k, end, c - up * Cage.POST_L / 2.0,
                                c + up * Cage.POST_L / 2.0))
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
        # THE MOUNT'S OWN RACK GOES BESIDE THE CAGE, and `_mount_rack` is
        # what places it.  Racked beyond the end racks -- on the diagonals'
        # own pitch, which is where it started -- the chosen truss wanted
        # 1766 mm of a 1400 mm X axis.
        mr = [r for r in self.g.mount_rods]
        if mr:
            out += self._mount_rack(mr, out, z)
        return out

    @staticmethod
    def _x_span(slots, posts_at):
        xs = list(posts_at)
        for s in slots:
            xs += [float(s.p0[0]), float(s.p1[0])]
        return min(xs), max(xs)

    def mount_span(self):
        """(lo, hi) the mount's rack may use in x, mm.

        EXACTLY THE X THE CELL ALREADY HAS TO REACH FOR THE TRUSS, and not
        a millimetre more -- so the mount costs no gantry.  The truss's own
        racks and thread posts set it; whether the mount then fits inside
        it is a question with an answer, which is the point.  Racked beyond
        the end racks instead, on the diagonals' pitch with each kit's nest
        padded by hand, the chosen truss wanted 1766 mm of a 1400 mm axis
        and nothing measured it: check_mount built its X-travel fixture
        with no camera on the truss.
        """
        truss = [s for s in self.slots if s.rod < len(self.g.rods)]
        return self._x_span(truss, (-self.post_off,
                                    self.t.length + self.post_off))

    def _mount_rack(self, mr, truss_slots, z):
        """Every mount part on the cage's flanks, at the free station
        NEAREST THE NOSE IT IS LAID AT.

        WHY BESIDE THE CAGE.  The -y flank carries the chord rack and the
        two ends carry the diagonals; the flanks are otherwise empty for the
        truss's whole length, and a part laid there costs no X travel at
        all.  The parts are stubby enough for it -- 35 to 85 mm against a
        diagonal's 130.  Racked beyond the end racks instead, on the
        diagonals' own pitch, the chosen truss wanted 1766 mm of a 1400 mm
        axis.

        TWO KINDS OF PART, and which kind a part is comes out of the head,
        not out of its name.  A part the gripper can TURN below the head is
        racked along x -- the chord rack's own pattern, and `pick_pose`
        reads the yaw off the slot so the mount phase turns to whatever the
        rack gives it.  A part it CANNOT turn is racked at the yaw
        `mount.lay_pose` says it is laid at, so that it never has to be: the
        head kit's module, collar and grid stand 17.5 mm above the boss the
        jaws hold and want 51 mm of a 40 mm stroke.  Racked square and
        turned anyway it came out 31.3 mm and 75.5 degrees off.

        THE BENCH IS NOT FREE AT THE SAME RADIUS EVERYWHERE.  Over the cage
        a rack has to clear the cage; past its end plates there is nothing
        there but the diagonal rack, 26 mm nearer the axis.  The kit is 61
        mm deep lying along y, and that difference is the whole of whether
        the mount costs Y travel or none -- so a station that costs none is
        taken over a nearer one that would, and only if there is no such
        station anywhere does a part stand further out.

        AND THE STATION IS THE NEAREST ONE TO ITS OWN NOSE.  Every part is
        fetched from the rack and carried to the nose, so what the phase
        costs is twice the distance, per part, and the ORDER it is laid in
        cannot change that -- only where it lies can.  Filled from one end
        of the bench the far nose's kit was racked 1124 mm from the nose it
        goes on, at the wrong end of the cell: the longest move in the whole
        build, and the one op that overran its plan.
        """
        from .spec import Carrier, Payload
        from . import mount as _mount
        lo, hi = self._x_span(truss_slots, (-self.post_off,
                                            self.t.length + self.post_off))
        nose = (0.0, float(self.t.length))
        y_hard = Gantry.Y_TRAVEL / 2.0 - Gripper.PAD_L
        # WHAT THE TRUSS ALREADY MAKES THE LOADER REACH.  A station inside
        # this costs the machine nothing; one outside it is the mount asking
        # for Y travel of its own, and is taken only when nothing else is
        # left.
        y_free = max([abs(float(q.p0[1] + q.p1[1]) / 2.0) for q in truss_slots]
                     + [0.0])
        cage_x = (-(self.end_free() + Cage.END_PLATE_T),
                  self.t.length + self.end_free() + Cage.END_PLATE_T)

        def floor_at(sgn, x0, x1):
            """How near the axis this flank is free, over the x it is asked
            about: the cage's own swept radius where the cage is, and
            whatever the truss's racks reach where they reach."""
            f = 0.0
            if x1 > cage_x[0] and x0 < cage_x[1]:
                f = self.cage_swept_r() + Magazine.CLEAR
            for q in truss_slots:
                qx = sorted((float(q.p0[0]), float(q.p1[0])))
                if qx[1] < x0 or qx[0] > x1:
                    continue
                f = max(f, max(sgn * float(q.p0[1]), sgn * float(q.p1[1]))
                        + Magazine.DIAG_PITCH / 2.0)
            return f

        def over_of(r):
            """(past the slot's p0, half across the slot) -- how far the
            PART reaches beyond the slot the magazine is racking.

            THE SLOT IS THE BOSS AND THE KIT IS NOT: the module, its collar
            and the wound grid hang off the boss's ROOT inside a printed
            pocket, 14 mm past it and 20 across.  Racked as if the boss were
            the part, a kit's grid stood 1.30 mm inside the strut in the
            next slot -- measured on the built scene."""
            if r.kind == "mcam":
                return Carrier.nest_over(Payload, self.t.d_diag)
            return 0.0, Magazine.DIAG_PITCH / 2.0

        def shape(r, sgn):
            """The part's footprint about its slot's centre, on flank `sgn`:
            (back along x, front along x, floor to the slot, slot to the
            part's far edge, the slot's own yaw)."""
            over, wide = over_of(r)
            hl = r.length / 2.0
            if _mount.yaw_stroke_for(r, self.t.d_diag) <= Head.GRIP_STROKE + 1e-9:
                return hl, hl, wide, wide, 0.0   # along x; the head turns it
            yaw = _mount.lay_pose(r)[1]
            ay = sin(radians(yaw))
            if abs(ay) < 1e-6:
                raise ValueError(
                    "mount part %d cannot be turned below the head and is "
                    "laid at %.1f degrees, which is along the bench it would "
                    "have to lie on" % (r.index, yaw))
            ends = [-hl * ay, hl * ay, -(hl + over) * ay]
            return (wide, wide, -min(sgn * e for e in ends),
                    max(sgn * e for e in ends), yaw)

        def gap_to(ha, hb):
            """Between two slots end to end.  THE GRIPPER NEVER REACHES PAST
            THE PART IT TAKES -- it grips at the middle and is BODY_W wide
            along the carriage -- so this is the seating clearance, plus
            whatever a part shorter than the gripper's own body would need
            on top of it."""
            return (2.0 * Process.SEAT_CLEAR
                    + max(0.0, Gripper.BODY_W / 2.0 - ha)
                    + max(0.0, Gripper.BODY_W / 2.0 - hb))

        placed = {1.0: [], -1.0: []}         # (x0, x1, outer, half-length)

        def at(r, sgn, x, cap):
            """Where this part would stand at x on this flank, or None if it
            would reach past `cap`."""
            w0, w1, near, far, yaw = shape(r, sgn)
            hl = r.length / 2.0
            if x - w0 < lo - 1e-9 or x + w1 > hi + 1e-9:
                return None
            f = floor_at(sgn, x - w0, x + w1)
            for a, b, outer, ohl in placed[sgn]:
                # FLUSH IS NOT OVERLAPPING.  With the slack the other way a
                # part laid exactly against its neighbour read as inside it
                # and was pushed a row out -- or, when the row it wanted was
                # at the cap, to the far end of the bench and the wrong
                # nose.  Three of the 300 mm truss's parts went there.
                g = gap_to(hl, ohl)
                if b + g > x - w0 + 1e-9 and a - g < x + w1 - 1e-9:
                    f = max(f, outer + Process.SEAT_CLEAR)
            if f + near > cap + 1e-9:
                return None
            return (f + near, f + near + far, w0, w1, yaw, hl)

        def stations(r, sgn):
            """Where it is worth trying: its own nose, the edges of what is
            already on this flank, and the plate faces -- a part's best x is
            flush against something or at the nose, never between."""
            w0, w1, _n, _f, _y = shape(r, sgn)
            hl = r.length / 2.0
            xs = {nose[r.chord], lo + w0, hi - w1,
                  cage_x[0] - w1, cage_x[0] + w0,
                  cage_x[1] - w1, cage_x[1] + w0}
            for a, b, _o, ohl in placed[sgn]:
                g = gap_to(hl, ohl)
                xs.add(b + g + w0)
                xs.add(a - g - w1)
            return sorted(x for x in xs
                          if lo + w0 - 1e-9 <= x <= hi - w1 + 1e-9)

        out = []
        # DEEPEST FIRST, which is first-fit decreasing and not tidiness: a
        # kit in its nest is 61 mm deep where a strut is 11, and only the
        # bench nearest the cage is ever that clear.  Packed in index order
        # the kits come last and find it taken.
        for r in sorted(mr, key=lambda q: (-sum(shape(q, 1.0)[2:4]),
                                           q.chord, q.index)):
            best = None
            for cap in (y_free, y_hard):
                for sgn in (1.0, -1.0):
                    for x in stations(r, sgn):
                        got = at(r, sgn, x, cap)
                        if got is None:
                            continue
                        d = abs(x - nose[r.chord])
                        if best is None or d < best[0] - 1e-9:
                            best = (d, x, sgn, got)
                if best is not None:
                    break
            if best is None:
                raise ValueError(
                    "the mount's rack has nowhere to stand part %d: it is "
                    "%.1f mm deep against a %.1f mm reach, in x %.1f..%.1f"
                    % (r.index, sum(shape(r, 1.0)[2:4]), y_hard, lo, hi))
            _d, x, sgn, (yc, outer, w0, w1, yaw, hl) = best
            placed[sgn].append((x - w0, x + w1, outer, hl))
            y = sgn * yc
            ax, ay = cos(radians(yaw)) * hl, sin(radians(yaw)) * hl
            out.append(Slot(r.index, np.array([x - ax, y - ay, z]),
                            np.array([x + ax, y + ay, z]),
                            np.array([0, 0, 1.0])))
        return out

    def x_range(self):
        """The ring-centre x the cell must reach, (min, max): the posts,
        and every slot with the gripper's offset from the ring plane."""
        off = self.post_off
        xs = [-off, self.t.length + off]
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
        """Y travel the loader needs, either side of the truss axis.

        AT THE GRIP POINT, NOT AT THE ROD'S FAR END.  Every rod is taken at
        its MIDDLE -- that is the whole reason the racks leave the middle
        free and the pins keep off it -- so a 130 mm diagonal lying across
        the axis costs the Y axis nothing at all, and this used to charge it
        65 mm plus a pad.  On the 300 mm truss that was the number: 116.3 mm
        of Y demanded to reach a point the gripper never goes to.

        A bench is not a travel.  It cost nothing to be wrong about until
        the head kit had to be racked along y -- it cannot be turned below
        the head, so it is racked at the yaw it is laid at -- and a part
        61 mm deep read as 167 mm of a 150 mm axis while its boss, which is
        what the jaws close on, sat at 133.
        """
        ys = [abs(float(s.p0[1] + s.p1[1]) / 2.0) for s in self.slots]
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
        # THE COLLAPSING MANDREL'S OWN RAILS.  They are real solids under
        # every chord and every face, and the head has to miss them --
        # which is why the chord ones are broken on exactly the intervals
        # this class already keeps clear of joints.  Left out of the
        # obstacle set they would be a structure the path planner cannot
        # see, which is how a fixture ends up unbuildable.
        for kind, sh, phi, r, (xa, xb) in self.collapse.rails:
            up = radial(phi)
            p0.append(Rm @ (np.array([xa, 0.0, 0.0]) + up * r))
            p1.append(Rm @ (np.array([xb, 0.0, 0.0]) + up * r))
            rr.append(max(Cage.ARM_W, self.collapse.rail_h()) / 2.0)
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
