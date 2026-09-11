"""The fixture has to come OUT, and as drawn it cannot.

THE PROBLEM, measured before it was designed around.  The truss is a closed
triangular lattice wound onto the cage; the cage is a spine on the same axis
with radial arms reaching out under every chord and cradles under every
diagonal end.  Nothing about that can leave:

    the chosen 85 mm truss     96 pins, 54 cradles, 6 posts
    the biggest cylinder that can slide out along the axis   21.54 mm
    the pins' notch floors reach                             46.95 mm
    the cradles reach                                        36.18 mm

Every locating feature in the cell stands 15 to 25 mm outside the hole it
would have to leave through.  The cage was drawn as a thing to wind onto and
never as a thing to get back, and no amount of clearance fixes it: the
lattice closes around it.

WHAT THIS MODULE IS.  A COLLAPSIBLE MANDREL, which is what filament winding
has always used for a closed part.  Four ideas, each forced by a measurement
rather than chosen:

  RAILS, NOT ARMS.  An arm 39 mm long lying down takes 36 mm of axial room,
  and the metre truss has thirty-two pins per chord: 1.2 m of folded arm for
  1.0 m of spine.  Arms that swing TOGETHER carrying a rail fold once -- the
  rail translates, it does not rotate -- so the axial room needed is one
  arm's, not thirty-two.  A PARALLELOGRAM is what keeps it translating, and
  that matters: a V-notch that rotated as it retreated would rake the rod.

  THE CHORD RAILS ARE SEGMENTED.  They sit under the chords, which is
  exactly where the ring comes down to wind a joint.  So they run only on
  the free intervals `fixture.free_spans` already computes -- between one
  joint's exclusion zone and the next -- and the ring passes through the
  gaps.  Twelve pieces per chord on the metre truss, not one.

  THE FACE RAILS DO NOT NEED TO BE.  The cradles sit on the face planes,
  sixty degrees from any chord, where the head never goes.  Those run whole.

  SIX SHAFTS, SIX LOCKS.  Thirty-nine rail pieces would be thirty-nine
  locks.  Every arm at one azimuth is keyed to a common TORSION SHAFT lying
  on the spine's surface -- at the arms' own hinge line, the one radius the
  head can never reach -- so one shaft turns a whole chord's worth of rail
  pieces together, and one OVER-CENTRE BRACE holds one shaft.

WHY OVER-CENTRE.  The lock has to hold against the seating force and the
winding tension and cannot be undone by either.  A toggle taken past dead
centre onto a stop is driven HARDER into the stop by the load it carries;
the only way out is to push the knee back across, which is a motion in a
direction the load never applies.  `delta` is how far past centre it sits,
and it is bracketed, not picked: below it the stop's pin brinells or the
linkage's own slop swallows the lock, above it a hand cannot break it.

WHY ONE DRAW ROD.  The knees sit against the spine and the lattice is closed
over them -- a finger cannot reach one.  A rod down the spine's bore with a
ramp under each knee trips all six from the end plate, which is outside the
truss.  That is the "manually freed" step, and it is one pull.

EVERYTHING HERE IS COMPUTED FROM THE TRUSS.  The exit radius is the
section's own inscribed circle less the fattest rod; the rails sit where the
pins and cradles already are, so nothing the approach solver or the loader
knows about moves; the arm pitch comes from the rail being straighter than
the chord it holds; the fold angle is bisected until every feature is inside
the exit; and `withdraws` sweeps the collapsed mandrel out along the axis
and reports the least clearance to any rod it passes.
"""
from dataclasses import dataclass
from functools import cached_property
from math import cos, radians, sin, sqrt

import numpy as np

from .spec import Cage, Process, Stock
from .geometry import radial, chord_phi


@dataclass(frozen=True)
class Arm:
    """One leg of a shaft's parallelogram, in the cage frame at theta 0."""
    shaft: int
    x:     float          # axial station of both pivots (the arm is radial)
    base:  np.ndarray     # on the shaft's axis
    tip:   np.ndarray     # on the rail's centreline
    up:    np.ndarray     # radial unit, the way it points when erect

    @property
    def length(self):
        return float(np.linalg.norm(self.tip - self.base))

    def posed(self, psi, fold=-1.0):
        """Where the tip is with the arm swung psi radians toward `fold`."""
        L = self.length
        return self.base + self.up * (L * cos(psi)) + np.array(
            [fold * L * sin(psi), 0.0, 0.0])


@dataclass(frozen=True)
class Brace:
    """The over-centre knee that holds one shaft erect."""
    shaft:  int
    anchor: np.ndarray    # on the spine, axially offset from the arm
    attach: np.ndarray    # on the arm, `c` out from its base
    knee:   np.ndarray    # p from the anchor, q from the attach, delta off line
    p:      float
    q:      float
    delta:  float         # mm past dead centre
    d_lo:   float         # the bracket delta had to sit in
    d_hi:   float

    @property
    def span(self):
        return float(np.linalg.norm(self.attach - self.anchor))

    @property
    def stroke(self):
        """How far the draw rod travels to trip this knee, mm: across the
        over-centre offset on a 45 degree ramp, plus the knee pin's own
        diameter so the ramp is clear of it when seated."""
        return 2.0 * self.delta + Cage.PIVOT_D


class Collapse:
    """The collapsing mandrel for one truss: shafts, rails, arms, braces,
    and the question of whether it comes out."""

    def __init__(self, fixture):
        self.f = fixture
        self.g = fixture.g
        self.t = fixture.t

    # ------------------------------------------------------- the envelope
    def exit_radius(self):
        """The biggest cylinder that can slide out along the axis, mm.

        The section's inscribed circle touches the three FACE planes, and a
        diagonal lies in a face; so what a withdrawing mandrel has to clear
        is that circle less the fattest rod's radius, less a clearance."""
        t = self.t
        inr = t.side / (2.0 * sqrt(3.0))
        return inr - max(t.d_chord, t.d_diag) / 2.0 - Process.SEAT_CLEAR

    def feature_radius(self):
        """How far out the cage's locating features reach when erect, mm --
        what has to come inside `exit_radius`."""
        rs = [float(np.linalg.norm(p.apex[1:])) + Cage.NOTCH_DEPTH
              for p in self.f.pins]
        rs += [float(np.linalg.norm(c.apex[1:])) + self.f.cradle_r()
               for c in self.f.cradles]
        return max(rs)

    # --------------------------------------------------- shafts and rails
    @cached_property
    def shafts(self):
        """(kind, index, azimuth deg) for the six torsion shafts."""
        return tuple([("chord", k, chord_phi(k)) for k in range(self.t.n_chords)]
                     + [("face", k, chord_phi(k) + 60.0)
                        for k in range(self.t.n_chords)])

    def shaft_span(self):
        """(x0, x1) a torsion shaft runs between, mm.

        ONLY AS FAR AS THE RAILS IT TURNS, plus a bearing at each end.  Run
        the length of the cage instead -- end plate to end plate, which is
        what they were -- and they pass straight through both NOSES: four of
        the six cross the camera module, and the one at the face the camera
        looks out of runs down the optical axis 9.5 mm in front of the lens.
        Nothing caught it, because check_collapse asked whether the mandrel
        cleared the TRUSS and check_mount asked whether the mount cleared the
        MODULE, and no check asked whether the cage cleared the camera.

        Found by looking at a frame.
        """
        x0 = min(s[0] for _k, _sh, _p, _r, s in self.rails)
        x1 = max(s[1] for _k, _sh, _p, _r, s in self.rails)
        xs = [a.x for a in self.arms]
        b = Cage.PIVOT_D
        return (min(x0, min(xs)) - b, max(x1, max(xs)) + b)

    def parts(self, end=0):
        """The whole mandrel as `mount.Strut` records, so the clearance and
        field-of-view tests written for the mount can be pointed at it.

        A check that only ever runs on the thing it was written for is how
        the shafts got into the camera."""
        from .mount import Strut
        out = []
        sx0, sx1 = self.shaft_span()
        for sh, (_kind, _k, phi) in enumerate(self.shafts):
            up = radial(phi)
            out.append(Strut("shaft", len(out),
                             np.array([sx0, 0.0, 0.0]) + up * self.shaft_r(),
                             np.array([sx1, 0.0, 0.0]) + up * self.shaft_r(),
                             Cage.PIVOT_D / 2.0, end))
        for _kind, _sh, phi, r, (a, b) in self.rails:
            up = radial(phi)
            out.append(Strut("rail", len(out),
                             np.array([a, 0.0, 0.0]) + up * r,
                             np.array([b, 0.0, 0.0]) + up * r,
                             max(Cage.ARM_W, self.rail_h()) / 2.0, end))
        for a in self.arms:
            out.append(Strut("marm", len(out), a.base, a.tip,
                             Cage.LINK_T / 2.0, end))
        for b in self.braces:
            out.append(Strut("brace", len(out), b.anchor, b.knee,
                             Cage.LINK_T / 2.0, end))
            out.append(Strut("brace", len(out), b.knee, b.attach,
                             Cage.LINK_T / 2.0, end))
        return tuple(out)

    def shaft_r(self):
        """Where a shaft's axis lies, mm from the cage's axis: on the
        spine's surface, one shaft radius out."""
        return self.f.spine_r() + Cage.PIVOT_D / 2.0

    def rail_h(self):
        """Depth of a rail's section, mm: the notch is cut into its top
        face, so the bar has to be deeper than the notch is."""
        return Cage.NOTCH_DEPTH + Cage.CRADLE_T

    def rail_r(self, kind, k):
        """Centreline radius of a chord rail, mm.

        The pins it carries stay exactly where the fixture already puts them
        -- a rail is a way of CARRYING the fixture, not a new fixture -- so
        what this sets is how far below them the bar runs, and that is the
        GRIPPER's number: the pads reach `PAD_UNDER` past a rod's own axis
        to take it, and a rail whose top face is inside that reach is a rail
        the jaws close on."""
        rs = [float(np.linalg.norm(self.g.chord_point(k, p.x)[1:]))
              for p in self.f.pins if p.chord == k]
        from .spec import Gripper
        return (min(rs) - Gripper.PAD_UNDER - Process.SEAT_CLEAR
                - self.rail_h() / 2.0)

    def feature_top(self, k):
        """The furthest anything on chord rail k reaches when erect, mm."""
        return max(float(np.linalg.norm(p.apex[1:])) + Cage.NOTCH_DEPTH
                   for p in self.f.pins if p.chord == k)

    def cradle_top(self):
        return max(float(np.linalg.norm(c.apex[1:])) + self.f.cradle_r()
                   for c in self.f.cradles)

    def segments(self, k):
        """(x0, x1) of every stretch of chord rail there is room for."""
        return tuple(self.f.free_spans(k))

    def arm_pitch(self):
        """How far apart a parallelogram's arms stand, mm.

        NOT the brief's pin pitch, and not a number anybody chose: the rail
        exists to hold a chord straighter than it would lie on its own, so
        the rail's own sag between arms must be under the chord's sag
        between pins.  Both are the same simply-supported formula on the
        same distributed load, so the ratio is (EI_rail / EI_chord)^(1/4)
        times the pin pitch, and the rail -- an aluminium bar deeper than a
        3 mm rod -- can span a good deal further."""
        from .spec import Bracket
        ei_rail = Bracket.E * Cage.ARM_W * self.rail_h() ** 3 / 12.0
        ei_chord = Stock.E * np.pi * self.t.d_chord ** 4 / 64.0
        return Cage.PIN_PITCH * (ei_rail / ei_chord) ** 0.25

    @cached_property
    def rails(self):
        """(kind, shaft, azimuth, radius, (x0, x1)) for every chord rail."""
        out = []
        for sh, (kind, k, phi) in enumerate(self.shafts):
            if kind != "chord":
                continue
            for seg in self.segments(k):
                out.append((kind, sh, phi, self.rail_r(kind, k), seg))
        return tuple(out)

    @cached_property
    def arms(self):
        """Every arm: a chord rail's parallelogram legs, and one leg per
        cradle.

        THE CRADLES GET ARMS OF THEIR OWN, and that is forced too.  A face
        rail would have to run under the diagonals it carries -- and a
        diagonal DIPS to the face plane's own radius between its ends,
        20 mm where its cradles sit at 36.  There is no bar that is both
        under the cradle and clear of the rod: check_load found it as a
        seated diagonal that could no longer be picked out of its own
        cradles.  So each cradle swings on its own leg off the face shaft,
        and the shaft is what makes them one lock instead of fifty-four.
        """
        out = []
        for kind, sh, phi, r, (x0, x1) in self.rails:
            up = radial(phi)
            n = max(2, int(np.ceil((x1 - x0) / self.arm_pitch())) + 1)
            for x in np.linspace(x0, x1, n):
                base = np.array([x, 0.0, 0.0]) + up * self.shaft_r()
                out.append(Arm(sh, float(x), base,
                               np.array([x, 0.0, 0.0]) + up * r, up))
        faces = [(sh, phi) for sh, (kind, k, phi) in enumerate(self.shafts)
                 if kind == "face"]
        for c in self.f.cradles:
            tip = np.asarray(c.apex, float) - np.asarray(c.up, float) * self.stem_len()
            phi_c = np.degrees(np.arctan2(float(tip[2]), float(tip[1]))) % 360.0
            sh, _phi = min(faces, key=lambda f: abs(((f[1] - phi_c) + 180.0)
                                                    % 360.0 - 180.0))
            up = np.array([0.0, float(tip[1]), float(tip[2])])
            up = up / np.linalg.norm(up)
            base = np.array([float(tip[0]), 0.0, 0.0]) + up * self.shaft_r()
            out.append(Arm(sh, float(tip[0]), base, tip, up))
        return tuple(out)

    def stem_len(self):
        """How far below a cradle's V floor its arm stops, mm.

        The arm is a link, not a wire, and the jaws come down onto the rod
        the cradle holds: the pads reach `PAD_UNDER` past the rod's own axis
        and the axis sits `drop` above the V's floor.  An arm that reached
        the floor itself is an arm the gripper closes on -- which is how a
        seated diagonal stopped being pickable."""
        from .spec import Gripper
        drop = (self.t.d_diag / 2.0) / sin(radians(Cage.CRADLE_ANGLE / 2.0))
        return (drop + Gripper.PAD_UNDER + Process.SEAT_CLEAR
                + Cage.LINK_T / 2.0)

    def arms_of(self, sh):
        return tuple(a for a in self.arms if a.shaft == sh)

    # ---------------------------------------------------------- the locks
    def working_load(self):
        """What one shaft's lock has to hold, N.

        Two things, both measured elsewhere: the rails' share of the truss
        they carry, under gravity at whatever angle the cage is turned to,
        and the WINDING TENSION -- the ring pulls `t.tension` on the thread
        and reacts it into the cluster, and the cluster is on a rail.
        Nothing here is a flight load: the fixture never leaves the cell."""
        return self.t.mass / 1000.0 / self.t.n_chords * 9.81 + self.t.tension

    def _slop(self, span):
        """Everything that could let a knee drift back toward dead centre,
        mm: the three pin fits in the brace's chain, and the brace's own
        elastic shortening under the working load."""
        from .spec import Bracket
        return (3.0 * Cage.PIVOT_FIT
                + self.working_load() * span / (Bracket.E * Cage.LINK_T * Cage.LINK_W))

    def delta_bracket(self, span):
        """(floor, ceiling) for the over-centre offset, mm.

        FLOOR, whichever binds: the stop's pin carries F x p / (2 delta) --
        the toggle's own mechanical advantage, which runs away as delta goes
        to zero -- and that has to stay inside the lug's allowable bearing;
        and a lock angle exists to survive the linkage's own looseness, so
        delta has to be an order of magnitude above everything that could
        move the knee back.
        CEILING: breaking the lock costs F x 2 delta / p at the knee, and a
        hand has to be able to do it."""
        p, f = span / 2.0, self.working_load()
        bear = f * p / (2.0 * Cage.PIVOT_SIGMA * Cage.PIVOT_D * Cage.LINK_T)
        return max(bear, 10.0 * self._slop(span)), Cage.HAND_F * p / (2.0 * f)

    def _brace_for(self, sh, c):
        """The brace that meets shaft `sh`'s middle arm `c` out from its base."""
        arms = self.arms_of(sh)
        a = arms[len(arms) // 2]
        # FORTY-FIVE DEGREES BETWEEN BRACE AND ARM.  The brace's moment about
        # the shaft goes as its length times sin x cos of that angle, and
        # sin x cos is largest at 45 -- so the anchor sits as far along the
        # spine from the hinge as the attachment is out along the arm.
        attach = a.base + a.up * c
        anchor = a.base - np.array([c, 0.0, 0.0])
        span = float(np.linalg.norm(attach - anchor))
        lo, hi = self.delta_bracket(span)
        delta = lo
        p = sqrt((span / 2.0) ** 2 + delta ** 2)
        mid = 0.5 * (anchor + attach)
        n = np.cross(attach - anchor, np.cross(a.up, np.array([1.0, 0.0, 0.0])))
        return Brace(sh, anchor, attach, mid + n / np.linalg.norm(n) * delta,
                     p, p, delta, lo, hi)

    def attach_c(self, sh):
        """How far out the brace meets the arm, mm.

        As far as it can: the shaft's rotational stiffness goes as c
        squared, so the brace wants the tip.  What stops it is the FOLD --
        once the toggle is broken the brace is a loose two-bar chain and its
        knee can be anywhere within a link of the anchor, so the bound that
        has to hold is the rigorous one, shaft plus one link.  Bisected on
        exactly that."""
        L = self.arms_of(sh)[0].length
        want = self.exit_radius() - Process.SEAT_CLEAR
        lo, hi = 0.05 * L, L
        if self.shaft_r() + self._brace_for(sh, lo).p > want:
            return lo
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self.shaft_r() + self._brace_for(sh, mid).p <= want:
                lo = mid
            else:
                hi = mid
        return lo

    @cached_property
    def braces(self):
        return tuple(self._brace_for(sh, self.attach_c(sh))
                     for sh in range(len(self.shafts)))

    def knee_radius(self):
        """The furthest a broken knee can reach, mm -- a link from its
        anchor, which is on the spine."""
        return self.shaft_r() + max(b.p for b in self.braces)

    # ------------------------------------------------------------ the fold
    def folded_radius(self, psi):
        """The widest radius anything on the mandrel reaches at swing psi.

        A chord rail translates -- its pins keep their offset from it and
        the whole set moves in together; a cradle swings on its own leg and
        its own block goes with it."""
        worst = self.shaft_r()
        for kind, sh, phi, r, _seg in self.rails:
            L = r - self.shaft_r()
            top = self.feature_top(self.shafts[sh][1])
            worst = max(worst, self.shaft_r() + L * cos(psi) + (top - r))
        for c in self.f.cradles:
            tip = np.asarray(c.apex, float) - np.asarray(c.up, float) * self.stem_len()
            r = float(np.linalg.norm(tip[1:]))
            L = r - self.shaft_r()
            over = (float(np.linalg.norm(c.apex[1:])) + self.f.cradle_r()) - r
            worst = max(worst, self.shaft_r() + L * cos(psi) + over)
        return worst

    def fold_angle(self, margin=None):
        """The least swing that brings every feature inside the exit, rad.

        Bisected on the posed geometry rather than solved in closed form,
        because what has to clear is the CRADLES as well as the pins and
        they sit at a different radius on a different plane."""
        margin = Process.SEAT_CLEAR if margin is None else margin
        want = self.exit_radius() - margin
        lo, hi = 0.0, radians(110.0)
        if self.folded_radius(hi) > want:
            return hi
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if self.folded_radius(mid) <= want:
                hi = mid
            else:
                lo = mid
        return hi

    def posed(self, psi, fold=-1.0):
        """Every arm's tip at swing psi, and the axial shift it took."""
        L = max(a.length for a in self.arms)
        return (tuple(a.posed(psi, fold) for a in self.arms),
                fold * L * sin(psi))

    # ------------------------------------------------- does it come out?
    def withdraws(self, steps=41):
        """Least clearance between the COLLAPSED mandrel and any rod of the
        finished truss, sweeping it out along the axis, mm.

        Negative means it is still trapped.  This is the whole point of the
        module and it is a swept test, not a radius against a radius: the
        cradles sit on the face planes, which is where the diagonals are."""
        psi = self.fold_angle()
        r_out = max(self.folded_radius(psi), self.knee_radius())
        rods = list(self.g.chords) + list(self.g.diags)
        span = self.t.length + 2.0 * (self.f.end_free() + Cage.END_PLATE_T)
        x0 = min(s[0] for _, _, _, _, s in self.rails)
        x1 = max(s[1] for _, _, _, _, s in self.rails)
        worst = float("inf")
        for s in np.linspace(0.0, span, steps):
            lo, hi = x0 + s, x1 + s
            for rod in rods:
                a, b = np.asarray(rod.p0, float), np.asarray(rod.p1, float)
                if max(a[0], b[0]) < lo or min(a[0], b[0]) > hi:
                    continue
                ts = np.linspace(0.0, 1.0, 21)[:, None]
                pts = a + ts * (b - a)
                keep = (pts[:, 0] >= lo) & (pts[:, 0] <= hi)
                if not keep.any():
                    continue
                rr = np.linalg.norm(pts[keep][:, 1:], axis=1).min() - rod.r
                worst = min(worst, float(rr) - r_out)
        return worst

    # ------------------------------------------------------- the draw rod
    def draw_stroke(self):
        """How far the release rod pulls, mm: the worst knee's own trip, so
        one pull frees all six shafts."""
        return max(b.stroke for b in self.braces)

    def draw_bore(self):
        """The bore the spine needs for the rod, mm."""
        return Cage.DRAW_D + 2.0 * Cage.PIVOT_FIT
