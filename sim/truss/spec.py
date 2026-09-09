"""Single source of truth for the truss cell: the task, the machine, the fixture.

All lengths MILLIMETRES, masses GRAMS, angles DEGREES, forces NEWTONS.
Convert with mm()/g() at the point of use -- MuJoCo works in metres and
kilograms.

Three kinds of thing live here, and the distinction is the whole point of
the file (CLAUDE.md, "constants that ARE legitimate"):

  * FACTS about the world -- the stock's modulus, a NEMA17's step, the
    ring's outer diameter, the camera's pixel pitch.  Measured or bought.
  * THE RULES OF THE PRODUCT -- the angular budget, the tip masses, the
    mass ceiling, the prop excitation band.  Given by the brief.
  * A TASK SPEC -- one particular truss.  `Truss` is a dataclass, not a
    module constant, because the machine has to make more than one, and
    the default trusses are the OUTPUT of structure.design(): derived from
    the rules above, never typed in.

What is deliberately NOT here: joint positions, head poses, lift heights,
cage angles, tool paths, cycle times.  Those are answers, and the modules
that compute them (geometry, approach, schedule) take a Truss and a cell
spec as arguments so they answer for the next truss too.
"""
from dataclasses import dataclass
from math import (pi, tan, sin, cos, asin, acos, atan, radians, sqrt, floor,
                  ceil, atan2, degrees)


# ---------------------------------------------------------------- conversions
def mm(x):
    return x / 1000.0


def g(x):
    return x / 1000.0


# ============================================================= THE MATERIAL
class Stock:
    """Pultruded unidirectional carbon rod (brief 2.3, 3.1).

    The brief's own numbers: 230 GPa fibre at ~65 % volume fraction gives
    ~150 GPa axial for the rod; 11.3 g/m for 3 mm, which is 1.60 g/cm^3.
    Other diameters scale by area -- same fibre, same fraction.
    """
    E           = 150e3        # N/mm^2, axial modulus of the rod
    RHO_LIN_3   = 11.3         # g/m, 3 mm rod, measured
    TG          = 170.0        # deg C
    DIAMETERS   = (1.0, 1.5, 2.0, 3.0)      # what the shop keeps
    CUT_TOL     = 0.3          # mm, hand-cut length scatter (1 sigma) [VERIFY]

    @staticmethod
    def rho_lin(d):
        """g/m for a rod of diameter d."""
        return Stock.RHO_LIN_3 * (d / 3.0) ** 2

    @staticmethod
    def area(d):
        return pi * d * d / 4.0

    @staticmethod
    def I_rod(d):
        """Second moment of the rod's own section, mm^4."""
        return pi * d ** 4 / 64.0

    @staticmethod
    def density():
        """g/cm^3, from the measured linear density."""
        return Stock.RHO_LIN_3 / (Stock.area(3.0) * 1e-3)


class Carrier:
    """The camera's kitting carrier, and why there has to be one.

    THE GRIPPER CANNOT HOLD A CAMERA.  Its jaws open 8 mm -- sized for a
    3 mm rod with clearance -- and the module is 11.3 mm through its thin
    way, its metal enclosure 10.8 mm square.  Nothing on it is 8 mm except
    the lens barrel, and that is the one part a machine must not touch.

    AND OPENING THE JAWS IS NOT THE FIX.  The magazine's slot pitch is
    bounded below by JAW_OPEN/2 -- a rod has to clear the pads that
    straddle its neighbour -- and the diagonal racks pitch along x, and the
    gantry's X travel is already 1359 mm of 1400 on the chosen truss.  Jaws
    wide enough for the enclosure put the racks at 1409.  The machine would
    need a longer axis to be able to pick up a camera.  Both measured, in
    check_mount.

    So the module is pressed into a printed carrier at kitting -- the same
    manual step as pre-cutting the rods -- and the carrier presents a BOSS
    of the largest stock diameter, coaxial with the spine.  To the loader
    it is then a chord: the jaws already span it, the V already captures
    it, the rack pitch does not move, and the pose is yaw zero at any cage
    angle.  Nothing about the machine changes.

    IT GRIPS THE BOARD, NOT THE ENCLOSURE.  The six struts bond to the
    enclosure, so nothing of the carrier may be on it.  The carrier takes
    the board's two long edges instead -- the part the mount deliberately
    does not use, and the part the module's own designers put four holes
    in.
    """
    WALL        = 1.5          # mm, printed
    FIT         = 0.10         # mm interference on the board's edges
    MASS        = 2.0          # g [VERIFY: print one and weigh it]
    GRIP_CLEAR  = 4.0          # mm of boss beyond the pads, both ends

    @classmethod
    def boss_d(cls):
        """The boss is a stock rod diameter ON PURPOSE: every clearance,
        capture and pitch rule in the cell has already been checked for
        it."""
        return max(Stock.DIAMETERS)

    @classmethod
    def boss_l(cls):
        """Long enough for the pads to take it with clearance either side."""
        return Gripper.PAD_L + cls.GRIP_CLEAR

    @classmethod
    def span(cls, payload=None):
        """What the jaws would have to open to if there were no carrier --
        the module's own thin way, and its enclosure's."""
        p = Payload if payload is None else payload
        return (p.BOX[1], p.CASE[0])

    @classmethod
    def reach(cls, standoff_mm, payload=None):
        """How far past the chord ends the loaded carrier reaches, mm.

        THE BOSS IS RADIAL, NOT AXIAL, and that is what keeps this number
        down to the camera's own outer face.  Pointed along the spine it
        would add its whole length to the room the cage has to leave, and
        the racks that sit beyond the end plate would follow it.  Pointed
        out of the module's BACK -- 180 degrees from the optical axis, so
        it can never be in shot -- it costs nothing axially, and the cage
        turns it under the jaws exactly as it does a diagonal.
        """
        p = Payload if payload is None else payload
        return standoff_mm + p.BOX[0] / 2.0

    @staticmethod
    def boss_axis(look):
        """The boss's direction: straight out of the camera's back, so it
        is as far from the field of view as a direction can be."""
        import numpy as _np
        v = _np.asarray(look, float)
        return -v / _np.linalg.norm(v)


# ============================================================ THE PRODUCT
class Load:
    """The stereo rig's load case and budgets (brief 2.2, 2.3, 6).

    The spine is centre-mounted, so each half is a cantilever with a camera
    head at its tip.  The budget is ANGULAR: relative yaw between the two
    camera faces, which is bending slope at the tips and nothing a
    software calibration can absorb (brief 2.2).
    """
    # ONE CAMERA HEAD.  This was a flat 50 g, carried from the brief, and
    # it is the most consequential number in the design: it sets the tip
    # force, dominates the Rayleigh mass, and so fixes both the angular
    # budget and the first mode.  The module is 4 g.  Derived from the part
    # now, so it cannot drift away from it again.
    BRIEF_TIP_MASS  = 50.0     # g, the brief's assumption -- kept because
                               # its published 228 Hz is reproducible only
                               # against the head it was computed with
    LATERAL_G       = 3.0      # manoeuvre load, multiples of gravity
    G               = 9.81
    SLOPE_BUDGET    = 0.005    # deg, total angular drift, both faces
    # A DESIGN MARGIN, NOT A TUNING: the brief's own worked truss sits at
    # half the budget (0.0025 of 0.005), and half is what is left for the
    # thermal soak, the joints' creep and the mount.  The optimiser designs
    # to SLOPE_BUDGET / SLOPE_SF.
    SLOPE_SF        = 2.0
    F1_MIN          = 200.0    # Hz, first bending mode with the tip masses
    BLADE_PASS      = (200.0, 330.0)   # Hz, 5-7 in props at 6-10 krpm
    # The spine's own mode may sit in that band -- the brief soft-mounts
    # it for exactly that reason -- but a MEMBER mode must sit above it: a
    # diagonal ringing at blade-pass is a joint being fatigued at 300 Hz,
    # and one below the band is crossed on every spin-up.
    MEMBER_F_MIN    = 330.0
    # The joint recipe (turns, tension, band) is qualified by the bench
    # DoE at one geometry; a diagonal angle outside this range is a joint
    # nobody has pulled to 50 N.  Also the range within which a mitre is a
    # mitre rather than a scarf.
    ALPHA_RANGE     = (35.0, 55.0)
    MASS_MAX        = 70.0     # g, 1 m truss acceptance (brief 6)
    MASS_MAX_300    = 25.0     # g, the 300 mm truss (target 19)
    # The airframe mount.  A deeper section is always stiffer and lighter
    # per joint, so without an envelope the optimiser runs to infinity; the
    # envelope is a fact about the product, not about the truss.  The
    # brief's own design is a 90 mm triangle.
    SECTION_MAX     = 100.0    # mm, largest triangle side the mount takes
    BUCKLE_SF       = 5.0      # chord Euler load over its working load
    TEST_FORCE      = 1.5      # N at the cantilever tip (brief 6)

    @classmethod
    def tip_mass(cls):
        """g, one camera head: the module plus what terminates on it."""
        return Payload.MASS + Payload.HEAD_EXTRA


# ================================================================ THE TASK
@dataclass(frozen=True)
class Truss:
    """One truss: what the machine is asked to make.

    Warren, triangular section, three continuous chords, zigzag diagonals
    on each of the three faces, no verticals (brief 3.1, 4.2).  The joint
    recipe rides with it because a joint is part of the product.

    THE FACES INTERLEAVE.  Each chord carries the diagonals of two faces.
    If both zigzags put a node at the same x the joint would take FOUR
    mitred ends, which the ring cannot wind past and the brief forbids
    ("two diagonals meet at each joint").  So face f's nodes sit on one
    chord at even multiples of the run and on the other at odd, and every
    chord ends up with a joint every `run`, alternating faces.

    `alpha` is the angle of the diagonal's CENTRELINE to the chord, and
    `run` follows from it and the lateral span the centreline actually
    crosses: the chord-to-chord side minus one chord diameter, because the
    diagonal's ends lie against the chords' surfaces, not on their axes.
    """
    length:     float           # mm, chord length
    side:       float           # mm, chord axis to chord axis
    alpha:      float           # deg, diagonal centreline to chord
    d_chord:    float = 3.0
    d_diag:     float = 2.0
    end_margin: float = 25.0    # mm of chord beyond the last joint, each end
    n_chords:   int   = 3
    # the joint recipe (brief 3.2) -- the bench DoE owns the real values
    turns:      int   = 18
    band:       float = 8.0     # mm, wound band width
    thread_d:   float = 0.15    # mm, Kevlar
    thread_rho: float = 1.44    # g/cm^3
    tension:    float = 2.0     # N, winding tension [VERIFY by DoE]
    joint_mass: float = 0.15    # g, thread + resin, the target
    name:       str   = "truss"

    # ---------------------------------------------------------- section
    @property
    def R(self):
        """Circumradius: chord axes sit on this circle."""
        return self.side / sqrt(3.0)

    @property
    def s_eff(self):
        """Lateral span of a diagonal's centreline, surface to surface."""
        return self.side - self.d_chord

    @property
    def run(self):
        """Axial pitch of the joints on any one chord."""
        return self.s_eff / tan(radians(self.alpha))

    @property
    def n_nodes(self):
        """Joints per chord, all faces counted."""
        return int(floor((self.length - 2.0 * self.end_margin) / self.run)) + 1

    @property
    def x0(self):
        """x of the first joint; the grid is centred on the chord."""
        return (self.length - (self.n_nodes - 1) * self.run) / 2.0

    @property
    def n_diag(self):
        return self.n_chords * (self.n_nodes - 1)

    @property
    def n_joints(self):
        return self.n_chords * self.n_nodes

    @property
    def L_diag(self):
        """Centreline length, tangent plane to tangent plane."""
        return sqrt(self.run ** 2 + self.s_eff ** 2)

    @property
    def L_cut(self):
        """Rod length as CUT, tip to tip along the longest generatrix: each
        mitre adds half a diameter over the tangent of the angle."""
        return self.L_diag + self.d_diag / tan(radians(self.alpha))

    @property
    def mitre_face(self):
        """Length of the flat the mitre presents along the chord."""
        return self.d_diag / sin(radians(self.alpha))

    def bind_half(self, thread_d=None):
        """Half-width of the band that actually BINDS, mm.

        A diagonal touches the chord only at the joint; a hoop laid `dx`
        along the chord finds it standing off by dx*tan(alpha) less its own
        half-section across the chord, d_diag/(2 cos alpha), so the two
        members are within a thread of each other only while

            |dx| <= d_diag / (2 sin alpha)  +  d_thread / tan alpha

        and a hoop laid outside that wraps the CHORD ALONE, passing under
        the diagonals.  Measured, wound, in check_ring: hoops at 3 mm from
        a 45-degree joint sat at 1.24 mm radius where the members' convex
        hull reaches 6.4 -- the thread does not bridge, it beds.  So the
        recipe's 8 mm band binds over 1.9 mm of its width and the rest is
        thread on a chord; the resin and the keeper hold the rest.
        """
        a = radians(self.alpha)
        d_t = self.thread_d if thread_d is None else thread_d
        return self.d_diag / (2.0 * sin(a)) + d_t / tan(a)

    # -------------------------------------------------------------- mass
    @property
    def mass_chords(self):
        return self.n_chords * self.length * Stock.rho_lin(self.d_chord) / 1000.0

    @property
    def mass_diags(self):
        return self.n_diag * self.L_cut * Stock.rho_lin(self.d_diag) / 1000.0

    @property
    def mass_joints(self):
        return self.n_joints * self.joint_mass

    @property
    def mass(self):
        return self.mass_chords + self.mass_diags + self.mass_joints

    def joint_x(self):
        """The joint grid along any chord."""
        return [self.x0 + i * self.run for i in range(self.n_nodes)]


# ============================================================ THE MACHINE
@dataclass(frozen=True)
class Mesh:
    """A solved gear mesh: what Ring.mesh() found (all mm, MPa)."""
    m:         float        # module
    n_ring:    int          # teeth on the rim
    n_pinion:  int
    r_pitch:   float        # ring's pitch radius
    r_pinion:  float        # pinion's pitch radius
    centre:    float        # ring centre to pinion centre
    tip_r:     float        # how far the pinion's tip reaches from the ring centre
    sigma:     float        # MPa in the pinion's tooth root at DRIVE_TORQUE

    @property
    def ratio(self):
        """Ring turns per pinion turn."""
        return self.n_pinion / float(self.n_ring)


_RING_MEMO = {}


class Ring:
    """The winding head: a gapped ring in a raceway, driven on its teeth by
    phased pinions.

    WHY IT IS DRIVEN FROM ITS EDGE.  A ring that encircles a rod cannot
    have a shaft through it -- the rod is where the shaft would go -- so
    the drive acts on its rim.  That is an orbital welding head, scaled
    down, and the mechanism is not in doubt.  What was wrong was the
    detail, and three faults were measured in the first version:

      * THE SPOOL AND THE DRIVE WANTED THE SAME SURFACE.  The spool block
        swept 20 to 32 mm of radius on the rim; a 6 mm friction wheel
        pressed on that rim occupies 20 to 32 mm too, and sat axially on
        the 4 mm ring inside the spool's own 14.  The orbiting spool
        struck each drive wheel once a revolution.  Nothing caught it
        because both were lumped into one head solid and a head is not
        checked against itself.
      * NOTHING LOCATED THE RING.  Two friction wheels are a drive, not a
        bearing, and no guide roller was ever specified.
      * THE FRICTION HAD NO MARGIN.  The thread tension alone asks about
        0.022 N.m about the ring's axis against a slip torque of 0.03 that
        was itself a guess, and friction slips silently, which is why the
        turns had to be counted by a fiducial rather than known.

    WHAT REPLACES IT, and what the geometry refused on the way.  Closing
    the gap for winding was tried first: a gate of the gap's own arc,
    slid aside to admit the rod.  The clearance solver refused it, and the
    reason generalises -- parked aside, the gate still spans the ring's
    full radial depth pointing at the work, so on the way down it scrapes
    the chord, and moved radially instead it fouls the pin arm.  There is
    nowhere clear for a gate to park.  So the gap stays open, and what is
    fixed is the drive:

      * TEETH, NOT FRICTION.  The rim is toothed and driven by pinions, so
        the turns are known at the motor rather than inferred, and the
        torque margin is a tooth's, not a coefficient's.
      * THREE PINIONS, PHASED, ON ONE BELT.  Spaced wider than the gap, so
        at least two are always meshed, and rigidly synchronised so a
        pinion re-enters the teeth in phase after the gap has passed it.
        One motor, not three: three independently commanded shafts geared
        to one ring are an over-constrained closed loop, and check_drive
        jammed with all three saturated and the ring turning backwards.
        They are most of the bearing as well as the drive -- the raceway
        cannot locate the ring in the mouth's own direction, so what does
        is the pinions (race_mouth, capture_wander, RUN_OUT).
      * THE GAP IS A WHOLE NUMBER OF TEETH.  Otherwise the far side of the
        gap arrives out of phase and the re-entering pinion butts a tooth
        instead of finding a space.
      * THE MOUTH IS AS NARROW AS THE WORK ALLOWS.  Written wide enough to
        contain the gap, it left a dead arc that let the ring wedge; see
        race_mouth for what that cost and why the premise was false.
      * THE RING IS ITS OWN SPOOL.  The whole metre truss takes 7.8 m of
        0.15 mm thread, which is 139 cubic millimetres, less than a drop.
        It is wound into a GROOVE IN THE RING'S OWN WEB, the way a
        toroidal winder's shuttle carries its wire.  Nothing protrudes,
        the rim is left for the drive, and the head's swept radius falls
        from 32 mm to the rim itself -- which is what let the cage's spine
        grow from 3 mm to 8.
    """
    OD          = 40.0
    ID          = 20.0
    GAP         = 60.0         # deg, open arc: how the rod gets in
    # A THIN PLATE, NOT A DRUM.  The ring sweeps the whole band as the x
    # axis feeds, so its body reaches band/2 + W/2 from the joint, where
    # the diagonals have risen toward the rim: at 10 mm wide no truss in
    # the qualified angle range fits a 20 mm bore (measured, -0.36 mm at
    # the band's end); at 4 mm the metre truss fits with 0.2 mm to spare.
    W           = 4.0          # mm, axial width of the ring body
    # the thread leaves the ring here, on a guide inside the rim
    EXIT_R      = 11.0
    # ---- the thread, wound into the ring's own web
    GROOVE_R    = 15.0         # mm, mean radius of the channel
    GROOVE_D    = 2.0          # mm, radial depth
    GROOVE_W    = 2.0          # mm, axial width
    PACKING     = 0.6          # of the channel, wound thread
    # ---- the drive: teeth on the rim, three phased pinions.
    # NOTHING BELOW IS A TOOTH COUNT.  The first version of this drive wrote
    # one: 72 teeth on the rim and a 12-tooth pinion of 6 mm pitch radius.
    # Those are module 0.556 and module 1.0, and two gears of different
    # module do not mesh -- the drive could not have turned.  Nothing caught
    # it, because a tooth count is exactly the kind of number that is only
    # ever wrong silently.  So the counts are SOLVED (Ring.mesh) from the
    # facts below, and check_geometry re-solves them.
    MODULES     = (0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0)
    PRESSURE_ANGLE = 20.0      # deg, standard involute
    BEARING_OD  = 6.0          # mm, an MR63 -- the smallest the pinion can
    BEARING_W   = 2.5          # mm, its width; two of them straddle the ring
    PINION_WALL = 0.8          # mm of metal between the bearing and the root
    PINION_SIGMA = 60.0        # MPa allowable in the pinion's tooth root
    PINION_SF   = 3.0          # on that, before a module is acceptable
    DRIVE_TORQUE = 0.12        # N.m the pinions can deliver at the ring
                               # [VERIFY: a NEMA 8 through 4:1 -- the number
                               # check_drive measures the demand against]
    ENCODER_CPR = 4000         # counts per turn on the pinion shaft
    MOTOR_INERTIA = 2.0e-7     # kg.m^2, a NEMA 8 rotor
    REDUCTION   = 4.0          # motor to pinion shaft
    STEP_DEG    = 1.8          # a 200-step hybrid stepper, at the motor
    # ---- the raceway the pinions and their idlers are carried in
    RACE_T      = 3.0          # mm of rail either side of the rim
    RACE_CLEAR  = 0.05         # mm, an H7/g6 running fit at 40 mm (0.009
                               # to 0.050)
    # HOW FAR THE RING'S CENTRE ACTUALLY MOVES, measured by check_drive with
    # the pinions in place and re-measured by it every run.  The clearance
    # is not the answer on its own: the mouth and the gap leave an arc where
    # no rail can touch the ring, so it travels until rail half a dead arc
    # away catches it (capture_wander bounds that at 0.30).
    #
    # THIS NUMBER IS WHY THE MOUTH IS SOLVED AND NOT CHOSEN.  With the
    # mouth at 100 degrees the dead arc was 160, the catch was 80 degrees
    # oblique, and a 2 N thread became 7 N on the rail: the drive JAMMED
    # every time the gap crossed a pinion, at any clearance under 0.12, and
    # only ran at all at 0.15 or looser -- where the ring travelled far
    # enough to fetch up against a pinion instead.  A design that needs a
    # LOOSE fit to turn is a design being held by the wrong part.  With the
    # mouth solved (60.5) the dead arc is 120, and it turns at every
    # clearance from 0.05 to 0.20, at RPM_MAX, at twice the tension.
    RUN_OUT     = 0.08         # mm, measured 0.076
    RACE_MU     = 0.15         # rail friction, dry [VERIFY: a real one runs
                               # on a film and is nearer 0.05]
    RPM         = 60.0         # winding speed
    RPM_MAX     = 150.0
    SPINUP_S    = 0.5          # 0 -> RPM
    STOP_TOL    = 3.0          # deg, how well the gap can be parked
    MASS        = 22.0         # g, ring + thread
    # The structure above the ring -- the raceway, the pinions, their motor,
    # the carriage plate -- as a box in the ring's frame, z up from the ring
    # centre: (x half, y half, z0, z1).
    HEAD_BOX    = (22.0, 30.0, 24.0, 90.0)

    @classmethod
    def groove_capacity(cls, thread_d):
        """Metres of thread the ring's own channel holds."""
        arc = 2.0 * pi * cls.GROOVE_R * (360.0 - cls.GAP) / 360.0
        vol = arc * cls.GROOVE_D * cls.GROOVE_W * cls.PACKING
        return vol / (pi * (thread_d / 2.0) ** 2) / 1000.0

    @classmethod
    def _sig(cls):
        """Everything mesh/race_mouth/pinion_az read, as a cache key.

        THESE ARE SOLVERS AND THEY ARE CALLED FROM INNER LOOPS.  pinion_az
        scans 18000 candidate spacings and mesh calls it once per module;
        approach.head_distance calls mesh on every sampled obstacle point,
        which is thousands of times per station.  Uncached, check_approach
        went from 4 minutes to over 50.  Keyed on the inputs rather than
        cached outright so that a rig which changes RACE_CLEAR (or a rim
        that is not this one) re-solves instead of getting a stale answer.
        """
        return (cls.OD, cls.ID, cls.GAP, cls.W, cls.MODULES, cls.PRESSURE_ANGLE,
                cls.BEARING_OD, cls.PINION_WALL, cls.PINION_SIGMA, cls.PINION_SF,
                cls.DRIVE_TORQUE, cls.RACE_CLEAR, cls.RACE_T)

    @classmethod
    def mesh(cls):
        """Solve the rim/pinion mesh.  Returns a Mesh, or raises.

        The ring's TIP circle is its rim -- the rim is the head's swept
        radius and nothing may stand proud of it -- so the pitch radius is
        OD/2 minus one addendum, and the module then fixes the tooth count.
        A module is admissible when

          * the rim takes a whole number of teeth, AND the gap spans a whole
            number of them, so the far side of the gap arrives in phase and
            a re-entering pinion finds a space rather than a tooth;
          * the smallest pinion that avoids undercut (2/sin^2(alpha) teeth)
            also clears its own bearing;
          * THE DRIVE DOES NOT COST ENVELOPE.  A pinion hangs below the
            ring's centre by its azimuth, and the head has to get down over
            a chord: the rule is that the RIM decides how far the head
            reaches down, not the drive.  This is what rules out the coarse
            modules -- 1.25 puts a pinion 21.4 mm down against the rim's 20.

        Of the admissible ones the drive takes the COARSEST, because that
        is the strongest tooth and nothing else discriminates: the pinion's
        tip radius varies by 0.4 mm over the whole admissible set and the
        rim sets the envelope regardless, so spending margin on a finer
        tooth buys nothing.  (Measured over the standard modules: 0.2
        admissible at 18.3 MPa, 0.5 at 9.9, 0.8 at 6.2; 1.25 and 2.0 are
        stronger still and refused on envelope.)
        """
        key = ("mesh",) + cls._sig()
        if key in _RING_MEMO:
            return _RING_MEMO[key]
        best = None
        n_undercut = int(ceil(2.0 / sin(radians(cls.PRESSURE_ANGLE)) ** 2))
        for m in cls.MODULES:
            r_pitch = cls.OD / 2.0 - m           # tip circle IS the rim
            n_ring = 2.0 * r_pitch / m
            if abs(n_ring - round(n_ring)) > 1e-9:
                continue
            n_ring = int(round(n_ring))
            gap_teeth = n_ring * cls.GAP / 360.0
            if abs(gap_teeth - round(gap_teeth)) > 1e-9:
                continue
            # the pinion: big enough not to undercut, and big enough that
            # its tooth root stands outside its own bearing
            n_p = n_undercut
            while n_p * m / 2.0 - 1.25 * m < cls.BEARING_OD / 2.0 + cls.PINION_WALL:
                n_p += 1
            r_p = n_p * m / 2.0
            # Lewis at the root: tangential load from the drive torque over
            # the ring's pitch radius, on the rim's own face width.  Y is
            # the 20-degree full-depth fit 0.484 - 2.87/N.
            F = cls.DRIVE_TORQUE / (r_pitch * 1e-3)
            Y = 0.484 - 2.87 / n_p
            sigma = F / (cls.W * 1e-3 * m * 1e-3 * Y) / 1e6
            if sigma * cls.PINION_SF > cls.PINION_SIGMA:
                continue
            M = Mesh(m=m, n_ring=n_ring, n_pinion=n_p, r_pitch=r_pitch,
                     r_pinion=r_p, centre=r_pitch + r_p, tip_r=r_pitch + r_p + m,
                     sigma=sigma)
            if cls.pinion_depth(M) > cls.OD / 2.0 + 1e-9:
                continue                          # the drive would cost envelope
            if best is None or m > best.m:
                best = M
        if best is None:
            raise ValueError("no standard module meshes a %.1f mm rim with a "
                             "%.0f degree gap" % (cls.OD, cls.GAP))
        _RING_MEMO[key] = best
        return best

    @classmethod
    def pinion_depth(cls, mesh=None):
        """How far the deepest pinion and its bearing block hang below the
        ring's centre, with the mouth aimed straight down."""
        M = cls.mesh() if mesh is None else mesh
        r_block = max(M.r_pinion + M.m, cls.BEARING_OD / 2.0)
        return max(M.centre * cos(radians(az)) + r_block
                   for az in cls.pinion_az(M))

    @classmethod
    def teeth_in_gap(cls):
        return cls.mesh().n_ring * cls.GAP / 360.0

    @classmethod
    def contact_half(cls, mesh=None):
        """Degrees of RING arc one pinion can have its teeth in: the arc its
        tip circle subtends at the ring's centre."""
        M = cls.mesh() if mesh is None else mesh
        return degrees(asin(min(1.0, (M.r_pinion + M.m) / M.centre)))

    @classmethod
    def pinion_az(cls, mesh=None):
        """Where the three pinions go, in degrees from the mouth's centre.

        Two constraints, both in degrees, so they trade against each other
        directly:

          * every pinion's contact arc stays off the mouth -- there is no
            raceway there to carry it, and the work comes in through it;
          * adjacent pinions are further apart than a gap plus two contact
            arcs, or one gap can unmesh two of them at once and the ring is
            left held by a single roller.

        The problem is symmetric about the mouth, so the family is one
        pinion opposite the mouth and two at +-s, and the solver takes the s
        that maximises the SMALLER of the two margins.  (The hand-picked
        110/205/300 had 48 degrees of mouth margin and 11 of spacing: it
        spent margin where it was already rich.)
        """
        ch = cls.contact_half(mesh)
        key = ("az", cls.race_mouth(), cls.GAP, ch)
        if key in _RING_MEMO:
            return _RING_MEMO[key]
        floor_ = cls.race_mouth() / 2.0 + ch
        need = cls.GAP + 2.0 * ch
        best, best_s = -1e9, 120.0
        for i in range(1, 18000):
            s = i * 0.01
            mouth = (180.0 - s) - floor_
            spacing = min(s, 360.0 - 2.0 * s) - need
            v = min(mouth, spacing)
            if v > best:
                best, best_s = v, s
        out = (180.0 - best_s, 180.0, 180.0 + best_s)
        _RING_MEMO[key] = out
        return out

    @classmethod
    def pinion_margin(cls, mesh=None):
        """The margin, in degrees, that pinion_az achieved."""
        az = cls.pinion_az(mesh)
        ch = cls.contact_half(mesh)
        floor_ = cls.race_mouth() / 2.0 + ch
        need = cls.GAP + 2.0 * ch
        mouth = min(abs(((a + 180.0) % 360.0) - 180.0) for a in az) - floor_
        sp = sorted(az)
        spacing = min(sp[1] - sp[0], sp[2] - sp[1], 360.0 - (sp[2] - sp[0])) - need
        return min(mouth, spacing)

    @classmethod
    def race_mouth(cls):
        """The opening in the raceway, degrees.  Solved, not chosen.

        WHAT THE MOUTH IS ACTUALLY FOR.  It was written 100 degrees so the
        ring's 60-degree gap would fit inside it, on the reasoning that the
        work comes in through both.  That is not the requirement.  Only the
        CHORD ever reaches the raceway's radius, and only at one azimuth:
        the diagonals at the ring's own x-window are inside 5 mm of the
        chord's axis (r_in_needed) and never see the rail at 23.  A 3 mm
        chord subtends 8.6 degrees at the rail.  So the mouth does not have
        to contain the gap, and the 100 degrees was 40 degrees of dead arc
        bought for nothing -- dead arc being what lets the ring wander and
        wedge (capture_wander, wedge_torque).

        What does pull the other way is depth: the rail's lowest point sits
        at rail_r_out * cos(mouth/2), and a narrow mouth hangs it below the
        rim, which is what the head has to get down past.  So the mouth is
        the NARROWEST that still costs no envelope -- the rim decides how
        far the head reaches down, the drive does not -- and that is
        2 acos(rim / rail).
        """
        return 2.0 * degrees(acos(min(1.0, (cls.OD / 2.0)
                                      / (cls.OD / 2.0 + cls.RACE_CLEAR + cls.RACE_T))))

    @classmethod
    def chord_at_rail(cls, d_chord, clear):
        """Degrees of mouth a chord of this diameter needs at the rail."""
        return 2.0 * degrees(asin(min(1.0, (d_chord / 2.0 + clear) / (cls.OD / 2.0))))

    @classmethod
    def wedge_torque(cls, force):
        """Rail friction when a side load pushes the ring into the dead arc,
        as a torque about the ring's axis, N.m.

        The catch is oblique: the ring is held by rail half a dead arc away,
        so a load F is carried by normals F / (2 cos phi) each and the
        friction they make is mu F r / cos phi.  This is the term that took
        the drive's whole rating at a 0.05 mm clearance and a 160-degree
        dead arc, and it is why the mouth is no wider than it has to be.
        """
        phi = radians(min(89.0, cls.capture_arc() / 2.0))
        return cls.RACE_MU * force * (cls.OD / 2.0 / 1000.0) / cos(phi)

    @classmethod
    def inertia(cls):
        """The ring's polar moment about its own axis, kg.m^2 -- a thin
        annulus of MASS between the bore and the rim."""
        r1, r2 = cls.ID / 2000.0, cls.OD / 2000.0
        return cls.MASS / 1000.0 * (r1 * r1 + r2 * r2) / 2.0

    @classmethod
    def drive_armature(cls):
        """The motor's rotor, reflected through the reduction and the mesh
        to the ring's own axis, kg.m^2.

        REAL HARDWARE, AND THE CELL NEEDS IT.  It is six times the ring's
        own inertia, so leaving it out does not just lose a little fidelity
        -- it makes the ring's velocity loop unintegrable.  The cell's
        hinge carried armature 1e-7, a placeholder, and that was survivable
        only while the spool's mass sat out on the rim; with the spool gone
        the ring got light, kv*dt/I went to 2.6, and the servo chattered
        between 15 and 0 rad/s every step.  check_hal saw it as a park that
        never landed.
        """
        return cls.REDUCTION ** 2 * cls.MOTOR_INERTIA / cls.mesh().ratio ** 2

    @classmethod
    def servo_kp(cls):
        """The ring's position loop, N.m per radian AT THE RING.

        A STEPPER IS A POSITION SOURCE, not a velocity one, and that is
        what makes it stiff: its torque runs up to the rating over about
        one full step of lag, so the stiffness is the rating over a step
        and there is nothing to choose.  Referred from the pinion shaft to
        the ring, torque divides by the mesh ratio and angle multiplies by
        it, so the stiffness divides by its square.

        The cell drove the ring as a VELOCITY instead, and any gain soft
        enough for the timestep to integrate was too soft to reject the
        hinge's own damping: the ring ran 340 deg/s of a commanded 360 and
        check_hal read it as a 5.6 per cent speed error.  A position loop
        has no steady error to reject.
        """
        return cls.shaft_kp() / cls.mesh().ratio ** 2

    @classmethod
    def shaft_kp(cls):
        """The same stiffness at the PINION SHAFT, where the motor is:
        N.m per radian.  The shaft carries DRIVE_TORQUE * ratio (power is
        conserved, the ring turns `ratio` of a shaft turn) and develops it
        over one full step of the motor through the reduction."""
        return (cls.DRIVE_TORQUE * cls.mesh().ratio
                / radians(cls.STEP_DEG / cls.REDUCTION))

    @classmethod
    def servo_kv(cls):
        """...critically damped on what that loop actually carries: the
        ring plus the drive's rotor reflected to the ring's axis."""
        return 2.0 * (cls.servo_kp() * (cls.inertia() + cls.drive_armature())) ** 0.5

    @classmethod
    def servo_acc(cls):
        """deg/s^2 the ring's command ramps at: 0 to RPM in SPINUP_S.  An
        acceleration, not a time to whatever speed is asked for."""
        return cls.RPM * 6.0 / cls.SPINUP_S

    @classmethod
    def capture_arc(cls):
        """The widest arc over which the raceway CANNOT touch the ring.

        A contact needs ring at that azimuth and rail at that azimuth.  The
        ring is missing over its gap, which turns; the rail is missing over
        the mouth, which does not.  Worst case they lie side by side, and
        the dead arc is their sum.
        """
        return cls.GAP + cls.race_mouth()

    @classmethod
    def capture_wander(cls):
        """How far the ring's CENTRE can move in the raceway, mm.

        A circle in a channel one clearance larger stops when its rim
        reaches the wall in the direction it moved -- at the clearance, if
        there is wall there.  Pushed into the dead arc there is not, and it
        runs on until the nearest live azimuth catches it, which takes
        clearance / cos(half the dead arc).  Beyond a dead arc of 180
        degrees nothing catches it and the ring leaves the machine, so this
        is the capture criterion as well as the number.

        Measured against the rig at 0.35 mm (check_drive): the pinions,
        which this ignores, are most of what holds the ring.
        """
        half = radians(cls.capture_arc() / 2.0)
        if cls.capture_arc() >= 180.0:
            return float("inf")
        return cls.RACE_CLEAR / cos(half)

    @classmethod
    def pinions_meshed(cls, gap_az=0.0):
        """How many pinions have teeth under them with the gap here."""
        ch = cls.contact_half()
        return sum(1 for az in cls.pinion_az()
                   if abs(((az - gap_az) + 180.0) % 360.0 - 180.0) > cls.GAP / 2.0 + ch)


class Gantry:
    """XYZ on SFU1204 ballscrews, MGN12 rails, NEMA17 direct (brief 4.3)."""
    # AS BUILT HERE, NOT AS THE BRIEF HAS THEM.  The brief's 1200 x +-50 was
    # laid out for a head that only winds; a head that also LOADS needs to
    # reach a chord rack beside the cage (y) and a diagonal rack beyond
    # each end plate (x).  fixture.x_range/y_reach compute the need and
    # check_geometry holds these against it: the metre truss needs 1330
    # of x and 134 of y.  1400 is the brief's own "400 mm beyond the
    # longest truss".
    X_TRAVEL    = 1400.0
    Y_TRAVEL    = 300.0
    Z_TRAVEL    = 150.0
    PITCH       = 4.0          # mm per screw revolution
    STEPS_PER_REV = 200        # full steps; no microstepping (brief 4.3)
    MM_PER_STEP = PITCH / STEPS_PER_REV
    V_MAX       = {"x": 60.0, "y": 40.0, "z": 40.0}     # mm/s, loaded
    A_MAX       = {"x": 400.0, "y": 300.0, "z": 300.0}  # mm/s^2
    REPEAT      = 0.05         # mm, bidirectional
    SCREW_CTE   = 12e-6        # /K, steel
    DT_K        = 10.0         # K, warm-up over a shift
    SETTLE_S    = 0.15         # after a move, before a process step
    STALL_N     = {"x": 120.0, "y": 120.0, "z": 120.0}  # thrust at stall


class Head:
    """What rides on the Z carriage, and where.  Tool positions are along
    the carriage's x (the truss axis) from the ring's plane; the dispenser
    and gripper each have their own short stroke so they can reach BELOW a
    ring that is lifted clear -- a single Z cannot serve a ring that must
    be seated on the chord and a nozzle that must be 20 mm under it.

    THE OFFSETS ARE DERIVED: a tool must sit outside the ring's swept
    torus in x, with room for its own width.  Typed here they would move
    the day the spool grew.
    """
    DISP_STROKE = 40.0
    GRIP_STROKE = 40.0
    GRIP_YAW    = 95.0         # deg, either way: a rack rod lies at 90
                               # (measured: a 75-degree range clipped the
                               # pick yaw and the pads landed on the rod)
    TOOL_CLEAR  = 4.0          # mm between a tool and the ring's envelope
    # parked tool tips sit this far ABOVE the ring centre: clear of a chord
    # the ring is seated on, and of the diagonals beside it
    TIP_PARK    = 6.0

    @staticmethod
    def ring_axial_half():
        """Half the axial width of everything on the head at the ring's
        radius: the rim and the rails that carry it."""
        return Ring.W / 2.0 + Ring.RACE_T

    @staticmethod
    def disp_x():
        # the camera bay sits between the ring's envelope and the dispenser
        return (Head.ring_axial_half() + Head.TOOL_CLEAR + Vision.cam_bay()
                + Dispenser.BODY_W / 2.0)

    @staticmethod
    def grip_x():
        return -(Head.ring_axial_half() + Head.TOOL_CLEAR + Gripper.BODY_W / 2.0)


class Gripper:
    """Parallel-jaw, flat pads, on a yaw servo (brief 7.4 made concrete).

    Flat pads rather than V-jaws: the rod is LOCATED by the fixture's
    notches and cradles, never by the gripper, so the gripper only has to
    carry it there.  The yaw servo is what lets one magazine hold every
    diagonal parallel to x and still place them at +-alpha.
    """
    BODY_W      = 12.0         # along the carriage x
    PAD_L       = 10.0         # along the rod
    PAD_T       = 2.0
    PAD_H       = 6.0          # tall, gripping the rod's upper half
    # A rod is gripped at its MIDPOINT, where the racks and the fixture
    # leave it free; anywhere a V sits under the rod, a pad reaching below
    # the axis lands on the flank first (measured: jaws that could not
    # close on a racked diagonal).  fixture.pin_xs keeps the pins off it.
    PAD_UNDER   = 1.0          # how far the pads reach below the grip point:
                               # round the axis of the thinnest rod, since
                               # nothing is under a rod where it is gripped
    FINGER_L    = 22.0
    JAW_OPEN    = 8.0          # gap between pads; rods are 3 mm at most
    JAW_CLOSE_S = 0.3
    YAW_V       = 200.0        # deg/s
    YAW_TOL     = 0.5          # deg, a hobby servo's repeatability


class Cage:
    """The rotating fixture (brief 4.4): a spine on the truss axis, radial
    arms carrying V-notched pins under each chord, cradles at each joint,
    thread posts on the end plates, driven through a worm.

    GEOMETRY IS HELD BY THE FIXTURE, NOT THE ROBOT (brief 4.1).  Every
    location feature is a V, because a V is a kinematic locator that turns
    a millimetre of arrival error into a tenth of nothing -- and check_load
    measures exactly how much.
    """
    THETA_RPM   = 10.0         # worm output
    THETA_TOL   = 0.05         # deg, worm index repeatability
    PIN_PITCH   = 50.0         # brief 4.4: chord straightness governs
    PLATE_SPOKES = 12          # how many radial capsules stand for an end
                               # plate in the approach solver's obstacle set
    PIN_T       = 3.0          # along x
    ARM_W       = 6.0
    NOTCH_ANGLE = 90.0         # included angle of the V
    NOTCH_DEPTH = 2.5          # from the pin's tip to the V's floor
    END_PLATE_T = 6.0
    # the plate stands clear of a ring parked at the post: post offset plus
    # the ring's axial half-extent plus clearance, asserted in CHECKS
    # PLATE FACE BEYOND THE CHORD ENDS, AND THE CAMERA IS WHAT SETS IT.
    # This was 20 mm, which is what the winding head parked at a thread
    # post needs (POST_OFF + ring_axial_half + SEAT_CLEAR = 14.5).  But the
    # camera mount goes on INSIDE the cage -- its rods are laid by the same
    # gripper and bonded by the same dispenser, and the cage is what
    # presents their angles -- so the module has to fit in there too, and
    # at 20 mm it lands in the end plate: the solved standoff puts a Camera
    # Module 3's outer face 33.9 mm past the chord ends of the chosen truss
    # and 38.0 mm past the largest section the cell is specified to build.
    #
    # The cage is ONE machine built once, so this is a machine fact rather
    # than something solved per truss -- but it is a DERIVED one, and
    # check_mount re-derives it against the payload, the field of view and
    # SECTION_MAX, so it cannot drift away from the camera it was sized
    # for.  It is not free: it pushes the end racks out with it, and the
    # chosen truss then wants 1399 mm of the gantry's 1400.
    END_FREE    = 40.0
    POST_R      = 1.5          # thread anchor post
    POST_OFF    = 8.0          # post from the chord end, axially
    CRADLE_L    = 6.0          # along the diagonal
    CRADLE_T    = 1.5          # wall
    CRADLE_ANGLE = 90.0
    # THE SPINE IS SIZED BY THE HEAD'S REACH BELOW THE CHORD.  The ring
    # sweeps its own rim round the chord and the raceway's rail reaches
    # deeper still at the mouth's edge; the chord is R from the cage's
    # axis, so whatever tube runs down that axis has to fit in what is
    # left (measured with the old spool on the rim: an 8 mm
    # spine met the spool at the first joint of the 300 mm truss).  So the
    # spine is as thick as the truss allows up to SPINE_R_MAX, and a truss
    # that leaves less than SPINE_R_MIN cannot be wound by this ring.
    SPINE_R_MAX = 8.0
    SPINE_R_MIN = 3.0

    @staticmethod
    def spine_r(R):
        return min(Cage.SPINE_R_MAX, R - ring_swept_r() - 2.0 * Process.SEAT_CLEAR)
    # Retention.  A rod in an upward-opening V falls out when the cage
    # turns it downward; the real fixture needs a keeper (spring clip, wax
    # dab).  Modelled as a weld once a rod is seated; check_load reports
    # the force a keeper has to hold, which is the rod's weight and nothing
    # more.
    KEEPER      = "weld"
    ARM_PITCH_MIN = 40.0


class Magazine:
    """Where the loader finds the rods.  Chords lie beside the cage,
    parallel to x; diagonals lie in a rack beyond the +x end, also parallel
    to x, and the gripper's yaw turns them.  Positions are derived from the
    truss in fixture.magazine()."""
    CHORD_PITCH = 12.0
    # wider than the jaw's outer span PLUS the neighbour's flank: a yawed
    # gripper straddles the rod it picks, and its pads' outer faces must
    # clear the next slot's V (measured: at 9 mm a pad landed on it and the
    # jaws closed on nothing)
    DIAG_PITCH  = 11.0
    SLOT_ANGLE  = 90.0
    SLOT_DEPTH  = 2.0
    BLOCK_L     = 12.0         # each of the two V-blocks a rod rests on
    BLOCK_IN    = 15.0         # from the rod's end to the block's centre
    CLEAR       = 40.0         # chord rack from the cage's swept radius
    END_CLEAR   = 10.0         # diagonal racks from the cage's end plates


class Dispenser:
    """Syringe on its own short Z, stepper plunger (brief 4.6).  One
    metered drop per joint; the dose is the joint mass budget minus the
    thread, which band.py measures."""
    BODY_W      = 14.0
    NOZZLE_D    = 0.6
    STANDOFF    = 1.0          # nozzle tip above the band when dosing
    DOSE_S      = 1.5
    RESIN_RHO   = 1.15         # g/cm^3, LY556/HY917
    DROP_SPREAD = 0.4          # mm, where a drop lands vs the nozzle [measured by check_effectors]


class Cutter:
    """Hot-wire thread cutter on the head, beside the ring's exit guide."""
    CUT_S       = 1.0
    WIRE_OFF    = 4.0          # from the exit guide, along the strand


class CameraModule:
    """What every camera module in this project has in common.

    THE FRAME.  x runs along the spine (the stereo baseline), y is the
    VIEWING direction, z is up.  So `BOX` is (along the baseline, thickness
    toward the scene, height) and the lens looks along +y.  The housing
    straddles the spine's axis in y and z, so its centre of mass sits ON the
    axis and its optical axis crosses it -- which is not tidiness: an
    off-axis mass turns a manoeuvre into camera yaw, the one error nothing
    downstream recovers.

    NOTHING HERE IS A NUMBER.  The numbers are in the subclasses, one per
    part, each read off that part's own drawing; this is the arithmetic
    that turns them into the units the design budget is written in, so that
    two modules can be compared rather than argued about.
    """

    @classmethod
    def holes(cls):
        """The four mounting-hole centres, (along the baseline, up) from the
        BOARD's centre, mm."""
        hx, hz = cls.HOLE_PITCH[0] / 2.0, cls.HOLE_PITCH[1] / 2.0
        return tuple((sx * hx, cls.HOLES_UP + sz * hz)
                     for sx in (-1.0, 1.0) for sz in (-1.0, 1.0))

    @classmethod
    def f_px(cls):
        """The calibrated focal length, in pixels."""
        return cls.FOCAL / cls.PIXEL

    @classmethod
    def fov_from_sensor(cls):
        """The field the MEASURED image area and the focal length imply,
        degrees (h, v) -- the cross-check on the published field."""
        return tuple(2.0 * degrees(atan(a / 2.0 / cls.FOCAL))
                     for a in cls.IMAGE_AREA)

    @classmethod
    def fov_diagonal(cls):
        """The diagonal field the published H and V imply, degrees."""
        h, v = (radians(a / 2.0) for a in cls.FOV)
        return 2.0 * degrees(atan(sqrt(tan(h) ** 2 + tan(v) ** 2)))

    @classmethod
    def pixel_angle(cls):
        """radians one pixel subtends -- the thing that actually sets depth
        precision, and the one number a bigger sensor does not buy you."""
        return cls.PIXEL / cls.FOCAL

    @classmethod
    def range_error_match(cls, range_m, baseline_mm=1000.0, sigma_px=0.1):
        """m of range error from matching to sigma_px pixels: the STOCHASTIC
        floor.  dZ = Z^2 dtheta / B with dtheta one pixel's angle."""
        return (range_m ** 2 * sigma_px * cls.pixel_angle()
                / (baseline_mm / 1000.0))

    @classmethod
    def range_error_axial(cls, dv_mm, range_m):
        """m of range error from an AXIAL lens shift of dv_mm.

        The lens moving along its own axis changes the image distance,
        which is what a calibration measures as the focal length.  Stereo
        reads Z = f B / d, and a fractional error in f is the same
        fractional error in Z: it does NOT cancel between the two cameras,
        because both focal lengths enter the same way.  An INTRINSIC, and
        the truss cannot help with it.
        """
        return range_m * dv_mm / cls.FOCAL

    @classmethod
    def range_error_yaw(cls, yaw_deg, range_m, baseline_mm=1000.0):
        """m of range error from a relative YAW between the camera faces --
        the truss's own budget, in the same units, so the two compare.
        dZ = Z^2 dtheta / B, and the focal length cancels."""
        return range_m ** 2 * radians(yaw_deg) / (baseline_mm / 1000.0)

    @classmethod
    def case_r(cls):
        """Circumradius of the lens housing, mm -- the largest lever arm a
        mount bonded to it can have."""
        return 0.5 * sqrt(cls.CASE[0] ** 2 + cls.CASE[1] ** 2)

    @classmethod
    def hole_r(cls):
        """...and the lever arm a mount through the four PCB holes gets,
        which is twice as big and goes through FR4 to get there."""
        return 0.5 * sqrt(cls.HOLE_PITCH[0] ** 2 + cls.HOLE_PITCH[1] ** 2)

    @classmethod
    def pcb_stiffness(cls, span_mm):
        """N/mm of the board itself over a span, for comparing with a strut.
        The board is what a hole-mounted strut has to work through, and it
        is not obviously the stiff part."""
        I = cls.BOX[0] * cls.PCB_T ** 3 / 12.0
        return 48.0 * cls.PCB_E * I / span_mm ** 3

    @classmethod
    def half_diagonal(cls):
        """Half the diagonal of the housing's section across the spine --
        the circle the mount's platform triangle has to clear."""
        return 0.5 * sqrt(cls.BOX[1] ** 2 + cls.BOX[2] ** 2)

    @classmethod
    def com_on_axis(cls):
        """True when the module is mounted so its mass straddles the spine's
        axis and makes no moment under a manoeuvre.

        THE OPTICAL AXIS IS NOT THE MODULE'S CENTRE.  The lens sits LENS_UP
        above the board's centre, so a module centred on the spine's axis
        does not look along it, and a module whose lens is on the axis has
        its mass LENS_UP off it.  The mount takes the second: the mass
        offset is a couple of millimetres and makes a moment, where the same
        offset as an aiming error is tens of thousands of microdegrees of
        pointing and would have to be calibrated out.  Both are true of the
        part, not of the design.
        """
        return abs(cls.LENS_UP) < cls.BOX[2] / 2.0


class Module2(CameraModule):
    """Raspberry Pi Camera Module 2 -- THE PART THE PRODUCT USES.

    CHOSEN FOR WHAT IT DOES NOT HAVE.  Module 3 has a motorised lens, and
    Module3 below records what that costs: an open-loop voice coil whose
    postural difference alone is 1.06% of the image distance and therefore
    1.06% of every range reported.  Module 2's lens is set by hand on a
    thread and then it stays where it is put.  The price is a shorter focal
    length on bigger pixels -- 0.367 mrad a pixel against 0.295 -- so the
    stochastic floor is a quarter worse, and that is a quarter of a much
    smaller number than the one it removes.  check_geometry does the sum.

    MEASURED OFF THE DRAWING, RP-008149-DS-1 (RPI-CAM-V2_1, 12/11/2015).
    Read from the PDF's own geometry rather than its dimension labels: the
    board outline, the four hole arcs and the housing square all scale at
    35.525 user units per millimetre, and against that the board comes back
    24.996 x 23.871 and the housing 8.500 x 8.500.
    """
    BOX         = (25.0, 9.0, 23.862)   # mm: along the baseline / toward the
                                        # scene / up.  The 25 and the 23.862
                                        # are the drawing's; the 9.0 is the
                                        # vendor table's overall thickness
                                        # and is the only source for it.
    CORNER_R    = 2.0                   # mm, the board's four corners
    HOLE_D      = 2.2                   # mm, labelled; 2.168 measured
    HOLE_PAD    = 4.75                  # mm, the keep-out around each
    HOLE_PITCH  = (21.00, 12.526)       # mm, measured off the hole arcs
    HOLES_UP    = -3.680                # mm, hole-pattern centre vs the board's
    # THE LENS HOUSING, and it is PLASTIC -- which is the difference that
    # matters against Module 3's metal can.  A square holder carrying a
    # threaded barrel, centred across the board and 2.477 mm above its
    # centre.  It is what the six struts bond to.
    CASE        = (8.5, 8.5)            # mm square, measured
    CASE_PLASTIC = True
    CASE_E      = 3.0e3                 # N/mm^2 [VERIFY: taken as an UNFILLED
                                        # thermoplastic, which is the weakest
                                        # thing the housing could be; a filled
                                        # one is three to five times this.
                                        # check_mount uses it to show the
                                        # struts are still the compliance]
    # NOT ON THE DRAWING: it is a top view and has no side.  The module is
    # 9 mm overall on the vendor table and the board is about 1, so the
    # holder AND its barrel stand about 8 proud; the SQUARE's own height is
    # less than that and nothing here says by how much.
    CASE_PROUD  = 5.0                   # mm [VERIFY: measure one.  check_mount
                                        # sweeps 3..8 and reports what moves]
    LENS_D      = 5.5                   # mm [VERIFY: no barrel on the drawing]
    LENS_UP     = 2.477                 # mm above the board's centre, measured
    PCB_T       = 1.0                   # mm [VERIFY]
    PCB_E       = 20.0e3                # N/mm^2 [VERIFY: typical FR4]

    # ---- optics.  The image area is the vendor's; the field is the
    # vendor's; and they agree on one focal length to a third of a percent,
    # which is the only cross-check these numbers have.
    SENSOR      = "IMX219"
    IMAGE_AREA  = (3.68, 2.76)          # mm, 4.6 diagonal
    PIXEL       = 1.12e-3               # mm
    FOCAL       = 3.04                  # mm
    F_NO        = 2.0
    FOV         = (62.2, 48.8)          # deg, horizontal / vertical
    FIXED_FOCUS = True                  # set on a thread, then locked

    MASS        = 3.0                   # g, vendor table
    HEAD_EXTRA  = 1.0                   # g [VERIFY: weigh a terminated head]
    ENVELOPE    = (25.0, 9.0, 24.0)     # mm, vendor table, BOX's axes
    BOND_MU     = 10.0                  # MPa allowable shear in a filleted joint
    FILLET_R    = 3.0                   # mm, the fillet the dispenser can lay


class Module3(CameraModule):
    """Raspberry Pi Camera Module 3 -- REJECTED, and this is the record why.

    The whole product argument is that a rigid spine removes the need to
    re-estimate the cameras' relative pose in flight.  That argument is
    about EXTRINSICS.  This module focuses with an open-loop voice coil,
    and its own sensor-assembly datasheet gives the coil's tolerances:

        postural difference  +-50 um   1.06% of the image distance
        hysteresis             8 um    0.17%
        dynamic tilt           8'      several pixels of principal point

    An axial lens shift is a focal-length error, and a fractional error in
    f is the same fractional error in Z -- it does not cancel between the
    cameras.  At 100 m the postural term alone is more than the entire
    inter-camera budget the truss exists to hold, from inside one camera,
    with nothing bent.  Kept here rather than deleted because the next
    person to look at a specification sheet will see 11.9 megapixels
    against 8 and want to know what it cost.
    """
    BOX         = (25.0, 11.3, 23.862)
    CORNER_R    = 2.0
    HOLE_D      = 2.2
    HOLE_PAD    = 4.75
    HOLE_PITCH  = (21.0, 12.5)
    HOLES_UP    = -3.70
    CASE        = (10.8, 10.8)          # a METAL can, and load-bearing
    CASE_PLASTIC = False
    CASE_E      = 70.0e3                # N/mm^2 [VERIFY: taken as aluminium]
    CASE_PROUD  = 3.875
    LENS_D      = 5.75
    LENS_UP     = 2.45
    PCB_T       = 1.12
    PCB_E       = 20.0e3

    SENSOR      = "IMX708-AAJH5-C"
    IMAGE_AREA  = (6.45, 3.63)
    PIXEL       = 1.4e-3
    FOCAL       = 4.74                  # back focal length
    F_NO        = 1.79
    FOV         = (66.0, 41.0)
    FOV_DIAG_SPEC = 75.0                # deg +-3, the lens's own spec
    IMAGE_CIRCLE = 8.4
    FIXED_FOCUS = False
    AF_STROKE   = (0.310, -0.050)       # mm, minimum travel
    AF_POSTURAL = 0.050                 # mm, lens shift with ORIENTATION at
                                        # a fixed drive current
    AF_HYST     = 0.008                 # mm
    AF_TILT     = 8.0 / 60.0            # deg, lens axis vs the sensor plane

    MASS        = 4.0
    HEAD_EXTRA  = 1.0
    ENVELOPE    = (25.0, 11.5, 24.0)
    BOND_MU     = 10.0
    FILLET_R    = 3.0


# THE PART THE PRODUCT USES.  Everything downstream says `Payload`, so the
# choice is made here, once, and the rejected alternative stays readable.
Payload = Module2


class Vision:
    """A camera on the carriage looking straight down past the ring.

    The pre-wind localisation of brief 5: find the chord's actual y and the
    mitre line's x before lowering, so the fixture's +-1 mm becomes a
    non-issue.  As with the competition rig, the SENSOR is not the limit:
    at 90 mm range a 1280-wide sensor at 40 degrees puts 20 pixels across a
    3 mm chord and an edge fit resolves a hundredth of a millimetre; what
    limits the cell is the camera-to-ring calibration, which is a bracket,
    and it is drawn ONCE per run as a bias, never averaged away.
    """
    W, H        = 1280, 720
    HFOV        = 40.0
    CAM_W       = 8.5          # mm, a board camera module's side
    CAM_CLEAR   = 3.0          # mm between the module and its neighbours
    FPS         = 30.0
    EDGE_SIGMA_PX = 0.15       # per-frame edge-fit noise
    # WHAT A LOOK IS GOOD TO ALONG THE CHORD, 1 sigma, mm.  MEASURED by
    # check_vision on rendered frames, and it is a LAW, not a constant: the
    # chord's own centreline is found to hundredths, but the joint's x comes
    # from where the two diagonals' LINES meet it, and their lateral fit
    # carries a lever arm of 1/sin(alpha).  So a shallower web is a worse
    # look, in proportion.
    #
    # Written as one number it was 0.15, measured on a 45-degree truss.  The
    # optimiser then moved the 300 mm design to 35 degrees and the rendered
    # rms went to 0.1951 -- which is 0.15 / (sin 35 / sin 45) = 0.185, to
    # within the scatter of nine joints.  A number would have read that as a
    # broken camera; the law says the truss got shallower.
    LOOK_SIGMA_REF   = 0.20    # mm [MEASURED by check_vision: 0.1951 rms
                               # over nine joints of the 300 mm truss, both
                               # run directions, rounded up]
    LOOK_ALPHA_REF   = 35.0    # deg, the web it was measured on
    # AND HOW MUCH CHORD THE LOOK NEEDS IN FRAME, mm.  A calibration, and
    # check_vision re-measures it: the joint's x comes from where the two
    # diagonals' lines meet the chord, and a line needs a run to be fitted
    # on.  The camera rides in a bay whose radius follows the head's, so
    # when the spool came off the rim the camera came in with it -- range
    # 55 mm to 47, field 22.7 mm of chord to 19.4 -- and the pixel path
    # lost the first joint of every chord and its rms went to 0.30 mm
    # against the look_sigma the web asks for.  So the standoff is solved
    # for the field as well as for the envelope.
    # ...swept 24/28/32/38/45 against the rendered path: 24 finds every
    # joint, 28 is the best rms, and past 32 the range grows faster than the
    # resolution and the bracket's bias stops passing through cleanly.
    LOOK_FIELD  = 28.0         # mm [MEASURED by check_vision]
    EXT_SIGMA   = 0.10         # mm, camera-to-ring calibration bias, 1 sigma
    EXT_ANG_SIGMA = 0.10       # deg
    SAG_MIN     = 1.5          # mm of strand sag that means tensioner slip

    @classmethod
    def look_sigma(cls, alpha):
        """1 sigma of a look along the chord, mm, for a web at `alpha`."""
        return (cls.LOOK_SIGMA_REF * sin(radians(cls.LOOK_ALPHA_REF))
                / sin(radians(alpha)))

    @classmethod
    def f_px(cls):
        return (cls.W / 2.0) / tan(radians(cls.HFOV / 2.0))

    @classmethod
    def mm_per_px(cls, range_mm):
        return range_mm / cls.f_px()

    # WHERE THE CAMERA RIDES.  In a bay of the carriage between the ring's
    # envelope and the dispenser, above the spool's sweep, looking down at
    # the joint a little off the nadir.  Measured, rendered: a camera on
    # the carriage over the ring plane looks into its own carriage box and
    # the spool; one outboard of the dispenser sees the chord but the
    # diagonals -- the mitre line that gives the joint's x -- hide behind
    # the nearest pin's flanks.  From the bay both diagonals are in view,
    # the far one through the ring's gap, which the plan parks downward.
    @classmethod
    def cam_bay(cls):
        return cls.CAM_W + 2.0 * cls.CAM_CLEAR

    @classmethod
    def look_standoff(cls):
        """Range at which LOOK_FIELD mm of chord fills the frame's height
        (the chord runs up the image -- see cam_frame)."""
        return cls.LOOK_FIELD * cls.f_px() / cls.H

    @classmethod
    def cam_pos(cls):
        """Ring frame, mm.  Outboard of the head's widest static part AND
        far enough back that the look's field fits: the head's envelope
        alone used to decide this, and it stopped being the binding term
        when the head shrank."""
        x = Head.ring_axial_half() + Head.TOOL_CLEAR + cls.cam_bay() / 2.0
        z_env = race_r_out() + cls.CAM_CLEAR + cls.CAM_W / 2.0
        r = cls.look_standoff()
        z_field = cls.cam_aim()[2] + sqrt(max(0.0, r * r - x * x))
        return (x, 0.0, max(z_env, z_field))

    @classmethod
    def cam_aim(cls):
        """The camera is set on the bore's floor under the ring centre: the
        chord it winds sits inside the bore when seated and a lift's
        worth below it at the look."""
        return (0.0, 0.0, -ring_r_in())

    @classmethod
    def cam_frame(cls):
        """(X, Y, Z) unit axes of the camera in the ring frame: MuJoCo's
        camera looks along -Z with +Y up the image.  Image right is world
        +y, so the chord runs up the image, away from the camera."""
        px, py, pz = cls.cam_pos()
        ax, ay, az = cls.cam_aim()
        f = (ax - px, ay - py, az - pz)
        n = sqrt(f[0] ** 2 + f[1] ** 2 + f[2] ** 2)
        Z = (-f[0] / n, -f[1] / n, -f[2] / n)
        X = (0.0, 1.0, 0.0)
        Y = (Z[1] * X[2] - Z[2] * X[1], Z[2] * X[0] - Z[0] * X[2], Z[0] * X[1] - Z[1] * X[0])
        return X, Y, Z

    @classmethod
    def range_nominal(cls):
        px, py, pz = cls.cam_pos()
        ax, ay, az = cls.cam_aim()
        return sqrt((ax - px) ** 2 + (ay - py) ** 2 + (az - pz) ** 2)


class Process:
    """Timing facts the schedule needs and the machine owns."""
    HZ          = 50.0
    VISION_SETTLE_S = 0.3
    GRIP_S      = 0.3
    ANCHOR_LEGS = 4            # a small loop round the post
    ANCHOR_S    = 1.5          # on top of the legs
    DOSE_SETTLE_S = 0.5
    LIFT_CLEAR  = 3.0          # mm over the computed minimum lift
    DROP_IN     = 1.0          # mm a rod is released above its V, to drop in
    SEAT_CLEAR  = 1.5          # mm the ring keeps off the cluster when seated
    SPEED_FACTOR = 1.0         # measured pace correction, calibrated later


# ============================================================ DERIVED + CHECKS
def stepper_scale_sigma():
    """Fractional screw growth over a shift: 12 ppm/K x 10 K."""
    return Gantry.SCREW_CTE * Gantry.DT_K


def ring_r_in():
    return Ring.ID / 2.0


def ring_r_out():
    return Ring.OD / 2.0


def ring_swept_r():
    """The largest radius anything ON THE RING reaches while it turns.

    The bobbin is recessed in the annulus now, so this is the rim -- plus
    the RUN_OUT, because the ring is not on a shaft: it runs in a raceway
    and check_drive measures its centre moving.  The raceway and the pinion
    reach further, but they do not turn: they are the head's static
    envelope, not its swept one, and race_r_out() is where they end.
    """
    return ring_r_out() + Ring.RUN_OUT


def rail_r_out():
    """Outer radius of the C-channel the ring runs in.  This is what the
    ring is captured by, and it is continuous over everything but the
    mouth."""
    return ring_r_out() + Ring.RACE_CLEAR + Ring.RACE_T


def race_r_out():
    """The head's widest STATIC radius: a pinion's tip, which stands further
    out than the rail does."""
    return max(rail_r_out(), Ring.mesh().tip_r)


def head_lowest():
    """How far the head reaches below the ring's centre, with the mouth
    aimed straight down at the work.

    Taken over the parts that are actually there, not over a blanket
    annulus at the widest static radius -- the widest static radius is a
    pinion, and a pinion is three discs at known azimuths, not a ring:

      * the rim, all the way round (the ring turns, so the gap is nowhere
        in particular): ring_r_out();
      * the rail, which starts at half the mouth off vertical and is
        deepest exactly there;
      * each pinion, a disc of its own tip radius hung at its azimuth,
        plus the bearing block that carries it.

    The blanket-annulus version of this said 22.5 mm; the parts say the rim
    wins at 20, because the pinions the annulus was drawn round sit high.
    """
    M = Ring.mesh()
    deep = [ring_r_out(),
            rail_r_out() * cos(radians(Ring.race_mouth() / 2.0))]
    r_block = max(M.r_pinion + M.m, Ring.BEARING_OD / 2.0)
    for az in Ring.pinion_az():
        deep.append(M.centre * cos(radians(az)) + r_block)
    return max(deep)


def gap_aim_limit():
    """How far off straight down the ring's gap may be parked.

    NOT a raceway constraint any more.  It was (Ring.race_mouth - GAP)/2, on
    the belief that the gap had to sit inside the mouth; Ring.race_mouth
    says why it does not, and with the mouth solved that formula returns a
    quarter of a degree and would have frozen the aiming the approach
    solver depends on.  What actually limits the aim is the gap's own
    half-width against the chord it has to keep admitting.
    """
    return Ring.GAP / 2.0 - Ring.chord_at_rail(max(Stock.DIAMETERS),
                                               Process.SEAT_CLEAR) / 2.0


def gap_chord():
    """Width of the gap's opening at the inner radius."""
    return 2.0 * ring_r_in() * sin(radians(Ring.GAP / 2.0))


def r_in_needed(truss, clear=None):
    """Inner radius the ring must have to turn round a joint of this truss.

    The ring's body sweeps half the wound band either side of the joint
    plus its own half-width, and across that reach the two diagonals rise
    from the chord at alpha; their upper edge at the far end of the sweep
    is what the inner rim must pass.  Chord radius, plus the rise over that
    reach, plus the rod's own half-thickness measured perpendicular to the
    chord.  This is brief 4.2's topology argument as a number, with the
    band in it: charged for the ring's width alone it passed trusses the
    solver then refused at the band's end.

    AND THE CLEARANCE IS TAKEN AT THE RIM'S CORNER.  Just outside the
    band the diagonal keeps rising and passes the corner of the rim
    obliquely, so the nearest approach is the radial slack times
    cos(alpha), not the slack itself.  Measured by the approach solver on
    a 50-degree truss: 1.87 mm of radial slack was 1.21 mm of clearance.
    """
    clear = Process.SEAT_CLEAR if clear is None else clear
    a = radians(truss.alpha)
    return (truss.d_chord / 2.0 + (Ring.W / 2.0 + truss.band / 2.0) * tan(a)
            + truss.d_diag / (2.0 * cos(a)) + clear / cos(a))


def notch_mouth():
    """Opening of a V-notch at the pin's tip -- the loader's capture range
    is this less the rod, split both ways."""
    return 2.0 * Cage.NOTCH_DEPTH * tan(radians(Cage.NOTCH_ANGLE / 2.0))


def capture_range(d_rod):
    return (notch_mouth() - d_rod) / 2.0


CHECKS = [
    ("the raceway's mouth passes the largest chord stocked, with seating clearance, "
     "which is the only thing that ever reaches the rail's radius",
     Ring.race_mouth() > Ring.chord_at_rail(max(Stock.DIAMETERS), Process.SEAT_CLEAR)),
    ("...and it is no wider than that, so the rim and not the raceway decides how "
     "far the head reaches below a chord",
     head_lowest() <= ring_r_out() + 1e-9),
    ("...and the gap can still be aimed as far as the cluster asks",
     gap_aim_limit() > 20.0),
    ("the ring cannot leave its raceway: the gap and the mouth side by side still "
     "leave live rail within a quarter turn of every direction",
     Ring.capture_arc() < 180.0),
    ("...what it can wander inside it fits in the bore the largest truss leaves",
     Ring.capture_wander() < 0.5),
    ("...and the wedge that oblique catch makes costs a tenth of the rating, "
     "not the whole of it",
     Ring.wedge_torque(2.0 + Ring.MASS / 1000.0 * 9.81) < Ring.DRIVE_TORQUE / 5.0),
    ("the rim and the pinion are the same module, so the drive can mesh at all",
     abs(2.0 * Ring.mesh().r_pitch / Ring.mesh().n_ring - Ring.mesh().m) < 1e-9
     and abs(2.0 * Ring.mesh().r_pinion / Ring.mesh().n_pinion - Ring.mesh().m) < 1e-9),
    ("...and nothing of the drive stands proud of the rim, which is the swept radius",
     Ring.mesh().r_pitch + Ring.mesh().m <= ring_r_out() + 1e-9),
    ("the pinion's tooth carries the drive torque with the stated factor",
     Ring.mesh().sigma * Ring.PINION_SF <= Ring.PINION_SIGMA),
    ("...and its root stands outside its own bearing",
     Ring.mesh().r_pinion - 1.25 * Ring.mesh().m
     >= Ring.BEARING_OD / 2.0 + Ring.PINION_WALL - 1e-9),
    ("the gap spans a whole number of teeth, so a pinion re-enters in phase",
     abs(Ring.teeth_in_gap() - round(Ring.teeth_in_gap())) < 1e-9),
    ("at least two pinions are meshed wherever the gap is",
     min(Ring.pinions_meshed(a) for a in range(0, 360, 1)) >= 2),
    ("...because the solver spaced them wider than a gap plus two contact arcs",
     Ring.pinion_margin() > 0.0),
    ("every pinion's contact arc sits clear of the mouth, where the work comes in",
     all(abs(((az + 180.0) % 360.0) - 180.0)
         > Ring.race_mouth() / 2.0 + Ring.contact_half() for az in Ring.pinion_az())),
    ("the thread groove is cut inside the ring's own section",
     ring_r_in() < Ring.GROOVE_R - Ring.GROOVE_D / 2.0
     and Ring.GROOVE_R + Ring.GROOVE_D / 2.0 < ring_r_out()
     and Ring.GROOVE_W < Ring.W),
    ("...and holds a whole truss of thread, so nothing is reloaded mid-cycle",
     Ring.groove_capacity(0.15) > 8.0),
    ("the gap admits the largest chord stocked, with room",
     gap_chord() >= max(Stock.DIAMETERS) + 2.0 * Process.SEAT_CLEAR),
    ("the exit guide is inside the rim and outside the inner clearance",
     ring_r_in() < Ring.EXIT_R < ring_r_out()),
    ("the head reaches less far below the chord than the old spool did",
     head_lowest() < 32.0),
    ("a full step is finer than the repeatability it has to deliver",
     Gantry.MM_PER_STEP < Gantry.REPEAT),
    ("screw growth over a shift exceeds the repeatability -- which is why the "
     "camera exists (brief 4.3)",
     stepper_scale_sigma() * 300.0 > Gantry.REPEAT * 0.5),
    ("tool strokes reach below a ring lifted clear of a joint",
     min(Head.DISP_STROKE, Head.GRIP_STROKE)
     > Head.TIP_PARK + ring_r_out() * 0.866 + 2.0),
    ("the dispenser sits outside the ring's swept envelope",
     Head.disp_x() - Dispenser.BODY_W / 2.0 > Head.ring_axial_half()),
    ("...and so does the gripper",
     -Head.grip_x() - Gripper.BODY_W / 2.0 > Head.ring_axial_half()),
    ("the gripper's jaws open wider than the largest rod",
     Gripper.JAW_OPEN > max(Stock.DIAMETERS) + 4.0),
    ("the diagonal rack pitches rods clear of the pads that straddle their neighbour",
     Magazine.DIAG_PITCH > Gripper.JAW_OPEN / 2.0 + Gripper.PAD_T
     + Magazine.SLOT_DEPTH * tan(radians(Magazine.SLOT_ANGLE / 2.0)) + 1.0 + 1.0),
    ("...and so does the chord rack",
     Magazine.CHORD_PITCH > Gripper.JAW_OPEN / 2.0 + Gripper.PAD_T
     + Magazine.SLOT_DEPTH * tan(radians(Magazine.SLOT_ANGLE / 2.0)) + 1.0 + 1.0),
    ("a V-notch captures more arrival error than the gantry and its screw "
     "growth can produce over a metre",
     capture_range(3.0) > Gantry.REPEAT + stepper_scale_sigma() * 1000.0),
    ("the notch is a V, not a slot: the included angle is under 120",
     Cage.NOTCH_ANGLE < 120.0),
    ("pins pitch at 50 mm, the brief's straightness rule",
     abs(Cage.PIN_PITCH - 50.0) < 1e-9),
    ("a ring parked at a thread post clears the end plate",
     Cage.END_FREE >= Cage.POST_OFF + Head.ring_axial_half() + Process.SEAT_CLEAR),
    ("the camera resolves a chord's edge to a hundredth of a millimetre",
     Vision.mm_per_px(Vision.range_nominal()) * Vision.EDGE_SIGMA_PX < 0.02),
    ("...so the bracket, not the sensor, is the budget",
     Vision.EXT_SIGMA > 5.0 * Vision.mm_per_px(Vision.range_nominal()) * Vision.EDGE_SIGMA_PX),
    # The payload has two independent descriptions -- a dimensioned drawing
    # and a vendor table -- and they are the only way to catch having read
    # the wrong module's drawing, which would move the standoff, the rod
    # lengths and the mount's whole geometry without failing anything else.
    ("the box read off the drawing agrees with the vendor's published envelope "
     "to a millimetre on every axis",
     all(abs(a - b) <= 1.0 for a, b in zip(Payload.BOX, Payload.ENVELOPE))),
    ("...and the vendor's two masses agree with the vendor's two envelopes: the "
     "gram between the modules comes with the 2.5 mm of depth a motorised lens "
     "needs, so neither number is a guess",
     abs((Module3.MASS - Module2.MASS) - 1.0) < 1e-9
     and abs((Module3.ENVELOPE[1] - Module2.ENVELOPE[1]) - 2.5) < 1e-9),
    ("a camera head is the module and its termination, and the guess is the "
     "smaller half",
     Payload.HEAD_EXTRA < Payload.MASS
     and abs(Load.tip_mass() - (Payload.MASS + Payload.HEAD_EXTRA)) < 1e-9),
    # The field of view is used to solve the mount's standoff, so it is
    # worth knowing it is the right field.  The published field and the
    # measured sensor area are independent, and they agree on one focal
    # length -- which is the only cross-check these optics have.
    ("the published 62.2 x 48.8 degree field and the measured 3.68 x 2.76 mm "
     "image area agree on one focal length",
     all(abs(a - b) < 0.5 for a, b in zip(Payload.fov_from_sensor(), Payload.FOV))),
    # ---- WHY THIS MODULE AND NOT THE OTHER ONE ----------------------
    # The whole product argument is that a rigid spine removes the need to
    # re-estimate the cameras' relative pose in flight.  That argument is
    # about EXTRINSICS.  Module 3 focuses with an open-loop voice coil, and
    # its own datasheet says the lens moves 50 um with ORIENTATION at a
    # fixed drive current -- 1.06% of the image distance, and therefore
    # 1.06% of every range it reports.  At 100 m that is more error than
    # the entire inter-camera budget the truss exists to hold, from inside
    # one camera, with nothing bent.
    ("the rejected module's autofocus alone moves the range answer further than "
     "the whole truss budget does",
     Module3.range_error_axial(Module3.AF_POSTURAL, 100.0)
     > Module3.range_error_yaw(Load.SLOPE_BUDGET, 100.0, 1000.0)),
    ("...and the tilt it adds while the lens is actually moving is worse again: "
     "a millimetre of lever on 8 arcmin is a pixel and a half of principal point",
     Module3.AF_TILT * pi / 180.0 * 1.0 / Module3.PIXEL > 1.0),
    # ...AND WHAT THE SWAP COSTS, which is not nothing: a shorter focal
    # length on bigger pixels is a coarser angle per pixel, and the
    # stochastic depth floor is proportional to it.
    ("the chosen module's fixed lens costs angular resolution -- it is a quarter "
     "coarser per pixel than the one it replaces",
     1.20 < Payload.pixel_angle() / Module3.pixel_angle() < 1.30),
    ("...and that price is a quarter of a much smaller number: the matching floor "
     "it gives up is under a tenth of the drift it removes",
     Payload.range_error_match(100.0) - Module3.range_error_match(100.0)
     < 0.10 * Module3.range_error_axial(Module3.AF_POSTURAL, 100.0)),
    ("...so the chosen module has no lens actuator at all, which is the point",
     Payload.FIXED_FOCUS and not Module3.FIXED_FOCUS),
]
