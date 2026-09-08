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
from math import pi, tan, sin, cos, radians, sqrt, floor


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


# ============================================================ THE PRODUCT
class Load:
    """The stereo rig's load case and budgets (brief 2.2, 2.3, 6).

    The spine is centre-mounted, so each half is a cantilever with a camera
    head at its tip.  The budget is ANGULAR: relative yaw between the two
    camera faces, which is bending slope at the tips and nothing a
    software calibration can absorb (brief 2.2).
    """
    TIP_MASS        = 50.0     # g, one camera head
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
class Ring:
    """The C-ring winding head (brief 4.5): a 40 mm ring with a 60 degree
    gap, a spool and tensioner riding on its rim, two friction wheels
    90 degrees apart on the outside edge.

    The gap is what lets a closed hoop of thread go round ONE rod without
    a hand-off; the inner radius is what has to clear the two diagonals
    that leave the joint at alpha.  Both are asserted against the truss in
    structure.ring_fit, not assumed here.
    """
    OD          = 40.0
    ID          = 20.0
    GAP         = 60.0         # deg, open arc
    # A THIN PLATE, NOT A DRUM.  The ring sweeps the whole band as the x
    # axis feeds, so its body reaches band/2 + W/2 from the joint, where
    # the diagonals have risen toward the rim: at 10 mm wide no truss in
    # the qualified angle range fits a 20 mm bore (measured, -0.36 mm at
    # the band's end); at 4 mm the metre truss fits with 0.2 mm to spare.
    W           = 4.0          # mm, axial width of the ring body
    # the thread leaves the ring here, on a guide inside the rim
    EXIT_R      = 11.0
    # spool + tensioner block on the outside of the rim: radial, tangential,
    # axial extents.  Azimuth is measured from the gap's centre, so 180 puts
    # it diametrically opposite -- uppermost when the gap points down.
    SPOOL       = (12.0, 14.0, 14.0)
    SPOOL_AZ    = 180.0
    RPM         = 60.0         # winding speed
    RPM_MAX     = 150.0        # friction-wheel drive ceiling
    SPINUP_S    = 0.5          # 0 -> RPM
    WHEEL_SPACING = 90.0       # deg between the two friction wheels
    WHEEL_R     = 6.0
    SLIP_TORQUE = 0.03         # N.m before a friction wheel slips [VERIFY]
    STOP_TOL    = 3.0          # deg, how well the gap can be parked
    MASS        = 40.0         # g, ring + spool + thread
    # The structure above the ring -- drive wheels, their motor, the
    # carriage plate -- as a box in the ring's frame, z up from the ring
    # centre: (x half, y half, z0, z1).  It has to clear the cage when the
    # ring is seated, and it is what the approach solver sweeps.
    HEAD_BOX    = (22.0, 30.0, 24.0, 90.0)


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
        return max(Ring.W, Ring.SPOOL[2]) / 2.0

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
    PIN_T       = 3.0          # along x
    ARM_W       = 6.0
    NOTCH_ANGLE = 90.0         # included angle of the V
    NOTCH_DEPTH = 2.5          # from the pin's tip to the V's floor
    END_PLATE_T = 6.0
    # the plate stands clear of a ring parked at the post: post offset plus
    # the ring's axial half-extent plus clearance, asserted in CHECKS
    END_FREE    = 20.0         # plate face beyond the chord ends
    POST_R      = 1.5          # thread anchor post
    POST_OFF    = 8.0          # post from the chord end, axially
    CRADLE_L    = 6.0          # along the diagonal
    CRADLE_T    = 1.5          # wall
    CRADLE_ANGLE = 90.0
    # THE SPINE IS SIZED BY THE SPOOL.  The ring's spool sweeps 32 mm round
    # the chord, and the chord is R from the cage's axis; whatever tube
    # runs down that axis has to fit in what is left (measured: an 8 mm
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
    # what a look is actually good to, along the chord, 1 sigma: MEASURED
    # by check_vision on rendered frames (the chord's centreline is found
    # to hundredths; the joint's x comes from where the diagonals' lines
    # meet it, and a lever arm of 1/sin(alpha) on their lateral fit is
    # what sets this).  ModelVision draws its noise from it.
    LOOK_SIGMA  = 0.15         # mm
    EXT_SIGMA   = 0.10         # mm, camera-to-ring calibration bias, 1 sigma
    EXT_ANG_SIGMA = 0.10       # deg
    SAG_MIN     = 1.5          # mm of strand sag that means tensioner slip

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
    def cam_pos(cls):
        """Ring frame, mm."""
        x = Head.ring_axial_half() + Head.TOOL_CLEAR + cls.cam_bay() / 2.0
        z = ring_r_out() + Ring.SPOOL[0] + cls.CAM_CLEAR + cls.CAM_W / 2.0
        return (x, 0.0, z)

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
    """The largest radius anything on the ring reaches while it turns."""
    return ring_r_out() + Ring.SPOOL[0]


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
    ("one friction wheel is always on the ring: wheel spacing exceeds the gap",
     Ring.WHEEL_SPACING > Ring.GAP),
    ("the gap admits the largest chord stocked, with room",
     gap_chord() >= max(Stock.DIAMETERS) + 2.0 * Process.SEAT_CLEAR),
    ("the exit guide is inside the rim and outside the inner clearance",
     ring_r_in() < Ring.EXIT_R < ring_r_out()),
    ("the spool is opposite the gap, so it is uppermost when the gap is down",
     abs(Ring.SPOOL_AZ - 180.0) < 1e-9),
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
]
