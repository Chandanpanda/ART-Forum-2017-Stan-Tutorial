"""Single source of truth for the IMU bat and the line that builds it.

All lengths MILLIMETRES, masses GRAMS, angles DEGREES, forces NEWTONS and
times SECONDS, unless a name says otherwise (_H hours, _MIN minutes,
_DAYS days).  Convert with mm()/g() at the point of use -- MuJoCo works in
metres and kilograms.

The same three kinds of thing as truss/spec.py, kept apart for the same
reason (CLAUDE.md, "constants that ARE legitimate"):

  * FACTS -- an AA cell's IEC envelope, the printer's build volume, the
    board's drawing, the knit's stretch, the robot's speed.  Measured,
    bought or standardised.  One nobody has measured carries [VERIFY: how].
  * PRODUCT AND LINE RULES -- two cells in line, a 30 mm band, the rated
    swing, the visit schedule.  Given by the brief or by chan; one chan has
    not set yet carries [OPEN].
  * TASK SPECS -- one bat (`Bat`) and one line (`Line`).  Dataclasses,
    because the line has to make more than one bat and run at more than
    one volume.

A fourth kind lives here because it is not yet any of those: ESTIMATES
(`Est`, `Faults`, `Person`).  Durations, rates and probabilities nobody
has measured, each naming the milestone whose check replaces it.  The line
model runs on them and says so.

What is deliberately NOT here: the core's section, the frame's truss, the
socket depths, the band positions, the sleeve blank, the tube, the tray,
the number of printers or mandrels, any buffer or store size.  Those are
answers.  product.py derives the bat from a Bat; line.py sizes the line
from a Line and a derived bat.

The station module is the truss cell's gantry, copied, so its axes are
truss.spec.Gantry's and are imported, not restated.
"""
from dataclasses import dataclass
from math import hypot
from typing import Optional

from truss.spec import mm, g, Stock, Gantry, Load as TrussLoad  # noqa: F401


# ============================================================== THE FACTS
class Cell:
    """An AA cell, IEC 60086-2 type LR6: the customer's, never the line's.

    The standard fixes the envelope, so the bay is sized off its extremes
    and not off one brand: the fattest, longest cell has to go in and the
    shortest pair still has to be held by the spring."""
    D_MAX       = 14.5         # mm, IEC 60086-2 LR6
    D_MIN       = 13.5
    L_MAX       = 50.5         # mm, overall, positive pip included
    L_MIN       = 49.2
    PIP_D       = 5.5          # mm, positive pip diameter, maximum
    PIP_H       = 1.0          # mm, positive pip projection, minimum
    MASS        = 23.0         # g, alkaline [VERIFY: weigh the brand the guide names]


class Print:
    """The in-house FDM process: cores, caps, collars, stretchers.

    Every number is a slicer setting or something a test print measures --
    a clearance gauge settles CLEAR, weighing a printed core settles INFILL
    -- not a design choice."""
    NOZZLE      = 0.4
    WALL        = 1.2          # mm, three perimeters: the thinnest wall a part may have
    CLEAR       = 0.25         # mm a side for a part that must slide [VERIFY: clearance gauge]
    TOL         = 0.15         # mm, scatter of a printed dimension [VERIFY: measure ten prints]
    INTERFERENCE = 0.10        # mm a side for a press fit [VERIFY: press-fit coupon]
    RHO         = 1.27e-3      # g/mm^3, PETG [VERIFY: the filament's datasheet]
    INFILL      = 0.30         # of a part's interior, beyond its walls [slicer setting]
    E           = 2.0e3        # N/mm^2, PETG as printed, along the layers [VERIFY]
    SNAP_STRAIN = 0.03         # allowable strain of a snap beam assembled once [VERIFY: Bayer's
                               # table gives 3-4 % for amorphous plastics; bend a printed beam]
    BOND_TAU    = 3.0          # N/mm^2 allowable shear, epoxy in a printed socket
                               # [VERIFY: pull a rod out of a coupon -- PETG bonds poorly]
    THREAD_PITCH = 2.0         # mm, the finest thread this process prints cleanly [VERIFY]
    THREAD_DEPTH = 1.0         # mm, radial
    THREAD_TURNS = 3.0         # turns of engagement a printed thread needs [VERIFY: torque one off]


class Printer:
    """One print-farm machine.  [VERIFY: chan's printers -- a 256 mm cube
    is the common size class, and nothing here assumes a brand]"""
    BUILD       = (256.0, 256.0, 256.0)   # mm, x y z
    PLATES      = 1            # beds it works through between visits with nobody there
    PART_GAP    = 5.0          # mm between parts on a bed, the slicer's spacing
    FLOW        = 12.0         # mm^3/s achieved, every move counted [VERIFY: time a print]


class Board:
    """The bought board: nRF52840-class radio, a +-32 g / +-4000 dps IMU,
    the hold-up capacitor, test pads on the top face, two contact fingers
    on the bottom.

    PLACEHOLDER GEOMETRY.  No board has been chosen, so this is the envelope
    the slot, the fingers and the pogo block are sized against until one is
    [VERIFY: every number, from the chosen board's drawing].  Every fit in
    product.py runs against these, which is why choosing the board early
    saves rework.  Positions are from the board's POMMEL edge (the edge
    nearest the pommel once fitted), along / across its centreline."""
    L           = 40.0
    W           = 12.0
    PCB_T       = 1.0
    PCB_TOL     = 0.10         # mm, board thickness tolerance (IPC class 2)
    L_TOL       = 0.10         # mm, routed outline tolerance
    TOP_H       = 2.2          # mm, tallest component on the top face (the radio module)
    EDGE_KEEP   = 1.0          # mm strip along each long edge kept clear, for the slot's ledges
    MASS        = 3.0
    IMU_AT      = (30.0, 0.0)
    # power, ground, SWDIO, SWCLK, reset: two staggered rows at 0.1 inch.
    # One row of five spans 11.7 mm and puts the outer pads under the slot's
    # ledges on a 12 mm board -- product.py's "pads between the ledges" fit
    # found that, so the placeholder keeps them in the parts' area.
    PADS_AT     = ((3.0, -2.54), (3.0, 0.0), (3.0, 2.54), (5.54, -1.27), (5.54, 1.27))
    PAD_D       = 1.5
    # the + finger and the return finger, (along, across): side by side
    # across the board so the two strips that reach them run side by side
    FINGERS_AT  = ((12.0, -2.5), (22.0, 2.5))
    FINGER_W    = 2.0
    FINGER_FREE = 1.5          # mm, a finger's free height below the board
    FINGER_MIN  = 0.3          # mm of deflection for its rated contact force
    FINGER_MAX  = 1.0          # mm of deflection before it yields


class ContactSet:
    """The bought stamped contacts: a + plate on a strip, and a return
    strip with a ring at the bay's mouth.  No wires, no solder: the board's
    fingers press on the strips' tabs and the cap's spring closes the
    circuit through the ring.  [VERIFY: a stock part, or a drawing to a
    stamping shop]"""
    T           = 0.3          # mm, nickel-plated spring steel
    W           = 4.0          # mm, strip width
    RING_W      = 3.0          # mm, the return ring, along the bat
    RHO         = 7.85e-3      # g/mm^3, steel


class CapSpring:
    """The conical spring in the pommel cap: a battery-holder spring,
    bought [VERIFY: the supplier's drawing].  Conical, so its coils nest
    and it closes to almost nothing."""
    L_FREE      = 10.0
    L_SOLID     = 1.5
    RATE        = 0.4          # N/mm
    D_BASE      = 17.0         # mm, the big coil: seated in the cap, it lands on the
                               # return ring round the bay's mouth when the cap is home
    D_TIP       = 5.0          # mm, the small coil, on the cell's flat negative end
    F_MIN       = 1.5          # N, least force for a reliable contact [VERIFY]
    MASS        = 0.5


class Knit:
    """The bought blank sleeve: a white polyester knit tube closed at one
    end, for sublimation.  [VERIFY: every number, on the blank chan buys]"""
    T           = 0.8          # mm, relaxed
    GSM         = 180.0        # g/m^2, relaxed
    STRETCH_MAX = 0.60         # hoop strain it takes without the print cracking
    STRETCH_MIN = 0.05         # hoop strain below which it wrinkles on the frame
    AXIAL_PER_HOOP = 0.30      # axial contraction per unit of hoop strain
    SQUEEZE     = 0.40         # of its thickness the collar crushes to hold the hem
    PRINT_TOL   = 2.0          # mm, where a band lands on the blank, printed by hand
    PLACE_TOL   = 3.0          # mm, where the sleeve lands on the frame [VERIFY: the sleeve rig, M7]


class Tube:
    """The bought shipping tube: spiral-wound kraft, a press-on cap at each
    end, the guide and the app's QR code printed on it by the supplier.
    [VERIFY: the supplier's sizes]"""
    WALL        = 1.5
    ID_TOL      = 0.5          # mm, on the inner diameter
    CAP_DEPTH   = 20.0         # mm a cap's skirt overlaps the tube
    RHO         = 0.75e-3      # g/mm^3, kraft board


class Epoxy:
    """The joint resin.  [OPEN: chosen before M5]  Its handling time is
    how long a mandrel stays in the cure rack, so it sets the number of
    mandrels; full cure happens in stock.  A hot cure would also decide
    the core's material."""
    HANDLING_H  = 6.0          # h at room temperature until the frame can leave its mandrel
    FULL_CURE_H = 24.0
    CARTRIDGE_ML = 50.0        # one dual cartridge [VERIFY]
    PURGE_ML    = 2.0          # lost to each new static mixer [VERIFY]
    MIXER_LIFE_MIN = 30.0      # pot life in the mixer: a new nozzle after an idle this long


class Thread:
    """The Kevlar joint thread, as bought.  [VERIFY: the spool]"""
    SPOOL_M     = 500.0


class Rod:
    """Carbon rod, as stored in the frame station's magazines."""
    MAGAZINE    = 120          # rods a magazine holds [VERIFY: the truss cell's racks]


class Label:
    ROLL        = 500          # labels a roll [VERIFY]


class Robot:
    """The owned mecanum robot, carrying one tray.  [VERIFY: measure it --
    its speed, deck and charge are not in the repo]"""
    V_MAX       = 300.0        # mm/s, loaded
    A_MAX       = 300.0        # mm/s^2
    DECK        = (600.0, 600.0)   # mm, (forward, across) (BOM J8 asks for this to be measured)


class Dock:
    """Where a tray is set down: two rails the robot drives between, each
    carrying steel balls on posts that the tray's V-grooves land on (plan,
    "Trays, docks and the mecanum robot").  So a tray must span the robot
    from rail to rail -- line.py works out which way round that puts it.
    [VERIFY: M9's dock]"""
    RAIL_W      = 30.0         # mm, one rail with its ball posts
    CLEAR       = 20.0         # mm, robot to rail, each side: its arrival spread [VERIFY: M10]


# ======================================================= PRODUCT RULES
class Rules:
    """What the bat must be, from the brief and chan's decisions."""
    N_CELLS     = 2            # AA in line, fitted by the customer (chan, 2026-10-01)
    BAND_W      = 30.0         # mm, each tracking band (design doc)
    # [OPEN] two hues far apart and rare in a living room
    BAND_RGB    = ((1.00, 0.10, 0.60), (0.10, 0.85, 0.25))
    # The swing the bat is rated for: the hardest in the accuracy study,
    # the one the +-4000 dps part was chosen against.  Named, so product.py
    # builds the study's own swing rather than a copy of its numbers.
    RATED_SWING = "hard, straight drive"
    # Margins the truss cell already holds its trusses to; the bat's frame
    # is built on the same machine, so it is held to the same.
    BUCKLE_SF   = TrussLoad.BUCKLE_SF
    ALPHA_RANGE = TrussLoad.ALPHA_RANGE
    D_CHORDS    = (2.0, 3.0)                 # structure.design's own menus
    D_DIAGS     = (1.0, 1.5, 2.0)
    LOAD_WALLS  = 2            # walls in a face that carries a load: cap ends, floors


# ============================================================ TASK SPECS
@dataclass(frozen=True)
class Bat:
    """One bat: what the line is asked to make.

    Only the envelope is given.  Everything inside it -- the section, the
    frame, the bays, the bands -- is derived by product.design().  The
    default sizes are placeholders until chan chooses them [OPEN]."""
    name:     str
    length:   float            # mm, pommel end to toe, caps on
    handle:   float            # mm, pommel end to the shoulder
    grip_d:   float            # mm
    width:    float            # mm, envelope across the face
    depth:    float            # mm, envelope face to back
    # product rules chan has not set: check_bat holds the bat to them once
    # they exist, and reports the bat against nothing until then
    mass_max: Optional[float] = None         # g, with cells [OPEN]
    balance:  Optional[tuple] = None         # (lo, hi) fraction of length from the pommel [OPEN]


# A child's cricket-style bat, about a size 5 (plan, "The product").
KID = Bat("kid", length=770.0, handle=270.0, grip_d=32.0, width=95.0, depth=55.0)


@dataclass(frozen=True)
class Line:
    """One line: the volume, the hours, the visits.

    Defaults are the design doc's: 300 a month, 16 hours a day, every day,
    a visit each morning and a restock once a week.  The rules marked
    [OPEN] are chan's to set; each sizes something."""
    demand_month:  float = 300.0
    days_month:    float = 30.0
    hours_day:     float = 16.0
    day_start_h:   float = 6.0
    visit_at_h:    float = 9.0
    visit_every_days: int = 1          # [OPEN] overnight, a weekend or a week?
    restock_every_days: int = 7
    bats_per_tray: int = 4             # [OPEN] the design doc's default
    outage_h:      float = 8.0         # [OPEN] the longest robot outage to ride through
    promise_days:  float = 3.0         # [OPEN] order to dispatch
    service:       float = 0.999       # [OPEN] chance a store lasts until its refill
    rods_by_hand:  bool = True         # rods cut to length at kitting, as the truss cell
                                       # assumes; False if bought cut [OPEN]
    farm_load:     float = 0.85        # [OPEN] share of the print farm's capacity the mean
                                       # demand may use: lower buys printers, higher buys stock
    name:          str = "line"

    @property
    def takt_s(self):
        """Seconds of running line per bat."""
        return self.days_month * self.hours_day * 3600.0 / self.demand_month

    @property
    def per_day(self):
        return self.demand_month / self.days_month


# ============================================================= ESTIMATES
class Est:
    """Process durations and rates nobody has measured.  Each names the
    milestone whose check replaces it; until then the line model runs on
    these and every report says so.  Moves are NOT here: a move's time is
    the gantry's trapezoid over a computed distance (line.py)."""
    INSERT_V     = 10.0        # mm/s, a force-watched insertion stroke (M6)
    PRESS_S      = 2.0         # s, a press fit to its stop, with the force curve (M6, M7)
    FLASH_S      = 40.0        # s, firmware, serial and self-test over the pogo block (M4)
    DUMMY_CELL_S = 8.0         # s, the battery path checked with a dummy cell (M6)
    SLEEVE_FEED_V = 20.0       # mm/s, the frame driven through the stretcher (M7)
    STRETCHER_S  = 6.0         # s, the fixture opening the split stretcher (M7)
    CODE_READ_S  = 2.0         # s, the hem code read by the head camera (M7)
    SPINDLE_RPM  = 60.0        # the pommel cap screwed on (M8)
    LABEL_S      = 12.0        # s, print a label, press it on, read it back (M8)
    COLLAPSE_S   = 20.0        # s, the draw rod pulled and the mandrel folded (M5)
    NOZZLE_S     = 30.0        # s, the dispenser changes its own mixing nozzle and purges (M5)
    # calibration (M3): the gimbal's recipe, from the design doc's steps
    GIMBAL_RATE  = 180.0       # deg/s between poses
    STILL_POSES  = 12
    STILL_S      = 2.0         # s averaged at each pose
    SPIN_AXES    = 3
    SPIN_TURNS   = 5           # whole turns about each axis
    SPIN_RPS     = 1.0
    LEVER_SPINS  = 2           # spins with the IMU off the axis
    LEVER_S      = 10.0
    IPHONE_S     = 60.0        # s, pair, find the bands, a slow fused swing (M4)
    SWING_EVERY  = 20          # one bat in this many is swung at full speed (M4)
    SWING_S      = 120.0
    GOLDEN_S     = 300.0       # s, the golden bat at the start of each line day (M3)
    TAG_READ_S   = 2.0         # s, a station reads a tray's tag
    DOCK_S       = 40.0        # s, a pose short of a dock to the tray seated and read (M9, M10)
    PACE         = 1.0         # measured pace correction, as truss.spec.Process.SPEED_FACTOR


class Faults:
    """What goes wrong, and how often [estimate: each is replaced by the
    first unattended runs].  Probabilities are per bat at a station, per
    plate at a printer."""
    REJECT      = {"frame": 0.010, "electronics": 0.010, "sleeve": 0.020,
                   "calibration": 0.010, "pack": 0.005}
    PRINT_FAIL  = 0.03
    STATION_MTBF_H = 40.0      # h of running between faults a station clears itself
    STATION_MTTR_MIN = 10.0    # min it takes to: self-check, re-home, carry on
    ROBOT_MTBF_H = 30.0
    ROBOT_MTTR_MIN = 5.0


class Person:
    """chan, as a resource of the line: minutes per unit of each manual
    task (plan, "Manual tasks, kept on purpose").  [estimate: time the first
    real batches]"""
    SLEEVE_MIN  = 3.5          # one sleeve: print, press, onto a stretcher, into a tray
    UNLOAD_MIN  = 0.5          # one plate off a printer, into tray nests, restarted
    STOCK_MIN   = 0.2          # per bat's worth of bought parts loaded
    CONSUMABLE_MIN = 2.0       # one spool, cartridge or label roll changed
    DISPATCH_MIN = 0.5         # one tube handed over; empties brought back
    REJECT_MIN  = 2.0          # one rejected bat cleared, its board recovered
    CUT_ROD_MIN = 0.15         # one rod cut to length against a stop (truss BOM, F4)
    FAULT_MIN   = 15.0         # one fault a station could not clear itself


# ================================================== THE SPEC, CHECKED
def _cells_fit_a_spring():
    """The cell stack's own scatter must fit inside the spring's working
    range, or no bay length holds both the longest and the shortest pair."""
    scatter = Rules.N_CELLS * (Cell.L_MAX - Cell.L_MIN)
    working = CapSpring.L_FREE - CapSpring.F_MIN / CapSpring.RATE - CapSpring.L_SOLID
    return working - scatter


CHECKS = [
    ("an AA cell's IEC envelope is a range, not a size",
     Cell.D_MIN < Cell.D_MAX and Cell.L_MIN < Cell.L_MAX),
    ("the cap spring's working range covers the scatter of two IEC cells",
     _cells_fit_a_spring() > 0.0),
    ("...and it still presses at its rated force at full length",
     (CapSpring.L_FREE - CapSpring.L_SOLID) * CapSpring.RATE > CapSpring.F_MIN),
    ("the spring's tip coil lands inside a cell's flat negative end",
     CapSpring.D_TIP < Cell.D_MIN),
    ("a board finger has a working window: its rated deflection is under its yield",
     Board.FINGER_MIN < Board.FINGER_MAX < Board.FINGER_FREE),
    ("the test pads sit on the board, apart from each other",
     all(Board.PAD_D / 2.0 < a < Board.L - Board.PAD_D / 2.0
         and abs(b) + Board.PAD_D / 2.0 < Board.W / 2.0 for a, b in Board.PADS_AT)
     and all(hypot(a0 - a1, b0 - b1) > Board.PAD_D
             for i, (a0, b0) in enumerate(Board.PADS_AT) for a1, b1 in Board.PADS_AT[i + 1:])),
    ("the fingers and the IMU sit on the board",
     all(0.0 < a < Board.L and abs(b) + Board.FINGER_W / 2.0 < Board.W / 2.0
         for a, b in Board.FINGERS_AT) and 0.0 < Board.IMU_AT[0] < Board.L),
    ("...and the two fingers are a strip's width apart across it, so their strips "
     "can run side by side",
     abs(Board.FINGERS_AT[0][1] - Board.FINGERS_AT[1][1]) > ContactSet.W),
    ("a sliding fit clears the process's own scatter",
     Print.CLEAR >= Print.TOL),
    ("a knit that wrinkles below its stretch limit has a window at all",
     Knit.STRETCH_MIN < Knit.STRETCH_MAX),
    ("the collar's squeeze is part of the knit, not more than it",
     0.0 < Knit.SQUEEZE < 1.0),
    ("a printed thread is coarser than the nozzle that prints it",
     Print.THREAD_PITCH > 2.0 * Print.NOZZLE and Print.THREAD_DEPTH < Print.THREAD_PITCH),
    ("the joint recipe's qualified angles are the truss cell's",
     Rules.ALPHA_RANGE == TrussLoad.ALPHA_RANGE),
    ("the rod menus are diameters the shop keeps",
     all(d in Stock.DIAMETERS for d in Rules.D_CHORDS + Rules.D_DIAGS)),
    ("every station a reject rate is given for exists, and only those",
     set(Faults.REJECT) == {"frame", "electronics", "sleeve", "calibration", "pack"}),
    ("a probability is a probability",
     all(0.0 <= p < 1.0 for p in list(Faults.REJECT.values()) + [Faults.PRINT_FAIL])),
    ("the default line's takt is the design doc's: 96 minutes at 300 a month",
     abs(Line().takt_s / 60.0 - 96.0) < 1e-9),
    ("...and 28.8 at 1,000",
     abs(Line(demand_month=1000.0).takt_s / 60.0 - 28.8) < 1e-9),
    ("the print farm is loaded below its capacity, or no stock is ever enough",
     0.0 < Line().farm_load < 1.0),
    ("the visit lands inside the line's day",
     Line().day_start_h <= Line().visit_at_h < Line().day_start_h + Line().hours_day),
    ("the band's two colours are two colours",
     len(Rules.BAND_RGB) == 2 and Rules.BAND_RGB[0] != Rules.BAND_RGB[1]),
]
