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
    # The cell as a source, alkaline, Energizer E91 class [VERIFY: the maker's datasheet and
    # discharge curve at the bat's drain; marked (mem) from memory].  Open-circuit volts
    # against the share of its capacity used, at a drain of a few mA where its own
    # resistance drops a few mV: the curve is flat, then falls steeply past 90 %
    CURVE       = ((0.0, 1.50), (0.50, 1.25), (0.75, 1.10), (0.875, 1.00), (1.0, 0.80))   # (mem)
    CAPACITY_AH = 2.9          # Ah at 25 mA to 0.8 V (an older E91 engineering sheet)
    R_FRESH     = 0.30         # ohm, DC pulse, the top of Energizer's 150-300 mohm
    R_END       = 0.75         # ohm, late in its discharge (read off a graph: low confidence)


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
    # the IMU's sensing centre above the board's top face, and its axes in
    # the bat frame with the board fitted: rows are the chip's x, y and z
    # [VERIFY: the board's drawing and the part's package drawing]
    IMU_Z       = 0.46         # mm, half an LGA package 0.91 mm high
    IMU_AXES    = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    # the chip on its board, 1 sigma: where pick-and-place leaves it and how
    # its solder tilts it [VERIFY: the board house's placement tolerance]
    CHIP_POS    = 0.05         # mm, along and across
    CHIP_ROT    = 0.5          # deg, about its normal
    CHIP_TILT   = 0.2          # deg, about either edge
    # how firmly what presses on the board holds it from sliding along its
    # slot: the pogo tips on its pads, its fingers on their strips, as a
    # friction coefficient [VERIFY: pull a clamped board along its slot]
    PAD_MU      = 0.2
    # the bulk capacitor on the supply, which carries the board while a cell's
    # contact is open.  A requirement until a board is chosen: board/power.py
    # derives the least that rides through Rules.BOUNCE_S (206 uF streaming to
    # the phone, its losses counted), and check_link holds this, at its low
    # tolerance, to it: the smallest E6 value that clears it.  220 uF did not
    # [VERIFY: the chosen board's BOM; 100-470 uF low-ESR is usual]
    HOLD_C      = 330e-6       # F
    HOLD_TOL    = 0.20         # of HOLD_C either way: a polymer or tantalum part's band
    # the firmware's numbers the record and the self-test check against: the
    # SWD port's IDCODE and the IMU's WHO_AM_I [VERIFY: the parts' datasheets]
    SWD_IDCODE  = 0x2BA01477   # Arm's SW-DP for a Cortex-M4 (mem)
    WHO_AM_I    = 0x44         # ICM-42686-P, register 0x75 (mem)


class Imu:
    """The board's IMU as the firmware runs it: a +-4000 dps / +-32 g part,
    ICM-42686-P class (README: the study saturated a +-2000 dps one).  Every
    number is the part's datasheet or the firmware's setting [VERIFY: every
    one, from the chosen part's datasheet; where the study
    (imu_fusion_sim.DATASHEET) had a value from memory, it is kept].  Spreads
    are 1 sigma, part to part, before any calibration."""
    GYRO_FS     = 4000.0       # dps, full scale
    ACC_FS      = 32.0         # g
    BITS        = 16           # signed codes, -32768..32767: one LSB is FS / 32768
    ODR         = 1000.0       # Hz, the output rate the firmware sets (the study's)
    FILTER_ORDER = 2           # the output path's low-pass, Butterworth
    FILTER_BW   = 250.0        # Hz, its -3 dB point (the study's noise bandwidth)
    GYRO_ND     = 0.0028       # dps/rtHz, white
    ACC_ND      = 70e-6        # g/rtHz
    GYRO_BIAS   = 0.5          # dps, zero-rate offset
    ACC_BIAS    = 0.020        # g, zero-g offset
    GYRO_SCALE  = 0.003        # of the sensitivity
    ACC_SCALE   = 0.003
    GYRO_CROSS  = 0.006        # each off-diagonal, the die's misalignment included
    ACC_CROSS   = 0.005
    GYRO_GSENS  = 0.05         # dps/g, each element
    GYRO_TC     = 0.005        # dps/degC, the zero-rate offset against temperature
    ACC_TC      = 0.15e-3      # g/degC
    # one power-up's offsets about the part's own: what a stored bias is
    # worth at the next power-up [VERIFY: power one part up twenty times]
    GYRO_POWERUP = 0.02        # dps
    ACC_POWERUP = 1.0e-3       # g
    ODR_TOL     = 0.005        # of the output rate: the part's own oscillator
    TEMP_LSB    = 1.0 / 132.48 # degC a code, the die's temperature, in every sample
    TEMP_NOISE  = 0.02         # degC
    # the die after power-up, in a bat in the clamp: how far it warms and
    # how fast [VERIFY: a thermocouple on a board in a bat in the clamp]
    SELF_HEAT   = 3.0          # degC
    HEAT_TAU    = 120.0        # s
    START_S     = 0.5          # s from power to its first sample: boot and configure [VERIFY: the firmware]
    # supply current (ICM-42686-P datasheet DS-000639 v1.0): both sensors in
    # low-noise mode, and asleep with the accelerometer waking it on motion
    I_RUN       = 0.70e-3      # A
    I_WOM       = 15e-6        # A [VERIFY: the accelerometer's low-power mode at the wake rate;
                               # the datasheet's sleep is 7.5 uA]
    VDD_MIN     = 1.71         # V
    # the part's own self-test: the response to its electrostatic actuation,
    # as a share of the response trimmed at its factory [VERIFY: the datasheet's
    # table; ST's LSM6DSV states a 50-1700 mg window, so a wide one is usual]
    ST_RATIO    = (0.5, 1.5)
    ST_S        = 0.4          # s its self-test takes, both sensors, on and off [VERIFY: the datasheet's procedure]


class Mcu:
    """The board's radio SoC, nRF52840 class, run straight from the cells on
    VDD (normal mode: no regulator ahead of it, README).  [VERIFY: every
    number, from Nordic's nRF52840 Product Specification; (mem) is from
    memory, the rest from Nordic's product brief v3.0 and staff answers on
    DevZone]  Currents typical, at 3 V on the DC/DC."""
    VDD_MIN     = 1.7          # V, recommended operating range, normal mode
    VDD_MAX     = 3.6
    V_POR       = 1.75         # V, the least it comes out of power-on reset at
    V_BOR       = 1.63         # V, the most its System ON brown-out reset trips at (typ 1.60) (mem)
    I_IDLE      = 3.16e-6      # A, System ON idle, RAM kept, RTC running
    I_CPU       = 3.3e-3       # A, the CPU running from flash at 64 MHz (52 uA/MHz)
    I_TX        = 6.40e-3      # A, radio sending at 0 dBm, its 32 MHz clock included
    I_RX        = 6.26e-3      # A, radio receiving
    # flash, programmed through the SWD port by the station's probe
    FLASH_WORD_S = 41e-6       # s, one 32-bit word written
    ERASE_ALL_S = 0.170        # s (mem; a page is 85 ms)
    SWD_HZ      = 8e6          # Hz, the port's fastest clock
    SWD_WORD_BITS = 46         # bits a 32-bit SWD read or write moves: request, ack, data, parity,
                               # turnarounds (Arm ADIv5)
    UICR_SERIAL = 0x10001080   # the UICR's customer words, where the station writes the serial
    UICR_WORDS  = 32
    PAGE        = 4096         # bytes a flash page holds: the record's two pages are two of these
    ERASE_PAGE_S = 0.085       # s, one page erased
    CLOCK_PPM   = 40.0         # its 32 MHz crystal, all tolerances: what the radio needs (Nordic)
    # the radio's cost in charge, measured with a Power Profiler Kit II on a
    # development kit (DevZone 78918): one connectable advertising event, one
    # connection event with nothing to send, and what a 20-byte notification
    # adds to it -- 950, 277 and 485 uA at a 20 ms interval
    Q_ADV       = 19.0e-6      # C
    Q_EVENT     = 5.5e-6       # C
    Q_NOTIFY    = 4.2e-6       # C, 20 bytes; each byte more costs its air time at I_TX


class Link:
    """Bluetooth LE as the bat uses it [the Core specification's rules;
    the centrals' own marked VERIFY].  The bat streams at Imu.ODR over the
    LE 2M PHY with data length extension, one notification a packet."""
    CI_STEP     = 1.25e-3      # s, a connection interval is a whole number of these (Core spec)
    CI_MIN      = 7.5e-3       # s, the Core spec's least: the station's own dongle
    # Apple's Accessory Design Guidelines R7, 11.6: an interval at least 15 ms
    # and a multiple of it, a supervision timeout of 2 to 6 s
    PHONE_CI    = 15e-3        # s
    SUPERVISION = 2.0          # s, the least either allows
    # packets a central takes in one connection event: iOS about five (Apple
    # staff, developer forums 773555); the station's dongle runs its own
    # firmware, whose event length is a setting [VERIFY: both]
    PER_EVENT   = {"phone": 5, "dongle": 6}
    MTU         = 247          # ATT MTU the bat asks for: 244 bytes a notification
    LL_PAYLOAD  = 251          # bytes, data length extension
    PHY_BPS     = 2e6          # LE 2M
    LL_OVERHEAD = 17           # bytes on air round a payload: preamble 2, access address 4,
                               # header 2, MIC 4, CRC 3, plus the L2CAP header's 2
    T_IFS       = 150e-6       # s between packets in an event (Core spec)
    # a packet lost on air is resent until acknowledged: the application
    # loses nothing, a lost packet costs latency, and the link drops only
    # past the supervision timeout.  An event closes on two CRC errors running
    PER         = 1e-2         # packets lost on air, indoors (Lodro et al. 2021) [VERIFY: at the
                               # station and at playing distance, carbon beside the antenna]
    CLOCK_PPM   = 50.0         # the most a central's active clock may be off (Core spec)
    # advertising, Apple's guideline 11.5: fast for 30 s after waking, then slow
    ADV_FAST_S  = 0.020
    ADV_FAST_FOR = 30.0
    ADV_SLOW_S  = 0.1525
    ADV_JITTER  = 0.010        # s, the Core spec's random advDelay added to each advertising event
    ADV_BYTES   = 31           # a legacy advertising payload


class Firmware:
    """The bat's firmware as the line builds it: settings within the parts'
    limits [VERIFY: the firmware, once written; until then the plan's].  Its
    boot, power to the first sample and to advertising, is Imu.START_S."""
    VERSION     = 0x00010000   # u32, the record's byte 24: 0x00MMmmpp, here 1.0.0
    IMAGE_BYTES = 225_000      # the SoftDevice S140 (about 156 kB) and the application [estimate]
    QUEUE       = 6            # notifications the stack holds; past them a packet is dropped
    CPU_PACKET_S = 0.3e-3      # s the CPU runs to read a packet's samples off the IMU's FIFO
                               # and hand it to the stack [estimate: 228 bytes of SPI at 8 MHz]
    WOM_G       = 0.05         # g off the rest reading that wakes it (the IMU's interrupt)
    IDLE_S      = 60.0         # s with no motion and no link before it sleeps
    ADV_FOR_S   = 120.0        # s it advertises after waking with no central before it sleeps


class Probe:
    """The station's SWD debug probe [VERIFY: the chosen probe; a J-Link
    class reaches the port's 8 MHz, a CMSIS-DAP dongle about half]."""
    SWD_HZ      = 4e6


class Smu:
    """What powers and measures the board through the pogo block: a
    source-measure unit, Nordic Power Profiler Kit II class, which sources
    0.8-5 V and samples the current at 100 kHz [VERIFY: every number, from
    the chosen instrument's datasheet; from memory]."""
    V_RANGE     = (0.8, 5.0)
    I_MAX       = 1.0          # A
    RATE        = 100e3        # Hz, current samples
    ACC         = 0.10         # of the reading, the current's accuracy
    FLOOR       = 0.2e-6       # A, its resolution at the bottom range


class DummyCell:
    """One cell of the swing rig's dummy pack: an AA-size supercapacitor,
    weighted to an alkaline cell's mass (README, "a sample at full speed").
    [VERIFY: an AA-size (14 x 50 mm) EDLC's datasheet; placeholders]"""
    C           = 10.0         # F
    V_RATED     = 2.7          # V
    ESR         = 0.10         # ohm
    D           = 14.0         # mm
    L           = 50.0         # mm, pip included: it is made to the AA's IEC envelope


class Servo:
    """The swing rig's motors: AC servomotors with absolute encoders, each
    through a single-stage planetary gearbox, run by drives that take a
    torque command every cycle over EtherCAT (cyclic synchronous torque)
    [VERIFY: every number, from the chosen maker's catalogue and the
    drive's manual; the catalogue is a Delta ECMA-C2 series' from memory].
    Rows: (name, rated N m, peak N m, rated rpm, max rpm, rotor kg m^2)."""
    CATALOGUE   = (("200 W, 60 mm flange", 0.64, 1.92, 3000.0, 5000.0, 0.177e-4),
                   ("400 W, 60 mm flange", 1.27, 3.82, 3000.0, 5000.0, 0.277e-4),
                   ("750 W, 80 mm flange", 2.39, 7.16, 3000.0, 5000.0, 1.13e-4),
                   ("1 kW, 100 mm flange", 3.18, 9.54, 3000.0, 5000.0, 2.65e-4),
                   ("2 kW, 100 mm flange", 6.37, 19.1, 3000.0, 5000.0, 4.45e-4))
    RATIOS      = (3.0, 4.0, 5.0, 7.0, 10.0)   # a single planetary stage's
    GEAR_EFF    = 0.97         # a single planetary stage
    # the load's inertia over the rotor's, reflected through the gearbox,
    # that the drive's auto-tuning holds a stiff loop on [VERIFY: the
    # drive's manual; makers state 10 for a stiff coupling]
    INERTIA_RATIO = 10.0
    CYCLE_S     = 1.0e-3       # s, the drive's command cycle
    ENCODER_BITS = 17          # absolute, per motor turn


class SwingRig:
    """The enclosed swing rig at station 4 (README, "a sample at full
    speed"): an arm on a hub carries the hands round the rated swing's arc,
    and the bat's clamp turns on a wrist at its end, driven through a
    timing belt by a second motor on the hub, so neither motor rides the
    arm.  [VERIFY: every number, on the build]"""
    TUBE        = (40.0, 2.0)  # mm, the arm's square aluminium tube, side and wall
    WRIST_MASS  = 350.0        # g, the wrist's bearings and housing, at the arm's end
    CLAMP_MASS  = 250.0        # g, the split collar on the grip and the wrist's pulley, turning with the bat
    BELT_RATIO  = 1.0          # the wrist belt's pulleys, hub to wrist


class Phone:
    """The factory iPhone's wide camera as the app runs it [VERIFY: the phone
    the line buys; chan's test phone is an iPhone 13; the study's values
    (imu_fusion_sim.camera_anchor) are from memory]."""
    W           = 1920         # px, 1080p
    H           = 1080
    F_PX        = 1500.0       # px, focal length at 1080p (about 65 deg across)
    FPS         = 60.0
    READOUT_S   = 1.0 / 120.0  # s, the rolling shutter's top row to its bottom (mem)
    EXPOSURE_S  = 1.0 / 500.0  # s, under the station's lights (a setting the app locks)
    SIGMA_PX    = 0.5          # px, a band's centroid, still (the study's)


class Material:
    """Densities, g/mm^3 [handbook values; VERIFY the ball's]."""
    ALU         = 2.70e-3
    STEEL       = 7.85e-3
    BALL        = 1.05e-3      # a retroreflective ball's moulded core


class Site:
    """Where the line stands [OPEN: chan's workshop; a Delhi placeholder].
    Gravity and the Earth's turn there are computed from it (imu/kin.py);
    the rig's heading is measured once when it is installed [VERIFY: a
    compass along the rig's x axis]."""
    LAT         = 28.6         # deg north
    ALT         = 216.0        # m above sea level
    HEADING     = 0.0          # deg, the rig's x axis from true north, clockwise seen from above
    AMBIENT     = 25.0         # degC, the room
    AMBIENT_SD  = 3.0          # degC, day to day
    LEVEL       = 0.1          # deg, the rig's base off level, 1 sigma [VERIFY: a spirit level on the rig's base]


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
    # how a cell's end comes back off a contact it strikes: the share of
    # its closing speed it leaves with [VERIFY: drop a cell onto a plate on
    # a load cell; a placeholder]
    RESTITUTION = 0.5


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


# ------------------------------------------- the station module's hardware
# Every station is one machine, copied (plan, "The station module"): three
# axes on MGN12 rails, each a NEMA 17 with an encoder on its shaft, a head
# carrying that station's tools on a 3-axis load cell, one camera looking
# down.  What follows is the hardware as bought.  station/spec.py derives a
# station from it -- its travel, which drive each axis needs, its head, its
# counterbalance, its stiffness -- and types none of it.

class Stepper:
    """The axis motor: a NEMA 17 hybrid stepper with an encoder on its
    shaft, run in full steps as the truss cell's are (brief 4.3).  The
    encoder is what catches a lost step (plan)."""
    TEETH       = 50           # rotor teeth: what makes a hybrid stepper's step 1.8 degrees
    HOLD_NM     = 0.45         # N m holding torque [VERIFY: the motor's datasheet]
    # pull-out torque against speed, (rpm, N m), at the driver's supply: the
    # torque a running motor has before it drops steps [VERIFY: the curve on
    # the datasheet, at the driver's voltage]
    PULLOUT     = ((0.0, 0.45), (300.0, 0.36), (600.0, 0.27), (900.0, 0.20), (1200.0, 0.14))
    ROTOR_J     = 6.8e-6       # kg m^2, rotor inertia [VERIFY: 68 g cm^2 is a 48 mm NEMA 17's]
    ENCODER_CPR = 4000         # counts a revolution: a 1000-line quadrature encoder [VERIFY]
    # the driver spreads each full step it is sent over the step's own period
    # in microsteps (Trinamic's MicroPlyer does this): the count, and every
    # place an axis stops, stay whole full steps, as the brief has them, but
    # the rotor no longer jerks a full step at a time.  check_gantry's rig:
    # without it a carriage stepped away from home rattles across its play
    # and the load cell reads 3.3 N of noise while it moves [VERIFY: the
    # chosen driver interpolates]
    INTERPOLATE = True
    # ... and it takes microsteps too: a step pulse can be a sixteenth of a
    # full step.  Only a touch uses them, creeping onto what it measures so
    # each reading rises by a sixteenth of a step's force; every move still
    # ends on a whole step [VERIFY: the driver's microstep setting]
    MICROSTEPS  = 16
    MASS        = 350.0        # g [VERIFY]
    FRAME       = 42.3         # mm square, a NEMA 17's flange (NEMA ICS 16)
    BODY_L      = 48.0         # mm, the can's length [VERIFY: the datasheet; a 0.45 N m NEMA 17 is a 48 mm can]
    # how quickly an axis stops ringing, as a damping ratio: the driver's
    # current loop and the rails' grease, which no drawing gives [VERIFY: a
    # ring-down on the built axis, an accelerometer on the head]
    ZETA        = 0.1


@dataclass(frozen=True)
class BallScrew:
    """A rolled ball screw with a standard single nut.  The plan's axes are
    SFU1204 [VERIFY: every column, from the screw's drawing]."""
    name:       str
    d0:         float          # mm, nominal diameter
    lead:       float          # mm a revolution
    d_root:     float          # mm: what whips, and what stretches
    play:       float          # mm, the nut's axial play (no preload) [VERIFY: a dial gauge]
    nut_k:      float          # N/um, the nut's axial stiffness
    lead_err:   float          # mm over any 300 mm: the accuracy class (C7)
    nut_l:      float          # mm, the nut's length


SFU1204 = BallScrew("SFU1204", d0=12.0, lead=4.0, d_root=9.9, play=0.03, nut_k=100.0,
                    lead_err=0.05, nut_l=35.0)
# the same family a size and two up, at the same lead so a step is the same
# 0.02 mm: what "bigger screws" would mean [VERIFY: both, from the drawings]
SFU1604 = BallScrew("SFU1604", d0=16.0, lead=4.0, d_root=13.5, play=0.03, nut_k=150.0,
                    lead_err=0.05, nut_l=42.0)
SFU2004 = BallScrew("SFU2004", d0=20.0, lead=4.0, d_root=17.5, play=0.03, nut_k=200.0,
                    lead_err=0.05, nut_l=42.0)
SCREWS = (SFU1204, SFU1604, SFU2004)


class ScrewEnds:
    """BK10 at the motor end and BF10 at the other, as SFU screws are sold:
    'fixed-supported' [VERIFY].  What sets how long a screw may be before
    it whips is this mounting and its root diameter, by THK's formula."""
    LAMBDA      = 15.1         # x 1e7: THK's critical-speed factor, fixed-supported, its 0.8 margin included
    LAMBDA_FF   = 21.9         # ... and fixed at both ends, for a screw too long to be supported
    DN_MAX      = 50000.0      # d0 (mm) x rpm: a rolled screw's ball-return limit [VERIFY: the vendor]
    END_L       = 30.0         # mm of screw beyond the travel at each end: half a block, half the nut
    BEARING_K   = 100.0        # N/um, the BK10's angular-contact pair, axially [VERIFY]
    COUPLING_K  = 60.0         # N m/rad, the jaw coupling's torsional stiffness [VERIFY]
    COUPLING_J  = 2.0e-6       # kg m^2 [VERIFY]
    EFFICIENCY  = 0.90         # a ball screw, either way: it backdrives
    E_STEEL     = 2.06e5       # N/mm^2
    RHO_STEEL   = 7.85e-3      # g/mm^3


class Rack:
    """Rack and pinion through a planetary gearbox: the drive for an axis
    too long to screw.  The plan had none; station/spec.py says when an axis
    needs one [VERIFY: every number, from the rack's and the gearbox's
    datasheets]."""
    MODULE      = 1.0          # mm
    PINION_Z    = 20
    # the ratios a planetary catalogue offers, one and two stages
    RATIOS      = (3, 4, 5, 7, 8, 10, 12, 15, 16, 20, 25, 28, 30, 35, 40, 50, 64, 70, 100)
    GEAR_PLAY   = 15.0         # arcmin of backlash at the output: an economy planetary
    GEAR_K      = 1.5          # N m/arcmin, torsional stiffness at the output
    GEAR_J      = 1.0e-6       # kg m^2 at the input
    GEAR_MASS   = 300.0        # g
    MESH_C      = 14.0         # N/(mm um) per mm of face: ISO 6336's single-pair stiffness c'
    FACE        = 10.0         # mm
    MESH_PLAY   = 0.05         # mm, rack to pinion
    EFFICIENCY  = 0.90         # gearbox and mesh
    MASS        = 0.8          # g/mm of rack


class Rail:
    """MGN12H carriages on MGN12 rail, as the truss cell has (BOM I) [VERIFY]."""
    DRAG        = 2.0          # N per carriage, light preload [VERIFY: pull one with a gauge]
    ZETA        = 0.05         # damping ratio of a carriage ringing on its drive: the rails, the nut
                               # and their grease [VERIFY: a ring-down on the built axis]
    BLOCK_MASS  = 52.0         # g a carriage
    MASS        = 0.65         # g/mm of rail
    BLOCK_L     = Gantry.BLOCK_L


class Frame:
    """The module's frame: aluminium extrusion and plate [VERIFY]."""
    BEAM        = 40.0         # mm, a 2040 profile's deep side
    BEAM_MASS   = 0.98         # g/mm of 2040
    PLATE_T     = 8.0          # mm, carriage plates
    RHO_AL      = 2.70e-3      # g/mm^3


class HomeSwitch:
    """An inductive proximity switch at each axis' low end [VERIFY]."""
    REPEAT      = 0.005        # mm, 1 sigma, where it trips at the creep speed below
    CREEP_V     = 1.0          # mm/s, the speed that repeatability is quoted at


class LoadCell:
    """The 3-axis load cell between the Z carriage and the tool plate: what
    every press, insertion and touch-off is judged by [VERIFY: the chosen
    sensor's datasheet]."""
    RATED       = 50.0         # N, each axis
    DEFLECT     = 0.10         # mm at the rated load: a load cell is a spring
    NOISE       = 0.02         # N rms, each axis, at the control rate
    ZETA        = 0.05         # damping ratio of the tools ringing on it [VERIFY: tap the plate]
    SIZE        = (40.0, 40.0, 20.0)   # mm
    MASS        = 120.0        # g


class HeadCam:
    """One global-shutter camera on the Z carriage, looking straight down
    (plan): an OV9281 module, 1280 x 800 at 3 um, an M12 lens [VERIFY]."""
    W, H        = 1280, 800
    PIXEL       = 3.0e-3       # mm
    F           = 6.0          # mm
    # a fiducial disc's centroid, 1 sigma, looked at centred.  Measured, not
    # chosen: check_gantry's pixels rig holds the model camera against
    # PixelVision on rendered frames, four sub-pixel phases of every disc,
    # and finds 0.016 px rms on the kid's electronics station and 0.069 on
    # the adult's pack station -- a bias common to its flat discs that is
    # not the light's fall-off, the headlight's specular or the renderer's
    # multisampling.  The larger is the one taken [VERIFY: on the camera]
    CENTROID_PX = 0.07
    MIN_PX      = 10.0         # px across a disc for that centroid to hold [VERIFY: with PixelVision]
    BOARD       = 25.0         # mm, the module's board, square
    LENS_L      = 20.0         # mm, board to the lens's front
    MASS        = 30.0         # g


class Fiducial:
    """A dark disc on a light ground, printed or engraved, SMEMA's 1-3 mm."""
    D           = 2.0          # mm
    CLEAR       = 3.0          # its keep-out, in diameters (SMEMA)


class Gripper2:
    """A two-jaw electric parallel gripper [VERIFY: the chosen model]."""
    STROKE      = 10.0         # mm, each jaw
    FORCE       = 30.0         # N, each jaw
    CLOSE_S     = 0.3
    BODY        = (40.0, 30.0, 60.0)   # mm, x y z
    JAW         = (12.0, 6.0, 20.0)    # mm: along the part, thick, long
    MASS        = 250.0


class Vacuum:
    """A bellows cup on a vacuum generator [VERIFY]."""
    CUP_D       = 10.0
    KPA         = 60.0         # kPa of vacuum at the cup (truss spec.StationB.VAC_KPA)
    TILT        = 5.0          # deg a face may lean and the bellows still seal
    BODY        = (16.0, 16.0, 40.0)
    MASS        = 40.0


class Magnet:
    """An electro-permanent magnet, switched by a pulse [VERIFY]."""
    D           = 20.0
    HOLD        = 30.0         # N on 2 mm mild steel, flush
    GAP         = 0.5          # mm of air gap at which it still holds a part
    BODY        = (20.0, 20.0, 25.0)
    MASS        = 60.0


class Pusher:
    """A flat face on the head, for pushing along the bat [VERIFY]."""
    FACE        = (20.0, 20.0)   # mm, y z
    BODY        = (20.0, 20.0, 40.0)
    MASS        = 80.0


class Pogo:
    """A block of spring probes, P75 class [VERIFY]."""
    N           = 5
    PITCH       = 2.54
    TIP_D       = 0.9
    TRAVEL      = 2.0          # mm, full compression
    WORK        = (1.0, 1.6)   # mm, the compression it is rated to make contact at
    FORCE       = 0.6          # N each, at the middle of WORK
    BODY        = (20.0, 12.0, 25.0)
    MASS        = 30.0
    R_CONTACT   = 0.020        # ohm a pin, worst [INGUN GKS-100's datasheet: 20 mohm or less]


class Spindle:
    """A geared DC motor with a collet [VERIFY]."""
    TORQUE      = 0.3          # N m at stall
    RPM         = 60.0
    COLLET_D    = 20.0
    BODY        = (30.0, 30.0, 60.0)
    MASS        = 150.0


class ToolSlide:
    """What lowers one tool below its neighbours: a compact two-position air
    slide, one per tool [VERIFY]."""
    STROKE      = 30.0         # mm
    FORCE       = 40.0         # N at the supply pressure
    TIME_S      = 0.3          # s, end to end
    BODY        = (20.0, 40.0, 50.0)   # mm, x y z, retracted
    MASS        = 90.0


class Build:
    """How far a built station lies from its drawing, 1 sigma [estimate:
    measure the first build].  The station measures all of it itself at
    commissioning (plan: "nothing is typed in"); the simulation draws it
    once per build, so there is something to find."""
    TOOL_XY     = 0.3          # mm, a tool on the plate
    TOOL_Z      = 0.3
    CAM_XY      = 0.5          # mm, the camera on the carriage
    CAM_TILT    = 0.3          # deg
    DOCK_XY     = 1.0          # mm, a dock's fiducials on the floor
    DOCK_YAW    = 0.2          # deg
    FIXTURE_XY  = 1.0          # mm
    HOME        = 0.5          # mm, where a home switch trips


class RigBuild:
    """How far the built camera rig lies from its drawing, 1 sigma
    [estimate: measure the first build].  Self-calibration finds all of it;
    the simulation draws it once per build, from its own random stream,
    so there is something to find."""
    CAM_POS     = 5.0          # mm, a camera on its bracket
    CAM_TILT    = 1.0          # deg
    FOCAL       = 0.015        # of the focal length (an M12 lens's +-5%, as 3 sigma)
    CENTRE_PX   = 15.0         # px, the principal point: lens thread to sensor
    DIST        = (0.02, 0.01, 0.003, 5e-4, 5e-4)   # k1 k2 k3 p1 p2, about the lens's curve
    COLOUR_GAIN = 0.05         # of each channel's gain: the sensor's white balance
    COLOUR_MIX  = 0.03         # each off-diagonal of the sensor's colour mixing (half-normal: a leak)
    MARKER      = 0.3          # mm, a marker's stem in its tapped hole
    AXIS_SKEW   = 0.1          # deg, the inner axis off square to the outer
    AXIS_OFFSET = 0.5          # mm, the inner axis off the outer
    ENCODER_ZERO = 0.2         # deg, each axis' encoder zero
    CENTRE      = 1.0          # mm, the gimbal's centre off its drawing
    LIGHT       = 0.05         # of a ring light's output
    # of the pull a marker's stem puts on its centroid, about the area
    # model (rig/vision.stem_bias_px): its scale, one number for the rig,
    # set by how bright a ball's edge is against its middle and by how the
    # centroider weights it [estimate: the rendered ring pulls 1.26-1.28 of
    # the model, check_cameras].  Self-calibration solves it
    STEM        = 0.3


class Artefact:
    """The module's own test pieces: what it commissions itself against and
    what check_gantry exercises every tool on (plan: "calibrated by the
    station itself").  Bought or turned [VERIFY: each]."""
    PIN_D       = 8.0          # mm, a hardened dowel standing on the fixture: touched from four
    PIN_H       = 20.0         # sides and on top by every tool, seen from above by the camera
    TOKEN_D     = 16.0         # mm, a mild-steel puck: gripped, sucked, held by the magnet, taken
    TOKEN_H     = 10.0         # by the collet -- one piece every pick tool can take
    TOKEN_RHO   = 7.85e-3      # g/mm^3
    NEST_WALL   = 3.0          # mm, the printed nest a token waits in
    PLUNGER_K   = 2.0          # N/mm, a spring plunger the press is checked against [VERIFY: its rate]
    PLUNGER_TRAVEL = 6.0       # mm
    PLUNGER_D   = 12.0
    STUD_LEAD   = 1.5          # mm, an M10 stud and nut: the screw op's test thread
    STUD_L      = 12.0         # mm of thread the nut runs down before it seats
    NUT_AF      = 17.0         # mm across flats; the collet takes it as a cylinder
    NUT_H       = 8.0
    PAD_W       = 12.0         # mm, the plate the pogo block lands on
    PAD_T       = 2.0          # mm thick


# ------------------------------------- the calibration station's camera rig
# Station 4 is the station module plus a two-axis gimbal that turns the bat
# through every orientation, and eight cameras round it that measure the
# clamp's pose in every frame (plan, "Calibration and test", milestone M2).
# What follows is that rig's hardware as bought.  rig/spec.py derives the
# rig from it and the bat -- the rings' radii, the gimbal's height, the
# marker size, the lens and how far the cameras stand off -- and
# rig/place.py places the cameras.  None of it is typed there.

class RingCam:
    """One of the eight: a global-shutter colour module, 1920 x 1200 at
    3.0 um (AR0234 class), an M12 lens, a ring light round the lens, all
    eight on one hardware trigger [VERIFY: the chosen module's datasheet]."""
    W, H        = 1920, 1200
    PIXEL       = 3.0e-3       # mm
    BITS        = 10
    FULL_WELL   = 10000.0      # e- at the gain the rig runs [VERIFY]
    READ_NOISE  = 4.0          # e- rms [VERIFY]
    # codes the sensor adds to every pixel before it is cut to BITS, so the
    # noise of a dark pixel is kept, not clipped at zero: clipped, a dark
    # channel reads high on average (a band's 1 % green read 2 to 5 %)
    BLACK_LEVEL = 42           # codes at 10 bits [VERIFY: the module's data pedestal]
    TRIGGER_SKEW = 1.0e-6      # s, 1 sigma, between cameras on one trigger line [VERIFY]
    # The M12 lenses stocked for a 1/2.6" sensor: (focal mm, k1, k2), the
    # radial terms fitted to each datasheet's distortion curve -- the
    # drawing a lens starts from; RigBuild says how far one lies from it.
    # Short lenses barrel hard [VERIFY: the supplier's curves]
    LENSES      = ((2.8, -0.30, 0.09), (3.6, -0.24, 0.06), (4.0, -0.20, 0.04), (6.0, -0.11, 0.015),
                   (8.0, -0.06, 0.005), (12.0, -0.03, 0.0), (16.0, -0.02, 0.0), (25.0, -0.01, 0.0))
    # px across a ball's image for its centroid to hold: the head camera's,
    # borrowed; nothing here has measured it for this camera and ball (the
    # rendered rig's smallest ball is 11 px) [VERIFY: with PixelVision]
    MIN_PX      = HeadCam.MIN_PX
    # a ball's centroid, 1 sigma per axis, against where the model camera
    # puts it: a BUDGET, the head camera's measured figure carried over,
    # not a measurement of this camera.  Every covariance here is stated at
    # it and the model camera draws its noise at it, so the rig is judged
    # at the worst it is allowed; check_cameras' rendered frames come in at
    # about a third of it (0.023 px), and fail the suite if they ever
    # exceed it [VERIFY: on the camera]
    CENTROID_PX = 0.07
    BODY        = (36.0, 36.0, 45.0)   # mm: board square, and its depth with lens and ring light
    LIGHT_D     = 60.0         # mm, the ring light's outside diameter
    MASS        = 90.0


class Marker:
    """A retroreflective ball on a threaded stem: what motion capture
    tracks.  Lit by a ring light round the lens that sees it, it returns
    hundreds of times what a white diffuse surface does, so a short exposure
    sees a uniformly bright disc on a dark frame [VERIFY: the supplier]."""
    D_MENU      = (6.4, 9.5, 12.7, 14.0, 19.0)   # mm, the sizes sold
    STEM_D      = 3.0          # mm, M3
    RETRO_GAIN  = 300.0        # its return over a white diffuse surface's, same light [VERIFY]


class Gimbal:
    """The two-axis gimbal: an outer ring on two bearings turning about the
    aisle's direction, an inner ring on two bearings in it, each axis a
    stepper with an absolute encoder.  Square aluminium tube [VERIFY]."""
    SECTION     = 20.0         # mm, the rings' square tube
    HUB         = (40.0, 30.0) # mm, a bearing housing's diameter and length
    POST        = 40.0         # mm, the outer axis' two posts, square
    ENCODER_BITS = 17          # absolute, per turn [VERIFY]
    WALL        = 1.5          # mm, the rings' tube wall [VERIFY]
    # each axis: a Stepper through a timing belt, its encoder on the axis.
    # The inner axis' motor rides the outer ring at one inner hub, with its
    # own mass as a counterweight at the other, so the outer axis stays
    # balanced [VERIFY: the build]
    GEAR        = 4.0          # motor turns an axis turn [VERIFY: a 15 to 60 tooth belt]
    ENCODER_HZ  = 1000.0       # Hz, how often the station reads both encoders [VERIFY: the controller]
    # the pogo block's cable crosses both axes: how far each may turn from
    # its zero, either way, before the wrap runs out [VERIFY: the wrap as
    # built; a slip ring makes it unlimited]
    WRAP        = 1.5          # turns


class Clamp:
    """What holds the bat in the inner ring: V-jaws closing on the handle
    against a stop at the pommel, the pogo block on the jaw over the
    window; a live centre engaging the tip [VERIFY: the build, M3]."""
    JAW_L       = 40.0         # mm along the handle the jaws hold
    JAW_T       = 12.0         # mm, a jaw outside the grip
    STOP_T      = 8.0          # mm, the pommel stop's plate
    BACK        = 30.0         # mm, stop to ring: the jaws' actuator
    CENTRE_D    = 24.0         # mm, the live centre's body
    # a colour card on the jaws' flanks and underside, four patches in a
    # row on each: matte white, red, green and blue, each a printed ink
    # with a measured reflectance (ISO 13655) [VERIFY: the card's certificate]
    CARD_PATCH  = 12.0         # mm, square
    CARD_RGB    = ((0.90, 0.90, 0.90), (0.75, 0.12, 0.10), (0.12, 0.60, 0.20), (0.10, 0.15, 0.65))
    # where a bat sits in the jaws, bat to bat and re-clamp to re-clamp,
    # 1 sigma: the V-jaws centre a grip that varies by Print.TOL [VERIFY:
    # clamp one bat ten times under the cameras]
    SEAT_POS    = 0.10         # mm, across the bat's axis
    SEAT_ROT    = 0.10         # deg


class CalBar:
    """The certified bar the gimbal turns for self-calibration: a carbon
    tube between the clamp's stop and the live centre, as long as the bat,
    with balls on short stems, each ball's place measured on a CMM
    [VERIFY: the certificate]."""
    BALLS       = 8
    CERT        = 0.005        # mm, 1 sigma, each ball's certified place (a CMM to ISO 10360)
    TUBE_D      = 25.0         # mm


# ===================================================== THE MODULE'S RULES
class Module:
    """What every copy of the station module is held to.  Rules, not facts:
    each comes from the brief, the plan or a design margin with its reason.
    station/spec.py derives a station that meets them or reports which one
    it misses."""
    REPEAT      = Gantry.REPEAT              # mm, bidirectional, anywhere in the travel (brief 4.3)
    STEP_MAX    = Gantry.mm_per_step("x")    # mm: the plan's 0.02, the truss cell's X and Y
    V_RATED     = Gantry.V_MAX               # what line.py's move times have assumed since M0:
    A_RATED     = Gantry.A_MAX               # the truss cell's own axes
    # Which drives an axis may have, in the order they are tried; the first
    # that reaches the rated speed at the axis' length is fitted.  The plan
    # specified 12 mm ball screws everywhere, and one of those whips long
    # before the line's longest axes reach the rated speed (station/spec.py
    # works out where).  chan chose rack and pinion for those, 2026-10-02.
    DRIVES      = ("screw", "rack")
    X_DRIVES    = 2            # X rides a rail each side and is driven on both, as a bridge is
    TORQUE_SF   = 2.0          # pull-out force over what a move asks: a stepper loses steps
                               # silently, so it is sized at half its curve (makers say 1.5-2)
    Z_SIGMAS    = 4.0          # sigmas of what is not yet measured that a moving tool clears by
    TOOL_GAP    = 5.0          # mm between neighbouring tools on the plate: room for a cap screw
    OVERRUN     = 3.0          # an op taking this many times what its sensors should need has failed:
                               # a watchdog, not a schedule (the schedule is the plan's times)
    TOUCH_FREE  = 5            # readings of free travel a touch takes to learn its own noise
                               # before it may call anything a contact
    TOUCH_POINTS = 3           # readings in contact a touch fits: a line needs two, the third
                               # is what tells a contact from a spike


class Rig:
    """What the calibration station's camera rig is held to.  Rules, each
    with its reason; rig/spec.py derives a rig that meets them or reports
    which one it misses."""
    CAMERAS     = 8            # the plan's ring
    # cameras that make a marker CHECKED: two triangulate it, the third is
    # what tells a wrong match or a glint from a marker without the rest
    MIN_VIEWS   = 3
    # checked markers the pose must rest on at every pose with any one
    # camera lost (blocked, knocked, dead): three not in a line fix a rigid
    # body, and PER_GAP keeps any three off a line.  The plan asked for
    # every marker checked at every pose; the ring hides its own balls from
    # too many directions for eight cameras to do that (README, M2)
    MIN_CHECKED = 3
    # balls between each neighbouring pair of the inner ring's four fixtures
    # (its two bearings, the clamp, the live centre), alternately above and
    # below the ring's plane: two per gap is the fewest that leaves no three
    # in a line and no four in a plane, so a pose never rests on a degenerate set
    PER_GAP     = 2
    # ball diameters of stem between a ball and the tube it stands on, as a
    # motion-capture base holds its ball: the tube then cuts into a ball's
    # outline only from behind the tube
    STANDOFF    = 1.0
    # of each calibration target (imu_fusion_sim.CALIBRATED) the rig's own
    # error may take: a 4:1 test uncertainty ratio, which grows the total
    # by 3% (sqrt(1 + 1/16)).  The plan's "well inside"
    SHARE       = 0.25
    # [OPEN] the CIE76 difference a band's colour may read back with: one
    # just-noticeable difference (Mahy et al. 1994), so the stored colour is
    # the printed one to the eye, until the app's band finder asks for less
    DE_MAX      = 2.3


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
    # [OPEN] how long two fresh alkaline cells must last: streaming, and
    # asleep in a cupboard.  The design doc's 200 h is its estimate of what
    # the board achieves, used here as the floor until chan sets one
    PLAY_H      = 200.0
    SLEEP_DAYS  = 365.0
    # [OPEN] the longest a cell's contact may open with the bat playing on:
    # a drop can open a cell's contacts for as long as 10 ms (Maxim AN3209,
    # 2004), and nothing here models a ball's impact [VERIFY: drop a bat]
    BOUNCE_S    = 10e-3
    # [OPEN] the share of the cells' capacity over which the bat rides
    # through BOUNCE_S: hold-up shrinks to nothing as they near the board's
    # least supply, so it is promised over most of their life, not all of it
    HOLD_LIFE   = 0.90
    # [OPEN] where the player's phone stands from the bat: the study's
    PLAY_Z      = 4.0          # m


class Quality:
    """How sure the line must be that a calibrated bat meets its targets
    (imu_fusion_sim.CALIBRATED: each a 1 sigma error over bats, as the
    study drew them).  Policy, not physics [OPEN: chan's]."""
    # a station is qualified on this many simulated bats, and a healthy one
    # may fail that qualification this often (README, Milestone 3)
    CAL_QUAL_BATS = 100
    CAL_QUAL_ALPHA = 1e-3
    # the golden bat: how many false stops a year a healthy rig may cause,
    # and how surely it catches a drift that would put bats out of target
    GOLDEN_STOPS_YEAR = 1.0
    GOLDEN_POWER = 0.99
    GOLDEN_RUNS = 20           # days at commissioning that certify it (imu/golden.py: on ten, the guard for
                               # measuring their scatter costs more runs than ten more days do)
    # a healthy board may fail the station's self-test, or the factory
    # iPhone's test, this often: each test's limits split it by Bonferroni
    TEST_ALPHA  = 1e-3
    # the bat's clock against the phone's, 1 sigma: what the study's
    # accuracy assumed (imu_fusion_sim.Anchor.sync_ms), so the link's clock
    # recovery is held to it
    SYNC_S      = 1e-3


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
    DUMMY_CELL_S = 8.0         # s, the battery path checked with a dummy cell (M6)
    SLEEVE_FEED_V = 20.0       # mm/s, the frame driven through the stretcher (M7)
    STRETCHER_S  = 6.0         # s, the fixture opening the split stretcher (M7)
    CODE_READ_S  = 2.0         # s, the hem code read by the head camera (M7)
    SPINDLE_RPM  = 60.0        # the pommel cap screwed on (M8)
    LABEL_S      = 12.0        # s, print a label, press it on, read it back (M8)
    COLLAPSE_S   = 20.0        # s, the draw rod pulled and the mandrel folded (M5)
    NOZZLE_S     = 30.0        # s, the dispenser changes its own mixing nozzle and purges (M5)
    # calibration: the gimbal's motion is the IMU plan's (imu/plan.py) and
    # the golden bat's day that plan's runs, as many as a drifted rig takes
    # to show (imu/golden.py), both computed (M3); station 2's self-test, the
    # factory iPhone's test and the swing rig's are their procedures' own
    # plans (board/procedures.py, phone/test.py, swing/rig.py), each held to
    # its simulation (check_link, check_swing) (M4)
    # one bat in this many is swung at full speed: the README's "first of
    # each batch, then as the bats' spread says" needs a few hundred
    # measured bats to set [estimate until then]
    SWING_EVERY  = 20
    TAG_READ_S   = 2.0         # s, a station reads a tray's tag
    DOCK_S       = 40.0        # s, a pose short of a dock to the tray seated and read (M9, M10)
    PACE         = 1.0         # measured pace correction, as truss.spec.Process.SPEED_FACTOR
    # a central's host stack, air to the app's callback: a floor, then an
    # exponential spread above it, and how well a lab characterisation of one
    # central model knows its floor (M4) [estimate: time a pin the bat toggles
    # as it sends against the app's callback, once per central model]
    HOST_FLOOR_S = {"phone": 2.0e-3, "dongle": 1.0e-3}
    HOST_SPREAD_S = {"phone": 3.0e-3, "dongle": 0.5e-3}
    HOST_FLOOR_SD = 0.2e-3
    # one board's currents about its parts' typical values, 1 sigma: datasheets
    # state typical values only (M4) [estimate: measure the first boards]
    CURRENT_SPREAD = 0.10


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
