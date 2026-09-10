"""What the camera loses when the spine moves, in metres of range error.

THE CONVERSION, which is why this package exists.  A stereo pair reads
range from disparity, Z = f B / d.  Rotate one camera by dtheta about the
vertical and every disparity shifts by f dtheta pixels, so

    dZ = Z^2 dtheta / B

and the focal length cancels: a long lens does not save you.  Stretch the
baseline by dB instead and the error is a pure scale, dZ = Z dB / B.  Both
land in metres at a stated range, so a stiffness, a temperature and a
resonance can be added up in one currency and compared with a tube.

WHICH ROTATION IS THE KILLER.  With the spine along x and the optical axes
along y, a rotation about z swings one camera's axis toward its partner
and biases disparity: that is the yaw, and nothing downstream can absorb
it, because it is indistinguishable from a change of depth scale.  A
rotation about x tips the axis up, which rectification handles as long as
it is static.  A rotation about y rolls the image.  Both are reported;
only yaw is converted into range error.

RELATIVE, NOT ABSOLUTE.  Rigid motion of the whole rig changes nothing:
calibration is between the two faces.  Every number here is a difference
of the two, which is also why a centre-mounted spine is twice as sensitive
as a single tip's slope suggests -- the two cantilevers point opposite
ways, so their faces rotate in opposite senses and the errors add.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Pose:
    """One load case's effect on the pair, SI (radians, metres).

    ALL SIX DEGREES OF FREEDOM, not just the two that cost range.  The
    first version kept the baseline component of the translation and threw
    the other two away, because only yaw and scale convert into metres of
    range error.  That made the model unable to answer the question anyone
    asks first -- which of the six moved? -- and the honest answer to "the
    other two do not move" is a measurement, not an assumption.

    The three translations are along the baseline, across it horizontally,
    and vertically.  Only `d_baseline` enters range_error: a cross-baseline
    shift shows up as vertical disparity, which a stereo pair can see and
    correct without a calibration target, so it does not bias depth.
    """
    yaw:   float
    pitch: float
    roll:  float
    d_baseline: float
    baseline:   float
    # the two that do not cost range, so a synthetic Pose may omit them
    d_lateral:  float = 0.0
    d_vertical: float = 0.0

    def range_error(self, Z):
        """Metres of range error at range Z, from yaw and from scale."""
        return abs(Z * Z * self.yaw / self.baseline) + abs(Z * self.d_baseline / self.baseline)

    def yaw_deg(self):
        return np.degrees(self.yaw)


def relative_pose(model, u):
    """The pair's relative pose under a solved displacement field."""
    tl, rl = model.face_pose(u, "left")
    tr, rr = model.face_pose(u, "right")
    cl, cr = model.face_centre("left"), model.face_centre("right")
    axis = cr - cl
    B = float(np.linalg.norm(axis))
    axis = axis / B
    d = rr - rl
    dt = tr - tl
    # a frame on the baseline: along it, across it in the horizontal plane,
    # and up.  The optical axes lie along the model's y.
    up = np.array([0.0, 0.0, 1.0])
    lat = np.cross(up, axis)
    n = np.linalg.norm(lat)
    lat = lat / n if n > 1e-12 else np.array([0.0, 1.0, 0.0])
    return Pose(yaw=float(d[2]), pitch=float(d[0]), roll=float(d[1]),
                d_baseline=float(dt @ axis), d_lateral=float(dt @ lat),
                d_vertical=float(dt @ up), baseline=B)


# ------------------------------------------------------------ load cases
def accel_cases(model, duty):
    """The manoeuvre load along each axis, as {name: load vector}.

    All three are run because which one drives yaw is a property of the
    design, not something to assume: fore-aft is the obvious one, but a
    lattice with an odd number of bays is not symmetric and can convert a
    vertical load into yaw.
    """
    a = duty.manoeuvre_g * duty.g
    return {"accel x (along the baseline)": model.accel_load([1, 0, 0], a),
            "accel y (fore-aft)":           model.accel_load([0, 1, 0], a),
            "accel z (vertical)":           model.accel_load([0, 0, 1], a)}


def angular_cases(model, duty):
    """The aircraft rotating about each axis through its mount.

    MEASURED, AND THE ANSWER IS NOTHING.  A rigid-body angular
    acceleration moves both camera faces the same way -- the pair rotates
    and its relative pose does not change -- so it cannot disturb a
    calibration, and the torsional softness of an open lattice, which is
    where a closed tube would have beaten it, never gets to matter.  The
    case is run rather than argued because that is a claim about a
    particular structure, and an asymmetric one (a heavier lens on one
    side, drag on one head) would break it.
    """
    return {"angular about %s" % ax: model.angular_accel_load(v, duty.ang_accel)
            for ax, v in (("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1)))}


def thermal_cases(model, duty, n_dir=12):
    """A uniform soak, and the WORST gradient direction across the section.

    Direction matters and cannot be guessed: a gradient along z leans on
    the chords' differing heights and bends the spine vertically, which
    costs pitch; the gradient that costs YAW is the one across y, and
    which lattice orientation is worst depends on where the chords sit.
    So the section is swept and the worst reported, because the sun does
    not consult the drawing.
    """
    def soak(p):
        return duty.soak_K

    cases = {"soak %+.0f K" % duty.soak_K: (model.thermal_load(soak), model.member_dT(soak))}
    # the gradient is a temperature DIFFERENCE ACROSS THE SECTION, so it is
    # normalised by the section's own depth, not by wherever the nodes lie
    depth = model.depth
    worst = None
    for th in np.linspace(0.0, np.pi, n_dir, endpoint=False):
        n = np.array([0.0, np.cos(th), np.sin(th)])

        def grad(p, n=n):
            return duty.gradient_K * float(p @ n) / depth
        f, dT = model.thermal_load(grad), model.member_dT(grad)
        y = abs(relative_pose(model, model.solve(f)).yaw)
        if worst is None or y > worst[0]:
            worst = (y, f, dT, np.degrees(th))
    cases["gradient %.0f K across, worst at %.0f deg" % (duty.gradient_K, worst[3])] = (worst[1], worst[2])
    return cases


# ---------------------------------------------------------------- modal
def modal_yaw(model, n=12):
    """(frequencies, relative yaw per mode, mass participation per mode).

    A mode only matters if it MOVES THE PAIR RELATIVE TO EACH OTHER.  A
    frequency rule cannot see that; a mode that swings both cameras the
    same way costs nothing at all.
    """
    f, shapes = model.modes(n)
    yaw = np.array([relative_pose(model, s).yaw for s in shapes])
    return f, yaw, shapes


def transmissibility(model, duty, direction=(0, 1, 0), n=20, freqs=None):
    """Relative yaw per g of base acceleration, across the excitation band.

    Modal superposition with the duty's damping.  This is what the
    "first mode above 200 Hz" rule is a proxy for, and it is a poor proxy:
    what matters is how much relative yaw the excitation actually drives,
    which depends on where the modes are, how they are shaped, and how
    well they couple to the base motion.
    """
    K, M = model.assemble()
    free = model.free()
    f, shapes = model.modes(n)
    r = np.zeros(model.n * model.DOF)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    for k in range(3):
        r[k::model.DOF] = d[k]
    if freqs is None:
        lo, hi = duty.band_hz
        freqs = np.linspace(lo, hi, 60)
    w = 2.0 * np.pi * np.asarray(freqs, float)
    resp = np.zeros(len(w), dtype=complex)
    for i in range(len(f)):
        phi = shapes[i][free]
        mm = float(phi @ M[np.ix_(free, free)] @ phi)
        if mm <= 0:
            continue
        gamma = float(phi @ (M @ r)[free]) / mm
        yaw_i = relative_pose(model, shapes[i]).yaw
        wi = 2.0 * np.pi * f[i]
        resp += yaw_i * gamma / (wi ** 2 - w ** 2 + 2j * duty.damping * wi * w)
    return np.asarray(freqs), np.abs(resp) * duty.g       # rad per g


# --------------------------------------------------------------- forces
def member_margins(model, u, dT=None):
    """Peak member forces and the Euler margin of the worst compression."""
    N = model.member_forces(u, dT)
    worst_t = float(N.max()) if len(N) else 0.0
    worst_c = float(N.min()) if len(N) else 0.0
    margin = np.inf
    for k, mb in enumerate(model.members):
        if N[k] >= 0:
            continue
        L = float(np.linalg.norm(model.nodes[mb.j] - model.nodes[mb.i]))
        p_cr = np.pi ** 2 * mb.material.E * min(mb.section.Iy, mb.section.Iz) / L ** 2
        margin = min(margin, p_cr / abs(N[k]))
    return {"tension_N": worst_t, "compression_N": worst_c, "buckling_margin": float(margin)}


# ------------------------------------------------------------- the roll-up
def evaluate(model, duty, n_modes=20):
    """Every calibration metric for one spine under one duty.

    THE STATIC BUDGET AND THE VIBRATION BUDGET ARE KEPT APART.  At
    resonance a 1% damped structure amplifies fifty-fold, so any spine
    whose modes sit in the excitation band blows the calibration budget by
    orders of magnitude, and adding that number to a manoeuvre deflection
    of a few thousandths of a degree buries every design difference under
    one system decision.  Vibration is answered by isolation and damping,
    which are mount properties; what the SPINE owes is a low susceptibility
    and, if it can manage it, no modes in the band at all.
    """
    out = {"name": model.name, "mass_g": model.structural_mass * 1e3,
           "nodes": model.n, "members": len(model.members),
           "web": getattr(model, "web", "-")}
    worst = None
    for nm, f in accel_cases(model, duty).items():
        u = model.solve(f)
        p = relative_pose(model, u)
        axis = nm.split()[1]
        out["yaw_a%s_udeg" % axis] = np.degrees(p.yaw) * 1e6
        out["roll_accel_udeg"] = max(out.get("roll_accel_udeg", 0.0),
                                     abs(np.degrees(p.roll) * 1e6))
        if worst is None or abs(p.yaw) > abs(worst[2].yaw):
            worst = (nm, p.range_error(duty.range_m), p)
        if axis == "y":
            out["margins"] = member_margins(model, u)
    out["worst_accel"] = worst[0]
    out["yaw_accel_udeg"] = abs(np.degrees(worst[2].yaw) * 1e6)
    out["dz_accel_m"] = worst[1]
    out["baseline_m"] = worst[2].baseline
    for nm, (f, dT) in thermal_cases(model, duty).items():
        u = model.solve(f)
        p = relative_pose(model, u)
        key = "soak" if "soak" in nm else "grad"
        out["yaw_%s_udeg" % key] = abs(np.degrees(p.yaw) * 1e6)
        out["dbase_%s_um" % key] = p.d_baseline * 1e6
        out["dz_%s_m" % key] = p.range_error(duty.range_m)
    worst_ang = 0.0
    for nm, f in angular_cases(model, duty).items():
        p = relative_pose(model, model.solve(f))
        worst_ang = max(worst_ang, max(abs(np.degrees(v)) * 1e6
                                       for v in (p.yaw, p.pitch, p.roll)))
    out["ang_worst_udeg"] = worst_ang
    f, yaw, _ = modal_yaw(model, n_modes)
    out["f1_hz"] = float(f[0])
    out["f_modes_hz"] = f
    lo, hi = duty.band_hz
    out["modes_in_band"] = int(((f >= lo) & (f <= hi)).sum())
    fr, amp = transmissibility(model, duty, n=n_modes)
    out["yaw_vib_udeg"] = float(np.degrees(amp.max()) * 1e6 * duty.base_g)
    out["dz_vib_m"] = duty.range_m ** 2 * float(amp.max()) * duty.base_g / out["baseline_m"]
    # the static budget: what a manoeuvre and the sun do between them
    out["yaw_static_udeg"] = out["yaw_accel_udeg"] + out["yaw_grad_udeg"] + out["yaw_soak_udeg"]
    out["dz_static_m"] = out["dz_accel_m"] + out["dz_grad_m"] + out["dz_soak_m"]
    out["budget_used"] = out["yaw_static_udeg"] / (duty.yaw_budget_deg * 1e6)
    out["ok"] = out["budget_used"] <= 1.0
    return out


def row(r):
    """One evaluation as a printable row."""
    return ("%-26s %6.1f %7.1f %8.0f %7.0f %8.0f %7.2f %8.2f %8.0f %9.0f %5d"
            % (r["name"][:26], r["mass_g"], r["f1_hz"], r["yaw_accel_udeg"],
               r["yaw_grad_udeg"], r["yaw_static_udeg"], r["budget_used"],
               r["dz_static_m"], r["roll_accel_udeg"], r["yaw_vib_udeg"],
               r["modes_in_band"]))


HEADER = ("%-26s %6s %7s %8s %7s %8s %7s %8s %8s %9s %5s"
          % ("spine", "mass g", "f1 Hz", "yaw acc", "yaw dT", "yaw tot",
             "budget", "dz@100m", "roll", "yaw vib", "band"))
