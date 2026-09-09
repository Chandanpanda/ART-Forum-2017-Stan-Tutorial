"""Tier 2, slow: the ring's DRIVE -- is this head turnable at all?

The design review that produced this head said a ring encircling a rod
cannot have a shaft through it, so it must be driven on its rim, and
answered with teeth: a toothed rim, three pinions phased so the gap
cannot unmesh two of them, one motor and a belt.  Every one of those
claims is arithmetic in spec.Ring.  Here they are a body in a frame.

The ring in this rig has NO hinge and NO weld.  It is a free body with
six degrees of freedom, and what holds it is a C-channel raceway and
three involute pinions.  If the mesh is inconsistent it jams; if the
phasing is wrong it jams; if three pinions and a raceway with a mouth in
it do not capture a ring, it falls out of the frame.

    python3 sim/scripts/truss/check_drive.py [-v]
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from truss.glenv import headless        # noqa: E402  (before mujoco)
headless()
import numpy as np

from truss import drive
from truss.spec import Ring, Gantry

VERBOSE = "-v" in sys.argv
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def slip(rec, mesh):
    """Ring angle minus what the shaft says it should be, degrees.  This is
    the measurement the whole design rests on: a friction drive's version
    of it was unbounded, which is why the rim is toothed."""
    a = np.asarray(rec)
    return a[:, 1] + a[:, 2] * mesh.ratio


def main():
    M = Ring.mesh()
    tooth = 360.0 / M.n_ring
    back_deg = np.degrees(drive.RIG_BACKLASH / M.r_pitch)

    R = drive.DriveRig()
    check("the rig compiles: %d teeth on the rim, %d on each of three pinions, "
          "and a ring with a freejoint" % (M.n_ring - round(Ring.teeth_in_gap()), M.n_pinion),
          R.m.ngeom > 150 and R.m.nq == 10,
          "ngeom %d, nq %d (7 of the ring, 3 of the shaft)" % (R.m.ngeom, R.m.nq))

    # ------------------------------------------------- IS IT EVEN HELD?
    # Mouth straight down, gravity pulling the ring at the opening, no
    # drive.  Three pinions in an arc are not a bearing on their own and
    # the raceway is not one either where its mouth is; between them they
    # are, and this is the only place that says so.
    R.hold(0.5)
    pen = R.penetration()
    check("with the mouth aimed down and nothing driving it, the ring stays in its "
          "raceway -- radially inside the running clearance",
          R.wander_radial() <= Ring.RACE_CLEAR + pen + 1e-3,
          "%.3f mm of %.3f (clearance %.2f + %.3f of contact yield)"
          % (R.wander_radial(), Ring.RACE_CLEAR + pen, Ring.RACE_CLEAR, pen))
    check("...and axially, where the flanges hold it",
          R.wander_axial() <= Ring.RACE_CLEAR + pen + 1e-3,
          "%.3f mm of %.3f" % (R.wander_axial(), Ring.RACE_CLEAR + pen))

    # ...and when the gantry throws it.  The head is carried, not fixed.
    A = np.array([Gantry.A_MAX["x"], Gantry.A_MAX["y"], Gantry.A_MAX["z"]])
    T = drive.DriveRig(accel=A)
    T.hold(0.5)
    check("...and it is still held when the gantry accelerates at A_MAX on all "
          "three axes at once", T.wander_radial() <= Ring.RACE_CLEAR + T.penetration() + 1e-3,
          "%.3f mm radial, %.3f axial, under %.0f mm/s^2"
          % (T.wander_radial(), T.wander_axial(), np.linalg.norm(A)))

    # ------------------------------------------------- DOES IT TURN?
    t0 = time.time()
    peak, contact, rec = R.run(360.0)
    el = time.time() - t0
    a = np.array(rec)
    sl = slip(rec, M)
    check("a revolution against the winding tension loses no tooth: the ring is where "
          "the shaft says, to a fraction of a tooth (%.0f s)" % el,
          abs(sl[-1]) < tooth / 2.0 and np.abs(sl).max() < tooth / 2.0,
          "%.3f deg at the end, %.3f at worst, of a %.2f tooth (backlash is %.2f)"
          % (sl[-1], np.abs(sl).max(), tooth, back_deg))
    check("...and it arrives where it was sent, inside the parking tolerance",
          abs(a[-1, 0] - a[-1, 1]) < Ring.STOP_TOL,
          "%.3f deg of %.1f" % (abs(a[-1, 0] - a[-1, 1]), Ring.STOP_TOL))
    check("at least two pinions are engaged the whole way round, which is what "
          "spec.Ring.pinions_meshed claims and what locates the ring",
          a[:, 5].min() >= 2, "fewest %d; contacts fell to %d at hand-over"
          % (int(a[:, 5].min()), contact))
    # NOT the running clearance: turning, the ring's gap swings past the
    # raceway's mouth, and where the two coincide there is nothing to touch.
    # Ring.capture_wander is what the rail alone promises; the rig comes in
    # under it because the pinions hold the ring too.
    check("the ring stays in its raceway while it turns, inside what the gap beside "
          "the mouth allows", a[:, 6].max() <= Ring.capture_wander() + 1e-3,
          "%.3f mm radial of %.3f allowed (clearance %.2f, dead arc %.0f deg), "
          "%.3f axial" % (a[:, 6].max(), Ring.capture_wander(), Ring.RACE_CLEAR,
                          Ring.capture_arc(), a[:, 7].max()))
    # RUN_OUT is a calibration: the spec's clearance arithmetic subtracts it
    # from the ring's bore and adds it to the swept radius, so it has to be
    # what a rig says and not what would be convenient.
    check("...and RUN_OUT, which the spec charges against the ring's bore and its "
          "swept radius, is what the rig measures",
          a[:, 6].max() <= Ring.RUN_OUT + 1e-6,
          "%.4f mm measured against %.2f carried" % (a[:, 6].max(), Ring.RUN_OUT))

    # --------------------------------- WHAT THE GAP DOES ON ITS WAY PAST
    # The gap is a whole number of teeth so that the far side arrives in
    # phase.  Each pinion is passed once a revolution; this is what it
    # costs when it is.
    lost = []
    for az in Ring.pinion_az():
        k = int(np.argmin(np.abs((a[:, 1] % 360.0) - az)))
        lo, hi = max(0, k - 10), min(len(a) - 1, k + 10)
        lost.append(sl[hi] - sl[lo])
    check("the gap re-enters in phase: crossing a pinion costs a fortieth of a tooth, "
          "not a tooth", max(abs(x) for x in lost) < tooth / 10.0,
          "worst %.3f deg of a %.2f tooth, over three transits (%s)"
          % (max(abs(x) for x in lost), tooth, ", ".join("%.3f" % x for x in lost)))

    # ------------------------------------------------ WHAT IT COSTS
    # DRIVE_TORQUE is marked VERIFY in the spec, which means this is the
    # measurement it is waiting for.
    thread = R.tension * Ring.EXIT_R / 1000.0
    warm = a[a[:, 10] > Ring.SPINUP_S + 0.05]
    steady, spinup = warm[:, 9], a[a[:, 10] <= Ring.SPINUP_S, 9]
    check("the steady demand is the thread's drag plus a rail's friction, and the "
          "rating carries it with a factor",
          float(steady.mean()) * 2.0 < Ring.DRIVE_TORQUE,
          "%.4f N.m mean, %.4f peak, of %.3f rated -- the thread alone asks %.4f"
          % (steady.mean(), steady.max(), Ring.DRIVE_TORQUE, thread))
    check("...and the spin-up transient, which is the worst the drive ever sees, "
          "is inside the rating too", float(spinup.max()) < Ring.DRIVE_TORQUE,
          "%.4f N.m over 0 to %.0f rpm in %.1f s, of %.3f rated"
          % (spinup.max(), Ring.RPM, Ring.SPINUP_S, Ring.DRIVE_TORQUE))

    # -------------------------------------------------- HARDER CASES
    F = drive.DriveRig(rpm=Ring.RPM_MAX, tension=2.0 * R.tension)
    F.hold(0.1)
    pk2, _, rec2 = F.run(360.0)
    s2 = slip(rec2, M)
    b = np.array(rec2)
    ramp2 = (Ring.RPM_MAX / Ring.RPM) * Ring.SPINUP_S
    hot = b[b[:, 10] > ramp2 + 0.05][:, 9]
    check("at RPM_MAX and twice the winding tension it still loses no tooth, and the "
          "steady demand is still half the rating",
          np.abs(s2).max() < tooth / 2.0 and float(hot.mean()) * 2.0 < Ring.DRIVE_TORQUE,
          "%.3f deg at worst, %.4f N.m steady of %.3f rated (the ramp touches the "
          "current limit at %.4f, and loses nothing by it)"
          % (np.abs(s2).max(), hot.mean(), Ring.DRIVE_TORQUE, pk2))

    # Reversing matters: the ring backs up to park its gap for the next
    # joint, and backlash is where a reversal loses a turn if anywhere.
    B = drive.DriveRig()
    B.hold(0.1)
    B.run(180.0)
    B.run(-180.0)
    s3 = B.ring_angle() + B.pinion_angle(0) * M.ratio
    check("reversing loses no tooth either -- the gap is parked by backing up",
          abs(s3) < tooth / 2.0, "%.3f deg after 180 out and 180 back" % s3)

    # -------------------------------------------- THE CONTROL THAT FAILS
    # A rig that cannot fail is not a measurement (CLAUDE.md).  Press ONE
    # pinion on a quarter of a pitch out and the belt cannot accommodate it.
    p_p = 360.0 / M.n_pinion
    J = drive.DriveRig(phase_error=p_p / 4.0)
    J.hold(0.1)
    pk3, _, rec3 = J.run(180.0)
    j = np.array(rec3)
    check("and the rig can fail: ONE pinion pressed on a quarter of a pitch out and "
          "the drive JAMS -- so pinion_phase is doing work",
          abs(j[-1, 1]) < 10.0 and pk3 >= Ring.DRIVE_TORQUE - 1e-6,
          "ring reached %.2f deg of %.2f commanded, torque saturated at %.4f"
          % (j[-1, 1], j[-1, 0], pk3))
    # ...and the control's own control.  The phase that matters is RELATIVE:
    # the ring is a free body, so shifting all three pinions together is not
    # a fault, it is a different starting angle.  Written as an all-three
    # shift, the check above passed for the wrong reason.
    K = drive.DriveRig(phase_error=(p_p / 2.0,) * 3)
    K.hold(0.1)
    pk4, _, rec4 = K.run(180.0)
    k = np.array(rec4)
    check("...while shifting all THREE by half a pitch is not a fault at all: the "
          "ring is free, so it turns half a tooth and meshes",
          abs(k[-1, 1] - k[-1, 0]) < Ring.STOP_TOL * 2.0 and pk4 < Ring.DRIVE_TORQUE,
          "ring reached %.2f of %.2f, peak %.4f" % (k[-1, 1], k[-1, 0], pk4))

    if VERBOSE:
        print("  mesh: module %.2f, %d/%d teeth, ratio %.5f, pinions at %s"
              % (M.m, M.n_ring, M.n_pinion, M.ratio,
                 ", ".join("%.1f" % x for x in Ring.pinion_az())))
        print("  one revolution: slip %.3f..%.3f deg, torque peak %.4f, "
              "engaged %d..%d, wander %.3f radial / %.3f axial"
              % (sl.min(), sl.max(), peak, int(a[:, 5].min()), int(a[:, 5].max()),
                 a[:, 6].max(), a[:, 7].max()))
        print("  servo: kp %.3f N.m/rad (rating over one step), kv %.4f, shaft J %.2e"
              % (drive.KP, drive.KV, drive._shaft_inertia()))

    bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det and (VERBOSE or not ok) else ""))
    print("check_drive: %d checks, %d failed" % (len(RESULTS), bad))
    return 1 if bad else 0



if __name__ == "__main__":
    sys.exit(main())
