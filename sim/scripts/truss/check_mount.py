"""Tier 0: can THIS cell put a camera on the truss it just built?

The mount is solved in truss/mount.py -- nine rods per end, bonded to the
module's own metal enclosure.  That answers where everything goes.  This
answers whether the machine standing in front of it can get there, and the
first two checks say it cannot, for a reason no amount of path planning
fixes: the gripper's jaws open 8 mm and the camera is 10.8 mm through its
narrowest rigid part.

    python3 sim/scripts/truss/check_mount.py [-v]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from math import tan, radians

import numpy as np

from truss import structure, geometry, fixture, mount
from truss.spec import (Truss, Cage, Carrier, Gantry, Gripper, Head, Load,
                        Magazine, Payload, Process, Stock, Dispenser)

VERBOSE = "-v" in sys.argv
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def pitch_bound(jaw):
    """The magazine pitch a jaw opening forces: a rod has to clear the pads
    that straddle its neighbour.  The same expression spec.CHECKS pins."""
    return (jaw / 2.0 + Gripper.PAD_T
            + Magazine.SLOT_DEPTH * tan(radians(Magazine.SLOT_ANGLE / 2.0)) + 1.0 + 1.0)


def reach_with(t, diag_pitch, chord_pitch, end_free):
    """The gantry travel a fixture wants with the magazine and the cage at
    the given pitches -- measured by building one, not re-derived."""
    d0, c0, e0 = Magazine.DIAG_PITCH, Magazine.CHORD_PITCH, Cage.END_FREE
    try:
        Magazine.DIAG_PITCH, Magazine.CHORD_PITCH, Cage.END_FREE = \
            diag_pitch, chord_pitch, end_free
        fx = fixture.Fixture(geometry.TrussGeometry(t))
        return fx.x_reach(), fx.y_reach()
    finally:
        Magazine.DIAG_PITCH, Magazine.CHORD_PITCH, Cage.END_FREE = d0, c0, e0


def main():
    from spine.chosen import OPTIMAL_1M as ch
    t = Truss(**ch.as_truss_kwargs())
    g = geometry.TrussGeometry(t)
    so = mount.fov_standoff(g, d_strut=t.d_diag)
    m = mount.solve(g, d_strut=t.d_diag, standoff_mm=so)
    fx = fixture.Fixture(g)

    # ---------------------------------------------- THE GRIPPER CANNOT
    thin, case = Carrier.span()
    check("the gripper cannot hold the camera: nothing on the module is inside its "
          "jaws except the lens barrel, and that is the one part a machine must "
          "not touch",
          Gripper.JAW_OPEN < case and Gripper.JAW_OPEN < thin
          and Payload.LENS_D < Gripper.JAW_OPEN,
          "jaws %.1f mm; module %.1f thin way, enclosure %.1f square, barrel %.2f"
          % (Gripper.JAW_OPEN, thin, case, Payload.LENS_D))
    # ...AND OPENING THEM IS NOT THE FIX.  The magazine's pitch is bounded
    # below by the jaw opening, the diagonal racks pitch along x, and the
    # gantry is already nearly full.  This is the check that decided there
    # would be a carrier.
    jaw_ok = case + 1.0                      # just enough to take the enclosure
    dp = max(Magazine.DIAG_PITCH, pitch_bound(jaw_ok))
    cp = max(Magazine.CHORD_PITCH, pitch_bound(jaw_ok) + 1.0)
    x_wide, _y = reach_with(t, dp, cp, Cage.END_FREE)
    x_now, y_now = fx.x_reach(), fx.y_reach()
    check("...and opening them is not the fix: jaws wide enough for the enclosure "
          "push the racks past the gantry's X travel, so the machine would need a "
          "longer axis to be able to pick up a camera",
          x_now <= Gantry.X_TRAVEL < x_wide,
          "%.0f mm of %.0f as built; %.0f with %.1f mm jaws (pitch %.0f/%.0f)"
          % (x_now, Gantry.X_TRAVEL, x_wide, jaw_ok, dp, cp))

    # ---------------------------------------------- SO THERE IS A CARRIER
    check("the carrier's boss is a stock rod diameter, so every clearance, capture "
          "and pitch rule in the cell has already been checked for it",
          Carrier.boss_d() in Stock.DIAMETERS
          and Carrier.boss_d() + 4.0 <= Gripper.JAW_OPEN,
          "%.1f mm boss in %s, jaws %.1f" % (Carrier.boss_d(), Stock.DIAMETERS,
                                             Gripper.JAW_OPEN))
    check("...and long enough for the pads to take it with clearance either side",
          Carrier.boss_l() >= Gripper.PAD_L + 2.0,
          "%.1f mm of boss for %.1f mm of pad" % (Carrier.boss_l(), Gripper.PAD_L))
    look = np.asarray(m.payload[0][1], float)
    axis = Carrier.boss_axis(look)
    check("the boss points straight out of the camera's back, 180 degrees from the "
          "optical axis, so it can never be in shot",
          float(axis @ (look / np.linalg.norm(look))) < -0.999,
          "%.1f degrees off the look" % np.degrees(np.arccos(
              np.clip(float(axis @ (look / np.linalg.norm(look))), -1.0, 1.0))))
    check("...and being radial rather than axial it costs the cage no room at all",
          abs(Carrier.reach(so) - (so + Payload.BOX[0] / 2.0)) < 1e-9,
          "reach %.2f mm is the module's own outer face" % Carrier.reach(so))
    check("the cage can turn the boss under the jaws with the axes it has",
          np.linalg.norm(mount.axis_from(*mount.pose_for(axis)) - axis) < 1e-9)

    # ---------------------------------------------- ROOM IN THE CAGE
    # THE CAMERA IS MOUNTED INSIDE THE CAGE, so the cage has to have room
    # for it, and as built it did not.
    worst_side = Load.SECTION_MAX
    wg = geometry.TrussGeometry(Truss(length=t.length, side=worst_side, alpha=45.0,
                                      d_chord=3.0, d_diag=min(Stock.DIAMETERS)))
    worst = Carrier.reach(mount.fov_standoff(wg, d_strut=min(Stock.DIAMETERS)))
    check("the cage's end freedom clears a loaded carrier on the LARGEST section the "
          "cell is specified to build, not just on the chosen one",
          Cage.END_FREE >= worst + Process.SEAT_CLEAR,
          "%.1f mm of end free against %.2f needed at a %.0f mm section"
          % (Cage.END_FREE, worst + Process.SEAT_CLEAR, worst_side))
    check("...and at the 20 mm the cage was first drawn with, the camera lands INSIDE "
          "the end plate -- which is why this number moved",
          20.0 < fx.nose_reach + Process.SEAT_CLEAR,
          "the chosen truss's carrier reaches %.2f mm past the chord ends"
          % fx.nose_reach)
    check("...and the winding head parked at a thread post still fits in the same room",
          Cage.END_FREE >= Cage.POST_OFF + Head.ring_axial_half() + Process.SEAT_CLEAR)
    check("making that room is NOT free: it pushes the end racks out with it, and the "
          "chosen truss is inside the gantry's X travel by a hair",
          x_now <= Gantry.X_TRAVEL
          and reach_with(t, Magazine.DIAG_PITCH, Magazine.CHORD_PITCH, 20.0)[0]
          < x_now,
          "%.0f mm of %.0f now, %.0f with the old 20 mm cage"
          % (x_now, Gantry.X_TRAVEL,
             reach_with(t, Magazine.DIAG_PITCH, Magazine.CHORD_PITCH, 20.0)[0]))

    # ---------------------------------------------- THE RODS THEMSELVES
    rods = m.rods
    check("the mount is nine rods an end and no machined part: three battens closing "
          "the triangle, six struts, no platform",
          len(m.of(0)) == 9 and len(m.of(1)) == 9
          and len(m.by_kind("platform")) == 0,
          "%d battens, %d struts, %d platform"
          % (len(m.by_kind("batten")), len(m.by_kind("strut")),
             len(m.by_kind("platform"))))
    check("every mount rod is long enough for the jaws to take it at its middle",
          all(r.length >= Gripper.PAD_L + 4.0 for r in rods),
          "shortest %.1f mm against %.1f of pad"
          % (min(r.length for r in rods), Gripper.PAD_L))
    check("...and every one is a stock diameter",
          all(abs(2.0 * r.r - t.d_diag) < 1e-9 for r in rods)
          and t.d_diag in Stock.DIAMETERS,
          "%.1f mm" % (2.0 * rods[0].r))
    check("every mount rod can be laid by the cage and the gripper together, with no "
          "axis the machine does not have",
          all(float(np.linalg.norm(mount.axis_from(*mount.pose_for(r.axis)) - r.axis))
              < 1e-9 for r in rods))
    # THE CAGE ANGLES THE MOUNT ASKS FOR.  A rod laid at a cage angle the
    # loading phase never uses is not a problem -- the cage turns freely --
    # but a mount that wants a hundred different ones is a slow mount.
    thetas = sorted({round(mount.pose_for(r.axis)[0], 3) for r in rods})
    check("the mount asks the cage for a handful of angles, not one per rod",
          len(thetas) <= len(rods) / 2.0,
          "%d angles for %d rods" % (len(thetas), len(rods)))

    # ---------------------------------------------- THE BONDS
    bonds = 2 * len(rods)                 # both ends of every rod
    check("every mount joint is bonded, not wound: the ring cannot reach past the "
          "last joint, and does not need to",
          min(abs(float(r.p0[0])) for r in rods) >= 0.0
          and all(float(r.p0[0]) <= t.length + Cage.END_FREE for r in rods),
          "%d fillets, %d rods" % (bonds, len(rods)))
    check("a filleted rod end carries the head's service load a thousand times over",
          mount.fillet_strength(t.d_diag)
          > 100.0 * Load.tip_mass() / 1000.0 * Load.G * Load.LATERAL_G,
          "%.0f N against %.3f N"
          % (mount.fillet_strength(t.d_diag),
             Load.tip_mass() / 1000.0 * Load.G * Load.LATERAL_G))
    check("...and is stiffer than the strut it holds, so the bond is not the compliance",
          mount.fillet_stiffness(t.d_diag)[0]
          > Stock.E * np.pi * (t.d_diag / 2.0) ** 2 / min(r.length for r in rods),
          "%.0f N/mm against the shortest strut's %.0f"
          % (mount.fillet_stiffness(t.d_diag)[0],
             Stock.E * np.pi * (t.d_diag / 2.0) ** 2 / min(r.length for r in rods)))
    check("the dispenser's nozzle is finer than the fillet it has to lay",
          Dispenser.NOZZLE_D < Payload.FILLET_R,
          "%.1f mm nozzle, %.1f mm fillet" % (Dispenser.NOZZLE_D, Payload.FILLET_R))

    # ---------------------------------------------- AND NOTHING IN SHOT
    check("with the mount built, nothing of the truss or of the mount is inside the "
          "camera's 66 x 41 degree field",
          mount.fov_clear(m) > 0.0, "%.2f mm to spare" % mount.fov_clear(m))
    check("...which took a solved standoff: the mechanical clearance alone leaves the "
          "truss in the picture",
          mount.fov_clear(mount.solve(g, d_strut=t.d_diag,
                                      standoff_mm=mount.mech_standoff())) < 0.0,
          "%.1f mm at the %.1f mm mechanical standoff, %.1f mm at the solved %.2f"
          % (mount.fov_clear(mount.solve(g, d_strut=t.d_diag,
                                         standoff_mm=mount.mech_standoff())),
             mount.mech_standoff(), mount.fov_clear(m), so))

    # ---------------------------------------------- REACHABLE AT ALL
    xs = [float(p[0]) for r in rods for p in (r.p0, r.p1)]
    ys = [float(p[1]) for r in rods for p in (r.p0, r.p1)]
    zs = [float(p[2]) for r in rods for p in (r.p0, r.p1)]
    check("every bond point is inside the gantry's own travel",
          max(xs) - min(xs) <= Gantry.X_TRAVEL
          and max(abs(y) for y in ys) <= Gantry.Y_TRAVEL / 2.0
          and max(abs(z) for z in zs) <= Gantry.Z_TRAVEL / 2.0,
          "x %.0f..%.0f, |y| <= %.0f of %.0f, |z| <= %.0f of %.0f"
          % (min(xs), max(xs), max(abs(y) for y in ys), Gantry.Y_TRAVEL / 2.0,
             max(abs(z) for z in zs), Gantry.Z_TRAVEL / 2.0))

    bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det and (VERBOSE or not ok) else ""))
    print("check_mount: %d checks, %d failed" % (len(RESULTS), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
