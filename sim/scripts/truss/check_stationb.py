"""Station B: does it make the part station A expects?

    python3 sim/scripts/truss/check_stationb.py [-v]

THE CONTRACT BETWEEN TWO MACHINES is the only thing that matters here.  B
makes a head kit; A picks one up and lands six struts on its four
crossings.  If B's crossings and A's landings are not the same points, the
mount is not a mount, and NOTHING ELSE IN EITHER SUITE WOULD SAY SO --
each machine would be self-consistent and wrong about the other.

So the first checks are that contract, derived independently from each
side and compared.
"""
import os
import sys
from math import sqrt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np

from truss import structure, geometry, fixture, mjcf, mount, stationb
from truss.spec import (StationB, Bracket, Payload, Gripper, Stock, Process,
                        Magazine, Head)

RESULTS = []
VERBOSE = "-v" in sys.argv


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))


def main():
    for T in (structure.TRUSS_300, structure.TRUSS_1M):
        tag = "[%s] " % T.name
        g = geometry.TrussGeometry(T)
        so = mount.fov_standoff(g, d_strut=T.d_diag)
        M = mount.solve(g, d_strut=T.d_diag, standoff_mm=so)
        kit = stationb.Kit(d_rod=T.d_diag)

        # ------------------------------------------- THE CONTRACT
        xh, look, up = mount.landing_frame(g)
        c = np.asarray(M.payload[0][0], float)
        land = []
        for r in M.of(0):
            if r.kind != "strut":
                continue
            v = np.asarray(r.p1, float) - c
            p = np.array([float(v @ xh), float(v @ up), float(v @ look)])
            if not any(np.allclose(p, q, atol=1e-6) for q in land):
                land.append(p)
        cross = list(kit.crossings())
        worst = max(min(float(np.linalg.norm(p - q)) for q in cross) for p in land)
        check(tag + "B's four crossings ARE A's four landings, to the micron -- the "
              "two machines derive them separately and they have to agree",
              len(land) == 4 and worst < 1e-6,
              "%d landings, %d crossings, worst %.2e mm" % (len(land), len(cross), worst))
        check(tag + "...and the kit's grid rods are the mount's grid rods",
              len(kit.rods) == 4
              and abs(sorted(r.length for r in kit.rods)[0]
                      - sorted(r.length for r in M.of(0) if r.kind == "grid")[0]) < 1e-9,
              "%s mm" % [round(r.length, 2) for r in kit.rods])

        # ------------------------------------------- IT CAN BE WOUND
        check(tag + "the winder's bore clears the crossing it wraps -- two rods at "
              "right angles, one lying on the other",
              StationB.bore_margin(T.d_diag) > Process.SEAT_CLEAR,
              "%.2f mm of margin on a %.1f mm rod"
              % (StationB.bore_margin(T.d_diag), T.d_diag))
        # WHY THERE ARE TWO MACHINES, stated as the geometry rather than as
        # an assertion: the main head would have to encircle a crossing with
        # its bore along the lower grid rod -- which runs along the spine --
        # so its ring lies in the END TRIANGLE'S OWN PLANE, centred 20 mm
        # off the axis.  Its widest static radius then reaches straight
        # through the chords.
        from truss.spec import race_r_out
        xc = np.asarray(cross[0], float)
        r_cross = float(np.hypot(xc[0], xc[1]))
        room = g.R - r_cross              # from the crossing out to a chord
        check(tag + "the MAIN cell's head cannot reach a crossing at all: centred on "
              "one, its own rim reaches past the chords",
              race_r_out() > room and StationB.ring_r_out() < room,
              "the head is %.1f mm to its rim and there is %.1f mm from the crossing "
              "to the chord circle; B's winder is %.1f"
              % (race_r_out(), room, StationB.ring_r_out()))
        # a hoop round one rod alone would not tie the two together
        check(tag + "the hoop is laid perpendicular to the LOWER rod, so it encircles "
              "both -- the upper one passes through that plane at the crossing",
              abs(float(kit.wind_axis(0) @ np.array([1.0, 0.0, 0.0])) - 1.0) < 1e-9
              and abs(kit.rods[0].axis[0]) > 0.999
              and abs(kit.rods[2].axis[1]) > 0.999,
              "lower layer along x, upper along y")
        check(tag + "one turn of thread round a crossing is the hull of the pair, not "
              "a circle round one",
              kit.hoop_perimeter() > np.pi * T.d_diag,
              "%.2f mm a turn, %.0f mm of thread for the kit"
              % (kit.hoop_perimeter(), kit.thread_len()))

        # ------------------------------------------- IT CAN BE BUILT
        check(tag + "every rod of the kit is long enough for the jaws to take it at "
              "its middle",
              all(r.length >= Gripper.PAD_L + 4.0 for r in kit.rods),
              "shortest %.1f mm" % min(r.length for r in kit.rods))
        check(tag + "...and every one is rackable -- its blocks clear the pads",
              all(Magazine.blocks(r.length) for r in kit.rods),
              "%s" % [len(Magazine.blocks(r.length)) for r in kit.rods])
        ex, ey, ez = kit.extent()
        check(tag + "B's gantry covers everything B holds",
              2 * ex <= StationB.X_TRAVEL and 2 * ey <= StationB.Y_TRAVEL
              and ez <= StationB.Z_TRAVEL,
              "needs %.0f x %.0f x %.0f of %.0f x %.0f x %.0f"
              % (2 * ex, 2 * ey, ez, StationB.X_TRAVEL, StationB.Y_TRAVEL,
                 StationB.Z_TRAVEL))
        # THE COLLAR IS BIGGER THAN THE MODULE, and the nest has to let it past
        check(tag + "the nest's walls stop under the board's front face, so the collar "
              "-- which is wider than the module -- drops onto the HOUSING and not "
              "onto the nest",
              2.0 * kit.px > Payload.BOX[0]
              and Bracket.seat_l()[0] - Process.SEAT_CLEAR < Payload.BOX[1] / 2.0,
              "collar %.1f mm across a %.1f mm board; walls stop at z %.2f, board face "
              "at %.2f" % (2 * kit.px, Payload.BOX[0],
                           Bracket.seat_l()[0] - Process.SEAT_CLEAR,
                           Payload.BOX[1] / 2.0 - Payload.CASE_PROUD))
        check(tag + "layer 0 goes down first: layer 1 lies ON it at every crossing",
              [r.layer for r in kit.rods] == [0, 0, 1, 1]
              and kit.l1 > kit.l0,
              "layers at z %.2f and %.2f" % (kit.l0, kit.l1))
        check(tag + "...and the whole kit clears the collar's own aperture, so nothing "
              "the station lays is in front of the lens",
              min(abs(float(r.p0[0])) for r in kit.rods if r.layer == 1)
              > Bracket.aperture()[0] / 2.0
              and min(abs(float(r.p0[1])) for r in kit.rods if r.layer == 0)
              > Bracket.aperture()[1] / 2.0,
              "grid at %.2f/%.2f, aperture %.2f/%.2f"
              % (kit.gx, kit.gu, Bracket.aperture()[0] / 2.0,
                 Bracket.aperture()[1] / 2.0))

    # ------------------------------------------- IT IS IN THE SCENE
    import mujoco
    T = structure.TRUSS_300
    g = geometry.TrussGeometry(T)
    M = mount.solve(g, d_strut=T.d_diag,
                    standoff_mm=mount.fov_standoff(g, d_strut=T.d_diag))
    g.attach_mount(M)
    fx = fixture.Fixture(g)
    kit = stationb.Kit(d_rod=T.d_diag)
    m0 = mujoco.MjModel.from_xml_string(mjcf.scene_cell(g, fx, n_drops=80))
    m1 = mujoco.MjModel.from_xml_string(mjcf.scene_cell(g, fx, kit=kit, n_drops=80))
    b_joints = [n for n in ("bx", "by", "bz", "bg", "bw", "bring", "bf_l", "bf_r")
                if mujoco.mj_name2id(m1, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0]
    check("station B is a SEPARATE SET OF ACTUATORS in the same scene -- it shares "
          "the floor with the cell and nothing else",
          m1.nu == m0.nu + 8 and len(b_joints) == 8,
          "%d actuators against the cell's %d; joints %s"
          % (m1.nu, m0.nu, ",".join(b_joints)))
    check("...and adding it moves nothing of the cell",
          m1.ngeom > m0.ngeom and m0.nu == 9,
          "%d geoms against %d" % (m1.ngeom, m0.ngeom))

    bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det and (VERBOSE or not ok) else ""))
    print("check_stationb: %d checks, %d failed" % (len(RESULTS), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
