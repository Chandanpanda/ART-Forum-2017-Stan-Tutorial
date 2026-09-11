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
                        Magazine, Head, Carrier)

RESULTS = []
VERBOSE = "-v" in sys.argv


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))


def _foot_gap(fx, geom, a, b):
    """Least gap between two racked parts' own footprints in the bench
    plane, mm.  Negative is an overlap.

    A rod's footprint is its slot; a KIT'S IS NOT -- the module, its collar
    and the wound grid hang off the boss's root inside a printed pocket,
    `Carrier.nest_over` past it and across it.  Every mount slot lies along
    one world axis or the other, so both footprints are axis-aligned boxes
    and the gap is the larger of the two separations.
    """
    def box(rod):
        s = fx.slot_of(rod.index)
        p0 = np.asarray(s.p0, float)[:2]
        p1 = np.asarray(s.p1, float)[:2]
        L = float(np.linalg.norm(p1 - p0))
        u = (p1 - p0) / L
        n = np.array([-u[1], u[0]])
        if rod.kind == "mcam":
            over, w = Carrier.nest_over(Payload, geom.t.d_diag)
        else:
            over, w = 0.0, rod.r
        mid = 0.5 * (p0 + p1)
        cs = [mid + su * u * (L / 2.0 + (over if su < 0 else 0.0)) + sn * n * w
              for su in (-1.0, 1.0) for sn in (-1.0, 1.0)]
        cs = np.array(cs)
        return cs.min(axis=0), cs.max(axis=0)

    lo_a, hi_a = box(a)
    lo_b, hi_b = box(b)
    return float(max(np.maximum(lo_b - hi_a, lo_a - hi_b)))


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
        # THE BORE IS ON THE CROSSING'S DIAGONAL, and it has to be: a ring
        # can only orbit what lies along its axis, and put on either rod the
        # OTHER rod lies in the ring's own plane.  Measured before this was
        # understood: one crossing of four wound, the rest at zero turns.
        n = kit.wind_axis(0)
        t = np.array([-n[1], n[0], 0.0])
        worst_ax = min(abs(float(r.axis @ n)) for r in kit.rods)
        check(tag + "the winder's bore lies on the crossing's DIAGONAL, so BOTH "
              "rods pierce the ring's plane at the crossing and neither lies in it",
              worst_ax > 0.5 and abs(float(n @ np.array([0.0, 0.0, 1.0]))) < 1e-9,
              "each rod is %.0f degrees to the ring's plane"
              % np.degrees(np.arcsin(worst_ax)))
        # ...and at the bore's own radius the rods are long clear of the ring
        s_bore = StationB.ring_r_in() / abs(float(kit.rods[0].axis @ t))
        check(tag + "...and where a rod reaches the bore's radius it is already "
              "clear of the ring's own thickness",
              s_bore * abs(float(kit.rods[0].axis @ n)) > StationB.RING_W / 2.0,
              "%.1f mm off the plane at the bore, against a ring %.1f thick"
              % (s_bore * abs(float(kit.rods[0].axis @ n)), StationB.RING_W))

        # ------------------------------------------- IT IS WOUND ON A JIG
        # A RING CANNOT ORBIT A JOINT WITH A PLATE UNDER IT.  This is the
        # whole reason there is a jig, and it is geometry, not preference.
        lx = 0.5 * (kit.l0 + kit.l1)
        drop = StationB.ring_r_out() - (lx - kit.seat[1])
        check(tag + "the winder could not wind a crossing where the crossing "
              "finally sits: it reaches below the collar it is bonded to",
              drop > 0.0,
              "the rim is %.1f mm under the collar's own face" % drop)
        check(tag + "so the frame is built on a JIG, and every post of it stands "
              "clear of the winder's swept solid",
              kit.post_clearance() > 0.0 and len(kit.jig_posts()) == 4,
              "least clearance %.2f mm on %d posts"
              % (kit.post_clearance(), len(kit.jig_posts())))
        check(tag + "...and the winder's lowest reach clears the jig's own base "
              "plate",
              kit.jig_floor_clear() > 0.0,
              "%.2f mm" % kit.jig_floor_clear())
        check(tag + "no post stands under layer 1: it rests on layer 0 at all four "
              "crossings, exactly as it does in the finished mount, which leaves "
              "the two rods with nothing over them free for the jaws",
              all(min(abs(float(p[1]) - float(r.p0[1]))
                      for r in kit.jig_rods() if r.layer == 0) < 1e-6
                  for p in kit.jig_posts()),
              "%s" % [tuple(round(v, 1) for v in p) for p in kit.jig_posts()])
        # THE MOUTH HAS TO BE PARKED, and that is a measurement, not a habit:
        # left where fourteen turns stopped it, the ring came down on the next
        # crossing through its own rim and drove a rod 32 degrees round.
        check(tag + "the winder's mouth is wider than the crossing needs to enter "
              "it, so there is a tolerance to park to at all",
              kit.park_tol() > 0.0,
              "mouth %.0f deg, the crossing wants %.1f, so %.1f deg of tolerance"
              % (StationB.RING_GAP, kit.entry_half_angle(), kit.park_tol()))

        # ------------------------------------------- IT CAN BE HANDED OVER
        under, along = kit.frame_grip_margin()
        check(tag + "the finished frame is lifted by a LAYER-1 rod -- on layer 0 "
              "the jaws reach below the collar they are setting it down on",
              kit.frame_rod().layer == 1 and under > 0.0,
              "%.2f mm under the pads on layer 1; layer 0 would be %.2f"
              % (under, under - (kit.l1 - kit.l0)))
        check(tag + "...at its middle, which is the one span on the frame with "
              "nothing under it and nothing over it",
              along > 0.0 and abs(float(kit.frame_rod().mid[0])) > 1e-9,
              "the pads stop %.2f mm short of the nearest crossing" % along)
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

    # =============================================== WHAT A HANDS OVER
    # THE HAND-OVER IS THE OTHER HALF OF THE CONTRACT.  B makes the kit; the
    # cell must not also try to make it, and must not plan against a part
    # different from the one that arrives.  Both were true at once for a
    # while: the cell laid its own four grid rods AND drew the camera with a
    # solid plate where the grid goes.
    for T in (structure.TRUSS_300, structure.TRUSS_1M):
        tag = "[%s] " % T.name
        g = geometry.TrussGeometry(T)
        M = mount.solve(g, d_strut=T.d_diag,
                        standoff_mm=mount.fov_standoff(g, d_strut=T.d_diag))
        g.attach_mount(M)
        kit = stationb.Kit(d_rod=T.d_diag)
        kinds = {r.kind for r in g.mount_rods}
        check(tag + "the cell lays no grid rod: the tic-tac-toe arrives wound on "
              "the kit, because this cell cannot wind it",
              "mgrid" not in kinds and "mcollar" not in kinds
              and len(M.by_kind("grid")) == 8,
              "the cell lays %s; the mount still HAS %d grid rods, made at B"
              % (sorted(kinds), len(M.by_kind("grid"))))
        # ...and every strut still lands on a crossing, which is now a point
        # on a part the cell only fetches
        xh, look, up = mount.landing_frame(g)
        c = np.asarray(M.payload[0][0], float)
        cross = list(kit.crossings())
        lands = []
        for r in M.of(0):
            if r.kind != "strut":
                continue
            v = np.asarray(r.p1, float) - c
            lands.append(np.array([float(v @ xh), float(v @ up), float(v @ look)]))
        check(tag + "...and all six struts still land on crossings that arrive with "
              "the kit, so nothing is bonded to air",
              len(lands) == 6
              and max(min(float(np.linalg.norm(p - q)) for q in cross)
                      for p in lands) < 1e-6,
              "%d landings on %d crossings" % (len(lands), len(cross)))
        # THE KIT IS BIGGER THAN THE MODULE and everything sized off the
        # module has to be re-sized off the kit
        kx, ku = Carrier.kit_half(Payload, T.d_diag)
        # THE RULE THE BOSS IMPLEMENTS, checked as the rule and not as the
        # number: with the gripper down on the boss, nothing of the kit is
        # inside the head's static envelope.  Measured against the plate
        # alone the boss is 6 mm short of what the part that arrives needs,
        # and the head lands on the grid before the jaws reach the boss.
        from truss.spec import race_r_out
        bl = Carrier.boss_l(Payload, T.d_diag)
        H = Head.GRIP_STROKE - Head.TIP_PARK          # ring centre over the grip
        R = race_r_out() + Process.SEAT_CLEAR
        corner = sqrt((bl / 2.0) ** 2 + (H - ku) ** 2)
        check(tag + "the carrier's boss reaches until the WHOLE kit is outside the "
              "head's envelope -- the grid overhangs the collar, so a boss sized "
              "against the plate is short",
              corner >= R - 1e-6 and ku > Bracket.plate_half(Payload, T.d_diag)[1],
              "the kit's nearest corner is %.2f mm from the ring's axis against a "
              "rim at %.2f; boss %.1f mm for a kit %.2f to its edge (the plate is "
              "%.2f)" % (corner, R, bl, ku,
                         Bracket.plate_half(Payload, T.d_diag)[1]))
        fx = fixture.Fixture(g)
        cam = [r for r in g.mount_rods if r.kind == "mcam"]
        others = [r for r in g.mount_rods if r.kind != "mcam"]
        # THE KIT'S FOOTPRINT AGAINST ITS NEIGHBOURS', not a pitch in x.
        # This compared two slot origins along the truss axis, which said
        # something only while every slot lay across it at a fixed pitch;
        # the magazine packs the mount's parts along x on the cage's flanks
        # now, and the same 13 mm that is plenty along a kit's own axis is
        # nothing across it.  What the rule was always about is whether the
        # nest can reach a neighbour, and that is a footprint.
        worst, who = 1e9, -1
        for c in cam:
            for r in others:
                gp_ = _foot_gap(fx, g, c, r)
                if gp_ < worst:
                    worst, who = gp_, r.index
        check(tag + "...and the magazine gives that nest its own room: the kit's "
              "footprint clears every other racked part, which a slot pitch cannot "
              "say once the parts stop lying parallel",
              worst >= 0.0,
              "%.1f mm to part %d, kit half-width %.1f" % (worst, who, kx))

    # ------------------------------------------- IT IS IN THE SCENE
    import mujoco

    # THE KIT HAS TO FIT ITS OWN RECEPTACLE, measured on the built scene and
    # not on the drawing.  Sized against the module, the nest's front wall
    # cleared the part that actually arrives by 0.10 mm -- an accident, not
    # a clearance, and nothing in the suite was looking at it.
    T = structure.TRUSS_300
    g = geometry.TrussGeometry(T)
    M = mount.solve(g, d_strut=T.d_diag,
                    standoff_mm=mount.fov_standoff(g, d_strut=T.d_diag))
    g.attach_mount(M)
    fx = fixture.Fixture(g)
    mk = mujoco.MjModel.from_xml_string(mjcf.scene_cell(g, fx, n_drops=80))
    dk = mujoco.MjData(mk)
    mujoco.mj_forward(mk, dk)
    cam = [r for r in g.mount_rods if r.kind == "mcam"][0]
    gid = lambda n: mujoco.mj_name2id(mk, mujoco.mjtObj.mjOBJ_GEOM, n)
    kit_g = [i for i in range(mk.ngeom)
             if (mujoco.mj_id2name(mk, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
             .startswith("rod%d_" % cam.index)]
    nest_g = [i for i in range(mk.ngeom)
              if (mujoco.mj_id2name(mk, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
              .startswith(("nest%d" % cam.index, "nestw%d" % cam.index))]
    # THE NEST BEARS ON THE BOARD AND ON NOTHING ELSE.  The board's
    # underside and its back face are what a kitting nest is allowed to
    # touch; the collar, the grid and the boss are the product and the
    # nest has to stand off them.  Measured pairwise on the built scene.
    nm = lambda i: mujoco.mj_id2name(mk, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
    bearing, clear_ = 1e9, 1e9
    pair_b, pair_c = ("", ""), ("", "")
    for a in kit_g:
        for b in nest_g:
            e = mujoco.mj_geomDistance(mk, dk, a, b, 0.05, None) * 1000.0
            if nm(a).endswith("_b0"):
                if e < bearing:
                    bearing, pair_b = e, (nm(a), nm(b))
            elif e < clear_:
                clear_, pair_c = e, (nm(a), nm(b))
    check("the nest bears on the board's own faces and does not dig into it",
          bearing >= -1e-6,
          "%.2f mm at %s / %s" % (bearing, pair_b[0], pair_b[1]))
    check("...and it stands off everything else on the kit -- the collar hangs "
          "2.5 mm below the board on every side and the grid 6.5 below that",
          clear_ >= Process.SEAT_CLEAR - 1e-6,
          "%.2f mm at %s / %s, against a process clearance of %.2f"
          % (clear_, pair_c[0], pair_c[1], Process.SEAT_CLEAR))
    check("...and the kit is DRAWN as the part that arrives: a collar of bands, a "
          "ring and ribs, and four grid rods -- not a slab",
          len([i for i in kit_g
               if "_q" in (mujoco.mj_id2name(mk, mujoco.mjtObj.mjOBJ_GEOM, i) or "")]) == 4
          and len([i for i in kit_g
                   if "_p" in (mujoco.mj_id2name(mk, mujoco.mjtObj.mjOBJ_GEOM, i) or "")])
          == len(mount.collar_segments(Payload, T.d_diag)),
          "%d geoms in the kit" % len(kit_g))
    T = structure.TRUSS_300
    g = geometry.TrussGeometry(T)
    M = mount.solve(g, d_strut=T.d_diag,
                    standoff_mm=mount.fov_standoff(g, d_strut=T.d_diag))
    g.attach_mount(M)
    fx = fixture.Fixture(g)
    kit = stationb.Kit(d_rod=T.d_diag)
    m0 = mujoco.MjModel.from_xml_string(mjcf.scene_cell(g, fx, n_drops=80))
    m1 = mujoco.MjModel.from_xml_string(mjcf.scene_cell(g, fx, kit=kit, n_drops=80))
    b_joints = [n for n in ("bx", "by", "bz", "bg", "bv", "bw", "bring",
                            "bf_l", "bf_r")
                if mujoco.mj_name2id(m1, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0]
    check("station B is a SEPARATE SET OF ACTUATORS in the same scene -- it shares "
          "the floor with the cell and nothing else",
          m1.nu == m0.nu + 9 and len(b_joints) == 9,
          "%d actuators against the cell's %d; joints %s"
          % (m1.nu, m0.nu, ",".join(b_joints)))
    # THE COLLAR IS AN OPEN FRAME AND THE MODEL HAS TO SAY SO.  Drawn as a
    # plate it covered the lens; drawn with its ring at the band's width it
    # would not pass over the housing and sat 2 mm proud of its seat.
    segs = mount.collar_segments(Payload, T.d_diag)
    ax, au = [v / 2.0 for v in Bracket.aperture(Payload)]
    inner = min(min(abs(a[0]), abs(b[0])) - w / 2.0 for a, b, w in segs
                if abs(a[1] - b[1]) > 1e-9 and abs(a[0]) > 1e-9)
    check("the collar's ring clears the lens housing it goes over",
          inner <= ax + 1e-9 and ax > Payload.CASE[0] / 2.0,
          "the ring's inner edge is at %.2f, the aperture at %.2f, the case at %.2f"
          % (inner, ax, Payload.CASE[0] / 2.0))
    check("...and the vacuum head lands on that ring and nowhere else -- it is the "
          "only metal at the plate's own centre, and it clears the housing",
          all(abs(c[0]) - h[0] >= Payload.CASE[0] / 2.0 - 1e-9
              or abs(c[1]) - h[1] >= Payload.CASE[1] / 2.0 - 1e-9
              for c, h in StationB.vac_pads(Payload)),
          "pads at %s" % [tuple(round(v, 2) for v in c)
                          for c, _h in StationB.vac_pads(Payload)])
    check("...and it can lift what it is asked to lift",
          StationB.vac_hold_N(Payload)
          > 10.0 * Bracket.mass(Payload, T.d_diag) * 9.81e-3,
          "%.2f N against a collar of %.3f N"
          % (StationB.vac_hold_N(Payload),
             Bracket.mass(Payload, T.d_diag) * 9.81e-3))
    # EVERY TOOL IS SEATED ABOVE THE RING, because they share its carriage
    kitc = stationb.Kit(d_rod=T.d_diag)
    check("every tool on B's carriage is seated clear of the tallest thing the "
          "winder straddles -- they ride with it, and when it winds, its centre "
          "is ON the crossing",
          StationB.tool_lift(kitc) > kitc.work_above_crossing(),
          "%.2f mm of seat over %.2f mm of work"
          % (StationB.tool_lift(kitc), kitc.work_above_crossing()))
    check("...and the carriage flies at the height the WINDER needs, not the "
          "tool's -- the ring does not retract",
          kitc.cruise_z() - StationB.ring_r_out() > kitc.table_top(),
          "cruise %.1f, rim reaches %.1f, the table stands %.1f"
          % (kitc.cruise_z(), kitc.cruise_z() - StationB.ring_r_out(),
             kitc.table_top()))
    check("...and every tool's stroke covers the drop from there to the lowest "
          "thing it has to touch",
          StationB.tool_stroke(kitc)
          >= kitc.cruise_z() + StationB.tool_lift(kitc) - kitc.lowest_tip(),
          "%.1f mm of stroke for a %.1f mm drop"
          % (StationB.tool_stroke(kitc),
             kitc.cruise_z() + StationB.tool_lift(kitc) - kitc.lowest_tip()))
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
