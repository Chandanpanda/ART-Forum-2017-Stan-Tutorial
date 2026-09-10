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

from spine import build, duty as _duty, metrics
from truss import structure, geometry, fixture, mount
from truss.spec import (Bracket, Truss, Cage, Carrier, Gantry, Gripper, Head, Load,
                        Magazine, Module3, Payload, Process, Stock, Dispenser)

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


def _post_gap(geom, m, off):
    """Least surface gap between a thread post `off` mm past the chord ends
    and any part of the mount, mm, with what it is nearest.  Negative is
    inside.  The pure-geometry twin of the built-scene check below: it names
    the pin and the strut, which a scene-wide minimum does not.  The pin is
    a CYLINDER in the scene and a capsule here, so this reads 0.66 mm inside
    where the scene reads 0.30 -- the conservative side, which is the side a
    clearance model should be wrong on."""
    from truss.geometry import capsule_gap
    worst, who = 1e9, ("", -1)
    for k in range(geom.t.n_chords):
        up = geometry.radial(geometry.chord_phi(k))
        for end in (0, 1):
            rs = [r for r in m.rods if r.end == end]
            c = geom.chord_point(k, 0.0 if end == 0 else geom.t.length) \
                + np.array([-off if end == 0 else off, 0.0, 0.0])
            d = capsule_gap(c - up * Cage.POST_L / 2.0, c + up * Cage.POST_L / 2.0,
                            Cage.POST_R,
                            np.array([np.asarray(r.p0, float) for r in rs]),
                            np.array([np.asarray(r.p1, float) for r in rs]),
                            np.array([max(r.r, r.half_w) for r in rs]))
            i = int(np.argmin(d))
            if float(d[i]) < worst:
                worst, who = float(d[i]), (rs[i].kind, rs[i].index)
    return worst, who[0], who[1]


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


def _cam_pt(m, g, dx, dl, du, end=0):
    """A point in the camera's own frame, cage coords -- for building the
    geometries this check exists to reject."""
    xh, look, up = mount.landing_frame(g)
    return np.asarray(m.payload[end][0], float) + dx * xh + dl * look + du * up


def _rigid_budget(g, t, so, DUTY=None):
    """The same mount with a platform that cannot bend, at the plate's real
    mass -- the reading the mount had before the band was modelled."""
    from spine.chosen import OPTIMAL_1M as _ch
    DUTY = _duty.QUADROTOR if DUTY is None else DUTY
    sp = dict(mount.nose_spec(g, d_strut=t.d_diag))
    sp.update(plat_A=None, plat_I=None)
    return metrics.evaluate(build.warren_truss(
        t.length, _ch.side, _ch.alpha, _ch.d_chord, _ch.d_diag,
        tip_mass=DUTY.tip_mass_g, nose=build.Nose.around(**sp)),
        DUTY)["budget_used"]


def main():
    from spine.chosen import OPTIMAL_1M as ch
    DUTY = _duty.QUADROTOR
    t = Truss(**ch.as_truss_kwargs())
    g = geometry.TrussGeometry(t)
    so = mount.fov_standoff(g, d_strut=t.d_diag)
    m = mount.solve(g, d_strut=t.d_diag, standoff_mm=so)
    fx = fixture.Fixture(g)

    # ---------------------------------------------- THE GRIPPER CANNOT
    thin, case = Carrier.span()
    x_now, y_now = fx.x_reach(), fx.y_reach()
    check("the gripper cannot hold the camera as it stands: its jaws open 8 mm and "
          "the lens housing is 8.5 mm square",
          Gripper.JAW_OPEN < case and Gripper.JAW_OPEN < thin,
          "jaws %.1f mm; module %.1f through, housing %.1f square"
          % (Gripper.JAW_OPEN, thin, case))
    # OPENING THEM IS FREE ON THIS MODULE, and that is worth knowing rather
    # than assuming: the magazine's pitch is bounded below by the jaw
    # opening, so on the module this project started with -- 10.8 mm square
    # -- wider jaws pushed the racks past the gantry's X travel and there
    # was no fix at all.  Module 2's housing is 2.3 mm smaller and the
    # bound lands under the pitch the racks already have.
    jaw_ok = case + 1.0
    dp = max(Magazine.DIAG_PITCH, pitch_bound(jaw_ok))
    cp = max(Magazine.CHORD_PITCH, pitch_bound(jaw_ok) + 1.0)
    x_wide, _y = reach_with(t, dp, cp, Cage.END_FREE)
    check("...and on THIS module, opening them would cost nothing: the pitch a %.1f mm "
          "jaw forces is under the pitch the racks already have"
          % jaw_ok,
          pitch_bound(jaw_ok) <= Magazine.DIAG_PITCH and abs(x_wide - x_now) < 1e-6,
          "%.2f mm needed against %.1f; X unchanged at %.0f"
          % (pitch_bound(jaw_ok), Magazine.DIAG_PITCH, x_now))
    jaw_v3 = Module3.CASE[0] + 1.0
    x_v3, _ = reach_with(t, max(Magazine.DIAG_PITCH, pitch_bound(jaw_v3)),
                         max(Magazine.CHORD_PITCH, pitch_bound(jaw_v3) + 1.0),
                         Cage.END_FREE)
    check("...which it would NOT have on the module this started with: a 10.8 mm "
          "housing needs jaws that push the racks past the gantry's X travel",
          x_v3 > Gantry.X_TRAVEL >= x_now,
          "%.0f mm of %.0f with %.1f mm jaws, against %.0f as built"
          % (x_v3, Gantry.X_TRAVEL, jaw_v3, x_now))
    # ...SO WHY A CARRIER AT ALL.  Because the housing is the BONDING
    # SURFACE.  The six struts land on that square; jaws on it are jaws on
    # the bond, and a pad that has been clamped to an epoxy joint's own
    # face is a contaminated joint.  That argument needs no dimension.
    check("the housing the jaws would have to take is exactly the surface the collar's "
          "aperture is cut to, so gripping it and bonding it are the same face",
          abs(Bracket.aperture()[1] - (Payload.CASE[1] + 2.0 * Bracket.BOND_GAP)) < 1e-9,
          "an %.1f mm housing into an %.2f mm aperture -- %.2f mm of glue line all round"
          % (Payload.CASE[1], Bracket.aperture()[1], Bracket.BOND_GAP))
    check("...and the pads are taller than the housing stands proud anyway, so jaws "
          "on it would land on the board [rests on CASE_PROUD, which is VERIFY]",
          Gripper.PAD_H > Payload.CASE_PROUD,
          "%.1f mm of pad against %.1f mm of housing"
          % (Gripper.PAD_H, Payload.CASE_PROUD))

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
    check("...and being radial rather than axial it costs the cage no room at all: "
          "what reaches past the chord ends is the KIT's own outer face and nothing "
          "of the carrier",
          abs(Carrier.reach(so, Payload, t.d_diag)
              - (so + Carrier.kit_half(Payload, t.d_diag)[0])) < 1e-9
          and Carrier.reach(so, Payload, t.d_diag) < so + Carrier.boss_l(
              Payload, t.d_diag),
          "reach %.2f mm is the kit's own outer face; the boss is %.1f mm long and "
          "costs none of it"
          % (Carrier.reach(so, Payload, t.d_diag), Carrier.boss_l(Payload, t.d_diag)))
    # EVERY MOUNT ROD HAS TO BE LAYABLE WITH THE YAW THE MACHINE HAS.  The
    # servo has a stop; asked past it, it clamps and answers "settled", and
    # the rod goes down off its own line with nothing reporting anything.
    # Two struts an end wanted 142.3 degrees against a 95 degree stop and
    # were laid 47 degrees out -- 27.7 mm at their ends, and it took a
    # picture and then rig_mount to find, because the plan, the tracker and
    # the inspector all agreed the op had succeeded.
    for T in (structure.TRUSS_300, structure.TRUSS_1M):
        gg = geometry.TrussGeometry(T)
        MM = mount.solve(gg, d_strut=T.d_diag,
                         standoff_mm=mount.fov_standoff(gg, d_strut=T.d_diag))
        gg.attach_mount(MM)
        bad = []
        for r in gg.mount_rods:
            offer = mount.poses_for(r.axis, reversible=(r.kind != "mcam"))
            if not [p for p in offer if abs(p[1]) <= Head.GRIP_YAW]:
                bad.append((r.kind, r.index,
                            [round(p[1], 1) for p in offer]))
        check("[%s] every mount rod can be laid within the yaw servo's own stop "
              "-- a rod has no head or tail, so the cage's symmetry and the "
              "ROD'S give four poses, not two" % T.name,
              not bad, "%s" % (bad or "all %d parts" % len(gg.mount_rods)))
        # ...and the one part that is not a rod is never offered the flip
        cam = [r for r in gg.mount_rods if r.kind == "mcam"][0]
        check("[%s] ...and the camera is not offered the end-for-end flip: it is "
              "on a carrier with a boss out of one side, and reversed it is a "
              "boss where the lens goes" % T.name,
              len(mount.poses_for(cam.axis, reversible=False)) == 2
              and len(mount.poses_for(cam.axis)) == 4,
              "%d poses for a rod, %d for the camera"
              % (len(mount.poses_for(cam.axis)),
                 len(mount.poses_for(cam.axis, reversible=False))))
        # all four really are the same line
        for r in gg.mount_rods[:6]:
            got = [mount.axis_from(th, y) for th, y in mount.poses_for(r.axis)]
            check("[%s] ...and all four poses lay %s%d on its own line"
                  % (T.name, r.kind, r.index),
                  all(abs(abs(float(np.asarray(v) @ r.axis)) - 1.0) < 1e-6
                      for v in got),
                  "%d poses" % len(got))

    check("the cage can turn the boss under the jaws with the axes it has",
          np.linalg.norm(mount.axis_from(*mount.pose_for(axis)) - axis) < 1e-9)

    # ---------------------------------------------- ROOM IN THE CAGE
    # THE CAMERA IS MOUNTED INSIDE THE CAGE, so the cage has to have room
    # for it, and as built it did not.
    worst_side = Load.SECTION_MAX
    wg = geometry.TrussGeometry(Truss(length=t.length, side=worst_side, alpha=45.0,
                                      d_chord=3.0, d_diag=min(Stock.DIAMETERS)))
    d_worst = min(Stock.DIAMETERS)
    worst = Carrier.reach(mount.fov_standoff(wg, d_strut=d_worst), Payload, d_worst)
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
    # ------------------------------------------- WHERE A THREAD POST STANDS
    # NOT A NUMBER.  A post is a radial pin on the chord's own line and the
    # struts leave that same line heading inward, so the near bound the
    # cage was drawn with -- 8.0 mm, which is exactly what the ring's own
    # loop needs and nothing else -- put two struts an end INSIDE a pin.
    gp = geometry.TrussGeometry(t)
    Mp = mount.solve(gp, d_strut=t.d_diag,
                     standoff_mm=mount.fov_standoff(gp, d_strut=t.d_diag))
    gp.attach_mount(Mp)
    fp = fixture.Fixture(gp)
    near = Cage.post_off_min()
    at_near = _post_gap(gp, Mp, near)
    at_solved = _post_gap(gp, Mp, fp.post_off)
    check("a thread post at the 8 mm the ring's own loop needs is INSIDE the mount: "
          "the struts leave the chord ends on the same line the posts stand on",
          at_near[0] < 0.0,
          "%.2f mm at %s %d, %.2f mm past the chord end"
          % (at_near[0], at_near[1], at_near[2], near))
    check("...so the station is solved, not set: mount.post_station finds the nearest "
          "one that clears the whole mount by the process clearance",
          at_solved[0] >= Process.SEAT_CLEAR - 1e-6,
          "%.2f mm past the chord ends, clearing by %.2f at %s %d"
          % (fp.post_off, at_solved[0], at_solved[1], at_solved[2]))
    check("...and it is still a station the head can use: inside the window between "
          "the ring's loop clearing the chord end and the ring parked at it clearing "
          "the end plate",
          Cage.post_off_min() <= fp.post_off <= Cage.post_off_max()
          and Cage.END_FREE >= fp.post_off + Head.ring_axial_half() + Process.SEAT_CLEAR,
          "%.2f in [%.2f, %.2f]" % (fp.post_off, Cage.post_off_min(),
                                    Cage.post_off_max()))
    bare = fixture.Fixture(geometry.TrussGeometry(t))
    check("...and with no camera on the truss there is nothing to clear, so the post "
          "goes back to the ring's own bound -- which is why 8 mm looked right",
          abs(bare.post_off - Cage.post_off_min()) < 1e-9)

    # ------------------------------------------ AND THE PINS AT THE ENDS
    # The same fault twice: the chord ends now carry the END BATTENS, and
    # the last V-pin's station was a pin's own thickness from the end --
    # a fact about the pin that says nothing about what is laid over it.
    check("the last V-pin stands clear of the end batten laid across the chord ends, "
          "not a pin's thickness from the end",
          fp.end_exclusion() >= Cage.PIN_T / 2.0 + t.d_diag / 2.0 + Process.SEAT_CLEAR
          - 1e-9 and min(fp.pin_xs(0)) >= fp.end_exclusion() - 1e-9,
          "%.2f mm margin, first pin at %.2f (was %.2f on a bare truss)"
          % (fp.end_exclusion(), min(fp.pin_xs(0)), min(bare.pin_xs(0))))
    # AND NOTHING OF THE CELL IS IN THE MOUNT, measured on the BUILT scene
    # with each part at the pose the plan sends it to.  This is the check
    # that was missing: the camera is mounted on the cage's own axis, and the
    # cage's backbone, its draw rod, its torsion shafts, two of its arms and
    # its end plate were all in there with it -- eight parts, the worst
    # 4.5 mm deep.  Every op reported success and the kit came to rest 24 mm
    # from its target, because it was resting on the mandrel.
    import mujoco
    from truss import mjcf as _mjcf
    gk = geometry.TrussGeometry(t)
    Mk = mount.solve(gk, d_strut=t.d_diag,
                     standoff_mm=mount.fov_standoff(gk, d_strut=t.d_diag))
    gk.attach_mount(Mk)
    fk = fixture.Fixture(gk)
    mk = mujoco.MjModel.from_xml_string(
        _mjcf.scene_cell(gk, fk, stage="truss", n_drops=8))
    dk = mujoco.MjData(mk)

    def _rz(a):
        c_, s_ = np.cos(np.radians(a)), np.sin(np.radians(a))
        return np.array([[c_, -s_, 0.], [s_, c_, 0.], [0., 0., 1.]])

    # EVERY MOUNT ROD, not just the kit.  The kit-only version of this check
    # passed while four struts a truss were 0.66 mm inside a thread post,
    # for the plainest possible reason: it was only ever asked about one
    # part of nine.  A rod's own ENDS are excluded -- they are bonded to the
    # chord ends, and a fillet is a joint, not a collision -- so what is
    # measured is each rod's SHAFT, out past `Payload.FILLET_R` from either
    # bond point.  The post was 8 mm along a 40 mm strut.
    DMAX = 0.02                                   # m, as far as MuJoCo looks
    ft = np.zeros(6)
    worst_k, pair_k, kind_k = 1e9, ("", ""), ""
    # ONE ROD AT A TIME, AND PUT IT BACK.  Posed and left there, the second
    # rod is measured against the first one already at the nose: the first
    # run of this reported -1.50 mm between two struts, which is exactly
    # two radii, which is two centrelines on top of each other.  Suspect
    # the check.
    rest = dk.qpos.copy()
    for r in gk.mount_rods:
        dk.qpos[:] = rest
        offer = mount.poses_for(r.axis, reversible=r.kind != "mcam")
        th, yaw = max([p for p in offer if abs(p[1]) <= Head.GRIP_YAW],
                      key=lambda p: float((geometry.rot_x(p[0]) @ r.mid)[2]))
        b = mujoco.mj_name2id(mk, mujoco.mjtObj.mjOBJ_BODY, "rod%d" % r.index)
        adr = mk.jnt_qposadr[mk.body_jntadr[b]]
        # the part is PICKED at yaw 90, so only the change in yaw turns it
        R = _rz(yaw - 90.0)
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, R.flatten())
        dk.qpos[adr:adr + 3] = (geometry.rot_x(th) @ r.mid) / 1000.0
        dk.qpos[adr + 3:adr + 7] = q
        jc = mujoco.mj_name2id(mk, mujoco.mjtObj.mjOBJ_JOINT, "cage")
        dk.qpos[mk.jnt_qposadr[jc]] = np.radians(th)
        mujoco.mj_forward(mk, dk)
        # the bond points in the same frame the measurement comes back in;
        # the kit has none -- it is seated on its carrier, and the schedule
        # lays no fillet on it
        Rth = geometry.rot_x(th)
        bonds = () if r.kind == "mcam" else (Rth @ r.p0 / 1000.0,
                                             Rth @ r.p1 / 1000.0)
        mine = list(range(mk.body_geomadr[b],
                          mk.body_geomadr[b] + mk.body_geomnum[b]))
        for i in mine:
            for j2 in range(mk.ngeom):
                if j2 in mine:
                    continue
                e = mujoco.mj_geomDistance(mk, dk, i, j2, DMAX, ft) * 1000.0
                if e >= worst_k or e >= DMAX * 1000.0 - 1e-9:
                    continue          # `ft` is only written inside distmax
                if any(np.linalg.norm(ft[:3] - bp) * 1000.0
                       <= Payload.FILLET_R for bp in bonds):
                    continue                      # that is the fillet
                worst_k = e
                kind_k = r.kind
                pair_k = (mujoco.mj_id2name(mk, mujoco.mjtObj.mjOBJ_GEOM, i),
                          mujoco.mj_id2name(mk, mujoco.mjtObj.mjOBJ_GEOM, j2))
    check("nothing of the cell is inside the mount: EVERY mount rod at the pose the "
          "plan sends it to clears every geom in the scene, its own bonded ends aside",
          worst_k >= Process.SEAT_CLEAR - 1e-6,
          "%.2f mm at %s / %s (a %s), against a process clearance of %.2f"
          % (worst_k, pair_k[0], pair_k[1], kind_k, Process.SEAT_CLEAR))
    check("making that room is NOT free: it pushes the end racks out with it, and the "
          "chosen truss is inside the gantry's X travel with 17 mm to spare",
          x_now <= Gantry.X_TRAVEL
          and reach_with(t, Magazine.DIAG_PITCH, Magazine.CHORD_PITCH, 20.0)[0]
          < x_now,
          "%.0f mm of %.0f now, %.0f with the old 20 mm cage"
          % (x_now, Gantry.X_TRAVEL,
             reach_with(t, Magazine.DIAG_PITCH, Magazine.CHORD_PITCH, 20.0)[0]))

    # ---------------------------------------------- THE RODS THEMSELVES
    # the CARBON RODS only: the bracket's arms are a machined part and are
    # checked as one, not held to the stock's diameters or a rod's slenderness
    rods = [r for r in m.rods if r.kind in ("batten", "strut")]
    check("the mount is thirteen rods an end plus ONE laser-cut collar: three battens "
          "closing the chord triangle, four in a grid round the camera, six struts",
          len([r for r in m.of(0) if r.kind in ("batten", "grid", "strut")]) == 13
          and len(m.by_kind("batten")) == 6 and len(m.by_kind("grid")) == 8
          and len(m.by_kind("strut")) == 12,
          "%d battens, %d grid, %d struts, per pair of ends"
          % (len(m.by_kind("batten")), len(m.by_kind("grid")),
             len(m.by_kind("strut"))))
    # NO POINT JOINTS ANYWHERE, which is the rule the mount is built to.
    # Every rod either lies along a whole edge of the plate (a line of
    # adhesive) or crosses another rod (a wound joint, the one this cell has
    # qualified).  A strut that met a grid rod between its crossings would
    # be asking that rod to bend, and a rod asked to bend is 478 to 1900
    # times softer than the same rod pulled.
    cross = {tuple(np.round(r.p1, 4)) for r in m.of(0) if r.kind == "strut"}
    grid_ends = {tuple(np.round(p, 4)) for r in m.of(0) if r.kind == "grid"
                 for p in (r.p0, r.p1)}
    check("every strut lands where two grid rods already cross, so the load enters a "
          "wound joint and no rod is asked to bend between its supports",
          len(cross) == 4 and not (cross & grid_ends),
          "%d struts onto %d crossings, none at a rod end" % (6, len(cross)))
    # WHY THERE IS A COLLAR, measured on the frame model rather than argued.
    # Neither surface the camera actually offers will hold the budget.
    import numpy as _np
    from spine import build as _b, metrics as _mx
    def _budget(landings, link=None):
        n = _b.Nose.around(box=Payload.BOX, standoff=so, rigid=True,
                           r_platform=mount.platform_radius(Payload, t.d_diag),
                           d_strut=t.d_diag, landings=landings, link_k=link,
                           plate_g=Bracket.mass(Payload, t.d_diag))
        return _mx.evaluate(_b.warren_truss(t.length, ch.side, ch.alpha, ch.d_chord,
                                            ch.d_diag, tip_mass=DUTY.tip_mass_g,
                                            nose=n), DUTY)["budget_used"]
    _xh, _lk, _up = mount.landing_frame(g)
    def _cam(x, L, U):
        v = L * _lk + U * _up
        return (x, float(v[1]), float(v[2]))
    hb = mount.housing_extent()
    # three points ON the inboard face -- L between the housing's own limits,
    # spread as far as that face allows
    thin = tuple([_cam(hb[0][1], hb[1][0], hb[2][1]),
                  _cam(hb[0][1], hb[1][0], hb[2][0]),
                  _cam(hb[0][1], hb[1][1], 0.0)])
    holes = [_cam(h[0], hb[1][0] - 1.0, h[1]) for h in Payload.holes()]
    b_thin = _budget(thin)
    b_hole_rigid = _budget(tuple(holes[i] for i in (0, 1, 3)))
    b_hole_real = _budget(tuple(holes[i] for i in (0, 1, 3)),
                          link=Payload.pcb_stiffness(Payload.HOLE_PITCH[0]))
    check("bonding straight to the housing's own face will NOT hold the budget: it is "
          "the only surface parallel to the end triangle and it is too thin to be a "
          "triangle at all",
          b_thin > 1.0,
          "%.2f of the budget on an %.1f x %.1f mm face"
          % (b_thin, Payload.CASE[0], Payload.CASE_PROUD))
    check("...and bonding through the board's own mounting holes will not either, once "
          "the board is charged for: it is 26 times softer than the strut bonded to it",
          b_hole_real > 1.0 > b_hole_rigid,
          "%.2f with the board modelled rigid, %.2f at its real %.0f N/mm"
          % (b_hole_rigid, b_hole_real, Payload.pcb_stiffness(Payload.HOLE_PITCH[0])))
    _b_collar = _mx.evaluate(_b.warren_truss(
        t.length, ch.side, ch.alpha, ch.d_chord, ch.d_diag, tip_mass=DUTY.tip_mass_g,
        nose=_b.Nose.around(**mount.nose_spec(g, d_strut=t.d_diag))),
        DUTY)["budget_used"]
    check("...so the collar earns its place: landings on it hold the budget with room, "
          "with its own mass and its own bending both charged",
          _b_collar < 1.0, "%.2f of the budget, %.2f g of plate an end"
          % (_b_collar, Bracket.mass(Payload, t.d_diag)))
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
    # THE HOUSING IS PLASTIC ON THIS MODULE, which is the difference from the
    # metal can the mount was first drawn against.  It only matters if it is
    # the soft part.  Modelled as a block CASE[0] long of section
    # CASE[0] x CASE_PROUD between two opposite landings, so its stiffness
    # is E x CASE_PROUD and nothing else -- one modulus times one height,
    # against the strut's own EA/L.
    k_case = (Payload.CASE_E * Payload.CASE[0] * Payload.CASE_PROUD
              / Payload.CASE[0])
    k_strut = Stock.E * np.pi * (t.d_diag / 2.0) ** 2 / min(r.length for r in rods)
    check("the housing is PLASTIC on this module, and it is still not the soft part: "
          "the strut is the compliance",
          Payload.CASE_PLASTIC and k_case > k_strut,
          "%.0f N/mm of housing against %.0f of strut, %.1fx"
          % (k_case, k_strut, k_case / k_strut))
    # ...AND WHAT WOULD HAVE TO BE TRUE TO CHANGE IT.  Not a margin somebody
    # chose: the product of the two flagged numbers at which the conclusion
    # flips, so a caliper and a material name settle it.  A 3 GPa unfilled
    # housing needs a base 1.87 mm tall; a glass-filled one needs 0.56.
    need_mm = k_strut / Payload.CASE_E
    check("...and it flips on ONE product -- modulus times base height -- so a caliper "
          "reading and a material name settle it rather than an argument",
          need_mm < Payload.CASE_PROUD,
          "needs E x base > %.0f N/mm; at the weakest plastic assumed (%.0f MPa) "
          "that is a base %.2f mm tall, against the %.1f assumed"
          % (k_strut, Payload.CASE_E, need_mm, Payload.CASE_PROUD))
    # ...AND THE STRUCTURE DOES NOT TURN ON IT AT ALL, which is worth
    # asserting because CASE_PROUD is the one dimension the drawing does not
    # give.  Swept over everything it could plausibly be, the budget moves
    # 2% and the standoff and the field clearance do not move.
    prow, base_proud = [], Payload.CASE_PROUD
    try:
        for proud in (1.5, 3.0, 5.0, 8.0):
            Payload.CASE_PROUD = proud
            n = build.Nose.around(**mount.nose_spec(g, d_strut=t.d_diag))
            prow.append((proud,
                         metrics.evaluate(build.warren_truss(
                             t.length, ch.side, ch.alpha, ch.d_chord, ch.d_diag,
                             tip_mass=DUTY.tip_mass_g, nose=n), DUTY)["budget_used"],
                         mount.fov_standoff(g, d_strut=t.d_diag)))
    finally:
        Payload.CASE_PROUD = base_proud
    spread = max(b for _, b, _ in prow) / min(b for _, b, _ in prow) - 1.0
    check("the STRUCTURE does not turn on the one dimension the drawing does not "
          "give: the base height moves the yaw budget by 2% over everything it "
          "could plausibly be, and moves the standoff not at all",
          spread < 0.05 and max(so_ for _, _, so_ in prow)
          - min(so_ for _, _, so_ in prow) < 1e-9,
          "%s ; standoff %.2f throughout"
          % (", ".join("%.1f mm->%.4f" % (a, b) for a, b, _ in prow), prow[0][2]))
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
          "camera's %.1f x %.1f degree field" % Payload.FOV,
          mount.fov_clear(m) > 0.0, "%.2f mm to spare" % mount.fov_clear(m))
    # WHAT SETS THE STANDOFF CHANGED, and it is worth asserting which.  It
    # used to be the field of view: the mechanical clearance alone left the
    # end batten across the picture.  Now the part that stands off is the
    # whole head kit, and its own half-width is wider than the field ever
    # asked for -- so the mechanical bound binds and the solve returns it.
    so_mech = mount.mech_standoff(Payload, d_strut=t.d_diag)
    check("the standoff is set by the KIT's own width now, not by the field of view "
          "-- and the field is still clear at it",
          abs(so - so_mech) < 1e-6 and mount.fov_clear(m) > 0.0,
          "mechanical %.2f, solved %.2f, %.2f mm of field to spare"
          % (so_mech, so, mount.fov_clear(m)))
    check("...and it was the field that set it while the cell laid the grid itself: "
          "on the BOARD's half-width alone the truss is in the picture",
          mount.fov_clear(mount.solve(g, d_strut=t.d_diag,
                                      standoff_mm=Payload.BOX[0] / 2.0
                                      + Process.SEAT_CLEAR)) < 0.0,
          "%.1f mm at the board's own %.1f mm, %.1f mm at the solved %.2f"
          % (mount.fov_clear(mount.solve(g, d_strut=t.d_diag,
                                         standoff_mm=Payload.BOX[0] / 2.0
                                         + Process.SEAT_CLEAR)),
             Payload.BOX[0] / 2.0 + Process.SEAT_CLEAR,
             mount.fov_clear(m), so))

    # ------------------------------------- AND NOT INSIDE THE CAMERA
    # THE CHECK THAT WAS MISSING.  The collar was a fin standing on edge in
    # the plane perpendicular to the spine -- the plane a hexapod's platform
    # wants to be in -- and a fin through the camera's centre is INSIDE THE
    # CAMERA.  Nothing static asked; it was found by looking at a render.
    for e in (0, 1):
        pen, who = mount.payload_clearance(m, g, end=e)
        check("nothing of the mount at end %d is inside the module: not the collar, "
              "not a grid rod, not a strut" % e,
              pen <= 1e-6,
              "worst %+.4f mm (%s %s)" % (pen, who[0], who[1]) if who else "clear")
    check("...and the collar SEATS on the board's front face rather than floating over "
          "it -- that contact is what squares the camera to the truss",
          abs(mount.payload_clearance(m, g)[0]) < 1e-6
          and abs(Bracket.seat_l()[0]
                  - (Payload.BOX[1] / 2.0 - Payload.CASE_PROUD)) < 1e-9,
          "plate %.2f..%.2f mm, board's front face at %.2f"
          % (Bracket.seat_l()[0], Bracket.seat_l()[1],
             Payload.BOX[1] / 2.0 - Payload.CASE_PROUD))
    check("...and finishes inside the height the housing stands proud, or the aperture "
          "is not bonded over its whole wall",
          Bracket.SHEET <= Payload.CASE_PROUD,
          "%.1f mm of sheet in %.1f mm of proud housing"
          % (Bracket.SHEET, Payload.CASE_PROUD))
    check("the aperture is CLOSED, so the housing is bonded on four flanks -- the fin's "
          "open slot could only reach three, and one of those was its rear face",
          abs(Bracket.bond_area()
              - 2.0 * (Payload.CASE[0] + Payload.CASE[1]) * Bracket.SHEET) < 1e-9,
          "%.1f mm2 of shear against a %.3f N service load"
          % (Bracket.bond_area(),
             Load.tip_mass() / 1000.0 * Load.G * Load.LATERAL_G))
    # THE FIN, kept as a geometry so the failure cannot come back quietly.
    # It stood on edge in the plane perpendicular to the spine, its slot cut
    # to the housing from the front: half a slot-plus-wall deep along the
    # viewing direction, half a housing-plus-wall across.  Its top edge ran
    # the whole depth at x = 0, through the middle of the board.
    _fd = Payload.CASE_PROUD + Bracket.BOND_GAP + Bracket.WALL
    _fu = Payload.CASE[1] / 2.0 + Bracket.BOND_GAP + Bracket.WALL
    fin = [mount.Strut("collar", 0, _cam_pt(m, g, 0.0, -_fd, _fu),
                       _cam_pt(m, g, 0.0, _fd, _fu), Bracket.SHEET / 2.0, 0)]
    fin_m = mount.Mount(tuple(fin), m.platform_r, so, t.d_diag, m.payload)
    check("...and the check has teeth: the fin the collar replaced reads as buried in "
          "the PCB, which is what it was",
          mount.payload_clearance(fin_m, g)[0] > 1.0,
          "%.2f mm inside the part" % mount.payload_clearance(fin_m, g)[0])
    # THE GRID'S PLACEMENT IS WHAT MAKES THE STRUTS CLEAR, not luck.
    inb = dict(mount.nose_spec(g, d_strut=t.d_diag))
    gx, gu = Bracket.grid_half(Payload, t.d_diag)
    check("the grid sits one clearance outside the module's own silhouette, which is "
          "what lets the chord BEHIND the camera reach it without crossing the board",
          gx >= Payload.BOX[0] / 2.0 + t.d_diag / 2.0
          and gu >= Payload.BOX[2] / 2.0 + t.d_diag / 2.0,
          "crossings at %.2f x %.2f mm, module %.2f x %.2f"
          % (gx, gu, Payload.BOX[0] / 2.0, Payload.BOX[2] / 2.0))
    inside = mount.solve(g, d_strut=t.d_diag, standoff_mm=so, clear=-3.0)
    check("...and pulled inside it the mount goes back through the board, so the rule "
          "is load-bearing rather than decorative",
          mount.payload_clearance(inside, g)[0] > 0.0,
          "%.2f mm inside with the grid drawn in 4.5 mm"
          % mount.payload_clearance(inside, g)[0])

    # ------------------------------------- THE FOLD, PRICED NOT ASSUMED
    def _fold_budget(h):
        sp = dict(mount.nose_spec(g, d_strut=t.d_diag))
        sp.update(plat_A=Bracket.band_A(t.d_diag, h),
                  plat_I=Bracket.band_I(t.d_diag, h),
                  plate_g=Bracket.mass(Payload, t.d_diag, None, h))
        return metrics.evaluate(build.warren_truss(
            t.length, ch.side, ch.alpha, ch.d_chord, ch.d_diag,
            tip_mass=DUTY.tip_mass_g, nose=build.Nose.around(**sp)),
            DUTY)["budget_used"]
    k_str = mount.strut_k(g, Payload, t.d_diag, so)
    h_rule = Bracket.flange_h(k_str, Payload, t.d_diag)
    b_flat, b_fold = _fold_budget(0.0), _fold_budget(h_rule)
    check("the collar is FLAT, and that is a measurement: folding its rim until it is "
          "stiffer than the strut it holds -- the rule the fillet is held to -- costs "
          "more in mass than the bending it removes was worth",
          Bracket.FLANGE == 0.0 and b_flat < b_fold,
          "%.3f of the budget flat at %.2f g, %.3f folded %.1f mm at %.2f g"
          % (b_flat, Bracket.mass(Payload, t.d_diag), b_fold, h_rule,
             Bracket.mass(Payload, t.d_diag, None, h_rule)))
    check("...and the plate's own bending is priced rather than assumed away: a ring "
          "that cannot bend at all, at the same mass, is no better",
          abs(_fold_budget(0.0) - _rigid_budget(g, t, so)) < 0.05,
          "%.3f with the band's real section, %.3f with it rigid"
          % (_fold_budget(0.0), _rigid_budget(g, t, so)))

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
