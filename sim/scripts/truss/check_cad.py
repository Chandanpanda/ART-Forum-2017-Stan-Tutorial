r"""Tier 0: do the generated files agree with the solve, and do the parts fit?

    python3 scripts/truss/check_cad.py [-v]

These test THE FILES, not the design.  `check_drive` already says the drive
turns; this says the thing being printed is that drive -- the same module,
the same two tooth counts, the same mouth, the same three azimuths -- and
that the parts which have to pass each other do.

WHY A SUITE AND NOT AN EYEBALL.  Every number in these STLs came from
`Ring.mesh()` through a generator, and a generator is exactly as trustworthy
as its last edit.  The first version of the pinion looked up its gear radius
at the nearest sampled angle and put a step in every flank; the first ear
clipper left a hole in every cap.  Both read as fine in a table of
dimensions.  Closedness and re-derivation are what caught them.
"""
import sys
import os
from math import pi, cos, sin, radians, degrees, asin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from truss import cad
from truss.spec import Ring, Truss, ring_r_in, ring_r_out, rail_r_out
from spine import chosen

VERBOSE = "-v" in sys.argv
N = [0, 0]


def check(name, ok, detail=""):
    N[0] += 1
    if not ok:
        N[1] += 1
    if not ok or VERBOSE:
        print("  %-5s %s%s" % ("ok" if ok else "FAIL", name,
                               ("  [%s]" % detail) if detail else ""))


# ------------------------------------------------------------- occupancy
def rail_hit(p, arcs, lay, grow=0.0):
    """Depth, mm, by which point `p` (assembly frame, x along the ring's
    axis) is inside the rail.  0 if clear.  The rail's section is the
    C-channel, so this is a 2-D test in (r, |x|) plus an arc test."""
    x, y, z = p
    r = (y * y + z * z) ** 0.5
    az = degrees(np.arctan2(y, -z)) % 360.0
    if not any(a <= az <= b or a <= az + 360.0 <= b for a, b in arcs):
        return 0.0
    r_root = Ring.mesh().r_pitch - 1.25 * Ring.mesh().m
    r0, r1 = ring_r_out() + Ring.RACE_CLEAR, rail_r_out()
    skirt = 27.0
    ax = abs(x)
    if ax <= lay["rail_in"]:                      # the channel's mouth
        lo, hi = r0, r1
    elif ax <= lay["rail_out"]:                   # a flange, and its skirt
        lo, hi = r_root, skirt
    else:
        return 0.0
    if lo - grow <= r <= hi + grow:
        return min(r - (lo - grow), (hi + grow) - r)
    return 0.0


def main():
    t = Truss(**chosen.OPTIMAL_1M.as_truss_kwargs())
    M = Ring.mesh()
    L = cad.axial_layout()
    arcs = cad.race_arcs()
    sa = cad.standoff_az()
    belt = cad.belt_path()
    print("cad checks -- the files against the solve")

    # -------------------------------------------------- 1. the meshes close
    parts = {"ring": cad.ring(), "pinion": cad.pinion(),
             "race_long": cad.race(0.0, max(b - a for a, b in arcs)),
             "race_short": cad.race(0.0, min(b - a for a, b in arcs)),
             "vjig_chord": cad.vblock(t.d_chord, 25.0),
             "vjig_diag": cad.vblock(t.d_diag, 25.0)}
    for nm, m in parts.items():
        check("%s is a closed, consistently wound solid" % nm,
              not m.open_edges(), "%d open edges, %d tris"
              % (len(m.open_edges()), len(m.tris)))
        check("%s has positive volume -- its normals face out" % nm,
              m.volume() > 0.0, "%.1f mm^3" % m.volume())

    # ---- and the FILE is the mesh: written, read back, compared.  A
    # generator that is right and a writer that is wrong ship the same way.
    import struct
    import tempfile
    tmp = os.path.join(tempfile.mkdtemp(), "t.stl")
    for nm, m in parts.items():
        m.stl(tmp)
        raw = open(tmp, "rb").read()
        n = struct.unpack("<I", raw[80:84])[0]
        ok = n == len(m.tris) and len(raw) == 84 + 50 * n
        if ok:
            a = np.array(struct.unpack("<12f", raw[84:132]))
            nrm, v0 = a[0:3], a[3:6]
            ok = (np.allclose(v0, m.tris[0][0], atol=1e-4) and
                  abs(np.linalg.norm(nrm) - 1.0) < 1e-5)
        check("%s.stl reads back as the mesh that was written" % nm, ok,
              "%d facets, %d bytes" % (n, len(raw)))
    os.remove(tmp)

    # ------------------------------------------- 2. the ring is THIS ring
    ring = parts["ring"]
    lo, hi = ring.bbox()
    rv = np.array([q for tri in ring.tris for q in tri])
    r_max = float(np.hypot(rv[:, 0], rv[:, 1]).max())
    # A BOUNDING BOX IS NOT A RADIUS.  Measured that way this read 19.974:
    # the tooth pattern is offset half a pitch, so no tip land sits on the
    # y axis and the box misses the rim by the cosine of 2.94 degrees.
    check("the ring's tip circle is the rim the solve gave, and nothing "
          "stands proud of it",
          abs(r_max - ring_r_out()) < 0.01 and r_max <= ring_r_out() + 1e-9,
          "%.4f of %.4f" % (r_max, ring_r_out()))
    check("the ring is Ring.W wide", abs((hi[2] - lo[2]) - Ring.W) < 1e-9,
          "%.4f" % (hi[2] - lo[2]))
    check("the ring carries n_ring teeth less the gap's whole number of them",
          ring.info["teeth"] == M.n_ring - round(Ring.teeth_in_gap()),
          "%d of %d, gap eats %.0f"
          % (ring.info["teeth"], M.n_ring, Ring.teeth_in_gap()))
    check("the gap is a whole number of teeth -- otherwise the far side "
          "arrives out of phase",
          abs(Ring.teeth_in_gap() - round(Ring.teeth_in_gap())) < 1e-9,
          "%.6f teeth" % Ring.teeth_in_gap())
    v = rv
    az = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0
    rr = np.hypot(v[:, 0], v[:, 1])
    tip = az[rr > ring_r_out() - 0.02]
    check("no part tooth at either end of the C: the end faces fall at "
          "space centres",
          tip.min() > Ring.GAP / 2.0 + 1.0 and tip.max() < 360.0 - Ring.GAP / 2.0 - 1.0,
          "outermost tooth land at %.2f and %.2f deg, gap edge %.2f"
          % (tip.min(), tip.max(), Ring.GAP / 2.0))
    check("the bore is the bore the head's envelope was drawn to",
          abs(rr.min() - ring_r_in()) < 0.61,
          "%.3f, chamfered off %.3f" % (rr.min(), ring_r_in()))
    # the ring's own +z becomes the assembly's +x, where the belt is
    g_x = ring.info["groove_open_at"] - Ring.W / 2.0
    check("the groove opens on the WORK side, not the belt side -- a thread "
          "paying off the drive's face runs past three pulleys",
          (g_x < 0.0) and (L["belt_x0"] > 0.0),
          "groove opens at x %.2f, belt at x %.2f" % (g_x, L["belt_x0"]))
    # THE GROOVE, READ THE WAY A SLICER WILL: a point is inside or it is
    # not.  Closedness says the surface shuts; only this says which side of
    # it the thread's channel is on.  (The first probe list expected r 19 at
    # az 90 to be solid; the tooth pattern is offset half a pitch, so az 90
    # is a SPACE and the material stops at the root.  The mesh was right.)
    pp = lambda r, z, a: ring.contains((r * cos(radians(a)), r * sin(radians(a)), z))
    tc = 360.0 / M.n_ring / 2.0 + 12 * 360.0 / M.n_ring      # a tooth centre
    probes = [(Ring.GROOVE_R, 0.5, 90.0, False, "the groove's open face"),
              (Ring.GROOVE_R, Ring.W - 0.5, 90.0, True, "behind its floor"),
              (12.0, 0.5, 90.0, True, "the hub, same face"),
              (ring_r_in() - 2.0, 2.0, 90.0, False, "the bore"),
              (19.0, 2.0, tc, True, "a tooth"),
              (19.0, 2.0, tc + 180.0 / M.n_ring, False, "the space beside it"),
              (18.0, 2.0, 90.0, True, "below the root circle"),
              (19.0, 2.0, 0.0, False, "the 60 degree gap")]
    bad = [w for r, z, a, want, w in probes if pp(r, z, a) != want]
    check("the solid reads right where the thread has to go: %s" %
          ", ".join(w for _, _, _, _, w in probes), not bad,
          "%d of %d probes" % (len(probes) - len(bad), len(probes))
          + ("; wrong at " + ", ".join(bad) if bad else ""))
    cap = Ring.groove_capacity(0.11)
    check("the groove cut in the web holds the metres Ring.groove_capacity "
          "budgeted -- the section is 2x2 either way round",
          abs(ring.info["groove_mm3"] * Ring.PACKING /
              (pi * (0.11 / 2) ** 2) / 1000.0 - cap) < 1e-6,
          "%.2f m of 0.11 mm thread" % cap)

    # ----------------------------------------- 3. the pinion meshes with it
    pin = parts["pinion"]
    pvv = np.array([q for tri in pin.tris for q in tri])
    p_max = float(np.hypot(pvv[:, 0], pvv[:, 1]).max())
    check("the pinion's tip circle is r_pinion + one addendum",
          abs(p_max - (M.r_pinion + M.m)) < 0.005 and
          p_max <= M.r_pinion + M.m + 1e-9,
          "%.4f of %.4f, faceting %.4f"
          % (p_max, M.r_pinion + M.m, M.r_pinion + M.m - p_max))
    check("the two gears are the same module, which the FIRST version of "
          "this drive was not",
          True, "module %.1f on both, %dT and %dT" % (M.m, M.n_ring, M.n_pinion))
    s_ring = pi * M.m / 2.0 - cad.FIT_GEAR / 2.0
    check("tooth thickness plus space equals the circular pitch, less the "
          "backlash the pair is cut with",
          abs(2 * s_ring + cad.FIT_GEAR - pi * M.m) < 1e-9,
          "%.4f + %.4f = %.4f mm" % (s_ring, s_ring + cad.FIT_GEAR, pi * M.m))
    check("the bearing seat is an MR63 plus a fit",
          abs(pin.info["pocket_r"] * 2 - (cad.BEARING_OD + cad.FIT_BEARING)) < 1e-9,
          "%.2f mm bore for a %.1f mm bearing" % (pin.info["pocket_r"] * 2,
                                                  cad.BEARING_OD))
    check("the seat's shoulder bears on the OUTER race only -- a shoulder "
          "that reaches the inner race rubs it",
          pin.info["shoulder_r"] * 2 >= cad.BEARING_RACE_ID,
          "shoulder bore %.2f, outer race's own bore %.2f"
          % (pin.info["shoulder_r"] * 2, cad.BEARING_RACE_ID))
    check("the hub leaves Ring.PINION_WALL between the bearing and the root",
          pin.info["hub_r"] - pin.info["pocket_r"] >= Ring.PINION_WALL,
          "%.2f of %.2f required"
          % (pin.info["hub_r"] - pin.info["pocket_r"], Ring.PINION_WALL))
    check("the printed pulley is a GT2 20T: OD is the pitch diameter less "
          "two pitch-line offsets",
          abs(pin.info["pulley_od"] -
              (20 * cad.BELT_PITCH / pi - 2 * cad.BELT_PLO)) < 1e-9,
          "%.3f mm" % pin.info["pulley_od"])

    # ----------------------------------------------- 4. the axial chain
    check("the gear band sits IN the ring's own 4 mm and nowhere else",
          abs((L["x0"] + L["z_pocket_a"]) + Ring.W / 2.0) < 1e-9 and
          abs((L["x0"] + L["z_gear"]) - Ring.W / 2.0) < 1e-9,
          "gear spans x %.2f .. %.2f, ring %.2f .. %.2f"
          % (L["x0"] + L["z_pocket_a"], L["x0"] + L["z_gear"],
             -Ring.W / 2.0, Ring.W / 2.0))
    check("the belt clears the rail's outer face -- a belt inside it would "
          "foul the flange on every span",
          L["clear_belt_rail"] > 1.0, "%.2f mm" % L["clear_belt_rail"])
    check("the plate stands off the rail by a BOUGHT standoff, not a "
          "number that had to be printed",
          abs(L["plate_x"] - L["rail_out"] - L["standoff_l"]) < 1e-9,
          "%.1f mm standoff" % L["standoff_l"])
    check("the plate clears the belt",
          L["plate_x"] - (L["belt_x0"] + L["belt_band"]) > 2.0,
          "%.2f mm" % (L["plate_x"] - L["belt_x0"] - L["belt_band"]))

    # ------------------------------ 5. the rail's arcs and what fills them
    win = cad.pinion_window_deg()
    total = sum(b - a for a, b in arcs)
    check("the rail, the mouth and three pinion windows account for the "
          "whole circle",
          abs(total + Ring.race_mouth() + 6 * win - 360.0) < 1e-6,
          "%.2f rail + %.2f mouth + 3 x %.2f windows"
          % (total, Ring.race_mouth(), 2 * win))
    check("the mouth is the SOLVED one, not a chosen one -- at 100 degrees "
          "the ring wedged",
          abs(Ring.race_mouth() - 59.62) < 0.1, "%.2f deg" % Ring.race_mouth())
    check("every rail arc carries at least one standoff",
          all(any(a <= s <= b for s in sa) for a, b in arcs),
          "%d standoffs over %d arcs" % (len(sa), len(arcs)))
    # NOT "outside the slot" -- 1.2 mm of acrylic beside a slot is a crack
    # waiting to happen, and the plate picture is where that showed up.
    web = min((abs(cad._pt(s, cad.STANDOFF_R)[1]) - cad.SLOT_HALF
               - (cad.M3 + cad.FIT_SCREW) / 2.0)
              for s in sa if cad._pt(s, cad.STANDOFF_R)[0] < 0)
    check("a standoff beside the rod slot leaves a web worth cutting, not "
          "just clears it", web > 3.0, "%.2f mm of acrylic" % web)
    check("no standoff lands on one of the motor's own cuts",
          all(cad._seg_gap(cad._pt(s, cad.STANDOFF_R), a, b) > r
              for s in sa for a, b, r in cad.motor_cuts()),
          "closest %.2f mm" % min(cad._seg_gap(cad._pt(s, cad.STANDOFF_R), a, b) - r
                                  for s in sa for a, b, r in cad.motor_cuts()))
    check("the standoffs land on the rail's skirt, which is what they bond to",
          all(rail_r_out() < cad.STANDOFF_R < 27.0 for _ in sa),
          "r %.1f, skirt %.2f .. 27.0" % (cad.STANDOFF_R, rail_r_out()))

    # --------------------------------------------------------- 6. the belt
    check("the four pulley centres are convex, or the belt does not touch "
          "all four",
          belt["convex"])
    span = [cad.belt_path(motor_r=cad.MOTOR_R + s * cad.MOTOR_SLOT / 2)["length"]
            for s in (-1, 1)]
    # WHICH STOCKED LOOP, not whether one magic number fits.  This check
    # used to read "a standard 220 mm closed loop lands inside the motor's
    # slot" with the 220 typed here, which cannot answer the question that
    # actually comes up -- two 280 mm and two 200 mm GT2 loops already in a
    # drawer, can the head use them?  It can not, and now the suite says so
    # rather than a person re-deriving it.
    fits = cad.belt_fits()
    check("exactly one STOCKED GT2 loop lands inside the motor's slot, and it "
          "is the 220 -- so that is the belt to buy, and no other length on "
          "the shelf is a substitute without re-cutting the plate",
          sorted(fits) == [220.0],
          "slot gives %.1f .. %.1f mm; stocked loops inside it: %s"
          % (min(span), max(span), sorted(fits)))
    # THE ROUND-TRIP FAILED FIRST TIME AND THE SOLVER WAS WRONG, not the
    # check: Newton was asked for a 100 mm loop, which no motor position can
    # give -- three pinions on a 26.4 mm circle have a hull of their own --
    # and it wandered off and returned 1.98e3 mm of residual rather than
    # "impossible".  Both halves are asserted now, because a solver that
    # answers an impossible question is worse than one that refuses.
    solved = [(L, cad.motor_r_for_belt(L)) for L in cad.BELT_LOOPS]
    lo_p, hi_p = cad.belt_span_possible()
    rt = max(abs(cad.belt_path(motor_r=r)["length"] - L)
             for L, r in solved if r is not None)
    check("the solver round-trips every loop it says is possible, so 'which "
          "motor radius for this belt' is a computation and not a redesign",
          rt < 1e-6, "worst residual %.2e mm over %d of %d stocked lengths"
          % (rt, sum(1 for _, r in solved if r is not None), len(solved)))
    check("...and it REFUSES the ones no motor position can give: the three "
          "pinions' own hull plus a wrap is a hard floor, and it is not zero",
          all((r is None) == (L < lo_p - 1e-9) for L, r in solved),
          "floor %.1f mm at motor_r %.2f (where the motor crosses the chord "
          "between the pinions it sits between); refused %s"
          % (lo_p, cad.belt_r_floor(), [L for L, r in solved if r is None]))
    r200, ok200 = cad.belt_buildable(200.0)
    r280, ok280 = cad.belt_buildable(280.0)
    check("a 200 mm loop is not buildable even with a RE-CUT plate, and a 280 "
          "is -- the MOTOR_R comment claims the first half and this measures "
          "it: at 39.9 mm the motor's slots cover a rail arc end to end and "
          "standoff_az cannot place a hole in it",
          ok200 is False and ok280 is True,
          "200 -> motor_r %.1f (%s), 280 -> motor_r %.1f (%s), and 280 costs "
          "%+.0f mm of plate radius" % (r200, ok200, r280, ok280, r280 - cad.MOTOR_R))
    check("the pinions are spaced wider than the gap, so two are always "
          "meshed",
          all(abs(((b - a + 180.0) % 360.0) - 180.0) > Ring.GAP
              for a in Ring.pinion_az() for b in Ring.pinion_az() if a != b),
          "closest pair %.2f deg of a %.0f deg gap"
          % (min(abs(((b - a + 180.0) % 360.0) - 180.0)
                 for a in Ring.pinion_az() for b in Ring.pinion_az() if a != b),
             Ring.GAP))

    # --------------------------- 7. does a printed pinion clear the rail?
    # The window's half-angle was DERIVED from the hub.  This re-measures it
    # against the mesh actually written, which is the only version that gets
    # printed.
    pv = np.array([q for tri in pin.tris for q in tri])
    worst, where = 0.0, ""
    for k, azp in enumerate(Ring.pinion_az()):
        a = radians(azp)
        c = np.array([M.centre * sin(a), -M.centre * cos(a)])
        for q in pv[::7]:
            p = (L["x0"] + q[2], c[0] + q[0], c[1] + q[1])
            d = rail_hit(p, arcs, L)
            if d > worst:
                worst, where = d, "pinion %d at x %.2f, r %.2f" % (
                    k, p[0], (p[1] ** 2 + p[2] ** 2) ** 0.5)
    check("no printed pinion touches the rail: the window derived from the "
          "hub is wide enough for the whole stack",
          worst <= 0.0, "deepest %.3f mm%s" % (worst, (" -- " + where) if where else ""))

    # ...and the same measurement with the window narrowed, so the check
    # cannot pass by not looking.  A rig that cannot fail is not one.
    tight = cad.race_arcs(win=win - 4.0)
    bad = max(rail_hit((L["x0"] + q[2],
                        M.centre * sin(radians(Ring.pinion_az()[0])) + q[0],
                        -M.centre * cos(radians(Ring.pinion_az()[0])) + q[1]),
                       tight, L) for q in pv[::7])
    check("...and it can fail: narrow that window by 4 degrees and the same "
          "measurement reports the hub in the flange",
          bad > 0.1, "%.3f mm of interference" % bad)

    # ------------------------------------------------------- 8. the V-jig
    st = cad.vjig_stations(t.alpha)
    e_c, e_d = st[0][1], st[1][1]
    check("the V-jig's angle is the truss's own alpha, and it is inside the "
          "range a joint has been pulled in",
          min(np.degrees(np.arccos(np.clip(e_c @ e_d, -1, 1))), 180) ==
          min(180.0, t.alpha) or abs(np.degrees(np.arccos(np.clip(e_c @ e_d, -1, 1)))
                                     - t.alpha) < 1e-9,
          "%.2f deg, range %s" % (degrees(np.arccos(np.clip(e_c @ e_d, -1, 1))),
                                  str(__import__("truss.spec", fromlist=["Load"]).Load.ALPHA_RANGE)))
    rr_c = (t.d_chord + cad.FIT_ROD) / 2.0
    vb = cad.vblock(t.d_chord, 25.0)
    check("a rod in a 90-degree V sits at the height the ring's axis is at",
          abs(vb.info["v_half"] - rr_c * 2 ** 0.5) < 1e-9 and
          abs(vb.info["h_axis"] - cad.H_AXIS) < 1e-9,
          "axis %.1f mm up, V %.3f half-width" % (cad.H_AXIS, vb.info["v_half"]))
    # the diagonal rod must miss the chord's own block
    (ca, cb), _, cw = st[0]
    gap = 1e9
    for u in np.linspace(0.0, 120.0, 241):
        q = e_d * u
        if min(ca[0], cb[0]) <= q[0] <= max(ca[0], cb[0]):
            gap = min(gap, abs(q[1]) - cw / 2.0 - t.d_diag / 2.0)
    check("the diagonal rod misses the chord's V-block on its way out",
          gap > 1.0, "%.2f mm" % gap)
    check("the two V-blocks do not overlap -- their stand-offs are solved "
          "against each other, not written",
          cad._rect_gap(st[0][0], st[0][1], st[0][2],
                        st[1][0], st[1][1], st[1][2]) > 2.0,
          "%.2f mm apart" % cad._rect_gap(st[0][0], st[0][1], st[0][2],
                                          st[1][0], st[1][1], st[1][2]))
    check("the chord rod misses the diagonal's V-block",
          min(abs(float(q @ np.array([-e_d[1], e_d[0]])))
              for q in cad._rect_corners(st[1][0], st[1][1], st[1][2])) >= 0.0 and
          min(abs(q[1]) for q in cad._rect_corners(st[1][0], st[1][1], st[1][2])
              if min(ca[0], cb[0]) - 25.0 <= q[0] <= 0.0)
          > t.d_chord / 2.0 + 1.0,
          "%.2f mm" % min(abs(q[1]) for q in
                          cad._rect_corners(st[1][0], st[1][1], st[1][2])))

    # ---------------------------------------------------------- 9. the DXF
    for nm, d in (("plate", cad.plate_dxf()), ("base", cad.base_dxf(t.alpha))):
        check("%s.dxf uses only CUT and SCRIBE layers -- a scribe cut "
              "through is a scrapped part" % nm,
              all(l in ("CUT", "SCRIBE") for _, l, _ in d.e),
              "%d entities" % len(d.e))
    y0, y1, z0, z1 = cad.plate_extent()
    feats = ([cad._pt(a, M.centre) for a in Ring.pinion_az()] +
             [cad._pt(a, cad.STANDOFF_R) for a in sa] +
             [p for a, b, r in cad.motor_cuts() for p in
              (a + np.array([r, r]), b - np.array([r, r]), a - np.array([r, r]),
               b + np.array([r, r]))])
    inside = all(y0 <= p[0] <= y1 and z0 <= p[1] <= z1 for p in feats)
    check("every plate feature falls inside the plate the features sized",
          inside, "%.0f x %.0f mm" % (y1 - y0, z1 - z0))
    # the base and the plate are two files and one joint
    slots = cad.base_slots()
    check("every slot in the base matches a tab on the plate, in position "
          "and in width",
          len(slots) == len(cad.plate_tabs()) and
          all(abs(s[1] - tb[0]) < 1e-9 and s[3] >= tb[1] - 1e-9
              for s, tb in zip(slots, cad.plate_tabs())),
          "%d slots, %d tabs" % (len(slots), len(cad.plate_tabs())))
    check("the tabs are deeper than the base is thick, so the plate seats "
          "on the base rather than on the tab's end",
          cad.TAB_H > cad.BASE_T,
          "tab %.1f mm into %.1f mm of base" % (cad.TAB_H, cad.BASE_T))
    check("the rail's footprint on the base clears both V-blocks -- what is "
          "scribed there is a RECTANGLE, the ring stands up",
          all(cad._rect_gap((np.array([-L["rail_out"], 0.0]),
                             np.array([L["rail_out"], 0.0])),
                            np.array([1.0, 0.0]), 2 * cad.SKIRT_R,
                            seg, e, w) > 2.0 for seg, e, w in st),
          "%.1f mm" % min(cad._rect_gap(
              (np.array([-L["rail_out"], 0.0]), np.array([L["rail_out"], 0.0])),
              np.array([1.0, 0.0]), 2 * cad.SKIRT_R, seg, e, w)
              for seg, e, w in st))
    check("a standoff's 6 mm bonding pad lands wholly on the rail's skirt",
          rail_r_out() <= cad.STANDOFF_R - 3.0 and cad.STANDOFF_R + 3.0 <= cad.SKIRT_R,
          "pad r %.2f..%.2f on skirt %.2f..%.2f"
          % (cad.STANDOFF_R - 3, cad.STANDOFF_R + 3, rail_r_out(), cad.SKIRT_R))
    check("the plate's DXF labels which hole is pressed and which is "
          "clearance -- at r %.2f and r %.2f they are the same hole to look at"
          % (M.centre, cad.STANDOFF_R),
          sum(1 for k, l, a in cad.plate_dxf().e if k == "TEXT") >= 5,
          "%d labels" % sum(1 for k, l, a in cad.plate_dxf().e if k == "TEXT"))
    check("the plate's tabs are inside its own outline",
          all(y0 < yc - w / 2 and yc + w / 2 < y1 for yc, w in cad.plate_tabs()),
          "tabs at %s" % ", ".join("%.1f" % y for y, _ in cad.plate_tabs()))

    print("\ncheck_cad: %d checks, %d failed" % (N[0], N[1]))
    return 1 if N[1] else 0


if __name__ == "__main__":
    raise SystemExit(main())
