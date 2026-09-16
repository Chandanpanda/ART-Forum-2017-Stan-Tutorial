r"""Labelled diagrams of the rig's parts, for somebody who has to build it.

    python3 scripts/truss/explain_rig_cad.py [--out /tmp/rig_guide]

One picture per part with its file name and its features named, then how the
parts stack up along the ring's axis, then the drive as a whole.  Generated
from the same geometry the STLs are, so a diagram cannot drift away from the
part it describes.
"""
import sys
import os
from math import cos, sin, radians

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from truss import cad, draw
from truss.spec import Ring, Truss, ring_r_in, ring_r_out, rail_r_out
from truss.drive import pinion_phase
from spine import chosen

M = Ring.mesh()
L = cad.axial_layout()
ARCS = cad.race_arcs()
T = Truss(**chosen.OPTIMAL_1M.as_truss_kwargs())


def at(az, r, x=0.0):
    """A point in the assembly frame from a clock position and a radius."""
    a = radians(az)
    return np.array([x, r * sin(a), -r * cos(a)])


def ring_asm():
    return draw.place(cad.ring(), draw.R_RING, np.array([-Ring.W / 2, 0, 0]))


def rail_asm():
    out = []
    for a, b in ARCS:
        out += draw.place(cad.race(0.0, b - a), draw.R_RING @ draw.rz(a),
                          np.array([-L["rail_out"], 0, 0]))
    return out


def pinions_asm(which=None):
    out = []
    pin = cad.pinion()
    for k, az in enumerate(Ring.pinion_az()):
        if which is not None and k not in which:
            continue
        out += draw.place(pin, draw.R_RING @ draw.rz(180.0 + az + pinion_phase(az, M)),
                          at(az, M.centre) + np.array([L["x0"], 0, 0]))
    return out


def work_asm(x0=-70.0, x1=40.0):
    out = draw.rod(T.d_chord / 2.0, [x0, 0, 0], [x1, 0, 0])
    e = np.array([cos(radians(T.alpha)), -sin(radians(T.alpha)), 0.0])
    out += draw.rod(T.d_diag / 2.0, -e * 60.0, e * 34.0)
    return out


def belt_asm(x=None, w=0.8):
    """The belt as a chain: the spans are the hull's edges pushed out by the
    pitch radius, the corners are arcs of it.  All four pulleys are the same
    size, so every span is parallel to its own edge."""
    b = cad.belt_path()
    x = (L["belt_x0"] + L["belt_band"] / 2.0) if x is None else x
    pts = b["pts"]
    r = b["pitch_r"]
    n = len(pts)
    seg, ang = [], []
    for i in range(n):
        p, q = pts[i], pts[(i + 1) % n]
        d = q - p
        d = d / np.linalg.norm(d)
        nn = np.array([-d[1], d[0]])
        if float(nn @ (p - np.mean(pts, axis=0))) < 0:
            nn = -nn
        seg.append((p + nn * r, q + nn * r))
        ang.append(nn)
    out = []
    for i in range(n):
        a, c = seg[i]
        out += draw.rod(w, [x, a[0], a[1]], [x, c[0], c[1]], n=8)
        n0, n1 = ang[i], ang[(i + 1) % n]
        a0 = np.degrees(np.arctan2(n0[1], n0[0]))
        a1 = np.degrees(np.arctan2(n1[1], n1[0]))
        a1 = a0 + ((a1 - a0 + 180.0) % 360.0 - 180.0)
        c0 = pts[(i + 1) % n]
        k = 8
        for j in range(k):
            t0 = radians(a0 + (a1 - a0) * j / k)
            t1 = radians(a0 + (a1 - a0) * (j + 1) / k)
            out += draw.rod(w, [x, c0[0] + r * cos(t0), c0[1] + r * sin(t0)],
                            [x, c0[0] + r * cos(t1), c0[1] + r * sin(t1)], n=6)
    return out


def motor_asm():
    c = cad._pt(cad.MOTOR_AZ, cad.MOTOR_R)
    h = cad.NEMA17_SQ / 2.0
    return draw.slab([L["plate_x"] + cad.PLATE_T + 20.0, c[0], c[1]],
                     [20.0, h, h])


def plate_asm():
    y0, y1, z0, z1 = cad.plate_extent()
    return draw.slab([L["plate_x"] + cad.PLATE_T / 2.0, (y0 + y1) / 2, (z0 + z1) / 2],
                     [cad.PLATE_T / 2.0, (y1 - y0) / 2, (z1 - z0) / 2])


# ============================================ the axial chain, as a section
SEC_W, SEC_H = 1250, 780
SEC_X = (-18.0, 31.0)           # mm along the ring's axis
SEC_R = (-38.0, 40.0)           # mm of radius; SIGNED, and the sign means
                                # which cut you are looking at


def _sec_xy(x, r):
    x0, x1 = SEC_X
    r0, r1 = SEC_R
    return (150 + (x - x0) / (x1 - x0) * (SEC_W - 210),
            120 + (r1 - r) / (r1 - r0) * (SEC_H - 170))


def axial_section():
    """The axial chain as a section, to scale.

    ABOVE the axis is a cut through a PINION; BELOW it is a cut through a
    RAIL ARC.  Two different cuts in one drawing because at a pinion's
    azimuth the rail has a window, and at a rail arc there is no pinion --
    drawing both on one side would show an interference that is not there.

    (A perspective render was tried first.  The plate and the motor took
    the frame, the 4 mm that decides everything was a few pixels, and eight
    leaders crossed.  Then the rail's bands were drawn without their sign
    and landed on the pinion's side.  Both faults were obvious in the
    picture and invisible in the code.)
    """
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (SEC_W, SEC_H), draw.BG)
    dr = ImageDraw.Draw(img)
    f, fs, fb = draw._font(17), draw._font(14), draw._font(23, True)
    EDGE = (70, 76, 84)

    def band(x0, x1, r0, r1, fill, outline=EDGE):
        """A rectangle in (x, signed r).  The sign is the caller's."""
        a, b = _sec_xy(x0, r0), _sec_xy(x1, r1)
        dr.rectangle([min(a[0], b[0]), min(a[1], b[1]),
                      max(a[0], b[0]), max(a[1], b[1])], fill=fill, outline=outline)

    def tag(x, r, text, dx=0, dy=0, small=True, col=draw.INK):
        px, py = _sec_xy(x, r)
        dr.text((px + dx, py + dy), text, font=fs if small else f, fill=col)

    def dim(x0, x1, r, text):
        a, b = _sec_xy(x0, r), _sec_xy(x1, r)
        dr.line([a, b], fill=draw.LEAD, width=1)
        for q in (a, b):
            dr.line([(q[0], q[1] - 5), (q[0], q[1] + 5)], fill=draw.LEAD, width=1)
        w = dr.textlength(text, font=fs)
        cx = (a[0] + b[0]) / 2
        dr.rectangle([cx - w / 2 - 3, a[1] - 9, cx + w / 2 + 3, a[1] + 9], fill=draw.BG)
        dr.text((cx - w / 2, a[1] - 7), text, font=fs, fill=draw.LEAD)

    BLUE, GREY, ORANGE, BRASS, PLATE = ((150, 178, 214), (168, 172, 180),
                                        (222, 176, 124), (208, 190, 138),
                                        (198, 202, 208))
    w2 = Ring.W / 2.0
    r_root = M.r_pitch - 1.25 * M.m
    gt = cad.gt2_r(20).R

    # ---------------------------------------------------------- the plate
    band(L["plate_x"], L["plate_x"] + cad.PLATE_T, SEC_R[0] + 3.0, SEC_R[1] - 5.0,
         PLATE)
    tag(L["plate_x"] + 1.0, SEC_R[0] + 6.0, "the PLATE", 0, 0, small=False)

    # ------------------------------------- the axis, and the work on it
    dr.line([_sec_xy(SEC_X[0], 0.0), _sec_xy(SEC_X[1], 0.0)],
            fill=(160, 160, 160), width=1)
    band(SEC_X[0], SEC_X[1], -T.d_chord / 2.0, T.d_chord / 2.0, (58, 62, 68),
         (40, 44, 50))
    tag(SEC_X[0] + 0.5, 0.0, "the CHORD, a 3.0 mm rod  -  THE WORK", 0, -34,
        small=False)

    # ---- ABOVE THE AXIS: a cut through a pinion ------------------------
    band(L["x0"] - 6.0, L["plate_x"] + 2.0, M.centre - 0.6, M.centre + 0.6,
         (118, 122, 130), (88, 92, 100))
    for x0, x1, rho in cad.pinion_bands():
        band(x0, x1, M.centre - rho, M.centre + rho, ORANGE)
    for sg in (-1, 1):
        band(L["belt_x0"] + 0.6, L["belt_x0"] + L["belt_band"] - 0.6,
             M.centre + sg * gt - 0.1, M.centre + sg * (gt + 1.5),
             (66, 70, 76), (46, 50, 56))
    tag(-w2 + 0.5, M.centre + M.r_pinion + M.m, "PINION  (gear)", 0, -26, small=False)
    tag(L["belt_x0"] + 1.2, M.centre + gt + 1.5, "the BELT runs here", 0, -20)
    tag(L["x0"] - 6.0, M.centre + 0.6, "3 mm SHAFT, fixed in the plate", -4, -20)
    for x in (L["x0"] + L["z_pocket_a"] / 2, L["x0"] + (L["z_belt"] + L["z_top"]) / 2):
        px, py = _sec_xy(x, M.centre + 3.0)
        dr.line([(px, py), (px, py - 26)], fill=draw.LEAD, width=1)
        dr.text((px - 22, py - 42), "bearing", font=fs, fill=draw.LEAD)
    dr.text((22, _sec_xy(0, M.centre)[1] - 58),
            "ABOVE:\ncut through\na PINION", font=f, fill=(96, 104, 114))

    # ---- the ring: it is there at EVERY azimuth, so it spans both halves
    for sg in (1, -1):
        band(-w2, w2, sg * ring_r_in(), sg * ring_r_out(), BLUE)
        band(-w2, -w2 + Ring.GROOVE_D,
             sg * (Ring.GROOVE_R - Ring.GROOVE_W / 2),
             sg * (Ring.GROOVE_R + Ring.GROOVE_W / 2), draw.BG)
    tag(w2, -(ring_r_in() + ring_r_out()) / 2, "the RING", 16, -10, small=False)
    px, py = _sec_xy(-w2 + Ring.GROOVE_D / 2, -Ring.GROOVE_R)
    dr.line([(px, py), (px - 176, py - 34)], fill=draw.LEAD, width=1)
    dr.text((px - 300, py - 44), "its GROOVE: the thread lives in here",
            font=fs, fill=draw.INK)

    # ---- BELOW THE AXIS: a cut through a rail arc ----------------------
    band(-L["rail_out"], -L["rail_in"], -r_root, -cad.SKIRT_R, GREY)
    band(L["rail_in"], L["rail_out"], -r_root, -cad.SKIRT_R, GREY)
    band(-L["rail_in"], L["rail_in"], -(ring_r_out() + Ring.RACE_CLEAR),
         -rail_r_out(), GREY)
    band(L["rail_out"], L["plate_x"], -(cad.STANDOFF_R - 3.0),
         -(cad.STANDOFF_R + 3.0), BRASS)
    tag(-L["rail_out"], -cad.SKIRT_R, "RAIL, left flange", -124, 6)
    tag(L["rail_out"], -cad.SKIRT_R, "right flange", -66, 6)
    px, py = _sec_xy(0.0, -(rail_r_out() + ring_r_out()) / 2)
    dr.line([(px, py), (px - 120, py + 40)], fill=draw.LEAD, width=1)
    dr.text((px - 188, py + 34), "back wall", font=fs, fill=draw.INK)
    tag(L["rail_out"] + 4.5, -(cad.STANDOFF_R + 3.0), "M3 STANDOFF, 12 mm", 0, 9)
    dr.text((22, _sec_xy(0, -M.centre)[1] - 30),
            "BELOW:\ncut through\na RAIL ARC\n(no pinion\nthere)",
            font=f, fill=(96, 104, 114))

    # ---- the four numbers that fix the whole chain
    dim(-w2, w2, ring_r_in() - 3.0, "4.00  the ring")
    dim(-L["rail_out"], -L["rail_in"], -cad.SKIRT_R - 5.0, "3.00 flange")
    dim(L["rail_out"], L["plate_x"], -cad.SKIRT_R - 5.0, "12.00  standoff, bought")
    dim(L["rail_out"], L["belt_x0"], M.centre + gt + 6.0, "2.00  belt clears the rail")

    dr.text((26, 20), "How it stacks up along the rod  -  a section, to scale",
            font=fb, fill=draw.INK)
    dr.text((26, 58), "Nothing here was chosen.  The gear has to sit inside the "
            "ring's 4 mm, the belt has to clear the rail, and the plate stands off "
            "on a BOUGHT 12 mm", font=fs, fill=(96, 104, 114))
    dr.text((26, 78), "standoff.  Those three facts fix every other position in "
            "the machine.", font=fs, fill=(96, 104, 114))
    return img


def main(argv):
    out = "/tmp/rig_guide"
    if "--out" in argv:
        out = argv[argv.index("--out") + 1]
    os.makedirs(out, exist_ok=True)
    made = []

    def save(name, img):
        p = os.path.join(out, name)
        img.save(p)
        made.append(p)

    BLUE, ORANGE, GREY, GREEN = ((118, 150, 192), (206, 150, 92),
                                 (146, 150, 158), (142, 170, 138))

    # ---------------------------------------------------------- 1. the ring
    img, pr = draw.render(ring_asm(), eye=(0.9, -1.3, 0.75), tint=BLUE)
    draw.callout(img, pr, [
        (at(200, ring_r_out(), 0.0), "teeth (the RIM) - 40 of them", 0.28, 0.17),
        (at(0, 15.0, 0.0), "the GAP, 60 deg = 8 teeth wide:\nhow the rod gets in", 0.60, 0.92),
        (at(240, Ring.GROOVE_R, -Ring.W / 2 + 0.6),
         "the GROOVE - the thread's own spool", 0.05, 0.66),
        (at(120, ring_r_in(), 0.0), "the BORE - the rod sits here", 0.93, 0.36),
    ], "ring.stl  -  \"the ring\" (a C-ring, or a split ring)",
        "40 mm across, 4 mm thick.  It turns around the rod and carries the thread.  Print 1, in resin.")
    save("01_ring.png", img)

    # -------------------------------------------------------- 2. the pinion
    img, pr = draw.render(cad.pinion().tris, eye=(0.95, -1.15, 0.45), tint=ORANGE,
                          margin=0.62)
    z = (L["z_pocket_a"], L["z_gear"], L["z_hub"], L["z_belt"], L["z_top"])
    img = draw.callout(img, pr, [
        ((M.r_pinion + M.m, 0.0, (z[0] + z[1]) / 2),
         "GEAR TEETH, 18 of them -\nthese drive the ring", 0.05, 0.62),
        ((cad.gt2_r(20).R, 0.0, (z[2] + z[3]) / 2),
         "BELT PULLEY (GT2, 20 teeth) -\nthe belt runs here", 0.95, 0.26),
        ((2.2, -2.4, 0.0), "BEARING SEAT - an MR63\npresses into each end", 0.95, 0.76),
        ((2.2, -2.4, z[4]), "the other BEARING SEAT", 0.05, 0.20),
        ((0.0, -2.0, z[4]), "the 3 mm SHAFT passes\nright through both", 0.50, 0.94),
    ], "pinion.stl  -  \"a pinion\" (a small gear that drives a big one)",
        "16 mm across, 21.7 mm tall.  Print 3, in resin.  All three are the same part.")
    save("02_pinion.png", img)

    # ---------------------------------------------------------- 3. the rail
    arc = max(b - a for a, b in ARCS)
    img, pr = draw.render(cad.race(0.0, arc).tris, eye=(0.75, -0.9, 0.95),
                          tint=GREY, margin=0.66)
    mid = radians(arc / 2.0)
    pt = lambda r, zz: (r * cos(mid), r * sin(mid), zz)
    img = draw.callout(img, pr, [
        (pt(19.0, 2 * L["rail_out"] / 2), "the CHANNEL - the ring runs in here,\n"
         "with 0.05 mm to spare", 0.06, 0.30),
        (pt(20.5, 2 * L["rail_out"] / 2), "the BACK WALL - stops the ring\n"
         "moving outwards", 0.96, 0.22),
        (pt(21.0, 0.6), "a FLANGE - one each side,\nthey stop it falling out sideways",
         0.96, 0.70),
        (pt(28.0, 0.6), "the SKIRT - a flat pad; the\nstandoffs glue to it", 0.06, 0.82),
    ], "race_long.stl / race_short.stl  -  \"the rail\" (or raceway)",
        "A C-shaped track. Print 2 long + 2 short. Four pieces because each pinion "
        "needs a gap to reach through.")
    save("03_rail.png", img)

    # ------------------------------------------------- 4. how they stack up
    # A PERSPECTIVE RENDER IS THE WRONG DRAWING FOR A STACK.  Tried that
    # first: the plate and the motor took most of the frame, the 4 mm that
    # decides everything was a few pixels, and eight leader lines crossed
    # each other.  A section is the drawing an axial chain wants.
    save("04_stack.png", axial_section())

    # --------------------------------------------------- 5. the drive, front
    asm = ring_asm() + rail_asm() + pinions_asm() + belt_asm() + work_asm()
    img, pr = draw.render(asm, eye=(1.0, -0.05, 0.04), up=(0, 0, 1),
                          size=(1100, 900), tint=(150, 163, 178), margin=0.70)
    mc = cad._pt(cad.MOTOR_AZ, cad.MOTOR_R)
    img = draw.callout(img, pr, [
        (at(Ring.pinion_az()[0], M.centre, 8.0), "pinion 1", 0.93, 0.30),
        (at(Ring.pinion_az()[1], M.centre, 8.0), "pinion 2", 0.60, 0.16),
        (at(Ring.pinion_az()[2], M.centre, 8.0), "pinion 3", 0.07, 0.30),
        (np.array([10.0, mc[0], mc[1]]), "MOTOR PULLEY goes here\n"
         "(on the far side of the plate)", 0.93, 0.17),
        (np.array([10.0, 40.0, -4.0]), "one BELT round all four:\n"
         "that is what keeps them in step", 0.62, 0.95),
        (at(0, 22.0, 0.0), "the MOUTH - the rail stops here\nso the rod can come in",
         0.10, 0.92),
    ], "The drive: one motor, one belt, three pinions, one ring",
        "The belt goes round the OUTSIDE of all four wheels, so all four turn the same "
        "way and stay in step.")
    save("05_drive.png", img)

    # ------------------------------------------------------- 6. the V-block
    img, pr = draw.render(cad.vblock(T.d_chord, 25.0).tris, eye=(0.8, -1.0, 0.5),
                          tint=GREEN, margin=0.62)
    img = draw.callout(img, pr, [
        ((0.0, cad.H_AXIS, 12.5), "the V - the rod drops in here\n"
         "and is glued", 0.93, 0.18),
        ((0.0, 3.0, 12.5), "the PEDESTAL - glue this\nto the base", 0.93, 0.80),
        ((0.0, cad.H_AXIS / 2, 0.0), "%.0f mm tall: that is the\nheight of the rod's axis"
         % cad.H_AXIS, 0.06, 0.44),
    ], "vjig_chord.stl / vjig_diag.stl  -  \"V-block\" (part of the JIG)",
        "A jig is anything that holds the work still in a known place.  One V-block for "
        "the 3 mm chord, one for the 1.5 mm diagonal.")
    save("06_vblock.png", img)

    # ----------------------------------------------------------- 7. the DXFs
    img, f = draw.dxf_image(cad.plate_dxf(), size=(1150, 980), top=86)
    draw.callout(img, lambda p: f(p[:2]), [
        (at(Ring.pinion_az()[0], M.centre)[1:], "PRESS fit, 2.95 mm:\n"
         "a 3 mm shaft grips in it", 0.95, 0.33),
        (cad._pt(cad.standoff_az()[2], cad.STANDOFF_R),
         "CLEARANCE, 3.4 mm: an M3\nscrew slides through", 0.05, 0.22),
        (cad._pt(cad.MOTOR_AZ, cad.MOTOR_R + 12.0),
         "SLOTS, not holes: sliding the\nmotor tightens the belt", 0.95, 0.13),
        (np.array([-30.0, 0.0]), "ROD SLOT: the rod comes in\nthrough here", 0.05, 0.56),
        (np.array([-cad.SKIRT_R - 20.0, -38.0]),
         "a LEGEND, engraved: which hole\nis which, and what not to cut",
         0.60, 0.95),
    ], "plate.dxf  -  \"the plate\"",
        "Laser cut, 6 mm acrylic.  Black lines are cut. It holds the three shafts and "
        "the motor in the right places.")
    save("07_plate.png", img)

    img, f = draw.dxf_image(cad.base_dxf(T.alpha), size=(1100, 980), top=86)
    draw.callout(img, lambda p: f(p[:2]), [
        (np.array([L["plate_x"] + cad.PLATE_T / 2, cad.plate_tabs()[1][0]]),
         "SLOTS: the plate's tabs\ndrop in here, standing it up", 0.05, 0.17),
        (np.array([-32.0, 30.0]), "SCRIBE box: glue the diagonal's\n"
         "V-block on this mark", 0.05, 0.62),
        (np.array([-32.0, 0.0]), "SCRIBE box: the chord's V-block", 0.42, 0.95),
        (np.array([0.0, cad.SKIRT_R]), "where the rail stands.  It is a\n"
         "RECTANGLE: the ring stands up", 0.95, 0.33),
    ], "base.dxf  -  \"the base\"",
        "Laser cut, 6 mm acrylic.  Everything sits on this.  The orange marks are where "
        "to glue, not where to cut.")
    save("08_base.png", img)

    for p in made:
        print("  ", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
