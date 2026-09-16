r"""Render the generated rig parts and the assembly, and the two DXFs.

    python3 scripts/truss/view_rig_cad.py [--out /tmp/rig]

A SIMULATION IS NOT REVIEWED UNTIL FRAMES FROM IT HAVE BEEN READ, and a
generated part is not reviewed until it has been looked at either.  The
numbers in `check_cad` say the mesh closes and re-derives the solve; they
cannot say the part is the part.  The first pinion passed every dimension
check with a step in each flank.

Rendered here rather than through MuJoCo on purpose: the compiler recentres
a mesh on its own centre of mass and records the shift in `mesh_pos`, which
silently moves every part off the assembled position this is drawing.
"""
import sys
import os
from math import cos, sin, radians

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from PIL import Image, ImageDraw

from truss import cad
from truss.spec import Ring, Truss, ring_r_out
from truss.drive import pinion_phase
from spine import chosen

# the ring's own frame -> the assembly's: the revolve axis becomes +x, and
# the part's theta becomes the azimuth measured from straight down
R_RING = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


def rz(deg):
    c, s = cos(radians(deg)), sin(radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def place(mesh, R, t):
    out = []
    for a, b, c in mesh.tris:
        out.append(tuple(R @ q + t for q in (a, b, c)))
    return out


def shade(tris, path, size=760, eye=(1.0, -1.6, 0.9), up=(0, 0, 1),
          light=(0.45, -0.7, 0.62), ss=2, bg=250, tint=None):
    """Painter's algorithm: sort by depth, fill, shade off the normal."""
    e = np.array(eye, float)
    e /= np.linalg.norm(e)
    u = np.cross(np.array(up, float), e)
    u /= np.linalg.norm(u)
    v = np.cross(e, u)
    l = np.array(light, float)
    l /= np.linalg.norm(l)
    T = np.array([[t[i][j] for j in range(3)] for t in tris for i in range(3)])
    P = np.stack([T @ u, T @ v], axis=1).reshape(len(tris), 3, 2)
    D = (T @ e).reshape(len(tris), 3).mean(axis=1)
    lo, hi = P.reshape(-1, 2).min(axis=0), P.reshape(-1, 2).max(axis=0)
    s = 0.92 * min(size / max(hi[0] - lo[0], 1e-6), size / max(hi[1] - lo[1], 1e-6))
    mid = (lo + hi) / 2.0
    W = size * ss
    img = Image.new("RGB", (W, W), (bg, bg, bg))
    dr = ImageDraw.Draw(img)
    order = np.argsort(D)[::-1]
    base = np.array(tint if tint else (150, 165, 180), float)
    for k in order:
        a, b, c = tris[k]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < 1e-12:
            continue
        n = n / ln
        f = 0.30 + 0.70 * max(0.0, abs(float(n @ l)))
        col = tuple(int(min(255, x * f)) for x in base)
        xy = [((P[k][i][0] - mid[0]) * s + W / 2,
               W / 2 - (P[k][i][1] - mid[1]) * s) for i in range(3)]
        dr.polygon(xy, fill=col)
    img.resize((size, size), Image.LANCZOS).save(path)
    return path


def dxf_png(d, path, size=900, pad=24):
    pts = []
    for kind, _, a in d.e:
        if kind == "TEXT":
            pts.append(np.array(a[0], float))
        elif kind == "LINE":
            pts += [np.array(a[0], float), np.array(a[1], float)]
        else:
            c, r = np.array(a[0], float), a[1]
            pts += [c - r, c + r]
    P = np.array(pts)
    lo, hi = P.min(axis=0), P.max(axis=0)
    s = min((size - 2 * pad) / max(hi[0] - lo[0], 1e-6),
            (size - 2 * pad) / max(hi[1] - lo[1], 1e-6))
    H = int((hi[1] - lo[1]) * s + 2 * pad)
    img = Image.new("RGB", (size, H), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    f = lambda p: ((p[0] - lo[0]) * s + pad, H - pad - (p[1] - lo[1]) * s)
    for kind, layer, a in d.e:
        col = (20, 20, 20) if layer == "CUT" else (215, 120, 60)
        w = 2 if layer == "CUT" else 1
        if kind == "TEXT":
            dr.text(f(a[0]), a[1], fill=col)
        elif kind == "LINE":
            dr.line([f(a[0]), f(a[1])], fill=col, width=w)
        elif kind == "CIRCLE":
            c, r = np.array(a[0], float), a[1]
            dr.ellipse([f(c - r)[0], f(c + r)[1], f(c + r)[0], f(c - r)[1]],
                       outline=col, width=w)
        else:
            c, r, a0, a1 = np.array(a[0], float), a[1], a[2], a[3]
            # A DXF ARC IS ALWAYS COUNTER-CLOCKWISE from start to end, so a
            # cap whose end angle is numerically below its start sweeps the
            # long way round.  Interpolated naively, every slot cap in
            # plate.dxf drew backwards and the slots read as open crescents
            # -- the RENDERER was wrong, not the file.
            a1 = a0 + ((a1 - a0) % 360.0 or 360.0)
            n = max(8, int(abs(a1 - a0)))
            q = [f(c + r * np.array([cos(radians(a0 + (a1 - a0) * i / n)),
                                     sin(radians(a0 + (a1 - a0) * i / n))]))
                 for i in range(n + 1)]
            dr.line(q, fill=col, width=w)
    img.save(path)
    return path


def main(argv):
    out = "/tmp/rig"
    if "--out" in argv:
        out = argv[argv.index("--out") + 1]
    os.makedirs(out, exist_ok=True)
    t = Truss(**chosen.OPTIMAL_1M.as_truss_kwargs())
    M = Ring.mesh()
    L = cad.axial_layout()
    arcs = cad.race_arcs()
    made = []

    ring = cad.ring()
    pin = cad.pinion()
    made.append(shade(place(ring, R_RING, np.array([-Ring.W / 2, 0, 0])),
                      out + "/part_ring.png", tint=(120, 150, 190)))
    made.append(shade(place(pin, np.eye(3), np.zeros(3)),
                      out + "/part_pinion.png", eye=(1.0, -1.3, 0.5),
                      tint=(205, 150, 90)))
    made.append(shade(place(cad.race(0.0, max(b - a for a, b in arcs)),
                            np.eye(3), np.zeros(3)),
                      out + "/part_race.png", tint=(140, 140, 150)))
    made.append(shade(place(cad.vblock(t.d_chord, 25.0), np.eye(3), np.zeros(3)),
                      out + "/part_vjig.png", tint=(150, 175, 140)))

    # ---- the assembly
    asm = place(ring, R_RING, np.array([-Ring.W / 2, 0, 0]))
    for a, b in arcs:
        asm += place(cad.race(0.0, b - a), R_RING @ rz(a),
                     np.array([-L["rail_out"], 0, 0]))
    for az in Ring.pinion_az():
        c = np.array([0.0, M.centre * sin(radians(az)), -M.centre * cos(radians(az))])
        asm += place(pin, R_RING @ rz(180.0 + az + pinion_phase(az, M)),
                     c + np.array([L["x0"], 0, 0]))
    # the work: a chord along the axis and a diagonal off it
    asm += _rod(t.d_chord / 2.0, np.array([-60.0, 0, 0]), np.array([30.0, 0, 0]))
    e = np.array([cos(radians(t.alpha)), -sin(radians(t.alpha)), 0.0])
    asm += _rod(t.d_diag / 2.0, -e * 60.0, e * 30.0)
    for nm, eye, up in (("iso", (1.1, -1.5, 0.85), (0, 0, 1)),
                        ("axial", (1.0, -0.06, 0.05), (0, 0, 1)),
                        ("mouth", (0.35, -0.5, -0.9), (0, 0, 1))):
        made.append(shade(asm, out + "/asm_%s.png" % nm, eye=eye, up=up))
    # and one close on a single mesh
    near = [tr for tr in asm
            if abs(np.mean([q[1] for q in tr]) - M.centre * sin(radians(66.09))) < 26
            and abs(np.mean([q[2] for q in tr]) + M.centre * cos(radians(66.09))) < 26]
    made.append(shade(near, out + "/asm_mesh.png", eye=(1.0, -0.25, 0.18)))

    made.append(dxf_png(cad.plate_dxf(), out + "/dxf_plate.png"))
    made.append(dxf_png(cad.base_dxf(t.alpha), out + "/dxf_base.png"))
    for p in made:
        print("  ", p)
    return 0


def _rod(r, p0, p1, n=20, tint=None):
    d = p1 - p0
    d = d / np.linalg.norm(d)
    a = np.cross(d, [0, 0, 1.0])
    if np.linalg.norm(a) < 1e-9:
        a = np.cross(d, [1.0, 0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(d, a)
    tris = []
    for i in range(n):
        t0, t1 = 2 * np.pi * i / n, 2 * np.pi * (i + 1) / n
        u = a * np.cos(t0) * r + b * np.sin(t0) * r
        v = a * np.cos(t1) * r + b * np.sin(t1) * r
        tris += [(p0 + u, p1 + u, p1 + v), (p0 + u, p1 + v, p0 + v)]
    return tris


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
