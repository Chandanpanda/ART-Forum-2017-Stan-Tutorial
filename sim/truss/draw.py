"""Drawing the generated parts: shaded views, DXF previews, and call-outs.

WHY THIS IS A MODULE.  `cad.py` makes geometry and `check_cad` measures it;
neither can say the part is the part.  The faults that got through both --
a rail window derived from the wrong feature, a scribe circle for a part
that stands up -- were found by looking.  So looking is a tool here, not a
one-off script, and a call-out that names a feature is the same tool used
to explain the part to somebody who has to build it.

Rendered without MuJoCo on purpose: its compiler recentres a mesh on its
own centre of mass and records the shift in `mesh_pos`, which silently
moves every part off the assembled position being drawn.
"""
from math import cos, sin, radians, atan2

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
INK = (28, 32, 38)
LEAD = (196, 92, 40)
BG = (250, 249, 247)


def _font(px, bold=False):
    try:
        return ImageFont.truetype(FONT_B if bold else FONT, px)
    except Exception:
        return ImageFont.load_default()


def render(tris, size=(1100, 760), eye=(1.0, -1.6, 0.9), up=(0, 0, 1),
           light=(0.45, -0.7, 0.62), ss=2, tint=(150, 165, 180),
           margin=0.80, edges=False):
    """Painter's algorithm.  Returns (image, project) where `project` maps a
    point in the part's own frame to a pixel, so a call-out can be anchored
    on a FEATURE rather than on a guessed pixel."""
    e = np.array(eye, float)
    e = e / np.linalg.norm(e)
    u = np.cross(np.array(up, float), e)
    u = u / np.linalg.norm(u)
    v = np.cross(e, u)
    l = np.array(light, float)
    l = l / np.linalg.norm(l)
    W, H = size
    T = np.array([q for t in tris for q in t])
    P2 = np.stack([T @ u, T @ v], axis=1)
    lo, hi = P2.min(axis=0), P2.max(axis=0)
    s = margin * min(W / max(hi[0] - lo[0], 1e-6), H / max(hi[1] - lo[1], 1e-6))
    mid = (lo + hi) / 2.0

    def project(p):
        p = np.asarray(p, float)
        return ((float(p @ u) - mid[0]) * s + W / 2.0,
                H / 2.0 - (float(p @ v) - mid[1]) * s)

    img = Image.new("RGB", (W * ss, H * ss), BG)
    dr = ImageDraw.Draw(img)
    depth = np.array([float(np.mean([q @ e for q in t])) for t in tris])
    base = np.array(tint, float)
    for k in np.argsort(depth)[::-1]:
        a, b, c = tris[k]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < 1e-12:
            continue
        f = 0.28 + 0.72 * max(0.0, abs(float((n / ln) @ l)))
        col = tuple(int(min(255, x * f)) for x in base)
        xy = [tuple(q * ss for q in project(p)) for p in (a, b, c)]
        dr.polygon(xy, fill=col, outline=col if not edges else (90, 98, 108))
    return img.resize((W, H), Image.LANCZOS), project


def callout(img, project, notes, title=None, subtitle=None, px=17):
    """Draw leader lines to named features.

    A note is (anchor3d, text, tx, ty) with tx, ty in 0..1 of the image --
    placed by hand, because six diagrams do not need a layout engine and a
    layout engine would put a label somewhere nobody chose.
    """
    dr = ImageDraw.Draw(img)
    f, fb = _font(px), _font(px + 5, True)
    W, H = img.size
    for anchor, text, tx, ty in notes:
        ax, ay = project(anchor)
        bx, by = tx * W, ty * H
        lines = text.split("\n")
        # PIL will not measure multiline text, so measure the lines
        w = max(dr.textlength(ln, font=f) for ln in lines)
        h = len(lines) * (px + 4)
        left = bx - w - 12 if bx > W / 2 else bx + 12
        top = by - h / 2.0
        dr.line([(ax, ay), (bx, by)], fill=LEAD, width=2)
        dr.ellipse([ax - 4, ay - 4, ax + 4, ay + 4], fill=LEAD)
        dr.rectangle((left - 7, top - 5, left + w + 7, top + h + 3),
                     fill=(255, 255, 255), outline=LEAD)
        for i, ln in enumerate(lines):
            dr.text((left, top + i * (px + 4)), ln, font=f, fill=INK)
    if title:
        dr.text((26, 20), title, font=fb, fill=INK)
    if subtitle:
        dr.text((26, 24 + px + 10), subtitle, font=_font(px - 1), fill=(96, 104, 114))
    return img


def dxf_image(d, size=(1100, 900), pad=34, labels=True, top=0):
    """A laser file as a picture: CUT black, SCRIBE orange, TEXT as written."""
    pts = []
    for kind, _, a in d.e:
        if kind == "LINE":
            pts += [np.array(a[0], float), np.array(a[1], float)]
        elif kind == "TEXT":
            pts.append(np.array(a[0], float))
        else:
            c, r = np.array(a[0], float), a[1]
            pts += [c - r, c + r]
    P = np.array(pts)
    lo, hi = P.min(axis=0), P.max(axis=0)
    W, H = size
    s = min((W - 2 * pad) / max(hi[0] - lo[0], 1e-6),
            (H - top - 2 * pad) / max(hi[1] - lo[1], 1e-6))
    img = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    ox = (W - (hi[0] - lo[0]) * s) / 2.0
    oy = (H - top - (hi[1] - lo[1]) * s) / 2.0
    f = lambda p: ((p[0] - lo[0]) * s + ox, H - oy - (p[1] - lo[1]) * s)
    for kind, layer, a in d.e:
        col = INK if layer == "CUT" else LEAD
        w = 3 if layer == "CUT" else 1
        if kind == "TEXT":
            if labels:
                dr.text(f(a[0]), a[1], font=_font(max(9, int(a[2] * s * 0.9))),
                        fill=col)
        elif kind == "LINE":
            dr.line([f(a[0]), f(a[1])], fill=col, width=w)
        elif kind == "CIRCLE":
            c, r = np.array(a[0], float), a[1]
            dr.ellipse([f(c - r)[0], f(c + r)[1], f(c + r)[0], f(c - r)[1]],
                       outline=col, width=w)
        else:
            c, r, a0, a1 = np.array(a[0], float), a[1], a[2], a[3]
            # A DXF ARC IS ALWAYS COUNTER-CLOCKWISE from start to end.  Drawn
            # the short way, every slot cap in plate.dxf came out an open
            # crescent -- the drawing was wrong, the file was right.
            a1 = a0 + ((a1 - a0) % 360.0 or 360.0)
            n = max(8, int(abs(a1 - a0)))
            q = [f(c + r * np.array([cos(radians(a0 + (a1 - a0) * i / n)),
                                     sin(radians(a0 + (a1 - a0) * i / n))]))
                 for i in range(n + 1)]
            dr.line(q, fill=col, width=w)
    return img, f


def rod(r, p0, p1, n=24):
    """A round rod as triangles, so the work can be drawn with the machine."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    d = d / np.linalg.norm(d)
    a = np.cross(d, [0, 0, 1.0])
    if np.linalg.norm(a) < 1e-9:
        a = np.cross(d, [1.0, 0, 0])
    a = a / np.linalg.norm(a)
    b = np.cross(d, a)
    out = []
    for i in range(n):
        t0, t1 = 2 * np.pi * i / n, 2 * np.pi * (i + 1) / n
        u = a * cos(t0) * r + b * sin(t0) * r
        v = a * cos(t1) * r + b * sin(t1) * r
        out += [(p0 + u, p1 + u, p1 + v), (p0 + u, p1 + v, p0 + v)]
    return out


def slab(c, half):
    """An axis-aligned box, for drawing the plate and the base in context."""
    c, half = np.asarray(c, float), np.asarray(half, float)
    out = []
    for ax in range(3):
        for sg in (-1, 1):
            i, j = [k for k in range(3) if k != ax]
            q = []
            for si, sj in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3)
                p[ax] = sg * half[ax]
                p[i] = si * half[i]
                p[j] = sj * half[j]
                q.append(c + p)
            out += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
    return out


def place(mesh_or_tris, R, t):
    """Move a mesh (or a triangle list) into the assembly's frame."""
    tris = mesh_or_tris.tris if hasattr(mesh_or_tris, "tris") else mesh_or_tris
    R, t = np.asarray(R, float), np.asarray(t, float)
    return [tuple(R @ np.asarray(q, float) + t for q in tri) for tri in tris]


def rz(deg):
    c, s = cos(radians(deg)), sin(radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# the ring's own frame -> the assembly's: the revolve axis becomes +x, and
# the part's theta becomes the azimuth measured from straight down
R_RING = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
