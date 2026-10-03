"""LOOK AT IT: what the factory iPhone sees at station 4 -- the bat turned
by the gimbal through the slow fused swing, rendered from the phone's
unfolded camera, with the app's own idea of where each band lands.

    python3 sim/scripts/batline/demo_phone.py              # the kid's bat
    python3 sim/scripts/batline/demo_phone.py --bat adult  # the adult's
    python3 sim/scripts/batline/demo_phone.py --out DIR

The swing is phone/test.py's design (slow_swing: the turn the test's power
asks for) and the picture phone/picture.py's: mjcf.py's bat, every part at
its design pose in the clamp, the gimbal itself left out, seen through the
camera's View (Phone.F_PX at Rules.PLAY_Z along the unfolded path through
the fold mirror, square on to the swing plane).  Each frame is the phone's
image at a time on the recorded path (the stance, the swing, the end),
half size, with red crosses where the app puts each band's centroid
(app.seen: its silhouette's middle) and a green one on the sweet spot.
The label gives, per band, how far the rendered band's pixels' centroid
lies from the app's, and from a colour threshold's centroid (what a naive
band finder would report).

These frames are how the bands' offset was found: on the rounded-triangle
section, seen side-on with a flat face edge-on to the lens, a band's
silhouette is centred R/4 off the axis, 2.6 px at 4 m on the kid's bat,
and the app took the centre;
check_swing 15 now holds the app's model to the rendered picture.  What
the frames are for is what check_link cannot see: a band out of the
picture or edge-on to the lens, the bat crossing the frame's edge, the
app's pinhole disagreeing with the picture it is meant to describe.
Default output: sim/out/batline/phone/<bat>/ (git ignores sim/out).
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from truss.glenv import headless   # noqa: E402  (before mujoco)
headless()

import numpy as np                  # noqa: E402

from truss import filmstrip         # noqa: E402
from batline import product         # noqa: E402
from batline.spec import KID, Bat, Rules          # noqa: E402
from batline.rig import spec as RS  # noqa: E402
from batline.phone import test as PT, app as A, picture as PP   # noqa: E402

SCALE = 2                           # the phone's pixels to one of the sheet's
N_FRAMES = 15
BATS = {"kid": KID,                 # the adult's as check_bat draws it
        "adult": Bat("adult", length=860.0, handle=300.0, grip_d=34.0, width=108.0, depth=62.0)}


def centroid(img, rgb):
    """(u, v) px, pixel centres, of the pixels nearer the band's colour than
    half its distance from the picture's mean colour, or None."""
    c = np.asarray(rgb, float) * 255.0
    dist = np.linalg.norm(img.astype(float) - c, axis=2)
    m = dist < 0.5 * np.linalg.norm(c - img.reshape(-1, 3).mean(0))
    if m.sum() < 4:
        return None
    v, u = np.nonzero(m)
    return np.array([u.mean(), v.mean()]) + 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bat", default="kid", choices=sorted(BATS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from PIL import Image, ImageDraw
    d = product.design(BATS[a.bat])
    out = a.out or os.path.join(HERE, "..", "..", "out", "batline", "phone", a.bat)
    os.makedirs(out, exist_ok=True)
    t_wall = time.time()
    sp = PT.slow_swing(d)
    v = sp.view
    print("%s: the swing turns %.1f deg, designed in %.0f s" % (a.bat, np.degrees(sp.q_b[1] - sp.q_a[1]),
                                                               time.time() - t_wall))
    pic = PP.Picture(d, sp, RS.design(d), scale=SCALE)
    frames, labels, worst, worst_c = [], [], 0.0, 0.0
    for t in np.linspace(0.0, sp.path.T, N_FRAMES):
        q = sp.path.state(np.array([t]))[0][0]
        pic.pose(q)
        img = pic.rgb()
        got, _ = pic.silhouettes()
        Rc, x = sp.kin.pose(np.atleast_2d(q))
        uv = np.vstack([v.project_level(A.seen(v.level(x[0])[0], v.R.T @ Rc[0], sp.geom)),
                        v.project(x[0] + Rc[0] @ sp.geom.ss)])
        col = [centroid(img, Rules.BAND_RGB[i]) for i in range(2)]
        gaps = np.linalg.norm(got - uv[:2], axis=1)
        gaps_c = [float(np.linalg.norm(c * SCALE - got[i])) if c is not None else np.nan for i, c in enumerate(col)]
        worst, worst_c = np.nanmax([worst, *gaps]), np.nanmax([worst_c, *gaps_c])
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        for k, (u, w) in enumerate(uv / SCALE):
            c = (255, 0, 0) if k < 2 else (0, 200, 0)
            dr.line([(u - 6, w), (u + 6, w)], fill=c, width=1)
            dr.line([(u, w - 6), (u, w + 6)], fill=c, width=1)
        frames.append(np.asarray(im))
        phase = ("stance" if t < sp.t_stance else ("swing" if t < sp.t_stance + sp.t_swing else "end"))
        labels.append("%.2f s %s  hinges (%.0f, %.0f) deg  app %s px, threshold %s px"
                      % (t, phase, *np.degrees(q), "/".join("%.1f" % g for g in gaps),
                         "/".join("%.1f" % g for g in gaps_c)))
    pic.close()
    path = os.path.join(out, "sweep.png")
    filmstrip.contact_sheet(frames, path, cols=3, labels=labels)
    print("%d frames: %s" % (len(frames), os.path.abspath(path)))
    print("the rendered bands' centroids sit at most %.2f phone px from the app's; a colour threshold's at most "
          "%.2f px from the rendered" % (worst, worst_c))
    print("%.0f s wall" % (time.time() - t_wall))


if __name__ == "__main__":
    main()
