"""LOOK AT IT: the derived bat, rendered, before any number about it is quoted.

    python3 sim/scripts/batline/demo_bat.py                 # the default bat
    python3 sim/scripts/batline/demo_bat.py --bat adult     # the size check_bat has never seen
    python3 sim/scripts/batline/demo_bat.py --out DIR       # where the frames go

Twelve views of the MuJoCo model as product.py derived it -- whole, the
frame bare, the handle cut open at the board and through the cells, the
shoulder and the tip cut open, four slices across it and the parts
exploded in build order -- each saved, and all of them tiled on one
contact sheet (truss/filmstrip.py) to be read in one glance.  The views
cut rather than fade, because a hundred faint voxel faces are opaque.

What the frames are for is what check_bat cannot see: a part that is
placed exactly and is the wrong part, a bay that holds the cells and is
in the wrong place, a frame that is light enough and does not look like
a bat.  Default output: sim/out/batline/<bat>/ (git ignores sim/out).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from truss.glenv import headless   # noqa: E402  (before mujoco)
headless()

from truss import filmstrip         # noqa: E402
from batline import product, mjcf   # noqa: E402
from batline.spec import KID, Bat  # noqa: E402

BATS = {"kid": KID,
        # the size check_bat derives and has never been tuned on
        "adult": Bat("adult", length=860.0, handle=300.0, grip_d=34.0, width=108.0, depth=62.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bat", default="kid", choices=sorted(BATS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=600)
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "..", "..", "out", "batline", a.bat)
    os.makedirs(out, exist_ok=True)
    d = product.design(BATS[a.bat])
    print(product.report(d))
    shots = mjcf.views(d, width=a.width, height=a.height)
    from PIL import Image
    for i, (label, img) in enumerate(shots):
        Image.fromarray(img).save(os.path.join(out, "%02d.png" % i))
    sheet = filmstrip.contact_sheet([img for _, img in shots], os.path.join(out, "sheet.png"), cols=3,
                                    labels=["%02d %s" % (i, l) for i, (l, _) in enumerate(shots)])
    print("%d views in %s; the contact sheet is %s" % (len(shots), os.path.abspath(out), sheet))


if __name__ == "__main__":
    main()
