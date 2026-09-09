"""Tier 1: the viewer's camera -- the controls, not the pixels.

The fault this suite exists for: demo_cell opened the viewer already on a
FIXED camera, where MuJoCo's mouse does nothing at all, so the demo
arrived with no zoom, no orbit and no pan and nothing on screen to say
why.  A window cannot be opened in a check, but every part of that fault
can be: the camera the rig leaves behind, the keys it answers to, and
whether the demo forces a fixed camera at startup.

    python3 sim/scripts/truss/check_view.py [-v]
"""
import contextlib
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from truss.glenv import headless        # noqa: E402  (before mujoco)
headless()
import numpy as np
import mujoco

from truss import spec, structure, fixture, view
from truss.geometry import TrussGeometry

VERBOSE = "-v" in sys.argv
RESULTS = []
FREE = mujoco.mjtCamera.mjCAMERA_FREE
FIXED = mujoco.mjtCamera.mjCAMERA_FIXED


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


class StubViewer:
    """A passive-viewer handle as far as the rig can tell: the real
    MjvCamera the viewer holds, and the same lock discipline."""

    def __init__(self):
        self.cam = mujoco.MjvCamera()

    def lock(self):
        return contextlib.nullcontext()


KEYS = ([view.K_UP, view.K_DOWN, view.K_LEFT, view.K_RIGHT,
         view.K_PGUP, view.K_PGDN, view.K_HOME, view.K_END]
        + [ord(ch) for ch in "-=12340."])


def main():
    t = structure.TRUSS_1M
    g = TrussGeometry(t)
    fx = fixture.Fixture(g)
    frame = view.cell_frame(g, fx)
    v = StubViewer()
    close = view.close_frame(spec.ring_swept_r())
    rig = view.CameraRig(v, None, frame, follow=lambda: np.array([0.5, 0.0, 0.05]),
                         close=close)

    # ---------------------------------------------------- the frame
    lo, hi = fx.x_range()
    check("the home frame is computed from the fixture, and covers the cell",
          abs(frame[0][0] * 1000.0 - (lo + hi) / 2.0) < 1e-6
          and frame[1] * 1000.0 > (hi - lo),
          "lookat %.3f m, distance %.3f m for a %.0f mm cell" % (frame[0][0], frame[1], hi - lo))
    check("...and it is on the truss's own axis, not the model's centre of mass",
          abs(frame[0][1]) < 1e-9 and abs(frame[0][2]) < 1e-9)

    # ------------------------------------------- THE FAULT ITSELF
    check("the rig leaves the FREE camera, where the mouse works",
          v.cam.type == FREE and v.cam.fixedcamid == -1,
          "type %s" % v.cam.type)
    bad = []
    for k in KEYS:
        v.cam.type, v.cam.fixedcamid = FIXED, 1
        rig.key(k)
        if v.cam.type != FREE or v.cam.fixedcamid != -1:
            bad.append(k)
    check("every camera key also returns from a fixed camera -- any control is the way out",
          not bad, "stuck on %s" % [chr(k) if 32 < k < 127 else k for k in bad])
    src = open(os.path.join(os.path.dirname(__file__), "demo_cell.py")).read()
    body = "\n".join(l.split("#")[0] for l in re.sub(r'"""[\s\S]*?"""', "", src).splitlines())
    check("the demo never opens on a fixed camera (the fault this suite is for)",
          "mjCAMERA_FIXED" not in body and "fixedcamid" not in body,
          "%d lines of demo_cell.py scanned" % len(body.splitlines()))
    check("...and it hands the viewer a key callback, or the keys reach nothing",
          "key_callback" in body and "CameraRig" in body)

    # ------------------------------------------------------ the keys
    letters = view.taken_letters()
    clash = [chr(k) for k in KEYS if 32 < k < 127 and chr(k).upper() in letters]
    check("no camera key is one the viewer has already bound to a visualisation flag",
          not clash, "clashes on %s (the viewer binds %d)" % (clash, len(letters)))
    check("...which is not only letters: the viewer takes punctuation too",
          any(not ch.isalpha() for ch in letters),
          "bound: %s" % "".join(sorted(ch for ch in letters if not ch.isalpha())))
    check("the rig answers its own keys and declines the rest",
          all(rig.key(k) for k in KEYS) and not rig.key(ord("q")) and not rig.key(ord("z")))

    # ---------------------------------------------------- the motion
    rig.home()
    d0 = v.cam.distance
    rig.key(ord("-")); far = v.cam.distance
    rig.key(ord("=")); back = v.cam.distance
    check("zoom out and in are inverse, and neither is a fixed step",
          far > d0 and abs(back - d0) < 1e-9, "%.3f -> %.3f -> %.3f m" % (d0, far, back))
    rig.home()
    near = np.array(v.cam.lookat)
    rig.key(view.K_UP)
    step_near = np.linalg.norm(np.array(v.cam.lookat) - near)
    rig.home()
    with v.lock():
        v.cam.distance *= 10.0
    wide = np.array(v.cam.lookat)
    rig.key(view.K_UP)
    step_wide = np.linalg.norm(np.array(v.cam.lookat) - wide)
    check("the pan step scales with the zoom, so one binding works across the whole cell",
          abs(step_wide / step_near - 10.0) < 0.01,
          "%.4f m pulled back vs %.4f m close in" % (step_wide, step_near))
    rig.home()
    rig.key(view.K_UP)
    up = np.array(v.cam.lookat)
    rig.key(view.K_DOWN)
    check("pan is reversible: up then down is where you were",
          np.linalg.norm(np.array(v.cam.lookat) - frame[0]) < 1e-9,
          "%.4f m off" % np.linalg.norm(np.array(v.cam.lookat) - frame[0]))
    check("...and it moves in the ground plane, which does not degenerate looking down",
          abs(up[2] - frame[0][2]) < 1e-9)
    rig.key(ord("1"))
    check("the top view stops short of straight down, where the image rolls at random",
          -90.0 < v.cam.elevation < -89.0, "%.1f deg" % v.cam.elevation)
    rig.key(ord("3"))
    check("the end-on view looks along the truss", abs(v.cam.azimuth % 360.0) < 1e-9,
          "azimuth %.0f" % v.cam.azimuth)

    # --------------------------------------------------- follow mode
    rig.home()
    check("follow is off until asked", not rig.following)
    rig.key(ord("."))
    rig.tick()
    check("follow carries the lookat to the head", rig.following
          and np.allclose(v.cam.lookat, [0.5, 0.0, 0.05]),
          "%s" % np.round(v.cam.lookat, 3))
    rig.key(ord("4"))
    rig.tick()
    check("...and 4 goes close to it, at a distance derived from the head, not typed",
          rig.following and abs(v.cam.distance - close) < 1e-9
          and close < frame[1] / 4.0
          and abs(view.close_frame(2.0 * spec.ring_swept_r()) - 2.0 * close) < 1e-12,
          "%.3f m for a %.0f mm head, against %.2f m for the cell"
          % (v.cam.distance, spec.ring_swept_r(), frame[1]))
    rig.key(view.K_LEFT)
    check("a pan takes the wheel back: follow stops when you steer", not rig.following)
    rig.key(ord("0"))
    check("0 returns to the whole cell, free and unfollowed",
          not rig.following and v.cam.type == FREE
          and abs(v.cam.distance - frame[1]) < 1e-9)

    # ---------------------------------------------------------- help
    for want in ("mouse", "free camera", "follow"):
        check("the help says what the mouse does: '%s'" % want, want in view.HELP.lower())

    bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det and (VERBOSE or not ok) else ""))
    print("check_view: %d checks, %d failed" % (len(RESULTS), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
