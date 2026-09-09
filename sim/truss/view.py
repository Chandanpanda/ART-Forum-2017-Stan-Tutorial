"""Camera controls for the cell's passive viewer: pan, orbit, zoom, follow.

WHY THIS EXISTS, AND THE TRAP IT CLEARS.  MuJoCo's viewer puts all three
Blender operations on the mouse -- left-drag orbits, right-drag pans,
scroll zooms -- but only on the FREE camera.  This model carries three
fixed viewpoints (`cell`, `side`, and the head's own `head_cam`), and one
press of `[` or `]` moves you onto one, where the mouse does nothing at
all.  Nothing on screen says so.  Worse, a script can open the viewer
already on a fixed camera, and then the mouse is dead from the first
frame -- which is exactly what demo_cell.py did, and what this module was
written to stop.

So: the demo opens on the free camera, framed on the cell, with the mouse
live; the trio also gets a keyboard binding; and EVERY key here snaps
back to the free camera first, so any control key is also the way out.

THE BINDINGS ARE THE ONES THE COMPETITION DEMO USES.  Same keys, same
meanings, so muscle memory carries across the two rigs in this repository.
They cannot be letters: the viewer binds A-Z to visualisation flags
(mjVISSTRING and mjRNDSTRING, listed by `taken_letters()` below), so a
letter would fire twice -- once for us, once to toggle wireframe or
contact forces.  Everything here is arrows, digits and punctuation.

The steps scale with the zoom distance, which is what lets one set of
bindings work both across a 1.4 m cell and inside an 8 mm band: close in
a tap moves a tenth of a millimetre, pulled back it moves 100 mm.

Nothing here knows the truss: the home frame and the thing to follow are
arguments, so the same rig serves any cell this package builds.
"""
import numpy as np
import mujoco

# GLFW key codes.  Spelled out rather than imported: mujoco.viewer pulls
# glfw in itself, but this module is also imported by checks that never
# open a window.
K_RIGHT, K_LEFT, K_DOWN, K_UP = 262, 263, 264, 265
K_PGUP, K_PGDN, K_HOME, K_END = 266, 267, 268, 269

PAN_FRAC  = 0.09      # of the zoom distance, per press
ZOOM_STEP = 1.18      # per press
ORBIT_DEG = 6.0       # per press
CLOSE_SPANS = 6.0     # head radii the close view frames

HELP = """  camera:  mouse works on the free camera -- left-drag orbits,
                    right-drag pans, scroll zooms
           arrows pan            Page Up/Down raise, lower
           Home/End orbit        - and = zoom out, in
           1 top   2 side   3 end-on   4 close on the head
           . follow the head     0 free camera, whole cell
           ([ and ] switch to the fixed cameras, where the mouse does
            nothing; any key above, or Esc, brings the free camera back)"""


def taken_letters():
    """The letters the viewer has already bound, and to what.

    Not used by the rig -- it is here so the claim in the header can be
    checked against the installed MuJoCo rather than believed.
    """
    out = {}
    for tbl in (mujoco.mjVISSTRING, mujoco.mjRNDSTRING):
        for row in tbl:
            row = list(row)
            if len(row) >= 3 and row[2].strip():
                out.setdefault(row[2].strip().upper(), []).append(row[0])
    return out


def cell_frame(geom, fixture):
    """(lookat, distance) in METRES that frames the whole cell: the truss,
    its racks and the head's travel.  Computed, never typed."""
    lo, hi = fixture.x_range()
    y = fixture.y_reach()
    r = fixture.cage_swept_r()
    lookat = np.array([(lo + hi) / 2.0, 0.0, 0.0]) / 1000.0
    span = max(hi - lo, 2.0 * y, 4.0 * r)
    return lookat, 1.25 * span / 1000.0


def close_frame(head_r_mm):
    """Distance in METRES that frames the head and a little of the cell
    around it.  Derived from the head's own swept radius, so the close
    view is the same view on a 300 mm cell and a metre one."""
    return CLOSE_SPANS * float(head_r_mm) / 1000.0


class CameraRig:
    """Keyboard camera for a `mujoco.viewer` passive handle.

    `home_frame` is (lookat_m, distance_m); `follow` an optional callable
    returning the world point in metres to keep centred while follow mode
    is on; `close` the distance the close-up uses, from close_frame().
    """

    def __init__(self, viewer, model, home_frame, follow=None, close=None):
        self.v, self.m, self._follow = viewer, model, follow
        self.lookat, self.dist = np.asarray(home_frame[0], float), float(home_frame[1])
        # a tenth of the cell is a poor close-up on a long truss and a fine
        # one on a short truss; pass close_frame(head radius) instead
        self.close = float(close) if close else 0.1 * self.dist
        self.following = False
        self.home()

    # ------------------------------------------------------------- frame
    def home(self):
        """Free camera, framed on the CELL from the side.

        Not the viewer's own default: that frames the whole model, and the
        model includes a floor plane a metre below the truss and racks
        beyond both ends, so the truss itself ends up a sliver.
        """
        with self.v.lock():
            c = self.v.cam
            c.type, c.fixedcamid = mujoco.mjtCamera.mjCAMERA_FREE, -1
            c.lookat[:] = self.lookat
            c.distance = self.dist
            c.azimuth, c.elevation = 90.0, -20.0
        self.following = False

    def _free(self):
        """Any control key is also the way off a fixed camera."""
        c = self.v.cam
        if c.type != mujoco.mjtCamera.mjCAMERA_FREE:
            c.type = mujoco.mjtCamera.mjCAMERA_FREE
            c.fixedcamid = -1

    # -------------------------------------------------------------- axes
    def _axes(self):
        """Ground-plane forward and right for the current azimuth.

        Panning on the GROUND plane, not the screen plane: looking down
        into the fixture, which is the useful view here, the screen-plane
        up vector degenerates and the pan direction swings wildly for a
        degree of elevation.  The ground plane is well behaved at every
        elevation, and along a truss laid on the x axis it is what you
        meant anyway.
        """
        az = np.radians(self.v.cam.azimuth)
        return (np.array([np.cos(az), np.sin(az), 0.0]),      # forward
                np.array([np.sin(az), -np.cos(az), 0.0]))     # right

    def _pan(self, d):
        with self.v.lock():
            self._free()
            self.v.cam.lookat[:] = np.asarray(self.v.cam.lookat) + \
                d * PAN_FRAC * max(self.v.cam.distance, 0.02)
        self.following = False       # you have taken the wheel

    # -------------------------------------------------------------- keys
    def key(self, code):
        """Handle one key press.  Returns True if it was ours."""
        fwd, right = self._axes()
        if code == K_UP:      self._pan(fwd)
        elif code == K_DOWN:  self._pan(-fwd)
        elif code == K_RIGHT: self._pan(right)
        elif code == K_LEFT:  self._pan(-right)
        elif code == K_PGUP:  self._pan(np.array([0.0, 0.0, 1.0]))
        elif code == K_PGDN:  self._pan(np.array([0.0, 0.0, -1.0]))
        elif code in (K_HOME, K_END):
            with self.v.lock():
                self._free()
                self.v.cam.azimuth += ORBIT_DEG * (1 if code == K_HOME else -1)
        elif code in (ord("-"), ord("=")):
            with self.v.lock():
                self._free()
                self.v.cam.distance *= ZOOM_STEP if code == ord("-") else 1 / ZOOM_STEP
        elif code in (ord("1"), ord("2"), ord("3")):
            # Elevation -89.5 rather than -90: at exactly -90 the camera's
            # own up vector is parallel to the view and the image rolls at
            # random.
            az, el = {ord("1"): (90.0, -89.5),      # top, down the cage's axis
                      ord("2"): (90.0, -20.0),      # from the side
                      ord("3"): (0.0, -15.0)}[code]  # end-on, along the truss
            with self.v.lock():
                self._free()
                self.v.cam.azimuth, self.v.cam.elevation = az, el
        elif code == ord("4"):
            self.following = self._follow is not None
            with self.v.lock():
                self._free()
                self.v.cam.distance = self.close
                self.v.cam.elevation = -25.0
        elif code == ord("."):
            # free FIRST: pressed on a fixed camera this used to leave you
            # there, following something you could not see move (found by
            # check_view, which asks every key the header's question)
            with self.v.lock():
                self._free()
            self.following = not self.following and self._follow is not None
        elif code == ord("0"):
            self.home()
        else:
            return False
        return True

    # -------------------------------------------------------------- tick
    def tick(self):
        """Call once per viewer sync.  Carries the lookat in follow mode."""
        if not self.following or self._follow is None:
            return
        p = self._follow()
        if p is None:
            return
        with self.v.lock():
            self._free()
            self.v.cam.lookat[:] = p
