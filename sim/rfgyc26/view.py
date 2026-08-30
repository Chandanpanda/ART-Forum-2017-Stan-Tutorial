"""Camera controls for the passive viewer: translate, orbit, zoom, follow.

WHY THIS EXISTS.  MuJoCo's viewer already has all three Blender operations on
the mouse -- left-drag orbits, right-drag translates, scroll zooms -- but ONLY
on the FREE camera.  The model carries four fixed viewpoints (`field`, `lab`,
`quar` and the robot's own `A_chase`), and one press of `[` or `]` puts you on
one of them, where the mouse does nothing at all: no pan, no orbit, no zoom.
There is no on-screen sign that this has happened.  That is the trap.

So this module does two things.  It gives the translate/orbit/zoom trio a
keyboard binding as well as a mouse one, and it makes every one of those keys
snap back to the free camera first -- so any control key is also the way out.

IT CANNOT USE LETTER KEYS.  The viewer binds every letter A-Z to a
visualisation flag (mjVISSTRING and mjRNDSTRING, checked at runtime by
`taken_letters()` below), so a letter here would fire twice: once for us and
once to toggle wireframe, or shadows, or contact forces.  Everything below is
on arrows, digits and punctuation, which the viewer leaves alone.

The steps scale with the zoom distance, which is what makes a single set of
bindings usable both across the 2 m field and inside the 60 mm magazine: close
in, a tap moves you a few millimetres; pulled back, it moves you a quarter of
the field.
"""
import numpy as np
import mujoco

from .params import Field

# GLFW key codes.  Spelled out rather than imported: mujoco.viewer pulls glfw in
# itself, but this module is also imported by scripts that never open a window.
K_RIGHT, K_LEFT, K_DOWN, K_UP = 262, 263, 264, 265
K_PGUP, K_PGDN, K_HOME, K_END = 266, 267, 268, 269

PAN_FRAC  = 0.09      # of the zoom distance, per press
ZOOM_STEP = 1.18      # per press
ORBIT_DEG = 6.0       # per press

HELP = """  camera:  arrows pan          Page Up/Down raise, lower
           Home/End orbit      - and = zoom out, in
           1 top  2 south  3 west  4 close on the robot
           . follow the robot   0 free camera, whole field
           (mouse still works on the free camera: left-drag orbits,
            right-drag pans, scroll zooms -- but not on a fixed camera,
            which is what [ and ] switch to)"""


def taken_letters():
    """The letters the viewer has already bound, and to what.

    Not used by the rig -- it is here so the claim in the docstring above can be
    checked against the installed MuJoCo rather than believed.
    """
    out = {}
    for tbl in (mujoco.mjVISSTRING, mujoco.mjRNDSTRING):
        for row in tbl:
            row = list(row)
            if len(row) >= 3 and row[2].strip():
                out.setdefault(row[2].strip().upper(), []).append(row[0])
    return out


class CameraRig:
    """Keyboard camera for a `mujoco.viewer` passive handle.

    `follow` is an optional callable returning the world point (metres) to keep
    centred while follow mode is on.
    """

    def __init__(self, viewer, model, follow=None):
        self.v, self.m, self._follow = viewer, model, follow
        self.following = False
        self.home()

    # ------------------------------------------------------------------ frame
    def home(self):
        """Free camera, framed on the FIELD.

        Not mjv_defaultFreeCamera, which is what the viewer opens with: that
        frames the whole model, and the model is a 2 m arena with a hospital and
        two PCC corners in it that Agent A never visits.  It puts the camera
        3.6 m out and the 1.14 x 1.18 m of field this robot actually works in
        ends up a third of the screen.  Frame the field instead, from the south,
        which is the orientation every drawing in the spec uses.
        """
        with self.v.lock():
            c = self.v.cam
            c.type, c.fixedcamid = mujoco.mjtCamera.mjCAMERA_FREE, -1
            c.lookat[:] = [Field.W/2000.0, Field.H/2000.0, 0.0]
            c.distance = 1.45*max(Field.W, Field.H)/1000.0
            c.azimuth, c.elevation = 90.0, -55.0
        self.following = False

    def _free(self):
        """Any control key is also the way off a fixed camera (see the header)."""
        c = self.v.cam
        if c.type != mujoco.mjtCamera.mjCAMERA_FREE:
            c.type = mujoco.mjtCamera.mjCAMERA_FREE
            c.fixedcamid = -1

    # ------------------------------------------------------------------- axes
    def _axes(self):
        """Ground-plane forward and right for the current azimuth.

        Panning on the GROUND plane, not in the screen plane: looking almost
        straight down -- which is the useful view of a field -- the screen-plane
        'up' vector degenerates, and the pan direction then swings wildly for a
        degree of elevation change.  The ground plane is well behaved at every
        elevation, and on a flat field it is what you meant anyway.
        """
        az = np.radians(self.v.cam.azimuth)
        return (np.array([np.cos(az), np.sin(az), 0.0]),      # forward
                np.array([np.sin(az), -np.cos(az), 0.0]))     # right

    def _pan(self, d):
        with self.v.lock():
            self._free()
            self.v.cam.lookat[:] = np.asarray(self.v.cam.lookat) + \
                d * PAN_FRAC * max(self.v.cam.distance, 0.05)
        self.following = False       # you have taken the wheel

    # ------------------------------------------------------------------- keys
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
                self.v.cam.distance *= ZOOM_STEP if code == ord("-") else 1/ZOOM_STEP
        elif code in (ord("1"), ord("2"), ord("3")):
            # Elevation -89.5 rather than -90: at exactly -90 the camera's own
            # up vector is parallel to the view and the image rolls at random.
            az, el = {ord("1"): (90.0, -89.5),      # top
                      ord("2"): (90.0, -22.0),      # from the south
                      ord("3"): (0.0, -22.0)}[code] # from the west
            with self.v.lock():
                self._free()
                self.v.cam.azimuth, self.v.cam.elevation = az, el
        elif code == ord("4"):
            self.following = True
            with self.v.lock():
                self._free()
                self.v.cam.distance = 0.55
                self.v.cam.elevation = -35.0
        elif code == ord("."):
            self.following = not self.following and self._follow is not None
        elif code == ord("0"):
            self.home()
        else:
            return False
        return True

    # ------------------------------------------------------------------- tick
    def tick(self):
        """Call once per viewer sync.  Carries the lookat along in follow mode."""
        if not self.following or self._follow is None:
            return
        p = self._follow()
        if p is None:
            return
        with self.v.lock():
            self._free()
            self.v.cam.lookat[:] = p
