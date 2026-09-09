"""Which OpenGL backend MuJoCo should use, decided by the platform.

MUJOCO_GL names the context MuJoCo renders through, and the legal values
differ by operating system: `osmesa` and `egl` are Linux-only, Windows
uses `wgl` and macOS `cgl`, both of which MuJoCo picks by itself.  A
script that writes `osmesa` unconditionally therefore does not merely
render slowly off a Windows machine -- it raises

    RuntimeError: invalid value for environment variable MUJOCO_GL: osmesa

from `import mujoco`, before any of its own code runs.  (Measured, on a
Windows 11 laptop with a working MuJoCo, against this package's own
demo.)

So the rule is: set nothing unless we are sure, respect anything the user
has set, and never force an offscreen context on a script that wants a
window.  Import this and call it BEFORE `import mujoco`.
"""
import os
import sys


def headless():
    """Ask for an offscreen context, if this platform needs to be told.

    Linux has no default GL context on a server, so name a software one.
    Windows and macOS have one, and MuJoCo finds it.
    """
    if os.environ.get("MUJOCO_GL"):
        return os.environ["MUJOCO_GL"]
    if sys.platform.startswith("linux"):
        os.environ["MUJOCO_GL"] = "osmesa"
    return os.environ.get("MUJOCO_GL", "")


def windowed():
    """A viewer needs the platform's own context: never an offscreen one.

    Returns False when a Linux session has no display, which is the case
    where a window cannot be opened at all and the caller should say so
    rather than fail inside the viewer.
    """
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True
