"""The station measures itself: nothing about where things are is typed in.

The plan: "each station calibrates itself against fiducials".  A built
station differs from its drawing by everything spec.Build lists -- each
tool on its plate, the camera on its carriage (offset, turned, tilted),
each dock and the fixture on the floor, each home switch on its rail --
and the controller only ever has the drawing.  Commissioning turns the
drawing into a StationFrame that is right about the station as built,
using nothing but the station's own sensors:

  1. home every axis
  2. the camera: a flat fiducial on the fixture, looked at with the head
     moved a known distance each way, gives the image's scale and turn
     (the head's own steps are the ruler); looked at again from lower down,
     how far its axis drifts with range -- the lens's tilt
  3. the reference pin's top, seen from above: where the pin is
  4. each region: its three fiducials, seen, give its place, its turn,
     and the head's scale along x and y across it
  5. each tool, against that pin: its tip's x and y by touching the pin
     from both sides of each -- the midpoint of two touches is the pin's
     centre whatever the tool's width, so no tool size is assumed -- then
     its tip's height by touching the pin's top.  The pogo block's pins
     are springs too soft for a force to find where they meet anything,
     so its height is found by its own sensor: lowered onto its pad until
     every pin closes, which they do at Pogo.WORK[0] of compression

THE FRAME IS THE CAMERA'S.  Everything is measured in believed head
coordinates, and where the lens sits on the head is the one thing taken
from the drawing: it defines where "the station" is, and the camera's
axis runs down from it at the lean the camera measures of itself.  An
error in it moves every tool and every region alike, and a tool sent to
a point the camera saw lands on it whatever that error was.

The surfaces' heights are the drawing's (regions are flat and level to
Build's tolerances, which do not include height); the tools' heights are
measured against the pin's top, so a home switch that trips high or low
is absorbed with them.
"""
from dataclasses import dataclass, field, replace
from math import atan2, sqrt

import numpy as np

from ..spec import (HeadCam, Artefact, Gripper2, Vacuum, Magnet, Pusher, Pogo, Spindle, Build, Module,
                    ToolSlide, Fiducial)
from . import spec as SS


# ================================================================ FRAME
@dataclass
class StationFrame:
    """What the executor knows about where things are."""
    tips: dict                 # kind -> (x, y, z): the EXTENDED tip from the head point
    c0: np.ndarray             # (x, y): the camera's axis from the head point, at range R0
    cam_z: float               # the lens's height above the head point
    M: np.ndarray              # 2x2: station mm per image px, at range R0
    tilt: np.ndarray           # (x, y) mm the axis moves per mm of range beyond R0
    R0: float
    regions: dict              # name -> (R 2x2, t 2): station point = R drawn + t
    source: str = "drawing"
    notes: dict = field(default_factory=dict)

    @classmethod
    def drawing(cls, mod):
        """The frame the drawing gives: what the controller has until it has
        measured anything."""
        tips = {k: np.array(mod.tip(k, True), float) for k in mod.head.tools}
        R0 = mod.look["fixture"]
        mpp = R0 * HeadCam.PIXEL / HeadCam.F
        # the image lies long side along x, rows running down -y
        M = mpp * np.array([[1.0, 0.0], [0.0, -1.0]])
        regs = {k: (np.eye(2), np.zeros(2)) for k in mod.regions}
        return cls(tips, np.array(mod.head.camera[:2], float), float(mod.head.camera[2]), M, np.zeros(2),
                   float(R0), regs)

    # --------------------------------------------------------------- camera
    def range(self, z_head, z_surface):
        return z_head + self.cam_z - z_surface

    def to_station(self, head_xy, z_head, uv, z_surface):
        """The station (x, y) of what is seen at pixel uv."""
        R = self.range(z_head, z_surface)
        d = np.asarray(uv, float) - np.array([HeadCam.W / 2.0, HeadCam.H / 2.0])
        return np.asarray(head_xy, float) + self.c0 + self.tilt * (R - self.R0) + (R / self.R0) * (self.M @ d)

    def camera_head(self, p, z_head, z_surface):
        """The head (x, y) that puts station point p at the image's centre."""
        R = self.range(z_head, z_surface)
        return tuple(np.asarray(p[:2], float) - self.c0 - self.tilt * (R - self.R0))

    def predict_uv(self, head_xy, z_head, p, z_surface):
        R = self.range(z_head, z_surface)
        d = np.asarray(p[:2], float) - np.asarray(head_xy, float) - self.c0 - self.tilt * (R - self.R0)
        return np.linalg.solve((R / self.R0) * self.M, d) + np.array([HeadCam.W / 2.0, HeadCam.H / 2.0])

    # ---------------------------------------------------------------- tools
    def tip(self, kind, extended=True):
        t = self.tips[kind].copy()
        if not extended:
            t[2] += ToolSlide.STROKE
        return t

    def head_for(self, kind, p, extended=True):
        """The head point that puts this tool's tip on station point p."""
        return tuple(np.asarray(p, float) - self.tip(kind, extended))

    # -------------------------------------------------------------- regions
    def place(self, region, p):
        """Where drawn point p (x, y[, z]) of a region is, in the station
        frame as measured."""
        R, t = self.regions.get(region, (np.eye(2), np.zeros(2)))
        q = R @ np.asarray(p[:2], float) + t
        return (float(q[0]), float(q[1])) + tuple(float(v) for v in p[2:])


# ===================================================== TOOL GEOMETRY
def contact_half(kind):
    """(x, y) half-size of what a tool touches a pin with, drawn: only to
    start a touch clear of it; the touch itself measures the truth."""
    if kind == "gripper":                      # jaws closed: two jaws side by side
        return Gripper2.JAW[0] / 2.0, Gripper2.JAW[1]
    if kind == "vacuum":
        return Vacuum.CUP_D / 2.0, Vacuum.CUP_D / 2.0
    if kind == "magnet":
        return Magnet.BODY[0] / 2.0, Magnet.BODY[1] / 2.0
    if kind == "pusher":
        return Pusher.BODY[0] / 2.0, Pusher.BODY[1] / 2.0
    if kind == "pogo":
        # its block, not its plungers: they are 0.9 mm of spring steel and
        # sit inside the block's footprint, TRAVEL below it, so the block's
        # faces meet the pin's sides first at the same height
        return Pogo.BODY[0] / 2.0, Pogo.BODY[1] / 2.0
    if kind == "spindle":
        return Spindle.COLLET_D / 2.0, Spindle.COLLET_D / 2.0
    raise ValueError(kind)


def _pad_at(mod):
    """The pogo block's pad, where test_plate lays it, drawn: on the fixture."""
    from .mjcf import test_plate
    return next(p.at for p in test_plate(mod) if p.kind == "pad")


def fit_pose(drawn, seen):
    """(A, t) taking drawn points onto seen ones, least squares: the
    affine map three fiducials fix -- place, turn, and a scale along each
    axis and their squareness.  The scales are not the region's: the
    head's steps are the ruler, and a rack or a screw a few kelvin warmer
    than when it was cut is a longer ruler, each by its own amount.  Fitted
    rigid, two fiducials split that growth between their ends -- 0.14 mm
    at each end of the adult's 1.7 m fixture, for 160 ppm -- and sent every
    tool off by it; fitted as a similarity, they took the mean growth and
    left the difference at the corners nobody looked at."""
    a, b = np.asarray(drawn, float), np.asarray(seen, float)
    X = np.linalg.lstsq(np.hstack([a, np.ones((len(a), 1))]), b, rcond=None)[0]
    return X[:2].T, X[2]


# ======================================================== COMMISSIONING
def commission(ex, mod, log=None):
    """A generator: run it on the executor's clock.  Returns (frame, notes)
    and leaves the measured frame on the executor."""
    say = log or (lambda s: None)
    fr = StationFrame.drawing(mod)
    ex.frame = fr
    notes = {}

    d = yield from ex.home()
    if not d.ok:
        return None, {"failed": d.why}

    # ------------------------------------------------- 2. the camera
    fx = mod.regions["fixture"]
    fid = fx.fiducial_points()[0]
    z_top = fx.top
    d = yield from ex.look(fid[:2], z_top, over="fixture")
    if not d.ok:
        return None, {"failed": "camera: " + d.why}
    H0 = np.array(d.data["head"])
    uv0 = np.array(d.data["uv"])
    # the head's own steps are the ruler: half the image's short side
    mpp = fr.R0 * HeadCam.PIXEL / HeadCam.F
    D = 0.5 * min(HeadCam.W, HeadCam.H) / 2.0 * mpp
    dH, duv = [], []
    for dx, dy in ((D, 0.0), (-D, 0.0), (0.0, D), (0.0, -D)):
        d = yield from ex.move({"x": H0[0] + dx, "y": H0[1] + dy}, op="camera ruler")
        if not d.ok:
            return None, {"failed": "camera ruler: " + d.why}
        near = fr.predict_uv(H0 + np.array([dx, dy]), ex.hal.at("z"), fr.to_station(H0, ex.hal.at("z"), uv0, z_top),
                             z_top)
        uv = ex.vision.locate(tuple(near))
        if uv is None:
            return None, {"failed": "camera ruler: fiducial left the view"}
        dH.append([ex.hal.at("x") - H0[0], ex.hal.at("y") - H0[1]])
        duv.append(np.array(uv) - uv0)
    dH, duv = np.array(dH).T, np.array(duv).T
    R_ruler = fr.range(ex.hal.at("z"), z_top)
    M = -(dH @ np.linalg.pinv(duv)) * (fr.R0 / R_ruler)
    fr = replace(fr, M=M)
    ex.frame = fr
    # the tilt: the same fiducial from as low as the head may go, same x, y
    d = yield from ex.move({"x": H0[0], "y": H0[1]}, op="camera ruler")
    uv_hi = np.array(ex.vision.locate(tuple(uv0)))
    z_hi = ex.hal.at("z")
    d = yield from ex.move({"z": mod.z_safe}, op="camera tilt")
    if not d.ok:
        return None, {"failed": "camera tilt: " + d.why}
    uv_lo = ex.vision.locate(tuple(uv0))
    if uv_lo is None:
        return None, {"failed": "camera tilt: fiducial left the view"}
    c = np.array([HeadCam.W / 2.0, HeadCam.H / 2.0])
    R_hi, R_lo = fr.range(z_hi, z_top), fr.range(mod.z_safe, z_top)
    tilt = ((R_lo / fr.R0) * (M @ (np.array(uv_lo) - c)) - (R_hi / fr.R0) * (M @ (uv_hi - c))) / (R_hi - R_lo)
    # the drawing knows where the lens sits on its bracket, not where its
    # axis lands: a 0.3 degree lean puts that 1.4 mm away at the range the
    # camera works from.  Carry the drawn lens point down the axis as
    # measured
    fr = replace(fr, tilt=tilt, c0=fr.c0 + tilt * fr.R0)
    ex.frame = fr
    yield from ex.lift()
    notes["camera"] = {"mm_per_px": float(sqrt(abs(np.linalg.det(M)))),
                       "turn_deg": float(np.degrees(atan2(M[1, 0], M[0, 0]))), "tilt": tuple(tilt)}
    say("    camera: %.5f mm/px, turned %.3f deg, axis drifts %.4f, %.4f mm per mm of range"
        % (notes["camera"]["mm_per_px"], notes["camera"]["turn_deg"], tilt[0], tilt[1]))

    # ------------------------------------------------- 3. the reference pin
    pin = mod.regions["pin"]
    pin_top = pin.top + Artefact.PIN_H
    d = yield from ex.look(pin.centre, pin_top, over="pin")
    if not d.ok:
        return None, {"failed": "pin: " + d.why}
    P = np.array(d.data["p"])
    notes["pin"] = tuple(P)
    say("    pin seen at %.3f, %.3f" % tuple(P))

    # ---------------------------------------------------- 4. each region
    regs = dict(fr.regions)
    notes["regions"] = {}
    for name, r in mod.regions.items():
        pts = r.fiducial_points()
        if not pts:
            continue
        seen = []
        for p in pts:
            d = yield from ex.look(p[:2], r.top, over=name)
            if not d.ok:
                return None, {"failed": "%s fiducial: %s" % (name, d.why)}
            seen.append(d.data["p"])
        regs[name] = fit_pose([p[:2] for p in pts], seen)
        R, t = regs[name]
        notes["regions"][name] = {"turn_deg": float(np.degrees(atan2(R[1, 0], R[0, 0]))), "shift": tuple(t),
                                  "scale_ppm": tuple(float((np.hypot(*R[:, i]) - 1.0) * 1e6) for i in (0, 1))}
    # the pin stands on the fixture, so it moves with it
    if "pin" in mod.regions and "fixture" in regs:
        regs["pin"] = regs["fixture"]
    fr = replace(fr, regions=regs)
    ex.frame = fr

    # ------------------------------------------------------ 5. each tool
    gz, gxy = SS.gaps()
    tips = dict(fr.tips)
    notes["tools"] = {}
    for k in mod.head.tools:
        drawn = fr.tips[k]
        d = yield from ex.slide(k, True)
        if not d.ok:
            return None, {"failed": "%s: %s" % (k, d.why)}
        if k == "gripper":
            ex.hal.grip(k)                     # the jaws closed on nothing: one block to touch with
            yield from ex._wait(Gripper2.CLOSE_S)
        # its x and y, on the pin's sides at half its height (the height
        # still the drawing's: the pin is far taller than that is wrong by)
        tip_z = drawn[2]
        above = P - drawn[:2]
        half = contact_half(k)
        z_side = pin_top - Artefact.PIN_H / 2.0 - tip_z
        z_over = pin_top + 2.0 * gz - tip_z
        centre = {}
        for ax, i in (("x", 0), ("y", 1)):
            hit = {}
            for sgn in (+1, -1):
                start = above.copy()
                start[i] -= sgn * (Artefact.PIN_D / 2.0 + half[i] + gxy)
                for step in (ex.lift(max(z_over, ex.hal.at("z"))), ex.move({"x": start[0], "y": start[1]}, op="beside"),
                             ex.move({"z": z_side}, frac=0.5, op="down beside")):
                    d = yield from step
                    if not d.ok:
                        return None, {"failed": "%s beside the pin: %s" % (k, d.why)}
                t = yield from ex.touch(ax, sgn, 2.0 * gxy, op="touch %s %s%s" % (k, "+" if sgn > 0 else "-", ax))
                if not t.ok:
                    return None, {"failed": "%s side: %s" % (k, t.why)}
                hit[sgn] = t.data["at"]
            centre[ax] = (hit[+1] + hit[-1]) / 2.0
            notes["tools"].setdefault(k, {})["width_" + ax] = abs(hit[-1] - hit[+1]) - Artefact.PIN_D
        tip_xy = np.array([P[0] - centre["x"], P[1] - centre["y"]])
        d = yield from ex.lift(z_over)
        # its height: on the pin's top, or the pogo block on its pad
        if k == "pogo":
            pad = mod.regions["fixture"].top + Artefact.PAD_T
            pxy = np.array(fr.place("fixture", _pad_at(mod)))
            over = pxy - tip_xy
            d = yield from ex.travel((over[0], over[1], pad + gz - drawn[2]))
            if not d.ok:
                return None, {"failed": "pogo over its pad: %s" % d.why}
            t = yield from ex.touch_contacts(2.0 * gz + Pogo.WORK[1], op="touch pogo pad")
            if not t.ok:
                return None, {"failed": "pogo pad: %s" % t.why}
            tip_z = pad - Pogo.WORK[0] - t.data["at"]
        else:
            over = P - tip_xy
            d = yield from ex.travel((over[0], over[1], pin_top + gz - drawn[2]))
            if not d.ok:
                return None, {"failed": "%s over the pin: %s" % (k, d.why)}
            t = yield from ex.touch("z", -1, 2.0 * gz, op="touch %s top" % k)
            if not t.ok:
                return None, {"failed": "%s top: %s" % (k, t.why)}
            tip_z = pin_top - t.data["at"]
        tips[k] = np.array([tip_xy[0], tip_xy[1], tip_z])
        notes["tools"][k].update({"tip": tuple(tips[k]), "drawn": tuple(drawn)})
        say("    %-8s tip %8.3f %8.3f %8.3f  (drawn %8.3f %8.3f %8.3f)" % ((k,) + tuple(tips[k]) + tuple(drawn)))
        if k == "gripper":
            ex.hal.release(k)
            yield from ex._wait(Gripper2.CLOSE_S)
        yield from ex.lift()
        yield from ex.slide(k, False)
    fr = replace(fr, tips=tips)
    ex.frame = fr

    fr = replace(fr, source="commissioned", notes=notes)
    ex.frame = fr
    return fr, notes
