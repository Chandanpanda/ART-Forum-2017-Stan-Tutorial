"""The factory iPhone's camera as the app uses it: a pinhole at the
player's distance from the bat, the fold mirror that puts it there inside a
station, and where a band lands in its image.  Production, numpy only.

THE VIEW.  The accuracy study's camera (imu_fusion_sim.camera_anchor): the
phone's wide camera at 1080p, Phone.F_PX, at Rules.PLAY_Z from the bat,
finding two colour bands to Phone.SIGMA_PX on a still frame.  A station has
no 4 m to spare, so the phone looks at the bat through a plane mirror.  A
mirror leaves every ray's geometry as it was, with the lens at its image
behind the glass, so the bands see a camera at PLAY_Z along the unfolded
path; the app's arithmetic is the unfolded camera's, and mirror_size gives
the least glass that passes every ray from the bands to the lens.

THE FRAMES.  The camera's level frame is the study's world: X right as the
camera sees it, Y along its line of sight, Z up (the phone's own
accelerometer levels it; the station commissions where it stands against
the rig, View.c and View.R).  Pixels: u right, v down, from the top-left.
A View maps the station's frame (m) to pixels.
"""
from dataclasses import dataclass

import numpy as np

from ..spec import Phone, Rules


@dataclass(frozen=True)
class View:
    """The unfolded camera in the station's frame: its centre c (m) and its
    level frame R (columns X, Y, Z in station axes)."""
    c: np.ndarray
    R: np.ndarray
    f: float = Phone.F_PX
    w: int = Phone.W
    h: int = Phone.H

    def level(self, p):
        """(n, 3) m: station points in the camera's level frame."""
        return (np.atleast_2d(p) - self.c) @ self.R

    def project(self, p):
        """(n, 2) px of station points p (n, 3)."""
        q = self.level(p)
        if np.any(q[:, 1] <= 0.0):
            raise ValueError("a point behind the camera")
        return np.stack([self.w / 2.0 + self.f * q[:, 0] / q[:, 1], self.h / 2.0 - self.f * q[:, 2] / q[:, 1]], 1)

    def project_level(self, q):
        """(n, 2) px of points already in the level frame."""
        q = np.atleast_2d(q)
        return np.stack([self.w / 2.0 + self.f * q[:, 0] / q[:, 1], self.h / 2.0 - self.f * q[:, 2] / q[:, 1]], 1)

    def in_image(self, uv, margin=0.0):
        uv = np.atleast_2d(uv)
        return bool(np.all((uv[:, 0] >= margin) & (uv[:, 0] <= self.w - margin)
                           & (uv[:, 1] >= margin) & (uv[:, 1] <= self.h - margin)))


def view_at(target, up, sight, z=Rules.PLAY_Z):
    """The level View looking along the horizontal part of `sight` at
    `target` (station m) from Rules.PLAY_Z, `up` the station's true up."""
    up = np.asarray(up, float) / np.linalg.norm(up)
    s = np.asarray(sight, float)
    y = s - up * (s @ up)
    y /= np.linalg.norm(y)
    x = np.cross(y, up)
    R = np.stack([x, y, up], 1)
    return View(np.asarray(target, float) - z * y, R)


def mirror_size(view, pts, d1):
    """(across, up) m: the least plane mirror square to the line of sight at
    d1 from the lens that passes every ray from the station points pts (the
    bands wherever they go) to it.  Tilted to fold the path through an angle
    2a, its side across the fold grows by 1/cos(a); a 45 degree fold, by
    sqrt(2)."""
    q = view.level(pts)
    x = q[:, 0] * d1 / q[:, 1]
    z = q[:, 2] * d1 / q[:, 1]
    return float(x.max() - x.min()), float(z.max() - z.min())


def motion_px(speed, z=Rules.PLAY_Z):
    """px a band moves in one exposure, and across its own rows' readout,
    at `speed` m/s square to the line of sight: what a still frame's
    centroid noise does not include."""
    v_img = speed * Phone.F_PX / z
    return v_img * Phone.EXPOSURE_S, v_img * Phone.READOUT_S / Phone.H


def still_speed(band_px, z=Rules.PLAY_Z):
    """m/s square to the line of sight below which a band `band_px` rows
    tall moves less than its still centroid noise in one exposure and
    across its own readout: the frame is as good as a still one."""
    t = max(Phone.EXPOSURE_S, band_px * Phone.READOUT_S / Phone.H)
    return Phone.SIGMA_PX * z / (Phone.F_PX * t)
