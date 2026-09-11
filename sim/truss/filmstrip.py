"""Frames from a run, and a contact sheet to read them on.

WHY THIS IS A MODULE AND NOT A FLAG SOMEWHERE.  A simulation is not
reviewed until frames from it have been read.  The mount phase once
reported every rod placed within 0.45 mm and 0.0 degrees of plan, the
inspector passed the truss, 955 of 955 ops ran -- and the thing it built
was a mess.  Every one of those numbers was measuring the model against
itself.  A number can say the machine went where the plan said; it cannot
say the plan was the design.

    sweep   0.1 fps -- one frame per ten seconds of simulated time, over the
            whole run, tiled into one sheet.  Cheap, and it is what catches
            "this phase ran and achieved nothing".
    window  1 fps over a stretch that looked wrong on the sweep.

Both are in SIMULATED time, not wall time: a run that takes forty minutes
of wall clock for twenty of simulated time must not produce twice the
frames when the machine is busy.
"""
import os

import numpy as np


class Filmstrip:
    """Captures frames at a rate in SIMULATED seconds, optionally only
    inside a window."""

    def __init__(self, renderer, camera, fps=0.1, window=None, limit=4000):
        self.r, self.cam = renderer, camera
        self.period = 1.0 / float(fps) if fps > 0 else 0.0
        self.window = window
        self.limit = limit
        self.frames = []
        self.times = []
        self._next = None

    def maybe(self, sim_t, data):
        """Call once a tick with the sim time; captures when it is due."""
        if self.window is not None and not (self.window[0] <= sim_t <= self.window[1]):
            return False
        if self._next is None:
            self._next = sim_t
        if sim_t < self._next or len(self.frames) >= self.limit:
            return False
        self.r.update_scene(data, self.cam)
        self.frames.append(self.r.render().copy())
        self.times.append(float(sim_t))
        self._next = sim_t + self.period
        return True


def contact_sheet(frames, path, cols=None, scale=1, labels=None, pad=4,
                  bg=(24, 24, 26)):
    """Tile frames into one image, in order, left to right and top down.

    Reading fifty frames one at a time is how a sweep stops being done; on
    one sheet it is a glance."""
    from PIL import Image, ImageDraw
    if not frames:
        return None
    n = len(frames)
    cols = cols or max(1, int(np.ceil(np.sqrt(n * 1.4))))
    rows = int(np.ceil(n / cols))
    h, w, _ = frames[0].shape
    w2, h2 = max(1, w // scale), max(1, h // scale)
    sheet = Image.new("RGB", (cols * (w2 + pad) + pad, rows * (h2 + pad) + pad), bg)
    dr = ImageDraw.Draw(sheet)
    for i, f in enumerate(frames):
        im = Image.fromarray(f)
        if scale != 1:
            im = im.resize((w2, h2))
        x = pad + (i % cols) * (w2 + pad)
        y = pad + (i // cols) * (h2 + pad)
        sheet.paste(im, (x, y))
        if labels is not None and i < len(labels):
            dr.text((x + 3, y + 2), str(labels[i]), fill=(255, 240, 120))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    sheet.save(path)
    return path
