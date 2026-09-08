"""Pre-wind localisation: where the joint is, relative to the ring.

Two implementations of hal.VisionHAL with one contract, as in the
competition rig:

  * ModelVision -- the synthetic camera.  Geometry plus the error budget:
    the chord's true offset from the ring, a camera-to-ring calibration
    bias drawn ONCE per run (a bracket is wrong by the same amount all
    shift; redrawing it per look would average it away and flatter the
    cell), and per-look noise from the edge fit.  Fast; the regression
    double.
  * PixelVision -- frames from the head camera through a pipeline that
    finds the chord's two edges and the mitre line.  Slow; the release
    gate (check_vision).

What is measured, and why it is enough.  The head is over the joint to
the gantry's belief; the fixture's notches put the chord within a
millimetre of where the geometry says.  The ring needs the chord under
its centre to a fraction of its seating clearance, so the look has to
resolve tenths of a millimetre at 90 mm range, which 20 pixels across a
3 mm chord do easily.  The bracket, not the sensor, is the budget.
"""
import numpy as np

from . import hal
from .spec import Vision


class ModelVision(hal.VisionHAL):
    def __init__(self, cell, rng=None):
        self.cell = cell
        self.rng = rng or np.random.default_rng(1)
        self.bias = self.rng.normal(0.0, Vision.EXT_SIGMA, 2)
        self.sigma = Vision.LOOK_SIGMA

    def locate(self, joint_index):
        j = self.cell.geom.joints[joint_index]
        centre = self.cell.ring_centre()
        cj, _n = self.cell.geom.joint_at(j, self.cell.cage_truth())
        # in view: the chord within the frame's footprint at its range
        span = Vision.mm_per_px(Vision.range_nominal()) * Vision.W / 2.0
        if abs(cj[0] - centre[0]) > span or abs(cj[1] - centre[1]) > span * 0.5:
            return None
        dx = float(cj[0] - centre[0]) + self.bias[0] + float(self.rng.normal(0.0, self.sigma))
        dy = float(cj[1] - centre[1]) + self.bias[1] + float(self.rng.normal(0.0, self.sigma))
        return dx, dy


class PixelVision(hal.VisionHAL):
    """The rendered path.

    The camera is a calibrated pinhole at a known pose on the head, so a
    pixel is a ray in the head's frame and a point on the chord's axis
    plane is where that ray meets a known height (the head's believed
    height less the plan's lift).  The image supplies two things: the
    chord's centreline, a dark stripe up the frame, and the mitre, where
    the diagonals' centrelines meet it.  Both are searched for NEAR WHERE
    THE PLAN PREDICTS THEM -- the frame also holds pins, cradles and the
    cage's end plate -- and what comes back is the measured minus the
    predicted image feature, back-projected, added to the predicted
    offset.  So a bias in the camera's bracket shows up as a bias in the
    look, which is what the model camera assumes and check_vision pins.
    """

    def __init__(self, camera, cell, rng=None):
        self.cam = camera
        self.cell = cell
        self.rng = rng or np.random.default_rng(2)
        self.bias = self.rng.normal(0.0, Vision.EXT_SIGMA, 2)
        self.last = None

    # ------------------------------------------------------------ rays
    def _ray(self, cal, u, v):
        d = np.array([(u - cal["cx"]) / cal["f"], -(v - cal["cy"]) / cal["f"], -1.0])
        d = cal["R"] @ d
        return d / np.linalg.norm(d)

    def _on_plane(self, cal, u, v, z):
        d = self._ray(cal, u, v)
        return cal["pos"] + ((z - cal["pos"][2]) / d[2]) * d

    def _pixel(self, cal, p):
        q = cal["R"].T @ (np.asarray(p, float) - cal["pos"])
        return (cal["cx"] + cal["f"] * q[0] / -q[2], cal["cy"] - cal["f"] * q[1] / -q[2])

    # ---------------------------------------------------------- pixels
    @staticmethod
    def _otsu(vals):
        hist, edges = np.histogram(vals, bins=64)
        mids = (edges[:-1] + edges[1:]) / 2.0
        w0 = np.cumsum(hist).astype(float)
        w1 = w0[-1] - w0
        m0 = np.cumsum(hist * mids) / np.maximum(w0, 1e-9)
        m1 = (np.sum(hist * mids) - np.cumsum(hist * mids)) / np.maximum(w1, 1e-9)
        between = w0 * w1 * (m0 - m1) ** 2
        return float(mids[int(np.argmax(between))])

    @staticmethod
    def _runs(dark):
        cols = np.where(dark)[0]
        if len(cols) == 0:
            return []
        runs, start = [], cols[0]
        for a, b in zip(cols[:-1], cols[1:]):
            if b != a + 1:
                runs.append((start, a)); start = b
        runs.append((start, cols[-1]))
        return runs

    def _centre_near(self, g, thr, v, u_pred, tol_px, w_min, w_max, avoid=None):
        """Sub-pixel centre of the dark run on row v nearest column u_pred,
        of plausible width, not the one at `avoid`; None if none."""
        v = int(round(v))
        if v < 0 or v >= g.shape[0]:
            return None
        line = g[v]
        best = None
        for a, b in self._runs(line < thr):
            w = b - a + 1
            if w < w_min or w > w_max:
                continue
            c = (a + b) / 2.0
            if avoid is not None and abs(c - avoid[0]) < avoid[1]:
                continue
            if abs(c - u_pred) > tol_px:
                continue
            if best is None or abs(c - u_pred) < abs(best[0] - u_pred):
                def cross(i0, i1):
                    y0, y1 = line[i0], line[i1]
                    return i0 + (thr - y0) / (y1 - y0) if y1 != y0 else float(i0)
                left = cross(a - 1, a) if a > 0 else float(a)
                right = cross(b, b + 1) if b + 1 < len(line) else float(b)
                best = ((left + right) / 2.0, right - left)
        return best

    @staticmethod
    def _fit(us, vs):
        A = np.stack([np.ones(len(vs)), np.asarray(vs, float)], axis=1)
        return np.linalg.lstsq(A, np.asarray(us, float), rcond=None)[0]   # u = a + b v

    @staticmethod
    def _fit_uv(us, vs):
        """A line as v = c + d u (for the diagonals, which cross rows)."""
        A = np.stack([np.ones(len(us)), np.asarray(us, float)], axis=1)
        return np.linalg.lstsq(A, np.asarray(vs, float), rcond=None)[0]

    @staticmethod
    def _meet(chord, diag):
        """Where u = a + b v meets v = c + d u."""
        a, b = chord
        c, d = diag
        v = (c + d * a) / (1.0 - d * b)
        return a + b * v, v

    # ------------------------------------------------------------ look
    def locate(self, joint_index):
        img, _t = self.cam.frame()
        cal = self.cam.calib()
        g = img.astype(float).mean(axis=2)
        H, W = g.shape
        geom = self.cell.geom
        t = geom.t
        j = geom.joints[joint_index]
        theta = self.cell.theta()
        head = np.array([self.cell.at("x"), self.cell.at("y"), self.cell.at("z")])
        cj, _n = geom.joint_at(j, theta)
        pj = cj - head                                    # the joint, head frame
        rng_mm = float(np.linalg.norm(pj - cal["pos"]))
        mm_px = rng_mm / cal["f"]
        # ---- the chord: dark runs near its predicted line, a band's
        # length either side of the joint
        u_pred = lambda s: self._pixel(cal, pj + np.array([s, 0.0, 0.0]))
        thr_rows = [u_pred(s) for s in np.linspace(-t.band - 4.0, t.band + 4.0, 41)]
        r_lo, r_hi = int(min(v for _u, v in thr_rows)), int(max(v for _u, v in thr_rows))
        c_lo, c_hi = int(min(u for u, _v in thr_rows) - 6.0 / mm_px), int(max(u for u, _v in thr_rows) + 6.0 / mm_px)
        r_lo, r_hi, c_lo, c_hi = max(r_lo, 0), min(r_hi, H - 1), max(c_lo, 0), min(c_hi, W - 1)
        if r_hi - r_lo < 8 or c_hi - c_lo < 8:
            return None
        thr = self._otsu(g[r_lo:r_hi + 1, c_lo:c_hi + 1].ravel())
        w_c = t.d_chord / mm_px
        us, vs = [], []
        for u0, v0 in thr_rows:
            got = self._centre_near(g, thr, v0, u0, 4.0 / mm_px, 0.5 * w_c, 2.5 * w_c)
            if got is not None:
                us.append(got[0]); vs.append(v0)
        if len(us) < 8:
            return None
        chord = self._fit(us, vs)
        chord_pred = self._fit(*zip(*[(u, v) for u, v in thr_rows]))
        # ---- the diagonals: dark runs near each predicted centreline, 3 to
        # 12 mm out from the joint, avoiding the chord's own run
        w_d = t.d_diag / mm_px
        dxs = []
        for ri in j.diags:
            rod = geom.rods[ri]
            from .geometry import rot_x
            Rm = rot_x(theta)
            p0, p1 = Rm @ rod.p0 - head, Rm @ rod.p1 - head
            if np.linalg.norm(p0 - pj) > np.linalg.norm(p1 - pj):
                p0, p1 = p1, p0
            u_d = (p1 - p0) / np.linalg.norm(p1 - p0)
            pu, pv, mu, mv = [], [], [], []
            for s_mm in np.linspace(3.0, 12.0, 19):
                q = p0 + u_d * s_mm
                u0, v0 = self._pixel(cal, q)
                got = self._centre_near(g, thr, v0, u0, 3.0 / mm_px, 0.4 * w_d, 3.0 * w_d,
                                        avoid=(chord[0] + chord[1] * v0, 0.75 * w_c))
                pu.append(u0); pv.append(v0)
                if got is not None:
                    mu.append(got[0]); mv.append(v0)
            if len(mu) < 6:
                continue
            fit = self._fit_uv(mu, mv)
            resid = float(np.sqrt(np.mean((np.array(mv) - (fit[0] + fit[1] * np.array(mu))) ** 2)))
            meas = self._meet(chord, fit)
            pred = self._meet(chord_pred, self._fit_uv(pu, pv))
            dm = self._on_plane(cal, meas[0], meas[1], float(pj[2]))
            dp = self._on_plane(cal, pred[0], pred[1], float(pj[2]))
            dxs.append((float(dm[0] - dp[0]), resid, len(mu)))
        if not dxs:
            return None
        # two diagonals that agree are averaged; two that do not -- one of
        # them caught a cradle's edge -- defer to the cleaner fit
        if len(dxs) == 2 and abs(dxs[0][0] - dxs[1][0]) > 0.5:
            dxs = [min(dxs, key=lambda r: r[1] / r[2])]
        dxs = [r[0] for r in dxs]
        # ---- dy from the chord line at the joint's predicted row
        v_j = self._pixel(cal, pj)[1]
        dm = self._on_plane(cal, chord[0] + chord[1] * v_j, v_j, float(pj[2]))
        dp = self._on_plane(cal, chord_pred[0] + chord_pred[1] * v_j, v_j, float(pj[2]))
        dy = float(dm[1] - dp[1])
        dx = float(np.mean(dxs))
        self.last = (dx, dy, len(dxs), thr)
        return float(pj[0]) + dx + self.bias[0], float(pj[1]) + dy + self.bias[1]
