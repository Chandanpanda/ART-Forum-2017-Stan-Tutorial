"""Pixel perception: pixels in, robot-frame measurements out.

This module is the SAME code on both machines.  In simulation the frames come
from mujoco.Renderer through the tail cameras (robot.SimCameras); on the Pi
they come from picamera2.  Nothing in here imports mujoco -- numpy and params
only -- because this file ships to the robot.

WHAT IT MEASURES, AND HOW THE GEOMETRY IS USED.  The laboratory slot is a
O60 bore through a 5 mm plate standing on a FLAT floor, seen by a camera whose
pose in the robot frame is calibrated.  That means one eye is enough for a
full 3-D fix: every boundary pixel back-projects along its ray onto a KNOWN
horizontal plane.  The pipeline exploits it twice, because the slot's visible
outline is two different circles (the DET_BIAS finding in params.Vision):

  * the NEAR arc of the outline is the plate's top rim -- the chamfer's outer
    edge, radius HOLE_D/2 + LAB_CHAMFER at the plate's top face.  (The 1 mm
    45-deg chamfer means the grazing edge sits between r=30 @ z=4 and r=31 @
    z=5; at dock viewing angles the two are nearly collinear, so the model
    uses the midpoint.  NEAR_R / NEAR_Z below, verified by check_perception.)
  * the FAR arc is the BASE of the far bore wall -- the same bore, at the
    floor: radius HOLE_D/2 at z=0.  The wall itself is plate-coloured, so
    chroma cannot see the far top rim at all; using the far arc on the WRONG
    plane is exactly the crescent-centroid bias params warns about.

Each arc is back-projected onto ITS plane and a circle of KNOWN radius is
fitted for the centre.  Fitting radius-free first is the refusal gate: a blob
whose free radius is not the bore's is not a slot, whatever it looks like.

TWO MEASUREMENTS THAT MUST AGREE, twice over.  The two arcs are independent
fits and must land coaxial within COAX_GATE; the two eyes are independent
measurements and must agree within EYE_GATE.  Disagreement is a REFUSAL, not
an average -- a wrong dock loses the sample and does it confidently.

The output contract is see_lab's: [(x_mm, y_mm, z_mm, mode)] in the robot
frame (x out the nose, y out the left flank, origin at the axle), mode
"stereo" when both eyes confirmed, "mono" otherwise.  look_lab and the whole
dock consume it unchanged.
"""
import numpy as np
from .params import Field, Vision

# ---- what the plate looks like: blue-grey against a warm floor -------------
# B-R chroma, 8-bit: plate face and bore wall +26, floor -5, robot deck +123
# (measured on rendered frames; the band excludes both).  On the real field
# the plate gets a coloured film if raw wood is too close to the board.
PLATE_DB_LO   = 10.0
PLATE_DB_HI   = 70.0
# ---- cluster gates ---------------------------------------------------------
MIN_CLUSTER_PX = 800          # a slot at Z_MAX-ish is bigger than this
MAX_CLUSTER_PX = 60000
MIN_ARC_PTS    = 20           # per-arc fit needs a real arc, not a speck
# ---- the two known circles -------------------------------------------------
NEAR_R = Field.LAB_HOLE_D/2.0 + Field.LAB_CHAMFER/2.0     # 30.5
NEAR_Z = Field.LAB_PLATE_T - Field.LAB_CHAMFER/2.0        # 4.5
FAR_R  = Field.LAB_HOLE_D/2.0                             # 30.0
FAR_Z  = 0.0
# ---- refusal gates ---------------------------------------------------------
R_GATE    = 6.0               # |fitted free radius - known| beyond this: not a slot
COAX_GATE = 5.0               # the two arcs' centres must be coaxial to this
EYE_GATE  = 8.0               # the two eyes must agree to this
TRIM_MM   = 2.5               # robust trim: drop points this far off the circle


class Eye:
    """One camera's calibration: pinhole + pose in the robot frame (mm).

    R's columns are the image axes expressed in the robot frame (x right in
    the image, y up in the image, z back out of the lens -- the camera looks
    down -z, MuJoCo's own convention and OpenCV's up to a y/z sign flip).
    `dist` is the Brown model (k1 k2 p1 p2 k3); None means an ideal camera
    (the renderer).  On the real pair these numbers come from the OpenCV
    chessboard flow; here they come from Vision.cam_pose plus the mount bias.
    """
    def __init__(self, f, cx, cy, R, T, dist=None):
        self.f, self.cx, self.cy = float(f), float(cx), float(cy)
        self.R = np.asarray(R, float)
        self.T = np.asarray(T, float)
        self.dist = None if dist is None else np.asarray(dist, float)

    def undistort(self, u, v):
        """Normalised, distortion-free image coordinates from pixels."""
        x = (u - self.cx) / self.f
        y = (self.cy - v) / self.f              # rows grow downward
        if self.dist is None:
            return x, y
        k1, k2, p1, p2, k3 = self.dist
        xd, yd = x.copy(), y.copy()
        for _ in range(5):                       # fixed-point inversion
            r2 = xd*xd + yd*yd
            ic = 1.0 / (1.0 + r2*(k1 + r2*(k2 + r2*k3)))
            dx = 2*p1*xd*yd + p2*(r2 + 2*xd*xd)
            dy = p1*(r2 + 2*yd*yd) + 2*p2*xd*yd
            xd, yd = (x - dx)*ic, (y - dy)*ic
        return xd, yd

    def ground(self, u, v, z_mm):
        """Back-project pixels onto the horizontal plane z=z_mm, robot frame.
        Returns (N,2) x-y points; rays parallel to the plane give NaN."""
        x, y = self.undistort(np.asarray(u, float), np.asarray(v, float))
        d = self.R @ np.stack([x, y, -np.ones_like(x)])       # (3, N)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (z_mm - self.T[2]) / d[2]
        s = np.where((s > 0) & np.isfinite(s), s, np.nan)
        return (self.T[:2, None] + d[:2]*s).T

    def project(self, p_robot):
        """(u, v) pixels for robot-frame points; NaN when behind the lens."""
        p = np.atleast_2d(np.asarray(p_robot, float))
        cam = (p - self.T) @ self.R              # rows: (X, Y, Z) cam frame
        z = -cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.cx + self.f*cam[:, 0]/z
            v = self.cy - self.f*cam[:, 1]/z
        bad = z <= 1e-6
        u[bad] = np.nan; v[bad] = np.nan
        return u, v


def sim_calib(rng=None):
    """The rendered pair's calibration, from the ONE mount statement.

    With `rng`, the per-match mounting bias is drawn and folded in -- the
    same physics the synthetic camera models: where the plate sits relative
    to the bore is a bracket on a robot that hits walls on purpose, wrong by
    the same amount all match.  The RENDER stays at the true pose (the world
    does not move); the CALIBRATION the pipeline is handed is what is wrong,
    which is how it is wrong on the real robot too.  One plate, one draw:
    both eyes get the same bias.
    """
    from . import hal
    f = Vision.f_px()
    ext = np.zeros(3)
    ca, sa = 1.0, 0.0
    if rng is not None:
        ext = rng.normal(0.0, Vision.EXT_SIGMA, 3)
        a = np.radians(rng.normal(0.0, Vision.EXT_ANG_SIGMA))
        ca, sa = np.cos(a), np.sin(a)
    rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    eyes = {}
    for side, tag in ((1, "left"), (-1, "right")):
        pos, xax, yax = Vision.cam_pose(side)
        zax = np.cross(xax, yax)
        R = np.column_stack([xax, yax, zax])
        # w = T + R m_true; the pipeline believes w' = T' + R' m with
        # T' = T + R e and R' = R Rz -- identical to the synthetic model's
        # m -> Rz m + e applied in the camera frame.
        eyes[tag] = Eye(f, Vision.W/2.0, Vision.H/2.0,
                        R @ rz, np.asarray(pos) + R @ ext)
    return hal.StereoCalib((Vision.W, Vision.H), eyes["left"], eyes["right"])


# --------------------------------------------------------------- segmentation
def plate_mask(img):
    """Boolean mask of lab-plate-coloured pixels (uint8 RGB in)."""
    d = img[..., 2].astype(np.float32) - img[..., 0].astype(np.float32)
    return (d > PLATE_DB_LO) & (d < PLATE_DB_HI)


def _components(hole):
    """Connected components of `hole` via row runs + union-find.

    Returns (labels int32 HxW with 0 = background, count).  Pure numpy/python
    on the run list -- a few hundred runs a frame -- so no scipy needed.
    """
    H, W = hole.shape
    labels = np.zeros((H, W), np.int32)
    parent = [0]

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    prev = []                                   # (c0, c1, label) of row above
    for r in range(H):
        row = hole[r]
        if not row.any():
            prev = []
            continue
        dif = np.diff(row.astype(np.int8))
        starts = list(np.flatnonzero(dif == 1) + 1)
        ends = list(np.flatnonzero(dif == -1) + 1)
        if row[0]:
            starts.insert(0, 0)
        if row[-1]:
            ends.append(W)
        cur = []
        for c0, c1 in zip(starts, ends):
            lab = 0
            for p0, p1, pl in prev:
                if p0 < c1 and c0 < p1:          # 4-connected overlap
                    pl = find(pl)
                    if lab == 0:
                        lab = pl
                    elif pl != lab:
                        parent[pl] = lab         # merge
            if lab == 0:
                parent.append(len(parent))
                lab = len(parent) - 1
            labels[r, c0:c1] = lab
            cur.append((c0, c1, lab))
        prev = cur
    if len(parent) == 1:
        return labels, 0
    flat = np.array([find(i) for i in range(len(parent))])
    remap = np.zeros_like(flat)
    uniq = np.unique(flat[1:])
    remap[uniq] = np.arange(1, len(uniq) + 1)
    return remap[flat][labels], len(uniq)


def _boundary(d, m, plate):
    """Sub-pixel boundary points of one component AGAINST THE PLATE.

    Only transitions whose outside pixel is plate-coloured count: the hole's
    outline against anything else (the robot's own tail, another piece) is an
    occlusion edge, not the bore, and fitting it would drag the circle.  The
    sub-pixel position is where the chroma profile crosses PLATE_DB_LO
    between the two pixels -- which is also (measured) close to the profile's
    midpoint, so the estimate is unbiased.  Fully vectorised: this runs per
    frame on the Pi too.  Returns (u, v) arrays.
    """
    us, vs = [], []

    def frac(a, b):
        return np.clip((a - PLATE_DB_LO) / np.where(a == b, 1e-9, a - b), 0.0, 1.0)

    # left/right: hole px whose horizontal neighbour is plate
    rr, cc = np.nonzero(m[:, 1:] & plate[:, :-1])
    cc = cc + 1
    t = frac(d[rr, cc-1], d[rr, cc])
    us.append(cc - 1 + t); vs.append(rr.astype(np.float64))
    rr, cc = np.nonzero(m[:, :-1] & plate[:, 1:])
    t = frac(d[rr, cc+1], d[rr, cc])
    us.append(cc + 1 - t); vs.append(rr.astype(np.float64))
    # top/bottom: hole px whose vertical neighbour is plate
    rr, cc = np.nonzero(m[1:, :] & plate[:-1, :])
    rr = rr + 1
    t = frac(d[rr-1, cc], d[rr, cc])
    vs.append(rr - 1 + t); us.append(cc.astype(np.float64))
    rr, cc = np.nonzero(m[:-1, :] & plate[1:, :])
    t = frac(d[rr+1, cc], d[rr, cc])
    vs.append(rr + 1 - t); us.append(cc.astype(np.float64))
    return np.concatenate(us), np.concatenate(vs)


# ------------------------------------------------------------- circle fitting
def _kasa(pts):
    """Algebraic circle fit: centre (2,), radius.  pts (N,2)."""
    A = np.column_stack([2*pts[:, 0], 2*pts[:, 1], np.ones(len(pts))])
    b = (pts**2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]
    r = float(np.sqrt(max(sol[2] + c @ c, 0.0)))
    return c, r

def _fixed_r(pts, r_known, c0):
    """Centre of a circle of KNOWN radius: Gauss-Newton from c0, one robust
    trim.  A half-arc pins a free circle poorly but a fixed-radius one well,
    and the radius here is the rulebook's, not a guess."""
    c = np.asarray(c0, float)
    keep = np.ones(len(pts), bool)
    for it in range(6):
        d = pts[keep] - c
        rho = np.linalg.norm(d, axis=1)
        rho[rho < 1e-9] = 1e-9
        res = rho - r_known
        J = -d / rho[:, None]
        try:
            step = np.linalg.lstsq(J, res, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        c = c - step
        if it == 2:                              # one mid-flight trim pass
            d = pts - c
            res_all = np.abs(np.linalg.norm(d, axis=1) - r_known)
            keep = res_all < max(TRIM_MM, 3.0*np.median(res_all))
            if keep.sum() < MIN_ARC_PTS:
                return None
    return c


# ------------------------------------------------------------------ the slots
def _fit_arc(pts_near, pts_far, sel):
    """Fit both known circles to one split of the boundary points.
    Returns {(name): (centre, npts)} for the arcs that pass their gates."""
    fits = {}
    for name, pts, r_known in (("near", pts_near[sel], NEAR_R),
                               ("far",  pts_far[~sel], FAR_R)):
        pts = pts[np.isfinite(pts).all(1)]
        if len(pts) < MIN_ARC_PTS:
            continue
        c0, r_free = _kasa(pts)
        if abs(r_free - r_known) > R_GATE:        # not bore-sized: refuse
            continue
        c = _fixed_r(pts, r_known, c0)
        if c is not None:
            fits[name] = (c, len(pts))
    return fits


def _slots_one_eye(img, eye):
    """[(x, y, npts)] robot-frame slot centres this eye can defend."""
    d = img[..., 2].astype(np.float32) - img[..., 0].astype(np.float32)
    plate = (d > PLATE_DB_LO) & (d < PLATE_DB_HI)
    if plate.sum() < 4000:                       # the plate is not in view
        return []
    labels, n = _components(~plate)
    out = []
    H, W = plate.shape
    for lab in range(1, n + 1):
        m = labels == lab
        area = int(m.sum())
        if not (MIN_CLUSTER_PX <= area <= MAX_CLUSTER_PX):
            continue
        rr, cc = np.nonzero(m)
        if rr.min() == 0 or rr.max() == H-1 or cc.min() == 0 or cc.max() == W-1:
            continue                             # clipped by the frame: refuse
        u, v = _boundary(d, m, plate)
        if len(u) < 2*MIN_ARC_PTS:
            continue
        # Both plane liftings of every point, computed once.
        pts_near = eye.ground(u, v, NEAR_Z)
        pts_far = eye.ground(u, v, FAR_Z)
        # First split by image row -- near arc is the lower half -- then one
        # reassignment by fit residual: an OBLIQUE slot's outline tilts in
        # the image, and a row split lumps lateral rim points into the
        # wall-base set (measured: -3 mm on the slot 280 mm to the side).
        sel = v >= 0.5*(v.min() + v.max())
        fits = _fit_arc(pts_near, pts_far, sel)
        if "near" in fits and "far" in fits:
            rn = np.abs(np.linalg.norm(pts_near - fits["near"][0], axis=1) - NEAR_R)
            rf = np.abs(np.linalg.norm(pts_far - fits["far"][0], axis=1) - FAR_R)
            sel2 = np.where(np.isfinite(rn) & np.isfinite(rf), rn < rf, sel)
            if (sel2 != sel).any():
                fits = _fit_arc(pts_near, pts_far, sel2) or fits
        if not fits:
            continue
        cs = [c for c, _ in fits.values()]
        ws = np.array([float(np_) for _, np_ in fits.values()])
        if len(cs) == 2 and np.linalg.norm(cs[0] - cs[1]) > COAX_GATE:
            continue                             # the two arcs disagree: refuse
        c = (np.asarray(cs) * ws[:, None]).sum(0) / ws.sum()
        out.append((float(c[0]), float(c[1]), int(ws.sum())))
    return out


class LabPipeline:
    """The full pair: per-eye measurement, cross-eye confirmation."""

    def __init__(self, calib):
        self.calib = calib

    def slots(self, imgL, imgR):
        """[(x_mm, y_mm, z_mm, mode)] in the ROBOT frame -- see_lab's contract."""
        L = _slots_one_eye(imgL, self.calib.left)
        R = _slots_one_eye(imgR, self.calib.right)
        out, used = [], set()
        for x, y, _ in L:
            mate = None
            for j, (x2, y2, _) in enumerate(R):
                if j not in used and np.hypot(x2-x, y2-y) < 35.0:
                    mate = j
                    break
            if mate is not None:
                used.add(mate)
                x2, y2, _ = R[mate]
                if np.hypot(x2-x, y2-y) > EYE_GATE:
                    continue                     # the eyes disagree: refuse
                out.append(((x+x2)/2.0, (y+y2)/2.0, Field.LAB_PLATE_T, "stereo"))
            else:
                out.append((x, y, Field.LAB_PLATE_T, "mono"))
        for j, (x2, y2, _) in enumerate(R):
            if j not in used:
                out.append((x2, y2, Field.LAB_PLATE_T, "mono"))
        return out


# ------------------------------------------------------- cylinder colours
# The patients stand at KNOWN sticker positions; only their colour is random.
# So this is not detection: the caller projects each sticker it wants read
# into the frame (through its own pose estimate) and asks what colour the
# patch is.  Classification bands from the model's palette; the real robot
# recalibrates them from one photo of the actual pieces.
def classify_patch(img, eye, p_robot, half_px=5):
    """'red' | 'green' | 'yellow' | None for a robot-frame point."""
    u, v = eye.project(p_robot)
    u, v = float(u[0]), float(v[0])
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    H, W = img.shape[:2]
    c, r = int(round(u)), int(round(v))
    if not (half_px <= c < W-half_px and half_px <= r < H-half_px):
        return None
    patch = img[r-half_px:r+half_px+1, c-half_px:c+half_px+1].reshape(-1, 3)
    rgb = np.median(patch.astype(float), axis=0)
    r_, g_, b_ = rgb
    if r_ > 1.4*g_ and r_ > 1.4*b_ and r_ > 80:
        return "red"
    if g_ > 1.2*r_ and g_ > 1.2*b_ and g_ > 60:
        return "green"
    if r_ > 100 and g_ > 80 and b_ < 0.7*g_:
        return "yellow"
    return None
