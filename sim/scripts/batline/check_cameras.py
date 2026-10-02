"""Tier 1: the calibration station's camera rig -- eight cameras round the
gimbal -- placed, calibrated and run on two bats, with both
implementations of its cameras.

    python3 sim/scripts/batline/check_cameras.py [-v]

The plan's contract for this suite: it fails when a clamp fiducial is seen
by fewer than three cameras, when self-calibration misses planted lens
errors, or when pose or colour reads back wrong.  The rig was developed on
the kid's bat; the adult's (check_bat's second size) it has never seen.
Each bat's run draws its own build of the rig -- every camera's place,
turn and lens, the gimbal's axes and zeros, the markers, each sensor's
colour response, the stems' pull -- so self-calibration has something to
find, and runs in its own process.

PLACEMENT, each bat:
  the cameras rig/place.py places meet the rule (spec.Rig) on poses it
  never saw (a grid twice as fine, a quarter step over): with any one
  camera lost, every pose rests on MIN_CHECKED markers each seen by
  MIN_VIEWS cameras, and the pose is fixed within the rig's share of the
  per-unit calibration's budget.  Must fail: the best seven cameras.  And
  the solver's eight leave more to spare than eight cameras evenly round
  the gimbal at its height, the layout a person would draw.

TIER 0, arithmetic, both bats: the rig, its site and station 4's module
meet their rules; no procedure (track, colour, calib, pose) reads the
scene's truth.

TIER 1, MuJoCo, each bat, the model cameras:
  self-calibration  the bar over an 8 x 8 grid of gimbal poses.  Every
                    camera as calibrated puts every ball within
                    RingCam.CENTROID_PX rms (Z_SIGMAS of it at worst) of
                    where the camera as built puts it; every lens
                    parameter, marker and the stems' pull's scale lie
                    within Z_SIGMAS of their own sigmas.  Must fail: the
                    supplier's lens curve held instead of solved.
  pose              the clamp's pose at poses no calibration saw, within
                    the rig's share per axis, 1 sigma.  Must fail: the
                    gimbal's encoders alone.
  drift             the shift-start check passes the rig as calibrated
                    and catches any camera knocked by the rig's rotation
                    share, naming it.  Must fail: the stems' pull left
                    uncorrected.
  colour            the card, then the bands: written colours read back
                    within Rig.DE_MAX -- the design's, the two swapped,
                    both white, and one printed 2 DE_MAX off -- and the
                    line refuses every one but the design's.

TIER 1 with rendered frames, each bat:
  centroids         every usable ball's blob against where the scene's
                    truth puts its centre: the stems' pull has the model's
                    shape and a scale within Z_SIGMAS of the build's prior
                    (RigBuild.STEM), and what the scale leaves is within
                    CENTROID_PX rms, Z_SIGMAS of it at worst -- what makes
                    CENTROID_PX a measurement.  Must fail: uncorrected.
  self-calibration  and the pose, as with the model cameras; the pull's
                    scale self-calibration finds is the one the centroids
                    measure.
  colour (kid)      the design's colours and the swap, from rendered frames.
"""
import copy
import os
import re
import sys
import time
from multiprocessing import get_context

# four workers on four cores: one BLAS thread each, or the solves fight
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from truss.glenv import headless        # noqa: E402  (before mujoco)
headless()
import numpy as np

from batline import product, line as L
from batline.spec import KID, Bat, Module, RingCam, Rig, RigBuild, Rules

VERBOSE = "-v" in sys.argv
RESULTS = []
# check_bat's second size: a bat the rig was never developed on
ADULT = Bat("adult", length=860.0, handle=300.0, grip_d=34.0, width=108.0, depth=62.0)
BATS = {"developed": (KID, 7), "unseen": (ADULT, 8)}
Z = Module.Z_SIGMAS
HERE = os.path.dirname(os.path.abspath(__file__))
PROCEDURES = ("track.py", "colour.py", "calib.py", "pose.py")


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def _station(bat):
    from batline.rig import spec as RS
    d = product.design(bat)
    sts, dg = L.stations(d, timed=False)
    st = sts["calibration"]
    rig = RS.design(d)
    return d, st, rig, RS.site(rig, st, dg, st.module)


def _fits(fs):
    bad = [f for f in fs if not f.ok]
    return not bad, "; ".join("%s (%.4f) %s" % (f.name, f.margin, f.detail) for f in bad) or "%d rules" % len(fs)


# ============================================================= PLACEMENT
def eye_ring(cands):
    """Eight of the solver's own candidates as a person would place them:
    evenly round the gimbal, as near its height as the site allows."""
    C = np.array([c.centre for c in cands])
    el = np.degrees(np.arcsin(C[:, 2] / np.linalg.norm(C, axis=1)))
    az = np.degrees(np.arctan2(C[:, 1], C[:, 0]))
    pick = []
    for k in range(Rig.CAMERAS):
        want = -180.0 + (k + 0.5) * 360.0 / Rig.CAMERAS
        score = el ** 2 + ((az - want + 180.0) % 360.0 - 180.0) ** 2
        score[pick] = np.inf
        pick.append(int(np.argmin(score)))
    return [cands[i] for i in pick]


def placed(bat):
    from batline.rig import spec as RS, place as PL, track as T
    d, st, rig, site = _station(bat)
    pl, cov = PL.place(rig, site)
    held = RS.grid(24, 0.25)
    j = PL.judge(rig, site, pl.cams, held)
    # the views the placement counts, against the ones the tracker will use
    # with the bat in the clamp
    seen = PL.coverage(rig, site, pl.cams, held).seen
    pred = T.Predictor(rig, site, pl.cams, "bat")
    K = len(rig.markers)
    refused = 0
    for ip, (a, c) in enumerate(held):
        T0 = T.encoder_pose(a, c)
        for k, (_uv, _f, ok, _p) in enumerate(pred(pl.cams, a, c, T0[:, :3], T0[:, 3], rig.markers, rig.ball_r)):
            refused += int((seen[k, ip] & ~ok[:K]).sum())
    p7 = PL.solve(cov, n=Rig.CAMERAS - 1, at=PL.bat_points(rig))
    j7 = PL.judge(rig, site, p7.cams, held)
    ring = eye_ring(cov.cams)
    jr = PL.judge(rig, site, ring, held)
    return bat.name, dict(cams=pl.cams, rig_fits=rig.fits, site_fits=site.fits, module_fits=st.module.fits,
                          fits=PL.fits(rig, site, pl.cams, j), j=j, fits7=PL.fits(rig, site, p7.cams, j7),
                          j7=j7, jr=jr, report=RS.report(rig), refused=refused, counted=int(seen.sum()))


def placement_checks(name, p, developed):
    j = p["j"]
    rule7 = [f for f in p["fits7"] if f.name.startswith("any one camera lost, every pose")][0]
    for f in p["fits"]:
        check("%s: %s" % (name, f.name), f.ok, "%s; margin %.4f%s" % (
            f.detail, f.margin, "" if developed or f is not p["fits"][1] else
            "; the best %d alone: fewest checked %d" % (Rig.CAMERAS - 1, p["j7"].least_checked)))
    if developed:
        # the bat the rig was developed on needs all eight; a bigger bat's
        # rig stands further off, and each camera sees more of it
        check("must fail: %s's best %d cameras -- %s" % (name, Rig.CAMERAS - 1, rule7.name), not rule7.ok,
              "fewest checked %d; %s" % (p["j7"].least_checked, rule7.detail))
    check("%s: the placement counts no view the tracker refuses with the bat in the clamp" % name,
          p["refused"] == 0, "%d of the %d it counts on %d held-out poses" % (p["refused"], p["counted"], j.poses))
    jr = p["jr"]
    check("%s: the solver's cameras keep more checked markers to spare than an even ring at the gimbal's height"
          % name, j.least_checked > jr.least_checked,
          "fewest checked, any one camera lost: %d against the ring's %d; ring sigma %.4f deg, %.4f mm"
          % (j.least_checked, jr.least_checked, jr.worst_rot, jr.worst_pos))


# ================================================================ TIER 0
def tier0(placements):
    for name, p in placements.items():
        for what in ("rig", "site", "module"):
            ok, det = _fits(p[what + "_fits"])
            check("%s: the %s meets every rule it is held to" % (name, {"rig": "rig", "site": "rig's site",
                                                                       "module": "calibration station's module"}[what]),
                  ok, det)
    # the procedures never read the scene's truth: the truth's methods, the
    # build as drawn for the scene, or a vision's private parts
    rig_dir = os.path.join(HERE, "..", "..", "batline", "rig")
    banned = re.compile(r"truth_\w+|\.build\b|RigBuildDraw\.draw\b|\.rng\b")
    hits = []
    for f in PROCEDURES:
        src = open(os.path.join(rig_dir, f)).read()
        hits += ["%s: %s" % (f, m.group(0)) for m in banned.finditer(src)]
    check("no procedure reads the scene's truth (%s)" % ", ".join(PROCEDURES), not hits, "; ".join(hits) or "none")


# ================================================================ TIER 1
def _built(bat, cams, seed):
    from batline.rig import mjcf as RM
    d, st, rig, site = _station(bat)
    return d, rig, site, RM.RigBuildDraw.draw(rig, cams, np.random.default_rng(seed))


def calibration_checks(label, cal, scene, b, rig, stems=True):
    """Self-calibration against the scene's truth; `stems`: whether the
    stems pull as the build drew them (the model camera; a rendered frame
    pulls as it pulls, which centroid_checks measures)."""
    from batline.rig import judge as J
    from batline.rig.calib import LENS
    out = []
    G = J.gauge(cal, scene)
    rp = np.array(J.reprojection(cal, scene, G))
    out.append(("%s: self-calibration finds every camera's planted lens -- as calibrated, each puts every ball "
                "within %.2f px rms of where the camera as built does, %.2f at worst"
                % (label, RingCam.CENTROID_PX, Z * RingCam.CENTROID_PX),
                rp[:, 0].max() <= RingCam.CENTROID_PX and rp[:, 1].max() <= Z * RingCam.CENTROID_PX,
                "rms %.3f..%.3f px, worst %.3f; %d rounds, %d sightings, residual rms %.3f px"
                % (rp[:, 0].min(), rp[:, 0].max(), rp[:, 1].max(), len(cal.rounds), len(cal.obs), cal.sol.rms_px)))
    z = J.intrinsics_z(cal, scene)
    c, k = np.unravel_index(np.argmax(np.abs(z)), z.shape)
    out.append(("%s: every camera's lens parameters lie within %.0f of their own sigmas" % (label, Z),
                np.abs(z).max() <= Z, "worst %.2f (cam%d %s)" % (np.abs(z).max(), c, LENS[k])))
    K = len(rig.markers)
    sd = cal.sol.cov["point"][:K]
    zm = (cal.sol.P[:K] - b.markers) / np.where(sd > 0, sd, np.inf)
    out.append(("%s: every marker lies within %.0f of its own sigma" % (label, Z), np.abs(zm).max() <= Z,
                "worst %.2f; error rms %.4f mm, sigma %.4f mm" % (np.abs(zm).max(),
                float(np.sqrt(np.mean((cal.sol.P[:K] - b.markers) ** 2))), float(sd.mean()))))
    if stems:
        out.append(("%s: the stems' pull is solved to within %.0f of its sigma" % (label, Z),
                    abs(cal.stem - b.stem) <= Z * cal.stem_sd, "%.3f +- %.3f against %.3f as built"
                    % (cal.stem, cal.stem_sd, b.stem)))
    return out, G


def pose_checks(label, vision, cal, rig, site, cams, held, G):
    from batline.rig import track as T, judge as J, place as PL
    from batline.rig.pose import PoseFit
    pred = T.Predictor(rig, site, cams, "bat")
    fits = [T.track(vision, cal, rig, pred, a, b).fit for a, b in held]
    got = [i for i, f in enumerate(fits) if f is not None]
    b_rot, b_pos = PL.budget()
    out = []
    if got:
        e = J.pose_errors([fits[i] for i in got], held[got], vision, G, PL.bat_points(rig))
        zr = np.sqrt(np.mean(e.z ** 2, axis=0))
        out.append(("%s: the clamp's pose at %d poses no calibration saw is within %.3f deg and the bat's ends "
                    "within %.3f mm, per axis, 1 sigma" % (label, len(held), b_rot, b_pos),
                    len(got) == len(held) and e.rms_rot() <= b_rot and e.rms_pos() <= b_pos,
                    "%.4f deg (worst %.4f), %.4f mm (worst %.4f); %d of %d tracked; its own sigma's z rms %s"
                    % (e.rms_rot(), np.abs(e.rot).max(), e.rms_pos(), np.abs(e.pos).max(), len(got), len(held),
                       np.round(zr, 2))))
    else:
        out.append(("%s: the clamp's pose at poses no calibration saw" % label, False, "nothing tracked"))
    # must fail: the encoders' own word, in the rig's own frame
    enc = []
    for a, b in held:
        T0 = T.encoder_pose(a, b)
        enc.append(PoseFit(T0[:, :3], T0[:, 3], 0.0, 0, 0, None, np.eye(6), 0.0, 0.0, 0, True))
    e = J.pose_errors(enc, held, vision, (np.eye(3), np.zeros(3)), PL.bat_points(rig))
    out.append(("must fail: %s, the gimbal's encoders alone are within the share" % label,
                not (e.rms_rot() <= b_rot and e.rms_pos() <= b_pos),
                "%.4f deg, %.4f mm" % (e.rms_rot(), e.rms_pos())))
    return out, fits


def drift_checks(label, rig, site, cams, b, cal):
    from batline.rig import track as T, vision as V, place as PL, spec as RS
    from batline.rig.lens import Camera, rodrigues
    out = []
    pred = T.Predictor(rig, site, cams, "none")
    poses = RS.grid(4, 0.25)
    r = T.drift_check(V.ModelVision(rig, site, cams, b, "none", np.random.default_rng(11)), cal, rig, pred, poses)
    out.append(("%s: the shift-start check passes the rig as calibrated" % label, r["ok"],
                "chi2 %.0f, limit %.0f" % (r["chi2"], r["limit"])))
    knock = PL.budget()[0]
    caught, named = 0, 0
    weakest = None
    rng = np.random.default_rng(12)
    for k in range(len(cams)):
        bb = copy.deepcopy(b)
        c = bb.cams[k]
        ax = rng.normal(size=3)
        bb.cams[k] = Camera.at(c.name, c.lens, c.centre, rodrigues(np.radians(knock) * ax / np.linalg.norm(ax)) @ c.R,
                               **c.meta)
        r = T.drift_check(V.ModelVision(rig, site, cams, bb, "none", np.random.default_rng(13 + k)), cal, rig, pred,
                          poses)
        caught += not r["ok"]
        named += r["worst_index"] == k
        p = float(r["per_cam_p"][k])
        weakest = p if weakest is None else max(weakest, p)
    out.append(("%s: the shift-start check catches any one camera knocked by the rig's rotation share, %.3f deg, "
                "and names it" % (label, knock), caught == len(cams) and named == len(cams),
                "%d of %d caught, %d named; the knocked camera's own p at most %.2g" % (caught, len(cams), named,
                                                                                        weakest)))
    # a camera that sees nothing it can match leaves no residuals: dark, or
    # knocked so far (twice the association gate) that nothing matches
    gate = Z * T.kinematic_px(rig)
    far = np.degrees(2.0 * gate / rig.lens_drawn().fx)

    class Dark(V.ModelVision):
        def snap(self, a, c):
            o = V.ModelVision.snap(self, a, c)
            o[self.dark] = o[self.dark][:0]
            return o
    caught, named = 0, 0
    for k in range(len(cams)):
        mv = Dark(rig, site, cams, b, "none", np.random.default_rng(21 + k))
        mv.dark = k
        r = T.drift_check(mv, cal, rig, pred, poses)
        caught += not r["ok"]
        named += r["worst_index"] == k
        bb = copy.deepcopy(b)
        c = bb.cams[k]
        ax = rng.normal(size=3)
        bb.cams[k] = Camera.at(c.name, c.lens, c.centre, rodrigues(np.radians(far) * ax / np.linalg.norm(ax)) @ c.R,
                               **c.meta)
        r = T.drift_check(V.ModelVision(rig, site, cams, bb, "none", np.random.default_rng(31 + k)), cal, rig, pred,
                          poses)
        caught += not r["ok"]
        named += r["worst_index"] == k
    out.append(("%s: the shift-start check catches any one camera gone dark, or knocked %.2f deg (past matching), "
                "and names it" % (label, far), caught == 2 * len(cams) and named == 2 * len(cams),
                "%d of %d caught, %d named" % (caught, 2 * len(cams), named)))
    # must fail: the stems' pull left uncorrected
    raw = copy.copy(cal)
    raw.stem = 0.0
    r = T.drift_check(V.ModelVision(rig, site, cams, b, "none", np.random.default_rng(11)), raw, rig, pred, poses)
    out.append(("must fail: %s, the shift-start check passes centroids the stems pull, uncorrected" % label,
                not r["ok"], "chi2 %.0f, limit %.0f" % (r["chi2"], r["limit"])))
    return out


def _mix(design, other, de):
    """The colour on the line from `design` toward `other` that lies `de` from it."""
    from batline.rig import colour as CO
    lo, hi = 0.0, 1.0
    for _ in range(50):
        t = (lo + hi) / 2.0
        if CO.delta_e((1 - t) * np.asarray(design) + t * np.asarray(other), design) < de:
            lo = t
        else:
            hi = t
    return (1 - lo) * np.asarray(design) + lo * np.asarray(other)


def colour_checks(label, vision, d, rig, site, cams, cal, palettes):
    from batline.rig import track as T, colour as CO, spec as RS
    out = []
    pred = T.Predictor(rig, site, cams, "bat")
    guard = T.guard_mm()
    cp = CO.card_poses(rig, cal.sol.cams, pred, RS.grid(12), guard)
    fc = [T.track(vision, cal, rig, pred, a, b).fit for a, b in cp]
    card = CO.fit_card(vision, cal.sol.cams, fc, cp, rig, pred, guard)
    missing = [k for k, A in enumerate(card.A) if A is None]
    out.append(("%s: every camera reads the card" % label, not missing,
                "%d gimbal stops; patches per camera %s%s" % (len(cp), card.patches,
                "; none: %s" % missing if missing else "")))
    poses = RS.grid(4, 0.5)
    fb = [T.track(vision, cal, rig, pred, a, b).fit for a, b in poses]
    sleeve = d.parts["sleeve"].rgba[:3]
    design = [np.asarray(c, float) for c in Rules.BAND_RGB]
    for what in palettes:
        if what == "design":
            printed = design
        elif what == "swapped":
            printed = design[::-1]
        elif what == "white":
            printed = [np.asarray(sleeve, float)] * 2
        else:                                              # a band printed 2 DE_MAX off
            printed = [_mix(design[0], sleeve, 2.0 * Rig.DE_MAX), design[1]]
        for i, rgb in enumerate(printed):
            vision.paint("band%d" % (i + 1), rgb)
        reads = CO.read_bands(vision, cal.sol.cams, card, fb, poses, rig, pred, guard, sleeve)
        _, back = CO.verdict(reads, printed)
        ok, de = CO.verdict(reads, design)
        n = [r.n for r in reads]
        how = {"design": "in the design's colours", "swapped": "in each other's colours",
               "white": "the sleeve's white", "near": "with one %.1f dE off" % (2 * Rig.DE_MAX)}[what]
        out.append(("%s: the bands printed %s read back as printed, within %.1f dE" % (label, how, Rig.DE_MAX),
                    max(back) <= Rig.DE_MAX, "dE %s from %s samples" % (np.round(back, 2), n)))
        if what == "design":
            raw = CO.read_bands(vision, cal.sol.cams, card, fb, poses, rig, pred, guard, sleeve, correct=False)
            out.append(("%s: the line passes the design's bands" % label, ok,
                        "dE %s; without the card %s" % (np.round(de, 2), np.round(CO.verdict(raw, design)[1], 2))))
        else:
            out.append(("must fail: %s, the line passes bands printed %s" % (label, how), not ok,
                        "dE from the design %s" % np.round(de, 2)))
    for i, rgb in enumerate(design):
        vision.paint("band%d" % (i + 1), rgb)
    return out


def model(which):
    from batline.rig import track as T, vision as V, spec as RS
    from batline.rig.lens import DIST
    bat, seed = BATS[which]
    cams = PLACED[bat.name]["cams"]
    d, rig, site, b = _built(bat, cams, seed)
    label = "%s, model cameras" % bat.name
    mv = V.ModelVision(rig, site, cams, b, "bar", np.random.default_rng(1))
    cal = T.self_calibrate(mv, rig, site, RS.grid(8), b.bar_cert)
    out, G = calibration_checks(label, cal, mv, b, rig)
    # must fail: the supplier's curve, held
    sup = T.self_calibrate(V.ModelVision(rig, site, cams, b, "bar", np.random.default_rng(1)), rig, site, RS.grid(8),
                           b.bar_cert, fix_dist=DIST)
    o, Gs = calibration_checks(label, sup, mv, b, rig)
    bv = V.ModelVision(rig, site, cams, b, "bat", np.random.default_rng(3))
    held = RS.grid(12, 0.5)
    ps, _f = pose_checks(label, bv, sup, rig, site, cams, held, Gs)
    out.append(("must fail: %s, the supplier's lens curve held, every camera puts every ball within %.2f px rms of "
                "where the camera as built does" % (label, RingCam.CENTROID_PX), not o[0][1],
                "%s; its pose: %s" % (o[0][2], ps[0][2])))
    o, _f = pose_checks(label, bv, cal, rig, site, cams, held, G)
    out += o
    out += drift_checks(label, rig, site, cams, b, cal)
    out += colour_checks(label, bv, d, rig, site, cams, cal, ("design", "swapped", "white", "near"))
    return out


def centroid_checks(label, pv, rig, b, poses):
    """The rendered blobs against the truth: the stems' pull and the noise.
    Returns the checks and a function of a scale: what that scale leaves
    along the pull where it pulls more than CENTROID_PX, px rms."""
    from batline.rig import track as T, vision as V
    P_all = np.vstack([b.markers, b.bar])
    pred = T.Predictor(rig, pv.site, b.cams, "bar")        # the cameras as built: no calibration in the way
    E, D = [], []
    for a, c in poses:
        blobs = pv.snap(a, c)
        R, t = pv.truth_pose(a, c)
        for k, (uv, front, ok, pull) in enumerate(pred(b.cams, a, c, R, t, P_all, rig.ball_r)):
            B = blobs[k]
            for j in np.flatnonzero(ok):
                if not len(B):
                    continue
                dd = np.hypot(B[:, 0] - uv[j, 0], B[:, 1] - uv[j, 1])
                i = int(np.argmin(dd))
                if dd[i] <= V.GAP / 2.0:                    # nearer than any other ball's image can be
                    E.append(B[i, :2] - uv[j])
                    D.append(pull[j])
    E, D = np.array(E), np.array(D)
    n2 = float(np.sum(D * D))
    k = float(np.sum(E * D)) / n2
    rest = E - k * D
    noise = 1.4826 * float(np.median(np.abs(rest - np.median(rest, axis=0))))     # per axis, robust
    size = np.linalg.norm(D, axis=1)
    big = size > RingCam.CENTROID_PX
    u = D[big] / size[big, None]

    def along(scale):
        return float(np.sqrt(np.mean(np.einsum("ij,ij->i", E[big] - scale * D[big], u) ** 2)))

    across = float(np.sqrt(np.mean((rest[big, 0] * u[:, 1] - rest[big, 1] * u[:, 0]) ** 2)))
    out = []
    out.append(("%s: the stems' pull has the model's shape -- what one scale leaves along it, where it pulls more "
                "than %.2f px, is within %.2f px rms" % (label, RingCam.CENTROID_PX, RingCam.CENTROID_PX),
                big.sum() > 0 and along(k) <= RingCam.CENTROID_PX,
                "%.3f px rms over %d sightings, against %.3f across it" % (along(k), big.sum(), across)))
    out.append(("%s: its scale lies within %.0f of the build's prior, 1 +- %.2f" % (label, Z, RigBuild.STEM),
                abs(k - 1.0) <= Z * RigBuild.STEM, "%.3f +- %.3f" % (k, noise / np.sqrt(n2))))
    rms = float(np.sqrt(np.mean(rest ** 2)))
    out.append(("%s: corrected, every usable ball's centroid is within %.2f px rms of its centre, %.2f at worst"
                % (label, RingCam.CENTROID_PX, Z * RingCam.CENTROID_PX),
                rms <= RingCam.CENTROID_PX and np.abs(rest).max() <= Z * RingCam.CENTROID_PX,
                "%.3f px rms per axis (%.3f robust), worst %.3f, over %d sightings"
                % (rms, noise, np.abs(rest).max(), len(rest))))
    out.append(("must fail: %s, uncorrected, where the stems pull more than %.2f px the centroids are within it rms"
                % (label, RingCam.CENTROID_PX), not along(0.0) <= RingCam.CENTROID_PX, "%.3f px rms" % along(0.0)))
    return out, along


def pixels(which):
    from batline.rig import track as T, vision as V, spec as RS
    bat, seed = BATS[which]
    cams = PLACED[bat.name]["cams"]
    d, rig, site, b = _built(bat, cams, seed)
    label = "%s, rendered frames" % bat.name
    pv = V.PixelVision(rig, site, cams, b, "bar", np.random.default_rng(1))
    out, along = centroid_checks(label, pv, rig, b, RS.grid(3, 0.5))
    cal = T.self_calibrate(pv, rig, site, RS.grid(8), b.bar_cert)
    pv.close()
    o, G = calibration_checks(label, cal, pv, b, rig, stems=False)
    out += o
    out.append(("%s: the stems' pull as self-calibration solved it leaves the rendered centroids, where it pulls "
                "more than %.2f px, within it rms" % (label, RingCam.CENTROID_PX),
                along(cal.stem) <= RingCam.CENTROID_PX, "scale %.3f +- %.3f: %.3f px rms"
                % (cal.stem, cal.stem_sd, along(cal.stem))))
    bv = V.PixelVision(rig, site, cams, b, "bat", np.random.default_rng(3))
    o, _f = pose_checks(label, bv, cal, rig, site, cams, RS.grid(4, 0.5), G)
    bv.close()
    return out + o


def colour_pixels(which):
    from batline.rig import track as T, vision as V, spec as RS
    bat, seed = BATS[which]
    cams = PLACED[bat.name]["cams"]
    d, rig, site, b = _built(bat, cams, seed)
    cal = T.self_calibrate(V.ModelVision(rig, site, cams, b, "bar", np.random.default_rng(1)), rig, site,
                           RS.grid(8), b.bar_cert)
    bv = V.PixelVision(rig, site, cams, b, "bat", np.random.default_rng(3))
    out = colour_checks("%s, rendered frames" % bat.name, bv, d, rig, site, cams, cal, ("design", "swapped"))
    bv.close()
    return out


SCENARIOS = {
    "model developed": lambda: model("developed"),
    "model unseen": lambda: model("unseen"),
    "pixels developed": lambda: pixels("developed"),
    "pixels unseen": lambda: pixels("unseen"),
    "colour pixels developed": lambda: colour_pixels("developed"),
}
PLACED = {}


def scenario(arg):
    name, placed_cams = arg
    PLACED.update(placed_cams)
    t0 = time.time()
    try:
        res = SCENARIOS[name]()
    except Exception as e:                          # a crash is a failure, with its reason
        import traceback
        res = [("%s ran to the end" % name, False, "%s: %s" % (type(e).__name__, traceback.format_exc()[-600:]))]
    return name, res, time.time() - t0


def main():
    t0 = time.time()
    import mujoco
    # the slowest first, so the pool's last worker is not left with them
    order = ["pixels developed", "pixels unseen", "colour pixels developed", "model developed", "model unseen"]
    with get_context("spawn").Pool(min(len(SCENARIOS), os.cpu_count() or 1)) as pool:
        placements = dict(pool.map(placed, [bat for bat, _s in BATS.values()]))
        cams = {name: {"cams": p["cams"]} for name, p in placements.items()}
        job = pool.map_async(scenario, [(n, cams) for n in order], chunksize=1)
        for name, p in placements.items():
            placement_checks(name, p, name == BATS["developed"][0].name)
        tier0(placements)
        done = job.get()
    for name, res, el in done:
        for nm, ok, det in res:
            check(nm, ok, det)
        if VERBOSE:
            print("  (%s: %.0f s)" % (name, el))
    if VERBOSE:
        for name, p in placements.items():
            print(p["report"])
    bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det and (VERBOSE or not ok) else ""))
    print("check_cameras: %d checks, %d failed, %.0f s (MuJoCo %s)"
          % (len(RESULTS), bad, time.time() - t0, mujoco.__version__))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
