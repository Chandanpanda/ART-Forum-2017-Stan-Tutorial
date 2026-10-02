"""Tier 1: the station module -- one gantry, its tools, its load cell and
its head camera, and the executor that runs them -- on two travels.

    python3 sim/scripts/batline/check_gantry.py [-v]

The plan's contract for this suite: it fails when the module misses its
repeatability anywhere in its travel, including on a travel it has never
seen.  The module was developed on the kid's bat's electronics station
(four tools, 1.6 m of X); the travel it has never seen is the adult's bat's
pack station (five tools, among them a spindle and a vacuum cup it never
carried, and 2.7 m of X).  Each station is built from its own draw of build
errors, so commissioning has something to find, and runs in its own
process.

TIER 0, arithmetic, every station of both bats:
  every module meets the rules it is held to; its travel takes in every
  region wherever 200 drawn builds put it; Z reaches below the lowest top
  by a touch's search and the pogo block's sink; a touch's fitted readings
  stay under every preload they push against.  And a module that must
  fail: the pack station with 12 mm screws only, whose X whips.

TIER 1, MuJoCo, each station:
  repeatability   X, Y and Z together, at both ends and the middle of each
                  travel, from both sides, twice: the carriage's true place
                  against where the controller believes it is, spread
                  under Module.REPEAT.
  holding         each axis parked mid-travel and pushed either way by the
                  largest force the module lets a process push it with
                  moves less than Module.REPEAT.  Must fail: X with its
                  preload taken off, pushed across its play.
  lost steps      a jam mid-move -- something in the carriage's way -- is
                  caught by the encoder and the move fails; the same move
                  unjammed passes; homing again, from the middle of the
                  travel, puts the axis back where it was to the switch's
                  repeat.
  commissioning   from the drawing alone, every tool measured against the
                  pin; then each tool's tip sent over the pin and over the
                  four corners of the farthest region it works on --
                  three at its fiducials, one where nothing was looked at --
                  lands on them to Module.Z_SIGMAS of what the camera can
                  know there (HeadCam.CENTROID_PX a look, weighed by the
                  fit), as the sim's truth measures it; and at the height
                  asked for within Module.REPEAT.  Must fail: the same,
                  sent by the drawing's own frame.
  ops             each pick tool takes its token from its nest and sets it
                  back inside the nest's clearance; the press reads the
                  plunger's rate; the spindle runs the nut down its stud
                  until it seats; the pogo block closes every pin on its
                  pad.
  pixels          the rendered frame's disc centre against the model
                  camera's, looked at as look() looks -- centred, at four
                  sub-pixel phases, and on the first look as far
                  off-centre as the build can put a disc: what makes
                  HeadCam.CENTROID_PX a measurement.  Must fail: the tall
                  pin seen off-centre, which is why look() centres first.
"""
import copy
import os
import sys
import time
from dataclasses import replace
from math import radians, sqrt
from multiprocessing import get_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from truss.glenv import headless        # noqa: E402  (before mujoco)
headless()
import numpy as np

from batline import product, line as L
from batline.spec import (KID, Bat, Module, Build, LoadCell, Stepper, Artefact, HeadCam, HomeSwitch, Print,
                          Pogo, Gripper2)
from batline.station import spec as SS
from truss.spec import Gantry

VERBOSE = "-v" in sys.argv
RESULTS = []
# check_bat's second size: a bat the module was never developed on
ADULT = Bat("adult", length=860.0, handle=300.0, grip_d=34.0, width=108.0, depth=62.0)
STATIONS = {"developed": (KID, "electronics", 11), "unseen": (ADULT, "pack", 12)}
BUILDS = 200           # drawn builds the travel is tried against


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def bat_modules(bat):
    """Every station's module for a bat, and -- for the must-fail -- the
    unseen station's rules again on the plan's 12 mm screws only."""
    d = product.design(bat)
    sts, dg = L.stations(d, timed=False)
    out = {"%s %s" % (bat.name, name): st.module for name, st in sts.items()}
    _b, name, _s = STATIONS["unseen"]
    screws = SS.derive(SS.from_line(sts[name], dg, d, drives=("screw",))).fits if bat.name == _b.name else None
    return out, screws


def module(mods, which):
    bat, name, seed = STATIONS[which]
    label = "%s %s" % (bat.name, name)
    return mods[label], label, seed


# ================================================================ TIER 0
def tier0(mods, screws):
    from batline.station import mjcf as SM
    for label, mod in mods.items():
        bad = [f for f in mod.fits if not f.ok]
        check("%s: the module meets every rule it is held to" % label, not bad,
              "; ".join("%s (%.3f)" % (f.name, f.margin) for f in bad) or "%d rules" % len(mod.fits))

    # the travel takes in every region as built: each tool's tip over every
    # corner of every region it works, and the camera over every fiducial
    rng = np.random.default_rng(5)
    for label, mod in mods.items():
        lim = {a: (ax.lo, ax.hi) for a, ax in mod.axes.items()}
        out = -np.inf
        for _ in range(BUILDS):
            b = SM.BuildDraw.draw(mod, rng)
            for r in mod.spec.regions:
                corners = [(x, y) for x in (r.lo[0], r.hi[0]) for y in (r.lo[1], r.hi[1])]
                for kind in r.reach:
                    if kind == "camera":
                        pts = [b.place(mod, r.name, p[:2]) for p in r.fiducial_points()]
                        off = np.array(mod.head.camera[:2]) + np.array(b.cam[:2])
                    else:
                        pts = [b.place(mod, r.name, c) for c in corners]
                        off = np.array(mod.tip(kind)[:2]) + np.array(b.tool[kind][:2])
                    for p in pts:
                        h = np.array(p) - off
                        for i, a in enumerate("xy"):
                            out = max(out, lim[a][0] - h[i], h[i] - lim[a][1])
        check("%s: its travel takes in every region and fiducial, as %d drawn builds put them"
              % (label, BUILDS), out <= 0.0, "%.2f mm %s" % (abs(out), "short" if out > 0 else "to spare"))

    # Z: an extended tip below the lowest top by a touch's search, and the
    # pogo block's sink into its pad
    for label, mod in mods.items():
        tops = min(r.top for r in mod.spec.regions)
        need = tops - max(SS.sink(k) for k in mod.head.tools) - SS.gaps()[0] - mod.head.z_extended
        check("%s: Z goes as low as a touch on the lowest top may have to search" % label,
              mod.axes["z"].lo <= need + 1e-9, "z from %.2f, needs %.2f" % (mod.axes["z"].lo, need))

    # a touch's fitted readings, against the stiffest head there can be --
    # the load cell alone -- at a microstep a reading, stay under every
    # preload they push against
    k_head = LoadCell.RATED / LoadCell.DEFLECT
    for label, mod in mods.items():
        for a, ax in mod.axes.items():
            for sign in (+1, -1):
                f = (Module.TOUCH_POINTS + 1) * k_head * ax.drive.step / Stepper.MICROSTEPS \
                    + Module.Z_SIGMAS * LoadCell.NOISE
                cap = SS.touch_cap(ax, sign)
                if f > cap:
                    check("%s: a touch along %s%s stays under what holds the carriage on its flank"
                          % (label, "+" if sign > 0 else "-", a), False, "%.2f N against %.2f" % (f, cap))
    check("every touch's fit on every axis stays under the preload it pushes against",
          not [r for r in RESULTS if "stays under what holds" in r[0]],
          "worst %.2f N fitted" % ((Module.TOUCH_POINTS + 1) * k_head * Module.STEP_MAX / Stepper.MICROSTEPS
                                    + Module.Z_SIGMAS * LoadCell.NOISE))

    # must fail: the plan's 12 mm screws everywhere, on the longest X
    bat, name, _ = STATIONS["unseen"]
    whip = [f for f in screws if not f.ok]
    check("must fail: the %s %s on 12 mm screws only misses its rules (the plan's screws, before "
          "rack and pinion)" % (bat.name, name), bool(whip),
          "; ".join(f.name for f in whip)[:160] or "it passed")


# ================================================================ TIER 1
def _sim(mod, seed, pieces=True, thermal=True):
    from batline.station import mjcf as SM, sim as SIM
    b = SM.BuildDraw.draw(mod, np.random.default_rng(seed))
    pcs = SM.test_plate(mod) if pieces else ()
    m, d, sim, clock = SIM.build(mod, b, pcs, rng=np.random.default_rng(seed + 100), thermal=thermal)
    return b, pcs, m, d, sim, clock


def _executor(mod, sim, clock, m, d, seed, frame=None):
    from batline.station import vision as V, ops as O, commission as C
    vis = V.ModelVision(m, d, np.random.default_rng(seed + 200))
    return O.Executor(sim, clock, mod, frame or C.StationFrame.drawing(mod), vis)


def _hold(ex, sim, a, mid):
    """mm the carriage moves between the largest force a process may push
    it with away from home and the same force toward it, parked at mid."""
    ax = ex.mod.axes[a]
    f = SS.touch_cap(ax, ax.toward)              # creeping toward home, the work pushes it away
    dn = yield from ex.move({a: mid}, op="hold")
    at = []
    for sgn in (-ax.toward, ax.toward):
        sim.disturb(a, sgn * f)
        yield from ex._wait(Gantry.SETTLE_S)
        at.append(sim.carriage(a))
    sim.disturb(a, None)
    yield from ex._wait(Gantry.SETTLE_S)
    return dn.ok, abs(at[1] - at[0]), f


def repeatability(mods, which):
    """Every axis at both ends and the middle, from both sides, twice."""
    out = []
    mod, label, seed = module(mods, which)
    b, pcs, m, d, sim, clock = _sim(mod, seed, pieces=False)
    ex = _executor(mod, sim, clock, m, d, seed)

    def run():
        dn = yield from ex.home()
        out.append(("%s: homes every axis" % label, dn.ok, dn.why))
        over = {a: Module.Z_SIGMAS * ax.drive.backlash() for a, ax in mod.axes.items()}
        T = {a: (ax.lo + over[a], (ax.lo + ax.hi) / 2.0, ax.hi - over[a]) for a, ax in mod.axes.items()}
        seen = {(a, t): [] for a in T for t in range(3)}
        bad = []
        for rep in range(2):
            for side in (-1, +1):
                for t in ((0, 1, 2) if side < 0 else (2, 1, 0)):
                    goal = {a: T[a][t] for a in T}
                    pre = {a: T[a][t] + side * over[a] for a in T}
                    for g in (pre, goal):
                        dn = yield from ex.move(g, op="repeat")
                        if not dn.ok:
                            bad.append(dn.why)
                    for a in T:
                        seen[(a, t)].append((side, sim.carriage(a) - sim.at(a)))
        out.append(("%s: every move of the repeatability run finishes without losing a step" % label,
                    not bad and not ex.lost, "; ".join(bad[:3]) or "%d moves" % (24)))
        for a in "xyz":
            spread = [max(v for _, v in seen[(a, t)]) - min(v for _, v in seen[(a, t)]) for t in range(3)]
            out.append(("%s: %s repeats to %.3f mm at both ends and the middle of its %.0f mm, from both sides"
                        % (label, a, Module.REPEAT, mod.axes[a].travel), max(spread) <= Module.REPEAT,
                        "spread %s mm (%s)" % (" ".join("%.4f" % s for s in spread), mod.axes[a].drive.name)))
        # what the preload is for: a process force either way, under the
        # largest one the module lets a process push with; parked in the
        # middle of all three, where the run above has already been
        yield from ex.move({a: T[a][1] for a in T}, op="park")
        for a in "xyz":
            ok, mv, f = yield from _hold(ex, sim, a, T[a][1])
            out.append(("%s: %s holds its place to %.3f mm under a process's largest force, %.1f N, either way"
                        % (label, a, Module.REPEAT, f), ok and mv <= Module.REPEAT, "moved %.4f mm" % mv))
        # lost steps: a jam the drive cannot push through, mid-move
        a = "x"
        ax = mod.axes[a]
        far = T[a][1] + 0.25 * (T[a][2] - T[a][1])
        dn = yield from ex.move({a: far}, op="clean")
        out.append(("%s: a long X move with nothing in the way keeps every step" % label, dn.ok, dn.why))
        before = sim.carriage(a) - sim.at(a)
        # something in the way, half way there
        sim.block(a, (sim.carriage(a) + T[a][1]) / 2.0)
        dn = yield from ex.move({a: T[a][1]}, op="jammed")
        sim.block(a, None)
        caught = (not dn.ok) and any(op == "jammed" for op, _a, _n in ex.lost)
        out.append(("%s: a jam mid-move is caught by the encoder and the move fails" % label, caught,
                    dn.why or "the move passed"))
        dn = yield from ex.home((a,))
        if not dn.ok:
            out.append(("%s: homed again from the middle of its travel" % label, False, dn.why))
            return
        dn2 = yield from ex.move({a: far}, op="after")
        after = sim.carriage(a) - sim.at(a)
        lim = Module.Z_SIGMAS * sqrt(2.0) * HomeSwitch.REPEAT
        out.append(("%s: homed again from the middle of its travel, X is back where it was to the switch's "
                    "repeat (%.3f mm)" % (label, lim),
                    dn.ok and dn2.ok and abs(after - before) <= lim,
                    "%.4f mm %s" % (after - before, dn.why or dn2.why)))

    clock.run(run())
    # must fail: X with its preload taken off, which a process force pushes
    # across its play
    mod2 = copy.copy(mod)
    mod2.axes = dict(mod.axes, x=replace(mod.axes["x"], preload=0.0))
    b, pcs, m, d, sim, clock = _sim(mod2, seed, pieces=False)
    ex = _executor(mod2, sim, clock, m, d, seed)
    res = []

    def run2():
        yield from ex.home()
        ax = mod2.axes["x"]
        over = Module.Z_SIGMAS * ax.drive.backlash()
        mid = (ax.lo + ax.hi) / 2.0
        for _rep in range(2):
            for side in (-1, +1):
                yield from ex.move({"x": mid + side * over}, op="repeat")
                yield from ex.move({"x": mid}, op="repeat")
                res.append(sim.carriage("x") - sim.at("x"))
        # the force is the preloaded module's: what a process may push with
        ex.mod = mod
        res.append((yield from _hold(ex, sim, "x", mid)))
        ex.mod = mod2

    clock.run(run2())
    ok, mv, f = res.pop()
    sp = max(res) - min(res)
    out.append(("must fail: %s's X without its preload is pushed across its %.3f mm play by %.1f N"
                % (label, mod2.axes["x"].drive.backlash(), f), mv > Module.REPEAT,
                "moved %.4f mm; moves alone, from both sides, still repeat to %.4f mm" % (mv, sp)))
    return out


def commissioned(mods, which):
    """Commission from the drawing, send every tool to what the camera saw,
    then every op on the test pieces."""
    from batline.station import commission as C
    out = []
    mod, label, seed = module(mods, which)
    b, pcs, m, d, sim, clock = _sim(mod, seed)
    ex = _executor(mod, sim, clock, m, d, seed)
    t0 = time.time()
    fr, notes = clock.run(C.commission(ex, mod))
    out.append(("%s: commissions itself from the drawing: homes, measures its camera, finds its regions "
                "and measures every tool against the pin" % label, fr is not None,
                notes.get("failed", "") if fr is None else "%d tools, %.0f s of station time, %.0f s wall"
                % (len(fr.tips), d.time, time.time() - t0)))
    if fr is None:
        return out
    pin = mod.regions["pin"]
    pin_top = pin.top + Artefact.PIN_H
    hover = SS.work_gap()

    # what the camera can know: one look is HeadCam.CENTROID_PX at the look
    # range, each axis.  Every target carries the pin's look, through the
    # tools' tips; a region's point carries its fiducials' looks as the
    # affine fit weighs them there (its leverage: 1 on a fiducial, 3 on the
    # corner none of three is on)
    s_look = HeadCam.CENTROID_PX * mod.mm_per_px(mod.look["fixture"])

    def leverage(r, p):
        A = np.array([[f[0], f[1], 1.0] for f in r.fiducial_points()])
        a = np.array([p[0], p[1], 1.0])
        return float(a @ np.linalg.pinv(A.T @ A) @ a)

    def targets(k):
        """(what, station point as measured, true world xy, sigma per axis):
        the pin's top as the camera saw it, and the four corners of the
        region it works on farthest from the pin -- three where its
        fiducials are, one where nothing was looked at."""
        P = np.array(notes["pin"])
        tg = [("the pin", (P[0], P[1], pin_top + hover), b.place(mod, "pin", pin.centre), s_look)]
        far = [r for r in mod.spec.regions if k in r.reach and r.fiducials and r.name != "pin"]
        if far:
            r = max(far, key=lambda r: np.hypot(r.centre[0] - pin.centre[0], r.centre[1] - pin.centre[1]))
            fid = r.fiducial_points()
            (x0, y0), (x1, y1) = (fid[0][0], fid[0][1]), (fid[1][0], fid[1][1])        # opposite corners
            for (x, y) in ((x0, y0), (x1, y1), (x1, y0), (x0, y1)):
                what = "%s %s" % (r.name, "fiducial" if any(abs(x - f[0]) + abs(y - f[1]) < 1e-9 for f in fid)
                                  else "corner")
                q = fr.place(r.name, (x, y))
                tg.append((what, (q[0], q[1], r.top + hover), b.place(mod, r.name, (x, y)),
                           s_look * sqrt(1.0 + leverage(r, (x, y)))))
        return tg

    def send(frame, k, p):
        ex.frame = frame
        head = frame.head_for(k, p)
        d1 = yield from ex.travel((head[0], head[1], None))
        d2 = yield from ex.slide(k, True)
        d3 = yield from ex.move({"z": head[2]}, frac=0.5, op="hover")
        tip = sim.tip_truth(k)
        yield from ex.lift()
        yield from ex.slide(k, False)
        ex.frame = fr
        return (d1.ok and d2.ok and d3.ok), tip

    def accuracy():
        errs, zs, rr = [], [], []
        for k in mod.head.tools:
            for what, p, true, sg in targets(k):
                ok, tip = yield from send(fr, k, p)
                e = np.array(tip[:2]) - np.array(true[:2])
                errs.append((float(np.max(np.abs(e))) / sg, float(np.hypot(*e)), sg, k, what))
                rr.append(float(np.hypot(*e)))
                zs.append(abs(tip[2] - p[2]))
                if not ok:
                    errs.append((np.inf, np.inf, sg, k, what + " unreachable"))
        z, e, sg, k, what = max(errs)
        out.append(("%s: every tool sent over the pin and over a far region's corners lands on it to %d sigmas of "
                    "what its camera can know" % (label, Module.Z_SIGMAS), z <= Module.Z_SIGMAS,
                    "worst %.1f sigma, %.4f mm, the %s over the %s (sigma %.4f); %.4f mm rms over %d targets; "
                    "one look is %.4f mm" % (z, e, k, what, sg, float(np.sqrt(np.mean(np.square(rr)))), len(rr),
                                             s_look)))
        out.append(("%s: ... and at the height asked for, which touches measured, within %.3f mm"
                    % (label, Module.REPEAT), max(zs) <= Module.REPEAT, "worst %.4f mm" % max(zs)))
        # must fail: the drawing's frame, over the pin, with the first tool
        k = next(iter(mod.head.tools))
        drawn = C.StationFrame.drawing(mod)
        ok, tip = yield from send(drawn, k, (pin.centre[0], pin.centre[1], pin_top + hover))
        true = b.place(mod, "pin", pin.centre)
        e = float(np.max(np.abs(np.array(tip[:2]) - np.array(true[:2]))))
        out.append(("must fail: %s's %s sent by the drawing alone misses the pin by more than %d sigmas"
                    % (label, k, Module.Z_SIGMAS), e > Module.Z_SIGMAS * s_look, "%.3f mm" % e))

    clock.run(accuracy())
    out += ops(mod, label, b, pcs, m, d, sim, clock, ex, fr)
    return out


def ops(mod, label, b, pcs, m, d, sim, clock, ex, fr):
    import mujoco
    out = []
    A = Artefact
    body = lambda n: d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)] * 1000.0

    def top_of(pc):
        r = mod.regions[pc.region]
        xy = fr.place(pc.region, pc.at)
        h = {"token": A.TOKEN_H, "plunger": A.PLUNGER_TRAVEL + 10.0, "stud": A.STUD_L + A.NUT_H,
             "pad": A.PAD_T}[pc.kind]
        return (xy[0], xy[1], r.top + h)

    def run():
        for pc in pcs:
            if pc.kind == "pin":
                continue
            p = top_of(pc)
            if pc.kind == "token":
                k = pc.name.split("_", 1)[1]
                r = mod.regions[pc.region]
                # the jaws close round the token's side, above its nest's wall
                grip = r.top + A.TOKEN_H / 2.0 + Module.Z_SIGMAS * Print.TOL if k == "gripper" else None
                nest = np.array(b.place(mod, pc.region, pc.at))
                dn = yield from ex.pick(k, p, grip)
                up = body(pc.name)[2] - r.top
                if not dn.ok:
                    out.append(("%s: the %s takes its token" % (label, k), False, dn.why))
                    continue
                dn = yield from ex.place(k, p, grip)
                tok = body(pc.name)
                R = d.xmat[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, pc.name)].reshape(3, 3)
                lean = float(np.degrees(np.arccos(min(1.0, abs(R[2, 2])))))
                off = float(np.hypot(tok[0] - nest[0], tok[1] - nest[1]))
                out.append(("%s: the %s takes its token, lifts it and sets it back upright in its nest, "
                            "inside the nest's %.2f mm clearance" % (label, k, Print.CLEAR),
                            dn.ok and up > A.TOKEN_H and off <= Print.CLEAR + 1e-6 and lean < 1.0,
                            "lifted to %.0f mm, back %.3f mm off centre, leaning %.2f deg" % (up, off, lean)))
            elif pc.kind == "plunger":
                k = "pusher" if "pusher" in mod.head.tools else "gripper"
                depth = A.PLUNGER_TRAVEL / 2.0
                dn = yield from ex.press(k, p, depth, LoadCell.RATED / Module.TORQUE_SF)
                cv = np.array(dn.data.get("curve") or [(0.0, 0.0)])
                cv = cv[cv[:, 0] > 0.0]
                slope = float(np.polyfit(cv[:, 0], cv[:, 1], 1)[0]) if len(cv) > 3 else 0.0
                out.append(("%s: the %s presses the plunger %.0f mm and reads its rate, %.1f N/mm, to 10%%"
                            % (label, k, depth, A.PLUNGER_K), dn.ok and abs(slope / A.PLUNGER_K - 1.0) < 0.10,
                            "%.3f N/mm over %d readings %s" % (slope, len(cv), dn.why)))
            elif pc.kind == "stud":
                dn = yield from ex.screw(p, A.STUD_L / A.STUD_LEAD + 2.0)
                seat = mod.regions[pc.region].top + A.NUT_H / 2.0
                z = body("nut")[2]
                out.append(("%s: the spindle runs the nut down its stud until it seats" % label,
                            dn.ok and abs(z - seat) < A.STUD_LEAD,
                            "%.1f turns, %.2f mm from its seat %s" % (dn.data.get("turns", 0.0), z - seat, dn.why)))
            elif pc.kind == "pad":
                dn = yield from ex.probe(p)
                out.append(("%s: the pogo block lands on its pad and every pin closes" % label, dn.ok,
                            dn.why or "%d pins" % len(dn.data.get("contacts", []))))

    clock.run(run())
    return out


def pixels(mods, which):
    """The model camera against a rendered frame, looked at as look() looks."""
    import mujoco
    from batline.station import sim as SIM, vision as V
    out = []
    mod, label, seed = module(mods, which)
    b, pcs, m, d, sim, clock = _sim(mod, seed, thermal=False)
    cam = SIM.SimCamera(m, d)
    mv = V.ModelVision(m, d)
    pv = V.PixelVision(cam)

    def head_at(xy, z):
        for a, v in zip("xyz", (xy[0], xy[1], z)):
            d.qpos[sim.axes[a].qd] = (v - mod.axes[a].home) * 1e-3
        mujoco.mj_forward(m, d)

    def err(off_mm):
        """(dpx, radius px) of the disc nearest the image centre."""
        truth = mv.truth()
        near = min(truth, key=lambda s: (s[0] - HeadCam.W / 2.0) ** 2 + (s[1] - HeadCam.H / 2.0) ** 2)
        img, _t = cam.frame()
        got = pv.locate(near=near[:2], img=img)
        return (np.inf if got is None else float(np.hypot(got[0] - near[0], got[1] - near[1]))), near[2]

    centred, first = [], []
    discs = [("pin", b.place(mod, "pin", mod.regions["pin"].centre))]
    discs += [(n, b.place(mod, n, p[:2])) for n, p in mod.fiducials]
    for name, p in discs:
        hx, hy = mod.camera_over(p)
        mpp = mod.mm_per_px(mod.look[name if name != "pin" else "fixture"])
        # look() measures centred, and a centroid's error depends on where
        # the disc falls across a pixel, so centred is four phases of it
        # (quarters of a pixel, each way).  Its first look is off-centre by
        # as far as the build can put the disc from where the drawing says
        # -- the region's place and turn and the camera's mount, at
        # Z_SIGMAS -- and only has to be good enough to centre on.  The pin
        # is first looked at where commissioning has put the camera over it
        offs = [(i / 4.0 * mpp, j / 4.0 * mpp) for i, j in ((0, 0), (1, 2), (2, 1), (3, 3))]
        if name != "pin":
            r = mod.regions[name]
            offs.append((SS.build_margin(r, 0, Build.CAM_XY), -SS.build_margin(r, 1, Build.CAM_XY)))
        for k, off in enumerate(offs):
            head_at((hx + off[0], hy + off[1]), mod.z_carry)
            e, r_ = err(off)
            (first if k == 4 else centred).append(e)
    c = np.array(centred)
    rms = float(np.sqrt(np.mean(c ** 2)))
    out.append(("%s: centred, as look() measures, the model camera's disc centres agree with the rendered "
                "frame's to %.2f px rms, %.2f px worst (Z_SIGMAS of it)"
                % (label, HeadCam.CENTROID_PX, Module.Z_SIGMAS * HeadCam.CENTROID_PX),
                rms <= HeadCam.CENTROID_PX and c.max() <= Module.Z_SIGMAS * HeadCam.CENTROID_PX,
                "%.3f px rms, %.3f worst, over %d looks" % (rms, c.max(), len(c))))
    f = np.array(first)
    out.append(("%s: a flat fiducial's first look, as far off-centre as the build can put it, is within "
                "%.2f px of the model's" % (label, Module.Z_SIGMAS * HeadCam.CENTROID_PX),
                f.max() <= Module.Z_SIGMAS * HeadCam.CENTROID_PX,
                "%.3f px rms, %.3f worst, over %d looks" % (float(np.sqrt(np.mean(f ** 2))), f.max(), len(f))))
    # must fail: the pin is 20 mm tall and as dark down its side as on top;
    # seen a quarter of the view off-centre, its side joins the blob
    hx, hy = mod.camera_over(discs[0][1])
    mpp = mod.mm_per_px(mod.look["fixture"])
    head_at((hx + 0.25 * HeadCam.W * mpp, hy - 0.15 * HeadCam.H * mpp), mod.z_carry)
    e, _r = err(None)
    out.append(("must fail: %s's tall pin seen off-centre is found further off than %.2f px -- why look() "
                "centres first" % (label, Module.Z_SIGMAS * HeadCam.CENTROID_PX),
                e > Module.Z_SIGMAS * HeadCam.CENTROID_PX, "%.2f px" % e))
    return out


SCENARIOS = {
    "commission developed": lambda m: commissioned(m, "developed"),
    "commission unseen": lambda m: commissioned(m, "unseen"),
    "repeat developed": lambda m: repeatability(m, "developed") + pixels(m, "developed"),
    "repeat unseen": lambda m: repeatability(m, "unseen") + pixels(m, "unseen"),
}


def scenario(arg):
    name, mods = arg
    t0 = time.time()
    try:
        res = SCENARIOS[name](mods)
    except Exception as e:                          # a crash is a failure, with its reason
        import traceback
        res = [("%s ran to the end" % name, False, "%s: %s" % (type(e).__name__, traceback.format_exc()[-400:]))]
    return name, res, time.time() - t0


def main():
    t0 = time.time()
    import mujoco
    with get_context("spawn").Pool(min(len(SCENARIOS), os.cpu_count() or 1)) as pool:
        mods, screws = {}, None
        for m, f in pool.map(bat_modules, (KID, ADULT)):
            mods.update(m)
            screws = f if f is not None else screws
        # the scenarios get only the modules they run
        need = {module(mods, w)[1] for w in STATIONS}
        sub = {k: v for k, v in mods.items() if k in need}
        job = pool.map_async(scenario, [(n, sub) for n in SCENARIOS])
        tier0(mods, screws)
        done = job.get()
    for name, res, el in done:
        for nm, ok, det in res:
            check(nm, ok, det)
        if VERBOSE:
            print("  (%s: %.0f s)" % (name, el))
    bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det and (VERBOSE or not ok) else ""))
    print("check_gantry: %d checks, %d failed, %.0f s (MuJoCo %s)"
          % (len(RESULTS), bad, time.time() - t0, mujoco.__version__))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
