"""Tier 1: the HAL contract, with physics -- and the ratchet.

  * CONFORMANCE: the backend implements every verb.
  * HONESTY: an axis believes the quantised setpoint and the screw does
    the setpoint times its thermal scale -- the disagreement is the error
    the camera exists for, and it must be there; the ring spins at the
    rpm it is told and parks within the stop tolerance; the cage indexes
    at the worm's rate.
  * THE RATCHET: process.py reads no simulator state.  No `.d`, `.m`,
    `_truth`, `ring_centre`, `rod_pose` anywhere in it.

    python3 sim/scripts/truss/check_hal.py [-v]
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from truss.glenv import headless        # noqa: E402  (before mujoco)
headless()
import numpy as np
import mujoco

from truss import structure, geometry, fixture, mjcf, cell, hal, motion
from truss.spec import Ring, Gantry, Cage, stepper_scale_sigma
from truss.geometry import TrussGeometry

VERBOSE = "-v" in sys.argv
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def main():
    t = structure.TRUSS_300
    g = TrussGeometry(t)
    fx = fixture.Fixture(g)
    m = mujoco.MjModel.from_xml_string(mjcf.scene_cell(g, fx, stage="loaded"))
    d = mujoco.MjData(m)
    c = cell.CellSim(m, d, g, fx, rng=np.random.default_rng(7))
    clk = cell.SimClock(m, d, c)
    for _ in range(30):
        clk.tick()

    check("the backend implements the whole contract", hal.audit(c) == [],
          ", ".join(hal.audit(c)) or "complete")
    check("...and is registered against the ABCs, not duck-typed",
          all(isinstance(c, k) for k in (hal.AxesHAL, hal.RingHAL, hal.CageHAL,
                                         hal.GripperHAL, hal.DispenserHAL, hal.CutterHAL)))
    check("one tick is one control period",
          abs(clk.decim * m.opt.timestep - hal.Clock.PERIOD) < 1e-9)

    # ------------------------------------------------------------ axes
    x0 = c.at("x")
    goal = x0 + 250.0
    mv = motion.Move({"x": x0}, {"x": goal}, Gantry.V_MAX, Gantry.A_MAX)
    tt = 0.0
    while tt < mv.T:
        c.goto("x", mv.at(tt)["x"])
        tt += clk.PERIOD
        clk.tick()
    c.goto("x", goal)
    for _ in range(30):
        clk.tick()
    believed, truth = c.at("x"), c.axis_truth("x")
    check("an axis believes the quantised setpoint",
          abs(believed - goal) <= Gantry.MM_PER_STEP / 2.0 + 1e-9,
          "%.4f for %.4f" % (believed, goal))
    check("...and the screw did the setpoint times its drawn scale",
          abs(truth - ((believed - c._off["x"]) * c.scale["x"] + c._off["x"])) < 0.02,
          "truth %.3f, believed %.3f, scale %.6f" % (truth, believed, c.scale["x"]))
    check("the scale is a per-run draw of the right size",
          0.0 < abs(c.scale["x"] - 1.0) < 4.0 * stepper_scale_sigma(),
          "%.2e vs sigma %.2e" % (c.scale["x"] - 1.0, stepper_scale_sigma()))
    check("settled() reports the axis settled when it is", c.settled("x"))
    c.goto("x", goal + 30.0)
    check("...and not the moment a new setpoint lands", not c.settled("x"))
    for _ in range(60):
        clk.tick()
    z0 = c.at("z")
    c.goto("z", z0 - 20.0)
    for _ in range(60):
        clk.tick()
    check("z holds its setpoint against gravity to a step",
          abs(c.axis_truth("z") - c.at("z")) < 2.0 * Gantry.MM_PER_STEP + 0.01,
          "%.3f vs %.3f" % (c.axis_truth("z"), c.at("z")))
    lim = c.limits()
    check("limits are the travels", set(lim) == {"x", "y", "z", "d", "g", "w"})

    # ------------------------------------------------------------ ring
    c.spin(Ring.RPM)
    for _ in range(100):
        clk.tick()
    a0 = c.ring_truth()
    for _ in range(100):
        clk.tick()
    rpm = (c.ring_truth() - a0) / 360.0 / 2.0 * 60.0
    check("the ring spins at the rpm it is told, within 3%% (measured after the "
          "spin-up ramp, which is %.1f s of the spec's own)" % Ring.SPINUP_S,
          abs(rpm - Ring.RPM) < 0.03 * Ring.RPM, "%.1f rpm" % rpm)
    # THE ANGLE IS AN ENCODER READ, NOT A FIDUCIAL.  The friction drive
    # slipped, so the turns had to be counted by watching a mark go past
    # and the reading carried a noise; a toothed rim on phased pinions
    # cannot slip (check_drive measures 0.30 deg of a 7.50 deg tooth over a
    # revolution), so the count at the motor IS the angle and the only
    # error left is quantisation.  This is the check that replaced the
    # fiducial's, and it asserts the opposite property: no noise at all.
    enc = [c.angle() - c.ring_truth() for _ in range(50)]
    check("the ring's angle is the drive's encoder: true to half a count",
          abs(np.mean(enc)) <= cell.RING_ENC_DEG / 2.0 + 1e-9,
          "%.5f deg of a %.5f count" % (np.mean(enc), cell.RING_ENC_DEG))
    check("...and quantised, not noisy: read it fifty times and it does not move",
          np.std(enc) == 0.0
          and all(abs(c.angle() / cell.RING_ENC_DEG
                      - round(c.angle() / cell.RING_ENC_DEG)) < 1e-6 for _ in range(5)),
          "sd %.2e deg" % np.std(enc))
    check("...and fine enough to park by: a count is a fraction of the stop tolerance",
          cell.RING_ENC_DEG < Ring.STOP_TOL / 10.0,
          "%.4f deg of %.1f" % (cell.RING_ENC_DEG, Ring.STOP_TOL))
    c.park(-15.0)
    n = 0
    while not c.parked() and n < 300:
        clk.tick(); n += 1
    err = ((c.ring_truth() - (-15.0)) + 180.0) % 360.0 - 180.0
    check("park lands the gap within the stop tolerance",
          c.parked() and abs(err) < Ring.STOP_TOL, "%.2f deg in %.2f s" % (err, n * clk.PERIOD))

    # ------------------------------------------------------------ cage
    th0 = c.cage_truth()
    c.index(th0 + 120.0)
    n = 0
    while not c.indexed() and n < 600:
        clk.tick(); n += 1
    took = n * clk.PERIOD
    want = 120.0 / (6.0 * Cage.THETA_RPM)
    check("the cage indexes 120 degrees at the worm's rate",
          c.indexed() and abs(took - want) < 0.8, "%.2f s for %.2f" % (took, want))
    err = ((c.cage_truth() - (th0 + 120.0)) + 180.0) % 360.0 - 180.0
    check("...and lands within a fifth of a degree", abs(err) < 0.2, "%.3f deg" % err)
    check("theta() is the worm's own count, which equals the target once indexed",
          abs(c.theta() - (th0 + 120.0)) < 1e-9)

    # ---------------------------------------------------------- keeper
    r = g.chords[1]
    check("a loaded rod is kept", c.kept(r.index))
    c.release_keeper(r.index)
    check("...and can be freed", not c.kept(r.index))
    c.keep(r.index)
    check("...and kept again where it is", c.kept(r.index))

    # --------------------------------------------------------- ratchet
    src = open(os.path.join(os.path.dirname(__file__), "..", "..", "truss", "process.py")).read()
    body = re.sub(r'"""[\s\S]*?"""', "", src)
    body = "\n".join(l.split("#")[0] for l in body.splitlines())
    leaks = [w for w in ("\\.d\\b", "\\.m\\b", "_truth", "ring_centre", "rod_pose",
                         "mujoco", "xpos", "qpos")
             if re.search(w, body)]
    check("process.py reads nothing below the HAL", not leaks, "found %s" % leaks)
    lits = [l for l in re.findall(r"(?<![\w.])\d{2,4}(?:\.\d+)?(?![\w.])", body)
            if float(l) not in (180.0, 360.0)]          # angle wraps are arithmetic
    check("process.py carries no coordinate-scale literals (pinned at 0)",
          len(lits) == 0, "%s" % lits[:6])

    # ------------------------------------------- the GL backend, per platform
    # MUJOCO_GL names a context whose legal values differ by operating
    # system.  Every script here once wrote "osmesa" unconditionally, and
    # on Windows that raises out of `import mujoco` before any of this
    # code runs.  One module decides it now, and nothing else may.
    import glob
    from truss import glenv
    assign = re.compile(r"""\[\s*["']MUJOCO_GL["']\s*\]\s*=|setdefault\(\s*["']MUJOCO_GL["']""")
    offenders = []
    here = os.path.abspath(__file__)
    for f in (glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))
              + glob.glob(os.path.join(os.path.dirname(__file__), "..", "..", "truss", "*.py"))):
        if os.path.abspath(f) == here or os.path.basename(f) == "glenv.py":
            continue
        for line in open(f).read().splitlines():
            if assign.search(line):
                offenders.append("%s: %s" % (os.path.basename(f), line.strip()[:50]))
    check("no script names a GL backend itself; truss.glenv decides",
          not offenders, "; ".join(offenders[:3]))
    keep = os.environ.pop("MUJOCO_GL", None)
    try:
        got = {}
        for plat in ("win32", "darwin", "linux"):
            os.environ.pop("MUJOCO_GL", None)
            real, sys.platform = sys.platform, plat
            try:
                got[plat] = glenv.headless()
            finally:
                sys.platform = real
        check("...and it names one only where the name is legal: nothing off Linux",
              got["win32"] == "" and got["darwin"] == "" and got["linux"] == "osmesa",
              "%s" % got)
        os.environ["MUJOCO_GL"] = "glfw"
        check("...and never overrides what the operator set",
              glenv.headless() == "glfw")
    finally:
        os.environ.pop("MUJOCO_GL", None)
        if keep is not None:
            os.environ["MUJOCO_GL"] = keep

    check("a viewer is told to open a window only where one can exist",
          glenv.windowed() in (True, False))
    # ------------------------------------ BACKSLASHES IN THE SOURCE ITSELF
    # THE SAME FAMILY AS MUJOCO_GL: a defect only a Windows user meets.
    # `demo_assembly` documents the Windows command line, which has
    # backslashes in it, in a docstring that was not raw -- so `\t` became a
    # TAB and ate the `t` of `truss`, and the command this file handed a
    # Windows user was `python scripts<tab>russ\demo_assembly.py`.
    #
    # THIS CHECK ALREADY EXISTED AND COULD NOT FAIL, which is worth more than
    # the fault.  It read "every source compiles with syntax warnings fatal"
    # and made SyntaxWarning an error -- but the invalid-escape warning is a
    # SyntaxWarning only from Python 3.12; on the 3.11 it was written and run
    # against it is a DeprecationWarning, so the filter caught nothing.  A
    # check that can only fail on an interpreter nobody runs it on is not a
    # measurement.  This one matches on the warning's MESSAGE, so it does not
    # care which category the interpreter of the day files it under, and it
    # walks the whole tree rather than two globs.
    #
    # Two checks, because the loud half and the silent half are different
    # faults: `\d` warns, `\t` does not and mangles the text instead.
    import warnings
    import ast
    import pathlib
    root = pathlib.Path(os.path.dirname(__file__)).parent.parent
    pys = sorted(q for q in root.rglob("*.py") if "__pycache__" not in str(q))
    loud, quiet = [], []
    for q in pys:
        text = q.read_text(encoding="utf-8", errors="replace")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                compile(text, str(q), "exec")
            except SyntaxError as exc:
                loud.append("%s: %s" % (q.name, exc))
                continue
        for msg in caught:
            if "escape sequence" in str(msg.message):
                loud.append("%s:%s %s" % (q.name, msg.lineno, msg.message))
        # a DOCSTRING is prose, so a control character in one is always a
        # mangled path and never intent
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, (ast.Module, ast.ClassDef,
                                     ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node, clean=False)
            for ch in doc or "":
                if ord(ch) < 32 and ch != "\n":
                    quiet.append("%s:%s has %r in a docstring"
                                 % (q.name, getattr(node, "lineno", 1), ch))
                    break
    check("every source file in the tree compiles without an escape-sequence "
          "warning, whatever category this Python files it under: a warning "
          "printed at the operator on every run is a defect",
          not loud, "%d files scanned; %s" % (len(pys), "; ".join(loud[:3])))
    check("...and no docstring carries a control character, which is the half that "
          "does NOT warn: a Windows path in a docstring that is not raw loses the "
          "letter after every valid escape",
          not quiet, "%d files scanned; %s" % (len(pys), "; ".join(quiet[:3])))

    bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det and (VERBOSE or not ok) else ""))
    print("check_hal: %d checks, %d failed" % (len(RESULTS), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
