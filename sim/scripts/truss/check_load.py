"""Tier 1: the loader -- where a released rod actually ends up.

The brief's claim is placement to +-0.1 mm, and the design's answer is
that the FIXTURE does it: a V-notch is a kinematic locator, so a rod let
go anywhere in the V's mouth settles on its apex line.  That is a claim
about a mechanism, and this suite measures it the way check_effectors
measures the competition's hoppers: release the rod with a deliberate
arrival error, let the physics settle it, and read where it lies.

Also measured: the capture range (how far off a rod may arrive and still
drop in), the seating of a diagonal into two cradles, and RETENTION --
what a rod does when the cage turns it downward with no keeper.

    python3 sim/scripts/truss/check_load.py [-v]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from truss.glenv import headless        # noqa: E402  (before mujoco)
headless()
import numpy as np
import mujoco

from truss import structure, geometry, fixture, mjcf, cell, inspector, schedule, approach
from truss.spec import Head, Gripper, Process, Cage
from truss.geometry import TrussGeometry, theta_chord_up, theta_face_up

VERBOSE = "-v" in sys.argv
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def fresh(t, stage="empty"):
    g = TrussGeometry(t)
    fx = fixture.Fixture(g)
    m = mujoco.MjModel.from_xml_string(mjcf.scene_cell(g, fx, stage=stage))
    d = mujoco.MjData(m)
    c = cell.CellSim(m, d, g, fx, rng=np.random.default_rng(0), thermal=False)
    clk = cell.SimClock(m, d, c)
    for _ in range(30):
        clk.tick()
    return g, fx, c, clk


def ticks(clk, n):
    for _ in range(n):
        clk.tick()


def goto_settle(c, clk, n=120, **axes):
    for a, v in axes.items():
        c.goto(a, v)
    ticks(clk, n)


def pick(c, clk, fx, rod):
    mid, yaw = fx.pick_pose(rod)
    goto_settle(c, clk, x=mid[0] - Head.grip_x(), y=mid[1],
                z=schedule.grip_z(mid[2]) + 30.0)
    goto_settle(c, clk, 40, w=yaw)
    goto_settle(c, clk, 60, z=schedule.grip_z(mid[2]))
    goto_settle(c, clk, 80, g=Head.GRIP_STROKE)
    c.close()
    ticks(clk, 40)
    held = c.holding()
    goto_settle(c, clk, 80, g=0.0)
    goto_settle(c, clk, 60, z=schedule.grip_z(mid[2]) + 30.0)
    return held


def place(c, clk, fx, rod, dx=0.0, dy=0.0, dz=None, settle=1.0):
    """Carry a held rod to its place pose, offset by (dx, dy) in the world
    and released dz above the seat, and let it settle."""
    dz = Process.DROP_IN if dz is None else dz
    p, yaw, th = fx.place_pose(rod)
    c.index(th)
    ticks(clk, 300)
    goto_settle(c, clk, 200, x=p[0] - Head.grip_x() + dx, y=p[1] + dy)
    if abs(((yaw - c.at("w")) + 180.0) % 360.0 - 180.0) > 1e-6:
        # the plan's order: down to the solved height, gripper out, turn
        # there, finish the descent (approach.yaw_height says why)
        g = c.geom
        half = (g.t.L_cut if rod.kind == "diag" else rod.length) / 2.0
        obs = approach.Obstacles(g, fx, th, skip=(rod.index,))
        z_yaw = approach.yaw_height(obs, p, c.at("w"), yaw, half, rod.r, Process.SEAT_CLEAR,
                                    z_max=approach.index_lift(g, fx), stroke=Head.GRIP_STROKE)
        goto_settle(c, clk, 120, z=schedule.grip_z(z_yaw))
        goto_settle(c, clk, 80, g=Head.GRIP_STROKE)
        goto_settle(c, clk, 60, w=yaw)
        goto_settle(c, clk, 60, z=schedule.grip_z(p[2]) + 3.0)
    else:
        goto_settle(c, clk, 60, z=schedule.grip_z(p[2]) + 3.0)
        goto_settle(c, clk, 80, g=Head.GRIP_STROKE)
    goto_settle(c, clk, 40, z=schedule.grip_z(p[2]) + dz)
    c.open()
    ticks(clk, int(settle * 50))
    err = inspector.seat_error(c.geom, rod, *c.rod_pose(rod.index), theta=c.cage_truth())
    goto_settle(c, clk, 80, g=0.0)
    goto_settle(c, clk, 60, z=schedule.grip_z(p[2]) + 30.0)
    return err


def main():
    t = structure.TRUSS_300
    # ------------------------------------------------ a chord, dead on
    g, fx, c, clk = fresh(t)
    r = g.chords[0]
    held = pick(c, clk, fx, r)
    check("a chord is picked from its rack", held)
    e = place(c, clk, fx, r)
    check("...and released a millimetre above its notches it seats within %.2f mm" % inspector.SEAT_TOL,
          e < inspector.SEAT_TOL, "%.3f mm" % e)
    # ------------------------------------------- the capture range
    # THE CAPTURE RANGE, MEASURED: release further and further off the V
    # until the rod no longer seats.  The closed form (the mouth less the
    # rod, halved) is a floor -- a rod touching a flank's top edge still
    # rolls in -- and the measured range must not be under it.
    cap = fx.capture_range(r)
    rows = []
    for dy in (0.5, 1.0, cap, cap + 0.7, cap + 1.5, cap + 2.5):
        g, fx, c, clk = fresh(t)
        r = g.chords[0]
        pick(c, clk, fx, r)
        e = place(c, clk, fx, r, dy=dy)
        rows.append((dy, e))
    seated = [dy for dy, e in rows if e < inspector.SEAT_TOL]
    missed = [dy for dy, e in rows if e >= inspector.SEAT_TOL]
    check("a chord arriving anywhere inside the V's mouth seats within %.2f mm" % inspector.SEAT_TOL,
          all(e < inspector.SEAT_TOL for dy, e in rows if dy <= cap + 1e-9),
          " ".join("%.1f->%.3f" % (dy, e) for dy, e in rows))
    check("the measured capture range is at least the closed form's %.1f mm, and finite" % cap,
          bool(seated) and max(seated) >= cap - 1e-9 and bool(missed)
          and min(missed) > max(seated),
          "seats to %.1f, misses from %s" % (max(seated) if seated else -1,
                                             min(missed) if missed else "never"))
    # ------------------------------------------------ a diagonal
    g, fx, c, clk = fresh(t, stage="loaded")
    # free the chords' keepers?  No: chords stay welded; a diagonal is
    # placed into its two cradles against them.  Take diagonal 0 out of
    # the fixture first: unkeep it, lift it into the gripper, then re-place
    r = g.diags[0]
    p, yaw, th = fx.place_pose(r)
    c.index(th)
    ticks(clk, 300)
    c.release_keeper(r.index)          # free it only once its face is up
    ticks(clk, 30)
    goto_settle(c, clk, 200, x=p[0] - Head.grip_x(), y=p[1], z=schedule.grip_z(p[2]) + 30.0)
    goto_settle(c, clk, 40, w=yaw)
    goto_settle(c, clk, 60, z=schedule.grip_z(p[2]))
    goto_settle(c, clk, 80, g=Head.GRIP_STROKE)
    c.close()
    ticks(clk, 40)
    check("a seated diagonal is picked out of its cradles", c.holding())
    goto_settle(c, clk, 80, g=0.0)
    goto_settle(c, clk, 60, z=schedule.grip_z(p[2]) + 30.0)
    e = place(c, clk, fx, r, dx=0.4, dy=-0.4)
    check("...and put back 0.4 mm off in x and y it seats within %.2f mm" % inspector.SEAT_TOL,
          e < inspector.SEAT_TOL, "%.3f mm" % e)
    # ------------------------------------------------ retention
    g, fx, c, clk = fresh(t, stage="loaded")
    r = g.diags[1]
    th = theta_face_up(r.face)
    c.index(th)
    ticks(clk, 400)
    c.release_keeper(r.index)
    ticks(clk, 100)
    e_up = inspector.seat_error(g, r, *c.rod_pose(r.index), theta=c.cage_truth())
    check("a free diagonal, face up, rests in its cradles", e_up < 0.2, "%.3f mm" % e_up)
    c.index(th + 180.0)
    ticks(clk, 700)
    e_down = inspector.seat_error(g, r, *c.rod_pose(r.index), theta=c.cage_truth())
    check("...and turned face DOWN with no keeper it leaves them -- the keeper is a "
          "requirement, not a nicety (Cage.KEEPER)", e_down > 5.0, "%.1f mm off" % e_down)
    w = t.mass_diags / max(t.n_diag, 1) * 9.81e-3
    check("the force a keeper must hold is the rod's own weight, millinewtons",
          w < 0.05, "%.4f N" % w)
    # ------------------------------------------ the whole load, by plan
    g, fx, c, clk = fresh(t)
    from truss import process
    st = approach.plan_truss(g, fx, span=0.0)
    P = schedule.plan(g, fx, st)
    load_ops = [o for o in P.ops if o.phase == "load"]
    P2 = schedule.Plan()
    P2.ops = load_ops
    msgs = []
    ex = process.Executor(c, c, c, c, c, c, clk, g, fx, P2, vision=None, log=msgs.append)
    clk.run(ex.run())
    errs = {r.index: inspector.seat_error(g, r, *c.rod_pose(r.index), theta=c.cage_truth())
            for r in g.rods}
    check("the plan's loading phase seats every rod of the 300 mm truss within %.2f mm"
          % inspector.SEAT_TOL, all(e < inspector.SEAT_TOL for e in errs.values()),
          "worst %.3f mm; %s" % (max(errs.values()), msgs[:3]))
    check("...with every rod kept", all(c.kept(r.index) for r in g.rods))
    planned = sum(o.dt for o in load_ops)
    check("...in the time the plan said, within 20%%",
          abs(clk.now() / planned - 1.0) < 0.20, "%.0f s vs %.0f planned" % (clk.now(), planned))

    bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det and (VERBOSE or not ok) else ""))
    print("check_load: %d checks, %d failed" % (len(RESULTS), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
