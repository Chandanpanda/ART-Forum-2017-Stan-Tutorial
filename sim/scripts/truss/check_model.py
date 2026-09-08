"""Tier 1: the BUILT cell against its parameters, with physics.

check_geometry re-derives the numbers; this builds the MJCF from them,
settles it, and interrogates what stands there: are the rods where the
racks say, do the notches hold a chord where the geometry says, is the
ring the ring the solver reasoned about, does a seated head touch
nothing.  Every one of these has been wrong once already in this file's
short life (a V whose flanks stood proud, a spool on a spine).

    python3 sim/scripts/truss/check_model.py [-v]
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import mujoco

from truss import structure, geometry, fixture, mjcf, cell, approach, inspector
from truss.spec import Ring, Gantry, Head, Cage, ring_r_in, ring_r_out
from truss.geometry import TrussGeometry, theta_chord_up, rot_x

VERBOSE = "-v" in sys.argv
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def build(t, stage):
    g = TrussGeometry(t)
    fx = fixture.Fixture(g)
    m = mujoco.MjModel.from_xml_string(mjcf.scene_cell(g, fx, stage=stage))
    d = mujoco.MjData(m)
    c = cell.CellSim(m, d, g, fx, rng=np.random.default_rng(0), thermal=False)
    clk = cell.SimClock(m, d, c)
    for _ in range(40):
        clk.tick()
    return g, fx, m, d, c, clk


def ring_radii(m, d, c):
    """Inner and outer radius of the built ring, from its boxes."""
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ring")
    centre = c.ring_centre()
    r_in, r_out = 1e9, 0.0
    for gid in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if not n.startswith("ring"):
            continue
        p = d.geom_xpos[gid] * 1000.0 - centre
        rho = np.hypot(p[1], p[2])
        half = m.geom_size[gid][2] * 1000.0
        r_in, r_out = min(r_in, rho - half), max(r_out, rho + half)
    return r_in, r_out


def main():
    for t, tag in ((structure.TRUSS_300, "300"), (structure.TRUSS_1M, "1m")):
        g, fx, m, d, c, clk = build(t, "empty")
        check("%s: the model compiles and is numerically alive after settling" % tag,
              np.all(np.isfinite(d.qpos)), "nbody %d ngeom %d" % (m.nbody, m.ngeom))
        # rods rest in their racks where the slots say, to a tenth
        worst = 0.0
        for r in g.rods:
            s = fx.slot_of(r.index)
            p0, p1 = c.rod_pose(r.index)
            mid = (p0 + p1) / 2.0
            worst = max(worst, float(np.linalg.norm(mid - (s.p0 + s.p1) / 2.0)))
        check("%s: every rod rests in its rack slot within 0.1 mm" % tag, worst < 0.1,
              "worst %.3f mm" % worst)
        check("%s: nothing on the head touches anything at rest" % tag,
              c.contacts_between(mjcf.HEAD_B, mjcf.ROD | mjcf.CAGE) == 0)
        r_in, r_out = ring_radii(m, d, c)
        check("%s: the built ring has the radii the solver reasoned about" % tag,
              abs(r_in - ring_r_in()) < 0.6 and abs(r_out - ring_r_out()) < 0.6,
              "built %.1f..%.1f, spec %.1f..%.1f" % (r_in, r_out, ring_r_in(), ring_r_out()))
        # the gap: no ring box within the gap's arc, straight down at angle 0
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ring")
        centre = c.ring_centre()
        in_gap = 0
        for gid in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if n.startswith("ring"):
                p = d.geom_xpos[gid] * 1000.0 - centre
                psi = np.degrees(np.arctan2(p[1], -p[2]))
                if abs(psi) < Ring.GAP / 2.0 - 5.0:
                    in_gap += 1
        check("%s: the gap is open where the spec says, straight down at angle zero" % tag,
              in_gap == 0, "%d boxes in the gap" % in_gap)
        # the axes' zero and travels agree with the plan's frame
        lim = c.limits()
        check("%s: the carriage travels are the spec's" % tag,
              abs((lim["x"][1] - lim["x"][0]) - Gantry.X_TRAVEL) < 1e-6 and
              abs((lim["z"][1] - lim["z"][0]) - Gantry.Z_TRAVEL) < 1e-6)
        lo, hi = fx.x_range()
        check("%s: ...and cover the fixture's needs" % tag,
              lim["x"][0] <= lo + 1e-6 and lim["x"][1] >= hi - 1e-6,
              "x %.0f..%.0f for %.0f..%.0f" % (lim["x"][0], lim["x"][1], lo, hi))
        # LOADED: rods welded in the fixture sit where the geometry says
        g, fx, m, d, c, clk = build(t, "loaded")
        worst = 0.0
        for r in g.rods:
            worst = max(worst, inspector.seat_error(g, r, *c.rod_pose(r.index), theta=c.cage_truth()))
        check("%s loaded: every rod is where geometry.py put it, to a hundredth" % tag,
              worst < 0.01, "worst %.4f mm" % worst)
        # notches hold the chord: unclip chord 0, settle, it stays put (chord up)
        c.index(theta_chord_up(0))
        for _ in range(150):
            clk.tick()
        c.release_keeper(g.chords[0].index)
        for _ in range(100):
            clk.tick()
        e = inspector.seat_error(g, g.chords[0], *c.rod_pose(0), theta=c.cage_truth())
        check("%s loaded: a chord freed in its notches, chord up, stays seated to 0.1 mm" % tag,
              e < 0.1, "%.3f mm" % e)
        # the seated head touches nothing at a solver station
        th = theta_chord_up(0)
        j = g.joints_on(0)[1]
        a = approach.station(g, fx, j, th)
        check("%s: the solver has a station for a mid-chord joint" % tag, a is not None)
        if a is not None:
            c.index(th)
            for _ in range(60):
                clk.tick()
            c.park(a.gap_down)
            for _ in range(60):
                clk.tick()
            for x in (a.centre[0] - t.band / 2.0, a.centre[0] + t.band / 2.0):
                c.goto("x", x); c.goto("y", a.centre[1]); c.goto("z", a.centre[2] + a.lift)
                for _ in range(120):
                    clk.tick()
                c.goto("z", a.centre[2])
                for _ in range(60):
                    clk.tick()
                n0 = c.contacts_between(mjcf.HEAD_B, mjcf.ROD | mjcf.CAGE)
                a0 = c.ring_truth()
                c.spin(Ring.RPM)
                for _ in range(100):
                    clk.tick()
                turns = (c.ring_truth() - a0) / 360.0
                n1 = c.contacts_between(mjcf.HEAD_B, mjcf.ROD | mjcf.CAGE)
                check("%s: seated at x %.0f, the head touches nothing and the ring turns freely"
                      % (tag, x), n0 == 0 and n1 == 0 and turns > 1.5,
                      "%d/%d contacts, %.1f turns in 2 s" % (n0, n1, turns))
                c.spin(0.0); c.park(a.gap_down)
                for _ in range(60):
                    clk.tick()
                c.goto("z", a.centre[2] + a.lift)
                for _ in range(60):
                    clk.tick()
        # physics cost, for the record
        import time
        t0 = time.time()
        for _ in range(200):
            mujoco.mj_step(m, d)
        us = 1e6 * (time.time() - t0) / 200
        check("%s: a physics step costs under two milliseconds" % tag, us < 2000.0, "%.0f us" % us)

    bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det and (VERBOSE or not ok) else ""))
    print("check_model: %d checks, %d failed" % (len(RESULTS), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
