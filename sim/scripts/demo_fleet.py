"""BOTH robots, the whole 250-point board, live.

Robot 1 sweeps, docks, delivers and seals exactly as in demo_pick_place;
robot 2 -- the detached actuator -- scoots from the east end of the
deployment box, shakes its two kits into PCC_R, works the east patient
columns before robot 1's climb arrives, and holds the dead corner while
its teammate finishes.  Every robot-2 command crosses the LinkHAL v0
wire; its only sensor is robot 1's camera spotting the ArUco.

    python scripts/demo_fleet.py [--seed N] [--gui] [--xray] [--speed 1.0]

Viewer controls: left-drag orbits, right-drag pans, scroll zooms,
X toggles the chassis transparent.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mujoco
from rfgyc26 import mjcf, referee, robot2
from rfgyc26.params import Field, AgentA, M2
from rfgyc26.robot import AgentARobot
from rfgyc26.route import mission_agent_a
from rfgyc26 import fleet as fleetmod

CHUTE_OFFSET = AgentA.AXLE_X - AgentA.CHUTE_X
CTRL_DECIM = 20
HOLD_R1 = 2.0


def random_discs(rng, n=3):
    pts = []
    while len(pts) < n:
        p = (rng.uniform(60, 245), rng.uniform(80, 230))
        if all(np.hypot(p[0]-q[0], p[1]-q[1]) > 90 for q in pts):
            pts.append(p)
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--xray", action="store_true")
    ap.add_argument("--speed", type=float, default=0.0,
                    help="playback rate vs real time; 0 = unthrottled")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    discs0 = random_discs(rng)
    xml = mjcf.scene_full_match(discs0, rng=rng, r2=True)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    rb = AgentARobot(m, d, rng=rng)
    link = robot2.SimLink(m, d, rng=np.random.default_rng(a.seed + 5000))
    spot = robot2.sim_spot(m, d, np.random.default_rng(a.seed + 9000))
    FLT = fleetmod.Fleet()
    FLT.join('r1', 185.0)
    FLT.join('r2', robot2.R2_CIRCUM)
    rb.fleet = FLT
    ctl = robot2.R2Controller(link, spot, clock=lambda: d.time)
    g1 = mission_agent_a(rb, list(Field.LAB_HOLE_X), mjcf.LAB_HOLE_Y,
                         CHUTE_OFFSET, log=print, clock=lambda: d.time,
                         discs=discs0)      # the opening survey plans the sweep
    g2 = robot2.mission_robot2(ctl, m, d=d, rb=rb, flt=FLT, log=print, clock=lambda: d.time)

    shown = {"xray": a.xray}
    mjcf.set_xray(m, a.xray)

    def on_key(key):
        if key in (ord("x"), ord("X")):
            shown["xray"] = not shown["xray"]
            mjcf.set_xray(m, shown["xray"])

    viewer = None
    if a.gui:
        import mujoco.viewer as _mjv
        viewer = _mjv.launch_passive(m, d, key_callback=on_key)
        if a.speed == 0.0:
            a.speed = 1.0

    t_wall = time.time()
    d1 = d2 = False
    k = 0
    while d.time < 121.0:
        if not d1 and d.time >= HOLD_R1:
            try:
                next(g1)
            except StopIteration:
                d1 = True
        if not d2:
            try:
                next(g2)
            except StopIteration:
                d2 = True
        # the link advances physics with the wheel servo closed around each
        # substep -- a driver that also stepped would run the sim at double
        # rate and halve every control loop's effective frequency
        FLT.track(rb, ctl, t=d.time)
        link.step(CTRL_DECIM)
        k += 1
        if viewer is not None and k % 2 == 0:
            if not viewer.is_running():
                return
            viewer.sync()
        if a.speed > 0.0:
            lag = d.time / a.speed - (time.time() - t_wall)
            if lag > 0.0:
                time.sleep(min(lag, 0.05))
        if d1 and d2:
            break

    # ------------------------------------------------------------- referee
    discs, beams, kits, cyl = [], [], [], []
    for i in range(3):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "disc%d" % i)
        p = d.xpos[b]*1000; discs.append((p[0], p[1], p[2]))
    for i in (1, 2):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "beam%d" % i)
        p = d.xpos[b]*1000
        beams.append((p[0], p[1], p[2], d.xquat[b].copy()))
    for i in range(M2.N_KITS):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "kit%d" % i)
        p = d.xpos[b]*1000; kits.append((p[0], p[1]))
    for i in range(M2.N_CYL):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cyl%d" % i)
        p = d.xpos[b]*1000
        g = m.geom_rgba[m.body_geomadr[b]]
        col = ("red" if g[0] > 0.6 and g[1] < 0.4 else
               ("green" if g[1] > 0.5 and g[0] < 0.5 else "yellow"))
        cyl.append((p[0], p[1], col))
    tot, out = referee.score_match(discs, beams, kits, cyl)
    print("\n================= FLEET SCORE =================")
    for name, pts, detail in out:
        print("  %-10s %+5.0f" % (name.upper(), pts))
        for _, what, p_ in detail:
            print("      %-52s %+4.0f" % (what, p_))
    print("  TOTAL AT THE BUZZER  %+.0f" % tot)
    if viewer is not None:
        while viewer.is_running():
            time.sleep(0.2)


if __name__ == "__main__":
    main()
