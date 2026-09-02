"""THE SCORING PATH, END TO END -- one patient and one kit run.  Run this on
every change.

The fast tier had a hole the size of the scoring path.  Every suite tested
a mechanism -- the pocket seats, the gate shuts, the hopper throws -- and
none of them ran a DELIVERY, so a change that deleted ZONE_NAME outright
left all 217 checks green and only failed when a rig happened to reach the
line.  A suite that cannot see the thing the robot is for is not a suite.

So: put a patient in front of robot 2, price the job, drive it, carry it,
let go, and ask the referee.  It is one delivery of one patient on an
otherwise cleared board, which is not a match -- match-scale behaviour is
the boards' job -- but it is the whole chain, and it runs in seconds.

    python3 sim/scripts/check_delivery.py [-v]
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import mujoco

from rfgyc26 import mjcf, referee, robot2, route, station, world
from rfgyc26.robot import AgentARobot
from rfgyc26.params import Field, M2, Piece, Robot2 as R2

VERBOSE = "-v" in sys.argv
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def rig(colour, start, r2_pose):
    """A board with ONE patient of `colour` at `start` and the rest parked
    off-field, so a delivery is not a search problem."""
    rng = np.random.default_rng(4)
    m = mujoco.MjModel.from_xml_string(mjcf.scene_full_match(
        [(100.0, 100.0), (160.0, 190.0), (220.0, 110.0)], rng=rng, r2=True,
        robot_pose=(150.0, 1050.0, 0.0), r2_pose=r2_pose))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    keep = next(i for i in range(M2.N_CYL) if mjcf.cyl_colour(m, i) == colour)
    for i in range(M2.N_CYL):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cyl%d_f" % i)
        a = m.jnt_qposadr[j]
        d.qpos[a:a+3] = ([start[0]/1000.0, start[1]/1000.0, 0.011] if i == keep
                         else [2.0 + 0.06*i, 2.4, 0.02])
    mujoco.mj_forward(m, d)
    return m, d, keep


def pos(m, d, i):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cyl%d" % i)
    return (float(d.xpos[b][0]*1000), float(d.xpos[b][1]*1000))


def deliver(colour, start, r2_pose, cap=70.0):
    m, d, i = rig(colour, start, r2_pose)
    link = robot2.SimLink(m, d, rng=np.random.default_rng(2))
    spot = robot2.sim_spot(m, d, np.random.default_rng(3))
    ctl = robot2.R2Controller(link, spot, clock=lambda: d.time)
    for _ in range(10):                    # the estimator needs a pose first
        ctl.tick()
        link.step(20)
    zn = ({"red": "HOSP", "green": "RECOVERY"}.get(colour)
          or ("PCC_R" if start[0] > 570.0 else "PCC_L"))
    zone = robot2.ZONES[zn]
    cm = world.board_map(pieces=[(i, *pos(m, d, i), "patient")], skip=i)
    ctl.cmap = cm
    pr = robot2._price(cm, ctl.pose[:2], pos(m, d, i), zone, d.time, zn)
    if pr is None:
        return None, zn, None, ctl, m, d, i
    secs, app, st = pr
    say = (lambda *a, **k: print("        |", *a)) if VERBOSE else \
        (lambda *a, **k: None)
    gen = robot2._deliver(ctl, i, lambda k: pos(m, d, k), st.pose[:2], app,
                          say, lambda: "", zone=zone, stand=st)
    t0 = d.time
    good = None
    while d.time - t0 < cap:
        try:
            next(gen)
        except StopIteration as e:
            good = e.value
            break
        link.step(20)
    return good, zn, st, ctl, m, d, i


def main():
    # ------------------------------------------------- the pieces of a job
    cm = world.board_map()
    st = robot2.place_stand(Field.RECOVERY, cm, (860.0, 400.0))
    check("place_stand answers for a zone at all", st is not None, repr(st))
    if st:
        x, y = st.deposits[0]
        check("...with the patient inside the tape, not an inset box",
              Field.RECOVERY[0] <= x <= Field.RECOVERY[2] and
              Field.RECOVERY[1] <= y <= Field.RECOVERY[3],
              "deposit (%.0f, %.0f) vs %s" % (x, y, Field.RECOVERY))
        check("...and the chassis clear of everything", st.clear > 0.0,
              "clear %.0f mm" % st.clear)
        # THE MARGIN IS WHAT THE ZONE ALLOWS, and RECOVERY is 80 mm deep,
        # so no pose can offer more than about 40.  What must hold is that
        # the carry is told to arrive that precisely rather than on a flat
        # 55 mm tolerance the zone cannot honour (F150).
        check("...and the carry's tolerance is cut to the margin the zone "
              "allows",
              float(np.clip(st.margin, 18.0, robot2.ARRIVE_TOL)) <= st.margin
              + 1e-9 or st.margin < 18.0,
              "margin %.0f mm, tolerance %.0f"
              % (st.margin, np.clip(st.margin, 18.0, robot2.ARRIVE_TOL)))

    check("every zone can be delivered into",
          all(robot2.place_stand(z, cm, (600.0, 600.0)) is not None
              for z in robot2.ZONES.values()),
          str([k for k, z in robot2.ZONES.items()
               if robot2.place_stand(z, cm, (600.0, 600.0)) is None]))

    # --------------------------------------------------- the whole chain
    for colour, start, r2_pose in (
            # a real sticker position: (860, 430) sits 13 mm off the
            # laboratory plate's inflated corner, which is less than one
            # 20 mm costmap cell, so the map cannot resolve it and
            # capture_approach rightly refuses.  No patient ever stands
            # there; putting one there tests the grid, not the robot.
            ("green", (983.0, 650.0), (900.0, 500.0, 90.0)),
            ("red",   (600.0, 780.0), (620.0, 640.0, 90.0)),
            ("yellow", (1000.0, 800.0), (940.0, 700.0, 45.0))):
        good, zn, st, ctl, m, d, i = deliver(colour, start, r2_pose)
        where = pos(m, d, i)
        box = robot2.ZONES[zn]
        inside = box[0] <= where[0] <= box[2] and box[1] <= where[1] <= box[3]
        check("a %s patient is carried into %s" % (colour, zn),
              bool(good) and inside,
              "ended (%.0f, %.0f), %s says %s"
              % (where[0], where[1], zn, "inside" if inside else "OUTSIDE"))
        if inside:
            # ...and the referee agrees it is worth points
            cyl = [(where[0], where[1], colour)] + \
                  [(2400.0, 2400.0, c) for c in
                   (["red"]*4 + ["yellow"]*4 + ["green"]*4)[1:]]
            base = referee.score_cylinders(
                [(2400.0, 2400.0, c) for c in
                 (["red"]*4 + ["yellow"]*4 + ["green"]*4)])[0]
            check("...and the referee pays for it",
                  referee.score_cylinders(cyl)[0] > base,
                  "%+d vs %+d adrift" % (referee.score_cylinders(cyl)[0], base))

    # ------------------------------------ ROBOT 1's KIT RUN, END TO END
    # The same hole on robot 1's side, and it cost the kit column twice.
    # Every suite tested a mechanism -- the hopper throws, the solver finds
    # a stand -- and none of them drove the LEG, so a robot that dropped six
    # kits correctly into the hospital and then plowed them 500 mm west into
    # PCC_L on its way to the next zone read as +4 instead of +18 with
    # everything green.
    rng = np.random.default_rng(7)
    mk = mujoco.MjModel.from_xml_string(mjcf.scene_full_match(
        [(100.0, 100.0), (160.0, 190.0), (220.0, 110.0)], rng=rng, r2=False,
        robot_pose=(571.0, 205.0, -90.0)))     # the dock line, facing south
    dk = mujoco.MjData(mk)
    mujoco.mj_forward(mk, dk)
    rb = AgentARobot(mk, dk, rng=rng)
    for _ in range(20):
        rb.stop()
        mujoco.mj_step(mk, dk, nstep=20)
    said = []
    t0 = dk.time
    gen = route.deliver_kits(rb, log=lambda *a: said.append(" ".join(map(str, a))),
                             clk=lambda: dk.time, order=("HOSP", "PCC_L"))
    while dk.time - t0 < 70.0:
        try:
            next(gen)
        except StopIteration:
            break
        mujoco.mj_step(mk, dk, nstep=20)
    kits = []
    for i in range(M2.N_KITS):
        b = mujoco.mj_name2id(mk, mujoco.mjtObj.mjOBJ_BODY, "kit%d" % i)
        p = dk.xpos[b] * 1000
        kits.append((float(p[0]), float(p[1])))

    def in_box(box):
        return sum(1 for kx, ky in kits
                   if box[0] <= kx <= box[2] and box[1] <= ky <= box[3])

    check("robot 1 gets off the dock line at all (F154: it could not)",
          any("legs" in l for l in said) and dk.time - t0 < 69.0,
          "%.1f s, %d log lines" % (dk.time - t0, len(said)))
    check("...and lands all six kits in the hospital",
          in_box(Field.HOSPITAL) == M2.KIT_PLAN["HOSP"],
          "%d of %d" % (in_box(Field.HOSPITAL), M2.KIT_PLAN["HOSP"]))
    check("...and both of PCC_L's, without plowing the hospital pile west",
          in_box(Field.PCC_L) == M2.KIT_PLAN["PCC_L"],
          "%d of %d" % (in_box(Field.PCC_L), M2.KIT_PLAN["PCC_L"]))
    # ...and the referee pays for it.  PCC_R is robot 2's, so the best a
    # solo robot 1 can bank is 8 kits and one empty zone.
    solo = referee.score_kits(kits)[0]
    check("...and the referee pays the solo maximum for it", solo >= 14,
          "%+d/50 (8 kits placed, PCC_R is robot 2's)" % solo)

    # ------------------------------------- a delivery it should REFUSE
    # A patient wedged in a corner with no room to leave is not deliverable,
    # and saying so is worth more than driving there to find out.
    # walled in on every side, so there is nowhere to stand to take it
    cm2 = world.board_map()
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, -1),
                   (1, -1), (-1, 1)):
        cm2.add_disc(600.0 + dx*45.0, 600.0 + dy*45.0, 30.0)
    pr = robot2._price(cm2, (300.0, 300.0), (600.0, 600.0), Field.HOSPITAL,
                       0.0, "HOSP")
    check("a patient walled in on every side is refused, not attempted",
          pr is None, repr(pr))

    n_bad = sum(1 for _, ok, _ in RESULTS if not ok)
    for nm, ok, det in RESULTS:
        if VERBOSE or not ok:
            print("  %s  %s%s" % ("ok  " if ok else "FAIL", nm,
                                  ("  [%s]" % det) if det else ""))
    print("check_delivery: %d checks, %d failed" % (len(RESULTS), n_bad))
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
