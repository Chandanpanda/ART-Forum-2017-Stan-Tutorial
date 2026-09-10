"""The cell's MuJoCo backend of the HAL (hal.py owns the contract).

Steppers are modelled as what they are: the axis is told a setpoint,
quantised to a whole step, and BELIEVES it went there.  What the screw
actually did is the setpoint times a scale drawn once per run -- 12 ppm
per kelvin of steel over a shift's warm-up -- which is the error the
camera exists to absorb (brief 4.3).  The ring is a friction drive with
a fiducial: spin is a velocity command under a slip-torque cap, park is a
P-loop on the fiducial that the firmware would run.  The cage is a worm:
its setpoint moves at the worm's rate and nothing back-drives it.

Two things happen by constraint rather than by contact, and both are
declared stand-ins with a rig behind them:

  * A GRIPPED ROD IS WELDED to the jaw body from the moment the pads meet
    it (the competition's "a carried beam is clamped, not loose").  The
    RELEASE is physics: the rod falls into its V and settles, and
    check_load measures where.
  * A SEATED ROD IS CLIPPED -- welded to the cage -- once the process
    has seen it settle.  The real fixture needs a keeper (Cage.KEEPER),
    and check_load reports the force it must hold.

Everything below the "truth" marker exists only in simulation: the
inspector and the checks read it, the process never does.
"""
import numpy as np
import mujoco

from . import hal, band as _band
from .spec import (mm, Gantry, Ring, Gripper, Cage, Vision,
                   stepper_scale_sigma)

AXES = ("x", "y", "z", "d", "g", "w")
_JOINT = {"x": "gx", "y": "gy", "z": "gz", "d": "gd", "g": "gg", "w": "gw"}
_ACT = {"x": "a_x", "y": "a_y", "z": "a_z", "d": "a_d", "g": "a_g", "w": "a_w"}
YAW_STEP = 0.1                  # deg, a hobby servo's command resolution
# THE RING'S ANGLE IS AN ENCODER READ NOW, NOT A FIDUCIAL.  Friction drive
# slipped, so the turns had to be counted by watching a mark go past; a
# toothed rim on phased pinions cannot slip, so the motor's own count is
# the ring's angle and the only error left is quantisation.  The encoder is
# on the PINION, and the ring turns Mesh.ratio of a pinion turn.
RING_ENC_DEG = 360.0 * Ring.mesh().ratio / Ring.ENCODER_CPR


def _quat_from_mat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


class CellSim(hal.AxesHAL, hal.RingHAL, hal.CageHAL, hal.GripperHAL,
              hal.DispenserHAL, hal.CutterHAL):
    def __init__(self, model, data, geom, fixture, rng=None, thermal=True):
        self.m, self.d = model, data
        self.geom, self.fixture = geom, fixture
        self.rng = rng or np.random.default_rng(0)
        gid = lambda t, n: mujoco.mj_name2id(model, t, n)
        J, A, B, S, E = (mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR,
                         mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_SITE,
                         mujoco.mjtObj.mjOBJ_EQUALITY)
        self.j = {a: gid(J, _JOINT[a]) for a in AXES}
        self.a = {a: gid(A, _ACT[a]) for a in AXES}
        self.a_f, self.a_ring, self.a_cage = gid(A, "a_f"), gid(A, "a_ring"), gid(A, "a_cage")
        self.j_ring, self.j_cage = gid(J, "ring"), gid(J, "cage")
        self.b_cage, self.b_gripw, self.b_gz = gid(B, "cage"), gid(B, "gripw"), gid(B, "gz")
        self.s_grip, self.s_nozzle = gid(S, "grip_pt"), gid(S, "nozzle_tip")
        self.s_centre = gid(S, "ring_centre")
        self.b_rod = {r.index: gid(B, "rod%d" % r.index) for r in geom.all_rods}
        self.eq_keep = {r.index: gid(E, "keep%d" % r.index) for r in geom.all_rods}
        self.eq_hold = {r.index: gid(E, "hold%d" % r.index) for r in geom.all_rods}
        # the nest's detent, on the parts that have one
        self.eq_rack = {}
        for r in geom.all_rods:
            k = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY,
                                  "rack%d" % r.index)
            if k >= 0:
                self.eq_rack[r.index] = k
        self.b_drop = [i for i in range(64)
                       if gid(B, "drop%d" % i) >= 0]
        self.eq_stick = [gid(E, "stick%d" % i) for i in self.b_drop]
        # the axes' world offsets: joint zero is where the body was built
        self._off = {}
        for a in ("x", "y", "z"):
            b = gid(B, {"x": "gx", "y": "gy", "z": "gz"}[a])
            self._off[a] = float(model.body_pos[b][{"x": 0, "y": 1, "z": 2}[a]]) * 1000.0
        self._off["d"] = self._off["g"] = self._off["w"] = 0.0
        # THE SCREW'S SCALE, drawn once.  Believed mm -> actual mm.
        sd = stepper_scale_sigma() if thermal else 0.0
        self.scale = {a: 1.0 + float(self.rng.normal(0.0, sd)) for a in ("x", "y", "z")}
        self.scale.update({"d": 1.0, "g": 1.0, "w": 1.0})
        self._set = {a: self._read_joint(a) for a in AXES}
        for a in AXES:
            self.goto(a, self._set[a])
        self._park = None
        self._rate = 0.0
        self._ring_cmd = float(np.degrees(self.d.qpos[self.m.jnt_qposadr[self.j_ring]]))
        self._ring_w = 0.0
        self._cage_target = 0.0
        self._cage_cmd = 0.0
        # rods the scene built already kept take the kept masks
        for r in geom.all_rods:
            if self.d.eq_active[self.eq_keep[r.index]]:
                self._rod_masks(r.index, True)
        self._closing = False
        self._held = None
        self._n_dosed = 0
        self._drops_live = []          # (drop k, joint index)
        self.hits = {}                 # joint -> world hit point of its drop
        self._cuts = 0
        self.log = []

    # ------------------------------------------------------------- axes
    def _read_joint(self, a):
        v = float(self.d.qpos[self.m.jnt_qposadr[self.j[a]]])
        return (np.degrees(v) if a == "w" else v * 1000.0) + self._off[a]

    def _quant(self, a, value):
        step = YAW_STEP if a == "w" else Gantry.MM_PER_STEP
        return round(float(value) / step) * step

    def goto(self, axis, value):
        a = axis
        lo, hi = self.limits()[a]
        v = self._quant(a, min(max(float(value), lo), hi))
        self._set[a] = v
        actual = (v - self._off[a]) * self.scale[a]
        self.d.ctrl[self.a[a]] = np.radians(actual) if a == "w" else actual / 1000.0

    def at(self, axis):
        return self._set[axis]

    def settled(self, axis, tol=0.05):
        a = axis
        truth = self._read_joint(a)
        want = self._set[a] - self._off[a]
        want = want * self.scale[a] + self._off[a]
        vel = float(self.d.qvel[self.m.jnt_dofadr[self.j[a]]])
        vel = np.degrees(vel) if a == "w" else vel * 1000.0
        return abs(truth - want) < tol and abs(vel) < 20.0 * tol

    def limits(self):
        out = {}
        for a in AXES:
            lo, hi = self.m.jnt_range[self.j[a]]
            if a == "w":
                out[a] = (float(np.degrees(lo)), float(np.degrees(hi)))
            else:
                out[a] = (lo * 1000.0 + self._off[a], hi * 1000.0 + self._off[a])
        return out

    # ------------------------------------------------------------- ring
    def spin(self, rpm):
        """Wind at this speed.  A RATE, SERVED AS A RAMPING POSITION -- the
        drive is a stepper, and a stepper is a position source (see
        Ring.servo_kp: driven as a velocity, any gain the timestep could
        integrate was too soft to reject the hinge's own damping)."""
        self._park = None
        self._rate = float(rpm) * 6.0            # deg/s at the ring

    def park(self, az_deg):
        self._park = float(az_deg)
        self._rate = 0.0

    def angle(self):
        """The ring's angle, from the drive's encoder: quantised, not noisy."""
        q = float(np.degrees(self.d.qpos[self.m.jnt_qposadr[self.j_ring]]))
        return float(np.round(q / RING_ENC_DEG) * RING_ENC_DEG)

    def parked(self, tol=None):
        tol = Ring.STOP_TOL if tol is None else tol
        if self._park is None:
            return False
        q = np.degrees(self.d.qpos[self.m.jnt_qposadr[self.j_ring]])
        w = np.degrees(self.d.qvel[self.m.jnt_dofadr[self.j_ring]])
        err = ((self._park - q) + 180.0) % 360.0 - 180.0
        return abs(err) < tol and abs(w) < 30.0

    def _serve_ring(self, dt):
        """March the commanded ring angle at Ring.servo_acc, toward a rate
        while winding and toward an angle while parking."""
        acc, w_max = Ring.servo_acc(), Ring.RPM_MAX * 6.0
        if self._park is None:
            want = float(np.clip(self._rate, -w_max, w_max))
            step = acc * dt
            self._ring_w += float(np.clip(want - self._ring_w, -step, step))
            self._ring_cmd += self._ring_w * dt
        else:
            q = self._ring_cmd
            err = ((self._park - q) + 180.0) % 360.0 - 180.0
            # brake in time: stop inside the distance this rate needs
            if abs(err) <= self._ring_w ** 2 / (2.0 * acc):
                self._ring_w = max(0.0, self._ring_w - acc * dt)
            else:
                self._ring_w = min(w_max, self._ring_w + acc * dt)
            step = np.sign(err) * self._ring_w * dt
            if abs(step) > abs(err):
                step, self._ring_w = err, 0.0
            self._ring_cmd += step
        self.d.ctrl[self.a_ring] = np.radians(self._ring_cmd)

    # ------------------------------------------------------------- cage
    def index(self, theta_deg):
        self._cage_target = float(theta_deg)

    def theta(self):
        return self._cage_cmd

    def indexed(self, tol=None):
        tol = Cage.THETA_TOL if tol is None else tol
        q = np.degrees(self.d.qpos[self.m.jnt_qposadr[self.j_cage]])
        return (abs(self._cage_cmd - self._cage_target) < 1e-9 and
                abs(((q - self._cage_target) + 180.0) % 360.0 - 180.0) < max(tol, 0.2))

    def _serve_cage(self, dt):
        """The worm: the setpoint walks toward the target at THETA_RPM."""
        d = ((self._cage_target - self._cage_cmd) + 180.0) % 360.0 - 180.0
        step = 6.0 * Cage.THETA_RPM * dt
        if abs(d) <= step:
            self._cage_cmd = self._cage_target
        else:
            self._cage_cmd += np.sign(d) * step
        self.d.ctrl[self.a_cage] = np.radians(self._cage_cmd)

    # ---------------------------------------------------------- gripper
    def free_from_rack(self, rod_index):
        """Let a detented part out of its nest -- what closing the jaws on
        it does."""
        k = self.eq_rack.get(rod_index)
        if k is not None:
            self.d.eq_active[k] = 0

    def close(self):
        self.d.ctrl[self.a_f] = -2.0 * mm(Gripper.JAW_OPEN / 2.0 - 0.3)
        self._closing = True

    def open(self):
        self.d.ctrl[self.a_f] = 0.0
        self._closing = False
        if self._held is not None:
            self.d.eq_active[self.eq_hold[self._held]] = 0
            self._held = None

    def holding(self):
        return self._held is not None

    def _serve_grip(self):
        """The pads have met something: weld it.  A rod is 'between the
        pads' when its axis passes within a pad's reach of the grip point
        and the jaws have closed to its diameter."""
        if not self._closing or self._held is not None:
            return
        gap = -float(self.d.qpos[self.m.jnt_qposadr[mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_JOINT, "gf_l")]]) * 1000.0
        opening = Gripper.JAW_OPEN / 2.0 - gap
        g = self.d.site_xpos[self.s_grip] * 1000.0
        best, best_d = None, 1e9
        for r in self.geom.all_rods:
            p0, p1 = self.rod_pose(r.index)
            u = (p1 - p0) / np.linalg.norm(p1 - p0)
            v = g - p0
            s = float(np.clip(v @ u, 0.0, np.linalg.norm(p1 - p0)))
            dist = float(np.linalg.norm(v - s * u))
            if dist < best_d:
                best, best_d = r, dist
        if best is not None and best_d < best.r + 1.5 and opening <= best.r + 0.6:
            self._weld_now(self.eq_hold[best.index], self.b_gripw, self.b_rod[best.index])
            self._held = best.index
            # ...AND THEN STOP CLOSING.  `close` asks for a fixed travel,
            # not for the rod's own diameter, so the jaws go on past it and
            # sit 0.2 mm into a 1 mm rod with 15 N behind them -- a fight
            # against the weld that holds it, on a body whose inertia is
            # 5e-9.  The jaws close ONTO the rod and hold there.
            self.d.ctrl[self.a_f] = -2.0 * mm(Gripper.JAW_OPEN / 2.0 - best.r)

    # ----------------------------------------------------------- keeper
    # A KEPT ROD IS PART OF THE FIXTURE, so it stops colliding with the
    # fixture and the other kept rods.  Welded AND in contact with the
    # notch flanks and the chord it touches, a diagonal drifted 30 degrees
    # out of its weld over one 210-degree index -- a soft constraint
    # against a persistent contact loses a little every step.  It keeps
    # colliding with the head and the drops, which is what matters.
    KEPT_TYPE, KEPT_AFF = 16, 4 | 8
    FREE_TYPE, FREE_AFF = 1, 1 | 2 | 4 | 8

    def _rod_masks(self, rod_index, kept):
        b = self.b_rod[rod_index]
        for gid in range(self.m.body_geomadr[b], self.m.body_geomadr[b] + self.m.body_geomnum[b]):
            self.m.geom_contype[gid] = self.KEPT_TYPE if kept else self.FREE_TYPE
            self.m.geom_conaffinity[gid] = self.KEPT_AFF if kept else self.FREE_AFF

    def keep(self, rod_index):
        """Engage the keeper: weld the rod to the cage where it now lies."""
        self._weld_now(self.eq_keep[rod_index], self.b_cage, self.b_rod[rod_index])
        self._rod_masks(rod_index, True)

    def release_keeper(self, rod_index):
        self.d.eq_active[self.eq_keep[rod_index]] = 0
        self._rod_masks(rod_index, False)

    def kept(self, rod_index):
        return bool(self.d.eq_active[self.eq_keep[rod_index]])

    def _weld_now(self, eq, b1, b2):
        """Activate a weld holding body2 where it is relative to body1."""
        R1 = self.d.xmat[b1].reshape(3, 3)
        R2 = self.d.xmat[b2].reshape(3, 3)
        rel_p = R1.T @ (self.d.xpos[b2] - self.d.xpos[b1])
        rel_R = R1.T @ R2
        self.m.eq_data[eq][0:3] = 0.0
        self.m.eq_data[eq][3:6] = rel_p
        self.m.eq_data[eq][6:10] = _quat_from_mat(rel_R)
        self.m.eq_data[eq][10] = 1.0
        self.d.eq_active[eq] = 1

    # -------------------------------------------------------- dispenser
    def dose(self, mg, joint=None):
        """Release one drop of `mg` from the nozzle tip."""
        if self._n_dosed >= len(self.b_drop):
            raise RuntimeError("out of drops")
        k = self._n_dosed
        b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "drop%d" % self.b_drop[k])
        jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "drop%d_f" % self.b_drop[k])
        adr = self.m.jnt_qposadr[jid]
        vadr = self.m.jnt_dofadr[jid]
        tip = self.d.site_xpos[self.s_nozzle].copy()
        # the drop is the dose's own size, a sphere of resin, hanging from
        # the tip
        gid = self.m.body_geomadr[b]
        r = _band.drop_radius_mm(mg) / 1000.0
        self.m.geom_size[gid][0] = r
        self.m.geom_rbound[gid] = r
        self.d.qpos[adr:adr + 3] = tip - np.array([0.0, 0.0, r + 0.0005])
        self.d.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.d.qvel[vadr:vadr + 6] = 0.0
        self.m.body_mass[b] = max(mg, 1.0) * 1e-6
        # refresh positions and contacts NOW: the contact list still says
        # the drop is on the floor at its park, and a stick test that read
        # it welded every drop where it had been parked
        mujoco.mj_forward(self.m, self.d)
        self._drops_live.append((k, joint, self.d.time))
        self._n_dosed += 1
        return 1

    def dosed(self):
        return self._n_dosed

    def _serve_drops(self):
        """A drop that touches the truss sticks where it landed: resin
        wicks in, it does not roll off.  What is recorded is where it
        landed, which is what the dispenser's aim is judged by."""
        if not self._drops_live:
            return
        still = []
        for k, joint, t_rel in self._drops_live:
            b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "drop%d" % self.b_drop[k])
            # the first tick it touches the truss or the fixture it is
            # welded there.  Measured: judged two ticks after release, a
            # drop that had met the chord's crest had already rolled off
            # it, and seven of twelve went to the floor
            hit, floor = False, False
            for i in range(self.d.ncon):
                c = self.d.contact[i]
                bb = (self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2])
                if b not in bb:
                    continue
                other = bb[1] if bb[0] == b else bb[0]
                if other == 0:
                    floor = True
                else:
                    hit = True
                    break
            if hit or floor:
                # recorded in the CAGE frame, so it can be compared with the
                # joint's fixture position whatever the cage does afterwards
                from .geometry import rot_x
                p = rot_x(-self.cage_truth()) @ (self.d.xpos[b] * 1000.0)
                self.hits[joint if joint is not None else k] = p
                if hit:
                    self._weld_now(self.eq_stick[k], self.b_cage, b)
                    # and it stops colliding: resin wicks into the band, it
                    # does not go on pressing the chord.  Measured before
                    # this: each stuck drop, welded to the cage while its
                    # soft contact still bore on the chord, walked the
                    # chord a millimetre out of its notches, to 4.5 mm by
                    # the fourth joint.
                    for gid in range(self.m.body_geomadr[b],
                                     self.m.body_geomadr[b] + self.m.body_geomnum[b]):
                        self.m.geom_contype[gid] = 0
                        self.m.geom_conaffinity[gid] = 0
            else:
                still.append((k, joint, t_rel))
        self._drops_live = still

    # ----------------------------------------------------------- cutter
    def cut(self):
        self._cuts += 1
        return True

    # ------------------------------------------------------------- tick
    def tick_hook(self, dt):
        """The firmware's own loops, once per control tick."""
        self._serve_cage(dt)
        self._serve_grip()

    def substep_hook(self):
        """Physics the firmware does not see, after every physics step.

        THE RING'S STEP GENERATOR IS HARDWARE, not a firmware loop.  Served
        once per 50 Hz tick, the ring's commanded angle jumped 7.2 degrees
        at a time and the ring arrived in 2.5 ms and waited 17.5 -- the mean
        speed was right and the instantaneous one was eight times it.  A
        stepper's driver clocks far faster than the loop that tells it what
        speed to run, so the command marches here.

        Resin also wets what it touches the instant it touches it: judged
        once a control tick, a drop that had met the chord's crest and been
        pushed off it again inside the tick was already falling.
        """
        self._serve_ring(float(self.m.opt.timestep))
        if self._drops_live:
            self._serve_drops()

    # ============================================================ truth
    # Sim only: the inspector and the checks.  The process never reads it.
    def ring_centre(self):
        return self.d.site_xpos[self.s_centre].copy() * 1000.0

    def axis_truth(self, axis):
        return self._read_joint(axis)

    def cage_truth(self):
        return float(np.degrees(self.d.qpos[self.m.jnt_qposadr[self.j_cage]]))

    def ring_truth(self):
        return float(np.degrees(self.d.qpos[self.m.jnt_qposadr[self.j_ring]]))

    def rod_pose(self, rod_index):
        """World endpoints of the rod's centreline, mm -- read off the
        geom itself (a capsule's axis is its frame's z), so it is right
        whether the rod was built in a rack or in the fixture."""
        b = self.b_rod[rod_index]
        g = int(self.m.body_geomadr[b])
        c = self.d.geom_xpos[g] * 1000.0
        u = self.d.geom_xmat[g].reshape(3, 3)[:, 2]
        half = float(self.m.geom_size[g][1]) * 1000.0
        return c - u * half, c + u * half

    def grip_point(self):
        return self.d.site_xpos[self.s_grip].copy() * 1000.0

    def nozzle_tip(self):
        return self.d.site_xpos[self.s_nozzle].copy() * 1000.0

    def contacts_between(self, mask_a, mask_b):
        """Contacts whose geoms carry these collision types -- e.g. head
        against anything not the head."""
        # a kept rod carries its own type; asking about rods asks about
        # those too (a nozzle pressing a kept diagonal 1.8 mm out of its
        # cradle went uncounted until it did)
        if mask_a & 1:
            mask_a |= self.KEPT_TYPE
        if mask_b & 1:
            mask_b |= self.KEPT_TYPE
        n = 0
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            ta, tb = self.m.geom_contype[c.geom1], self.m.geom_contype[c.geom2]
            if (ta & mask_a and tb & mask_b) or (tb & mask_a and ta & mask_b):
                n += 1
        return n


class SimClock(hal.Clock):
    """One tick = CTRL substeps of physics, preceded by the firmware
    loops the backend owns."""

    def __init__(self, model, data, cell, decim=None):
        self.m, self.d, self.cell = model, data, cell
        self.decim = decim or int(round(self.PERIOD / model.opt.timestep))

    def now(self):
        return float(self.d.time)

    def tick(self):
        self.cell.tick_hook(self.PERIOD)
        for _ in range(self.decim):
            mujoco.mj_step(self.m, self.d)
            self.cell.substep_hook()


class SimCamera(hal.CameraHAL):
    """The head camera, rendered.  Needs an offscreen GL context."""

    def __init__(self, model, data):
        self.m, self.d = model, data
        self._r = None

    def frame(self):
        if self._r is None:
            self._r = mujoco.Renderer(self.m, Vision.H, Vision.W)
        self._r.update_scene(self.d, camera="head_cam")
        return self._r.render().copy(), float(self.d.time)

    def calib(self):
        X, Y, Z = Vision.cam_frame()
        return {"f": Vision.f_px(), "cx": Vision.W / 2.0, "cy": Vision.H / 2.0,
                "pos": np.array(Vision.cam_pos()),
                "R": np.array([X, Y, Z]).T}         # camera axes as columns
