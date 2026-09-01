"""Agent A's MuJoCo backend of the HAL (hal.py owns the contract).

The drive is a stepper model -- commanded wheel speed is quantised to whole
full-steps per second, and step loss can be injected, because the spec calls a
skipped step 'silent' and that is the design's main odometry risk.  odometry()
books exactly what a TMC2209 step counter would: the commanded steps, losses
included, integrated over the time each command was actually in force.

Everything below the "ground truth" marker exists only in simulation: the
referee and the check suites read it freely, mission code is losing it step
by step (design doc paragraph 13 -- pose and qvel go in step 3, see_lab is the
synthetic stand-in for the rendered camera pipeline that lands in step 2).
"""
import numpy as np, mujoco
from . import hal
from .params import Chassis, AgentA, Piece, Field, Vision, M2, mm

WHEEL_R = Chassis.WHEEL_D / 2000.0          # m
HALF_TRACK = Chassis.TRACK / 2000.0         # m
STEP_RAD = 2 * np.pi / Chassis.STEPS_PER_REV


class AgentARobot(hal.DriveHAL, hal.DeviceHAL):
    def __init__(self, model, data, step_loss=0.0, rng=None, vision="model"):
        self.m, self.d = model, data
        self.rng = rng or np.random.default_rng(0)
        self.step_loss = step_loss
        # "model": the synthetic camera (geometry + the Vision error budget,
        # fast -- the regression double).  "render": real frames through
        # mujoco.Renderer into perception.LabPipeline -- the same pixels-in
        # pipeline the Pi cameras feed, and the one that gates releases.
        # Headless machines need MUJOCO_GL=osmesa (or egl) for "render".
        self.vision_mode = vision
        self._pix = None
        gid = lambda t, n: mujoco.mj_name2id(model, t, n)
        self.bid   = gid(mujoco.mjtObj.mjOBJ_BODY, "agentA")
        self.a_l   = gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "a_drive_l")
        self.a_r   = gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "a_drive_r")
        self.a_fl  = gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "a_finger_l")
        self.a_fr  = gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "a_finger_r")
        self.a_shim  = gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "a_shim")
        self.a_roller= gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "a_roller")
        self.a_gate= gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "a_gate")
        self.a_feed= gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "a_feed")
        self.a_blade = gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "a_blade")
        self.a_cr = [gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "a_cradle1"),
                     gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "a_cradle2")]
        self.eq_beam = [gid(mujoco.mjtObj.mjOBJ_EQUALITY, "beam%d_hold" % i)
                        for i in (1, 2)]
        self.s_tof = gid(mujoco.mjtObj.mjOBJ_SENSOR, "a_tof")
        self.s_gyro= gid(mujoco.mjtObj.mjOBJ_SENSOR, "a_gyro")
        self.s_mag = gid(mujoco.mjtObj.mjOBJ_SENSOR, "a_mag")
        self.a_trim= gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "a_trim")
        self.j_trim= gid(mujoco.mjtObj.mjOBJ_JOINT, "A_trim_j")
        # Commanded-step odometry: wheel mm/s currently in force, the mm each
        # wheel has accumulated, and when the accumulator was last brought up
        # to date.  (Replaces odo_steps, which integrated each command over
        # ONE physics substep instead of the whole control period it was in
        # force -- a 20x under-count.  Nothing consumed it, so it never bit.)
        self._odo_v  = np.zeros(2)
        self._odo_mm = np.zeros(2)
        self._odo_t  = data.time

    # ------------------------- ground truth (sim only; not in the HAL) -----
    # The referee and check suites read these freely.  Mission code still
    # does too, pending step 3 (the estimator); check_hal.py pins its reads.
    @property
    def pose(self):
        """(x_mm, y_mm, heading_deg) ground truth."""
        p = self.d.xpos[self.bid]
        q = self.d.qpos[3:7]
        th = np.degrees(np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2)))
        return p[0]*1000.0, p[1]*1000.0, th

    def to_local(self, world_xyz):
        x, y, th = self.pose
        t = np.radians(th)
        dx, dy = world_xyz[0]*1000.0 - x, world_xyz[1]*1000.0 - y
        return (dx*np.cos(-t) - dy*np.sin(-t), dx*np.sin(-t) + dy*np.cos(-t), world_xyz[2]*1000.0)

    def chute_xy(self, offset_mm):
        """World position of the O58 chute axis -- it sits `offset` behind the axle."""
        x, y, th = self.pose
        t = np.radians(th)
        return x - offset_mm*np.cos(t), y - offset_mm*np.sin(t)

    def tof_mm(self):
        v = self.d.sensordata[self.m.sensor_adr[self.s_tof]]
        return 1e9 if v < 0 else v*1000.0

    # ------------------------------------------------------------ actuation
    def _odo_flush(self):
        """Book the command in force since the last flush."""
        t = self.d.time
        if t > self._odo_t:
            self._odo_mm += self._odo_v * (t - self._odo_t)
        self._odo_t = t

    def drive(self, v_mm_s, omega_deg_s):
        """Differential drive.  Wheel speeds are quantised to whole steps/s."""
        self._odo_flush()
        v, w = v_mm_s/1000.0, np.radians(omega_deg_s)
        wl = (v - w*HALF_TRACK) / WHEEL_R
        wr = (v + w*HALF_TRACK) / WHEEL_R
        out = []
        for om in (wl, wr):
            steps = np.round(om / STEP_RAD)                    # whole steps/s
            if self.step_loss and self.rng.random() < self.step_loss:
                steps -= np.sign(steps)                        # a silent skipped step
            out.append(steps * STEP_RAD)
        self.d.ctrl[self.a_l], self.d.ctrl[self.a_r] = out
        # What odometry believes: the QUANTISED command, losses included --
        # which is precisely what the TMC2209's step counter would say.
        self._odo_v = np.array(out) * WHEEL_R * 1000.0

    def stop(self):
        self._odo_flush()
        self.d.ctrl[self.a_l] = self.d.ctrl[self.a_r] = 0.0
        self._odo_v[:] = 0.0

    def odometry(self):
        """(dL_mm, dR_mm) commanded wheel travel since the previous call."""
        self._odo_flush()
        dl, dr = self._odo_mm
        self._odo_mm[:] = 0.0
        return float(dl), float(dr)

    def fingers(self, opened=True):
        # RADIANS -- actuator ctrlrange is not converted by compiler angle="degree"
        if self.a_fl < 0:            # fingerless build: ctrl[-1] would hit the cradle
            return
        a = np.radians(AgentA.FINGER_OPEN if opened else AgentA.FINGER_RAKE)
        self.d.ctrl[self.a_fl] = a
        self.d.ctrl[self.a_fr] = -a

    def intake(self, collecting, rpm=None):
        """Knife down + brush roller spinning, or roller stopped + knife up.

        The knife is servo-lifted SHIM_LIFT deg on every non-collecting leg: pressed
        to the floor it bulldozes already-placed discs.  Ordering matters on
        the real machine -- spinning fingers strike a LIFTED knife tip (they
        clear a lowered one by ~2 mm) -- so the roller stops before the knife
        lifts and the knife drops before the roller starts; here both commands
        land the same tick and the model's masks make the order moot.
        """
        if self.a_shim < 0:
            return
        if collecting:
            self.d.ctrl[self.a_shim]   = np.radians(AgentA.SHIM_DROOP)
            self.d.ctrl[self.a_roller] = 2*np.pi*(rpm or AgentA.ROLL_RPM)/60.0
        else:
            self.d.ctrl[self.a_roller] = 0.0
            self.d.ctrl[self.a_shim]   = -np.radians(AgentA.SHIM_LIFT)

    # ------------------------------------------------------------ mission 2
    def _find_kits(self):
        if not hasattr(self, "_kit_eq"):
            self._kit_eq = []
            for i in range(64):
                e = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_EQUALITY,
                                      "kit%d_hold" % i)
                if e < 0: break
                self._kit_eq.append(e)
        return self._kit_eq

    def kits_aboard(self):
        eq = self._find_kits()
        return sum(1 for e in eq if self.d.eq_active[e])

    def drop_kits(self, n):
        """Release the next n kits, in index order.  Kept for the rigs."""
        eq = self._find_kits()
        live = [e for e in eq if self.d.eq_active[e]]
        for e in live[:n]:
            self.d.eq_active[e] = 0
        return min(n, len(live))

    def open_hopper(self, dest):
        """Open one destination's flap.  The rules let kits start ON the robot
        (g.1) and they are loaded already grouped, so this is the ONLY verb
        Mission 2 needs for a kit -- no pick-up, no sorting, no singulation.
        One MG90S per hopper, opened once each in the match.

        In the model it is a weld going inactive: the kits then fall under
        gravity from wherever the hopper is, and the referee reads where they
        land, exactly as it does for a beam.
        """
        eq = self._find_kits()
        n = 0
        for i in M2.KIT_GROUPS.get(dest, ()):
            if i < len(eq) and self.d.eq_active[eq[i]]:
                self.d.eq_active[eq[i]] = 0
                n += 1
        return n

    def feed(self, down):
        """Positive-feed plunger.  Parked its face is 33 mm above the highest a
        piece ever reaches, so the drop path stays clear; one stroke presses the
        column down onto the stack and re-seats anything shaken loose."""
        self.d.ctrl[self.a_feed] = -mm(AgentA.FEED_STROKE) if down else 0.0

    def gate(self, opened):
        """Escapement shelf: two leaves that carry the column and retract to
        OPPOSITE sides to release one.  The command is the coupled tendon's
        length, which is the sum of both leaf strokes -- one pinion, two racks
        (F68)."""
        self.d.ctrl[self.a_gate] = 2*mm(AgentA.ESC_Y) if opened else 0.0

    def blade(self, inserted):
        """Escapement retainer: two 1 mm knives that take the column at the
        joint above the bottom disc, so the shelf can retract from under just
        that one.  Parked they are clear of the bore."""
        self.d.ctrl[self.a_blade] = 2*mm(AgentA.ESC_BLADE_PARK) if inserted else 0.0

    def cradle(self, which, carry):
        """Beam cradle 1 (pocket R, beam 1) or 2 (pocket L, beam 2).

        carry=True lifts the beam CARRY_Z off the field -- which is how it
        crosses the laboratory and how the robot is allowed to pivot at all
        (F46).  carry=False sets it down on the field; the shelves then sit in
        the floor plane and the robot can simply back away from the piece.
        """
        a = self.a_cr[which - 1]
        if a >= 0:
            self.d.ctrl[a] = 0.0 if carry else mm(AgentA.CARRY_Z + AgentA.CRADLE_DROP)
        # The clamp goes with it: carried means wedged against the pocket wall,
        # released means standing on the field on its own (F50).
        e = self.eq_beam[which - 1]
        if e >= 0:
            self.d.eq_active[e] = 1 if carry else 0

    def cradle_down(self, which, tol=0.6):
        """Has the cradle finished its stroke?  A limit switch reads this."""
        j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "A_cr%d_j" % which)
        full = AgentA.CARRY_Z + AgentA.CRADLE_DROP
        return self.d.qpos[self.m.jnt_qposadr[j]]*1000.0 > full - tol

    # ------------------------------------------------------------- trim slide
    def trim(self, y_mm):
        """Aim the posting head.  +y is the robot's LEFT (F68)."""
        if self.a_trim < 0:
            return
        self.d.ctrl[self.a_trim] = mm(float(np.clip(y_mm, -AgentA.TRIM_Y,
                                                    AgentA.TRIM_Y)))

    def trim_at(self):
        """Where the slide actually IS, mm -- a servo's own feedback pot."""
        if self.j_trim < 0:
            return 0.0
        return self.d.qpos[self.m.jnt_qposadr[self.j_trim]]*1000.0

    def trim_settled(self, tol=0.25):
        want = self.d.ctrl[self.a_trim]*1000.0 if self.a_trim >= 0 else 0.0
        return abs(self.trim_at() - want) < tol

    def mag_count(self):
        """Pieces in the magazine, from the bore rangefinder.  Empty reads the
        shelf at Za 11; each piece brings the surface up 5 mm."""
        r = self.d.sensordata[self.m.sensor_adr[self.s_mag]]
        if r < 0:                        # no return at all
            return 0
        top = 70.0 - r*1000.0            # site is at Za 70 looking down
        return int(max(0.0, round((top - AgentA.CHUTE_Z0) / Piece.DISC_T)))

    # ------------------------------------------------------------ perception
    def _cam(self):
        if not hasattr(self, "_cam_id"):
            self._cam_id = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA,
                                              "A_cam_%s" % t) for t in ("l", "r")]
            # THE EXTRINSIC IS A BIAS, NOT NOISE.  Where the camera plate sits
            # relative to the bore is a bracket on a robot that gets driven into
            # walls on purpose; it is wrong by the same amount all match, and a
            # model that redraws it every frame averages it away and flatters
            # itself.  Drawn once, at construction, from the match's own seed.
            self._ext = self.rng.normal(0.0, Vision.EXT_SIGMA, 3)
            self._ext_a = np.radians(self.rng.normal(0.0, Vision.EXT_ANG_SIGMA))
            # Per-SLOT detection bias: a rim fit's residual is a property of
            # that slot's marking and the angle it is seen from, so it is the
            # same all match for a given slot and different between slots.
            self._det = self.rng.normal(0.0, Vision.DET_BIAS,
                                        (len(Field.LAB_HOLE_X), 2))
        return self._cam_id

    def cam_frame(self, i=0):
        """(origin_world_mm, R) for one camera.  MuJoCo's camera looks down its
        own -z with +x right and +y up, which is also how a rectified pair is
        described, so the columns of R are the image axes."""
        c = self._cam()[i]
        return self.d.cam_xpos[c]*1000.0, self.d.cam_xmat[c].reshape(3, 3)

    def _project(self, p_world_mm, i=0):
        """(u, v, z) in pixels and mm along the view axis, or None if behind."""
        o, R = self.cam_frame(i)
        cam = R.T @ (np.asarray(p_world_mm, float) - o)
        z = -cam[2]
        if z <= 1e-6:
            return None
        f = Vision.f_px()
        return (f*cam[0]/z, f*cam[1]/z, z)

    def _visible(self, p_world_mm, i=0):
        """In frame AND in line of sight, for camera i.

        Occlusion is cast with mj_ray rather than argued about: the robot's own
        tail shell, the mast, a carried beam and the laboratory's lip are all in
        the model, so the honest test is whether the ray gets there.
        """
        uv = self._project(p_world_mm, i)
        if uv is None:
            return None
        u, v, z = uv
        if abs(u) > Vision.W/2 or abs(v) > Vision.H/2 or z > Vision.Z_MAX:
            return None
        o, _ = self.cam_frame(i)
        vec = np.asarray(p_world_mm, float) - o
        rng = np.linalg.norm(vec)/1000.0
        u3 = vec/np.linalg.norm(vec)
        # START THE RAY OUTSIDE THE HOUSING.  Cast from the optical centre it
        # hits the camera's own plate a few mm out and every slot reads
        # occluded.  45 mm clears the plate and the mast, and everything that
        # could really be in the way -- the tail shell, a carried beam, the
        # posting head -- is further out than that.
        SKIN = 0.045
        gid = np.zeros(1, np.int32)
        hit = mujoco.mj_ray(self.m, self.d, o/1000.0 + u3*SKIN, u3, None, 1, -1, gid)
        if hit >= 0 and hit < rng - SKIN - 0.0015:
            return None
        return (u, v, z)

    def _sees_slot(self, p, i):
        """Is the WHOLE rim of a slot centred at p in camera i's frame?

        Checked against an actual depth render: a blob that runs off the edge of
        the image has no centroid worth having, and the model was reporting
        those as measurements.  Four rim points catch it and cost nothing.
        """
        if self._visible(p, i) is None:
            return None
        r = Field.LAB_HOLE_D/2.0
        for o_ in ((r, 0, 0), (-r, 0, 0), (0, r, 0), (0, -r, 0)):
            if self._visible(p + np.array(o_), i) is None:
                return None
        return self._visible(p, i)

    def see_lab(self, plate_top=None):
        """Measure the laboratory's slots.  Returns [(x, y, z, mode)] in the
        ROBOT frame, for the slots the rig can actually see.

        Two modes, and they are different physics on the same pair of images:

          * 'stereo' -- the slot is in BOTH frames, so its centre triangulates.
            No disparity search, therefore no minimum range: this is why a
            self-built pair beats a depth-map module here, whose 173-697 mm
            floor would blind it for the whole approach.
          * 'mono'   -- only one camera has it, but the slot is a circle of
            KNOWN diameter (rules 3.2 give 60 mm), so its range comes from the
            apparent size instead.  Lateral is unaffected; range costs a factor
            of two, and lateral is the axis the trim slide spends.

        The error that survives either way is the plate-to-bore calibration and
        the mast's flex, which is why both are modelled as bias rather than
        noise -- see Vision.

        vision="render" swaps this synthetic model for the real thing: the
        frames are rendered and perception.LabPipeline measures them, with
        the same output contract.  (The rendered path carries the mount bias
        through its calibration; mast flex stays model-only for now -- the
        camplate is welded in the MJCF -- and the budget carries it as
        margin.)
        """
        if self.vision_mode == "render":
            return self._see_lab_px()
        self._cam()
        top = Field.LAB_PLATE_T if plate_top is None else plate_top
        out = []
        x, y, th = self.pose
        t = np.radians(th)
        speed = float(np.linalg.norm(self.d.qvel[0:2]))*1000.0
        flex = np.radians(Vision.FLEX_DEG * min(speed/Vision.FLEX_REF, 2.0))
        for si, hx in enumerate(Field.LAB_HOLE_X):
            p = np.array([hx, _LAB_HOLE_Y(), top])
            vis = [self._sees_slot(p, i) for i in (0, 1)]
            if vis[0] is None and vis[1] is None:
                continue
            i = 0 if vis[0] is not None else 1
            u, v, z = vis[i]
            both = vis[0] is not None and vis[1] is not None
            sl = Vision.sigma_lat(z)
            sz = (Vision.sigma_z_stereo(z) if both
                  else Vision.sigma_z_mono(z, Field.LAB_HOLE_D))
            # flex tilts the whole rig, so it moves the answer by range*angle
            sz = np.hypot(sz, z*flex)
            sl = np.hypot(sl, z*flex)
            f = Vision.f_px()
            o, R = self.cam_frame(i)
            m = np.array([(u*z/f) + self.rng.normal(0, sl) + self._det[si, 0],
                          (v*z/f) + self.rng.normal(0, sl) + self._det[si, 1],
                          -(z + self.rng.normal(0, sz))])
            ca, sa = np.cos(self._ext_a), np.sin(self._ext_a)
            m = np.array([ca*m[0] - sa*m[1], sa*m[0] + ca*m[1], m[2]]) + self._ext
            w = o + R @ m
            dx, dy = w[0] - x, w[1] - y
            out.append((dx*np.cos(-t) - dy*np.sin(-t),
                        dx*np.sin(-t) + dy*np.cos(-t),
                        w[2], "stereo" if both else "mono"))
        return out

    def _see_lab_px(self):
        """The rendered path: frames from the tail cameras, measurements from
        perception.LabPipeline.  Built lazily so the model-camera path never
        pays for a GL context -- and never consumes the rng draws the bias
        needs, keeping model-camera runs bit-identical with or without this
        code existing."""
        if self._pix is None:
            from . import perception
            cams = SimCameras(self.m, self.d, rng=self.rng)
            self._pix = (cams, perception.LabPipeline(cams.calib()))
        cams, pipe = self._pix
        imgL, imgR, _ = cams.frames()
        return pipe.slots(imgL, imgR)

    # ------------------------------------------------------------ stallguard
    def stalled(self, thresh=0.42):
        """TMC2209 StallGuard stand-in: both drivers at torque saturation.
        Unreliable below ~0.1 m/s, exactly as the real part is."""
        return (abs(self.d.actuator_force[self.a_l]) > thresh and
                abs(self.d.actuator_force[self.a_r]) > thresh)


def _LAB_HOLE_Y():
    """The laboratory's slot line.  Imported lazily so robot.py does not depend
    on mjcf.py at import time (mjcf imports params, params imports nothing)."""
    from .mjcf import LAB_HOLE_Y
    return LAB_HOLE_Y


class SimClock(hal.Clock):
    """The simulator's control tick: one tick = CTRL_DECIM physics substeps.

    Owns the decimation the run scripts hard-code today (1 kHz physics to
    50 Hz control), derived from the model's own timestep so a model change
    cannot silently skew the control rate.  The Pi's clock sleeps to the
    next 20 ms wall boundary instead; mission code cannot tell them apart.
    """
    def __init__(self, model, data, decim=None):
        self.m, self.d = model, data
        self.decim = decim or int(round(self.PERIOD / model.opt.timestep))

    def now(self):
        return self.d.time

    def tick(self):
        for _ in range(self.decim):
            mujoco.mj_step(self.m, self.d)


class SimCameras(hal.CameraHAL):
    """The tail pair, rendered: the simulator's CameraHAL.

    frames() renders both eyes offscreen at the perception resolution; the
    physics is untouched (rendering reads d, never writes it).  calib() is
    built from Vision.cam_pose -- the same statement the MJCF emits the
    <camera> tags from -- plus, when an rng is given, the per-match mounting
    bias, folded into the calibration rather than the render: the world does
    not move when a bracket is bent, the robot's belief does.
    """
    def __init__(self, model, data, rng=None):
        self.m, self.d = model, data
        self._r = None
        from . import perception
        self._calib = perception.sim_calib(rng)

    def _renderer(self):
        if self._r is None:
            self._r = mujoco.Renderer(self.m, Vision.H, Vision.W)
        return self._r

    def frames(self):
        r = self._renderer()
        r.update_scene(self.d, camera="A_cam_l")
        imgL = r.render().copy()
        r.update_scene(self.d, camera="A_cam_r")
        return imgL, r.render(), float(self.d.time)

    def calib(self):
        return self._calib
