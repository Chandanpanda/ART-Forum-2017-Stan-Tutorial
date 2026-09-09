"""Tier 2: one joint wound with a real thread -- the cable rig.

Tier 1 lays the band as arithmetic: a turn is a turn the fiducial counted,
placed where the feed had reached.  This rig winds a THREAD -- a MuJoCo
cable, a chain of capsules with bending stiffness from the elasticity
plugin -- round one joint of the truss and lets the physics say where the
hoops lie, how much thread each takes, whether the mitred diagonals stay
where the fixture left them, and whether the wound band alone holds them
once the keeper lets go.

THE RIG IS A LATHE.  The joint's cluster -- its chord, its diagonals in
their cradles -- sits on a spindle turning about the chord's axis under a
fixed exit guide that feeds along x, and the thread pays out from the
guide under a constant tension.  Relative motion is all winding is; a
spinning ring with a spool on its rim needs its pay-out reserve carried
round with it, and a 50 mm reserve on a 20 mm ring sweeps the cage.  So
the rig turns the work instead, and the exit guide is the ring's, at
Ring.EXIT_R from the chord's axis.

One thing the lathe does not share with the ring: the anchor turns with
the work, so the strand between the anchor and the fixed guide wraps at
the ANCHOR's x, not the guide's.  Anchored a 10 mm lead-in before the
band, every hoop landed 10 mm before the band.  The anchor is at the
band's first hoop.

WHAT IS A PROXY AND WHY.  MuJoCo's contacts and welds are impedances that
scale with the mass they act on: a 2 N tension through segments weighing
what 1 mm of 0.15 mm Kevlar weighs (25 micrograms) put the thread a metre
into whatever it touched.  So the thread here is 1 g a millimetre and
twice its real radius, and it is wound at 0.5 N instead of 2 N.  The
ratios that decide what the rig measures survive: tension to the thread's
own weight per span, thousands to one either way; tension to a diagonal's
weight, 100:1 here against 400:1 on the machine.  The hoop's pitch, the
radius it settles at, the thread a hoop takes and the hold on the mitres
are geometry and friction, and those are the real ones.  Measured while
choosing these (sim/scripts/truss/check_ring.py records them): at 2 N with
this proxy the thread cut 0.17 mm into the chord and the pay-out stalled;
an exit guide with the rods' friction (0.5) held the thread back and the
first hoop went nowhere -- a real guide is a ceramic eyelet, EXIT_MU.
"""
from math import pi, radians

import numpy as np
import mujoco

from .spec import Ring, Cage
from .geometry import theta_chord_up, rot_x
from . import mjcf as _m

# rig facts and proxies (see the module docstring)
PROXY_R_SCALE = 2.0          # thread radius, times the real one
PROXY_SEG_MM = 1.0           # cable segment length
PROXY_SEG_MASS = 1e-3        # kg per segment
RIG_TENSION = 0.5            # N, scaled from the recipe's 2 N with the mass
EXIT_MU = 0.1                # a ceramic eyelet
LEAD_IN = 1.0                # mm, the anchor's tail before the first hoop
FLYWHEEL_KG = 0.1            # the spindle's inertia, so its servo is tame
TIMESTEP = 2e-4
# the keeper stands for a CLAMP: the rig's question is what the thread does
# to a rod the fixture is holding, so the constraint must not be the thing
# that yields.  At the default weld stiffness the diagonals sagged 0.06 mm
# under their own weight and 0.40 mm under the winding tension, and that
# number was the solver's, not the machine's.
KEEP_SOLREF = 5.0 * TIMESTEP


class JointRig:
    def __init__(self, geom, fixture, joint, turns=4, thread=True, rng=None):
        self.geom, self.fixture, self.joint = geom, fixture, joint
        self.turns, self.thread = turns, thread
        t = geom.t
        self.r_cable = PROXY_R_SCALE * t.thread_d / 2.0
        # THE BAND'S WIDTH IS THE RECIPE'S; THE TURN COUNT IS NOT.  The
        # recipe's 18 turns is 289 cable bodies at 240 ms a step -- an hour
        # for one joint.  Four hoops at band/4 cover the same width, cross
        # the same mitres and lie on the same cluster; what they cannot
        # measure is how a hoop beds against its neighbour.  (At the
        # recipe's own pitch four turns covered 1.8 mm of the 8 and never
        # reached the mitres, so the hold test had nothing to hold.)
        self.pitch = t.band / self.turns
        self.x_start = -t.band / 2.0                # band start, rig frame (joint at 0)
        self.xml = self._build()
        self.m = mujoco.MjModel.from_xml_string(self.xml)
        self.d = mujoco.MjData(self.m)
        nm = lambda typ, n: mujoco.mj_name2id(self.m, typ, n)
        B, J, S, A, E = (mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_SITE,
                         mujoco.mjtObj.mjOBJ_ACTUATOR, mujoco.mjtObj.mjOBJ_EQUALITY)
        self.j_spin = self.m.jnt_qposadr[nm(J, "spin")]
        self.v_spin = self.m.jnt_dofadr[nm(J, "spin")]
        self.j_feed = self.m.jnt_qposadr[nm(J, "feed")]
        self.a_spin, self.a_feed = nm(A, "a_spin"), nm(A, "a_feed")
        self.s_far = nm(S, "far")
        self.b_spindle = nm(B, "spindle")
        self.b_diag = {ri: nm(B, "diag%d" % ri) for ri in joint.diags}
        self.eq_keep = {ri: nm(E, "keep%d" % ri) for ri in joint.diags}
        names = [mujoco.mj_id2name(self.m, B, i) for i in range(self.m.nbody)]
        self.cable = [i for i, n in enumerate(names) if n and n.startswith("B_")]
        self.b_last = names.index("B_last") if "B_last" in names else None
        mujoco.mj_forward(self.m, self.d)
        self.z_end0 = float(self.d.xpos[self.b_last][2]) if self.b_last is not None else 0.0

    # ------------------------------------------------------------ frame
    def _T(self):
        """Cage-frame point -> rig frame (chord axis along x, joint at 0)."""
        th = theta_chord_up(self.joint.chord)
        Rm = rot_x(th)
        cj = Rm @ self.geom.chord_point(self.joint.chord, self.joint.x)
        return Rm, cj

    # ------------------------------------------------------------ build
    def _build(self):
        g, t, j = self.geom, self.geom.t, self.joint
        Rm, cj = self._T()
        T = lambda p: Rm @ np.asarray(p, float) - cj
        R = lambda v: Rm @ np.asarray(v, float)
        r_c = t.d_chord / 2.0
        half = t.band + 30.0
        parts = ['<body name="spindle" pos="0 0 0">',
                 '  <joint name="spin" type="hinge" axis="1 0 0" damping="1e-3"/>',
                 '  ' + _m.capsule("chord", (-half, 0, 0), (half, 0, 0), r_c, _m.C_ROD, 1, 1,
                                   mass=t.mass_chords / t.n_chords * (2 * half / t.length),
                                   cls=' friction="0.6 0.005 0.0001"'),
                 '  ' + _m.cylinder("flywheel", (-half - 4.0, 0, 0), (-half - 2.0, 0, 0), 20.0,
                                    "0.4 0.4 0.45 0.3", 0, 0, mass=FLYWHEEL_KG * 1000.0)]
        for i, c in enumerate(self.fixture.cradles):
            if c.rod not in j.diags:
                continue
            parts += ["  " + s for s in _m.v_flanks("cradle%d" % i, T(c.apex), R(c.axis), R(c.up),
                                                    Cage.CRADLE_L, Cage.NOTCH_DEPTH * 0.8,
                                                    Cage.CRADLE_T, Cage.CRADLE_ANGLE / 2.0,
                                                    _m.C_CAGE, 1, 1)]
            parts.append("  " + _m.capsule("cstem%d" % i, T(c.apex) - R(c.up) * 2.0,
                                           T(c.apex) - R(c.up) * 14.0, 1.5, _m.C_CAGE, 1, 1))
        if self.thread:
            # the thread: anchored on the chord's crest a lead-in before the
            # band, up through the exit guide, then the reserve straight up
            reserve = self.thread_expected() * 1.4 + 10.0
            a = np.array([self.x_start - LEAD_IN, 0.0, r_c + self.r_cable])
            e = np.array([self.x_start, 0.0, Ring.EXIT_R])
            top = e + np.array([0.0, 0.0, reserve])
            V = [a]
            for p, q in ((a, e), (e, top)):
                n = max(1, int(round(np.linalg.norm(q - p) / PROXY_SEG_MM)))
                V += [p + (q - p) * k / n for k in range(1, n + 1)]
            verts = " ".join("%.6f %.6f %.6f" % tuple(_m.mm(x) for x in p) for p in V)
            parts += ['  <composite type="cable" vertex="%s" initial="none">' % verts,
                      '    <plugin plugin="mujoco.elasticity.cable">',
                      '      <config key="twist" value="1e6"/>',
                      '      <config key="bend" value="1e6"/>',
                      '      <config key="vmax" value="0"/>',
                      '    </plugin>',
                      '    <joint kind="main" damping="1e-4"/>',
                      '    <geom type="capsule" size="%.6f" mass="%.6f" friction="0.5 0.005 0.0001" '
                      'rgba="0.9 0.75 0.1 1"/>' % (_m.mm(self.r_cable), PROXY_SEG_MASS),
                      '  </composite>']
        parts.append('</body>')
        # the exit guide on the feed carriage: a square tube round the thread
        h = self.r_cable + 0.6
        e = (self.x_start, 0.0, Ring.EXIT_R)
        parts += ['<body name="carriage" pos="0 0 0">',
                  '  <joint name="feed" type="slide" axis="1 0 0" damping="1"/>']
        for tag, c, sz in (("l", (e[0], h + 0.5, e[2]), (0.6, 0.5, 2.0)),
                           ("r", (e[0], -h - 0.5, e[2]), (0.6, 0.5, 2.0)),
                           ("f", (e[0] + h + 0.5, 0.0, e[2]), (0.5, h, 2.0)),
                           ("b", (e[0] - h - 0.5, 0.0, e[2]), (0.5, h, 2.0))):
            parts.append("  " + _m.box("eye_%s" % tag, c, sz, (1, 0, 0), (0, 1, 0), "0.8 0.8 0.85 1",
                                       1, 1, mass=10.0, extra=' friction="%g 0.005 0.0001"' % EXIT_MU))
        parts += ['  <site name="far" pos="%s" size="0.001"/>' % _m._v((e[0], 0.0, 300.0)),
                  '</body>']
        # the diagonals: free bodies at their seated poses, kept by welds
        for ri in j.diags:
            rod = g.rods[ri]
            p0, p1 = (T(p) for p in g.diag_body_ends(rod))
            mid = (p0 + p1) / 2.0
            parts += ['<body name="diag%d" pos="%s">' % (ri, _m._v(mid)),
                      '  <freejoint name="diag%d_f"/>' % ri,
                      '  ' + _m.capsule("diag%d_g" % ri, p0 - mid, p1 - mid, rod.r, _m.C_ROD, 1, 1,
                                        mass=t.mass_diags / max(t.n_diag, 1),
                                        cls=' friction="0.4 0.005 0.0001"'),
                      '</body>']
        parts.append('<geom name="floor" type="plane" pos="0 0 -0.2" size="1 1 0.1" contype="1" conaffinity="1"/>')
        eq = ["  <equality>"] + [
            '    <weld name="keep%d" body1="spindle" body2="diag%d" solref="%g 1" '
            'solimp="0.999 0.9999 0.0001"/>' % (ri, ri, KEEP_SOLREF) for ri in j.diags] + ["  </equality>"]
        return """<mujoco model="joint_rig">
  <extension><plugin plugin="mujoco.elasticity.cable"/></extension>
  <compiler angle="degree" autolimits="true"/>
  <option timestep="%g" integrator="implicitfast" gravity="0 0 -9.81"/>
  <default>
    <geom condim="3" friction="0.5 0.005 0.0001" solref="0.002 1" solimp="0.99 0.999 0.001"/>
  </default>
  <visual><headlight ambient="0.7 0.7 0.7" diffuse="0.5 0.5 0.5"/></visual>
  <worldbody>
%s
  </worldbody>
%s
  <actuator>
    <velocity name="a_spin" joint="spin" kv="0.05" forcerange="-2 2"/>
    <position name="a_feed" joint="feed" kp="500" kv="5"/>
  </actuator>
</mujoco>""" % (TIMESTEP, "\n".join("    " + p for p in parts), "\n".join(eq))

    # ------------------------------------------------------------ drive
    def _pull(self):
        if self.b_last is None:
            return
        p = self.d.xpos[self.b_last]
        f = self.d.site_xpos[self.s_far]
        u = (f - p) / np.linalg.norm(f - p)
        self.d.xfrc_applied[self.b_last, :3] = RIG_TENSION * u

    def step(self, n):
        for _ in range(n):
            self._pull()
            mujoco.mj_step(self.m, self.d)

    def settle(self, seconds=0.3):
        self.step(int(seconds / TIMESTEP))

    def angle(self):
        return float(self.d.qpos[self.j_spin])

    def wind(self, rpm=None):
        """Spin the recipe's turns at the ring's rpm, feeding a pitch per
        turn; returns the wall time per step."""
        import time
        rpm = Ring.RPM if rpm is None else rpm
        w = rpm / 60.0 * 2 * pi
        a0 = self.angle()
        n = int(self.turns / (rpm / 60.0) / TIMESTEP)
        t0 = time.time()
        for _ in range(n):
            self.d.ctrl[self.a_spin] = w
            self.d.ctrl[self.a_feed] = _m.mm(self.pitch) * ((self.angle() - a0) / (2 * pi))
            self._pull()
            mujoco.mj_step(self.m, self.d)
        el = time.time() - t0
        self.d.ctrl[self.a_spin] = 0.0
        self.settle(0.5)
        return el / n

    def turn_to(self, deg, seconds=2.0):
        """Turn the spindle to `deg` (rig frame) at a steady rate."""
        a0 = self.angle()
        target = radians(deg)
        n = int(seconds / TIMESTEP)
        for k in range(n):
            want = a0 + (target - a0) * (k + 1) / n
            self.d.ctrl[self.a_spin] = (want - self.angle()) / TIMESTEP * 0.1
            self._pull()
            mujoco.mj_step(self.m, self.d)
        self.d.ctrl[self.a_spin] = 0.0

    def keeper_yield(self, force=None):
        """How far the keeper lets a diagonal move under a steady pull of
        `force` newtons -- the constraint's own compliance, so a movement
        measured during winding can be told from the thread dragging the
        rod.  Restores the load before returning."""
        force = RIG_TENSION if force is None else force
        base = self.seat_errors()
        for b in self.b_diag.values():
            self.d.xfrc_applied[b, :3] = np.array([0.0, 0.0, -force])
        self.settle(0.3)
        loaded = self.seat_errors()
        for b in self.b_diag.values():
            self.d.xfrc_applied[b, :3] = 0.0
        self.settle(0.3)
        return max(abs(loaded[k] - base[k]) for k in base)

    def release_keepers(self):
        for e in self.eq_keep.values():
            self.d.eq_active[e] = 0

    # ---------------------------------------------------------- measure
    def spun_turns(self):
        return self.angle() / (2 * pi)

    def pay_out(self):
        """mm of thread that left the reserve."""
        if self.b_last is None:
            return 0.0
        return (self.z_end0 - float(self.d.xpos[self.b_last][2])) * 1000.0

    def hoops(self):
        """The thread on the cluster: (turns wrapped, x of the on-chord
        segments (mm), their radii from the chord axis (mm))."""
        if not self.cable:
            return 0.0, np.zeros(0), np.zeros(0)
        P = np.array([self.d.xpos[i] for i in self.cable]) * 1000.0
        rho = np.hypot(P[:, 1], P[:, 2])
        # THE CLUSTER GROWS WITH |x|: a hoop 4 mm from the joint rides over
        # a diagonal 4 mm out from the axis.  Judged against the reach at
        # the joint alone, every segment that had climbed a diagonal was
        # thrown away as "off the cluster", and the thread the rig reported
        # was the thread on the chord.
        reach = np.array([self.geom.cluster_reach(self.joint, float(x)) for x in P[:, 0]])
        on = rho < reach + self.r_cable + 1.0
        idx = np.where(on)[0]
        if len(idx) < 3:
            return 0.0, np.zeros(0), np.zeros(0)
        az = np.unwrap(np.arctan2(P[idx, 2], P[idx, 1]))
        return abs(az[-1] - az[0]) / (2 * pi), P[idx, 0], rho[idx]

    def seat_errors(self):
        """Each diagonal's distance from where the fixture holds it, in the
        spindle's frame, mm: the larger of its two ends' offsets."""
        Rm, cj = self._T()
        Rs = rot_x(np.degrees(self.angle()))
        out = {}
        for ri, b in self.b_diag.items():
            rod = self.geom.rods[ri]
            a, c = (Rs @ (Rm @ p - cj) for p in self.geom.diag_body_ends(rod))
            u = (c - a) / np.linalg.norm(c - a)
            # the CAPSULE's frame, not the body's: fromto rotates the geom
            # inside a body whose own frame stays the identity, and a body
            # z-axis read instead put every diagonal 35 mm off before
            # anything had moved
            gid = self.m.body_geomadr[b]
            pos = self.d.geom_xpos[gid] * 1000.0
            ax = self.d.geom_xmat[gid].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
            hl = self.m.geom_size[gid][1] * 1000.0
            ends = [pos - ax * hl, pos + ax * hl]

            def off(p):
                v = p - a
                return float(np.linalg.norm(v - (v @ u) * u))
            out[ri] = max(off(p) for p in ends)
        return out

    def thread_expected(self):
        """Thread the band's arithmetic says these turns take, mm."""
        xs = self.x_start + (np.arange(self.turns) + 0.5) * self.pitch
        return float(sum(self.geom.hoop_perimeter(self.joint, float(x), 2.0 * self.r_cable)
                         for x in xs))

    def arc_on_cluster(self):
        """Length of cable actually lying on the cluster, mm -- what the
        hoops took, measured rather than paid out (the pay-out also holds
        the tail and whatever is still in the guide)."""
        if not self.cable:
            return 0.0
        P = np.array([self.d.xpos[i] for i in self.cable]) * 1000.0
        rho = np.hypot(P[:, 1], P[:, 2])
        reach = np.array([self.geom.cluster_reach(self.joint, float(x)) for x in P[:, 0]])
        on = rho < reach + self.r_cable + 1.0
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        return float(seg[on[:-1] & on[1:]].sum())

    def hoop_radii(self):
        """(x, radius) of every cable segment on the cluster, and the
        cluster's own hull radius in that segment's direction -- the
        measurement that says whether the thread rides the hull or beds
        against the chord under the diagonals."""
        if not self.cable:
            return np.zeros((0, 3))
        P = np.array([self.d.xpos[i] for i in self.cable]) * 1000.0
        rho = np.hypot(P[:, 1], P[:, 2])
        reach = np.array([self.geom.cluster_reach(self.joint, float(x)) for x in P[:, 0]])
        out = []
        for i in np.where(rho < reach + self.r_cable + 1.0)[0]:
            u = np.array([P[i, 1], P[i, 2]]) / max(rho[i], 1e-9)
            pts = self.geom.cluster_points(self.joint, float(P[i, 0]))
            out.append((P[i, 0], rho[i], float(max(pts @ u))))
        return np.array(out) if out else np.zeros((0, 3))
