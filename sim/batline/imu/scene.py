"""The gimbal and the board's IMU in MuJoCo.  TRUTH: the simulation reads
the IMU's true motion here, and check_imu holds kin.py, the dynamics and
the numpy masses to it.  No procedure imports this.

The scene is rig/mjcf.py's gimbal as built (the same XML the cameras
render), weighed -- every geom given its material's density, the bat its
parts' own masses, the solver's proxy boxes none -- with each motor's
rotor reflected through its belt as the hinge's armature.  On the inner
body sit two sites, each with a gyro and an accelerometer: `clamp` at the
body's origin and `imu`, which place() moves to any board without a
recompile.

MuJoCo knows nothing of the Earth's turn, so a reading here is the motion
relative to the room; sim.py adds the Earth's rate.
"""
from math import radians

import numpy as np
import mujoco

from ..spec import Stepper, Gimbal
from ..rig import mjcf as RM
from ..rig.inertia import inertials
from .kin import Chain


def armature():
    """kg m^2: each motor's rotor as its hinge feels it, through the belt."""
    return Stepper.ROTOR_J * Gimbal.GEAR ** 2


def chain_of(b):
    """The build's true chain (kin.Chain) from a RigBuildDraw."""
    return Chain.of(b.centre, b.axis_o, b.axis_i, b.offset_i, b.zero)


def imu_xml(rig, b):
    sites = ['<site name="clamp" pos="0 0 0" size="0.002"/>',
             '<site name="imu" pos="0 0 0" size="0.002"/>']
    bodies = RM.gimbal_xml(rig, b, payload="bat", weighed=True, inner_extra=sites)
    arm = armature()
    sens = []
    for s in ("clamp", "imu"):
        sens += ['<gyro name="%s_gyro" site="%s"/>' % (s, s),
                 '<accelerometer name="%s_acc" site="%s"/>' % (s, s)]
    sens += ['<framepos name="imu_pos" objtype="site" objname="imu"/>',
             '<framequat name="imu_quat" objtype="site" objname="imu"/>']
    xml = """<mujoco model="imu">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option gravity="0 0 0" timestep="0.001"/>
  <asset>
    <material name="matte" specular="0" shininess="0" reflectance="0"/>
    <material name="retro" rgba="1 1 1 1" emission="1" specular="0" shininess="0"/>
  </asset>
  <default>
    <geom material="matte" contype="0" conaffinity="0" density="%.6f"/>
    <joint armature="%.12g"/>
  </default>
  <worldbody>
%s
  </worldbody>
  <sensor>
%s
  </sensor>
</mujoco>
""" % (RM.DEFAULT_DENSITY, arm, "\n".join("    " + w for w in bodies), "\n".join("    " + s for s in sens))
    return xml, bodies


class Scene:
    """The weighed gimbal as built, with its two sensor sites."""

    def __init__(self, rig, b):
        self.rig, self.b = rig, b
        self.xml, self.bodies = imu_xml(rig, b)
        self.m = mujoco.MjModel.from_xml_string(self.xml)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.qa = m.joint("alpha").qposadr[0]
        self.qb = m.joint("beta").qposadr[0]
        self.va = m.joint("alpha").dofadr[0]
        self.vb = m.joint("beta").dofadr[0]
        self.inner = m.body("inner").id
        self.site = m.site("imu").id

        def adr(name):
            s = m.sensor(name)
            return slice(s.adr[0], s.adr[0] + s.dim[0])
        self.s = {n: adr(n) for n in ("clamp_gyro", "clamp_acc", "imu_gyro", "imu_acc", "imu_pos", "imu_quat")}
        self.zero = np.radians(np.asarray(b.zero, float))
        self.chain = chain_of(b)

    def numpy_inertials(self):
        """The same bodies' masses, by rig/inertia.py: (outer, inner)."""
        I = inertials("\n".join(self.bodies))
        return I["outer"], I["inner"]

    def gravity(self, g_w):
        self.m.opt.gravity[:] = g_w

    def place(self, r_c, R_ci):
        """The imu site at r_c (m, clamp frame) with the chip's axes R_ci
        (columns, clamp frame)."""
        self.m.site_pos[self.site] = r_c
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, np.asarray(R_ci, float).ravel())
        self.m.site_quat[self.site] = q
        # the site was compiled at the body's origin, so the compiler marked
        # it as sharing the body's frame and the kinematics copy the body's
        # pose to it, ignoring site_pos and site_quat: unmark it, or the imu
        # site's sensors read the clamp's origin wherever it is placed
        self.m.site_sameframe[self.site] = mujoco.mjtSameFrame.mjSAMEFRAME_NONE

    def set(self, q, qd, qdd):
        """Hinges at encoder angles q (rad), rates and accelerations: inverse
        dynamics, which also runs the sensors."""
        d = self.d
        d.qpos[self.qa] = q[0] + self.zero[0]
        d.qpos[self.qb] = q[1] + self.zero[1]
        d.qvel[self.va], d.qvel[self.vb] = qd[0], qd[1]
        d.qacc[self.va], d.qacc[self.vb] = qdd[0], qdd[1]
        mujoco.mj_inverse(self.m, d)

    def read(self, name):
        return self.d.sensordata[self.s[name]].copy()

    def pose(self):
        """(R, x m): the inner body (the clamp) in the world."""
        return self.d.xmat[self.inner].reshape(3, 3).copy(), self.d.xpos[self.inner].copy()

    def ang_acc(self):
        """rad/s^2: the inner body's angular acceleration, its own frame.
        (Asked for in the world: a body's "local" frame in MuJoCo is its
        inertial frame -- the principal axes at its centre of mass -- not
        the frame its geoms are written in.)"""
        res = np.zeros(6)
        mujoco.mj_objectAcceleration(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, self.inner, res, 0)
        return self.d.xmat[self.inner].reshape(3, 3).T @ res[:3]

    def torque(self):
        """N m: what each hinge's motor must give (the inverse dynamics)."""
        return np.array([self.d.qfrc_inverse[self.va], self.d.qfrc_inverse[self.vb]])

    def mass(self):
        """kg m^2 (2, 2): the hinges' mass matrix where set() left them."""
        M = np.zeros((self.m.nv, self.m.nv))
        try:
            mujoco.mj_fullM(self.m, self.d, M)            # MuJoCo >= 3.3
        except TypeError:
            mujoco.mj_fullM(self.m, M, self.d.qM)
        return M[np.ix_([self.va, self.vb], [self.va, self.vb])]

    def mass_matrix(self, q):
        self.set(q, (0.0, 0.0), (0.0, 0.0))
        return self.mass()


def inner_motion(scene, q, qd, qdd):
    """The clamp's true motion at hinge states (n, 2) each, from MuJoCo's
    inverse dynamics with gravity off: (R (n,3,3), x (n,3) m, w_c (n,3)
    rad/s, wd_c (n,3) rad/s^2, a_c (n,3) m/s^2 -- the origin's acceleration
    in the clamp frame), and each hinge's torque (n, 2) without gravity."""
    n = len(q)
    scene.gravity(np.zeros(3))
    R = np.empty((n, 3, 3))
    x = np.empty((n, 3))
    w = np.empty((n, 3))
    wd = np.empty((n, 3))
    a = np.empty((n, 3))
    tau = np.empty((n, 2))
    for k in range(n):
        scene.set(q[k], qd[k], qdd[k])
        R[k], x[k] = scene.pose()
        w[k] = scene.read("clamp_gyro")
        a[k] = scene.read("clamp_acc")
        wd[k] = scene.ang_acc()
        tau[k] = scene.torque()
    return R, x, w, wd, a, tau


def at_point(mot, r_c, g_w):
    """(w_c, f_c) clamp frame at clamp point r_c, from inner_motion's
    output and gravity g_w (world): the rigid body's own law."""
    R, x, w, wd, a = mot[:5]
    f = a + np.cross(wd, r_c) + np.cross(w, np.cross(w, r_c)) - np.einsum("nji,j->ni", R, g_w)
    return w, f
