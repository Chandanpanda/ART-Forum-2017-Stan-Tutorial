"""The factory iPhone's picture of the bat, rendered.  TRUTH: simulation
and checks only (check_swing, demo_phone); the app never sees it.

mjcf.scene's bat, every part at its design pose in the gimbal's clamp,
seen through the phone's unfolded camera (camera.View: Phone.F_PX at
Rules.PLAY_Z), the image Phone.W x Phone.H divided by `scale`.  The gimbal
itself is left out.  A pixel's centre is at its index plus a half, the
convention View.project's continuous coordinates take.

The near plane is moved out to PICTURE_ZNEAR.  MuJoCo reads znear as a
multiple of the model's extent, not a distance: mjcf.scene's 0.001 is
0.8 mm on the kid's scene, where a 24-bit depth buffer resolves about a
millimetre at 4 m and the bands stand only mjcf.INK proud of the sleeve;
0.05 is 41 mm, where it resolves about 0.02 mm.

The caller must have called truss.glenv.headless() before MuJoCo was
first imported, as every renderer in this repository does.
"""
import numpy as np
import mujoco

from ..spec import Phone, Rules
from .. import mjcf

PICTURE_ZNEAR = 0.05           # of the model's extent (41 mm on the kid's scene): under anything the phone sees


class Picture:
    """One renderer on the phone's view of design d in station 4's gimbal
    (sp a test.SwingPlan, rig its rig.spec design)."""

    def __init__(self, d, sp, rig, scale=1):
        self.sp, self.scale = sp, int(scale)
        self.W, self.H = Phone.W // self.scale, Phone.H // self.scale
        v = sp.view
        xml = mjcf.scene(d, width=self.W, height=self.H, floor=False)
        near = '<map znear="0.001" zfar="20"/>'
        if near not in xml:
            raise ValueError("mjcf.scene's near plane is no longer %s" % near)
        fovy = 2.0 * np.degrees(np.arctan(Phone.H / 2.0 / Phone.F_PX))
        cam = ('<camera name="phone" pos="%.6f %.6f %.6f" xyaxes="%.6f %.6f %.6f %.6f %.6f %.6f" fovy="%.6f"/>'
               % (*v.c, *v.R[:, 0], *v.R[:, 2], fovy))
        xml = xml.replace("</worldbody>", "    " + cam + "\n  </worldbody>")
        xml = xml.replace(near, '<map znear="%g" zfar="20"/>' % PICTURE_ZNEAR)
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.dd = mujoco.MjData(self.m)
        self.ren = mujoco.Renderer(self.m, self.H, self.W)
        self.adr = [self.m.jnt_qposadr[self.m.body_jntadr[b]] for b in range(1, self.m.nbody)
                    if self.m.body_jntnum[b] == 1]
        self.off = rig.bat_to_body(np.zeros(3)) / 1000.0        # m: the product frame's origin in the clamp's
        self.bands = [np.array([g for g in range(self.m.ngeom) if self.m.geom(g).name.startswith("band%d/" % (i + 1))])
                      for i in range(len(Rules.BAND_RGB))]
        if any(len(b) == 0 for b in self.bands):
            raise ValueError("mjcf.scene drew no geom for a band")

    def pose(self, q):
        """Every part where the gimbal's hinges q put the clamp."""
        Rc, x = self.sp.kin.pose(np.atleast_2d(q))
        pos = x[0] + Rc[0] @ self.off
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, Rc[0].ravel())
        for j in self.adr:
            self.dd.qpos[j:j + 3] = pos
            self.dd.qpos[j + 3:j + 7] = quat
        mujoco.mj_forward(self.m, self.dd)

    def rgb(self):
        self.ren.update_scene(self.dd, camera="phone")
        return self.ren.render()

    def silhouettes(self):
        """(2, 2) px, the phone's full-size pixels: each band's visible pixels'
        centroid (NaN where none shows), and how many pixels each covers."""
        self.ren.enable_segmentation_rendering()
        try:
            self.ren.update_scene(self.dd, camera="phone")
            seg = self.ren.render()
        finally:
            self.ren.disable_segmentation_rendering()
        geom = seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        out, n = np.full((2, 2), np.nan), np.zeros(2, int)
        for i, ids in enumerate(self.bands):
            v, u = np.nonzero(geom & np.isin(seg[:, :, 0], ids))
            n[i] = len(u)
            if n[i]:
                out[i] = (np.array([u.mean(), v.mean()]) + 0.5) * self.scale
        return out, n

    def close(self):
        self.ren.close()
