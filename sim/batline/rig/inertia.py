"""The gimbal's mass properties in numpy, from the very MJCF the scene is
built from (mjcf.gimbal_xml(..., weighed=True)).

The planner times the gimbal by its motors' torque against these, and it
runs inside line.py, which must not need MuJoCo.  So the shapes' masses
and inertias are computed here exactly as MuJoCo computes them from geoms
(inertiafromgeom: a box, a cylinder or a sphere of uniform density, each
moved to its body's frame by the parallel-axis theorem), and check_imu
holds the two to each other: MuJoCo's own body masses, centres and
inertias, and its joint-space mass matrix.

    inertials(xml)   {body: Inertial} SI, in each body's own frame, with
                     every welded child (no joint) folded into its parent:
                     what moves on each hinge
"""
import xml.etree.ElementTree as ET

import numpy as np

from ..imu.kin import Inertial


def _floats(s):
    return np.array([float(v) for v in s.split()])


def _frame_from_xyaxes(xy):
    """MuJoCo's reading of xyaxes: x normalised, y made square to it and
    normalised, z their cross product."""
    x = xy[:3] / np.linalg.norm(xy[:3])
    y = xy[3:] - x * (x @ xy[3:])
    y = y / np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def _quat_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _vec_to_R(v):
    """A frame whose z is along v, as MuJoCo orients a fromto geom (the
    rotation taking z to v by the shortest arc)."""
    z = v / np.linalg.norm(v)
    e = np.array([0.0, 0.0, 1.0])
    c = float(e @ z)
    if c > 1.0 - 1e-15:
        return np.eye(3)
    if c < -1.0 + 1e-15:
        return np.diag([1.0, -1.0, -1.0])
    ax = np.cross(e, z)
    s = np.linalg.norm(ax)
    ax = ax / s
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    ang = np.arctan2(s, c)
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def geom_inertial(g, density):
    """(m, c, I about c in the body frame) of one geom element."""
    typ = g.get("type", "sphere")
    size = _floats(g.get("size"))
    if typ == "box":
        a, b, c = size
        vol = 8.0 * a * b * c
        Il = np.diag([b * b + c * c, a * a + c * c, a * a + b * b]) / 3.0
    elif typ == "cylinder":
        r = size[0]
        if g.get("fromto") is not None:
            ft = _floats(g.get("fromto"))
            h = np.linalg.norm(ft[3:] - ft[:3]) / 2.0
        else:
            h = size[1]
        vol = np.pi * r * r * 2.0 * h
        Il = np.diag([r * r / 4.0 + h * h / 3.0, r * r / 4.0 + h * h / 3.0, r * r / 2.0])
    elif typ == "sphere":
        r = size[0]
        vol = 4.0 / 3.0 * np.pi * r ** 3
        Il = np.eye(3) * 0.4 * r * r
    else:
        raise ValueError("geom type %s has no inertia here" % typ)
    if g.get("mass") is not None:
        m = float(g.get("mass"))
    elif g.get("density") is not None:
        m = float(g.get("density")) * vol
    elif density is not None:
        m = density * vol
    else:
        raise ValueError("geom %s is not weighed" % g.get("name"))
    if g.get("fromto") is not None:
        ft = _floats(g.get("fromto"))
        c = (ft[:3] + ft[3:]) / 2.0
        R = _vec_to_R(ft[3:] - ft[:3])
    else:
        c = _floats(g.get("pos")) if g.get("pos") is not None else np.zeros(3)
        if g.get("xyaxes") is not None:
            R = _frame_from_xyaxes(_floats(g.get("xyaxes")))
        elif g.get("quat") is not None:
            R = _quat_to_R(_floats(g.get("quat")))
        else:
            R = np.eye(3)
    return m, c, m * (R @ Il @ R.T)


def combine(parts):
    """Inertial of several (m, c, I_about_c) in one frame."""
    m = sum(p[0] for p in parts)
    if m <= 0.0:
        return Inertial(0.0, np.zeros(3), np.zeros((3, 3)))
    c = sum(p[0] * p[1] for p in parts) / m
    I = np.zeros((3, 3))
    for mi, ci, Ii in parts:
        d = ci - c
        I += Ii + mi * ((d @ d) * np.eye(3) - np.outer(d, d))
    return Inertial(m, c, I)


def _body_parts(el, density, R=np.eye(3), p=np.zeros(3)):
    """Every geom of body element el and of its welded descendants, in el's
    frame (R, p: a descendant's frame in el's)."""
    out = []
    for g in el.findall("geom"):
        m, c, I = geom_inertial(g, density)
        out.append((m, p + R @ c, R @ I @ R.T))
    for ch in el.findall("body"):
        if ch.find("joint") is not None:
            continue
        pc = _floats(ch.get("pos")) if ch.get("pos") is not None else np.zeros(3)
        Rc = _quat_to_R(_floats(ch.get("quat"))) if ch.get("quat") is not None else np.eye(3)
        out += _body_parts(ch, density, R @ Rc, p + R @ pc)
    return out


def inertials(xml, density=None):
    """{body name: Inertial}, SI, for every body with a joint in an MJCF
    fragment of bodies.  A geom weighs its `mass`, else its `density` times
    its volume, else `density` here times its volume; with none of them
    it is an error, so an unweighed geom cannot pass for a weighed one."""
    root = ET.fromstring("<root>%s</root>" % xml)
    out = {}
    for b in root.iter("body"):
        if b.find("joint") is None:
            continue
        out[b.get("name")] = combine(_body_parts(b, density))
    return out
