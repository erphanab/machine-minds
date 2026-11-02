"""Ground-truth geometry from datagen meta.json -- for scoring only, never runtime."""

import math

import numpy as np
from scipy.spatial.transform import Rotation

import scene_config as C

# Object centre in the root prim's frame (cell.py's origins differ per class).
_LOCAL_CENTRE = {
    "cube": (0.0, 0.0, C.CUBE_EDGE / 2),
    "cap": (0.0, 0.0, C.CAP_HEIGHT / 2),
    "aaa_cell": (0.0, 0.0, 0.0),
    "tube": (0.0, 0.0, C.VIAL_LENGTH / 2),
    "screw": (0.0, 0.0, -(C.SCREW_HEAD_H + C.SCREW_LENGTH) / 2),  # shank middle
    "bin": (0.0, 0.0, 0.0),  # base centre
}
YAW_PERIOD = {"cube": 90.0, "aaa_cell": 180.0, "tube": 180.0, "screw": 180.0, "bin": 90.0}


def _rot(obj):
    w, x, y, z = obj["quat_wxyz"]
    return Rotation.from_quat([x, y, z, w])


def gt_centre_xy(obj):
    c = _rot(obj).apply(_LOCAL_CENTRE[obj["class"]]) + obj["position"]
    return [float(c[0]), float(c[1])]


def gt_yaw_deg(obj):
    """Cube: face direction about Z. AAA cell / tube: long axis (local Z) in XY.
    Screw: head -> tip direction in XY."""
    r = _rot(obj)
    if obj["class"] == "bin":
        a = r.apply([1.0, 0.0, 0.0])
    elif obj["class"] == "cube":
        # A dropped cube may rest on any face: use its most horizontal axis.
        axes = r.apply(np.eye(3))
        a = axes[int(np.argmin(np.abs(axes[:, 2])))]
    elif obj["class"] in ("aaa_cell", "tube"):
        a = r.apply([0.0, 0.0, 1.0])
    elif obj["class"] == "screw":
        a = r.apply([0.0, 0.0, -1.0])  # head -> tip
    else:
        return None
    return math.degrees(math.atan2(a[1], a[0]))


def yaw_error(cls, yaw, truth):
    """Absolute yaw error in degrees, modulo the object's symmetry."""
    if yaw is None or truth is None or cls not in YAW_PERIOD:
        return None
    p = YAW_PERIOD[cls]
    d = (yaw - truth) % p
    return min(d, p - d)
