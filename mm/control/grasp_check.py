"""
Did the gripper take the object? (Machine Minds step B6.) Host-side, numpy.

Uses the wrist camera as mounted (scene_config.WRIST_CAM, v0.5 calibration).
That camera sees the table under and ahead of the gripper, not the fingers,
so the guide's "pixel region between the fingers" is not visible. What it does
see, from the identical hover pose before descending and after lifting, is the
spot the object was lying on:

    pick spot still shows the object  -> MISS
    pick spot now shows bare table and has changed -> TAKEN

The spot is projected from the planned object position, the arm's joint angles
(forward kinematics) and the camera's calibrated mount -- no simulator labels,
so the same check runs on the real 2D wrist camera. An object knocked away
rather than taken also reads TAKEN here; the re-capture after every pick (B7)
then finds it on the table and replans it.
"""

import cv2
import numpy as np

import fk
import scene_config as C

# Footprint radius per class around the planned centre, metres.
SPOT_RADIUS = {
    "cube": C.CUBE_EDGE * 0.75,
    "screw": (C.SCREW_LENGTH + C.SCREW_HEAD_H) / 2,
    "aaa_cell": C.AAA_CELL_LENGTH / 2,
}
# Thresholds on the spot pixels (fractions), see verdict().
TABLE_DIFF = 40  # colour distance from the table counted as "not table"
CHANGE_DIFF = 40  # pre/post distance counted as "changed"
MAX_OBJECT_FRAC = 0.20  # TAKEN needs at most this much non-table left in the spot
MIN_CHANGE_FRAC = 0.15  # ... and at least this much of the spot changed


def wrist_camera():
    """CameraModel of the wrist camera in the J6 (flange) frame."""
    from test_support import camera_model

    return camera_model(dict(C.WRIST_CAM))


def spot_mask(cam_j6, q, xy, cls, shape, table_z=None):
    """Pixels of the object's footprint on the table, seen from joint pose q.

    xy is the planned object centre (robot base frame). The wrist camera is
    fixed to J6, so a world point maps to the camera through the flange pose.
    """
    rot, pos = fk.fk(q)
    z = (C.SURFACE_DZ if table_z is None else table_z) + 0.002
    r = SPOT_RADIUS[cls]
    ring = [(xy[0] + r * np.cos(a), xy[1] + r * np.sin(a), z) for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
    uv = []
    for p in ring:
        p_j6 = rot.T @ (np.asarray(p) - pos)
        pix = cam_j6.project(p_j6)
        if pix is not None:
            uv.append(pix)
    mask = np.zeros(shape[:2], np.uint8)
    if len(uv) >= 3:
        hull = cv2.convexHull(np.array(uv, np.float32)).astype(np.int32)
        cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool)


def _table_colour(img, mask):
    """Median colour of a ring around the spot: the table near it."""
    big = cv2.dilate(mask.astype(np.uint8), np.ones((61, 61), np.uint8)).astype(bool)
    ring = big & ~cv2.dilate(mask.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
    px = img[ring] if ring.any() else img.reshape(-1, 3)
    return np.median(px, axis=0)


def verdict(pre, post, mask):
    """TAKEN / MISS for one grasp, with the evidence.

    pre, post: BGR wrist images at the same hover pose (before descend, after
    lift). mask: spot_mask() for that pose.
    """
    if mask.sum() < 30:
        return {"taken": None, "reason": "spot_not_in_view", "spot_px": int(mask.sum())}
    pre_f, post_f = pre.astype(np.float32), post.astype(np.float32)
    table = _table_colour(post_f, mask)
    not_table = np.linalg.norm(post_f[mask] - table, axis=1) > TABLE_DIFF
    changed = np.linalg.norm(post_f[mask] - pre_f[mask], axis=1) > CHANGE_DIFF
    obj_frac, change_frac = float(not_table.mean()), float(changed.mean())
    taken = obj_frac <= MAX_OBJECT_FRAC and change_frac >= MIN_CHANGE_FRAC
    return {
        "taken": bool(taken),
        "reason": "spot_cleared" if taken else ("object_still_there" if obj_frac > MAX_OBJECT_FRAC else "no_change"),
        "object_frac": round(obj_frac, 3),
        "change_frac": round(change_frac, 3),
        "spot_px": int(mask.sum()),
    }
