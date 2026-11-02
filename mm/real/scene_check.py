"""
Is the table ready (v6 step 5)? Host-side (OpenCV + numpy).

  stable(prev, now)  the same targets and bins in two captures in a row: per class the same count, and every
                     detection within STABLE_M of one of the same class in the other capture
  foreign(...)       anything large on the table that is none of the detections: a hand or an arm. Object pixels
                     (saturated, skin-toned or dark) inside the work ROI, minus every detection's outline (grown by
                     PAD_PX) and minus the robot (its base column and the arm at its current joints), as
                     components; the largest above FOREIGN_PX is reported.
The 28 Oct 2026 run had a hand on the table in rollout_05 act 22 and an arm in rollout_07 act 2.
"""

import math

import cv2
import numpy as np

STABLE_M = 0.010
# A 10 mm cube is ~400 px, an AAA cell ~1000 px in the 1080p table image. 28 Oct 2026 captures: hands and arms
# 9100-50700 px (22 captures), everything else <= 3700 px (the far table edge, the frame at the right).
FOREIGN_PX = 5000
PAD_PX = 12
DARK = 0.7  # darker than this share of the local table brightness
COUNTED = ("cube", "screw", "aaa_cell", "cap", "tube", "bin")


def signature(dets, conf_min=0.5):
    return [(d["class"], tuple(d["xy_robot_m"])) for d in dets
            if d["class"] in COUNTED and d.get("conf", 0) >= conf_min and d.get("xy_robot_m") is not None]


def stable(prev, now):
    if prev is None:
        return False
    for a, b in ((prev, now), (now, prev)):
        if sorted(c for c, _ in a) != sorted(c for c, _ in b):
            return False
        for c, xy in a:
            if not any(c2 == c and math.dist(xy, xy2) <= STABLE_M for c2, xy2 in b):
                return False
    return True


def arm_mask(cam, shape, arm_points, pad_px=0):
    """The arm (points along its links, base frame) as a filled outline in the table image, grown by pad_px."""
    m = np.zeros(shape[:2], np.uint8)
    uv = np.array([p for p in (cam.project(np.asarray(q) + [0.0, 0.0, cam.dz]) for q in arm_points) if p is not None],
                  np.float32)
    if len(uv) >= 3:
        cv2.fillConvexPoly(m, cv2.convexHull(uv).astype(np.int32), 1)
        if pad_px:
            m = cv2.dilate(m, np.ones((2 * pad_px + 1, 2 * pad_px + 1), np.uint8))
    return m.astype(bool)


def robot_mask(cam, shape, arm_points=None):
    """The robot's base column (radius 0.13 m, up to the shoulder) and, if given, the arm's points, as a mask."""
    m = np.zeros(shape[:2], np.uint8)
    col = [cam.project([0.13 * math.cos(a), 0.13 * math.sin(a), cam.dz + z])
           for a in np.linspace(0, 2 * math.pi, 24, endpoint=False) for z in (0.0, 0.40)]
    col = np.array([p for p in col if p is not None], np.float32)
    if len(col) >= 3:
        cv2.fillConvexPoly(m, cv2.convexHull(col).astype(np.int32), 1)
    if arm_points is not None:
        uv = np.array([p for p in (cam.project(np.asarray(q) + [0.0, 0.0, cam.dz]) for q in arm_points) if p is not None],
                      np.float32)
        if len(uv) >= 3:
            cv2.fillConvexPoly(m, cv2.convexHull(uv).astype(np.int32), 1)
            m = cv2.dilate(m, np.ones((41, 41), np.uint8))
    return m.astype(bool)


def foreign(img_bgr, dets, roi_box, cam=None, arm_points=None):
    """Largest unexplained object-pixel area: {"px", "box"} or None."""
    x0, y0, w, h = roi_box
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    # The white table's brightness varies across the image (V ~115 in the far shaded corner, ~165 in the middle):
    # dark is relative to the local table, from a median over a window much larger than any target.
    small = cv2.resize(v, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.medianBlur(small, 15), (v.shape[1], v.shape[0]), interpolation=cv2.INTER_LINEAR)
    obj = ((s >= 60) & (v >= 40)) | (v.astype(np.float32) < DARK * bg.astype(np.float32))
    # The yellow caution tape at the table's edge enters the ROI's corner; yellow targets and bins are detections
    # (excluded below anyway). Skin and sleeves are red-orange (H < 20 or > 160), so they stay.
    hue = hsv[:, :, 0]
    obj &= ~((hue >= 22) & (hue <= 38) & (s >= 60))
    known = np.zeros(obj.shape, np.uint8)
    for d in dets:
        pts = np.asarray(d["polygon_px"], np.float32).reshape(-1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(known, [pts.astype(np.int32)], 1)
    known = cv2.dilate(known, np.ones((2 * PAD_PX + 1, 2 * PAD_PX + 1), np.uint8)).astype(bool)
    if cam is not None:
        known |= robot_mask(cam, obj.shape, arm_points)
    left = np.zeros(obj.shape, np.uint8)
    left[y0:y0 + h, x0:x0 + w] = (obj & ~known)[y0:y0 + h, x0:x0 + w]
    left = cv2.morphologyEx(left, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(left, 8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[k, cv2.CC_STAT_AREA])
    if area < FOREIGN_PX:
        return None
    x, y, bw, bh = (int(stats[k, i]) for i in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    return {"px": area, "box": [x, y, bw, bh]}
