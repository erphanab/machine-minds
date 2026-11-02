"""
Grasp pose for cubes and screws straight from the YOLO outline (v6). No 3D model is fitted and there is no
match-score gate: the 28 Oct 2026 real run refused 691 of 694 cube and all 163 screw detections at pose_fit's
0.85 IoU gate, because a 10 mm cube is ~20 px in the 1080p table image.

Every mask pixel is back-projected onto the table at the object's mid height (pose_fit.MID_HEIGHT), so the
rectangle and the profile below are in metres on the table, not skewed by the oblique camera.
  cube   centre: the mask centroid. Angle: the minimum-area rectangle's edges, mod 90 (the jaws close on a face pair).
  screw  any type (hex bolt, countersunk, ...): the rectangle's long axis is the screw axis. The head is the end
         whose width is clearly larger than the shank's; the grasp centre is the middle of the shank only, so the
         jaws never close on the head. yaw_deg points head -> tip (pose_fit's convention). No clear head (both ends
         alike): the middle of the whole length, and head_end "unclear" in the record.
Diagnostics (pixel centre, rectangle, widths) go into the record for previews and logs.
"""

import math

import cv2
import numpy as np

from mm.perception.pose_fit import MID_HEIGHT, mask_centroid

HEAD_RATIO = 1.3  # a slice this much wider than the shank's median width belongs to the head
SLICE_M = 0.001  # profile slice along the screw axis
MIN_SHANK_M = 0.004  # shorter shank than this: fall back to the whole length's middle


def table_points(cam, mask, z):
    """Mask pixels (centres) -> table-plane xy (N, 2) at height z, and their pixel coords."""
    ys, xs = np.nonzero(mask)
    pts, pix = [], []
    for u, v in zip(xs + 0.5, ys + 0.5):
        p = cam.to_plane(u, v, z=z)
        if p is not None:
            pts.append(p[:2])
            pix.append((u, v))
    return np.asarray(pts, float), np.asarray(pix, float)


def min_area_rect(pts):
    """((cx, cy), (w, h), angle_deg) of table points, in metres; angle of the w edge from +x."""
    scale = 1e4  # minAreaRect works in float32: 0.1 mm units keep the precision
    (cx, cy), (w, h), a = cv2.minAreaRect((pts * scale).astype(np.float32))
    return (cx / scale, cy / scale), (w / scale, h / scale), a


def _pixel(cam, xy, z):
    uv = cam.project([xy[0], xy[1], z])
    return None if uv is None else [float(uv[0]), float(uv[1])]


def cube_grasp(cam, mask):
    z = cam.table_z + MID_HEIGHT["cube"]
    u, v = mask_centroid(mask)
    c = cam.to_plane(u, v, z=z)
    if c is None:
        return {"xy": None, "yaw_deg": None}
    pts, _ = table_points(cam, mask, z)
    _, (w, h), a = min_area_rect(pts)
    yaw = (a + 45.0) % 90.0 - 45.0  # square: any edge, mod 90
    return {"xy": [float(c[0]), float(c[1])], "yaw_deg": float(yaw), "centre_px": [float(u), float(v)],
            "rect_mm": [round(w * 1000, 1), round(h * 1000, 1)]}


def screw_grasp(cam, mask):
    z = cam.table_z + MID_HEIGHT["screw"]
    pts, _ = table_points(cam, mask, z)
    if len(pts) < 5:
        return {"xy": None, "yaw_deg": None}
    (cx, cy), (w, h), a = min_area_rect(pts)
    if w < h:
        a, (w, h) = a + 90.0, (h, w)
    axis = np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
    side = np.array([-axis[1], axis[0]])
    rel = pts - [cx, cy]
    t, s = rel @ axis, rel @ side
    edges = np.arange(t.min(), t.max() + SLICE_M, SLICE_M)
    width, mid = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (t >= lo) & (t < hi)
        width.append(float(s[k].max() - s[k].min()) if k.sum() >= 2 else 0.0)
        mid.append(float((s[k].max() + s[k].min()) / 2) if k.any() else 0.0)
    width, mid = np.asarray(width), np.asarray(mid)
    n = len(width)
    # The head is the wider end: compare the outer thirds' widest slices (the very last slice is narrow where
    # the outline rounds off, so a single end slice says little). Then grow the head inward from its widest
    # slice while slices stay clearly wider than the shank (median of the tip-side half).
    third = max(1, n // 3)
    lo_peak, hi_peak = width[:third].max(), width[n - third:].max()
    if max(lo_peak, hi_peak) < HEAD_RATIO * min(lo_peak, hi_peak) or n < 4:
        head_end, start, stop = "unclear", 0, n
        shank_w = float(np.median(width[width > 0]))
    else:
        head_lo = lo_peak > hi_peak
        prof = width if head_lo else width[::-1]  # head at index 0 of prof
        tail = prof[n // 2: n - 1] if n - 1 > n // 2 else prof[n // 2:]
        shank_w = float(np.median(tail[tail > 0])) if (tail > 0).any() else float(prof.min())
        k = int(np.argmax(prof[:third]))
        while k + 1 < n and prof[k + 1] > HEAD_RATIO * shank_w:
            k += 1
        head_len = k + 1  # slices 0..k of prof are head
        head_end = "-axis" if head_lo else "+axis"
        start, stop = (head_len, n) if head_lo else (0, n - head_len)
    if (stop - start) * SLICE_M < MIN_SHANK_M:
        head_end, start, stop = "unclear", 0, n
    t_mid = (edges[start] + edges[stop]) / 2
    s_mid = float(np.median(mid[start:stop]))
    centre = np.array([cx, cy]) + t_mid * axis + s_mid * side
    yaw = a if head_end != "+axis" else a + 180.0  # head -> tip
    yaw = (yaw + 180.0) % 360.0 - 180.0
    idx = np.arange(n)
    head_w = width[(idx < start) | (idx >= stop)]
    return {"xy": [float(centre[0]), float(centre[1])], "yaw_deg": float(yaw), "centre_px": _pixel(cam, centre, z),
            "head_end": head_end, "length_mm": round(w * 1000, 1), "shank_mm": round(shank_w * 1000, 1),
            "head_mm": round(float(head_w.max()) * 1000, 1) if len(head_w) else None,
            "grip_from_mm": round(float(edges[start] - t.min()) * 1000, 1),
            "grip_to_mm": round(float(edges[stop] - t.min()) * 1000, 1)}


def grasp(cls, cam, mask):
    return cube_grasp(cam, mask) if cls == "cube" else screw_grasp(cam, mask)
