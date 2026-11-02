"""
Grasp check from the wrist camera (v6 step 4). Host-side (OpenCV + numpy).

The policy takes one wrist image at the hover pose before descending and one at the same pose after lifting. At the
hover pose the pick spot is well inside the wrist image (~(1022, 867) at 1920x1080) and the fingertips are below
its bottom edge, so the second image shows the spot, not the held object.

  object pixels   saturated colour (cube, AAA label) or dark (screw, AAA body); the table is white
  blob            in the first image, the object-pixel component at (or nearest to) the pick spot's projection
  still           in the second image, the share of that blob (grown by GROW_PX: the arm stops within 3 mm, ~10 px
                  at this range) that is still object-coloured
  verdict         still >= MISSED_AT: missed (the object never left); still <= PICKED_AT: picked; else unclear.
                  No usable blob in the first image (out of frame, too small): unclear, and the table image judges
                  the pick at the next capture, as in v5.
"""

import cv2
import numpy as np

SIZE_M = {"cube": 0.010, "screw": 0.030, "aaa_cell": 0.045}
MISSED_AT = 0.5
PICKED_AT = 0.25
GROW_PX = 6
MIN_BLOB_PX = 25


def object_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    return ((s >= 70) & (v >= 40)) | (v <= 90)


def window(cam, q, p_base, cls):
    """(u, v, half) of the pick spot's window in the wrist image, or None if it is not in the image."""
    uv = cam.project(q, p_base)
    if uv is None:
        return None
    edge = cam.project(q, np.asarray(p_base, float) + [SIZE_M.get(cls, 0.02), 0.0, 0.0])
    size_px = 30.0 if edge is None else float(np.linalg.norm(edge - uv))
    half = int(max(20, 0.9 * size_px))
    u, v = int(round(uv[0])), int(round(uv[1]))
    if not (half <= u < cam.W - half and half <= v < cam.H - half):
        return None
    return u, v, half


def blob(img_bgr, win):
    """The object-pixel component at (or nearest to) the window centre, as a full-window bool mask."""
    u, v, half = win
    m = object_mask(img_bgr[v - half:v + half, u - half:u + half]).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    best = min(range(1, n), key=lambda k: np.hypot(cents[k][0] - half, cents[k][1] - half) - 0.02 * stats[k, cv2.CC_STAT_AREA])
    out = lab == best
    return out if out.sum() >= MIN_BLOB_PX else None


def verdict(before_bgr, after_bgr, win):
    """("picked" | "missed" | "unclear", details)."""
    if before_bgr is None or after_bgr is None or win is None:
        return "unclear", {"why": "no wrist image or the spot is outside it"}
    b = blob(before_bgr, win)
    if b is None:
        return "unclear", {"why": "no object at the spot in the first wrist image"}
    u, v, half = win
    grown = cv2.dilate(b.astype(np.uint8), np.ones((2 * GROW_PX + 1, 2 * GROW_PX + 1), np.uint8)).astype(bool)
    after = object_mask(after_bgr[v - half:v + half, u - half:u + half])
    still = float((after & grown).sum()) / float(b.sum())
    still = min(still, 1.0)
    out = "missed" if still >= MISSED_AT else "picked" if still <= PICKED_AT else "unclear"
    return out, {"blob_px": int(b.sum()), "still": round(still, 2), "window": [u, v, half]}


def crop(img_bgr, win, pad=10):
    if img_bgr is None or win is None:
        return None
    u, v, half = win
    h = half + pad
    return img_bgr[max(0, v - h):v + h, max(0, u - h):u + h]
