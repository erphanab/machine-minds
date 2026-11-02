"""
Tabletop pose refinement from a YOLO instance mask (Machine Minds A7).

No model or Isaac imports: runs and is tested on the host. YOLO says what and
roughly where; this measures where, to the ~1 mm a 12 mm jaw opening needs.

One method for every target class: fit the projected silhouette of the known
object shape, resting upright on the calibrated table, to the instance mask.
  cube     x, y, yaw (mod 90), then a joint outline + top-face polish for yaw
  cap      x, y, for both orientations; the better one is reported
  aaa_cell  x, y, yaw (full turn: + terminal direction)
  screw    x, y, yaw (full turn: head -> tip), lying with its rest tilt; xy is
           the shank middle, where the jaws close

The guide seeds the starter's cube_pose.refine_pose instead. That segments by
hue alone, and with randomised light the warm-lit table passes its yellow
gate around yellow cubes (1306 px component for a 706 px cube), so it failed
about a third of cubes. Same silhouette idea, observed mask from YOLO instead.

The predicted silhouette is supersampled, so IoU varies smoothly with sub-pixel
moves and small yaw changes on a 26 px cube. Every fit is refused at a search
boundary, as in the starter, so a poor seed cannot become an unchecked grasp.
"""

import cv2
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.transform import Rotation

import scene_config as C
from mm.datagen.masks_to_yolo import object_parts

FIT_IOU_MIN = 0.85  # same gate as the starter's image fit
# A lying AAA cell's mask is ~4:1. One standing on end or propped on a
# neighbour is near-round, yet a lying-cell fit could still clear the IoU gate
# (up to 16 mm off in eval), so such masks are refused as not resting flat.
# A lying screw's outline is ~2.5:1 (30 mm by its 12 mm head).
MIN_ELONGATION = {"aaa_cell": 2.0, "screw": 1.8}
AA_MIN_ELONGATION = MIN_ELONGATION["aaa_cell"]
_R_SCREW = Rotation.from_euler("y", -90 + C.SCREW_REST_TILT_DEG, degrees=True)
XY_SEARCH = 0.006  # +/- around the mask-centroid seed, metres
SUPERSAMPLE = 4
CUBE_HUES = {  # OpenCV hue gates from cube_pose.refine_pose; colour is a report
    "yellow": lambda h: (h >= 12) & (h <= 42),
    "blue": lambda h: (h >= 95) & (h <= 135),
    "red": lambda h: (h <= 12) | (h >= 170),
}
# Height of the point whose image the mask centroid approximates.
MID_HEIGHT = {
    "cube": C.CUBE_EDGE / 2,
    "cap": C.CAP_HEIGHT / 2,
    "aaa_cell": C.AAA_CELL_RADIUS,
    "tube": C.VIAL_RADIUS,
    "screw": (C.SCREW_HEAD_D + C.SCREW_SHANK_D) / 4,  # axis height mid-length
    "bin": 0.065 / 2,  # bins.yaml design wall_h / 2
}
# A tall bin's mask centroid sits well off its base centre at this camera angle.
BIN_XY_SEARCH = 0.025


def rest_pose(cls, xy, yaw_deg, table_z):
    """Root position and rotation of an upright object resting on the table."""
    yaw = Rotation.from_euler("z", yaw_deg, degrees=True)
    if cls == "aaa_cell":
        return np.array([*xy, table_z + C.AAA_CELL_RADIUS]), yaw * Rotation.from_euler(
            "y", 90, degrees=True
        )
    if cls in ("cube", "cap", "bin"):
        return np.array([*xy, table_z]), yaw
    if cls == "screw":  # root at the head face; xy is the shank middle
        hx, hy = C.screw_head_xy(xy, yaw_deg)
        return np.array([hx, hy, table_z + C.SCREW_REST_HEAD_Z]), yaw * _R_SCREW
    raise ValueError(cls)


def silhouette(cls, cam, xy, yaw_deg, shape, origin=(0, 0), orientation=None, ss=1):
    """Projected object in an image window at `origin`.

    ss=1 gives a binary mask; ss>1 gives per-pixel coverage in [0, 1] from an
    ss x ss supersampled fill.
    """
    pos, rot = rest_pose(cls, xy, yaw_deg, cam.table_z)
    big = np.zeros((shape[0] * ss, shape[1] * ss), np.uint8)
    for part in object_parts(cls, orientation):  # union of convex pieces
        uv = np.array([cam.project(p) for p in part @ rot.as_matrix().T + pos], np.float32)
        hull = cv2.convexHull(uv).reshape(-1, 2) - np.asarray(origin, np.float32)
        # Continuous coords: sub-pixel centres sit at (index + 0.5) / ss.
        pts16 = np.round((hull * ss - 0.5) * 16).astype(np.int32)
        cv2.fillConvexPoly(big, pts16, 1, lineType=cv2.LINE_8, shift=4)
    if ss == 1:
        return big
    return big.reshape(shape[0], ss, shape[1], ss).mean(axis=(1, 3), dtype=np.float32)


def mask_centroid(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean() + 0.5, ys.mean() + 0.5])


def seed_xy(cam, mask, cls):
    u, v = mask_centroid(mask)
    p = cam.to_plane(u, v, z=cam.table_z + MID_HEIGHT[cls])
    return None if p is None else p[:2]


def cube_colour(img_bgr, mask):
    """Colour whose hue gate holds most of the masked, saturated pixels."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[mask.astype(bool)]
    sat = hsv[(hsv[:, 1] >= 45) & (hsv[:, 2] >= 35)]
    if not len(sat):
        return None
    counts = {c: int(gate(sat[:, 0]).sum()) for c, gate in CUBE_HUES.items()}
    return max(counts, key=counts.get)


def _window(mask, pad):
    ys, xs = np.nonzero(mask)
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x1 = min(mask.shape[1], xs.max() + pad + 1)
    y1 = min(mask.shape[0], ys.max() + pad + 1)
    return x0, y0, x1, y1


def fit_silhouette(cls, cam, mask, xy0, yaw_range=None, orientation=None, seed=8, xy_search=XY_SEARCH):
    """Fit (x, y[, yaw]) so the projected silhouette matches `mask` (full image).

    Returns (xy, yaw_deg, iou). IoU is zeroed at a search boundary.
    """
    x0, y0, x1, y1 = _window(mask, pad=40)
    observed = mask[y0:y1, x0:x1].astype(np.float32)
    shape = observed.shape

    def iou(p, ss):  # soft IoU: fractional predicted coverage vs the binary mask
        yaw = p[2] if yaw_range else 0.0
        pred = silhouette(cls, cam, p[:2], yaw, shape, (x0, y0), orientation, ss)
        return np.minimum(pred, observed).sum() / max(1.0, np.maximum(pred, observed).sum())

    bounds = [(xy0[0] - xy_search, xy0[0] + xy_search), (xy0[1] - xy_search, xy0[1] + xy_search)]
    if yaw_range:
        bounds.append(yaw_range)
    # Global search on the cheap binary silhouette, then a supersampled polish.
    coarse = differential_evolution(
        lambda p: -iou(p, 1), bounds, seed=seed, popsize=12, maxiter=40, polish=False, tol=1e-3
    )
    scale = np.array([0.0005, 0.0005, 1.0][: len(bounds)])  # ~equal IoU sensitivity
    fine = minimize(
        lambda z: -iou(coarse.x + z * scale, SUPERSAMPLE),
        np.zeros(len(bounds)),
        method="Powell",
        bounds=[(-4, 4)] * len(bounds),
        options={"xtol": 1e-2, "ftol": 1e-5},
    )
    p = coarse.x + fine.x * scale
    fit = float(iou(p, SUPERSAMPLE))
    if np.any(np.abs(p[:2] - xy0) > xy_search - 0.0002):
        fit = 0.0  # boundary optimum: the true pose may be outside the search
    yaw = float(p[2]) if yaw_range else 0.0
    return p[:2], yaw, fit


def _soft_fill(poly, shape, ss=SUPERSAMPLE):
    big = np.zeros((shape[0] * ss, shape[1] * ss), np.uint8)
    pts16 = np.round((poly * ss - 0.5) * 16).astype(np.int32)
    cv2.fillConvexPoly(big, pts16, 1, lineType=cv2.LINE_8, shift=4)
    return big.reshape(shape[0], ss, shape[1], ss).mean(axis=(1, 3), dtype=np.float32)


def cube_top_polish(img_bgr, cam, mask, xy, yaw_deg):
    """Tighten cube yaw with the top face, which the outline barely constrains.

    A 26 px outline leaves ~2.5 deg yaw noise (p95 4.6 deg); the edge between
    the top face and the sides carries most of the yaw. The top face is the
    brighter Otsu class inside the mask. Jointly fitting outline and top face
    gave p95 1.6 deg on rendered frames. Returns (xy, yaw, outline IoU), or
    None when the bright region does not look like a top face.

    YOLO's mask edge spills onto the white table, which is brighter than the
    top face, so table pixels (unsaturated, unlike the coloured cube) are left
    out of the brightness split. The outline fit still uses the YOLO mask:
    a saturation cut there loses shaded side faces under dim or tinted light.
    """
    x0, y0, x1, y1 = _window(mask, pad=20)
    win = mask[y0:y1, x0:x1].astype(bool)
    hsv = cv2.cvtColor(img_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    s_thr, _ = cv2.threshold(s[win].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    body = win & (s > min(s_thr, 60))  # coloured pixels; the table is ~S 50 at most
    if body.sum() < 0.5 * win.sum():
        body = win
    thr, _ = cv2.threshold(v[body].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    top = (body & (v > thr)).astype(np.float32)
    if not 0.15 <= top.sum() / max(1, win.sum()) <= 0.85:
        return None
    outline = win.astype(np.float32)
    origin = np.array([x0, y0], np.float32)

    def ious(p):
        t = np.radians(p[2])
        rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        tp = np.array(
            [
                cam.project([*(p[:2] + rot @ np.array(c) * C.CUBE_EDGE / 2), cam.table_z + C.CUBE_EDGE])
                for c in corners
            ],
            np.float32,
        )
        pt = _soft_fill(tp - origin, win.shape)
        ph = silhouette("cube", cam, p[:2], p[2], win.shape, (x0, y0), ss=SUPERSAMPLE)
        i_out = np.minimum(ph, outline).sum() / max(1.0, np.maximum(ph, outline).sum())
        i_top = np.minimum(pt, top).sum() / max(1.0, np.maximum(pt, top).sum())
        return i_out, i_top

    scale = np.array([0.0005, 0.0005, 1.0])
    base = np.array([*xy, yaw_deg])
    runs = [
        minimize(
            lambda z: -sum(ious(base + z * scale)),
            [0.0, 0.0, s],
            method="Powell",
            bounds=[(-4, 4), (-4, 4), (-8, 8)],
        )
        for s in (-4.0, 0.0, 4.0)
    ]
    p = base + min(runs, key=lambda r: r.fun).x * scale
    return p[:2], float((p[2] + 45) % 90 - 45), float(ious(p)[0])


def aa_polarity(img_bgr, cam, mask, xy, yaw_deg):
    """+1 if the coloured terminal band is at the yaw_deg end, else -1.

    The mask outline cannot tell the ends apart (the nub is ~1 mm), and a
    backwards fit is ~1 mm off along the axis. The band is the most saturated
    part of the cell, so compare saturation in the two end quarters.
    """
    a = np.radians(yaw_deg)
    z = cam.table_z + C.AAA_CELL_RADIUS
    ends = [np.asarray(xy) + s * C.AAA_CELL_LENGTH / 2 * np.array([np.cos(a), np.sin(a)]) for s in (1, -1)]
    pu, nu = (cam.project([*e, z]) for e in ends)
    ys, xs = np.nonzero(mask)
    pix = np.stack([xs + 0.5, ys + 0.5], 1)
    axis = pu - nu
    t = (pix - nu) @ axis / max(1e-9, axis @ axis)  # 0 at - end, 1 at + end
    sat = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[ys, xs, 1].astype(float)
    hi, lo = sat[t > 0.75], sat[t < 0.25]
    if not len(hi) or not len(lo):
        return 1
    return 1 if hi.mean() >= lo.mean() else -1


def _elongation(mask):
    ys, xs = np.nonzero(mask)
    (_, _), (w, h), _ = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
    return max(w, h) / max(1e-6, min(w, h))


def refine(cls, img_bgr, cam, mask):
    """Pose for one detection. Returns a dict with xy, yaw_deg, fit_iou and extras."""
    xy0 = seed_xy(cam, mask, cls)
    if xy0 is None:
        return {"xy": None, "yaw_deg": None, "fit_iou": 0.0}
    if cls == "cube":
        xy, yaw, fit = fit_silhouette("cube", cam, mask, xy0, yaw_range=(-45.0, 45.0))
        if fit >= FIT_IOU_MIN:
            polished = cube_top_polish(img_bgr, cam, mask, xy, yaw)
            if polished is not None and polished[2] >= FIT_IOU_MIN:
                xy, yaw, fit = polished
        return {
            "xy": [float(v) for v in xy],
            "yaw_deg": yaw,
            "fit_iou": fit,
            "colour": cube_colour(img_bgr, mask),  # reported, not used to pick
        }
    if cls == "cap":
        best = None
        for orient in ("narrow_end_up", "wide_end_up"):
            xy, _, fit = fit_silhouette("cap", cam, mask, xy0, orientation=orient)
            if best is None or fit > best[2]:
                best = (xy, orient, fit)
        xy, orient, fit = best
        return {"xy": [float(v) for v in xy], "yaw_deg": None, "fit_iou": fit, "orientation": orient}
    if cls == "aaa_cell":
        # Full turn: the nub makes the cell end-to-end asymmetric, and a 180 deg
        # search fitted half of them backwards, ~1 mm off along the axis.
        # yaw_deg points along + (nub) end, so it also gives polarity.
        if _elongation(mask) < MIN_ELONGATION["aaa_cell"]:
            return {"xy": xy0.tolist(), "yaw_deg": None, "fit_iou": 0.0, "resting_flat": False}
        xy, yaw, fit = fit_silhouette("aaa_cell", cam, mask, xy0, yaw_range=(-180.0, 180.0))
        if aa_polarity(img_bgr, cam, mask, xy, yaw) < 0:
            flipped = (yaw + 180.0 + 180.0) % 360.0 - 180.0
            xy, yaw, fit = fit_silhouette(
                "aaa_cell", cam, mask, xy0, yaw_range=(flipped - 5.0, flipped + 5.0)
            )
        return {"xy": [float(v) for v in xy], "yaw_deg": yaw, "fit_iou": fit}
    if cls == "screw":
        # Full turn: the head makes the outline clearly one-ended, so the fit
        # itself resolves head -> tip. The mask centroid sits ~3 mm toward the
        # head from the shank middle, inside the search range.
        if _elongation(mask) < MIN_ELONGATION["screw"]:
            return {"xy": xy0.tolist(), "yaw_deg": None, "fit_iou": 0.0, "resting_flat": False}
        xy, yaw, fit = fit_silhouette("screw", cam, mask, xy0, yaw_range=(-180.0, 180.0))
        # The global search sometimes settles head-for-tip (IoU ~0.75, centre
        # ~5 mm off along the axis). Fit the flipped hypothesis too; the head
        # makes the right one clearly better.
        flipped = (yaw + 360.0) % 360.0 - 180.0
        alt = fit_silhouette("screw", cam, mask, xy0, yaw_range=(flipped - 10.0, flipped + 10.0))
        if alt[2] > fit:
            xy, yaw, fit = alt
        return {"xy": [float(v) for v in xy], "yaw_deg": yaw, "fit_iou": fit}
    if cls == "bin":
        # B8: where the bin is and how it is turned (square: yaw mod 90), and
        # which bin it is -- they share one shape, so by colour.
        xy, yaw, fit = fit_silhouette(
            "bin", cam, mask, xy0, yaw_range=(-45.0, 45.0), xy_search=BIN_XY_SEARCH
        )
        return {"xy": [float(v) for v in xy], "yaw_deg": yaw, "fit_iou": fit, "colour": cube_colour(img_bgr, mask)}
    # Distractors: centroid only, never refined or picked.
    return {"xy": xy0.tolist(), "yaw_deg": None, "fit_iou": 0.0}
