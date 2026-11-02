"""
Instance masks -> YOLO-seg labels on a fixed workspace crop (Machine Minds A4).

Host-side. Reads gen_yolo_data.py output and writes an Ultralytics dataset:
  <dst>/images/<split>/<name>.jpg   ROI crop of the 4K table image
  <dst>/labels/<split>/<name>.txt   "class x1 y1 x2 y2 ..." normalised to the crop
  <dst>/index_<split>.json          per-instance kept/dropped status, for eval

    .venv/bin/python -m mm.datagen.masks_to_yolo --write-roi
    .venv/bin/python -m mm.datagen.masks_to_yolo --src data/synth --dst data/yolo

The crop keeps 10 mm cubes at ~26 px so they survive resizing to the training
image size. The same crop (configs/roi.yaml) must be applied at inference.

Instances under MIN_VISIBLE_PX, or under MIN_VISIBLE_FRAC of their unoccluded
silhouette, are dropped from the labels and listed as hard in the index. The
unoccluded silhouette is exact here: every object is convex (caps nearly so),
so it is the convex hull of its projected collider shape at the settled pose.
"""

import argparse
import functools
import glob
import json
import math
import os
from multiprocessing import Pool

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

import scene_config as C
from backproject import CameraModel
from mm.datagen import randomize as R

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROI_PATH = os.path.join(ROOT, "configs", "roi.yaml")
CLASSES_PATH = os.path.join(ROOT, "configs", "classes.yaml")

MIN_VISIBLE_PX = 30
MIN_VISIBLE_FRAC = 0.40
POLY_EPS_PX = 1.0
ROI_MARGIN_PX = 64
ROI_ALIGN = 32
NOMINAL_RES = tuple(2 * v for v in C.TABLE_CAM["resolution"])  # 4K planning capture


# ------------------------------------------------------------------ geometry
def _ring(radius, z, n=32):
    a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(a), radius * np.sin(a), np.full(n, z)], axis=1)


def object_points(cls, orientation=None):
    """Collider surface samples in the root prim's frame (as cell.py builds it)."""
    if cls == "cube":  # origin = bottom-face centre
        h = C.CUBE_EDGE / 2
        return np.array(
            [(x, y, z) for x in (-h, h) for y in (-h, h) for z in (0.0, C.CUBE_EDGE)]
        )
    if cls == "cap":  # origin = bottom face; flange and tapered plug
        f, p = C.CAP_FLANGE_THICK, C.CAP_PLUG_LENGTH
        rb, rt = C.CAP_PLUG_BASE_RADIUS, C.CAP_PLUG_TIP_RADIUS
        if orientation == "wide_end_up":
            parts = [(rt, 0.0), (rb, p), (C.CAP_RADIUS, p), (C.CAP_RADIUS, p + f)]
        else:
            parts = [(C.CAP_RADIUS, 0.0), (C.CAP_RADIUS, f), (rb, f), (rt, f + p)]
        return np.concatenate([_ring(r, z) for r, z in parts])
    if cls == "aaa_cell":  # origin = centre, axis local Z; visible parts, not collider
        L, r, t = C.AAA_CELL_LENGTH, C.AAA_CELL_RADIUS, C.AAA_CELL_END_T
        nub_r, nub_h = C.AAA_CELL_NUB
        parts = [
            (0.92 * r, -L / 2),  # negative end disc
            (r, -L / 2 + t),  # wrap
            (r, L / 2 - nub_h - t),
            (0.92 * r, L / 2 - nub_h),  # positive end disc
            (nub_r, L / 2),  # terminal nub
        ]
        return np.concatenate([_ring(rad, z) for rad, z in parts])
    if cls == "tube":  # origin = tail, axis local Z
        return np.concatenate([_ring(C.VIAL_RADIUS, z) for z in (0.0, C.VIAL_LENGTH)])
    if cls == "screw":
        return np.concatenate(object_parts(cls))
    if cls == "bin":  # origin = base centre (cell._make_bin), the outer box
        d = R.load_bin_design()
        hx, hy = d["outer"][0] / 2, d["outer"][1] / 2
        return np.array([(x, y, z) for x in (-hx, hx) for y in (-hy, hy) for z in (0.0, d["wall_h"])])
    raise ValueError(cls)


def object_parts(cls, orientation=None):
    """Convex pieces whose union is the object's silhouette.

    One piece for convex objects. A screw is not convex (12 mm head on a 6 mm
    shank): a single hull would add ~25% area along the shank and cap a perfect
    fit at IoU ~0.79, so it is the head-with-countersink plus the shank.
    """
    if cls != "screw":
        return [object_points(cls, orientation)]
    # origin = head face, tip along local -Z (cell.add_screws)
    hr, sr = C.SCREW_HEAD_D / 2, C.SCREW_SHANK_D / 2
    L, hh = C.SCREW_LENGTH, C.SCREW_HEAD_H
    head = [(hr, 0.0), (hr, -(hh - (hr - sr))), (sr, -hh)]
    shank = [(sr, -hh), (sr, -(L - 0.0008)), (0.8 * sr, -L)]
    return [np.concatenate([_ring(r, z) for r, z in part]) for part in (head, shank)]


def project_object(obj, cam):
    """Pixel coordinates of an object's collider samples at its settled pose."""
    w, x, y, z = obj["quat_wxyz"]
    rot = Rotation.from_quat([x, y, z, w]).as_matrix()
    pts = object_points(obj["class"], obj.get("orientation")) @ rot.T + obj["position"]
    uv = [cam.project(p) for p in pts]
    return np.array([p for p in uv if p is not None], np.float32)


def amodal_area(obj, cam):
    """Unoccluded silhouette area in pixels (union of projected convex parts)."""
    w, x, y, z = obj["quat_wxyz"]
    rot = Rotation.from_quat([x, y, z, w]).as_matrix()
    hulls = []
    for part in object_parts(obj["class"], obj.get("orientation")):
        uv = [cam.project(p) for p in part @ rot.T + obj["position"]]
        uv = np.array([p for p in uv if p is not None], np.float32)
        if len(uv) >= 3:
            hulls.append(cv2.convexHull(uv).reshape(-1, 2))
    if not hulls:
        return 0.0
    if len(hulls) == 1:
        return float(cv2.contourArea(hulls[0]))
    pts = np.concatenate(hulls)  # rasterise the union at 4x in its bounding box
    lo = np.floor(pts.min(axis=0)) - 1
    size = (np.ceil(pts.max(axis=0)) - lo + 2).astype(int) * 4
    canvas = np.zeros((size[1], size[0]), np.uint8)
    for h in hulls:
        cv2.fillConvexPoly(canvas, np.round((h - lo) * 4 * 16).astype(np.int32), 1, cv2.LINE_8, 4)
    return float(canvas.sum()) / 16.0


# ------------------------------------------------------------------ polygons
def mask_to_polygon(binary, eps=POLY_EPS_PX):
    """Largest external contour, simplified. Returns (N, 2) pixel coords or None.

    Coordinates are continuous (pixel centre = index + 0.5), matching
    CameraModel.project/ray.
    """
    contours, _ = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    if len(c) < 3:
        return None
    return c.astype(np.float64) + 0.5


def polygon_to_mask(poly, shape):
    """Inverse of mask_to_polygon, for round-trip checks."""
    m = np.zeros(shape, np.uint8)
    pts = np.round(np.asarray(poly) - 0.5).astype(np.int32)
    cv2.fillPoly(m, [pts], 1)
    return m


def label_line(class_id, poly, width, height):
    xy = np.clip(np.asarray(poly) / [width, height], 0.0, 1.0)
    return f"{class_id} " + " ".join(f"{v:.6f}" for v in xy.reshape(-1))


# ------------------------------------------------------------------ ROI
def nominal_camera(resolution=NOMINAL_RES):
    """Table camera from scene_config (no jitter), at the given resolution."""
    from test_support import camera_model

    return camera_model(dict(C.TABLE_CAM, resolution=tuple(resolution)))


@functools.lru_cache(maxsize=4)
def bin_region_centres(step=0.02):
    """Grid of every bin centre randomize.bin_pose_ok accepts (B8 bin region).

    ~1,400 IK checks (~40 s): cached per process.
    """
    xs = np.arange(R.BIN_REGION_X[0], R.BIN_REGION_X[1] + 1e-9, step)
    ys = np.arange(R.BIN_REGION_Y[0], R.BIN_REGION_Y[1] + 1e-9, step)
    return tuple((float(x), float(y)) for x in xs for y in ys if R.bin_pose_ok((x, y)))


def roi_world_points(bin_centres=None):
    """Points the crop must contain: targets, tube overhang, and every bin pose
    the B8 region allows (a bin's circumscribed circle, base and rim)."""
    reach = C.VIAL_LENGTH / 2 + C.VIAL_RADIUS
    boxes = [
        (R.TARGET_X, R.TARGET_Y, 0.03),
        (
            (R.DISTRACTOR_X[0] - reach, R.DISTRACTOR_X[1] + reach),
            (R.DISTRACTOR_Y[0] - reach, R.DISTRACTOR_Y[1] + reach),
            0.02,
        ),
    ]
    z0 = C.TABLE_TOP_Z + C.SURFACE_DZ
    pts = [(x, y, z0 + z) for xs, ys, h in boxes for x in xs for y in ys for z in (0.0, h)]
    centres = list(bin_region_centres()) if bin_centres is None else bin_centres
    d, rad = R.load_bin_design(), R._bin_radius()
    for cx, cy in centres + [tuple(b["xy"]) for b in R.load_bins().values()]:
        for k in range(8):
            a = 2 * math.pi * k / 8
            for h in (0.0, d["wall_h"]):
                pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a), z0 + h))
    return pts


def compute_roi(cam=None, margin=ROI_MARGIN_PX, align=ROI_ALIGN):
    cam = cam or nominal_camera()
    uv = np.array([cam.project(p) for p in roi_world_points()])
    x0, y0 = np.floor(uv.min(axis=0)) - margin
    x1, y1 = np.ceil(uv.max(axis=0)) + margin
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    w = min(cam.W - x0, int(math.ceil((x1 - x0) / align) * align))
    h = min(cam.H - y0, int(math.ceil((y1 - y0) / align) * align))
    return {"resolution": [cam.W, cam.H], "x0": x0, "y0": y0, "width": w, "height": h}


def load_roi(path=ROI_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def roi_for(roi, width, height):
    """Scale the stored ROI to an image of another resolution."""
    sx, sy = width / roi["resolution"][0], height / roi["resolution"][1]
    return (
        int(round(roi["x0"] * sx)),
        int(round(roi["y0"] * sy)),
        int(round(roi["width"] * sx)),
        int(round(roi["height"] * sy)),
    )


# ------------------------------------------------------------------ conversion
def convert_frame(meta, image, mask, roi):
    """Return (crop, label lines, per-instance records) for one frame."""
    cam = CameraModel(meta["camera"])
    x0, y0, w, h = roi_for(roi, cam.W, cam.H)
    crop = image[y0 : y0 + h, x0 : x0 + w]
    mcrop = mask[y0 : y0 + h, x0 : x0 + w]
    lines, records = [], []
    for obj in meta["objects"]:
        inst = mcrop == obj["index"]
        visible = int(inst.sum())
        full = amodal_area(obj, cam)
        frac = visible / full if full > 0 else 0.0
        rec = {
            "index": obj["index"],
            "class": obj["class"],
            "visible_px": visible,
            "visible_frac": round(frac, 3),
        }
        if visible < MIN_VISIBLE_PX:
            rec["status"] = "dropped:few_px"
        elif frac < MIN_VISIBLE_FRAC:
            rec["status"] = "dropped:occluded"
        else:
            poly = mask_to_polygon(inst)
            if poly is None:
                rec["status"] = "dropped:no_polygon"
            else:
                lines.append(label_line(obj["class_id"], poly, w, h))
                rec["status"] = "kept"
        records.append(rec)
    return crop, lines, records


def _convert_one(job):
    meta_path, src_split, dst, split, roi = job
    with open(meta_path) as f:
        meta = json.load(f)
    image = cv2.imread(os.path.join(src_split, meta["image"]))
    mask = cv2.imread(os.path.join(src_split, meta["mask"]), cv2.IMREAD_UNCHANGED)
    crop, lines, records = convert_frame(meta, image, mask, roi)
    stem = os.path.splitext(os.path.basename(meta_path))[0]
    cv2.imwrite(
        os.path.join(dst, "images", split, stem + ".jpg"),
        crop,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    with open(os.path.join(dst, "labels", split, stem + ".txt"), "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return stem, records


def convert_split(src, dst, split, roi, workers=None):
    src_split = os.path.join(src, split)
    metas = sorted(glob.glob(os.path.join(src_split, "meta", "*.json")))
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(dst, sub, split), exist_ok=True)
    jobs = [(m, src_split, dst, split, roi) for m in metas]
    with Pool(workers) as pool:
        results = pool.map(_convert_one, jobs, chunksize=8)
    index = {stem: recs for stem, recs in results}
    with open(os.path.join(dst, f"index_{split}.json"), "w") as f:
        json.dump(index, f)
    counts = {}
    for recs in index.values():
        for r in recs:
            key = (r["class"], r["status"])
            counts[key] = counts.get(key, 0) + 1
    return len(metas), counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=os.path.join(ROOT, "data", "synth"))
    parser.add_argument("--dst", default=os.path.join(ROOT, "data", "yolo"))
    parser.add_argument("--splits", nargs="+", default=list(R.SPLITS))
    parser.add_argument("--write-roi", action="store_true", help="recompute configs/roi.yaml")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    if args.write_roi or not os.path.exists(ROI_PATH):
        roi = compute_roi()
        with open(ROI_PATH, "w") as f:
            f.write(
                "# Fixed table-camera crop for YOLO training and inference (A4/A7).\n"
                "# Pixel box in the full table image at `resolution`; roi_for() scales it.\n"
                "# Generated by masks_to_yolo.py --write-roi from scene_config's\n"
                "# table camera, target/distractor areas and bins. Regenerate if any change.\n"
            )
            yaml.safe_dump(roi, f, sort_keys=False)
        print(f"[masks_to_yolo] wrote {ROI_PATH}: {roi}")
    roi = load_roi()

    for split in args.splits:
        if not os.path.isdir(os.path.join(args.src, split, "meta")):
            print(f"[masks_to_yolo] {split}: no frames, skipped")
            continue
        n, counts = convert_split(args.src, args.dst, split, roi, args.workers)
        summary = ", ".join(f"{c}/{s}={k}" for (c, s), k in sorted(counts.items()))
        print(f"[masks_to_yolo] {split}: {n} frames; {summary}")


if __name__ == "__main__":
    main()
