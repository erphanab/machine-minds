"""
YOLO detection + silhouette pose refinement -> detections.json (Machine Minds A7).

Runs in the YOLO environment. Replaces perceive.py (Cosmos) in the pipeline;
Cosmos stays available as an optional end-of-run bin check.

    $YOLO_PY mm/perception/perceive_yolo.py --run runs/stack_01
    # reads  runs/stack_01/world_camera.png, camera.json  (capture_scene.py)
    # writes runs/stack_01/detections.json

Steps: crop the configs/roi.yaml workspace -> YOLO-seg -> per instance, refine
the tabletop pose from its mask (pose_fit.py, IoU >= 0.85 gate) -> robot XY.
Uses only the image and the calibrated camera; no simulator ground truth.

Per object: id, class, conf, polygon_px, xy_robot_m, yaw_deg, fit_iou,
eligible, skip_reason (+ colour for cubes, orientation for caps).
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from backproject import CameraModel  # noqa: E402
from mm.datagen.masks_to_yolo import load_roi, roi_for  # noqa: E402
from mm import paths  # noqa: E402
from mm.perception import outline_grasp as OG  # noqa: E402
from mm.perception import pose_fit  # noqa: E402

# v4: tangerine bins, 1080p (the real camera), inferred at 1600; perceive_server
# resizes 4K sim captures to its card's input_resolution. v3: 4K, coloured bins.
DEFAULT_RUN = "yolo_seg_v4_1080"
CONF_MIN = 0.5  # detections below this are reported but never eligible
# v1/v2 models named class 2 "aa_cell"; v3 renamed it (AAA cells). v2 detects
# AAA cells under the old name, so old models still run against v3 configs.
LEGACY_NAMES = {"aa_cell": "aaa_cell"}
PREDICT_CONF = 0.25  # floor passed to YOLO, so low-confidence objects still show up
# v6 step 2: cubes and screws are grasped from their outline (outline_grasp.py), with no pose fit or match-score
# gate. MM_GRASP=pose_fit (or "2" in MM_V5_STEPS, see mm/real/brain.py) switches back to v5 (pose_fit.refine and
# its 0.85 IoU gate). AAA cells, caps and bins always take pose_fit.refine, unchanged.
_V5_STEPS = {s.strip() for s in os.environ.get("MM_V5_STEPS", "").split(",")}
GRASP_MODE = os.environ.get("MM_GRASP", "pose_fit" if "2" in _V5_STEPS else "outline")
OUTLINE_CLASSES = ("cube", "screw")


def load_classes():
    with open(os.path.join(ROOT, "configs", "classes.yaml")) as f:
        return {c["name"]: c for c in yaml.safe_load(f)["classes"]}


def model_imgsz(weights, default=1280):
    """Inference size from the model's card.json: "inference_imgsz" if set (the
    1080p fine-tune fits finer masks upsampled to 1600), else the training size."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(weights))), "card.json")
    try:
        with open(path) as f:
            card = json.load(f)
        return int(card.get("inference_imgsz") or card["hyperparameters"]["imgsz"])
    except (OSError, KeyError, ValueError):
        return default


def model_card(weights):
    """card.json next to weights/best.pt, if train_yolo.py wrote one."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(weights))), "card.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        card = json.load(f)
    return {
        "card": os.path.relpath(path, ROOT).replace(os.sep, "/"),
        "name": card.get("name"),
        "dataset_sha256": card.get("dataset", {}).get("sha256"),
        "versions": card.get("versions"),
    }


def skip_reason(det, cls_cfg, conf_min):
    if cls_cfg["role"] == "bin":
        return "bin"  # a destination (B8), never picked
    if cls_cfg["role"] != "target":
        return "distractor"
    if det["conf"] < conf_min:
        return "low_confidence"
    if det["xy_robot_m"] is None:
        return "no_table_intersection"
    if det.get("grasp") == "outline":
        return None  # v6 cubes and screws: YOLO's detection is enough; the planner checks reach, TCP box and bins
    if det.get("resting_flat") is False:
        return "not_resting_flat"
    if det["fit_iou"] < pose_fit.FIT_IOU_MIN:
        return "pose_fit_below_gate"
    if det["class"] == "cap" and det.get("orientation") != "narrow_end_up":
        return "cap_wide_end_up"  # the starter only picks narrow-end-up caps
    return None


def detect(model, img_bgr, cam, roi, imgsz=1280):
    """YOLO on the ROI crop. Returns raw instances with a tight mask window.

    Each: class, conf, polygon (full-image px), window (x, y) and its bool mask.
    Kept small so refinement can run in other processes (eval_perception.py).
    """
    x0, y0, w, h = roi_for(roi, cam.W, cam.H)
    crop = img_bgr[y0 : y0 + h, x0 : x0 + w]
    result = model.predict(
        # agnostic_nms: one object, one class. v2 reported AAA cells as both
        # aa_cell and screw; the phantom twin then blocked the fingers.
        crop, imgsz=imgsz, conf=PREDICT_CONF, retina_masks=True, verbose=False, agnostic_nms=True
    )[0]
    raw = []
    if result.masks is None:
        return raw
    masks = result.masks.data.cpu().numpy() > 0.5  # (N, h, w) at crop resolution
    for k in range(len(result.boxes)):
        ys, xs = np.nonzero(masks[k])
        if len(xs) < 10:
            continue
        wx0, wy0 = xs.min(), ys.min()
        raw.append(
            {
                "class": LEGACY_NAMES.get(result.names[int(result.boxes.cls[k])], result.names[int(result.boxes.cls[k])]),
                "conf": float(result.boxes.conf[k]),
                "polygon_px": np.round(result.masks.xy[k] + [x0, y0], 1).tolist(),
                "window": (int(x0 + wx0), int(y0 + wy0)),
                "mask": masks[k][wy0 : ys.max() + 1, wx0 : xs.max() + 1],
            }
        )
    return raw


def full_mask(inst, cam):
    m = np.zeros((cam.H, cam.W), np.uint8)
    (x, y), win = inst["window"], inst["mask"]
    m[y : y + win.shape[0], x : x + win.shape[1]] = win
    return m


def finalize(raw, img_bgr, cam, classes, conf_min=CONF_MIN, grasp_mode=None):
    """Refine each raw instance's pose and decide eligibility. grasp_mode "outline" (v6): cubes and screws take
    their grasp from the mask outline (outline_grasp.py), with no fit or match-score gate; "pose_fit": v5."""
    grasp_mode = grasp_mode or GRASP_MODE
    dets = []
    for inst in raw:
        mask = full_mask(inst, cam)
        if grasp_mode == "outline" and inst["class"] in OUTLINE_CLASSES:
            pose = dict(OG.grasp(inst["class"], cam, mask), fit_iou=None, grasp="outline")
            if inst["class"] == "cube":
                pose["colour"] = pose_fit.cube_colour(img_bgr, mask)  # reported, not used to pick
        else:
            pose = pose_fit.refine(inst["class"], img_bgr, cam, mask)
        det = {
            "id": len(dets),
            "class": inst["class"],
            "conf": inst["conf"],
            "polygon_px": inst["polygon_px"],
            "xy_robot_m": pose["xy"],
            "yaw_deg": pose["yaw_deg"],
            "fit_iou": None if pose["fit_iou"] is None else round(float(pose["fit_iou"]), 4),
        }
        for extra in ("colour", "orientation", "resting_flat", "grasp", "centre_px", "rect_mm", "head_end",
                      "length_mm", "shank_mm", "head_mm", "grip_from_mm", "grip_to_mm"):
            if extra in pose:
                det[extra] = pose[extra]
        reason = skip_reason(det, classes[inst["class"]], conf_min)
        det["eligible"] = reason is None
        det["skip_reason"] = reason
        dets.append(det)
    return dets


def perceive(model, img_bgr, cam, roi, classes, imgsz=1280, conf_min=CONF_MIN, grasp_mode=None):
    """Detections for one full-resolution table image. Returns a list of dicts."""
    return finalize(detect(model, img_bgr, cam, roi, imgsz), img_bgr, cam, classes, conf_min, grasp_mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="folder with world_camera.png, camera.json")
    parser.add_argument(
        "--weights", default=os.path.join(paths.model_dir(DEFAULT_RUN), "weights", "best.pt")
    )
    parser.add_argument("--imgsz", type=int, default=None, help="default: the model card's")
    parser.add_argument("--conf", type=float, default=CONF_MIN)
    parser.add_argument("--out", default="detections.json", help="file name inside --run")
    args = parser.parse_args()

    from ultralytics import YOLO

    img = cv2.imread(os.path.join(args.run, "world_camera.png"))
    if img is None:
        raise SystemExit(f"no world_camera.png in {args.run}")
    cam = CameraModel(os.path.join(args.run, "camera.json"))
    if (img.shape[1], img.shape[0]) != (cam.W, cam.H):
        raise SystemExit(f"image {img.shape[1]}x{img.shape[0]} != camera.json {cam.W}x{cam.H}")
    roi = load_roi()
    t = time.time()
    imgsz = args.imgsz or model_imgsz(args.weights)
    dets = perceive(YOLO(args.weights), img, cam, roi, load_classes(), imgsz, args.conf)
    out = {
        "source": "perceive_yolo",
        "image": "world_camera.png",
        "model": model_card(args.weights)
        or {"weights": os.path.relpath(args.weights, ROOT).replace(os.sep, "/")},
        "roi": roi,
        "conf_min": args.conf,
        "fit_iou_min": pose_fit.FIT_IOU_MIN,
        "seconds": round(time.time() - t, 2),
        "objects": dets,
    }
    path = os.path.join(args.run, args.out)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    n_ok = sum(d["eligible"] for d in dets)
    print(f"[perceive_yolo] {len(dets)} objects, {n_ok} eligible -> {path}")


if __name__ == "__main__":
    main()
