"""
Perception metrics that matter for grasping (Machine Minds step A6).

Runs in the YOLO environment over a datagen split (full 4K frames + meta.json),
running the same detect -> refine path as perceive_yolo.py.

    $YOLO_PY mm/eval/eval_perception.py --split test
    $YOLO_PY mm/eval/eval_perception.py --split hard

Per class, at the chosen confidence:
  recall / precision     prediction <-> GT matched by mask IoU >= 0.5,
                         class-agnostic, so misclassifications show as confusion
  confusion              GT class x predicted class (+ missed / spurious)
  centroid error (mm)    refined table XY vs GT centre, eligible upright objects
  yaw error (deg)        cubes mod 90, AAA cells mod 180 (+ polarity rate)
  eligible rate          how many GT targets reach a pickable detection
Dropped (tilted) GT objects are reported apart; the risk there is one being
marked eligible. Mask mAP comes from training (card.json).

Targets (guide): recall >= 98%, confusion <= 1%, centroid <= 1 mm, yaw <= 3 deg.
Writes models/<run>/perception_<split>.json and prints a summary table.
GT is used here for scoring only.
"""

import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from backproject import CameraModel  # noqa: E402
from mm.datagen.masks_to_yolo import (  # noqa: E402
    MIN_VISIBLE_FRAC,
    MIN_VISIBLE_PX,
    amodal_area,
    load_roi,
)
from mm.eval.gt import gt_centre_xy, gt_yaw_deg, yaw_error  # noqa: E402
from mm import paths  # noqa: E402
from mm.perception import perceive_yolo as PY  # noqa: E402

MATCH_IOU = 0.5
TARGETS = {"recall": 0.98, "confusion": 0.01, "centroid_mm": 1.0, "yaw_deg": 3.0}


def scaled(meta_camera, scale):
    """The capture camera at another resolution (same view; a smaller sensor)."""
    w, h = meta_camera["resolution"]
    return dict(meta_camera, resolution=[round(w * scale), round(h * scale)])


def load_frame(meta, split_dir, scale=1.0):
    img = cv2.imread(os.path.join(split_dir, meta["image"]))
    if scale != 1.0:
        img = cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return img, CameraModel(scaled(meta["camera"], scale))


def score_frame(job):
    """Refine one frame's raw detections and match them to GT (worker process)."""
    meta_path, split_dir, raw, classes, conf_min = job[:5]
    scale = job[5] if len(job) > 5 else 1.0
    with open(meta_path) as f:
        meta = json.load(f)
    img, cam = load_frame(meta, split_dir, scale)
    gt_full = cv2.imread(os.path.join(split_dir, meta["mask"]), cv2.IMREAD_UNCHANGED)
    gt_mask = gt_full if scale == 1.0 else cv2.resize(gt_full, (cam.W, cam.H), interpolation=cv2.INTER_NEAREST)
    cam_full = CameraModel(meta["camera"])
    dets = PY.finalize(raw, img, cam, classes, conf_min)
    det_masks = [PY.full_mask(r, cam).astype(bool) for r in raw]

    gts = []
    for o in meta["objects"]:
        m = gt_mask == o["index"]
        # countable is decided at full resolution, so every scale scores the same objects
        vis = int((gt_full == o["index"]).sum())
        full = amodal_area(o, cam_full)
        countable = vis >= MIN_VISIBLE_PX and (vis / full if full else 0) >= MIN_VISIBLE_FRAC
        gts.append({"obj": o, "mask": m, "countable": countable})

    # Greedy class-agnostic matching by mask IoU, best pairs first.
    pairs = []
    for i, dm in enumerate(det_masks):
        for j, g in enumerate(gts):
            inter = np.logical_and(dm, g["mask"]).sum()
            if inter:
                iou = inter / np.logical_or(dm, g["mask"]).sum()
                if iou >= MATCH_IOU:
                    pairs.append((iou, i, j))
    used_d, used_g, rows = set(), set(), []
    for iou, i, j in sorted(pairs, reverse=True):
        if i in used_d or j in used_g:
            continue
        used_d.add(i)
        used_g.add(j)
        o, d = gts[j]["obj"], dets[i]
        row = {
            "frame": os.path.basename(meta_path)[:-5],
            "gt_class": o["class"],
            "pred_class": d["class"],
            "conf": d["conf"],
            "mask_iou": round(float(iou), 3),
            "countable": gts[j]["countable"],
            "dropped": o["dropped"],
            "eligible": d["eligible"],
            "skip_reason": d["skip_reason"],
            "fit_iou": d["fit_iou"],
            "gt_xy": [round(v, 4) for v in gt_centre_xy(o)],
        }
        if d["xy_robot_m"] is not None:
            row["centroid_mm"] = float(np.linalg.norm(np.subtract(d["xy_robot_m"], gt_centre_xy(o))) * 1000)
        truth = gt_yaw_deg(o)
        row["yaw_err_deg"] = yaw_error(o["class"], d["yaw_deg"], truth)
        if o["class"] in ("aaa_cell", "screw") and d["yaw_deg"] is not None and truth is not None:
            row["polarity_ok"] = bool(abs((d["yaw_deg"] - truth + 180) % 360 - 180) < 90)
        if o["class"] == "cap":
            row["orientation_ok"] = d.get("orientation") == o.get("orientation")
        rows.append(row)
    for j, g in enumerate(gts):
        if j not in used_g:
            rows.append(
                {
                    "frame": os.path.basename(meta_path)[:-5],
                    "gt_class": g["obj"]["class"],
                    "pred_class": None,
                    "countable": g["countable"],
                    "dropped": g["obj"]["dropped"],
                    "gt_xy": [round(v, 4) for v in gt_centre_xy(g["obj"])],
                }
            )
    for i, d in enumerate(dets):
        if i not in used_d:
            rows.append({"frame": os.path.basename(meta_path)[:-5], "gt_class": None, "pred_class": d["class"], "conf": d["conf"], "eligible": d["eligible"]})
    return rows


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def summarise(rows, class_names, conf_min):
    out = {"per_class": {}, "confusion": {}}
    hits = [r for r in rows if r.get("gt_class") and r.get("pred_class") and r.get("conf", 0) >= conf_min]
    for c in class_names:
        gt_c = [r for r in rows if r.get("gt_class") == c and r["countable"]]
        tp = [r for r in hits if r["gt_class"] == c and r["countable"] and r["pred_class"] == c]
        preds = [r for r in rows if r.get("pred_class") == c and r.get("conf", 0) >= conf_min]
        good = [r for r in preds if r.get("gt_class") == c]
        upright = [r for r in tp if not r["dropped"]]
        # Bins are never "eligible" (never picked); score every bin whose fit
        # passes the threshold B8 uses (perception/bins.MIN_FIT).
        ok = [r for r in upright if r["eligible"] or (c == "bin" and r.get("fit_iou", 0) >= 0.80)]
        cent = [r["centroid_mm"] for r in ok if "centroid_mm" in r]
        yaw = [r["yaw_err_deg"] for r in ok if r.get("yaw_err_deg") is not None]
        stats = {
            "gt": len(gt_c),
            "recall": len(tp) / len(gt_c) if gt_c else None,
            "precision": len(good) / len(preds) if preds else None,
            "eligible_rate": len(ok) / len(upright) if upright else None,
            "centroid_mm": {"median": _pct(cent, 50), "p95": _pct(cent, 95), "max": max(cent) if cent else None},
            "yaw_deg": {"median": _pct(yaw, 50), "p95": _pct(yaw, 95), "max": max(yaw) if yaw else None},
            "dropped_marked_eligible": sum(1 for r in tp if r["dropped"] and r["eligible"]),
            "skip_reasons": {},
        }
        for r in upright:
            if r["skip_reason"]:
                stats["skip_reasons"][r["skip_reason"]] = stats["skip_reasons"].get(r["skip_reason"], 0) + 1
        if c in ("aaa_cell", "screw"):
            pol = [r["polarity_ok"] for r in ok if "polarity_ok" in r]
            stats["polarity_rate"] = float(np.mean(pol)) if pol else None
        if c == "cap":
            ori = [r["orientation_ok"] for r in tp if "orientation_ok" in r]
            stats["orientation_rate"] = float(np.mean(ori)) if ori else None
        out["per_class"][c] = stats
        row = {p: 0 for p in class_names + ["missed"]}
        for r in rows:
            if r.get("gt_class") == c and r["countable"]:
                p = r.get("pred_class") if r.get("pred_class") and r.get("conf", 0) >= conf_min else "missed"
                row[p] += 1
        out["confusion"][c] = row
    total = sum(sum(v for k, v in row.items() if k != "missed") for row in out["confusion"].values())
    wrong = sum(v for c, row in out["confusion"].items() for k, v in row.items() if k not in (c, "missed"))
    out["confusion_rate"] = wrong / total if total else None
    out["spurious"] = sum(1 for r in rows if r.get("gt_class") is None and r.get("conf", 0) >= conf_min)
    return out


def print_table(summary):
    print(f"{'class':8s} {'GT':>5s} {'recall':>7s} {'prec':>6s} {'elig':>6s} {'cent p95':>9s} {'cent max':>9s} {'yaw p95':>8s}")
    for c, s in summary["per_class"].items():
        f = lambda v, d=3: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
        print(
            f"{c:8s} {s['gt']:5d} {f(s['recall'])} {f(s['precision'])} {f(s['eligible_rate'])} "
            f"{f(s['centroid_mm']['p95'], 2):>9s} {f(s['centroid_mm']['max'], 2):>9s} {f(s['yaw_deg']['p95'], 1):>8s}"
        )
    print(f"confusion rate {summary['confusion_rate']:.4f}, spurious detections {summary['spurious']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--src", default=os.path.join(ROOT, "data", "synth_v3"))
    parser.add_argument("--run", default=PY.DEFAULT_RUN, help="models/<run>")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--conf", type=float, default=PY.CONF_MIN)
    parser.add_argument("--imgsz", type=int, default=None, help="default: the model card's")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--scale", type=float, default=1.0, help="0.5: score the 4K frames at 1080p (the real camera)")
    parser.add_argument("--tag", default=None, help="output suffix: perception_<split>_<tag>.json")
    args = parser.parse_args()

    from ultralytics import YOLO

    run_dir = paths.model_dir(args.run)
    weights = args.weights or os.path.join(run_dir, "weights", "best.pt")
    model = YOLO(weights)
    classes = PY.load_classes()
    roi = load_roi()
    split_dir = os.path.join(args.src, args.split)
    metas = sorted(glob.glob(os.path.join(split_dir, "meta", "*.json")))[: args.limit]

    t0 = time.time()
    jobs = []
    for mp in metas:  # GPU stage, in this process
        with open(mp) as f:
            meta = json.load(f)
        img, cam = load_frame(meta, split_dir, args.scale)
        raw = PY.detect(model, img, cam, roi, args.imgsz or PY.model_imgsz(weights))
        jobs.append((mp, split_dir, raw, classes, args.conf, args.scale))
    t_detect = time.time() - t0
    with Pool(args.workers) as pool:  # CPU stage: pose refinement + matching
        rows = [r for frame in pool.imap(score_frame, jobs, chunksize=2) for r in frame]
    summary = summarise(rows, list(classes), args.conf)
    summary.update(
        split=args.split,
        frames=len(metas),
        weights=os.path.relpath(weights, ROOT).replace(os.sep, "/"),
        model=PY.model_card(weights),
        conf_min=args.conf,
        scale=args.scale,
        targets=TARGETS,
        seconds={"detect": round(t_detect), "total": round(time.time() - t0)},
    )
    os.makedirs(run_dir, exist_ok=True)
    out = os.path.join(run_dir, f"perception_{args.split}{'_' + args.tag if args.tag else ''}.json")
    with open(out, "w") as f:
        json.dump({"summary": summary, "objects": rows}, f, indent=1)
    print_table(summary)
    print(f"[eval_perception] {len(metas)} frames -> {out}")


if __name__ == "__main__":  # required on Windows for the worker pool
    main()
