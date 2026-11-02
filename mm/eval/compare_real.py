"""
Old vs new YOLO weights on the held-out real rollouts, through the robot's own detection (perceive_yolo.detect:
ROI crop, class-agnostic NMS, the card's inference size), against reviewed labels. YOLO environment:

    C:\\isim\\yolo\\python.exe -m mm.eval.compare_real --old models/yolo_seg_v4_1080/weights/best.pt \\
        --new models/yolo_seg_v5_real_1080/weights/best.pt --labels data/real_labels_20261027

Labels are clipped to the ROI; those (nearly) outside it are dropped: the robot never sees them, and both models
would miss them alike. A prediction matches a label of its class at mask IoU >= 0.5 (greedy, highest confidence first). Reports per
class recall and precision at the pick gate (0.5) and at detect's floor (0.25), and the sim test split's mask
mAP50 (configs/dataset_v4_1080.yaml) so a real-image gain does not hide a sim loss. Writes --out.

"Clearly better", fixed before any result is seen; every condition must hold:
  - targets (cube, aaa_cell, screw pooled): recall at the gate up by >= 0.10
  - no target class: recall at the gate down by > 0.05
  - targets pooled: precision at the gate down by <= 0.05
  - bins: recall at the gate down by <= 0.02
  - sim test: mask mAP50 down by <= 0.02
The held-out labels must have been reviewed by hand: unreviewed, they are the old model's own output and the
comparison favours it.
"""

import argparse
import json
import os
from types import SimpleNamespace

import cv2
import numpy as np

from mm.datagen.build_real_finetune import MIN_PX
from mm.datagen.masks_to_yolo import roi_for
from mm.perception import perceive_yolo as PY

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TARGETS = ("cube", "aaa_cell", "screw")
GATE, FLOOR = PY.CONF_MIN, PY.PREDICT_CONF


def poly_mask(pts, shape):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(pts, float).reshape(-1, 2)).astype(np.int32)], 1)
    return m.astype(bool)


def evaluate(weights, images, names):
    from ultralytics import YOLO

    model, imgsz, roi = YOLO(weights), PY.model_imgsz(weights), PY.load_roi()
    tally = {}  # (class, threshold) -> [tp, fp, fn]
    for img_path, lab_path in images:
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        x0, y0, w, h = roi_for(roi, W, H)
        inside = np.zeros((H, W), bool)
        inside[y0:y0 + h, x0:x0 + w] = True
        gts = []  # labels clipped to the ROI the robot sees; ones (nearly) outside it dropped, as in build_real_finetune
        for line in (open(lab_path).read().splitlines() if os.path.exists(lab_path) else []):
            v = line.split()
            if len(v) >= 7:
                gm = poly_mask(np.asarray(v[1:], float).reshape(-1, 2) * [W, H], (H, W)) & inside
                if gm.sum() >= MIN_PX:
                    gts.append((names[int(v[0])], gm))
        preds = [(d["class"], d["conf"], poly_mask(d["polygon_px"], (H, W)))
                 for d in PY.detect(model, img, SimpleNamespace(W=W, H=H), roi, imgsz)]
        for thr in (GATE, FLOOR):
            used = set()
            for cls, conf, pm in sorted((p for p in preds if p[1] >= thr), key=lambda p: -p[1]):
                best, best_iou = None, 0.5
                for k, (gc, gm) in enumerate(gts):
                    if k in used or gc != cls:
                        continue
                    iou = np.logical_and(pm, gm).sum() / max(1, np.logical_or(pm, gm).sum())
                    if iou >= best_iou:
                        best, best_iou = k, iou
                t = tally.setdefault((cls, thr), [0, 0, 0])
                if best is None:
                    t[1] += 1
                else:
                    used.add(best)
                    t[0] += 1
            for k, (gc, _) in enumerate(gts):
                if k not in used:
                    tally.setdefault((gc, thr), [0, 0, 0])[2] += 1
    out = {}
    for thr in (GATE, FLOOR):
        per = {}
        for cls in names.values():
            tp, fp, fn = tally.get((cls, thr), [0, 0, 0])
            per[cls] = {"labels": tp + fn, "tp": tp, "fp": fp,
                        "recall": round(tp / (tp + fn), 3) if tp + fn else None,
                        "precision": round(tp / (tp + fp), 3) if tp + fp else None}
        tp = sum(per[c]["tp"] for c in TARGETS)
        fp = sum(per[c]["fp"] for c in TARGETS)
        lab = sum(per[c]["labels"] for c in TARGETS)
        per["targets"] = {"labels": lab, "tp": tp, "fp": fp, "recall": round(tp / lab, 3) if lab else None,
                          "precision": round(tp / (tp + fp), 3) if tp + fp else None}
        out[f"conf_{thr}"] = per
    return out


def sim_map50(weights, data):
    from ultralytics import YOLO

    from mm.perception.train_yolo import resolve_dataset
    import tempfile

    import yaml

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(resolve_dataset(data), tmp)
    r = YOLO(weights).val(data=tmp.name, split="test", imgsz=PY.model_imgsz(weights), batch=4, plots=False, verbose=False)
    return round(float(r.seg.map50), 4)


def verdict(old, new):
    o, n = old["real"][f"conf_{GATE}"], new["real"][f"conf_{GATE}"]
    d = lambda part, key: (n[part][key] or 0) - (o[part][key] or 0)  # noqa: E731
    checks = {
        "targets recall +>= 0.10": d("targets", "recall") >= 0.10,
        "no target class recall -> 0.05": all(d(c, "recall") >= -0.05 for c in TARGETS),
        "targets precision ->= -0.05": d("targets", "precision") >= -0.05,
        "bin recall ->= -0.02": d("bin", "recall") >= -0.02,
        "sim test mAP50 ->= -0.02": new["sim_test_mask_map50"] - old["sim_test_mask_map50"] >= -0.02,
    }
    return {"clearly_better": all(checks.values()), "checks": checks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--labels", required=True, help="reviewed labels, YOLO-seg layout, full frames")
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--sim-data", default=os.path.join(ROOT, "configs", "dataset_v4_1080.yaml"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "compare_real.json"))
    a = ap.parse_args()
    names = {int(k): v for k, v in enumerate(["cube", "cap", "aaa_cell", "tube", "screw", "bin"])}
    d = os.path.join(a.labels, "images", a.split)
    images = [(os.path.join(d, f), os.path.join(a.labels, "labels", a.split, os.path.splitext(f)[0] + ".txt"))
              for f in sorted(os.listdir(d))]
    res = {}
    for tag, w in (("old", a.old), ("new", a.new)):
        res[tag] = {"weights": os.path.abspath(w), "real": evaluate(w, images, names), "sim_test_mask_map50": sim_map50(w, a.sim_data)}
    res["images"] = len(images)
    res["verdict"] = verdict(res["old"], res["new"])
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    for thr in (GATE, FLOOR):
        print(f"conf >= {thr}:")
        for cls in [*names.values(), "targets"]:
            o, n = res["old"]["real"][f"conf_{thr}"][cls], res["new"]["real"][f"conf_{thr}"][cls]
            print(f"  {cls:9s} labels {o['labels']:4d}  recall {o['recall']} -> {n['recall']}  "
                  f"precision {o['precision']} -> {n['precision']}")
    print(f"sim test mask mAP50 {res['old']['sim_test_mask_map50']} -> {res['new']['sim_test_mask_map50']}")
    print("verdict:", json.dumps(res["verdict"]))


if __name__ == "__main__":
    main()
