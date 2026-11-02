"""
Auto-label the real table images a cell run saved (runs/real/<run>/rollout_*/act_*_table.jpg) with our YOLO
model, as a starting point for a person to correct in a labeling tool. Nothing here trains. YOLO environment:

    C:\\isim\\yolo\\python.exe -m mm.datagen.autolabel_real --run runs/real/20261027_165715 \\
        --heldout rollout_05 rollout_07 --out data/real_labels_20261027

Split by rollout (images of one rollout are near-duplicates, so they never straddle the split): --heldout
rollouts go to heldout/, the rest to train/. Writes, per split:
  images/<split>/<rollout>_act_NNN.jpg, labels/<split>/*.txt   Ultralytics YOLO segmentation (polygons, 0..1)
  coco_<split>.json                                             COCO instances (polygons, pixel units)
  preview/<split>/*.jpg                                         the auto-labels drawn on the image
and data.yaml, split.json and autolabel_report.json (counts per class, labels below the pick gate's 0.5).
The confidence threshold is low on purpose (--conf 0.10): deleting a wrong polygon is quicker than drawing a
missed one. Every label carries its score (COCO attribute "score") so the reviewer can sort by it.
"""

import argparse
import glob
import json
import os
import shutil

import cv2
import numpy as np

from mm import paths

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COLOURS = [(0, 0, 255), (255, 0, 255), (0, 200, 0), (255, 255, 0), (255, 128, 0), (0, 165, 255)]


def area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--heldout", nargs="+", required=True, help="rollout folder names kept out of training")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=os.path.join(paths.model_dir("yolo_seg_v4_1080"), "weights", "best.pt"))
    ap.add_argument("--imgsz", type=int, default=1600)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(a.weights)
    names = {int(k): v for k, v in model.names.items()}
    rollouts = sorted(os.path.basename(d) for d in glob.glob(os.path.join(a.run, "rollout_*"))
                      if glob.glob(os.path.join(d, "act_*_table.jpg")))
    missing = [r for r in a.heldout if r not in rollouts]
    if missing:
        raise SystemExit(f"no images in {missing}; rollouts with images: {rollouts}")
    split = {r: ("heldout" if r in a.heldout else "train") for r in rollouts}
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    report = {"run": os.path.abspath(a.run), "weights": os.path.abspath(a.weights), "imgsz": a.imgsz, "conf": a.conf,
              "split": split, "splits": {}}
    for s in ("train", "heldout"):
        for sub in ("images", "labels", "preview"):
            os.makedirs(os.path.join(a.out, sub, s), exist_ok=True)
        coco = {"images": [], "annotations": [], "categories": [{"id": k + 1, "name": v} for k, v in names.items()]}
        stats = {"rollouts": [r for r in rollouts if split[r] == s], "images": 0, "labels": {v: 0 for v in names.values()},
                 "below_0.5": {v: 0 for v in names.values()}, "images_without_labels": 0}
        for r in stats["rollouts"]:
            for path in sorted(glob.glob(os.path.join(a.run, r, "act_*_table.jpg"))):
                stem = f"{r}_{os.path.basename(path)[:-len('_table.jpg')]}"
                img = cv2.imread(path)
                h, w = img.shape[:2]
                shutil.copyfile(path, os.path.join(a.out, "images", s, stem + ".jpg"))
                res = model.predict(img, imgsz=a.imgsz, conf=a.conf, device=a.device, retina_masks=True, verbose=False)[0]
                image_id = len(coco["images"]) + 1
                coco["images"].append({"id": image_id, "file_name": stem + ".jpg", "width": w, "height": h})
                lines, view = [], img.copy()
                polys = res.masks.xy if res.masks is not None else []
                for poly, cls, conf in zip(polys, res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                    if len(poly) < 3:
                        continue
                    cls, name = int(cls), names[int(cls)]
                    lines.append(f"{cls} " + " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in poly))
                    x0, y0 = poly.min(axis=0)
                    x1, y1 = poly.max(axis=0)
                    coco["annotations"].append({
                        "id": len(coco["annotations"]) + 1, "image_id": image_id, "category_id": cls + 1,
                        "segmentation": [[round(float(v), 1) for v in poly.reshape(-1)]],
                        "bbox": [round(float(x0), 1), round(float(y0), 1), round(float(x1 - x0), 1), round(float(y1 - y0), 1)],
                        "area": round(area(poly), 1), "iscrowd": 0, "score": round(conf, 3),
                        "attributes": {"score": round(conf, 3)}})
                    stats["labels"][name] += 1
                    stats["below_0.5"][name] += conf < 0.5
                    col = COLOURS[cls % len(COLOURS)]
                    cv2.polylines(view, [poly.astype(np.int32)], True, col, 2)
                    cv2.putText(view, f"{name} {conf:.2f}", (int(x0), max(12, int(y0) - 4)), 0, 0.6, col, 2)
                with open(os.path.join(a.out, "labels", s, stem + ".txt"), "w") as f:
                    f.write("\n".join(lines) + ("\n" if lines else ""))
                cv2.imwrite(os.path.join(a.out, "preview", s, stem + ".jpg"), view, [cv2.IMWRITE_JPEG_QUALITY, 85])
                stats["images"] += 1
                stats["images_without_labels"] += not lines
                print(f"[autolabel] {s} {stem}: {len(lines)} labels", flush=True)
        with open(os.path.join(a.out, f"coco_{s}.json"), "w") as f:
            json.dump(coco, f)
        report["splits"][s] = stats
    with open(os.path.join(a.out, "data.yaml"), "w") as f:
        f.write("# Auto-labels of real cell images, for review. Not for training until reviewed (and allowed).\n"
                f"path: .\ntrain: images/train\nval: images/heldout\nnames:\n"
                + "".join(f"  {k}: {v}\n" for k, v in names.items()))
    with open(os.path.join(a.out, "split.json"), "w") as f:
        json.dump(split, f, indent=1)
    with open(os.path.join(a.out, "autolabel_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report["splits"], indent=1))


if __name__ == "__main__":
    main()
