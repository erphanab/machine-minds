"""
The fine-tune set for real images: the v4 sim set plus reviewed real-cell labels, both as the robot sees them
(configs/roi.yaml workspace crop of the 1080p image, as perceive_yolo.detect). Host python:

    python -m mm.datagen.build_real_finetune --real data/real_labels_20261027 --out data/yolo_v5_real_1080

--real: reviewed labels in Ultralytics YOLO segmentation layout, full 1920x1080 frames (images/<split>,
labels/<split>, split train / heldout; autolabel_real.py's layout). Writes <out>/:
  train  the sim train split (hard links) + every real train image, cropped, repeated --repeat times (103 real
         images against 1600 sim ones would barely count otherwise)
  val    the sim val split (hard links): picks best.pt, so the real held-out images play no part in training
  test   the real held-out rollouts, cropped: evaluation only
and configs/dataset_v5_real_1080.yaml.
"""

import argparse
import os
import shutil

import cv2
import numpy as np

from mm.datagen.masks_to_yolo import load_roi, roi_for

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MIN_PX = 20  # a label cropped down to fewer pixels than this is dropped


def crop_labels(lines, size, box):
    """YOLO-seg lines of a full frame (W, H) -> lines of its crop box (x0, y0, w, h): each polygon clipped to the
    box (largest piece), normalized to the crop; dropped if under MIN_PX inside it."""
    (W, H), (x0, y0, w, h) = size, box
    out = []
    for line in lines:
        v = line.split()
        if len(v) < 7:
            continue
        pts = np.asarray(v[1:], float).reshape(-1, 2) * [W, H]
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
        c = np.ascontiguousarray(m[y0:y0 + h, x0:x0 + w])
        if int(c.sum()) < MIN_PX:
            continue
        contours, _ = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        big = max(contours, key=cv2.contourArea).reshape(-1, 2)
        if len(big) < 3:
            continue
        out.append(f"{v[0]} " + " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in big))
    return out


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--sim", default=os.path.join("data", "yolo_v4_1080"))
    ap.add_argument("--out", default=os.path.join("data", "yolo_v5_real_1080"))
    ap.add_argument("--repeat", type=int, default=4)
    a = ap.parse_args()
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    for kind in ("images", "labels"):
        for s in ("train", "val", "test"):
            os.makedirs(os.path.join(a.out, kind, s))
    counts = {"sim_train": 0, "sim_val": 0, "real_train": 0, "real_test": 0, "real_labels_dropped": 0}
    for split, key in (("train", "sim_train"), ("val", "sim_val")):
        for name in sorted(os.listdir(os.path.join(a.sim, "images", split))):
            stem = os.path.splitext(name)[0]
            link(os.path.join(a.sim, "images", split, name), os.path.join(a.out, "images", split, name))
            link(os.path.join(a.sim, "labels", split, stem + ".txt"), os.path.join(a.out, "labels", split, stem + ".txt"))
            counts[key] += 1
    roi = load_roi()
    for split, dst, reps, key in (("train", "train", a.repeat, "real_train"), ("heldout", "test", 1, "real_test")):
        for name in sorted(os.listdir(os.path.join(a.real, "images", split))):
            stem = os.path.splitext(name)[0]
            img = cv2.imread(os.path.join(a.real, "images", split, name))
            H, W = img.shape[:2]
            box = roi_for(roi, W, H)
            x0, y0, w, h = box
            crop = img[y0:y0 + h, x0:x0 + w]
            lab = os.path.join(a.real, "labels", split, stem + ".txt")
            lines = open(lab).read().splitlines() if os.path.exists(lab) else []
            kept = crop_labels(lines, (W, H), box)
            counts["real_labels_dropped"] += len([x for x in lines if x.strip()]) - len(kept)
            for k in range(reps):
                out = f"real_{stem}" + (f"_r{k}" if reps > 1 else "")
                cv2.imwrite(os.path.join(a.out, "images", dst, out + ".jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                with open(os.path.join(a.out, "labels", dst, out + ".txt"), "w") as f:
                    f.write("\n".join(kept) + ("\n" if kept else ""))
            counts[key] += 1
    cfg = os.path.join(ROOT, "configs", "dataset_v5_real_1080.yaml")
    with open(cfg, "w") as f:
        f.write("# v5: the v4 sim set + reviewed real-cell images (27 Oct 2026, 4 training rollouts x "
                f"{a.repeat}), ROI crops at 1080p.\n# val = sim val (picks best.pt); test = the 2 held-out real "
                "rollouts (evaluation only). Built by mm.datagen.build_real_finetune.\n"
                f"path: {os.path.relpath(a.out, ROOT).replace(os.sep, '/')}\ntrain: images/train\nval: images/val\n"
                "test: images/test\nnames:\n  0: cube\n  1: cap\n  2: aaa_cell\n  3: tube\n  4: screw\n  5: bin\n")
    print(counts, "->", cfg)


if __name__ == "__main__":
    main()
