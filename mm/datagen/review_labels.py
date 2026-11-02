"""
Review aids for a labeled real-image set (autolabel_real.py's output, or the same layout after correction):
previews of every image grouped by rollout, one contact sheet per rollout, and a list of the images most likely
wrong, to check first. Reads coco_<split>.json, so it can be re-run on corrected labels. Any Python with OpenCV:

    python -m mm.datagen.review_labels --labels data/real_labels_20261027

Writes <labels>/review/: by_rollout/<rollout>/<image>.jpg (class + score drawn; '?' under the pick gate's 0.5,
'DUP' on a second label of another class over the same object), sheets/<rollout>.jpg, check_first.csv/.md.

"Likely wrong" scores each image on what can be seen without ground truth:
  - two labels of different classes on one object (mask IoU > 0.5): one is wrong
  - labels under 0.5: the model was unsure
  - not exactly 3 bins: a bin missed or an extra one
  - per-class counts unlike the rest of its rollout (the median): while the arm is parked nothing moves, so a
    count that changes from frame to frame is a miss or a false label
Missed objects that are missed in every frame of a rollout do not show up here: look at each sheet once.
"""

import argparse
import collections
import csv
import json
import os
import shutil
import statistics

import cv2
import numpy as np

COLOURS = {"cube": (0, 0, 255), "cap": (255, 0, 255), "aaa_cell": (0, 200, 0), "tube": (255, 255, 0),
           "screw": (255, 128, 0), "bin": (0, 165, 255)}
GATE = 0.5


def mask(poly, shape):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.asarray(poly, np.float32).reshape(-1, 2).astype(np.int32)], 1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--top", type=int, default=30, help="images listed in check_first.md")
    a = ap.parse_args()
    out = os.path.join(a.labels, "review")
    for sub in ("by_rollout", "sheets"):  # only its own outputs: review/ also holds e.g. changed_round1/
        if os.path.exists(os.path.join(out, sub)):
            shutil.rmtree(os.path.join(out, sub))
    rows, by_roll = [], collections.defaultdict(list)
    for split in ("train", "heldout"):
        coco = json.load(open(os.path.join(a.labels, f"coco_{split}.json")))
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        anns = collections.defaultdict(list)
        for ann in coco["annotations"]:
            anns[ann["image_id"]].append(ann)
        for im in coco["images"]:
            roll = im["file_name"].split("_act_")[0]
            labels = [{"cls": cats[x["category_id"]], "score": x.get("score", x.get("attributes", {}).get("score", 1.0)),
                       "poly": x["segmentation"][0]} for x in anns[im["id"]]]
            by_roll[roll].append({"split": split, "file": im["file_name"], "size": (im["height"], im["width"]),
                                  "labels": labels})
    for roll, ims in sorted(by_roll.items()):
        ims.sort(key=lambda r: r["file"])
        counts = [collections.Counter(l["cls"] for l in r["labels"]) for r in ims]
        classes = sorted({c for k in counts for c in k} | {"bin"})
        median = {c: statistics.median(k.get(c, 0) for k in counts) for c in classes}
        os.makedirs(os.path.join(out, "by_rollout", roll), exist_ok=True)
        thumbs = []
        for r, cnt in zip(ims, counts):
            img = cv2.imread(os.path.join(a.labels, "images", r["split"], r["file"]))
            masks = [mask(l["poly"], r["size"]) for l in r["labels"]]
            dup = set()
            for i in range(len(masks)):
                for j in range(i + 1, len(masks)):
                    if r["labels"][i]["cls"] == r["labels"][j]["cls"]:
                        continue
                    inter = np.logical_and(masks[i], masks[j]).sum()
                    union = np.logical_or(masks[i], masks[j]).sum()
                    if union and inter / union > 0.5:
                        dup.add(j if r["labels"][j]["score"] < r["labels"][i]["score"] else i)
            low = [l for l in r["labels"] if l["score"] < GATE]
            odd = {c: cnt.get(c, 0) - median[c] for c in classes if cnt.get(c, 0) != median[c]}
            reasons = []
            if dup:
                reasons.append(f"{len(dup)} object(s) with two classes")
            if low:
                reasons.append(f"{len(low)} label(s) < {GATE}: " + ", ".join(sorted(f"{l['cls']} {l['score']:.2f}" for l in low)))
            if cnt.get("bin", 0) != 3:
                reasons.append(f"{cnt.get('bin', 0)} bins")
            if odd:
                reasons.append("unlike its rollout: " + ", ".join(f"{c} {v:+g}" for c, v in sorted(odd.items())))
            score = 3 * len(dup) + len(low) + 2 * abs(cnt.get("bin", 0) - 3) + sum(abs(v) for v in odd.values())
            for k, l in enumerate(r["labels"]):
                col = COLOURS.get(l["cls"], (255, 255, 255))
                pts = np.asarray(l["poly"], np.float32).reshape(-1, 2).astype(np.int32)
                cv2.polylines(img, [pts], True, col, 2)
                tag = f"{l['cls']} {l['score']:.2f}" + ("?" if l["score"] < GATE else "") + (" DUP" if k in dup else "")
                x0, y0 = pts.min(axis=0)
                cv2.putText(img, tag, (int(x0), max(14, int(y0) - 5)), 0, 0.6, (0, 0, 0), 4)
                cv2.putText(img, tag, (int(x0), max(14, int(y0) - 5)), 0, 0.6, col, 2)
            cv2.putText(img, f"{r['file']}  ({r['split']})  score {score:g}", (20, 40), 0, 1.0, (0, 0, 0), 5)
            cv2.putText(img, f"{r['file']}  ({r['split']})  score {score:g}", (20, 40), 0, 1.0, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(out, "by_rollout", roll, r["file"]), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            thumbs.append(cv2.resize(img[180:1080, 240:1760], (456, 270), interpolation=cv2.INTER_AREA))
            rows.append({"image": r["file"], "rollout": roll, "split": r["split"], "score": score,
                         "labels": len(r["labels"]), "reasons": "; ".join(reasons),
                         "preview": os.path.join("by_rollout", roll, r["file"])})
        cols = 5
        while len(thumbs) % cols:
            thumbs.append(np.full_like(thumbs[0], 255))
        sheet = np.vstack([np.hstack(thumbs[i:i + cols]) for i in range(0, len(thumbs), cols)])
        os.makedirs(os.path.join(out, "sheets"), exist_ok=True)
        cv2.imwrite(os.path.join(out, "sheets", f"{roll}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    rows.sort(key=lambda r: -r["score"])
    with open(os.path.join(out, "check_first.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "check_first.md"), "w") as f:
        f.write(f"# Images to check first ({len(rows)} in all; full list in check_first.csv)\n\n"
                "| # | Image | Split | Score | Why |\n|---|---|---|---|---|\n")
        for k, r in enumerate(rows[:a.top], 1):
            f.write(f"| {k} | [{r['image']}]({r['preview'].replace(os.sep, '/')}) | {r['split']} | {r['score']:g} | {r['reasons']} |\n")
    print(f"{len(rows)} images; top scores: " + ", ".join(f"{r['image']} {r['score']:g}" for r in rows[:5]))


if __name__ == "__main__":
    main()
