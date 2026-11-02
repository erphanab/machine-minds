"""
Bin detection and pose at 1080p, overall and in the hard spots for yellow bins (v4). Host-side.

    .venv/bin/python -m mm.eval.bin_check --run yolo_seg_v3_1080 --src data/synth_v4 --tag v4_1080p

Reads models/<run>/perception_<split>_<tag>.json (eval_perception.py --scale 0.5 --tag ...) and the frames'
meta. Groups every countable GT bin:
  near_arm   the arm is in view in that frame, or the bin centre is within 0.30 m of the robot base axis
             (the yellow arm and its grey base column)
  near_tape  the bin's outer edge within 40 mm of the yellow caution tape on the table edges
Reports recall (any class, IoU >= 0.5, predicted as bin) and pose error (centroid mm, yaw deg) for bins
whose fit passes bins.MIN_FIT, as B8 uses them.
"""

import argparse
import json
import math
import os

import numpy as np

import scene_config as C
from mm.datagen import randomize as R
from mm.perception.bins import MIN_FIT

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tape_gap(xy):
    """Distance from a bin's outer edge (circumscribed) to the inner edge of the yellow tape."""
    rad = R._bin_radius()
    x, y = xy
    edges = [(C.TABLE_X[1] - C.EDGE_BAND_W) - x, y - (C.TABLE_Y[0] + C.EDGE_BAND_W), (C.TABLE_Y[1] - C.EDGE_BAND_W) - y]
    return min(edges) - rad


def stats(rows):
    gt = [r for r in rows]
    hit = [r for r in gt if r.get("pred_class") == "bin"]
    used = [r for r in hit if r.get("fit_iou", 0) >= MIN_FIT and "centroid_mm" in r]
    c = np.array([r["centroid_mm"] for r in used]) if used else np.array([])
    y = np.array([r["yaw_err_deg"] for r in used if r.get("yaw_err_deg") is not None])
    return {
        "bins": len(gt), "recall": round(len(hit) / len(gt), 4) if gt else None,
        "fit_ok": len(used),
        "centroid_mm": {"median": round(float(np.median(c)), 2), "p95": round(float(np.percentile(c, 95)), 2),
                        "max": round(float(c.max()), 2), "over_2mm": int((c > 2.0).sum())} if len(c) else None,
        "yaw_deg": {"p95": round(float(np.percentile(y, 95)), 2), "max": round(float(y.max()), 2)} if len(y) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="yolo_seg_v3_1080")
    ap.add_argument("--src", default=os.path.join(ROOT, "data", "synth_v4"))
    ap.add_argument("--tag", default="v4_1080p")
    ap.add_argument("--splits", nargs="+", default=["test", "hard"])
    a = ap.parse_args()
    out = {}
    for split in a.splits:
        rows = json.load(open(os.path.join(ROOT, "models", a.run, f"perception_{split}_{a.tag}.json")))["objects"]
        bins = [r for r in rows if r.get("gt_class") == "bin" and r.get("countable")]
        arm_in_view = {}
        for r in bins:
            if r["frame"] not in arm_in_view:
                meta = json.load(open(os.path.join(a.src, split, "meta", r["frame"] + ".json")))
                arm_in_view[r["frame"]] = meta.get("layout", {}).get("arm", {}).get("mode") == "in_view"
        groups = {
            "all": bins,
            "near_arm": [r for r in bins if arm_in_view[r["frame"]] or math.hypot(*r["gt_xy"]) < 0.30],
            "near_tape": [r for r in bins if tape_gap(r["gt_xy"]) < 0.04],
        }
        out[split] = {g: stats(v) for g, v in groups.items()}
        for g, s in out[split].items():
            print(f"{split:5s} {g:9s} {s}")
    path = os.path.join(ROOT, "models", a.run, f"bin_check_{a.tag}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("->", path)


if __name__ == "__main__":
    main()
