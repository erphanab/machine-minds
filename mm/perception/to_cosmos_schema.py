"""
Rewrite YOLO detections in the schema the starter's plan_sort.py reads (B0).

The baseline run keeps the starter's planner and executor unchanged and only
swaps Cosmos for YOLO, since Cosmos3-Nano is not installed here. B3's own
planner reads perceive_yolo output directly; this bridge is for B0 only.

    .venv/bin/python -m mm.perception.to_cosmos_schema --run runs/b0 \
        --src detections_yolo.json --out detections.json

Mapping per object (unit image coordinates, as perceive.py writes them):
  center              projection of the refined pose at mid height
  bbox                bounds of the YOLO polygon
  top_edge_endpoints  two adjacent top-face corners of the refined cube pose
                      (plan_sort.edge_pose turns them back into a yaw)
  pose_confidence     the pose fit IoU when eligible, else 0 (plan_sort needs
                      >= 0.8, and the fit gate is 0.85)
AAA cells have no category in the starter planner and are left out.
"""

import argparse
import json
import math
import os

import numpy as np

import scene_config as C
from backproject import CameraModel
from mm.perception.pose_fit import MID_HEIGHT

CATEGORY = {"cube": "cube", "cap": "cap", "tube": "vial"}


def cube_top_edge(cam, xy, yaw_deg):
    t = math.radians(yaw_deg)
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    z = cam.table_z + C.CUBE_EDGE
    pts = [cam.project([*(np.asarray(xy) + rot @ np.array(c) * C.CUBE_EDGE / 2), z]) for c in ((-1, -1), (1, -1))]
    return [[float(p[0] / cam.W), float(p[1] / cam.H)] for p in pts]


def convert(det, cam):
    objects = []
    for d in det["objects"]:
        cat = CATEGORY.get(d["class"])
        if cat is None or d["xy_robot_m"] is None:
            continue
        u, v = cam.project([*d["xy_robot_m"], cam.table_z + MID_HEIGHT[d["class"]]])
        poly = np.asarray(d["polygon_px"])
        obj = {
            "label": f"{d.get('colour') or ''} {d['class']}".strip(),
            "category": cat,
            "center": [float(u / cam.W), float(v / cam.H)],
            "bbox": [
                float(poly[:, 0].min() / cam.W),
                float(poly[:, 1].min() / cam.H),
                float(poly[:, 0].max() / cam.W),
                float(poly[:, 1].max() / cam.H),
            ],
            "graspable": bool(d["eligible"]),
            "confidence": d["conf"],
            "yolo_id": d["id"],
        }
        if cat == "cube":
            obj["colour"] = d.get("colour") or "unknown"
            if d["yaw_deg"] is not None:
                obj["top_edge_endpoints"] = cube_top_edge(cam, d["xy_robot_m"], d["yaw_deg"])
            obj["pose_confidence"] = d["fit_iou"] if d["eligible"] else 0.0
        if cat == "cap":
            obj["colour"] = "yellow"
            obj["orientation"] = d.get("orientation", "unknown")
            obj["orientation_confidence"] = d["fit_iou"]
        objects.append(obj)
    return {
        "source": "perceive_yolo via to_cosmos_schema",
        "coordinate_convention": "unit",
        "model": det.get("model"),
        "objects": objects,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--src", default="detections_yolo.json")
    parser.add_argument("--out", default="detections.json")
    args = parser.parse_args()
    with open(os.path.join(args.run, args.src)) as f:
        det = json.load(f)
    cam = CameraModel(os.path.join(args.run, "camera.json"))
    out = convert(det, cam)
    with open(os.path.join(args.run, args.out), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[to_cosmos_schema] {len(out['objects'])} objects -> {os.path.join(args.run, args.out)}")


if __name__ == "__main__":
    main()
