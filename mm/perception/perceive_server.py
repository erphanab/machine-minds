"""
Long-lived perception for the closed loop (Machine Minds B6-B8).

Runs in the YOLO environment and keeps the model loaded; the Isaac process
(sim_backend) sends one JSON line per capture on stdin and reads one JSON
line back, so a re-capture costs inference only (~2 s), not a process start.

    request : {"image": "<abs path .png>", "camera": "<abs path camera.json>"}
    response: {"objects": [...perceive_yolo detections...], "seconds": t}
              or {"error": "..."}

Model: MM_YOLO_RUN (models/<run>) or perceive_yolo.DEFAULT_RUN. A model whose
card sets "input_resolution" (the 1080p models: the real table camera) sees
the capture resized to it, with the camera scaled to match, as on the cell;
outlines (polygon_px) are returned in the capture's own pixels.
"""

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402

from backproject import CameraModel  # noqa: E402
from mm.datagen.masks_to_yolo import load_roi  # noqa: E402
from mm import paths  # noqa: E402
from mm.perception import perceive_yolo as PY  # noqa: E402


def main():
    from ultralytics import YOLO

    run = os.environ.get("MM_YOLO_RUN") or PY.DEFAULT_RUN
    weights = os.path.join(paths.model_dir(run), "weights", "best.pt")
    model, imgsz = YOLO(weights), PY.model_imgsz(weights)
    with open(os.path.join(paths.model_dir(run), "card.json")) as f:
        input_res = json.load(f).get("input_resolution")
    classes, roi = PY.load_classes(), load_roi()
    print(json.dumps({"ready": True, "model": PY.model_card(weights), "input_resolution": input_res}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            t = time.time()
            img = cv2.imread(req["image"])
            with open(req["camera"]) as f:
                cam_json = json.load(f)
            scale = 1.0
            if input_res and img.shape[1] > input_res[0]:
                scale = input_res[0] / img.shape[1]
                img = cv2.resize(img, tuple(input_res), interpolation=cv2.INTER_AREA)
                cam_json = dict(cam_json, resolution=list(input_res))
            dets = PY.perceive(model, img, CameraModel(cam_json), roi, classes, imgsz)
            for d in dets:  # back to the capture's pixels
                d["polygon_px"] = [[u / scale, v / scale] for u, v in d["polygon_px"]]
            print(json.dumps({"objects": dets, "seconds": round(time.time() - t, 2)}), flush=True)
        except Exception as e:  # keep serving; the caller decides
            print(json.dumps({"error": f"{type(e).__name__}: {e}"}), flush=True)


if __name__ == "__main__":
    main()
