"""
Wrist-camera frames from a downloaded cell recording (uv run run.py --download -> LeRobot v3 dataset), one folder per
rollout, for the wrist grasp check (v6 step 4). Needs an OpenCV that decodes AV1 (the YOLO environment's does).

    C:\\isim\\yolo\\python.exe -m mm.datagen.wrist_frames --dataset %USERPROFILE%\\.cache\\huggingface\\lerobot\\manipbench\\r20261028-716eb3 \\
        --rollout-offset 2 --out data/wrist_20261028

Episode k is videos/observation.images.wrist/chunk-000/file-00k.mp4 (one file per episode in this recording; checked
against manipbench.json's trial ids). Writes <out>/rollout_<k + offset>/frame_<i>.jpg (i = frame index at the
dataset's fps, so t = i / fps from the episode start), plus sheet.jpg (one frame per 5 s) and index.json.
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--camera", default="wrist")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rollout-offset", type=int, default=2, help="our brain logs number rollouts from 02 on 28 Oct")
    ap.add_argument("--every", type=int, default=1, help="keep every n-th frame")
    a = ap.parse_args()
    info = json.load(open(os.path.join(a.dataset, "meta", "info.json")))
    trials = json.load(open(os.path.join(a.dataset, "manipbench.json")))
    files = sorted(glob.glob(os.path.join(a.dataset, "videos", f"observation.images.{a.camera}", "chunk-000", "file-*.mp4")))
    if len(files) != len(trials):
        raise SystemExit(f"{len(files)} video files for {len(trials)} episodes: not one file per episode")
    index = {"dataset": os.path.basename(a.dataset.rstrip("/\\")), "camera": a.camera, "fps": info["fps"], "rollouts": {}}
    for k, (path, tr) in enumerate(zip(files, trials)):
        roll = f"rollout_{k + a.rollout_offset:02d}"
        d = os.path.join(a.out, roll)
        os.makedirs(d, exist_ok=True)
        cap = cv2.VideoCapture(path)
        i, kept, thumbs = 0, 0, []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % a.every == 0:
                cv2.imwrite(os.path.join(d, f"frame_{i:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                kept += 1
            if i % (5 * info["fps"]) == 0:
                t = frame.copy()
                cv2.putText(t, f"{roll} {i / info['fps']:.0f}s", (8, 28), 0, 0.9, (0, 0, 0), 4)
                cv2.putText(t, f"{roll} {i / info['fps']:.0f}s", (8, 28), 0, 0.9, (255, 255, 255), 2)
                thumbs.append(cv2.resize(t, (320, 180)))
            i += 1
        cols = 8
        while len(thumbs) % cols:
            thumbs.append(np.full((180, 320, 3), 255, np.uint8))
        cv2.imwrite(os.path.join(d, "sheet.jpg"), np.vstack([np.hstack(thumbs[r:r + cols]) for r in range(0, len(thumbs), cols)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
        index["rollouts"][roll] = {"episode_index": tr["episode_index"], "trial_id": tr["trial_id"], "end_reason": tr.get("end_reason"),
                                   "frames": i, "kept": kept, "seconds": round(i / info["fps"], 1), "video": os.path.relpath(path, a.dataset)}
        print(roll, tr["trial_id"], tr.get("end_reason"), i, "frames", flush=True)
    with open(os.path.join(a.out, "index.json"), "w") as f:
        json.dump(index, f, indent=1)


if __name__ == "__main__":
    main()
