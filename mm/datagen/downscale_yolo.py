"""
A YOLO-seg dataset at a lower camera resolution, without re-rendering.

    .venv/bin/python -m mm.datagen.downscale_yolo --src data/yolo_v3 --dst data/yolo_v3_1080 --scale 0.5

The v3 images are ROI crops of 4K table frames; the real table camera gives at
most 1080p, where the same ROI is half the size. Each crop is resized by
--scale with area averaging (what a lower-resolution sensor integrates);
labels are normalised polygons and are copied unchanged.
"""

import argparse
import glob
import os
import shutil
from multiprocessing import Pool

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _one(job):
    src, dst, scale = job
    img = cv2.imread(src)
    h, w = img.shape[:2]
    small = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst, small)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "data", "yolo_v3"))
    ap.add_argument("--dst", default=os.path.join(ROOT, "data", "yolo_v3_1080"))
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    jobs = []
    for split in sorted(os.listdir(os.path.join(a.src, "images"))):
        os.makedirs(os.path.join(a.dst, "images", split), exist_ok=True)
        os.makedirs(os.path.join(a.dst, "labels", split), exist_ok=True)
        for p in glob.glob(os.path.join(a.src, "images", split, "*")):
            jobs.append((p, os.path.join(a.dst, "images", split, os.path.basename(p)), a.scale))
        for p in glob.glob(os.path.join(a.src, "labels", split, "*.txt")):
            shutil.copyfile(p, os.path.join(a.dst, "labels", split, os.path.basename(p)))
    for p in glob.glob(os.path.join(a.src, "index_*.json")):
        shutil.copyfile(p, os.path.join(a.dst, os.path.basename(p)))
    with Pool(a.workers) as pool:
        pool.map(_one, jobs, chunksize=16)
    print(f"[downscale_yolo] {len(jobs)} images x{a.scale} -> {a.dst}")


if __name__ == "__main__":
    main()
