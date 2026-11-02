"""YOLO-env test for mm/perception/perceive_yolo.py (step A7).

On 3 saved sim frames (the guide's check): every countable object is found
with the right class, and every upright target is located within 1 mm.
Eligibility is not required: the 0.85 fit gate deliberately refuses a few
correctly located but thin-masked screws (eval: fits of 0.80-0.85 included a
4 mm error on the hard split), so a refused object is fine, a wrong one is not.

    $YOLO_PY -m unittest tests.test_perceive_yolo

Skips when Ultralytics, the trained weights or the generated frames are absent
(weights and data are git-ignored).
"""

import json
import os
import unittest

import numpy as np

from mm.perception import perceive_yolo as PY

ROOT = PY.ROOT
WEIGHTS = os.path.join(ROOT, "models", PY.DEFAULT_RUN, "weights", "best.pt")
SPLIT_DIR = os.path.join(ROOT, "data", "synth_v3", "test")  # matches the v3 model
FRAMES = ("test_200000", "test_200001", "test_200002")

try:
    import ultralytics  # noqa: F401

    HAVE_YOLO = True
except ImportError:
    HAVE_YOLO = False


@unittest.skipUnless(HAVE_YOLO, "Ultralytics not installed (run in the YOLO env)")
@unittest.skipUnless(os.path.exists(WEIGHTS), f"no trained weights at {WEIGHTS}")
@unittest.skipUnless(os.path.isdir(SPLIT_DIR), "no generated test frames")
class PerceiveYoloTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ultralytics import YOLO

        cls.model = YOLO(WEIGHTS)
        cls.classes = PY.load_classes()
        cls.roi = PY.load_roi()

    def test_saved_frames(self):
        from mm.eval.eval_perception import score_frame

        for stem in FRAMES:
            meta_path = os.path.join(SPLIT_DIR, "meta", stem + ".json")
            with open(meta_path) as f:
                meta = json.load(f)
            import cv2
            from backproject import CameraModel

            img = cv2.imread(os.path.join(SPLIT_DIR, meta["image"]))
            raw = PY.detect(self.model, img, CameraModel(meta["camera"]), self.roi)
            rows = score_frame((meta_path, SPLIT_DIR, raw, self.classes, PY.CONF_MIN))
            for r in rows:
                if r.get("gt_class") and r.get("countable"):
                    self.assertEqual(r.get("pred_class"), r["gt_class"], (stem, r))
                    fitted = r.get("fit_iou", 0) > 0  # refused before fitting: no pose to judge
                    if not r["dropped"] and fitted and self.classes[r["gt_class"]]["role"] == "target":
                        self.assertLessEqual(r["centroid_mm"], 1.0, (stem, r))
                        if r.get("polarity_ok") is False:
                            self.fail(("head/tip or polarity wrong", stem, r))
                if r.get("eligible") and r.get("gt_class"):
                    self.assertLessEqual(r.get("centroid_mm", 99), 1.0, (stem, r))
            spurious = [r for r in rows if r.get("gt_class") is None and r["conf"] >= PY.CONF_MIN]
            self.assertFalse(spurious, (stem, spurious))


if __name__ == "__main__":
    unittest.main()
