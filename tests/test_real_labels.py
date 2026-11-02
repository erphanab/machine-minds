"""Host tests for the real-image fine-tune set: labels cropped to the workspace ROI.

    .venv/bin/python -m unittest tests.test_real_labels
"""

import unittest

import numpy as np

from mm.datagen.build_real_finetune import crop_labels


def line(cls, pts, size):
    return f"{cls} " + " ".join(f"{x / size[0]:.6f} {y / size[1]:.6f}" for x, y in pts)


class CropLabelTests(unittest.TestCase):
    size, box = (1920, 1080), (344, 128, 1280, 912)

    def test_inside_is_shifted_and_normalized_to_the_crop(self):
        sq = [(500, 300), (540, 300), (540, 340), (500, 340)]
        (out,) = crop_labels([line(0, sq, self.size)], self.size, self.box)
        v = out.split()
        self.assertEqual(v[0], "0")
        pts = np.asarray(v[1:], float).reshape(-1, 2) * [1280, 912]
        np.testing.assert_allclose(pts.min(axis=0), [500 - 344, 300 - 128], atol=1.0)
        np.testing.assert_allclose(pts.max(axis=0), [540 - 344, 340 - 128], atol=1.0)

    def test_straddling_is_clipped_outside_is_dropped(self):
        edge = [(320, 400), (380, 400), (380, 440), (320, 440)]  # crosses x0 = 344
        away = [(100, 50), (140, 50), (140, 90), (100, 90)]  # left of and above the ROI
        out = crop_labels([line(5, edge, self.size), line(2, away, self.size)], self.size, self.box)
        self.assertEqual(len(out), 1)
        pts = np.asarray(out[0].split()[1:], float).reshape(-1, 2) * [1280, 912]
        self.assertGreaterEqual(pts[:, 0].min(), 0.0)
        self.assertAlmostEqual(pts[:, 0].max(), 380 - 344, delta=1.0)


if __name__ == "__main__":
    unittest.main()
