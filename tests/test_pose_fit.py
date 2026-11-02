"""Host tests for mm/perception/pose_fit.py (step A7 refinement).

    .venv/bin/python -m unittest tests.test_pose_fit
"""

import unittest

import cv2
import numpy as np

import scene_config as C
from mm.datagen.masks_to_yolo import nominal_camera
from mm.eval.gt import yaw_error
from mm.perception import pose_fit as P


class PoseFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cam = nominal_camera()

    def mask_of(self, cls, xy, yaw, orientation=None):
        """Full-frame binary mask as a renderer would produce it."""
        return P.silhouette(cls, self.cam, xy, yaw, (self.cam.H, self.cam.W), orientation=orientation)

    def test_cube_xy_and_yaw(self):
        for xy, yaw in (((0.30, 0.05), 17.0), ((0.42, -0.10), -33.0), ((0.36, 0.22), 2.0)):
            r = P.refine("cube", np.zeros((self.cam.H, self.cam.W, 3), np.uint8), self.cam, self.mask_of("cube", xy, yaw))
            self.assertGreaterEqual(r["fit_iou"], P.FIT_IOU_MIN)
            self.assertLess(np.linalg.norm(np.subtract(r["xy"], xy)) * 1000, 0.3, xy)
            self.assertLess(yaw_error("cube", r["yaw_deg"], yaw), 3.0, (xy, yaw))

    def test_cap_orientation_and_xy(self):
        img = np.zeros((self.cam.H, self.cam.W, 3), np.uint8)
        for orient in ("narrow_end_up", "wide_end_up"):
            xy = (0.35, 0.10)
            r = P.refine("cap", img, self.cam, self.mask_of("cap", xy, 0.0, orient))
            self.assertEqual(r["orientation"], orient)
            self.assertLess(np.linalg.norm(np.subtract(r["xy"], xy)) * 1000, 0.3)

    def test_aaa_cell_polarity_from_band(self):
        xy, yaw = (0.33, 0.0), 120.0
        mask = self.mask_of("aaa_cell", xy, yaw)
        # Paint a saturated band near the + end, as the copper band renders.
        img = np.full((self.cam.H, self.cam.W, 3), 40, np.uint8)
        a = np.radians(yaw)
        plus = np.add(xy, 0.020 * np.array([np.cos(a), np.sin(a)]))
        u, v = self.cam.project([*plus, self.cam.table_z + C.AAA_CELL_RADIUS])
        cv2.circle(img, (int(u), int(v)), 12, (40, 110, 200), -1)
        r = P.refine("aaa_cell", img, self.cam, mask)
        self.assertGreaterEqual(r["fit_iou"], P.FIT_IOU_MIN)
        self.assertLess(abs((r["yaw_deg"] - yaw + 180) % 360 - 180), 3.0)
        self.assertLess(np.linalg.norm(np.subtract(r["xy"], xy)) * 1000, 0.3)

    def test_screw_xy_yaw_and_polarity(self):
        img = np.zeros((self.cam.H, self.cam.W, 3), np.uint8)
        for xy, yaw in (((0.34, 0.08), 35.0), ((0.40, -0.05), -150.0)):
            r = P.refine("screw", img, self.cam, self.mask_of("screw", xy, yaw))
            self.assertGreaterEqual(r["fit_iou"], P.FIT_IOU_MIN)
            self.assertLess(np.linalg.norm(np.subtract(r["xy"], xy)) * 1000, 0.3, xy)
            self.assertLess(abs((r["yaw_deg"] - yaw + 180) % 360 - 180), 3.0, (xy, yaw))

    def test_screw_silhouette_is_not_convex(self):
        m = self.mask_of("screw", (0.34, 0.08), 0.0)
        hull = cv2.convexHull(np.stack(np.nonzero(m)[::-1], 1).astype(np.int32))
        self.assertLess(m.sum() / cv2.contourArea(hull), 0.9)  # head + shank, not a taper

    def test_aaa_cell_standing_on_end_is_refused(self):
        img = np.zeros((self.cam.H, self.cam.W, 3), np.uint8)
        u, v = self.cam.project([0.35, 0.05, self.cam.table_z + C.AAA_CELL_LENGTH])
        mask = cv2.ellipse(np.zeros((self.cam.H, self.cam.W), np.uint8), (int(u), int(v)), (20, 26), 0, 0, 360, 1, -1)
        r = P.refine("aaa_cell", img, self.cam, mask)
        self.assertFalse(r["resting_flat"])
        self.assertEqual(r["fit_iou"], 0.0)

    def test_seed_outside_search_is_refused(self):
        mask = self.mask_of("cube", (0.35, 0.05), 10.0)
        _, _, fit = P.fit_silhouette("cube", self.cam, mask, np.array([0.35, 0.065]), yaw_range=(-45, 45))
        self.assertEqual(fit, 0.0)  # true pose is 15 mm away: boundary optimum

    def test_supersampled_silhouette_area_matches_binary(self):
        a = P.silhouette("cube", self.cam, (0.35, 0.05), 20.0, (self.cam.H, self.cam.W))
        win = P._window(a, 10)
        b = P.silhouette("cube", self.cam, (0.35, 0.05), 20.0, (win[3] - win[1], win[2] - win[0]), win[:2], ss=4)
        self.assertAlmostEqual(b.sum() / a.sum(), 1.0, delta=0.08)


if __name__ == "__main__":
    unittest.main()
