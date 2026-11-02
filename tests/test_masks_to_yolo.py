"""Host tests for mm/datagen/masks_to_yolo.py (step A4).

    .venv/bin/python -m unittest tests.test_masks_to_yolo
"""

import unittest

import cv2
import numpy as np

import scene_config as C
from mm.datagen import masks_to_yolo as M
from mm.datagen import randomize as R


def iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return (a & b).sum() / max(1, (a | b).sum())


def rotated_rect(shape, centre, size, angle):
    m = np.zeros(shape, np.uint8)
    box = cv2.boxPoints((centre, size, angle)).astype(np.int32)
    cv2.fillPoly(m, [box], 1)
    return m


class PolygonTests(unittest.TestCase):
    # Sizes as rendered at 4K: cube ~26 px, cap ~70 px, AAA cell ~85 x 21 px.
    SHAPES = {
        "cube": lambda s: rotated_rect(s, (60, 60), (26, 26), 27),
        "cap": lambda s: cv2.circle(np.zeros(s, np.uint8), (80, 80), 35, 1, -1),
        "aaa_cell": lambda s: rotated_rect(s, (90, 70), (95, 28), -38),
        "tube": lambda s: rotated_rect(s, (100, 100), (260, 32), 12),
    }

    def test_round_trip_iou(self):
        shape = (200, 220)
        for name, make in self.SHAPES.items():
            mask = make(shape)
            poly = M.mask_to_polygon(mask)
            self.assertIsNotNone(poly, name)
            self.assertGreaterEqual(iou(mask, M.polygon_to_mask(poly, shape)), 0.9, name)

    def test_occluded_nonconvex_round_trip(self):
        shape = (200, 220)
        mask = self.SHAPES["aaa_cell"](shape)
        mask[60:80, 70:200] = 0  # arm crossing the cell
        mask = mask * (cv2.connectedComponents(mask)[1] == 1)  # keep one side
        poly = M.mask_to_polygon(mask)
        self.assertGreaterEqual(iou(mask, M.polygon_to_mask(poly, shape)), 0.9)

    def test_largest_component_wins(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[10:20, 10:20] = 1
        mask[40:80, 40:80] = 1
        poly = M.mask_to_polygon(mask)
        self.assertTrue(np.all(poly >= 39) and np.all(poly <= 81))

    def test_label_line_format(self):
        poly = M.mask_to_polygon(self.SHAPES["cap"]((200, 220)))
        parts = M.label_line(1, poly, 220, 200).split()
        self.assertEqual(parts[0], "1")
        coords = np.array(parts[1:], float)
        self.assertEqual(len(coords) % 2, 0)
        self.assertGreaterEqual(len(coords), 6)
        self.assertTrue(np.all((coords >= 0) & (coords <= 1)))


class VisibilityTests(unittest.TestCase):
    def setUp(self):
        self.cam = M.nominal_camera()
        self.z = C.TABLE_TOP_Z + C.SURFACE_DZ

    def obj(self, cls, xy, yaw, z_off=0.0, orientation=None):
        from scipy.spatial.transform import Rotation

        rot = Rotation.from_euler("z", yaw, degrees=True)
        if cls in ("aaa_cell", "tube"):
            rot = rot * Rotation.from_euler("y", 90, degrees=True)
        if cls == "screw":
            rot = rot * Rotation.from_euler("y", -90 + C.SCREW_REST_TILT_DEG, degrees=True)
        x, y, zq, w = rot.as_quat()
        return {
            "class": cls,
            "orientation": orientation,
            "position": [xy[0], xy[1], self.z + z_off],
            "quat_wxyz": [w, x, y, zq],
        }

    def rendered(self, obj):
        from scipy.spatial.transform import Rotation

        w, x, y, z = obj["quat_wxyz"]
        rot = Rotation.from_quat([x, y, z, w]).as_matrix()
        mask = np.zeros((self.cam.H, self.cam.W), np.uint8)
        for part in M.object_parts(obj["class"], obj.get("orientation")):
            uv = np.array([self.cam.project(p) for p in part @ rot.T + obj["position"]], np.float32)
            cv2.fillPoly(mask, [np.round(cv2.convexHull(uv) - 0.5).astype(np.int32)], 1)
        return mask

    def test_full_object_is_fully_visible(self):
        for cls, z_off in (
            ("cube", 0.0),
            ("cap", 0.0),
            ("aaa_cell", C.AAA_CELL_RADIUS),
            ("screw", C.SCREW_HEAD_D / 2),
        ):
            o = self.obj(cls, (0.35, 0.05), 20.0, z_off, "narrow_end_up")
            frac = self.rendered(o).sum() / M.amodal_area(o, self.cam)
            self.assertTrue(0.9 <= frac <= 1.15, (cls, frac))

    def test_half_hidden_object(self):
        o = self.obj("aaa_cell", (0.35, 0.05), 0.0, C.AAA_CELL_RADIUS)
        mask = self.rendered(o)
        ys, xs = np.nonzero(mask)
        mask[:, : int(np.median(xs))] = 0
        frac = mask.sum() / M.amodal_area(o, self.cam)
        self.assertTrue(0.4 <= frac <= 0.6, frac)

    def test_cube_size_at_4k(self):
        o = self.obj("cube", (0.35, 0.05), 0.0)
        self.assertTrue(400 <= M.amodal_area(o, self.cam) <= 1500)


class RoiTests(unittest.TestCase):
    def test_roi_covers_work_area_and_bins(self):
        cam = M.nominal_camera()
        roi = M.compute_roi(cam)
        x0, y0, w, h = M.roi_for(roi, cam.W, cam.H)
        self.assertEqual(w % M.ROI_ALIGN, 0)
        for p in M.roi_world_points():
            u, v = cam.project(p)
            self.assertTrue(x0 <= u <= x0 + w and y0 <= v <= y0 + h, p)

    def test_roi_scales_with_resolution(self):
        roi = {"resolution": [3840, 2160], "x0": 1000, "y0": 500, "width": 1600, "height": 1200}
        self.assertEqual(M.roi_for(roi, 1920, 1080), (500, 250, 800, 600))

    def test_target_area_corners_project_inside_image(self):
        cam = M.nominal_camera()
        z = C.TABLE_TOP_Z + C.SURFACE_DZ
        for x in R.TARGET_X:
            for y in R.TARGET_Y:
                u, v = cam.project((x, y, z))
                self.assertTrue(0 <= u < cam.W and 0 <= v < cam.H)


if __name__ == "__main__":
    unittest.main()
