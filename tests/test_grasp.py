"""Host tests for mm/planning/grasp.py and plan_bins.py (step B3).

    .venv/bin/python -m unittest tests.test_grasp
"""

import math
import unittest

import cv2
import numpy as np

import scene_config as C
from mm.datagen.masks_to_yolo import nominal_camera, project_object
from mm.perception.pose_fit import rest_pose
from mm.planning import grasp as G
from mm.planning import plan_bins as PB


def make_det(cam, ident, cls, xy, yaw=0.0, eligible=True, skip=None):
    """A perceive_yolo-style detection whose outline is the projected object."""
    if cls == "tube":
        a = math.radians(yaw)
        pos = np.array([xy[0] - C.VIAL_LENGTH / 2 * math.cos(a), xy[1] - C.VIAL_LENGTH / 2 * math.sin(a), cam.table_z + C.VIAL_RADIUS])
        from scipy.spatial.transform import Rotation

        rot = Rotation.from_euler("z", yaw, degrees=True) * Rotation.from_euler("y", 90, degrees=True)
    else:
        pos, rot = rest_pose(cls, xy, yaw or 0.0, cam.table_z)
    x, y, z, w = rot.as_quat()
    uv = project_object({"class": cls, "orientation": "narrow_end_up", "position": pos, "quat_wxyz": [w, x, y, z]}, cam)
    hull = cv2.convexHull(uv).reshape(-1, 2)
    return {
        "id": ident,
        "class": cls,
        "conf": 0.9,
        "polygon_px": hull.tolist(),
        "xy_robot_m": list(xy),
        "yaw_deg": yaw if cls in ("cube", "aaa_cell", "screw") else None,
        "fit_iou": 0.95,
        "eligible": eligible,
        "skip_reason": skip,
        "orientation": "narrow_end_up" if cls == "cap" else None,
    }


class GraspTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cam = nominal_camera()
        cls.reach = G.Reach()
        from mm.planning.bin_assign import default_table

        cls.bins = default_table()  # v4: class -> one of the identical bins

    def plan(self, dets):
        return PB.build_plan({"objects": dets}, self.cam, bins=self.bins, reach=self.reach)

    def test_tip_gap_and_width_rules(self):
        self.assertAlmostEqual(G.TIP_OPEN_GAP, 0.016, delta=0.0002)
        self.assertTrue(G.width_ok("cube"))  # 10 + 2 x 2.5 <= 16
        self.assertTrue(G.width_ok("aaa_cell"))  # scallop fit, 0.75 mm per side
        self.assertFalse(G.width_ok("cap"))  # 23.6 mm plug

    def test_finger_rects_straddle_the_object(self):
        rects = G.finger_rects((0.0, 0.0), (0.0, 1.0), margin=0.0)
        ys = sorted(abs(r[:, 1]).min() for r in rects)
        self.assertAlmostEqual(ys[0], G.TIP_OPEN_GAP / 2, places=6)
        for r in rects:
            self.assertAlmostEqual(np.ptp(r[:, 0]), C.FINGER_WIDTH, places=6)
            self.assertAlmostEqual(np.ptp(r[:, 1]), C.FINGER_THICK, places=6)

    def test_isolated_cube_and_aaa_cell_are_planned(self):
        dets = [make_det(self.cam, 0, "cube", (0.33, 0.05), 20.0), make_det(self.cam, 1, "aaa_cell", (0.40, 0.15), -30.0)]
        plan = self.plan(dets)
        self.assertEqual(len(plan["picks"]), 2, plan["skipped"])
        by_class = {p["category"]: p for p in plan["picks"]}  # order is B4's choice
        cube, cell = by_class["cube"], by_class["aaa_cell"]
        np.testing.assert_allclose(cube["spread_dir"][:2], [-math.sin(math.radians(20)), math.cos(math.radians(20))], atol=1e-9)
        np.testing.assert_allclose(cell["spread_dir"][:2], [-math.sin(math.radians(-30)), math.cos(math.radians(-30))], atol=1e-9)
        self.assertEqual(cube["place_xy"], list(self.bins["cube"]["xy"]))
        self.assertEqual(cell["place_xy"], list(self.bins["aaa_cell"]["xy"]))
        for p in plan["picks"]:  # everything sort_scene.segments() reads
            for key in ("xy", "pick_z", "place_xy", "place_z", "spread_dir", "place_spread_dir", "category"):
                self.assertIn(key, p)

    def test_screw_is_planned_across_its_shank(self):
        screw = make_det(self.cam, 0, "screw", (0.36, 0.10), 40.0)
        plan = self.plan([screw])
        self.assertEqual(len(plan["picks"]), 1, plan["skipped"])
        p = plan["picks"][0]
        self.assertEqual(p["bin"], "screw")
        self.assertAlmostEqual(p["width_m"], C.SCREW_SHANK_D)
        np.testing.assert_allclose(p["spread_dir"][:2], [-math.sin(math.radians(40)), math.cos(math.radians(40))], atol=1e-9)
        self.assertTrue(G.width_ok("screw"))

    def test_cube_falls_back_to_other_face_pair(self):
        cube = make_det(self.cam, 0, "cube", (0.33, 0.05), 0.0)
        # A cube 20 mm along +Y: inside the Y-closing finger zone (5-17 mm out),
        # outside the X-closing fingers' reach (+/-9 mm in Y).
        blocker = make_det(self.cam, 1, "cube", (0.33, 0.05 + 0.020), 0.0, eligible=False, skip="x")
        plan = self.plan([cube, blocker])
        self.assertEqual(len(plan["picks"]), 1, plan["skipped"])
        p = plan["picks"][0]
        np.testing.assert_allclose(np.abs(p["spread_dir"][:2]), [1.0, 0.0], atol=1e-9)
        self.assertTrue(p["alternatives_rejected"][0]["finger_collision"])

    def test_boxed_in_cube_is_skipped(self):
        cube = make_det(self.cam, 0, "cube", (0.33, 0.05), 0.0)
        ring = [
            make_det(self.cam, k + 1, "cap", (0.33 + dx, 0.05 + dy), eligible=False, skip="x")
            for k, (dx, dy) in enumerate(((0.024, 0), (-0.024, 0), (0, 0.024), (0, -0.024)))
        ]
        plan = self.plan([cube] + ring)
        self.assertEqual(plan["picks"], [])
        cube_skip = next(s for s in plan["skipped"] if s["detection_id"] == 0)
        self.assertEqual(cube_skip["skip_reason"], "finger_collision")

    def test_unreachable_and_ungraspable_are_skipped(self):
        far = make_det(self.cam, 0, "cube", (0.70, 0.0), 0.0)
        cap = make_det(self.cam, 1, "cap", (0.30, -0.05))
        tube = make_det(self.cam, 2, "tube", (0.42, 0.10), 30.0, eligible=False, skip="distractor")
        plan = self.plan([far, cap, tube])
        reasons = {s["detection_id"]: s["skip_reason"] for s in plan["skipped"]}
        # Caps are distractors now (no bin); were one marked eligible, the planner
        # still refuses it. The width rule itself is checked in test_tip_gap.
        self.assertEqual(reasons, {0: "ik_unreachable", 1: "no_bin_for_class", 2: "distractor"})

    def test_bin_walls_block_fingers(self):
        b = self.bins["cube"]
        near = make_det(self.cam, 0, "cube", (b["xy"][0], b["xy"][1] + b["outer"][1] / 2 + 0.008), 0.0)
        plan = self.plan([near])
        skip = plan["skipped"][0] if plan["skipped"] else None
        if plan["picks"]:  # allowed only if the chosen fingers close along X, away from the wall
            np.testing.assert_allclose(np.abs(plan["picks"][0]["spread_dir"][:2]), [1.0, 0.0], atol=1e-9)
        else:
            self.assertEqual(skip["skip_reason"], "finger_collision")


if __name__ == "__main__":
    unittest.main()
