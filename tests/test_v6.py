"""Host tests for the v6 changes: outline grasp (cubes, screws), wrist grasp check, scene checks, bins re-found every
capture, the five-move pick and the clear-view retreat.

    .venv/bin/python -m unittest tests.test_v6
"""

import math
import os
import unittest

import cv2
import numpy as np

import scene_config as C
from mm.control import executor as X
from mm.perception import bins as B
from mm.perception import outline_grasp as OG
from mm.perception import perceive_yolo as PY
from mm.perception import pose_fit as P
from mm.planning import grasp as G
from mm.real import brain as BR
from mm.real import frames as F
from mm.real import scene_check as SC
from mm.real import wrist_check as WC

CLIENT_CAMS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "PhysicalAI-Hackathon-2026-v6", "calibration", "cameras.json")


def yaw90(a, b):
    return abs((a - b + 45.0) % 90.0 - 45.0)


@unittest.skipUnless(os.path.exists(CLIENT_CAMS), "needs the v6 client's calibration")
class OutlineGraspTests(unittest.TestCase):
    """On the real table camera's calibration at 1080p, where a 10 mm cube is ~20 px."""

    @classmethod
    def setUpClass(cls):
        cls.cam = F.CellCamera.load(CLIENT_CAMS, "table", (1920, 1080))

    def mask(self, cls, xy, yaw):
        return P.silhouette(cls, self.cam, xy, yaw, (self.cam.H, self.cam.W))

    def test_cube_centre_and_square_edges(self):
        for xy, yaw in (((0.35, 0.05), 20.0), ((0.42, -0.10), -35.0), ((0.30, 0.15), 5.0)):
            g = OG.cube_grasp(self.cam, self.mask("cube", xy, yaw))
            self.assertLess(math.dist(g["xy"], xy) * 1000, 1.5, (xy, yaw, g))
            self.assertLess(yaw90(g["yaw_deg"], yaw), 8.0, (xy, yaw, g))

    def test_cube_one_pixel_loose_outline_still_grasped_near_centre(self):
        m = cv2.dilate(self.mask("cube", (0.38, 0.0), 30.0), np.ones((3, 3), np.uint8))
        g = OG.cube_grasp(self.cam, m)
        self.assertLess(math.dist(g["xy"], (0.38, 0.0)) * 1000, 1.5)
        self.assertLess(yaw90(g["yaw_deg"], 30.0), 8.0)

    def test_screw_grip_on_the_shank_middle_not_the_head(self):
        for xy, yaw in (((0.35, 0.05), 30.0), ((0.40, -0.08), -120.0), ((0.32, 0.12), 170.0)):
            g = OG.screw_grasp(self.cam, self.mask("screw", xy, yaw))
            self.assertNotEqual(g["head_end"], "unclear", (xy, yaw, g))
            d = abs((g["yaw_deg"] - yaw + 180.0) % 360.0 - 180.0)
            self.assertLess(d, 10.0, (xy, yaw, g))  # head -> tip, the same way round
            self.assertLess(math.dist(g["xy"], xy) * 1000, 3.0, (xy, yaw, g))  # xy: the shank middle
            head = C.screw_head_xy(xy, yaw)
            self.assertGreater(math.dist(g["xy"], head) * 1000, 12.0)  # well clear of the head

    def test_finalize_outline_has_no_fit_gate_but_aaa_cells_are_unchanged(self):
        img = np.full((1080, 1920, 3), 235, np.uint8)
        cube = cv2.dilate(self.mask("cube", (0.36, 0.02), 10.0), np.ones((5, 5), np.uint8))  # a loose outline
        aaa = self.mask("aaa_cell", (0.34, -0.06), 40.0)
        img[cube.astype(bool)] = (40, 40, 200)
        img[aaa.astype(bool)] = (30, 30, 30)

        def raw(cls, m):
            ys, xs = np.nonzero(m)
            return {"class": cls, "conf": 0.8, "polygon_px": [[float(x), float(y)] for x, y in zip(xs[:3], ys[:3])],
                    "window": (int(xs.min()), int(ys.min())), "mask": m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(bool)}

        insts = [raw("cube", cube), raw("aaa_cell", aaa)]
        classes = PY.load_classes()
        v6 = PY.finalize(insts, img, self.cam, classes, grasp_mode="outline")
        v5 = PY.finalize(insts, img, self.cam, classes, grasp_mode="pose_fit")
        self.assertTrue(v6[0]["eligible"])
        self.assertEqual(v6[0]["grasp"], "outline")
        self.assertIsNone(v6[0]["fit_iou"])
        self.assertEqual(v5[0]["skip_reason"], "pose_fit_below_gate")
        self.assertEqual({k: v6[1][k] for k in ("xy_robot_m", "yaw_deg", "fit_iou", "eligible", "skip_reason")},
                         {k: v5[1][k] for k in ("xy_robot_m", "yaw_deg", "fit_iou", "eligible", "skip_reason")})

    def test_touching_a_bin(self):
        b = dict(B.load_bin_design(), xy=[0.40, 0.10], yaw_deg=0.0)
        half = b["outer"][0] / 2
        poly = lambda xy: [list(self.cam.project([xy[0] + dx, xy[1] + dy, self.cam.table_z + 0.005]))
                           for dx, dy in ((-0.005, -0.005), (0.005, -0.005), (0.005, 0.005), (-0.005, 0.005))]
        near = {"class": "cube", "polygon_px": poly((0.40 + half + 0.004, 0.10))}
        far = {"class": "cube", "polygon_px": poly((0.40 + half + 0.03, 0.10))}
        self.assertEqual(G.touches_bin(near, self.cam, {"bin_1": b}), "bin_1")
        self.assertIsNone(G.touches_bin(far, self.cam, {"bin_1": b}))


class WristCheckTests(unittest.TestCase):
    def setUp(self):
        self.table = np.full((1080, 1920, 3), 230, np.uint8)
        self.with_cube = self.table.copy()
        cv2.rectangle(self.with_cube, (1010, 855), (1035, 880), (30, 30, 200), -1)
        self.win = (1022, 867, 40)

    def test_still_there_is_a_miss(self):
        shifted = self.table.copy()
        cv2.rectangle(shifted, (1016, 859), (1041, 884), (30, 30, 200), -1)  # the arm stopped ~6 px off
        self.assertEqual(WC.verdict(self.with_cube, shifted, self.win)[0], "missed")

    def test_gone_is_a_pick(self):
        self.assertEqual(WC.verdict(self.with_cube, self.table, self.win)[0], "picked")

    def test_nothing_to_compare_is_unclear(self):
        self.assertEqual(WC.verdict(self.table, self.table, self.win)[0], "unclear")
        self.assertEqual(WC.verdict(self.with_cube, None, self.win)[0], "unclear")

    @unittest.skipUnless(os.path.exists(CLIENT_CAMS), "needs the v6 client's calibration")
    def test_spot_below_the_hover_pose_is_in_the_wrist_image(self):
        import json

        reach = G.Reach()
        with open(CLIENT_CAMS) as f:
            real = json.load(f)["wrist"]
        for cal in (real, F.sim_wrist_calibration()):
            cam = F.WristCamera(cal, (1920, 1080))
            _, tip = __import__("fk").tool_pose(reach.home, reach.tool)
            win = WC.window(cam, reach.home, tip - [0.0, 0.0, X.HOVER_DZ], "cube")
            self.assertIsNotNone(win)
            self.assertLess(abs(win[0] - 1022), 15)
            self.assertLess(abs(win[1] - 867), 15)


class SceneTests(unittest.TestCase):
    def test_stable(self):
        a = [("cube", (0.30, 0.10)), ("bin", (0.40, -0.20))]
        self.assertTrue(SC.stable(a, [("bin", (0.402, -0.199)), ("cube", (0.305, 0.10))]))
        self.assertFalse(SC.stable(a, [("cube", (0.30, 0.10))]))
        self.assertFalse(SC.stable(a, [("cube", (0.33, 0.10)), ("bin", (0.40, -0.20))]))
        self.assertFalse(SC.stable(None, a))

    def test_foreign_finds_a_hand_not_known_objects(self):
        img = np.full((1080, 1920, 3), 200, np.uint8)
        roi = (344, 128, 1280, 912)
        cv2.rectangle(img, (900, 600), (1100, 800), (90, 110, 170), -1)  # skin-toned, 40000 px
        self.assertIsNotNone(SC.foreign(img, [], roi))
        known = [{"polygon_px": [[900, 600], [1100, 600], [1100, 800], [900, 800]]}]
        self.assertIsNone(SC.foreign(img, known, roi))
        small = np.full((1080, 1920, 3), 200, np.uint8)
        cv2.rectangle(small, (900, 600), (930, 630), (20, 20, 20), -1)  # an unlisted cube-sized thing
        self.assertIsNone(SC.foreign(small, [], roi))


class BinsTests(unittest.TestCase):
    def test_bins_refound_at_any_distance(self):
        prev = {"bin_1": dict(B.load_bin_design(), xy=[0.30, 0.20], yaw_deg=0.0, found=True)}
        det = [{"class": "bin", "xy_robot_m": [0.30, 0.05], "yaw_deg": 5.0, "fit_iou": 0.95, "conf": 0.9}]
        self.assertFalse(B.track_bins(det, prev)["bin_1"]["found"])  # v5: beyond TRACK_RADIUS
        moved = B.track_bins(det, prev, radius=math.inf)["bin_1"]
        self.assertTrue(moved["found"])
        self.assertEqual(moved["xy"], [0.30, 0.05])

    def test_unseen_or_unreachable_bin_is_not_used(self):
        b = dict(B.load_bin_design(), xy=[0.40, 0.10], yaw_deg=0.0)
        self.assertIsNone(BR.Brain._bin_unusable(dict(b, found=True)))
        self.assertEqual(BR.Brain._bin_unusable(dict(b, found=False)), "bin_not_seen_now")
        self.assertEqual(BR.Brain._bin_unusable(dict(b, xy=[0.80, 0.10], found=True)), "bin_outside_tcp_box")


class SymmetricGraspTests(unittest.TestCase):
    def test_every_symmetric_angle_after_the_v5_ones(self):
        mod = lambda a: round(a % 360.0, 6)
        cube = [mod(g) for _, g in G.spread_options({"class": "cube", "yaw_deg": 10.0})]
        self.assertEqual(cube[:2], [100.0, 10.0])  # v5's two first: a grasp that worked before is unchanged
        self.assertEqual(sorted(cube), sorted(mod(10.0 + k) for k in (0, 90, 180, 270)))
        for cls in ("screw", "aaa_cell"):
            self.assertEqual([mod(g) for _, g in G.spread_options({"class": cls, "yaw_deg": 10.0})], [100.0, 280.0])
        v = [s for s, _ in G.spread_options({"class": "screw", "yaw_deg": 10.0})]
        np.testing.assert_allclose(v[1], -v[0], atol=1e-12)  # the same jaw line, the wrist turned 180 deg


class MovesTests(unittest.TestCase):
    def test_five_moves_meet_end_to_end(self):
        pick = {"xy": [0.35, 0.05], "pick_z": C.SURFACE_DZ + 0.004, "place_xy": [0.40, -0.20], "place_z": C.SURFACE_DZ + 0.05,
                "spread_dir": [0.0, 1.0, 0.0], "place_spread_dir": [0.0, 1.0, 0.0], "category": "cube"}
        segs = X.pick_segments(pick, np.array([0.3, 0.0, 0.24]))
        segs.append(("park", segs[-1][2], np.array(BR.RETREAT_XYZ), 2.0, C.JAW_OPEN, C.JAW_OPEN, np.array([0.0, 1.0, 0.0])))
        moves, shut = BR.split_pick_wrist(segs, "cube", 0.0)
        self.assertEqual(len(moves), 5)
        up = [0.35, 0.05, pick["pick_z"] + X.HOVER_DZ]
        np.testing.assert_allclose(F.uf0_to_base(moves[0][-1][:3]), up, atol=1e-6)  # hover, before the descent
        np.testing.assert_allclose(F.uf0_to_base(moves[2][-1][:3]), up, atol=1e-6)  # hover, after the lift
        self.assertEqual(moves[1][-1][6], 0.0)  # open at the grasp pose
        self.assertEqual(moves[2][-1][6], 1.0)  # closed after the lift
        np.testing.assert_allclose(F.uf0_to_base(moves[4][-1][:3]), BR.RETREAT_XYZ, atol=1e-6)

    def test_clear_pose_on_the_bin_side(self):
        brain = BR.Brain.__new__(BR.Brain)
        brain.park = np.asarray(BR.PARK_XYZ, float)
        np.testing.assert_allclose(brain._clear_pose([0.4, -0.2]), BR.RETREAT_XYZ)
        np.testing.assert_allclose(brain._clear_pose([0.4, 0.2]), BR.PARK_XYZ)
        self.assertTrue(F.in_tcp_box(F.base_to_uf0(BR.RETREAT_XYZ) + [0, 0, 0, 0]))


if __name__ == "__main__":
    unittest.main()
