"""Host tests for the v3 bins: STL geometry, random placement, rotated checks.

    .venv/bin/python -m unittest tests.test_sort_bin
"""

import collections
import math
import os
import struct
import tempfile
import unittest

import cv2
import numpy as np

import scene_config as C
from mm.cad import sort_bin as S
from mm.control import executor as X
from mm.datagen import randomize as R
from mm.datagen.masks_to_yolo import nominal_camera
from mm.perception import pose_fit as P


class StlTests(unittest.TestCase):
    def setUp(self):
        d = R.load_bin_design()
        self.d = d
        self.pts, self.faces = S.bin_mesh(d["outer"], d["pocket"], d["pocket_r"], d["wall_h"], d["floor_t"])

    def test_watertight_and_consistently_wound(self):
        edges = collections.Counter()
        for a, b, c in S.triangles(self.pts, self.faces):
            for u, v in ((a, b), (b, c), (c, a)):
                edges[(u, v)] += 1
        for (u, v), n in edges.items():
            self.assertEqual(n, 1, "an edge used twice in one direction: inconsistent winding")
            self.assertEqual(edges[(v, u)], 1, "an edge without its opposite: not closed")

    def test_dimensions_and_volume(self):
        d = self.d
        np.testing.assert_allclose(self.pts.min(axis=0), [-d["outer"][0] / 2, -d["outer"][1] / 2, 0.0], atol=1e-12)
        np.testing.assert_allclose(self.pts.max(axis=0), [d["outer"][0] / 2, d["outer"][1] / 2, d["wall_h"]], atol=1e-12)
        box = d["outer"][0] * d["outer"][1] * d["wall_h"]
        pocket = (d["pocket"] ** 2 - (4 - math.pi) * d["pocket_r"] ** 2) * (d["wall_h"] - d["floor_t"])
        self.assertAlmostEqual(S.volume(self.pts, self.faces), box - pocket, delta=box * 0.01)

    def test_stl_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bin.stl")
            n = S.write_stl(path, self.pts, self.faces)
            with open(path, "rb") as f:
                data = f.read()
        self.assertEqual(struct.unpack("<I", data[80:84])[0], n)
        self.assertEqual(len(data), 84 + 50 * n)
        first = struct.unpack("<12f", data[84:132])
        self.assertLessEqual(max(abs(v) for v in first[3:]), self.d["outer"][0] * 1000)  # millimetres


class RandomBinTests(unittest.TestCase):
    def test_random_bins_reachable_separated_varied(self):
        yaws = []
        for seed in range(15):
            layout = R.sample_layout(seed, bin_mode="random")
            bins = layout["bins"]
            self.assertEqual(set(bins), set(R.load_bins()))
            for b in bins.values():
                self.assertTrue(R.bin_pose_ok(b["xy"]))
                yaws.append(b["yaw_deg"])
            for (_, a), (_, b) in [(x, y) for i, x in enumerate(bins.items()) for y in list(bins.items())[i + 1 :]]:
                self.assertGreaterEqual(math.dist(a["xy"], b["xy"]), 2 * R._bin_radius() + R.BIN_SEPARATION - 1e-9)
            for key in ("cubes", "screws", "aaa_cells", "vials", "caps"):
                kind = {v: k for k, v in R.LAYOUT_KEYS.items()}[key]
                for o in layout[key]:
                    f = R.footprint(kind, o["xy"], o["yaw_deg"])
                    for b in bins.values():
                        self.assertGreaterEqual(R.bin_gap(f, b), R.BIN_CLEARANCE - 1e-9)
        self.assertGreater(np.ptp(yaws), 45.0)

    def test_robot_base_and_far_table_rejected(self):
        self.assertFalse(R.bin_pose_ok((0.05, 0.05)))  # on the robot base
        self.assertFalse(R.bin_pose_ok((0.75, 0.0)))  # off the table edge
        self.assertFalse(R.bin_pose_ok((0.60, 0.60)))  # out of reach

    def test_rotated_checks(self):
        b = dict(R.load_bins()["bin_1"], xy=[0.3, 0.1], yaw_deg=45.0)
        z = C.TABLE_TOP_Z + C.SURFACE_DZ + 0.03
        self.assertTrue(X.in_bin([0.3 + 0.02, 0.1 + 0.02, z], b)[0])  # along the bin's axis
        corner = [0.3 + 0.045 * math.sqrt(2) * math.cos(math.radians(45)), 0.1 + 0.045 * math.sqrt(2) * math.sin(math.radians(45)), z]
        self.assertFalse(X.in_bin(corner, b)[0])  # a pocket corner direction is wall at 45 deg yaw
        # bin_gap: a cube just beyond the rotated wall vs just inside it
        out = R.footprint("cube", (0.3 + 0.064 + 0.02, 0.1), 0.0)
        self.assertGreater(R.bin_gap(out, b), 0.0)
        self.assertLess(R.bin_gap(R.footprint("cube", (0.3 + 0.05, 0.1), 0.0), b), 0.0)


class BinPoseFitTests(unittest.TestCase):
    def test_bin_xy_yaw_colour(self):
        cam = nominal_camera()
        img = np.zeros((cam.H, cam.W, 3), np.uint8)
        for xy, yaw in (((0.30, -0.20), 20.0), ((0.10, 0.35), -35.0)):
            mask = P.silhouette("bin", cam, xy, yaw, (cam.H, cam.W))
            img[:] = 0
            img[mask.astype(bool)] = (200, 60, 40)  # BGR blue
            r = P.refine("bin", img, cam, mask)
            self.assertGreaterEqual(r["fit_iou"], P.FIT_IOU_MIN)
            self.assertLess(math.dist(r["xy"], xy) * 1000, 1.0)
            self.assertLess(abs((r["yaw_deg"] - yaw + 45) % 90 - 45), 1.0)
            self.assertEqual(r["colour"], "blue")


if __name__ == "__main__":
    unittest.main()
