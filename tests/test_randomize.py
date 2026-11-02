"""Host tests for mm/datagen/randomize.py (step A2).

    .venv/bin/python -m unittest tests.test_randomize
"""

import importlib
import itertools
import json
import math
import os
import tempfile
import unittest

import numpy as np

import ik
import scene_config as C
from mm.datagen import randomize as R

KEYS = R.LAYOUT_KEYS
DISTRACTORS = set(R.DISTRACTOR_KINDS)


def objects(layout):
    for kind, key in KEYS.items():
        for o in layout[key]:
            yield kind, o


class RandomizeTests(unittest.TestCase):
    def test_same_seed_same_layout(self):
        for seed in (0, 1, 123, 300_005):
            self.assertEqual(R.sample_layout(seed), R.sample_layout(seed))
        self.assertNotEqual(R.sample_layout(1)["cubes"], R.sample_layout(2)["cubes"])

    def test_counts_and_classes(self):
        for seed in range(100):
            layout = R.sample_layout(seed)
            targets = [k for k, _ in objects(layout) if k not in DISTRACTORS]
            self.assertTrue(R.N_TARGETS[0] <= len(targets) <= R.N_TARGETS[1], seed)
            self.assertEqual(set(targets), set(R.TARGET_CLASSES), seed)
            self.assertLessEqual(len(layout["vials"]) + len(layout["caps"]), R.N_DISTRACTORS[1])
            for kind, key in KEYS.items():
                self.assertLessEqual(len(layout[key]), R.MAX_PER_CLASS[kind])

    def test_spacing_rules_hold(self):
        for seed in itertools.chain(range(100), R.split_seeds("hard")[:30]):
            layout = R.sample_layout(seed, hard=seed >= R.SPLITS["hard"][0])
            self.assertIn(layout["spacing_m"], R.SPACINGS)
            objs = list(objects(layout))
            for (k1, a), (k2, b) in itertools.combinations(objs, 2):
                self.assertGreaterEqual(
                    math.dist(a["xy"], b["xy"]), layout["spacing_m"] - 1e-12, seed
                )
                gap = R.footprint_gap(
                    R.footprint(k1, a["xy"], a["yaw_deg"]),
                    R.footprint(k2, b["xy"], b["yaw_deg"]),
                )
                self.assertGreaterEqual(gap, R.MIN_GAP - 1e-12, seed)

    def test_about_a_fifth_of_scenes_cluttered(self):
        spacings = [R.sample_layout(s)["spacing_m"] for s in range(300)]
        cluttered = sum(s < R.SPACINGS[0] for s in spacings) / len(spacings)
        self.assertTrue(0.1 <= cluttered <= 0.3, cluttered)
        hard = [R.sample_layout(s, hard=True)["spacing_m"] for s in range(20)]
        self.assertTrue(all(s < R.SPACINGS[0] for s in hard))

    def test_footprint_gap(self):
        cube = R.footprint("cube", (0.0, 0.0), 0.0)
        r = cube[2]
        self.assertAlmostEqual(R.footprint_gap(cube, R.footprint("cube", (0.05, 0), 0)), 0.05 - 2 * r)
        # Crossing tubes overlap even though their centres are far from the ends.
        t1 = R.footprint("vial", (0.0, 0.0), 0.0)
        t2 = R.footprint("vial", (0.0, 0.0), 90.0)
        self.assertLess(R.footprint_gap(t1, t2), 0)

    def test_randomized_ranges(self):
        for seed in range(50):
            layout = R.sample_layout(seed)
            light = layout["lighting"]
            self.assertTrue(R.LIGHT_SCALE[0] <= light["scale"] <= R.LIGHT_SCALE[1])
            self.assertTrue(R.LIGHT_KELVIN[0] <= light["kelvin"] <= R.LIGHT_KELVIN[1])
            for key in ("eye", "look_at"):
                d = np.subtract(layout["table_cam"][key], C.TABLE_CAM[key])
                self.assertTrue(np.all(np.abs(d) <= R.CAM_JITTER + 1e-12))
            self.assertIn(layout["arm"]["mode"], ("parked", "in_view"))

    def test_targets_inside_ik_reachable_area(self):
        ik.set_tool_axis(C.TOOL_AXIS)
        ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
        limits = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])
        q_home = np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)])
        tool = tuple(np.array(C.TOOL_AXIS) * C.TOOL_LENGTH)
        grasp_z = C.SURFACE_DZ + C.TIP_CLEARANCE
        down = [0.0, 0.0, -1.0]

        def reach(xy, spreads):
            for spread in spreads:
                q, ok = q_home, True
                for z in (grasp_z + 0.06, grasp_z):  # hover, then grasp
                    q, ok = ik.solve(q, [*xy, z], down, limits, tool_offset=tool, spread_dir=spread)
                    if not ok:
                        break
                if ok:
                    return True
            return False

        for seed in range(12):
            for kind, o in objects(R.sample_layout(seed)):
                if kind in DISTRACTORS:
                    continue
                x, y = o["xy"]
                self.assertTrue(R.TARGET_X[0] <= x <= R.TARGET_X[1], (seed, kind))
                self.assertTrue(R.TARGET_Y[0] <= y <= R.TARGET_Y[1], (seed, kind))
                a = math.radians(o["yaw_deg"])
                if kind == "cube":  # jaws close on opposite faces: yaw or yaw + 90
                    spreads = [(math.cos(a), math.sin(a), 0), (-math.sin(a), math.cos(a), 0)]
                elif kind in ("aaa_cell", "screw"):  # jaws close across the axis
                    spreads = [(-math.sin(a), math.cos(a), 0)]
                else:
                    spreads = [None]
                self.assertTrue(reach((x, y), spreads), (seed, kind, o["xy"]))

    def test_in_view_arm_pose_is_solved(self):
        poses = [R.sample_layout(s, hard=True)["arm"] for s in range(20)]
        in_view = [p for p in poses if p["mode"] == "in_view"]
        self.assertTrue(in_view)
        for p in in_view:
            for j, deg in p["joints_deg"].items():
                lo, hi = C.JOINT_LIMITS_DEG[j]
                self.assertTrue(lo <= deg <= hi, (j, deg))

    def test_splits_are_disjoint(self):
        ranges = sorted(R.SPLITS.values())
        for (_, hi), (lo, _) in zip(ranges, ranges[1:]):
            self.assertLessEqual(hi, lo)


class BinTests(unittest.TestCase):
    """B1: bins from configs/bins.yaml."""

    def test_one_bin_per_target_class(self):
        """v4: as many identical bins as target classes; the robot assigns them."""
        import yaml

        from mm.planning.bin_assign import default_table

        with open(os.path.join(os.path.dirname(R.BINS_PATH), "classes.yaml")) as f:
            classes = yaml.safe_load(f)["classes"]
        targets = {c["name"] for c in classes if c["role"] == "target"}
        bins = R.load_bins()
        self.assertEqual(set(bins), {"bin_1", "bin_2", "bin_3"})
        self.assertEqual(len({(b["colour"], b["mat"]) for b in bins.values()}), 1)  # all alike
        self.assertEqual(set(default_table()), targets)

    def test_objects_clear_bins(self):
        for seed in itertools.chain(range(100), R.split_seeds("hard")[:30]):
            layout = R.sample_layout(seed, hard=seed >= R.SPLITS["hard"][0])
            self.assertEqual(set(layout["bins"]), set(R.load_bins()))
            for kind, o in objects(layout):
                f = R.footprint(kind, o["xy"], o["yaw_deg"])
                for name, b in layout["bins"].items():
                    self.assertGreaterEqual(R.bin_gap(f, b), R.BIN_CLEARANCE - 1e-9, (seed, kind, name))

    def test_bins_do_not_overlap(self):
        for (n1, a), (n2, b) in itertools.combinations(R.load_bins().items(), 2):
            gap_x = abs(a["xy"][0] - b["xy"][0]) - (a["outer"][0] + b["outer"][0]) / 2
            gap_y = abs(a["xy"][1] - b["xy"][1]) - (a["outer"][1] + b["outer"][1]) / 2
            self.assertGreater(max(gap_x, gap_y), 0.0, (n1, n2))

    def test_bins_reachable_and_in_camera_crop(self):
        from mm.datagen.masks_to_yolo import load_roi, nominal_camera, roi_for

        ik.set_tool_axis(C.TOOL_AXIS)
        ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
        limits = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])
        q0 = np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)])
        tool = tuple(np.array(C.TOOL_AXIS) * C.TOOL_LENGTH)
        cam = nominal_camera()
        x0, y0, w, h = roi_for(load_roi(), cam.W, cam.H)
        z0 = C.TABLE_TOP_Z + C.SURFACE_DZ
        for name, b in R.load_bins().items():
            for z in (b["release_z"], b["release_z"] + 0.06):  # release, then hover
                _, ok = ik.solve(q0, [*b["xy"], C.SURFACE_DZ + z], [0, 0, -1], limits, tool_offset=tool)
                self.assertTrue(ok, (name, z))
            hx, hy = b["outer"][0] / 2, b["outer"][1] / 2
            for dx in (-hx, hx):
                for dy in (-hy, hy):
                    for dz in (0.0, b["wall_h"]):
                        u, v = cam.project([b["xy"][0] + dx, b["xy"][1] + dy, z0 + dz])
                        self.assertTrue(x0 <= u <= x0 + w and y0 <= v <= y0 + h, name)

    def test_bin_free_layouts_still_available(self):
        layout = R.sample_layout(3, bins=False)
        self.assertNotIn("bins", layout)


class SceneOverrideTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("MM_SCENE_JSON", None)
        importlib.reload(C)

    def test_layout_loads_through_scene_config(self):
        layout = R.sample_layout(7)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "layout.json")
            with open(path, "w") as f:
                json.dump(layout, f)
            os.environ["MM_SCENE_JSON"] = path
            importlib.reload(C)
        self.assertEqual(len(C.SORT_CUBES), len(layout["cubes"]))
        self.assertEqual(len(C.SORT_CUBE_YAWS_DEG), len(C.SORT_CUBES))
        self.assertEqual(len(C.CAP_ORIENTATIONS), len(C.SORT_CAPS))
        self.assertEqual(len(C.SORT_VIALS), len(layout["vials"]))
        self.assertEqual(len(C.SORT_AAA_CELLS), len(layout["aaa_cells"]))
        self.assertEqual(len(C.SCREWS), len(layout["screws"]))
        self.assertEqual(C.SORT_CUBES[0][2], layout["cubes"][0]["colour"])
        # Screw layout xy is the shank middle; SCREWS holds the head face.
        s, (hx, hy, yaw) = layout["screws"][0], C.SCREWS[0]
        self.assertAlmostEqual(
            math.dist(s["xy"], (hx, hy)), (C.SCREW_HEAD_H + C.SCREW_LENGTH) / 2, places=9
        )
        self.assertTrue(C.SHOW_BINS)
        self.assertEqual(set(C.SORT_BINS), set(layout["bins"]))
        self.assertEqual(C.SORT_BINS["bin_3"]["xy"], tuple(layout["bins"]["bin_3"]["xy"]))
        self.assertAlmostEqual(C.BIN_ROUGHNESS, layout["bin_roughness"])

    def test_starter_scene_unchanged_without_override(self):
        os.environ.pop("MM_SCENE_JSON", None)
        importlib.reload(C)
        self.assertEqual(len(C.SORT_CUBES), 6)
        self.assertEqual(C.SORT_AAA_CELLS, [])
        self.assertEqual(C.CAP_ORIENTATIONS, ["narrow_end_up"] * 2)
        self.assertFalse(C.SHOW_BINS)
        self.assertEqual(set(C.SORT_BINS), {"cube", "cap", "vial"})
        self.assertEqual(C.SCREWS, [])


if __name__ == "__main__":
    unittest.main()
