"""Host tests for mm/control/executor.py (step B5).

    .venv/bin/python -m unittest tests.test_executor
"""

import copy
import math
import unittest

import numpy as np

import scene_config as C
from mm.control import executor as X
from mm.datagen import randomize as R
from mm.planning import order as O
from mm.planning.bin_assign import default_table

BINS = default_table()  # v4: class -> one of the identical bins


def plan_with(picks):
    for k, p in enumerate(picks):
        p["order_index"] = k
    return {"source": X.PLAN_SOURCE, "task": "bin_sort", "open_gap_m": 2 * C.JAW_OPEN, "bins": BINS, "picks": picks}


def pick(xy, cat="cube", bin_name="cube"):
    b = BINS[bin_name]
    return {
        "detection_id": 0,
        "category": cat,
        "bin": bin_name,
        "xy": list(xy),
        "pick_z": C.SURFACE_DZ + C.TIP_CLEARANCE,
        "place_xy": list(b["xy"]),
        "place_z": C.SURFACE_DZ + b["release_z"],
        "spread_dir": [0.0, 1.0, 0.0],
        "place_spread_dir": [0.0, 1.0, 0.0],
    }


class ExecutorTests(unittest.TestCase):
    def test_picks_chain_without_returning_home(self):
        picks = [pick((0.30, 0.0)), pick((0.40, 0.1), "screw", "screw"), pick((0.35, -0.1), "aaa_cell", "aaa_cell")]
        out = list(X.chained_segments(picks))
        self.assertEqual(len(out), 3)
        for (_, prev), (_, nxt) in zip(out, out[1:]):
            np.testing.assert_allclose(nxt[0][1], prev[9][2])  # approach starts at last retreat end
            self.assertNotIn("home", [s[0] for s in prev])
        self.assertEqual(out[-1][1][-1][0], "home")
        np.testing.assert_allclose(out[0][1][0][1], X.HOME_XYZ)
        np.testing.assert_allclose(out[-1][1][-1][2], X.HOME_XYZ)

    def test_chained_path_matches_order_cost_model(self):
        picks = [pick((0.30, 0.05)), pick((0.42, 0.2), "screw", "screw"), pick((0.28, -0.1), "aaa_cell", "aaa_cell")]
        length = sum(
            float(np.linalg.norm(np.subtract(p1, p0))) for _, segs in X.chained_segments(picks) for _, p0, p1, *_ in segs
        )
        self.assertAlmostEqual(length, O.tour_cost(picks), places=9)

    def test_jaw_targets(self):
        self.assertEqual(X.jaw_closed("cube"), C.CUBE_JAW_HALF_TARGET)
        self.assertEqual(X.jaw_closed("screw"), C.JAW_CLOSED)
        segs = X.pick_segments(pick((0.3, 0.0), "screw", "screw"), X.HOME_XYZ)
        by = {s[0]: s for s in segs}
        self.assertEqual((by["close"][4], by["close"][5]), (C.JAW_OPEN, C.JAW_CLOSED))
        self.assertEqual((by["release"][4], by["release"][5]), (C.JAW_CLOSED, C.JAW_OPEN))

    def test_long_moves_respect_speed_limit(self):
        for _, segs in X.chained_segments([pick((0.46, 0.28)), pick((0.24, -0.15))]):
            for name, p0, p1, dur, *_ in segs:
                self.assertLessEqual(float(np.linalg.norm(np.subtract(p1, p0))) / dur, X.MAX_TCP_SPEED + 1e-9, name)

    def test_validate_plan(self):
        good = plan_with([pick((0.3, 0.0))])
        X.validate_plan(good)
        for mutate, msg in (
            (lambda p: p.update(source="Cosmos3-Nano"), "source"),
            (lambda p: p["picks"][0].update(place_xy=[0.0, 0.0]), "bin xy"),
            (lambda p: p["picks"][0].update(spread_dir=[0.0, 0.0, 1.0]), "vertical spread"),
            (lambda p: p["picks"][0].update(order_index=3), "order"),
            (lambda p: p["picks"][0].update(category="vial"), "category"),
        ):
            bad = copy.deepcopy(good)
            mutate(bad)
            with self.assertRaises(ValueError, msg=msg):
                X.validate_plan(bad)

    def test_in_bin(self):
        b = BINS["screw"]
        z0 = C.TABLE_TOP_Z + C.SURFACE_DZ
        self.assertTrue(X.in_bin([b["xy"][0], b["xy"][1], z0 + b["floor_t"] + 0.003], b)[0])
        self.assertFalse(X.in_bin([b["xy"][0] + b["outer"][0], b["xy"][1], z0 + 0.02], b)[0])  # beside it
        self.assertFalse(X.in_bin([b["xy"][0], b["xy"][1], z0 + b["wall_h"] + 0.02], b)[0])  # above the rim

    def test_path_log(self):
        log = X.PathLog(1 / 30)
        log.start("approach")
        for k in range(4):
            log.step([0.1 * k, 0, 0], [0.1 * k, 0.001, 0])
        seg = log.end()
        self.assertAlmostEqual(seg["commanded_m"], 0.3, places=9)
        self.assertAlmostEqual(seg["time_s"], 4 / 30, places=4)
        self.assertEqual(X.summarise([seg, seg])["tcp_path_commanded_m"], 0.6)

    def test_screw_centre_is_shank_middle(self):
        c = X.object_centre("Screw0", [0, 0, 0], np.eye(3))
        self.assertAlmostEqual(-c[2], (C.SCREW_HEAD_H + C.SCREW_LENGTH) / 2)


if __name__ == "__main__":
    unittest.main()
