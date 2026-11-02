"""Host tests for the real-cell policy pieces (mm/real): frames, camera, rows.

    .venv/bin/python -m unittest tests.test_real_policy
"""

import json
import os
import unittest

import numpy as np

import fk
import scene_config as C
from mm.control import executor as X
from mm.planning import grasp as G
from mm.real import brain as BR
from mm.real import frames as F
from test_support import camera_model

CLIENT_CAMS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "PhysicalAI-Hackathon-2026-v6", "calibration", "cameras.json")
CELL_HOME_DEG = [-0.0065, 3.2909, -11.8967, -0.0007, -74.8218, -0.0286]  # mock_cell.HOME


class FrameTests(unittest.TestCase):
    def test_uf0(self):
        p = np.array([0.3, -0.1, -0.0051])
        self.assertAlmostEqual(F.base_to_uf0(p)[2], -335.1)
        np.testing.assert_allclose(F.uf0_to_base(F.base_to_uf0(p)), p)

    def test_r_spread_round_trip_and_symmetry(self):
        for r in (-170.0, -90.0, 0.0, 45.0, 135.0):
            s = F.r_to_spread(r)
            self.assertAlmostEqual(F.spread_to_r(s, r), r)
            self.assertAlmostEqual(F.spread_to_r(-s, r), r)  # symmetric jaws: no half turn for -s
        self.assertAlmostEqual(F.spread_to_r(F.r_to_spread(10.0), 170.0), 190.0)

    def test_cell_home_is_the_sort_home_fingers_down(self):
        reach = G.Reach()
        q = np.deg2rad(CELL_HOME_DEG)
        _, tip = fk.tool_pose(q, reach.tool)
        np.testing.assert_allclose(tip, X.HOME_XYZ, atol=0.002)
        r_j6, _ = fk.fk(q)
        w, p, r = F.wpr_from_rotation(r_j6)
        self.assertAlmostEqual(abs(w), 180.0, delta=0.5)
        self.assertAlmostEqual(p, 0.0, delta=0.5)
        # the R it reports closes the jaws along the twin's spread axis
        spread = r_j6 @ np.array(C.FINGER_SPREAD_AXIS)
        self.assertGreater(abs(np.dot(F.r_to_spread(r)[:2], spread[:2])), 0.999)

    def test_cell_camera_matches_the_sim_camera(self):
        cfg = dict(C.TABLE_CAM)
        sim = camera_model(cfg)
        cell = F.CellCamera(F.sim_calibration(sim.j))
        for p in ([0.3, 0.1, sim.table_z], [0.45, -0.2, sim.table_z + 0.02]):
            np.testing.assert_allclose(cell.project(p), sim.project(p), atol=1e-6)
        np.testing.assert_allclose(cell.to_plane(800.0, 600.0), sim.to_plane(800.0, 600.0), atol=1e-9)

    @unittest.skipUnless(os.path.exists(CLIENT_CAMS), "organizers' client not next to the project")
    def test_real_calibration_close_to_twin(self):
        sim = camera_model(dict(C.TABLE_CAM))
        real = F.CellCamera.load(CLIENT_CAMS, "table", (sim.W, sim.H))
        p = [0.35, 0.05, sim.table_z]
        self.assertLess(np.linalg.norm(real.project(p) - sim.project(p)), 2.0)


def pick(xy=(0.35, 0.05), bin_xy=(0.32, -0.24)):
    return {"category": "cube", "xy": list(xy), "pick_z": C.SURFACE_DZ + C.TIP_CLEARANCE,
            "place_xy": list(bin_xy), "place_z": C.SURFACE_DZ + 0.095,
            "spread_dir": [0.0, 1.0, 0.0], "place_spread_dir": [0.0, 1.0, 0.0]}


def pick_route(p=None):
    p = p or pick()
    segs = [s[:6] + (np.array(p["spread_dir"]),) for s in X.pick_segments(p, X.HOME_XYZ)]
    segs.append(("park", segs[-1][2], np.array(BR.PARK_XYZ), 1.5, C.JAW_OPEN, C.JAW_OPEN, np.array(p["spread_dir"])))
    return p, segs


class RowTests(unittest.TestCase):
    def test_pick_rows(self):
        p, segs = pick_route()
        rows = BR.segments_to_rows(segs, r_start=180.0)
        self.assertTrue(all(F.in_tcp_box(r) for r in rows))
        g = [r[6] for r in rows]
        # open, then closed from the close row through the carry, then open from the release
        first_closed, last_closed = g.index(1.0), len(g) - 1 - g[::-1].index(1.0)
        self.assertTrue(all(v == 1.0 for v in g[first_closed:last_closed + 1]))
        self.assertEqual(sum(1 for a, b in zip(g, g[1:]) if a != b), 2)
        # closed at the grasp height, fingers down, R continuous
        z_closed = rows[first_closed][2]
        self.assertAlmostEqual(z_closed, F.base_to_uf0([0, 0, p["pick_z"]])[2], places=3)
        self.assertTrue(all(r[3] == -180.0 and r[4] == 0.0 for r in rows))
        self.assertLess(max(abs(a[5] - b[5]) for a, b in zip(rows, rows[1:])), 30.0)
        # jaws close across y at the object: R = +-90 + 180k
        self.assertGreater(abs(np.dot(F.r_to_spread(rows[first_closed][5])[:2], [0.0, 1.0])), 0.999)
        np.testing.assert_allclose(F.uf0_to_base(rows[-1][:3]), BR.PARK_XYZ, atol=1e-6)

    def test_speed_cap(self):
        segs = [("approach", np.array(X.HOME_XYZ), np.array([0.45, 0.25, 0.14]), 0.5, C.JAW_OPEN, C.JAW_OPEN,
                 np.array([1.0, 0.0, 0.0]))]
        rows = BR.segments_to_rows(segs, r_start=180.0)
        steps = [np.linalg.norm(np.subtract(a[:3], b[:3])) for a, b in zip(rows, rows[1:])]
        self.assertLessEqual(max(steps), 1.6 * BR.TCP_SPEED * 1000 * BR.DT)  # eased: peak 1.5x the mean
        self.assertLess(max(steps) / BR.DT, 50.0)  # under the real arm's ~50 mm/s (27 Oct logs)

    def test_split_pick(self):
        p, segs = pick_route()
        a, b, c, shut = BR.split_pick(segs, p["category"], 180.0)
        self.assertLess(shut, C.JAW_OPEN)
        grasp_z = F.base_to_uf0([0, 0, p["pick_z"]])[2]
        settle = round(BR.SETTLE_S / BR.DT)
        # move 1: open all the way, ends held at the grasp pose
        self.assertTrue(all(r[6] == 0.0 for r in a))
        self.assertTrue(all(np.allclose(r[:3], a[-1][:3]) for r in a[-settle:]))
        self.assertAlmostEqual(a[-1][2], grasp_z, places=3)
        # move 2: closes on its first row, where move 1 ended; closed all the way, ends held above the bin
        self.assertTrue(all(r[6] == 1.0 for r in b))
        np.testing.assert_allclose(b[0][:3], a[-1][:3], atol=1e-6)
        self.assertTrue(all(np.allclose(r[:3], b[-1][:3]) for r in b[-settle:]))
        np.testing.assert_allclose(F.uf0_to_base(b[-1][:3])[:2], p["place_xy"], atol=1e-6)
        # move 3: opens on its first row, where move 2 ended; ends at park
        self.assertTrue(all(r[6] == 0.0 for r in c))
        np.testing.assert_allclose(c[0][:3], b[-1][:3], atol=1e-6)
        np.testing.assert_allclose(F.uf0_to_base(c[-1][:3]), BR.PARK_XYZ, atol=1e-6)
        rows = a + b + c
        self.assertTrue(all(F.in_tcp_box(r) for r in rows))
        self.assertTrue(all(r[3] == -180.0 and r[4] == 0.0 for r in rows))
        self.assertLess(max(abs(x[5] - y[5]) for x, y in zip(rows, rows[1:])), 30.0)


def bare_brain():
    """Brain without YOLO: enough for the move-following logic (no capture)."""
    b = BR.Brain.__new__(BR.Brain)
    b.reach, b.log_dir, b.rollout, b.bins_mode = G.Reach(), None, 0, "detected"
    b.home, b.park = np.asarray(X.HOME_XYZ, float), np.asarray(BR.PARK_XYZ, float)
    b.reset()
    b.acts = 3
    return b


def arm_at(p_base, spread):
    """joints (deg + gripper) and tcp_pose (UF0 mm, w, p, r) of the arm with its fingertip at p_base."""
    import ik

    reach = G.Reach()
    q, ok = ik.solve(np.deg2rad(CELL_HOME_DEG), np.asarray(p_base, float), [0.0, 0.0, -1.0], reach.limits,
                     tool_offset=reach.tool, spread_dir=np.asarray(spread, float), iters=500, damping=0.02)
    assert ok
    rot, tip = fk.tool_pose(q, reach.tool)
    r = F.spread_to_r(rot @ np.array(C.FINGER_SPREAD_AXIS))
    return [*np.rad2deg(q), 0.0], [*F.base_to_uf0(tip), -180.0, 0.0, r]


class FollowTests(unittest.TestCase):
    def setUp(self):
        p, segs = pick_route()
        self.a, self.b, self.c, shut = BR.split_pick(segs, p["category"], 180.0)
        self.brain = bare_brain()
        self.brain._expect("grasp", self.a, BR.ARRIVE_TOL, C.JAW_OPEN, then=[
            {"decision": "carry", "label": "release", "rows": self.b, "tol": BR.ARRIVE_TOL, "jaw": shut},
            {"decision": "release", "label": "park", "rows": self.c, "tol": BR.PARK_TOL, "jaw": C.JAW_OPEN}])
        self.grasp = F.uf0_to_base(self.a[-1][:3])
        self.spread = F.r_to_spread(self.a[-1][5])

    def test_short_of_the_grasp_pose_waits_open(self):
        joints, tcp = arm_at(self.grasp + [0.0, 0.0, 0.03], self.spread)  # 30 mm above: still descending
        out = self.brain.act(None, joints, tcp)
        self.assertEqual(out["info"]["decision"], "wait")
        self.assertTrue(all(r[6] == 0.0 for r in out["rows"]))  # never closes on the clock
        np.testing.assert_allclose(out["rows"][-1][:3], self.a[-1][:3], atol=1e-6)
        self.assertEqual(self.brain.expect["label"], "grasp")

    def test_at_the_grasp_pose_closes_and_carries(self):
        joints, tcp = arm_at(self.grasp, self.spread)
        out = self.brain.act(None, joints, tcp)
        self.assertEqual(out["info"]["decision"], "carry")
        self.assertEqual(out["rows"][0][6], 1.0)
        self.assertEqual(self.brain.expect["label"], "release")
        np.testing.assert_allclose(self.brain.expect["target"], self.b[-1][:6])

    def test_never_reaching_the_grasp_pose_aborts_open(self):
        joints, tcp = arm_at(self.grasp + [0.0, 0.0, 0.01], self.spread)
        for _ in range(BR.MAX_WAITS):
            self.assertEqual(self.brain.act(None, joints, tcp)["info"]["decision"], "wait")
        out = self.brain.act(None, joints, tcp)
        self.assertEqual(out["info"]["decision"], "abort")
        self.assertTrue(all(r[6] == 0.0 for r in out["rows"]))
        np.testing.assert_allclose(F.uf0_to_base(out["rows"][-1][:3]), BR.PARK_XYZ, atol=1e-6)
        self.assertEqual(self.brain.expect["label"], "park")

    def test_park_not_reached_waits_without_capture(self):
        self.brain._expect("park", self.c, BR.PARK_TOL, C.JAW_OPEN)
        joints, tcp = arm_at(self.brain.park + [0.06, -0.08, 0.0], self.spread)  # where v4 stopped short
        out = self.brain.act(None, joints, tcp)  # no image: a capture would fail
        self.assertEqual(out["info"]["decision"], "wait")
        np.testing.assert_allclose(F.uf0_to_base(out["rows"][-1][:3]), BR.PARK_XYZ, atol=1e-6)
        steps = [np.linalg.norm(np.subtract(x[:3], y[:3])) for x, y in zip(out["rows"], out["rows"][1:])]
        self.assertLess(max(steps) / BR.DT, 50.0)

    def test_parked_hands_back_to_capture(self):
        self.brain._expect("park", self.c, BR.PARK_TOL, C.JAW_OPEN)
        joints, tcp = arm_at(self.brain.park + [0.005, 0.0, 0.0], self.spread)
        self.assertIsNone(self.brain._follow(None, joints, tcp))
        self.assertIsNone(self.brain.expect)


if __name__ == "__main__":
    unittest.main()
