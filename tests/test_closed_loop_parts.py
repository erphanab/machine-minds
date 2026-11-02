"""Host tests for B6-B8 parts: grasp check, tracker, bins from detections.

    .venv/bin/python -m unittest tests.test_closed_loop_parts
"""

import math
import unittest

import numpy as np

import ik
import scene_config as C
from mm.control import grasp_check as GC
from mm.datagen.randomize import load_bins
from mm.perception.bins import first_bins, track_bins
from mm.planning import grasp as G
from mm.planning.tracker import MAX_ATTEMPTS, Tracker


def hover_q(xy):
    ik.set_tool_axis(C.TOOL_AXIS)
    ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
    reach = G.Reach()
    q, ok = ik.solve(reach.home, [*xy, C.SURFACE_DZ + C.TIP_CLEARANCE + G.HOVER_DZ], [0, 0, -1], reach.limits,
                     tool_offset=reach.tool, spread_dir=[0, 1, 0])
    assert ok
    return q


class GraspCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cam = GC.wrist_camera()
        cls.xy = (0.35, 0.05)
        cls.q = hover_q(cls.xy)

    def test_pick_spot_is_in_the_wrist_view(self):
        for cls_ in GC.SPOT_RADIUS:
            m = GC.spot_mask(self.cam, self.q, self.xy, cls_, (720, 1280))
            self.assertGreater(m.sum(), 200, cls_)

    def scene(self, with_object):
        img = np.full((720, 1280, 3), 205, np.uint8)  # table
        m = GC.spot_mask(self.cam, self.q, self.xy, "cube", img.shape)
        if with_object:
            ys, xs = np.nonzero(m)
            cy, cx = int(ys.mean()), int(xs.mean())
            img[cy - 12 : cy + 12, cx - 12 : cx + 12] = (30, 30, 200)  # a red cube
        return img, m

    def test_verdicts(self):
        pre, m = self.scene(True)
        empty, _ = self.scene(False)
        self.assertTrue(GC.verdict(pre, empty, m)["taken"])  # spot cleared
        self.assertFalse(GC.verdict(pre, pre.copy(), m)["taken"])  # object still there
        noisy = np.clip(pre.astype(int) + np.random.default_rng(0).integers(-6, 7, pre.shape), 0, 255).astype(np.uint8)
        self.assertFalse(GC.verdict(pre, noisy, m)["taken"])  # sensor noise is not a pick

    def test_spot_out_of_view(self):
        m = np.zeros((720, 1280), bool)
        self.assertIsNone(GC.verdict(np.zeros((720, 1280, 3), np.uint8), np.zeros((720, 1280, 3), np.uint8), m)["taken"])


def pick(cat, xy):
    return {"category": cat, "xy": list(xy), "pick_z": C.SURFACE_DZ + C.TIP_CLEARANCE,
            "place_xy": [0.2, -0.24], "place_z": C.SURFACE_DZ + 0.095}


class TrackerTests(unittest.TestCase):
    def test_match_move_add_vanish(self):
        here = (0.3, 0.0, 0.24)
        t = Tracker()
        c0 = t.update([pick("cube", (0.30, 0.0)), pick("screw", (0.40, 0.1)), pick("aaa_cell", (0.35, -0.1))], here, 0)
        self.assertEqual(len(c0["new"]), 3)
        ids = {t.tracks[i]["pick"]["category"]: i for i in t.tracks}
        # cube moved 10 mm, screw gone, a new cube appears
        c1 = t.update([pick("cube", (0.31, 0.0)), pick("aaa_cell", (0.35, -0.1)), pick("cube", (0.25, 0.2))], here, 1)
        self.assertEqual(c1["moved"], [ids["cube"]])
        self.assertEqual(c1["vanished"], [ids["screw"]])
        self.assertEqual(len(c1["new"]), 1)
        self.assertEqual(sorted(t.order), sorted(t.tracks))
        self.assertEqual(t.tracks[ids["cube"]]["pick"]["xy"], [0.31, 0.0])

    def test_class_must_match(self):
        t = Tracker()
        t.update([pick("cube", (0.30, 0.0))], (0.3, 0, 0.24), 0)
        c = t.update([pick("screw", (0.30, 0.0))], (0.3, 0, 0.24), 1)
        self.assertEqual(len(c["new"]), 1)
        self.assertEqual(len(c["vanished"]), 1)

    def test_retries_then_flag(self):
        t = Tracker()
        t.update([pick("cube", (0.30, 0.0))], (0.3, 0, 0.24), 0)
        tid, _ = t.next()
        for k in range(MAX_ATTEMPTS):
            gave_up = t.attempt_failed(tid)
            self.assertEqual(gave_up, k == MAX_ATTEMPTS - 1)
            if not gave_up:
                t.update([pick("cube", (0.30, 0.0))], (0.3, 0, 0.24), k + 1)
                self.assertEqual(t.next()[0], tid)  # retried, same identity
        self.assertEqual(t.next(), (None, None))
        c = t.update([pick("cube", (0.301, 0.0))], (0.3, 0, 0.24), 9)
        self.assertEqual(c["new"], [])  # a flagged object is not offered again


    def test_give_up_flags_now(self):
        t = Tracker()
        t.update([pick("cube", (0.30, 0.0)), pick("screw", (0.40, 0.1))], (0.3, 0, 0.24), 0)
        tid, _ = t.next()
        t.give_up(tid)
        self.assertIn(tid, t.flagged)
        self.assertNotIn(tid, t.order)
        c = t.update([pick("cube", (0.30, 0.0)), pick("screw", (0.40, 0.1))], (0.3, 0, 0.24), 1)
        self.assertEqual(c["new"], [])  # the unreachable object is not offered again


def full_pick(cat, xy):
    return dict(pick(cat, xy), spread_dir=[0.0, 1.0, 0.0], place_spread_dir=[1.0, 0.0, 0.0])


class RouteTests(unittest.TestCase):
    """IK fix: reroute or skip instead of ending the run (closed_loop.choose_route)."""

    def setUp(self):
        from mm.control import closed_loop as CL

        self.CL = CL
        self.here = np.array([0.2, -0.24, 0.2])
        self.picks, self.places = CL.route_options(full_pick("cube", (0.35, 0.1)), self.here,
                                                   np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))

    def test_options(self):
        labels = [p[0] for p in self.picks]
        self.assertEqual(labels, ["direct", "wrist_flipped", "via_home", "via_home_wrist_flipped"])
        via = dict(self.picks)["via_home"]
        self.assertEqual(via[0][0], "reroute")
        np.testing.assert_allclose(via[0][1], self.here)
        np.testing.assert_allclose(via[1][1], via[0][2])  # approach starts at home
        flipped = dict(self.picks)["wrist_flipped"]
        np.testing.assert_allclose(flipped[0][6], [0.0, -1.0, 0.0])
        places = dict(self.places(dict(self.picks)["direct"]))
        carry = places["via_home"]
        self.assertEqual([s[0] for s in carry], ["carry", "carry", "lower", "release", "settle", "retreat"])
        np.testing.assert_allclose(carry[1][2], places["direct"][0][2])  # still ends above the bin

    def test_choose(self):
        ok = self.CL.choose_route(lambda segs: None, self.picks, self.places)
        self.assertEqual((ok["pick"], ok["place"], ok["failures"]), ("direct", "direct", []))
        self.assertEqual(ok["n_pick"], 5)

        def no_straight_approach(segs):  # the straight approach leaves the reach
            return ("approach", 30) if segs[0][0] == "approach" and segs[0][6][1] > 0 else None

        r = self.CL.choose_route(no_straight_approach, self.picks, self.places)
        self.assertEqual(r["pick"], "wrist_flipped")
        self.assertEqual(r["failures"], [{"part": "pick", "route": "direct", "segment": "approach", "frame": 30}])

        def no_direct_carry(segs):
            return ("carry", 12) if len(segs) > 5 and segs[5][0] == "carry" and segs[6][0] == "lower" else None

        r = self.CL.choose_route(no_direct_carry, self.picks, self.places)
        self.assertEqual((r["pick"], r["place"]), ("direct", "via_home"))

        r = self.CL.choose_route(lambda segs: ("descend", 3), self.picks, self.places)
        self.assertIsNone(r["segments"])  # nothing reaches: skip the object
        self.assertEqual(len(r["failures"]), 4)


class DeliberateMissTests(unittest.TestCase):
    def test_offset_clears_object_and_avoids_neighbours(self):
        from mm.control import closed_loop as CL

        p = {"category": "aaa_cell", "xy": [0.35, 0.10]}
        off = CL.miss_offset(p, [0.0, 1.0, 0.0], avoid=[(0.45, 0.10)])
        # across the jaws = along the cell's axis (x), past its end + half a finger
        self.assertAlmostEqual(abs(off[1]), 0.0)
        self.assertGreater(abs(off[0]), C.AAA_CELL_LENGTH / 2 + C.FINGER_WIDTH / 2)
        self.assertLess(off[0], 0)  # away from the neighbour at +x
        cube = CL.miss_offset({"category": "cube", "xy": [0.3, 0.0]}, [1.0, 0.0, 0.0], avoid=[])
        self.assertGreater(np.linalg.norm(cube), C.CUBE_EDGE / 2 + C.FINGER_WIDTH / 2)


class DemoLayoutTests(unittest.TestCase):
    def test_rules_hold(self):
        import math as m

        from mm.datagen import randomize as R
        from mm.eval import demo_layout as D

        layout, loop = D.build(3)
        objs = [dict(o, kind=k) for k, key in R.LAYOUT_KEYS.items() for o in layout[key]]
        self.assertEqual(sorted(o["kind"] for o in objs), ["aaa_cell", "aaa_cell", "cap", "cube", "cube", "screw", "vial"])
        for i, a in enumerate(objs):
            for b in objs[i + 1:]:
                self.assertGreaterEqual(m.dist(a["xy"], b["xy"]), D.MIN_CENTRES)
            for b in layout["bins"].values():
                self.assertGreaterEqual(R.bin_gap(R.footprint(a["kind"], a["xy"], a["yaw_deg"]), b), D.BIN_WALL_GAP)
            if a["kind"] not in D.DISTRACTORS:
                self.assertIsNone(D.target_ok(a, [o for o in objs if o is not a], layout["bins"]))
        self.assertEqual(loop["inject"]["deliberate_attempt"], 3)
        self.assertFalse([e for e in loop["events"] if e["type"] == "add"])
        self.assertTrue(all(len(s) > 0 for s in loop["events"][0]["spots"].values()))


def bin_det(xy, yaw=0.0, fit=0.95):
    return {"class": "bin", "xy_robot_m": list(xy), "yaw_deg": yaw, "fit_iou": fit, "conf": 0.9}


class BinTrackingTests(unittest.TestCase):
    """v4: identical bins, named by position, tracked by position."""

    def test_fewer_than_three_means_capture_again(self):
        self.assertIsNone(first_bins([bin_det((0.2, -0.24)), bin_det((0.32, -0.24))]))
        self.assertIsNone(first_bins([bin_det((0.2, -0.24)), bin_det((0.32, -0.24)), bin_det((0.44, -0.24), fit=0.6)]))
        dup = [bin_det((0.2, -0.24)), bin_det((0.21, -0.24), fit=0.9), bin_det((0.32, -0.24))]
        self.assertIsNone(first_bins(dup))  # two detections of one bin are one bin

    def test_named_by_position_and_never_swapped(self):
        dets = [bin_det((0.44, -0.24), 100.0), bin_det((0.20, -0.24)), bin_det((0.32, -0.24))]
        b = first_bins(dets)
        self.assertEqual([b[n]["xy"][0] for n in ("bin_1", "bin_2", "bin_3")], [0.20, 0.32, 0.44])
        self.assertAlmostEqual(b["bin_3"]["yaw_deg"], 10.0)
        self.assertEqual(b["bin_1"]["outer"], load_bins()["bin_1"]["outer"])
        # next capture: detections in another order, small pose noise, one bin hidden
        now = track_bins([bin_det((0.321, -0.239)), bin_det((0.203, -0.241))], b)
        self.assertEqual(now["bin_1"]["xy"], [0.203, -0.241])
        self.assertEqual(now["bin_2"]["xy"], [0.321, -0.239])
        self.assertFalse(now["bin_3"]["found"])
        self.assertEqual(now["bin_3"]["xy"], [0.44, -0.24])  # kept, not guessed


class BinAssignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mm.datagen.masks_to_yolo import nominal_camera

        cls.cam = nominal_camera()
        cls.reach = G.Reach()

    def test_shortest_travel_assignment(self):
        from mm.planning import bin_assign as BA
        from tests.test_grasp import make_det

        # bins in a row at y = -0.24; two cubes far left, screw and AAA cell to the right
        phys = load_bins()
        dets = [make_det(self.cam, 0, "cube", (0.25, 0.20), 10.0), make_det(self.cam, 1, "cube", (0.30, 0.22), 40.0),
                make_det(self.cam, 2, "screw", (0.42, 0.05), 80.0), make_det(self.cam, 3, "aaa_cell", (0.38, -0.10), 0.0)]
        assignment, log = BA.assign_bins({"objects": dets}, self.cam, phys, reach=self.reach)
        self.assertEqual(len(log["tried"]), 6)
        self.assertEqual(sorted(assignment.values()), ["bin_1", "bin_2", "bin_3"])
        self.assertEqual(log["picks"], 4)
        self.assertEqual(log["travel_m"], min(t["travel_m"] for t in log["tried"]))
        self.assertGreaterEqual(log["saving_m"], 0.0)
        self.assertEqual(log["worst_travel_m"], max(t["travel_m"] for t in log["tried"]))
        # the plan uses it: each class to its bin
        from mm.planning import plan_bins as PB

        plan = PB.build_plan({"objects": dets}, self.cam, bins=BA.table(phys, assignment), reach=self.reach)
        for p in plan["picks"]:
            self.assertEqual(p["place_xy"], list(phys[assignment[p["category"]]]["xy"]))

    def test_success_rule(self):
        from mm.control.executor import sort_ok

        self.assertTrue(sort_ok({"bin_1": ["cube", "cube"], "bin_2": ["screw"], "bin_3": []}, 3))
        self.assertFalse(sort_ok({"bin_1": ["cube", "screw"], "bin_2": [], "bin_3": []}, 2))  # mixed bin
        self.assertFalse(sort_ok({"bin_1": ["cube"], "bin_2": [], "bin_3": []}, 2))  # a target left out
        self.assertFalse(sort_ok({"bin_1": ["cube"], "bin_2": ["cap"], "bin_3": []}, 1))  # cap picked


if __name__ == "__main__":
    unittest.main()
