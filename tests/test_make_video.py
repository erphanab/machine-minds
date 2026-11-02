"""Host tests for mm/eval/make_video.py: captions and totals follow the motion."""

import os
import tempfile
import unittest

from mm.eval import make_video as M

NAMES = ("approach", "descend", "close", "dwell", "lift", "carry", "lower", "release", "settle", "retreat")


def segs(names, n=4):
    return [{"segment": s, "frames": n + 1} for s in names]  # PathLog counts the start sample too


def attempt(cycle, track, lifted, taken, placed, names=NAMES):
    return {"cycle": cycle, "track": track, "attempt": 1, "category": "cube", "bin": "cube", "injected_offset_mm": 0,
            "gt_lifted": lifted, "verdict_taken": taken, "placement_verified": placed, "segments": segs(names)}


class TimelineTests(unittest.TestCase):
    def test_closed_loop_totals_change_when_outcome_known(self):
        report = {
            "mode": "closed_loop", "targets_initial": 2, "cycles": [],
            "events": [{"type": "add", "prim": "SortCube3", "kind": "cube", "cycle": 1}],
            "picks": [
                attempt(0, "T1", False, False, False, NAMES[:5] + ("open",)),  # caught miss
                attempt(1, "T1", True, True, True),
            ],
        }
        tl = M._closed_loop_timeline(report)
        self.assertEqual(len(tl), 4 * (6 + 10))
        caps = [c for c, _ in tl]
        # first attempt: the miss is counted only after the lift
        lift_end = 4 * 5
        self.assertEqual(tl[lift_end - 1][1]["misses"], 0)
        self.assertEqual(tl[lift_end][1]["misses"], 1)
        self.assertEqual(tl[lift_end][1]["caught"], 1)
        self.assertIn("still on the table", caps[lift_end])
        # second attempt: "placed" only from settle on, sorted counted after release
        start2 = 4 * 6
        release = start2 + 4 * NAMES.index("release")
        self.assertFalse(any("placed" in c for c in caps[:release + 4]))
        self.assertTrue(all("placed" in c for c in caps[release + 4:]))
        self.assertEqual(tl[release + 3][1]["sorted"], 0)
        self.assertEqual(tl[release + 4][1]["sorted"], 1)
        # the added target appears in the total from its cycle on
        self.assertEqual(tl[0][1]["targets"], 2)
        self.assertEqual(tl[start2][1]["targets"], 3)
        self.assertIn("added mid-run", caps[start2])

    def test_open_loop(self):
        report = {"picks_planned": 1, "picks": [{"category": "screw", "bin": "screw", "pickup_verified": True,
                                                 "placement_verified": False, "segments": segs(NAMES)}]}
        tl = M._open_loop_timeline(report)
        self.assertEqual(len(tl), 40)
        self.assertIn("not in the bin", tl[-1][0])
        self.assertEqual(tl[-1][1], {"planned": 1, "picked": 1, "placed": 0})

    def test_duplicate_frac(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i, data in enumerate((b"a", b"a", b"b", b"c")):
                p = os.path.join(d, f"f{i}.png")
                with open(p, "wb") as f:
                    f.write(data)
                paths.append(p)
            self.assertAlmostEqual(M.duplicate_frac(paths), 1 / 3)


if __name__ == "__main__":
    unittest.main()
