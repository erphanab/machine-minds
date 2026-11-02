"""Host tests for mm/planning/order.py (step B4).

    .venv/bin/python -m unittest tests.test_order
"""

import itertools
import math
import random
import unittest

import scene_config as C
from mm.datagen import randomize as R
from mm.planning import order as O
from mm.planning.bin_assign import default_table
from mm.planning.grasp import HOME_XYZ, HOVER_DZ

BINS = default_table()  # v4: class -> one of the identical bins


def random_picks(rng, n):
    """Plan-shaped picks: objects in the sampling area, each bound for a bin."""
    picks = []
    for _ in range(n):
        b = BINS[rng.choice(sorted(BINS))]
        picks.append(
            {
                "xy": [rng.uniform(*R.TARGET_X), rng.uniform(*R.TARGET_Y)],
                "pick_z": C.SURFACE_DZ + C.TIP_CLEARANCE,
                "place_xy": list(b["xy"]),
                "place_z": C.SURFACE_DZ + b["release_z"],
            }
        )
    return picks


class OrderTests(unittest.TestCase):
    def test_two_opt_never_longer_than_greedy_on_100_cases(self):
        rng = random.Random(4)
        gains = []
        for case in range(100):
            picks = random_picks(rng, rng.randint(6, 10))
            g = O.tour_cost(picks, O.greedy(picks))
            t = O.tour_cost(picks, O.greedy_2opt(picks))
            self.assertLessEqual(t, g + 1e-9, case)
            gains.append(g - t)
        self.assertGreater(max(gains), 0.0)  # 2-opt does improve some cases

    def test_sequences_are_permutations(self):
        rng = random.Random(1)
        for _ in range(20):
            picks = random_picks(rng, rng.randint(1, 10))
            for s in O.STRATEGIES:
                seq, _, _ = O.order(picks, s)
                self.assertEqual(sorted(seq), list(range(len(picks))), s)
        self.assertEqual(O.fixed(random_picks(rng, 5)), [0, 1, 2, 3, 4])

    def test_greedy_2opt_close_to_brute_force_optimum(self):
        rng = random.Random(7)
        for _ in range(15):
            picks = random_picks(rng, rng.randint(4, 7))
            best = min(O.tour_cost(picks, p) for p in itertools.permutations(range(len(picks))))
            got = O.tour_cost(picks, O.greedy_2opt(picks))
            self.assertGreaterEqual(got, best - 1e-9)
            self.assertLessEqual(got, best * 1.05)  # within 5% of optimal on these sizes

    def test_tour_cost_arithmetic(self):
        b = BINS["cube"]
        p = {"xy": [0.30, 0.0], "pick_z": 0.0, "place_xy": list(b["xy"]), "place_z": 0.0}
        above_obj, above_bin = (0.30, 0.0, HOVER_DZ), (*b["xy"], HOVER_DZ)
        expected = (
            math.dist(HOME_XYZ, above_obj) + 2 * HOVER_DZ + math.dist(above_obj, above_bin)
            + 2 * HOVER_DZ + math.dist(above_bin, HOME_XYZ)
        )
        self.assertAlmostEqual(O.tour_cost([p]), expected, places=12)
        self.assertEqual(O.tour_cost([]), 0.0)

    def test_order_independent_part_is_constant(self):
        """Descents and object -> bin legs are the same in every order."""
        rng = random.Random(3)
        picks = random_picks(rng, 6)
        fixed_part = sum(4 * HOVER_DZ + math.dist(*O.hover_points(p)) for p in picks)
        for seq in itertools.islice(itertools.permutations(range(6)), 50):
            self.assertGreaterEqual(O.tour_cost(picks, seq), fixed_part)

    def test_cheapest_insertion(self):
        rng = random.Random(5)
        for _ in range(30):
            picks = random_picks(rng, rng.randint(3, 8))
            new = len(picks) - 1
            seq = O.greedy_2opt(picks[:-1])
            got, added = O.cheapest_insertion(picks, seq, new)
            self.assertEqual(sorted(got), list(range(len(picks))))
            appended = O.tour_cost(picks, seq + [new]) - O.tour_cost(picks, seq)
            self.assertLessEqual(added, appended + 1e-12)
            self.assertAlmostEqual(O.tour_cost(picks, got) - O.tour_cost(picks, seq), added, places=12)

    def test_deterministic(self):
        picks = random_picks(random.Random(9), 9)

        def strip(c):  # planning time varies run to run
            return {s: (v["sequence"], v["travel_m"]) for s, v in c.items()}

        self.assertEqual(strip(O.compare(picks)), strip(O.compare(picks)))


if __name__ == "__main__":
    unittest.main()
