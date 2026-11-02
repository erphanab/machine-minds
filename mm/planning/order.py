"""
Job ordering for the bin sort (Machine Minds step B4). Host-side, no I/O.

Cost model -- TCP travel for chained execution (B5 makes sort_scene run picks
back to back instead of returning home after each one):

    home -> above object_1 -> down/up -> above its bin -> down/up
         -> above object_2 -> ... -> above bin_n -> home

Straight Cartesian lines between hover points, as sort_scene.segments
interpolates. Each pick's descents and its object -> bin leg are the same in
any order; only the home and bin -> next-object legs depend on it, so savings
are bounded by those legs (expect modest numbers; report them as measured).

Strategies:
  fixed            picks as given (detection order)
  greedy           from the current position, always the nearest object next
  greedy_2opt      greedy, then 2-opt segment reversals while any shortens the
                   exact tour cost (so it is never longer than greedy)
cheapest_insertion() adds a newly seen object to an existing order (B7).
"""

import itertools
import math
import time

from mm.planning.grasp import HOME_XYZ, HOVER_DZ

STRATEGIES = ("fixed", "greedy", "greedy_2opt")


def hover_points(pick):
    """(above object, above its bin) for one plan pick."""
    obj = (pick["xy"][0], pick["xy"][1], pick["pick_z"] + HOVER_DZ)
    dst = (pick["place_xy"][0], pick["place_xy"][1], pick["place_z"] + HOVER_DZ)
    return obj, dst


def _d(a, b):
    return math.dist(a, b)


def tour_cost(picks, seq=None, home=HOME_XYZ):
    """TCP path length (m) of executing picks in order seq (default: as given)."""
    seq = range(len(picks)) if seq is None else seq
    pts = [hover_points(picks[i]) for i in seq]
    if not pts:
        return 0.0
    total, here = 0.0, tuple(home)
    for obj, dst in pts:
        total += _d(here, obj) + 2 * HOVER_DZ + _d(obj, dst) + 2 * HOVER_DZ
        here = dst
    return total + _d(here, home)


def fixed(picks, home=HOME_XYZ):
    return list(range(len(picks)))


def greedy(picks, home=HOME_XYZ):
    """Nearest next object from wherever the tool is (home, then each bin)."""
    pts = [hover_points(p) for p in picks]
    left, seq, here = set(range(len(picks))), [], tuple(home)
    while left:
        # Ties broken by index so the order is reproducible.
        nxt = min(left, key=lambda i: (_d(here, pts[i][0]), i))
        seq.append(nxt)
        left.remove(nxt)
        here = pts[nxt][1]
    return seq


def two_opt(picks, seq, home=HOME_XYZ, eps=1e-9):
    """Reverse sub-sequences while any reversal lowers the exact tour cost."""
    seq, best = list(seq), tour_cost(picks, seq, home)
    improved = True
    while improved:
        improved = False
        for i, j in itertools.combinations(range(len(seq)), 2):
            cand = seq[:i] + seq[i : j + 1][::-1] + seq[j + 1 :]
            c = tour_cost(picks, cand, home)
            if c < best - eps:
                seq, best, improved = cand, c, True
    return seq


def greedy_2opt(picks, home=HOME_XYZ):
    return two_opt(picks, greedy(picks, home), home)


def order(picks, strategy="greedy_2opt", home=HOME_XYZ):
    """Returns (sequence of pick indices, predicted travel m, planning ms)."""
    fn = {"fixed": fixed, "greedy": greedy, "greedy_2opt": greedy_2opt}[strategy]
    t = time.perf_counter()
    seq = fn(picks, home)
    ms = (time.perf_counter() - t) * 1000
    return seq, tour_cost(picks, seq, home), ms


def compare(picks, home=HOME_XYZ):
    """All strategies on the same picks: {strategy: {sequence, travel_m, planning_ms}}."""
    out = {}
    for s in STRATEGIES:
        seq, cost, ms = order(picks, s, home)
        out[s] = {"sequence": seq, "travel_m": round(cost, 4), "planning_ms": round(ms, 3)}
    return out


def cheapest_insertion(picks, seq, new_index, home=HOME_XYZ):
    """Insert picks[new_index] where it adds least travel to the remaining order.

    For a newly seen object between picks (B7). Returns (new sequence, added m).
    """
    base = tour_cost(picks, seq, home)
    best = None
    for k in range(len(seq) + 1):
        cand = seq[:k] + [new_index] + seq[k:]
        added = tour_cost(picks, cand, home) - base
        if best is None or added < best[1] - 1e-12:
            best = (cand, added)
    return best
