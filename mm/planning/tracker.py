"""
Object tracking across re-captures (Machine Minds step B7). Host-side.

After every pick the table is captured again. The tracker keeps an identity
for each target and a pick order across those captures:

  matched   a detection of the same class within MATCH_RADIUS of a track
            (closest pairs first): the track takes the new pose; it moved if
            the shift exceeds MOVED_MM
  new       an unmatched detection: a new track, put into the remaining order
            by cheapest insertion (order.cheapest_insertion)
  vanished  an unmatched track: dropped (picked, or knocked off the table)

Only eligible, planned picks are tracked; the planner decides pickability
every cycle. B6 attempts are counted per track; a track that fails
MAX_ATTEMPTS times is flagged and never offered again, as is one the arm has
no IK path to (give_up).
"""

import math

from mm.planning import order as O

MATCH_RADIUS = 0.025  # m
MOVED_MM = 3.0
MAX_ATTEMPTS = 3


class Tracker:
    def __init__(self):
        self.tracks = {}  # id -> {"pick": plan pick, "attempts": int, "history": [...]}
        self.order = []  # remaining track ids, next first
        self.flagged = {}  # id -> last pick, given up after MAX_ATTEMPTS
        self.done = []  # ids sorted into a bin
        self._next = 0
        self.log = []

    def _new_id(self):
        self._next += 1
        return f"T{self._next}"

    def update(self, picks, here, cycle):
        """Match this cycle's planned picks to tracks; returns the change summary."""
        alive = {tid: t for tid, t in self.tracks.items()}
        pairs = []
        for k, p in enumerate(picks):
            for tid, t in alive.items():
                if t["pick"]["category"] != p["category"]:
                    continue
                d = math.dist(t["pick"]["xy"], p["xy"])
                if d <= MATCH_RADIUS:
                    pairs.append((d, k, tid))
        used_p, used_t, moved = set(), set(), []
        for d, k, tid in sorted(pairs):
            if k in used_p or tid in used_t:
                continue
            used_p.add(k)
            used_t.add(tid)
            if d * 1000 > MOVED_MM:
                moved.append(tid)
            self.tracks[tid]["pick"] = picks[k]
            self.tracks[tid]["history"].append((cycle, picks[k]["xy"]))
        vanished = [tid for tid in self.tracks if tid not in used_t]
        for tid in vanished:
            del self.tracks[tid]
            if tid in self.order:
                self.order.remove(tid)
        new = []
        for k, p in enumerate(picks):
            if k in used_p:
                continue
            if any(math.dist(p["xy"], f["xy"]) <= MATCH_RADIUS and p["category"] == f["category"] for f in self.flagged.values()):
                continue  # a flagged object seen again: still given up
            tid = self._new_id()
            self.tracks[tid] = {"pick": p, "attempts": 0, "history": [(cycle, p["xy"])]}
            new.append(tid)
        # First capture: order everything with the planner's order; later:
        # cheapest insertion of the new ones into the remaining order.
        if cycle == 0:
            ids = list(self.tracks)
            seq, _, _ = O.order([self.tracks[i]["pick"] for i in ids], "greedy_2opt", here)
            self.order = [ids[i] for i in seq]
        else:
            for tid in new:
                ids = self.order + [tid]
                pk = [self.tracks[i]["pick"] for i in ids]
                seq, _ = O.cheapest_insertion(pk, list(range(len(self.order))), len(self.order), here)
                self.order = [ids[i] for i in seq]
        change = {"cycle": cycle, "new": new, "moved": moved, "vanished": vanished, "order": list(self.order)}
        self.log.append(change)
        return change

    def next(self):
        return (self.order[0], self.tracks[self.order[0]]["pick"]) if self.order else (None, None)

    def attempt_failed(self, tid):
        """Count a failed attempt; True if the track is now given up."""
        t = self.tracks[tid]
        t["attempts"] += 1
        if t["attempts"] >= MAX_ATTEMPTS:
            self.flagged[tid] = t["pick"]
            del self.tracks[tid]
            self.order.remove(tid)
            return True
        return False

    def give_up(self, tid):
        """Flag a track now (no arm path to it); never offered again."""
        self.flagged[tid] = self.tracks.pop(tid)["pick"]
        if tid in self.order:
            self.order.remove(tid)

    def picked(self, tid):
        self.done.append(tid)
        self.tracks.pop(tid, None)
        if tid in self.order:
            self.order.remove(tid)
