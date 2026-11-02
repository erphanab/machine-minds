"""
A curated demo scene and closed-loop script (layout.json + loop.json). Host-side.

    .venv/bin/python -m mm.eval.demo_layout --seed 1 --out runs/demo_x
    .venv/bin/python -m mm.eval.run_trial --run runs/demo_x --layout runs/demo_x/layout.json \
        --loop-json runs/demo_x/loop.json --seed 1 --render

Seven objects: 2 cubes, 2 AAA cells, 1 screw (targets), 1 cap, 1 tube
(distractors), all lying flat, and three bins at random poses. Every rule is
checked here, before Isaac:
  - object centres >= MIN_CENTRES apart, footprints >= MIN_FOOTPRINT_GAP
  - every footprint >= BIN_WALL_GAP from every bin's outer wall
  - objects and bins inside the comfortable reach band (REACH_R from the base)
  - per target, every jaw direction the planner may choose: open fingers clear
    of all other objects and bins, and IK at hover, grasp, lift, above the
    target bin and release (grasp.Reach, the planner's own check)
loop.json: grasp check on, bins found by perception, one deliberate miss on
the third pick (a first try, never a retry), and one nudge to a spot that
passes the same rules (spots precomputed per target, re-checked against the
live table when the nudge runs). No objects added.
"""

import argparse
import json
import math
import os
import random

import numpy as np

import scene_config as C
from mm.control.executor import PRIMS
from mm.datagen import randomize as R
from mm.planning import grasp as G

MIN_CENTRES = 0.080
MIN_FOOTPRINT_GAP = 0.025  # a 140 mm tube can pass close to a neighbour 80 mm from its centre
BIN_WALL_GAP = 0.060
REACH_R = (0.25, 0.40)  # object centre to the robot base axis
BIN_REACH_R = (0.25, 0.42)
OBJ_BOX = (R.TARGET_X, R.TARGET_Y)  # where the training data put objects (camera view)
DIST_BOX = (R.DISTRACTOR_X, R.DISTRACTOR_Y)
TARGETS = ("cube", "cube", "aaa_cell", "aaa_cell", "screw")
DISTRACTORS = ("cap", "vial")
NUDGE_SPOTS = 6  # candidate destinations per target
NUDGE_MIN_MOVE = 0.05
DELIBERATE_ATTEMPT = 3
NUDGE_AFTER_PICKS = 3

_REACH = None


def reach():
    global _REACH
    if _REACH is None:
        _REACH = G.Reach()
    return _REACH


def capsule_poly(f, n=12):
    """Capsule footprint (a, b, r) as a polygon."""
    (a, b, r) = f
    d = np.subtract(b, a)
    ang = math.atan2(d[1], d[0]) if np.linalg.norm(d) > 0 else 0.0
    pts = []
    for centre, start in ((b, ang - math.pi / 2), (a, ang + math.pi / 2)):
        for k in range(n + 1):
            t = start + math.pi * k / n
            pts.append((centre[0] + r * math.cos(t), centre[1] + r * math.sin(t)))
    return np.array(pts)


def in_band(xy, band):
    return band[0] <= math.hypot(*xy) <= band[1]


def target_ok(obj, others, bins):
    """The planner's checks for a target at obj, with others on the table. v4: the robot picks
    each class's bin at run time, so IK must reach every bin (any assignment is then feasible)."""
    shapes = [("object", i, capsule_poly(R.footprint(o["kind"], o["xy"], o["yaw_deg"]))) for i, o in enumerate(others)]
    shapes += G.bin_shapes(bins)
    g = G.GRASP[obj["kind"]]
    for spread, _ in G.spread_options({"class": obj["kind"], "yaw_deg": obj["yaw_deg"]}):
        if G.finger_collisions(np.asarray(obj["xy"]), spread, shapes):
            return "finger_collision"
        for b in bins.values():
            if reach().check(obj["xy"], g["pick_z"], spread, b["xy"], C.SURFACE_DZ + b["release_z"], spread):
                return "ik"
    return None


def spot_ok(kind, xy, yaw, placed, bins):
    """Spacing, bin and reach-band rules for one object (not IK)."""
    f = R.footprint(kind, xy, yaw)
    box = DIST_BOX if kind in DISTRACTORS else OBJ_BOX
    if not (box[0][0] <= xy[0] <= box[0][1] and box[1][0] <= xy[1] <= box[1][1]):
        return False
    if kind not in DISTRACTORS and not in_band(xy, REACH_R):
        return False
    if any(R.bin_gap(f, b) < BIN_WALL_GAP for b in bins.values()):
        return False
    return all(
        math.dist(xy, o["xy"]) >= MIN_CENTRES
        and R.footprint_gap(f, R.footprint(o["kind"], o["xy"], o["yaw_deg"])) >= MIN_FOOTPRINT_GAP
        for o in placed
    )


def sample_bins(rng, tries=4000):
    bins = {}
    rad = R._bin_radius()
    for name, fixed in sorted(R.load_bins().items()):
        for _ in range(tries):
            xy = (rng.uniform(-0.1, 0.45), rng.uniform(-0.45, 0.45))
            if not in_band(xy, BIN_REACH_R):
                continue
            if any(math.dist(xy, b["xy"]) < 2 * rad + R.BIN_SEPARATION for b in bins.values()):
                continue
            if R.bin_pose_ok(xy):
                bins[name] = dict(fixed, xy=[float(xy[0]), float(xy[1])], yaw_deg=rng.uniform(0.0, 90.0))
                break
        else:
            return None
    return bins


def sample(seed, max_restarts=400, targets=TARGETS, distractors=DISTRACTORS, fixed_bins=False):
    rng = random.Random(seed)
    for _ in range(max_restarts):
        bins = R.load_bins() if fixed_bins else sample_bins(rng)
        if bins is None:
            continue
        placed = []
        for kind in tuple(distractors) + tuple(targets):  # long tube first
            box = DIST_BOX if kind in DISTRACTORS else OBJ_BOX
            for _ in range(3000):
                xy = (rng.uniform(*box[0]), rng.uniform(*box[1]))
                yaw = rng.uniform(-180.0, 180.0)
                if spot_ok(kind, xy, yaw, placed, bins):
                    placed.append({"kind": kind, "xy": [float(xy[0]), float(xy[1])], "yaw_deg": float(yaw)})
                    break
            else:
                break
        else:
            bad = [(o["kind"], target_ok(o, [p for p in placed if p is not o], bins))
                   for o in placed if o["kind"] not in DISTRACTORS]
            if all(r is None for _, r in bad):
                return placed, bins, rng
    raise ValueError(f"seed {seed}: no layout passes the rules")


def nudge_spots(placed, bins, rng):
    """Per target prim: destinations that pass every rule with the others where they start."""
    counters, spots = {}, {}
    for o in placed:
        k = o["kind"]
        idx = counters.get(k, 0)
        counters[k] = idx + 1
        if k in DISTRACTORS:
            continue
        prim = f"{PRIMS[k]}{idx}"  # cell.py numbers each kind in layout order
        others = [p for p in placed if p is not o]
        found = []
        for _ in range(4000):
            xy = [rng.uniform(*OBJ_BOX[0]), rng.uniform(*OBJ_BOX[1])]
            if math.dist(xy, o["xy"]) < NUDGE_MIN_MOVE:
                continue
            if not spot_ok(k, xy, o["yaw_deg"], others, bins):
                continue
            moved = dict(o, xy=xy)
            if target_ok(moved, others, bins):
                continue
            if any(p["kind"] not in DISTRACTORS and target_ok(p, [q for q in others if q is not p] + [moved], bins)
                   for p in others):
                continue
            found.append([round(xy[0], 4), round(xy[1], 4)])
            if len(found) == NUDGE_SPOTS:
                break
        spots[prim] = found
    return spots


def eval_kinds(seed):
    """Real-arm evaluation scene (submission/placement.yaml): 4-5 targets with
    every class at least once, and one distractor (cap or tube)."""
    rng = random.Random(seed * 31 + 7)
    n = rng.choice((4, 5))
    kinds = ["cube", "aaa_cell", "screw"] + [rng.choice(("cube", "aaa_cell", "screw")) for _ in range(n - 3)]
    return tuple(sorted(kinds)), (rng.choice(DISTRACTORS),)


def build(seed, kind="demo"):
    """kind "demo": the curated 7-object scene with random bins; "eval": the
    real-arm evaluation scene with the fixed configs/bins.yaml bins."""
    if kind == "eval":
        targets, distractors = eval_kinds(seed)
        placed, bins, rng = sample(seed, targets=targets, distractors=distractors, fixed_bins=True)
    else:
        placed, bins, rng = sample(seed)
    layout = {
        "seed": seed, "hard": False, "spacing_m": MIN_CENTRES, "demo": True,
        **{key: [] for key in R.LAYOUT_KEYS.values()},
        "bins": bins,
        "bin_roughness": random.Random(seed * 104729 + 3).uniform(*C.BIN_ROUGHNESS_RANGE),
        "lighting": {"scale": 1.0, "kelvin": 5500.0},
        "table_cam": {"eye": list(C.TABLE_CAM["eye"]), "look_at": list(C.TABLE_CAM["look_at"]), "up": list(C.TABLE_CAM["up"])},
        "arm": {"mode": "parked", "joints_deg": dict(C.CAPTURE_POSE_DEG)},
    }
    colours = list(R.CUBE_COLOURS)
    for o in placed:
        entry = {"xy": o["xy"], "yaw_deg": o["yaw_deg"]}
        if o["kind"] == "cube":
            entry["colour"] = rng.choice(colours)
        if o["kind"] == "cap":
            entry["orientation"] = "narrow_end_up"
        layout[R.LAYOUT_KEYS[o["kind"]]].append(entry)
    loop = {
        "grasp_check": True, "bins": "detected", "seed": seed, "max_cycles": 40,
        "inject": {"prob": 0.0, "deliberate_attempt": DELIBERATE_ATTEMPT},
        "events": [{"after_picks": NUDGE_AFTER_PICKS, "type": "nudge", "spots": nudge_spots(placed, bins, rng)}],
    }
    return layout, loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True, help="run folder: writes layout.json and loop.json")
    ap.add_argument("--kind", choices=["demo", "eval"], default="demo")
    a = ap.parse_args()
    layout, loop = build(a.seed, a.kind)
    os.makedirs(a.out, exist_ok=True)
    for name, data in (("layout.json", layout), ("loop.json", loop)):
        with open(os.path.join(a.out, name), "w") as f:
            json.dump(data, f, indent=2)
    print(f"[demo_layout] seed {a.seed}: bins {[(k, [round(v, 3) for v in b['xy']], round(b['yaw_deg'])) for k, b in layout['bins'].items()]}, "
          f"nudge spots {[len(v) for v in loop['events'][0]['spots'].values()]} -> {a.out}")


if __name__ == "__main__":
    main()
