"""
detections.json -> sort_plan.json for the bin sort (Machine Minds step B3).

Host-side. Reads perceive_yolo.py output, configs/classes.yaml (class -> bin)
and configs/bins.yaml, chooses a top-down grasp per eligible object
(mm/planning/grasp.py) and writes the plan sort_scene.py executes:

    .venv/bin/python -m mm.planning.plan_bins --run runs/<id>

Each pick carries the fields sort_scene.segments() reads -- xy, pick_z,
place_xy, place_z, spread_dir, place_spread_dir, category -- plus the
detection's id, class, confidence, pose fit and the chosen grasp. Objects that
cannot be picked go to "skipped" with a skip_reason (perception's, or
no_bin_for_class / exceeds_jaw_opening / finger_collision / ik_unreachable).
Picks are ordered by --order (mm/planning/order.py, B4; default greedy_2opt);
the plan also records every strategy's predicted travel for benchmarking.
No ground truth is read.
"""

import argparse
import json
import os
import time

import yaml

import scene_config as C
from backproject import CameraModel
from mm.datagen import randomize as R
from mm.planning import grasp as G
from mm.planning import order as O

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE = "MachineMinds-plan_bins"


def load_yaml(name, key):
    with open(os.path.join(ROOT, "configs", name)) as f:
        return yaml.safe_load(f)[key]


def build_plan(det, cam, classes=None, bins=None, reach=None, strategy="greedy_2opt"):
    classes = classes or {c["name"]: c for c in load_yaml("classes.yaml", "classes")}
    bins = bins or R.load_bins()  # B8 passes the bins perception found instead
    reach = reach or G.Reach()
    assignment_log = None
    slots = {c.get("bin") for c in classes.values() if c.get("role") == "target"}
    if not slots <= set(bins):  # v4: physical bins (bin_1..3); the robot assigns them
        from mm.planning.bin_assign import assign_bins, table

        assignment, assignment_log = assign_bins(det, cam, bins, classes, reach, strategy)
        bins = table(bins, assignment)
    dets = det["objects"]
    picks, skipped = [], []
    for d in dets:
        base = {
            "detection_id": d["id"],
            "category": d["class"],
            "conf": d["conf"],
            "xy": d["xy_robot_m"],
            "yaw_deg": d["yaw_deg"],
            "fit_iou": d["fit_iou"],
        }
        for extra in ("colour", "orientation"):
            if extra in d:
                base[extra] = d[extra]
        if not d["eligible"]:
            skipped.append(dict(base, skip_reason=d["skip_reason"], stage="perception"))
            continue
        bin_name = classes.get(d["class"], {}).get("bin")
        if bin_name not in bins:
            skipped.append(dict(base, skip_reason="no_bin_for_class", stage="planning"))
            continue
        if d.get("grasp") == "outline":  # v6 cubes and screws: in or touching a bin is already sorted or unsafe
            near = G.touches_bin(d, cam, bins)
            if near:
                skipped.append(dict(base, skip_reason="in_or_touching_bin", stage="planning", detail={"bin": near}))
                continue
        chosen, why = G.plan_grasp(d, dets, cam, bins, bin_name, reach)
        if chosen is None:
            reason, detail = why
            skipped.append(dict(base, skip_reason=reason, stage="planning", detail=detail))
            continue
        picks.append(dict(base, bin=bin_name, **chosen))
    ordering = O.compare(picks)
    seq = ordering[strategy]["sequence"]
    picks = [dict(picks[i], order_index=k) for k, i in enumerate(seq)]
    return {
        "source": SOURCE,
        "task": "bin_sort",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "perception": det.get("model"),
        "gripper_model": C.GRIPPER_MODEL,
        "open_gap_m": 2 * C.JAW_OPEN,  # sort_scene checks this against the twin
        "tip_open_gap_m": G.TIP_OPEN_GAP,
        "bins": bins,
        "bin_assignment": assignment_log,
        "order": strategy,
        "predicted_travel_m": ordering[strategy]["travel_m"],
        "ordering": ordering,  # every strategy on these picks; sequences index detection order
        "picks": picks,
        "skipped": skipped,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--det", default="detections.json")
    parser.add_argument("--out", default="sort_plan.json")
    parser.add_argument("--order", choices=O.STRATEGIES, default="greedy_2opt")
    args = parser.parse_args()
    with open(os.path.join(args.run, args.det)) as f:
        det = json.load(f)
    cam = CameraModel(os.path.join(args.run, "camera.json"))
    t = time.time()
    plan = build_plan(det, cam, strategy=args.order)
    plan["planning_ms"] = round((time.time() - t) * 1000)
    with open(os.path.join(args.run, args.out), "w") as f:
        json.dump(plan, f, indent=2)
    travel = ", ".join(f"{s} {v['travel_m']:.3f} m" for s, v in plan["ordering"].items())
    print(f"[plan_bins] {len(plan['picks'])} picks, {len(plan['skipped'])} skipped "
          f"({plan['planning_ms']} ms), order {args.order}; travel {travel} "
          f"-> {os.path.join(args.run, args.out)}")
    for s in plan["skipped"]:
        print(f"  skip {s['detection_id']:2d} {s['category']:8s} {s['skip_reason']}")


if __name__ == "__main__":
    main()
