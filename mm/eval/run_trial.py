"""
One sim trial: layout -> capture -> perceive -> plan -> execute (Machine Minds).

Cross-platform replacement for run_sort_pipeline.sh for the bin sort. Runs each
stage with its own interpreter, from environment variables (defaults are this
repo's Windows setup, docs/setup_notes.md):

    ISAAC_PY  Isaac Sim python      (C:\\isim\\env\\python.exe)
    YOLO_PY   YOLO env python        (C:\\isim\\yolo\\python.exe)
    HOST_PY   host .venv python      (this interpreter)

    python -m mm.eval.run_trial --seed 930000 --order greedy_2opt [--render]

Writes runs/trial_<seed>_<order>/: layout.json, capture outputs, detections,
sort_plan.json, execution_report.json and trial.json (seed, strategy, model
card, per-stage times, outcome summary). Ground truth is for scoring only.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULTS = {
    "ISAAC_PY": r"C:\isim\env\python.exe" if platform.system() == "Windows" else "/isaacsim/python.sh",
    "YOLO_PY": r"C:\isim\yolo\python.exe" if platform.system() == "Windows" else "python",
    "HOST_PY": sys.executable,
}


def interp(name):
    return os.environ.get(name, DEFAULTS[name])


def stage(name, cmd, run, env=None):
    t = time.time()
    with open(os.path.join(run, f"{name}.log"), "w") as log:
        code = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    return {"stage": name, "exit": code, "seconds": round(time.time() - t, 1)}


def write_loop_config(run, layout_path, args):
    """run/loop.json for the closed loop (sim_backend -> closed_loop)."""
    import random

    from mm.control.executor import PRIMS
    from mm.datagen import randomize as R

    cfg = {"grasp_check": True, "bins": "detected" if args.loop in ("b8", "all") else "config", "seed": args.seed,
           "max_cycles": 60, "inject": {"prob": 0.0}}
    if args.loop in ("b6", "all"):
        cfg["inject"] = {"prob": 0.5 if args.inject_prob is None else args.inject_prob,
                         "offset_mm": list(args.inject_mm or (3.0, 5.0))}
    if args.loop in ("b7", "all"):
        # Spares parked off the table (behind the table camera), dropped onto
        # free spots part-way through; plus one target nudged 35 mm.
        with open(layout_path) as f:
            lay = json.load(f)
        rng = random.Random(args.seed + 99)
        placed = [
            {"kind": kind, "xy": o["xy"], "yaw_deg": o["yaw_deg"]}
            for kind, key in R.LAYOUT_KEYS.items() for o in lay[key]
        ]
        events = []
        for i, (kind, after) in enumerate((("screw", 2), ("aaa_cell", 2), ("cube", 4))):
            obj = R._place(rng, kind, placed, 0.04, (R.TARGET_X, R.TARGET_Y), list(lay["bins"].values()))
            if obj is None:
                continue
            placed.append(obj)
            key = R.LAYOUT_KEYS[kind]
            spare = {"xy": [2.0 + 0.2 * i, 1.6], "yaw_deg": 0.0}
            if kind == "cube":
                spare["colour"] = "blue"
            events.append({"after_picks": after, "type": "add", "prim": f"{PRIMS[kind]}{len(lay[key])}",
                           "kind": kind, "xy": [float(v) for v in obj["xy"]], "yaw_deg": float(obj["yaw_deg"])})
            lay[key].append(spare)
        events.append({"after_picks": 3, "type": "nudge", "dist_mm": 35})
        lay["spares"] = [e["prim"] for e in events if e["type"] == "add"]
        with open(layout_path, "w") as f:
            json.dump(lay, f, indent=2)
        cfg["events"] = events
    with open(os.path.join(run, "loop.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--order", default="greedy_2opt", choices=["fixed", "greedy", "greedy_2opt"])
    parser.add_argument("--run", default=None, help="default runs/trial_<seed>_<order>")
    parser.add_argument("--render", action="store_true", help="render frames (slow); default physics only")
    parser.add_argument("--hard", action="store_true")
    parser.add_argument(
        "--loop", choices=["none", "b6", "b7", "b8", "all"], default="none",
        help="none: open-loop plan (B5); b6: grasp check + retries with injected misses; "
        "b7: + objects added/nudged mid-run; b8: + bins placed at random and found by perception; "
        "all: b6 + b7 + b8 in one run",
    )
    parser.add_argument("--inject-prob", type=float, default=None, help="b6: share of grasps offset on purpose")
    parser.add_argument("--inject-mm", type=float, nargs=2, default=None, help="b6: offset range, mm")
    parser.add_argument("--layout", default=None, help="use this layout.json instead of randomize.py")
    parser.add_argument("--loop-json", default=None, help="use this loop.json (closed loop) instead of --loop")
    args = parser.parse_args()
    run = os.path.abspath(args.run or os.path.join(ROOT, "runs", f"trial_{args.seed}_{args.order}"))
    os.makedirs(run, exist_ok=True)
    if os.path.exists(os.path.join(run, "execution_report.json")):
        raise SystemExit(f"{run} already has a result; use a fresh run directory")

    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES")
    env.pop("MM_SCENE_JSON", None)
    host, isaac, yolo = interp("HOST_PY"), interp("ISAAC_PY"), interp("YOLO_PY")
    env["YOLO_PY"] = yolo  # the closed loop starts the perception server with it
    layout = os.path.join(run, "layout.json")
    bins_mode = "random" if args.loop in ("b8", "all") else "fixed"
    if args.layout:  # a prepared scene (mm.eval.demo_layout)
        import shutil

        if os.path.abspath(args.layout) != layout:
            shutil.copyfile(args.layout, layout)
        bins_mode = "given"
        stages = [{"stage": "layout", "exit": 0, "seconds": 0.0, "from": args.layout}]
    else:
        stages = [stage("layout", [host, "-m", "mm.datagen.randomize", "--seed", str(args.seed), "--out", layout,
                                   "--bins", bins_mode] + (["--hard"] if args.hard else []), run, env)]
    if args.loop_json:  # a prepared closed-loop script
        with open(args.loop_json) as f:
            cfg = json.load(f)
        with open(os.path.join(run, "loop.json"), "w") as f:
            json.dump(cfg, f, indent=2)
    elif args.loop != "none" and stages[-1]["exit"] == 0:
        write_loop_config(run, layout, args)
    steps = [
        ("capture", [isaac, "capture_scene.py", "--out", run], dict(env, MM_SCENE_JSON=layout)),
        ("perceive", [yolo, os.path.join("mm", "perception", "perceive_yolo.py"), "--run", run], env),
        ("plan", [host, "-m", "mm.planning.plan_bins", "--run", run, "--order", args.order], env),
        ("execute", [isaac, "sort_scene.py", "--run", run] + ([] if args.render else ["--physics-only"]), env),
    ]
    for name, cmd, e in steps:
        if stages[-1]["exit"] != 0:
            break
        stages.append(stage(name, cmd, run, e))

    def load(name):
        p = os.path.join(run, name)
        return json.load(open(p)) if os.path.exists(p) else None

    plan, report, det = load("sort_plan.json"), load("execution_report.json"), load("detections.json")
    lay = load("layout.json") or {}
    spares = len(lay.get("spares", []))  # b7: parked off-table until added mid-run
    targets = sum(len(lay.get(k, [])) for k in ("cubes", "aaa_cells", "screws")) - spares
    trial = {
        "seed": args.seed,
        "hard": args.hard,
        "loop": args.loop,
        "bins": bins_mode,
        "strategy": args.order,
        "git_commit": None,  # git is not installed on the development PC
        "model": det.get("model") if det else None,
        "stages": stages,
        "targets_in_scene": targets,
        "picks_planned": len(plan["picks"]) if plan else None,
        "skipped": [{k: s.get(k) for k in ("category", "skip_reason")} for s in plan["skipped"]] if plan else None,
        "predicted_travel_m": plan.get("predicted_travel_m") if plan else None,
    }
    if report:
        trial.update(
            {k: report[k] for k in ("task_complete", "verified_pickups", "verified_placements", "cycle_time_s",
                                    "tcp_path_measured_m", "abort_reason", "picks_attempted")}
        )
        trial["per_pick"] = [
            {k: e.get(k) for k in ("order_index", "cycle", "track", "attempt", "category", "injected_offset_mm",
                                    "pickup_verified", "verdict_taken", "placement_verified",
                                    "perception_error_mm", "time_s", "tcp_path_measured_m", "others_disturbed")}
            for e in report["picks"]
        ]
        for k in ("mode", "grasp_check", "events", "targets_added", "targets_left_on_table", "sorted_into_bins", "flagged",
                  "ik", "render_check"):
            if k in report:
                trial[k] = report[k]
    with open(os.path.join(run, "trial.json"), "w") as f:
        json.dump(trial, f, indent=2)
    ok = report and report["task_complete"]
    render = report.get("render_check") if report else None
    if render and not render["ok"]:
        print(f"[run_trial] RENDER FAILED: identical consecutive frames {render['duplicate_frac']} "
              f"(max {render['max_duplicate_frac']})")
        sys.exit(2)
    if report and report.get("mode") == "closed_loop":
        summary = (f"{report['sorted_into_bins']} sorted of {targets}+{report['targets_added']} added targets, "
                   f"{len(report['targets_left_on_table'])} left, {len(report['picks'])} attempts, "
                   f"check {report['grasp_check']}, IK rerouted {report['ik']['rerouted']} skipped "
                   f"{report['ik']['skipped']}, cycle {report['cycle_time_s']} s")
    elif report:
        summary = (f"{trial.get('verified_placements')}/{trial['picks_planned']} placed of {targets} targets, "
                   f"cycle {trial.get('cycle_time_s')} s")
    else:
        summary = f"stopped at {stages[-1]['stage']}"
    print(f"[run_trial] seed {args.seed} {args.order}: {'OK' if ok else 'INCOMPLETE'} -- {summary} -> {run}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
