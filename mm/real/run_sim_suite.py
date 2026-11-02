"""
policy.py on N evaluation layouts in the twin (layout -> capture -> sim_eval). Host python:

    .venv/bin/python -m mm.real.run_sim_suite --seeds 1 2 3 4 5 --out runs/real_sim

Layouts: mm.eval.demo_layout "eval" scenes (4-5 targets, every class, 1 distractor, >= 80 mm apart, fixed
configs/bins.yaml bins), the scenes planned for the real-arm evaluation. Prints sorted / failed per layout and
writes <out>/summary.json.
"""

import argparse
import json
import os
import subprocess
import sys
import time

from mm.eval import demo_layout as D
from mm.eval.run_trial import interp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "real_sim"))
    ap.add_argument("--client", default=os.path.join(os.path.dirname(ROOT), "PhysicalAI-Hackathon-2026-v6"))
    ap.add_argument("--arm-speed", type=float, default=None, help="sim_eval --arm-speed (m/s; the real cell: ~0.05)")
    ap.add_argument("--brain-root", default=None, help="sim_eval --brain-root")
    ap.add_argument("--weights", default=None, help="sim_eval --weights")
    ap.add_argument("--track-moves", action="store_true", help="sim_eval --track-moves")
    a = ap.parse_args()
    extra = [f for k, v in (("--arm-speed", a.arm_speed), ("--brain-root", a.brain_root), ("--weights", a.weights))
             if v is not None for f in (k, str(v))] + (["--track-moves"] if a.track_moves else [])
    isaac = interp("ISAAC_PY")
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES")
    rows = []
    for seed in a.seeds:
        run = os.path.abspath(os.path.join(a.out, f"layout_{seed}"))
        os.makedirs(run, exist_ok=True)
        layout, _ = D.build(seed, "eval")
        with open(os.path.join(run, "layout.json"), "w") as f:
            json.dump(layout, f, indent=2)
        t = time.time()
        with open(os.path.join(run, "capture.log"), "w") as log:
            subprocess.call([isaac, "capture_scene.py", "--out", run], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env=dict(env, MM_SCENE_JSON=os.path.join(run, "layout.json")))
        with open(os.path.join(run, "sim_eval.log"), "w") as log:
            subprocess.call([isaac, os.path.join("mm", "real", "sim_eval.py"), "--run", run, "--client", a.client, *extra],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        p = os.path.join(run, "sim_eval.json")
        r = json.load(open(p)) if os.path.exists(p) else {"fault": "no sim_eval.json"}
        line = {"seed": seed, "targets": r.get("n_targets"), "sorted": r.get("sorted"), "success": r.get("success"),
                "distractor_moved_mm": {k: v["moved_mm"] for k, v in (r.get("distractors") or {}).items()},
                "fault": r.get("fault"), "acts": len(r.get("acts", [])), "picks": r.get("picks"),
                "table_moves_mm": r.get("table_moves_mm"),
                "minutes": round((time.time() - t) / 60, 1)}
        rows.append(line)
        print(f"[suite] layout {seed}: sorted {line['sorted']}/{line['targets']} success {line['success']} "
              f"picks {line['picks']} distractor {line['distractor_moved_mm']} table moves {line['table_moves_mm']} "
              f"fault {line['fault']} "
              f"({line['minutes']} min)", flush=True)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[suite] {sum(bool(r['success']) for r in rows)}/{len(rows)} layouts fully sorted")


if __name__ == "__main__":
    sys.exit(main())
