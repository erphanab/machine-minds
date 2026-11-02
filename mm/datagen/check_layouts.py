"""
Load N randomized layouts with bins in Isaac and check them (Machine Minds B1).

    $ISAAC/python.sh mm/datagen/check_layouts.py --seeds 20 --out runs/b1_layouts

One Isaac process, a fresh stage per layout. For each: every object and bin
prim exists, and after one second of physics no cube or cap has been pushed
more than --max-drift-mm (it would mean it was placed in contact with a bin or
another object). Lying AAA cells and tubes roll freely; their roll is reported,
not judged. The first layout also writes a
table-camera render (table_<seed>.png) to look at. Writes report.json.
"""

import argparse
import importlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, default=20)
parser.add_argument("--first-seed", type=int, default=800_000)
parser.add_argument("--out", default=os.path.join(ROOT, "runs", "b1_layouts"))
parser.add_argument("--max-drift-mm", type=float, default=3.0)
parser.add_argument("--render", type=int, default=1, help="render this many layouts")
args = parser.parse_args()
args.out = os.path.abspath(args.out)
os.makedirs(args.out, exist_ok=True)

from mm.datagen import randomize as R  # noqa: E402

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import RigidPrim  # noqa: E402

import scene_config as C  # noqa: E402
import cell  # noqa: E402

PREFIX = {"cubes": "SortCube", "caps": "SortCap", "vials": "SortVial", "aaa_cells": "SortAAACell", "screws": "Screw"}
results = []
for k in range(args.seeds):
    seed = args.first_seed + k
    layout = R.sample_layout(seed)  # with bins, no drops: scene_config ignores drops
    path = os.path.join(args.out, f"layout_{seed}.json")
    with open(path, "w") as f:
        json.dump(layout, f, indent=2)
    os.environ["MM_SCENE_JSON"] = path
    importlib.reload(C)  # cell holds the same module object, so it sees the layout

    omni.usd.get_context().new_stage()
    World.clear_instance()
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    cell.build(stage)
    cell.enable_ccd(world)
    world.reset()

    missing = [f"/World/Bin_{b}" for b in layout["bins"] if not stage.GetPrimAtPath(f"/World/Bin_{b}").IsValid()]
    paths, want = [], []
    for key, prefix in PREFIX.items():
        for i, o in enumerate(layout[key]):
            p = f"/World/{prefix}{i}"
            if not stage.GetPrimAtPath(p).IsValid():
                missing.append(p)
                continue
            paths.append(p)
            want.append(o["xy"])
    bodies = RigidPrim(paths, name=f"chk{seed}")
    bodies.initialize(world.physics_sim_view)
    start = np.asarray(bodies.get_world_poses()[0])[:, :2].copy()
    for _ in range(60):
        world.step(render=False)
    end = np.asarray(bodies.get_world_poses()[0])[:, :2]
    drift = np.linalg.norm(end - start, axis=1) * 1000
    # Lying cylinders (AAA cells, tubes) roll freely, so only cubes and caps
    # show a bad placement (an object started in contact with a bin or another).
    fixed = np.array([("SortCube" in p or "SortCap" in p) for p in paths])
    rolls = ~fixed
    rec = {
        "seed": seed,
        "objects": len(paths),
        "bins": len(layout["bins"]),
        "missing": missing,
        "max_drift_mm": float(drift[fixed].max()) if fixed.any() else 0.0,
        "worst": paths[int(np.argmax(np.where(fixed, drift, -1)))] if fixed.any() else None,
        "max_cylinder_roll_mm": float(drift[rolls].max()) if rolls.any() else 0.0,
    }
    rec["ok"] = not missing and rec["max_drift_mm"] <= args.max_drift_mm
    if k < args.render:
        rp = rep.create.render_product(cell.TABLE_CAM_PATH, tuple(2 * v for v in C.TABLE_CAM["resolution"]))
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach(rp)
        for _ in range(60):
            world.step(render=True)
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=False)
        img = np.asarray(ann.get_data())[:, :, :3]
        rec["render"] = f"table_{seed}.png"
        cv2.imwrite(os.path.join(args.out, rec["render"]), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        ann.detach()
        rp.destroy()
    print(f"[check_layouts] seed {seed}: {rec['objects']} objects, {rec['bins']} bins, "
          f"cube/cap drift max {rec['max_drift_mm']:.2f} mm, cylinder roll max "
          f"{rec['max_cylinder_roll_mm']:.1f} mm {'OK' if rec['ok'] else 'FAIL ' + str(rec)}", flush=True)
    results.append(rec)

summary = {"layouts": len(results), "ok": sum(r["ok"] for r in results), "results": results}
with open(os.path.join(args.out, "report.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"[check_layouts] {summary['ok']}/{summary['layouts']} layouts OK -> {args.out}", flush=True)
app.close()
