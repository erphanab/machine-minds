"""
The organizers' client policy.py, run against the twin with the cell's observation format. Isaac Sim python:

    C:\\isim\\env\\python.exe mm/real/sim_eval.py --run runs/real_sim/l1 --client ../PhysicalAI-Hackathon-2026-v6

The run folder needs layout.json and capture_scene.py's scene_state.json and camera.json. Each step, like the
cell: render the table (1920x1080) and wrist cameras (1280x720, resized to IMAGE_SIZE), build obs (RGB images,
joints in the twin's degrees + gripper 0/1, tcp_pose in UF0 mm + w, p, r), call policy.act(obs), and play the
returned UF0 rows, one per DT, with the twin's IK (straight line within a row, wrist turned to the row's R). An
unreachable row is a fault and ends the rollout, as on the cell. Stops after two holds in a row or --max-acts.
--arm-speed plays the rows the way the real arm did: at most that fast toward each row, stopping where it is
when they run out, the gripper acting on the row's time wherever the arm has got to.

The policy perceives through the twin's calibration in cameras.json form (frames.sim_calibration), so it runs
the same code as on the cell. Ground truth (simulator poses) is read only here, to score: every target inside
its own bin, and the distractor moved < 5 mm. Writes run/sim_eval.json.
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--client", default=str(ROOT.parent / "PhysicalAI-Hackathon-2026-v6"))
ap.add_argument("--max-acts", type=int, default=150)
ap.add_argument("--weights", default=None)
ap.add_argument("--arm-speed", type=float, default=None,
                help="m/s: the twin arm moves at most this fast toward each row (the real cell, 27 Oct 2026: ~0.05); "
                     "default: every row reached in its dt")
ap.add_argument("--brain-root", default=None, help="project whose mm/real/brain.py the policy runs (default: this one)")
ap.add_argument("--track-moves", action="store_true",
                help="after every row, note each object that has moved > 2 mm since the act began: the row, the "
                     "fingertip there and the move (acts[].moved); finds what knocked an object")
args = ap.parse_args()
run = Path(args.run).resolve()
os.environ["MM_SCENE_JSON"] = str(run / "layout.json")

import numpy as np  # noqa: E402

import scene_config as C  # noqa: E402
from mm.real import frames as F  # noqa: E402

# The policy (and its YOLO brain process) starts before Kit edits PATH.
cal = {"table": F.sim_calibration(json.loads((run / "camera.json").read_text())), "wrist": F.sim_wrist_calibration()}
(run / "sim_cameras.json").write_text(json.dumps(cal, indent=1))
os.environ["MM_CAMERAS_JSON"] = str(run / "sim_cameras.json")
os.environ["MM_ROOT"] = str(Path(args.brain_root).resolve() if args.brain_root else ROOT)
spec = importlib.util.spec_from_file_location("client_policy", Path(args.client) / "policy.py")
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)
policy = client.Policy(args.weights)
W, H = policy.IMAGE_SIZE
DT = getattr(policy, "DT", 0.1)

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
report = {"run": str(run), "acts": [], "fault": None}
try:
    import cv2
    import omni.replicator.core as rep

    import cell
    from mm.control import executor as X
    from mm.control.sim_backend import SimCell, _load_layout

    _load_layout(run)
    state = json.loads((run / "scene_state.json").read_text())
    simcell = SimCell(SimpleNamespace(physics_only=True), run, run / "frames", state, closed_loop=True)
    rp = rep.create.render_product(cell.TABLE_CAM_PATH, tuple(C.TABLE_CAM["resolution"]))
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach(rp)
    rp.hydra_texture.set_updates_enabled(False)
    simcell.products["table1080"] = (rp, ann, cell.TABLE_CAM_PATH, tuple(C.TABLE_CAM["resolution"]))
    simcell.settle(60)
    start = simcell.poses()
    grip = 0.0
    import fk

    def observe():
        table = cv2.cvtColor(simcell.grab("table1080"), cv2.COLOR_BGR2RGB)
        wrist = cv2.cvtColor(cv2.resize(simcell.grab("wrist"), (W, H), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        if table.shape[:2] != (H, W):
            table = cv2.resize(table, (W, H), interpolation=cv2.INTER_AREA)
        q = simcell.arm.get_joint_positions()[simcell.arm_idx]
        rot, tip = fk.tool_pose(q, simcell.tool)
        r_j6, _ = fk.fk(q)
        return {"images": {"table": table, "wrist": wrist}, "joints": np.array([*np.rad2deg(q), grip], np.float32),
                "tcp_pose": [*F.base_to_uf0(tip), *F.wpr_from_rotation(r_j6)], "t": time.time(),
                "trial": {"status": "running", "instruction": "sort the targets into their bins"}}

    policy.reset()
    holds = 0
    for k in range(args.max_acts):
        obs = observe()
        rows = np.asarray(policy.act(obs), float).reshape(-1, 7)
        info = (policy.last or {}).get("info", {})
        rec = {"act": k + 1, "decision": info.get("decision"), "rows": len(rows), "pick": info.get("pick"),
               "previous_pick": info.get("previous_pick"), "route": info.get("route"), "error": (policy.last or {}).get("error"),
               "reason": info.get("reason"), "move": info.get("move"), "error_mm": info.get("error_mm"),
               "wrist_verdict": info.get("wrist_verdict"), "wrist": info.get("wrist"), "seconds": info.get("seconds")}
        if info.get("bin_assignment"):
            rec["bin_assignment"] = {k: v for k, v in info["bin_assignment"].items() if k != "tried"}
        report["acts"].append(rec)
        arrival = f" {rec['move']} off by {rec['error_mm']} mm" if rec["move"] else ""
        print(f"[sim_eval] act {k + 1}: {rec['decision']}{arrival} {rec['pick'] or ''} prev {rec['previous_pick'] or ''} "
              f"rows {len(rows)}", flush=True)
        if any(not F.in_tcp_box(r) for r in rows):
            report["fault"] = f"act {k + 1}: row outside the TCP box"
            break
        if rec["decision"] == "stopped":  # v6: the brain stopped the rollout (home, hold)
            break
        waiting = rec["decision"] == "hold" and "capture again" in (rec["reason"] or "")
        holds = holds + 1 if rec["decision"] == "hold" else 0
        # v6 stops by itself on its third hold in a row (and goes home): let it
        if holds >= (5 if waiting else 3):  # bins not all seen: a few more captures before giving up
            break
        before = simcell.poses() if args.track_moves else None
        if before is not None:
            rec["centres"] = {n: [round(v, 5) for v in before[n]["centre"]] for n in simcell.names}
        try:
            for i, r in enumerate(rows):
                p = F.uf0_to_base(r[:3])
                if args.arm_speed:  # the real arm: at most arm_speed toward this row, then on to the next
                    step = p - simcell.tip_now
                    n = float(np.linalg.norm(step))
                    if n > args.arm_speed * DT:
                        p = simcell.tip_now + step * (args.arm_speed * DT / n)
                jaw = C.JAW_CLOSED if r[6] >= 0.5 else C.JAW_OPEN
                simcell.move("row", simcell.tip_now, p, DT, simcell._jaw, jaw, F.r_to_spread(r[5]))
                grip = 1.0 if r[6] >= 0.5 else 0.0
                if before is not None:
                    now, seen = simcell.poses(), {m["object"] for m in rec.setdefault("moved", [])}
                    for n in simcell.names:
                        d = math.dist(now[n]["centre"], before[n]["centre"]) * 1000
                        if n not in seen and d > 2.0:
                            rec["moved"].append({"object": n, "row": i, "of_rows": len(rows), "moved_mm": round(d, 1),
                                                 "from": [round(v, 4) for v in before[n]["centre"]],
                                                 "tip": [round(float(v), 4) for v in simcell.tip_now],
                                                 "gripper": grip})
        except RuntimeError as e:
            report["fault"] = f"act {k + 1}: {e}"
            print(f"[sim_eval] fault: {e}", flush=True)
            break

    simcell.settle(30)
    final = simcell.poses()
    bins = C.SORT_BINS  # the physical bins (v4: bin_1..3, identical)
    targets, distractors = {}, {}
    for n in simcell.names:
        kind = next((k for k, pre in X.PRIMS.items() if n.startswith(pre)), None)
        if final[n]["centre"][0] > 1.0:  # parked spare (not in this scene)
            continue
        inside = next((name for name, b in bins.items() if X.in_bin(final[n]["centre"], b)[0]), None)
        if kind in ("cube", "aaa_cell", "screw"):
            targets[n] = {"kind": kind, "bin": inside}
        elif kind in ("cap", "tube"):
            moved = math.dist(final[n]["centre"][:2], start[n]["centre"][:2]) * 1000
            distractors[n] = {"kind": kind, "bin": inside, "moved_mm": round(moved, 1)}
    contents = X.bin_contents({n: final[n] for n in list(targets) + list(distractors)}, bins)
    # v4 success: each bin holds one type at most, every target is in a bin, the distractor is not
    # picked (not in a bin; it may roll in the twin, so its movement is reported, not judged)
    report.update(
        targets=targets, distractors=distractors, bin_contents=contents,
        assignment=next((a.get("bin_assignment") for a in report["acts"] if a.get("bin_assignment")), None),
        sorted=sum(t["bin"] is not None for t in targets.values()), n_targets=len(targets),
        success=report["fault"] is None and X.sort_ok(contents, len(targets)),
    )
    if args.track_moves:  # how far each object moved on the table before it was lifted (or by the end)
        snaps = [a["centres"] for a in report["acts"] if a.get("centres")] + [{n: final[n]["centre"] for n in simcell.names}]
        table = {}
        for n in simcell.names:
            s0 = start[n]["centre"]
            for snap in snaps:
                if snap[n][2] - s0[2] > 0.02:  # lifted: picked
                    break
                table[n] = max(table.get(n, 0.0), math.dist(snap[n][:2], s0[:2]) * 1000)
        report["table_moves_mm"] = {n: round(d, 1) for n, d in table.items()}
        print(f"[sim_eval] moved on the table before lift (mm): {report['table_moves_mm']}", flush=True)
    # v5 judges a pick from the next table image (previous_pick); v6 mostly from the wrist camera (wrist_verdict)
    verdicts = [a["previous_pick"]["verdict"] for a in report["acts"] if a.get("previous_pick")]
    verdicts += [a["wrist_verdict"] for a in report["acts"] if a.get("wrist_verdict") in ("picked", "missed")]
    decisions = [a["decision"] for a in report["acts"]]
    report["picks"] = {"attempts": decisions.count("pick"), "picked": sum(v.startswith("picked") for v in verdicts),
                       "missed": sum(v.startswith("missed") for v in verdicts), "retries": decisions.count("retry"),
                       "aborted": decisions.count("abort"),
                       "waits": decisions.count("wait"), "arm_speed": args.arm_speed,
                       "arm_seconds": round(sum(a["rows"] for a in report["acts"]) * DT, 1)}
    print(f"[sim_eval] picks {report['picks']}", flush=True)
    print(f"[sim_eval] in bins {report['sorted']}/{report['n_targets']}, contents {contents}, distractors "
          f"{ {n: (d['bin'], d['moved_mm']) for n, d in distractors.items()} }, fault {report['fault']}, "
          f"success {report['success']}", flush=True)
except Exception:
    import traceback

    traceback.print_exc()
    report["fault"] = report["fault"] or "exception: " + traceback.format_exc(limit=1)
    raise
finally:
    (run / "sim_eval.json").write_text(json.dumps(report, indent=1, default=float))
    policy.close()
    app.close()
