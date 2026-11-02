"""
Closed-loop sorting in the twin (Machine Minds steps B6, B7, B8).

Each cycle: capture the table -> perceive (YOLO server) -> find the bins (B8)
-> plan -> track and order (B7) -> one pick attempt with a grasp check (B6).

  B6  grasp check   wrist images at the hover pose before descending and after
                    lifting (grasp_check.verdict). A miss opens the jaws; the
                    next cycle re-captures and retries the same track, up to
                    tracker.MAX_ATTEMPTS, then flags and skips it.
                    loop.json "inject" offsets some grasps by 3-5 mm to cause
                    misses on purpose and measure the check.
  B7  moving objects re-detect every cycle; the tracker matches, moves, adds
                    (cheapest insertion) and drops. loop.json "events" add
                    spare objects onto the table or nudge one mid-run.
  B8  bins          "bins": "detected" takes each bin's pose from the capture
                    (YOLO "bin" class + pose fit), every cycle. v4: the bins
                    are identical; the first capture with all three names them
                    by position (fewer: capture again), later ones track them
                    by position, and the class -> bin assignment is chosen
                    once by shortest planned travel (bin_assign) and kept.
  IK  paths         planning checks IK only at waypoints; a straight move
                    between them can still leave the arm's reach. Every
                    attempt's whole path is solved frame by frame before the
                    arm moves; if it fails, the wrist turned the other way and
                    a route via the home point are tried; with none, the object
                    is skipped and flagged. Logged in the report's "ik".

Configuration: run/loop.json (written by mm.eval.run_trial). Ground truth is
read only to score (and to place the injected events), never to move the arm.
"""

import json
import math
import os
import random
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

import fk
import scene_config as C
from backproject import CameraModel
from mm.control import executor as X
from mm.control import grasp_check as GC
from mm.control.sim_backend import write_report
from mm.datagen import randomize as R
from mm.perception.bins import bins_seen, first_bins, track_bins
from mm.planning import bin_assign as BA
from mm.planning import grasp as G
from mm.planning import plan_bins as PB
from mm.planning.tracker import Tracker

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TARGETS = ("cube", "screw", "aaa_cell")


class PerceptionClient:
    """perceive_server.py in the YOLO environment, over stdin/stdout."""

    def __init__(self, env, log_path):
        yolo = env.get("YOLO_PY") or (r"C:\isim\yolo\python.exe" if os.name == "nt" else "python")
        server = os.path.join(ROOT, "mm", "perception", "perceive_server.py")
        self.err = open(log_path, "w")
        self.p = subprocess.Popen([yolo, server], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err,
                                  text=True, cwd=ROOT, env=env, bufsize=1)
        self.ready = self._read()
        if not self.ready.get("ready"):
            raise RuntimeError(f"perception server failed: {self.ready}")

    def _read(self):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("perception server exited; see perceive_server.log")
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue  # a library printed to stdout

    def perceive(self, image, camera):
        self.p.stdin.write(json.dumps({"image": str(image), "camera": str(camera)}) + "\n")
        self.p.stdin.flush()
        resp = self._read()
        if "error" in resp:
            raise RuntimeError(f"perception: {resp['error']}")
        return resp

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()
        self.err.close()


def rest_pose(kind, xy, yaw_deg):
    """Root pose of an object resting upright at xy (cell.py origins)."""
    z0 = C.TABLE_TOP_Z + C.SURFACE_DZ
    yaw = Rotation.from_euler("z", yaw_deg, degrees=True)
    if kind == "cube":
        pos, rot = [xy[0], xy[1], z0], yaw
    elif kind == "aaa_cell":
        pos, rot = [xy[0], xy[1], z0 + C.AAA_CELL_RADIUS + 0.0005], yaw * Rotation.from_euler("y", 90, degrees=True)
    elif kind == "screw":
        hx, hy = C.screw_head_xy(xy, yaw_deg)
        pos = [hx, hy, z0 + C.SCREW_REST_HEAD_Z + 0.0002]
        rot = yaw * Rotation.from_euler("y", -90 + C.SCREW_REST_TILT_DEG, degrees=True)
    else:
        raise ValueError(kind)
    x, y, z, w = rot.as_quat()
    return np.array(pos), np.array([w, x, y, z])


def on_table_targets(cell, poses):
    """Scoring: target objects lying on the work area, not in a bin, not parked."""
    z0 = C.TABLE_TOP_Z + C.SURFACE_DZ
    out = {}
    for name in cell.names:
        kind = next((k for k in TARGETS if name.startswith(X.PRIMS[k])), None)
        c = poses[name]["centre"]
        if kind is None or c[0] > 1.0 or not (z0 - 0.01 < c[2] < z0 + 0.03):
            continue
        if any(X.in_bin(c, b)[0] for b in C.SORT_BINS.values()):
            continue
        out[name] = {"kind": kind, "xy": [round(c[0], 4), round(c[1], 4)]}
    return out


def apply_event(cell, ev, tracker, here):
    """B7: put a spare object on the table, or nudge one still to be sorted."""
    poses = cell.poses()
    if ev["type"] == "add":
        pos, quat = rest_pose(ev["kind"], ev["xy"], ev["yaw_deg"])
        cell.teleport({ev["prim"]: (pos, quat)})
        cell.settle(20)
        return {"type": "add", "prim": ev["prim"], "kind": ev["kind"], "xy": ev["xy"]}
    if ev["type"] == "nudge" and "spots" in ev:
        # Curated demo (mm.eval.demo_layout): destinations checked before the
        # run; here only re-checked against where everything is now.
        table = on_table_targets(cell, poses)
        later = [tracker.tracks[t]["pick"]["xy"] for t in tracker.order[1:]] or [p["xy"] for p in table.values()]
        if not table or not later:
            return {"type": "nudge", "skipped": "nothing to nudge"}
        name = min(table, key=lambda n: math.dist(table[n]["xy"], later[-1]))
        z0 = C.TABLE_TOP_Z + C.SURFACE_DZ
        others = [poses[n]["centre"][:2] for n in cell.names if n != name and poses[n]["centre"][0] < 1.0
                  and z0 - 0.01 < poses[n]["centre"][2] < z0 + 0.03
                  and not any(X.in_bin(poses[n]["centre"], b)[0] for b in C.SORT_BINS.values())]
        for xy in ev["spots"].get(name, []):
            if all(math.dist(xy, o) >= ev.get("min_centres", 0.08) for o in others):
                p, q = poses[name]["position"], poses[name]["orientation"]
                shift = np.array([xy[0] - table[name]["xy"][0], xy[1] - table[name]["xy"][1], 0.002])
                cell.teleport({name: (np.asarray(p) + shift, np.asarray(q))})
                cell.settle(20)
                return {"type": "nudge", "prim": name, "from": table[name]["xy"], "to": list(xy)}
        return {"type": "nudge", "skipped": "no planned spot clear"}
    if ev["type"] == "nudge":
        table = on_table_targets(cell, poses)
        # the object the tracker would visit last, so the move happens unseen
        later = [tracker.tracks[t]["pick"]["xy"] for t in tracker.order[1:]] or [p["xy"] for p in table.values()]
        if not table or not later:
            return {"type": "nudge", "skipped": "nothing to nudge"}
        name = min(table, key=lambda n: math.dist(table[n]["xy"], later[-1]))
        d = ev.get("dist_mm", 35) / 1000
        others = [p["xy"] for n, p in table.items() if n != name]
        bins = list(C.SORT_BINS.values())
        for k in range(12):
            a = 2 * math.pi * k / 12
            xy = (table[name]["xy"][0] + d * math.cos(a), table[name]["xy"][1] + d * math.sin(a))
            f = R.footprint(table[name]["kind"], xy, 0.0)
            if (R.TARGET_X[0] <= xy[0] <= R.TARGET_X[1] and R.TARGET_Y[0] <= xy[1] <= R.TARGET_Y[1]
                    and all(math.dist(xy, o) > 0.03 for o in others)
                    and all(R.bin_gap(f, b) > R.BIN_CLEARANCE for b in bins)):
                p, q = poses[name]["position"], poses[name]["orientation"]
                shift = np.array([xy[0] - table[name]["xy"][0], xy[1] - table[name]["xy"][1], 0.002])
                cell.teleport({name: (np.asarray(p) + shift, np.asarray(q))})
                cell.settle(20)
                return {"type": "nudge", "prim": name, "from": table[name]["xy"], "to": [round(v, 4) for v in xy]}
        return {"type": "nudge", "skipped": "no free spot"}
    raise ValueError(ev["type"])


def route_options(cmd, here, spread, dest, home=X.HOME_XYZ):
    """Candidate pick legs and, per pick leg, candidate place legs.

    Pick: straight from here, the wrist turned the other way round, or via
    the home point (both ways). Place: the same for the carry to the bin.
    Segments are (name, p0, p1, duration, jaw0, jaw1, direction).
    """
    spread, dest = np.asarray(spread, float), np.asarray(dest, float)
    here, home = np.asarray(here, float), np.asarray(home, float)
    o, c = C.JAW_OPEN, X.jaw_closed(cmd["category"])
    base, via = X.pick_segments(cmd, here), X.pick_segments(cmd, home)
    reroute = ("reroute", here, home, X._duration(1.1, here, home), o, o)
    picks = [
        ("direct", [s[:6] + (spread,) for s in base[:5]]),
        ("wrist_flipped", [s[:6] + (-spread,) for s in base[:5]]),
        ("via_home", [reroute + (spread,)] + [s[:6] + (spread,) for s in via[:5]]),
        ("via_home_wrist_flipped", [reroute + (-spread,)] + [s[:6] + (-spread,) for s in via[:5]]),
    ]
    up, above = base[5][1], base[5][2]
    carry_via = [("carry", up, home, X._duration(1.4, up, home), c, c), ("carry", home, above, X._duration(1.4, home, above), c, c)]

    def places(pick_segs):
        s = pick_segs[-1][6]
        d0 = dest if np.dot(s, dest) >= 0 else -dest
        return [
            ("direct", [p[:6] + (d0,) for p in base[5:]]),
            ("wrist_flipped", [p[:6] + (-d0,) for p in base[5:]]),
            ("via_home", [p + (d0,) for p in carry_via] + [p[:6] + (d0,) for p in base[6:]]),
            ("via_home_wrist_flipped", [p + (-d0,) for p in carry_via] + [p[:6] + (-d0,) for p in base[6:]]),
        ]

    return picks, places


def choose_route(check, picks, places):
    """First pick + place combination whose path solves at every frame.

    check(segments) -> None or (segment name, frame). Returns a dict with the
    chosen labels, the segments (None if no route), the pick-leg length and
    every failed try.
    """
    failures = []
    for pl, pick_segs in picks:
        f = check(pick_segs)
        if f:
            failures.append({"part": "pick", "route": pl, "segment": f[0], "frame": f[1]})
            continue
        for ql, place_segs in places(pick_segs):
            f = check(pick_segs + place_segs)
            if f:
                failures.append({"part": "place", "route": f"{pl}+{ql}", "segment": f[0], "frame": f[1]})
                continue
            return {"pick": pl, "place": ql, "segments": pick_segs + place_segs, "n_pick": len(pick_segs),
                    "failures": failures}
    return {"pick": None, "place": None, "segments": None, "n_pick": 0, "failures": failures}


def miss_offset(pick, spread, avoid):
    """Deliberate miss: shift the grasp across the jaws (perpendicular to the
    closing direction) far enough that both fingers close beside the object,
    towards the side away from the nearest thing in `avoid` (xy points)."""
    s = np.asarray(spread, float)[:2]
    t = np.array([-s[1], s[0]])
    half = {"cube": C.CUBE_EDGE / 2, "aaa_cell": C.AAA_CELL_LENGTH / 2,
            "screw": (C.SCREW_HEAD_H + C.SCREW_LENGTH) / 2}[pick["category"]]
    d = half + C.FINGER_WIDTH / 2 + 0.005
    xy = np.asarray(pick["xy"], float)
    near = lambda v: min((np.linalg.norm(xy + v - np.asarray(a)) for a in avoid), default=1.0)  # noqa: E731
    return max((d * t, -d * t), key=near)


def attempt(cell, pick, here, cfg, rng, cam_j6, deliberate_avoid=None):
    """One grasp attempt; returns (record, new tool position, taken).

    record["skipped"] = "no_ik_path" when no route solves: the arm only
    pauses (a short hold, so the video shows it), nothing is grasped.
    deliberate_avoid (xy list) makes this the scripted deliberate miss.
    """
    r, _ = fk.fk(cell.q)
    current = r @ np.array(C.FINGER_SPREAD_AXIS)
    spread = np.array(pick["spread_dir"], float)
    if np.dot(current, spread) < 0:
        spread = -spread
    dest = np.array(pick["place_spread_dir"], float)
    if np.dot(spread, dest) < 0:
        dest = -dest
    start = cell.poses()
    tracked = cell.track_target(pick, start, radius=0.03)
    inject = cfg.get("inject") or {}
    offset = np.zeros(2)
    if deliberate_avoid is not None:
        offset = miss_offset(pick, spread, deliberate_avoid)
    elif rng.random() < inject.get("prob", 0.0):
        lo, hi = inject.get("offset_mm", [3, 5])
        a = rng.uniform(0, 2 * math.pi)
        offset = rng.uniform(lo, hi) / 1000 * np.array([math.cos(a), math.sin(a)])
    cmd = dict(pick, xy=list(np.asarray(pick["xy"]) + offset))
    rec = {
        "category": pick["category"], "bin": pick["bin"], "planned_xy": pick["xy"],
        "injected_offset_mm": round(float(np.linalg.norm(offset)) * 1000, 2), "tracked_object": tracked,
    }
    if deliberate_avoid is not None:
        rec["deliberate_miss"] = True
    if tracked:
        rec["perception_error_mm"] = round(float(np.linalg.norm(np.array(start[tracked]["centre"][:2]) - pick["xy"]) * 1000), 3)
    route = choose_route(lambda segs: cell.check_path([(s[0], s[1], s[2], s[3], s[6]) for s in segs]),
                         *route_options(cmd, here, spread, dest, cell.home))
    rec["route"] = {k: route[k] for k in ("pick", "place", "failures")}
    if route["segments"] is None:
        logs = [cell.hold("ik_skip", 1.0, C.JAW_OPEN, C.JAW_OPEN)]
        rec.update(skipped="no_ik_path", gt_lifted=False, pickup_verified=False, placement_verified=False,
                   verdict_taken=False, segments=logs)
        rec.update(X.summarise(logs))
        return rec, np.asarray(here, float), False
    pick_legs, place_legs = route["segments"][:route["n_pick"]], route["segments"][route["n_pick"]:]
    logs = []

    def run(seg):
        logs.append(cell.move(*seg))

    for seg in pick_legs:
        run(seg)
        if seg[0] == "approach":
            pre = cell.grab("wrist") if cfg.get("grasp_check") else None
        if seg[0] == "dwell":
            rec["jaw_gap_mm"] = cell.jaw_gap_mm()
    lifted = cell.poses()
    rec["gt_lifted"] = bool(tracked and lifted[tracked]["centre"][2] - start[tracked]["centre"][2] > 0.025)
    if cfg.get("grasp_check"):
        post = cell.grab("wrist")
        mask = GC.spot_mask(cam_j6, cell.q, pick["xy"], pick["category"], post.shape)
        v = GC.verdict(pre, post, mask)
        rec["check"] = v
        taken = True if v["taken"] is None else v["taken"]  # spot out of view: carry on
    else:
        taken = True
    rec["verdict_taken"] = taken
    if taken:
        for seg in place_legs:
            run(seg)
        finish = cell.poses()
        if tracked:
            ok, where = X.in_bin(finish[tracked]["centre"], X.bin_at(pick["place_xy"], C.SORT_BINS))
            rec["placement_verified"], rec["placement"] = bool(ok), where
        here = place_legs[-1][2]
    else:
        logs.append(cell.hold("open", 0.5, X.jaw_closed(pick["category"]), C.JAW_OPEN))
        here = pick_legs[-1][2]
    rec["segments"] = logs
    rec.update(X.summarise(logs))
    rec["pickup_verified"] = rec["gt_lifted"]
    rec.setdefault("placement_verified", False)
    return rec, np.asarray(here, float), taken


def run_closed_loop(cell, plan0, cfg, env):
    client = PerceptionClient(env, cell.run / "perceive_server.log")
    rng = random.Random(cfg.get("seed", 0))
    tracker = Tracker()
    reach = G.Reach()
    cam_j6 = GC.wrist_camera()
    # v4: physical bins by name (bin_1..3), found and tracked by position; which class goes to
    # which bin is chosen once per rollout (bin_assign) and kept.
    phys = None if cfg.get("bins") == "detected" else R.load_bins()
    assignment, assignment_log, bins = None, None, None
    here = np.asarray(cell.home, float)
    attempts, cycles, events_done, ik_events, abort_reason = [], [], [], [], None
    picks_done = 0
    pending = list(enumerate(cfg.get("events", [])))
    initial = on_table_targets(cell, cell.poses())
    plan = plan0
    try:
        for cycle in range(cfg.get("max_cycles", 60)):
            for k, ev in list(pending):
                if picks_done >= ev["after_picks"]:
                    res = apply_event(cell, ev, tracker, here)
                    res.update(cycle=cycle, after_picks=picks_done)
                    events_done.append(res)
                    pending.remove((k, ev))
                    print(f"[closed_loop] event: {res}", flush=True)
            img, camj = cell.table_capture(cycle)
            resp = client.perceive(img, camj)
            dets = resp["objects"]
            info = {"cycle": cycle, "capture": os.path.relpath(img, cell.run), "perceive_s": resp["seconds"],
                    "detections": {c: sum(d["class"] == c for d in dets) for c in set(d["class"] for d in dets)}}
            cam = CameraModel(str(camj))
            if cfg.get("bins") == "detected":
                phys = first_bins(dets) if phys is None else track_bins(dets, phys)
                if phys is None:  # fewer than 3 bins at the start: capture again, never guess
                    info["bins"] = f"{len(bins_seen(dets))} of 3 found: capture again"
                    cycles.append(info)
                    print(f"[closed_loop] cycle {cycle}: {info['bins']}", flush=True)
                    continue
                info["bins"] = {}
                for n, b in phys.items():
                    truth = X.bin_at(b["xy"], C.SORT_BINS)  # scoring only
                    info["bins"][n] = {
                        "found": b["found"], "xy": [round(v, 4) for v in b["xy"]], "yaw_deg": round(b["yaw_deg"], 2),
                        "error_mm": round(math.dist(b["xy"], truth["xy"]) * 1000, 2),
                        "yaw_error_deg": round(abs((b["yaw_deg"] - truth.get("yaw_deg", 0.0) + 45) % 90 - 45), 2)}
            # Sorted objects stay visible inside their bins: never plan those.
            for d in dets:
                if d["eligible"] and d["xy_robot_m"] and any(X.xy_in_bin(d["xy_robot_m"], b) for b in phys.values()):
                    d["eligible"], d["skip_reason"] = False, "in_bin"
            if assignment is None:
                assignment, assignment_log = BA.assign_bins({"objects": dets}, cam, phys, reach=reach)
                info["bin_assignment"] = {k: v for k, v in assignment_log.items() if k != "tried"}
                print(f"[closed_loop] bins assigned {assignment}: travel {assignment_log['travel_m']} m vs worst "
                      f"{assignment_log['worst_travel_m']} m", flush=True)
            bins = BA.table(phys, assignment)
            plan = PB.build_plan({"objects": dets}, cam, bins=bins, reach=reach)
            change = tracker.update(plan["picks"], here, cycle)
            info.update(planned=len(plan["picks"]), skipped=[(s["category"], s["skip_reason"]) for s in plan["skipped"]
                                                            if s["category"] in TARGETS],
                        tracker={k: change[k] for k in ("new", "moved", "vanished")})
            cycles.append(info)
            tid, pick = tracker.next()
            print(f"[closed_loop] cycle {cycle}: {len(dets)} dets, {len(plan['picks'])} pickable, next {tid}, "
                  f"tracker new {change['new']} moved {change['moved']} vanished {change['vanished']}", flush=True)
            if tid is None:
                if pending:  # events still due: they need picks first; nothing left -> run them now
                    picks_done = max(picks_done, min(ev["after_picks"] for _, ev in pending))
                    continue
                break
            # Scripted deliberate miss: the Nth grasp attempt if it is a first
            # try, else the next first try; once per run, never on a retry.
            deliberate = None
            nth = (cfg.get("inject") or {}).get("deliberate_attempt")
            done_tries = sum(1 for a in attempts if not a.get("skipped"))
            if (nth and not any(a.get("deliberate_miss") for a in attempts) and done_tries + 1 >= nth
                    and tracker.tracks[tid]["attempts"] == 0):
                deliberate = [tracker.tracks[t]["pick"]["xy"] for t in tracker.order if t != tid]
                deliberate += [b["xy"] for b in (bins or {}).values()]
            rec, here, taken = attempt(cell, pick, here, cfg, rng, cam_j6, deliberate)
            rec.update(cycle=cycle, track=tid, attempt=tracker.tracks[tid]["attempts"] + 1)
            route = rec["route"]
            if rec.get("skipped"):
                tracker.give_up(tid)
                ik_events.append({"cycle": cycle, "track": tid, "category": rec["category"], "action": "skipped",
                                  "failures": route["failures"]})
            elif taken:
                tracker.picked(tid)
                picks_done += 1
            else:
                rec["gave_up"] = tracker.attempt_failed(tid)
            if route["failures"] and not rec.get("skipped"):
                ik_events.append({"cycle": cycle, "track": tid, "category": rec["category"], "action": "rerouted",
                                  "pick": route["pick"], "place": route["place"], "failures": route["failures"]})
            if route["failures"]:
                print(f"[closed_loop]   IK: {ik_events[-1]}", flush=True)
            attempts.append(rec)
            print(f"[closed_loop]   {tid} {rec['category']} attempt {rec['attempt']}: inject {rec['injected_offset_mm']} mm, "
                  f"lifted {rec['gt_lifted']}, check {rec.get('check', {}).get('reason')}, "
                  f"placed {rec['placement_verified']}", flush=True)
            (cell.run / "execution_events.json").write_text(json.dumps(attempts, indent=2, default=float) + "\n")
        home = ("home", cell.tip_now, cell.home, X._duration(1.1, cell.tip_now, cell.home), C.JAW_OPEN, C.JAW_OPEN, None)
        failed = cell.check_path([(home[0], home[1], home[2], home[3], home[6])])
        if failed:
            ik_events.append({"cycle": None, "action": "home_skipped", "failures": [{"segment": failed[0], "frame": failed[1]}]})
        else:
            segs = cell.move(*home)
            if attempts:
                attempts[-1]["segments"].append(segs)
    except RuntimeError as e:
        abort_reason = str(e)
        print(f"[closed_loop] abort: {abort_reason}", flush=True)
    finally:
        client.close()

    after = cell.poses()
    final = on_table_targets(cell, after)
    if phys is None and abort_reason is None:
        abort_reason = "fewer than 3 bins found at every capture"
    # scored under the robot's own bin names (by position), each mapped to the true bin nearest it
    named = {n: X.bin_at(b["xy"], C.SORT_BINS) for n, b in phys.items()} if phys else C.SORT_BINS
    contents = X.bin_contents({n: p for n, p in after.items() if p["centre"][0] < 1.0}, named)
    n_targets = len(initial) + sum(e.get("type") == "add" and "prim" in e for e in events_done)
    checked = [a for a in attempts if "check" in a and a["check"].get("taken") is not None]
    misses = [a for a in checked if not a["gt_lifted"]]
    extra = {
        "mode": "closed_loop",
        "loop": cfg,
        "cycles": cycles,
        "events": events_done,
        "tracker": tracker.log,
        "flagged": list(tracker.flagged),
        "targets_initial": len(initial),
        "targets_added": sum(e.get("type") == "add" and "prim" in e for e in events_done),
        "targets_left_on_table": final,
        "sorted_into_bins": sum(bool(a["placement_verified"]) for a in attempts),
        "task_complete": abort_reason is None and not final,
        "bin_assignment": assignment_log,
        "bin_contents": contents,  # ground truth, scoring only
        "sort_ok": abort_reason is None and X.sort_ok(contents, n_targets),
        "grasp_check": {
            "attempts_checked": len(checked),
            "actual_misses": len(misses),
            "misses_caught": sum(not a["check"]["taken"] for a in misses),
            "false_alarms": sum(a["gt_lifted"] and not a["check"]["taken"] for a in checked),
            "injected": sum(a["injected_offset_mm"] > 0 for a in attempts if not a.get("skipped")),
        },
        "ik": {
            "rerouted": sum(e["action"] == "rerouted" for e in ik_events),
            "skipped": sum(e["action"] == "skipped" for e in ik_events),
            "events": ik_events,
        },
    }
    pseudo_plan = {"order": "tracker", "predicted_travel_m": None, "picks": attempts, "skipped": plan.get("skipped", [])}
    write_report(cell, pseudo_plan, attempts, abort_reason, extra)
