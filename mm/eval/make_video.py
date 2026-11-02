"""
Captioned demo video from a rendered run (Machine Minds B5-B8 videos).

    .venv/bin/python -m mm.eval.make_video --run runs/<id> --title "B6 grasp check + retries"

Reads run/frames (demo camera), run/wrist_frames (wrist camera) and
run/execution_report.json. Layout: a title/caption band above the camera
image, a running-totals footer below it, the wrist view as a picture-in-picture
in the image's upper right. Captions and totals follow the per-segment frame
counts in the report, so they change when things happen: the grasp-check
verdict after the lift, "placed" only once the jaws have released. A closing
card holds the last frame with the run's result.

Refuses runs whose frames are stale: more than 5% of consecutive frames
identical (sim_backend renders every frame fresh; older runs did not).
"""

import argparse
import glob
import hashlib
import json
import os

import cv2
import imageio_ffmpeg
import numpy as np

FPS = 30
TOP, BOTTOM = 118, 58  # bands outside the camera image: nothing covers the arm
PIP_W, PIP_MARGIN = 560, 22
MAX_DUPLICATE_FRAC = 0.05
END_HOLD_S = 3.0


def duplicate_frac(paths):
    """Share of frames byte-identical to the one before."""
    prev, dup = None, 0
    for p in paths:
        with open(p, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
        dup += h == prev
        prev = h
    return dup / max(1, len(paths) - 1)


def rendered(segment):
    """Video frames of a segment: PathLog also counts the start sample."""
    return max(0, segment["frames"] - 1)


def _closed_loop_timeline(report):
    """Per frame: (caption, totals dict). Closed loop (B6-B8)."""
    events = report.get("events", [])
    cycles = {c["cycle"]: c for c in report.get("cycles", [])}
    added_at = sorted(e["cycle"] for e in events if e.get("type") == "add" and "prim" in e)
    t = dict(targets=report.get("targets_initial", 0), sorted=0, attempts=0, misses=0, caught=0, false_alarms=0, dropped=0,
             reroutes=0, ik_skips=0)
    seen_cycles, frames = set(), []
    for a in report["picks"]:
        head = f"cycle {a['cycle']}: {a['category']} {a['track']}" + ("" if a.get("skipped") else f", attempt {a['attempt']}")
        if a.get("deliberate_miss"):
            head += f" | deliberate miss: fingers offset {a['injected_offset_mm']:.0f} mm beside it"
        elif a.get("injected_offset_mm"):
            head += f" | grasp offset {a['injected_offset_mm']:.0f} mm injected"
        news = []
        if a["cycle"] not in seen_cycles:
            seen_cycles.add(a["cycle"])
            t["targets"] = report.get("targets_initial", 0) + sum(c <= a["cycle"] for c in added_at)
            for e in events:
                if e.get("cycle") == a["cycle"] and "prim" in e:
                    news.append(f"{e['kind']} added mid-run" if e["type"] == "add" else
                                f"{e['prim']} nudged {np.hypot(*np.subtract(e['to'], e['from'])) * 1000:.0f} mm")
            bins = cycles.get(a["cycle"], {}).get("bins")
            if isinstance(bins, dict) and bins:
                news.append(f"bins found {sum(b['found'] for b in bins.values())}/{len(bins)}")
            ba = cycles.get(a["cycle"], {}).get("bin_assignment")
            if ba:
                news.append("robot assigns the bins: " + ", ".join(f"{PLURAL[c]} -> {b}" for c, b in sorted(ba["assignment"].items(), key=lambda kv: kv[1])))
        miss = not a["gt_lifted"]
        taken = a.get("verdict_taken", True)
        route = a.get("route") or {}
        rerouted = bool(route.get("failures")) and not a.get("skipped")
        if rerouted:
            t["reroutes"] += 1
        for s in a.get("segments", []):
            name = s["segment"]
            if name == "ik_skip":
                phase = "IK: no reachable path (straight, wrist turned, via home) -> object skipped and flagged"
                if news:
                    phase = "re-capture: " + "; ".join(news) + " | " + phase
            elif name == "reroute":
                phase = "IK: straight path out of reach -> rerouting via home"
            elif name == "approach":
                t["attempts"] += 1
                phase = "re-captured, re-planned: moving to target"
                if news:
                    phase = "re-capture: " + "; ".join(news) + " | moving to target"
                if rerouted and route.get("pick") == "wrist_flipped":
                    phase += " | IK: straight path out of reach -> wrist turned the other way"
                elif rerouted and route.get("pick") == "direct":
                    phase += f" | IK: carry rerouted ({route.get('place', '').replace('_', ' ')})"
            elif name == "descend":
                phase = "descending"
            elif name in ("close", "dwell"):
                phase = "closing jaws"
            elif name == "lift":
                phase = "lifting; wrist camera looks at the pick spot"
            elif name == "open":
                phase = "wrist check: object still on the table -> open, re-capture, retry"
                if a["gt_lifted"]:
                    phase += " (false alarm: it was in the jaws)"
                if a.get("gave_up"):
                    phase += " (3rd try: flagged, skipped)"
            elif name in ("carry", "lower"):
                phase = f"wrist check: spot cleared -> carry to {a['bin']} bin"
                if rerouted and route.get("place", "direct") != "direct":
                    phase += f" (IK: {route['place'].replace('_', ' ')})"
            elif name == "release":
                phase = f"releasing over {a['bin']} bin"
            elif name in ("settle", "retreat"):
                if a["placement_verified"]:
                    phase = f"placed in {a['bin']} bin"
                elif a["gt_lifted"]:
                    phase = "dropped on the way: bin empty (next capture looks for it)"
                else:
                    phase = "bin empty: the grasp had missed and the check passed it"
            elif name == "home":
                phase = "no pickable target left: home"
            else:
                phase = name
            for _ in range(rendered(s)):
                frames.append((head + " | " + phase, dict(t)))
            # totals change at the moment the outcome is known
            if name == "lift":
                t["misses"] += miss
                t["caught"] += miss and not taken
                t["false_alarms"] += (not miss) and not taken
            if name == "release":
                t["sorted"] += bool(a["placement_verified"])
                t["dropped"] += bool(a["gt_lifted"] and not a["placement_verified"])
            if name == "ik_skip":
                t["ik_skips"] += 1
    return frames


def _open_loop_timeline(report):
    """Per frame: (caption, totals dict). Open loop (B5)."""
    n = len(report["picks"])
    t = dict(planned=report["picks_planned"], picked=0, placed=0)
    frames = []
    for k, a in enumerate(report["picks"]):
        head = f"pick {k + 1}/{n}: {a['category']} -> {a['bin']} bin"
        for s in a.get("segments", []):
            name = s["segment"]
            phase = {
                "approach": "moving to the planned position", "descend": "descending", "close": "closing jaws",
                "dwell": "closing jaws", "lift": "lifting", "release": f"releasing over {a['bin']} bin",
                "home": "plan finished: home",
            }.get(name)
            if name in ("carry", "lower"):
                phase = f"carrying to {a['bin']} bin" if a["pickup_verified"] else "nothing in the jaws (no check in open loop)"
            elif name in ("settle", "retreat"):
                phase = f"placed in {a['bin']} bin" if a["placement_verified"] else "not in the bin"
            for _ in range(rendered(s)):
                frames.append((head + " | " + (phase or name), dict(t)))
            if name == "lift":
                t["picked"] += bool(a["pickup_verified"])
            if name == "release":
                t["placed"] += bool(a["placement_verified"])
    return frames


def footer_text(report, totals, frame):
    sim = f"motion time {frame / FPS:.0f} s"
    if report.get("mode") == "closed_loop":
        return (f"sorted {totals['sorted']} of {totals['targets']} targets | attempts {totals['attempts']} | "
                f"missed grasps (sim truth) {totals['misses']}, caught by wrist check {totals['caught']} | "
                f"false alarms {totals['false_alarms']} | dropped {totals['dropped']} | "
                f"IK reroutes {totals['reroutes']}, skips {totals['ik_skips']} | {sim}")
    return f"placed {totals['placed']} of {totals['planned']} planned picks | lifted {totals['picked']} | {sim}"


PLURAL = {"cube": "cubes", "aaa_cell": "AAA cells", "screw": "screws"}
SINGULAR = {"cube": "cube", "aaa_cell": "AAA cell", "screw": "screw"}


def assignment_line(report):
    """v4: the robot's bin choice, shown for the whole run (it never changes)."""
    ba = report.get("bin_assignment")
    if not ba:
        return None
    parts = " | ".join(f"{b} = {PLURAL[c]}" for c, b in sorted(ba["assignment"].items(), key=lambda kv: kv[1]))
    return (f"Bins chosen by the robot at the first capture, kept all run: {parts}   "
            f"(planned travel {ba['travel_m']:.2f} m, worst choice {ba['worst_travel_m']:.2f} m)")


def bin_labels(report):
    """[(text, xy)] for each bin, at its perceived position, once assigned."""
    ba = report.get("bin_assignment")
    first = next((c["bins"] for c in report.get("cycles", []) if isinstance(c.get("bins"), dict) and c["bins"]), None)
    if not ba or not first:
        return []
    holds = {b: PLURAL[c] for c, b in ba["assignment"].items()}
    return [(f"{name}: {holds.get(name, 'empty')}", b["xy"]) for name, b in first.items()]


def draw_bin_labels(img, labels, vcam):
    if not labels or vcam is None:
        return img
    import scene_config as C

    sx, sy = img.shape[1] / vcam.W, img.shape[0] / vcam.H
    for text, xy in labels:
        p = vcam.project([xy[0], xy[1], C.TABLE_TOP_Z + C.SURFACE_DZ + 0.09])
        if p is None:
            continue
        u, v = int(p[0] * sx), int(p[1] * sy)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x0, y0 = u - tw // 2, v - 30
        cv2.rectangle(img, (x0 - 8, y0 - th - 8), (x0 + tw + 8, y0 + 8), (255, 255, 255), -1)
        cv2.rectangle(img, (x0 - 8, y0 - th - 8), (x0 + tw + 8, y0 + 8), (40, 40, 40), 1)
        cv2.putText(img, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (25, 25, 25), 2, cv2.LINE_AA)
    return img


def video_camera():
    """The demo camera (sim_backend.VIDEO_CAM) as a CameraModel, for labels."""
    try:
        from mm.control.sim_backend import VIDEO_CAM
        from test_support import camera_model

        return camera_model(VIDEO_CAM)
    except Exception:  # noqa: BLE001  (labels are optional)
        return None


def result_text(report):
    if "sort_ok" in report:  # v4 rule
        total = report.get("targets_initial", 0) + report.get("targets_added", 0)
        contents = report.get("bin_contents") or {}
        binned = sum(k in PLURAL for v in contents.values() for k in v)
        distractor_binned = [k for v in contents.values() for k in v if k not in PLURAL]
        one_type = all(len(set(v)) <= 1 for v in contents.values())
        return (f"Result: {binned} of {total} targets in bins | each bin one type: {'yes' if one_type else 'NO'} | "
                f"cap and tube picked: {'no' if not distractor_binned else 'YES'} | "
                + "; ".join(f"{b}: {len(v)} {(PLURAL.get(v[0], v[0]) if len(v) > 1 else SINGULAR.get(v[0], v[0])) if v else ''}".strip()
                            for b, v in sorted(contents.items()))
                + f" | {'SUCCESS' if report['sort_ok'] else 'FAIL'}")
    if report.get("mode") == "closed_loop":
        g = report["grasp_check"]
        total = report.get("targets_initial", 0) + report.get("targets_added", 0)
        return (f"Result: sorted {report['sorted_into_bins']} of {total} targets, "
                f"{len(report['targets_left_on_table'])} left on the table (refused at planning or lost); "
                f"missed grasps {g['actual_misses']}, caught {g['misses_caught']}"
                + (f"; IK reroutes {report['ik']['rerouted']}, skips {report['ik']['skipped']}" if "ik" in report else ""))
    refused = [s for s in report.get("skipped_at_planning", []) if s.get("skip_reason") not in ("bin", "distractor")]
    return (f"Result: {report['verified_placements']} of {report['picks_planned']} planned picks placed; "
            f"{len(refused)} targets refused at planning (" + ", ".join(sorted({s['skip_reason'] for s in refused})) + ")")


def put_fit(img, text, org, scale, color, thick, max_w):
    """putText, shrunk until it fits max_w pixels."""
    while scale > 0.45 and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] > max_w:
        scale *= 0.95
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def compose(img, wrist, title, caption, footer, subtitle=None, labels=None, vcam=None):
    h, w = img.shape[:2]
    top = TOP + (40 if subtitle else 0)
    canvas = np.full((top + h + BOTTOM, w, 3), 255, np.uint8)
    canvas[top:top + h] = draw_bin_labels(img.copy(), labels, vcam)
    if wrist is not None:
        ph = round(wrist.shape[0] * PIP_W / wrist.shape[1])
        x0, y0 = w - PIP_W - PIP_MARGIN, top + PIP_MARGIN
        cv2.rectangle(canvas, (x0 - 4, y0 - 4), (x0 + PIP_W + 3, y0 + ph + 31), (255, 255, 255), -1)
        canvas[y0:y0 + ph, x0:x0 + PIP_W] = cv2.resize(wrist, (PIP_W, ph), interpolation=cv2.INTER_AREA)
        cv2.putText(canvas, "wrist camera", (x0 + 6, y0 + ph + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 40, 40), 1, cv2.LINE_AA)
    put_fit(canvas, title, (20, 46), 1.15, (20, 20, 20), 2, w - 40)
    if subtitle:
        put_fit(canvas, subtitle, (20, 88), 0.74, (150, 70, 0), 2, w - 40)
    put_fit(canvas, caption, (20, top - 24), 0.78, (45, 45, 45), 2, w - 40)
    put_fit(canvas, footer, (20, top + h + 38), 0.72, (55, 55, 55), 1, w - 40)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report = json.load(open(os.path.join(args.run, "execution_report.json")))
    frames = sorted(glob.glob(os.path.join(args.run, "frames", "f*.png")))
    wrists = sorted(glob.glob(os.path.join(args.run, "wrist_frames", "f*.jpg")))
    if not frames:
        raise SystemExit("no frames: run the trial with --render")
    for name, paths in (("frames", frames), ("wrist_frames", wrists)):
        frac = duplicate_frac(paths) if paths else 0.0
        print(f"[make_video] {name}: {len(paths)}, identical consecutive {frac:.1%}")
        if frac > MAX_DUPLICATE_FRAC:
            raise SystemExit(f"{name}: {frac:.1%} identical consecutive frames (max {MAX_DUPLICATE_FRAC:.0%}): stale render")
    if wrists and len(wrists) != len(frames):
        raise SystemExit(f"{len(frames)} frames but {len(wrists)} wrist frames")
    timeline = _closed_loop_timeline(report) if report.get("mode") == "closed_loop" else _open_loop_timeline(report)
    if len(timeline) != len(frames):
        raise SystemExit(f"report segments give {len(timeline)} frames, run has {len(frames)}")

    subtitle, labels, vcam = assignment_line(report), bin_labels(report), video_camera()
    first = cv2.imread(frames[0])
    size = (first.shape[1], first.shape[0] + TOP + (40 if subtitle else 0) + BOTTOM)
    out = args.out or os.path.join(args.run, "demo.mp4")
    writer = imageio_ffmpeg.write_frames(out, size, fps=FPS, quality=7, macro_block_size=8)
    writer.send(None)
    for i, path in enumerate(frames):
        caption, totals = timeline[i]
        wrist = cv2.imread(wrists[i]) if wrists else None
        img = compose(cv2.imread(path), wrist, args.title, caption, footer_text(report, totals, i), subtitle, labels, vcam)
        writer.send(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tobytes())
    # closing card: final totals (all outcomes are in by the last frame)
    final = dict(timeline[-1][1])
    img = compose(cv2.imread(frames[-1]), wrist, args.title, result_text(report), footer_text(report, final, len(frames)),
                  subtitle, labels, vcam)
    for _ in range(round(END_HOLD_S * FPS)):
        writer.send(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tobytes())
    writer.close()
    print(f"[make_video] {len(frames)} frames + {END_HOLD_S:.0f} s result card -> {out}")


if __name__ == "__main__":
    main()
