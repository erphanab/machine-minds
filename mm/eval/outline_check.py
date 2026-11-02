"""
Today's real pictures as a test set (never trained on): what each code version would pick, per detection.

    C:\\isim\\yolo\\python.exe -m mm.eval.outline_check dump --run ../v5/runs/real/20261028_163407 --out runs/oc/v6.json
    C:\\isim\\yolo\\python.exe -m mm.eval.outline_check dump --code-root ../v5 --run ... --out runs/oc/v5.json
    C:\\isim\\yolo\\python.exe -m mm.eval.outline_check report --v5 runs/oc/v5.json --v6 runs/oc/v6.json --out runs/oc

dump: every capture act (act_*.json with detections) -> YOLO on its saved table image -> that code's perception
(v6: MM_GRASP, default outline) -> bins found in the same image, the brain's in-bin rule, bin assignment and the
planner -> per detection: picked or skip reason. --code-root runs another tree's modules (v5) on the same images.
Saved images are JPEGs of the PNGs the run used, so detections can differ slightly from the run's own logs.

report: per-class counts (v5 vs v6), AAA cells compared field by field, outline quality for cubes and screws
against the object's own pixels (saturated colour for cubes, dark pixels for screws), and previews with the mask,
grasp centre and jaw direction drawn on (review/<rollout>/<act>.jpg, suspects/*.jpg).
"""

import argparse
import glob
import json
import math
import os
import sys
import time


def dump(a):
    root = os.path.abspath(a.code_root)
    for name in list(sys.modules):  # this tree's mm/ (and top-level modules) are loaded: use root's instead
        if name.split(".")[0] in ("mm", "scene_config", "fk", "ik", "backproject"):
            del sys.modules[name]
    sys.path.insert(0, root)
    os.chdir(root)
    import cv2
    import numpy as np
    from ultralytics import YOLO

    from mm.datagen.masks_to_yolo import load_roi
    from mm.perception import perceive_yolo as PY
    from mm.perception.bins import first_bins
    from mm.planning import bin_assign as BA
    from mm.planning import grasp as G
    from mm.planning import plan_bins as PB
    from mm.control import executor as X
    from mm.real import brain as BR
    from mm.real import frames as F

    weights = a.weights or BR.default_weights()
    model, imgsz = YOLO(weights), PY.model_imgsz(weights)
    classes, roi = PY.load_classes(), load_roi()
    cam = F.CellCamera.load(a.cameras, "table", (1920, 1080))
    reach = G.Reach()
    out = {"code_root": root, "weights": weights, "grasp_mode": getattr(PY, "GRASP_MODE", "pose_fit"), "images": []}
    acts = sorted(glob.glob(os.path.join(a.run, "rollout_*", "act_*.json")))
    for path in acts:
        rec = json.load(open(path))
        if "detections" not in rec:
            continue
        img = cv2.imread(path.replace(".json", "_table.jpg"))
        t0 = time.time()
        raw = PY.detect(model, img, cam, roi, imgsz)
        dets = PY.finalize(raw, img, cam, classes)
        t_perc = time.time() - t0
        phys = first_bins(dets)
        status = {}
        if phys is None:
            for d in dets:
                status[d["id"]] = "fewer_than_3_bins" if d["eligible"] else d["skip_reason"]
        else:
            for d in dets:
                if d["eligible"] and d["xy_robot_m"] and any(X.xy_in_bin(d["xy_robot_m"], b) for b in phys.values()):
                    d["eligible"], d["skip_reason"] = False, "in_bin"
            assignment, _ = BA.assign_bins({"objects": dets}, cam, phys, reach=reach)
            plan = PB.build_plan({"objects": dets}, cam, bins=BA.table(phys, assignment), reach=reach)
            for p in plan["picks"]:
                status[p["detection_id"]] = "pick"
            for s in plan["skipped"]:
                status[s["detection_id"]] = s["skip_reason"]
        rows = []
        for d, inst in zip(dets, raw):
            ys, xs = np.nonzero(inst["mask"])
            row = {k: d.get(k) for k in ("class", "conf", "xy_robot_m", "yaw_deg", "fit_iou", "skip_reason", "colour",
                                          "centre_px", "rect_mm", "head_end", "length_mm", "shank_mm", "head_mm",
                                          "grip_from_mm", "grip_to_mm", "grasp")}
            row["status"] = status.get(d["id"], d["skip_reason"])
            row["mask_px"] = int(len(xs))
            row["mask_rle"] = [inst["window"], list(inst["mask"].shape), np.packbits(inst["mask"]).tolist()]
            rows.append(row)
        out["images"].append({"file": os.path.relpath(path.replace(".json", "_table.jpg"), a.run).replace(os.sep, "/"),
                              "bins": 0 if phys is None else len(phys), "seconds": round(t_perc, 2), "dets": rows})
        print(f"{out['images'][-1]['file']}: {len(rows)} dets, {t_perc:.1f} s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f)


def mask_of(row):
    import numpy as np

    (x, y), (h, w), bits = row["mask_rle"]
    m = np.unpackbits(np.array(bits, np.uint8))[: h * w].reshape(h, w).astype(bool)
    return x, y, m


def own_pixels(img, row, pad=6):
    """The object's own pixels near its mask: saturated colour (cube) or dark (screw); (x0, y0, bool window)."""
    import cv2
    import numpy as np

    x, y, m = mask_of(row)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img.shape[1], x + m.shape[1] + pad), min(img.shape[0], y + m.shape[0] + pad)
    hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    if row["class"] == "cube":
        own = (hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 40)
    else:
        own = hsv[:, :, 2] <= 90
    n, lab, stats, _ = cv2.connectedComponentsWithStats(own.astype(np.uint8), 8)
    big = np.zeros_like(own)
    win = np.zeros_like(own)
    win[y - y0:y - y0 + m.shape[0], x - x0:x - x0 + m.shape[1]] = m
    for k in range(1, n):  # components that overlap the mask
        comp = lab == k
        if (comp & win).sum() >= 3:
            big |= comp
    return x0, y0, big, win


def report(a):
    import cv2
    import numpy as np

    v5, v6 = json.load(open(a.v5)), json.load(open(a.v6))
    run = a.run
    os.makedirs(a.out, exist_ok=True)
    classes = ("cube", "screw", "aaa_cell", "cap")
    counts = {v: {c: {"detected": 0, "pick": 0, "skips": {}} for c in classes} for v in ("v5", "v6")}
    for v, data in (("v5", v5), ("v6", v6)):
        for im in data["images"]:
            for d in im["dets"]:
                c = counts[v].get(d["class"])
                if c is None:
                    continue
                c["detected"] += 1
                if d["status"] == "pick":
                    c["pick"] += 1
                else:
                    c["skips"][d["status"]] = c["skips"].get(d["status"], 0) + 1
    # AAA cells: same image, same mask -> identical pose and decision?
    aaa_same, aaa_diff = 0, []
    for i5, i6 in zip(v5["images"], v6["images"]):
        assert i5["file"] == i6["file"]
        a5 = [d for d in i5["dets"] if d["class"] == "aaa_cell"]
        a6 = [d for d in i6["dets"] if d["class"] == "aaa_cell"]
        for d5, d6 in zip(a5, a6):
            keys = ("conf", "xy_robot_m", "yaw_deg", "fit_iou", "skip_reason")
            if all(d5[k] == d6[k] for k in keys) and d5["mask_rle"] == d6["mask_rle"]:
                aaa_same += 1
            else:
                aaa_diff.append({"file": i5["file"], **{k: [d5[k], d6[k]] for k in keys + ("status",)}})
        if len(a5) != len(a6):
            aaa_diff.append({"file": i5["file"], "count": [len(a5), len(a6)]})
    aaa_status = {"same_pose_and_gate": aaa_same, "different": aaa_diff,
                  "status_v5_v6": [sum(d["status"] == "pick" for im in x["images"] for d in im["dets"] if d["class"] == "aaa_cell")
                                   for x in (v5, v6)]}
    # outline quality + previews
    quality = {"cube": [], "screw": []}
    suspects = []
    for im in v6["images"]:
        img = cv2.imread(os.path.join(run, im["file"]))
        canvas = img.copy()
        dets = im["dets"]
        boxes = []
        for d in dets:
            x, y, m = mask_of(d)
            boxes.append((x, y, x + m.shape[1], y + m.shape[0]))
        for k, d in enumerate(dets):
            x, y, m = mask_of(d)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnts = [c + [x, y] for c in cnts]
            if d["class"] not in quality:
                cv2.drawContours(canvas, cnts, -1, (160, 160, 160), 1)
                continue
            col = (0, 200, 0) if d["status"] == "pick" else (0, 0, 255)
            cv2.drawContours(canvas, cnts, -1, col, 1)
            x0, y0, own, win = own_pixels(img, d)
            inter, union = (own & win).sum(), (own | win).sum()
            ys, xs = np.nonzero(own)
            mys, mxs = np.nonzero(win)
            off = math.hypot(xs.mean() - mxs.mean(), ys.mean() - mys.mean()) if len(xs) else None
            q = {"file": im["file"], "conf": round(d["conf"], 2), "mask_px": int(win.sum()), "own_px": int(own.sum()),
                 "ratio": round(win.sum() / max(1, own.sum()), 2), "iou": round(inter / max(1, union), 2),
                 "offset_px": None if off is None else round(off, 1), "side_px": round(math.sqrt(win.sum()), 1),
                 "status": d["status"]}
            for e in ("rect_mm", "length_mm", "shank_mm", "head_mm", "head_end"):
                if d.get(e) is not None:
                    q[e] = d[e]
            # suspect cases: touching another detection, cut by the image/ROI, mask much bigger than the object
            touch = any(j != k and not (b[2] + 2 < boxes[k][0] or boxes[k][2] + 2 < b[0] or b[3] + 2 < boxes[k][1]
                                        or boxes[k][3] + 2 < b[1]) for j, b in enumerate(boxes))
            why = []
            if touch:
                why.append("touching another object")
            if q["ratio"] >= 1.5:
                why.append(f"outline {q['ratio']}x the object's pixels")
            if q["ratio"] <= 0.6:
                why.append(f"partial outline ({q['ratio']}x)")
            if d["class"] == "screw" and d.get("head_end") == "unclear":
                why.append("screw head not found")
            if off is not None and off > 3:
                why.append(f"centre {off:.1f} px off the object's pixels")
            q["why"] = why
            quality[d["class"]].append(q)
            if d.get("centre_px") and d.get("yaw_deg") is not None:
                u, v = d["centre_px"]
                cv2.circle(canvas, (round(u), round(v)), 2, (255, 255, 255), -1)
                cv2.circle(canvas, (round(u), round(v)), 3, (0, 0, 0), 1)
                # jaw-closing direction in the image: project centre +- 8 mm across the grasp
                jaw = math.radians(d["yaw_deg"] + 90.0)
                cx, cy = d["xy_robot_m"]
                ends = []
                for s in (-1, 1):
                    p = [cx + s * 0.008 * math.cos(jaw), cy + s * 0.008 * math.sin(jaw)]
                    ends.append(project_px(a.cameras, p, d["class"]))
                if all(e is not None for e in ends):
                    cv2.line(canvas, tuple(map(round, ends[0])), tuple(map(round, ends[1])), (255, 255, 0), 1)
            if why:
                suspects.append((im["file"], d, why))
        roll, act = im["file"].split("/")
        os.makedirs(os.path.join(a.out, "review", roll), exist_ok=True)
        crop = canvas[300:1040, 300:1700]
        cv2.imwrite(os.path.join(a.out, "review", roll, act), cv2.resize(crop, None, fx=1.3, fy=1.3,
                                                                         interpolation=cv2.INTER_NEAREST),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    # suspects sheet: 6x zoom crops around each suspect, from the review canvases
    os.makedirs(os.path.join(a.out, "suspects"), exist_ok=True)
    tiles = []
    for f, d, why in suspects:
        canvas = cv2.imread(os.path.join(a.out, "review", *f.split("/")))
        u, v = d["centre_px"] or [0, 0]
        cu, cv_ = (u - 300) * 1.3, (v - 300) * 1.3
        x0, y0 = int(max(0, cu - 45)), int(max(0, cv_ - 45))
        t = canvas[y0:y0 + 90, x0:x0 + 90]
        if t.shape[:2] != (90, 90):
            continue
        t = cv2.resize(t, (270, 270), interpolation=cv2.INTER_NEAREST)
        cv2.putText(t, f"{f.split('/')[0][-2:]} {f.split('/')[1][4:7]} {d['class'][:5]}", (4, 14), 0, 0.45, (0, 0, 0), 3)
        cv2.putText(t, f"{f.split('/')[0][-2:]} {f.split('/')[1][4:7]} {d['class'][:5]}", (4, 14), 0, 0.45, (255, 255, 255), 1)
        cv2.putText(t, why[0][:34], (4, 262), 0, 0.4, (0, 0, 0), 3)
        cv2.putText(t, why[0][:34], (4, 262), 0, 0.4, (255, 255, 255), 1)
        tiles.append(t)
    for s in range(0, len(tiles), 48):
        page = tiles[s:s + 48]
        while len(page) % 8:
            page.append(np.full((270, 270, 3), 255, np.uint8))
        sheet = np.vstack([np.hstack(page[i:i + 8]) for i in range(0, len(page), 8)])
        cv2.imwrite(os.path.join(a.out, "suspects", f"suspects_{s // 48:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])

    def summary(rows):
        if not rows:
            return {}
        r = np.array([q["ratio"] for q in rows])
        iou = np.array([q["iou"] for q in rows])
        off = np.array([q["offset_px"] for q in rows if q["offset_px"] is not None])
        out = {"n": len(rows), "ratio_median": round(float(np.median(r)), 2), "ratio_p10_p90": [round(float(np.percentile(r, 10)), 2), round(float(np.percentile(r, 90)), 2)],
               "loose_ratio_ge_1.25": int((r >= 1.25).sum()), "tight_ratio_le_0.8": int((r <= 0.8).sum()),
               "iou_median": round(float(np.median(iou)), 2), "offset_px_median": round(float(np.median(off)), 1),
               "offset_gt_3px": int((off > 3).sum()), "side_px_median": round(float(np.median([q["side_px"] for q in rows])), 1),
               "mask_px_median": int(np.median([q["mask_px"] for q in rows])), "own_px_median": int(np.median([q["own_px"] for q in rows]))}
        for e in ("rect_mm", ):
            vals = [q[e] for q in rows if e in q]
            if vals:
                out["rect_mm_median"] = [round(float(np.median([min(v) for v in vals])), 1), round(float(np.median([max(v) for v in vals])), 1)]
        for e in ("length_mm", "shank_mm", "head_mm"):
            vals = [q[e] for q in rows if q.get(e) is not None]
            if vals:
                out[e + "_median"] = round(float(np.median(vals)), 1)
        if rows[0].get("head_end") is not None or any("head_end" in q for q in rows):
            he = {}
            for q in rows:
                he[q.get("head_end")] = he.get(q.get("head_end"), 0) + 1
            out["head_end"] = he
        return out

    res = {"counts": counts, "aaa": aaa_status, "outline_quality": {c: summary(v) for c, v in quality.items()},
           "suspects": [{"file": f, "class": d["class"], "status": d["status"], "why": w} for f, d, w in suspects],
           "seconds_per_image": {v: round(float(np.mean([im["seconds"] for im in x["images"]])), 2) for v, x in (("v5", v5), ("v6", v6))},
           "images": len(v6["images"])}
    with open(os.path.join(a.out, "outline_check.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(a.out, "outline_quality.csv"), "w") as f:
        keys = ["class", "file", "conf", "status", "mask_px", "own_px", "ratio", "iou", "offset_px", "side_px", "rect_mm",
                "length_mm", "shank_mm", "head_mm", "head_end", "why"]
        f.write(",".join(keys) + "\n")
        for c, rows in quality.items():
            for q in rows:
                f.write(",".join(json.dumps(q.get(k, c if k == "class" else "")).replace(",", ";") for k in keys) + "\n")
    print(json.dumps({k: res[k] for k in ("counts", "outline_quality", "seconds_per_image", "images")}, indent=1))
    print("AAA same:", aaa_same, "different:", len(aaa_diff), "AAA picks v5/v6:", aaa_status["status_v5_v6"])
    print("suspects:", len(suspects))


_CAM = {}


def project_px(cameras, xy, cls):
    from mm.perception.pose_fit import MID_HEIGHT
    from mm.real import frames as F

    if cameras not in _CAM:
        _CAM[cameras] = F.CellCamera.load(cameras, "table", (1920, 1080))
    cam = _CAM[cameras]
    return cam.project([xy[0], xy[1], cam.table_z + MID_HEIGHT[cls]])


def main():
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--run", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--code-root", default=here)
    d.add_argument("--weights", default=None)
    r = sub.add_parser("report")
    r.add_argument("--run", required=True)
    r.add_argument("--v5", required=True)
    r.add_argument("--v6", required=True)
    r.add_argument("--out", required=True)
    for p in (d, r):
        p.add_argument("--cameras", default=os.path.join(os.path.dirname(here), "PhysicalAI-Hackathon-2026-v6", "calibration",
                                                         "cameras.json"))
    a = ap.parse_args()
    for k in ("run", "out", "v5", "v6", "cameras", "weights"):
        if getattr(a, k, None):
            setattr(a, k, os.path.abspath(getattr(a, k)))
    dump(a) if a.cmd == "dump" else report(a)


if __name__ == "__main__":
    main()
