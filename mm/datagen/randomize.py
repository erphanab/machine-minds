"""
Seeded random cell layouts for synthetic YOLO data (Machine Minds step A2).

No Isaac imports, so it runs and is tested on the host. One seed gives one
layout dict. scene_config.py loads it through MM_SCENE_JSON; gen_yolo_data.py
applies one per frame without rebuilding the stage.

    .venv/bin/python -m mm.datagen.randomize --seed 7 --out runs/layout_7.json

Layout keys read by scene_config._apply_layout: cubes, caps, vials, aaa_cells, screws,
bins (configs/bins.yaml, B1). Keys for datagen only: lighting, table_cam, arm,
and per-object drop_rpy_deg.
"""

import argparse
import json
import math
import os
import random

import yaml

import scene_config as C

BINS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "bins.yaml")
BIN_CLEARANCE = 0.015  # object footprint to bin outer wall, metres

# Reachable area the starter samples cube centres from (hackathon_cubes).
TARGET_X = (0.24, 0.46)
TARGET_Y = (-0.15, 0.28)
# Distractor tubes may overhang it; their centres stay close to the work area.
DISTRACTOR_X = (0.22, 0.48)
DISTRACTOR_Y = (-0.18, 0.30)

N_TARGETS = (6, 10)
N_DISTRACTORS = (0, 4)  # tubes and caps together
TARGET_CLASSES = ("cube", "screw", "aaa_cell")  # screws replaced caps, 2026-10-23
DISTRACTOR_KINDS = ("vial", "cap")
CUBE_COLOURS = tuple(C.CUBE_MATS)
# Minimum centre spacing; footprints must also clear each other by MIN_GAP.
SPACINGS = (0.065, 0.030, 0.0)
SPACING_WEIGHTS = (0.8, 0.1, 0.1)  # ~20% of scenes cluttered
HARD_SPACING_WEIGHTS = (0.0, 0.5, 0.5)
MIN_GAP = 0.0005
P_DROPPED = 0.08  # released tilted above the mat; settles under physics
P_CAP_WIDE_UP = 0.15
LIGHT_SCALE = (0.6, 1.4)
LIGHT_KELVIN = (4000.0, 6500.0)
CAM_JITTER = 0.005  # +/- per axis on eye and look_at
P_ARM_IN_VIEW = 0.3
HARD_P_ARM_IN_VIEW = 0.7
ARM_TIP_Z = (0.10, 0.22)  # fingertips above the work surface when in view

# Largest count per class one frame can hold; datagen builds this pool once.
MAX_PER_CLASS = {"cube": 10, "screw": 10, "aaa_cell": 10, "vial": 4, "cap": 4}
LAYOUT_KEYS = {"cube": "cubes", "screw": "screws", "aaa_cell": "aaa_cells", "vial": "vials", "cap": "caps"}

# Seed ranges keep splits disjoint, so near-identical frames never cross them.
SPLITS = {
    "train": (0, 2000),
    "val": (100_000, 100_300),
    "test": (200_000, 200_300),
    "hard": (300_000, 300_200),
}


def footprint(kind, xy, yaw_deg):
    """Tabletop footprint as a capsule: (segment end a, end b, radius)."""
    if kind == "cube":
        half, r = 0.0, C.CUBE_EDGE * math.sqrt(2) / 2
    elif kind == "cap":
        half, r = 0.0, C.CAP_RADIUS
    elif kind == "aaa_cell":
        half, r = C.AAA_CELL_LENGTH / 2 - C.AAA_CELL_RADIUS, C.AAA_CELL_RADIUS
    elif kind == "vial":
        half, r = C.VIAL_LENGTH / 2 - C.VIAL_RADIUS, C.VIAL_RADIUS
    elif kind == "screw":
        # xy is the shank middle, 16.65 mm from the head face: a capsule of
        # head radius reaching the head face, 3.3 mm long at the tip end.
        r = C.SCREW_HEAD_D / 2
        half = (C.SCREW_HEAD_H + C.SCREW_LENGTH) / 2 - r
    else:
        raise ValueError(f"unknown object kind {kind!r}")
    ux, uy = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    a = (xy[0] - half * ux, xy[1] - half * uy)
    b = (xy[0] + half * ux, xy[1] + half * uy)
    return a, b, r


def _point_segment(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = dx * dx + dy * dy
    t = 0.0 if n == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / n))
    return math.dist(p, (a[0] + t * dx, a[1] + t * dy))


def _segments_cross(a, b, c, d):
    def side(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    return side(a, b, c) * side(a, b, d) < 0 and side(c, d, a) * side(c, d, b) < 0


def footprint_gap(f, g):
    """Clearance between two capsule footprints; negative means overlap."""
    (a, b, r1), (c, d, r2) = f, g
    if _segments_cross(a, b, c, d):
        return -(r1 + r2)
    dist = min(
        _point_segment(a, c, d),
        _point_segment(b, c, d),
        _point_segment(c, a, b),
        _point_segment(d, a, b),
    )
    return dist - r1 - r2


_BIN_CFG = None


def _bin_cfg(path=BINS_PATH):
    global _BIN_CFG
    if _BIN_CFG is None:
        with open(path) as f:
            _BIN_CFG = yaml.safe_load(f)
    return _BIN_CFG


def load_bin_design():
    """The one bin design shared by every bin (configs/bins.yaml "design")."""
    return dict(_bin_cfg()["design"])


def load_bins():
    """Fixed bin table: each bin = design + its colour, material, xy, yaw."""
    design = load_bin_design()
    return {name: dict(design, **cfg) for name, cfg in _bin_cfg()["bins"].items()}


def bin_gap(f, b, samples=11):
    """Clearance from a capsule footprint to a bin's outer square, in the bin's
    own (yawed) frame. <= -r means inside."""
    (a, c, r) = f
    cx, cy = b["xy"]
    hx, hy = b["outer"][0] / 2, b["outer"][1] / 2
    t_ = math.radians(-b.get("yaw_deg", 0.0))
    ct, st = math.cos(t_), math.sin(t_)
    best = math.inf
    for k in range(samples):
        t = k / (samples - 1)
        px, py = a[0] + t * (c[0] - a[0]) - cx, a[1] + t * (c[1] - a[1]) - cy
        px, py = ct * px - st * py, st * px + ct * py
        dx, dy = abs(px) - hx, abs(py) - hy
        best = min(best, math.hypot(max(dx, 0.0), max(dy, 0.0)) if dx > 0 or dy > 0 else 0.0)
    return best - r


# ---- B8: bins anywhere within reach, at any yaw
BIN_REGION_X = (-0.30, 0.54)  # proposal box; bin_pose_ok decides
BIN_REGION_Y = (-0.54, 0.54)
BIN_BASE_KEEPOUT = 0.19  # bin centre to robot base axis (base + cabling)
BIN_SEPARATION = 0.020  # between bins' circumscribed circles
BIN_IMAGE_MARGIN = 40  # px at 4K: the whole bin stays inside the table image


def _bin_radius():
    d = load_bin_design()
    return math.hypot(d["outer"][0], d["outer"][1]) / 2


_REACH = None


def bin_pose_ok(xy):
    """A bin centred at xy (any yaw) is on the table, clear of the robot base,
    reachable at release and hover height, and fully in the table camera."""
    global _REACH
    import numpy as np

    import ik
    from test_support import camera_model

    d, R = load_bin_design(), _bin_radius()
    x, y = xy
    if math.hypot(x, y) < BIN_BASE_KEEPOUT:
        return False
    if not (C.TABLE_X[0] + R < x < C.TABLE_X[1] - C.EDGE_BAND_W - R):
        return False
    if not (C.TABLE_Y[0] + C.EDGE_BAND_W + R < y < C.TABLE_Y[1] - C.EDGE_BAND_W - R):
        return False
    if _REACH is None:
        ik.set_tool_axis(C.TOOL_AXIS)
        ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
        _REACH = dict(
            limits=np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)]),
            q0=np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)]),
            tool=tuple(np.array(C.TOOL_AXIS) * C.TOOL_LENGTH),
            cam=camera_model(dict(C.TABLE_CAM, resolution=tuple(2 * v for v in C.TABLE_CAM["resolution"]))),
        )
    r = _REACH
    for dz in (d["release_z"], d["release_z"] + 0.13):  # release, then hover (executor)
        if not ik.solve(r["q0"], [x, y, C.SURFACE_DZ + dz], [0, 0, -1], r["limits"], tool_offset=r["tool"])[1]:
            return False
    cam, z0 = r["cam"], C.TABLE_TOP_Z + C.SURFACE_DZ
    m = BIN_IMAGE_MARGIN
    for k in range(16):
        a = 2 * math.pi * k / 16
        for h in (0.0, d["wall_h"]):
            p = cam.project([x + R * math.cos(a), y + R * math.sin(a), z0 + h])
            if p is None or not (m <= p[0] < cam.W - m and m <= p[1] < cam.H - m):
                return False
    return True


def sample_bins(rng, tries=4000):
    """Random reachable, visible, separated bin poses for the bins in bins.yaml."""
    fixed = load_bins()
    R = _bin_radius()
    out = {}
    for name in sorted(fixed):
        for _ in range(tries):
            xy = (rng.uniform(*BIN_REGION_X), rng.uniform(*BIN_REGION_Y))
            if any(math.dist(xy, b["xy"]) < 2 * R + BIN_SEPARATION for b in out.values()):
                continue
            if bin_pose_ok(xy):
                out[name] = dict(fixed[name], xy=[float(xy[0]), float(xy[1])], yaw_deg=rng.uniform(0.0, 90.0))
                break
        else:
            raise ValueError("cannot place bins")
    return out


def _bin_in_crop(xy, roi_px):
    """The whole bin (base and rim) inside the 4K perception crop (configs/roi.yaml)."""
    import numpy as np

    bin_pose_ok((0.4, 0.0))  # builds the cached camera
    cam, z0, d, rad = _REACH["cam"], C.TABLE_TOP_Z + C.SURFACE_DZ, load_bin_design(), _bin_radius()
    x0, y0, w, h = roi_px
    for k in range(16):
        a = 2 * math.pi * k / 16
        for hh in (0.0, d["wall_h"]):
            p = cam.project([xy[0] + rad * math.cos(a), xy[1] + rad * math.sin(a), z0 + hh])
            if p is None or not (x0 <= p[0] < x0 + w and y0 <= p[1] < y0 + h):
                return False
    return bool(np.isfinite(xy).all())


def sample_bins_stress(rng, tries=20000):
    """v4 detection stress test (not reachable): one bin 5-40 mm from the yellow
    edge tape, one against the robot base, one anywhere; all inside the crop."""
    import yaml

    with open(os.path.join(os.path.dirname(BINS_PATH), "roi.yaml")) as f:
        roi = yaml.safe_load(f)
    roi_px = (roi["x0"], roi["y0"], roi["width"], roi["height"])
    fixed, rad = load_bins(), _bin_radius()
    inner_y = (C.TABLE_Y[0] + C.EDGE_BAND_W, C.TABLE_Y[1] - C.EDGE_BAND_W)
    out = {}
    for k, name in enumerate(sorted(fixed)):
        for _ in range(tries):
            if k == 0:  # against a side tape
                side = rng.choice((0, 1))
                gap = rng.uniform(0.005, 0.040)
                y = inner_y[0] + rad + gap if side == 0 else inner_y[1] - rad - gap
                xy = (rng.uniform(C.TABLE_X[0] + rad, 0.6), y)
            elif k == 1:  # against the robot base column
                a, r = rng.uniform(-math.pi, math.pi), rng.uniform(0.16, 0.24)
                xy = (r * math.cos(a), r * math.sin(a))
            else:
                xy = (rng.uniform(C.TABLE_X[0] + rad, 0.6), rng.uniform(inner_y[0] + rad, inner_y[1] - rad))
            if math.hypot(*xy) < 0.16 or any(math.dist(xy, b["xy"]) < 2 * rad + BIN_SEPARATION for b in out.values()):
                continue
            if _bin_in_crop(xy, roi_px):
                out[name] = dict(fixed[name], xy=[float(xy[0]), float(xy[1])], yaw_deg=rng.uniform(0.0, 90.0))
                break
        else:
            raise ValueError("cannot place stress bins")
    return out


def _place(rng, kind, placed, spacing, box, bins=(), tries=2000):
    for _ in range(tries):
        xy = (rng.uniform(*box[0]), rng.uniform(*box[1]))
        yaw = rng.uniform(-180.0, 180.0)
        f = footprint(kind, xy, yaw)
        if all(bin_gap(f, b) >= BIN_CLEARANCE for b in bins) and all(
            math.dist(xy, q["xy"]) >= spacing
            and footprint_gap(f, footprint(q["kind"], q["xy"], q["yaw_deg"])) >= MIN_GAP
            for q in placed
        ):
            return {"kind": kind, "xy": xy, "yaw_deg": yaw}
    return None


def _drop_rpy(rng, kind):
    """Release orientation for an object dropped tilted onto the mat."""
    if kind == "aaa_cell" and rng.random() < 0.5:
        return [0.0, rng.choice((-90.0, 90.0)), rng.uniform(-180.0, 180.0)]  # on end
    return [rng.uniform(-60.0, 60.0), rng.uniform(-60.0, 60.0), rng.uniform(-180.0, 180.0)]


def _arm_pose(rng, hard):
    """Parked capture pose, or fingertips over the table to occlude objects."""
    if rng.random() >= (HARD_P_ARM_IN_VIEW if hard else P_ARM_IN_VIEW):
        return {"mode": "parked", "joints_deg": dict(C.CAPTURE_POSE_DEG)}
    import numpy as np
    import ik

    ik.set_tool_axis(C.TOOL_AXIS)
    ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
    limits = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])
    q0 = np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)])
    tool = tuple(np.array(C.TOOL_AXIS) * C.TOOL_LENGTH)
    for _ in range(20):
        tip = [
            rng.uniform(*TARGET_X),
            rng.uniform(*TARGET_Y),
            C.SURFACE_DZ + rng.uniform(*ARM_TIP_Z),
        ]
        q, ok = ik.solve(q0, tip, [0.0, 0.0, -1.0], limits, tool_offset=tool)
        if ok:
            deg = np.rad2deg(q)
            return {
                "mode": "in_view",
                "tip_xyz": tip,
                "joints_deg": {f"J{i + 1}": float(deg[i]) for i in range(6)},
            }
    return {"mode": "parked", "joints_deg": dict(C.CAPTURE_POSE_DEG)}


def _jitter(rng, v):
    return [float(c + rng.uniform(-CAM_JITTER, CAM_JITTER)) for c in v]


def sample_layout(seed, hard=False, max_restarts=50, bins=True, bin_mode="fixed"):
    """One reproducible layout. hard=True biases to clutter, occlusion, odd light.

    bins=True adds the bins (scene_config then shows them) and keeps every
    object BIN_CLEARANCE away from them; bins=False is the bin-free scene Part
    A's dataset was generated with. bin_mode "fixed" uses configs/bins.yaml
    (B5-B7); "random" places them anywhere reachable and visible, any yaw (B8).
    """
    rng = random.Random(seed)
    if not bins:
        bin_table = {}
    elif bin_mode == "random":
        bin_table = sample_bins(random.Random(seed * 7919 + 1))  # own stream: objects stay seeded
    elif bin_mode == "stress":
        bin_table = sample_bins_stress(random.Random(seed * 7919 + 1))
    elif bin_mode == "fixed":
        bin_table = load_bins()
    else:
        raise ValueError(bin_mode)
    spacing = rng.choices(SPACINGS, HARD_SPACING_WEIGHTS if hard else SPACING_WEIGHTS)[0]
    for _ in range(max_restarts):
        n_targets = rng.randint(*N_TARGETS)
        kinds = list(TARGET_CLASSES) + [
            rng.choice(TARGET_CLASSES) for _ in range(n_targets - len(TARGET_CLASSES))
        ]
        distractors = [rng.choice(DISTRACTOR_KINDS) for _ in range(rng.randint(*N_DISTRACTORS))]
        counts = {k: (kinds + distractors).count(k) for k in MAX_PER_CLASS}
        if any(counts[k] > MAX_PER_CLASS[k] for k in counts):
            continue
        placed = []
        # Distractors first, long tubes leading: hardest to fit once the table fills.
        distractors.sort(key=lambda k: k != "vial")
        for kind, box in [(k, (DISTRACTOR_X, DISTRACTOR_Y)) for k in distractors] + [
            (k, (TARGET_X, TARGET_Y)) for k in kinds
        ]:
            obj = _place(rng, kind, placed, spacing, box, list(bin_table.values()))
            if obj is None:
                break
            placed.append(obj)
        else:
            break
    else:
        raise ValueError(f"seed {seed}: cannot place objects at spacing {spacing}")

    layout = {
        "seed": seed,
        "hard": hard,
        "spacing_m": spacing,
        **{key: [] for key in LAYOUT_KEYS.values()},
    }
    if bin_table:
        layout["bins"] = {k: dict(v) for k, v in bin_table.items()}
        # v4: printed bins' gloss, own stream so the objects stay as seeded
        layout["bin_roughness"] = random.Random(seed * 104729 + 3).uniform(*C.BIN_ROUGHNESS_RANGE)
    for obj in placed:
        kind, xy, yaw = obj["kind"], [float(v) for v in obj["xy"]], float(obj["yaw_deg"])
        entry = {"xy": xy, "yaw_deg": yaw}
        if kind == "cube":
            entry["colour"] = rng.choice(CUBE_COLOURS)
        if kind == "cap":
            entry["orientation"] = (
                "wide_end_up" if rng.random() < P_CAP_WIDE_UP else "narrow_end_up"
            )
        if kind != "vial" and rng.random() < P_DROPPED:
            entry["drop_rpy_deg"] = _drop_rpy(rng, kind)
        layout[LAYOUT_KEYS[kind]].append(entry)

    if hard:
        lo, hi = LIGHT_SCALE
        scale = rng.choice((rng.uniform(lo, lo + 0.15), rng.uniform(hi - 0.15, hi)))
        kelvin = rng.choice(LIGHT_KELVIN)
    else:
        scale, kelvin = rng.uniform(*LIGHT_SCALE), rng.uniform(*LIGHT_KELVIN)
    layout["lighting"] = {"scale": scale, "kelvin": kelvin}
    layout["table_cam"] = {
        "eye": _jitter(rng, C.TABLE_CAM["eye"]),
        "look_at": _jitter(rng, C.TABLE_CAM["look_at"]),
        "up": list(C.TABLE_CAM["up"]),
    }
    layout["arm"] = _arm_pose(rng, hard)
    return layout


def split_seeds(split):
    lo, hi = SPLITS[split]
    return range(lo, hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--no-bins", action="store_true", help="bin-free scene (Part A data)")
    parser.add_argument("--bins", choices=["fixed", "random"], default="fixed", help="B8: random")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    layout = sample_layout(args.seed, hard=args.hard, bins=not args.no_bins, bin_mode=args.bins)
    with open(args.out, "w") as f:
        json.dump(layout, f, indent=2)
    print(f"[randomize] seed {args.seed} -> {args.out}")


if __name__ == "__main__":
    main()
