"""
Backend-independent plan execution for the bin sort (Machine Minds step B5).

Shared by the sim backend (sim_backend.py, Isaac) and the real-arm backend to
come (fanuc_backend.py, B8). No Isaac imports.

  validate_plan()     checks a plan_bins sort_plan.json before any motion
  chained_segments()  the TCP trajectory: picks run back to back in plan order
                      (home -> object -> its bin -> next object ... -> home),
                      not home between picks as the starter's cube task does
  PathLog             TCP path length and time per segment and per pick
  in_bin()            placement check: an object resting inside its bin pocket

Segment tuples match sort_scene.segments():
  (name, p0, p1, duration_s, jaw0, jaw1, spread_dir)
"""

import math

import numpy as np

import scene_config as C

PLAN_SOURCE = "MachineMinds-plan_bins"
HOVER_DZ = 0.13  # as sort_scene.segments
HOME_XYZ = (0.30, 0.0, 0.24)  # sort_scene's home tip position
# Long moves are slowed to this TCP speed; short ones keep the starter's times.
MAX_TCP_SPEED = 0.35  # m/s
PRIMS = {"cube": "SortCube", "aaa_cell": "SortAAACell", "screw": "Screw", "cap": "SortCap", "tube": "SortVial"}


def validate_plan(plan):
    """Raise ValueError if a plan_bins plan cannot be executed as written."""
    if plan.get("source") != PLAN_SOURCE or plan.get("task") != "bin_sort":
        raise ValueError("not a plan_bins bin_sort plan")
    if not np.isclose(plan.get("open_gap_m", 0), 2 * C.JAW_OPEN):
        raise ValueError("plan uses a different jaw opening; regenerate perception and plan")
    bins = plan.get("bins") or {}
    for p in plan["picks"]:
        if p["category"] not in PRIMS:
            raise ValueError(f"unknown category {p['category']!r}")
        if p.get("bin") not in bins:
            raise ValueError(f"pick {p['detection_id']} has no bin")
        if not np.allclose(p["place_xy"], bins[p["bin"]]["xy"]):
            raise ValueError(f"pick {p['detection_id']} is not placed at its bin")
        for key in ("xy", "place_xy"):
            if len(p[key]) != 2 or not np.all(np.isfinite(p[key])):
                raise ValueError("invalid plan coordinates")
        for key in ("spread_dir", "place_spread_dir"):
            s = np.asarray(p[key], float)
            if s.shape != (3,) or not np.isclose(np.linalg.norm(s), 1) or abs(s[2]) > 1e-9:
                raise ValueError(f"pick {p['detection_id']}: {key} must be a horizontal unit vector")
    if [p.get("order_index") for p in plan["picks"]] != list(range(len(plan["picks"]))):
        raise ValueError("picks are not in plan order")


def jaw_closed(category):
    """Per-jaw closing target: cubes seat on their faces; others clamp fully."""
    return C.CUBE_JAW_HALF_TARGET if category == "cube" else C.JAW_CLOSED


def _duration(base, p0, p1):
    return max(base, float(np.linalg.norm(np.subtract(p1, p0))) / MAX_TCP_SPEED)


def pick_segments(pick, start):
    """One pick from `start` (a hover point) to above its bin. No return home."""
    p = np.array([*pick["xy"], pick["pick_z"]])
    d = np.array([*pick["place_xy"], pick["place_z"]])
    up, above = p + [0, 0, HOVER_DZ], d + [0, 0, HOVER_DZ]
    start = np.asarray(start, float)
    spread, dest = pick["spread_dir"], pick["place_spread_dir"]
    shut = jaw_closed(pick["category"])
    o, c = C.JAW_OPEN, shut
    return [
        ("approach", start, up, _duration(1.2, start, up), o, o, spread),
        ("descend", up, p, 0.8, o, o, spread),
        ("close", p, p, 0.6, o, c, spread),
        ("dwell", p, p, 0.5, c, c, spread),
        ("lift", p, up, 0.8, c, c, spread),
        ("carry", up, above, _duration(1.4, up, above), c, c, dest),
        ("lower", above, d, 0.8, c, c, dest),
        ("release", d, d, 0.6, c, o, dest),
        ("settle", d, d, 0.6, o, o, dest),
        ("retreat", d, above, 0.8, o, o, dest),
    ]


def chained_segments(picks, home=HOME_XYZ):
    """Yields (pick, segments): each pick starts where the previous one ended.

    The last pick also carries the final ("home", ...) segment.
    """
    here = np.asarray(home, float)
    for k, pick in enumerate(picks):
        segs = pick_segments(pick, here)
        here = segs[-1][2]
        if k == len(picks) - 1:
            segs.append(("home", here, np.asarray(home, float), _duration(1.1, here, home), C.JAW_OPEN, C.JAW_OPEN, None))
        yield pick, segs


SAMPLE_HZ = 30  # IK samples per second of motion: the sim's frame rate
# Segments that turn the wrist roll from its current direction to the given one.
TURNING = ("approach", "carry", "reroute", "row")


def solve_segment(q, name, p0, p1, duration, direction, limits, tool, hz=SAMPLE_HZ):
    """Joint solution for every sample of one straight TCP segment from joint
    state q, fingers down, without moving anything (sim and real-arm planning).

    Returns ([(q, target, t), ...], failed sample or None); t is the eased
    progress 0..1. direction: the jaw-closing direction to hold (TURNING
    segments turn to it from q's), or None for any wrist roll.
    """
    import fk
    import ik

    spread_start = None
    if direction is not None and name in TURNING:
        r, _ = fk.fk(q)
        s = r @ np.array(C.FINGER_SPREAD_AXIS)
        s[2] = 0
        spread_start = s / np.linalg.norm(s)
    n = max(1, round(duration * hz))
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    down = [0.0, 0.0, -1.0]
    out = []
    for i in range(n):
        t = (i + 1) / n
        t = t * t * (3 - 2 * t)  # sort_scene.ease
        target = p0 + (p1 - p0) * t
        d = direction
        if spread_start is not None:
            ang = np.arctan2(spread_start[0] * direction[1] - spread_start[1] * direction[0], np.dot(spread_start[:2], direction[:2])) * t
            cs, sn = np.cos(ang), np.sin(ang)
            d = np.array([spread_start[0] * cs - spread_start[1] * sn, spread_start[0] * sn + spread_start[1] * cs, 0.0])
        q2, ok = ik.solve(q, target, down, limits, tool_offset=tool, spread_dir=d)
        if not ok:
            q2, ok = ik.solve(q, target, down, limits, tool_offset=tool, spread_dir=d, iters=500, damping=0.02)
        rr, actual = fk.tool_pose(q2, tool)
        aligned = d is None or abs(np.dot(rr @ np.array(C.FINGER_SPREAD_AXIS), d)) > 0.999
        if not ok or np.linalg.norm(actual - target) > 0.001 or not aligned:
            return out, i
        q = q2
        out.append((q, target, t))
    return out, None


def check_path(q, legs, limits, tool):
    """IK along a chain of (name, p0, p1, duration, direction) legs from joint
    state q. None if every sample solves, else (leg name, sample)."""
    for name, p0, p1, duration, direction in legs:
        frames, failed = solve_segment(q, name, p0, p1, duration, direction, limits, tool)
        if failed is not None:
            return name, failed
        q = frames[-1][0]
    return None


class PathLog:
    """Accumulates commanded and measured TCP path length and time."""

    def __init__(self, dt):
        self.dt = dt
        self.segments = []
        self._cur = None

    def start(self, name):
        self._cur = {"segment": name, "frames": 0, "commanded_m": 0.0, "measured_m": 0.0, "_last": None}

    def step(self, commanded, measured):
        cur = self._cur
        if cur["_last"] is not None:
            cur["commanded_m"] += float(np.linalg.norm(np.subtract(commanded, cur["_last"][0])))
            cur["measured_m"] += float(np.linalg.norm(np.subtract(measured, cur["_last"][1])))
        cur["_last"] = (np.asarray(commanded, float), np.asarray(measured, float))
        cur["frames"] += 1

    def end(self):
        cur = self._cur
        cur.pop("_last")
        cur["time_s"] = round(cur["frames"] * self.dt, 4)
        cur["commanded_m"] = round(cur["commanded_m"], 5)
        cur["measured_m"] = round(cur["measured_m"], 5)
        self.segments.append(cur)
        self._cur = None
        return cur


def summarise(segments):
    return {
        "time_s": round(sum(s["time_s"] for s in segments), 3),
        "tcp_path_commanded_m": round(sum(s["commanded_m"] for s in segments), 4),
        "tcp_path_measured_m": round(sum(s["measured_m"] for s in segments), 4),
    }


def in_bin(centre_world, bin_cfg, margin=0.002):
    """Object centre inside the bin pocket (xy) and below its rim, above its floor."""
    z0 = C.TABLE_TOP_Z + C.SURFACE_DZ
    half = bin_cfg["pocket"] / 2 + margin
    dx, dy = (centre_world[0] - bin_cfg["xy"][0], centre_world[1] - bin_cfg["xy"][1])
    t = math.radians(-bin_cfg.get("yaw_deg", 0.0))  # into the bin's frame
    lx, ly = math.cos(t) * dx - math.sin(t) * dy, math.sin(t) * dx + math.cos(t) * dy
    inside_xy = abs(lx) <= half and abs(ly) <= half
    z = centre_world[2] - z0
    return bool(inside_xy and bin_cfg["floor_t"] - 0.002 <= z <= bin_cfg["wall_h"]), {
        "offset_mm": [round(dx * 1000, 1), round(dy * 1000, 1)],
        "height_mm": round(z * 1000, 1),
    }


def bin_at(xy, bins):
    """The bin (of a {name: bin} table) whose centre is nearest xy: v4 bins are
    identical, so a pick's destination is identified by its place_xy."""
    return min(bins.values(), key=lambda b: math.dist(xy[:2], b["xy"]))


def bin_contents(poses, bins, prims=PRIMS):
    """Scoring (ground truth): {bin name: sorted object kinds inside it}."""
    out = {name: [] for name in bins}
    for obj, p in poses.items():
        kind = next((k for k, pre in prims.items() if obj.startswith(pre)), None)
        if kind is None:
            continue
        for name, b in bins.items():
            if in_bin(p["centre"], b)[0]:
                out[name].append(kind)
    return {name: sorted(v) for name, v in out.items()}


def sort_ok(contents, n_targets, distractors=("cap", "tube")):
    """v4 success rule: each bin holds one type at most, every target is in a bin,
    no distractor is in a bin."""
    kinds = [k for v in contents.values() for k in v]
    return (all(len(set(v)) <= 1 for v in contents.values())
            and sum(k not in distractors for k in kinds) == n_targets
            and not any(k in distractors for k in kinds))


def xy_in_bin(xy, bin_cfg, margin=0.005):
    """Table-plane point inside a bin's outer square (its own yawed frame)."""
    dx, dy = xy[0] - bin_cfg["xy"][0], xy[1] - bin_cfg["xy"][1]
    t = math.radians(-bin_cfg.get("yaw_deg", 0.0))
    lx, ly = math.cos(t) * dx - math.sin(t) * dy, math.sin(t) * dx + math.cos(t) * dy
    return abs(lx) <= bin_cfg["outer"][0] / 2 + margin and abs(ly) <= bin_cfg["outer"][1] / 2 + margin


def object_centre(prim_name, pos, rot):
    """World centre of a task object from its root pose (cell.py origins)."""
    local = (0.0, 0.0, 0.0)
    if prim_name.startswith("SortVial"):
        local = (0, 0, C.VIAL_LENGTH / 2)
    elif prim_name.startswith("SortCap"):
        local = (0, 0, C.CAP_HEIGHT / 2)
    elif prim_name.startswith("SortCube"):
        local = (0, 0, C.CUBE_EDGE / 2)
    elif prim_name.startswith("Screw"):
        local = (0, 0, -(C.SCREW_HEAD_H + C.SCREW_LENGTH) / 2)  # shank middle
    return np.asarray(pos, float) + np.asarray(rot) @ np.array(local)


def yaw_tilt(rot):
    rot = np.asarray(rot)
    return (
        float(math.degrees(math.atan2(rot[1, 0], rot[0, 0]))),
        float(math.degrees(math.acos(np.clip(rot[2, 2], -1, 1)))),
    )
