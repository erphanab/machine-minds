"""
Top-down grasp selection and pickability checks (Machine Minds step B3).

Host-side, image + calibrated camera only. For one detected object:
  1. width     the object's width across the jaws, plus clearance on each side,
               must fit the open fingertip gap
  2. fingers   the two finger footprints at the grasp yaw, grown by a margin,
               must not overlap any neighbour or bin
  3. reach     IK must succeed at hover, grasp, above-bin and release poses
Cubes try both face pairs (yaw and yaw + 90) before being skipped; v6 also the symmetric twins (+180 deg).

Gripper numbers (scene_config, SMC MHZL2-16D2, 6 mm per jaw): the straight
faces open to 12 mm, but the hooked tips that close on a tabletop part open to
2 * (JAW_OPEN + FINGER_INNER_TIP) = 16 mm. The guide's rule "width <= opening
- 2 x 2.5 mm" is applied to that 16 mm tip gap (against 12 mm it would reject
the 10 mm cubes the starter picks). AAA cells (10.5 mm) leave 2.75 mm per
side; AA cells (0.75 mm) proved marginal and were replaced. Screws are gripped
across the 6 mm shank. Caps (20.8-23.6 mm plug) do not fit this gripper at
all, which is why screws replaced them as a target.
"""

import math
import os

import cv2
import numpy as np

import ik
import scene_config as C
from mm.perception.pose_fit import MID_HEIGHT
from mm.real import frames as F

TIP_OPEN_GAP = 2 * (C.JAW_OPEN + C.FINGER_INNER_TIP)  # 16 mm
FINGER_MARGIN = 0.003  # finger footprint growth vs neighbours (guide: +3 mm)
HOVER_DZ = 0.13  # sort_scene.segments lifts 0.13 m above pick and place
HOME_XYZ = (0.30, 0.0, 0.24)  # sort_scene's home tip position
# v6: before a target is called unreachable, try its symmetric grasp angles too (spread_options)
SYMMETRIC = "6" not in {s.strip() for s in os.environ.get("MM_V5_STEPS", "").split(",")}

# Per class: width across the jaws (m), clearance per side (m), fingertip height.
GRASP = {
    "cube": dict(width=C.CUBE_EDGE, clearance=0.0025, pick_z=C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE),
    "aaa_cell": dict(width=2 * C.AAA_CELL_RADIUS, clearance=0.0025, pick_z=C.SURFACE_DZ + C.TIP_CLEARANCE),
    "cap": dict(
        width=2 * C.CAP_PLUG_BASE_RADIUS,  # narrow end up: jaws close on the plug
        clearance=0.0025,
        pick_z=C.SURFACE_DZ + C.CAP_PICK_HEIGHT,
    ),
    # Jaws close across the shank middle. Tips at TIP_CLEARANCE stay under the
    # lying screw's axis (~4.5 mm), as the v0.5 screw-picking cell was set up.
    "screw": dict(width=C.SCREW_SHANK_D, clearance=0.0025, pick_z=C.SURFACE_DZ + C.TIP_CLEARANCE),
}


def width_ok(cls):
    g = GRASP[cls]
    return g["width"] + 2 * g["clearance"] <= TIP_OPEN_GAP + 1e-9


def spread_options(det):
    """Jaw-closing directions to try, as (unit 2D vector, grasp yaw deg).

    grasp yaw is the direction the jaws close, from world +X.
    v6: then every symmetric twin (the jaws close the same way, but the wrist is turned 180 deg, so the IK differs):
    cube yaw + 90 + 180 and yaw + 180 (with the two above: all four of +0/90/180/270), screw / AAA cell + 180.
    "6" in MM_V5_STEPS: v5's list only.
    """
    cls, yaw = det["class"], det.get("yaw_deg")
    if cls == "cube":
        # plan_sort convention: spread = (-sin yaw, cos yaw) closes on one face
        # pair; spread along yaw closes on the other.
        base = [yaw + 90.0, yaw]
    elif cls in ("aaa_cell", "screw"):
        base = [yaw + 90.0]  # across the long axis
    else:
        return [_unit(90.0)]  # round: any; world +Y as sort_scene's default
    if not SYMMETRIC:
        return [_unit(a) for a in base]
    return [_unit(a) for a in base] + [_unit(a + 180.0) for a in base]


def _unit(deg):
    a = math.radians(deg)
    return np.array([math.cos(a), math.sin(a)]), float((deg + 180.0) % 360.0 - 180.0)


def finger_rects(xy, spread, margin=FINGER_MARGIN):
    """Table-plane footprints of the two open fingers, grown by margin."""
    s = np.asarray(spread, float)
    t = np.array([-s[1], s[0]])
    half_in = TIP_OPEN_GAP / 2 - margin
    half_out = TIP_OPEN_GAP / 2 + C.FINGER_THICK + margin
    half_w = C.FINGER_WIDTH / 2 + margin
    rects = []
    for sign in (1, -1):
        a, b = sign * half_in, sign * half_out
        rects.append(
            np.array([np.asarray(xy) + u * s + v * t for u, v in ((a, -half_w), (b, -half_w), (b, half_w), (a, half_w))])
        )
    return rects


def neighbour_shapes(dets, cam, exclude_id):
    """Other detections as table-plane polygons (YOLO outline back-projected
    at the object's mid height)."""
    shapes = []
    for d in dets:
        if d["id"] == exclude_id:
            continue
        z = cam.table_z + MID_HEIGHT.get(d["class"], 0.005)
        pts = [cam.to_plane(u, v, z=z) for u, v in d["polygon_px"]]
        pts = [p[:2] for p in pts if p is not None]
        if len(pts) >= 3:
            shapes.append(("object", d["id"], np.array(pts)))
    return shapes


def bin_shapes(bins):
    """Each bin's outer square in the table plane, at its yaw."""
    out = []
    for name, b in bins.items():
        (cx, cy), (w, h) = b["xy"], b["outer"]
        t = math.radians(b.get("yaw_deg", 0.0))
        rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        corners = np.array([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)])
        out.append(("bin", name, corners @ rot.T + [cx, cy]))
    return out


def finger_collisions(xy, spread, shapes, res=0.00025, half=0.06):
    """Which shapes the finger footprints touch, by rasterising a local window."""
    def grid(poly):
        return np.round((np.asarray(poly) - np.asarray(xy) + half) / res).astype(np.int32)

    n = int(2 * half / res)
    fingers = np.zeros((n, n), np.uint8)
    for r in finger_rects(xy, spread):
        cv2.fillPoly(fingers, [grid(r)], 1)
    hits = []
    for kind, ident, poly in shapes:
        if np.min(np.linalg.norm(poly - xy, axis=1)) > half + 0.1:
            continue  # far away: cannot reach the window
        m = np.zeros((n, n), np.uint8)
        cv2.fillPoly(m, [grid(poly)], 1)
        if (m & fingers).any():
            hits.append(f"{kind}:{ident}")
    return hits


class Reach:
    """Sequential IK from sort_scene's home pose, wrist roll pinned per pose."""

    def __init__(self):
        ik.set_tool_axis(C.TOOL_AXIS)
        ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
        self.limits = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])
        self.tool = tuple(np.array(C.TOOL_AXIS) * C.TOOL_LENGTH)
        q = np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)])
        self.home, ok = ik.solve(q, HOME_XYZ, [0, 0, -1], self.limits, tool_offset=self.tool)
        if not ok:
            raise RuntimeError("home IK failed")

    def _solve(self, q, xyz, spread):
        q2, ok = ik.solve(q, xyz, [0, 0, -1], self.limits, tool_offset=self.tool, spread_dir=spread)
        if not ok:  # sort_scene's retry
            q2, ok = ik.solve(q, xyz, [0, 0, -1], self.limits, tool_offset=self.tool,
                              spread_dir=spread, iters=500, damping=0.02)
        return q2, ok

    def check(self, xy, pick_z, spread, place_xy, place_z, place_spread):
        """Returns the first failing stage name, or None if all poses solve."""
        sp = [*spread, 0.0]
        psp = [*place_spread, 0.0]
        q = self.home
        for stage, xyz, s in (
            ("hover", [*xy, pick_z + HOVER_DZ], sp),
            ("grasp", [*xy, pick_z], sp),
            ("lift", [*xy, pick_z + HOVER_DZ], sp),
            ("above_bin", [*place_xy, place_z + HOVER_DZ], psp),
            ("release", [*place_xy, place_z], psp),
        ):
            q, ok = self._solve(q, xyz, s)
            if not ok:
                return stage
        return None


def plan_grasp(det, dets, cam, bins, bin_name, reach):
    """Choose a grasp for one eligible detection.

    Returns (grasp dict, None) or (None, skip_reason) with details.
    """
    cls = det["class"]
    if cls not in GRASP:
        return None, ("no_grasp_model", {})
    if not width_ok(cls):
        g = GRASP[cls]
        return None, ("exceeds_jaw_opening", {"width_mm": g["width"] * 1000, "tip_gap_mm": TIP_OPEN_GAP * 1000})
    g = GRASP[cls]
    b = bins[bin_name]
    place_xy = list(b["xy"])
    place_z = C.SURFACE_DZ + b["release_z"]
    shapes = neighbour_shapes(dets, cam, det["id"]) + bin_shapes(bins)
    outline = det.get("grasp") == "outline"  # v6 cubes and screws: only bins block the fingers
    options = spread_options(det)
    if outline:  # a collision-free option first; neighbours touched are logged, not refused
        hits_of = {i: finger_collisions(np.asarray(det["xy_robot_m"]), s, shapes) for i, (s, _) in enumerate(options)}
        order = sorted(range(len(options)), key=lambda i: (any(h.startswith("bin:") for h in hits_of[i]), len(hits_of[i])))
        options = [options[i] for i in order]
    tried = []
    for spread, grasp_yaw in options:
        hits = finger_collisions(np.asarray(det["xy_robot_m"]), spread, shapes)
        if outline and hits and not any(h.startswith("bin:") for h in hits):
            tried.append({"grasp_yaw_deg": grasp_yaw, "finger_touches_neighbour": hits})
        elif hits:
            tried.append({"grasp_yaw_deg": grasp_yaw, "finger_collision": hits})
            continue
        if outline:
            g_rows = [F.base_to_uf0([*det["xy_robot_m"], z]) for z in (g["pick_z"], g["pick_z"] + HOVER_DZ)]
            if not all(F.in_tcp_box([*r, 0, 0, 0, 0]) for r in g_rows):
                tried.append({"grasp_yaw_deg": grasp_yaw, "outside_tcp_box": [round(v, 1) for v in g_rows[0]]})
                continue
        place_spread = spread  # keep the wrist roll: no turn needed for a bin
        failed = reach.check(det["xy_robot_m"], g["pick_z"], spread, place_xy, place_z, place_spread)
        if failed:
            tried.append({"grasp_yaw_deg": grasp_yaw, "ik_failed": failed})
            continue
        return {
            "spread_dir": [float(spread[0]), float(spread[1]), 0.0],
            "place_spread_dir": [float(place_spread[0]), float(place_spread[1]), 0.0],
            "grasp_yaw_deg": grasp_yaw,
            "pick_z": g["pick_z"],
            "jaw_half": g["width"] / 2,
            "width_m": g["width"],
            "clearance_m": g["clearance"],
            "place_xy": place_xy,
            "place_z": place_z,
            "alternatives_rejected": tried,
        }, None
    if outline and tried and all("outside_tcp_box" in t for t in tried):
        return None, ("outside_tcp_box", {"tried": tried})
    if outline and all("finger_collision" in t for t in tried):
        return None, ("fingers_touch_bin", {"tried": tried})
    if all("finger_collision" in t for t in tried):
        return None, ("finger_collision", {"tried": tried})
    return None, ("ik_unreachable", {"tried": tried})


def touches_bin(det, cam, bins, margin=0.002):
    """Bin name if the object's outline (back-projected at its mid height) is inside or within margin of a bin's
    outer square, else None."""
    z = cam.table_z + MID_HEIGHT.get(det["class"], 0.005)
    pts = [cam.to_plane(u, v, z=z) for u, v in det["polygon_px"]]
    pts = np.array([p[:2] for p in pts if p is not None])
    if not len(pts):
        return None
    for name, b in bins.items():
        t = math.radians(-b.get("yaw_deg", 0.0))
        rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        local = (pts - b["xy"]) @ rot.T
        if np.any((np.abs(local[:, 0]) <= b["outer"][0] / 2 + margin) & (np.abs(local[:, 1]) <= b["outer"][1] / 2 + margin)):
            return name
    return None
