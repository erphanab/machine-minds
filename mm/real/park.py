"""
Choose the park pose: fingers down, inside the cell's TCP box, reachable by
the twin's IK, and with the whole arm clear of the table camera's view of the
work area and the bins. Host-side.

    .venv/bin/python -m mm.real.park [--cameras ../PhysicalAI-Hackathon-2026-v6/calibration/cameras.json]

The arm is approximated by points along its links (calibration geometry's
joint chain, the twin's convention) and the gripper body; the work area is
the target box plus every fixed bin, projected into the 1080p table image.
Prints each candidate's clearance in pixels (negative = overlap).
"""

import argparse
import os

import cv2
import numpy as np

import ik
import scene_config as C
from mm.datagen import randomize as R
from mm.planning import grasp as G
from mm.real import frames as F

CHAIN = [((0.0, 0.0, 0.33), "z", +1), ((0.0, 0.0, 0.00), "y", +1), ((0.0, 0.0, 0.26), "y", -1),
         ((0.0, 0.0, 0.02), "x", -1), ((0.29, 0.0, 0.0), "y", -1), ((0.0, 0.0, 0.00), "x", -1)]  # calibration/geometry.py


def _rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def arm_points(q, tool_len=C.TOOL_LENGTH, n=12):
    """Points along the arm (base frame, m): each link sampled, plus the tool."""
    rot, p, joints = np.eye(3), np.zeros(3), [np.zeros(3)]
    for (off, axis, sign), a in zip(CHAIN, q):
        p = p + rot @ np.array(off)
        rot = rot @ _rot(axis, sign * a)
        joints.append(p.copy())
    joints.append(p + rot @ np.array([tool_len, 0.0, 0.0]))
    pts = []
    for a, b in list(zip(joints[:-1], joints[1:]))[2:]:  # from the shoulder on: the base column never moves
        for t in np.linspace(0, 1, n):
            c = a + (b - a) * t
            for dx in (-0.05, 0.05):  # link thickness, roughly
                for dy in (-0.05, 0.05):
                    pts.append(c + [dx, dy, 0.0])
    return np.array(pts)


def work_polygon(cam):
    z = C.TABLE_TOP_Z + C.SURFACE_DZ
    pts = [(x, y) for x in R.DISTRACTOR_X for y in R.DISTRACTOR_Y]
    for b in R.load_bins().values():
        h = b["outer"][0] / 2 + 0.01
        pts += [(b["xy"][0] + sx * h, b["xy"][1] + sy * h) for sx in (-1, 1) for sy in (-1, 1)]
    uv = np.array([cam.project([x, y, z + dz]) for x, y in pts for dz in (0.0, 0.07)], np.float32)
    return cv2.convexHull(uv)


def clearance(q, cam, hull):
    uv = [cam.project(p + [0.0, 0.0, C.TABLE_TOP_Z]) for p in arm_points(q)]
    d = [cv2.pointPolygonTest(hull, (float(u[0]), float(u[1])), True) for u in uv if u is not None]
    return -max(d)  # pointPolygonTest: positive inside


def main():
    ap = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--cameras", default=os.path.join(os.path.dirname(root), "PhysicalAI-Hackathon-2026-v6", "calibration", "cameras.json"))
    a = ap.parse_args()
    cam = F.CellCamera.load(a.cameras, "table", (1920, 1080))
    hull = work_polygon(cam)
    reach = G.Reach()
    best = None
    for x in (0.15, 0.18, 0.22):
        for y in (-0.42, -0.38, 0.30, 0.36, 0.40, 0.44):
            for z in (0.20, 0.25, 0.30, 0.35):
                if not F.in_tcp_box(F.base_to_uf0((x, y, z)) + [0, 0, 0, 0]):
                    continue
                q, ok = ik.solve(reach.home, [x, y, z], [0, 0, -1], reach.limits, tool_offset=reach.tool,
                                 iters=500, damping=0.02)
                if not ok:
                    continue
                c = clearance(q, cam, hull)
                print(f"park {x:.2f} {y:+.2f} {z:.2f}: clearance {c:7.1f} px")
                if best is None or c > best[0]:
                    best = (c, (x, y, z))
    print("best:", best)


if __name__ == "__main__":
    main()
