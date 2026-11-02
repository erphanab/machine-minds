"""
Sort-bin geometry: one source for the Isaac twin and the 3D-printed bins.

The shape is the starter's procedural bin (cell._make_bin, its "vial" size):
a square outer wall, a rounded-square pocket and a solid floor. cell.py builds
every bin's USD mesh from bin_mesh(); write_stl() exports the same triangles
for printing, so the printed bins match the twin exactly.

    .venv/bin/python -m mm.cad.sort_bin --out cad/sort_bin_90x90x65.stl

Units: metres in code; the STL is written in millimetres (slicer default).
"""

import argparse
import math
import os
import struct

import numpy as np


def rounded_square(half, r, per_corner=9):
    """Counter-clockwise outline of a square with rounded corners.

    per_corner is odd so each arc includes its 45 deg midpoint, which lets
    ray_to_square land exactly on the outer square's corners.
    """
    pts = []
    for k, (cx, cy) in enumerate(((half - r, half - r), (-half + r, half - r), (-half + r, -half + r), (half - r, -half + r))):
        for j in range(per_corner):
            a = math.radians(90 * k + 90.0 * j / (per_corner - 1))
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def ray_to_square(p, half):
    """Push a point radially out onto an axis-aligned square of half-size half."""
    m = max(abs(p[0]), abs(p[1]))
    return (p[0] * half / m, p[1] * half / m)


def bin_mesh(outer, pocket, pocket_r, wall_h, floor_t):
    """Closed, outward-wound mesh in the bin's frame (origin: base centre).

    Returns (points (N, 3), faces) with faces as vertex-index lists (quads and
    triangles), as USD wants them.
    """
    inner = rounded_square(pocket / 2, pocket_r)
    outer_pts = [ray_to_square(p, outer[0] / 2) for p in inner]
    if not math.isclose(outer[0], outer[1]):
        raise ValueError("the bin design is square")
    n = len(inner)
    pts = []

    def ring(poly, z):
        base = len(pts)
        pts.extend((x, y, z) for x, y in poly)
        return base

    o_top, i_top = ring(outer_pts, wall_h), ring(inner, wall_h)
    o_bot, i_bot = ring(outer_pts, 0.0), ring(inner, floor_t)
    c_bot = len(pts)
    pts.append((0.0, 0.0, 0.0))
    c_fl = len(pts)
    pts.append((0.0, 0.0, floor_t))
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([o_top + i, o_top + j, i_top + j, i_top + i])  # rim, up
        faces.append([o_bot + i, o_bot + j, o_top + j, o_top + i])  # outer wall, out
        faces.append([i_top + i, i_top + j, i_bot + j, i_bot + i])  # pocket wall, inward
        faces.append([c_bot, o_bot + j, o_bot + i])  # base, down
        faces.append([c_fl, i_bot + i, i_bot + j])  # pocket floor, up
    return np.array(pts, float), faces


def triangles(points, faces):
    """Fan-split faces into triangles, keeping their winding."""
    tris = []
    for f in faces:
        for k in range(1, len(f) - 1):
            tris.append((f[0], f[k], f[k + 1]))
    return np.array(tris, int)


def write_stl(path, points, faces, scale=1000.0, name=b"machine_minds_sort_bin"):
    """Binary STL (millimetres by default)."""
    tris = triangles(points, faces)
    p = points * scale
    with open(path, "wb") as f:
        f.write(name.ljust(80, b" ")[:80])
        f.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            n = np.cross(p[b] - p[a], p[c] - p[a])
            n = n / (np.linalg.norm(n) or 1.0)
            f.write(struct.pack("<12fH", *n, *p[a], *p[b], *p[c], 0))
    return len(tris)


def volume(points, faces):
    """Signed volume (positive when the mesh is closed and outward-wound)."""
    tris = triangles(points, faces)
    v = points[tris]
    return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)


def main():
    from mm.datagen.randomize import load_bin_design

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    d = load_bin_design()
    pts, faces = bin_mesh(d["outer"], d["pocket"], d["pocket_r"], d["wall_h"], d["floor_t"])
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mm = [round(v * 1000) for v in (d["outer"][0], d["outer"][1], d["wall_h"])]
    out = args.out or os.path.join(root, "cad", f"sort_bin_{mm[0]}x{mm[1]}x{mm[2]}.stl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n = write_stl(out, pts, faces)
    print(f"[sort_bin] {n} triangles, volume {volume(pts, faces) * 1e6:.1f} cm^3 -> {out}")


if __name__ == "__main__":
    main()
