"""
The cell's frames and calibration for the real-arm policy (organizers' client).

  robot base   x forward, y left, z up, metres; the twin's planning frame
  UF0          the controller frame of obs["tcp_pose"] and "tcp" actions:
               x, y in mm as the base, z_UF0 = z_base - 330 mm
  w, p, r      fingers down is (-180, 0, R); R turns the fingers about z

Jaw direction <-> R. The controller's tool frame (UTOOL1) has z along the
approach, which is the twin's J6_link +x (C.TOOL_AXIS). At zero joints a
FANUC faceplate frame is x down, y left, z forward, so tool y = J6_link +y,
the twin's jaw-closing axis (C.FINGER_SPREAD_AXIS), and tool x = -J6_link z.
With w, p, r applied as Rz(r) Ry(p) Rx(w), fingers down gives
tool y = (sin R, -cos R, 0): the jaws close along that line.
This is derived from conventions, not measured on the cell: check it on the
first real rollout (the logged R against the jaw direction in the wrist view).

CellCamera serves backproject.CameraModel's interface (W, H, table_z,
project, to_plane) from the cell's calibration/cameras.json, in the twin's
world frame (base + TABLE_TOP_Z in z), so perception runs unchanged.
"""

import json
import math

import numpy as np

import scene_config as C

UF0_Z_MM = -330.0
TCP_BOX = [(150.0, 650.0), (-450.0, 450.0), (-334.0, 300.0)]  # UF0 mm, the cell's limits (mock_cell.py)
# Tool frame axes in J6_link coordinates (columns: tool x, y, z).
TOOL_IN_J6 = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


def base_to_uf0(p):
    return [float(p[0]) * 1000.0, float(p[1]) * 1000.0, float(p[2]) * 1000.0 + UF0_Z_MM]


def uf0_to_base(xyz_mm):
    return np.array([xyz_mm[0] / 1000.0, xyz_mm[1] / 1000.0, (xyz_mm[2] - UF0_Z_MM) / 1000.0])


def spread_to_r(spread, r_prev=None):
    """Jaw-closing direction (x, y) -> R (deg). Jaws are symmetric, so R and
    R +- 180 are the same grasp: the one nearest r_prev is used."""
    r = math.degrees(math.atan2(spread[0], -spread[1]))
    if r_prev is None:
        return r
    cands = [r + 180.0 * k for k in range(-3, 4) if abs(r + 180.0 * k) <= 350.0]
    return min(cands, key=lambda c: abs(c - r_prev))


def r_to_spread(r_deg):
    a = math.radians(r_deg)
    return np.array([math.sin(a), -math.cos(a), 0.0])


def wpr_from_rotation(rot_j6):
    """J6_link rotation (base frame) -> controller w, p, r (deg)."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(np.asarray(rot_j6) @ TOOL_IN_J6).as_euler("xyz", degrees=True).tolist()


def in_tcp_box(row):
    return all(lo - 1e-6 <= row[k] <= hi + 1e-6 for k, (lo, hi) in enumerate(TCP_BOX))


class CellCamera:
    """A fixed cell camera (cameras.json entry) as a backproject.CameraModel."""

    def __init__(self, cal, size=None):
        self.T = np.array(cal["T_base_cam"], float)
        self.Tinv = np.linalg.inv(self.T)
        w0, h0 = cal["size"]
        self.W, self.H = (int(size[0]), int(size[1])) if size else (int(w0), int(h0))
        self.fx = float(cal["fx"]) * self.W / w0  # same aspect: the image is resized, not cropped
        self.k1 = float(cal.get("k1", 0.0))
        self.dz = C.TABLE_TOP_Z  # world z = base z + TABLE_TOP_Z
        self.table_z = C.TABLE_TOP_Z + C.SURFACE_DZ

    @classmethod
    def load(cls, path, name="table", size=None):
        with open(path) as f:
            return cls(json.load(f)[name], size)

    def project(self, p_world):
        p = np.asarray(p_world, float) - [0.0, 0.0, self.dz]
        pc = self.Tinv[:3, :3] @ p + self.Tinv[:3, 3]
        if pc[2] <= 1e-9:
            return None
        x, y = pc[0] / pc[2], pc[1] / pc[2]
        d = 1 + self.k1 * (x * x + y * y)
        return np.array([self.fx * x * d + self.W / 2, self.fx * y * d + self.H / 2])

    def ray(self, u, v):
        xd, yd = (u - self.W / 2) / self.fx, (v - self.H / 2) / self.fx
        x, y = xd, yd
        for _ in range(10):  # invert the radial term (geometry.pixel_to_base)
            d = 1 + self.k1 * (x * x + y * y)
            x, y = xd / d, yd / d
        r = self.T[:3, :3] @ np.array([x, y, 1.0])
        return self.T[:3, 3] + [0.0, 0.0, self.dz], r / np.linalg.norm(r)

    def to_plane(self, u, v, z=None):
        z = self.table_z if z is None else z
        o, d = self.ray(u, v)
        if abs(d[2]) < 1e-9:
            return None
        t = (z - o[2]) / d[2]
        return None if t <= 0 else o + t * d

    def to_table_xy(self, u_norm, v_norm, height_above_table=0.0):
        p = self.to_plane(u_norm * self.W, v_norm * self.H, z=self.table_z + height_above_table)
        return None if p is None else np.array([p[0], p[1]])


class WristCamera:
    """The wrist camera (cameras.json "wrist": T_flange_cam on the twin's J6_link) at the served image size; its pose
    follows the joints (fk.fk). Points are in the robot base frame (not the twin's world)."""

    def __init__(self, cal, size=None):
        self.T_fc = np.array(cal["T_flange_cam"], float)
        w0, h0 = cal["size"]
        self.W, self.H = (int(size[0]), int(size[1])) if size else (int(w0), int(h0))
        self.fx = float(cal["fx"]) * self.W / w0
        self.k1 = float(cal.get("k1", 0.0))

    @classmethod
    def load(cls, path, size=None):
        with open(path) as f:
            cal = json.load(f).get("wrist")
        return None if cal is None or "T_flange_cam" not in cal else cls(cal, size)

    def project(self, q_rad, p_base):
        import fk

        rot, pos = fk.fk(q_rad)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = rot, pos
        Tinv = np.linalg.inv(T @ self.T_fc)
        pc = Tinv[:3, :3] @ np.asarray(p_base, float) + Tinv[:3, 3]
        if pc[2] <= 1e-9:
            return None
        x, y = pc[0] / pc[2], pc[1] / pc[2]
        d = 1 + self.k1 * (x * x + y * y)
        return np.array([self.fx * x * d + self.W / 2, self.fx * y * d + self.H / 2])


def sim_wrist_calibration():
    """The twin's wrist camera (C.WRIST_CAM: eye / look_at / up on J6_link) as a cameras.json "wrist" entry. The twin
    was built from the cell's wrist calibration, so this matches it up to k1."""
    eye, at, up = (np.asarray(C.WRIST_CAM[k], float) for k in ("eye", "look_at", "up"))
    z = (at - eye) / np.linalg.norm(at - eye)  # forward
    x = np.cross(z, up)
    x /= np.linalg.norm(x)  # right
    y = np.cross(z, x)  # down
    t = np.eye(4)
    t[:3, 0], t[:3, 1], t[:3, 2], t[:3, 3] = x, y, z, eye
    w, h = C.WRIST_CAM["resolution"]
    return {"T_flange_cam": t.tolist(), "fx": w * C.WRIST_CAM["focal_length"] / C.WRIST_CAM["horiz_aperture"], "k1": 0.0,
            "size": [int(w), int(h)], "solved": "twin", "transform": "T_flange_cam: camera -> flange (J6_link)"}


def sim_calibration(cam_json):
    """A cameras.json "table" entry for a twin camera (capture camera.json):
    lets the policy run on sim frames through the same code as on the cell."""
    m = np.array(cam_json["camera_to_world"], float)  # USD: rows = camera axes, row 3 = eye
    x, y, z, eye = m[0, :3], m[1, :3], m[2, :3], m[3, :3]
    t = np.eye(4)
    t[:3, 0], t[:3, 1], t[:3, 2] = x, -y, -z  # USD looks down -Z with +Y up; cameras.json: +z forward, +y down
    t[:3, 3] = eye - [0.0, 0.0, C.TABLE_TOP_Z]
    w, h = cam_json["resolution"]
    return {"T_base_cam": t.tolist(), "fx": w * cam_json["focal_length_mm"] / cam_json["horizontal_aperture_mm"],
            "k1": 0.0, "size": [int(w), int(h)], "solved": "twin", "transform": "T_base_cam: camera -> robot base"}
