"""
Synthetic table-camera frames with instance masks (Machine Minds step A3).

Runs under Isaac's interpreter. Builds the cell once with a pool of objects,
then loops over seeds without rebuilding: each seed's layout (randomize.py)
moves the objects it uses into place, parks the rest out of view, and sets
lights, table-camera jitter and the arm pose.

    $ISAAC/python.sh mm/datagen/gen_yolo_data.py --split train --start 0 --count 50

Per frame, under <out>/<split>/:
  images/<name>.jpg   table camera RGB (4K, as capture_scene.py)
  masks/<name>.png    uint16 object index per pixel, 0 = background
  meta/<name>.json    seed, layout, camera model, object classes and settled
                      poses -- labels only, never used at runtime
  aux_cams/<name>_{side,wrist}.jpg for about 20% of frames (grasp check, B6)
  ("aux" itself is a reserved device name on Windows.)
"""

import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="train", choices=["train", "val", "test", "hard"])
parser.add_argument("--start", type=int, default=0, help="offset into the split's seeds")
parser.add_argument("--count", type=int, default=None, help="frames to make (default: all)")
parser.add_argument("--out", default=os.path.join(ROOT, "data", "synth"))
parser.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("W", "H"))
parser.add_argument("--aux-every", type=int, default=5, help="side+wrist every Nth seed")
parser.add_argument("--jpeg-quality", type=int, default=95)
parser.add_argument("--settle-steps", type=int, default=10)
parser.add_argument("--drop-steps", type=int, default=60)
parser.add_argument("--render-steps", type=int, default=4)
parser.add_argument("--rt-subframes", type=int, default=4)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--debug", action="store_true", help="print timeline/pose checks")
parser.add_argument(
    "--no-bins", action="store_true", help="bin-free scene, as the Part A dataset was made"
)
parser.add_argument(
    "--bins", choices=["random", "fixed", "stress"], default="random",
    help="v3: bins at random reachable poses each frame (B8), labelled as class 'bin'; v4 'stress': "
    "a bin against the yellow edge tape and one against the robot base (detection test, not reachable)",
)
args = parser.parse_args()
# Kit changes the working directory; resolve before it starts.
args.out = os.path.abspath(args.out)

from mm.datagen import randomize as R  # noqa: E402  (host-safe, no Isaac imports)

# Pool layout: built once, every object parked. Cap orientation changes the
# collider, so the pool holds both orientations; cube colour is a visual
# variant and is switched per frame instead.
# Anything that recomposes USD under a rigid body (a cube colour variant switch,
# a cap's orientation-dependent collider) makes PhysX re-parse every body from
# USD and undo the frame's teleports. So the pool bakes those attributes in:
# full sets of each cube colour and both cap orientations.
POOL_CAPS = {"narrow_end_up": R.MAX_PER_CLASS["cap"], "wide_end_up": 2}
# Parking: on the floor behind the table camera (x > its eye) and behind the
# side camera (y > its eye). The pool is built here, so nothing starts
# interpenetrating.
PARK_X = (1.8, 2.0, 2.2, 2.4, 2.6, 2.8)
PARK_Y = tuple(0.8 + 0.2 * k for k in range(11))
PARK_SLOTS = [(x, y) for x in PARK_X for y in PARK_Y]


def _pool_layout():
    slots = iter(PARK_SLOTS)

    def kind(n, **extra):
        return [{"xy": list(next(slots)), "yaw_deg": 0.0, **extra} for _ in range(n)]

    caps = [o for orient, n in POOL_CAPS.items() for o in kind(n, orientation=orient)]
    cubes = [o for col in R.CUBE_COLOURS for o in kind(R.MAX_PER_CLASS["cube"], colour=col)]
    bins = {} if args.no_bins else {"bins": R.load_bins()}  # static: built once
    return {
        **bins,
        "cubes": cubes,
        "caps": caps,
        "vials": kind(R.MAX_PER_CLASS["vial"]),
        "aaa_cells": kind(R.MAX_PER_CLASS["aaa_cell"]),
        "screws": kind(R.MAX_PER_CLASS["screw"]),
    }


split_dir = os.path.join(args.out, args.split)
for sub in ("images", "masks", "meta", "aux_cams"):
    os.makedirs(os.path.join(split_dir, sub), exist_ok=True)
pool_path = os.path.join(split_dir, "pool_layout.json")
with open(pool_path, "w") as f:
    json.dump(_pool_layout(), f, indent=2)
os.environ["MM_SCENE_JSON"] = pool_path
# randomize already imported scene_config without the override; reload in place
# so cell.py (and randomize) see the pool.
import importlib  # noqa: E402

importlib.reload(R.C)

from isaacsim import SimulationApp  # noqa: E402

sim_app = SimulationApp({"headless": True})

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
import yaml  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import RigidPrim, SingleArticulation  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402

import scene_config as C  # noqa: E402
import cell  # noqa: E402

with open(os.path.join(ROOT, "configs", "classes.yaml")) as f:
    CLASSES = yaml.safe_load(f)["classes"]
CLASS_BY_PRIM = {c["prim"]: c for c in CLASSES}
LAYOUT_KEY = {"cube": "cubes", "cap": "caps", "aaa_cell": "aaa_cells", "tube": "vials", "screw": "screws"}
ROOT_RE = re.compile(r"^/World/(SortCube|SortCap|SortVial|SortAAACell|Screw)(\d+)(/|$)")
BIN_RE = re.compile(r"^/World/Bin_([A-Za-z0-9_]+?)(/|$)")  # v4 names bin_1..3
BIN_CLASS = next(c for c in CLASSES if c["role"] == "bin")

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
cell.build(stage)
# The starter's mat collider is a 2 mm slab. A narrow-end-up cap's flange
# penetrates it far enough that PhysX resolves the overlap downward, and the cap
# drops onto the table box 2 mm below (seen in the unmodified starter scene too).
# Half of all such caps rendered with the flange hidden under the backdrop.
# A thick invisible support whose top is flush with the work surface pushes
# them back up. Collision only; the calibrated surface height is unchanged.
_SUPPORT_T = 0.02
_support = cell.box(
    stage,
    "/World/MatSupport",
    (C.TABLE_SIZE[0], C.TABLE_SIZE[1], _SUPPORT_T),
    (*C.TABLE_CENTER, C.TABLE_TOP_Z + C.SURFACE_DZ - _SUPPORT_T / 2),
    (1.0, 1.0, 1.0),
)
UsdGeom.Imageable(_support).CreateVisibilityAttr().Set("invisible")
cell.bind_physics_material(_support)
cell.small_contact_offsets(_support)
cell.enable_ccd(world)
world.reset()

arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
arm.initialize()
dof = list(arm.dof_names)

# ---- object pool: prim name -> handle and the attributes baked into it
POOL_SIZE = {
    "cube": len(C.SORT_CUBES),
    "cap": len(C.SORT_CAPS),
    "aaa_cell": len(C.SORT_AAA_CELLS),
    "tube": len(C.SORT_VIALS),
    "screw": len(C.SCREWS),
}
pool = {}
for c in CLASSES:
    for i in range(POOL_SIZE.get(c["name"], 0)):  # bins are static, not in the pool
        name = f"{c['prim']}{i}"
        entry = {"index": len(pool), "class": c}
        if c["name"] == "cap":
            entry["orientation"] = C.CAP_ORIENTATIONS[i]
        if c["name"] == "cube":
            entry["colour"] = C.SORT_CUBES[i][2]
        pool[name] = entry
assert len(PARK_SLOTS) >= len(pool), "not enough parking slots"
# One physics-backed view over the pool. Without a valid physics handle the
# pose setters fall back to writing USD, which PhysX ignores mid-simulation.
bodies = RigidPrim([f"/World/{n}" for n in pool], name="mm_pool")
bodies.initialize(world.physics_sim_view)
if not bodies.is_physics_handle_valid():
    raise RuntimeError("pool RigidPrim has no physics handle; poses would not apply")

Z = C.TABLE_TOP_Z + C.SURFACE_DZ  # world height of the work surface
R_LIE = Rotation.from_euler("y", 90, degrees=True)  # cell axis -> world +X
# cell.add_screws: local -Z (head face -> tip) -> world +X, then the rest tilt
# dips the tip so the head rim and the shank both touch the mat.
R_SCREW = Rotation.from_euler("y", -90 + C.SCREW_REST_TILT_DEG, degrees=True)
DROP_LIFT = {
    "cube": C.CUBE_EDGE,
    "cap": 2 * C.CAP_RADIUS,
    "aaa_cell": C.AAA_CELL_LENGTH / 2,
    "screw": C.SCREW_LENGTH,
}


def _wxyz(rot):
    x, y, z, w = rot.as_quat()
    return np.array([w, x, y, z])


def rest_pose(cls, o):
    """Root prim pose for a layout entry, matching cell.add_sort_objects()."""
    x, y = o["xy"]
    yaw = Rotation.from_euler("z", o["yaw_deg"], degrees=True)
    if cls == "cube":
        return np.array([x, y, Z]), yaw
    if cls == "cap":
        return np.array([x, y, Z]), yaw
    if cls == "aaa_cell":
        return np.array([x, y, Z + C.AAA_CELL_RADIUS + 0.0005]), yaw * R_LIE
    if cls == "tube":  # root sits at the tail, not the centre
        a = np.radians(o["yaw_deg"])
        tail = (x - C.VIAL_LENGTH / 2 * np.cos(a), y - C.VIAL_LENGTH / 2 * np.sin(a))
        return np.array([*tail, Z + C.VIAL_RADIUS + 0.001]), yaw * R_LIE
    if cls == "screw":  # root at the head face; layout xy is the shank middle
        hx, hy = C.screw_head_xy((x, y), o["yaw_deg"])
        return np.array([hx, hy, Z + C.SCREW_HEAD_D / 2 + 0.0005]), yaw * R_SCREW
    raise ValueError(cls)


def drop_pose(cls, o):
    """Released tilted above the mat; physics settles it."""
    x, y = o["xy"]
    roll, pitch, dyaw = o["drop_rpy_deg"]
    tilt = Rotation.from_euler("ZYX", [dyaw, pitch, roll], degrees=True)
    _, rot = rest_pose(cls, o)
    return np.array([x, y, Z + 0.02 + DROP_LIFT[cls]]), tilt * rot


def park_pose(entry, slot):
    name = entry["class"]["name"]
    rot = R_LIE if name in ("aaa_cell", "tube") else R_SCREW if name == "screw" else Rotation.identity()
    return np.array([*slot, 0.05]), rot


def assign(layout):
    """Map layout entries to pool prims. Returns [(prim name, class, entry)].

    Cube colour and cap orientation come from the pool prim; when the pool is
    short of the requested one, the entry records what was actually used.
    """
    free = {n: e for n, e in pool.items()}
    used = []
    for c in CLASSES:
        if c["name"] not in LAYOUT_KEY:  # bins: static, placed by place_bins
            continue
        for o in layout[LAYOUT_KEY[c["name"]]]:
            cands = [n for n, e in free.items() if e["class"] is c]
            for attr in ("colour", "orientation"):
                if attr in o:
                    same = [n for n in cands if free[n].get(attr) == o[attr]]
                    cands = same or cands
            name = sorted(cands, key=lambda n: int(re.sub(r"\D", "", n)))[0]
            entry = free.pop(name)
            truth = {a: entry[a] for a in ("colour", "orientation") if a in entry}
            used.append((name, c, dict(o, **truth)))
    return used, free


lights = {
    p: UsdLux.LightAPI(stage.GetPrimAtPath(p))
    for p in ("/World/DomeLight", "/World/KeyLight", "/World/FillLight", "/World/RimLight")
}
base_intensity = {p: l.GetIntensityAttr().Get() for p, l in lights.items()}


def set_lighting(cfg):
    for p, light in lights.items():
        light.GetIntensityAttr().Set(base_intensity[p] * cfg["scale"])
        light.CreateEnableColorTemperatureAttr().Set(True)
        light.CreateColorTemperatureAttr().Set(float(cfg["kelvin"]))


def set_table_cam(cfg):
    view = Gf.Matrix4d().SetLookAt(*(Gf.Vec3d(*cfg[k]) for k in ("eye", "look_at", "up")))
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(cell.TABLE_CAM_PATH))
    xf.GetOrderedXformOps()[0].Set(view.GetInverse())


def set_arm(joints_deg):
    q = arm.get_joint_positions()
    for j, deg in joints_deg.items():
        q[dof.index(j)] = np.deg2rad(deg)
    for jn, v in cell.jaw_targets(C.JAW_OPEN).items():
        q[dof.index(jn)] = v
    arm.set_joint_positions(q)
    arm.apply_action(ArticulationAction(joint_positions=q))


def camera_model(path, resolution):
    """Intrinsics/extrinsics read back from the stage, as capture_scene.py does."""
    prim = stage.GetPrimAtPath(path)
    gcam = UsdGeom.Camera(prim)
    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    return {
        "resolution": list(resolution),
        "focal_length_mm": float(gcam.GetFocalLengthAttr().Get()),
        "horizontal_aperture_mm": float(gcam.GetHorizontalApertureAttr().Get()),
        "vertical_aperture_mm": float(gcam.GetVerticalApertureAttr().Get()),
        "camera_to_world": [[float(xf[r][c]) for c in range(4)] for r in range(4)],
        "table_top_z": float(Z),
        "robot_base_xy": [0.0, 0.0],
    }


# ---- render products: table camera at the planning capture's 4K
table_res = tuple(args.resolution) if args.resolution else tuple(
    2 * v for v in C.TABLE_CAM["resolution"]
)
if hasattr(rep.orchestrator, "set_capture_on_play"):
    rep.orchestrator.set_capture_on_play(False)
rp_table = rep.create.render_product(cell.TABLE_CAM_PATH, table_res)
ann_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
ann_rgb.attach(rp_table)
ann_inst = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation_fast")
ann_inst.attach(rp_table)
aux = {}
for tag, path, cfg in (
    ("side", cell.SIDE_CAM_PATH, C.SIDE_CAM),
    ("wrist", cell.WRIST_CAM_PATH, C.WRIST_CAM),
):
    if stage.GetPrimAtPath(path).IsValid():
        rp = rep.create.render_product(path, tuple(cfg["resolution"]))
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach(rp)
        rp.hydra_texture.set_updates_enabled(False)
        aux[tag] = (rp, ann, path, tuple(cfg["resolution"]))

# MDL materials compile over the first rendered frames.
for _ in range(60):
    world.step(render=True)


def place_bins(bins):
    """Move each bin by its root Xform (translate + rotateZ, cell._make_bin).

    Attribute changes on static colliders, not structural edits, so PhysX
    moves them without re-parsing the rigid bodies (whose teleports follow).
    """
    for name, b in bins.items():
        prim = stage.GetPrimAtPath(f"/World/Bin_{name}")
        ops = {op.GetOpName(): op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}
        ops["xformOp:translate"].Set(Gf.Vec3d(b["xy"][0], b["xy"][1], C.TABLE_TOP_Z + C.SURFACE_DZ))
        ops["xformOp:rotateZ"].Set(float(b.get("yaw_deg", 0.0)))


def capture(seed, layout):
    if layout.get("bins"):
        place_bins(layout["bins"])
        world.step(render=False)  # let PhysX take the moved static colliders
    used, free = assign(layout)
    positions = np.zeros((len(pool), 3))
    quats = np.zeros((len(pool), 4))
    for name, entry in free.items():  # parked objects keep their own slot
        pos, rot = park_pose(entry, PARK_SLOTS[entry["index"]])
        positions[entry["index"]], quats[entry["index"]] = pos, _wxyz(rot)
    dropped = False
    for name, c, o in used:
        if "drop_rpy_deg" in o:
            pos, rot = drop_pose(c["name"], o)
            dropped = True
        else:
            pos, rot = rest_pose(c["name"], o)
        k = pool[name]["index"]
        positions[k], quats[k] = pos, _wxyz(rot)
    bodies.set_world_poses(positions, quats)
    bodies.set_velocities(np.zeros((len(pool), 6)))
    if args.debug:
        k0 = pool[used[0][0]]["index"]
        print(f"[debug] playing={world.is_playing()} want={positions[k0].round(4)} "
              f"got={np.asarray(bodies.get_world_poses()[0])[k0].round(4)}", flush=True)
    set_lighting(layout["lighting"])
    if "bin_roughness" in layout:  # v4: printed tangerine bins, mild gloss variation
        cell.set_roughness(stage, "bin_tangerine", layout["bin_roughness"])
    set_table_cam(layout["table_cam"])
    set_arm(layout["arm"]["joints_deg"])

    for _ in range(args.drop_steps if dropped else args.settle_steps):
        world.step(render=False)
    for _ in range(args.render_steps):
        world.step(render=True)
    if args.debug:
        print(f"[debug] after steps playing={world.is_playing()} "
              f"got={np.asarray(bodies.get_world_poses()[0])[k0].round(4)}", flush=True)

    want_aux = args.aux_every > 0 and seed % args.aux_every == 0
    for rp, *_ in aux.values():
        rp.hydra_texture.set_updates_enabled(want_aux)
    rep.orchestrator.step(rt_subframes=args.rt_subframes, delta_time=0.0, pause_timeline=False)
    if args.debug:
        print(f"[debug] after capture playing={world.is_playing()}", flush=True)

    rgb = np.asarray(ann_rgb.get_data())[:, :, :3]
    inst = ann_inst.get_data()
    ids = np.asarray(inst["data"]).astype(np.uint32)
    id_to_path = inst["info"]["idToLabels"]

    # Merge every mesh under one object root into that object's index; bins
    # follow the objects.
    index_of = {name: k + 1 for k, (name, _, _) in enumerate(used)}
    bins = layout.get("bins") or {}
    bin_index = {name: len(used) + 1 + k for k, name in enumerate(bins)}
    mask = np.zeros(ids.shape, np.uint16)
    for sid, path in id_to_path.items():
        m = ROOT_RE.match(str(path))
        if m and f"{m.group(1)}{m.group(2)}" in index_of:
            mask[ids == int(sid)] = index_of[f"{m.group(1)}{m.group(2)}"]
            continue
        b = BIN_RE.match(str(path))
        if b and b.group(1) in bin_index:
            mask[ids == int(sid)] = bin_index[b.group(1)]

    stem = f"{args.split}_{seed:06d}"
    cv2.imwrite(
        os.path.join(split_dir, "images", stem + ".jpg"),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
    )
    cv2.imwrite(os.path.join(split_dir, "masks", stem + ".png"), mask)

    aux_files = {}
    if want_aux:
        for tag, (rp, ann, path, res) in aux.items():
            img = np.asarray(ann.get_data())[:, :, :3]
            fname = f"{stem}_{tag}.jpg"
            cv2.imwrite(
                os.path.join(split_dir, "aux_cams", fname),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )
            aux_files[tag] = {"file": f"aux_cams/{fname}", "camera": camera_model(path, res)}

    settled_pos, settled_quat = (np.asarray(a) for a in bodies.get_world_poses())
    objects = []
    for name, c, o in used:
        k = pool[name]["index"]
        pos, quat = settled_pos[k], settled_quat[k]
        objects.append(
            {
                "index": index_of[name],
                "prim": f"/World/{name}",
                "class": c["name"],
                "class_id": c["id"],
                "role": c["role"],
                "colour": o.get("colour"),
                "orientation": o.get("orientation"),
                "dropped": "drop_rpy_deg" in o,
                "layout_xy": o["xy"],
                "layout_yaw_deg": o["yaw_deg"],
                "position": [float(v) for v in pos],
                "quat_wxyz": [float(v) for v in quat],
                "visible_px": int((mask == index_of[name]).sum()),
            }
        )
    for name, b in bins.items():  # static: the layout pose is the truth
        w, x, y, z = _wxyz(Rotation.from_euler("z", b.get("yaw_deg", 0.0), degrees=True))
        objects.append(
            {
                "index": bin_index[name],
                "prim": f"/World/Bin_{name}",
                "class": "bin",
                "class_id": BIN_CLASS["id"],
                "role": "bin",
                "bin_for": name,
                "colour": b.get("colour"),
                "orientation": None,
                "dropped": False,
                "layout_xy": list(b["xy"]),
                "layout_yaw_deg": b.get("yaw_deg", 0.0),
                "position": [float(b["xy"][0]), float(b["xy"][1]), float(Z)],
                "quat_wxyz": [float(w), float(x), float(y), float(z)],
                "visible_px": int((mask == bin_index[name]).sum()),
            }
        )
    meta = {
        "seed": seed,
        "split": args.split,
        "image": f"images/{stem}.jpg",
        "mask": f"masks/{stem}.png",
        "camera": camera_model(cell.TABLE_CAM_PATH, table_res),
        "objects": objects,
        "aux": aux_files,
        "layout": layout,
    }
    with open(os.path.join(split_dir, "meta", stem + ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    return stem, len(objects)


seeds = list(R.split_seeds(args.split))[args.start:]
if args.count is not None:
    seeds = seeds[: args.count]
t0 = time.time()
done = 0
for seed in seeds:
    stem = f"{args.split}_{seed:06d}"
    if not args.overwrite and os.path.exists(os.path.join(split_dir, "meta", stem + ".json")):
        continue
    layout = R.sample_layout(seed, hard=args.split == "hard", bins=not args.no_bins, bin_mode=args.bins)
    t = time.time()
    _, n = capture(seed, layout)
    done += 1
    print(f"[datagen] {stem}: {n} objects, {time.time() - t:.1f}s", flush=True)

print(f"[datagen] {done} frames in {time.time() - t0:.0f}s -> {split_dir}", flush=True)
sim_app.close()
