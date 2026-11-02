"""
Isaac Sim backend: execute bin-sort plans (Machine Minds steps B5-B8).

Called by sort_scene.py for plan_bins plans (the starter's cube-stack path is
unchanged). Motion is the starter's: per-frame IK along straight Cartesian
segments with the wrist roll pinned to the jaw direction (turned during
approach and carry), two 1/60 s physics steps per 30 fps frame.

  open loop (B5)    the plan from the initial capture, picks back to back in
                    plan order (executor.chained_segments)
  closed loop       run/loop.json present: closed_loop.py re-captures and
  (B6-B8)           re-plans every cycle, checks each grasp, retries, tracks
                    added or moved objects, and finds the bins each capture

Ground truth (simulator object poses) is used only to score: pickup = the
tracked object rose 25 mm at lift, placement = it rests inside its bin. It
never moves the arm.

Writes run/execution_events.json as it goes and run/execution_report.json.
With --render: run/frames (demo camera, VIDEO_CAM) and run/wrist_frames, each
rendered fresh per frame; the run fails if more than 5% of consecutive frames
are identical (report render_check). sort_scene's --camera is not used here.
"""

import importlib
import json
import os
import time
from pathlib import Path

import numpy as np

import scene_config as C
import fk
import ik
from mm.control import executor as X

FPS = 30
# Demo video (--render only; not part of the twin's calibrated cameras): a
# three-quarter view from the front right that keeps the whole arm, the work
# area and random bins in frame, with the upper right free for the wrist
# picture-in-picture. Both views are rendered fresh for every saved frame.
VIDEO_CAM = dict(
    eye=(1.0, -0.95, C.TABLE_TOP_Z + 0.60), look_at=(0.2, 0.08, C.TABLE_TOP_Z + 0.20), up=(0.0, 0.0, 1.0),
    resolution=(1920, 1080), focal_length=12.0, horiz_aperture=20.955,
)
VIDEO_CAM_PATH = "/World/MMVideoCam"
PIP_RES = (640, 360)
RT_SUBFRAMES = 2
MAX_DUPLICATE_FRAC = 0.05  # identical consecutive frames above this: the render failed
# Task objects sleep once at rest and wake when touched. PhysX has no rolling resistance: with the arm still,
# a lying tube or AAA cell on the level mat crept off and sped up to ~30 mm/s (layout 3: the tube ~104 mm into
# an AAA cell), which the real cell never showed. Mass-normalised kinetic energy, m^2/s^2 (default 5e-5).
REST_SLEEP_THRESHOLD = 1e-4


def _load_layout(run):
    """Rebuild the captured scene: the layout that made it (bins, screws, ...)."""
    layout = run / "layout.json"
    if layout.exists():
        os.environ["MM_SCENE_JSON"] = str(layout)
    elif not os.environ.get("MM_SCENE_JSON"):
        raise SystemExit(f"{layout} missing and MM_SCENE_JSON unset: cannot rebuild the scene")
    importlib.reload(C)


def main(a, run, out, plan, state):
    run, out = Path(run), Path(out)
    _load_layout(run)
    loop_cfg = json.loads((run / "loop.json").read_text()) if (run / "loop.json").exists() else None
    clean_env = dict(os.environ)  # before Kit edits PATH/PYTHONPATH: for the YOLO server
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        cell = SimCell(a, run, out, state, closed_loop=loop_cfg is not None)
        if loop_cfg is None:
            run_open_loop(cell, plan)
        else:
            from mm.control.closed_loop import run_closed_loop

            run_closed_loop(cell, plan, loop_cfg, clean_env)
        if cell.render and not cell.render_check()["ok"]:
            raise RuntimeError(f"render check failed: {cell.render_check()}")
    except Exception:
        import traceback

        traceback.print_exc()  # Kit shutdown may hide it otherwise
        raise
    finally:
        app.close()


class SimCell:
    """The twin, set up once: arm motion, object poses, cameras, teleports."""

    def __init__(self, a, run, out, state, closed_loop=False):
        import omni.replicator.core as rep
        import omni.usd
        from pxr import UsdGeom, UsdPhysics
        from isaacsim.core.api import World
        from isaacsim.core.prims import RigidPrim, SingleArticulation, SingleXFormPrim
        from isaacsim.core.utils.types import ArticulationAction
        import cell

        self.a, self.run, self.out, self.cellmod, self.rep = a, run, out, cell, rep
        self.render = not a.physics_only
        self.Action = ArticulationAction
        self.world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 30)
        self.stage = omni.usd.get_context().get_stage()
        cell.build(self.stage)
        # Same thick support under the 2 mm mat as datagen (gen_yolo_data.py):
        # without it narrow-end-up caps sink through the mat slab.
        support = cell.box(
            self.stage, "/World/MatSupport", (C.TABLE_SIZE[0], C.TABLE_SIZE[1], 0.02),
            (*C.TABLE_CENTER, C.TABLE_TOP_Z + C.SURFACE_DZ - 0.01), (1.0, 1.0, 1.0),
        )
        UsdGeom.Imageable(support).CreateVisibilityAttr().Set("invisible")
        cell.bind_physics_material(support)
        cell.small_contact_offsets(support)

        self.names = []
        for name, saved in state["objects"].items():
            prim = self.stage.GetPrimAtPath("/World/" + name)
            if not prim.IsValid():
                raise ValueError(f"captured object missing: {name}")
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(True)
            SingleXFormPrim("/World/" + name).set_world_pose(np.array(saved["position"]), np.array(saved["orientation"]))
            self.names.append(name)

        # Nothing renders unasked: every image comes from an explicit
        # orchestrator step (video frames in move(), captures in grab()).
        if (self.render or closed_loop) and hasattr(rep.orchestrator, "set_capture_on_play"):
            rep.orchestrator.set_capture_on_play(False)
        # Video (--render): the demo camera and the wrist picture-in-picture.
        self.video = {}
        if self.render:
            cell.camera(self.stage, VIDEO_CAM_PATH, VIDEO_CAM)
            for tag, path, res in (("video", VIDEO_CAM_PATH, VIDEO_CAM["resolution"]), ("pip", cell.WRIST_CAM_PATH, PIP_RES)):
                rp = rep.create.render_product(path, tuple(res))
                ann = rep.AnnotatorRegistry.get_annotator("rgb")
                ann.attach(rp)
                self.video[tag] = (rp, ann)
        # Captures for the closed loop: 4K table camera (as capture_scene.py) and
        # the wrist camera (B6). Rendered only when asked for.
        self.products = {}
        if closed_loop:
            for tag, path, res in (
                ("table4k", cell.TABLE_CAM_PATH, tuple(2 * v for v in C.TABLE_CAM["resolution"])),
                ("wrist", cell.WRIST_CAM_PATH, tuple(C.WRIST_CAM["resolution"])),
            ):
                rp = rep.create.render_product(path, res)
                ann = rep.AnnotatorRegistry.get_annotator("rgb")
                ann.attach(rp)
                rp.hydra_texture.set_updates_enabled(False)
                self.products[tag] = (rp, ann, path, res)
        cell.enable_ccd(self.world)
        self.world.reset()

        self.arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
        self.arm.initialize()
        names = list(self.arm.dof_names)
        self.arm_idx = [names.index(f"J{i}") for i in range(1, 7)]
        self.idx = {n: i for i, n in enumerate(names)}
        self.limits = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])
        ik.set_tool_axis(C.TOOL_AXIS)
        ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
        self.tool = tuple(np.array(C.TOOL_AXIS) * cell.tip_offset())
        self.down = [0.0, 0.0, -1.0]
        self.home = np.array(X.HOME_XYZ)
        q = np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)])
        self.q, ok = ik.solve(q, self.home, self.down, self.limits, tool_offset=self.tool)
        if not ok:
            raise RuntimeError("home IK failed")
        full = self.arm.get_joint_positions().copy()
        full[self.arm_idx] = self.q
        for joint, value in cell.jaw_targets(C.JAW_OPEN).items():
            full[self.idx[joint]] = value
        self.arm.set_joint_positions(full)
        for _ in range(120):
            self.command(self.q, C.JAW_OPEN)
            self.world.step(render=self.render)
        self.bodies = RigidPrim([f"/World/{n}" for n in self.names], name="mm_task_objects")
        self.bodies.initialize(self.world.physics_sim_view)
        # Through the physics view: PhysX reads 0 (never sleep) for these bodies, and a USD sleepThreshold set at
        # build time does not reach it.
        self.bodies.set_sleep_thresholds(np.full(len(self.names), REST_SLEEP_THRESHOLD))
        for name in self.names:  # release the captured snapshot to physics
            UsdPhysics.RigidBodyAPI(self.stage.GetPrimAtPath("/World/" + name)).CreateKinematicEnabledAttr().Set(False)
        for _ in range(30):
            self.command(self.q, C.JAW_OPEN)
            self.world.step(render=self.render)

        if self.render:
            out.mkdir(parents=True, exist_ok=True)
            (run / "wrist_frames").mkdir(exist_ok=True)
        self.frame = 0
        self.render_stats = {"frames": 0, "duplicates": {"video": 0, "pip": 0}}
        self._last_img = {}
        self.log = X.PathLog(1.0 / FPS)
        self.tip_now = fk.tool_pose(self.arm.get_joint_positions()[self.arm_idx], self.tool)[1]
        self.t_wall0 = time.time()

    # ---------------------------------------------------------------- basics
    def command(self, q6, jaw):
        full = self.arm.get_joint_positions().copy()
        full[self.arm_idx] = q6
        for joint, value in self.cellmod.jaw_targets(jaw).items():
            full[self.idx[joint]] = value
        self.arm.apply_action(self.Action(joint_positions=full))

    def jaw_gap_mm(self):
        jp = self.arm.get_joint_positions()
        return round(float((2 * C.JAW_OPEN - abs(jp[self.idx["jaw_l"]]) - abs(jp[self.idx["jaw_r"]])) * 1000), 3)

    def poses(self):
        pos, quat = (np.asarray(v) for v in self.bodies.get_world_poses())
        values = {}
        for k, name in enumerate(self.names):
            w, x, y, z = quat[k]
            rot = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )
            yaw, tilt = X.yaw_tilt(rot)
            values[name] = {
                "position": pos[k].tolist(),
                "orientation": quat[k].tolist(),
                "centre": X.object_centre(name, pos[k], rot).tolist(),
                "yaw_deg": yaw,
                "tilt_deg": tilt,
            }
        return values

    def teleport(self, poses):
        """{name: (position, quat_wxyz)} through the physics view; others kept."""
        pos, quat = (np.asarray(v).copy() for v in self.bodies.get_world_poses())
        for name, (p, q) in poses.items():
            k = self.names.index(name)
            pos[k], quat[k] = p, q
        self.bodies.set_world_poses(pos, quat)
        self.bodies.set_velocities(np.zeros((len(self.names), 6)))

    def settle(self, n=20):
        for _ in range(n):
            self.command(self.q, self._jaw)
            self.world.step(render=False)

    _jaw = C.JAW_OPEN

    # ---------------------------------------------------------------- motion
    def solve_segment(self, q, name, p0, p1, duration, direction):
        """executor.solve_segment with this arm's limits and tool, at FPS."""
        return X.solve_segment(q, name, p0, p1, duration, direction, self.limits, self.tool, hz=FPS)

    def check_path(self, legs):
        """IK along a chain of (name, p0, p1, duration, direction) legs from the
        current joint state. None if every frame solves, else (leg name, frame)."""
        return X.check_path(self.q, legs, self.limits, self.tool)

    def move(self, name, p0, p1, duration, j0, j1, direction):
        """One straight segment; returns its PathLog entry. Raises on IK failure,
        before any motion (closed_loop checks the whole path first)."""
        frames, failed = self.solve_segment(self.q, name, p0, p1, duration, direction)
        if failed is not None:
            raise RuntimeError(f"IK failed: {name}, frame {failed}")
        self.log.start(name)
        self.log.step(self.tip_now, self.tip_now)
        for q, target, t in frames:
            self.q = q
            self._jaw = j0 + (j1 - j0) * t
            self.command(q, self._jaw)
            for _ in range(2):
                self.world.step(render=False)
            self.tip_now = fk.tool_pose(self.arm.get_joint_positions()[self.arm_idx], self.tool)[1]
            self.log.step(target, self.tip_now)
            if self.video:
                self._save_frame()
            self.frame += 1
        return self.log.end()

    def hold(self, name, seconds, jaw0, jaw1):
        """Stay put (jaw may move): a segment from here to here."""
        here = self.tip_now.copy()
        r, _ = fk.fk(self.q)
        s = r @ np.array(C.FINGER_SPREAD_AXIS)
        s[2] = 0
        return self.move(name, here, here, seconds, jaw0, jaw1, s / np.linalg.norm(s))

    # ---------------------------------------------------------------- cameras
    @staticmethod
    def _cv2():
        import cv2

        return cv2

    def _save_frame(self):
        """Render the video and wrist views for this frame and write both.

        Counts frames identical to the previous one: a stale render product
        (capture-on-play off, no step) gives copies and a jumping arm.
        """
        cv2 = self._cv2()
        self.rep.orchestrator.step(rt_subframes=RT_SUBFRAMES, delta_time=0.0, pause_timeline=False)
        for tag, path in (("video", self.out / f"f{self.frame:05d}.png"),
                          ("pip", self.run / "wrist_frames" / f"f{self.frame:05d}.jpg")):
            img = np.asarray(self.video[tag][1].get_data())
            if img.ndim != 3 or img.size == 0:
                raise RuntimeError(f"{tag} camera returned no image")
            img = img[:, :, :3]
            last = self._last_img.get(tag)
            if last is not None and np.array_equal(img, last):
                self.render_stats["duplicates"][tag] += 1
            self._last_img[tag] = img.copy()
            cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.render_stats["frames"] += 1

    def render_check(self):
        """Share of identical consecutive frames per view; ok when all <= 5%."""
        n = max(1, self.render_stats["frames"] - 1)
        frac = {k: round(v / n, 4) for k, v in self.render_stats["duplicates"].items()}
        return {
            "frames": self.render_stats["frames"], "duplicates": dict(self.render_stats["duplicates"]),
            "duplicate_frac": frac, "max_duplicate_frac": MAX_DUPLICATE_FRAC,
            "ok": all(f <= MAX_DUPLICATE_FRAC for f in frac.values()),
        }

    def grab(self, tag, warmup=4):
        """BGR image from a capture product (table4k, wrist), rendered now."""
        cv2 = self._cv2()
        rp, ann, _, _ = self.products[tag]
        rp.hydra_texture.set_updates_enabled(True)
        for _ in range(warmup):
            self.command(self.q, self._jaw)
            self.world.step(render=True)
        self.rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=False)
        img = np.asarray(ann.get_data())[:, :, :3]
        rp.hydra_texture.set_updates_enabled(False)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def table_capture(self, k):
        """4K table image + camera.json, as capture_scene.py writes them."""
        from pxr import UsdGeom

        cv2 = self._cv2()
        cap = self.run / "captures"
        cap.mkdir(exist_ok=True)
        img = self.grab("table4k")
        img_path, cam_path = cap / f"cap_{k:03d}.png", cap / f"cap_{k:03d}_camera.json"
        cv2.imwrite(str(img_path), img)
        _, _, path, res = self.products["table4k"]
        gcam = UsdGeom.Camera(self.stage.GetPrimAtPath(path))
        xf = UsdGeom.Xformable(gcam.GetPrim()).ComputeLocalToWorldTransform(0)
        cam_json = {
            "camera": "table",
            "resolution": list(res),
            "focal_length_mm": float(gcam.GetFocalLengthAttr().Get()),
            "horizontal_aperture_mm": float(gcam.GetHorizontalApertureAttr().Get()),
            "vertical_aperture_mm": float(gcam.GetVerticalApertureAttr().Get()),
            "camera_to_world": [[float(xf[r][c]) for c in range(4)] for r in range(4)],
            "table_top_z": float(C.TABLE_TOP_Z + C.SURFACE_DZ),
            "robot_base_xy": [0.0, 0.0],
        }
        cam_path.write_text(json.dumps(cam_json, indent=2))
        return img_path, cam_path

    def track_target(self, pick, poses, radius=0.02):
        """Scoring only: the simulated object a pick aims at, if any."""
        prefix = X.PRIMS[pick["category"]]
        cands = [n for n in self.names if n.startswith(prefix)]
        best = min(cands, key=lambda n: np.linalg.norm(np.array(poses[n]["centre"][:2]) - pick["xy"]), default=None)
        if best and np.linalg.norm(np.array(poses[best]["centre"][:2]) - pick["xy"]) > radius:
            return None
        return best


def run_open_loop(cell, plan):
    """B5: the initial plan, chained in plan order."""
    events, abort_reason = [], None
    for pick, segs in X.chained_segments(plan["picks"], cell.home):
        pick = dict(pick)
        r, _ = fk.fk(cell.q)
        current = r @ np.array(C.FINGER_SPREAD_AXIS)
        spread = np.array(pick["spread_dir"], float)
        if np.dot(current, spread) < 0:
            spread = -spread
        dest = np.array(pick["place_spread_dir"], float)
        if np.dot(spread, dest) < 0:
            dest = -dest
        start = cell.poses()
        tracked = cell.track_target(pick, start)
        event = {
            "order_index": pick["order_index"], "detection_id": pick["detection_id"],
            "category": pick["category"], "bin": pick["bin"], "tracked_object": tracked,
            "pickup_verified": False, "placement_verified": False,
        }
        if tracked:
            event["perception_error_mm"] = round(float(np.linalg.norm(np.array(start[tracked]["centre"][:2]) - pick["xy"]) * 1000), 3)
        seg_logs = []
        try:
            for name, p0, p1, duration, j0, j1, s in segs:
                direction = None if s is None else (spread if name in ("approach", "descend", "close", "dwell", "lift") else dest)
                seg_logs.append(cell.move(name, p0, p1, duration, j0, j1, direction))
                if name == "dwell":
                    event["jaw_gap_mm"] = cell.jaw_gap_mm()
                if name == "descend":
                    event["grasp_tip_error_mm"] = round(float(np.linalg.norm(cell.tip_now - p1) * 1000), 3)
                if name == "lift":
                    lifted = cell.poses()
                    event["objects_lifted"] = [k for k in cell.names if lifted[k]["centre"][2] - start[k]["centre"][2] > 0.025]
                    event["pickup_verified"] = tracked in event["objects_lifted"]
                    print(f"[sim_backend] pick {pick['order_index']} ({pick['category']}): lifted {event['objects_lifted']}", flush=True)
        except RuntimeError as e:
            abort_reason = str(e)
            print(f"[sim_backend] abort: {abort_reason}", flush=True)
        finish = cell.poses()
        if tracked:
            ok_bin, where = X.in_bin(finish[tracked]["centre"], X.bin_at(pick["place_xy"], C.SORT_BINS))
            event["placement_verified"] = bool(event["pickup_verified"] and ok_bin)
            event["placement"] = where
        event["others_disturbed"] = disturbed(cell, start, finish, tracked)
        event["segments"] = seg_logs
        event.update(X.summarise(seg_logs))
        events.append(event)
        (cell.run / "execution_events.json").write_text(json.dumps(events, indent=2) + "\n")
        if abort_reason:
            break
    write_report(cell, plan, events, abort_reason)


def disturbed(cell, start, finish, tracked):
    """Objects on the table, other than the target, moved > 5 mm."""
    def binned(c):
        return any(X.in_bin(c, b)[0] for b in C.SORT_BINS.values())

    return [
        k for k in cell.names
        if k != tracked and not binned(start[k]["centre"])  # already sorted: jostled by the drop
        and np.linalg.norm(np.subtract(finish[k]["centre"], start[k]["centre"])) > 0.005
    ]


def write_report(cell, plan, events, abort_reason, extra=None):
    after = cell.poses()
    totals = X.summarise([s for e in events for s in e.get("segments", [])])
    report = {
        "source": "sim_backend",
        "task": "bin_sort",
        "order": plan.get("order"),
        "predicted_travel_m": plan.get("predicted_travel_m"),
        "cycle_time_s": totals["time_s"],
        "tcp_path_commanded_m": totals["tcp_path_commanded_m"],
        "tcp_path_measured_m": totals["tcp_path_measured_m"],
        "wall_time_s": round(time.time() - cell.t_wall0, 1),
        "frames": cell.frame,
        "rendered": cell.render,
        "abort_reason": abort_reason,
        "picks_planned": len(plan["picks"]),
        "picks_attempted": len(events),
        "verified_pickups": sum(bool(e.get("pickup_verified")) for e in events),
        "verified_placements": sum(bool(e.get("placement_verified")) for e in events),
        "task_complete": bool(plan["picks"]) and abort_reason is None and len(events) == len(plan["picks"])
        and all(e.get("placement_verified") for e in events),
        "skipped_at_planning": [{k: s.get(k) for k in ("detection_id", "category", "skip_reason")} for s in plan["skipped"]],
        "picks": events,
        "after": after,
        "hardware": {"gripper_model": C.GRIPPER_MODEL, "jaw_stroke_m": C.JAW_STROKE, "open_gap_m": 2 * C.JAW_OPEN},
    }
    if extra:
        report.update(extra)
    if cell.render:
        report["render_check"] = cell.render_check()
    (cell.run / "execution_report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(
        f"[sim_backend] {report['verified_pickups']} pickups, {report['verified_placements']} in bin of "
        f"{report['picks_planned']} planned; cycle {report['cycle_time_s']} s, TCP {report['tcp_path_measured_m']} m",
        flush=True,
    )
    return report
