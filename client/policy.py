"""Machine Minds policy (v6): the closed loop (YOLO perception -> grasp/bin/order planning -> IK-checked TCP
waypoints), from our Isaac Sim project. Three identical tangerine-yellow bins; the robot assigns cube / screw /
AAA cell to them at the first capture of each rollout (shortest planned travel) and keeps that choice.
v5: waypoints timed to the real arm's speed, and the gripper acts only once tcp_pose shows the arm has arrived.
v6: cubes and screws grasped from their YOLO outline (no pose-fit gate); retreat to a clear-view pose on the bin's
side instead of parking after every pick; a wrist-camera grasp check at the hover pose with an immediate retry;
wait for a settled table with no hand on it; stop at home after 3 captures with nothing pickable.

act(obs) returns the arm's next targets; when they have played, act is called again with a new observation.
Each call is one decision of the loop, made by mm/real/brain.py in our project, which runs in the YOLO
environment behind a JSON-lines pipe (torch/ultralytics stay out of this client's environment):
  - first call of a rollout: park the arm clear of the table camera
  - then: detect on the table image, plan the next pick and return it in moves, each once the arm has finished the
    one before: hover | descend | close, lift to hover | carry, lower | release, retreat. At the hover pose the
    wrist image before the descent and after the lift says picked or missed (missed: open and descend again, at
    most 3 tries); unclear: the next table image judges, as in v5. A move the arm has not finished is finished
    first ("wait", no new detection). Nothing pickable 3 times in a row: go home and hold still
  MM_V5_STEPS=2,3,4,5 (any subset) runs those steps as in v5 (mm/real/brain.py).
Where things are (environment variables, defaults for our Windows PC):
  MM_ROOT   our project            (../v6, next to this folder)
  YOLO_PY   its YOLO interpreter   (C:\\isim\\yolo\\python.exe)
  MM_CAMERAS_JSON  calibration     (calibration/cameras.json here; the sim test passes the twin's)
--checkpoint <weights.pt> overrides the YOLO weights (default: our 1080p model).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PARK_ROW = [180.0, 400.0, 20.0, -180.0, 0.0, 0.0, 0.0]  # UF0: the brain's park pose, used only if it fails


class Policy:
    CAMERAS = ["table", "wrist"]  # the cameras you want, any subset
    IMAGE_SIZE = (1920, 1080)     # (width, height): the table camera's full 1080p frame
    IMAGE_FORMAT = "png"          # lossless: perception fits silhouettes to the mask edges
    ACTION = "tcp"                # [x, y, z mm, w, p, r deg, gripper] in UF0
    PREFETCH = False              # each act needs a fresh image of the table after the last pick
    DEPTH = []
    DT = 0.2                      # s per row (mm/real/brain.py DT)

    def __init__(self, checkpoint: str | None = None):
        self.root = Path(os.environ.get("MM_ROOT", HERE.parent / "v6")).resolve()
        yolo = os.environ.get("YOLO_PY", r"C:\isim\yolo\python.exe" if os.name == "nt" else "python")
        cams = os.environ.get("MM_CAMERAS_JSON", str(HERE / "calibration" / "cameras.json"))
        self.tmp = Path(tempfile.mkdtemp(prefix="mm_policy_"))
        cmd = [yolo, "-u", "-m", "mm.real.brain_server", "--cameras", cams,
               "--width", str(self.IMAGE_SIZE[0]), "--height", str(self.IMAGE_SIZE[1])]
        if checkpoint:
            cmd += ["--weights", str(Path(checkpoint).resolve())]
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
        self.err = open(self.tmp / "brain_stderr.log", "w")
        self.p = subprocess.Popen(cmd, cwd=self.root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.err, text=True, bufsize=1, env=env)
        ready = self._read()
        if not ready.get("ready"):
            raise RuntimeError(f"brain did not start: {ready} (see {self.tmp / 'brain_stderr.log'})")
        self.info = ready
        self.last = None
        print(f"[policy] brain ready: {ready['weights']} imgsz {ready['imgsz']}, logs {ready['log_dir']}", flush=True)

    def _read(self) -> dict:
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"brain exited; see {self.tmp / 'brain_stderr.log'}")
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    def _ask(self, req: dict) -> dict:
        self.p.stdin.write(json.dumps(req) + "\n")
        self.p.stdin.flush()
        return self._read()

    def reset(self) -> None:
        """Called at the start of every rollout."""
        self._ask({"cmd": "reset"})

    def act(self, obs: dict):
        table = self.tmp / "table.png"
        cv2.imwrite(str(table), cv2.cvtColor(obs["images"]["table"], cv2.COLOR_RGB2BGR))
        tcp = obs.get("tcp_pose")
        wrist = None
        if obs["images"].get("wrist") is not None:
            wrist = self.tmp / "wrist.png"
            cv2.imwrite(str(wrist), cv2.cvtColor(obs["images"]["wrist"], cv2.COLOR_RGB2BGR))
        out = self._ask({"cmd": "act", "table": str(table), "joints": [float(v) for v in obs["joints"]],
                         "tcp_pose": None if tcp is None else [float(v) for v in tcp],
                         "wrist": None if wrist is None else str(wrist)})
        self.last = out
        if "error" in out:  # hold still rather than guess
            print(f"[policy] brain error: {out['error']}", file=sys.stderr, flush=True)
            if tcp is None:
                return np.array([PARK_ROW])
            return np.array([[*map(float, tcp[:6]), float(obs["joints"][6])]])
        info = out["info"]
        print(f"[policy] act {info.get('act')}: {info.get('decision')} {info.get('pick', '')} "
              f"rows {len(out['rows'])}", flush=True)
        return np.asarray(out["rows"], dtype=float)

    def close(self) -> None:
        try:
            self.p.stdin.close()
            self.p.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.p.kill()
        self.err.close()
