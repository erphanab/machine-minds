"""
mm.real.brain.Brain over stdin/stdout JSON lines, for the client's policy.py.
Runs in the YOLO environment:

    C:\\isim\\yolo\\python.exe -m mm.real.brain_server [--weights ...] [--log-dir ...]

Requests (one JSON object per line):
  {"cmd": "reset"}
  {"cmd": "act", "table": png path, "joints": [J1..J6 deg, gripper], "tcp_pose": [...] | null,
   "wrist": png path | null (v6: the wrist grasp check)}
Replies: {"ready": true} once loaded, then {"ok": true} / {"rows", "dt", "info"} / {"error"}.
Libraries that print to stdout are tolerated by the client (non-JSON lines).
"""

import argparse
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--cameras", default=None, help="cameras.json (default: MM_CAMERAS_JSON or the client's)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--bins", choices=["detected", "fixed"], default="detected")
    ap.add_argument("--log-dir", default=os.path.join(ROOT, "runs", "real", time.strftime("%Y%m%d_%H%M%S")))
    a = ap.parse_args()

    import cv2

    from mm.real.brain import Brain

    out = sys.stdout
    sys.stdout = sys.stderr  # keep stray prints off the protocol stream
    brain = Brain(a.weights, a.cameras, (a.width, a.height), a.log_dir, a.bins)

    def reply(obj):
        out.write(json.dumps(obj, default=float) + "\n")
        out.flush()

    reply({"ready": True, "weights": brain.weights, "imgsz": brain.imgsz, "cameras": brain.cameras_json,
           "log_dir": a.log_dir})
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            if req["cmd"] == "reset":
                brain.reset()
                reply({"ok": True})
            elif req["cmd"] == "act":
                img = cv2.imread(req["table"])
                if img is None:
                    raise ValueError(f"cannot read {req['table']}")
                wrist = cv2.imread(req["wrist"]) if req.get("wrist") else None  # v6 step 4
                reply(brain.act(img, req["joints"], req.get("tcp_pose"), wrist))
            else:
                raise ValueError(f"unknown cmd {req['cmd']!r}")
        except Exception as e:  # noqa: BLE001  (reported to the client)
            reply({"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()})


if __name__ == "__main__":
    main()
