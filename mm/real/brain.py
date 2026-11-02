"""
The v6 closed loop as an act(obs) policy for the real cell (organizers' client). v6 changes: see V5_STEPS.

Runs in the YOLO environment, behind mm/real/brain_server.py; the client's
policy.py forwards each observation to it. One act() = one decision:

  1. capture: YOLO + pose fit on the table image (1080p model). Bins (three
     identical tangerine bins): the first capture with all three names them
     by position, and later ones track them by position (fewer than three at
     the start: hold and capture again, never guess). Which class goes to which
     bin is chosen once per rollout, by shortest planned travel (bin_assign)
  2. judge the previous pick from this image: the object still at its spot
     means a miss (the gripper reports only open/closed) -> retry, at most
     tracker.MAX_ATTEMPTS, then skip; gone means picked
  3. plan (plan_bins: grasp, bin, finger and reach checks), track and order
     (tracker), take the next target
  4. solve the whole path frame by frame with the twin's IK (straight, wrist
     the other way, via home; none -> skip the object)
  5. return it as UF0 rows in three moves, each sent only once the arm has
     reached the end of the one before (the cell's tcp_pose, ARRIVE_TOL):
     hover + descend + pause | close, lift, carry, lower + pause | release,
     retreat, park. The real arm lags its rows and stops where it is when
     they run out, so a move it has not finished is finished ("wait", no new
     capture) before the gripper acts; never closed or opened on the clock
Nothing left: hold still (one row, where the arm is).

The first act of a rollout only parks (the operator's home pose may hide the
table); the next capture waits until the arm is there. Ground truth is never
read. Each act is logged under --log-dir.
"""

import json
import math
import os
import time

import cv2
import numpy as np

import fk
import scene_config as C
from mm.control import closed_loop as CL
from mm.control import executor as X
from mm.datagen import randomize as R
from mm.perception import perceive_yolo as PY
from mm.perception.bins import bins_seen, first_bins, track_bins
from mm.planning import bin_assign as BA
from mm.planning import grasp as G
from mm.planning import plan_bins as PB
from mm.planning.tracker import MATCH_RADIUS, Tracker
from mm import paths
from mm.datagen.masks_to_yolo import roi_for
from mm.perception.pose_fit import MID_HEIGHT
from mm.real import frames as F
from mm.real import scene_check as SC
from mm.real import wrist_check as WC
from mm.real.park import arm_points

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DT = 0.2  # s per row (the cell plays one row per dt)
# The real arm moved ~50 mm/s however fast its rows asked (27 Oct 2026 logs: 146 mm in 2.8 s, 50 mm in 1.2 s) and
# stopped where it was when they ran out. Rows are eased (peak 1.5x the mean): a 30 mm/s mean peaks at 45 mm/s.
TCP_SPEED = 0.030  # m/s mean along each segment (the sim uses executor.MAX_TCP_SPEED)
ARRIVE_TOL = 0.003  # m: fingertip to the grasp / release pose before the gripper acts
ARRIVE_R_TOL = 3.0  # deg: jaw direction (R, mod 180) to the grasp / release pose
SETTLE_S = 0.6  # s held at the grasp / release pose before the arrival check
MAX_WAITS = 4  # acts spent finishing one move before giving it up
RELEASE_XY_TOL = 0.015  # m: after MAX_WAITS, still release if this close above the bin (opening ~80 mm)
# Park: fingers down beside the work area; mm/real/park.py: the arm stays
# > 300 px clear of the work area and bins in the 1080p table image.
PARK_XYZ = (0.18, 0.40, 0.35)  # base frame, m; UF0 (180, 400, 20) mm
PARK_TOL = 0.02
MAX_ROWS = 100  # the cell's chunk limit (mock_cell.MAX_CHUNK, MAX_QUEUE)
# v6. MM_V5_STEPS lists the steps that run as in v5, e.g. "3,4" (step 2, the outline grasp, is read in
# perceive_yolo; MM_GRASP=pose_fit does the same):
#   3  after a release, retreat to a clear-view pose on the bin's side, not to park; park only at the start and
#      when a tracked target or a bin is hidden behind the arm
#   4  wrist-camera grasp check at the hover pose (before descending / after lifting); a miss retries at once
#   5  stop (home, then hold) after IDLE_STOP_ACTS captures with nothing pickable; wait for a settled table with no
#      hand on it; bins re-found every capture (any distance), none used that is unseen or outside the TCP box
V5_STEPS = {s.strip() for s in os.environ.get("MM_V5_STEPS", "").split(",") if s.strip()}
# mm/real/park.py: fingers down at (0.18, -0.38, 0.35) keeps the arm 244 px clear of the work area and fixed bins in
# the 1080p table image (the +y park, 313 px). Bins on the -y side retreat there, on the +y side to PARK_XYZ.
RETREAT_XYZ = (0.18, -0.38, 0.35)  # base frame, m; UF0 (180, -380, 20) mm
IDLE_STOP_ACTS = 3
STOP_HOLD_ROWS = 25  # 5 s still at home per act once stopped
HIDDEN_PAD_PX = 25  # a tracked target this close to the arm's outline in the table image counts as hidden
# First that exists. v5_real leads by the owner's decision (28 Oct, docs/setup_notes.md) although it failed the
# compare_real rule (it calls a silver AA an aaa_cell); v4 found 2 of 255 real targets. Back: delete the first entry.
DEFAULT_WEIGHTS = ("yolo_seg_v5_real_1080", "yolo_seg_v4_1080", "yolo_seg_v3_1080", "yolo_seg_v3")


def default_weights():
    for run in DEFAULT_WEIGHTS:
        p = os.path.join(paths.model_dir(run), "weights", "best.pt")
        if os.path.exists(p):
            return p
    raise FileNotFoundError("no YOLO weights under models/ (v6 or its fallback, mm/paths.py)")


def gripper(jaw):
    """Sim jaw target -> the cell's binary gripper (0 open, 1 closed)."""
    return 0.0 if jaw >= C.JAW_OPEN - 1e-6 else 1.0


def r_error(a, b):
    """Jaw-direction difference in deg; the jaws are symmetric, so mod 180."""
    return abs((a - b + 90.0) % 180.0 - 90.0)


def segments_to_rows(segments, r_start, dt=DT, speed=TCP_SPEED):
    """Executor segments (name, p0, p1, duration, jaw0, jaw1, direction) ->
    UF0 rows [x, y, z mm, -180, 0, R deg, gripper], one per dt.

    Positions follow each straight segment with sort_scene's easing; the
    jaw direction turns as in the sim (TURNING segments) and is held
    otherwise; R stays continuous (jaws are symmetric, R ~ R + 180).
    """
    rows, r_prev = [], r_start
    spread = F.r_to_spread(r_start)
    for name, p0, p1, duration, j0, j1, direction in segments:
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        duration = max(duration, float(np.linalg.norm(p1 - p0)) / speed)
        n = max(1, round(duration / dt))
        start = spread
        for i in range(n):
            t = (i + 1) / n
            e = t * t * (3 - 2 * t)
            p = p0 + (p1 - p0) * e
            d = start
            if direction is not None:
                d = np.asarray(direction, float)
                if name in X.TURNING:
                    ang = math.atan2(start[0] * d[1] - start[1] * d[0], start[0] * d[0] + start[1] * d[1]) * e
                    d = np.array([start[0] * math.cos(ang) - start[1] * math.sin(ang),
                                  start[0] * math.sin(ang) + start[1] * math.cos(ang), 0.0])
            r_prev = F.spread_to_r(d, r_prev)
            rows.append([*F.base_to_uf0(p), -180.0, 0.0, r_prev, gripper(j1)])
        spread = F.r_to_spread(r_prev)
    return rows


def split_pick(segments, category, r_start):
    """A pick's segments -> its three moves as rows: to the grasp pose (+ pause, open), close and carry to the
    release pose (+ pause, closed), release and park. Each move ends where the next one starts."""
    names = [s[0] for s in segments]
    i_close, i_release = names.index("close"), names.index("release")
    grasp, place = segments[i_close - 1], segments[i_release - 1]  # descend, lower
    shut = X.jaw_closed(category)
    rows_a = segments_to_rows(segments[:i_close] + [("hold", grasp[2], grasp[2], SETTLE_S, C.JAW_OPEN, C.JAW_OPEN, grasp[6])],
                              r_start)
    rows_b = segments_to_rows(segments[i_close:i_release] + [("hold", place[2], place[2], SETTLE_S, shut, shut, place[6])],
                              rows_a[-1][5])
    rows_c = segments_to_rows(segments[i_release:], rows_b[-1][5])
    return rows_a, rows_b, rows_c, shut


def split_pick_wrist(segments, category, r_start):
    """v6 step 4: the pick in five moves, so the wrist camera can look from the hover pose before the descent and
    after the lift: to hover (+ pause) | descend (+ pause) | close, lift back to hover (+ pause) | carry, lower
    (+ pause) | release, retreat. Returns the moves' rows and the jaw when shut."""
    names = [s[0] for s in segments]
    i_desc, i_close, i_lift, i_release = (names.index(n) for n in ("descend", "close", "lift", "release"))
    shut = X.jaw_closed(category)
    up, grasp, place = segments[i_desc][1], segments[i_desc][2], segments[i_release - 1][2]
    d_pick, d_place = segments[i_desc][6], segments[i_release - 1][6]
    o = C.JAW_OPEN
    moves, r = [], r_start
    for segs in (segments[:i_desc] + [("hold", up, up, SETTLE_S, o, o, d_pick)],
                 segments[i_desc:i_close] + [("hold", grasp, grasp, SETTLE_S, o, o, d_pick)],
                 segments[i_close:i_lift + 1] + [("hold", up, up, SETTLE_S, shut, shut, d_pick)],
                 segments[i_lift + 1:i_release] + [("hold", place, place, SETTLE_S, shut, shut, d_place)],
                 segments[i_release:]):
        moves.append(segments_to_rows(segs, r))
        r = moves[-1][-1][5]
    return moves, shut


class Brain:
    def __init__(self, weights=None, cameras_json=None, image_size=(1920, 1080), log_dir=None, bins_mode="detected"):
        from ultralytics import YOLO

        self.weights = weights or default_weights()
        self.model = YOLO(self.weights)
        self.imgsz = PY.model_imgsz(self.weights)
        self.classes = PY.load_classes()
        self.roi = PY.load_roi()
        self.cameras_json = cameras_json or os.environ.get("MM_CAMERAS_JSON") or os.path.join(
            os.path.dirname(ROOT), "PhysicalAI-Hackathon-2026-v6", "calibration", "cameras.json")
        self.cam = F.CellCamera.load(self.cameras_json, "table", image_size)
        self.reach = G.Reach()
        self.home = np.asarray(X.HOME_XYZ, float)
        self.park = np.asarray(PARK_XYZ, float)
        self.bins_mode = bins_mode
        self.log_dir = log_dir
        self.wrist_cam = None if "4" in V5_STEPS else F.WristCamera.load(self.cameras_json, image_size)
        self.roi_box = roi_for(self.roi, *image_size)
        self.rollout = 0
        self.reset()

    def reset(self):
        self.tracker = Tracker()
        self.phys = None  # physical bins by name (bin_1..3), tracked by position
        self.assignment = None  # {class: bin name}, chosen at the first full view, kept for the rollout
        self.bins = None  # the planner's table: class -> its bin
        self.pending = None  # the pick played last: judged from the next image
        self.queued = []  # rows of a path longer than one chunk, sent on the next act
        self.expect = None  # where the move sent last ends, and the moves after it (_follow)
        self.note = None  # a move given up, for the next capture's log
        self.cycle = 0
        self.acts = 0
        self.history = []
        self.rollout += 1
        self.idle = 0  # step 5: captures in a row with nothing pickable
        self.stopped = None  # step 5: why the rollout was stopped
        self.scene_prev = None  # step 5: the last capture's targets and bins, for the settled-table check
        self.scene_ready = "5" in V5_STEPS
        self.wcheck = None  # step 4: the wrist image at hover before the descent, its window, the pick's moves
        self.wrist_bgr = None  # the wrist image of the current observation

    # ------------------------------------------------------------ helpers
    def _log(self, record, table_bgr=None):
        if not self.log_dir:
            return
        d = os.path.join(self.log_dir, f"rollout_{self.rollout:02d}")
        os.makedirs(d, exist_ok=True)
        stem = os.path.join(d, f"act_{self.acts:03d}")
        if table_bgr is not None:
            cv2.imwrite(stem + "_table.jpg", table_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        with open(stem + ".json", "w") as f:
            json.dump(record, f, indent=1, default=float)

    def _check(self, q):
        return lambda segs: X.check_path(q, [(s[0], s[1], s[2], s[3], s[6]) for s in segs], self.reach.limits, self.reach.tool)

    def _park_leg(self, start, direction, to=None):
        to = self.park if to is None else np.asarray(to, float)
        return ("park", np.asarray(start, float), to, X._duration(1.1, start, to), C.JAW_OPEN, C.JAW_OPEN,
                np.asarray(direction, float))

    def _clear_pose(self, bin_xy):
        """Step 3: where the arm waits for the next capture after a release: the clear-view pose on the bin's side
        (v5: always park)."""
        if "3" in V5_STEPS or bin_xy is None or bin_xy[1] >= 0:
            return self.park
        return np.asarray(RETREAT_XYZ, float)

    def _at_clear(self, tip):
        return min(np.linalg.norm(tip - self.park), np.linalg.norm(tip - np.asarray(RETREAT_XYZ))) <= PARK_TOL

    def _log_image(self, name, img):
        if self.log_dir and img is not None:
            d = os.path.join(self.log_dir, f"rollout_{self.rollout:02d}")
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, f"act_{self.acts:03d}_{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # ------------------------------------------------------------ step 4: wrist check
    def _wrist_before(self, rec, joints_deg):
        """At the hover pose before the descent: keep the wrist image and the pick spot's window in it."""
        w, p = self.wcheck, self.pending
        if w is None or p is None or self.wrist_cam is None or self.wrist_bgr is None:
            return
        q = np.deg2rad(np.asarray(joints_deg[:6], float))
        spot = [p["xy"][0], p["xy"][1], C.SURFACE_DZ + MID_HEIGHT.get(p["category"], 0.005)]
        w.update(before=self.wrist_bgr.copy(), win=WC.window(self.wrist_cam, q, spot, p["category"]))
        rec["wrist_window"] = w["win"]
        self._log_image("wrist_before", self.wrist_bgr)

    def _after_lift(self, rec, table_bgr, tip, r_now):
        """Back at the hover pose with the jaws shut: is the object still at its spot? picked or unclear: carry on
        (unclear: the next table capture judges, as in v5). missed: open and descend again at once, at most
        tracker.MAX_ATTEMPTS tries in all; then open, give the object up and go to the clear-view pose."""
        w, p = self.wcheck, self.pending
        if w is None or p is None or w.get("before") is None:
            return None
        verdict, detail = WC.verdict(w["before"], self.wrist_bgr, w["win"])
        self._log_image("wrist_after", self.wrist_bgr)
        rec.update(wrist_verdict=verdict, wrist=detail, attempt=p["attempt"])
        if verdict == "unclear":
            return None
        if verdict == "picked":
            if p["track"] in self.tracker.tracks:
                self.tracker.picked(p["track"])
            self.history.append(dict(p, verdict="picked (wrist)"))
            self.pending = None
            return None
        gave_up = self.tracker.attempt_failed(p["track"]) if p["track"] in self.tracker.tracks else True
        self.history.append(dict(p, verdict="missed (wrist)" + (" (gave up)" if gave_up else "")))
        d = F.r_to_spread(r_now)
        self.acts += 1
        if not gave_up:
            p["attempt"] += 1
            rows = segments_to_rows([("hold", tip, tip, SETTLE_S, C.JAW_OPEN, C.JAW_OPEN, d)], r_now) + w["moves"][1]
            self._expect("grasp", rows, ARRIVE_TOL, C.JAW_OPEN, then=w["then"])
            rec.update(act=self.acts, decision="retry", reason=f"wrist: still at its spot, try {p['attempt']}",
                       rows=len(rows))
        else:
            self.pending = None
            dest = self._clear_pose(None)
            segs = [("hold", tip, tip, SETTLE_S, C.JAW_OPEN, C.JAW_OPEN, d), self._park_leg(tip, d, dest)]
            rows = segments_to_rows(segs, r_now)
            self._expect("park", rows, PARK_TOL, C.JAW_OPEN)
            rec.update(act=self.acts, decision="give_up", reason="wrist: missed on every try", rows=len(rows))
        self._log(rec, table_bgr)
        return {"rows": rows, "dt": DT, "info": rec}

    def _hold(self, tip, r, grip):
        return [[*F.base_to_uf0(tip), -180.0, 0.0, r, grip]]

    def _state(self, joints_deg, tcp_pose):
        """Joints (rad), gripper, fingertip (base), jaw direction, R and the fingertip in UF0 mm (the cell's
        tcp_pose when it reports one, else the twin's FK of the joints)."""
        q = np.deg2rad(np.asarray(joints_deg[:6], float))
        grip = float(joints_deg[6]) if len(joints_deg) > 6 else 0.0
        rot, tip = fk.tool_pose(q, self.reach.tool)
        spread = rot @ np.array(C.FINGER_SPREAD_AXIS)
        r = float(tcp_pose[5]) if tcp_pose is not None else F.spread_to_r(spread)
        here = np.asarray(tcp_pose[:3] if tcp_pose is not None else F.base_to_uf0(tip), float)
        return q, grip, tip, spread, r, here

    def _expect(self, label, rows, tol, jaw, then=()):
        """The move just sent ends at rows[-1]: label is what waits on it (grasp: close, release: open, park:
        capture). then: the moves after it, each {decision, label, rows, tol, jaw}."""
        self.expect = {"label": label, "target": list(rows[-1][:6]), "tol": tol, "jaw": jaw, "then": list(then), "waits": 0}

    def act(self, table_bgr, joints_deg, tcp_pose=None, wrist_bgr=None):
        """{"rows": UF0 rows (<= MAX_ROWS), "dt", "info"}; a longer path continues on the next call."""
        self.wrist_bgr = wrist_bgr
        if self.queued:
            rows, self.queued = self.queued[:MAX_ROWS], self.queued[MAX_ROWS:]
            self.acts += 1
            return {"rows": rows, "dt": DT, "info": {"act": self.acts, "decision": "continue", "rows": len(rows)}}
        out = self._follow(table_bgr, joints_deg, tcp_pose) if self.expect else None
        if out is None:
            out = self._decide(table_bgr, joints_deg, tcp_pose)
        rows = out["rows"]
        if len(rows) > MAX_ROWS:
            out["rows"], self.queued = rows[:MAX_ROWS], rows[MAX_ROWS:]
        return out

    # ------------------------------------------------------------ follow a move
    def _follow(self, table_bgr, joints_deg, tcp_pose):
        """The arm against the end of the move sent last. Not there yet: finish that move from where the arm is
        (no capture), at most MAX_WAITS times. There: send the next move of the pick, or None (the caller
        captures and decides)."""
        e = self.expect
        q, _, tip, spread_now, r_now, here = self._state(joints_deg, tcp_pose)
        target = np.asarray(e["target"][:3], float)
        err = float(np.linalg.norm(here - target)) / 1000.0
        r_err = r_error(r_now, e["target"][5]) if tcp_pose is not None and e["label"] != "park" else 0.0
        arrived = err <= e["tol"] and r_err <= ARRIVE_R_TOL
        rec = {"joints": list(map(float, joints_deg)), "tcp_pose": tcp_pose, "move": e["label"],
               "target": e["target"], "error_mm": round(err * 1000, 1), "r_error_deg": round(r_err, 1), "waits": e["waits"]}
        if not arrived and e["waits"] >= MAX_WAITS:
            if e["label"] in ("grasp", "hover"):  # never close where the object is not
                return self._abort(rec, table_bgr, q, tip, spread_now, r_now)
            if e["label"] == "park":  # capture from here rather than stall
                self.expect, self.note = None, dict(rec, gave_up=True)
                return None
            xy = float(np.linalg.norm(here[:2] - target[:2])) / 1000.0
            if xy <= RELEASE_XY_TOL and abs(here[2] - target[2]) / 1000.0 <= 0.01:
                arrived, rec["released_short"] = True, True
            # else: keep finishing the move; never open outside the bin
        if arrived:
            if e["label"] == "hover":
                self._wrist_before(rec, joints_deg)
            elif e["label"] == "lifted":
                out = self._after_lift(rec, table_bgr, tip, r_now)
                if out is not None:
                    return out
            if not e["then"]:
                self.expect = None
                return None
            nxt = e["then"][0]
            self._expect(nxt["label"], nxt["rows"], nxt["tol"], nxt["jaw"], e["then"][1:])
            self.acts += 1
            rec.update(act=self.acts, decision=nxt["decision"], rows=len(nxt["rows"]))
            self._log(rec, table_bgr)
            return {"rows": nxt["rows"], "dt": DT, "info": rec}
        e["waits"] += 1
        end = F.uf0_to_base(e["target"][:3])
        d = F.r_to_spread(e["target"][5])
        segs = [("wait", tip, end, max(DT, float(np.linalg.norm(end - tip)) / TCP_SPEED), e["jaw"], e["jaw"], d)]
        if e["label"] != "park":
            segs.append(("hold", end, end, SETTLE_S, e["jaw"], e["jaw"], d))
        self.acts += 1
        if self._check(q)(segs) is None:
            rows = segments_to_rows(segs, r_now)
            rec.update(act=self.acts, decision="wait", rows=len(rows))
        else:
            rows = self._hold(tip, r_now, gripper(e["jaw"]))
            rec.update(act=self.acts, decision="wait", reason="no IK path to finish the move: hold", rows=1)
        self._log(rec, table_bgr)
        return {"rows": rows, "dt": DT, "info": rec}

    def _abort(self, rec, table_bgr, q, tip, spread_now, r_now):
        """The arm never reached the grasp pose: lift straight up and park with the gripper open; a failed
        attempt for the tracker (the gripper never closed)."""
        up = tip + np.array([0.0, 0.0, X.HOVER_DZ])
        segs = [("lift", tip, up, 0.8, C.JAW_OPEN, C.JAW_OPEN, spread_now), self._park_leg(up, spread_now)]
        p = self.pending
        if p and p["track"] in self.tracker.tracks:
            gave_up = self.tracker.attempt_failed(p["track"])
            self.history.append(dict(p, verdict="aborted: grasp pose not reached" + (" (gave up)" if gave_up else "")))
        self.pending = None
        self.acts += 1
        rows = segments_to_rows(segs, r_now) if self._check(q)(segs) is None else []
        if rows and all(F.in_tcp_box(r) for r in rows):
            self._expect("park", rows, PARK_TOL, C.JAW_OPEN)
        else:
            rows, self.expect = self._hold(tip, r_now, 0.0), None
        rec.update(act=self.acts, decision="abort", reason="grasp pose not reached", rows=len(rows))
        self._log(rec, table_bgr)
        return {"rows": rows, "dt": DT, "info": rec}

    # ------------------------------------------------------------ decide
    def _decide(self, table_bgr, joints_deg, tcp_pose=None):
        t0 = time.time()
        self.acts += 1
        q, grip, tip, spread_now, r_now, _ = self._state(joints_deg, tcp_pose)
        rec = {"act": self.acts, "joints": list(map(float, joints_deg)), "tcp_pose": tcp_pose, "tip_base": tip.tolist()}
        if self.note:
            rec["move_given_up"], self.note = self.note, None

        # First act of a rollout: park, so the arm does not hide the table.
        if self.acts == 1 and np.linalg.norm(tip - self.park) > PARK_TOL:
            leg = [self._park_leg(tip, spread_now)]
            route = [("direct", leg), ("via_home", [("reroute", tip, self.home, X._duration(1.1, tip, self.home),
                                                    C.JAW_OPEN, C.JAW_OPEN, spread_now), self._park_leg(self.home, spread_now)])]
            for label, segs in route:
                if self._check(q)(segs) is None:
                    rows = segments_to_rows(segs, r_now)
                    self._expect("park", rows, PARK_TOL, C.JAW_OPEN)
                    rec.update(decision="park", route=label, rows=len(rows))
                    self._log(rec, table_bgr)
                    return {"rows": rows, "dt": DT, "info": rec}
            rec.update(decision="hold", reason="no IK path to park")
            self._log(rec, table_bgr)
            return {"rows": self._hold(tip, r_now, grip), "dt": DT, "info": rec}

        if self.stopped:  # step 5: stopped at home; stay still until the operator ends the rollout
            rec.update(decision="stopped", reason=self.stopped, rows=STOP_HOLD_ROWS)
            return {"rows": self._hold(tip, r_now, 0.0) * STOP_HOLD_ROWS, "dt": DT, "info": rec}

        # 1. perceive; bins: named by position at the first full view, then tracked
        dets = PY.perceive(self.model, table_bgr, self.cam, self.roi, self.classes, self.imgsz)
        if self.bins_mode == "fixed":
            self.phys = R.load_bins()
        elif self.phys is None:
            self.phys = first_bins(dets)
        else:  # step 5: re-found every capture wherever they are now (v5: within TRACK_RADIUS of the last pose)
            self.phys = track_bins(dets, self.phys, radius=math.inf if "5" not in V5_STEPS else None)
        if "5" not in V5_STEPS:
            hand = SC.foreign(table_bgr, dets, self.roi_box, self.cam, arm_points(q))
            if hand:  # never move while someone reaches over the table; the table must settle again first
                self.scene_ready, self.scene_prev = False, None
                return self._wait(rec, table_bgr, tip, r_now, grip, f"something large on the table ({hand['px']} px "
                                  f"at {hand['box']}): a hand? waiting", t0)
            if not self.scene_ready:
                sig = SC.signature(dets)
                if not SC.stable(self.scene_prev, sig):
                    self.scene_prev = sig
                    return self._wait(rec, table_bgr, tip, r_now, grip, "waiting for the same objects and bins in two "
                                      "captures in a row", t0)
                self.scene_ready = True
                rec["scene_ready"] = True
        if "3" not in V5_STEPS and np.linalg.norm(tip - self.park) > PARK_TOL:
            hidden = self._hidden(dets, q)
            if hidden or self.phys is None:  # step 3: something may be behind the arm: look from the park pose
                why = f"hidden behind the arm: {hidden}" if hidden else "fewer than 3 bins seen"
                return self._go(rec, table_bgr, q, tip, spread_now, r_now, self.park, "park", why, t0)
        if self.phys is None:  # fewer than 3 bins: hold and capture again, never guess
            rec.update(detections=[{k: d.get(k) for k in ("class", "conf", "xy_robot_m", "fit_iou")} for d in dets])
            return self._nothing(rec, table_bgr, q, tip, r_now, grip, f"{len(bins_seen(dets))} of 3 bins found: "
                                 "capture again", [], t0)
        for d in dets:
            if d["eligible"] and d["xy_robot_m"] and any(X.xy_in_bin(d["xy_robot_m"], b) for b in self.phys.values()):
                d["eligible"], d["skip_reason"] = False, "in_bin"
        if self.assignment is None:  # once per rollout
            self.assignment, log = BA.assign_bins({"objects": dets}, self.cam, self.phys, reach=self.reach)
            rec["bin_assignment"] = log
        self.bins = BA.table(self.phys, self.assignment)
        plan = PB.build_plan({"objects": dets}, self.cam, bins=self.bins, reach=self.reach)
        if "5" not in V5_STEPS:  # step 5: never release into a bin not seen now, or one outside the TCP box
            keep = []
            for p in plan["picks"]:
                why = self._bin_unusable(self.bins[p["bin"]])
                if why:
                    plan["skipped"].append({k: p[k] for k in ("detection_id", "category", "conf", "xy", "yaw_deg")}
                                           | {"skip_reason": why, "stage": "bins", "bin": self.bins[p["bin"]]["id"]})
                else:
                    keep.append(p)
            plan["picks"] = keep

        # 2. judge the last pick from this image
        if self.pending:
            p = self.pending
            still = [d for d in dets if d["class"] == p["category"] and d["xy_robot_m"]
                     and math.dist(d["xy_robot_m"], p["xy"]) <= MATCH_RADIUS]
            verdict = "missed" if still else "picked"
            if p["track"] in self.tracker.tracks:
                if still:
                    gave_up = self.tracker.attempt_failed(p["track"])
                    verdict += " (gave up)" if gave_up else ""
                else:
                    self.tracker.picked(p["track"])
            rec["previous_pick"] = dict(p, verdict=verdict)
            self.history.append(rec["previous_pick"])
            self.pending = None

        # 3. track and order, 4. route
        change = self.tracker.update(plan["picks"], tip, self.cycle)
        self.cycle += 1
        rec.update(
            detections=[{k: d.get(k) for k in ("class", "conf", "xy_robot_m", "yaw_deg", "fit_iou", "eligible", "skip_reason", "colour")}
                        for d in dets],
            bins={n: {"xy": b["xy"], "yaw_deg": b.get("yaw_deg", 0.0), "found": b.get("found", True),
                      "holds": [s for s, bid in self.assignment.items() if bid == n]} for n, b in self.phys.items()},
            skipped=[(s["category"], s["skip_reason"]) for s in plan["skipped"]],
            tracker={k: change[k] for k in ("new", "moved", "vanished", "order")},
        )
        check = self._check(q)
        while True:
            tid, pick = self.tracker.next()
            if tid is None:
                clear = self._at_clear(tip) if "3" not in V5_STEPS else np.linalg.norm(tip - self.park) <= PARK_TOL
                if not clear:
                    segs = [self._park_leg(tip, spread_now)]
                    if check(segs) is None:
                        rows = segments_to_rows(segs, r_now)
                        self._expect("park", rows, PARK_TOL, C.JAW_OPEN)
                        rec.update(decision="park", rows=len(rows), seconds=round(time.time() - t0, 2))
                        self._log(rec, table_bgr)
                        return {"rows": rows, "dt": DT, "info": rec}
                return self._nothing(rec, table_bgr, q, tip, r_now, grip, "nothing left to pick", plan["skipped"], t0)
            spread = np.asarray(pick["spread_dir"], float)
            if np.dot(spread_now, spread) < 0:
                spread = -spread
            dest = np.asarray(pick["place_spread_dir"], float)
            if np.dot(spread, dest) < 0:
                dest = -dest
            picks, places = CL.route_options(pick, tip, spread, dest, self.home)

            def with_park(pick_segs, places=places, to=self._clear_pose(pick.get("place_xy"))):
                return [(label, segs + [self._park_leg(segs[-1][2], segs[-1][6], to)]) for label, segs in places(pick_segs)]

            route = CL.choose_route(check, picks, with_park)
            if route["segments"] is None:
                self.tracker.give_up(tid)
                rec.setdefault("ik_skipped", []).append({"track": tid, "category": pick["category"], "failures": route["failures"]})
                continue
            break
        wrist = self.wrist_cam is not None and self.wrist_bgr is not None
        if wrist:  # step 4: five moves, the wrist camera looking from the hover pose before and after
            moves, shut = split_pick_wrist(route["segments"], pick["category"], r_now)
        else:
            rows_a, rows_b, rows_c, shut = split_pick(route["segments"], pick["category"], r_now)
            moves = [rows_a, rows_b, rows_c]
        bad = [r for m in moves for r in m if not F.in_tcp_box(r)]
        if bad:  # never send what the cell would reject: skip this object
            self.tracker.give_up(tid)
            rec.update(reason=f"{len(bad)} rows outside the TCP box", track=tid)
            return self._nothing(rec, table_bgr, q, tip, r_now, grip, rec["reason"], plan["skipped"], t0)
        self.idle = 0
        self.pending = {"track": tid, "category": pick["category"], "xy": list(pick["xy"]), "bin": pick["bin"],
                        "attempt": self.tracker.tracks[tid]["attempts"] + 1}
        o = C.JAW_OPEN
        if wrist:
            then = [{"decision": "close_lift", "label": "lifted", "rows": moves[2], "tol": ARRIVE_TOL, "jaw": shut},
                    {"decision": "carry", "label": "release", "rows": moves[3], "tol": ARRIVE_TOL, "jaw": shut},
                    {"decision": "release", "label": "park", "rows": moves[4], "tol": PARK_TOL, "jaw": o}]
            self.wcheck = {"moves": moves, "then": then}
            self._expect("hover", moves[0], ARRIVE_TOL, o, then=[
                {"decision": "descend", "label": "grasp", "rows": moves[1], "tol": ARRIVE_TOL, "jaw": o}] + then)
        else:
            self.wcheck = None
            self._expect("grasp", moves[0], ARRIVE_TOL, o, then=[
                {"decision": "carry", "label": "release", "rows": moves[1], "tol": ARRIVE_TOL, "jaw": shut},
                {"decision": "release", "label": "park", "rows": moves[2], "tol": PARK_TOL, "jaw": o}])
        rec.update(decision="pick", pick=self.pending, route={k: route[k] for k in ("pick", "place", "failures")},
                   rows=len(moves[0]), moves=[len(m) for m in moves], wrist_check=wrist,
                   clear_pose=[round(v, 3) for v in self._clear_pose(pick.get("place_xy"))],
                   seconds=round(time.time() - t0, 2))
        self._log(rec, table_bgr)
        return {"rows": moves[0], "dt": DT, "info": rec}

    # ------------------------------------------------------------ v6 helpers for _decide
    def _wait(self, rec, table_bgr, tip, r_now, grip, why, t0):
        """Hold and capture again; not counted as idle (step 5)."""
        rec.update(decision="wait_scene", reason=why, seconds=round(time.time() - t0, 2))
        self._log(rec, table_bgr)
        return {"rows": self._hold(tip, r_now, grip), "dt": DT, "info": rec}

    def _go(self, rec, table_bgr, q, tip, spread_now, r_now, to, label, why, t0):
        segs = [self._park_leg(tip, spread_now, to)]
        if self._check(q)(segs) is not None:
            return self._wait(rec, table_bgr, tip, r_now, 0.0, why + "; no IK path to move: hold", t0)
        rows = segments_to_rows(segs, r_now)
        self._expect("park", rows, PARK_TOL, C.JAW_OPEN)
        rec.update(decision=label, reason=why, rows=len(rows), seconds=round(time.time() - t0, 2))
        self._log(rec, table_bgr)
        return {"rows": rows, "dt": DT, "info": rec}

    def _nothing(self, rec, table_bgr, q, tip, r_now, grip, why, skipped, t0):
        """Nothing pickable this capture. v5: hold and capture again. Step 5: the IDLE_STOP_ACTS-th time in a row,
        go home and stop, logging why each object was skipped."""
        self.idle += 1
        if "5" in V5_STEPS or self.idle < IDLE_STOP_ACTS:
            rec.update(decision="hold", reason=why, idle=self.idle, seconds=round(time.time() - t0, 2))
            self._log(rec, table_bgr)
            return {"rows": self._hold(tip, r_now, grip), "dt": DT, "info": rec}
        self.stopped = f"nothing pickable for {self.idle} captures in a row ({why})"
        rec["skipped_why"] = {
            "this_capture": [{k: s.get(k) for k in ("category", "xy", "skip_reason", "stage", "detail", "bin")} for s in skipped],
            "given_up": [{"track": t, "category": p["category"], "xy": p["xy"]} for t, p in self.tracker.flagged.items()],
            "history": self.history}
        segs = [("home", tip, self.home, X._duration(1.1, tip, self.home), C.JAW_OPEN, C.JAW_OPEN, F.r_to_spread(r_now))]
        if self._check(q)(segs) is None:
            rows = segments_to_rows(segs, r_now)
        else:
            rows = self._hold(tip, r_now, 0.0)
            self.stopped += "; no IK path home: stopped where it is"
        rec.update(decision="stop", reason=self.stopped, rows=len(rows), seconds=round(time.time() - t0, 2))
        self._log(rec, table_bgr)
        return {"rows": rows, "dt": DT, "info": rec}

    def _hidden(self, dets, q):
        """Step 3: tracked targets not detected now whose spot is behind the arm in the table image."""
        mask = SC.arm_mask(self.cam, (self.cam.H, self.cam.W), arm_points(q), HIDDEN_PAD_PX)
        out = []
        for tid in self.tracker.order:
            p = self.tracker.tracks[tid]["pick"]
            if any(d["class"] == p["category"] and d["xy_robot_m"] and math.dist(d["xy_robot_m"], p["xy"]) <= MATCH_RADIUS
                   for d in dets):
                continue
            uv = self.cam.project([p["xy"][0], p["xy"][1], self.cam.table_z])
            if uv is not None and 0 <= uv[0] < self.cam.W and 0 <= uv[1] < self.cam.H and mask[int(uv[1]), int(uv[0])]:
                out.append(tid)
        for name, b in (self.phys or {}).items():
            uv = self.cam.project([b["xy"][0], b["xy"][1], self.cam.table_z])
            if not b.get("found", True) and uv is not None and 0 <= uv[0] < self.cam.W and 0 <= uv[1] < self.cam.H \
                    and mask[int(uv[1]), int(uv[0])]:
                out.append(name)
        return out

    @staticmethod
    def _bin_unusable(b):
        if not b.get("found", True):
            return "bin_not_seen_now"
        above = F.base_to_uf0([b["xy"][0], b["xy"][1], C.SURFACE_DZ + b.get("release_z", 0.0) + X.HOVER_DZ])
        if not F.in_tcp_box([*above, 0, 0, 0, 0]):
            return "bin_outside_tcp_box"
        return None
