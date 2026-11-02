# Machine Minds: closed-loop sorting of mixed small parts on the CSI robot cell

Team Machine Minds, Track 3, NVIDIA x Rockwell Physical AI Hackathon (Fall 2026), UWM Connected Systems Institute.

A FANUC ER-4iA arm with an SMC MHZL2-16D2 parallel gripper sorts loose cubes, screws and AAA cells from a table into
three identical yellow bins, one part type per bin, and leaves distractors (caps, tubes) alone. Bins and parts are
placed differently in every trial. The robot finds the bins from the camera, chooses which bin gets which type, picks
every target, checks each grasp with the wrist camera, and stops by itself when the table is clear.

The submitted policy is **v6** (`MachineMinds-v6-ClosedLoop`). **v5** is the baseline it is measured against.

## Media

| Clip | What it shows |
|---|---|
| [media/slide09_rollout.mp4](media/slide09_rollout.mp4) | One full real-cell trial (30 Oct, trial t05), 10x speed: mixed table to all targets sorted, cap left, robot stops itself |
| [media/slide06_clip1.mp4](media/slide06_clip1.mp4), [media/slide06_clip2.mp4](media/slide06_clip2.mp4) | Real cell, two different bin layouts (trials t02 and t07), 3x and 4x speed: pick and place into bins found that trial |
| [media/slide07_twin.mp4](media/slide07_twin.mp4) | Isaac Sim digital twin: a screw is nudged 183 mm mid-run, re-detected and sorted |

Each clip has a still with the same name (`.jpg`).

## The manufacturing problem

Kitting and sorting of mixed small parts (fasteners, cells, blocks) is still largely manual or done with fixed
fixtures and per-part programming. In high-mix, low-volume production the parts arrive loose and the bins move, so a
cell that needs fixed positions or a CAD-based pose model for every part is slow to change over. The goal is a cell
that sorts whatever it sees into whatever bins are on the table, with no reprogramming between layouts.

## Technical approach (v6)

One camera image in, TCP waypoints out, every step (closed loop):

1. **Perception.** YOLO11s-seg fine-tuned on real cell images (`models/yolo_seg_v5_real_1080`, 1080p input,
   inference at 1600 px) segments cubes, screws, AAA cells, bins, caps and tubes in the table-camera image.
2. **Outline grasp (hybrid perception).** Each mask is back-projected through the calibrated table camera onto the
   table plane at the part's mid height. Cube: mask centre and rectangle yaw. Screw: a width profile along its length
   finds the head; the grasp is the middle of the shank, never the head. AAA cell: pose fit. The image front end
   supplies the quantitative pose; this is not general 6D pose estimation.
3. **Bins and assignment.** Bins are re-detected every step. The robot chooses the bin-to-type assignment with the
   shortest planned travel.
4. **Planning.** Pick order, grasp angle (including the symmetric twins), IK and workspace checks for every waypoint;
   fingers may not touch a bin.
5. **Execution.** Five moves per pick: hover, descend, close and lift, carry and lower, release and retreat.
6. **Checks.** The wrist camera looks at the pick spot before and after the lift (picked / missed / unclear; a miss is
   retried). The policy waits for a stable scene before the first pick, pauses while a hand or other foreign object
   is on the table, and goes home and stops when no pickable target is left.

Each v6 change can be switched back to v5 behaviour (`MM_V5_STEPS=2,3,4,5,6`, see `docs/setup_notes.md`).

## How it uses physical AI and the CSI equipment

- **Real cell (CSI):** the policy ran on the CSI FANUC ER-4iA cell through the organizers' client (`client/`), using
  the cell's table camera, wrist camera, joint and TCP feedback. Placements for each trial are in
  `client/submission/placement.yaml`.
- **Digital twin:** an Isaac Sim model of the cell (arm, gripper, calibrated cameras, identical yellow bins) was used
  to generate synthetic training data and to test the full closed loop at the real arm's speed before cell time.
- **Learning from the cell:** real images from earlier cell runs were auto-labelled and hand-corrected, then used to
  fine-tune the detector.

## Data

- **Synthetic:** 2,800 4K frames from Isaac Sim Replicator with randomized layouts, lighting and materials, converted
  to YOLO segmentation labels (`mm/datagen/`).
- **Real:** 182 table-camera images from the 27 Oct cell runs, auto-labelled then hand-corrected (not included here
  for size; the fine-tuned weights are).
- **Run logs:** every decision the policy made on the cell: one JSON per action with the detections, bins, chosen
  pick and plan, plus the table and wrist images it saw (`runs/real/`, `runs/real_v5_baseline/`).

## Results against the baseline

**Real cell.** The two runs were on different days and testbeds with different layouts, so this is not a matched
comparison. A "release" is the robot opening the gripper over the chosen bin, from its own logs.

| | v5 baseline (28 Oct, 8 trials) | v6 (30 Oct, trials t04-t09, 6 trials) |
|---|---|---|
| Pick attempts (cube / screw / AAA) | 17 (3 / 0 / 14) | 17 (5 / 4 / 8) |
| Releases over a bin (cube / screw / AAA) | 12 (2 / 0 / 10) | 13 (4 / 4 / 5) |
| Screws attempted | 0 | 4, all 4 released |
| Trials with every target cleared | 0 | 2 (t04, t05; checked on the table-camera video) |
| Trial ended by a controller fault | 4 | 5 |

v5 attempted almost no cubes and no screws: its pose-fit gate refused them. v6's outline grasp removed that gate.

**Same images, both versions (offline replay of the 28 Oct cell images).**

| | Detected | v5 would pick | v6 would pick |
|---|---|---|---|
| Cubes | 705 | 2 | 676 |
| Screws | 165 | 0 | 131 |
| AAA cells | 39 | 34 | 26 |
| Decision time per image | | 8.7 s | 2.9 s |

v6 picks fewer AAA cells here because its bin assignment leaves some cells for a bin it cannot reach in those
layouts; the AAA pose itself is identical in all 39.

**Digital twin, 5 layouts at the real arm speed (50 mm/s).**

| | v5 | v6 |
|---|---|---|
| Layouts fully sorted | 3 / 5 | 5 / 5 |
| Targets sorted | 20 / 22 | 22 / 22 |
| Arm time per pick (median) | 69.0 s | 56.5 s |
| Wrist check verdicts correct | n/a | 22 / 22 |

Raw numbers: `runs/sim/*_summary.json`, `runs/offline_replay/`, `runs/compare_real.json`, `docs/setup_notes.md`.

## Limitations

- **Controller faults** ended 5 of the 6 v6 trials (t05 was ended by the operator after the robot had stopped). The
  cause has not been established from our side.
- **Part size.** The open fingertip gap is 16 mm. In an earlier 30 Oct run with 13-15 mm cubes, a finger landed on the
  cube's top during the descent. With 10 mm cubes this did not happen.
- **Wrist camera calibration** on the real cell is about 2 cm off; when the wrist check is unclear the table camera
  decides.
- **Reach.** Targets whose bin is outside the arm's reach are refused rather than attempted.
- **Retries.** In t06 an AAA cell had to be picked a second time after its first placement.
- **Waiting.** The scene-ready check waits when a detection flickers near the confidence threshold, which costs time
  before the first pick.
- **Speed.** About a minute per pick at the cell's arm speed.
- **Perception scope.** Hybrid perception: a trained detector plus calibrated geometry. It is not independent
  vision-language pose estimation and not arbitrary 6D pose.

## How it generalizes

- The grasp comes from the part's outline, so a new part type needs only segmentation labels, not a CAD model or a
  pose template.
- Bins are found every step, so moving bins or adding parts mid-run needs no reprogramming.
- The digital twin produces labelled training data and a closed-loop test bench before any cell time, and the same
  policy code runs in the twin and on the cell.
- The checks (wrist verification, foreign-object pause, stop when done) are independent of the part types.

## Repository layout

| Path | Contents |
|---|---|
| `mm/` | Policy code: perception, planning, real-cell brain, data generation, evaluation |
| `client/` | Organizers' cell client with our `policy.py` and submission files (no request credentials) |
| `models/yolo_seg_v5_real_1080/` | Detector weights v6 runs (Git LFS) |
| `runs/real/` | v6 real-cell logs, 30 Oct (`20261030_144522`: t00-t02, `20261030_153410`: t04-t09) |
| `runs/real_v5_baseline/` | v5 real-cell logs, 28 Oct |
| `runs/sim/`, `runs/offline_replay/` | Twin and offline-replay results |
| `media/` | Video clips and stills |
| `docs/` | Setup notes (full engineering log), real-run reports, the starter's original README |
| `tests/` | Host unit tests |
| top level | The organizers' starter (scene, camera, FK/IK, gripper geometry); see `docs/STARTER_README.md` |

## Running it

- Host tests: `python -m unittest discover -s tests` from the repo root, in the host virtual environment
  (`requirements.txt`): 114 tests, 113 pass and 1 is skipped (run on this repository before publishing). The
  starter's own tests: `python -m unittest test_sort_planning test_cube_perception test_cube_pose
  test_runtime_config` (33 tests; on Windows one `test_runtime_config` test errors on `Path.home()`).
- Client protocol check without the cell: in `client/`, `uv run run.py --mock`.
- On the cell: in `client/`, `uv run run.py` (needs an approved request from the organizers; the request file is
  private and not in this repository). Set `MM_ROOT` to this repository's root and `YOLO_PY` to a Python with
  `ultralytics`.

On our PC the code ran as `v6/` next to the client folder `PhysicalAI-Hackathon-2026-v6/`; some tools and tests
still default to that sibling folder for the cell calibration (`--cameras` / `MM_CAMERAS_JSON` override it; the same
file is `client/calibration/cameras.json`). The detector needs `ultralytics` (`requirements-yolo.txt`); the digital
twin needs Isaac Sim 6.2.
