# Setup notes — local GPU machine (B0 baseline)

Machine Minds, step B0. Exact paths, versions and measured run times for the
Windows development PC. The starter was validated on Linux with Isaac Sim 6.0;
everything below was run on Windows 11 with Isaac Sim 6.2.

## Machine

| Item | Value |
| --- | --- |
| OS | Windows 11 Education, build 26200 |
| CPU / RAM | Intel Core Ultra 9 285, 63 GB |
| GPU | NVIDIA GeForce RTX 5060, 8 GB VRAM, driver 610.60 |
| Conda | Anaconda 2026.07-1 (conda 26.5.3), `%LOCALAPPDATA%\anaconda3` |

## Interpreters

The three roles from `AGENTS.md` are kept separate. Environments live under
`C:\isim\` because Windows long paths are disabled (no admin rights) and Isaac
Sim's extension cache exceeds 260 characters inside the default conda folder.

| Role | Interpreter | Key versions |
| --- | --- | --- |
| Isaac (`$ISAAC/python.sh` in the guide) | `C:\isim\env\python.exe` | Python 3.12.15, isaacsim 6.2.0.0 (pip, `isaacsim[all,extscache]`), numpy 2.3.1 |
| YOLO | `C:\isim\yolo\python.exe` | Python 3.12.15, torch 2.14.1+cu130, ultralytics 8.4.183, `requirements-yolo.txt` |
| Host (`.venv/bin/python` in the guide) | `.venv\Scripts\python.exe` | Python 3.12.15, numpy 2.5.4, scipy 1.18.1, opencv 5.0.0, pyyaml 6.0.3 |

Set `OMNI_KIT_ACCEPT_EULA=YES` for Isaac runs (the NVIDIA Omniverse licence was
accepted on this machine). The FANUC ER-4iA USD loads from NVIDIA's asset
server on first use (`.../Assets/Isaac/6.2/Isaac/Robots/Fanuc/er4ia/er4ia.usd`).

## Windows differences from SETUP.md

- `bootstrap.sh` is bash-only. Equivalent, from the repo root:
  `C:\isim\env\python.exe -m venv .venv`, then
  `.venv\Scripts\python.exe -m pip install -r requirements.txt`,
  `.venv\Scripts\python.exe make_textures.py`, and the unittest line below.
- `run_sort_pipeline.sh` is bash-only; the stages below run it by hand.
- Host tests: 58/59 pass. `test_runtime_config.test_shared_hub_cache_is_respected`
  fails on Windows only: it clears the environment, so `Path.home()` cannot
  find `USERPROFILE`. Not a code fault.
- `build_scene.py --shoot` raises after writing its images under Isaac 6.2
  (`arm.get_joint_positions()` returns None after the Replicator capture), in
  the unmodified starter too. The images are usable; the metadata JSON is not
  written.
- `assets/mat_print.png` is not in the starter (warning only; plain mat).
- `aux` is a reserved device name on Windows; datagen writes `aux_cams/`.

## Known twin issue

Narrow-end-up caps sink 2 mm through the 2 mm mat slab (the flange's overlap is
resolved downwards onto the table box), in the unmodified starter scene too.
`mm/datagen/gen_yolo_data.py` adds an invisible 20 mm support slab under the
work surface. `cell.py` is not yet fixed; this matters for cap picks (B3/B5).

## B0 baseline run (Cosmos replaced by YOLO)

Cosmos3-Nano is not installed. The baseline keeps the starter's planner and
executor unchanged and swaps only perception: `perceive_yolo.py` writes
`detections_yolo.json`, and `mm/perception/to_cosmos_schema.py` rewrites it in
the schema `plan_sort.py` reads. Starter scene, no layout override.

```
set OMNI_KIT_ACCEPT_EULA=YES
C:\isim\env\python.exe capture_scene.py --out runs\b0_baseline
C:\isim\yolo\python.exe mm\perception\perceive_yolo.py --run runs\b0_baseline --out detections_yolo.json
.venv\Scripts\python.exe -m mm.perception.to_cosmos_schema --run runs\b0_baseline
.venv\Scripts\python.exe plan_sort.py --run runs\b0_baseline
C:\isim\env\python.exe sort_scene.py --run runs\b0_baseline --physics-only
```

| Stage | Interpreter | Time |
| --- | --- | --- |
| 4K table capture | Isaac | 22 s (incl. Kit start-up) |
| YOLO + pose refinement | YOLO | 11 s (10 objects) |
| Schema adapter | Host | 1 s |
| Plan (`plan_sort.py`) | Host | 5 s |
| Execute, physics only | Isaac | 23 s (1656 frames) |
| Execute, rendered 1080p | Isaac | 4.6 min (1656 frames) |
| Encode video (`encode_video.py`) | Host | 41 s (55 s of 1080p30, 62 MB) |

Result, both physics-only and rendered: `execution_report.json` —
`task_complete: true`, `verified_pickups: 6`, `verified_placements: 6`,
`abort_reason: null`. Physics-only report kept as
`execution_report_physics_only.json`; video in `runs/b0_baseline/sort_table.mp4`.

## B1 scene and bins

Bins come from `configs/bins.yaml` (one per target class in
`configs/classes.yaml`; **positions are placeholders** until the real cell is
measured). They are switched on only through a layout (`MM_SCENE_JSON`), so the
starter scene above stays bin-free. `randomize.py` copies the bins into every
layout and keeps objects 15 mm clear of them (`--no-bins` reproduces the Part A
data). Check in Isaac:

```
C:\isim\env\python.exe mm\datagen\check_layouts.py --seeds 20 --out runs\b1_layouts
```

20/20 layouts load with 3 bins (1.1 min); cubes and caps move at most 0.36 mm
under physics. Lying AA cells and tubes roll up to 19 mm in 1 s of free
physics, so planning must use the captured (kinematic) poses. The Part A model
detects all objects in a scene with bins and makes no detections on the bins.

## B3 grasp planning

```
C:\isim\yolo\python.exe mm\perception\perceive_yolo.py --run runs\<id>
.venv\Scripts\python.exe -m mm.planning.plan_bins --run runs\<id>
```

`mm/planning/grasp.py` checks width (on the 16 mm hooked-tip gap; AA cells use
the 14.5 mm scallop fit), finger footprints vs neighbours and bins (+3 mm) and
IK at hover, grasp, lift, above-bin and release; cubes try both face pairs.
Caps are never pickable with this gripper (plug 20.8–23.6 mm > 16 mm tip
gap), so on 2026-10-23 **screws (M6 x 30 countersunk, ISO 10642) replaced caps**
as the third target; caps stay as distractors (class id 1, id 4 = screw). The
starter's v0.5 `cell.add_screws` is used again, now authored in the measured
resting pose (5.3 deg tilt, head face 5.67 mm up). Jaws close across the 6 mm
shank middle.

v2 data and model (`data/synth_v2`, `data/yolo_v2`, `models/yolo_seg_v2`;
v1 kept): datagen 54 min, training 100 epochs 3.0 h, test mask mAP50 0.995 /
mAP50-95 0.843. Perception eval, test split: recall 99.6–100% every class,
0 confusions; screws: centroid max 0.65 mm, yaw p95 1.1 deg, polarity 100%,
97% eligible (the rest refused by the 0.85 fit gate, kept deliberately).

On 5 fresh bin scenes (`runs/b3v2_9200xx`): 36/37 targets (all 16 screws)
planned to the right bin within 1 mm, 0 wrong picks, < 130 ms per scene; one
cube correctly refused (a 140 mm tube lies along it). `check_layouts.py`:
20/20 layouts with screws and bins OK. `sort_plan.json` has
`source: MachineMinds-plan_bins`; `sort_scene.py` accepts it only after B5.

## B4 job ordering

`mm/planning/order.py`; `plan_bins.py --order {fixed,greedy,greedy_2opt}`
(default greedy_2opt). Every plan also logs all three strategies' predicted
travel (`ordering`) for B9. Cost = TCP path for chained execution
(home → object → its bin → next object … → home, straight lines between
hover points 0.13 m up, as `sort_scene.segments`); B5 must make `sort_scene.py`
chain picks instead of returning home after each, or order changes nothing.

Measured, 100 random 6–10-pick cases: greedy_2opt 1.4% shorter than greedy on
average (max 3.2%, better in 97/100), never longer (tested); greedy alone ≈
fixed (0.1%), sometimes worse. Planning < 2 ms. Reason: 71% of travel does not
depend on order (0.52 m of descents per pick plus each object → bin leg); on
the order-dependent 29%, 2-opt saves ~5%. 5 real scenes: 9.8–11.3 m per scene,
greedy_2opt up to 1.7% shorter than fixed, once 1.2% longer (greedy start).

## B5 sim execution

`sort_scene.py` hands `plan_bins` plans to `mm/control/sim_backend.py` (the
starter's cube-stack path is unchanged); `mm/control/executor.py` holds the
backend-independent parts (plan checks, chained segments, path/time logs,
in-bin check) for the real-arm backend later. Picks run back to back in plan
order (no return home between picks), long moves capped at 0.35 m/s, released
at the bin's `release_z`. The scene is rebuilt from the run's `layout.json`.
`execution_report.json` logs per pick and per segment: sim time, commanded and
measured TCP path, jaw gap, grasp tip error, bin placement, other objects moved.
Pickup is still scored with the starter's sim-pose lift check (B6 replaces it).

One trial, all stages (Windows, all interpreters from env or defaults):

```
.venv\Scripts\python.exe -m mm.eval.run_trial --seed 930000 --order greedy_2opt [--render]
```

10 random scenes (`runs/trial_93000x_greedy_2opt`): **10/10 complete, 78/78
planned picks placed in the right bin** (26 cubes, 21 AA cells, 31 screws);
2 of 80 targets skipped at planning (finger collision); 0 table objects
disturbed. Mean cycle 67 s sim (8.6 s per pick), TCP 9.18 m measured vs 9.20 m
predicted. Stage times: capture 18 s, perceive 18 s, plan 1 s, execute 23 s
(physics only). Screws marked `grip_validated` in sim.

**AA cell grasp is marginal.** Physics-only runs are repeatable (the same plan
3x: 10/10 each), but the rendered run of the same scene (`runs/demo_930005`,
video `sort_table.mp4`) lost both AA cells: one stayed wedged in the finger
scallop when the jaws opened and was carried off, one was not seated (jaw gap
6 mm instead of ~1.5 mm) and dropped on the way. Cubes and screws were
unaffected. With 0.75 mm clearance per side the AA grasp tips on small
perturbations, so physics-only success overstates it; a real arm will vary
more. Needs a grasp/release check (B6) and AA grasp tuning.

**AAA instead of AA (test, 2026-10-23; v3 made it permanent and removed the
`MM_CELL` switch).** The test swapped the cell for an AAA (10.5 x 44.5 mm, 11.5 g) under the same `aa_cell`
class, bin and the unchanged v2 YOLO model; the finger scallop stays AA-sized.
Clearance per side in the 16 mm tip gap: 2.75 mm (AA 0.75).

| Cell | Physics-only (10 scenes) | Rendered (3 scenes) |
| --- | --- | --- |
| AA  | 21/21 placed | 4/7 placed (stuck on release, dropped, flung off table) |
| AAA | 18/18 placed | 7/7 placed |

Everything else placed in every run. YOLO, never trained on AAA, detects them
as `aa_cell` (confidence mean 0.86, min 0.70; AA 0.90) and the pose fit puts
them within 0.64 mm (median 0.24). 3 of 21 AAA cells were not attempted
(finger collision with a neighbour). Runs: `runs/cell_AAA_*`, `runs/cell_AA_render_*`;
side-by-side video `runs/compare_AA_vs_AAA_930007.mp4`. Not done: YOLO
retraining on AAA (confidence would recover), a test on the real gripper.

## v3: AAA cells, one bin design, bin class (2026-10-24)

AAA cells (class `aaa_cell`) replace AA everywhere. All three bins use the
starter's large vial-bin design (90 x 90 x 65 mm, 65 x 65 mm pocket,
`configs/bins.yaml: design`); the printable STL comes from the same function
the twin uses: `.venv\Scripts\python.exe -m mm.cad.sort_bin` →
`cad/sort_bin_90x90x65.stl`. Classes: targets cube, aaa_cell, screw;
distractors cap, tube; `bin` (id 5, never picked). Bin floor collider reaches
30 mm into the table (a cube once tunnelled through the 15 mm floor).

Model `models/yolo_seg_v3` (yolo11s-seg, imgsz 1600, v3 ROI): training stopped
at epoch 24 of 100 (converged); `best.pt` = epoch 11. Test mask mAP50 0.995,
mAP50-95 0.862, P 0.998, R 0.996.

Perception eval (`eval_perception.py`, test 300 frames / hard 200 frames):

| Class | Recall test / hard | Centroid p95 / max (mm), test | Yaw p95 (deg), test | Eligible |
| --- | --- | --- | --- | --- |
| cube | 99.9 / 99.6% | 0.40 / 0.76 | 7.7 (mod 90) | 99% |
| aaa_cell | 99.6 / 99.6% | 0.60 / 2.36 | 1.3, polarity 99.5% | 98% |
| screw | 99.8 / 99.6% | 0.43 / 0.75 | 1.3, polarity 100% | 80% |
| bin | 99.3 / 99.6% | 0.67 / 8.07 | 0.48 | 99% (fit ≥ 0.80) |
| cap, tube | 99.3–100% | distractors | | |

Confusion 0.05% (test) / 0.12% (hard); 10 / 11 spurious detections. Bin pose:
1,400 bins scored, 3 off by > 2 mm (8 mm/12 deg and 15 mm/22 deg, fits 0.86 and
0.80, cause not yet looked at); 50 more fall below the 0.80 fit and B8 keeps the
previous pose. Screws: 20% refused by the 0.85 fit gate (kept deliberately).
Known, deferred: cube yaw p95 7.7 deg (max 23 deg on hard); tilted (dropped)
objects still marked eligible (53 cubes, 59 AAA, 39 screws on test).

## B6–B8 closed loop, v3 checks (2026-10-24, physics only)

`run_trial --loop b6|b7|b8`; runs in `runs/checks/`, demos (rendered) in
`runs/video_b5..b8/demo.mp4`. 31 runs, 190 grasp attempts.

| Check | Runs | Targets sorted | Notes |
| --- | --- | --- | --- |
| B6, 3–5 mm injected | 6 | 32 / 52 | 1 abort (IK, see below) |
| B6, 6–10 mm injected | 6 | 39 / 47 | 2 complete |
| B7, +3 added, 1 nudge each | 5 | 52 / 57 | 15 objects added, 4 nudged; 2 complete |
| B8, random bins | 8 | 44 / 67 | 1 abort; 0 wrong-bin placements |

- **Grasp check (B6):** 21 real misses, 17 caught (81%, target 95%), 0 false
  alarms in 190 checks. All 4 uncaught misses knocked the object 0.2–0.5 m
  away: the spot was empty, so the check said "taken". It checks only that the
  object left the spot, not that it is in the jaws. The next capture finds the
  object again if it stays on the table.
- **Grasps without injection:** cubes 57/58 lifted, screws 51/51, **AAA cells
  43/54 (80%)**. Failed AAA grasps close fully (jaw gap ~0) and the cell rolls
  or pops away. Likely cause: fingertips at 2.5 mm meet the 10.5 mm cell below
  its centre (5.25 mm), and the AA-sized scallop sits higher (9.5 mm). A lying
  cylinder has no rolling resistance in PhysX. Not yet verified in a dedicated
  test.
- **B5 open loop (`video_b5`): 3/5 placed.** All three AAA cells were already
  off their captured spots when the first pick ran (rolled 8–200 mm), so the
  plan-once path missed them. The closed loop re-detects every cycle.
- **Aborts:** 2 runs stopped with `IK failed: approach`, on the move from the
  previous drop to the next object's hover. The pick passed planning (IK is
  checked only at waypoints, not along the straight line), and one failure
  ends the run.
- **B7:** every add/nudge was re-detected and tracked; 13 of the 15 added
  objects were sorted (2 refused by gates).
- **B8:** bins found in 170 of 174 per-cycle detections; error median 0.2 mm,
  max 1.78 mm, yaw max 1.9 deg.
- Objects left on the table are mostly gate refusals (screw fit < 0.85,
  finger collision, IK), kept as designed. 4 objects dropped in transit.

### IK fix (2026-10-24)

Planning checks IK only at waypoints, so a straight move between them could
fail mid-run and end the run. Now `closed_loop` solves each attempt's whole
path frame by frame before the arm moves (`SimCell.check_path`). If the path
fails, it tries in order: the wrist turned the other way round, then via the
home point, for the pick and the carry separately (`route_options`,
`choose_route`). With no route, the object is skipped and flagged
(`Tracker.give_up`) after a 1 s pause the video shows. The report's
`ik` field logs every reroute and skip with the failed segment and frame.
`move()` now raises before moving rather than part-way. The grasp check and
the AAA grasp are unchanged.

Rerun of the same 25 check seeds (`runs/checks_ikfix/`, physics only):

| Check | Sorted before → after | Aborts before → after | IK |
| --- | --- | --- | --- |
| B6 3–5 mm (6) | 32/52 → 37/52 | 1 → 0 | 1 skipped |
| B6 6–10 mm (6) | 39/47 → 36/47 | 0 → 0 | – |
| B7 (5) | 52/57 → 52/57 | 0 → 0 | – |
| B8 (8) | 44/67 → 44/67 | 1 → 0 | 1 rerouted (via home) |

Grasp check over all 25: 14 of 19 misses caught, 0 false alarms. B8 bins:
163 of 168 per-cycle detections, median 0.19 mm, max 1.86 mm, yaw max 2.3 deg.
The skipped AAA cell (seed 950005) failed every route near the end of the
approach, even the one starting from home. Planning's single IK solve at the
hover point succeeds, but tracking the straight line there frame by frame does
not. Cause not investigated.

## v4: three identical tangerine-yellow bins, robot-chosen assignment (2026-10-25)

This folder (`Desktop/Hackathon/v4`) is a copy of v3 (`fanuc_er4ia_sim`, without `.venv`; host Python is
v3's `.venv`). Its client is `../PhysicalAI-Hackathon-2026-v4`, a copy of the client without `.venv` and
without `.request.json`. The original client keeps the v3 submission (request q481f22) and still points at
v3.

- **Bins.** `configs/bins.yaml` has bin_1..3, all colour tangerine, material `bin_tangerine`: #FFC72C
  sRGB, linear (1.000, 0.571, 0.025). Roughness is `C.BIN_ROUGHNESS` (0.45). A layout's `bin_roughness`
  (0.35–0.55, its own random stream so object layouts are unchanged) sets the gloss per scene, and datagen
  also sets it per frame (`cell.set_roughness`).
- **Identity.**
  - `mm/perception/bins.py`: `first_bins()` names the bins by position at the first capture where all
    three are found with fit ≥ 0.80. With fewer than three, the closed loop captures again and the policy
    holds and captures again; neither guesses.
  - `track_bins()` then matches each name to the nearest bin within 60 mm, so names never swap. A bin
    not seen keeps its last pose.
- **Assignment.** `mm/planning/bin_assign.py` plans all 6 class → bin assignments with the real planner
  (grasp, finger, reach and IK at that bin). It keeps the one with the most plannable picks, then the
  shortest predicted travel, and logs every assignment's travel and the saving against the worst.
  - The choice is made once per rollout and kept (`closed_loop.py`, `mm/real/brain.py`).
  - `plan_bins.build_plan` runs it itself when given physical bins (open loop).
  - The executor finds a pick's bin by `place_xy` (`executor.bin_at`).
- **Success rule** (`executor.sort_ok`): each bin holds one type at most, every target is in a bin, and
  the distractor (cap or tube) is in no bin. The closed-loop report has `bin_contents` and `sort_ok`;
  `mm/real/sim_eval.py` scores the same way. A distractor's movement is reported, not judged, because the
  twin's tube can roll by itself.
- `demo_layout` "eval" scenes now need IK from every target to every bin, so any assignment is feasible.

**YOLO on tangerine bins at 1080p** (`eval_perception --scale 0.5`, then `mm.eval.bin_check`):
- **Sets.** `data/synth_v4` test/hard use v3's seeds, so only the bins differ. `data/synth_v4_stress`
  (`--bins stress`, hard seeds, arm in view 70%) puts one bin 5–40 mm from the yellow edge tape (only the
  back corners, where the tape enters the crop) and one 0.16–0.24 m from the base axis. Not reachable: a
  detection test only. Within reach, no bin can be near the tape (at least 140 mm).
- **v3_1080 model:** test/hard bin recall 99.8%, p95 ≤ 0.78 mm, but stress recall 96.2% (91.2% near the
  tape). Under 99%, so retrained.
- **`models/yolo_seg_v4_1080`:** v3_1080 fine-tuned 15 epochs (best: epoch 3) on 1,000 reachable-bin
  frames + 600 stress frames (+250 val) at 1080p, with randomized light and gloss; inference at 1600. Test
  mask mAP50 0.994, mAP50-95 0.776.
  - Bin recall: test 99.89%, hard 99.65%, stress 99.81% (tape 100%, base/arm 99.75%).
  - Bin position p95 ≤ 1.21 mm everywhere. Max over 2 mm: 1 bin each in test and hard, 12 in stress (bins
    half behind the base column).
  - Screws are now 96% pickable (89% with v3_1080).

**Sim test of the v4 `policy.py`** (`mm.real.run_sim_suite --out runs/real_sim_v4`, seeds 1–5): **5/5
layouts succeed** under the v4 rule, every target binned and one type per bin. Travel saved by the chosen
assignment vs the worst: 3.7, 2.5, 1.1, 3.0 and 2.0%. On layout 3 the tube rolled about 100 mm by itself
(the twin, as in v3), but it is not in a bin. Layout 5's screw, refused by v3's model, is picked now.
`uv run run.py --mock` (v4 client): **"✓ Ready for the real robot"**, 54 actions, 0 rejected. With no
bins in the mock images, the policy parks and then holds while capturing again.

**v4 demo video** (`runs/demo7v4_render_s4/demo.mp4`, `demo_layout --seed 4`, run with `run_trial
--layout/--loop-json --render`, checked with `mm.eval.demo_check`).
- **Closed-loop perception now matches the cell.** `perceive_server` loads `MM_YOLO_RUN` or `DEFAULT_RUN`
  (now `yolo_seg_v4_1080`) and resizes 4K captures to the card's `input_resolution` (1920×1080). Outlines
  are returned in capture pixels.
- **The video** adds the robot's assignment as a fixed line under the title, plus a label on each bin at
  its perceived position (`make_video.bin_labels`). The result card states the v4 rule.
- **Seed 4 passed every check**, physics-only and rendered alike:
  - no gate refusals, and all 3 bins found at the first capture;
  - assignment bin_1 = screws, bin_2 = cubes, bin_3 = AAA cells (6.13 m planned travel vs 6.64 m worst);
  - deliberate miss on attempt 3 (AAA cell, fingers 33 mm off), caught and placed on the retry;
  - Screw0 nudged 183 mm and sorted;
  - contents bin_1 1 screw, bin_2 2 cubes, bin_3 2 AAA cells; cap not moved.
- **The tube rolled 129 mm** sideways by itself between frames 100 and 280 while the gripper was at a bin
  (the twin's cylinder roll). It never entered a bin.
- **Seeds 3 and 5 failed physics-only.** Seed 3: the nudge found no planned spot clear. Seed 5: after its
  nudge, the AAA cell was refused as `ik_unreachable` at a spot `demo_layout` had passed; not investigated.

## Real arm: organizers' client (2026-10-25)

The client is `../PhysicalAI-Hackathon-2026`, downloaded as a zip because git is not installed here.
uv is at `C:\isim\uvtool\Scripts\uv.exe`. uv's own Python download fails here (a link it cannot make without
admin), so the client venv was created with
`UV_PYTHON_DOWNLOADS=never uv sync --python C:\isim\env\python.exe`.
Run it with `C:\isim\uvtool\Scripts\uv.exe run run.py --mock` from the client folder.

- **`policy.py` (client):** `ACTION = "tcp"`, `CAMERAS = ["table", "wrist"]`, `IMAGE_SIZE = (1920, 1080)`,
  `DT = 0.2`. It starts `mm.real.brain_server` with `C:\isim\yolo\python.exe` (torch stays out of the client
  env) and forwards each observation to it over JSON lines.
- **`mm/real/brain.py`, one decision per act():**
  - The first act parks.
  - Each act then perceives with the 1080p model, finds the bins (fixed `bins.yaml` poses until each one
    is seen), and judges the last pick from the new image: the object still within 25 mm of its spot is a
    miss, and is retried up to 3 times then skipped.
  - It plans, tracks and orders, then solves the whole path with the twin's IK (straight, wrist turned the
    other way, via home; no route means skip).
  - It returns UF0 rows: hover, descend, close, lift, carry, release, retreat, park. Rows are capped at
    0.15 m/s, R is kept continuous, and a path over 100 rows continues on the next act.
  - With nothing left, it holds still.
  - Every act is logged to `runs/real/<time>/rollout_NN/act_NNN.json` with the table image.
- **Frames (`mm/real/frames.py`):** z_UF0 = z_base − 330 mm, fingers down is w, p, r = −180, 0, R. The jaws
  close along (sin R, −cos R), derived from the FANUC faceplate convention, not measured.
  - Check it on the cell first: at the cell's home joints (`mock_cell.HOME`), the twin's FK gives exactly
    our sort home, UF0 (300, 0, −90) mm, fingers down.
  - `CellCamera` serves `cameras.json` through the perception code's camera interface. The real table
    calibration is within 0.5 px (4K) of the twin camera.
- **Park:** (0.18, 0.40, 0.35) m base = UF0 (180, 400, 20) mm, the arm more than 300 px clear of the work area
  and bins in the 1080p image (`mm.real.park`). The capture pose is outside the cell's TCP box (x 50 mm).

**YOLO at 1080p.** `models/yolo_seg_v3_1080` is v3 `best.pt` fine-tuned 25 epochs (AdamW, lr0 5e-4) on the
v3 crops halved (`mm.datagen.downscale_yolo`, `configs/dataset_1080.yaml`). The best epoch was an early one,
and later epochs dropped; test mask mAP50 0.995, mAP50-95 0.798, recall 0.995.
`eval_perception --scale 0.5` scores the 4K test/hard frames at 1080p:

| Test split @1080p | Recall cube / AAA / screw | Screws eligible | Centroid p95 / max (mm) | Spurious |
| --- | --- | --- | --- | --- |
| v3 (4K-trained), imgsz 1600 | 99.9 / 99.6 / 99.6% | 88% | ≤ 0.71 / 2.94 | 9 |
| v3_1080, imgsz 1280 | 99.7 / 99.4 / 99.9% | 40% | ≤ 0.82 / 2.74 | 4 |
| **v3_1080, imgsz 1600 (used)** | 100 / 99.5 / 99.9% | 89% | ≤ 0.78 / 1.42 | 4 |

At 1280 the masks are too coarse for the screw fit gate. Its card now sets `inference_imgsz: 1600`, and
`perceive_yolo.model_imgsz` honours it. Cube yaw p95 is about 10 deg at 1080p (13 deg at 4K). Hard split at
1600 not yet run.

**Sim test of `policy.py`** (`mm.real.run_sim_suite`; `mm/real/sim_eval.py` gives the policy rendered 1080p
twin frames in the cell's obs format and plays its UF0 rows with the twin's IK). Layouts are the
`demo_layout` "eval" scenes, seeds 1–5, which are also rollouts 1–5 in `placement.yaml`:

| Layout | Targets | Sorted | Distractor | Result |
| --- | --- | --- | --- | --- |
| 1 | 5 | 5 | cap 0 mm | success |
| 2 | 4 | 4 | cap 0 mm | success |
| 3 | 4 | 4 | tube moved 100 mm | fail |
| 4 | 5 | 5 | cap 0 mm | success |
| 5 | 4 | 3 | cap 0 mm | fail |

21 of 22 targets sorted, no faults, no IK reroutes, every pick took one attempt, about 1.2 min per layout.
- **Layout 3:** the tube rolled on its own, perpendicular to its axis. It had moved 57 mm before the first
  capture, then moved 48 mm more while the arm worked 170 mm away. That is the twin's lying-cylinder roll,
  not arm contact as far as the logs show.
- **Layout 5:** one screw fit 0.80–0.81 in every capture, below the 0.85 gate. It was skipped, and the gate
  is unchanged.

`uv run run.py --mock`: **"✓ Ready for the real robot"**, 3 rollouts, 169 actions, 0 rejected, act() about
1.8 s. The mock has no kinematics, so its joints stay at home and the policy keeps parking there.

### Curated 7-object demo (`runs/demo7_render_s3/demo.mp4`)

`mm/eval/demo_layout.py` builds a scene and its `loop.json`, checking every
rule before Isaac runs:
- 2 cubes, 2 AAA cells, 1 screw, 1 cap, 1 tube, all flat.
- Centres at least 80 mm apart, footprints at least 60 mm from every bin
  wall, targets 0.25–0.40 m from the base and bins 0.25–0.42 m.
- Per jaw direction: fingers clear of all objects and bins, plus `grasp.Reach`
  IK at hover, grasp, lift, above the bin and release.
- One deliberate miss on the 3rd attempt, first tries only: the fingers are
  shifted across the jaws to just past the object (`closed_loop.miss_offset`).
- One nudge, to destinations precomputed under the same rules and re-checked
  live.

Run it with
`run_trial --layout <dir>/layout.json --loop-json <dir>/loop.json [--render]`.

Layout seed 3, first try:
- **Physics-only and rendered runs were identical.** 5/5 sorted in 6 attempts,
  with no target refused at any capture and all 3 bins found in all 7 captures.
- **Deliberate miss:** attempt 3, an AAA cell with the fingers 33 mm off. The
  check said "object still there"; the retry placed it.
- **Nudge:** SortCube1 moved 56 mm, re-detected and sorted.
- **Distractors:** the arm never touched the cap or the tube. The tube rolled
  86 mm sideways when the scene's snapshot was released to physics, before the
  first capture. The rules still hold at its new spot. This is the same
  lying-cylinder roll as the AAA cells in B5; cause not investigated.

### Combined demo (`runs/demo_all_954018/demo.mp4`)

`run_trial --loop all` runs B6 + B7 + B8 together: random bins found by
perception, 6–10 mm offsets injected on half the grasps, 3 objects added and
1 nudged. Seed 954018 (rendered): 7 of 10 targets sorted (all three classes;
caps and tubes on the table). 6 misses, all caught; 5 of them injected, and
those objects were all placed on a later try. The 3 added objects and the
nudged screw (dropped once on the way, re-detected) were sorted. Bins were
found in 37 of 45 captures; when one is missed, the previous pose is kept.
3 targets were refused at planning. No IK reroute or skip happened in this
run.

Rendered closed-loop runs reproduce themselves (954000 twice, identical) but
not the physics-only run of the same seed after the first missed grasp. So
demo seeds were chosen from rendered runs (`runs/demo_all_*`). Physics-only
candidates are in `runs/demo_candidates/`.

### Demo videos (re-rendered 2026-10-24: `runs/video2_b5..b8/demo.mp4`)

The first B6–B8 videos (`runs/video_b6..b8`) were ~92% duplicate frames.
The closed loop switches Replicator's capture-on-play off, so the video
camera refreshed only when a 4K capture or wrist grab was taken. Now
`sim_backend` renders every saved frame with an explicit orchestrator step
from its own demo camera (`VIDEO_CAM`: three-quarter view, whole arm in frame)
and the wrist camera (`run/wrist_frames`). A run fails if more than 5% of
consecutive frames are identical (`execution_report.json: render_check`;
`run_trial` exits 2). `make_video` checks the same thing again on the files.
It puts the title, caption and footer bands outside the camera image, adds
the wrist view as a picture-in-picture, and builds captions and running totals
per segment ("placed" only after the release). A 3 s result card ends each
video. Re-render: `run_trial --render` (~6–9 min per run), then
`make_video --run ... --title ...`. Rendering does not change outcomes: the
B5 and B7 seeds reproduced their earlier outcomes exactly. B6 and B8 differ
from the earlier renders of the same seeds; the cause has not been checked.

## Machine Minds additions (Part A), for reference

| Job | Command | Time here |
| --- | --- | --- |
| Synthetic frames, 2,800 at 4K | `C:\isim\env\python.exe mm\datagen\gen_yolo_data.py --split <train/val/test/hard>` | 50 min (~1 s/frame) |
| Masks to YOLO-seg | `.venv\Scripts\python.exe -m mm.datagen.masks_to_yolo` | 31 s |
| Train yolo11s-seg, 100 epochs | `C:\isim\yolo\python.exe mm\perception\train_yolo.py --batch 4` | 3.0 h |
| Perception eval, 300 frames | `C:\isim\yolo\python.exe mm\eval\eval_perception.py --split test` | ~7 min |

Training uses batch 4, not the guide's 8: at imgsz 1280, batch 8 needs 10.8 GB
and spills into shared memory on this 8 GB GPU (27 min/epoch instead of ~2).

## v5: real-arm timing, arrival checks, a twin that holds still (2026-10-27)

v5 is a copy of v4 (`../v4` stays the known baseline); its client is `../PhysicalAI-Hackathon-2026-v5`
(MM_ROOT default `../v5`; venv rebuilt with `uv sync --offline --python C:\isim\env\python.exe`).

- **Why:** on the 27 Oct real run the arm moved ~50 mm/s whatever its rows asked (146 mm in 2.8 s, 50 mm in
  1.2 s; `runs/real/20261027_165715`) and stopped where it was when the rows ran out. Rows timed for 0.15 m/s
  left every park 6-8 cm short (six parks per trial) and closed the gripper before the arm reached the AAA cell.
- **`mm/real/brain.py`:**
  - `TCP_SPEED` 0.030 m/s mean (eased rows peak at 1.5x: 45 mm/s, under the arm's ~50).
  - A pick is three moves: hover + descend + 0.6 s pause | close, lift, carry, lower + pause | release,
    retreat, park. Each is sent only when `tcp_pose` is within 3 mm and 3 deg of the end of the one before.
  - Not there yet: "wait", which finishes the same move from where the arm is, with no new capture.
    After 4 waits: abort at the grasp (lift, park, gripper never closed). Over the bin, release if within
    15 mm in xy, else keep finishing; it never opens outside the bin.
  - Park works the same way, so the next capture waits until the arm is parked.
- **`mm/real/sim_eval.py`:**
  - `--arm-speed` plays rows the way the cell did.
  - `--track-moves` logs every object that moves and how far it moved on the table before being lifted.
  - `--brain-root` runs another project's brain.
- **Twin (`mm/control/sim_backend.py`):** task objects sleep at rest (`REST_SLEEP_THRESHOLD` 1e-4, set
  through the physics view: PhysX read 0, i.e. never sleep). Without it, lying round objects crept on the
  level mat with the arm still. In layout 3 the tube rolled ~104 mm into an AAA cell, and in other layouts
  AAA cells rolled 43-89 mm after being located. That was the "tube knock": no arm move caused it.
- **Sim, layouts 1-5 (eval scenes, all picks):**

| Run | Arm | Sorted | Picks | Moved on the table |
| --- | --- | --- | --- | --- |
| v4 brain | 50 mm/s | 0/5 layouts | 0 of 22 in a bin after 105 tries | - |
| v5, creeping twin | 50 mm/s | 5/5 | 22/22 | tube 99.8 mm (layout 3) |
| v5, creeping twin | 25 mm/s (layouts 1-2) | 2/2 | 9/9 (one retry), 68 waits | - |
| v5, twin at rest | 50 mm/s | 5/5 | 22/22, no waits | 0.0 mm, every object |

- **Time:** the arm needs ~72 s per target, 288-359 s per placement.yaml rollout. A capture + plan act
  takes ~6.3 s warm, and a wait or continue act 0.2-0.7 s.
- **Real labels:**
  - `mm/datagen/autolabel_real.py` makes `data/real_labels_20261027`, with 4 training rollouts and
    2 held out.
  - `mm/datagen/review_labels.py` makes the previews and `check_first.md`.
  - `mm/datagen/build_real_finetune.py` builds the fine-tune set as ROI crops, with real images repeated
    4x and sim val.
  - `mm/eval/compare_real.py` compares old vs new weights through `perceive_yolo.detect`, using a
    "clearly better" rule fixed in advance.

### Fine-tune on real images: not made the default (2026-10-28)

`yolo_seg_v5_real_1080` is `yolo_seg_v4_1080` fine-tuned on the v4 sim set plus 4 reviewed real rollouts (02, 03, 04, 06; each repeated 4×). It used the same recipe as v4: 15 epochs, imgsz 1280, AdamW, lr0 5e-4.

**Comparison:** `mm.eval.compare_real` on the 2 held-out rollouts (05, 07), run through the robot's own `detect`. Labels outside the ROI were dropped before any result was seen.

At the 0.5 pick gate (true positives / false positives / labels):

| Class | v4 | v5_real |
|---|---|---|
| cube | 2 / 0 / 193 | 193 / 0 / 193 |
| aaa_cell | 0 / 0 / 62 | 5 / 69 / 62 |
| cap | 22 / 0 / 22 | 22 / 0 / 22 |
| bin | 216 / 0 / 216 | 215 / 0 / 216 |

- **Sim test mask mAP50:** 0.995 for both.
- **Verdict:** not "clearly better". The targets' precision falls from 1.0 to 0.74, beyond the 0.05 limit. So `DEFAULT_WEIGHTS` still starts with `yolo_seg_v4_1080`.
- **Where the 69 false AAA detections are:** every one is on the thick silver AA distractor (rollout_05 acts 008–023, every rollout_07 frame). The training rollouts contain no silver AA.
- **The Rayovac AAA cell in rollout_07:** neither model ever detects it.
- **v4 on real frames:** through `detect`, v4 finds almost no cubes or AAA cells (2 of 255 targets at the gate).
- **To try v5_real anyway:** in the v5 client, run `uv run run.py --checkpoint ..\v5\models\yolo_seg_v5_real_1080\weights\best.pt`.

**Decision (owner, 2026-10-28): `yolo_seg_v5_real_1080` is the real-robot default for today's slot, even though it failed the rule above.**
- **Why:** on the held-out real rollouts, v4 finds only 2 of 255 targets at the pick gate. v5_real finds all 193 cubes.
- **Accepted risk:** v5_real calls a thick silver AA cell an `aaa_cell`, so an AA on the table may be picked as an AAA cell. It also still misses the rollout_07 Rayovac AAA cell.
- **Where:** `mm/real/brain.py` `DEFAULT_WEIGHTS` now starts with `yolo_seg_v5_real_1080`. `perceive_yolo.DEFAULT_RUN`, used by the sim perception tools, stays v4.
- **Switch back:** for one run, use `uv run run.py --checkpoint ..\v5\models\yolo_seg_v4_1080\weights\best.pt` in the v5 client. For good, delete the first entry of `DEFAULT_WEIGHTS`.
- **Checks with v5_real as the default (28 Oct):**
  - Sim layout 1 at 50 mm/s (`runs/real_sim_v5real_arm050`): 5/5 sorted, 5/5 picks, 0 misses, 0 waits, 0.0 mm table moves. This matches v4 on the same layout.
  - The mock from the v5 client passes: 216 actions, 0 rejected, act() 5.5 s.
  - 98 host tests pass.
  - `submission/placement.yaml` lists no AA cell. But it is unchanged since 27 Oct, when silver AA cells were on the table anyway, so ask the operator to use AAA cells only.

## v6: outline grasp, retreat instead of park, wrist check, stop and scene checks (2026-10-28)

v6 is a copy of v5's code, configs and client (`../v6`, `../PhysicalAI-Hackathon-2026-v6`); v5 stays the fallback.
v6 has its own empty `models/` and `data/` and reads v5's weights and datasets when it has none (`mm/paths.py`;
`configs/dataset*.yaml` point at `../v5/data`). The client's `MM_ROOT` defaults to `../v6`; the submission is named
`MachineMinds-v6-ClosedLoop`. Not submitted.

Each change has a switch back: `MM_V5_STEPS=2,3,4,5` (any subset) runs those steps as in v5 (`MM_GRASP=pose_fit` also
reverts step 2). With all four set, sim layout 3 at 50 mm/s gives exactly the v5 result (3/4 sorted, 223.4 s of arm time).

- **Step 2, grasp from the outline (`mm/perception/outline_grasp.py`).** Cubes and screws: no 3D fit and no match-score
  gate; YOLO conf >= 0.5 is enough. The mask is back-projected onto the table at the object's mid height. Cube: centre =
  mask centroid, angle = the minimum-area rectangle's edges (mod 90). Screw (any type): the rectangle's long axis; the
  head is the wider end (width profile along the axis, outer thirds compared); the grasp is the middle of the shank only.
  Skipped only when outside the TCP box, unreachable by IK, in or touching a bin, or when a finger would touch a bin; a
  finger touching a neighbour is logged, not refused. AAA cells, caps and bins use `pose_fit.refine` unchanged.
- **Step 3, no park after every pick.** After a release the arm goes to a clear-view pose on the bin's side: (0.18,
  -0.38, 0.35) m for bins at y < 0 (park.py: 244 px clear of the work area), the park pose otherwise, and the next table
  photo is taken there. It parks at the start of a rollout and when a tracked target or a bin is hidden behind the arm.
- **Step 4, wrist grasp check (`mm/real/wrist_check.py`).** The pick is five moves: to hover | descend | close + lift
  back to hover | carry + lower | release + retreat. At hover, one wrist image before the descent and one after the lift
  (the cell's wrist calibration; the spot is at ~(1022, 867) in the 1920x1080 wrist image, the fingertips below it). The
  object's blob at the spot still there (>= 50 %): missed, open and descend again at once, at most 3 tries; gone (<= 25 %):
  picked; otherwise the next table image judges, as in v5. Both images are logged. The stretch (refining the grasp from
  the wrist image) is not done.
- **Step 5.** No pickable target in 3 captures in a row: go home, log why each object was skipped, hold still (the
  client cannot end a trial). Before the first pick, the same targets and bins in two captures in a row; on every capture,
  nothing large on the table that is not a detection (`mm/real/scene_check.py`; a hand or arm). Bins re-found every
  capture at any distance (`track_bins(radius=inf)`); a bin not seen now or with its release pose outside the TCP box
  takes no picks.

**Checks on the 28 Oct real pictures** (`mm/eval/outline_check.py`; 128 captures of run r20261028-716eb3, test only):

| Class | Detected | v5 would pick | v6 would pick | v6 skips |
|---|---|---|---|---|
| cube | 705 | 2 | 676 | 20 fewer than 3 bins, 9 conf < 0.5 |
| screw | 165 | 0 | 131 | 22 IK, 10 fewer than 3 bins, 2 conf < 0.5 |
| aaa_cell | 39 | 34 | 26 | 8 IK (see below), 2 bins, 2 conf, 1 pose fit |

- AAA perception is identical in all 39 detections (pose, fit, gate). The 8 extra AAA skips are all in t02/t03 frames
  where one bin stood at y = +0.483 m, out of reach: with cubes now pickable, the bin assignment (most picks first) gives
  that bin to the AAA cells instead of to the cubes.
- Outline quality (1b): cube masks are loose but centred: 402 px median against 225 px for a 10 mm cube's silhouette at
  the same spot (~20 px against 15 px a side; bins, caps and AAA cells measure 1.01, 1.01 and 0.93 of theirs, so the
  calibration is right), 1.15x the cube's own coloured pixels. Grasp centres from the mask and from the coloured pixels
  agree to 0.5 mm median (p90 1.0 mm); angles to 4.5 deg median (p90 15 deg; a cube at any angle fits the 16 mm open
  jaws). Screws: 0.86x the model screw's area; length 28.4 mm, shank 6.3 mm, head 10.6 mm (medians); the head was found
  in 159 of 165.
- Cases that look off: a yellow cube touching the yellow cap (rollout_04: the mask runs into the cap, centre between
  them); cubes held in a hand (rollout_03; v6 waits while a hand is on the table); 6 screws with no clear head (grip at
  the middle of the whole length). Previews: `runs/oc/review/`, `runs/oc/suspects/`.
- Hand check on the same captures: 19 flagged, all a person reaching onto the table (9,100-50,700 px); none otherwise.
- Decision time per capture 2.9 s (v6) against 8.7 s (v5) on these images.
- The run's recording: t03-t06 ended with `end_reason: "controller fault"` (manipbench.json). Wrist frames by rollout:
  `data/wrist_20261028/` (`mm/datagen/wrist_frames.py`).

**Sim at the real arm's speed (50 mm/s), layouts 1-5, v5_real weights:**

| | v5 (`runs/sim_v5base_arm050`) | v6 (`runs/sim_v6_arm050`) |
|---|---|---|
| Fully sorted | 3/5 (layouts 3, 5: one target never planned) | 5/5 |
| Picks / misses / table moves | 20 / 0 / 0.0 mm | 22 / 0 / 0.0 mm |
| Arm time per pick (median) | 69.0 s | 56.5 s |
| Arm time to the last release, layouts 1, 2, 4 | 350, 282, 352 s | 283, 246, 293 s |

The wrist check called all 22 v6 picks "picked", as the twin's ground truth confirms. The mock from the v6 client
passes (218 actions, 0 rejected, act() 4.3 s); its synthetic images read as "something large on the table", so it only
checks the protocol. Host tests: 113 pass (15 new in `tests/test_v6.py`), 1 skipped; `test_runtime_config` still has its
v5 error (`Path.home()` under the test's cleared environment).

**Symmetric grasp angles (v6, 28 Oct, switch back: "6" in `MM_V5_STEPS`).** Before a target is called unreachable,
`grasp.spread_options` also tries each angle's symmetric twin, after v5's own: cube yaw + 90, 0, 270, 180; screw and AAA
cell yaw + 90, 270. The jaws close along the same line, but the wrist is turned 180 deg, so the IK is different.
- Today's pictures: 0 of the 22 unreachable screws recovered; every count is unchanged. All 22 fail at the "above_bin"
  stage, not at the grasp. In those t02/t03 frames two of the three bins were out of the arm's reach (bin_1 at y =
  +0.483 m, bin_3 at x = 0.555 m). Each screw could be sorted into bin_2, but bin_2 went to the cubes (more picks). No
  grasp angle reaches a bin the arm cannot reach.
- Sim at 50 mm/s, layouts 1-5 (`runs/sim_v6sym_arm050`): 5/5 sorted, 22/22 picks (wrist: all picked), 0 misses,
  0.0 mm table moves, no faults. The times are the same as before (56.5 s per pick median), because every grasp there
  worked at its first angle. Mock: 218 actions, 0 rejected, act() 3.5 s. Host tests: 114 pass, 1 skipped.
