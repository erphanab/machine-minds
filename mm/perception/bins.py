"""
Sort bins from perception (Machine Minds B8; v4: identical bins). Host-side.

The three bins share one shape and one colour (tangerine yellow), so nothing in
the image says which is which. YOLO finds them as class "bin" and the pose fit
gives each one's centre and yaw (square: mod 90). Identity is positional:

  first_bins()  the first capture with all three bins found (fit >= MIN_FIT)
                names them bin_1..3 by position (x, then y). Fewer than three:
                None, and the caller captures again rather than guess.
  track_bins()  every later capture matches each named bin to the nearest bin
                found (within TRACK_RADIUS), so names never swap. A bin not seen
                this capture keeps its last pose (found=False).

Which class goes to which bin is the robot's choice (mm/planning/bin_assign.py).
Entries have the form of randomize.load_bins(): design + colour, mat, xy, yaw_deg.
"""

import math

from mm.datagen.randomize import load_bin_design, load_bins

MIN_FIT = 0.80  # a bin is big; a lower fit means it is partly hidden or wrong
SAME_BIN = 0.045  # two detections closer than half a bin are one bin
TRACK_RADIUS = 0.06  # bins are not moved during a rollout; this allows for pose noise


def bins_seen(dets):
    """Bin detections good enough to use, one per bin (best fit wins)."""
    cands = sorted((d for d in dets if d["class"] == "bin" and d.get("xy_robot_m") is not None
                    and d.get("fit_iou", 0) >= MIN_FIT), key=lambda d: -d["fit_iou"])
    out = []
    for d in cands:
        if all(math.dist(d["xy_robot_m"], o["xy_robot_m"]) > SAME_BIN for o in out):
            out.append(d)
    return out


def _entry(d):
    style = next(iter(load_bins().values()))
    return dict(load_bin_design(), colour=style["colour"], mat=style["mat"], xy=list(d["xy_robot_m"]),
                yaw_deg=float(d["yaw_deg"]) % 90.0, found=True, fit_iou=d["fit_iou"], conf=d["conf"])


def first_bins(dets, n=3):
    """{bin_1..n: bin} named by position, or None if fewer than n bins are found."""
    seen = bins_seen(dets)
    if len(seen) < n:
        return None
    seen = sorted(seen[:n], key=lambda d: (round(d["xy_robot_m"][0], 3), d["xy_robot_m"][1]))
    return {f"bin_{k + 1}": _entry(d) for k, d in enumerate(seen)}


def track_bins(dets, previous, radius=None):
    """The same names, each moved to the nearest bin found now (closest pairs first), at most radius away (default
    TRACK_RADIUS; v6 passes math.inf: bins moved during the 28 Oct 2026 rollouts and are re-found wherever they are)."""
    radius = TRACK_RADIUS if radius is None else radius
    seen = bins_seen(dets)
    pairs = sorted((math.dist(b["xy"], d["xy_robot_m"]), name, k)
                   for name, b in previous.items() for k, d in enumerate(seen))
    out, used = {}, set()
    for dist, name, k in pairs:
        if dist > radius or name in out or k in used:
            continue
        out[name] = _entry(seen[k])
        used.add(k)
    for name, b in previous.items():
        if name not in out:
            out[name] = dict(b, found=False)
    return {name: out[name] for name in previous}
