"""
Which bin gets which class, chosen by the robot (v4: the three bins are identical). Host-side.

At a rollout's first capture with all three bins found, every assignment of the target classes
(cube, screw, aaa_cell) to the physical bins is planned with the real planner (plan_bins: grasp,
finger, reach and IK checks at that bin) and ordered (greedy + 2-opt). The chosen one plans the
most picks, then has the shortest predicted TCP travel. With 3 classes and 3 bins that is all 6
permutations: exact, and cheap (one plan each). The log keeps every assignment's travel, so the
saving against the worst is reported.

The result is a bins table keyed by class (the planner's slots), each entry a physical bin with
"id" (bin_1..3): plan_bins, the executor and the tracker use it unchanged.
"""

import itertools

from mm.planning import plan_bins as PB


def target_slots(classes=None):
    classes = classes or {c["name"]: c for c in PB.load_yaml("classes.yaml", "classes")}
    return sorted({c["bin"] for c in classes.values() if c.get("role") == "target" and c.get("bin")})


def table(phys, assignment):
    """{slot: physical bin + id} for an assignment {slot: bin id}."""
    return {slot: dict(phys[bid], id=bid) for slot, bid in assignment.items()}


def default_table(phys=None, classes=None):
    """A fixed class -> bin table (slots in name order onto bin_1..3), for tests and
    tools that plan without a scene to choose from."""
    from mm.datagen.randomize import load_bins

    phys = phys or load_bins()
    return table(phys, dict(zip(target_slots(classes), sorted(phys))))


def assign_bins(det, cam, phys, classes=None, reach=None, strategy="greedy_2opt"):
    """Best assignment of target slots to physical bins.

    det: {"objects": detections}; phys: {bin id: bin dict} (at least one bin per slot).
    Returns (assignment {slot: bin id}, log).
    """
    slots = target_slots(classes)
    if len(phys) < len(slots):
        raise ValueError(f"{len(phys)} bins for {len(slots)} classes")
    tried = []
    for perm in itertools.permutations(sorted(phys), len(slots)):
        assignment = dict(zip(slots, perm))
        plan = PB.build_plan(det, cam, classes, table(phys, assignment), reach, strategy)
        tried.append({"assignment": assignment, "picks": len(plan["picks"]),
                      "travel_m": round(plan["predicted_travel_m"], 4),
                      "skipped": sorted({s["skip_reason"] for s in plan["skipped"] if s.get("stage") == "planning"})})
    best = min(tried, key=lambda t: (-t["picks"], t["travel_m"]))
    peers = [t for t in tried if t["picks"] == best["picks"]]
    worst = max(peers, key=lambda t: t["travel_m"])
    log = {
        "assignment": best["assignment"],
        "picks": best["picks"],
        "travel_m": best["travel_m"],
        "worst_assignment": worst["assignment"],
        "worst_travel_m": worst["travel_m"],
        "saving_m": round(worst["travel_m"] - best["travel_m"], 4),
        "saving_pct": round(100 * (worst["travel_m"] - best["travel_m"]) / worst["travel_m"], 1) if worst["travel_m"] else 0.0,
        "tried": tried,
    }
    return best["assignment"], log
