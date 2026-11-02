"""
Check a curated demo run (mm.eval.demo_layout + run_trial --layout --loop-json) against the demo's rules. Host-side:

    .venv/bin/python -m mm.eval.demo_check runs/<run> [...]

Prints: target refusals at any capture (gates: fit, finger collision, reach, flatness; objects already in a bin
excluded), bins found and the robot's assignment, the deliberate miss (3rd attempt, a first try, caught, then
placed on retry), the nudge (applied, nudged object placed), bin contents (ground truth) and the v4 rule
(each bin one type, every target binned, cap and tube not in a bin). Exit status 0 only if all hold.
"""

import json
import os
import sys

GATES = ("pose_fit_below_gate", "finger_collision", "ik_unreachable", "not_resting_flat", "low_confidence",
         "no_bin_for_class", "exceeds_jaw_opening")


def check(run):
    r = json.load(open(os.path.join(run, "execution_report.json")))
    att, ok, notes = r["picks"], {}, []
    # gates: targets on the table refused at a capture. A capture lists refusals of objects already sorted too
    # (their skip reason is in_bin when eligible, else the gate that fired on the bin view), so count only
    # captures where fewer targets were planned than remain on the table.
    on_table = r["targets_initial"] + r.get("targets_added", 0)
    refused = []
    for c in r["cycles"]:
        if not isinstance(c.get("bins"), dict):
            continue
        sorted_before = sum(1 for a in att if a["cycle"] < c["cycle"] and a["placement_verified"])
        remaining = on_table - sorted_before
        if c["planned"] < remaining:
            refused.append((c["cycle"], remaining, c["planned"], [s for s in c["skipped"] if s[1] in GATES]))
    ok["no_gate_refusals"] = not refused
    first = next((c for c in r["cycles"] if isinstance(c.get("bins"), dict)), None)
    ok["bins_found_at_start"] = bool(first and len(first["bins"]) == 3 and all(b["found"] for b in first["bins"].values()))
    ba = r.get("bin_assignment") or {}
    ok["assignment_logged"] = bool(ba.get("assignment"))
    d = [a for a in att if a.get("deliberate_miss")]
    ok["one_deliberate_miss_on_attempt_3"] = len(d) == 1 and att.index(d[0]) == 2 and d[0]["attempt"] == 1
    ok["deliberate_caught_and_retried"] = bool(d) and not d[0]["verdict_taken"] and not d[0]["gt_lifted"] and any(
        a.get("tracked_object") == d[0]["tracked_object"] and a["placement_verified"] for a in att[att.index(d[0]) + 1:])
    ok["no_injection_on_retry"] = all(a["injected_offset_mm"] == 0 for a in att if a.get("attempt", 1) > 1)
    nudges = [e for e in r["events"] if e.get("type") == "nudge"]
    ok["nudge_applied_and_sorted"] = len(nudges) == 1 and "prim" in nudges[0] and any(
        a.get("tracked_object") == nudges[0]["prim"] and a["placement_verified"] for a in att)
    ok["no_objects_added"] = not any(e.get("type") == "add" for e in r["events"])
    ok["sort_ok"] = bool(r.get("sort_ok"))
    ok["no_abort"] = r.get("abort_reason") is None
    print(f"== {run}: {'PASS' if all(ok.values()) else 'FAIL'}")
    for k, v in ok.items():
        print(f"   {'ok ' if v else 'NO '} {k}")
    print(f"   assignment {ba.get('assignment')} travel {ba.get('travel_m')} m vs worst {ba.get('worst_travel_m')} m")
    print(f"   bin contents {r.get('bin_contents')}")
    print(f"   attempts: " + "; ".join(f"{a['category']}{' (deliberate)' if a.get('deliberate_miss') else ''} "
                                      f"{'placed' if a['placement_verified'] else 'missed' if not a['gt_lifted'] else 'dropped'}"
                                      for a in att))
    print(f"   nudge {nudges}")
    if refused:
        print(f"   refusals (cycle, on table, planned, gates): {refused}")
    return all(ok.values())


if __name__ == "__main__":
    results = [check(run) for run in sys.argv[1:]]
    sys.exit(0 if results and all(results) else 1)
