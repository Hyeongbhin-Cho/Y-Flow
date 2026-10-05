#!/usr/bin/env python
import argparse
import glob
import json
import os
from collections import Counter

import pandas as pd

METRICS = [
    ("score", "CLS"),
    ("no_ego_at_fault_collisions", "collision"),
    ("drivable_area_compliance", "drivable"),
    ("driving_direction_compliance", "direction"),
    ("ego_is_making_progress", "making_prog"),
    ("time_to_collision_within_bound", "TTC"),
    ("speed_limit_compliance", "speed_lim"),
    ("ego_progress_along_expert_route", "progress"),
    ("ego_is_comfortable", "comfort"),
]


def uid_from_path(p, exp_root):
    rel = os.path.relpath(p, exp_root)
    parts = rel.split(os.sep)
    if "flow_drive" in parts:
        i = parts.index("flow_drive")
        return "/".join(parts[i:i + 5])
    return os.path.dirname(os.path.dirname(rel))


def yflow_stats(uid, results_dir):
    d = os.path.join(results_dir, "yflow_stats", uid.replace("/", "_"))
    files = glob.glob(os.path.join(d, "*.jsonl"))
    if not files:
        return {}
    recs = [json.loads(l) for f in files for l in open(f) if l.strip()]
    if not recs:
        return {}
    lv = Counter(l.replace("(rs)", "") for r in recs for l in (r.get("levels") or [r.get("level")]))
    n = sum(lv.values())
    full = max(lv, key=lambda k: k.count("+")) if lv else ""
    fails = sum(v for k, v in lv.items() if "fail" in k)
    viol = pd.DataFrame([r["viol"] for r in recs if r.get("viol")])
    out = {
        "calls": len(recs),
        "full_level": full,
        "full_ok_%": 100.0 * lv.get(full, 0) / n,
        "fallback_%": 100.0 * (n - lv.get(full, 0)) / n,
        "fail_%": 100.0 * fails / n,
        "ms_mean": sum(r["total_ms"] for r in recs) / len(recs),
        "shift_m": sum(r.get("shift_m", 0) for r in recs) / len(recs),
    }
    if any("resampled" in r for r in recs):
        out["rs_%"] = 100.0 * sum(1 for r in recs if r.get("resampled", 0) > 0) / len(recs)
        out["corr_skip_%"] = 100.0 * sum(1 for r in recs if r.get("corr_skipped")) / len(recs)
    for k in ("speed", "accel", "corridor", "obstacle"):
        if not viol.empty and k in viol:
            out[f"viol_{k}_p99"] = viol[k].quantile(0.99)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", default=os.environ.get("NUPLAN_EXP_ROOT", "/root/nuplan/exp"))
    ap.add_argument("--results_dir", default=os.environ.get("RESULTS_DIR", "/root/fd_yflow_results"))
    ap.add_argument("--filter", default="")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(a.exp_root, "**", "aggregator_metric", "*.parquet"), recursive=True)):
        if a.filter and a.filter not in p:
            continue
        df = pd.read_parquet(p)
        fin = df[df["scenario"] == "final_score"] if "scenario" in df else df.tail(1)
        if fin.empty:
            continue
        fin = fin.iloc[0]
        uid = uid_from_path(p, a.exp_root)
        row = {"run": uid, "n_scen": int(fin.get("num_scenarios", -1)) if "num_scenarios" in fin else -1}
        for col, name in METRICS:
            if col in fin:
                row[name] = round(100 * float(fin[col]), 2) if col == "score" else round(float(fin[col]), 4)
        row.update({k: (round(v, 3) if isinstance(v, float) else v) for k, v in yflow_stats(uid, a.results_dir).items()})
        rows.append(row)
    if not rows:
        print(f"[summary] {a.exp_root} 아래 aggregator_metric parquet 없음 (시뮬레이션이 끝났는지 확인)")
        return
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(out.to_string(index=False))
    if a.csv:
        out.to_csv(a.csv, index=False)
        print(f"[summary] -> {a.csv}")


if __name__ == "__main__":
    main()
