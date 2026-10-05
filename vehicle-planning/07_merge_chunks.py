#!/usr/bin/env python
import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd

METRICS = [
    ("no_ego_at_fault_collisions", "collision"),
    ("drivable_area_compliance", "drivable"),
    ("driving_direction_compliance", "direction"),
    ("ego_is_making_progress", "making_prog"),
    ("time_to_collision_within_bound", "TTC"),
    ("speed_limit_compliance", "speed_lim"),
    ("ego_progress_along_expert_route", "progress"),
    ("ego_is_comfortable", "comfort"),
]
TAG_RE = re.compile(r"^(?P<base>.+?)-(?P<split>val14|test14-hard|test14-random)-c(?P<id>\d+)(?:-(?P<cfg>.+))?$")


def scenario_rows(df):
    df = df[df["scenario"] != "final_score"]
    if "log_name" in df:
        return df[df["log_name"].notna()]
    types = set(df["scenario_type"]) if "scenario_type" in df else set()
    return df[~df["scenario"].isin(types)]


def aggregate(rows):
    n = len(rows)
    out = {"n_scen": n, "CLS": round(100 * rows["score"].sum() / n, 2) if n else float("nan")}
    for col, name in METRICS:
        if col in rows:
            out[name] = round(float(rows[col].fillna(0).sum()) / n, 4)
    return out


def yflow_summary(uids, results_dir):
    recs = []
    for uid in uids:
        d = os.path.join(results_dir, "yflow_stats", uid.replace("/", "_"))
        for f in glob.glob(os.path.join(d, "*.jsonl")):
            recs += [json.loads(l) for l in open(f) if l.strip()]
    if not recs:
        return {}
    lv = Counter(l.replace("(rs)", "") for r in recs for l in (r.get("levels") or [r.get("level")]))
    n = sum(lv.values())
    full = max(lv, key=lambda k: k.count("+"))
    return {"calls": len(recs), "full_ok_%": round(100.0 * lv.get(full, 0) / n, 2),
            "fail_%": round(100.0 * sum(v for k, v in lv.items() if "fail" in k) / n, 2),
            "rs_%": round(100.0 * sum(1 for r in recs if r.get("resampled", 0) > 0) / len(recs), 2),
            "corr_skip_%": round(100.0 * sum(1 for r in recs if r.get("corr_skipped")) / len(recs), 2),
            "ms_mean": round(sum(r["total_ms"] for r in recs) / len(recs), 1)}


def find_runs(exp_root, split):
    runs = defaultdict(dict)
    pat = os.path.join(exp_root, "**", "flow_drive", split, "*", "*", "*", "aggregator_metric", "*.parquet")
    for p in sorted(glob.glob(pat, recursive=True)):
        parts = p.split(os.sep)
        i = len(parts) - 1 - parts[::-1].index("flow_drive")
        _, sp, ch, tag, stamp = parts[i:i + 5]
        m = TAG_RE.match(tag)
        if not m or m["split"] != split:
            continue
        group = m["base"] + (f"[{m['cfg']}]" if m["cfg"] else "")
        runs[(group, ch)][int(m["id"])] = (p, "/".join(parts[i:i + 5]))
    return runs


def selftest(exp_root):
    worst = 0.0
    n = 0
    for p in glob.glob(os.path.join(exp_root, "**", "aggregator_metric", "*.parquet"), recursive=True):
        df = pd.read_parquet(p)
        fin = df[df["scenario"] == "final_score"]
        if fin.empty:
            continue
        rows = scenario_rows(df)
        agg = aggregate(rows)
        ref = 100 * float(fin.iloc[0]["score"])
        d = abs(agg["CLS"] - ref)
        worst = max(worst, d)
        n += 1
        flag = "OK " if d < 0.01 and agg["n_scen"] == int(fin.iloc[0]["num_scenarios"]) else "BAD"
        print(f"[{flag}] n={agg['n_scen']:4d} 재계산 {agg['CLS']:6.2f} vs final_score {ref:6.2f}  {os.path.relpath(p, exp_root)[:90]}")
    print(f"[selftest] {n}개 실행, 최대 차이 {worst:.4f} -> {'PASS' if worst < 0.01 else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val14")
    ap.add_argument("--exp_root", default=os.environ.get("NUPLAN_EXP_ROOT", "/root/nuplan/exp"))
    ap.add_argument("--results_dir", default=os.environ.get("RESULTS_DIR", "/root/fd_yflow_results"))
    ap.add_argument("--data_root", default=os.environ.get("NUPLAN_DATA_ROOT", "/root/nuplan/dataset"))
    ap.add_argument("--csv", default="")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--check_chunk", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.exp_root)

    st_path = os.path.join(a.data_root, "chunks", a.split, "state.json")
    st = json.load(open(st_path)) if os.path.exists(st_path) else None
    expected = st["n_tokens"] if st else None
    runs = find_runs(a.exp_root, a.split)
    if not runs:
        print(f"[merge] {a.split} 청크 결과 없음")
        return

    if a.check_chunk:
        want = next((c["tokens"] for c in (st or {}).get("chunks", []) if c["id"] == a.chunk), None)
        for (group, ch), by_id in sorted(runs.items()):
            if a.chunk in by_id:
                n = len(scenario_rows(pd.read_parquet(by_id[a.chunk][0])))
                ok = "OK" if want is None or n == want else f"주의: 청크 토큰 {want}개와 다름"
                print(f"[check] c{a.chunk:02d} {group:28s} {ch}: 시나리오 {n}개 ({ok})")
        return

    table = []
    for (group, ch), by_id in sorted(runs.items()):
        frames = [scenario_rows(pd.read_parquet(p)) for p, _ in by_id.values()]
        rows = pd.concat(frames).drop_duplicates(subset="scenario", keep="last")
        row = {"method": group, "ch": ch, "chunks": ",".join(str(k) for k in sorted(by_id))}
        row.update(aggregate(rows))
        if expected:
            row["coverage"] = f"{row['n_scen']}/{expected}"
        row.update(yflow_summary([u for _, u in by_id.values()], a.results_dir))
        table.append(row)
    out = pd.DataFrame(table)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(out.to_string(index=False))
    if st:
        done = [c["id"] for c in st["chunks"] if c["status"] == "done"]
        print(f"[state] 토큰 {len(st['token_log'])}/{st['n_tokens']} 확보, 끝난 청크 {done}")
    if a.csv:
        out.to_csv(a.csv, index=False)
        print(f"[merge] -> {a.csv}")


if __name__ == "__main__":
    main()
