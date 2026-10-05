#!/usr/bin/env python
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

WEIGHTS = {"ego_progress_along_expert_route": 5.0, "time_to_collision_within_bound": 5.0,
           "speed_limit_compliance": 4.0, "ego_is_comfortable": 2.0}
MULTIPLIERS = ["no_ego_at_fault_collisions", "drivable_area_compliance",
               "driving_direction_compliance", "ego_is_making_progress"]
SHORT = {"ego_progress_along_expert_route": "progress", "time_to_collision_within_bound": "TTC",
         "speed_limit_compliance": "speed", "ego_is_comfortable": "comfort"}
SPLITS = "val14|test14-hard|test14-random"


def run_dirs(exp_root, split, ch, tag):
    base = glob.glob(os.path.join(exp_root, "**", "flow_drive", split, ch), recursive=True)
    out = []
    chunk_re = re.compile(rf"^{re.escape(tag)}-(?:{SPLITS})-c\d+$")
    for b in base:
        for d in sorted(os.listdir(b)):
            if d == tag or chunk_re.match(d):
                stamps = sorted(s for s in glob.glob(os.path.join(b, d, "*"))
                                if glob.glob(os.path.join(s, "aggregator_metric", "*.parquet")))
                if stamps:
                    out.append(stamps[-1])
    return out


def load(exp_root, split, ch, tag):
    dirs = run_dirs(exp_root, split, ch, tag)
    if not dirs:
        raise SystemExit(f"[err] 결과 없음: {split}/{ch}/{tag}")
    aggs, spds = [], []
    for d in dirs:
        a = pd.read_parquet(glob.glob(os.path.join(d, "aggregator_metric", "*.parquet"))[0])
        a = a[a["scenario"] != "final_score"]
        a = a[a["log_name"].notna()] if "log_name" in a else a
        aggs.append(a)
        sp = os.path.join(d, "metrics", "speed_limit_compliance.parquet")
        if os.path.exists(sp):
            spds.append(pd.read_parquet(sp))
    agg = pd.concat(aggs).drop_duplicates("scenario", keep="last").set_index("scenario")
    if spds:
        s = pd.concat(spds)
        key = "scenario_name" if "scenario_name" in s else "scenario"
        pick = {}
        for c in s.columns:
            if c.endswith("_stat_value"):
                if c.startswith("number_of_violations"):
                    pick[c] = "n_viol"
                elif c.startswith("max_violation"):
                    pick[c] = "max_over"
                elif c.startswith("mean_violation"):
                    pick[c] = "mean_over"
        s = s[[key] + list(pick)].rename(columns=pick).drop_duplicates(key, keep="last").set_index(key)
        agg = agg.join(s, how="left")
    for c in ("n_viol", "max_over", "mean_over"):
        if c not in agg:
            agg[c] = np.nan
    agg["n_viol"] = agg["n_viol"].fillna(0)
    return agg, len(dirs)


def decompose(df):
    mult = np.ones(len(df))
    for m in MULTIPLIERS:
        if m in df:
            mult = mult * df[m].fillna(1.0).to_numpy()
    W = sum(WEIGHTS.values())
    out = {}
    for m, w in WEIGHTS.items():
        if m in df:
            out[SHORT[m]] = 100 * np.mean(mult * w * df[m].fillna(0).to_numpy() / W)
    out["sum"] = sum(out.values())
    out["CLS(실제)"] = 100 * df["score"].mean()
    out["배수=0 시나리오"] = int((mult == 0).sum())
    return out


def summary_row(name, df):
    v = df["n_viol"] > 0
    return {"method": name, "n": len(df), "CLS": round(100 * df["score"].mean(), 2),
            "speed_score": round(df["speed_limit_compliance"].mean(), 4),
            "위반 시나리오 %": round(100 * v.mean(), 1),
            "평균 위반 횟수": round(df["n_viol"].mean(), 2),
            "최대초과 평균 m/s": round(df.loc[v, "max_over"].mean(), 2) if v.any() else 0.0,
            "최대초과 p95 m/s": round(df.loc[v, "max_over"].quantile(0.95), 2) if v.any() else 0.0,
            "progress": round(df["ego_progress_along_expert_route"].mean(), 4),
            "collision": round(df["no_ego_at_fault_collisions"].mean(), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--ch", default="nr")
    ap.add_argument("--base", required=True)
    ap.add_argument("--cmp", nargs="+", required=True)
    ap.add_argument("--exp_root", default=os.environ.get("NUPLAN_EXP_ROOT", "/root/nuplan/exp"))
    ap.add_argument("--csv", default="", help="시나리오별 비교표 저장")
    a = ap.parse_args()
    pd.set_option("display.width", 250, "display.max_columns", 40)

    data = {}
    for t in [a.base] + a.cmp:
        data[t], k = load(a.exp_root, a.split, a.ch, t)
        print(f"[load] {t}: 시나리오 {len(data[t])}개 (실행 {k}개)")
    common = sorted(set.intersection(*[set(d.index) for d in data.values()]))
    data = {t: d.loc[common] for t, d in data.items()}
    print(f"[load] 공통 시나리오 {len(common)}개로 비교\n")

    print("[1] 속도 위반 요약 (전체)")
    print(pd.DataFrame([summary_row(t, d) for t, d in data.items()]).to_string(index=False), "\n")

    base = data[a.base]
    viol = base.index[(base["n_viol"] > 0) | (base["speed_limit_compliance"] < 1.0)]
    print(f"[2] 기준({a.base})이 속도를 위반한 시나리오 {len(viol)}개 / {len(common)}개에서 비교")
    if len(viol):
        print(pd.DataFrame([summary_row(t, d.loc[viol]) for t, d in data.items()]).to_string(index=False))
        rest = base.index.difference(viol)
        print(f"    (참고) 위반 없던 나머지 {len(rest)}개의 CLS: "
              + ", ".join(f"{t} {100 * d.loc[rest, 'score'].mean():.2f}" for t, d in data.items()))
    print()

    print("[3] 시나리오 유형별 (기준의 위반 비율 높은 순)")
    rows = []
    for st, idx in base.groupby("scenario_type").groups.items():
        r = {"scenario_type": st, "n": len(idx),
             f"위반% {a.base}": round(100 * (base.loc[idx, "n_viol"] > 0).mean(), 0)}
        for t, d in data.items():
            r[f"speed {t}"] = round(d.loc[idx, "speed_limit_compliance"].mean(), 3)
        for t, d in data.items():
            r[f"CLS {t}"] = round(100 * d.loc[idx, "score"].mean(), 1)
        rows.append(r)
    print(pd.DataFrame(rows).sort_values(f"위반% {a.base}", ascending=False).to_string(index=False), "\n")

    print("[4] CLS 분해 (항별 기여, 점). 차이 = 방법 - 기준")
    dec = {t: decompose(d) for t, d in data.items()}
    tab = pd.DataFrame(dec).T
    print(tab.round(2).to_string())
    diff = tab.subtract(tab.loc[a.base], axis=1).drop(index=a.base)
    print("\n    기준 대비 차이")
    print(diff.drop(columns=["배수=0 시나리오"]).round(2).to_string())
    print("    (sum과 CLS(실제)가 거의 같아야 공식이 맞는 것. speed 항이 +면 속도 준수로 얻은 점수)")

    if a.csv:
        out = pd.concat({t: d[["scenario_type", "score", "speed_limit_compliance", "n_viol", "max_over",
                                "ego_progress_along_expert_route", "no_ego_at_fault_collisions"]]
                         for t, d in data.items()}, axis=1)
        out.to_csv(a.csv)
        print(f"\n[csv] -> {a.csv}")


if __name__ == "__main__":
    main()
