#!/usr/bin/env python
import argparse
import glob
import os

import numpy as np
import pandas as pd

ORDER = ["human", "pdmc", "fd", "fd_clip", "fd_post", "yflow-kin-v2", "yflow_post-kin-v2", "yflow-corr-v2",
         "yflow_post-corr-v2", "yflow-kin2-v2", "yflow_post-kin2-v2", "yflow_post-corr2-v2",
         "fdstar", "yflow_star-kin-v2", "yflow_star-corr-v2", "yflow_star-kin2-v2"]
GROUP = {"human": "참고", "pdmc": "참고", "fdstar": "Scorer", "yflow_star-kin-v2": "Scorer", "yflow_star-corr-v2": "Scorer",
         "yflow_star-kin2-v2": "Scorer"}
METRICS = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]
PAIRS = [("fd", "yflow-kin-v2"), ("fd", "fd_post"), ("fd_post", "yflow_post-kin-v2"), ("fd_post", "yflow_post-corr-v2"),
         ("yflow-kin-v2", "yflow_post-kin-v2"), ("fd", "fdstar"), ("fdstar", "yflow_star-kin-v2"),
         ("fd_post", "yflow_post-kin2-v2"), ("yflow_post-kin-v2", "yflow_post-kin2-v2"), ("fdstar", "yflow_star-kin2-v2")]


def load(root, split):
    out = {}
    for d in sorted(glob.glob(os.path.join(root, split, "*"))):
        files = sorted(glob.glob(os.path.join(d, "*.parquet")))
        if files:
            df = pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates(["token", "frame"], keep="last")
            out[os.path.basename(d)] = df
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--frame", default="now", choices=["now", "prev"])
    ap.add_argument("--root", default=os.path.join(os.environ.get("RESULTS_DIR", "/root/fd_yflow_results"), "navsim_eval"))
    ap.add_argument("--xlsx", default="")
    ap.add_argument("--pairs", nargs="*", default=[], help="추가 짝 비교 '기준:대상' (예: fdstar:yflow_star-kin2-v2-b6)")
    a = ap.parse_args()
    data = load(a.root, a.split)
    if not data:
        raise SystemExit(f"[err] 결과 없음: {a.root}/{a.split}")
    data = {k: v[v["frame"] == a.frame].set_index("token") for k, v in data.items()}
    common = sorted(set.intersection(*[set(d.index) for d in data.values()]))
    names = sorted(data, key=lambda k: (ORDER.index(k) if k in ORDER else 99, k))
    pd.set_option("display.width", 250, "display.max_columns", 30)

    rows = []
    for k in names:
        d = data[k].loc[common]
        r = {"그룹": GROUP.get(k, "No scorer"), "method": k, "n": len(d)}
        for m in METRICS:
            r[m] = round(100 * d[m].mean(), 1) if m in d and d[m].notna().any() else np.nan
        r["EPDMS"] = round(100 * d["EPDMS"].mean(), 2)
        r["0점 수"] = int((d["EPDMS"] == 0).sum())
        r["plan ms"] = round(d["plan_ms"].mean(), 0)
        rows.append(r)
    tab = pd.DataFrame(rows)
    print(f"[{a.split}] frame={a.frame}, 공통 시나리오 {len(common)}개  (지표는 0~100, EPDMS = NC·DAC·DDC·TLC × (5EP+5TTC+2LK+2HC+2EC)/16)")
    print(tab.to_string(index=False), "\n")

    rng = np.random.default_rng(0)
    prs = []
    for x, y in PAIRS + [tuple(p.split(":", 1)) for p in a.pairs]:
        if x in data and y in data:
            diff = 100 * (data[y].loc[common, "EPDMS"] - data[x].loc[common, "EPDMS"]).to_numpy()
            bs = rng.choice(diff, (10000, len(diff))).mean(1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
            prs.append({"기준": x, "대상": y, "Δ EPDMS": round(diff.mean(), 2), "CI 하한": round(lo, 2), "CI 상한": round(hi, 2),
                        "p": round(p, 3), "승": int((diff > 0.5).sum()), "패": int((diff < -0.5).sum())})
    pr = pd.DataFrame(prs)
    if len(pr):
        print("짝 bootstrap (같은 시나리오끼리, 10,000회)")
        print(pr.to_string(index=False), "\n")

    types = pd.DataFrame({k: 100 * data[k].loc[common].groupby("scenario_type")["EPDMS"].mean() for k in names}).round(1)
    types.insert(0, "n", data[names[0]].loc[common].groupby("scenario_type").size())
    print("유형별 EPDMS")
    print(types.to_string(), "\n")

    if a.xlsx:
        with pd.ExcelWriter(a.xlsx) as w:
            tab.to_excel(w, sheet_name="summary", index=False)
            pr.to_excel(w, sheet_name="pairs", index=False)
            types.to_excel(w, sheet_name="types")
            pd.concat({k: data[k].loc[common] for k in names}, axis=1).to_excel(w, sheet_name="per_scenario")
        print(f"[xlsx] -> {a.xlsx}")


if __name__ == "__main__":
    main()
