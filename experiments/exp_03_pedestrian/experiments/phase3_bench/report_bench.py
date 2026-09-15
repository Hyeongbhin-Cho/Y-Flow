# -*- coding: utf-8 -*-
# experiments/exp_03_pedestrian/experiments/phase3_bench/report_bench.py
"""Collect runs/exp_03_pedestrian/bench/<model>/<subset>/results/summary.csv into one ETH/UCY table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from experiments._layout import RUN_ROOT
from experiments.phase2_cfm.report_study import block_bootstrap

SUBSETS = ("eth", "hotel", "univ", "zara1", "zara2")
ORDER = ("CONST_VEL", "FLOWMATCH", "POSTHOC_PROJ", "HARDFLOW", "YFLOW", "SAFEFLOW", "UNICONFLOW", "GUIDEFLOW")


def load_raw(path: Path):
    """raw.csv -> {method: {scene: (seed-averaged min_ade, gt_feasible)}}."""
    acc: dict = {}
    for r in csv.DictReader(path.open()):
        if r["min_ade"] in ("", "nan"):
            continue
        d = acc.setdefault(r["method"], {}).setdefault(int(r["scene"]), [[], r["gt_feasible"] in ("True", "true", "1")])
        d[0].append(float(r["min_ade"]))
    return {m: {sc: (float(np.mean(v[0])), v[1]) for sc, v in d.items()} for m, d in acc.items()}


def paired_section(root: Path, have, methods) -> list[str]:
    """Paired per-scene minADE differences with a moving-block bootstrap CI (needs --save_raw)."""
    raws = {s: load_raw(root / s / "results" / "raw.csv") for s in have if (root / s / "results" / "raw.csv").is_file()}
    if not raws:
        return ["", "(paired statistics need raw.csv: run the benchmark with --save_raw)"]
    rng = np.random.default_rng(0)
    out = []
    for ref in ("FLOWMATCH", "POSTHOC_PROJ"):
        for scope in ("all", "gt_feasible"):
            out += ["", f"## Paired Δ minADE vs {ref} (m), {scope} scenes, seed-averaged, block-bootstrap 95% CI (50-scene blocks)", "",
                    "| method | " + " | ".join(raws) + " |", "| --- |" + " --- |" * len(raws)]
            for m in methods:
                if m in (ref, "CONST_VEL"):
                    continue
                cells = []
                for s, raw in raws.items():
                    if m not in raw or ref not in raw:
                        cells.append("–")
                        continue
                    scenes = sorted(sc for sc in raw[m] if sc in raw[ref] and (scope == "all" or raw[m][sc][1]))
                    d = np.array([raw[m][sc][0] - raw[ref][sc][0] for sc in scenes])
                    if d.size == 0:
                        cells.append("–")
                        continue
                    lo, hi = block_bootstrap(d, rng)
                    mark = " *" if (hi < 0 or lo > 0) else ""
                    cells.append(f"{d.mean() * 1000:+.2f} mm [{lo * 1000:+.2f}, {hi * 1000:+.2f}]{mark}")
                out.append(f"| {m} | " + " | ".join(cells) + " |")
    out += ["", "`*`: 95% CI excludes 0."]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(RUN_ROOT / "bench"))
    p.add_argument("--model", default="moflow")
    a = p.parse_args()
    root = Path(a.root) / a.model
    R = {}
    for s in SUBSETS:
        f = root / s / "results" / "summary.csv"
        if f.is_file():
            for r in csv.DictReader(f.open()):
                if r["mean"] not in ("", None):
                    R[(s, r["method"], r["metric"])] = float(r["mean"])
    have = [s for s in SUBSETS if any(k[0] == s for k in R)]
    methods = [m for m in ORDER if any(k[1] == m for k in R)]

    def cell(s, m, k, nd=3):
        return f"{R[(s, m, k)]:.{nd}f}" if (s, m, k) in R else "–"

    def avg(m, k, nd=3):
        v = [R[(s, m, k)] for s in have if (s, m, k) in R]
        return f"{sum(v) / len(v):.{nd}f}" if len(v) == len(have) and v else "–"

    out = [f"# Exp-03 pedestrian benchmark ({a.model}, ETH/UCY SocialGAN split, K=20)", "",
           "## min20 ADE / FDE (m)", "", "| method | " + " | ".join(have) + " | AVG |", "| --- |" + " --- |" * (len(have) + 1)]
    for m in methods:
        out.append(f"| {m} | " + " | ".join(f"{cell(s, m, 'min_ade')} / {cell(s, m, 'min_fde')}" for s in have)
                   + f" | {avg(m, 'min_ade')} / {avg(m, 'min_fde')} |")
    for title, k, nd in (("avg ADE (m)", "avg_ade", 3), ("KDE-NLL", "kde_nll", 3), ("COL@0.1", "col010_sample_rate", 4),
                         ("COL@0.2", "col020_sample_rate", 4), ("APD (m)", "apd", 3), ("kinematic violation rate", "viol_sample_rate", 4),
                         ("latency (s / scene)", "latency_s", 4), ("mean change vs FLOWMATCH (m)", "distortion_vs_flowmatch_m", 4),
                         ("mean distance to POSTHOC_PROJ (m): 0 = method equals projecting the unconstrained output afterwards",
                          "dist_to_posthoc_m", 5)):
        out += ["", f"## {title}", "", "| method | " + " | ".join(have) + " | AVG |", "| --- |" + " --- |" * (len(have) + 1)]
        for m in methods:
            out.append(f"| {m} | " + " | ".join(cell(s, m, k, nd) for s in have) + f" | {avg(m, k, nd)} |")
    out += paired_section(root, have, methods)
    (root / "BENCH_REPORT.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
