# -*- coding: utf-8 -*-
# experiments/exp_03_pedestrian/experiments/phase2_cfm/report_study.py
"""Aggregate the Exp-03 study and evaluate the pre-declared decision gates.

Inputs  <root>/<model>/<setting>/<subset>/results/raw.csv  (run_phase1 --save_raw)
        <root>/probe/<model>/<subset>/summary.json
Outputs <root>/REPORT_STUDY.md, <root>/gates.json, <root>/tables/*.csv

Statistics: per scene, metrics are first averaged over seeds; paired deltas are
taken per scene; 95% CIs use a moving-block bootstrap over consecutive scene
indices (SocialGAN windows overlap by one frame, so scenes are not
independent). Gates are fixed in docs/exp/exp_03_pedestrian.md before running.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SUBSETS = ("eth", "hotel", "univ", "zara1", "zara2")
STRESS = ("kin_v12", "kin_v10")
PAIRS = (("YFLOW", "FINAL_PROJECTION"), ("POV_ALWAYS", "FINAL_PROJECTION"),
         ("FINAL_PROJECTION", "PLAIN_FM"), ("YFLOW_NO_REPLACE", "PLAIN_FM"), ("PLAIN_FM", "CONST_VEL"))
TABLE_METRICS = ("min_ade", "min_fde", "avg_ade", "avg_fde", "kde_nll", "col010_sample_rate", "col020_sample_rate",
                 "apd", "viol", "distortion_vs_plain_m", "proj_fail_rate", "latency_s")
METHOD_ORDER = ("CONST_VEL", "PLAIN_FM", "FINAL_PROJECTION", "YFLOW", "YFLOW_NO_REPLACE", "POV_ALWAYS", "YFLOW_D05")

G0_PROBE_M = 0.01
G0_DISTORTION_M = 0.01
G1_VIOL = 1e-3
G1_FAIL = 0.01
H1_MIN_SUBSETS = 3
H2_APD_RATIO = 0.95
H2_LATENCY_RATIO = 3.0
BLOCK = 50
N_BOOT = 2000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    return p.parse_args()


def load_run(path: Path) -> dict | None:
    raw = path / "results" / "raw.csv"
    if not raw.is_file():
        return None
    acc: dict = defaultdict(lambda: defaultdict(list))
    feas: dict = {}
    has_col = False
    with raw.open() as f:
        for r in csv.DictReader(f):
            m, sc = r["method"], int(r["scene"])
            feas[sc] = r["gt_feasible"] in ("True", "true", "1")
            has_col = has_col or ("all_viol_sample_rate" in r and r["all_viol_sample_rate"] != "")
            for k, v in r.items():
                if k in ("method", "scene", "seed", "gt_feasible") or v in ("", None):
                    continue
                acc[(m, sc)][k].append(float(v))
    per = defaultdict(dict)
    for (m, sc), d in acc.items():
        per[m][sc] = {k: (float(np.mean(f)) if (f := np.asarray(v)[np.isfinite(v)]).size else float("nan")) for k, v in d.items()}
        per[m][sc]["viol"] = per[m][sc].get("all_viol_sample_rate" if has_col else "viol_sample_rate", float("nan"))
    return {"per": per, "feasible": feas, "col": has_col}


def block_bootstrap(d: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = d.size
    if n == 0:
        return float("nan"), float("nan")
    nb = max(1, int(np.ceil(n / BLOCK)))
    starts = np.arange(0, n, BLOCK)
    blocks = [d[s:s + BLOCK] for s in starts]
    sizes = np.array([b.size for b in blocks])
    sums = np.array([b.sum() for b in blocks])
    idx = rng.integers(0, len(blocks), size=(N_BOOT, nb))
    stats = sums[idx].sum(1) / sizes[idx].sum(1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def table_row(run: dict, method: str, feasible_only: bool) -> dict:
    scenes = sorted(run["per"].get(method, {}))
    if feasible_only:
        scenes = [s for s in scenes if run["feasible"].get(s, False)]
    out = {"n": len(scenes)}
    for k in TABLE_METRICS:
        vals = np.array([run["per"][method][s].get(k, np.nan) for s in scenes], dtype=float)
        out[k] = float(np.nanmean(vals)) if vals.size and np.isfinite(vals).any() else float("nan")
    return out


def paired(run: dict, a: str, b: str, metric: str, rng) -> dict:
    if a not in run["per"] or b not in run["per"]:
        return {}
    scenes = sorted(s for s in run["per"][a] if s in run["per"][b] and run["feasible"].get(s, False))
    d = np.array([run["per"][a][s][metric] - run["per"][b][s][metric] for s in scenes], dtype=float)
    d = d[np.isfinite(d)]
    lo, hi = block_bootstrap(d, rng)
    return {"n": int(d.size), "mean": float(d.mean()) if d.size else float("nan"), "ci_low": lo, "ci_high": hi}


def fmt(x, nd=4):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rng = np.random.default_rng(0)
    runs = {}
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name in ("cfm", "moflow")):
        for setting_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            for sub_dir in sorted(p for p in setting_dir.iterdir() if p.is_dir()):
                r = load_run(sub_dir)
                if r is not None:
                    runs[(model_dir.name, setting_dir.name, sub_dir.name)] = r
    probes = {}
    for f in root.glob("probe/*/*/summary.json"):
        probes[(f.parent.parent.name, f.parent.name)] = json.loads(f.read_text())

    (root / "tables").mkdir(exist_ok=True)
    table_rows, pair_rows = [], []
    for (model, setting, sub), run in sorted(runs.items()):
        for m in sorted(run["per"]):
            for feas in (False, True):
                t = table_row(run, m, feas)
                table_rows.append({"model": model, "setting": setting, "subset": sub, "method": m,
                                   "scenes": "gt_feasible" if feas else "all", **t})
        for a, b in PAIRS:
            for metric in ("min_ade", "min_fde", "apd"):
                pr = paired(run, a, b, metric, rng)
                if pr:
                    pair_rows.append({"model": model, "setting": setting, "subset": sub, "a": a, "b": b, "metric": metric, **pr})
    for name, rows in (("metrics", table_rows), ("paired", pair_rows)):
        if rows:
            with (root / "tables" / f"{name}.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    def get_pair(model, setting, sub, a, b, metric):
        return next((r for r in pair_rows if (r["model"], r["setting"], r["subset"], r["a"], r["b"], r["metric"]) ==
                     (model, setting, sub, a, b, metric)), None)

    def get_tab(model, setting, sub, method, scenes="all"):
        return next((r for r in table_rows if (r["model"], r["setting"], r["subset"], r["method"], r["scenes"]) ==
                     (model, setting, sub, method, scenes)), None)

    gates: dict = {"thresholds": {"G0_probe_m": G0_PROBE_M, "G0_distortion_m": G0_DISTORTION_M, "G1_viol": G1_VIOL,
                                  "G1_fail": G1_FAIL, "H1_min_subsets": H1_MIN_SUBSETS, "H2_apd_ratio": H2_APD_RATIO,
                                  "H2_latency_ratio": H2_LATENCY_RATIO, "block": BLOCK, "n_boot": N_BOOT}}
    g0 = {}
    for sub in SUBSETS:
        pr = probes.get(("cfm", sub), {})
        term = pr.get("terminal_change_m", {})
        probe_val = float(term.get("0.1", term.get(0.1, np.nan))) if term else float("nan")
        t = get_tab("cfm", "kin_v10", sub, "YFLOW_NO_REPLACE")
        dist = t["distortion_vs_plain_m"] if t else float("nan")
        g0[sub] = {"probe_terminal_delta0.1_m": probe_val, "no_replace_distortion_m": dist,
                   "pass": bool(probe_val >= G0_PROBE_M and dist >= G0_DISTORTION_M)}
    gates["G0_responsiveness_cfm"] = g0
    g0_pass = [s for s in SUBSETS if g0[s]["pass"]]

    g1 = {}
    for (model, setting, sub), run in runs.items():
        for m in ("FINAL_PROJECTION", "YFLOW", "POV_ALWAYS"):
            t = get_tab(model, setting, sub, m)
            if t:
                g1[f"{model}/{setting}/{sub}/{m}"] = bool(t["viol"] <= G1_VIOL and (not np.isfinite(t["proj_fail_rate"]) or t["proj_fail_rate"] <= G1_FAIL))
    gates["G1_safety"] = {"all_pass": all(g1.values()) if g1 else False, "failures": [k for k, v in g1.items() if not v]}

    h1 = {}
    for setting in STRESS:
        for a in ("YFLOW", "POV_ALWAYS"):
            wins, losses = [], []
            for sub in g0_pass:
                pr = get_pair("cfm", setting, sub, a, "FINAL_PROJECTION", "min_ade")
                if pr and pr["ci_high"] < 0:
                    wins.append(sub)
                if pr and pr["ci_low"] > 0:
                    losses.append(sub)
            h1[f"{setting}/{a}"] = {"better_subsets": wins, "worse_subsets": losses, "pass": len(wins) >= H1_MIN_SUBSETS}
    gates["H1_accuracy_vs_final_projection"] = h1

    apd_ratios, lat_ratios = [], []
    for setting in STRESS:
        for sub in SUBSETS:
            y, p = get_tab("cfm", setting, sub, "YFLOW"), get_tab("cfm", setting, sub, "PLAIN_FM")
            if y and p and p["apd"] > 0:
                apd_ratios.append(y["apd"] / p["apd"])
                lat_ratios.append(y["latency_s"] / p["latency_s"])
    h2_pass = bool(apd_ratios) and float(np.mean(apd_ratios)) >= H2_APD_RATIO and float(np.mean(lat_ratios)) <= H2_LATENCY_RATIO
    gates["H2_cost"] = {"apd_ratio_mean": float(np.mean(apd_ratios)) if apd_ratios else None,
                        "latency_ratio_mean": float(np.mean(lat_ratios)) if lat_ratios else None, "pass": h2_pass}

    h3 = {}
    for a in ("YFLOW", "POV_ALWAYS"):
        wins = [sub for sub in SUBSETS if (pr := get_pair("cfm", "col_r035", sub, a, "FINAL_PROJECTION", "min_ade")) and pr["ci_high"] < 0]
        h3[a] = {"better_subsets": wins, "pass": len(wins) >= H1_MIN_SUBSETS}
    gates["H3_planning_col_r035"] = h3

    h4 = {}
    for a in ("YFLOW", "POV_ALWAYS"):
        wins = [sub for sub in g0_pass if (pr := get_pair("cfm", "colcv_r020", sub, a, "FINAL_PROJECTION", "min_ade")) and pr["ci_high"] < 0]
        col = {}
        for sub in SUBSETS:
            t_a, t_p = get_tab("cfm", "colcv_r020", sub, a), get_tab("cfm", "colcv_r020", sub, "PLAIN_FM")
            if t_a and t_p:
                col[sub] = t_a["col020_sample_rate"] - t_p["col020_sample_rate"]
        h4[a] = {"better_subsets": wins, "pass": len(wins) >= H1_MIN_SUBSETS, "col020_minus_plain": col}
    gates["H4_social_colcv_r020"] = h4

    ctrl = {}
    for (model, setting, sub), run in runs.items():
        if model == "moflow":
            pr = get_pair(model, setting, sub, "YFLOW", "FINAL_PROJECTION", "min_ade")
            if pr:
                ctrl[f"{setting}/{sub}"] = pr["mean"]
    gates["control_moflow_yflow_minus_final_minade"] = ctrl

    any_h1 = any(v["pass"] for v in h1.values())
    partial_h1 = any(v["better_subsets"] for v in h1.values())
    if len(g0_pass) < H1_MIN_SUBSETS:
        verdict = "INCONCLUSIVE"
    elif gates["G1_safety"]["all_pass"] and any_h1 and h2_pass:
        verdict = "SUPPORT"
    elif gates["G1_safety"]["all_pass"] and partial_h1:
        verdict = "LIMITED_SUPPORT"
    else:
        verdict = "NO_SUPPORT"
    gates["verdict"] = verdict
    (root / "gates.json").write_text(json.dumps(gates, indent=2, default=str))

    lines = ["# Exp-03 study report", "", f"**Verdict: {verdict}**", "", "Gates are defined in `docs/exp/exp_03_pedestrian.md`. Full numbers: `tables/metrics.csv`, `tables/paired.csv`, `gates.json`.", ""]
    lines += ["## G0 responsiveness (cfm)", "", "| subset | probe terminal Δ (δ=0.1) m | NO_REPLACE distortion m | pass |", "| --- | --- | --- | --- |"]
    for sub in SUBSETS:
        g = g0[sub]
        lines.append(f"| {sub} | {fmt(g['probe_terminal_delta0.1_m'])} | {fmt(g['no_replace_distortion_m'])} | {g['pass']} |")
    def avg_over_subsets(model, setting, m, key, scenes="all"):
        vals = [t[key] for sub in SUBSETS if (t := get_tab(model, setting, sub, m, scenes)) and np.isfinite(t[key])]
        return float(np.mean(vals)) if len(vals) == len(SUBSETS) else float("nan")

    def order(methods):
        known = [m for m in METHOD_ORDER if m in methods]
        return known + sorted(m for m in methods if m not in known)

    for model in ("cfm", "moflow"):
        settings = sorted({k[1] for k in runs if k[0] == model})
        for setting in settings:
            methods = order({m for (mo, se, su), r in runs.items() if mo == model and se == setting for m in r["per"]})
            lines += ["", f"## {model} / {setting}", "",
                      "min20 ADE / FDE (m), all test scenes (CONST_VEL: single deterministic prediction)", "",
                      "| method | " + " | ".join(SUBSETS) + " | AVG |", "| --- |" + " --- |" * (len(SUBSETS) + 1)]
            for m in methods:
                cells = []
                for sub in SUBSETS:
                    t = get_tab(model, setting, sub, m)
                    cells.append(f"{fmt(t['min_ade'], 3)} / {fmt(t['min_fde'], 3)}" if t else "–")
                cells.append(f"{fmt(avg_over_subsets(model, setting, m, 'min_ade'), 3)} / {fmt(avg_over_subsets(model, setting, m, 'min_fde'), 3)}")
                lines.append(f"| {m} | " + " | ".join(cells) + " |")
            lines += ["", "Pedestrian quality, mean over the five subsets (COL@r: share of samples within r of a co-present "
                      "agent's GT future; KDE-NLL: Scott KDE on K samples, log-pdf floor -20; viol: kinematic limits of this setting)", "",
                      "| method | minADE (GT-feasible) | COL@0.1 | COL@0.2 | KDE-NLL | APD | viol | latency s/scene |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- |"]
            for m in methods:
                lines.append(f"| {m} | {fmt(avg_over_subsets(model, setting, m, 'min_ade', 'gt_feasible'), 3)} | "
                             f"{fmt(avg_over_subsets(model, setting, m, 'col010_sample_rate'))} | "
                             f"{fmt(avg_over_subsets(model, setting, m, 'col020_sample_rate'))} | "
                             f"{fmt(avg_over_subsets(model, setting, m, 'kde_nll'), 3)} | "
                             f"{fmt(avg_over_subsets(model, setting, m, 'apd'), 3)} | "
                             f"{fmt(avg_over_subsets(model, setting, m, 'viol'))} | "
                             f"{fmt(avg_over_subsets(model, setting, m, 'latency_s'))} |")
            lines += ["", "Paired Δ minADE on GT-feasible scenes (m, block-bootstrap 95% CI):", "",
                      "| pair | " + " | ".join(SUBSETS) + " |", "| --- |" + " --- |" * len(SUBSETS)]
            for a, b in PAIRS:
                cells = []
                for sub in SUBSETS:
                    pr = get_pair(model, setting, sub, a, b, "min_ade")
                    cells.append(f"{fmt(pr['mean'])} [{fmt(pr['ci_low'])}, {fmt(pr['ci_high'])}] n={pr['n']}" if pr else "–")
                lines.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
    lines += ["", "## Gates", "", "```json", json.dumps({k: v for k, v in gates.items() if k != "thresholds"}, indent=2, default=str), "```"]
    (root / "REPORT_STUDY.md").write_text("\n".join(lines) + "\n")
    print(f"verdict {verdict}; wrote {root / 'REPORT_STUDY.md'}")


if __name__ == "__main__":
    main()
