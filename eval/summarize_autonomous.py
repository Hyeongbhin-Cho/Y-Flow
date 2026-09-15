"""Aggregate repeated AV2 evaluations into mean and sample standard deviation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from utils.paths import ROOT


PRIMARY = (
    "minADE",
    "minFDE",
    "top1ADE",
    "top1FDE",
    "safe_ratio",
    "speed_viol_rate",
    "accel_viol_rate",
    "intervention_ADE_m",
    "inference_time_s",
)


def summarize(run_names: list[str]) -> dict:
    if len(run_names) < 2:
        raise ValueError("at least two evaluation runs are required")
    collected: dict[str, list[dict]] = {}
    signatures: set[tuple[int, int, int]] = set()
    for run_name in run_names:
        root = ROOT / "runs" / run_name
        if not root.is_dir():
            raise FileNotFoundError(f"missing run directory: {root}")
        for path in sorted(root.glob("*/metrics.json")):
            metrics = json.loads(path.read_text())
            method = str(metrics["method"])
            collected.setdefault(method, []).append(metrics)
            signatures.add(
                (int(metrics["n_scenarios"]), int(metrics["n_modes"]), int(metrics["n_steps"]))
            )
    if len(signatures) != 1:
        raise ValueError(f"evaluation shapes or step counts differ: {sorted(signatures)}")

    methods = {}
    for method, records in sorted(collected.items()):
        if len(records) != len(run_names):
            raise ValueError(f"{method} has {len(records)} records, expected {len(run_names)}")
        stats = {}
        for key in PRIMARY:
            values = [float(record[key]) for record in records if key in record]
            if len(values) == len(records):
                stats[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                }
        methods[method] = stats
    return {
        "runs": run_names,
        "n_repeats": len(run_names),
        "evaluation_signature": list(next(iter(signatures))),
        "methods": methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", default="runs/exp_05_multiseed_summary.json")
    args = parser.parse_args()
    payload = summarize(args.runs)
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
