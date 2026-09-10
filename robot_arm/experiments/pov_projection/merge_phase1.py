"""Merge independent Phase-1 task shards and regenerate aggregate artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from safe_flow_mpc.RobotModel import RobotModel
from safe_flow_mpc.SafeFlowMPC.ObstacleManager import ObstacleManager

from .run_phase1 import (
    bootstrap_summary,
    plot_bars,
    plot_cases,
    plot_pareto,
    plot_pov_steps,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()

    raw_frames = []
    step_frames = []
    trajectories: dict[str, np.ndarray] = {}
    configs = []
    for shard in args.inputs:
        results = shard / "results"
        raw_frames.append(pd.read_csv(results / "raw_results.csv"))
        step_frames.append(pd.read_csv(results / "per_step_pov.csv"))
        with np.load(results / "trajectories.npz") as archive:
            for key in archive.files:
                if key in trajectories:
                    raise ValueError(f"duplicate trajectory key: {key}")
                trajectories[key] = archive[key]
        configs.append(json.loads((results / "config.json").read_text()))

    raw = pd.concat(raw_frames, ignore_index=True)
    steps = pd.concat(step_frames, ignore_index=True)
    expected_rows = len(set(raw.task_id)) * len(set(raw.seed)) * 4
    if len(raw) != expected_rows:
        raise ValueError(f"expected {expected_rows} raw rows, found {len(raw)}")
    if raw.duplicated(["method", "task_id", "seed"]).any():
        raise ValueError("duplicate method/task/seed rows in shards")

    results_dir = args.output / "results"
    plots_dir = args.output / "plots"
    visualizations_dir = args.output / "visualizations"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = bootstrap_summary(raw, args.bootstrap_samples)
    raw.to_csv(results_dir / "raw_results.csv", index=False)
    steps.to_csv(results_dir / "per_step_pov.csv", index=False)
    summary.to_csv(results_dir / "summary.csv", index=False)
    np.savez_compressed(results_dir / "trajectories.npz", **trajectories)

    config = configs[0]
    config["experiment"] = "phase1_id_one_shot_merged_shards"
    config["tasks"] = sorted({int(value) for cfg in configs for value in cfg["tasks"]})
    config["seeds_per_task"] = len(set(raw.seed))
    config["bootstrap_samples"] = args.bootstrap_samples
    config["shard_count"] = len(args.inputs)
    (results_dir / "config.json").write_text(json.dumps(config, indent=2))

    robot_model = RobotModel()
    obstacle_manager = ObstacleManager()
    obstacle_manager.add_default_obstacles()
    plot_bars(summary, plots_dir)
    plot_pov_steps(steps, plots_dir)
    plot_pareto(summary, plots_dir)
    plot_cases(raw, trajectories, robot_model, obstacle_manager, visualizations_dir)
    write_report(raw, summary, args.output)
    print(
        f"merged {len(args.inputs)} shards: {len(raw)} raw rows, "
        f"{len(steps)} per-step rows -> {args.output}"
    )


if __name__ == "__main__":
    main()
