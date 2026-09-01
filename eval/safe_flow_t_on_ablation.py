# -*- coding: utf-8 -*-
# eval/safe_flow_t_on_ablation.py
"""Reproduce the Exp-01 SafeFlow t_on ablation without overwriting canonical evals."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from constraints.swiss_roll import SwissRollConstraint
from data.swiss_roll import build_swiss_roll, denormalize, nearest_u
from eval.evaluate import _save_scatter, make_eval_x0
from eval.metrics import evaluate_points
from eval.safe_flow import sample
from utils.device import get_device
from utils.paths import ROOT, flowmatch_ckpt, method_dir, run_name_of

DEFAULT_T_ON = (0.5, 0.7, 0.8, 0.9)


def _tag(value: float) -> str:
    return f"t_on_{value:g}".replace(".", "p")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _save_comparison(
    path: Path,
    samples: list[tuple[float, np.ndarray]],
    reference: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    for ax, (t_on, points) in zip(axes.flat, samples):
        ax.scatter(reference[:, 0], reference[:, 1], s=3, alpha=0.18, c="0.65")
        ax.scatter(points[:, 0], points[:, 1], s=5, alpha=0.65, c="C0")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"t_on={t_on:g}")
    fig.suptitle("SafeFlow t_on ablation")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_u_histogram(
    path: Path,
    samples: list[tuple[float, np.ndarray]],
    reference: np.ndarray,
    *,
    a: float,
    u_min: float,
    u_max: float,
) -> None:
    bins = np.linspace(u_min, u_max, 7)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    reference_u = nearest_u(reference, a, u_min, u_max)
    ax.hist(
        reference_u,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.0,
        color="black",
        label="data",
    )
    for t_on, points in samples:
        sample_u = nearest_u(points, a, u_min, u_max)
        ax.hist(
            sample_u,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            label=f"t_on={t_on:g}",
        )
    ax.set_xlabel("nearest spiral parameter u")
    ax.set_ylabel("density")
    ax.set_title("SafeFlow density along the spiral")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def run_ablation(
    cfg: DictConfig,
    t_on_values: tuple[float, ...] = DEFAULT_T_ON,
    *,
    output_dir: Path | None = None,
    device: torch.device | None = None,
) -> Path:
    if not t_on_values:
        raise ValueError("at least one t_on value is required")
    if len(t_on_values) != 4:
        raise ValueError("comparison plot requires exactly four t_on values")
    if any(value < 0.0 or value >= 1.0 for value in t_on_values):
        raise ValueError("t_on values must lie in [0, 1)")

    device = device or get_device(cfg)
    checkpoint = flowmatch_ckpt(cfg)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing FlowMatch checkpoint: {checkpoint}")

    bundle = build_swiss_roll(cfg)
    meta = bundle["meta"]
    constraint = SwissRollConstraint(meta)
    x0 = make_eval_x0(cfg, device)
    output_dir = output_dir or method_dir(cfg, "safeflow") / "t_on_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    generated: list[tuple[float, np.ndarray]] = []
    u_edges = np.linspace(float(meta.u_min), float(meta.u_max), 7)
    for t_on in t_on_values:
        run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        run_cfg.safeflow.t_on = float(t_on)

        _synchronize(device)
        started = time.perf_counter()
        result = sample(run_cfg, device, x0)
        _synchronize(device)
        elapsed = time.perf_counter() - started

        points = denormalize(
            result.samples, bundle["mean"], bundle["std"]
        ).cpu().numpy()
        u = nearest_u(
            points,
            float(meta.a),
            float(meta.u_min),
            float(meta.u_max),
        )
        metrics = evaluate_points(points, bundle["eval_raw"], constraint)
        metrics.update(
            {
                "method": "safeflow",
                "run_name": run_name_of(cfg),
                "integrator": str(run_cfg.safeflow.integrator),
                "t_on": float(t_on),
                "n_samples": int(x0.shape[0]),
                "n_steps": int(run_cfg.sample.n_steps),
                "inference_time_s": float(elapsed),
                "inference_time_s_per_1k": float(
                    elapsed / max(x0.shape[0] / 1000.0, 1.0e-12)
                ),
                "mean_u": float(u.mean()),
                "u_bin_edges": u_edges.tolist(),
                "u_bin_counts": np.histogram(u, bins=u_edges)[0].tolist(),
            }
        )
        metrics.update(result.diagnostics)

        setting_dir = output_dir / _tag(float(t_on))
        setting_dir.mkdir(parents=True, exist_ok=True)
        np.save(setting_dir / "eval_samples.npy", points)
        (setting_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        OmegaConf.save(run_cfg, setting_dir / "config.yaml")
        _save_scatter(
            setting_dir / "eval_samples.png",
            points,
            bundle["train_raw"],
            f"SafeFlow t_on={t_on:g}",
        )
        rows.append(metrics)
        generated.append((float(t_on), points))

    eval_u = nearest_u(
        bundle["eval_raw"],
        float(meta.a),
        float(meta.u_min),
        float(meta.u_max),
    )
    summary = {
        "run_name": run_name_of(cfg),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "device": str(device),
        "platform": platform.platform(),
        "seed": int(cfg.seed),
        "n_samples": int(x0.shape[0]),
        "integrator": str(cfg.safeflow.integrator),
        "t_on_values": [float(value) for value in t_on_values],
        "reference_mean_u": float(eval_u.mean()),
        "reference_u_bin_counts": np.histogram(eval_u, bins=u_edges)[0].tolist(),
        "results": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _save_comparison(output_dir / "comparison.png", generated, bundle["train_raw"])
    _save_u_histogram(
        output_dir / "u_histogram.png",
        generated,
        bundle["eval_raw"],
        a=float(meta.a),
        u_min=float(meta.u_min),
        u_max=float(meta.u_max),
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "exp_01_swiss_roll.yaml"),
    )
    parser.add_argument("--run_name", default="exp_01_swiss_roll")
    parser.add_argument("--integrator", choices=("euler", "dopri5"), default="euler")
    parser.add_argument("--t_on", type=float, nargs=4, default=DEFAULT_T_ON)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.run_name = args.run_name
    cfg.safeflow.integrator = args.integrator
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None and not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    result = run_ablation(cfg, tuple(args.t_on), output_dir=output_dir)
    print(result)


if __name__ == "__main__":
    main()
