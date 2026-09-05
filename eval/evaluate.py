# -*- coding: utf-8 -*-
# eval/evaluate.py

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

from data.base import build_dataset, denormalize
from eval.metrics import evaluate_points
from eval.sample_result import SampleResult
from utils.device import get_device
from utils.paths import ROOT, method_dir, run_name_of

_SAMPLE_MODULES = {
    "flowmatch": "eval.flow_match",
    "guideflow": "eval.guide_flow",
    "safeflow": "eval.safe_flow",
    "uniconflow": "eval.unicon_flow",
    "hardflow": "eval.hard_flow",
    "yflow": "eval.y_flow",
}


def _save_scatter(path: Path, points: np.ndarray, reference: np.ndarray | None, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    if reference is not None:
        ax.scatter(reference[:, 0], reference[:, 1], s=4, alpha=0.25, c="0.6", label="data")
    ax.scatter(points[:, 0], points[:, 1], s=6, alpha=0.7, c="C0", label="samples")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _sample_fn(method: str):
    if method not in _SAMPLE_MODULES:
        raise NotImplementedError(f"{method} eval is not implemented yet")
    mod = importlib.import_module(_SAMPLE_MODULES[method])
    return mod.sample


def make_eval_x0(cfg: DictConfig, device: torch.device) -> torch.Tensor:
    """Same (seed, n, dim) always yields the same x0, independent of other RNG use."""
    n = int(cfg.sample.n_samples)
    dim = int(cfg.model.get("dim", 2))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(cfg.seed))
    return torch.randn(n, dim, generator=gen).to(device=device)


def run_eval(cfg: DictConfig, method: str, device: torch.device | None = None) -> dict:
    sample = _sample_fn(method)
    device = device or get_device(cfg)
    bundle = build_dataset(cfg)
    x0 = make_eval_x0(cfg, device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    sample_output = sample(cfg, device, x0)
    if isinstance(sample_output, SampleResult):
        z = sample_output.samples
        diagnostics = sample_output.diagnostics
    else:
        z = sample_output
        diagnostics = {}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    p = denormalize(z, bundle["mean"], bundle["std"]).detach().cpu().numpy()
    constraint = bundle["constraint"]
    metrics = evaluate_points(p, bundle["eval_raw"], constraint)
    metrics.update(
        {
            "method": method,
            "run_name": str(cfg.run_name),
            "n_samples": int(x0.shape[0]),
            "n_steps": int(cfg.sample.n_steps),
            "inference_time_s": float(elapsed),
            "inference_time_s_per_1k": float(elapsed / max(x0.shape[0] / 1000.0, 1e-12)),
        }
    )
    metrics.update(diagnostics)

    out_dir = method_dir(cfg, method)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_scatter(out_dir / "eval_samples.png", p, bundle["train_raw"], f"{method} eval")
    np.save(out_dir / "eval_samples.npy", p)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    integrator = diagnostics.get("integrator")
    if integrator:
        tag = str(integrator).lower()
        _save_scatter(
            out_dir / f"eval_samples_{tag}.png",
            p,
            bundle["train_raw"],
            f"{method} {tag} eval",
        )
        np.save(out_dir / f"eval_samples_{tag}.npy", p)
        (out_dir / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2))
    write_run_metrics(cfg)
    print(json.dumps(metrics, indent=2))
    return metrics


def write_run_metrics(cfg: DictConfig) -> Path:
    """Merge per-command metrics into runs/{run_name}/metrics.json."""
    root = ROOT / "runs" / run_name_of(cfg)
    root.mkdir(parents=True, exist_ok=True)
    methods: dict[str, dict] = {}
    if root.is_dir():
        for child in sorted(root.iterdir()):
            path = child / "metrics.json"
            if child.is_dir() and path.is_file():
                methods[child.name] = json.loads(path.read_text())
    summary = {"run_name": run_name_of(cfg), "methods": methods}
    out = root / "metrics.json"
    out.write_text(json.dumps(summary, indent=2))
    return out
