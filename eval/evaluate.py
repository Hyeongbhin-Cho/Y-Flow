# -*- coding: utf-8 -*-
# eval/evaluate.py

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

from constraints.swiss_roll import SwissRollConstraint
from data.swiss_roll import build_swiss_roll, denormalize
from eval.metrics import evaluate_points
from model import build_model
from sample.euler import EulerSampler
from train.checkpoint import load_checkpoint
from train.ema import EMA
from train.flow_match import ConditionalFlowMatching
from utils.device import get_device
from utils.paths import ROOT, flowmatch_ckpt, method_dir, run_name_of


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


def _sample_flowmatch(cfg: DictConfig, device: torch.device, x0: torch.Tensor) -> torch.Tensor:
    ckpt_path = flowmatch_ckpt(cfg)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing flowmatch checkpoint: {ckpt_path}")
    model = build_model(cfg).to(device)
    ema = EMA(model, decay=float(cfg.train.ema_decay))
    load_checkpoint(ckpt_path, model, ema=ema, map_location=device)
    ema.copy_to(model)
    model.eval()
    method = ConditionalFlowMatching(cfg)
    sampler = EulerSampler(n_steps=int(cfg.sample.n_steps))
    return sampler.sample(model, method, x0)


_SAMPLERS = {
    "flowmatch": _sample_flowmatch,
}


def run_eval(cfg: DictConfig, method: str, device: torch.device | None = None) -> dict:
    if method not in _SAMPLERS:
        raise NotImplementedError(f"{method} eval is not implemented yet")

    device = device or get_device(cfg)
    bundle = build_swiss_roll(cfg)
    n = int(cfg.sample.n_samples)
    dim = int(cfg.model.get("dim", 2))
    x0 = torch.randn(n, dim, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    z = _SAMPLERS[method](cfg, device, x0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    p = denormalize(z, bundle["mean"], bundle["std"]).detach().cpu().numpy()
    constraint = SwissRollConstraint(bundle["meta"])
    metrics = evaluate_points(p, bundle["eval_raw"], constraint)
    metrics.update(
        {
            "method": method,
            "run_name": str(cfg.run_name),
            "n_samples": n,
            "n_steps": int(cfg.sample.n_steps),
            "inference_time_s": float(elapsed),
            "inference_time_s_per_1k": float(elapsed / max(n / 1000.0, 1e-12)),
        }
    )

    out_dir = method_dir(cfg, method)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_scatter(out_dir / "eval_samples.png", p, bundle["train_raw"], f"{method} eval")
    np.save(out_dir / "eval_samples.npy", p)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
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
