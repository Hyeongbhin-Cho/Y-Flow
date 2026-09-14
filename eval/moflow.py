"""Evaluate conditional K-shot trajectory forecasts from the AV2 MoFlow teacher."""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from omegaconf import DictConfig

from data.base import build_dataset, denormalize
from model.moflow import build_moflow_model
from train.checkpoint import load_checkpoint
from train.ema import EMA
from utils.device import get_device
from utils.paths import method_dir


def _context_to_device(context, device):
    return {name: torch.as_tensor(value, device=device) for name, value in context.items()}


@torch.no_grad()
def sample_moflow(model, context, n_modes: int, dim: int, n_steps: int, seed: int):
    device = next(model.parameters()).device
    batch = next(iter(context.values())).shape[0]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    x = torch.randn(batch, n_modes, dim, generator=generator).to(device)
    feature = model.encode_context(context)
    logits = None
    dt = 1.0 / n_steps
    model.eval()
    for step in range(n_steps):
        t = torch.full((batch,), step / n_steps, device=device, dtype=x.dtype)
        velocity, logits = model(x, t, context, context_feature=feature)
        x = x + dt * velocity
    final_t = torch.ones(batch, device=device, dtype=x.dtype)
    _, logits = model(x, final_t, context, context_feature=feature)
    return x, logits


def trajectory_metrics(
    predictions: np.ndarray,
    target: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, float]:
    pred = predictions.reshape(*predictions.shape[:2], -1, 2)
    truth = target.reshape(target.shape[0], -1, 2)
    distance = np.linalg.norm(pred - truth[:, None], axis=-1)
    ade = distance.mean(axis=-1)
    fde = distance[..., -1]
    metrics = {
        "minADE": float(ade.min(axis=1).mean()),
        "minFDE": float(fde.min(axis=1).mean()),
        "meanADE": float(ade.mean()),
        "meanFDE": float(fde.mean()),
    }
    if probabilities is not None:
        chosen = np.asarray(probabilities).argmax(axis=1)
        rows = np.arange(pred.shape[0])
        metrics["top1ADE"] = float(ade[rows, chosen].mean())
        metrics["top1FDE"] = float(fde[rows, chosen].mean())
    return metrics


def run_eval_moflow(cfg: DictConfig, device: torch.device | None = None) -> dict:
    device = device or get_device(cfg)
    bundle = build_dataset(cfg)
    if bundle.eval_context is None:
        raise ValueError("moflow requires evaluation context")
    model = build_moflow_model(cfg).to(device)
    ema = EMA(model, decay=float(cfg.train.ema_decay))
    checkpoint = method_dir(cfg, "moflow") / "last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing moflow checkpoint: {checkpoint}")
    load_checkpoint(checkpoint, model, ema=ema, map_location=device)
    ema.copy_to(model)
    context = _context_to_device(bundle.eval_context, device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    predictions_z, logits = sample_moflow(
        model, context, int(cfg.moflow.n_modes), int(cfg.model.dim),
        int(cfg.sample.n_steps), int(cfg.seed),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    predictions = denormalize(predictions_z, bundle.mean, bundle.std).cpu().numpy()
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
    metrics = trajectory_metrics(predictions, bundle.eval_raw, probabilities)
    h = bundle.constraint.h(predictions.reshape(-1, predictions.shape[-1]))
    stacked = np.stack(list(h.values()), axis=-1)
    safe = (stacked <= 0).all(axis=-1).reshape(predictions.shape[:2])
    top1 = probabilities.argmax(axis=1)
    rows = np.arange(predictions.shape[0])
    metrics["safe_ratio"] = float(safe.mean())
    metrics["oracle_safe_ratio"] = float(safe.any(axis=1).mean())
    metrics["top1_safe_ratio"] = float(safe[rows, top1].mean())
    for name, value in h.items():
        metrics[f"{name}_viol_rate"] = float((value > 0).mean())
        metrics[f"{name}_viol_mean"] = float(np.maximum(value, 0).mean())
    metrics.update({
        "method": "moflow", "run_name": str(cfg.run_name),
        "n_scenarios": int(predictions.shape[0]), "n_modes": int(predictions.shape[1]),
        "n_steps": int(cfg.sample.n_steps), "inference_time_s": float(elapsed),
    })
    out_dir = method_dir(cfg, "moflow")
    np.save(out_dir / "eval_predictions.npy", predictions)
    np.save(out_dir / "eval_probabilities.npy", probabilities)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics
