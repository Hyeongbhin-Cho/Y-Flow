# -*- coding: utf-8 -*-
# eval/clevrer_recognition.py
"""Evaluate video-conditioned CLEVRER state recognition (FlowMatch baseline)."""

from __future__ import annotations

import json

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data.base import build_dataset
from data.clevrer_state import (
    collate_clevrer_recognition,
    denormalize_state,
    unpack_state,
    layout_from_cfg,
)
from eval._backbone import load_frozen_velocity
from sample.euler import EulerSampler
from utils.device import get_device
from utils.paths import method_dir


def _attr_sets(parts: dict, active: np.ndarray) -> list[set[tuple[str, str, str]]]:
    from data.clevrer_state import CLEVRER_COLORS, CLEVRER_MATERIALS, CLEVRER_SHAPES

    colors = parts["color"].argmax(axis=-1)
    materials = parts["material"].argmax(axis=-1)
    shapes = parts["shape"].argmax(axis=-1)
    out = []
    for b in range(active.shape[0]):
        items = set()
        for k in range(active.shape[1]):
            if active[b, k] <= 0.5:
                continue
            items.add(
                (
                    CLEVRER_COLORS[int(colors[b, k])],
                    CLEVRER_MATERIALS[int(materials[b, k])],
                    CLEVRER_SHAPES[int(shapes[b, k])],
                )
            )
        out.append(items)
    return out


def _set_f1(pred: set, truth: set) -> float:
    if not pred and not truth:
        return 1.0
    tp = len(pred & truth)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(truth) if truth else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run_eval(cfg: DictConfig, method: str, device: torch.device | None = None) -> dict:
    if method != "flowmatch":
        raise NotImplementedError(
            f"{method} eval on clevrer_recognition is not implemented yet; "
            "train/eval the unconstrained FlowMatch baseline first"
        )
    device = device or get_device(cfg)
    bundle = build_dataset(cfg)
    loader = DataLoader(
        bundle.eval,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_clevrer_recognition,
    )
    model, cfm = load_frozen_velocity(cfg, device)
    sampler = EulerSampler(n_steps=int(cfg.sample.n_steps))
    layout = layout_from_cfg(cfg)
    mean = np.asarray(bundle.meta["mean"], dtype=np.float32)
    std = np.asarray(bundle.meta["std"], dtype=np.float32)
    constraint = bundle.constraint

    f1s, ades, safes = [], [], []
    n = 0
    limit = int(cfg.sample.get("n_samples", 100))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(cfg.seed))
    for batch in loader:
        if n >= limit:
            break
        video = batch["video"].to(device)
        gt = batch["state"].numpy()
        cond = model.encode(video)
        x0 = torch.randn(video.shape[0], layout.dim, generator=gen).to(device=device)
        z = sampler.sample(model, cfm, x0, cond=cond)
        pred = denormalize_state(z, mean, std).detach().cpu().numpy()
        h = constraint.h(pred)
        stacked = np.stack(list(h.values()), axis=0)
        safes.append(float((stacked.max(axis=0) <= 0).mean()))
        pred_parts = unpack_state(pred, layout)
        gt_parts = unpack_state(gt, layout)
        pred_active = pred_parts["vis"].max(axis=-1)
        gt_active = gt_parts["vis"].max(axis=-1)
        pred_sets = _attr_sets({k: np.asarray(v) for k, v in pred_parts.items()}, pred_active)
        gt_sets = _attr_sets({k: np.asarray(v) for k, v in gt_parts.items()}, gt_active)
        f1s.extend(_set_f1(p, t) for p, t in zip(pred_sets, gt_sets))
        vis = gt_parts["vis"]
        delta = np.linalg.norm(pred_parts["pos"] - gt_parts["pos"], axis=-1)
        mask = vis > 0.5
        if mask.any():
            ades.append(float(delta[mask].mean()))
        n += video.shape[0]

    metrics = {
        "method": method,
        "run_name": str(cfg.run_name),
        "n_samples": n,
        "n_steps": int(cfg.sample.n_steps),
        "safe_ratio": float(np.mean(safes)) if safes else 0.0,
        "attribute_f1": float(np.mean(f1s)) if f1s else 0.0,
        "ade_visible": float(np.mean(ades)) if ades else 0.0,
    }
    out_dir = method_dir(cfg, method)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return metrics
