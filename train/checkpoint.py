# -*- coding: utf-8 -*-
# train/checkpoint.py

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

from train.ema import EMA


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    ema: EMA | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location=None,
) -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if ema is not None and "ema" in payload:
        ema.load_state_dict(payload["ema"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
