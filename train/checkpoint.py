# -*- coding: utf-8 -*-
# train/checkpoint.py

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

from train.ema import EMA
from utils.paths import published_ckpt


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


def _is_foundation_weight_tree(directory: Path) -> bool:
    return (directory / "transformer").exists() or (directory / "vae").exists()


def publish_checkpoint(
    src: str | Path,
    dst: str | Path,
    extra: dict | None = None,
) -> Path:
    """Copy a run-local last.pt into a stable checkpoints/ path for reuse."""
    src = Path(src)
    dst = Path(dst)
    if not src.is_file():
        raise FileNotFoundError(f"missing checkpoint to publish: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copy2(src, tmp)
        tmp.replace(dst)
    cfg_src = src.parent / "config.yaml"
    if cfg_src.is_file():
        shutil.copy2(cfg_src, dst.parent / "config.yaml")
    ready = {
        "checkpoint": dst.name,
        "source": str(src),
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if extra:
        ready.update(extra)
    (dst.parent / "READY.json").write_text(json.dumps(ready, indent=2) + "\n")
    return dst


def maybe_publish_checkpoint(
    cfg,
    src: str | Path,
    extra: dict | None = None,
) -> Path | None:
    """Publish to model.local_dir/last.pt when set. Skip Wan HF weight trees."""
    dst = published_ckpt(cfg)
    if dst is None:
        return None
    if _is_foundation_weight_tree(dst.parent):
        return None
    return publish_checkpoint(src, dst, extra=extra)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    ema: EMA | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location=None,
) -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("extra", {}).get("is_foundation_model", False):
        return payload
    model.load_state_dict(payload["model"])
    if ema is not None and "ema" in payload:
        ema.load_state_dict(payload["ema"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
