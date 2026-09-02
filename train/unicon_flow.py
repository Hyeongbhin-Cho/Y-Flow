# -*- coding: utf-8 -*-
# train/unicon_flow.py
"""UniConFlow is training-free and reuses the FlowMatch backbone."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from utils.paths import flowmatch_ckpt


def ensure_flowmatch_ckpt(cfg: DictConfig) -> Path:
    path = flowmatch_ckpt(cfg)
    if path.is_file():
        print(f"pretrained flowmatch model already exists: {path}")
        return path
    from train.trainer import run_train

    print(f"missing {path}, training flowmatch")
    return run_train(cfg, method="flowmatch")
