# -*- coding: utf-8 -*-
# utils/device.py

from __future__ import annotations

import torch
from omegaconf import DictConfig


def get_device(cfg: DictConfig | str = "cuda") -> torch.device:
    name = cfg if isinstance(cfg, str) else str(cfg.get("device", "cuda"))
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
