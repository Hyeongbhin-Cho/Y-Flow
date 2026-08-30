# -*- coding: utf-8 -*-
# eval/flow_match.py

from __future__ import annotations

import torch
from omegaconf import DictConfig

from eval._backbone import load_frozen_velocity
from sample.euler import EulerSampler


def sample(cfg: DictConfig, device: torch.device, x0: torch.Tensor) -> torch.Tensor:
    model, method = load_frozen_velocity(cfg, device)
    sampler = EulerSampler(n_steps=int(cfg.sample.n_steps))
    return sampler.sample(model, method, x0)
