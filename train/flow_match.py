# -*- coding: utf-8 -*-
# train/flow_match.py
"""Unconstrained linear Conditional Flow Matching. No h, C, or P."""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch import nn


class ConditionalFlowMatching:
    def __init__(self, cfg: DictConfig | None = None, sigma_min: float = 0.0):
        if cfg is not None:
            sigma_min = float(cfg.method.get("sigma_min", sigma_min))
        self.sigma_min = sigma_min

    def interpolant(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = t.reshape(-1, *([1] * (x1.ndim - 1)))
        sigma = self.sigma_min
        xt = (1.0 - (1.0 - sigma) * t) * x0 + t * x1
        ut = x1 - (1.0 - sigma) * x0
        return xt, ut

    def training_losses(self, model: nn.Module, x1: torch.Tensor) -> torch.Tensor:
        t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        x0 = torch.randn_like(x1)
        xt, ut = self.interpolant(x0, x1, t)
        v = model(xt, t)
        return (v - ut).square().mean()

    def velocity(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return model(x, t)
