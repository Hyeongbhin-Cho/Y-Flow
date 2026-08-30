# -*- coding: utf-8 -*-
# model/base.py
"""Velocity field interface: v_theta(x, t)."""

from __future__ import annotations

from abc import abstractmethod

import torch
from omegaconf import DictConfig
from torch import nn


class VelocityNet(nn.Module):
    """Maps state x and time t to a velocity of the same shape as x."""

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D]
            t: [B] or [B, 1], values in [0, 1]
        Returns:
            v: [B, D]
        """
        raise NotImplementedError


def build_model(cfg: DictConfig) -> VelocityNet:
    name = str(cfg.model.name)
    if name == "mlp":
        from model.mlp import VelocityMLP

        return VelocityMLP(
            dim=int(cfg.model.get("dim", 2)),
            hidden=tuple(int(h) for h in cfg.model.hidden),
            time_embed_dim=int(cfg.model.time_embed_dim),
        )
    raise KeyError(f"unknown model {name!r}")
