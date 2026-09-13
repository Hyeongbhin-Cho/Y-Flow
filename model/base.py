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


def build_model(cfg: DictConfig) -> nn.Module:
    name = str(cfg.model.name)
    if name == "mlp":
        from model.mlp import VelocityMLP

        return VelocityMLP(
            dim=int(cfg.model.get("dim", 2)),
            hidden=tuple(int(h) for h in cfg.model.hidden),
            time_embed_dim=int(cfg.model.time_embed_dim),
        )
    if name in ("wan2.1", "wan", "wan_transformer"):
        from model.wan import build_wan_model

        return build_wan_model(cfg)
    if name in ("clevrer_flow", "clevrer_recognition"):
        from model.clevrer_flow import build_clevrer_flow_model

        return build_clevrer_flow_model(cfg)
    if name in ("clevrer_resnet34", "clevrer_video_recognizer"):
        from model.clevrer_recognition import build_clevrer_resnet34_model

        return build_clevrer_resnet34_model(cfg)
    raise KeyError(f"unknown model {name!r}")
