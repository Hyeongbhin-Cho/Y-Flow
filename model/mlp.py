# -*- coding: utf-8 -*-
# model/mlp.py
"""MLP velocity field for low-dimensional states (Exp-01 Swiss roll)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from model.base import VelocityNet
from model.time_embed import SinusoidalTimeEmbedding


class VelocityMLP(VelocityNet):
    def __init__(
        self,
        dim: int = 2,
        hidden: Sequence[int] = (64, 64, 64),
        time_embed_dim: int = 32,
    ):
        super().__init__()
        self.dim = dim
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        layers: list[nn.Module] = []
        in_dim = dim + time_embed_dim
        for width in hidden:
            layers.extend([nn.Linear(in_dim, width), nn.SiLU()])
            in_dim = width
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.time_embed(t.to(device=x.device, dtype=x.dtype))
        return self.net(torch.cat([x, te], dim=-1))
