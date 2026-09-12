# -*- coding: utf-8 -*-
# model/clevrer_flow.py
"""Video-conditioned velocity field for CLEVRER scene-state recognition."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from omegaconf import DictConfig
from torch import nn

from data.clevrer_state import layout_from_cfg
from model.base import VelocityNet
from model.time_embed import SinusoidalTimeEmbedding


class ClipEncoder(nn.Module):
    """Maps a clip [B, 3, T, H, W] in [-1, 1] to a condition vector [B, cond_dim]."""

    def __init__(self, cond_dim: int = 128):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Linear(128, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"expected video [B,3,T,H,W], got {tuple(video.shape)}")
        batch, _c, time, _h, _w = video.shape
        frames = video.permute(0, 2, 1, 3, 4).reshape(batch * time, 3, video.shape[-2], video.shape[-1])
        features = self.spatial(frames).flatten(1).reshape(batch, time, -1).mean(dim=1)
        return self.proj(features)


class CLEVRERVelocityNet(VelocityNet):
    """v_theta(S_t, t, E(V)) for packed scene state S."""

    def __init__(
        self,
        dim: int,
        hidden: Sequence[int] = (256, 256, 256),
        time_embed_dim: int = 64,
        cond_dim: int = 128,
    ):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.encoder = ClipEncoder(cond_dim=cond_dim)
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        layers: list[nn.Module] = []
        in_dim = dim + time_embed_dim + cond_dim
        for width in hidden:
            layers.extend([nn.Linear(in_dim, width), nn.SiLU()])
            in_dim = width
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        return self.encoder(video)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        video: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cond is None:
            if video is None:
                cond = x.new_zeros(x.shape[0], self.cond_dim)
            else:
                cond = self.encode(video)
        te = self.time_embed(t.to(device=x.device, dtype=x.dtype))
        return self.net(torch.cat([x, te, cond], dim=-1))


def build_clevrer_flow_model(cfg: DictConfig) -> CLEVRERVelocityNet:
    layout = layout_from_cfg(cfg)
    dim = layout.dim
    hidden = tuple(int(h) for h in cfg.model.get("hidden", [256, 256, 256]))
    return CLEVRERVelocityNet(
        dim=dim,
        hidden=hidden,
        time_embed_dim=int(cfg.model.get("time_embed_dim", 64)),
        cond_dim=int(cfg.model.get("cond_dim", 128)),
    )
