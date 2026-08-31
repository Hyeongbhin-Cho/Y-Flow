# -*- coding: utf-8 -*-
# model/cond_mlp.py
"""Conditional MLP velocity field for GuideFlow classifier-free guidance.

Implements v_theta(x, t, c) of Eq. (12): intent (plan anchor or driving command) and reward
are embedded, masked independently with probability p, and fused with the flow state.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from omegaconf import DictConfig
from torch import nn

from model.base import VelocityNet
from model.time_embed import SinusoidalTimeEmbedding


class ConditionalVelocityMLP(VelocityNet):
    def __init__(
        self,
        dim: int = 2,
        hidden: Sequence[int] = (128, 128, 128),
        time_embed_dim: int = 32,
        intent_dim: int = 2,
        cond_embed_dim: int = 32,
        use_reward: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.intent_dim = intent_dim
        self.use_reward = use_reward
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.intent_proj = nn.Linear(intent_dim, cond_embed_dim)
        self.null_intent = nn.Parameter(torch.zeros(cond_embed_dim))
        if use_reward:
            self.reward_proj = nn.Linear(1, cond_embed_dim)
            self.null_reward = nn.Parameter(torch.zeros(cond_embed_dim))

        layers: list[nn.Module] = []
        in_dim = dim + time_embed_dim + cond_embed_dim * (2 if use_reward else 1)
        for width in hidden:
            layers.extend([nn.Linear(in_dim, width), nn.SiLU()])
            in_dim = width
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)

    def _gate(self, embedded: torch.Tensor, null: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.reshape(-1, 1).to(dtype=embedded.dtype)
        return mask * embedded + (1.0 - mask) * null.unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        intent: torch.Tensor | None = None,
        reward: torch.Tensor | None = None,
        intent_mask: torch.Tensor | None = None,
        reward_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = x.shape[0]
        te = self.time_embed(t.to(device=x.device, dtype=x.dtype))
        if intent is None:
            intent = x.new_zeros(b, self.intent_dim)
            intent_mask = x.new_zeros(b)
        if intent_mask is None:
            intent_mask = x.new_ones(b)
        parts = [x, te, self._gate(self.intent_proj(intent), self.null_intent, intent_mask)]
        if self.use_reward:
            if reward is None:
                reward = x.new_zeros(b, 1)
                reward_mask = x.new_zeros(b)
            if reward_mask is None:
                reward_mask = x.new_ones(b)
            r = self._gate(self.reward_proj(reward.reshape(-1, 1)), self.null_reward, reward_mask)
            parts.append(r)
        return self.net(torch.cat(parts, dim=-1))


def build_cond_model(cfg: DictConfig, intent_dim: int) -> ConditionalVelocityMLP:
    g = cfg.guideflow.guidance
    return ConditionalVelocityMLP(
        dim=int(cfg.model.get("dim", 2)),
        hidden=tuple(int(h) for h in g.get("cond_hidden", [128, 128, 128])),
        time_embed_dim=int(cfg.model.time_embed_dim),
        intent_dim=intent_dim,
        cond_embed_dim=int(g.get("cond_embed_dim", 32)),
        use_reward=bool(g.get("reward", True)),
    )
