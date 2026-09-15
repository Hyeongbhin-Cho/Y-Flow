"""MoFlow-style conditional K-shot velocity model for AV2 focal forecasting.

This is an AV2 adaptation: it preserves MoFlow's conditional flow and K-shot
best-of-K objective, while replacing the original dataset-specific scene encoder
with a compact focal/neighbor history encoder.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch import nn

from model.time_embed import SinusoidalTimeEmbedding


class HistoryEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_size=2, hidden_size=hidden_dim, batch_first=True)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        _, state = self.gru(history)
        return state[-1]


class MoFlowAV2(nn.Module):
    """Conditional velocity field v(x, t, context) with K proposal heads."""

    def __init__(
        self,
        dim: int = 120,
        hidden_dim: int = 256,
        time_embed_dim: int = 64,
        type_count: int = 5,
        type_embed_dim: int = 16,
        context_scale_m: float = 50.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.context_scale_m = float(context_scale_m)
        self.history_encoder = HistoryEncoder(hidden_dim)
        self.type_embedding = nn.Embedding(type_count + 1, type_embed_dim, padding_idx=0)
        self.neighbor_projection = nn.Linear(hidden_dim + type_embed_dim, hidden_dim)
        self.context_projection = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        self.velocity = nn.Sequential(
            nn.Linear(dim + time_embed_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim + dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def encode_context(self, context: dict[str, torch.Tensor]) -> torch.Tensor:
        focal = context["focal_history"] / self.context_scale_m
        neighbors = context["neighbor_history"] / self.context_scale_m
        mask = context["neighbor_mask"].bool()
        types = context["neighbor_types"].long()
        batch, count, steps, _ = neighbors.shape

        focal_feature = self.history_encoder(focal)
        neighbor_feature = self.history_encoder(neighbors.reshape(batch * count, steps, 2))
        neighbor_feature = neighbor_feature.reshape(batch, count, -1)
        neighbor_feature = self.neighbor_projection(
            torch.cat([neighbor_feature, self.type_embedding(types)], dim=-1)
        )
        weights = mask.unsqueeze(-1).to(neighbor_feature.dtype)
        pooled = (neighbor_feature * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.context_projection(torch.cat([focal_feature, pooled], dim=-1))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: dict[str, torch.Tensor],
        context_feature: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim not in (2, 3):
            raise ValueError(f"x must have shape [B,D] or [B,K,D], got {tuple(x.shape)}")
        feature = self.encode_context(context) if context_feature is None else context_feature
        if x.ndim == 3:
            k = x.shape[1]
            feature = feature[:, None, :].expand(-1, k, -1)
            if t.ndim == 1:
                t = t[:, None].expand(-1, k)
        time_feature = self.time_embedding(t.reshape(-1)).reshape(*x.shape[:-1], -1)
        velocity = self.velocity(torch.cat([x, time_feature, feature], dim=-1))
        # Confidence must see each proposal; context-only logits would be identical
        # across K and could not learn the best-of-K classification target.
        logits = self.confidence(torch.cat([feature, x], dim=-1)).squeeze(-1)
        return velocity, logits


def build_moflow_model(cfg: DictConfig) -> MoFlowAV2:
    settings = cfg.moflow
    return MoFlowAV2(
        dim=int(cfg.model.dim),
        hidden_dim=int(settings.get("hidden_dim", 256)),
        time_embed_dim=int(cfg.model.time_embed_dim),
        type_count=len(cfg.data.context.actor_types),
        type_embed_dim=int(settings.get("type_embed_dim", 16)),
        context_scale_m=float(settings.get("context_scale_m", cfg.data.context.radius_m)),
    )
