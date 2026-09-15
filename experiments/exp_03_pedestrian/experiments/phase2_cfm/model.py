# -*- coding: utf-8 -*-
# experiments/exp_03_pedestrian/experiments/phase2_cfm/model.py
"""Plain conditional flow matching for ETH/UCY with the MoFlow sampler interface.

Unlike the MoFlow teacher (K query heads, near-deterministic per head), this
model draws every one of the K futures from independent Gaussian noise and
learns the rectified-flow velocity v(x_t, t | past). Diversity therefore lives
in the flow state, so inference-time guidance on x_t can change the outcome.

The sampler exposes the attributes the Phase-1 runner uses from
models.flow_matching.FlowMatcher: ``model_predictions``, ``bwd_sample_t``,
``sample`` and ``device``. Data are the upstream ETHDataset tensors
(min-max normalized, rotated), so constraints and metrics are shared.
"""

from __future__ import annotations

import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn

ModelPrediction = namedtuple("ModelPrediction", ["pred_vel", "pred_data", "pred_score"])


def sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    ang = (t.float() * 1000.0)[:, None] * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.film = nn.Linear(cond_dim, 2 * dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(c).chunk(2, dim=-1)
        return h + self.mlp(self.norm(h) * (1.0 + scale) + shift)


class CondFlowNet(nn.Module):
    def __init__(self, past_frames: int = 8, past_feat: int = 6, out_dim: int = 24,
                 hidden: int = 512, cond_dim: int = 256, n_blocks: int = 6, t_dim: int = 128):
        super().__init__()
        self.t_dim = t_dim
        self.ctx = nn.Sequential(nn.Linear(past_frames * past_feat, cond_dim), nn.SiLU(),
                                 nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.tmlp = nn.Sequential(nn.Linear(t_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.inp = nn.Linear(out_dim, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden, cond_dim) for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, y: torch.Tensor, t: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        """y [B, K, A, D], t [B], past [B, A, P, 6] -> velocity [B, K, A, D]."""
        B, K, A, _ = y.shape
        c = self.ctx(past.reshape(B, A, -1))[:, None] + self.tmlp(sinusoidal(t, self.t_dim))[:, None, None]
        c = c.expand(B, K, A, -1)
        h = self.inp(y)
        for blk in self.blocks:
            h = blk(h, c)
        return self.out(h)


def time_grid(solver: str, steps: int, lin_poly_p: int = 5, lin_poly_long_step: int = 1000):
    if solver == "euler":
        dt = 1.0 / steps
        return [dt * i for i in range(steps)], [dt] * steps
    n_lin = steps // 2
    n_poly = steps - n_lin
    dt_lin = 1.0 / lin_poly_long_step
    t_lin = [dt_lin * i for i in range(n_lin)]
    a = t_lin[-1] + dt_lin
    pts = [a + (1.0 - a) * ((i - 1) ** lin_poly_p) / ((n_poly) ** lin_poly_p) for i in range(1, n_poly + 2)]
    dt_poly = list(np.diff(pts))
    return t_lin + pts[:-1], [dt_lin] * n_lin + dt_poly


class CFMSampler(nn.Module):
    def __init__(self, cfg, net: CondFlowNet):
        super().__init__()
        self.cfg = cfg
        self.model = net
        self.out_dim = int(cfg.future_frames) * 2
        self.num_agents = int(cfg.agents)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def model_predictions(self, y_t, x_data, t, flag_print=False):
        v = self.model(y_t, t, x_data["past_traj"])
        tt = t.reshape(-1, 1, 1, 1).to(y_t.dtype)
        return ModelPrediction(v, y_t + (1.0 - tt) * v, None)

    def bwd_sample_t(self, y_t, t, dt, x_data, flag_print=False):
        B = y_t.shape[0]
        bt = torch.full((B,), t, device=y_t.device, dtype=torch.float)
        preds = self.model_predictions(y_t, x_data, bt)
        return y_t + preds.pred_vel * dt, preds.pred_data, preds

    @torch.no_grad()
    def sample(self, x_data, num_trajs, return_all_states=False):
        B = int(x_data["batch_size"])
        y = torch.randn((B, num_trajs, self.num_agents, self.out_dim), device=self.device)
        ts, dts = time_grid(self.cfg.solver, int(self.cfg.sampling_steps),
                            int(self.cfg.get("lin_poly_p", 5)), int(self.cfg.get("lin_poly_long_step", 1000)))
        for t, dt in zip(ts, dts):
            y, _, _ = self.bwd_sample_t(y, float(t), float(dt), x_data)
        return y, None, None, None, None
