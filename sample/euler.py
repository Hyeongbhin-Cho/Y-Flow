# -*- coding: utf-8 -*-
# sample/euler.py
"""Fixed-step Euler ODE sampler. Unguided: x <- x + dt * v_theta(x, t)."""

from __future__ import annotations

import torch
from torch import nn

from train.flow_match import ConditionalFlowMatching


class EulerSampler:
    def __init__(self, n_steps: int = 100):
        self.n_steps = n_steps

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        method: ConditionalFlowMatching,
        x0: torch.Tensor,
        n_steps: int | None = None,
    ) -> torch.Tensor:
        steps = int(n_steps or self.n_steps)
        x = x0
        dt = 1.0 / steps
        was_training = model.training
        model.eval()
        for i in range(steps):
            t = torch.full((x.shape[0],), i / steps, device=x.device, dtype=x.dtype)
            v = method.velocity(model, x, t)
            x = x + dt * v
        if was_training:
            model.train()
        return x

    @torch.no_grad()
    def trajectory(
        self,
        model: nn.Module,
        method: ConditionalFlowMatching,
        x0: torch.Tensor,
        n_steps: int | None = None,
    ) -> torch.Tensor:
        steps = int(n_steps or self.n_steps)
        xs = [x0]
        x = x0
        dt = 1.0 / steps
        was_training = model.training
        model.eval()
        for i in range(steps):
            t = torch.full((x.shape[0],), i / steps, device=x.device, dtype=x.dtype)
            v = method.velocity(model, x, t)
            x = x + dt * v
            xs.append(x)
        if was_training:
            model.train()
        return torch.stack(xs, dim=1)
