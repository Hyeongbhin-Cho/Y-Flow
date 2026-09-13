# -*- coding: utf-8 -*-
# eval/clevrer_safe_flow.py
"""Training-free SafeFlow on video-conditioned CLEVRER packed state."""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from data.base import BaseConstraint, barrier_gain, solve_composite_fmbf
from train.flow_match import ConditionalFlowMatching


class _SafeVelocity:
    def __init__(self, model, method, fmbf, mean, std, cond, cfg):
        self.model = model
        self.method = method
        self.fmbf = fmbf
        self.mean = mean
        self.std = std
        self.cond = cond
        self.cfg = cfg

    def __call__(self, t: float, z: torch.Tensor, *, corrected: bool) -> torch.Tensor:
        t_batch = torch.full((z.shape[0],), float(t), device=z.device, dtype=z.dtype)
        v_z = self.method.velocity(self.model, z, t_batch, cond=self.cond)
        if not corrected or not bool(self.cfg.enabled):
            return v_z
        p = z * self.std + self.mean
        v_p = v_z * self.std
        h, gradients = self.fmbf.values_and_gradients(p)
        gain = barrier_gain(
            torch.as_tensor(t, device=z.device, dtype=z.dtype),
            h,
            phi0=float(self.cfg.phi0),
            schedule=str(self.cfg.phi_schedule),
            gamma=float(self.cfg.phi_gamma),
            omega=float(self.cfg.phi_omega),
            terminal_eps=float(self.cfg.terminal_eps),
        )
        a = (gradients * v_p.unsqueeze(-2)).sum(dim=-1) + gain * h
        solution = solve_composite_fmbf(
            a,
            gradients,
            slack_weight=float(self.cfg.slack_weight),
            active_tol=float(self.cfg.qp_active_tol),
        )
        return v_z + solution.correction / self.std


def sample_state(
    cfg: DictConfig,
    model: torch.nn.Module,
    cfm: ConditionalFlowMatching,
    x0: torch.Tensor,
    cond: torch.Tensor,
    constraint: BaseConstraint,
    mean,
    std,
) -> torch.Tensor:
    mean_t = torch.as_tensor(mean, device=x0.device, dtype=x0.dtype)
    std_t = torch.as_tensor(std, device=x0.device, dtype=x0.dtype)
    fmbf = constraint.get_fmbf(
        temperature=float(cfg.safeflow.get("smooth_box_temperature", 0.05)),
    )
    field = _SafeVelocity(model, cfm, fmbf, mean_t, std_t, cond, cfg.safeflow)
    steps = int(cfg.sample.n_steps)
    dt = 1.0 / steps
    t_on = float(cfg.safeflow.t_on)
    z = x0
    model.eval()
    for i in range(steps):
        t = i / steps
        z = z + dt * field(t, z, corrected=t >= t_on)
    if bool(cfg.safeflow.enabled) and bool(cfg.safeflow.terminal_filter.enabled):
        p = (z * std_t + mean_t).detach().cpu().numpy()
        p_final, _stats = fmbf.terminal_filter(
            p,
            max_iter=int(cfg.safeflow.terminal_filter.max_iter),
            ftol=float(cfg.safeflow.terminal_filter.ftol),
            constraint_tol=float(cfg.safeflow.terminal_filter.constraint_tol),
        )
        z = torch.from_numpy(p_final.astype(z.detach().cpu().numpy().dtype)).to(
            device=x0.device, dtype=x0.dtype
        )
        z = (z - mean_t) / std_t
    return z
