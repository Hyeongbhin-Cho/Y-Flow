# -*- coding: utf-8 -*-
# eval/clevrer_y_flow.py
"""Training-free YFlow on video-conditioned CLEVRER packed state."""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from data.base import BaseConstraint
from eval.y_flow import solve_terminal_pgd
from train.flow_match import ConditionalFlowMatching


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
    yflow_cfg = cfg.get("yflow", {})
    t_on = float(yflow_cfg.get("t_on", 0.5))
    lambda_oc = float(yflow_cfg.get("lambda_oc", 10.0))
    mu_val = float(yflow_cfg.get("mu", 1.0))
    delta = float(yflow_cfg.get("delta", 0.1))
    gamma_max = float(yflow_cfg.get("gamma_max", 1.0))
    max_iter = int(yflow_cfg.get("max_iter", 10))
    safety_buffer = float(yflow_cfg.get("safety_buffer", 1e-4))
    steps = int(cfg.sample.n_steps)
    dt = 1.0 / steps
    x = x0
    model.eval()
    for i in range(steps):
        t = i / steps
        t_tensor = torch.full((x.shape[0],), t, device=x0.device, dtype=x0.dtype)
        v = cfm.velocity(model, x, t_tensor, cond=cond)
        x1_raw = x + (1.0 - t) * v
        is_terminal_step = i == steps - 1
        if not is_terminal_step and t < t_on:
            eta = dt / max(1.0 - t, 1e-8)
            x = (1.0 - eta) * x + eta * x1_raw
            continue
        p_raw = x1_raw * std_t + mean_t
        p_phys = constraint.project_physical(p_raw)
        z_phys = (p_phys - mean_t) / std_t
        L_P = constraint.estimate_lipschitz(p_raw)
        g_val = gamma_max * min(max((t - t_on) / max(1.0 - t_on, 1e-8), 0.0), 1.0)
        gamma = torch.where(
            L_P <= 1.0 + delta,
            torch.full_like(L_P, g_val),
            torch.zeros_like(L_P),
        )
        lam = lambda_oc * (t ** 2) / max(dt, 1e-8)
        n_iters = max_iter if is_terminal_step else max(3, max_iter // 2)
        with torch.enable_grad():
            z_star = solve_terminal_pgd(
                x1_raw,
                z_phys,
                mean_t,
                std_t,
                constraint=constraint,
                lam=lam,
                mu=mu_val,
                n_iters=n_iters,
                buffer=safety_buffer,
            )
        if not is_terminal_step:
            mask = (gamma == 0.0).unsqueeze(-1)
            z_star = torch.where(mask, x1_raw, z_star)
        eta = 1.0 if is_terminal_step else dt / max(1.0 - t, 1e-8)
        x = (1.0 - eta) * x + eta * z_star
    return x
