# -*- coding: utf-8 -*-
# eval/clevrer_hard_flow.py
"""Training-free HardFlow on video-conditioned CLEVRER packed state."""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from data.base import BaseConstraint
from eval.hard_flow import solve_terminal_pgd_hardflow
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
    hf_cfg = cfg.get("hardflow", {})
    t_on = float(hf_cfg.get("t_on", 0.5))
    lambda_oc = float(hf_cfg.get("lambda_oc", 10.0))
    max_iter = int(hf_cfg.get("max_iter", 10))
    safety_buffer = float(hf_cfg.get("safety_buffer", 1e-4))
    steps = int(cfg.sample.n_steps)
    dt = 1.0 / steps
    x = x0
    model.eval()
    for i in range(steps):
        t = i / steps
        t_next = (i + 1) / steps
        t_tensor = torch.full((x.shape[0],), t, device=x0.device, dtype=x0.dtype)
        v = cfm.velocity(model, x, t_tensor, cond=cond)
        bar_x = x + dt * v
        if t < t_on and i < steps - 1:
            x = bar_x
            continue
        t_next_tensor = torch.full((x.shape[0],), t_next, device=x0.device, dtype=x0.dtype)
        v_next = cfm.velocity(model, bar_x, t_next_tensor, cond=cond)
        bar_x1 = bar_x + (1.0 - t_next) * v_next
        lam = lambda_oc * (t_next ** 2) / max(dt, 1e-8)
        with torch.enable_grad():
            z_star = solve_terminal_pgd_hardflow(
                bar_x1,
                mean_t,
                std_t,
                constraint=constraint,
                lam=lam,
                n_iters=max_iter,
                buffer=safety_buffer,
            )
        w0 = bar_x - t_next * v_next
        x = t_next * z_star + (1.0 - t_next) * w0
    return x
