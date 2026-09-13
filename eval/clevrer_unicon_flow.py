# -*- coding: utf-8 -*-
# eval/clevrer_unicon_flow.py
"""Training-free UniConFlow on video-conditioned CLEVRER packed state."""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from data.base import BaseConstraint
from eval.unicon_flow import constraint_jacobian, constraint_values, ptzf_reference, qp_guidance, terminal_refinement
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
    uc = cfg.get("uniconflow", {})
    free_until = float(uc.get("free_until", 0.0))
    ptzf_rate = float(uc.get("ptzf_rate", 1.0))
    gamma = float(uc.get("gamma", 1.0))
    slack_weight = float(uc.get("slack_weight", 100.0))
    max_guidance_norm = float(uc.get("max_guidance_norm", 20.0))
    terminal = bool(uc.get("terminal_refinement", True))
    safety_buffer = float(uc.get("safety_buffer", 1e-4))
    steps = int(cfg.sample.n_steps)
    dt = 1.0 / steps
    x = x0
    hbar0: torch.Tensor | None = None
    model.eval()
    for i in range(steps):
        t = i / steps
        t_tensor = torch.full((x.shape[0],), t, device=x0.device, dtype=x0.dtype)
        with torch.no_grad():
            v = cfm.velocity(model, x, t_tensor, cond=cond)
        with torch.enable_grad():
            z = x.detach().requires_grad_(True)
            h = constraint_values(z, mean_t, std_t, constraint=constraint)
            eta = constraint_jacobian(h, z)
        if hbar0 is None:
            hbar0 = torch.clamp(h.detach(), min=0.0) + safety_buffer
        if t < free_until:
            guidance = torch.zeros_like(v)
        else:
            local_t = (t - free_until) / max(1.0 - free_until, 1e-8)
            hbar, hbar_dot_local = ptzf_reference(hbar0, local_t, ptzf_rate)
            hbar_dot = hbar_dot_local / max(1.0 - free_until, 1e-8)
            nominal_dh = torch.einsum("bmd,bd->bm", eta, v)
            rho = nominal_dh - gamma * (hbar - h.detach()) - hbar_dot
            guidance = qp_guidance(rho, eta.detach(), slack_weight=slack_weight)
            norm = torch.linalg.vector_norm(guidance, dim=-1, keepdim=True)
            guidance = guidance * torch.clamp(
                max_guidance_norm / norm.clamp_min(1e-12), max=1.0
            )
        x = x + dt * (v + guidance)
    if terminal:
        x = terminal_refinement(x, mean_t, std_t, constraint=constraint, buffer=safety_buffer)
    return x.detach()
