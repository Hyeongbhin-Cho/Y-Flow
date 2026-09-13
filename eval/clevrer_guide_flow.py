# -*- coding: utf-8 -*-
# eval/clevrer_guide_flow.py
"""Training-free GuideFlow (CVF+CF+RFE) on video-conditioned CLEVRER packed state."""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig

from data.base import BaseConstraint
from eval.guide_flow import _constrain_velocity, _nearest_anchor, energy_weight
from train.flow_match import ConditionalFlowMatching

_EPS = 1e-12


def sample_state(
    cfg: DictConfig,
    model: torch.nn.Module,
    cfm: ConditionalFlowMatching,
    x0: torch.Tensor,
    cond: torch.Tensor,
    constraint: BaseConstraint,
    mean,
    std,
    anchors_z: torch.Tensor | None = None,
) -> torch.Tensor:
    mean_t = torch.as_tensor(mean, device=x0.device, dtype=x0.dtype)
    std_t = torch.as_tensor(std, device=x0.device, dtype=x0.dtype)
    mean_np = np.asarray(mean, dtype=np.float64).reshape(-1)
    std_np = np.asarray(std, dtype=np.float64).reshape(-1)
    gf = cfg.guideflow
    use_cvf = bool(gf.get("cvf", True)) and anchors_z is not None
    use_cf = bool(gf.get("cf", True)) and anchors_z is not None
    cf_mode = str(gf.get("cf_mode", "interp"))
    use_rfe = bool(gf.get("rfe", True))
    lam = float(gf.get("lambda_cvf", 0.1))
    t_on = float(gf.get("cvf_t_on", 0.0))
    k_c = int(gf.get("k_c", 25))
    tau_star = float(gf.get("tau_star", 0.5))
    eta_max = float(gf.get("eta_max", 0.5))
    n_refine = int(gf.get("n_refine", 10))
    slack = float(gf.get("slack", 0.01))
    del slack

    def velocity(x, t_tensor):
        return cfm.velocity(model, x, t_tensor, cond=cond)

    def refine(x: torch.Tensor, eta: float) -> torch.Tensor:
        p = (x * std_t + mean_t).detach().cpu().numpy()
        g = constraint.energy_grad(p)
        z = ((p - eta * g) - mean_np) / std_np
        return torch.from_numpy(z.astype(np.float32)).to(device=x.device, dtype=x.dtype)

    def re_anchor(x: torch.Tensor, t: float) -> torch.Tensor:
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
        v = velocity(x, t_tensor)
        x1_c = _nearest_anchor(x + (1.0 - t) * v, anchors_z)
        if cf_mode == "replace":
            return x1_c
        return (1.0 - t) * x0 + t * x1_c

    steps = int(cfg.sample.n_steps)
    dt = 1.0 / steps
    x = x0
    model.eval()
    for i in range(steps):
        t = i / steps
        t_next = (i + 1) / steps
        if use_cf and i == k_c:
            x = re_anchor(x, t)
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
        v = velocity(x, t_tensor)
        if use_cvf and t >= t_on:
            x1_c = _nearest_anchor(x + (1.0 - t) * v, anchors_z)
            v = _constrain_velocity(v, x1_c - x0, lam)
        x = x + dt * v
        if use_rfe:
            eta = energy_weight(t_next, tau_star, eta_max)
            if eta > 0.0:
                x = refine(x, eta)
    if use_rfe:
        for _ in range(n_refine):
            x = refine(x, eta_max)
    return x
