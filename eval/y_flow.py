# -*- coding: utf-8 -*-
# eval/y_flow.py
"""Physical Guidance Flow Matching (YFlow) GPU-accelerated sampling with PyTorch Autograd."""

from __future__ import annotations

import math
import numpy as np
import torch
from data.base import BaseConstraint, build_dataset
from eval._backbone import load_frozen_velocity


def torch_spiral(u: torch.Tensor, a: float) -> torch.Tensor:
    from data.swiss_roll import torch_spiral as _ts
    return _ts(u, a)


def torch_nearest_u(p: torch.Tensor, a: float, u_min: float, u_max: float) -> torch.Tensor:
    from data.swiss_roll import torch_nearest_u as _tnu
    return _tnu(p, a, u_min, u_max)


def torch_project_to_manifold(p: torch.Tensor, a: float, u_min: float, u_max: float) -> torch.Tensor:
    from data.swiss_roll import torch_project_to_manifold as _tpm
    return _tpm(p, a, u_min, u_max)


def estimate_lipschitz(
    p: np.ndarray | torch.Tensor,
    constraint_or_a: BaseConstraint | float,
    u_min: float | None = None,
    u_max: float | None = None,
    eps: float = 1e-4,
) -> np.ndarray | torch.Tensor:
    """Estimate local Lipschitz constant of physical projection operator P."""
    if isinstance(constraint_or_a, BaseConstraint):
        return constraint_or_a.estimate_lipschitz(p, eps=eps)
    from data.swiss_roll import estimate_lipschitz as _el
    return _el(p, constraint_or_a, u_min=u_min, u_max=u_max, eps=eps)


def project_feasible(
    p: torch.Tensor,
    a: float,
    u_min: float,
    u_max: float,
    tau: float,
    rho_min: float,
    R: float,
    buffer: float = 1e-4,
) -> torch.Tensor:
    """Exact projection onto feasible set h_tube, h_core, h_box <= 0."""
    # 1. Tube constraint projection
    u = torch_nearest_u(p, a, u_min, u_max)
    g = torch_spiral(u, a)
    v = p - g
    d = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-12)
    max_d = tau - buffer
    scale = torch.clamp(max_d / d, max=1.0)
    p_tube = g + scale * v

    # 2. Core constraint projection
    r = torch.norm(p_tube, dim=-1, keepdim=True).clamp(min=1e-12)
    p_core = torch.where(r < rho_min + buffer, p_tube * ((rho_min + buffer) / r), p_tube)

    # 3. Box constraint projection
    p_box = torch.clamp(p_core, -R + buffer, R - buffer)
    return p_box


def solve_terminal_pgd(
    z_raw: torch.Tensor,
    z_phys: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    constraint: BaseConstraint | None = None,
    lam: float = 10.0,
    mu: float = 1.0,
    n_iters: int = 15,
    buffer: float = 1e-4,
    *,
    a: float | None = None,
    u_min: float | None = None,
    u_max: float | None = None,
    tau: float | None = None,
    rho_min: float | None = None,
    R: float | None = None,
) -> torch.Tensor:
    """GPU-batched Projected Gradient Descent with PyTorch Autograd."""
    denom = max(lam + mu, 1e-8)
    z_quad = (lam * z_raw + mu * z_phys) / denom

    p_init = z_quad * std + mean
    if constraint is not None:
        p_feas = constraint.project_feasible(p_init, buffer=buffer)
    else:
        p_feas = project_feasible(p_init, a, u_min, u_max, tau, rho_min, R, buffer=buffer)
    z = (p_feas - mean) / std

    lr = 1.0 / (denom + 2.0)

    for _ in range(n_iters):
        z = z.detach().requires_grad_(True)
        p = z * std + mean
        if constraint is not None:
            cost = constraint.cost(p)
        else:
            u = torch_nearest_u(p, a, u_min, u_max)
            g = torch_spiral(u, a)
            cost = ((p - g) ** 2).sum(dim=-1)
        raw_penalty = 0.5 * lam * ((z - z_raw) ** 2).sum(dim=-1)
        phys_penalty = 0.5 * mu * ((z - z_phys) ** 2).sum(dim=-1) if mu > 0 else 0.0

        total_loss = (cost + raw_penalty + phys_penalty).sum()
        grad = torch.autograd.grad(total_loss, z)[0]

        z_step = z - lr * grad
        p_step = z_step * std + mean
        if constraint is not None:
            p_feas = constraint.project_feasible(p_step, buffer=buffer)
        else:
            p_feas = project_feasible(p_step, a, u_min, u_max, tau, rho_min, R, buffer=buffer)
        z = (p_feas - mean) / std

    return z.detach()


@torch.no_grad()
def sample(cfg: DictConfig, device: torch.device, x0: torch.Tensor) -> torch.Tensor:
    model, method = load_frozen_velocity(cfg, device)
    bundle = build_dataset(cfg)
    constraint = bundle["constraint"]

    mean_t = bundle["mean"].to(device=device, dtype=x0.dtype)
    std_t = bundle["std"].to(device=device, dtype=x0.dtype)

    yflow_cfg = cfg.get("yflow", {})
    t_on = float(yflow_cfg.get("t_on", 0.5))
    lambda_oc = float(yflow_cfg.get("lambda_oc", 10.0))
    mu_val = float(yflow_cfg.get("mu", 1.0))
    delta = float(yflow_cfg.get("delta", 0.1))
    gamma_max = float(yflow_cfg.get("gamma_max", 1.0))
    max_iter = int(yflow_cfg.get("max_iter", 20))
    safety_buffer = float(yflow_cfg.get("safety_buffer", 1e-4))

    steps = int(cfg.sample.n_steps)
    dt = 1.0 / steps
    x = x0
    model.eval()

    for i in range(steps):
        t = i / steps
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
        v = method.velocity(model, x, t_tensor)
        x1_raw = x + (1.0 - t) * v

        is_terminal_step = i == steps - 1

        if not is_terminal_step and t < t_on:
            eta = dt / max(1.0 - t, 1e-8)
            x = (1.0 - eta) * x + eta * x1_raw
            continue

        p_raw = x1_raw * std_t + mean_t
        p_phys = constraint.project_physical(p_raw)
        if p_phys is not None:
            z_phys = (p_phys - mean_t) / std_t
            step_mu = mu_val
        else:
            z_phys = x1_raw
            step_mu = 0.0

        L_P = constraint.estimate_lipschitz(p_raw)
        if L_P is None:
            L_P = torch.zeros(p_raw.shape[0], device=device, dtype=p_raw.dtype)

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
                mu=step_mu,
                n_iters=n_iters,
                buffer=safety_buffer,
            )

        if not is_terminal_step:
            mask = (gamma == 0.0).unsqueeze(-1)
            z_star = torch.where(mask, x1_raw, z_star)

        eta = 1.0 if is_terminal_step else dt / max(1.0 - t, 1e-8)
        x = (1.0 - eta) * x + eta * z_star

    return x
