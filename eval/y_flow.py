# -*- coding: utf-8 -*-
# eval/y_flow.py
"""Physical Guidance Flow Matching (YFlow) GPU-accelerated sampling with PyTorch Autograd."""

from __future__ import annotations

import math
import numpy as np
import torch
from omegaconf import DictConfig

from constraints.swiss_roll import SwissRollConstraint
from data.swiss_roll import build_swiss_roll
from eval._backbone import load_frozen_velocity


def torch_spiral(u: torch.Tensor, a: float) -> torch.Tensor:
    return torch.stack([a * u * torch.cos(u), a * u * torch.sin(u)], dim=-1)


def torch_nearest_u(p: torch.Tensor, a: float, u_min: float, u_max: float) -> torch.Tensor:
    phi = torch.atan2(p[:, 1], p[:, 0])
    k0 = int(math.floor(u_min / (2.0 * math.pi))) - 1
    k1 = int(math.ceil(u_max / (2.0 * math.pi))) + 1
    ks = torch.arange(k0, k1 + 1, device=p.device, dtype=p.dtype)
    u_cands = phi[:, None] + 2.0 * math.pi * ks[None, :]
    u_min_t = torch.full((p.shape[0], 1), u_min, device=p.device, dtype=p.dtype)
    u_max_t = torch.full((p.shape[0], 1), u_max, device=p.device, dtype=p.dtype)
    u_all = torch.cat([u_cands, u_min_t, u_max_t], dim=1)
    u_all = torch.clamp(u_all, u_min, u_max)
    g = torch_spiral(u_all, a)
    d2 = ((g - p[:, None, :]) ** 2).sum(dim=-1)
    min_idx = d2.argmin(dim=1)
    return u_all[torch.arange(p.shape[0], device=p.device), min_idx]


def torch_project_to_manifold(p: torch.Tensor, a: float, u_min: float, u_max: float) -> torch.Tensor:
    u = torch_nearest_u(p, a, u_min, u_max)
    return torch_spiral(u, a)


def estimate_lipschitz(
    p: np.ndarray | torch.Tensor,
    constraint_or_a: SwissRollConstraint | float,
    u_min: float | None = None,
    u_max: float | None = None,
    eps: float = 1e-4,
) -> np.ndarray | torch.Tensor:
    """Estimate local Lipschitz constant of physical projection operator P via batched PyTorch."""
    is_np = isinstance(p, np.ndarray)
    if isinstance(constraint_or_a, SwissRollConstraint):
        a = float(constraint_or_a.meta.a)
        u_min = float(constraint_or_a.meta.u_min)
        u_max = float(constraint_or_a.meta.u_max)
    else:
        a = float(constraint_or_a)
        assert u_min is not None and u_max is not None

    if is_np:
        p_t = torch.from_numpy(p.astype(np.float32))
    else:
        p_t = p

    p_proj = torch_project_to_manifold(p_t, a, u_min, u_max)
    dirs = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], device=p_t.device, dtype=p_t.dtype)
    max_L = torch.zeros(p_t.shape[0], device=p_t.device, dtype=p_t.dtype)
    for d in dirs:
        p_pert = p_t + eps * d[None, :]
        p_pert_proj = torch_project_to_manifold(p_pert, a, u_min, u_max)
        diff = torch.norm(p_pert_proj - p_proj, dim=-1)
        max_L = torch.maximum(max_L, diff / eps)

    if is_np:
        return max_L.cpu().numpy()
    return max_L


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
    a: float,
    u_min: float,
    u_max: float,
    tau: float,
    rho_min: float,
    R: float,
    lam: float,
    mu: float,
    n_iters: int = 15,
    buffer: float = 1e-4,
) -> torch.Tensor:
    """GPU-batched Projected Gradient Descent with PyTorch Autograd."""
    denom = max(lam + mu, 1e-8)
    z_quad = (lam * z_raw + mu * z_phys) / denom

    p_init = z_quad * std + mean
    p_feas = project_feasible(p_init, a, u_min, u_max, tau, rho_min, R, buffer=buffer)
    z = (p_feas - mean) / std

    lr = 1.0 / (denom + 2.0)

    for _ in range(n_iters):
        z = z.detach().requires_grad_(True)
        p = z * std + mean
        u = torch_nearest_u(p, a, u_min, u_max)
        g = torch_spiral(u, a)
        cost = ((p - g) ** 2).sum(dim=-1)
        raw_penalty = 0.5 * lam * ((z - z_raw) ** 2).sum(dim=-1)
        phys_penalty = 0.5 * mu * ((z - z_phys) ** 2).sum(dim=-1) if mu > 0 else 0.0

        total_loss = (cost + raw_penalty + phys_penalty).sum()
        grad = torch.autograd.grad(total_loss, z)[0]

        z_step = z - lr * grad
        p_step = z_step * std + mean
        p_feas = project_feasible(p_step, a, u_min, u_max, tau, rho_min, R, buffer=buffer)
        z = (p_feas - mean) / std

    return z.detach()


@torch.no_grad()
def sample(cfg: DictConfig, device: torch.device, x0: torch.Tensor) -> torch.Tensor:
    model, method = load_frozen_velocity(cfg, device)
    bundle = build_swiss_roll(cfg)
    meta = bundle["meta"]
    a = float(meta.a)
    u_min = float(meta.u_min)
    u_max = float(meta.u_max)
    tau = float(meta.tau)
    rho_min = float(meta.rho_min)
    R = float(meta.R)

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
        p_phys = torch_project_to_manifold(p_raw, a, u_min, u_max)
        z_phys = (p_phys - mean_t) / std_t

        L_P = estimate_lipschitz(p_raw, a, u_min, u_max)

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
                a,
                u_min,
                u_max,
                tau,
                rho_min,
                R,
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
