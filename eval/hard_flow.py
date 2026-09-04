# -*- coding: utf-8 -*-
# eval/hard_flow.py
"""Training-free HardFlow sampling with PyTorch Autograd GPU-batched optimization."""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from data.swiss_roll import build_swiss_roll
from eval._backbone import load_frozen_velocity
from eval.y_flow import project_feasible, torch_nearest_u, torch_spiral


def solve_terminal_pgd_hardflow(
    z_bar: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    a: float,
    u_min: float,
    u_max: float,
    tau: float,
    rho_min: float,
    R: float,
    lam: float,
    n_iters: int = 20,
    buffer: float = 1e-4,
) -> torch.Tensor:
    """GPU-batched Projected Gradient Descent for HardFlow terminal optimization."""
    p_init = z_bar * std + mean
    p_feas = project_feasible(p_init, a, u_min, u_max, tau, rho_min, R, buffer=buffer)
    z = (p_feas - mean) / std

    lr = 1.0 / (lam + 2.0)

    for _ in range(n_iters):
        z = z.detach().requires_grad_(True)
        p = z * std + mean
        u = torch_nearest_u(p, a, u_min, u_max)
        g = torch_spiral(u, a)
        cost = ((p - g) ** 2).sum(dim=-1)
        bar_penalty = 0.5 * lam * ((z - z_bar) ** 2).sum(dim=-1)

        total_loss = (cost + bar_penalty).sum()
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

    hf_cfg = cfg.get("hardflow", {})
    t_on = float(hf_cfg.get("t_on", 0.5))
    lambda_oc = float(hf_cfg.get("lambda_oc", 10.0))
    max_iter = int(hf_cfg.get("max_iter", 20))
    safety_buffer = float(hf_cfg.get("safety_buffer", 1e-4))

    steps = int(cfg.sample.n_steps)
    dt = 1.0 / steps
    x = x0
    model.eval()

    for i in range(steps):
        t = i / steps
        t_next = (i + 1) / steps
        t_tensor = torch.full((x.shape[0],), t, device=device, dtype=x.dtype)
        v = method.velocity(model, x, t_tensor)
        bar_x = x + dt * v

        if t < t_on and i < steps - 1:
            x = bar_x
            continue

        t_next_tensor = torch.full((x.shape[0],), t_next, device=device, dtype=x.dtype)
        v_next = method.velocity(model, bar_x, t_next_tensor)
        bar_x1 = bar_x + (1.0 - t_next) * v_next

        lam = lambda_oc * (t_next ** 2) / max(dt, 1e-8)
        with torch.enable_grad():
            z_star = solve_terminal_pgd_hardflow(
                bar_x1,
                mean_t,
                std_t,
                a=a,
                u_min=u_min,
                u_max=u_max,
                tau=tau,
                rho_min=rho_min,
                R=R,
                lam=lam,
                n_iters=max_iter,
                buffer=safety_buffer,
            )

        w0 = bar_x - t_next * v_next
        x = t_next * z_star + (1.0 - t_next) * w0

    return x
