from __future__ import annotations

import torch


def solve_terminal_pgd(z_raw, z_phys, std, cons, lam, mu, n_iters, buffer, sweeps=None, max_step=0.5):
    denom = max(lam + mu, 1e-8)
    z_quad = (lam * z_raw + mu * z_phys) / denom
    z0 = cons.project_feasible(z_quad * std, buffer=buffer, sweeps=sweeps) / std
    z = z0
    lr = 1.0 / (denom + 2.0)
    for _ in range(n_iters):
        z = z.detach().requires_grad_(True)
        p = z * std
        loss = cons.cost(p) + 0.5 * lam * (z - z_raw).square().sum(dim=(-1, -2))
        if mu > 0:
            loss = loss + 0.5 * mu * (z - z_phys).square().sum(dim=(-1, -2))
        grad = torch.autograd.grad(loss.sum(), z)[0]
        step = lr * grad
        n = step.flatten(-2).norm(dim=-1)[..., None, None]
        step = step * (max_step / n.clamp_min(1e-9)).clamp(max=1.0)
        z = cons.project_feasible((z - step).detach() * std, buffer=buffer, sweeps=sweeps) / std
    bad = ~torch.isfinite(z).flatten(-2).all(dim=-1)
    if bad.any():
        z = torch.where(bad[..., None, None], z0, z)
    return z.detach()


@torch.no_grad()
def sample(model, batch, fut_std, k, n_steps, cons, ycfg: dict, generator=None, final_sweeps=None):
    t_on = float(ycfg.get("t_on", 0.5))
    lambda_oc = float(ycfg.get("lambda_oc", 10.0))
    mu_val = float(ycfg.get("mu", 1.0))
    delta = float(ycfg.get("delta", 0.1))
    gamma_max = float(ycfg.get("gamma_max", 1.0))
    max_iter = int(ycfg.get("max_iter", 20))
    buffer = float(ycfg.get("safety_buffer", 1e-3))
    max_step = float(ycfg.get("max_step", 0.5))

    memory, pad = model.encode(batch)
    b = memory.shape[0]
    z = torch.randn((b, k, model.T, 2), generator=generator,
                    device="cpu" if generator is not None else memory.device)
    z = z.to(memory.device, memory.dtype)
    std = fut_std.to(z.device, z.dtype)
    dt = 1.0 / n_steps
    stats = {"gated_frac": [], "lipschitz_mean": []}
    for i in range(n_steps):
        t = i * dt
        v = model.velocity(z, torch.full((b,), t, device=z.device, dtype=z.dtype), memory, pad)
        x1_raw = z + (1.0 - t) * v
        terminal = i == n_steps - 1
        if not terminal and t < t_on:
            z = z + dt * v
            continue
        p_raw = x1_raw * std
        if mu_val > 0:
            z_phys, step_mu = cons.project_physical(p_raw) / std, mu_val
        else:
            z_phys, step_mu = x1_raw, 0.0
        lp = cons.estimate_lipschitz(p_raw)
        g_val = gamma_max * min(max((t - t_on) / max(1.0 - t_on, 1e-8), 0.0), 1.0)
        gamma = torch.where(lp <= 1.0 + delta, torch.full_like(lp, g_val), torch.zeros_like(lp))
        stats["gated_frac"].append(float((gamma == 0).float().mean()))
        stats["lipschitz_mean"].append(float(lp.mean()))
        lam = lambda_oc * t ** 2 / dt
        n_iters = max_iter if terminal else max(3, max_iter // 2)
        with torch.enable_grad():
            z_star = solve_terminal_pgd(x1_raw, z_phys, std, cons, lam, step_mu, n_iters, buffer,
                                        sweeps=final_sweeps if terminal else None, max_step=max_step)
        if not terminal:
            z_star = torch.where((gamma == 0)[..., None, None], x1_raw, z_star)
        eta = 1.0 if terminal else dt / max(1.0 - t, 1e-8)
        z = (1.0 - eta) * z + eta * z_star
    return z * std, stats
