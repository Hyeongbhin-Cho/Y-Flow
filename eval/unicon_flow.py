# -*- coding: utf-8 -*-
# eval/unicon_flow.py
"""Training-free UniConFlow specialization for the 2D Swiss-roll benchmark.

The sampler follows Eqs. (42)-(51) of UniConFlow: a PTZF constraint
certificate defines linear conditions on the flow velocity and a batched
minimum-norm QP supplies the guidance. The paper's terminal refinement is
specialized to the exact Swiss-roll feasible-set projection.
"""

from __future__ import annotations

import math

import torch
from omegaconf import DictConfig

from data.swiss_roll import build_swiss_roll
from eval._backbone import load_frozen_velocity
from eval.y_flow import project_feasible, torch_nearest_u, torch_spiral


def constraint_values(
    z: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    a: float,
    u_min: float,
    u_max: float,
    tau: float,
    rho_min: float,
    radius: float,
) -> torch.Tensor:
    """Return [tube, core, box] constraints in the paper's h <= 0 form."""
    p = z * std + mean
    u = torch_nearest_u(p, a, u_min, u_max)
    curve = torch_spiral(u, a)
    tube = torch.linalg.vector_norm(p - curve, dim=-1) - tau
    core = rho_min - torch.linalg.vector_norm(p, dim=-1)
    box = torch.amax(torch.abs(p), dim=-1) - radius
    return torch.stack((tube, core, box), dim=-1)


def constraint_jacobian(h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Compute eta_h = dh/dz for every batched scalar constraint."""
    grads = []
    for j in range(h.shape[-1]):
        grad = torch.autograd.grad(h[:, j].sum(), z, retain_graph=True)[0]
        grads.append(grad)
    return torch.stack(grads, dim=1)


def ptzf_reference(
    initial: torch.Tensor,
    t: float,
    rate: float,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PTZF r(t) and its derivative (paper Eq. 36 with T_pre=1)."""
    one_minus_t = max(1.0 - t, eps)
    exponent = -rate * t / one_minus_t
    reference = initial * math.exp(max(exponent, -80.0))
    derivative = -rate * reference / (one_minus_t * one_minus_t)
    return reference, derivative


def qp_guidance(
    rho: torch.Tensor,
    eta: torch.Tensor,
    *,
    slack_weight: float,
    ridge: float = 1e-6,
) -> torch.Tensor:
    """Closed-form slack QP from paper Eqs. (49)-(51), batched."""
    batch, n_constraints, dim = eta.shape
    active = rho > 0
    rho_active = torch.where(active, rho, torch.zeros_like(rho))
    eta_active = eta * active.unsqueeze(-1)

    eye_m = torch.eye(n_constraints, device=eta.device, dtype=eta.dtype)
    eye_m = eye_m.expand(batch, -1, -1)
    eta_z = torch.cat((eta_active, -eye_m * active.unsqueeze(-1)), dim=-1)

    inv_diag = torch.cat(
        (
            torch.ones(dim, device=eta.device, dtype=eta.dtype),
            torch.full(
                (n_constraints,),
                1.0 / max(slack_weight, 1e-8),
                device=eta.device,
                dtype=eta.dtype,
            ),
        )
    )
    weighted_eta_t = inv_diag.view(1, -1, 1) * eta_z.transpose(1, 2)
    gram = eta_z @ weighted_eta_t + ridge * eye_m
    multipliers = torch.linalg.solve(gram, rho_active.unsqueeze(-1))
    z = -(weighted_eta_t @ multipliers).squeeze(-1)
    return z[:, :dim]


def terminal_refinement(
    z: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    a: float,
    u_min: float,
    u_max: float,
    tau: float,
    rho_min: float,
    radius: float,
    buffer: float,
) -> torch.Tensor:
    """Exact 2D analogue of UniConFlow's terminal constraint refinement."""
    p = z * std + mean
    p = project_feasible(
        p, a, u_min, u_max, tau, rho_min, radius, buffer=buffer
    )
    return (p - mean) / std


def sample(cfg: DictConfig, device: torch.device, x0: torch.Tensor) -> torch.Tensor:
    model, method = load_frozen_velocity(cfg, device)
    bundle = build_swiss_roll(cfg)
    meta = bundle["meta"]
    mean = bundle["mean"].to(device=device, dtype=x0.dtype)
    std = bundle["std"].to(device=device, dtype=x0.dtype)

    uc = cfg.get("uniconflow", {})
    free_until = float(uc.get("free_until", 0.0))
    ptzf_rate = float(uc.get("ptzf_rate", 1.0))
    gamma = float(uc.get("gamma", 1.0))
    slack_weight = float(uc.get("slack_weight", 100.0))
    max_guidance_norm = float(uc.get("max_guidance_norm", 20.0))
    terminal = bool(uc.get("terminal_refinement", True))
    safety_buffer = float(uc.get("safety_buffer", 1e-4))

    kwargs = {
        "a": float(meta.a),
        "u_min": float(meta.u_min),
        "u_max": float(meta.u_max),
        "tau": float(meta.tau),
        "rho_min": float(meta.rho_min),
        "radius": float(meta.R),
    }
    steps = int(cfg.sample.n_steps)
    dt = 1.0 / steps
    x = x0
    hbar0: torch.Tensor | None = None
    model.eval()

    for i in range(steps):
        t = i / steps
        t_tensor = torch.full((x.shape[0],), t, device=device, dtype=x.dtype)
        with torch.no_grad():
            v = method.velocity(model, x, t_tensor)

        with torch.enable_grad():
            z = x.detach().requires_grad_(True)
            h = constraint_values(z, mean, std, **kwargs)
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
        x = terminal_refinement(
            x, mean, std, buffer=safety_buffer, **kwargs
        )
    return x.detach()
