# -*- coding: utf-8 -*-
# constraints/fmbf.py
"""Flow Matching Barrier Function schedules and small batched QP solvers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CompositeQPSolution:
    correction: torch.Tensor
    slack: torch.Tensor
    raw_residual: torch.Tensor
    relaxed_residual: torch.Tensor
    active_mask: torch.Tensor


def barrier_gain(
    t: float | torch.Tensor,
    h: torch.Tensor,
    *,
    phi0: float = 1.0,
    schedule: str = "paper_piecewise",
    gamma: float = 0.9,
    omega: float = 3.0,
    terminal_eps: float = 1.0e-3,
) -> torch.Tensor:
    """Return phi(t, h), using phi0 inside and a blow-up gain outside."""
    if not 0.0 < terminal_eps < 1.0:
        raise ValueError("terminal_eps must lie in (0, 1)")
    t_tensor = torch.as_tensor(t, device=h.device, dtype=h.dtype)
    while t_tensor.ndim < h.ndim:
        t_tensor = t_tensor.unsqueeze(-1)
    one_minus_t = (1.0 - t_tensor).clamp_min(float(terminal_eps))
    if schedule == "paper_piecewise":
        unsafe_gain = torch.where(
            t_tensor < float(gamma),
            1.0 + 4.0 * t_tensor.pow(3),
            one_minus_t.reciprocal(),
        )
    elif schedule == "inverse_polynomial":
        if omega <= 2.0:
            raise ValueError("inverse_polynomial requires omega > 2")
        unsafe_gain = float(omega) / one_minus_t.square()
    elif schedule == "exponential":
        z = float(omega) * one_minus_t
        unsafe_gain = float(omega) * torch.exp(z) / torch.expm1(z).clamp_min(
            torch.finfo(h.dtype).eps
        )
    else:
        raise ValueError(f"unknown FMBF schedule: {schedule}")
    return torch.where(h >= 0.0, torch.full_like(h, float(phi0)), unsafe_gain)


def solve_single_fmbf(a: torch.Tensor, b: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    """Closed-form minimum-norm solution of min ||u||^2 s.t. a + b^T u >= 0."""
    if b.shape[:-1] != a.shape:
        raise ValueError(f"incompatible shapes: a={tuple(a.shape)}, b={tuple(b.shape)}")
    norm2 = b.square().sum(dim=-1)
    impossible = (a < 0.0) & (norm2 <= float(eps))
    if bool(impossible.any()):
        raise ValueError("infeasible single FMBF constraint with zero gradient")
    scale = torch.where(a < 0.0, -a / norm2.clamp_min(float(eps)), torch.zeros_like(a))
    return scale.unsqueeze(-1) * b


def solve_composite_fmbf(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    slack_weight: float = 1.0,
    active_tol: float = 1.0e-6,
    max_constraints: int = 12,
) -> CompositeQPSolution:
    """Solve the relaxed CFMBF QP by enumerating its piecewise-quadratic active sets.

    The solved problem is
        min ||u||^2 + w ||delta||^2
        s.t. a_j + b_j^T u + delta_j >= 0, delta_j >= 0.
    This is intended for the small number of constraints used in Exp-01.
    """
    if a.ndim < 1 or b.ndim != a.ndim + 1 or b.shape[:-1] != a.shape:
        raise ValueError(f"incompatible shapes: a={tuple(a.shape)}, b={tuple(b.shape)}")
    if slack_weight <= 0.0:
        raise ValueError("slack_weight must be positive")
    n_constraints = int(a.shape[-1])
    dim = int(b.shape[-1])
    if n_constraints == 0 or n_constraints > int(max_constraints):
        raise ValueError(f"constraint count must be in [1, {max_constraints}]")

    batch_shape = a.shape[:-1]
    flat_a = a.reshape(-1, n_constraints)
    flat_b = b.reshape(-1, n_constraints, dim)
    batch = int(flat_a.shape[0])
    eye = torch.eye(dim, device=b.device, dtype=b.dtype).expand(batch, dim, dim)
    weight = float(slack_weight)

    candidates_u: list[torch.Tensor] = []
    candidates_slack: list[torch.Tensor] = []
    candidates_raw: list[torch.Tensor] = []
    candidates_valid: list[torch.Tensor] = []
    candidates_obj: list[torch.Tensor] = []
    mask_values: list[int] = []

    for mask_value in range(1 << n_constraints):
        active = torch.tensor(
            [(mask_value >> j) & 1 for j in range(n_constraints)],
            device=b.device,
            dtype=torch.bool,
        )
        if bool(active.any()):
            b_active = flat_b[:, active, :]
            a_active = flat_a[:, active]
            lhs = eye + weight * torch.matmul(b_active.transpose(-1, -2), b_active)
            rhs = -weight * torch.matmul(
                b_active.transpose(-1, -2), a_active.unsqueeze(-1)
            )
            u = torch.linalg.solve(lhs, rhs).squeeze(-1)
        else:
            u = torch.zeros(batch, dim, device=b.device, dtype=b.dtype)

        raw = flat_a + torch.einsum("bnd,bd->bn", flat_b, u)
        slack = torch.relu(-raw)
        active_ok = (raw[:, active] <= float(active_tol)).all(dim=-1)
        inactive_ok = (raw[:, ~active] >= -float(active_tol)).all(dim=-1)
        finite = torch.isfinite(u).all(dim=-1) & torch.isfinite(raw).all(dim=-1)
        valid = active_ok & inactive_ok & finite
        objective = u.square().sum(dim=-1) + weight * slack.square().sum(dim=-1)

        candidates_u.append(u)
        candidates_slack.append(slack)
        candidates_raw.append(raw)
        candidates_valid.append(valid)
        candidates_obj.append(objective)
        mask_values.append(mask_value)

    stacked_u = torch.stack(candidates_u, dim=1)
    stacked_slack = torch.stack(candidates_slack, dim=1)
    stacked_raw = torch.stack(candidates_raw, dim=1)
    valid = torch.stack(candidates_valid, dim=1)
    objective = torch.stack(candidates_obj, dim=1)
    inf = torch.full_like(objective, torch.inf)
    selected = torch.where(valid, objective, inf).argmin(dim=1)
    has_valid = valid.any(dim=1)
    if not bool(has_valid.all()):
        failed = int((~has_valid).sum())
        raise RuntimeError(f"CFMBF active-set solver failed for {failed} batch items")
    row = torch.arange(batch, device=b.device)

    chosen_u = stacked_u[row, selected]
    chosen_slack = stacked_slack[row, selected]
    chosen_raw = stacked_raw[row, selected]
    relaxed = chosen_raw + chosen_slack

    masks = torch.tensor(mask_values, device=b.device, dtype=torch.long)
    chosen_mask = masks[selected]
    return CompositeQPSolution(
        correction=chosen_u.reshape(*batch_shape, dim),
        slack=chosen_slack.reshape(*batch_shape, n_constraints),
        raw_residual=chosen_raw.reshape(*batch_shape, n_constraints),
        relaxed_residual=relaxed.reshape(*batch_shape, n_constraints),
        active_mask=chosen_mask.reshape(batch_shape),
    )
