# -*- coding: utf-8 -*-
# data/base.py
"""Unified dataset and constraint interface for physical flow matching."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset


# =====================================================================
# 1. Base Constraint Interface
# =====================================================================

class BaseConstraint(ABC):
    """Abstract interface for dataset-dependent constraints and physical operators."""

    @abstractmethod
    def h(self, p: torch.Tensor | np.ndarray) -> dict[str, torch.Tensor | np.ndarray]:
        """Hard constraint function: h_j(p) <= 0.
        
        Returns:
            Dictionary mapping constraint names (e.g., 'tube', 'core', 'box')
            to violation tensors/arrays where values <= 0 represent feasible points.
        """
        raise NotImplementedError

    @abstractmethod
    def project_feasible(self, p: torch.Tensor, buffer: float = 1e-4) -> torch.Tensor:
        """Exact or proximal projection onto feasible set S = {p | h(p) <= 0}.
        
        Used by HardFlow, UniConFlow, YFlow terminal steps, and SafeFlow filters.
        """
        raise NotImplementedError

    def project_physical(
        self, p: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Physical projection operator P: maps state to ideal manifold target.
        
        Default: Identity mapping P(x) = x when no domain-specific physical manifold
        prior is defined. Allows YFlow to degrade gracefully into constraint-guided
        flow matching (similar to HardFlow with trajectory regularization).
        """
        return p

    def estimate_lipschitz(
        self, p: torch.Tensor | np.ndarray, eps: float = 1e-4
    ) -> torch.Tensor | np.ndarray:
        """Estimate local Lipschitz constant of physical operator P.
        
        Default: 1.0 for the identity operator P(x) = x.
        """
        if isinstance(p, torch.Tensor):
            return torch.ones(p.shape[0], device=p.device, dtype=p.dtype)
        return np.ones(p.shape[0], dtype=np.float64)

    def cost(self, p: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Cost function C(p) to minimize (default: 0)."""
        if isinstance(p, torch.Tensor):
            return torch.zeros(p.shape[:-1], device=p.device, dtype=p.dtype)
        return np.zeros(p.shape[:-1], dtype=p.dtype)

    def energy_grad(self, p: np.ndarray, **kwargs) -> np.ndarray:
        """Energy gradient grad_p E(p) used by GuideFlow RFE."""
        raise NotImplementedError("energy_grad is not implemented for this constraint")

    def energy(
        self,
        p: torch.Tensor,
        w_tube: float = 1.0,
        w_core: float = 1.0,
        w_box: float = 1.0,
        w_cost: float = 0.0,
        slack: float = 0.01,
    ) -> torch.Tensor:
        """Scalar energy function for GuideFlow RFE training/evaluation."""
        raise NotImplementedError("energy is not implemented for this constraint")

    def progress(self, p: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Scalar progress metric in [0, 1] along the manifold/task."""
        raise NotImplementedError("progress is not implemented for this constraint")

    def command_bins(
        self,
        p: torch.Tensor | np.ndarray,
        n_commands: int,
    ) -> torch.Tensor | np.ndarray:
        """Discrete command bin index in {0, ..., n_commands - 1} based on progress."""
        prog = self.progress(p)
        if isinstance(prog, torch.Tensor):
            frac = prog.clamp(0.0, 1.0)
            return torch.clamp((frac * n_commands).long(), 0, n_commands - 1)
        else:
            frac = np.clip(np.asarray(prog, dtype=np.float64), 0.0, 1.0)
            return np.clip((frac * n_commands).astype(np.int64), 0, int(n_commands) - 1)

    def build_anchor_vocabulary(
        self,
        points: np.ndarray,
        n_anchors: int = 256,
        seed: int = 42,
    ) -> np.ndarray:
        """Build anchor vocabulary via farthest point sampling over feasible points."""
        return build_anchor_vocabulary(points, self, n_anchors=n_anchors, seed=seed)

    def get_fmbf(self, **kwargs) -> Any:
        """Return smooth FMBF barrier object for SafeFlow CFMBF-QP."""
        raise NotImplementedError("smooth FMBF is not implemented for this constraint")


def build_anchor_vocabulary(
    points: np.ndarray,
    constraint: BaseConstraint,
    n_anchors: int = 256,
    seed: int = 42,
) -> np.ndarray:
    """Build anchor vocabulary via farthest point sampling over feasible points."""
    p = np.asarray(points, dtype=np.float64)
    h = constraint.h(p)
    h_stack = np.stack(list(h.values()), axis=-1)
    feasible = h_stack.max(axis=-1) <= 0.0
    pool = (
        p[feasible]
        if bool(feasible.any())
        else np.asarray(constraint.project_feasible(p), dtype=np.float64)
    )
    n = int(min(max(n_anchors, 1), pool.shape[0]))
    rng = np.random.default_rng(int(seed))
    picked = [int(rng.integers(pool.shape[0]))]
    d2 = ((pool - pool[picked[0]]) ** 2).sum(axis=-1)
    for _ in range(n - 1):
        j = int(d2.argmax())
        picked.append(j)
        d2 = np.minimum(d2, ((pool - pool[j]) ** 2).sum(axis=-1))
    return pool[np.asarray(picked, dtype=np.int64)]


# =====================================================================
# 2. Standard DataBundle Container
# =====================================================================

@dataclass
class DataBundle:
    """Unified container packaging dataset tensors and bound constraints.
    
    Supports both attribute access (bundle.train) and dict indexing (bundle['train'])
    to preserve 100% backward compatibility with existing training and evaluation code.
    """

    train: Dataset
    train_raw: np.ndarray
    eval_raw: np.ndarray
    eval_z: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    constraint: BaseConstraint
    meta: Any
    meta_dict: dict

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"Key {key!r} not found in DataBundle")

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def keys(self):
        return [
            "train",
            "train_raw",
            "eval_raw",
            "eval_z",
            "mean",
            "std",
            "constraint",
            "meta",
            "meta_dict",
        ]


# =====================================================================
# 3. Dynamic Registry & Routing Factory
# =====================================================================

_DATASET_REGISTRY: dict[str, Callable[[DictConfig], DataBundle]] = {}


def register_dataset(name: str):
    """Decorator to register a dataset builder function."""
    def decorator(fn: Callable[[DictConfig], DataBundle]):
        _DATASET_REGISTRY[name.lower()] = fn
        return fn

    return decorator


def build_dataset(cfg_or_name: DictConfig | str, **kwargs) -> DataBundle:
    """Route and build dataset bundle by config or string name.
    
    Examples:
        bundle = build_dataset(cfg)
        bundle = build_dataset("swiss_roll", cfg=cfg)
    """
    if isinstance(cfg_or_name, str):
        name = cfg_or_name.lower()
        cfg = kwargs.get("cfg", None)
        if cfg is None:
            raise ValueError("When passing a dataset name string, 'cfg' must be supplied in kwargs")
    else:
        name = str(cfg_or_name.data.name).lower()
        cfg = cfg_or_name

    if name not in _DATASET_REGISTRY:
        # Lazy load built-in datasets if not yet imported
        if name == "swiss_roll":
            import data.swiss_roll  # noqa: F401

    if name not in _DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset {name!r}. Registered datasets: {list(_DATASET_REGISTRY.keys())}"
        )

    return _DATASET_REGISTRY[name](cfg)


# =====================================================================
# 4. Common Normalization Utilities
# =====================================================================

def denormalize(z: torch.Tensor | np.ndarray, mean: Any, std: Any) -> torch.Tensor | np.ndarray:
    """Transform z-score normalized samples back to physical space p."""
    if isinstance(z, torch.Tensor):
        m = (
            mean.to(device=z.device, dtype=z.dtype)
            if isinstance(mean, torch.Tensor)
            else torch.as_tensor(mean, device=z.device, dtype=z.dtype)
        )
        s = (
            std.to(device=z.device, dtype=z.dtype)
            if isinstance(std, torch.Tensor)
            else torch.as_tensor(std, device=z.device, dtype=z.dtype)
        )
        return z * s + m
    return np.asarray(z) * np.asarray(std) + np.asarray(mean)


def normalize(p: torch.Tensor | np.ndarray, mean: Any, std: Any) -> torch.Tensor | np.ndarray:
    """Normalize physical space p into zero-mean, unit-variance z."""
    if isinstance(p, torch.Tensor):
        m = (
            mean.to(device=p.device, dtype=p.dtype)
            if isinstance(mean, torch.Tensor)
            else torch.as_tensor(mean, device=p.device, dtype=p.dtype)
        )
        s = (
            std.to(device=p.device, dtype=p.dtype)
            if isinstance(std, torch.Tensor)
            else torch.as_tensor(std, device=p.device, dtype=p.dtype)
        )
        return (p - m) / s
    return (np.asarray(p) - np.asarray(mean)) / np.asarray(std)


# =====================================================================
# 5. Generic CFMBF Active-set QP Solver & Barrier Schedules
# =====================================================================

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
    """Solve relaxed CFMBF QP by enumerating its piecewise-quadratic active sets.

    The solved problem is:
        min ||u||^2 + w ||delta||^2
        s.t. a_j + b_j^T u + delta_j >= 0, delta_j >= 0.
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
