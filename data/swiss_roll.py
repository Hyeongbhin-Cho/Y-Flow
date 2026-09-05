# -*- coding: utf-8 -*-
# data/swiss_roll.py
"""2D Swiss roll points and unified constraints for Exp-01."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from scipy.optimize import minimize
from torch.utils.data import Dataset

from data.base import BaseConstraint, DataBundle, register_dataset
from utils.paths import ROOT


# =====================================================================
# 1. Metadata and PyTorch Dataset
# =====================================================================

@dataclass
class SwissRollMeta:
    a: float
    u_min: float
    u_max: float
    n_turns: float
    sigma_obs: float
    n_train: int
    n_eval: int
    margin: float
    tau: float
    rho_min: float
    R: float
    seed: int
    mean: tuple[float, float]
    std: tuple[float, float]
    arc_length: float


class PointDataset(Dataset):
    def __init__(self, points: torch.Tensor):
        self.points = points

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.points[idx]


# =====================================================================
# 2. Geometry Functions (NumPy & PyTorch)
# =====================================================================

def spiral(u: np.ndarray, a: float) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    return np.stack([a * u * np.cos(u), a * u * np.sin(u)], axis=-1)


def torch_spiral(u: torch.Tensor, a: float) -> torch.Tensor:
    return torch.stack([a * u * torch.cos(u), a * u * torch.sin(u)], dim=-1)


def arc_length(a: float, u_min: float, u_max: float) -> float:
    def F(u: float) -> float:
        return 0.5 * a * (u * math.sqrt(1.0 + u * u) + math.asinh(u))

    return F(u_max) - F(u_min)


def sample_points(
    n: int,
    a: float,
    u_min: float,
    u_max: float,
    sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    u = rng.uniform(u_min, u_max, size=n)
    p = spiral(u, a) + sigma * rng.normal(size=(n, 2))
    return p.astype(np.float32), u.astype(np.float32)


def nearest_u(p: np.ndarray, a: float, u_min: float, u_max: float) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    phi = np.arctan2(p[:, 1], p[:, 0])
    k0 = int(math.floor(u_min / (2.0 * math.pi))) - 1
    k1 = int(math.ceil(u_max / (2.0 * math.pi))) + 1
    ks = np.arange(k0, k1 + 1)
    u = phi[:, None] + 2.0 * math.pi * ks[None, :]
    u = np.concatenate(
        [u, np.full((p.shape[0], 1), u_min), np.full((p.shape[0], 1), u_max)],
        axis=1,
    )
    u = np.clip(u, u_min, u_max)
    g = spiral(u, a)
    d2 = ((g - p[:, None, :]) ** 2).sum(axis=-1)
    return u[np.arange(p.shape[0]), d2.argmin(axis=1)]


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


def manifold_distance(p: np.ndarray, a: float, u_min: float, u_max: float) -> np.ndarray:
    u = nearest_u(p, a, u_min, u_max)
    return np.linalg.norm(p - spiral(u, a), axis=-1)


def project_to_manifold(p: np.ndarray, a: float, u_min: float, u_max: float) -> np.ndarray:
    u = nearest_u(p, a, u_min, u_max)
    return spiral(u, a).astype(np.float32)


def torch_project_to_manifold(
    p: torch.Tensor, a: float, u_min: float, u_max: float
) -> torch.Tensor:
    u = torch_nearest_u(p, a, u_min, u_max)
    return torch_spiral(u, a)


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
    """Exact projection onto Swiss roll feasible set h_tube, h_core, h_box <= 0."""
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


def estimate_lipschitz(
    p: np.ndarray | torch.Tensor,
    constraint_or_a: SwissRollConstraint | float,
    u_min: float | None = None,
    u_max: float | None = None,
    eps: float = 1e-4,
) -> np.ndarray | torch.Tensor:
    """Estimate local Lipschitz constant of physical projection operator P via batched PyTorch."""
    is_np = isinstance(p, np.ndarray)
    if isinstance(constraint_or_a, (SwissRollConstraint, BaseConstraint)) or hasattr(constraint_or_a, "meta"):
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
    dirs = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        device=p_t.device,
        dtype=p_t.dtype,
    )
    max_L = torch.zeros(p_t.shape[0], device=p_t.device, dtype=p_t.dtype)
    for d in dirs:
        p_pert = p_t + eps * d[None, :]
        p_pert_proj = torch_project_to_manifold(p_pert, a, u_min, u_max)
        diff = torch.norm(p_pert_proj - p_proj, dim=-1)
        max_L = torch.maximum(max_L, diff / eps)

    if is_np:
        return max_L.cpu().numpy()
    return max_L


# =====================================================================
# 3. SwissRollConstraint (BaseConstraint Implementation)
# =====================================================================

class SwissRollConstraint(BaseConstraint):
    def __init__(self, meta: SwissRollMeta):
        self.meta = meta

    def h(self, p: np.ndarray | torch.Tensor) -> dict[str, np.ndarray | torch.Tensor]:
        if isinstance(p, torch.Tensor):
            u = torch_nearest_u(
                p, float(self.meta.a), float(self.meta.u_min), float(self.meta.u_max)
            )
            g = torch_spiral(u, float(self.meta.a))
            dist_curve = torch.linalg.vector_norm(p - g, dim=-1)
            r = torch.linalg.vector_norm(p, dim=-1)
            box_max = torch.amax(torch.abs(p), dim=-1)
            return {
                "tube": dist_curve - float(self.meta.tau),
                "rad": torch.abs(r - float(self.meta.a) * u) - float(self.meta.tau),
                "core": float(self.meta.rho_min) - r,
                "box": box_max - float(self.meta.R),
            }

        p_np = np.asarray(p, dtype=np.float64)
        r = np.linalg.norm(p_np, axis=-1)
        u = nearest_u(p_np, self.meta.a, self.meta.u_min, self.meta.u_max)
        return {
            "tube": manifold_distance(p_np, self.meta.a, self.meta.u_min, self.meta.u_max)
            - self.meta.tau,
            "rad": np.abs(r - self.meta.a * u) - self.meta.tau,
            "core": self.meta.rho_min - r,
            "box": np.max(np.abs(p_np), axis=-1) - self.meta.R,
        }

    def cost(self, p: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(p, torch.Tensor):
            u = torch_nearest_u(p, float(self.meta.a), float(self.meta.u_min), float(self.meta.u_max))
            g = torch_spiral(u, float(self.meta.a))
            return ((p - g) ** 2).sum(dim=-1)
        return manifold_distance(p, self.meta.a, self.meta.u_min, self.meta.u_max) ** 2

    def project(self, p: np.ndarray) -> np.ndarray:
        return project_to_manifold(p, self.meta.a, self.meta.u_min, self.meta.u_max)

    def project_feasible(
        self, p: torch.Tensor | np.ndarray, buffer: float = 1e-4
    ) -> torch.Tensor | np.ndarray:
        is_np = isinstance(p, np.ndarray)
        p_t = torch.from_numpy(p.astype(np.float32)) if is_np else p
        res = project_feasible(
            p_t,
            float(self.meta.a),
            float(self.meta.u_min),
            float(self.meta.u_max),
            float(self.meta.tau),
            float(self.meta.rho_min),
            float(self.meta.R),
            buffer=buffer,
        )
        return res.cpu().numpy() if is_np else res

    def project_physical(
        self, p: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        is_np = isinstance(p, np.ndarray)
        p_t = torch.from_numpy(p.astype(np.float32)) if is_np else p
        res = torch_project_to_manifold(
            p_t, float(self.meta.a), float(self.meta.u_min), float(self.meta.u_max)
        )
        return res.cpu().numpy() if is_np else res

    def estimate_lipschitz(
        self, p: torch.Tensor | np.ndarray, eps: float = 1e-4
    ) -> torch.Tensor | np.ndarray:
        return estimate_lipschitz(p, self, eps=eps)

    def radius_error(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        r = np.linalg.norm(p, axis=-1)
        u = nearest_u(p, self.meta.a, self.meta.u_min, self.meta.u_max)
        return np.abs(r - self.meta.a * u)

    def curve(self, n: int = 400) -> np.ndarray:
        u = np.linspace(self.meta.u_min, self.meta.u_max, n)
        return spiral(u, self.meta.a)

    def energy(
        self,
        p: torch.Tensor,
        w_tube: float = 1.0,
        w_core: float = 1.0,
        w_box: float = 1.0,
        w_cost: float = 0.0,
        slack: float = 0.01,
    ) -> torch.Tensor:
        meta = self.meta
        proj = torch.from_numpy(self.project(p.detach().cpu().numpy())).to(
            device=p.device, dtype=p.dtype
        )
        diff = p - proj
        d = diff.norm(dim=-1)
        r = p.norm(dim=-1)
        tube = (d - (meta.tau - slack)).clamp_min(0.0)
        core = ((meta.rho_min + slack) - r).clamp_min(0.0)
        box = (p.abs() - (meta.R - slack)).clamp_min(0.0)
        return (
            w_tube * tube.square()
            + w_cost * d.square()
            + w_core * core.square()
            + w_box * box.square().sum(dim=-1)
        )

    def progress(self, p: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        meta = self.meta
        eps = 1e-12
        if isinstance(p, torch.Tensor):
            p_np = p.detach().cpu().numpy().astype(np.float64)
            u = nearest_u(p_np, meta.a, meta.u_min, meta.u_max)
            frac = (u - meta.u_min) / max(meta.u_max - meta.u_min, eps)
            return torch.from_numpy(np.clip(frac, 0.0, 1.0)).to(
                device=p.device, dtype=p.dtype
            )
        p_np = np.asarray(p, dtype=np.float64)
        u = nearest_u(p_np, meta.a, meta.u_min, meta.u_max)
        frac = (u - meta.u_min) / max(meta.u_max - meta.u_min, eps)
        return np.clip(frac, 0.0, 1.0)

    def energy_grad(
        self,
        p: np.ndarray,
        w_tube: float = 1.0,
        w_core: float = 1.0,
        w_box: float = 1.0,
        w_cost: float = 0.0,
        slack: float = 0.01,
    ) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        meta = self.meta
        proj = self.project(p).astype(np.float64)
        diff = p - proj
        d = np.linalg.norm(diff, axis=-1, keepdims=True)
        unit = diff / np.clip(d, 1e-12, None)
        tube = np.maximum(d - (meta.tau - slack), 0.0)
        grad = 2.0 * w_tube * tube * unit + 2.0 * w_cost * diff

        r = np.linalg.norm(p, axis=-1, keepdims=True)
        core = np.maximum((meta.rho_min + slack) - r, 0.0)
        grad = grad - 2.0 * w_core * core * (p / np.clip(r, 1e-12, None))

        box = np.maximum(np.abs(p) - (meta.R - slack), 0.0)
        grad = grad + 2.0 * w_box * box * np.sign(p)
        return grad

    def get_fmbf(
        self,
        *,
        radius_eps: float = 1.0e-2,
        tube_margin: float = 2.0e-3,
        box_temperature: float = 1.0e-3,
    ) -> SwissRollFMBF:
        return SwissRollFMBF(
            self.meta,
            radius_eps=radius_eps,
            tube_margin=tube_margin,
            box_temperature=box_temperature,
        )


# =====================================================================
# 4. Smooth FMBF Implementation for SafeFlow
# =====================================================================

@dataclass(frozen=True)
class TerminalFilterStats:
    filtered: int


class SwissRollFMBF:
    """Smooth h>=0 barriers in original point coordinates for SafeFlow."""

    names = ("tube", "core", "outer", "box")
    reference_names = ("tube", "core", "box")

    def __init__(
        self,
        meta: SwissRollMeta,
        *,
        radius_eps: float = 1.0e-2,
        tube_margin: float = 2.0e-3,
        box_temperature: float = 1.0e-3,
    ) -> None:
        if radius_eps <= 0.0 or box_temperature <= 0.0:
            raise ValueError("smooth barrier scales must be positive")
        if not 0.0 <= tube_margin < float(meta.tau):
            raise ValueError("tube_margin must lie in [0, tau)")
        self.meta = meta
        self.radius_eps = float(radius_eps)
        self.tube_radius = float(meta.tau) - float(tube_margin)
        self.box_temperature = float(box_temperature)

        a = float(meta.a)
        ratio = self.tube_radius / (2.0 * a)
        if ratio >= 1.0:
            raise ValueError("smooth tube radius must be smaller than 2 * a")
        self.phase_error_bound = 2.0 * a * math.asin(ratio)
        self.inner_radius = a * float(meta.u_min) + self.phase_error_bound
        self.outer_radius = a * float(meta.u_max) - self.phase_error_bound
        if self.inner_radius >= self.outer_radius:
            raise ValueError("smooth radial guards leave an empty spiral interval")
        eps2 = self.radius_eps * self.radius_eps
        self.smooth_inner_radius = math.sqrt(self.inner_radius**2 + eps2)
        self.smooth_outer_radius = math.sqrt(self.outer_radius**2 + eps2)
        smoothing_error = self.smooth_inner_radius - self.inner_radius
        if self.phase_error_bound + smoothing_error >= float(meta.tau):
            raise ValueError("smooth tube parameters do not imply reference safety")

    def values_and_gradients(self, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if p.shape[-1] != 2:
            raise ValueError(f"expected [..., 2], got {tuple(p.shape)}")
        batch_shape = p.shape[:-1]
        flat = p.reshape(-1, 2)
        x, y = flat[:, 0], flat[:, 1]

        radius2 = flat.square().sum(dim=-1)
        radius = torch.sqrt(radius2 + self.radius_eps * self.radius_eps)
        a = float(self.meta.a)
        phase = radius / a
        phase_cos = torch.cos(phase)
        phase_sin = torch.sin(phase)
        alignment_numerator = x * phase_cos + y * phase_sin
        alignment = alignment_numerator / radius
        phase_distance2 = 2.0 * a * a * (1.0 - alignment)
        h_tube = (
            self.tube_radius * self.tube_radius - phase_distance2
        ) / (2.0 * self.tube_radius)

        phase_tangent = (-x * phase_sin + y * phase_cos) / a
        grad_numerator = torch.stack((phase_cos, phase_sin), dim=-1)
        grad_numerator = grad_numerator + phase_tangent.unsqueeze(-1) * flat / radius.unsqueeze(-1)
        grad_alignment = grad_numerator / radius.unsqueeze(-1)
        grad_alignment = grad_alignment - (
            alignment_numerator / radius.pow(3)
        ).unsqueeze(-1) * flat
        grad_tube = (a * a / self.tube_radius) * grad_alignment

        radial_gradient = flat / radius.unsqueeze(-1)
        h_core = radius - self.smooth_inner_radius
        grad_core = radial_gradient
        h_outer = self.smooth_outer_radius - radius
        grad_outer = -radial_gradient

        box_faces = torch.stack((x, -x, y, -y), dim=-1)
        box_logits = box_faces / self.box_temperature
        box_weights = torch.softmax(box_logits, dim=-1)
        smooth_box = self.box_temperature * torch.logsumexp(box_logits, dim=-1)
        grad_smooth_box = torch.stack(
            (
                box_weights[:, 0] - box_weights[:, 1],
                box_weights[:, 2] - box_weights[:, 3],
            ),
            dim=-1,
        )
        h_box = float(self.meta.R) - smooth_box
        grad_box = -grad_smooth_box

        values = torch.stack((h_tube, h_core, h_outer, h_box), dim=-1)
        gradients = torch.stack((grad_tube, grad_core, grad_outer, grad_box), dim=-2)
        return (
            values.reshape(*batch_shape, len(self.names)),
            gradients.reshape(*batch_shape, len(self.names), 2),
        )

    def terminal_filter(
        self,
        p: np.ndarray,
        *,
        max_iter: int = 100,
        ftol: float = 1.0e-7,
        constraint_tol: float = 1.0e-7,
    ) -> tuple[np.ndarray, TerminalFilterStats]:
        points = np.asarray(p, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"expected [N, 2], got {points.shape}")
        out = points.copy()
        values, _ = self._numpy_values_and_gradients(out)
        indices = np.flatnonzero((values < -float(constraint_tol)).any(axis=-1))

        for idx in indices:
            target = points[idx]
            seed = self._initial_point(target)
            objective_scale = max(float(self.meta.R) ** 2, 1.0)

            def objective(candidate: np.ndarray) -> float:
                return 0.5 * float(np.square(candidate - target).sum()) / objective_scale

            def objective_jac(candidate: np.ndarray) -> np.ndarray:
                return (np.asarray(candidate, dtype=np.float64) - target) / objective_scale

            def barrier_values(candidate: np.ndarray) -> np.ndarray:
                candidate_values, _ = self._numpy_values_and_gradients(candidate[None, :])
                return candidate_values[0]

            def barrier_jac(candidate: np.ndarray) -> np.ndarray:
                _, candidate_gradients = self._numpy_values_and_gradients(candidate[None, :])
                return candidate_gradients[0]

            result = minimize(
                objective,
                seed,
                jac=objective_jac,
                method="SLSQP",
                constraints={"type": "ineq", "fun": barrier_values, "jac": barrier_jac},
                bounds=[(-float(self.meta.R), float(self.meta.R))] * 2,
                options={"maxiter": int(max_iter), "ftol": float(ftol), "disp": False},
            )
            candidate = np.asarray(result.x, dtype=np.float64).reshape(2)
            candidate_values = barrier_values(candidate)
            if not result.success or not np.isfinite(candidate).all():
                raise RuntimeError(
                    "terminal safety filter failed: "
                    f"sample={idx}, status={result.status}, min_h={candidate_values.min():.3e}, "
                    f"message={result.message}"
                )
            if float(candidate_values.min()) < -float(constraint_tol):
                raise RuntimeError(
                    "terminal safety filter returned an infeasible point: "
                    f"sample={idx}, min_h={candidate_values.min():.3e}"
                )
            out[idx] = candidate

        final_values, _ = self._numpy_values_and_gradients(out)
        if float(final_values.min()) < -float(constraint_tol):
            raise RuntimeError("terminal safety filter returned an unsafe batch")
        self._check_reference_constraints(out, float(constraint_tol))
        return out, TerminalFilterStats(filtered=int(indices.size))

    def _initial_point(self, target: np.ndarray) -> np.ndarray:
        meta = self.meta
        u = nearest_u(
            np.asarray(target, dtype=np.float64).reshape(1, 2),
            float(meta.a),
            float(meta.u_min),
            float(meta.u_max),
        )[0]
        guard_u = self.phase_error_bound / float(meta.a)
        u = np.clip(u, float(meta.u_min) + guard_u, float(meta.u_max) - guard_u)
        return spiral(np.array([u]), float(meta.a))[0]

    def _numpy_values_and_gradients(
        self, p: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        tensor = torch.as_tensor(p, dtype=torch.float64)
        values, gradients = self.values_and_gradients(tensor)
        return values.detach().numpy(), gradients.detach().numpy()

    def _check_reference_constraints(self, p: np.ndarray, tol: float) -> None:
        reference_h = SwissRollConstraint(self.meta).h(p)
        if not all(
            bool((reference_h[name] <= tol).all()) for name in self.reference_names
        ):
            raise RuntimeError("smooth terminal filter violated the Exp-01 constraints")


# =====================================================================
# 5. Dataset Bundle Builders & Serialization
# =====================================================================

def meta_from_dict(d: dict) -> SwissRollMeta:
    payload = dict(d)
    payload["mean"] = (float(payload["mean"][0]), float(payload["mean"][1]))
    payload["std"] = (float(payload["std"][0]), float(payload["std"][1]))
    return SwissRollMeta(**payload)


def resolve_cache_dir(cfg: DictConfig) -> Path | None:
    raw = cfg.data.get("cache_dir", None)
    if not raw:
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = ROOT / path
    return path


def save_swiss_roll(
    cache_dir: str | Path,
    train_raw: np.ndarray,
    eval_raw: np.ndarray,
    meta: SwissRollMeta,
) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "train.npy", np.asarray(train_raw, dtype=np.float32))
    np.save(cache_dir / "eval.npy", np.asarray(eval_raw, dtype=np.float32))
    (cache_dir / "meta.json").write_text(json.dumps(asdict(meta), indent=2))


def bundle_from_arrays(
    train_raw: np.ndarray, eval_raw: np.ndarray, meta: SwissRollMeta
) -> DataBundle:
    mean = np.asarray(meta.mean, dtype=np.float32)
    std = np.asarray(meta.std, dtype=np.float32)
    train_z = (train_raw - mean) / std
    eval_z = (eval_raw - mean) / std
    constraint = SwissRollConstraint(meta)
    return DataBundle(
        train=PointDataset(torch.from_numpy(train_z.astype(np.float32))),
        train_raw=np.asarray(train_raw, dtype=np.float32),
        eval_raw=np.asarray(eval_raw, dtype=np.float32),
        eval_z=torch.from_numpy(eval_z.astype(np.float32)),
        mean=torch.from_numpy(mean),
        std=torch.from_numpy(std),
        constraint=constraint,
        meta=meta,
        meta_dict=asdict(meta),
    )


def load_swiss_roll(cache_dir: str | Path) -> DataBundle:
    cache_dir = Path(cache_dir)
    train_raw = np.load(cache_dir / "train.npy")
    eval_raw = np.load(cache_dir / "eval.npy")
    meta = meta_from_dict(json.loads((cache_dir / "meta.json").read_text()))
    return bundle_from_arrays(train_raw, eval_raw, meta)


def _u_range(cfg: DictConfig) -> tuple[float, float, float]:
    a = float(cfg.data.a)
    n_turns = float(cfg.data.n_turns)
    u_min = float(cfg.data.get("u_min", 1.5 * math.pi))
    if "u_max" in cfg.data:
        u_max = float(cfg.data.u_max)
    else:
        u_max = u_min + n_turns * 2.0 * math.pi
    return a, u_min, u_max


@register_dataset("swiss_roll")
def build_swiss_roll(cfg: DictConfig) -> DataBundle:
    cache_dir = resolve_cache_dir(cfg)
    regenerate = bool(cfg.data.get("regenerate", False))
    if cache_dir is not None and (cache_dir / "meta.json").is_file() and not regenerate:
        return load_swiss_roll(cache_dir)

    a, u_min, u_max = _u_range(cfg)
    n_turns = (u_max - u_min) / (2.0 * math.pi)
    sigma = float(cfg.data.sigma_obs)
    n_train = int(cfg.data.n_train)
    n_eval = int(cfg.data.n_eval)
    margin = float(cfg.data.margin)
    seed = int(cfg.seed)
    tau = float(cfg.data.get("tau", 3.0 * sigma))

    rng = np.random.default_rng(seed)
    train_raw, _ = sample_points(n_train, a, u_min, u_max, sigma, rng)
    eval_raw, _ = sample_points(n_eval, a, u_min, u_max, sigma, rng)

    R = float(np.abs(train_raw).max()) + margin
    rho_min = a * u_min - tau
    mean = train_raw.mean(axis=0)
    std = train_raw.std(axis=0).clip(min=1e-6)

    meta = SwissRollMeta(
        a=a,
        u_min=u_min,
        u_max=u_max,
        n_turns=n_turns,
        sigma_obs=sigma,
        n_train=n_train,
        n_eval=n_eval,
        margin=margin,
        tau=tau,
        rho_min=rho_min,
        R=R,
        seed=seed,
        mean=(float(mean[0]), float(mean[1])),
        std=(float(std[0]), float(std[1])),
        arc_length=arc_length(a, u_min, u_max),
    )
    if cache_dir is not None:
        save_swiss_roll(cache_dir, train_raw, eval_raw, meta)
    return bundle_from_arrays(train_raw, eval_raw, meta)


# Legacy normalization aliases
from data.base import denormalize, normalize  # noqa: E402, F401
