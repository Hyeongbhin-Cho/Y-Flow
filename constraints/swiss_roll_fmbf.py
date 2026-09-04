# -*- coding: utf-8 -*-
# constraints/swiss_roll_fmbf.py
"""Smooth paper-sign FMBFs and terminal projection for Exp-01."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import minimize

from constraints.swiss_roll import SwissRollConstraint
from data.swiss_roll import SwissRollMeta, nearest_u, spiral


@dataclass(frozen=True)
class TerminalFilterStats:
    filtered: int


class SwissRollFMBF:
    """Smooth h>=0 barriers in original point coordinates.

    The tube is expressed through the periodic phase relation of an
    Archimedean spiral, avoiding nearest-point branches and absolute values.
    Smooth radial guards keep that infinite spiral inside the finite Exp-01
    parameter interval. The box uses a log-sum-exp smooth maximum.
    """

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
