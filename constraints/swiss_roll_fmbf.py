# -*- coding: utf-8 -*-
# constraints/swiss_roll_fmbf.py
"""Paper-sign FMBFs and terminal projection for the Exp-01 Swiss roll."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import minimize

from constraints.swiss_roll import SwissRollConstraint
from data.swiss_roll import SwissRollMeta


@dataclass(frozen=True)
class TerminalFilterStats:
    filtered: int
    fallbacks: int


class SwissRollFMBF:
    """Differentiable-almost-everywhere h>=0 barriers in original point coordinates."""

    names = ("tube", "core", "box")

    def __init__(self, meta: SwissRollMeta, eps: float = 1.0e-8):
        self.meta = meta
        self.eps = float(eps)

    def values_and_gradients(self, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if p.shape[-1] != 2:
            raise ValueError(f"expected [..., 2], got {tuple(p.shape)}")
        batch_shape = p.shape[:-1]
        flat = p.reshape(-1, 2)
        x, y = flat[:, 0], flat[:, 1]
        r = flat.norm(dim=-1)
        safe_r = r.clamp_min(self.eps)
        radial = flat / safe_r.unsqueeze(-1)
        origin = r < self.eps
        fallback_radial = torch.zeros_like(radial)
        fallback_radial[:, 0] = 1.0
        radial = torch.where(origin.unsqueeze(-1), fallback_radial, radial)
        tangent = torch.stack((-radial[:, 1], radial[:, 0]), dim=-1)

        angle = torch.atan2(y, x)
        k0 = int(math.floor(self.meta.u_min / (2.0 * math.pi))) - 1
        k1 = int(math.ceil(self.meta.u_max / (2.0 * math.pi))) + 1
        ks = torch.arange(k0, k1 + 1, device=p.device, dtype=p.dtype)
        raw_phase = angle[:, None] + 2.0 * math.pi * ks[None, :]
        raw = torch.cat(
            (
                raw_phase,
                torch.full((flat.shape[0], 1), float(self.meta.u_min), device=p.device, dtype=p.dtype),
                torch.full((flat.shape[0], 1), float(self.meta.u_max), device=p.device, dtype=p.dtype),
            ),
            dim=1,
        )
        u_candidates = raw.clamp(float(self.meta.u_min), float(self.meta.u_max))
        q_candidates = float(self.meta.a) * u_candidates.unsqueeze(-1) * torch.stack(
            (torch.cos(u_candidates), torch.sin(u_candidates)), dim=-1
        )
        d2 = (q_candidates - flat[:, None, :]).square().sum(dim=-1)
        selected = d2.argmin(dim=1)
        row = torch.arange(flat.shape[0], device=p.device)
        u = u_candidates[row, selected]
        raw_u = raw[row, selected]
        q = q_candidates[row, selected]
        distance = (flat - q).norm(dim=-1)

        phase_count = raw_phase.shape[1]
        interior = (
            (selected < phase_count)
            & (raw_u > float(self.meta.u_min))
            & (raw_u < float(self.meta.u_max))
            & (~origin)
        )
        signed_radial_error = r - float(self.meta.a) * u
        sign = torch.sign(signed_radial_error)
        interior_grad_d = sign.unsqueeze(-1) * (
            radial - float(self.meta.a) * tangent / safe_r.unsqueeze(-1)
        )
        endpoint_grad_d = (flat - q) / distance.clamp_min(self.eps).unsqueeze(-1)
        grad_distance = torch.where(interior.unsqueeze(-1), interior_grad_d, endpoint_grad_d)
        grad_distance = torch.where(
            (distance < self.eps).unsqueeze(-1), torch.zeros_like(grad_distance), grad_distance
        )

        h_tube = float(self.meta.tau) - distance
        grad_tube = -grad_distance
        h_core = r - float(self.meta.rho_min)
        grad_core = radial

        abs_flat = flat.abs()
        box_axis = abs_flat.argmax(dim=-1)
        box_value = abs_flat.gather(1, box_axis[:, None]).squeeze(1)
        box_sign = flat.gather(1, box_axis[:, None]).squeeze(1).sign()
        box_sign = torch.where(box_sign == 0.0, torch.ones_like(box_sign), box_sign)
        grad_box = torch.zeros_like(flat)
        grad_box.scatter_(1, box_axis[:, None], (-box_sign)[:, None])
        h_box = float(self.meta.R) - box_value

        values = torch.stack((h_tube, h_core, h_box), dim=-1)
        gradients = torch.stack((grad_tube, grad_core, grad_box), dim=-2)
        return (
            values.reshape(*batch_shape, len(self.names)),
            gradients.reshape(*batch_shape, len(self.names), 2),
        )

    def terminal_filter(
        self,
        p: np.ndarray,
        *,
        max_iter: int = 50,
        ftol: float = 1.0e-9,
    ) -> tuple[np.ndarray, TerminalFilterStats]:
        points = np.asarray(p, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"expected [N, 2], got {points.shape}")
        constraint = SwissRollConstraint(self.meta)
        out = points.copy()
        unsafe = ~_safe_mask(out, constraint)
        indices = np.flatnonzero(unsafe)
        if indices.size == 0:
            return out, TerminalFilterStats(filtered=0, fallbacks=0)

        safety_margin = max(1.0e-6, 10.0 * float(ftol))
        curve = constraint.curve(4096)
        curve = curve[_safe_mask(curve, constraint, margin=safety_margin)]
        if curve.shape[0] == 0:
            raise RuntimeError("Swiss-roll terminal filter has no feasible fallback point")

        fallbacks = 0
        for idx in indices:
            target = points[idx]
            seed = curve[np.square(curve - target).sum(axis=-1).argmin()]

            def objective(candidate: np.ndarray) -> float:
                return 0.5 * float(np.square(candidate - target).sum())

            constraints = [
                {
                    "type": "ineq",
                    "fun": lambda candidate, name=name: float(
                        -constraint.h(np.asarray(candidate).reshape(1, 2))[name][0]
                    ),
                }
                for name in self.names
            ]
            result = minimize(
                objective,
                seed,
                method="SLSQP",
                constraints=constraints,
                options={"maxiter": int(max_iter), "ftol": float(ftol), "disp": False},
            )
            candidate = np.asarray(result.x, dtype=np.float64).reshape(2)
            candidate = _repair_toward_seed(
                candidate, seed, constraint, margin=safety_margin
            )
            if candidate is None:
                candidate = seed
                fallbacks += 1
            out[idx] = candidate

        if not bool(_safe_mask(out, constraint).all()):
            raise RuntimeError("terminal safety filter returned an unsafe point")
        return out, TerminalFilterStats(filtered=int(indices.size), fallbacks=fallbacks)


def _safe_mask(
    p: np.ndarray,
    constraint: SwissRollConstraint,
    margin: float = 0.0,
) -> np.ndarray:
    h = constraint.h(np.asarray(p, dtype=np.float64))
    return np.stack(
        [h[name] <= -float(margin) for name in SwissRollFMBF.names], axis=-1
    ).all(axis=-1)


def _repair_toward_seed(
    candidate: np.ndarray,
    seed: np.ndarray,
    constraint: SwissRollConstraint,
    margin: float,
) -> np.ndarray | None:
    if bool(_safe_mask(candidate.reshape(1, 2), constraint, margin=margin)[0]):
        return candidate
    for fraction in (1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2):
        repaired = (1.0 - fraction) * candidate + fraction * seed
        if bool(_safe_mask(repaired.reshape(1, 2), constraint, margin=margin)[0]):
            return repaired
    return None
