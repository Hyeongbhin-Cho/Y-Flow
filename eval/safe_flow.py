# -*- coding: utf-8 -*-
# eval/safe_flow.py
"""Training-free SafeFlow with CFMBF-QP correction and a terminal safety filter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import DictConfig

from data.base import barrier_gain, build_dataset, solve_composite_fmbf
from eval._backbone import load_frozen_velocity
from eval.sample_result import SampleResult


@dataclass
class _Diagnostics:
    nfe: int = 0
    correction_evals: int = 0
    correction_items: int = 0
    correction_norm_sum: torch.Tensor | None = None
    correction_norm_max: torch.Tensor | None = None
    slack_items: int = 0
    slack_sum: torch.Tensor | None = None
    slack_max: torch.Tensor | None = None
    min_raw_residual: torch.Tensor | None = None
    min_relaxed_residual: torch.Tensor | None = None

    def update(
        self,
        correction: torch.Tensor,
        slack: torch.Tensor,
        raw: torch.Tensor,
        relaxed: torch.Tensor,
    ) -> None:
        norms = correction.norm(dim=-1).detach()
        slack_value = slack.detach()
        self.correction_evals += 1
        self.correction_items += int(norms.numel())
        self.correction_norm_sum = _add(self.correction_norm_sum, norms.sum())
        self.correction_norm_max = _maximum(self.correction_norm_max, norms.max())
        self.slack_items += int(slack_value.numel())
        self.slack_sum = _add(self.slack_sum, slack_value.sum())
        self.slack_max = _maximum(self.slack_max, slack_value.max())
        self.min_raw_residual = _minimum(self.min_raw_residual, raw.detach().min())
        self.min_relaxed_residual = _minimum(
            self.min_relaxed_residual, relaxed.detach().min()
        )


class _SafeVelocity:
    def __init__(self, model, method, fmbf, mean, std, cfg, diagnostics: _Diagnostics):
        self.model = model
        self.method = method
        self.fmbf = fmbf
        self.mean = mean
        self.std = std
        self.cfg = cfg
        self.diagnostics = diagnostics

    def __call__(
        self,
        t: float | torch.Tensor,
        z: torch.Tensor,
        *,
        corrected: bool,
    ) -> torch.Tensor:
        t_scalar = torch.as_tensor(t, device=z.device, dtype=z.dtype)
        t_batch = t_scalar.expand(z.shape[0])
        v_z = self.method.velocity(self.model, z, t_batch)
        self.diagnostics.nfe += 1
        if not corrected or not bool(self.cfg.enabled):
            return v_z

        p = z * self.std + self.mean
        v_p = v_z * self.std
        h, gradients = self.fmbf.values_and_gradients(p)
        gain = barrier_gain(
            t_scalar,
            h,
            phi0=float(self.cfg.phi0),
            schedule=str(self.cfg.phi_schedule),
            gamma=float(self.cfg.phi_gamma),
            omega=float(self.cfg.phi_omega),
            terminal_eps=float(self.cfg.terminal_eps),
        )
        a = (gradients * v_p.unsqueeze(-2)).sum(dim=-1) + gain * h
        solution = solve_composite_fmbf(
            a,
            gradients,
            slack_weight=float(self.cfg.slack_weight),
            active_tol=float(self.cfg.qp_active_tol),
        )
        self.diagnostics.update(
            solution.correction,
            solution.slack,
            solution.raw_residual,
            solution.relaxed_residual,
        )
        return v_z + solution.correction / self.std


@torch.no_grad()
def sample(cfg: DictConfig, device: torch.device, x0: torch.Tensor) -> SampleResult:
    model, method = load_frozen_velocity(cfg, device)
    bundle = build_dataset(cfg)
    constraint = bundle["constraint"]
    mean = bundle["mean"].to(device=device, dtype=x0.dtype)
    std = bundle["std"].to(device=device, dtype=x0.dtype)
    fmbf = constraint.get_fmbf(
        radius_eps=float(cfg.safeflow.smooth_radius_eps),
        tube_margin=float(cfg.safeflow.smooth_tube_margin),
        box_temperature=float(cfg.safeflow.smooth_box_temperature),
    )
    diagnostics = _Diagnostics()
    field = _SafeVelocity(model, method, fmbf, mean, std, cfg.safeflow, diagnostics)
    integrator = str(cfg.safeflow.integrator).lower()
    if integrator == "euler":
        z_pre = _sample_euler(cfg, field, x0)
    elif integrator == "dopri5":
        z_pre = _sample_dopri5(cfg, field, x0)
    else:
        raise ValueError(f"unknown SafeFlow integrator: {integrator}")

    p_pre = (z_pre * std + mean).detach().cpu().numpy().astype(np.float64)
    existing_h = constraint.h(p_pre)
    safe_pre = np.stack(list(existing_h.values()), axis=-1).max(axis=-1) <= 0.0

    terminal_filtered = 0
    p_final = p_pre
    if bool(cfg.safeflow.enabled) and bool(cfg.safeflow.terminal_filter.enabled):
        p_final, terminal_stats = fmbf.terminal_filter(
            p_pre,
            max_iter=int(cfg.safeflow.terminal_filter.max_iter),
            ftol=float(cfg.safeflow.terminal_filter.ftol),
            constraint_tol=float(cfg.safeflow.terminal_filter.constraint_tol),
        )
        terminal_filtered = terminal_stats.filtered

    if terminal_filtered == 0:
        z_final = z_pre
    else:
        z_final = torch.from_numpy(p_final).to(device=device, dtype=x0.dtype)
        z_final = (z_final - mean) / std
    correction_mean = _as_float(diagnostics.correction_norm_sum) / max(
        diagnostics.correction_items, 1
    )
    slack_mean = _as_float(diagnostics.slack_sum) / max(diagnostics.slack_items, 1)
    min_raw = _as_float(diagnostics.min_raw_residual)
    min_relaxed = _as_float(diagnostics.min_relaxed_residual)
    payload = {
        "integrator": integrator,
        "nfe": diagnostics.nfe,
        "correction_evals": diagnostics.correction_evals,
        "mean_correction_norm": correction_mean,
        "max_correction_norm": _as_float(diagnostics.correction_norm_max),
        "mean_slack": slack_mean,
        "max_slack": _as_float(diagnostics.slack_max),
        "min_raw_fmbf_residual": min_raw,
        "min_relaxed_fmbf_residual": min_relaxed,
        "pre_filter_safe_ratio": float(safe_pre.mean()),
        "terminal_filter_rate": float(terminal_filtered / max(x0.shape[0], 1)),
    }
    return SampleResult(samples=z_final, diagnostics=payload)


def _add(current: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
    return value if current is None else current + value


def _maximum(current: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
    return value if current is None else torch.maximum(current, value)


def _minimum(current: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
    return value if current is None else torch.minimum(current, value)


def _as_float(value: torch.Tensor | None) -> float:
    return 0.0 if value is None else float(value.cpu())


def _sample_euler(cfg: DictConfig, field: _SafeVelocity, x0: torch.Tensor) -> torch.Tensor:
    steps = int(cfg.sample.n_steps)
    if steps <= 0:
        raise ValueError("sample.n_steps must be positive")
    dt = 1.0 / steps
    t_on = float(cfg.safeflow.t_on)
    z = x0
    for i in range(steps):
        t = i / steps
        velocity = field(t, z, corrected=t >= t_on)
        z = z + dt * velocity
    return z


def _sample_dopri5(cfg: DictConfig, field: _SafeVelocity, x0: torch.Tensor) -> torch.Tensor:
    from torchdiffeq import odeint

    options = cfg.safeflow.dopri5
    rtol = float(options.rtol)
    atol = float(options.atol)
    terminal_time = 1.0 - float(cfg.safeflow.terminal_eps)
    t_on = min(max(float(cfg.safeflow.t_on), 0.0), terminal_time)
    z = x0
    if t_on > 0.0:
        time = torch.tensor([0.0, t_on], device=x0.device, dtype=x0.dtype)
        z = odeint(
            lambda t, state: field(t, state, corrected=False),
            z,
            time,
            method="dopri5",
            rtol=rtol,
            atol=atol,
        )[-1]
    if terminal_time > t_on:
        time = torch.tensor([t_on, terminal_time], device=x0.device, dtype=x0.dtype)
        z = odeint(
            lambda t, state: field(t, state, corrected=True),
            z,
            time,
            method="dopri5",
            rtol=rtol,
            atol=atol,
        )[-1]
    return z
