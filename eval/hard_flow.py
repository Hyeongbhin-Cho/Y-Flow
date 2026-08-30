# -*- coding: utf-8 -*-
# eval/hard_flow.py
"""Training-free HardFlow sampling. Terminal h, C on predicted x_1, then affine map back."""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig
from scipy.optimize import minimize

from constraints.swiss_roll import SwissRollConstraint
from data.swiss_roll import build_swiss_roll
from eval._backbone import load_frozen_velocity

_H_NAMES = ("tube", "core", "box")


def _to_p(z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return z * std + mean


def _to_z(p: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (p - mean) / std


def _feasible(p: np.ndarray, constraint: SwissRollConstraint) -> bool:
    h = constraint.h(np.asarray(p, dtype=np.float64).reshape(1, 2))
    return all(float(h[name][0]) <= 1e-8 for name in _H_NAMES)


def _project_fallback(p_bar: np.ndarray, constraint: SwissRollConstraint) -> np.ndarray:
    p = constraint.project(p_bar.reshape(1, 2))[0]
    r = float(np.linalg.norm(p))
    if r < constraint.meta.rho_min and r > 1e-12:
        p = p * (constraint.meta.rho_min / r)
    return np.clip(p, -constraint.meta.R, constraint.meta.R)


def _solve_terminal(
    z_bar: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    constraint: SwissRollConstraint,
    lam: float,
    max_iter: int,
) -> np.ndarray:
    z_bar = np.asarray(z_bar, dtype=np.float64).reshape(2)
    mean = np.asarray(mean, dtype=np.float64).reshape(2)
    std = np.asarray(std, dtype=np.float64).reshape(2)

    def fun(z: np.ndarray) -> float:
        p = _to_p(z, mean, std)
        cost = float(constraint.cost(p.reshape(1, 2))[0])
        return cost + 0.5 * lam * float(np.sum((z - z_bar) ** 2))

    def ineq(name: str):
        return {
            "type": "ineq",
            "fun": lambda z, n=name: float(-constraint.h(_to_p(z, mean, std).reshape(1, 2))[n][0]),
        }

    res = minimize(
        fun,
        z_bar,
        method="SLSQP",
        constraints=[ineq("tube"), ineq("core"), ineq("box")],
        options={"maxiter": max_iter, "ftol": 1e-9, "disp": False},
    )
    z_hat = np.asarray(res.x, dtype=np.float64).reshape(2)
    if _feasible(_to_p(z_hat, mean, std), constraint):
        return z_hat
    p = _project_fallback(_to_p(z_bar, mean, std), constraint)
    return _to_z(p, mean, std)


@torch.no_grad()
def sample(cfg: DictConfig, device: torch.device, x0: torch.Tensor) -> torch.Tensor:
    model, method = load_frozen_velocity(cfg, device)
    bundle = build_swiss_roll(cfg)
    mean_t = bundle["mean"].to(device=device, dtype=x0.dtype)
    std_t = bundle["std"].to(device=device, dtype=x0.dtype)
    mean_np = bundle["mean"].detach().cpu().numpy()
    std_np = bundle["std"].detach().cpu().numpy()
    constraint = SwissRollConstraint(bundle["meta"])
    t_on = float(cfg.hardflow.get("t_on", 0.5))
    lambda_oc = float(cfg.hardflow.get("lambda_oc", 10.0))
    max_iter = int(cfg.hardflow.get("max_iter", 20))
    steps = int(cfg.sample.n_steps)
    dt = 1.0 / steps
    x = x0
    model.eval()
    for i in range(steps):
        t = i / steps
        t_next = (i + 1) / steps
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
        v = method.velocity(model, x, t_tensor)
        bar_x = x + dt * v
        if t < t_on and i < steps - 1:
            x = bar_x
            continue
        t_next_tensor = torch.full((x.shape[0],), t_next, device=x.device, dtype=x.dtype)
        v_next = method.velocity(model, bar_x, t_next_tensor)
        bar_x1 = bar_x + (1.0 - t_next) * v_next
        z_bar = bar_x1.detach().cpu().numpy()
        lam = lambda_oc * (t_next ** 2) / max(dt, 1e-8)
        z_star_np = np.stack(
            [
                _solve_terminal(z_bar[j], mean_np, std_np, constraint, lam, max_iter)
                for j in range(z_bar.shape[0])
            ],
            axis=0,
        )
        z_star = torch.from_numpy(z_star_np.astype(np.float32)).to(device=x.device, dtype=x.dtype)
        w0 = bar_x - t_next * v_next
        x = t_next * z_star + (1.0 - t_next) * w0
    return x
