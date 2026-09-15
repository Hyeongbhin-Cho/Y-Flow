# -*- coding: utf-8 -*-
# experiments/exp_03_pedestrian/experiments/phase1_yflow/sampling.py
"""Small, testable rules that mirror the remote-main Y-Flow update."""

from __future__ import annotations


def is_terminal_step(t: float, dt: float, eps: float = 1e-6) -> bool:
    return t + dt >= 1.0 - eps


def yflow_projection_active(t: float, dt: float, t_on: float) -> bool:
    return is_terminal_step(t, dt) or t >= t_on


def damped_target(raw, projected, damping: float):
    """x1* = raw + damping * (P(raw) - raw); damping=1 is the exact projection."""
    return raw + damping * (projected - raw)


def yflow_next_state(y_t, x1_star, t: float, dt: float):
    """Linear interpolation x_{t+dt} = x_t + dt * (x1* - x_t) / (1 - t).

    At the terminal step (t + dt = 1) this returns x1* exactly.
    """
    if is_terminal_step(t, dt):
        return x1_star
    return y_t + dt * (x1_star - y_t) / (1.0 - t)
