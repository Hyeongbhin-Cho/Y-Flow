"""Small, testable primitives matching the remote-main Y-Flow update."""

from __future__ import annotations

from typing import TypeVar


Array = TypeVar("Array")


def yflow_eta(t: float, dt: float, terminal: bool) -> float:
    return 1.0 if terminal else dt / max(1.0 - t, 1e-8)


def yflow_interpolate(
    current: Array,
    terminal_target: Array,
    t: float,
    dt: float,
    terminal: bool,
) -> tuple[Array, float]:
    eta = yflow_eta(t, dt, terminal)
    return (1.0 - eta) * current + eta * terminal_target, eta


def yflow_projection_active(
    flow_step: int, flow_steps: int, t: float, t_on: float
) -> bool:
    terminal = flow_step == flow_steps - 1
    return terminal or t >= t_on
