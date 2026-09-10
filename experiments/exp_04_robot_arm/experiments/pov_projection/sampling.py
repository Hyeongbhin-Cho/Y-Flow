"""Small, testable sampling rules for the POV experiment."""

from __future__ import annotations

from typing import TypeVar


Array = TypeVar("Array")


def primary_pov_velocity(current: Array, projected_endpoint: Array) -> Array:
    """Primary POV direction; intentionally not normalized by ``1 - t``."""

    return projected_endpoint - current


def primary_pov_euler_step(
    current: Array, projected_endpoint: Array, dt: float
) -> Array:
    return current + dt * primary_pov_velocity(current, projected_endpoint)
