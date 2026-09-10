"""Sampling primitives for the corrected endpoint-projection dynamics."""

from __future__ import annotations

from typing import TypeVar


Array = TypeVar("Array")


def corrected_pov_velocity(
    current: Array,
    raw_velocity: Array,
    projected_endpoint: Array,
    t: float,
    lambda_pov: float,
    epsilon: float = 1e-3,
) -> tuple[Array, Array, Array]:
    """Return flow-preserving guided velocity, endpoint correction and normalized correction.

    If projection is the identity, ``delta`` is zero and the raw Flow Matching
    velocity is returned exactly.
    """

    endpoint = current + (1.0 - t) * raw_velocity
    delta = projected_endpoint - endpoint
    normalized_delta = delta / max(1.0 - t, epsilon)
    guided = raw_velocity + lambda_pov * normalized_delta
    return guided, delta, normalized_delta


def corrected_pov_euler_step(
    current: Array,
    raw_velocity: Array,
    projected_endpoint: Array,
    t: float,
    dt: float,
    lambda_pov: float,
    epsilon: float = 1e-3,
) -> tuple[Array, Array, Array, Array]:
    guided, delta, normalized_delta = corrected_pov_velocity(
        current,
        raw_velocity,
        projected_endpoint,
        t,
        lambda_pov,
        epsilon,
    )
    return current + dt * guided, guided, delta, normalized_delta
