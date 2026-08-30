# -*- coding: utf-8 -*-
# eval/metrics.py

from __future__ import annotations

import numpy as np

from constraints.swiss_roll import SwissRollConstraint


def _subsample(p: np.ndarray, n_max: int, rng: np.random.Generator) -> np.ndarray:
    if p.shape[0] <= n_max:
        return p
    idx = rng.choice(p.shape[0], n_max, replace=False)
    return p[idx]


def rbf_mmd(
    x: np.ndarray,
    y: np.ndarray,
    sigmas: tuple[float, ...] | None = None,
    n_max: int = 1024,
) -> float:
    rng = np.random.default_rng(0)
    x = _subsample(np.asarray(x, dtype=np.float64), n_max, rng)
    y = _subsample(np.asarray(y, dtype=np.float64), n_max, rng)
    if sigmas is None:
        z = np.concatenate([x, y], axis=0)
        probe = _subsample(z, min(512, z.shape[0]), rng)
        d2 = ((probe[:, None, :] - probe[None, :, :]) ** 2).sum(axis=-1)
        med = float(np.median(d2[d2 > 0])) if np.any(d2 > 0) else 1.0
        sigmas = tuple(float(np.sqrt(med) * s) for s in (0.5, 1.0, 2.0))

    def k(a: np.ndarray, b: np.ndarray, sigma: float) -> np.ndarray:
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
        return np.exp(-d2 / (2.0 * sigma * sigma + 1e-12))

    n, m = x.shape[0], y.shape[0]
    total = 0.0
    for sigma in sigmas:
        kxx = k(x, x, sigma)
        kyy = k(y, y, sigma)
        kxy = k(x, y, sigma)
        np.fill_diagonal(kxx, 0.0)
        np.fill_diagonal(kyy, 0.0)
        mmd2 = kxx.sum() / (n * (n - 1) + 1e-12)
        mmd2 += kyy.sum() / (m * (m - 1) + 1e-12)
        mmd2 -= 2.0 * kxy.mean()
        total += max(mmd2, 0.0)
    return float(total / len(sigmas))


def evaluate_points(
    samples: np.ndarray,
    test: np.ndarray,
    constraint: SwissRollConstraint,
) -> dict[str, float]:
    h = constraint.h(samples)
    stacked = np.stack([h["tube"], h["core"], h["box"]], axis=-1)
    safe = (stacked <= 0).all(axis=-1)

    def viol(arr: np.ndarray) -> tuple[float, float]:
        pos = np.maximum(arr, 0.0)
        return float((arr > 0).mean()), float(pos.mean())

    tube_rate, tube_mean = viol(h["tube"])
    core_rate, core_mean = viol(h["core"])
    box_rate, box_mean = viol(h["box"])
    return {
        "safe_ratio": float(safe.mean()),
        "tube_viol_rate": tube_rate,
        "tube_viol_mean": tube_mean,
        "core_viol_rate": core_rate,
        "core_viol_mean": core_mean,
        "box_viol_rate": box_rate,
        "box_viol_mean": box_mean,
        "mmd": rbf_mmd(samples, test),
        "radius_mae": float(constraint.radius_error(samples).mean()),
    }
