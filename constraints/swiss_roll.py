# -*- coding: utf-8 -*-
# constraints/swiss_roll.py

from __future__ import annotations

import numpy as np

from data.swiss_roll import SwissRollMeta, manifold_distance, nearest_u, project_to_manifold, spiral


class SwissRollConstraint:
    def __init__(self, meta: SwissRollMeta):
        self.meta = meta

    def h(self, p: np.ndarray) -> dict[str, np.ndarray]:
        p = np.asarray(p, dtype=np.float64)
        r = np.linalg.norm(p, axis=-1)
        u = nearest_u(p, self.meta.a, self.meta.u_min, self.meta.u_max)
        return {
            "tube": manifold_distance(p, self.meta.a, self.meta.u_min, self.meta.u_max) - self.meta.tau,
            "rad": np.abs(r - self.meta.a * u) - self.meta.tau,
            "core": self.meta.rho_min - r,
            "box": np.max(np.abs(p), axis=-1) - self.meta.R,
        }

    def cost(self, p: np.ndarray) -> np.ndarray:
        return manifold_distance(p, self.meta.a, self.meta.u_min, self.meta.u_max) ** 2

    def project(self, p: np.ndarray) -> np.ndarray:
        return project_to_manifold(p, self.meta.a, self.meta.u_min, self.meta.u_max)

    def radius_error(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        r = np.linalg.norm(p, axis=-1)
        u = nearest_u(p, self.meta.a, self.meta.u_min, self.meta.u_max)
        return np.abs(r - self.meta.a * u)

    def curve(self, n: int = 400) -> np.ndarray:
        u = np.linspace(self.meta.u_min, self.meta.u_max, n)
        return spiral(u, self.meta.a)
