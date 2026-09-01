# -*- coding: utf-8 -*-
# test/test_fmbf.py

from __future__ import annotations

import unittest

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.optimize import minimize

from constraints.fmbf import barrier_gain, solve_composite_fmbf, solve_single_fmbf
from constraints.swiss_roll import SwissRollConstraint
from constraints.swiss_roll_fmbf import SwissRollFMBF
from data.swiss_roll import build_swiss_roll, spiral
from utils.paths import ROOT


class TestFMBF(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cfg = OmegaConf.load(ROOT / "configs" / "exp_01_swiss_roll.yaml")
        cls.bundle = build_swiss_roll(cfg)
        cls.constraint = SwissRollConstraint(cls.bundle["meta"])
        cls.fmbf = SwissRollFMBF(cls.bundle["meta"])

    def test_paper_gain_switches_on_barrier_sign(self) -> None:
        h = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
        early = barrier_gain(0.5, h, phi0=1.0, schedule="paper_piecewise", gamma=0.9)
        late = barrier_gain(0.95, h, phi0=1.0, schedule="paper_piecewise", gamma=0.9)
        self.assertAlmostEqual(float(early[0, 0]), 1.0)
        self.assertAlmostEqual(float(early[0, 1]), 1.5)
        self.assertAlmostEqual(float(late[0, 0]), 1.0)
        self.assertAlmostEqual(float(late[0, 1]), 20.0)

    def test_single_closed_form_satisfies_kkt(self) -> None:
        a = torch.tensor([-2.0, 1.0], dtype=torch.float64)
        b = torch.tensor([[3.0, 4.0], [1.0, -2.0]], dtype=torch.float64)
        u = solve_single_fmbf(a, b)
        residual = a + (b * u).sum(dim=-1)
        self.assertTrue(torch.all(residual >= -1.0e-12))
        self.assertTrue(torch.allclose(u[1], torch.zeros(2, dtype=torch.float64)))
        self.assertTrue(torch.allclose(u[0], torch.tensor([0.24, 0.32], dtype=torch.float64)))

    def test_composite_active_set_matches_scipy(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(8):
            a_np = rng.normal(size=3)
            b_np = rng.normal(size=(3, 2))
            solution = solve_composite_fmbf(
                torch.tensor(a_np, dtype=torch.float64),
                torch.tensor(b_np, dtype=torch.float64),
            )

            def objective(x: np.ndarray) -> float:
                return float(np.square(x[:2]).sum() + np.square(x[2:]).sum())

            constraints = [
                {
                    "type": "ineq",
                    "fun": lambda x, j=j: float(a_np[j] + b_np[j] @ x[:2] + x[2 + j]),
                    "jac": lambda _x, j=j: np.concatenate((b_np[j], np.eye(3)[j])),
                }
                for j in range(3)
            ]
            constraints.extend(
                {
                    "type": "ineq",
                    "fun": lambda x, j=j: float(x[2 + j]),
                    "jac": lambda _x, j=j: np.concatenate((np.zeros(2), np.eye(3)[j])),
                }
                for j in range(3)
            )
            scipy_result = minimize(
                objective,
                np.concatenate((np.zeros(2), np.maximum(-a_np, 0.0) + 0.1)),
                jac=lambda x: 2.0 * x,
                method="SLSQP",
                constraints=constraints,
                options={"ftol": 1.0e-12, "maxiter": 200},
            )
            self.assertTrue(scipy_result.success)
            self.assertGreaterEqual(float(solution.relaxed_residual.min()), -1.0e-9)
            ours = float(
                solution.correction.square().sum() + solution.slack.square().sum()
            )
            self.assertAlmostEqual(ours, objective(scipy_result.x), places=7)
            self.assertTrue(
                np.allclose(solution.correction.numpy(), scipy_result.x[:2], atol=2e-6)
            )

    def test_smooth_safe_set_is_conservative(self) -> None:
        rng = np.random.default_rng(42)
        radius = float(self.bundle["meta"].R)
        points = rng.uniform(-radius, radius, size=(50_000, 2))
        values, _ = self.fmbf.values_and_gradients(
            torch.tensor(points, dtype=torch.float64)
        )
        smooth_safe = (values.numpy() >= 0.0).all(axis=-1)
        self.assertGreater(int(smooth_safe.sum()), 0)
        existing = self.constraint.h(points[smooth_safe])
        for name in self.fmbf.reference_names:
            self.assertTrue(np.all(existing[name] <= 0.0))

    def test_barrier_gradients_match_finite_difference(self) -> None:
        mid_u = 0.5 * (self.bundle["meta"].u_min + self.bundle["meta"].u_max)
        center = spiral(np.array([mid_u]), self.bundle["meta"].a)[0]
        points = torch.from_numpy(
            np.asarray([[0.0, 0.0], [5.0, 5.0], center], dtype=np.float64)
        )
        _, gradients = self.fmbf.values_and_gradients(points)
        for row in range(points.shape[0]):
            eps = 1.0e-9 if row == 0 else 1.0e-6
            for dim in range(2):
                plus = points.clone()
                minus = points.clone()
                plus[row, dim] += eps
                minus[row, dim] -= eps
                value_plus, _ = self.fmbf.values_and_gradients(plus)
                value_minus, _ = self.fmbf.values_and_gradients(minus)
                numerical = (value_plus[row] - value_minus[row]) / (2.0 * eps)
                self.assertTrue(
                    torch.allclose(
                        numerical,
                        gradients[row, :, dim],
                        atol=2.0e-5,
                        rtol=2.0e-5,
                    ),
                    msg=(
                        f"row={row} dim={dim} numerical={numerical} "
                        f"analytic={gradients[row, :, dim]}"
                    ),
                )

    def test_sampled_safety_boundaries_have_nonzero_gradients(self) -> None:
        rng = np.random.default_rng(101)
        radius = float(self.bundle["meta"].R)
        points = torch.from_numpy(rng.uniform(-radius, radius, size=(50_000, 2)))
        values, gradients = self.fmbf.values_and_gradients(points)
        for index, name in enumerate(self.fmbf.names):
            near_boundary = values[:, index].abs() < 5.0e-3
            self.assertTrue(bool(near_boundary.any()), msg=f"no {name} boundary samples")
            norms = gradients[near_boundary, index].norm(dim=-1)
            self.assertGreater(float(norms.min()), 0.9, msg=name)

    def test_smooth_barriers_and_terminal_filter_are_safe(self) -> None:
        values, gradients = self.fmbf.values_and_gradients(
            torch.zeros(1, 2, dtype=torch.float64)
        )
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(torch.isfinite(gradients).all())
        filtered, stats = self.fmbf.terminal_filter(
            np.array([[0.0, 0.0], [100.0, 100.0]]), max_iter=100
        )
        self.assertEqual(stats.filtered, 2)
        h = self.constraint.h(filtered)
        for name in self.fmbf.reference_names:
            self.assertTrue(np.all(h[name] <= 0.0))

    def test_terminal_filter_does_not_hide_solver_failure(self) -> None:
        with self.assertRaises(RuntimeError):
            self.fmbf.terminal_filter(np.array([[100.0, 100.0]]), max_iter=1)


if __name__ == "__main__":
    unittest.main()
