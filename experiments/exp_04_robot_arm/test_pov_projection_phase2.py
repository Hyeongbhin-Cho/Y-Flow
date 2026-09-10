import unittest

import numpy as np

from experiments.pov_projection_phase2.sampling import (
    corrected_pov_euler_step,
    corrected_pov_velocity,
)


class CorrectedPovDynamicsTest(unittest.TestCase):
    def test_identity_projection_preserves_velocity_exactly(self):
        rng = np.random.default_rng(7)
        current = rng.normal(size=(16, 7))
        velocity = rng.normal(size=(16, 7))
        t = 6 / 7
        endpoint = current + (1 - t) * velocity
        guided, delta, normalized = corrected_pov_velocity(
            current, velocity, endpoint, t, 1.0
        )
        np.testing.assert_array_equal(delta, np.zeros_like(delta))
        np.testing.assert_array_equal(normalized, np.zeros_like(normalized))
        np.testing.assert_array_equal(guided, velocity)

    def test_lambda_scales_only_projection_residual(self):
        current = np.zeros((2, 2))
        velocity = np.ones((2, 2))
        projected = current + 0.5 * velocity + 0.2
        full, _, _ = corrected_pov_velocity(current, velocity, projected, 0.5, 1.0)
        half, _, _ = corrected_pov_velocity(current, velocity, projected, 0.5, 0.5)
        np.testing.assert_allclose(half - velocity, 0.5 * (full - velocity))

    def test_euler_step_uses_mandatory_normalized_rule(self):
        current = np.zeros((1, 1))
        velocity = np.array([[2.0]])
        projected = np.array([[1.5]])
        result, guided, delta, normalized = corrected_pov_euler_step(
            current, velocity, projected, 0.5, 0.5, 0.5
        )
        np.testing.assert_allclose(delta, [[0.5]])
        np.testing.assert_allclose(normalized, [[1.0]])
        np.testing.assert_allclose(guided, [[2.5]])
        np.testing.assert_allclose(result, [[1.25]])


if __name__ == "__main__":
    unittest.main()
