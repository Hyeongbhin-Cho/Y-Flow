import unittest

import numpy as np

from experiments.pov_projection.metrics import signed_distance_to_aabb
from experiments.pov_projection.sampling import primary_pov_euler_step


class MetricTests(unittest.TestCase):
    def test_signed_distance_is_positive_outside(self):
        box = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        self.assertAlmostEqual(signed_distance_to_aabb(np.array([2, 0.5, 0.5]), box), 1.0)

    def test_signed_distance_is_negative_inside(self):
        box = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        self.assertAlmostEqual(signed_distance_to_aabb(np.array([0.5, 0.5, 0.5]), box), -0.5)

    def test_primary_pov_rule_has_no_one_minus_t_normalization(self):
        current = np.array([2.0])
        projected_endpoint = np.array([6.0])
        np.testing.assert_allclose(
            primary_pov_euler_step(current, projected_endpoint, dt=0.25),
            np.array([3.0]),
        )


if __name__ == "__main__":
    unittest.main()
