"""CPU-only outer-loop integrity tests for the Y-Flow port."""

import numpy as np

from experiments.pov_projection_phase15_yflow_main.sampling import (
    yflow_eta,
    yflow_interpolate,
    yflow_projection_active,
)


def test_t05_schedule_has_three_active_steps_on_seven_step_grid():
    active = [
        i
        for i in range(7)
        if yflow_projection_active(i, 7, i / 7, 0.5)
    ]
    assert active == [4, 5, 6]


def test_raw_target_interpolation_equals_vanilla_euler():
    x = np.array([1.0, -2.0])
    v = np.array([0.3, 0.4])
    t = 2 / 7
    dt = 1 / 7
    raw = x + (1.0 - t) * v
    got, _ = yflow_interpolate(x, raw, t, dt, terminal=False)
    np.testing.assert_allclose(got, x + dt * v)


def test_terminal_step_replaces_state_with_optimized_target():
    x = np.array([1.0, 2.0])
    target = np.array([-3.0, 4.0])
    got, eta = yflow_interpolate(x, target, 6 / 7, 1 / 7, terminal=True)
    assert yflow_eta(6 / 7, 1 / 7, True) == 1.0
    assert eta == 1.0
    np.testing.assert_allclose(got, target)
