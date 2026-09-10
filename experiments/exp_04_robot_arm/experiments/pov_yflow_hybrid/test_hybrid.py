"""CPU-only logic tests for the hybrid activation policy."""

from experiments.pov_projection_phase3.run_phase3 import hard_safety_violation


def _snapshot(**changes):
    values = {
        "min_clearance": 0.1,
        "joint_violation": 0.0,
        "velocity_violation": 0.0,
        "acceleration_violation": 0.0,
        "goal_error": 1e9,
    }
    values.update(changes)
    return values


def test_goal_never_triggers_safety_gate():
    assert not hard_safety_violation(_snapshot(goal_error=1e12))


def test_true_safety_components_trigger():
    assert hard_safety_violation(_snapshot(min_clearance=-1e-3))
    assert hard_safety_violation(_snapshot(joint_violation=1e-3))
    assert hard_safety_violation(_snapshot(velocity_violation=1e-3))
    assert hard_safety_violation(_snapshot(acceleration_violation=1e-3))


def test_late_grid_has_three_steps():
    assert [i for i in range(7) if i / 7 >= 0.5] == [4, 5, 6]
