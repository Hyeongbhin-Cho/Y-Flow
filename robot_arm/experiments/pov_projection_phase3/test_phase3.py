"""CPU-only integrity tests for the Phase-3 trigger and gate."""

import pandas as pd

from experiments.pov_projection_phase3.run_phase3 import (
    METHODS,
    hard_safety_violation,
    phase3a_gate,
)


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


def test_goal_error_is_excluded_from_adaptive_trigger():
    assert not hard_safety_violation(_snapshot(goal_error=1e12))


def test_every_hard_safety_component_can_trigger():
    assert hard_safety_violation(_snapshot(min_clearance=-1e-3))
    assert hard_safety_violation(_snapshot(joint_violation=1e-3))
    assert hard_safety_violation(_snapshot(velocity_violation=1e-3))
    assert hard_safety_violation(_snapshot(acceleration_violation=1e-3))


def test_gate_requires_call_and_latency_reduction():
    rows = []
    for method in METHODS:
        for sample in range(10):
            rows.append(
                {
                    "method": method,
                    "task_id": 0,
                    "seed": sample,
                    "collision": 0.0,
                    "projection_calls": 7.0,
                    "planning_latency_sec": 1.0,
                    "projection_solver_latency_sec": 0.8,
                    "terminal_goal_error": 0.4,
                }
            )
    raw = pd.DataFrame(rows)
    transition_rows = []
    for adaptive in ("POV_L05_ADAPTIVE", "POV_L025_ADAPTIVE"):
        for reference in ("PLAIN_FM", "POV_L05_ALWAYS"):
            for reference_state in ("safe", "collision"):
                for adaptive_state in ("safe", "collision"):
                    transition_rows.append(
                        {
                            "adaptive_method": adaptive,
                            "reference_method": reference,
                            "reference_state": reference_state,
                            "adaptive_state": adaptive_state,
                            "count": 10
                            if reference_state == adaptive_state == "safe"
                            else 0,
                        }
                    )
    gate = phase3a_gate(raw, pd.DataFrame(transition_rows))
    assert not gate["calls_reduced_at_least_25pct"]
    assert not gate["latency_reduced_meaningfully"]
    assert not gate["passed"]
