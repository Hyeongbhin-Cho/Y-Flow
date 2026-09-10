"""Robot-trajectory metrics shared by the POV samplers and projector."""

from __future__ import annotations

from typing import Any

import numpy as np


def signed_distance_to_aabb(point: np.ndarray, bbox: np.ndarray) -> float:
    """Signed point-to-box distance: positive outside and negative inside."""

    point = np.asarray(point, dtype=float)
    bbox = np.asarray(bbox, dtype=float)
    lower, upper = bbox[:3], bbox[3:]
    outside = np.maximum(np.maximum(lower - point, point - upper), 0.0)
    if np.any(outside > 0.0):
        return float(np.linalg.norm(outside))
    return -float(np.min(np.minimum(point - lower, upper - point)))


def minimum_clearance(
    trajectory: np.ndarray, robot_model: Any, obstacle_manager: Any
) -> float:
    """Minimum collision-sphere clearance from the true obstacle boxes."""

    import pinocchio as pin

    _, _, bboxes = obstacle_manager.get_obstacles()
    if not bboxes:
        return float("inf")
    minimum = float("inf")
    for q in np.asarray(trajectory):
        data = robot_model.model.createData()
        pin.forwardKinematics(robot_model.model, data, q)
        pin.framesForwardKinematics(robot_model.model, data, q)
        for index, radius in enumerate(robot_model.col_joint_sizes):
            object_id = robot_model.col_ids[index]
            center = (
                data.oMi[object_id].translation
                if index < 4
                else data.oMf[object_id].translation
            )
            center = np.asarray(center, dtype=float).reshape(3)
            for bbox in bboxes:
                clearance = signed_distance_to_aabb(center, bbox) - float(radius)
                minimum = min(minimum, clearance)
    return float(minimum)


def trajectory_metrics(
    trajectory: np.ndarray,
    q_goal: np.ndarray,
    p_goal: np.ndarray,
    q_reference: np.ndarray,
    robot_model: Any,
    obstacle_manager: Any,
    dt: float = 0.1,
    goal_tolerance: float = 0.03,
    violation_tolerance: float = 1e-5,
) -> dict[str, float]:
    """Compute safety, task-quality, and path metrics for one trajectory."""

    q = np.asarray(trajectory, dtype=float)
    dq = np.diff(q, axis=0) / dt
    ddq = np.diff(dq, axis=0) / dt
    dddq = np.diff(ddq, axis=0) / dt

    joint_low = np.maximum(robot_model.q_lim_lower - q, 0.0)
    joint_high = np.maximum(q - robot_model.q_lim_upper, 0.0)
    joint_violation = float(np.max(np.maximum(joint_low, joint_high), initial=0.0))

    velocity_violation = 0.0
    if dq.size:
        velocity_violation = float(
            np.max(
                np.maximum(
                    np.maximum(robot_model.dq_lim_lower - dq, 0.0),
                    np.maximum(dq - robot_model.dq_lim_upper, 0.0),
                ),
                initial=0.0,
            )
        )
    acceleration_violation = 0.0
    if ddq.size:
        acceleration_violation = float(
            np.max(
                np.maximum(
                    np.maximum(robot_model.ddq_lim_lower - ddq, 0.0),
                    np.maximum(ddq - robot_model.ddq_lim_upper, 0.0),
                ),
                initial=0.0,
            )
        )

    clearance = minimum_clearance(q, robot_model, obstacle_manager)
    ee = np.asarray([robot_model.fk_pos(x) for x in q], dtype=float)
    goal_error = float(np.linalg.norm(ee[-1] - np.asarray(p_goal)[:3]))
    joint_path = float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
    ee_path = float(np.linalg.norm(np.diff(ee, axis=0), axis=1).sum())
    smoothness = float(np.square(np.diff(q, n=2, axis=0)).sum())
    acceleration_cost = float(np.square(ddq).sum() * dt) if ddq.size else 0.0
    jerk_cost = float(np.square(dddq).sum() * dt) if dddq.size else 0.0

    reference_distance = float("nan")
    if q_reference.shape == q.shape:
        reference_distance = float(np.linalg.norm(q - q_reference, axis=1).mean())

    collision = clearance < -violation_tolerance
    feasible = (
        not collision
        and joint_violation <= violation_tolerance
        and velocity_violation <= violation_tolerance
        and acceleration_violation <= violation_tolerance
    )
    return {
        "collision": float(collision),
        "success": float(feasible and goal_error <= goal_tolerance),
        "min_clearance": clearance,
        "joint_limit_violation": joint_violation,
        "velocity_violation": velocity_violation,
        "acceleration_violation": acceleration_violation,
        "terminal_goal_error": goal_error,
        "joint_path_length": joint_path,
        "ee_path_length": ee_path,
        "smoothness": smoothness,
        "acceleration_cost": acceleration_cost,
        "jerk_cost": jerk_cost,
        "reference_distance": reference_distance,
        "terminal_joint_error": float(np.linalg.norm(q[-1] - q_goal)),
    }


def constraint_snapshot(
    trajectory: np.ndarray,
    p_goal: np.ndarray,
    robot_model: Any,
    obstacle_manager: Any,
    dt: float = 0.1,
) -> dict[str, float]:
    """Compact constraint diagnostics without requiring a reference trajectory."""

    q = np.asarray(trajectory, dtype=float)
    dq = np.diff(q, axis=0) / dt
    ddq = np.diff(dq, axis=0) / dt
    joint_violation = float(
        np.max(
            np.maximum(
                np.maximum(robot_model.q_lim_lower - q, 0.0),
                np.maximum(q - robot_model.q_lim_upper, 0.0),
            ),
            initial=0.0,
        )
    )
    velocity_violation = (
        float(
            np.max(
                np.maximum(
                    np.maximum(robot_model.dq_lim_lower - dq, 0.0),
                    np.maximum(dq - robot_model.dq_lim_upper, 0.0),
                ),
                initial=0.0,
            )
        )
        if dq.size
        else 0.0
    )
    acceleration_violation = (
        float(
            np.max(
                np.maximum(
                    np.maximum(robot_model.ddq_lim_lower - ddq, 0.0),
                    np.maximum(ddq - robot_model.ddq_lim_upper, 0.0),
                ),
                initial=0.0,
            )
        )
        if ddq.size
        else 0.0
    )
    clearance = minimum_clearance(q, robot_model, obstacle_manager)
    goal_error = float(np.linalg.norm(robot_model.fk_pos(q[-1]) - np.asarray(p_goal)[:3]))
    total_violation = (
        max(-clearance, 0.0)
        + joint_violation
        + velocity_violation
        + acceleration_violation
    )
    return {
        "min_clearance": clearance,
        "joint_violation": joint_violation,
        "velocity_violation": velocity_violation,
        "acceleration_violation": acceleration_violation,
        "goal_error": goal_error,
        "total_violation": float(total_violation),
    }
