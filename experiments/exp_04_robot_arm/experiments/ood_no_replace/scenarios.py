"""Deterministic, geometry-only OOD obstacle scenario generation."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.pov_projection.metrics import minimum_clearance, signed_distance_to_aabb
from experiments.pov_projection.run_phase1 import load_task
from safe_flow_mpc.SafeFlowMPC.ObstacleManager import (
    ObstacleDescription,
    ObstacleManager,
)
from safe_flow_mpc.utils import compute_polytope_vertices, make_box, normalize_set_size


CATEGORIES = (
    "SHIFTED_OBSTACLE",
    "ENLARGED_OBSTACLE",
    "MULTI_OBSTACLE",
    "NARROW_PASSAGE",
    "GOAL_NEAR_OBSTACLE",
)
PREFIXES = {
    "SHIFTED_OBSTACLE": "ood_shifted",
    "ENLARGED_OBSTACLE": "ood_enlarged",
    "MULTI_OBSTACLE": "ood_multi",
    "NARROW_PASSAGE": "ood_narrow",
    "GOAL_NEAR_OBSTACLE": "ood_goalnear",
}
DEFAULT_BBOXES = (
    np.array([0.45, -0.06, 0.725, 1.10, 0.06, 0.90], dtype=float),
    np.array([0.45, -0.06, 0.000, 1.10, 0.06, 0.38], dtype=float),
)


def set_bboxes(manager: ObstacleManager, bboxes: list[list[float]] | list[np.ndarray]) -> None:
    manager.obstacles.clear()
    for index, values in enumerate(bboxes):
        bbox = np.asarray(values, dtype=float)
        a_set, b_set = make_box(bbox[:3], bbox[3:])
        points = np.asarray(compute_polytope_vertices(a_set, b_set))
        manager.obstacles[f"ood_obs{index}"] = ObstacleDescription(
            set=normalize_set_size([[a_set, b_set]])[0],
            points=points,
            bbox=np.concatenate((points.min(axis=0), points.max(axis=0))),
        )


def _clip_box(box: np.ndarray) -> np.ndarray:
    lower = np.maximum(box[:3], np.array([-0.20, -1.10, 0.01]))
    upper = np.minimum(box[3:], np.array([1.25, 1.10, 1.30]))
    return np.concatenate((lower, upper))


def _scaled(box: np.ndarray, scale: np.ndarray) -> np.ndarray:
    center = (box[:3] + box[3:]) / 2
    half = (box[3:] - box[:3]) * scale / 2
    return _clip_box(np.concatenate((center - half, center + half)))


def _candidate(
    category: str,
    index: int,
    attempt: int,
    p_goal: np.ndarray,
) -> tuple[list[np.ndarray], str]:
    category_index = CATEGORIES.index(category)
    rng = np.random.default_rng(20260908 + 10000 * category_index + 101 * index + attempt)
    severity = ("mild", "medium", "hard")[index % 3]
    low, high = (x.copy() for x in DEFAULT_BBOXES)

    if category == "SHIFTED_OBSTACLE":
        signs = rng.choice([-1.0, 1.0], size=3)
        shift = signs * rng.uniform([0.12, 0.10, 0.03], [0.32, 0.30, 0.14])
        boxes = [_clip_box(low + np.tile(shift, 2)), _clip_box(high + np.tile(shift, 2))]
    elif category == "ENLARGED_OBSTACLE":
        base = {"mild": 1.10, "medium": 1.22, "hard": 1.35}[severity]
        scale = np.array([base, base + rng.uniform(0.05, 0.25), base])
        boxes = [_scaled(low, scale), _scaled(high, scale)]
    elif category == "MULTI_OBSTACLE":
        dims = rng.uniform([0.12, 0.10, 0.12], [0.25, 0.22, 0.30])
        center = rng.uniform([0.20, -0.72, 0.18], [1.02, 0.72, 0.92])
        extra = _clip_box(np.concatenate((center - dims / 2, center + dims / 2)))
        boxes = [low, high, extra]
    elif category == "NARROW_PASSAGE":
        lower = high.copy()
        upper = low.copy()
        gap = {"mild": 0.30, "medium": 0.24, "hard": 0.18}[severity]
        gap += rng.uniform(-0.015, 0.015)
        center_z = rng.uniform(0.50, 0.60)
        y_half = rng.uniform(0.06, 0.12)
        x_shift = rng.uniform(-0.22, 0.22)
        y_center = rng.uniform(-0.28, 0.28)
        lower[0] += x_shift
        lower[3] += x_shift
        upper[0] += x_shift
        upper[3] += x_shift
        lower[1], lower[4], lower[5] = y_center - y_half, y_center + y_half, center_z - gap / 2
        upper[1], upper[4], upper[2] = y_center - y_half, y_center + y_half, center_z + gap / 2
        boxes = [upper, lower]
    elif category == "GOAL_NEAR_OBSTACLE":
        dims = rng.uniform([0.10, 0.10, 0.10], [0.20, 0.18, 0.22])
        directions = np.eye(3).tolist() + (-np.eye(3)).tolist()
        direction = np.asarray(directions[(index + attempt) % len(directions)])
        gap = rng.uniform(0.045, 0.13)
        center = np.asarray(p_goal)[:3] + direction * (dims / 2 + gap)
        near = _clip_box(np.concatenate((center - dims / 2, center + dims / 2)))
        boxes = [low, high, near]
    else:
        raise ValueError(category)
    return boxes, severity


def _validate(
    bboxes: list[np.ndarray],
    task: dict[str, Any],
    robot_model: Any,
    manager: ObstacleManager,
) -> tuple[bool, dict[str, Any]]:
    if any(np.any(box[3:] <= box[:3]) or not np.isfinite(box).all() for box in bboxes):
        return False, {"reason": "invalid_geometry"}
    set_bboxes(manager, bboxes)
    q = np.asarray(task["q"])
    p_goal = np.asarray(robot_model.fk_pos(q[-1]))
    start_clearance = minimum_clearance(q[:1], robot_model, manager)
    goal_clearance = minimum_clearance(q[-1:], robot_model, manager)
    reference_clearance = minimum_clearance(q, robot_model, manager)
    goal_inside = any(
        np.all(p_goal >= box[:3]) and np.all(p_goal <= box[3:]) for box in bboxes
    )
    valid = bool(
        start_clearance >= 0.0
        and goal_clearance >= 0.0
        and not goal_inside
    )
    return valid, {
        "start_clearance": float(start_clearance),
        "goal_clearance": float(goal_clearance),
        "known_reference_path_clearance": float(reference_clearance),
        "goal_inside_obstacle": bool(goal_inside),
        "reason": "accepted" if valid else "failed_geometry_validation",
    }


def generate_scenarios(robot_model: Any) -> list[dict[str, Any]]:
    """Generate exactly 20 accepted scenarios per category without FM-result filtering."""
    manager = ObstacleManager()
    scenarios: list[dict[str, Any]] = []
    for category in CATEGORIES:
        for index in range(20):
            accepted = None
            for task_offset in range(6):
                task_id = (index + task_offset) % 6
                task = load_task(task_id)
                p_goal = np.asarray(robot_model.fk_pos(task["q"][-1]))
                for attempt in range(200):
                    boxes, severity = _candidate(category, index, attempt, p_goal)
                    valid, validation = _validate(boxes, task, robot_model, manager)
                    different = not (
                        len(boxes) == len(DEFAULT_BBOXES)
                        and all(np.allclose(a, b) for a, b in zip(boxes, DEFAULT_BBOXES))
                    )
                    if valid and different:
                        accepted = {
                            "scenario_id": f"{PREFIXES[category]}_{index:03d}",
                            "category": category,
                            "category_index": index,
                            "task_id": task_id,
                            "severity": severity,
                            "task_offset": task_offset,
                            "generation_attempt": attempt,
                            "bboxes": [box.tolist() for box in boxes],
                            "validation": validation,
                        }
                        break
                if accepted is not None:
                    break
            if accepted is None:
                raise RuntimeError(f"could not construct valid {category} scenario {index}")
            scenarios.append(accepted)
    if len(scenarios) != 100 or len({x["scenario_id"] for x in scenarios}) != 100:
        raise RuntimeError("scenario count or IDs are not exactly 100 unique entries")
    return scenarios


def geometry_checker_matches_bboxes(scenario: dict[str, Any]) -> bool:
    """Confirm the signed-distance checker agrees with boxes used by plots."""
    for values in scenario["bboxes"]:
        box = np.asarray(values, dtype=float)
        center = (box[:3] + box[3:]) / 2
        outside = box[3:] + 0.1
        if signed_distance_to_aabb(center, box) >= 0:
            return False
        if signed_distance_to_aabb(outside, box) <= 0:
            return False
    return True
