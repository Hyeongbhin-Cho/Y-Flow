"""Headless matched one-shot extraction of the upstream SafeFlowMPC mechanism.

The implementation deliberately uses the repository's unmodified
``SafetyFilterAcados.step`` and a line-for-line copy of the upstream Cartesian
guidance calculation.  It avoids constructing ``SafeFlowMPC`` itself only
because that class unconditionally launches a MuJoCo viewer and owns closed-loop
simulation state.  The unsafe FM checkpoint is technically compatible and is
supplied by the Phase-3 runner.
"""

from __future__ import annotations

import contextlib
import io
import time
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from safe_flow_mpc.SafetyFilter import SafetyFilterAcados

from experiments.pov_projection.metrics import constraint_snapshot
from experiments.pov_projection.run_phase1 import model_velocity, synchronize


def upstream_cartesian_guidance(
    x_current: torch.Tensor,
    dx_flow: torch.Tensor,
    q_start: np.ndarray,
    p_goal: np.ndarray,
    robot_model: Any,
    horizon: int = 16,
) -> np.ndarray:
    """Copy of ``SafeFlowMPC._compute_guidance`` for 7-DoF trajectories."""

    q_des = (x_current + dx_flow).reshape((-1, 7)).detach().cpu().numpy()
    p_init, _, _ = robot_model.forward_kinematics(q_start, 0 * q_start)
    q_des_old = x_current.reshape((-1, 7)).detach().cpu().numpy()
    for index in range(1, horizon):
        current_pose, jacobian, _ = robot_model.forward_kinematics(
            q_des[index, :7], 0 * q_des[index, :7]
        )
        desired_pose = np.copy(p_goal[:3])
        desired_pose[2] = current_pose[2]
        initial_distance = np.linalg.norm(desired_pose - p_init[:3])
        current_distance = np.linalg.norm(desired_pose - current_pose[:3])
        if current_distance <= 0.1:
            initial_distance = np.linalg.norm(p_goal[:3] - p_init[:3])
            current_distance = np.linalg.norm(p_goal[:3] - current_pose[:3])
        scale = max(0, np.exp(-10 * (current_distance / initial_distance - 0.09)))
        scale = min(1, scale)
        pose_delta = np.zeros(6)
        pose_delta[:3] = desired_pose - current_pose[:3]
        current_rotation = R.from_rotvec(current_pose[3:])
        goal_rotation = R.from_rotvec(p_goal[3:])
        rotation_delta = current_rotation * goal_rotation.inv()
        pose_delta[3:] = -rotation_delta.as_rotvec()
        alpha = 0.001
        jacobian_pinv = (
            np.linalg.inv(jacobian.T @ jacobian + alpha * np.eye(jacobian.shape[1]))
            @ jacobian.T
        )
        desired_joint_delta = jacobian_pinv @ pose_delta
        guidance_weight = scale * (index + 1) / horizon
        q_des[index, :7] = (1 - guidance_weight) * q_des[index, :7] + guidance_weight * (
            q_des_old[index, :7] + desired_joint_delta
        )
    return q_des[:, :7]


class OriginalSafeFlowMPCOneShot:
    """Exact upstream safety-filter step semantics without the viewer/loop."""

    def __init__(
        self,
        obstacle_manager: Any,
        robot_model: Any,
        *,
        horizon: int = 16,
        build: bool = False,
    ) -> None:
        self.obstacle_manager = obstacle_manager
        self.robot_model = robot_model
        self.horizon = horizon
        self.filter = SafetyFilterAcados(
            N=horizon,
            smooth=True,
            use_term=True,
            use_sets=True,
            obstacle_manager=obstacle_manager,
            build=build,
            workspace_min=[-0.25, -1.3, 0.1],
            workspace_max=[1.3, 1.3, 1.5],
        )

    def reset(self, obstacle_manager: Any) -> None:
        self.obstacle_manager = obstacle_manager
        self.filter.ocp_solver.reset()
        self.filter.reset()
        self.filter.set_finder.set_obstacles(obstacle_manager)

    def sample(
        self,
        initial_noise: torch.Tensor,
        condition: torch.Tensor,
        field: Any,
        q_start: np.ndarray,
        p_goal: np.ndarray,
        obstacle_manager: Any,
        task_id: int,
        seed: int,
        flow_steps: int = 7,
    ) -> tuple[np.ndarray, dict[str, float], list[dict[str, Any]]]:
        self.reset(obstacle_manager)
        x = initial_noise.clone()
        dt = 1.0 / flow_steps
        network_latency = 0.0
        optimization_total = 0.0
        solver_total = 0.0
        failures = 0
        rows: list[dict[str, Any]] = []
        synchronize(field.device)
        planning_started = time.perf_counter()
        for flow_step in range(flow_steps):
            t = flow_step / flow_steps
            velocity, elapsed = model_velocity(field, x, t, condition)
            network_latency += elapsed
            endpoint = x + (1.0 - t) * velocity
            snapshot = constraint_snapshot(
                endpoint.detach().cpu().numpy()[0],
                p_goal,
                self.robot_model,
                obstacle_manager,
            )
            q_des = upstream_cartesian_guidance(
                x, velocity * dt, q_start, p_goal, self.robot_model, self.horizon
            )
            call_started = time.perf_counter()
            before_solver_time = self.filter.t_total
            error = ""
            success = False
            try:
                # Upstream set-finder warnings are diagnostic noise; the exact
                # returned values and RuntimeError behavior are unchanged.
                with contextlib.redirect_stdout(io.StringIO()):
                    safe_columns = self.filter.step(q_start, q_des)
                safe = safe_columns.T
                x = torch.as_tensor(
                    safe, dtype=torch.float32, device=initial_noise.device
                )[None, ...]
                success = True
            except RuntimeError as exc:
                failures += 1
                error = f"{type(exc).__name__}: {exc}"
                # This is the upstream catch-path: reuse q_last, which is only
                # promoted after a successful solve, never the failed iterate.
                safe = self.filter.q_last[:, :-1].T
                x = torch.as_tensor(
                    safe, dtype=torch.float32, device=initial_noise.device
                )[None, ...]
            call_total = time.perf_counter() - call_started
            solver_elapsed = self.filter.t_total - before_solver_time
            optimization_total += call_total
            solver_total += solver_elapsed
            rows.append(
                {
                    "method": "ORIGINAL_SAFEFLOWMPC",
                    "task_id": task_id,
                    "seed": seed,
                    "flow_step": flow_step,
                    "t": t,
                    "lambda_pov": 0.0,
                    "raw_velocity_norm": float(torch.linalg.vector_norm(velocity).item()),
                    "endpoint_constraint_violation_before_projection": snapshot[
                        "total_violation"
                    ],
                    "endpoint_goal_error": snapshot["goal_error"],
                    "endpoint_min_clearance": snapshot["min_clearance"],
                    "endpoint_feasible": float(snapshot["total_violation"] <= 1e-5),
                    "adaptive_trigger": 0.0,
                    "projection_called": 1.0,
                    "projection_solver_success": float(success),
                    "projection_solver_status": 0 if success else -1,
                    "projection_solver_iterations": 1,
                    "projection_runtime_sec": solver_elapsed,
                    "projection_total_runtime_sec": call_total,
                    "correction_norm": float(np.linalg.norm(safe - q_des)),
                    "normalized_correction_norm": 0.0,
                    "velocity_change_norm": 0.0,
                    "goal_error_before_projection": snapshot["goal_error"],
                    "goal_error_after_projection": float(
                        np.linalg.norm(self.robot_model.fk_pos(safe[-1]) - p_goal[:3])
                    ),
                    "min_clearance_before_projection": snapshot["min_clearance"],
                    "min_clearance_after_projection": constraint_snapshot(
                        safe, p_goal, self.robot_model, obstacle_manager
                    )["min_clearance"],
                    "violation_before_projection": snapshot["total_violation"],
                    "violation_after_projection": constraint_snapshot(
                        safe, p_goal, self.robot_model, obstacle_manager
                    )["total_violation"],
                    "projection_error": error,
                }
            )
        synchronize(field.device)
        return (
            x.detach().cpu().numpy()[0],
            {
                "planning_latency_sec": time.perf_counter() - planning_started,
                "fm_latency_sec": network_latency,
                "projection_latency_sec": optimization_total,
                "projection_solver_latency_sec": solver_total,
                "projection_calls": float(flow_steps),
                "failed_projection_solves": float(failures),
                "projection_correction_sum": float(
                    sum(row["correction_norm"] for row in rows)
                ),
            },
            rows,
        )
