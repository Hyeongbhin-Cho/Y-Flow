"""Acados-backed trajectory projection used by all projection baselines.

This deliberately wraps the repository's existing Acados safety-filter OCP. Its
stage cost is squared joint-space distance to the candidate trajectory with the
upstream smoothness weights. Collision avoidance is enforced through a local
convex corridor with soft nonlinear constraints. It is therefore a weighted,
local projection surrogate, not an exact global Euclidean projection.
"""

from __future__ import annotations

import contextlib
import io
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from safe_flow_mpc.SafetyFilter import SafetyFilterAcados
from safe_flow_mpc.utils import normalize_set_size

from .metrics import constraint_snapshot


@dataclass
class ProjectionDiagnostics:
    solver_success: bool
    solver_runtime_sec: float
    total_runtime_sec: float
    objective_value: float
    correction_norm: float
    min_clearance_before: float
    min_clearance_after: float
    total_violation_before: float
    total_violation_after: float
    joint_violation_before: float
    joint_violation_after: float
    velocity_violation_before: float
    velocity_violation_after: float
    acceleration_violation_before: float
    acceleration_violation_after: float
    goal_error_before: float
    goal_error_after: float
    solver_iterations: int
    corridor_collision_count: int
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrajectoryProjector:
    """Clean projection interface around the upstream Acados OCP."""

    def __init__(
        self,
        obstacle_manager: Any,
        robot_model: Any,
        horizon: int = 16,
        dt: float = 0.1,
        build: bool = False,
        workspace_min: tuple[float, float, float] = (-0.25, -1.3, 0.1),
        workspace_max: tuple[float, float, float] = (1.3, 1.3, 1.5),
        max_projection_iterations: int = 12,
    ) -> None:
        self.obstacle_manager = obstacle_manager
        self.robot_model = robot_model
        self.horizon = horizon
        self.dt = dt
        self.max_projection_iterations = max_projection_iterations
        self.filter = SafetyFilterAcados(
            N=horizon,
            smooth=True,
            use_term=True,
            use_sets=True,
            obstacle_manager=obstacle_manager,
            build=build,
            workspace_min=list(workspace_min),
            workspace_max=list(workspace_max),
        )

    def _set_candidate_warm_start(
        self, candidate: np.ndarray, start_state: np.ndarray
    ) -> None:
        q = np.asarray(candidate, dtype=float).copy()
        q[0] = np.asarray(start_state, dtype=float)
        q_ext = np.vstack((q, q[-1]))
        dq = np.vstack((np.zeros((1, 7)), np.diff(q_ext, axis=0) / self.dt))
        ddq = np.vstack((np.zeros((1, 7)), np.diff(dq, axis=0) / self.dt))
        jerk = np.vstack((np.zeros((1, 7)), np.diff(ddq, axis=0) / self.dt))
        self.filter.q_last = q_ext.T
        self.filter.dq_last = dq.T
        self.filter.ddq_last = ddq.T
        self.filter.dddq_last = jerk.T
        self.filter.q = np.asarray(start_state, dtype=float).copy()
        self.filter.dq = np.zeros(7)
        self.filter.ddq = np.zeros(7)
        self.filter.dddq = np.zeros(7)
        self.filter.n_steps = 1
        self.filter.updated = True
        self.filter.t_array = []

    def _load_last_acados_iterate(self) -> np.ndarray:
        """Promote a failed RTI iterate to the next warm start."""

        x_opt = np.asarray(
            [self.filter.ocp_solver.get(i, "x") for i in range(self.horizon + 1)]
        )
        u_opt = np.asarray(
            [self.filter.ocp_solver.get(i, "u") for i in range(self.horizon)]
        ).T
        q_opt = x_opt[:, :7].T
        dq_opt = x_opt[:, 7:14].T
        ddq_opt = x_opt[:, 14:21].T
        dddq0 = np.zeros((7, 1))
        self.filter.q_last = q_opt
        self.filter.dq_last = dq_opt
        self.filter.ddq_last = ddq_opt
        self.filter.dddq_last = np.hstack((dddq0, u_opt))
        self.filter.updated = True
        return q_opt[:, :-1].T

    def _stage_parameters(
        self, trajectory: np.ndarray
    ) -> tuple[list[np.ndarray], int]:
        """Build a local collision corridor for every trajectory segment."""

        parameters: list[np.ndarray] = []
        collision_count = 0
        radii = self.robot_model.col_joint_sizes
        for stage in range(self.horizon):
            q1 = trajectory[stage]
            q0 = trajectory[max(0, stage - 1)]
            sets = []
            for index, radius in enumerate(radii):
                p0 = self.robot_model.fk_pos_col(q0, index)
                p1 = self.robot_model.fk_pos_col(q1, index)
                # The upstream helper prints every touching segment. Preserve the
                # information as a count in diagnostics without flooding logs.
                with contextlib.redirect_stdout(io.StringIO()):
                    a_set, b_set, collision = (
                        self.filter.set_finder.find_set_collision_avoidance(
                            p0, p1, limit_space=False, e_max=0.7
                        )
                    )
                collision_count += int(collision)
                sets.append([a_set, b_set - radius])
            normalized = normalize_set_size(sets, 15)
            parameters.append(
                np.concatenate(
                    [
                        np.concatenate((a_set.T.flatten(), b_set))
                        for a_set, b_set in normalized
                    ]
                )
            )
        return parameters, collision_count

    def _solve_once(
        self,
        candidate: np.ndarray,
        start_state: np.ndarray,
        linearization: np.ndarray,
    ) -> tuple[np.ndarray, int, float, float, int]:
        """Solve one RTI projection using stage-specific local corridors."""

        solver = self.filter.ocp_solver
        self._set_candidate_warm_start(linearization, start_state)
        parameters, collision_count = self._stage_parameters(linearization)
        x_mat = np.vstack(
            (
                self.filter.q_last,
                self.filter.dq_last,
                self.filter.ddq_last,
                self.filter.dddq_last,
            )
        )
        x0 = np.concatenate((start_state, np.zeros(21)))
        solver.set(0, "x", x0)
        solver.set(0, "lbx", x0)
        solver.set(0, "ubx", x0)
        for stage in range(1, self.horizon + 1):
            solver.set(stage, "x", x_mat[:, stage])
        yref = np.hstack((candidate, np.zeros((self.horizon, 28))))
        for stage in range(self.horizon):
            solver.set(stage, "yref", yref[stage])
            solver.set(stage, "p", parameters[stage])

        started = time.perf_counter()
        status = int(solver.solve())
        solver_runtime = time.perf_counter() - started
        slacks = [solver.get(stage, "su") for stage in range(1, self.horizon)]
        max_slack = float(np.max(slacks)) if slacks else 0.0
        projected = self._load_last_acados_iterate()
        return projected, status, max_slack, solver_runtime, collision_count

    def project_trajectory(
        self,
        candidate_trajectory: np.ndarray,
        start_state: np.ndarray,
        goal: np.ndarray,
        obstacles: Any | None = None,
    ) -> tuple[np.ndarray, ProjectionDiagnostics]:
        """Project a candidate and always expose failures in diagnostics.

        ``obstacles`` is accepted to keep the requested interface explicit; the
        projector owns the matching upstream ``ObstacleManager`` instance.
        """

        del obstacles
        candidate = np.asarray(candidate_trajectory, dtype=float).reshape(
            self.horizon, 7
        )
        before = constraint_snapshot(
            candidate, goal, self.robot_model, self.obstacle_manager, self.dt
        )
        # P must depend only on this candidate and scenario. Do not leak Acados
        # dual/primal warm-start state from a previous method or seed.
        self.filter.ocp_solver.reset()
        started = time.perf_counter()
        error = ""
        success = False
        projected = candidate.copy()
        linearization = candidate.copy()
        solver_runtime = 0.0
        corridor_collision_count = 0
        iterations = 0
        for iterations in range(1, self.max_projection_iterations + 1):
            try:
                projected, status, max_slack, elapsed, collisions = self._solve_once(
                    candidate, np.asarray(start_state, dtype=float), linearization
                )
                solver_runtime += elapsed
                corridor_collision_count += collisions
                finite = bool(np.isfinite(projected).all())
                success = status == 0 and max_slack <= 1e-3 and finite
                if success:
                    error = ""
                    break
                error = (
                    f"Acados projection did not converge: status={status}, "
                    f"max_slack={max_slack:.6g}, finite={finite}"
                )
                if not finite:
                    projected = candidate.copy()
                    break
                linearization = projected
            except (RuntimeError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                projected = candidate.copy()
                break
        total_runtime = time.perf_counter() - started
        after = constraint_snapshot(
            projected, goal, self.robot_model, self.obstacle_manager, self.dt
        )
        correction = projected - candidate
        diagnostics = ProjectionDiagnostics(
            solver_success=success,
            solver_runtime_sec=solver_runtime,
            total_runtime_sec=total_runtime,
            objective_value=float(np.square(correction).sum()),
            correction_norm=float(np.linalg.norm(correction)),
            min_clearance_before=before["min_clearance"],
            min_clearance_after=after["min_clearance"],
            total_violation_before=before["total_violation"],
            total_violation_after=after["total_violation"],
            joint_violation_before=before["joint_violation"],
            joint_violation_after=after["joint_violation"],
            velocity_violation_before=before["velocity_violation"],
            velocity_violation_after=after["velocity_violation"],
            acceleration_violation_before=before["acceleration_violation"],
            acceleration_violation_after=after["acceleration_violation"],
            goal_error_before=before["goal_error"],
            goal_error_after=after["goal_error"],
            solver_iterations=iterations,
            corridor_collision_count=corridor_collision_count,
            error=error,
        )
        return projected, diagnostics
