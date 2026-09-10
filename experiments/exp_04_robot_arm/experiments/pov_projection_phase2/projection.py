"""Goal-aware Acados trajectory projection for POV Phase 2.

The solver keeps the upstream discrete jerk dynamics, joint/velocity/
acceleration bounds, and local convex collision corridors.  The projected
trajectory's last returned knot (stage ``N-1``) is made goal-aware either by a
hard Cartesian tolerance or by a strong nonlinear least-squares goal penalty.

Unlike Phase 1, an unsuccessful solve never returns an unconverged iterate: the
original candidate is returned with ``solver_success=False``.
"""

from __future__ import annotations

import contextlib
import io
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import casadi as ca
import numpy as np
from acados_template import ACADOS_INFTY, AcadosModel, AcadosOcp, AcadosOcpSolver

from safe_flow_mpc.ConvexSetFinder import ConvexSetFinder
from safe_flow_mpc.RobotModel import RobotModel
from safe_flow_mpc.utils import normalize_set_size

from experiments.pov_projection.metrics import constraint_snapshot


GoalMode = Literal["hard", "soft"]


@dataclass
class ProjectionDiagnostics:
    solver_success: bool
    solver_status: int
    solver_runtime_sec: float
    total_runtime_sec: float
    solver_iterations: int
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
    corridor_collision_count: int
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_solver(
    *,
    horizon: int,
    dt: float,
    goal_mode: GoalMode,
    goal_tolerance: float,
    goal_weight: float,
    build: bool,
) -> tuple[AcadosOcpSolver, int]:
    """Build a Phase-2-specific Acados OCP and return its stage yref size."""

    nr_joints = 7
    max_set_size = 15
    nr_p_col = 10
    robot_model = RobotModel()

    q = ca.SX.sym("q", nr_joints)
    dq = ca.SX.sym("dq", nr_joints)
    ddq = ca.SX.sym("ddq", nr_joints)
    uprev = ca.SX.sym("uprev", nr_joints)
    u = ca.SX.sym("u", nr_joints)
    x = ca.vertcat(q, dq, ddq, uprev)

    q_new = (
        ddq * dt**2 / 2.0
        + dq * dt
        + q
        + uprev * dt**3 / 8.0
        + u * dt**3 / 24.0
    )
    dq_new = dq + ddq * dt + uprev * dt**2 / 3.0 + u * dt**2 / 6.0
    ddq_new = ddq + uprev * dt / 2.0 + u * dt / 2.0

    model = AcadosModel()
    model.name = f"pov_phase2_goal_{goal_mode}"
    model.x = x
    model.u = u
    model.disc_dyn_expr = ca.vertcat(q_new, dq_new, ddq_new, u)

    collision_constraints = ca.SX.sym("collision_constraints", 0)
    corridor_params = ca.SX.sym("corridor_params", 0)
    for index in range(nr_p_col):
        position = robot_model.fk_pos_col(q, index)
        a_set = ca.SX.sym(f"a_set_{index}", max_set_size, 3)
        b_set = ca.SX.sym(f"b_set_{index}", max_set_size)
        corridor_params = ca.vertcat(
            corridor_params, a_set.reshape((-1, 1)), b_set
        )
        collision_constraints = ca.vertcat(
            collision_constraints, a_set @ position - b_set
        )

    goal = ca.SX.sym("cartesian_goal", 3)
    goal_activation = ca.SX.sym("goal_activation", 1)
    model.p = ca.vertcat(corridor_params, goal, goal_activation)
    goal_residual = robot_model.fk_pos(q) - goal

    ocp = AcadosOcp()
    code_dir = Path(f"/tmp/acados_code_pov_phase2_{goal_mode}")
    json_path = Path(f"/tmp/acados_ocp_pov_phase2_{goal_mode}.json")
    ocp.code_export_directory = str(code_dir)
    ocp.model = model
    ocp.solver_options.N_horizon = horizon
    ocp.solver_options.tf = horizon * dt

    nx = int(x.rows())
    nu = int(u.rows())
    base_ny = nx + nu
    if goal_mode == "soft":
        model.cost_y_expr = ca.vertcat(x, u, goal_activation * goal_residual)
        stage_yref_size = base_ny + 3
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.W = np.eye(stage_yref_size)
        ocp.cost.W[-3:, -3:] *= goal_weight
        ocp.cost.yref = np.zeros(stage_yref_size)
    else:
        stage_yref_size = base_ny
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.Vx = np.zeros((base_ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((base_ny, nu))
        ocp.cost.Vu[-nu:, :] = np.eye(nu)
        ocp.cost.W = np.eye(base_ny)
        ocp.cost.yref = np.zeros(base_ny)

    # Preserve the Phase-1 weighting: q tracks the candidate strongly while
    # derivative/control components act as a small smoothness regularizer.
    ocp.cost.W[nr_joints : 2 * nr_joints, nr_joints : 2 * nr_joints] *= 1e-3
    ocp.cost.W[2 * nr_joints : 3 * nr_joints, 2 * nr_joints : 3 * nr_joints] *= 1e-3
    ocp.cost.W[3 * nr_joints : base_ny, 3 * nr_joints : base_ny] *= 1e-4
    ocp.cost.W_0 = ocp.cost.W.copy()

    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.Vx_e = np.eye(nx)[nr_joints:, :]
    ocp.cost.W_e = 10.0 * np.eye(nx - nr_joints)
    ocp.cost.yref_e = np.zeros(nx - nr_joints)

    ocp.constraints.lbu = robot_model.u_min * np.ones(nu)
    ocp.constraints.ubu = robot_model.u_max * np.ones(nu)
    ocp.constraints.idxbu = np.arange(nu)
    ocp.constraints.x0 = np.zeros(nx)
    ocp.constraints.lbx = np.concatenate(
        (
            robot_model.q_lim_lower,
            robot_model.dq_lim_lower,
            robot_model.ddq_lim_lower,
            ocp.constraints.lbu,
        )
    )
    ocp.constraints.ubx = np.concatenate(
        (
            robot_model.q_lim_upper,
            robot_model.dq_lim_upper,
            robot_model.ddq_lim_upper,
            ocp.constraints.ubu,
        )
    )
    ocp.constraints.idxbx = np.arange(nx)

    number_collision_constraints = int(collision_constraints.rows())
    if goal_mode == "hard":
        hard_goal = goal_activation * (
            ca.sumsqr(goal_residual) - goal_tolerance**2
        )
        model.con_h_expr = ca.vertcat(collision_constraints, hard_goal)
        ocp.constraints.lh = np.concatenate(
            (-ACADOS_INFTY * np.ones(number_collision_constraints), [-ACADOS_INFTY])
        )
        ocp.constraints.uh = np.zeros(number_collision_constraints + 1)
    else:
        model.con_h_expr = collision_constraints
        ocp.constraints.lh = -ACADOS_INFTY * np.ones(number_collision_constraints)
        ocp.constraints.uh = np.zeros(number_collision_constraints)

    # Collision corridors stay soft as upstream.  The hard Cartesian goal, when
    # selected, is deliberately excluded from idxsh and has no slack.
    ocp.constraints.idxsh = np.arange(number_collision_constraints)
    ocp.cost.Zl = np.ones(number_collision_constraints)
    ocp.cost.Zu = np.ones(number_collision_constraints)
    ocp.cost.zl = 1000.0 * np.ones(number_collision_constraints)
    ocp.cost.zu = 1000.0 * np.ones(number_collision_constraints)

    ocp.parameter_values = np.zeros(int(model.p.rows()))
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.print_level = 0
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.globalization = "FIXED_STEP"
    ocp.solver_options.nlp_solver_max_iter = 50
    ocp.solver_options.nlp_solver_tol_stat = 1e-6
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"

    if build:
        AcadosOcpSolver.generate(ocp, json_file=str(json_path))
        solver = AcadosOcpSolver(ocp, json_file=str(json_path))
    else:
        solver = AcadosOcpSolver(
            ocp,
            json_file=str(json_path),
            build=False,
            generate=False,
        )
    return solver, stage_yref_size


class GoalAwareTrajectoryProjector:
    """Local weighted trajectory projector with an explicit Cartesian goal."""

    def __init__(
        self,
        obstacle_manager: Any,
        robot_model: Any,
        *,
        horizon: int = 16,
        dt: float = 0.1,
        goal_mode: GoalMode = "hard",
        goal_tolerance: float = 0.03,
        goal_weight: float = 1000.0,
        max_projection_iterations: int = 12,
        build: bool = False,
        workspace_min: tuple[float, float, float] = (-0.25, -1.3, 0.1),
        workspace_max: tuple[float, float, float] = (1.3, 1.3, 1.5),
    ) -> None:
        self.obstacle_manager = obstacle_manager
        self.robot_model = robot_model
        self.horizon = horizon
        self.dt = dt
        self.goal_mode = goal_mode
        self.goal_tolerance = goal_tolerance
        self.goal_weight = goal_weight
        self.max_projection_iterations = max_projection_iterations
        self.solver, self.stage_yref_size = _build_solver(
            horizon=horizon,
            dt=dt,
            goal_mode=goal_mode,
            goal_tolerance=goal_tolerance,
            goal_weight=goal_weight,
            build=build,
        )
        self.set_finder = ConvexSetFinder(
            obstacle_manager,
            e_max=list(workspace_max),
            e_min=list(workspace_min),
            max_set_size=30,
        )

    @property
    def formulation(self) -> str:
        if self.goal_mode == "hard":
            return (
                "hard Cartesian constraint at returned knot H-1: "
                f"||FK(q_H)-x_goal|| <= {self.goal_tolerance:g} m"
            )
        return (
            "strong soft Cartesian cost at returned knot H-1: "
            f"{self.goal_weight:g} * ||FK(q_H)-x_goal||^2"
        )

    def _warm_start(self, candidate: np.ndarray, start_state: np.ndarray) -> None:
        q = np.asarray(candidate, dtype=float).copy()
        q[0] = np.asarray(start_state, dtype=float)
        q_ext = np.vstack((q, q[-1]))
        dq = np.vstack((np.zeros((1, 7)), np.diff(q_ext, axis=0) / self.dt))
        ddq = np.vstack((np.zeros((1, 7)), np.diff(dq, axis=0) / self.dt))
        jerk = np.vstack((np.zeros((1, 7)), np.diff(ddq, axis=0) / self.dt))
        x = np.hstack((q_ext, dq, ddq, jerk))
        for stage in range(self.horizon + 1):
            self.solver.set(stage, "x", x[stage])
        for stage in range(self.horizon):
            self.solver.set(stage, "u", jerk[stage + 1])

    def _stage_parameters(
        self, trajectory: np.ndarray, goal: np.ndarray
    ) -> tuple[list[np.ndarray], int]:
        parameters: list[np.ndarray] = []
        collision_count = 0
        for stage in range(self.horizon):
            q1 = trajectory[stage]
            q0 = trajectory[max(0, stage - 1)]
            sets = []
            for index, radius in enumerate(self.robot_model.col_joint_sizes):
                p0 = self.robot_model.fk_pos_col(q0, index)
                p1 = self.robot_model.fk_pos_col(q1, index)
                with contextlib.redirect_stdout(io.StringIO()):
                    a_set, b_set, collision = self.set_finder.find_set_collision_avoidance(
                        p0, p1, limit_space=False, e_max=0.7
                    )
                collision_count += int(collision)
                sets.append([a_set, b_set - radius])
            normalized = normalize_set_size(sets, 15)
            corridor = np.concatenate(
                [np.concatenate((a.T.flatten(), b)) for a, b in normalized]
            )
            activation = float(stage == self.horizon - 1)
            parameters.append(
                np.concatenate((corridor, np.asarray(goal)[:3], [activation]))
            )
        return parameters, collision_count

    def _load_iterate(self) -> np.ndarray:
        x_opt = np.asarray(
            [self.solver.get(stage, "x") for stage in range(self.horizon + 1)]
        )
        return x_opt[:-1, :7]

    def _solve_once(
        self,
        candidate: np.ndarray,
        start_state: np.ndarray,
        goal: np.ndarray,
        linearization: np.ndarray,
    ) -> tuple[np.ndarray, int, float, float, int]:
        self._warm_start(linearization, start_state)
        parameters, collision_count = self._stage_parameters(linearization, goal)
        x0 = np.concatenate((start_state, np.zeros(21)))
        self.solver.set(0, "x", x0)
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)
        for stage in range(self.horizon):
            if self.goal_mode == "soft":
                yref = np.concatenate((candidate[stage], np.zeros(28 + 3)))
            else:
                yref = np.concatenate((candidate[stage], np.zeros(28)))
            self.solver.set(stage, "yref", yref)
            self.solver.set(stage, "p", parameters[stage])

        started = time.perf_counter()
        status = int(self.solver.solve())
        elapsed = time.perf_counter() - started
        slacks = [self.solver.get(stage, "su") for stage in range(1, self.horizon)]
        max_slack = float(max((np.max(x) for x in slacks), default=0.0))
        return self._load_iterate(), status, max_slack, elapsed, collision_count

    def project_trajectory(
        self,
        candidate_trajectory: np.ndarray,
        start_state: np.ndarray,
        goal: np.ndarray,
        obstacles: Any | None = None,
    ) -> tuple[np.ndarray, ProjectionDiagnostics]:
        del obstacles
        candidate = np.asarray(candidate_trajectory, dtype=float).reshape(
            self.horizon, 7
        )
        start_state = np.asarray(start_state, dtype=float)
        before = constraint_snapshot(
            candidate, goal, self.robot_model, self.obstacle_manager, self.dt
        )
        self.solver.reset()
        started = time.perf_counter()
        projected_iterate = candidate.copy()
        linearization = candidate.copy()
        success = False
        error = ""
        status = -1
        solver_runtime = 0.0
        collision_count = 0
        iterations = 0
        for iterations in range(1, self.max_projection_iterations + 1):
            try:
                projected_iterate, status, max_slack, elapsed, collisions = (
                    self._solve_once(candidate, start_state, goal, linearization)
                )
                solver_runtime += elapsed
                collision_count += collisions
                finite = bool(np.isfinite(projected_iterate).all())
                goal_error = (
                    float(
                        np.linalg.norm(
                            self.robot_model.fk_pos(projected_iterate[-1])
                            - np.asarray(goal)[:3]
                        )
                    )
                    if finite
                    else float("inf")
                )
                hard_goal_ok = (
                    self.goal_mode != "hard"
                    or goal_error <= self.goal_tolerance + 1e-4
                )
                success = status == 0 and max_slack <= 1e-3 and finite and hard_goal_ok
                if success:
                    error = ""
                    break
                error = (
                    f"Acados projection failed: status={status}, "
                    f"max_slack={max_slack:.6g}, finite={finite}, "
                    f"goal_error={goal_error:.6g}"
                )
                if not finite:
                    break
                linearization = projected_iterate
            except (RuntimeError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                break

        # Mandatory Phase-2 rule: never consume an unconverged correction.
        projected = projected_iterate if success else candidate.copy()
        total_runtime = time.perf_counter() - started
        after = constraint_snapshot(
            projected, goal, self.robot_model, self.obstacle_manager, self.dt
        )
        correction = projected - candidate
        smoothness = float(np.square(np.diff(projected, n=2, axis=0)).sum())
        objective = float(np.square(correction).sum() + 1e-3 * smoothness)
        if self.goal_mode == "soft":
            objective += self.goal_weight * after["goal_error"] ** 2
        diagnostics = ProjectionDiagnostics(
            solver_success=success,
            solver_status=status,
            solver_runtime_sec=solver_runtime,
            total_runtime_sec=total_runtime,
            solver_iterations=iterations,
            objective_value=objective,
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
            corridor_collision_count=collision_count,
            error=error,
        )
        return projected, diagnostics
