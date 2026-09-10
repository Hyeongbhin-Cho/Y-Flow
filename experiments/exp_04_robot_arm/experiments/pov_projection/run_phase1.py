"""Run the matched-seed Phase-1 POV projection experiment.

The experiment evaluates one 16-knot global-planner sample per task/seed. This
keeps start, goal, condition and initial noise identical between methods and
isolates projection placement. It does not claim to reproduce the upstream
closed-loop MPC rollout metric.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation as R

from safe_flow_mpc.RobotModel import RobotModel
from safe_flow_mpc.SafeFlowMPC import FlowMatchingField, PlannerConfig
from safe_flow_mpc.SafeFlowMPC.ObstacleManager import ObstacleManager

from experiments._layout import DATASET_ROOT, RUN_ROOT
from .metrics import constraint_snapshot, trajectory_metrics
from .projection import ProjectionDiagnostics, TrajectoryProjector
from .sampling import primary_pov_euler_step


METHODS = (
    "PLAIN_FM",
    "FINAL_PROJECTION",
    "CURRENT_STATE_PROJECTION",
    "POV_ENDPOINT_PROJECTION",
)


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def load_task(task_id: int) -> dict[str, np.ndarray]:
    data = np.load(DATASET_ROOT / f"traj_example_{task_id}.npz", allow_pickle=True)
    return {name: data[name] for name in data.files}


def build_condition(
    robot_model: RobotModel,
    q_start: np.ndarray,
    q_prev: np.ndarray,
    p_goal: np.ndarray,
    device: str,
) -> torch.Tensor:
    """Match ``SafeFlowMPC.create_condition_vector`` exactly."""

    h = robot_model.hom_transform_endeffector(q_start)
    p0 = h[:3, 3]
    r0 = h[:3, :3].flatten()
    r_final = R.from_rotvec(p_goal[3:]).as_matrix().flatten()
    collision_positions = [robot_model.fk_pos_col(q_start, i) for i in range(7)]
    values = list(q_prev) + [p0, r0, p_goal[:3], r_final] + collision_positions
    return torch.cat([torch.as_tensor(x, dtype=torch.float32) for x in values]).to(
        device
    )


def model_velocity(
    field: FlowMatchingField,
    x: torch.Tensor,
    t: float,
    condition: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    t_tensor = torch.tensor([t], dtype=torch.float32, device=x.device)
    synchronize(field.device)
    started = time.perf_counter()
    with torch.inference_mode():
        velocity = field.ema.ema_model(x, t_tensor, condition[None, :])
    synchronize(field.device)
    return velocity, time.perf_counter() - started


def projection_row(
    method: str,
    task_id: int,
    seed: int,
    flow_step: int,
    t: float,
    velocity_norm: float,
    endpoint_snapshot: dict[str, float],
    diagnostics: ProjectionDiagnostics | None,
    target: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": method,
        "task_id": task_id,
        "seed": seed,
        "flow_step": flow_step,
        "t": t,
        "projection_target": target,
        "velocity_norm": velocity_norm,
        "endpoint_total_violation": endpoint_snapshot["total_violation"],
        "endpoint_min_clearance": endpoint_snapshot["min_clearance"],
        "endpoint_goal_error": endpoint_snapshot["goal_error"],
        "projection_called": float(diagnostics is not None),
    }
    if diagnostics is None:
        row.update(
            {
                "solver_success": float("nan"),
                "solver_runtime_sec": 0.0,
                "projection_total_runtime_sec": 0.0,
                "projection_correction_norm": 0.0,
                "projection_objective": 0.0,
                "min_clearance_before_projection": float("nan"),
                "min_clearance_after_projection": float("nan"),
                "violation_before_projection": float("nan"),
                "violation_after_projection": float("nan"),
                "projection_error": "",
                "solver_iterations": 0,
                "corridor_collision_count": 0,
            }
        )
    else:
        row.update(
            {
                "solver_success": float(diagnostics.solver_success),
                "solver_runtime_sec": diagnostics.solver_runtime_sec,
                "projection_total_runtime_sec": diagnostics.total_runtime_sec,
                "projection_correction_norm": diagnostics.correction_norm,
                "projection_objective": diagnostics.objective_value,
                "min_clearance_before_projection": diagnostics.min_clearance_before,
                "min_clearance_after_projection": diagnostics.min_clearance_after,
                "violation_before_projection": diagnostics.total_violation_before,
                "violation_after_projection": diagnostics.total_violation_after,
                "projection_error": diagnostics.error,
                "solver_iterations": diagnostics.solver_iterations,
                "corridor_collision_count": diagnostics.corridor_collision_count,
            }
        )
    return row


def run_sampler(
    method: str,
    initial_noise: torch.Tensor,
    condition: torch.Tensor,
    field: FlowMatchingField,
    projector: TrajectoryProjector,
    robot_model: RobotModel,
    obstacle_manager: ObstacleManager,
    q_start: np.ndarray,
    p_goal: np.ndarray,
    task_id: int,
    seed: int,
    flow_steps: int,
) -> tuple[np.ndarray, dict[str, float], list[dict[str, Any]]]:
    x = initial_noise.clone()
    dt = 1.0 / flow_steps
    network_latency = 0.0
    projection_latency = 0.0
    solver_latency = 0.0
    projection_calls = 0
    failed_projections = 0
    correction_sum = 0.0
    final_correction = 0.0
    step_rows: list[dict[str, Any]] = []
    synchronize(field.device)
    planning_started = time.perf_counter()

    for flow_step in range(flow_steps):
        t = flow_step / flow_steps
        current_diag = None
        if method == "CURRENT_STATE_PROJECTION":
            projected, current_diag = projector.project_trajectory(
                x.detach().cpu().numpy()[0], q_start, p_goal, obstacle_manager
            )
            x = torch.as_tensor(projected[None, ...], dtype=torch.float32, device=x.device)
            projection_calls += 1
            failed_projections += int(not current_diag.solver_success)
            correction_sum += current_diag.correction_norm
            projection_latency += current_diag.total_runtime_sec
            solver_latency += current_diag.solver_runtime_sec

        velocity, elapsed = model_velocity(field, x, t, condition)
        network_latency += elapsed
        endpoint_pred = x + (1.0 - t) * velocity
        endpoint_np = endpoint_pred.detach().cpu().numpy()[0]
        endpoint_state = constraint_snapshot(
            endpoint_np, p_goal, robot_model, obstacle_manager
        )

        pov_diag = None
        if method == "POV_ENDPOINT_PROJECTION":
            endpoint_projected, pov_diag = projector.project_trajectory(
                endpoint_np, q_start, p_goal, obstacle_manager
            )
            endpoint_projected_t = torch.as_tensor(
                endpoint_projected[None, ...], dtype=torch.float32, device=x.device
            )
            # Primary POV rule: do not divide by (1 - t).
            x = primary_pov_euler_step(x, endpoint_projected_t, dt)
            projection_calls += 1
            failed_projections += int(not pov_diag.solver_success)
            correction_sum += pov_diag.correction_norm
            final_correction = pov_diag.correction_norm
            projection_latency += pov_diag.total_runtime_sec
            solver_latency += pov_diag.solver_runtime_sec
        else:
            x = x + dt * velocity

        active_diag = pov_diag if pov_diag is not None else current_diag
        target = (
            "predicted_endpoint"
            if pov_diag is not None
            else "current_state"
            if current_diag is not None
            else "none"
        )
        step_rows.append(
            projection_row(
                method,
                task_id,
                seed,
                flow_step,
                t,
                float(torch.linalg.vector_norm(velocity).item()),
                endpoint_state,
                active_diag,
                target,
            )
        )

    if method == "FINAL_PROJECTION":
        candidate = x.detach().cpu().numpy()[0]
        projected, diag = projector.project_trajectory(
            candidate, q_start, p_goal, obstacle_manager
        )
        x = torch.as_tensor(projected[None, ...], dtype=torch.float32, device=x.device)
        projection_calls = 1
        failed_projections = int(not diag.solver_success)
        correction_sum = diag.correction_norm
        final_correction = diag.correction_norm
        projection_latency = diag.total_runtime_sec
        solver_latency = diag.solver_runtime_sec
        endpoint_state = constraint_snapshot(
            candidate, p_goal, robot_model, obstacle_manager
        )
        step_rows.append(
            projection_row(
                method,
                task_id,
                seed,
                flow_steps,
                1.0,
                0.0,
                endpoint_state,
                diag,
                "final_endpoint",
            )
        )

    synchronize(field.device)
    planning_latency = time.perf_counter() - planning_started
    trajectory = x.detach().cpu().numpy()[0]
    sampler_metrics = {
        "planning_latency_sec": planning_latency,
        "fm_latency_sec": network_latency,
        "projection_latency_sec": projection_latency,
        "projection_solver_latency_sec": solver_latency,
        "projection_calls": float(projection_calls),
        "failed_projection_solves": float(failed_projections),
        "projection_correction_sum": correction_sum,
        "final_correction_magnitude": final_correction,
    }
    return trajectory, sampler_metrics, step_rows


def bootstrap_summary(
    raw: pd.DataFrame, bootstrap_samples: int
) -> pd.DataFrame:
    rng = np.random.default_rng(20260907)
    metrics = [
        "collision",
        "success",
        "min_clearance",
        "joint_limit_violation",
        "velocity_violation",
        "acceleration_violation",
        "terminal_goal_error",
        "joint_path_length",
        "ee_path_length",
        "smoothness",
        "acceleration_cost",
        "jerk_cost",
        "reference_distance",
        "trajectory_distortion_vs_plain",
        "planning_latency_sec",
        "fm_latency_sec",
        "projection_latency_sec",
        "projection_solver_latency_sec",
        "projection_calls",
        "failed_projection_solves",
        "projection_correction_sum",
        "final_correction_magnitude",
    ]
    rows = []
    for method, group in raw.groupby("method", sort=False):
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            samples = rng.choice(values, size=(bootstrap_samples, len(values)), replace=True)
            boot_means = samples.mean(axis=1)
            row = {
                "method": method,
                "metric": metric,
                "count": len(values),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "median": float(np.median(values)),
                "ci95_low": float(np.quantile(boot_means, 0.025)),
                "ci95_high": float(np.quantile(boot_means, 0.975)),
                "numerator": "",
                "denominator": "",
            }
            if metric in {"collision", "success"}:
                row["numerator"] = int(values.sum())
                row["denominator"] = len(values)
            rows.append(row)
    return pd.DataFrame(rows)


def plot_bars(summary: pd.DataFrame, output_dir: Path) -> None:
    plot_specs = {
        "success_rate.png": ("success", "Success rate", True),
        "collision_rate.png": ("collision", "Collision rate", True),
        "goal_error.png": ("terminal_goal_error", "Terminal EE goal error (m)", False),
        "min_clearance.png": ("min_clearance", "Minimum clearance (m)", False),
        "latency.png": ("planning_latency_sec", "Planning latency (s)", False),
        "projection_calls.png": ("projection_calls", "Projection calls", False),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, (metric, ylabel, bounded) in plot_specs.items():
        frame = summary[summary.metric == metric].set_index("method").reindex(METHODS)
        means = frame["mean"].to_numpy()
        errors = np.vstack((means - frame["ci95_low"], frame["ci95_high"] - means))
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(range(len(METHODS)), means, yerr=errors, capsize=4)
        ax.set_xticks(range(len(METHODS)), [x.replace("_", "\n") for x in METHODS])
        ax.set_ylabel(ylabel)
        if bounded:
            ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)


def plot_pov_steps(steps: pd.DataFrame, output_dir: Path) -> None:
    pov = steps[
        (steps.method == "POV_ENDPOINT_PROJECTION")
        & (steps.projection_called == 1.0)
    ]
    grouped = pov.groupby("t")["projection_correction_norm"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"], marker="o")
    ax.set_xlabel("Flow time t")
    ax.set_ylabel("Endpoint projection correction norm")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "correction_vs_flow_time.png", dpi=160)
    plt.close(fig)


def plot_pareto(summary: pd.DataFrame, output_dir: Path) -> None:
    collision = summary[summary.metric == "collision"].set_index("method")
    latency = summary[summary.metric == "planning_latency_sec"].set_index("method")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in METHODS:
        ax.scatter(latency.loc[method, "mean"], collision.loc[method, "mean"], s=60)
        ax.annotate(method, (latency.loc[method, "mean"], collision.loc[method, "mean"]))
    ax.set_xlabel("Mean planning latency (s)")
    ax.set_ylabel("Collision rate")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "safety_latency_pareto.png", dpi=160)
    plt.close(fig)


def select_cases(raw: pd.DataFrame) -> list[tuple[str, int, int]]:
    pivot_success = raw.pivot_table(
        index=["task_id", "seed"], columns="method", values="success"
    )
    easy = pivot_success.mean(axis=1).sort_values(ascending=False).index[0]
    disagreements = pivot_success[pivot_success.nunique(axis=1) > 1]
    difficult = disagreements.index[0] if len(disagreements) else pivot_success.mean(axis=1).sort_values().index[0]
    pov_fail = raw[
        (raw.method == "POV_ENDPOINT_PROJECTION") & (raw.success == 0.0)
    ].sort_values(["collision", "terminal_goal_error"], ascending=False)
    failure = (
        (int(pov_fail.iloc[0].task_id), int(pov_fail.iloc[0].seed))
        if len(pov_fail)
        else pivot_success.mean(axis=1).sort_values().index[0]
    )
    return [
        ("easy", int(easy[0]), int(easy[1])),
        ("difficult", int(difficult[0]), int(difficult[1])),
        ("failure_or_disagreement", int(failure[0]), int(failure[1])),
    ]


def plot_cases(
    raw: pd.DataFrame,
    trajectories: dict[str, np.ndarray],
    robot_model: RobotModel,
    obstacle_manager: ObstacleManager,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, bboxes = obstacle_manager.get_obstacles()
    for label, task_id, seed in select_cases(raw):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for method in METHODS:
            key = f"task{task_id}_seed{seed}_{method}"
            q = trajectories[key]
            ee = np.asarray([robot_model.fk_pos(x) for x in q])
            axes[0].plot(ee[:, 0], ee[:, 1], marker=".", label=method)
            axes[1].plot(ee[:, 0], ee[:, 2], marker=".", label=method)
        for bbox in bboxes:
            lower, upper = bbox[:3], bbox[3:]
            axes[0].add_patch(
                plt.Rectangle(
                    (lower[0], lower[1]), upper[0] - lower[0], upper[1] - lower[1],
                    color="black", alpha=0.18,
                )
            )
            axes[1].add_patch(
                plt.Rectangle(
                    (lower[0], lower[2]), upper[0] - lower[0], upper[2] - lower[2],
                    color="black", alpha=0.18,
                )
            )
        axes[0].set(xlabel="x (m)", ylabel="y (m)", title="Top view")
        axes[1].set(xlabel="x (m)", ylabel="z (m)", title="Side view")
        for ax in axes:
            ax.axis("equal")
            ax.grid(alpha=0.2)
        axes[1].legend(fontsize=7)
        fig.suptitle(f"{label}: task={task_id}, seed={seed}")
        fig.tight_layout()
        fig.savefig(output_dir / f"{label}.png", dpi=160)
        plt.close(fig)


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate extra."""

    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        cells = []
        for value in values:
            if isinstance(value, float):
                cells.append(f"{value:.6g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def environment_record() -> dict[str, str]:
    acados_root = Path(os.environ.get("ACADOS_SOURCE_DIR", ""))
    cpu_model = "unavailable"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "python": sys.version.replace("\n", " "),
        "pytorch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "gpu": command_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
        ),
        "cpu": cpu_model,
        "platform": platform.platform(),
        "safeflowmpc_commit": command_output(["git", "rev-parse", "HEAD"]),
        "acados_commit": command_output(
            ["git", "-C", str(acados_root), "rev-parse", "HEAD"]
        )
        if acados_root.is_dir()
        else "unavailable",
        "acados_version": command_output(
            ["git", "-C", str(acados_root), "describe", "--tags", "--always"]
        )
        if acados_root.is_dir()
        else "unavailable",
    }


def write_report(raw: pd.DataFrame, summary: pd.DataFrame, output_root: Path) -> None:
    means = summary.pivot(index="metric", columns="method", values="mean")
    pov = "POV_ENDPOINT_PROJECTION"
    plain = "PLAIN_FM"
    projections = ["FINAL_PROJECTION", "CURRENT_STATE_PROJECTION"]
    lower_collision = means.loc["collision", pov] < means.loc["collision", plain]
    similar_projection_safety = means.loc["collision", pov] <= min(
        means.loc["collision", x] for x in projections
    ) + 0.02
    lower_distortion = means.loc["trajectory_distortion_vs_plain", pov] <= min(
        means.loc["trajectory_distortion_vs_plain", x] for x in projections
    )
    acceptable_latency = means.loc["planning_latency_sec", pov] <= 1.0
    promising = lower_collision and similar_projection_safety and lower_distortion and acceptable_latency
    verdict = "GO" if promising else "NO-GO"

    table_metrics = [
        "collision", "success", "terminal_goal_error", "min_clearance",
        "trajectory_distortion_vs_plain", "planning_latency_sec",
        "projection_solver_latency_sec", "failed_projection_solves",
    ]
    table = summary[summary.metric.isin(table_metrics)][
        ["method", "metric", "mean", "std", "median", "ci95_low", "ci95_high", "numerator", "denominator"]
    ]
    failures = int(raw.failed_projection_solves.sum())
    table_markdown = dataframe_to_markdown(table)
    report = f"""# POV endpoint-projection Phase-1 report

## Decision

**{verdict}** under the predeclared first gate.

- Lower collision rate than plain FM: `{lower_collision}`
- Within 2 percentage points of the safer projection baseline: `{similar_projection_safety}`
- No more distortion than both projection baselines: `{lower_distortion}`
- Mean one-shot planning latency at most 1 second: `{acceptable_latency}`
- Projection solver failures (not hidden): `{failures}`

## Scope

This is the controlled one-shot ID experiment: six supplied tasks, matched seeds,
one 16-knot sample per task/seed, the pretrained unsafe Flow Matching checkpoint,
and no retraining. It is not a full closed-loop MPC success-rate reproduction.

The projection reuses SafeFlowMPC's Acados OCP. The objective is weighted squared
joint-trajectory distance plus the upstream small smoothness regularizer. Collision
constraints use a candidate-dependent local convex corridor and soft nonlinear
constraints; consequently this is a local weighted projection surrogate rather
than an exact global Euclidean projection. The upstream terminal option constrains
terminal derivatives, not a hard Cartesian goal equality.
When all 12 RTI relinearizations fail, the last finite solver iterate is retained
as that method's planning output and the solve remains marked failed; it is never
counted as a successful projection.

## Summary

{table_markdown}

## Primary POV update

At each original Euler flow time `t=k/7`, the implementation evaluates the raw
velocity `v_theta`, forms `x1_pred = x_t + (1-t) v_theta`, projects that endpoint,
and advances with `x_t <- x_t + (1/7) * (P(x1_pred)-x_t)`. It deliberately does
not divide by `1-t`.

## Reproduction and commands

The upstream `inference_global_planner.py` was reproduced before experiment code
was added, using Xvfb for the unchanged MuJoCo viewer and Acados v0.5.1. The run
reached its sampled goal and exited with status 0; its log is preserved separately.

```bash
export ACADOS_SOURCE_DIR=/workspace/acados-v0.5.1
export LD_LIBRARY_PATH=/workspace/Y-Flow/experiments/exp_04_robot_arm/.venv/lib/python3.12/site-packages/cmeel.prefix/lib:/workspace/acados-v0.5.1/lib:$LD_LIBRARY_PATH
python -m experiments.pov_projection.run_phase1 --seeds 32
```

## Limitations

- The six examples are ID tasks and share the repository's default obstacles.
- Candidate feasibility metrics use true obstacle boxes and collision-sphere radii,
  while the optimizer uses the upstream local convex corridor approximation.
- Finite-difference velocity and acceleration diagnostics are conservative proxies
  for one-shot trajectories.
- Per the gate, no safety-model run, OOD scenarios, adaptive trigger, normalized
  POV, or retraining was performed before this verdict.
"""
    (output_root / "REPORT.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--tasks", type=int, default=6)
    parser.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help="Comma-separated task IDs for an independent benchmark shard",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--output", type=Path, default=RUN_ROOT / "pov_projection" / "phase1"
    )
    parser.add_argument("--build-solver", action="store_true")
    args = parser.parse_args()
    task_ids = (
        list(range(args.tasks))
        if args.task_ids is None
        else [int(value) for value in args.task_ids.split(",")]
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Phase-1 benchmark requires cuda:0")
    device = "cuda:0"
    torch.set_grad_enabled(False)

    config = PlannerConfig(
        use_safety_filter=False,
        use_safe_model=False,
        use_safe_dist=False,
        use_term=True,
        use_sets=True,
        use_guidance=False,
        n_horizon=16,
        flow_steps=7,
        compile_fm=False,
        fm_dim=32,
        fm_dim_mults=(1, 2, 4, 8),
        real_time=False,
        experiment=False,
        limit_time=False,
        sleep=False,
        model_path="checkpoints/",
    )
    robot_model = RobotModel()
    obstacle_manager = ObstacleManager()
    obstacle_manager.add_default_obstacles()
    field = FlowMatchingField(config, device)
    projector = TrajectoryProjector(
        obstacle_manager, robot_model, horizon=16, build=args.build_solver
    )

    # Warm up both GPU and optimizer before timing.
    warm_task = load_task(0)
    q_start = warm_task["q"][0]
    q_goal = warm_task["q"][-1]
    p_goal, _, _ = robot_model.forward_kinematics(q_goal, np.zeros(7))
    condition = build_condition(robot_model, q_start, warm_task["q_prev0"], p_goal, device)
    for _ in range(10):
        noise = torch.randn((1, 16, 7), device=device)
        model_velocity(field, noise, 0.0, condition)
    warm_candidate = warm_task["q"][:16]
    for _ in range(3):
        projector.project_trajectory(warm_candidate, q_start, p_goal, obstacle_manager)

    raw_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    trajectories: dict[str, np.ndarray] = {}
    total = len(task_ids) * args.seeds
    completed = 0
    for task_id in task_ids:
        task = load_task(task_id)
        q_start = task["q"][0]
        q_goal = task["q"][-1]
        q_reference = task["q"][:16]
        p_goal, _, _ = robot_model.forward_kinematics(q_goal, np.zeros(7))
        condition = build_condition(
            robot_model, q_start, task["q_prev0"], p_goal, device
        )
        for seed in range(args.seeds):
            generator = torch.Generator(device=device).manual_seed(seed)
            initial_noise = torch.randn(
                (1, 16, 7), generator=generator, device=device
            )
            matched: dict[str, np.ndarray] = {}
            method_rows: dict[str, dict[str, Any]] = {}
            for method in METHODS:
                trajectory, timing, steps = run_sampler(
                    method,
                    initial_noise,
                    condition,
                    field,
                    projector,
                    robot_model,
                    obstacle_manager,
                    q_start,
                    p_goal,
                    task_id,
                    seed,
                    config.flow_steps,
                )
                metrics = trajectory_metrics(
                    trajectory,
                    q_goal,
                    p_goal,
                    q_reference,
                    robot_model,
                    obstacle_manager,
                )
                row = {"method": method, "task_id": task_id, "seed": seed}
                row.update(metrics)
                row.update(timing)
                method_rows[method] = row
                matched[method] = trajectory
                trajectories[f"task{task_id}_seed{seed}_{method}"] = trajectory
                step_rows.extend(steps)
            plain = matched["PLAIN_FM"]
            for method in METHODS:
                method_rows[method]["trajectory_distortion_vs_plain"] = float(
                    np.linalg.norm(matched[method] - plain)
                )
                raw_rows.append(method_rows[method])
            completed += 1
            print(f"completed {completed}/{total}: task={task_id} seed={seed}", flush=True)

    results_dir = args.output / "results"
    plots_dir = args.output / "plots"
    visualizations_dir = args.output / "visualizations"
    results_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(raw_rows)
    steps = pd.DataFrame(step_rows)
    summary = bootstrap_summary(raw, args.bootstrap_samples)
    raw.to_csv(results_dir / "raw_results.csv", index=False)
    steps.to_csv(results_dir / "per_step_pov.csv", index=False)
    summary.to_csv(results_dir / "summary.csv", index=False)
    np.savez_compressed(results_dir / "trajectories.npz", **trajectories)
    config_record = {
        "experiment": "phase1_id_one_shot",
        "tasks": task_ids,
        "seeds_per_task": args.seeds,
        "methods": METHODS,
        "checkpoint": config.model_name,
        "horizon": config.n_horizon,
        "dof": 7,
        "flow_steps": config.flow_steps,
        "fm_dim": config.fm_dim,
        "fm_dim_mults": config.fm_dim_mults,
        "flow_times": [k / config.flow_steps for k in range(config.flow_steps)],
        "pov_rule": "v_pov = P(x_t + (1-t)*v_theta) - x_t",
        "projection": "upstream Acados weighted local projection surrogate",
        "bootstrap_samples": args.bootstrap_samples,
        "environment": environment_record(),
    }
    (results_dir / "config.json").write_text(json.dumps(config_record, indent=2))
    plot_bars(summary, plots_dir)
    plot_pov_steps(steps, plots_dir)
    plot_pareto(summary, plots_dir)
    plot_cases(raw, trajectories, robot_model, obstacle_manager, visualizations_dir)
    write_report(raw, summary, args.output)
    print(f"artifacts written to {args.output}")


if __name__ == "__main__":
    main()
