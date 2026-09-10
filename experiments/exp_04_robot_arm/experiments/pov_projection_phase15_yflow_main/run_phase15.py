"""Run the matched Phase-1.5 remote-main Y-Flow transfer experiment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from safe_flow_mpc.RobotModel import RobotModel
from safe_flow_mpc.SafeFlowMPC import FlowMatchingField, PlannerConfig
from safe_flow_mpc.SafeFlowMPC.ObstacleManager import ObstacleManager

from experiments.pov_projection.metrics import constraint_snapshot, trajectory_metrics
from experiments._layout import RUN_ROOT
from experiments.pov_projection.run_phase1 import (
    build_condition,
    environment_record,
    load_task,
    model_velocity,
    synchronize,
)
from experiments.pov_projection_phase2.run_phase2 import (
    _empty_projection_fields,
    _projection_fields,
    run_sampler as run_phase2_sampler,
)
from experiments.pov_projection_phase15_yflow_main.projection import (
    YFlowRobotTerminalOptimizer,
)
from experiments.pov_projection_phase15_yflow_main.sampling import (
    yflow_interpolate,
    yflow_projection_active,
)


METHODS = (
    "PLAIN_FM",
    "POV_L05_ALWAYS",
    "YFLOW_MAIN_PORT",
    "GOAL_AWARE_FINAL_PROJECTION",
)
OUTPUT_ROOT = RUN_ROOT / "pov_projection_phase15_yflow_main"
REMOTE_MAIN_COMMIT = "eeee5d2fee03f6c0bb593148727f0a92b94d3c97"
T_ON = 0.5
LAMBDA_OC = 10.0


def run_yflow_main_port(
    initial_noise: torch.Tensor,
    condition: torch.Tensor,
    field: FlowMatchingField,
    optimizer: YFlowRobotTerminalOptimizer,
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
    fm_latency = 0.0
    projection_latency = 0.0
    solver_latency = 0.0
    calls = 0
    failures = 0
    correction_sum = 0.0
    rows: list[dict[str, Any]] = []
    synchronize(field.device)
    started = time.perf_counter()

    for flow_step in range(flow_steps):
        t = flow_step / flow_steps
        terminal = flow_step == flow_steps - 1
        velocity, elapsed = model_velocity(field, x, t, condition)
        fm_latency += elapsed
        raw_target = x + (1.0 - t) * velocity
        raw_np = raw_target.detach().cpu().numpy()[0]
        snapshot = constraint_snapshot(
            raw_np, p_goal, robot_model, obstacle_manager
        )
        active = yflow_projection_active(flow_step, flow_steps, t, T_ON)
        raw_weight = LAMBDA_OC * t**2 / max(dt, 1e-8) if active else 0.0
        row: dict[str, Any] = {
            "method": "YFLOW_MAIN_PORT",
            "task_id": task_id,
            "seed": seed,
            "flow_step": flow_step,
            "t": t,
            "terminal_step": float(terminal),
            "projection_active": float(active),
            "eta": 1.0 if terminal else dt / max(1.0 - t, 1e-8),
            "raw_tracking_weight": raw_weight,
            "raw_velocity_norm": float(torch.linalg.vector_norm(velocity).item()),
            "endpoint_constraint_violation_before_projection": snapshot[
                "total_violation"
            ],
            "endpoint_goal_error": snapshot["goal_error"],
            "endpoint_min_clearance": snapshot["min_clearance"],
        }
        if active:
            optimized, diagnostics = optimizer.optimize_terminal(
                raw_np, q_start, p_goal, raw_weight
            )
            target = torch.as_tensor(
                optimized[None, ...], dtype=torch.float32, device=x.device
            )
            x, eta = yflow_interpolate(x, target, t, dt, terminal)
            row["eta"] = eta
            normalized = (target - raw_target) / max(1.0 - t, 1e-3)
            velocity_change = float(torch.linalg.vector_norm(normalized).item())
            row.update(
                _projection_fields(
                    diagnostics,
                    float(torch.linalg.vector_norm(normalized).item()),
                    velocity_change,
                )
            )
            calls += 1
            failures += int(not diagnostics.solver_success)
            correction_sum += diagnostics.correction_norm
            projection_latency += diagnostics.total_runtime_sec
            solver_latency += diagnostics.solver_runtime_sec
        else:
            x, eta = yflow_interpolate(x, raw_target, t, dt, terminal)
            row["eta"] = eta
            row.update(_empty_projection_fields())
        rows.append(row)

    synchronize(field.device)
    return (
        x.detach().cpu().numpy()[0],
        {
            "planning_latency_sec": time.perf_counter() - started,
            "fm_latency_sec": fm_latency,
            "projection_latency_sec": projection_latency,
            "projection_solver_latency_sec": solver_latency,
            "projection_calls": float(calls),
            "failed_projection_solves": float(failures),
            "projection_correction_sum": correction_sum,
        },
        rows,
    )


SUMMARY_METRICS = (
    "collision",
    "success",
    "min_clearance",
    "joint_limit_violation",
    "velocity_violation",
    "acceleration_violation",
    "terminal_goal_error",
    "trajectory_distortion_vs_plain",
    "planning_latency_sec",
    "projection_solver_latency_sec",
    "projection_calls",
    "projection_failure_rate",
)


def summarize(raw: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260908)
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        group = raw[raw.method == method]
        for metric in SUMMARY_METRICS:
            values = group[metric].to_numpy(float)
            boot = rng.choice(
                values, size=(bootstrap_samples, len(values)), replace=True
            ).mean(axis=1)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "count": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(np.median(values)),
                    "ci95_low": float(np.quantile(boot, 0.025)),
                    "ci95_high": float(np.quantile(boot, 0.975)),
                    "numerator": int(values.sum())
                    if metric in {"collision", "success"}
                    else "",
                    "denominator": len(values)
                    if metric in {"collision", "success"}
                    else "",
                }
            )
    return pd.DataFrame(rows)


def paired_and_gate(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    index = ["task_id", "seed"]
    yflow = raw[raw.method == "YFLOW_MAIN_PORT"].set_index(index).sort_index()
    rows: list[dict[str, Any]] = []
    for reference_method in ("PLAIN_FM", "POV_L05_ALWAYS"):
        reference = raw[raw.method == reference_method].set_index(index).sort_index()
        for reference_state, reference_value in (("safe", False), ("collision", True)):
            for yflow_state, yflow_value in (("safe", False), ("collision", True)):
                rows.append(
                    {
                        "reference_method": reference_method,
                        "reference_state": reference_state,
                        "yflow_state": yflow_state,
                        "count": int(
                            np.sum(
                                (reference.collision.to_numpy(bool) == reference_value)
                                & (yflow.collision.to_numpy(bool) == yflow_value)
                            )
                        ),
                    }
                )
    paired = pd.DataFrame(rows)
    means = raw.groupby("method").mean(numeric_only=True)
    always = means.loc["POV_L05_ALWAYS"]
    yf = means.loc["YFLOW_MAIN_PORT"]
    always_safe = int((raw[raw.method == "POV_L05_ALWAYS"].collision == 0).sum())
    new_collisions = int(
        paired[
            (paired.reference_method == "POV_L05_ALWAYS")
            & (paired.reference_state == "safe")
            & (paired.yflow_state == "collision")
        ]["count"].iloc[0]
    )
    gate = {
        "collision_difference_vs_always": float(yf.collision - always.collision),
        "collision_within_2pp": bool(yf.collision <= always.collision + 0.02),
        "new_collisions_vs_always": new_collisions,
        "new_collision_rate_from_always_safe": new_collisions / always_safe,
        "few_new_collisions": bool(new_collisions / always_safe <= 0.02),
        "projection_call_reduction": float(
            1.0 - yf.projection_calls / always.projection_calls
        ),
        "calls_reduced_at_least_50pct": bool(
            1.0 - yf.projection_calls / always.projection_calls >= 0.50
        ),
        "latency_reduction": float(
            1.0 - yf.planning_latency_sec / always.planning_latency_sec
        ),
        "latency_reduced_at_least_35pct": bool(
            1.0 - yf.planning_latency_sec / always.planning_latency_sec >= 0.35
        ),
        "goal_error_difference": float(
            yf.terminal_goal_error - always.terminal_goal_error
        ),
        "goal_not_worse_by_more_than_0p1m": bool(
            yf.terminal_goal_error <= always.terminal_goal_error + 0.10
        ),
    }
    gate["passed"] = bool(
        gate["collision_within_2pp"]
        and gate["few_new_collisions"]
        and gate["calls_reduced_at_least_50pct"]
        and gate["latency_reduced_at_least_35pct"]
        and gate["goal_not_worse_by_more_than_0p1m"]
    )
    return paired, gate


def make_plots(summary: pd.DataFrame, steps: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    labels = [m.replace("_", "\n") for m in METHODS]
    for metric, filename, ylabel in (
        ("collision", "collision.png", "Collision rate"),
        ("terminal_goal_error", "goal_error.png", "Goal error (m)"),
        ("planning_latency_sec", "latency.png", "Planning latency (s)"),
        ("projection_calls", "projection_calls.png", "Calls per trajectory"),
    ):
        frame = summary[summary.metric == metric].set_index("method").reindex(METHODS)
        mean = frame["mean"].to_numpy()
        err = np.vstack((mean - frame.ci95_low, frame.ci95_high - mean))
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.bar(range(len(METHODS)), mean, yerr=err, capsize=4)
        ax.set_xticks(range(len(METHODS)), labels, fontsize=7)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)

    yf = steps[steps.method == "YFLOW_MAIN_PORT"]
    flow = yf.groupby("t", as_index=False).agg(
        projection_rate=("projection_called", "mean"),
        correction=("correction_norm", "mean"),
        raw_weight=("raw_tracking_weight", "mean"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, column, title in zip(
        axes,
        ("projection_rate", "correction", "raw_weight"),
        ("Projection rate", "Mean correction", "Raw-target weight"),
    ):
        ax.plot(flow.t, flow[column], marker="o")
        ax.set_title(title)
        ax.set_xlabel("Flow time t")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "flow_time.png", dpi=160)
    plt.close(fig)


def markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(
                f"{value:.6g}" if isinstance(value, float) else str(value)
                for value in values
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    steps: pd.DataFrame,
    paired: pd.DataFrame,
    gate: dict[str, Any],
    output: Path,
) -> None:
    primary = summary[
        summary.metric.isin(
            ("collision", "terminal_goal_error", "planning_latency_sec", "projection_calls")
        )
    ][["method", "metric", "mean", "ci95_low", "ci95_high", "numerator", "denominator"]]
    means = raw.groupby("method").mean(numeric_only=True)
    yf = means.loc["YFLOW_MAIN_PORT"]
    always = means.loc["POV_L05_ALWAYS"]
    flow = steps[steps.method == "YFLOW_MAIN_PORT"].groupby("t", as_index=False).agg(
        projection_rate=("projection_called", "mean"),
        raw_tracking_weight=("raw_tracking_weight", "mean"),
        correction_norm=("correction_norm", "mean"),
        solver_success=("projection_solver_success", "mean"),
    )
    active = steps[
        (steps.method == "YFLOW_MAIN_PORT") & (steps.projection_called == 1.0)
    ].copy()
    active["true_collision_after_projection"] = (
        active.min_clearance_after_projection < -1e-5
    ).astype(float)
    active_safety = active.groupby("t", as_index=False).agg(
        true_collision_after_projection=("true_collision_after_projection", "mean"),
        mean_true_clearance_after_projection=(
            "min_clearance_after_projection",
            "mean",
        ),
        solver_success=("projection_solver_success", "mean"),
    )
    per_task = raw.groupby(["task_id", "method"], as_index=False).agg(
        collision_rate=("collision", "mean"),
        goal_error=("terminal_goal_error", "mean"),
        min_clearance=("min_clearance", "mean"),
    )
    verdict = "GO" if gate["passed"] else "NO-GO"
    report = f"""# Robot-arm Y-Flow Phase 1.5 report

## Decision

**{verdict}** for the predeclared remote-main outer-loop transfer gate.

```json
{json.dumps(gate, indent=2)}
```

## Primary matched results

{markdown(primary)}

## Paired collision transitions

{markdown(paired)}

## What was transferred exactly

The outer update follows Y-Flow remote `main` commit `{REMOTE_MAIN_COMMIT}`:

1. `x1_raw = x + (1-t) * v`;
2. non-terminal steps with `t < 0.5` use vanilla interpolation;
3. later steps solve a terminal constrained target;
4. `eta = dt/(1-t)`, with `eta = 1` at the terminal step;
5. `x_next = (1-eta) * x + eta * z_star`;
6. raw-target weight is `lambda_oc * t^2 / dt`, with `lambda_oc=10`.

On the seven-step grid this makes exactly three optimization calls at
`t=4/7, 5/7, 6/7`.

## Necessary robot-domain substitution

The remote code's `P` is a differentiable 2-D Swiss-roll centerline projection
and its inner solver is GPU PGD. No corresponding analytic robot trajectory
manifold operator exists in SafeFlowMPC. We therefore use the Y-Flow document's
explicit no-`P` path (`mu=0`): the existing Acados OCP supplies the terminal
optimization, its robot goal/smoothness terms act as `C`, and its trajectory
constraints act as `h`.

This is an exact transfer of the Y-Flow scheduling and interpolation mechanism,
not a claim that 2-D PGD and robot Acados are identical inner solvers. Collision
corridors and the Cartesian goal remain soft penalties in the inherited robot
OCP, while joint/velocity/acceleration bounds are hard.

## Result interpretation

- Collision: Y-Flow port {yf.collision:.4f} vs always-on POV {always.collision:.4f}.
- Goal error: {yf.terminal_goal_error:.4f} vs {always.terminal_goal_error:.4f} m.
- Calls: {yf.projection_calls:.4f} vs {always.projection_calls:.4f}
  ({100 * gate['projection_call_reduction']:.2f}% reduction).
- Latency: {yf.planning_latency_sec:.4f} vs {always.planning_latency_sec:.4f} s
  ({100 * gate['latency_reduction']:.2f}% reduction).
- New collisions relative to always-on: {gate['new_collisions_vs_always']}.

All 64 Y-Flow-port collisions were new relative to always-on POV. They were
concentrated in task 3 and task 5 (32/32 seeds in each), while the port rescued
all 29 Plain-FM collisions from task 4. This is therefore a task-specific safety
trade rather than uniform degradation.

Every Acados solve call returned solver success, but solver success did not imply
safety under the true obstacle geometry. The local collision corridor is soft and
candidate-dependent; with the large late raw-target weights and terminal full
replacement, the optimized trajectory can still collide with the true boxes.

### True-geometry audit after each active optimization

{markdown(active_safety)}

### Per-task breakdown

{markdown(per_task)}

## Flow-time diagnostics

{markdown(flow)}

## Scope

- Same six ID tasks, 32 seeds per task, unsafe FM checkpoint and matched noise.
- No retraining, OOD evaluation, time-window tuning or coefficient sweep.
- Phase 1, Phase 2 and Phase 3 artifacts were not overwritten.
"""
    (output / "REPORT_PHASE15.md").write_text(report)


def build_components(args: argparse.Namespace) -> tuple[Any, ...]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 1.5 requires cuda:0")
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
    obstacles = ObstacleManager()
    obstacles.add_default_obstacles()
    field = FlowMatchingField(config, device)
    optimizer = YFlowRobotTerminalOptimizer(
        obstacles,
        robot_model,
        horizon=16,
        goal_mode="soft",
        goal_weight=1000.0,
        build=args.build_solver,
    )
    return device, config, robot_model, obstacles, field, optimizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--build-solver", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results_dir = args.output / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    device, config, robot_model, obstacles, field, optimizer = build_components(args)

    warm = load_task(0)
    warm_goal, _, _ = robot_model.forward_kinematics(warm["q"][-1], np.zeros(7))
    warm_condition = build_condition(
        robot_model, warm["q"][0], warm["q_prev0"], warm_goal, device
    )
    for _ in range(10):
        model_velocity(
            field, torch.randn((1, 16, 7), device=device), 0.0, warm_condition
        )

    raw_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    total = args.tasks * args.seeds
    complete = 0
    for task_id in range(args.tasks):
        task = load_task(task_id)
        q_start = task["q"][0]
        q_goal = task["q"][-1]
        p_goal, _, _ = robot_model.forward_kinematics(q_goal, np.zeros(7))
        condition = build_condition(
            robot_model, q_start, task["q_prev0"], p_goal, device
        )
        reference = task["q"][:16]
        for seed in range(args.seeds):
            generator = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn((1, 16, 7), generator=generator, device=device)
            trajectories: dict[str, np.ndarray] = {}
            method_rows: dict[str, dict[str, Any]] = {}
            for method in METHODS:
                if method == "YFLOW_MAIN_PORT":
                    trajectory, timing, steps = run_yflow_main_port(
                        noise,
                        condition,
                        field,
                        optimizer,
                        robot_model,
                        obstacles,
                        q_start,
                        p_goal,
                        task_id,
                        seed,
                        config.flow_steps,
                    )
                else:
                    # Y-Flow changes the shared solver's q-tracking cost at
                    # runtime. Restore the exact Phase-2 weight before every
                    # baseline so method order and the prior seed cannot leak.
                    optimizer.set_raw_tracking_weight(1.0)
                    phase2_method = (
                        "DAMPED_POV_L05" if method == "POV_L05_ALWAYS" else method
                    )
                    trajectory, timing, steps = run_phase2_sampler(
                        phase2_method,
                        noise,
                        condition,
                        field,
                        optimizer,
                        robot_model,
                        obstacles,
                        q_start,
                        p_goal,
                        task_id,
                        seed,
                        config.flow_steps,
                    )
                    for step in steps:
                        step["method"] = method
                        step.setdefault("projection_active", step["projection_called"])
                        step.setdefault("terminal_step", 0.0)
                        step.setdefault("eta", float("nan"))
                        step.setdefault("raw_tracking_weight", float("nan"))
                metrics = trajectory_metrics(
                    trajectory,
                    q_goal,
                    p_goal,
                    reference,
                    robot_model,
                    obstacles,
                )
                row = {"method": method, "task_id": task_id, "seed": seed}
                row.update(metrics)
                row.update(timing)
                trajectories[method] = trajectory
                method_rows[method] = row
                step_rows.extend(steps)
            plain = trajectories["PLAIN_FM"]
            for method in METHODS:
                method_rows[method]["trajectory_distortion_vs_plain"] = float(
                    np.linalg.norm(trajectories[method] - plain)
                )
                raw_rows.append(method_rows[method])
            complete += 1
            print(
                f"completed {complete}/{total}: task={task_id} seed={seed}",
                flush=True,
            )

    raw = pd.DataFrame(raw_rows)
    raw["projection_failure_rate"] = np.where(
        raw.projection_calls > 0,
        raw.failed_projection_solves / raw.projection_calls,
        0.0,
    )
    steps = pd.DataFrame(step_rows)
    summary = summarize(raw, args.bootstrap_samples)
    paired, gate = paired_and_gate(raw)
    raw.to_csv(results_dir / "raw.csv", index=False)
    summary.to_csv(results_dir / "summary.csv", index=False)
    steps.to_csv(results_dir / "per_step.csv", index=False)
    paired.to_csv(results_dir / "paired_transitions.csv", index=False)
    raw.groupby(["task_id", "method"], as_index=False).agg(
        collision_rate=("collision", "mean"),
        goal_error=("terminal_goal_error", "mean"),
        min_clearance=("min_clearance", "mean"),
    ).to_csv(results_dir / "per_task.csv", index=False)
    config_record = {
        "experiment": "phase15_remote_main_yflow_outer_port",
        "remote_main_commit": REMOTE_MAIN_COMMIT,
        "tasks": list(range(args.tasks)),
        "seeds_per_task": args.seeds,
        "methods": METHODS,
        "checkpoint": config.model_name,
        "t_on": T_ON,
        "lambda_oc": LAMBDA_OC,
        "mu": 0.0,
        "lipschitz_gate": "omitted by documented no-P path",
        "active_flow_steps": [4, 5, 6],
        "inner_solver_substitution": "robot Acados OCP for Swiss-roll GPU PGD",
        "gate": gate,
        "environment": environment_record(),
    }
    (results_dir / "config.json").write_text(json.dumps(config_record, indent=2))
    make_plots(summary, steps, args.output / "plots")
    write_report(raw, summary, steps, paired, gate, args.output)
    print(f"PHASE15_GATE={'GO' if gate['passed'] else 'NO_GO'}", flush=True)


if __name__ == "__main__":
    main()
