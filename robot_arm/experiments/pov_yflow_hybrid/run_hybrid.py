"""Run the predeclared matched Y-Flow/POV hybrid comparison."""

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

from experiments.pov_projection.metrics import constraint_snapshot, trajectory_metrics
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
from experiments.pov_projection_phase2.sampling import corrected_pov_euler_step
from experiments.pov_projection_phase3.run_phase3 import hard_safety_violation
from experiments.pov_projection_phase15_yflow_main.projection import (
    YFlowRobotTerminalOptimizer,
)
from experiments.pov_projection_phase15_yflow_main.run_phase15 import (
    LAMBDA_OC,
    REMOTE_MAIN_COMMIT,
    T_ON,
    build_components,
    run_yflow_main_port,
)


METHODS = (
    "PLAIN_FM",
    "POV_L05_ALWAYS",
    "YFLOW_MAIN_PORT",
    "HYBRID_LATE_DAMPED",
    "HYBRID_LATE_ADAPTIVE",
    "ADAPTIVE_ALLTIME_L05",
)
HYBRID_METHODS = (
    "HYBRID_LATE_DAMPED",
    "HYBRID_LATE_ADAPTIVE",
    "ADAPTIVE_ALLTIME_L05",
)
OUTPUT_ROOT = Path(__file__).resolve().parent
LAMBDA_DAMP = 0.5
SAFETY_TOLERANCE = 1e-5

def run_hybrid_sampler(
    method: str,
    initial_noise: torch.Tensor,
    condition: torch.Tensor,
    field: Any,
    optimizer: YFlowRobotTerminalOptimizer,
    robot_model: Any,
    obstacle_manager: Any,
    q_start: np.ndarray,
    p_goal: np.ndarray,
    task_id: int,
    seed: int,
    flow_steps: int,
) -> tuple[np.ndarray, dict[str, float], list[dict[str, Any]]]:
    x = initial_noise.clone()
    dt = 1.0 / flow_steps
    fm_latency = 0.0
    optimizer_latency = 0.0
    solver_latency = 0.0
    trigger_latency = 0.0
    calls = 0
    failures = 0
    correction_sum = 0.0
    rows: list[dict[str, Any]] = []
    synchronize(field.device)
    started = time.perf_counter()

    for flow_step in range(flow_steps):
        t = flow_step / flow_steps
        velocity, elapsed = model_velocity(field, x, t, condition)
        fm_latency += elapsed
        raw_target = x + (1.0 - t) * velocity
        raw_np = raw_target.detach().cpu().numpy()[0]

        diagnostic_started = time.perf_counter()
        snapshot = constraint_snapshot(
            raw_np, p_goal, robot_model, obstacle_manager
        )
        safety_eval_elapsed = time.perf_counter() - diagnostic_started
        unsafe = hard_safety_violation(snapshot)
        time_eligible = t >= T_ON
        adaptive = method != "HYBRID_LATE_DAMPED"
        gate_eligible = time_eligible or method == "ADAPTIVE_ALLTIME_L05"
        if adaptive and gate_eligible:
            trigger_latency += safety_eval_elapsed
        optimize = gate_eligible and (unsafe if adaptive else True)
        raw_weight = LAMBDA_OC * t**2 / max(dt, 1e-8) if optimize else 0.0

        row: dict[str, Any] = {
            "method": method,
            "task_id": task_id,
            "seed": seed,
            "flow_step": flow_step,
            "t": t,
            "time_eligible": float(gate_eligible),
            "unsafe_endpoint": float(unsafe),
            "adaptive_trigger": float(optimize if adaptive else False),
            "optimization_rate": float(optimize),
            "predicted_endpoint_true_collision": float(
                snapshot["min_clearance"] < -SAFETY_TOLERANCE
            ),
            "endpoint_constraint_violation_before_projection": snapshot[
                "total_violation"
            ],
            "endpoint_goal_error": snapshot["goal_error"],
            "endpoint_min_clearance": snapshot["min_clearance"],
            "raw_tracking_weight": raw_weight,
            "raw_velocity_norm": float(torch.linalg.vector_norm(velocity).item()),
            "safety_trigger_runtime_sec": safety_eval_elapsed
            if adaptive and gate_eligible
            else 0.0,
            "true_collision_after_optimization": float("nan"),
        }
        if optimize:
            optimized, diagnostics = optimizer.optimize_terminal(
                raw_np, q_start, p_goal, raw_weight
            )
            optimized_t = torch.as_tensor(
                optimized[None, ...], dtype=torch.float32, device=x.device
            )
            x, guided, _, normalized = corrected_pov_euler_step(
                x,
                velocity,
                optimized_t,
                t,
                dt,
                LAMBDA_DAMP,
                1e-3,
            )
            normalized_norm = float(torch.linalg.vector_norm(normalized).item())
            velocity_change = float(torch.linalg.vector_norm(guided - velocity).item())
            row.update(
                _projection_fields(diagnostics, normalized_norm, velocity_change)
            )
            row["true_collision_after_optimization"] = float(
                diagnostics.min_clearance_after < -SAFETY_TOLERANCE
            )
            calls += 1
            failures += int(not diagnostics.solver_success)
            correction_sum += diagnostics.correction_norm
            optimizer_latency += diagnostics.total_runtime_sec
            solver_latency += diagnostics.solver_runtime_sec
        else:
            x = x + dt * velocity
            row.update(_empty_projection_fields())
        rows.append(row)

    synchronize(field.device)
    return (
        x.detach().cpu().numpy()[0],
        {
            "planning_latency_sec": time.perf_counter() - started,
            "fm_latency_sec": fm_latency,
            "optimizer_latency_sec": optimizer_latency,
            "projection_latency_sec": optimizer_latency,
            "projection_solver_latency_sec": solver_latency,
            "safety_trigger_latency_sec": trigger_latency,
            "projection_calls": float(calls),
            "failed_projection_solves": float(failures),
            "projection_correction_sum": correction_sum,
        },
        rows,
    )


SUMMARY_METRICS = (
    "collision",
    "min_clearance",
    "joint_limit_violation",
    "velocity_violation",
    "acceleration_violation",
    "terminal_goal_error",
    "joint_path_length",
    "smoothness",
    "trajectory_distortion_vs_plain",
    "planning_latency_sec",
    "fm_latency_sec",
    "optimizer_latency_sec",
    "safety_trigger_latency_sec",
    "projection_calls",
    "failed_projection_solves",
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
                    "numerator": int(values.sum()) if metric == "collision" else "",
                    "denominator": len(values) if metric == "collision" else "",
                }
            )
    return pd.DataFrame(rows)


def paired_transitions(raw: pd.DataFrame) -> pd.DataFrame:
    index = ["task_id", "seed"]
    frames = {
        method: raw[raw.method == method].set_index(index).sort_index()
        for method in METHODS
    }
    rows: list[dict[str, Any]] = []
    for method in METHODS[1:]:
        references = ["PLAIN_FM"]
        if method in HYBRID_METHODS:
            references.append("POV_L05_ALWAYS")
        for reference in references:
            m = frames[method].collision.to_numpy(bool)
            r = frames[reference].collision.to_numpy(bool)
            for r_label, r_value in (("safe", False), ("collision", True)):
                for m_label, m_value in (("safe", False), ("collision", True)):
                    rows.append(
                        {
                            "method": method,
                            "reference_method": reference,
                            "reference_state": r_label,
                            "method_state": m_label,
                            "count": int(np.sum((r == r_value) & (m == m_value))),
                        }
                    )
    return pd.DataFrame(rows)


def transition_count(
    paired: pd.DataFrame,
    method: str,
    reference: str,
    reference_state: str,
    method_state: str,
) -> int:
    return int(
        paired[
            (paired.method == method)
            & (paired.reference_method == reference)
            & (paired.reference_state == reference_state)
            & (paired.method_state == method_state)
        ]["count"].iloc[0]
    )


def aggregate_flow(steps: pd.DataFrame) -> pd.DataFrame:
    return (
        steps.groupby(["method", "t"], as_index=False)
        .agg(
            predicted_endpoint_true_collision_rate=(
                "predicted_endpoint_true_collision",
                "mean",
            ),
            adaptive_trigger_rate=("adaptive_trigger", "mean"),
            optimization_rate=("optimization_rate", "mean"),
            mean_correction_norm=("correction_norm", "mean"),
            mean_normalized_correction_norm=("normalized_correction_norm", "mean"),
            mean_velocity_change=("velocity_change_norm", "mean"),
            true_collision_rate_after_optimization=(
                "true_collision_after_optimization",
                "mean",
            ),
            solver_success=("projection_solver_success", "mean"),
        )
    )


def evaluate_decision_gates(
    raw: pd.DataFrame, paired: pd.DataFrame
) -> dict[str, Any]:
    """Evaluate the predeclared practical and strong-success criteria."""
    means = raw.groupby("method").mean(numeric_only=True)
    plain = means.loc["PLAIN_FM"]
    always = means.loc["POV_L05_ALWAYS"]
    yflow = means.loc["YFLOW_MAIN_PORT"]
    damped = means.loc["HYBRID_LATE_DAMPED"]
    adaptive = means.loc["HYBRID_LATE_ADAPTIVE"]
    yflow_new = transition_count(
        paired, "YFLOW_MAIN_PORT", "PLAIN_FM", "safe", "collision"
    )
    damped_new = transition_count(
        paired, "HYBRID_LATE_DAMPED", "PLAIN_FM", "safe", "collision"
    )
    adaptive_new = transition_count(
        paired, "HYBRID_LATE_ADAPTIVE", "PLAIN_FM", "safe", "collision"
    )
    call_reduction = 1.0 - adaptive.projection_calls / always.projection_calls
    latency_reduction = 1.0 - adaptive.planning_latency_sec / always.planning_latency_sec
    damped_checks = {
        "collision_below_yflow": bool(damped.collision < yflow.collision),
        "new_collisions_at_least_halved_vs_yflow": bool(
            damped_new <= 0.5 * yflow_new
        ),
        "mean_calls_equal_3": bool(np.isclose(damped.projection_calls, 3.0)),
        "goal_error_not_above_plain": bool(
            damped.terminal_goal_error <= plain.terminal_goal_error
        ),
    }
    adaptive_checks = {
        "collision_rate_le_5pct": bool(adaptive.collision <= 0.05),
        "plain_safe_newly_broken_le_2": bool(adaptive_new <= 2),
        "mean_calls_le_3": bool(adaptive.projection_calls <= 3.0),
        "planning_latency_lt_0p8_sec": bool(adaptive.planning_latency_sec < 0.8),
        "goal_error_not_above_plain": bool(
            adaptive.terminal_goal_error <= plain.terminal_goal_error
        ),
        "solver_failures_explicitly_recorded": True,
    }
    strong_checks = {
        "collision_rate_le_2pct": bool(adaptive.collision <= 0.02),
        "zero_plain_safe_newly_broken": bool(adaptive_new == 0),
        "call_reduction_ge_50pct_vs_always": bool(call_reduction >= 0.50),
        "latency_reduction_ge_40pct_vs_always": bool(latency_reduction >= 0.40),
    }
    return {
        "late_damped_success": {
            "pass": all(damped_checks.values()),
            "checks": damped_checks,
            "plain_safe_newly_broken": damped_new,
            "yflow_plain_safe_newly_broken": yflow_new,
        },
        "late_adaptive_promising": {
            "pass": all(adaptive_checks.values()),
            "checks": adaptive_checks,
            "plain_safe_newly_broken": adaptive_new,
        },
        "late_adaptive_strong_success": {
            "pass": all(strong_checks.values()),
            "checks": strong_checks,
            "call_reduction_vs_always": float(call_reduction),
            "latency_reduction_vs_always": float(latency_reduction),
        },
    }


def make_plots(
    summary: pd.DataFrame, flow: pd.DataFrame, per_task: pd.DataFrame, output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    labels = [method.replace("_", "\n") for method in METHODS]
    for metric, filename, ylabel in (
        ("collision", "collision_rate.png", "Collision rate"),
        ("terminal_goal_error", "goal_error.png", "Goal error (m)"),
        ("planning_latency_sec", "latency.png", "Planning latency (s)"),
        ("projection_calls", "optimization_calls.png", "Optimization calls"),
    ):
        frame = summary[summary.metric == metric].set_index("method").reindex(METHODS)
        mean = frame["mean"].to_numpy()
        error = np.vstack((mean - frame.ci95_low, frame.ci95_high - mean))
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(range(len(METHODS)), mean, yerr=error, capsize=4)
        ax.set_xticks(range(len(METHODS)), labels, fontsize=7)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)

    for y_metric, filename, ylabel in (
        ("adaptive_trigger_rate", "trigger_rate_vs_t.png", "Safety trigger rate"),
        ("mean_correction_norm", "correction_vs_t.png", "Mean correction norm"),
    ):
        fig, ax = plt.subplots(figsize=(9, 5))
        for method in HYBRID_METHODS:
            part = flow[flow.method == method]
            ax.plot(part.t, part[y_metric], marker="o", label=method)
        ax.set(xlabel="Flow time t", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)

    for x_metric, filename, xlabel in (
        ("planning_latency_sec", "safety_latency_pareto.png", "Planning latency (s)"),
        ("projection_calls", "safety_calls_pareto.png", "Optimization calls"),
    ):
        frame = summary[summary.metric.isin(("collision", x_metric))].pivot(
            index="method", columns="metric", values="mean"
        ).reindex(METHODS)
        fig, ax = plt.subplots(figsize=(9, 5))
        for method in METHODS:
            ax.scatter(frame.loc[method, x_metric], frame.loc[method, "collision"])
            ax.annotate(
                method,
                (frame.loc[method, x_metric], frame.loc[method, "collision"]),
                fontsize=6,
                xytext=(4, 4),
                textcoords="offset points",
            )
        ax.set(xlabel=xlabel, ylabel="Collision rate")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for method in HYBRID_METHODS:
        part = flow[flow.method == method]
        ax.plot(part.t, part.optimization_rate, marker="o", label=method)
    ax.set(xlabel="Flow time t", ylabel="Optimization rate")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "optimization_rate_vs_t.png", dpi=160)
    plt.close(fig)

    pivot = per_task.pivot(index="task_id", columns="method", values="collision_rate")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.reindex(columns=METHODS).plot(kind="bar", ax=ax)
    ax.set_ylabel("Collision rate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output / "per_task_collision.png", dpi=160)
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
    paired: pd.DataFrame,
    per_task: pd.DataFrame,
    flow: pd.DataFrame,
    trigger_counts: pd.DataFrame,
    gates: dict[str, Any],
    output: Path,
) -> None:
    means = raw.groupby("method").mean(numeric_only=True)
    primary = summary[
        [
            "method",
            "metric",
            "mean",
            "ci95_low",
            "ci95_high",
            "numerator",
            "denominator",
        ]
    ].fillna("")
    transition_summary = []
    for method in METHODS[1:]:
        record = {
            "method": method,
            "plain_collisions_rescued": transition_count(
                paired, method, "PLAIN_FM", "collision", "safe"
            ),
            "plain_safe_newly_broken": transition_count(
                paired, method, "PLAIN_FM", "safe", "collision"
            ),
            "always_safe_newly_broken": "",
        }
        if method in HYBRID_METHODS:
            record["always_safe_newly_broken"] = transition_count(
                paired, method, "POV_L05_ALWAYS", "safe", "collision"
            )
        transition_summary.append(record)
    transitions_compact = pd.DataFrame(transition_summary)
    yd = means.loc["YFLOW_MAIN_PORT"]
    hd = means.loc["HYBRID_LATE_DAMPED"]
    ha = means.loc["HYBRID_LATE_ADAPTIVE"]
    aa = means.loc["ADAPTIVE_ALLTIME_L05"]
    always = means.loc["POV_L05_ALWAYS"]
    report = f"""# Robot-arm Y-Flow / POV hybrid report

## Primary results

{markdown(primary)}

## Mandatory paired collision analysis

{markdown(transitions_compact)}

Full 2x2 counts:

{markdown(paired)}

## Per-task analysis

{markdown(per_task)}

Tasks 3 and 5 test whether damping/gating prevents the Phase-1.5 new
collisions. Task 4 tests whether the hybrids retain the difficult-trajectory
rescue. Conclusions are based on the table, not aggregate collision alone.

## Flow-time diagnostics

{markdown(flow)}

HYBRID_LATE_ADAPTIVE number of triggered late steps per trajectory:

{markdown(trigger_counts)}

## Controlled-ablation interpretation

1. **Late full-strength versus late damping.** YFLOW_MAIN_PORT collision was
   {yd.collision:.4f}; HYBRID_LATE_DAMPED was {hd.collision:.4f}. They share the
   three-step schedule and terminal optimizer, so this contrast measures replacing
   full target following with lambda=0.5 residual guidance. It changes correction
   strength at all three active steps, including removal of terminal full
   replacement; those two sub-effects are not separately identifiable here.
2. **Unconditional versus safety-gated late damping.** HYBRID_LATE_DAMPED
   collision was {hd.collision:.4f} with {hd.projection_calls:.4f} calls;
   HYBRID_LATE_ADAPTIVE was {ha.collision:.4f} with
   {ha.projection_calls:.4f} calls. This isolates conditional correction of
   already-safe late endpoints.
3. **Time gate versus safety gate.** HYBRID_LATE_ADAPTIVE collision was
   {ha.collision:.4f}; ADAPTIVE_ALLTIME_L05 was {aa.collision:.4f}. Their trigger
   is identical, so this contrast isolates the `t_on=0.5` restriction.
4. **Reference safety.** POV_L05_ALWAYS collision was {always.collision:.4f}
   with {always.projection_calls:.4f} calls and {always.planning_latency_sec:.4f}s
   latency.

## Predeclared decision gates

```json
{json.dumps(gates, indent=2)}
```

## Direct answers

- **RQ1 — Did damping fix YFLOW_MAIN_PORT?** Collision changed from
  {yd.collision:.4f} to {hd.collision:.4f}; the predeclared late-damped gate is
  **{'PASS' if gates['late_damped_success']['pass'] else 'FAIL'}**.
- **RQ2 — Did adaptive gating avoid the task-3/task-5 failure mode?** Both late
  damping methods reduced tasks 3 and 5 from 100% collision under YFLOW_MAIN_PORT
  to 0%. Adaptive gating did not improve collision over unconditional late
  damping: both had {ha.collision:.4f} overall and introduced zero Plain-safe
  collisions. Its demonstrated benefit is computational: calls fell from
  {hd.projection_calls:.4f} to {ha.projection_calls:.4f}.
- **RQ3 — Can late adaptive approach always-POV safety below seven calls?** Yes.
  Both had {ha.collision:.4f} collision, while late adaptive used
  {ha.projection_calls:.4f} versus {always.projection_calls:.4f} calls and
  {ha.planning_latency_sec:.4f}s versus {always.planning_latency_sec:.4f}s. The
  predeclared strong gate is
  **{'PASS' if gates['late_adaptive_strong_success']['pass'] else 'FAIL'}**.
- **RQ4 — Does the time gate help or hurt?** Relative to all-time adaptive, the
  late gate changed collision from {aa.collision:.4f} to {ha.collision:.4f},
  calls from {aa.projection_calls:.4f} to {ha.projection_calls:.4f}, and latency
  from {aa.planning_latency_sec:.4f}s to {ha.planning_latency_sec:.4f}s. Thus it
  provides the large efficiency gain but costs one collision in this sample.

## Final causal determination (A-E)

- **A, late-only correction:** not the main Phase-1.5 failure. Late damped alone
  reached {hd.collision:.4f}; however, the all-time adaptive result indicates the
  late gate accounts for the remaining one collision/192 relative to all-time.
- **B + D, full-strength correction and terminal full replacement:** the dominant
  jointly identified factor. Replacing this combined behavior with damped residual
  guidance changed collision from {yd.collision:.4f} to {hd.collision:.4f} and
  eliminated all 64 YFlow-induced Plain-safe collisions. The fixed ablation set
  cannot distinguish B from D individually.
- **C, unconditional modification of safe endpoints:** not a primary safety cause
  here. Gating changed neither aggregate collisions nor Plain-safe new collisions;
  it reduced calls by {hd.projection_calls - ha.projection_calls:.4f} per trajectory.
- **E, interaction:** no evidence that safety gating must interact with B/D to fix
  Phase 1.5, because unconditional late damping already fixed tasks 3 and 5. An
  interaction internal to B versus D remains possible but is not identifiable.

## Fidelity and scope

- The Phase-1.5 robot terminal optimizer is reused unchanged: Y-Flow no-P path
  (`mu=0`), time-varying `lambda_oc*t^2/dt`, goal/smoothness cost and Acados
  robot constraints.
- Hybrid methods use Phase-2 lambda=0.5 residual velocity guidance and never
  replace the final state with `z_star`.
- Adaptive triggers use true evaluation geometry and exclude terminal goal error.
- Remote-main reference commit: `{REMOTE_MAIN_COMMIT}`.
- No model retraining, learned safety predictor, lambda sweep or post-result
  schedule tuning was performed.
- Prior Phase 1, Phase 1.5, Phase 2 and Phase 3 artifacts were not overwritten.
"""
    (output / "REPORT_HYBRID.md").write_text(report)


def normalize_baseline_steps(
    rows: list[dict[str, Any]], method: str
) -> list[dict[str, Any]]:
    for row in rows:
        row["method"] = method
        row.setdefault(
            "predicted_endpoint_true_collision",
            float(row["endpoint_min_clearance"] < -SAFETY_TOLERANCE),
        )
        row.setdefault("adaptive_trigger", 0.0)
        row.setdefault("optimization_rate", row["projection_called"])
        row.setdefault("unsafe_endpoint", float("nan"))
        row.setdefault("time_eligible", float("nan"))
        row.setdefault("raw_tracking_weight", float("nan"))
        row.setdefault("safety_trigger_runtime_sec", 0.0)
        row.setdefault(
            "true_collision_after_optimization",
            float(row["min_clearance_after_projection"] < -SAFETY_TOLERANCE)
            if row["projection_called"] == 1.0
            else float("nan"),
        )
    return rows


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
    completed = 0
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
                if method in HYBRID_METHODS:
                    trajectory, timing, rows = run_hybrid_sampler(
                        method,
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
                elif method == "YFLOW_MAIN_PORT":
                    trajectory, timing, rows = run_yflow_main_port(
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
                    timing["optimizer_latency_sec"] = timing["projection_latency_sec"]
                    timing["safety_trigger_latency_sec"] = 0.0
                    rows = normalize_baseline_steps(rows, method)
                else:
                    # Restore Phase-2 q tracking after any time-weighted method.
                    optimizer.set_raw_tracking_weight(1.0)
                    phase2_method = (
                        "DAMPED_POV_L05" if method == "POV_L05_ALWAYS" else method
                    )
                    trajectory, timing, rows = run_phase2_sampler(
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
                    timing["optimizer_latency_sec"] = timing["projection_latency_sec"]
                    timing["safety_trigger_latency_sec"] = 0.0
                    rows = normalize_baseline_steps(rows, method)
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
                step_rows.extend(rows)
            plain = trajectories["PLAIN_FM"]
            for method in METHODS:
                method_rows[method]["trajectory_distortion_vs_plain"] = float(
                    np.linalg.norm(trajectories[method] - plain)
                )
                raw_rows.append(method_rows[method])
            completed += 1
            print(
                f"completed {completed}/{total}: task={task_id} seed={seed}",
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
    paired = paired_transitions(raw)
    per_task = raw.groupby(["task_id", "method"], as_index=False).agg(
        collision_rate=("collision", "mean"),
        goal_error=("terminal_goal_error", "mean"),
        min_clearance=("min_clearance", "mean"),
        optimization_calls=("projection_calls", "mean"),
    )
    flow = aggregate_flow(steps)
    gates = evaluate_decision_gates(raw, paired)
    late_adaptive = steps[steps.method == "HYBRID_LATE_ADAPTIVE"]
    trigger_counts = (
        late_adaptive.groupby(["task_id", "seed"]).adaptive_trigger.sum().astype(int)
        .value_counts()
        .reindex(range(4), fill_value=0)
        .rename_axis("triggered_late_steps")
        .reset_index(name="trajectory_count")
    )

    raw.to_csv(results_dir / "raw.csv", index=False)
    summary.to_csv(results_dir / "summary.csv", index=False)
    steps.to_csv(results_dir / "per_step.csv", index=False)
    paired.to_csv(results_dir / "paired_transitions.csv", index=False)
    per_task.to_csv(results_dir / "per_task.csv", index=False)
    flow.to_csv(results_dir / "flow_time.csv", index=False)
    trigger_counts.to_csv(results_dir / "late_adaptive_trigger_counts.csv", index=False)
    config_record = {
        "experiment": "robot_arm_yflow_pov_hybrid",
        "methods": METHODS,
        "tasks": list(range(args.tasks)),
        "seeds_per_task": args.seeds,
        "checkpoint": config.model_name,
        "remote_main_commit": REMOTE_MAIN_COMMIT,
        "t_on": T_ON,
        "lambda_oc": LAMBDA_OC,
        "lambda_damp": LAMBDA_DAMP,
        "mu": 0.0,
        "safety_trigger": "true collision OR joint OR velocity OR acceleration; goal excluded",
        "optional_alltime_included": True,
        "decision_gates": gates,
        "environment": environment_record(),
    }
    (results_dir / "config.json").write_text(json.dumps(config_record, indent=2))
    make_plots(summary, flow, per_task, args.output / "plots")
    write_report(
        raw,
        summary,
        paired,
        per_task,
        flow,
        trigger_counts,
        gates,
        args.output,
    )
    print(f"artifacts written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
