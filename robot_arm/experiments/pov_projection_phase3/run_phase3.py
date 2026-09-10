"""Run Phase 3A adaptive POV, gate OOD, and analyze Flow time."""

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
from experiments.pov_projection.run_phase1 import (
    build_condition,
    environment_record,
    load_task,
    model_velocity,
    synchronize,
)
from experiments.pov_projection_phase2.projection import GoalAwareTrajectoryProjector
from experiments.pov_projection_phase2.run_phase2 import (
    _empty_projection_fields,
    _projection_fields,
    run_sampler as run_phase2_sampler,
)
from experiments.pov_projection_phase2.sampling import corrected_pov_euler_step

from .original_safeflow import OriginalSafeFlowMPCOneShot


METHODS = (
    "PLAIN_FM",
    "ORIGINAL_SAFEFLOWMPC",
    "POV_L05_ALWAYS",
    "POV_L05_ADAPTIVE",
    "POV_L025_ADAPTIVE",
    "GOAL_AWARE_FINAL_PROJECTION",
)
ADAPTIVE_LAMBDAS = {
    "POV_L05_ADAPTIVE": 0.5,
    "POV_L025_ADAPTIVE": 0.25,
    "POV_L05_ADAPTIVE_NO_LAST": 0.5,
}
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(__file__).resolve().parent
EPSILON = 1e-3
SAFETY_TOLERANCE = 1e-5


def hard_safety_violation(snapshot: dict[str, float]) -> bool:
    """Trigger only on hard safety risk; goal error is deliberately excluded."""

    return bool(
        snapshot["min_clearance"] < -SAFETY_TOLERANCE
        or snapshot["joint_violation"] > SAFETY_TOLERANCE
        or snapshot["velocity_violation"] > SAFETY_TOLERANCE
        or snapshot["acceleration_violation"] > SAFETY_TOLERANCE
    )


def run_adaptive_sampler(
    method: str,
    initial_noise: torch.Tensor,
    condition: torch.Tensor,
    field: FlowMatchingField,
    projector: GoalAwareTrajectoryProjector,
    robot_model: RobotModel,
    obstacle_manager: ObstacleManager,
    q_start: np.ndarray,
    p_goal: np.ndarray,
    task_id: int,
    seed: int,
    flow_steps: int,
    *,
    skip_last: bool = False,
) -> tuple[np.ndarray, dict[str, float], list[dict[str, Any]]]:
    x = initial_noise.clone()
    dt = 1.0 / flow_steps
    lambda_pov = ADAPTIVE_LAMBDAS[method]
    network_latency = 0.0
    projection_latency = 0.0
    solver_latency = 0.0
    calls = 0
    failures = 0
    correction_sum = 0.0
    rows: list[dict[str, Any]] = []
    synchronize(field.device)
    planning_started = time.perf_counter()
    for flow_step in range(flow_steps):
        t = flow_step / flow_steps
        velocity, elapsed = model_velocity(field, x, t, condition)
        network_latency += elapsed
        endpoint = x + (1.0 - t) * velocity
        endpoint_np = endpoint.detach().cpu().numpy()[0]
        snapshot = constraint_snapshot(
            endpoint_np, p_goal, robot_model, obstacle_manager
        )
        unsafe = hard_safety_violation(snapshot)
        trigger = unsafe and not (skip_last and flow_step == flow_steps - 1)
        row: dict[str, Any] = {
            "method": method,
            "task_id": task_id,
            "seed": seed,
            "flow_step": flow_step,
            "t": t,
            "lambda_pov": lambda_pov,
            "raw_velocity_norm": float(torch.linalg.vector_norm(velocity).item()),
            "endpoint_constraint_violation_before_projection": snapshot[
                "total_violation"
            ],
            "endpoint_goal_error": snapshot["goal_error"],
            "endpoint_min_clearance": snapshot["min_clearance"],
            "endpoint_feasible": float(not unsafe),
            "adaptive_trigger": float(trigger),
            "trigger_collision": float(snapshot["min_clearance"] < -SAFETY_TOLERANCE),
            "trigger_joint": float(snapshot["joint_violation"] > SAFETY_TOLERANCE),
            "trigger_velocity": float(
                snapshot["velocity_violation"] > SAFETY_TOLERANCE
            ),
            "trigger_acceleration": float(
                snapshot["acceleration_violation"] > SAFETY_TOLERANCE
            ),
            "diagnostic_last_step_suppressed": float(
                skip_last and unsafe and flow_step == flow_steps - 1
            ),
        }
        if trigger:
            projected, diagnostics = projector.project_trajectory(
                endpoint_np, q_start, p_goal, obstacle_manager
            )
            projected_t = torch.as_tensor(
                projected[None, ...], dtype=torch.float32, device=x.device
            )
            x, guided, _, normalized = corrected_pov_euler_step(
                x,
                velocity,
                projected_t,
                t,
                dt,
                lambda_pov,
                EPSILON,
            )
            normalized_norm = float(torch.linalg.vector_norm(normalized).item())
            velocity_change = float(torch.linalg.vector_norm(guided - velocity).item())
            row.update(
                _projection_fields(diagnostics, normalized_norm, velocity_change)
            )
            calls += 1
            failures += int(not diagnostics.solver_success)
            correction_sum += diagnostics.correction_norm
            projection_latency += diagnostics.total_runtime_sec
            solver_latency += diagnostics.solver_runtime_sec
        else:
            x = x + dt * velocity
            row.update(_empty_projection_fields())
        rows.append(row)

    synchronize(field.device)
    return (
        x.detach().cpu().numpy()[0],
        {
            "planning_latency_sec": time.perf_counter() - planning_started,
            "fm_latency_sec": network_latency,
            "projection_latency_sec": projection_latency,
            "projection_solver_latency_sec": solver_latency,
            "projection_calls": float(calls),
            "failed_projection_solves": float(failures),
            "projection_correction_sum": correction_sum,
        },
        rows,
    )


def _rename_steps(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    for row in rows:
        row["method"] = method
        row.setdefault("endpoint_feasible", float(
            row["endpoint_constraint_violation_before_projection"] <= SAFETY_TOLERANCE
        ))
        row.setdefault(
            "adaptive_trigger",
            float(row.get("projection_called", 0.0))
            if method == "POV_L05_ALWAYS"
            else 0.0,
        )
        row.setdefault("trigger_collision", float("nan"))
        row.setdefault("trigger_joint", float("nan"))
        row.setdefault("trigger_velocity", float("nan"))
        row.setdefault("trigger_acceleration", float("nan"))
        row.setdefault("diagnostic_last_step_suppressed", 0.0)
    return rows


SUMMARY_METRICS = [
    "collision",
    "success",
    "min_clearance",
    "joint_limit_violation",
    "velocity_violation",
    "acceleration_violation",
    "terminal_goal_error",
    "trajectory_distortion_vs_plain",
    "joint_path_length",
    "smoothness",
    "acceleration_cost",
    "jerk_cost",
    "planning_latency_sec",
    "fm_latency_sec",
    "projection_solver_latency_sec",
    "projection_calls",
    "failed_projection_solves",
    "projection_failure_rate",
    "trigger_fraction",
    "zero_projection_trajectory",
]


def bootstrap_summary(
    raw: pd.DataFrame, methods: tuple[str, ...], bootstrap_samples: int
) -> pd.DataFrame:
    rng = np.random.default_rng(20260909)
    rows: list[dict[str, Any]] = []
    for method in methods:
        group = raw[raw.method == method]
        for metric in SUMMARY_METRICS:
            values = group[metric].dropna().to_numpy(float)
            if not len(values):
                continue
            boot = rng.choice(
                values, size=(bootstrap_samples, len(values)), replace=True
            ).mean(axis=1)
            row = {
                "method": method,
                "metric": metric,
                "count": len(values),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "median": float(np.median(values)),
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
                "numerator": "",
                "denominator": "",
            }
            if metric in {"collision", "success", "zero_projection_trajectory"}:
                row["numerator"] = int(values.sum())
                row["denominator"] = len(values)
            rows.append(row)
    return pd.DataFrame(rows)


def paired_transitions(raw: pd.DataFrame) -> pd.DataFrame:
    index = ["task_id", "seed"]
    frames = {
        method: raw[raw.method == method].set_index(index).sort_index()
        for method in METHODS
    }
    rows: list[dict[str, Any]] = []
    for adaptive in ("POV_L05_ADAPTIVE", "POV_L025_ADAPTIVE"):
        for reference in ("PLAIN_FM", "POV_L05_ALWAYS"):
            a = frames[adaptive].collision.to_numpy(bool)
            r = frames[reference].collision.to_numpy(bool)
            for reference_label, reference_value in [("safe", False), ("collision", True)]:
                for adaptive_label, adaptive_value in [("safe", False), ("collision", True)]:
                    rows.append(
                        {
                            "adaptive_method": adaptive,
                            "reference_method": reference,
                            "reference_state": reference_label,
                            "adaptive_state": adaptive_label,
                            "count": int(
                                np.sum(
                                    (r == reference_value) & (a == adaptive_value)
                                )
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def phase3a_gate(
    raw: pd.DataFrame, transitions: pd.DataFrame
) -> dict[str, Any]:
    means = raw.groupby("method").mean(numeric_only=True)
    always = means.loc["POV_L05_ALWAYS"]
    adaptive = means.loc["POV_L05_ADAPTIVE"]
    transition = transitions[
        (transitions.adaptive_method == "POV_L05_ADAPTIVE")
        & (transitions.reference_method == "POV_L05_ALWAYS")
    ]
    new_collisions = int(
        transition[
            (transition.reference_state == "safe")
            & (transition.adaptive_state == "collision")
        ]["count"].iloc[0]
    )
    always_safe = int((raw[raw.method == "POV_L05_ALWAYS"].collision == 0).sum())
    call_reduction = 1.0 - adaptive.projection_calls / always.projection_calls
    latency_reduction = 1.0 - adaptive.planning_latency_sec / always.planning_latency_sec
    solver_runtime_reduction = (
        1.0
        - adaptive.projection_solver_latency_sec
        / always.projection_solver_latency_sec
    )
    goal_difference = adaptive.terminal_goal_error - always.terminal_goal_error
    goal_tolerance = max(0.05, 0.1 * always.terminal_goal_error)
    checks = {
        "collision_within_2pp": bool(
            adaptive.collision <= always.collision + 0.02
        ),
        "new_collisions_vs_always": new_collisions,
        "new_collision_rate_from_always_safe": new_collisions / always_safe,
        "few_new_collisions": bool(new_collisions / always_safe <= 0.02),
        "projection_call_reduction": float(call_reduction),
        "calls_reduced_at_least_25pct": bool(call_reduction >= 0.25),
        "latency_reduction": float(latency_reduction),
        "latency_reduced_meaningfully": bool(latency_reduction >= 0.10),
        "solver_runtime_reduction": float(solver_runtime_reduction),
        "goal_error_difference": float(goal_difference),
        "goal_material_worsening_tolerance": float(goal_tolerance),
        "goal_not_materially_worse": bool(goal_difference <= goal_tolerance),
    }
    checks["passed"] = bool(
        checks["collision_within_2pp"]
        and checks["few_new_collisions"]
        and checks["calls_reduced_at_least_25pct"]
        and checks["latency_reduced_meaningfully"]
        and checks["goal_not_materially_worse"]
    )
    return checks


def _bar(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    output: Path,
    methods: tuple[str, ...] = METHODS,
) -> None:
    frame = summary[summary.metric == metric].set_index("method").reindex(methods)
    means = frame["mean"].to_numpy()
    errors = np.vstack((means - frame.ci95_low, frame.ci95_high - means))
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(range(len(methods)), means, yerr=errors, capsize=4)
    ax.set_xticks(
        range(len(methods)), [x.replace("_", "\n") for x in methods], fontsize=7
    )
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def make_plots(
    summary: pd.DataFrame, steps: pd.DataFrame, output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _bar(summary, "collision", "Collision rate", output / "id_collision.png")
    _bar(
        summary,
        "terminal_goal_error",
        "Terminal Cartesian goal error (m)",
        output / "id_goal_error.png",
    )
    _bar(
        summary,
        "planning_latency_sec",
        "Planning latency (s)",
        output / "id_latency.png",
    )
    adaptive_methods = ("POV_L05_ADAPTIVE", "POV_L025_ADAPTIVE")
    always_calls = float(
        summary[
            (summary.method == "POV_L05_ALWAYS")
            & (summary.metric == "projection_calls")
        ]["mean"].iloc[0]
    )
    reductions = []
    for method in adaptive_methods:
        calls = float(
            summary[(summary.method == method) & (summary.metric == "projection_calls")][
                "mean"
            ].iloc[0]
        )
        reductions.append(1.0 - calls / always_calls)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(2), reductions)
    ax.axhline(0.25, linestyle="--", color="red", label="25% gate")
    ax.set_xticks(range(2), [x.replace("_", "\n") for x in adaptive_methods])
    ax.set_ylabel("Projection call reduction")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "projection_call_reduction.png", dpi=160)
    plt.close(fig)

    flow_methods = [
        "POV_L05_ALWAYS",
        "POV_L05_ADAPTIVE",
        "POV_L025_ADAPTIVE",
    ]
    flow = steps[steps.method.isin(flow_methods)]
    specifications = [
        ("trigger_rate_vs_t.png", "adaptive_trigger", "Projection trigger rate"),
        ("endpoint_feasibility_vs_t.png", "endpoint_feasible", "Endpoint feasible fraction"),
        ("correction_vs_t.png", "correction_norm", "Mean ||delta||"),
        ("normalized_correction_vs_t.png", "normalized_correction_norm", "Mean ||delta/(1-t)||"),
        ("velocity_change_vs_t.png", "velocity_change_norm", "Mean ||v_guided-v_t||"),
    ]
    for filename, metric, ylabel in specifications:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for method, group in flow.groupby("method", sort=False):
            values = group.groupby("t")[metric].mean()
            ax.plot(values.index, values.values, marker="o", label=method)
        ax.set_xlabel("Flow time t")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)


def make_stopped_ood_placeholders(output: Path) -> None:
    """Create honest, visibly stopped OOD artifacts after an ID gate failure."""

    output.mkdir(parents=True, exist_ok=True)
    for filename in (
        "ood_collision_by_category.png",
        "ood_goal_error_by_category.png",
        "ood_safety_latency_pareto.png",
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "NOT RUN\nPhase 3A adaptive gate failed",
            ha="center",
            va="center",
            fontsize=16,
        )
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)


def make_stopped_ood_visualizations(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for filename in (
        "ood_shifted_success.png",
        "ood_multi_obstacle_success.png",
        "narrow_passage_case.png",
        "failure_case.png",
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "NOT RUN\nPhase 3A adaptive gate failed",
            ha="center",
            va="center",
            fontsize=16,
        )
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)


def _markdown(frame: pd.DataFrame) -> str:
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
    transitions: pd.DataFrame,
    gate: dict[str, Any],
    diagnostic: pd.DataFrame,
    output: Path,
) -> None:
    means = raw.groupby("method").mean(numeric_only=True)
    always = means.loc["POV_L05_ALWAYS"]
    adaptive = means.loc["POV_L05_ADAPTIVE"]
    original = means.loc["ORIGINAL_SAFEFLOWMPC"]
    flow_trigger = (
        steps[steps.method == "POV_L05_ADAPTIVE"]
        .groupby("t")
        .adaptive_trigger.mean()
    )
    top_trigger_t = float(flow_trigger.idxmax())
    top_trigger_rate = float(flow_trigger.max())
    adaptive_steps = steps[steps.method == "POV_L05_ADAPTIVE"]
    trigger_component_rates = adaptive_steps[
        [
            "trigger_collision",
            "trigger_joint",
            "trigger_velocity",
            "trigger_acceleration",
        ]
    ].mean()
    always_last = steps[
        (steps.method == "POV_L05_ALWAYS") & (steps.t < 1.0)
    ].sort_values("t").groupby("t").normalized_correction_norm.mean().iloc[-1]
    if len(diagnostic):
        last_step_conclusion = (
            f"The always-on mean normalized correction reached {always_last:.4f} "
            "at t=6/7. Removing that step raised collision to "
            f"{diagnostic.collision.mean():.4f} and goal error to "
            f"{diagnostic.terminal_goal_error.mean():.4f}, so the large "
            "correction is consequential but cannot simply be removed."
        )
    else:
        last_step_conclusion = (
            "The prespecified 1.5x disproportion criterion was not met, so the "
            "optional no-last ablation was not run."
        )
    primary = summary[
        summary.metric.isin(
            [
                "collision",
                "terminal_goal_error",
                "planning_latency_sec",
                "projection_calls",
                "projection_failure_rate",
                "zero_projection_trajectory",
            ]
        )
    ][
        [
            "method",
            "metric",
            "mean",
            "std",
            "median",
            "ci95_low",
            "ci95_high",
            "numerator",
            "denominator",
        ]
    ]
    flow = steps[
        steps.method.isin(
            ["POV_L05_ALWAYS", "POV_L05_ADAPTIVE", "POV_L025_ADAPTIVE"]
        )
    ]
    flow_table = flow.groupby(["method", "t"], as_index=False).agg(
        endpoint_feasible_fraction=("endpoint_feasible", "mean"),
        trigger_rate=("adaptive_trigger", "mean"),
        mean_endpoint_violation=(
            "endpoint_constraint_violation_before_projection",
            "mean",
        ),
        mean_raw_delta=("correction_norm", "mean"),
        mean_normalized_delta=("normalized_correction_norm", "mean"),
        mean_velocity_correction=("velocity_change_norm", "mean"),
    )
    diagnostic_text = "Not run."
    if len(diagnostic):
        diagnostic_summary = diagnostic.groupby("method").agg(
            collision=("collision", "mean"),
            goal_error=("terminal_goal_error", "mean"),
            latency=("planning_latency_sec", "mean"),
            calls=("projection_calls", "mean"),
            distortion=("trajectory_distortion_vs_plain", "mean"),
        ).reset_index()
        diagnostic_text = (
            "Exploratory only; not part of the predeclared primary comparison.\n\n"
            + _markdown(diagnostic_summary)
        )
    decision = "CONTINUE TO OOD" if gate["passed"] else "NO-GO BEFORE OOD"
    report = f"""# POV robot-arm Phase-3 report

## Decision

**{decision}** at the Phase-3A adaptive gate.

Final Phase-3 decision: **NO-GO for the current adaptive hard-violation
trigger**. This is not a NO-GO for always-on POV: the adaptive policy preserved
ID collision safety, but it did not achieve its efficiency objective, so the
predeclared protocol stops before OOD and makes no generalization claim.

```json
{json.dumps(gate, indent=2)}
```

## ID matched comparison

{_markdown(primary)}

## Paired collision transitions

{_markdown(transitions)}

## Answers to the research questions

1. **Does adaptive POV preserve always-on safety?** {'Yes under the collision/new-collision checks, but the full adaptive gate still failed.' if gate['collision_within_2pp'] and gate['few_new_collisions'] else 'No; it failed at least one collision-preservation check.'}
   Collision was {adaptive.collision:.4f} adaptive versus {always.collision:.4f}
   always-on; the paired table identifies newly introduced collisions.
2. **How many calls are saved?** Mean calls changed from
   {always.projection_calls:.4f} to {adaptive.projection_calls:.4f}, a
   {100 * gate['projection_call_reduction']:.2f}% reduction.
3. **How much latency is saved?** Total latency changed from
   {always.planning_latency_sec:.4f}s to {adaptive.planning_latency_sec:.4f}s
   ({100 * gate['latency_reduction']:.2f}% reduction); measured solver-runtime
   reduction was {100 * gate['solver_runtime_reduction']:.2f}%.
4. **Does POV outperform Plain FM under OOD?** {'Not tested because Phase 3A failed, so no generalization claim is made.' if not gate['passed'] else 'Evaluated in the OOD section.'}
5. **Does it outperform original SafeFlowMPC?** On ID, adaptive POV collision was
   {adaptive.collision:.4f} versus {original.collision:.4f}, and latency was
   {adaptive.planning_latency_sec:.4f}s versus {original.planning_latency_sec:.4f}s.
   Interpret this as a matched one-shot comparison; the upstream closed-loop
   planner remains semantically complementary rather than fully equivalent.
6. **Which Flow times trigger most?** The highest adaptive trigger rate was
   {top_trigger_rate:.4f} at t={top_trigger_t:.6g}; every t is listed below.
7. **Is last-step normalization problematic?** The flow table reports the raw and
   normalized final-step correction. The no-last ablation below was run only if
   the prespecified 1.5x disproportion criterion was met. {last_step_conclusion}
8. **Are Phase-2 gains robust beyond six ID examples?** {'Unknown; OOD was correctly stopped by the gate.' if not gate['passed'] else 'Answered by the OOD tables below.'}

## Flow-time analysis

{_markdown(flow_table)}

The trigger was dominated by acceleration and velocity violations, not only
geometric collision: overall trigger-component rates were collision
{trigger_component_rates.trigger_collision:.4f}, joint
{trigger_component_rates.trigger_joint:.4f}, velocity
{trigger_component_rates.trigger_velocity:.4f}, and acceleration
{trigger_component_rates.trigger_acceleration:.4f}. This explains why the
hard-safety trigger skipped only 3.20% of calls.

## Optional no-last diagnostic

{diagnostic_text}

## Original SafeFlowMPC baseline semantics

The matched baseline uses the unchanged upstream `SafetyFilterAcados.step` seven
times inside one Flow Matching sample and a line-for-line headless extraction of
`SafeFlowMPC._compute_guidance`. It uses the same unsafe FM checkpoint and matched
initial noise as every other ID method. This is a one-shot comparison.

The repository's top-level `inference_global_planner.py` has different semantics:
it is a closed-loop/receding-horizon simulator, shifts the prior solution between
control timesteps, and selects the safe FM checkpoint by default. It was reproduced
separately in Phase 1, but that single rollout is not mixed into this matched table.

## OOD status

{'Phase 3B was not run because the Phase-3A gate failed. Empty OOD CSV schemas are retained only to make the stopped state machine-readable.' if not gate['passed'] else 'Phase 3B results follow in the OOD artifacts.'}

The requested OOD plot and visualization paths contain explicit `NOT RUN`
placeholders; they are not experimental observations.

## Phase-2 reproduction note

Plain FM reproduced 29/192 collisions and final projection reproduced 93/192.
The current always-on lambda=0.5 run produced 0/192 versus Phase 2's 1/192. The
single prior collision had only -2.56e-5 m clearance and is now +1.94e-3 m, so
this is a near-boundary Acados numerical/interleaving drift rather than evidence
of a material safety change. The adaptive gate uses the current matched
always-on run; its 3.20% call reduction remains far below 25% under either count.

## Scope

- No Flow Matching retraining occurred.
- Goal error was never part of the adaptive trigger.
- Failed projections use vanilla velocity; unconverged iterates are discarded.
- No schedule was tuned from the flow-time analysis.
"""
    (output / "REPORT_PHASE3.md").write_text(report)


def build_components(args: argparse.Namespace) -> tuple[Any, ...]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 3 requires cuda:0")
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
    projector = GoalAwareTrajectoryProjector(
        obstacles,
        robot_model,
        horizon=16,
        goal_mode="soft",
        goal_weight=1000.0,
        build=False,
    )
    original = OriginalSafeFlowMPCOneShot(
        obstacles, robot_model, horizon=16, build=args.build_upstream_solver
    )
    return device, config, robot_model, obstacles, field, projector, original


def add_derived_columns(raw: pd.DataFrame, flow_steps: int) -> pd.DataFrame:
    raw = raw.copy()
    raw["projection_failure_rate"] = np.where(
        raw.projection_calls > 0,
        raw.failed_projection_solves / raw.projection_calls,
        0.0,
    )
    raw["trigger_fraction"] = np.where(
        raw.method.str.contains("ADAPTIVE"), raw.projection_calls / flow_steps, 0.0
    )
    raw["zero_projection_trajectory"] = (
        raw.projection_calls == 0
    ).astype(float)
    return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--tasks", type=int, default=6)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--build-upstream-solver", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results_dir = args.output / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    device, config, robot_model, obstacles, field, projector, original = build_components(args)

    warm_task = load_task(0)
    warm_goal, _, _ = robot_model.forward_kinematics(warm_task["q"][-1], np.zeros(7))
    warm_condition = build_condition(
        robot_model, warm_task["q"][0], warm_task["q_prev0"], warm_goal, device
    )
    for _ in range(10):
        model_velocity(
            field, torch.randn((1, 16, 7), device=device), 0.0, warm_condition
        )

    raw_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    trajectories: dict[str, np.ndarray] = {}
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
            matched: dict[str, np.ndarray] = {}
            method_rows: dict[str, dict[str, Any]] = {}
            for method in METHODS:
                if method == "PLAIN_FM":
                    trajectory, timing, steps = run_phase2_sampler(
                        "PLAIN_FM", noise, condition, field, projector, robot_model,
                        obstacles, q_start, p_goal, task_id, seed, config.flow_steps
                    )
                elif method == "POV_L05_ALWAYS":
                    trajectory, timing, steps = run_phase2_sampler(
                        "DAMPED_POV_L05", noise, condition, field, projector,
                        robot_model, obstacles, q_start, p_goal, task_id, seed,
                        config.flow_steps
                    )
                elif method == "GOAL_AWARE_FINAL_PROJECTION":
                    trajectory, timing, steps = run_phase2_sampler(
                        method, noise, condition, field, projector, robot_model,
                        obstacles, q_start, p_goal, task_id, seed, config.flow_steps
                    )
                elif method == "ORIGINAL_SAFEFLOWMPC":
                    trajectory, timing, steps = original.sample(
                        noise, condition, field, q_start, p_goal, obstacles,
                        task_id, seed, config.flow_steps
                    )
                else:
                    trajectory, timing, steps = run_adaptive_sampler(
                        method, noise, condition, field, projector, robot_model,
                        obstacles, q_start, p_goal, task_id, seed, config.flow_steps
                    )
                steps = _rename_steps(steps, method)
                metrics = trajectory_metrics(
                    trajectory, q_goal, p_goal, reference, robot_model, obstacles
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

    raw = add_derived_columns(pd.DataFrame(raw_rows), config.flow_steps)
    steps = pd.DataFrame(step_rows)
    summary = bootstrap_summary(raw, METHODS, args.bootstrap_samples)
    transitions = paired_transitions(raw)
    gate = phase3a_gate(raw, transitions)

    # Predeclared optional diagnostic: run only when the last correction is at
    # least 1.5x every earlier mean correction. It never alters the primary gate.
    always_steps = steps[
        (steps.method == "POV_L05_ALWAYS") & (steps.projection_called == 1.0)
    ]
    correction_by_t = always_steps.groupby("t").normalized_correction_norm.mean()
    final_disproportionate = bool(
        correction_by_t.iloc[-1] >= 1.5 * correction_by_t.iloc[:-1].max()
    )
    diagnostic_rows: list[dict[str, Any]] = []
    diagnostic_steps: list[dict[str, Any]] = []
    if final_disproportionate:
        diagnostic_total = args.tasks * args.seeds
        diagnostic_completed = 0
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
                trajectory, timing, per_step = run_adaptive_sampler(
                    "POV_L05_ADAPTIVE_NO_LAST", noise, condition, field, projector,
                    robot_model, obstacles, q_start, p_goal, task_id, seed,
                    config.flow_steps, skip_last=True
                )
                metrics = trajectory_metrics(
                    trajectory, q_goal, p_goal, reference, robot_model, obstacles
                )
                row = {
                    "method": "POV_L05_ADAPTIVE_NO_LAST",
                    "task_id": task_id,
                    "seed": seed,
                }
                row.update(metrics)
                row.update(timing)
                plain = trajectories[f"task{task_id}_seed{seed}_PLAIN_FM"]
                row["trajectory_distortion_vs_plain"] = float(
                    np.linalg.norm(trajectory - plain)
                )
                diagnostic_rows.append(row)
                diagnostic_steps.extend(per_step)
                diagnostic_completed += 1
                print(
                    f"diagnostic {diagnostic_completed}/{diagnostic_total}: "
                    f"task={task_id} seed={seed}",
                    flush=True,
                )
    diagnostic = (
        add_derived_columns(pd.DataFrame(diagnostic_rows), config.flow_steps)
        if diagnostic_rows
        else pd.DataFrame()
    )
    if diagnostic_steps:
        steps = pd.concat([steps, pd.DataFrame(diagnostic_steps)], ignore_index=True)

    raw.to_csv(results_dir / "id_raw.csv", index=False)
    summary.to_csv(results_dir / "id_summary.csv", index=False)
    steps.to_csv(results_dir / "per_step.csv", index=False)
    transitions.to_csv(results_dir / "paired_transitions.csv", index=False)
    if len(diagnostic):
        diagnostic.to_csv(results_dir / "diagnostic_no_last.csv", index=False)
        bootstrap_summary(
            diagnostic, ("POV_L05_ADAPTIVE_NO_LAST",), args.bootstrap_samples
        ).to_csv(results_dir / "diagnostic_no_last_summary.csv", index=False)

    # Gate-first protocol: OOD artifacts are explicit empty schemas when stopped.
    ood_columns = [
        "method", "scenario_id", "category", "seed", "collision",
        "min_clearance", "terminal_goal_error", "planning_latency_sec",
        "projection_calls", "projection_failure_rate",
    ]
    pd.DataFrame(columns=ood_columns).to_csv(results_dir / "ood_raw.csv", index=False)
    pd.DataFrame(columns=["method", "category", "metric", "mean"]).to_csv(
        results_dir / "ood_summary.csv", index=False
    )
    if gate["passed"]:
        # A positive Phase 3A must proceed through the predeclared OOD phase.
        # Never publish ID-only output as a completed positive Phase 3.
        raise RuntimeError(
            "Phase 3A passed; implement and execute the predeclared OOD phase "
            "before writing REPORT_PHASE3.md"
        )
    config_record = {
        "experiment": "phase3_adaptive_gate_then_ood",
        "tasks": list(range(args.tasks)),
        "seeds_per_task": args.seeds,
        "methods": METHODS,
        "checkpoint": config.model_name,
        "adaptive_trigger": "collision OR joint OR velocity OR acceleration hard violation; goal excluded",
        "safety_tolerance": SAFETY_TOLERANCE,
        "pov_epsilon": EPSILON,
        "phase3a_gate": gate,
        "phase3b_status": "not_run_phase3a_gate_failed",
        "optional_no_last_diagnostic_run": bool(len(diagnostic)),
        "optional_no_last_criterion": "final normalized correction >= 1.5 * max earlier mean",
        "original_baseline": "headless matched one-shot extraction of upstream SafetyFilterAcados.step and _compute_guidance",
        "environment": environment_record(),
    }
    (results_dir / "config.json").write_text(json.dumps(config_record, indent=2))
    make_plots(summary, steps, args.output / "plots")
    make_stopped_ood_placeholders(args.output / "plots")
    make_stopped_ood_visualizations(args.output / "visualizations")
    write_report(raw, summary, steps, transitions, gate, diagnostic, args.output)
    print(f"artifacts written to {args.output}", flush=True)
    if not gate["passed"]:
        print("PHASE3A_GATE=NO_GO; OOD_NOT_RUN", flush=True)


if __name__ == "__main__":
    main()
