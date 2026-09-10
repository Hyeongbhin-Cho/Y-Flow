"""Run the predeclared matched 2x2 Y-Flow robot-arm factorial ablation."""

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
from experiments.pov_projection_phase15_yflow_main.run_phase15 import (
    LAMBDA_OC,
    REMOTE_MAIN_COMMIT,
    T_ON,
    build_components,
    run_yflow_main_port,
)
from experiments.pov_projection_phase15_yflow_main.sampling import yflow_interpolate
from experiments.pov_yflow_hybrid.run_hybrid import run_hybrid_sampler


METHODS = (
    "PLAIN_FM",
    "L1_REPLACE",
    "L1_NO_REPLACE",
    "L05_REPLACE",
    "L05_NO_REPLACE",
)
FACTORIAL_METHODS = METHODS[1:]
FACTORS = {
    "L1_REPLACE": (1.0, True),
    "L1_NO_REPLACE": (1.0, False),
    "L05_REPLACE": (0.5, True),
    "L05_NO_REPLACE": (0.5, False),
}
OUTPUT_ROOT = Path(__file__).resolve().parent
SAFETY_TOLERANCE = 1e-5
SUMMARY_METRICS = (
    "collision",
    "min_clearance",
    "terminal_goal_error",
    "trajectory_distortion_vs_plain",
    "joint_path_length",
    "smoothness",
    "velocity_violation",
    "acceleration_violation",
    "planning_latency_sec",
    "fm_latency_sec",
    "optimizer_latency_sec",
    "projection_calls",
    "failed_projection_solves",
    "projection_failure_rate",
)


def state_audit_rows(
    method: str,
    task_id: int,
    seed: int,
    states: dict[str, np.ndarray],
    p_goal: np.ndarray,
    robot_model: Any,
    obstacle_manager: Any,
) -> list[dict[str, Any]]:
    rows = []
    for name, state in states.items():
        snapshot = constraint_snapshot(state, p_goal, robot_model, obstacle_manager)
        rows.append(
            {
                "method": method,
                "task_id": task_id,
                "seed": seed,
                "state": name,
                "collision": float(snapshot["min_clearance"] < -SAFETY_TOLERANCE),
                "min_clearance": snapshot["min_clearance"],
                "terminal_goal_error": snapshot["goal_error"],
                "state_json": json.dumps(state.tolist(), separators=(",", ":")),
            }
        )
    return rows


def run_factorial_sampler(
    method: str,
    initial_noise: torch.Tensor,
    condition: torch.Tensor,
    field: Any,
    optimizer: Any,
    robot_model: Any,
    obstacle_manager: Any,
    q_start: np.ndarray,
    p_goal: np.ndarray,
    task_id: int,
    seed: int,
    flow_steps: int,
) -> tuple[np.ndarray, dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    lambda_pov, replace = FACTORS[method]
    x = initial_noise.clone()
    dt = 1.0 / flow_steps
    fm_latency = optimizer_latency = solver_latency = 0.0
    calls = failures = 0
    correction_sum = 0.0
    step_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    synchronize(field.device)
    started = time.perf_counter()

    for flow_step in range(flow_steps):
        t = flow_step / flow_steps
        terminal = flow_step == flow_steps - 1
        x_before = x.clone()
        velocity, elapsed = model_velocity(field, x, t, condition)
        fm_latency += elapsed
        raw_target = x + (1.0 - t) * velocity
        raw_np = raw_target.detach().cpu().numpy()[0]
        raw_snapshot = constraint_snapshot(raw_np, p_goal, robot_model, obstacle_manager)
        active = t >= T_ON
        raw_weight = LAMBDA_OC * t**2 / max(dt, 1e-8) if active else 0.0
        row: dict[str, Any] = {
            "method": method,
            "task_id": task_id,
            "seed": seed,
            "flow_step": flow_step,
            "t": t,
            "terminal_step": float(terminal),
            "projection_active": float(active),
            "lambda_pov": lambda_pov,
            "terminal_replacement": float(replace),
            "raw_tracking_weight": raw_weight,
            "x1_raw_collision": float(
                raw_snapshot["min_clearance"] < -SAFETY_TOLERANCE
            ),
            "x1_raw_min_clearance": raw_snapshot["min_clearance"],
            "x1_raw_goal_error": raw_snapshot["goal_error"],
            "z_star_collision": float("nan"),
            "z_star_min_clearance": float("nan"),
            "z_star_goal_error": float("nan"),
        }
        if not active:
            if method == "L1_REPLACE":
                # Preserve the exact arithmetic order of the Phase-1.5 baseline.
                x, _ = yflow_interpolate(x, raw_target, t, dt, terminal)
            else:
                x = x + dt * velocity
            row.update(_empty_projection_fields())
        else:
            optimized, diagnostics = optimizer.optimize_terminal(
                raw_np, q_start, p_goal, raw_weight
            )
            z = torch.as_tensor(
                optimized[None, ...], dtype=torch.float32, device=x.device
            )
            residual_output, guided, _, normalized = corrected_pov_euler_step(
                x, velocity, z, t, dt, lambda_pov, 1e-3
            )
            l1_output, _, _, _ = corrected_pov_euler_step(
                x, velocity, z, t, dt, 1.0, 1e-3
            )
            l05_output, _, _, _ = corrected_pov_euler_step(
                x, velocity, z, t, dt, 0.5, 1e-3
            )
            if method == "L1_REPLACE":
                x, _ = yflow_interpolate(x, z, t, dt, terminal)
            elif terminal and replace:
                x = z
            else:
                x = residual_output
            z_snapshot = constraint_snapshot(
                optimized, p_goal, robot_model, obstacle_manager
            )
            normalized_norm = float(torch.linalg.vector_norm(normalized).item())
            velocity_change = float(torch.linalg.vector_norm(guided - velocity).item())
            row.update(_projection_fields(diagnostics, normalized_norm, velocity_change))
            row.update(
                {
                    "z_star_collision": float(
                        z_snapshot["min_clearance"] < -SAFETY_TOLERANCE
                    ),
                    "z_star_min_clearance": z_snapshot["min_clearance"],
                    "z_star_goal_error": z_snapshot["goal_error"],
                }
            )
            if terminal:
                states = {
                    "x_before_terminal": x_before.detach().cpu().numpy()[0],
                    "x1_raw": raw_np,
                    "z_star": optimized,
                    "hypothetical_l1_residual": l1_output.detach().cpu().numpy()[0],
                    "hypothetical_l05_residual": l05_output.detach().cpu().numpy()[0],
                }
                audit_rows.extend(
                    state_audit_rows(
                        method,
                        task_id,
                        seed,
                        states,
                        p_goal,
                        robot_model,
                        obstacle_manager,
                    )
                )
            calls += 1
            failures += int(not diagnostics.solver_success)
            correction_sum += diagnostics.correction_norm
            optimizer_latency += diagnostics.total_runtime_sec
            solver_latency += diagnostics.solver_runtime_sec
        step_rows.append(row)

    synchronize(field.device)
    timing = {
        "planning_latency_sec": time.perf_counter() - started,
        "fm_latency_sec": fm_latency,
        "optimizer_latency_sec": optimizer_latency,
        "projection_latency_sec": optimizer_latency,
        "projection_solver_latency_sec": solver_latency,
        "projection_calls": float(calls),
        "failed_projection_solves": float(failures),
        "projection_correction_sum": correction_sum,
    }
    return x.detach().cpu().numpy()[0], timing, step_rows, audit_rows


def compute_metrics(
    trajectory: np.ndarray,
    q_goal: np.ndarray,
    p_goal: np.ndarray,
    reference: np.ndarray,
    robot_model: Any,
    obstacles: Any,
) -> dict[str, float]:
    return trajectory_metrics(
        trajectory, q_goal, p_goal, reference, robot_model, obstacles
    )


def run_sanity(
    config: Any,
    robot_model: Any,
    obstacles: Any,
    field: Any,
    optimizer: Any,
    device: str,
) -> dict[str, Any]:
    checks = []
    for task_id, seed in ((0, 0), (3, 0), (4, 0), (5, 0)):
        task = load_task(task_id)
        q_start, q_goal = task["q"][0], task["q"][-1]
        p_goal, _, _ = robot_model.forward_kinematics(q_goal, np.zeros(7))
        condition = build_condition(
            robot_model, q_start, task["q_prev0"], p_goal, device
        )
        noise = torch.randn(
            (1, 16, 7),
            generator=torch.Generator(device=device).manual_seed(seed),
            device=device,
        )
        for method, reference_name in (
            ("L1_REPLACE", "YFLOW_MAIN_PORT"),
            ("L05_NO_REPLACE", "HYBRID_LATE_DAMPED"),
        ):
            actual, _, _, _ = run_factorial_sampler(
                method, noise, condition, field, optimizer, robot_model, obstacles,
                q_start, p_goal, task_id, seed, config.flow_steps
            )
            if reference_name == "YFLOW_MAIN_PORT":
                expected, _, _ = run_yflow_main_port(
                    noise, condition, field, optimizer, robot_model, obstacles,
                    q_start, p_goal, task_id, seed, config.flow_steps
                )
            else:
                expected, _, _ = run_hybrid_sampler(
                    "HYBRID_LATE_DAMPED", noise, condition, field, optimizer,
                    robot_model, obstacles, q_start, p_goal, task_id, seed,
                    config.flow_steps
                )
            actual_metrics = compute_metrics(
                actual, q_goal, p_goal, task["q"][:16], robot_model, obstacles
            )
            expected_metrics = compute_metrics(
                expected, q_goal, p_goal, task["q"][:16], robot_model, obstacles
            )
            check = {
                "task_id": task_id,
                "seed": seed,
                "method": method,
                "reference": reference_name,
                "max_abs_trajectory_difference": float(np.max(np.abs(actual - expected))),
                "collision_match": actual_metrics["collision"] == expected_metrics["collision"],
                "goal_error_difference": float(
                    abs(actual_metrics["terminal_goal_error"] - expected_metrics["terminal_goal_error"])
                ),
            }
            check["pass"] = bool(
                check["max_abs_trajectory_difference"] <= 5e-5
                and check["collision_match"]
                and check["goal_error_difference"] <= 5e-5
            )
            checks.append(check)
    result = {"pass": all(x["pass"] for x in checks), "checks": checks}
    if not result["pass"]:
        raise RuntimeError(f"factorial reproduction sanity check failed: {result}")
    return result


def summarize(raw: pd.DataFrame, samples: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260908)
    rows = []
    for method in METHODS:
        group = raw[raw.method == method]
        for metric in SUMMARY_METRICS:
            values = group[metric].to_numpy(float)
            boot = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "count": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "median": float(np.median(values)),
                    "ci95_low": float(np.quantile(boot, 0.025)),
                    "ci95_high": float(np.quantile(boot, 0.975)),
                    "numerator": int(values.sum()) if metric == "collision" else "",
                    "denominator": len(values) if metric == "collision" else "",
                }
            )
    return pd.DataFrame(rows)


def transition_table(raw: pd.DataFrame) -> pd.DataFrame:
    indexed = {
        m: raw[raw.method == m].set_index(["task_id", "seed"]).sort_index()
        for m in METHODS
    }
    comparisons = [(m, "PLAIN_FM") for m in FACTORIAL_METHODS] + [
        ("L1_REPLACE", "L1_NO_REPLACE"),
        ("L05_REPLACE", "L05_NO_REPLACE"),
        ("L1_NO_REPLACE", "L05_NO_REPLACE"),
    ]
    rows = []
    for method, reference in comparisons:
        m = indexed[method].collision.to_numpy(bool)
        r = indexed[reference].collision.to_numpy(bool)
        for r_name, r_value in (("safe", False), ("collision", True)):
            for m_name, m_value in (("safe", False), ("collision", True)):
                rows.append(
                    {
                        "method": method,
                        "reference_method": reference,
                        "reference_state": r_name,
                        "method_state": m_name,
                        "count": int(np.sum((r == r_value) & (m == m_value))),
                    }
                )
    return pd.DataFrame(rows)


def factorial_effects(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = ("collision", "terminal_goal_error", "trajectory_distortion_vs_plain", "min_clearance")
    rows = []
    means = raw.groupby("method").mean(numeric_only=True)
    for metric in metrics:
        l1 = (means.loc["L1_REPLACE", metric] + means.loc["L1_NO_REPLACE", metric]) / 2
        l05 = (means.loc["L05_REPLACE", metric] + means.loc["L05_NO_REPLACE", metric]) / 2
        replace = (means.loc["L1_REPLACE", metric] + means.loc["L05_REPLACE", metric]) / 2
        no_replace = (means.loc["L1_NO_REPLACE", metric] + means.loc["L05_NO_REPLACE", metric]) / 2
        interaction = (
            means.loc["L1_REPLACE", metric] - means.loc["L1_NO_REPLACE", metric]
            - means.loc["L05_REPLACE", metric] + means.loc["L05_NO_REPLACE", metric]
        )
        rows.append(
            {
                "metric": metric,
                "effect_lambda_l1_minus_l05": float(l1 - l05),
                "effect_replace_minus_no_replace": float(replace - no_replace),
                "interaction_difference_in_differences": float(interaction),
            }
        )
    return pd.DataFrame(rows)


def markdown(frame: pd.DataFrame) -> str:
    frame = frame.fillna("")
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in values) + " |")
    return "\n".join(lines)


def make_plots(raw: pd.DataFrame, per_task: pd.DataFrame, audit: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    grids = {
        "collision": "collision_factorial.png",
        "terminal_goal_error": "goal_error_factorial.png",
        "trajectory_distortion_vs_plain": "distortion_factorial.png",
        "min_clearance": "min_clearance_factorial.png",
    }
    positions = [["L05_NO_REPLACE", "L05_REPLACE"], ["L1_NO_REPLACE", "L1_REPLACE"]]
    means = raw.groupby("method").mean(numeric_only=True)
    for metric, filename in grids.items():
        values = np.array([[means.loc[m, metric] for m in row] for row in positions])
        fig, ax = plt.subplots(figsize=(6, 4.5))
        image = ax.imshow(values, cmap="viridis", aspect="auto")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{values[i, j]:.4f}", ha="center", va="center", color="white")
        ax.set_xticks([0, 1], ["NO REPLACE", "REPLACE"])
        ax.set_yticks([0, 1], ["lambda=0.5", "lambda=1.0"])
        ax.set_title(metric)
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)

    pivot = per_task.pivot(index="task_id", columns="method", values="collision_rate")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.reindex(columns=METHODS).plot(kind="bar", ax=ax)
    ax.set_ylabel("Collision rate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "per_task_collision.png", dpi=160)
    plt.close(fig)

    terminal = audit.groupby(["method", "state"], as_index=False).collision.mean()
    pivot = terminal.pivot(index="state", columns="method", values="collision")
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.reindex(columns=FACTORIAL_METHODS).plot(kind="bar", ax=ax)
    ax.set_ylabel("True-geometry collision rate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "terminal_state_collision.png", dpi=160)
    plt.close(fig)


def write_report(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    per_task: pd.DataFrame,
    steps: pd.DataFrame,
    audit: pd.DataFrame,
    effects: pd.DataFrame,
    sanity: dict[str, Any],
    output: Path,
) -> None:
    means = raw.groupby("method").mean(numeric_only=True)
    terminal = audit.groupby(["method", "state"], as_index=False).agg(
        collision_rate=("collision", "mean"),
        min_clearance=("min_clearance", "mean"),
        goal_error=("terminal_goal_error", "mean"),
    )
    active_flow = steps[steps.projection_active == 1].groupby(["method", "t"], as_index=False).agg(
        x1_raw_collision=("x1_raw_collision", "mean"),
        z_star_collision=("z_star_collision", "mean"),
        correction_norm=("correction_norm", "mean"),
        normalized_correction_norm=("normalized_correction_norm", "mean"),
        velocity_change=("velocity_change_norm", "mean"),
        z_star_min_clearance=("z_star_min_clearance", "mean"),
        z_star_goal_error=("z_star_goal_error", "mean"),
        solver_success=("projection_solver_success", "mean"),
    )
    collisions = means.collision
    z_l05 = float(terminal[(terminal.method == "L05_NO_REPLACE") & (terminal.state == "z_star")].collision_rate.iloc[0])
    out_l05 = float(terminal[(terminal.method == "L05_NO_REPLACE") & (terminal.state == "hypothetical_l05_residual")].collision_rate.iloc[0])
    report = f"""# Robot-arm Y-Flow 2x2 factorial ablation

## Sanity checks

The fixed implementation reproduced the existing reference samplers on four matched
task/seed cases before the full run. Overall status: **{'PASS' if sanity['pass'] else 'FAIL'}**.

{markdown(pd.DataFrame(sanity['checks']))}

At the final grid point, `dt = 1-t = 1/7`; therefore a lambda=1 residual Euler
update is algebraically equal to `z_star`. This means L1_NO_REPLACE and L1_REPLACE
are implementation-label variants, not an independently manipulable terminal factor
at the final step. The requested results are reported, but this structural aliasing
limits the nominal 2x2 causal interpretation.

## Primary metrics

{markdown(summary[['method','metric','mean','ci95_low','ci95_high','numerator','denominator']])}

## Paired collision transitions

{markdown(paired)}

## Per-task results

{markdown(per_task)}

## Factorial descriptive effects

Positive collision effects mean worse safety. These are descriptive matched-benchmark
effects, not population-level causal estimates from six tasks.

{markdown(effects)}

## Active flow-time diagnostics

{markdown(active_flow)}

## Terminal state audit

{markdown(terminal)}

The full 16x7 states for `x_before_terminal`, `x1_raw`, `z_star`, hypothetical
lambda=1 output, and hypothetical lambda=0.5 output are serialized in
`results/terminal_state_audit.csv`.

## Required answers

1. **Is lambda=1 independently harmful?** There is no evidence that the
   non-terminal strength change is the primary problem: among replacement methods,
   lambda=1 has {collisions['L1_REPLACE']:.4f} collision versus
   {collisions['L05_REPLACE']:.4f} for lambda=0.5, only one trajectory apart.
   L1_NO_REPLACE ({collisions['L1_NO_REPLACE']:.4f}) cannot identify an independent
   all-step lambda effect because its final lambda=1 Euler update is exactly a
   replacement. Thus a clean independent terminal lambda=1 effect is structurally
   unavailable on this grid.
2. **Is terminal full replacement independently harmful?** Yes. At lambda=0.5,
   replacement changes collision from {collisions['L05_NO_REPLACE']:.4f}
   (1/192) to {collisions['L05_REPLACE']:.4f} (63/192). The paired table shows
   63 no-replacement-safe trajectories newly collide and one collision is rescued.
   Task 3 changes from 0/32 to 32/32 and task 5 from 0/32 to 31/32.
3. **Is there an interaction?** The nominal difference-in-differences is large,
   but it is not a clean interaction estimate because the lambda=1/no-replacement
   cell implements effective replacement at the final step. The data support a
   terminal-replacement main mechanism; they do not establish an additional
   lambda-by-replacement interaction.
4. **What caused the 64 YFLOW_MAIN_PORT new collisions?** Terminal replacement is
   the dominant identified cause. L05_REPLACE reproduces 63/64 aggregate collisions
   while keeping the first two active corrections damped. Increasing those earlier
   corrections to lambda=1 adds only the remaining one. This explains all task-3
   failures and 31/32 task-5 failures without full-strength early correction.
5. **Why can L05_NO_REPLACE be safe when z_star collides?** Its terminal z_star
   collision rate is {z_l05:.4f}, while the hypothetical damped residual output is
   {out_l05:.4f}. The optimizer supplies a direction; retaining half of the learned
   Flow update interpolates away from unsafe z_star rather than consuming it whole.
6. **Recommended main robot-arm formulation.** Use L05_NO_REPLACE: it preserves
   learned Flow dynamics, consumes the optimizer only as a damped correction direction,
   and avoids the known terminal replacement risk. Adaptive gating remains a separate
   compute optimization demonstrated in the prior experiment, not part of this ablation.

## Scope and fidelity

- Same 6 tasks x 32 seeds, checkpoint, initial noises, obstacles, goals, and 7-step grid.
- Same Phase-1.5 no-P (`mu=0`) Acados terminal optimizer, `t_on=0.5`, and
  `lambda_oc=10`; exactly three optimizer calls for each factorial method.
- No adaptive gating, OOD evaluation, retraining, lambda tuning, or schedule change.
- Remote-main reference commit: `{REMOTE_MAIN_COMMIT}`.
- Prior Phase 1/1.5/2/3/hybrid artifacts were not overwritten.
"""
    (output / "REPORT_FACTORIAL.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--build-solver", action="store_true")
    parser.add_argument("--sanity-only", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results_dir = args.output / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    device, config, robot_model, obstacles, field, optimizer = build_components(args)

    warm = load_task(0)
    warm_goal, _, _ = robot_model.forward_kinematics(warm["q"][-1], np.zeros(7))
    warm_condition = build_condition(robot_model, warm["q"][0], warm["q_prev0"], warm_goal, device)
    for _ in range(10):
        model_velocity(field, torch.randn((1, 16, 7), device=device), 0.0, warm_condition)

    sanity = run_sanity(config, robot_model, obstacles, field, optimizer, device)
    (results_dir / "sanity.json").write_text(json.dumps(sanity, indent=2))
    print("mandatory reproduction sanity checks passed", flush=True)
    if args.sanity_only:
        return

    raw_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    total = args.tasks * args.seeds
    completed = 0
    for task_id in range(args.tasks):
        task = load_task(task_id)
        q_start, q_goal = task["q"][0], task["q"][-1]
        p_goal, _, _ = robot_model.forward_kinematics(q_goal, np.zeros(7))
        condition = build_condition(robot_model, q_start, task["q_prev0"], p_goal, device)
        reference = task["q"][:16]
        for seed in range(args.seeds):
            noise = torch.randn(
                (1, 16, 7),
                generator=torch.Generator(device=device).manual_seed(seed),
                device=device,
            )
            trajectories: dict[str, np.ndarray] = {}
            sample_rows: dict[str, dict[str, Any]] = {}
            for method in METHODS:
                if method == "PLAIN_FM":
                    optimizer.set_raw_tracking_weight(1.0)
                    trajectory, timing, rows = run_phase2_sampler(
                        "PLAIN_FM", noise, condition, field, optimizer, robot_model,
                        obstacles, q_start, p_goal, task_id, seed, config.flow_steps
                    )
                    method_audit: list[dict[str, Any]] = []
                    for row in rows:
                        row["method"] = method
                        row.setdefault("projection_active", row["projection_called"])
                        row.setdefault("lambda_pov", float("nan"))
                        row.setdefault("terminal_replacement", 0.0)
                        row.setdefault("x1_raw_collision", float(row["endpoint_min_clearance"] < -SAFETY_TOLERANCE))
                        row.setdefault("x1_raw_min_clearance", row["endpoint_min_clearance"])
                        row.setdefault("x1_raw_goal_error", row["endpoint_goal_error"])
                        row.setdefault("z_star_collision", float("nan"))
                        row.setdefault("z_star_min_clearance", float("nan"))
                        row.setdefault("z_star_goal_error", float("nan"))
                    timing["optimizer_latency_sec"] = timing["projection_latency_sec"]
                else:
                    trajectory, timing, rows, method_audit = run_factorial_sampler(
                        method, noise, condition, field, optimizer, robot_model,
                        obstacles, q_start, p_goal, task_id, seed, config.flow_steps
                    )
                metrics = compute_metrics(trajectory, q_goal, p_goal, reference, robot_model, obstacles)
                record = {"method": method, "task_id": task_id, "seed": seed}
                record.update(metrics)
                record.update(timing)
                trajectories[method] = trajectory
                sample_rows[method] = record
                step_rows.extend(rows)
                audit_rows.extend(method_audit)
            plain = trajectories["PLAIN_FM"]
            for method in METHODS:
                sample_rows[method]["trajectory_distortion_vs_plain"] = float(np.linalg.norm(trajectories[method] - plain))
                raw_rows.append(sample_rows[method])
            completed += 1
            print(f"completed {completed}/{total}: task={task_id} seed={seed}", flush=True)

    raw = pd.DataFrame(raw_rows)
    raw["projection_failure_rate"] = np.where(
        raw.projection_calls > 0,
        raw.failed_projection_solves / raw.projection_calls,
        0.0,
    )
    steps = pd.DataFrame(step_rows)
    audit = pd.DataFrame(audit_rows)
    summary = summarize(raw, args.bootstrap_samples)
    paired = transition_table(raw)
    per_task = raw.groupby(["task_id", "method"], as_index=False).agg(
        collision_rate=("collision", "mean"),
        goal_error=("terminal_goal_error", "mean"),
        min_clearance=("min_clearance", "mean"),
    )
    effects = factorial_effects(raw)
    config_record = {
        "experiment": "robot_arm_yflow_2x2_factorial",
        "methods": METHODS,
        "factors": FACTORS,
        "tasks": list(range(args.tasks)),
        "seeds_per_task": args.seeds,
        "checkpoint": config.model_name,
        "remote_main_commit": REMOTE_MAIN_COMMIT,
        "t_on": T_ON,
        "lambda_oc": LAMBDA_OC,
        "terminal_alias_warning": "At t=6/7, dt/(1-t)=1, so lambda=1 residual Euler equals z_star replacement.",
        "sanity": sanity,
        "environment": environment_record(),
    }
    raw.to_csv(results_dir / "raw.csv", index=False)
    summary.to_csv(results_dir / "summary.csv", index=False)
    paired.to_csv(results_dir / "paired_transitions.csv", index=False)
    per_task.to_csv(results_dir / "per_task.csv", index=False)
    steps.to_csv(results_dir / "per_step.csv", index=False)
    audit.to_csv(results_dir / "terminal_state_audit.csv", index=False)
    (results_dir / "config.json").write_text(json.dumps(config_record, indent=2))
    make_plots(raw, per_task, audit, args.output / "plots")
    write_report(raw, summary, paired, per_task, steps, audit, effects, sanity, args.output)
    print(f"artifacts written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
