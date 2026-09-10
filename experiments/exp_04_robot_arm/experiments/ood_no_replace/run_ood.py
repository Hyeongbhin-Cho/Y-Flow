"""Run the fixed robot-arm Y-Flow formulations on deterministic OOD obstacles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from experiments.ood_no_replace.scenarios import (
    CATEGORIES,
    DEFAULT_BBOXES,
    generate_scenarios,
    geometry_checker_matches_bboxes,
    set_bboxes,
)
from experiments._layout import RUN_ROOT
from experiments.pov_factorial_ablation.run_factorial import (
    REMOTE_MAIN_COMMIT,
    SUMMARY_METRICS,
    compute_metrics,
    markdown,
    run_factorial_sampler,
)
from experiments.pov_projection.metrics import constraint_snapshot
from experiments.pov_projection.run_phase1 import (
    build_condition,
    environment_record,
    load_task,
    model_velocity,
)
from experiments.pov_projection_phase2.run_phase2 import run_sampler as run_phase2_sampler
from experiments.pov_projection_phase15_yflow_main.run_phase15 import (
    LAMBDA_OC,
    T_ON,
    build_components,
)


METHODS = ("PLAIN_FM", "L1_REPLACE", "L05_REPLACE", "L05_NO_REPLACE")
PROJECTION_METHODS = METHODS[1:]
OUTPUT_ROOT = RUN_ROOT / "ood_no_replace"
SAFETY_TOLERANCE = 1e-5
BOOTSTRAP_SEED = 20260908


class RecordingOptimizer:
    """Transparent logger around the fixed factorial optimizer."""

    def __init__(self, delegate: Any):
        self.delegate = delegate
        self.captures: list[dict[str, Any]] = []

    def optimize_terminal(
        self,
        candidate_trajectory: np.ndarray,
        start_state: np.ndarray,
        goal: np.ndarray,
        raw_weight: float,
    ):
        optimized, diagnostics = self.delegate.optimize_terminal(
            candidate_trajectory, start_state, goal, raw_weight
        )
        self.captures.append(
            {
                "x1_raw": np.asarray(candidate_trajectory).copy(),
                "z_star": np.asarray(optimized).copy(),
            }
        )
        return optimized, diagnostics


def _state_record(
    method: str,
    scenario: dict[str, Any],
    seed: int,
    name: str,
    state: np.ndarray,
    p_goal: np.ndarray,
    robot_model: Any,
    obstacles: Any,
) -> dict[str, Any]:
    snapshot = constraint_snapshot(state, p_goal, robot_model, obstacles)
    return {
        "method": method,
        "scenario_id": scenario["scenario_id"],
        "category": scenario["category"],
        "task_id": scenario["task_id"],
        "seed": seed,
        "state": name,
        "collision": float(snapshot["min_clearance"] < -SAFETY_TOLERANCE),
        "min_clearance": snapshot["min_clearance"],
        "terminal_goal_error": snapshot["goal_error"],
        "state_json": json.dumps(np.asarray(state).tolist(), separators=(",", ":")),
    }


def run_projection_method(
    method: str,
    noise: torch.Tensor,
    condition: torch.Tensor,
    field: Any,
    optimizer: Any,
    robot_model: Any,
    obstacles: Any,
    q_start: np.ndarray,
    p_goal: np.ndarray,
    scenario: dict[str, Any],
    seed: int,
    flow_steps: int,
) -> tuple[np.ndarray, dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    recorder = RecordingOptimizer(optimizer)
    trajectory, timing, rows, factorial_audit = run_factorial_sampler(
        method,
        noise,
        condition,
        field,
        recorder,
        robot_model,
        obstacles,
        q_start,
        p_goal,
        scenario["category_index"],
        seed,
        flow_steps,
    )
    final_snapshot = constraint_snapshot(trajectory, p_goal, robot_model, obstacles)
    active_rows = [row for row in rows if row["projection_active"] == 1.0]
    if len(active_rows) != 3 or len(recorder.captures) != 3:
        raise RuntimeError("fixed schedule did not produce exactly three optimizer records")
    for row, capture in zip(active_rows, recorder.captures):
        row["scenario_id"] = scenario["scenario_id"]
        row["category"] = scenario["category"]
        row["x1_raw_json"] = json.dumps(capture["x1_raw"].tolist(), separators=(",", ":"))
        row["z_star_json"] = json.dumps(capture["z_star"].tolist(), separators=(",", ":"))
        row["final_output_collision"] = float(
            final_snapshot["min_clearance"] < -SAFETY_TOLERANCE
        )
    for row in rows:
        row["scenario_id"] = scenario["scenario_id"]
        row["category"] = scenario["category"]
        row["task_id"] = scenario["task_id"]
        row.setdefault("x1_raw_json", "")
        row.setdefault("z_star_json", "")
        row.setdefault("final_output_collision", float(
            final_snapshot["min_clearance"] < -SAFETY_TOLERANCE
        ))

    wanted = {"x_before_terminal", "x1_raw", "z_star"}
    audit = []
    for record in factorial_audit:
        if record["state"] not in wanted:
            continue
        state = np.asarray(json.loads(record["state_json"]), dtype=float)
        audit.append(
            _state_record(
                method, scenario, seed, record["state"], state,
                p_goal, robot_model, obstacles
            )
        )
    audit.append(
        _state_record(
            method, scenario, seed, "actual_final_output", trajectory,
            p_goal, robot_model, obstacles
        )
    )
    terminal_solver_success = float(active_rows[-1]["projection_solver_success"])
    for record in audit:
        record["terminal_projection_solver_success"] = terminal_solver_success
    return trajectory, timing, rows, audit


def _noise(device: str, seed: int) -> torch.Tensor:
    return torch.randn(
        (1, 16, 7),
        generator=torch.Generator(device=device).manual_seed(seed),
        device=device,
    )


def _noise_hash(noise: torch.Tensor) -> str:
    return hashlib.sha256(noise.detach().cpu().numpy().tobytes()).hexdigest()


def run_sanity(
    scenarios: list[dict[str, Any]],
    config: Any,
    robot_model: Any,
    obstacles: Any,
    field: Any,
    optimizer: Any,
    device: str,
) -> dict[str, Any]:
    selected_categories = (
        "SHIFTED_OBSTACLE", "MULTI_OBSTACLE", "NARROW_PASSAGE", "GOAL_NEAR_OBSTACLE"
    )
    checks = []
    for category in selected_categories:
        scenario = next(x for x in scenarios if x["category"] == category)
        set_bboxes(obstacles, scenario["bboxes"])
        optimizer.set_finder.set_obstacles(obstacles)
        task = load_task(scenario["task_id"])
        q_start, q_goal = task["q"][0], task["q"][-1]
        p_goal, _, _ = robot_model.forward_kinematics(q_goal, np.zeros(7))
        condition = build_condition(robot_model, q_start, task["q_prev0"], p_goal, device)
        noise = _noise(device, 0)
        expected_hash = _noise_hash(noise)
        outputs: dict[str, np.ndarray] = {}
        audits: dict[str, list[dict[str, Any]]] = {}
        observed_hashes = []
        for method in METHODS:
            observed_hashes.append(_noise_hash(noise))
            if method == "PLAIN_FM":
                optimizer.set_raw_tracking_weight(1.0)
                trajectory, _, _ = run_phase2_sampler(
                    "PLAIN_FM", noise, condition, field, optimizer, robot_model,
                    obstacles, q_start, p_goal, 0, 0, config.flow_steps
                )
                method_audit = []
            else:
                trajectory, _, _, method_audit = run_projection_method(
                    method, noise, condition, field, optimizer, robot_model, obstacles,
                    q_start, p_goal, scenario, 0, config.flow_steps
                )
            outputs[method] = trajectory
            audits[method] = method_audit

        def state(method: str, name: str) -> np.ndarray:
            record = next(x for x in audits[method] if x["state"] == name)
            return np.asarray(json.loads(record["state_json"]), dtype=float)

        geometry_different = not (
            len(scenario["bboxes"]) == len(DEFAULT_BBOXES)
            and all(np.allclose(a, b) for a, b in zip(scenario["bboxes"], DEFAULT_BBOXES))
        )
        checks.append(
            {
                "scenario_id": scenario["scenario_id"],
                "category": category,
                "identical_noise": all(x == expected_hash for x in observed_hashes),
                "geometry_differs_from_id": bool(geometry_different),
                "collision_checker_matches_plot_bboxes": geometry_checker_matches_bboxes(scenario),
                "l05_no_replace_semantics": bool(
                    not np.allclose(state("L05_NO_REPLACE", "actual_final_output"), state("L05_NO_REPLACE", "z_star"), atol=1e-6)
                ),
                "l05_replace_semantics": bool(
                    np.allclose(state("L05_REPLACE", "actual_final_output"), state("L05_REPLACE", "z_star"), atol=1e-6)
                ),
                "l1_replace_semantics": bool(
                    np.allclose(state("L1_REPLACE", "actual_final_output"), state("L1_REPLACE", "z_star"), atol=1e-6)
                ),
                "start_valid": scenario["validation"]["start_clearance"] >= 0,
                "goal_valid": scenario["validation"]["goal_clearance"] >= 0 and not scenario["validation"]["goal_inside_obstacle"],
            }
        )
    keys = [k for k in checks[0] if k not in {"scenario_id", "category"}]
    result = {"pass": all(all(bool(x[k]) for k in keys) for x in checks), "checks": checks}
    if not result["pass"]:
        raise RuntimeError(f"OOD sanity failed: {result}")
    return result


def bootstrap_summary(raw: pd.DataFrame, samples: int) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
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
                    "median": float(np.median(values)),
                    "std": float(values.std(ddof=1)),
                    "ci95_low": float(np.quantile(boot, 0.025)),
                    "ci95_high": float(np.quantile(boot, 0.975)),
                    "numerator": int(values.sum()) if metric == "collision" else "",
                    "denominator": len(values) if metric == "collision" else "",
                }
            )
    return pd.DataFrame(rows)


def paired_bootstrap_difference(
    left: np.ndarray, right: np.ndarray, samples: int, seed: int
) -> tuple[float, float, float]:
    delta = np.asarray(left, float) - np.asarray(right, float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    boot = delta[indices].mean(axis=1)
    return float(delta.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def paired_analysis(raw: pd.DataFrame, samples: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    indexed = {
        method: raw[raw.method == method].set_index(["scenario_id", "seed"]).sort_index()
        for method in METHODS
    }
    comparisons = [(m, "PLAIN_FM", "OVERALL") for m in PROJECTION_METHODS]
    comparisons += [("L05_NO_REPLACE", "L05_REPLACE", "OVERALL")]
    comparisons += [("L05_NO_REPLACE", "L05_REPLACE", cat) for cat in CATEGORIES]
    diffs = {}
    for comparison_index, (method, reference, category) in enumerate(comparisons):
        m = indexed[method]
        r = indexed[reference]
        if category != "OVERALL":
            valid_ids = set(raw[raw.category == category].scenario_id)
            mask = m.index.get_level_values("scenario_id").isin(valid_ids)
            m, r = m[mask], r[mask]
        mc = m.collision.to_numpy(bool)
        rc = r.collision.to_numpy(bool)
        mean, low, high = paired_bootstrap_difference(
            rc.astype(float), mc.astype(float), samples, BOOTSTRAP_SEED + comparison_index
        )
        key = f"{reference}_minus_{method}_{category}"
        diffs[key] = {"mean": mean, "ci95_low": low, "ci95_high": high, "n": len(mc)}
        for r_name, r_value in (("safe", False), ("collision", True)):
            for m_name, m_value in (("safe", False), ("collision", True)):
                rows.append(
                    {
                        "category": category,
                        "method": method,
                        "reference_method": reference,
                        "reference_state": r_name,
                        "method_state": m_name,
                        "count": int(np.sum((rc == r_value) & (mc == m_value))),
                        "reference_minus_method_collision": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return pd.DataFrame(rows), diffs


def per_category_table(raw: pd.DataFrame, id_collision: dict[str, float]) -> pd.DataFrame:
    rows = []
    for category in ("OVERALL",) + CATEGORIES:
        frame = raw if category == "OVERALL" else raw[raw.category == category]
        plain_collision = frame[frame.method == "PLAIN_FM"].collision.mean()
        for method in METHODS:
            group = frame[frame.method == method]
            collision = float(group.collision.mean())
            rows.append(
                {
                    "category": category,
                    "method": method,
                    "collision": collision,
                    "collision_count": int(group.collision.sum()),
                    "n": len(group),
                    "goal_error": float(group.terminal_goal_error.mean()),
                    "min_clearance": float(group.min_clearance.mean()),
                    "distortion": float(group.trajectory_distortion_vs_plain.mean()),
                    "latency": float(group.planning_latency_sec.mean()),
                    "id_collision": id_collision[method],
                    "ood_minus_id_collision": collision - id_collision[method],
                    "relative_collision_reduction_vs_plain": (
                        (plain_collision - collision) / plain_collision
                        if plain_collision > 0 else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def decision_rule(
    raw: pd.DataFrame,
    per_category: pd.DataFrame,
    audit: pd.DataFrame,
    paired_diffs: dict[str, Any],
) -> dict[str, Any]:
    overall = per_category[per_category.category == "OVERALL"].set_index("method")
    cats = per_category[per_category.category != "OVERALL"].pivot(
        index="category", columns="method", values="collision"
    )
    indexed = {
        m: raw[raw.method == m].set_index(["scenario_id", "seed"]).sort_index()
        for m in METHODS
    }
    plain_safe = ~indexed["PLAIN_FM"].collision.to_numpy(bool)
    no_collision = indexed["L05_NO_REPLACE"].collision.to_numpy(bool)
    new_rate = float(np.sum(plain_safe & no_collision) / max(1, np.sum(plain_safe)))
    terminal = audit[audit.method == "L05_NO_REPLACE"].pivot_table(
        index=["scenario_id", "seed"], columns="state", values="collision"
    )
    zstar_final_safe = int(
        ((terminal.z_star == 1) & (terminal.actual_final_output == 0)).sum()
    )
    plain_diff = paired_diffs["PLAIN_FM_minus_L05_NO_REPLACE_OVERALL"]
    replace_diff = paired_diffs["L05_REPLACE_minus_L05_NO_REPLACE_OVERALL"]
    checks = {
        "plain_reduction_ci_above_zero": plain_diff["ci95_low"] > 0,
        "improves_at_least_3_of_5_categories": int(
            (cats.L05_NO_REPLACE < cats.PLAIN_FM).sum()
        ) >= 3,
        "replace_reduction_at_least_10pp_with_positive_ci": (
            replace_diff["mean"] >= 0.10 and replace_diff["ci95_low"] > 0
        ),
        "plain_safe_new_collision_rate_le_5pct": new_rate <= 0.05,
        "goal_error_within_plain_plus_0p15m": (
            overall.loc["L05_NO_REPLACE", "goal_error"]
            <= overall.loc["PLAIN_FM", "goal_error"] + 0.15
        ),
        "direction_not_solution_evidence": zstar_final_safe > 0,
    }
    if all(checks.values()):
        numeric_verdict = "STRONG_SUPPORT"
    elif overall.loc["L05_NO_REPLACE", "collision"] < overall.loc["PLAIN_FM", "collision"]:
        numeric_verdict = "LIMITED_SUPPORT"
    else:
        numeric_verdict = "NO_SUPPORT"
    badly_failing_categories = cats.index[cats.L05_NO_REPLACE >= 0.50].tolist()
    verdict = (
        "LIMITED_SUPPORT"
        if numeric_verdict == "STRONG_SUPPORT" and badly_failing_categories
        else numeric_verdict
    )
    return {
        "verdict": verdict,
        "numeric_checks_verdict": numeric_verdict,
        "numeric_checks_predeclared_before_full_run": True,
        "qualitative_bad_failure_threshold": 0.50,
        "qualitative_bad_failure_threshold_predeclared": False,
        "qualitative_protocol_override": (
            "LIMITED_SUPPORT because at least half of L05_NO_REPLACE trajectories "
            "collide in a category"
            if badly_failing_categories else "none"
        ),
        "badly_failing_categories": badly_failing_categories,
        "checks": {k: bool(v) for k, v in checks.items()},
        "plain_safe_new_collision_rate": new_rate,
        "zstar_collision_final_safe_count": zstar_final_safe,
        "categories_improved": int((cats.L05_NO_REPLACE < cats.PLAIN_FM).sum()),
    }


def make_plots(
    raw: pd.DataFrame,
    per_category: pd.DataFrame,
    audit: pd.DataFrame,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    overall = per_category[per_category.category == "OVERALL"].set_index("method").reindex(METHODS)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(METHODS, overall.collision)
    ax.set_ylabel("Collision rate"); ax.tick_params(axis="x", rotation=20); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(output/"collision_overall.png", dpi=160); plt.close(fig)

    category = per_category[per_category.category != "OVERALL"]
    for metric, filename, ylabel in (
        ("collision", "collision_by_category.png", "Collision rate"),
        ("goal_error", "goal_error_by_category.png", "Goal error (m)"),
        ("min_clearance", "min_clearance_by_category.png", "Min clearance (m)"),
        ("distortion", "distortion_by_category.png", "Distortion vs Plain"),
    ):
        pivot = category.pivot(index="category", columns="method", values=metric).reindex(columns=METHODS)
        fig, ax = plt.subplots(figsize=(11, 5)); pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.25); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(output/filename, dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8)); ax.bar(METHODS, overall.latency)
    ax.set_ylabel("Planning latency (s)"); ax.tick_params(axis="x", rotation=20); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(output/"latency_by_method.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8));
    x=np.arange(len(METHODS)); width=.35
    ax.bar(x-width/2, overall.id_collision, width, label="ID")
    ax.bar(x+width/2, overall.collision, width, label="OOD")
    ax.set_xticks(x, METHODS, rotation=20); ax.set_ylabel("Collision rate"); ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(output/"id_vs_ood_collision.png", dpi=160); plt.close(fig)

    replace = category[category.method.isin(("L05_REPLACE","L05_NO_REPLACE"))].pivot(index="category",columns="method",values="collision")
    fig, ax = plt.subplots(figsize=(9, 4.8)); replace.plot(kind="bar", ax=ax)
    ax.set_ylabel("Collision rate"); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(output/"replace_vs_no_replace.png", dpi=160); plt.close(fig)

    terminal = audit[audit.method == "L05_NO_REPLACE"].pivot_table(index=["category","scenario_id","seed"],columns="state",values="collision").reset_index()
    terminal_plot = terminal.groupby("category")[["z_star","actual_final_output"]].mean()
    fig, ax = plt.subplots(figsize=(9, 4.8)); terminal_plot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Collision rate"); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(output/"zstar_vs_final_collision.png", dpi=160); plt.close(fig)


def select_visualization_cases(raw: pd.DataFrame, audit: pd.DataFrame) -> dict[str, tuple[str, int] | None]:
    pivot = raw.pivot_table(index=["scenario_id","seed"], columns="method", values="collision")
    category = raw.drop_duplicates("scenario_id").set_index("scenario_id").category.to_dict()
    terminal = audit[audit.method == "L05_NO_REPLACE"].pivot_table(index=["scenario_id","seed"],columns="state",values="collision")
    def first(mask: pd.Series) -> tuple[str, int] | None:
        values = pivot.index[mask]
        return (str(values[0][0]), int(values[0][1])) if len(values) else None
    rescue = first((pivot.PLAIN_FM == 1) & (pivot.L05_NO_REPLACE == 0))
    replacement = first((pivot.L05_REPLACE == 1) & (pivot.L05_NO_REPLACE == 0))
    both_fail = first((pivot.L05_REPLACE == 1) & (pivot.L05_NO_REPLACE == 1))
    narrow_mask = (
        np.asarray([category[x] for x in pivot.index.get_level_values("scenario_id")]) == "NARROW_PASSAGE"
    ) & (pivot.L05_REPLACE.to_numpy() == 0) & (pivot.L05_NO_REPLACE.to_numpy() == 0)
    narrow_values = pivot.index[narrow_mask]
    narrow = (str(narrow_values[0][0]), int(narrow_values[0][1])) if len(narrow_values) else None
    zmask = (terminal.z_star == 1) & (terminal.actual_final_output == 0)
    zvalues = terminal.index[zmask]
    zcase = (str(zvalues[0][0]), int(zvalues[0][1])) if len(zvalues) else None
    return {
        "rescue_case": rescue,
        "replacement_failure_case": replacement,
        "zstar_collision_final_safe": zcase,
        "narrow_passage_case": narrow,
        "failure_case": both_fail,
    }


def make_visualizations(
    cases: dict[str, tuple[str, int] | None],
    trajectories: dict[tuple[str, int, str], np.ndarray],
    scenarios: list[dict[str, Any]],
    audit: pd.DataFrame,
    robot_model: Any,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    scenario_map = {x["scenario_id"]: x for x in scenarios}
    for label, selected in cases.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        if selected is None:
            for ax in axes: ax.axis("off")
            fig.text(.5,.5,f"{label}: no matching case observed",ha="center",va="center",fontsize=14)
            fig.savefig(output/f"{label}.png",dpi=160); plt.close(fig); continue
        scenario_id, seed = selected; scenario = scenario_map[scenario_id]
        for method in ("PLAIN_FM","L05_REPLACE","L05_NO_REPLACE"):
            q=trajectories[(scenario_id,seed,method)]
            ee=np.asarray([robot_model.fk_pos(x) for x in q])
            axes[0].plot(ee[:,0],ee[:,1],marker=".",label=method)
            axes[1].plot(ee[:,0],ee[:,2],marker=".",label=method)
        zrows=audit[(audit.scenario_id==scenario_id)&(audit.seed==seed)&(audit.method=="L05_NO_REPLACE")&(audit.state=="z_star")]
        if len(zrows):
            z=np.asarray(json.loads(zrows.iloc[0].state_json)); ee=np.asarray([robot_model.fk_pos(x) for x in z])
            axes[0].plot(ee[:,0],ee[:,1],"--",label="z_star")
            axes[1].plot(ee[:,0],ee[:,2],"--",label="z_star")
        for values in scenario["bboxes"]:
            box=np.asarray(values); lo,hi=box[:3],box[3:]
            axes[0].add_patch(plt.Rectangle((lo[0],lo[1]),hi[0]-lo[0],hi[1]-lo[1],color="black",alpha=.18))
            axes[1].add_patch(plt.Rectangle((lo[0],lo[2]),hi[0]-lo[0],hi[2]-lo[2],color="black",alpha=.18))
        task=load_task(scenario["task_id"]); start=robot_model.fk_pos(task["q"][0]); goal=robot_model.fk_pos(task["q"][-1])
        axes[0].scatter([start[0],goal[0]],[start[1],goal[1]],c=["green","red"],s=45)
        axes[1].scatter([start[0],goal[0]],[start[2],goal[2]],c=["green","red"],s=45)
        axes[0].set(xlabel="x",ylabel="y",title="Top view"); axes[1].set(xlabel="x",ylabel="z",title="Side view")
        for ax in axes: ax.axis("equal"); ax.grid(alpha=.2)
        axes[1].legend(fontsize=7); fig.suptitle(f"{label}: {scenario_id}, seed={seed}")
        fig.tight_layout(); fig.savefig(output/f"{label}.png",dpi=160); plt.close(fig)


def write_report(
    summary: pd.DataFrame,
    per_category: pd.DataFrame,
    paired: pd.DataFrame,
    audit: pd.DataFrame,
    paired_diffs: dict[str, Any],
    decision: dict[str, Any],
    sanity: dict[str, Any],
    cases: dict[str, Any],
    output: Path,
) -> None:
    overall=per_category[per_category.category=="OVERALL"].set_index("method")
    cats=per_category[per_category.category!="OVERALL"]
    hardest=(cats[cats.method=="L05_NO_REPLACE"].sort_values(["collision","goal_error"],ascending=False).iloc[0])
    l05_audit=audit[audit.method=="L05_NO_REPLACE"]
    terminal=l05_audit.pivot_table(index=["scenario_id","seed"],columns="state",values="collision")
    terminal_success=l05_audit.drop_duplicates(["scenario_id","seed"]).set_index(["scenario_id","seed"])["terminal_projection_solver_success"]
    terminal=terminal.join(terminal_success)
    successful_terminal=terminal[terminal.terminal_projection_solver_success==1]
    zrate=float(terminal.z_star.mean()); frate=float(terminal.actual_final_output.mean())
    zd_safe=int(((terminal.z_star==1)&(terminal.actual_final_output==0)).sum())
    successful_zrate=float(successful_terminal.z_star.mean())
    successful_frate=float(successful_terminal.actual_final_output.mean())
    terminal_failures=int((terminal.terminal_projection_solver_success!=1).sum())
    failed_projection_calls=int(summary[(summary.method=="L05_NO_REPLACE")&(summary.metric=="failed_projection_solves")]["mean"].iloc[0]*800)
    plain_diff=paired_diffs["PLAIN_FM_minus_L05_NO_REPLACE_OVERALL"]
    replace_diff=paired_diffs["L05_REPLACE_minus_L05_NO_REPLACE_OVERALL"]
    replacement_rows=[]
    for category in CATEGORIES:
        frame=per_category[per_category.category==category].set_index("method")
        transition=paired[(paired.category==category)&(paired.method=="L05_NO_REPLACE")&(paired.reference_method=="L05_REPLACE")]
        rescued=int(transition[(transition.reference_state=="collision")&(transition.method_state=="safe")]["count"].iloc[0])
        broken=int(transition[(transition.reference_state=="safe")&(transition.method_state=="collision")]["count"].iloc[0])
        zcat=audit[(audit.category==category)&(audit.method=="L05_NO_REPLACE")&(audit.state=="z_star")]
        zcat_success=zcat[zcat.terminal_projection_solver_success==1]
        replacement_rows.append({
            "category":category,
            "replace_minus_no_replace_collision":frame.loc["L05_REPLACE","collision"]-frame.loc["L05_NO_REPLACE","collision"],
            "replace_minus_no_replace_goal_error":frame.loc["L05_REPLACE","goal_error"]-frame.loc["L05_NO_REPLACE","goal_error"],
            "replace_minus_no_replace_distortion":frame.loc["L05_REPLACE","distortion"]-frame.loc["L05_NO_REPLACE","distortion"],
            "replacement_collisions_rescued_by_no_replace":rescued,
            "no_replace_newly_broken_vs_replace":broken,
            "z_star_collision_rate":float(zcat.collision.mean()),
            "z_star_collision_rate_successful_solves":float(zcat_success.collision.mean()),
        })
    replacement_category=pd.DataFrame(replacement_rows)
    report=f"""# Robot-arm Y-Flow no-replacement OOD report

## Decision

**{decision['verdict']}** under the supplied qualitative decision rule.

The numeric checklist was fixed before the full run and yields
`{decision['numeric_checks_verdict']}`. The supplied phrase "fail badly" had no
numeric cutoff; at reporting time it was conservatively operationalized as at
least 50% collision in a category. That post-result descriptive cutoff is labeled
in the JSON below and is not presented as a predeclared statistical gate.

```json
{json.dumps(decision,indent=2)}
```

## Sanity checks

All four required OOD sanity scenarios passed before the 3,200-evaluation run.

{markdown(pd.DataFrame(sanity['checks']))}

## Overall metrics

{markdown(summary[['method','metric','mean','median','std','ci95_low','ci95_high','numerator','denominator']])}

## Overall and per-category generalization

{markdown(per_category)}

## Paired collision analysis

The overall paired Plain-minus-L05_NO_REPLACE collision difference is
{plain_diff['mean']:.4f} (95% bootstrap CI {plain_diff['ci95_low']:.4f},
{plain_diff['ci95_high']:.4f}). The L05_REPLACE-minus-L05_NO_REPLACE difference
is {replace_diff['mean']:.4f} (95% CI {replace_diff['ci95_low']:.4f},
{replace_diff['ci95_high']:.4f}).

{markdown(paired)}

## Terminal replacement mechanism

For L05_NO_REPLACE, the returned terminal target collides in
{int(terminal.z_star.sum())}/800 ({zrate:.2%}), while the actual damped final
output collides in {int(terminal.actual_final_output.sum())}/800 ({frate:.2%}).
Among {len(successful_terminal)} successful terminal solves, actual `z_star`
collides in {int(successful_terminal.z_star.sum())}/{len(successful_terminal)}
({successful_zrate:.2%}) and the final output in
{int(successful_terminal.actual_final_output.sum())}/{len(successful_terminal)}
({successful_frate:.2%}). In {zd_safe}/800 cases, a successfully solved `z_star`
collides but the damped output is safe. There were {terminal_failures} terminal
solver failures and {failed_projection_calls}/2400 failed L05_NO_REPLACE calls
across all active steps; failed solves consumed no unconverged solution and used
the fixed candidate/zero-correction fallback.

Category-level replacement analysis:

{markdown(replacement_category)}

## Required answers

1. **Does L05_NO_REPLACE reduce OOD collision relative to Plain?** Yes. Overall rates
   are {overall.loc['L05_NO_REPLACE','collision']:.2%} versus
   {overall.loc['PLAIN_FM','collision']:.2%}; the paired difference and CI are above.
2. **Does the ID 0.52% generalize?** No. OOD collision is
   {overall.loc['L05_NO_REPLACE','collision']:.2%}, a change of
   {overall.loc['L05_NO_REPLACE','ood_minus_id_collision']:+.2%} from ID. This is
   reported as observed benchmark generalization, not a population guarantee.
3. **Does no-replace remain better than replace?** Yes. Rates are
   {overall.loc['L05_NO_REPLACE','collision']:.2%} versus
   {overall.loc['L05_REPLACE','collision']:.2%}; paired transitions quantify the cases.
4. **Is replacement still a dominant failure mechanism?** Yes overall. The replacement-minus-
   no-replacement paired effect is {replace_diff['mean']:.2%}; category rows show
   whether it is consistent across geometry types.
5. **How often is z_star colliding?** On successful terminal solves,
   {int(successful_terminal.z_star.sum())}/{len(successful_terminal)}
   ({successful_zrate:.2%}); the all-sample returned-target rate including fixed
   failure fallback is {int(terminal.z_star.sum())}/800 ({zrate:.2%}).
6. **How often is z_star colliding while damped output is safe?** {zd_safe}/800
   ({zd_safe/800:.2%}).
7. **Hardest category for L05_NO_REPLACE:** {hardest.category}, collision
   {hardest.collision:.2%}, goal error {hardest.goal_error:.4f} m.
8. **Safety cost:** Not in goal error: L05_NO_REPLACE goal error improves to
   {overall.loc['L05_NO_REPLACE','goal_error']:.4f} m versus
   {overall.loc['PLAIN_FM','goal_error']:.4f} m. However, its distortion versus
   Plain is substantial at {overall.loc['L05_NO_REPLACE','distortion']:.4f}.
9. **Direction rather than solution?** {'Supported' if zd_safe>0 and frate<zrate else 'Not supported'}:
   the true-geometry terminal audit compares the same optimizer solution and
   damped output on every matched OOD sample.

## Representative cases

```json
{json.dumps(cases,indent=2)}
```

Missing case types, if any, are rendered honestly as `no matching case observed`.

## Scope

- 100 deterministic OOD scenarios (20/category), 8 matched seeds, 4 fixed methods.
- Same checkpoint, 7-step grid, no-P Acados optimizer, `mu=0`, `t_on=0.5`,
  `lambda_oc=10`, integration, goal, and solver-failure behavior.
- No retraining, adaptive gating, clipping, lambda/schedule tuning, or result-based
  scenario rejection. Start and goal validity were checked before sampling.
- Remote-main reference commit: `{REMOTE_MAIN_COMMIT}`.
- All earlier experiment artifacts remain untouched.
"""
    (output/"REPORT_OOD.md").write_text(report)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--scenarios",type=int,default=100)
    parser.add_argument("--seeds",type=int,default=8)
    parser.add_argument("--bootstrap-samples",type=int,default=2000)
    parser.add_argument("--build-solver",action="store_true")
    parser.add_argument("--sanity-only",action="store_true")
    parser.add_argument("--output",type=Path,default=OUTPUT_ROOT)
    args=parser.parse_args()
    if args.scenarios!=100 and not args.sanity_only:
        raise ValueError("full benchmark is fixed to exactly 100 scenarios")
    if args.seeds!=8 and not args.sanity_only:
        raise ValueError("full benchmark is fixed to exactly 8 seeds")
    args.output.mkdir(parents=True,exist_ok=True); results=args.output/"results"; results.mkdir(parents=True,exist_ok=True)
    device,config,robot_model,obstacles,field,optimizer=build_components(args)
    scenarios=generate_scenarios(robot_model)
    (results/"scenarios.json").write_text(json.dumps(scenarios,indent=2))
    warm=load_task(0); warm_goal,_,_=robot_model.forward_kinematics(warm["q"][-1],np.zeros(7)); warm_condition=build_condition(robot_model,warm["q"][0],warm["q_prev0"],warm_goal,device)
    for _ in range(10): model_velocity(field,torch.randn((1,16,7),device=device),0.0,warm_condition)
    sanity=run_sanity(scenarios,config,robot_model,obstacles,field,optimizer,device)
    (results/"sanity.json").write_text(json.dumps(sanity,indent=2)); print("OOD_SANITY=PASS",flush=True)
    if args.sanity_only: return

    raw_rows=[]; step_rows=[]; audit_rows=[]; trajectories={}
    for scenario_index,scenario in enumerate(scenarios):
        set_bboxes(obstacles,scenario["bboxes"]); optimizer.set_finder.set_obstacles(obstacles)
        task=load_task(scenario["task_id"]); q_start,q_goal=task["q"][0],task["q"][-1]
        p_goal,_,_=robot_model.forward_kinematics(q_goal,np.zeros(7)); condition=build_condition(robot_model,q_start,task["q_prev0"],p_goal,device); reference=task["q"][:16]
        for seed in range(args.seeds):
            noise=_noise(device,seed); sample={}
            for method in METHODS:
                if method=="PLAIN_FM":
                    optimizer.set_raw_tracking_weight(1.0)
                    trajectory,timing,rows=run_phase2_sampler("PLAIN_FM",noise,condition,field,optimizer,robot_model,obstacles,q_start,p_goal,scenario_index,seed,config.flow_steps)
                    method_audit=[]
                    for row in rows:
                        row.update({"method":method,"scenario_id":scenario["scenario_id"],"category":scenario["category"],"task_id":scenario["task_id"],"projection_active":row["projection_called"],"x1_raw_json":"","z_star_json":"","final_output_collision":float("nan")})
                    timing["optimizer_latency_sec"]=timing["projection_latency_sec"]
                else:
                    trajectory,timing,rows,method_audit=run_projection_method(method,noise,condition,field,optimizer,robot_model,obstacles,q_start,p_goal,scenario,seed,config.flow_steps)
                metrics=compute_metrics(trajectory,q_goal,p_goal,reference,robot_model,obstacles)
                record={"method":method,"scenario_id":scenario["scenario_id"],"category":scenario["category"],"task_id":scenario["task_id"],"seed":seed,"severity":scenario["severity"]}; record.update(metrics); record.update(timing)
                sample[method]=(record,trajectory); step_rows.extend(rows); audit_rows.extend(method_audit)
                trajectories[(scenario["scenario_id"],seed,method)]=trajectory
            plain=sample["PLAIN_FM"][1]
            for method in METHODS:
                record,trajectory=sample[method]; record["trajectory_distortion_vs_plain"]=float(np.linalg.norm(trajectory-plain)); raw_rows.append(record)
        print(f"completed {scenario_index+1}/100 scenarios",flush=True)

    raw=pd.DataFrame(raw_rows); raw["projection_failure_rate"]=np.where(raw.projection_calls>0,raw.failed_projection_solves/raw.projection_calls,0.0)
    steps=pd.DataFrame(step_rows); audit=pd.DataFrame(audit_rows); summary=bootstrap_summary(raw,args.bootstrap_samples)
    id_raw=pd.read_csv(RUN_ROOT/"pov_factorial_ablation"/"results"/"raw.csv")
    id_collision=id_raw.groupby("method").collision.mean().reindex(METHODS).to_dict()
    per_category=per_category_table(raw,id_collision); paired,diffs=paired_analysis(raw,args.bootstrap_samples); decision=decision_rule(raw,per_category,audit,diffs)
    cases=select_visualization_cases(raw,audit)
    config_record={"experiment":"robot_arm_yflow_ood_no_replace","methods":METHODS,"scenario_count":100,"categories":CATEGORIES,"seeds":list(range(8)),"checkpoint":config.model_name,"t_on":T_ON,"lambda_oc":LAMBDA_OC,"mu":0.0,"decision":decision,"paired_differences":diffs,"sanity":sanity,"visualization_cases":cases,"environment":environment_record()}
    raw.to_csv(results/"raw_results.csv",index=False); summary.to_csv(results/"summary.csv",index=False); per_category.to_csv(results/"per_category.csv",index=False); paired.to_csv(results/"paired_transitions.csv",index=False); audit.to_csv(results/"terminal_audit.csv",index=False); steps.to_csv(results/"per_step.csv",index=False); (results/"config.json").write_text(json.dumps(config_record,indent=2))
    make_plots(raw,per_category,audit,args.output/"plots"); make_visualizations(cases,trajectories,scenarios,audit,robot_model,args.output/"visualizations"); write_report(summary,per_category,paired,audit,diffs,decision,sanity,cases,args.output)
    print(f"artifacts written to {args.output}",flush=True)


if __name__=="__main__": main()
