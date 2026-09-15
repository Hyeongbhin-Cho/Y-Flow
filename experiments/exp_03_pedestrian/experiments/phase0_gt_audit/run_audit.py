# -*- coding: utf-8 -*-
# experiments/exp_03_pedestrian/experiments/phase0_gt_audit/run_audit.py
"""Phase 0: kinematic limits from train GT and GT violation rates on every split."""

from __future__ import annotations

import argparse
import csv
import json
import pickle

import numpy as np

from experiments._layout import DATASET_ROOT, RUN_ROOT, SUBSETS


PAST = 8
FUTURE = 12
DT = 0.4
SPLITS = ("train", "val", "test")
CALIBRATIONS = ("traj", "elem")


def load_traj(subset: str, split: str) -> np.ndarray:
    with (DATASET_ROOT / subset / f"{subset}_{split}.pkl").open("rb") as f:
        return np.asarray(pickle.load(f)["traj"], dtype=np.float64)


def kinematics(traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    anchor = traj[:, PAST - 2 : PAST + FUTURE]
    vel = np.diff(anchor, axis=1) / DT
    speed = np.linalg.norm(vel[:, 1:], axis=-1)
    acc = np.linalg.norm(np.diff(vel, axis=1), axis=-1) / DT
    return speed, acc


def violation_stats(values: np.ndarray, limit: float) -> dict[str, float]:
    excess = values - limit
    return {
        "elem_rate": float((excess > 0.0).mean()),
        "traj_rate": float((excess > 0.0).any(axis=1).mean()),
        "max_excess": float(max(excess.max(), 0.0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantiles", type=float, nargs="+", default=[99.0, 99.5, 99.9])
    parser.add_argument("--out_dir", type=str, default=str(RUN_ROOT / "phase0_gt_audit"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = RUN_ROOT.__class__(args.out_dir)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)

    thresholds: dict[str, dict] = {}
    rows: list[dict] = []
    for subset in SUBSETS:
        kin = {split: kinematics(load_traj(subset, split)) for split in SPLITS}
        train_speed, train_acc = kin["train"]
        pools = {
            "elem": (train_speed, train_acc),
            "traj": (train_speed.max(axis=1), train_acc.max(axis=1)),
        }
        thresholds[subset] = {
            f"{mode}_q{q:g}": {
                "v_max": float(np.percentile(pools[mode][0], q)),
                "a_max": float(np.percentile(pools[mode][1], q)),
            }
            for mode in CALIBRATIONS
            for q in args.quantiles
        }
        for mode in CALIBRATIONS:
            for q in args.quantiles:
                lim = thresholds[subset][f"{mode}_q{q:g}"]
                for split in SPLITS:
                    speed, acc = kin[split]
                    spd = violation_stats(speed, lim["v_max"])
                    acc_s = violation_stats(acc, lim["a_max"])
                    any_rate = float(
                        ((speed > lim["v_max"]) | (acc > lim["a_max"])).any(axis=1).mean()
                    )
                    rows.append(
                        {
                            "subset": subset,
                            "calibration": mode,
                            "quantile": q,
                            "split": split,
                            "n_traj": int(speed.shape[0]),
                            "v_max": lim["v_max"],
                            "a_max": lim["a_max"],
                            "speed_elem_rate": spd["elem_rate"],
                            "speed_traj_rate": spd["traj_rate"],
                            "speed_max_excess": spd["max_excess"],
                            "acc_elem_rate": acc_s["elem_rate"],
                            "acc_traj_rate": acc_s["traj_rate"],
                            "acc_max_excess": acc_s["max_excess"],
                            "any_traj_rate": any_rate,
                        }
                    )

        for split in SPLITS:
            speed, acc = kin[split]
            thresholds[subset].setdefault("distribution", {})[split] = {
                "speed_p50": float(np.percentile(speed, 50)),
                "speed_p99": float(np.percentile(speed, 99)),
                "speed_max": float(speed.max()),
                "acc_p50": float(np.percentile(acc, 50)),
                "acc_p99": float(np.percentile(acc, 99)),
                "acc_max": float(acc.max()),
                "speed_zero_frac": float((speed < 1e-6).mean()),
                "acc_below_0p01_frac": float((acc < 1e-2).mean()),
            }

    meta = {"dt": DT, "past_frames": PAST, "future_frames": FUTURE,
            "speed_terms": FUTURE, "acc_terms": FUTURE,
            "velocity_anchor": "v0 = (p_obs[-1] - p_obs[-2]) / dt",
            "calibration": {"traj": "quantile of per-trajectory max", "elem": "quantile of all terms"}}
    (out_dir / "results" / "thresholds.json").write_text(
        json.dumps({"meta": meta, "subsets": thresholds}, indent=2)
    )
    with (out_dir / "results" / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_dir / 'results'}")


if __name__ == "__main__":
    main()
