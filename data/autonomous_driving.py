# -*- coding: utf-8 -*-
# data/autonomous_driving.py
"""Argoverse 2 focal-vehicle trajectories and kinematic constraints for Exp-05."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from data.base import BaseConstraint, DataBundle, register_dataset
from utils.paths import ROOT


@dataclass
class AutonomousDrivingMeta:
    dataset: str
    sample_hz: float
    history_steps: int
    future_steps: int
    difference_window: int
    v_max: float
    a_max: float
    n_train: int
    n_eval: int
    skipped_train: int
    skipped_eval: int
    coordinate_frame: str
    seed: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    train_scenario_ids: tuple[str, ...]
    eval_scenario_ids: tuple[str, ...]


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories: torch.Tensor):
        self.trajectories = trajectories

    def __len__(self) -> int:
        return int(self.trajectories.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.trajectories[idx]


def _reshape_trajectories(
    p: np.ndarray | torch.Tensor, future_steps: int
) -> np.ndarray | torch.Tensor:
    if p.shape[-1] != 2 * future_steps:
        raise ValueError(
            f"expected final dimension {2 * future_steps}, got {p.shape[-1]}"
        )
    return p.reshape(*p.shape[:-1], future_steps, 2)


class AutonomousDrivingConstraint(BaseConstraint):
    """Speed and acceleration envelope for flattened ego-centric trajectories."""

    def __init__(self, meta: AutonomousDrivingMeta):
        if meta.difference_window < 1:
            raise ValueError("difference_window must be positive")
        if 2 * meta.difference_window > meta.future_steps:
            raise ValueError("future_steps must be at least twice difference_window")
        self.meta = meta

    def _kinematics(
        self, p: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        traj = _reshape_trajectories(p, self.meta.future_steps)
        if isinstance(traj, torch.Tensor):
            origin = torch.zeros(
                *traj.shape[:-2], 1, 2, device=traj.device, dtype=traj.dtype
            )
            points = torch.cat([origin, traj], dim=-2)
        else:
            origin = np.zeros((*traj.shape[:-2], 1, 2), dtype=traj.dtype)
            points = np.concatenate([origin, traj], axis=-2)

        window = int(self.meta.difference_window)
        window_dt = window / float(self.meta.sample_hz)
        velocity = (points[..., window:, :] - points[..., :-window, :]) / window_dt
        acceleration = (
            velocity[..., window:, :] - velocity[..., :-window, :]
        ) / window_dt
        return velocity, acceleration

    def h(self, p: np.ndarray | torch.Tensor) -> dict[str, np.ndarray | torch.Tensor]:
        velocity, acceleration = self._kinematics(p)
        if isinstance(p, torch.Tensor):
            max_speed = torch.linalg.vector_norm(velocity, dim=-1).amax(dim=-1)
            max_accel = torch.linalg.vector_norm(acceleration, dim=-1).amax(dim=-1)
        else:
            max_speed = np.linalg.norm(velocity, axis=-1).max(axis=-1)
            max_accel = np.linalg.norm(acceleration, axis=-1).max(axis=-1)
        return {
            "speed": max_speed - float(self.meta.v_max),
            "accel": max_accel - float(self.meta.a_max),
        }

    def cost(self, p: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        values = self.h(p)
        if isinstance(p, torch.Tensor):
            return 0.5 * sum(torch.clamp_min(v, 0.0).square() for v in values.values())
        return 0.5 * sum(np.maximum(v, 0.0) ** 2 for v in values.values())

    def project_feasible(
        self, p: torch.Tensor | np.ndarray, buffer: float = 1e-4
    ) -> torch.Tensor | np.ndarray:
        values = self.h(p)
        speed_limit = max(float(self.meta.v_max) - float(buffer), 1e-8)
        accel_limit = max(float(self.meta.a_max) - float(buffer), 1e-8)
        max_speed = values["speed"] + float(self.meta.v_max)
        max_accel = values["accel"] + float(self.meta.a_max)

        if isinstance(p, torch.Tensor):
            one = torch.ones_like(max_speed)
            scale = torch.minimum(one, speed_limit / max_speed.clamp_min(1e-12))
            scale = torch.minimum(scale, accel_limit / max_accel.clamp_min(1e-12))
            return p * scale.unsqueeze(-1)

        one = np.ones_like(max_speed)
        scale = np.minimum(one, speed_limit / np.maximum(max_speed, 1e-12))
        scale = np.minimum(scale, accel_limit / np.maximum(max_accel, 1e-12))
        return p * np.expand_dims(scale, axis=-1)

def _resolve_path(raw: Any) -> Path:
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else ROOT / path


def _extract_focal_trajectory(
    path: Path, history_steps: int, future_steps: int
) -> tuple[np.ndarray, str] | None:
    frame = pd.read_parquet(path)
    required = {
        "observed",
        "track_id",
        "object_type",
        "timestep",
        "position_x",
        "position_y",
        "heading",
        "focal_track_id",
        "scenario_id",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    focal_id = frame["focal_track_id"].iloc[0]
    focal = frame[frame["track_id"] == focal_id].sort_values("timestep")
    observed = focal[focal["observed"]]
    future = focal[~focal["observed"]]
    expected = history_steps + future_steps
    complete = (
        len(focal) == expected
        and len(observed) == history_steps
        and len(future) == future_steps
        and int(focal["timestep"].min()) == 0
        and int(focal["timestep"].max()) == expected - 1
        and str(focal["object_type"].iloc[0]).lower() == "vehicle"
    )
    if not complete:
        return None

    origin = observed[["position_x", "position_y"]].iloc[-1].to_numpy(dtype=np.float64)
    heading = float(observed["heading"].iloc[-1])
    delta = future[["position_x", "position_y"]].to_numpy(dtype=np.float64) - origin
    cosine, sine = np.cos(heading), np.sin(heading)
    local = np.column_stack(
        [
            cosine * delta[:, 0] + sine * delta[:, 1],
            -sine * delta[:, 0] + cosine * delta[:, 1],
        ]
    )
    scenario_id = str(frame["scenario_id"].iloc[0])
    return local.astype(np.float32).reshape(-1), scenario_id


def _load_split(
    split_dir: Path,
    history_steps: int,
    future_steps: int,
    limit: int,
) -> tuple[np.ndarray, tuple[str, ...], int]:
    paths = sorted(split_dir.rglob("scenario_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no scenario parquet files found below {split_dir}")

    trajectories: list[np.ndarray] = []
    scenario_ids: list[str] = []
    skipped = 0
    for path in paths:
        item = _extract_focal_trajectory(path, history_steps, future_steps)
        if item is None:
            skipped += 1
            continue
        trajectory, scenario_id = item
        trajectories.append(trajectory)
        scenario_ids.append(scenario_id)
        if limit > 0 and len(trajectories) >= limit:
            break

    if not trajectories:
        raise ValueError(f"no complete focal-vehicle trajectories found below {split_dir}")
    return np.stack(trajectories), tuple(scenario_ids), skipped


def _bundle_from_arrays(
    train_raw: np.ndarray,
    eval_raw: np.ndarray,
    meta: AutonomousDrivingMeta,
) -> DataBundle:
    mean = np.asarray(meta.mean, dtype=np.float32)
    std = np.asarray(meta.std, dtype=np.float32)
    train_z = (np.asarray(train_raw, dtype=np.float32) - mean) / std
    eval_z = (np.asarray(eval_raw, dtype=np.float32) - mean) / std
    return DataBundle(
        train=TrajectoryDataset(torch.from_numpy(train_z)),
        train_raw=np.asarray(train_raw, dtype=np.float32),
        eval_raw=np.asarray(eval_raw, dtype=np.float32),
        eval_z=torch.from_numpy(eval_z),
        mean=torch.from_numpy(mean),
        std=torch.from_numpy(std),
        constraint=AutonomousDrivingConstraint(meta),
        meta=meta,
        meta_dict=asdict(meta),
    )


def _meta_from_dict(payload: dict[str, Any]) -> AutonomousDrivingMeta:
    data = dict(payload)
    for name in ("mean", "std", "train_scenario_ids", "eval_scenario_ids"):
        data[name] = tuple(data[name])
    return AutonomousDrivingMeta(**data)


def save_autonomous_driving(
    cache_dir: str | Path,
    train_raw: np.ndarray,
    eval_raw: np.ndarray,
    meta: AutonomousDrivingMeta,
) -> None:
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "train.npy", np.asarray(train_raw, dtype=np.float32))
    np.save(path / "eval.npy", np.asarray(eval_raw, dtype=np.float32))
    (path / "meta.json").write_text(json.dumps(asdict(meta), indent=2))


def load_autonomous_driving(cache_dir: str | Path) -> DataBundle:
    path = Path(cache_dir)
    train_raw = np.load(path / "train.npy")
    eval_raw = np.load(path / "eval.npy")
    meta = _meta_from_dict(json.loads((path / "meta.json").read_text()))
    return _bundle_from_arrays(train_raw, eval_raw, meta)


@register_dataset("autonomous_driving")
def build_autonomous_driving(cfg: DictConfig) -> DataBundle:
    cache_dir = _resolve_path(cfg.data.cache_dir)
    regenerate = bool(cfg.data.get("regenerate", False))
    if (cache_dir / "meta.json").is_file() and not regenerate:
        return load_autonomous_driving(cache_dir)

    raw_dir = _resolve_path(cfg.data.raw_dir)
    history_steps = int(cfg.data.history_steps)
    future_steps = int(cfg.data.future_steps)
    train_raw, train_ids, skipped_train = _load_split(
        raw_dir / "train", history_steps, future_steps, int(cfg.data.n_train)
    )
    eval_raw, eval_ids, skipped_eval = _load_split(
        raw_dir / "val", history_steps, future_steps, int(cfg.data.n_eval)
    )
    mean = train_raw.mean(axis=0)
    std = train_raw.std(axis=0).clip(min=1e-6)
    meta = AutonomousDrivingMeta(
        dataset="Argoverse 2 Motion Forecasting",
        sample_hz=float(cfg.data.sample_hz),
        history_steps=history_steps,
        future_steps=future_steps,
        difference_window=int(cfg.data.difference_window),
        v_max=float(cfg.data.v_max),
        a_max=float(cfg.data.a_max),
        n_train=int(train_raw.shape[0]),
        n_eval=int(eval_raw.shape[0]),
        skipped_train=skipped_train,
        skipped_eval=skipped_eval,
        coordinate_frame="focal-agent ego-centric at final observed timestep",
        seed=int(cfg.seed),
        mean=tuple(float(x) for x in mean),
        std=tuple(float(x) for x in std),
        train_scenario_ids=train_ids,
        eval_scenario_ids=eval_ids,
    )
    save_autonomous_driving(cache_dir, train_raw, eval_raw, meta)
    return _bundle_from_arrays(train_raw, eval_raw, meta)
