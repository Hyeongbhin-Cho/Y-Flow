# -*- coding: utf-8 -*-
# data/swiss_roll.py
"""2D Swiss roll points for Exp-01. Not a pixel grid, not sklearn's 3D roll."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from utils.paths import ROOT


@dataclass
class SwissRollMeta:
    a: float
    u_min: float
    u_max: float
    n_turns: float
    sigma_obs: float
    n_train: int
    n_eval: int
    margin: float
    tau: float
    rho_min: float
    R: float
    seed: int
    mean: tuple[float, float]
    std: tuple[float, float]
    arc_length: float


def spiral(u: np.ndarray, a: float) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    return np.stack([a * u * np.cos(u), a * u * np.sin(u)], axis=-1)


def arc_length(a: float, u_min: float, u_max: float) -> float:
    def F(u: float) -> float:
        return 0.5 * a * (u * math.sqrt(1.0 + u * u) + math.asinh(u))

    return F(u_max) - F(u_min)


def sample_points(
    n: int,
    a: float,
    u_min: float,
    u_max: float,
    sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    u = rng.uniform(u_min, u_max, size=n)
    p = spiral(u, a) + sigma * rng.normal(size=(n, 2))
    return p.astype(np.float32), u.astype(np.float32)


def nearest_u(p: np.ndarray, a: float, u_min: float, u_max: float) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    phi = np.arctan2(p[:, 1], p[:, 0])
    k0 = int(math.floor(u_min / (2.0 * math.pi))) - 1
    k1 = int(math.ceil(u_max / (2.0 * math.pi))) + 1
    ks = np.arange(k0, k1 + 1)
    u = phi[:, None] + 2.0 * math.pi * ks[None, :]
    u = np.concatenate(
        [u, np.full((p.shape[0], 1), u_min), np.full((p.shape[0], 1), u_max)],
        axis=1,
    )
    u = np.clip(u, u_min, u_max)
    g = spiral(u, a)
    d2 = ((g - p[:, None, :]) ** 2).sum(axis=-1)
    return u[np.arange(p.shape[0]), d2.argmin(axis=1)]


def manifold_distance(p: np.ndarray, a: float, u_min: float, u_max: float) -> np.ndarray:
    u = nearest_u(p, a, u_min, u_max)
    return np.linalg.norm(p - spiral(u, a), axis=-1)


def project_to_manifold(p: np.ndarray, a: float, u_min: float, u_max: float) -> np.ndarray:
    u = nearest_u(p, a, u_min, u_max)
    return spiral(u, a).astype(np.float32)


class PointDataset(Dataset):
    def __init__(self, points: torch.Tensor):
        self.points = points

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.points[idx]


def meta_from_dict(d: dict) -> SwissRollMeta:
    payload = dict(d)
    payload["mean"] = (float(payload["mean"][0]), float(payload["mean"][1]))
    payload["std"] = (float(payload["std"][0]), float(payload["std"][1]))
    return SwissRollMeta(**payload)


def resolve_cache_dir(cfg: DictConfig) -> Path | None:
    raw = cfg.data.get("cache_dir", None)
    if not raw:
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = ROOT / path
    return path


def save_swiss_roll(cache_dir: str | Path, train_raw: np.ndarray, eval_raw: np.ndarray, meta: SwissRollMeta) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "train.npy", np.asarray(train_raw, dtype=np.float32))
    np.save(cache_dir / "eval.npy", np.asarray(eval_raw, dtype=np.float32))
    (cache_dir / "meta.json").write_text(json.dumps(asdict(meta), indent=2))


def bundle_from_arrays(train_raw: np.ndarray, eval_raw: np.ndarray, meta: SwissRollMeta) -> dict:
    mean = np.asarray(meta.mean, dtype=np.float32)
    std = np.asarray(meta.std, dtype=np.float32)
    train_z = (train_raw - mean) / std
    eval_z = (eval_raw - mean) / std
    return {
        "train": PointDataset(torch.from_numpy(train_z.astype(np.float32))),
        "train_raw": np.asarray(train_raw, dtype=np.float32),
        "eval_raw": np.asarray(eval_raw, dtype=np.float32),
        "eval_z": torch.from_numpy(eval_z.astype(np.float32)),
        "mean": torch.from_numpy(mean),
        "std": torch.from_numpy(std),
        "meta": meta,
        "meta_dict": asdict(meta),
    }


def load_swiss_roll(cache_dir: str | Path) -> dict:
    cache_dir = Path(cache_dir)
    train_raw = np.load(cache_dir / "train.npy")
    eval_raw = np.load(cache_dir / "eval.npy")
    meta = meta_from_dict(json.loads((cache_dir / "meta.json").read_text()))
    return bundle_from_arrays(train_raw, eval_raw, meta)


def _u_range(cfg: DictConfig) -> tuple[float, float, float]:
    a = float(cfg.data.a)
    n_turns = float(cfg.data.n_turns)
    u_min = float(cfg.data.get("u_min", 1.5 * math.pi))
    if "u_max" in cfg.data:
        u_max = float(cfg.data.u_max)
    else:
        u_max = u_min + n_turns * 2.0 * math.pi
    return a, u_min, u_max


def build_swiss_roll(cfg: DictConfig) -> dict:
    cache_dir = resolve_cache_dir(cfg)
    regenerate = bool(cfg.data.get("regenerate", False))
    if cache_dir is not None and (cache_dir / "meta.json").is_file() and not regenerate:
        return load_swiss_roll(cache_dir)

    a, u_min, u_max = _u_range(cfg)
    n_turns = (u_max - u_min) / (2.0 * math.pi)
    sigma = float(cfg.data.sigma_obs)
    n_train = int(cfg.data.n_train)
    n_eval = int(cfg.data.n_eval)
    margin = float(cfg.data.margin)
    seed = int(cfg.seed)
    tau = float(cfg.data.get("tau", 3.0 * sigma))

    rng = np.random.default_rng(seed)
    train_raw, _ = sample_points(n_train, a, u_min, u_max, sigma, rng)
    eval_raw, _ = sample_points(n_eval, a, u_min, u_max, sigma, rng)

    R = float(np.abs(train_raw).max()) + margin
    rho_min = a * u_min - tau
    mean = train_raw.mean(axis=0)
    std = train_raw.std(axis=0).clip(min=1e-6)

    meta = SwissRollMeta(
        a=a,
        u_min=u_min,
        u_max=u_max,
        n_turns=n_turns,
        sigma_obs=sigma,
        n_train=n_train,
        n_eval=n_eval,
        margin=margin,
        tau=tau,
        rho_min=rho_min,
        R=R,
        seed=seed,
        mean=(float(mean[0]), float(mean[1])),
        std=(float(std[0]), float(std[1])),
        arc_length=arc_length(a, u_min, u_max),
    )
    if cache_dir is not None:
        save_swiss_roll(cache_dir, train_raw, eval_raw, meta)
    return bundle_from_arrays(train_raw, eval_raw, meta)


def denormalize(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return z * std.to(device=z.device, dtype=z.dtype) + mean.to(device=z.device, dtype=z.dtype)


def normalize(p: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (p - mean.to(device=p.device, dtype=p.dtype)) / std.to(device=p.device, dtype=p.dtype)
