from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

KEYS = ("hist", "hist_mask", "fut", "fut_mask", "nbr", "nbr_mask", "lane", "lane_mask")
OPTIONAL_KEYS = ("focal_size", "obs", "obs_mask", "sdf", "nbr_size", "nbr_yaw")


def load_meta(cache_dir: str | Path) -> dict:
    return json.loads((Path(cache_dir) / "meta.json").read_text())


def load_split(cache_dir: str | Path, split: str, limit: int | None = None) -> dict[str, np.ndarray]:
    with np.load(Path(cache_dir) / f"{split}.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    missing = [k for k in KEYS if k not in arrays]
    if missing:
        raise KeyError(f"{split}.npz is missing {missing}")
    if limit is not None and limit > 0:
        arrays = {k: v[:limit] for k, v in arrays.items()}
    validate(arrays)
    return arrays


def validate(a: dict[str, np.ndarray]) -> None:
    n, h, _ = a["hist"].shape
    t = a["fut"].shape[1]
    checks = {
        "hist_mask": (n, h),
        "fut": (n, t, 2),
        "fut_mask": (n, t),
        "nbr_mask": a["nbr"].shape[:3],
        "lane_mask": a["lane"].shape[:3],
    }
    for key, shape in checks.items():
        if tuple(a[key].shape) != tuple(shape):
            raise ValueError(f"{key} shape {a[key].shape} != {shape}")
    if a["nbr"].shape[0] != n or a["nbr"].shape[2] != h or a["lane"].shape[0] != n:
        raise ValueError("nbr/lane leading dims do not match hist")
    if not a["hist_mask"][:, -1].all():
        raise ValueError("last history step must be observed for every sample (frame origin)")
    if not np.isfinite(a["fut"][a["fut_mask"]]).all():
        raise ValueError("non-finite values in valid future steps")


def save_split(cache_dir: str | Path, split: str, arrays: dict[str, np.ndarray]) -> None:
    validate(arrays)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(cache_dir) / f"{split}.npz", **arrays)


def future_std(arrays: dict[str, np.ndarray], mode: str = "per_step") -> np.ndarray:
    fut, mask = arrays["fut"].astype(np.float64), arrays["fut_mask"]
    if mode == "global":
        return np.maximum(fut[mask].std(axis=0), 1e-3).astype(np.float32)
    w = mask[..., None].astype(np.float64)
    n = np.maximum(w.sum(axis=0), 1.0)
    mean = (fut * w).sum(axis=0) / n
    var = (((fut - mean) ** 2) * w).sum(axis=0) / n
    return np.maximum(np.sqrt(var), 0.05).astype(np.float32)


class VehicleTrajDataset(Dataset):

    def __init__(self, arrays: dict[str, np.ndarray], optional: bool = False):
        keys = list(KEYS) + ([k for k in OPTIONAL_KEYS if k in arrays] if optional else [])
        self.tensors = {k: torch.from_numpy(np.ascontiguousarray(arrays[k])) for k in keys}
        for k in keys:
            if k.endswith("mask"):
                self.tensors[k] = self.tensors[k].bool()
            elif k == "sdf":
                self.tensors[k] = self.tensors[k].to(torch.int8)
            else:
                self.tensors[k] = torch.nan_to_num(self.tensors[k].float())

    def __len__(self) -> int:
        return int(self.tensors["hist"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self.tensors.items()}


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
