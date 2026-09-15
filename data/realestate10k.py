"""RealEstate10K prepared-frame loader and world-to-camera geometry.

Source format: https://google.github.io/realestate10k/download.html
No downloading, learned matcher, or RGB hard-constraint claims in this module.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.base import register_dataset
from data.clevrer import VideoDataBundle

ROOT = Path(__file__).resolve().parents[1]
WAN_TEMPORAL_COMPRESSION = 4
WAN_SPATIAL_COMPRESSION = 8
WAN_TRANSFORMER_SPATIAL_PATCH = 2


def _wan_video_input(video: torch.Tensor, timestamps_us: torch.Tensor):
    """Pad an RGB clip to Wan2.1's 1 + 4k causal-VAE frame contract."""
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError("video must have shape [3,T,H,W]")
    _, frames, height, width = video.shape
    spatial_multiple = WAN_SPATIAL_COMPRESSION * WAN_TRANSFORMER_SPATIAL_PATCH
    if height % spatial_multiple or width % spatial_multiple:
        raise ValueError(
            f"Wan2.1 requires input height/width divisible by {spatial_multiple}; "
            f"got {(height, width)}"
        )
    padded_frames = 1 + ((frames - 1 + WAN_TEMPORAL_COMPRESSION - 1) // WAN_TEMPORAL_COMPRESSION) * WAN_TEMPORAL_COMPRESSION
    padding = padded_frames - frames
    if padding:
        video = torch.cat([video, video[:, -1:].expand(-1, padding, -1, -1)], dim=1)
        timestamps_us = torch.cat([timestamps_us, timestamps_us[-1:].expand(padding)])
    frame_mask = torch.arange(padded_frames) < frames
    return video.contiguous(), timestamps_us.contiguous(), frame_mask


def parse_camera_file(path: str | Path) -> dict:
    lines = Path(path).read_text().splitlines()
    if not lines or not lines[0].strip():
        raise ValueError(f"Missing video URL: {path}")
    rows = [line.split() for line in lines[1:] if line.strip()]
    if len(rows) < 2 or any(len(row) != 19 for row in rows):
        raise ValueError(f"Expected >=2 camera rows with 19 columns: {path}")
    timestamps = np.array([int(row[0]) for row in rows], dtype=np.int64)
    values = np.asarray([row[1:] for row in rows], dtype=np.float64)
    if not np.isfinite(values).all() or np.any(timestamps < 0) or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"Invalid camera values/timestamps: {path}")
    if np.any(values[:, :2] <= 0):
        raise ValueError("Focal lengths must be positive")
    K = np.zeros((len(rows), 3, 3), dtype=np.float64)
    K[:, 0, 0], K[:, 1, 1] = values[:, 0], values[:, 1]
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = values[:, 2], values[:, 3], 1
    # Columns 5,6 (zero based) are reserved; the final 12 are [R|t].
    poses = values[:, 6:].reshape(-1, 3, 4)
    R = poses[:, :, :3]
    if not np.allclose(R @ R.transpose(0, 2, 1), np.eye(3), atol=2e-3) or not np.allclose(np.linalg.det(R), 1, atol=2e-3):
        raise ValueError("Camera rotations must be proper orthonormal matrices")
    return dict(source_url=lines[0].strip(), timestamps_us=timestamps, K_normalized=K, world_to_camera=poses)


def fundamental_matrix(K1, P1, K2, P2):
    """Return scale-normalized F and validity; zero translation has invalid F.

    Validity here is algebraic only; small parallax needs a measured oracle.
    """
    R = P2[:, :3] @ P1[:, :3].T
    t = P2[:, 3] - R @ P1[:, 3]
    norm = np.linalg.norm(t)
    if not np.isfinite(norm) or norm <= 1e-12:
        return np.zeros((3, 3), dtype=np.float64), False
    x, y, z = t / norm
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    F = np.linalg.inv(K2).T @ skew @ R @ np.linalg.inv(K1)
    return F / np.linalg.norm(F), True


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes dataset root: {relative}")
    return path


class RealEstate10KEpipolarConstraint:
    """Coordinate-space epipolar oracle for matched pixel correspondences.

    This is intentionally not a ``BaseConstraint``: RealEstate10K supplies a
    different F per clip/frame pair, and exact projection is only defined for
    correspondence coordinates, not RGB frames or Wan latents.
    """

    def __init__(self, tolerance_px: float = 1.0, min_line_norm: float = 1e-10):
        if not np.isfinite(tolerance_px) or tolerance_px < 0:
            raise ValueError("tolerance_px must be finite and nonnegative")
        if not np.isfinite(min_line_norm) or min_line_norm <= 0:
            raise ValueError("min_line_norm must be finite and positive")
        self.tolerance_px = float(tolerance_px)
        self.min_line_norm = float(min_line_norm)

    @staticmethod
    def _line(reference_points, fundamental_matrices):
        if reference_points.shape[-1] != 2 or fundamental_matrices.shape[-2:] != (3, 3):
            raise ValueError("Expected pixel points [...,2] and F matrices [...,3,3]")
        if isinstance(reference_points, torch.Tensor) != isinstance(fundamental_matrices, torch.Tensor):
            raise TypeError("points and fundamental matrices must use the same array type")
        if isinstance(reference_points, torch.Tensor):
            ones = torch.ones_like(reference_points[..., :1])
            p1 = torch.cat([reference_points, ones], dim=-1)
            F = fundamental_matrices
            if F.ndim == reference_points.ndim and F.shape[:-2] == reference_points.shape[:-2]:
                F = F.unsqueeze(-3)
            return (F @ p1.unsqueeze(-1)).squeeze(-1)
        points = np.asarray(reference_points)
        matrices = np.asarray(fundamental_matrices)
        p1 = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
        if matrices.ndim == points.ndim and matrices.shape[:-2] == points.shape[:-2]:
            matrices = np.expand_dims(matrices, axis=-3)
        return (matrices @ p1[..., None])[..., 0]

    def h(self, reference_points, target_points, fundamental_matrices, valid=None):
        """Return point-to-epipolar-line violations in pixels (feasible <= 0)."""
        if reference_points.shape != target_points.shape or target_points.shape[-1] != 2:
            raise ValueError("reference_points and target_points must have the same [...,2] shape")
        line = self._line(reference_points, fundamental_matrices)
        if isinstance(target_points, torch.Tensor):
            target_h = torch.cat([target_points, torch.ones_like(target_points[..., :1])], dim=-1)
            norm = torch.linalg.vector_norm(line[..., :2], dim=-1)
            distance = (line * target_h).sum(dim=-1).abs() / norm.clamp_min(self.min_line_norm)
            violation = distance - self.tolerance_px
            feasible_line = norm >= self.min_line_norm
            if valid is not None:
                valid_mask = torch.as_tensor(valid, device=target_points.device, dtype=torch.bool)
                if valid_mask.ndim == feasible_line.ndim - 1 and valid_mask.shape == feasible_line.shape[:-1]:
                    valid_mask = valid_mask.unsqueeze(-1)
                feasible_line = feasible_line & valid_mask
            return {"epipolar_px": torch.where(feasible_line, violation, torch.full_like(violation, torch.inf))}
        target = np.asarray(target_points)
        target_h = np.concatenate([target, np.ones_like(target[..., :1])], axis=-1)
        norm = np.linalg.norm(line[..., :2], axis=-1)
        violation = np.abs(np.sum(line * target_h, axis=-1)) / np.maximum(norm, self.min_line_norm) - self.tolerance_px
        feasible_line = norm >= self.min_line_norm
        if valid is not None:
            valid_mask = np.asarray(valid, dtype=bool)
            if valid_mask.ndim == feasible_line.ndim - 1 and valid_mask.shape == feasible_line.shape[:-1]:
                valid_mask = np.expand_dims(valid_mask, axis=-1)
            feasible_line = feasible_line & valid_mask
        return {"epipolar_px": np.where(feasible_line, violation, np.inf)}

    def project_feasible(self, reference_points, target_points, fundamental_matrices, valid=None):
        """Project target pixels to their lines; invalid/degenerate pairs stay unchanged.

        Returns ``(projected_points, projectable_mask)``. For every true mask entry,
        the result is the exact least-distance Euclidean projection to ``F p1``.
        """
        if reference_points.shape != target_points.shape or target_points.shape[-1] != 2:
            raise ValueError("reference_points and target_points must have the same [...,2] shape")
        line = self._line(reference_points, fundamental_matrices)
        if isinstance(target_points, torch.Tensor):
            norm_sq = line[..., :2].square().sum(dim=-1)
            projectable = norm_sq >= self.min_line_norm**2
            if valid is not None:
                valid_mask = torch.as_tensor(valid, device=target_points.device, dtype=torch.bool)
                if valid_mask.ndim == projectable.ndim - 1 and valid_mask.shape == projectable.shape[:-1]:
                    valid_mask = valid_mask.unsqueeze(-1)
                projectable = projectable & valid_mask
            signed = (line[..., :2] * target_points).sum(dim=-1) + line[..., 2]
            projected = target_points - signed.div(norm_sq.clamp_min(self.min_line_norm**2))[..., None] * line[..., :2]
            return torch.where(projectable[..., None], projected, target_points), projectable
        target = np.asarray(target_points)
        norm_sq = np.sum(line[..., :2] ** 2, axis=-1)
        projectable = norm_sq >= self.min_line_norm**2
        if valid is not None:
            valid_mask = np.asarray(valid, dtype=bool)
            if valid_mask.ndim == projectable.ndim - 1 and valid_mask.shape == projectable.shape[:-1]:
                valid_mask = np.expand_dims(valid_mask, axis=-1)
            projectable = projectable & valid_mask
        signed = np.sum(line[..., :2] * target, axis=-1) + line[..., 2]
        projected = target - (signed / np.maximum(norm_sq, self.min_line_norm**2))[..., None] * line[..., :2]
        return np.where(projectable[..., None], projected, target), projectable


class RealEstate10KDataset(Dataset):
    """Lazy float32 RGB [C,T,H,W] in [-1,1], float64 camera geometry."""

    def __init__(self, root, split="train", limit=None, prompt=None, noise_seed=None):
        if split not in ("train", "test"):
            raise ValueError("split must be train or test")
        if limit is not None and (isinstance(limit, bool) or int(limit) != limit or limit <= 0):
            raise ValueError("limit must be a positive integer")
        self.root, self.split = Path(root).resolve(), split
        path = self.root / "manifests" / f"{split}.json"
        manifest = json.loads(path.read_text())
        if manifest.get("schema_version") != 1 or manifest.get("split") != split:
            raise ValueError(f"Unsupported or mismatched manifest: {path}")
        self.records = manifest["clips"][:limit]
        if not self.records:
            raise ValueError(f"No prepared clips: {path}; run scripts/setup_realestate10k.py")
        if len({r["clip_id"] for r in self.records}) != len(self.records):
            raise ValueError("Duplicate clip IDs")
        self.meta = manifest.get("settings", {})
        self.prompt = prompt or self.meta.get("prompt", "a realistic real estate interior")
        self.noise_seed = int(self.meta.get("seed", 42) if noise_seed is None else noise_seed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        camera_path = safe_path(self.root, record["cameras"])
        with np.load(camera_path, allow_pickle=False) as data:
            cameras = {k: data[k].copy() for k in data.files}
        frames = []
        for relative in record["frames"]:
            with Image.open(safe_path(self.root, relative)) as image:
                frames.append(np.asarray(image.convert("RGB")).copy())
        video = np.stack(frames)
        T, H, W, _ = video.shape
        K, P = cameras["K"], cameras["world_to_camera"]
        if K.shape != (T, 3, 3) or P.shape != (T, 3, 4):
            raise ValueError("Frame/camera count mismatch")
        pairs = sorted({(0, k) for k in range(1, T)} | {(k, k+1) for k in range(T-1)})
        matrices, valid = zip(*(fundamental_matrix(K[i], P[i], K[j], P[j]) for i, j in pairs))
        video = torch.from_numpy(video).permute(3, 0, 1, 2).float() / 127.5 - 1
        timestamps = cameras["actual_timestamps_us"]
        wan_video, wan_timestamps, wan_frame_mask = _wan_video_input(
            video, torch.as_tensor(timestamps, dtype=torch.int64)
        )
        return {
            "clip_id": record["clip_id"], "source_url": record["source_url"],
            "prompt": record.get("prompt", self.prompt),
            "noise_seed": torch.tensor(int(record.get("noise_seed", self.noise_seed + index)), dtype=torch.int64),
            "video": video,
            "wan_video": wan_video,
            "wan_timestamps_us": wan_timestamps,
            "wan_frame_mask": wan_frame_mask,
            **{key: torch.from_numpy(value) for key, value in cameras.items()},
            "pair_indices": torch.tensor(pairs, dtype=torch.long),
            "fundamental_matrices": torch.from_numpy(np.stack(matrices)),
            "fundamental_valid": torch.tensor(valid, dtype=torch.bool),
            "valid_region": torch.ones((T, H, W), dtype=torch.bool),
            "static_scene_review": record.get("static_scene_review", "pending"),
        }


def collate_realestate10k(batch):
    """Stack equal-size clips; preserve string metadata as lists."""
    return {key: torch.stack([item[key] for item in batch]) if isinstance(batch[0][key], torch.Tensor)
            else [item[key] for item in batch] for key in batch[0]}


@register_dataset("realestate10k")
def build_realestate10k(cfg):
    options = cfg.data
    root = Path(str(options.get("cache_dir", "datasets/realestate10k"))).expanduser()
    if not root.is_absolute():
        root = ROOT / root
    evaluation = RealEstate10KDataset(
        root, options.get("eval_split", "train"), options.get("n_eval", 10),
        prompt=str(options.get("prompt", "a realistic real estate interior")),
        noise_seed=int(options.get("noise_seed", options.get("seed", 42))),
    )
    constraint = RealEstate10KEpipolarConstraint(
        tolerance_px=float(options.get("epipolar_tolerance_px", 1.0))
    )
    return VideoDataBundle(train=None, eval=evaluation, constraint=constraint, meta={
        "dataset": "realestate10k", "root": str(root), "eval_split": evaluation.split,
        "n_eval": len(evaluation), "video_layout": "CTHW", "pixel_range": [-1, 1],
        "wan_video_layout": "CTHW", "wan_input_frames": "1 + 4k",
        "wan_temporal_compression": WAN_TEMPORAL_COMPRESSION,
        "wan_spatial_compression": WAN_SPATIAL_COMPRESSION,
        "wan_transformer_spatial_patch": WAN_TRANSFORMER_SPATIAL_PATCH,
        "camera_convention": "world_to_camera", "geometry_coordinates": "pixel",
        "constraint_status": "epipolar_correspondence_space_only",
        "constraint_tolerance_px": constraint.tolerance_px,
        **evaluation.meta,
    })
