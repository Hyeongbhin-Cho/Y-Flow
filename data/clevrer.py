# -*- coding: utf-8 -*-
# data/clevrer.py
"""Lazy CLEVRER video loading with original simulator annotations."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import Dataset

from data.base import register_dataset
from utils.paths import ROOT

SPLIT_RANGES = {"train": (0, 10000), "validation": (10000, 15000), "test": (15000, 20000)}


def _index_files(directory: Path, prefix: str, suffix: str) -> dict[int, Path]:
    records = {}
    for path in sorted(directory.rglob(f"{prefix}_*.{suffix}")):
        identifier = path.stem.removeprefix(prefix + "_")
        if not identifier.isdigit():
            continue
        scene_id = int(identifier)
        if scene_id in records:
            raise ValueError(f"Duplicate scene {scene_id} in {directory}")
        records[scene_id] = path
    return records


def _decode_video(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    try:
        import av
    except ImportError as exc:
        raise ImportError("CLEVRER video decoding requires PyAV: pip install av") from exc
    frames, times = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        for index, frame in enumerate(container.decode(stream)):
            if frame.time is None and fps <= 0:
                raise ValueError(f"No frame timestamps or frame rate: {path}")
            times.append(float(frame.time) if frame.time is not None else index / fps)
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"No decoded frames: {path}")
    timestamps = np.asarray(times, dtype=np.float64)
    timestamps -= timestamps[0]
    if len(times) > 1 and not np.all(np.diff(timestamps) > 0):
        raise ValueError(f"Non-increasing video timestamps: {path}")
    if fps <= 0 and len(times) > 1:
        fps = 1.0 / float(np.median(np.diff(timestamps)))
    return np.stack(frames), timestamps, fps


class CLEVRERDataset(Dataset):
    """One item is a RGB clip [C,T,H,W] plus unmodified reference annotations.

    Directory layout: videos/{split}/**/video_XXXXX.mp4 and
    annotations/{split}/**/annotation_XXXXX.json. No model or download is run.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "validation",
        *,
        n_frames: int | None = None,
        fps: float | None = None,
        height: int | None = None,
        width: int | None = None,
        start_time: float = 0.0,
        limit: int | None = None,
        load_questions: bool = False,
    ) -> None:
        if split not in SPLIT_RANGES:
            raise ValueError(f"Unknown CLEVRER split: {split}")
        if n_frames is not None and (isinstance(n_frames, bool) or int(n_frames) != n_frames or n_frames <= 0):
            raise ValueError("n_frames must be a positive integer or None")
        if fps is not None and (not math.isfinite(fps) or fps <= 0):
            raise ValueError("fps must be positive or None")
        if (height is None) != (width is None) or any(v is not None and (v <= 0 or int(v) != v) for v in (height, width)):
            raise ValueError("height and width must both be positive integers or None")
        if not math.isfinite(start_time) or start_time < 0:
            raise ValueError("start_time must be finite and nonnegative")
        if limit is not None and (limit <= 0 or int(limit) != limit):
            raise ValueError("limit must be a positive integer or None")
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.n_frames = int(n_frames) if n_frames is not None else None
        self.fps = fps
        self.height = int(height) if height is not None else None
        self.width = int(width) if width is not None else None
        self.start_time = start_time
        videos = _index_files(self.root / "videos" / split, "video", "mp4")
        if not videos:
            raise FileNotFoundError(
                f"No CLEVRER {split} videos in {self.root}. Run "
                f"python scripts/setup_clevrer.py --root {self.root} --splits {split}"
            )
        annotations = _index_files(self.root / "annotations" / split, "annotation", "json")
        self.records = []
        low, high = SPLIT_RANGES[split]
        for scene_id, video in videos.items():
            if not low <= scene_id < high:
                raise ValueError(f"Scene {scene_id} does not belong to {split}")
            annotation = annotations.get(scene_id)
            if split != "test" and annotation is None:
                raise FileNotFoundError(f"Missing annotation for {video}")
            self.records.append((scene_id, video, annotation if split != "test" else None))
        if limit is not None:
            self.records = self.records[:int(limit)]
        self.questions: dict[int, Any] = {}
        if load_questions:
            question_file = self.root / "questions" / f"{split}.json"
            for entry in json.loads(question_file.read_text()):
                self.questions[int(entry["scene_index"])] = entry["questions"]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        scene_id, video_path, annotation_path = self.records[index]
        frames, times, source_fps = _decode_video(video_path)
        if self.start_time > times[-1] + 1e-8:
            raise ValueError(f"Clip starts after the video ends: {video_path}")
        if self.fps is None:
            first = int(np.searchsorted(times, self.start_time - 1e-8))
            stop = len(times) if self.n_frames is None else first + self.n_frames
            if stop > len(times):
                raise ValueError(f"Clip requires {stop} frames but {video_path} has {len(times)}")
            indices = np.arange(first, stop)
            requested_times = times[indices]
        else:
            count = self.n_frames
            if count is None:
                count = int(np.floor((times[-1] - self.start_time) * self.fps + 1e-8)) + 1
            requested_times = self.start_time + np.arange(count) / self.fps
            if requested_times[-1] > times[-1] + 1e-8:
                raise ValueError(f"Requested clip extends past {video_path}; no frame repetition/padding is applied")
            right = np.searchsorted(times, requested_times).clip(0, len(times) - 1)
            left = (right - 1).clip(0)
            indices = np.where(abs(times[left] - requested_times) <= abs(times[right] - requested_times), left, right)
        video = torch.from_numpy(frames[indices].copy()).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        source_h, source_w = video.shape[-2:]
        height, width = self.height or source_h, self.width or source_w
        scale = min(height / source_h, width / source_w)
        resized_h, resized_w = max(1, round(source_h * scale)), max(1, round(source_w * scale))
        if (resized_h, resized_w) != (source_h, source_w):
            video = F.interpolate(video, size=(resized_h, resized_w), mode="bilinear", align_corners=False, antialias=True)
        top, left = (height - resized_h) // 2, (width - resized_w) // 2
        video = F.pad(video, (left, width - resized_w - left, top, height - resized_h - top), value=-1.0)
        video = video.permute(1, 0, 2, 3).contiguous()
        valid_region = torch.zeros((1, height, width), dtype=torch.bool)
        valid_region[:, top:top + resized_h, left:left + resized_w] = True
        annotation = json.loads(annotation_path.read_text()) if annotation_path else None
        frame_annotations = None
        if annotation is not None:
            if annotation.get("scene_index") != scene_id or Path(annotation.get("video_filename", "")).name != video_path.name:
                raise ValueError(f"Annotation/video mismatch: {annotation_path}")
            by_frame = {int(entry["frame_id"]): entry for entry in annotation["motion_trajectory"]}
            missing = [int(i) for i in indices if int(i) not in by_frame]
            if missing:
                raise ValueError(f"Missing frame annotations {missing} in {annotation_path}")
            frame_annotations = [by_frame[int(i)] for i in indices]
        return {
            "scene_index": scene_id,
            "split": self.split,
            "video_path": str(video_path),
            "video": video,
            "initial_frame": video[:, 0].clone(),
            "frame_indices": torch.as_tensor(indices, dtype=torch.long),
            "frame_times": torch.as_tensor(requested_times.copy(), dtype=torch.float64),
            "source_frame_times": torch.as_tensor(times[indices].copy(), dtype=torch.float64),
            "source_fps": source_fps,
            "fps": self.fps or source_fps,
            "valid_region": valid_region,
            "spatial_transform": {
                "source_height": source_h, "source_width": source_w,
                "scale_x": resized_w / source_w, "scale_y": resized_h / source_h,
                "pad_left": left, "pad_top": top,
            },
            # Original world coordinates/events are preserved, never relabeled as image coordinates.
            "annotation": annotation,
            "frame_annotations": frame_annotations,
            "questions": self.questions.get(scene_id),
        }


def collate_clevrer(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack fixed-size tensors; keep variable-length object/event metadata per video."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    tensor_keys = {"video", "initial_frame", "frame_indices", "frame_times", "source_frame_times", "valid_region"}
    return {
        key: torch.stack([item[key] for item in samples]) if key in tensor_keys else [item[key] for item in samples]
        for key in samples[0]
    }


@dataclass
class VideoDataBundle:
    """Lazy video counterpart of DataBundle; optional recognition constraint."""

    train: Dataset | None
    eval: Dataset
    meta: dict[str, Any]
    constraint: Any = None
    train_raw: Any = None

    @property
    def meta_dict(self) -> dict[str, Any]:
        return self.meta

    def keys(self) -> list[str]:
        return ["train", "eval", "meta", "meta_dict", "constraint", "train_raw"]

    def __getitem__(self, key: str) -> Any:
        if key not in self.keys():
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return key in self.keys()

    def get(self, key: str, default: Any = None) -> Any:
        return self[key] if key in self else default


@register_dataset("clevrer")
def build_clevrer(cfg: DictConfig) -> VideoDataBundle:
    data = cfg.data
    root = Path(str(data.get("cache_dir", "datasets/clevrer"))).expanduser()
    if not root.is_absolute():
        root = ROOT / root
    options = {key: data.get(key, None) for key in ("n_frames", "fps", "height", "width")}
    options["start_time"] = float(data.get("start_time", 0.0))
    options["load_questions"] = bool(data.get("load_questions", False))
    evaluation = CLEVRERDataset(root, str(data.get("eval_split", "validation")), limit=data.get("n_eval", None), **options)
    train_split = data.get("train_split", None)
    train = CLEVRERDataset(root, str(train_split), limit=data.get("n_train", None), **options) if train_split else None
    meta = {
        "dataset": "clevrer", "root": str(root), "eval_split": evaluation.split,
        "n_eval": len(evaluation), "n_train": len(train) if train is not None else 0,
        "video_layout": "CTHW", "pixel_range": [-1.0, 1.0],
        "annotation_coordinates": "original_simulator_world", "physics_evaluation": "not_implemented",
        "eval_scene_indices": [record[0] for record in evaluation.records],
        **options,
    }
    return VideoDataBundle(train=train, eval=evaluation, meta=meta)
