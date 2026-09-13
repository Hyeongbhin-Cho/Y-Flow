# -*- coding: utf-8 -*-
"""Evaluation for the directly supervised CLEVRER video recognizer."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from data.base import build_dataset
from data.clevrer_state import (
    CLEVRER_COLORS,
    CLEVRER_MATERIALS,
    CLEVRER_SHAPES,
    collate_clevrer_recognition,
    layout_from_cfg,
    pack_state,
    task_safety,
    unpack_state,
)
from model import build_model
from train.clevrer_recognition import CLEVRERRecognitionCriterion
from utils.device import get_device
from utils.paths import ROOT, method_dir, published_dir


def _checkpoint_path(cfg: DictConfig, method: str) -> Path:
    run_dir = method_dir(cfg, method)
    for candidate in (run_dir / "best.pt", run_dir / "last.pt"):
        if candidate.is_file():
            return candidate
    directory = published_dir(cfg)
    if directory is not None:
        for candidate in (directory / "last.pt", directory / "best.pt"):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"no recognition checkpoint found in {run_dir} or model.local_dir; "
        "run `python main.py recognition --mode train` first"
    )


def _f1(tp: int, fp: int, fn: int) -> float:
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def _attributes(parts: dict[str, torch.Tensor], batch_i: int, slot: int) -> tuple[str, str, str]:
    return (
        CLEVRER_COLORS[int(parts["color"][batch_i, slot].argmax())],
        CLEVRER_MATERIALS[int(parts["material"][batch_i, slot].argmax())],
        CLEVRER_SHAPES[int(parts["shape"][batch_i, slot].argmax())],
    )


def _event_f1(predicted: list[tuple[int, int, int]], truth: list[tuple[int, int, int]], tolerance: int) -> tuple[int, int, int]:
    """Greedily match collision events by object pair and temporal tolerance."""
    remaining = set(range(len(truth)))
    tp = 0
    for frame, obj_i, obj_j in predicted:
        candidates = [
            idx
            for idx in remaining
            if truth[idx][1:] == (obj_i, obj_j) and abs(truth[idx][0] - frame) <= tolerance
        ]
        if candidates:
            chosen = min(candidates, key=lambda idx: abs(truth[idx][0] - frame))
            remaining.remove(chosen)
            tp += 1
    return tp, len(predicted) - tp, len(remaining)


@torch.inference_mode()
def run_eval(cfg: DictConfig, method: str = "recognition", device: torch.device | None = None) -> dict:
    if method not in ("recognition", "clevrer_resnet34"):
        raise ValueError(
            f"{method!r} is not an Exp-02-sub recognition method; use the `recognition` command"
        )
    device = device or get_device(cfg)
    bundle = build_dataset(cfg)
    layout = layout_from_cfg(cfg)
    run_method = "recognition"
    checkpoint_path = _checkpoint_path(cfg, run_method)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("extra", {}).get("task") != "clevrer_video_recognition":
        raise ValueError(f"checkpoint is not a direct CLEVRER recognition model: {checkpoint_path}")

    # The full training checkpoint already contains the pretrained backbone. Avoid requiring
    # the separate ImageNet bootstrap file merely to restore it for evaluation.
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    model_cfg.model.pretrained = False
    model = build_model(model_cfg).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    stats = payload["extra"]["criterion"]
    criterion = CLEVRERRecognitionCriterion(
        cfg,
        stats["position_mean"],
        stats["position_std"],
        stats["velocity_mean"],
        stats["velocity_std"],
        stats["collision_pos_weight"],
    ).to(device)
    thresholds = {
        "objectness": float(cfg.eval.get("objectness_threshold", 0.5)),
        "visibility": float(cfg.eval.get("visibility_threshold", 0.5)),
        "collision": float(cfg.eval.get("collision_threshold", 0.5)),
    }
    thresholds.update(payload.get("extra", {}).get("thresholds", {}))

    loader = DataLoader(
        bundle.eval,
        batch_size=int(cfg.eval.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(cfg.eval.get("num_workers", 0)),
        collate_fn=collate_clevrer_recognition,
        pin_memory=device.type == "cuda",
    )
    limit = int(cfg.data.get("n_eval", len(bundle.eval)))
    count = 0
    attribute_tp = attribute_fp = attribute_fn = 0
    object_tp = object_fp = object_fn = 0
    object_count_abs_error: list[float] = []
    visibility_tp = visibility_fp = visibility_fn = 0
    ade_values: list[float] = []
    fde_values: list[float] = []
    velocity_errors: list[float] = []
    track_objects_matched = track_objects_total = 0
    collision_tp = collision_fp = collision_fn = 0
    safety_rows: dict[str, list[float]] = {key: [] for key in ("attr", "track", "collision", "total")}
    inference_seconds = 0.0
    pair_i = model.pair_i
    pair_j = model.pair_j

    for batch in loader:
        if count >= limit:
            break
        video = batch["video"].to(device, non_blocking=True)
        target_packed = batch["state"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        outputs = model(video)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - start
        gt_parts = unpack_state(target_packed, layout)
        matches = criterion.match(outputs, target_packed)
        object_probability = outputs["objectness_logits"].sigmoid()
        visibility_probability = outputs["visibility_logits"].sigmoid()
        collision_probability = outputs["collision_logits"].sigmoid()

        batch_size = video.shape[0]
        for batch_i in range(batch_size):
            if count >= limit:
                break
            predicted_active = object_probability[batch_i] >= float(thresholds["objectness"])
            ground_truth_active = gt_parts["color"][batch_i].sum(dim=-1) > 0.5
            object_count_abs_error.append(
                abs(float(predicted_active.sum()) - float(ground_truth_active.sum()))
            )
            mapping = dict(matches[batch_i])

            predicted_attributes = [
                _attributes(
                    {
                        "color": outputs["color_logits"],
                        "material": outputs["material_logits"],
                        "shape": outputs["shape_logits"],
                    },
                    batch_i,
                    pred_slot,
                )
                for pred_slot in torch.where(predicted_active)[0].tolist()
            ]
            truth_attributes = [
                _attributes(gt_parts, batch_i, gt_slot)
                for gt_slot in torch.where(ground_truth_active)[0].tolist()
            ]
            matched_attributes = sum((Counter(predicted_attributes) & Counter(truth_attributes)).values())
            attribute_tp += matched_attributes
            attribute_fp += len(predicted_attributes) - matched_attributes
            attribute_fn += len(truth_attributes) - matched_attributes

            active_mapping = {
                pred_slot: gt_slot
                for pred_slot, gt_slot in mapping.items()
                if bool(predicted_active[pred_slot])
            }
            object_tp += len(active_mapping)
            object_fp += int(predicted_active.sum()) - len(active_mapping)
            object_fn += int(ground_truth_active.sum()) - len(active_mapping)
            track_objects_matched += len(active_mapping)
            track_objects_total += int(ground_truth_active.sum())
            for pred_slot, gt_slot in mapping.items():
                if pred_slot not in active_mapping:
                    visibility_fn += int((gt_parts["vis"][batch_i, gt_slot] > 0.5).sum())
            for pred_slot, gt_slot in active_mapping.items():
                gt_visible = gt_parts["vis"][batch_i, gt_slot] > 0.5
                pred_visible = visibility_probability[batch_i, pred_slot] >= float(thresholds["visibility"])
                visibility_tp += int((pred_visible & gt_visible).sum())
                visibility_fp += int((pred_visible & ~gt_visible).sum())
                visibility_fn += int((~pred_visible & gt_visible).sum())
                if gt_visible.any():
                    pred_pos = outputs["positions"][batch_i, pred_slot] * criterion.position_std + criterion.position_mean
                    pred_vel = outputs["velocities"][batch_i, pred_slot] * criterion.velocity_std + criterion.velocity_mean
                    delta = (pred_pos[gt_visible] - gt_parts["pos"][batch_i, gt_slot, gt_visible]).norm(dim=-1)
                    ade_values.extend(delta.cpu().tolist())
                    last_visible = torch.where(gt_visible)[0][-1]
                    fde_values.append(float((pred_pos[last_visible] - gt_parts["pos"][batch_i, gt_slot, last_visible]).norm()))
                    velocity_delta = (pred_vel[gt_visible] - gt_parts["vel"][batch_i, gt_slot, gt_visible]).norm(dim=-1)
                    velocity_errors.extend(velocity_delta.cpu().tolist())

            truth_events: list[tuple[int, int, int]] = []
            for frame in range(layout.n_frames):
                for gt_i in range(layout.n_slots):
                    for gt_j in range(gt_i + 1, layout.n_slots):
                        if (
                            gt_parts["coll"][batch_i, frame, gt_i, gt_j] > 0.5
                            and gt_parts["vis"][batch_i, gt_i, frame] > 0.5
                            and gt_parts["vis"][batch_i, gt_j, frame] > 0.5
                        ):
                            truth_events.append((frame, gt_i, gt_j))
            predicted_events: list[tuple[int, int, int]] = []
            for pair_idx, (pred_i, pred_j) in enumerate(zip(pair_i.tolist(), pair_j.tolist())):
                if pred_i not in active_mapping or pred_j not in active_mapping:
                    continue
                gt_i, gt_j = sorted((active_mapping[pred_i], active_mapping[pred_j]))
                pair_visible = (
                    visibility_probability[batch_i, pred_i] >= float(thresholds["visibility"])
                ) & (visibility_probability[batch_i, pred_j] >= float(thresholds["visibility"]))
                event_frames = torch.where(
                    (collision_probability[batch_i, :, pair_idx] >= float(thresholds["collision"]))
                    & pair_visible
                )[0]
                predicted_events.extend((int(frame), gt_i, gt_j) for frame in event_frames.tolist())
            tp, fp, fn = _event_f1(
                predicted_events,
                truth_events,
                tolerance=int(cfg.eval.get("collision_frame_tolerance", 2)),
            )
            collision_tp += tp
            collision_fp += fp
            collision_fn += fn

            # Quantize the prediction for physical diagnostics; learned task metrics above
            # continue to use their native per-head probabilities and logits.
            color = torch.zeros_like(gt_parts["color"][batch_i])
            material = torch.zeros_like(gt_parts["material"][batch_i])
            shape = torch.zeros_like(gt_parts["shape"][batch_i])
            for pred_slot in torch.where(predicted_active)[0].tolist():
                color[pred_slot, outputs["color_logits"][batch_i, pred_slot].argmax()] = 1.0
                material[pred_slot, outputs["material_logits"][batch_i, pred_slot].argmax()] = 1.0
                shape[pred_slot, outputs["shape_logits"][batch_i, pred_slot].argmax()] = 1.0
            pred_vis = (visibility_probability[batch_i] >= float(thresholds["visibility"])) & predicted_active[:, None]
            pred_pos = outputs["positions"][batch_i] * criterion.position_std + criterion.position_mean
            pred_vel = outputs["velocities"][batch_i] * criterion.velocity_std + criterion.velocity_mean
            pred_coll = torch.zeros_like(gt_parts["coll"][batch_i])
            for pair_idx, (pred_i, pred_j) in enumerate(zip(pair_i.tolist(), pair_j.tolist())):
                collision = collision_probability[batch_i, :, pair_idx] >= float(thresholds["collision"])
                collision &= pred_vis[pred_i] & pred_vis[pred_j]
                pred_coll[:, pred_i, pred_j] = collision.float()
                pred_coll[:, pred_j, pred_i] = collision.float()
            packed_prediction = pack_state(
                color,
                material,
                shape,
                pred_vis.float(),
                pred_pos,
                pred_vel,
                pred_coll,
                layout,
            )
            safety = task_safety(bundle.constraint.h(packed_prediction))
            for key in safety_rows:
                safety_rows[key].append(float(safety[key]))
            count += 1

    metrics = {
        "method": "recognition",
        "run_name": str(cfg.get("run_name", "default")),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)) if checkpoint_path.is_relative_to(ROOT) else str(checkpoint_path),
        "n_samples": count,
        "thresholds": thresholds,
        "object_count_mae": float(np.mean(object_count_abs_error)) if object_count_abs_error else 0.0,
        "object_precision": float(object_tp / max(object_tp + object_fp, 1)),
        "object_recall": float(object_tp / max(object_tp + object_fn, 1)),
        "object_f1": _f1(object_tp, object_fp, object_fn),
        "attribute_f1": _f1(attribute_tp, attribute_fp, attribute_fn),
        "visibility_f1": _f1(visibility_tp, visibility_fp, visibility_fn),
        "track_object_coverage": float(track_objects_matched / max(track_objects_total, 1)),
        "ade_visible": float(np.mean(ade_values)) if ade_values else 0.0,
        "fde_visible": float(np.mean(fde_values)) if fde_values else 0.0,
        "velocity_mae": float(np.mean(velocity_errors)) if velocity_errors else 0.0,
        "collision_event_f1": _f1(collision_tp, collision_fp, collision_fn),
        "collision_event_precision": float(collision_tp / max(collision_tp + collision_fp, 1)),
        "collision_event_recall": float(collision_tp / max(collision_tp + collision_fn, 1)),
        "collision_tp": collision_tp,
        "collision_fp": collision_fp,
        "collision_fn": collision_fn,
        "attr_safe": float(np.mean(safety_rows["attr"])) if safety_rows["attr"] else 0.0,
        "track_safe": float(np.mean(safety_rows["track"])) if safety_rows["track"] else 0.0,
        "collision_safe": float(np.mean(safety_rows["collision"])) if safety_rows["collision"] else 0.0,
        "total_safe": float(np.mean(safety_rows["total"])) if safety_rows["total"] else 0.0,
        "inference_time_s": inference_seconds,
        "inference_time_s_per_clip": inference_seconds / max(count, 1),
    }
    out_dir = method_dir(cfg, "recognition")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return metrics
