# -*- coding: utf-8 -*-
# model/clevrer_eval.py
"""CLEVRER physics-eval parsers: official proposals, PropNet preds, Mask R-CNN."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model.download import DEFAULT_CLEVRER_EVAL_DIR
from utils.paths import ROOT

ATTR_KEYS = ("color", "material", "shape")
COCO_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


def _attr_tuple(obj: dict[str, Any]) -> tuple[str, str, str]:
    return tuple(str(obj[k]).lower() for k in ATTR_KEYS)  # type: ignore[return-value]


def find_proposal_path(root: Path, scene_id: int) -> Path:
    for name in (f"proposal_{scene_id:05d}.json", f"sim_{scene_id:05d}.json"):
        direct = root / "visual_masks" / name
        if direct.is_file():
            return direct
    matches = list((root / "visual_masks").rglob(f"proposal_{scene_id:05d}.json"))
    matches += list((root / "visual_masks").rglob(f"sim_{scene_id:05d}.json"))
    if not matches:
        raise FileNotFoundError(f"No visual mask for scene {scene_id} in {root}")
    return matches[0]


def find_propnet_pred_path(root: Path, scene_id: int, *, supervised: bool = True) -> Path:
    matches = list((root / "propnet_preds").rglob(f"sim_{scene_id:05d}.json"))
    if not matches:
        raise FileNotFoundError(f"No PropNet pred for scene {scene_id} in {root}")
    if supervised:
        hits = [p for p in matches if "with_edge_supervision" in p.as_posix()]
    else:
        hits = [p for p in matches if "without_edge_supervision" in p.as_posix()]
    return (hits or matches)[0]


def load_visual_mask(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_propnet_pred(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def proposal_attribute_set(payload: dict[str, Any]) -> set[tuple[str, str, str]]:
    objects = payload.get("ground_truth", {}).get("objects") or []
    if objects:
        attrs = {_attr_tuple(obj) for obj in objects if all(k in obj for k in ATTR_KEYS)}
        if attrs:
            return attrs
    seen: set[tuple[str, str, str]] = set()
    for frame in payload.get("frames") or []:
        for obj in frame.get("objects") or []:
            if float(obj.get("score", 1.0)) < 0.9:
                continue
            if all(k in obj for k in ATTR_KEYS):
                seen.add(_attr_tuple(obj))
    return seen


def annotation_attribute_set(annotation: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {_attr_tuple(obj) for obj in annotation.get("object_property") or []}


def visible_counts_from_annotation(annotation: dict[str, Any], n_frames: int = 128) -> list[int]:
    by_frame = {int(entry["frame_id"]): entry for entry in annotation.get("motion_trajectory") or []}
    counts = []
    for frame_id in range(n_frames):
        entry = by_frame.get(frame_id, {"objects": []})
        counts.append(sum(1 for obj in entry.get("objects") or [] if obj.get("inside_camera_view")))
    return counts


def visible_counts_from_proposal(payload: dict[str, Any], n_frames: int = 128) -> list[int]:
    frames = payload.get("frames") or []
    by_index: dict[int, int] = {}
    for index, frame in enumerate(frames):
        key = int(frame.get("frame_index", frame.get("frame_id", index)))
        by_index[key] = sum(1 for obj in frame.get("objects") or [] if float(obj.get("score", 1.0)) >= 0.9)
    return [by_index.get(i, 0) for i in range(n_frames)]


def set_scores(pred: set, truth: set) -> dict[str, float]:
    if not truth and not pred:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_pred": 0, "n_truth": 0}
    tp = len(pred & truth)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "n_pred": len(pred), "n_truth": len(truth)}


def annotation_collisions(annotation: dict[str, Any]) -> list[tuple[int, frozenset[int]]]:
    events = []
    for item in annotation.get("collision") or []:
        ids = item.get("object_ids") or item.get("object") or []
        events.append((int(item["frame_id"]), frozenset(int(i) for i in ids)))
    return events


def proposal_collisions(payload: dict[str, Any]) -> list[tuple[int, frozenset[int]]]:
    events = []
    for item in payload.get("ground_truth", {}).get("collisions") or []:
        frame = item.get("frame", item.get("frame_id"))
        ids = item.get("object") or item.get("object_ids") or []
        events.append((int(frame), frozenset(int(i) for i in ids)))
    return events


def propnet_collisions(payload: dict[str, Any]) -> list[tuple[int, frozenset[tuple[str, str, str]]]]:
    """Collisions keyed by attribute triples; PropNet ids are not GT object ids."""
    objects = payload.get("objects") or []
    id_to_attr = {}
    for obj in objects:
        if all(k in obj for k in ATTR_KEYS):
            id_to_attr[obj.get("id")] = _attr_tuple(obj)
    events = []
    preds = payload.get("predictions") or []
    if not preds:
        return events
    for item in preds[0].get("collisions") or []:
        frame = int(item.get("frame", item.get("frame_index", -1)))
        pair = []
        for obj in item.get("objects") or []:
            if isinstance(obj, dict) and all(k in obj for k in ATTR_KEYS):
                pair.append(_attr_tuple(obj))
            elif obj in id_to_attr:
                pair.append(id_to_attr[obj])
        if len(pair) == 2 and 0 <= frame < 128:
            events.append((frame, frozenset(pair)))
    return events


def annotation_collisions_by_attr(annotation: dict[str, Any]) -> list[tuple[int, frozenset[tuple[str, str, str]]]]:
    id_to_attr = {int(obj["object_id"]): _attr_tuple(obj) for obj in annotation.get("object_property") or []}
    events = []
    for item in annotation.get("collision") or []:
        ids = [int(i) for i in (item.get("object_ids") or [])]
        attrs = [id_to_attr[i] for i in ids if i in id_to_attr]
        if len(attrs) == 2:
            events.append((int(item["frame_id"]), frozenset(attrs)))
    return events


def match_collisions(
    pred: list[tuple[int, frozenset]],
    truth: list[tuple[int, frozenset]],
    *,
    frame_tol: int = 5,
) -> dict[str, float]:
    used = set()
    tp = 0
    for frame, pair in pred:
        for index, (t_frame, t_pair) in enumerate(truth):
            if index in used:
                continue
            if pair == t_pair and abs(frame - t_frame) <= frame_tol:
                used.add(index)
                tp += 1
                break
    precision = tp / len(pred) if pred else (1.0 if not truth else 0.0)
    recall = tp / len(truth) if truth else (1.0 if not pred else 0.0)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "n_pred": len(pred),
        "n_truth": len(truth),
        "frame_tol": frame_tol,
    }


def load_mask_rcnn(root: str | Path = DEFAULT_CLEVRER_EVAL_DIR, device: str | None = None):
    """Load the downloaded COCO Mask R-CNN. Not a CLEVRER-finetuned parser."""
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn

    root = Path(root)
    ready = json.loads((root / "mask_rcnn" / "READY.json").read_text())
    if ready.get("clevrer_finetuned"):
        raise ValueError("Unexpected CLEVRER-finetuned flag on COCO weights")
    weights_path = root / "mask_rcnn" / "maskrcnn_resnet50_fpn_coco.pth"
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device), device


def frame_to_uint8_hwc(frame_chw: torch.Tensor) -> np.ndarray:
    """Convert a CLEVRER loader frame in [-1, 1] CHW to uint8 HWC."""
    img = ((frame_chw.detach().cpu().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return img.permute(1, 2, 0).numpy()


@torch.no_grad()
def detect_mask_rcnn(
    model,
    frame_chw: torch.Tensor,
    device: str,
    *,
    score_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    rgb = frame_to_uint8_hwc(frame_chw)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    outputs = model([tensor.to(device)])[0]
    detections = []
    scores = outputs["scores"].detach().cpu()
    labels = outputs["labels"].detach().cpu()
    boxes = outputs["boxes"].detach().cpu()
    for score, label, box in zip(scores, labels, boxes):
        if float(score) < score_threshold:
            continue
        class_id = int(label)
        detections.append({
            "score": float(score),
            "label": class_id,
            "name": COCO_NAMES[class_id] if class_id < len(COCO_NAMES) else str(class_id),
            "box": [float(v) for v in box.tolist()],
        })
    return detections


def evaluate_visual_mask(scene_id: int, annotation: dict[str, Any], root: Path) -> dict[str, Any]:
    payload = load_visual_mask(find_proposal_path(root, scene_id))
    attr = set_scores(proposal_attribute_set(payload), annotation_attribute_set(annotation))
    gt_vis = visible_counts_from_annotation(annotation)
    pr_vis = visible_counts_from_proposal(payload, n_frames=len(gt_vis) or 128)
    n = min(len(gt_vis), len(pr_vis))
    mae = float(np.mean(np.abs(np.array(gt_vis[:n]) - np.array(pr_vis[:n])))) if n else 0.0
    return {
        "scene_index": scene_id,
        "attributes": attr,
        "visible_count_mae": mae,
        "n_frames_compared": n,
        "collisions": match_collisions(proposal_collisions(payload), annotation_collisions(annotation)),
    }


def evaluate_propnet_pred(scene_id: int, annotation: dict[str, Any], root: Path) -> dict[str, Any]:
    payload = load_propnet_pred(find_propnet_pred_path(root, scene_id))
    objects = payload.get("objects") or []
    pred_attrs = {_attr_tuple(obj) for obj in objects if all(k in obj for k in ATTR_KEYS)}
    return {
        "scene_index": scene_id,
        "attributes": set_scores(pred_attrs, annotation_attribute_set(annotation)),
        "collisions": match_collisions(
            propnet_collisions(payload),
            annotation_collisions_by_attr(annotation),
        ),
        "n_predictions": len(payload.get("predictions") or []),
    }


def summarize_reports(reports: list[dict[str, Any]], key: str) -> dict[str, float]:
    if not reports:
        return {}
    values = [report[key] for report in reports]
    if isinstance(values[0], dict):
        out = {}
        for field in ("precision", "recall", "f1"):
            if field in values[0]:
                out[field] = float(np.mean([v[field] for v in values]))
        return out
    return {"mean": float(np.mean(values))}


def resolve_eval_root(root: str | Path | None = None) -> Path:
    path = Path(root or DEFAULT_CLEVRER_EVAL_DIR)
    return path if path.is_absolute() else ROOT / path
