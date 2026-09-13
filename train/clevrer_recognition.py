# -*- coding: utf-8 -*-
"""Direct supervised training for the CLEVRER ResNet video recognizer."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.base import build_dataset
from data.clevrer_state import collate_clevrer_recognition, layout_from_cfg, unpack_state
from model import build_model
from train.checkpoint import maybe_publish_checkpoint
from utils.device import get_device
from utils.paths import method_dir


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


class CLEVRERRecognitionCriterion(nn.Module):
    """Hungarian-matched multi-task loss for packed CLEVRER scene labels."""

    def __init__(
        self,
        cfg: DictConfig,
        position_mean: list[float] | tuple[float, ...],
        position_std: list[float] | tuple[float, ...],
        velocity_mean: list[float] | tuple[float, ...],
        velocity_std: list[float] | tuple[float, ...],
        collision_pos_weight: float,
    ) -> None:
        super().__init__()
        self.layout = layout_from_cfg(cfg)
        self.dt = 1.0 / float(cfg.data.get("fps", 16))
        loss_cfg = cfg.get("loss", {})
        self.weights = {
            key: float(loss_cfg.get(key, default))
            for key, default in (
                ("objectness", 1.0),
                ("attributes", 1.0),
                ("visibility", 1.0),
                ("position", 5.0),
                ("velocity", 2.0),
                ("collision", 2.0),
                ("kinematic", 0.1),
            )
        }
        self.match_weights = {
            "objectness": float(loss_cfg.get("match_objectness", 1.0)),
            "attributes": float(loss_cfg.get("match_attributes", 1.0)),
            "position": float(loss_cfg.get("match_position", 5.0)),
        }
        self.register_buffer("position_mean", torch.tensor(position_mean, dtype=torch.float32))
        self.register_buffer("position_std", torch.tensor(position_std, dtype=torch.float32))
        self.register_buffer("velocity_mean", torch.tensor(velocity_mean, dtype=torch.float32))
        self.register_buffer("velocity_std", torch.tensor(velocity_std, dtype=torch.float32))
        self.register_buffer(
            "collision_pos_weight", torch.tensor(float(collision_pos_weight), dtype=torch.float32)
        )

    @torch.no_grad()
    def match(self, outputs: dict[str, torch.Tensor], packed_target: torch.Tensor) -> list[list[tuple[int, int]]]:
        """Return (prediction slot, GT slot) pairs, one clip-level assignment per sample."""
        target = unpack_state(packed_target, self.layout)
        matches: list[list[tuple[int, int]]] = []
        for batch_i in range(packed_target.shape[0]):
            active_gt = torch.where(target["color"][batch_i].sum(dim=-1) > 0.5)[0]
            if active_gt.numel() == 0:
                matches.append([])
                continue
            pred_count = outputs["objectness_logits"].shape[1]
            cost = outputs["objectness_logits"].new_zeros(pred_count, active_gt.numel())
            obj_cost = F.softplus(-outputs["objectness_logits"][batch_i])
            cost = cost + self.match_weights["objectness"] * obj_cost[:, None]
            attr_cost = cost.new_zeros(cost.shape)
            pos_cost = cost.new_zeros(cost.shape)
            for column, gt_slot_tensor in enumerate(active_gt):
                gt_slot = int(gt_slot_tensor)
                attr_cost[:, column] = (
                    F.cross_entropy(
                        outputs["color_logits"][batch_i],
                        target["color"][batch_i, gt_slot].argmax().expand(pred_count),
                        reduction="none",
                    )
                    + F.cross_entropy(
                        outputs["material_logits"][batch_i],
                        target["material"][batch_i, gt_slot].argmax().expand(pred_count),
                        reduction="none",
                    )
                    + F.cross_entropy(
                        outputs["shape_logits"][batch_i],
                        target["shape"][batch_i, gt_slot].argmax().expand(pred_count),
                        reduction="none",
                    )
                )
                visible = target["vis"][batch_i, gt_slot] > 0.5
                if visible.any():
                    pred_pos = outputs["positions"][batch_i]
                    gt_pos = (
                        target["pos"][batch_i, gt_slot] - self.position_mean
                    ) / self.position_std
                    distances = (pred_pos - gt_pos[None]).abs().mean(dim=-1)
                    pos_cost[:, column] = distances[:, visible].mean(dim=-1)
            cost = cost + self.match_weights["attributes"] * attr_cost
            cost = cost + self.match_weights["position"] * pos_cost
            pred_indices, active_columns = linear_sum_assignment(cost.detach().cpu().numpy())
            matches.append(
                [(int(pred), int(active_gt[column])) for pred, column in zip(pred_indices, active_columns)]
            )
        return matches

    def forward(
        self, outputs: dict[str, torch.Tensor], packed_target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[list[tuple[int, int]]]]:
        target = unpack_state(packed_target, self.layout)
        matches = self.match(outputs, packed_target)
        device = packed_target.device
        zero = outputs["objectness_logits"].sum() * 0.0
        components = {key: zero for key in self.weights}

        object_target = torch.zeros_like(outputs["objectness_logits"])
        for batch_i, pairs in enumerate(matches):
            for pred_slot, _gt_slot in pairs:
                object_target[batch_i, pred_slot] = 1.0
        components["objectness"] = F.binary_cross_entropy_with_logits(
            outputs["objectness_logits"], object_target
        )

        attr_losses: list[torch.Tensor] = []
        vis_losses: list[torch.Tensor] = []
        pos_losses: list[torch.Tensor] = []
        vel_losses: list[torch.Tensor] = []
        kin_losses: list[torch.Tensor] = []
        collision_logits: list[torch.Tensor] = []
        collision_targets: list[torch.Tensor] = []
        collision_masks: list[torch.Tensor] = []
        pred_to_gt: list[dict[int, int]] = []

        for batch_i, pairs in enumerate(matches):
            mapping = {pred_slot: gt_slot for pred_slot, gt_slot in pairs}
            pred_to_gt.append(mapping)
            for pred_slot, gt_slot in pairs:
                attr_losses.append(
                    F.cross_entropy(
                        outputs["color_logits"][batch_i, pred_slot][None],
                        target["color"][batch_i, gt_slot].argmax().view(1),
                    )
                    + F.cross_entropy(
                        outputs["material_logits"][batch_i, pred_slot][None],
                        target["material"][batch_i, gt_slot].argmax().view(1),
                    )
                    + F.cross_entropy(
                        outputs["shape_logits"][batch_i, pred_slot][None],
                        target["shape"][batch_i, gt_slot].argmax().view(1),
                    )
                )
                visible = target["vis"][batch_i, gt_slot] > 0.5
                vis_losses.append(
                    F.binary_cross_entropy_with_logits(
                        outputs["visibility_logits"][batch_i, pred_slot],
                        target["vis"][batch_i, gt_slot],
                    )
                )
                if visible.any():
                    pred_pos = outputs["positions"][batch_i, pred_slot]
                    pred_vel = outputs["velocities"][batch_i, pred_slot]
                    gt_pos = (target["pos"][batch_i, gt_slot] - self.position_mean) / self.position_std
                    gt_vel = (target["vel"][batch_i, gt_slot] - self.velocity_mean) / self.velocity_std
                    pos_element = F.smooth_l1_loss(pred_pos, gt_pos, reduction="none").mean(dim=-1)
                    vel_element = F.smooth_l1_loss(pred_vel, gt_vel, reduction="none").mean(dim=-1)
                    pos_losses.append(_masked_mean(pos_element, visible))
                    vel_losses.append(_masked_mean(vel_element, visible))
                adjacent_visible = visible[:-1] & visible[1:]
                if adjacent_visible.any():
                    pred_pos_world = outputs["positions"][batch_i, pred_slot] * self.position_std + self.position_mean
                    pred_vel_world = outputs["velocities"][batch_i, pred_slot] * self.velocity_std + self.velocity_mean
                    finite_difference = (pred_pos_world[1:] - pred_pos_world[:-1]) / self.dt
                    kin_error = F.smooth_l1_loss(finite_difference, pred_vel_world[:-1], reduction="none").mean(dim=-1)
                    kin_losses.append(_masked_mean(kin_error, adjacent_visible))

            pair_i = outputs.get("pair_i")
            pair_j = outputs.get("pair_j")
            if pair_i is None or pair_j is None:
                # pair ordering is i<j; fall back to the canonical layout order.
                pair_i, pair_j = torch.triu_indices(self.layout.n_slots, self.layout.n_slots, 1, device=device)
            for pair_idx, (pred_i, pred_j) in enumerate(zip(pair_i.tolist(), pair_j.tolist())):
                if pred_i not in mapping or pred_j not in mapping:
                    continue
                gt_i, gt_j = mapping[pred_i], mapping[pred_j]
                visible_pair = (target["vis"][batch_i, gt_i] > 0.5) & (target["vis"][batch_i, gt_j] > 0.5)
                if not visible_pair.any():
                    continue
                collision_logits.append(outputs["collision_logits"][batch_i, :, pair_idx])
                collision_targets.append(target["coll"][batch_i, :, gt_i, gt_j])
                collision_masks.append(visible_pair)

        if attr_losses:
            components["attributes"] = torch.stack(attr_losses).mean()
        if vis_losses:
            components["visibility"] = torch.stack(vis_losses).mean()
        if pos_losses:
            components["position"] = torch.stack(pos_losses).mean()
        if vel_losses:
            components["velocity"] = torch.stack(vel_losses).mean()
        if kin_losses:
            components["kinematic"] = torch.stack(kin_losses).mean()
        if collision_logits:
            logits = torch.cat(collision_logits)
            labels = torch.cat(collision_targets)
            mask = torch.cat(collision_masks)
            components["collision"] = _masked_mean(
                F.binary_cross_entropy_with_logits(
                    logits,
                    labels,
                    reduction="none",
                    pos_weight=self.collision_pos_weight,
                ),
                mask,
            )

        total = sum(self.weights[key] * components[key] for key in self.weights)
        return total, components, matches

    def stats(self) -> dict[str, Any]:
        return {
            "position_mean": self.position_mean.detach().cpu().tolist(),
            "position_std": self.position_std.detach().cpu().tolist(),
            "velocity_mean": self.velocity_mean.detach().cpu().tolist(),
            "velocity_std": self.velocity_std.detach().cpu().tolist(),
            "collision_pos_weight": float(self.collision_pos_weight.item()),
        }


def _training_stats(bundle, indices: np.ndarray, cfg: DictConfig) -> dict[str, Any]:
    layout = layout_from_cfg(cfg)
    state = torch.as_tensor(np.asarray(bundle.train_raw)[indices], dtype=torch.float32)
    parts = unpack_state(state, layout)
    visible = parts["vis"] > 0.5
    values: dict[str, Any] = {}
    for key in ("pos", "vel"):
        mask = visible.unsqueeze(-1).expand_as(parts[key])
        selected = parts[key][mask].reshape(-1, 3)
        if selected.numel() == 0:
            raise ValueError(f"no visible {key} targets in the training split")
        mean = selected.mean(dim=0)
        std = selected.std(dim=0, unbiased=False).clamp_min(0.01 if key == "pos" else 0.05)
        values[f"{key}_mean"] = mean.tolist()
        values[f"{key}_std"] = std.tolist()

    visible_t = visible.transpose(1, 2)
    pair_visible = visible_t.unsqueeze(-1) & visible_t.unsqueeze(-2)
    eye = torch.eye(layout.n_slots, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
    eligible = pair_visible & ~eye
    collision = parts["coll"] > 0.5
    positive = int((collision & eligible).sum())
    negative = int((~collision & eligible).sum())
    cap = float(cfg.loss.get("collision_pos_weight_cap", 20.0))
    values["collision_pos_weight"] = min(cap, negative / max(positive, 1))
    return values


def _make_criterion(cfg: DictConfig, stats: dict[str, Any]) -> CLEVRERRecognitionCriterion:
    return CLEVRERRecognitionCriterion(
        cfg,
        stats["pos_mean"],
        stats["pos_std"],
        stats["vel_mean"],
        stats["vel_std"],
        stats["collision_pos_weight"],
    )


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _checkpoint_payload(model, optimizer, scheduler, epoch, step, cfg, criterion, val_loss, split):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "extra": {
            "task": "clevrer_video_recognition",
            "criterion": criterion.stats(),
            "val_loss": float(val_loss),
            "thresholds": {"objectness": 0.5, "visibility": 0.5, "collision": 0.5},
            "split": split,
        },
    }


def _binary_f1_threshold(scores: list[float], labels: list[int]) -> float:
    if not scores or not any(labels):
        return 0.5
    score = np.asarray(scores, dtype=np.float64)
    target = np.asarray(labels, dtype=bool)
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.1, 0.9, 17):
        prediction = score >= threshold
        tp = np.logical_and(prediction, target).sum()
        fp = np.logical_and(prediction, ~target).sum()
        fn = np.logical_and(~prediction, target).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), float(f1)
    return best_threshold


@torch.no_grad()
def _calibrate_thresholds(model, loader, criterion, device) -> dict[str, float]:
    model.eval()
    object_scores: list[float] = []
    object_labels: list[int] = []
    visibility_scores: list[float] = []
    visibility_labels: list[int] = []
    collision_scores: list[float] = []
    collision_labels: list[int] = []
    for batch in tqdm(loader, desc="calibrate", leave=False):
        video = batch["video"].to(device)
        target = batch["state"].to(device)
        output = model(video)
        _, _, matches = criterion(output, target)
        parts = unpack_state(target, criterion.layout)
        probs = output["objectness_logits"].sigmoid()
        for batch_i, pairs in enumerate(matches):
            mapping = {pred_slot: gt_slot for pred_slot, gt_slot in pairs}
            matched_prediction = set(mapping)
            object_scores.extend(probs[batch_i].cpu().tolist())
            object_labels.extend([int(i in matched_prediction) for i in range(probs.shape[1])])
            for pred_slot, gt_slot in pairs:
                vis_prob = output["visibility_logits"][batch_i, pred_slot].sigmoid()
                visibility_scores.extend(vis_prob.cpu().tolist())
                visibility_labels.extend((parts["vis"][batch_i, gt_slot] > 0.5).int().cpu().tolist())
            pair_i, pair_j = output.get("pair_i"), output.get("pair_j")
            if pair_i is None or pair_j is None:
                pair_i, pair_j = torch.triu_indices(
                    criterion.layout.n_slots, criterion.layout.n_slots, 1, device=device
                )
            for pair_idx, (pred_i, pred_j) in enumerate(zip(pair_i.tolist(), pair_j.tolist())):
                if pred_i not in mapping or pred_j not in mapping:
                    continue
                gt_i, gt_j = mapping[pred_i], mapping[pred_j]
                visible = (parts["vis"][batch_i, gt_i] > 0.5) & (parts["vis"][batch_i, gt_j] > 0.5)
                if not visible.any():
                    continue
                collision_scores.extend(output["collision_logits"][batch_i, :, pair_idx].sigmoid()[visible].cpu().tolist())
                collision_labels.extend((parts["coll"][batch_i, :, gt_i, gt_j][visible] > 0.5).int().cpu().tolist())
    return {
        "objectness": _binary_f1_threshold(object_scores, object_labels),
        "visibility": _binary_f1_threshold(visibility_scores, visibility_labels),
        "collision": _binary_f1_threshold(collision_scores, collision_labels),
    }


def run_train_recognition(
    cfg: DictConfig,
    method: str = "recognition",
    device: torch.device | None = None,
) -> Path:
    """Train the direct ResNet recognizer and select checkpoints on a train-only dev split."""
    if str(cfg.data.name).lower() != "clevrer_recognition":
        raise ValueError("the recognition trainer requires data.name=clevrer_recognition")
    if str(cfg.model.name).lower() not in ("clevrer_resnet34", "clevrer_video_recognizer"):
        raise ValueError("the recognition trainer requires model.name=clevrer_resnet34")
    if method != "recognition":
        raise ValueError("CLEVRER direct recognition training must use method='recognition'")
    device = device or get_device(cfg)
    out_dir = method_dir(cfg, method)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    bundle = build_dataset(cfg)
    n_samples = len(bundle.train)
    if np.asarray(bundle.train_raw).shape[0] != n_samples:
        raise ValueError(
            "train_raw annotations and video dataset have different lengths; "
            "ensure every training scene has a matching annotation"
        )
    dev_fraction = float(cfg.train.get("dev_fraction", 0.1))
    if not 0.0 < dev_fraction < 1.0 or n_samples < 2:
        raise ValueError("recognition training requires at least two scenes and 0 < dev_fraction < 1")
    permutation = torch.randperm(n_samples, generator=torch.Generator().manual_seed(int(cfg.seed))).numpy()
    n_dev = min(max(int(round(n_samples * dev_fraction)), 1), n_samples - 1)
    dev_indices, train_indices = permutation[:n_dev], permutation[n_dev:]

    loader_options = {
        "batch_size": int(cfg.train.batch_size),
        "num_workers": int(cfg.train.get("num_workers", 0)),
        "collate_fn": collate_clevrer_recognition,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        Subset(bundle.train, train_indices.tolist()), shuffle=True, drop_last=False, **loader_options
    )
    dev_loader = DataLoader(
        Subset(bundle.train, dev_indices.tolist()), shuffle=False, drop_last=False, **loader_options
    )
    stats = _training_stats(bundle, train_indices, cfg)
    criterion = _make_criterion(cfg, stats).to(device)
    model = build_model(cfg).to(device)

    backbone_params = [p for name, p in model.named_parameters() if p.requires_grad and name.startswith("layer3.")]
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]
    param_groups = [{"params": head_params, "lr": float(cfg.train.get("lr", 3e-4))}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": float(cfg.train.get("backbone_lr", 1e-5))})
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=float(cfg.train.get("weight_decay", 0.01)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg.train.get("lr_factor", 0.5)),
        patience=int(cfg.train.get("lr_patience", 2)),
    )

    max_epochs = int(cfg.train.get("epochs", 30))
    patience = int(cfg.train.get("early_stopping_patience", 8))
    grad_clip = float(cfg.train.get("grad_clip", 1.0))
    best_loss = math.inf
    stale_epochs = 0
    global_step = 0
    best_path, last_path = out_dir / "best.pt", out_dir / "last.pt"
    log_rows: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_totals: dict[str, float] = {}
        train_batches = 0
        progress = tqdm(train_loader, desc=f"recognition train {epoch}/{max_epochs}", leave=False)
        for batch in progress:
            video = batch["video"].to(device, non_blocking=True)
            target = batch["state"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(video)
            loss, components, _matches = criterion(outputs, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            global_step += 1
            train_batches += 1
            train_totals["loss"] = train_totals.get("loss", 0.0) + float(loss.detach())
            for key, value in components.items():
                train_totals[key] = train_totals.get(key, 0.0) + float(value.detach())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        dev_totals: dict[str, float] = {}
        dev_batches = 0
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"recognition dev {epoch}/{max_epochs}", leave=False):
                video = batch["video"].to(device, non_blocking=True)
                target = batch["state"].to(device, non_blocking=True)
                outputs = model(video)
                loss, components, _matches = criterion(outputs, target)
                dev_totals["loss"] = dev_totals.get("loss", 0.0) + float(loss)
                for key, value in components.items():
                    dev_totals[key] = dev_totals.get(key, 0.0) + float(value)
                dev_batches += 1
        if train_batches == 0 or dev_batches == 0:
            raise RuntimeError("recognition train/dev split produced an empty loader")
        train_mean = {key: value / train_batches for key, value in train_totals.items()}
        dev_mean = {key: value / dev_batches for key, value in dev_totals.items()}
        scheduler.step(dev_mean["loss"])
        row = {"epoch": epoch, "step": global_step, "train": train_mean, "dev": dev_mean}
        log_rows.append(row)
        print(json.dumps(row))

        payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            epoch,
            global_step,
            cfg,
            criterion,
            dev_mean["loss"],
            {"seed": int(cfg.seed), "train_indices": train_indices.tolist(), "dev_indices": dev_indices.tolist()},
        )
        _atomic_save(payload, last_path)
        if dev_mean["loss"] < best_loss:
            best_loss = dev_mean["loss"]
            stale_epochs = 0
            _atomic_save(payload, best_path)
        else:
            stale_epochs += 1
        (out_dir / "history.json").write_text(json.dumps(log_rows, indent=2) + "\n")
        if stale_epochs >= patience:
            print(f"early stopping after {epoch} epochs; best dev loss={best_loss:.6f}")
            break

    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model"])
    thresholds = _calibrate_thresholds(model, dev_loader, criterion, device)
    best_payload["extra"]["thresholds"] = thresholds
    _atomic_save(best_payload, best_path)
    # Publish the development-selected model as the reusable default checkpoint.
    published = maybe_publish_checkpoint(
        cfg,
        best_path,
        extra={"epoch": best_payload["epoch"], "task": "clevrer_video_recognition", "thresholds": thresholds},
    )
    if published is not None:
        print(f"published best development checkpoint to {published}")
    print(f"best recognition checkpoint: {best_path}; dev thresholds: {thresholds}")
    return best_path
