# -*- coding: utf-8 -*-
# utils/paths.py

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[1]


def run_name_of(cfg: DictConfig) -> str:
    name = str(cfg.get("run_name", "default"))
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"invalid run_name {name!r}")
    return name


def method_dir(cfg: DictConfig, method: str) -> Path:
    return ROOT / "runs" / run_name_of(cfg) / method


def flowmatch_ckpt(cfg: DictConfig) -> Path:
    return method_dir(cfg, "flowmatch") / "last.pt"


def moflow_backbone_run_name(cfg: DictConfig) -> str:
    configured = cfg.moflow.get("backbone_run_name", None)
    name = run_name_of(cfg) if configured is None or str(configured) == "" else str(configured)
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"invalid MoFlow backbone run name {name!r}")
    return name


def moflow_ckpt(cfg: DictConfig) -> Path:
    """Checkpoint for evaluation, optionally decoupled from the output run."""
    name = moflow_backbone_run_name(cfg)
    return ROOT / "runs" / name / "moflow" / "last.pt"
