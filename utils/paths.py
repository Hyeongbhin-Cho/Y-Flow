# -*- coding: utf-8 -*-
# utils/paths.py

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEVRER_FLOW_DIR = "checkpoints/clevrer_flow"


def run_name_of(cfg: DictConfig) -> str:
    name = str(cfg.get("run_name", "default"))
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"invalid run_name {name!r}")
    return name


def method_dir(cfg: DictConfig, method: str) -> Path:
    return ROOT / "runs" / run_name_of(cfg) / method


def flowmatch_ckpt(cfg: DictConfig) -> Path:
    return method_dir(cfg, "flowmatch") / "last.pt"


def published_dir(cfg: DictConfig) -> Path | None:
    """Stable reuse directory from model.local_dir, or None if unset."""
    model = cfg.get("model") if cfg is not None else None
    if model is None:
        return None
    local = model.get("local_dir")
    if local is None:
        return None
    text = str(local).strip()
    if not text or text.lower() in ("none", "null"):
        return None
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def published_ckpt(cfg: DictConfig) -> Path | None:
    directory = published_dir(cfg)
    return None if directory is None else directory / "last.pt"


def resolve_flowmatch_ckpt(cfg: DictConfig) -> Path:
    """Prefer runs/{run_name}/flowmatch/last.pt, else published local_dir/last.pt."""
    run_path = flowmatch_ckpt(cfg)
    if run_path.is_file():
        return run_path
    published = published_ckpt(cfg)
    if published is not None and published.is_file():
        return published
    return run_path


def missing_ckpt_message(cfg: DictConfig) -> str:
    run_path = flowmatch_ckpt(cfg)
    published = published_ckpt(cfg)
    msg = f"missing flowmatch checkpoint: {run_path}"
    if published is not None:
        msg += f" (published fallback also missing: {published})"
    return msg
