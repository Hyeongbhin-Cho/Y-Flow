# -*- coding: utf-8 -*-
# utils/config.py

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    return OmegaConf.load(path)


def flatten_leaves(cfg: DictConfig) -> dict[str, Any]:
    container = OmegaConf.to_container(cfg, resolve=True)
    out: dict[str, Any] = {}

    def rec(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for key, val in obj.items():
                rec(val, f"{path}.{key}" if path else str(key))
            return
        out[path] = obj

    rec(container, "")
    return out


def _leaf_aliases(dotted: dict[str, Any]) -> dict[str, str]:
    names = [path.split(".")[-1] for path in dotted]
    counts = Counter(names)
    aliases: dict[str, str] = {}
    for path in dotted:
        leaf = path.split(".")[-1]
        if counts[leaf] == 1 and leaf != path:
            aliases[leaf] = path
    return aliases


def _coerce(raw: str, reference: Any) -> Any:
    if isinstance(reference, bool):
        return raw.lower() in ("1", "true", "yes", "y")
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(float(raw))
    if isinstance(reference, float):
        return float(raw)
    if isinstance(reference, list):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            parsed = [item.strip() for item in raw.split(",") if item.strip()]
        if reference and parsed:
            elem_t = type(reference[0])
            parsed = [elem_t(x) if not isinstance(x, elem_t) else x for x in parsed]
        return parsed
    return raw


def add_config_override_args(parser: argparse.ArgumentParser, cfg: DictConfig) -> None:
    dotted = flatten_leaves(cfg)
    aliases = _leaf_aliases(dotted)
    seen: set[str] = set()
    for path in dotted:
        if path in seen:
            continue
        seen.add(path)
        parser.add_argument(
            f"--{path}",
            default=None,
            help=f"override {path} (yaml default: {dotted[path]!r})",
        )
    for leaf, path in aliases.items():
        if leaf in seen:
            continue
        seen.add(leaf)
        parser.add_argument(
            f"--{leaf}",
            default=None,
            help=f"override {path} (yaml default: {dotted[path]!r})",
        )


def apply_overrides(cfg: DictConfig, args: argparse.Namespace) -> DictConfig:
    dotted = flatten_leaves(cfg)
    aliases = _leaf_aliases(dotted)
    provided = {k: v for k, v in vars(args).items() if v is not None}

    updates: dict[str, Any] = {}
    for key, raw in provided.items():
        if key in dotted:
            path = key
        elif key in aliases:
            path = aliases[key]
        else:
            continue
        updates[path] = _coerce(str(raw), dotted[path])

    if not updates:
        return cfg
    overlay = OmegaConf.create({})
    for path, value in updates.items():
        OmegaConf.update(overlay, path, value, merge=True)
    return OmegaConf.merge(cfg, overlay)
