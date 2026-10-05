from __future__ import annotations

from pathlib import Path

import yaml


class Cfg(dict):

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Cfg(value) if isinstance(value, dict) and not isinstance(value, Cfg) else value

    def __setattr__(self, name, value):
        self[name] = value


def _wrap(obj):
    if isinstance(obj, dict):
        return Cfg({k: _wrap(v) for k, v in obj.items()})
    return obj


def _unwrap(obj):
    if isinstance(obj, dict):
        return {k: _unwrap(v) for k, v in obj.items()}
    return obj


def load_config(path: str | Path, overrides: list[str] | None = None) -> Cfg:
    cfg = _wrap(yaml.safe_load(Path(path).read_text()))
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, Cfg())
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def dump_config(cfg: Cfg, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(_unwrap(cfg), sort_keys=False, allow_unicode=True))
