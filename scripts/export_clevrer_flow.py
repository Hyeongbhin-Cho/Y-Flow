# -*- coding: utf-8 -*-
# scripts/export_clevrer_flow.py
"""Copy a trained CLEVRER FlowMatch checkpoint into checkpoints/ for reuse."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

from train.checkpoint import publish_checkpoint
from utils.config import load_config
from utils.paths import DEFAULT_CLEVRER_FLOW_DIR, flowmatch_ckpt, published_dir


def export_clevrer_flow(
    src: str | Path | None = None,
    dst_dir: str | Path | None = None,
    *,
    config_path: str = "configs/exp_02_sub_video_recognition.yaml",
    run_name: str | None = None,
) -> Path:
    cfg = load_config(config_path)
    if run_name:
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"run_name": run_name}))
    elif cfg.get("run_name") in (None, ""):
        cfg = OmegaConf.merge(
            cfg, OmegaConf.create({"run_name": "exp_02_sub_video_recognition"})
        )

    src_path = Path(src) if src is not None else flowmatch_ckpt(cfg)
    if not src_path.is_absolute():
        src_path = ROOT / src_path
    if not src_path.is_file():
        raise FileNotFoundError(
            f"missing run checkpoint {src_path}. "
            "Pass --src or --run_name pointing at runs/<name>/flowmatch/last.pt"
        )

    if dst_dir is not None:
        out_dir = Path(dst_dir)
    else:
        out_dir = published_dir(cfg) or (ROOT / DEFAULT_CLEVRER_FLOW_DIR)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    extra: dict = {"source": str(src_path)}
    payload = torch.load(src_path, map_location="cpu", weights_only=False)
    extra["step"] = payload.get("step")
    saved_cfg = payload.get("cfg") or {}
    model_cfg = saved_cfg.get("model") or {}
    extra["model"] = model_cfg.get("name") or str(cfg.model.get("name", "clevrer_flow"))
    extra["run_name"] = saved_cfg.get("run_name") or str(cfg.get("run_name", ""))

    dst = publish_checkpoint(src_path, out_dir / "last.pt", extra=extra)
    print(f"[Y-Flow] published {src_path} -> {dst}")
    print(f"[Y-Flow] READY.json: {out_dir / 'READY.json'}")
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a trained CLEVRER FlowMatch checkpoint to checkpoints/."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exp_02_sub_video_recognition.yaml",
        help="YAML used to resolve run_name and model.local_dir.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run whose runs/<name>/flowmatch/last.pt is exported.",
    )
    parser.add_argument(
        "--src",
        type=str,
        default=None,
        help="Explicit last.pt path. Overrides --run_name.",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=None,
        help="Destination directory (default: model.local_dir or checkpoints/clevrer_flow).",
    )
    args = parser.parse_args()
    export_clevrer_flow(
        src=args.src,
        dst_dir=args.dst,
        config_path=args.config,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
