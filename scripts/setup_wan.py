# -*- coding: utf-8 -*-
# scripts/setup_wan.py
"""Setup, download, and verification script for Wan2.1 video flow-matching foundation model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

from model.download import download_wan_model, is_already_downloaded
from model.wan import build_wan_model
from train.trainer import setup_foundation_model
from utils.config import load_config
from utils.device import get_device
from utils.paths import method_dir


def setup_wan(
    config_path: str = "configs/exp_02_video.yaml",
    subcomponent: str = "core",
    force_download: bool = False,
    run_name: str | None = None,
) -> Path:
    cfg = load_config(config_path)
    if run_name:
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"run_name": run_name}))

    local_dir = Path(cfg.model.get("local_dir", "checkpoints/Wan2.1-T2V-1.3B"))
    repo_id = str(cfg.model.get("pretrained_path", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"))

    print("=" * 65)
    print(" [Y-Flow] Wan2.1 Foundation Model Setup & Verification")
    print(f"  Config: {config_path}")
    print(f"  Model:  {cfg.model.name} ({cfg.model.get('variant', '1.3B')})")
    print(f"  Target: {local_dir}")
    print("=" * 65)

    # 1. Download / Verify weights
    if not is_already_downloaded(local_dir, subcomponent=subcomponent) or force_download:
        print(f"[Y-Flow] Downloading weights ({subcomponent}) from {repo_id}...")
        download_wan_model(
            repo_id=repo_id,
            local_dir=local_dir,
            subcomponent=subcomponent,
            force_download=force_download,
        )
    else:
        print(f"[Y-Flow] Weights already available at: {local_dir}")

    # 2. Verify model architecture and parameters
    print("[Y-Flow] Verifying model loading...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_wan_model(cfg)
    total_params = sum(p.numel() for p in model.transformer.parameters()) / 1e9
    print(f"[Y-Flow] WanTransformer3DModel successfully loaded: {total_params:.2f}B parameters.")
    if model.vae is not None:
        vae_params = sum(p.numel() for p in model.vae.parameters()) / 1e6
        print(f"[Y-Flow] AutoencoderKLWan (3D Causal VAE) loaded: {vae_params:.2f}M parameters.")

    # 3. Create flowmatch checkpoint bundle under runs/{run_name}/flowmatch/last.pt
    out_dir = method_dir(cfg, "flowmatch")
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    ckpt_path = setup_foundation_model(cfg, out_dir, device=device)
    print(f"[Y-Flow] Setup complete! FlowMatch backbone checkpoint ready at: {ckpt_path}")
    print("=" * 65)
    return ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup and verify Wan2.1 model for Y-Flow.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exp_02_video.yaml",
        help="Path to experiment config file.",
    )
    parser.add_argument(
        "--subcomponent",
        type=str,
        default="core",
        choices=["core", "transformer", "vae", "all"],
        help="Components to ensure: 'core' (DiT+VAE, ~3GB), 'transformer', 'vae', 'all'.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run name override (e.g. exp_02_video).",
    )
    args = parser.parse_args()

    setup_wan(
        config_path=args.config,
        subcomponent=args.subcomponent,
        force_download=args.force,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
