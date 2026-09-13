# -*- coding: utf-8 -*-
"""Download and save the pinned torchvision ImageNet-1K ResNet-34 weights."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torchvision.models import ResNet34_Weights

DEFAULT_OUTPUT = ROOT / "checkpoints" / "resnet34_imagenet1k_v1.pth"


def ensure_resnet34_weights(output: str | Path = DEFAULT_OUTPUT, *, force: bool = False) -> Path:
    output = Path(output).expanduser().resolve()
    if output.is_file() and not force:
        payload = torch.load(output, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "conv1.weight" not in payload:
            raise ValueError(f"Not a torchvision ResNet-34 state dict: {output}")
        print(f"[Y-Flow] ResNet-34 ImageNet weights already exist: {output}")
        return output

    weights = ResNet34_Weights.IMAGENET1K_V1
    print(f"[Y-Flow] Loading {weights.name} from torchvision: {weights.url}")
    state_dict = weights.get_state_dict(progress=True, check_hash=True)
    # Validate the official tensor set before publishing the local file.
    from torchvision.models import resnet34

    resnet34(weights=None).load_state_dict(state_dict, strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    torch.save(state_dict, temporary)
    os.replace(temporary, output)
    metadata = {
        "architecture": "torchvision.resnet34",
        "weights": "ResNet34_Weights.IMAGENET1K_V1",
        "url": weights.url,
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[Y-Flow] Saved pretrained weights: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="Download again and replace the local file.")
    args = parser.parse_args()
    ensure_resnet34_weights(args.output, force=args.force)


if __name__ == "__main__":
    main()
