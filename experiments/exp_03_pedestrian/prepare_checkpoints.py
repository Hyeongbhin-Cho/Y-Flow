# -*- coding: utf-8 -*-
# experiments/exp_03_pedestrian/prepare_checkpoints.py
"""Download the MoFlow ETH/UCY teacher checkpoints and lay them out for eval_eth.py.

Hugging Face layout:  eth_ucy/moflow/<subset>/{checkpoint_best.pt, cor_fm.yml}
Upstream eval layout: <dir>/cor_fm_updated.yml + <dir>/models/checkpoint_best.pt
(eval_eth.py --cfg auto looks for *_updated.yml next to the models/ folder).

    python prepare_checkpoints.py            # all five subsets
    python prepare_checkpoints.py --subsets eth zara1
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from experiments._layout import CHECKPOINT_ROOT, SUBSETS


REPO = "fyxfelixfu/moflow"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=list(SUBSETS))
    args = p.parse_args()

    local = Path(snapshot_download(REPO, allow_patterns=[f"eth_ucy/moflow/{s}/*" for s in args.subsets]))
    for s in args.subsets:
        src = local / "eth_ucy" / "moflow" / s
        dst = CHECKPOINT_ROOT / s
        (dst / "models").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "checkpoint_best.pt", dst / "models" / "checkpoint_best.pt")
        shutil.copy2(src / "cor_fm.yml", dst / "cor_fm.yml")
        shutil.copy2(src / "cor_fm.yml", dst / "cor_fm_updated.yml")
        print(f"{s}: {dst}")


if __name__ == "__main__":
    main()
