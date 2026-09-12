# -*- coding: utf-8 -*-
# train/clevrer_flow.py
"""Train video-conditioned CFM for CLEVRER scene-state recognition."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.base import build_dataset
from data.clevrer_state import collate_clevrer_recognition
from model import build_model
from train.checkpoint import save_checkpoint
from train.ema import EMA
from train.flow_match import ConditionalFlowMatching
from utils.device import get_device
from utils.paths import method_dir


def run_train_recognition(
    cfg: DictConfig,
    method: str = "flowmatch",
    device: torch.device | None = None,
) -> Path:
    device = device or get_device(cfg)
    out_dir = method_dir(cfg, method)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    bundle = build_dataset(cfg)
    if bundle.train is None:
        raise FileNotFoundError(
            "clevrer_recognition train split is empty. "
            "Run: python scripts/setup_clevrer.py --splits train"
        )
    loader = DataLoader(
        bundle.train,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        drop_last=True,
        collate_fn=collate_clevrer_recognition,
        num_workers=int(cfg.train.get("num_workers", 0)),
    )
    model = build_model(cfg).to(device)
    cfm = ConditionalFlowMatching(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr))
    ema = EMA(model, decay=float(cfg.train.ema_decay))

    steps = int(cfg.train.steps)
    save_every = int(cfg.log.get("plot_every", 500))
    it = iter(loader)
    model.train()
    pbar = tqdm(range(1, steps + 1), desc="clevrer-recognition")
    for step in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        x1 = batch["state_z"].to(device)
        video = batch["video"].to(device)
        cond = model.encode(video)
        loss = cfm.training_losses(model, x1, cond=cond)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema.update(model)
        pbar.set_postfix(loss=f"{float(loss.item()):.4f}")
        if step % save_every == 0 or step == steps:
            save_checkpoint(
                out_dir / "last.pt",
                model,
                ema,
                opt,
                step,
                cfg,
                extra={"meta": bundle.meta_dict},
            )
    return out_dir / "last.pt"
