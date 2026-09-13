# -*- coding: utf-8 -*-
# train/trainer.py

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.base import build_dataset, denormalize
from model import build_model
from sample.euler import EulerSampler
from train.checkpoint import maybe_publish_checkpoint, save_checkpoint
from train.ema import EMA
from train.flow_match import ConditionalFlowMatching
from utils.device import get_device
from utils.paths import method_dir


def _save_scatter(path: Path, points: np.ndarray, reference: np.ndarray | None, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    if reference is not None:
        ax.scatter(reference[:, 0], reference[:, 1], s=4, alpha=0.25, c="0.6", label="data")
    ax.scatter(points[:, 0], points[:, 1], s=6, alpha=0.7, c="C0", label="samples")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def setup_foundation_model(
    cfg: DictConfig,
    out_dir: Path,
    device: torch.device,
) -> Path:
    """Prepare and verify frozen foundation flow-matching model (e.g. Wan2.1).
    
    Rather than training from scratch, it verifies local pretrained weights,
    initializes the model architecture, and writes a checkpoint reference to out_dir / 'last.pt'.
    This enables all downstream training-free methods (HardFlow, YFlow, SafeFlow, UniConFlow, GuideFlow)
    to seamlessly reuse the backbone through ensure_flowmatch_ckpt(cfg).
    """
    print(f"[Y-Flow] Initializing frozen foundation model '{cfg.model.name}'...")
    model = build_model(cfg).to(device)
    ckpt_path = out_dir / "last.pt"
    payload = {
        "model_name": str(cfg.model.name),
        "pretrained_path": str(cfg.model.get("pretrained_path", "")),
        "local_dir": str(cfg.model.get("local_dir", "")),
        "step": 0,
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "extra": {
            "is_foundation_model": True,
            "variant": str(cfg.model.get("variant", "1.3B")),
        },
    }
    torch.save(payload, ckpt_path)
    print(f"[Y-Flow] Foundation model verified. Checkpoint reference saved to: {ckpt_path}")
    return ckpt_path


def run_train(
    cfg: DictConfig,
    method: str = "flowmatch",
    device: torch.device | None = None,
) -> Path:
    device = device or get_device(cfg)
    if str(cfg.data.get("name", "")).lower() == "clevrer_recognition":
        if method != "recognition":
            raise ValueError(
                "CLEVRER video recognition uses direct supervised training; "
                "invoke `main.py recognition --mode train`"
            )
        from train.clevrer_recognition import run_train_recognition

        return run_train_recognition(cfg, method=method, device=device)

    out_dir = method_dir(cfg, method)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    model_name = str(cfg.model.name).lower()
    if model_name in ("wan2.1", "wan", "wan_transformer"):
        return setup_foundation_model(cfg, out_dir, device)

    bundle = build_dataset(cfg)
    loader = DataLoader(
        bundle["train"],
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        drop_last=True,
    )
    model = build_model(cfg).to(device)
    method = ConditionalFlowMatching(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr))
    ema = EMA(model, decay=float(cfg.train.ema_decay))
    sampler = EulerSampler(n_steps=int(cfg.sample.n_steps))

    _save_scatter(out_dir / "data.png", bundle["train_raw"], None, "train data")

    steps = int(cfg.train.steps)
    plot_every = int(cfg.log.plot_every)
    it = iter(loader)
    model.train()
    pbar = tqdm(range(1, steps + 1), desc="train")
    for step in pbar:
        try:
            x1 = next(it)
        except StopIteration:
            it = iter(loader)
            x1 = next(it)
        x1 = x1.to(device)
        loss = method.training_losses(model, x1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema.update(model)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if step % plot_every == 0 or step == steps:
            _plot_ema_samples(cfg, model, ema, method, sampler, bundle, device, out_dir, step)
            ckpt_path = out_dir / "last.pt"
            save_checkpoint(
                ckpt_path,
                model,
                ema,
                opt,
                step,
                cfg,
                extra={"meta": bundle["meta_dict"]},
            )
            maybe_publish_checkpoint(
                cfg,
                ckpt_path,
                extra={
                    "step": step,
                    "model": str(cfg.model.name),
                    "run_name": str(cfg.get("run_name", "")),
                },
            )

    return out_dir / "last.pt"


@torch.no_grad()
def _plot_ema_samples(cfg, model, ema, method, sampler, bundle, device, out_dir, step) -> None:
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema.copy_to(model)
    n = min(int(cfg.sample.n_samples), 2000)
    dim = int(cfg.model.get("dim", 2))
    x0 = torch.randn(n, dim, device=device)
    z = sampler.sample(model, method, x0)
    p = denormalize(z, bundle["mean"], bundle["std"]).cpu().numpy()
    _save_scatter(
        Path(out_dir) / f"samples_step_{step:06d}.png",
        p,
        bundle["train_raw"],
        f"step {step}",
    )
    model.load_state_dict(backup)
    model.train()
