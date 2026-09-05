# -*- coding: utf-8 -*-
# train/guide_flow.py
"""GuideFlow backbone. Training-free by default, with two opt-in training modes.

  guidance.enabled : conditional velocity field of Eq. (12) with condition masking, which
                     classifier-free guidance (Eq. 13) needs at sampling time.
  rfe_train.rfe_loss: EBM-unified objective of Eq. (18), which raises the energy of the
                     model's generated endpoint and lowers it on the ground truth.

Both off (default): reuse the frozen FlowMatch checkpoint, so GuideFlow is compared against
the other constrained methods on the same v_theta. See docs/GuideFlow.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.paths import flowmatch_ckpt


def ensure_flowmatch_ckpt(cfg: DictConfig) -> Path:
    path = flowmatch_ckpt(cfg)
    if path.is_file():
        print(f"pretrained flowmatch model already exists: {path}")
        return path
    from train.trainer import run_train

    print(f"missing {path}, training flowmatch")
    return run_train(cfg, method="flowmatch")


def build_conditions(cfg: DictConfig, bundle: dict) -> dict[str, torch.Tensor]:
    from eval.guide_flow import intent_dim_of

    g = cfg.guideflow.guidance
    constraint = bundle["constraint"]
    train_raw = bundle["train_raw"].astype(np.float64)
    mean = bundle["mean"].numpy().astype(np.float64)
    std = bundle["std"].numpy().astype(np.float64)

    anchors_p = constraint.build_anchor_vocabulary(
        train_raw, int(cfg.guideflow.get("n_anchors", 256)), int(cfg.seed)
    )
    d2 = ((train_raw[:, None, :] - anchors_p[None, :, :]) ** 2).sum(axis=-1)
    nearest = d2.argmin(axis=1)
    n_commands = int(g.get("n_commands", 5))

    if str(g.get("signal", "anchor")) == "command":
        idx = constraint.command_bins(anchors_p[nearest], n_commands)
        intent = np.eye(n_commands, dtype=np.float32)[idx]
    else:
        intent = ((anchors_p[nearest] - mean) / std).astype(np.float32)

    return {
        "x1": bundle["train"].points,
        "intent": torch.from_numpy(intent),
        "reward": torch.as_tensor(constraint.progress(train_raw), dtype=torch.float32),
        "intent_dim": intent_dim_of(cfg),
    }


def _velocity(model, use_cond, x, t, intent, reward, intent_mask, reward_mask):
    if use_cond:
        return model(x, t, intent, reward, intent_mask, reward_mask)
    return model(x, t)


def _null_conditions(bundle: dict) -> dict:
    n = bundle["train"].points.shape[0]
    return {
        "x1": bundle["train"].points,
        "intent": torch.zeros(n, 1),
        "reward": torch.zeros(n),
        "intent_dim": 0,
    }


def run_train_guideflow(cfg: DictConfig, device: torch.device | None = None) -> Path:
    from data.base import build_dataset
    from eval.guide_flow import energy_weights_of
    from model import build_model
    from model.cond_mlp import build_cond_model
    from train.checkpoint import save_checkpoint
    from train.ema import EMA
    from train.flow_match import ConditionalFlowMatching
    from utils.device import get_device
    from utils.paths import method_dir

    g = cfg.guideflow.guidance
    rfe = cfg.guideflow.rfe_train
    use_cond = bool(g.get("enabled", False))
    use_rfe_loss = bool(rfe.get("rfe_loss", False))
    device = device or get_device(cfg)
    out_dir = method_dir(cfg, "guideflow")
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    bundle = build_dataset(cfg)
    constraint = bundle["constraint"]
    cond = build_conditions(cfg, bundle) if use_cond else _null_conditions(bundle)
    loader = DataLoader(
        TensorDataset(cond["x1"], cond["intent"], cond["reward"]),
        batch_size=int(g.get("cond_batch_size", cfg.train.batch_size)),
        shuffle=True,
        drop_last=True,
    )

    intent_dim = cond["intent_dim"] if use_cond else 0
    model = (build_cond_model(cfg, intent_dim) if use_cond else build_model(cfg)).to(device)
    method = ConditionalFlowMatching(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=float(g.get("cond_lr", cfg.train.lr)))
    ema = EMA(model, decay=float(cfg.train.ema_decay))

    p_uncond = float(g.get("p_uncond", 0.2))
    use_reward = bool(g.get("reward", True))
    steps = int(g.get("cond_steps", cfg.train.steps))
    lambda_rfe = float(rfe.get("lambda_rfe", 1.0))
    t_min = float(rfe.get("t_min", 0.5))
    rollout_steps = int(rfe.get("rollout_steps", -1))
    n_grid = int(cfg.sample.n_steps)
    n_roll = (
        max(1, int(round((1.0 - t_min) * n_grid))) if rollout_steps < 0 else rollout_steps
    )
    slack = float(cfg.guideflow.get("slack", 0.01))
    weights = energy_weights_of(cfg)
    mean = bundle["mean"].to(device)
    std = bundle["std"].to(device)
    it = iter(loader)
    model.train()
    pbar = tqdm(range(1, steps + 1), desc="train guideflow")
    for step in pbar:
        try:
            x1, intent, reward = next(it)
        except StopIteration:
            it = iter(loader)
            x1, intent, reward = next(it)
        x1 = x1.to(device)
        intent = intent.to(device)
        reward = reward.to(device)

        t = torch.rand(x1.shape[0], device=device, dtype=x1.dtype)
        x0 = torch.randn_like(x1)
        xt, ut = method.interpolant(x0, x1, t)
        intent_mask = (torch.rand(x1.shape[0], device=device) >= p_uncond).to(x1.dtype)
        reward_mask = (torch.rand(x1.shape[0], device=device) >= p_uncond).to(x1.dtype)
        if not use_reward:
            reward_mask = torch.zeros_like(reward_mask)
        v = _velocity(model, use_cond, xt, t, intent, reward, intent_mask, reward_mask)
        loss = (v - ut).square().mean()

        if use_rfe_loss:
            gate = (t >= t_min).to(x1.dtype)
            if rollout_steps == 0:
                x1_hat = xt + (1.0 - t).reshape(-1, 1) * v
            else:
                x1_hat = xt
                tau = t.clone()
                for j in range(n_roll):
                    if rollout_steps < 0:
                        dtau = (1.0 - tau).clamp(min=0.0, max=1.0 / n_grid)
                    else:
                        dtau = (1.0 - tau) / n_roll
                    vv = v if j == 0 else _velocity(
                        model, use_cond, x1_hat, tau, intent, reward, intent_mask, reward_mask
                    )
                    x1_hat = x1_hat + dtau.reshape(-1, 1) * vv
                    tau = tau + dtau
            e_gen = constraint.energy(x1_hat * std + mean, *weights, slack=slack)
            e_gt = constraint.energy(x1 * std + mean, *weights, slack=slack).detach()
            loss = loss + lambda_rfe * (gate * (e_gen - e_gt)).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema.update(model)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if step % int(cfg.log.plot_every) == 0 or step == steps:
            save_checkpoint(
                out_dir / "last.pt",
                model,
                ema,
                opt,
                step,
                cfg,
                extra={
                    "meta": bundle["meta_dict"],
                    "intent_dim": intent_dim,
                    "rfe_loss": use_rfe_loss,
                },
            )
            import os
            if os.environ.get("GF_SNAPSHOT"):
                save_checkpoint(
                    out_dir / f"snap_{step:06d}.pt",
                    model, ema, opt, step, cfg,
                    extra={"meta": bundle["meta_dict"], "intent_dim": intent_dim, "rfe_loss": use_rfe_loss},
                )

    return out_dir / "last.pt"
