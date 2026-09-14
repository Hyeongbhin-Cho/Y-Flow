"""Train the AV2 MoFlow-style K-shot teacher."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.base import build_dataset
from model.moflow import MoFlowAV2, build_moflow_model
from train.checkpoint import save_checkpoint
from train.ema import EMA
from utils.device import get_device
from utils.paths import method_dir


class ContextTrajectoryDataset(Dataset):
    def __init__(self, trajectories: torch.Tensor, context: dict[str, object]):
        self.trajectories = trajectories
        self.context = {name: torch.as_tensor(value) for name, value in context.items()}
        if any(len(value) != len(trajectories) for value in self.context.values()):
            raise ValueError("trajectory and context lengths differ")

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, index: int):
        return self.trajectories[index], {name: value[index] for name, value in self.context.items()}


def _to_device(context: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in context.items()}


def moflow_loss(
    model: MoFlowAV2,
    target: torch.Tensor,
    context: dict[str, torch.Tensor],
    n_modes: int,
    classification_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch, dim = target.shape
    x0 = torch.randn(batch, n_modes, dim, device=target.device, dtype=target.dtype)
    t = torch.rand(batch, device=target.device, dtype=target.dtype)
    target_k = target[:, None, :].expand(-1, n_modes, -1)
    time = t[:, None, None]
    xt = (1.0 - time) * x0 + time * target_k
    true_velocity = target_k - x0
    predicted_velocity, logits = model(xt, t, context)
    endpoint = xt + (1.0 - time) * predicted_velocity
    endpoint_error = (endpoint - target_k).square().mean(dim=-1)
    winner = endpoint_error.detach().argmin(dim=1)
    rows = torch.arange(batch, device=target.device)
    regression = (predicted_velocity[rows, winner] - true_velocity[rows, winner]).square().mean()
    classification = F.cross_entropy(logits, winner)
    loss = regression + float(classification_weight) * classification
    return loss, {
        "regression": float(regression.detach()),
        "classification": float(classification.detach()),
    }


def run_train_moflow(cfg: DictConfig, device: torch.device | None = None) -> Path:
    if str(cfg.data.name) != "autonomous_driving":
        raise ValueError("moflow currently supports data.name=autonomous_driving only")
    device = device or get_device(cfg)
    out_dir = method_dir(cfg, "moflow")
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    bundle = build_dataset(cfg)
    if bundle.train_context is None:
        raise ValueError("moflow requires trajectory context")
    dataset = ContextTrajectoryDataset(bundle.train.trajectories, bundle.train_context)
    loader = DataLoader(dataset, batch_size=int(cfg.train.batch_size), shuffle=True, drop_last=False)
    model = build_moflow_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr))
    ema = EMA(model, decay=float(cfg.train.ema_decay))
    n_modes = int(cfg.moflow.n_modes)
    classification_weight = float(cfg.moflow.get("classification_weight", 0.1))

    iterator = iter(loader)
    model.train()
    pbar = tqdm(range(1, int(cfg.train.steps) + 1), desc="train moflow")
    for step in pbar:
        try:
            target, context = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            target, context = next(iterator)
        target = target.to(device)
        context = _to_device(context, device)
        loss, parts = moflow_loss(model, target, context, n_modes, classification_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.get("grad_clip", 5.0)))
        optimizer.step()
        ema.update(model)
        pbar.set_postfix(loss=f"{loss.item():.4f}", reg=f"{parts['regression']:.4f}")

        save_every = int(cfg.train.get("save_every", cfg.train.steps))
        if step % save_every == 0 or step == int(cfg.train.steps):
            save_checkpoint(
                out_dir / "last.pt", model, ema, optimizer, step, cfg,
                extra={"meta": bundle.meta_dict, "adaptation": "MoFlow-style AV2 focal teacher"},
            )
    return out_dir / "last.pt"
