from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vfm.config import dump_config, load_config
from vfm.data import VehicleTrajDataset, future_std, load_meta, load_split, to_device
from vfm.flow import fm_loss, sample
from vfm.metrics import summarize
from vfm.model import build_model


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EMA:

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.lerp_(p.detach(), 1.0 - self.decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def run(cfg) -> Path:
    seed_all(int(cfg.seed))
    device = pick_device(str(cfg.get("device", "cuda")))
    out_dir = Path(cfg.out_dir) / str(cfg.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out_dir / "config.yaml")

    meta = load_meta(cfg.data.cache_dir)
    train = load_split(cfg.data.cache_dir, cfg.data.get("train_split", "train"), cfg.data.get("limit_train"))
    std = torch.from_numpy(future_std(train, str(cfg.flow.get("norm", "per_step"))))
    ds = VehicleTrajDataset(train)
    loader = DataLoader(
        ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        drop_last=len(ds) > int(cfg.train.batch_size),
        num_workers=int(cfg.train.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    model = build_model(cfg, future_steps=int(train["fut"].shape[1])).to(device)
    ema = EMA(model, float(cfg.train.ema_decay))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))
    std_d = std.to(device)
    steps = int(cfg.train.steps)
    warmup = int(cfg.train.get("warmup", 1000))
    n_samples = int(cfg.flow.get("train_samples_per_scene", 4))
    schedule = str(cfg.flow.get("t_schedule", "uniform"))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} train={len(ds)} params={n_params/1e6:.2f}M norm={cfg.flow.get('norm', 'per_step')} fut_std[first,last]={std.reshape(-1, 2)[[0, -1]].tolist()}")

    val = None
    if cfg.train.get("select_on_minade", True):
        val = load_split(cfg.data.cache_dir, cfg.data.get("eval_split", "val"), int(cfg.train.get("select_limit", 2000)))
        val_loader = DataLoader(VehicleTrajDataset(val), batch_size=int(cfg.eval.batch_size), shuffle=False)
        sel_k = int(cfg.eval.get("k") or meta.get("eval_k", 6))
        sel_steps = int(cfg.train.get("select_n_steps", cfg.eval.n_steps))
    best = float("inf")

    def select_score() -> float:
        gen = torch.Generator().manual_seed(0)
        preds = [sample(ema.shadow, to_device(b, device), std_d, sel_k, sel_steps, gen).float().cpu().numpy()
                 for b in val_loader]
        m = summarize(np.concatenate(preds), val["fut"], val["fut_mask"], meta)
        return float(m[f"minADE_{sel_k}"])

    log = open(out_dir / "train_log.jsonl", "a")
    it = iter(loader)
    model.train()
    started = time.time()
    running = 0.0
    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = to_device(batch, device)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, warmup, float(cfg.train.lr))
        loss = fm_loss(model, batch, std_d, n_samples, schedule)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
        opt.step()
        ema.update(model)
        running += loss.item()
        if (step + 1) % int(cfg.train.log_every) == 0:
            rec = {"step": step + 1, "loss": running / int(cfg.train.log_every), "lr": opt.param_groups[0]["lr"],
                   "elapsed_s": round(time.time() - started, 1)}
            print(json.dumps(rec))
            log.write(json.dumps(rec) + "\n")
            log.flush()
            running = 0.0
        if (step + 1) % int(cfg.train.save_every) == 0 or step + 1 == steps:
            ckpt = {"model": model.state_dict(), "ema": ema.shadow.state_dict(), "step": step + 1,
                    "fut_std": std, "future_steps": int(train["fut"].shape[1]), "meta": meta, "cfg": dict(cfg)}
            torch.save(ckpt, out_dir / "last.pt")
            if val is not None:
                score = select_score()
                rec = {"step": step + 1, "select_minADE": score, "best": min(best, score)}
                print(json.dumps(rec))
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if score < best:
                    best = score
                    torch.save(ckpt, out_dir / "best.pt")
    log.close()
    return out_dir / ("best.pt" if val is not None else "last.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    print(run(load_config(args.config, args.overrides)))


if __name__ == "__main__":
    main()
