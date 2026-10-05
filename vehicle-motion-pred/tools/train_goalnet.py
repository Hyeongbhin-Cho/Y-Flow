from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vfm.config import load_config
from vfm.data import VehicleTrajDataset, load_meta, load_split, to_device
from vfm.goalnet import GoalScorer, losses, propose
from vfm.model import build_model
from vfm.train import pick_device


def loader(cfg, split, batch_size, shuffle, limit=None):
    a = load_split(cfg.data.cache_dir, split, limit)
    return DataLoader(VehicleTrajDataset(a, optional=True), batch_size=batch_size, shuffle=shuffle,
                      drop_last=shuffle)


@torch.no_grad()
def evaluate(net, model, dl, ccfg, horizon_s, dt, device, k):
    net.eval()
    hit, dist, n = 0.0, [], 0
    for batch in dl:
        batch = to_device(batch, device)
        g = propose(net, model, batch, torch.zeros(batch["hist"].shape[0], 1, 2, device=device), k, ccfg,
                    horizon_s, dt)
        fmask = batch["fut_mask"]
        last = fmask.shape[1] - 1 - fmask.flip(-1).float().argmax(-1)
        end = batch["fut"][torch.arange(len(last), device=device), last]
        d = (g - end[:, None]).norm(dim=-1)
        dist.append(d.min(1).values.cpu())
        hit += float((d.min(1).values < 2.0).sum())
        n += len(d)
    net.train()
    d = torch.cat(dist)
    return hit / max(n, 1), float(d.median()), float(d.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k", type=int, default=6, help="goals kept at evaluation time")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_limit", type=int, default=1000)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    device = pick_device(str(cfg.get("device", "cuda")))
    meta = load_meta(cfg.data.cache_dir)
    hz = float(meta["sample_hz"])
    dt, horizon_s = 1.0 / hz, int(meta["future_steps"]) / hz
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_model(cfg, future_steps=int(ck["future_steps"])).to(device)
    model.load_state_dict(ck["ema" if cfg.eval.get("use_ema", True) else "model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ccfg = dict(cfg.get("constraints") or {})
    net = GoalScorer(d_model=int(cfg.model.d_model)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    tr = loader(cfg, cfg.data.train_split, args.batch_size, True)
    ev = loader(cfg, cfg.data.eval_split, 64, False, args.eval_limit)
    out = Path(args.out or (Path(args.ckpt).parent / "goalnet.pt"))
    step, t0, best = 0, time.time(), float("inf")
    while step < args.steps:
        for batch in tr:
            batch = to_device(batch, device)
            ce, reg, frac = losses(net, model, batch, ccfg, horizon_s, dt)
            loss = ce + reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"step {step:5d}  ce {float(ce):.3f}  reg {float(reg):.3f}  usable {float(frac):.2f}  "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                hit, med, mean = evaluate(net, model, ev, ccfg, horizon_s, dt, device, args.k)
                print(f"  eval  goal<2m {100 * hit:.1f}%   median {med:.2f} m   mean {mean:.2f} m", flush=True)
                if med < best:
                    best = med
                    torch.save({"state_dict": net.state_dict(), "d_model": int(cfg.model.d_model),
                                "step": step, "median_m": med}, out)
                    print(f"  saved {out}", flush=True)
            if step >= args.steps:
                break
    print("done. best median goal error", round(best, 3), "m ->", out)


if __name__ == "__main__":
    main()
