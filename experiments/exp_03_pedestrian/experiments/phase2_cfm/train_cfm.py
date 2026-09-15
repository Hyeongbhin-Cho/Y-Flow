# -*- coding: utf-8 -*-
# experiments/exp_03_pedestrian/experiments/phase2_cfm/train_cfm.py
"""Train the plain conditional flow-matching base model on one ETH/UCY split.

Same data pipeline as the MoFlow teacher (SocialGAN splits, rotation at frame 6,
min-max normalization from the train split). No test data are used for model
selection: the EMA weights after the last epoch are saved.

    COMMAND=train_cfm ./run_exp_03_pedestrian.sh --subset zara2
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
from easydict import EasyDict
from torch.utils.data import DataLoader

from data.dataloader_eth_ucy import ETHDataset, seq_collate_eth

from experiments._layout import CFM_ROOT, TASK_ROOT
from experiments.phase1_yflow.run_phase1 import make_logger
from experiments.phase2_cfm.model import CondFlowNet


def base_cfg(subset: str) -> EasyDict:
    return EasyDict(dict(
        subset=subset, dataset="eth_ucy", past_frames=8, future_frames=12, rotate=True, rotate_aug=False,
        data_norm="min_max", agents=1, denoising_head_preds=20, fm_wrapper="velocity",
        MODEL=EasyDict(CONTEXT_ENCODER=EasyDict(AGENTS=1)),
    ))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True, choices=["eth", "hotel", "univ", "zara1", "zara2"])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--cond_dim", type=int, default=256)
    p.add_argument("--n_blocks", type=int, default=6)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--rotate_time_frame", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir) if args.out_dir else CFM_ROOT / args.subset
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = make_logger(out_dir / "train.log")

    cfg = base_cfg(args.subset)
    dset = ETHDataset(cfg=cfg, training=True, data_dir=str(TASK_ROOT / "data" / "eth_ucy"), subset=args.subset,
                      rotate_time_frame=args.rotate_time_frame, type="original")
    loader = DataLoader(dset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, collate_fn=seq_collate_eth)
    stats = {k: float(cfg[k]) for k in ("fut_traj_min", "fut_traj_max", "past_traj_min", "past_traj_max")}
    net_kwargs = dict(past_frames=8, past_feat=6, out_dim=24, hidden=args.hidden, cond_dim=args.cond_dim, n_blocks=args.n_blocks)
    net = CondFlowNet(**net_kwargs).to(device)
    ema = copy.deepcopy(net).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    logger.info(f"subset={args.subset} train={len(dset)} steps/epoch={len(loader)} params={sum(p.numel() for p in net.parameters()):,} device={device}")

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        net.train()
        run = 0.0
        for data in loader:
            past = data["past_traj"].to(device)
            x1 = data["fut_traj"].to(device)
            B, A = x1.shape[:2]
            x1 = x1.reshape(B, 1, A, -1)
            x0 = torch.randn_like(x1)
            t = torch.rand(B, device=device)
            xt = t.view(-1, 1, 1, 1) * x1 + (1 - t.view(-1, 1, 1, 1)) * x0
            v = net(xt, t, past)
            loss = ((v - (x1 - x0)) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.mul_(args.ema).add_(pn.detach(), alpha=1 - args.ema)
            run += float(loss)
            step += 1
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            logger.info(f"epoch {epoch:4d} loss {run / len(loader):.5f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")

    blob = {"model": ema.state_dict(), "net_kwargs": net_kwargs, "cfg": {k: v for k, v in cfg.items() if k != "MODEL"},
            **stats, "args": vars(args), "steps": step}
    blob["cfg"].update(stats)
    torch.save(blob, out_dir / "model.pt")
    (out_dir / "config.json").write_text(json.dumps({"net_kwargs": net_kwargs, "stats": stats, "args": vars(args), "steps": step}, indent=2))
    logger.info(f"saved {out_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
