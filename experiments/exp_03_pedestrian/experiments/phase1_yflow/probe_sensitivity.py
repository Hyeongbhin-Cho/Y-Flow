# -*- coding: utf-8 -*-
# experiments/exp_03_pedestrian/experiments/phase1_yflow/probe_sensitivity.py
"""Does the teacher's clean prediction depend on the flow state y_t?

Along the plain sampling path, at every step the prediction is re-evaluated
with y_t perturbed by delta * N(0, I) (normalized space) and the mean change of
x1_hat is reported in metres. Y-Flow can only steer a model whose curve is
clearly above zero at the steps where projection is active (t >= t_on) and,
above all, at the terminal step. The run also prints the y-embedding dropout
settings stored in the checkpoint config and the implied training drop rate.

    COMMAND=probe ./run_exp_03_pedestrian.sh --subset zara1
    COMMAND=probe ./run_exp_03_pedestrian.sh --subset zara2 --ckpt_dir results_eth_ucy/cor_fm/<tag>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import types
from pathlib import Path

import torch

from experiments._layout import DATASET_ROOT, RUN_ROOT
from experiments.phase1_yflow.run_phase1 import make_logger, setup_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True, choices=["eth", "hotel", "univ", "zara1", "zara2"])
    p.add_argument("--model", choices=["moflow", "cfm"], default="moflow")
    p.add_argument("--ckpt_dir", type=str, default=None)
    p.add_argument("--deltas", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    p.add_argument("--probe_t", type=float, nargs="+", default=[0.9, 0.95, 0.98, 0.99, 0.999],
                   help="extra t values probed on the analytic path x_t = t*x1 + (1-t)*eps, x1 = plain sample")
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--sampling_steps", type=int, default=None)
    p.add_argument("--solver", choices=["euler", "lin_poly"], default=None)
    p.add_argument("--lin_poly_p", type=int, default=5)
    p.add_argument("--lin_poly_long_step", type=int, default=1000)
    p.add_argument("--rotate_time_frame", type=int, default=6)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tag = args.model if args.ckpt_dir is None else Path(args.ckpt_dir).name
    out_dir = Path(args.out_dir) if args.out_dir else RUN_ROOT / "probe_sensitivity" / f"{args.subset}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = make_logger(out_dir / "run.log")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    cfg, loader, denoiser, ckpt = setup_model(args, DATASET_ROOT, logger, device)
    drop = {k: cfg.get(k, None) for k in ("drop_method", "drop_logi_k", "drop_logi_m")}
    logger.info(f"checkpoint {ckpt} | y-embedding dropout in cfg: {drop}")
    F = int(cfg.future_frames)
    fut_min, fut_max = float(cfg.fut_traj_min), float(cfg.fut_traj_max)
    scale = 2.0 / (fut_max - fut_min)

    rows: list[dict] = []
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 1)

    def bwd_sample_t(self, y_t, t, dt, x_data, flag_print=False):
        B = y_t.shape[0]
        bt = torch.full((B,), t, device=self.device, dtype=torch.float)
        preds = self.model_predictions(y_t, x_data, bt, False)
        base = preds.pred_data.reshape(*preds.pred_data.shape[:-1], F, 2)
        for d in args.deltas:
            eps = torch.randn(y_t.shape, generator=gen, device=y_t.device, dtype=y_t.dtype)
            other = self.model_predictions(y_t + d * eps, x_data, bt, False).pred_data
            other = other.reshape(*other.shape[:-1], F, 2)
            change = float((other - base).norm(dim=-1).mean()) / scale
            rows.append({"t": float(t), "dt": float(dt), "delta": d, "mean_abs_change_m": change, "path": "sampler"})
        return y_t + preds.pred_vel * dt, preds.pred_data, preds

    denoiser.bwd_sample_t = types.MethodType(bwd_sample_t, denoiser)
    data = next(iter(loader))
    data = {k: v.to(device) for k, v in data.items()}
    torch.manual_seed(args.seed)
    with torch.no_grad():
        x1_plain, _, _, _, _ = denoiser.sample(data, num_trajs=cfg.denoising_head_preds)
        B = x1_plain.shape[0]
        for t in args.probe_t:
            eps0 = torch.randn(x1_plain.shape, generator=gen, device=device, dtype=x1_plain.dtype)
            y_t = t * x1_plain + (1.0 - t) * eps0
            bt = torch.full((B,), t, device=device, dtype=torch.float)
            base = denoiser.model_predictions(y_t, data, bt, False).pred_data.reshape(*x1_plain.shape[:-1], F, 2)
            for d in args.deltas:
                eps = torch.randn(y_t.shape, generator=gen, device=device, dtype=y_t.dtype)
                other = denoiser.model_predictions(y_t + d * eps, data, bt, False).pred_data.reshape(*x1_plain.shape[:-1], F, 2)
                change = float((other - base).norm(dim=-1).mean()) / scale
                rows.append({"t": float(t), "dt": "", "delta": d, "mean_abs_change_m": change, "path": "analytic"})

    if drop["drop_method"] == "emb" and drop["drop_logi_k"] is not None:
        k, m = float(drop["drop_logi_k"]), float(drop["drop_logi_m"])
        for r in rows:
            r["train_drop_prob"] = 1.0 / (1.0 + math.exp(-k * (r["t"] - m)))
    with (out_dir / "sensitivity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), restval="")
        w.writeheader()
        w.writerows(rows)

    ts = sorted({r["t"] for r in rows if r["path"] == "sampler"})
    marks = [ts[0], ts[len(ts) // 2], *[x for x in ts if x >= 0.3], ts[-1]]
    seen = set()
    for t in marks:
        if t in seen:
            continue
        seen.add(t)
        vals = {r["delta"]: r["mean_abs_change_m"] for r in rows if r["t"] == t and r["path"] == "sampler"}
        extra = next((f" train_drop_prob {r['train_drop_prob']:.4f}" for r in rows if r["t"] == t and r["path"] == "sampler" and "train_drop_prob" in r), "")
        logger.info(f"t={t:.3f} " + " ".join(f"delta {d}: {v:.5f} m" for d, v in vals.items()) + extra)
    terminal = {r["delta"]: r["mean_abs_change_m"] for r in rows if r["t"] == ts[-1] and r["path"] == "sampler"}
    for t in args.probe_t:
        vals = {r["delta"]: r["mean_abs_change_m"] for r in rows if r["path"] == "analytic" and r["t"] == t}
        logger.info(f"[analytic] t={t:.3f} " + " ".join(f"delta {d}: {v:.5f} m" for d, v in vals.items()))
    (out_dir / "summary.json").write_text(json.dumps({"checkpoint": str(ckpt), "dropout_cfg": drop,
                                                        "terminal_step_t": ts[-1], "terminal_change_m": terminal,
                                                        "n_scenes": int(data["batch_size"])}, indent=2, default=str))
    logger.info(f"terminal step t={ts[-1]:.3f}: {terminal}  -> Y-Flow needs this clearly > 0")
    logger.info(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
