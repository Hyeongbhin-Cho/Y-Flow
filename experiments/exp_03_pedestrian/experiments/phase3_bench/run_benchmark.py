# -*- coding: utf-8 -*-
# experiments/exp_03_pedestrian/experiments/phase3_bench/run_benchmark.py
"""Pedestrian-forecasting benchmark of the Y-Flow repository methods.

FlowMatch (unconstrained), HardFlow, YFlow, SafeFlow, UniConFlow, GuideFlow
(training-free) with the hyper-parameters of configs/exp_03_pedestrian.yaml,
on the ETH/UCY SocialGAN test split, conditioned on the observed past through
the pretrained MoFlow teacher (or --model cfm). Uniform Euler grid with
sample.n_steps steps, matched noise per seed. Metrics: min20 ADE/FDE,
avg ADE/FDE, KDE-NLL, COL@0.1/0.2 against co-present agents' GT futures
(evaluation only), APD, kinematic violation, latency; plus a
constant-velocity baseline.

    COMMAND=bench ./run_exp_03_pedestrian.sh --subset zara1
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from experiments._layout import DATASET_ROOT, REPOSITORY_ROOT, RUN_ROOT
from experiments.phase1_yflow.constraints import KinematicConstraint
from experiments.phase1_yflow.neighbors import NeighbourTable, rotation_matrices
from experiments.phase1_yflow.run_phase1 import (
    FRAME_ERR_FAIL_M, kinematic_limits, make_logger, ped_metrics, scene_metrics, setup_model, summarize,
)
from experiments.phase3_bench.methods import SAMPLERS, ForecastConstraint, StepModel, anchor_vocabulary

ORDER = ("FLOWMATCH", "HARDFLOW", "YFLOW", "SAFEFLOW", "UNICONFLOW", "GUIDEFLOW")
KEYS = ("min_ade", "min_fde", "avg_ade", "avg_fde", "kde_nll", "col010_sample_rate", "col020_sample_rate",
        "apd", "viol_sample_rate", "max_viol", "latency_s", "distortion_vs_flowmatch_m", "dist_to_posthoc_m")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True, choices=["eth", "hotel", "univ", "zara1", "zara2"])
    p.add_argument("--model", choices=["moflow", "cfm"], default="moflow")
    p.add_argument("--ckpt_dir", type=str, default=None)
    p.add_argument("--config", type=str, default=str(REPOSITORY_ROOT / "configs" / "exp_03_pedestrian.yaml"))
    p.add_argument("--methods", nargs="+", default=list(ORDER), choices=list(ORDER))
    p.add_argument("--steps", type=int, default=None, help="default: sample.n_steps of the config (100)")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed_base", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--limit_batches", type=int, default=None)
    p.add_argument("--calibration", choices=["traj", "elem"], default="traj")
    p.add_argument("--quantile", type=float, default=99.5)
    p.add_argument("--v_max", type=float, default=None)
    p.add_argument("--a_max", type=float, default=None)
    p.add_argument("--rotate_time_frame", type=int, default=6)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--save_raw", action="store_true")
    p.add_argument("--no_posthoc", action="store_true",
                   help="skip POSTHOC_PROJ: exact projection of the FLOWMATCH output after sampling (equivalence reference)")
    a = p.parse_args()
    a.solver, a.sampling_steps, a.lin_poly_p, a.lin_poly_long_step = "euler", 10, 5, 1000   # loader setup only
    return a


def train_anchor_pool(subset: str, rtf: int) -> tuple[np.ndarray, np.ndarray]:
    with (DATASET_ROOT / subset / f"{subset}_train.pkl").open("rb") as f:
        traj = np.asarray(pickle.load(f)["traj"], dtype=np.float64)
    R = rotation_matrices(traj, rtf, rotate=True)
    fut = np.einsum("nij,nfj->nfi", R, traj[:, 8:] - traj[:, 7:8])
    last_step = np.einsum("nij,nj->ni", R, traj[:, 7] - traj[:, 6])
    return fut, last_step


def main() -> None:
    args = parse_args()
    cfg_yaml = yaml.safe_load(Path(args.config).read_text())
    steps = int(args.steps or cfg_yaml.get("sample", {}).get("n_steps", 100))
    out_dir = Path(args.out_dir) if args.out_dir else RUN_ROOT / "bench" / args.model / args.subset
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    logger = make_logger(out_dir / "run.log")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"bench subset={args.subset} model={args.model} steps={steps} (uniform Euler) methods={args.methods} config={args.config}")

    cfg, loader, denoiser, ckpt = setup_model(args, DATASET_ROOT, logger, device)
    F = int(cfg.future_frames)
    fut_min, fut_max = float(cfg.fut_traj_min), float(cfg.fut_traj_max)
    scale = 2.0 / (fut_max - fut_min)
    offset = -1.0 - scale * fut_min
    limits, limit_info = kinematic_limits(args, DATASET_ROOT)
    logger.info(f"limits v_max={limits.v_max:.3f} a_max={limits.a_max:.3f} | scale={scale:.4f}")
    constraint_m = KinematicConstraint(limits, F)
    fc = ForecastConstraint(limits, scale, offset, F,
                            fmbf_eps=float(cfg_yaml["safeflow"].get("smooth_radius_eps", 1e-2)),
                            fmbf_margin=float(cfg_yaml["safeflow"].get("smooth_tube_margin", 2e-3)))
    neighbours = NeighbourTable(DATASET_ROOT / args.subset / f"{args.subset}_test.pkl", rotate=True,
                                rotate_time_frame=args.rotate_time_frame)

    anchors_z = None
    if "GUIDEFLOW" in args.methods:
        fut, last_step = train_anchor_pool(args.subset, args.rotate_time_frame)
        n_anch = int(cfg_yaml["guideflow"].get("n_anchors", 256))
        anchors_p = anchor_vocabulary(fut, last_step, limits, n_anch, int(cfg_yaml.get("seed", 0)))
        anchors_z = torch.as_tensor(anchors_p * scale + offset, device=device, dtype=torch.float32)
        logger.info(f"GuideFlow anchor vocabulary: {anchors_p.shape[0]} feasible train futures (FPS)")

    tied = bool(cfg.get("tied_noise", False))
    posthoc_on = (not args.no_posthoc) and "FLOWMATCH" in args.methods
    K = int(cfg.denoising_head_preds)
    raw_rows: list[dict] = []
    scene_offset = 0
    for i_batch, data in enumerate(loader):
        if args.limit_batches is not None and i_batch >= args.limit_batches:
            break
        data = {k: v.to(device) for k, v in data.items()}
        gt_m = data["fut_traj_original_scale"].reshape(-1, F, 2)
        rel = data["past_traj_original_scale"][..., 2:4]
        v0 = (rel[:, :, -1] - rel[:, :, -2]).reshape(-1, 2)
        B = gt_m.shape[0]
        A = int(cfg.agents)
        idx = data["indexes"].reshape(-1).cpu().numpy().astype(np.int64)
        err = float(np.abs(neighbours.own_future(idx) - gt_m.cpu().numpy()).max())
        if err > FRAME_ERR_FAIL_M:
            raise RuntimeError(f"neighbour frame mismatch {err:.4g} m")
        qa, ma = neighbours.gather(idx)
        q_all = torch.as_tensor(qa, device=device, dtype=gt_m.dtype)
        mask_all = torch.as_tensor(ma, device=device)
        p0 = torch.zeros_like(v0)
        p_prev, _ = constraint_m.clip_anchor(p0, p0 - v0)
        fc.bind(p_prev, K * A)
        gt_met = scene_metrics(gt_m[:, None], gt_m, v0, constraint_m)
        gt_feasible = gt_met["viol_any"] < 0.5

        pred_cv = v0[:, None, None, :] * torch.arange(1, F + 1, device=device, dtype=gt_m.dtype)[None, None, :, None]
        met = scene_metrics(pred_cv, gt_m, v0, constraint_m)
        met.update(ped_metrics(pred_cv, gt_m, q_all, mask_all))
        for j in range(B):
            raw_rows.append({"method": "CONST_VEL", "seed": 0, "scene": scene_offset + j,
                             **{k: float(met[k][j]) for k in met}, "latency_s": 0.0,
                             "distortion_vs_flowmatch_m": float("nan"), "dist_to_posthoc_m": float("nan"),
                             "gt_feasible": bool(gt_feasible[j])})

        step = StepModel(denoiser, data, B, K, A)
        for s_idx in range(args.seeds):
            seed = args.seed_base + 1000 * s_idx + i_batch
            base = None
            posthoc = None
            for m in args.methods:
                torch.manual_seed(seed)
                x0 = torch.randn((B, K, A, 2 * F), device=device)
                if tied:
                    x0 = x0[:, :1].expand(-1, K, -1, -1).contiguous()
                x0 = x0.reshape(B * K * A, 2 * F)
                diag: dict = {}
                t0 = time.perf_counter()
                with torch.no_grad():
                    if m == "GUIDEFLOW":
                        x1 = SAMPLERS[m](step, fc, x0, cfg_yaml, steps, diag, anchors_z)
                    else:
                        x1 = SAMPLERS[m](step, fc, x0, cfg_yaml, steps, diag)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                lat = (time.perf_counter() - t0) / B
                pred_m = fc.to_p(x1).reshape(B, K * A, F, 2)
                if m == "FLOWMATCH":
                    base = pred_m
                    if posthoc_on:
                        t1 = time.perf_counter()
                        with torch.no_grad():
                            ph = fc.to_p(fc.project_physical(x1)).reshape(B, K * A, F, 2)
                        ph_lat = (time.perf_counter() - t1) / B + lat
                        posthoc = ph
                dist = ((pred_m - base).norm(dim=-1).mean(dim=(1, 2)).cpu().numpy() if base is not None
                        else np.full(B, np.nan))
                dph = ((pred_m - posthoc).norm(dim=-1).mean(dim=(1, 2)).cpu().numpy() if posthoc is not None
                       else np.full(B, np.nan))
                met = scene_metrics(pred_m, gt_m, v0, constraint_m)
                met.update(ped_metrics(pred_m, gt_m, q_all, mask_all))
                for j in range(B):
                    raw_rows.append({"method": m, "seed": s_idx, "scene": scene_offset + j,
                                     **{k: float(met[k][j]) for k in met}, "latency_s": lat,
                                     "distortion_vs_flowmatch_m": float(dist[j]), "dist_to_posthoc_m": float(dph[j]),
                                     "gt_feasible": bool(gt_feasible[j])})
                if m == "FLOWMATCH" and posthoc_on:
                    met_ph = scene_metrics(posthoc, gt_m, v0, constraint_m)
                    met_ph.update(ped_metrics(posthoc, gt_m, q_all, mask_all))
                    d_ph = (posthoc - base).norm(dim=-1).mean(dim=(1, 2)).cpu().numpy()
                    for j in range(B):
                        raw_rows.append({"method": "POSTHOC_PROJ", "seed": s_idx, "scene": scene_offset + j,
                                         **{k: float(met_ph[k][j]) for k in met_ph}, "latency_s": ph_lat,
                                         "distortion_vs_flowmatch_m": float(d_ph[j]), "dist_to_posthoc_m": 0.0,
                                         "gt_feasible": bool(gt_feasible[j])})
                logger.info(f"batch {i_batch} seed {s_idx} {m:11s} minADE {met['min_ade'].mean():.4f} minFDE {met['min_fde'].mean():.4f} "
                            f"NLL {np.nanmean(met['kde_nll']):.3f} COL@0.2 {met['col020_sample_rate'].mean():.4f} "
                            f"viol {met['viol_sample_rate'].mean():.4f} APD {met['apd'].mean():.3f} "
                            f"dist {np.nanmean(dist):.4f} to_posthoc {np.nanmean(dph):.5f} {lat * B:.1f}s {json.dumps({k: round(v, 4) for k, v in diag.items()})}")
        scene_offset += B

    methods_sum = list(args.methods) + (["POSTHOC_PROJ"] if posthoc_on else []) + ["CONST_VEL"]
    summary = summarize(raw_rows, methods_sum, list(KEYS))
    for flag, tag in ((True, "gt_feasible"), (False, "gt_infeasible")):
        sub = [r for r in raw_rows if r["gt_feasible"] == flag]
        if sub:
            for r in summarize(sub, methods_sum, ["min_ade", "min_fde"]):
                r["metric"] = f"{r['metric']}|{tag}"
                summary.append(r)
    with (out_dir / "results" / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "metric", "count", "mean", "std", "median", "ci95_low", "ci95_high"], extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)
    if args.save_raw:
        with (out_dir / "results" / "raw.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
            w.writeheader()
            w.writerows(raw_rows)
    (out_dir / "results" / "config.json").write_text(json.dumps(
        {"args": vars(args), "steps": steps, "limits": limit_info, "checkpoint": str(ckpt), "tied_noise": tied,
         "method_config": {k: cfg_yaml.get(k) for k in ("hardflow", "yflow", "safeflow", "uniconflow", "guideflow")}},
        indent=2, default=str))

    get = {(r["method"], r["metric"]): r["mean"] for r in summary}
    lines = ["| method | minADE | minFDE | avgADE | KDE-NLL | COL@0.1 | COL@0.2 | APD | viol | latency s/scene | Δ vs FLOWMATCH m | Δ vs POSTHOC m |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m in methods_sum:
        g = lambda k, nd=4: (f"{get[(m, k)]:.{nd}f}" if (m, k) in get else "–")
        lines.append(f"| {m} | {g('min_ade')} | {g('min_fde')} | {g('avg_ade')} | {g('kde_nll', 3)} | {g('col010_sample_rate')} | "
                     f"{g('col020_sample_rate')} | {g('apd', 3)} | {g('viol_sample_rate')} | {g('latency_s')} | "
                     f"{g('distortion_vs_flowmatch_m')} | {g('dist_to_posthoc_m', 5)} |")
    (out_dir / "results" / "table.md").write_text("\n".join(lines) + "\n")
    logger.info("\n" + "\n".join(lines))
    logger.info(f"wrote {out_dir / 'results'}")


if __name__ == "__main__":
    main()
