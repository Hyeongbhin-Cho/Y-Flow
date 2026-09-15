# -*- coding: utf-8 -*-
# experiments/exp_03_pedestrian/experiments/phase1_yflow/run_phase1.py
"""Phase 1: Y-Flow kinematic terminal projection on the pretrained MoFlow teacher.

The upstream MoFlow sampler already uses the Y-Flow update
    x1_raw = pred_data,  v = (x1_raw - y_t) / (1 - t),  y_{t+dt} = y_t + v * dt,
so this runner only replaces ``pred_data`` by its projection onto the
kinematic feasible set once t >= t_on (and always at the terminal step).
The upstream code is not modified; ``bwd_sample_t`` is overridden per method.

Run from experiments/exp_03_pedestrian (or via run_exp_03_pedestrian.sh):
    COMMAND=phase1 ./run_exp_03_pedestrian.sh --subset zara1
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pickle
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader

from data.dataloader_eth_ucy import ETHDataset, seq_collate_eth
from models.backbone_eth_ucy import ETHMotionTransformer
from models.flow_matching import FlowMatcher
from utils.normalization import unnormalize_min_max

from experiments._layout import CFM_ROOT, CHECKPOINT_ROOT, DATASET_ROOT, RUN_ROOT, TASK_ROOT
from experiments.phase0_gt_audit.run_audit import kinematics
from experiments.phase1_yflow.constraints import CollisionKinematicConstraint, KinematicConstraint, KinematicLimits
from experiments.phase1_yflow.neighbors import NeighbourTable
from experiments.phase1_yflow.sampling import (
    damped_target,
    is_terminal_step,
    yflow_next_state,
    yflow_projection_active,
)


DT = 0.4
FRAME_ERR_WARN_M = 1e-3
FRAME_ERR_FAIL_M = 5e-2
METHODS = {
    "PLAIN_FM": dict(active=False),
    "FINAL_PROJECTION": dict(active=True, t_on=1.0, damping=1.0, replace=True),
    "YFLOW": dict(active=True, t_on=None, damping=1.0, replace=True),
    "YFLOW_NO_REPLACE": dict(active=True, t_on=None, damping=1.0, replace=False),
    "YFLOW_D05": dict(active=True, t_on=None, damping=0.5, replace=True),
    "POV_ALWAYS": dict(active=True, t_on=0.0, damping=1.0, replace=True),
}
CFG_FALLBACKS = {
    "rotate": True,
    "rotate_aug": False,
    "data_norm": "min_max",
    "tied_noise": True,
    "fm_in_scaling": True,
    "past_frames": 8,
    "future_frames": 12,
    "dataset": "eth_ucy",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True, choices=["eth", "hotel", "univ", "zara1", "zara2"])
    p.add_argument("--model", choices=["moflow", "cfm"], default="moflow",
                   help="moflow: released teacher; cfm: phase-2 plain conditional flow matching (x_t-responsive)")
    p.add_argument("--ckpt_dir", type=str, default=None,
                   help="moflow: dir with (models/)checkpoint_best.pt and yml; cfm: dir with model.pt")
    p.add_argument("--data_dir", type=str, default=None, help="dir containing <subset>/<subset>_{train,test}.pkl")
    p.add_argument("--methods", nargs="+", default=["PLAIN_FM", "FINAL_PROJECTION", "YFLOW", "YFLOW_NO_REPLACE"], choices=list(METHODS),
                   help="POV_ALWAYS (projection at every step) is optional: with lin_poly/100 steps it is ~15x slower")
    p.add_argument("--constraint", choices=["kin", "kin_col"], default="kin_col",
                   help="kin: speed/acc only; kin_col: + clearance to neighbours' GT futures (planning setting)")
    p.add_argument("--r_safe", type=float, default=0.2, help="clearance radius (m) for kin_col")
    p.add_argument("--neighbour_source", choices=["gt", "cv"], default="gt",
                   help="kin_col obstacles: gt = neighbours' GT futures (planning, oracle); "
                        "cv = constant-velocity extrapolation of their observed past (forecasting, no oracle)")
    p.add_argument("--seeds", type=int, default=1, help="matched noise seeds per batch")
    p.add_argument("--seed_base", type=int, default=42)
    p.add_argument("--calibration", choices=["traj", "elem"], default="traj")
    p.add_argument("--quantile", type=float, default=99.5)
    p.add_argument("--v_max", type=float, default=None, help="override (m/s)")
    p.add_argument("--a_max", type=float, default=None, help="override (m/s^2)")
    p.add_argument("--t_on", type=float, default=0.5)
    p.add_argument("--admm_iters", type=int, default=100)
    p.add_argument("--admm_iters_terminal", type=int, default=300)
    p.add_argument("--admm_tol", type=float, default=1e-5, help="terminal ADMM tolerance (m)")
    p.add_argument("--admm_tol_guide", type=float, default=1e-3, help="non-terminal ADMM tolerance (m)")
    p.add_argument("--seal_buffer", type=float, default=1e-4, help="relative radius shrink of the terminal seal")
    p.add_argument("--sampling_steps", type=int, default=None, help="default: moflow 100, cfm 20")
    p.add_argument("--solver", choices=["euler", "lin_poly"], default=None, help="default: moflow lin_poly, cfm euler")
    p.add_argument("--lin_poly_p", type=int, default=5)
    p.add_argument("--lin_poly_long_step", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=1000)
    p.add_argument("--limit_batches", type=int, default=None, help="smoke test")
    p.add_argument("--rotate_time_frame", type=int, default=6)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--save_raw", action="store_true", help="write per-scene raw.csv (large for univ)")
    return p.parse_args()


# ----------------------------------------------------------------------------- setup
def make_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("exp03_phase1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(levelname)5s  %(message)s")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path)):
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.propagate = False
    return logger


def find_checkpoint(ckpt_dir: Path) -> tuple[Path, Path]:
    cands = [ckpt_dir / "models" / "checkpoint_best.pt", ckpt_dir / "checkpoint_best.pt"]
    ckpt = next((c for c in cands if c.is_file()), None)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint_best.pt under {ckpt_dir}; run prepare_checkpoints.py first")
    ymls = sorted(ckpt_dir.glob("*_updated.yml")) or sorted(ckpt_dir.glob("*.yml"))
    if not ymls:
        raise FileNotFoundError(f"no yml config under {ckpt_dir}")
    return ckpt, ymls[0]


def load_cfg(yml: Path, args: argparse.Namespace, logger: logging.Logger) -> EasyDict:
    cfg = EasyDict(yaml.safe_load(yml.read_text()))
    for k, v in CFG_FALLBACKS.items():
        if k not in cfg:
            logger.warning(f"cfg key {k!r} missing in {yml.name}; falling back to {v!r}")
            cfg[k] = v
    if "fm_wrapper" not in cfg:
        raise KeyError("fm_wrapper missing from the checkpoint yml; refusing to guess")
    cfg.subset = args.subset
    cfg.sampling_steps = args.sampling_steps
    cfg.solver = args.solver
    cfg.lin_poly_p = args.lin_poly_p
    cfg.lin_poly_long_step = args.lin_poly_long_step
    cfg.train_batch_size = args.batch_size
    cfg.test_batch_size = args.batch_size
    cfg.denoising_method = cfg.get("denoising_method", "fm")
    return cfg


def check_normalization(stored: dict, cfg: EasyDict, logger: logging.Logger) -> None:
    """The checkpoint yml may carry the normalization statistics of its training data.

    If they differ from the ones recomputed from our train split, the checkpoint
    was trained on another data version (e.g. LED instead of SocialGAN) and every
    input is mis-normalized: stop instead of producing silent garbage.
    """
    logger.info(f"checkpoint cfg: {stored}")
    src = stored.get("data_source")
    if src is not None and str(src) != "original":
        raise RuntimeError(f"checkpoint trained with data_source={src!r}; this runner loads 'original' (SocialGAN) splits")
    for k in ("fut_traj_min", "fut_traj_max", "past_traj_min", "past_traj_max"):
        if stored.get(k) is None:
            logger.warning(f"{k} not stored in the checkpoint yml; data-version match cannot be verified")
            continue
        a, b = float(stored[k]), float(cfg[k])
        if abs(a - b) > 1e-3 * max(1.0, abs(a)):
            raise RuntimeError(f"{k}: checkpoint {a:.4f} vs recomputed {b:.4f}; data version or subset mismatch")
    logger.info("normalization statistics match the checkpoint config")


def build_loaders(cfg: EasyDict, args: argparse.Namespace, data_dir: Path):
    data_dir = TASK_ROOT / "data" / "eth_ucy"
    # the training dataset call sets cfg.fut_traj_min/max and cfg.agents (upstream behaviour)
    train_dset = ETHDataset(cfg=cfg, training=True, data_dir=str(data_dir), subset=cfg.subset,
                            rotate_time_frame=args.rotate_time_frame, type="original")
    test_dset = ETHDataset(cfg=cfg, training=False, data_dir=str(data_dir), subset=cfg.subset,
                           rotate_time_frame=args.rotate_time_frame, type="original")
    loader = DataLoader(test_dset, batch_size=cfg.test_batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=seq_collate_eth)
    return train_dset, test_dset, loader


def resolve_sampler_defaults(args: argparse.Namespace) -> None:
    if args.solver is None:
        args.solver = "lin_poly" if args.model == "moflow" else "euler"
    if args.sampling_steps is None:
        args.sampling_steps = 100 if args.model == "moflow" else 20


def setup_model(args: argparse.Namespace, data_dir: Path, logger: logging.Logger, device: torch.device):
    """Returns (cfg, loader, denoiser, ckpt) for --model moflow|cfm with the shared ETHDataset pipeline."""
    resolve_sampler_defaults(args)
    if args.model == "moflow":
        ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else CHECKPOINT_ROOT / args.subset
        ckpt, yml = find_checkpoint(ckpt_dir)
        cfg = load_cfg(yml, args, logger)
        stored = {k: cfg.get(k, None) for k in ("data_source", "fut_traj_min", "fut_traj_max", "past_traj_min", "past_traj_max", "rotate", "subset")}
        _, _, loader = build_loaders(cfg, args, data_dir)
        check_normalization(stored, cfg, logger)
        cfg.device = str(device)
        denoiser = build_denoiser(cfg, ckpt, device, logger)
    else:
        from experiments.phase2_cfm.model import CFMSampler, CondFlowNet
        ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else CFM_ROOT / args.subset
        ckpt = ckpt_dir / "model.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"{ckpt} missing; run COMMAND=train_cfm first")
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = EasyDict(blob["cfg"])
        cfg.MODEL = EasyDict(CONTEXT_ENCODER=EasyDict(AGENTS=1))
        stored = {k: blob[k] for k in ("fut_traj_min", "fut_traj_max", "past_traj_min", "past_traj_max")}
        stored.update(data_source="original", rotate=cfg.rotate, subset=cfg.subset)
        if cfg.subset != args.subset:
            raise RuntimeError(f"cfm checkpoint trained on {cfg.subset}, requested {args.subset}")
        cfg.sampling_steps = args.sampling_steps
        cfg.solver = args.solver
        cfg.lin_poly_p = args.lin_poly_p
        cfg.lin_poly_long_step = args.lin_poly_long_step
        cfg.train_batch_size = args.batch_size
        cfg.test_batch_size = args.batch_size
        _, _, loader = build_loaders(cfg, args, data_dir)
        check_normalization(stored, cfg, logger)
        cfg.device = str(device)
        net = CondFlowNet(**blob["net_kwargs"])
        net.load_state_dict(blob["model"])
        denoiser = CFMSampler(cfg, net).to(device).eval()
        logger.info(f"cfm checkpoint {ckpt} ({blob.get('steps')} steps, {sum(p.numel() for p in net.parameters()):,} params)")
    logger.info(f"model={args.model} solver={args.solver} steps={args.sampling_steps}")
    return cfg, loader, denoiser, ckpt


def build_denoiser(cfg: EasyDict, ckpt: Path, device: torch.device, logger: logging.Logger) -> FlowMatcher:
    model = ETHMotionTransformer(model_config=cfg.MODEL, logger=logger, config=cfg)
    denoiser = FlowMatcher(cfg, model, logger=logger)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    agent_keys = [k for k in state["model"] if k.endswith("agent_order_embedding.weight")]
    if agent_keys:
        logger.info(f"checkpoint {agent_keys[0]} shape {tuple(state['model'][agent_keys[0]].shape)} "
                    f"(rows = agents per sample at training; SocialGAN 'original' split gives 1)")
    denoiser.load_state_dict(state["model"])
    denoiser.to(device).eval()
    return denoiser


def kinematic_limits(args: argparse.Namespace, data_dir: Path) -> tuple[KinematicLimits, dict]:
    with (data_dir / args.subset / f"{args.subset}_train.pkl").open("rb") as f:
        traj = np.asarray(pickle.load(f)["traj"], dtype=np.float64)
    speed, acc = kinematics(traj)
    if args.calibration == "traj":
        speed, acc = speed.max(axis=1), acc.max(axis=1)
    v_max = float(np.percentile(speed, args.quantile)) if args.v_max is None else args.v_max
    a_max = float(np.percentile(acc, args.quantile)) if args.a_max is None else args.a_max
    info = {"v_max": v_max, "a_max": a_max, "calibration": args.calibration, "quantile": args.quantile,
            "v_max_override": args.v_max, "a_max_override": args.a_max}
    return KinematicLimits(v_max=v_max, a_max=a_max, dt=DT), info


# ---------------------------------------------------------------------- sampler hook
class ProjectionContext:
    """Per-batch anchors in normalized space and diagnostics of one sampling run."""

    def __init__(self, constraint_z: KinematicConstraint, scale: float, offset: float, F: int):
        self.constraint = constraint_z
        self.scale = scale
        self.offset = offset
        self.F = F
        self.reset()

    def reset(self):
        self.p0 = None
        self.p_prev = None
        self.q = None
        self.mask = None
        self.state = None
        self.active_steps = 0
        self.correction_sum = 0.0
        self.correction_count = 0
        self.iters = []
        self.residuals = []
        self.clipped_frac = None
        self.last_raw = None
        self.terminal_res = None

    def set_neighbours(self, q_z, mask):
        self.q = q_z
        self.mask = mask

    def set_batch(self, x_data: dict):
        past = x_data["past_traj_original_scale"]                     # [B, A, P, 6] (abs, rel, vel)
        rel = past[..., 2:4]
        v0_disp = rel[:, :, -1] - rel[:, :, -2]                        # [B, A, 2] metres per frame (rotated frame)
        B, A, _ = v0_disp.shape
        p0 = torch.full((B, 1, A, 2), self.offset, device=past.device, dtype=past.dtype)
        p_prev = p0 - self.scale * v0_disp[:, None]
        p_prev, clipped = self.constraint.clip_anchor(p0, p_prev)
        self.p0, self.p_prev = p0, p_prev
        self.state = None
        self.clipped_frac = float(clipped.float().mean())


def install_yflow(denoiser: FlowMatcher, ctx: ProjectionContext, spec: dict, args: argparse.Namespace):
    F = ctx.F

    def bwd_sample_t(self, y_t, t, dt, x_data, flag_print=False):
        B = y_t.shape[0]
        batched_t = torch.full((B,), t, device=self.device, dtype=torch.float)
        preds = self.model_predictions(y_t, x_data, batched_t, flag_print)
        raw = preds.pred_data
        terminal = is_terminal_step(t, dt)
        if terminal:
            ctx.last_raw = raw
        if spec["active"] and yflow_projection_active(t, dt, spec["t_on"]) and (spec["replace"] or not terminal):
            x = raw.reshape(*raw.shape[:-1], F, 2)
            n_iters = args.admm_iters_terminal if terminal else args.admm_iters
            if ctx.q is not None:
                n = ctx.constraint.normals(x, ctx.q)
                proj, iters, res, ctx.state = ctx.constraint.project_mixed(
                    x, ctx.p0, ctx.p_prev, ctx.q, ctx.mask, n, n_iters=n_iters, tol=(args.admm_tol if terminal else args.admm_tol_guide) * ctx.scale, warm=ctx.state)
            else:
                proj, iters, res, ctx.state = ctx.constraint.project(
                    x, ctx.p0, ctx.p_prev, n_iters=n_iters, tol=(args.admm_tol if terminal else args.admm_tol_guide) * ctx.scale, warm=ctx.state)
            if terminal:
                ctx.terminal_res = res / ctx.scale
            if terminal:
                proj = ctx.constraint.project_feasible(proj, ctx.p0, ctx.p_prev, buffer=args.seal_buffer)
            star = damped_target(raw, proj.reshape(raw.shape), spec["damping"])
            ctx.active_steps += 1
            ctx.iters.append(int(iters))
            ctx.residuals.append(float(res.max()) / ctx.scale)
            corr = ((star - raw).reshape(*raw.shape[:-1], F, 2).norm(dim=-1).mean(dim=-1)) / ctx.scale
            ctx.correction_sum += float(corr.sum())
            ctx.correction_count += int(corr.numel())
        else:
            star = raw
        y_next = yflow_next_state(y_t, star, t, dt)
        return y_next, star, preds

    denoiser.bwd_sample_t = types.MethodType(bwd_sample_t, denoiser)


# ------------------------------------------------------------------------- metrics
def pairwise_apd(pred: torch.Tensor) -> torch.Tensor:
    """pred [N, K, F, 2] -> [N] mean pairwise L2 distance (averaged over frames)."""
    K = pred.shape[1]
    if K < 2:
        return torch.zeros(pred.shape[0], device=pred.device, dtype=pred.dtype)
    d = (pred[:, :, None] - pred[:, None]).norm(dim=-1).mean(dim=-1)      # [N, K, K]
    return d.sum(dim=(1, 2)) / (K * (K - 1))


def scene_metrics(pred_m: torch.Tensor, gt_m: torch.Tensor, v0_disp_m: torch.Tensor,
                  constraint_m: KinematicConstraint, q_m=None, mask=None, r_safe: float = 0.0) -> dict[str, np.ndarray]:
    """pred_m [N, K, F, 2] metres, gt_m [N, F, 2], v0_disp_m [N, 2] metres/frame,
    q_m [N, M, F, 2] neighbour futures (metres, agent frame), mask [N, M]."""
    dist = (pred_m - gt_m[:, None]).norm(dim=-1)                          # [N, K, F]
    ade = dist.mean(dim=-1)
    fde = dist[..., -1]
    p0 = torch.zeros_like(v0_disp_m)[:, None]                             # [N, 1, 2]
    p_prev_raw = p0 - v0_disp_m[:, None]
    p_prev_clip, clipped = constraint_m.clip_anchor(p0, p_prev_raw)
    viol_clip = constraint_m.max_violation(pred_m, p0, p_prev_clip)       # [N, K]
    viol_raw = constraint_m.max_violation(pred_m, p0, p_prev_raw)
    out = {
        "min_ade": ade.min(dim=1).values.cpu().numpy(),
        "min_fde": fde.min(dim=1).values.cpu().numpy(),
        "avg_ade": ade.mean(dim=1).cpu().numpy(),
        "viol_sample_rate": (viol_clip > 1e-6).float().mean(dim=1).cpu().numpy(),
        "viol_any": (viol_clip > 1e-6).any(dim=1).float().cpu().numpy(),
        "viol_sample_rate_raw_anchor": (viol_raw > 1e-6).float().mean(dim=1).cpu().numpy(),
        "max_viol": viol_clip.clamp_min(0.0).max(dim=1).values.cpu().numpy(),
        "apd": pairwise_apd(pred_m).cpu().numpy(),
        "anchor_clipped": clipped.reshape(-1).float().cpu().numpy(),
    }
    if q_m is not None:
        d = (pred_m[:, :, None] - q_m[:, None]).norm(dim=-1)              # [N, K, M, F]
        d = torch.where(mask[:, None, :, None], d, torch.full_like(d, 1e9))
        clear = d.flatten(2).min(dim=-1).values                            # [N, K]
        col = clear < (r_safe - 1e-3)
        has_nb = mask.any(dim=1)[:, None]
        nan = torch.full_like(clear, float("nan"))
        clear_rep = torch.where(has_nb, clear, nan)
        any_viol = col | (viol_clip > 1e-6)
        out.update({
            "col_sample_rate": col.float().mean(dim=1).cpu().numpy(),
            "col_any": col.any(dim=1).float().cpu().numpy(),
            "min_clearance": clear_rep.min(dim=1).values.cpu().numpy(),
            "mean_clearance": clear_rep.mean(dim=1).cpu().numpy(),
            "all_viol_sample_rate": any_viol.float().mean(dim=1).cpu().numpy(),
            "all_viol_any": any_viol.any(dim=1).float().cpu().numpy(),
        })
    return out


KDE_LOGPDF_FLOOR = -20.0      # Trajectron++ evaluation convention


def kde_nll(pred_m: torch.Tensor, gt_m: torch.Tensor, reg: float = 1e-4) -> torch.Tensor:
    """Per-scene KDE negative log-likelihood of the GT, averaged over future frames.

    Gaussian KDE with Scott's bandwidth on the K samples of each frame (as
    scipy.stats.gaussian_kde / Trajectron++), log-pdf clipped at -20, plus a
    small covariance regularizer for near-degenerate sample sets. NaN if K < 2.
    """
    N, K, F, _ = pred_m.shape
    if K < 2:
        return torch.full((N,), float("nan"), device=pred_m.device)
    x = pred_m.permute(0, 2, 1, 3).double()                                # [N, F, K, 2]
    xc = x - x.mean(dim=2, keepdim=True)
    cov = xc.transpose(-1, -2) @ xc / (K - 1)                               # [N, F, 2, 2]
    eye = torch.eye(2, device=x.device, dtype=x.dtype)
    H = cov * (K ** (-2.0 / 6.0)) + reg * eye
    diff = gt_m[:, :, None, :].double() - x                                 # [N, F, K, 2]
    maha = torch.einsum("nfki,nfij,nfkj->nfk", diff, torch.linalg.inv(H), diff)
    logp = torch.logsumexp(-0.5 * maha, dim=-1) - math.log(K) - 0.5 * torch.logdet(H) - math.log(2 * math.pi)
    return (-logp.clamp_min(KDE_LOGPDF_FLOOR)).mean(dim=-1).float()


def ped_metrics(pred_m: torch.Tensor, gt_m: torch.Tensor, q_all, mask_all) -> dict[str, np.ndarray]:
    """Pedestrian-forecasting metrics beyond min ADE/FDE.

    COL@r: a predicted trajectory collides if it comes within r of any
    co-present agent's ground-truth future at the same frame (Social-NCE style,
    evaluation only; the forecasting methods never see these futures).
    """
    dist = (pred_m - gt_m[:, None]).norm(dim=-1)
    out = {"avg_fde": dist[..., -1].mean(dim=1).cpu().numpy(),
           "kde_nll": kde_nll(pred_m, gt_m).cpu().numpy()}
    if q_all is not None:
        d = (pred_m[:, :, None] - q_all[:, None]).norm(dim=-1)              # [N, K, M, F]
        d = torch.where(mask_all[:, None, :, None], d, torch.full_like(d, 1e9))
        clear = d.flatten(2).min(dim=-1).values                              # [N, K]
        for r, tag in ((0.1, "col010"), (0.2, "col020")):
            c = clear < r
            out[f"{tag}_sample_rate"] = c.float().mean(dim=1).cpu().numpy()
            out[f"{tag}_any"] = c.any(dim=1).float().cpu().numpy()
    return out


def summarize(raw_rows: list[dict], methods: list[str], keys: list[str]) -> list[dict]:
    out = []
    by_method = {m: [r for r in raw_rows if r["method"] == m] for m in methods}
    plain = {(r["scene"], r["seed"]): r for r in by_method.get("PLAIN_FM", [])}
    for m in methods:
        rows = by_method[m]
        for k in keys:
            v = np.asarray([r[k] for r in rows], dtype=np.float64)
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            n = v.size
            std = float(v.std(ddof=1)) if n > 1 else 0.0
            out.append({"method": m, "metric": k, "count": n, "mean": float(v.mean()), "std": std,
                        "median": float(np.median(v)), "ci95_low": float(v.mean() - 1.96 * std / math.sqrt(max(n, 1))),
                        "ci95_high": float(v.mean() + 1.96 * std / math.sqrt(max(n, 1)))})
        if m != "PLAIN_FM" and plain:
            for k in ("min_ade", "min_fde", "apd"):
                d = np.asarray([r[k] - plain[(r["scene"], r["seed"])][k] for r in rows if (r["scene"], r["seed"]) in plain])
                if d.size:
                    std = float(d.std(ddof=1)) if d.size > 1 else 0.0
                    out.append({"method": m, "metric": f"paired_delta_{k}_vs_plain", "count": int(d.size), "mean": float(d.mean()),
                                "std": std, "median": float(np.median(d)),
                                "ci95_low": float(d.mean() - 1.96 * std / math.sqrt(d.size)),
                                "ci95_high": float(d.mean() + 1.96 * std / math.sqrt(d.size))})
    return out


# ----------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else DATASET_ROOT
    out_dir = Path(args.out_dir) if args.out_dir else RUN_ROOT / "phase1_yflow" / f"{args.subset}_{args.model}"
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    logger = make_logger(out_dir / "run.log")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"device={device} subset={args.subset} model={args.model} data_dir={data_dir}")

    cfg, loader, denoiser, ckpt = setup_model(args, data_dir, logger, device)
    if int(cfg.agents) != 1:
        raise RuntimeError(f"expected single-agent ETH/UCY layout (A=1), got agents={cfg.agents}")
    F = int(cfg.future_frames)
    fut_min, fut_max = float(cfg.fut_traj_min), float(cfg.fut_traj_max)
    scale = 2.0 / (fut_max - fut_min)                 # z = scale * p + offset
    offset = -1.0 - scale * fut_min
    logger.info(f"normalization: fut_traj_min={fut_min:.4f} fut_traj_max={fut_max:.4f} scale={scale:.4f}")

    limits, limit_info = kinematic_limits(args, data_dir)
    logger.info(f"limits: v_max={limits.v_max:.3f} m/s, a_max={limits.a_max:.3f} m/s^2 ({limit_info['calibration']} q{limit_info['quantile']})")
    constraint_m = KinematicConstraint(limits, F)
    use_col = args.constraint == "kin_col"
    limits_z = KinematicLimits(limits.v_max * scale, limits.a_max * scale, DT)
    neighbours = NeighbourTable(data_dir / args.subset / f"{args.subset}_test.pkl",
                                rotate=bool(cfg.rotate), rotate_time_frame=args.rotate_time_frame)
    if use_col:
        constraint_z = CollisionKinematicConstraint(limits_z, args.r_safe * scale, F)
        what = "GT futures (planning setting, oracle)" if args.neighbour_source == "gt" else "constant-velocity forecasts from observed past (no oracle)"
        logger.info(f"constraint: kin + clearance r_safe={args.r_safe} m to neighbours' {what}")
    else:
        constraint_z = KinematicConstraint(limits_z, F)
        logger.info("constraint: kin (speed/acc only); neighbours used for COL evaluation only")
    ctx = ProjectionContext(constraint_z, scale, offset, F)

    upstream_bwd = denoiser.bwd_sample_t
    methods = list(args.methods)
    if "PLAIN_FM" not in methods:
        methods = ["PLAIN_FM"] + methods
    specs = {}
    for m in methods:
        s = dict(METHODS[m])
        if s.get("t_on") is None and s["active"]:
            s["t_on"] = args.t_on
        specs[m] = s

    keys = ["min_ade", "min_fde", "avg_ade", "viol_sample_rate", "viol_any", "viol_sample_rate_raw_anchor",
            "max_viol", "apd", "anchor_clipped", "correction_m", "active_steps", "admm_iters_mean", "admm_residual_max", "latency_s",
            "terminal_raw_shift_m", "distortion_vs_plain_m", "proj_fail_rate",
            "avg_fde", "kde_nll", "col010_sample_rate", "col010_any", "col020_sample_rate", "col020_any"]
    if use_col:
        keys += ["col_sample_rate", "col_any", "min_clearance", "mean_clearance", "all_viol_sample_rate", "all_viol_any"]
    raw_rows: list[dict] = []
    gt_rows: list[dict] = []
    scene_offset = 0
    for i_batch, data in enumerate(loader):
        if args.limit_batches is not None and i_batch >= args.limit_batches:
            break
        data = {k: v.to(device) for k, v in data.items()}
        gt_m = data["fut_traj_original_scale"].reshape(-1, F, 2)          # [N, F, 2] (A=1 for ETH/UCY)
        v0_disp_m = (data["past_traj_original_scale"][..., 2:4][:, :, -1] - data["past_traj_original_scale"][..., 2:4][:, :, -2]).reshape(-1, 2)
        N = gt_m.shape[0]
        q_m = mask_t = q_eval = mask_eval = None
        idx = data["indexes"].reshape(-1).cpu().numpy().astype(np.int64)
        own = neighbours.own_future(idx)
        frame_err = float(np.abs(own - gt_m.cpu().numpy()).max())
        if frame_err > FRAME_ERR_FAIL_M:
            raise RuntimeError(f"neighbour frame mismatch vs loader: max |own - gt| = {frame_err:.4g} m")
        if frame_err > FRAME_ERR_WARN_M:
            logger.warning(f"batch {i_batch}: neighbour frame max err {frame_err:.2e} m (float32 rotation in the loader; "
                           f"< {FRAME_ERR_FAIL_M} m tolerated, negligible vs clearance radii)")
        qa_np, ma_np = neighbours.gather(idx)
        q_all = torch.as_tensor(qa_np, device=device, dtype=gt_m.dtype)
        mask_all = torch.as_tensor(ma_np, device=device)
        if use_col:
            reach = limits.v_max * DT * np.arange(1, F + 1) + args.r_safe
            q_np, mask_np = neighbours.gather(idx, reach=reach, source=args.neighbour_source)
            q_m = torch.as_tensor(q_np, device=device, dtype=gt_m.dtype)
            mask_t = torch.as_tensor(mask_np, device=device)
            if args.neighbour_source == "gt":
                q_eval, mask_eval = q_all, mask_all
            else:
                qc_np, mc_np = neighbours.gather(idx, source="cv")
                q_eval = torch.as_tensor(qc_np, device=device, dtype=gt_m.dtype)
                mask_eval = torch.as_tensor(mc_np, device=device)
        if i_batch == 0:
            logger.info(f"neighbours: frame check max err {frame_err:.2e} m, max M in batch {qa_np.shape[1]}, "
                        f"mean neighbours {ma_np.sum(1).mean():.1f}")

        gt_metrics = scene_metrics(gt_m[:, None], gt_m, v0_disp_m, constraint_m, q_eval, mask_eval, args.r_safe)
        gt_ped = ped_metrics(gt_m[:, None], gt_m, q_all, mask_all)
        for j in range(N):
            gt_rows.append({"scene": scene_offset + j, "viol_any": float(gt_metrics["viol_any"][j]),
                            "viol_any_raw_anchor": float(gt_metrics["viol_sample_rate_raw_anchor"][j]),
                            "anchor_clipped": float(gt_metrics["anchor_clipped"][j]),
                            "col_any": float(gt_metrics["col_any"][j]) if use_col else 0.0,
                            "col020_any": float(gt_ped["col020_any"][j])})

        gt_feasible = (gt_metrics["all_viol_any"] if use_col else gt_metrics["viol_any"]) < 0.5

        steps = torch.arange(1, F + 1, device=device, dtype=gt_m.dtype)
        pred_cv = (v0_disp_m[:, None, None, :] * steps[None, None, :, None])  # [N, 1, F, 2] constant velocity
        met_cv = scene_metrics(pred_cv, gt_m, v0_disp_m, constraint_m, q_eval, mask_eval, args.r_safe)
        met_cv.update(ped_metrics(pred_cv, gt_m, q_all, mask_all))
        for j in range(N):
            raw_rows.append({"method": "CONST_VEL", "seed": 0, "scene": scene_offset + j,
                             **{k: float(met_cv[k][j]) for k in met_cv},
                             "correction_m": 0.0, "active_steps": 0, "admm_iters_mean": 0.0, "admm_residual_max": 0.0,
                             "latency_s": 0.0, "terminal_raw_shift_m": 0.0, "distortion_vs_plain_m": float("nan"),
                             "proj_fail_rate": 0.0, "gt_feasible": bool(gt_feasible[j])})
        for seed_idx in range(args.seeds):
            seed = args.seed_base + 1000 * seed_idx + i_batch
            plain_y1 = None
            for m in methods:
                spec = specs[m]
                ctx.reset()
                ctx.set_batch(data)
                if use_col:
                    ctx.set_neighbours((q_m * scale + offset)[:, None, None], mask_t[:, None, None])
                if spec["active"]:
                    install_yflow(denoiser, ctx, spec, args)
                else:
                    denoiser.bwd_sample_t = upstream_bwd
                torch.manual_seed(seed)
                t0 = time.perf_counter()
                with torch.no_grad():
                    y1, _, _, _, _ = denoiser.sample(data, num_trajs=cfg.denoising_head_preds)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latency = time.perf_counter() - t0
                pred_m = unnormalize_min_max(y1.reshape(N, -1, F, 2), fut_min, fut_max, -1, 1)
                met = scene_metrics(pred_m, gt_m, v0_disp_m, constraint_m, q_eval, mask_eval, args.r_safe)
                met.update(ped_metrics(pred_m, gt_m, q_all, mask_all))
                corr = ctx.correction_sum / max(ctx.correction_count, 1)
                if m == "PLAIN_FM":
                    plain_y1 = y1
                    shift = np.zeros(N)
                    distortion = np.zeros(N)
                else:
                    last_raw = ctx.last_raw if ctx.last_raw is not None else y1
                    diff = (last_raw - plain_y1).reshape(N, -1, F, 2).norm(dim=-1).mean(dim=(1, 2)) / scale
                    shift = diff.cpu().numpy()
                    distortion = ((y1 - plain_y1).reshape(N, -1, F, 2).norm(dim=-1).mean(dim=(1, 2)) / scale).cpu().numpy()
                if ctx.terminal_res is not None:
                    fail = (ctx.terminal_res.reshape(N, -1) > 1e-3).float().mean(dim=1).cpu().numpy()
                else:
                    fail = np.zeros(N)
                for j in range(N):
                    raw_rows.append({"method": m, "seed": seed_idx, "scene": scene_offset + j,
                                     **{k: float(met[k][j]) for k in met},
                                     "correction_m": corr, "active_steps": ctx.active_steps,
                                     "admm_iters_mean": float(np.mean(ctx.iters)) if ctx.iters else 0.0,
                                     "admm_residual_max": float(np.max(ctx.residuals)) if ctx.residuals else 0.0,
                                     "latency_s": latency / N,
                                     "terminal_raw_shift_m": float(shift[j]),
                                     "distortion_vs_plain_m": float(distortion[j]),
                                     "proj_fail_rate": float(fail[j]),
                                     "gt_feasible": bool(gt_feasible[j])})
                logger.info(f"batch {i_batch} seed {seed_idx} {m:18s} minADE {met['min_ade'].mean():.4f} minFDE {met['min_fde'].mean():.4f} "
                            f"viol {met['viol_sample_rate'].mean():.4f} maxviol {met['max_viol'].max():.2e} "
                            + (f"col {met['col_sample_rate'].mean():.4f} minclr {np.nanmin(met['min_clearance']):.3f} " if use_col else "")
                            + f"COL@0.2 {met['col020_sample_rate'].mean():.4f} NLL {np.nanmean(met['kde_nll']):.3f} APD {met['apd'].mean():.4f} corr {corr:.4f} m "
                            f"steps {ctx.active_steps} iters {np.mean(ctx.iters) if ctx.iters else 0:.1f} rawshift {shift.mean():.4f} m {latency:.1f}s")
        scene_offset += N
    denoiser.bwd_sample_t = upstream_bwd

    methods_sum = methods + ["CONST_VEL"]
    summary = summarize(raw_rows, methods_sum, keys)
    split_keys = ["min_ade", "min_fde", "apd", "viol_sample_rate", "col020_sample_rate"] + (["col_sample_rate", "all_viol_sample_rate"] if use_col else [])
    for flag, tag in ((True, "gt_feasible"), (False, "gt_infeasible")):
        subset = [r for r in raw_rows if r["gt_feasible"] == flag]
        if subset:
            for r in summarize(subset, methods_sum, split_keys):
                r["metric"] = f"{r['metric']}|{tag}"
                summary.append(r)
    gt_viol = float(np.mean([r["viol_any"] for r in gt_rows]))
    gt_viol_raw = float(np.mean([r["viol_any_raw_anchor"] for r in gt_rows]))
    summary.append({"method": "GT", "metric": "viol_any", "count": len(gt_rows), "mean": gt_viol, "std": 0.0, "median": gt_viol, "ci95_low": gt_viol, "ci95_high": gt_viol})
    if use_col:
        gt_col = float(np.mean([r["col_any"] for r in gt_rows]))
        summary.append({"method": "GT", "metric": "col_any", "count": len(gt_rows), "mean": gt_col, "std": 0.0, "median": gt_col, "ci95_low": gt_col, "ci95_high": gt_col})
        flag = "all_viol_any"
        by = {(r["method"], r["scene"], r["seed"]): r[flag] > 0.5 for r in raw_rows}
        for ref in ("PLAIN_FM", "FINAL_PROJECTION"):
            if ref not in methods:
                continue
            for m in methods:
                if m == ref:
                    continue
                pairs = [(by[(ref, sc, sd)], by[(m, sc, sd)]) for (mm, sc, sd) in by if mm == m and (ref, sc, sd) in by]
                for a, b, name in ((False, True, "safe_to_viol"), (True, False, "viol_to_safe")):
                    cnt = sum(1 for x, y in pairs if x == a and y == b)
                    summary.append({"method": m, "metric": f"transition_{name}_vs_{ref}", "count": len(pairs), "mean": cnt / max(len(pairs), 1),
                                    "std": "", "median": "", "ci95_low": "", "ci95_high": "",
                                    "numerator": cnt, "denominator": len(pairs)})
    gt_c = float(np.mean([r["col020_any"] for r in gt_rows]))
    summary.append({"method": "GT", "metric": "col020_any", "count": len(gt_rows), "mean": gt_c, "std": 0.0, "median": gt_c, "ci95_low": gt_c, "ci95_high": gt_c})
    summary.append({"method": "GT", "metric": "viol_any_raw_anchor", "count": len(gt_rows), "mean": gt_viol_raw, "std": 0.0, "median": gt_viol_raw, "ci95_low": gt_viol_raw, "ci95_high": gt_viol_raw})
    with (out_dir / "results" / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "metric", "count", "mean", "std", "median", "ci95_low", "ci95_high",
                                           "numerator", "denominator"], restval="")
        w.writeheader()
        w.writerows(summary)
    if args.save_raw:
        with (out_dir / "results" / "raw.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
            w.writeheader()
            w.writerows(raw_rows)
    import hashlib
    here = Path(__file__).resolve().parent
    fingerprints = {name: hashlib.sha256((here / name).read_bytes()).hexdigest()[:16]
                    for name in ("run_phase1.py", "constraints.py", "neighbors.py", "sampling.py")}
    fingerprints["checkpoint"] = hashlib.sha256(ckpt.read_bytes()).hexdigest()[:16]
    config = {"model": args.model, "code_sha256_16": fingerprints, "args": vars(args), "limits": limit_info, "constraint": args.constraint, "r_safe": args.r_safe, "normalization": {"fut_traj_min": fut_min, "fut_traj_max": fut_max, "scale": scale, "offset": offset},
              "checkpoint": str(ckpt), "methods": specs, "n_scenes": scene_offset, "device": str(device),
              "fm_wrapper": cfg.get("fm_wrapper"), "tied_noise": bool(cfg.get("tied_noise", False)), "solver": cfg.solver, "sampling_steps": cfg.sampling_steps}
    (out_dir / "results" / "config.json").write_text(json.dumps(config, indent=2, default=str))
    logger.info(f"GT viol_any={gt_viol:.4f} (raw anchor {gt_viol_raw:.4f})")
    for r in summary:
        if r["metric"] in ("min_ade", "min_fde", "viol_sample_rate", "max_viol", "apd", "correction_m", "admm_iters_mean", "latency_s",
                           "terminal_raw_shift_m", "distortion_vs_plain_m", "proj_fail_rate", "col_sample_rate", "min_clearance",
                           "all_viol_sample_rate", "min_ade|gt_feasible", "min_ade|gt_infeasible", "apd|gt_feasible",
                           "col_any", "kde_nll", "col020_sample_rate", "col020_any") or r["metric"].startswith("transition_"):
            if r["metric"].startswith("transition_"):
                logger.info(f"{r['method']:18s} {r['metric']:18s} {r['mean']:.4f} ({r['numerator']}/{r['denominator']})")
            else:
                logger.info(f"{r['method']:18s} {r['metric']:18s} {r['mean']:.4f} ± {r['std']:.4f}")
    logger.info(f"wrote {out_dir / 'results'}")


if __name__ == "__main__":
    main()
