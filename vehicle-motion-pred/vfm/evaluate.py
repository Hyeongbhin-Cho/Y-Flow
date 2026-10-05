from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vfm.config import load_config
from vfm.constraints import NuScenesConstraint, violation_summary
from vfm.data import VehicleTrajDataset, load_meta, load_split, to_device
from vfm import anchors
from vfm.flow import sample
from vfm.indep import collisions, gt_conditional, load_agents
from vfm.metrics import displacement, summarize
from vfm.model import build_model
from vfm.train import pick_device
from vfm import yflow


def constant_velocity(arrays: dict[str, np.ndarray], future_steps: int) -> np.ndarray:
    hist, mask = arrays["hist"], arrays["hist_mask"]
    prev = np.where(mask[:, -2:-1, None], hist[:, -2:-1], hist[:, -1:])
    v = hist[:, -1] - prev[:, 0]
    steps = np.arange(1, future_steps + 1, dtype=np.float32)[None, :, None]
    return (hist[:, -1][:, None] + steps * v[:, None])[:, None]


_GOALNET: dict = {}


def goalnet_for(ccfg, cfg, device):
    path = ccfg.get("goal_net")
    if not path:
        return None
    if path not in _GOALNET:
        from vfm.goalnet import GoalScorer
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = GoalScorer(d_model=int(ck.get("d_model", cfg.model.d_model))).to(device)
        net.load_state_dict(ck["state_dict"])
        net.eval()
        _GOALNET[path] = net
        print(f"[goals] learned scorer {path} (median goal error {ck.get('median_m', float('nan')):.2f} m)",
              flush=True)
    return _GOALNET[path]


def propose_goals(model, batch, std, k, n_steps, gen, cons, ccfg, meta, cfg=None) -> None:
    if str(ccfg.get("goal_source", "gt")) == "gt" or "goal" not in cons.targets:
        return
    src = str(ccfg.get("goal_source", "gt"))
    hz = float(meta["sample_hz"])
    m = int(ccfg.get("anchor_samples", 100))
    x = sample(model, batch, std, m, n_steps, gen)
    horizon_s, dt = x.shape[2] / hz, 1.0 / hz
    net = goalnet_for(ccfg, cfg, x.device) if src == "learned" else None
    if net is not None:
        from vfm import goalnet
        g = goalnet.propose(net, model, batch, x[:, :, -1], k, dict(ccfg), horizon_s, dt)
    else:
        g = anchors.propose(batch, x[:, :, -1], k, dict(ccfg), horizon_s, dt)
    cons.set_target("goal", g, step=x.shape[2] - 1)


def constraint_metrics(pred: np.ndarray, arrays, meta, ccfg, batch_size: int, device, gt: bool = False,
                       goals: torch.Tensor | None = None) -> dict:
    ds = VehicleTrajDataset(arrays, optional=True)
    hs, fp = {}, []
    fp_tol = float(ccfg.get("footprint_tol", 0.5))
    for s in range(0, len(ds), batch_size):
        batch = to_device({k: v[s:s + batch_size] for k, v in ds.tensors.items()}, device)
        cons = NuScenesConstraint(batch, meta, ccfg)
        if goals is not None and "goal" in cons.targets:
            cons.set_target("goal", goals[s:s + batch_size].to(device))
        p = torch.from_numpy(pred[s:s + batch_size]).to(device)
        h = cons.h(p, step_mask=batch["fut_mask"] if gt else None)
        for n, v in h.items():
            hs.setdefault(n, []).append(v.cpu())
        if cons.sdf is not None:
            size = batch.get("focal_size")
            length = size[:, 0] if size is not None else torch.full((p.shape[0],), 4.5, device=device)
            width = size[:, 1] if size is not None else torch.full((p.shape[0],), 1.9, device=device)
            pp = p
            if gt:
                pp = torch.where(batch["fut_mask"][:, None, :, None], p, p[:, :, :1].expand_as(p))
            fp.append(cons.footprint_offroad(pp, length, width).cpu())
    h = {n: torch.cat(v) for n, v in hs.items()}
    out = violation_summary(h, tuple(ccfg.get("hard", ())), float(ccfg.get("tol", 1e-3)))
    if fp:
        f = torch.cat(fp) > fp_tol
        out["footprint_offroad_rate"] = float(f.float().mean())
        out["scene_footprint_offroad_rate"] = float(f.any(dim=-1).float().mean())
    if gt:
        for n, v in h.items():
            v = v.flatten().numpy()
            out[f"gt_h_pct_{n}"] = {str(q): float(np.percentile(v, q)) for q in (50, 90, 99, 99.9)}
    return out


def indep_metrics(pred: np.ndarray, arrays, meta, ccfg, cache_dir, split: str, batch_size: int, device) -> dict:
    out = {}
    gt = arrays["fut"][:, None].copy()
    fm = arrays["fut_mask"]
    gt = np.where(fm[:, None, :, None], gt, gt[:, :, :1])
    size = arrays.get("focal_size")
    if size is None:
        size = np.tile([[4.5, 1.9]], (len(pred), 1)).astype(np.float32)
    ag = load_agents(cache_dir, split, arrays["scene_id"])
    if ag is not None:
        hit, _ = collisions(pred, size, ag)
        ghit, _ = collisions(gt, size, ag)
        for k, v in gt_conditional(hit, ghit[:, 0]).items():
            out[f"indep_coll_{k}"] = v
    if "sdf" in arrays and "sdf" in meta:
        tol = float(ccfg.get("footprint_tol", 0.5))
        ds = VehicleTrajDataset(arrays, optional=True)
        fp, gfp, fpv = [], [], []
        for s in range(0, len(ds), batch_size):
            batch = to_device({k: v[s:s + batch_size] for k, v in ds.tensors.items()}, device)
            cons = NuScenesConstraint(batch, meta, ccfg)
            L, W = cons.focal_l, cons.focal_w
            v = cons.footprint_offroad(torch.from_numpy(pred[s:s + batch_size]).float().to(device), L, W).cpu().numpy()
            fpv.append(v)
            fp.append(v > tol)
            gfp.append((cons.footprint_offroad(torch.from_numpy(gt[s:s + batch_size]).float().to(device), L, W) > tol).cpu().numpy())
        for k, v in gt_conditional(np.concatenate(fp), np.concatenate(gfp)[:, 0]).items():
            out[f"indep_offroad_{k}"] = v
        gok = ~np.concatenate(gfp)[:, 0]
        ex = np.clip(np.concatenate(fpv) - tol, 0.0, None)
        out["indep_offroad_excess_gtok"] = float(ex[gok].mean()) if gok.any() else float("nan")
    return out


def run(cfg, method: str = "fm", ckpt_path: str | None = None, ref_pred: str | None = None) -> dict:
    meta = load_meta(cfg.data.cache_dir)
    meta = {**meta, "extra_k": list(meta.get("extra_k", [])) + list(cfg.eval.get("extra_k") or [])}
    split = str(cfg.data.get("eval_split", "val"))
    arrays = load_split(cfg.data.cache_dir, split, cfg.data.get("limit_eval"))
    out_dir = Path(cfg.out_dir) / str(cfg.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    kin = cfg.eval.get("kinematic_limits") or {}
    ccfg = dict(cfg.get("constraints") or {})
    T = int(arrays["fut"].shape[1])
    device = pick_device(str(cfg.get("device", "cuda")))
    seed = int(cfg.eval.get("seed", cfg.seed))

    if method in ("cv", "gt"):
        pred = constant_velocity(arrays, T) if method == "cv" else arrays["fut"][:, None].copy()
        metrics = summarize(pred, arrays["fut"], arrays["fut_mask"], meta, kin)
        metrics.update({"method": {"cv": "constant_velocity", "gt": "ground_truth"}[method], "split": split})
    else:
        default = out_dir / "best.pt" if (out_dir / "best.pt").exists() else out_dir / "last.pt"
        ckpt = torch.load(ckpt_path or default, map_location="cpu", weights_only=False)
        model = build_model(cfg, future_steps=int(ckpt["future_steps"])).to(device)
        model.load_state_dict(ckpt["ema" if cfg.eval.get("use_ema", True) else "model"])
        model.eval()
        std = ckpt["fut_std"].to(device)
        k = int(cfg.eval.get("k") or meta.get("eval_k", 6))
        n_steps = int(cfg.eval.n_steps)
        gen = torch.Generator().manual_seed(seed)
        loader = DataLoader(VehicleTrajDataset(arrays, optional=method in ("yflow", "proj")),
                            batch_size=int(cfg.eval.batch_size), shuffle=False)
        preds, ystats, goal_store = [], {"gated_frac": [], "lipschitz_mean": []}, []
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for batch in loader:
            batch = to_device(batch, device)
            if method == "yflow":
                cons = NuScenesConstraint(batch, meta, ccfg)
                propose_goals(model, batch, std, k, n_steps, gen, cons, ccfg, meta, cfg)
                x, st = yflow.sample(model, batch, std, k, n_steps, cons, dict(cfg.get("yflow") or {}), gen,
                                     final_sweeps=int(ccfg.get("proj_sweeps_final", 50)))
                for n in ystats:
                    ystats[n].extend(st[n])
            elif method == "proj":
                cons = NuScenesConstraint(batch, meta, ccfg)
                propose_goals(model, batch, std, k, n_steps, gen, cons, ccfg, meta, cfg)
                x = sample(model, batch, std, k, n_steps, gen)
                x = cons.project_feasible(x, sweeps=int(ccfg.get("proj_sweeps_final", 50)))
            else:
                x = sample(model, batch, std, k, n_steps, gen)
            if method in ("yflow", "proj") and str(ccfg.get("goal_source", "gt")) != "gt" \
                    and "goal" in getattr(cons, "targets", {}):
                goal_store.append(cons.targets["goal"][1].detach().cpu())
            preds.append(x.float().cpu().numpy())
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        pred = np.concatenate(preds, axis=0)
        metrics = summarize(pred, arrays["fut"], arrays["fut_mask"], meta, kin)
        metrics.update({"method": {"fm": "flow_matching", "yflow": "yflow", "proj": "fm_posthoc_projection"}[method], "split": split,
                        "n_steps": n_steps, "step": int(ckpt["step"]),
                        "ms_per_scene": 1000.0 * elapsed / max(1, pred.shape[0])})
        if method == "yflow":
            metrics["yflow_gated_frac_mean"] = float(np.mean(ystats["gated_frac"])) if ystats["gated_frac"] else 0.0
            metrics["yflow_lipschitz_mean"] = float(np.mean(ystats["lipschitz_mean"])) if ystats["lipschitz_mean"] else 0.0
        if cfg.eval.get("save_predictions", False):
            np.save(out_dir / f"pred_{method}_{split}_seed{seed}.npy", pred.astype(np.float32))

    if ccfg:
        goals = torch.cat(goal_store) if ("goal_store" in dir() and goal_store) else None
        metrics.update(constraint_metrics(pred, arrays, meta, ccfg, int(cfg.eval.batch_size), device,
                                          gt=method == "gt", goals=goals))
        if method != "gt" and cfg.eval.get("indep", True):
            metrics.update(indep_metrics(pred, arrays, meta, ccfg, cfg.data.cache_dir, split, int(cfg.eval.batch_size), device))
    if ref_pred:
        ref = np.load(ref_pred)
        thr = float(meta.get("miss_threshold_m", 2.0))
        _, f_ref = displacement(ref, arrays["fut"], arrays["fut_mask"])
        _, f_cur = displacement(pred, arrays["fut"], arrays["fut_mask"])
        metrics["mode_lost_rate"] = float(((f_ref.min(1) <= thr) & (f_cur.min(1) > thr)).mean())
        metrics["mode_gained_rate"] = float(((f_ref.min(1) > thr) & (f_cur.min(1) <= thr)).mean())

    tag = f"metrics_{method}_{split}_seed{seed}.json"
    (out_dir / tag).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", default="fm", choices=["fm", "yflow", "proj", "cv", "gt"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--ref_pred", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    run(load_config(args.config, args.overrides), args.method, args.ckpt, args.ref_pred)


if __name__ == "__main__":
    main()
