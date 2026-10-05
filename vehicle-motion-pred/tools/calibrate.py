from __future__ import annotations

import argparse
import copy
import json
import math

import numpy as np
import torch

from vfm.config import dump_config, load_config
from vfm.constraints import NuScenesConstraint, violation_summary
from vfm.data import VehicleTrajDataset, load_meta, load_split, to_device

HARD = ["speed", "continuity", "accel", "drivable_fp", "static", "coll"]


def gt_raw(cache, split, meta, ccfg, device, limit, bs=512):
    a = load_split(cache, split, limit)
    ds = VehicleTrajDataset(a, optional=True)
    cfg = {**ccfg, "fp_margin": 0.0, "coll_margin": 0.0, "coll_horizon_s": 1e9}
    fp, coll = [], []
    for s in range(0, len(ds), bs):
        batch = to_device({k: v[s:s + bs] for k, v in ds.tensors.items()}, device)
        cons = NuScenesConstraint(batch, meta, cfg)
        hs = cons.h_steps(batch["fut"][:, None])
        m = batch["fut_mask"][:, None]
        if "drivable_fp" in hs:
            fp.append(hs["drivable_fp"].masked_fill(~m, -1.0)[:, 0].cpu().numpy())
        if "coll" in hs:
            coll.append(hs["coll"].masked_fill(~m, -1.0)[:, 0].cpu().numpy())
    return (np.concatenate(fp) if fp else None), (np.concatenate(coll) if coll else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--check_split", default="train_val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--target", type=float, default=0.001, help="allowed GT violation rate per constraint")
    ap.add_argument("--max_overlap", type=float, default=0.5, help="largest allowed GT circle overlap (m)")
    ap.add_argument("--horizons", nargs="+", type=float, default=[1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    cfg.data.cache_dir = args.cache
    ccfg = dict(cfg.get("constraints") or {})
    meta = load_meta(args.cache)
    dt = 1.0 / float(meta["sample_hz"])
    q = 1.0 - args.target
    fp, coll = gt_raw(args.cache, args.split, meta, ccfg, device, args.limit or None)
    report = {"split": args.split, "target_violation": args.target}
    new = {}
    if fp is not None:
        need = float(np.quantile(fp.max(1), q))
        new["fp_margin"] = max(0.0, math.ceil(need * 10) / 10)
        report["fp_quantiles"] = {str(p): float(np.quantile(fp.max(1), p)) for p in (0.9, 0.99, 0.999)}
    if coll is not None:
        table = {}
        for H in args.horizons:
            k = int(round(H / dt))
            need = float(np.quantile(coll[:, :k].max(1), q))
            table[H] = max(0.0, need)
            report.setdefault("coll_quantiles_by_horizon", {})[H] = {
                str(p): float(np.quantile(coll[:, :k].max(1), p)) for p in (0.9, 0.99, 0.999)}
        report["coll_allowance_by_horizon"] = table
        ok = [H for H, v in table.items() if v <= args.max_overlap]
        H = max(ok) if ok else min(table)
        new["coll_horizon_s"] = float(H)
        new["coll_margin"] = -math.ceil(table[H] * 20) / 20
    hard = [h for h in HARD if (h != "drivable_fp" or fp is not None) and (h != "coll" or coll is not None)]
    out_cfg = copy.deepcopy(cfg)
    out_cfg.constraints.update(new)
    out_cfg.constraints["hard"] = hard
    dump_config(out_cfg, args.out)
    report["new"] = new
    report["hard"] = hard

    a = load_split(args.cache, args.check_split)
    ds = VehicleTrajDataset(a, optional=True)
    hs = {}
    for s in range(0, len(ds), 512):
        batch = to_device({k: v[s:s + 512] for k, v in ds.tensors.items()}, device)
        cons = NuScenesConstraint(batch, meta, dict(out_cfg.constraints))
        for n, v in cons.h(batch["fut"][:, None], step_mask=batch["fut_mask"]).items():
            hs.setdefault(n, []).append(v.cpu())
    summ = violation_summary({n: torch.cat(v) for n, v in hs.items()}, tuple(hard), float(ccfg.get("tol", 1e-3)))
    report["check"] = {args.check_split: {k: v for k, v in summ.items() if k.startswith(("viol_rate", "all_hard"))}}
    print(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
