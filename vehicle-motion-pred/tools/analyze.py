from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from vfm.config import load_config
from vfm.constraints import NuScenesConstraint
from vfm.data import VehicleTrajDataset, load_meta, load_split, to_device
from vfm.metrics import displacement

COLORS = {"cv": "#8a8983", "fm": "#2a78d6", "yflow": "#eb6834"}
NAMES = {"cv": "Const. velocity", "fm": "FM baseline", "yflow": "Y-Flow"}


def per_scene(pred, a, meta, ccfg, device, bs=256):
    ds = VehicleTrajDataset(a, optional=True)
    hard = tuple(ccfg.get("hard", ()))
    tol = float(ccfg.get("tol", 1e-3))
    fp_tol = float(ccfg.get("footprint_tol", 0.5))
    worst, fp = [], []
    for s in range(0, len(ds), bs):
        batch = to_device({k: v[s:s + bs] for k, v in ds.tensors.items()}, device)
        cons = NuScenesConstraint(batch, meta, ccfg)
        p = torch.from_numpy(pred[s:s + bs]).float().to(device)
        h = cons.h(p)
        w = torch.stack([h[n] for n in hard if n in h]).max(0).values
        worst.append((w > tol).cpu().numpy())
        if cons.sdf is not None:
            size = batch.get("focal_size")
            n = p.shape[0]
            length = size[:, 0] if size is not None else torch.full((n,), 4.5, device=device)
            width = size[:, 1] if size is not None else torch.full((n,), 1.9, device=device)
            fp.append((cons.footprint_offroad(p, length, width) > fp_tol).cpu().numpy())
    ade, fde = displacement(pred, a["fut"], a["fut_mask"])
    out = {"minADE": ade.min(1), "minFDE": fde.min(1), "viol": np.concatenate(worst)}
    if fp:
        out["fp"] = np.concatenate(fp)
    return out


def subsets(a, fm_viol):
    fut, fm = a["fut"], a["fut_mask"]
    n, t = fut.shape[:2]
    last = t - 1 - np.argmax(fm[:, ::-1], axis=1)
    prev = np.maximum(last - 1, 0)
    d = fut[np.arange(n), last] - fut[np.arange(n), prev]
    ang = np.abs(np.degrees(np.arctan2(d[:, 1], d[:, 0])))
    dist = np.linalg.norm(fut[np.arange(n), last], axis=-1)
    speed = dist / np.maximum(last + 1, 1)
    out = {"all": np.ones(n, bool), "turn": ang > 30, "fast": speed >= np.quantile(speed, 0.75)}
    if "sdf" in a:
        g = SDF_META
        ix = np.clip(((fut[..., 0] - g["x_min"]) / g["res"]).astype(int), 0, g["size"] - 1)
        iy = np.clip(((fut[..., 1] - g["y_min"]) / g["res"]).astype(int), 0, g["size"] - 1)
        sdf = a["sdf"][np.arange(n)[:, None], iy, ix].astype(np.float32) * g["scale"]
        out["near_edge"] = (np.where(fm, np.abs(sdf), np.inf) < 2.0).any(1)
    if "obs" in a:
        c = a["obs"][:, None, :, :2]
        dd = np.linalg.norm(fut[:, :, None] - c, axis=-1)
        dd = np.where(a["obs_mask"][:, None, :] & fm[:, :, None], dd, np.inf)
        out["near_static"] = dd.min(axis=(1, 2)) < 5.0
    if fm_viol is not None:
        out["fm_violating"] = fm_viol.any(1)
    return out


SDF_META: dict = {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/fm_trainval")
    ap.add_argument("--cache", default="datasets/nuscenes_trainval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--config", default=None, help="constraints block; default {run}/tune/best.yaml, else {run}/config.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run = Path(args.run)
    cfg_path = args.config or next(str(p) for p in (run / "tune/best.yaml", run / "config.yaml",
                                                    Path("configs/nuscenes.yaml")) if p.exists())
    ccfg = dict(load_config(cfg_path).get("constraints") or {})
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    meta = load_meta(args.cache)
    SDF_META.update(meta.get("sdf", {}))
    preds = {m: np.load(run / f"pred_{m}_{args.split}_seed{args.seed}.npy") for m in ("fm", "yflow")
             if (run / f"pred_{m}_{args.split}_seed{args.seed}.npy").exists()}
    if not preds:
        raise SystemExit(f"no pred_*_{args.split}_seed{args.seed}.npy in {run}")
    n = min(len(p) for p in preds.values())
    a = {k: v[:n] for k, v in load_split(args.cache, args.split, n).items()}
    preds = {m: p[:n] for m, p in preds.items()}
    thr = float(meta.get("miss_threshold_m", 2.0))
    res = {m: per_scene(p, a, meta, ccfg, device) for m, p in preds.items()}
    subs = subsets(a, res["fm"]["viol"] if "fm" in res else None)
    table = {}
    k = next(iter(preds.values())).shape[1]
    hdr = ["subset", "n", "method", f"minADE_{k}", f"minFDE_{k}", "MR", "traj_viol%", "scene_viol%",
           "fp_offroad%", "scene_fp%"]
    print(("{:<13}{:>6}  {:<12}" + "{:>12}" * 7).format(*hdr))
    for s, mask in subs.items():
        table[s] = {"n": int(mask.sum())}
        for m, r in res.items():
            if not mask.any():
                continue
            row = {"minADE": float(r["minADE"][mask].mean()), "minFDE": float(r["minFDE"][mask].mean()),
                   "MR": float((r["minFDE"][mask] > thr).mean()),
                   "traj_viol": float(100 * r["viol"][mask].mean()),
                   "scene_viol": float(100 * r["viol"][mask].any(1).mean())}
            if "fp" in r:
                row["fp_offroad"] = float(100 * r["fp"][mask].mean())
                row["scene_fp"] = float(100 * r["fp"][mask].any(1).mean())
            table[s][m] = row
            vals = [row["minADE"], row["minFDE"], row["MR"], row["traj_viol"], row["scene_viol"],
                    row.get("fp_offroad", float("nan")), row.get("scene_fp", float("nan"))]
            print(("{:<13}{:>6}  {:<12}" + "{:>12.3f}" * 7).format(s, int(mask.sum()), m, *vals))
    (run / f"analysis_{args.split}.json").write_text(json.dumps({"config": cfg_path, "table": table}, indent=2))

    names = [s for s in table if all(m in table[s] for m in res)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    x = np.arange(len(names))
    w = 0.8 / len(res)
    for j, (key, title) in enumerate((("minADE", f"minADE@{k} (m)"), ("scene_viol", "scenes with a hard violation (%)"),
                                      ("scene_fp", "scenes with footprint off-road (%)"))):
        ax = axes[j]
        for i, m in enumerate(res):
            v = [table[s][m].get(key, np.nan) for s in names]
            xs = x + (i - (len(res) - 1) / 2) * w
            ax.bar(xs, v, w * 0.92, color=COLORS[m], label=NAMES[m])
            for xi, vi in zip(xs, v):
                if np.isfinite(vi):
                    ax.text(xi, vi, f"{vi:.2f}" if key == "minADE" else f"{vi:.1f}", ha="center", va="bottom",
                            fontsize=6.5, color="#6b6a64")
        ax.set_xticks(x, [f"{s}\n(n={table[s]['n']})" for s in names], fontsize=7.5)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", color="#e4e3dd", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(run / f"analysis_{args.split}.png", dpi=110)
    print("wrote", run / f"analysis_{args.split}.json", run / f"analysis_{args.split}.png")


if __name__ == "__main__":
    main()
