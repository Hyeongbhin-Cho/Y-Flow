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
from vfm.evaluate import constant_velocity
from vfm.indep import boxes, collisions, gt_conditional, headings, sat_overlap
from vfm.metrics import displacement

COLORS = {"gt": "#1f1f1e", "cv": "#8a8983", "fm": "#2a78d6", "yflow": "#eb6834"}
NAMES = {"gt": "GT (reference)", "cv": "Const. velocity", "fm": "FM baseline", "yflow": "Y-Flow"}
COMFORT = {"lon_acc_min": -4.05, "lon_acc_max": 2.40, "lat_acc": 4.89, "yaw_rate": 0.95, "lon_jerk": 4.13}


def lane_metrics(pred, lane, lmask, hz, lk_dist, chunk=64):
    n, k, t = pred.shape[:3]
    a = lane[:, :, :-1].reshape(n, -1, 2)
    b = lane[:, :, 1:].reshape(n, -1, 2)
    valid = (lmask[:, :, :-1] & lmask[:, :, 1:]).reshape(n, -1)
    ddc = np.zeros((n, k), bool)
    lk = np.zeros((n, k), bool)
    q = np.concatenate([np.zeros_like(pred[:, :, :1]), pred], axis=2)
    d = np.diff(q, axis=2)
    need = max(1, int(round(2.0 * hz)))
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        ab = (b[s:e] - a[s:e])[:, None, None]
        L = np.linalg.norm(ab, axis=-1).clip(1e-6)
        u = ab / L[..., None]
        p = pred[s:e][:, :, :, None]
        w = ((p - a[s:e][:, None, None]) * ab).sum(-1) / (L ** 2)
        c = a[s:e][:, None, None] + w.clip(0, 1)[..., None] * ab
        dist = np.linalg.norm(p - c, axis=-1)
        dist = np.where(valid[s:e][:, None, None], dist, np.inf)
        dmin = dist.min(-1)
        off = dmin >= lk_dist
        run = np.zeros(off.shape[:2], int)
        worst = np.zeros(off.shape[:2], int)
        for i in range(t):
            run = np.where(off[:, :, i], run + 1, 0)
            worst = np.maximum(worst, run)
        lk[s:e] = worst >= need
        mv = d[s:e]
        step = np.linalg.norm(mv, axis=-1)
        cosang = (mv[:, :, :, None] * u).sum(-1) / step.clip(1e-6)[..., None]
        near = dist <= 2.0
        against = (near & (cosang < -0.5)).any(-1) & ~(near & (cosang > 0.3)).any(-1) & (step > 0.5)
        ddc[s:e] = (np.where(against, step, 0).sum(-1) >= 2.0)
    return ddc, lk


def comfort(pred, hist, hmask, hz):
    v0 = np.where(hmask[:, -2:-1, None], (hist[:, -1:] - hist[:, -2:-1]) * hz, 0.0)[:, None]
    q = np.concatenate([np.zeros_like(pred[:, :, :1]), pred], axis=2)
    vel = np.concatenate([np.broadcast_to(v0, pred[:, :, :1].shape), np.diff(q, axis=2) * hz], axis=2)
    speed = np.linalg.norm(vel, axis=-1)
    yaw = np.concatenate([np.zeros_like(pred[:, :, :1, 0]), headings(pred)], axis=2)
    lon = np.diff(speed, axis=2) * hz
    yr = np.diff(np.unwrap(yaw, axis=2), axis=2) * hz
    lat = speed[:, :, 1:] * yr
    jerk = np.diff(lon, axis=2) * hz
    ok = ((lon >= COMFORT["lon_acc_min"]) & (lon <= COMFORT["lon_acc_max"])).all(-1)
    ok &= (np.abs(lat) <= COMFORT["lat_acc"]).all(-1)
    ok &= (np.abs(yr) <= COMFORT["yaw_rate"]).all(-1)
    ok &= (np.abs(jerk) <= COMFORT["lon_jerk"]).all(-1)
    return ok


def dac(pred, a, meta, ccfg, device, bs=256):
    ds = VehicleTrajDataset(a, optional=True)
    tol = float(ccfg.get("footprint_tol", 0.5))
    out = []
    for s in range(0, len(ds), bs):
        batch = to_device({k: v[s:s + bs] for k, v in ds.tensors.items()}, device)
        cons = NuScenesConstraint(batch, meta, ccfg)
        if cons.sdf is None:
            return None
        size = batch["focal_size"]
        v = cons.footprint_offroad(torch.from_numpy(pred[s:s + bs]).float().to(device), size[:, 0], size[:, 1])
        out.append((v <= tol).cpu().numpy())
    return np.concatenate(out)


def aggregate(fail_or_ok: np.ndarray, best: np.ndarray, is_ok: bool) -> dict:
    ok = fail_or_ok if is_ok else ~fail_or_ok
    return {"traj": float(100 * ok.mean()), "scene": float(100 * ok.all(1).mean()),
            "oracle": float(100 * ok[np.arange(len(ok)), best].mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/fm_trainval")
    ap.add_argument("--cache", default="datasets/nuscenes_trainval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--lk_dist", type=float, default=1.75)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run = Path(args.run)
    cfg_path = args.config or next(str(p) for p in (run / "tune/best.yaml", run / "config.yaml",
                                                    Path("configs/nuscenes.yaml")) if p.exists())
    ccfg = dict(load_config(cfg_path).get("constraints") or {})
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    meta = load_meta(args.cache)
    hz = float(meta["sample_hz"])
    preds = {m: np.load(run / f"pred_{m}_{args.split}_seed{args.seed}.npy") for m in ("fm", "yflow")
             if (run / f"pred_{m}_{args.split}_seed{args.seed}.npy").exists()}
    if not preds:
        raise SystemExit(f"no pred_*_{args.split}_seed{args.seed}.npy in {run}")
    n = min(len(p) for p in preds.values())
    a = load_split(args.cache, args.split, n)
    T = a["fut"].shape[1]
    preds = {"gt": a["fut"][:, None].copy(), "cv": constant_velocity(a, T), **{m: p[:n] for m, p in preds.items()}}
    agf = Path(args.cache) / f"{args.split}_agents.npz"
    ag = None
    if agf.exists():
        with np.load(agf) as z:
            ag = {k: z[k][:n] for k in z.files}
        if not (ag["scene_id"] == a["scene_id"]).all():
            raise SystemExit(f"{agf} is not aligned with {args.split}.npz")
    else:
        print(f"[warn] {agf} missing: NC skipped (run tools.extract_agents)")
    idx = [int(round(s * hz)) - 1 for s in (1, 2, 3)]
    size = a.get("focal_size", np.tile([[4.5, 1.9]], (n, 1)).astype(np.float32))
    res, raw = {}, {}
    for m, p in preds.items():
        raw[m] = {}
        ade, _ = displacement(p, a["fut"], a["fut_mask"])
        best = ade.argmin(1)
        d = np.linalg.norm(p - a["fut"][:, None], axis=-1)
        r = {"K": int(p.shape[1])}
        orc = d[np.arange(n), best]
        r["L2_oracle"] = {f"{s}s": float(orc[:, i].mean()) for s, i in zip((1, 2, 3), idx)}
        r["L2_oracle"]["avg"] = float(np.mean(list(r["L2_oracle"].values())))
        r["L2_mean_all"] = {f"{s}s": float(d[:, :, i].mean()) for s, i in zip((1, 2, 3), idx)}
        r["L2_mean_all"]["avg"] = float(np.mean(list(r["L2_mean_all"].values())))
        if ag is not None:
            hit, by = collisions(p, size, ag)
            r["NC"] = aggregate(hit, best, False)
            raw[m]["NC"] = ~hit
            r["collision_rate_by_class_traj"] = {name: float(100 * by[c].mean()) for c, name in
                                                 ((0, "vehicle"), (1, "pedestrian"), (2, "cyclist"), (3, "static"))}
        ok = dac(p, a, meta, ccfg, device)
        if ok is not None:
            r["DAC"] = aggregate(ok, best, True)
            raw[m]["DAC"] = ok
        ddc, lk = lane_metrics(p, a["lane"], a["lane_mask"], hz, args.lk_dist)
        r["DDC"] = aggregate(ddc, best, False)
        r["LK"] = aggregate(lk, best, False)
        ec = comfort(p, a["hist"], a["hist_mask"], hz)
        r["EC_comfort"] = aggregate(ec, best, True)
        raw[m].update({"DDC": ~ddc, "LK": ~lk, "EC_comfort": ec})
        mj = run / f"metrics_{m}_{args.split}_seed{args.seed}.json"
        if mj.exists() and m in ("fm", "yflow"):
            ms = json.loads(mj.read_text()).get("ms_per_scene")
            r["FPS_scenes_per_s"] = float(1000.0 / ms) if ms else None
        res[m] = r
    for m in res:
        for key, ok in raw[m].items():
            if key in raw.get("gt", {}):
                g = raw["gt"][key][:, 0]
                res[m][key]["traj_gtok"] = float(100 * ok[g].mean()) if g.any() else float("nan")
                res[m][key]["scene_gtok"] = float(100 * ok[g].all(1).mean()) if g.any() else float("nan")
                res[m][key]["n_gtok"] = int(g.sum())

    print(f"\nsplit={args.split}  n={n}  config={cfg_path}")
    print("L2 (m, oracle = minADE sample | mean over all samples)")
    for m, r in res.items():
        o, al = r["L2_oracle"], r["L2_mean_all"]
        print(f"  {m:<6} K={r['K']:<3} oracle 1s {o['1s']:.3f} 2s {o['2s']:.3f} 3s {o['3s']:.3f} avg {o['avg']:.3f}"
              f"   | all 1s {al['1s']:.3f} 2s {al['2s']:.3f} 3s {al['3s']:.3f} avg {al['avg']:.3f}")
    print("Compliance (% of trajectories / % of scenes with all K compliant / % oracle sample) - higher is better")
    for key in ("NC", "DAC", "DDC", "LK", "EC_comfort"):
        for m, r in res.items():
            if key in r:
                v = r[key]
                extra = (f"  | GT-ok scenes (n={v['n_gtok']}): traj {v['traj_gtok']:6.2f}  scene {v['scene_gtok']:6.2f}"
                         if "scene_gtok" in v else "")
                print(f"  {key:<11}{m:<6} traj {v['traj']:6.2f}  scene {v['scene']:6.2f}  oracle {v['oracle']:6.2f}{extra}")
    for m, r in res.items():
        if "collision_rate_by_class_traj" in r:
            print(f"  collisions by class (% traj) {m:<6} " + "  ".join(f"{k} {v:.3f}" for k, v in r["collision_rate_by_class_traj"].items()))
        if r.get("FPS_scenes_per_s"):
            print(f"  FPS {m:<6} {r['FPS_scenes_per_s']:.1f} scenes/s (batched; not comparable to single-scene planner FPS)")
    out = {"split": args.split, "n": n, "config": cfg_path, "lk_dist": args.lk_dist, "comfort_limits": COMFORT,
           "results": res}
    (run / f"guideflow_metrics_{args.split}.json").write_text(json.dumps(out, indent=2))

    print("  note: EC at 2 Hz is dominated by annotation noise (see GT); not used in the figure")
    keys = [k for k in ("NC", "DAC", "DDC", "LK") if all(k in r for r in res.values())]
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.8), gridspec_kw={"width_ratios": [1, 2.2]})
    ax = axes[0]
    ms = [m for m in res if m != "gt"]
    for i, m in enumerate(ms):
        v = [res[m]["L2_oracle"][s] for s in ("1s", "2s", "3s")]
        ax.plot([1, 2, 3], v, color=COLORS[m], lw=2, marker="o", ms=6, label=NAMES[m])
    ax.set(title="L2 of the minADE sample (m)", xlabel="horizon (s)", xticks=[1, 2, 3])
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    x = np.arange(len(keys))
    w = 0.8 / len(res)
    for i, m in enumerate(res):
        v = [100 - res[m][k].get("scene_gtok", res[m][k]["scene"]) for k in keys]
        xs = x + (i - (len(res) - 1) / 2) * w
        ax.bar(xs, v, w * 0.92, color=COLORS[m], label=NAMES[m])
        for xi, vi in zip(xs, v):
            ax.text(xi, vi, f"{vi:.1f}", ha="center", va="bottom", fontsize=6.5, color="#6b6a64")
    ax.set_xticks(x, [{"NC": "collision (NC)", "DAC": "off drivable (DAC)", "DDC": "wrong direction (DDC)",
                       "LK": "lane departure (LK)"}[k] for k in keys])
    ax.set(title="Scenes (GT passes) where at least one of K samples fails (%) - lower is better", ylim=(0, 112))
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for a_ in axes:
        a_.grid(color="#e4e3dd", lw=0.6)
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(run / f"guideflow_metrics_{args.split}.png", dpi=110)
    print("wrote", run / f"guideflow_metrics_{args.split}.json", run / f"guideflow_metrics_{args.split}.png")


if __name__ == "__main__":
    main()
