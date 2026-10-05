from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sh(*a: str) -> None:
    print("+", " ".join(a), flush=True)
    subprocess.run([sys.executable, "-m", *a], cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/nuscenes")
    ap.add_argument("--old_cache", default="datasets/nuscenes_trainval")
    ap.add_argument("--cache", default="datasets/nuscenes_v2")
    ap.add_argument("--old_run", default="outputs/fm_trainval")
    ap.add_argument("--run", default="v2")
    ap.add_argument("--base_config", default="configs/nuscenes.yaml")
    ap.add_argument("--config", default="configs/nuscenes_v2.yaml")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cache, run = ROOT / args.cache, ROOT / "outputs" / args.run
    splits = ("train", "train_val", "val")

    todo = [s for s in splits if not (cache / f"{s}.npz").exists()]
    if todo:
        sh("tools.convert_nuscenes", "--root", args.root, "--out", args.cache, "--splits", *todo,
           "--workers", str(args.workers))

    from vfm.indep import load_agents
    need = []
    for s in ("train_val", "val"):
        dst = cache / f"{s}_agents.npz"
        if dst.exists():
            continue
        src = ROOT / args.old_cache / f"{s}_agents.npz"
        if src.exists():
            shutil.copy(src, dst)
            with np.load(cache / f"{s}.npz") as z:
                ids = z["scene_id"]
            try:
                load_agents(cache, s, ids)
                print(f"copied {src} (aligned)")
                continue
            except ValueError:
                dst.unlink()
        need.append(s)
    if need:
        sh("tools.extract_agents", "--root", args.root, "--out", args.cache, "--splits", *need, "--workers", str(args.workers))

    if not (ROOT / args.config).exists():
        sh("tools.calibrate", "--config", args.base_config, "--cache", args.cache, "--out", args.config,
           "--device", args.device)

    run.mkdir(parents=True, exist_ok=True)
    old = ROOT / args.old_run
    for f in ("best.pt", "config.yaml"):
        if not (run / f).exists():
            shutil.copy(old / f, run / f)
    common = [f"device={args.device}", f"data.cache_dir={args.cache}", "data.train_split=train",
              "data.eval_split=train_val", f"run_name={args.run}", "out_dir=outputs"]
    if not (run / "tune" / "best.yaml").exists():
        sh("vfm.tune", "--config", args.config, "--trials", str(args.trials), *common)

    final = [f"data.eval_split=val", "eval.save_predictions=true", "data.limit_eval=null"]
    for m in ("cv", "fm"):
        if not (run / f"metrics_{m}_val_seed0.json").exists():
            sh("vfm.evaluate", "--config", args.config, "--method", m, *common, *final)
    if not (run / "metrics_yflow_val_seed0.json").exists():
        sh("vfm.evaluate", "--config", str(run / "tune" / "best.yaml"), "--method", "yflow", "--ref_pred",
           str(run / "pred_fm_val_seed0.npy"), *common, *final)
    for tool in ("tools.guideflow_metrics", "tools.analyze", "tools.visualize"):
        sh(tool, "--run", str(run), "--cache", args.cache, "--split", "val")

    import torch
    from vfm.config import load_config
    from vfm.data import load_meta, load_split
    from vfm.evaluate import indep_metrics
    from vfm.metrics import summarize

    cfg = load_config(run / "tune" / "best.yaml")
    ccfg = dict(cfg.constraints)
    meta = load_meta(cache)
    a = load_split(cache, "val")
    dev = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    preds = {"FM": run / "pred_fm_val_seed0.npy", "Y-Flow v1": old / "pred_yflow_val_seed0.npy",
             f"Y-Flow {args.run}": run / "pred_yflow_val_seed0.npy"}
    v1_fm = old / "pred_fm_val_seed0.npy"
    if v1_fm.exists():
        same = np.allclose(np.load(v1_fm), np.load(preds["FM"]), atol=1e-4)
        print(f"sanity: v2 FM predictions identical to v1 FM predictions: {same}")
    rows = {}
    for name, f in preds.items():
        if not f.exists():
            continue
        p = np.load(f)
        m = summarize(p, a["fut"], a["fut_mask"], meta)
        m.update(indep_metrics(p, a, meta, ccfg, cache, "val", 256, dev))
        rows[name] = m
    keys = [("minADE_10", "minADE@10"), ("minFDE_10", "minFDE@10"), ("MR_10@2m", "MR@10"),
            ("indep_coll_traj_gtok", "coll traj %"), ("indep_coll_scene_gtok", "coll scene %"),
            ("indep_offroad_traj_gtok", "offroad traj %"), ("indep_offroad_scene_gtok", "offroad scene %"),
            ("indep_offroad_excess_gtok", "offroad excess m")]
    print("\n== val, all checks on the v2 cache (collision / footprint off-road over scenes where GT passes) ==")
    print(f"{'':12}" + "".join(f"{h:>16}" for _, h in keys))
    for name, m in rows.items():
        vals = [m.get(k, float("nan")) * (100 if "%" in h else 1) for k, h in keys]
        print(f"{name:12}" + "".join(f"{v:>16.4f}" for v in vals))
    (run / "compare_val.json").write_text(json.dumps(rows, indent=2))
    print("wrote", run / "compare_val.json")


if __name__ == "__main__":
    main()
