from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"fm": "#2a78d6", "yflow": "#eb6834"}
NAMES = {"fm": "FM baseline", "yflow": "Y-Flow"}
KEYS = ("minADE_10", "minFDE_10", "MR_10@2m", "all_hard_safe", "scene_all_hard_safe", "scene_footprint_offroad_rate",
        "ms_per_scene")


def sh(*a: str) -> None:
    print("+", " ".join(a), flush=True)
    subprocess.run([sys.executable, "-m", *a], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def evaluate(cfg: str, run_name: str, ckpt: str, common: list[str], split: str) -> dict:
    out = {}
    for m in ("fm", "yflow"):
        sh("vfm.evaluate", "--config", cfg, "--method", m, "--ckpt", ckpt, f"run_name={run_name}", *common)
        r = json.loads((ROOT / "outputs" / run_name / f"metrics_{m}_{split}_seed0.json").read_text())
        out[m] = {k: r.get(k) for k in KEYS}
    return out


def plot(rows: list, what: str, xkey: str, xlabel: str, out: Path) -> None:
    rows = [r for r in rows if r["what"] == what]
    if not rows:
        return
    xs = [r[xkey] for r in rows]
    panels = (("minADE_10", "minADE@10 (m)", 1.0), ("scene_all_hard_safe", "scenes with a hard violation (%)", -100.0),
              ("scene_footprint_offroad_rate", "scenes with footprint off-road (%)", 100.0))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, (k, title, scale) in zip(axes, panels):
        for m in ("fm", "yflow"):
            ys = [r[m].get(k) for r in rows]
            ys = [None if y is None else (100 + y * scale if scale < 0 else y * scale) for y in ys]
            ax.plot(xs, ys, color=COLORS[m], lw=2, marker="o", ms=6, label=NAMES[m])
        ax.set(title=title, xlabel=xlabel)
        ax.grid(color="#e4e3dd", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        if what == "data":
            ax.set_xscale("log")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"ablation_{what}.png", dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/fm_trainval")
    ap.add_argument("--cache", default="datasets/nuscenes_trainval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--what", nargs="+", default=["steps", "data"], choices=["steps", "data"])
    ap.add_argument("--steps", nargs="+", type=int, default=[2, 4, 10, 20])
    ap.add_argument("--fracs", nargs="+", type=float, default=[0.1, 0.25, 1.0])
    ap.add_argument("--train_steps", type=int, default=15000)
    args = ap.parse_args()
    run = ROOT / args.run
    cfg = str(run / "tune" / "best.yaml")
    if not Path(cfg).exists():
        raise SystemExit(f"{cfg} not found: finish tuning first")
    ckpt = str(run / ("best.pt" if (run / "best.pt").exists() else "last.pt"))
    out = run / "ablation"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "ablation.jsonl"
    common = [f"device={args.device}", f"data.cache_dir={args.cache}", f"data.eval_split={args.split}",
              f"data.limit_eval={args.limit}", "eval.save_predictions=false", f"out_dir=outputs"]
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()] if log.exists() else []
    done = {(r["what"], r.get("n_steps"), r.get("frac")) for r in rows}

    if "steps" in args.what:
        for n in args.steps:
            if ("steps", n, None) in done:
                continue
            r = {"what": "steps", "n_steps": n, "frac": None,
                 **evaluate(cfg, f"ablation/steps{n}", ckpt, common + [f"eval.n_steps={n}"], args.split)}
            rows.append(r)
            with log.open("a") as fh:
                fh.write(json.dumps(r) + "\n")
    if "data" in args.what:
        counts = json.loads((ROOT / args.cache / "meta.json").read_text()).get("counts", {})
        if "train" in counts:
            n_train = int(counts["train"])
        else:
            import numpy as np
            with np.load(ROOT / args.cache / "train.npz") as z:
                n_train = int(z["hist"].shape[0])
        for f in args.fracs:
            if ("data", None, f) in done:
                continue
            name = f"ablation/frac{f:g}"
            if f >= 1.0:
                c = ckpt
            else:
                sh("vfm.train", "--config", cfg, f"run_name={name}", f"device={args.device}", f"data.cache_dir={args.cache}",
                   "data.train_split=train", "data.eval_split=train_val", f"data.limit_train={int(f * n_train)}",
                   f"train.steps={args.train_steps}", f"train.save_every={max(500, args.train_steps // 15)}",
                   "data.limit_eval=null", "train.select_limit=2000", "out_dir=outputs")
                c = str(ROOT / "outputs" / name / "best.pt")
            r = {"what": "data", "n_steps": None, "frac": f, **evaluate(cfg, name, c, common, args.split)}
            rows.append(r)
            with log.open("a") as fh:
                fh.write(json.dumps(r) + "\n")

    for r in rows:
        tag = f"steps={r['n_steps']}" if r["what"] == "steps" else f"frac={r['frac']:g}"
        for m in ("fm", "yflow"):
            v = r[m]
            print(f"{r['what']:<6}{tag:<12}{m:<7}" + "  ".join(
                f"{k}={v[k]:.4f}" for k in KEYS if v.get(k) is not None))
    plot(rows, "steps", "n_steps", "Euler steps", out)
    plot(rows, "data", "frac", "fraction of train used", out)
    print("wrote", log)


if __name__ == "__main__":
    main()
