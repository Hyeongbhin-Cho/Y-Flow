from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sh(args: list[str], gpu: str | None) -> int:
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    r = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    if r.returncode:
        print(r.stdout[-2000:])
    return r.returncode


def tune_plot(trials: Path, out: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(l) for l in trials.read_text().splitlines() if l.strip()]
    base = next((r for r in rows if r.get("trial") == -1), None)
    ok = [r for r in rows if r.get("trial", -1) >= 0 and "objective" in r]
    if not ok:
        return "tuning: no finished trial yet"
    t = [r["trial"] for r in ok]
    obj = [r["objective"] for r in ok]
    best = [min(obj[: i + 1]) for i in range(len(obj))]
    k = [kk for kk in ok[0]["metrics"] if kk.startswith("minADE_") and kk != "minADE_5"][0]
    ade = [r["metrics"][k] for r in ok]
    safe = [r["metrics"].get("all_hard_safe", float("nan")) for r in ok]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    axes[0].plot(t, obj, "o", color="#eb6834", ms=5, alpha=0.6, label="trial")
    axes[0].plot(t, best, color="#eb6834", lw=2, label="best so far")
    if base:
        axes[0].axhline(base["objective"], color="#2a78d6", lw=1.5, ls="--", label="FM baseline")
    axes[0].set(title="Tuning objective (lower is better)", xlabel="trial")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(t, ade, "o", color="#eb6834", ms=5)
    if base:
        axes[1].axhline(base["metrics"][k], color="#2a78d6", lw=1.5, ls="--", label="FM baseline")
        axes[1].legend(frameon=False, fontsize=8)
    axes[1].set(title=f"{k} per trial (m)", xlabel="trial")
    axes[2].plot(t, safe, "o", color="#eb6834", ms=5)
    axes[2].set(title="all_hard_safe per trial", xlabel="trial", ylim=(min(0.9, min(safe) - 0.01), 1.005))
    from matplotlib.ticker import MaxNLocator

    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(color="#e4e3dd", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "tuning_progress.png", dpi=110)
    plt.close(fig)
    i = min(range(len(ok)), key=lambda j: obj[j])
    return f"tuning: {len(ok)} trials, best trial {t[i]} obj {obj[i]:.4f} {k} {ade[i]:.4f} safe {safe[i]:.4f}"


def snapshot(args, last_ckpt_mtime: float | None) -> float | None:
    run = ROOT / args.run
    stamp = datetime.now().strftime("%m%d_%H%M")
    out = ROOT / "outputs" / "watch" / stamp
    out.mkdir(parents=True, exist_ok=True)
    line = [stamp]
    if (run / "train_log.jsonl").exists():
        shutil.copy(run / "train_log.jsonl", out / "train_log.jsonl")
        rows = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines() if l.strip()]
        step = max((r.get("step", 0) for r in rows), default=0)
        sel = [r["select_minADE"] for r in rows if "select_minADE" in r]
        line.append(f"train step {step}" + (f", select_minADE best {min(sel):.4f} last {sel[-1]:.4f}" if sel else ""))
    ckpt = run / ("best.pt" if (run / "best.pt").exists() else "last.pt")
    mtime = ckpt.stat().st_mtime if ckpt.exists() else None
    if ckpt.exists() and (args.force_eval or mtime != last_ckpt_mtime):
        local = out / "ckpt.pt"
        shutil.copy(ckpt, local)
        cfg = run / "tune" / "best.yaml"
        ycfg = str(cfg) if cfg.exists() else args.config
        common = ["--ckpt", str(local), f"device={args.device}", f"data.cache_dir={args.cache}",
                  f"data.eval_split={args.split}", f"data.limit_eval={args.limit}",
                  f"out_dir={out.parent}", f"run_name={stamp}", "eval.save_predictions=true"]
        sh(["vfm.evaluate", "--config", args.config, "--method", "fm", *common], args.gpu)
        if not args.no_yflow:
            ref = out / f"pred_fm_{args.split}_seed0.npy"
            sh(["vfm.evaluate", "--config", ycfg, "--method", "yflow", "--ref_pred", str(ref), *common], args.gpu)
        for m in ("fm", "yflow"):
            p = out / f"metrics_{m}_{args.split}_seed0.json"
            if p.exists():
                r = json.loads(p.read_text())
                k = int(r["K"])
                line.append(f"{m} minADE_{k} {r[f'minADE_{k}']:.4f} minFDE_{k} {r[f'minFDE_{k}']:.4f} "
                            f"safe {r.get('all_hard_safe', float('nan')):.4f}")
        line.append(f"ckpt {ckpt.name}" + (" + tune/best.yaml" if cfg.exists() else ""))
        local.unlink(missing_ok=True)
    elif ckpt.exists():
        line.append("checkpoint unchanged, eval skipped")
    trials = run / "tune" / "trials.jsonl"
    if trials.exists():
        line.append(tune_plot(trials, out))
    sh(["tools.visualize", "--run", str(out), "--cache", args.cache, "--split", args.split, "--n", str(args.n)], None)
    msg = " | ".join(line)
    print(msg, flush=True)
    with open(ROOT / "outputs" / "watch" / "summary.tsv", "a") as f:
        f.write("\t".join(line) + "\n")
    return mtime


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/fm_trainval")
    ap.add_argument("--cache", default="datasets/nuscenes_trainval")
    ap.add_argument("--config", default=None, help="default: {run}/config.yaml, i.e. the config the run was trained with")
    ap.add_argument("--split", default="train_val")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--every", type=float, default=30.0, help="minutes")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--no_yflow", action="store_true")
    ap.add_argument("--force_eval", action="store_true")
    args = ap.parse_args()
    if args.config is None:
        cfg = ROOT / args.run / "config.yaml"
        args.config = str(cfg if cfg.exists() else ROOT / "configs" / "nuscenes.yaml")
    last = None
    while True:
        done = (ROOT / args.run / "final_summary.json").exists()
        last = snapshot(args, last)
        if args.once or done:
            break
        time.sleep(args.every * 60)


if __name__ == "__main__":
    main()
