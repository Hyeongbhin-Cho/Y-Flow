from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
from matplotlib.ticker import ScalarFormatter

from vfm.data import load_meta, load_split
from vfm.metrics import displacement

INK, INK2, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
COLORS = {"cv": "#8a8983", "fm": "#2a78d6", "proj": "#1f9d61", "yflow": "#eb6834"}
NAMES = {"cv": "Const. velocity", "fm": "FM baseline", "proj": "FM + projection", "yflow": "Y-Flow"}

plt.rcParams.update({
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "axes.titlecolor": INK,
    "legend.frameon": False, "figure.dpi": 110,
})


def training_curves(run: Path, out: Path) -> None:
    rows = [json.loads(l) for l in (run / "train_log.jsonl").read_text().splitlines() if l.strip()]
    loss = [(r["step"], r["loss"]) for r in rows if "loss" in r]
    sel = [(r["step"], r["select_minADE"]) for r in rows if "select_minADE" in r]
    fig, axes = plt.subplots(1, 2 if sel else 1, figsize=(10 if sel else 5, 3.4), squeeze=False)
    ax = axes[0, 0]
    ax.plot(*zip(*loss), color=COLORS["fm"], lw=2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(ScalarFormatter())
    ax.set(title="Flow-matching loss (train)", xlabel="step", ylabel="loss (log scale)")
    if sel:
        ax = axes[0, 1]
        s, v = zip(*sel)
        ax.plot(s, v, color=COLORS["fm"], lw=2, marker="o", ms=5)
        i = int(np.argmin(v))
        ax.annotate(f"best {v[i]:.3f} m @ {s[i]}", (s[i], v[i]), xytext=(0, 12), textcoords="offset points",
                    ha="center", color=INK)
        ax.set(title="Checkpoint selection: minADE on selection split", xlabel="step", ylabel="minADE (m)")
    fig.tight_layout()
    fig.savefig(out / "training_curves.png")
    plt.close(fig)


def horizon_error(preds: dict, a: dict, hz: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    t = np.arange(1, a["fut"].shape[1] + 1) / hz
    for m, p in preds.items():
        ade, _ = displacement(p, a["fut"], a["fut_mask"])
        best = ade.argmin(1)
        d = np.linalg.norm(p[np.arange(len(p)), best] - a["fut"], axis=-1)
        d = np.where(a["fut_mask"], d, np.nan)
        y = np.nanmean(d, axis=0)
        ax.plot(t, y, color=COLORS[m], lw=2, marker="o", ms=4, label=NAMES[m])
    ax.set(title="Error vs horizon (sample chosen by minADE)", xlabel="horizon (s)", ylabel="mean L2 error (m)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "horizon_error.png")
    plt.close(fig)


def violations(metrics: dict, out: Path, hard: tuple = ()) -> None:
    names = sorted({k[len("viol_rate_"):] for m in metrics.values() for k in m if k.startswith("viol_rate_")})
    if not names:
        return
    methods = [m for m in ("cv", "fm", "proj", "yflow") if m in metrics]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.8 / len(methods)
    x = np.arange(len(names))
    for i, m in enumerate(methods):
        v = [100 * metrics[m].get(f"viol_rate_{n}", np.nan) for n in names]
        xs = x + (i - (len(methods) - 1) / 2) * w
        ax.bar(xs, v, w * 0.92, color=COLORS[m], label=NAMES[m])
        for xi, vi in zip(xs, v):
            if np.isfinite(vi):
                ax.text(xi, vi + 1.5, f"{vi:.1f}" if vi < 10 else f"{vi:.0f}", ha="center", va="bottom",
                        fontsize=6.5, color=INK2)
    ax.set_xticks(x, [f"{n}\n(hard)" if n in hard else f"{n}\n(soft)" for n in names])
    ax.set_ylim(0, 112)
    ax.set(title="Predicted trajectories violating each constraint", ylabel="% of trajectories")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "constraint_violations.png")
    plt.close(fig)


def _box(ax, o):
    x, y, yaw, l, w = o
    c, s = np.cos(yaw), np.sin(yaw)
    pts = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]) / 2 @ np.array([[c, s], [-s, c]]) + [x, y]
    ax.add_patch(Polygon(pts, closed=True, fc="#c9c8c1", ec=INK2, lw=0.6, zorder=2))


OFFROAD = ListedColormap(["#dedcd4"])


def scene(ax, a: dict, i: int, meta: dict, preds: dict, title: str, frame: dict | None = None) -> None:
    fut, fm = a["fut"][i], a["fut_mask"][i]
    frame = frame or preds
    view = np.concatenate([a["hist"][i][a["hist_mask"][i]], fut[fm]] + [p[i].reshape(-1, 2) for p in frame.values()])
    lo, hi = view.min(0), view.max(0)
    c, r = (lo + hi) / 2, max(15.0, float((hi - lo).max()) / 2 + 8)
    if "sdf" in a and "sdf" in meta:
        g = meta["sdf"]
        ext = [g["x_min"], g["x_min"] + g["res"] * g["size"], g["y_min"], g["y_min"] + g["res"] * g["size"]]
        off = np.ma.masked_where(a["sdf"][i] <= 0, np.ones_like(a["sdf"][i], dtype=float))
        ax.imshow(off, origin="lower", extent=ext, cmap=OFFROAD, zorder=0, interpolation="nearest")
    for l, m in zip(a["lane"][i], a["lane_mask"][i]):
        if m.any():
            ax.plot(l[m, 0], l[m, 1], color="#b9b8b0", lw=0.8, zorder=1)
    for nb, m in zip(a["nbr"][i], a["nbr_mask"][i]):
        if m.any():
            ax.plot(nb[m, 0], nb[m, 1], color="#a3a29a", lw=1, zorder=2)
    if "obs" in a:
        for o, m in zip(a["obs"][i], a["obs_mask"][i]):
            if m:
                _box(ax, o)
    for mth, p in preds.items():
        for k in range(p.shape[1]):
            ax.plot(np.r_[0, p[i, k, :, 0]], np.r_[0, p[i, k, :, 1]], color=COLORS[mth], lw=1.2, alpha=0.55, zorder=3)
    h = a["hist"][i][a["hist_mask"][i]]
    ax.plot(h[:, 0], h[:, 1], color=INK, lw=2, marker="o", ms=3, zorder=4)
    ax.plot(np.r_[0, fut[fm, 0]], np.r_[0, fut[fm, 1]], color=INK, lw=2, ls="--", marker="o", ms=3, zorder=5)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8.5)
    ax.tick_params(labelsize=7)


def scenes(a: dict, meta: dict, preds: dict, pick: str, n: int, seed: int, out: Path) -> None:
    ade = {m: displacement(p, a["fut"], a["fut_mask"])[0].min(1) for m, p in preds.items()}
    idx = np.arange(len(a["fut"]))
    if pick == "fm_worst" and "fm" in ade:
        order = np.argsort(-ade["fm"])
    elif pick in ("yflow_gain", "yflow_loss") and {"fm", "yflow"} <= set(ade):
        diff = ade["fm"] - ade["yflow"]
        order = np.argsort(-diff if pick == "yflow_gain" else diff)
    else:
        order = np.random.default_rng(seed).permutation(idx)
    sel = order[:n]
    cols = len(preds)
    fig, axes = plt.subplots(n, cols, figsize=(4.2 * cols, 4.0 * n), squeeze=False)
    for r, i in enumerate(sel):
        for cidx, (m, p) in enumerate(preds.items()):
            scene(axes[r, cidx], a, int(i), meta, {m: p}, f"{NAMES[m]}  scene {int(i)}  minADE {ade[m][i]:.2f} m",
                  frame=preds)
    handles = [plt.Line2D([], [], color=INK, lw=2, marker="o", ms=3, label="history"),
               plt.Line2D([], [], color=INK, lw=2, ls="--", marker="o", ms=3, label="ground truth")]
    handles += [plt.Line2D([], [], color=COLORS[m], lw=1.5, label=f"{NAMES[m]} samples") for m in preds]
    handles += [plt.Line2D([], [], color="#b9b8b0", lw=1, label="lane centerline"),
                plt.Line2D([], [], color="#a3a29a", lw=1, label="neighbor history"),
                plt.Rectangle((0, 0), 1, 1, fc="#dedcd4", label="off drivable area"),
                plt.Rectangle((0, 0), 1, 1, fc="#c9c8c1", ec=INK2, lw=0.6, label="static obstacle")]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_size_inches()[1]))
    fig.savefig(out / f"scenes_{pick}.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/fm_trainval")
    ap.add_argument("--cache", default="datasets/nuscenes_trainval")
    ap.add_argument("--split", default="train_val")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--picks", nargs="+", default=["random", "fm_worst", "yflow_gain", "yflow_loss"])
    args = ap.parse_args()
    run = Path(args.run)
    out = run / "viz"
    out.mkdir(parents=True, exist_ok=True)
    made = []
    if (run / "train_log.jsonl").exists():
        training_curves(run, out)
        made.append("training_curves.png")
    metrics = {m: json.loads((run / f"metrics_{m}_{args.split}_seed0.json").read_text())
               for m in ("cv", "fm", "proj", "yflow") if (run / f"metrics_{m}_{args.split}_seed0.json").exists()}
    if metrics:
        hard = ()
        if (run / "config.yaml").exists():
            import yaml
            hard = tuple((yaml.safe_load((run / "config.yaml").read_text()).get("constraints") or {}).get("hard", ()))
        violations(metrics, out, hard or ("speed", "continuity", "accel", "drivable", "static"))
        made.append("constraint_violations.png")
        cols = ("K", "minADE", "minFDE", "MR", "minADE_5", "minFDE_5", "all_hard_safe", "ms_per_scene")
        print(f"{'':8}" + "".join(f"{c:>14}" for c in cols))
        for m, r in metrics.items():
            k = int(r["K"])
            mr = next((v for kk, v in r.items() if kk.startswith(f"MR_{k}@")), float("nan"))
            vals = (k, r.get(f"minADE_{k}"), r.get(f"minFDE_{k}"), mr, r.get("minADE_5"), r.get("minFDE_5"),
                    r.get("all_hard_safe"), r.get("ms_per_scene"))
            print(f"{m:8}" + "".join(f"{'-':>14}" if v is None else f"{v:>14.4f}" for v in vals))
    preds = {m: np.load(run / f"pred_{m}_{args.split}_seed0.npy") for m in ("fm", "proj", "yflow")
             if (run / f"pred_{m}_{args.split}_seed0.npy").exists()}
    if preds:
        meta = load_meta(args.cache)
        a = load_split(args.cache, args.split)
        n = min(len(a["fut"]), *(len(p) for p in preds.values()))
        a = {k: v[:n] for k, v in a.items()}
        preds = {m: p[:n] for m, p in preds.items()}
        horizon_error(preds, a, float(meta["sample_hz"]), out)
        made.append("horizon_error.png")
        for pick in args.picks:
            if pick in ("yflow_gain", "yflow_loss") and "yflow" not in preds:
                continue
            scenes(a, meta, preds, pick, args.n, args.seed, out)
            made.append(f"scenes_{pick}.png")
    print("wrote", ", ".join(str(out / m) for m in made) if made else "nothing (no logs/metrics/predictions yet)")


if __name__ == "__main__":
    main()
