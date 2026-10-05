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

from vfm.data import load_meta, load_split
from vfm.metrics import displacement

INK, INK2, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
COLORS = {"cv": "#8a8983", "fm": "#2a78d6", "proj": "#1f9d61", "yflow": "#eb6834", "gt": INK}
NAMES = {"cv": "Const. velocity", "fm": "FM (no guidance)", "proj": "FM + post-hoc projection",
         "yflow": "FM + Y-Flow", "gt": "Ground truth"}
EXTRA = ["#7b4fb5", "#c9a227", "#0f8c8c", "#b5446e", "#4f6d7a"]
OFFROAD_FC = "#dedcd4"
OFF_CMAP = ListedColormap([OFFROAD_FC])

def _cjk_font() -> str | None:
    from matplotlib import font_manager
    want = ("NanumGothic", "Nanum Gothic", "Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic", "AppleGothic",
            "NanumBarunGothic", "UnDotum", "Source Han Sans KR")
    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((w for w in want if w in have), None)


CJK = _cjk_font()
if CJK:
    plt.rcParams["font.family"] = CJK
    plt.rcParams["axes.unicode_minus"] = False

plt.rcParams.update({
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "axes.titlecolor": INK,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def save(fig, out: Path, name: str, made: list[str]) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    made.append(f"{name}.png")


def label(name: str) -> str:
    return NAMES.get(name, name)


def register(names) -> None:
    for n in names:
        if n not in COLORS:
            COLORS[n] = EXTRA[sum(1 for v in COLORS.values() if v in EXTRA) % len(EXTRA)]


def parse_pairs(items) -> dict:
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = v
    return out


def mr_key(m: dict) -> str | None:
    return next((k for k in m if k.startswith("MR_") and "@" in k), None)


def k_of(r: dict) -> int:
    return int(r.get("K", 1))


def value(r: dict, kind: str) -> float:
    k = k_of(r)
    alts = {
        "minADE": [f"minADE_{k}", "minADE_1", "ADE_1"],
        "minFDE": [f"minFDE_{k}", "minFDE_1", "FDE_1"],
        "MR": [mr_key(r)],
        "offroad": ["indep_offroad_scene_gtok", "scene_footprint_offroad_rate", "footprint_offroad_rate",
                    "scene_viol_rate_drivable_fp", "scene_viol_rate_drivable", "viol_rate_drivable"],
        "coll": ["indep_coll_scene_gtok", "scene_viol_rate_coll", "viol_rate_coll"],
        "hard_safe": ["scene_all_hard_safe", "all_hard_safe"],
    }[kind]
    for a in alts:
        if a and a in r:
            return float(r[a])
    return float("nan")


ROWS = [("minADE", "minADE$_K$ (m)", 1.0, "\u2193"), ("minFDE", "minFDE$_K$ (m)", 1.0, "\u2193"),
        ("MR", "MR$_K$ (%)", 100.0, "\u2193"), ("offroad", "off-road scenes (%)", 100.0, "\u2193"),
        ("coll", "collision scenes (%)", 100.0, "\u2193"),
        ("hard_safe", "all hard constraints met (%)", 100.0, "\u2191")]


def panel_bars(ax, metrics: dict, kind: str, title: str, scale: float, better: str) -> None:
    v = [value(metrics[m], kind) * scale for m in metrics]
    x = np.arange(len(metrics))
    ax.bar(x, v, 0.68, color=[COLORS.get(m, INK2) for m in metrics])
    for xi, vi in zip(x, v):
        if np.isfinite(vi):
            ax.annotate(f"{vi:.2f}" if abs(vi) < 10 else f"{vi:.1f}", (xi, vi), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x, ["" for _ in metrics])
    ax.set_title(f"{title}  ({better})", fontsize=9)
    ax.margins(y=0.22)
    ax.tick_params(bottom=False)


def fig_accuracy_safety(metrics: dict, out: Path, made: list[str]) -> None:
    rows = [r for r in ROWS if any(np.isfinite(value(m, r[0])) for m in metrics.values())]
    fig, axes = plt.subplots(1, len(rows), figsize=(2.3 * len(rows), 3.4), squeeze=False)
    for ax, (k, t, s, b) in zip(axes[0], rows):
        panel_bars(ax, metrics, k, t, s, b)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS.get(m, INK2), label=f"{label(m)}  (K={k_of(metrics[m])})")
               for m in metrics]
    fig.legend(handles=handles, loc="upper center", ncol=len(metrics), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, out, "fig1_accuracy_safety", made)


def pareto(points: np.ndarray) -> np.ndarray:
    order = np.argsort(points[:, 0])
    keep, best = [], np.inf
    for i in order:
        if points[i, 1] < best - 1e-12:
            keep.append(i)
            best = points[i, 1]
    return np.array(keep, dtype=int)


def fig_tradeoff(metrics: dict, trials: list[dict], out: Path, made: list[str]) -> None:
    xk, yk = "minADE", "offroad"
    pts = {m: (value(r, xk), value(r, yk)) for m, r in metrics.items() if m != "cv"}
    pts = {m: v for m, v in pts.items() if np.isfinite(v[0]) and np.isfinite(v[1])}
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    tp = np.array([[value(t, xk), value(t, yk)] for t in trials], dtype=float) if trials else np.empty((0, 2))
    tp = tp[np.isfinite(tp).all(1)] if len(tp) else tp
    if len(tp):
        ax.scatter(tp[:, 0], 100 * tp[:, 1], s=16, color="#c9c8c1", edgecolor="none", zorder=1,
                   label=f"Y-Flow tuning trials (n={len(tp)})")
        f = pareto(tp)
        ax.plot(tp[f, 0], 100 * tp[f, 1], color="#c9c8c1", lw=1.2, ls="--", zorder=1, label="trial Pareto front")
    base = pts.get("fm")
    for m, (x, y) in pts.items():
        lbl = label(m)
        if base and m != "fm" and base[0]:
            lbl += f"  ({100 * (x / base[0] - 1):+.1f}% minADE, {100 * (y - base[1]):+.1f} pt off-road)"
        ax.scatter([x], [100 * y], s=110, color=COLORS.get(m, INK2), zorder=3, label=lbl)
    ax.set(xlabel="minADE$_K$ (m)  \u2014 accuracy cost",
           ylabel="off-road scenes (%)  \u2014 constraint violation",
           title="Accuracy vs constraint satisfaction")
    ax.margins(0.12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=7.5, ncol=1)
    fig.tight_layout()
    save(fig, out, "fig2_tradeoff", made)


def fig_horizon(preds: dict, a: dict, hz: float, out: Path, made: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    t = np.arange(1, a["fut"].shape[1] + 1) / hz
    for m, p in preds.items():
        ade, _ = displacement(p, a["fut"], a["fut_mask"])
        best = ade.argmin(1)
        d = np.linalg.norm(p[np.arange(len(p)), best] - a["fut"], axis=-1)
        y = np.nanmean(np.where(a["fut_mask"], d, np.nan), axis=0)
        ax.plot(t, y, color=COLORS.get(m, INK2), lw=2, marker="o", ms=4, label=label(m))
    ax.set(title="Error vs horizon (sample chosen by minADE)", xlabel="horizon (s)", ylabel="mean L2 error (m)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    save(fig, out, "fig3_horizon", made)


def fig_violations(metrics: dict, hard: tuple, out: Path, made: list[str]) -> None:
    names = sorted({k[len("viol_rate_"):] for m in metrics.values() for k in m if k.startswith("viol_rate_")})
    if not names:
        return
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(names))
    w = 0.8 / len(metrics)
    for i, (m, r) in enumerate(metrics.items()):
        v = [100 * r.get(f"viol_rate_{n}", np.nan) for n in names]
        xs = x + (i - (len(metrics) - 1) / 2) * w
        ax.bar(xs, v, w * 0.9, color=COLORS.get(m, INK2), label=label(m))
        for xi, vi in zip(xs, v):
            if np.isfinite(vi):
                ax.annotate(f"{vi:.1f}" if vi < 10 else f"{vi:.0f}", (xi, vi), xytext=(0, 2),
                            textcoords="offset points", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(x, [f"{n}\n(hard)" if n in hard else f"{n}\n(soft)" for n in names], fontsize=8)
    ax.set(title="Predicted trajectories violating each constraint", ylabel="% of trajectories")
    ax.margins(y=0.18)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, out, "fig4_violations", made)


def fig_cost(metrics: dict, out: Path, made: list[str]) -> None:
    pts = {m: (r.get("ms_per_scene", np.nan), value(r, "offroad")) for m, r in metrics.items()}
    pts = {m: v for m, v in pts.items() if np.isfinite(v[0]) and np.isfinite(v[1])}
    if len(pts) < 2:
        return
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for m, (x, y) in pts.items():
        ax.scatter([x], [100 * y], s=90, color=COLORS.get(m, INK2), zorder=3)
        ax.annotate(label(m), (x, 100 * y), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
    ax.set(xlabel="inference time (ms / scene)", ylabel="off-road scenes (%)",
           title="Cost of enforcing the constraints")
    ax.margins(0.2)
    fig.tight_layout()
    save(fig, out, "fig5_cost", made)


def sdf_at(sdf: np.ndarray, g: dict, p: np.ndarray) -> np.ndarray:
    res, n = float(g["res"]), int(g["size"])
    fx = (p[..., 0] - float(g["x_min"])) / res - 0.5
    fy = (p[..., 1] - float(g["y_min"])) / res - 0.5
    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    tx, ty = fx - x0, fy - y0
    ok = (x0 >= 0) & (y0 >= 0) & (x0 < n - 1) & (y0 < n - 1)
    xc, yc = np.clip(x0, 0, n - 2), np.clip(y0, 0, n - 2)
    v = (sdf[yc, xc] * (1 - tx) * (1 - ty) + sdf[yc, xc + 1] * tx * (1 - ty)
         + sdf[yc + 1, xc] * (1 - tx) * ty + sdf[yc + 1, xc + 1] * tx * ty) * float(g["scale"])
    return np.where(ok, v, 0.0)


def _box(ax, o, fc="#c9c8c1", z=2):
    x, y, yaw, l, w = o
    c, s = np.cos(yaw), np.sin(yaw)
    pts = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]) / 2 @ np.array([[c, s], [-s, c]]) + [x, y]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=INK2, lw=0.6, zorder=z))


def scene(ax, a: dict, i: int, meta: dict, preds: dict, title: str, frame: dict, margin: float) -> None:
    fut, fm = a["fut"][i], a["fut_mask"][i]
    view = np.concatenate([a["hist"][i][a["hist_mask"][i]], fut[fm]] + [p[i].reshape(-1, 2) for p in frame.values()])
    lo, hi = view.min(0), view.max(0)
    c, r = (lo + hi) / 2, max(15.0, float((hi - lo).max()) / 2 + 8)
    g = meta.get("sdf")
    if g and "sdf" in a:
        ext = [g["x_min"], g["x_min"] + g["res"] * g["size"], g["y_min"], g["y_min"] + g["res"] * g["size"]]
        off = np.ma.masked_where(a["sdf"][i] <= 0, np.ones_like(a["sdf"][i], dtype=float))
        ax.imshow(off, origin="lower", extent=ext, cmap=OFF_CMAP, zorder=0, interpolation="nearest")
    for l, m in zip(a["lane"][i], a["lane_mask"][i]):
        if m.any():
            ax.plot(l[m, 0], l[m, 1], color="#b9b8b0", lw=0.8, zorder=1)
    for nb, m in zip(a["nbr"][i], a["nbr_mask"][i]):
        if m.any():
            ax.plot(nb[m, 0], nb[m, 1], color="#a3a29a", lw=1, zorder=2)
            ax.scatter(nb[m][-1:, 0], nb[m][-1:, 1], s=10, color="#a3a29a", zorder=2)
    for o, m in zip(a.get("obs", np.zeros((0, 0, 5)))[i] if "obs" in a else [], a.get("obs_mask", [[]])[i]):
        if m:
            _box(ax, o)
    for mth, p in preds.items():
        col = COLORS.get(mth, INK2)
        for k in range(p.shape[1]):
            xy = np.concatenate([np.zeros((1, 2)), p[i, k]])
            ax.plot(xy[:, 0], xy[:, 1], color=col, lw=1.2, alpha=0.6, zorder=3)
            if g and "sdf" in a:
                bad = sdf_at(a["sdf"][i], g, p[i, k]) > margin
                if bad.any():
                    ax.scatter(p[i, k][bad, 0], p[i, k][bad, 1], s=9, color="#d1344a", zorder=5, linewidths=0)
    h = a["hist"][i][a["hist_mask"][i]]
    ax.plot(h[:, 0], h[:, 1], color=INK, lw=2, marker="o", ms=3, zorder=4)
    ax.plot(np.r_[0, fut[fm, 0]], np.r_[0, fut[fm, 1]], color=INK, lw=2, ls="--", marker="o", ms=3, zorder=5)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8.5)
    ax.tick_params(labelsize=7)


def offroad_rate(a: dict, meta: dict, p: np.ndarray, margin: float) -> np.ndarray:
    g = meta.get("sdf")
    if not g or "sdf" not in a:
        return np.zeros(len(p))
    return np.array([(sdf_at(a["sdf"][i], g, p[i]) > margin).any(-1).mean() for i in range(len(p))])


def pick_scenes(a: dict, meta: dict, preds: dict, pick: str, n: int, seed: int, margin: float) -> np.ndarray:
    ade = {m: displacement(p, a["fut"], a["fut_mask"])[0].min(1) for m, p in preds.items()}
    off = {m: offroad_rate(a, meta, p, margin) for m, p in preds.items()}
    have = set(preds)
    if pick == "offroad_fixed" and "fm" in have:
        other = [m for m in ("yflow", "proj") if m in have]
        gain = off["fm"] - (np.minimum(*[off[m] for m in other]) if len(other) == 2 else
                            off[other[0]] if other else 0.0)
        order = np.argsort(-gain)
    elif pick == "yflow_vs_proj" and {"yflow", "proj"} <= have:
        order = np.argsort(-np.abs(ade["yflow"] - ade["proj"]))
    elif pick == "accuracy_loss" and "fm" in have and len(have) > 1:
        other = [m for m in ("yflow", "proj") if m in have]
        loss = np.max([ade[m] - ade["fm"] for m in other], axis=0) if other else np.zeros(len(a["fut"]))
        order = np.argsort(-loss)
    elif pick == "fm_offroad" and "fm" in have:
        order = np.argsort(-off["fm"])
    elif pick == "fm_worst" and "fm" in have:
        order = np.argsort(-ade["fm"])
    else:
        order = np.random.default_rng(seed).permutation(len(a["fut"]))
    return order[:n]


def fig_scenes(a: dict, meta: dict, preds: dict, pick: str, n: int, seed: int, margin: float, out: Path,
               made: list[str]) -> None:
    sel = pick_scenes(a, meta, preds, pick, n, seed, margin)
    ade = {m: displacement(p, a["fut"], a["fut_mask"])[0].min(1) for m, p in preds.items()}
    off = {m: offroad_rate(a, meta, p, margin) for m, p in preds.items()}
    cols = len(preds)
    fig, axes = plt.subplots(len(sel), cols, figsize=(4.0 * cols, 3.9 * len(sel)), squeeze=False)
    for r, i in enumerate(sel):
        i = int(i)
        for c, (m, p) in enumerate(preds.items()):
            scene(axes[r, c], a, i, meta, {m: p}, f"{label(m)}   scene {i}\nminADE {ade[m][i]:.2f} m   "
                  f"off-road {100 * off[m][i]:.0f}% of samples", frame=preds, margin=margin)
    handles = [plt.Line2D([], [], color=INK, lw=2, marker="o", ms=3, label="history"),
               plt.Line2D([], [], color=INK, lw=2, ls="--", marker="o", ms=3, label="ground truth")]
    handles += [plt.Line2D([], [], color=COLORS.get(m, INK2), lw=1.5, label=f"{label(m)} samples") for m in preds]
    handles += [plt.Line2D([], [], color="#b9b8b0", lw=1, label="lane centerline"),
                plt.Line2D([], [], color="#a3a29a", lw=1, label="neighbor history"),
                plt.Line2D([], [], color="#d1344a", lw=0, marker="o", ms=4, label="off-road step"),
                plt.Rectangle((0, 0), 1, 1, fc=OFFROAD_FC, label="off drivable area"),
                plt.Rectangle((0, 0), 1, 1, fc="#c9c8c1", ec=INK2, lw=0.6, label="static obstacle")]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.7 / fig.get_size_inches()[1]))
    save(fig, out, f"fig6_scenes_{pick}", made)


def table(metrics: dict) -> str:
    cols = [("minADE", "minADE_K"), ("minFDE", "minFDE_K"), ("MR", "MR"), ("offroad", "offroad"),
            ("coll", "coll"), ("hard_safe", "hard_safe")]
    lines = [f"{'method':<26}{'K':>4}" + "".join(f"{c:>12}" for _, c in cols) + f"{'ms/scene':>12}"]
    for m, r in metrics.items():
        vals = [value(r, kind) for kind, _ in cols]
        lines.append(f"{label(m):<26}{k_of(r):>4}"
                     + "".join(f"{'-':>12}" if not np.isfinite(v) else f"{v:>12.4f}" for v in vals)
                     + (f"{r['ms_per_scene']:>12.2f}" if "ms_per_scene" in r else f"{'-':>12}"))
    missing = sorted({kind for kind, _ in cols if all(not np.isfinite(value(r, kind)) for r in metrics.values())})
    if missing:
        lines.append(f"missing everywhere: {', '.join(missing)}  "
                     f"(re-run vfm.evaluate with a `constraints` block to get them)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/fm_trainval")
    ap.add_argument("--cache", default="datasets/nuscenes")
    ap.add_argument("--split", default="val")
    ap.add_argument("--methods", nargs="+", default=["cv", "fm", "proj", "yflow"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tune", default=None, help="tune trials.jsonl (default {run}/tune/trials.jsonl)")
    ap.add_argument("--n", type=int, default=4, help="scenes per qualitative figure")
    ap.add_argument("--picks", nargs="+", default=["offroad_fixed", "yflow_vs_proj", "accuracy_loss"])
    ap.add_argument("--margin", type=float, default=0.0, help="SDF threshold (m) counted as off-road in the plots")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pred", nargs="+", default=None, metavar="NAME=PATH",
                    help="extra prediction files, e.g. yflow_nolane=outputs/abl/pred_yflow_val_seed0.npy")
    ap.add_argument("--metrics", nargs="+", default=None, metavar="NAME=PATH", help="extra metrics json files")
    ap.add_argument("--label", nargs="+", default=None, metavar="NAME=TEXT", help="display name for a method")
    args = ap.parse_args()
    NAMES.update(parse_pairs(args.label))
    if CJK is None and any(ord(c) > 0x2000 for v in NAMES.values() for c in v):
        print("[figures] 한글 폰트가 없어 라벨이 네모로 나와. 'apt-get install -y fonts-nanum' 후 "
              "'rm -rf ~/.cache/matplotlib' 하거나 --label 을 영문으로 줘.", flush=True)
    run = Path(args.run)
    out = Path(args.out) if args.out else run / "figs"
    out.mkdir(parents=True, exist_ok=True)
    made: list[str] = []

    metrics = {}
    for m in args.methods:
        f = run / f"metrics_{m}_{args.split}_seed{args.seed}.json"
        if f.exists():
            metrics[m] = json.loads(f.read_text())
    for n, f in parse_pairs(args.metrics).items():
        metrics[n] = json.loads(Path(f).read_text())
    register(metrics)
    trials = []
    tf = Path(args.tune) if args.tune else run / "tune" / "trials.jsonl"
    if tf.exists():
        trials = [json.loads(l) for l in tf.read_text().splitlines() if l.strip()]
        trials = [t.get("metrics", t) for t in trials]
    hard = ("speed", "continuity", "accel", "drivable", "static")
    if (run / "config.yaml").exists():
        import yaml
        hard = tuple((yaml.safe_load((run / "config.yaml").read_text()).get("constraints") or {}).get("hard", hard))
    if metrics:
        fig_accuracy_safety(metrics, out, made)
        fig_tradeoff(metrics, trials, out, made)
        fig_violations(metrics, hard, out, made)
        fig_cost(metrics, out, made)
        print(table(metrics))
        (out / "table.txt").write_text(table(metrics) + "\n")

    preds = {}
    for m in args.methods:
        f = run / f"pred_{m}_{args.split}_seed{args.seed}.npy"
        if f.exists():
            preds[m] = np.load(f)
    for n, f in parse_pairs(args.pred).items():
        preds[n] = np.load(f)
    register(preds)
    if preds:
        meta = load_meta(args.cache)
        a = load_split(args.cache, args.split)
        n = min(len(a["fut"]), *(len(p) for p in preds.values()))
        a = {k: v[:n] for k, v in a.items()}
        preds = {m: p[:n] for m, p in preds.items()}
        fig_horizon(preds, a, float(meta["sample_hz"]), out, made)
        for pick in args.picks:
            fig_scenes(a, meta, preds, pick, args.n, args.seed, args.margin, out, made)
    print("wrote", ", ".join(str(out / m) for m in made) if made else "nothing (no metrics/predictions found)")


if __name__ == "__main__":
    main()
