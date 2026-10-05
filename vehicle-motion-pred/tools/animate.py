from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from tools.figures import CJK, COLORS, GRID as GRID_EC, INK, INK2, NAMES, OFF_CMAP, OFFROAD_FC, label, \
    offroad_rate, parse_pairs, pick_scenes, register, sdf_at
from vfm.data import load_meta, load_split
from vfm.indep import load_agents
from vfm.metrics import displacement


def box(center: np.ndarray, yaw: float, length: float, width: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[length, width], [length, -width], [-length, -width], [-length, width]]) / 2 \
        @ np.array([[c, s], [-s, c]]) + center


def upsample(x: np.ndarray, sub: int) -> np.ndarray:
    if sub <= 1:
        return x
    T = x.shape[-2]
    src = np.arange(T, dtype=np.float64)
    dst = np.linspace(0.0, T - 1, (T - 1) * sub + 1)
    flat = x.reshape(-1, T, 2)
    out = np.empty((flat.shape[0], len(dst), 2), dtype=np.float32)
    for j in range(2):
        out[..., j] = np.stack([np.interp(dst, src, f[:, j]) for f in flat])
    return out.reshape(*x.shape[:-2], len(dst), 2)


def heading(path: np.ndarray, k: int, fallback: float = 0.0) -> float:
    a, b = path[max(k - 1, 0)], path[min(k + 1, len(path) - 1)]
    d = b - a
    return float(np.arctan2(d[1], d[0])) if np.hypot(*d) > 1e-3 else fallback


def scene_panel(ax, a: dict, i: int, meta: dict, xlim, ylim, clean: bool = True) -> None:
    g = meta.get("sdf")
    if g and "sdf" in a:
        ext = [g["x_min"], g["x_min"] + g["res"] * g["size"], g["y_min"], g["y_min"] + g["res"] * g["size"]]
        off = np.ma.masked_where(a["sdf"][i] <= 0, np.ones_like(a["sdf"][i], dtype=float))
        ax.imshow(off, origin="lower", extent=ext, cmap=OFF_CMAP, zorder=0, interpolation="nearest")
    for l, m in zip(a["lane"][i], a["lane_mask"][i]):
        if m.any():
            ax.plot(l[m, 0], l[m, 1], color="#b9b8b0", lw=0.8, zorder=1)
    if "obs" in a:
        for o, m in zip(a["obs"][i], a["obs_mask"][i]):
            if m:
                ax.add_patch(Polygon(box(o[:2], o[2], o[3], o[4]), closed=True, fc="#c9c8c1", ec=INK2, lw=0.6,
                                     zorder=2))
    h = a["hist"][i][a["hist_mask"][i]]
    ax.plot(h[:, 0], h[:, 1], color=INK2, lw=1.8, ls=":", zorder=3)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    if clean:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_facecolor("#fbfaf7")
    else:
        ax.tick_params(labelsize=7)


def animate_scene(a: dict, meta: dict, preds: dict, i: int, out: Path, fps: int, fmt: str, margin: float,
                  trail: bool, sub: int, hold: float, follow: bool, zoom: float, ghost: bool,
                  clean: bool, show: str, flash: bool, ag: dict | None = None) -> Path:
    fut, fm = a["fut"][i], a["fut_mask"][i]
    hz = float(meta["sample_hz"])
    gt = upsample(np.concatenate([np.zeros((1, 2)), fut]), sub)
    T = gt.shape[0] - 1
    fps_data = hz * sub
    size = a["focal_size"][i] if "focal_size" in a else np.array([4.5, 1.9], np.float32)
    nbr, nmask = a["nbr"][i], a["nbr_mask"][i]
    if ag is not None:
        pos = np.concatenate([ag["ag_fut"][i][:, :1], ag["ag_fut"][i]], axis=1)
        yaw = np.concatenate([ag["ag_yaw"][i][:, :1], ag["ag_yaw"][i]], axis=1)
        nb_pos = upsample(pos, sub)
        fwd = upsample(np.stack([np.cos(yaw), np.sin(yaw)], -1), sub)
        nb_yaw = np.arctan2(fwd[..., 1], fwd[..., 0])
        nb_size = ag["ag_size"][i]
        nb_valid_t = np.concatenate([ag["ag_mask"][i][:, :1], ag["ag_mask"][i]], axis=1).repeat(sub, 1)[:, :nb_pos.shape[1]]
        nb_src = "ground truth"
    else:
        nsize = a["nbr_size"][i] if "nbr_size" in a else np.tile([4.5, 1.9], (len(nbr), 1))
        ny = a["nbr_yaw"][i] if "nbr_yaw" in a else np.zeros(len(nbr))
        last = nbr[:, -1]
        v = np.zeros((len(nbr), 2), np.float32)
        if nbr.shape[1] > 1:
            v = np.where((nmask[:, -1] & nmask[:, -2])[:, None], nbr[:, -1] - nbr[:, -2], 0.0)
        steps = np.arange(fut.shape[0] * sub + 1)[None, :, None] / sub
        nb_pos = last[:, None] + v[:, None] * steps
        nb_yaw = np.repeat(ny[:, None], nb_pos.shape[1], 1)
        nb_size = nsize
        nb_valid_t = np.repeat(nmask[:, -1:], nb_pos.shape[1], 1)
        nb_src = "const. velocity"
    valid = nb_valid_t.any(1)

    view = np.concatenate([a["hist"][i][a["hist_mask"][i]], gt] + [p[i].reshape(-1, 2) for p in preds.values()])
    lo, hi = view.min(0), view.max(0)
    c, r = (lo + hi) / 2, max(15.0, float((hi - lo).max()) / 2 + 8)
    xlim, ylim = (c[0] - r, c[0] + r), (c[1] - r, c[1] + r)

    ades = {m: displacement(p[i:i + 1], fut[None], fm[None])[0][0] for m, p in preds.items()}
    ade = {m: float(v.min()) for m, v in ades.items()}
    up = {m: upsample(np.concatenate([np.zeros((p.shape[1], 1, 2)), p[i]], axis=1), sub)[:, 1:] for m, p in preds.items()}
    draw = {m: (np.array([int(ades[m].argmin())]) if show == "best" else np.arange(up[m].shape[0]))
            for m in preds}
    g = meta.get("sdf")
    cols = len(preds)
    fig, axes = plt.subplots(1, cols, figsize=(4.4 * cols, 4.8), squeeze=False)
    art = []
    for ax, (m, p) in zip(axes[0], preds.items()):
        scene_panel(ax, a, i, meta, xlim, ylim, clean)
        ax.set_title(f"{label(m)}\nscene {i}   minADE {ade[m]:.2f} m", fontsize=9)
        col = COLORS.get(m, INK2)
        K = p.shape[1]
        idx = draw[m]
        if ghost:
            for k in range(K):
                xy = np.concatenate([np.zeros((1, 2)), up[m][k]])
                ax.plot(xy[:, 0], xy[:, 1], color=col, lw=1.0, alpha=0.10 if show == "best" else 0.16, zorder=3)
        lw = 3.4 if show == "best" else 2.0
        samples = [ax.plot([], [], color=col, lw=lw, alpha=0.9, zorder=4,
                           solid_capstyle="round")[0] for _ in idx]
        heads = ax.scatter([], [], s=70 if show == "best" else 26, color=col, zorder=5, linewidths=0)
        bad = ax.scatter([], [], s=44, color="#d1344a", zorder=6, linewidths=0)
        border = Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc="none", ec="#d1344a", lw=0, zorder=10)
        ax.add_patch(border)
        alarm = ax.text(0.5, 0.06, "", transform=ax.transAxes, ha="center", fontsize=16, color="#d1344a",
                        fontweight="bold", zorder=11)
        verdict = ax.text(0.5, 0.80, "", transform=ax.transAxes, ha="center", va="center", fontsize=11,
                          color=INK, fontweight="bold", zorder=12, linespacing=1.5,
                          bbox=dict(fc="white", ec=GRID_EC, alpha=0.93, pad=5))
        gt_line, = ax.plot([], [], color=INK, lw=2.4, ls="--", zorder=5)
        focal = Polygon(box(np.zeros(2), 0.0, size[0], size[1]), closed=True, fc=INK, ec=INK, alpha=0.85, zorder=7)
        ax.add_patch(focal)
        nb_boxes = [Polygon(box(nb_pos[j, 0], float(nb_yaw[j, 0]), nb_size[j][0], nb_size[j][1]), closed=True,
                            fc="#9a9992", ec=INK2, lw=0.6, alpha=0.9, zorder=6) for j in range(len(nb_pos))]
        for j, b in enumerate(nb_boxes):
            b.set_visible(bool(valid[j]))
            ax.add_patch(b)
        clock = ax.text(0.03, 0.96, "", transform=ax.transAxes, va="top", fontsize=11, color=INK,
                        bbox=dict(fc="white", ec="none", alpha=0.7, pad=2))
        art.append({"ax": ax, "m": m, "p": up[m][idx], "all": up[m], "heads": heads, "samples": samples,
                    "border": border, "alarm": alarm, "verdict": verdict, "bad": bad, "gt": gt_line, "focal": focal,
                    "nb": nb_boxes, "clock": clock})

    def frame(k: int):
        out_art = []
        for d in art:
            pts = []
            for s, line in enumerate(d["samples"]):
                xy = np.concatenate([np.zeros((1, 2)), d["p"][s, :k]])
                seg = xy if trail else xy[max(0, len(xy) - 2):]
                line.set_data(seg[:, 0], seg[:, 1])
                if g and "sdf" in a and k:
                    off = sdf_at(a["sdf"][i], g, d["p"][s, :k]) > margin
                    if off.any():
                        pts.append(d["p"][s, :k][off])
            d["bad"].set_offsets(np.concatenate(pts) if pts else np.empty((0, 2)))
            d["heads"].set_offsets(d["p"][:, k - 1] if k else np.zeros((d["p"].shape[0], 2)))
            now_off = False
            if g and "sdf" in a and k:
                now_off = bool((sdf_at(a["sdf"][i], g, d["p"][:, k - 1]) > margin).any())
            if flash:
                d["border"].set_linewidth(7 if now_off else 0)
                d["alarm"].set_text("OFF-ROAD" if now_off else "")
            if k >= T:
                frac = (sdf_at(a["sdf"][i], g, d["all"]) > margin).any(-1).mean() if (g and "sdf" in a) else np.nan
                d["verdict"].set_text(f"{label(d['m'])}\nminADE {ade[d['m']]:.2f} m"
                                      + (f"\noff-road {100 * frac:.0f}% of samples" if np.isfinite(frac) else ""))
            else:
                d["verdict"].set_text("")
            d["gt"].set_data(gt[:k + 1, 0], gt[:k + 1, 1])
            yaw = heading(gt, k)
            d["focal"].set_xy(box(gt[k], yaw, size[0], size[1]))
            for j, b in enumerate(d["nb"]):
                kk = min(k, nb_pos.shape[1] - 1)
                on = bool(valid[j] and nb_valid_t[j, kk])
                b.set_visible(on)
                if on:
                    b.set_xy(box(nb_pos[j, kk], float(nb_yaw[j, kk]), nb_size[j][0], nb_size[j][1]))
            d["clock"].set_text(f"t = {k / fps_data:.1f} s")
            if follow:
                d["ax"].set_xlim(gt[k, 0] - zoom, gt[k, 0] + zoom)
                d["ax"].set_ylim(gt[k, 1] - zoom, gt[k, 1] + zoom)
            out_art += d["samples"] + [d["bad"], d["heads"], d["gt"], d["focal"], d["clock"], d["border"],
                                       d["alarm"], d["verdict"]] + d["nb"]
        return out_art

    handles = [plt.Line2D([], [], color=INK, lw=2, ls="--", label="ground truth"),
               plt.Line2D([], [], color=INK2, lw=1.6, ls=":", label="history"),
               plt.Rectangle((0, 0), 1, 1, fc=INK, label="focal vehicle"),
               plt.Rectangle((0, 0), 1, 1, fc="#9a9992", ec=INK2, label=f"neighbour ({nb_src})"),
               plt.Line2D([], [], color="#d1344a", lw=0, marker="o", ms=4, label="off-road step"),
               plt.Rectangle((0, 0), 1, 1, fc=OFFROAD_FC, label="off drivable area")]
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    frames = list(range(T + 1)) + [T] * int(round(hold * fps))
    anim = animation.FuncAnimation(fig, frame, frames=frames, interval=1000 / fps, blit=False)
    path = out / f"scene{i}.{fmt}"
    if fmt == "mp4":
        try:
            anim.save(str(path), writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
        except Exception as e:
            print(f"  ffmpeg unavailable ({e.__class__.__name__}), writing a gif instead")
            path = out / f"scene{i}.gif"
            anim.save(str(path), writer=animation.PillowWriter(fps=fps))
    else:
        anim.save(str(path), writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/fm_trainval")
    ap.add_argument("--cache", default="datasets/nuscenes")
    ap.add_argument("--split", default="val")
    ap.add_argument("--methods", nargs="+", default=["fm", "proj", "yflow"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pick", default="offroad_fixed",
                    help="offroad_fixed | yflow_vs_proj | accuracy_loss | fm_offroad | fm_worst | random")
    ap.add_argument("--scenes", nargs="+", type=int, default=None, help="explicit scene indices")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--fps", type=int, default=12, help="playback frame rate")
    ap.add_argument("--sub", type=int, default=6, help="interpolated frames per 0.5 s data step (smoothness)")
    ap.add_argument("--hold", type=float, default=1.0, help="seconds to freeze on the final step")
    ap.add_argument("--format", default="mp4", choices=["mp4", "gif"])
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--no_trail", action="store_true", help="draw only the current segment of each sample")
    ap.add_argument("--no_follow", action="store_true", help="keep the whole scene in view instead of riding along")
    ap.add_argument("--zoom", type=float, default=32.0, help="half-window (m) of the following camera")
    ap.add_argument("--no_ghost", action="store_true", help="hide the faint full predictions")
    ap.add_argument("--axes", action="store_true", help="keep plot axes, ticks and grid")
    ap.add_argument("--show", default="best", choices=["best", "all"],
                    help="best (default): animate only the sample closest to the ground truth; all: every sample")
    ap.add_argument("--no_flash", action="store_true", help="no red border / OFF-ROAD warning")
    ap.add_argument("--cv_neighbours", action="store_true",
                    help="extrapolate the neighbours at constant velocity even if their recorded futures exist")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pred", nargs="+", default=None, metavar="NAME=PATH", help="extra prediction files")
    ap.add_argument("--label", nargs="+", default=None, metavar="NAME=TEXT", help="display name for a method")
    args = ap.parse_args()
    NAMES.update(parse_pairs(args.label))
    if CJK is None and any(ord(c) > 0x2000 for v in NAMES.values() for c in v):
        print("[animate] 한글 폰트가 없어 라벨이 네모로 나와. 'apt-get install -y fonts-nanum' 후 "
              "'rm -rf ~/.cache/matplotlib' 하거나 --label 을 영문으로 줘.", flush=True)
    run = Path(args.run)
    out = Path(args.out) if args.out else run / "anim"
    out.mkdir(parents=True, exist_ok=True)
    preds = {m: np.load(run / f"pred_{m}_{args.split}_seed{args.seed}.npy") for m in args.methods
             if (run / f"pred_{m}_{args.split}_seed{args.seed}.npy").exists()}
    preds.update({n: np.load(f) for n, f in parse_pairs(args.pred).items()})
    register(preds)
    if not preds:
        raise SystemExit(f"no pred_*_{args.split}_seed{args.seed}.npy in {run}")
    meta = load_meta(args.cache)
    a = load_split(args.cache, args.split)
    n = min(len(a["fut"]), *(len(p) for p in preds.values()))
    a = {k: v[:n] for k, v in a.items()}
    preds = {m: p[:n] for m, p in preds.items()}
    ag = load_agents(args.cache, args.split, a["scene_id"]) if not args.cv_neighbours else None
    if ag is not None:
        ag = {k: v[:n] for k, v in ag.items()}
    print("neighbours:", "recorded futures" if ag is not None else "constant velocity", flush=True)
    sel = args.scenes if args.scenes else [int(i) for i in
                                           pick_scenes(a, meta, preds, args.pick, args.n, args.seed, args.margin)]
    off = {m: offroad_rate(a, meta, p, args.margin) for m, p in preds.items()}
    for i in sel:
        p = animate_scene(a, meta, preds, int(i), out, args.fps, args.format, args.margin, not args.no_trail,
                          args.sub, args.hold, not args.no_follow, args.zoom, not args.no_ghost, not args.axes,
                          args.show, not args.no_flash, ag)
        print(f"wrote {p}   " + "  ".join(f"{m} off-road {100 * off[m][i]:.0f}%" for m in preds))


if __name__ == "__main__":
    main()
