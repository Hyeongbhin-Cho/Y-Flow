from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from tools.convert_nuscenes import HZ, STATIC_CATEGORIES, T, _yaw
from tools.geometry import to_local

_G: dict = {}


def _cls(cat: str) -> int:
    if cat.startswith(("vehicle.bicycle", "vehicle.motorcycle")):
        return 2
    if cat.startswith("vehicle."):
        return 0
    if cat.startswith("human.pedestrian"):
        return 1
    return 3


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def _build(tok):
    helper, args = _G["helper"], _G["args"]
    instance, sample = tok.split("_")
    ann = helper.get_sample_annotation(instance, sample)
    origin = np.asarray(ann["translation"], np.float64)
    yaw0 = _yaw(ann["rotation"])
    m = args.n_agents
    fut = np.zeros((m, T, 2), np.float32)
    yaw = np.zeros((m, T), np.float32)
    mask = np.zeros((m, T), bool)
    size = np.zeros((m, 2), np.float32)
    cls = np.full(m, 3, np.int8)
    cands = []
    for other in helper.get_annotations_for_sample(sample):
        if other["instance_token"] == instance:
            continue
        d = float(np.linalg.norm(np.asarray(other["translation"][:2]) - origin[:2]))
        if d <= args.radius:
            cands.append((d, other))
    cands.sort(key=lambda x: x[0])
    for j, (_, other) in enumerate(cands[:m]):
        c = _cls(other["category_name"])
        cls[j] = c
        size[j] = (float(other["size"][1]), float(other["size"][0]))
        recs = helper.get_future_for_agent(other["instance_token"], sample, seconds=T / HZ,
                                           in_agent_frame=False, just_xy=False)
        recs = list(recs)[:T]
        for i, r in enumerate(recs):
            fut[j, i] = to_local(np.asarray(r["translation"][:2])[None], origin, yaw0)[0]
            yaw[j, i] = _wrap(_yaw(r["rotation"]) - yaw0)
            mask[j, i] = True
        if not recs and (c == 3 or other["category_name"].startswith(STATIC_CATEGORIES)):
            fut[j] = to_local(np.asarray(other["translation"][:2])[None], origin, yaw0)[0]
            yaw[j] = _wrap(_yaw(other["rotation"]) - yaw0)
            mask[j] = True
    return tok, fut, yaw, mask, size, cls


def extract_split(helper, split, args, out: Path) -> int:
    from nuscenes.eval.prediction.splits import get_prediction_challenge_split

    tokens = get_prediction_challenge_split(split, dataroot=args.root)
    if args.limit:
        tokens = tokens[: args.limit]
    _G.update(helper=helper, args=args)
    workers = int(getattr(args, "workers", 1) or 1)
    t0 = time.time()
    if workers > 1:
        pool = mp.get_context("fork").Pool(workers)
        it = pool.imap(_build, tokens, chunksize=16)
    else:
        pool, it = None, map(_build, tokens)
    rows = []
    try:
        for i, row in enumerate(it, 1):
            rows.append(row)
            if i % 1000 == 0 or i == len(tokens):
                el = time.time() - t0
                print(f"{split}: {i}/{len(tokens)}  {i / el:.1f}/s  ETA {el / i * (len(tokens) - i) / 60:.1f} min", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    names = ("scene_id", "ag_fut", "ag_yaw", "ag_mask", "ag_size", "ag_cls")
    arrays = {k: np.stack([r[i] for r in rows]) for i, k in enumerate(names)}
    ref = out / f"{split}.npz"
    if ref.exists() and not args.limit:
        with np.load(ref) as z:
            ids = z["scene_id"]
        if len(ids) != len(arrays["scene_id"]) or not (ids == arrays["scene_id"]).all():
            raise SystemExit(f"{split}: token order differs from {ref}; regenerate both with the same devkit data")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{split}_agents.npz", **arrays)
    print(f"{split}: wrote {out / f'{split}_agents.npz'} ({len(rows)} scenes)", flush=True)
    return len(rows)


def main() -> None:
    from nuscenes import NuScenes
    from nuscenes.prediction import PredictHelper

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--out", default="datasets/nuscenes_trainval")
    ap.add_argument("--splits", nargs="+", default=["val", "train_val"])
    ap.add_argument("--n_agents", type=int, default=32)
    ap.add_argument("--radius", type=float, default=80.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()
    print(f"loading nuScenes {args.version} ... (workers={args.workers})", flush=True)
    nusc = NuScenes(args.version, dataroot=args.root, verbose=False)
    helper = PredictHelper(nusc)
    for s in args.splits:
        extract_split(helper, s, args, Path(args.out))


if __name__ == "__main__":
    main()
