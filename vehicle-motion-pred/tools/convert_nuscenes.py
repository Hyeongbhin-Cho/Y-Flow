from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from tools.geometry import drivable_sdf, pick_lanes, split_polyline, to_local
from vfm.data import save_split

H, T, HZ = 5, 12, 2.0
SDF_GRID = {"x_min": -32.0, "y_min": -64.0, "res": 1.0, "size": 128, "scale": 0.25}
STATIC_CATEGORIES = ("movable_object.barrier", "movable_object.trafficcone")


def _yaw(rotation) -> float:
    from nuscenes.eval.common.utils import quaternion_yaw
    from pyquaternion import Quaternion

    return float(quaternion_yaw(Quaternion(rotation)))


def _track(helper, instance, sample, seconds, direction):
    ann = helper.get_sample_annotation(instance, sample)
    fn = helper.get_past_for_agent if direction == "past" else helper.get_future_for_agent
    xy = fn(instance, sample, seconds=seconds, in_agent_frame=False)
    xy = np.zeros((0, 2)) if len(xy) == 0 else np.asarray(xy)[:, :2]
    return ann, xy


def _is_static(nusc, ann) -> bool:
    if ann["category_name"].startswith(STATIC_CATEGORIES):
        return True
    if not ann["category_name"].startswith("vehicle."):
        return False
    names = {nusc.get("attribute", t)["name"] for t in ann["attribute_tokens"]}
    return "vehicle.parked" in names


_POLY_CACHE: dict = {}
_G: dict = {}


def _global_polygon(nmap, token):
    key = (id(nmap), token)
    if key not in _POLY_CACHE:
        poly = nmap.extract_polygon(token)
        if poly.is_empty:
            _POLY_CACHE[key] = None
        else:
            _POLY_CACHE[key] = (np.asarray(poly.exterior.coords)[:, :2],
                                [np.asarray(r.coords)[:, :2] for r in poly.interiors])
    return _POLY_CACHE[key]


def _drivable_polygons(nmap, origin, yaw, radius, layers):
    recs = nmap.get_records_in_radius(origin[0], origin[1], radius, list(layers))
    polys = []
    for layer in layers:
        for tok in recs.get(layer, []):
            rec = nmap.get(layer, tok)
            ptoks = rec["polygon_tokens"] if "polygon_tokens" in rec else [rec["polygon_token"]]
            for pt in ptoks:
                g = _global_polygon(nmap, pt)
                if g is None:
                    continue
                polys.append((to_local(g[0], origin, yaw), [to_local(h, origin, yaw) for h in g[1]]))
    return polys


def _build(tok):
    nusc, helper, maps, args = _G["nusc"], _G["helper"], _G["maps"], _G["args"]
    instance, sample = tok.split("_")
    ann, past = _track(helper, instance, sample, (H - 1) / HZ, "past")
    _, future = _track(helper, instance, sample, T / HZ, "future")
    origin = np.asarray(ann["translation"], np.float64)
    yaw = _yaw(ann["rotation"])

    hist = np.zeros((H, 2), np.float32)
    hmask = np.zeros(H, bool)
    seq = np.concatenate([past[::-1], origin[None, :2]], axis=0)[-H:]
    hist[H - len(seq):] = to_local(seq, origin, yaw)
    hmask[H - len(seq):] = True
    fut = np.zeros((T, 2), np.float32)
    fmask = np.zeros(T, bool)
    if len(future):
        fut[: len(future)] = to_local(future[:T], origin, yaw)
        fmask[: len(future[:T])] = True

    nbr = np.zeros((args.n_nbr, H, 2), np.float32)
    nmask = np.zeros((args.n_nbr, H), bool)
    nsize = np.zeros((args.n_nbr, 2), np.float32)
    nyaw = np.zeros(args.n_nbr, np.float32)
    cands = []
    statics = []
    for other in helper.get_annotations_for_sample(sample):
        if other["instance_token"] == instance:
            continue
        d = float(np.linalg.norm(np.asarray(other["translation"][:2]) - origin[:2]))
        if d <= args.radius:
            cands.append((d, other))
            if _is_static(nusc, other):
                statics.append((d, other))
    cands.sort(key=lambda x: x[0])
    for j, (_, other) in enumerate(cands[: args.n_nbr]):
        _, op = _track(helper, other["instance_token"], sample, (H - 1) / HZ, "past")
        oseq = np.concatenate([op[::-1], np.asarray(other["translation"][:2])[None]], axis=0)[-H:]
        nbr[j, H - len(oseq):] = to_local(oseq, origin, yaw)
        nmask[j, H - len(oseq):] = True
        nsize[j] = (float(other["size"][1]), float(other["size"][0]))
        nyaw[j] = np.arctan2(np.sin(_yaw(other["rotation"]) - yaw), np.cos(_yaw(other["rotation"]) - yaw))

    obs = np.zeros((args.n_obs, 5), np.float32)
    omask = np.zeros(args.n_obs, bool)
    statics.sort(key=lambda x: x[0])
    for j, (_, other) in enumerate(statics[: args.n_obs]):
        c = to_local(np.asarray(other["translation"][:2])[None], origin, yaw)[0]
        w, l = float(other["size"][0]), float(other["size"][1])
        oyaw = np.arctan2(np.sin(_yaw(other["rotation"]) - yaw), np.cos(_yaw(other["rotation"]) - yaw))
        obs[j] = (c[0], c[1], oyaw, l, w)
        omask[j] = True
    focal_size = np.array([ann["size"][1], ann["size"][0]], np.float32)

    map_name = helper.get_map_name_from_sample_token(sample)
    nmap = maps[map_name]
    recs = nmap.get_records_in_radius(origin[0], origin[1], args.radius, ["lane", "lane_connector"])
    ids = recs["lane"] + recs["lane_connector"]
    polys = []
    for poses in nmap.discretize_lanes(ids, 1.0).values():
        if len(poses) < 2:
            continue
        for part in split_polyline(np.asarray(poses)[:, :2], args.lane_len):
            polys.append(to_local(part, origin, yaw))
    lane, lmask = pick_lanes(polys, args.n_lanes, args.n_points)
    da = _drivable_polygons(nmap, origin, yaw, args.sdf_radius, args.sdf_layers)
    sdf = drivable_sdf(da, SDF_GRID)
    sdf_q = np.clip(np.round(sdf / SDF_GRID["scale"]), -127, 127).astype(np.int8)

    return tok, hist, hmask, fut, fmask, nbr, nmask, lane, lmask, focal_size, obs, omask, sdf_q, nsize, nyaw


def convert_split(nusc, helper, maps, split, args, out: Path) -> int:
    from nuscenes.eval.prediction.splits import get_prediction_challenge_split

    tokens = get_prediction_challenge_split(split, dataroot=args.root)
    if args.limit:
        tokens = tokens[: args.limit]
    names = ("scene_id", "hist", "hist_mask", "fut", "fut_mask", "nbr", "nbr_mask", "lane", "lane_mask",
             "focal_size", "obs", "obs_mask", "sdf", "nbr_size", "nbr_yaw")
    rows = {k: [] for k in names}
    _G.update(nusc=nusc, helper=helper, maps=maps, args=args)
    workers = int(getattr(args, "workers", 1) or 1)
    every = int(getattr(args, "log_every", 500) or 500)
    t0 = time.time()
    if workers > 1:
        pool = mp.get_context("fork").Pool(workers)
        it = pool.imap(_build, tokens, chunksize=16)
    else:
        pool, it = None, map(_build, tokens)
    try:
        for i, row in enumerate(it, 1):
            for k, v in zip(rows, row):
                rows[k].append(v)
            if i % every == 0 or i == len(tokens):
                el = time.time() - t0
                eta = el / i * (len(tokens) - i)
                print(f"{split}: {i}/{len(tokens)}  {i / el:.1f} agents/s  elapsed {el / 60:.1f} min  "
                      f"ETA {eta / 60:.1f} min", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    arrays = {k: np.stack(v) for k, v in rows.items()}
    save_split(out, split, arrays)
    print(f"{split}: {len(tokens)} agents, full-future {int(arrays['fut_mask'].all(1).sum())}", flush=True)
    return len(tokens)


def main() -> None:
    from nuscenes import NuScenes
    from nuscenes.map_expansion.map_api import NuScenesMap
    from nuscenes.prediction import PredictHelper

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--out", default="datasets/nuscenes")
    ap.add_argument("--splits", nargs="+", default=["train", "train_val", "val"])
    ap.add_argument("--n_nbr", type=int, default=16)
    ap.add_argument("--n_lanes", type=int, default=48)
    ap.add_argument("--n_points", type=int, default=20)
    ap.add_argument("--lane_len", type=float, default=20.0)
    ap.add_argument("--radius", type=float, default=60.0)
    ap.add_argument("--n_obs", type=int, default=32)
    ap.add_argument("--sdf_radius", type=float, default=140.0)
    ap.add_argument("--sdf_layers", nargs="+", default=["drivable_area", "carpark_area"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--log_every", type=int, default=500)
    args = ap.parse_args()
    print(f"loading nuScenes {args.version} ... (workers={args.workers})", flush=True)
    nusc = NuScenes(args.version, dataroot=args.root, verbose=False)
    helper = PredictHelper(nusc)
    names = ("singapore-onenorth", "singapore-hollandvillage", "singapore-queenstown", "boston-seaport")
    maps = {n: NuScenesMap(dataroot=args.root, map_name=n) for n in names}
    print("loaded; converting", flush=True)
    out = Path(args.out)
    counts = {s: convert_split(nusc, helper, maps, s, args, out) for s in args.splits}
    meta = {"dataset": "nuscenes_prediction", "sample_hz": HZ, "history_steps": H, "future_steps": T,
            "eval_k": 10, "extra_k": [5], "miss_threshold_m": 2.0, "counts": counts,
            "sdf": {**SDF_GRID, "layers": args.sdf_layers},
            "frame": "focal-centric at current keyframe, +x = focal heading"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
