from __future__ import annotations

from pathlib import Path

import numpy as np


def headings(p: np.ndarray, min_move: float = 0.2) -> np.ndarray:
    q = np.concatenate([np.zeros_like(p[..., :1, :]), p], axis=-2)
    d = np.diff(q, axis=-2)
    yaw = np.zeros(p.shape[:-1], np.float32)
    cur = np.zeros(p.shape[:-2], np.float32)
    for i in range(p.shape[-2]):
        mv = np.linalg.norm(d[..., i, :], axis=-1) > min_move
        cur = np.where(mv, np.arctan2(d[..., i, 1], d[..., i, 0]), cur)
        yaw[..., i] = cur
    return yaw


def boxes(c: np.ndarray, yaw: np.ndarray, length, width) -> np.ndarray:
    f = np.stack([np.cos(yaw), np.sin(yaw)], -1)
    l = np.stack([-f[..., 1], f[..., 0]], -1)
    hl, hw = (np.asarray(length)[..., None] / 2), (np.asarray(width)[..., None] / 2)
    return np.stack([c + f * hl + l * hw, c + f * hl - l * hw, c - f * hl - l * hw, c - f * hl + l * hw], -2)


def sat_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.ones(np.broadcast_shapes(a.shape[:-2], b.shape[:-2]), bool)
    for poly in (a, b):
        for e in range(2):
            edge = poly[..., e + 1, :] - poly[..., e, :]
            ax = np.stack([-edge[..., 1], edge[..., 0]], -1)[..., None, :]
            pa = (a * ax).sum(-1)
            pb = (b * ax).sum(-1)
            out &= (pa.max(-1) >= pb.min(-1)) & (pb.max(-1) >= pa.min(-1))
    return out


def collisions(pred, focal_size, ag, chunk=64):
    n, k, t = pred.shape[:3]
    yaw = headings(pred)
    hit = np.zeros((n, k), bool)
    hit_by = {c: np.zeros((n, k), bool) for c in (0, 1, 2, 3)}
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        fb = boxes(pred[s:e], yaw[s:e], focal_size[s:e, 0, None, None], focal_size[s:e, 1, None, None])
        ab = boxes(ag["ag_fut"][s:e], ag["ag_yaw"][s:e], ag["ag_size"][s:e, :, 0, None], ag["ag_size"][s:e, :, 1, None])
        ov = sat_overlap(fb[:, :, None], ab[:, None])
        ov &= ag["ag_mask"][s:e, None]
        ov &= (ag["ag_size"][s:e, :, 0] > 0)[:, None, :, None]
        hit[s:e] = ov.any(axis=(2, 3))
        for c in hit_by:
            hit_by[c][s:e] = (ov & (ag["ag_cls"][s:e] == c)[:, None, :, None]).any(axis=(2, 3))
    return hit, hit_by


def load_agents(cache_dir, split: str, scene_id: np.ndarray) -> dict | None:
    f = Path(cache_dir) / f"{split}_agents.npz"
    if not f.exists():
        return None
    with np.load(f) as z:
        ag = {k: z[k][: len(scene_id)] for k in z.files}
    if len(ag["scene_id"]) != len(scene_id) or not (ag["scene_id"] == scene_id).all():
        raise ValueError(f"{f} is not aligned with the {split} cache")
    return ag


def gt_conditional(fail: np.ndarray, gt_fail: np.ndarray) -> dict:
    ok = ~gt_fail
    out = {"traj": float(fail.mean()), "scene": float(fail.any(1).mean())}
    out["traj_gtok"] = float(fail[ok].mean()) if ok.any() else float("nan")
    out["scene_gtok"] = float(fail[ok].any(1).mean()) if ok.any() else float("nan")
    out["gt"] = float(gt_fail.mean())
    return out
