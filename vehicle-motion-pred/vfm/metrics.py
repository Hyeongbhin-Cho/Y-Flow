from __future__ import annotations

import numpy as np


def displacement(pred: np.ndarray, fut: np.ndarray, fut_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.linalg.norm(pred - fut[:, None], axis=-1)
    m = fut_mask[:, None].astype(np.float64)
    ade = (d * m).sum(-1) / np.maximum(m.sum(-1), 1.0)
    last = fut_mask.shape[1] - 1 - np.argmax(fut_mask[:, ::-1], axis=1)
    fde = d[np.arange(d.shape[0]), :, last]
    return ade, fde


def kinematics(pred: np.ndarray, sample_hz: float) -> tuple[np.ndarray, np.ndarray]:
    origin = np.zeros((*pred.shape[:-2], 1, 2), dtype=pred.dtype)
    p = np.concatenate([origin, pred], axis=-2)
    v = np.diff(p, axis=-2) * sample_hz
    a = np.diff(v, axis=-2) * sample_hz
    return np.linalg.norm(v, axis=-1), np.linalg.norm(a, axis=-1)


def summarize(pred: np.ndarray, fut: np.ndarray, fut_mask: np.ndarray, meta: dict, kin: dict | None = None) -> dict:
    ade, fde = displacement(pred, fut, fut_mask)
    k = pred.shape[1]
    thr = float(meta.get("miss_threshold_m", 2.0))
    out = {
        "n_scenes": int(pred.shape[0]),
        "K": int(k),
        f"minADE_{k}": float(ade.min(1).mean()),
        f"minFDE_{k}": float(fde.min(1).mean()),
        f"MR_{k}@{thr:g}m": float((fde.min(1) > thr).mean()),
        "ADE_1": float(ade[:, 0].mean()),
        "FDE_1": float(fde[:, 0].mean()),
        "meanADE": float(ade.mean()),
        "meanFDE": float(fde.mean()),
    }
    for kk in sorted({int(x) for x in meta.get("extra_k", [])}):
        if 1 < kk < k:
            out[f"minADE_{kk}"] = float(ade[:, :kk].min(1).mean())
            out[f"minFDE_{kk}"] = float(fde[:, :kk].min(1).mean())
    spread = pred[..., -1, :] - pred[..., -1, :].mean(axis=1, keepdims=True)
    out["endpoint_spread_m"] = float(np.linalg.norm(spread, axis=-1).mean())
    speed, acc = kinematics(pred, float(meta["sample_hz"]))
    out["speed_p99"] = float(np.percentile(speed.max(-1), 99))
    out["accel_p99"] = float(np.percentile(acc.max(-1), 99))
    if kin:
        if "v_max" in kin:
            out["speed_viol_rate"] = float((speed.max(-1) > float(kin["v_max"])).mean())
        if "a_max" in kin:
            out["accel_viol_rate"] = float((acc.max(-1) > float(kin["a_max"])).mean())
    return out
