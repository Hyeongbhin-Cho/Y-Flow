from __future__ import annotations

import argparse
import json

import numpy as np

from vfm.data import save_split


def _arc(speed, turn, t, hz):
    dt = 1.0 / hz
    yaw_rate = turn * (np.pi / 2) / 3.0
    x, y, yaw = 0.0, 0.0, 0.0
    out = []
    for i in range(t):
        s = i * dt
        r = yaw_rate if 1.0 <= s <= 4.0 else 0.0
        yaw += r * dt
        x += speed * np.cos(yaw) * dt
        y += speed * np.sin(yaw) * dt
        out.append((x, y))
    return np.asarray(out, dtype=np.float32)


def make(n, rng, h=20, t=30, hz=5.0, a=4, l=6, p=10):
    arr = {
        "hist": np.zeros((n, h, 2), np.float32), "hist_mask": np.ones((n, h), bool),
        "fut": np.zeros((n, t, 2), np.float32), "fut_mask": np.ones((n, t), bool),
        "nbr": np.zeros((n, a, h, 2), np.float32), "nbr_mask": np.zeros((n, a, h), bool),
        "lane": np.zeros((n, l, p, 2), np.float32), "lane_mask": np.zeros((n, l, p), bool),
    }
    ids = []
    for i in range(n):
        speed = rng.uniform(5.0, 15.0)
        arr["hist"][i] = (np.arange(-h + 1, 1, dtype=np.float32)[:, None] * np.array([speed / hz, 0.0], np.float32))
        options = [o for o in (-1, 0, 1) if rng.random() < 0.7] or [0]
        turn = options[rng.integers(len(options))]
        arr["fut"][i] = _arc(speed, turn, t, hz) + rng.normal(0, 0.05, (t, 2)).astype(np.float32)
        for j, o in enumerate(options):
            path = _arc(speed, o, t, hz)
            idx = np.linspace(0, t - 1, p).astype(int)
            arr["lane"][i, j] = path[idx]
            arr["lane_mask"][i, j] = True
        for j in range(rng.integers(0, a + 1)):
            off = rng.uniform(-30, 30, 2).astype(np.float32)
            v = rng.uniform(-10, 10, 2).astype(np.float32) / hz
            arr["nbr"][i, j] = off + np.arange(-h + 1, 1, dtype=np.float32)[:, None] * v
            arr["nbr_mask"][i, j] = True
        ids.append(f"syn_{i:07d}")
    arr["scene_id"] = np.asarray(ids)
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/synthetic")
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    save_split(args.out, "train", make(args.n_train, rng))
    save_split(args.out, "val", make(args.n_val, rng))
    meta = {"dataset": "synthetic_junction", "sample_hz": 5.0, "history_steps": 20, "future_steps": 30,
            "eval_k": 6, "miss_threshold_m": 2.0, "frame": "focal-centric, +x heading"}
    (__import__("pathlib").Path(args.out) / "meta.json").write_text(json.dumps(meta, indent=2))
    print(meta)


if __name__ == "__main__":
    main()
