from __future__ import annotations

import argparse
import json

import numpy as np

from vfm.data import load_meta, load_split
from vfm.metrics import kinematics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    meta = load_meta(args.cache)
    a = load_split(args.cache, args.split)
    full = a["fut_mask"].all(axis=1)
    speed, acc = kinematics(a["fut"][full][:, None], float(meta["sample_hz"]))
    vmax, amax = speed[:, 0].max(-1), acc[:, 0].max(-1)
    q = [50, 90, 99, 99.5, 99.9]
    report = {
        "dataset": meta.get("dataset"), "split": args.split, "n_full_future": int(full.sum()),
        "n_total": int(len(full)),
        "max_speed_pct": {str(p): float(np.percentile(vmax, p)) for p in q},
        "max_accel_pct": {str(p): float(np.percentile(amax, p)) for p in q},
        "n_neighbors_mean": float(a["nbr_mask"].any(-1).sum(-1).mean()),
        "n_lanes_mean": float(a["lane_mask"].any(-1).sum(-1).mean()),
        "final_disp_mean_m": float(np.linalg.norm(a["fut"][full][:, -1], axis=-1).mean()),
        "final_x_mean_m": float(a["fut"][full][:, -1, 0].mean()),
        "final_abs_y_mean_m": float(np.abs(a["fut"][full][:, -1, 1]).mean()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
