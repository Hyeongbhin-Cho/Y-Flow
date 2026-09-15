"""Measure RealEstate10K matches before/after the exact epipolar projector."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.realestate10k import RealEstate10KDataset, RealEstate10KEpipolarConstraint


def _sift_matches(a, b, max_matches):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install opencv-python-headless to run this diagnostic") from exc
    detector = cv2.SIFT_create(nfeatures=4000)
    ka, da = detector.detectAndCompute(a, None)
    kb, db = detector.detectAndCompute(b, None)
    if da is None or db is None:
        return np.empty((0, 2)), np.empty((0, 2))
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
    good.sort(key=lambda m: m.distance)
    good = good[:max_matches]
    return np.asarray([ka[m.queryIdx].pt for m in good]), np.asarray([kb[m.trainIdx].pt for m in good])


def verify(root="datasets/realestate10k", split="train", limit=10, max_matches=1000):
    dataset = RealEstate10KDataset(root, split=split, limit=limit)
    constraint = RealEstate10KEpipolarConstraint(tolerance_px=0.0)
    reports = []
    for sample in dataset:
        frames = ((sample["video"].permute(1, 2, 3, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        pairs = []
        for no, (i, j) in enumerate(sample["pair_indices"].tolist()):
            if not bool(sample["fundamental_valid"][no]):
                continue
            p1, p2 = _sift_matches(frames[i], frames[j], max_matches)
            if len(p1) < 8:
                pairs.append({"pair": [i, j], "matches": int(len(p1)), "status": "insufficient_matches"})
                continue
            F = np.broadcast_to(sample["fundamental_matrices"][no].numpy(), (len(p1), 3, 3))
            raw = constraint.h(p1[None], p2[None], F[None])["epipolar_px"][0]
            projected, mask = constraint.project_feasible(p1[None], p2[None], F[None])
            after = constraint.h(p1[None], projected, F[None])["epipolar_px"][0]
            pairs.append({"pair": [i, j], "matches": int(len(p1)),
                          "raw_median_px": float(np.median(raw)),
                          "raw_p90_px": float(np.percentile(raw, 90)),
                          "projected_max_abs_px": float(np.max(np.abs(after[mask[0]]))),
                          "projectable": int(mask[0].sum()), "status": "ok"})
        reports.append({"clip_id": sample["clip_id"], "pairs": pairs})
    return reports


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("datasets/realestate10k"))
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--max-matches", type=int, default=1000)
    args = p.parse_args()
    print(json.dumps(verify(args.root, args.split, args.limit, args.max_matches), indent=2))


if __name__ == "__main__":
    main()
