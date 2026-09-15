"""Apply sparse epipolar projections as RGB warps and re-measure SIFT error.

This is a controlled image-space experiment, not a latent-space projection.
Each later generated frame is independently warped toward the first frame's
camera F.  The JSON report separates control-point algebra from fresh matches
measured after the RGB warp.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.realestate10k import RealEstate10KDataset, RealEstate10KEpipolarConstraint


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install opencv-python-headless to run RGB warp evaluation") from exc
    return cv2


def sift_matches(source, target, max_matches, nfeatures=8000, ratio=0.80):
    cv2 = _cv2()
    detector = cv2.SIFT_create(nfeatures=nfeatures)
    source_keys, source_desc = detector.detectAndCompute(source, None)
    target_keys, target_desc = detector.detectAndCompute(target, None)
    if source_desc is None or target_desc is None:
        return np.empty((0, 2)), np.empty((0, 2))
    candidates = cv2.BFMatcher(cv2.NORM_L2).knnMatch(source_desc, target_desc, k=2)
    good = [first for first, second in candidates if first.distance < ratio * second.distance]
    good.sort(key=lambda match: match.distance)
    good = good[:max_matches]
    return (np.asarray([source_keys[m.queryIdx].pt for m in good], dtype=np.float64),
            np.asarray([target_keys[m.trainIdx].pt for m in good], dtype=np.float64))


def decode_video(path: Path, frames: int, height: int, width: int):
    import av

    result = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            result.append(frame.to_ndarray(format="rgb24"))
            if len(result) == frames:
                break
    if len(result) < frames:
        raise ValueError(f"{path} has {len(result)} frames; expected at least {frames}")
    if result[0].shape[:2] != (height, width):
        raise ValueError(f"{path} has {result[0].shape[:2]}, expected {(height, width)}")
    return np.stack(result)


def export_video(frames, path: Path, fps=16):
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
        for array in frames:
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def metrics(constraint, source, target, F, max_matches, nfeatures=8000, ratio=0.80):
    p1, p2 = sift_matches(source, target, max_matches, nfeatures, ratio)
    if len(p1) < 8:
        return {"matches": int(len(p1)), "status": "insufficient_matches"}
    matrices = np.broadcast_to(F, (len(p1), 3, 3))[None]
    residual = constraint.h(p1[None], p2[None], matrices)["epipolar_px"][0]
    return {"matches": int(len(p1)), "median_px": float(np.median(residual)),
            "p90_px": float(np.percentile(residual, 90)), "status": "ok"}


def warp_target(
    constraint, source, target, F, max_matches, max_displacement, support_radius,
    min_controls=12, nfeatures=8000, ratio=0.80,
):
    """Return an inverse-warped target and its accepted exact control points."""
    p1, p2 = sift_matches(source, target, max_matches, nfeatures, ratio)
    if len(p1) < 8:
        return target.copy(), {"matches": int(len(p1)), "accepted_controls": 0, "status": "insufficient_matches"}
    matrices = np.broadcast_to(F, (len(p1), 3, 3))[None]
    projected, mask = constraint.project_feasible(p1[None], p2[None], matrices)
    projected, mask = projected[0], mask[0]
    displacement = p2 - projected  # inverse sampling: output q* reads from original q
    distance = np.linalg.norm(displacement, axis=1)
    accepted = mask & np.isfinite(distance) & (distance <= max_displacement)
    if accepted.sum() < min_controls:
        return target.copy(), {"matches": int(len(p1)), "accepted_controls": int(accepted.sum()),
                               "status": "insufficient_accepted_controls"}
    controls = projected[accepted]
    values = displacement[accepted]
    height, width = target.shape[:2]
    cv2 = _cv2()
    hull = cv2.convexHull(controls.astype(np.float32))
    coverage = float(cv2.contourArea(hull) / max(height * width, 1))
    # Boundary anchors preserve the frame edge. Interpolation only affects
    # pixels close to actual projected correspondence controls.
    anchors = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], dtype=np.float64)
    points = np.concatenate([controls, anchors])
    vectors = np.concatenate([values, np.zeros((len(anchors), 2))])
    yy, xx = np.mgrid[0:height, 0:width]
    field_x = griddata(points, vectors[:, 0], (xx, yy), method="linear", fill_value=0.0)
    field_y = griddata(points, vectors[:, 1], (xx, yy), method="linear", fill_value=0.0)
    nearest, _ = cKDTree(controls).query(np.column_stack([xx.ravel(), yy.ravel()]), k=1)
    supported = nearest.reshape(height, width) <= support_radius
    field_x = np.where(supported, field_x, 0.0).astype(np.float32)
    field_y = np.where(supported, field_y, 0.0).astype(np.float32)
    warped = cv2.remap(target, xx.astype(np.float32) + field_x, yy.astype(np.float32) + field_y,
                        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    control_residual = constraint.h(p1[None, accepted], projected[None, accepted],
                                    np.broadcast_to(F, (accepted.sum(), 3, 3))[None])["epipolar_px"][0]
    return warped, {"matches": int(len(p1)), "accepted_controls": int(accepted.sum()),
                    "control_coverage": coverage,
                    "control_max_abs_px": float(np.max(np.abs(control_residual))), "status": "ok"}


def run(root, generated_dir, output, split, clip_id, max_matches, max_displacement, support_radius,
        min_controls=12, control_features=8000, control_ratio=0.80):
    dataset = RealEstate10KDataset(root, split=split)
    sample = next((item for item in dataset if item["clip_id"] == clip_id), None)
    if sample is None:
        raise ValueError(f"No clip {clip_id!r} in {root}/manifests/{split}.json")
    original = sample["video"]
    frames = decode_video(Path(generated_dir) / f"{clip_id}.mp4", original.shape[1], original.shape[2], original.shape[3])
    constraint = RealEstate10KEpipolarConstraint(tolerance_px=0.0)
    F_by_pair = {tuple(pair.tolist()): sample["fundamental_matrices"][n].numpy()
                 for n, pair in enumerate(sample["pair_indices"])
                 if bool(sample["fundamental_valid"][n])}
    warped = frames.copy()
    report_pairs = []
    for target_index in range(1, len(frames)):
        pair = (0, target_index)
        F = F_by_pair.get(pair)
        if F is None:
            report_pairs.append({"pair": list(pair), "status": "invalid_fundamental_matrix"})
            continue
        before = metrics(constraint, frames[0], frames[target_index], F, max_matches, control_features, control_ratio)
        warped[target_index], controls = warp_target(constraint, frames[0], frames[target_index], F, max_matches,
                                                      max_displacement, support_radius, min_controls,
                                                      control_features, control_ratio)
        after = metrics(constraint, warped[0], warped[target_index], F, max_matches, control_features, control_ratio)
        report_pairs.append({"pair": list(pair), "before": before, "controls": controls, "after": after})
    output.mkdir(parents=True, exist_ok=True)
    video_path = output / f"{clip_id}_warped.mp4"
    export_video(warped, video_path)
    report = {"clip_id": clip_id, "input_video": str(Path(generated_dir) / f"{clip_id}.mp4"),
              "warped_video": str(video_path), "max_displacement_px": max_displacement,
              "support_radius_px": support_radius, "min_controls": min_controls,
              "control_features": control_features, "control_ratio": control_ratio,
              "warning": "Each target frame is independently warped toward frame 0; this is not a Wan latent constraint.",
              "pairs": report_pairs}
    (output / f"{clip_id}_warp_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/realestate10k"))
    parser.add_argument("--generated-dir", type=Path, default=Path("outputs/realestate10k_wan"))
    parser.add_argument("--output", type=Path, default=Path("outputs/realestate10k_warp"))
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--max-matches", type=int, default=2000)
    parser.add_argument("--max-displacement", type=float, default=64.0)
    parser.add_argument("--support-radius", type=float, default=48.0)
    parser.add_argument("--min-controls", type=int, default=12)
    parser.add_argument("--control-features", type=int, default=8000)
    parser.add_argument("--control-ratio", type=float, default=0.80)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
