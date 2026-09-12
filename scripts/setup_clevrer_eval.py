# -*- coding: utf-8 -*-
# scripts/setup_clevrer_eval.py
"""Download CLEVRER physics-eval artifacts and score object/interaction parsers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.clevrer_eval import (
    detect_mask_rcnn,
    evaluate_propnet_pred,
    evaluate_visual_mask,
    frame_to_uint8_hwc,
    load_mask_rcnn,
    resolve_eval_root,
    summarize_reports,
)
from model.download import (
    CLEVRER_ARTIFACTS,
    DEFAULT_CLEVRER_EVAL_DIR,
    download_clevrer_artifact,
    is_clevrer_artifact_ready,
    write_clevrer_artifact_manifest,
)


def _load_annotation(scene_id: int) -> dict:
    path = ROOT / "datasets" / "clevrer" / "annotations" / "validation" / f"annotation_{scene_id:05d}.json"
    return json.loads(path.read_text())


def _eval_scene_ids(limit: int) -> list[int]:
    manifest = ROOT / "datasets" / "clevrer" / "manifests" / "eval_100.json"
    if manifest.is_file():
        ids = json.loads(manifest.read_text())["scene_indices"]
        return ids[:limit]
    return list(range(10000, 10000 + limit))


def download_artifacts(names: list[str], root: Path, *, force: bool) -> dict[str, str]:
    status = {}
    for name in names:
        try:
            path = download_clevrer_artifact(name, root, force=force)
            status[name] = f"ready:{path}"
        except Exception as exc:  # noqa: BLE001 — record each artifact independently
            status[name] = f"failed:{exc}"
            print(f"[Y-Flow] {name} failed: {exc}", flush=True)
    write_clevrer_artifact_manifest(root)
    return status


def evaluate_parsers(root: Path, *, n_scenes: int, n_detect: int) -> dict:
    scene_ids = _eval_scene_ids(n_scenes)
    report: dict = {"n_scenes": len(scene_ids), "scene_indices": scene_ids}

    if is_clevrer_artifact_ready(root, "visual_masks"):
        mask_reports = []
        missing = []
        for scene_id in scene_ids:
            try:
                mask_reports.append(evaluate_visual_mask(scene_id, _load_annotation(scene_id), root))
            except FileNotFoundError:
                missing.append(scene_id)
        report["visual_masks"] = {
            "n_scored": len(mask_reports),
            "n_missing": len(missing),
            "attributes": summarize_reports(mask_reports, "attributes"),
            "visible_count_mae": summarize_reports(mask_reports, "visible_count_mae"),
            "collisions": summarize_reports(mask_reports, "collisions"),
        }
    else:
        report["visual_masks"] = {"ready": False}

    if is_clevrer_artifact_ready(root, "propnet_preds"):
        pred_reports = []
        missing = []
        for scene_id in scene_ids:
            try:
                pred_reports.append(evaluate_propnet_pred(scene_id, _load_annotation(scene_id), root))
            except FileNotFoundError:
                missing.append(scene_id)
        report["propnet_preds"] = {
            "n_scored": len(pred_reports),
            "n_missing": len(missing),
            "attributes": summarize_reports(pred_reports, "attributes"),
            "collisions": summarize_reports(pred_reports, "collisions"),
        }
    else:
        report["propnet_preds"] = {"ready": False}

    if is_clevrer_artifact_ready(root, "mask_rcnn"):
        from data import CLEVRERDataset

        model, device = load_mask_rcnn(root)
        dataset = CLEVRERDataset(ROOT / "datasets" / "clevrer", "validation", n_frames=1, fps=None)
        id_to_index = {rec[0]: i for i, rec in enumerate(dataset.records)}
        detections = []
        inspect = root / "inspect"
        inspect.mkdir(parents=True, exist_ok=True)
        for scene_id in scene_ids[:n_detect]:
            sample = dataset[id_to_index[scene_id]]
            dets = detect_mask_rcnn(model, sample["initial_frame"], device)
            annotation = _load_annotation(scene_id)
            n_visible = sum(
                1 for obj in annotation["motion_trajectory"][0]["objects"] if obj.get("inside_camera_view")
            )
            detections.append({
                "scene_index": scene_id,
                "n_gt_visible": n_visible,
                "n_det": len(dets),
                "names": [d["name"] for d in dets],
                "scores": [d["score"] for d in dets],
            })
            try:
                from PIL import Image, ImageDraw

                image = Image.fromarray(frame_to_uint8_hwc(sample["initial_frame"]))
                draw = ImageDraw.Draw(image)
                for det in dets:
                    x1, y1, x2, y2 = det["box"]
                    draw.rectangle((x1, y1, x2, y2), outline=(255, 220, 0), width=2)
                    draw.text((x1, max(0, y1 - 12)), f"{det['name']} {det['score']:.2f}", fill=(255, 220, 0))
                image.save(inspect / f"maskrcnn_{scene_id:05d}.png")
            except Exception:  # noqa: BLE001
                pass
        report["mask_rcnn"] = {
            "clevrer_finetuned": False,
            "n_images": len(detections),
            "mean_detections": sum(d["n_det"] for d in detections) / max(len(detections), 1),
            "mean_gt_visible": sum(d["n_gt_visible"] for d in detections) / max(len(detections), 1),
            "coco_class_histogram": _histogram([n for d in detections for n in d["names"]]),
            "samples": detections,
        }
    else:
        report["mask_rcnn"] = {"ready": False}

    report["propnet_weights"] = {
        "ready": is_clevrer_artifact_ready(root, "propnet"),
        "note": "DCL .pth files consume tube proposals, not raw RGB. Interaction scoring uses propnet_preds.",
    }
    return report


def _histogram(names: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_CLEVRER_EVAL_DIR))
    parser.add_argument(
        "--artifacts",
        nargs="+",
        choices=list(CLEVRER_ARTIFACTS),
        default=list(CLEVRER_ARTIFACTS),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--eval", action="store_true", help="Score parsers on original CLEVRER eval clips.")
    parser.add_argument("--n-scenes", type=int, default=100)
    parser.add_argument("--n-detect", type=int, default=10)
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else ROOT / args.root

    print("=" * 65)
    print(" [Y-Flow] CLEVRER physics-eval artifacts")
    print(f"  Target: {root}")
    print("=" * 65)
    if args.dry_run:
        for name, spec in CLEVRER_ARTIFACTS.items():
            print(f"{name}: {spec['kind']} | {spec['weight_status']}")
            print(f"  {spec['source']}")
            print(f"  applies_to={spec['applies_to']}")
        return

    status = {}
    if not args.skip_download:
        status = download_artifacts(args.artifacts, root, force=args.force)
        print("[Y-Flow] download status:", json.dumps(status, indent=2))

    if args.eval:
        report = evaluate_parsers(root, n_scenes=args.n_scenes, n_detect=args.n_detect)
        report["download_status"] = status
        out = root / "eval_report.json"
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "scene_indices"}, indent=2))
        print(f"[Y-Flow] wrote {out}")


if __name__ == "__main__":
    main()
