# -*- coding: utf-8 -*-
# model/download.py
"""Hugging Face checkpoint downloader for Wan2.1 video flow matching models."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Sequence

from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_LOCAL_DIR = "checkpoints/Wan2.1-T2V-1.3B"

# Pre-defined subcomponent filter patterns
SUBCOMPONENT_PATTERNS: dict[str, list[str]] = {
    # All files (includes ~10GB UMT5 text encoder)
    "all": ["*"],
    # Core flow-matching files: DiT transformer + 3D VAE + scheduler (~3.0 GB total)
    "core": [
        "model_index.json",
        "scheduler/*",
        "transformer/*",
        "vae/*",
    ],
    # Only the DiT velocity backbone (~2.6 GB)
    "transformer": [
        "model_index.json",
        "transformer/*",
    ],
    # Only the 3D Causal VAE (~400 MB)
    "vae": [
        "vae/*",
    ],
    # Text encoder only (~10 GB)
    "text_encoder": [
        "text_encoder/*",
        "tokenizer/*",
    ],
}


def is_already_downloaded(local_dir: Path | str, subcomponent: str = "core") -> bool:
    """Check if the requested model files already exist locally."""
    path = Path(local_dir)
    if not path.exists():
        return False

    if subcomponent in ("all", "core", "transformer"):
        trans_cfg = path / "transformer" / "config.json"
        trans_weights = list((path / "transformer").glob("*.safetensors"))
        if not (trans_cfg.exists() and len(trans_weights) > 0):
            return False

    if subcomponent in ("all", "core", "vae"):
        vae_cfg = path / "vae" / "config.json"
        vae_weights = list((path / "vae").glob("*.safetensors"))
        if not (vae_cfg.exists() and len(vae_weights) > 0):
            return False

    if subcomponent in ("all", "text_encoder"):
        te_cfg = path / "text_encoder" / "config.json"
        te_weights = list((path / "text_encoder").glob("*.safetensors"))
        if not (te_cfg.exists() and len(te_weights) > 0):
            return False

    return True


def download_wan_model(
    repo_id: str = DEFAULT_REPO_ID,
    local_dir: str | Path = DEFAULT_LOCAL_DIR,
    subcomponent: str = "core",
    force_download: bool = False,
    max_workers: int = 4,
) -> Path:
    """Download Wan2.1 weights from Hugging Face into a persistent local directory.
    
    If weights are already present and force_download is False, the download is skipped.
    
    Args:
        repo_id: HuggingFace repository ID (e.g. 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers').
        local_dir: Target local directory to save the checkpoint files.
        subcomponent: One of ['all', 'core', 'transformer', 'vae', 'text_encoder'].
            'core' (default) downloads the DiT velocity backbone and 3D VAE (~3.0 GB).
        force_download: If True, re-downloads even if files exist locally.
        max_workers: Number of concurrent download workers.
        
    Returns:
        Path to the local model directory.
    """
    out_path = Path(local_dir).resolve()
    subcomponent = subcomponent.lower()
    if subcomponent not in SUBCOMPONENT_PATTERNS:
        raise ValueError(
            f"Unknown subcomponent: {subcomponent!r}. Choose from {list(SUBCOMPONENT_PATTERNS.keys())}"
        )

    if not force_download and is_already_downloaded(out_path, subcomponent):
        print(f"[Y-Flow] Checkpoint for {subcomponent!r} already exists at: {out_path}")
        print("[Y-Flow] Skipping download. Using existing local weights.")
        return out_path

    allow_patterns = SUBCOMPONENT_PATTERNS[subcomponent]
    print(f"[Y-Flow] Downloading Wan2.1 ({subcomponent}) from '{repo_id}'...")
    print(f"[Y-Flow] Target directory: {out_path}")
    print(f"[Y-Flow] Allowed patterns: {allow_patterns}")

    out_path.mkdir(parents=True, exist_ok=True)

    downloaded_dir = snapshot_download(
        repo_id=repo_id,
        local_dir=str(out_path),
        allow_patterns=allow_patterns,
        local_files_only=False,
        max_workers=max_workers,
    )

    print(f"[Y-Flow] Successfully downloaded Wan2.1 weights to: {downloaded_dir}")
    return Path(downloaded_dir)


DEFAULT_CLEVRER_EVAL_DIR = "checkpoints/clevrer"

# Official CLEVRER / DCL artifacts for physics evaluation of generated video.
# CLEVRER-finetuned Mask R-CNN weights were never published (Yi et al., ICLR 2020).
CLEVRER_ARTIFACTS: dict[str, dict[str, object]] = {
    "visual_masks": {
        "kind": "parser_output",
        "applies_to": "original_clevrer_only",
        "weight_status": "not_a_checkpoint",
        "source": "https://data.csail.mit.edu/clevrer/derender_proposals.zip",
        "license": "CC0 (CLEVRER dataset page)",
        "description": (
            "Mask R-CNN video-frame parser outputs (object masks and attributes) "
            "for original CLEVRER clips. Not a runnable detector for new video."
        ),
    },
    "propnet_preds": {
        "kind": "precomputed_dynamics",
        "applies_to": "original_clevrer_only",
        "weight_status": "not_a_checkpoint",
        "gdrive_id": "1u2OdG59Zl1PqNAnXZjDVMmhXSy3czR44",
        "source": "https://drive.google.com/file/d/1u2OdG59Zl1PqNAnXZjDVMmhXSy3czR44/view",
        "license": "CLEVRER official code release",
        "description": (
            "Precomputed PropNet trajectories and collisions on original CLEVRER. "
            "Not a checkpoint that can be run on Wan-generated clips."
        ),
    },
    "mask_rcnn": {
        "kind": "runnable_detector",
        "applies_to": "any_rgb_frame",
        "weight_status": "public_coco_not_clevrer_finetuned",
        "source": "torchvision MaskRCNN_ResNet50_FPN COCO",
        "license": "BSD (torchvision) / COCO pretrained",
        "description": (
            "Runnable Mask R-CNN for arbitrary frames. Official CLEVRER-finetuned "
            "weights are unpublished; COCO weights are the public substitute."
        ),
    },
    "propnet": {
        "kind": "runnable_dynamics",
        "applies_to": "proposal_tubes_not_raw_video",
        "weight_status": "dcl_release_not_official_clevrer",
        "gdrive_folder": "16FnmnZBb11ge_gJNWUMp8EACRZ5nKe-W",
        "source": "https://drive.google.com/drive/folders/16FnmnZBb11ge_gJNWUMp8EACRZ5nKe-W",
        "license": "DCL (Chen et al., ICLR 2021) release",
        "description": (
            "DCL PropNet .pth files. Official CLEVRER training checkpoints are "
            "unpublished. These weights consume tube proposals, not raw RGB."
        ),
    },
}


def _http_download(url: str, destination: Path, *, force: bool = False) -> Path:
    """Stream to a partial file, resuming only when the server honors Range."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not force:
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    if force:
        partial.unlink(missing_ok=True)
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"User-Agent": "Y-Flow-CLEVRER/1.0"})
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                resumed = response.status == 206
                if resumed:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise ValueError(f"Unexpected Content-Range: {content_range}")
                else:
                    offset = 0
                length = response.headers.get("Content-Length")
                expected = offset + int(length) if length is not None else None
                written = offset
                last_report = time.monotonic()
                with partial.open("ab" if resumed else "wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        written += len(chunk)
                        if time.monotonic() - last_report > 5:
                            total = f" / {expected / 2**20:.0f}" if expected else ""
                            print(f"  {destination.name}: {written / 2**20:.0f}{total} MiB", flush=True)
                            last_report = time.monotonic()
                if expected is not None and written != expected:
                    raise OSError(f"Incomplete download: {written} of {expected} bytes")
            partial.replace(destination)
            return destination
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError(f"Could not download {url}")


def _safe_members(names: list[str]) -> list[str]:
    kept = []
    for name in names:
        path = PurePosixPath(name.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe archive member: {name}")
        kept.append(name)
    return kept


def _flatten_extracted(source: Path, destination: Path, pattern: str, *, keep_parents: int = 0) -> int:
    """Copy matching files to destination, optionally keeping trailing parent dirs."""
    import re

    regex = re.compile(pattern)
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in source.rglob("*"):
        if not path.is_file() or "__MACOSX" in path.parts:
            continue
        if regex.fullmatch(path.name) is None:
            continue
        rel = path.relative_to(source)
        kept = Path(*rel.parts[-(keep_parents + 1):]) if keep_parents else Path(path.name)
        target = destination / kept
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.resolve() == path.resolve():
            count += 1
            continue
        shutil.copy2(path, target)
        count += 1
    return count


def extract_clevrer_archive(
    archive: Path,
    destination: Path,
    name_pattern: str,
    *,
    keep_parents: int = 0,
) -> int:
    """Extract zip or tar.gz, flatten matching members, reject path traversal."""
    staging = destination.parent / (destination.name + ".extracting")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        import re

        regex = re.compile(name_pattern)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as source:
                names = _safe_members(source.namelist())
                count = 0
                destination.mkdir(parents=True, exist_ok=True)
                for info in source.infolist():
                    if info.is_dir():
                        continue
                    if stat.S_ISLNK(info.external_attr >> 16):
                        raise ValueError(f"Archive symlink: {info.filename}")
                    path = PurePosixPath(info.filename)
                    if regex.fullmatch(path.name) is None:
                        continue
                    kept = Path(*path.parts[-(keep_parents + 1):]) if keep_parents else Path(path.name)
                    target = destination / kept
                    target.parent.mkdir(parents=True, exist_ok=True)
                    partial = target.with_suffix(target.suffix + ".part")
                    with source.open(info) as stream, partial.open("wb") as output:
                        shutil.copyfileobj(stream, output)
                    partial.replace(target)
                    count += 1
                if count == 0:
                    raise ValueError(f"No files matching {name_pattern} in {archive}")
                return count
        else:
            with tarfile.open(archive, "r:*") as source:
                members = []
                for member in source.getmembers():
                    if member.issym() or member.islnk():
                        raise ValueError(f"Archive symlink: {member.name}")
                    _safe_members([member.name])
                    if member.isfile():
                        members.append(member)
                source.extractall(staging, members=members)
        count = _flatten_extracted(staging, destination, name_pattern, keep_parents=keep_parents)
        if count == 0:
            raise ValueError(f"No files matching {name_pattern} in {archive}")
        return count
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _gdown_file(file_id: str, destination: Path, *, force: bool = False) -> Path:
    if destination.is_file() and not force:
        return destination
    try:
        import gdown
    except ImportError as exc:
        raise ImportError("Google Drive downloads require gdown: pip install gdown") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[Y-Flow] gdown {url} -> {destination}", flush=True)
    out = gdown.download(id=file_id, output=str(destination), quiet=False)
    if not out or not Path(out).is_file():
        raise FileNotFoundError(f"gdown failed for {file_id}")
    return Path(out)


def _gdown_folder(folder_id: str, destination: Path, *, force: bool = False) -> Path:
    if destination.is_dir() and any(destination.iterdir()) and not force:
        return destination
    try:
        import gdown
    except ImportError as exc:
        raise ImportError("Google Drive downloads require gdown: pip install gdown") from exc
    destination.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    print(f"[Y-Flow] gdown folder {url} -> {destination}", flush=True)
    gdown.download_folder(url=url, output=str(destination), quiet=False)
    return destination


def is_clevrer_artifact_ready(root: Path | str, name: str) -> bool:
    root = Path(root)
    if name == "visual_masks":
        files = list((root / "visual_masks").glob("proposal_*.json")) + list((root / "visual_masks").glob("sim_*.json"))
        return len(files) >= 100
    if name == "propnet_preds":
        files = list((root / "propnet_preds").rglob("sim_*.json")) + list((root / "propnet_preds").rglob("proposal_*.json"))
        return len(files) >= 100
    if name == "mask_rcnn":
        return (root / "mask_rcnn" / "READY.json").is_file()
    if name == "propnet":
        return any((root / "propnet").rglob("*.pth"))
    raise ValueError(f"Unknown CLEVRER artifact: {name}")


def download_clevrer_artifact(
    name: str,
    root: str | Path = DEFAULT_CLEVRER_EVAL_DIR,
    *,
    force: bool = False,
) -> Path:
    """Download one CLEVRER evaluation artifact into checkpoints/clevrer."""
    if name not in CLEVRER_ARTIFACTS:
        raise ValueError(f"Unknown artifact {name!r}. Choose from {list(CLEVRER_ARTIFACTS)}")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    spec = CLEVRER_ARTIFACTS[name]
    print(f"[Y-Flow] {name}: {spec['kind']} / {spec['weight_status']}", flush=True)

    if name == "visual_masks":
        target = root / "visual_masks"
        if not force and is_clevrer_artifact_ready(root, name):
            print(f"[Y-Flow] visual_masks already present in {target}")
            return target
        archive = root / "archives" / "derender_proposals.zip"
        _http_download(str(spec["source"]), archive, force=force)
        n = extract_clevrer_archive(archive, target, r"(proposal|sim)_\d{5}\.json")
        print(f"[Y-Flow] extracted {n} visual-mask json files")
        return target

    if name == "propnet_preds":
        target = root / "propnet_preds"
        if not force and is_clevrer_artifact_ready(root, name):
            print(f"[Y-Flow] propnet_preds already present in {target}")
            return target
        archive = root / "archives" / "propnet_preds.bin"
        _gdown_file(str(spec["gdrive_id"]), archive, force=force)
        n = extract_clevrer_archive(archive, target, r"(proposal|sim)_\d{5}\.json", keep_parents=1)
        print(f"[Y-Flow] extracted {n} PropNet prediction json files")
        return target

    if name == "mask_rcnn":
        target = root / "mask_rcnn"
        target.mkdir(parents=True, exist_ok=True)
        if not force and is_clevrer_artifact_ready(root, name):
            print(f"[Y-Flow] mask_rcnn already present in {target}")
            return target
        import torch
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights

        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn(weights=weights)
        ckpt = target / "maskrcnn_resnet50_fpn_coco.pth"
        torch.save(model.state_dict(), ckpt)
        payload = {
            "source": "torchvision.models.detection.maskrcnn_resnet50_fpn",
            "weights": str(weights),
            "url": weights.url,
            "num_classes": 91,
            "clevrer_finetuned": False,
            "applies_to": "any_rgb_frame",
            "note": "Official CLEVRER Mask R-CNN checkpoint was never published.",
        }
        (target / "READY.json").write_text(json.dumps(payload, indent=2) + "\n")
        return target

    if name == "propnet":
        target = root / "propnet"
        if not force and is_clevrer_artifact_ready(root, name):
            print(f"[Y-Flow] propnet weights already present in {target}")
            return target
        _gdown_folder(str(spec["gdrive_folder"]), target, force=force)
        if not any(target.rglob("*.pth")):
            raise FileNotFoundError(
                f"DCL PropNet folder downloaded but no .pth found in {target}. "
                "Official CLEVRER PropNet training weights remain unpublished."
            )
        return target

    raise AssertionError(name)


def write_clevrer_artifact_manifest(root: str | Path = DEFAULT_CLEVRER_EVAL_DIR) -> Path:
    root = Path(root)
    records = {}
    for name, spec in CLEVRER_ARTIFACTS.items():
        records[name] = {**spec, "ready": is_clevrer_artifact_ready(root, name), "path": str(root / name)}
    path = root / "artifacts.json"
    path.write_text(json.dumps({"dataset": "clevrer", "artifacts": records}, indent=2) + "\n")
    return path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download Wan2.1 weights from Hugging Face.")
    parser.add_argument(
        "--repo_id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace repo ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=DEFAULT_LOCAL_DIR,
        help=f"Local target directory (default: {DEFAULT_LOCAL_DIR})",
    )
    parser.add_argument(
        "--subcomponent",
        type=str,
        default="core",
        choices=list(SUBCOMPONENT_PATTERNS.keys()),
        help=(
            "Subcomponents to download: 'core' (DiT + VAE, ~3GB), 'all' (full pipeline, ~13GB), "
            "'transformer' (DiT only, ~2.6GB), 'vae' (~400MB), 'text_encoder' (~10GB)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of parallel download threads.",
    )
    args = parser.parse_args(argv)

    download_wan_model(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        subcomponent=args.subcomponent,
        force_download=args.force,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
