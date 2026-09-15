"""Prepare 10 small RealEstate10K development clips, keeping only sampled PNGs.

Official poses + original source video timestamps; no pose interpolation.
Local --poses-dir/--videos-dir inputs support reproducible offline preparation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
from data.realestate10k import parse_camera_file

POSE_URL = "https://storage.googleapis.com/realestate10k-public-files/RealEstate10K.tar.gz"


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def fetch_poses(root):
    destination = root / "poses"
    if (destination / "READY.json").exists():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    # Temporary archive is removed on success or failure; never extract paths.
    with tempfile.TemporaryDirectory(dir=root, prefix="poses-download-") as work:
        archive = Path(work) / "poses.tar.gz"
        print(f"Downloading camera metadata: {POSE_URL}", flush=True)
        with urllib.request.urlopen(POSE_URL, timeout=60) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        with tarfile.open(archive, "r:gz") as source:
            for member in source:
                parts = Path(member.name).parts
                if not member.isfile() or len(parts) < 2 or parts[-2] not in ("train", "test") or not parts[-1].endswith(".txt"):
                    continue
                output = destination / parts[-2] / parts[-1]
                output.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as incoming, output.open("wb") as target:
                    shutil.copyfileobj(incoming, target)
    if not all(list((destination / split).glob("*.txt")) for split in ("train", "test")):
        raise ValueError("Camera archive is missing train/test files")
    atomic_json(destination / "READY.json", {"source": POSE_URL})
    return destination


def select_indices(timestamps, n_frames, interval_s):
    requested = timestamps[0] + np.arange(n_frames) * interval_s * 1e6
    if requested[-1] > timestamps[-1]:
        raise ValueError("Clip is shorter than requested sampling window")
    right = np.searchsorted(timestamps, requested).clip(0, len(timestamps)-1)
    left = (right-1).clip(0)
    indices = np.where(abs(timestamps[left]-requested) <= abs(timestamps[right]-requested), left, right)
    if len(np.unique(indices)) != n_frames:
        raise ValueError("Requested sampling repeats camera rows; increase --interval")
    return indices, np.rint(requested).astype(np.int64)


def extract_frames(video, timestamps_us, output, width, height, tolerance_us):
    """Stream nearest PTS frames without loading the source video into RAM."""
    import av
    targets = timestamps_us / 1e6
    actual, files, sizes = [], [], []
    previous = None
    index = 0

    def save(candidate):
        timestamp, rgb = candidate
        if abs(timestamp * 1e6 - timestamps_us[index]) > tolerance_us:
            raise ValueError(f"Video/pose alignment exceeds {tolerance_us} us")
        path = output / f"{index:04d}.png"
        image = Image.fromarray(rgb)
        sizes.append(image.size)
        image.resize((width, height), Image.Resampling.LANCZOS).save(path)
        actual.append(round(timestamp * 1e6))
        files.append(path.name)

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        # PTS remain absolute source-video time; do not subtract first decoded PTS.
        for frame in container.decode(stream):
            if frame.time is None:
                raise ValueError("Source video has no frame timestamps")
            now = (float(frame.time), frame.to_ndarray(format="rgb24"))
            if previous is not None and now[0] <= previous[0]:
                raise ValueError("Non-increasing video timestamps")
            while index < len(targets) and now[0] >= targets[index]:
                candidate = min([now] + ([previous] if previous else []), key=lambda v: abs(v[0]-targets[index]))
                save(candidate)
                index += 1
            if index == len(targets):
                break
            previous = now
    if index != len(targets) or len(set(actual)) != len(actual):
        raise ValueError("Missing or duplicate sampled video frames")
    return files, np.asarray(actual, dtype=np.int64), np.asarray(sizes, dtype=np.int64)


def download_video(url, work):
    try:
        import yt_dlp
    except ImportError as error:
        raise RuntimeError("Install yt-dlp or supply --videos-dir for offline setup") from error
    # Prefer a single progressive stream so PyAV can open it without relying
    # on an external ffmpeg merge. The final /best handles videos whose
    # available stream height is above 720 or whose container is not mp4.
    options = {
        "format": "best[height<=720]/best",
        "outtmpl": str(work / "source.%(ext)s"),
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 2,
        "cachedir": False,
        "quiet": True,
        # YouTube format discovery now needs a JS runtime. yt-dlp[default]
        # supplies the EJS solver; Deno must be available on PATH.
        "js_runtimes": {"deno": {}},
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        try:
            info = downloader.extract_info(url, download=True)
        except yt_dlp.utils.DownloadError as error:
            raise RuntimeError(str(error)) from error
        return Path(downloader.prepare_filename(info))


def prepare(args):
    if args.width % 16 or args.height % 16:
        raise ValueError("Wan2.1 setup dimensions must be divisible by 16 (VAE × transformer patch)")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    poses = args.poses_dir.resolve() if args.poses_dir else fetch_poses(root)
    candidates = sorted((poses / args.split).glob("*.txt"))
    if not candidates:
        raise ValueError(f"No pose files in {poses / args.split}")
    np.random.default_rng(args.seed).shuffle(candidates)
    settings = dict(n_frames=args.n_frames, interval_s=args.interval, width=args.width,
                    height=args.height, seed=args.seed, tolerance_us=args.tolerance_us,
                    pose_source=str(poses), video_source=str(args.videos_dir.resolve()) if args.videos_dir else "source_url")
    manifest_path = root / "manifests" / f"{args.split}.json"
    manifest = dict(schema_version=1, split=args.split, settings=settings, clips=[], failures=[])
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["settings"] != settings:
            raise ValueError("Existing manifest settings differ; use a separate --root")
        for clip in manifest["clips"]:
            if not all((root / p).is_file() for p in clip["frames"] + [clip["cameras"]]):
                raise ValueError("Incomplete prepared clip; use a clean output root")
    used_ids = {c["clip_id"] for c in manifest["clips"]}
    # These failures cannot be repaired by retrying the same source with a
    # different format selector or downloader runtime.
    permanent_failure_ids = {
        failure["clip_id"] for failure in manifest["failures"]
        if "Private video" in failure["error"]
        or "Clip is shorter than requested sampling window" in failure["error"]
    }
    used_urls = {c["source_url"] for c in manifest["clips"]}
    other = root / "manifests" / f"{'test' if args.split == 'train' else 'train'}.json"
    if other.exists():
        used_urls.update(c["source_url"] for c in json.loads(other.read_text())["clips"])
    attempts = 0
    for pose_path in candidates:
        if len(manifest["clips"]) >= args.n_clips or attempts >= args.max_attempts:
            break
        if pose_path.stem in used_ids or pose_path.stem in permanent_failure_ids:
            continue
        attempts += 1
        try:
            camera = parse_camera_file(pose_path)
            if camera["source_url"] in used_urls:
                continue
            indices, requested = select_indices(camera["timestamps_us"], args.n_frames, args.interval)
            with tempfile.TemporaryDirectory(dir=root, prefix="prepare-") as temporary:
                work = Path(temporary)
                video = args.videos_dir / f"{pose_path.stem}.mp4" if args.videos_dir else download_video(camera["source_url"], work)
                images = work / "frames"
                images.mkdir()
                times = camera["timestamps_us"][indices]
                files, actual, sizes = extract_frames(video, times, images, args.width, args.height, args.tolerance_us)
                K = np.diag([args.width, args.height, 1.0])[None] @ camera["K_normalized"][indices]
                A = np.stack([np.diag([args.width/w, args.height/h, 1.0]) for w, h in sizes])
                np.savez(images / "cameras.npz", K=K, K_normalized=camera["K_normalized"][indices],
                         world_to_camera=camera["world_to_camera"][indices],
                         timestamps_us=times, requested_timestamps_us=requested,
                         actual_timestamps_us=actual, camera_indices=indices,
                         spatial_transform=A, source_sizes_wh=sizes)
                shutil.copy2(pose_path, images / "source_camera.txt")
                destination = root / "clips" / args.split / pose_path.stem
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise ValueError(f"Unindexed output exists: {destination}")
                images.rename(destination)
            relative = destination.relative_to(root)
            manifest["clips"].append(dict(clip_id=pose_path.stem, source_url=camera["source_url"],
                frames=[str(relative / f) for f in files], cameras=str(relative / "cameras.npz"),
                static_scene_review="pending"))
            used_urls.add(camera["source_url"])
            print(f"Prepared {len(manifest['clips'])}/{args.n_clips}: {pose_path.stem}", flush=True)
        except (ValueError, OSError, RuntimeError) as error:
            manifest["failures"].append(dict(clip_id=pose_path.stem, error=str(error)))
            print(f"Skipped {pose_path.stem}: {error}", flush=True)
        atomic_json(manifest_path, manifest)
    atomic_json(manifest_path, manifest)
    if len(manifest["clips"]) < args.n_clips:
        raise RuntimeError(f"Only {len(manifest['clips'])}/{args.n_clips} clips prepared; see {manifest_path}")
    print(f"Ready: {manifest_path} (static-scene review still required)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "datasets/realestate10k")
    parser.add_argument("--poses-dir", type=Path, help="Directory containing train/*.txt and/or test/*.txt")
    parser.add_argument("--videos-dir", type=Path, help="Offline original videos named <clip_id>.mp4")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--n-clips", type=int, default=10)
    parser.add_argument("--n-frames", type=int, default=8)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--tolerance-us", type=int, default=20000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.n_clips, args.width, args.height, args.max_attempts, args.tolerance_us) <= 0 or args.n_frames < 2 or not np.isfinite(args.interval) or args.interval <= 0:
        parser.error("Positive sizes/counts/tolerance/interval and n-frames >= 2 required")
    if args.dry_run:
        print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
        return
    required = [name for name in ("av",) if importlib.util.find_spec(name) is None]
    if args.videos_dir is None and importlib.util.find_spec("yt_dlp") is None:
        required.append("yt-dlp")
    if required:
        raise RuntimeError(
            "Missing setup dependencies: " + ", ".join(required)
            + ". Install them in this Python environment with: "
            "python -m pip install yt-dlp av pillow"
        )
    prepare(args)


if __name__ == "__main__":
    main()
