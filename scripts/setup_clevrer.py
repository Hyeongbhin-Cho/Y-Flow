# -*- coding: utf-8 -*-
# scripts/setup_clevrer.py
"""Download official CLEVRER videos and annotations into datasets/clevrer."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import stat
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://data.csail.mit.edu/clevrer"
SPLITS = {"train": (0, 10000), "validation": (10000, 15000), "test": (15000, 20000)}


def download_file(url: str, destination: Path, *, force: bool = False) -> Path:
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


def extract_archive(archive: Path, destination: Path, kind: str, split: str) -> None:
    """Flatten official archive subdirectories; reject unsafe/duplicate members."""
    prefix, suffix = ("video", "mp4") if kind == "videos" else ("annotation", "json")
    pattern = re.compile(rf"{prefix}_(\d{{5}})\.{suffix}")
    start, end = SPLITS[split]
    with zipfile.ZipFile(archive) as source:
        members = []
        names = set()
        for member in source.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Archive symlink: {member.filename}")
            if member.is_dir() or "__MACOSX" in path.parts:
                continue
            match = pattern.fullmatch(path.name)
            if match is None:
                continue
            if not start <= int(match[1]) < end or path.name in names:
                raise ValueError(f"Invalid or duplicate archive member: {member.filename}")
            names.add(path.name)
            members.append(member)
        if not members:
            raise ValueError(f"No {kind} found in {archive}")
        destination.mkdir(parents=True, exist_ok=True)
        for member in members:
            target = destination / PurePosixPath(member.filename).name
            partial = target.with_suffix(target.suffix + ".part")
            # Reading to EOF checks the ZIP CRC before committing each file.
            with source.open(member) as stream, partial.open("wb") as output:
                shutil.copyfileobj(stream, output)
            partial.replace(target)


def write_manifest(root: Path, split: str, *, expected_count: int | None = None) -> Path:
    records = []
    for video in sorted((root / "videos" / split).glob("video_*.mp4")):
        scene_id = int(video.stem.removeprefix("video_"))
        annotation = root / "annotations" / split / f"annotation_{scene_id:05d}.json"
        if video.stat().st_size == 0:
            raise ValueError(f"Empty video: {video}")
        if split != "test":
            payload = json.loads(annotation.read_text())
            if payload.get("scene_index") != scene_id:
                raise ValueError(f"Annotation/video mismatch: {annotation}")
        records.append({
            "scene_index": scene_id,
            "video": video.relative_to(root).as_posix(),
            "annotation": annotation.relative_to(root).as_posix() if split != "test" else None,
        })
    if not records or (expected_count is not None and len(records) != expected_count):
        raise ValueError(f"Incomplete {split} split: {len(records)} videos, expected {expected_count}")
    path = root / "manifests" / f"{split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": "clevrer", "split": split, "source": BASE_URL, "records": records}
    partial = path.with_suffix(".json.part")
    partial.write_text(json.dumps(payload, indent=2) + "\n")
    partial.replace(path)
    return path


def setup_clevrer(root: Path, splits: list[str], *, questions: bool = False, force: bool = False) -> None:
    for split in splits:
        for kind, filename in [("videos", f"video_{split}.zip"), ("annotations", f"annotation_{split}.zip")]:
            if split == "test" and kind == "annotations":
                continue
            url = f"{BASE_URL}/{kind}/{split}/{filename}"
            archive = root / "archives" / filename
            print(f"[Y-Flow] {url}", flush=True)
            download_file(url, archive, force=force)
            marker = archive.with_suffix(".extracted")
            target = root / kind / split
            expected = SPLITS[split][1] - SPLITS[split][0]
            extension = "*.mp4" if kind == "videos" else "*.json"
            if force or not marker.exists() or len(list(target.glob(extension))) != expected:
                extract_archive(archive, target, kind, split)
                marker.touch()
        if questions:
            path = root / "questions" / f"{split}.json"
            download_file(f"{BASE_URL}/questions/{split}.json", path, force=force)
            json.loads(path.read_text())
        path = write_manifest(root, split, expected_count=expected)
        print(f"[Y-Flow] Ready: {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "datasets" / "clevrer")
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=["validation"])
    parser.add_argument("--questions", action="store_true", help="Also download optional VQA files.")
    parser.add_argument("--force", action="store_true", help="Download and extract again.")
    parser.add_argument("--dry-run", action="store_true", help="Print URLs without downloading.")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else ROOT / args.root
    if args.dry_run:
        print(f"Target: {root}")
        for split in args.splits:
            print(f"{BASE_URL}/videos/{split}/video_{split}.zip")
            if split != "test":
                print(f"{BASE_URL}/annotations/{split}/annotation_{split}.zip")
            if args.questions:
                print(f"{BASE_URL}/questions/{split}.json")
        return
    setup_clevrer(root, args.splits, questions=args.questions, force=args.force)


if __name__ == "__main__":
    main()
