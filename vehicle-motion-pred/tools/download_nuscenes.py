from __future__ import annotations

import argparse
import os
import tarfile
import urllib.request
import zipfile
from pathlib import Path

TABLES = ("attribute", "calibrated_sensor", "category", "ego_pose", "instance", "log", "map", "sample",
          "sample_annotation", "sample_data", "scene", "sensor", "visibility")


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"reusing {dest}")
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(req, timeout=120) as r, part.open("ab" if offset else "wb") as f:
        total = r.headers.get("Content-Length")
        done = 0
        while True:
            block = r.read(1 << 22)
            if not block:
                break
            f.write(block)
            done += len(block)
            if total and done % (1 << 28) < (1 << 22):
                print(f"  {dest.name}: {(offset + done) / 1e9:.2f} GB", flush=True)
    part.replace(dest)
    return dest


def extract(archive: Path, root: Path, version: str) -> int:
    root.mkdir(parents=True, exist_ok=True)
    n = 0

    def keep(name: str) -> Path | None:
        parts = Path(name).parts
        for i, p in enumerate(parts):
            if p == version:
                return Path(*parts[i:])
            if p == "maps":
                return Path(*parts[i:])
            if p in ("expansion", "prediction", "basemap"):
                return Path("maps", *parts[i:])
        return None

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for m in z.infolist():
                rel = keep(m.filename)
                if rel is None or m.is_dir():
                    continue
                out = root / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as src, out.open("wb") as dst:
                    dst.write(src.read())
                n += 1
    else:
        with tarfile.open(archive, "r:*") as t:
            for m in t:
                rel = keep(m.name)
                if rel is None or not m.isfile():
                    continue
                out = root / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                with t.extractfile(m) as src, out.open("wb") as dst:
                    dst.write(src.read())
                n += 1
    return n


def verify(root: Path, version: str) -> None:
    missing = [f"{version}/{t}.json" for t in TABLES if not (root / version / f"{t}.json").is_file()]
    missing += [p for p in ("maps/prediction/prediction_scenes.json",) if not (root / p).is_file()]
    for name in ("boston-seaport", "singapore-onenorth", "singapore-queenstown", "singapore-hollandvillage"):
        if not (root / "maps/expansion" / f"{name}.json").is_file():
            missing.append(f"maps/expansion/{name}.json")
    if not list((root / "maps").glob("*.png")):
        missing.append("maps/*.png (base map rasters from the metadata archive)")
    if missing:
        raise SystemExit("missing: " + ", ".join(missing))
    print(f"nuScenes {version} layout OK at {root}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/nuscenes")
    ap.add_argument("--profile", choices=["mini", "trainval"], default="trainval")
    ap.add_argument("--meta_url", default=os.environ.get("NUSCENES_META_URL"))
    ap.add_argument("--map_url", default=os.environ.get("NUSCENES_MAP_URL"))
    ap.add_argument("--archives", default="data/archives")
    ap.add_argument("--verify_only", action="store_true")
    args = ap.parse_args()
    root, version = Path(args.root), f"v1.0-{args.profile}"
    if not args.verify_only:
        if not args.meta_url or not args.map_url:
            raise SystemExit("set NUSCENES_META_URL and NUSCENES_MAP_URL (direct links from nuscenes.org, login required)")
        arc = Path(args.archives)
        for label, url in ((f"meta_{args.profile}", args.meta_url), ("maps", args.map_url)):
            suffix = ".zip" if ".zip" in url.split("?")[0] else ".tgz"
            path = download(url, arc / f"{label}{suffix}")
            print(f"{label}: extracted {extract(path, root, version)} files")
    verify(root, version)


if __name__ == "__main__":
    main()
