#!/usr/bin/env python
import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("fetch_chunk", os.path.join(HERE, "05_fetch_chunk.py"))
fc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=sorted(fc.URLS))
    ap.add_argument("--tokens", nargs="+", required=True)
    a = ap.parse_args()

    data_root = os.environ.get("NUPLAN_DATA_ROOT", "/root/nuplan/dataset")
    base = os.path.join(data_root, "chunks", a.split)
    st_path = os.path.join(base, "state.json")
    if not os.path.exists(st_path):
        sys.exit(f"[err] {st_path} 없음 (05_fetch_chunk.py 로 받은 적이 있어야 토큰->로그를 알 수 있음)")
    st = json.load(open(st_path))
    logs = {}
    for t in a.tokens:
        name = st["token_log"].get(t.lower())
        if not name:
            sys.exit(f"[err] 토큰 {t} 가 {a.split} state.json 에 없음")
        logs.setdefault(name, []).append(t)
    out = os.path.join(base, "viz")
    os.makedirs(out, exist_ok=True)
    todo = [n for n in logs if not os.path.exists(os.path.join(out, n))]
    for n in logs:
        print(f"[log] {n}  <- 토큰 {logs[n]}" + ("" if n in todo else "  (이미 있음)"))
    if not todo:
        print(f"[done] {out}")
        return
    rf = fc.HTTPRangeFile(st.get("url") or fc.URLS[a.split])
    zf = zipfile.ZipFile(rf)
    by_name = {os.path.basename(zi.filename): zi for zi in zf.infolist() if zi.filename.endswith(".db")}
    for n in todo:
        zi = by_name.get(n)
        if zi is None:
            sys.exit(f"[err] zip 안에 {n} 없음")
        t0 = time.time()
        part = os.path.join(out, n + ".partial")
        with zf.open(zi) as src, open(part, "wb") as dst:
            shutil.copyfileobj(src, dst, 8 << 20)
        os.replace(part, os.path.join(out, n))
        print(f"[get] {n} {zi.file_size / 1e9:.2f} GB, {time.time() - t0:.0f}s", flush=True)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
