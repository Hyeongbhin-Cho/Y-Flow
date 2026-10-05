#!/usr/bin/env python
import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import OrderedDict

import yaml

URLS = {
    "val14": "https://motional-nuplan.s3.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_val.zip",
    "test14-hard": "https://motional-nuplan.s3.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_test.zip",
    "test14-random": "https://motional-nuplan.s3.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_test.zip",
}


class HTTPRangeFile:

    def __init__(self, url, block=16 << 20, cache_blocks=4, retries=8):
        self.url, self.block, self.cache_blocks, self.retries = url, block, cache_blocks, retries
        self.pos = 0
        self.cache = OrderedDict()
        self.size = self._head_size()
        self.bytes_fetched = 0

    def _open(self, req):
        for i in range(self.retries):
            try:
                return urllib.request.urlopen(req, timeout=120).read()
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                wait = min(60, 2 ** i)
                print(f"[net] {type(e).__name__}: {e} -> retry in {wait}s", flush=True)
                time.sleep(wait)
        raise RuntimeError(f"download failed after {self.retries} retries")

    def _head_size(self):
        req = urllib.request.Request(self.url, method="HEAD")
        for i in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    if r.headers.get("Accept-Ranges", "").lower() != "bytes":
                        raise RuntimeError("server does not support HTTP Range")
                    return int(r.headers["Content-Length"])
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                time.sleep(min(60, 2 ** i))
        raise RuntimeError("HEAD failed")

    def _get_block(self, idx):
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]
        start = idx * self.block
        end = min(self.size, start + self.block) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        data = self._open(req)
        if len(data) != end - start + 1:
            raise RuntimeError(f"short read {len(data)} != {end - start + 1}")
        self.bytes_fetched += len(data)
        self.cache[idx] = data
        while len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return data

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            idx, off = divmod(self.pos, self.block)
            blk = self._get_block(idx)
            take = min(n, len(blk) - off)
            out += blk[off:off + take]
            self.pos += take
            n -= take
        return bytes(out)

    def close(self):
        self.cache.clear()


def load_tokens(split, fd_root):
    p = os.path.join(fd_root, "flow_drive", "config", "scenario_filter", f"{split}.yaml")
    with open(p) as f:
        d = yaml.safe_load(f)
    toks = d.get("scenario_tokens")
    if not toks:
        sys.exit(f"[err] {p} 에 scenario_tokens 없음")
    return {t.lower() for t in toks}


def tokens_in_db(path, wanted):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT token FROM lidar_pc").fetchall()
    finally:
        con.close()
    found = set()
    for (t,) in rows:
        h = t.hex() if isinstance(t, (bytes, bytearray, memoryview)) else str(t).lower()
        if h in wanted:
            found.add(h)
    return found


def save_state(path, st):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, path)


def gb(x):
    return x / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=sorted(URLS))
    ap.add_argument("--budget_gb", type=float, default=float(os.environ.get("CHUNK_GB", 45)))
    ap.add_argument("--min_free_gb", type=float, default=12.0, help="이보다 여유 공간이 적어지면 멈춤")
    ap.add_argument("--max_pending", type=int, default=1, help="아직 실행 안 끝난 청크가 이만큼 있으면 새로 안 받음")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--url", default=None)
    a = ap.parse_args()

    data_root = os.environ.get("NUPLAN_DATA_ROOT", "/root/nuplan/dataset")
    fd_root = os.environ.get("FD_ROOT", "/root/code/flow_drive_planner")
    base = os.path.join(data_root, "chunks", a.split)
    os.makedirs(base, exist_ok=True)
    st_path = os.path.join(base, "state.json")
    wanted = load_tokens(a.split, fd_root)
    st = json.load(open(st_path)) if os.path.exists(st_path) else {
        "split": a.split, "url": a.url or URLS[a.split], "n_tokens": len(wanted),
        "processed": {}, "chunks": [], "token_log": {}, "zip_members": None}

    found = set(st["token_log"])
    pending = [c for c in st["chunks"] if c["status"] == "fetched"]
    print(f"[state] {a.split}: 토큰 {len(found)}/{len(wanted)} 찾음 | 처리한 db {len(st['processed'])}"
          + (f"/{st['zip_members']}" if st["zip_members"] else "")
          + f" | 청크 {[(c['id'], c['status'], len(c['logs']), round(c['gb'], 1)) for c in st['chunks']]}")
    if a.status:
        return
    if found >= wanted:
        print("[done] 모든 벤치마크 토큰을 이미 찾음. 더 받을 것 없음")
        return
    if len(pending) >= a.max_pending:
        print(f"[wait] 실행이 안 끝난 청크 {[c['id'] for c in pending]} 가 있음 -> 06_run_chunk.sh 먼저 (또는 --max_pending 2)")
        return

    if st["chunks"] and st["chunks"][-1]["status"] == "fetching":
        chunk = st["chunks"][-1]
        cid, cdir = chunk["id"], chunk["dir"]
        print(f"[resume] c{cid:02d} 이어서 받음")
    else:
        cid = len(st["chunks"]) + 1
        cdir = os.path.join(base, f"c{cid:02d}")
        chunk = {"id": cid, "dir": cdir, "logs": [], "tokens": 0, "gb": 0.0, "status": "fetching"}
        st["chunks"].append(chunk)
    os.makedirs(cdir, exist_ok=True)
    save_state(st_path, st)

    rf = HTTPRangeFile(st["url"])
    zf = zipfile.ZipFile(rf)
    members = [zi for zi in zf.infolist() if zi.filename.endswith(".db")]
    st["zip_members"] = len(members)
    print(f"[zip] {gb(rf.size):.1f} GB, .db {len(members)}개 | 청크 c{cid:02d} 목표 {a.budget_gb:.0f} GB", flush=True)
    t0 = time.time()
    for zi in members:
        name = os.path.basename(zi.filename)
        if name in st["processed"]:
            continue
        free = shutil.disk_usage(cdir).free
        if gb(free) - gb(zi.file_size) < a.min_free_gb:
            print(f"[stop] 여유 공간 부족 ({gb(free):.1f} GB)")
            break
        if chunk["gb"] > 0 and chunk["gb"] + gb(zi.file_size) > a.budget_gb:
            print(f"[stop] 청크 용량 {chunk['gb']:.1f} GB 도달")
            break
        part = os.path.join(cdir, name + ".partial")
        with zf.open(zi) as src, open(part, "wb") as dst:
            shutil.copyfileobj(src, dst, 8 << 20)
        hit = tokens_in_db(part, wanted - found)
        if hit:
            os.replace(part, os.path.join(cdir, name))
            chunk["logs"].append(name)
            chunk["tokens"] += len(hit)
            chunk["gb"] += gb(zi.file_size)
            for h in hit:
                st["token_log"][h] = name
            found |= hit
        else:
            os.remove(part)
        st["processed"][name] = len(hit)
        save_state(st_path, st)
        el = time.time() - t0
        print(f"[db] {len(st['processed'])}/{len(members)} {name} {gb(zi.file_size):.2f}GB "
              f"{'KEEP +' + str(len(hit)) if hit else 'skip'} | 토큰 {len(found)}/{len(wanted)} "
              f"| 청크 {chunk['gb']:.1f}GB | {gb(rf.bytes_fetched) / max(el, 1) * 1e3:.0f} MB/s", flush=True)
        if found >= wanted:
            print("[stop] 모든 토큰 찾음")
            break
    chunk["status"] = "fetched" if chunk["logs"] else "empty"
    if not chunk["logs"]:
        shutil.rmtree(cdir, ignore_errors=True)
        chunk["status"] = "done"
    save_state(st_path, st)
    print(f"[chunk] c{cid:02d}: db {len(chunk['logs'])}개, 시나리오 토큰 {chunk['tokens']}개, {chunk['gb']:.1f} GB "
          f"| 전체 {len(found)}/{len(wanted)}")
    if chunk["logs"]:
        print(f"[next] bash 06_run_chunk.sh {a.split} {cid}")


if __name__ == "__main__":
    main()
