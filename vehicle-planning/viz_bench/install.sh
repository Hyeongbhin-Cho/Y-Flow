#!/usr/bin/env bash
set -euo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${1:-/root/flowdrive_yflow_vessl}"
mkdir -p "$DST/.viz1_backup"
for f in 14_render_compare.py 15_viz_run.sh; do [ -f "$DST/$f" ] && cp "$DST/$f" "$DST/.viz1_backup/"; done
cp "$SRC"/14_render_compare.py "$SRC"/15_viz_run.sh "$SRC"/16_fetch_logs.py "$SRC"/17_viz_bench.sh "$DST/"
chmod +x "$DST"/15_viz_run.sh "$DST"/17_viz_bench.sh
grep -q "PICK" "$DST/14_render_compare.py" && echo "[ok] 설치 완료 -> $DST"
