#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
ensure_sysdeps
activate_env
export LD_LIBRARY_PATH="${CONDA_PREFIX:-$CONDA_DIR/envs/$ENV_NAME}/lib:${LD_LIBRARY_PATH:-}"

SPLIT="${1:?split: mini | val14 | test14-hard}"
ID="${2:?chunk id (mini는 0)}"
METHODS="${METHODS:-human pdmc fd fd_post yflow-kin yflow_post-kin fdstar yflow_star-kin}"
NPROC="${NPROC:-4}"
LOGD="$RESULTS_DIR/logs/navsim"; mkdir -p "$LOGD"
EXTRA=(); [ -n "${LIMIT:-}" ] && EXTRA+=(--limit "$LIMIT")
if [ -n "${YSET:-}" ]; then read -r -a _ys <<< "$YSET"; EXTRA+=(--yset "${_ys[@]}" --ytag "${YTAG:?YSET을 쓰면 YTAG도 필요}"); fi
if [ "$ID" != 0 ]; then
    CID=$(printf "c%02d" "$ID"); CDIR="$NUPLAN_DATA_ROOT/chunks/$SPLIT/$CID"
    ls "$CDIR"/*.db >/dev/null 2>&1 || { echo "[err] $CDIR 에 .db 없음 (05_fetch_chunk.py 먼저)"; exit 1; }
    EXTRA+=(--chunk "$ID"); TAGP="${SPLIT}_${CID}"
else
    TAGP="${SPLIT}_all"
fi
[ -n "${LIMIT:-}" ] && NPROC=1

echo "[navsim] $SPLIT id=$ID  methods=[$METHODS]  NPROC=$NPROC  (로그: $LOGD/${TAGP}_s*.log)"
pids=()
for k in $(seq 0 $((NPROC - 1))); do
    SH=(); [ "$NPROC" -gt 1 ] && SH=(--nshard "$NPROC" --shard "$k")
    python "$HERE/11_navsim_eval.py" --split "$SPLIT" --methods $METHODS ${SH[@]+"${SH[@]}"} ${EXTRA[@]+"${EXTRA[@]}"} \
        > "$LOGD/${TAGP}_s$k.log" 2>&1 &
    pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
for k in $(seq 0 $((NPROC - 1))); do echo "--- shard $k"; grep -E "^\[(done|skip|warn)\]" "$LOGD/${TAGP}_s$k.log" | tail -n 12 || true; done
[ "$fail" = 0 ] || { echo "[err] 실패한 shard 있음 -> 로그 확인: $LOGD/${TAGP}_s*.log"; exit 1; }

if [ "$ID" != 0 ] && [ -z "${LIMIT:-}" ] && [ "${KEEP_DB:-0}" != "1" ]; then
    rm -f "$CDIR"/*.db && rmdir "$CDIR" 2>/dev/null || true
    echo "[clean] $CDIR 삭제"
    python - "$NUPLAN_DATA_ROOT/chunks/$SPLIT/state.json" "$ID" <<'PY'
import json, sys
p, cid = sys.argv[1], int(sys.argv[2])
s = json.load(open(p))
for c in s["chunks"]:
    if c["id"] == cid:
        c["status"] = "done"
json.dump(s, open(p, "w"), indent=1)
PY
    echo "[done] $SPLIT $CID 완료. 다음: python 05_fetch_chunk.py --split $SPLIT"
fi
echo "[navsim] 요약: python $HERE/13_navsim_summary.py --split $SPLIT"
