#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
activate_env

SPLIT="${1:?split: val14 | test14-hard | test14-random}"
ID="${2:?chunk id (05_fetch_chunk.py 출력의 cNN 숫자)}"
CID=$(printf "c%02d" "$ID")
CDIR="$NUPLAN_DATA_ROOT/chunks/$SPLIT/$CID"
STATE="$NUPLAN_DATA_ROOT/chunks/$SPLIT/state.json"
[ -d "$CDIR" ] && ls "$CDIR"/*.db >/dev/null 2>&1 || { echo "[err] $CDIR 에 .db 없음 (05_fetch_chunk.py 먼저)"; exit 1; }

METHODS="${METHODS:-fd fdstar yflow yflow_star}"
CHALLENGES="${CHALLENGES:-nr r}"
STAGE="${YFLOW_STAGE:-corr}"
BASE_TAG="${SPLIT}-${CID}"
COMMON=("scenario_builder.data_root=$CDIR")
[ "${SIMLOG:-0}" = "1" ] || COMMON+=("~callback.simulation_log_callback")

run_tag() {
    case "$1" in yflow*) echo "${BASE_TAG}${CFG:+-$CFG}" ;; *) echo "$BASE_TAG" ;; esac
}
tag_full() {
    local m=$1 t
    case "$m" in
        fd|fdstar|fd_clip|fd_post) t="$m" ;;
        yflow_off) t="$m" ;;
        *)         t="$m-$STAGE" ;;
    esac
    case "$m" in fd|fdstar|fd_clip|fd_post) ;; *) [ "${YFLOW_V2:-0}" = "1" ] && t="$t-v2" ;; esac
    echo "$t-$(run_tag "$m")"
}

echo "[chunk] $SPLIT $CID: $(ls "$CDIR"/*.db | wc -l) db | methods=[$METHODS] ch=[$CHALLENGES] stage=$STAGE v2=${YFLOW_V2:-0} ${YFLOW_ARGS:-}"
for m in $METHODS; do
    for ch in $CHALLENGES; do
        tf=$(tag_full "$m")
        if ls "$NUPLAN_EXP_ROOT"/exp/simulation/*/flow_drive/"$SPLIT"/"$ch"/"$tf"/*/aggregator_metric/*.parquet >/dev/null 2>&1; then
            echo "[skip] $tf / $ch (이미 끝남)"; continue
        fi
        extra=()
        case "$m" in yflow*) [ -n "${YFLOW_ARGS:-}" ] && read -r -a extra <<< "$YFLOW_ARGS" ;; esac
        echo "[run ] $tf / $ch  ($(date +%H:%M))"
        TAG="$(run_tag "$m")" bash "$HERE/02_sim.sh" "$m" "$SPLIT" "$ch" "${COMMON[@]}" ${extra[@]+"${extra[@]}"}
        ls "$NUPLAN_EXP_ROOT"/exp/simulation/*/flow_drive/"$SPLIT"/"$ch"/"$tf"/*/aggregator_metric/*.parquet >/dev/null 2>&1 \
            || { echo "[err] $tf / $ch 결과 parquet 없음 -> 중단 (로그: $RESULTS_DIR/logs/)"; exit 1; }
    done
done

python "$HERE/07_merge_chunks.py" --split "$SPLIT" --chunk "$ID" --check_chunk
if [ "${KEEP_DB:-0}" != "1" ]; then
    rm -f "$CDIR"/*.db && rmdir "$CDIR" 2>/dev/null || true
    echo "[clean] $CDIR 삭제"
fi
python - "$STATE" "$ID" <<'PY'
import json, sys
p, cid = sys.argv[1], int(sys.argv[2])
s = json.load(open(p))
for c in s["chunks"]:
    if c["id"] == cid:
        c["status"] = "done"
json.dump(s, open(p, "w"), indent=1)
PY
echo "[done] $SPLIT $CID 완료. 다음: python 05_fetch_chunk.py --split $SPLIT"
