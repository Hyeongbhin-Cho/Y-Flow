#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
activate_env

SPLIT="${1:?split: val14 | test14-hard}"; TOKEN="${2:-}"
CH="${CH:-nr}"; YST="${YSTAGE:-kin}"
M=(fd yflow fdstar yflow_star)
REF=(fd "yflow-$YST-v2" fdstar "yflow_star-$YST-v2")
VIZ=(fd-viz "yflow-$YST-v2-viz" fdstar-viz "yflow_star-$YST-v2-viz")
LAB=("FlowDrive" "Y-Flow ($YST)" "FlowDrive*" "Y-Flow* ($YST)")
OUT="$RESULTS_DIR/viz/$SPLIT"; mkdir -p "$OUT"
R="$HERE/14_render_compare.py"

if [ -z "$TOKEN" ]; then
    python "$R" --list --split "$SPLIT" --ch "$CH" --top 8 --tags "${REF[@]}" | tee "$OUT/list_$CH.txt"
    TOKEN=$(awk '/^PICK /{print $2}' "$OUT/list_$CH.txt")
    [ -n "$TOKEN" ] || { echo "[err] 후보 없음"; exit 1; }
fi
echo "[viz] $SPLIT / $CH / token $TOKEN"

python "$HERE/16_fetch_logs.py" --split "$SPLIT" --tokens "$TOKEN"
DB_DIR="$NUPLAN_DATA_ROOT/chunks/$SPLIT/viz"

for i in 0 1 2 3; do
    if find "$NUPLAN_EXP_ROOT" -path "*/flow_drive/$SPLIT/$CH/${VIZ[$i]}/*" -name "$TOKEN.msgpack.xz" 2>/dev/null | grep -q .; then
        echo "[skip] ${VIZ[$i]} (기록 있음)"; continue
    fi
    TAG=viz YFLOW_V2=1 YFLOW_STAGE="$YST" bash "$HERE/02_sim.sh" "${M[$i]}" "$SPLIT" "$CH" \
        "scenario_builder.data_root=$DB_DIR" "scenario_filter.scenario_tokens=[\"$TOKEN\"]"
done

for i in 0 1 2 3; do
    python "$R" --split "$SPLIT" --ch "$CH" --token "$TOKEN" --tags "${VIZ[$i]}" --labels "${LAB[$i]}" \
        --out "$OUT/${SPLIT}_${TOKEN}_${M[$i]}.gif"
done
python "$R" --split "$SPLIT" --ch "$CH" --token "$TOKEN" --tags "${VIZ[@]}" --labels "${LAB[@]}" \
    --out "$OUT/${SPLIT}_${TOKEN}_2x2.gif"

[ "${KEEP_DB:-0}" = "1" ] || { rm -f "$DB_DIR"/*.db; echo "[clean] $DB_DIR/*.db 삭제"; }
echo "[done] $OUT"; ls -1 "$OUT" | grep "$TOKEN"
