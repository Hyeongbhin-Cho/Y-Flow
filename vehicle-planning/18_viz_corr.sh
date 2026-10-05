#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"; activate_env
SPLIT="$1"; TOKEN="$2"; CH="${CH:-nr}"
OUT="$RESULTS_DIR/viz/$SPLIT"; mkdir -p "$OUT"
DB="$NUPLAN_DATA_ROOT/chunks/$SPLIT/viz"
python "$HERE/16_fetch_logs.py" --split "$SPLIT" --tokens "$TOKEN"
if ! find "$NUPLAN_EXP_ROOT" -path "*/flow_drive/$SPLIT/$CH/yflow-corr-v2-viz/*" -name "$TOKEN.msgpack.xz" | grep -q .; then
  TAG=viz YFLOW_V2=1 YFLOW_STAGE=corr bash "$HERE/02_sim.sh" yflow "$SPLIT" "$CH" \
    "scenario_builder.data_root=$DB" "scenario_filter.scenario_tokens=[\"$TOKEN\"]"
fi
R="$HERE/14_render_compare.py"
python "$R" --split "$SPLIT" --ch "$CH" --token "$TOKEN" --tags yflow-corr-v2-viz --labels "Y-Flow (corr)" \
  --out "$OUT/${SPLIT}_${TOKEN}_yflow_corr.gif"
python "$R" --split "$SPLIT" --ch "$CH" --token "$TOKEN" --tags fd-viz yflow-corr-v2-viz \
  --labels "FlowDrive" "Y-Flow (corr)" --out "$OUT/${SPLIT}_${TOKEN}_fd_vs_corr.gif"
[ "${KEEP_DB:-0}" = "1" ] || rm -f "$DB"/*.db
ls -1 "$OUT" | grep "$TOKEN" | grep corr
