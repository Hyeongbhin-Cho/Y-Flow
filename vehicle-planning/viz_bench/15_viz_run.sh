#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPLIT="${1:?split}"; CH="${2:?nr|r}"; TOKENS="${3:?\"tok1 tok2 ...\"}"; METHODS="${4:?\"fd_post yflow_post:kin2 ...\"}"
TL=$(printf '"%s",' $TOKENS); TL="[${TL%,}]"
EXTRA=("scenario_filter.scenario_tokens=$TL")
[ -n "${DATA_ROOT:-}" ] && EXTRA+=("scenario_builder.data_root=$DATA_ROOT")
[ "$SPLIT" = mini ] && EXTRA+=(scenario_filter.num_scenarios_per_type=20) && export MINI_N=1000
for spec in $METHODS; do
    m="${spec%%:*}"; st="${spec#*:}"; [ "$st" = "$spec" ] && st=kin
    echo "[viz] $m (stage $st) / $SPLIT / $CH / tokens: $TOKENS"
    TAG=viz YFLOW_V2=1 YFLOW_STAGE="$st" bash "$HERE/02_sim.sh" "$m" "$SPLIT" "$CH" "${EXTRA[@]}"
done
echo "[viz] 끝. 결과 이름 예: yflow_post-kin2-v2-viz, fd_post-viz  -> 14_render_compare.py 로 그리기"
