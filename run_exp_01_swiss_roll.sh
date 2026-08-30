#!/usr/bin/env bash
# Train FlowMatch on Exp-01 Swiss roll, then run eval.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

RUN_NAME="${RUN_NAME:-exp_01_swiss_roll}"
CONFIG="${CONFIG:-configs/exp_01_swiss_roll.yaml}"
PYTHON="${PYTHON:-python}"

echo "==> train flowmatch  run_name=${RUN_NAME}"
"${PYTHON}" main.py flowmatch --mode train --run_name "${RUN_NAME}" --config "${CONFIG}" "$@"

echo "==> eval flowmatch   run_name=${RUN_NAME}"
"${PYTHON}" main.py flowmatch --mode eval --run_name "${RUN_NAME}" --config "${CONFIG}" "$@"

echo "==> done"
echo "    ckpt:    runs/${RUN_NAME}/flowmatch/last.pt"
echo "    metrics: runs/${RUN_NAME}/flowmatch/metrics.json"
echo "    summary: runs/${RUN_NAME}/metrics.json"
