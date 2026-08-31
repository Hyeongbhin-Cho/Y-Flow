#!/usr/bin/env bash
# Train (or reuse FlowMatch ckpt) on Exp-01 Swiss roll, then eval.
# COMMAND: all | flowmatch | hardflow | yflow | safeflow | uniconflow | guideflow
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

RUN_NAME="${RUN_NAME:-exp_01_swiss_roll}"
CONFIG="${CONFIG:-configs/exp_01_swiss_roll.yaml}"
PYTHON="${PYTHON:-python}"
COMMAND="${COMMAND:-yflow}"

echo "==> train ${COMMAND}  run_name=${RUN_NAME}"
"${PYTHON}" main.py "${COMMAND}" --mode train --run_name "${RUN_NAME}" --config "${CONFIG}" "$@"

echo "==> eval ${COMMAND}   run_name=${RUN_NAME}"
"${PYTHON}" main.py "${COMMAND}" --mode eval --run_name "${RUN_NAME}" --config "${CONFIG}" "$@"

echo "==> done"
echo "    backbone: runs/${RUN_NAME}/flowmatch/last.pt"
echo "    metrics:  runs/${RUN_NAME}/${COMMAND}/metrics.json"
echo "    summary:  runs/${RUN_NAME}/metrics.json"
