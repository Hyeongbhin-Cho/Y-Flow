#!/usr/bin/env bash
# Train then eval CLEVRER FlowMatch recognition (Exp-02-sub).
# Requires CLEVRER train+validation on disk:
#   python scripts/setup_clevrer.py --splits train validation
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

RUN_NAME="${RUN_NAME:-exp_02_sub_video_recognition}"
CONFIG="${CONFIG:-configs/exp_02_sub_video_recognition.yaml}"
PYTHON="${PYTHON:-python}"
COMMAND="${COMMAND:-flowmatch}"

echo "==> train ${COMMAND}  run_name=${RUN_NAME}"
"${PYTHON}" main.py "${COMMAND}" --mode train --run_name "${RUN_NAME}" --config "${CONFIG}" "$@"

echo "==> eval ${COMMAND}   run_name=${RUN_NAME}"
"${PYTHON}" main.py "${COMMAND}" --mode eval --run_name "${RUN_NAME}" --config "${CONFIG}" "$@"

echo "==> done"
echo "    backbone: runs/${RUN_NAME}/flowmatch/last.pt"
echo "    published: checkpoints/clevrer_flow/last.pt"
echo "    metrics:  runs/${RUN_NAME}/flowmatch/metrics.json"
