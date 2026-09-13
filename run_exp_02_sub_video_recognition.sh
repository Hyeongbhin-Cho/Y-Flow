#!/usr/bin/env bash
# Prepare ImageNet ResNet-34, train the CLEVRER recognizer, evaluate it, and run focused tests.
# Set SETUP_DATA=1 to download and extract the full CLEVRER train + validation splits first.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

RUN_NAME="${RUN_NAME:-exp_02_sub_video_recognition}"
CONFIG="${CONFIG:-configs/exp_02_sub_video_recognition.yaml}"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-datasets/clevrer}"
RESNET_WEIGHTS="${RESNET_WEIGHTS:-checkpoints/resnet34_imagenet1k_v1.pth}"
SETUP_DATA="${SETUP_DATA:-0}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"

if [[ "$SETUP_DATA" != "0" && "$SETUP_DATA" != "1" ]]; then
  echo "SETUP_DATA must be 0 or 1 (got: $SETUP_DATA)" >&2
  exit 2
fi
if [[ "$RUN_UNIT_TESTS" != "0" && "$RUN_UNIT_TESTS" != "1" ]]; then
  echo "RUN_UNIT_TESTS must be 0 or 1 (got: $RUN_UNIT_TESTS)" >&2
  exit 2
fi

echo "==> prepare ImageNet ResNet-34 weights"
"${PYTHON}" scripts/setup_resnet34.py --output "$RESNET_WEIGHTS"

if [[ "$SETUP_DATA" == "1" ]]; then
  echo "==> download and prepare CLEVRER train + validation splits"
  "${PYTHON}" scripts/setup_clevrer.py --root "$DATA_ROOT" --splits train validation
fi

for split in train validation; do
  if [[ ! -f "$DATA_ROOT/manifests/$split.json" ]]; then
    echo "Missing CLEVRER $split manifest: $DATA_ROOT/manifests/$split.json" >&2
    echo "Prepare data with: ${PYTHON} scripts/setup_clevrer.py --root $DATA_ROOT --splits train validation" >&2
    echo "Or run this script with SETUP_DATA=1 to download both splits." >&2
    exit 2
  fi
done

echo "==> train ResNet-34 CLEVRER recognizer  run_name=$RUN_NAME"
"${PYTHON}" main.py recognition --mode train --run_name "$RUN_NAME" --config "$CONFIG" \
  --data.cache_dir "$DATA_ROOT" --model.weights_path "$RESNET_WEIGHTS" "$@"

echo "==> evaluate on the configured CLEVRER validation subset  run_name=$RUN_NAME"
"${PYTHON}" main.py recognition --mode eval --run_name "$RUN_NAME" --config "$CONFIG" \
  --data.cache_dir "$DATA_ROOT" --model.weights_path "$RESNET_WEIGHTS" "$@"

if [[ "$RUN_UNIT_TESTS" == "1" ]]; then
  echo "==> run recognition, data-state, and CLI tests"
  "${PYTHON}" -m unittest \
    test.test_clevrer_recognition_model \
    test.test_clevrer_recognition_loss \
    test.test_clevrer_state \
    test.test_cli \
    -v
fi

echo "==> completed"
echo "    best checkpoint: runs/$RUN_NAME/recognition/best.pt"
echo "    last checkpoint: runs/$RUN_NAME/recognition/last.pt"
echo "    published model: checkpoints/clevrer_recognition/last.pt"
echo "    eval metrics:    runs/$RUN_NAME/recognition/metrics.json"
