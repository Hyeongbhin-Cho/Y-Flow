#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/exp_05_autonomous_driving_10k.yaml"
RUN_NAME="exp_05_av2_10k"

echo "==> build and validate 10k/2k cache"
python - <<'PY'
from data.base import build_dataset
from utils.config import load_config

cfg = load_config("configs/exp_05_autonomous_driving_10k.yaml")
bundle = build_dataset(cfg)
print("train:", len(bundle.train), bundle.train_raw.shape)
print("eval:", bundle.eval_raw.shape)
print("train context:", {k: v.shape for k, v in bundle.train_context.items()})
print("eval context:", {k: v.shape for k, v in bundle.eval_context.items()})
print("skipped:", bundle.meta.skipped_train, bundle.meta.skipped_eval)
if len(bundle.train) != int(cfg.data.n_train):
    raise RuntimeError(f"expected {cfg.data.n_train} train trajectories, got {len(bundle.train)}")
if len(bundle.eval_raw) != int(cfg.data.n_eval):
    raise RuntimeError(f"expected {cfg.data.n_eval} eval trajectories, got {len(bundle.eval_raw)}")
PY

echo "==> train MoFlow teacher"
python main.py moflow \
  --config "$CONFIG" \
  --mode train \
  --run_name "$RUN_NAME" \
  --device cuda

echo "==> seed-0 comparison"
python main.py all \
  --config "$CONFIG" \
  --mode eval \
  --run_name "$RUN_NAME" \
  --device cuda

echo "==> Exp-05 10k/2k seed-0 run complete"
