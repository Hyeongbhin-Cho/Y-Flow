#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/exp_05_autonomous_driving.yaml"
BACKBONE="exp_05_moflow"
SEEDS=(0 1 2 3 4)
ITERS=(1 3 5 10 20)

run_eval() {
  local method="$1"
  local run_name="$2"
  local seed="$3"
  shift 3
  python main.py "$method" \
    --config "$CONFIG" \
    --mode eval \
    --run_name "$run_name" \
    --device cuda \
    --seed "$seed" \
    --moflow.backbone_run_name "$BACKBONE" \
    "$@"
}

echo "==> YFlow max_iter ablation"
for max_iter in "${ITERS[@]}"; do
  runs=()
  for seed in "${SEEDS[@]}"; do
    run_name="exp_05_yflow_iter_${max_iter}_seed_${seed}"
    runs+=("$run_name")
    run_eval yflow "$run_name" "$seed" --yflow.max_iter "$max_iter"
  done
  python -m eval.summarize_autonomous \
    --runs "${runs[@]}" \
    --output "runs/exp_05_yflow_iter_${max_iter}_summary.json"
done

echo "==> terminal projection ablation"
for method in safeflow uniconflow yflow; do
  runs=()
  for seed in "${SEEDS[@]}"; do
    run_name="exp_05_${method}_no_terminal_seed_${seed}"
    runs+=("$run_name")
    case "$method" in
      safeflow)
        run_eval "$method" "$run_name" "$seed" --safeflow.terminal_filter.enabled false
        ;;
      uniconflow)
        run_eval "$method" "$run_name" "$seed" --uniconflow.terminal_refinement false
        ;;
      yflow)
        run_eval "$method" "$run_name" "$seed" --yflow.terminal_refinement false
        ;;
    esac
  done
  python -m eval.summarize_autonomous \
    --runs "${runs[@]}" \
    --output "runs/exp_05_${method}_no_terminal_summary.json"
done

echo "==> Exp-05 ablations complete"
