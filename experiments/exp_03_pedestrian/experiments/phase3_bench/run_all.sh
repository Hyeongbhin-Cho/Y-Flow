#!/usr/bin/env bash
# experiments/exp_03_pedestrian/experiments/phase3_bench/run_all.sh
# Full Exp-03 benchmark. Run from the repository root:
#   nohup bash experiments/exp_03_pedestrian/experiments/phase3_bench/run_all.sh > bench_all.log 2>&1 &
# Env: SUBSETS (default all five), CFM_SEEDS=3, MOFLOW_SEEDS=1, BATCH=1000, SKIP_MOFLOW=1 to skip the control rerun.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
SUBSETS="${SUBSETS:-eth hotel zara1 zara2 univ}"
CFM_SEEDS="${CFM_SEEDS:-3}"
MOFLOW_SEEDS="${MOFLOW_SEEDS:-1}"
BATCH="${BATCH:-1000}"
R=runs/exp_03_pedestrian
say() { echo "[$(date '+%F %T')] $*"; }
run() { local log="$1"; shift; say "RUN $* (log: ${log})"; "$@" > "${log}" 2>&1 && say "OK" || { say "FAIL (tail ${log})"; return 1; }; }

mkdir -p ${R}/logs
for s in ${SUBSETS}; do
  if [[ ! -f experiments/exp_03_pedestrian/checkpoints/cfm/${s}/model.pt ]]; then
    run ${R}/logs/train_cfm_${s}.log env COMMAND=train_cfm bash run_exp_03_pedestrian.sh --subset ${s} || continue
  else
    say "cfm ${s}: checkpoint exists, skip training"
  fi
  run ${R}/logs/probe_cfm_${s}.log env COMMAND=probe bash run_exp_03_pedestrian.sh --subset ${s} --model cfm \
      --out_dir ../../${R}/bench/probe/cfm_${s}
  grep -h "terminal step" ${R}/bench/probe/cfm_${s}/run.log | sed "s/^/  [G0 cfm ${s}] /"
  run ${R}/logs/bench_cfm_${s}.log env COMMAND=bench bash run_exp_03_pedestrian.sh --subset ${s} --model cfm \
      --seeds ${CFM_SEEDS} --batch_size ${BATCH} --save_raw
done
COMMAND=bench_report bash run_exp_03_pedestrian.sh --model cfm > /dev/null && say "report: ${R}/bench/cfm/BENCH_REPORT.md"

if [[ -z "${SKIP_MOFLOW:-}" ]]; then
  for s in ${SUBSETS}; do
    run ${R}/logs/bench_moflow_${s}.log env COMMAND=bench bash run_exp_03_pedestrian.sh --subset ${s} --model moflow \
        --seeds ${MOFLOW_SEEDS} --batch_size ${BATCH} --save_raw
  done
  COMMAND=bench_report bash run_exp_03_pedestrian.sh --model moflow > /dev/null && say "report: ${R}/bench/moflow/BENCH_REPORT.md"
fi
say "all done"
