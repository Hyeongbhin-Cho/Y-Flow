#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_root="${repo_root}/experiments/exp_03_pedestrian"
python_bin="${PYTHON_BIN:-python}"
command="${COMMAND:-points}"

# Y-Flow point pipeline (data/pedestrian.py, same protocol as Exp-01):
#   COMMAND=points METHOD=all|flowmatch|hardflow|yflow|safeflow|uniconflow|guideflow ./run_exp_03_pedestrian.sh [--data.subset zara2]
if [[ "${command}" == "points" ]]; then
  method="${METHOD:-all}"
  run_name="${RUN_NAME:-exp_03_pedestrian}"
  config="${CONFIG:-configs/exp_03_pedestrian.yaml}"
  cd "${repo_root}"
  echo "==> train ${method}  run_name=${run_name}"
  "${python_bin}" main.py "${method}" --mode train --run_name "${run_name}" --config "${config}" "$@"
  echo "==> eval ${method}   run_name=${run_name}"
  exec "${python_bin}" main.py "${method}" --mode eval --run_name "${run_name}" --config "${config}" "$@"
fi

export PYTHONPATH="${task_root}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${task_root}"

case "${command}" in
  prepare)
    module="prepare_checkpoints"
    ;;
  audit)
    module="experiments.phase0_gt_audit.run_audit"
    ;;
  phase1)
    module="experiments.phase1_yflow.run_phase1"
    ;;
  train)
    module="fm_eth"
    ;;
  probe)
    module="experiments.phase1_yflow.probe_sensitivity"
    ;;
  bench)
    module="experiments.phase3_bench.run_benchmark"
    ;;
  bench_report)
    module="experiments.phase3_bench.report_bench"
    ;;
  train_cfm)
    module="experiments.phase2_cfm.train_cfm"
    ;;
  report)
    module="experiments.phase2_cfm.report_study"
    ;;
  study)
    exec bash experiments/phase2_cfm/run_study.sh "$@"
    ;;
  eval)
    module="eval_eth"
    ;;
  *)
    echo "Unknown COMMAND=${command}. Use points (Y-Flow pipeline), or the MoFlow task: prepare, audit, phase1, probe, train, train_cfm, study, report, eval." >&2
    exit 2
    ;;
esac

exec "${python_bin}" -m "${module}" "$@"
