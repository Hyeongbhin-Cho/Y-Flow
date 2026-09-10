#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_root="${repo_root}/experiments/exp_04_robot_arm"
python_bin="${PYTHON_BIN:-python}"
command="${COMMAND:-ood}"

export PYTHONPATH="${task_root}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${task_root}"

case "${command}" in
  phase1)
    module="experiments.pov_projection.run_phase1"
    ;;
  phase2)
    module="experiments.pov_projection_phase2.run_phase2"
    ;;
  phase3)
    module="experiments.pov_projection_phase3.run_phase3"
    ;;
  phase15)
    module="experiments.pov_projection_phase15_yflow_main.run_phase15"
    ;;
  hybrid)
    module="experiments.pov_yflow_hybrid.run_hybrid"
    ;;
  factorial)
    module="experiments.pov_factorial_ablation.run_factorial"
    ;;
  ood)
    module="experiments.ood_no_replace.run_ood"
    ;;
  *)
    echo "Unknown COMMAND=${command}. Use phase1, phase2, phase3, phase15, hybrid, factorial, or ood." >&2
    exit 2
    ;;
esac

exec "${python_bin}" -m "${module}" "$@"
