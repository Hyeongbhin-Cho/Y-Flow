#!/usr/bin/env bash
# experiments/exp_03_pedestrian/experiments/phase2_cfm/run_study.sh
# Exp-03 main study. Run from experiments/exp_03_pedestrian (COMMAND=study does this).
#   STAGES="train probe run report" (default: all)   SUBSETS="eth hotel univ zara1 zara2"
#   CFM_SEEDS=5  MOFLOW_SEEDS=1  PY=python
set -uo pipefail

PY="${PY:-python}"
STAGES="${STAGES:-train probe run report}"
SUBSETS="${SUBSETS:-eth hotel univ zara1 zara2}"
CFM_SEEDS="${CFM_SEEDS:-5}"
MOFLOW_SEEDS="${MOFLOW_SEEDS:-1}"
ROOT="../../runs/exp_03_pedestrian/study"
mkdir -p "${ROOT}"
LOG="${ROOT}/study.log"
METHODS_CFM="PLAIN_FM FINAL_PROJECTION YFLOW YFLOW_NO_REPLACE POV_ALWAYS"
METHODS_MOFLOW="PLAIN_FM FINAL_PROJECTION YFLOW YFLOW_NO_REPLACE"

# setting name | extra phase1 arguments
SETTINGS=(
  "kin_data|--constraint kin"
  "kin_v12|--constraint kin --v_max 1.2"
  "kin_v10|--constraint kin --v_max 1.0"
  "colcv_r020|--constraint kin_col --neighbour_source cv --r_safe 0.2"
  "col_r020|--constraint kin_col --r_safe 0.2"
  "col_r035|--constraint kin_col --r_safe 0.35"
)

say() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }
run() { say "RUN $*"; "$@" >> "${LOG}" 2>&1 || { say "FAIL $*"; return 1; }; }

has() { [[ " ${STAGES} " == *" $1 "* ]]; }

if has train; then
  for s in ${SUBSETS}; do
    if [[ -f checkpoints/cfm/${s}/model.pt && -z "${RETRAIN:-}" ]]; then say "skip train ${s} (exists)"; continue; fi
    run ${PY} -m experiments.phase2_cfm.train_cfm --subset "${s}"
  done
fi

if has probe; then
  for s in ${SUBSETS}; do
    run ${PY} -m experiments.phase1_yflow.probe_sensitivity --subset "${s}" --model cfm --out_dir "${ROOT}/probe/cfm/${s}"
    run ${PY} -m experiments.phase1_yflow.probe_sensitivity --subset "${s}" --model moflow --out_dir "${ROOT}/probe/moflow/${s}"
  done
fi

if has run; then
  for entry in "${SETTINGS[@]}"; do
    name="${entry%%|*}"; extra="${entry#*|}"
    for s in ${SUBSETS}; do
      run ${PY} -m experiments.phase1_yflow.run_phase1 --subset "${s}" --model cfm --seeds "${CFM_SEEDS}" \
        --methods ${METHODS_CFM} ${extra} --save_raw --out_dir "${ROOT}/cfm/${name}/${s}"
      run ${PY} -m experiments.phase1_yflow.run_phase1 --subset "${s}" --model moflow --seeds "${MOFLOW_SEEDS}" \
        --methods ${METHODS_MOFLOW} ${extra} --save_raw --out_dir "${ROOT}/moflow/${name}/${s}"
    done
  done
fi

if has report; then
  run ${PY} -m experiments.phase2_cfm.report_study --root "${ROOT}"
fi
say "done: ${STAGES}"
