#!/usr/bin/env bash

export WORK_ROOT="${WORK_ROOT:-/root}"
export CODE_DIR="$WORK_ROOT/code"
export CONDA_DIR="$WORK_ROOT/miniforge3"
export ENV_NAME="${ENV_NAME:-flowdrive}"

export NUPLAN_DEVKIT_COMMIT="e9241677997dd86bfc0bcd44817ab04fe631405b"
export TUPLAN_COMMIT="b51d5d04fac1bd4389653b9ab2ff73ea88f435a3"
export INTERPLAN_COMMIT="a85024a48935c92ec0d7c2a8cc2e583d01514469"
export FLOWDRIVE_COMMIT="06f941e3681a6a508b3e0e09ec35e2256ac26771"

export NUPLAN_DEVKIT_ROOT="$CODE_DIR/nuplan-devkit"
export TUPLAN_ROOT="$CODE_DIR/tuplan_garage"
export INTERPLAN_DEVKIT_ROOT="$CODE_DIR/interPlan"
export FD_ROOT="$CODE_DIR/flow_drive_planner"
export CKPT_PATH="${CKPT_PATH:-$FD_ROOT/flow_drive/checkpoint/flow_drive_model.pth}"

export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-$WORK_ROOT/nuplan/dataset}"
export NUPLAN_MAPS_ROOT="$NUPLAN_DATA_ROOT/maps"
export SPLITS_DIR="$NUPLAN_DATA_ROOT/nuplan-v1.1/splits"
export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-$WORK_ROOT/nuplan/exp}"
export RESULTS_DIR="${RESULTS_DIR:-$WORK_ROOT/fd_yflow_results}"

export HYDRA_FULL_ERROR=1

activate_env() {
    if [ ! -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
        echo "[err] conda가 없음. 먼저 bash 00_setup_env.sh" >&2; return 1
    fi
    source "$CONDA_DIR/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
}

ensure_sysdeps() {
    local need=()
    for c in wget unzip tmux git; do command -v $c >/dev/null || need+=($c); done
    ldconfig -p 2>/dev/null | grep -q libGL.so.1       || need+=(libgl1)
    ldconfig -p 2>/dev/null | grep -q libglib-2.0.so.0 || need+=(libglib2.0-0)
    ldconfig -p 2>/dev/null | grep -q libgeos_c        || true
    if [ ${#need[@]} -gt 0 ]; then
        echo "[sysdeps] installing: ${need[*]}"
        local SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
        $SUDO apt-get update -qq && DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq "${need[@]}"
    fi
}
