#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
ensure_sysdeps
activate_env

METHOD="${1:?method: fd|fdstar|yflow|yflow_off|yflow_star|fd_clip|fd_post|yflow_post}"
SPLIT="${2:?split: mini|val14|test14-random|test14-hard|interplan}"
CH="${3:?challenge: nr|r}"
shift 3
EXTRA=("$@")

case "$CH" in
    nr) CHALLENGE=closed_loop_nonreactive_agents ;;
    r)  CHALLENGE=closed_loop_reactive_agents ;;
    *)  echo "[err] challenge는 nr 또는 r"; exit 1 ;;
esac
STAGE="${YFLOW_STAGE:-corr}"
THREADS="${THREADS:-$(nproc)}"
GPU_FRAC="${GPU_FRAC:-0.1}"

YP=planner.flow_drive_yflow
case "$METHOD" in
    fd)         PLANNER=flow_drive;       PO=(planner.flow_drive.post_mode=0) ;;
    fdstar)     PLANNER=flow_drive;       PO=(planner.flow_drive.post_mode=1) ;;
    yflow)      PLANNER=flow_drive_yflow; PO=($YP.post_mode=2) ;;
    yflow_off)  PLANNER=flow_drive_yflow; PO=($YP.post_mode=2 $YP.yflow.enabled=false) ;;
    yflow_star) PLANNER=flow_drive_yflow; PO=($YP.post_mode=3) ;;
    fd_clip)    PLANNER=flow_drive_yflow; PO=($YP.post_mode=4) ;;
    fd_post)    PLANNER=flow_drive_yflow; PO=($YP.post_mode=5) ;;
    yflow_post) PLANNER=flow_drive_yflow; PO=($YP.post_mode=6) ;;
    *) echo "[err] unknown method $METHOD"; exit 1 ;;
esac

TAG_FULL="${METHOD}"
if [ "$PLANNER" = flow_drive ]; then
    PO+=(planner.flow_drive.ckpt_path="$CKPT_PATH"
         planner.flow_drive.mlflow_exp_name=None planner.flow_drive.load_run_name=None
         planner.flow_drive.load_epoch=0 planner.flow_drive.render=false planner.flow_drive.video_dir=None)
else
    case "$STAGE" in
        kin)  PO+=($YP.yflow.use_corridor=false $YP.yflow.use_obstacles=false) ;;
        corr) PO+=($YP.yflow.use_corridor=true  $YP.yflow.use_obstacles=false) ;;
        full) PO+=($YP.yflow.use_corridor=true  $YP.yflow.use_obstacles=true) ;;
        kin2) PO+=($YP.yflow.use_corridor=false $YP.yflow.use_obstacles=false $YP.yflow.accel_mode=lonlat) ;;
        corr2) PO+=($YP.yflow.use_corridor=true $YP.yflow.use_obstacles=false $YP.yflow.accel_mode=lonlat) ;;
        *) echo "[err] YFLOW_STAGE=kin|corr|full|kin2|corr2"; exit 1 ;;
    esac
    case "$METHOD" in yflow_off|fd_clip|fd_post) ;; *) TAG_FULL="${METHOD}-${STAGE}" ;; esac
    if [ "${YFLOW_V2:-0}" = "1" ] && [ "$METHOD" != fd_clip ] && [ "$METHOD" != fd_post ]; then
        PO+=($YP.yflow.fallback_resample=true $YP.yflow.corridor_precheck=true)
        TAG_FULL="${TAG_FULL}-v2"
    fi
    PO+=($YP.ckpt_path="$CKPT_PATH")
fi
[ -n "${TAG:-}" ] && TAG_FULL="${TAG_FULL}-${TAG}"
STAMP=$(date +%Y%m%d-%H%M%S)
UID_="flow_drive/${SPLIT}/${CH}/${TAG_FULL}/${STAMP}"
if [ "$PLANNER" = flow_drive_yflow ]; then
    PO+=($YP.stats_dir="$RESULTS_DIR/yflow_stats/$(echo "$UID_" | tr '/' '_')")
fi
mkdir -p "$NUPLAN_EXP_ROOT" "$RESULTS_DIR/logs"
LOG="$RESULTS_DIR/logs/$(echo "$UID_" | tr '/' '_').log"
echo "[sim] $METHOD / $SPLIT / $CHALLENGE / stage=$STAGE -> $LOG"

COMMON=(
    planner="$PLANNER" "${PO[@]}"
    verbose=false
    worker=ray_distributed
    worker.threads_per_node="$THREADS"
    distributed_mode='SINGLE_NODE'
    number_of_gpus_allocated_per_simulation="$GPU_FRAC"
    enable_simulation_progress_bar=true
)

if [ "$SPLIT" = interplan ]; then
    [ "$CH" = r ] || { echo "[err] interplan은 reactive(r)만 (upstream 설정과 동일)"; exit 1; }
    export NUPLAN_EXP_ROOT="$NUPLAN_EXP_ROOT/interplan_exp"; mkdir -p "$NUPLAN_EXP_ROOT"
    python "$INTERPLAN_DEVKIT_ROOT/interplan/planning/script/run_simulation.py" \
        +simulation=default_interplan_benchmark \
        scenario_filter=interplan10 \
        experiment_name="$(echo "$UID_" | tr '/' '_')" \
        "${COMMON[@]}" \
        hydra.searchpath="[pkg://flow_drive.config.scenario_filter, pkg://flow_drive.config, pkg://interplan.planning.script.config.common, pkg://interplan.planning.script.config.simulation, pkg://interplan.planning.script.experiments, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.config.simulation, pkg://nuplan.planning.script.experiments]" \
        ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee "$LOG"
else
    if [ "$SPLIT" = mini ]; then
        SC=(scenario_builder=nuplan_mini scenario_filter=one_of_each_scenario_type
            scenario_filter.limit_total_scenarios="${MINI_N:-10}")
    elif [ "$SPLIT" = val14 ]; then
        SC=(scenario_builder=nuplan scenario_filter=val14)
    else
        SC=(scenario_builder=nuplan_challenge scenario_filter="$SPLIT")
    fi
    python "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
        +simulation="$CHALLENGE" \
        "${SC[@]}" \
        experiment_uid="$UID_" \
        "${COMMON[@]}" \
        hydra.searchpath="[pkg://flow_drive.config.scenario_filter, pkg://flow_drive.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \
        ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee "$LOG"
fi
echo "[sim] done. 요약: python $HERE/03_summarize.py"
