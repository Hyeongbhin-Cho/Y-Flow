#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
ensure_sysdeps

if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    wget -q -O /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash /tmp/miniforge.sh -b -p "$CONDA_DIR"; rm -f /tmp/miniforge.sh
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" || conda create -y -n "$ENV_NAME" python=3.9
conda activate "$ENV_NAME"

mkdir -p "$CODE_DIR"
clone_at() {
    [ -d "$2/.git" ] || git clone -q "$1" "$2"
    git -C "$2" fetch -q origin "$3" 2>/dev/null || true
    git -C "$2" checkout -q "$3"
}
clone_at https://github.com/motional/nuplan-devkit.git          "$NUPLAN_DEVKIT_ROOT"    "$NUPLAN_DEVKIT_COMMIT"
clone_at https://github.com/autonomousvision/tuplan_garage.git  "$TUPLAN_ROOT"           "$TUPLAN_COMMIT"
clone_at https://github.com/einsteinguang/flow_drive_planner.git "$FD_ROOT"              "$FLOWDRIVE_COMMIT"
[ "${INSTALL_INTERPLAN:-1}" = "1" ] && clone_at https://github.com/mh0797/interPlan.git "$INTERPLAN_DEVKIT_ROOT" "$INTERPLAN_COMMIT"

pip install -q -U "pip<24.1"

( cd "$NUPLAN_DEVKIT_ROOT" && pip install -q -e . && pip install -q -r requirements.txt )

( cd "$TUPLAN_ROOT" && pip install -q -e . --no-deps && pip install -q scikit-learn==1.2.2 positional-encodings==6.0.1 )
cp "$FD_ROOT/assets/adapted_tuplan_code/pdm_object_manager.py" \
   "$TUPLAN_ROOT/tuplan_garage/planning/simulation/planner/pdm_planner/observation/pdm_object_manager.py"

if [ "${INSTALL_INTERPLAN:-1}" = "1" ]; then
    ( cd "$INTERPLAN_DEVKIT_ROOT" && pip install -q -e . --no-deps )
    sed -i 's|${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1/trainval|${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/test|' \
        "$INTERPLAN_DEVKIT_ROOT/interplan/planning/script/config/common/scenario_builder/interplan.yaml" || true
fi

if [ -n "${TORCH_INDEX_URL:-}" ]; then
    pip install -q "torch>=2.8,<2.9" "torchvision>=0.23,<0.24" --index-url "$TORCH_INDEX_URL"
fi
( cd "$FD_ROOT" && pip install -q -e . )

cp -r "$HERE/overlay/." "$FD_ROOT/"

mkdir -p "$SPLITS_DIR"/{mini,trainval,test}
ln -sfn "$SPLITS_DIR/trainval" "$NUPLAN_DATA_ROOT/nuplan-v1.1/trainval"
ln -sfn "$SPLITS_DIR/test"     "$NUPLAN_DATA_ROOT/nuplan-v1.1/test"

python - <<'PY'
import torch, diffusers, nuplan, tuplan_garage, flow_drive
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| gpus", torch.cuda.device_count())
from flow_drive.planner.yflow_planner import FlowDriveYFlowPlannerWrapper
print("imports OK")
PY
( cd "$FD_ROOT" && python tests/test_yflow_equivalence.py && python tests/test_yflow_constraints.py )
echo "[done] 다음: bash 01_download_nuplan.sh"
