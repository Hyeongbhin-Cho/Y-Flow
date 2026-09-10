# Reproduction environment

The original verified server checkout was `/workspace/yflow-robot-arm` on branch
`feature/robot-arm`. In the numbered Y-Flow layout, run from
`/workspace/Y-Flow/experiments/exp_04_robot_arm`. Both are based on SafeFlowMPC commit
`3efe4d9522f4112b291a868866e7e4934697a261`.

The repository's Python requirements need two additional system/runtime pins on
the Ubuntu 24.04 / Python 3.12 server:

- `libcdd-dev` to build `pycddlib==3.0.2`
- `cmeel-urdfdom==4.0.1` and `cmeel-tinyxml2==10.0.0`, because Pinocchio 3.8.0
  was built against those SONAMEs while their unbounded transitive dependencies
  now resolve to incompatible major versions

Acados is pinned to `v0.5.1` (`48e223e85f0408ebfd1d8c6d6fb0589e9c41b3aa`).
The current Acados main branch changed the `AcadosOcpSolver.generate` API used by
SafeFlowMPC and does not reproduce the unmodified script.

```bash
apt-get update
apt-get install -y libcdd-dev xvfb

cd /workspace/Y-Flow/experiments/exp_04_robot_arm
uv venv --python /venv/main/bin/python .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python \
  'cmeel-urdfdom==4.0.1' 'cmeel-tinyxml2==10.0.0'

git clone --branch v0.5.1 --depth 1 --recursive \
  https://github.com/acados/acados.git /workspace/acados-v0.5.1
cmake -S /workspace/acados-v0.5.1 -B /workspace/acados-v0.5.1/build \
  -DACADOS_WITH_QPOASES=ON
cmake --build /workspace/acados-v0.5.1/build -j8 --target install
uv pip install --python .venv/bin/python -e \
  /workspace/acados-v0.5.1/interfaces/acados_template
uv pip install --python .venv/bin/python -r requirements.txt
```

Every run uses:

```bash
export ACADOS_SOURCE_DIR=/workspace/acados-v0.5.1
export LD_LIBRARY_PATH=/workspace/Y-Flow/experiments/exp_04_robot_arm/.venv/lib/python3.12/site-packages/cmeel.prefix/lib:/workspace/acados-v0.5.1/lib:$LD_LIBRARY_PATH
```

The unmodified upstream example was reproduced under `xvfb-run` after building
its `/tmp/acados_code` solver once. It reached the sampled goal and exited 0.
The full controlled run command is:

```bash
python -m experiments.pov_projection.run_phase1 \
  --tasks 6 --seeds 32 --bootstrap-samples 2000 \
  --output /workspace/Y-Flow/runs/exp_04_robot_arm/pov_projection/phase1
```

Task-disjoint shards can be run concurrently with `--task-ids 0,1` (and the
matching `2,3` / `4,5` shards). Merge them with:

```bash
python -m experiments.pov_projection.merge_phase1 \
  --inputs artifacts/shard01 artifacts/shard23 artifacts/shard45 \
  --output /workspace/Y-Flow/runs/exp_04_robot_arm/pov_projection/phase1
```
