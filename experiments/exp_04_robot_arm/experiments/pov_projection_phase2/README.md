# POV robot-arm Phase 2

This directory is isolated from the preserved Phase-1 implementation and
artifacts. It evaluates a flow-preserving endpoint correction using the same
unsafe Flow Matching checkpoint, six ID tasks, time grid, and matched seeds.

The primary update is:

```text
x1_pred = x_t + (1-t) * v_t
delta = P_goal(x1_pred) - x1_pred
v_guided = v_t + lambda_pov * delta / max(1-t, 1e-3)
```

An unsuccessful Acados projection discards the unconverged iterate and uses the
vanilla velocity for that step. The full benchmark refuses to start unless the
mandatory correctness tests in `results/correctness_tests.json` pass.

Run on the configured RTX 3090 host:

```bash
export ACADOS_SOURCE_DIR=/workspace/acados-v0.5.1
export LD_LIBRARY_PATH=/workspace/Y-Flow/experiments/exp_04_robot_arm/.venv/lib/python3.12/site-packages/cmeel.prefix/lib:/workspace/acados-v0.5.1/lib:$LD_LIBRARY_PATH
python -m experiments.pov_projection_phase2.run_phase2 --build-solver --correctness-only
python -m experiments.pov_projection_phase2.run_phase2 --skip-correctness
```
