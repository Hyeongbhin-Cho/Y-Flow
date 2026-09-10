# POV endpoint-projection proof of concept

This module keeps the upstream SafeFlowMPC implementation unchanged and compares
four samplers with the unsafe pretrained Flow Matching checkpoint and matched
initial noise:

- `PLAIN_FM`
- `FINAL_PROJECTION`
- `CURRENT_STATE_PROJECTION`
- `POV_ENDPOINT_PROJECTION`

The primary POV update is exactly
`x1_pred = x_t + (1-t) v_theta`,
`v_pov = P(x1_pred) - x_t`, and
`x_t <- x_t + (1/7) v_pov`. There is no division by `1-t`.

The projector wraps the upstream Acados OCP. It minimizes weighted joint-space
distance to the candidate with the existing small smoothness weights, dynamics,
joint/velocity/acceleration/jerk bounds and local convex collision corridor. The
corridor and soft collision constraints make it a local projection surrogate,
not an exact global Euclidean projection. The upstream terminal option enforces
terminal derivative behavior rather than a hard Cartesian goal equality.
The upstream implementation uses one convex corridor for all stages. The POC
instead reuses its `ConvexSetFinder` constraints per trajectory segment and sends
those as stage-specific Acados parameters. Because the solver is configured for
one SQP-RTI iteration, the wrapper reuses its returned iterate and relinearizes up
to 12 times. Exhausted iterations remain explicit solver failures in the CSV.
The last finite iterate is retained as the failed planner output so its safety and
quality can be measured, but it is never relabeled as a successful projection.
Acados state is reset at the beginning of every `P(...)` call so method ordering
cannot leak solver warm starts across candidates; RTI iterates are reused only
inside that single projection call.

Run from the repository root after building the upstream Acados solver:

```bash
export ACADOS_SOURCE_DIR=/workspace/acados-v0.5.1
export LD_LIBRARY_PATH=/workspace/yflow-robot-arm/.venv/lib/python3.12/site-packages/cmeel.prefix/lib:/workspace/acados-v0.5.1/lib:$LD_LIBRARY_PATH
python -m experiments.pov_projection.run_phase1 --seeds 32
```

Use `--tasks 1 --seeds 1` for a smoke test. Use `--build-solver` only when the
matching `/tmp/acados_code` has not already been generated.
