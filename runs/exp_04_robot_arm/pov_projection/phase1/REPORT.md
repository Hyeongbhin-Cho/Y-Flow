# POV endpoint-projection Phase-1 report

## Decision

**NO-GO** under the predeclared first gate.

- Lower collision rate than plain FM: `False`
- Within 2 percentage points of the safer projection baseline: `False`
- No more distortion than both projection baselines: `False`
- Mean one-shot planning latency at most 1 second: `False`
- Projection solver failures (not hidden): `1563`

## Scope

This is the controlled one-shot ID experiment: six supplied tasks, matched seeds,
one 16-knot sample per task/seed, the pretrained unsafe Flow Matching checkpoint,
and no retraining. It is not a full closed-loop MPC success-rate reproduction.

The projection reuses SafeFlowMPC's Acados OCP. The objective is weighted squared
joint-trajectory distance plus the upstream small smoothness regularizer. Collision
constraints use a candidate-dependent local convex corridor and soft nonlinear
constraints; consequently this is a local weighted projection surrogate rather
than an exact global Euclidean projection. The upstream terminal option constrains
terminal derivatives, not a hard Cartesian goal equality.
When all 12 RTI relinearizations fail, the last finite solver iterate is retained
as that method's planning output and the solve remains marked failed; it is never
counted as a successful projection.

## Summary

| method | metric | mean | std | median | ci95_low | ci95_high | numerator | denominator |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PLAIN_FM | collision | 0.151042 | 0.359026 | 0 | 0.0989583 | 0.203125 | 29 | 192 |
| PLAIN_FM | success | 0 | 0 | 0 | 0 | 0 | 0 | 192 |
| PLAIN_FM | min_clearance | 0.0660799 | 0.0464402 | 0.0722477 | 0.0592817 | 0.0725584 |  |  |
| PLAIN_FM | terminal_goal_error | 0.581712 | 0.29237 | 0.456701 | 0.541122 | 0.622538 |  |  |
| PLAIN_FM | trajectory_distortion_vs_plain | 0 | 0 | 0 | 0 | 0 |  |  |
| PLAIN_FM | planning_latency_sec | 0.0713058 | 0.00169557 | 0.0710638 | 0.0711122 | 0.0715876 |  |  |
| PLAIN_FM | projection_solver_latency_sec | 0 | 0 | 0 | 0 | 0 |  |  |
| PLAIN_FM | failed_projection_solves | 0 | 0 | 0 | 0 | 0 |  |  |
| FINAL_PROJECTION | collision | 0.03125 | 0.174448 | 0 | 0.0104167 | 0.0572917 | 6 | 192 |
| FINAL_PROJECTION | success | 0 | 0 | 0 | 0 | 0 | 0 | 192 |
| FINAL_PROJECTION | min_clearance | 0.167679 | 0.131801 | 0.171512 | 0.149201 | 0.186641 |  |  |
| FINAL_PROJECTION | terminal_goal_error | 0.946302 | 0.352992 | 0.986234 | 0.895663 | 0.997179 |  |  |
| FINAL_PROJECTION | trajectory_distortion_vs_plain | 3.17443 | 1.34265 | 3.13119 | 2.99725 | 3.36545 |  |  |
| FINAL_PROJECTION | planning_latency_sec | 0.386467 | 0.169617 | 0.356426 | 0.363517 | 0.410272 |  |  |
| FINAL_PROJECTION | projection_solver_latency_sec | 0.0266111 | 0.0178455 | 0.0245388 | 0.0240992 | 0.0291788 |  |  |
| FINAL_PROJECTION | failed_projection_solves | 0.015625 | 0.124344 | 0 | 0 | 0.0364583 |  |  |
| CURRENT_STATE_PROJECTION | collision | 0.151042 | 0.359026 | 0 | 0.104167 | 0.208333 | 29 | 192 |
| CURRENT_STATE_PROJECTION | success | 0 | 0 | 0 | 0 | 0 | 0 | 192 |
| CURRENT_STATE_PROJECTION | min_clearance | 0.0814145 | 0.0902506 | 0.0433359 | 0.0688503 | 0.0939333 |  |  |
| CURRENT_STATE_PROJECTION | terminal_goal_error | 0.771634 | 0.281266 | 0.699543 | 0.734263 | 0.8107 |  |  |
| CURRENT_STATE_PROJECTION | trajectory_distortion_vs_plain | 1.84166 | 0.903652 | 1.62538 | 1.71184 | 1.97276 |  |  |
| CURRENT_STATE_PROJECTION | planning_latency_sec | 6.52661 | 1.0546 | 6.84993 | 6.37932 | 6.66902 |  |  |
| CURRENT_STATE_PROJECTION | projection_solver_latency_sec | 0.706898 | 0.12559 | 0.743725 | 0.68884 | 0.724007 |  |  |
| CURRENT_STATE_PROJECTION | failed_projection_solves | 5.19792 | 1.0744 | 5 | 5.04688 | 5.34896 |  |  |
| POV_ENDPOINT_PROJECTION | collision | 0.692708 | 0.462578 | 1 | 0.625 | 0.755208 | 133 | 192 |
| POV_ENDPOINT_PROJECTION | success | 0 | 0 | 0 | 0 | 0 | 0 | 192 |
| POV_ENDPOINT_PROJECTION | min_clearance | -0.0187196 | 0.105519 | -0.0540098 | -0.0333784 | -0.00349403 |  |  |
| POV_ENDPOINT_PROJECTION | terminal_goal_error | 1.1832 | 0.248084 | 1.19268 | 1.14895 | 1.21689 |  |  |
| POV_ENDPOINT_PROJECTION | trajectory_distortion_vs_plain | 6.54818 | 1.02598 | 6.351 | 6.39743 | 6.68893 |  |  |
| POV_ENDPOINT_PROJECTION | planning_latency_sec | 6.55465 | 0.990536 | 6.72837 | 6.41274 | 6.70183 |  |  |
| POV_ENDPOINT_PROJECTION | projection_solver_latency_sec | 0.711902 | 0.120908 | 0.728266 | 0.69437 | 0.728793 |  |  |
| POV_ENDPOINT_PROJECTION | failed_projection_solves | 2.92708 | 1.47044 | 3 | 2.71354 | 3.14583 |  |  |

## Primary POV update

At each original Euler flow time `t=k/7`, the implementation evaluates the raw
velocity `v_theta`, forms `x1_pred = x_t + (1-t) v_theta`, projects that endpoint,
and advances with `x_t <- x_t + (1/7) * (P(x1_pred)-x_t)`. It deliberately does
not divide by `1-t`.

## Reproduction and commands

The upstream `inference_global_planner.py` was reproduced before experiment code
was added, using Xvfb for the unchanged MuJoCo viewer and Acados v0.5.1. The run
reached its sampled goal and exited with status 0; its log is preserved separately.

```bash
export ACADOS_SOURCE_DIR=/workspace/acados-v0.5.1
export LD_LIBRARY_PATH=/workspace/yflow-robot-arm/.venv/lib/python3.12/site-packages/cmeel.prefix/lib:/workspace/acados-v0.5.1/lib:$LD_LIBRARY_PATH
python -m experiments.pov_projection.run_phase1 --seeds 32
```

## Limitations

- The six examples are ID tasks and share the repository's default obstacles.
- Candidate feasibility metrics use true obstacle boxes and collision-sphere radii,
  while the optimizer uses the upstream local convex corridor approximation.
- Finite-difference velocity and acceleration diagnostics are conservative proxies
  for one-shot trajectories.
- Per the gate, no safety-model run, OOD scenarios, adaptive trigger, normalized
  POV, or retraining was performed before this verdict.
