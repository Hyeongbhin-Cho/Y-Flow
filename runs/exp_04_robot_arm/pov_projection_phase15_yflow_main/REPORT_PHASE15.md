# Robot-arm Y-Flow Phase 1.5 report

## Decision

**NO-GO** for the predeclared remote-main outer-loop transfer gate.

```json
{
  "collision_difference_vs_always": 0.328125,
  "collision_within_2pp": false,
  "new_collisions_vs_always": 64,
  "new_collision_rate_from_always_safe": 0.33507853403141363,
  "few_new_collisions": false,
  "projection_call_reduction": 0.5714285714285714,
  "calls_reduced_at_least_50pct": true,
  "latency_reduction": 0.6507507388924104,
  "latency_reduced_at_least_35pct": true,
  "goal_error_difference": -0.086101457598516,
  "goal_not_worse_by_more_than_0p1m": true,
  "passed": false
}
```

## Primary matched results

| method | metric | mean | ci95_low | ci95_high | numerator | denominator |
| --- | --- | --- | --- | --- | --- | --- |
| PLAIN_FM | collision | 0.151042 | 0.104167 | 0.203125 | 29 | 192 |
| PLAIN_FM | terminal_goal_error | 0.581712 | 0.540992 | 0.622951 | nan | nan |
| PLAIN_FM | planning_latency_sec | 0.0700913 | 0.0699265 | 0.070345 | nan | nan |
| PLAIN_FM | projection_calls | 0 | 0 | 0 | nan | nan |
| POV_L05_ALWAYS | collision | 0.00520833 | 0 | 0.015625 | 1 | 192 |
| POV_L05_ALWAYS | terminal_goal_error | 0.441042 | 0.413182 | 0.469434 | nan | nan |
| POV_L05_ALWAYS | planning_latency_sec | 1.36272 | 1.33321 | 1.39507 | nan | nan |
| POV_L05_ALWAYS | projection_calls | 7 | 7 | 7 | nan | nan |
| YFLOW_MAIN_PORT | collision | 0.333333 | 0.265625 | 0.401042 | 64 | 192 |
| YFLOW_MAIN_PORT | terminal_goal_error | 0.354941 | 0.336543 | 0.372188 | nan | nan |
| YFLOW_MAIN_PORT | planning_latency_sec | 0.475929 | 0.462963 | 0.488981 | nan | nan |
| YFLOW_MAIN_PORT | projection_calls | 3 | 3 | 3 | nan | nan |
| GOAL_AWARE_FINAL_PROJECTION | collision | 0.484375 | 0.416667 | 0.557292 | 93 | 192 |
| GOAL_AWARE_FINAL_PROJECTION | terminal_goal_error | 0.386458 | 0.369521 | 0.40318 | nan | nan |
| GOAL_AWARE_FINAL_PROJECTION | planning_latency_sec | 0.180609 | 0.177323 | 0.184022 | nan | nan |
| GOAL_AWARE_FINAL_PROJECTION | projection_calls | 1 | 1 | 1 | nan | nan |

## Paired collision transitions

| reference_method | reference_state | yflow_state | count |
| --- | --- | --- | --- |
| PLAIN_FM | safe | safe | 99 |
| PLAIN_FM | safe | collision | 64 |
| PLAIN_FM | collision | safe | 29 |
| PLAIN_FM | collision | collision | 0 |
| POV_L05_ALWAYS | safe | safe | 127 |
| POV_L05_ALWAYS | safe | collision | 64 |
| POV_L05_ALWAYS | collision | safe | 1 |
| POV_L05_ALWAYS | collision | collision | 0 |

## What was transferred exactly

The outer update follows Y-Flow remote `main` commit `eeee5d2fee03f6c0bb593148727f0a92b94d3c97`:

1. `x1_raw = x + (1-t) * v`;
2. non-terminal steps with `t < 0.5` use vanilla interpolation;
3. later steps solve a terminal constrained target;
4. `eta = dt/(1-t)`, with `eta = 1` at the terminal step;
5. `x_next = (1-eta) * x + eta * z_star`;
6. raw-target weight is `lambda_oc * t^2 / dt`, with `lambda_oc=10`.

On the seven-step grid this makes exactly three optimization calls at
`t=4/7, 5/7, 6/7`.

## Necessary robot-domain substitution

The remote code's `P` is a differentiable 2-D Swiss-roll centerline projection
and its inner solver is GPU PGD. No corresponding analytic robot trajectory
manifold operator exists in SafeFlowMPC. We therefore use the Y-Flow document's
explicit no-`P` path (`mu=0`): the existing Acados OCP supplies the terminal
optimization, its robot goal/smoothness terms act as `C`, and its trajectory
constraints act as `h`.

This is an exact transfer of the Y-Flow scheduling and interpolation mechanism,
not a claim that 2-D PGD and robot Acados are identical inner solvers. Collision
corridors and the Cartesian goal remain soft penalties in the inherited robot
OCP, while joint/velocity/acceleration bounds are hard.

## Result interpretation

- Collision: Y-Flow port 0.3333 vs always-on POV 0.0052.
- Goal error: 0.3549 vs 0.4410 m.
- Calls: 3.0000 vs 7.0000
  (57.14% reduction).
- Latency: 0.4759 vs 1.3627 s
  (65.08% reduction).
- New collisions relative to always-on: 64.

All 64 Y-Flow-port collisions were new relative to always-on POV. They were
concentrated in task 3 and task 5 (32/32 seeds in each), while the port rescued
all 29 Plain-FM collisions from task 4. This is therefore a task-specific safety
trade rather than uniform degradation.

Every Acados solve call returned solver success, but solver success did not imply
safety under the true obstacle geometry. The local collision corridor is soft and
candidate-dependent; with the large late raw-target weights and terminal full
replacement, the optimized trajectory can still collide with the true boxes.

### True-geometry audit after each active optimization

| t | true_collision_after_projection | mean_true_clearance_after_projection | solver_success |
| --- | --- | --- | --- |
| 0.571429 | 0.286458 | 0.0116655 | 1 |
| 0.714286 | 0.317708 | 0.00938362 | 1 |
| 0.857143 | 0.333333 | 0.0144307 | 1 |

### Per-task breakdown

| task_id | method | collision_rate | goal_error | min_clearance |
| --- | --- | --- | --- | --- |
| 0 | GOAL_AWARE_FINAL_PROJECTION | 0 | 0.434995 | 0.0503821 |
| 0 | PLAIN_FM | 0 | 0.447589 | 0.0675023 |
| 0 | POV_L05_ALWAYS | 0 | 0.435119 | 0.0411191 |
| 0 | YFLOW_MAIN_PORT | 0 | 0.415852 | 0.0338531 |
| 1 | GOAL_AWARE_FINAL_PROJECTION | 0.0625 | 0.302382 | 0.0202669 |
| 1 | PLAIN_FM | 0 | 0.472374 | 0.0254297 |
| 1 | POV_L05_ALWAYS | 0 | 0.314706 | 0.0328623 |
| 1 | YFLOW_MAIN_PORT | 0 | 0.19367 | 0.0233912 |
| 2 | GOAL_AWARE_FINAL_PROJECTION | 0.78125 | 0.431838 | -0.00810936 |
| 2 | PLAIN_FM | 0 | 0.393215 | 0.0755328 |
| 2 | POV_L05_ALWAYS | 0 | 0.345335 | 0.0507461 |
| 2 | YFLOW_MAIN_PORT | 0 | 0.340921 | 0.0566219 |
| 3 | GOAL_AWARE_FINAL_PROJECTION | 1 | 0.379313 | -0.0566762 |
| 3 | PLAIN_FM | 0 | 0.9141 | 0.106214 |
| 3 | POV_L05_ALWAYS | 0 | 0.644064 | 0.0739622 |
| 3 | YFLOW_MAIN_PORT | 1 | 0.526718 | -0.0263448 |
| 4 | GOAL_AWARE_FINAL_PROJECTION | 0.0625 | 0.210954 | 0.0273105 |
| 4 | PLAIN_FM | 0.90625 | 0.226989 | -0.00695405 |
| 4 | POV_L05_ALWAYS | 0.03125 | 0.173216 | 0.0142634 |
| 4 | YFLOW_MAIN_PORT | 0 | 0.205106 | 0.0385641 |
| 5 | GOAL_AWARE_FINAL_PROJECTION | 1 | 0.559268 | -0.0398128 |
| 5 | PLAIN_FM | 0 | 1.036 | 0.128755 |
| 5 | POV_L05_ALWAYS | 0 | 0.733814 | 0.0826539 |
| 5 | YFLOW_MAIN_PORT | 1 | 0.447376 | -0.0395014 |

## Flow-time diagnostics

| t | projection_rate | raw_tracking_weight | correction_norm | solver_success |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | nan |
| 0.142857 | 0 | 0 | 0 | nan |
| 0.285714 | 0 | 0 | 0 | nan |
| 0.428571 | 0 | 0 | 0 | nan |
| 0.571429 | 1 | 22.8571 | 1.75331 | 1 |
| 0.714286 | 1 | 35.7143 | 1.49051 | 1 |
| 0.857143 | 1 | 51.4286 | 1.17483 | 1 |

## Scope

- Same six ID tasks, 32 seeds per task, unsafe FM checkpoint and matched noise.
- No retraining, OOD evaluation, time-window tuning or coefficient sweep.
- Phase 1, Phase 2 and Phase 3 artifacts were not overwritten.
