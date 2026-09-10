# POV robot-arm Phase-2 report

## Decision

**PROMISING / LIMITED GO** under the predeclared corrected-POV gate.

This is a promising signal rather than a strong-positive endpoint-projection
result unless the gate table below marks `strong_positive=True`. Safety, goal,
distortion, new-collision transitions, and failure handling are all considered;
the verdict is not based on collision reduction alone.

The benchmark used the unchanged unsafe Flow Matching checkpoint and the exact
Phase-1 matched ID set. No retraining, OOD scenarios, clipping, adaptive trigger,
or late-start tuning was run before this decision.

## Mandatory correctness tests

```json
{
  "projection_formulation": "strong soft Cartesian cost at returned knot H-1: 1000 * ||FK(q_H)-x_goal||^2",
  "identity_projection_invariance": {
    "max_abs_difference": 0.0,
    "l2_difference": 0.0,
    "passed": true
  },
  "projection_fixed_point": {
    "count": 6,
    "median_correction": 4.093694065024967e-08,
    "mean_correction": 2.5909332612071196e-07,
    "maximum_correction": 7.578531051036021e-07,
    "first_projection_successes": 6,
    "passed": true
  },
  "all_passed": true,
  "hard_constraint_probe": {
    "projection_formulation": "hard Cartesian constraint at returned knot H-1: ||FK(q_H)-x_goal|| <= 0.03 m",
    "identity_projection_invariance": {
      "max_abs_difference": 0.0,
      "l2_difference": 0.0,
      "passed": true
    },
    "projection_fixed_point": {
      "count": 6,
      "median_correction": 7.767507410795321e-08,
      "mean_correction": 3.955784332104985e-07,
      "maximum_correction": 1.578206885270726e-06,
      "first_projection_successes": 6,
      "passed": true
    },
    "successful_projection_count": 0,
    "attempted_projection_count": 6,
    "all_passed": false
  }
}
```

Goal preservation passed: `True` using
`6` successful
projections out of `6`.

## Projection formulation

`strong soft Cartesian cost at returned knot H-1: 1000 * ||FK(q_H)-x_goal||^2`

The objective otherwise retains the upstream weighted joint-trajectory distance
and derivative/control smoothness terms. Collision constraints remain candidate-
dependent local convex corridors with soft nonlinear slacks. A failed solve returns
the original candidate; its unconverged iterate is never consumed.

## Primary summary

| method | metric | mean | std | median | ci95_low | ci95_high | numerator | denominator |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PLAIN_FM | collision | 0.151042 | 0.359026 | 0 | 0.104167 | 0.203125 | 29 | 192 |
| PLAIN_FM | min_clearance | 0.0660799 | 0.0464402 | 0.0722477 | 0.0593746 | 0.0726474 |  |  |
| PLAIN_FM | terminal_goal_error | 0.581712 | 0.29237 | 0.456701 | 0.540992 | 0.622951 |  |  |
| PLAIN_FM | trajectory_distortion_vs_plain | 0 | 0 | 0 | 0 | 0 |  |  |
| PLAIN_FM | planning_latency_sec | 0.0707673 | 0.00151678 | 0.0706833 | 0.0705897 | 0.0710186 |  |  |
| PLAIN_FM | failed_projection_solves | 0 | 0 | 0 | 0 | 0 |  |  |
| PLAIN_FM | projection_failure_rate | 0 | 0 | 0 | 0 | 0 |  |  |
| GOAL_AWARE_FINAL_PROJECTION | collision | 0.484375 | 0.501062 | 0 | 0.411458 | 0.557292 | 93 | 192 |
| GOAL_AWARE_FINAL_PROJECTION | min_clearance | -0.00110647 | 0.0401367 | 0.00828292 | -0.00681329 | 0.00441677 |  |  |
| GOAL_AWARE_FINAL_PROJECTION | terminal_goal_error | 0.386458 | 0.117702 | 0.38499 | 0.369605 | 0.402383 |  |  |
| GOAL_AWARE_FINAL_PROJECTION | trajectory_distortion_vs_plain | 3.24864 | 0.602826 | 3.14411 | 3.16775 | 3.33664 |  |  |
| GOAL_AWARE_FINAL_PROJECTION | planning_latency_sec | 0.180683 | 0.0236444 | 0.173991 | 0.177734 | 0.184073 |  |  |
| GOAL_AWARE_FINAL_PROJECTION | failed_projection_solves | 0 | 0 | 0 | 0 | 0 |  |  |
| GOAL_AWARE_FINAL_PROJECTION | projection_failure_rate | 0 | 0 | 0 | 0 | 0 |  |  |
| NORMALIZED_POV_L1 | collision | 0.208333 | 0.407178 | 0 | 0.15625 | 0.265625 | 40 | 192 |
| NORMALIZED_POV_L1 | min_clearance | 0.0295234 | 0.0598081 | 0.0284561 | 0.0212136 | 0.0380434 |  |  |
| NORMALIZED_POV_L1 | terminal_goal_error | 0.398662 | 0.174631 | 0.406192 | 0.373259 | 0.425136 |  |  |
| NORMALIZED_POV_L1 | trajectory_distortion_vs_plain | 3.82213 | 1.00131 | 3.74531 | 3.68235 | 3.96146 |  |  |
| NORMALIZED_POV_L1 | planning_latency_sec | 1.44375 | 0.224671 | 1.42864 | 1.41181 | 1.47797 |  |  |
| NORMALIZED_POV_L1 | failed_projection_solves | 0 | 0 | 0 | 0 | 0 |  |  |
| NORMALIZED_POV_L1 | projection_failure_rate | 0 | 0 | 0 | 0 | 0 |  |  |
| DAMPED_POV_L05 | collision | 0.00520833 | 0.0721688 | 0 | 0 | 0.015625 | 1 | 192 |
| DAMPED_POV_L05 | min_clearance | 0.0493887 | 0.0243283 | 0.0472141 | 0.0460763 | 0.052666 |  |  |
| DAMPED_POV_L05 | terminal_goal_error | 0.43945 | 0.197076 | 0.377425 | 0.411713 | 0.467468 |  |  |
| DAMPED_POV_L05 | trajectory_distortion_vs_plain | 1.95452 | 0.367732 | 1.99082 | 1.90123 | 2.00635 |  |  |
| DAMPED_POV_L05 | planning_latency_sec | 1.3676 | 0.211118 | 1.33714 | 1.33943 | 1.398 |  |  |
| DAMPED_POV_L05 | failed_projection_solves | 0 | 0 | 0 | 0 | 0 |  |  |
| DAMPED_POV_L05 | projection_failure_rate | 0 | 0 | 0 | 0 | 0 |  |  |
| DAMPED_POV_L025 | collision | 0.0208333 | 0.1432 | 0 | 0.00520833 | 0.0416667 | 4 | 192 |
| DAMPED_POV_L025 | min_clearance | 0.0610744 | 0.0377885 | 0.0577136 | 0.0559104 | 0.0664915 |  |  |
| DAMPED_POV_L025 | terminal_goal_error | 0.506958 | 0.24632 | 0.417729 | 0.472609 | 0.54241 |  |  |
| DAMPED_POV_L025 | trajectory_distortion_vs_plain | 0.993961 | 0.166419 | 1.02947 | 0.97084 | 1.01765 |  |  |
| DAMPED_POV_L025 | planning_latency_sec | 1.36905 | 0.20999 | 1.33607 | 1.33877 | 1.39858 |  |  |
| DAMPED_POV_L025 | failed_projection_solves | 0 | 0 | 0 | 0 | 0 |  |  |
| DAMPED_POV_L025 | projection_failure_rate | 0 | 0 | 0 | 0 | 0 |  |  |

## Gate details

| method | collision_below_plain | goal_substantially_better_than_old_pov | distortion_substantially_below_old_pov | new_collision_rate_from_plain_safe | few_new_collisions | near_final_safety | goal_better_than_final | distortion_better_than_final | promising | strong_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NORMALIZED_POV_L1 | False | True | True | 0.239264 | False | True | False | False | False | False |
| DAMPED_POV_L05 | True | True | True | 0 | True | True | False | True | True | False |
| DAMPED_POV_L025 | True | True | True | 0 | True | True | False | True | True | False |

## Paired collision transitions

| method | plain_state | pov_state | count |
| --- | --- | --- | --- |
| NORMALIZED_POV_L1 | safe | safe | 124 |
| NORMALIZED_POV_L1 | safe | collision | 39 |
| NORMALIZED_POV_L1 | collision | safe | 28 |
| NORMALIZED_POV_L1 | collision | collision | 1 |
| DAMPED_POV_L05 | safe | safe | 163 |
| DAMPED_POV_L05 | safe | collision | 0 |
| DAMPED_POV_L05 | collision | safe | 28 |
| DAMPED_POV_L05 | collision | collision | 1 |
| DAMPED_POV_L025 | safe | safe | 163 |
| DAMPED_POV_L025 | safe | collision | 0 |
| DAMPED_POV_L025 | collision | safe | 25 |
| DAMPED_POV_L025 | collision | collision | 4 |

## Paired statistics

| method | metric | paired_mean_difference_vs_plain | paired_median_difference_vs_plain | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- |
| NORMALIZED_POV_L1 | collision | 0.0572917 | 0 | -0.0208333 | 0.140625 |
| NORMALIZED_POV_L1 | terminal_goal_error | -0.18305 | -0.134855 | -0.216505 | -0.150989 |
| NORMALIZED_POV_L1 | trajectory_distortion_vs_plain | 3.82213 | 3.74531 | 3.68352 | 3.96556 |
| DAMPED_POV_L05 | collision | -0.145833 | 0 | -0.197917 | -0.0989583 |
| DAMPED_POV_L05 | terminal_goal_error | -0.142262 | -0.098656 | -0.158171 | -0.126461 |
| DAMPED_POV_L05 | trajectory_distortion_vs_plain | 1.95452 | 1.99082 | 1.90128 | 2.0065 |
| DAMPED_POV_L025 | collision | -0.130208 | 0 | -0.177083 | -0.0833333 |
| DAMPED_POV_L025 | terminal_goal_error | -0.0747541 | -0.0497734 | -0.0822889 | -0.0671558 |
| DAMPED_POV_L025 | trajectory_distortion_vs_plain | 0.993961 | 1.02947 | 0.97035 | 1.01693 |

## Final-step stability diagnostics

| method | t | mean_delta | mean_normalized_delta | mean_velocity_change | projection_failure_rate | endpoint_violation |
| --- | --- | --- | --- | --- | --- | --- |
| DAMPED_POV_L025 | 0.857143 | 3.12312 | 21.8619 | 5.46546 | 0 | 1.01988 |
| DAMPED_POV_L05 | 0.857143 | 3.05028 | 21.352 | 10.676 | 0 | 1.32222 |
| NORMALIZED_POV_L1 | 0.857143 | 2.92912 | 20.5038 | 20.5038 | 0 | 4.15431 |

The last sampled flow time is `t=6/7`, so normalization amplifies endpoint
corrections by a factor of seven. Mean normalized corrections near `20--22` are
large; damping reduces the velocity change in direct proportion to lambda but
does not remove the underlying late-time amplification.

The primary corrected update was exactly:

`v_guided = v_t + lambda_pov * (P(x_t + (1-t)v_t) - (x_t + (1-t)v_t)) / max(1-t, 1e-3)`.

No arbitrary correction clipping was applied. Projection failures fell back to
the vanilla `v_t` for that step.

## Scope and limitations

- This is a controlled one-shot 16-knot ID benchmark, not a closed-loop MPC rollout.
- The Cartesian goal is the full demonstration endpoint, which can be farther than
  the first 16 reference knots reach; goal feasibility therefore depends on the OCP
  dynamics and horizon rather than on copying the reference prefix.
- Candidate metrics use true obstacle boxes and collision-sphere radii, while the
  optimizer uses SafeFlowMPC's local corridor approximation.
- Phase 3 is intentionally not run automatically.
