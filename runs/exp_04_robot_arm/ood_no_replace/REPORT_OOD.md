# Robot-arm Y-Flow no-replacement OOD report

## Decision

**LIMITED_SUPPORT** under the supplied qualitative decision rule.

The numeric checklist was fixed before the full run and yields
`STRONG_SUPPORT`. The supplied phrase "fail badly" had no
numeric cutoff; at reporting time it was conservatively operationalized as at
least 50% collision in a category. That post-result descriptive cutoff is labeled
in the JSON below and is not presented as a predeclared statistical gate.

```json
{
  "verdict": "LIMITED_SUPPORT",
  "numeric_checks_verdict": "STRONG_SUPPORT",
  "numeric_checks_predeclared_before_full_run": true,
  "qualitative_bad_failure_threshold": 0.5,
  "qualitative_bad_failure_threshold_predeclared": false,
  "qualitative_protocol_override": "LIMITED_SUPPORT because at least half of L05_NO_REPLACE trajectories collide in a category",
  "badly_failing_categories": [
    "ENLARGED_OBSTACLE"
  ],
  "checks": {
    "plain_reduction_ci_above_zero": true,
    "improves_at_least_3_of_5_categories": true,
    "replace_reduction_at_least_10pp_with_positive_ci": true,
    "plain_safe_new_collision_rate_le_5pct": true,
    "goal_error_within_plain_plus_0p15m": true,
    "direction_not_solution_evidence": true
  },
  "plain_safe_new_collision_rate": 0.0,
  "zstar_collision_final_safe_count": 201,
  "categories_improved": 4
}
```

## Sanity checks

All four required OOD sanity scenarios passed before the 3,200-evaluation run.

| scenario_id | category | identical_noise | geometry_differs_from_id | collision_checker_matches_plot_bboxes | l05_no_replace_semantics | l05_replace_semantics | l1_replace_semantics | start_valid | goal_valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ood_shifted_000 | SHIFTED_OBSTACLE | True | True | True | True | True | True | True | True |
| ood_multi_000 | MULTI_OBSTACLE | True | True | True | True | True | True | True | True |
| ood_narrow_000 | NARROW_PASSAGE | True | True | True | True | True | True | True | True |
| ood_goalnear_000 | GOAL_NEAR_OBSTACLE | True | True | True | True | True | True | True | True |

## Overall metrics

| method | metric | mean | median | std | ci95_low | ci95_high | numerator | denominator |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PLAIN_FM | collision | 0.38125 | 0 | 0.485998 | 0.3475 | 0.415 | 305 | 800 |
| PLAIN_FM | min_clearance | 0.0408211 | 0.0297317 | 0.0836781 | 0.0353406 | 0.046304 |  |  |
| PLAIN_FM | terminal_goal_error | 0.571038 | 0.457342 | 0.270615 | 0.55215 | 0.590388 |  |  |
| PLAIN_FM | trajectory_distortion_vs_plain | 0 | 0 | 0 | 0 | 0 |  |  |
| PLAIN_FM | joint_path_length | 2.5227 | 2.52576 | 0.754597 | 2.46876 | 2.57467 |  |  |
| PLAIN_FM | smoothness | 0.0406765 | 0.0408255 | 0.00967615 | 0.0400692 | 0.0413667 |  |  |
| PLAIN_FM | velocity_violation | 0.142417 | 0.159212 | 0.126134 | 0.133955 | 0.150774 |  |  |
| PLAIN_FM | acceleration_violation | 0.880079 | 0.749512 | 0.896597 | 0.82149 | 0.942799 |  |  |
| PLAIN_FM | planning_latency_sec | 0.0739363 | 0.0707317 | 0.00462674 | 0.073624 | 0.074246 |  |  |
| PLAIN_FM | fm_latency_sec | 0.0444513 | 0.0443971 | 0.000429786 | 0.0444222 | 0.0444818 |  |  |
| PLAIN_FM | optimizer_latency_sec | 0 | 0 | 0 | 0 | 0 |  |  |
| PLAIN_FM | projection_calls | 0 | 0 | 0 | 0 | 0 |  |  |
| PLAIN_FM | failed_projection_solves | 0 | 0 | 0 | 0 | 0 |  |  |
| PLAIN_FM | projection_failure_rate | 0 | 0 | 0 | 0 | 0 |  |  |
| L1_REPLACE | collision | 0.43875 | 0 | 0.496545 | 0.405 | 0.4725 | 351 | 800 |
| L1_REPLACE | min_clearance | 0.0169249 | 0.00366382 | 0.0588758 | 0.0130859 | 0.0211973 |  |  |
| L1_REPLACE | terminal_goal_error | 0.381166 | 0.385775 | 0.155217 | 0.370076 | 0.392019 |  |  |
| L1_REPLACE | trajectory_distortion_vs_plain | 1.40325 | 1.37499 | 0.411529 | 1.37515 | 1.42975 |  |  |
| L1_REPLACE | joint_path_length | 2.04708 | 2.07837 | 0.562972 | 2.00694 | 2.08613 |  |  |
| L1_REPLACE | smoothness | 0.0252293 | 0.0242138 | 0.00823719 | 0.0246683 | 0.0257981 |  |  |
| L1_REPLACE | velocity_violation | 0.00254028 | 0 | 0.0188218 | 0.00131802 | 0.00399264 |  |  |
| L1_REPLACE | acceleration_violation | 0.013304 | 0 | 0.118563 | 0.0060829 | 0.0229708 |  |  |
| L1_REPLACE | planning_latency_sec | 0.679832 | 0.590261 | 0.504228 | 0.648456 | 0.714664 |  |  |
| L1_REPLACE | fm_latency_sec | 0.0453993 | 0.0453685 | 0.000453198 | 0.0453687 | 0.045432 |  |  |
| L1_REPLACE | optimizer_latency_sec | 0.544525 | 0.46503 | 0.502372 | 0.511985 | 0.581946 |  |  |
| L1_REPLACE | projection_calls | 3 | 3 | 0 | 3 | 3 |  |  |
| L1_REPLACE | failed_projection_solves | 0.06 | 0 | 0.420263 | 0.03375 | 0.09 |  |  |
| L1_REPLACE | projection_failure_rate | 0.02 | 0 | 0.140088 | 0.01125 | 0.03 |  |  |
| L05_REPLACE | collision | 0.45 | 0 | 0.497805 | 0.415 | 0.485 | 360 | 800 |
| L05_REPLACE | min_clearance | 0.0171499 | 0.00312297 | 0.0607648 | 0.0129863 | 0.0213553 |  |  |
| L05_REPLACE | terminal_goal_error | 0.386331 | 0.382749 | 0.150761 | 0.375999 | 0.396451 |  |  |
| L05_REPLACE | trajectory_distortion_vs_plain | 1.36272 | 1.34354 | 0.399043 | 1.33464 | 1.38949 |  |  |
| L05_REPLACE | joint_path_length | 2.07013 | 2.13609 | 0.573775 | 2.0314 | 2.10793 |  |  |
| L05_REPLACE | smoothness | 0.0255471 | 0.0248942 | 0.00838171 | 0.0249694 | 0.026102 |  |  |
| L05_REPLACE | velocity_violation | 0.00288858 | 0 | 0.0197268 | 0.00164416 | 0.00429524 |  |  |
| L05_REPLACE | acceleration_violation | 0.0137233 | 0 | 0.119942 | 0.00654113 | 0.0227007 |  |  |
| L05_REPLACE | planning_latency_sec | 0.683653 | 0.592219 | 0.500898 | 0.650892 | 0.719743 |  |  |
| L05_REPLACE | fm_latency_sec | 0.0462175 | 0.0461642 | 0.000546518 | 0.0461818 | 0.0462553 |  |  |
| L05_REPLACE | optimizer_latency_sec | 0.547493 | 0.466042 | 0.499126 | 0.515232 | 0.58365 |  |  |
| L05_REPLACE | projection_calls | 3 | 3 | 0 | 3 | 3 |  |  |
| L05_REPLACE | failed_projection_solves | 0.06 | 0 | 0.414264 | 0.03375 | 0.09 |  |  |
| L05_REPLACE | projection_failure_rate | 0.02 | 0 | 0.138088 | 0.0104167 | 0.0291667 |  |  |
| L05_NO_REPLACE | collision | 0.27375 | 0 | 0.446161 | 0.2425 | 0.305 | 219 | 800 |
| L05_NO_REPLACE | min_clearance | 0.0412683 | 0.0316072 | 0.0662287 | 0.0366944 | 0.045871 |  |  |
| L05_NO_REPLACE | terminal_goal_error | 0.476079 | 0.411319 | 0.200525 | 0.462075 | 0.489817 |  |  |
| L05_NO_REPLACE | trajectory_distortion_vs_plain | 0.752299 | 0.724494 | 0.231034 | 0.735765 | 0.768499 |  |  |
| L05_NO_REPLACE | joint_path_length | 2.21228 | 2.22779 | 0.678593 | 2.16581 | 2.25701 |  |  |
| L05_NO_REPLACE | smoothness | 0.0212343 | 0.0217399 | 0.0060121 | 0.0208235 | 0.0216439 |  |  |
| L05_NO_REPLACE | velocity_violation | 0.0176578 | 0 | 0.0327326 | 0.0154153 | 0.0200405 |  |  |
| L05_NO_REPLACE | acceleration_violation | 0.0218689 | 0 | 0.135874 | 0.0132508 | 0.0324926 |  |  |
| L05_NO_REPLACE | planning_latency_sec | 0.682512 | 0.591165 | 0.499505 | 0.650236 | 0.718506 |  |  |
| L05_NO_REPLACE | fm_latency_sec | 0.0462176 | 0.0461726 | 0.000524218 | 0.0461823 | 0.0462542 |  |  |
| L05_NO_REPLACE | optimizer_latency_sec | 0.54634 | 0.465173 | 0.497737 | 0.514163 | 0.583921 |  |  |
| L05_NO_REPLACE | projection_calls | 3 | 3 | 0 | 3 | 3 |  |  |
| L05_NO_REPLACE | failed_projection_solves | 0.05875 | 0 | 0.41293 | 0.0337187 | 0.09 |  |  |
| L05_NO_REPLACE | projection_failure_rate | 0.0195833 | 0 | 0.137643 | 0.0108333 | 0.03 |  |  |

## Overall and per-category generalization

| category | method | collision | collision_count | n | goal_error | min_clearance | distortion | latency | id_collision | ood_minus_id_collision | relative_collision_reduction_vs_plain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OVERALL | PLAIN_FM | 0.38125 | 305 | 800 | 0.571038 | 0.0408211 | 0 | 0.0739363 | 0.151042 | 0.230208 | 0 |
| OVERALL | L1_REPLACE | 0.43875 | 351 | 800 | 0.381166 | 0.0169249 | 1.40325 | 0.679832 | 0.333333 | 0.105417 | -0.15082 |
| OVERALL | L05_REPLACE | 0.45 | 360 | 800 | 0.386331 | 0.0171499 | 1.36272 | 0.683653 | 0.328125 | 0.121875 | -0.180328 |
| OVERALL | L05_NO_REPLACE | 0.27375 | 219 | 800 | 0.476079 | 0.0412683 | 0.752299 | 0.682512 | 0.00520833 | 0.268542 | 0.281967 |
| SHIFTED_OBSTACLE | PLAIN_FM | 0.2875 | 46 | 160 | 0.571038 | 0.091304 | 0 | 0.0699751 | 0.151042 | 0.136458 | 0 |
| SHIFTED_OBSTACLE | L1_REPLACE | 0.24375 | 39 | 160 | 0.389915 | 0.059542 | 1.40379 | 0.516028 | 0.333333 | -0.0895833 | 0.152174 |
| SHIFTED_OBSTACLE | L05_REPLACE | 0.26875 | 43 | 160 | 0.398456 | 0.0607605 | 1.36912 | 0.516336 | 0.328125 | -0.059375 | 0.0652174 |
| SHIFTED_OBSTACLE | L05_NO_REPLACE | 0.15 | 24 | 160 | 0.481007 | 0.0876872 | 0.75681 | 0.515359 | 0.00520833 | 0.144792 | 0.478261 |
| ENLARGED_OBSTACLE | PLAIN_FM | 0.5 | 80 | 160 | 0.571038 | 0.00982718 | 0 | 0.0702121 | 0.151042 | 0.348958 | 0 |
| ENLARGED_OBSTACLE | L1_REPLACE | 0.76875 | 123 | 160 | 0.351983 | -0.00919197 | 1.39982 | 0.602129 | 0.333333 | 0.435417 | -0.5375 |
| ENLARGED_OBSTACLE | L05_REPLACE | 0.75625 | 121 | 160 | 0.358901 | -0.0061971 | 1.35727 | 0.612374 | 0.328125 | 0.428125 | -0.5125 |
| ENLARGED_OBSTACLE | L05_NO_REPLACE | 0.5 | 80 | 160 | 0.463516 | 0.0123742 | 0.74685 | 0.61204 | 0.00520833 | 0.494792 | 0 |
| MULTI_OBSTACLE | PLAIN_FM | 0.425 | 68 | 160 | 0.571038 | 0.0249705 | 0 | 0.0794755 | 0.151042 | 0.273958 | 0 |
| MULTI_OBSTACLE | L1_REPLACE | 0.3375 | 54 | 160 | 0.366298 | 0.0135474 | 1.40648 | 0.888189 | 0.333333 | 0.00416667 | 0.205882 |
| MULTI_OBSTACLE | L05_REPLACE | 0.3375 | 54 | 160 | 0.377116 | 0.0147916 | 1.35016 | 0.888945 | 0.328125 | 0.009375 | 0.205882 |
| MULTI_OBSTACLE | L05_NO_REPLACE | 0.25 | 40 | 160 | 0.470988 | 0.0326213 | 0.746758 | 0.884254 | 0.00520833 | 0.244792 | 0.411765 |
| NARROW_PASSAGE | PLAIN_FM | 0.54375 | 87 | 160 | 0.571038 | 0.0152936 | 0 | 0.0704308 | 0.151042 | 0.392708 | 0 |
| NARROW_PASSAGE | L1_REPLACE | 0.54375 | 87 | 160 | 0.446584 | 0.00293834 | 1.36708 | 0.741307 | 0.333333 | 0.210417 | 0 |
| NARROW_PASSAGE | L05_REPLACE | 0.5875 | 94 | 160 | 0.444312 | 0.00177398 | 1.33213 | 0.752552 | 0.328125 | 0.259375 | -0.0804598 |
| NARROW_PASSAGE | L05_NO_REPLACE | 0.46875 | 75 | 160 | 0.506689 | 0.0185139 | 0.734613 | 0.752465 | 0.00520833 | 0.463542 | 0.137931 |
| GOAL_NEAR_OBSTACLE | PLAIN_FM | 0.15 | 24 | 160 | 0.571038 | 0.0627101 | 0 | 0.0795881 | 0.151042 | -0.00104167 | 0 |
| GOAL_NEAR_OBSTACLE | L1_REPLACE | 0.3 | 48 | 160 | 0.351049 | 0.0177889 | 1.4391 | 0.651506 | 0.333333 | -0.0333333 | -1 |
| GOAL_NEAR_OBSTACLE | L05_REPLACE | 0.3 | 48 | 160 | 0.352871 | 0.0146204 | 1.40491 | 0.648058 | 0.328125 | -0.028125 | -1 |
| GOAL_NEAR_OBSTACLE | L05_NO_REPLACE | 0 | 0 | 160 | 0.458195 | 0.0551448 | 0.776465 | 0.648441 | 0.00520833 | -0.00520833 | 1 |

## Paired collision analysis

The overall paired Plain-minus-L05_NO_REPLACE collision difference is
0.1075 (95% bootstrap CI 0.0862,
0.1300). The L05_REPLACE-minus-L05_NO_REPLACE difference
is 0.1762 (95% CI 0.1388,
0.2112).

| category | method | reference_method | reference_state | method_state | count | reference_minus_method_collision | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OVERALL | L1_REPLACE | PLAIN_FM | safe | safe | 308 | -0.0575 | -0.1025 | -0.01125 |
| OVERALL | L1_REPLACE | PLAIN_FM | safe | collision | 187 | -0.0575 | -0.1025 | -0.01125 |
| OVERALL | L1_REPLACE | PLAIN_FM | collision | safe | 141 | -0.0575 | -0.1025 | -0.01125 |
| OVERALL | L1_REPLACE | PLAIN_FM | collision | collision | 164 | -0.0575 | -0.1025 | -0.01125 |
| OVERALL | L05_REPLACE | PLAIN_FM | safe | safe | 305 | -0.06875 | -0.11 | -0.0249688 |
| OVERALL | L05_REPLACE | PLAIN_FM | safe | collision | 190 | -0.06875 | -0.11 | -0.0249688 |
| OVERALL | L05_REPLACE | PLAIN_FM | collision | safe | 135 | -0.06875 | -0.11 | -0.0249688 |
| OVERALL | L05_REPLACE | PLAIN_FM | collision | collision | 170 | -0.06875 | -0.11 | -0.0249688 |
| OVERALL | L05_NO_REPLACE | PLAIN_FM | safe | safe | 495 | 0.1075 | 0.08625 | 0.13 |
| OVERALL | L05_NO_REPLACE | PLAIN_FM | safe | collision | 0 | 0.1075 | 0.08625 | 0.13 |
| OVERALL | L05_NO_REPLACE | PLAIN_FM | collision | safe | 86 | 0.1075 | 0.08625 | 0.13 |
| OVERALL | L05_NO_REPLACE | PLAIN_FM | collision | collision | 219 | 0.1075 | 0.08625 | 0.13 |
| OVERALL | L05_NO_REPLACE | L05_REPLACE | safe | safe | 380 | 0.17625 | 0.13875 | 0.21125 |
| OVERALL | L05_NO_REPLACE | L05_REPLACE | safe | collision | 60 | 0.17625 | 0.13875 | 0.21125 |
| OVERALL | L05_NO_REPLACE | L05_REPLACE | collision | safe | 201 | 0.17625 | 0.13875 | 0.21125 |
| OVERALL | L05_NO_REPLACE | L05_REPLACE | collision | collision | 159 | 0.17625 | 0.13875 | 0.21125 |
| SHIFTED_OBSTACLE | L05_NO_REPLACE | L05_REPLACE | safe | safe | 97 | 0.11875 | 0.03125 | 0.2125 |
| SHIFTED_OBSTACLE | L05_NO_REPLACE | L05_REPLACE | safe | collision | 20 | 0.11875 | 0.03125 | 0.2125 |
| SHIFTED_OBSTACLE | L05_NO_REPLACE | L05_REPLACE | collision | safe | 39 | 0.11875 | 0.03125 | 0.2125 |
| SHIFTED_OBSTACLE | L05_NO_REPLACE | L05_REPLACE | collision | collision | 4 | 0.11875 | 0.03125 | 0.2125 |
| ENLARGED_OBSTACLE | L05_NO_REPLACE | L05_REPLACE | safe | safe | 32 | 0.25625 | 0.175 | 0.3375 |
| ENLARGED_OBSTACLE | L05_NO_REPLACE | L05_REPLACE | safe | collision | 7 | 0.25625 | 0.175 | 0.3375 |
| ENLARGED_OBSTACLE | L05_NO_REPLACE | L05_REPLACE | collision | safe | 48 | 0.25625 | 0.175 | 0.3375 |
| ENLARGED_OBSTACLE | L05_NO_REPLACE | L05_REPLACE | collision | collision | 73 | 0.25625 | 0.175 | 0.3375 |
| MULTI_OBSTACLE | L05_NO_REPLACE | L05_REPLACE | safe | safe | 96 | 0.0875 | 0.0125 | 0.15625 |
| MULTI_OBSTACLE | L05_NO_REPLACE | L05_REPLACE | safe | collision | 10 | 0.0875 | 0.0125 | 0.15625 |
| MULTI_OBSTACLE | L05_NO_REPLACE | L05_REPLACE | collision | safe | 24 | 0.0875 | 0.0125 | 0.15625 |
| MULTI_OBSTACLE | L05_NO_REPLACE | L05_REPLACE | collision | collision | 30 | 0.0875 | 0.0125 | 0.15625 |
| NARROW_PASSAGE | L05_NO_REPLACE | L05_REPLACE | safe | safe | 43 | 0.11875 | 0.01875 | 0.21875 |
| NARROW_PASSAGE | L05_NO_REPLACE | L05_REPLACE | safe | collision | 23 | 0.11875 | 0.01875 | 0.21875 |
| NARROW_PASSAGE | L05_NO_REPLACE | L05_REPLACE | collision | safe | 42 | 0.11875 | 0.01875 | 0.21875 |
| NARROW_PASSAGE | L05_NO_REPLACE | L05_REPLACE | collision | collision | 52 | 0.11875 | 0.01875 | 0.21875 |
| GOAL_NEAR_OBSTACLE | L05_NO_REPLACE | L05_REPLACE | safe | safe | 112 | 0.3 | 0.23125 | 0.375 |
| GOAL_NEAR_OBSTACLE | L05_NO_REPLACE | L05_REPLACE | safe | collision | 0 | 0.3 | 0.23125 | 0.375 |
| GOAL_NEAR_OBSTACLE | L05_NO_REPLACE | L05_REPLACE | collision | safe | 48 | 0.3 | 0.23125 | 0.375 |
| GOAL_NEAR_OBSTACLE | L05_NO_REPLACE | L05_REPLACE | collision | collision | 0 | 0.3 | 0.23125 | 0.375 |

## Terminal replacement mechanism

For L05_NO_REPLACE, the returned terminal target collides in
361/800 (45.12%), while the actual damped final
output collides in 219/800 (27.38%).
Among 784 successful terminal solves, actual `z_star`
collides in 345/784
(44.01%) and the final output in
203/784
(25.89%). In 201/800 cases, a successfully solved `z_star`
collides but the damped output is safe. There were 16 terminal
solver failures and 47/2400 failed L05_NO_REPLACE calls
across all active steps; failed solves consumed no unconverged solution and used
the fixed candidate/zero-correction fallback.

Category-level replacement analysis:

| category | replace_minus_no_replace_collision | replace_minus_no_replace_goal_error | replace_minus_no_replace_distortion | replacement_collisions_rescued_by_no_replace | no_replace_newly_broken_vs_replace | z_star_collision_rate | z_star_collision_rate_successful_solves |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SHIFTED_OBSTACLE | 0.11875 | -0.0825509 | 0.61231 | 39 | 20 | 0.26875 | 0.26875 |
| ENLARGED_OBSTACLE | 0.25625 | -0.104615 | 0.610424 | 48 | 7 | 0.75625 | 0.75625 |
| MULTI_OBSTACLE | 0.0875 | -0.0938722 | 0.603399 | 24 | 10 | 0.33125 | 0.296053 |
| NARROW_PASSAGE | 0.11875 | -0.0623772 | 0.597521 | 42 | 23 | 0.6 | 0.578947 |
| GOAL_NEAR_OBSTACLE | 0.3 | -0.105324 | 0.628443 | 48 | 0 | 0.3 | 0.3 |

## Required answers

1. **Does L05_NO_REPLACE reduce OOD collision relative to Plain?** Yes. Overall rates
   are 27.38% versus
   38.12%; the paired difference and CI are above.
2. **Does the ID 0.52% generalize?** No. OOD collision is
   27.38%, a change of
   +26.85% from ID. This is
   reported as observed benchmark generalization, not a population guarantee.
3. **Does no-replace remain better than replace?** Yes. Rates are
   27.38% versus
   45.00%; paired transitions quantify the cases.
4. **Is replacement still a dominant failure mechanism?** Yes overall. The replacement-minus-
   no-replacement paired effect is 17.62%; category rows show
   whether it is consistent across geometry types.
5. **How often is z_star colliding?** On successful terminal solves,
   345/784
   (44.01%); the all-sample returned-target rate including fixed
   failure fallback is 361/800 (45.12%).
6. **How often is z_star colliding while damped output is safe?** 201/800
   (25.12%).
7. **Hardest category for L05_NO_REPLACE:** ENLARGED_OBSTACLE, collision
   50.00%, goal error 0.4635 m.
8. **Safety cost:** Not in goal error: L05_NO_REPLACE goal error improves to
   0.4761 m versus
   0.5710 m. However, its distortion versus
   Plain is substantial at 0.7523.
9. **Direction rather than solution?** Supported:
   the true-geometry terminal audit compares the same optimizer solution and
   damped output on every matched OOD sample.

## Representative cases

```json
{
  "rescue_case": [
    "ood_goalnear_004",
    0
  ],
  "replacement_failure_case": [
    "ood_enlarged_003",
    0
  ],
  "zstar_collision_final_safe": [
    "ood_enlarged_003",
    0
  ],
  "narrow_passage_case": [
    "ood_narrow_000",
    0
  ],
  "failure_case": [
    "ood_enlarged_001",
    0
  ]
}
```

Missing case types, if any, are rendered honestly as `no matching case observed`.

## Scope

- 100 deterministic OOD scenarios (20/category), 8 matched seeds, 4 fixed methods.
- Same checkpoint, 7-step grid, no-P Acados optimizer, `mu=0`, `t_on=0.5`,
  `lambda_oc=10`, integration, goal, and solver-failure behavior.
- No retraining, adaptive gating, clipping, lambda/schedule tuning, or result-based
  scenario rejection. Start and goal validity were checked before sampling.
- Remote-main reference commit: `eeee5d2fee03f6c0bb593148727f0a92b94d3c97`.
- All earlier experiment artifacts remain untouched.
