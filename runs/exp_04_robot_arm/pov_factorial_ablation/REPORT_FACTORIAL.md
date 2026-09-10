# Robot-arm Y-Flow 2x2 factorial ablation

## Sanity checks

The fixed implementation reproduced the existing reference samplers on four matched
task/seed cases before the full run. Overall status: **PASS**.

| task_id | seed | method | reference | max_abs_trajectory_difference | collision_match | goal_error_difference | pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | L1_REPLACE | YFLOW_MAIN_PORT | 0 | True | 0 | True |
| 0 | 0 | L05_NO_REPLACE | HYBRID_LATE_DAMPED | 0 | True | 0 | True |
| 3 | 0 | L1_REPLACE | YFLOW_MAIN_PORT | 0 | True | 0 | True |
| 3 | 0 | L05_NO_REPLACE | HYBRID_LATE_DAMPED | 0 | True | 0 | True |
| 4 | 0 | L1_REPLACE | YFLOW_MAIN_PORT | 0 | True | 0 | True |
| 4 | 0 | L05_NO_REPLACE | HYBRID_LATE_DAMPED | 0 | True | 0 | True |
| 5 | 0 | L1_REPLACE | YFLOW_MAIN_PORT | 0 | True | 0 | True |
| 5 | 0 | L05_NO_REPLACE | HYBRID_LATE_DAMPED | 0 | True | 0 | True |

At the final grid point, `dt = 1-t = 1/7`; therefore a lambda=1 residual Euler
update is algebraically equal to `z_star`. This means L1_NO_REPLACE and L1_REPLACE
are implementation-label variants, not an independently manipulable terminal factor
at the final step. The requested results are reported, but this structural aliasing
limits the nominal 2x2 causal interpretation.

## Primary metrics

| method | metric | mean | ci95_low | ci95_high | numerator | denominator |
| --- | --- | --- | --- | --- | --- | --- |
| PLAIN_FM | collision | 0.151042 | 0.104167 | 0.203125 | 29 | 192 |
| PLAIN_FM | min_clearance | 0.0660799 | 0.0593572 | 0.0724806 |  |  |
| PLAIN_FM | terminal_goal_error | 0.581712 | 0.540359 | 0.622926 |  |  |
| PLAIN_FM | trajectory_distortion_vs_plain | 0 | 0 | 0 |  |  |
| PLAIN_FM | joint_path_length | 2.50436 | 2.39617 | 2.61074 |  |  |
| PLAIN_FM | smoothness | 0.0400362 | 0.0384968 | 0.0414949 |  |  |
| PLAIN_FM | velocity_violation | 0.141085 | 0.124002 | 0.157671 |  |  |
| PLAIN_FM | acceleration_violation | 0.869476 | 0.748884 | 0.996602 |  |  |
| PLAIN_FM | planning_latency_sec | 0.0693736 | 0.0692886 | 0.0694711 |  |  |
| PLAIN_FM | fm_latency_sec | 0.0437689 | 0.0437088 | 0.0438434 |  |  |
| PLAIN_FM | optimizer_latency_sec | 0 | 0 | 0 |  |  |
| PLAIN_FM | projection_calls | 0 | 0 | 0 |  |  |
| PLAIN_FM | failed_projection_solves | 0 | 0 | 0 |  |  |
| PLAIN_FM | projection_failure_rate | 0 | 0 | 0 |  |  |
| L1_REPLACE | collision | 0.333333 | 0.265625 | 0.401042 | 64 | 192 |
| L1_REPLACE | min_clearance | 0.0142318 | 0.00922712 | 0.0194065 |  |  |
| L1_REPLACE | terminal_goal_error | 0.354502 | 0.336485 | 0.371691 |  |  |
| L1_REPLACE | trajectory_distortion_vs_plain | 1.46636 | 1.41666 | 1.51503 |  |  |
| L1_REPLACE | joint_path_length | 2.01201 | 1.93162 | 2.09422 |  |  |
| L1_REPLACE | smoothness | 0.0229677 | 0.0218321 | 0.0240795 |  |  |
| L1_REPLACE | velocity_violation | 0.000464572 | 0.000282666 | 0.000661666 |  |  |
| L1_REPLACE | acceleration_violation | 0 | 0 | 0 |  |  |
| L1_REPLACE | planning_latency_sec | 0.504254 | 0.490796 | 0.518285 |  |  |
| L1_REPLACE | fm_latency_sec | 0.0446362 | 0.0445867 | 0.0446887 |  |  |
| L1_REPLACE | optimizer_latency_sec | 0.381453 | 0.368068 | 0.39427 |  |  |
| L1_REPLACE | projection_calls | 3 | 3 | 3 |  |  |
| L1_REPLACE | failed_projection_solves | 0 | 0 | 0 |  |  |
| L1_REPLACE | projection_failure_rate | 0 | 0 | 0 |  |  |
| L1_NO_REPLACE | collision | 0.333333 | 0.265625 | 0.401042 | 64 | 192 |
| L1_NO_REPLACE | min_clearance | 0.0142662 | 0.00919831 | 0.0192384 |  |  |
| L1_NO_REPLACE | terminal_goal_error | 0.352987 | 0.334849 | 0.370465 |  |  |
| L1_NO_REPLACE | trajectory_distortion_vs_plain | 1.46373 | 1.41537 | 1.50996 |  |  |
| L1_NO_REPLACE | joint_path_length | 2.01561 | 1.93694 | 2.09921 |  |  |
| L1_NO_REPLACE | smoothness | 0.0231989 | 0.0220447 | 0.0243616 |  |  |
| L1_NO_REPLACE | velocity_violation | 0.000504613 | 0.000302012 | 0.000738145 |  |  |
| L1_NO_REPLACE | acceleration_violation | 0 | 0 | 0 |  |  |
| L1_NO_REPLACE | planning_latency_sec | 0.509587 | 0.496699 | 0.523003 |  |  |
| L1_NO_REPLACE | fm_latency_sec | 0.0454396 | 0.0453954 | 0.045485 |  |  |
| L1_NO_REPLACE | optimizer_latency_sec | 0.386054 | 0.373265 | 0.399337 |  |  |
| L1_NO_REPLACE | projection_calls | 3 | 3 | 3 |  |  |
| L1_NO_REPLACE | failed_projection_solves | 0 | 0 | 0 |  |  |
| L1_NO_REPLACE | projection_failure_rate | 0 | 0 | 0 |  |  |
| L05_REPLACE | collision | 0.328125 | 0.265625 | 0.390755 | 63 | 192 |
| L05_REPLACE | min_clearance | 0.0120251 | 0.00557992 | 0.0179952 |  |  |
| L05_REPLACE | terminal_goal_error | 0.354661 | 0.337005 | 0.371888 |  |  |
| L05_REPLACE | trajectory_distortion_vs_plain | 1.42528 | 1.37899 | 1.47218 |  |  |
| L05_REPLACE | joint_path_length | 2.04488 | 1.95776 | 2.12884 |  |  |
| L05_REPLACE | smoothness | 0.0237997 | 0.0226971 | 0.0249519 |  |  |
| L05_REPLACE | velocity_violation | 0.000721033 | 0.000505295 | 0.000965759 |  |  |
| L05_REPLACE | acceleration_violation | 0 | 0 | 0 |  |  |
| L05_REPLACE | planning_latency_sec | 0.50687 | 0.494168 | 0.519371 |  |  |
| L05_REPLACE | fm_latency_sec | 0.0454515 | 0.0454025 | 0.045503 |  |  |
| L05_REPLACE | optimizer_latency_sec | 0.383335 | 0.371105 | 0.397034 |  |  |
| L05_REPLACE | projection_calls | 3 | 3 | 3 |  |  |
| L05_REPLACE | failed_projection_solves | 0 | 0 | 0 |  |  |
| L05_REPLACE | projection_failure_rate | 0 | 0 | 0 |  |  |
| L05_NO_REPLACE | collision | 0.00520833 | 0 | 0.015625 | 1 | 192 |
| L05_NO_REPLACE | min_clearance | 0.057124 | 0.0530671 | 0.061314 |  |  |
| L05_NO_REPLACE | terminal_goal_error | 0.465158 | 0.437273 | 0.493736 |  |  |
| L05_NO_REPLACE | trajectory_distortion_vs_plain | 0.787176 | 0.757278 | 0.814144 |  |  |
| L05_NO_REPLACE | joint_path_length | 2.18746 | 2.08943 | 2.28598 |  |  |
| L05_NO_REPLACE | smoothness | 0.0201056 | 0.0192886 | 0.0208795 |  |  |
| L05_NO_REPLACE | velocity_violation | 0.0188454 | 0.0142683 | 0.0236917 |  |  |
| L05_NO_REPLACE | acceleration_violation | 0.00663169 | 2.4531e-05 | 0.0170641 |  |  |
| L05_NO_REPLACE | planning_latency_sec | 0.506835 | 0.494644 | 0.520375 |  |  |
| L05_NO_REPLACE | fm_latency_sec | 0.0454442 | 0.0453958 | 0.0454991 |  |  |
| L05_NO_REPLACE | optimizer_latency_sec | 0.383267 | 0.370141 | 0.395685 |  |  |
| L05_NO_REPLACE | projection_calls | 3 | 3 | 3 |  |  |
| L05_NO_REPLACE | failed_projection_solves | 0 | 0 | 0 |  |  |
| L05_NO_REPLACE | projection_failure_rate | 0 | 0 | 0 |  |  |

## Paired collision transitions

| method | reference_method | reference_state | method_state | count |
| --- | --- | --- | --- | --- |
| L1_REPLACE | PLAIN_FM | safe | safe | 99 |
| L1_REPLACE | PLAIN_FM | safe | collision | 64 |
| L1_REPLACE | PLAIN_FM | collision | safe | 29 |
| L1_REPLACE | PLAIN_FM | collision | collision | 0 |
| L1_NO_REPLACE | PLAIN_FM | safe | safe | 99 |
| L1_NO_REPLACE | PLAIN_FM | safe | collision | 64 |
| L1_NO_REPLACE | PLAIN_FM | collision | safe | 29 |
| L1_NO_REPLACE | PLAIN_FM | collision | collision | 0 |
| L05_REPLACE | PLAIN_FM | safe | safe | 100 |
| L05_REPLACE | PLAIN_FM | safe | collision | 63 |
| L05_REPLACE | PLAIN_FM | collision | safe | 29 |
| L05_REPLACE | PLAIN_FM | collision | collision | 0 |
| L05_NO_REPLACE | PLAIN_FM | safe | safe | 163 |
| L05_NO_REPLACE | PLAIN_FM | safe | collision | 0 |
| L05_NO_REPLACE | PLAIN_FM | collision | safe | 28 |
| L05_NO_REPLACE | PLAIN_FM | collision | collision | 1 |
| L1_REPLACE | L1_NO_REPLACE | safe | safe | 128 |
| L1_REPLACE | L1_NO_REPLACE | safe | collision | 0 |
| L1_REPLACE | L1_NO_REPLACE | collision | safe | 0 |
| L1_REPLACE | L1_NO_REPLACE | collision | collision | 64 |
| L05_REPLACE | L05_NO_REPLACE | safe | safe | 128 |
| L05_REPLACE | L05_NO_REPLACE | safe | collision | 63 |
| L05_REPLACE | L05_NO_REPLACE | collision | safe | 1 |
| L05_REPLACE | L05_NO_REPLACE | collision | collision | 0 |
| L1_NO_REPLACE | L05_NO_REPLACE | safe | safe | 127 |
| L1_NO_REPLACE | L05_NO_REPLACE | safe | collision | 64 |
| L1_NO_REPLACE | L05_NO_REPLACE | collision | safe | 1 |
| L1_NO_REPLACE | L05_NO_REPLACE | collision | collision | 0 |

## Per-task results

| task_id | method | collision_rate | goal_error | min_clearance |
| --- | --- | --- | --- | --- |
| 0 | L05_NO_REPLACE | 0 | 0.42721 | 0.041739 |
| 0 | L05_REPLACE | 0 | 0.413079 | 0.033906 |
| 0 | L1_NO_REPLACE | 0 | 0.417524 | 0.0330152 |
| 0 | L1_REPLACE | 0 | 0.415852 | 0.0338531 |
| 0 | PLAIN_FM | 0 | 0.447589 | 0.0675023 |
| 1 | L05_NO_REPLACE | 0 | 0.327635 | 0.0330112 |
| 1 | L05_REPLACE | 0 | 0.212826 | 0.0248936 |
| 1 | L1_NO_REPLACE | 0 | 0.192858 | 0.0247622 |
| 1 | L1_REPLACE | 0 | 0.19367 | 0.0233912 |
| 1 | PLAIN_FM | 0 | 0.472374 | 0.0254297 |
| 2 | L05_NO_REPLACE | 0 | 0.358595 | 0.0694211 |
| 2 | L05_REPLACE | 0 | 0.336994 | 0.0601502 |
| 2 | L1_NO_REPLACE | 0 | 0.337109 | 0.0562339 |
| 2 | L1_REPLACE | 0 | 0.340921 | 0.0566219 |
| 2 | PLAIN_FM | 0 | 0.393215 | 0.0755328 |
| 3 | L05_NO_REPLACE | 0 | 0.724196 | 0.102867 |
| 3 | L05_REPLACE | 1 | 0.528175 | -0.0300067 |
| 3 | L1_NO_REPLACE | 1 | 0.526211 | -0.0270286 |
| 3 | L1_REPLACE | 1 | 0.526718 | -0.0263448 |
| 3 | PLAIN_FM | 0 | 0.9141 | 0.106214 |
| 4 | L05_NO_REPLACE | 0.03125 | 0.22023 | 0.0203106 |
| 4 | L05_REPLACE | 0 | 0.204417 | 0.0397662 |
| 4 | L1_NO_REPLACE | 0 | 0.19755 | 0.0387772 |
| 4 | L1_REPLACE | 0 | 0.202476 | 0.0373707 |
| 4 | PLAIN_FM | 0.90625 | 0.226989 | -0.00695405 |
| 5 | L05_NO_REPLACE | 0 | 0.733081 | 0.0753945 |
| 5 | L05_REPLACE | 0.96875 | 0.432476 | -0.0565587 |
| 5 | L1_NO_REPLACE | 1 | 0.446669 | -0.0401627 |
| 5 | L1_REPLACE | 1 | 0.447376 | -0.0395014 |
| 5 | PLAIN_FM | 0 | 1.036 | 0.128755 |

## Factorial descriptive effects

Positive collision effects mean worse safety. These are descriptive matched-benchmark
effects, not population-level causal estimates from six tasks.

| metric | effect_lambda_l1_minus_l05 | effect_replace_minus_no_replace | interaction_difference_in_differences |
| --- | --- | --- | --- |
| collision | 0.166667 | 0.161458 | -0.322917 |
| terminal_goal_error | -0.056165 | -0.0544905 | 0.112012 |
| trajectory_distortion_vs_plain | 0.358818 | 0.32037 | -0.635464 |
| min_clearance | -0.0203255 | -0.0225666 | 0.0450645 |

## Active flow-time diagnostics

| method | t | x1_raw_collision | z_star_collision | correction_norm | normalized_correction_norm | velocity_change | z_star_min_clearance | z_star_goal_error | solver_success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L05_NO_REPLACE | 0.571429 | 0.0625 | 0.291667 | 1.74142 | 4.06331 | 2.03165 | 0.0119387 | 0.362415 | 1 |
| L05_NO_REPLACE | 0.714286 | 0.0989583 | 0.333333 | 1.50775 | 5.27712 | 2.63856 | 0.00671323 | 0.358294 | 1 |
| L05_NO_REPLACE | 0.857143 | 0.125 | 0.328125 | 1.28382 | 8.98671 | 4.49336 | 0.0120722 | 0.35488 | 1 |
| L05_REPLACE | 0.571429 | 0.0625 | 0.291667 | 1.74185 | 4.06431 | 2.03215 | 0.0119241 | 0.362472 | 1 |
| L05_REPLACE | 0.714286 | 0.0989583 | 0.333333 | 1.51081 | 5.28782 | 2.64391 | 0.00674503 | 0.358406 | 1 |
| L05_REPLACE | 0.857143 | 0.125 | 0.328125 | 1.28275 | 8.97927 | 4.48964 | 0.0120251 | 0.354661 | 1 |
| L1_NO_REPLACE | 0.571429 | 0.0625 | 0.291667 | 1.74155 | 4.06361 | 4.06361 | 0.0120476 | 0.362338 | 1 |
| L1_NO_REPLACE | 0.714286 | 0.109375 | 0.307292 | 1.46973 | 5.14405 | 5.14405 | 0.0101244 | 0.356661 | 1 |
| L1_NO_REPLACE | 0.857143 | 0.109375 | 0.333333 | 1.1747 | 8.22291 | 8.22291 | 0.0142662 | 0.352987 | 1 |
| L1_REPLACE | 0.571429 | 0.0625 | 0.286458 | 1.7533 | 4.09104 | 4.09104 | 0.011534 | 0.36587 | 1 |
| L1_REPLACE | 0.714286 | 0.104167 | 0.317708 | 1.48883 | 5.21091 | 5.21091 | 0.00922688 | 0.359048 | 1 |
| L1_REPLACE | 0.857143 | 0.114583 | 0.333333 | 1.17386 | 8.21702 | 8.21702 | 0.0142318 | 0.354502 | 1 |

## Terminal state audit

| method | state | collision_rate | min_clearance | goal_error |
| --- | --- | --- | --- | --- |
| L05_NO_REPLACE | hypothetical_l05_residual | 0.00520833 | 0.057124 | 0.465158 |
| L05_NO_REPLACE | hypothetical_l1_residual | 0.328125 | 0.0120722 | 0.35488 |
| L05_NO_REPLACE | x1_raw | 0.125 | 0.0638632 | 0.572941 |
| L05_NO_REPLACE | x_before_terminal | 0.765625 | -0.0368639 | 0.624229 |
| L05_NO_REPLACE | z_star | 0.328125 | 0.0120722 | 0.35488 |
| L05_REPLACE | hypothetical_l05_residual | 0.00520833 | 0.0570992 | 0.465068 |
| L05_REPLACE | hypothetical_l1_residual | 0.328125 | 0.0120251 | 0.354661 |
| L05_REPLACE | x1_raw | 0.125 | 0.0638688 | 0.572988 |
| L05_REPLACE | x_before_terminal | 0.765625 | -0.0368645 | 0.6243 |
| L05_REPLACE | z_star | 0.328125 | 0.0120251 | 0.354661 |
| L1_NO_REPLACE | hypothetical_l05_residual | 0 | 0.0564973 | 0.459292 |
| L1_NO_REPLACE | hypothetical_l1_residual | 0.333333 | 0.0142662 | 0.352987 |
| L1_NO_REPLACE | x1_raw | 0.109375 | 0.0614943 | 0.563506 |
| L1_NO_REPLACE | x_before_terminal | 0.880208 | -0.0557422 | 0.553306 |
| L1_NO_REPLACE | z_star | 0.333333 | 0.0142662 | 0.352987 |
| L1_REPLACE | hypothetical_l05_residual | 0 | 0.0568239 | 0.460121 |
| L1_REPLACE | hypothetical_l1_residual | 0.333333 | 0.0142318 | 0.354502 |
| L1_REPLACE | x1_raw | 0.114583 | 0.0615729 | 0.563947 |
| L1_REPLACE | x_before_terminal | 0.880208 | -0.0554792 | 0.554264 |
| L1_REPLACE | z_star | 0.333333 | 0.0142318 | 0.354502 |

The full 16x7 states for `x_before_terminal`, `x1_raw`, `z_star`, hypothetical
lambda=1 output, and hypothetical lambda=0.5 output are serialized in
`results/terminal_state_audit.csv`.

## Required answers

1. **Is lambda=1 independently harmful?** There is no evidence that the
   non-terminal strength change is the primary problem: among replacement methods,
   lambda=1 has 0.3333 collision versus
   0.3281 for lambda=0.5, only one trajectory apart.
   L1_NO_REPLACE (0.3333) cannot identify an independent
   all-step lambda effect because its final lambda=1 Euler update is exactly a
   replacement. Thus a clean independent terminal lambda=1 effect is structurally
   unavailable on this grid.
2. **Is terminal full replacement independently harmful?** Yes. At lambda=0.5,
   replacement changes collision from 0.0052
   (1/192) to 0.3281 (63/192). The paired table shows
   63 no-replacement-safe trajectories newly collide and one collision is rescued.
   Task 3 changes from 0/32 to 32/32 and task 5 from 0/32 to 31/32.
3. **Is there an interaction?** The nominal difference-in-differences is large,
   but it is not a clean interaction estimate because the lambda=1/no-replacement
   cell implements effective replacement at the final step. The data support a
   terminal-replacement main mechanism; they do not establish an additional
   lambda-by-replacement interaction.
4. **What caused the 64 YFLOW_MAIN_PORT new collisions?** Terminal replacement is
   the dominant identified cause. L05_REPLACE reproduces 63/64 aggregate collisions
   while keeping the first two active corrections damped. Increasing those earlier
   corrections to lambda=1 adds only the remaining one. This explains all task-3
   failures and 31/32 task-5 failures without full-strength early correction.
5. **Why can L05_NO_REPLACE be safe when z_star collides?** Its terminal z_star
   collision rate is 0.3281, while the hypothetical damped residual output is
   0.0052. The optimizer supplies a direction; retaining half of the learned
   Flow update interpolates away from unsafe z_star rather than consuming it whole.
6. **Recommended main robot-arm formulation.** Use L05_NO_REPLACE: it preserves
   learned Flow dynamics, consumes the optimizer only as a damped correction direction,
   and avoids the known terminal replacement risk. Adaptive gating remains a separate
   compute optimization demonstrated in the prior experiment, not part of this ablation.

## Scope and fidelity

- Same 6 tasks x 32 seeds, checkpoint, initial noises, obstacles, goals, and 7-step grid.
- Same Phase-1.5 no-P (`mu=0`) Acados terminal optimizer, `t_on=0.5`, and
  `lambda_oc=10`; exactly three optimizer calls for each factorial method.
- No adaptive gating, OOD evaluation, retraining, lambda tuning, or schedule change.
- Remote-main reference commit: `eeee5d2fee03f6c0bb593148727f0a92b94d3c97`.
- Prior Phase 1/1.5/2/3/hybrid artifacts were not overwritten.
