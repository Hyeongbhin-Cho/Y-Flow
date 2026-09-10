# POV robot-arm Phase-3 report

## Decision

**NO-GO BEFORE OOD** at the Phase-3A adaptive gate.

Final Phase-3 decision: **NO-GO for the current adaptive hard-violation
trigger**. This is not a NO-GO for always-on POV: the adaptive policy preserved
ID collision safety, but it did not achieve its efficiency objective, so the
predeclared protocol stops before OOD and makes no generalization claim.

```json
{
  "collision_within_2pp": true,
  "new_collisions_vs_always": 0,
  "new_collision_rate_from_always_safe": 0.0,
  "few_new_collisions": true,
  "projection_call_reduction": 0.03199404761904756,
  "calls_reduced_at_least_25pct": false,
  "latency_reduction": 0.018925983929992563,
  "latency_reduced_meaningfully": false,
  "solver_runtime_reduction": 0.01913080158747449,
  "goal_error_difference": 0.03582690862717314,
  "goal_material_worsening_tolerance": 0.05,
  "goal_not_materially_worse": true,
  "passed": false
}
```

## ID matched comparison

| method | metric | mean | std | median | ci95_low | ci95_high | numerator | denominator |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PLAIN_FM | collision | 0.151042 | 0.359026 | 0 | 0.0989583 | 0.208333 | 29 | 192 |
| PLAIN_FM | terminal_goal_error | 0.581712 | 0.29237 | 0.456701 | 0.542078 | 0.624474 | nan | nan |
| PLAIN_FM | planning_latency_sec | 0.0703553 | 0.00169387 | 0.070483 | 0.0701462 | 0.0706419 | nan | nan |
| PLAIN_FM | projection_calls | 0 | 0 | 0 | 0 | 0 | nan | nan |
| PLAIN_FM | projection_failure_rate | 0 | 0 | 0 | 0 | 0 | nan | nan |
| PLAIN_FM | zero_projection_trajectory | 1 | 0 | 1 | 1 | 1 | 192 | 192 |
| ORIGINAL_SAFEFLOWMPC | collision | 0 | 0 | 0 | 0 | 0 | 0 | 192 |
| ORIGINAL_SAFEFLOWMPC | terminal_goal_error | 0.880841 | 0.22932 | 0.940497 | 0.847615 | 0.913414 | nan | nan |
| ORIGINAL_SAFEFLOWMPC | planning_latency_sec | 0.188399 | 0.00285545 | 0.188975 | 0.187993 | 0.188794 | nan | nan |
| ORIGINAL_SAFEFLOWMPC | projection_calls | 7 | 0 | 7 | 7 | 7 | nan | nan |
| ORIGINAL_SAFEFLOWMPC | projection_failure_rate | 0 | 0 | 0 | 0 | 0 | nan | nan |
| ORIGINAL_SAFEFLOWMPC | zero_projection_trajectory | 0 | 0 | 0 | 0 | 0 | 0 | 192 |
| POV_L05_ALWAYS | collision | 0 | 0 | 0 | 0 | 0 | 0 | 192 |
| POV_L05_ALWAYS | terminal_goal_error | 0.43922 | 0.19768 | 0.377424 | 0.411026 | 0.468064 | nan | nan |
| POV_L05_ALWAYS | planning_latency_sec | 1.35739 | 0.209868 | 1.33646 | 1.32799 | 1.38681 | nan | nan |
| POV_L05_ALWAYS | projection_calls | 7 | 0 | 7 | 7 | 7 | nan | nan |
| POV_L05_ALWAYS | projection_failure_rate | 0 | 0 | 0 | 0 | 0 | nan | nan |
| POV_L05_ALWAYS | zero_projection_trajectory | 0 | 0 | 0 | 0 | 0 | 0 | 192 |
| POV_L05_ADAPTIVE | collision | 0 | 0 | 0 | 0 | 0 | 0 | 192 |
| POV_L05_ADAPTIVE | terminal_goal_error | 0.475047 | 0.248246 | 0.425836 | 0.438714 | 0.511699 | nan | nan |
| POV_L05_ADAPTIVE | planning_latency_sec | 1.3317 | 0.242345 | 1.33045 | 1.2966 | 1.36614 | nan | nan |
| POV_L05_ADAPTIVE | projection_calls | 6.77604 | 0.593904 | 7 | 6.6875 | 6.85417 | nan | nan |
| POV_L05_ADAPTIVE | projection_failure_rate | 0 | 0 | 0 | 0 | 0 | nan | nan |
| POV_L05_ADAPTIVE | zero_projection_trajectory | 0 | 0 | 0 | 0 | 0 | 0 | 192 |
| POV_L025_ADAPTIVE | collision | 0.0208333 | 0.1432 | 0 | 0.00520833 | 0.0416667 | 4 | 192 |
| POV_L025_ADAPTIVE | terminal_goal_error | 0.526544 | 0.277784 | 0.42317 | 0.488519 | 0.564971 | nan | nan |
| POV_L025_ADAPTIVE | planning_latency_sec | 1.33787 | 0.247546 | 1.33334 | 1.30302 | 1.37368 | nan | nan |
| POV_L025_ADAPTIVE | projection_calls | 6.75521 | 0.645138 | 7 | 6.66146 | 6.84375 | nan | nan |
| POV_L025_ADAPTIVE | projection_failure_rate | 0 | 0 | 0 | 0 | 0 | nan | nan |
| POV_L025_ADAPTIVE | zero_projection_trajectory | 0 | 0 | 0 | 0 | 0 | 0 | 192 |
| GOAL_AWARE_FINAL_PROJECTION | collision | 0.484375 | 0.501062 | 0 | 0.416667 | 0.552083 | 93 | 192 |
| GOAL_AWARE_FINAL_PROJECTION | terminal_goal_error | 0.386458 | 0.117702 | 0.38499 | 0.370039 | 0.403093 | nan | nan |
| GOAL_AWARE_FINAL_PROJECTION | planning_latency_sec | 0.180191 | 0.0238365 | 0.174442 | 0.177108 | 0.183808 | nan | nan |
| GOAL_AWARE_FINAL_PROJECTION | projection_calls | 1 | 0 | 1 | 1 | 1 | nan | nan |
| GOAL_AWARE_FINAL_PROJECTION | projection_failure_rate | 0 | 0 | 0 | 0 | 0 | nan | nan |
| GOAL_AWARE_FINAL_PROJECTION | zero_projection_trajectory | 0 | 0 | 0 | 0 | 0 | 0 | 192 |

## Paired collision transitions

| adaptive_method | reference_method | reference_state | adaptive_state | count |
| --- | --- | --- | --- | --- |
| POV_L05_ADAPTIVE | PLAIN_FM | safe | safe | 163 |
| POV_L05_ADAPTIVE | PLAIN_FM | safe | collision | 0 |
| POV_L05_ADAPTIVE | PLAIN_FM | collision | safe | 29 |
| POV_L05_ADAPTIVE | PLAIN_FM | collision | collision | 0 |
| POV_L05_ADAPTIVE | POV_L05_ALWAYS | safe | safe | 192 |
| POV_L05_ADAPTIVE | POV_L05_ALWAYS | safe | collision | 0 |
| POV_L05_ADAPTIVE | POV_L05_ALWAYS | collision | safe | 0 |
| POV_L05_ADAPTIVE | POV_L05_ALWAYS | collision | collision | 0 |
| POV_L025_ADAPTIVE | PLAIN_FM | safe | safe | 163 |
| POV_L025_ADAPTIVE | PLAIN_FM | safe | collision | 0 |
| POV_L025_ADAPTIVE | PLAIN_FM | collision | safe | 25 |
| POV_L025_ADAPTIVE | PLAIN_FM | collision | collision | 4 |
| POV_L025_ADAPTIVE | POV_L05_ALWAYS | safe | safe | 188 |
| POV_L025_ADAPTIVE | POV_L05_ALWAYS | safe | collision | 4 |
| POV_L025_ADAPTIVE | POV_L05_ALWAYS | collision | safe | 0 |
| POV_L025_ADAPTIVE | POV_L05_ALWAYS | collision | collision | 0 |

## Answers to the research questions

1. **Does adaptive POV preserve always-on safety?** Yes under the collision/new-collision checks, but the full adaptive gate still failed.
   Collision was 0.0000 adaptive versus 0.0000
   always-on; the paired table identifies newly introduced collisions.
2. **How many calls are saved?** Mean calls changed from
   7.0000 to 6.7760, a
   3.20% reduction.
3. **How much latency is saved?** Total latency changed from
   1.3574s to 1.3317s
   (1.89% reduction); measured solver-runtime
   reduction was 1.91%.
4. **Does POV outperform Plain FM under OOD?** Not tested because Phase 3A failed, so no generalization claim is made.
5. **Does it outperform original SafeFlowMPC?** On ID, adaptive POV collision was
   0.0000 versus 0.0000, and latency was
   1.3317s versus 0.1884s.
   Interpret this as a matched one-shot comparison; the upstream closed-loop
   planner remains semantically complementary rather than fully equivalent.
6. **Which Flow times trigger most?** The highest adaptive trigger rate was
   1.0000 at t=0; every t is listed below.
7. **Is last-step normalization problematic?** The flow table reports the raw and
   normalized final-step correction. The no-last ablation below was run only if
   the prespecified 1.5x disproportion criterion was met. The always-on mean normalized correction reached 21.4214 at t=6/7. Removing that step raised collision to 0.1250 and goal error to 0.5776, so the large correction is consequential but cannot simply be removed.
8. **Are Phase-2 gains robust beyond six ID examples?** Unknown; OOD was correctly stopped by the gate.

## Flow-time analysis

| method | t | endpoint_feasible_fraction | trigger_rate | mean_endpoint_violation | mean_raw_delta | mean_normalized_delta | mean_velocity_correction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| POV_L025_ADAPTIVE | 0 | 0 | 1 | 11.7139 | 3.83123 | 3.83123 | 0.957807 |
| POV_L025_ADAPTIVE | 0.142857 | 0 | 1 | 8.57658 | 3.74384 | 4.36781 | 1.09195 |
| POV_L025_ADAPTIVE | 0.285714 | 0 | 1 | 6.37382 | 3.64246 | 5.09945 | 1.27486 |
| POV_L025_ADAPTIVE | 0.428571 | 0 | 1 | 4.57077 | 3.64159 | 6.37278 | 1.59319 |
| POV_L025_ADAPTIVE | 0.571429 | 0.0260417 | 0.973958 | 3.1831 | 3.32396 | 7.7559 | 1.93897 |
| POV_L025_ADAPTIVE | 0.714286 | 0.0625 | 0.9375 | 1.93484 | 3.16384 | 11.0734 | 2.76836 |
| POV_L025_ADAPTIVE | 0.857143 | 0.15625 | 0.84375 | 1.01227 | 2.53214 | 17.725 | 4.43125 |
| POV_L05_ADAPTIVE | 0 | 0 | 1 | 11.7139 | 3.87385 | 3.87385 | 1.93692 |
| POV_L05_ADAPTIVE | 0.142857 | 0 | 1 | 8.60936 | 3.73628 | 4.35899 | 2.17949 |
| POV_L05_ADAPTIVE | 0.285714 | 0 | 1 | 6.37945 | 3.722 | 5.2108 | 2.6054 |
| POV_L05_ADAPTIVE | 0.428571 | 0.00520833 | 0.994792 | 4.63524 | 3.68546 | 6.44956 | 3.22478 |
| POV_L05_ADAPTIVE | 0.571429 | 0.0104167 | 0.989583 | 3.29771 | 3.48429 | 8.13001 | 4.06501 |
| POV_L05_ADAPTIVE | 0.714286 | 0.0572917 | 0.942708 | 2.20522 | 3.12557 | 10.9395 | 5.46974 |
| POV_L05_ADAPTIVE | 0.857143 | 0.151042 | 0.848958 | 1.32787 | 2.53779 | 17.7645 | 8.88227 |
| POV_L05_ALWAYS | 0 | 0 | 1 | 11.7139 | 3.8052 | 3.8052 | 1.9026 |
| POV_L05_ALWAYS | 0.142857 | 0 | 1 | 8.60498 | 3.78088 | 4.41103 | 2.20551 |
| POV_L05_ALWAYS | 0.285714 | 0 | 1 | 6.3763 | 3.72957 | 5.2214 | 2.6107 |
| POV_L05_ALWAYS | 0.428571 | 0.00520833 | 1 | 4.63919 | 3.63099 | 6.35424 | 3.17712 |
| POV_L05_ALWAYS | 0.571429 | 0.015625 | 1 | 3.28631 | 3.4998 | 8.16621 | 4.0831 |
| POV_L05_ALWAYS | 0.714286 | 0.0625 | 1 | 2.19838 | 3.31459 | 11.6011 | 5.80054 |
| POV_L05_ALWAYS | 0.857143 | 0.130208 | 1 | 1.31832 | 3.0602 | 21.4214 | 10.7107 |

The trigger was dominated by acceleration and velocity violations, not only
geometric collision: overall trigger-component rates were collision
0.0781, joint
0.0000, velocity
0.6890, and acceleration
0.9621. This explains why the
hard-safety trigger skipped only 3.20% of calls.

## Optional no-last diagnostic

Exploratory only; not part of the predeclared primary comparison.

| method | collision | goal_error | latency | calls | distortion |
| --- | --- | --- | --- | --- | --- |
| POV_L05_ADAPTIVE_NO_LAST | 0.125 | 0.577558 | 1.25814 | 5.92188 | 0.508691 |

## Original SafeFlowMPC baseline semantics

The matched baseline uses the unchanged upstream `SafetyFilterAcados.step` seven
times inside one Flow Matching sample and a line-for-line headless extraction of
`SafeFlowMPC._compute_guidance`. It uses the same unsafe FM checkpoint and matched
initial noise as every other ID method. This is a one-shot comparison.

The repository's top-level `inference_global_planner.py` has different semantics:
it is a closed-loop/receding-horizon simulator, shifts the prior solution between
control timesteps, and selects the safe FM checkpoint by default. It was reproduced
separately in Phase 1, but that single rollout is not mixed into this matched table.

## OOD status

Phase 3B was not run because the Phase-3A gate failed. Empty OOD CSV schemas are retained only to make the stopped state machine-readable.

The requested OOD plot and visualization paths contain explicit `NOT RUN`
placeholders; they are not experimental observations.

## Phase-2 reproduction note

Plain FM reproduced 29/192 collisions and final projection reproduced 93/192.
The current always-on lambda=0.5 run produced 0/192 versus Phase 2's 1/192. The
single prior collision had only -2.56e-5 m clearance and is now +1.94e-3 m, so
this is a near-boundary Acados numerical/interleaving drift rather than evidence
of a material safety change. The adaptive gate uses the current matched
always-on run; its 3.20% call reduction remains far below 25% under either count.

## Scope

- No Flow Matching retraining occurred.
- Goal error was never part of the adaptive trigger.
- Failed projections use vanilla velocity; unconverged iterates are discarded.
- No schedule was tuned from the flow-time analysis.
