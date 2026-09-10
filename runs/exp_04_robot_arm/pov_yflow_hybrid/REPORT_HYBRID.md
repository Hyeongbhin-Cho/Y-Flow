# Robot-arm Y-Flow / POV hybrid report

## Primary results

| method | metric | mean | ci95_low | ci95_high | numerator | denominator |
| --- | --- | --- | --- | --- | --- | --- |
| PLAIN_FM | collision | 0.151042 | 0.104167 | 0.203125 | 29 | 192 |
| PLAIN_FM | min_clearance | 0.0660799 | 0.0593572 | 0.0724806 |  |  |
| PLAIN_FM | joint_limit_violation | 0 | 0 | 0 |  |  |
| PLAIN_FM | velocity_violation | 0.141085 | 0.124405 | 0.157436 |  |  |
| PLAIN_FM | acceleration_violation | 0.869476 | 0.745149 | 0.996321 |  |  |
| PLAIN_FM | terminal_goal_error | 0.581712 | 0.54315 | 0.624617 |  |  |
| PLAIN_FM | joint_path_length | 2.50436 | 2.39079 | 2.61354 |  |  |
| PLAIN_FM | smoothness | 0.0400362 | 0.0385187 | 0.0415237 |  |  |
| PLAIN_FM | trajectory_distortion_vs_plain | 0 | 0 | 0 |  |  |
| PLAIN_FM | planning_latency_sec | 0.070263 | 0.0700835 | 0.0705425 |  |  |
| PLAIN_FM | fm_latency_sec | 0.0436347 | 0.0435624 | 0.0437064 |  |  |
| PLAIN_FM | optimizer_latency_sec | 0 | 0 | 0 |  |  |
| PLAIN_FM | safety_trigger_latency_sec | 0 | 0 | 0 |  |  |
| PLAIN_FM | projection_calls | 0 | 0 | 0 |  |  |
| PLAIN_FM | failed_projection_solves | 0 | 0 | 0 |  |  |
| PLAIN_FM | projection_failure_rate | 0 | 0 | 0 |  |  |
| POV_L05_ALWAYS | collision | 0.00520833 | 0 | 0.015625 | 1 | 192 |
| POV_L05_ALWAYS | min_clearance | 0.0494099 | 0.0459651 | 0.0527057 |  |  |
| POV_L05_ALWAYS | joint_limit_violation | 0 | 0 | 0 |  |  |
| POV_L05_ALWAYS | velocity_violation | 0.00991496 | 0.00667589 | 0.0135132 |  |  |
| POV_L05_ALWAYS | acceleration_violation | 0.0258174 | 0.00841605 | 0.0483569 |  |  |
| POV_L05_ALWAYS | terminal_goal_error | 0.438917 | 0.410958 | 0.466922 |  |  |
| POV_L05_ALWAYS | joint_path_length | 1.90812 | 1.80486 | 2.00873 |  |  |
| POV_L05_ALWAYS | smoothness | 0.0201815 | 0.0192129 | 0.0211244 |  |  |
| POV_L05_ALWAYS | trajectory_distortion_vs_plain | 1.96006 | 1.90376 | 2.01392 |  |  |
| POV_L05_ALWAYS | planning_latency_sec | 1.37328 | 1.34295 | 1.40278 |  |  |
| POV_L05_ALWAYS | fm_latency_sec | 0.047977 | 0.0478625 | 0.0480963 |  |  |
| POV_L05_ALWAYS | optimizer_latency_sec | 1.243 | 1.21286 | 1.27398 |  |  |
| POV_L05_ALWAYS | safety_trigger_latency_sec | 0 | 0 | 0 |  |  |
| POV_L05_ALWAYS | projection_calls | 7 | 7 | 7 |  |  |
| POV_L05_ALWAYS | failed_projection_solves | 0 | 0 | 0 |  |  |
| POV_L05_ALWAYS | projection_failure_rate | 0 | 0 | 0 |  |  |
| YFLOW_MAIN_PORT | collision | 0.333333 | 0.270703 | 0.401042 | 64 | 192 |
| YFLOW_MAIN_PORT | min_clearance | 0.0143391 | 0.00924748 | 0.0195453 |  |  |
| YFLOW_MAIN_PORT | joint_limit_violation | 6.06403e-10 | 0 | 1.81921e-09 |  |  |
| YFLOW_MAIN_PORT | velocity_violation | 0.000468419 | 0.000287847 | 0.000671823 |  |  |
| YFLOW_MAIN_PORT | acceleration_violation | 0 | 0 | 0 |  |  |
| YFLOW_MAIN_PORT | terminal_goal_error | 0.354574 | 0.336092 | 0.372263 |  |  |
| YFLOW_MAIN_PORT | joint_path_length | 2.0114 | 1.93225 | 2.09362 |  |  |
| YFLOW_MAIN_PORT | smoothness | 0.0229115 | 0.0218424 | 0.0240012 |  |  |
| YFLOW_MAIN_PORT | trajectory_distortion_vs_plain | 1.46733 | 1.41809 | 1.51693 |  |  |
| YFLOW_MAIN_PORT | planning_latency_sec | 0.478574 | 0.465483 | 0.492178 |  |  |
| YFLOW_MAIN_PORT | fm_latency_sec | 0.0453823 | 0.0453069 | 0.0454599 |  |  |
| YFLOW_MAIN_PORT | optimizer_latency_sec | 0.381739 | 0.368841 | 0.39588 |  |  |
| YFLOW_MAIN_PORT | safety_trigger_latency_sec | 0 | 0 | 0 |  |  |
| YFLOW_MAIN_PORT | projection_calls | 3 | 3 | 3 |  |  |
| YFLOW_MAIN_PORT | failed_projection_solves | 0 | 0 | 0 |  |  |
| YFLOW_MAIN_PORT | projection_failure_rate | 0 | 0 | 0 |  |  |
| HYBRID_LATE_DAMPED | collision | 0.00520833 | 0 | 0.0208333 | 1 | 192 |
| HYBRID_LATE_DAMPED | min_clearance | 0.0571041 | 0.0531759 | 0.0610322 |  |  |
| HYBRID_LATE_DAMPED | joint_limit_violation | 0 | 0 | 0 |  |  |
| HYBRID_LATE_DAMPED | velocity_violation | 0.0188388 | 0.0142064 | 0.0235901 |  |  |
| HYBRID_LATE_DAMPED | acceleration_violation | 0.00663169 | 2.4531e-05 | 0.017027 |  |  |
| HYBRID_LATE_DAMPED | terminal_goal_error | 0.46506 | 0.437997 | 0.49254 |  |  |
| HYBRID_LATE_DAMPED | joint_path_length | 2.18767 | 2.09251 | 2.28447 |  |  |
| HYBRID_LATE_DAMPED | smoothness | 0.0201209 | 0.019329 | 0.0208775 |  |  |
| HYBRID_LATE_DAMPED | trajectory_distortion_vs_plain | 0.78669 | 0.75767 | 0.815955 |  |  |
| HYBRID_LATE_DAMPED | planning_latency_sec | 0.480784 | 0.468318 | 0.493865 |  |  |
| HYBRID_LATE_DAMPED | fm_latency_sec | 0.0453854 | 0.0453227 | 0.0454505 |  |  |
| HYBRID_LATE_DAMPED | optimizer_latency_sec | 0.383801 | 0.370295 | 0.396476 |  |  |
| HYBRID_LATE_DAMPED | safety_trigger_latency_sec | 0 | 0 | 0 |  |  |
| HYBRID_LATE_DAMPED | projection_calls | 3 | 3 | 3 |  |  |
| HYBRID_LATE_DAMPED | failed_projection_solves | 0 | 0 | 0 |  |  |
| HYBRID_LATE_DAMPED | projection_failure_rate | 0 | 0 | 0 |  |  |
| HYBRID_LATE_ADAPTIVE | collision | 0.00520833 | 0 | 0.015625 | 1 | 192 |
| HYBRID_LATE_ADAPTIVE | min_clearance | 0.0636471 | 0.058311 | 0.0689949 |  |  |
| HYBRID_LATE_ADAPTIVE | joint_limit_violation | 0 | 0 | 0 |  |  |
| HYBRID_LATE_ADAPTIVE | velocity_violation | 0.0188681 | 0.0144322 | 0.0239868 |  |  |
| HYBRID_LATE_ADAPTIVE | acceleration_violation | 0.00663169 | 2.4531e-05 | 0.0172218 |  |  |
| HYBRID_LATE_ADAPTIVE | terminal_goal_error | 0.50627 | 0.470285 | 0.54395 |  |  |
| HYBRID_LATE_ADAPTIVE | joint_path_length | 2.19927 | 2.10349 | 2.29267 |  |  |
| HYBRID_LATE_ADAPTIVE | smoothness | 0.0220478 | 0.0213978 | 0.0227001 |  |  |
| HYBRID_LATE_ADAPTIVE | trajectory_distortion_vs_plain | 0.670849 | 0.624994 | 0.716027 |  |  |
| HYBRID_LATE_ADAPTIVE | planning_latency_sec | 0.454757 | 0.436011 | 0.474125 |  |  |
| HYBRID_LATE_ADAPTIVE | fm_latency_sec | 0.0453254 | 0.0452437 | 0.0454172 |  |  |
| HYBRID_LATE_ADAPTIVE | optimizer_latency_sec | 0.359922 | 0.341838 | 0.377971 |  |  |
| HYBRID_LATE_ADAPTIVE | safety_trigger_latency_sec | 0.0110295 | 0.0110144 | 0.0110446 |  |  |
| HYBRID_LATE_ADAPTIVE | projection_calls | 2.75 | 2.66146 | 2.83854 |  |  |
| HYBRID_LATE_ADAPTIVE | failed_projection_solves | 0 | 0 | 0 |  |  |
| HYBRID_LATE_ADAPTIVE | projection_failure_rate | 0 | 0 | 0 |  |  |
| ADAPTIVE_ALLTIME_L05 | collision | 0 | 0 | 0 | 0 | 192 |
| ADAPTIVE_ALLTIME_L05 | min_clearance | 0.0629073 | 0.0578813 | 0.0679449 |  |  |
| ADAPTIVE_ALLTIME_L05 | joint_limit_violation | 0 | 0 | 0 |  |  |
| ADAPTIVE_ALLTIME_L05 | velocity_violation | 0.0159234 | 0.0117662 | 0.0201487 |  |  |
| ADAPTIVE_ALLTIME_L05 | acceleration_violation | 0.00652429 | 0 | 0.0150039 |  |  |
| ADAPTIVE_ALLTIME_L05 | terminal_goal_error | 0.50092 | 0.466006 | 0.535816 |  |  |
| ADAPTIVE_ALLTIME_L05 | joint_path_length | 2.17545 | 2.0804 | 2.26769 |  |  |
| ADAPTIVE_ALLTIME_L05 | smoothness | 0.0222089 | 0.0214957 | 0.0229936 |  |  |
| ADAPTIVE_ALLTIME_L05 | trajectory_distortion_vs_plain | 0.737443 | 0.690154 | 0.786558 |  |  |
| ADAPTIVE_ALLTIME_L05 | planning_latency_sec | 1.3578 | 1.32163 | 1.39327 |  |  |
| ADAPTIVE_ALLTIME_L05 | fm_latency_sec | 0.0486146 | 0.0484685 | 0.0487723 |  |  |
| ADAPTIVE_ALLTIME_L05 | optimizer_latency_sec | 1.22584 | 1.1897 | 1.26254 |  |  |
| ADAPTIVE_ALLTIME_L05 | safety_trigger_latency_sec | 0.0259669 | 0.0259247 | 0.0260152 |  |  |
| ADAPTIVE_ALLTIME_L05 | projection_calls | 6.77604 | 6.69271 | 6.85938 |  |  |
| ADAPTIVE_ALLTIME_L05 | failed_projection_solves | 0 | 0 | 0 |  |  |
| ADAPTIVE_ALLTIME_L05 | projection_failure_rate | 0 | 0 | 0 |  |  |

## Mandatory paired collision analysis

| method | plain_collisions_rescued | plain_safe_newly_broken | always_safe_newly_broken |
| --- | --- | --- | --- |
| POV_L05_ALWAYS | 28 | 0 |  |
| YFLOW_MAIN_PORT | 29 | 64 |  |
| HYBRID_LATE_DAMPED | 28 | 0 | 1 |
| HYBRID_LATE_ADAPTIVE | 28 | 0 | 1 |
| ADAPTIVE_ALLTIME_L05 | 29 | 0 | 0 |

Full 2x2 counts:

| method | reference_method | reference_state | method_state | count |
| --- | --- | --- | --- | --- |
| POV_L05_ALWAYS | PLAIN_FM | safe | safe | 163 |
| POV_L05_ALWAYS | PLAIN_FM | safe | collision | 0 |
| POV_L05_ALWAYS | PLAIN_FM | collision | safe | 28 |
| POV_L05_ALWAYS | PLAIN_FM | collision | collision | 1 |
| YFLOW_MAIN_PORT | PLAIN_FM | safe | safe | 99 |
| YFLOW_MAIN_PORT | PLAIN_FM | safe | collision | 64 |
| YFLOW_MAIN_PORT | PLAIN_FM | collision | safe | 29 |
| YFLOW_MAIN_PORT | PLAIN_FM | collision | collision | 0 |
| HYBRID_LATE_DAMPED | PLAIN_FM | safe | safe | 163 |
| HYBRID_LATE_DAMPED | PLAIN_FM | safe | collision | 0 |
| HYBRID_LATE_DAMPED | PLAIN_FM | collision | safe | 28 |
| HYBRID_LATE_DAMPED | PLAIN_FM | collision | collision | 1 |
| HYBRID_LATE_DAMPED | POV_L05_ALWAYS | safe | safe | 190 |
| HYBRID_LATE_DAMPED | POV_L05_ALWAYS | safe | collision | 1 |
| HYBRID_LATE_DAMPED | POV_L05_ALWAYS | collision | safe | 1 |
| HYBRID_LATE_DAMPED | POV_L05_ALWAYS | collision | collision | 0 |
| HYBRID_LATE_ADAPTIVE | PLAIN_FM | safe | safe | 163 |
| HYBRID_LATE_ADAPTIVE | PLAIN_FM | safe | collision | 0 |
| HYBRID_LATE_ADAPTIVE | PLAIN_FM | collision | safe | 28 |
| HYBRID_LATE_ADAPTIVE | PLAIN_FM | collision | collision | 1 |
| HYBRID_LATE_ADAPTIVE | POV_L05_ALWAYS | safe | safe | 190 |
| HYBRID_LATE_ADAPTIVE | POV_L05_ALWAYS | safe | collision | 1 |
| HYBRID_LATE_ADAPTIVE | POV_L05_ALWAYS | collision | safe | 1 |
| HYBRID_LATE_ADAPTIVE | POV_L05_ALWAYS | collision | collision | 0 |
| ADAPTIVE_ALLTIME_L05 | PLAIN_FM | safe | safe | 163 |
| ADAPTIVE_ALLTIME_L05 | PLAIN_FM | safe | collision | 0 |
| ADAPTIVE_ALLTIME_L05 | PLAIN_FM | collision | safe | 29 |
| ADAPTIVE_ALLTIME_L05 | PLAIN_FM | collision | collision | 0 |
| ADAPTIVE_ALLTIME_L05 | POV_L05_ALWAYS | safe | safe | 191 |
| ADAPTIVE_ALLTIME_L05 | POV_L05_ALWAYS | safe | collision | 0 |
| ADAPTIVE_ALLTIME_L05 | POV_L05_ALWAYS | collision | safe | 1 |
| ADAPTIVE_ALLTIME_L05 | POV_L05_ALWAYS | collision | collision | 0 |

## Per-task analysis

| task_id | method | collision_rate | goal_error | min_clearance | optimization_calls |
| --- | --- | --- | --- | --- | --- |
| 0 | ADAPTIVE_ALLTIME_L05 | 0 | 0.428879 | 0.0403928 | 7 |
| 0 | HYBRID_LATE_ADAPTIVE | 0 | 0.42721 | 0.041739 | 3 |
| 0 | HYBRID_LATE_DAMPED | 0 | 0.42721 | 0.041739 | 3 |
| 0 | PLAIN_FM | 0 | 0.447589 | 0.0675023 | 0 |
| 0 | POV_L05_ALWAYS | 0 | 0.433645 | 0.0406376 | 7 |
| 0 | YFLOW_MAIN_PORT | 0 | 0.415852 | 0.0338531 | 3 |
| 1 | ADAPTIVE_ALLTIME_L05 | 0 | 0.331771 | 0.0324447 | 6.875 |
| 1 | HYBRID_LATE_ADAPTIVE | 0 | 0.340482 | 0.0322754 | 2.90625 |
| 1 | HYBRID_LATE_DAMPED | 0 | 0.327635 | 0.0330112 | 3 |
| 1 | PLAIN_FM | 0 | 0.472374 | 0.0254297 | 0 |
| 1 | POV_L05_ALWAYS | 0 | 0.307009 | 0.0327494 | 7 |
| 1 | YFLOW_MAIN_PORT | 0 | 0.19367 | 0.0233912 | 3 |
| 2 | ADAPTIVE_ALLTIME_L05 | 0 | 0.357619 | 0.0688532 | 7 |
| 2 | HYBRID_LATE_ADAPTIVE | 0 | 0.358595 | 0.0694211 | 3 |
| 2 | HYBRID_LATE_DAMPED | 0 | 0.358595 | 0.0694211 | 3 |
| 2 | PLAIN_FM | 0 | 0.393215 | 0.0755328 | 0 |
| 2 | POV_L05_ALWAYS | 0 | 0.344217 | 0.0513761 | 7 |
| 2 | YFLOW_MAIN_PORT | 0 | 0.340921 | 0.0566219 | 3 |
| 3 | ADAPTIVE_ALLTIME_L05 | 0 | 0.720679 | 0.103088 | 7 |
| 3 | HYBRID_LATE_ADAPTIVE | 0 | 0.724196 | 0.102867 | 3 |
| 3 | HYBRID_LATE_DAMPED | 0 | 0.724196 | 0.102867 | 3 |
| 3 | PLAIN_FM | 0 | 0.9141 | 0.106214 | 0 |
| 3 | POV_L05_ALWAYS | 0 | 0.644074 | 0.0739573 | 7 |
| 3 | YFLOW_MAIN_PORT | 1 | 0.526733 | -0.0263235 | 3 |
| 4 | ADAPTIVE_ALLTIME_L05 | 0 | 0.227784 | 0.0223627 | 7 |
| 4 | HYBRID_LATE_ADAPTIVE | 0.03125 | 0.219692 | 0.0201622 | 3 |
| 4 | HYBRID_LATE_DAMPED | 0.03125 | 0.219641 | 0.020191 | 3 |
| 4 | PLAIN_FM | 0.90625 | 0.226989 | -0.00695405 | 0 |
| 4 | POV_L05_ALWAYS | 0.03125 | 0.170745 | 0.0150853 | 7 |
| 4 | YFLOW_MAIN_PORT | 0 | 0.20289 | 0.0379931 | 3 |
| 5 | ADAPTIVE_ALLTIME_L05 | 0 | 0.938786 | 0.110303 | 5.78125 |
| 5 | HYBRID_LATE_ADAPTIVE | 0 | 0.967443 | 0.115417 | 1.59375 |
| 5 | HYBRID_LATE_DAMPED | 0 | 0.733081 | 0.0753945 | 3 |
| 5 | PLAIN_FM | 0 | 1.036 | 0.128755 | 0 |
| 5 | POV_L05_ALWAYS | 0 | 0.733814 | 0.0826539 | 7 |
| 5 | YFLOW_MAIN_PORT | 1 | 0.447376 | -0.0395014 | 3 |

Tasks 3 and 5 test whether damping/gating prevents the Phase-1.5 new
collisions. Task 4 tests whether the hybrids retain the difficult-trajectory
rescue. Conclusions are based on the table, not aggregate collision alone.

## Flow-time diagnostics

| method | t | predicted_endpoint_true_collision_rate | adaptive_trigger_rate | optimization_rate | mean_correction_norm | mean_normalized_correction_norm | mean_velocity_change | true_collision_rate_after_optimization | solver_success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADAPTIVE_ALLTIME_L05 | 0 | 0.0416667 | 1 | 1 | 4.7307 | 4.7307 | 2.36535 | 0.380208 | 1 |
| ADAPTIVE_ALLTIME_L05 | 0.142857 | 0.0520833 | 1 | 1 | 3.55599 | 4.14866 | 2.07433 | 0.307292 | 1 |
| ADAPTIVE_ALLTIME_L05 | 0.285714 | 0.0520833 | 1 | 1 | 2.5974 | 3.63636 | 1.81818 | 0.208333 | 1 |
| ADAPTIVE_ALLTIME_L05 | 0.428571 | 0.0572917 | 1 | 1 | 2.0195 | 3.53413 | 1.76707 | 0.213542 | 1 |
| ADAPTIVE_ALLTIME_L05 | 0.571429 | 0.0833333 | 0.984375 | 0.984375 | 1.65058 | 3.85135 | 1.92567 | 0.253968 | 1 |
| ADAPTIVE_ALLTIME_L05 | 0.714286 | 0.119792 | 0.932292 | 0.932292 | 1.32822 | 4.64878 | 2.32439 | 0.26257 | 1 |
| ADAPTIVE_ALLTIME_L05 | 0.857143 | 0.130208 | 0.859375 | 0.859375 | 1.03618 | 7.25325 | 3.62663 | 0.242424 | 1 |
| HYBRID_LATE_ADAPTIVE | 0 | 0.0416667 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| HYBRID_LATE_ADAPTIVE | 0.142857 | 0.0520833 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| HYBRID_LATE_ADAPTIVE | 0.285714 | 0.0520833 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| HYBRID_LATE_ADAPTIVE | 0.428571 | 0.0520833 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| HYBRID_LATE_ADAPTIVE | 0.571429 | 0.0625 | 0.973958 | 0.973958 | 1.68998 | 3.94329 | 1.97164 | 0.272727 | 1 |
| HYBRID_LATE_ADAPTIVE | 0.714286 | 0.0989583 | 0.927083 | 0.927083 | 1.38137 | 4.8348 | 2.4174 | 0.280899 | 1 |
| HYBRID_LATE_ADAPTIVE | 0.857143 | 0.125 | 0.848958 | 0.848958 | 1.06124 | 7.4287 | 3.71435 | 0.226994 | 1 |
| HYBRID_LATE_DAMPED | 0 | 0.0416667 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| HYBRID_LATE_DAMPED | 0.142857 | 0.0520833 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| HYBRID_LATE_DAMPED | 0.285714 | 0.0520833 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| HYBRID_LATE_DAMPED | 0.428571 | 0.0520833 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| HYBRID_LATE_DAMPED | 0.571429 | 0.0625 | 0 | 1 | 1.74155 | 4.06361 | 2.03181 | 0.291667 | 1 |
| HYBRID_LATE_DAMPED | 0.714286 | 0.0989583 | 0 | 1 | 1.50721 | 5.27525 | 2.63763 | 0.333333 | 1 |
| HYBRID_LATE_DAMPED | 0.857143 | 0.125 | 0 | 1 | 1.28316 | 8.98213 | 4.49106 | 0.328125 | 1 |
| PLAIN_FM | 0 | 0.0416667 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| PLAIN_FM | 0.142857 | 0.0520833 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| PLAIN_FM | 0.285714 | 0.0520833 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| PLAIN_FM | 0.428571 | 0.0520833 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| PLAIN_FM | 0.571429 | 0.0625 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| PLAIN_FM | 0.714286 | 0.0989583 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| PLAIN_FM | 0.857143 | 0.151042 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| POV_L05_ALWAYS | 0 | 0.0416667 | 0 | 1 | 3.82442 | 3.82442 | 1.91221 | 0.328125 | 1 |
| POV_L05_ALWAYS | 0.142857 | 0.0520833 | 0 | 1 | 3.7581 | 4.38445 | 2.19223 | 0.307292 | 1 |
| POV_L05_ALWAYS | 0.285714 | 0.0520833 | 0 | 1 | 3.66475 | 5.13065 | 2.56533 | 0.265625 | 1 |
| POV_L05_ALWAYS | 0.428571 | 0.0625 | 0 | 1 | 3.65044 | 6.38828 | 3.19414 | 0.234375 | 1 |
| POV_L05_ALWAYS | 0.571429 | 0.0833333 | 0 | 1 | 3.5399 | 8.25977 | 4.12989 | 0.276042 | 1 |
| POV_L05_ALWAYS | 0.714286 | 0.130208 | 0 | 1 | 3.32193 | 11.6268 | 5.81338 | 0.369792 | 1 |
| POV_L05_ALWAYS | 0.857143 | 0.130208 | 0 | 1 | 3.06058 | 21.424 | 10.712 | 0.333333 | 1 |
| YFLOW_MAIN_PORT | 0 | 0.0416667 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| YFLOW_MAIN_PORT | 0.142857 | 0.0520833 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| YFLOW_MAIN_PORT | 0.285714 | 0.0520833 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| YFLOW_MAIN_PORT | 0.428571 | 0.0520833 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| YFLOW_MAIN_PORT | 0.571429 | 0.0625 | 0 | 1 | 1.75271 | 4.08966 | 4.08966 | 0.286458 | 1 |
| YFLOW_MAIN_PORT | 0.714286 | 0.104167 | 0 | 1 | 1.48989 | 5.21462 | 5.21462 | 0.317708 | 1 |
| YFLOW_MAIN_PORT | 0.857143 | 0.114583 | 0 | 1 | 1.17425 | 8.21972 | 8.21972 | 0.333333 | 1 |

HYBRID_LATE_ADAPTIVE number of triggered late steps per trajectory:

| triggered_late_steps | trajectory_count |
| --- | --- |
| 0 | 5 |
| 1 | 9 |
| 2 | 15 |
| 3 | 163 |

## Controlled-ablation interpretation

1. **Late full-strength versus late damping.** YFLOW_MAIN_PORT collision was
   0.3333; HYBRID_LATE_DAMPED was 0.0052. They share the
   three-step schedule and terminal optimizer, so this contrast measures replacing
   full target following with lambda=0.5 residual guidance. It changes correction
   strength at all three active steps, including removal of terminal full
   replacement; those two sub-effects are not separately identifiable here.
2. **Unconditional versus safety-gated late damping.** HYBRID_LATE_DAMPED
   collision was 0.0052 with 3.0000 calls;
   HYBRID_LATE_ADAPTIVE was 0.0052 with
   2.7500 calls. This isolates conditional correction of
   already-safe late endpoints.
3. **Time gate versus safety gate.** HYBRID_LATE_ADAPTIVE collision was
   0.0052; ADAPTIVE_ALLTIME_L05 was 0.0000. Their trigger
   is identical, so this contrast isolates the `t_on=0.5` restriction.
4. **Reference safety.** POV_L05_ALWAYS collision was 0.0052
   with 7.0000 calls and 1.3733s
   latency.

## Predeclared decision gates

```json
{
  "late_damped_success": {
    "pass": true,
    "checks": {
      "collision_below_yflow": true,
      "new_collisions_at_least_halved_vs_yflow": true,
      "mean_calls_equal_3": true,
      "goal_error_not_above_plain": true
    },
    "plain_safe_newly_broken": 0,
    "yflow_plain_safe_newly_broken": 64
  },
  "late_adaptive_promising": {
    "pass": true,
    "checks": {
      "collision_rate_le_5pct": true,
      "plain_safe_newly_broken_le_2": true,
      "mean_calls_le_3": true,
      "planning_latency_lt_0p8_sec": true,
      "goal_error_not_above_plain": true,
      "solver_failures_explicitly_recorded": true
    },
    "plain_safe_newly_broken": 0
  },
  "late_adaptive_strong_success": {
    "pass": true,
    "checks": {
      "collision_rate_le_2pct": true,
      "zero_plain_safe_newly_broken": true,
      "call_reduction_ge_50pct_vs_always": true,
      "latency_reduction_ge_40pct_vs_always": true
    },
    "call_reduction_vs_always": 0.6071428571428572,
    "latency_reduction_vs_always": 0.6688520691243545
  }
}
```

## Direct answers

- **RQ1 — Did damping fix YFLOW_MAIN_PORT?** Collision changed from
  0.3333 to 0.0052; the predeclared late-damped gate is
  **PASS**.
- **RQ2 — Did adaptive gating avoid the task-3/task-5 failure mode?** Both late
  damping methods reduced tasks 3 and 5 from 100% collision under YFLOW_MAIN_PORT
  to 0%. Adaptive gating did not improve collision over unconditional late
  damping: both had 0.0052 overall and introduced zero Plain-safe
  collisions. Its demonstrated benefit is computational: calls fell from
  3.0000 to 2.7500.
- **RQ3 — Can late adaptive approach always-POV safety below seven calls?** Yes.
  Both had 0.0052 collision, while late adaptive used
  2.7500 versus 7.0000 calls and
  0.4548s versus 1.3733s. The
  predeclared strong gate is
  **PASS**.
- **RQ4 — Does the time gate help or hurt?** Relative to all-time adaptive, the
  late gate changed collision from 0.0000 to 0.0052,
  calls from 6.7760 to 2.7500, and latency
  from 1.3578s to 0.4548s. Thus it
  provides the large efficiency gain but costs one collision in this sample.

## Final causal determination (A-E)

- **A, late-only correction:** not the main Phase-1.5 failure. Late damped alone
  reached 0.0052; however, the all-time adaptive result indicates the
  late gate accounts for the remaining one collision/192 relative to all-time.
- **B + D, full-strength correction and terminal full replacement:** the dominant
  jointly identified factor. Replacing this combined behavior with damped residual
  guidance changed collision from 0.3333 to 0.0052 and
  eliminated all 64 YFlow-induced Plain-safe collisions. The fixed ablation set
  cannot distinguish B from D individually.
- **C, unconditional modification of safe endpoints:** not a primary safety cause
  here. Gating changed neither aggregate collisions nor Plain-safe new collisions;
  it reduced calls by 0.2500 per trajectory.
- **E, interaction:** no evidence that safety gating must interact with B/D to fix
  Phase 1.5, because unconditional late damping already fixed tasks 3 and 5. An
  interaction internal to B versus D remains possible but is not identifiable.

## Fidelity and scope

- The Phase-1.5 robot terminal optimizer is reused unchanged: Y-Flow no-P path
  (`mu=0`), time-varying `lambda_oc*t^2/dt`, goal/smoothness cost and Acados
  robot constraints.
- Hybrid methods use Phase-2 lambda=0.5 residual velocity guidance and never
  replace the final state with `z_star`.
- Adaptive triggers use true evaluation geometry and exclude terminal goal error.
- Remote-main reference commit: `eeee5d2fee03f6c0bb593148727f0a92b94d3c97`.
- No model retraining, learned safety predictor, lambda sweep or post-result
  schedule tuning was performed.
- Prior Phase 1, Phase 1.5, Phase 2 and Phase 3 artifacts were not overwritten.
