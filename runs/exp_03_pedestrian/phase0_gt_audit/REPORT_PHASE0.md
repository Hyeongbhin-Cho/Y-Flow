# Phase 0. GT kinematic constraint audit

## Setting

- Data: `datasets/pedestrian/default/`, 8 past + 12 future frames, $\Delta t=0.4$ s
- Speed terms: $\|p_k-p_{k-1}\|/\Delta t$, $k=1..12$, $p_0$ = last observed position
- Acceleration terms: $\|v_k-v_{k-1}\|/\Delta t$, $k=1..12$, $v_0$ = last observed velocity
- Limits from each subset's **train** split only
  - `traj`: quantile of the per-trajectory maximum
  - `elem`: quantile over all 12 terms
- Reproduce: `COMMAND=audit ./run_exp_03_pedestrian.sh`

## Results (quantile 99.5, fraction of GT trajectories violating any term)

| subset | calib | $v_{\max}$ | $a_{\max}$ | train | test |
| --- | --- | --- | --- | --- | --- |
| eth | traj | 1.82 | 2.34 | 0.009 | 0.564 |
| hotel | traj | 1.85 | 2.38 | 0.009 | 0.043 |
| univ | traj | 2.09 | 2.62 | 0.007 | 0.003 |
| zara1 | traj | 1.88 | 2.38 | 0.009 | 0.006 |
| zara2 | traj | 1.88 | 2.38 | 0.009 | 0.004 |
| eth | elem | 1.70 | 1.43 | 0.063 | 0.663 |
| hotel | elem | 1.71 | 1.44 | 0.059 | 0.217 |
| univ | elem | 1.84 | 1.45 | 0.049 | 0.055 |
| zara1 | elem | 1.71 | 1.47 | 0.059 | 0.019 |
| zara2 | elem | 1.71 | 1.48 | 0.059 | 0.036 |

Full grid (99 / 99.5 / 99.9, all splits): `results/summary.csv`, limits and distributions: `results/thresholds.json`.

## Observations

1. Element-wise calibration compounds over 24 terms: 5–6% of train GT is infeasible at q99.5. Trajectory-level calibration keeps train GT infeasibility below 1%.
2. eth test is kinematically shifted from every train split: median speed 1.11 m/s vs 0.57 m/s (eth train), and 56% of eth test GT violates the traj-q99.5 limits. The train maxima of the other four subsets (3.90 m/s, 6.89 m/s²) are identical because they all contain the eth sequence.
3. More than half of all acceleration terms are below 0.01 m/s² in every train split, while the 99th percentile is about 1.2 m/s². The acceleration signal is dominated by a small number of kinks.
4. hotel test contains many stationary agents: 35% of speed terms are exactly zero.
