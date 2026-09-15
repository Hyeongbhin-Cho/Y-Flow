# Pedestrian Y-Flow study

This Exp-03 directory is intentionally isolated from the existing Y-Flow implementation.
It vendors the MoFlow (CVPR 2025) source tree at commit
`f1b89b0e80ce95646214923e271e3c280d305cfa` and adds the pedestrian Y-Flow
experiments under `experiments/`.

## Scope

- The pretrained MoFlow teacher checkpoints (ETH/UCY, five leave-one-out subsets) are reused.
- Existing Y-Flow method code is not used or modified by the pedestrian runtime.
- Upstream code is unmodified. The bundled NBA/SDD data, the LED-format ETH/UCY
  arrays, and the raw ETH/UCY text files were removed, and `.gitignore` gained a
  `checkpoints/` entry. `prepare_checkpoints.py` is an addition.
- `data/eth_ucy/original` is a symlink to `../../../../datasets/pedestrian/default`,
  the canonical dataset location in Y-Flow.

## Constraint

Per-agent kinematic hard constraints on the 12 predicted frames (dt = 0.4 s):
speed `||p_k - p_{k-1}||/dt <= v_max` and acceleration
`||p_k - 2p_{k-1} + p_{k-2}||/dt^2 <= a_max`, anchored on the last observed
position and velocity. Limits are the q99.5 of the per-trajectory maximum on the
subset's train split (Phase 0). The physical operator P is absent, so the Y-Flow
terminal problem is solved on its documented `mu = 0`, `C = 0` path, which is the
exact Euclidean projection onto the feasible set (ADMM, then a closed-form
forward seal at the terminal step). The observed anchor velocity is clipped to
`v_max` so the set is never empty; the clip rate is reported.

## Clearance constraint (planning setting)

`--constraint kin_col` (default) adds `||p_k - q_{j,k}|| >= r_safe` for every
co-present agent j of the same 20-frame window, where `q_{j,k}` is j's
**ground-truth** future mapped into the agent's MoFlow frame
(`experiments/phase1_yflow/neighbors.py`). Using neighbours' GT futures is an
oracle: results are a planning-with-known-moving-obstacles setting (the Exp-04
analogue), not a forecasting benchmark.

Clearance is non-convex. Each row is convexified into the half-plane through
the candidate direction (`n_{jk}` from `q_{j,k}` to the candidate `p_k`), so the
projection is exact on a candidate-dependent convex inner set, not the global
nearest collision-free trajectory. Neighbours that never enter the kinematic
reachable disc are dropped exactly. Terminal feasibility is not guaranteed when
the convexified set is empty; `proj_fail_rate` reports it.

## Pretrained checkpoint limitation

With the released teachers the clean prediction is nearly independent of the
flow state: outputs are identical across noise seeds, `terminal_raw_shift_m = 0`
after intermediate corrections, and `COMMAND=probe` shows a perturbation of one
normalized unit (about 9.5 m on zara1) changing x1_hat by only 5-9 mm at every
sampler step, including the terminal one. With these checkpoints Y-Flow
therefore collapses to the terminal projection.

The cause is not established. The checkpoints were trained with
`--drop_method emb` (y-embedding dropout growing with t), but the probe shows
the same insensitivity at small t where the training drop rate is near zero, so
the dropout alone does not explain it. Retraining without it is a test, not a
known fix:

```bash
COMMAND=train ./run_exp_03_pedestrian.sh --exp nodrop --subset zara2 --rotate --rotate_time_frame 6 \
  --tied_noise --fm_in_scaling --checkpt_freq 1 --batch_size 32 --init_lr 1e-4 --drop_method None
COMMAND=phase1 ./run_exp_03_pedestrian.sh --subset zara2 --ckpt_dir results_eth_ucy/cor_fm/<run_tag>
```

Training needs CUDA (the upstream validation loop uses `torch.cuda.Event`).
Before comparing methods, check the retrained model: accuracy near the released
teacher, outputs that differ across `--seeds`, and `terminal_raw_shift_m > 0`
for `YFLOW_NO_REPLACE`.

## Main study (Phase 2)

```bash
COMMAND=prepare ./run_exp_03_pedestrian.sh    # MoFlow teachers (once)
COMMAND=study   ./run_exp_03_pedestrian.sh    # train cfm x5, probes, all runs, report
```

Stages can be selected, e.g. `STAGES="run report" SUBSETS="zara1 zara2" COMMAND=study ...`.
Progress and failures go to `../../runs/exp_03_pedestrian/study/study.log`; the
final report is `runs/exp_03_pedestrian/study/REPORT_STUDY.md` with `gates.json`.
The gates are fixed in `docs/exp/exp_03_pedestrian.md`.

## Checkpoints

```bash
pip install -U huggingface_hub
COMMAND=prepare ./run_exp_03_pedestrian.sh          # all five subsets
```

This downloads `fyxfelixfu/moflow` and lays each subset out as
`checkpoints/eth_ucy/moflow/<subset>/{cor_fm.yml, cor_fm_updated.yml, models/checkpoint_best.pt}`
so that both `eval_eth.py --cfg auto` and `run_phase1.py` find it. The directory is git-ignored.

## Running

```bash
COMMAND=phase1 ./run_exp_03_pedestrian.sh --subset zara1 --limit_batches 1                  # smoke test
COMMAND=phase1 ./run_exp_03_pedestrian.sh --subset zara2                                     # kin + clearance
COMMAND=phase1 ./run_exp_03_pedestrian.sh --subset zara2 --constraint kin                    # kinematic only
COMMAND=eval   ./run_exp_03_pedestrian.sh --subset zara1 --ckpt_path checkpoints/eth_ucy/moflow/zara1/models/checkpoint_best.pt \
  --rotate --rotate_time_frame 6 --batch_size 1000 --sampling_steps 100 --solver lin_poly --lin_poly_p 5 --lin_poly_long_step 1000
```

`phase1` runs on CPU or GPU. The upstream `eval` command needs CUDA
(`trainer/denoising_model_trainers.py` uses `torch.cuda.Event`); use it only for
the paper-number reproduction check on a GPU machine. Its `--solver euler` path
has an upstream bug in the print schedule; use `lin_poly`.
