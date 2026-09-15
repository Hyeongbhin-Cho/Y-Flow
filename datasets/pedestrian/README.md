# Exp-03 pedestrian dataset

`default/` contains the ETH/UCY leave-one-out splits in the SocialGAN / EigenTrajectory
format distributed with MoFlow (`data/eth_ucy/original` at commit
`f1b89b0e80ce95646214923e271e3c280d305cfa`). Each `<subset>/<subset>_<split>.pkl`
stores a dictionary with:

- `traj`: `[N, 20, 2]` world coordinates in meters (8 past + 12 future frames, 0.4 s)
- `seq_start_end`: `[S, 2]` agent index ranges of each 20-frame window
- `num_peds_in_seq`: `[S]` agent counts per window
- `frame_list`: `[S]` frame ids per window

The implementation exposes the same files through
`experiments/exp_03_pedestrian/data/eth_ucy/original` for compatibility with the vendored
upstream scripts. The compatibility path is a symlink; this directory is the canonical
dataset location in Y-Flow.
