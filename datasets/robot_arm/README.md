# Exp-04 robot-arm dataset

`default/` contains the six fixed SafeFlowMPC example tasks used by the robot-arm
Y-Flow / POV study. Each `traj_example_<id>.npz` stores the start, previous state,
reference trajectory, and goal information consumed by the pretrained robot-arm
Flow Matching model.

The implementation exposes the same files through
`experiments/exp_04_robot_arm/data` for compatibility with the vendored upstream
scripts. The compatibility path is a symlink; this directory is the canonical
dataset location in Y-Flow.
