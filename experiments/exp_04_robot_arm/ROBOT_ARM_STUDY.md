# Robot-arm Y-Flow / POV study

This Exp-04 directory is intentionally isolated from the existing Y-Flow implementation.
It vendors the unmodified SafeFlowMPC source tree at commit
`3efe4d9522f4112b291a868866e7e4934697a261` and adds the robot-arm Y-Flow / POV
experiments under `experiments/`.

## Scope

- The pretrained unsafe Flow Matching checkpoint and original robot model are reused.
- Existing Y-Flow method code is not used or modified by the robot-arm runtime.
- Projection uses the candidate-dependent Acados local weighted projection surrogate;
  it is not claimed to be an exact global Euclidean projection.
- `L05_NO_REPLACE` uses the optimizer result as a damped correction direction and
  does not replace the terminal trajectory with `z_star`.

## Main artifacts

- `../../runs/exp_04_robot_arm/pov_projection/phase1/REPORT.md`
- `../../runs/exp_04_robot_arm/pov_projection_phase2/REPORT_PHASE2.md`
- `../../runs/exp_04_robot_arm/pov_projection_phase3/REPORT_PHASE3.md`
- `../../runs/exp_04_robot_arm/pov_projection_phase15_yflow_main/REPORT_PHASE15.md`
- `../../runs/exp_04_robot_arm/pov_yflow_hybrid/REPORT_HYBRID.md`
- `../../runs/exp_04_robot_arm/pov_factorial_ablation/REPORT_FACTORIAL.md`
- `../../runs/exp_04_robot_arm/ood_no_replace/REPORT_OOD.md`

The final OOD benchmark contains 100 deterministic obstacle scenarios, eight
matched noise seeds, and four fixed methods (3,200 trajectory evaluations). Its
final verdict is `LIMITED_SUPPORT`: no-replacement improves overall collision
relative to Plain FM and terminal replacement, but the ID collision rate does not
generalize absolutely and enlarged-obstacle scenes remain difficult.

## Running

Run commands through `../../run_exp_04_robot_arm.sh`, or from this directory, so imports such as
`safe_flow_mpc` and `experiments` resolve against this isolated tree. The reports
record the exact experiment settings and reproduction commands. Acados v0.5.1 and
the dependency versions described by the experiment setup documents are required.
