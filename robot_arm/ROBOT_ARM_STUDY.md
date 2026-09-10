# Robot-arm Y-Flow / POV study

This directory is intentionally isolated from the existing Y-Flow implementation.
It vendors the unmodified SafeFlowMPC source tree at commit
`3efe4d9522f4112b291a868866e7e4934697a261` and adds the robot-arm Y-Flow / POV
experiments under `experiments/`.

## Scope

- The pretrained unsafe Flow Matching checkpoint and original robot model are reused.
- Existing Y-Flow source files outside `robot_arm/` are not modified.
- Projection uses the candidate-dependent Acados local weighted projection surrogate;
  it is not claimed to be an exact global Euclidean projection.
- `L05_NO_REPLACE` uses the optimizer result as a damped correction direction and
  does not replace the terminal trajectory with `z_star`.

## Main artifacts

- `experiments/pov_projection/artifacts/phase1/REPORT.md`
- `experiments/pov_projection_phase2/REPORT_PHASE2.md`
- `experiments/pov_projection_phase3/REPORT_PHASE3.md`
- `experiments/pov_projection_phase15_yflow_main/REPORT_PHASE15.md`
- `experiments/pov_yflow_hybrid/REPORT_HYBRID.md`
- `experiments/pov_factorial_ablation/REPORT_FACTORIAL.md`
- `experiments/ood_no_replace/REPORT_OOD.md`

The final OOD benchmark contains 100 deterministic obstacle scenarios, eight
matched noise seeds, and four fixed methods (3,200 trajectory evaluations). Its
final verdict is `LIMITED_SUPPORT`: no-replacement improves overall collision
relative to Plain FM and terminal replacement, but the ID collision rate does not
generalize absolutely and enlarged-obstacle scenes remain difficult.

## Running

Run commands from this `robot_arm/` directory so imports such as
`safe_flow_mpc` and `experiments` resolve against this isolated tree. The reports
record the exact experiment settings and reproduction commands. Acados v0.5.1 and
the dependency versions described by the experiment setup documents are required.
