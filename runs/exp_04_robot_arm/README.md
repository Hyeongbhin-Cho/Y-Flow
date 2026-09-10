# Exp-04 robot-arm artifacts

This directory contains the immutable reports, raw tables, logs, plots, and
representative trajectories produced by the robot-arm Y-Flow / POV study.

The study directories are ordered by purpose rather than by chronology:

- `pov_projection/`: initial endpoint-projection proof of concept
- `pov_projection_phase2/`: corrected POV dynamics and goal-aware projection
- `pov_projection_phase3/`: adaptive/late projection evaluation
- `pov_projection_phase15_yflow_main/`: remote-main Y-Flow semantics transfer
- `pov_yflow_hybrid/`: hybrid trigger study
- `pov_factorial_ablation/`: lambda and terminal-replacement factorial analysis
- `ood_no_replace/`: fixed 100-scenario OOD benchmark and final report

The primary conclusion is in `ood_no_replace/REPORT_OOD.md`.
