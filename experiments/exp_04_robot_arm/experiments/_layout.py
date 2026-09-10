"""Canonical Exp-04 paths inside the Y-Flow repository."""

from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TASK_ROOT.parents[1]
DATASET_ROOT = REPOSITORY_ROOT / "datasets" / "robot_arm" / "default"
RUN_ROOT = REPOSITORY_ROOT / "runs" / "exp_04_robot_arm"
