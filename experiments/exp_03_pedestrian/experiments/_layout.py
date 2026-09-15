# -*- coding: utf-8 -*-
# experiments/exp_03_pedestrian/experiments/_layout.py
"""Canonical Exp-03 paths inside the Y-Flow repository."""

from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TASK_ROOT.parents[1]
DATASET_ROOT = REPOSITORY_ROOT / "datasets" / "pedestrian" / "default"
CHECKPOINT_ROOT = TASK_ROOT / "checkpoints" / "eth_ucy" / "moflow"
CFM_ROOT = TASK_ROOT / "checkpoints" / "cfm"
RUN_ROOT = REPOSITORY_ROOT / "runs" / "exp_03_pedestrian"
SUBSETS = ("eth", "hotel", "univ", "zara1", "zara2")
