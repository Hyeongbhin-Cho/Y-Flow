# -*- coding: utf-8 -*-
# test/test_exp_03_pedestrian.py
"""Exp-03 layout isolation and ETH/UCY dataset integrity."""

import pickle
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "experiments" / "exp_03_pedestrian"
DATASET_ROOT = ROOT / "datasets" / "pedestrian" / "default"
SUBSETS = ("eth", "hotel", "univ", "zara1", "zara2")
SPLITS = ("train", "val", "test")


class TestExp03Pedestrian(unittest.TestCase):
    def test_numbered_layout(self):
        self.assertTrue((ROOT / "configs" / "exp_03_pedestrian.yaml").is_file())
        self.assertTrue((ROOT / "run_exp_03_pedestrian.sh").is_file())
        self.assertTrue((ROOT / "docs" / "exp" / "exp_03_pedestrian.md").is_file())
        self.assertTrue((ROOT / "runs" / "exp_03_pedestrian").is_dir())
        self.assertTrue(DATASET_ROOT.is_dir())
        self.assertTrue((TASK_ROOT / "models" / "flow_matching.py").is_file())
        self.assertTrue((TASK_ROOT / "eval_eth.py").is_file())
        self.assertTrue((TASK_ROOT / "data" / "eth_ucy" / "original").resolve().samefile(
            DATASET_ROOT
        ))

    def test_dataset_integrity(self):
        for subset in SUBSETS:
            for split in SPLITS:
                with self.subTest(subset=subset, split=split):
                    path = DATASET_ROOT / subset / f"{subset}_{split}.pkl"
                    with path.open("rb") as f:
                        payload = pickle.load(f)
                    traj = np.asarray(payload["traj"])
                    seq = np.asarray(payload["seq_start_end"])
                    num = np.asarray(payload["num_peds_in_seq"])
                    self.assertEqual(traj.ndim, 3)
                    self.assertEqual(traj.shape[1:], (20, 2))
                    self.assertTrue(np.isfinite(traj).all())
                    self.assertEqual(int(seq[0, 0]), 0)
                    self.assertEqual(int(seq[-1, 1]), traj.shape[0])
                    self.assertTrue(np.array_equal(seq[1:, 0], seq[:-1, 1]))
                    self.assertTrue(np.array_equal(seq[:, 1] - seq[:, 0], num))

    def test_audit_kinematics(self):
        import sys
        sys.path.insert(0, str(TASK_ROOT))
        from experiments.phase0_gt_audit.run_audit import kinematics
        steps = np.arange(20, dtype=np.float64)[None, :, None] * np.array([0.4, 0.0])
        speed, acc = kinematics(steps)
        self.assertEqual(speed.shape, (1, 12))
        self.assertEqual(acc.shape, (1, 12))
        self.assertTrue(np.allclose(speed, 1.0))
        self.assertTrue(np.allclose(acc, 0.0))

    def test_cpu_integrity_suite(self):
        subprocess.run(
            [sys.executable, "-m", "experiments.phase1_yflow.test_phase1"],
            cwd=TASK_ROOT, check=True, capture_output=True, text=True,
        )


if __name__ == "__main__":
    unittest.main()
