# -*- coding: utf-8 -*-
# test/test_constraints.py

from __future__ import annotations

import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf

from constraints.swiss_roll import SwissRollConstraint
from data.swiss_roll import build_swiss_roll
from eval.metrics import evaluate_points
from utils.paths import ROOT


class TestConstraints(unittest.TestCase):
    def test_oracle_train_points_are_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = OmegaConf.load(ROOT / "configs" / "exp_01_swiss_roll.yaml")
            cfg.data.cache_dir = tmp
            cfg.data.n_train = 200
            cfg.data.n_eval = 80
            cfg.data.regenerate = True
            bundle = build_swiss_roll(cfg)
            cons = SwissRollConstraint(bundle["meta"])
            metrics = evaluate_points(bundle["train_raw"], bundle["eval_raw"], cons)
            self.assertGreaterEqual(metrics["safe_ratio"], 0.95)
            origin = evaluate_points(
                np.zeros((8, 2), dtype=np.float64),
                bundle["eval_raw"],
                cons,
            )
            self.assertGreater(origin["core_viol_rate"], 0.9)


if __name__ == "__main__":
    unittest.main()
