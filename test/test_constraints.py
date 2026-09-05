# -*- coding: utf-8 -*-
# test/test_constraints.py

from __future__ import annotations

import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf

from data.swiss_roll import SwissRollConstraint, build_swiss_roll
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

    def test_autograd_differentiability(self) -> None:
        import torch
        from data.swiss_roll import SwissRollMeta

        meta = SwissRollMeta(
            a=1.0,
            u_min=4.71,
            u_max=14.14,
            n_turns=1.5,
            sigma_obs=0.05,
            n_train=10,
            n_eval=10,
            margin=0.2,
            tau=0.15,
            rho_min=4.56,
            R=14.43,
            seed=0,
            mean=(0.0, 0.0),
            std=(1.0, 1.0),
            arc_length=100.0,
        )
        cons = SwissRollConstraint(meta)
        p = torch.randn(10, 2, requires_grad=True)

        # 1. Cost differentiability
        cost = cons.cost(p)
        self.assertTrue(cost.requires_grad)
        grad_cost = torch.autograd.grad(cost.sum(), p, retain_graph=True)[0]
        self.assertEqual(grad_cost.shape, (10, 2))
        self.assertTrue(torch.isfinite(grad_cost).all())

        # 2. Hard constraints (h) differentiability
        h_dict = cons.h(p)
        for key in ["tube", "core", "box"]:
            self.assertTrue(h_dict[key].requires_grad, f"h[{key}] should require grad")
            grad_h = torch.autograd.grad(h_dict[key].sum(), p, retain_graph=True)[0]
            self.assertEqual(grad_h.shape, (10, 2))
            self.assertTrue(torch.isfinite(grad_h).all())


if __name__ == "__main__":
    unittest.main()
