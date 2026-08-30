# -*- coding: utf-8 -*-
# test/test_flow_match.py

from __future__ import annotations

import unittest

import torch
from omegaconf import OmegaConf

from model.mlp import VelocityMLP
from sample.euler import EulerSampler
from train.flow_match import ConditionalFlowMatching
from utils.paths import ROOT


class TestFlowMatch(unittest.TestCase):
    def test_loss_finite(self) -> None:
        cfg = OmegaConf.load(ROOT / "configs" / "exp_01_swiss_roll.yaml")
        method = ConditionalFlowMatching(cfg)
        model = VelocityMLP(dim=2, hidden=(8, 8), time_embed_dim=8)
        loss = method.training_losses(model, torch.randn(16, 2))
        self.assertTrue(torch.isfinite(loss))

    def test_euler_shape(self) -> None:
        cfg = OmegaConf.load(ROOT / "configs" / "exp_01_swiss_roll.yaml")
        method = ConditionalFlowMatching(cfg)
        model = VelocityMLP(dim=2, hidden=(8, 8), time_embed_dim=8)
        sampler = EulerSampler(n_steps=5)
        x0 = torch.randn(4, 2)
        x1 = sampler.sample(model, method, x0)
        traj = sampler.trajectory(model, method, x0)
        self.assertEqual(tuple(x1.shape), (4, 2))
        self.assertEqual(tuple(traj.shape), (4, 6, 2))


if __name__ == "__main__":
    unittest.main()
