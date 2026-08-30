# -*- coding: utf-8 -*-
# test/test_model.py

from __future__ import annotations

import unittest

import torch
from omegaconf import OmegaConf

from model import build_model
from model.mlp import VelocityMLP
from utils.paths import ROOT


class TestVelocityMLP(unittest.TestCase):
    def test_forward_shape(self) -> None:
        net = VelocityMLP(dim=2, hidden=(16, 16), time_embed_dim=8)
        x = torch.randn(7, 2)
        t = torch.rand(7)
        v = net(x, t)
        self.assertEqual(tuple(v.shape), (7, 2))

    def test_build_model_from_cfg(self) -> None:
        cfg = OmegaConf.load(ROOT / "configs" / "exp_01_swiss_roll.yaml")
        cfg.model.hidden = [8, 8]
        cfg.model.time_embed_dim = 8
        net = build_model(cfg)
        v = net(torch.zeros(3, 2), torch.zeros(3))
        self.assertEqual(tuple(v.shape), (3, 2))


if __name__ == "__main__":
    unittest.main()
