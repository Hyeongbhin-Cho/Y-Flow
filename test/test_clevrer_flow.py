# -*- coding: utf-8 -*-
# test/test_clevrer_flow.py

from __future__ import annotations

import unittest

import torch
from omegaconf import OmegaConf

from data.clevrer_state import CLEVRERStateLayout
from model import build_model
from model.clevrer_flow import CLEVRERVelocityNet
from sample.euler import EulerSampler
from train.flow_match import ConditionalFlowMatching


class TestCLEVRERFlowModel(unittest.TestCase):
    def test_forward_and_conditional_loss(self) -> None:
        layout = CLEVRERStateLayout(n_slots=6, n_frames=4)
        net = CLEVRERVelocityNet(dim=layout.dim, hidden=(32, 32), time_embed_dim=8, cond_dim=16)
        x = torch.randn(3, layout.dim)
        t = torch.rand(3)
        video = torch.randn(3, 3, 4, 16, 24)
        v = net(x, t, video=video)
        self.assertEqual(tuple(v.shape), (3, layout.dim))
        cond = net.encode(video)
        self.assertEqual(tuple(cond.shape), (3, 16))
        method = ConditionalFlowMatching(sigma_min=0.0)
        loss = method.training_losses(net, x, cond=cond)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_euler_with_cond(self) -> None:
        layout = CLEVRERStateLayout(n_slots=6, n_frames=4)
        net = CLEVRERVelocityNet(dim=layout.dim, hidden=(16,), time_embed_dim=8, cond_dim=8)
        method = ConditionalFlowMatching(sigma_min=0.0)
        sampler = EulerSampler(n_steps=3)
        x0 = torch.randn(2, layout.dim)
        cond = torch.randn(2, 8)
        x1 = sampler.sample(net, method, x0, cond=cond)
        self.assertEqual(tuple(x1.shape), (2, layout.dim))

    def test_build_model_from_cfg(self) -> None:
        cfg = OmegaConf.create({
            "data": {"n_slots": 6, "n_frames": 4},
            "model": {"name": "clevrer_flow", "hidden": [16, 16], "time_embed_dim": 8, "cond_dim": 8},
        })
        net = build_model(cfg)
        self.assertIsInstance(net, CLEVRERVelocityNet)
        self.assertEqual(net.dim, CLEVRERStateLayout(n_slots=6, n_frames=4).dim)


if __name__ == "__main__":
    unittest.main()
