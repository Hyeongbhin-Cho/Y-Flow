from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from eval.moflow import trajectory_metrics
from eval.autonomous_methods import _SAMPLERS, _cyclic_halfspace_correction
from data.autonomous_driving import AutonomousDrivingConstraint
from model.moflow import build_moflow_model
from train.moflow import moflow_loss


def _cfg():
    return OmegaConf.create(
        {
            "model": {"dim": 120, "time_embed_dim": 16},
            "data": {
                "context": {
                    "radius_m": 50.0,
                    "actor_types": ["vehicle", "bus", "cyclist", "pedestrian", "motorcyclist"],
                }
            },
            "moflow": {
                "n_modes": 3,
                "hidden_dim": 32,
                "type_embed_dim": 8,
                "classification_weight": 0.1,
            },
            "sample": {"n_steps": 2},
            "hardflow": {"t_on": 0.5, "lambda_oc": 1.0, "max_iter": 1, "safety_buffer": 1e-4},
            "safeflow": {"t_on": 0.5, "slack_weight": 1.0, "av2_gain": 1.0, "terminal_filter": {"enabled": True}},
            "uniconflow": {"free_until": 0.0, "ptzf_rate": 1.0, "gamma": 1.0, "slack_weight": 10.0, "max_guidance_norm": 5.0, "terminal_refinement": True, "safety_buffer": 1e-4},
            "guideflow": {"tau_star": 0.5, "eta_max": 0.1, "av2_max_guidance_norm": 1.0},
            "yflow": {"t_on": 0.5, "lambda_oc": 1.0, "mu": 1.0, "max_iter": 1, "safety_buffer": 1e-4},
        }
    )


def _context(batch: int):
    return {
        "focal_history": torch.randn(batch, 50, 2),
        "neighbor_history": torch.randn(batch, 4, 50, 2),
        "neighbor_mask": torch.tensor([[True, True, False, False]] * batch),
        "neighbor_types": torch.tensor([[1, 2, 0, 0]] * batch),
    }


class TestMoFlow(unittest.TestCase):
    def test_k_shot_forward_and_loss(self):
        model = build_moflow_model(_cfg())
        target = torch.randn(2, 120)
        context = _context(2)
        velocity, logits = model(torch.randn(2, 3, 120), torch.rand(2), context)
        self.assertEqual(velocity.shape, (2, 3, 120))
        self.assertEqual(logits.shape, (2, 3))
        loss, parts = moflow_loss(model, target, context, 3, 0.1)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("regression", parts)
        loss.backward()

    def test_min_metrics_select_best_mode(self):
        truth = np.zeros((2, 120), dtype=np.float32)
        predictions = np.ones((2, 3, 120), dtype=np.float32)
        predictions[:, 1] = 0.0
        metrics = trajectory_metrics(predictions, truth)
        self.assertEqual(metrics["minADE"], 0.0)
        self.assertEqual(metrics["minFDE"], 0.0)

    def test_all_constraint_adapters_preserve_shape(self):
        cfg = _cfg()
        model = build_moflow_model(cfg)
        context = _context(2)
        feature = model.encode_context(context)
        x0 = torch.randn(2, 3, 120)
        mean = torch.zeros(120)
        std = torch.ones(120)
        meta = SimpleNamespace(
            difference_window=5, future_steps=60, sample_hz=10.0,
            v_max=25.0, a_max=15.0,
        )
        constraint = AutonomousDrivingConstraint(meta)
        for name, sampler in _SAMPLERS.items():
            result, diagnostics = sampler(
                cfg, model, context, feature, x0.clone(), mean, std, constraint
            )
            self.assertEqual(result.shape, x0.shape, name)
            self.assertTrue(torch.isfinite(result).all(), name)
            self.assertFalse(result.requires_grad, name)
            self.assertIsInstance(diagnostics, dict)
            if name == "guideflow":
                self.assertGreater(diagnostics["energy_correction_steps"], 0)

    def test_safe_flow_qp_fallback_satisfies_simple_halfspaces(self):
        a = torch.tensor([[-2.0, -3.0]])
        b = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        correction = _cyclic_halfspace_correction(a, b)
        residual = a + (b * correction.unsqueeze(-2)).sum(dim=-1)
        self.assertTrue((residual >= -1e-6).all())


if __name__ == "__main__":
    unittest.main()
