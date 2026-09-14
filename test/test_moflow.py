from __future__ import annotations

import unittest

import numpy as np
import torch
from omegaconf import OmegaConf

from eval.moflow import trajectory_metrics
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


if __name__ == "__main__":
    unittest.main()
