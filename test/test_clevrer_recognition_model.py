# -*- coding: utf-8 -*-

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torchvision.models import resnet34

from model import build_model
from model.clevrer_recognition import CLEVRERResNetRecognizer


class TestCLEVRERResNetRecognizer(unittest.TestCase):
    def _make_model(self, *, backbone_mode: str = "frozen") -> CLEVRERResNetRecognizer:
        return CLEVRERResNetRecognizer(
            n_slots=6,
            n_frames=4,
            dim=64,
            slot_layers=1,
            temporal_layers=1,
            n_heads=8,
            ff_dim=128,
            dropout=0.0,
            pretrained=False,
            backbone_mode=backbone_mode,
        )

    def test_forward_shapes_and_backward(self) -> None:
        model = self._make_model()
        video = torch.randn(2, 3, 4, 64, 96)
        output = model(video)
        expected = {
            "objectness_logits": (2, 6),
            "color_logits": (2, 6, 8),
            "material_logits": (2, 6, 2),
            "shape_logits": (2, 6, 3),
            "visibility_logits": (2, 6, 4),
            "positions": (2, 6, 4, 3),
            "velocities": (2, 6, 4, 3),
            "collision_logits": (2, 4, 15),
            "slot_features": (2, 6, 4, 64),
        }
        self.assertEqual({key: tuple(value.shape) for key, value in output.items()}, expected)
        self.assertTrue(all(torch.isfinite(value).all() for value in output.values()))
        loss = sum(value.square().mean() for value in output.values())
        loss.backward()
        self.assertIsNotNone(model.fuse[0].weight.grad)
        self.assertIsNone(model.layer2[0].conv1.weight.grad)

    def test_layer3_finetune_mode_only_unfreezes_layer3(self) -> None:
        model = self._make_model(backbone_mode="layer3")
        self.assertTrue(all(p.requires_grad for p in model.layer3.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.layer2.parameters()))
        model.train()
        self.assertFalse(model.layer2.training)
        self.assertTrue(model.layer3.training)

    def test_loads_local_official_layout_state_dict(self) -> None:
        state = resnet34(weights=None).state_dict()
        with tempfile.TemporaryDirectory() as tmp:
            weights_path = Path(tmp) / "resnet34.pth"
            torch.save(state, weights_path)
            model = CLEVRERResNetRecognizer(
                n_frames=2,
                dim=64,
                slot_layers=1,
                temporal_layers=1,
                n_heads=8,
                ff_dim=128,
                pretrained=True,
                weights_path=weights_path,
            )
        self.assertTrue(torch.equal(model.layer3[0].conv1.weight, state["layer3.0.conv1.weight"]))

    def test_build_model_factory(self) -> None:
        cfg = OmegaConf.create({
            "data": {"n_slots": 6, "n_frames": 3},
            "model": {
                "name": "clevrer_resnet34",
                "pretrained": False,
                "dim": 64,
                "slot_layers": 1,
                "temporal_layers": 1,
                "n_heads": 8,
                "ff_dim": 128,
            },
        })
        self.assertIsInstance(build_model(cfg), CLEVRERResNetRecognizer)


if __name__ == "__main__":
    unittest.main()
