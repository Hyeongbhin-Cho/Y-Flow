# -*- coding: utf-8 -*-
# test/test_wan.py
"""Tests for Wan2.1 model wrapper, configuration routing, and download logic."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from omegaconf import OmegaConf

from model import build_model
from model.download import SUBCOMPONENT_PATTERNS, is_already_downloaded
from model.wan import WanVelocityNet


class TestWanModel(unittest.TestCase):
    def test_subcomponent_patterns(self) -> None:
        self.assertIn("core", SUBCOMPONENT_PATTERNS)
        self.assertIn("transformer", SUBCOMPONENT_PATTERNS)
        self.assertIn("vae", SUBCOMPONENT_PATTERNS)
        self.assertIn("all", SUBCOMPONENT_PATTERNS)

    def test_is_already_downloaded_empty_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertFalse(is_already_downloaded(tmpdir, "core"))

    def test_is_already_downloaded_mock_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            (p / "transformer").mkdir(parents=True)
            (p / "transformer" / "config.json").write_text("{}")
            (p / "transformer" / "model.safetensors").write_text("")

            (p / "vae").mkdir(parents=True)
            (p / "vae" / "config.json").write_text("{}")
            (p / "vae" / "model.safetensors").write_text("")

            self.assertTrue(is_already_downloaded(p, "transformer"))
            self.assertTrue(is_already_downloaded(p, "vae"))
            self.assertTrue(is_already_downloaded(p, "core"))
            # text_encoder is not present yet
            self.assertFalse(is_already_downloaded(p, "text_encoder"))

    def test_wan_velocity_net_forward(self) -> None:
        mock_transformer = MagicMock()
        # Mock output of transformer: object with sample attribute
        mock_output = MagicMock()
        mock_output.sample = torch.randn(2, 16, 4, 16, 16)
        mock_transformer.return_value = mock_output

        net = WanVelocityNet(transformer=mock_transformer, text_dim=4096)
        x = torch.randn(2, 16, 4, 16, 16)
        t = torch.tensor([0.5, 0.8])

        v = net(x, t)
        self.assertEqual(tuple(v.shape), (2, 16, 4, 16, 16))
        mock_transformer.assert_called_once()
        _, kwargs = mock_transformer.call_args
        self.assertIn("hidden_states", kwargs)
        self.assertIn("timestep", kwargs)
        self.assertIn("encoder_hidden_states", kwargs)
        # Check timestep scaling: 0.5 * 1000 = 500, 0.8 * 1000 = 800
        torch.testing.assert_close(kwargs["timestep"], torch.tensor([500.0, 800.0]))

    def test_build_model_routing_wan(self) -> None:
        cfg = OmegaConf.create({
            "model": {
                "name": "wan2.1",
                "local_dir": "non_existent_path",
                "auto_download": False,
            }
        })
        # Should route to build_wan_model and raise FileNotFoundError because auto_download=False
        with self.assertRaises(FileNotFoundError):
            build_model(cfg)


if __name__ == "__main__":
    unittest.main()
