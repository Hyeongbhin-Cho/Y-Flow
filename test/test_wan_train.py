# -*- coding: utf-8 -*-
# test/test_wan_train.py
"""Tests for foundation model train mode, backbone checkpoint creation, and reuse."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from omegaconf import OmegaConf

from eval._backbone import load_frozen_velocity
from train.hard_flow import ensure_flowmatch_ckpt
from train.trainer import run_train, setup_foundation_model


class TestWanTrainPipeline(unittest.TestCase):
    def test_setup_foundation_model_creates_checkpoint_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "runs" / "test_exp" / "flowmatch"
            out_dir.mkdir(parents=True)

            cfg = OmegaConf.create({
                "model": {
                    "name": "wan2.1",
                    "variant": "1.3B",
                    "pretrained_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                    "local_dir": "checkpoints/Wan2.1-T2V-1.3B",
                }
            })

            with patch("train.trainer.build_model") as mock_build:
                mock_net = MagicMock()
                mock_build.return_value = mock_net

                device = torch.device("cpu")
                ckpt_path = setup_foundation_model(cfg, out_dir, device)

                self.assertTrue(ckpt_path.is_file())
                payload = torch.load(ckpt_path, weights_only=False)
                self.assertEqual(payload["model_name"], "wan2.1")
                self.assertTrue(payload["extra"]["is_foundation_model"])

    def test_ensure_flowmatch_ckpt_reuses_foundation_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_name = "test_wan_reuse"
            ckpt_dir = Path(tmpdir) / "runs" / run_name / "flowmatch"
            ckpt_dir.mkdir(parents=True)
            ckpt_path = ckpt_dir / "last.pt"

            # Pre-create a foundation model checkpoint
            payload = {
                "model_name": "wan2.1",
                "extra": {"is_foundation_model": True},
            }
            torch.save(payload, ckpt_path)

            cfg = OmegaConf.create({
                "run_name": run_name,
                "model": {"name": "wan2.1"},
            })

            with patch("utils.paths.ROOT", Path(tmpdir)):
                reused_path = ensure_flowmatch_ckpt(cfg)
                self.assertEqual(reused_path, ckpt_path)

    def test_load_frozen_velocity_with_foundation_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_name = "test_wan_eval"
            ckpt_dir = Path(tmpdir) / "runs" / run_name / "flowmatch"
            ckpt_dir.mkdir(parents=True)
            ckpt_path = ckpt_dir / "last.pt"

            payload = {
                "model_name": "wan2.1",
                "extra": {"is_foundation_model": True},
            }
            torch.save(payload, ckpt_path)

            cfg = OmegaConf.create({
                "run_name": run_name,
                "model": {"name": "wan2.1"},
                "method": {"sigma_min": 0.0},
            })

            with patch("utils.paths.ROOT", Path(tmpdir)):
                with patch("eval._backbone.build_model") as mock_build:
                    mock_model = MagicMock()
                    mock_model.to.return_value = mock_model
                    mock_build.return_value = mock_model

                    device = torch.device("cpu")
                    model, method = load_frozen_velocity(cfg, device)
                    self.assertIs(model, mock_model)
                    mock_model.eval.assert_called_once()


if __name__ == "__main__":
    unittest.main()
