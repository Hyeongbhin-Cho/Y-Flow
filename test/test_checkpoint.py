# -*- coding: utf-8 -*-
# test/test_checkpoint.py

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from eval._backbone import load_frozen_velocity
from model import build_model
from train.checkpoint import maybe_publish_checkpoint, publish_checkpoint, save_checkpoint
from train.ema import EMA
from train.hard_flow import ensure_flowmatch_ckpt
from utils.paths import (
    flowmatch_ckpt,
    missing_ckpt_message,
    published_ckpt,
    published_dir,
    resolve_flowmatch_ckpt,
)


def _tiny_cfg(run_name: str, local_dir: str) -> OmegaConf:
    return OmegaConf.create(
        {
            "run_name": run_name,
            "model": {
                "name": "mlp",
                "dim": 2,
                "hidden": [8],
                "time_embed_dim": 8,
                "local_dir": local_dir,
            },
            "train": {"ema_decay": 0.9},
            "method": {"sigma_min": 0.0},
        }
    )


def _write_ckpt(path: Path, cfg) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg)
    ema = EMA(model, decay=0.9)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    save_checkpoint(path, model, ema, opt, 7, cfg, extra={"meta": {"dim": 2}})
    (path.parent / "config.yaml").write_text("run_name: src\n")


class TestPublishedCheckpoint(unittest.TestCase):
    def test_published_dir_none_without_local_dir(self) -> None:
        cfg = OmegaConf.create({"model": {"name": "mlp"}})
        self.assertIsNone(published_dir(cfg))
        self.assertIsNone(published_ckpt(cfg))

    def test_publish_writes_last_pt_ready_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "runs" / "exp" / "flowmatch" / "last.pt"
            dst = root / "checkpoints" / "clevrer_flow" / "last.pt"
            cfg = _tiny_cfg("exp", "checkpoints/clevrer_flow")
            _write_ckpt(src, cfg)
            published = publish_checkpoint(
                src, dst, extra={"step": 7, "model": "mlp", "run_name": "exp"}
            )
            self.assertEqual(published, dst)
            self.assertTrue(dst.is_file())
            self.assertTrue((dst.parent / "config.yaml").is_file())
            ready = json.loads((dst.parent / "READY.json").read_text())
            self.assertEqual(ready["checkpoint"], "last.pt")
            self.assertEqual(ready["step"], 7)
            self.assertEqual(ready["model"], "mlp")
            self.assertEqual(ready["source"], str(src))

    def test_resolve_prefers_run_then_published(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = _tiny_cfg("exp_a", "checkpoints/clevrer_flow")
            run_path = root / "runs" / "exp_a" / "flowmatch" / "last.pt"
            pub_path = root / "checkpoints" / "clevrer_flow" / "last.pt"
            with patch("utils.paths.ROOT", root):
                self.assertEqual(resolve_flowmatch_ckpt(cfg), flowmatch_ckpt(cfg))
                pub_path.parent.mkdir(parents=True)
                pub_path.write_bytes(b"published")
                self.assertEqual(resolve_flowmatch_ckpt(cfg), pub_path)
                run_path.parent.mkdir(parents=True)
                run_path.write_bytes(b"run")
                self.assertEqual(resolve_flowmatch_ckpt(cfg), run_path)

    def test_load_frozen_velocity_falls_back_to_published(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = _tiny_cfg("other_exp", "checkpoints/clevrer_flow")
            pub = root / "checkpoints" / "clevrer_flow" / "last.pt"
            _write_ckpt(pub, cfg)
            with patch("utils.paths.ROOT", root):
                model, method = load_frozen_velocity(cfg, torch.device("cpu"))
            self.assertEqual(tuple(model(torch.zeros(2, 2), torch.zeros(2)).shape), (2, 2))
            self.assertTrue(hasattr(method, "training_losses"))

    def test_ensure_reuses_published_when_run_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pub = root / "checkpoints" / "clevrer_flow" / "last.pt"
            pub.parent.mkdir(parents=True)
            pub.write_bytes(b"dummy")
            cfg = OmegaConf.create(
                {
                    "run_name": "new_exp",
                    "model": {"local_dir": "checkpoints/clevrer_flow"},
                }
            )
            with patch("utils.paths.ROOT", root):
                got = ensure_flowmatch_ckpt(cfg)
            self.assertEqual(got, pub)

    def test_maybe_publish_skips_wan_weight_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wan = root / "checkpoints" / "Wan2.1-T2V-1.3B"
            (wan / "transformer").mkdir(parents=True)
            src = root / "runs" / "x" / "flowmatch" / "last.pt"
            src.parent.mkdir(parents=True)
            src.write_bytes(b"ckpt")
            cfg = OmegaConf.create({"model": {"local_dir": "checkpoints/Wan2.1-T2V-1.3B"}})
            with patch("utils.paths.ROOT", root):
                self.assertIsNone(maybe_publish_checkpoint(cfg, src))
            self.assertFalse((wan / "last.pt").exists())

    def test_missing_message_mentions_published(self) -> None:
        cfg = _tiny_cfg("gone", "checkpoints/clevrer_flow")
        msg = missing_ckpt_message(cfg)
        self.assertIn("gone", msg)
        self.assertIn("clevrer_flow", msg)


if __name__ == "__main__":
    unittest.main()
