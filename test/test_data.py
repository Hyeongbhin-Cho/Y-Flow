# -*- coding: utf-8 -*-
# test/test_data.py

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from data.swiss_roll import build_swiss_roll, load_swiss_roll, save_swiss_roll
from utils.paths import ROOT


def _tiny_cfg(cache_dir: str) -> OmegaConf:
    cfg = OmegaConf.load(ROOT / "configs" / "exp_01_swiss_roll.yaml")
    cfg.data.cache_dir = cache_dir
    cfg.data.n_train = 64
    cfg.data.n_eval = 32
    cfg.data.regenerate = True
    cfg.seed = 0
    return cfg


class TestSwissRollCache(unittest.TestCase):
    def test_save_load_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_cfg(tmp)
            first = build_swiss_roll(cfg)
            self.assertTrue((Path(tmp) / "train.npy").is_file())
            self.assertTrue((Path(tmp) / "eval.npy").is_file())
            self.assertTrue((Path(tmp) / "meta.json").is_file())

            cfg.data.regenerate = False
            second = build_swiss_roll(cfg)
            loaded = load_swiss_roll(tmp)
            self.assertEqual(first["train_raw"].shape, (64, 2))
            self.assertTrue((first["train_raw"] == second["train_raw"]).all())
            self.assertTrue((first["eval_raw"] == loaded["eval_raw"]).all())
            self.assertEqual(first["meta"].tau, loaded["meta"].tau)
            self.assertEqual(first["meta"].mean, loaded["meta"].mean)

    def test_load_ignores_new_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_cfg(tmp)
            original = build_swiss_roll(cfg)
            cfg.data.regenerate = False
            cfg.seed = 99
            cfg.data.n_train = 8
            again = build_swiss_roll(cfg)
            self.assertTrue((original["train_raw"] == again["train_raw"]).all())
            self.assertEqual(again["meta"].n_train, 64)

    def test_save_roundtrip_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_cfg(tmp)
            bundle = build_swiss_roll(cfg)
            other = Path(tmp) / "copy"
            save_swiss_roll(other, bundle["train_raw"], bundle["eval_raw"], bundle["meta"])
            copied = load_swiss_roll(other)
            self.assertTrue((copied["train_raw"] == bundle["train_raw"]).all())
            self.assertEqual(copied["meta"].R, bundle["meta"].R)


class TestDataRouter(unittest.TestCase):
    def test_build_dataset_routing(self) -> None:
        from data.base import BaseConstraint, DataBundle, build_dataset

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_cfg(tmp)
            bundle = build_dataset(cfg)
            self.assertIsInstance(bundle, DataBundle)
            self.assertIsInstance(bundle.constraint, BaseConstraint)
            # Test attribute and dict access
            self.assertEqual(bundle["train_raw"].shape, (64, 2))
            self.assertEqual(bundle.train_raw.shape, (64, 2))
            self.assertIn("train", bundle)

            # Test string routing
            bundle_str = build_dataset("swiss_roll", cfg=cfg)
            self.assertEqual(bundle_str.train_raw.shape, (64, 2))

    def test_unknown_dataset(self) -> None:
        from data.base import build_dataset

        with self.assertRaises(KeyError):
            build_dataset("unknown_dataset", cfg=OmegaConf.create({"data": {"name": "unknown_dataset"}}))

    def test_guideflow_constraint_methods(self) -> None:
        import numpy as np
        import torch
        from data.base import build_dataset

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_cfg(tmp)
            bundle = build_dataset(cfg)
            c = bundle.constraint

            # Anchor vocabulary
            anchors = c.build_anchor_vocabulary(bundle.train_raw, n_anchors=16, seed=0)
            self.assertEqual(anchors.shape, (16, 2))

            # Progress & Command bins (NumPy and PyTorch)
            pts_np = bundle.train_raw[:10]
            prog_np = c.progress(pts_np)
            self.assertEqual(prog_np.shape, (10,))
            self.assertTrue(np.all((prog_np >= 0.0) & (prog_np <= 1.0)))

            bins_np = c.command_bins(pts_np, n_commands=5)
            self.assertEqual(bins_np.shape, (10,))
            self.assertTrue(np.all((bins_np >= 0) & (bins_np < 5)))

            pts_t = torch.from_numpy(pts_np)
            prog_t = c.progress(pts_t)
            self.assertIsInstance(prog_t, torch.Tensor)
            self.assertEqual(prog_t.shape, (10,))

            bins_t = c.command_bins(pts_t, n_commands=5)
            self.assertIsInstance(bins_t, torch.Tensor)
            self.assertEqual(bins_t.shape, (10,))

            # Energy
            e = c.energy(pts_t)
            self.assertIsInstance(e, torch.Tensor)
            self.assertEqual(e.shape, (10,))


if __name__ == "__main__":
    unittest.main()
