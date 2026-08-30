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


if __name__ == "__main__":
    unittest.main()
