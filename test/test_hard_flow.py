# -*- coding: utf-8 -*-
# test/test_hard_flow.py

from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch
from omegaconf import OmegaConf

from eval.evaluate import make_eval_x0, run_eval
from train.hard_flow import ensure_flowmatch_ckpt
from utils.device import get_device
from utils.paths import ROOT, flowmatch_ckpt


class TestHardFlow(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.run_name = "unittest_hardflow"
        run_root = ROOT / "runs" / self.run_name
        if run_root.exists():
            shutil.rmtree(run_root)

    def tearDown(self) -> None:
        run_root = ROOT / "runs" / self.run_name
        if run_root.exists():
            shutil.rmtree(run_root)
        self.tmp.cleanup()

    def _cfg(self):
        cfg = OmegaConf.load(ROOT / "configs" / "exp_01_swiss_roll.yaml")
        cfg.data.cache_dir = self.tmp.name
        cfg.data.n_train = 64
        cfg.data.n_eval = 32
        cfg.data.regenerate = True
        cfg.train.steps = 4
        cfg.train.batch_size = 16
        cfg.sample.n_samples = 16
        cfg.sample.n_steps = 6
        cfg.log.plot_every = 4
        cfg.device = "cpu"
        cfg.run_name = self.run_name
        cfg.hardflow.t_on = 0.5
        cfg.hardflow.max_iter = 8
        return cfg

    def test_skip_when_ckpt_exists(self) -> None:
        cfg = self._cfg()
        path = flowmatch_ckpt(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"dummy")
        buf = io.StringIO()
        with redirect_stdout(buf):
            got = ensure_flowmatch_ckpt(cfg)
        self.assertEqual(got, path)
        self.assertIn("already exists", buf.getvalue())
        self.assertEqual(path.read_bytes(), b"dummy")

    def test_trains_when_missing_then_eval(self) -> None:
        cfg = self._cfg()
        buf = io.StringIO()
        with redirect_stdout(buf):
            ckpt = ensure_flowmatch_ckpt(cfg)
        self.assertTrue(Path(ckpt).is_file())
        self.assertIn("training flowmatch", buf.getvalue())
        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            again = ensure_flowmatch_ckpt(cfg)
        self.assertIn("already exists", buf2.getvalue())
        self.assertEqual(again, ckpt)

        metrics = run_eval(cfg, "hardflow")
        self.assertEqual(metrics["method"], "hardflow")
        self.assertIn("safe_ratio", metrics)
        self.assertTrue((ROOT / "runs" / self.run_name / "hardflow" / "metrics.json").is_file())

    def test_eval_x0_is_deterministic(self) -> None:
        cfg = self._cfg()
        device = get_device(cfg)
        a = make_eval_x0(cfg, device)
        b = make_eval_x0(cfg, device)
        self.assertTrue(torch.equal(a.cpu(), b.cpu()))
        cfg.seed = int(cfg.seed) + 1
        c = make_eval_x0(cfg, device)
        self.assertFalse(torch.equal(a.cpu(), c.cpu()))


if __name__ == "__main__":
    unittest.main()

