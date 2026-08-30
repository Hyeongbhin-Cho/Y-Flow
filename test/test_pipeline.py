# -*- coding: utf-8 -*-
# test/test_pipeline.py

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from eval.evaluate import run_eval
from train.trainer import run_train
from utils.paths import ROOT, flowmatch_ckpt, method_dir


class TestPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.run_name = "unittest_pipeline"
        run_root = ROOT / "runs" / self.run_name
        if run_root.exists():
            shutil.rmtree(run_root)

    def tearDown(self) -> None:
        run_root = ROOT / "runs" / self.run_name
        if run_root.exists():
            shutil.rmtree(run_root)
        self.tmp.cleanup()

    def test_short_train_then_eval_writes_json(self) -> None:
        cfg = OmegaConf.load(ROOT / "configs" / "exp_01_swiss_roll.yaml")
        cfg.data.cache_dir = self.tmp.name
        cfg.data.n_train = 64
        cfg.data.n_eval = 32
        cfg.data.regenerate = True
        cfg.train.steps = 4
        cfg.train.batch_size = 16
        cfg.train.lr = 1.0e-2
        cfg.sample.n_samples = 32
        cfg.sample.n_steps = 4
        cfg.log.plot_every = 4
        cfg.device = "cpu"
        cfg.run_name = self.run_name

        ckpt = run_train(cfg, method="flowmatch")
        self.assertTrue(Path(ckpt).is_file())
        self.assertTrue(flowmatch_ckpt(cfg).is_file())

        metrics = run_eval(cfg, "flowmatch")
        method_metrics = method_dir(cfg, "flowmatch") / "metrics.json"
        run_metrics = ROOT / "runs" / self.run_name / "metrics.json"
        self.assertTrue(method_metrics.is_file())
        self.assertTrue(run_metrics.is_file())
        payload = json.loads(run_metrics.read_text())
        self.assertEqual(payload["run_name"], self.run_name)
        self.assertIn("flowmatch", payload["methods"])
        self.assertIn("safe_ratio", metrics)
        self.assertIn("inference_time_s", metrics)


if __name__ == "__main__":
    unittest.main()
