# -*- coding: utf-8 -*-
# test/test_safe_flow.py

from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout

import torch
from omegaconf import OmegaConf

from eval.evaluate import make_eval_x0, run_eval
from eval.flow_match import sample as sample_flowmatch
from eval.safe_flow import sample as sample_safeflow
from eval.safe_flow_t_on_ablation import run_ablation
from train.safe_flow import ensure_flowmatch_ckpt
from utils.device import get_device
from utils.paths import ROOT, flowmatch_ckpt


class TestSafeFlow(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.tmp = tempfile.TemporaryDirectory()
        self.run_name = "unittest_safeflow"
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
        cfg.sample.n_samples = 12
        cfg.sample.n_steps = 8
        cfg.log.plot_every = 4
        cfg.device = "cpu"
        cfg.run_name = self.run_name
        cfg.safeflow.terminal_filter.max_iter = 100
        return cfg

    def _ensure_backbone(self, cfg) -> None:
        with redirect_stdout(io.StringIO()):
            ensure_flowmatch_ckpt(cfg)

    def test_reuses_flowmatch_checkpoint_without_own_checkpoint(self) -> None:
        cfg = self._cfg()
        self._ensure_backbone(cfg)
        self.assertTrue(flowmatch_ckpt(cfg).is_file())
        self.assertFalse((ROOT / "runs" / self.run_name / "safeflow" / "last.pt").exists())
        buf = io.StringIO()
        with redirect_stdout(buf):
            again = ensure_flowmatch_ckpt(cfg)
        self.assertEqual(again, flowmatch_ckpt(cfg))
        self.assertIn("already exists", buf.getvalue())

    def test_disabled_euler_matches_flowmatch(self) -> None:
        cfg = self._cfg()
        self._ensure_backbone(cfg)
        cfg.safeflow.enabled = False
        device = get_device(cfg)
        x0 = make_eval_x0(cfg, device)
        expected = sample_flowmatch(cfg, device, x0)
        got = sample_safeflow(cfg, device, x0)
        self.assertTrue(torch.allclose(got.samples, expected, atol=0.0, rtol=0.0))
        self.assertEqual(got.diagnostics["terminal_filter_rate"], 0.0)

    def test_euler_eval_is_safe_and_writes_diagnostics(self) -> None:
        cfg = self._cfg()
        self._ensure_backbone(cfg)
        with redirect_stdout(io.StringIO()):
            metrics = run_eval(cfg, "safeflow")
        self.assertEqual(metrics["safe_ratio"], 1.0)
        self.assertEqual(metrics["integrator"], "euler")
        self.assertEqual(metrics["nfe"], int(cfg.sample.n_steps))
        self.assertGreaterEqual(metrics["min_relaxed_fmbf_residual"], -1.0e-6)
        self.assertIn("pre_filter_safe_ratio", metrics)
        self.assertTrue((ROOT / "runs" / self.run_name / "safeflow" / "metrics.json").is_file())
        self.assertTrue(
            (ROOT / "runs" / self.run_name / "safeflow" / "metrics_euler.json").is_file()
        )

    def test_dopri5_eval_is_safe(self) -> None:
        cfg = self._cfg()
        cfg.sample.n_samples = 8
        cfg.safeflow.integrator = "dopri5"
        cfg.safeflow.dopri5.rtol = 1.0e-4
        cfg.safeflow.dopri5.atol = 1.0e-6
        self._ensure_backbone(cfg)
        with redirect_stdout(io.StringIO()):
            metrics = run_eval(cfg, "safeflow")
        self.assertEqual(metrics["safe_ratio"], 1.0)
        self.assertEqual(metrics["integrator"], "dopri5")
        self.assertGreater(metrics["nfe"], 0)
        self.assertGreaterEqual(metrics["min_relaxed_fmbf_residual"], -1.0e-6)
        self.assertTrue(
            (ROOT / "runs" / self.run_name / "safeflow" / "metrics_dopri5.json").is_file()
        )

    def test_t_on_ablation_writes_reproducible_artifacts(self) -> None:
        cfg = self._cfg()
        cfg.sample.n_samples = 4
        cfg.sample.n_steps = 4
        self._ensure_backbone(cfg)
        output_dir = ROOT / "runs" / self.run_name / "safeflow" / "t_on_ablation"
        result_dir = run_ablation(cfg, output_dir=output_dir)

        summary = json.loads((result_dir / "summary.json").read_text())
        self.assertEqual(summary["t_on_values"], [0.5, 0.7, 0.8, 0.9])
        self.assertEqual(len(summary["results"]), 4)
        self.assertTrue((result_dir / "comparison.png").is_file())
        self.assertTrue((result_dir / "u_histogram.png").is_file())
        for tag in ("t_on_0p5", "t_on_0p7", "t_on_0p8", "t_on_0p9"):
            setting_dir = result_dir / tag
            self.assertTrue((setting_dir / "config.yaml").is_file())
            self.assertTrue((setting_dir / "eval_samples.npy").is_file())
            self.assertTrue((setting_dir / "eval_samples.png").is_file())
            self.assertTrue((setting_dir / "metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
