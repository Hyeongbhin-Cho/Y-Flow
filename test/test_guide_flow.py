# -*- coding: utf-8 -*-
# test/test_guide_flow.py

from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from constraints.swiss_roll import SwissRollConstraint
from data.swiss_roll import build_swiss_roll
from eval.evaluate import run_eval
from eval.guide_flow import (
    _constrain_velocity,
    build_anchor_vocabulary,
    energy_grad,
    energy_weight,
)
from train.guide_flow import ensure_flowmatch_ckpt
from utils.paths import ROOT, flowmatch_ckpt


class TestGuideFlow(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.run_name = "unittest_guideflow"
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
        cfg.data.n_train = 256
        cfg.data.n_eval = 32
        cfg.data.regenerate = True
        cfg.train.steps = 4
        cfg.train.batch_size = 16
        cfg.sample.n_samples = 16
        cfg.sample.n_steps = 8
        cfg.log.plot_every = 4
        cfg.device = "cpu"
        cfg.run_name = self.run_name
        cfg.guideflow.n_anchors = 32
        cfg.guideflow.k_c = 4
        cfg.guideflow.n_refine = 2
        return cfg

    def test_energy_weight_schedule(self) -> None:
        self.assertEqual(energy_weight(0.2, 0.5, 0.4), 0.0)
        self.assertAlmostEqual(energy_weight(0.5, 0.5, 0.4), 0.0)
        self.assertAlmostEqual(energy_weight(0.75, 0.5, 0.4), 0.2)
        self.assertAlmostEqual(energy_weight(1.0, 0.5, 0.4), 0.4)
        self.assertAlmostEqual(energy_weight(1.5, 0.5, 0.4), 0.4)

    def test_cvf_reflection_and_magnitude(self) -> None:
        v = torch.tensor([[1.0, 2.0], [-0.5, 0.25]])
        vc = torch.tensor([[0.0, 3.0], [1.0, 1.0]])
        self.assertTrue(torch.allclose(_constrain_velocity(v, vc, 0.0), v))

        reflected = _constrain_velocity(v, vc, 1.0)
        self.assertTrue(torch.allclose(reflected.norm(dim=-1), v.norm(dim=-1), atol=1e-6))

        orthogonal = _constrain_velocity(v, vc, 0.5)
        self.assertTrue(torch.allclose((orthogonal * vc).sum(dim=-1), torch.zeros(2), atol=1e-6))

        lam = 0.1
        small = _constrain_velocity(v, vc, lam)
        rel = (small.norm(dim=-1) - v.norm(dim=-1)).abs() / v.norm(dim=-1)
        self.assertTrue(bool((rel <= 2.0 * lam + 1e-6).all()))

    def test_anchor_vocabulary_is_feasible(self) -> None:
        cfg = self._cfg()
        bundle = build_swiss_roll(cfg)
        constraint = SwissRollConstraint(bundle["meta"])
        anchors = build_anchor_vocabulary(bundle["train_raw"], constraint, 32, int(cfg.seed))
        self.assertEqual(anchors.shape[1], 2)
        self.assertLessEqual(anchors.shape[0], 32)
        h = constraint.h(anchors)
        for name in ("tube", "core", "box"):
            self.assertTrue(np.all(h[name] <= 0.0))

    def test_energy_grad_is_zero_inside_feasible_set(self) -> None:
        cfg = self._cfg()
        bundle = build_swiss_roll(cfg)
        constraint = SwissRollConstraint(bundle["meta"])
        on_manifold = constraint.curve(16)
        g = energy_grad(on_manifold, constraint, 1.0, 1.0, 1.0, 0.0, 0.01)
        self.assertTrue(np.allclose(g, 0.0, atol=1e-8))

        outside = on_manifold * 1.5
        g_out = energy_grad(outside, constraint, 1.0, 1.0, 1.0, 0.0, 0.01)
        self.assertGreater(float(np.abs(g_out).max()), 0.0)

    def test_cfg_off_by_default_uses_frozen_backbone(self) -> None:
        cfg = self._cfg()
        self.assertFalse(bool(cfg.guideflow.guidance.enabled))
        with redirect_stdout(io.StringIO()):
            ensure_flowmatch_ckpt(cfg)
            metrics = run_eval(cfg, "guideflow")
        self.assertIn("safe_ratio", metrics)
        self.assertFalse((ROOT / "runs" / self.run_name / "guideflow" / "last.pt").is_file())

    def test_conditional_training_then_cfg_eval(self) -> None:
        cfg = self._cfg()
        cfg.guideflow.guidance.enabled = True
        cfg.guideflow.guidance.cond_steps = 4
        cfg.guideflow.guidance.cond_batch_size = 16
        cfg.guideflow.guidance.cond_hidden = [16, 16]

        from train.guide_flow import run_train_guideflow

        with redirect_stdout(io.StringIO()):
            ckpt = run_train_guideflow(cfg)
        self.assertTrue(Path(ckpt).is_file())

        with redirect_stdout(io.StringIO()):
            metrics = run_eval(cfg, "guideflow")
        self.assertEqual(metrics["method"], "guideflow")
        self.assertIn("safe_ratio", metrics)

    def test_rfe_loss_training_then_eval(self) -> None:
        cfg = self._cfg()
        cfg.guideflow.rfe_train.rfe_loss = True
        cfg.guideflow.guidance.cond_steps = 4
        cfg.guideflow.guidance.cond_batch_size = 16

        from eval.guide_flow import owns_backbone
        from train.guide_flow import run_train_guideflow

        self.assertTrue(owns_backbone(cfg))
        self.assertFalse(bool(cfg.guideflow.guidance.enabled))
        with redirect_stdout(io.StringIO()):
            ckpt = run_train_guideflow(cfg)
        self.assertTrue(Path(ckpt).is_file())

        import torch as th

        extra = th.load(ckpt, map_location="cpu", weights_only=False)["extra"]
        self.assertEqual(int(extra["intent_dim"]), 0)
        self.assertTrue(bool(extra["rfe_loss"]))

        with redirect_stdout(io.StringIO()):
            metrics = run_eval(cfg, "guideflow")
        self.assertIn("safe_ratio", metrics)

    def test_energy_torch_matches_analytic_gradient(self) -> None:
        import torch as th

        from eval.guide_flow import energy_grad, energy_torch

        cfg = self._cfg()
        bundle = build_swiss_roll(cfg)
        constraint = SwissRollConstraint(bundle["meta"])
        pts = np.concatenate([constraint.curve(8) * 1.4, constraint.curve(8) * 0.2], axis=0)

        p = th.tensor(pts, dtype=th.float64, requires_grad=True)
        energy_torch(p, constraint, 1.0, 1.0, 1.0, 0.0, 0.01).sum().backward()
        expected = energy_grad(pts, constraint, 1.0, 1.0, 1.0, 0.0, 0.01)
        self.assertTrue(np.allclose(p.grad.numpy(), expected, atol=1e-8))

    def test_cfg_eval_without_checkpoint_raises(self) -> None:
        cfg = self._cfg()
        cfg.guideflow.guidance.enabled = True
        from eval.evaluate import make_eval_x0
        from eval.guide_flow import sample
        from utils.device import get_device

        device = get_device(cfg)
        with self.assertRaises(FileNotFoundError):
            sample(cfg, device, make_eval_x0(cfg, device))

    def test_command_bins_and_ego_progress(self) -> None:
        cfg = self._cfg()
        bundle = build_swiss_roll(cfg)
        constraint = SwissRollConstraint(bundle["meta"])
        curve = constraint.curve(32)

        from eval.guide_flow import command_bins, ego_progress

        bins = command_bins(curve, constraint, 5)
        self.assertEqual(bins.min(), 0)
        self.assertEqual(bins.max(), 4)
        self.assertTrue(np.all(np.diff(bins) >= 0))

        ep = ego_progress(curve, constraint)
        self.assertAlmostEqual(float(ep[0]), 0.0, places=5)
        self.assertAlmostEqual(float(ep[-1]), 1.0, places=5)
        self.assertTrue(np.all(np.diff(ep) >= -1e-9))

    def test_guidance_scale_interpolates(self) -> None:
        import torch as th

        from eval.guide_flow import _GuidedVelocity

        class Fake:
            def __call__(self, x, t, intent, reward, im, rm):
                return th.ones_like(x) * (2.0 if float(im[0]) > 0.5 else 1.0)

        g = _GuidedVelocity(Fake(), 0.0, False)
        x = th.zeros(3, 2)
        t = th.zeros(3)
        self.assertTrue(th.allclose(g(x, t), th.ones(3, 2)))
        self.assertTrue(th.allclose(_GuidedVelocity(Fake(), 1.0, False)(x, t), th.full((3, 2), 2.0)))
        self.assertTrue(th.allclose(_GuidedVelocity(Fake(), 0.5, False)(x, t), th.full((3, 2), 1.5)))

    def test_train_reuses_ckpt_then_eval(self) -> None:
        cfg = self._cfg()
        buf = io.StringIO()
        with redirect_stdout(buf):
            ckpt = ensure_flowmatch_ckpt(cfg)
        self.assertTrue(Path(ckpt).is_file())
        self.assertTrue(flowmatch_ckpt(cfg).is_file())
        self.assertIn("training flowmatch", buf.getvalue())

        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            again = ensure_flowmatch_ckpt(cfg)
        self.assertIn("already exists", buf2.getvalue())
        self.assertEqual(again, ckpt)

        metrics = run_eval(cfg, "guideflow")
        self.assertEqual(metrics["method"], "guideflow")
        self.assertIn("safe_ratio", metrics)
        self.assertTrue((ROOT / "runs" / self.run_name / "guideflow" / "metrics.json").is_file())

    def test_disabling_all_modules_matches_flowmatch(self) -> None:
        cfg = self._cfg()
        with redirect_stdout(io.StringIO()):
            ensure_flowmatch_ckpt(cfg)
        cfg.guideflow.cvf = False
        cfg.guideflow.cf = False
        cfg.guideflow.rfe = False

        from eval.evaluate import make_eval_x0
        from eval.flow_match import sample as fm_sample
        from eval.guide_flow import sample as gf_sample
        from utils.device import get_device

        device = get_device(cfg)
        x0 = make_eval_x0(cfg, device)
        self.assertTrue(torch.allclose(gf_sample(cfg, device, x0), fm_sample(cfg, device, x0)))


if __name__ == "__main__":
    unittest.main()
