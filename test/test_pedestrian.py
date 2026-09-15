# -*- coding: utf-8 -*-
# test/test_pedestrian.py
"""Exp-03 pedestrian dataset: NOTICE-required oracle safety, differentiability and shape checks."""

import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from data.base import build_dataset
from data.pedestrian import DIM, PedestrianConstraint, point_to_seq, seq_to_point

ROOT = Path(__file__).resolve().parents[1]


def _cfg(subset="zara1", n_eval=300):
    return OmegaConf.create({"seed": 0, "data": {"name": "pedestrian", "subset": subset, "cache_dir": None,
                                                  "source_dir": "datasets/pedestrian/default", "n_eval": n_eval}})


class TestPedestrianDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = build_dataset(_cfg())
        cls.c: PedestrianConstraint = cls.bundle.constraint
        cls.train = cls.bundle.train_raw.astype(np.float64)
        cls.rng = np.random.default_rng(0)
        h = cls.c.h(cls.train)
        cls.feasible_idx = np.flatnonzero(np.all(np.stack(list(h.values()), -1) <= 0, -1))
        # random rows: the pkl starts with an atypical block (mostly infeasible windows)
        cls.sample_idx = cls.rng.choice(cls.train.shape[0], 64, replace=False)
        cls.noisy = cls.train[cls.sample_idx] + cls.rng.normal(size=(64, DIM)) * 0.6

    def test_bundle_shapes_and_frame(self):
        self.assertEqual(self.train.shape[1], DIM)
        seq = point_to_seq(self.train[:5])
        self.assertEqual(seq.shape, (5, 20, 2))
        self.assertTrue(np.allclose(seq[:, 7], 0.0))
        self.assertTrue(np.allclose(seq_to_point(seq), self.train[:5]))
        self.assertTrue(np.abs(seq[:, 6, 1]).max() < 1e-3)          # rotated: frame-6 vector on the x axis

    def test_gt_is_feasible_and_fixed_by_P(self):
        h = self.c.h(self.train)
        safe = np.all(np.stack(list(h.values()), -1) <= 0, -1)
        self.assertGreater(safe.mean(), 0.98)
        x = self.train[safe][:200]
        self.assertLess(np.abs(self.c.project_physical(x) - x).max(), 1e-6)

    def test_project_feasible_strict(self):
        for p in (self.noisy, self.rng.normal(size=(32, DIM)) * 20.0):
            out = self.c.project_feasible(p, buffer=1e-4)
            for name, v in self.c.h(out).items():
                self.assertLessEqual(float(v.max()), -1e-4 + 1e-9, name)
            self.assertEqual(float(self.c.cost(out).max()), 0.0)
        out32 = self.c.project_feasible(self.noisy.astype(np.float32), buffer=1e-4).astype(np.float32)
        self.assertTrue(all(float(v.max()) <= 0.0 for v in self.c.h(out32).values()))

    def test_P_nonexpansive(self):
        a, b = self.noisy[:32], self.noisy[32:64]
        pa, pb = self.c.project_physical(a), self.c.project_physical(b)
        ratio = np.linalg.norm(pa - pb, axis=-1) / np.linalg.norm(a - b, axis=-1)
        self.assertLessEqual(ratio.max(), 1.0 + 1e-6)
        self.assertTrue(np.allclose(self.c.estimate_lipschitz(a), 1.0))

    def test_torch_paths_and_autograd(self):
        p = torch.tensor(self.noisy, dtype=torch.float32, requires_grad=True)
        h = self.c.h(p)
        for name, v in h.items():
            self.assertEqual(tuple(v.shape), (64,), name)
            self.assertTrue(np.allclose(v.detach().numpy(), self.c.h(self.noisy)[name], atol=1e-4), name)
        cost = self.c.cost(p)
        (g,) = torch.autograd.grad(cost.sum(), p)
        self.assertTrue(torch.isfinite(g).all())
        pf = self.c.project_feasible(p.detach(), buffer=1e-4)
        self.assertTrue(all(float(v.max()) <= 0.0 for v in self.c.h(pf).values()))
        pp = self.c.project_physical(p.detach()).numpy()
        self.assertTrue(np.allclose(pp, self.c.project_physical(self.noisy), atol=2e-3))

    def test_energy_is_distance_and_stable(self):
        p = self.noisy[:8]
        g = self.c.energy_grad(p, slack=0.01)
        target = self.c.project_feasible(p, buffer=0.01)
        self.assertTrue(np.allclose(g, p - target))
        e = self.c.energy(torch.tensor(p, dtype=torch.float64), slack=0.01).numpy()
        self.assertTrue(np.allclose(e, 0.5 * ((p - target) ** 2).sum(-1)))
        feasible = self.train[self.feasible_idx[:4]]
        self.assertLess(np.abs(self.c.energy_grad(feasible, slack=0.0)).max(), 1e-6)
        x = p.copy()
        for _ in range(10):                                   # GuideFlow refinement, eta_max = 0.5
            x = x - 0.5 * self.c.energy_grad(x, slack=0.01)
        self.assertTrue(np.isfinite(x).all())
        self.assertLess(float(self.c.cost(x).max()), float(self.c.cost(p).max()))

    def test_progress_and_command_bins(self):
        prog = self.c.progress(self.train)
        self.assertTrue(((prog >= 0.0) & (prog <= 1.0)).all())
        bins = self.c.command_bins(self.train, 5)
        self.assertEqual(bins.dtype.kind, "i")
        self.assertTrue(((bins >= 0) & (bins <= 4)).all())
        self.assertGreater(len(np.unique(bins)), 1)

    def test_fmbf_gradients_and_terminal_filter(self):
        fmbf = self.c.get_fmbf(radius_eps=1e-2, tube_margin=2e-3, box_temperature=1e-3)
        p = torch.tensor(self.noisy[:8], dtype=torch.float64)
        vals, grads = fmbf.values_and_gradients(p)
        self.assertEqual(tuple(vals.shape), (8, 3))
        self.assertEqual(tuple(grads.shape), (8, 3, DIM))
        eps = 1e-6
        for j in (2, 20):
            q = p.clone()
            q[:, j] += eps
            fd = (fmbf.values_and_gradients(q)[0] - vals) / eps
            self.assertTrue(torch.allclose(fd, grads[:, :, j], atol=1e-4), j)
        pick = self.rng.choice(self.feasible_idx, 256, replace=False)
        gt = torch.tensor(self.train[pick], dtype=torch.float64)
        gt_vals, _ = fmbf.values_and_gradients(gt)
        self.assertGreater(float((gt_vals >= 0).all(dim=-1).float().mean()), 0.95)
        out, stats = fmbf.terminal_filter(self.noisy[:16])
        self.assertGreater(stats.filtered, 0)
        self.assertTrue(all(float(v.max()) <= 1e-7 for v in self.c.h(out).values()))

    def test_anchor_vocabulary(self):
        anchors = self.c.build_anchor_vocabulary(self.train[:2000], n_anchors=16)
        self.assertEqual(anchors.shape, (16, DIM))
        self.assertTrue(all(float(v.max()) <= 0.0 for v in self.c.h(anchors).values()))


if __name__ == "__main__":
    unittest.main()
