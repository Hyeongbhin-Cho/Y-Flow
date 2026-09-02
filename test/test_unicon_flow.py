# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest

import torch

from eval.unicon_flow import ptzf_reference, qp_guidance


class TestUniConFlow(unittest.TestCase):
    def test_ptzf_reaches_zero_and_decreases(self) -> None:
        initial = torch.tensor([[2.0, 1.0]])
        early, _ = ptzf_reference(initial, 0.2, 1.0)
        late, derivative = ptzf_reference(initial, 0.9, 1.0)
        self.assertTrue(torch.all(late < early))
        self.assertTrue(torch.all(derivative <= 0))

    def test_qp_guidance_corrects_positive_residual(self) -> None:
        rho = torch.tensor([[2.0]])
        eta = torch.tensor([[[1.0, 0.0]]])
        u = qp_guidance(rho, eta, slack_weight=1.0e6)
        residual = rho + torch.einsum("bmd,bd->bm", eta, u)
        self.assertLess(float(residual[0, 0]), 1e-3)
        self.assertLess(float(u[0, 0]), 0.0)

    def test_qp_leaves_satisfied_certificate_unchanged(self) -> None:
        rho = torch.tensor([[-1.0]])
        eta = torch.tensor([[[1.0, 0.0]]])
        u = qp_guidance(rho, eta, slack_weight=100.0)
        torch.testing.assert_close(u, torch.zeros_like(u))


if __name__ == "__main__":
    unittest.main()
