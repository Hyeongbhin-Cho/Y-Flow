# -*- coding: utf-8 -*-
# test/test_clevrer_methods.py

from __future__ import annotations

import unittest

import torch
from omegaconf import OmegaConf

from data.clevrer_state import CLEVRERStateConstraint, CLEVRERStateLayout, CLEVRERStateMeta
from eval.clevrer_guide_flow import sample_state as sample_guideflow
from eval.clevrer_hard_flow import sample_state as sample_hardflow
from eval.clevrer_safe_flow import sample_state as sample_safeflow
from eval.clevrer_unicon_flow import sample_state as sample_uniconflow
from eval.clevrer_y_flow import sample_state as sample_yflow
from model.clevrer_flow import CLEVRERVelocityNet
from sample.euler import EulerSampler
from train.flow_match import ConditionalFlowMatching


def _meta(layout: CLEVRERStateLayout) -> CLEVRERStateMeta:
    return CLEVRERStateMeta(
        n_slots=layout.n_slots,
        n_frames=layout.n_frames,
        fps=16.0,
        dt=1.0 / 16.0,
        r_xy=12.0,
        z0=0.20,
        tau_z=0.05,
        v_max=3.2,
        tau_kin=0.20,
        tau_acc=0.60,
        d0=0.45,
        d_min=0.38,
        tau_col=0.20,
        delta_v=0.15,
        eps_ident=0.05,
        eps_null=1.0e-3,
        mean=(0.0,) * layout.dim,
        std=(1.0,) * layout.dim,
    )


def _cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "sample": {"n_steps": 3},
            "hardflow": {"t_on": 0.0, "lambda_oc": 10.0, "max_iter": 2, "safety_buffer": 1e-4},
            "yflow": {
                "t_on": 0.0,
                "lambda_oc": 10.0,
                "mu": 1.0,
                "max_iter": 2,
                "delta": 0.1,
                "gamma_max": 1.0,
                "safety_buffer": 1e-4,
            },
            "uniconflow": {
                "free_until": 0.0,
                "ptzf_rate": 1.0,
                "gamma": 1.0,
                "slack_weight": 100.0,
                "max_guidance_norm": 20.0,
                "terminal_refinement": True,
                "safety_buffer": 1e-4,
            },
            "safeflow": {
                "enabled": True,
                "integrator": "euler",
                "t_on": 0.0,
                "phi_schedule": "paper_piecewise",
                "phi0": 1.0,
                "phi_gamma": 0.9,
                "phi_omega": 3.0,
                "slack_weight": 1.0,
                "terminal_eps": 1.0e-3,
                "smooth_box_temperature": 0.05,
                "qp_active_tol": 1.0e-6,
                "terminal_filter": {
                    "enabled": True,
                    "max_iter": 8,
                    "ftol": 1.0e-7,
                    "constraint_tol": 1.0e-4,
                },
            },
            "guideflow": {
                "cvf": True,
                "lambda_cvf": 0.1,
                "cvf_t_on": 0.0,
                "cf": True,
                "cf_mode": "interp",
                "k_c": 1,
                "rfe": True,
                "tau_star": 0.0,
                "eta_max": 0.1,
                "n_refine": 1,
                "slack": 0.01,
            },
        }
    )


class TestCLEVRERConstrainedSamplers(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = CLEVRERStateLayout(n_slots=3, n_frames=4)
        self.net = CLEVRERVelocityNet(
            dim=self.layout.dim, hidden=(16,), time_embed_dim=8, cond_dim=8
        )
        self.cfm = ConditionalFlowMatching(sigma_min=0.0)
        self.cons = CLEVRERStateConstraint(_meta(self.layout), self.layout)
        self.mean = torch.zeros(self.layout.dim)
        self.std = torch.ones(self.layout.dim)
        self.x0 = torch.randn(2, self.layout.dim)
        self.cond = torch.randn(2, 8)
        self.cfg = _cfg()

    def _finite(self, z: torch.Tensor) -> None:
        self.assertEqual(tuple(z.shape), (2, self.layout.dim))
        self.assertTrue(torch.isfinite(z).all())

    def test_flowmatch_and_five_methods_are_finite(self) -> None:
        z_fm = EulerSampler(n_steps=3).sample(self.net, self.cfm, self.x0, cond=self.cond)
        self._finite(z_fm)
        self._finite(sample_hardflow(self.cfg, self.net, self.cfm, self.x0, self.cond, self.cons, self.mean, self.std))
        self._finite(sample_yflow(self.cfg, self.net, self.cfm, self.x0, self.cond, self.cons, self.mean, self.std))
        self._finite(sample_uniconflow(self.cfg, self.net, self.cfm, self.x0, self.cond, self.cons, self.mean, self.std))
        self._finite(sample_safeflow(self.cfg, self.net, self.cfm, self.x0, self.cond, self.cons, self.mean, self.std))
        anchors = torch.randn(4, self.layout.dim)
        self._finite(
            sample_guideflow(
                self.cfg, self.net, self.cfm, self.x0, self.cond, self.cons, self.mean, self.std, anchors_z=anchors
            )
        )


if __name__ == "__main__":
    unittest.main()
