from __future__ import annotations

import numpy as np
import torch

from tools.geometry import drivable_sdf
from vfm.constraints import NuScenesConstraint
from vfm.data import VehicleTrajDataset
from vfm.model import FlowForecaster
from vfm import yflow

GRID = {"x_min": -32.0, "y_min": -64.0, "res": 1.0, "size": 128, "scale": 0.25}
META = {"sample_hz": 2.0, "sdf": GRID}
CFG = {"v_max": 20.0, "a_max": 4.0, "a_cont": 4.0, "drivable_margin": 0.0, "lane_half_width": 2.0,
       "hard": ["speed", "continuity", "accel", "drivable", "static"], "proj_sweeps": 60}


def _batch(n=3, t=12, h=5):
    road = [(np.array([[-40, -4], [120, -4], [120, 4], [-40, 4]], float), [])]
    sdf = drivable_sdf(road, GRID)
    arr = {
        "hist": (np.arange(-h + 1, 1)[:, None] * [5.0, 0.0])[None].repeat(n, 0).astype(np.float32),
        "hist_mask": np.ones((n, h), bool),
        "fut": (np.arange(1, t + 1)[:, None] * [5.0, 0.0])[None].repeat(n, 0).astype(np.float32),
        "fut_mask": np.ones((n, t), bool),
        "nbr": np.zeros((n, 2, h, 2), np.float32), "nbr_mask": np.zeros((n, 2, h), bool),
        "lane": np.zeros((n, 2, 20, 2), np.float32), "lane_mask": np.zeros((n, 2, 20), bool),
        "focal_size": np.tile([[4.5, 1.9]], (n, 1)).astype(np.float32),
        "obs": np.tile([[[30.0, 1.0, 0.0, 4.5, 1.9]]], (n, 1, 1)).astype(np.float32),
        "obs_mask": np.ones((n, 1), bool),
        "sdf": np.clip(np.round(sdf / 0.25), -127, 127).astype(np.int8)[None].repeat(n, 0),
    }
    arr["lane"][:, 0, :, 0] = np.linspace(-20, 100, 20)
    arr["lane_mask"][:, 0] = True
    return {k: v for k, v in VehicleTrajDataset(arr, optional=True).tensors.items()}


def test_sdf_sign():
    sdf = drivable_sdf([(np.array([[-40, -4], [120, -4], [120, 4], [-40, 4]], float), [])], GRID)
    assert sdf[64, 40] < 0 and sdf[64 + 10, 40] > 5


def test_gt_like_is_feasible_except_static():
    b = _batch()
    cons = NuScenesConstraint(b, META, CFG)
    h = cons.h(b["fut"][:, None])
    for n in ("speed", "continuity", "accel", "drivable", "lane"):
        assert (h[n] <= 1e-3).all(), (n, h[n])
    assert (h["static"] > 0).all()


def test_projection_restores_feasibility():
    torch.manual_seed(0)
    b = _batch()
    cons = NuScenesConstraint(b, META, CFG)
    p = b["fut"][:, None].repeat(1, 4, 1, 1) + torch.randn(3, 4, 12, 2) * 4.0
    before = cons.h(p)
    q = cons.project_feasible(p, sweeps=100)
    after = cons.h(q)
    for n in ("speed", "continuity", "accel", "drivable", "static"):
        assert after[n].max() < 0.3, (n, before[n].max(), after[n].max())


def test_lane_projection_is_identity_inside_tube():
    b = _batch()
    cons = NuScenesConstraint(b, META, CFG)
    p = b["fut"][:, None].clone()
    p[..., 1] = 1.0
    torch.testing.assert_close(cons.project_physical(p), p)
    p[..., 1] = 5.0
    assert torch.allclose(cons.project_physical(p)[..., 1], torch.full_like(p[..., 1], 2.0), atol=1e-4)
    assert (cons.estimate_lipschitz(p) <= 1.0 + 1e-2).all()


def test_yflow_runs_and_gamma_zero_matches_euler():
    torch.manual_seed(0)
    b = _batch()
    m = FlowForecaster(future_steps=12, d_model=32, n_heads=4, enc_layers=1, dec_layers=1, dropout=0.0).eval()
    std = torch.full((12, 2), 5.0)
    cons = NuScenesConstraint(b, META, CFG)
    x, st = yflow.sample(m, b, std, 4, 6, cons, {"t_on": 0.5, "max_iter": 5}, torch.Generator().manual_seed(0),
                         final_sweeps=100)
    assert x.shape == (3, 4, 12, 2) and torch.isfinite(x).all()
    h = cons.h(x)
    for n in ("speed", "continuity", "accel", "drivable", "static"):
        assert h[n].max() < 0.5, (n, h[n].max())
    from vfm.flow import sample
    ref = sample(m, b, std, 4, 6, torch.Generator().manual_seed(0))
    x0, _ = yflow.sample(m, b, std, 4, 6, cons, {"t_on": 2.0, "gamma_max": 0.0, "max_iter": 0, "mu": 0.0},
                         torch.Generator().manual_seed(0), final_sweeps=0)
    torch.testing.assert_close(x0, ref, atol=1e-4, rtol=1e-4)


def _nbr_batch(nbr_x, nbr_v, y=0.0):
    b = _batch(1)
    h = b["hist"].shape[1]
    nbr = torch.zeros(1, 2, h, 2)
    nbr[0, 0, :, 0] = nbr_x + torch.arange(-h + 1, 1, dtype=torch.float32) * nbr_v * 0.5
    nbr[0, 0, :, 1] = y
    b["nbr"], b["nbr_mask"] = nbr, torch.zeros(1, 2, h, dtype=torch.bool)
    b["nbr_mask"][0, 0] = True
    b["nbr_size"] = torch.tensor([[[4.5, 1.9], [0.0, 0.0]]])
    b["nbr_yaw"] = torch.zeros(1, 2)
    b["obs_mask"] = torch.zeros_like(b["obs_mask"])
    return b


def test_coll_stopped_car_ahead_is_violated_and_projected():
    b = _nbr_batch(20.0, 0.0)
    cfg = {**CFG, "hard": ["speed", "continuity", "accel", "coll"], "coll_margin": 0.0, "proj_sweeps": 80}
    cons = NuScenesConstraint(b, META, cfg)
    p = b["fut"][:, None]
    assert cons.h(p)["coll"].item() > 1.0
    q = cons.project_feasible(p, sweeps=80)
    h = cons.h(q)
    assert h["coll"].item() < 0.05 and h["accel"].item() < 1e-3


def test_coll_constant_velocity_leader_is_free():
    b = _nbr_batch(15.0, 10.0)
    cons = NuScenesConstraint(b, META, {**CFG, "coll_margin": 0.0})
    assert cons.h(b["fut"][:, None])["coll"].item() < 0


def test_coll_horizon_limits_steps():
    b = _nbr_batch(40.0, 0.0)
    p = b["fut"][:, None]
    far = NuScenesConstraint(b, META, {**CFG, "coll_horizon_s": 6.0}).h(p)["coll"].item()
    near = NuScenesConstraint(b, META, {**CFG, "coll_horizon_s": 2.0}).h(p)["coll"].item()
    assert far > 0 and near < 0


def test_footprint_drivable_catches_corner_and_projects():
    b = _batch(1)
    b["obs_mask"] = torch.zeros_like(b["obs_mask"])
    p = b["fut"][:, None].clone()
    p[..., 1] = 3.4
    cfg = {**CFG, "hard": ["drivable_fp"], "fp_margin": 0.0, "drivable_margin": 0.0}
    cons = NuScenesConstraint(b, META, cfg)
    h = cons.h(p)
    assert h["drivable"].item() < 0 and h["drivable_fp"].item() > 0.2
    q = cons.project_feasible(p, sweeps=30)
    assert cons.h(q)["drivable_fp"].item() < 0.05


def test_neighbour_overlapping_at_t0_is_dropped():
    b = _nbr_batch(1.0, 10.0, y=0.5)
    cons = NuScenesConstraint(b, META, {**CFG, "coll_margin": 0.0})
    assert bool(cons.nbr_dropped[0, 0]) and not bool(cons.nbr_valid[0, 0])
    assert cons.h(b["fut"][:, None])["coll"].item() <= -1.0 + 1e-6


def test_fp_margin_relaxed_when_starting_off_road():
    b = _batch(1)
    b["obs_mask"] = torch.zeros_like(b["obs_mask"])
    b["sdf"] = torch.roll(b["sdf"], shifts=6, dims=1)
    cons = NuScenesConstraint(b, META, {**CFG, "fp_margin": 0.0})
    assert cons.fp_margin_b[0].item() > 1.0
    b2 = _batch(1)
    cons2 = NuScenesConstraint(b2, META, {**CFG, "fp_margin": 0.0})
    assert cons2.fp_margin_b[0].item() == 0.0
