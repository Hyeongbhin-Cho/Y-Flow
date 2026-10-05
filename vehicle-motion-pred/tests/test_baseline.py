from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.geometry import pick_lanes, resample, split_polyline, to_local
from tools.make_synthetic import make
from vfm.config import Cfg
from vfm.data import VehicleTrajDataset, future_std, validate
from vfm.evaluate import constant_velocity
from vfm.flow import fm_loss, sample
from vfm.metrics import displacement, kinematics, summarize
from vfm.model import FlowForecaster


def _batch(n=8, seed=0):
    arr = make(n, np.random.default_rng(seed))
    ds = VehicleTrajDataset(arr)
    return arr, {k: v for k, v in ds.tensors.items()}


def test_to_local_heading_is_plus_x():
    yaw = 0.7
    origin = np.array([10.0, -3.0])
    ahead = origin + 5.0 * np.array([np.cos(yaw), np.sin(yaw)])
    left = origin + 2.0 * np.array([-np.sin(yaw), np.cos(yaw)])
    out = to_local(np.stack([origin, ahead, left]), origin, yaw)
    np.testing.assert_allclose(out, [[0, 0], [5, 0], [0, 2]], atol=1e-5)


def test_polyline_utils():
    line = np.stack([np.linspace(0, 45, 10), np.zeros(10)], 1)
    parts = split_polyline(line, 20.0)
    assert len(parts) == 3
    r = resample(line, 7)
    np.testing.assert_allclose(r[:, 0], np.linspace(0, 45, 7), atol=1e-6)
    lanes, mask = pick_lanes([line + [100, 0], line], 1, 5)
    assert mask.all() and lanes[0, 0, 0] == pytest.approx(0.0)


def test_validate_rejects_unobserved_origin():
    arr = make(4, np.random.default_rng(0))
    arr["hist_mask"][0, -1] = False
    with pytest.raises(ValueError):
        validate(arr)


def test_constant_velocity_exact_on_straight_line():
    arr = make(16, np.random.default_rng(1))
    arr["fut"] = constant_velocity(arr, arr["fut"].shape[1])[:, 0]
    pred = constant_velocity(arr, arr["fut"].shape[1])
    ade, fde = displacement(pred, arr["fut"], arr["fut_mask"])
    assert ade.max() < 1e-5 and fde.max() < 1e-5


def test_fde_uses_last_valid_step():
    fut = np.zeros((1, 4, 2), np.float32)
    fut[0, :, 0] = [1, 2, 3, 99]
    mask = np.array([[True, True, True, False]])
    pred = np.zeros((1, 2, 4, 2), np.float32)
    pred[0, 1, :, 0] = [1, 2, 3, 0]
    ade, fde = displacement(pred, fut, mask)
    assert fde[0, 1] == 0.0 and fde[0, 0] == pytest.approx(3.0)
    assert ade[0, 1] == 0.0 and ade[0, 0] == pytest.approx(2.0)


def test_kinematics_constant_speed():
    pred = (np.arange(1, 11, dtype=np.float32)[:, None] * [2.0, 0.0])[None, None]
    speed, acc = kinematics(pred, 5.0)
    np.testing.assert_allclose(speed, 10.0, atol=1e-5)
    np.testing.assert_allclose(acc, 0.0, atol=1e-4)


def test_summarize_min_over_k():
    fut = np.zeros((2, 3, 2), np.float32)
    pred = np.stack([np.full((3, 2), 5.0), np.zeros((3, 2))])[None].repeat(2, 0).astype(np.float32)
    m = summarize(pred, fut, np.ones((2, 3), bool), {"sample_hz": 1.0, "miss_threshold_m": 2.0})
    assert m["minADE_2"] == 0.0 and m["ADE_1"] > 7.0 and m["MR_2@2m"] == 0.0


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return FlowForecaster(future_steps=30, d_model=32, n_heads=4, enc_layers=1, dec_layers=2, dropout=0.0).eval()


def test_shapes_and_sampling(model):
    arr, b = _batch()
    std = torch.from_numpy(future_std(arr))
    x = sample(model, b, std, k=5, n_steps=3, generator=torch.Generator().manual_seed(0))
    assert x.shape == (8, 5, 30, 2) and torch.isfinite(x).all()


def test_samples_are_independent_queries(model):
    _, b = _batch()
    mem, pad = model.encode(b)
    x = torch.randn(8, 4, 30, 2)
    t = torch.rand(8)
    v = model.velocity(x, t, mem, pad)
    perm = torch.tensor([2, 0, 3, 1])
    v_perm = model.velocity(x[:, perm], t, mem, pad)
    torch.testing.assert_close(v_perm, v[:, perm], atol=1e-5, rtol=1e-5)
    v_one = model.velocity(x[:, :1], t, mem, pad)
    torch.testing.assert_close(v_one, v[:, :1], atol=1e-5, rtol=1e-5)


def test_padding_is_ignored(model):
    _, b = _batch()
    mem, pad = model.encode(b)
    b2 = dict(b)
    b2["nbr"] = torch.where(b["nbr_mask"][..., None], b["nbr"], torch.full_like(b["nbr"], 1e3))
    b2["lane"] = torch.where(b["lane_mask"][..., None], b["lane"], torch.full_like(b["lane"], -1e3))
    mem2, _ = model.encode(b2)
    torch.testing.assert_close(mem[~pad], mem2[~pad], atol=1e-5, rtol=1e-5)


def test_map_changes_encoding(model):
    _, b = _batch()
    mem, _ = model.encode(b)
    b2 = dict(b)
    b2["lane"] = b["lane"] + 3.0
    mem2, _ = model.encode(b2)
    assert (mem[:, 0] - mem2[:, 0]).abs().max() > 1e-4


def test_loss_decreases_on_fixed_batch():
    torch.manual_seed(0)
    m = FlowForecaster(future_steps=30, d_model=32, n_heads=4, enc_layers=1, dec_layers=1, dropout=0.0)
    arr, b = _batch(32)
    std = torch.from_numpy(future_std(arr))
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    losses = []
    for _ in range(150):
        loss = fm_loss(m, b, std, n_samples=4)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-10:]) < 0.5 * np.mean(losses[:10])


def test_masked_future_steps_do_not_contribute():
    torch.manual_seed(0)
    m = FlowForecaster(future_steps=30, d_model=32, n_heads=4, enc_layers=1, dec_layers=1, dropout=0.0)
    arr, b = _batch(4)
    b["fut_mask"][:, 20:] = False
    std = torch.from_numpy(future_std(arr))
    torch.manual_seed(1)
    l1 = fm_loss(m, b, std)
    b2 = dict(b)
    b2["fut"] = b["fut"].clone()
    b2["fut"][:, 20:] += 1e3
    torch.manual_seed(1)
    l2 = fm_loss(m, b2, std)
    assert l1.item() == pytest.approx(l2.item(), rel=1e-6)


def test_config_overrides(tmp_path):
    from vfm.config import load_config
    p = tmp_path / "c.yaml"
    p.write_text("a: 1\nb:\n  c: 2\n")
    cfg = load_config(p, ["b.c=5", "b.d=[1, 2]"])
    assert isinstance(cfg, Cfg) and cfg.b.c == 5 and cfg.b.d == [1, 2]


def test_future_std_modes():
    arr = make(64, np.random.default_rng(3))
    arr["fut_mask"][:, -5:] = False
    per = future_std(arr)
    glob = future_std(arr, "global")
    assert per.shape == (30, 2) and glob.shape == (2,)
    assert per[0].max() < per[20].max()
    assert np.isfinite(per).all() and (per >= 0.05).all()


def test_tune_draw_and_apply():
    from vfm.tune import SPACE, apply, draw, objective
    rng = np.random.default_rng(0)
    p = draw(rng)
    assert set(p) == {k for k, v in SPACE.items() if v[0] != "offset"} and 0.3 <= p["yflow.t_on"] <= 0.8
    q = draw(rng, Cfg({"constraints": {"fp_margin": 0.7, "coll_margin": -0.3}}))
    assert 0.7 <= q["constraints.fp_margin"] <= 1.7 and -0.3 <= q["constraints.coll_margin"] <= 0.2
    cfg = apply(Cfg({"yflow": {"t_on": 0.5}, "constraints": {}}), p)
    assert cfg.yflow.t_on == p["yflow.t_on"] and cfg.constraints.w_lane == p["constraints.w_lane"]
    assert objective({"minADE_10": 1.0, "all_hard_safe": 0.5, "MR_10@2m": 0.2}, 10, 5.0, 1.0) == 1.0 + 2.5 + 0.2
    m = {"minADE_10": 1.0, "all_hard_safe": 1.0, "MR_10@2m": 0.0, "indep_coll_scene_gtok": 0.3, "indep_offroad_scene_gtok": 0.1}
    assert abs(objective(m, 10, 5.0, 1.0, 2.0, 2.0) - 1.8) < 1e-9
