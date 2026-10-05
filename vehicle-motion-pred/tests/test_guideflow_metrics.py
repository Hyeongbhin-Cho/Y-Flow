from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tools.guideflow_metrics import boxes, collisions, comfort, headings, lane_metrics, sat_overlap

T = 12


def _straight(v=5.0, n=1, k=1):
    p = np.zeros((n, k, T, 2), np.float32)
    p[..., 0] = np.arange(1, T + 1) * v * 0.5
    return p


def test_sat_overlap():
    a = boxes(np.array([0.0, 0.0]), np.array(0.0), 4.0, 2.0)
    assert sat_overlap(a, boxes(np.array([3.0, 0.0]), np.array(0.0), 4.0, 2.0))
    assert not sat_overlap(a, boxes(np.array([4.5, 0.0]), np.array(0.0), 4.0, 2.0))
    assert not sat_overlap(a, boxes(np.array([0.0, 2.5]), np.array(0.0), 4.0, 2.0))
    assert sat_overlap(a, boxes(np.array([0.0, 2.5]), np.array(np.pi / 2), 4.0, 2.0))
    assert not sat_overlap(a, boxes(np.array([3.3, 1.8]), np.array(np.pi / 4), 2.0, 1.0))


def test_headings_hold_when_stopped():
    p = _straight()
    p[0, 0, 6:] = p[0, 0, 5]
    y = headings(p)
    np.testing.assert_allclose(y, 0.0, atol=1e-6)


def test_collisions_with_agent_in_path():
    pred = _straight()
    ag = {"ag_fut": np.zeros((1, 2, T, 2), np.float32), "ag_yaw": np.zeros((1, 2, T), np.float32),
          "ag_mask": np.ones((1, 2, T), bool), "ag_size": np.array([[[4.5, 1.9], [4.5, 1.9]]], np.float32),
          "ag_cls": np.array([[0, 1]], np.int8)}
    ag["ag_fut"][0, 0] = [15.0, 0.0]
    ag["ag_fut"][0, 1] = [15.0, 10.0]
    hit, by = collisions(pred, np.array([[4.5, 1.9]], np.float32), ag)
    assert hit[0, 0] and by[0][0, 0] and not by[1][0, 0]
    ag["ag_fut"][0, 0] = [15.0, 4.0]
    hit, _ = collisions(pred, np.array([[4.5, 1.9]], np.float32), ag)
    assert not hit[0, 0]
    ag["ag_fut"][0, 0] = [15.0, 0.0]
    ag["ag_mask"][:] = False
    hit, _ = collisions(pred, np.array([[4.5, 1.9]], np.float32), ag)
    assert not hit[0, 0]


def _lanes():
    lane = np.zeros((1, 2, 20, 2), np.float32)
    lane[0, 0, :, 0] = np.linspace(-60, 60, 20)
    mask = np.zeros((1, 2, 20), bool)
    mask[0, 0] = True
    return lane, mask


def test_ddc_and_lk():
    lane, mask = _lanes()
    fwd = _straight()
    back = -fwd
    off = fwd.copy()
    off[..., 1] = 5.0
    pred = np.concatenate([fwd, back, off], axis=1)
    ddc, lk = lane_metrics(pred, lane, mask, 2.0, 1.75)
    assert ddc.tolist() == [[False, True, False]]
    assert lk.tolist() == [[False, False, True]]


def test_comfort():
    hist = np.stack([np.arange(-4, 1) * 2.5, np.zeros(5)], -1)[None].astype(np.float32)
    hmask = np.ones((1, 5), bool)
    smooth = _straight(5.0)
    stop = smooth.copy()
    stop[0, 0, 2:] = stop[0, 0, 1]
    ok = comfort(np.concatenate([smooth, stop], axis=1), hist, hmask, 2.0)
    assert ok.tolist() == [[True, False]]


def test_extract_agents_with_fake_helper(tmp_path, monkeypatch):
    import nuscenes.eval.prediction.splits as splits
    from tools import extract_agents as ea

    yaw = np.deg2rad(30.0)
    q = [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]
    d = np.array([np.cos(yaw), np.sin(yaw)])

    def ann(inst, k, cat="vehicle.car"):
        base = np.array([100.0, 50.0]) + (d * 10 if inst != "focal" else 0)
        return {"instance_token": inst, "translation": [*(base + d * 5 * k), 0.0], "rotation": q,
                "category_name": cat, "size": [2.0, 5.0, 1.5]}

    class H:
        def get_sample_annotation(self, inst, sample):
            return ann(inst, 0)

        def get_annotations_for_sample(self, sample):
            return [ann("focal", 0), ann("car", 0), {**ann("cone", 0, "movable_object.trafficcone"), "size": [0.4, 0.4, 1]}]

        def get_future_for_agent(self, inst, sample, seconds, in_agent_frame, just_xy):
            return [] if inst == "cone" else [ann(inst, k) for k in range(1, 13)]

    monkeypatch.setattr(splits, "get_prediction_challenge_split", lambda s, dataroot: ["focal_s0"])
    args = SimpleNamespace(root="x", limit=0, n_agents=4, radius=80.0, workers=1)
    ea.extract_split(H(), "val", args, tmp_path)
    z = np.load(tmp_path / "val_agents.npz")
    assert z["ag_mask"][0, 0].all() and z["ag_mask"][0, 1].all() and not z["ag_mask"][0, 2:].any()
    np.testing.assert_allclose(z["ag_fut"][0, 0, :, 0], 10 + 5 * np.arange(1, 13), atol=1e-3)
    np.testing.assert_allclose(z["ag_fut"][0, 0, :, 1], 0.0, atol=1e-3)
    np.testing.assert_allclose(z["ag_yaw"][0, 0], 0.0, atol=1e-5)
    assert z["ag_cls"][0, :2].tolist() == [0, 3]
    np.testing.assert_allclose(z["ag_size"][0, 0], [5.0, 2.0])


def test_comfort_batched_matches_single():
    hist = np.stack([np.arange(-4, 1) * 2.5, np.zeros(5)], -1)[None].astype(np.float32)
    hist3 = np.concatenate([hist, hist * 2, hist * 0.5])
    hmask = np.ones((3, 5), bool)
    hmask[2, -2] = False
    pred = np.concatenate([_straight(5.0), _straight(10.0), _straight(1.0)])
    batched = comfort(pred, hist3, hmask, 2.0)
    single = np.concatenate([comfort(pred[i:i + 1], hist3[i:i + 1], hmask[i:i + 1], 2.0) for i in range(3)])
    assert batched.shape == (3, 1)
    np.testing.assert_array_equal(batched, single)
