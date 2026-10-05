from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("nuscenes")

import nuscenes.eval.prediction.splits as splits

from tools import convert_nuscenes as cn
from vfm.data import load_split

YAW = np.deg2rad(120.0)
DIR = np.array([np.cos(YAW), np.sin(YAW)])
O = np.array([500.0, 800.0])
QUAT = [float(np.cos(YAW / 2)), 0.0, 0.0, float(np.sin(YAW / 2))]


def _pos(inst, k):
    speed, off = (10.0, 0.0) if inst == "focal" else (6.0, 3.0)
    left = np.array([-DIR[1], DIR[0]])
    return O + left * off + DIR * speed * 0.5 * k


class FakeHelper:

    def get_sample_annotation(self, inst, sample):
        return {"instance_token": inst, "translation": [*_pos(inst, 0), 0.0], "rotation": QUAT,
                "category_name": "vehicle.car", "attribute_tokens": ["moving"], "size": [1.9, 4.5, 1.6]}

    def get_past_for_agent(self, inst, sample, seconds, in_agent_frame):
        n = 4 if inst == "focal" else 2
        return np.stack([_pos(inst, -k) for k in range(1, n + 1)])

    def get_future_for_agent(self, inst, sample, seconds, in_agent_frame):
        return np.stack([_pos(inst, k) for k in range(1, int(seconds * 2) + 1)])

    def get_annotations_for_sample(self, sample):
        parked = {"instance_token": "parked", "translation": [*(O + DIR * 20.0 + np.array([-DIR[1], DIR[0]]) * -3.0), 0.0],
                  "rotation": QUAT, "category_name": "vehicle.car", "attribute_tokens": ["parked"],
                  "size": [2.0, 5.0, 1.6]}
        return [self.get_sample_annotation("focal", sample), self.get_sample_annotation("other", sample), parked]

    def get_map_name_from_sample_token(self, sample):
        return "boston-seaport"


class FakeNusc:

    def get(self, table, tok):
        return {"name": {"moving": "vehicle.moving", "parked": "vehicle.parked"}[tok]}


class FakeMap:

    def get_records_in_radius(self, x, y, r, layers):
        return {"lane": ["l1"], "lane_connector": [], "drivable_area": ["d1"]}

    def get(self, layer, tok):
        return {"polygon_tokens": ["poly1"]}

    def extract_polygon(self, tok):
        from shapely.geometry import Polygon
        left = np.array([-DIR[1], DIR[0]])
        pts = [O - DIR * 50 - left * 6, O + DIR * 150 - left * 6, O + DIR * 150 + left * 6, O - DIR * 50 + left * 6]
        return Polygon(pts)

    def discretize_lanes(self, ids, res):
        return {"l1": [(*(O + DIR * s), YAW) for s in np.arange(-10.0, 60.0, 1.0)]}


def test_convert_split(tmp_path, monkeypatch):
    monkeypatch.setattr(splits, "get_prediction_challenge_split", lambda split, dataroot: ["focal_s0"])
    args = SimpleNamespace(root="x", limit=0, n_nbr=4, n_lanes=8, n_points=10, lane_len=20.0, radius=60.0,
                           n_obs=4, sdf_radius=140.0, sdf_layers=["drivable_area"])
    cn.convert_split(FakeNusc(), FakeHelper(), {"boston-seaport": FakeMap()}, "train", args, tmp_path)
    a = load_split(tmp_path, "train")
    assert a["hist_mask"].all() and a["fut_mask"].all()
    np.testing.assert_allclose(a["hist"][0, :, 0], [-20, -15, -10, -5, 0], atol=1e-3)
    np.testing.assert_allclose(a["hist"][0, :, 1], 0.0, atol=1e-3)
    np.testing.assert_allclose(a["fut"][0, :, 0], 5.0 * np.arange(1, cn.T + 1), atol=1e-3)
    np.testing.assert_allclose(a["fut"][0, :, 1], 0.0, atol=1e-3)
    assert a["nbr_mask"][0, 0].tolist() == [False, False, True, True, True]
    np.testing.assert_allclose(a["nbr"][0, 0, 2:], [[-6, 3], [-3, 3], [0, 3]], atol=1e-3)
    assert np.abs(a["lane"][0][a["lane_mask"][0]][:, 1]).max() < 1e-3
    np.testing.assert_allclose(a["focal_size"][0], [4.5, 1.9])
    assert a["obs_mask"][0].tolist() == [True, False, False, False]
    np.testing.assert_allclose(a["obs"][0, 0], [20.0, -3.0, 0.0, 5.0, 2.0], atol=1e-3)
    sdf = a["sdf"][0].astype(np.float32) * 0.25
    g = cn.SDF_GRID
    row = lambda y: int((y - g["y_min"]) / g["res"])
    col = lambda x: int((x - g["x_min"]) / g["res"])
    assert sdf[row(0.5), col(10.5)] < -4.0 and sdf[row(10.5), col(10.5)] > 3.0
    assert a["nbr_mask"][0, 0].any()
    np.testing.assert_allclose(a["nbr_size"][0, 0], [4.5, 1.9])
    np.testing.assert_allclose(a["nbr_yaw"][0, 0], 0.0, atol=1e-5)


def test_convert_split_parallel_matches_serial(tmp_path, monkeypatch):
    monkeypatch.setattr(splits, "get_prediction_challenge_split", lambda split, dataroot: ["focal_s0"] * 40)
    kw = dict(root="x", limit=0, n_nbr=4, n_lanes=8, n_points=10, lane_len=20.0, radius=60.0,
              n_obs=4, sdf_radius=140.0, sdf_layers=["drivable_area"], log_every=10)
    out = {}
    for w in (1, 2):
        d = tmp_path / f"w{w}"
        cn.convert_split(FakeNusc(), FakeHelper(), {"boston-seaport": FakeMap()}, "train",
                         SimpleNamespace(**kw, workers=w), d)
        out[w] = load_split(d, "train")
    for k in out[1]:
        np.testing.assert_array_equal(out[1][k], out[2][k])
