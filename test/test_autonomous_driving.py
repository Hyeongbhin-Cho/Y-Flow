# -*- coding: utf-8 -*-
# test/test_autonomous_driving.py

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from data.autonomous_driving import AutonomousDrivingConstraint, build_autonomous_driving


def _write_scenario(root: Path, split: str, scenario_id: str, speed: float) -> None:
    directory = root / split / scenario_id
    directory.mkdir(parents=True)
    timestep = np.arange(110)
    observed = timestep < 50
    frame = pd.DataFrame(
        {
            "observed": observed,
            "track_id": "focal",
            "object_type": "vehicle",
            "object_category": 3,
            "timestep": timestep,
            "position_x": speed * 0.1 * timestep,
            "position_y": np.zeros(110),
            "heading": np.zeros(110),
            "velocity_x": np.full(110, speed),
            "velocity_y": np.zeros(110),
            "scenario_id": scenario_id,
            "start_timestamp": 0,
            "end_timestamp": 109,
            "num_timestamps": 110,
            "focal_track_id": "focal",
            "city": "TEST",
        }
    )
    frame.to_parquet(directory / f"scenario_{scenario_id}.parquet")


def _cfg(raw_dir: Path, cache_dir: Path) -> OmegaConf:
    return OmegaConf.create(
        {
            "seed": 0,
            "data": {
                "name": "autonomous_driving",
                "raw_dir": str(raw_dir),
                "cache_dir": str(cache_dir),
                "regenerate": True,
                "sample_hz": 10.0,
                "history_steps": 50,
                "future_steps": 60,
                "difference_window": 5,
                "n_train": 2,
                "n_eval": 1,
                "v_max": 25.0,
                "a_max": 15.0,
                "context": {
                    "radius_m": 50.0,
                    "max_neighbors": 16,
                    "actor_types": [
                        "vehicle",
                        "bus",
                        "motorcyclist",
                        "cyclist",
                        "pedestrian",
                    ],
                },
            },
        }
    )


class TestAutonomousDrivingData(unittest.TestCase):
    def test_build_cache_and_coordinate_transform(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            cache = root / "cache"
            _write_scenario(raw, "train", "train-a", 10.0)
            _write_scenario(raw, "train", "train-b", 5.0)
            _write_scenario(raw, "val", "val-a", 8.0)
            bundle = build_autonomous_driving(_cfg(raw, cache))

            self.assertEqual(bundle.train_raw.shape, (2, 120))
            self.assertEqual(bundle.eval_raw.shape, (1, 120))
            self.assertEqual(bundle.train_context["focal_history"].shape, (2, 50, 2))
            self.assertEqual(
                bundle.train_context["neighbor_history"].shape, (2, 16, 50, 2)
            )
            self.assertEqual(bundle.train_context["neighbor_mask"].shape, (2, 16))
            self.assertFalse(bundle.train_context["neighbor_mask"].any())
            self.assertAlmostEqual(float(bundle.train_raw[0, 0]), 1.0, places=5)
            self.assertAlmostEqual(float(bundle.train_raw[0, 1]), 0.0, places=5)
            self.assertTrue((cache / "train.npy").is_file())
            self.assertTrue((cache / "eval.npy").is_file())
            self.assertTrue((cache / "meta.json").is_file())
            self.assertTrue((cache / "train_context.npz").is_file())
            self.assertTrue((cache / "eval_context.npz").is_file())

    def test_constraints_autograd_and_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            _write_scenario(raw, "train", "train-a", 10.0)
            _write_scenario(raw, "train", "train-b", 5.0)
            _write_scenario(raw, "val", "val-a", 8.0)
            bundle = build_autonomous_driving(_cfg(raw, root / "cache"))
            constraint = AutonomousDrivingConstraint(bundle.meta)

            safe = torch.from_numpy(bundle.train_raw[:1]).requires_grad_(True)
            values = constraint.h(safe)
            self.assertLessEqual(float(values["speed"].item()), 0.0)
            self.assertLessEqual(float(values["accel"].item()), 0.0)

            unsafe = safe.detach() * 4.0
            unsafe.requires_grad_(True)
            cost = constraint.cost(unsafe)
            grad = torch.autograd.grad(cost.sum(), unsafe)[0]
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(cost.item()), 0.0)

            projected = constraint.project_feasible(unsafe)
            projected_h = constraint.h(projected)
            self.assertLessEqual(float(projected_h["speed"].item()), 1e-5)
            self.assertLessEqual(float(projected_h["accel"].item()), 1e-5)


if __name__ == "__main__":
    unittest.main()
