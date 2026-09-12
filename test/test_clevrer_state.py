# -*- coding: utf-8 -*-
# test/test_clevrer_state.py

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from data.clevrer_state import (
    CLEVRERStateConstraint,
    CLEVRERStateLayout,
    CLEVRERStateMeta,
    annotation_to_state,
    pack_parts,
    unpack_state,
)
try:
    import av
except ImportError:
    av = None


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


def _valid_annotation(n_frames: int = 8) -> dict:
    objects = [
        {"object_id": 0, "color": "blue", "material": "rubber", "shape": "cylinder"},
        {"object_id": 1, "color": "brown", "material": "metal", "shape": "cube"},
        {"object_id": 2, "color": "gray", "material": "metal", "shape": "sphere"},
    ]
    traj = []
    for i in range(n_frames):
        t = i / 16.0
        if i < 2:
            p0, p1, v0, v1 = 0.4 * t, 0.45 + 0.4 * t, 0.4, 0.4
        else:
            t2 = 2 / 16.0
            p0 = 0.4 * t2 + (-0.2) * (t - t2)
            p1 = 0.45 + 0.4 * t2 + 0.8 * (t - t2)
            v0, v1 = -0.2, 0.8
        traj.append({
            "frame_id": i,
            "objects": [
                {
                    "object_id": 0,
                    "location": [p0, 0.0, 0.20],
                    "velocity": [v0, 0.0, 0.0],
                    "inside_camera_view": True,
                },
                {
                    "object_id": 1,
                    "location": [p1, 0.0, 0.20],
                    "velocity": [v1, 0.0, 0.0],
                    "inside_camera_view": True,
                },
                {
                    "object_id": 2,
                    "location": [3.0, 3.0, 0.20],
                    "velocity": [0.0, 0.0, 0.0],
                    "inside_camera_view": True,
                },
            ],
        })
    return {
        "scene_index": 0,
        "object_property": objects,
        "motion_trajectory": traj,
        "collision": [{"object_ids": [0, 1], "frame_id": 2, "location": [0.2, 0.0, 0.2]}],
    }


class TestCLEVRERState(unittest.TestCase):
    def test_pack_unpack_roundtrip(self) -> None:
        layout = CLEVRERStateLayout(n_slots=6, n_frames=4)
        color = np.zeros((6, 8), np.float32)
        color[0, 0] = 1.0
        material = np.zeros((6, 2), np.float32)
        material[0, 1] = 1.0
        shape = np.zeros((6, 3), np.float32)
        shape[0, 2] = 1.0
        vis = np.zeros((6, 4), np.float32)
        vis[0] = 1.0
        pos = np.zeros((6, 4, 3), np.float32)
        pos[0, :, 2] = 0.2
        vel = np.zeros((6, 4, 3), np.float32)
        coll = np.zeros((4, 6, 6), np.float32)
        packed = pack_parts(color, material, shape, vis, pos, vel, coll, layout)
        self.assertEqual(packed.shape, (layout.dim,))
        parts = unpack_state(packed, layout)
        np.testing.assert_allclose(parts["pos"], pos)
        np.testing.assert_allclose(parts["vis"], vis)

    def test_annotation_to_state_and_gt_is_safe(self) -> None:
        layout = CLEVRERStateLayout(n_slots=6, n_frames=8)
        annotation = _valid_annotation(8)
        state = annotation_to_state(annotation, np.arange(8), layout, dt=1.0 / 16.0)
        parts = unpack_state(state, layout)
        self.assertEqual(int((parts["vis"].max(axis=1) > 0.5).sum()), 3)
        self.assertGreater(float(parts["coll"][2, 0, 1]), 0.5)
        cons = CLEVRERStateConstraint(_meta(layout), layout)
        h = cons.h(state)
        for name, value in h.items():
            self.assertLessEqual(float(np.max(value)), 1e-5, msg=name)

    def test_cost_autograd(self) -> None:
        layout = CLEVRERStateLayout(n_slots=6, n_frames=4)
        cons = CLEVRERStateConstraint(_meta(layout), layout)
        p = torch.randn(5, layout.dim, requires_grad=True)
        cost = cons.cost(p)
        self.assertEqual(tuple(cost.shape), (5,))
        grad = torch.autograd.grad(cost.sum(), p)[0]
        self.assertEqual(tuple(grad.shape), (5, layout.dim))
        self.assertTrue(torch.isfinite(grad).all())

    def test_count_violation(self) -> None:
        layout = CLEVRERStateLayout(n_slots=6, n_frames=4)
        annotation = _valid_annotation(4)
        annotation["object_property"] = annotation["object_property"][:1]
        for entry in annotation["motion_trajectory"]:
            entry["objects"] = [o for o in entry["objects"] if o["object_id"] == 0]
        annotation["collision"] = []
        state = annotation_to_state(annotation, np.arange(4), layout)
        cons = CLEVRERStateConstraint(_meta(layout), layout)
        self.assertGreater(float(cons.h(state)["count"]), 0.0)


class TestCLEVRERRecognitionBuild(unittest.TestCase):
    @unittest.skipIf(av is None, "PyAV required")
    def test_dataset_item_shapes(self) -> None:
        from test.test_clevrer import make_scene
        from data.clevrer import CLEVRERDataset
        from data.clevrer_state import CLEVRERRecognitionDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_scene(root, 10000, "validation")
            path = root / "annotations/validation/annotation_10000.json"
            payload = json.loads(path.read_text())
            payload["object_property"] = _valid_annotation()["object_property"]
            payload["motion_trajectory"] = [
                {
                    "frame_id": i,
                    "objects": [
                        {
                            "object_id": o["object_id"],
                            "location": [0.2 * o["object_id"], 0.0, 0.20],
                            "velocity": [0.0, 0.0, 0.0],
                            "inside_camera_view": True,
                        }
                        for o in payload["object_property"]
                    ],
                }
                for i in range(12)
            ]
            payload["collision"] = []
            path.write_text(json.dumps(payload))
            videos = CLEVRERDataset(root, "validation", n_frames=4, fps=4)
            layout = CLEVRERStateLayout(n_slots=6, n_frames=4)
            mean = np.zeros(layout.dim, np.float32)
            std = np.ones(layout.dim, np.float32)
            dataset = CLEVRERRecognitionDataset(videos, layout, mean, std, dt=0.25)
            item = dataset[0]
            self.assertEqual(tuple(item["state"].shape), (layout.dim,))
            self.assertEqual(item["video"].shape[1], 4)


if __name__ == "__main__":
    unittest.main()
