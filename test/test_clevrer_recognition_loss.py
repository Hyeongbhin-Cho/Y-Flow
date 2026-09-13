import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from data.clevrer_state import CLEVRERStateLayout, pack_state
from train.clevrer_recognition import CLEVRERRecognitionCriterion, run_train_recognition


class TestCLEVRERRecognitionCriterion(unittest.TestCase):
    def test_clip_matching_and_multitask_loss_backpropagate(self):
        layout = CLEVRERStateLayout(n_slots=3, n_frames=4)
        color = torch.zeros(1, 3, 8)
        material = torch.zeros(1, 3, 2)
        shape = torch.zeros(1, 3, 3)
        color[0, 0, 1] = color[0, 2, 4] = 1
        material[0, 0, 0] = material[0, 2, 1] = 1
        shape[0, 0, 2] = shape[0, 2, 0] = 1
        vis = torch.zeros(1, 3, 4)
        vis[:, (0, 2), :] = 1
        pos = torch.zeros(1, 3, 4, 3)
        pos[0, 0, :, 0] = torch.arange(4) * 0.1
        pos[0, 2, :, 1] = torch.arange(4) * 0.1
        vel = torch.zeros_like(pos)
        coll = torch.zeros(1, 4, 3, 3)
        coll[0, 2, 0, 2] = coll[0, 2, 2, 0] = 1
        target = pack_state(color, material, shape, vis, pos, vel, coll, layout)

        cfg = OmegaConf.create(
            {
                "data": {"n_slots": 3, "n_frames": 4, "fps": 16},
                "loss": {},
            }
        )
        criterion = CLEVRERRecognitionCriterion(
            cfg,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            5.0,
        )
        outputs = {
            "objectness_logits": torch.randn(1, 3, requires_grad=True),
            "color_logits": torch.randn(1, 3, 8, requires_grad=True),
            "material_logits": torch.randn(1, 3, 2, requires_grad=True),
            "shape_logits": torch.randn(1, 3, 3, requires_grad=True),
            "visibility_logits": torch.randn(1, 3, 4, requires_grad=True),
            "positions": torch.randn(1, 3, 4, 3, requires_grad=True),
            "velocities": torch.randn(1, 3, 4, 3, requires_grad=True),
            "collision_logits": torch.randn(1, 4, 3, requires_grad=True),
        }
        loss, components, matches = criterion(outputs, target)
        self.assertEqual(len(matches[0]), 2)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(components), {"objectness", "attributes", "visibility", "position", "velocity", "collision", "kinematic"})
        loss.backward()
        for key in ("objectness_logits", "color_logits", "visibility_logits", "positions", "velocities", "collision_logits"):
            self.assertIsNotNone(outputs[key].grad)
            self.assertTrue(torch.isfinite(outputs[key].grad).all())

    def test_supervised_train_loop_writes_dev_selected_checkpoints(self):
        layout = CLEVRERStateLayout(n_slots=2, n_frames=2)
        color = torch.zeros(2, 8)
        material = torch.zeros(2, 2)
        shape = torch.zeros(2, 3)
        color[1, 0] = material[1, 1] = shape[1, 2] = 1
        vis = torch.zeros(2, 2)
        vis[1] = 1
        pos = torch.zeros(2, 2, 3)
        pos[1, :, 0] = torch.tensor([0.0, 0.1])
        vel = torch.zeros_like(pos)
        coll = torch.zeros(2, 2, 2)
        packed = pack_state(color, material, shape, vis, pos, vel, coll, layout)

        class TinyVideoDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, index):
                return {"video": torch.full((3, 2, 32, 32), float(index) / 4), "state": packed.clone()}

        class TinyRecognizer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.objectness = torch.nn.Parameter(torch.zeros(2))
                self.color = torch.nn.Parameter(torch.zeros(2, 8))
                self.material = torch.nn.Parameter(torch.zeros(2, 2))
                self.shape = torch.nn.Parameter(torch.zeros(2, 3))
                self.visibility = torch.nn.Parameter(torch.zeros(2, 2))
                self.position = torch.nn.Parameter(torch.zeros(2, 2, 3))
                self.velocity = torch.nn.Parameter(torch.zeros(2, 2, 3))
                self.collision = torch.nn.Parameter(torch.zeros(2, 1))

            def forward(self, video):
                scalar = video.mean(dim=(1, 2, 3, 4))
                return {
                    "objectness_logits": self.objectness[None] + scalar[:, None],
                    "color_logits": self.color[None] + scalar[:, None, None],
                    "material_logits": self.material[None] + scalar[:, None, None],
                    "shape_logits": self.shape[None] + scalar[:, None, None],
                    "visibility_logits": self.visibility[None] + scalar[:, None, None],
                    "positions": self.position[None] + scalar[:, None, None, None],
                    "velocities": self.velocity[None] + scalar[:, None, None, None],
                    "collision_logits": self.collision[None] + scalar[:, None, None],
                }

        def tiny_collate(items):
            return {key: torch.stack([item[key] for item in items]) for key in items[0]}

        cfg = OmegaConf.create(
            {
                "seed": 7,
                "data": {"name": "clevrer_recognition", "n_slots": 2, "n_frames": 2, "fps": 16},
                "model": {"name": "clevrer_resnet34", "local_dir": None, "backbone_mode": "frozen"},
                "train": {
                    "batch_size": 2,
                    "epochs": 1,
                    "dev_fraction": 0.25,
                    "early_stopping_patience": 1,
                    "lr": 1.0e-3,
                    "num_workers": 0,
                },
                "loss": {},
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "recognition"
            bundle = SimpleNamespace(
                train=TinyVideoDataset(),
                train_raw=torch.stack([packed] * 4).numpy(),
            )
            with (
                patch("train.clevrer_recognition.build_dataset", return_value=bundle),
                patch("train.clevrer_recognition.build_model", return_value=TinyRecognizer()),
                patch("train.clevrer_recognition.collate_clevrer_recognition", side_effect=tiny_collate),
                patch("train.clevrer_recognition.method_dir", return_value=output_dir),
            ):
                best = run_train_recognition(cfg, device=torch.device("cpu"))
            self.assertTrue(best.is_file())
            self.assertTrue((output_dir / "last.pt").is_file())
            checkpoint = torch.load(best, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["extra"]["task"], "clevrer_video_recognition")
            self.assertEqual(set(checkpoint["extra"]["thresholds"]), {"objectness", "visibility", "collision"})


if __name__ == "__main__":
    unittest.main()
