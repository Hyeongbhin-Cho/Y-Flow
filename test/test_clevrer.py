# -*- coding: utf-8 -*-
# test/test_clevrer.py

from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data import CLEVRERDataset, VideoDataBundle, build_dataset, collate_clevrer
from scripts.setup_clevrer import download_file, extract_archive, write_manifest

try:
    import av
except ImportError:
    av = None


def make_scene(root: Path, scene_id: int = 10000, split: str = "validation") -> None:
    video = root / "videos" / split / f"video_{scene_id:05d}.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 24, 16, "yuv420p"
        for index in range(12):
            pixels = np.full((16, 24, 3), index * 20, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    if split != "test":
        annotation = root / "annotations" / split / f"annotation_{scene_id:05d}.json"
        annotation.parent.mkdir(parents=True, exist_ok=True)
        annotation.write_text(json.dumps({
            "scene_index": scene_id, "video_filename": video.name,
            "object_property": [{"object_id": 0, "color": "blue", "material": "rubber", "shape": "sphere"}],
            "motion_trajectory": [{"frame_id": i, "objects": [{"object_id": 0, "location": [i, 0, 0]}]} for i in range(12)],
            "collision": [{"object_ids": [0, 1], "frame_id": 3, "location": [1, 0, 0]}],
        }))


@unittest.skipIf(av is None, "PyAV is required for real MP4 decoding tests")
class TestCLEVRERDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        make_scene(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_native_video_and_raw_annotations(self) -> None:
        dataset = CLEVRERDataset(self.root)
        item = dataset[0]
        self.assertEqual(item["video"].shape, (3, 12, 16, 24))
        self.assertEqual(item["video"].dtype, torch.float32)
        self.assertTrue(bool((item["video"].abs() <= 1).all()))
        self.assertTrue(torch.equal(item["initial_frame"], item["video"][:, 0]))
        self.assertEqual(item["annotation"]["collision"][0]["frame_id"], 3)
        self.assertEqual(item["frame_annotations"][5]["objects"][0]["location"], [5, 0, 0])

    def test_clip_timestamps_letterbox_and_alignment(self) -> None:
        item = CLEVRERDataset(self.root, n_frames=5, fps=5, height=32, width=64)[0]
        self.assertEqual(item["video"].shape, (3, 5, 32, 64))
        self.assertEqual(item["frame_indices"].tolist(), [0, 2, 4, 6, 8])
        torch.testing.assert_close(item["frame_times"], torch.tensor([0, .2, .4, .6, .8], dtype=torch.float64))
        self.assertEqual([entry["frame_id"] for entry in item["frame_annotations"]], [0, 2, 4, 6, 8])
        self.assertEqual(item["spatial_transform"]["pad_left"], 8)
        self.assertEqual(int(item["valid_region"].sum()), 32 * 48)
        self.assertTrue(bool((item["video"][..., :8] == -1).all()))
        self.assertGreater(float(item["video"][:, -1, :, 8:56].mean()), float(item["video"][:, 0, :, 8:56].mean()))
        # Events between sampled frames are retained in the raw annotation.
        self.assertEqual(item["annotation"]["collision"][0]["frame_id"], 3)

    def test_registry_lazy_bundle_and_dataloader(self) -> None:
        make_scene(self.root, 10001)
        cfg = OmegaConf.create({"data": {"name": "clevrer", "cache_dir": str(self.root), "n_frames": 5, "fps": 5}})
        with patch("data.clevrer._decode_video", side_effect=AssertionError("eager decode")):
            bundle = build_dataset(cfg)
        self.assertIsInstance(bundle, VideoDataBundle)
        self.assertIsNone(bundle.constraint)
        self.assertIsNone(bundle.train)
        self.assertNotIn("eval_z", bundle)
        self.assertEqual(bundle["meta"]["n_eval"], 2)
        batch = next(iter(DataLoader(bundle.eval, batch_size=2, collate_fn=collate_clevrer)))
        self.assertEqual(batch["video"].shape, (2, 3, 5, 16, 24))
        self.assertEqual(batch["scene_index"], [10000, 10001])
        self.assertEqual(len(batch["annotation"]), 2)

    def test_test_split_has_no_ground_truth(self) -> None:
        make_scene(self.root, 15000, "test")
        sample = CLEVRERDataset(self.root, "test")[0]
        self.assertIsNone(sample["annotation"])
        self.assertIsNone(sample["frame_annotations"])
        self.assertIsNone(collate_clevrer([sample])["annotation"][0])

    def test_optional_train_and_questions(self) -> None:
        make_scene(self.root, 0, "train")
        directory = self.root / "questions"
        directory.mkdir()
        (directory / "validation.json").write_text(json.dumps([{"scene_index": 10000, "questions": [{"question": "Example?"}]}]))
        self.assertEqual(CLEVRERDataset(self.root, load_questions=True)[0]["questions"][0]["question"], "Example?")
        cfg = OmegaConf.create({"data": {"name": "clevrer", "cache_dir": str(self.root), "train_split": "train"}})
        self.assertEqual(len(build_dataset("clevrer", cfg=cfg).train), 1)

    def test_short_clip_and_missing_or_mismatched_annotations_fail(self) -> None:
        with self.assertRaises(ValueError):
            CLEVRERDataset(self.root, n_frames=33, fps=16)[0]
        path = self.root / "annotations/validation/annotation_10000.json"
        payload = json.loads(path.read_text())
        payload["scene_index"] = 10001
        path.write_text(json.dumps(payload))
        with self.assertRaises(ValueError):
            CLEVRERDataset(self.root)[0]
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            CLEVRERDataset(self.root)


class TestCLEVRERSetup(unittest.TestCase):
    def test_archive_layout_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for kind, filename, payload in [
                ("videos", "video_10000.mp4", b"fixture"),
                ("annotations", "annotation_10000.json", b'{"scene_index":10000}'),
            ]:
                archive = root / f"{kind}.zip"
                with zipfile.ZipFile(archive, "w") as output:
                    output.writestr("nested/10000-11000/" + filename, payload)
                extract_archive(archive, root / kind / "validation", kind, "validation")
            manifest = json.loads(write_manifest(root, "validation", expected_count=1).read_text())
            self.assertEqual(manifest["records"][0]["video"], "videos/validation/video_10000.mp4")
            with self.assertRaises(ValueError):
                write_manifest(root, "validation", expected_count=2)

    def test_archive_rejects_traversal_and_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for names in [["../video_10000.mp4"], ["a/video_10000.mp4", "b/video_10000.mp4"]]:
                archive = root / "bad.zip"
                with zipfile.ZipFile(archive, "w") as output:
                    for name in names:
                        output.writestr(name, b"data")
                with self.assertRaises(ValueError):
                    extract_archive(archive, root / "out", "videos", "validation")
                self.assertFalse((root / "out").exists())

    def test_download_resume_and_server_ignoring_range(self) -> None:
        for status, body, headers in [
            (206, b"def", {"Content-Length": "3", "Content-Range": "bytes 3-5/6"}),
            (200, b"abcdef", {"Content-Length": "6"}),
        ]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "file.zip"
                path.with_suffix(".zip.part").write_bytes(b"abc")
                response = io.BytesIO(body)
                response.status, response.headers = status, headers
                with patch("urllib.request.urlopen", return_value=response) as request:
                    download_file("https://example.org/file.zip", path)
                self.assertEqual(request.call_args.args[0].get_header("Range"), "bytes=3-")
                self.assertEqual(path.read_bytes(), b"abcdef")
                self.assertFalse(path.with_suffix(".zip.part").exists())
                with patch("urllib.request.urlopen", side_effect=AssertionError("downloaded twice")):
                    download_file("https://example.org/file.zip", path)


if __name__ == "__main__":
    unittest.main()
