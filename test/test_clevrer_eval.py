# -*- coding: utf-8 -*-
# test/test_clevrer_eval.py

from __future__ import annotations

import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from model.clevrer_eval import (
    annotation_attribute_set,
    annotation_collisions_by_attr,
    evaluate_propnet_pred,
    evaluate_visual_mask,
    match_collisions,
    proposal_attribute_set,
    propnet_collisions,
    set_scores,
)
from model.download import (
    CLEVRER_ARTIFACTS,
    extract_clevrer_archive,
    is_clevrer_artifact_ready,
    write_clevrer_artifact_manifest,
)


def _proposal(scene_id: int = 10000) -> dict:
    return {
        "frames": [
            {
                "frame_index": 0,
                "objects": [
                    {"color": "blue", "material": "rubber", "shape": "cylinder"},
                    {"color": "gray", "material": "metal", "shape": "sphere"},
                ],
            }
        ],
        "ground_truth": {
            "objects": [
                {"id": 0, "color": "brown", "material": "metal", "shape": "cube"},
                {"id": 1, "color": "blue", "material": "rubber", "shape": "cylinder"},
                {"id": 2, "color": "yellow", "material": "metal", "shape": "cube"},
                {"id": 3, "color": "gray", "material": "metal", "shape": "sphere"},
            ],
            "collisions": [{"object": [0, 1], "frame": 34}],
        },
    }


def _annotation() -> dict:
    return {
        "scene_index": 10000,
        "object_property": [
            {"object_id": 0, "color": "brown", "material": "metal", "shape": "cube"},
            {"object_id": 1, "color": "blue", "material": "rubber", "shape": "cylinder"},
            {"object_id": 2, "color": "yellow", "material": "metal", "shape": "cube"},
            {"object_id": 3, "color": "gray", "material": "metal", "shape": "sphere"},
        ],
        "motion_trajectory": [
            {
                "frame_id": i,
                "objects": [
                    {"object_id": 0, "inside_camera_view": i > 10},
                    {"object_id": 1, "inside_camera_view": True},
                    {"object_id": 2, "inside_camera_view": False},
                    {"object_id": 3, "inside_camera_view": True},
                ],
            }
            for i in range(5)
        ],
        "collision": [{"object_ids": [0, 1], "frame_id": 34, "location": [0, 0, 0]}],
    }


def _propnet_pred() -> dict:
    return {
        "objects": [
            {"id": 0, "color": "brown", "material": "metal", "shape": "cube"},
            {"id": 1, "color": "blue", "material": "rubber", "shape": "cylinder"},
        ],
        "predictions": [
            {
                "what_if": -1,
                "trajectory": [],
                "collisions": [
                    {
                        "frame": 36,
                        "objects": [
                            {"color": "brown", "material": "metal", "shape": "cube"},
                            {"color": "blue", "material": "rubber", "shape": "cylinder"},
                        ],
                    }
                ],
            }
        ],
    }


class TestCLEVRERArtifacts(unittest.TestCase):
    def test_catalog_covers_requested_models(self) -> None:
        self.assertEqual(set(CLEVRER_ARTIFACTS), {"visual_masks", "propnet_preds", "mask_rcnn", "propnet"})
        self.assertEqual(CLEVRER_ARTIFACTS["visual_masks"]["weight_status"], "not_a_checkpoint")
        self.assertEqual(CLEVRER_ARTIFACTS["mask_rcnn"]["weight_status"], "public_coco_not_clevrer_finetuned")
        self.assertEqual(CLEVRER_ARTIFACTS["propnet"]["applies_to"], "proposal_tubes_not_raw_video")

    def test_extract_zip_flattens_and_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "masks.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("nested/processed_proposals/sim_10000.json", b"{}")
            n = extract_clevrer_archive(archive, root / "visual_masks", r"(proposal|sim)_\d{5}\.json")
            self.assertEqual(n, 1)
            self.assertTrue((root / "visual_masks" / "sim_10000.json").is_file())
            bad = root / "bad.zip"
            with zipfile.ZipFile(bad, "w") as output:
                output.writestr("../sim_10000.json", b"{}")
            with self.assertRaises(ValueError):
                extract_clevrer_archive(bad, root / "out", r"sim_\d{5}\.json")

    def test_extract_tar_keeps_supervision_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "preds.tar.gz"
            inner = root / "sim_10000.json"
            inner.write_text("{}")
            with tarfile.open(archive, "w:gz") as output:
                output.add(inner, arcname="propnet_preds/with_edge_supervision/sim_10000.json")
            n = extract_clevrer_archive(archive, root / "propnet_preds", r"sim_\d{5}\.json", keep_parents=1)
            self.assertEqual(n, 1)
            self.assertTrue((root / "propnet_preds" / "with_edge_supervision" / "sim_10000.json").is_file())

    def test_ready_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "visual_masks").mkdir()
            for scene_id in range(10000, 10100):
                (root / "visual_masks" / f"sim_{scene_id:05d}.json").write_text("{}")
            self.assertTrue(is_clevrer_artifact_ready(root, "visual_masks"))
            self.assertFalse(is_clevrer_artifact_ready(root, "mask_rcnn"))
            manifest = json.loads(write_clevrer_artifact_manifest(root).read_text())
            self.assertTrue(manifest["artifacts"]["visual_masks"]["ready"])
            self.assertFalse(manifest["artifacts"]["propnet"]["ready"])


class TestCLEVREREvalMetrics(unittest.TestCase):
    def test_attribute_and_collision_matching(self) -> None:
        proposal = _proposal()
        annotation = _annotation()
        self.assertEqual(proposal_attribute_set(proposal), annotation_attribute_set(annotation))
        scores = set_scores(proposal_attribute_set(proposal), annotation_attribute_set(annotation))
        self.assertEqual(scores["f1"], 1.0)
        collisions = match_collisions(
            [(36, frozenset({("brown", "metal", "cube"), ("blue", "rubber", "cylinder")}))],
            annotation_collisions_by_attr(annotation),
        )
        self.assertEqual(collisions["tp"], 1)
        self.assertGreater(collisions["f1"], 0.99)

    def test_evaluate_visual_mask_and_propnet_from_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "visual_masks").mkdir()
            (root / "propnet_preds" / "with_edge_supervision").mkdir(parents=True)
            (root / "visual_masks" / "sim_10000.json").write_text(json.dumps(_proposal()))
            (root / "propnet_preds" / "with_edge_supervision" / "sim_10000.json").write_text(json.dumps(_propnet_pred()))
            annotation = _annotation()
            mask = evaluate_visual_mask(10000, annotation, root)
            self.assertEqual(mask["attributes"]["f1"], 1.0)
            pred = evaluate_propnet_pred(10000, annotation, root)
            self.assertGreater(pred["collisions"]["f1"], 0.99)
            self.assertEqual(len(propnet_collisions(_propnet_pred())), 1)


if __name__ == "__main__":
    unittest.main()
