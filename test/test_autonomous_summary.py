from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.summarize_autonomous import summarize


class TestAutonomousSummary(unittest.TestCase):
    def test_aggregate_mean_and_sample_std(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed, ade in ((0, 3.0), (1, 5.0)):
                directory = root / "runs" / f"seed_{seed}" / "moflow"
                directory.mkdir(parents=True)
                payload = {
                    "method": "moflow",
                    "n_scenarios": 10,
                    "n_modes": 6,
                    "n_steps": 100,
                    "minADE": ade,
                    "minFDE": 9.0,
                }
                (directory / "metrics.json").write_text(json.dumps(payload))
            with patch("eval.summarize_autonomous.ROOT", root):
                result = summarize(["seed_0", "seed_1"])
            self.assertEqual(result["methods"]["moflow"]["minADE"]["mean"], 4.0)
            self.assertAlmostEqual(
                result["methods"]["moflow"]["minADE"]["std"], 2**0.5
            )


if __name__ == "__main__":
    unittest.main()
