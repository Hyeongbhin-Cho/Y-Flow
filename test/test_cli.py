# -*- coding: utf-8 -*-
# test/test_cli.py

from __future__ import annotations

import unittest

from main import parse_args
from utils.paths import method_dir


class TestCLI(unittest.TestCase):
    def test_overrides_and_run_dir(self) -> None:
        args, cfg = parse_args(
            [
                "flowmatch",
                "--mode",
                "eval",
                "--run_name",
                "cli_unit",
                "--steps",
                "11",
                "--device",
                "cpu",
            ]
        )
        self.assertEqual(args.command, "flowmatch")
        self.assertEqual(args.mode, "eval")
        self.assertEqual(str(cfg.run_name), "cli_unit")
        self.assertEqual(int(cfg.train.steps), 11)
        self.assertEqual(str(cfg.device), "cpu")
        self.assertTrue(str(method_dir(cfg, "flowmatch")).endswith("runs/cli_unit/flowmatch"))

    def test_unimplemented_eval(self) -> None:
        from main import main

        with self.assertRaises(NotImplementedError):
            main(["yflow", "--mode", "eval", "--run_name", "cli_unit"])


if __name__ == "__main__":
    unittest.main()
