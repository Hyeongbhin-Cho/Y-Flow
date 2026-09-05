# -*- coding: utf-8 -*-
# test/test_sample_result.py

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

import torch

from eval.sample_result import SampleResult


class TestSampleResult(unittest.TestCase):
    def test_default_diagnostics(self) -> None:
        samples = torch.randn(10, 2)
        res = SampleResult(samples=samples)
        self.assertIs(res.samples, samples)
        self.assertEqual(res.diagnostics, {})

    def test_explicit_diagnostics(self) -> None:
        samples = torch.randn(5, 2)
        payload = {"nfe": 10, "pre_filter_safe_ratio": 0.85}
        res = SampleResult(samples=samples, diagnostics=payload)
        self.assertIs(res.samples, samples)
        self.assertEqual(res.diagnostics["nfe"], 10)
        self.assertEqual(res.diagnostics["pre_filter_safe_ratio"], 0.85)

    def test_frozen_immutability(self) -> None:
        samples = torch.randn(4, 2)
        res = SampleResult(samples=samples)
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            res.samples = torch.zeros(4, 2)  # type: ignore

    def test_with_diagnostics(self) -> None:
        samples = torch.randn(4, 2)
        res1 = SampleResult(samples=samples, diagnostics={"step": 1})
        res2 = res1.with_diagnostics(step=2, metric="ok")

        self.assertEqual(res1.diagnostics, {"step": 1})
        self.assertEqual(res2.diagnostics, {"step": 2, "metric": "ok"})
        self.assertIs(res2.samples, samples)

    def test_tuple_unpacking(self) -> None:
        samples = torch.randn(6, 2)
        payload = {"key": "val"}
        res = SampleResult(samples=samples, diagnostics=payload)

        unpacked_samples, unpacked_diag = res
        self.assertIs(unpacked_samples, samples)
        self.assertEqual(unpacked_diag, payload)


if __name__ == "__main__":
    unittest.main()
