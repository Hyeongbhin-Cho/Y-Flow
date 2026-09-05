# -*- coding: utf-8 -*-
# eval/sample_result.py

from __future__ import annotations

"""Standard return contract between samplers and evaluators."""

from dataclasses import dataclass, field
from typing import Any, Iterator

import torch


@dataclass(frozen=True)
class SampleResult:
    """Unified return container for sampling outputs and runtime telemetry.

    Acts as the standard data protocol between method-specific samplers
    (`eval/*.py`) and the central evaluation runner (`eval/evaluate.py`).

    Attributes:
        samples: Generated sample points in normalized space, typically shape `[B, D]`.
        diagnostics: Telemetry and algorithm-specific runtime diagnostics that are
            automatically merged into `metrics.json` (e.g., SafeFlow's NFE / terminal filter
            rate, or YFlow's PGD convergence iterations / Lipschitz gating ratio).
    """

    samples: torch.Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def with_diagnostics(self, **kwargs: Any) -> SampleResult:
        """Return a new SampleResult with additional or updated diagnostic entries."""
        merged = {**self.diagnostics, **kwargs}
        return SampleResult(samples=self.samples, diagnostics=merged)

    def __iter__(self) -> Iterator[Any]:
        """Support convenient tuple unpacking: `samples, diagnostics = result`."""
        yield self.samples
        yield self.diagnostics

