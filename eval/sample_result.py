# -*- coding: utf-8 -*-
# eval/sample_result.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SampleResult:
    samples: torch.Tensor
    diagnostics: dict[str, Any]
