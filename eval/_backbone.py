# -*- coding: utf-8 -*-
# eval/_backbone.py

from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch import nn

from model import build_model
from train.checkpoint import load_checkpoint
from train.ema import EMA
from train.flow_match import ConditionalFlowMatching
from utils.paths import flowmatch_ckpt


def load_frozen_velocity(
    cfg: DictConfig, device: torch.device
) -> tuple[nn.Module, ConditionalFlowMatching]:
    ckpt_path = flowmatch_ckpt(cfg)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing flowmatch checkpoint: {ckpt_path}")
    model = build_model(cfg).to(device)
    ema = EMA(model, decay=float(cfg.train.ema_decay))
    load_checkpoint(ckpt_path, model, ema=ema, map_location=device)
    ema.copy_to(model)
    model.eval()
    return model, ConditionalFlowMatching(cfg)
