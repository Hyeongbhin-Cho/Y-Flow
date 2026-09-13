# -*- coding: utf-8 -*-
# model/__init__.py

from model.base import VelocityNet, build_model
from model.mlp import VelocityMLP
from model.time_embed import SinusoidalTimeEmbedding


def download_wan_model(*args, **kwargs):
    from model.download import download_wan_model as _dl

    return _dl(*args, **kwargs)


def download_clevrer_artifact(*args, **kwargs):
    from model.download import download_clevrer_artifact as _dl

    return _dl(*args, **kwargs)


def WanVelocityNet(*args, **kwargs):
    from model.wan import WanVelocityNet as _wvn

    return _wvn(*args, **kwargs)


def build_wan_model(*args, **kwargs):
    from model.wan import build_wan_model as _bwm

    return _bwm(*args, **kwargs)


def CLEVRERVelocityNet(*args, **kwargs):
    from model.clevrer_flow import CLEVRERVelocityNet as _net

    return _net(*args, **kwargs)


def CLEVRERResNetRecognizer(*args, **kwargs):
    from model.clevrer_recognition import CLEVRERResNetRecognizer as _net

    return _net(*args, **kwargs)


__all__ = [
    "SinusoidalTimeEmbedding",
    "VelocityMLP",
    "VelocityNet",
    "CLEVRERVelocityNet",
    "CLEVRERResNetRecognizer",
    "WanVelocityNet",
    "build_model",
    "build_wan_model",
    "download_clevrer_artifact",
    "download_wan_model",
]
