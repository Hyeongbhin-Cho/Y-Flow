from types import SimpleNamespace

import pytest
import torch
from torch import nn

from model.wan import WanVelocityNet


class FakePosterior:
    def __init__(self, value):
        self.value = value

    def mode(self):
        return self.value


class FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            scale_factor_temporal=4,
            scale_factor_spatial=8,
            latents_mean=[1., -1.],
            latents_std=[2., 4.],
        )

    def encode(self, video):
        # Deterministic stand-in: mimic Wan's temporal/spatial compression shape.
        raw = video[:, :2, ::4, ::8, ::8]
        return SimpleNamespace(latent_dist=FakePosterior(raw))

    def decode(self, latents):
        return SimpleNamespace(sample=latents)


class FakeTransformer(nn.Module):
    def forward(self, **kwargs):
        return SimpleNamespace(sample=kwargs["hidden_states"])


def test_wan_encode_normalize_decode_roundtrip():
    model = WanVelocityNet(FakeTransformer(), vae=FakeVAE())
    video = torch.linspace(-1, 1, 2 * 3 * 5 * 32 * 32).reshape(2, 3, 5, 32, 32)
    latent = model.encode_video(video)
    assert latent.shape == (2, 2, 2, 4, 4)
    raw = video[:, :2, ::4, ::8, ::8]
    mean = torch.tensor([1., -1.]).view(1, 2, 1, 1, 1)
    std = torch.tensor([2., 4.]).view(1, 2, 1, 1, 1)
    torch.testing.assert_close(latent, (raw - mean) / std)
    torch.testing.assert_close(model.decode_latents(latent), raw)


def test_wan_encode_requires_causal_frame_count():
    model = WanVelocityNet(FakeTransformer(), vae=FakeVAE())
    with pytest.raises(ValueError, match=r"1\+4k"):
        model.encode_video(torch.zeros(1, 3, 8, 32, 32))


def test_wan_velocity_interface_shape_and_time():
    transformer = FakeTransformer()
    model = WanVelocityNet(transformer, text_dim=6)
    x = torch.randn(2, 4, 2, 3, 3)
    output = model(x, torch.tensor([0.1, 0.4]))
    torch.testing.assert_close(output, x)
