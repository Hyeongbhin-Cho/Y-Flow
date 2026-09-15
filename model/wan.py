# -*- coding: utf-8 -*-
# model/wan.py
"""Wan2.1 Flow-Matching Video Transformer & 3D VAE wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from model.base import VelocityNet
from model.download import DEFAULT_LOCAL_DIR, DEFAULT_REPO_ID, download_wan_model, is_already_downloaded


class WanVelocityNet(VelocityNet):
    """Wrapper around WanTransformer3DModel conforming to the VelocityNet interface.
    
    Maps latent video state z and time t in [0, 1] to the velocity field v.
    """

    def __init__(
        self,
        transformer: nn.Module,
        vae: nn.Module | None = None,
        text_dim: int = 4096,
        time_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.text_dim = text_dim
        self.time_scale = time_scale

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Evaluate flow-matching velocity v_t(x).
        
        Args:
            x: Latent video tensor of shape [B, C, T', H', W'].
            t: Normalized time tensor in [0, 1] of shape [B] or scalar.
            encoder_hidden_states: Text conditioning tensor of shape [B, seq_len, text_dim].
                If None, an unconditional zero embedding is supplied.
        Returns:
            v: Predicted velocity tensor of the same shape as x.
        """
        B = x.shape[0]

        # Ensure t has shape [B] and is scaled to [0, 1000] as expected by Wan2.1
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
        if t.ndim == 0:
            t = t.expand(B)
        elif t.ndim == 1 and t.shape[0] == 1 and B > 1:
            t = t.expand(B)

        # Scale from [0, 1] to Wan DiT discrete/continuous timestep range
        timestep = t * self.time_scale
        if timestep.dtype != x.dtype:
            timestep = timestep.to(dtype=x.dtype)

        # Unconditional fallback if no text embedding is provided
        if encoder_hidden_states is None:
            encoder_hidden_states = torch.zeros(
                (B, 1, self.text_dim),
                device=x.device,
                dtype=x.dtype,
            )

        out = self.transformer(
            hidden_states=x,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True,
            **kwargs,
        )

        return out.sample

    @torch.no_grad()
    def decode_latents(self, z: torch.Tensor) -> torch.Tensor:
        """Decode Wan-normalized z [B,C,T',H',W'] to RGB video in [-1,1]."""
        if self.vae is None:
            raise RuntimeError("VAE is not loaded in WanVelocityNet. Set load_vae=True in config.")
        mean, std = self._latent_stats(z)
        vae_dtype = next(self.vae.parameters()).dtype
        unnormalized = z.float() * std.float() + mean.float()
        out = self.vae.decode(unnormalized.to(dtype=vae_dtype))
        video = out.sample if hasattr(out, "sample") else out
        return video

    @torch.no_grad()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode Wan-ready RGB [B,3,T,H,W] into deterministic normalized latents.

        Wan2.1 uses causal temporal compression by four, so input clips must have
        ``T = 1 + 4k`` frames. RealEstate10K samples expose ``wan_video`` padded
        to this contract; the separate ``video`` field keeps only observed frames.
        """
        if self.vae is None:
            raise RuntimeError("VAE is not loaded in WanVelocityNet. Set load_vae=True in config.")
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"Expected video [B,3,T,H,W], got {tuple(video.shape)}")
        temporal_factor = int(getattr(self.vae.config, "scale_factor_temporal", 4) or 4)
        spatial_factor = int(getattr(self.vae.config, "scale_factor_spatial", 8) or 8)
        if (video.shape[2] - 1) % temporal_factor:
            raise ValueError(
                f"Wan VAE requires T=1+{temporal_factor}k frames; got T={video.shape[2]}. "
                "Use the dataset's wan_video field."
            )
        if video.shape[-2] % spatial_factor or video.shape[-1] % spatial_factor:
            raise ValueError(f"Wan VAE requires H,W divisible by {spatial_factor}")
        vae_param = next(self.vae.parameters())
        video = video.to(device=vae_param.device, dtype=vae_param.dtype)
        out = self.vae.encode(video)
        posterior = out.latent_dist if hasattr(out, "latent_dist") else None
        if posterior is None:
            latents = out
        elif hasattr(posterior, "mode"):
            latents = posterior.mode()
        else:
            raise TypeError("Wan VAE encoder output must expose a deterministic latent_dist.mode()")
        mean, std = self._latent_stats(latents)
        return ((latents.float() - mean.float()) / std.float()).to(dtype=latents.dtype)

    def _latent_stats(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return broadcastable Wan posterior mean/std in latent channel order."""
        config = self.vae.config
        mean_values = getattr(config, "latents_mean", None)
        std_values = getattr(config, "latents_std", None)
        if mean_values is None or std_values is None:
            raise ValueError("Wan VAE config must define latents_mean and latents_std")
        mean = torch.as_tensor(mean_values, device=reference.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.as_tensor(std_values, device=reference.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        if mean.shape[1] != reference.shape[1] or std.shape[1] != reference.shape[1]:
            raise ValueError("Wan latent statistics do not match latent channel count")
        if torch.any(std <= 0):
            raise ValueError("Wan latent standard deviations must be positive")
        return mean, std


def build_wan_model(cfg: DictConfig) -> WanVelocityNet:
    """Build or load frozen Wan2.1 velocity model and optional 3D VAE."""
    from diffusers import AutoencoderKLWan, WanTransformer3DModel

    model_cfg = cfg.get("model", {})
    repo_id = str(model_cfg.get("pretrained_path", DEFAULT_REPO_ID))
    local_dir = Path(model_cfg.get("local_dir", DEFAULT_LOCAL_DIR))
    auto_download = bool(model_cfg.get("auto_download", True))
    load_vae = bool(model_cfg.get("load_vae", True))
    dtype_str = str(model_cfg.get("dtype", "bfloat16")).lower()

    dtype = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }.get(dtype_str, torch.bfloat16)

    # 1. Check local weights or trigger download
    if not is_already_downloaded(local_dir, subcomponent="core" if load_vae else "transformer"):
        if auto_download:
            print(f"[Y-Flow] Local checkpoint not found at: {local_dir}")
            print(f"[Y-Flow] Initiating automatic download from Hugging Face: {repo_id}")
            download_wan_model(
                repo_id=repo_id,
                local_dir=local_dir,
                subcomponent="core" if load_vae else "transformer",
            )
        else:
            raise FileNotFoundError(
                f"Wan2.1 weights not found at '{local_dir}'. "
                f"Run 'python -m model.download --local_dir {local_dir}' first."
            )

    # 2. Load DiT Transformer
    transformer_dir = local_dir / "transformer" if (local_dir / "transformer").exists() else local_dir
    print(f"[Y-Flow] Loading WanTransformer3DModel from: {transformer_dir} ({dtype_str})")
    transformer = WanTransformer3DModel.from_pretrained(
        str(transformer_dir),
        torch_dtype=dtype,
        local_files_only=True,
    )

    # 3. Load 3D Causal VAE (optional)
    vae = None
    if load_vae:
        vae_dir = local_dir / "vae"
        if vae_dir.exists():
            print(f"[Y-Flow] Loading AutoencoderKLWan from: {vae_dir} ({dtype_str})")
            vae = AutoencoderKLWan.from_pretrained(
                str(vae_dir),
                torch_dtype=dtype,
                local_files_only=True,
            )
        else:
            print(f"[Y-Flow] Warning: VAE directory not found at {vae_dir}. Skipping VAE load.")

    text_dim = int(getattr(transformer.config, "text_dim", 4096))
    return WanVelocityNet(transformer=transformer, vae=vae, text_dim=text_dim)
