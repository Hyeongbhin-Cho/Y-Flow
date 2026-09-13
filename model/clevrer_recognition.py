# -*- coding: utf-8 -*-
"""ResNet-34 video recognizer for CLEVRER object and physical state."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from utils.paths import ROOT


def _sincos_2d(height: int, width: int, dim: int, *, device, dtype) -> torch.Tensor:
    if dim % 4:
        raise ValueError(f"2D sine/cosine embedding dimension must be divisible by 4, got {dim}")
    quarter = dim // 4
    omega = torch.arange(quarter, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(quarter - 1, 1)))
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    x_phase = x.reshape(-1, 1) * omega.reshape(1, -1)
    y_phase = y.reshape(-1, 1) * omega.reshape(1, -1)
    embedding = torch.cat(
        [x_phase.sin(), x_phase.cos(), y_phase.sin(), y_phase.cos()], dim=-1
    )
    return embedding.to(dtype=dtype).unsqueeze(0)


def _load_resnet_state(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(
            f"ResNet-34 weights not found at {path}. Run "
            "python scripts/setup_resnet34.py before building the pretrained model."
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Invalid ResNet-34 state dict in {path}")
    return payload


class CLEVRERResNetRecognizer(nn.Module):
    """Predict fixed-slot CLEVRER scene state directly from RGB clips.

    Input video is RGB [B, 3, T, H, W] in [-1, 1]. The ResNet-34 stem through
    layer3 produces spatial maps; object queries read each frame, and a temporal
    Transformer predicts attributes, tracks, visibility, and collisions.
    """

    def __init__(
        self,
        *,
        n_slots: int = 6,
        n_frames: int = 33,
        n_color: int = 8,
        n_material: int = 2,
        n_shape: int = 3,
        dim: int = 256,
        slot_layers: int = 2,
        temporal_layers: int = 4,
        n_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        pretrained: bool = True,
        weights_path: str | Path | None = None,
        backbone_mode: str = "frozen",
    ) -> None:
        super().__init__()
        if dim % 4:
            raise ValueError("dim must be divisible by 4 for 2D sine/cosine positions")
        if backbone_mode not in ("frozen", "layer3"):
            raise ValueError("backbone_mode must be 'frozen' or 'layer3'")
        if n_slots <= 0 or n_frames <= 0:
            raise ValueError("n_slots and n_frames must be positive")

        from torchvision.models import resnet34

        resnet = resnet34(weights=None)
        if pretrained:
            if weights_path is None:
                weights_path = ROOT / "checkpoints" / "resnet34_imagenet1k_v1.pth"
            resnet.load_state_dict(_load_resnet_state(weights_path), strict=True)

        self.n_slots = int(n_slots)
        self.n_frames = int(n_frames)
        self.dim = int(dim)
        self.backbone_mode = backbone_mode
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        # Do not retain the unused ImageNet layer4, global pool, or classifier.
        del resnet

        self.lateral2 = nn.Conv2d(128, dim, kernel_size=1)
        self.lateral3 = nn.Conv2d(256, dim, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        self.slot_queries = nn.Parameter(torch.empty(1, n_slots, dim))
        nn.init.normal_(self.slot_queries, std=0.02)
        slot_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.slot_decoder = nn.TransformerDecoder(
            slot_layer, num_layers=slot_layers, norm=nn.LayerNorm(dim)
        )

        self.temporal_position = nn.Parameter(torch.empty(1, n_frames, dim))
        nn.init.normal_(self.temporal_position, std=0.01)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, num_layers=temporal_layers, norm=nn.LayerNorm(dim)
        )

        self.pool_score = nn.Linear(dim, 1)
        self.objectness_head = self._mlp(dim, 1, dropout)
        self.color_head = self._mlp(dim, n_color, dropout)
        self.material_head = self._mlp(dim, n_material, dropout)
        self.shape_head = self._mlp(dim, n_shape, dropout)
        self.visibility_head = self._mlp(dim, 1, dropout)
        self.position_head = self._mlp(dim, 3, dropout)
        self.velocity_head = self._mlp(dim, 3, dropout)
        self.collision_head = nn.Sequential(
            nn.Linear(3 * dim + 6, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )
        pair_indices = torch.triu_indices(n_slots, n_slots, offset=1)
        self.register_buffer("pair_i", pair_indices[0], persistent=False)
        self.register_buffer("pair_j", pair_indices[1], persistent=False)

        self.register_buffer(
            "imagenet_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "imagenet_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=True,
        )
        self._configure_backbone_gradients()

    @staticmethod
    def _mlp(dim: int, out_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, out_dim),
        )

    def _configure_backbone_gradients(self) -> None:
        for parameter in self.stem.parameters():
            parameter.requires_grad_(False)
        for module in (self.layer1, self.layer2, self.layer3):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        if self.backbone_mode == "layer3":
            for parameter in self.layer3.parameters():
                parameter.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.backbone_mode == "frozen":
            self.stem.eval()
            self.layer1.eval()
            self.layer2.eval()
            self.layer3.eval()
        elif mode and self.backbone_mode == "layer3":
            self.stem.eval()
            self.layer1.eval()
            self.layer2.eval()
            self.layer3.train()
        return self

    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        # Dataset frames are [-1, 1]; torchvision ImageNet weights expect normalized [0, 1].
        frames = (frames + 1.0) * 0.5
        frames = (frames - self.imagenet_mean) / self.imagenet_std
        x = self.stem(frames)
        x = self.layer1(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        layer2 = self.lateral2(layer2)
        layer3 = self.lateral3(layer3)
        layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(layer2 + layer3)

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"expected video [B,3,T,H,W], got {tuple(video.shape)}")
        batch, _channels, time, height, width = video.shape
        if time > self.n_frames:
            raise ValueError(f"got T={time} frames, configured maximum is {self.n_frames}")
        if height < 32 or width < 32:
            raise ValueError("ResNet-34 input height and width must both be at least 32")

        frames = video.permute(0, 2, 1, 3, 4).reshape(batch * time, 3, height, width)
        feature_map = self._encode_frames(frames)
        _, _dim, feat_h, feat_w = feature_map.shape
        spatial_pos = _sincos_2d(
            feat_h, feat_w, self.dim, device=feature_map.device, dtype=feature_map.dtype
        )
        memory = feature_map.flatten(2).transpose(1, 2) + spatial_pos
        queries = self.slot_queries.expand(batch * time, -1, -1)
        frame_slots = self.slot_decoder(queries, memory)
        frame_slots = frame_slots.reshape(batch, time, self.n_slots, self.dim)

        # Apply temporal attention independently to each persistent, clip-matched slot.
        temporal = frame_slots.permute(0, 2, 1, 3).reshape(batch * self.n_slots, time, self.dim)
        temporal = temporal + self.temporal_position[:, :time]
        temporal = self.temporal_encoder(temporal)
        temporal = temporal.reshape(batch, self.n_slots, time, self.dim).permute(0, 2, 1, 3)

        pool_weights = self.pool_score(temporal).squeeze(-1).softmax(dim=1)
        pooled = (temporal * pool_weights.unsqueeze(-1)).sum(dim=1)
        positions = self.position_head(temporal)
        velocities = self.velocity_head(temporal)

        pair_i, pair_j = self.pair_i, self.pair_j
        slot_i, slot_j = temporal[:, :, pair_i], temporal[:, :, pair_j]
        symmetric_features = torch.cat(
            [slot_i + slot_j, (slot_i - slot_j).abs(), slot_i * slot_j], dim=-1
        )
        relative_state = torch.cat(
            [
                (positions[:, :, pair_i] - positions[:, :, pair_j]).abs(),
                (velocities[:, :, pair_i] - velocities[:, :, pair_j]).abs(),
            ],
            dim=-1,
        )
        collision_logits = self.collision_head(
            torch.cat([symmetric_features, relative_state], dim=-1)
        ).squeeze(-1)

        return {
            "objectness_logits": self.objectness_head(pooled).squeeze(-1),
            "color_logits": self.color_head(pooled),
            "material_logits": self.material_head(pooled),
            "shape_logits": self.shape_head(pooled),
            "visibility_logits": self.visibility_head(temporal).squeeze(-1).permute(0, 2, 1),
            "positions": positions.permute(0, 2, 1, 3),
            "velocities": velocities.permute(0, 2, 1, 3),
            "collision_logits": collision_logits,
            "slot_features": temporal.permute(0, 2, 1, 3),
        }


def build_clevrer_resnet34_model(cfg: DictConfig) -> CLEVRERResNetRecognizer:
    model_cfg = cfg.model
    return CLEVRERResNetRecognizer(
        n_slots=int(cfg.data.get("n_slots", 6)),
        n_frames=int(cfg.data.get("n_frames", 33)),
        n_color=int(model_cfg.get("n_color", 8)),
        n_material=int(model_cfg.get("n_material", 2)),
        n_shape=int(model_cfg.get("n_shape", 3)),
        dim=int(model_cfg.get("dim", 256)),
        slot_layers=int(model_cfg.get("slot_layers", 2)),
        temporal_layers=int(model_cfg.get("temporal_layers", 4)),
        n_heads=int(model_cfg.get("n_heads", 8)),
        ff_dim=int(model_cfg.get("ff_dim", 1024)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        pretrained=bool(model_cfg.get("pretrained", True)),
        weights_path=model_cfg.get("weights_path"),
        backbone_mode=str(model_cfg.get("backbone_mode", "frozen")),
    )
