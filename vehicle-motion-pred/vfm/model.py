from __future__ import annotations

import math

import torch
from torch import nn


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h, o))


class PolylineEncoder(nn.Module):

    def __init__(self, in_dim: int, d: int):
        super().__init__()
        self.point = _mlp(in_dim, d, d)
        self.out = _mlp(2 * d, d, d)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.point(feats)
        neg = torch.finfo(h.dtype).min
        pooled = h.masked_fill(~mask[..., None], neg).amax(dim=-2)
        valid = mask.any(dim=-1)
        pooled = torch.where(valid[..., None], pooled, torch.zeros_like(pooled))
        h = torch.cat([h, pooled[..., None, :].expand_as(h)], dim=-1)
        h = self.out(h).masked_fill(~mask[..., None], neg).amax(dim=-2)
        h = torch.where(valid[..., None], h, torch.zeros_like(h))
        return h, valid


def _point_features(xy: torch.Tensor, mask: torch.Tensor, scale: float, with_time: bool) -> torch.Tensor:
    xy = xy / scale
    prev = torch.cat([xy[..., :1, :], xy[..., :-1, :]], dim=-2)
    pmask = torch.cat([mask[..., :1], mask[..., :-1]], dim=-1)
    delta = torch.where((mask & pmask)[..., None], xy - prev, torch.zeros_like(xy))
    feats = [xy, delta]
    if with_time:
        n = xy.shape[-2]
        tt = torch.linspace(-1.0, 0.0, n, device=xy.device, dtype=xy.dtype)
        feats.append(tt.expand(*xy.shape[:-1])[..., None])
    return torch.cat(feats, dim=-1) * mask[..., None]


class TimeEmbedding(nn.Module):

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.d // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
        args = 1000.0 * t[..., None] * freqs
        return self.mlp(torch.cat([args.sin(), args.cos()], dim=-1))


class CrossBlock(nn.Module):

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm_f = nn.LayerNorm(d, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, q, temb, memory, memory_pad):
        s1, b1, g1, s2, b2, g2 = self.ada(temb).chunk(6, dim=-1)
        h = self.norm_q(q) * (1 + s1) + b1
        h, _ = self.attn(h, memory, memory, key_padding_mask=memory_pad, need_weights=False)
        q = q + (1 + g1) * h
        h = self.norm_f(q) * (1 + s2) + b2
        return q + (1 + g2) * self.ffn(h)


class FlowForecaster(nn.Module):

    def __init__(
        self,
        future_steps: int,
        d_model: int = 128,
        n_heads: int = 8,
        enc_layers: int = 3,
        dec_layers: int = 3,
        dropout: float = 0.1,
        ctx_scale_m: float = 50.0,
    ):
        super().__init__()
        self.T = int(future_steps)
        self.ctx_scale = float(ctx_scale_m)
        d = d_model
        self.agent_enc = PolylineEncoder(5, d)
        self.lane_enc = PolylineEncoder(4, d)
        self.kind = nn.Embedding(3, d)
        layer = nn.TransformerEncoderLayer(
            d, n_heads, 4 * d, dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.scene = nn.TransformerEncoder(layer, enc_layers, enable_nested_tensor=False)
        self.scene_norm = nn.LayerNorm(d)
        self.time = TimeEmbedding(d)
        self.x_in = _mlp(2 * self.T, 2 * d, d)
        self.blocks = nn.ModuleList(CrossBlock(d, n_heads, dropout) for _ in range(dec_layers))
        self.out_norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 2 * self.T)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def encode(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        hist = batch["hist"][:, None]
        hmask = batch["hist_mask"][:, None]
        agents = torch.cat([hist, batch["nbr"]], dim=1)
        amask = torch.cat([hmask, batch["nbr_mask"]], dim=1)
        a_tok, a_valid = self.agent_enc(_point_features(agents, amask, self.ctx_scale, True), amask)
        kind = torch.ones(a_tok.shape[:2], dtype=torch.long, device=a_tok.device)
        kind[:, 0] = 0
        a_tok = a_tok + self.kind(kind)
        lmask = batch["lane_mask"]
        l_tok, l_valid = self.lane_enc(_point_features(batch["lane"], lmask, self.ctx_scale, False), lmask)
        l_tok = l_tok + self.kind.weight[2]
        tokens = torch.cat([a_tok, l_tok], dim=1)
        pad = ~torch.cat([a_valid, l_valid], dim=1)
        pad[:, 0] = False
        mem = self.scene_norm(self.scene(tokens, src_key_padding_mask=pad))
        return mem, pad

    def velocity(self, x: torch.Tensor, t: torch.Tensor, memory: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        b, k = x.shape[:2]
        if t.ndim == 1:
            t = t[:, None].expand(b, k)
        temb = self.time(t)
        q = self.x_in(x.reshape(b, k, -1)) + temb + memory[:, :1]
        for block in self.blocks:
            q = block(q, temb, memory, pad)
        return self.out(self.out_norm(q)).reshape(b, k, self.T, 2)

    def forward(self, x, t, batch):
        memory, pad = self.encode(batch)
        return self.velocity(x, t, memory, pad)


def build_model(cfg, future_steps: int) -> FlowForecaster:
    m = cfg.model
    return FlowForecaster(
        future_steps=future_steps,
        d_model=int(m.get("d_model", 128)),
        n_heads=int(m.get("n_heads", 8)),
        enc_layers=int(m.get("enc_layers", 3)),
        dec_layers=int(m.get("dec_layers", 3)),
        dropout=float(m.get("dropout", 0.1)),
        ctx_scale_m=float(m.get("ctx_scale_m", 50.0)),
    )
