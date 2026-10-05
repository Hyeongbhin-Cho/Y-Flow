from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vfm.anchors import kmeans, lane_candidates, nms_select

SCALE = 40.0


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h, o))


def cand_features(cand: torch.Tensor, reach: torch.Tensor) -> torch.Tensor:
    d = cand.norm(dim=-1, keepdim=True)
    u = cand / d.clamp_min(1e-6)
    return torch.cat([cand / SCALE, d / SCALE, u, d / reach[:, None, None].clamp_min(1.0)], dim=-1)


class GoalScorer(nn.Module):

    def __init__(self, d_model: int = 128, hidden: int = 128):
        super().__init__()
        self.ctx = nn.Linear(d_model, hidden)
        self.cand = _mlp(6, hidden, hidden)
        self.score = _mlp(2 * hidden, hidden, 1)
        self.offset = _mlp(2 * hidden, hidden, 2)

    def forward(self, memory: torch.Tensor, pad: torch.Tensor, cand: torch.Tensor, reach: torch.Tensor):
        m = (~pad).to(memory.dtype)[..., None]
        ctx = self.ctx((memory * m).sum(1) / m.sum(1).clamp_min(1))
        h = torch.cat([self.cand(cand_features(cand, reach)),
                       ctx[:, None].expand(-1, cand.shape[1], -1)], dim=-1)
        return self.score(h).squeeze(-1), self.offset(h) * SCALE / 8.0


def reach_of(batch: dict, horizon_s: float, dt: float) -> torch.Tensor:
    hist, hmask = batch["hist"], batch["hist_mask"]
    v = torch.zeros(hist.shape[0], device=hist.device, dtype=hist.dtype)
    if hist.shape[1] > 1:
        v = torch.where(hmask[:, -1] & hmask[:, -2], (hist[:, -1] - hist[:, -2]).norm(dim=-1) / dt, v)
    return (v * horizon_s).clamp_min(5.0)


def losses(net: GoalScorer, model, batch: dict, cfg: dict, horizon_s: float, dt: float):
    with torch.no_grad():
        memory, pad = model.encode(batch)
    cand, ok = lane_candidates(batch, horizon_s, dt, band=tuple(cfg.get("goal_band", (0.25, 1.35))),
                               behind=float(cfg.get("goal_behind", -5.0)))
    fmask = batch["fut_mask"]
    last = fmask.shape[1] - 1 - fmask.flip(-1).float().argmax(-1)
    end = batch["fut"][torch.arange(len(last), device=last.device), last]
    d = (cand - end[:, None]).norm(dim=-1).masked_fill(~ok, float("inf"))
    tgt = d.argmin(dim=1)
    usable = torch.isfinite(d.gather(1, tgt[:, None]).squeeze(1)) & ok.any(1)
    if not usable.any():
        z = memory.sum() * 0.0
        return z, z, usable.float().mean()
    logits, off = net(memory, pad, cand, reach_of(batch, horizon_s, dt))
    logits = logits.masked_fill(~ok, float("-inf"))
    ce = F.cross_entropy(logits[usable], tgt[usable])
    ar = torch.arange(len(tgt), device=tgt.device)
    reg = F.smooth_l1_loss(off[ar, tgt][usable], (end - cand[ar, tgt])[usable])
    return ce, reg, usable.float().mean()


@torch.no_grad()
def propose(net: GoalScorer, model, batch: dict, ends: torch.Tensor, k: int, cfg: dict, horizon_s: float,
            dt: float) -> torch.Tensor:
    memory, pad = model.encode(batch)
    cand, ok = lane_candidates(batch, horizon_s, dt, band=tuple(cfg.get("goal_band", (0.25, 1.35))),
                               behind=float(cfg.get("goal_behind", -5.0)))
    logits, off = net(memory, pad, cand, reach_of(batch, horizon_s, dt))
    moved = cand + off.clamp(-float(cfg.get("goal_offset_max", 6.0)), float(cfg.get("goal_offset_max", 6.0)))
    goals, n = nms_select(moved, logits.masked_fill(~ok, float("-inf")), k, float(cfg.get("goal_nms", 4.0)))
    fallback, _ = kmeans(ends, k)
    return torch.where((n >= k)[:, None, None], goals, fallback)
