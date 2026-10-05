from __future__ import annotations

import torch


def kmeans(x: torch.Tensor, k: int, iters: int = 15) -> tuple[torch.Tensor, torch.Tensor]:
    b, m, _ = x.shape
    k = min(k, m)
    ar = torch.arange(b, device=x.device)
    idx = torch.zeros(b, k, dtype=torch.long, device=x.device)
    idx[:, 0] = x.norm(dim=-1).argmax(dim=1)
    d = torch.full((b, m), float("inf"), device=x.device, dtype=x.dtype)
    for j in range(1, k):
        d = torch.minimum(d, (x - x[ar, idx[:, j - 1]][:, None]).norm(dim=-1))
        idx[:, j] = d.argmax(dim=1)
    cen = x[ar[:, None], idx]
    for _ in range(iters):
        a = (x[:, :, None] - cen[:, None]).norm(dim=-1).argmin(dim=-1)
        oh = torch.nn.functional.one_hot(a, k).to(x.dtype)
        w = oh.sum(1)
        cen = torch.where(w[..., None] > 0, torch.einsum("bmk,bmd->bkd", oh, x) / w.clamp_min(1)[..., None], cen)
    a = (x[:, :, None] - cen[:, None]).norm(dim=-1).argmin(dim=-1)
    w = torch.nn.functional.one_hot(a, k).to(x.dtype).sum(1)
    return cen, w / w.sum(-1, keepdim=True).clamp_min(1)


def lane_candidates(batch: dict, horizon_s: float, dt: float, band=(0.25, 1.35), behind: float = -5.0):
    lane, lmask = batch["lane"], batch["lane_mask"]
    b, L, P, _ = lane.shape
    hist, hmask = batch["hist"], batch["hist_mask"]
    v = torch.zeros(b, device=lane.device, dtype=lane.dtype)
    if hist.shape[1] > 1:
        step = (hist[:, -1] - hist[:, -2]).norm(dim=-1) / dt
        v = torch.where(hmask[:, -1] & hmask[:, -2], step, v)
    reach = (v * horizon_s).clamp_min(5.0)
    c = lane.reshape(b, L * P, 2)
    m = lmask.reshape(b, L * P)
    d = c.norm(dim=-1)
    ok = m & (d >= band[0] * reach[:, None]) & (d <= band[1] * reach[:, None]) & (c[..., 0] > behind)
    return c, ok


def score_candidates(cand: torch.Tensor, ok: torch.Tensor, ends: torch.Tensor, sigma: float = 3.0):
    d2 = (cand[:, :, None] - ends[:, None]).square().sum(-1)
    s = torch.exp(-d2 / (2 * sigma ** 2)).sum(-1)
    return s.masked_fill(~ok, float("-inf"))


def nms_select(cand: torch.Tensor, score: torch.Tensor, k: int, radius: float = 4.0) -> tuple[torch.Tensor, torch.Tensor]:
    b = cand.shape[0]
    out = torch.zeros(b, k, 2, dtype=cand.dtype, device=cand.device)
    n = torch.zeros(b, dtype=torch.long, device=cand.device)
    s = score.clone()
    ar = torch.arange(b, device=cand.device)
    for j in range(k):
        best = s.argmax(dim=1)
        alive = torch.isfinite(s[ar, best])
        out[:, j] = torch.where(alive[:, None], cand[ar, best], out[:, max(j - 1, 0)])
        n = n + alive.long()
        s = s.masked_fill((cand - cand[ar, best][:, None]).norm(dim=-1) < radius, float("-inf"))
    return out, n


def propose(batch: dict, ends: torch.Tensor, k: int, cfg: dict, horizon_s: float, dt: float) -> torch.Tensor:
    src = str(cfg.get("goal_source", "lane"))
    fallback, _ = kmeans(ends, k)
    if src != "lane":
        return fallback
    cand, ok = lane_candidates(batch, horizon_s, dt, band=tuple(cfg.get("goal_band", (0.25, 1.35))),
                               behind=float(cfg.get("goal_behind", -5.0)))
    goals, n = nms_select(cand, score_candidates(cand, ok, ends, float(cfg.get("goal_sigma", 3.0))), k,
                          float(cfg.get("goal_nms", 4.0)))
    enough = n >= k
    return torch.where(enough[:, None, None], goals, fallback)
