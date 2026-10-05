from __future__ import annotations

import torch


def sample_t(n: int, k: int, device, dtype, schedule: str = "uniform") -> torch.Tensor:
    if schedule == "logit_normal":
        return torch.sigmoid(torch.randn(n, k, device=device, dtype=dtype))
    return torch.rand(n, k, device=device, dtype=dtype)


def fm_loss(model, batch, fut_std: torch.Tensor, n_samples: int = 1, t_schedule: str = "uniform"):
    z1 = batch["fut"] / fut_std
    mask = batch["fut_mask"].to(z1.dtype)
    b = z1.shape[0]
    z1 = z1[:, None].expand(b, n_samples, *z1.shape[1:])
    z0 = torch.randn_like(z1)
    t = sample_t(b, n_samples, z1.device, z1.dtype, t_schedule)
    tt = t[..., None, None]
    zt = (1.0 - tt) * z0 + tt * z1
    ut = z1 - z0
    memory, pad = model.encode(batch)
    v = model.velocity(zt, t, memory, pad)
    err = (v - ut).square().sum(dim=-1) * mask[:, None]
    return err.sum() / (mask.sum() * n_samples * 2).clamp_min(1.0)


@torch.no_grad()
def sample(model, batch, fut_std: torch.Tensor, k: int, n_steps: int, generator: torch.Generator | None = None):
    memory, pad = model.encode(batch)
    b = memory.shape[0]
    shape = (b, k, model.T, 2)
    z = torch.randn(shape, generator=generator, device="cpu" if generator is not None else memory.device)
    z = z.to(memory.device, memory.dtype)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((b,), i * dt, device=z.device, dtype=z.dtype)
        z = z + dt * model.velocity(z, t, memory, pad)
    return z * fut_std
