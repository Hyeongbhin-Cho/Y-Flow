from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Rows:
    G: torch.Tensor
    c: torch.Tensor
    r: torch.Tensor
    A: torch.Tensor
    lo: torch.Tensor
    hi: torch.Tensor


def _stack(rows: Rows):
    B, Nb, _, D = rows.G.shape
    M = torch.cat([rows.G.reshape(B, Nb * 2, D), rows.A], dim=1)
    return M, Nb


def _project(w: torch.Tensor, rows: Rows, Nb: int) -> torch.Tensor:
    B = w.shape[0]
    wb = w[:, : Nb * 2].reshape(B, Nb, 2) - rows.c
    n = wb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    r = rows.r[..., None]
    wb = torch.where(n > r, wb / n * r, wb) + rows.c
    ws = torch.maximum(torch.minimum(w[:, Nb * 2:], rows.hi), rows.lo)
    return torch.cat([wb.reshape(B, Nb * 2), ws], dim=1)


def violation(z: torch.Tensor, rows: Rows) -> torch.Tensor:
    B = z.shape[0]
    gb = torch.einsum("bnkd,bd->bnk", rows.G, z) - rows.c
    vb = (gb.norm(dim=-1) - rows.r).clamp_min(0.0)
    s = torch.einsum("bnd,bd->bn", rows.A, z)
    vs = torch.maximum((rows.lo - s).clamp_min(0.0), (s - rows.hi).clamp_min(0.0))
    vs = torch.nan_to_num(vs, nan=0.0, posinf=0.0)
    vb_max = vb.amax(dim=1) if vb.shape[1] else torch.zeros(B, dtype=z.dtype, device=z.device)
    vs_max = vs.amax(dim=1) if vs.shape[1] else torch.zeros(B, dtype=z.dtype, device=z.device)
    return torch.maximum(vb_max, vs_max)


def admm_project(
    zbar: torch.Tensor,
    rows: Rows,
    iters: int = 400,
    rho: float = 300.0,
    alpha: float = 1.6,
    z0: Optional[torch.Tensor] = None,
    tol: float = 1e-5,
    check_every: int = 50,
) -> torch.Tensor:
    M, Nb = _stack(rows)
    B, R, D = M.shape
    eye = torch.eye(D, dtype=zbar.dtype, device=zbar.device).expand(B, D, D)
    K = eye + rho * M.transpose(1, 2) @ M
    L = torch.linalg.cholesky(K)
    z = zbar.clone() if z0 is None else z0.clone()
    y = _project(torch.einsum("brd,bd->br", M, z), rows, Nb)
    u = torch.zeros_like(y)
    Mt = M.transpose(1, 2)
    for it in range(iters):
        rhs = zbar + rho * torch.einsum("bdr,br->bd", Mt, y - u)
        z = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
        Mz = torch.einsum("brd,bd->br", M, z)
        Mz_hat = alpha * Mz + (1 - alpha) * y
        y = _project(Mz_hat + u, rows, Nb)
        u = u + Mz_hat - y
        if (it + 1) % check_every == 0 and violation(z, rows).max() < tol:
            break
    return z
