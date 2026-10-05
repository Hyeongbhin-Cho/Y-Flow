import torch


def _second_diff_matrix(T: int, dtype, device):
    D = torch.zeros(T, T, dtype=dtype, device=device)
    for k in range(T):
        D[k, k] = 1.0
        if k - 1 >= 0:
            D[k, k - 1] = -2.0
        if k - 2 >= 0:
            D[k, k - 2] = 1.0
    return D


def smooth_positions(pos: torch.Tensor, v0: torch.Tensor, dt: float, alpha: float) -> torch.Tensor:
    if alpha <= 0:
        return pos
    B, T, _ = pos.shape
    D = _second_diff_matrix(T, pos.dtype, pos.device)
    b = torch.zeros(B, T, 2, dtype=pos.dtype, device=pos.device)
    b[:, 0] = -v0 * dt
    A = torch.eye(T, dtype=pos.dtype, device=pos.device) + alpha * D.T @ D
    rhs = pos - alpha * torch.einsum("kt,bkc->btc", D, b)
    return torch.linalg.solve(A.expand(B, T, T), rhs)


def heading_from_positions(pos: torch.Tensor, raw_cs: torch.Tensor, min_step: float = 0.3) -> torch.Tensor:
    B, T, _ = pos.shape
    full = torch.cat([torch.zeros(B, 1, 2, dtype=pos.dtype, device=pos.device), pos], dim=1)
    d = torch.empty_like(pos)
    d[:, :-1] = full[:, 2:] - full[:, :-2]
    d[:, -1] = full[:, -1] - full[:, -2]
    n = d.norm(dim=-1, keepdim=True)
    motion = d / n.clamp_min(1e-9)
    out = torch.empty_like(pos)
    prev = torch.zeros(B, 2, dtype=pos.dtype, device=pos.device)
    prev[:, 0] = 1.0
    for k in range(T):
        ok = (n[:, k] > min_step) & ((motion[:, k] * prev).sum(-1, keepdim=True) > 0)
        prev = torch.where(ok, motion[:, k], prev)
        out[:, k] = prev
    return out
