from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

from flow_drive.yflow.solver import Rows

EGO_HALF_WIDTH = 1.1485
REAR_OVERHANG = 1.127
FRONT_FROM_REAR_AXLE = 4.049
_HALF_LEN_PER_CIRCLE = (REAR_OVERHANG + FRONT_FROM_REAR_AXLE) / 4.0
CIRCLE_OFFSETS = (-REAR_OVERHANG + _HALF_LEN_PER_CIRCLE,
                  FRONT_FROM_REAR_AXLE - _HALF_LEN_PER_CIRCLE)
CIRCLE_RADIUS = float(np.hypot(_HALF_LEN_PER_CIRCLE, EGO_HALF_WIDTH))


@dataclass
class CorridorData:
    center: torch.Tensor
    normal: torch.Tensor
    tangent: torch.Tensor
    lo: torch.Tensor
    hi: torch.Tensor
    valid: torch.Tensor


def kin_rows(B, T, v0, v_limit, a_max, dt, v_tol, dtype, device, decel=None):
    D = 2 * T
    G = torch.zeros(B, 2 * T, 2, D, dtype=dtype, device=device)
    c = torch.zeros(B, 2 * T, 2, dtype=dtype, device=device)
    r = torch.zeros(B, 2 * T, dtype=dtype, device=device)
    I2 = torch.eye(2, dtype=dtype, device=device)
    speed0 = v0.norm(dim=-1)
    k = torch.arange(T, dtype=dtype, device=device)
    dec = a_max if decel is None else decel
    vmax = torch.maximum(v_limit[:, None] * (1 + v_tol), speed0[:, None] - dec * (k[None] + 1) * dt)
    for t in range(T):
        G[:, t, :, 2 * t:2 * t + 2] = I2
        if t >= 1:
            G[:, t, :, 2 * (t - 1):2 * t] = -I2
        r[:, t] = vmax[:, t] * dt
        j = T + t
        G[:, j, :, 2 * t:2 * t + 2] = I2
        if t >= 1:
            G[:, j, :, 2 * (t - 1):2 * t] = -2 * I2
        if t >= 2:
            G[:, j, :, 2 * (t - 2):2 * (t - 1)] = I2
        if t == 0:
            c[:, j] = v0 * dt
        r[:, j] = a_max * dt * dt
    return G, c, r


def accel_lonlat_rows(G_acc, c_acc, cs_lin, a_lon_max, a_lon_min, a_lat_max, dt):
    B, T = G_acc.shape[:2]
    h = torch.empty(B, T, 2, dtype=G_acc.dtype, device=G_acc.device)
    h[:, 0, 0], h[:, 0, 1] = 1.0, 0.0
    h[:, 1:] = cs_lin[:, :-1].to(G_acc.dtype)
    h = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    n = torch.stack([-h[..., 1], h[..., 0]], -1)
    dt2 = dt * dt
    A_lon = torch.einsum("btk,btkd->btd", h, G_acc)
    A_lat = torch.einsum("btk,btkd->btd", n, G_acc)
    s_lon = (h * c_acc).sum(-1)
    s_lat = (n * c_acc).sum(-1)
    A = torch.cat([A_lon, A_lat], 1)
    lo = torch.cat([s_lon + a_lon_min * dt2, s_lat - a_lat_max * dt2], 1)
    hi = torch.cat([s_lon + a_lon_max * dt2, s_lat + a_lat_max * dt2], 1)
    return A, lo, hi


def retime_lon(pos, v0, dt, a_lon_min, a_lon_max):
    B, T, _ = pos.shape
    full = torch.cat([torch.zeros(B, 1, 2, dtype=pos.dtype, device=pos.device), pos], 1)
    seg = (full[:, 1:] - full[:, :-1]).norm(dim=-1)
    s = torch.cat([torch.zeros(B, 1, dtype=pos.dtype, device=pos.device), seg.cumsum(1)], 1)
    v_raw = seg / dt
    v = torch.empty_like(v_raw)
    prev = v0.norm(dim=-1)
    for k in range(T):
        prev = torch.minimum(torch.maximum(v_raw[:, k], prev + a_lon_min * dt), prev + a_lon_max * dt).clamp_min(0.0)
        v[:, k] = prev
    s_new = (v * dt).cumsum(1)
    d_end = full[:, -1] - full[:, -2]
    d_end = torch.where(d_end.norm(dim=-1, keepdim=True) > 1e-6, d_end,
                        torch.tensor([1.0, 0.0], dtype=pos.dtype, device=pos.device).expand_as(d_end))
    d_end = d_end / d_end.norm(dim=-1, keepdim=True)
    out = torch.empty_like(pos)
    for b in range(B):
        idx = torch.searchsorted(s[b].contiguous(), s_new[b].contiguous()).clamp(1, T)
        s0, s1 = s[b, idx - 1], s[b, idx]
        w = ((s_new[b] - s0) / (s1 - s0).clamp_min(1e-9)).clamp(0, 1)[:, None]
        q = full[b, idx - 1] + w * (full[b, idx] - full[b, idx - 1])
        beyond = (s_new[b] - s[b, -1]).clamp_min(0)[:, None]
        out[b] = q + beyond * d_end[b]
    return out


def circle_maps(pos_lin, cs_lin, offset: float, min_step: float = 0.3):
    B, T, _ = pos_lin.shape
    D = 2 * T
    dtype, device = pos_lin.dtype, pos_lin.device
    full = torch.cat([torch.zeros(B, 1, 2, dtype=dtype, device=device), pos_lin], 1)
    L = torch.zeros(B, T, 2, D, dtype=dtype, device=device)
    const = torch.zeros(B, T, 2, dtype=dtype, device=device)
    I2 = torch.eye(2, dtype=dtype, device=device)
    for k in range(1, T + 1):
        j = k - 1
        L[:, j, :, 2 * j:2 * j + 2] = I2
        a, b = (k + 1, k - 1) if k < T else (k, k - 1)
        dbar = full[:, a] - full[:, b]
        s = dbar.norm(dim=-1)
        h = dbar / s.clamp_min(1e-9)[:, None]
        moving = (s > min_step) & ((h * cs_lin[:, j]).sum(-1) > 0)
        hbar = torch.where(moving[:, None], h, cs_lin[:, j])
        J = (offset / s.clamp_min(1e-9))[:, None, None] * (I2 - hbar[:, :, None] * hbar[:, None, :])
        J = torch.where(moving[:, None, None], J, torch.zeros_like(J))
        L[:, j, :, 2 * (a - 1):2 * a] += J
        if b >= 1:
            L[:, j, :, 2 * (b - 1):2 * b] -= J
        const[:, j] = offset * hbar - torch.einsum("bij,bj->bi", J, dbar)
    Q = torch.einsum("btkd,bd->btk", L, pos_lin.reshape(B, D)) + const
    return L, const, Q


def corridor_rows(pos_lin, cs_lin, corr: CorridorData, margin: float, max_dist: float = 8.0):
    B, T, _ = pos_lin.shape
    D = 2 * T
    dtype, device = pos_lin.dtype, pos_lin.device
    A = torch.zeros(B, 2 * T, D, dtype=dtype, device=device)
    lo = torch.full((B, 2 * T), -float("inf"), dtype=dtype, device=device)
    hi = torch.full((B, 2 * T), float("inf"), dtype=dtype, device=device)
    center = corr.center.to(dtype=dtype, device=device)
    normal = corr.normal.to(dtype=dtype, device=device)
    for ci, off in enumerate(CIRCLE_OFFSETS):
        L, const, Q = circle_maps(pos_lin, cs_lin, off)
        dist, idx = (Q[:, :, None, :] - center[None, None]).norm(dim=-1).min(dim=-1)
        n, c = normal[idx], center[idx]
        l = corr.lo.to(dtype=dtype, device=device)[idx] + margin
        h = corr.hi.to(dtype=dtype, device=device)[idx] - margin
        ok = corr.valid.to(device)[idx] & (dist < max_dist) & (h > l)
        a = torch.einsum("btk,btkd->btd", n, L)
        shift = (n * (c - const)).sum(-1)
        sl = slice(ci * T, (ci + 1) * T)
        A[:, sl] = torch.where(ok[..., None], a, torch.zeros_like(a))
        lo[:, sl] = torch.where(ok, l + shift, lo[:, sl])
        hi[:, sl] = torch.where(ok, h + shift, hi[:, sl])
    return A, lo, hi


def corridor_feasible_now(corr: CorridorData, margin: float, tol: float = 0.05,
                          max_dist: float = 8.0) -> bool:
    center = corr.center.detach().cpu().double()
    normal = corr.normal.detach().cpu().double()
    lo_all = corr.lo.detach().cpu().double()
    hi_all = corr.hi.detach().cpu().double()
    valid = corr.valid.detach().cpu()
    for off in CIRCLE_OFFSETS:
        q = torch.tensor([off, 0.0], dtype=torch.float64)
        dist, idx = (q - center).norm(dim=-1).min(dim=0)
        l, h = lo_all[idx] + margin, hi_all[idx] - margin
        if not (bool(valid[idx]) and float(dist) < max_dist and float(h) > float(l)):
            continue
        d = float((normal[idx] * (q - center[idx])).sum())
        if d < float(l) - tol or d > float(h) + tol:
            return False
    return True


def _frenet(pts, corr: CorridorData, return_idx: bool = False):
    center = corr.center.detach().cpu().double().numpy()
    normal = corr.normal.detach().cpu().double().numpy()
    tangent = corr.tangent.detach().cpu().double().numpy()
    s_samp = corr_s(corr).detach().cpu().double().numpy()
    d2 = ((pts[..., None, :] - center) ** 2).sum(-1)
    j = d2.argmin(-1)
    rel = pts - center[j]
    s, d = s_samp[j] + (rel * tangent[j]).sum(-1), (rel * normal[j]).sum(-1)
    return (s, d, j) if return_idx else (s, d)


def corr_s(corr: CorridorData):
    seg = (corr.center[1:] - corr.center[:-1]).norm(dim=-1)
    s = torch.cat([torch.zeros(1, dtype=seg.dtype, device=seg.device), seg.cumsum(0)])
    j0 = corr.center.norm(dim=-1).argmin()
    return s - s[j0] - (corr.tangent[j0] * (-corr.center[j0])).sum()


def obstacle_rows(pos_lin, cs_lin, obstacles: Optional[Sequence], radius: float,
                  corr: Optional[CorridorData] = None, max_per_step: int = 2,
                  query_dist: float = 6.0, lat_clear: float = 0.3, block_buffer: float = 1.0,
                  blocking_cache: Optional[dict] = None, corr_margin: float = EGO_HALF_WIDTH + 0.1,
                  frenet_cache: Optional[dict] = None):
    B, T, _ = pos_lin.shape
    D = 2 * T
    dtype, device = pos_lin.dtype, pos_lin.device
    n_lat = T * len(CIRCLE_OFFSETS) * max_per_step
    n_rows = T + n_lat
    A = torch.zeros(B, n_rows, D, dtype=dtype, device=device)
    lo = torch.full((B, n_rows), -float("inf"), dtype=dtype, device=device)
    hi = torch.full((B, n_rows), float("inf"), dtype=dtype, device=device)
    if obstacles is None:
        return A, lo, hi, 0
    import shapely

    n_active = 0
    H = cs_lin.detach().cpu().double().numpy()
    maps = [circle_maps(pos_lin, cs_lin, off) for off in CIRCLE_OFFSETS]
    Qs = [m[2].detach().cpu().double().numpy() for m in maps]
    half_w = EGO_HALF_WIDTH + lat_clear
    if corr is not None:
        center = corr.center.to(dtype=dtype, device=device)
        tangent = corr.tangent.to(dtype=dtype, device=device)
        s_samp = corr_s(corr).to(dtype=dtype, device=device)

    for t in range(T):
        geoms = obstacles[t] if t < len(obstacles) else None
        if geoms is None or len(geoms) == 0:
            continue
        blocking = np.zeros((B, len(geoms)), dtype=bool)
        s_min_obj = np.full(len(geoms), np.inf)
        if corr is not None:
            _, d_front, jf = _frenet(Qs[-1][:, t], corr, return_idx=True)
            lo_np = corr.lo.detach().cpu().double().numpy()[jf] + corr_margin
            hi_np = corr.hi.detach().cpu().double().numpy()[jf] - corr_margin
            d_front = np.where(lo_np < hi_np, np.clip(d_front, lo_np, hi_np), d_front)
            key = ("frenet", t)
            if frenet_cache is not None and key in frenet_cache:
                s_min_obj, d_lo, d_hi = frenet_cache[key]
            else:
                d_lo = np.zeros(len(geoms)); d_hi = np.zeros(len(geoms))
                for gi, g in enumerate(geoms):
                    s_v, d_v = _frenet(shapely.get_coordinates(g), corr)
                    s_min_obj[gi], d_lo[gi], d_hi[gi] = s_v.min(), d_v.min(), d_v.max()
                if frenet_cache is not None:
                    frenet_cache[key] = (s_min_obj, d_lo, d_hi)
            ahead = s_min_obj > 0.0
            blocking = ahead[None] & (d_lo[None] < d_front[:, None] + half_w) & (d_hi[None] > d_front[:, None] - half_w)
            if blocking_cache is not None:
                blocking = blocking_cache.setdefault(t, blocking)
            P = pos_lin
            for b in range(B):
                if not blocking[b].any():
                    continue
                s_lim = s_min_obj[blocking[b]].min() - FRONT_FROM_REAR_AXLE - block_buffer
                j = (P[b, t][None] - center).norm(dim=-1).argmin()
                tj, cj = tangent[j], center[j]
                A[b, t, 2 * t:2 * t + 2] = tj
                hi[b, t] = s_lim - float(s_samp[j]) + float(tj @ cj)
                n_active += 1
        for ci, off in enumerate(CIRCLE_OFFSETS):
            L, const, _ = maps[ci]
            base = T + (t * len(CIRCLE_OFFSETS) + ci) * max_per_step
            for b in range(B):
                q = Qs[ci][b, t]
                pt = shapely.points(q)
                dist = shapely.distance(geoms, pt)
                cand = np.where((dist < query_dist) & ~blocking[b])[0]
                if cand.size == 0:
                    continue
                cand = cand[np.argsort(dist[cand])][:max_per_step]
                for m, gi in enumerate(cand):
                    g = geoms[gi]
                    if dist[gi] > 1e-6:
                        qq = np.asarray(shapely.shortest_line(g, pt).coords[0])
                        nvec = q - qq
                        if np.dot(qq - q, H[b, t]) < -0.5:
                            continue
                    else:
                        nvec = q - np.asarray(g.centroid.coords[0])
                        qq = np.asarray(shapely.shortest_line(g.exterior, pt).coords[0])
                    nn = np.linalg.norm(nvec)
                    if nn < 1e-6:
                        continue
                    nv = torch.as_tensor(nvec / nn, dtype=dtype, device=device)
                    A[b, base + m] = nv @ L[b, t]
                    lo[b, base + m] = float(nv @ torch.as_tensor(qq, dtype=dtype, device=device)) \
                        + radius - float(nv @ const[b, t])
                    n_active += 1
    return A, lo, hi, n_active


def assemble(kin, corr=None, obs=None, acc=None) -> Rows:
    G, c, r = kin
    As, los, his = [], [], []
    for part in (acc, corr, obs):
        if part is not None:
            As.append(part[0]); los.append(part[1]); his.append(part[2])
    B, D = G.shape[0], G.shape[-1]
    if As:
        A = torch.cat(As, 1); lo = torch.cat(los, 1); hi = torch.cat(his, 1)
    else:
        A = torch.zeros(B, 0, D, dtype=G.dtype, device=G.device)
        lo = torch.zeros(B, 0, dtype=G.dtype, device=G.device); hi = lo.clone()
    return Rows(G, c, r, A, lo, hi)


def physical_violations(z, kin, corr=None, obs=None, dt=0.1, acc=None):
    G, c, r = kin
    g = torch.einsum("bnkd,bd->bnk", G, z) - c
    ex = (g.norm(dim=-1) - r).clamp_min(0)
    if acc is None:
        T = G.shape[1] // 2
        out = {"speed": (ex[:, :T].amax(1) / dt), "accel": (ex[:, T:].amax(1) / dt / dt)}
    else:
        s = torch.einsum("bnd,bd->bn", acc[0], z)
        va = torch.maximum((acc[1] - s).clamp_min(0), (s - acc[2]).clamp_min(0))
        out = {"speed": ex.amax(1) / dt, "accel": va.amax(1) / dt / dt}
    for name, part in (("corridor", corr), ("obstacle", obs)):
        if part is None or part[0].shape[1] == 0:
            out[name] = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
            continue
        s = torch.einsum("bnd,bd->bn", part[0], z)
        v = torch.maximum((part[1] - s).clamp_min(0), (s - part[2]).clamp_min(0))
        out[name] = torch.nan_to_num(v, nan=0.0, posinf=0.0).amax(1)
    return out
