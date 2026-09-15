# -*- coding: utf-8 -*-
# experiments/exp_03_pedestrian/experiments/phase1_yflow/constraints.py
"""Kinematic hard constraints h(p) <= 0 with an exact Euclidean projection.

Works on numpy arrays and torch tensors with one code path. All geometry is on
the future trajectory x = (p_1, ..., p_F) with fixed anchors p_0 (last observed
position) and p_{-1} = p_0 - v_0 * dt.

    speed_k : ||p_k - p_{k-1}|| / dt - v_max <= 0
    acc_k   : ||p_k - 2 p_{k-1} + p_{k-2}|| / dt^2 - a_max <= 0

Every constraint is a Euclidean ball on a linear stencil of the sequence, so
the projection is a small second-order-cone QP solved with ADMM (the x-update
matrix is constant and pre-inverted). ``project_feasible`` is a closed-form
forward pass that always returns a feasible point and is used to seal the
terminal state after ADMM.

The set is empty when ||v_0|| > v_max + a_max * dt. ``clip_anchor`` therefore
clips the observed anchor velocity to v_max before the set is built and
reports how often that happened.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _is_torch(x) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def _cat(xs, axis):
    return torch.cat(xs, dim=axis) if _is_torch(xs[0]) else np.concatenate(xs, axis=axis)


def _stack(xs, axis):
    return torch.stack(xs, dim=axis) if _is_torch(xs[0]) else np.stack(xs, axis=axis)


def _clip_max(x, hi: float):
    return torch.clamp(x, max=hi) if _is_torch(x) else np.minimum(x, hi)


def _where(cond, a, b):
    return torch.where(cond, a, b) if _is_torch(a) else np.where(cond, a, b)


def _sqrt(x):
    return torch.sqrt(x) if _is_torch(x) else np.sqrt(x)


def _norm_last(x):
    return ((x * x).sum(-1)) ** 0.5


def _ball(v, radius: float, eps: float = 1e-12):
    n = _norm_last(v)[..., None]
    return v * _clip_max(radius / (n + eps), 1.0)


def _as_like(a: np.ndarray, ref):
    if _is_torch(ref):
        return torch.as_tensor(a, device=ref.device, dtype=ref.dtype)
    return np.asarray(a, dtype=ref.dtype)


@dataclass(frozen=True)
class KinematicLimits:
    v_max: float
    a_max: float
    dt: float

    @property
    def speed_radius(self) -> float:
        return self.v_max * self.dt

    @property
    def acc_radius(self) -> float:
        return self.a_max * self.dt * self.dt


class KinematicConstraint:
    def __init__(self, limits: KinematicLimits, n_future: int = 12, rho: float = 10.0, relax: float = 1.8):
        self.limits = limits
        self.n_future = int(n_future)
        self.rho = float(rho)
        self.relax = float(relax)
        F = self.n_future
        # rows over the extended sequence index (p_{-1}, p_0, p_1, ..., p_F)
        rows = []
        radii = []
        for k in range(1, F + 1):
            r = np.zeros(F + 2)
            r[k + 1], r[k] = 1.0, -1.0
            rows.append(r)
            radii.append(limits.speed_radius)
        for k in range(1, F + 1):
            r = np.zeros(F + 2)
            r[k + 1], r[k], r[k - 1] = 1.0, -2.0, 1.0
            rows.append(r)
            radii.append(limits.acc_radius)
        self.full_stencil = np.stack(rows)              # [2F, F+2]
        self.A = self.full_stencil[:, 2:]               # variable part [2F, F]
        self.C = self.full_stencil[:, :2]               # anchor part   [2F, 2]
        self.radii = np.asarray(radii)                  # [2F]
        self.G = np.linalg.inv(np.eye(F) + self.rho * self.A.T @ self.A)
        self._cache: dict = {}

    # ------------------------------------------------------------------ utils
    def _mats(self, ref):
        key = (type(ref), getattr(ref, "device", None), getattr(ref, "dtype", None))
        if key not in self._cache:
            self._cache[key] = tuple(_as_like(m, ref) for m in (self.A, self.C, self.G, self.radii))
        return self._cache[key]

    def _anchor_term(self, p0, p_prev, C):
        anchors = _stack([p_prev, p0], -2)                                  # [..., 2, 2]
        return torch.einsum("jk,...kd->...jd", C, anchors) if _is_torch(p0) else np.einsum("jk,...kd->...jd", C, anchors)

    def _apply_A(self, A, x):
        return torch.einsum("jf,...fd->...jd", A, x) if _is_torch(x) else np.einsum("jf,...fd->...jd", A, x)

    def clip_anchor(self, p0, p_prev):
        """Clip the anchor velocity so the feasible set is non-empty; returns (p_prev, clipped_mask)."""
        d0 = p0 - p_prev
        d0c = _ball(d0, self.limits.speed_radius)
        clipped = _norm_last(d0) > self.limits.speed_radius + 1e-12
        return p0 - d0c, clipped

    # ------------------------------------------------------------ constraints
    def residuals(self, x, p0, p_prev):
        """[..., 2F, 2] stencil values (differences in position units)."""
        A, C, _, _ = self._mats(x)
        return self._apply_A(A, x) + self._anchor_term(p0, p_prev, C)

    def h(self, x, p0, p_prev) -> dict[str, object]:
        """{"speed": [..., F] in m/s, "acc": [..., F] in m/s^2}; h <= 0 is feasible."""
        F = self.n_future
        n = _norm_last(self.residuals(x, p0, p_prev))
        dt = self.limits.dt
        return {
            "speed": n[..., :F] / dt - self.limits.v_max,
            "acc": n[..., F:] / (dt * dt) - self.limits.a_max,
        }

    def max_violation(self, x, p0, p_prev):
        hv = self.h(x, p0, p_prev)
        both = _cat([hv["speed"], hv["acc"]], -1)
        return both.max(-1)[0] if _is_torch(x) else both.max(-1)

    # ------------------------------------------------------------- projection
    def project(self, x, p0, p_prev, n_iters: int = 100, tol: float = 1e-5, warm=None):
        """ADMM Euclidean projection of x onto {h <= 0}.

        Returns (x_proj, iters, max_residual, state); pass ``state`` back as
        ``warm`` on the next call with a nearby ``x`` to warm-start the duals.
        """
        A, C, G, radii = self._mats(x)
        rad = radii[..., None]
        c = self._anchor_term(p0, p_prev, C)
        if warm is None:
            z = self._apply_A(A, x) + c
            z = z * _clip_max(rad / (_norm_last(z)[..., None] + 1e-12), 1.0)
            u = z * 0.0
        else:
            z, u = warm
        cur = x
        used = 0
        worst = None
        AT = A.T if not _is_torch(A) else A.transpose(0, 1)
        for it in range(n_iters):
            rhs = x + self.rho * self._apply_A(AT, z - c - u)
            cur = self._apply_A(G, rhs)
            Ax = self._apply_A(A, cur) + c
            Ax_r = self.relax * Ax + (1.0 - self.relax) * z
            v = Ax_r + u
            z = v * _clip_max(rad / (_norm_last(v)[..., None] + 1e-12), 1.0)
            u = u + Ax_r - z
            used = it + 1
            worst = _norm_last(Ax - z).max(-1)[0] if _is_torch(x) else _norm_last(Ax - z).max(-1)
            if float(worst.max()) <= tol:
                break
        return cur, used, worst, (z, u)

    def project_feasible(self, x, p0, p_prev, buffer: float = 0.0):
        """Forward pass: each step is the exact projection of the candidate
        displacement onto ball(0, Rv) ∩ ball(d_{k-1}, Ra). Always feasible when
        ||p_0 - p_{-1}|| <= Rv (see ``clip_anchor``).

        ``buffer`` shrinks both radii by that relative fraction so the result
        satisfies h <= -buffer * limit, which keeps it feasible after float32
        rounding (NOTICE: project_feasible must return h <= -buffer).
        """
        Rv = self.limits.speed_radius * (1.0 - buffer)
        Ra = self.limits.acc_radius * (1.0 - buffer)
        d_prev = p0 - p_prev
        p_last = p0
        out = []
        for k in range(self.n_future):
            cand = x[..., k, :] - p_last
            d = _two_disc_projection(cand, d_prev, Rv, Ra)
            p_last = p_last + d
            out.append(p_last)
            d_prev = d
        return _stack(out, -2)


def _two_disc_projection(q, center, Rv, Ra, eps: float = 1e-12):
    """Project q onto {||d|| <= Rv} ∩ {||d - center|| <= Ra} in 2-D (closed form).

    Assumes the intersection is non-empty (||center|| <= Rv + Ra).
    """
    pa = _ball(q, Rv)
    in_b = _norm_last(pa - center) <= Ra + 1e-9
    pb = center + _ball(q - center, Ra)
    in_a = _norm_last(pb) <= Rv + 1e-9
    # circle-circle intersection points (fallback when neither single projection works)
    dvec = center
    dist = _norm_last(dvec) + eps
    a = (Rv * Rv - Ra * Ra + dist * dist) / (2.0 * dist)
    hh = _sqrt(_abs_clip(Rv * Rv - a * a))
    base = dvec * (a / dist)[..., None]
    perp = _stack([-dvec[..., 1], dvec[..., 0]], -1) / dist[..., None]
    i1 = base + perp * hh[..., None]
    i2 = base - perp * hh[..., None]
    pick1 = _norm_last(q - i1) <= _norm_last(q - i2)
    inter = _where(pick1[..., None], i1, i2)
    out = _where(in_a[..., None], pb, inter)
    out = _where(in_b[..., None], pa, out)
    return out


def _abs_clip(x):
    return torch.clamp(x, min=0.0) if _is_torch(x) else np.maximum(x, 0.0)


def _halfspace(v, n, r):
    """Project v onto {v : n·v >= r} (n unit, r broadcast); r = -inf disables."""
    gap = r - (v * n).sum(-1)
    gap = gap.clamp(min=0.0) if _is_torch(v) else np.maximum(gap, 0.0)
    return v + gap[..., None] * n


def _unit(v, eps: float = 1e-9):
    return v / (_norm_last(v)[..., None] + eps)


class CollisionKinematicConstraint(KinematicConstraint):
    """Kinematic balls plus neighbour clearance ||p_k - q_{j,k}|| >= r_safe.

    Clearance is non-convex. Following the robot-arm (Exp-04) local-corridor
    idea, each clearance row is replaced by the candidate-dependent half-plane
    n_{jk}·(p_k - q_{jk}) >= r_safe with n_{jk} the unit vector from q_{jk} to
    the candidate p_k. The half-plane lies outside the disc, so any point of the
    convexified set satisfies the true clearance. The convexified problem is an
    exact Euclidean projection onto a candidate-dependent convex inner set; it is
    not the global nearest collision-free trajectory.

    Neighbour tensors: q [..., M, F, 2], mask [..., M] (True = real neighbour).
    """

    def __init__(self, limits: KinematicLimits, r_safe: float, n_future: int = 12, rho: float = 10.0, relax: float = 1.8):
        super().__init__(limits, n_future, rho=rho, relax=relax)
        self.r_safe = float(r_safe)
        self._g_cache: dict = {}

    def _g(self, M: int, ref):
        key = (M, type(ref), getattr(ref, "device", None), getattr(ref, "dtype", None))
        if key not in self._g_cache:
            F = self.n_future
            G = np.linalg.inv(np.eye(F) + self.rho * (self.A.T @ self.A + M * np.eye(F)))
            self._g_cache[key] = _as_like(G, ref)
        return self._g_cache[key]

    @staticmethod
    def normals(candidate, q):
        """Unit vectors from neighbours to the candidate: [..., M, F, 2]."""
        v = candidate[..., None, :, :] - q
        fallback = v * 0.0
        fallback[..., 0] = 1.0
        tiny = _norm_last(v)[..., None] < 1e-9
        return _unit(_where(tiny, fallback, v))

    def clearance(self, x, q, mask):
        """[..., M, F] distances; masked neighbours get +inf."""
        d = _norm_last(x[..., None, :, :] - q)
        big = d * 0.0 + 1e9
        return _where(mask[..., None], d, big)

    def min_clearance(self, x, q, mask):
        d = self.clearance(x, q, mask)
        flat = d.reshape(*d.shape[:-2], -1)
        return flat.min(-1)[0] if _is_torch(x) else flat.min(-1)

    def project_mixed(self, x, p0, p_prev, q, mask, n, n_iters: int = 100, tol: float = 1e-5, warm=None):
        """ADMM projection onto {kinematic balls} ∩ {clearance half-planes}.

        Returns (x_proj, iters, max_residual, state).
        """
        F = self.n_future
        M = int(q.shape[-3])
        A, C, _, radii = self._mats(x)
        G = self._g(M, x)
        rad = radii[..., None]
        c_kin = self._anchor_term(p0, p_prev, C)                               # [..., 2F, 2]
        q_b = q + x[..., None, :, :] * 0.0                                     # broadcast to x batch
        n_b = n + q_b * 0.0
        c_col = -q_b.reshape(*q_b.shape[:-3], M * F, 2)                        # [..., MF, 2]
        n_col = n_b.reshape(*n_b.shape[:-3], M * F, 2)
        zeros_mf = q_b[..., 0].reshape(*q_b.shape[:-3], M * F) * 0.0
        if _is_torch(q):
            m_b = mask[..., None].expand(*q_b.shape[:-1])
        else:
            m_b = np.broadcast_to(mask[..., None], q_b.shape[:-1])
        r_col = _where(m_b.reshape(*m_b.shape[:-2], M * F), zeros_mf + self.r_safe, zeros_mf - 1e9)

        def apply_all(xx):
            kin = self._apply_A(A, xx) + c_kin
            col = _cat([xx] * M, -2) + c_col if M > 0 else xx[..., :0, :]
            return kin, col

        def apply_t(zk, zc):
            AT = A.transpose(0, 1) if _is_torch(A) else A.T
            out = self._apply_A(AT, zk)
            if M > 0:
                out = out + zc.reshape(*zc.shape[:-2], M, F, 2).sum(-3)
            return out

        if warm is None:
            kin0, col0 = apply_all(x)
            zk = kin0 * _clip_max(rad / (_norm_last(kin0)[..., None] + 1e-12), 1.0)
            zc = _halfspace(col0, n_col, r_col)
            uk, uc = zk * 0.0, zc * 0.0
        else:
            zk, zc, uk, uc = warm
        cur = x
        used = 0
        worst = None
        for it in range(n_iters):
            rhs = x + self.rho * apply_t(zk - c_kin - uk, zc - c_col - uc)
            cur = self._apply_A(G, rhs)
            kin, col = apply_all(cur)
            kin_r = self.relax * kin + (1.0 - self.relax) * zk
            col_r = self.relax * col + (1.0 - self.relax) * zc
            vk, vc = kin_r + uk, col_r + uc
            zk = vk * _clip_max(rad / (_norm_last(vk)[..., None] + 1e-12), 1.0)
            zc = _halfspace(vc, n_col, r_col)
            uk = uk + kin_r - zk
            uc = uc + col_r - zc
            used = it + 1
            res = _cat([_norm_last(kin - zk), _norm_last(col - zc)], -1)
            worst = res.max(-1)[0] if _is_torch(x) else res.max(-1)
            if float(worst.max()) <= tol:
                break
        return cur, used, worst, (zk, zc, uk, uc)
