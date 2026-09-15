# -*- coding: utf-8 -*-
# data/pedestrian.py
"""ETH/UCY pedestrian trajectories as points with kinematic constraints (Exp-03).

One sample is a whole 20-frame trajectory (8 observed + 12 future, 0.4 s) in the
agent frame used by MoFlow / EigenTrajectory: translated so that the last
observed frame (index 7) is the origin and rotated so that the frame-6 -> 7
heading is the +x axis. Frame 7 is identically zero and is dropped, so a point
is p in R^38 = frames (0..6, 8..19) x (x, y), in metres.

Constraints follow data/NOTICE.md. All of them are per-agent kinematics, so
the point is self-contained (the anchors are inside p):

    speed_k : ||p_k - p_{k-1}|| / dt - v_max <= 0        k = 1..19
    acc_k   : ||p_k - 2 p_{k-1} + p_{k-2}|| / dt^2 - a_max <= 0   k = 2..19
    box     : max |p| - R <= 0

h(p) reports one value per family (max over frames). v_max / a_max are the
q99.5 of the per-trajectory maximum on the train split; R is the train
bounding box plus a margin. P(p) = exact Euclidean projection onto the
kinematic ∩ box set (convex, hence 1-Lipschitz, and P(x) = x for the feasible
>= 99% of ground-truth trajectories). project_feasible shrinks the radii by
`buffer` and seals the result with a closed-form pass, so h <= -buffer holds
after float32 round-off. The box is enforced by isotropic scaling about the
origin, which cannot break the kinematic rows.
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from data.base import BaseConstraint, DataBundle, register_dataset
from utils.paths import ROOT

N_OBS = 8
N_FRAMES = 20
ANCHOR = N_OBS - 1
VAR_FRAMES = tuple(i for i in range(N_FRAMES) if i != ANCHOR)
DIM = 2 * len(VAR_FRAMES)
SUBSETS = ("eth", "hotel", "univ", "zara1", "zara2")


# =====================================================================
# 1. Metadata and dataset
# =====================================================================

@dataclass
class PedestrianMeta:
    subset: str
    dt: float
    n_obs: int
    n_fut: int
    v_max: float
    a_max: float
    R: float
    margin: float
    quantile: float
    rotate_time_frame: int
    n_train: int
    n_eval: int
    seed: int
    mean: list
    std: list


class PointDataset(Dataset):
    def __init__(self, points: torch.Tensor):
        self.points = points

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.points[idx]


# =====================================================================
# 2. Frame conversion
# =====================================================================

def to_agent_frame(traj: np.ndarray, rotate_time_frame: int = 6) -> np.ndarray:
    """[N, 20, 2] absolute -> [N, 20, 2] agent frame (same convention as MoFlow rotate_traj)."""
    t = np.asarray(traj, dtype=np.float64)
    rel = t - t[:, ANCHOR : ANCHOR + 1]
    d = rel[:, rotate_time_frame]
    theta = np.arctan(d[:, 1] / (d[:, 0] + 1e-5))
    theta = np.where(d[:, 0] < 0, theta + math.pi, theta)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], -2)
    return np.einsum("nij,nfj->nfi", rot, rel)


def seq_to_point(seq) -> Any:
    """[..., 20, 2] -> [..., 38] (drop the zero anchor frame)."""
    idx = list(VAR_FRAMES)
    return seq[..., idx, :].reshape(*seq.shape[:-2], DIM)


def point_to_seq(p) -> Any:
    """[..., 38] -> [..., 20, 2] with the anchor frame inserted as zeros."""
    is_t = isinstance(p, torch.Tensor)
    v = p.reshape(*p.shape[:-1], len(VAR_FRAMES), 2)
    zero = v[..., :1, :] * 0.0
    parts = [v[..., :ANCHOR, :], zero, v[..., ANCHOR:, :]]
    return torch.cat(parts, dim=-2) if is_t else np.concatenate(parts, axis=-2)


# =====================================================================
# 3. Small numpy/torch helpers
# =====================================================================

def _is_t(x) -> bool:
    return isinstance(x, torch.Tensor)


def _norm(x, eps: float = 0.0):
    s = (x * x).sum(-1)
    return torch.sqrt(s + eps) if _is_t(x) else np.sqrt(s + eps)


def _amax(x):
    return x.max(-1)[0] if _is_t(x) else x.max(-1)


def _clip_max(x, hi):
    return torch.clamp(x, max=hi) if _is_t(x) else np.minimum(x, hi)


def _relu(x):
    return torch.clamp(x, min=0.0) if _is_t(x) else np.maximum(x, 0.0)


def _where(c, a, b):
    return torch.where(c, a, b) if _is_t(a) else np.where(c, a, b)


def _stack(xs, axis):
    return torch.stack(xs, dim=axis) if _is_t(xs[0]) else np.stack(xs, axis=axis)


def _cat(xs, axis):
    return torch.cat(xs, dim=axis) if _is_t(xs[0]) else np.concatenate(xs, axis=axis)


def _like(a: np.ndarray, ref):
    return torch.as_tensor(a, device=ref.device, dtype=ref.dtype) if _is_t(ref) else np.asarray(a, dtype=ref.dtype)


def _ball(v, r):
    return v * _clip_max(r / (_norm(v) + 1e-12)[..., None], 1.0)


def _two_disc(q, center, rv, ra):
    """Project q onto {||d|| <= rv} ∩ {||d - center|| <= ra} in 2-D (closed form)."""
    pa = _ball(q, rv)
    in_b = _norm(pa - center) <= ra + 1e-9
    pb = center + _ball(q - center, ra)
    in_a = _norm(pb) <= rv + 1e-9
    dist = _norm(center) + 1e-12
    a = (rv * rv - ra * ra + dist * dist) / (2.0 * dist)
    hh = _relu(rv * rv - a * a) ** 0.5
    base = center * (a / dist)[..., None]
    perp = _stack([-center[..., 1], center[..., 0]], -1) / dist[..., None]
    i1, i2 = base + perp * hh[..., None], base - perp * hh[..., None]
    inter = _where((_norm(q - i1) <= _norm(q - i2))[..., None], i1, i2)
    out = _where(in_a[..., None], pb, inter)
    return _where(in_b[..., None], pa, out)


# =====================================================================
# 4. PedestrianConstraint
# =====================================================================

class PedestrianConstraint(BaseConstraint):
    def __init__(self, meta: PedestrianMeta, rho: float = 10.0, relax: float = 1.8):
        self.meta = meta
        self.rho = float(rho)
        self.relax = float(relax)
        rows, radii = [], []
        for k in range(1, N_FRAMES):
            r = np.zeros(N_FRAMES)
            r[k], r[k - 1] = 1.0, -1.0
            rows.append(r)
            radii.append(meta.v_max * meta.dt)
        for k in range(2, N_FRAMES):
            r = np.zeros(N_FRAMES)
            r[k], r[k - 1], r[k - 2] = 1.0, -2.0, 1.0
            rows.append(r)
            radii.append(meta.a_max * meta.dt * meta.dt)
        self.stencil = np.stack(rows)                       # [37, 20]
        self.A = self.stencil[:, list(VAR_FRAMES)]           # [37, 19]
        self.radii = np.asarray(radii)
        self.n_speed = N_FRAMES - 1
        nv = len(VAR_FRAMES)
        self.G = np.linalg.inv(np.eye(nv) + self.rho * (self.A.T @ self.A + np.eye(nv)))
        self._cache: dict = {}

    # ------------------------------------------------------------ helpers
    def _mats(self, ref):
        key = (type(ref), getattr(ref, "device", None), getattr(ref, "dtype", None))
        if key not in self._cache:
            self._cache[key] = tuple(_like(m, ref) for m in (self.A, self.G, self.radii))
        return self._cache[key]

    def _frames(self, p):
        return p.reshape(*p.shape[:-1], len(VAR_FRAMES), 2)

    def _residuals(self, p):
        A, _, _ = self._mats(p)
        x = self._frames(p)
        return torch.einsum("jf,...fd->...jd", A, x) if _is_t(p) else np.einsum("jf,...fd->...jd", A, x)

    def _terms(self, p, eps: float = 1e-12):
        """speed [..., 19] (m/s) and acc [..., 18] (m/s^2)."""
        n = _norm(self._residuals(p), eps)
        dt = self.meta.dt
        return n[..., : self.n_speed] / dt, n[..., self.n_speed :] / (dt * dt)

    # ------------------------------------------------------ NOTICE 2.1 h
    def h(self, p):
        speed, acc = self._terms(p)
        box = _amax(abs(p)) - self.meta.R
        return {"speed": _amax(speed) - self.meta.v_max, "acc": _amax(acc) - self.meta.a_max, "box": box}

    def h_per_frame(self, p):
        speed, acc = self._terms(p)
        return {"speed": speed - self.meta.v_max, "acc": acc - self.meta.a_max}

    # ---------------------------------------------------- NOTICE 2.2 cost
    def cost(self, p):
        speed, acc = self._terms(p)
        c = 0.5 * (_relu(speed - self.meta.v_max) ** 2).sum(-1)
        c = c + 0.5 * (_relu(acc - self.meta.a_max) ** 2).sum(-1)
        return c + 0.5 * (_relu(abs(p) - self.meta.R) ** 2).sum(-1)

    # ------------------------------------------------ projections (2.3/2.4)
    def _admm(self, p, v_max, a_max, R, n_iters: int = 300, tol: float = 1e-6):
        A, G, _ = self._mats(p)
        rad = _like(np.concatenate([np.full(self.n_speed, v_max * self.meta.dt),
                                    np.full(N_FRAMES - 2, a_max * self.meta.dt ** 2)]), p)[..., None]
        x = self._frames(p)
        AT = A.transpose(0, 1) if _is_t(p) else A.T
        apply = (lambda M, v: torch.einsum("jf,...fd->...jd", M, v)) if _is_t(p) else (lambda M, v: np.einsum("jf,...fd->...jd", M, v))
        zk = _ball(apply(A, x), rad)
        zb = _where(x > R, x * 0 + R, _where(x < -R, x * 0 - R, x))
        uk, ub = zk * 0.0, zb * 0.0
        cur = x
        for _ in range(n_iters):
            rhs = x + self.rho * (apply(AT, zk - uk) + (zb - ub))
            cur = apply(G, rhs)
            kin = apply(A, cur)
            kin_r = self.relax * kin + (1.0 - self.relax) * zk
            box_r = self.relax * cur + (1.0 - self.relax) * zb
            zk = _ball(kin_r + uk, rad)
            vb = box_r + ub
            zb = _where(vb > R, vb * 0 + R, _where(vb < -R, vb * 0 - R, vb))
            uk = uk + kin_r - zk
            ub = ub + box_r - zb
            res = max(float(_amax(_norm(kin - zk)).max()), float(abs(cur - zb).max()))
            if res <= tol:
                break
        return cur.reshape(*p.shape[:-1], DIM)

    def _seal(self, p, v_max, a_max, R):
        """Closed-form pass that returns a point with h <= 0 for the given radii."""
        rv, ra = v_max * self.meta.dt, a_max * self.meta.dt ** 2
        seq = point_to_seq(p)
        # backward from the anchor: frames 6, 5, ..., 0
        out_b = []
        last = seq[..., ANCHOR, :]
        e_prev = None
        e_first = None
        for j in range(ANCHOR - 1, -1, -1):
            cand = seq[..., j, :] - last
            e = _ball(cand, rv) if e_prev is None else _two_disc(cand, e_prev, rv, ra)
            last = last + e
            out_b.append(last)
            e_prev = e
            if e_first is None:
                e_first = e
        d_prev = -e_first if e_first is not None else None          # d_7 = p_7 - p_6 = -(p_6 - p_7)
        # forward from the anchor: frames 8, ..., 19
        out_f = []
        last = seq[..., ANCHOR, :]
        for k in range(ANCHOR + 1, N_FRAMES):
            cand = seq[..., k, :] - last
            d = _ball(cand, rv) if d_prev is None else _two_disc(cand, d_prev, rv, ra)
            last = last + d
            out_f.append(last)
            d_prev = d
        frames = _stack(list(reversed(out_b)) + out_f, -2)          # [..., 19, 2]
        flat = frames.reshape(*p.shape[:-1], DIM)
        # box by isotropic scaling about the anchor (keeps kinematic rows feasible)
        m = _amax(abs(flat))
        s = _clip_max(R / (m + 1e-12), 1.0)
        return flat * s[..., None]

    def _project(self, p, v, a, R):
        if _is_t(p):
            with torch.no_grad():
                out = self._admm(p.detach(), v, a, R)
                return self._seal(out, v, a, R)
        pt = np.asarray(p, dtype=np.float64)
        return self._seal(self._admm(pt, v, a, R), v, a, R)

    def project_physical(self, p):
        return self._project(p, self.meta.v_max, self.meta.a_max, self.meta.R)

    def project_feasible(self, p, buffer: float = 1e-4):
        b = buffer * 1.001 + 1e-9                       # slack for round-off in the closed-form seal
        return self._project(p, self.meta.v_max - b, self.meta.a_max - b, self.meta.R - b)

    def project(self, p: np.ndarray) -> np.ndarray:
        return self.project_physical(np.asarray(p, dtype=np.float64)).astype(np.float32)

    # --------------------------------------------------- NOTICE 2.5 lipschitz
    def estimate_lipschitz(self, p, eps: float = 1e-4):
        # P is the Euclidean projection onto a convex set: non-expansive, L = 1 exactly.
        if _is_t(p):
            return torch.ones(p.shape[0], device=p.device, dtype=p.dtype)
        return np.ones(p.shape[0], dtype=np.float64)

    # ------------------------------------------------------ NOTICE 2.6 energy
    def energy(self, p: torch.Tensor, w_tube: float = 1.0, w_core: float = 1.0, w_box: float = 1.0,
               w_cost: float = 0.0, slack: float = 0.01) -> torch.Tensor:
        """E = w_tube/2 * dist(p, S_slack)^2 (+ w_cost * C). S_slack: limits shrunk by `slack`.

        Pedestrian counterpart of the Swiss-roll hinge (d - tau)_+^2, which is the squared distance
        to the tube. Zero inside, convex, 1-Lipschitz gradient (Moreau envelope), so GuideFlow's
        refinement step p <- p - eta * grad E is stable for eta * w_tube < 2. A hinge on per-frame
        speed / acceleration in m/s units has curvature ~1e2-1e3 at dt = 0.4 s and diverges with
        eta_max = 0.5. w_core / w_box are absorbed: the set couples speed, acceleration and box.
        """
        with torch.no_grad():
            target = self.project_feasible(p.detach(), buffer=slack)
        e = 0.5 * w_tube * ((p - target) ** 2).sum(-1)
        if w_cost > 0.0:
            e = e + w_cost * self.cost(p)
        return e

    def energy_grad(self, p: np.ndarray, w_tube: float = 1.0, w_core: float = 1.0, w_box: float = 1.0,
                    w_cost: float = 0.0, slack: float = 0.01) -> np.ndarray:
        """grad E = w_tube * (p - P_slack(p)) (+ w_cost * grad C, by autograd)."""
        p = np.asarray(p, dtype=np.float64)
        g = w_tube * (p - self.project_feasible(p, buffer=slack))
        if w_cost > 0.0:
            pt = torch.from_numpy(p).requires_grad_(True)
            (gc,) = torch.autograd.grad(self.cost(pt).sum(), pt)
            g = g + w_cost * gc.numpy()
        return g

    # ---------------------------------------------- NOTICE 2.7 progress/command
    def progress(self, p):
        """Heading of the final future point in the agent frame, mapped to [0, 1]."""
        seq = point_to_seq(p)
        end = seq[..., -1, :]
        ang = torch.atan2(end[..., 1], end[..., 0]) if _is_t(p) else np.arctan2(end[..., 1], end[..., 0])
        return (ang + math.pi) / (2.0 * math.pi)

    # ------------------------------------------------------- NOTICE 2.8 FMBF
    def get_fmbf(self, *, radius_eps: float = 1.0e-2, tube_margin: float = 0.0, box_temperature: float = 1.0e-3,
                 softmin_temperature: float = 0.05) -> "PedestrianFMBF":
        return PedestrianFMBF(self, radius_eps=radius_eps, margin=tube_margin, box_temperature=box_temperature,
                              softmin_temperature=softmin_temperature)


@dataclass(frozen=True)
class TerminalFilterStats:
    filtered: int


class PedestrianFMBF:
    """Three C^1 barriers h_bar >= 0 (speed, acc, box) for SafeFlow's CFMBF-QP.

    Per-frame terms are aggregated with a soft-min (log-sum-exp) so the QP sees
    three constraints instead of 41 (the active-set solver enumerates 2^K).
    """

    names = ("speed", "acc", "box")
    reference_names = ("speed", "acc", "box")

    def __init__(self, constraint: PedestrianConstraint, *, radius_eps: float, margin: float,
                 box_temperature: float, softmin_temperature: float) -> None:
        self.c = constraint
        self.eps2 = float(radius_eps) ** 2
        self.margin = float(margin)
        self.box_T = float(box_temperature)
        self.T = float(softmin_temperature)

    def _values(self, p: torch.Tensor) -> torch.Tensor:
        m = self.c.meta
        n = torch.sqrt((self.c._residuals(p) ** 2).sum(-1) + self.eps2)
        speed = (m.v_max - self.margin) - n[..., : self.c.n_speed] / m.dt
        acc = (m.a_max - self.margin) - n[..., self.c.n_speed :] / (m.dt * m.dt)
        h_speed = -self.T * torch.logsumexp(-speed / self.T, dim=-1)
        h_acc = -self.T * torch.logsumexp(-acc / self.T, dim=-1)
        faces = torch.cat([p, -p], dim=-1)
        h_box = m.R - self.box_T * torch.logsumexp(faces / self.box_T, dim=-1)
        return torch.stack([h_speed, h_acc, h_box], dim=-1)

    def values_and_gradients(self, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = p.shape[:-1]
        flat = p.detach().reshape(-1, DIM).requires_grad_(True)
        with torch.enable_grad():
            vals = self._values(flat)
            grads = torch.stack([torch.autograd.grad(vals[:, i].sum(), flat, retain_graph=i < 2)[0] for i in range(3)], dim=1)
        return vals.detach().reshape(*batch, 3), grads.detach().reshape(*batch, 3, DIM)

    def terminal_filter(self, p: np.ndarray, *, max_iter: int = 100, ftol: float = 1e-7,
                        constraint_tol: float = 1e-7) -> tuple[np.ndarray, TerminalFilterStats]:
        points = np.asarray(p, dtype=np.float64)
        vals, _ = self.values_and_gradients(torch.as_tensor(points, dtype=torch.float64))
        bad = (vals.numpy() < -float(constraint_tol)).any(axis=-1)
        out = points.copy()
        if bad.any():
            out[bad] = self.c.project_feasible(points[bad].astype(np.float32), buffer=max(self.margin, 1e-4)).astype(np.float64)
        ref = self.c.h(out)
        if not all(bool((ref[k] <= constraint_tol).all()) for k in self.reference_names):
            raise RuntimeError("pedestrian terminal filter returned an infeasible batch")
        return out, TerminalFilterStats(filtered=int(bad.sum()))


# =====================================================================
# 5. Dataset builder
# =====================================================================

def _load_split(cache_dir: Path, subset: str, split: str) -> np.ndarray:
    with (cache_dir / subset / f"{subset}_{split}.pkl").open("rb") as f:
        return np.asarray(pickle.load(f)["traj"], dtype=np.float64)


def _kinematic_limits(points: np.ndarray, dt: float, quantile: float) -> tuple[float, float]:
    seq = point_to_seq(points)
    d = np.diff(seq, axis=1)
    speed = np.linalg.norm(d, axis=-1) / dt
    acc = np.linalg.norm(np.diff(d, axis=1), axis=-1) / (dt * dt)
    return float(np.percentile(speed.max(1), quantile)), float(np.percentile(acc.max(1), quantile))


def bundle_from_arrays(train_raw: np.ndarray, eval_raw: np.ndarray, meta: PedestrianMeta) -> DataBundle:
    mean = np.asarray(meta.mean, dtype=np.float32)
    std = np.asarray(meta.std, dtype=np.float32)
    train_z = (train_raw - mean) / std
    eval_z = (eval_raw - mean) / std
    return DataBundle(
        train=PointDataset(torch.from_numpy(train_z.astype(np.float32))),
        train_raw=np.asarray(train_raw, dtype=np.float32),
        eval_raw=np.asarray(eval_raw, dtype=np.float32),
        eval_z=torch.from_numpy(eval_z.astype(np.float32)),
        mean=torch.from_numpy(mean),
        std=torch.from_numpy(std),
        constraint=PedestrianConstraint(meta),
        meta=meta,
        meta_dict=asdict(meta),
    )


def _resolve(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(str(path_str))
    return path if path.is_absolute() else ROOT / path


@register_dataset("pedestrian")
def build_pedestrian(cfg: DictConfig) -> DataBundle:
    subset = str(cfg.data.get("subset", "zara2"))
    if subset not in SUBSETS:
        raise ValueError(f"unknown ETH/UCY subset {subset!r}")
    src = _resolve(cfg.data.get("source_dir", "datasets/pedestrian/default"))
    cache = _resolve(cfg.data.get("cache_dir", "datasets/pedestrian/points"))
    cache_sub = cache / subset if cache is not None else None
    regenerate = bool(cfg.data.get("regenerate", False))
    if cache_sub is not None and (cache_sub / "meta.json").is_file() and not regenerate:
        train_raw = np.load(cache_sub / "train.npy")
        eval_raw = np.load(cache_sub / "eval.npy")
        meta = PedestrianMeta(**json.loads((cache_sub / "meta.json").read_text()))
        return bundle_from_arrays(train_raw, eval_raw, meta)

    dt = float(cfg.data.get("dt", 0.4))
    rtf = int(cfg.data.get("rotate_time_frame", 6))
    quantile = float(cfg.data.get("quantile", 99.5))
    margin = float(cfg.data.get("margin", 0.5))
    n_eval = cfg.data.get("n_eval", None)
    seed = int(cfg.get("seed", 0))

    train_raw = seq_to_point(to_agent_frame(_load_split(src, subset, "train"), rtf))
    eval_raw = seq_to_point(to_agent_frame(_load_split(src, subset, "test"), rtf))
    if n_eval is not None and int(n_eval) < eval_raw.shape[0]:
        rng = np.random.default_rng(seed)
        eval_raw = eval_raw[np.sort(rng.choice(eval_raw.shape[0], int(n_eval), replace=False))]
    v_max, a_max = _kinematic_limits(train_raw, dt, quantile)
    R = float(np.abs(train_raw).max()) + margin
    meta = PedestrianMeta(
        subset=subset, dt=dt, n_obs=N_OBS, n_fut=N_FRAMES - N_OBS, v_max=v_max, a_max=a_max, R=R,
        margin=margin, quantile=quantile, rotate_time_frame=rtf, n_train=int(train_raw.shape[0]),
        n_eval=int(eval_raw.shape[0]), seed=seed,
        mean=[float(x) for x in train_raw.mean(axis=0)],
        std=[float(x) for x in train_raw.std(axis=0).clip(min=1e-6)],
    )
    if cache_sub is not None:
        cache_sub.mkdir(parents=True, exist_ok=True)
        np.save(cache_sub / "train.npy", train_raw.astype(np.float32))
        np.save(cache_sub / "eval.npy", eval_raw.astype(np.float32))
        (cache_sub / "meta.json").write_text(json.dumps(asdict(meta), indent=2))
    return bundle_from_arrays(train_raw.astype(np.float32), eval_raw.astype(np.float32), meta)
