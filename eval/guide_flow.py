# -*- coding: utf-8 -*-
# eval/guide_flow.py
"""Training-free GuideFlow sampling: CVF + CF + RFE on the frozen FlowMatch v_theta.

Paper mapping (GuideFlow, arXiv:2511.18729):
  * trajectory anchor set V_a  -> farthest point sampling over feasible train points
  * CFG  (Eq. 12, 13)          -> v_guide = (1 - gamma) v(x,t) + gamma v(x,t,c), opt-in
  * CVF  (Eq. 14)              -> v <- v - 2 lambda (v . v^c) / ||v^c||^2 v^c
  * CF   (Eq. 16)              -> re-anchor the flow state at k_c onto the path to x_1^c
  * RFE  (Eq. 5, 8, 17)        -> x <- x - eta(t) grad E, E from the analytic constraint
"""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig

from constraints.swiss_roll import SwissRollConstraint
from data.swiss_roll import build_swiss_roll, nearest_u
from eval._backbone import load_frozen_velocity

_H_NAMES = ("tube", "core", "box")
_EPS = 1e-12


def build_anchor_vocabulary(
    train_raw: np.ndarray,
    constraint: SwissRollConstraint,
    n_anchors: int,
    seed: int,
) -> np.ndarray:
    p = np.asarray(train_raw, dtype=np.float64)
    h = constraint.h(p)
    feasible = np.stack([h[name] for name in _H_NAMES], axis=-1).max(axis=-1) <= 0.0
    pool = p[feasible] if bool(feasible.any()) else constraint.project(p).astype(np.float64)
    n = int(min(max(n_anchors, 1), pool.shape[0]))
    rng = np.random.default_rng(int(seed))
    picked = [int(rng.integers(pool.shape[0]))]
    d2 = ((pool - pool[picked[0]]) ** 2).sum(axis=-1)
    for _ in range(n - 1):
        j = int(d2.argmax())
        picked.append(j)
        d2 = np.minimum(d2, ((pool - pool[j]) ** 2).sum(axis=-1))
    return pool[np.asarray(picked, dtype=np.int64)]


def energy_grad(
    p: np.ndarray,
    constraint: SwissRollConstraint,
    w_tube: float,
    w_core: float,
    w_box: float,
    w_cost: float,
    slack: float,
) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    meta = constraint.meta
    proj = constraint.project(p).astype(np.float64)
    diff = p - proj
    d = np.linalg.norm(diff, axis=-1, keepdims=True)
    unit = diff / np.clip(d, _EPS, None)
    tube = np.maximum(d - (meta.tau - slack), 0.0)
    grad = 2.0 * w_tube * tube * unit + 2.0 * w_cost * diff

    r = np.linalg.norm(p, axis=-1, keepdims=True)
    core = np.maximum((meta.rho_min + slack) - r, 0.0)
    grad = grad - 2.0 * w_core * core * (p / np.clip(r, _EPS, None))

    box = np.maximum(np.abs(p) - (meta.R - slack), 0.0)
    grad = grad + 2.0 * w_box * box * np.sign(p)
    return grad


def energy_torch(
    p: torch.Tensor,
    constraint: SwissRollConstraint,
    w_tube: float,
    w_core: float,
    w_box: float,
    w_cost: float,
    slack: float,
) -> torch.Tensor:
    meta = constraint.meta
    proj = torch.from_numpy(constraint.project(p.detach().cpu().numpy())).to(
        device=p.device, dtype=p.dtype
    )
    diff = p - proj
    d = diff.norm(dim=-1)
    r = p.norm(dim=-1)
    tube = (d - (meta.tau - slack)).clamp_min(0.0)
    core = ((meta.rho_min + slack) - r).clamp_min(0.0)
    box = (p.abs() - (meta.R - slack)).clamp_min(0.0)
    return (
        w_tube * tube.square()
        + w_cost * d.square()
        + w_core * core.square()
        + w_box * box.square().sum(dim=-1)
    )


def energy_weights_of(cfg: DictConfig) -> tuple[float, float, float, float]:
    gf = cfg.guideflow
    return (
        float(gf.get("w_tube", 1.0)),
        float(gf.get("w_core", 1.0)),
        float(gf.get("w_box", 1.0)),
        float(gf.get("w_cost", 0.0)),
    )


def owns_backbone(cfg: DictConfig) -> bool:
    gf = cfg.guideflow
    return bool(gf.guidance.get("enabled", False)) or bool(gf.rfe_train.get("rfe_loss", False))


def energy_weight(t: float, tau_star: float, eta_max: float) -> float:
    if t < tau_star:
        return 0.0
    if t >= 1.0:
        return float(eta_max)
    return float(eta_max) * (t - tau_star) / max(1.0 - tau_star, _EPS)


def command_bins(p: np.ndarray, constraint: SwissRollConstraint, n_commands: int) -> np.ndarray:
    meta = constraint.meta
    u = nearest_u(np.asarray(p, dtype=np.float64), meta.a, meta.u_min, meta.u_max)
    frac = (u - meta.u_min) / max(meta.u_max - meta.u_min, _EPS)
    return np.clip((frac * n_commands).astype(np.int64), 0, int(n_commands) - 1)


def ego_progress(p: np.ndarray, constraint: SwissRollConstraint) -> np.ndarray:
    meta = constraint.meta
    u = nearest_u(np.asarray(p, dtype=np.float64), meta.a, meta.u_min, meta.u_max)
    return np.clip((u - meta.u_min) / max(meta.u_max - meta.u_min, _EPS), 0.0, 1.0)


def one_hot(idx: torch.Tensor, n: int, like: torch.Tensor) -> torch.Tensor:
    out = like.new_zeros(idx.shape[0], int(n))
    return out.scatter_(1, idx.reshape(-1, 1).long(), 1.0)


def intent_dim_of(cfg: DictConfig) -> int:
    g = cfg.guideflow.guidance
    if str(g.get("signal", "anchor")) == "command":
        return int(g.get("n_commands", 5))
    return int(cfg.model.get("dim", 2))


def _nearest_anchor(x1_hat: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    idx = torch.cdist(x1_hat, anchors).argmin(dim=1)
    return anchors.index_select(0, idx)


def _constrain_velocity(v: torch.Tensor, vc: torch.Tensor, lam: float) -> torch.Tensor:
    num = (v * vc).sum(dim=-1, keepdim=True)
    den = vc.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS)
    return v - 2.0 * lam * (num / den) * vc


class _GuidedVelocity:

    def __init__(self, model, gamma: float, use_reward: bool):
        self.model = model
        self.gamma = float(gamma)
        self.use_reward = use_reward

    def __call__(self, x, t, intent=None, reward=None):
        ones = x.new_ones(x.shape[0])
        zeros = x.new_zeros(x.shape[0])
        v_cond = self.model(x, t, intent, reward, ones, ones if self.use_reward else zeros)
        if self.gamma == 1.0:
            return v_cond
        v_uncond = self.model(x, t, intent, reward, zeros, zeros)
        return (1.0 - self.gamma) * v_uncond + self.gamma * v_cond


def _load_guideflow_backbone(cfg: DictConfig, device: torch.device):
    from model import build_model
    from model.cond_mlp import build_cond_model
    from train.checkpoint import load_checkpoint
    from train.ema import EMA
    from utils.paths import method_dir

    ckpt = method_dir(cfg, "guideflow") / "last.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"missing guideflow checkpoint: {ckpt}. "
            "run `python main.py guideflow --mode train` with the same "
            "guidance.enabled / rfe_train.rfe_loss settings"
        )
    use_cond = bool(cfg.guideflow.guidance.get("enabled", False))
    intent_dim = intent_dim_of(cfg) if use_cond else 0
    saved = torch.load(ckpt, map_location="cpu", weights_only=False).get("extra", {})
    saved_dim = int(saved.get("intent_dim", intent_dim))
    if saved_dim != intent_dim:
        raise ValueError(
            f"{ckpt} was trained with intent dim {saved_dim}, but the current config needs "
            f"{intent_dim} (guidance.enabled={use_cond}, signal="
            f"{cfg.guideflow.guidance.get('signal', 'anchor')!r}). retrain, or restore the config."
        )
    model = (build_cond_model(cfg, intent_dim) if use_cond else build_model(cfg)).to(device)
    ema = EMA(model, decay=float(cfg.train.ema_decay))
    load_checkpoint(ckpt, model, ema=ema, map_location=device)
    ema.copy_to(model)
    model.eval()
    return model


@torch.no_grad()
def sample(cfg: DictConfig, device: torch.device, x0: torch.Tensor) -> torch.Tensor:
    bundle = build_swiss_roll(cfg)
    constraint = SwissRollConstraint(bundle["meta"])
    gf = cfg.guideflow

    use_cvf = bool(gf.get("cvf", True))
    use_cf = bool(gf.get("cf", True))
    cf_mode = str(gf.get("cf_mode", "interp"))
    use_rfe = bool(gf.get("rfe", True))
    lam = float(gf.get("lambda_cvf", 0.1))
    t_on = float(gf.get("cvf_t_on", 0.0))
    k_c = int(gf.get("k_c", 50))
    tau_star = float(gf.get("tau_star", 0.5))
    eta_max = float(gf.get("eta_max", 0.5))
    n_refine = int(gf.get("n_refine", 10))
    slack = float(gf.get("slack", 0.01))
    guidance = gf.get("guidance", {})
    use_cfg = bool(guidance.get("enabled", False))
    signal = str(guidance.get("signal", "anchor"))
    n_commands = int(guidance.get("n_commands", 5))
    gamma = float(guidance.get("gamma", 1.0))
    use_reward = bool(guidance.get("reward", True))
    ep_value = float(guidance.get("ep", 1.0))
    weights = energy_weights_of(cfg)

    mean_t = bundle["mean"].to(device=device, dtype=x0.dtype)
    std_t = bundle["std"].to(device=device, dtype=x0.dtype)
    mean_np = bundle["mean"].detach().cpu().numpy().astype(np.float64)
    std_np = bundle["std"].detach().cpu().numpy().astype(np.float64)

    anchors_p = build_anchor_vocabulary(
        bundle["train_raw"],
        constraint,
        int(gf.get("n_anchors", 256)),
        int(cfg.seed),
    )
    anchors_z = torch.from_numpy(((anchors_p - mean_np) / std_np).astype(np.float32)).to(
        device=device, dtype=x0.dtype
    )
    anchor_cmd = torch.from_numpy(command_bins(anchors_p, constraint, n_commands)).to(device)

    if owns_backbone(cfg):
        own_model = _load_guideflow_backbone(cfg, device)
    if use_cfg:
        guided = _GuidedVelocity(own_model, gamma, use_reward)
        reward_vec = x0.new_full((x0.shape[0], 1), ep_value) if use_reward else None

        def velocity(x, t_tensor, x1_c=None):
            if x1_c is None:
                x1_c = _nearest_anchor(x, anchors_z)
            if signal == "command":
                idx = torch.cdist(x1_c, anchors_z).argmin(dim=1)
                intent = one_hot(anchor_cmd.index_select(0, idx), n_commands, x)
            else:
                intent = x1_c
            return guided(x, t_tensor, intent, reward_vec)
    elif owns_backbone(cfg):

        def velocity(x, t_tensor, x1_c=None):
            return own_model(x, t_tensor)
    else:
        model, method = load_frozen_velocity(cfg, device)

        def velocity(x, t_tensor, x1_c=None):
            return method.velocity(model, x, t_tensor)

    def refine(x: torch.Tensor, eta: float) -> torch.Tensor:
        p = (x * std_t + mean_t).detach().cpu().numpy().astype(np.float64)
        g = energy_grad(p, constraint, *weights, slack)
        z = ((p - eta * g) - mean_np) / std_np
        return torch.from_numpy(z.astype(np.float32)).to(device=x.device, dtype=x.dtype)

    def re_anchor(x: torch.Tensor, t: float) -> torch.Tensor:
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
        v = velocity(x, t_tensor)
        x1_c = _nearest_anchor(x + (1.0 - t) * v, anchors_z)
        if cf_mode == "replace":
            return x1_c
        return (1.0 - t) * x0 + t * x1_c

    steps = int(cfg.sample.n_steps)
    dt = 1.0 / steps
    x = x0
    for i in range(steps):
        t = i / steps
        t_next = (i + 1) / steps
        if use_cf and i == k_c:
            x = re_anchor(x, t)
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
        v = velocity(x, t_tensor)
        if use_cvf and t >= t_on:
            x1_c = _nearest_anchor(x + (1.0 - t) * v, anchors_z)
            v = _constrain_velocity(v, x1_c - x0, lam)
        x = x + dt * v
        if use_rfe:
            eta = energy_weight(t_next, tau_star, eta_max)
            if eta > 0.0:
                x = refine(x, eta)
    if use_rfe:
        for _ in range(n_refine):
            x = refine(x, eta_max)
    return x
