# -*- coding: utf-8 -*-
# experiments/exp_03_pedestrian/experiments/phase3_bench/methods.py
"""The five Y-Flow repository methods, specialized to pedestrian forecasting.

Each sampler reproduces the update rule of the corresponding `eval/*.py` in the
Y-Flow repository (HardFlow, YFlow, SafeFlow, UniConFlow, GuideFlow; plus the
unconstrained FlowMatch baseline) with the same hyper-parameters, but the
velocity field is the conditional MoFlow teacher (or the Phase-2 CFM) and the
constraint is the per-agent kinematic set on the 12 predicted frames, anchored
on the observed past. The repository's BaseConstraint operations map as

    h(p)              speed / acc families (max over frames)      -> ForecastConstraint.h
    C(p)              1/2 sum max(0, h_k)^2 over frames           -> .cost
    P(p)              exact Euclidean projection (ADMM + seal)    -> .project_physical
    project_feasible  projection with radii shrunk by `buffer`    -> .project_feasible
    energy / grad     w_tube/2 * dist(p, S_slack)^2               -> .energy_grad_z
    FMBF              2 soft-min C^1 barriers (speed, acc)        -> .fmbf

All operations take normalized flow states z [N, 24] (z = scale * p + offset,
isotropic) and evaluate the constraint in metres, as data/NOTICE.md requires.
Where a repository detail is Swiss-roll specific it is replaced by its
pedestrian counterpart and noted in-line.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass

import numpy as np
import torch

from experiments._layout import REPOSITORY_ROOT
from experiments.phase1_yflow.constraints import KinematicConstraint, KinematicLimits


def _load_repo_base():
    """Load the repository's data/base.py without clashing with MoFlow's `data` package."""
    spec = importlib.util.spec_from_file_location("yflow_repo_data_base", REPOSITORY_ROOT / "data" / "base.py")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_REPO_BASE = _load_repo_base()
barrier_gain = _REPO_BASE.barrier_gain
solve_composite_fmbf = _REPO_BASE.solve_composite_fmbf
_EPS = 1e-12


# =====================================================================
# Constraint on the forecast (flattened [N, 24] states)
# =====================================================================

class ForecastConstraint:
    def __init__(self, limits: KinematicLimits, scale: float, offset: float, F: int = 12,
                 fmbf_T: float = 0.05, fmbf_eps: float = 1e-2, fmbf_margin: float = 2e-3):
        self.lim = limits
        self.scale = float(scale)
        self.offset = float(offset)
        self.F = F
        self.kin_z = KinematicConstraint(KinematicLimits(limits.v_max * scale, limits.a_max * scale, limits.dt), F)
        self.T, self.eps, self.margin = fmbf_T, fmbf_eps, fmbf_margin
        self.p_prev = None                      # [N, 2] metres, anchor p_{-1}; p_0 = 0

    def bind(self, p_prev_scene: torch.Tensor, repeats: int) -> None:
        """p_prev_scene [B, 2] (clipped anchor, metres) -> per-sample anchors [B*repeats, 2]."""
        self.p_prev = p_prev_scene.repeat_interleave(repeats, dim=0)

    # ---------------------------------------------------------- mappings
    def to_p(self, z: torch.Tensor) -> torch.Tensor:
        return ((z - self.offset) / self.scale).reshape(z.shape[0], self.F, 2)

    def to_z(self, p: torch.Tensor) -> torch.Tensor:
        return (p.reshape(p.shape[0], -1) * self.scale + self.offset)

    def _terms(self, p: torch.Tensor, soft: float = 1e-12):
        """p [N, F, 2] metres -> speed [N, F] m/s, acc [N, F] m/s^2."""
        p0 = torch.zeros_like(self.p_prev)
        seq = torch.cat([self.p_prev[:, None], p0[:, None], p], dim=1)
        d = seq[:, 1:] - seq[:, :-1]                                     # [N, F+1, 2]
        speed = torch.sqrt((d[:, 1:] ** 2).sum(-1) + soft) / self.lim.dt
        acc = torch.sqrt(((d[:, 1:] - d[:, :-1]) ** 2).sum(-1) + soft) / self.lim.dt ** 2
        return speed, acc

    # ------------------------------------------------------------- h, C, E
    def h(self, z: torch.Tensor) -> torch.Tensor:
        speed, acc = self._terms(self.to_p(z))
        return torch.stack([speed.max(-1)[0] - self.lim.v_max, acc.max(-1)[0] - self.lim.a_max], dim=-1)

    def cost(self, z: torch.Tensor) -> torch.Tensor:
        speed, acc = self._terms(self.to_p(z))
        return 0.5 * ((speed - self.lim.v_max).clamp_min(0) ** 2).sum(-1) + 0.5 * ((acc - self.lim.a_max).clamp_min(0) ** 2).sum(-1)

    def energy_grad_z(self, z: torch.Tensor, weight: float, slack: float) -> torch.Tensor:
        """GuideFlow energy E = weight/2 * dist(p, S_slack)^2, the pedestrian counterpart of the
        Swiss-roll hinge (d - tau)_+^2 (= squared distance to the tube). grad_p E = weight (p - P(p)),
        1-Lipschitz, so the repository step p <- p - eta grad E is stable for eta * weight < 2.
        (A hinge on per-frame speed / acceleration in m/s units has curvature up to ~1e2-1e3 at dt = 0.4 s
        and diverges with eta_max = 0.5.) Returned in z units: grad_z = weight * (z - P_z(z))."""
        return weight * (z - self.project_feasible(z, buffer=slack))

    # ---------------------------------------------------------- projections
    def _anchors_z(self):
        p0z = torch.full_like(self.p_prev, self.offset)
        return p0z, p0z + self.scale * self.p_prev

    def project_physical(self, z: torch.Tensor, n_iters: int = 200) -> torch.Tensor:
        p0z, ppz = self._anchors_z()
        x = z.reshape(z.shape[0], self.F, 2)
        proj, _, _, _ = self.kin_z.project(x, p0z, ppz, n_iters=n_iters, tol=1e-6 * self.scale)
        return self.kin_z.project_feasible(proj, p0z, ppz, buffer=0.0).reshape(z.shape)

    def project_feasible(self, z: torch.Tensor, buffer: float = 1e-4, n_iters: int = 60) -> torch.Tensor:
        # ADMM(n_iters) + closed-form seal: always strictly feasible; near-exact projection
        # (60 iterations keep the 20-iteration PGD loops of HardFlow / YFlow tractable).
        p0z, ppz = self._anchors_z()
        x = z.detach().reshape(z.shape[0], self.F, 2)
        proj, _, _, _ = self.kin_z.project(x, p0z, ppz, n_iters=n_iters, tol=1e-6 * self.scale)
        rel = buffer / max(self.lim.v_max, 1e-6)
        return self.kin_z.project_feasible(proj, p0z, ppz, buffer=rel * 1.001 + 1e-9).reshape(z.shape)

    # ---------------------------------------------------------------- FMBF
    def fmbf(self, p_flat: torch.Tensor):
        """Smooth barriers h_bar >= 0 (speed, acc) and gradients w.r.t. p [N, 24] metres."""
        with torch.enable_grad():
            q = p_flat.detach().requires_grad_(True)
            p = q.reshape(q.shape[0], self.F, 2)
            p0 = torch.zeros_like(self.p_prev)
            seq = torch.cat([self.p_prev[:, None], p0[:, None], p], dim=1)
            d = seq[:, 1:] - seq[:, :-1]
            n_v = torch.sqrt((d[:, 1:] ** 2).sum(-1) + self.eps ** 2) / self.lim.dt
            n_a = torch.sqrt(((d[:, 1:] - d[:, :-1]) ** 2).sum(-1) + self.eps ** 2) / self.lim.dt ** 2
            hv = -self.T * torch.logsumexp(-((self.lim.v_max - self.margin) - n_v) / self.T, dim=-1)
            ha = -self.T * torch.logsumexp(-((self.lim.a_max - self.margin) - n_a) / self.T, dim=-1)
            vals = torch.stack([hv, ha], dim=-1)
            grads = torch.stack([torch.autograd.grad(vals[:, i].sum(), q, retain_graph=(i == 0))[0] for i in range(2)], dim=1)
        return vals.detach(), grads.detach()


# =====================================================================
# Repository helpers (verbatim ports)
# =====================================================================

def ptzf_reference(initial: torch.Tensor, t: float, rate: float, eps: float = 1e-5):
    """eval/unicon_flow.py: PTZF r(t) and its derivative (Eq. 36, T_pre=1)."""
    one_minus_t = max(1.0 - t, eps)
    exponent = -rate * t / one_minus_t
    reference = initial * math.exp(max(exponent, -80.0))
    derivative = -rate * reference / (one_minus_t * one_minus_t)
    return reference, derivative


def qp_guidance(rho: torch.Tensor, eta: torch.Tensor, *, slack_weight: float, ridge: float = 1e-6) -> torch.Tensor:
    """eval/unicon_flow.py: closed-form slack QP, Eqs. (49)-(51)."""
    batch, n_constraints, dim = eta.shape
    active = rho > 0
    rho_active = torch.where(active, rho, torch.zeros_like(rho))
    eta_active = eta * active.unsqueeze(-1)
    eye_m = torch.eye(n_constraints, device=eta.device, dtype=eta.dtype).expand(batch, -1, -1)
    eta_z = torch.cat((eta_active, -eye_m * active.unsqueeze(-1)), dim=-1)
    inv_diag = torch.cat((torch.ones(dim, device=eta.device, dtype=eta.dtype),
                          torch.full((n_constraints,), 1.0 / max(slack_weight, 1e-8), device=eta.device, dtype=eta.dtype)))
    weighted_eta_t = inv_diag.view(1, -1, 1) * eta_z.transpose(1, 2)
    gram = eta_z @ weighted_eta_t + ridge * eye_m
    multipliers = torch.linalg.solve(gram, rho_active.unsqueeze(-1))
    return (-(weighted_eta_t @ multipliers).squeeze(-1))[:, :dim]


def energy_weight(t: float, tau_star: float, eta_max: float) -> float:
    """eval/guide_flow.py."""
    if t < tau_star:
        return 0.0
    if t >= 1.0:
        return float(eta_max)
    return float(eta_max) * (t - tau_star) / max(1.0 - tau_star, _EPS)


def nearest_anchor(x1_hat: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """eval/guide_flow.py: _nearest_anchor."""
    return anchors.index_select(0, torch.cdist(x1_hat, anchors).argmin(dim=1))


def constrain_velocity(v: torch.Tensor, vc: torch.Tensor, lam: float) -> torch.Tensor:
    """eval/guide_flow.py: _constrain_velocity (GuideFlow Eq. 14)."""
    num = (v * vc).sum(dim=-1, keepdim=True)
    den = vc.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS)
    return v - 2.0 * lam * (num / den) * vc


def pgd_terminal(fc: ForecastConstraint, z_raw: torch.Tensor, z_phys: torch.Tensor | None, lam: float, mu: float,
                 n_iters: int, buffer: float) -> torch.Tensor:
    """eval/y_flow.py solve_terminal_pgd (mu=0 and z_phys=None give eval/hard_flow.py solve_terminal_pgd_hardflow)."""
    if z_phys is None or mu <= 0:
        z0, denom, mu = z_raw, max(lam, 1e-8), 0.0
    else:
        denom = max(lam + mu, 1e-8)
        z0 = (lam * z_raw + mu * z_phys) / denom
    z = fc.project_feasible(z0, buffer)
    lr = 1.0 / (denom + 2.0)
    for _ in range(n_iters):
        with torch.enable_grad():
            zz = z.detach().requires_grad_(True)
            loss = fc.cost(zz) + 0.5 * lam * ((zz - z_raw) ** 2).sum(-1)
            if mu > 0:
                loss = loss + 0.5 * mu * ((zz - z_phys) ** 2).sum(-1)
            (g,) = torch.autograd.grad(loss.sum(), zz)
        z = fc.project_feasible(z - lr * g, buffer)
    return z.detach()


# =====================================================================
# Samplers: one Euler grid, uniform dt = 1/steps (repository protocol)
# =====================================================================

@dataclass
class StepModel:
    """Wraps the conditional backbone: (z [N,24], t) -> (velocity, clean prediction) in z space."""
    denoiser: object
    x_data: dict
    B: int
    K: int
    A: int

    def __call__(self, z: torch.Tensor, t: float):
        y = z.reshape(self.B, self.K, self.A, -1)
        bt = torch.full((self.B,), float(t), device=z.device, dtype=torch.float)
        preds = self.denoiser.model_predictions(y, self.x_data, bt, False)
        return preds.pred_vel.reshape(z.shape), preds.pred_data.reshape(z.shape)


def sample_flowmatch(step: StepModel, fc: ForecastConstraint, x0: torch.Tensor, cfg: dict, steps: int, diag: dict):
    dt = 1.0 / steps
    x = x0
    for i in range(steps):
        v, _ = step(x, i * dt)
        x = x + dt * v
    return x


def sample_hardflow(step, fc, x0, cfg, steps, diag):
    hc = cfg.get("hardflow", {})
    t_on, lambda_oc = float(hc.get("t_on", 0.5)), float(hc.get("lambda_oc", 10.0))
    max_iter, buffer = int(hc.get("max_iter", 20)), float(hc.get("safety_buffer", 1e-4))
    dt = 1.0 / steps
    x = x0
    for i in range(steps):
        t, t_next = i * dt, (i + 1) * dt
        v, _ = step(x, t)
        bar_x = x + dt * v
        if t < t_on and i < steps - 1:
            x = bar_x
            continue
        _, bar_x1 = step(bar_x, t_next)                     # bar_x + (1 - t_next) v(bar_x, t_next)
        lam = lambda_oc * (t_next ** 2) / max(dt, 1e-8)
        z_star = pgd_terminal(fc, bar_x1, None, lam, 0.0, max_iter, buffer)
        diag["active_steps"] = diag.get("active_steps", 0) + 1
        if t_next >= 1.0 - 1e-9:
            x = z_star                                      # repo: t_next * z* + (1 - t_next) * w0 with t_next = 1
        else:
            w0 = (bar_x - t_next * bar_x1) / (1.0 - t_next)  # = bar_x - t_next * v_next
            x = t_next * z_star + (1.0 - t_next) * w0
    return x


def sample_yflow(step, fc, x0, cfg, steps, diag):
    yc = cfg.get("yflow", {})
    t_on, lambda_oc, mu = float(yc.get("t_on", 0.5)), float(yc.get("lambda_oc", 10.0)), float(yc.get("mu", 1.0))
    delta, gamma_max = float(yc.get("delta", 0.1)), float(yc.get("gamma_max", 1.0))
    max_iter, buffer = int(yc.get("max_iter", 20)), float(yc.get("safety_buffer", 1e-4))
    dt = 1.0 / steps
    x = x0
    for i in range(steps):
        t = i * dt
        _, x1_raw = step(x, t)
        terminal = i == steps - 1
        if not terminal and t < t_on:
            eta = dt / max(1.0 - t, 1e-8)
            x = (1.0 - eta) * x + eta * x1_raw
            continue
        z_phys = fc.project_physical(x1_raw)
        L_P = torch.ones(x.shape[0], device=x.device)       # exact projection onto a convex set: L_P = 1
        g_val = gamma_max * min(max((t - t_on) / max(1.0 - t_on, 1e-8), 0.0), 1.0)
        gamma = torch.where(L_P <= 1.0 + delta, torch.full_like(L_P, g_val), torch.zeros_like(L_P))
        lam = lambda_oc * (t ** 2) / max(dt, 1e-8)
        n_iters = max_iter if terminal else max(3, max_iter // 2)
        z_star = pgd_terminal(fc, x1_raw, z_phys, lam, mu, n_iters, buffer)
        if not terminal:
            z_star = torch.where((gamma == 0.0).unsqueeze(-1), x1_raw, z_star)
        diag["active_steps"] = diag.get("active_steps", 0) + 1
        eta = 1.0 if terminal else dt / max(1.0 - t, 1e-8)
        x = (1.0 - eta) * x + eta * z_star
    return x


def sample_safeflow(step, fc, x0, cfg, steps, diag):
    sc = cfg.get("safeflow", {})
    t_on = float(sc.get("t_on", 0.5))
    kw = dict(phi0=float(sc.get("phi0", 1.0)), schedule=str(sc.get("phi_schedule", "paper_piecewise")),
              gamma=float(sc.get("phi_gamma", 0.9)), omega=float(sc.get("phi_omega", 3.0)),
              terminal_eps=float(sc.get("terminal_eps", 1e-3)))
    slack_w, active_tol = float(sc.get("slack_weight", 1.0)), float(sc.get("qp_active_tol", 1e-6))
    dt = 1.0 / steps
    x = x0
    for i in range(steps):
        t = i * dt
        v, _ = step(x, t)
        if t >= t_on:
            p = fc.to_p(x).reshape(x.shape[0], -1)
            v_p = v / fc.scale
            h, grads = fc.fmbf(p)
            gain = barrier_gain(torch.tensor(t, device=x.device, dtype=h.dtype), h, **kw)
            a = (grads * v_p.unsqueeze(-2)).sum(dim=-1) + gain * h
            sol = solve_composite_fmbf(a, grads, slack_weight=slack_w, active_tol=active_tol)
            v = v + sol.correction * fc.scale
            diag["correction_m"] = diag.get("correction_m", 0.0) + float(sol.correction.norm(dim=-1).mean())
            diag["active_steps"] = diag.get("active_steps", 0) + 1
        x = x + dt * v
    if bool(sc.get("terminal_filter", {}).get("enabled", True)):
        vals, _ = fc.fmbf(fc.to_p(x).reshape(x.shape[0], -1))
        bad = (vals < -float(sc.get("terminal_filter", {}).get("constraint_tol", 1e-7))).any(dim=-1)
        diag["terminal_filter_rate"] = float(bad.float().mean())
        if bool(bad.any()):
            x = torch.where(bad.unsqueeze(-1), fc.project_feasible(x, max(fc.margin, 1e-4)), x)
    return x


def sample_uniconflow(step, fc, x0, cfg, steps, diag):
    uc = cfg.get("uniconflow", {})
    free_until, rate, gamma = float(uc.get("free_until", 0.0)), float(uc.get("ptzf_rate", 1.0)), float(uc.get("gamma", 1.0))
    slack_w, max_norm = float(uc.get("slack_weight", 100.0)), float(uc.get("max_guidance_norm", 20.0))
    terminal, buffer = bool(uc.get("terminal_refinement", True)), float(uc.get("safety_buffer", 1e-4))
    dt = 1.0 / steps
    x = x0
    hbar0 = None
    for i in range(steps):
        t = i * dt
        v, _ = step(x, t)
        with torch.enable_grad():
            z = x.detach().requires_grad_(True)
            h = fc.h(z)
            eta = torch.stack([torch.autograd.grad(h[:, j].sum(), z, retain_graph=True)[0] for j in range(h.shape[-1])], dim=1)
        if hbar0 is None:
            hbar0 = torch.clamp(h.detach(), min=0.0) + buffer
        if t < free_until:
            guidance = torch.zeros_like(v)
        else:
            local_t = (t - free_until) / max(1.0 - free_until, 1e-8)
            hbar, hbar_dot_local = ptzf_reference(hbar0, local_t, rate)
            hbar_dot = hbar_dot_local / max(1.0 - free_until, 1e-8)
            rho = torch.einsum("bmd,bd->bm", eta, v) - gamma * (hbar - h.detach()) - hbar_dot
            guidance = qp_guidance(rho, eta.detach(), slack_weight=slack_w)
            norm = torch.linalg.vector_norm(guidance, dim=-1, keepdim=True)
            guidance = guidance * torch.clamp(max_norm / norm.clamp_min(1e-12), max=1.0)
            diag["active_steps"] = diag.get("active_steps", 0) + 1
        x = x + dt * (v + guidance)
    if terminal:
        x = fc.project_feasible(x, buffer)
    return x


def sample_guideflow(step, fc, x0, cfg, steps, diag, anchors_z: torch.Tensor):
    gf = cfg.get("guideflow", {})
    use_cvf, use_cf, use_rfe = bool(gf.get("cvf", True)), bool(gf.get("cf", True)), bool(gf.get("rfe", True))
    cf_mode, lam, t_on = str(gf.get("cf_mode", "interp")), float(gf.get("lambda_cvf", 0.1)), float(gf.get("cvf_t_on", 0.0))
    k_c, tau_star, eta_max = int(gf.get("k_c", 50)), float(gf.get("tau_star", 0.5)), float(gf.get("eta_max", 0.5))
    n_refine, slack = int(gf.get("n_refine", 10)), float(gf.get("slack", 0.01))
    weight = float(gf.get("w_tube", 1.0))
    if bool(gf.get("guidance", {}).get("enabled", False)):
        raise NotImplementedError("GuideFlow CFG needs a conditional backbone; the benchmark uses the training-free setting")
    dt = 1.0 / steps

    def refine(x, eta):
        return x - eta * fc.energy_grad_z(x, weight, slack)

    x = x0
    for i in range(steps):
        t, t_next = i * dt, (i + 1) * dt
        if use_cf and i == k_c:
            _, x1_hat = step(x, t)
            x1_c = nearest_anchor(x1_hat, anchors_z)
            x = x1_c if cf_mode == "replace" else (1.0 - t) * x0 + t * x1_c
        v, x1_hat = step(x, t)
        if use_cvf and t >= t_on:
            x1_c = nearest_anchor(x1_hat, anchors_z)
            v = constrain_velocity(v, x1_c - x0, lam)
        x = x + dt * v
        if use_rfe:
            eta = energy_weight(t_next, tau_star, eta_max)
            if eta > 0.0:
                x = refine(x, eta)
    if use_rfe:
        for _ in range(n_refine):
            x = refine(x, eta_max)
    return x


SAMPLERS = {
    "FLOWMATCH": sample_flowmatch,
    "HARDFLOW": sample_hardflow,
    "YFLOW": sample_yflow,
    "SAFEFLOW": sample_safeflow,
    "UNICONFLOW": sample_uniconflow,
    "GUIDEFLOW": sample_guideflow,
}


def anchor_vocabulary(fut_rot: np.ndarray, past_last_step: np.ndarray, limits: KinematicLimits,
                      n_anchors: int, seed: int) -> np.ndarray:
    """GuideFlow trajectory vocabulary V_a: FPS over feasible train futures (agent frame, metres) -> [n, 24]."""
    kc = KinematicConstraint(limits, fut_rot.shape[1])
    p0 = np.zeros((fut_rot.shape[0], 2))
    p_prev, _ = kc.clip_anchor(p0, p0 - past_last_step)
    feasible = kc.max_violation(fut_rot, p0, p_prev) <= 0
    pool = fut_rot[feasible].reshape(int(feasible.sum()), -1)
    n = int(min(max(n_anchors, 1), pool.shape[0]))
    rng = np.random.default_rng(seed)
    picked = [int(rng.integers(pool.shape[0]))]
    d2 = ((pool - pool[picked[0]]) ** 2).sum(-1)
    for _ in range(n - 1):
        j = int(d2.argmax())
        picked.append(j)
        d2 = np.minimum(d2, ((pool - pool[j]) ** 2).sum(-1))
    return pool[np.asarray(picked)]
