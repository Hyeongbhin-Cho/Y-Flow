import time
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import torch

from flow_drive.yflow import constraints as C
from flow_drive.yflow.smoothing import heading_from_positions, smooth_positions
from flow_drive.yflow.solver import admm_project


@dataclass
class YFlowConfig:
    enabled: bool = True
    t_on: float = 0.5
    gamma_max: float = 1.0
    lam: float = 1.0
    mu: float = 1.0
    smooth_alpha: float = 2.0
    w_progress: float = 0.0
    dt: float = 0.1
    a_max: float = 4.0
    accel_mode: str = "ball"
    a_lon_max: float = 2.2
    a_lon_min: float = -3.8
    a_lat_max: float = 4.5
    retime_final_only: bool = False
    v_tol: float = 0.0
    use_corridor: bool = True
    corridor_margin: float = C.EGO_HALF_WIDTH + 0.1
    use_obstacles: bool = False
    obstacle_radius: float = C.CIRCLE_RADIUS + 0.2
    iters: int = 150
    final_iters: int = 600
    rho: float = 300.0
    scp_rounds: int = 4
    scp_damping: float = 1.0
    tol_speed: float = 0.1
    tol_accel: float = 0.3
    tol_pos: float = 0.05
    solver_device: str = "cpu"
    fallback_resample: bool = False
    corridor_precheck: bool = False


@dataclass
class YFlowContext:
    v0: torch.Tensor
    v_limit: torch.Tensor
    corridor: Optional[C.CorridorData] = None
    obstacles: Optional[list] = None
    cache: dict = field(default_factory=dict)


@dataclass
class YFlowStats:
    active_steps: int = 0
    level: str = "raw"
    levels: list = field(default_factory=list)
    viol: dict = field(default_factory=dict)
    shift_m: float = 0.0
    n_obs_rows: int = 0
    solve_ms: float = 0.0
    resampled: int = 0
    corr_skipped: bool = False

    def as_dict(self):
        return {"active_steps": self.active_steps, "level": self.level, "levels": self.levels,
                "viol": {k: float(v) for k, v in self.viol.items()},
                "shift_m": self.shift_m, "n_obs_rows": self.n_obs_rows, "solve_ms": self.solve_ms,
                "resampled": self.resampled, "corr_skipped": self.corr_skipped}


def _levels(cfg: YFlowConfig, ctx: YFlowContext):
    full = ["kin"]
    if cfg.use_corridor and ctx.corridor is not None:
        full.append("corr")
    if cfg.use_obstacles and ctx.obstacles is not None:
        full.append("obs")
    lv = [tuple(full)]
    while len(full) > 1:
        full = full[:-1]
        lv.append(tuple(full))
    return lv


def project_target(x1_norm, normalizer, cfg: YFlowConfig, ctx: YFlowContext, gamma: float,
                   final: bool, stats: YFlowStats):
    t0 = time.perf_counter()
    dev, dt_ = x1_norm.device, x1_norm.dtype
    sd = torch.device(cfg.solver_device)
    phys = normalizer.inverse(x1_norm).to(sd, torch.float64)
    pos_raw, cs_raw = phys[..., :2], phys[..., 2:4]
    cs_raw = cs_raw / cs_raw.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    B, T, _ = pos_raw.shape
    v0 = ctx.v0.to(sd, torch.float64)
    vlim = ctx.v_limit.to(sd, torch.float64)

    z_phys = smooth_positions(pos_raw, v0, cfg.dt, cfg.smooth_alpha)
    cs_lin = heading_from_positions(z_phys, cs_raw)
    mu = cfg.mu * gamma
    zbar = (cfg.lam * pos_raw + mu * z_phys) / (cfg.lam + mu)
    if cfg.w_progress > 0:
        zbar[:, -1] += cfg.w_progress / (cfg.lam + mu) * cs_lin[:, -1]
    if cfg.accel_mode == "lonlat" and (final or not cfg.retime_final_only):
        zbar = C.retime_lon(zbar, v0, cfg.dt, cfg.a_lon_min, cfg.a_lon_max)
    zbar_f = zbar.reshape(B, 2 * T)

    lonlat = cfg.accel_mode == "lonlat"
    kin = C.kin_rows(B, T, v0, vlim, cfg.a_max, cfg.dt, cfg.v_tol, torch.float64, sd,
                     decel=-cfg.a_lon_min if lonlat else None)
    if lonlat:
        G_acc, c_acc = kin[0][:, T:], kin[1][:, T:]
        kin = (kin[0][:, :T], kin[1][:, :T], kin[2][:, :T])

    blocking_cache = {}

    def build(pos_lin, cs_l):
        parts = {"kin": kin}
        if lonlat:
            parts["acc"] = C.accel_lonlat_rows(G_acc, c_acc, cs_l, cfg.a_lon_max, cfg.a_lon_min,
                                               cfg.a_lat_max, cfg.dt)
        if cfg.use_corridor and ctx.corridor is not None:
            parts["corr"] = C.corridor_rows(pos_lin, cs_l, ctx.corridor, cfg.corridor_margin)
        if cfg.use_obstacles and ctx.obstacles is not None:
            A, lo, hi, n = C.obstacle_rows(pos_lin, cs_l, ctx.obstacles, cfg.obstacle_radius,
                                           corr=ctx.corridor if cfg.use_corridor else None,
                                           blocking_cache=blocking_cache, corr_margin=cfg.corridor_margin,
                                           frenet_cache=ctx.cache)
            parts["obs"] = (A, lo, hi)
            stats.n_obs_rows = n
        return parts

    levels = _levels(cfg, ctx) if final else _levels(cfg, ctx)[:1]
    iters = cfg.final_iters if final else max(30, int(cfg.iters * gamma))
    rounds = cfg.scp_rounds if final else 1

    def check(zz, lv):
        pl = zz.reshape(B, T, 2)
        parts = build(pl, heading_from_positions(pl, cs_lin))
        pv = C.physical_violations(zz, kin, parts.get("corr") if "corr" in lv else None,
                                   parts.get("obs") if "obs" in lv else None, cfg.dt, acc=parts.get("acc"))
        score = torch.stack([pv["speed"] / cfg.tol_speed, pv["accel"] / cfg.tol_accel,
                             pv["corridor"] / cfg.tol_pos, pv["obstacle"] / cfg.tol_pos]).amax(0)
        return score, pv

    z_out = None
    accepted = torch.zeros(B, dtype=torch.bool)
    level_of = ["raw"] * B
    viol_of = [None] * B
    for li, lv in enumerate(levels):
        pos_lin, cs_l, z0 = z_phys, cs_lin, None
        best_z, best_s, best_pv = None, None, None
        for r_i in range(rounds):
            parts = build(pos_lin, cs_l)
            corr = parts.get("corr") if "corr" in lv else None
            obs = parts.get("obs") if "obs" in lv else None
            target = zbar_f if z0 is None else (zbar_f + cfg.scp_damping * z0) / (1 + cfg.scp_damping)
            z = admm_project(target, C.assemble(kin, corr, obs, acc=parts.get("acc")), iters=iters, rho=cfg.rho, z0=z0)
            z0 = z
            pos_lin = z.reshape(B, T, 2)
            cs_l = heading_from_positions(pos_lin, cs_lin)
            if final:
                s, pv = check(z, lv)
                if best_z is None:
                    best_z, best_s, best_pv = z.clone(), s.clone(), {k: v.clone() for k, v in pv.items()}
                else:
                    better = s < best_s
                    best_z[better] = z[better]; best_s[better] = s[better]
                    for k in pv:
                        best_pv[k][better] = pv[k][better]
                if bool((best_s <= 1).all()):
                    break
        if not final:
            z_out = z
            break
        if z_out is None:
            z_out = best_z.clone()
        take = (~accepted) & ((best_s <= 1) | torch.tensor(li == len(levels) - 1))
        z_out[take] = best_z[take]
        for b in torch.nonzero(take).flatten().tolist():
            level_of[b] = "+".join(lv) + ("" if best_s[b] <= 1 else "(fail)")
            viol_of[b] = {k: best_pv[k][b].item() for k in best_pv}
        accepted |= best_s <= 1
        if bool(accepted.all()):
            break
    z = z_out
    if final:
        from collections import Counter
        stats.level = ",".join(f"{k}:{v}" for k, v in Counter(level_of).items()) if B > 1 else level_of[0]
        stats.levels = level_of
        stats.viol = {k: max(v[k] for v in viol_of if v is not None) for k in viol_of[0]}
    pos = z.reshape(B, T, 2)
    if final:
        stats.shift_m = (pos - pos_raw).norm(dim=-1).mean().item()
    cs = heading_from_positions(pos, cs_lin)
    out = normalizer(torch.cat([pos, cs], -1).to(dev, dt_))
    stats.solve_ms += (time.perf_counter() - t0) * 1e3
    return out


def sample_action_yflow(params, noise_pred_net, obs_cond, sampler, ego_current_state,
                        normalizer, cfg: YFlowConfig, ctx: Optional[YFlowContext],
                        x_init: Optional[torch.Tensor] = None,
                        step_hook: Optional[Callable] = None, hook_index: Optional[int] = None):
    device = obs_cond["encoding"].device
    B = obs_cond["encoding"].shape[0]
    T = params.diffuser.pred_horizon
    active = cfg.enabled and ctx is not None
    corr_skipped = False
    if (active and cfg.corridor_precheck and cfg.use_corridor and ctx.corridor is not None
            and not C.corridor_feasible_now(ctx.corridor, cfg.corridor_margin, cfg.tol_pos)):
        ctx = replace(ctx, corridor=None)
        corr_skipped = True
    if x_init is None:
        x_init = torch.randn((B, T, 4), device=device)
    run = lambda c: _sample_once(params, noise_pred_net, obs_cond, sampler, ego_current_state,
                                 normalizer, c, ctx, x_init, step_hook, hook_index)
    out, stats = run(cfg)
    stats.corr_skipped = corr_skipped
    if not (active and cfg.fallback_resample):
        return out, stats

    levels = _levels(cfg, ctx)
    full = "+".join(levels[0])
    bad = [lv != full for lv in stats.levels]
    if len(levels) == 1 or not any(bad):
        return out, stats
    out = out.clone()
    for li, lv in enumerate(levels[1:], start=1):
        cfg2 = replace(cfg, use_corridor="corr" in lv, use_obstacles="obs" in lv)
        out2, st2 = run(cfg2)
        stats.resampled += 1
        stats.solve_ms += st2.solve_ms
        stats.active_steps += st2.active_steps
        for k, v in st2.viol.items():
            stats.viol[k] = max(stats.viol.get(k, 0.0), v)
        name, last = "+".join(lv), li == len(levels) - 1
        for b in range(B):
            if bad[b] and (st2.levels[b] == name or last):
                out[b] = out2[b]
                stats.levels[b] = st2.levels[b] + "(rs)"
                bad[b] = False
        if not any(bad):
            break
    if B > 1:
        from collections import Counter
        stats.level = ",".join(f"{k}:{v}" for k, v in Counter(stats.levels).items())
    else:
        stats.level = stats.levels[0]
    return out, stats


def _sample_once(params, noise_pred_net, obs_cond, sampler, ego_current_state, normalizer,
                 cfg: YFlowConfig, ctx: Optional[YFlowContext], x_init: torch.Tensor,
                 step_hook: Optional[Callable], hook_index: Optional[int]):
    sampler.set_timesteps(params.inference.flow_inference_iter)
    device = obs_cond["encoding"].device
    B = obs_cond["encoding"].shape[0]
    T = params.diffuser.pred_horizon
    x = x_init.clone()
    x = torch.cat([ego_current_state.unsqueeze(1), x], dim=1)

    sig = sampler.sigmas.to(device=device, dtype=torch.float32)
    float_ts = sampler.timesteps
    int_ts = torch.linspace(0, sampler.config.num_train_timesteps - 1, steps=len(float_ts),
                            device=device).long()
    N = len(int_ts)
    stats = YFlowStats()
    for i, t_int in enumerate(int_ts):
        with torch.no_grad():
            v_fd = noise_pred_net(x, t_int, global_cond=obs_cond)
        s_i, s_n = sig[i], sig[i + 1]
        x1 = x - s_i * v_fd
        x1[:, 0, :] = ego_current_state
        t = float(1.0 - s_i)
        final = i == N - 1
        gamma = 0.0 if t < cfg.t_on else cfg.gamma_max * (t - cfg.t_on) / max(1e-6, 1.0 - cfg.t_on)
        if cfg.enabled and ctx is not None and (gamma > 0 or final):
            stats.active_steps += 1
            x1[:, 1:] = project_target(x1[:, 1:], normalizer, cfg, ctx,
                                       1.0 if final else min(1.0, gamma), final, stats)
        eta = (s_i - s_n) / s_i
        x = x + eta * (x1 - x)
        x[:, 0, :] = ego_current_state
        if step_hook is not None and i == hook_index:
            x = step_hook(x)
        x = x.detach()
    return x[:, 1:, :], stats
