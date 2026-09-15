"""Constraint-guided AV2 evaluation on one frozen MoFlow-style teacher.

The samplers retain each method's core intervention but operate on conditional
K-shot trajectories. They are domain adaptations, not exact benchmark code from
the respective papers.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from omegaconf import DictConfig

from data.base import build_dataset, solve_composite_fmbf
from eval.hard_flow import solve_terminal_pgd_hardflow
from eval.moflow import _context_to_device, trajectory_metrics
from eval.unicon_flow import ptzf_reference, qp_guidance
from eval.y_flow import solve_terminal_pgd
from model.moflow import build_moflow_model
from train.checkpoint import load_checkpoint
from train.ema import EMA
from utils.device import get_device
from utils.paths import method_dir


SUPPORTED = frozenset({"hardflow", "safeflow", "uniconflow", "guideflow", "yflow"})


def _cyclic_halfspace_correction(
    a: torch.Tensor, b: torch.Tensor, passes: int = 4
) -> torch.Tensor:
    """Robust projection fallback for constraints a_j + b_j^T u >= 0."""
    correction = torch.zeros(*a.shape[:-1], b.shape[-1], device=b.device, dtype=b.dtype)
    for _ in range(int(passes)):
        for j in range(a.shape[-1]):
            normal = b[..., j, :]
            residual = a[..., j] + (normal * correction).sum(dim=-1)
            step = torch.clamp_min(-residual, 0.0) / normal.square().sum(dim=-1).clamp_min(1e-12)
            correction = correction + step.unsqueeze(-1) * normal
    return correction


def _load(cfg, device):
    bundle = build_dataset(cfg)
    model = build_moflow_model(cfg).to(device)
    ema = EMA(model, decay=float(cfg.train.ema_decay))
    checkpoint = method_dir(cfg, "moflow") / "last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing shared MoFlow checkpoint: {checkpoint}")
    load_checkpoint(checkpoint, model, ema=ema, map_location=device)
    ema.copy_to(model)
    model.eval()
    context = _context_to_device(bundle.eval_context, device)
    generator = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
    shape = (len(bundle.eval_raw), int(cfg.moflow.n_modes), int(cfg.model.dim))
    x0 = torch.randn(*shape, generator=generator).to(device)
    mean = bundle.mean.to(device=device, dtype=x0.dtype)
    std = bundle.std.to(device=device, dtype=x0.dtype)
    with torch.no_grad():
        feature = model.encode_context(context).detach()
    return bundle, model, context, feature, x0, mean, std


def _velocity(model, context, feature, x, t: float):
    batch = x.shape[0]
    times = torch.full((batch,), t, device=x.device, dtype=x.dtype)
    with torch.no_grad():
        return model(x, times, context, context_feature=feature)[0]


@torch.no_grad()
def _nominal_sample(cfg, model, context, feature, x0):
    x = x0.clone()
    steps = int(cfg.sample.n_steps)
    for i in range(steps):
        x = x + _velocity(model, context, feature, x, i / steps) / steps
    return x


def _hardflow(cfg, model, context, feature, x0, mean, std, constraint):
    settings = cfg.hardflow
    steps, x = int(cfg.sample.n_steps), x0
    dt = 1.0 / steps
    for i in range(steps):
        t, t_next = i / steps, (i + 1) / steps
        v = _velocity(model, context, feature, x, t)
        bar = x + dt * v
        if t < float(settings.t_on) and i < steps - 1:
            x = bar
            continue
        v_next = _velocity(model, context, feature, bar, t_next)
        endpoint = bar + (1.0 - t_next) * v_next
        lam = float(settings.lambda_oc) * t_next**2 / dt
        with torch.enable_grad():
            target = solve_terminal_pgd_hardflow(
                endpoint, mean, std, constraint=constraint, lam=lam,
                n_iters=int(settings.max_iter),
                buffer=float(settings.get("safety_buffer", 1e-4)),
            )
        x = t_next * target + (1.0 - t_next) * (bar - t_next * v_next)
    return x.detach(), {}


def _yflow(cfg, model, context, feature, x0, mean, std, constraint):
    settings = cfg.yflow
    steps, x = int(cfg.sample.n_steps), x0
    dt = 1.0 / steps
    for i in range(steps):
        t = i / steps
        v = _velocity(model, context, feature, x, t)
        raw = x + (1.0 - t) * v
        terminal = i == steps - 1
        if not terminal and t < float(settings.t_on):
            eta = dt / max(1.0 - t, 1e-8)
            x = (1.0 - eta) * x + eta * raw
            continue
        physical = raw * std + mean
        projected = constraint.project_physical(physical)
        z_physical = (projected - mean) / std
        lam = float(settings.lambda_oc) * t**2 / dt
        with torch.enable_grad():
            target = solve_terminal_pgd(
                raw, z_physical, mean, std, constraint=constraint, lam=lam,
                mu=float(settings.mu),
                n_iters=int(settings.max_iter) if terminal else max(3, int(settings.max_iter) // 2),
                buffer=float(settings.get("safety_buffer", 1e-4)),
            )
        eta = 1.0 if terminal else dt / max(1.0 - t, 1e-8)
        x = (1.0 - eta) * x + eta * target
    return x.detach(), {}


def _uniconflow(cfg, model, context, feature, x0, mean, std, constraint):
    settings = cfg.uniconflow
    steps, x, initial = int(cfg.sample.n_steps), x0, None
    dt = 1.0 / steps
    for i in range(steps):
        t = i / steps
        with torch.no_grad():
            nominal = _velocity(model, context, feature, x, t)
        with torch.enable_grad():
            z = x.detach().requires_grad_(True)
            h = torch.stack(list(constraint.h(z * std + mean).values()), dim=-1)
            grads = torch.stack(
                [torch.autograd.grad(h[..., j].sum(), z, retain_graph=True)[0] for j in range(h.shape[-1])],
                dim=-2,
            )
        if initial is None:
            initial = torch.clamp(h.detach(), min=0.0) + float(settings.safety_buffer)
        if t < float(settings.free_until):
            guidance = torch.zeros_like(nominal)
        else:
            local_t = (t - float(settings.free_until)) / max(1.0 - float(settings.free_until), 1e-8)
            reference, derivative = ptzf_reference(initial, local_t, float(settings.ptzf_rate))
            derivative = derivative / max(1.0 - float(settings.free_until), 1e-8)
            nominal_dh = (grads * nominal.unsqueeze(-2)).sum(dim=-1)
            rho = nominal_dh - float(settings.gamma) * (reference - h.detach()) - derivative
            flat_guidance = qp_guidance(
                rho.reshape(-1, rho.shape[-1]),
                grads.reshape(-1, grads.shape[-2], grads.shape[-1]),
                slack_weight=float(settings.slack_weight),
            )
            guidance = flat_guidance.reshape_as(x)
            norm = torch.linalg.vector_norm(guidance, dim=-1, keepdim=True)
            guidance = guidance * torch.clamp(float(settings.max_guidance_norm) / norm.clamp_min(1e-12), max=1.0)
        x = x + dt * (nominal + guidance)
    pre_terminal = x.detach()
    if bool(settings.terminal_refinement):
        x = (constraint.project_feasible(x * std + mean, buffer=float(settings.safety_buffer)) - mean) / std
    return x.detach(), {"_pre_terminal_z": pre_terminal}


def _safeflow(cfg, model, context, feature, x0, mean, std, constraint):
    settings = cfg.safeflow
    steps, x = int(cfg.sample.n_steps), x0
    dt, corrected_steps, fallback_steps = 1.0 / steps, 0, 0
    for i in range(steps):
        t = i / steps
        nominal = _velocity(model, context, feature, x, t)
        if t >= float(settings.t_on):
            with torch.enable_grad():
                z = x.detach().requires_grad_(True)
                h = torch.stack(list(constraint.h(z * std + mean).values()), dim=-1)
                grads = torch.stack(
                    [torch.autograd.grad(h[..., j].sum(), z, retain_graph=True)[0] for j in range(h.shape[-1])],
                    dim=-2,
                )
            gain = float(settings.get("av2_gain", 2.0)) / max(1.0 - t, 1.0 / steps)
            a = -(grads * nominal.unsqueeze(-2)).sum(dim=-1) - gain * h.detach()
            flat_a = a.reshape(-1, a.shape[-1])
            flat_b = (-grads).reshape(-1, grads.shape[-2], grads.shape[-1])
            # Scaling a half-space by a positive value leaves it unchanged and
            # greatly improves conditioning for AV2 acceleration gradients.
            row_scale = torch.linalg.vector_norm(flat_b, dim=-1).clamp_min(1e-6)
            flat_a = flat_a / row_scale
            flat_b = flat_b / row_scale.unsqueeze(-1)
            try:
                correction = solve_composite_fmbf(
                    flat_a, flat_b, slack_weight=float(settings.slack_weight)
                ).correction
            except RuntimeError:
                correction = _cyclic_halfspace_correction(flat_a, flat_b)
                fallback_steps += 1
            nominal = nominal + correction.reshape_as(x)
            corrected_steps += 1
        x = x + dt * nominal
    pre = constraint.h(x * std + mean)
    pre_safe = torch.stack(list(pre.values()), dim=-1).le(0).all(dim=-1).float().mean()
    pre_terminal = x.detach()
    if bool(settings.terminal_filter.enabled):
        x = (constraint.project_feasible(x * std + mean) - mean) / std
    return x.detach(), {
        "pre_filter_safe_ratio": float(pre_safe.cpu()),
        "correction_steps": corrected_steps,
        "qp_fallback_steps": fallback_steps,
        "_pre_terminal_z": pre_terminal,
    }


def _guideflow(cfg, model, context, feature, x0, mean, std, constraint):
    settings = cfg.guideflow
    steps, x = int(cfg.sample.n_steps), x0
    dt, corrections, accepted_items, attempted_items = 1.0 / steps, 0, 0, 0
    t_on = float(settings.get("tau_star", 0.5))
    max_norm = float(settings.get("av2_max_guidance_norm", 5.0))
    for i in range(steps):
        t = i / steps
        t_next = (i + 1) / steps
        x = x + dt * _velocity(model, context, feature, x, t)
        if t_next > t_on:
            with torch.enable_grad():
                z = x.detach().requires_grad_(True)
                energy = constraint.cost(z * std + mean).sum()
                grad = torch.autograd.grad(energy, z)[0]
            norm = torch.linalg.vector_norm(grad, dim=-1, keepdim=True)
            grad = grad * torch.clamp(max_norm / norm.clamp_min(1e-12), max=1.0)
            weight = float(settings.eta_max) * (t_next - t_on) / max(1.0 - t_on, 1e-8)
            base_cost = constraint.cost(x * std + mean).detach()
            accepted = torch.zeros_like(base_cost, dtype=torch.bool)
            refined = x
            step_size = dt * weight
            for _ in range(int(settings.get("av2_line_search_steps", 8))):
                candidate = x - step_size * grad
                candidate_cost = constraint.cost(candidate * std + mean).detach()
                take = (~accepted) & (candidate_cost <= base_cost)
                refined = torch.where(take.unsqueeze(-1), candidate, refined)
                accepted = accepted | take
                step_size *= 0.5
            x = refined.detach()
            corrections += 1
            accepted_items += int(accepted.sum())
            attempted_items += int(accepted.numel())
    return x.detach(), {
        "energy_correction_steps": corrections,
        "energy_acceptance_rate": accepted_items / max(attempted_items, 1),
    }


_SAMPLERS = {
    "hardflow": _hardflow,
    "safeflow": _safeflow,
    "uniconflow": _uniconflow,
    "guideflow": _guideflow,
    "yflow": _yflow,
}


def run_eval_autonomous(cfg: DictConfig, method: str, device=None) -> dict:
    if method not in SUPPORTED:
        raise KeyError(f"unsupported autonomous method: {method}")
    device = device or get_device(cfg)
    bundle, model, context, feature, x0, mean, std = _load(cfg, device)
    nominal_z = _nominal_sample(cfg, model, context, feature, x0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    predictions_z, diagnostics = _SAMPLERS[method](
        cfg, model, context, feature, x0, mean, std, bundle.constraint
    )
    # Constraint samplers temporarily enable autograd to obtain Jacobians or
    # optimization gradients. Generated trajectories are inference artifacts.
    predictions_z = predictions_z.detach()
    final_t = torch.ones(x0.shape[0], device=device, dtype=x0.dtype)
    with torch.no_grad():
        _, logits = model(predictions_z, final_t, context, context_feature=feature)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    predictions = (predictions_z * std + mean).detach().cpu().numpy()
    nominal = (nominal_z * std + mean).detach().cpu().numpy()
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
    metrics = trajectory_metrics(predictions, bundle.eval_raw, probabilities)
    pre_terminal_z = diagnostics.pop("_pre_terminal_z", None)
    if pre_terminal_z is not None:
        pre_terminal = (pre_terminal_z * std + mean).detach().cpu().numpy()
        pre_metrics = trajectory_metrics(pre_terminal, bundle.eval_raw)
        metrics["pre_terminal_minADE"] = pre_metrics["minADE"]
        metrics["pre_terminal_minFDE"] = pre_metrics["minFDE"]
        pre_h = bundle.constraint.h(pre_terminal.reshape(-1, pre_terminal.shape[-1]))
        pre_stacked = np.stack(list(pre_h.values()), axis=-1)
        metrics["pre_terminal_safe_ratio"] = float((pre_stacked <= 0).all(axis=-1).mean())
        terminal_shift = np.linalg.norm(
            predictions.reshape(*predictions.shape[:2], -1, 2)
            - pre_terminal.reshape(*pre_terminal.shape[:2], -1, 2),
            axis=-1,
        )
        metrics["terminal_projection_ADE_m"] = float(terminal_shift.mean())
        metrics["terminal_projection_FDE_m"] = float(terminal_shift[..., -1].mean())
    displacement = np.linalg.norm(
        predictions.reshape(*predictions.shape[:2], -1, 2)
        - nominal.reshape(*nominal.shape[:2], -1, 2),
        axis=-1,
    )
    metrics["intervention_ADE_m"] = float(displacement.mean())
    metrics["intervention_FDE_m"] = float(displacement[..., -1].mean())
    metrics["intervention_max_m"] = float(displacement.max(axis=-1).mean())
    h = bundle.constraint.h(predictions.reshape(-1, predictions.shape[-1]))
    stacked = np.stack(list(h.values()), axis=-1)
    safe = (stacked <= 0).all(axis=-1).reshape(predictions.shape[:2])
    rows, top1 = np.arange(len(predictions)), probabilities.argmax(axis=1)
    metrics.update({
        "safe_ratio": float(safe.mean()),
        "oracle_safe_ratio": float(safe.any(axis=1).mean()),
        "top1_safe_ratio": float(safe[rows, top1].mean()),
    })
    for name, value in h.items():
        metrics[f"{name}_viol_rate"] = float((value > 0).mean())
        metrics[f"{name}_viol_mean"] = float(np.maximum(value, 0).mean())
    metrics.update({
        "method": method, "backbone": "moflow", "adaptation": "AV2 conditional K-shot",
        "run_name": str(cfg.run_name), "n_scenarios": len(predictions),
        "n_modes": predictions.shape[1], "n_steps": int(cfg.sample.n_steps),
        "inference_time_s": float(elapsed), **diagnostics,
    })
    out_dir = method_dir(cfg, method)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "eval_predictions.npy", predictions)
    np.save(out_dir / "eval_probabilities.npy", probabilities)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    from eval.evaluate import write_run_metrics

    write_run_metrics(cfg)
    print(json.dumps(metrics, indent=2))
    return metrics
