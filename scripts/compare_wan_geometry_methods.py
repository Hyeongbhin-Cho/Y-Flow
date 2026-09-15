"""Compare FlowMatch, terminal warp, YFlow-Geo, and HardFlow-Geo on Wan.

The `-Geo` methods use an RGB/VAE bridge, never a fictional pixel constraint
on Wan latents.  They are experimental ablations, not implementations of the
original state-space HardFlow/YFlow guarantees.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import build_dataset
from data.realestate10k import RealEstate10KEpipolarConstraint
from scripts.verify_realestate10k_geometry import verify
from scripts.warp_realestate10k_geometry import export_video, metrics, warp_target


def _stats(vae, reference):
    mean = torch.as_tensor(vae.config.latents_mean, device=reference.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    std = torch.as_tensor(vae.config.latents_std, device=reference.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    return mean, std


class GeometryBridge:
    """Decode, warp accepted frame pairs, and deterministically re-encode."""

    def __init__(self, pipe, sample, max_matches, max_displacement, support_radius, min_matches):
        self.pipe, self.sample = pipe, sample
        self.max_matches = max_matches
        self.max_displacement = max_displacement
        self.support_radius = support_radius
        self.min_matches = min_matches
        self.constraint = RealEstate10KEpipolarConstraint(tolerance_px=0.0)
        self.calls = []
        self.F_by_pair = {
            tuple(pair.tolist()): sample["fundamental_matrices"][n].cpu().numpy()
            for n, pair in enumerate(sample["pair_indices"])
            if bool(sample["fundamental_valid"][n])
        }

    @torch.no_grad()
    def _decode(self, latents):
        mean, std = _stats(self.pipe.vae, latents)
        raw = latents.float() * std + mean
        video = self.pipe.vae.decode(raw.to(dtype=self.pipe.vae.dtype), return_dict=False)[0]
        return ((video[0].permute(1, 2, 3, 0).float().cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    @torch.no_grad()
    def _encode(self, frames, dtype):
        video = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float().div(127.5).sub(1.0)
        parameter = next(self.pipe.vae.parameters())
        posterior = self.pipe.vae.encode(video.to(device=parameter.device, dtype=parameter.dtype)).latent_dist
        raw = posterior.mode()
        mean, std = _stats(self.pipe.vae, raw)
        return ((raw.float() - mean) / std).to(dtype=dtype)

    @torch.no_grad()
    def __call__(self, clean_latents, *, step, sigma):
        frames = self._decode(clean_latents)
        warped = frames.copy()
        pairs = []
        accepted_pairs = 0
        # Camera/F is supplied for observed frames only. The padded Wan frame
        # is deliberately left unchanged.
        for target_index in range(1, min(len(frames), int(self.sample["video"].shape[1]))):
            F = self.F_by_pair.get((0, target_index))
            if F is None:
                continue
            before = metrics(self.constraint, frames[0], frames[target_index], F, self.max_matches)
            candidate, controls = warp_target(
                self.constraint, frames[0], frames[target_index], F, self.max_matches,
                self.max_displacement, self.support_radius,
            )
            after = metrics(self.constraint, frames[0], candidate, F, self.max_matches)
            accept = (
                controls.get("status") == "ok" and after.get("status") == "ok"
                and after["matches"] >= self.min_matches
                and after["median_px"] < before.get("median_px", float("inf"))
            )
            if accept:
                warped[target_index] = candidate
                accepted_pairs += 1
            pairs.append({"pair": [0, target_index], "before": before, "controls": controls,
                          "after": after, "accepted": accept})
        event = {"step": int(step), "sigma": float(sigma), "accepted_pairs": accepted_pairs, "pairs": pairs}
        self.calls.append(event)
        if accepted_pairs == 0:
            return None
        return self._encode(warped, clean_latents.dtype)


def _predict_velocity(pipe, latents, timestep, prompt_embeds, negative_prompt_embeds, attention_kwargs):
    dtype = pipe.transformer.dtype
    model_input = latents.to(dtype)
    with pipe.transformer.cache_context("cond"):
        prediction = pipe.transformer(
            hidden_states=model_input, timestep=timestep, encoder_hidden_states=prompt_embeds,
            attention_kwargs=attention_kwargs, return_dict=False,
        )[0]
    if pipe.do_classifier_free_guidance:
        with pipe.transformer.cache_context("uncond"):
            uncond = pipe.transformer(
                hidden_states=model_input, timestep=timestep, encoder_hidden_states=negative_prompt_embeds,
                attention_kwargs=attention_kwargs, return_dict=False,
            )[0]
        prediction = uncond + pipe.guidance_scale * (prediction - uncond)
    return prediction


@torch.no_grad()
def sample_method(pipe, sample, method, steps, guidance_scale, alpha, correction_last_steps, bridge):
    """Copied Wan2.1 loop with only the explicitly listed latent interventions."""
    if getattr(pipe, "transformer_2", None) is not None or pipe.config.boundary_ratio is not None:
        raise ValueError("This comparison script currently supports Wan2.1's single-transformer pipeline only")
    device = pipe._execution_device
    pipe._guidance_scale = guidance_scale
    pipe._attention_kwargs = None
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=sample["prompt"], negative_prompt=None,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        num_videos_per_prompt=1, max_sequence_length=512, device=device,
    )
    prompt_embeds = prompt_embeds.to(pipe.transformer.dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(pipe.transformer.dtype)
    pipe.scheduler.set_timesteps(steps, device=device)
    pipe.scheduler.set_begin_index(0)
    generator = torch.Generator(device=device).manual_seed(int(sample["noise_seed"]))
    height, width = sample["wan_video"].shape[-2:]
    frames = sample["wan_video"].shape[1]
    latents = pipe.prepare_latents(1, pipe.transformer.config.in_channels, height, width, frames,
                                   torch.float32, device, generator, None)
    mask = torch.ones_like(latents, dtype=torch.float32)
    for index, timestep_value in enumerate(pipe.scheduler.timesteps):
        timestep = timestep_value.expand(latents.shape[0])
        velocity = _predict_velocity(pipe, latents, timestep, prompt_embeds, negative_prompt_embeds, None)
        sigma = float(pipe.scheduler.sigmas[index])
        next_sigma = float(pipe.scheduler.sigmas[index + 1])
        late = index >= len(pipe.scheduler.timesteps) - correction_last_steps and next_sigma > 0.0
        clean = latents - sigma * velocity
        if method == "yflow_geo" and late:
            geo = bridge(clean, step=index, sigma=sigma)
            if geo is not None:
                target = (1.0 - alpha) * clean + alpha * geo
                velocity = (latents - target) / max(sigma, 1e-6)
        if method == "hardflow_geo" and late:
            geo = bridge(clean, step=index, sigma=sigma)
            if geo is not None:
                # Preserve the estimated noise component while replacing only
                # the terminal target; with geo=clean this equals Euler step.
                target = (1.0 - alpha) * clean + alpha * geo
                noise = (latents - (1.0 - sigma) * clean) / max(sigma, 1e-6)
                pipe.scheduler.step(velocity, timestep_value, latents, return_dict=False)
                latents = (1.0 - next_sigma) * target + next_sigma * noise
                continue
        latents = pipe.scheduler.step(velocity, timestep_value, latents, return_dict=False)[0]
    if method == "terminal_warp":
        geo = bridge(latents, step=steps, sigma=0.0)
        if geo is not None:
            latents = geo
    return bridge._decode(latents)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/realestate10k.yaml")
    parser.add_argument("--model", default="checkpoints/Wan2.1-T2V-1.3B")
    parser.add_argument("--output", type=Path, default=Path("outputs/realestate10k_geo_compare"))
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--correction-last-steps", type=int, default=2)
    parser.add_argument("--max-matches", type=int, default=1000)
    parser.add_argument("--max-displacement", type=float, default=64.0)
    parser.add_argument("--support-radius", type=float, default=48.0)
    parser.add_argument("--min-matches", type=int, default=8)
    parser.add_argument("--methods", nargs="+", default=["flowmatch", "terminal_warp", "yflow_geo", "hardflow_geo"],
                        choices=["flowmatch", "terminal_warp", "yflow_geo", "hardflow_geo"])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Wan geometry comparison requires CUDA")
    if not 0.0 <= args.alpha <= 1.0 or args.correction_last_steps < 1:
        parser.error("alpha must be in [0,1] and correction-last-steps must be positive")
    from diffusers import WanPipeline

    cfg = OmegaConf.load(args.config)
    cfg.data.n_eval = args.limit
    bundle = build_dataset(cfg)
    pipe = WanPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.to("cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    run = {"methods": {}, "settings": {key: value for key, value in vars(args).items() if key != "output"}}
    for sample_index in range(len(bundle.eval)):
        sample = bundle.eval[sample_index]
        for method in args.methods:
            destination = args.output / method
            destination.mkdir(parents=True, exist_ok=True)
            bridge = GeometryBridge(pipe, sample, args.max_matches, args.max_displacement, args.support_radius, args.min_matches)
            frames = sample_method(pipe, sample, method, args.steps, args.guidance_scale, args.alpha,
                                   args.correction_last_steps, bridge)
            video_path = destination / f"{sample['clip_id']}.mp4"
            export_video(frames, video_path)
            geometry = verify(cfg.data.cache_dir, cfg.data.eval_split, 1, args.max_matches, destination,
                              clip_ids=[sample["clip_id"]])
            record = {"video": str(video_path), "bridge_calls": bridge.calls, "geometry": geometry}
            run["methods"].setdefault(method, {})[sample["clip_id"]] = record
            (destination / f"{sample['clip_id']}_report.json").write_text(json.dumps(record, indent=2))
            print(f"completed {method}: {video_path}", flush=True)
    (args.output / "comparison.json").write_text(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
