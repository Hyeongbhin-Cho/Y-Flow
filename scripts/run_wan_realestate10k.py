"""Generate Wan T2V videos for RealEstate10K prompts and record seeds.

This uses the original clip only for prompt/metadata; its RGB target is not
fed to Wan.  Geometry evaluation must re-match the generated video separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from data import build_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/realestate10k.yaml")
    p.add_argument("--model", default="checkpoints/Wan2.1-T2V-1.3B")
    p.add_argument("--output", type=Path, default=Path("outputs/realestate10k_wan"))
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Wan generation requires a CUDA GPU")
    from diffusers import WanPipeline
    from diffusers.utils import export_to_video

    cfg = OmegaConf.load(args.config)
    cfg.data.n_eval = args.limit
    bundle = build_dataset(cfg)
    pipe = WanPipeline.from_pretrained(str(args.model), torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.to("cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index in range(len(bundle.eval)):
        sample = bundle.eval[index]
        seed = int(sample["noise_seed"])
        generator = torch.Generator(device="cuda").manual_seed(seed)
        result = pipe(prompt=sample["prompt"], height=sample["wan_video"].shape[-2],
                      width=sample["wan_video"].shape[-1], num_frames=sample["wan_video"].shape[1],
                      num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
                      generator=generator)
        frames = result.frames[0]
        video_path = args.output / f"{sample['clip_id']}.mp4"
        export_to_video(frames, str(video_path), fps=16)
        manifest.append({"clip_id": sample["clip_id"], "prompt": sample["prompt"],
                         "noise_seed": seed, "video": str(video_path),
                         "source_pair_indices": sample["pair_indices"].tolist()})
        print(f"generated {index + 1}/{len(bundle.eval)}: {video_path}", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
