#!/usr/bin/env python3
"""
Delta-zero comparison: Flow-Factory LTXUnionAdapter vs original ValidationSampler.

Runs both pipelines with identical inputs (same seed, prompt, reference video)
and asserts torch.equal() at every stage:
  1. Sigma schedule
  2. Reference latents (VAE-encoded)
  3. Initial noise
  4. Each denoising step
  5. Final latents
  6. Decoded video pixels

Usage:
    LTX_MODEL_PATH=/path/to/model \
    LTX_UNION_LORA_PATH=/path/to/lora.safetensors \
    LTX_GEMMA_PATH=/path/to/gemma \
    python scripts/delta_zero_compare.py

Requirements: GPU + ltx_core + ltx_trainer + flow_factory installed
"""
from __future__ import annotations

import os
import sys
import torch
import numpy as np
from dataclasses import replace

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ["LTX_MODEL_PATH"]
LORA_PATH = os.environ["LTX_UNION_LORA_PATH"]
GEMMA_PATH = os.environ["LTX_GEMMA_PATH"]

SEED = 42
PROMPT = "A person dancing in a studio"
NUM_STEPS = 10
HEIGHT = 256
WIDTH = 416
NUM_FRAMES = 9
DEVICE = "cuda"
DTYPE = torch.bfloat16


def _report(name: str, a: torch.Tensor, b: torch.Tensor):
    """Report match/mismatch between two tensors."""
    if torch.equal(a, b):
        print(f"  [PASS] {name}: bitwise identical")
    else:
        diff = (a.float() - b.float()).abs()
        print(
            f"  [FAIL] {name}: max_diff={diff.max().item():.2e}, "
            f"mean_diff={diff.mean().item():.2e}, "
            f"mismatched_elements={(a != b).sum().item()}/{a.numel()}"
        )


# ---------------------------------------------------------------------------
# 1. Load model once (shared by both pipelines)
# ---------------------------------------------------------------------------
print("=" * 60)
print("Loading model components...")
print("=" * 60)

from ltx_trainer.model_loader import (
    load_transformer,
    load_video_vae_encoder,
    load_video_vae_decoder,
    load_text_encoder,
    load_embeddings_processor,
)
from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.components.diffusion_step import EulerDiffusionStep
from ltx_core.model.x0_model import X0Model
from ltx_core.model.transformer.modality import Modality
from ltx_core.types import VideoLatentShape, VIDEO_SCALE_FACTORS
from ltx_core.components.patchifiers import get_pixel_coords

# Load all components
transformer = load_transformer(MODEL_PATH)

# Merge LoRA
from flow_factory.models.ltx.pipeline import _merge_lora_into_model
reference_downscale_factor = _merge_lora_into_model(transformer, LORA_PATH)
print(f"reference_downscale_factor = {reference_downscale_factor}")

vae_encoder = load_video_vae_encoder(MODEL_PATH)
vae_decoder = load_video_vae_decoder(MODEL_PATH)
text_encoder = load_text_encoder(GEMMA_PATH)
embeddings_processor = load_embeddings_processor(MODEL_PATH)
patchifier = VideoLatentPatchifier(patch_size=1)

# Move to device
transformer.to(DEVICE, DTYPE).eval()
vae_encoder.to(DEVICE).eval()
vae_decoder.to(DEVICE).eval()
text_encoder.to(DEVICE).eval()
embeddings_processor.to(DEVICE).eval()

dsf = reference_downscale_factor

# ---------------------------------------------------------------------------
# 2. Encode prompt (shared)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Encoding prompt...")
print("=" * 60)

with torch.no_grad():
    hidden_states, attention_mask = text_encoder.encode(PROMPT)
    enc_output = embeddings_processor.process_hidden_states(hidden_states, attention_mask)

prompt_embeds = enc_output.video_encoding  # [1, seq, dim]
audio_embeds = enc_output.audio_encoding
prompt_mask = enc_output.attention_mask
print(f"  prompt_embeds: {prompt_embeds.shape}, dtype={prompt_embeds.dtype}")

# ---------------------------------------------------------------------------
# 3. Create synthetic reference video (deterministic)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Creating reference video...")
print("=" * 60)

# Use a fixed random reference video in [-1, 1] (BCFHW format)
ref_h = HEIGHT // dsf
ref_w = WIDTH // dsf
# Valid frames: (F-1)//8*8 + 1
valid_frames = (NUM_FRAMES - 1) // 8 * 8 + 1

torch.manual_seed(999)  # Fixed seed for reference video
ref_video = torch.randn(1, 3, valid_frames, ref_h, ref_w, device=DEVICE, dtype=torch.float32)
ref_video = ref_video.clamp(-1, 1)  # Realistic range
print(f"  ref_video: {ref_video.shape} (pixel space, [-1,1])")

# VAE encode reference
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
    ref_latents = vae_encoder(ref_video)
print(f"  ref_latents: {ref_latents.shape}, dtype={ref_latents.dtype}")

# ---------------------------------------------------------------------------
# 4. Sigma schedule
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Comparing sigma schedules...")
print("=" * 60)

ltx_sched = LTX2Scheduler()
orig_sigmas = ltx_sched.execute(steps=NUM_STEPS)
print(f"  Original sigmas ({len(orig_sigmas)}): {orig_sigmas.tolist()}")

from flow_factory.models.ltx.scheduler_bridge import LTXSDEScheduler
ff_sched = LTXSDEScheduler(num_inference_steps=NUM_STEPS, dynamics_type="ODE", noise_level=0.0)
ff_sched.set_timesteps(NUM_STEPS, device=DEVICE)
ff_sigmas = ff_sched.sigmas.cpu()
print(f"  FF sigmas ({len(ff_sigmas)}): {ff_sigmas.tolist()}")

_report("sigma_schedule", orig_sigmas.cpu().float(), ff_sigmas.float())

# ---------------------------------------------------------------------------
# 5. Initial noise
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Comparing initial noise...")
print("=" * 60)

F_lat = (NUM_FRAMES - 1) // 8 + 1
H_lat = HEIGHT // 32
W_lat = WIDTH // 32

gen_orig = torch.Generator(device=DEVICE).manual_seed(SEED)
noise_orig = torch.randn(1, 128, F_lat, H_lat, W_lat, device=DEVICE, dtype=DTYPE, generator=gen_orig)

from diffusers.utils.torch_utils import randn_tensor
gen_ff = torch.Generator(device=DEVICE).manual_seed(SEED)
noise_ff = randn_tensor((1, 128, F_lat, H_lat, W_lat), generator=gen_ff, device=DEVICE, dtype=DTYPE)

_report("initial_noise", noise_orig, noise_ff)

# ---------------------------------------------------------------------------
# 6. Step-by-step denoising comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step-by-step denoising comparison...")
print("=" * 60)

# -- Prepare shared inputs --
_, _, ref_F, ref_H, ref_W = ref_latents.shape
ref_3d = patchifier.patchify(ref_latents)  # [1, seq_ref, 128]
seq_ref = ref_3d.shape[1]

ref_shape = VideoLatentShape(batch=1, channels=128, frames=ref_F, height=ref_H, width=ref_W)
target_shape = VideoLatentShape(batch=1, channels=128, frames=F_lat, height=H_lat, width=W_lat)

ref_positions = patchifier.get_patch_grid_bounds(ref_shape, device=DEVICE)
target_positions = patchifier.get_patch_grid_bounds(target_shape, device=DEVICE)

# Scale ref positions
if dsf != 1:
    ref_positions = ref_positions.clone()
    ref_positions[:, 1, ...] *= dsf
    ref_positions[:, 2, ...] *= dsf

all_positions = torch.cat([ref_positions, target_positions], dim=2)
all_positions = get_pixel_coords(all_positions, VIDEO_SCALE_FACTORS, causal_fix=True)
fps = 25.0
all_positions = all_positions.to(dtype=DTYPE)
all_positions[:, 0, ...] = all_positions[:, 0, ...] / fps

sigmas = orig_sigmas.to(DEVICE)

# -- Original pipeline denoising --
print("\nRunning ORIGINAL pipeline denoising...")
x0_model = X0Model(transformer)
stepper = EulerDiffusionStep(x0_model)

# Start from same noise
latents_orig = noise_orig.clone()
latents_orig_3d = patchifier.patchify(latents_orig)

for step_idx in range(len(sigmas) - 1):
    sigma = sigmas[step_idx]
    sigma_next = sigmas[step_idx + 1]

    # Build combined latent
    combined = torch.cat([ref_3d.to(DTYPE), latents_orig_3d], dim=1)
    seq_target = latents_orig_3d.shape[1]

    # Build timesteps
    ref_ts = torch.zeros(1, seq_ref, device=DEVICE, dtype=torch.float32)
    target_ts = torch.full((1, seq_target), sigma.item(), device=DEVICE, dtype=torch.float32)
    per_token_ts = torch.cat([ref_ts, target_ts], dim=1)

    modality = Modality(
        enabled=True,
        latent=combined,
        sigma=sigma.unsqueeze(0),
        timesteps=per_token_ts,
        positions=all_positions,
        context=prompt_embeds,
        context_mask=None,
    )

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
        # X0Model forward: get x0 prediction
        denoised_video, _ = x0_model(video=modality, audio=None, perturbations=None)

    # Apply denoise mask (ref tokens keep clean latent)
    # denoise_mask: 0 for ref, 1 for target
    denoise_mask = torch.cat([
        torch.zeros(1, seq_ref, 1, device=DEVICE, dtype=DTYPE),
        torch.ones(1, seq_target, 1, device=DEVICE, dtype=DTYPE),
    ], dim=1)
    clean_latent = torch.cat([ref_3d.to(DTYPE), torch.zeros_like(latents_orig_3d)], dim=1)
    denoised_masked = denoised_video * denoise_mask + clean_latent * (1 - denoise_mask)

    # Euler step on full combined (but we only care about target part)
    # velocity = (sample - denoised) / sigma
    velocity = (combined - denoised_masked) / sigma
    dt = sigma_next - sigma
    combined_next = combined + velocity * dt

    # Extract target portion
    latents_orig_3d = combined_next[:, seq_ref:]

    print(f"  Step {step_idx}: sigma={sigma.item():.4f} → {sigma_next.item():.4f}")

# Unpatchify original result
latents_orig_5d = patchifier.unpatchify(latents_orig_3d, target_shape)
print(f"  Original final latents: {latents_orig_5d.shape}")

# -- Flow-Factory pipeline denoising --
print("\nRunning FLOW-FACTORY pipeline denoising...")
latents_ff = noise_ff.clone()

ff_timesteps = ff_sched.timesteps  # sigma * 1000

for i in range(len(ff_timesteps)):
    t = ff_timesteps[i]
    t_next = (
        ff_timesteps[i + 1]
        if i + 1 < len(ff_timesteps)
        else torch.tensor(0.0, device=DEVICE, dtype=ff_timesteps.dtype)
    )

    sigma_ff = t / 1000.0

    # Patchify
    target_3d = patchifier.patchify(latents_ff)
    seq_target = target_3d.shape[1]
    combined_ff = torch.cat([ref_3d.to(DTYPE), target_3d], dim=1)

    # Build timesteps
    ref_ts = torch.zeros(1, seq_ref, device=DEVICE, dtype=torch.float32)
    target_ts = torch.full((1, seq_target), sigma_ff.item(), device=DEVICE, dtype=torch.float32)
    per_token_ts = torch.cat([ref_ts, target_ts], dim=1)

    modality = Modality(
        enabled=True,
        latent=combined_ff,
        sigma=sigma_ff.unsqueeze(0) if isinstance(sigma_ff, torch.Tensor) else torch.tensor([sigma_ff], device=DEVICE),
        timesteps=per_token_ts,
        positions=all_positions,
        context=prompt_embeds,
        context_mask=None,
    )

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
        video_pred, _ = transformer(video=modality, audio=None, perturbations=None)

    # Extract target velocity prediction
    target_v_pred_3d = video_pred[:, seq_ref:]
    v_pred_5d = patchifier.unpatchify(target_v_pred_3d, target_shape)

    # Scheduler step (ODE mode)
    output = ff_sched.step(
        noise_pred=v_pred_5d,
        timestep=t,
        latents=latents_ff,
        timestep_next=t_next,
        noise_level=0.0,
        compute_log_prob=False,
        return_dict=True,
        return_kwargs=["next_latents"],
    )

    latents_ff = output.next_latents
    print(f"  Step {i}: t={t.item():.1f} → {t_next.item():.1f}")

latents_ff_5d = latents_ff
print(f"  FF final latents: {latents_ff_5d.shape}")

# ---------------------------------------------------------------------------
# 7. Compare final latents
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

_report("final_latents", latents_orig_5d, latents_ff_5d)

# Check dtype consistency
print(f"\n  Original dtype: {latents_orig_5d.dtype}")
print(f"  FF dtype: {latents_ff_5d.dtype}")

# If not equal, find where divergence starts
if not torch.equal(latents_orig_5d, latents_ff_5d):
    print("\n  Investigating divergence source...")
    print("  Re-running step-by-step with per-step comparison...")

    # Re-run both with per-step comparison
    lat_o = noise_orig.clone()
    lat_f = noise_ff.clone()

    for step_idx in range(len(sigmas) - 1):
        sigma = sigmas[step_idx]
        sigma_next = sigmas[step_idx + 1]

        # Patchify both
        o_3d = patchifier.patchify(lat_o)
        f_3d = patchifier.patchify(lat_f)
        seq_t = o_3d.shape[1]

        _report(f"  step_{step_idx}_input_latents", lat_o, lat_f)

        # Original: combined + X0Model + denoise mask + Euler
        combined_o = torch.cat([ref_3d.to(DTYPE), o_3d], dim=1)
        ref_ts = torch.zeros(1, seq_ref, device=DEVICE, dtype=torch.float32)
        tgt_ts = torch.full((1, seq_t), sigma.item(), device=DEVICE, dtype=torch.float32)
        pts = torch.cat([ref_ts, tgt_ts], dim=1)

        mod_o = Modality(
            enabled=True, latent=combined_o, sigma=sigma.unsqueeze(0),
            timesteps=pts, positions=all_positions,
            context=prompt_embeds, context_mask=None,
        )

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
            denoised_o, _ = x0_model(video=mod_o, audio=None, perturbations=None)

        # Extract original's target x0 prediction
        orig_target_x0 = denoised_o[:, seq_ref:]

        # FF: velocity prediction
        combined_f = torch.cat([ref_3d.to(DTYPE), f_3d], dim=1)
        mod_f = Modality(
            enabled=True, latent=combined_f, sigma=sigma.unsqueeze(0),
            timesteps=pts, positions=all_positions,
            context=prompt_embeds, context_mask=None,
        )

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
            v_pred_f, _ = transformer(video=mod_f, audio=None, perturbations=None)

        ff_target_v = v_pred_f[:, seq_ref:]

        # Convert original x0 to velocity: v = (sample - x0) / sigma
        orig_target_v_from_x0 = (o_3d - orig_target_x0) / sigma
        _report(f"  step_{step_idx}_velocity_pred", orig_target_v_from_x0, ff_target_v)

        # Original Euler step on target
        dt = sigma_next - sigma
        lat_o_next_3d = o_3d + orig_target_v_from_x0 * dt
        lat_o = patchifier.unpatchify(lat_o_next_3d, target_shape)

        # FF scheduler step
        ff_v_5d = patchifier.unpatchify(ff_target_v, target_shape)
        t_val = sigma * 1000
        t_next_val = sigma_next * 1000
        out_f = ff_sched.step(
            noise_pred=ff_v_5d, timestep=t_val, latents=lat_f,
            timestep_next=t_next_val, noise_level=0.0,
            compute_log_prob=False, return_dict=True,
            return_kwargs=["next_latents"],
        )
        lat_f = out_f.next_latents

        _report(f"  step_{step_idx}_output_latents", lat_o, lat_f)

        if not torch.equal(lat_o, lat_f):
            print(f"\n  >>> DIVERGENCE at step {step_idx} <<<")
            break

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
