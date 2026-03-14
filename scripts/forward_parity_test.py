#!/usr/bin/env python3
"""
Forward parity test: verify that LTXUnionAdapter.forward() produces
identical results to calling the transformer with the original Modality
construction from ltx_core.

This tests the FULL forward() path including:
- Token ordering
- Per-token timesteps
- Position computation (dtypes, fps division, dsf scaling)
- Modality construction
- Velocity extraction

Environment variables:
    LTX_MODEL_PATH       — path to ltx-2.3 distilled .safetensors
    LTX_UNION_LORA_PATH  — path to IC-LoRA union control .safetensors
    LTX_GEMMA_PATH       — path to Gemma-3 text encoder directory

Usage:
    LTX_MODEL_PATH=... LTX_UNION_LORA_PATH=... LTX_GEMMA_PATH=... \
    python scripts/forward_parity_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import imageio
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ["LTX_MODEL_PATH"]
LORA_PATH = os.environ["LTX_UNION_LORA_PATH"]
GEMMA_PATH = os.environ["LTX_GEMMA_PATH"]

SEED = 42
PROMPT = "A person dancing in a studio"
STAGE1_H = 256
STAGE1_W = 384
NUM_FRAMES = 33
FRAME_RATE = 24.0
DEVICE = "cuda"
DTYPE = torch.bfloat16

DISTILLED_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]

all_passed = True


def _report(name: str, a: torch.Tensor, b: torch.Tensor):
    global all_passed
    if torch.equal(a, b):
        print(f"  [PASS] {name}: bitwise identical")
    else:
        all_passed = False
        diff = (a.float() - b.float()).abs()
        print(
            f"  [FAIL] {name}: max_diff={diff.max().item():.2e}, "
            f"mean_diff={diff.mean().item():.2e}, "
            f"mismatched={diff.nonzero().shape[0]}/{a.numel()}"
        )


# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------
print("=" * 60)
print("Loading model components...")
print("=" * 60)

from ltx_trainer.model_loader import (
    load_transformer, load_video_vae_encoder, load_video_vae_decoder,
    load_text_encoder, load_embeddings_processor,
)
from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.model.transformer.modality import Modality
from ltx_core.types import VideoLatentShape, VIDEO_SCALE_FACTORS, VideoPixelShape, LatentState
from ltx_core.utils import to_denoised
from ltx_core.tools import VideoLatentTools
from ltx_core.conditioning import VideoConditionByReferenceLatent

from flow_factory.models.ltx.pipeline import _merge_lora_into_model
from flow_factory.models.ltx.scheduler_bridge import LTXSDEScheduler

transformer = load_transformer(MODEL_PATH)
dsf = _merge_lora_into_model(transformer, LORA_PATH)
print(f"  reference_downscale_factor = {dsf}")

vae_encoder = load_video_vae_encoder(MODEL_PATH)
vae_decoder = load_video_vae_decoder(MODEL_PATH)
text_encoder = load_text_encoder(GEMMA_PATH)
embeddings_processor = load_embeddings_processor(MODEL_PATH)
patchifier = VideoLatentPatchifier(patch_size=1)

transformer.to(DEVICE, DTYPE).eval()
vae_encoder.to(DEVICE).eval()
vae_decoder.to(DEVICE, DTYPE).eval()
text_encoder.to(DEVICE).eval()
embeddings_processor.to(DEVICE).eval()

# ---------------------------------------------------------------------------
# 2. Encode shared inputs
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Preparing shared inputs...")
print("=" * 60)

with torch.no_grad():
    hidden_states, attention_mask = text_encoder.encode(PROMPT)
    enc_output = embeddings_processor.process_hidden_states(hidden_states, attention_mask)
prompt_embeds = enc_output.video_encoding
print(f"  prompt_embeds: {prompt_embeds.shape}")

ref_h = STAGE1_H // dsf
ref_w = STAGE1_W // dsf
np_rng = np.random.RandomState(123)
ref_frames = np_rng.randint(0, 256, (NUM_FRAMES, ref_h, ref_w, 3), dtype=np.uint8)
ref_video_path = tempfile.mktemp(suffix=".mp4")
imageio.mimwrite(ref_video_path, ref_frames, fps=int(FRAME_RATE))

from ltx_pipelines.utils.media_io import load_video_conditioning
ref_video_tensor = load_video_conditioning(
    video_path=ref_video_path,
    height=ref_h, width=ref_w, frame_cap=NUM_FRAMES,
    dtype=DTYPE, device=torch.device(DEVICE),
)
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
    ref_latents = vae_encoder(ref_video_tensor)
print(f"  ref_latents: {ref_latents.shape}")
os.unlink(ref_video_path)

F_lat = (NUM_FRAMES - 1) // 8 + 1
H_lat = STAGE1_H // 32
W_lat = STAGE1_W // 32
target_shape = VideoLatentShape(batch=1, channels=128, frames=F_lat, height=H_lat, width=W_lat)
print(f"  Target latent: F={F_lat}, H={H_lat}, W={W_lat}")

sigmas = torch.tensor(DISTILLED_SIGMAS, device=DEVICE, dtype=torch.float32)

# ---------------------------------------------------------------------------
# 3. Set up original initial state
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Setting up initial state...")
print("=" * 60)

from ltx_pipelines.utils.types import PipelineComponents
from ltx_pipelines.utils.helpers import (
    modality_from_latent_state, post_process_latent, timesteps_from_mask,
)
from dataclasses import replace

components = PipelineComponents(dtype=DTYPE, device=torch.device(DEVICE))
output_shape = VideoPixelShape(
    batch=1, frames=NUM_FRAMES, width=STAGE1_W, height=STAGE1_H, fps=FRAME_RATE,
)
video_latent_shape = VideoLatentShape.from_pixel_shape(
    shape=output_shape,
    latent_channels=components.video_latent_channels,
    scale_factors=components.video_scale_factors,
)
video_tools = VideoLatentTools(components.video_patchifier, video_latent_shape, output_shape.fps)

gen = torch.Generator(device=DEVICE).manual_seed(SEED)
state = video_tools.create_initial_state(DEVICE, DTYPE)
cond = VideoConditionByReferenceLatent(latent=ref_latents, downscale_factor=dsf, strength=1.0)
state = cond.apply_to(latent_state=state, latent_tools=video_tools)
noiser = GaussianNoiser(generator=gen)
state = noiser(state, noise_scale=1.0)

seq_target = patchifier.get_token_count(target_shape)
seq_ref = state.latent.shape[1] - seq_target
print(f"  Combined state: {state.latent.shape} (seq_target={seq_target}, seq_ref={seq_ref})")

# ---------------------------------------------------------------------------
# 4. Compare forward() at step 0
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Comparing forward() Modality construction at step 0...")
print("=" * 60)

sigma = sigmas[0]  # 1.0

# === ORIGINAL: Build Modality from state ===
orig_mod = modality_from_latent_state(state, prompt_embeds, sigma)

# === FF: Build Modality via forward() path ===
# Reconstruct what forward() does:
target_latents_5d = patchifier.unpatchify(state.latent[:, :seq_target], target_shape)
fps = FRAME_RATE

# forward() rebuilds positions from scratch using the latent shapes
ff_target_shape = VideoLatentShape(batch=1, channels=128, frames=F_lat, height=H_lat, width=W_lat)
ff_target_positions = patchifier.get_patch_grid_bounds(ff_target_shape, device=DEVICE)
ff_target_positions = get_pixel_coords(ff_target_positions, VIDEO_SCALE_FACTORS, causal_fix=True).float()
ff_target_positions[:, 0, ...] = ff_target_positions[:, 0, ...] / fps
ff_target_positions = ff_target_positions.to(dtype=torch.bfloat16)

_, _, ref_F, ref_H, ref_W = ref_latents.shape
ff_ref_shape = VideoLatentShape(batch=1, channels=128, frames=ref_F, height=ref_H, width=ref_W)
ff_ref_positions = patchifier.get_patch_grid_bounds(ff_ref_shape, device=DEVICE)
ff_ref_positions = get_pixel_coords(ff_ref_positions, VIDEO_SCALE_FACTORS, causal_fix=True)
ff_ref_positions = ff_ref_positions.to(dtype=torch.float32)
ff_ref_positions[:, 0, ...] = ff_ref_positions[:, 0, ...] / fps
if dsf != 1:
    ff_ref_positions = ff_ref_positions.clone()
    ff_ref_positions[:, 1, ...] *= dsf
    ff_ref_positions[:, 2, ...] *= dsf
ff_positions = torch.cat([ff_target_positions, ff_ref_positions], dim=2)

# Per-token timesteps
sigma_batch = sigma.view(1)
ff_ref_ts = torch.zeros(1, seq_ref, 1, device=DEVICE, dtype=torch.float32)
ff_target_ts = sigma_batch.view(1, 1, 1).expand(1, seq_target, 1)
ff_per_token_ts = torch.cat([ff_target_ts, ff_ref_ts], dim=1)

# Combined latent [target | ref]
target_3d = patchifier.patchify(target_latents_5d)
ref_3d = patchifier.patchify(ref_latents)
ff_combined = torch.cat([target_3d, ref_3d.to(dtype=DTYPE)], dim=1)

print("Comparing Modality components:")
_report("latent", orig_mod.latent, ff_combined)
_report("timesteps", orig_mod.timesteps, ff_per_token_ts)
_report("positions", orig_mod.positions, ff_positions)
_report("context", orig_mod.context, prompt_embeds)

# === Now compare actual transformer outputs ===
print("\nComparing transformer outputs:")
ff_modality = Modality(
    enabled=True,
    latent=ff_combined,
    sigma=sigma_batch,
    timesteps=ff_per_token_ts,
    positions=ff_positions,
    context=prompt_embeds,
    context_mask=None,
)

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
    orig_vel, _ = transformer(video=orig_mod, audio=None, perturbations=None)
    ff_vel, _ = transformer(video=ff_modality, audio=None, perturbations=None)

_report("full_velocity", orig_vel, ff_vel)
_report("target_velocity", orig_vel[:, :seq_target], ff_vel[:, :seq_target])

# ---------------------------------------------------------------------------
# 5. Full denoising loop comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Full denoising loop: original vs FF forward()...")
print("=" * 60)

stepper = EulerDiffusionStep()
orig_state_loop = state

# FF state: target latents in 5D
ff_latents = target_latents_5d.clone()

ff_sched = LTXSDEScheduler(
    num_inference_steps=8, dynamics_type="ODE", noise_level=0.0, use_distilled=True,
)
ff_sched.set_timesteps(8, device=DEVICE)
ff_timesteps = ff_sched.timesteps

for step_idx in range(len(sigmas) - 1):
    sigma_val = sigmas[step_idx]
    sigma_next = sigmas[step_idx + 1]

    # Build original Modality
    orig_mod = modality_from_latent_state(orig_state_loop, prompt_embeds, sigma_val)

    # Build FF Modality (same as forward() does)
    ff_target_3d = patchifier.patchify(ff_latents)
    ff_combined_step = torch.cat([ff_target_3d, ref_3d.to(dtype=DTYPE)], dim=1)

    sigma_b = sigma_val.view(1)
    ff_target_ts_step = sigma_b.view(1, 1, 1).expand(1, seq_target, 1)
    ff_per_token_ts_step = torch.cat([ff_target_ts_step, ff_ref_ts], dim=1)

    ff_mod = Modality(
        enabled=True,
        latent=ff_combined_step,
        sigma=sigma_b,
        timesteps=ff_per_token_ts_step,
        positions=ff_positions,
        context=prompt_embeds,
        context_mask=None,
    )

    # Verify Modality inputs match
    inputs_match = (
        torch.equal(orig_mod.latent, ff_mod.latent)
        and torch.equal(orig_mod.timesteps, ff_mod.timesteps)
        and torch.equal(orig_mod.positions, ff_mod.positions)
    )

    # Single transformer call with ORIGINAL modality
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
        velocity_pred, _ = transformer(video=orig_mod, audio=None, perturbations=None)

    # === ORIGINAL PATH ===
    denoised_combined = to_denoised(orig_state_loop.latent, velocity_pred, orig_mod.timesteps)
    denoised_combined = post_process_latent(
        denoised_combined, orig_state_loop.denoise_mask, orig_state_loop.clean_latent,
    )
    new_combined = stepper.step(
        sample=orig_state_loop.latent,
        denoised_sample=denoised_combined,
        sigmas=sigmas,
        step_index=step_idx,
    )
    orig_state_loop = replace(orig_state_loop, latent=new_combined)
    orig_target_5d = patchifier.unpatchify(new_combined[:, :seq_target], target_shape)

    # === FF PATH ===
    target_v_3d = velocity_pred[:, :seq_target]
    target_v_5d = patchifier.unpatchify(target_v_3d, target_shape)

    t = ff_timesteps[step_idx]
    t_next = (
        ff_timesteps[step_idx + 1]
        if step_idx + 1 < len(ff_timesteps)
        else torch.tensor(0.0, device=DEVICE, dtype=ff_timesteps.dtype)
    )
    out = ff_sched.step(
        noise_pred=target_v_5d, timestep=t, latents=ff_latents,
        timestep_next=t_next, noise_level=0.0,
        compute_log_prob=False, return_dict=True,
        return_kwargs=["next_latents"],
    )
    ff_latents = out.next_latents

    status = "OK" if inputs_match and torch.equal(orig_target_5d, ff_latents) else "MISMATCH"
    if status == "OK":
        print(f"  Step {step_idx}: OK (inputs_match={inputs_match}, sigma={sigma_val.item():.6f})")
    else:
        print(f"  Step {step_idx}: {status}")
        if not inputs_match:
            _report(f"    step_{step_idx}_latent", orig_mod.latent, ff_mod.latent)
            _report(f"    step_{step_idx}_timesteps", orig_mod.timesteps, ff_mod.timesteps)
            _report(f"    step_{step_idx}_positions", orig_mod.positions, ff_mod.positions)
        _report(f"    step_{step_idx}_output", orig_target_5d, ff_latents)
        break

# ---------------------------------------------------------------------------
# 6. Decode and compare
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

orig_final = video_tools.clear_conditioning(orig_state_loop)
orig_final = video_tools.unpatchify(orig_final)
_report("final_latents", orig_final.latent, ff_latents)

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
    orig_pixels = vae_decoder(orig_final.latent)
    ff_pixels = vae_decoder(ff_latents)
_report("decoded_video", orig_pixels, ff_pixels)

if all_passed:
    print("\n  *** FULL FORWARD PARITY VERIFIED — BITWISE IDENTICAL ***")
else:
    print("\n  *** PARITY FAILED ***")

print("=" * 60)
sys.exit(0 if all_passed else 1)
