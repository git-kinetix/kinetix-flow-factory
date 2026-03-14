#!/usr/bin/env python3
"""
Pipeline parity test: verify that Flow-Factory's denoising logic produces
bitwise identical results to the original ICLoraPipeline (Stage 1).

Uses a SINGLE transformer call per step, then applies both the original
ltx_core path (to_denoised + post_process + EulerStep on combined state)
and our FF path (extract target velocity + scheduler.step on target only).

This eliminates CUDA non-determinism from the comparison.

Environment variables:
    LTX_MODEL_PATH       — path to ltx-2.3 distilled .safetensors
    LTX_UNION_LORA_PATH  — path to IC-LoRA union control .safetensors
    LTX_GEMMA_PATH       — path to Gemma-3 text encoder directory

Usage:
    LTX_MODEL_PATH=... LTX_UNION_LORA_PATH=... LTX_GEMMA_PATH=... \
    python scripts/pipeline_parity_test.py
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
# 1. Load shared model components
# ---------------------------------------------------------------------------
print("=" * 60)
print("Loading model components...")
print("=" * 60)

from ltx_trainer.model_loader import (
    load_transformer, load_video_vae_encoder,
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
text_encoder = load_text_encoder(GEMMA_PATH)
embeddings_processor = load_embeddings_processor(MODEL_PATH)
patchifier = VideoLatentPatchifier(patch_size=1)

transformer.to(DEVICE, DTYPE).eval()
vae_encoder.to(DEVICE).eval()
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
# 3. Set up ORIGINAL initial state using ltx_core tools
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Setting up initial state (ltx_core tools)...")
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

orig_gen = torch.Generator(device=DEVICE).manual_seed(SEED)
state = video_tools.create_initial_state(DEVICE, DTYPE)

cond = VideoConditionByReferenceLatent(latent=ref_latents, downscale_factor=dsf, strength=1.0)
state = cond.apply_to(latent_state=state, latent_tools=video_tools)

noiser = GaussianNoiser(generator=orig_gen)
state = noiser(state, noise_scale=1.0)

seq_target = patchifier.get_token_count(target_shape)
seq_ref = state.latent.shape[1] - seq_target

print(f"  Combined state: {state.latent.shape} (seq_target={seq_target}, seq_ref={seq_ref})")
print(f"  Positions: {state.positions.shape}, dtype={state.positions.dtype}")

# ---------------------------------------------------------------------------
# 4. Set up FF initial noise (same generator pattern)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Setting up FF initial state...")
print("=" * 60)

ff_gen = torch.Generator(device=DEVICE).manual_seed(SEED)
ref_3d = patchifier.patchify(ref_latents)

target_zeros_5d = torch.zeros(1, 128, F_lat, H_lat, W_lat, device=DEVICE, dtype=DTYPE)
target_zeros_3d = patchifier.patchify(target_zeros_5d)
combined_3d = torch.cat([target_zeros_3d, ref_3d.to(DTYPE)], dim=1)
noise_3d = torch.randn(*combined_3d.shape, device=DEVICE, dtype=DTYPE, generator=ff_gen)
denoise_mask_3d = torch.cat([
    torch.ones(1, seq_target, 1, device=DEVICE, dtype=DTYPE),
    torch.zeros(1, seq_ref, 1, device=DEVICE, dtype=DTYPE),
], dim=1)
noised_3d = noise_3d * denoise_mask_3d + combined_3d * (1 - denoise_mask_3d)
ff_latents = patchifier.unpatchify(noised_3d[:, :seq_target], target_shape)

# Verify initial noise matches
orig_target_5d = patchifier.unpatchify(state.latent[:, :seq_target], target_shape)
_report("initial_noise", orig_target_5d, ff_latents)

# FF positions
target_positions = patchifier.get_patch_grid_bounds(target_shape, device=DEVICE)
target_positions = get_pixel_coords(target_positions, VIDEO_SCALE_FACTORS, causal_fix=True).float()
target_positions[:, 0, ...] /= FRAME_RATE
target_positions = target_positions.to(dtype=torch.bfloat16)

_, _, ref_F, ref_H, ref_W = ref_latents.shape
ref_shape_vls = VideoLatentShape(batch=1, channels=128, frames=ref_F, height=ref_H, width=ref_W)
ref_positions = patchifier.get_patch_grid_bounds(ref_shape_vls, device=DEVICE)
ref_positions = get_pixel_coords(ref_positions, VIDEO_SCALE_FACTORS, causal_fix=True).to(torch.float32)
ref_positions[:, 0, ...] /= FRAME_RATE
if dsf != 1:
    ref_positions = ref_positions.clone()
    ref_positions[:, 1, ...] *= dsf
    ref_positions[:, 2, ...] *= dsf
ff_positions = torch.cat([target_positions, ref_positions], dim=2)

_report("positions", state.positions, ff_positions)

# FF scheduler
ff_sched = LTXSDEScheduler(
    num_inference_steps=8, dynamics_type="ODE", noise_level=0.0, use_distilled=True,
)
ff_sched.set_timesteps(8, device=DEVICE)
_report("sigma_schedule", sigmas, ff_sched.sigmas)

# ---------------------------------------------------------------------------
# 5. Step-by-step: SINGLE transformer call, BOTH step computations
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step-by-step comparison (single transformer call per step)...")
print("=" * 60)

stepper = EulerDiffusionStep()
orig_state = state
ff_timesteps = ff_sched.timesteps

for step_idx in range(len(sigmas) - 1):
    sigma = sigmas[step_idx]
    sigma_next = sigmas[step_idx + 1]

    # Verify inputs match
    orig_target_3d = orig_state.latent[:, :seq_target]
    orig_target_5d = patchifier.unpatchify(orig_target_3d, target_shape)
    _report(f"step_{step_idx}_input", orig_target_5d, ff_latents)

    # Build combined latent for both paths (use orig_state directly)
    # For the FF path, we also build the combined state
    ff_target_3d = patchifier.patchify(ff_latents)
    ff_combined = torch.cat([ff_target_3d, ref_3d.to(DTYPE)], dim=1)

    # Build Modality using original state (positions, denoise_mask, etc.)
    mod = modality_from_latent_state(orig_state, prompt_embeds, sigma)

    # SINGLE transformer call (velocity model)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
        velocity_pred, _ = transformer(video=mod, audio=None, perturbations=None)

    # ===== ORIGINAL PATH =====
    # X0Model: to_denoised(latent, velocity, timesteps)
    denoised_combined = to_denoised(orig_state.latent, velocity_pred, mod.timesteps)
    denoised_combined = post_process_latent(
        denoised_combined, orig_state.denoise_mask, orig_state.clean_latent,
    )
    new_combined = stepper.step(
        sample=orig_state.latent,
        denoised_sample=denoised_combined,
        sigmas=sigmas,
        step_index=step_idx,
    )
    orig_state = replace(orig_state, latent=new_combined)
    orig_target_5d = patchifier.unpatchify(new_combined[:, :seq_target], target_shape)

    # ===== FF PATH =====
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

    _report(f"step_{step_idx}_output", orig_target_5d, ff_latents)

    if not torch.equal(orig_target_5d, ff_latents):
        # Diagnose
        print(f"  >>> DIVERGENCE at step {step_idx}")
        print(f"      sigma={sigma.item():.6f} → {sigma_next.item():.6f}")

        # Manual euler on target only (same as FF should compute)
        denoised_target_3d = denoised_combined[:, :seq_target]
        denoised_target_5d = patchifier.unpatchify(denoised_target_3d, target_shape)
        _report(f"  denoised_target", denoised_target_5d,
                (ff_latents_prev.float() - target_v_5d.float() * sigma.float()).to(DTYPE)
                if 'ff_latents_prev' in dir() else denoised_target_5d)

        break

    ff_latents_prev = ff_latents.clone()
    print(f"  Step {step_idx}: OK (sigma={sigma.item():.6f})")

# ---------------------------------------------------------------------------
# 6. Final
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

orig_final = video_tools.clear_conditioning(orig_state)
orig_final = video_tools.unpatchify(orig_final)
_report("final_latents", orig_final.latent, ff_latents)

if all_passed:
    print("\n  *** PIPELINE PARITY VERIFIED — BITWISE IDENTICAL ***")
else:
    mse = ((orig_final.latent.float() - ff_latents.float()) ** 2).mean().item()
    print(f"\n  *** PARITY FAILED — MSE = {mse:.2e} ***")

print("=" * 60)
sys.exit(0 if all_passed else 1)
