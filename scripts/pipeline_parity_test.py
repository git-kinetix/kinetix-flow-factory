#!/usr/bin/env python3
"""
Pipeline parity test: verify that Flow-Factory's inference logic produces
bitwise identical results to the original ICLoraPipeline (Stage 1).

Loads model components ONCE, then runs both pipelines' denoising logic
with the same seed and inputs. Compares step-by-step.

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
# Stage 1 resolution (original pipeline runs at HEIGHT//2, WIDTH//2)
STAGE1_H = 256
STAGE1_W = 384
NUM_FRAMES = 33
FRAME_RATE = 24.0
DEVICE = "cuda"
DTYPE = torch.bfloat16

# Distilled sigma schedule (from ltx_pipelines/utils/constants.py)
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

# Prompt encoding
with torch.no_grad():
    hidden_states, attention_mask = text_encoder.encode(PROMPT)
    enc_output = embeddings_processor.process_hidden_states(hidden_states, attention_mask)
prompt_embeds = enc_output.video_encoding
print(f"  prompt_embeds: {prompt_embeds.shape}")

# Reference video encoding
ref_h = STAGE1_H // dsf
ref_w = STAGE1_W // dsf
np_rng = np.random.RandomState(123)
ref_frames = np_rng.randint(0, 256, (NUM_FRAMES, ref_h, ref_w, 3), dtype=np.uint8)
ref_video_path = tempfile.mktemp(suffix=".mp4")
imageio.mimwrite(ref_video_path, ref_frames, fps=int(FRAME_RATE))
print(f"  Reference video: {ref_h}x{ref_w}, {NUM_FRAMES} frames")

# Load and encode reference video (same way as original pipeline)
from ltx_pipelines.utils.media_io import load_video_conditioning
ref_video_tensor = load_video_conditioning(
    video_path=ref_video_path,
    height=ref_h,
    width=ref_w,
    frame_cap=NUM_FRAMES,
    dtype=DTYPE,
    device=torch.device(DEVICE),
)
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
    ref_latents = vae_encoder(ref_video_tensor)
print(f"  ref_latents: {ref_latents.shape}")
os.unlink(ref_video_path)

# Latent dimensions
F_lat = (NUM_FRAMES - 1) // 8 + 1
H_lat = STAGE1_H // 32
W_lat = STAGE1_W // 32
print(f"  Target latent: F={F_lat}, H={H_lat}, W={W_lat}")

sigmas = torch.tensor(DISTILLED_SIGMAS, device=DEVICE, dtype=torch.float32)

# ---------------------------------------------------------------------------
# 3. ORIGINAL PATH: Use ltx_core tools to set up and run denoising
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Running ORIGINAL path (ltx_core tools)...")
print("=" * 60)

# Create the noised state using the exact original pipeline tools
output_shape = VideoPixelShape(
    batch=1, frames=NUM_FRAMES, width=STAGE1_W, height=STAGE1_H, fps=FRAME_RATE,
)
from ltx_pipelines.utils.types import PipelineComponents
components = PipelineComponents(dtype=DTYPE, device=torch.device(DEVICE))

video_latent_shape = VideoLatentShape.from_pixel_shape(
    shape=output_shape,
    latent_channels=components.video_latent_channels,
    scale_factors=components.video_scale_factors,
)
video_tools = VideoLatentTools(components.video_patchifier, video_latent_shape, output_shape.fps)

# Create initial state (empty target)
orig_gen = torch.Generator(device=DEVICE).manual_seed(SEED)
state = video_tools.create_initial_state(DEVICE, DTYPE)

# Apply conditioning (appends ref tokens)
cond = VideoConditionByReferenceLatent(
    latent=ref_latents,
    downscale_factor=dsf,
    strength=1.0,
)
state = cond.apply_to(latent_state=state, latent_tools=video_tools)

print(f"  State after conditioning:")
print(f"    latent: {state.latent.shape}, dtype={state.latent.dtype}")
print(f"    denoise_mask: {state.denoise_mask.shape}")
print(f"    positions: {state.positions.shape}, dtype={state.positions.dtype}")

# Noise using GaussianNoiser (same generator as original pipeline)
noiser = GaussianNoiser(generator=orig_gen)
state = noiser(state, noise_scale=1.0)

# Run denoising loop
stepper = EulerDiffusionStep()

from ltx_pipelines.utils.helpers import (
    modality_from_latent_state, post_process_latent,
)
from ltx_core.model.transformer import X0Model
from dataclasses import replace

x0_model = X0Model(transformer)

# Save initial state for later comparison
initial_noised_state = state

orig_state = state
orig_step0_velocity = None
for step_idx in range(len(sigmas) - 1):
    sigma = sigmas[step_idx]

    mod = modality_from_latent_state(orig_state, prompt_embeds, sigma)

    # Capture step 0 diagnostics
    if step_idx == 0:
        print(f"\n  --- Step 0 Diagnostics (ORIGINAL) ---")
        print(f"  mod.latent: {mod.latent.shape}, dtype={mod.latent.dtype}")
        print(f"  mod.sigma: {mod.sigma.shape}, dtype={mod.sigma.dtype}, val={mod.sigma}")
        print(f"  mod.timesteps: {mod.timesteps.shape}, dtype={mod.timesteps.dtype}")
        print(f"  mod.positions: {mod.positions.shape}, dtype={mod.positions.dtype}")
        print(f"  mod.context: {mod.context.shape}")
        print(f"  mod.attention_mask: {mod.attention_mask}")
        # Also get raw velocity for comparison
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
            orig_step0_velocity, _ = transformer(video=mod, audio=None, perturbations=None)
        print(f"  raw velocity norm: {orig_step0_velocity.float().norm().item():.4f}")

    # Pass audio=None (matches our FF forward() which also uses audio=None)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
        denoised_video, _ = x0_model(video=mod, audio=None, perturbations=None)

    denoised_video = post_process_latent(denoised_video, orig_state.denoise_mask, orig_state.clean_latent)

    new_latent = stepper.step(
        sample=orig_state.latent,
        denoised_sample=denoised_video,
        sigmas=sigmas,
        step_index=step_idx,
    )
    orig_state = replace(orig_state, latent=new_latent)

    if step_idx < 2 or step_idx == len(sigmas) - 2:
        print(f"  Step {step_idx}: sigma={sigma.item():.6f}, latent norm={new_latent.float().norm().item():.2f}")

# Extract target tokens and unpatchify
orig_final_state = video_tools.clear_conditioning(orig_state)
orig_final_state = video_tools.unpatchify(orig_final_state)
orig_final_latent = orig_final_state.latent
print(f"  Original final latent: {orig_final_latent.shape}")

# ---------------------------------------------------------------------------
# 4. FLOW-FACTORY PATH: Replicate our inference() logic
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Running FLOW-FACTORY path (our inference logic)...")
print("=" * 60)

# Replicate the noise generation from our fixed inference() method
ff_gen = torch.Generator(device=DEVICE).manual_seed(SEED)

target_shape = VideoLatentShape(
    batch=1, channels=128, frames=F_lat, height=H_lat, width=W_lat,
)

# Create empty target in 3D form (same as create_initial_state)
target_zeros_5d = torch.zeros(1, 128, F_lat, H_lat, W_lat, device=DEVICE, dtype=DTYPE)
target_zeros_3d = patchifier.patchify(target_zeros_5d)
seq_target = target_zeros_3d.shape[1]

ref_3d = patchifier.patchify(ref_latents)
seq_ref = ref_3d.shape[1]

# Combined [target | ref] in 3D (matching our fixed token order)
combined_3d = torch.cat([target_zeros_3d, ref_3d.to(DTYPE)], dim=1)

# Generate noise for full combined state
noise_3d = torch.randn(
    *combined_3d.shape,
    device=DEVICE, dtype=DTYPE,
    generator=ff_gen,
)

# Apply denoise mask
denoise_mask_3d = torch.cat([
    torch.ones(1, seq_target, 1, device=DEVICE, dtype=DTYPE),
    torch.zeros(1, seq_ref, 1, device=DEVICE, dtype=DTYPE),
], dim=1)
noised_3d = noise_3d * denoise_mask_3d + combined_3d * (1 - denoise_mask_3d)

# Extract target and unpatchify
target_noised_3d = noised_3d[:, :seq_target]
ff_latents = patchifier.unpatchify(target_noised_3d, target_shape)

# Compare initial noise with original
# Original target noise: first seq_target tokens of orig_state.latent (before denoising started)
orig_target_3d = state.latent[:, :seq_target]  # state still has the initial noised state
orig_target_5d = patchifier.unpatchify(orig_target_3d, target_shape)
_report("initial_noise", orig_target_5d, ff_latents)

# Set up our scheduler
ff_sched = LTXSDEScheduler(
    num_inference_steps=8, dynamics_type="ODE", noise_level=0.0,
    use_distilled=True,
)
ff_sched.set_timesteps(8, device=DEVICE)

_report("sigma_schedule", sigmas, ff_sched.sigmas)

ff_timesteps = ff_sched.timesteps

# Build positions matching our fixed forward()
target_positions = patchifier.get_patch_grid_bounds(target_shape, device=DEVICE)
target_positions = get_pixel_coords(target_positions, VIDEO_SCALE_FACTORS, causal_fix=True).float()
target_positions[:, 0, ...] = target_positions[:, 0, ...] / FRAME_RATE
target_positions = target_positions.to(dtype=torch.bfloat16)

_, _, ref_F, ref_H, ref_W = ref_latents.shape
ref_shape_vls = VideoLatentShape(batch=1, channels=128, frames=ref_F, height=ref_H, width=ref_W)
ref_positions = patchifier.get_patch_grid_bounds(ref_shape_vls, device=DEVICE)
ref_positions = get_pixel_coords(ref_positions, VIDEO_SCALE_FACTORS, causal_fix=True)
ref_positions = ref_positions.to(dtype=torch.float32)
ref_positions[:, 0, ...] = ref_positions[:, 0, ...] / FRAME_RATE
if dsf != 1:
    ref_positions = ref_positions.clone()
    ref_positions[:, 1, ...] *= dsf
    ref_positions[:, 2, ...] *= dsf

positions = torch.cat([target_positions, ref_positions], dim=2)

# Compare positions with original
_report("positions", state.positions, positions)

# Denoising loop (our forward() logic)
for step_idx in range(len(sigmas) - 1):
    sigma = sigmas[step_idx]
    sigma_next = sigmas[step_idx + 1]

    # Build combined latent [target | ref]
    target_3d = patchifier.patchify(ff_latents)
    combined = torch.cat([target_3d, ref_3d.to(DTYPE)], dim=1)

    # Per-token timesteps [target | ref]
    target_ts = torch.full((1, seq_target, 1), sigma.item(), device=DEVICE, dtype=torch.float32)
    ref_ts = torch.zeros(1, seq_ref, 1, device=DEVICE, dtype=torch.float32)
    per_token_ts = torch.cat([target_ts, ref_ts], dim=1)

    # Build Modality (match sigma shape with original: pass as-is, not unsqueezed)
    mod = Modality(
        enabled=True, latent=combined,
        sigma=sigma.unsqueeze(0), timesteps=per_token_ts,
        positions=positions, context=prompt_embeds, context_mask=None,
    )

    if step_idx == 0:
        print(f"\n  --- Step 0 Diagnostics (FF) ---")
        print(f"  mod.latent: {mod.latent.shape}, dtype={mod.latent.dtype}")
        print(f"  mod.sigma: {mod.sigma.shape}, dtype={mod.sigma.dtype}, val={mod.sigma}")
        print(f"  mod.timesteps: {mod.timesteps.shape}, dtype={mod.timesteps.dtype}")
        print(f"  mod.positions: {mod.positions.shape}, dtype={mod.positions.dtype}")
        print(f"  mod.context: {mod.context.shape}")
        print(f"  mod.attention_mask: {mod.attention_mask}")
        # Compare Modality fields with original step 0
        orig_mod = modality_from_latent_state(initial_noised_state, prompt_embeds, sigmas[0])
        _report("step0_latent", orig_mod.latent, mod.latent)
        _report("step0_timesteps", orig_mod.timesteps, mod.timesteps)
        _report("step0_context", orig_mod.context, mod.context)

    # Raw transformer forward (velocity model, not X0Model)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
        velocity_pred, _ = transformer(video=mod, audio=None, perturbations=None)

    if step_idx == 0 and orig_step0_velocity is not None:
        _report("step0_velocity", orig_step0_velocity, velocity_pred)
        print(f"  FF velocity norm: {velocity_pred.float().norm().item():.4f}")

    # Extract target velocity [target | ref] → first seq_target tokens
    target_v_3d = velocity_pred[:, :seq_target]
    target_v_5d = patchifier.unpatchify(target_v_3d, target_shape)

    # Scheduler step
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

    if step_idx < 2 or step_idx == len(sigmas) - 2:
        print(f"  Step {step_idx}: sigma={sigma.item():.6f}, latent norm={ff_latents.float().norm().item():.2f}")

print(f"  FF final latent: {ff_latents.shape}")

# ---------------------------------------------------------------------------
# 5. Final comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

_report("final_latents", orig_final_latent, ff_latents)
print(f"  dtype: orig={orig_final_latent.dtype}, ff={ff_latents.dtype}")
print(f"  shape: orig={orig_final_latent.shape}, ff={ff_latents.shape}")

if all_passed:
    print("\n  *** PIPELINE PARITY VERIFIED — BITWISE IDENTICAL ***")
else:
    print("\n  *** PARITY FAILED — SEE ABOVE ***")
    mse = ((orig_final_latent.float() - ff_latents.float()) ** 2).mean().item()
    print(f"  MSE = {mse:.2e}")

print("=" * 60)
sys.exit(0 if all_passed else 1)
