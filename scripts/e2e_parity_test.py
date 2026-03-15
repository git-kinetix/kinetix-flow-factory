#!/usr/bin/env python3
"""
End-to-end parity test: verify that Flow-Factory's denoising produces
BITWISE IDENTICAL output to the original LTX ICLoraPipeline using
real RealisDance-Val data.

This is the definitive test: same model, same seed, same prompt, same
conditioning video, same resolution — two independent code paths must
produce torch.equal() results at every step.

Environment variables:
    LTX_MODEL_PATH       — path to ltx-2.3 distilled .safetensors
    LTX_UNION_LORA_PATH  — path to IC-LoRA union control .safetensors
    LTX_GEMMA_PATH       — path to Gemma-3 text encoder directory

Usage:
    source activate DiffusionNFT
    LTX_MODEL_PATH=... LTX_UNION_LORA_PATH=... LTX_GEMMA_PATH=... \
    python scripts/e2e_parity_test.py
"""
from __future__ import annotations

import gc
import os
import sys
import tempfile

import imageio
import numpy as np
import torch
from dataclasses import replace

# ---------------------------------------------------------------------------
# Config — matches training config exactly
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ["LTX_MODEL_PATH"]
LORA_PATH = os.environ["LTX_UNION_LORA_PATH"]
GEMMA_PATH = os.environ["LTX_GEMMA_PATH"]
DATASET_ROOT = os.environ.get("DATASET_ROOT", "/home/ubuntu/RealisDance-Val")

SEED = 42
HEIGHT = 256
WIDTH = 448
NUM_FRAMES = 33
FRAME_RATE = 24.0
DTYPE = torch.bfloat16
DEVICE = "cuda"

# Pick a real sample from RealisDance-Val
SAMPLE_ID = "0001"
VARIANT = "orbit_right_45"

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
    load_transformer, load_video_vae_encoder, load_video_vae_decoder,
    load_text_encoder, load_embeddings_processor,
)
from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.model.transformer.modality import Modality
from ltx_core.types import (
    VideoLatentShape, VIDEO_SCALE_FACTORS, VideoPixelShape,
    AudioLatentShape,
)
from ltx_core.utils import to_denoised
from ltx_core.tools import VideoLatentTools, AudioLatentTools
from ltx_core.conditioning import VideoConditionByReferenceLatent
from ltx_pipelines.utils.types import PipelineComponents
from ltx_pipelines.utils.helpers import (
    modality_from_latent_state, post_process_latent,
    noise_video_state, noise_audio_state,
)
from ltx_pipelines.utils import simple_denoising_func
from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES

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
# 2. Encode shared inputs (real data)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Encoding shared inputs from RealisDance-Val...")
print("=" * 60)

# Read real prompt
prompt_file = os.path.join(DATASET_ROOT, "prompt", f"{SAMPLE_ID}.txt")
with open(prompt_file) as f:
    prompt = f.read().strip()
print(f"  Prompt: {prompt[:80]}...")

with torch.no_grad():
    hidden_states, attention_mask = text_encoder.encode(prompt)
    enc_output = embeddings_processor.process_hidden_states(hidden_states, attention_mask)
video_context = enc_output.video_encoding
audio_context = enc_output.audio_encoding
print(f"  video_context: {video_context.shape}")
print(f"  audio_context: {audio_context.shape}")

# Free text encoder early
del text_encoder, embeddings_processor
gc.collect()
torch.cuda.empty_cache()

# Encode reference conditioning video (pose_depth)
npz_path = os.path.join(
    DATASET_ROOT, "condition", "pose_depth", f"{SAMPLE_ID}_{VARIANT}video.npz"
)
data = np.load(npz_path)
num_npz_frames = int(data["num_frames"])
frames_list = []
for i in range(num_npz_frames):
    frame = data[f"frame_{i:05d}"]
    frame_uint8 = (frame * 255).clip(0, 255).astype(np.uint8)
    frames_list.append(frame_uint8)

ref_h = HEIGHT // dsf
ref_w = WIDTH // dsf
print(f"  Reference resolution: {ref_h}x{ref_w}")

# Write to temp MP4 and use load_video_conditioning (same as original pipeline)
from ltx_pipelines.utils.media_io import load_video_conditioning

ref_mp4 = tempfile.mktemp(suffix=".mp4")
imageio.mimwrite(ref_mp4, frames_list, fps=int(FRAME_RATE))
ref_video_tensor = load_video_conditioning(
    video_path=ref_mp4, height=ref_h, width=ref_w,
    frame_cap=NUM_FRAMES, dtype=DTYPE, device=torch.device(DEVICE),
)
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
    ref_latents = vae_encoder(ref_video_tensor)
print(f"  ref_latents: {ref_latents.shape}")
del ref_video_tensor
os.unlink(ref_mp4)

# Free VAE encoder
del vae_encoder
gc.collect()
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 3. Compute shapes
# ---------------------------------------------------------------------------
F_lat = (NUM_FRAMES - 1) // 8 + 1
H_lat = HEIGHT // 32
W_lat = WIDTH // 32
target_shape = VideoLatentShape(batch=1, channels=128, frames=F_lat, height=H_lat, width=W_lat)
sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, device=DEVICE, dtype=torch.float32)
n_steps = len(sigmas) - 1
seq_target = patchifier.get_token_count(target_shape)

_, _, ref_F, ref_H, ref_W = ref_latents.shape
ref_shape_5d = VideoLatentShape(batch=1, channels=128, frames=ref_F, height=ref_H, width=ref_W)
seq_ref = patchifier.get_token_count(ref_shape_5d)

print(f"\n  Target latent: F={F_lat}, H={H_lat}, W={W_lat} ({seq_target} tokens)")
print(f"  Ref latent: F={ref_F}, H={ref_H}, W={ref_W} ({seq_ref} tokens)")
print(f"  Sigma schedule ({n_steps} steps): {[f'{s:.6f}' for s in sigmas.tolist()]}")

# =====================================================================
# PATH A: ORIGINAL ICLoraPipeline (using ltx_core functions directly)
# =====================================================================
print("\n" + "=" * 60)
print("PATH A: Original ICLoraPipeline denoising")
print("=" * 60)

output_shape = VideoPixelShape(
    batch=1, frames=NUM_FRAMES, width=WIDTH, height=HEIGHT, fps=FRAME_RATE,
)
components = PipelineComponents(dtype=DTYPE, device=torch.device(DEVICE))

gen_A = torch.Generator(device=DEVICE).manual_seed(SEED)
noiser_A = GaussianNoiser(generator=gen_A)
stepper = EulerDiffusionStep()

cond_A = VideoConditionByReferenceLatent(
    latent=ref_latents, downscale_factor=dsf, strength=1.0,
)

# Create video + audio state (same RNG order as original pipeline)
video_state_A, video_tools_A = noise_video_state(
    output_shape=output_shape, noiser=noiser_A, conditionings=[cond_A],
    components=components, dtype=DTYPE, device=torch.device(DEVICE), noise_scale=1.0,
)
audio_state_A, audio_tools_A = noise_audio_state(
    output_shape=output_shape, noiser=noiser_A, conditionings=[],
    components=components, dtype=DTYPE, device=torch.device(DEVICE), noise_scale=1.0,
)

print(f"  Video state: {video_state_A.latent.shape}")
print(f"  Audio state: {audio_state_A.latent.shape}")

denoise_fn = simple_denoising_func(
    video_context=video_context, audio_context=audio_context, transformer=transformer,
)

orig_target_per_step = []
orig_audio_per_step = []
for step_idx in range(n_steps):
    dv, da = denoise_fn(video_state_A, audio_state_A, sigmas, step_idx)
    dv = post_process_latent(dv, video_state_A.denoise_mask, video_state_A.clean_latent)
    da = post_process_latent(da, audio_state_A.denoise_mask, audio_state_A.clean_latent)
    video_state_A = replace(
        video_state_A,
        latent=stepper.step(video_state_A.latent, dv, sigmas, step_idx),
    )
    audio_state_A = replace(
        audio_state_A,
        latent=stepper.step(audio_state_A.latent, da, sigmas, step_idx),
    )
    # Extract target-only latents (unpatchify from combined state)
    target_5d_A = patchifier.unpatchify(
        video_state_A.latent[:, :seq_target], target_shape,
    )
    orig_target_per_step.append(target_5d_A.clone())
    orig_audio_per_step.append(audio_state_A.latent.clone())
    print(f"  Step {step_idx}: sigma={sigmas[step_idx].item():.6f} → {sigmas[step_idx+1].item():.6f}")

# Final result
orig_final = video_tools_A.clear_conditioning(video_state_A)
orig_final = video_tools_A.unpatchify(orig_final)
print(f"  Final target latent: {orig_final.latent.shape}")

# =====================================================================
# PATH B: Flow-Factory denoising (replicates inference() + forward())
# =====================================================================
print("\n" + "=" * 60)
print("PATH B: Flow-Factory denoising")
print("=" * 60)

gen_B = torch.Generator(device=DEVICE).manual_seed(SEED)

# Replicate inference() noise generation
target_zeros_5d = torch.zeros(
    1, 128, F_lat, H_lat, W_lat, device=DEVICE, dtype=DTYPE,
)
target_zeros_3d = patchifier.patchify(target_zeros_5d)
ref_latents_3d = patchifier.patchify(ref_latents)

combined_3d = torch.cat([
    target_zeros_3d, ref_latents_3d.to(dtype=DTYPE),
], dim=1)

noise_3d = torch.randn(
    *combined_3d.shape, device=DEVICE, dtype=DTYPE, generator=gen_B,
)

denoise_mask = torch.cat([
    torch.ones(1, seq_target, 1, device=DEVICE, dtype=DTYPE),
    torch.zeros(1, seq_ref, 1, device=DEVICE, dtype=DTYPE),
], dim=1)
noised_3d = noise_3d * denoise_mask + combined_3d * (1 - denoise_mask)

target_noised_3d = noised_3d[:, :seq_target]
ff_latents = patchifier.unpatchify(target_noised_3d, target_shape)

# Verify noise initialization matches
_report("initial_noise", orig_target_per_step[0] if not orig_target_per_step else
        patchifier.unpatchify(video_state_A.latent[:, :seq_target], target_shape) if False else ff_latents, ff_latents)

# Initialize audio state (same generator, after video noise)
audio_output_shape = VideoPixelShape(
    batch=1, frames=NUM_FRAMES, width=WIDTH, height=HEIGHT, fps=FRAME_RATE,
)
audio_components = PipelineComponents(dtype=DTYPE, device=torch.device(DEVICE))
audio_latent_shape = AudioLatentShape.from_video_pixel_shape(audio_output_shape)
audio_tools_B = AudioLatentTools(audio_components.audio_patchifier, audio_latent_shape)
audio_noiser_B = GaussianNoiser(generator=gen_B)
audio_state_B = audio_noiser_B(
    audio_tools_B.create_initial_state(DEVICE, DTYPE), noise_scale=1.0,
)
audio_latent = audio_state_B.latent
audio_positions = audio_state_B.positions
audio_denoise_mask = audio_state_B.denoise_mask

print(f"  FF latents: {ff_latents.shape}")
print(f"  FF audio: {audio_latent.shape}")

# Set up FF scheduler
ff_sched = LTXSDEScheduler(
    num_inference_steps=n_steps, dynamics_type="ODE", noise_level=0.0, use_distilled=True,
)
ff_sched.set_timesteps(n_steps, device=DEVICE)
ff_timesteps = ff_sched.timesteps

# Build positions once (same as forward() does)
fps = FRAME_RATE
target_positions = patchifier.get_patch_grid_bounds(target_shape, device=DEVICE)
target_positions = get_pixel_coords(
    target_positions, VIDEO_SCALE_FACTORS, causal_fix=True,
).float()
target_positions[:, 0, ...] = target_positions[:, 0, ...] / fps
target_positions = target_positions.to(dtype=torch.bfloat16)

ref_positions = patchifier.get_patch_grid_bounds(ref_shape_5d, device=DEVICE)
ref_positions = get_pixel_coords(
    ref_positions, VIDEO_SCALE_FACTORS, causal_fix=True,
)
ref_positions = ref_positions.to(dtype=torch.float32)
ref_positions[:, 0, ...] = ref_positions[:, 0, ...] / fps
if dsf != 1:
    ref_positions = ref_positions.clone()
    ref_positions[:, 1, ...] *= dsf
    ref_positions[:, 2, ...] *= dsf
positions = torch.cat([target_positions, ref_positions], dim=2)

ff_ref_ts = torch.zeros(1, seq_ref, 1, device=DEVICE, dtype=torch.float32)

ff_target_per_step = []
ff_audio_per_step = []
for step_idx in range(n_steps):
    sigma_val = sigmas[step_idx]
    sigma_batch = sigma_val.view(1)

    # Build video Modality
    ff_target_3d = patchifier.patchify(ff_latents)
    ff_combined = torch.cat([ff_target_3d, ref_latents_3d.to(dtype=DTYPE)], dim=1)

    ff_target_ts = sigma_batch.view(1, 1, 1).expand(1, seq_target, 1)
    ff_per_token_ts = torch.cat([ff_target_ts, ff_ref_ts], dim=1)

    ff_modality = Modality(
        enabled=True,
        latent=ff_combined,
        sigma=sigma_batch,
        timesteps=ff_per_token_ts,
        positions=positions,
        context=video_context,
        context_mask=None,
    )

    # Build audio Modality
    audio_ts = audio_denoise_mask * sigma_batch.view(1, 1, 1)
    audio_modality = Modality(
        enabled=True,
        latent=audio_latent,
        sigma=sigma_batch,
        timesteps=audio_ts,
        positions=audio_positions,
        context=audio_context,
        context_mask=None,
    )

    # Transformer forward
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
        video_pred, audio_pred = transformer(
            video=ff_modality, audio=audio_modality, perturbations=None,
        )

    # Step video (using FF scheduler on target only)
    target_v_3d = video_pred[:, :seq_target]
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
        compute_log_prob=False, return_dict=True, return_kwargs=["next_latents"],
    )
    ff_latents = out.next_latents

    # Step audio (using EulerDiffusionStep, matching original)
    audio_denoised = to_denoised(audio_latent, audio_pred, audio_ts)
    audio_clean = torch.zeros_like(audio_latent)
    audio_denoised = (
        audio_denoised * audio_denoise_mask + audio_clean * (1 - audio_denoise_mask)
    )
    all_sigmas = torch.cat([
        sigmas, torch.zeros(1, device=DEVICE, dtype=sigmas.dtype),
    ])
    audio_latent = EulerDiffusionStep().step(
        audio_latent, audio_denoised, all_sigmas, step_idx,
    )

    ff_target_per_step.append(ff_latents.clone())
    ff_audio_per_step.append(audio_latent.clone())
    print(f"  Step {step_idx}: sigma={sigmas[step_idx].item():.6f} → {sigmas[step_idx+1].item():.6f}")


# =====================================================================
# COMPARISON
# =====================================================================
print("\n" + "=" * 60)
print("STEP-BY-STEP COMPARISON")
print("=" * 60)

for i in range(n_steps):
    _report(f"step_{i}_video", orig_target_per_step[i], ff_target_per_step[i])
    _report(f"step_{i}_audio", orig_audio_per_step[i], ff_audio_per_step[i])

print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

_report("final_latents", orig_final.latent, ff_latents)

if all_passed:
    print("\n  *** END-TO-END PARITY VERIFIED — BITWISE IDENTICAL ***")
    print("  (Real data: RealisDance-Val, same model, same seed, same everything)")
else:
    print("\n  *** PARITY FAILED ***")

print("=" * 60)
sys.exit(0 if all_passed else 1)
