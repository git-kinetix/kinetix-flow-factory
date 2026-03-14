#!/usr/bin/env python3
"""
Inference parity test: verify that LTXUnionAdapter's inference path produces
bitwise identical output to the original ICLoraPipeline Stage 1 denoising.

Runs a single shared denoising loop, calling the transformer ONCE per step,
then applies both the original EulerDiffusionStep and the FF LTXSDEScheduler
to the SAME velocity prediction. Compares the resulting latents at each step.

Also tests whether audio=None vs audio=Modality changes video output.

Environment variables:
    LTX_MODEL_PATH       — path to ltx-2.3 distilled .safetensors
    LTX_UNION_LORA_PATH  — path to IC-LoRA union control .safetensors
    LTX_GEMMA_PATH       — path to Gemma-3 text encoder directory
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
# 1. Load components
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
from ltx_core.conditioning import VideoConditionByReferenceLatent
from ltx_core.model.transformer.modality import Modality
from ltx_core.types import VideoLatentShape, VideoPixelShape, VIDEO_SCALE_FACTORS
from ltx_core.tools import VideoLatentTools
from ltx_core.utils import to_denoised

from flow_factory.models.ltx.pipeline import _merge_lora_into_model
from flow_factory.models.ltx.scheduler_bridge import LTXSDEScheduler

from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.helpers import (
    modality_from_latent_state, post_process_latent,
    noise_video_state, noise_audio_state,
)
from ltx_pipelines.utils.types import PipelineComponents

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
# 2. Prepare inputs
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Preparing shared inputs...")
print("=" * 60)

with torch.no_grad():
    hidden_states, attention_mask = text_encoder.encode(PROMPT)
    enc_output = embeddings_processor.process_hidden_states(hidden_states, attention_mask)
prompt_embeds = enc_output.video_encoding
audio_embeds = enc_output.audio_encoding
print(f"  prompt_embeds: {prompt_embeds.shape}")
print(f"  audio_embeds: {audio_embeds.shape if audio_embeds is not None else None}")

# Free text encoder
del text_encoder, embeddings_processor
gc.collect(); torch.cuda.empty_cache()

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

# Free VAE encoder
del vae_encoder, ref_video_tensor
gc.collect(); torch.cuda.empty_cache()

F_lat = (NUM_FRAMES - 1) // 8 + 1
H_lat = STAGE1_H // 32
W_lat = STAGE1_W // 32
target_shape = VideoLatentShape(batch=1, channels=128, frames=F_lat, height=H_lat, width=W_lat)
print(f"  Target latent: F={F_lat}, H={H_lat}, W={W_lat}")

sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, device=DEVICE, dtype=torch.float32)
components = PipelineComponents(dtype=DTYPE, device=torch.device(DEVICE))
output_shape = VideoPixelShape(
    batch=1, frames=NUM_FRAMES, width=STAGE1_W, height=STAGE1_H, fps=FRAME_RATE,
)

# ---------------------------------------------------------------------------
# 3. Test: audio=None vs audio=Modality — single step comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Test: does audio modality affect video output?")
print("=" * 60)

gen_test = torch.Generator(device=DEVICE).manual_seed(SEED)
noiser_test = GaussianNoiser(generator=gen_test)
cond = VideoConditionByReferenceLatent(latent=ref_latents, downscale_factor=dsf, strength=1.0)

test_video_state, test_video_tools = noise_video_state(
    output_shape=output_shape, noiser=noiser_test, conditionings=[cond],
    components=components, dtype=DTYPE, device=torch.device(DEVICE), noise_scale=1.0,
)
test_audio_state, _ = noise_audio_state(
    output_shape=output_shape, noiser=noiser_test, conditionings=[],
    components=components, dtype=DTYPE, device=torch.device(DEVICE), noise_scale=1.0,
)

seq_target = patchifier.get_token_count(target_shape)
seq_ref = test_video_state.latent.shape[1] - seq_target

sigma = sigmas[0]
video_mod = modality_from_latent_state(test_video_state, prompt_embeds, sigma)
audio_mod = modality_from_latent_state(test_audio_state, audio_embeds, sigma)

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
    vel_with_audio, _ = transformer(video=video_mod, audio=audio_mod, perturbations=None)
    vel_no_audio, _ = transformer(video=video_mod, audio=None, perturbations=None)

_report("video_velocity_with_vs_without_audio", vel_with_audio, vel_no_audio)

del test_video_state, test_audio_state, test_video_tools, video_mod, audio_mod
del vel_with_audio, vel_no_audio
gc.collect(); torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 4. Full denoising: shared transformer call, two stepping paths
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Full denoising loop: original EulerDiffusionStep vs FF LTXSDEScheduler")
print("=" * 60)

stepper = EulerDiffusionStep()

# Initialize original state
gen_orig = torch.Generator(device=DEVICE).manual_seed(SEED)
noiser_orig = GaussianNoiser(generator=gen_orig)

video_state, video_tools = noise_video_state(
    output_shape=output_shape, noiser=noiser_orig, conditionings=[cond],
    components=components, dtype=DTYPE, device=torch.device(DEVICE), noise_scale=1.0,
)

# Initialize FF state (same noise)
gen_ff = torch.Generator(device=DEVICE).manual_seed(SEED)
ref_latents_3d = patchifier.patchify(ref_latents)
seq_ref_ff = ref_latents_3d.shape[1]

target_zeros_5d = torch.zeros(1, 128, F_lat, H_lat, W_lat, device=DEVICE, dtype=DTYPE)
target_zeros_3d = patchifier.patchify(target_zeros_5d)
seq_target_ff = target_zeros_3d.shape[1]

combined_3d = torch.cat([target_zeros_3d, ref_latents_3d.to(dtype=DTYPE)], dim=1)
noise_3d = torch.randn(*combined_3d.shape, device=DEVICE, dtype=DTYPE, generator=gen_ff)
denoise_mask_ff = torch.cat([
    torch.ones(1, seq_target_ff, 1, device=DEVICE, dtype=DTYPE),
    torch.zeros(1, seq_ref_ff, 1, device=DEVICE, dtype=DTYPE),
], dim=1)
noised_3d = noise_3d * denoise_mask_ff + combined_3d * (1 - denoise_mask_ff)
target_noised_3d = noised_3d[:, :seq_target_ff]
ff_latents = patchifier.unpatchify(target_noised_3d, target_shape)

# Verify initial noise matches
orig_target_5d = patchifier.unpatchify(video_state.latent[:, :seq_target], target_shape)
print("\nComparing initial state:")
_report("initial_noised_latents", orig_target_5d, ff_latents)

# FF scheduler
ff_sched = LTXSDEScheduler(
    num_inference_steps=8, dynamics_type="ODE", noise_level=0.0, use_distilled=True,
)
ff_sched.set_timesteps(8, device=DEVICE)
ff_timesteps = ff_sched.timesteps

# Ref tokens for FF
ref_3d = patchifier.patchify(ref_latents).to(dtype=DTYPE)
ff_ref_ts = torch.zeros(1, seq_ref_ff, 1, device=DEVICE, dtype=torch.float32)

# FF positions
ff_target_positions = patchifier.get_patch_grid_bounds(target_shape, device=DEVICE)
ff_target_positions = get_pixel_coords(ff_target_positions, VIDEO_SCALE_FACTORS, causal_fix=True).float()
ff_target_positions[:, 0, ...] = ff_target_positions[:, 0, ...] / FRAME_RATE
ff_target_positions = ff_target_positions.to(dtype=torch.bfloat16)

_, _, rF, rH, rW = ref_latents.shape
ff_ref_shape = VideoLatentShape(batch=1, channels=128, frames=rF, height=rH, width=rW)
ff_ref_positions = patchifier.get_patch_grid_bounds(ff_ref_shape, device=DEVICE)
ff_ref_positions = get_pixel_coords(ff_ref_positions, VIDEO_SCALE_FACTORS, causal_fix=True)
ff_ref_positions = ff_ref_positions.to(dtype=torch.float32)
ff_ref_positions[:, 0, ...] = ff_ref_positions[:, 0, ...] / FRAME_RATE
if dsf != 1:
    ff_ref_positions = ff_ref_positions.clone()
    ff_ref_positions[:, 1, ...] *= dsf
    ff_ref_positions[:, 2, ...] *= dsf
ff_positions = torch.cat([ff_target_positions, ff_ref_positions], dim=2)

print("\nDenoising loop:")
n_steps = len(sigmas) - 1
for step_idx in range(n_steps):
    sigma_val = sigmas[step_idx]

    # Build Modality using original state (which we verified matches FF)
    orig_mod = modality_from_latent_state(video_state, prompt_embeds, sigma_val)

    # Single transformer call with audio=None (what forward() does)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
        velocity_pred, _ = transformer(video=orig_mod, audio=None, perturbations=None)

    # === ORIGINAL PATH: EulerDiffusionStep ===
    denoised_combined = to_denoised(video_state.latent, velocity_pred, orig_mod.timesteps)
    denoised_combined = post_process_latent(
        denoised_combined, video_state.denoise_mask, video_state.clean_latent,
    )
    new_combined = stepper.step(
        sample=video_state.latent,
        denoised_sample=denoised_combined,
        sigmas=sigmas,
        step_index=step_idx,
    )
    video_state = replace(video_state, latent=new_combined)
    orig_target_5d = patchifier.unpatchify(new_combined[:, :seq_target], target_shape)

    # === FF PATH: LTXSDEScheduler ===
    target_v_3d = velocity_pred[:, :seq_target_ff]
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

    match = torch.equal(orig_target_5d, ff_latents)
    status = "OK" if match else "MISMATCH"
    if not match:
        all_passed = False
    print(f"  Step {step_idx}: {status} (sigma={sigma_val.item():.6f})")
    if not match:
        _report(f"    step_{step_idx}_output", orig_target_5d, ff_latents)
        break

# ---------------------------------------------------------------------------
# 5. Final comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

orig_final = video_tools.clear_conditioning(video_state)
orig_final = video_tools.unpatchify(orig_final)
_report("final_latents", orig_final.latent, ff_latents)

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
    orig_pixels = vae_decoder(orig_final.latent)
    ff_pixels = vae_decoder(ff_latents)
_report("decoded_video", orig_pixels, ff_pixels)

if all_passed:
    print("\n  *** FULL INFERENCE PARITY VERIFIED — BITWISE IDENTICAL ***")
else:
    print("\n  *** PARITY FAILED ***")

print("=" * 60)
sys.exit(0 if all_passed else 1)
