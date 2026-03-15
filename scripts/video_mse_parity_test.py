#!/usr/bin/env python3
"""
Zero-MSE video parity test.

Runs two INDEPENDENT denoising loops + VAE decode:
  Path A: Original ICLoraPipeline (ltx_core functions)
  Path B: Flow-Factory (replicating inference() + forward())

Asserts MSE(video_A, video_B) == 0.0

NO PROXIES. NO SHARED CALLS. TWO INDEPENDENT RUNS.

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

# ---------------------------------------------------------------------------
# Force deterministic CUDA operations
# ---------------------------------------------------------------------------
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Force math SDP backend (deterministic) over flash/mem-efficient (non-deterministic)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ---------------------------------------------------------------------------
# Config
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
SAMPLE_ID = "0001"
VARIANT = "orbit_right_45"

all_passed = True


def _report(name, a, b):
    global all_passed
    mse = ((a.float() - b.float()) ** 2).mean().item()
    if mse == 0.0 and torch.equal(a, b):
        print(f"  [PASS] {name}: MSE=0.0, bitwise identical")
    else:
        all_passed = False
        diff = (a.float() - b.float()).abs()
        print(
            f"  [FAIL] {name}: MSE={mse:.2e}, max_diff={diff.max().item():.2e}, "
            f"mismatched={diff.nonzero().shape[0]}/{a.numel()}"
        )


# ===========================================================================
# 1. Load model + encode shared inputs
# ===========================================================================
print("=" * 60)
print("Loading model + encoding shared inputs...")
print("=" * 60)

from ltx_trainer.model_loader import (
    load_transformer, load_video_vae_encoder, load_video_vae_decoder,
    load_text_encoder, load_embeddings_processor,
)
from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.types import (
    VideoLatentShape, VIDEO_SCALE_FACTORS, VideoPixelShape, AudioLatentShape,
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
from ltx_pipelines.utils.media_io import load_video_conditioning

from flow_factory.models.ltx.pipeline import _merge_lora_into_model
from flow_factory.models.ltx.scheduler_bridge import LTXSDEScheduler

transformer = load_transformer(MODEL_PATH)
dsf = _merge_lora_into_model(transformer, LORA_PATH)
print(f"  dsf={dsf}")

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

# Encode prompt
prompt_file = os.path.join(DATASET_ROOT, "prompt", f"{SAMPLE_ID}.txt")
with open(prompt_file) as f:
    prompt = f.read().strip()
print(f"  Prompt: {prompt[:60]}...")

with torch.no_grad():
    hidden_states, attention_mask = text_encoder.encode(prompt)
    enc_output = embeddings_processor.process_hidden_states(hidden_states, attention_mask)
video_context = enc_output.video_encoding
audio_context = enc_output.audio_encoding

del text_encoder, embeddings_processor
gc.collect(); torch.cuda.empty_cache()

# Encode reference conditioning video
npz_path = os.path.join(
    DATASET_ROOT, "condition", "pose_depth", f"{SAMPLE_ID}_{VARIANT}video.npz"
)
data = np.load(npz_path)
num_npz_frames = int(data["num_frames"])
frames_list = [
    (data[f"frame_{i:05d}"] * 255).clip(0, 255).astype(np.uint8)
    for i in range(num_npz_frames)
]
ref_h, ref_w = HEIGHT // dsf, WIDTH // dsf
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

del vae_encoder
gc.collect(); torch.cuda.empty_cache()

# Shapes
F_lat = (NUM_FRAMES - 1) // 8 + 1
H_lat, W_lat = HEIGHT // 32, WIDTH // 32
target_shape = VideoLatentShape(batch=1, channels=128, frames=F_lat, height=H_lat, width=W_lat)
sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, device=DEVICE, dtype=torch.float32)
n_steps = len(sigmas) - 1
seq_target = patchifier.get_token_count(target_shape)
_, _, ref_F, ref_H, ref_W = ref_latents.shape
ref_shape_5d = VideoLatentShape(batch=1, channels=128, frames=ref_F, height=ref_H, width=ref_W)
seq_ref = patchifier.get_token_count(ref_shape_5d)
ref_latents_3d = patchifier.patchify(ref_latents)

output_shape = VideoPixelShape(
    batch=1, frames=NUM_FRAMES, width=WIDTH, height=HEIGHT, fps=FRAME_RATE,
)

print(f"  target: {seq_target} tokens, ref: {seq_ref} tokens, steps: {n_steps}")


# ===========================================================================
# 2. PATH A: Original ICLoraPipeline (fully independent)
# ===========================================================================
print("\n" + "=" * 60)
print("PATH A: Original ICLoraPipeline")
print("=" * 60)

components = PipelineComponents(dtype=DTYPE, device=torch.device(DEVICE))
gen_A = torch.Generator(device=DEVICE).manual_seed(SEED)
noiser_A = GaussianNoiser(generator=gen_A)
stepper = EulerDiffusionStep()

cond_A = VideoConditionByReferenceLatent(
    latent=ref_latents, downscale_factor=dsf, strength=1.0,
)
video_state_A, video_tools_A = noise_video_state(
    output_shape=output_shape, noiser=noiser_A, conditionings=[cond_A],
    components=components, dtype=DTYPE, device=torch.device(DEVICE), noise_scale=1.0,
)
audio_state_A, audio_tools_A = noise_audio_state(
    output_shape=output_shape, noiser=noiser_A, conditionings=[],
    components=components, dtype=DTYPE, device=torch.device(DEVICE), noise_scale=1.0,
)

denoise_fn = simple_denoising_func(
    video_context=video_context, audio_context=audio_context, transformer=transformer,
)

for step_idx in range(n_steps):
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
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
    del dv, da
    torch.cuda.empty_cache()
    print(f"  Step {step_idx}: sigma={sigmas[step_idx].item():.6f}")

# Extract final target latents
orig_final = video_tools_A.clear_conditioning(video_state_A)
orig_final = video_tools_A.unpatchify(orig_final)
latents_A = orig_final.latent  # [1, 128, F, H, W]
print(f"  Final latents A: {latents_A.shape}")

# Decode to video pixels
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
    frame_chunks_A = list(vae_decode_video(
        latents_A[0], vae_decoder, tiling_config=None, generator=None,
    ))
video_A = torch.cat(frame_chunks_A, dim=0)  # [T, H, W, C] uint8
video_A = video_A.permute(0, 3, 1, 2).float() / 255.0  # [T, C, H, W] float [0,1]
print(f"  Video A: {video_A.shape}, range=[{video_A.min():.3f}, {video_A.max():.3f}]")

# Move to CPU and free GPU
latents_A_cpu = latents_A.cpu()
video_A_cpu = video_A.cpu()
del (
    video_state_A, audio_state_A, orig_final, latents_A, video_A,
    frame_chunks_A, video_tools_A, audio_tools_A, denoise_fn,
    noiser_A, cond_A, gen_A, components,
)
gc.collect(); torch.cuda.empty_cache()
print(f"  GPU freed. CUDA memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")


# ===========================================================================
# 3. PATH B: Flow-Factory (fully independent, replicating inference+forward)
# ===========================================================================
print("\n" + "=" * 60)
print("PATH B: Flow-Factory")
print("=" * 60)

gen_B = torch.Generator(device=DEVICE).manual_seed(SEED)

# Noise initialization (matching inference() exactly)
target_zeros_5d = torch.zeros(1, 128, F_lat, H_lat, W_lat, device=DEVICE, dtype=DTYPE)
target_zeros_3d = patchifier.patchify(target_zeros_5d)
combined_3d = torch.cat([target_zeros_3d, ref_latents_3d.to(dtype=DTYPE)], dim=1)
noise_3d = torch.randn(*combined_3d.shape, device=DEVICE, dtype=DTYPE, generator=gen_B)
denoise_mask = torch.cat([
    torch.ones(1, seq_target, 1, device=DEVICE, dtype=DTYPE),
    torch.zeros(1, seq_ref, 1, device=DEVICE, dtype=DTYPE),
], dim=1)
noised_3d = noise_3d * denoise_mask + combined_3d * (1 - denoise_mask)
ff_latents = patchifier.unpatchify(noised_3d[:, :seq_target], target_shape)
del target_zeros_5d, target_zeros_3d, combined_3d, noise_3d, noised_3d, denoise_mask

# Audio state (same generator, after video noise)
audio_components = PipelineComponents(dtype=DTYPE, device=torch.device(DEVICE))
audio_latent_shape = AudioLatentShape.from_video_pixel_shape(output_shape)
audio_tools_B = AudioLatentTools(audio_components.audio_patchifier, audio_latent_shape)
audio_noiser_B = GaussianNoiser(generator=gen_B)
audio_state_B = audio_noiser_B(
    audio_tools_B.create_initial_state(DEVICE, DTYPE), noise_scale=1.0,
)
ff_audio = audio_state_B.latent
audio_positions = audio_state_B.positions
audio_denoise_mask = audio_state_B.denoise_mask
del audio_noiser_B, audio_state_B

# Verify initial states match
print("  Verifying initial states match Path A...")
# Re-create Path A initial state for comparison (same seed, same code)
gen_verify = torch.Generator(device=DEVICE).manual_seed(SEED)
noiser_verify = GaussianNoiser(generator=gen_verify)
cond_verify = VideoConditionByReferenceLatent(
    latent=ref_latents, downscale_factor=dsf, strength=1.0,
)
components_verify = PipelineComponents(dtype=DTYPE, device=torch.device(DEVICE))
vs_verify, vt_verify = noise_video_state(
    output_shape=output_shape, noiser=noiser_verify, conditionings=[cond_verify],
    components=components_verify, dtype=DTYPE, device=torch.device(DEVICE), noise_scale=1.0,
)
init_target_A = patchifier.unpatchify(vs_verify.latent[:, :seq_target], target_shape)
_report("initial_noise_cross_check", init_target_A, ff_latents)
del gen_verify, noiser_verify, cond_verify, components_verify, vs_verify, vt_verify, init_target_A

# FF scheduler
ff_sched = LTXSDEScheduler(
    num_inference_steps=n_steps, dynamics_type="ODE", noise_level=0.0, use_distilled=True,
)
ff_sched.set_timesteps(n_steps, device=DEVICE)
ff_timesteps = ff_sched.timesteps

# Positions (matching forward())
fps = FRAME_RATE
target_positions = patchifier.get_patch_grid_bounds(target_shape, device=DEVICE)
target_positions = get_pixel_coords(target_positions, VIDEO_SCALE_FACTORS, causal_fix=True).float()
target_positions[:, 0, ...] = target_positions[:, 0, ...] / fps
target_positions = target_positions.to(dtype=torch.bfloat16)

ref_positions = patchifier.get_patch_grid_bounds(ref_shape_5d, device=DEVICE)
ref_positions = get_pixel_coords(ref_positions, VIDEO_SCALE_FACTORS, causal_fix=True)
ref_positions = ref_positions.to(dtype=torch.float32)
ref_positions[:, 0, ...] = ref_positions[:, 0, ...] / fps
if dsf != 1:
    ref_positions = ref_positions.clone()
    ref_positions[:, 1, ...] *= dsf
    ref_positions[:, 2, ...] *= dsf
ff_positions = torch.cat([target_positions, ref_positions], dim=2)
ff_ref_ts = torch.zeros(1, seq_ref, 1, device=DEVICE, dtype=torch.float32)

# Denoising loop (fully independent from Path A)
for step_idx in range(n_steps):
    sigma_val = sigmas[step_idx]
    sigma_batch = sigma_val.view(1)

    # Video Modality
    ff_target_3d = patchifier.patchify(ff_latents)
    ff_combined = torch.cat([ff_target_3d, ref_latents_3d.to(dtype=DTYPE)], dim=1)
    ff_target_ts = sigma_batch.view(1, 1, 1).expand(1, seq_target, 1)
    ff_per_token_ts = torch.cat([ff_target_ts, ff_ref_ts], dim=1)

    ff_video_mod = Modality(
        enabled=True, latent=ff_combined, sigma=sigma_batch,
        timesteps=ff_per_token_ts, positions=ff_positions,
        context=video_context, context_mask=None,
    )

    # Audio Modality
    audio_ts = audio_denoise_mask * sigma_batch.view(1, 1, 1)
    ff_audio_mod = Modality(
        enabled=True, latent=ff_audio, sigma=sigma_batch,
        timesteps=audio_ts, positions=audio_positions,
        context=audio_context, context_mask=None,
    )

    # Independent transformer call
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
        video_pred, audio_pred = transformer(
            video=ff_video_mod, audio=ff_audio_mod, perturbations=None,
        )

    # Step video (FF scheduler)
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

    # Step audio
    audio_denoised = to_denoised(ff_audio, audio_pred, audio_ts)
    audio_clean = torch.zeros_like(ff_audio)
    audio_denoised = audio_denoised * audio_denoise_mask + audio_clean * (1 - audio_denoise_mask)
    all_sigmas = torch.cat([sigmas, torch.zeros(1, device=DEVICE, dtype=sigmas.dtype)])
    ff_audio = EulerDiffusionStep().step(ff_audio, audio_denoised, all_sigmas, step_idx)

    # Free intermediates
    del (
        ff_target_3d, ff_combined, ff_target_ts, ff_per_token_ts,
        ff_video_mod, ff_audio_mod, video_pred, audio_pred,
        target_v_3d, target_v_5d, audio_denoised, audio_clean, audio_ts, out,
    )
    torch.cuda.empty_cache()
    print(f"  Step {step_idx}: sigma={sigma_val.item():.6f}")

latents_B = ff_latents  # [1, 128, F, H, W]
print(f"  Final latents B: {latents_B.shape}")

# Decode to video pixels
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
    frame_chunks_B = list(vae_decode_video(
        latents_B[0], vae_decoder, tiling_config=None, generator=None,
    ))
video_B = torch.cat(frame_chunks_B, dim=0)
video_B = video_B.permute(0, 3, 1, 2).float() / 255.0
print(f"  Video B: {video_B.shape}, range=[{video_B.min():.3f}, {video_B.max():.3f}]")

latents_B_cpu = latents_B.cpu()
video_B_cpu = video_B.cpu()


# ===========================================================================
# 4. THE COMPARISON — MSE must be 0.0
# ===========================================================================
print("\n" + "=" * 60)
print("THE COMPARISON")
print("=" * 60)

_report("latents", latents_A_cpu, latents_B_cpu)
_report("decoded_video", video_A_cpu, video_B_cpu)

mse = ((video_A_cpu.float() - video_B_cpu.float()) ** 2).mean().item()
print(f"\n  MSE = {mse}")

if all_passed and mse == 0.0:
    print("\n  *** ZERO MSE VERIFIED — VIDEOS ARE PIXEL-IDENTICAL ***")
else:
    print("\n  *** MSE != 0 — VIDEOS DIFFER ***")

print("=" * 60)
sys.exit(0 if all_passed else 1)
