#!/usr/bin/env python3
"""
Delta-zero comparison: Flow-Factory LTXUnionAdapter vs original ltx_core pipeline.

Runs both pipelines with identical inputs (same seed, prompt, reference latents)
and asserts torch.equal() at every stage:
  1. Sigma schedule
  2. Initial noise
  3. Each denoising step (transformer input identical → step output identical)
  4. Final latents

Usage:
    LTX_MODEL_PATH=/path/to/model \
    LTX_UNION_LORA_PATH=/path/to/lora.safetensors \
    LTX_GEMMA_PATH=/path/to/gemma \
    python scripts/delta_zero_compare.py

Requirements: GPU + ltx_core + ltx_trainer + flow_factory installed
"""
from __future__ import annotations

import os
import torch

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
WIDTH = 448  # Must satisfy: (WIDTH // dsf) % 32 == 0 for VAE compatibility
NUM_FRAMES = 9
DEVICE = "cuda"
DTYPE = torch.bfloat16

all_passed = True


def _report(name: str, a: torch.Tensor, b: torch.Tensor):
    """Report match/mismatch between two tensors."""
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
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.model.transformer import X0Model
from ltx_core.model.transformer.modality import Modality
from ltx_core.types import VideoLatentShape, VIDEO_SCALE_FACTORS
from ltx_core.components.patchifiers import get_pixel_coords

# Load all components
transformer = load_transformer(MODEL_PATH)

from flow_factory.models.ltx.pipeline import _merge_lora_into_model
dsf = _merge_lora_into_model(transformer, LORA_PATH)
print(f"reference_downscale_factor = {dsf}")

vae_encoder = load_video_vae_encoder(MODEL_PATH)
text_encoder = load_text_encoder(GEMMA_PATH)
embeddings_processor = load_embeddings_processor(MODEL_PATH)
patchifier = VideoLatentPatchifier(patch_size=1)

# Move to device
transformer.to(DEVICE, DTYPE).eval()
vae_encoder.to(DEVICE).eval()
text_encoder.to(DEVICE).eval()
embeddings_processor.to(DEVICE).eval()

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
print(f"  prompt_embeds: {prompt_embeds.shape}, dtype={prompt_embeds.dtype}")

# ---------------------------------------------------------------------------
# 3. Create reference latents (deterministic synthetic)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Creating reference latents...")
print("=" * 60)

ref_h = HEIGHT // dsf
ref_w = WIDTH // dsf
valid_frames = (NUM_FRAMES - 1) // 8 * 8 + 1

torch.manual_seed(999)
ref_video = torch.randn(1, 3, valid_frames, ref_h, ref_w, device=DEVICE, dtype=torch.float32)
ref_video = ref_video.clamp(-1, 1)

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
print(f"  Original: {orig_sigmas.tolist()}")

from flow_factory.models.ltx.scheduler_bridge import LTXSDEScheduler
ff_sched = LTXSDEScheduler(num_inference_steps=NUM_STEPS, dynamics_type="ODE", noise_level=0.0)
ff_sched.set_timesteps(NUM_STEPS, device=DEVICE)
ff_sigmas = ff_sched.sigmas.cpu()
print(f"  FF:       {ff_sigmas.tolist()}")

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
# 6. Prepare shared geometry (positions)
# ---------------------------------------------------------------------------
_, _, ref_F, ref_H, ref_W = ref_latents.shape
ref_3d = patchifier.patchify(ref_latents)
seq_ref = ref_3d.shape[1]

ref_shape = VideoLatentShape(batch=1, channels=128, frames=ref_F, height=ref_H, width=ref_W)
target_shape = VideoLatentShape(batch=1, channels=128, frames=F_lat, height=H_lat, width=W_lat)

ref_positions = patchifier.get_patch_grid_bounds(ref_shape, device=DEVICE)
target_positions = patchifier.get_patch_grid_bounds(target_shape, device=DEVICE)

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

# ---------------------------------------------------------------------------
# 7. Step-by-step comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step-by-step denoising comparison...")
print("=" * 60)

x0_model = X0Model(transformer)
stepper = EulerDiffusionStep()

lat_orig = noise_orig.clone()  # 5D
lat_ff = noise_ff.clone()  # 5D

ff_timesteps = ff_sched.timesteps

for step_idx in range(len(sigmas) - 1):
    sigma = sigmas[step_idx]
    sigma_next = sigmas[step_idx + 1]
    print(f"\n  --- Step {step_idx}: sigma {sigma.item():.6f} → {sigma_next.item():.6f} ---")

    # Check input latents match
    _report(f"step_{step_idx}_input", lat_orig, lat_ff)

    # ----- ORIGINAL PATH: X0Model + EulerDiffusionStep on combined 3D -----
    orig_3d = patchifier.patchify(lat_orig)
    seq_t = orig_3d.shape[1]
    combined_orig = torch.cat([ref_3d.to(DTYPE), orig_3d], dim=1)

    # per-token timesteps: [B, seq, 1] for broadcast with [B, seq, C]
    ref_ts = torch.zeros(1, seq_ref, 1, device=DEVICE, dtype=torch.float32)
    tgt_ts = torch.full((1, seq_t, 1), sigma.item(), device=DEVICE, dtype=torch.float32)
    per_token_ts = torch.cat([ref_ts, tgt_ts], dim=1)

    mod = Modality(
        enabled=True, latent=combined_orig,
        sigma=sigma.unsqueeze(0), timesteps=per_token_ts,
        positions=all_positions, context=prompt_embeds, context_mask=None,
    )

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
        denoised_orig, _ = x0_model(video=mod, audio=None, perturbations=None)

    # Apply denoise_mask
    dmask = torch.cat([
        torch.zeros(1, seq_ref, 1, device=DEVICE, dtype=DTYPE),
        torch.ones(1, seq_t, 1, device=DEVICE, dtype=DTYPE),
    ], dim=1)
    clean = torch.cat([ref_3d.to(DTYPE), torch.zeros_like(orig_3d)], dim=1)
    denoised_masked = denoised_orig * dmask + clean * (1 - dmask)

    # EulerDiffusionStep.step on combined
    combined_next = stepper.step(
        sample=combined_orig,
        denoised_sample=denoised_masked,
        sigmas=sigmas,
        step_index=step_idx,
    )

    # Extract target and unpatchify
    orig_target_3d = combined_next[:, seq_ref:]
    lat_orig = patchifier.unpatchify(orig_target_3d, target_shape)

    # ----- FF PATH: raw transformer + scheduler.step on target-only 5D -----
    ff_3d = patchifier.patchify(lat_ff)
    combined_ff = torch.cat([ref_3d.to(DTYPE), ff_3d], dim=1)

    mod_ff = Modality(
        enabled=True, latent=combined_ff,
        sigma=sigma.unsqueeze(0), timesteps=per_token_ts,
        positions=all_positions, context=prompt_embeds, context_mask=None,
    )

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
        v_pred_ff, _ = transformer(video=mod_ff, audio=None, perturbations=None)

    ff_target_v_3d = v_pred_ff[:, seq_ref:]
    ff_target_v_5d = patchifier.unpatchify(ff_target_v_3d, target_shape)

    t = ff_timesteps[step_idx]
    t_next = (
        ff_timesteps[step_idx + 1]
        if step_idx + 1 < len(ff_timesteps)
        else torch.tensor(0.0, device=DEVICE, dtype=ff_timesteps.dtype)
    )

    out_ff = ff_sched.step(
        noise_pred=ff_target_v_5d, timestep=t, latents=lat_ff,
        timestep_next=t_next, noise_level=0.0,
        compute_log_prob=False, return_dict=True,
        return_kwargs=["next_latents"],
    )
    lat_ff = out_ff.next_latents

    # Compare step outputs
    _report(f"step_{step_idx}_output", lat_orig, lat_ff)

    if not torch.equal(lat_orig, lat_ff):
        # Diagnose: is it the transformer prediction or the step computation?
        # Both paths used the same input, so if transformer is deterministic,
        # the velocity should match. Let's check:
        # Original x0 for target = denoised_orig[:, seq_ref:]
        orig_x0_target = denoised_orig[:, seq_ref:]
        # v = (sample - x0) / sigma (what to_velocity does)
        orig_v_from_x0 = (orig_3d.float() - orig_x0_target.float()) / sigma.float()
        orig_v_from_x0 = orig_v_from_x0.to(DTYPE)
        _report(f"step_{step_idx}_velocity_match", orig_v_from_x0, ff_target_v_3d)

        # Also compare x0 derived from FF velocity
        ff_x0 = (lat_ff.float() - ff_target_v_5d.float() * sigma.float()).to(DTYPE)
        print(f"  (diagnostic) ff_x0 dtype={ff_x0.dtype}, orig_x0 dtype={orig_x0_target.dtype}")
        print(f"  >>> DIVERGENCE at step {step_idx}")
        break

# ---------------------------------------------------------------------------
# 8. Final comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

_report("final_latents", lat_orig, lat_ff)
print(f"  Original dtype: {lat_orig.dtype}, FF dtype: {lat_ff.dtype}")

if all_passed:
    print("\n  *** ALL CHECKS PASSED — DELTA ZERO VERIFIED ***")
else:
    print("\n  *** SOME CHECKS FAILED — SEE ABOVE ***")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
