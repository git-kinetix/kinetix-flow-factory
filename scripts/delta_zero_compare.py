#!/usr/bin/env python3
"""
Delta-zero comparison: Flow-Factory scheduler step vs original ltx_core step.

Uses a SINGLE transformer call per step, then applies both the original
EulerDiffusionStep and our LTXSDEScheduler to verify they produce
bitwise identical results.

This eliminates CUDA non-determinism from the comparison by using the
same velocity prediction for both paths.

Usage:
    LTX_MODEL_PATH=/path/to/model \
    LTX_UNION_LORA_PATH=/path/to/lora.safetensors \
    LTX_GEMMA_PATH=/path/to/gemma \
    python scripts/delta_zero_compare.py
"""
from __future__ import annotations

import os
import torch

MODEL_PATH = os.environ["LTX_MODEL_PATH"]
LORA_PATH = os.environ["LTX_UNION_LORA_PATH"]
GEMMA_PATH = os.environ["LTX_GEMMA_PATH"]

SEED = 42
PROMPT = "A person dancing in a studio"
NUM_STEPS = 10
HEIGHT = 256
WIDTH = 448
NUM_FRAMES = 9
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
# 1. Load model
# ---------------------------------------------------------------------------
print("=" * 60)
print("Loading model components...")
print("=" * 60)

from ltx_trainer.model_loader import (
    load_transformer, load_video_vae_encoder,
    load_text_encoder, load_embeddings_processor,
)
from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.model.transformer import X0Model
from ltx_core.utils import to_denoised, to_velocity
from ltx_core.model.transformer.modality import Modality
from ltx_core.types import VideoLatentShape, VIDEO_SCALE_FACTORS
from ltx_core.components.patchifiers import get_pixel_coords

transformer = load_transformer(MODEL_PATH)

from flow_factory.models.ltx.pipeline import _merge_lora_into_model
dsf = _merge_lora_into_model(transformer, LORA_PATH)
print(f"reference_downscale_factor = {dsf}")

vae_encoder = load_video_vae_encoder(MODEL_PATH)
text_encoder = load_text_encoder(GEMMA_PATH)
embeddings_processor = load_embeddings_processor(MODEL_PATH)
patchifier = VideoLatentPatchifier(patch_size=1)

transformer.to(DEVICE, DTYPE).eval()
vae_encoder.to(DEVICE).eval()
text_encoder.to(DEVICE).eval()
embeddings_processor.to(DEVICE).eval()

# ---------------------------------------------------------------------------
# 2. Shared inputs
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Preparing shared inputs...")
print("=" * 60)

with torch.no_grad():
    hidden_states, attention_mask = text_encoder.encode(PROMPT)
    enc_output = embeddings_processor.process_hidden_states(hidden_states, attention_mask)
prompt_embeds = enc_output.video_encoding
print(f"  prompt_embeds: {prompt_embeds.shape}")

ref_h, ref_w = HEIGHT // dsf, WIDTH // dsf
valid_frames = (NUM_FRAMES - 1) // 8 * 8 + 1
torch.manual_seed(999)
ref_video = torch.randn(1, 3, valid_frames, ref_h, ref_w, device=DEVICE, dtype=torch.float32).clamp(-1, 1)
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
    ref_latents = vae_encoder(ref_video)
print(f"  ref_latents: {ref_latents.shape}")

F_lat = (NUM_FRAMES - 1) // 8 + 1
H_lat = HEIGHT // 32
W_lat = WIDTH // 32

# Sigma schedule
ltx_sched = LTX2Scheduler()
sigmas = ltx_sched.execute(steps=NUM_STEPS).to(DEVICE)

from flow_factory.models.ltx.scheduler_bridge import LTXSDEScheduler
ff_sched = LTXSDEScheduler(num_inference_steps=NUM_STEPS, dynamics_type="ODE", noise_level=0.0)
ff_sched.set_timesteps(NUM_STEPS, device=DEVICE)
_report("sigma_schedule", sigmas.cpu().float(), ff_sched.sigmas.cpu().float())

# Initial noise
gen = torch.Generator(device=DEVICE).manual_seed(SEED)
noise = torch.randn(1, 128, F_lat, H_lat, W_lat, device=DEVICE, dtype=DTYPE, generator=gen)

# Geometry
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
all_positions = all_positions.to(dtype=DTYPE)
all_positions[:, 0, ...] = all_positions[:, 0, ...] / 25.0

# ---------------------------------------------------------------------------
# 3. First: verify transformer determinism
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Verifying transformer determinism...")
print("=" * 60)

target_3d = patchifier.patchify(noise)
seq_t = target_3d.shape[1]
combined = torch.cat([ref_3d.to(DTYPE), target_3d], dim=1)

ref_ts = torch.zeros(1, seq_ref, 1, device=DEVICE, dtype=torch.float32)
tgt_ts = torch.full((1, seq_t, 1), sigmas[0].item(), device=DEVICE, dtype=torch.float32)
per_token_ts = torch.cat([ref_ts, tgt_ts], dim=1)

mod = Modality(
    enabled=True, latent=combined,
    sigma=sigmas[0].unsqueeze(0), timesteps=per_token_ts,
    positions=all_positions, context=prompt_embeds, context_mask=None,
)

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
    v1, _ = transformer(video=mod, audio=None, perturbations=None)
    v2, _ = transformer(video=mod, audio=None, perturbations=None)

_report("transformer_determinism", v1, v2)

# ---------------------------------------------------------------------------
# 4. Step-by-step: single transformer call, both step computations
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step-by-step comparison (single transformer call per step)...")
print("=" * 60)

stepper = EulerDiffusionStep()
lat_orig = noise.clone()
lat_ff = noise.clone()
ff_timesteps = ff_sched.timesteps

for step_idx in range(len(sigmas) - 1):
    sigma = sigmas[step_idx]
    sigma_next = sigmas[step_idx + 1]
    print(f"\n  --- Step {step_idx}: sigma {sigma.item():.6f} → {sigma_next.item():.6f} ---")

    _report(f"step_{step_idx}_input", lat_orig, lat_ff)

    # Patchify (identical for both since inputs match)
    target_3d = patchifier.patchify(lat_orig)
    seq_t = target_3d.shape[1]
    combined = torch.cat([ref_3d.to(DTYPE), target_3d], dim=1)

    ref_ts = torch.zeros(1, seq_ref, 1, device=DEVICE, dtype=torch.float32)
    tgt_ts = torch.full((1, seq_t, 1), sigma.item(), device=DEVICE, dtype=torch.float32)
    per_token_ts = torch.cat([ref_ts, tgt_ts], dim=1)

    mod = Modality(
        enabled=True, latent=combined,
        sigma=sigma.unsqueeze(0), timesteps=per_token_ts,
        positions=all_positions, context=prompt_embeds, context_mask=None,
    )

    # SINGLE transformer call
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
        velocity_pred, _ = transformer(video=mod, audio=None, perturbations=None)

    # ===== ORIGINAL PATH: to_denoised + denoise_mask + EulerDiffusionStep =====
    # to_denoised: x0 = (sample.f32 - velocity.f32 * sigma_pertoken.f32).bf16
    denoised_3d = to_denoised(combined, velocity_pred, per_token_ts)

    # Denoise mask: ref keeps clean, target keeps denoised
    dmask = torch.cat([
        torch.zeros(1, seq_ref, 1, device=DEVICE, dtype=DTYPE),
        torch.ones(1, seq_t, 1, device=DEVICE, dtype=DTYPE),
    ], dim=1)
    clean = torch.cat([ref_3d.to(DTYPE), torch.zeros_like(target_3d)], dim=1)
    denoised_masked = denoised_3d * dmask + clean * (1 - dmask)

    # EulerDiffusionStep on combined
    combined_next = stepper.step(
        sample=combined, denoised_sample=denoised_masked,
        sigmas=sigmas, step_index=step_idx,
    )
    orig_target_3d = combined_next[:, seq_ref:]
    lat_orig = patchifier.unpatchify(orig_target_3d, target_shape)

    # ===== FF PATH: extract target velocity, unpatchify, scheduler.step =====
    target_v_3d = velocity_pred[:, seq_ref:]
    target_v_5d = patchifier.unpatchify(target_v_3d, target_shape)

    t = ff_timesteps[step_idx]
    t_next = (
        ff_timesteps[step_idx + 1]
        if step_idx + 1 < len(ff_timesteps)
        else torch.tensor(0.0, device=DEVICE, dtype=ff_timesteps.dtype)
    )

    out_ff = ff_sched.step(
        noise_pred=target_v_5d, timestep=t, latents=lat_ff,
        timestep_next=t_next, noise_level=0.0,
        compute_log_prob=False, return_dict=True,
        return_kwargs=["next_latents"],
    )
    lat_ff = out_ff.next_latents

    _report(f"step_{step_idx}_output", lat_orig, lat_ff)

    if not torch.equal(lat_orig, lat_ff):
        # Diagnose: compare the individual sub-computations
        # to_denoised for target tokens
        orig_x0_target = denoised_3d[:, seq_ref:]
        ff_x0_target = (target_3d.float() - target_v_3d.float() * sigma.float()).to(DTYPE)
        _report(f"  to_denoised_target", orig_x0_target, ff_x0_target)

        # to_velocity: original uses sigma.item() (Python float)
        orig_v_recon = to_velocity(
            target_3d, sigma, denoised_masked[:, seq_ref:]
        )
        ff_v_recon = ((target_3d.float() - ff_x0_target.float()) / sigma.float()).to(DTYPE)
        _report(f"  to_velocity_target", orig_v_recon, ff_v_recon)

        # Euler step
        dt = sigma_next - sigma
        orig_step = (target_3d.float() + orig_v_recon.float() * dt).to(DTYPE)
        ff_step_manual = (target_3d.float() + ff_v_recon.float() * dt.float()).to(DTYPE)
        _report(f"  euler_step_manual", orig_step, ff_step_manual)

        # Check sigma roundtrip precision
        sigma_from_ff = t / 1000.0
        print(f"  sigma_orig={sigma.item():.15f}, sigma_ff={sigma_from_ff.item():.15f}")
        print(f"  >>> DIVERGENCE at step {step_idx}")
        break

# ---------------------------------------------------------------------------
# 5. Final
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

_report("final_latents", lat_orig, lat_ff)
print(f"  dtype: orig={lat_orig.dtype}, ff={lat_ff.dtype}")

if all_passed:
    print("\n  *** ALL CHECKS PASSED — DELTA ZERO VERIFIED ***")
else:
    print("\n  *** SOME CHECKS FAILED — SEE ABOVE ***")

print("=" * 60)
