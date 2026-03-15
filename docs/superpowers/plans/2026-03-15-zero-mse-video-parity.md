# Zero-MSE Video Parity: Original Pipeline vs Training Sampling

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Achieve `MSE(video_A, video_B) == 0.0` between a video from the original LTX ICLoraPipeline (pose+depth, all steps) and a video from Flow-Factory's NFT sampling phase, under identical conditions.

**Architecture:** Write a single script that runs both pipelines independently (no shared transformer calls) in the same process, decodes both to pixel video, and asserts zero MSE. Run paths sequentially to avoid OOM on the 22B model. Fix any divergence source until MSE is zero.

**Tech Stack:** PyTorch, ltx_core, ltx_pipelines, flow_factory, CUDA

---

## Known Divergence Sources

| # | Source | Status | Impact |
|---|--------|--------|--------|
| 1 | FPS (was 25, should be 24) | FIXED in adapter __init__ | Position mismatch |
| 2 | Audio modality (was None) | FIXED in forward() | 99.2% velocity mismatch |
| 3 | Image conditioning (ref image) | **GAP** — original has it, FF doesn't | Different token count, positions, attention |
| 4 | Latent-level parity (same inputs) | VERIFIED via e2e_parity_test.py | All 8 steps bitwise identical |
| 5 | VAE decode parity | **UNTESTED** | Same latents should decode identically |

## Strategy

**Phase 1 (Task 1-2):** Prove zero MSE for matching setups — both pipelines without image conditioning. This isolates the denoising+decode path. If this fails, something fundamental is broken.

**Phase 2 (Task 3-5):** Add image conditioning to Flow-Factory so it matches the original pipeline exactly. Then prove zero MSE with image conditioning included.

---

### Task 1: Write the independent video comparison script

**Files:**
- Create: `scripts/video_mse_parity_test.py`

This script runs BOTH paths independently — no shared transformer calls, no proxies. Sequential execution to avoid OOM.

- [ ] **Step 1: Write the script**

```python
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
    # Free intermediates aggressively
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
```

- [ ] **Step 2: Run the test on H200**

```bash
ssh RD-H100-2 "cd /home/ubuntu/kinetix-flow-factory && git pull origin main"
ssh RD-H100-2 "source activate DiffusionNFT && cd /home/ubuntu/kinetix-flow-factory && \
  LTX_MODEL_PATH=... LTX_UNION_LORA_PATH=... LTX_GEMMA_PATH=... \
  python scripts/video_mse_parity_test.py"
```

Expected: `MSE = 0.0` and `ZERO MSE VERIFIED`

If OOM: add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` env var.

- [ ] **Step 3: If FAIL — diagnose which component diverges**

Inspect the _report output:
- If `latents` PASS but `decoded_video` FAIL → VAE decoder non-determinism
- If `latents` FAIL → denoising divergence (noise init, stepping, or Modality construction)

- [ ] **Step 4: Fix divergence and re-run until MSE=0**

- [ ] **Step 5: Commit**

```bash
git add scripts/video_mse_parity_test.py
git commit -m "test: zero-MSE video parity test (no proxies, independent runs)"
```

---

### Task 2: Add image conditioning to Flow-Factory

The original LTX ICLoraPipeline uses `combined_image_conditionings` to encode the reference image (person appearance) as additional reference tokens. Flow-Factory's training currently skips this — only pose_depth video conditioning is used.

This means the original pipeline generates [target | image_ref | video_ref] tokens, while Flow-Factory generates [target | video_ref] tokens. Different token count → different attention → different output.

**Files:**
- Modify: `src/flow_factory/models/ltx/ltx_union.py` (encode_image, preprocess_func, inference, forward)
- Modify: `src/flow_factory/data_utils/dataset.py` (pass ref images to preprocess)

- [ ] **Step 1: Understand image conditioning in the original pipeline**

The original script does:
```python
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils import combined_image_conditionings

images = [ImageConditioningInput(path=ref_image, frame_idx=0, strength=1.0)]
image_conditionings = combined_image_conditionings(
    images=images, height=H, width=W,
    video_encoder=video_encoder, dtype=dtype, device=device,
)
# Each image_conditioning is a VideoConditionByReferenceLatent
# conditionings = image_conditionings + [video_cond]
# noise_video_state applies all conditionings, appending ref tokens
```

Read `combined_image_conditionings` source to understand:
- What shape are the image latents?
- What downscale_factor is used?
- How many tokens does each image add?

- [ ] **Step 2: Implement encode_image in LTXUnionAdapter**

Currently returns None. Update to:
```python
def encode_image(self, images, **kwargs):
    # Load each image, resize to ref resolution, encode with VAE
    # Return as reference latents with appropriate dsf
    ...
```

- [ ] **Step 3: Update preprocess_func to include image conditioning**

```python
def preprocess_func(self, prompt, videos=None, images=None, **kwargs):
    results = {}
    if prompt is not None:
        results.update(self.encode_prompt(prompt=prompt, **kwargs))
    if videos is not None:
        results.update(self.encode_video(videos=videos, **kwargs))
    if images is not None:
        img_result = self.encode_image(images=images, **kwargs)
        if img_result is not None:
            results["image_ref_latents"] = img_result["reference_latents"]
    return results
```

- [ ] **Step 4: Update inference() to include image ref tokens**

In the noise initialization, combine:
```python
# [target | image_ref | video_ref]
combined_3d = torch.cat([
    target_zeros_3d,
    image_ref_latents_3d,
    video_ref_latents_3d,
], dim=1)
```

- [ ] **Step 5: Update forward() to handle additional image ref tokens**

Build positions for three segments instead of two:
```python
positions = torch.cat([target_pos, image_ref_pos, video_ref_pos], dim=2)
per_token_ts = torch.cat([target_ts, image_ref_ts, video_ref_ts], dim=1)
```

- [ ] **Step 6: Update dataset to provide reference images**

Ensure the dataloader passes ref images from `ref/{sample_id}.png` to preprocess_func.

- [ ] **Step 7: Test with image conditioning**

Re-run the zero-MSE test with both paths using image conditioning.

- [ ] **Step 8: Commit**

```bash
git add src/flow_factory/models/ltx/ltx_union.py src/flow_factory/data_utils/dataset.py
git commit -m "feat: add image conditioning to match original ICLoraPipeline"
```

---

### Task 3: Full pipeline zero-MSE test with image conditioning

- [ ] **Step 1: Update video_mse_parity_test.py**

Add image conditioning to Path A (using combined_image_conditionings) and Path B (using the new FF encode_image).

- [ ] **Step 2: Run and verify MSE=0**

- [ ] **Step 3: Restart training with image conditioning enabled**

---

## Verification

The ONLY acceptance criterion:

```
MSE = 0.0
*** ZERO MSE VERIFIED — VIDEOS ARE PIXEL-IDENTICAL ***
```

No proxies. No "close enough." Zero.
