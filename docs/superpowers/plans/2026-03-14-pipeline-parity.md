# IC-LoRA Pipeline Parity Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Flow-Factory inference produce bitwise-identical output to the original ICLoraPipeline (Stage 1, skip_stage_2=True) with the same seed.

**Architecture:** Fix 4 differences: sigma schedule, token ordering, noise generation, and positions dtype. Then write a delta-zero comparison test that runs both pipelines with the same seed and asserts torch.equal on the final latents.

**Tech Stack:** PyTorch, ltx_core, ltx_pipelines (original), Flow-Factory

---

## Identified Differences

| # | Difference | Original | Flow-Factory (current) |
|---|-----------|----------|----------------------|
| 1 | Sigma schedule | `[1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]` (8 steps) | `LTX2Scheduler().execute(steps=50)` (50 steps, shifted+stretched) |
| 2 | Token order | `[target \| ref]` (ref appended) | `[ref \| target]` (ref prepended) |
| 3 | Noise generation | Combined 3D noise `(B, seq_target+seq_ref, 128)` with mask | Target-only 5D noise `(B, 128, F, H, W)` |
| 4 | Positions dtype | Target=bf16, ref=f32, cat→f32 | All cast to bf16 |

---

### Task 1: Fix Sigma Schedule

**Files:**
- Modify: `src/flow_factory/models/ltx/scheduler_bridge.py:44-48`

- [ ] **Step 1: Replace DISTILLED_SIGMA_VALUES with exact original values**

```python
DISTILLED_SIGMA_VALUES: List[float] = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0
]
```

- [ ] **Step 2: Verify the schedule has 8 steps (9 sigma values including terminal 0)**

---

### Task 2: Fix Token Ordering

**Files:**
- Modify: `src/flow_factory/models/ltx/ltx_union.py` — `forward()` method

Changes needed in forward():
1. `combined = torch.cat([target_latents_3d, ref_latents_3d], dim=1)` (swap order)
2. `per_token_timesteps = torch.cat([target_ts, ref_ts], dim=1)` (swap order)
3. `positions = torch.cat([target_positions, ref_positions], dim=2)` (swap order)
4. Extract velocity: `target_v_pred_3d = video_pred[:, :seq_target]` (first tokens, not last)

---

### Task 3: Fix Positions Dtype

**Files:**
- Modify: `src/flow_factory/models/ltx/ltx_union.py` — `forward()` method

Changes:
1. Target positions: `get_pixel_coords → float32 → /fps → to(bfloat16)` (matching create_initial_state)
2. Ref positions: `get_pixel_coords → to(float32) → /fps → scale by dsf` (matching VideoConditionByReferenceLatent)
3. Concatenate: `torch.cat([target_bf16, ref_f32])` → auto-upcasts to f32
4. Remove the `positions = positions.to(dtype=torch.bfloat16)` line

---

### Task 4: Fix Noise Generation

**Files:**
- Modify: `src/flow_factory/models/ltx/ltx_union.py` — `inference()` method

Instead of generating 5D target-only noise, replicate the original's combined-state noise:
1. Patchify zeros target → target_3d
2. Combine: `[target_3d, ref_3d]` (matching token order)
3. Generate `torch.randn(B, seq_total, 128, generator=gen, dtype=bf16)`
4. Apply denoise mask: `noise * mask + combined * (1 - mask)`
5. Extract target tokens, unpatchify to 5D

---

### Task 5: Write Comparison Test

**Files:**
- Create: `scripts/pipeline_parity_test.py`

Script that runs on H200:
1. Loads model weights (same as delta_zero_compare.py)
2. Runs original ICLoraPipeline(skip_stage_2=True) to get Stage 1 latents
3. Runs our LTXUnionAdapter.inference() with same seed, prompt, reference
4. Asserts `torch.equal(original_latents, ff_latents)`
5. If not equal, reports per-step divergence

---

### Task 6: Run Test on H200

- [ ] Push fixes to remote
- [ ] Run test
- [ ] Fix any remaining issues until 0 MSE verified
