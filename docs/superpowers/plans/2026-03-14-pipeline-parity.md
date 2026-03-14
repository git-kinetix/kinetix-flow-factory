# IC-LoRA Pipeline Parity Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Flow-Factory inference produce bitwise-identical output to the original ICLoraPipeline (Stage 1, skip_stage_2=True) with the same seed.

**Architecture:** Fix 5 differences: sigma schedule, token ordering, noise generation, positions dtype, and ODE velocity division. Then write a delta-zero comparison test that runs both pipelines with the same seed and asserts torch.equal on the final latents.

**Tech Stack:** PyTorch, ltx_core, ltx_pipelines (original), Flow-Factory

**Status: COMPLETE** — All 5 differences fixed, parity verified bitwise on H200 (2026-03-14).

---

## Identified Differences

| # | Difference | Original | Flow-Factory (fixed) |
|---|-----------|----------|----------------------|
| 1 | Sigma schedule | `[1.0, 0.99375, ..., 0.0]` (8 steps) | Fixed: exact distilled values |
| 2 | Token order | `[target \| ref]` (ref appended) | Fixed: `[target \| ref]` |
| 3 | Noise generation | Combined 3D noise with mask | Fixed: combined 3D noise with mask |
| 4 | Positions dtype | Target=bf16, ref=f32, cat→f32 | Fixed: matching dtypes |
| 5 | ODE velocity division | `to_velocity` uses `sigma.item()` (Python float) | Fixed: uses `.item()` to match CUDA kernel |

---

### Task 1: Fix Sigma Schedule — DONE

- [x] Replace DISTILLED_SIGMA_VALUES with exact original values
- [x] Fix sub-sampling check (`num_inference_steps >= num_distilled_steps`)

### Task 2: Fix Token Ordering — DONE

- [x] Swap to `[target | ref]` order in forward()
- [x] Extract velocity from first tokens (`[:, :seq_target]`)

### Task 3: Fix Positions Dtype — DONE

- [x] Target positions: bf16 roundtrip
- [x] Ref positions: stay f32
- [x] Cat auto-upcasts to f32

### Task 4: Fix Noise Generation — DONE

- [x] Combined 3D noise matching GaussianNoiser pattern

### Task 5: Fix ODE Velocity Division — DONE

- [x] Use `sigma.item()` in scheduler ODE path to match `to_velocity()`'s Python float division
- [x] Root cause: CUDA kernel for `f32/f64` differs from `f32/f32`, producing ~6e-8 diffs that round to different bf16 values

### Task 6: Write & Run Comparison Test — DONE

- [x] `scripts/pipeline_parity_test.py` — single-transformer-call approach
- [x] All 8 steps bitwise identical
- [x] Final latents bitwise identical
